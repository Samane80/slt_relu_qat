# coding: utf-8
"""
Decoding helpers — QAT-compatible (embed returns (x, sf)).
"""
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from signjoey.decoders import Decoder, TransformerDecoder
from signjoey.embeddings import Embeddings
from signjoey.helpers import tile

__all__ = ["greedy", "transformer_greedy", "beam_search"]


def _unpack_embed(embed_out):
    """Embeddings may return Tensor or (Tensor, sf)."""
    if isinstance(embed_out, tuple):
        return embed_out[0], embed_out[1]
    return embed_out, None


def greedy(
    src_mask: Tensor,
    embed: Embeddings,
    bos_index: int,
    eos_index: int,
    max_output_length: int,
    decoder: Decoder,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
    encoder_output_sf: Tensor = None,
    pad_index: int = None,
) -> (np.array, np.array):
    if isinstance(decoder, TransformerDecoder):
        greedy_fun = transformer_greedy
    else:
        greedy_fun = recurrent_greedy

    return greedy_fun(
        src_mask=src_mask,
        embed=embed,
        bos_index=bos_index,
        eos_index=eos_index,
        max_output_length=max_output_length,
        decoder=decoder,
        encoder_output=encoder_output,
        encoder_hidden=encoder_hidden,
        encoder_output_sf=encoder_output_sf,
        pad_index=pad_index,
    )


def recurrent_greedy(
    src_mask: Tensor,
    embed: Embeddings,
    bos_index: int,
    eos_index: int,
    max_output_length: int,
    decoder: Decoder,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
    encoder_output_sf: Tensor = None,
    pad_index: int = None,
) -> (np.array, np.array):
    batch_size = src_mask.size(0)
    if pad_index is None:
        # Backwards-compatible fallback for callers that do not provide a
        # target PAD id.  The model path passes it explicitly.
        pad_index = eos_index
    prev_y = src_mask.new_full(
        size=[batch_size, 1], fill_value=bos_index, dtype=torch.long
    )
    output = []
    attention_scores = []
    hidden = None
    prev_att_vector = None
    finished = src_mask.new_zeros((batch_size, 1), dtype=torch.bool)

    for t in range(max_output_length):
        trg_embed, trg_sf = _unpack_embed(
            embed(prev_y, mask=src_mask.new_ones((batch_size, 1, 1)))
        )
        logits, hidden, att_probs, prev_att_vector = decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            trg_embed=trg_embed,
            hidden=hidden,
            prev_att_vector=prev_att_vector,
            unroll_steps=1,
            act_scaling_factor=trg_sf,
            encoder_output_sf=encoder_output_sf,
        )
        next_word = torch.argmax(logits, dim=-1)
        was_finished = finished.bool()
        next_word = torch.where(
            was_finished,
            next_word.new_full(next_word.shape, pad_index),
            next_word,
        )
        output.append(next_word.squeeze(1).detach().cpu().numpy())
        prev_y = next_word
        attention_scores.append(att_probs.squeeze(1).detach().cpu().numpy())
        is_eos = torch.eq(next_word, eos_index) & ~was_finished
        finished |= is_eos
        if bool(finished.all()):
            break

    stacked_output = np.stack(output, axis=1)
    stacked_attention_scores = np.stack(attention_scores, axis=1)
    return stacked_output, stacked_attention_scores


def transformer_greedy(
    src_mask: Tensor,
    embed: Embeddings,
    bos_index: int,
    eos_index: int,
    max_output_length: int,
    decoder: Decoder,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
    encoder_output_sf: Tensor = None,
    pad_index: int = None,
) -> (np.array, None):
    """Greedy decoding for a Transformer decoder.

    Every sequence in a batch is unrolled to the same tensor length, but a
    sequence that has emitted EOS is frozen and receives PAD thereafter.
    The old implementation kept feeding model predictions after EOS, which
    produced strings such as ``... </s> . </s>`` and made raw hypotheses
    depend on the other examples in the batch.
    """
    batch_size = src_mask.size(0)
    if pad_index is None:
        pad_index = eos_index

    ys = encoder_output.new_full([batch_size, 1], bos_index, dtype=torch.long)
    # Valid-prefix mask: BOS is valid for every sequence.  It is passed to
    # both the embedding normalizer and the decoder, in addition to the
    # causal mask constructed inside TransformerDecoder.
    prefix_mask = src_mask.new_ones([batch_size, 1, 1], dtype=torch.bool)
    finished = src_mask.new_zeros((batch_size), dtype=torch.bool)

    for _ in range(max_output_length):
        trg_embed, trg_sf = _unpack_embed(embed(ys, mask=prefix_mask))
        with torch.no_grad():
            (logits, _), _, _, _ = decoder(
                trg_embed=trg_embed,
                encoder_output=encoder_output,
                encoder_hidden=None,
                src_mask=src_mask,
                unroll_steps=None,
                hidden=None,
                trg_mask=prefix_mask,
                act_scaling_factor=trg_sf,
                encoder_output_sf=encoder_output_sf,
            )
            next_word = torch.argmax(logits[:, -1], dim=1)
            was_finished = finished
            next_word = torch.where(
                was_finished,
                next_word.new_full(next_word.shape, pad_index),
                next_word,
            )
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)

        # EOS itself remains valid and is kept in the returned hypothesis;
        # only tokens appended after it are PAD.
        is_eos = torch.eq(next_word, eos_index) & ~was_finished
        finished |= is_eos
        prefix_mask = torch.cat(
            [prefix_mask, (~was_finished).view(batch_size, 1, 1)], dim=-1
        )
        if bool(finished.all()):
            break

    return ys[:, 1:].detach().cpu().numpy(), None


def _transformer_beam_search(
    decoder: TransformerDecoder,
    size: int,
    bos_index: int,
    eos_index: int,
    pad_index: int,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
    src_mask: Tensor,
    max_output_length: int,
    alpha: float,
    embed: Embeddings,
    encoder_output_sf: Tensor = None,
) -> (np.array, None):
    """Correctness-first beam search for the full-prefix Transformer API.

    The previous implementation was adapted from the recurrent JoeyNMT
    search code.  It used ``decoder.output_size`` (unset by
    TransformerDecoder), passed the wrong beam offsets after removing a
    finished sentence, and kept expanding hypotheses after EOS.  This
    implementation keeps each example independent, which is slightly less
    parallel but makes the state transitions and EOS semantics explicit.
    """
    batch_size = src_mask.size(0)
    vocab_size = decoder.vocab_size
    if size <= 0:
        raise ValueError("Beam size must be > 0")
    if max_output_length < 0:
        raise ValueError("max_output_length must be non-negative")

    def length_penalized_score(raw_score, length):
        if alpha is None or alpha < 0:
            return raw_score
        return raw_score / (((5.0 + length) / 6.0) ** alpha)

    def select_scale(scale, batch_index):
        if scale is None or scale.numel() == 1:
            return scale
        if scale.dim() > 0 and scale.size(0) == batch_size:
            return scale[batch_index : batch_index + 1]
        return scale

    outputs = []
    with torch.no_grad():
        for batch_index in range(batch_size):
            memory = encoder_output[batch_index : batch_index + 1]
            memory_hidden = (
                encoder_hidden[batch_index : batch_index + 1]
                if encoder_hidden is not None
                and encoder_hidden.dim() > 0
                and encoder_hidden.size(0) == batch_size
                else encoder_hidden
            )
            memory_sf = select_scale(encoder_output_sf, batch_index)
            source_mask = src_mask[batch_index : batch_index + 1]

            # Each item is (token_ids including BOS, raw log probability,
            # finished).  Finished hypotheses are carried forward unchanged.
            beams = [(
                torch.tensor([bos_index], dtype=torch.long, device=memory.device),
                0.0,
                False,
            )]

            for _ in range(max_output_length):
                # Keep finished hypotheses, but evaluate all active beams in
                # one decoder call for this example.  This preserves the
                # explicit EOS handling without turning beam search into
                # ``batch * beam * length`` separate Transformer forwards.
                candidates = [item for item in beams if item[2]]
                active = [item for item in beams if not item[2]]
                if active:
                    decoder_input = torch.stack([item[0] for item in active])
                    active_size = decoder_input.size(0)
                    prefix_mask = torch.ones(
                        active_size, 1, decoder_input.size(1),
                        dtype=torch.bool, device=memory.device
                    )
                    trg_embed, trg_sf = _unpack_embed(
                        embed(decoder_input, mask=prefix_mask)
                    )
                    active_memory = memory.expand(
                        active_size, -1, -1
                    )
                    active_source_mask = source_mask.expand(
                        active_size, -1, -1
                    )
                    active_memory_sf = memory_sf
                    if (
                        active_memory_sf is not None
                        and active_memory_sf.numel() > 1
                        and active_memory_sf.size(0) == 1
                    ):
                        active_memory_sf = active_memory_sf.expand(
                            active_size, *active_memory_sf.shape[1:]
                        )
                    (logits, _), _, _, _ = decoder(
                        trg_embed=trg_embed,
                        encoder_output=active_memory,
                        encoder_hidden=memory_hidden,
                        src_mask=active_source_mask,
                        trg_mask=prefix_mask,
                        hidden=None,
                        unroll_steps=None,
                        act_scaling_factor=trg_sf,
                        encoder_output_sf=active_memory_sf,
                    )
                    next_log_probs = F.log_softmax(
                        logits[:, -1], dim=-1
                    )
                    top_k = min(size, vocab_size)
                    values, indices = torch.topk(next_log_probs, top_k, dim=-1)
                    for row, (token_ids, raw_score, _) in enumerate(active):
                        for value, index in zip(values[row], indices[row]):
                            next_token = int(index.item())
                            next_ids = torch.cat(
                                [token_ids, index.view(1)], dim=0
                            )
                            candidates.append((
                                next_ids,
                                raw_score + float(value.item()),
                                next_token == eos_index,
                            ))

                candidates.sort(
                    key=lambda item: length_penalized_score(
                        item[1], item[0].numel() - 1
                    ),
                    reverse=True,
                )
                beams = candidates[:size]
                if beams and all(item[2] for item in beams):
                    break

            finished_beams = [item for item in beams if item[2]]
            ranked = finished_beams if finished_beams else beams
            ranked.sort(
                key=lambda item: length_penalized_score(
                    item[1], item[0].numel() - 1
                ),
                reverse=True,
            )
            outputs.append(ranked[0][0][1:].detach().cpu().numpy())

    max_len = max((output.shape[0] for output in outputs), default=0)
    stacked = np.full((batch_size, max_len), pad_index, dtype=np.int64)
    for batch_index, output in enumerate(outputs):
        stacked[batch_index, : output.shape[0]] = output
    return stacked, None


def beam_search(
    decoder: Decoder,
    size: int,
    bos_index: int,
    eos_index: int,
    pad_index: int,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
    src_mask: Tensor,
    max_output_length: int,
    alpha: float,
    embed: Embeddings,
    n_best: int = 1,
    encoder_output_sf: Tensor = None,
) -> (np.array, np.array):
    if isinstance(decoder, TransformerDecoder):
        if n_best != 1:
            raise ValueError("Transformer beam_search currently supports n_best=1")
        return _transformer_beam_search(
            decoder=decoder,
            size=size,
            bos_index=bos_index,
            eos_index=eos_index,
            pad_index=pad_index,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            max_output_length=max_output_length,
            alpha=alpha,
            embed=embed,
            encoder_output_sf=encoder_output_sf,
        )
    return _legacy_beam_search(
        decoder=decoder,
        size=size,
        bos_index=bos_index,
        eos_index=eos_index,
        pad_index=pad_index,
        encoder_output=encoder_output,
        encoder_hidden=encoder_hidden,
        src_mask=src_mask,
        max_output_length=max_output_length,
        alpha=alpha,
        embed=embed,
        n_best=n_best,
        encoder_output_sf=encoder_output_sf,
    )


def _legacy_beam_search(
    decoder: Decoder,
    size: int,
    bos_index: int,
    eos_index: int,
    pad_index: int,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
    src_mask: Tensor,
    max_output_length: int,
    alpha: float,
    embed: Embeddings,
    n_best: int = 1,
    encoder_output_sf: Tensor = None,
) -> (np.array, np.array):
    assert size > 0, "Beam size must be >0."
    assert n_best <= size, "Can only return {} best hypotheses.".format(size)

    transformer = isinstance(decoder, TransformerDecoder)
    batch_size = src_mask.size(0)
    att_vectors = None

    if not transformer:
        hidden = decoder._init_hidden(encoder_hidden)
    else:
        hidden = None

    if hidden is not None:
        hidden = tile(hidden, size, dim=1)

    encoder_output = tile(encoder_output.contiguous(), size, dim=0)
    src_mask = tile(src_mask, size, dim=0)

    # tile encoder sf if present (scalar or [1] → broadcast ok; if batched, tile)
    if encoder_output_sf is not None and encoder_output_sf.dim() > 0:
        if encoder_output_sf.numel() > 1:
            encoder_output_sf = tile(encoder_output_sf, size, dim=0)

    if transformer:
        trg_mask = src_mask.new_ones([1, 1, 1])
    else:
        trg_mask = None

    batch_offset = torch.arange(
        batch_size, dtype=torch.long, device=encoder_output.device
    )
    beam_offset = torch.arange(
        0, batch_size * size, step=size, dtype=torch.long, device=encoder_output.device
    )
    alive_seq = torch.full(
        [batch_size * size, 1],
        bos_index,
        dtype=torch.long,
        device=encoder_output.device,
    )

    topk_log_probs = torch.zeros(batch_size, size, device=encoder_output.device)
    topk_log_probs[:, 1:] = float("-inf")

    hypotheses = [[] for _ in range(batch_size)]
    results = {
        "predictions": [[] for _ in range(batch_size)],
        "scores": [[] for _ in range(batch_size)],
        "gold_score": [0] * batch_size,
    }

    for step in range(max_output_length):
        if transformer:
            decoder_input = alive_seq
        else:
            decoder_input = alive_seq[:, -1].view(-1, 1)

        trg_embed, trg_sf = _unpack_embed(embed(decoder_input))

        # logits, hidden, att_scores, att_vectors = decoder(
        #     encoder_output=encoder_output,
        #     encoder_hidden=encoder_hidden,
        #     src_mask=src_mask,
        #     trg_embed=trg_embed,
        #     hidden=hidden,
        #     prev_att_vector=att_vectors,
        #     unroll_steps=1,
        #     trg_mask=trg_mask,
        #     act_scaling_factor=trg_sf,
        #     encoder_output_sf=encoder_output_sf,
        # )
        (logits, _), hidden, att_scores, att_vectors = decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            trg_embed=trg_embed,
            hidden=hidden,
            prev_att_vector=att_vectors,
            unroll_steps=1,
            trg_mask=trg_mask,
            act_scaling_factor=trg_sf,
            encoder_output_sf=encoder_output_sf,  
        )

        if transformer:
            logits = logits[:, -1]
            hidden = None

        log_probs = F.log_softmax(logits, dim=-1).squeeze(1)
        log_probs += topk_log_probs.view(-1).unsqueeze(1)
        curr_scores = log_probs.clone()

        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty

        curr_scores = curr_scores.reshape(-1, size * decoder.output_size)
        topk_scores, topk_ids = curr_scores.topk(size, dim=-1)

        if alpha > -1:
            topk_log_probs = topk_scores * length_penalty
        else:
            topk_log_probs = topk_scores.clone()

        topk_beam_index = topk_ids.div(decoder.output_size)
        topk_ids = topk_ids.fmod(decoder.output_size)

        batch_index = topk_beam_index + beam_offset[: topk_beam_index.size(0)].unsqueeze(
            1
        )
        select_indices = batch_index.view(-1)

        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1
        )

        is_finished = topk_ids.eq(eos_index)
        if step + 1 == max_output_length:
            is_finished.fill_(True)

        end_condition = is_finished[:, 0].eq(True)

        if is_finished.any():
            predictions = alive_seq.view(-1, size, alive_seq.size(-1))
            for i in range(is_finished.size(0)):
                b = batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill_(True)
                finished_hyp = is_finished[i].nonzero().view(-1)
                for j in finished_hyp:
                    if (predictions[i, j, 1:] == eos_index).nonzero().numel() < 2:
                        hypotheses[b].append(
                            (topk_scores[i, j], predictions[i, j, 1:])
                        )
                if end_condition[i]:
                    best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)

            non_finished = end_condition.eq(False).nonzero().view(-1)
            if len(non_finished) == 0:
                break

            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished).view(
                -1, alive_seq.size(-1)
            )

        select_indices = batch_index.view(-1)
        encoder_output = encoder_output.index_select(0, select_indices)
        src_mask = src_mask.index_select(0, select_indices)

        if hidden is not None and not transformer:
            if isinstance(hidden, tuple):
                h, c = hidden
                h = h.index_select(1, select_indices)
                c = c.index_select(1, select_indices)
                hidden = (h, c)
            else:
                hidden = hidden.index_select(1, select_indices)

        if att_vectors is not None:
            att_vectors = att_vectors.index_select(0, select_indices)

    def pad_and_stack_hyps(hyps, pad_value):
        filled = (
            np.ones((len(hyps), max([h.shape[0] for h in hyps])), dtype=int) * pad_value
        )
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    assert n_best == 1
    final_outputs = pad_and_stack_hyps(
        [r[0].cpu().numpy() for r in results["predictions"]], pad_value=pad_index
    )
    return final_outputs, None
