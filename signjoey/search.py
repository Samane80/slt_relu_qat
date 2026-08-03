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
) -> (np.array, np.array):
    batch_size = src_mask.size(0)
    prev_y = src_mask.new_full(
        size=[batch_size, 1], fill_value=bos_index, dtype=torch.long
    )
    output = []
    attention_scores = []
    hidden = None
    prev_att_vector = None
    finished = src_mask.new_zeros((batch_size, 1)).byte()

    for t in range(max_output_length):
        trg_embed, trg_sf = _unpack_embed(embed(prev_y))
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
        output.append(next_word.squeeze(1).detach().cpu().numpy())
        prev_y = next_word
        attention_scores.append(att_probs.squeeze(1).detach().cpu().numpy())
        is_eos = torch.eq(next_word, eos_index)
        finished += is_eos
        if (finished >= 1).sum() == batch_size:
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
) -> (np.array, None):
    batch_size = src_mask.size(0)
    ys = encoder_output.new_full([batch_size, 1], bos_index, dtype=torch.long)
    trg_mask = src_mask.new_ones([1, 1, 1])
    finished = src_mask.new_zeros((batch_size)).byte()

    for _ in range(max_output_length):
        trg_embed, trg_sf = _unpack_embed(embed(ys))
        with torch.no_grad():
            # logits, out, _, _ = decoder(
            #     trg_embed=trg_embed,
            #     encoder_output=encoder_output,
            #     encoder_hidden=None,
            #     src_mask=src_mask,
            #     unroll_steps=None,
            #     hidden=None,
            #     trg_mask=trg_mask,
            #     act_scaling_factor=trg_sf,
            #     encoder_output_sf=encoder_output_sf,
            # )
            # logits = logits[:, -1]
            (logits, _), out, _, _ = decoder(
                trg_embed=trg_embed,
                encoder_output=encoder_output,
                encoder_hidden=None,
                src_mask=src_mask,
                unroll_steps=None,
                hidden=None,
                trg_mask=trg_mask,
                act_scaling_factor=trg_sf,
                encoder_output_sf=encoder_output_sf,  
            )
            logits = logits[:, -1]
            _, next_word = torch.max(logits, dim=1)
            next_word = next_word.data
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)

        is_eos = torch.eq(next_word, eos_index)
        finished += is_eos
        if (finished >= 1).sum() == batch_size:
            break

    ys = ys[:, 1:]
    return ys.detach().cpu().numpy(), None


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