# coding: utf-8
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import groupby
from signjoey.initialization import initialize_model
from signjoey.embeddings import Embeddings, SpatialEmbeddings
from signjoey.encoders import Encoder, TransformerEncoder
from signjoey.decoders import Decoder, TransformerDecoder
from signjoey.search import beam_search, greedy
from signjoey.vocabulary import (
    TextVocabulary,
    GlossVocabulary,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
)
from signjoey.batch import Batch
from signjoey.helpers import freeze_params
from quantization_utils.quant_modules import QuantLinear
from quantization_utils.model_utils import set_quantize_mode
from torch import Tensor
from typing import Union


class SignModel(nn.Module):
    """
    Base Model class
    """

    def __init__(
        self,
        encoder: Encoder,
        gloss_output_layer: nn.Module,
        decoder: Decoder,
        sgn_embed: SpatialEmbeddings,
        txt_embed: Embeddings,
        gls_vocab: GlossVocabulary,
        txt_vocab: TextVocabulary,
        do_recognition: bool = True,
        do_translation: bool = True,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.sgn_embed = sgn_embed
        self.txt_embed = txt_embed

        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab

        self.txt_bos_index = self.txt_vocab.stoi[BOS_TOKEN]
        self.txt_pad_index = self.txt_vocab.stoi[PAD_TOKEN]
        self.txt_eos_index = self.txt_vocab.stoi[EOS_TOKEN]

        self.gloss_output_layer = gloss_output_layer
        self.do_recognition = do_recognition
        self.do_translation = do_translation

    # pylint: disable=arguments-differ
    def forward(
        self,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        txt_input: Tensor,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.
        """
        encoder_output, encoder_hidden = self.encode(
            sgn=sgn, sgn_mask=sgn_mask, sgn_length=sgn_lengths
        )

        if self.do_recognition:
            gloss_scores = self.gloss_output_layer(encoder_output)
            gloss_probabilities = gloss_scores.log_softmax(2)
            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
        else:
            gloss_probabilities = None

        if self.do_translation:
            unroll_steps = txt_input.size(1)
            decoder_outputs = self.decode(
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                sgn_mask=sgn_mask,
                txt_input=txt_input,
                unroll_steps=unroll_steps,
                txt_mask=txt_mask,
            )
        else:
            decoder_outputs = None

        return decoder_outputs, gloss_probabilities

    def encode(
        self, sgn: Tensor, sgn_mask: Tensor, sgn_length: Tensor
    ) -> (Tensor, Tensor):
        """
        Encodes the source sentence.
        ✅ تغییر: ساخت query_mask و handle خروجی tuple از sgn_embed
        """
        # ✅ سازگاری با QAT (sgn_embed دو خروجی برمی‌گرداند)
        sgn_embed_result = self.sgn_embed(x=sgn, mask=sgn_mask)
        if isinstance(sgn_embed_result, tuple):
            sgn_embed, sgn_embed_sf = sgn_embed_result
        else:
            sgn_embed = sgn_embed_result
            sgn_embed_sf = None

        # ✅ ساخت query_mask برای encoder
        encoder_query_mask = sgn_mask.squeeze(1) if sgn_mask is not None else None

        return self.encoder(
            embed_src=sgn_embed,
            src_length=sgn_length,
            mask=sgn_mask,
            act_scaling_factor=sgn_embed_sf,
            query_mask=encoder_query_mask,
        )

    def decode(
        self,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        sgn_mask: Tensor,
        txt_input: Tensor,
        unroll_steps: int,
        decoder_hidden: Tensor = None,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.
        ✅ تغییر: ساخت query_mask، handle tuple از txt_embed، پاس دادن sf
        """
        # ✅ سازگاری با QAT (txt_embed دو خروجی برمی‌گرداند)
        txt_embed_result = self.txt_embed(x=txt_input, mask=txt_mask)
        if isinstance(txt_embed_result, tuple):
            txt_embed, txt_embed_sf = txt_embed_result
        else:
            txt_embed = txt_embed_result
            txt_embed_sf = None

        # ✅ ساخت query_mask برای decoder
        decoder_query_mask = txt_mask.squeeze(1) if txt_mask is not None else None

        return self.decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=sgn_mask,
            trg_embed=txt_embed,
            trg_mask=txt_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
            act_scaling_factor=txt_embed_sf,
            encoder_output_sf=encoder_hidden,  # ✅ در QAT: encoder_hidden = sf
            query_mask=decoder_query_mask,
        )

    def get_attention_reg_loss(self) -> Tensor:
        """
        Sum of the ReLUFormer attention-regularization losses.
        """
        reg_losses = []
        encoder_reg_loss = getattr(self.encoder, "reg_loss", None)
        if encoder_reg_loss is not None:
            reg_losses.append(encoder_reg_loss)
        if self.decoder is not None:
            decoder_reg_loss = getattr(self.decoder, "reg_loss", None)
            if decoder_reg_loss is not None:
                reg_losses.append(decoder_reg_loss)
        if reg_losses:
            return torch.stack(reg_losses).sum()
        return None

    def get_loss_for_batch(
        self,
        batch: Batch,
        recognition_loss_function: nn.Module,
        translation_loss_function: nn.Module,
        recognition_loss_weight: float,
        translation_loss_weight: float,
        attn_reg_loss_weight: float = 0.0,
    ) -> (Tensor, Tensor):
        """
        Compute non-normalized loss and number of tokens for a batch
        """
        decoder_outputs, gloss_probabilities = self.forward(
            sgn=batch.sgn,
            sgn_mask=batch.sgn_mask,
            sgn_lengths=batch.sgn_lengths,
            txt_input=batch.txt_input,
            txt_mask=batch.txt_mask,
        )

        if self.do_recognition:
            assert gloss_probabilities is not None
            recognition_loss = (
                recognition_loss_function(
                    gloss_probabilities,
                    batch.gls,
                    batch.sgn_lengths.long(),
                    batch.gls_lengths.long(),
                )
                * recognition_loss_weight
            )
        else:
            recognition_loss = None

        if self.do_translation:
            assert decoder_outputs is not None
            # ✅ تغییر: سازگاری با QAT (خروجی decoder tuple است)
            first_output, _, _, _ = decoder_outputs
            if isinstance(first_output, tuple):
                word_outputs = first_output[0]  # ✅ فقط tensor
            else:
                word_outputs = first_output

            txt_log_probs = F.log_softmax(word_outputs, dim=-1)
            translation_loss = (
                translation_loss_function(txt_log_probs, batch.txt)
                * translation_loss_weight
            )
        else:
            translation_loss = None

        if attn_reg_loss_weight and attn_reg_loss_weight > 0.0:
            attn_reg_loss = self.get_attention_reg_loss()
            if attn_reg_loss is not None:
                if self.do_translation:
                    translation_loss = translation_loss + attn_reg_loss_weight * attn_reg_loss
                elif self.do_recognition:
                    recognition_loss = recognition_loss + attn_reg_loss_weight * attn_reg_loss

        return recognition_loss, translation_loss

    def run_batch(
        self,
        batch: Batch,
        recognition_beam_size: int = 1,
        translation_beam_size: int = 1,
        translation_beam_alpha: float = -1,
        translation_max_output_length: int = 100,
    ) -> (np.array, np.array, np.array):
        """
        Get outputs and attentions scores for a given batch
        """
        encoder_output, encoder_hidden = self.encode(
            sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths
        )

        if self.do_recognition:
            gloss_scores = self.gloss_output_layer(encoder_output)
            gloss_probabilities = gloss_scores.log_softmax(2)
            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
            gloss_probabilities = gloss_probabilities.cpu().detach().numpy()
            tf_gloss_probabilities = np.concatenate(
                (gloss_probabilities[:, :, 1:], gloss_probabilities[:, :, 0, None]),
                axis=-1,
            )

            assert recognition_beam_size > 0
            ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
                inputs=tf_gloss_probabilities,
                sequence_length=batch.sgn_lengths.cpu().detach().numpy(),
                beam_width=recognition_beam_size,
                top_paths=1,
            )
            ctc_decode = ctc_decode[0]
            tmp_gloss_sequences = [[] for i in range(gloss_scores.shape[0])]
            for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
                tmp_gloss_sequences[dense_idx[0]].append(
                    ctc_decode.values[value_idx].numpy() + 1
                )
            decoded_gloss_sequences = []
            for seq_idx in range(0, len(tmp_gloss_sequences)):
                decoded_gloss_sequences.append(
                    [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
                )
        else:
            decoded_gloss_sequences = None

        if self.do_translation:
            if translation_beam_size < 2:
                stacked_txt_output, stacked_attention_scores = greedy(
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    encoder_output_sf=encoder_hidden,
                    src_mask=batch.sgn_mask,
                    embed=self.txt_embed,
                    bos_index=self.txt_bos_index,
                    eos_index=self.txt_eos_index,
                    decoder=self.decoder,
                    max_output_length=translation_max_output_length,
                )
            else:
                stacked_txt_output, stacked_attention_scores = beam_search(
                    size=translation_beam_size,
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    encoder_output_sf=encoder_hidden,
                    src_mask=batch.sgn_mask,
                    embed=self.txt_embed,
                    max_output_length=translation_max_output_length,
                    alpha=translation_beam_alpha,
                    eos_index=self.txt_eos_index,
                    pad_index=self.txt_pad_index,
                    bos_index=self.txt_bos_index,
                    decoder=self.decoder,
                )
        else:
            stacked_txt_output = stacked_attention_scores = None

        return decoded_gloss_sequences, stacked_txt_output, stacked_attention_scores

    def __repr__(self) -> str:
        return (
            "%s(\n"
            "\tencoder=%s,\n"
            "\tdecoder=%s,\n"
            "\tsgn_embed=%s,\n"
            "\ttxt_embed=%s)"
            % (
                self.__class__.__name__,
                self.encoder,
                self.decoder,
                self.sgn_embed,
                self.txt_embed,
            )
        )


def build_model(
    cfg: dict,
    sgn_dim: int,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    do_recognition: bool = True,
    do_translation: bool = True,
) -> SignModel:
    """
    Build and initialize the model according to the configuration.
    """
    txt_padding_idx = txt_vocab.stoi[PAD_TOKEN]

    sgn_embed: SpatialEmbeddings = SpatialEmbeddings(
        **cfg["encoder"]["embeddings"],
        num_heads=cfg["encoder"]["num_heads"],
        input_size=sgn_dim,
    )

    enc_dropout = cfg["encoder"].get("dropout", 0.0)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert (
            cfg["encoder"]["embeddings"]["embedding_dim"]
            == cfg["encoder"]["hidden_size"]
        ), "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )
    else:
        raise NotImplementedError(
            "encoder.type='{}' is not supported. Only 'transformer' is "
            "implemented in this codebase (no RecurrentEncoder is defined "
            "in signjoey.encoders). Set encoder.type: transformer in your "
            "config.".format(cfg["encoder"].get("type", "recurrent"))
        )

    if do_recognition:
        # gloss_output_layer = nn.Linear(encoder.output_size, len(gls_vocab))
        gloss_output_layer = QuantLinear(encoder.output_size, len(gls_vocab))
        if cfg["encoder"].get("freeze", False):
            freeze_params(gloss_output_layer)
    else:
        gloss_output_layer = None

    if do_translation:
        txt_embed: Union[Embeddings, None] = Embeddings(
            **cfg["decoder"]["embeddings"],
            num_heads=cfg["decoder"]["num_heads"],
            vocab_size=len(txt_vocab),
            padding_idx=txt_padding_idx,
        )
        dec_dropout = cfg["decoder"].get("dropout", 0.0)
        dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
        if cfg["decoder"].get("type", "recurrent") == "transformer":
            decoder = TransformerDecoder(
                **cfg["decoder"],
                encoder=encoder,
                vocab_size=len(txt_vocab),
                emb_size=txt_embed.embedding_dim,
                emb_dropout=dec_emb_dropout,
            )
        else:
            raise NotImplementedError(
                "decoder.type='{}' is not supported. Only 'transformer' is "
                "implemented in this codebase (no RecurrentDecoder is defined "
                "in signjoey.decoders). Set decoder.type: transformer in your "
                "config.".format(cfg["decoder"].get("type", "recurrent"))
            )
    else:
        txt_embed = None
        decoder = None

    model: SignModel = SignModel(
        encoder=encoder,
        gloss_output_layer=gloss_output_layer,
        decoder=decoder,
        sgn_embed=sgn_embed,
        txt_embed=txt_embed,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        do_recognition=do_recognition,
        do_translation=do_translation,
    )

    if do_translation:
        if cfg.get("tied_softmax", False):
            if txt_embed.lut.weight.shape == model.decoder.output_layer.weight.shape:
                model.decoder.output_layer.weight = txt_embed.lut.weight
            else:
                raise ValueError(
                    "For tied_softmax, the decoder embedding_dim and decoder "
                    "hidden_size must be the same."
                    "The decoder must be a Transformer."
                )

    initialize_model(model, cfg, txt_padding_idx)

    # Single switch for the whole model: cfg["quantize"]: false -> plain
    # FP32 forward pass (nn.Linear/nn.LayerNorm/nn.GELU/nn.Softmax math,
    # computed on these SAME parameters); true (default) -> QAT fake-quant.
    # Toggle again later, mid-training, with:
    #   from signjoey.model_utils import set_quantize_mode
    #   set_quantize_mode(model, True)   # e.g. when switching from
    #                                     # pretraining to QAT fine-tuning
    set_quantize_mode(model, cfg.get("quantize", True))

    return model