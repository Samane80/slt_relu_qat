# coding: utf-8
"""
Quantization-aware Transformer Decoder
"""
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from signjoey.helpers import freeze_params, subsequent_mask
from signjoey.transformer_layers import PositionalEncoding, TransformerDecoderLayer
from quantization_utils.quant_modules import QuantLinear, QuantAct, IntLayerNorm


class Decoder(nn.Module):
    """Base decoder class"""

    @property
    def output_size(self):
        return self._output_size


class TransformerDecoder(Decoder):
    """Transformer Decoder."""

    def __init__(
        self,
        num_layers: int = 6,
        num_heads: int = 8,
        hidden_size: int = 512,
        ff_size: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        vocab_size: int = 1,
        attention_type: str = "softmax",
        **kwargs,
    ):
        super().__init__()

        self.pe = PositionalEncoding(size=hidden_size)

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    attention_type=attention_type,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = IntLayerNorm(hidden_size, eps=1e-6)
        self.qact_after_dropout = QuantAct(16)
        self.qact_after_ln = QuantAct()
        self.dropout = nn.Dropout(emb_dropout)
        self.vocab_size = vocab_size

        self.output_layer = QuantLinear(hidden_size, vocab_size)
        self.qact_output = QuantAct(16)

    @property
    def reg_loss(self):
        losses = [
            l for l in (layer.reg_loss for layer in self.layers) if l is not None
        ]
        if losses:
            return torch.stack(losses).sum()
        return None

    def forward(
        self,
        trg_embed: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor,
        trg_mask: Tensor,
        act_scaling_factor: Tensor = None,
        encoder_output_sf: Tensor = None,
        encoder_hidden: Tensor = None,
        hidden: Tensor = None,           
        unroll_steps: int = None,     
        query_mask: Tensor = None,     
        **kwargs,
    ):
        """Target forward pass."""

        if encoder_output_sf is None:
            enc_min = encoder_output.detach().min()
            enc_max = encoder_output.detach().max()
            encoder_output_sf = (enc_max - enc_min).clamp(min=1e-8) / 127.0

        x = trg_embed
        x_sf = act_scaling_factor
        if trg_mask is not None:
            seq_len = trg_embed.size(1)
            causal = subsequent_mask(seq_len).to(trg_mask.device)
            trg_mask = trg_mask.unsqueeze(2) & causal.unsqueeze(0)

        x, x_sf = self.pe(x, x_sf)
        x = self.dropout(x)
        
        x, x_sf = self.qact_after_dropout(x, x_sf)

        for layer in self.layers:
            x, x_sf = layer(
                x,
                x_sf,
                memory=encoder_output,
                memory_sf=encoder_output_sf,
                src_mask=src_mask,
                trg_mask=trg_mask,
                query_mask=query_mask,  
            )

        x, x_sf = self.layer_norm(x, x_sf)
        x, x_sf = self.qact_after_ln(x, x_sf)

        output, output_sf = self.output_layer(x, x_sf)
        output, output_sf = self.qact_output(output, output_sf)

        return (output, output_sf), None, None, output_sf

    
    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            len(self.layers),
            self.layers[0].trg_trg_att.num_heads,
        )
