# coding: utf-8
"""
Encoder implementations with QAT support.
"""
import torch
import torch.nn as nn
from torch import Tensor

from quantization_utils.quant_modules import QuantAct, IntLayerNorm
from signjoey.transformer_layers import TransformerEncoderLayer,PositionalEncoding


class Encoder(nn.Module):
    """Base encoder class."""
    
    @property
    def output_size(self):
        """Return the output size of the encoder."""
        return self._output_size


class TransformerEncoder(Encoder):
    """Transformer Encoder."""

    def __init__(
        self,
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        attention_type: str = "reluformer",
        **kwargs,
    ):
        super().__init__()

        # ✅ تنظیم output_size در base class
        self._output_size = hidden_size
        self.pe = PositionalEncoding(size=hidden_size)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
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

    @property
    def reg_loss(self):
        """
        Return the sum of regularization losses from all layers.
        """
        losses = []
        for layer in self.layers:
            if hasattr(layer, 'reg_loss') and layer.reg_loss is not None:
                losses.append(layer.reg_loss)
        
        if losses:
            return torch.stack(losses).sum()
        return None

    def forward(
        self,
        embed_src: Tensor,
        src_length: Tensor,
        mask: Tensor,
        act_scaling_factor: Tensor = None,
        query_mask: Tensor = None,
    ):
        """
        Pass the input through the transformer encoder layers.
        """
        x = embed_src
        x_sf = act_scaling_factor
        
        x, x_sf = self.pe(x, x_sf)
        x = self.dropout(x)
        
        x, x_sf = self.qact_after_dropout(x, x_sf)

        for layer in self.layers:
            x, x_sf = layer(x, x_sf, mask=mask, query_mask=query_mask)

        x, x_sf = self.layer_norm(x, x_sf)
        x, x_sf = self.qact_after_ln(x, x_sf)

        return x, x_sf


# سایر encoder ها (RecurrentEncoder و ...) در اینجا قرار می‌گیرند