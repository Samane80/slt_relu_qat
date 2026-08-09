# coding: utf-8
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from signjoey.helpers import freeze_params
from quantization_utils.quant_modules import (
    QuantLinear,
    QuantAct,
    IntLayerNorm,
    IntGELU,
    IntBatchNorm1d,
    IntSoftsign,
)


def get_activation(activation_type):
    """
    integer-only QAT pipeline: از gelu (IntGELU) و softsign (IntSoftsign)
    پشتیبانی می‌کند. سایر Activationها مثل nn.ReLU امضای (x, scaling_factor)
    را پشتیبانی نمی‌کنند.
    """
    if activation_type == "gelu":
        return IntGELU()
    elif activation_type == "softsign":
        return IntSoftsign()
    else:
        raise ValueError(
            f"Unsupported activation type '{activation_type}' for integer-only QAT. "
            "Only 'gelu' (IntGELU) and 'softsign' (IntSoftsign) are supported."
        )


class MaskedNorm(nn.Module):
    """
    Masked LayerNorm/BatchNorm for integer-only QAT pipeline.

    اصلاحات:
    - GroupNorm همچنان پشتیبانی نمی‌شود (فرمول per-group statistics با
      همان STE-reciprocal trick سازگار نیست).
    - از این پس هم IntLayerNorm و هم IntBatchNorm1d پشتیبانی می‌شوند، دقیقاً
      مثل نسخه‌ی plain (که از nn.BatchNorm1d/nn.LayerNorm استفاده می‌کرد)،
      و با همان state_dict روی quantize=False و quantize=True کار می‌کنند.
    """

    def __init__(self, norm_type, num_groups, num_features):
        super().__init__()
        self.norm_type = norm_type

        if self.norm_type == "group":
            raise ValueError(
                "GroupNorm is not supported in the integer-only QAT pipeline. "
                "Please use norm_type='layer' or norm_type='batch'."
            )
        elif self.norm_type == "batch":
            self.norm = IntBatchNorm1d(num_features)
        elif self.norm_type == "layer":
            self.norm = IntLayerNorm(normalized_shape=num_features)
        else:
            raise ValueError(f"Unsupported Normalization Layer: {norm_type}")

        self.num_features = num_features

    def forward(self, x: Tensor, mask: Tensor, act_scaling_factor=None):
        if act_scaling_factor is None:
            raise ValueError("act_scaling_factor is required for MaskedNorm")

        sf = act_scaling_factor.reshape(-1).mean().view(1)

        if self.training:
            reshaped = x.reshape([-1, self.num_features])
            reshaped_mask = mask.reshape([-1, 1]) > 0
            selected = torch.masked_select(reshaped, reshaped_mask).reshape(
                [-1, self.num_features]
            )
            normed, sf_out = self.norm(selected, sf)
            scattered = reshaped.masked_scatter(reshaped_mask, normed)
            return (
                scattered.reshape([x.shape[0], -1, self.num_features]),
                sf_out,
            )
        else:
            reshaped = x.reshape([-1, self.num_features])
            normed, sf_out = self.norm(reshaped, sf)
            return normed.reshape([x.shape[0], -1, self.num_features]), sf_out

# def get_activation(activation_type):
#     """
#     در پایپ‌لاین Integer-Only QAT، فقط از IntGELU پشتیبانی می‌شود.
#     سایر Activationها مثل nn.ReLU امضای (x, scaling_factor) را پشتیبانی نمی‌کنند.
#     """
#     if activation_type == "gelu":
#         return IntGELU()
#     else:
#         raise ValueError(
#             f"Unsupported activation type '{activation_type}' for integer-only QAT. "
#             "Only 'gelu' (IntGELU) is supported."
#         )


# class MaskedNorm(nn.Module):
#     """
#     Masked LayerNorm for integer-only QAT pipeline.
    
#     اصلاحات:
#     - BatchNorm و GroupNorm به دلیل عدم سازگاری با Per-Tensor QuantAct 
#       به صورت عمدی غیرمجاز (Raise Error) اعلام شدند.
#     - فقط از IntLayerNorm استفاده می‌شود.
#     """

#     def __init__(self, norm_type, num_groups, num_features):
#         super().__init__()
#         self.norm_type = norm_type
        
#         if self.norm_type in ["batch", "group"]:
#             raise ValueError(
#                 "BatchNorm and GroupNorm are not supported in the integer-only QAT pipeline. "
#                 "Please use norm_type='layer' (IntLayerNorm)."
#             )
#         elif self.norm_type == "layer":
#             self.norm = IntLayerNorm(normalized_shape=num_features)
#         else:
#             raise ValueError(f"Unsupported Normalization Layer: {norm_type}")

#         self.num_features = num_features

#     def forward(self, x: Tensor, mask: Tensor, act_scaling_factor=None):
#         if act_scaling_factor is None:
#             raise ValueError("act_scaling_factor is required for IntLayerNorm")
        
#         # اطمینان از اسکالر بودن scaling factor
#         sf = act_scaling_factor.reshape(-1).mean().view(1)
        
#         if self.training:
#             reshaped = x.reshape([-1, self.num_features])
#             reshaped_mask = mask.reshape([-1, 1]) > 0
#             selected = torch.masked_select(reshaped, reshaped_mask).reshape(
#                 [-1, self.num_features]
#             )
#             normed, sf_out = self.norm(selected, sf)
#             scattered = reshaped.masked_scatter(reshaped_mask, normed)
#             return (
#                 scattered.reshape([x.shape[0], -1, self.num_features]),
#                 sf_out,
#             )
#         else:
#             reshaped = x.reshape([-1, self.num_features])
#             normed, sf_out = self.norm(reshaped, sf)
#             return normed.reshape([x.shape[0], -1, self.num_features]), sf_out


# Embeddings  —  Word/Gloss embedding (text side, decoder)

class Embeddings(nn.Module):
    """
    Quantization-aware word/gloss embeddings.

    تغییرات:
    - nn.Embedding: کوانتیزه نمیشه (integer index داره)
    - norm_type=="layer": IntLayerNorm به جای nn.LayerNorm
    - activation=="gelu": IntGELU به جای nn.GELU
    - QuantAct بعد از embedding و بعد از هر مرحله
    - forward: (x, mask) → (x, sf)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 8,
        scale: bool = False,
        scale_factor: float = None,
        norm_type: str = None,
        activation_type: str = None,
        vocab_size: int = 0,
        padding_idx: int = 1,
        freeze: bool = False,
        **kwargs
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.lut = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=padding_idx)
        nn.init.normal_(self.lut.weight, mean=0, std=embedding_dim ** -0.5)

        self.qact_input = QuantAct()

        self.norm_type = norm_type
        if self.norm_type:
            self.norm = MaskedNorm(
                norm_type=norm_type, num_groups=num_heads, num_features=embedding_dim
            )
            self.qact_after_norm = QuantAct()

        self.activation_type = activation_type
        if self.activation_type:
            self.activation = get_activation(activation_type)
            # چون فقط gelu پشتیبانی میشه، همیشه به QuantAct بعد از اون نیاز داریم
            self.qact_after_act = QuantAct()

        self.scale = scale
        if self.scale:
            self.scale_factor = scale_factor if scale_factor else math.sqrt(embedding_dim)

        if freeze:
            freeze_params(self)

    def forward(self, x: Tensor, mask: Tensor = None):
        """
        :param x: [B, U] — index کلمات
        :param mask: [B, 1, U] — padding mask
        :return: (x, sf) — embedding کوانتیزه + scalar sf
        """
        # lookup — float32
        x = self.lut(x)  # [B, U, dim]


        if self.scale:
            x = x * self.scale_factor

        x, sf = self.qact_input(x)

        if self.norm_type:
            x, sf = self.norm(x, mask, sf)
            x, sf = self.qact_after_norm(x, sf)

        if self.activation_type:
            if self.activation_type == "softsign":
                x, sf = self.activation(x, sf)
                x, sf = self.qact_after_act(x, sf)
            else:
                raise ValueError(
                    f"Only 'gelu' (IntGELU) is supported in the integer-only QAT pipeline. "
                    f"Got activation_type='{self.activation_type}'"
                )

        return x, sf

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__,
            self.embedding_dim,
            self.vocab_size,
        )


# SpatialEmbeddings  —  Sign/video embedding (encoder side)

class SpatialEmbeddings(nn.Module):
    """
    Quantization-aware spatial embedding برای feature vector فریم‌های ویدیو.

    تغییرات:
    - nn.Linear → QuantLinear (per-channel weight)
    - norm_type=="layer" → IntLayerNorm
    - activation=="gelu" → IntGELU
    - QuantAct بعد از هر مرحله
    - forward: (x, mask) → (x, sf)
    """

    def __init__(
        self,
        embedding_dim: int,
        input_size: int,
        num_heads: int,
        freeze: bool = False,
        norm_type: str = None,
        activation_type: str = None,
        scale: bool = False,
        scale_factor: float = None,
        **kwargs
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.input_size = input_size

        # اولین QuantAct — sf اولیه از feature خام ویدیو
        self.qact_input = QuantAct()

        # QuantLinear به جای nn.Linear
        self.ln = QuantLinear(self.input_size, self.embedding_dim)

        # QuantAct بعد از linear
        self.qact_after_ln = QuantAct(16)

        self.norm_type = norm_type
        if self.norm_type:
            self.norm = MaskedNorm(
                norm_type=norm_type, num_groups=num_heads, num_features=embedding_dim
            )
            self.qact_after_norm = QuantAct()

        self.activation_type = activation_type
        if self.activation_type:
            self.activation = get_activation(activation_type)
            self.qact_after_act = QuantAct()

        self.scale = scale
        if self.scale:
            self.scale_factor = scale_factor if scale_factor else math.sqrt(embedding_dim)

        if freeze:
            freeze_params(self)

    def forward(self, x: Tensor, mask: Tensor):
        """
        :param x: [B, T, input_size] — feature vector فریم‌های ویدیو
        :param mask: [B, 1, T] — frame mask
        :return: (x, sf) — embedding کوانتیزه + scalar sf
        """
        #  sf اول اینجا ساخته میشه از feature خام
        x, sf = self.qact_input(x)

        #  QuantLinear — weight per-channel کوانتیزه
        x, sf = self.ln(x, sf)
        x, sf = self.qact_after_ln(x, sf)

        if self.norm_type:
            x, sf = self.norm(x, mask, sf)
            x, sf = self.qact_after_norm(x, sf)

        if self.activation_type:
            if self.activation_type == "softsign":
                x, sf = self.activation(x, sf)
                x, sf = self.qact_after_act(x, sf)
            else:
                raise ValueError(
                    f"Only 'gelu' (IntGELU) is supported in the integer-only QAT pipeline. "
                    f"Got activation_type='{self.activation_type}'"
                )


        if self.scale:
            x = x * self.scale_factor
            sf = sf * self.scale_factor  

        return x, sf

    def __repr__(self):
        return "%s(embedding_dim=%d, input_size=%d)" % (
            self.__class__.__name__,
            self.embedding_dim,
            self.input_size,
        )
