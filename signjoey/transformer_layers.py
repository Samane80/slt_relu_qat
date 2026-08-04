import math
import torch
import torch.nn as nn
import torch.nn.functional as F  
from torch import Tensor

from quantization_utils.quant_modules import (
    QuantAct, QuantConv2d, QuantLinear, QuantMatMul,
    IntGELU, IntLayerNorm, IntSoftmax
)
from quantization_utils.layers_quant import Mlp
from signjoey.relu_attention import IntegerReLUFormerAttention


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, size, dropout=0.1,
                 attention_type: str = "softmax",
                 softmax_bits: int = 16):
        super().__init__()
        assert size % num_heads == 0
        assert attention_type in ["softmax", "reluformer"]

        self.num_heads = num_heads
        self.head_size = size // num_heads
        self.scale = self.head_size ** -0.5
        self.attention_type = attention_type

        self.k_layer = QuantLinear(size, size)
        self.v_layer = QuantLinear(size, size)
        self.q_layer = QuantLinear(size, size)

        self.qact_k = QuantAct()
        self.qact_v = QuantAct()
        self.qact_q = QuantAct()
        self.qact_qscale = QuantAct()

        self.matmul_qk = QuantMatMul()
        self.qact_scores = QuantAct()
        self.softmax = IntSoftmax(softmax_bits)
        self.matmul_av = QuantMatMul()

        self.qact_attn_pre = QuantAct()
        self.qact_attn_post = QuantAct()
        self.qact_ctx = QuantAct()

        self.output_layer = QuantLinear(size, size)
        self.qact_out = QuantAct(16)
        self.dropout = nn.Dropout(dropout)

        self.reluformer = IntegerReLUFormerAttention(num_heads)
        # NOTE: no separate qact_relu_attn here -- IntegerReLUFormerAttention
        # quantizes its own output (see relu_attention.py). Adding another
        # QuantAct after it would double-quantize the reluformer path.

    @property
    def reg_loss(self):
        if (self.attention_type == "reluformer"
                and hasattr(self, "reluformer")
                and self.reluformer.last_reg_loss is not None):
            return self.reluformer.last_reg_loss
        return None

    def _prepare_mask(self, mask, scores):
        if mask is None:
            return None
        mask = mask.to(dtype=torch.bool, device=scores.device)
        if mask.dim() == 2:
            mask = mask[:, None, None, :]
        elif mask.dim() == 3:
            mask = mask[:, None, :, :]
        elif mask.dim() == 4:
            pass
        else:
            raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")
        return mask

    def _mask_scores_for_softmax(self, scores, mask):
        if mask is None:
            return scores
        neg_large = scores.new_tensor(-1e4)
        return scores.masked_fill(~mask, neg_large)

    def forward(self, k, v, q, k_sf, v_sf, q_sf, mask=None,
                attention_type=None, query_mask=None): 
        attn_type = attention_type or self.attention_type
        if attn_type not in ["softmax", "reluformer"]:
            raise ValueError(f"attention_type must be 'softmax' or 'reluformer'")

        B, Nq, _ = q.shape
        Nk = k.shape[1]

        k, k_sf = self.k_layer(k, k_sf)
        v, v_sf = self.v_layer(v, v_sf)
        q, q_sf = self.q_layer(q, q_sf)

        k, k_sf = self.qact_k(k, k_sf)
        v, v_sf = self.qact_v(v, v_sf)
        q, q_sf = self.qact_q(q, q_sf)

        k = k.contiguous().view(B, Nk, self.num_heads, self.head_size).transpose(1, 2)
        v = v.contiguous().view(B, Nk, self.num_heads, self.head_size).transpose(1, 2)
        q = q.contiguous().view(B, Nq, self.num_heads, self.head_size).transpose(1, 2)

        q = q * self.scale
        q, q_sf = self.qact_qscale(q, q_sf)

        scores, s_sf = self.matmul_qk(q, q_sf, k.transpose(2, 3), k_sf)
        m = self._prepare_mask(mask, scores)
        scores, s_sf = self.qact_scores(scores, s_sf)

        if attn_type == "softmax":
            scores_sm = self._mask_scores_for_softmax(scores, m)
            attn, attn_sf = self.softmax(scores_sm, s_sf)
            attn, attn_sf = self.qact_attn_pre(attn, attn_sf)
        else:
            attn, attn_sf = self.reluformer(scores, s_sf, m, query_mask=query_mask)

        attn = self.dropout(attn)
        attn, attn_sf = self.qact_attn_post(attn, attn_sf)

        ctx, ctx_sf = self.matmul_av(attn, attn_sf, v, v_sf)
        ctx = ctx.transpose(1, 2).contiguous().view(B, Nq, self.num_heads * self.head_size)
        ctx, ctx_sf = self.qact_ctx(ctx, ctx_sf)

        out, out_sf = self.output_layer(ctx, ctx_sf)
        out, out_sf = self.qact_out(out, out_sf)
        return out, out_sf


class PositionalEncoding(nn.Module):
    def __init__(self, size: int = 0, max_len: int = 5000):
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super().__init__()
        self.register_buffer("pe", pe)
        self.dim = size
        self.qact = QuantAct(16)

    def forward(self, emb: Tensor, act_scaling_factor: Tensor = None):
        pe = self.pe[:, : emb.size(1)]
        x = emb + pe
        x, sf = self.qact(x)
        return x, sf


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        size: int = 0,
        ff_size: int = 0,
        num_heads: int = 0,
        dropout: float = 0.1,
        attention_type: str = "reluformer",
        dim: int = None,
        mlp_ratio: float = None,
        drop: float = None,
        **kwargs,
    ):
        super().__init__()

        if dim is not None:
            size = dim
        if drop is not None:
            dropout = drop
        if mlp_ratio is not None and ff_size == 0:
            ff_size = int(size * mlp_ratio)

        self.size = size
        self.layer_norm = IntLayerNorm(size, eps=1e-6)
        self.qact_ln = QuantAct()

        self.src_src_att = MultiHeadedAttention(
            num_heads, size, dropout=dropout, attention_type=attention_type
        )
        self.qact_attn = QuantAct()
        self.qact_res1 = QuantAct()
        # self.qact_res2 = QuantAct()

        self.feed_forward = Mlp(
            in_features=size,
            hidden_features=ff_size,
            out_features=size,
            drop=dropout
        )
        self.dropout = nn.Dropout(dropout)

    @property
    def reg_loss(self):
        return getattr(self.src_src_att, "reg_loss", None)

    def forward(self, x: Tensor, x_sf: Tensor, mask: Tensor = None,
                    query_mask: Tensor = None):
            x_norm, ln_sf = self.layer_norm(x, x_sf)
            x_norm, ln_sf = self.qact_ln(x_norm, ln_sf)

            h, h_sf = self.src_src_att(
                x_norm, x_norm, x_norm,
                ln_sf, ln_sf, ln_sf,
                mask=mask,
                query_mask=query_mask,  
            )
            h, h_sf = self.qact_attn(h, h_sf)
            h = self.dropout(h)
            

            res, res_sf = self.qact_res1(
                h, h_sf, identity=x, identity_scaling_factor=x_sf
            )
            o, o_sf = self.feed_forward(res, res_sf)
            return o, o_sf


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        size: int = 0,
        ff_size: int = 0,
        num_heads: int = 0,
        dropout: float = 0.1,
        attention_type: str = "softmax",
        dim: int = None,
        mlp_ratio: float = None,
        drop: float = None,
        **kwargs,
    ):
        super().__init__()
        if dim is not None:
            size = dim
        if drop is not None:
            dropout = drop
        if mlp_ratio is not None and ff_size == 0:
            ff_size = int(size * mlp_ratio)

        self.size = size

        self.x_layer_norm = IntLayerNorm(size, eps=1e-6)
        self.qact_ln1 = QuantAct()
        self.trg_trg_att = MultiHeadedAttention(
            num_heads, size, dropout=dropout, attention_type=attention_type
        )
        self.qact_attn1 = QuantAct()
        self.qact_res1 = QuantAct()

        self.dec_layer_norm = IntLayerNorm(size, eps=1e-6)
        self.qact_ln2 = QuantAct()
        self.src_trg_att = MultiHeadedAttention(
            num_heads, size, dropout=dropout, attention_type=attention_type
        )
        self.qact_attn2 = QuantAct()
        self.qact_res2 = QuantAct()

        self.feed_forward = Mlp(
            in_features=size,
            hidden_features=ff_size,
            out_features=size,
            drop=dropout
        )
        self.dropout = nn.Dropout(dropout)
        # self.qact_res3 = QuantAct()

    @property
    def reg_loss(self):
        losses = [
            l for l in (
                getattr(self.trg_trg_att, "reg_loss", None),
                getattr(self.src_trg_att, "reg_loss", None),
            ) if l is not None
        ]
        if losses:
            return torch.stack(losses).sum()
        return None

    def forward(
        self,
        x: Tensor,
        x_sf: Tensor,
        memory: Tensor,
        memory_sf: Tensor,
        src_mask: Tensor = None,
        trg_mask: Tensor = None,
        query_mask: Tensor = None, 
    ):
        x_norm, ln1_sf = self.x_layer_norm(x, x_sf)
        x_norm, ln1_sf = self.qact_ln1(x_norm, ln1_sf)

        h1, h1_sf = self.trg_trg_att(
            x_norm, x_norm, x_norm,
            ln1_sf, ln1_sf, ln1_sf,
            mask=trg_mask,
            query_mask=query_mask, 
        )
        h1, h1_sf = self.qact_attn1(h1, h1_sf)
        h1 = self.dropout(h1)
        h1, h1_sf = self.qact_res1(
            h1, h1_sf, identity=x, identity_scaling_factor=x_sf
        )

        h1_norm, ln2_sf = self.dec_layer_norm(h1, h1_sf)
        h1_norm, ln2_sf = self.qact_ln2(h1_norm, ln2_sf)

        h2, h2_sf = self.src_trg_att(
            memory, memory, h1_norm,
            memory_sf, memory_sf, ln2_sf,
            mask=src_mask,
        )
        h2, h2_sf = self.qact_attn2(h2, h2_sf)
        h2 = self.dropout(h2)
        h2, h2_sf = self.qact_res2(
            h2, h2_sf, identity=h1, identity_scaling_factor=h1_sf
        )

        o, o_sf = self.feed_forward(h2, h2_sf)
        # ff, ff_sf = self.feed_forward(h2, h2_sf)
        # o, o_sf = self.qact_res3(ff, ff_sf, identity=h2, identity_scaling_factor=h2_sf)

        return o, o_sf
