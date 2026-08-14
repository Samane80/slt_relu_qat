# coding: utf-8
"""
Implements custom initialization (QAT-safe)
"""
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import _calculate_fan_in_and_fan_out


def orthogonal_rnn_init_(cell: nn.RNNBase, gain: float = 1.0):
    with torch.no_grad():
        for _, hh, _, _ in cell.all_weights:
            for i in range(0, hh.size(0), cell.hidden_size):
                nn.init.orthogonal_(hh.data[i : i + cell.hidden_size], gain=gain)


def lstm_forget_gate_init_(cell: nn.RNNBase, value: float = 1.0) -> None:
    with torch.no_grad():
        for _, _, ih_b, hh_b in cell.all_weights:
            l = len(ih_b)
            ih_b.data[l // 4 : l // 2].fill_(value)
            hh_b.data[l // 4 : l // 2].fill_(value)


def xavier_uniform_n_(w: Tensor, gain: float = 1.0, n: int = 4) -> None:
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out //= n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)


# نام‌هایی که نباید با init معمولی خراب شوند (buffers / quant state / ReLUFormer)
_SKIP_INIT_SUBSTRINGS = (
    "min_val",
    "max_val",
    "act_scaling_factor",
    "fc_scaling_factor",
    "conv_scaling_factor",
    "weight_integer",
    "bias_integer",
    "norm_scaling_factor",
    "num_batches_tracked",
    "inv_sqrt_lut",       # LUT ثابت ReLUFormer
    "zero_point",
    "scale",              # اگر buffer به اسم scale داشته باشی
)


def initialize_model(model: nn.Module, cfg: dict, txt_padding_idx: int) -> None:
    gain = float(cfg.get("init_gain", 1.0))
    init = cfg.get("initializer", "xavier")
    init_weight = float(cfg.get("init_weight", 0.01))
    embed_init = cfg.get("embed_initializer", "normal")
    embed_init_weight = float(cfg.get("embed_init_weight", 0.01))
    embed_gain = float(cfg.get("embed_init_gain", 1.0))
    bias_init = cfg.get("bias_initializer", "zeros")
    bias_init_weight = float(cfg.get("bias_init_weight", 0.01))

    def _parse_init(s, scale, _gain):
        scale = float(scale)
        assert scale > 0.0, "incorrect init_weight"
        s = s.lower()
        if s == "xavier":
            return lambda p: nn.init.xavier_uniform_(p, gain=_gain)
        if s == "uniform":
            return lambda p: nn.init.uniform_(p, a=-scale, b=scale)
        if s == "normal":
            return lambda p: nn.init.normal_(p, mean=0.0, std=scale)
        if s == "zeros":
            return lambda p: nn.init.zeros_(p)
        raise ValueError("unknown initializer")

    init_fn_ = _parse_init(init, init_weight, gain)
    embed_init_fn_ = _parse_init(embed_init, embed_init_weight, embed_gain)
    bias_init_fn_ = _parse_init(bias_init, bias_init_weight, gain)

    with torch.no_grad():
        for name, p in model.named_parameters():
            # --- skip quant / LUT / observer state ---
            if any(s in name for s in _SKIP_INIT_SUBSTRINGS):
                continue

            # ReLUFormer: log_gamma باید 0 بماند (gamma=1)
            if "log_gamma" in name:
                p.zero_()
                continue

            # Normalization scale parameters must start at one, just like
            # PyTorch's BatchNorm/LayerNorm defaults.  The old generic
            # one-dimensional-parameter branch initialized them to zero,
            # collapsing the pre-LN Transformer until those scales learned
            # their way out of zero.  This check comes before the embedding
            # branch because embedding norms are nested under txt/sgn_embed.
            if name.endswith("norm.weight"):
                p.fill_(1.0)
                continue

            # --- embeddings ---
            if any(k in name for k in ("txt_embed", "gls_embed", "sgn_embed")):
                if "lut" in name:
                    embed_init_fn_(p)
                elif "bias" in name:
                    bias_init_fn_(p)
                elif p.dim() > 1:
                    init_fn_(p)
                else:
                    bias_init_fn_(p)
                continue

            # --- biases and other one-dimensional parameters ---
            if "bias" in name or p.dim() == 1:
                bias_init_fn_(p)
                continue

            # --- weight matrices ---
            if p.dim() > 1:
                if init == "xavier" and "rnn" in name:
                    n = 1
                    if "encoder" in name and hasattr(model.encoder, "rnn"):
                        n = 4 if isinstance(model.encoder.rnn, nn.LSTM) else 3
                    elif "decoder" in name and hasattr(model.decoder, "rnn"):
                        n = 4 if isinstance(model.decoder.rnn, nn.LSTM) else 3
                    xavier_uniform_n_(p.data, gain=gain, n=n)
                else:
                    init_fn_(p)

        # zero padding row in text embedding
        if getattr(model, "txt_embed", None) is not None:
            if hasattr(model.txt_embed, "lut"):
                model.txt_embed.lut.weight.data[txt_padding_idx].zero_()

        # RNN extras
        orthogonal = cfg.get("init_rnn_orthogonal", False)
        lstm_forget_gate = cfg.get("lstm_forget_gate", 1.0)

        if hasattr(model.encoder, "rnn"):
            if orthogonal:
                orthogonal_rnn_init_(model.encoder.rnn)
            if isinstance(model.encoder.rnn, nn.LSTM):
                lstm_forget_gate_init_(model.encoder.rnn, lstm_forget_gate)

        if hasattr(model.decoder, "rnn"):
            if orthogonal:
                orthogonal_rnn_init_(model.decoder.rnn)
            if isinstance(model.decoder.rnn, nn.LSTM):
                lstm_forget_gate_init_(model.decoder.rnn, lstm_forget_gate)