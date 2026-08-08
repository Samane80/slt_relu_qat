# coding: utf-8
"""
Integer-only ReLUFormer attention, following the I-ViT methodology
(Li & Gu, "I-ViT: Integer-only Quantization for Efficient ViT Inference", ICCV 2023).

I-ViT's contributions:
  * Dyadic Arithmetic (DA) requantization: a real scale ratio S is represented as
    M / 2**shift, so a rescale becomes   y_int = (M * x_int) >> shift  (integer only).
  * Shiftmax / ShiftGELU / I-LayerNorm to make softmax / GELU / LayerNorm integer.

ReLUFormer has NO softmax, so Shiftmax is not needed here. Its attention is:
    weights = ReLU(scores) * (1/gamma) * 1/sqrt(n/2)
  - ReLU is integer-exact: max(x, 0).
  - (1/gamma) and 1/sqrt(n/2) are positive rescales -> implemented with DA.

Training  -> `forward`          : returns (weights, scaling_factor) -- a real tuple,
              like every other layer in quant_modules.py, so it can be used
              as a drop-in replacement with no external re-quantization step.
              `self.quantize` (toggled by model_utils.set_quantize_mode)
              switches between:
                - quantize=True  (QAT):  fake-quant + STE, computes reg loss.
                  Takes an EXPLICIT `scaling_factor` for the incoming scores
                  (the same one produced by MultiHeadedAttention.qact_scores)
                  instead of re-calibrating its own -- this module is the
                  ONLY place that quantizes the reluformer *output*; it must
                  not also quantize the *input* independently, or two
                  uncalibrated scales get stacked.
                - quantize=False (FP32): plain float attention, `scores` used
                  as-is (no STE rounding), dummy sf=1.0 returned. Needed
                  because with quantize=True and a dummy sf=1.0 (as produced
                  by an upstream QuantAct(quantize=False)), the STE rounding
                  step would collapse attention scores to the nearest
                  integer -- this branch avoids that entirely.
Inference -> `forward_integer`  : pure integer graph, returns (weights_int, out_scale).
              `1/gamma` and `1/sqrt(n/2)` are reduced to ONE scalar dyadic
              coefficient (M, shift) per call -- computed on a single float,
              never on the [B, H, Q, K] attention tensor -- so the only ops
              touching the full tensor are integer add/mul/shift/clamp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Observer (symmetric option added for signed scores)
class MinMaxObserver(nn.Module):
    """Min/Max observer + fake-quantize (STE) with EMA (momentum) support."""

    def __init__(self, unsigned=False, num_bits=8, eps=1e-8,
                 symmetric=False, momentum=0.9):
        super().__init__()
        self.unsigned = unsigned
        self.num_bits = num_bits
        self.eps = eps
        self.symmetric = symmetric
        self.momentum = momentum

        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0, dtype=torch.int32))

        if unsigned:
            self.qmin, self.qmax = 0, 2 ** num_bits - 1
        else:
            self.qmin, self.qmax = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1

    def _update_qparams(self):
        min_val, max_val = self.min_val, self.max_val
        if self.unsigned:
            min_val = torch.zeros_like(min_val)
        if self.symmetric and not self.unsigned:
            max_abs = torch.maximum(min_val.abs(), max_val.abs())
            min_val, max_val = -max_abs, max_abs

        scale = (max_val - min_val).clamp(min=self.eps) / float(self.qmax - self.qmin)
        zp = self.qmin - torch.round(min_val / scale)
        zp = zp.clamp(self.qmin, self.qmax).to(torch.int32)
        self.scale.copy_(scale)
        self.zero_point.copy_(zp)

    def forward(self, x: Tensor) -> Tensor:
        if self.training or self.num_batches_tracked == 0:
            cur_min = x.detach().min()
            cur_max = x.detach().max()
            if self.unsigned:
                cur_min = torch.zeros_like(cur_min)
            if self.num_batches_tracked == 0:
                self.min_val, self.max_val = cur_min, cur_max
            else:
                self.min_val = self.min_val * self.momentum + cur_min * (1 - self.momentum)
                self.max_val = self.max_val * self.momentum + cur_max * (1 - self.momentum)
            self.num_batches_tracked += 1
            self._update_qparams()

        scale, zp = self.scale, self.zero_point
        x_int = torch.clamp(torch.round(x / scale) + zp, self.qmin, self.qmax)
        x_dequant = (x_int - zp) * scale
        return x + (x_dequant - x).detach()  # STE

    def calculate_qparams(self):
        return self.scale.clone(), self.zero_point.clone()

    def fix(self):
        self.training_frozen = True

    def reset(self):
        self.min_val.fill_(float("inf"))
        self.max_val.fill_(float("-inf"))
        self.num_batches_tracked.zero_()
        self.scale.fill_(1.0)
        self.zero_point.zero_()


# I-ViT Dyadic Arithmetic helpers
def to_dyadic(scale_ratio: Tensor, mult_bits: int = 16):
    """
    Approximate a positive real (scalar or tensor) `scale_ratio` by a dyadic
    number M / 2**shift. Chooses the largest shift that keeps M within
    `mult_bits` precision, matching I-ViT's get_scale_approximation.

    IMPORTANT: for a true integer-only *inference* graph this must only ever
    be called on small, parameter-like tensors (ideally a single scalar) --
    never on a tensor whose size scales with the attention matrix -- since
    log2/pow here are float ops. See `forward_integer` below for the correct
    usage (one scalar coefficient per call).
    """
    scale_ratio = scale_ratio.clamp(min=1e-12)
    shift = torch.floor(torch.log2((2 ** mult_bits - 1) / scale_ratio)).clamp(min=0.0)
    M = torch.floor(scale_ratio * torch.pow(2.0, shift) + 0.5)
    return M, shift


def dyadic_apply(x_int: Tensor, M: Tensor, shift: Tensor):
    """Integer rescale: round((M * x_int) / 2**shift) — i.e. (M * x_int) >> shift."""
    two_pow = torch.pow(2.0, shift)
    return torch.floor((M * x_int) / two_pow + 0.5)


def _quantize_small_tensor(x: Tensor, num_bits: int = 8, unsigned: bool = True, eps: float = 1e-8):
    """
    One-shot (no EMA state) symmetric/affine quantization for small,
    parameter-like tensors (e.g. `1/gamma`, only `num_heads` values, or a
    static LUT). Returns (x_integer, scale) with a single per-tensor scale.
    """
    x = x.detach()
    if unsigned:
        min_val = torch.zeros_like(x.min())
        max_val = x.max()
        qmin, qmax = 0, 2 ** num_bits - 1
    else:
        max_abs = x.abs().max()
        min_val, max_val = -max_abs, max_abs
        qmin, qmax = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1

    scale = (max_val - min_val).clamp(min=eps) / float(qmax - qmin)
    zero_point = qmin - torch.round(min_val / scale)
    zero_point = zero_point.clamp(qmin, qmax)

    x_int = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax) - zero_point
    return x_int, scale.view(1)


# Integer-only ReLUFormer attention
class IntegerReLUFormerAttention(nn.Module):
    def __init__(self, num_heads, entropy_margin_coeff=0.7, eps=1e-8,
                 max_seq_len=512, mult_bits=16, score_bits=8, out_bits=8,
                 lut_bit=8, gamma_bit=8, quantize=True):
        super().__init__()
        self.num_heads = num_heads
        self.log_gamma = nn.Parameter(torch.zeros(num_heads))
        self.entropy_margin_coeff = entropy_margin_coeff
        self.eps = eps
        # Same convention as QuantAct/QuantLinear/... in quant_modules.py:
        # toggled in-place by model_utils.set_quantize_mode(model, ...).
        # quantize=False -> plain float attention: no STE rounding of the
        # input scores, no fake-quantization of the output weights.
        self.quantize = quantize
        self.mult_bits = mult_bits
        self.score_bits = score_bits
        self.out_bits = out_bits
        self.lut_bit = lut_bit
        self.gamma_bit = gamma_bit
        self.qmax_out = 2 ** out_bits - 1
        self.max_seq_len = max_seq_len
        self.last_reg_loss = None

        # ---- float LUT: used only on the QAT/training path (forward) ----
        n_values = torch.arange(1, max_seq_len + 1, dtype=torch.float32)
        self.register_buffer("inv_sqrt_lut", 1.0 / torch.sqrt(n_values / 2.0))

        # ---- pre-quantized INTEGER LUT: used on the integer inference
        # path (forward_integer). Computed ONCE here since 1/sqrt(n/2)
        # never changes -- no per-call float math needed at inference. ----
        lut_int, lut_scale = _quantize_small_tensor(self.inv_sqrt_lut, num_bits=lut_bit, unsigned=True)
        self.register_buffer("inv_sqrt_lut_integer", lut_int)
        self.register_buffer("inv_sqrt_lut_scale", lut_scale)

        # attention weights are >= 0 -> unsigned output quantizer.
        self.weight_observer = MinMaxObserver(unsigned=True, num_bits=out_bits)

        self._gamma_frozen = False

    def fix(self):
        """Freeze gamma for deployment/export (stop training it, and lock
        the calibrated output scale). Call this AFTER QAT fine-tuning and
        BEFORE using `forward_integer` for real deployment."""
        self.log_gamma.requires_grad_(False)
        self.weight_observer.running_stat = False
        self._gamma_frozen = True

    def unfix(self):
        self.log_gamma.requires_grad_(True)
        self.weight_observer.running_stat = True
        self._gamma_frozen = False

    # ----- mask helpers -----
    @staticmethod
    def _normalize_mask(mask, B, Q, K):
        if mask is None:
            return None
        mask = mask.to(dtype=torch.bool)
        if mask.dim() == 2:
            mask = mask[:, None, :]
        elif mask.dim() == 3:
            pass
        elif mask.dim() == 4:
            mask = mask.squeeze(1) if mask.size(1) == 1 else mask[:, 0]
        else:
            raise ValueError(f"Unsupported mask dim: {mask.dim()}")
        if mask.size(-1) != K:
            raise ValueError(f"Mask last dim {mask.size(-1)} != k_len {K}")
        return mask

    @staticmethod
    def _normalize_query_mask(query_mask, B, Q):
        if query_mask is None:
            return None
        query_mask = query_mask.to(dtype=torch.bool)
        if query_mask.dim() == 2:
            pass
        elif query_mask.dim() == 3:
            if query_mask.size(1) == 1:
                query_mask = query_mask.squeeze(1)
            elif query_mask.size(2) == 1:
                query_mask = query_mask.squeeze(2)
            else:
                raise ValueError(f"bad query_mask 3D: {tuple(query_mask.shape)}")
        elif query_mask.dim() == 4:
            query_mask = query_mask.squeeze(1).squeeze(1)
        else:
            raise ValueError("query_mask must be 2, 3 or 4 dims")
        if query_mask.size(-1) != Q:
            raise ValueError(f"query_mask last dim {query_mask.size(-1)} != q_len {Q}")
        return query_mask

    @staticmethod
    def _valid_counts(mask, B, Q, K, device, dtype):
        if mask is None:
            return torch.full((B, Q), K, device=device, dtype=dtype)
        counts = mask.sum(dim=-1).to(dtype)
        if counts.size(1) == 1 and Q > 1:
            counts = counts.expand(-1, Q)
        return counts

    def _lookup_inv_sqrt(self, n: Tensor) -> Tensor:
        idx = n.long().clamp(min=1, max=self.inv_sqrt_lut.numel()) - 1
        return self.inv_sqrt_lut[idx]

    def _lookup_inv_sqrt_integer(self, n: Tensor) -> Tensor:
        idx = n.long().clamp(min=1, max=self.inv_sqrt_lut_integer.numel()) - 1
        return self.inv_sqrt_lut_integer[idx]

    # QAT / simulation path (drop-in replacement, returns float weights)
    def forward(self, scores, scaling_factor, mask=None, query_mask=None):
        """
        :param scores: [B, H, Q, K] dequantized (int * scale) pre-activation,
            already produced upstream by MultiHeadedAttention.qact_scores.
        :param scaling_factor: scalar tensor, the scale of `scores`. Passed
            in explicitly -- this module does NOT run its own score
            observer, so the reluformer *input* is only ever quantized
            once (by the caller), and this module only quantizes its
            *output* (via `weight_observer`).
        """
        B, H, Q, K = scores.shape
        mask = self._normalize_mask(mask, B, Q, K)
        qm = self._normalize_query_mask(query_mask, B, Q)

        if not self.quantize:
            # ---- FP32 path: use `scores` as-is. Also protects against a
            # dummy sf=1.0 coming from an upstream QuantAct(quantize=False),
            # which would otherwise round attention scores to the nearest
            # integer and destroy the signal.
            scores_q = scores
        else:
            # STE re-alignment to the grid implied by the given scale (no
            # new calibration happens here -- `sf` is fixed, this just
            # guarantees `scores` sits on an integer*scale grid before ReLU).
            sf = scaling_factor.reshape(-1).mean()
            scores_int = torch.round(scores / sf)
            scores_q = scores_int * sf
            scores_q = scores + (scores_q - scores).detach()  # STE passthrough

        relu_scores = F.relu(scores_q)
        if mask is not None:
            relu_scores = relu_scores.masked_fill(~mask.unsqueeze(1), 0.0)

        n = self._valid_counts(mask, B, Q, K, scores.device, scores.dtype).clamp(min=1.0)
        gamma = torch.exp(self.log_gamma).view(1, H, 1, 1)
        inv_gamma = 1.0 / (gamma + self.eps)
        inv_sqrt = self._lookup_inv_sqrt(n).unsqueeze(1).unsqueeze(-1)  # [B,1,Q,1]
        weights = relu_scores * inv_gamma * inv_sqrt

        # regularization loss (train only) — on continuous weights
        if self.training:
            sum_w = weights.sum(dim=-1)  # [B,H,Q]
            if qm is not None:
                valid_q = qm.unsqueeze(1).float()
            else:
                valid_q = (sum_w > 0).float()
            valid_count = valid_q.sum().clamp(min=1e-8)

            norm_reg = (torch.log(sum_w + self.eps).abs() * valid_q).sum() / valid_count

            probs = weights / (sum_w.unsqueeze(-1) + self.eps)
            probs_safe = probs.clamp(min=1e-8)
            entropy = -(probs * torch.log(probs_safe)).sum(dim=-1)
            c_bound = (self.entropy_margin_coeff * torch.log(n.clamp(min=2.0))).unsqueeze(1)
            entropy_reg = (F.relu(entropy - c_bound) * valid_q).sum() / valid_count

            self.last_reg_loss = norm_reg + entropy_reg
        else:
            self.last_reg_loss = None

        if not self.quantize:
            # FP32 path: plain float weights, dummy sf (matches the
            # convention used by every other layer in quant_modules.py).
            return weights, torch.ones(1, device=weights.device)

        # single quantization point for this module's OUTPUT
        weights_q = self.weight_observer(weights)
        return weights_q, self.weight_observer.scale.view(1)

    # Pure integer inference path (I-ViT deployment kernel)
    @torch.no_grad()
    def forward_integer(self, scores_int, scores_scale, mask=None, out_scale=None):
        """
        Args:
            scores_int   : INT tensor [B,H,Q,K] from the integer QK^T matmul
                            (already on the qact_scores integer grid).
            scores_scale : python float / 0-d tensor, real_scores = scores_int * scores_scale.
            out_scale    : float S_out for the returned weights (defaults to the
                           calibrated weight_observer scale; call `.fix()`
                           first so this scale is actually frozen/stable).
        Returns:
            weights_int (int32), out_scale (float);
            real_weights = weights_int * out_scale.

        Only ONE float computation happens here (`to_dyadic` on a single
        scalar `combined_scale`). Every operation touching the full
        [B, H, Q, K] tensor -- ReLU, masking, the two multiplications, and
        the final rescale -- is plain integer add/mul/shift/clamp.
        """
        if not self._gamma_frozen:
            raise RuntimeError(
                "Call self.fix() before forward_integer(): gamma and the "
                "output scale must be frozen for a stable integer graph."
            )

        B, H, Q, K = scores_int.shape
        device = scores_int.device
        mask = self._normalize_mask(mask, B, Q, K)

        # ReLU is integer-exact
        relu_int = torch.clamp(scores_int, min=0)
        if mask is not None:
            relu_int = relu_int.masked_fill(~mask.unsqueeze(1), 0)

        n = self._valid_counts(mask, B, Q, K, device, torch.float32).clamp(min=1.0)
        idx = n.round().long().clamp(min=1, max=self.max_seq_len) - 1
        inv_sqrt_int = self.inv_sqrt_lut_integer[idx]          # [B, Q], static, pre-quantized
        inv_sqrt_int = inv_sqrt_int.view(B, 1, Q, 1)

        # 1/gamma: only `num_heads` scalars -> cheap one-shot quantization,
        # NOT a per-forward calibration loop over the attention tensor.
        inv_gamma = 1.0 / (torch.exp(self.log_gamma) + self.eps)
        inv_gamma_int, inv_gamma_scale = _quantize_small_tensor(
            inv_gamma, num_bits=self.gamma_bit, unsigned=True
        )
        inv_gamma_int = inv_gamma_int.view(1, H, 1, 1).to(device)

        # ---- pure integer multiply over the full tensor ----
        weights_int = relu_int * inv_gamma_int * inv_sqrt_int

        if out_scale is None:
            out_scale = float(self.weight_observer.scale)

        # ---- ONE scalar dyadic coefficient, computed once per call ----
        combined_scale = (
            float(scores_scale)
            * float(inv_gamma_scale)
            * float(self.inv_sqrt_lut_scale)
            / out_scale
        )
        M, shift = to_dyadic(torch.tensor([combined_scale], device=device), self.mult_bits)
        weights_int = dyadic_apply(weights_int, M.view(1, 1, 1, 1), shift.view(1, 1, 1, 1))
        weights_int = torch.clamp(weights_int, 0, self.qmax_out).to(torch.int32)
        return weights_int, out_scale

    def __repr__(self):
        return (f"IntegerReLUFormerAttention(num_heads={self.log_gamma.numel()}, "
                f"mult_bits={self.mult_bits}, out_bits={self.out_bits}, "
                f"frozen={self._gamma_frozen})")
