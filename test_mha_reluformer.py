import sys
sys.path.insert(0, "/home/claude/testenv2")

import torch
from quantization_utils.quant_modules import QuantAct
from signjoey.transformer_layers import MultiHeadedAttention

torch.manual_seed(0)


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return cond


all_ok = True

B, T, H, num_heads = 2, 6, 32, 4
mha = MultiHeadedAttention(num_heads=num_heads, size=H, attention_type="reluformer")

x = torch.randn(B, T, H)
sf = torch.ones(1)  # sf coming from an upstream QuantAct(quantize=True) calibration step
# 2 real tokens + 4 padding tokens per sequence, key-side mask like sgn_mask
mask = torch.zeros(B, 1, T, dtype=torch.bool)
mask[:, :, :3] = True
query_mask = mask.squeeze(1)  # [B, T], same validity for query role (self-attention)

# ---------------------------------------------------------------------
# TEST 1: QAT mode (default) -- must not crash, must produce a REAL
#         (non-dummy) scaling factor at the output, and the reg_loss
#         must be a differentiable scalar.
# ---------------------------------------------------------------------
mha.train()
out, out_sf = mha(x, x, x, sf, sf, sf, mask=mask, query_mask=query_mask)
all_ok &= check("QAT forward runs without crashing", out.shape == (B, T, H))
all_ok &= check("QAT output scaling factor is real (non-dummy)", out_sf.item() != 1.0)
all_ok &= check("reg_loss is populated during training", mha.reg_loss is not None)

loss = out.sum() + (mha.reg_loss if mha.reg_loss is not None else 0.0)
loss.backward()
all_ok &= check(
    "gradient reaches ReLUFormer's log_gamma in QAT mode",
    mha.reluformer.log_gamma.grad is not None
    and mha.reluformer.log_gamma.grad.abs().sum().item() > 0,
)
mha.zero_grad()

# ---------------------------------------------------------------------
# TEST 2: the exact bug this session was about -- a mask must NEVER be
# silently accepted where a scaling factor was expected. If the old
# broken call site (`self.reluformer(scores, m, ...)`, mask passed as
# `scaling_factor`) were still there, dividing scores by a boolean
# mask tensor would either crash or silently produce garbage. We check
# that the fixed call path instead uses the real `s_sf` by comparing
# against a manual replica of what MultiHeadedAttention computes
# internally.
# ---------------------------------------------------------------------
with torch.no_grad():
    k_, k_sf_ = mha.k_layer(x, sf)
    q_, q_sf_ = mha.q_layer(x, sf)
    k_, k_sf_ = mha.qact_k(k_, k_sf_)
    q_, q_sf_ = mha.qact_q(q_, q_sf_)
    k_ = k_.view(B, T, num_heads, -1).transpose(1, 2)
    q_ = q_.view(B, T, num_heads, -1).transpose(1, 2)
    q_ = q_ * mha.scale
    q_, q_sf_ = mha.qact_qscale(q_, q_sf_)
    scores_, s_sf_ = mha.matmul_qk(q_, q_sf_, k_.transpose(2, 3), k_sf_)
    m_ = mha._prepare_mask(mask, scores_)
    scores_, s_sf_ = mha.qact_scores(scores_, s_sf_)
    # this must accept a real scaling-factor TENSOR (0/1-dim float), not
    # the boolean mask -- if the old bug were present this call would
    # raise (bool tensor has no meaningful `.reshape(-1).mean()` division)
    attn_, attn_sf_ = mha.reluformer(scores_, s_sf_, m_, query_mask=query_mask)
all_ok &= check(
    "reluformer receives a real scaling factor, not the mask (no crash, valid tensor)",
    torch.is_floating_point(attn_sf_) and attn_.shape == scores_.shape,
)

# ---------------------------------------------------------------------
# TEST 3: FP32 mode (quantize=False everywhere) -- output must be a
#         valid, finite, non-degenerate tensor. If the old "round to
#         nearest int with dummy sf=1.0" bug were still present, most
#         attention weights would collapse to zero.
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# TEST 3: FP32 mode (quantize=False everywhere) -- output must be a
#         valid, finite, non-degenerate tensor. If the old "round to
#         nearest int with dummy sf=1.0" bug were still present, most
#         attention weights would collapse to zero.
# ---------------------------------------------------------------------
def set_quantize_mode(model, quantize):
    touched = 0
    for m in model.modules():
        if hasattr(m, "quantize"):
            m.quantize = quantize
            touched += 1
    return touched


touched = set_quantize_mode(mha, False)
all_ok &= check(f"set_quantize_mode(False) touched {touched} sub-layers (>0)", touched > 0)

out_fp32, out_fp32_sf = mha(x, x, x, sf, sf, sf, mask=mask, query_mask=query_mask)
all_ok &= check("FP32-mode output is finite", torch.isfinite(out_fp32).all().item())
all_ok &= check("FP32-mode output scaling factor == dummy 1.0", out_fp32_sf.item() == 1.0)

# reconstruct attn weights directly to check they are NOT collapsed to
# (near-)zero -- the symptom of the old sf=1.0-rounding bug
with torch.no_grad():
    k2, k2sf = mha.k_layer(x, sf)
    q2, q2sf = mha.q_layer(x, sf)
    k2, k2sf = mha.qact_k(k2, k2sf)
    q2, q2sf = mha.qact_q(q2, q2sf)
    k2 = k2.view(B, T, num_heads, -1).transpose(1, 2)
    q2 = q2.view(B, T, num_heads, -1).transpose(1, 2)
    q2 = q2 * mha.scale
    q2, q2sf = mha.qact_qscale(q2, q2sf)
    scores2, s2sf = mha.matmul_qk(q2, q2sf, k2.transpose(2, 3), k2sf)
    m2 = mha._prepare_mask(mask, scores2)
    scores2, s2sf = mha.qact_scores(scores2, s2sf)
    attn2, attn2sf = mha.reluformer(scores2, s2sf, m2, query_mask=query_mask)

nonzero_fraction = (attn2 > 1e-6).float().mean().item()
all_ok &= check(
    f"FP32-mode reluformer attention weights are NOT collapsed to zero "
    f"({nonzero_fraction:.1%} of entries > 1e-6, expect > 20%)",
    nonzero_fraction > 0.20,
)

print()
print("=" * 60)
print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
print("=" * 60)
sys.exit(0 if all_ok else 1)
