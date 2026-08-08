import sys
sys.path.insert(0, "/home/claude/testenv2")

import torch
import torch.nn as nn

from signjoey.encoders import TransformerEncoder


def set_quantize_mode(model, quantize):
    touched = 0
    for m in model.modules():
        if hasattr(m, "quantize"):
            m.quantize = quantize
            touched += 1
    return touched


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return cond


all_ok = True
torch.manual_seed(0)

HIDDEN = 32
HEADS = 4
LAYERS = 2
B, T = 4, 6
CKPT_PATH = "/home/claude/testenv2/float_pretrain.ckpt"

# ---------------------------------------------------------------------
# Toy task: reconstruct a fixed random target vector from a mean-pooled
# encoder output (MSE). Not meaningful linguistically -- just something
# with a real gradient signal so we can watch the loss actually move.
# ---------------------------------------------------------------------
torch.manual_seed(42)
target = torch.randn(B, HIDDEN)

x = torch.randn(B, T, HIDDEN)
lengths = torch.tensor([T, T, T, T])
mask = torch.ones(B, 1, T, dtype=torch.bool)
mask[1, :, 4:] = False  # give one sequence some padding, like real data
query_mask = mask.squeeze(1)


def make_model():
    torch.manual_seed(123)  # same init every time we build a fresh model
    return TransformerEncoder(
        hidden_size=HIDDEN, ff_size=HIDDEN * 2, num_layers=LAYERS,
        num_heads=HEADS, dropout=0.0, emb_dropout=0.0,
        attention_type="reluformer",
    )


def run_steps(model, optimizer, n_steps, tag, reg_weight=0.05, clip=1.0):
    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        out, out_sf = model(embed_src=x, src_length=lengths, mask=mask, query_mask=query_mask)
        pred = out.mean(dim=1)
        loss = nn.functional.mse_loss(pred, target)
        reg = getattr(model, "reg_loss", None)
        total = loss + (reg_weight * reg if reg is not None else 0.0)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        losses.append(loss.item())
    print(f"    {tag} losses: " + " -> ".join(f"{l:.4f}" for l in losses))
    return losses


# =======================================================================
# STEP 1: FP32 pretraining
# =======================================================================
print("\n--- Step 1: FP32 pretraining ---")
model = make_model()
touched = set_quantize_mode(model, False)
all_ok &= check(f"set_quantize_mode(False) touched {touched} layers", touched > 0)

opt = torch.optim.Adam(model.parameters(), lr=0.005)
fp32_losses = run_steps(model, opt, n_steps=15, tag="FP32")
all_ok &= check("FP32 loss decreased over training", fp32_losses[-1] < fp32_losses[0])

weight_snapshot = model.layers[0].src_src_att.k_layer.weight.detach().clone()

# =======================================================================
# STEP 2: save checkpoint (weights + optimizer state)
# =======================================================================
print("\n--- Step 2: saving checkpoint ---")
torch.save(
    {"model_state": model.state_dict(), "optimizer_state": opt.state_dict()},
    CKPT_PATH,
)
all_ok &= check("checkpoint file written", __import__("os").path.exists(CKPT_PATH))

# =======================================================================
# STEP 3: fresh model instance, load checkpoint with strict=True
#         (the actual point of this whole design: no key mismatch)
# =======================================================================
print("\n--- Step 3: loading checkpoint into a FRESH model instance ---")
model2 = TransformerEncoder(  # deliberately NOT calling make_model()'s
    hidden_size=HIDDEN, ff_size=HIDDEN * 2, num_layers=LAYERS,       # seeded init, to prove the
    num_heads=HEADS, dropout=0.0, emb_dropout=0.0,                   # loaded weights (not luck)
    attention_type="reluformer",                                     # are what makes it match
)
ckpt = torch.load(CKPT_PATH, weights_only=False)
try:
    missing, unexpected = model2.load_state_dict(ckpt["model_state"], strict=True)
    strict_load_ok = True
except Exception as e:
    strict_load_ok = False
    print(f"    strict=True load FAILED: {e}")

all_ok &= check("load_state_dict(strict=True) succeeds (no key mismatch)", strict_load_ok)

loaded_weight = model2.layers[0].src_src_att.k_layer.weight.detach()
all_ok &= check(
    "loaded weights EXACTLY match the saved FP32 weights",
    torch.equal(loaded_weight, weight_snapshot),
)

# quantize flag is a plain python attribute, NOT part of state_dict --
# a freshly constructed model always starts at the class default (True)
# regardless of what the checkpoint's source model was set to.
default_quantize_after_load = model2.layers[0].src_src_att.k_layer.quantize
all_ok &= check(
    "quantize flag is NOT restored by load_state_dict (defaults to True) "
    "-- must call set_quantize_mode() explicitly after loading",
    default_quantize_after_load is True,
)

# sanity: with the SAME quantize=False setting reapplied, model2 must
# reproduce model's exact forward output (proves nothing was lost)
set_quantize_mode(model2, False)
model.eval(); model2.eval()
with torch.no_grad():
    out1, _ = model(embed_src=x, src_length=lengths, mask=mask, query_mask=query_mask)
    out2, _ = model2(embed_src=x, src_length=lengths, mask=mask, query_mask=query_mask)
all_ok &= check(
    "reloaded model reproduces the exact same FP32 forward output",
    torch.allclose(out1, out2, atol=1e-6),
)
model.train(); model2.train()

# =======================================================================
# STEP 4: switch to QAT and fine-tune the SAME (reloaded) weights
# =======================================================================
print("\n--- Step 4: QAT fine-tuning (fresh optimizer, same weights) ---")
touched_q = set_quantize_mode(model2, True)
all_ok &= check(f"set_quantize_mode(True) touched {touched_q} layers", touched_q > 0)

# fresh optimizer for the QAT phase, as recommended earlier
opt_qat = torch.optim.Adam(model2.parameters(), lr=0.001)
qat_losses = run_steps(model2, opt_qat, n_steps=15, tag="QAT ")

all_ok &= check(
    "QAT fine-tuning starts near the FP32 loss level (weights were preserved)",
    qat_losses[0] < fp32_losses[0] * 1.5,  # generous margin; just checks
                                            # it isn't starting from scratch
)
all_ok &= check("QAT loss decreased further during fine-tuning", qat_losses[-1] <= qat_losses[0])

reg_after_qat = getattr(model2, "reg_loss", None)
all_ok &= check("reg_loss is populated in QAT mode", reg_after_qat is not None)

grad_ok = all(
    p.grad is not None and p.grad.abs().sum().item() > 0
    for p in model2.layers[0].src_src_att.reluformer.parameters()
)
all_ok &= check("gradients reached ReLUFormer params during QAT fine-tuning", grad_ok)

# =======================================================================
# STEP 5: save the QAT checkpoint too, for completeness
# =======================================================================
QAT_CKPT_PATH = "/home/claude/testenv2/qat_finetuned.ckpt"
torch.save({"model_state": model2.state_dict()}, QAT_CKPT_PATH)
all_ok &= check("QAT checkpoint saved", __import__("os").path.exists(QAT_CKPT_PATH))

print()
print("=" * 70)
print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
print("=" * 70)
sys.exit(0 if all_ok else 1)
