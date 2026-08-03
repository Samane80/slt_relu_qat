#!/usr/bin/env python3
"""
تست ساده: FP32 → QAT Workflow
فقط forward/backward را تست می‌کند، بدون نیاز به دیتاست واقعی
"""
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from signjoey.helpers import load_config
from signjoey.model import build_model
from signjoey.vocabulary import TextVocabulary, GlossVocabulary
from quantization_utils.model_utils import set_quantize_mode

def create_dummy_vocab():
    """ساخت dummy vocabulary"""
    gls_vocab = GlossVocabulary()
    gls_vocab.stoi = {"<pad>": 0, "<s>": 1, "</s>": 2, "WORD1": 3, "WORD2": 4, "WORD3": 5}
    gls_vocab.itos = {v: k for k, v in gls_vocab.stoi.items()}
    
    txt_vocab = TextVocabulary()
    txt_vocab.stoi = {"<pad>": 0, "<s>": 1, "</s>": 2, "word1": 3, "word2": 4, "word3": 5}
    txt_vocab.itos = {v: k for k, v in txt_vocab.stoi.items()}
    
    return gls_vocab, txt_vocab

def test_forward_backward(model, mode_name):
    """تست forward و backward"""
    print(f"\n🔍 Testing {mode_name} forward/backward...")
    
    # Create dummy batch
    batch_size = 2
    seq_len = 10
    txt_len = 5
    
    sgn = torch.randn(batch_size, seq_len, 408)  # feature_size = 408
    sgn_mask = torch.ones(batch_size, 1, seq_len)
    sgn_lengths = torch.tensor([seq_len, seq_len])
    txt_input = torch.randint(0, 6, (batch_size, txt_len))
    txt = torch.randint(0, 6, (batch_size, txt_len))
    txt_mask = torch.ones(batch_size, 1, txt_len)
    
    # Forward
    model.train()
    decoder_outputs, gloss_probabilities = model(
        sgn=sgn,
        sgn_mask=sgn_mask,
        sgn_lengths=sgn_lengths,
        txt_input=txt_input,
        txt_mask=txt_mask,
    )
    
    # Compute loss
    word_outputs = decoder_outputs[0]
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(word_outputs.view(-1, word_outputs.size(-1)), txt.view(-1))
    
    # Backward
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  ✅ Forward pass: {word_outputs.shape}")
    print(f"  ✅ Loss: {loss.item():.4f}")
    print(f"  ✅ Backward pass: gradients computed")
    
    return loss.item()

def main():
    print("\n" + "=" * 70)
    print("🧪 Testing FP32 → QAT Workflow")
    print("=" * 70)
    
    # Load config
    print("\n[1/8] Loading configs...")
    fp32_cfg = load_config("configs/sign_fp32.yaml")
    qat_cfg = load_config("configs/sign_qat.yaml")
    
    # Create vocabs
    print("[2/8] Creating vocabularies...")
    gls_vocab, txt_vocab = create_dummy_vocab()
    
    # ===== PHASE 1: FP32 =====
    print("\n" + "=" * 70)
    print("🔵 PHASE 1: FP32 Mode (quantize=False)")
    print("=" * 70)
    
    print("\n[3/8] Building FP32 model...")
    model_fp32 = build_model(
        cfg=fp32_cfg["model"],
        sgn_dim=408,  # feature_size
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
    )
    
    # Check quantize flags
    fp32_count = sum(1 for m in model_fp32.modules() if hasattr(m, "quantize") and not m.quantize)
    qat_count = sum(1 for m in model_fp32.modules() if hasattr(m, "quantize") and m.quantize)
    print(f"  ✅ FP32 layers: {fp32_count}")
    print(f"  ✅ QAT layers: {qat_count} (should be 0)")
    
    if qat_count > 0:
        print("  ❌ ERROR: Some layers are in QAT mode!")
        return
    
    # Test forward/backward
    loss_fp32 = test_forward_backward(model_fp32, "FP32")
    
    # Save checkpoint
    print("\n[4/8] Saving FP32 checkpoint...")
    checkpoint_path = "test_fp32_checkpoint.pt"
    torch.save({
        'model_state': model_fp32.state_dict(),
        'step': 0,
    }, checkpoint_path)
    print(f"  ✅ Saved to {checkpoint_path}")
    
    # Save weights for comparison
    weights_before = copy.deepcopy(model_fp32.state_dict())
    
    # ===== PHASE 2: QAT =====
    print("\n" + "=" * 70)
    print("🟢 PHASE 2: QAT Mode (quantize=True)")
    print("=" * 70)
    
    print("\n[5/8] Loading FP32 checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    
    print("\n[6/8] Building QAT model...")
    model_qat = build_model(
        cfg=qat_cfg["model"],
        sgn_dim=408,  # feature_size
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
    )
    
    # Check quantize flags
    fp32_count = sum(1 for m in model_qat.modules() if hasattr(m, "quantize") and not m.quantize)
    qat_count = sum(1 for m in model_qat.modules() if hasattr(m, "quantize") and m.quantize)
    print(f"  ✅ FP32 layers: {fp32_count} (should be 0)")
    print(f"  ✅ QAT layers: {qat_count}")
    
    if fp32_count > 0:
        print("  ❌ ERROR: Some layers are still in FP32 mode!")
        return
    
    # Load weights
    print("\n[7/8] Loading FP32 weights into QAT model...")
    missing, unexpected = model_qat.load_state_dict(checkpoint['model_state'], strict=False)
    
    print(f"  ✅ Matched keys: {len(checkpoint['model_state']) - len(missing)}")
    print(f"  ⚠️  Missing keys: {len(missing)} (QAT-specific buffers)")
    print(f"  ❌ Unexpected keys: {len(unexpected)} (should be 0)")
    
    if len(unexpected) > 0:
        print("  ❌ ERROR: Unexpected keys found!")
        print(f"  Unexpected: {unexpected[:5]}")
        return
    
    # Compare weights
    print("\n  🔍 Comparing weights...")
    weights_match = True
    for key in weights_before:
        if key in model_qat.state_dict():
            if not torch.equal(weights_before[key], model_qat.state_dict()[key]):
                print(f"  ❌ Mismatch: {key}")
                weights_match = False
    
    if weights_match:
        print("  ✅ All weights match perfectly!")
    else:
        print("  ❌ ERROR: Weights don't match!")
        return
    
    # Test forward/backward
    loss_qat = test_forward_backward(model_qat, "QAT")
    
    # Save QAT checkpoint
    print("\n[8/8] Saving QAT checkpoint...")
    qat_checkpoint_path = "test_qat_checkpoint.pt"
    torch.save({
        'model_state': model_qat.state_dict(),
        'step': 0,
    }, qat_checkpoint_path)
    print(f"  ✅ Saved to {qat_checkpoint_path}")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\n📋 Summary:")
    print(f"  • FP32 loss: {loss_fp32:.4f}")
    print(f"  • QAT loss: {loss_qat:.4f}")
    print(f"  • Checkpoints: {checkpoint_path}, {qat_checkpoint_path}")
    print("\n🎯 Ready for full training!")
    print("\nNext steps:")
    print("  1. FP32 training: python -m signjoey training configs/sign_fp32.yaml")
    print("  2. QAT fine-tune: python -m signjoey training configs/sign_qat.yaml")

if __name__ == "__main__":
    main()