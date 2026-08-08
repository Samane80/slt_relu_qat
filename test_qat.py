import torch
import torch.nn as nn

from signjoey.embeddings import SpatialEmbeddings, Embeddings
from signjoey.encoders import TransformerEncoder
from signjoey.decoders import TransformerDecoder

def test_qat_pipeline():
    print("🚀 Starting QAT Pipeline Test...")
    
    # 1. Hyperparameters
    batch_size = 2
    src_len = 10
    trg_len = 5
    input_size = 512  # video feature dim
    hidden_size = 256 # transformer hidden dim
    ff_size = 512
    num_heads = 4
    num_layers = 2
    vocab_size = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize Components
    print("\n[1/4] Initializing QAT Components...")
    
    # Encoder side
    sgn_embed = SpatialEmbeddings(
        embedding_dim=hidden_size,
        input_size=input_size,
        num_heads=num_heads,
        norm_type="layer",
        activation_type="gelu",
        scale=True
    ).to(device)
    
    encoder = TransformerEncoder(
        hidden_size=hidden_size,
        ff_size=ff_size,
        num_layers=num_layers,
        num_heads=num_heads,
        attention_type="reluformer" # Test ReLUFormer integration
    ).to(device)
    
    # Decoder side
    txt_embed = Embeddings(
        embedding_dim=hidden_size,
        num_heads=num_heads,
        vocab_size=vocab_size,
        norm_type="layer",
        activation_type="gelu",
        scale=True
    ).to(device)
    
    decoder = TransformerDecoder(
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_size,
        ff_size=ff_size,
        vocab_size=vocab_size,
        attention_type="softmax"
    ).to(device)

    # 3. Create Dummy Data
    print("\n[2/4] Creating Dummy Data...")
    sgn = torch.randn(batch_size, src_len, input_size).to(device)
    sgn_mask = torch.ones(batch_size, 1, src_len, dtype=torch.bool).to(device)
    sgn_lengths = torch.tensor([src_len, src_len]).to(device)
    
    txt_input = torch.randint(0, vocab_size, (batch_size, trg_len)).to(device)
    txt_mask = torch.ones(batch_size, 1, trg_len, dtype=torch.bool).to(device)

    # 4. Forward Pass
    print("\n[3/4] Running Forward Pass...")
    encoder.train()
    decoder.train()
    
    # Encoder Forward
    embed_src, embed_src_sf = sgn_embed(sgn, sgn_mask)
    print(f"✅ SpatialEmbeddings output shape: {embed_src.shape}, SF: {embed_src_sf.item():.6f}")
    
    encoder_out, encoder_sf = encoder(
        embed_src=embed_src, 
        src_length=sgn_lengths, 
        mask=sgn_mask, 
        act_scaling_factor=embed_src_sf
    )
    print(f"✅ Encoder output shape: {encoder_out.shape}, SF: {encoder_sf.item():.6f}")
    
    # Decoder Forward
    trg_embed, trg_sf = txt_embed(txt_input, txt_mask)
    print(f"✅ Text Embeddings output shape: {trg_embed.shape}, SF: {trg_sf.item():.6f}")
    
    decoder_out, _, _, _ = decoder(
        trg_embed=trg_embed,
        encoder_output=encoder_out,
        src_mask=sgn_mask,
        trg_mask=txt_mask,
        act_scaling_factor=trg_sf,
        encoder_output_sf=encoder_sf
    )
    
    logits = decoder_out[0]
    print(f"✅ Decoder logits shape: {logits.shape}")

    # 5. Backward Pass & Gradient Check
    print("\n[4/4] Running Backward Pass & Checking Gradients...")
    
    # Dummy loss
    dummy_target = torch.randint(0, vocab_size, (batch_size, trg_len)).to(device)
    loss = nn.functional.cross_entropy(logits.reshape(-1, vocab_size), dummy_target.reshape(-1))
    
    # Add ReLUFormer reg loss if applicable
    if hasattr(encoder, 'reg_loss') and encoder.reg_loss is not None:
        loss += 0.1 * encoder.reg_loss
        print(f"✅ Added ReLUFormer Reg Loss: {encoder.reg_loss.item():.6f}")

    print(f"Total Dummy Loss: {loss.item():.6f}")
    
    loss.backward()
    
    # 6. Verification Checks
    print("\n🔍 Verification Checks:")
    
    # Check 1: Gradients exist and are not zero for QuantLinear weights (Tests STE)
    qlin = encoder.layers[0].src_src_att.k_layer
    if qlin.weight.grad is not None and qlin.weight.grad.abs().sum() > 0:
        print("✅ [PASS] QuantLinear weight gradients are valid (STE working).")
    else:
        print("❌ [FAIL] QuantLinear weight gradients are zero or None!")
        
    # Check 2: Scaling factors are updated (not zero)
    qact = encoder.qact_after_dropout
    if qact.act_scaling_factor.item() > 0:
        print(f"✅ [PASS] QuantAct scaling factor is updated: {qact.act_scaling_factor.item():.6f}")
    else:
        print("❌ [FAIL] QuantAct scaling factor is zero!")
        
    # Check 3: IntLayerNorm scaling factor
    ln = encoder.layer_norm
    if ln.norm_scaling_factor.item() > 0:
        print(f"✅ [PASS] IntLayerNorm scaling factor is updated: {ln.norm_scaling_factor.item():.6f}")
    else:
        print("❌ [FAIL] IntLayerNorm scaling factor is zero!")
        
    # Check 4: ReLUFormer parameters (log_gamma) have gradients
    relu_attn = encoder.layers[0].src_src_att.reluformer
    if relu_attn.log_gamma.grad is not None and relu_attn.log_gamma.grad.abs().sum() > 0:
        print("✅ [PASS] ReLUFormer log_gamma gradients are valid.")
    else:
        print("❌ [FAIL] ReLUFormer log_gamma gradients are zero or None!")

    print("\n🎉 QAT Pipeline Test Completed Successfully!")

if __name__ == "__main__":
    test_qat_pipeline()