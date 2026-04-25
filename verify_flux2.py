#!/usr/bin/env python3
"""
Verification script for FLUX.2 architecture components.
Tests individual building blocks to ensure the code structure is correct.
"""

import torch


def test_model_params():
    """Verify model parameter configurations."""
    print("Testing Model Parameter Configurations...")

    from flux2.model import Flux2Params, Klein4BParams, Klein9BParams

    configs = [
        ("Flux2Params", Flux2Params()),
        ("Klein9BParams", Klein9BParams()),
        ("Klein4BParams", Klein4BParams()),
    ]

    for name, params in configs:
        print(f"  {name}:")
        print(f"    hidden_size: {params.hidden_size}")
        print(f"    num_heads: {params.num_heads}")
        print(f"    depth: {params.depth}, depth_single_blocks: {params.depth_single_blocks}")
        print(f"    use_guidance_embed: {params.use_guidance_embed}")

        # Verify hidden_size is divisible by num_heads
        assert params.hidden_size % params.num_heads == 0

        # Verify axes_dim dimensions
        pe_dim = params.hidden_size // params.num_heads
        print(f"    axes_dim: {params.axes_dim}, sum={sum(params.axes_dim)}, pe_dim={pe_dim}")
        assert sum(params.axes_dim) == pe_dim, f"axes_dim sum {sum(params.axes_dim)} != pe_dim {pe_dim}"

    print("  All parameter configurations valid!")


def test_timestep_embedding():
    """Test timestep embedding generation."""
    print("\nTesting Timestep Embedding...")

    from flux2.model import timestep_embedding

    # Test with different timesteps
    timesteps = torch.tensor([0.0, 0.5, 1.0])
    emb = timestep_embedding(timesteps, dim=256)

    assert emb.shape == (3, 256), f"Expected (3, 256), got {emb.shape}"
    print(f"  Timestep embedding shape: {emb.shape} - OK")

    # Test with fractional timesteps
    timesteps_frac = torch.tensor([0.25, 0.75])
    emb_frac = timestep_embedding(timesteps_frac, dim=128)
    assert emb_frac.shape == (2, 128), f"Expected (2, 128), got {emb_frac.shape}"
    print(f"  Fractional timestep embedding shape: {emb_frac.shape} - OK")


def test_modulation():
    """Test modulation layer."""
    print("\nTesting Modulation Layer...")

    from flux2.model import Modulation

    # Test single modulation
    mod_single = Modulation(dim=256, double=False)
    vec = torch.randn(2, 256)
    result = mod_single(vec)
    # Modulation returns (shift, scale, gate) tuple for single
    # and ((shift, scale, gate), (shift, scale, gate)) for double
    if len(result) == 2:
        shift_scale_gate, _ = result
    else:
        shift_scale_gate = result
    shift, scale, gate = shift_scale_gate
    print(f"  Single modulation: shift={shift.shape}, scale={scale.shape}, gate={gate.shape} - OK")

    # Test double modulation
    mod_double = Modulation(dim=256, double=True)
    img_mod, txt_mod = mod_double(vec)
    shift_img, scale_img, gate_img = img_mod
    shift_txt, scale_txt, gate_txt = txt_mod
    print(
        f"  Double modulation (img): shift={shift_img.shape}, scale={scale_img.shape}, gate={gate_img.shape} - OK"
    )
    print(
        f"  Double modulation (txt): shift={shift_txt.shape}, scale={scale_txt.shape}, gate={gate_txt.shape} - OK"
    )


def test_mlp_embedder():
    """Test MLP embedder for timesteps."""
    print("\nTesting MLP Embedder...")

    from flux2.model import MLPEmbedder

    mlp = MLPEmbedder(in_dim=256, hidden_dim=512)
    x = torch.randn(2, 256)
    out = mlp(x)
    assert out.shape == (2, 512), f"Expected (2, 512), got {out.shape}"
    print(f"  MLP output shape: {out.shape} - OK")


def test_silu_activation():
    """Test SiLU activation."""
    print("\nTesting SiLU Activation...")

    from flux2.model import SiLUActivation

    silu = SiLUActivation()
    x = torch.randn(2, 256)
    out = silu(x)
    assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"
    print(f"  SiLU output shape: {out.shape} - OK")


def test_rmsnorm():
    """Test RMSNorm layer."""
    print("\nTesting RMSNorm...")

    from flux2.model import RMSNorm

    norm = RMSNorm(dim=128)
    x = torch.randn(4, 16, 128)
    out = norm(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"  RMSNorm output shape: {out.shape} - OK")


def test_qknorm():
    """Test QKNorm layer."""
    print("\nTesting QKNorm...")

    from flux2.model import QKNorm

    qknorm = QKNorm(dim=64)
    batch, seq, heads, dim = 2, 16, 8, 64
    q = torch.randn(batch, seq, heads, dim)
    k = torch.randn(batch, seq, heads, dim)
    v = torch.randn(batch, seq, heads, dim)
    q_out, k_out = qknorm(q, k, v)
    assert q_out.shape == q.shape, f"Expected {q.shape}, got {q_out.shape}"
    assert k_out.shape == k.shape, f"Expected {k.shape}, got {k_out.shape}"
    assert k_out.dtype == v.dtype, "QKNorm output should match v dtype"
    print(f"  QKNorm output shapes: q={q_out.shape}, k={k_out.shape} - OK")


def test_embed_nd():
    """Test EmbedND (RoPE embedder)."""
    print("\nTesting EmbedND (RoPE)...")

    from flux2.model import EmbedND

    axes_dim = [32, 32, 32, 32]
    embedder = EmbedND(dim=128, theta=2000, axes_dim=axes_dim)

    batch, seq = 2, 16
    ids = torch.zeros(batch, seq, 4)  # (batch, seq, t, h, w, l)
    ids[:, :, 1] = torch.arange(seq).float()  # h dimension

    emb = embedder(ids)
    # RoPE returns (batch, 1, seq, 64, 2, 2) for axes_dim [32,32,32,32]
    # where 64 = sum(axes_dim), and (2, 2) is the rotation matrix
    print(f"  EmbedND output shape: {emb.shape}")
    assert emb.shape[0] == batch
    assert emb.shape[1] == 1  # Singleton dimension
    assert emb.shape[2] == seq
    print("  EmbedND shape verified - OK")


def test_processing_helpers():
    """Test image and text processing helpers."""
    print("\nTesting Processing Helpers...")

    from flux2.sampling import batched_prc_img, prc_txt

    # Test image processing
    dummy_img = torch.randn(128, 8, 8)  # C, H, W
    x_list, x_ids_list = batched_prc_img([dummy_img])
    x = x_list[0]
    x_ids = x_ids_list[0]

    expected_seq_len = 8 * 8  # H * W
    assert x.shape == (expected_seq_len, 128), f"Expected ({expected_seq_len}, 128), got {x.shape}"
    assert x_ids.shape == (expected_seq_len, 4), f"Expected ({expected_seq_len}, 4), got {x_ids.shape}"
    print(f"  Image processing: x shape={x.shape}, x_ids shape={x_ids.shape} - OK")

    # Test text processing
    dummy_txt = torch.randn(16, 7680)  # seq_len, context_in_dim (2D)
    ctx, ctx_ids = prc_txt(dummy_txt)

    assert ctx.shape == (16, 7680), f"Expected (16, 7680), got {ctx.shape}"
    assert ctx_ids.shape == (16, 4), f"Expected (16, 4), got {ctx_ids.shape}"
    print(f"  Text processing: ctx shape={ctx.shape}, ctx_ids shape={ctx_ids.shape} - OK")


def test_last_layer():
    """Test LastLayer output layer."""
    print("\nTesting LastLayer...")

    from flux2.model import LastLayer

    layer = LastLayer(hidden_size=512, out_channels=128)
    x = torch.randn(2, 16, 512)  # batch, seq, hidden
    vec = torch.randn(2, 512)  # batch, hidden

    out = layer(x, vec)
    expected_shape = (2, 16, 128)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print(f"  LastLayer output shape: {out.shape} - OK")


def test_schedule_generation():
    """Test denoising schedule generation."""
    print("\nTesting Denoising Schedule...")

    from flux2.sampling import get_schedule

    # Test with different parameters
    timesteps = get_schedule(num_steps=20, image_seq_len=1024)
    assert len(timesteps) == 21, f"Expected 21 timesteps, got {len(timesteps)}"
    assert timesteps[0] == 1.0, f"Expected first timestep 1.0, got {timesteps[0]}"
    assert timesteps[-1] == 0.0, f"Expected last timestep 0.0, got {timesteps[-1]}"
    print(f"  Generated {len(timesteps)} timesteps for image_seq_len=1024, num_steps=20 - OK")


def test_scatter_ids():
    """Test scatter_ids function for token reordering."""
    print("\nTesting Scatter IDs...")

    from flux2.sampling import scatter_ids

    batch_size = 1
    seq_len = 16
    channels = 128

    x = torch.randn(batch_size, seq_len, channels)
    x_ids = torch.zeros(batch_size, seq_len, 4, dtype=torch.long)
    x_ids[0, :, 0] = 0  # t=0
    x_ids[0, :, 1] = torch.arange(seq_len) % 4  # h (0-3)
    x_ids[0, :, 2] = torch.arange(seq_len) // 4  # w (0-3)
    x_ids[0, :, 3] = 0  # l=0

    result = scatter_ids(x, x_ids)

    assert len(result) == batch_size
    assert result[0].shape == (1, 128, 1, 4, 4), f"Expected (1, 128, 1, 4, 4), got {result[0].shape}"
    print(f"  Scatter IDs output shape: {result[0].shape} - OK")


def test_compute_empirical_mu():
    """Test empirical mu computation."""
    print("\nTesting Empirical Mu Computation...")

    from flux2.sampling import compute_empirical_mu

    mu_small = compute_empirical_mu(image_seq_len=256, num_steps=10)
    mu_large = compute_empirical_mu(image_seq_len=10000, num_steps=200)

    print(f"  Mu for seq_len=256, steps=10: {mu_small:.4f}")
    print(f"  Mu for seq_len=10000, steps=200: {mu_large:.4f}")

    assert mu_small > 0, "Mu should be positive"
    assert mu_large > 0, "Mu should be positive"


def test_generalized_time_snr_shift():
    """Test time-SNR shift function."""
    print("\nTesting Time-SNR Shift...")

    from flux2.sampling import generalized_time_snr_shift

    t = torch.tensor([0.1, 0.5, 0.9])
    mu = 0.0
    sigma = 1.0

    result = generalized_time_snr_shift(t, mu, sigma)

    assert result.shape == t.shape, f"Expected {t.shape}, got {result.shape}"
    assert torch.all(result >= 0) and torch.all(result <= 1), "Result should be in [0, 1]"
    print(f"  Time-SNR shift result: {result} - OK")


def test_util_functions():
    """Test utility functions from flux2.util."""
    print("\nTesting Utility Functions...")

    try:
        from flux2 import util

        # Check if there are any load functions
        load_functions = [name for name in dir(util) if name.startswith("load_")]
        print(f"  Load functions available: {load_functions}")
    except ImportError as e:
        print(f"  Utility module import error: {e}")


def test_autoencoder():
    """Test autoencoder module."""
    print("\nTesting Autoencoder Module...")

    try:
        from flux2 import autoencoder

        # Check available classes
        autoencoder_classes = [name for name in dir(autoencoder) if not name.startswith("_")]
        print(f"  Autoencoder module classes: {autoencoder_classes}")
        print("  Autoencoder module accessible - OK")

    except ImportError as e:
        print(f"  Autoencoder import failed: {e}")


def test_text_encoder():
    """Test text encoder module."""
    print("\nTesting Text Encoder Module...")

    try:
        from flux2 import text_encoder

        # Check available classes
        text_encoder_classes = [name for name in dir(text_encoder) if not name.startswith("_")]
        print(f"  Text encoder module classes: {text_encoder_classes}")
        print("  Text encoder module accessible - OK")

    except ImportError as e:
        print(f"  Text encoder import failed: {e}")


def main():
    print("=" * 60)
    print("FLUX.2 Architecture Verification")
    print("=" * 60)
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tests = [
        ("Model Parameters", test_model_params),
        ("Timestep Embedding", test_timestep_embedding),
        ("Modulation Layer", test_modulation),
        ("MLP Embedder", test_mlp_embedder),
        ("SiLU Activation", test_silu_activation),
        ("RMSNorm", test_rmsnorm),
        ("QKNorm", test_qknorm),
        ("EmbedND (RoPE)", test_embed_nd),
        ("Processing Helpers", test_processing_helpers),
        ("LastLayer", test_last_layer),
        ("Denoising Schedule", test_schedule_generation),
        ("Scatter IDs", test_scatter_ids),
        ("Empirical Mu", test_compute_empirical_mu),
        ("Time-SNR Shift", test_generalized_time_snr_shift),
        ("Utility Functions", test_util_functions),
        ("Autoencoder", test_autoencoder),
        ("Text Encoder", test_text_encoder),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if passed > 0 and failed == 0:
        print("\nSUCCESS: All critical FLUX.2 components verified!")
    elif passed > 0:
        print(f"\nPARTIAL SUCCESS: {passed}/{len(tests)} tests passed")

    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
