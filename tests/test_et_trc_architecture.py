import torch
import torch.nn as nn
import pytest
from opentad.models.backbones.et_trc_videomae import (
    TemporalLowRankJVP,
    TaylorResidualBlock,
    ETTRCVisionTransformerAdapter,
)


def test_temporal_low_rank_jvp_linearity():
    B, T, S, C = 2, 8, 100, 384
    jvp = TemporalLowRankJVP(embed_dims=C, rank=64, kernel_size=3)
    
    # 1. Zero input -> Zero output: J(0) = 0
    zeros = torch.zeros(B, T, S, C)
    out_zero = jvp(zeros)
    assert torch.allclose(out_zero, zeros, atol=1e-6)
    
    # 2. Additivity: J(a + b) = J(a) + J(b)
    a = torch.randn(B, T, S, C)
    b = torch.randn(B, T, S, C)
    out_ab = jvp(a + b)
    out_a_plus_b = jvp(a) + jvp(b)
    assert torch.allclose(out_ab, out_a_plus_b, atol=1e-5)
    
    # 3. Homogeneity: J(c * a) = c * J(a)
    c = 2.5
    out_ca = jvp(c * a)
    out_c_times_a = c * jvp(a)
    assert torch.allclose(out_ca, out_c_times_a, atol=1e-5)


def test_taylor_residual_block_dense_parity():
    B, N, C = 2, 800, 384
    x = torch.randn(B, N, C, requires_grad=True)
    
    # When enable_taylor=False, block executes exact dense formulation
    block = TaylorResidualBlock(
        embed_dims=C,
        num_heads=6,
        mlp_ratio=4.0,
        stride_k=4,
        enable_taylor=False,
    )
    out = block(x, h=10, w=10, num_frames=16)
    assert out.shape == (B, N, C)
    assert not torch.isnan(out).any()


def test_et_trc_videomae_return_feat_map():
    # 2 chunks of 16 frames each = 16 frames per clip, 160x160 resolution
    B, C_in, T, H, W = 2, 3, 16, 160, 160
    video = torch.randn(B, C_in, T, H, W)
    
    model = ETTRCVisionTransformerAdapter(
        img_size=160,
        patch_size=16,
        embed_dims=384,
        depth=2,
        num_heads=6,
        mlp_ratio=4.0,
        total_frames=32,
        num_frames=16,
        tubelet_size=2,
        stride_k=4,
        enable_taylor=True,
        return_feat_map=True,
    )
    
    feats = model(video)
    # Expected feature map shape: [B, C, T_tubelets, H_patches, W_patches]
    # T_tubelets = 16 // 2 = 8, H_patches = 160 // 16 = 10, W_patches = 160 // 16 = 10
    assert feats.shape == (B, 384, 8, 10, 10), f"Expected (B, 384, 8, 10, 10) but got {feats.shape}"
    assert not torch.isnan(feats).any()
