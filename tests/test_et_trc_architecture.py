import torch
import pytest
from opentad.models.backbones.et_trc_videomae import (
    TaylorJacobianApproximator,
    TaylorResidualBlock,
    ETTRCVisionTransformerAdapter,
)


def test_taylor_jacobian_approximator():
    B, N, C = 2, 512, 384
    delta_h = torch.randn(B, N, C)
    approximator = TaylorJacobianApproximator(embed_dims=C, kernel_size=3)
    out = approximator(delta_h)
    assert out.shape == (B, N, C)
    assert not torch.isnan(out).any()


def test_taylor_residual_block_dense_identity():
    B, N, C = 2, 800, 384  # 8 tubelets, 10x10=100 spatial patches
    x = torch.randn(B, N, C, requires_grad=True)
    
    block = TaylorResidualBlock(
        embed_dims=C,
        num_heads=6,
        mlp_ratio=4.0,
        stride_k=4,
        enable_taylor=True,
    )
    
    out = block(x, h=10, w=10, num_frames=16)
    assert out.shape == (B, N, C), "State Multiplicity failed: shape must be exactly preserved"
    
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (B, N, C)
    assert not torch.isnan(x.grad).any()


def test_et_trc_videomae_forward_backward():
    # 2 chunks of 16 frames each = 32 frames, 160x160 resolution
    B, C_in, T, H, W = 2, 3, 16, 160, 160
    video = torch.randn(B, C_in, T, H, W)
    
    model = ETTRCVisionTransformerAdapter(
        img_size=160,
        patch_size=16,
        embed_dims=384,
        depth=2,  # 2 layers for fast testing
        num_heads=6,
        mlp_ratio=4.0,
        total_frames=32,
        num_frames=16,
        tubelet_size=2,
        stride_k=4,
        enable_taylor=True,
        adapter_index=[0, 1],
    )
    
    feats = model(video)
    # Expected output shape: (B, num_patches, embed_dims)
    # num_patches = (160/16) * (160/16) * (16/2) = 10 * 10 * 8 = 800
    assert feats.shape == (B, 800, 384)
    assert not torch.isnan(feats).any()
    
    loss = feats.sum()
    loss.backward()
    assert model.blocks[0].jacobian_approx.gain.grad is not None
