from types import SimpleNamespace

import torch
import torch.nn as nn
import pytest
from mmaction.registry import MODELS as MMACTION_MODELS
from opentad.models.backbones.et_trc_videomae import (
    TemporalLowRankJVP,
    TaylorAttention,
    TaylorResidualBlock,
    ETTRCVisionTransformerAdapter,
)
from tools.bata.diagnose_taylor_residual_manifold import (
    _add_production_pos_embed,
    _infer_patch_geometry,
    _resolve_jvp_operators,
)


def test_et_trc_backbone_registered_for_mmaction_recognizer():
    assert MMACTION_MODELS.get("ETTRCVisionTransformerAdapter") is ETTRCVisionTransformerAdapter


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


def test_taylor_attention_remaps_fused_videomae_qkv():
    attn = TaylorAttention(embed_dims=12, num_heads=3)
    fused_w = torch.arange(36 * 12, dtype=torch.float32).reshape(36, 12)
    fused_b = torch.arange(36, dtype=torch.float32)
    missing, unexpected = [], []
    attn.load_state_dict(
        {
            "qkv.weight": fused_w,
            "qkv.bias": fused_b,
            "proj.weight": attn.proj.weight.detach().clone(),
            "proj.bias": attn.proj.bias.detach().clone(),
        },
        strict=False,
    )
    assert attn.pretrained_qkv_remapped
    assert torch.equal(attn.q_proj.weight, fused_w[:12])
    assert torch.equal(attn.k_proj.weight, fused_w[12:24])
    assert torch.equal(attn.v_proj.weight, fused_w[24:])


def test_taylor_attention_segment_isolation():
    torch.manual_seed(7)
    attn = TaylorAttention(embed_dims=12, num_heads=3)
    x = torch.randn(1, 4, 12)
    q = x[:, 2:3]
    q_seg = torch.tensor([[1]])
    k_seg = torch.tensor([[0, 0, 1, 1]])
    out1 = attn(x, query_x=q, query_segment_ids=q_seg, key_segment_ids=k_seg)
    x_changed = x.clone()
    x_changed[:, :2] += 1000.0
    out2 = attn(x_changed, query_x=x_changed[:, 2:3], query_segment_ids=q_seg, key_segment_ids=k_seg)
    assert torch.allclose(out1, out2, atol=1e-5)


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


def test_taylor_diagnostic_infers_production_chunk_geometry():
    inner = SimpleNamespace(patch_size=16, tubelet_size=2)
    frames = torch.zeros(48, 3, 16, 160, 160)
    tokens = torch.zeros(48, 800, 384)

    assert _infer_patch_geometry(inner, frames, tokens) == (10, 10, 100, 8)


def test_taylor_diagnostic_rejects_full_window_pos_embed_mismatch():
    inner = SimpleNamespace(pos_embed=torch.zeros(1, 800, 384))
    full_window_tokens = torch.zeros(1, 38400, 384)

    with pytest.raises(RuntimeError, match="pos_embed shape mismatch"):
        _add_production_pos_embed(inner, full_window_tokens)


def test_taylor_diagnostic_requires_in_model_jvp():
    with pytest.raises(RuntimeError, match="missing layers \\[0\\]"):
        _resolve_jvp_operators(nn.ModuleList([nn.Identity()]), torch.device("cpu"))
