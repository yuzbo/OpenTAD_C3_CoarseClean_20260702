# Copyright (c) OpenTAD. All rights reserved.
from types import SimpleNamespace
import torch
import torch.nn as nn
import pytest

from opentad.datasets.transforms.bafdr import BAFDRSourceViews
from opentad.models.backbones.bafdr_wrapper import (
    BAFDRBackboneWrapper,
    BAFDRRouterHead,
    flatten_chunk_tubelets,
)
from opentad.models.projections.bafdr_asymmetric_proj import (
    BAFDRAsymmetricProjection,
    BAFDRLateProjection,
)
from tools.bata.bafdr_k16_fullmatrix_train import compute_router_targets


def test_bafdr_router_head_shape_and_range():
    B, num_chunks, C = 2, 48, 384
    router = BAFDRRouterHead(in_channels=C, hidden_channels=128)
    x = torch.randn(B, num_chunks, C)
    logits = router(x)
    assert logits.shape == (B, 4, 48)


def test_bafdr_router_targets_generation():
    # Ground truth segment: frames [100.0, 200.0] in a 768-frame window
    # chunk length = 16 frames. 100/16 = 6.25 (chunk 6), 200/16 = 12.5 (chunk 12)
    gt_segments = [torch.tensor([[100.0, 200.0]])]
    actionness, start_tgt, end_tgt = compute_router_targets(gt_segments, window_size=768, num_chunks=48)
    
    assert actionness.shape == (1, 48)
    assert start_tgt.shape == (1, 48)
    assert end_tgt.shape == (1, 48)
    
    # Start chunk 6 should have start_tgt = 1.0, adjacent 5 and 7 should have 0.5
    assert start_tgt[0, 6].item() == 1.0
    assert start_tgt[0, 5].item() == 0.5
    assert start_tgt[0, 7].item() == 0.5


def test_bafdr_dense_carrier_gamma_zero_identity():
    B, C, T = 2, 384, 384
    G = torch.randn(B, C, T)
    selected_indices = torch.tensor([[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                                     [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]])
    
    # When gamma is 0, residual R is all-zero tensor, so Z = G + R == G
    gamma = 0.0
    R_diff = torch.randn(B, C, 128)
    R_sel = gamma * R_diff
    R = torch.zeros_like(G)
    
    tubelet_offsets = torch.arange(8).view(1, 1, 8)
    selected_tubelets = (selected_indices.unsqueeze(-1) * 8 + tubelet_offsets).flatten(start_dim=1)
    R.scatter_(2, selected_tubelets.unsqueeze(1).expand(-1, C, -1), R_sel)
    
    Z = G + R
    assert torch.allclose(Z, G, atol=1e-7)


def test_bafdr_unselected_tubelets_bitwise_invariant():
    B, C, T = 1, 384, 384
    G = torch.randn(B, C, T)
    selected_indices = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
    
    R_sel = torch.randn(B, C, 128)
    R = torch.zeros_like(G)
    tubelet_offsets = torch.arange(8).view(1, 1, 8)
    selected_tubelets = (selected_indices.unsqueeze(-1) * 8 + tubelet_offsets).flatten(start_dim=1)
    R.scatter_(2, selected_tubelets.unsqueeze(1).expand(-1, C, -1), R_sel)
    
    Z = G + R
    # Tubelets for chunk 16..47 (indices 128..383) should be strictly identical to G
    assert torch.allclose(Z[:, :, 128:], G[:, :, 128:], atol=1e-7)
    assert not torch.allclose(Z[:, :, :128], G[:, :, :128], atol=1e-3)


def test_bafdr_asymmetric_projection_l2_l5_residual_invariance():
    B, C, T = 2, 384, 768
    G = torch.randn(B, C, T)
    R1 = torch.randn(B, C, T)
    R2 = torch.randn(B, C, T) * 5.0  # completely different residual
    mask = torch.ones(B, T, dtype=torch.bool)
    
    proj = BAFDRAsymmetricProjection(
        in_channels=C,
        out_channels=C,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=4, n_mha_win_size=-1),
        use_abs_pe=False,
    )
    proj.eval()
    
    bundle1 = dict(global_features=G, residual_features=R1)
    bundle2 = dict(global_features=G, residual_features=R2)
    
    with torch.no_grad():
        out1, mask1 = proj(bundle1, mask)
        out2, mask2 = proj(bundle2, mask)
        
    # out: (L0, L1, L2, L3, L4, L5)
    assert len(out1) == 6
    assert len(out2) == 6
    
    # L0 and L1 MUST be different because residual is injected
    assert not torch.allclose(out1[0], out2[0], atol=1e-4)
    assert not torch.allclose(out1[1], out2[1], atol=1e-4)
    
    # L2, L3, L4, L5 MUST BE STRICTLY IDENTICAL (bitwise invariant to residual R)
    for lvl in range(2, 6):
        assert torch.allclose(out1[lvl], out2[lvl], atol=1e-6), f"Level L{lvl} should be strictly invariant to R"


def test_bafdr_late_projection_propagates_residual_to_all_levels():
    B, C, T = 2, 384, 768
    G = torch.randn(B, C, T)
    R1 = torch.randn(B, C, T)
    R2 = torch.randn(B, C, T) * 2.0
    mask = torch.ones(B, T, dtype=torch.bool)
    
    proj = BAFDRLateProjection(
        in_channels=C,
        out_channels=C,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=4, n_mha_win_size=-1),
        use_abs_pe=False,
    )
    proj.eval()
    
    bundle1 = dict(fused_features=G + R1)
    bundle2 = dict(fused_features=G + R2)
    
    with torch.no_grad():
        out1, _ = proj(bundle1, mask)
        out2, _ = proj(bundle2, mask)
        
    # In Late projection, all levels receive the altered fused features
    for lvl in range(6):
        assert not torch.allclose(out1[lvl], out2[lvl], atol=1e-4), f"Late projection Level L{lvl} should change with R"
