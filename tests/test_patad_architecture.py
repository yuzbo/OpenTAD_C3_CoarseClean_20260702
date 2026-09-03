from __future__ import annotations

import torch
import pytest
from opentad.models.projections.pyramid_aware_asymmetric_proj import PyramidAwareAsymmetricProj
from opentad.models.projections.actionformer_proj import Conv1DTransformerProj


def test_patad_projection_shape_and_forward():
    conv_cfg = dict(kernel_size=3, proj_pdrop=0.0)
    attn_cfg = dict(n_head=4, n_mha_win_size=-1, attn_pdrop=0.0)
    norm_cfg = dict(type="LN")

    patad_proj = PyramidAwareAsymmetricProj(
        in_channels=16,
        out_channels=16,
        arch=(1, 1, 5),
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        attn_cfg=attn_cfg,
        max_seq_len=64,
        asymmetric_split_level=2,
    )

    B, C, T = 1, 16, 64
    g_feat = torch.randn(B, C, T, requires_grad=True)
    r_feat = torch.randn(B, C, T, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.bool)

    input_bundle = {"global_features": g_feat, "residual_features": r_feat}
    feats, masks = patad_proj(input_bundle, mask)
    assert len(feats) == 6, f"Expected 6 pyramid levels (L0-L5), got {len(feats)}"
    assert len(masks) == 6

    expected_lens = [64, 32, 16, 8, 4, 2]
    for i, (f, m, exp_len) in enumerate(zip(feats, masks, expected_lens)):
        assert f.shape == (B, C, exp_len), f"Level {i} shape mismatch: {f.shape} vs {(B, C, exp_len)}"
        assert m.shape == (B, exp_len)

    # Backward gradient flow check
    loss = sum(f.sum() for f in feats)
    loss.backward()
    assert g_feat.grad is not None and torch.isfinite(g_feat.grad).all()
    assert r_feat.grad is not None and torch.isfinite(r_feat.grad).all()


def test_patad_l2_to_l5_bitwise_invariance():
    conv_cfg = dict(kernel_size=3, proj_pdrop=0.0)
    attn_cfg = dict(n_head=4, n_mha_win_size=-1, attn_pdrop=0.0)

    patad_proj = PyramidAwareAsymmetricProj(
        in_channels=16,
        out_channels=16,
        arch=(1, 1, 5),
        conv_cfg=conv_cfg,
        attn_cfg=attn_cfg,
        asymmetric_split_level=2,
    )
    patad_proj.eval()

    B, C, T = 1, 16, 64
    g_feat = torch.randn(B, C, T)
    r_feat_1 = torch.randn(B, C, T)
    r_feat_2 = torch.randn(B, C, T) * 100.0  # massively different residual
    mask = torch.ones(B, T, dtype=torch.bool)

    with torch.no_grad():
        feats_1, _ = patad_proj({"global_features": g_feat, "residual_features": r_feat_1}, mask)
        feats_2, _ = patad_proj({"global_features": g_feat, "residual_features": r_feat_2}, mask)

    # L0 and L1 must differ because residuals are injected
    assert not torch.allclose(feats_1[0], feats_2[0]), "L0 must reflect residual changes"
    assert not torch.allclose(feats_1[1], feats_2[1]), "L1 must reflect residual changes"

    # L2 to L5 MUST be strictly bitwise identical
    for lvl in range(2, 6):
        assert torch.equal(feats_1[lvl], feats_2[lvl]), f"Level L{lvl} must be strictly invariant to residual R"


def test_patad_rejects_an_unimplemented_split_level():
    with pytest.raises(ValueError, match="frozen"):
        PyramidAwareAsymmetricProj(
            in_channels=16,
            out_channels=16,
            arch=(1, 1, 5),
            conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
            attn_cfg=dict(n_head=4, n_mha_win_size=-1, attn_pdrop=0.0),
            asymmetric_split_level=3,
        )
