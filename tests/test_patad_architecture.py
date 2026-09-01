from __future__ import annotations

import torch
import pytest
from opentad.models.projections.pyramid_aware_asymmetric_proj import PyramidAwareAsymmetricProj
from opentad.models.projections.actionformer_proj import Conv1DTransformerProj


def test_patad_projection_shape_and_parameter_parity():
    conv_cfg = dict(kernel_size=3, proj_pdrop=0.0)
    attn_cfg = dict(n_head=4, n_mha_win_size=19, attn_pdrop=0.0)
    norm_cfg = dict(type="LN")

    std_proj = Conv1DTransformerProj(
        in_channels=384,
        out_channels=384,
        arch=(2, 2, 5),
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        attn_cfg=attn_cfg,
        max_seq_len=768,
    )

    patad_proj = PyramidAwareAsymmetricProj(
        in_channels=384,
        out_channels=384,
        arch=(2, 2, 5),
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        attn_cfg=attn_cfg,
        max_seq_len=768,
        asymmetric_split_level=2,
    )

    # Parity check: zero extra parameters
    std_params = sum(p.numel() for p in std_proj.parameters())
    patad_params = sum(p.numel() for p in patad_proj.parameters())
    assert std_params == patad_params, f"PA-TAD must have 0 extra parameters, got {patad_params} vs {std_params}"

    # Forward check
    B, C, T = 2, 384, 768
    x = torch.randn(B, C, T, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.bool)
    burst_mask = torch.zeros(B, T, dtype=torch.bool)
    burst_mask[:, 100:200] = True

    feats, masks = patad_proj(x, mask, burst_mask=burst_mask)
    assert len(feats) == 6, f"Expected 6 pyramid levels (L0-L5), got {len(feats)}"
    assert len(masks) == 6

    # Verify pyramid strides: 768, 384, 192, 96, 48, 24
    expected_lens = [768, 384, 192, 96, 48, 24]
    for i, (f, m, exp_len) in enumerate(zip(feats, masks, expected_lens)):
        assert f.shape == (B, C, exp_len), f"Level {i} shape mismatch: {f.shape} vs {(B, C, exp_len)}"
        assert m.shape == (B, exp_len)

    # Backward gradient flow check
    loss = sum(f.sum() for f in feats)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_patad_asymmetric_split_level():
    conv_cfg = dict(kernel_size=3, proj_pdrop=0.0)
    attn_cfg = dict(n_head=4, n_mha_win_size=-1, attn_pdrop=0.0)

    patad_proj = PyramidAwareAsymmetricProj(
        in_channels=256,
        out_channels=256,
        arch=(2, 2, 5),
        conv_cfg=conv_cfg,
        attn_cfg=attn_cfg,
        asymmetric_split_level=2,
    )
    assert patad_proj.asymmetric_split_level == 2
