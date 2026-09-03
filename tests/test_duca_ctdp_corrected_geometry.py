"""Focused contracts for the corrected CT-DP geometry path."""
from __future__ import annotations

import pytest
try:
    import torch
    import torch.nn as nn
except OSError as exc:  # pragma: no cover - Windows CI without a Torch runtime
    pytest.skip(f"torch runtime unavailable in this environment: {exc}", allow_module_level=True)

from opentad.models.backbones.vit_adapter import Block
from opentad.models.bricks.scale_adaptive_conv1d import ContinuousTimeScaleAdaptiveConv1d
from opentad.models.selectors.dual_phase_frame_selector import DualPhaseFrameSelector


def test_four_time_objects_are_distinct_and_short_padding_is_masked():
    selector = DualPhaseFrameSelector(total_budget=8, scaffold_budget=4, burst_budget=4, force_uniform=True)
    inputs = torch.randn(1, 3, 5, 16, 16)
    mask = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.bool)
    meta = [{"video_name": "short"}]
    out = selector.forward_train(inputs, mask, meta)
    assert out["selected_positions"].shape == (1, 8)
    assert out["masks"].tolist() == [[True] * 5 + [False] * 3]
    for key in ("raw_frame_index", "selected_frame_rank", "tubelet_midpoint_physical_time", "detector_feature_physical_time"):
        assert key in meta[0]
    assert meta[0]["temporal_position_unit"] == "raw_frame_index"
    assert meta[0]["raw_frame_index"].dtype == torch.long
    assert meta[0]["selected_frame_rank"][5:].tolist() == [-1, -1, -1]
    assert torch.equal(meta[0]["detector_feature_physical_time"], out["detector_feature_physical_time"][0])


def test_ct_conv_uniform_exact_parity_all_stride_dilation_padding():
    torch.manual_seed(3)
    for stride in (1, 2):
        for dilation in (1, 2):
            padding = dilation
            ct = ContinuousTimeScaleAdaptiveConv1d(2, 3, kernel_size=3, stride=stride, dilation=dilation, padding=padding, bias=True, enable_learned_modulation=False)
            ref = nn.Conv1d(2, 3, 3, stride=stride, dilation=dilation, padding=padding, bias=True)
            ref.weight.data.copy_(ct.weight.data)
            ref.bias.data.copy_(ct.bias.data)
            x = torch.randn(2, 2, 32)
            tau = torch.arange(32, dtype=torch.float32).expand(2, -1)
            assert (ct(x, temporal_positions=tau) - ref(x)).abs().max() < 1e-5


def test_ct_conv_level_nominal_spacing_and_finite_offsets():
    conv = ContinuousTimeScaleAdaptiveConv1d(1, 1, kernel_size=3, padding=1, reference_spacing_mode="level_nominal", level_nominal_spacing=2.0, bias=False)
    x = torch.zeros(1, 1, 8)
    x[0, 0, 3] = 1.0
    tau = torch.arange(8, dtype=torch.float32).mul(2).view(1, -1)
    out = conv(x, temporal_positions=tau)
    assert torch.isfinite(out).all()


def test_bamod_selected_queries_use_full_kv_and_never_pad():
    block = Block(embed_dims=16, num_heads=4, mlp_ratio=2.0, use_adapter=False, temporal_size=4)
    block.eval()
    x = torch.randn(1, 8, 16)
    scores = torch.full((1, 8), -10.0)
    scores[:, :4] = 10.0
    mask = torch.ones(1, 8, dtype=torch.bool)
    out = block.forward_amod(x, 2, 2, scores, capacity=0.5, temporal_token_mask=mask)
    assert torch.equal(out[:, 4:], x[:, 4:])
    changed = x.clone()
    changed[:, 7] += 5.0
    out_changed = block.forward_amod(changed, 2, 2, scores, capacity=0.5, temporal_token_mask=mask)
    assert not torch.equal(out[:, :4], out_changed[:, :4])


def test_bamod_supports_mixed_valid_token_counts_in_batched_execution():
    block = Block(embed_dims=16, num_heads=4, mlp_ratio=2.0, use_adapter=False, temporal_size=4)
    x = torch.randn(2, 8, 16)
    scores = torch.randn(2, 8)
    mask = torch.tensor([[1] * 8, [1] * 6 + [0] * 2], dtype=torch.bool)
    out = block.forward_amod(x, 2, 2, scores, capacity=0.5, temporal_token_mask=mask)
    assert out.shape == x.shape
    assert torch.equal(out[1, 6:], x[1, 6:])
