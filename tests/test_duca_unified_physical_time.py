import pytest
import torch
import torch.nn as nn

from opentad.models.bricks.scale_adaptive_conv1d import ContinuousTimeScaleAdaptiveConv1d


def test_continuous_time_conv_rejects_duplicate_timestamps():
    conv = ContinuousTimeScaleAdaptiveConv1d(2, 3, kernel_size=3, padding=1)
    x = torch.randn(1, 2, 6)
    timestamps = torch.tensor([[0.0, 1.0, 2.0, 2.0, 4.0, 5.0]])

    with pytest.raises(ValueError, match="strictly increasing"):
        conv(x, temporal_positions=timestamps)


def test_continuous_time_conv_reports_uniform_grid_parity_diagnostics():
    torch.manual_seed(7)
    conv = ContinuousTimeScaleAdaptiveConv1d(2, 3, kernel_size=3, padding=1, bias=True)
    dense = nn.Conv1d(2, 3, kernel_size=3, padding=1, bias=True)
    dense.weight.data.copy_(conv.weight.data)
    dense.bias.data.copy_(conv.bias.data)

    x = torch.randn(2, 2, 10)
    timestamps = torch.arange(10, dtype=torch.float32).unsqueeze(0).expand(2, -1)

    out_ct = conv(x, temporal_positions=timestamps)
    out_dense = dense(x)

    assert torch.allclose(out_ct, out_dense, atol=1e-5, rtol=1e-5)
    summary = conv.latest_temporal_geometry_summary
    assert summary["uses_temporal_positions"] is True
    assert summary["duplicate_timestamps_forbidden"] is True
    assert summary["finite_offsets"] is True
    assert summary["offset_abs_max"] < 1e-6


def test_continuous_time_conv_rejects_nonpositive_delta_t():
    conv = ContinuousTimeScaleAdaptiveConv1d(2, 3, kernel_size=3, padding=1)
    x = torch.randn(1, 2, 6)
    delta_t = torch.ones(1, 6)
    delta_t[0, 3] = 0.0

    with pytest.raises(ValueError, match="strictly positive"):
        conv(x, delta_t=delta_t)
