import pytest
import torch
import torch.nn as nn
from mmengine.config import Config
from pathlib import Path

from opentad.models.bricks.scale_adaptive_conv1d import ContinuousTimeScaleAdaptiveConv1d
from opentad.models.selectors.duca_online_frame_selector import DucaOnlineFrameSelector


ROOT = Path(__file__).resolve().parents[1]


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


def test_physical_time_matrix_config_keeps_gt_on_native_axis():
    cfg = Config.fromfile(
        str(
            ROOT
            / "configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_c11_seed3407.py"
        )
    )

    selector_cfg = cfg.model.frame_selector
    assert selector_cfg.remap_gt_to_selected_axis is False
    assert selector_cfg.selected_axis_remap_required is False
    assert cfg.model.rpn_head.physical_grid_actionformer.enabled is True


def test_selector_outputs_temporal_positions_for_physical_grid():
    selector = DucaOnlineFrameSelector(
        in_channels=3,
        budget=4,
        dense_window_size=8,
        selector_hidden_channels=0,
        acquisition_policy="exact_uniform",
        remap_gt_to_selected_axis=False,
        selected_axis_remap_required=False,
        actionness_source_cfg={
            "type": "ZeroShotMotionActionnessSource",
            "mode": "motion",
            "source_name": "unit_motion",
            "thumos_trained": False,
            "uses_labels": False,
            "uses_teacher": False,
            "uses_gt": False,
            "uses_prediction_cache": False,
            "calibration_split": "none",
        },
    )
    inputs = torch.randn(1, 3, 8)
    masks = torch.ones(1, 8, dtype=torch.bool)

    outputs = selector.forward_test(inputs, masks, metas=[{}])

    assert outputs["temporal_positions"].shape == (1, 4)
    assert outputs["temporal_positions"].dtype == inputs.dtype
    values = outputs["temporal_positions"][0].tolist()
    assert values == sorted(values)
    assert outputs["metas"][0]["irregular_native_axis"] is True
    assert outputs["metas"][0]["temporal_positions"] == [int(value) for value in values]
