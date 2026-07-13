from pathlib import Path

import pytest
from mmengine.config import Config

torch = pytest.importorskip("torch")

from opentad.models import build_detector


ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "configs/adatad/thumos/phystime_g1a_selected_axis_native_j192.py"
PHYSICAL = ROOT / "configs/adatad/thumos/phystime_g1a_physical_metric_native_j192.py"


def _pipeline_step(cfg, split, step_type):
    return next(step for step in cfg.dataset[split].pipeline if step["type"] == step_type)


def _parameter_schema(model):
    return {
        name: (tuple(parameter.shape), bool(parameter.requires_grad))
        for name, parameter in model.named_parameters()
    }


def test_g1a_configs_use_native_j192_without_feature_interpolation():
    configs = [
        Config.fromfile(SELECTED, lazy_import=False),
        Config.fromfile(PHYSICAL, lazy_import=False),
    ]

    for cfg in configs:
        assert cfg.raw_observation_count == 384
        assert cfg.native_token_count == 192
        assert cfg.model.native_temporal_geometry.expected_raw_count == 384
        assert cfg.model.native_temporal_geometry.expected_token_count == 192
        assert cfg.model.backbone.custom.strict_temporal_padding_mask is True
        assert cfg.model.projection.max_seq_len == 192
        post_types = [step["type"] for step in cfg.model.backbone.custom.post_processing_pipeline]
        assert post_types == ["Reduce", "Rearrange"]
        assert "Interpolate" not in post_types
        for split in ("train", "val", "test"):
            assert _pipeline_step(cfg, split, "LoadFrames")["remap_gt_to_selected_axis"] is False
            assert _pipeline_step(cfg, split, "BuildPhysTimeNativeTubeletGeometry")["tubelet_size"] == 2
            raw = _pipeline_step(cfg, split, "BuildPhysTimeRawFrameGeometry")
            assert raw["fps_relative_tolerance"] == pytest.approx(0.0125)
            assert raw["duration_relative_tolerance"] == pytest.approx(0.0125)
            assert raw["frame_count_relative_tolerance"] == pytest.approx(0.0001)


def test_g1a_configs_change_only_the_common_seconds_axis_tensor():
    selected = Config.fromfile(SELECTED, lazy_import=False)
    physical = Config.fromfile(PHYSICAL, lazy_import=False)

    for split in ("train", "val", "test"):
        assert _pipeline_step(selected, split, "BuildPhysTimeNativeTubeletGeometry")[
            "coordinate_mode"
        ] == "uniform_rank_seconds"
        assert _pipeline_step(physical, split, "BuildPhysTimeNativeTubeletGeometry")[
            "coordinate_mode"
        ] == "physical_time_seconds"

    assert selected.model.rpn_head.physical_grid_actionformer.enabled is True
    assert physical.model.rpn_head.physical_grid_actionformer.enabled is True
    assert (
        physical.model.rpn_head.physical_grid_actionformer.positions_key
        == "phystime_g1a_axis_positions_sec"
    )
    assert (
        physical.model.rpn_head.physical_grid_actionformer.axis_start_key
        == "phystime_g1a_axis_start_sec"
    )
    assert physical.model.rpn_head.physical_grid_actionformer.axis_end_key == "phystime_g1a_axis_end_sec"
    assert selected.model == physical.model

    for cfg in (selected, physical):
        for split in ("train", "val"):
            raw = _pipeline_step(cfg, split, "BuildPhysTimeRawFrameGeometry")
            assert raw["convert_gt_to_seconds"] is True
        assert _pipeline_step(cfg, "test", "BuildPhysTimeRawFrameGeometry")[
            "convert_gt_to_seconds"
        ] is False


def test_g1a_models_have_identical_parameter_schema_and_native_query_capacity():
    selected_cfg = Config.fromfile(SELECTED, lazy_import=False)
    physical_cfg = Config.fromfile(PHYSICAL, lazy_import=False)
    selected_cfg.model.backbone.custom.pretrain = None
    physical_cfg.model.backbone.custom.pretrain = None
    selected_model = build_detector(selected_cfg.model)
    physical_model = build_detector(physical_cfg.model)

    assert _parameter_schema(selected_model) == _parameter_schema(physical_model)
    expected_level_lengths = [192, 96, 48, 24, 12, 6]
    assert sum(expected_level_lengths) == 378
    assert selected_model.max_seq_len == physical_model.max_seq_len == 192
    assert selected_model.rpn_head.prior_generator.strides == [1, 2, 4, 8, 16, 32]
    assert physical_model.rpn_head.prior_generator.strides == [1, 2, 4, 8, 16, 32]
