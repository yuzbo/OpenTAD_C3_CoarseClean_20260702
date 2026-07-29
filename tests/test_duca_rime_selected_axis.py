from __future__ import annotations

import pytest
import torch
from mmengine.config import Config

from opentad.models.detectors.actionformer import ActionFormer
from opentad.models.selectors.duca_rime_frame_selector import (
    DucaRimeFrameSelector,
)
from opentad.models.utils.truetime_geometry import (
    SELECTED_AXIS,
    TRUE_TIME_AXIS,
    truetime_map_from_metadata,
)


def _selector(*, mode="selected_axis_plugin"):
    return DucaRimeFrameSelector(
        in_channels=3,
        rime_arm="uniform_mixed_k",
        candidate_budgets=(16, 32),
        candidate_costs=(16.0, 32.0),
        fixed_budget=16,
        dense_window_size=32,
        target_mean_cost=24.0,
        execution_quantum=16,
        require_frozen_protocol=False,
        mixed_k_schedule_counts=(1, 1),
        mixed_k_schedule_seed=1,
        detector_bridge_gradient_scale=0.0,
        actionness_source_cfg=None,
        detector_coordinate_mode=mode,
    )


def _batch(*, training: bool):
    inputs = torch.randn(1, 3, 32, 2, 2)
    masks = torch.ones(1, 32, dtype=torch.bool)
    meta = {
        "frame_inds": list(range(32)),
        "avg_fps": 1.0,
        "video_name": "test",
    }
    if training:
        meta.update(
            {
                "duca_stateless_epoch": 0,
                "duca_stateless_sample_index": 0,
            }
        )
    return inputs, masks, [meta]


def test_selected_axis_rime_remaps_gt_and_emits_inverse_map():
    selector = _selector()
    selector.train()
    inputs, masks, metas = _batch(training=True)
    original = torch.tensor([[4.0, 12.0]])
    output = selector.forward_train(
        inputs,
        masks,
        metas,
        gt_segments=[original],
        gt_labels=[torch.tensor([1])],
    )
    meta = output["metas"][0]
    assert selector.selector_variant == "duca_rime_selected_axis"
    assert meta["duca_contract"] == "duca_rime_selected_axis_plugin_v2"
    assert meta["detector_output_coordinate_space"] == SELECTED_AXIS
    assert meta["detector_prediction_inverse_map_required"] is True
    assert meta["gt_remapped_to_selected_axis"] is True
    assert "physical_grid_contract" not in meta
    assert output["inputs"].shape[2] == output["masks"].shape[1]
    assert bool(output["masks"].all().item())

    time_map = truetime_map_from_metadata(meta, require_inverse_map=True)
    roundtrip = time_map.remap_segments(
        output["gt_segments"][0],
        source_coordinate_space=SELECTED_AXIS,
        target_coordinate_space=TRUE_TIME_AXIS,
    )
    assert torch.allclose(roundtrip, original, atol=1.0e-5)


def test_selected_axis_inference_does_not_claim_physical_head():
    selector = _selector().eval()
    inputs, masks, metas = _batch(training=False)
    output = selector.forward_test(inputs, masks, metas)
    meta = output["metas"][0]
    assert meta["proposal_axis"] == SELECTED_AXIS
    assert meta["gt_remapped_to_selected_axis"] is False
    assert meta.get("physical_grid_contract") is None


def test_selected_axis_detector_rejects_inverse_map_outside_dense_valid_axis():
    selector = _selector().eval()
    inputs, masks, metas = _batch(training=False)
    output = selector.forward_test(inputs, masks, metas)
    invalid_meta = dict(output["metas"][0])
    invalid_positions = list(
        invalid_meta["selected_axis_to_true_time_dense_index"]
    )
    invalid_positions[-1] = int(invalid_meta["irregular_dense_valid_len"])
    invalid_meta["selected_axis_to_true_time_dense_index"] = invalid_positions

    detector = ActionFormer.__new__(ActionFormer)
    torch.nn.Module.__init__(detector)
    detector.rpn_head = torch.nn.Identity()
    detector.frame_selector = selector

    with pytest.raises(
        RuntimeError,
        match="inverse-map positions exceed the declared dense valid axis",
    ):
        detector._validate_rime_selected_axis_contract(
            output["inputs"],
            output["masks"],
            [invalid_meta],
        )


def test_physical_integration_mode_remains_backward_compatible():
    selector = _selector(mode="physical_head_integration").eval()
    inputs, masks, metas = _batch(training=False)
    output = selector.forward_test(inputs, masks, metas)
    meta = output["metas"][0]
    assert selector.selector_variant == "duca_rime_physical"
    assert meta["duca_contract"] == "duca_rime_physical_dynamic_k_v1"
    assert meta["physical_grid_contract"] == "duca_rime_physical_dynamic_k_v1"
    assert meta["detector_prediction_inverse_map_required"] is False


def test_rime_amp_replay_state_restores_python_caches():
    selector = _selector()
    snapshot = selector.capture_amp_replay_state()
    selector._last_replay_effective_k = (16,)
    selector._last_decision_provenance = ({"source": "mutated"},)
    selector._last_mixed_k_schedule_indices = (7,)
    selector._last_mixed_k_schedule_source = "mutated"
    selector.restore_amp_replay_state(snapshot)
    assert selector._last_replay_effective_k == ()
    assert selector._last_decision_provenance == ()
    assert selector._last_mixed_k_schedule_indices == ()
    assert selector._last_mixed_k_schedule_source is None


def test_selected_axis_configs_restore_standard_heads(monkeypatch):
    monkeypatch.setenv("DUCA_RIME_TRAIN_BLOCK_LIST", "/tmp/train.json")
    monkeypatch.setenv("DUCA_RIME_DEVELOPMENT_BLOCK_LIST", "/tmp/dev.json")
    monkeypatch.setenv("DUCA_RIME_TARGETS_JSONL", "/tmp/targets.jsonl")
    monkeypatch.setenv("DUCA_RIME_TARGETS_SHA256", "a" * 64)
    for config_path in (
        "configs/adatad/thumos/duca_rime_full_selected_axis_total60.py",
        "configs/adatad/thumos/duca_rime_uniform_mixed_k_selected_axis_total60.py",
        "configs/adatad/thumos/duca_rime_full_tridet_selected_axis_total60.py",
    ):
        config = Config.fromfile(config_path)
        assert (
            config.model.frame_selector.detector_coordinate_mode
            == "selected_axis_plugin"
        )
        assert config.model.rpn_head.physical_grid_actionformer is None
        assert config.duca_rime_contract.pre_backbone_plugin is True
        assert config.duca_rime_contract.detector_head_modified is False
