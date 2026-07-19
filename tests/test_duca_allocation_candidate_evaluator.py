from __future__ import annotations

from pathlib import Path

import pytest


try:
    import torch
except (ImportError, OSError) as exc:
    pytest.skip(f"torch runtime unavailable: {exc}", allow_module_level=True)

from tools.bata.evaluate_duca_allocation_candidates import (  # noqa: E402
    evaluate_one_candidate,
    prepare_candidate_sample,
)


class _FakeRpnHead:
    def collect_debug_state(self):
        return {
            "physical_grid_actionformer_enabled": True,
            "physical_grid_actionformer_selected_count": 4,
        }


class _FakeDetector:
    def __init__(self):
        self.rpn_head = _FakeRpnHead()
        self.observed = None

    def forward_train(
        self,
        inputs,
        masks,
        metas,
        gt_segments,
        gt_labels,
        **kwargs,
    ):
        self.observed = {
            "inputs": inputs,
            "masks": masks,
            "metas": metas,
            "gt_segments": gt_segments,
            "gt_labels": gt_labels,
            "kwargs": kwargs,
        }
        assert metas[0]["irregular_native_axis"] is True
        assert metas[0]["remap_gt_to_selected_axis"] is False
        assert metas[0]["gt_remapped_to_selected_axis"] is False
        assert metas[0]["irregular_dense_valid_len"] == 8
        assert gt_segments[0].tolist() == [[1.0, 6.0]]
        return {
            "cls_loss": inputs.sum() * 0 + 2.0,
            "reg_loss": inputs.sum() * 0 + 0.5,
        }


def test_candidate_preparation_gathers_raw_frames_and_preserves_dense_gt() -> None:
    inputs = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    masks = torch.ones((1, 8), dtype=torch.bool)
    prepared = prepare_candidate_sample(
        inputs=inputs,
        masks=masks,
        meta={"video_name": "v", "window_start_frame": 0},
        gt_segments=torch.tensor([[1.0, 6.0]]),
        gt_labels=torch.tensor([2]),
        positions=[0, 2, 5, 7],
        requested_budget=4,
    )
    assert prepared["inputs"].flatten().tolist() == [0.0, 2.0, 5.0, 7.0]
    assert prepared["masks"].tolist() == [[True, True, True, True]]
    assert prepared["metas"][0]["selected_dense_indices"] == [0, 2, 5, 7]
    assert prepared["metas"][0]["selected_valid_len"] == 4
    assert prepared["gt_segments"][0].tolist() == [[1.0, 6.0]]


def test_short_valid_prefix_is_padded_without_inventing_physical_positions() -> None:
    inputs = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5)
    masks = torch.ones((1, 5), dtype=torch.bool)
    prepared = prepare_candidate_sample(
        inputs=inputs,
        masks=masks,
        meta={"video_name": "v", "window_start_frame": 0},
        gt_segments=torch.tensor([[0.0, 4.0]]),
        gt_labels=torch.tensor([1]),
        positions=[0, 1, 2, 3, 4],
        requested_budget=8,
    )
    assert prepared["inputs"].shape[-1] == 8
    assert prepared["inputs"].flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0]
    assert prepared["masks"].tolist() == [[True, True, True, True, True, False, False, False]]
    assert prepared["metas"][0]["irregular_selected_positions"] == [0, 1, 2, 3, 4]


def test_frozen_candidate_loss_uses_official_cls_and_reg_terms_only() -> None:
    detector = _FakeDetector()
    prepared = prepare_candidate_sample(
        inputs=torch.arange(8, dtype=torch.float32).reshape(1, 1, 8),
        masks=torch.ones((1, 8), dtype=torch.bool),
        meta={"video_name": "v", "window_start_frame": 0},
        gt_segments=torch.tensor([[1.0, 6.0]]),
        gt_labels=torch.tensor([2]),
        positions=[0, 2, 5, 7],
        requested_budget=4,
    )
    result = evaluate_one_candidate(detector, prepared)
    assert result["cls_loss"] == 2.0
    assert result["reg_loss"] == 0.5
    assert result["detector_loss"] == 2.5
    assert detector.observed["kwargs"]["_duca_skip_frame_selector"] is True
    assert detector.observed["kwargs"]["_duca_counterfactual_eval"] is True


def test_evaluator_config_explicitly_forbids_selected_axis_gt_remap() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "adatad"
        / "thumos"
        / "duca_allocation_ceiling_physical_grid_evaluator.py"
    )
    text = path.read_text(encoding="utf-8")
    assert "physical_grid_actionformer=True" in text
    assert "dense_axis_gt=True" in text
    assert "selected_axis_gt_remap=False" in text
    assert "trains_model=False" in text


def test_replay_map_runner_is_sealed_validation_only_and_hash_bound() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_duca_allocation_replay_map.sh"
    )
    text = path.read_text(encoding="utf-8")
    assert "DUCA_ALLOCATION_VALIDATION_AUTHORIZED" in text
    assert "validate_duca_allocation_ceiling_artifact" in text
    assert "DUCA_ALLOCATION_ARTIFACT_SHA256" in text
    assert "DUCA_ALLOCATION_ALLOW_PRIVILEGED" in text
    assert "selected_axis_gt_remap" in text
    assert "tools/test.py" in text
