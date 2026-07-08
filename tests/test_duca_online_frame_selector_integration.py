from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"torch is unavailable in this environment: {exc}", allow_module_level=True)

from opentad.models import build_detector  # noqa: E402
from opentad.models.detectors.single_stage import SingleStageDetector  # noqa: E402
from opentad.models.selectors.truetime_joint_selector import selector_grad_norm  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PRECHECK = ROOT / "tools" / "bata" / "validate_duca_online_adatad_precheck.py"


def _duca_model_cfg(in_channels: int = 3, budget: int = 4) -> dict:
    return dict(
        type="SingleStageDetector",
        frame_selector=dict(
            type="DucaOnlineFrameSelector",
            in_channels=in_channels,
            budget=budget,
            max_radius=2,
            selector_hidden_channels=4,
            detector_gradient_mode="st_sparse_gather",
        ),
        rpn_head=dict(type="DucaOnlinePrecheckHead", in_channels=in_channels),
    )


def _labels(batch: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    gt_segments = [torch.tensor([[0.0, 1.0]], dtype=torch.float32) for _ in range(batch)]
    gt_labels = [torch.tensor([0], dtype=torch.long) for _ in range(batch)]
    return gt_segments, gt_labels


def test_build_detector_constructs_duca_selector_and_standard_forward_trains_selector() -> None:
    torch.manual_seed(3)
    model = build_detector(_duca_model_cfg(in_channels=3, budget=4))
    assert model.frame_selector.__class__.__name__ == "DucaOnlineFrameSelector"
    model.train()
    inputs = torch.randn(2, 3, 8, requires_grad=True)
    masks = torch.ones(2, 8, dtype=torch.bool)
    metas = [{"video_name": "a"}, {"video_name": "b"}]
    gt_segments, gt_labels = _labels(2)

    losses = model(
        inputs,
        masks,
        metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
        return_loss=True,
        teacher_utility=torch.randn(2, 8),
    )
    losses["cost"].backward()

    assert "loss_detector" in losses
    assert "selector_entropy_anti_collapse_loss" in losses
    assert selector_grad_norm(model.frame_selector) > 0.0
    assert model.rpn_head.last_gt_segments is gt_segments
    assert model.rpn_head.last_gt_labels is gt_labels
    assert model.rpn_head.last_masks.sum(dim=1).tolist() == [4, 4]


def test_duca_selector_supports_dynamic_budget_masks_and_video_input_shape() -> None:
    torch.manual_seed(5)
    model = build_detector(_duca_model_cfg(in_channels=3, budget=7))
    inputs = torch.randn(2, 1, 3, 8, 2, 2, requires_grad=True)
    masks = torch.ones(2, 8, dtype=torch.bool)
    metas = [{"video_name": "a"}, {"video_name": "b"}]
    gt_segments, gt_labels = _labels(2)

    losses = model(
        inputs,
        masks,
        metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
        return_loss=True,
        budget=torch.tensor([4, 7]),
    )
    losses["cost"].backward()

    assert model.rpn_head.last_input_shape == (2, 1, 3, 7, 2, 2)
    assert model.rpn_head.last_masks.sum(dim=1).tolist() == [4, 7]
    assert selector_grad_norm(model.frame_selector) > 0.0


def test_duca_forward_test_rejects_nested_forbidden_payloads_and_writes_remap_metadata() -> None:
    torch.manual_seed(7)
    model = build_detector(_duca_model_cfg(in_channels=3, budget=4))
    inputs = torch.randn(1, 3, 8)
    masks = torch.ones(1, 8, dtype=torch.bool)

    with pytest.raises(ValueError, match="raw_predictions"):
        model.forward_test(
            inputs,
            masks,
            metas=[{"video_name": "bad", "nested": {"raw_predictions": [1, 2, 3]}}],
        )

    predictions = model.forward_test(
        inputs,
        masks,
        metas=[
            {
                "video_name": "ok",
                "fps": 1.0,
                "snippet_stride": 1.0,
                "offset_frames": 0.0,
                "window_start_frame": 0.0,
                "duration": 20.0,
            }
        ],
    )
    meta = model.rpn_head.last_metas[0]

    assert meta["detector_prediction_inverse_map_required"] is True
    assert meta["detector_output_coordinate_space"] == "selected_axis_index"
    assert len(meta["selected_axis_to_true_time_dense_index"]) == 4
    assert meta["irregular_selected_valid_len"] == 4
    assert predictions[0][0].shape[-1] == 2

    post_results = SingleStageDetector().post_processing(
        predictions,
        model.rpn_head.last_metas,
        SimpleNamespace(sliding_window=False, nms=None, pre_nms_thresh=0.0, pre_nms_topk=10),
        ["action"],
    )
    assert "ok" in post_results
    assert post_results["ok"]


def test_duca_online_precheck_script_uses_registry_model(tmp_path: Path) -> None:
    output_json = tmp_path / "duca_online_real_precheck.json"
    completed = subprocess.run(
        [sys.executable, str(PRECHECK), "--output-json", str(output_json), "--budget", "4", "--dense-len", "8"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    stdout_payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert stdout_payload == payload
    assert payload["status"] == "ok"
    assert payload["implementation"] == "opentad.models registry"
    assert payload["build_detector"] is True
    assert payload["standard_forward_train"] is True
    assert payload["real_detector_loss_selector_grad_nonzero"] is True
    assert payload["train_gt_reaches_detector"] is True
    assert payload["teacher_free_inference"] is True
    assert payload["uses_ledger_for_decision"] is False
    assert payload["remap_metadata_present"] is True
