from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from opentad.models.builder import HEADS
from opentad.models.detectors.single_stage import SingleStageDetector
from opentad.models.selectors.truetime_joint_selector import selector_grad_norm


@HEADS.register_module(force=True)
class _TruetimeSelectorGradientHead(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv = nn.Conv1d(int(in_channels), 1, kernel_size=1)

    def forward_train(self, feat_list, mask_list, gt_segments=None, gt_labels=None, **kwargs):
        logits = self.conv(feat_list).squeeze(1)
        target = torch.linspace(0.0, 1.0, logits.shape[-1], device=logits.device, dtype=logits.dtype)
        target = target.unsqueeze(0).expand_as(logits)
        per_time = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        valid = mask_list.to(device=logits.device, dtype=logits.dtype)
        return {"loss_detector": (per_time * valid).sum() / valid.sum().clamp_min(1.0)}


def test_single_stage_detector_loss_backpropagates_to_truetime_selector() -> None:
    detector = SingleStageDetector(
        frame_selector=dict(
            type="TrueTimeRelaxedHardTopKSelector",
            in_channels=3,
            selected_count=4,
            dense_len=8,
            temperature=0.7,
            selector_hidden_channels=4,
            allow_gt_selection=False,
            allow_teacher_utility=False,
            coordinate_space="selected_axis_index",
            true_time_source_axis="true_time_dense_index",
            detector_gradient_mode="st_sparse_gather",
            slot_softmax_temperature=0.7,
            slot_distance_penalty=1.0,
        ),
        rpn_head=dict(type="_TruetimeSelectorGradientHead", in_channels=3),
    )
    detector.train()
    inputs = torch.randn(2, 3, 8, requires_grad=True)
    masks = torch.ones(2, 8, dtype=torch.bool)
    metas = [{"video_name": "a"}, {"video_name": "b"}]

    losses = detector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=metas,
        gt_segments=[torch.empty(0, 2), torch.empty(0, 2)],
        gt_labels=[torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)],
    )

    assert "selector_selected_count_mean" not in losses
    assert "selector_selected_count_std" not in losses
    losses["cost"].backward()

    assert "loss_detector" in losses
    assert "selector_entropy_loss" in losses
    assert selector_grad_norm(detector.frame_selector) > 0.0


def test_single_stage_detector_without_selector_keeps_existing_path() -> None:
    detector = SingleStageDetector(rpn_head=dict(type="_TruetimeSelectorGradientHead", in_channels=3))
    inputs = torch.randn(2, 3, 4, requires_grad=True)
    masks = torch.ones(2, 4, dtype=torch.bool)

    losses = detector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=[{}, {}],
        gt_segments=[torch.empty(0, 2), torch.empty(0, 2)],
        gt_labels=[torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)],
    )

    assert not hasattr(detector, "frame_selector")
    assert "selector_entropy_loss" not in losses
    assert "loss_detector" in losses


def test_single_stage_post_processing_remaps_selector_predictions_to_true_time_seconds() -> None:
    detector = SingleStageDetector()
    predictions = (
        torch.tensor([[[0.0, 2.0], [1.5, 3.0]]]),
        torch.tensor([[[0.9], [0.8]]]),
    )
    metas = [
        {
            "video_name": "v1",
            "fps": 1.0,
            "snippet_stride": 1.0,
            "offset_frames": 0.0,
            "window_start_frame": 0.0,
            "duration": 200.0,
            "detector_prediction_inverse_map_required": True,
            "detector_output_coordinate_space": "selected_axis_index",
            "selected_axis_to_true_time_dense_index": [10, 20, 50, 100],
            "irregular_dense_valid_len": 120,
            "irregular_selected_valid_len": 4,
            "irregular_selected_count": 4,
            "irregular_native_axis": True,
        }
    ]
    results = detector.post_processing(
        predictions,
        metas,
        SimpleNamespace(sliding_window=False, nms=None),
        ["action"],
    )

    assert results["v1"][0]["segment"] == [10.0, 50.0]
    assert results["v1"][1]["segment"] == [35.0, 100.0]
