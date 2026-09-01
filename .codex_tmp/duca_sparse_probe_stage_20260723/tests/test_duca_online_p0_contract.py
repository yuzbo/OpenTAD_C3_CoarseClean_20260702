from __future__ import annotations

from types import SimpleNamespace

import pytest

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"torch is unavailable in this environment: {exc}", allow_module_level=True)

from opentad.models import build_detector  # noqa: E402
from opentad.models.detectors.actionformer import ActionFormer  # noqa: E402
from opentad.models.utils.truetime_geometry import SELECTED_AXIS  # noqa: E402


def _duca_single_stage_cfg(in_channels: int = 3, budget: int = 4) -> dict:
    return dict(
        type="SingleStageDetector",
        frame_selector=dict(
            type="DucaOnlineFrameSelector",
            in_channels=in_channels,
            budget=budget,
            max_radius=2,
            selector_hidden_channels=4,
            detector_gradient_mode="st_sparse_gather_soft_context",
            actionness_source_cfg=dict(
                type="ZeroShotMotionActionnessSource",
                mode="motion",
                source_name="test_motion",
                thumos_trained=False,
                uses_labels=False,
                uses_teacher=False,
                uses_gt=False,
                uses_prediction_cache=False,
                calibration_split="none",
            ),
        ),
        rpn_head=dict(type="DucaOnlinePrecheckHead", in_channels=in_channels, require_gt_in_train=True),
    )


def _labels(batch: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    gt_segments = [torch.tensor([[0.0, 1.0]], dtype=torch.float32) for _ in range(batch)]
    gt_labels = [torch.tensor([0], dtype=torch.long) for _ in range(batch)]
    return gt_segments, gt_labels


def test_duca_live_selector_forbids_raw_prediction_cache_in_forward_detection() -> None:
    model = build_detector(_duca_single_stage_cfg())
    inputs = torch.randn(1, 3, 8)
    masks = torch.ones(1, 8, dtype=torch.bool)
    metas = [{"video_name": "v"}]
    post_cfg = SimpleNamespace(sliding_window=False, nms=None, pre_nms_thresh=0.0, pre_nms_topk=10)

    for infer_cfg in (
        SimpleNamespace(load_from_raw_predictions=True, save_raw_prediction=False, folder="unused"),
        SimpleNamespace(load_from_raw_predictions=False, save_raw_prediction=True, folder="unused"),
    ):
        with pytest.raises(ValueError, match="DUCA.*raw-prediction"):
            model.forward_detection(inputs, masks, metas, infer_cfg, post_cfg, ext_cls=["action"])


class _MissingRemapSelector(nn.Module):
    forbid_raw_prediction_cache = True

    def forward_test(self, inputs, masks, metas=None, **kwargs):
        out_metas = [
            {
                "video_name": "bad",
                "detector_prediction_inverse_map_required": True,
                "detector_output_coordinate_space": SELECTED_AXIS,
            }
        ]
        return {"inputs": inputs, "masks": masks, "metas": out_metas}


class _RecordingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.called = False

    def forward_test(self, feat_list, mask_list, metas=None, **kwargs):
        self.called = True
        return [torch.zeros(1, 2)], [torch.ones(1, 1)]


def _bare_actionformer_for_forward_test() -> ActionFormer:
    model = ActionFormer.__new__(ActionFormer)
    nn.Module.__init__(model)
    model.frame_selector = _MissingRemapSelector()
    model.rpn_head = _RecordingHead()
    model.max_seq_len = 4
    model.max_div_factor = 1
    model.token_compressor = None
    model.pc_ot_mras_reader = None
    model.pc_ot_mras_reader_eval_override = None
    return model


def test_actionformer_forward_test_requires_duca_selected_axis_remap_metadata() -> None:
    model = _bare_actionformer_for_forward_test()

    with pytest.raises(RuntimeError, match="selected_axis_to_true_time_dense_index"):
        model.forward_test(torch.randn(1, 3, 4), torch.ones(1, 4, dtype=torch.bool), metas=[{"video_name": "bad"}])

    assert model.rpn_head.called is False


def test_standard_forward_detector_loss_optimizer_step_moves_duca_selector_parameters() -> None:
    torch.manual_seed(17)
    model = build_detector(_duca_single_stage_cfg(in_channels=3, budget=4))
    model.train()
    inputs = torch.randn(2, 3, 8)
    masks = torch.ones(2, 8, dtype=torch.bool)
    metas = [{"video_name": "a"}, {"video_name": "b"}]
    gt_segments, gt_labels = _labels(2)
    selector_params = [param for param in model.frame_selector.parameters() if param.requires_grad]
    before = [param.detach().clone() for param in selector_params]

    optimizer = torch.optim.SGD(model.frame_selector.parameters(), lr=0.5)
    losses = model(
        inputs,
        masks,
        metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
        return_loss=True,
    )
    optimizer.zero_grad()
    losses["cost"].backward()
    grad_norm = sum(float(param.grad.detach().abs().sum().item()) for param in selector_params if param.grad is not None)
    optimizer.step()
    delta = sum(float((param.detach() - old).abs().sum().item()) for param, old in zip(selector_params, before))

    assert "loss_detector" in losses
    assert grad_norm > 0.0
    assert delta > 0.0
    assert model.rpn_head.last_gt_segments is not None
    assert model.rpn_head.last_gt_labels is gt_labels
    assert len(model.rpn_head.last_gt_segments) == len(gt_segments)
    assert model.rpn_head.last_metas[0]["gt_remapped_to_selected_axis"] is True
    assert model.rpn_head.last_metas[0]["gt_segments_original_time"] == gt_segments[0].tolist()
    assert model.rpn_head.last_masks.sum(dim=1).tolist() == [4, 4]
