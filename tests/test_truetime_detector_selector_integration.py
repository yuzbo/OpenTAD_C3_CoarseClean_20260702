from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def _install_mmaction_registry_shim() -> None:
    if "mmaction.registry" in sys.modules:
        return

    class _RegistryShim:
        def register_module(self, *args, **kwargs):
            def _decorator(cls):
                return cls

            if args and isinstance(args[0], type):
                return args[0]
            return _decorator

    mmaction = types.ModuleType("mmaction")
    mmaction.__version__ = "test-shim"
    mmaction.__path__ = []
    registry = types.ModuleType("mmaction.registry")
    registry.MODELS = _RegistryShim()
    utils = types.ModuleType("mmaction.utils")
    utils.ConfigType = dict
    utils.OptConfigType = object
    models = types.ModuleType("mmaction.models")
    models.__path__ = []
    backbones = types.ModuleType("mmaction.models.backbones")
    backbones.__path__ = []
    swin = types.ModuleType("mmaction.models.backbones.swin")
    vit_mae = types.ModuleType("mmaction.models.backbones.vit_mae")

    class _UnusedBackboneShim(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    def _unused(*args, **kwargs):
        raise RuntimeError("mmaction swin shim should not be executed in TrueTime focused tests")

    swin.PatchEmbed3D = _UnusedBackboneShim
    swin.PatchMerging = _UnusedBackboneShim
    swin.WindowAttention3D = _UnusedBackboneShim
    swin.Mlp = _UnusedBackboneShim
    swin.get_window_size = _unused
    swin.compute_mask = _unused
    swin.window_partition = _unused
    swin.window_reverse = _unused
    vit_mae.get_sinusoid_encoding = _unused
    nms_1d_cpu = types.ModuleType("nms_1d_cpu")
    nms_1d_cpu.nms = _unused
    nms_1d_cpu.softnms = _unused
    align_1d = types.ModuleType("Align1D")
    align_1d.forward = _unused
    align_1d.backward = _unused
    boundary_max_pooling_cuda = types.ModuleType("boundary_max_pooling_cuda")
    boundary_max_pooling_cuda.forward = _unused
    boundary_max_pooling_cuda.backward = _unused

    sys.modules["mmaction"] = mmaction
    sys.modules["mmaction.registry"] = registry
    sys.modules["mmaction.utils"] = utils
    sys.modules["mmaction.models"] = models
    sys.modules["mmaction.models.backbones"] = backbones
    sys.modules["mmaction.models.backbones.swin"] = swin
    sys.modules["mmaction.models.backbones.vit_mae"] = vit_mae
    sys.modules.setdefault("nms_1d_cpu", nms_1d_cpu)
    sys.modules.setdefault("Align1D", align_1d)
    sys.modules.setdefault("boundary_max_pooling_cuda", boundary_max_pooling_cuda)


_install_mmaction_registry_shim()

from opentad.models.builder import HEADS
import opentad.models.detectors.single_stage as single_stage_module
from opentad.models.detectors.single_stage import SingleStageDetector
from opentad.models.selectors.truetime_joint_selector import selector_grad_norm


ROOT = Path(__file__).resolve().parents[1]
PRECHECK_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_truetime_joint_selector_adatad_precheck.py"
PRECHECK_RUNNER = ROOT / "tools" / "bata" / "run_truetime_joint_selector_precheck.py"


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
    losses["cost"].backward()

    assert "loss_detector" in losses
    assert "selector_entropy_loss" in losses
    assert selector_grad_norm(detector.frame_selector) > 0.0


def test_truetime_precheck_runner_proves_real_actionformer_selector_gradient() -> None:
    spec = importlib.util.spec_from_file_location("run_truetime_joint_selector_precheck_test", PRECHECK_RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    payload = module.run_precheck(config_path=str(PRECHECK_CONFIG), seed=17)

    assert payload["real_detector_proof_source"] == "opentad_actionformer_forward_train_cost_backward"
    assert payload["real_detector_loss_selector_grad_passed"] is True
    assert payload["real_detector_loss_selector_grad_norm"] > 0.0
    assert payload["actionformer_selected_axis_smoke"] is False
    assert payload["actionformer_physical_grid_precheck"] is True
    assert "cls_loss" in payload["real_detector_loss_keys"]
    assert "reg_loss" in payload["real_detector_loss_keys"]


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


def test_single_stage_post_processing_remaps_selector_predictions_before_nms(monkeypatch) -> None:
    detector = SingleStageDetector()
    predictions = (
        torch.tensor([[[0.0, 2.0], [1.0, 3.0]]]),
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
    seen = {}

    def _capture_batched_nms(segments, scores, labels, **kwargs):
        seen["segments"] = segments.clone()
        return segments, scores, labels

    monkeypatch.setattr(single_stage_module, "batched_nms", _capture_batched_nms)

    detector.post_processing(
        predictions,
        metas,
        SimpleNamespace(sliding_window=False, nms=dict(iou_threshold=0.5, use_soft_nms=False)),
        ["action"],
    )

    assert torch.allclose(seen["segments"], torch.tensor([[10.0, 50.0], [20.0, 100.0]]))
