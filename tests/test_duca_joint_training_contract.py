from __future__ import annotations

import os
import types

import pytest

if os.name == "nt":
    pytest.skip("local Windows torch/c10.dll import is unstable; Linux remote runs this suite", allow_module_level=True)

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"torch is unavailable in this environment: {exc}", allow_module_level=True)

from opentad.models import build_detector
from opentad.models.detectors.actionformer import ActionFormer
from opentad.models.selectors.duca_online_frame_selector import DucaOnlineFrameSelector
from opentad.utils.ema import ModelEma


def _grad_sum(module: nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.requires_grad and param.grad is not None:
            total += float(param.grad.detach().abs().sum().item())
    return total


def _official_asformer_source_cfg() -> dict:
    return {
        "type": "C3CoarseProbeActionnessSource",
        "source_name": "online_c3_official_asformer_coarse_actionness",
        "probe_model": "official-action-seg",
        "official_action_seg_backend": "official_asformer",
        "spatial_size": 16,
        "tcn_hidden_dim": 16,
        "official_num_layers": 1,
        "dropout": 0.0,
        "frozen": False,
        "trainable": True,
        "thumos_trained": False,
        "uses_labels": False,
        "uses_teacher": False,
        "uses_gt": False,
        "uses_prediction_cache": False,
        "calibration_split": "none",
    }


def test_actionformer_optimizer_covers_every_trainable_duca_frame_selector_parameter() -> None:
    selector = DucaOnlineFrameSelector(
        in_channels=3,
        budget=4,
        dense_window_size=8,
        max_radius=2,
        selector_hidden_channels=8,
        actionness_source_cfg=_official_asformer_source_cfg(),
    )
    model = nn.Module()
    model.frame_selector = selector
    model.rpn_head = nn.Linear(1, 1)

    groups = ActionFormer.get_optim_groups(model, {"lr": 1e-4, "weight_decay": 0.05})
    covered = {id(param) for group in groups for param in group["params"]}
    missing = [
        name
        for name, param in model.named_parameters()
        if name.startswith("frame_selector.") and param.requires_grad and id(param) not in covered
    ]

    assert not missing


@pytest.mark.parametrize("forbidden_key", ["total_loss", "detector_utility_distribution_loss"])
def test_actionformer_rejects_selector_aggregate_and_alias_losses(forbidden_key: str) -> None:
    with pytest.raises(ValueError, match="aggregate or alias"):
        ActionFormer._merge_selector_losses({}, {forbidden_key: torch.ones(())})


def test_official_asformer_duca_model_is_ema_deepcopyable() -> None:
    model = build_detector(
        {
            "type": "SingleStageDetector",
            "frame_selector": {
                "type": "DucaOnlineFrameSelector",
                "in_channels": 3,
                "budget": 4,
                "dense_window_size": 8,
                "max_radius": 2,
                "selector_hidden_channels": 8,
                "actionness_source_cfg": _official_asformer_source_cfg(),
            },
            "rpn_head": {"type": "DucaOnlinePrecheckHead", "in_channels": 3},
        }
    )

    probe_state = vars(model.frame_selector.raw_actionness_source.probe)
    module_attrs = [name for name, value in probe_state.items() if isinstance(value, types.ModuleType)]
    assert not module_attrs
    ema = ModelEma(model)

    assert ema.module is not model
    assert ema.module.frame_selector.raw_actionness_source is not model.frame_selector.raw_actionness_source


def test_train_forward_builds_gt_action_target_for_selector_loss() -> None:
    selector = DucaOnlineFrameSelector(
        in_channels=3,
        budget=1,
        dense_window_size=20,
        max_radius=1,
        selector_hidden_channels=4,
        actionness_source_cfg={
            "type": "ZeroShotMotionActionnessSource",
            "source_name": "zero_shot_motion_actionness",
            "mode": "motion",
            "thumos_trained": False,
            "uses_labels": False,
            "uses_teacher": False,
            "uses_gt": False,
            "uses_prediction_cache": False,
            "calibration_split": "none",
            "checkpoint_hash": "no_checkpoint_motion_energy",
        },
        loss_weights={
            "actionness": 1.0,
            "teacher": 0.0,
            "boundary": 1.0,
            "hole": 1.0,
            "redundancy": 0.0,
            "radius": 0.0,
            "entropy": 0.0,
            "budget": 0.0,
        },
    )
    inputs = torch.randn(1, 3, 20)
    masks = torch.ones(1, 20, dtype=torch.bool)
    gt_segments = [torch.tensor([[0.0, 20.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    out = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=[{"video_name": "v"}],
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )

    assert out["selector_outputs"]["action_target"].shape == (1, 20)
    assert out["selector_outputs"]["boundary_target"].shape == (1, 20)
    assert out["selector_outputs"]["boundary_utility_proxy_target"].shape == (1, 20)
    assert out["selector_outputs"]["boundary_utility_proxy_target_kind"] == (
        "instance_normalized_start_end_context_proxy"
    )
    assert out["selector_outputs"]["detector_utility_target_kind"] == "deprecated_alias_to_gt_boundary_utility_proxy"
    assert out["selector_outputs"]["boundary_target"].detach().sum().item() > 0.0
    assert out["selector_outputs"]["boundary_utility_proxy_target"].detach().sum().item() > 0.0
    assert out["losses"]["actionness_bce_loss"].detach().item() > 0.0
    assert out["losses"]["boundary_coverage_loss"].detach().item() >= 0.0
    assert out["losses"]["action_local_hole_loss"].detach().item() > 0.0
    training_provenance = out["selector_outputs"]["training_provenance"]
    assert training_provenance["uses_labels"] is True
    assert training_provenance["uses_gt_segments"] is True
    assert training_provenance["target_kinds"] == [
        "coarse_actionness",
        "start_endpoint",
        "end_endpoint",
        "boundary_context",
    ]
    assert training_provenance["uses_labels_at_inference"] is False


def test_endpoint_targets_are_instance_mass_normalized_across_action_duration() -> None:
    masks = torch.ones(2, 64, dtype=torch.bool)
    segments = [
        torch.tensor([[8.0, 12.0]], dtype=torch.float32),
        torch.tensor([[8.0, 56.0]], dtype=torch.float32),
    ]

    start, end, context = DucaOnlineFrameSelector._endpoint_targets_from_gt_segments(segments, masks)

    assert torch.allclose(start.sum(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(end.sum(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(context.sum(dim=1), torch.ones(2), atol=1e-5)
    assert start[0].sum().item() == pytest.approx(start[1].sum().item(), abs=1e-5)
    assert end[0].sum().item() == pytest.approx(end[1].sum().item(), abs=1e-5)


def test_boundary_utility_proxy_is_endpoint_normalized_not_action_body_mass() -> None:
    masks = torch.ones(1, 64, dtype=torch.bool)
    target = DucaOnlineFrameSelector._boundary_utility_proxy_target_from_gt_segments(
        [torch.tensor([[8.0, 56.0]], dtype=torch.float32)],
        masks,
        boundary_radius=4,
    )

    assert target.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert target[0, 32].item() < 1e-4
    assert target[0, 8].item() > target[0, 32].item()
    assert target[0, 56].item() > target[0, 32].item()


def test_endpoint_and_utility_heads_receive_direct_leaf_loss_gradients() -> None:
    selector = DucaOnlineFrameSelector(
        in_channels=3,
        budget=4,
        dense_window_size=24,
        max_radius=2,
        selector_hidden_channels=8,
        actionness_source_cfg={
            "type": "ZeroShotMotionActionnessSource",
            "source_name": "zero_shot_motion_actionness",
            "mode": "motion",
            "thumos_trained": False,
            "uses_labels": False,
            "uses_teacher": False,
            "uses_gt": False,
            "uses_prediction_cache": False,
            "calibration_split": "none",
            "checkpoint_hash": "no_checkpoint_motion_energy",
        },
        loss_weights={
            "start": 1.0,
            "end": 1.0,
            "context": 0.0,
            "detector_utility": 1.0,
            "actionness": 0.0,
            "teacher": 0.0,
            "boundary": 0.0,
            "hole": 0.0,
            "max_gap_hole": 0.0,
            "redundancy": 0.0,
            "radius": 0.0,
            "entropy": 0.0,
            "budget": 0.0,
        },
    )
    out = selector.forward_train(
        inputs=torch.randn(1, 3, 24),
        masks=torch.ones(1, 24, dtype=torch.bool),
        metas=[{"video_name": "v"}],
        gt_segments=[torch.tensor([[5.0, 18.0]], dtype=torch.float32)],
        gt_labels=[torch.tensor([1], dtype=torch.long)],
    )

    endpoint_losses = (
        out["losses"]["start_endpoint_distribution_loss"]
        + out["losses"]["end_endpoint_distribution_loss"]
        + out["losses"]["boundary_utility_proxy_distribution_loss"]
    )
    endpoint_losses.backward()

    assert _grad_sum(selector.adapter.start_head) > 0.0
    assert _grad_sum(selector.adapter.end_head) > 0.0
    assert _grad_sum(selector.adapter.utility_head) > 0.0


def test_full_window_structured_selector_has_exact_hard_forward_and_detector_gradient() -> None:
    selector = DucaOnlineFrameSelector(
        in_channels=3,
        budget=4,
        dense_window_size=9,
        selector_hidden_channels=8,
        acquisition_policy="global_structured_topk",
        structured_temperature=0.7,
        max_unselected_hole=2,
        hard_max_gap_repair=False,
        detector_gradient_mode="structured_zero_forward",
        actionness_source_cfg={
            "type": "ZeroShotMotionActionnessSource",
            "source_name": "zero_shot_motion_actionness",
            "mode": "motion",
            "thumos_trained": False,
            "uses_labels": False,
            "uses_teacher": False,
            "uses_gt": False,
            "uses_prediction_cache": False,
            "calibration_split": "none",
            "checkpoint_hash": "no_checkpoint_motion_energy",
        },
    )
    inputs = torch.randn(1, 3, 9)
    out = selector.forward_train(
        inputs=inputs,
        masks=torch.ones(1, 9, dtype=torch.bool),
        metas=[{"video_name": "v"}],
        gt_segments=[torch.tensor([[2.0, 7.0]], dtype=torch.float32)],
        gt_labels=[torch.tensor([1], dtype=torch.long)],
    )
    positions = out["selector_outputs"]["grid"].selected_positions[0]
    hard_gather = inputs[:, :, positions]

    assert out["selector_outputs"]["decode_metadata"]["decoder"] == "global_structured_topk"
    assert out["selector_outputs"]["decode_metadata"]["selection_scope"] == "full_window_non_streaming"
    assert torch.equal(out["inputs"].detach(), hard_gather)
    assert positions.numel() == 4
    holes = torch.tensor(
        [
            int(positions[0]),
            *[int(right - left - 1) for left, right in zip(positions[:-1], positions[1:])],
            9 - int(positions[-1]) - 1,
        ]
    )
    assert int(holes.max().item()) <= 2

    out["inputs"].square().mean().backward()
    assert _grad_sum(selector.adapter.center_head) > 0.0


def test_structured_curriculum_uses_stable_hard_input_before_learned_policy() -> None:
    selector = DucaOnlineFrameSelector(
        in_channels=3,
        budget=4,
        dense_window_size=9,
        selector_hidden_channels=8,
        acquisition_policy="global_structured_topk",
        structured_temperature=0.7,
        max_unselected_hole=2,
        hard_max_gap_repair=False,
        detector_gradient_mode="structured_zero_forward",
        loss_weight_schedule={
            "type": "progressive_joint",
            "warmup_steps": 0,
            "transition_steps": 1,
            "detector_gradient": {"start": 0.0, "end": 1.0},
        },
        actionness_source_cfg={
            "type": "ZeroShotMotionActionnessSource",
            "source_name": "zero_shot_motion_actionness",
            "mode": "motion",
            "thumos_trained": False,
            "uses_labels": False,
            "uses_teacher": False,
            "uses_gt": False,
            "uses_prediction_cache": False,
            "calibration_split": "none",
            "checkpoint_hash": "no_checkpoint_motion_energy",
        },
    )
    common = {
        "masks": torch.ones(1, 9, dtype=torch.bool),
        "metas": [{"video_name": "v"}],
        "gt_segments": [torch.tensor([[2.0, 7.0]], dtype=torch.float32)],
        "gt_labels": [torch.tensor([1], dtype=torch.long)],
    }
    first = selector.forward_train(inputs=torch.randn(1, 3, 9), **common)
    second = selector.forward_train(inputs=torch.randn(1, 3, 9) * 100.0, **common)

    assert first["selector_outputs"]["selection_path"] == "stable_structured_reference"
    assert second["selector_outputs"]["selection_path"] == "stable_structured_reference"
    assert torch.equal(
        first["selector_outputs"]["grid"].selected_positions,
        second["selector_outputs"]["grid"].selected_positions,
    )

    selector.after_optimizer_step()
    learned = selector.forward_train(inputs=torch.randn(1, 3, 9), **common)
    assert learned["selector_outputs"]["selection_path"] == "learned_global_structured"


def test_train_forward_builds_detector_utility_distribution_target_without_teacher() -> None:
    selector = DucaOnlineFrameSelector(
        in_channels=3,
        budget=2,
        dense_window_size=20,
        max_radius=2,
        selector_hidden_channels=4,
        actionness_source_cfg={
            "type": "ZeroShotMotionActionnessSource",
            "source_name": "zero_shot_motion_actionness",
            "mode": "motion",
            "thumos_trained": False,
            "uses_labels": False,
            "uses_teacher": False,
            "uses_gt": False,
            "uses_prediction_cache": False,
            "calibration_split": "none",
            "checkpoint_hash": "no_checkpoint_motion_energy",
        },
        loss_weights={
            "actionness": 0.0,
            "detector_utility": 1.0,
            "teacher": 0.0,
            "boundary": 0.0,
            "hole": 0.0,
            "redundancy": 0.0,
            "radius": 0.0,
            "entropy": 0.0,
            "budget": 0.0,
        },
        loss_weight_schedule={
            "type": "progressive_joint",
            "warmup_steps": 0,
            "transition_steps": 1,
            "detector_utility": {"start": 0.0, "end": 1.0},
        },
    )
    inputs = torch.randn(1, 3, 20)
    masks = torch.ones(1, 20, dtype=torch.bool)
    gt_segments = [torch.tensor([[5.0, 12.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    warmup = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=[{"video_name": "v"}],
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )
    selector.after_optimizer_step()
    joint = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=[{"video_name": "v"}],
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )

    target = joint["selector_outputs"]["boundary_utility_proxy_target"]
    assert target.shape == (1, 20)
    assert target.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert target[0, 5].item() > target[0, 9].item()
    assert target[0, 12].item() > target[0, 9].item()
    assert joint["losses"]["boundary_utility_proxy_distribution_loss"].detach().item() > 0.0
    assert warmup["losses"]["boundary_utility_proxy_distribution_loss"].detach().item() == pytest.approx(0.0)
    assert "teacher_utility_gain_loss_unweighted" not in joint["losses"]
    assert joint["selector_outputs"]["loss_weight_schedule"]["weights"]["detector_utility"] == pytest.approx(1.0)


def test_progressive_loss_schedule_starts_with_actionness_and_shifts_to_detector_selection_losses() -> None:
    selector = DucaOnlineFrameSelector(
        in_channels=3,
        budget=1,
        dense_window_size=20,
        max_radius=1,
        selector_hidden_channels=4,
        actionness_source_cfg={
            "type": "ZeroShotMotionActionnessSource",
            "source_name": "zero_shot_motion_actionness",
            "mode": "motion",
            "thumos_trained": False,
            "uses_labels": False,
            "uses_teacher": False,
            "uses_gt": False,
            "uses_prediction_cache": False,
            "calibration_split": "none",
            "checkpoint_hash": "no_checkpoint_motion_energy",
        },
        loss_weights={
            "actionness": 1.0,
            "detector": 1.0,
            "hole": 1.0,
            "budget": 1.0,
            "entropy": 1.0,
            "teacher": 0.0,
            "boundary": 0.0,
            "redundancy": 0.0,
            "radius": 0.0,
        },
        loss_weight_schedule={
            "type": "progressive_joint",
            "warmup_steps": 0,
            "transition_steps": 1,
            "actionness": {"start": 1.0, "end": 0.25},
            "detector_gradient": {"start": 0.0, "end": 1.0},
            "hole": {"start": 0.0, "end": 1.0},
            "budget": {"start": 0.0, "end": 1.0},
            "entropy": {"start": 0.0, "end": 0.2},
        },
    )
    inputs = torch.randn(1, 3, 20)
    masks = torch.ones(1, 20, dtype=torch.bool)
    gt_segments = [torch.tensor([[0.0, 20.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    first = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=[{"video_name": "v"}],
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )
    schedule_step = selector.after_optimizer_step()
    second = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=[{"video_name": "v"}],
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )

    first_schedule = first["selector_outputs"]["loss_weight_schedule"]
    second_schedule = second["selector_outputs"]["loss_weight_schedule"]

    assert first_schedule["step"] == 0
    assert first_schedule["phase"] == "coarse_actionness_warmup"
    assert first_schedule["weights"]["actionness"] == pytest.approx(1.0)
    assert first_schedule["detector_gradient_weight"] == pytest.approx(0.0)
    assert first_schedule["weights"]["hole"] == pytest.approx(0.0)
    assert schedule_step["updated"] is True
    assert schedule_step["source"] == "optimizer_step"
    assert schedule_step["step_before"] == 0
    assert schedule_step["step_after"] == 1
    assert second_schedule["step"] == 1
    assert second_schedule["phase"] == "joint_detection_selection"
    assert second_schedule["weights"]["actionness"] == pytest.approx(0.25)
    assert second_schedule["detector_gradient_weight"] == pytest.approx(1.0)
    assert second_schedule["weights"]["hole"] == pytest.approx(1.0)


def test_detector_loss_is_not_muted_while_selector_gradient_bridge_warms_up() -> None:
    model = build_detector(
        {
            "type": "SingleStageDetector",
            "frame_selector": {
                "type": "DucaOnlineFrameSelector",
                "in_channels": 3,
                "budget": 2,
                "dense_window_size": 8,
                "max_radius": 1,
                "selector_hidden_channels": 4,
                "detector_gradient_mode": "st_sparse_gather_soft_context",
                "actionness_source_cfg": {
                    "type": "ZeroShotMotionActionnessSource",
                    "source_name": "zero_shot_motion_actionness",
                    "mode": "motion",
                    "thumos_trained": False,
                    "uses_labels": False,
                    "uses_teacher": False,
                    "uses_gt": False,
                    "uses_prediction_cache": False,
                    "calibration_split": "none",
                    "checkpoint_hash": "no_checkpoint_motion_energy",
                },
                "loss_weights": {
                    "actionness": 1.0,
                    "detector": 1.0,
                    "hole": 0.0,
                    "teacher": 0.0,
                    "boundary": 0.0,
                    "redundancy": 0.0,
                    "radius": 0.0,
                    "entropy": 0.0,
                    "budget": 0.0,
                },
                "loss_weight_schedule": {
                    "type": "progressive_joint",
                    "warmup_steps": 0,
                    "transition_steps": 1,
                    "actionness": {"start": 1.0, "end": 0.25},
                    "detector_gradient": {"start": 0.0, "end": 1.0},
                },
            },
            "rpn_head": {"type": "DucaOnlinePrecheckHead", "in_channels": 3},
        }
    )
    losses = model(
        torch.randn(1, 3, 8, 8, 8),
        torch.ones(1, 8, dtype=torch.bool),
        [{"video_name": "v"}],
        gt_segments=[torch.tensor([[1.0, 6.0]], dtype=torch.float32)],
        gt_labels=[torch.tensor([1], dtype=torch.long)],
        return_loss=True,
    )

    assert losses["loss_detector"].detach().item() > 0.0
    assert losses["selector_actionness_bce_loss"].detach().item() > 0.0


def test_detector_gradient_bridge_is_scheduled_independently_from_detector_loss() -> None:
    torch.manual_seed(7)
    selector = DucaOnlineFrameSelector(
        in_channels=3,
        budget=2,
        dense_window_size=8,
        max_radius=1,
        selector_hidden_channels=8,
        detector_gradient_mode="st_sparse_gather_soft_context",
        actionness_source_cfg={
            "type": "ZeroShotMotionActionnessSource",
            "source_name": "zero_shot_motion_actionness",
            "mode": "motion",
            "thumos_trained": False,
            "uses_labels": False,
            "uses_teacher": False,
            "uses_gt": False,
            "uses_prediction_cache": False,
            "calibration_split": "none",
            "checkpoint_hash": "no_checkpoint_motion_energy",
        },
        loss_weights={
            "actionness": 0.0,
            "detector": 1.0,
            "hole": 0.0,
            "teacher": 0.0,
            "boundary": 0.0,
            "redundancy": 0.0,
            "radius": 0.0,
            "entropy": 0.0,
            "budget": 0.0,
        },
        loss_weight_schedule={
            "type": "progressive_joint",
            "warmup_steps": 0,
            "transition_steps": 1,
            "detector_gradient": {"start": 0.0, "end": 1.0},
        },
    )
    selector.train()
    inputs = torch.randn(1, 3, 8)
    masks = torch.ones(1, 8, dtype=torch.bool)
    gt_segments = [torch.tensor([[1.0, 6.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    warmup = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=[{"video_name": "v"}],
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )
    warmup["inputs"].pow(2).mean().backward()
    warmup_center_grad = _grad_sum(selector.adapter.center_head)
    schedule_step = selector.after_optimizer_step()

    selector.zero_grad(set_to_none=True)
    joint = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=[{"video_name": "v"}],
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )
    joint["inputs"].pow(2).mean().backward()
    joint_center_grad = _grad_sum(selector.adapter.center_head)

    assert warmup["selector_outputs"]["loss_weight_schedule"]["detector_gradient_weight"] == pytest.approx(0.0)
    assert schedule_step["updated"] is True
    assert schedule_step["step_after"] == 1
    assert joint["selector_outputs"]["loss_weight_schedule"]["detector_gradient_weight"] == pytest.approx(1.0)
    assert warmup_center_grad == pytest.approx(0.0)
    assert joint_center_grad > 0.0


def test_progressive_loss_schedule_does_not_advance_without_optimizer_step() -> None:
    selector = DucaOnlineFrameSelector(
        in_channels=3,
        budget=1,
        dense_window_size=8,
        max_radius=1,
        selector_hidden_channels=4,
        actionness_source_cfg={
            "type": "ZeroShotMotionActionnessSource",
            "source_name": "zero_shot_motion_actionness",
            "mode": "motion",
            "thumos_trained": False,
            "uses_labels": False,
            "uses_teacher": False,
            "uses_gt": False,
            "uses_prediction_cache": False,
            "calibration_split": "none",
            "checkpoint_hash": "no_checkpoint_motion_energy",
        },
        loss_weight_schedule={
            "type": "progressive_joint",
            "warmup_steps": 0,
            "transition_steps": 1,
            "detector_gradient": {"start": 0.0, "end": 1.0},
        },
    )
    inputs = torch.randn(1, 3, 8)
    masks = torch.ones(1, 8, dtype=torch.bool)
    gt_segments = [torch.tensor([[1.0, 6.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    first = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=[{"video_name": "v"}],
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )
    second = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=[{"video_name": "v"}],
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )

    assert first["selector_outputs"]["loss_weight_schedule"]["step"] == 0
    assert second["selector_outputs"]["loss_weight_schedule"]["step"] == 0


def test_detector_cost_backpropagates_to_official_asformer_probe_selector_and_budget_controller() -> None:
    torch.manual_seed(11)
    model = build_detector(
        {
            "type": "SingleStageDetector",
            "frame_selector": {
                "type": "DucaOnlineFrameSelector",
                "in_channels": 3,
                "budget": None,
                "budget_mode": "dynamic_must",
                "budget_min": 2,
                "budget_max": 4,
                "budget_multiple": 1,
                "target_budget": 3,
                "allow_external_budget_override": False,
                "dense_window_size": 8,
                "max_radius": 2,
                "selector_hidden_channels": 8,
                "detector_gradient_mode": "st_sparse_gather_soft_context",
                "actionness_source_cfg": _official_asformer_source_cfg(),
                "loss_weights": {
                    "actionness": 0.5,
                    "lagrangian_budget": 1.0,
                    "marginal_monotonic": 0.01,
                    "hole": 0.1,
                    "teacher": 0.0,
                    "boundary": 0.0,
                    "redundancy": 0.0,
                    "radius": 0.0,
                    "entropy": 0.0,
                    "budget": 0.0,
                },
            },
            "rpn_head": {"type": "DucaOnlinePrecheckHead", "in_channels": 3},
        }
    )
    model.train()
    inputs = torch.randn(1, 3, 8, 16, 16)
    masks = torch.ones(1, 8, dtype=torch.bool)
    gt_segments = [torch.tensor([[1.0, 6.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    losses = model(
        inputs,
        masks,
        [{"video_name": "v"}],
        gt_segments=gt_segments,
        gt_labels=gt_labels,
        return_loss=True,
    )
    losses["cost"].backward()

    assert losses["selector_action_local_hole_loss"].detach().item() >= 0.0
    assert _grad_sum(model.frame_selector.raw_actionness_source) > 0.0
    assert _grad_sum(model.frame_selector.adapter.center_head) > 0.0
    assert _grad_sum(model.frame_selector.adapter.budget_controller) > 0.0
