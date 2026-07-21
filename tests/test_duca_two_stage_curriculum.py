from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"torch is unavailable in this environment: {exc}", allow_module_level=True)

from mmengine.config import Config

from opentad.models.detectors.actionformer import ActionFormer
from tools.bata.create_duca_frontend_split import create_split
from tools.bata.duca_frontend_initialization import (
    initialize_frame_selector_from_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"


class _FrontendOnlySelector(nn.Module):
    selector_variant = "transition_only"
    require_counterfactual_utility_teacher = False
    separate_detector_rng = False

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.last_forward_summary = {}

    def forward_train(self, *, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        del kwargs
        action = self.weight.square()
        transition = (self.weight - 1.0).square()
        return {
            "inputs": inputs,
            "masks": masks,
            "metas": metas,
            "gt_segments": gt_segments,
            "gt_labels": gt_labels,
            "losses": {
                "actionness_bce_loss": action,
                "transition_distribution_loss": transition,
            },
            "selector_outputs": {},
            "counterfactual_request": None,
        }


class _ExplodingDetector(nn.Module):
    def forward(self, *_args, **_kwargs):  # pragma: no cover - failure path only.
        raise AssertionError("detector path must not run during frontend-only training")


def _bare_frontend_only_actionformer() -> ActionFormer:
    model = ActionFormer.__new__(ActionFormer)
    nn.Module.__init__(model)
    model.frame_selector = _FrontendOnlySelector()
    model.backbone = _ExplodingDetector()
    model.projection = _ExplodingDetector()
    model.neck = _ExplodingDetector()
    model.rpn_head = _ExplodingDetector()
    model.token_compressor = None
    model.pc_ot_mras_reader = None
    model.selector_train_only = True
    model.selector_train_only_skip_detector = True
    return model


def test_frontend_only_actionformer_skips_the_complete_detector_path() -> None:
    model = _bare_frontend_only_actionformer()
    loss = model.forward_train(
        torch.randn(1, 3, 8),
        torch.ones(1, 8, dtype=torch.bool),
        [{"video_name": "train"}],
        [torch.tensor([[1.0, 4.0]])],
        [torch.tensor([0])],
    )
    assert set(loss) == {
        "actionness_bce_loss",
        "transition_distribution_loss",
        "cost",
    }
    assert torch.equal(
        loss["cost"],
        loss["actionness_bce_loss"] + loss["transition_distribution_loss"],
    )
    loss["cost"].backward()
    assert model.frame_selector.weight.grad is not None
    assert model.frame_selector.last_forward_summary[
        "frontend_only_detector_skipped"
    ] is True


class _TinySelector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.register_buffer(
            "_loss_weight_schedule_step", torch.zeros((), dtype=torch.long)
        )


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frame_selector = _TinySelector()
        self.detector = nn.Linear(2, 1)


def test_selector_initialization_is_strict_hash_bound_and_resets_schedule(
    tmp_path: Path,
) -> None:
    source = _TinyModel()
    source.frame_selector.linear.weight.data.fill_(3.0)
    source.frame_selector.linear.bias.data.fill_(4.0)
    source.frame_selector._loss_weight_schedule_step.fill_(1600)
    state = {
        f"module.frame_selector.{key}": value.detach().clone()
        for key, value in source.frame_selector.state_dict().items()
    }
    state["module.detector.weight"] = source.detector.weight.detach().clone()
    checkpoint = tmp_path / "frontend.pth"
    torch.save({"epoch": 19, "state_dict_ema": state}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    target = _TinyModel()
    detector_before = target.detector.weight.detach().clone()
    receipt = initialize_frame_selector_from_checkpoint(
        target,
        {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": digest,
            "state_key": "state_dict_ema",
            "expected_checkpoint_epoch": 19,
            "reset_state_keys": ["_loss_weight_schedule_step"],
        },
    )
    assert receipt is not None
    assert receipt["detector_state_loaded"] is False
    assert torch.equal(
        target.frame_selector.linear.weight,
        source.frame_selector.linear.weight,
    )
    assert int(target.frame_selector._loss_weight_schedule_step.item()) == 0
    assert torch.equal(target.detector.weight, detector_before)

    with pytest.raises(RuntimeError, match="SHA256"):
        initialize_frame_selector_from_checkpoint(
            _TinyModel(),
            {
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": "0" * 64,
                "state_key": "state_dict_ema",
            },
        )


def test_train_only_frontend_split_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    annotation = tmp_path / "anno.json"
    database = {
        **{f"train_{idx}": {"subset": "training"} for idx in range(10)},
        "test_0": {"subset": "validation"},
    }
    annotation.write_text(json.dumps({"database": database}), encoding="utf-8")
    first = create_split(annotation, tmp_path / "split", seed=3407)
    second = create_split(annotation, tmp_path / "split", seed=3407)
    assert first == second
    assert set(first["train_videos"]).isdisjoint(first["holdout_videos"])
    assert set(first["train_videos"]) | set(first["holdout_videos"]) == {
        f"train_{idx}" for idx in range(10)
    }
    assert first["test_subset_consumed"] is False


def test_two_stage_configs_freeze_only_the_declared_coarse_branch(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    monkeypatch.setenv("DUCA_FRONTEND_TRAIN_BLOCK_LIST", "train_block.txt")
    monkeypatch.setenv("DUCA_FRONTEND_HOLDOUT_BLOCK_LIST", "holdout_block.txt")
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT", "frontend.pth")
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_EPOCH", "19")

    frontend = Config.fromfile(
        str(
            CONFIG_DIR
            / "duca_frontend_pretrain_lr_coarse50_action100_scorer25.py"
        )
    )
    assert frontend.model.selector_train_only is True
    assert frontend.model.selector_train_only_skip_detector is True
    assert frontend.workflow.end_epoch == 20
    assert frontend.workflow.checkpoint_interval == 5
    assert frontend.dataset.val is None and frontend.dataset.test is None
    assert frontend.optimizer.type == "AdamW"
    assert frontend.optimizer.paramwise is True
    assert "backbone" not in frontend.optimizer
    assert frontend.solver.clip_grad_norm <= 0.0
    assert frontend.model.frame_selector.strict_loss_contract is True
    assert frontend.model.frame_selector.actionness_loss_mode == "class_balanced_mean"
    assert frontend.model.frame_selector.auxiliary_hidden_gradient_scale == 0.0
    assert frontend.model.frame_selector.actionness_source_cfg.spatial_norm == "groupnorm"
    assert frontend.model.frame_selector.coarse_trunk_lr == 5.0e-5
    assert frontend.model.frame_selector.action_head_lr == 1.0e-4
    assert frontend.model.frame_selector.transition_scorer_lr == 2.5e-5
    assert set(frontend.model.frame_selector.loss_weights) == {
        "detector",
        "actionness",
        "budget",
        "boundary",
        "hole",
        "max_gap_hole",
        "redundancy",
        "radius",
        "entropy",
        "teacher",
        "detector_utility",
        "start",
        "end",
        "context",
        "lagrangian_budget",
        "marginal_monotonic",
        "hard_budget_cap",
        "transition",
        "transition_boundary",
    }
    assert frontend.model.frame_selector.loss_weights.detector == 0.0
    assert frontend.model.frame_selector.loss_weights.actionness == 1.0
    assert frontend.model.frame_selector.loss_weight_schedule.transition.end == 0.10
    assert (
        frontend.model.frame_selector.loss_weight_schedule.transition_boundary.end
        == 16.0
    )

    joint = Config.fromfile(
        str(CONFIG_DIR / "duca_two_stage_pretrained_joint_fixed384_official60.py")
    )
    frozen = Config.fromfile(
        str(CONFIG_DIR / "duca_two_stage_pretrained_frozen_fixed384_official60.py")
    )
    for cfg in (joint, frozen):
        schedule = cfg.model.frame_selector.loss_weight_schedule
        assert schedule.policy_alpha.warmup_steps == 1000
        assert schedule.policy_alpha.transition_steps == 1500
        assert schedule.detector_gradient.warmup_steps == 2500
        assert schedule.detector_gradient.transition_steps == 1500
        assert schedule.actionness.start == 0.0
        assert schedule.transition.start == 0.0
        assert schedule.transition_boundary.start == 0.0
        assert cfg.workflow.expected_successful_optimizer_updates == 6000
        assert cfg.workflow.selector_initialization.state_key == "state_dict_ema"
    assert joint.model.frame_selector.actionness_source_cfg.frozen is False
    assert joint.model.frame_selector.actionness_source_cfg.trainable is True
    assert frozen.model.frame_selector.actionness_source_cfg.frozen is True
    assert frozen.model.frame_selector.actionness_source_cfg.trainable is False
    assert frozen.model.frame_selector.allow_frozen_coarse_probe is True
    assert frozen.model.frame_selector.loss_weight_schedule.actionness.end == 0.0
