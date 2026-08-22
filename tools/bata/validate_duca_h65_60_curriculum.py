"""Read-only admission checks for the schedule-only H65 20+20+20 course."""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path
from typing import Any

import torch
from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    "STAGE1_20": ROOT / "configs/adatad/thumos/duca_h65_60_stage1_uniform20.py",
    "STAGE2_40": ROOT
    / "configs/adatad/thumos/duca_h65_60_stage2_transition20_joint20.py",
}
HISTORICAL = {
    "STAGE1_20": ROOT
    / "configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py",
    "STAGE2_40": ROOT
    / "configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py",
}


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    return value


def _model_without_course(cfg: Config) -> dict[str, Any]:
    model = copy.deepcopy(_plain(cfg.model))
    model["frame_selector"].pop("loss_weight_schedule", None)
    return model


def _load_configs(target: str, checkpoint: Path | None, digest: str | None) -> tuple[Config, Config]:
    if target == "STAGE2_40":
        if checkpoint is None or digest is None:
            raise SystemExit("STAGE2_40 requires Stage-1 checkpoint and SHA256")
        os.environ.update(
            DUCA_STAGE1_CHECKPOINT=str(checkpoint.resolve()),
            DUCA_STAGE1_CHECKPOINT_SHA256=digest.lower(),
            DUCA_STAGE1_CHECKPOINT_EPOCH="19",
        )
    return Config.fromfile(str(CONFIGS[target])), Config.fromfile(str(HISTORICAL[target]))


def _validate_checkpoint(checkpoint: Path, expected_sha256: str) -> None:
    actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual != expected_sha256.lower():
        raise SystemExit("Stage-1 checkpoint SHA256 mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("epoch") != 19:
        raise SystemExit("Stage-1 checkpoint must be terminal epoch 19")
    if not isinstance(state.get("state_dict_ema"), dict):
        raise SystemExit("Stage-1 checkpoint lacks state_dict_ema")


def _validate_contract(target: str, cfg: Config, historical: Config) -> None:
    if target == "STAGE1_20":
        if _plain(cfg.model) != _plain(historical.model):
            raise SystemExit("Stage-1 changed H65 model state outside the course")
        expected = (20, 2000, 19, 20)
    else:
        if _model_without_course(cfg) != _model_without_course(historical):
            raise SystemExit("Stage-2 changed H65 model state outside loss_weight_schedule")
        if _plain(cfg.optimizer) != _plain(historical.optimizer):
            raise SystemExit("Stage-2 changed optimizer parameter groups")
        schedule = cfg.model.frame_selector.loss_weight_schedule
        for name in ("actionness", "transition", "transition_boundary", "policy_alpha", "asformer_adapt"):
            if schedule[name].warmup_steps != 0 or schedule[name].transition_steps != 2000:
                raise SystemExit(f"invalid 20-epoch cosine schedule for {name}")
        for name in ("detector_gradient", "detector_contribution"):
            if (schedule[name].warmup_steps, schedule[name].transition_steps) != (667, 1333):
                raise SystemExit(f"invalid scaled feedback schedule for {name}")
        init = cfg.workflow.model_initialization
        if init.state_key != "state_dict_ema" or init.expected_checkpoint_epoch != 19:
            raise SystemExit("Stage-2 initialization is not bound to epoch-19 EMA")
        if init.reset_state_keys != ["frame_selector._loss_weight_schedule_step"]:
            raise SystemExit("Stage-2 may reset only the curriculum clock")
        expected = (40, 4000, 39, 40)

    observed = (
        cfg.workflow.end_epoch,
        cfg.workflow.expected_successful_optimizer_updates,
        cfg.workflow.primary_checkpoint_epoch,
        cfg.scheduler.max_epoch,
    )
    if observed != expected:
        raise SystemExit(f"training duration contract mismatch: {observed} != {expected}")
    if cfg.workflow.checkpoint_interval != 5:
        raise SystemExit("checkpoint interval must remain five epochs")
    if cfg.workflow.intermediate_validation_selects_checkpoint:
        raise SystemExit("intermediate validation must not select checkpoints")
    if cfg.seed != 3407:
        raise SystemExit("seed must remain 3407")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=CONFIGS, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--category", type=Path, required=True)
    parser.add_argument("--pretrain", type=Path, required=True)
    parser.add_argument("--stage1", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()

    for path in (args.annotation, args.category, args.pretrain):
        if not path.is_file():
            raise SystemExit(f"canonical resource unreadable: {path}")
    entries = list(args.video_root.iterdir())
    if len(entries) != 411 or any(not entry.is_symlink() or not entry.exists() for entry in entries):
        raise SystemExit("canonical video root must contain 411 valid symlinks")
    if args.target == "STAGE2_40":
        if args.stage1 is None or args.sha256 is None:
            raise SystemExit("STAGE2_40 requires Stage-1 checkpoint and SHA256")
        _validate_checkpoint(args.stage1, args.sha256)

    cfg, historical = _load_configs(args.target, args.stage1, args.sha256)
    _validate_contract(args.target, cfg, historical)
    print(f"PASS H65-60 {args.target}: schedule-only model identity preserved")


if __name__ == "__main__":
    main()

