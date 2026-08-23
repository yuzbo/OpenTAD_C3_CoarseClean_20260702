from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from mmengine.config import Config
import torch

from tools.bata.duca_p0_training import atomic_write_json


REQUIRED_CHECKPOINT_KEYS = (
    "state_dict",
    "state_dict_ema",
    "optimizer",
    "scheduler",
    "grad_scaler",
    "rng_state",
    "data_loader_state",
    "successful_optimizer_updates",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clock_values(state: Mapping[str, Any]) -> dict[str, float]:
    matches = {
        str(key): value
        for key, value in state.items()
        if str(key).endswith("relative_physical_time_scale")
    }
    values = {}
    for key, value in matches.items():
        _require(torch.is_tensor(value) and value.numel() == 1, f"invalid SingleClock scalar: {key}")
        scalar = float(value.detach().cpu().item())
        _require(torch.isfinite(value).all().item(), f"non-finite SingleClock scalar: {key}")
        values[key] = scalar
    return values


def audit_terminal_checkpoint(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path,
    stage1_path: str | Path,
    stage1_sha256: str,
    family: str,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    stage1_path = Path(stage1_path).expanduser().resolve()
    _require(checkpoint_path.is_file(), f"terminal checkpoint missing: {checkpoint_path}")
    _require(config_path.is_file(), f"source config missing: {config_path}")
    _require(stage1_path.is_file(), f"Stage-1 checkpoint missing: {stage1_path}")
    stage1_sha256 = str(stage1_sha256).lower()
    _require(_sha256(stage1_path) == stage1_sha256, "Stage-1 checkpoint SHA256 mismatch")
    family = str(family)
    _require(family in {"clock_on", "h65_off"}, "family must be clock_on or h65_off")

    stage1 = torch.load(stage1_path, map_location="cpu", weights_only=False)
    _require(
        isinstance(stage1, Mapping)
        and int(stage1.get("epoch", -1)) == 29
        and isinstance(stage1.get("state_dict_ema"), Mapping),
        "Stage-1 handoff must be epoch-29 EMA capable",
    )
    del stage1

    os.environ.update(
        DUCA_STAGE1_CHECKPOINT=str(stage1_path),
        DUCA_STAGE1_CHECKPOINT_SHA256=stage1_sha256,
        DUCA_STAGE1_CHECKPOINT_EPOCH="29",
    )
    cfg = Config.fromfile(str(config_path))
    _require(
        (int(cfg.seed), int(cfg.total_epochs), int(cfg.max_updates)) == (3407, 60, 6000),
        "terminal config must freeze seed3407, 60 epochs and 6000 updates",
    )
    _require(int(cfg.workflow.checkpoint_interval) == 5, "checkpoint interval must be 5 epochs")
    _require(
        str(cfg.duca_stage1_checkpoint) == str(stage1_path)
        and str(cfg.duca_stage1_checkpoint_sha256).lower() == stage1_sha256
        and int(cfg.duca_stage1_checkpoint_epoch) == 29,
        "resolved config does not bind the exact Stage-1 handoff",
    )
    admission = bool(cfg.model.get("single_clock_admission", False))
    _require(admission == (family == "clock_on"), "config SingleClock admission differs from family")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "terminal checkpoint is not a mapping")
    _require(int(checkpoint.get("epoch", -1)) == 59, "terminal checkpoint must be epoch 59")
    for key in REQUIRED_CHECKPOINT_KEYS:
        _require(key in checkpoint, f"terminal checkpoint is missing {key}")
    _require(
        int(checkpoint["successful_optimizer_updates"]) == 6000,
        "terminal checkpoint must contain exactly 6000 successful optimizer updates",
    )
    scheduler = checkpoint["scheduler"]
    _require(
        isinstance(scheduler, Mapping) and int(scheduler.get("last_epoch", -1)) == 6000,
        "terminal scheduler state must end at update 6000",
    )
    _require(isinstance(checkpoint["optimizer"], Mapping), "optimizer recovery state is invalid")
    _require(isinstance(checkpoint["grad_scaler"], Mapping), "AMP scaler recovery state is invalid")
    rng = checkpoint["rng_state"]
    _require(
        isinstance(rng, Mapping)
        and set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "terminal RNG recovery state is incomplete",
    )
    loader = checkpoint["data_loader_state"]
    _require(
        isinstance(loader, Mapping)
        and int(loader.get("completed_epoch", -1)) == 59
        and int(loader.get("next_epoch", -1)) == 60,
        "terminal DataLoader recovery state does not end at epoch 59",
    )
    clock = {}
    for state_key in ("state_dict", "state_dict_ema"):
        state = checkpoint[state_key]
        _require(isinstance(state, Mapping), f"{state_key} is invalid")
        values = _clock_values(state)
        if family == "clock_on":
            _require(len(values) == 1, f"{state_key} must contain exactly one SingleClock scalar")
            _require(next(iter(values.values())) != 0.0, f"{state_key} SingleClock scalar was not updated")
        else:
            _require(not values, f"{state_key} H65 OFF unexpectedly contains a SingleClock scalar")
        clock[state_key] = values

    payload = {
        "schema_version": "duca_h65_terminal_checkpoint_audit_v1",
        "family": family,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": 59,
        "successful_optimizer_updates": 6000,
        "scheduler_last_epoch": 6000,
        "state_dict_present": True,
        "state_dict_ema_present": True,
        "optimizer_present": True,
        "scheduler_present": True,
        "grad_scaler_present": True,
        "rng_state_complete": True,
        "data_loader_next_epoch": 60,
        "stage1_checkpoint_path": str(stage1_path),
        "stage1_checkpoint_sha256": stage1_sha256,
        "stage1_checkpoint_epoch": 29,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "single_clock_values": clock,
    }
    del checkpoint
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Audit a terminal H65/SingleClock checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--stage1-sha256", required=True)
    parser.add_argument("--family", choices=("clock_on", "h65_off"), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = audit_terminal_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        stage1_path=args.stage1,
        stage1_sha256=args.stage1_sha256,
        family=args.family,
    )
    atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
