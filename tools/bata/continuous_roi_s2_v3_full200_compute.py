from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from mmengine.config import Config


PROTOCOL_ID = "ZOOMTOKEN-CONTINUOUS-ROI-S2-V3-FULL200-COMPUTE-PARETO-3X3-v001"
ARMS = ("D160", "G96", "U128-A0")
SEEDS = (4407, 4408, 4409)
EXPECTED_TRAINING_IDENTITIES = 200
EXPECTED_EVALUATION_VIDEOS = 211
EXPECTED_EVALUATION_WINDOWS = 792
EXPECTED_UPDATES_PER_EPOCH = 100
EXPECTED_EPOCHS = 60
EXPECTED_TOTAL_UPDATES = 6000
EXPECTED_WORLD_SIZE = 2


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_publish_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def require_clean_commit(expected_commit: str, root: str | Path) -> None:
    root = Path(root).resolve()
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if actual != expected_commit:
        raise RuntimeError(f"candidate commit mismatch: {actual} != {expected_commit}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip()
    if status:
        raise RuntimeError("formal execution requires a clean candidate checkout")


def config_path(root: str | Path, arm: str, seed: int) -> Path:
    names = {
        "D160": "d160",
        "G96": "g96",
        "U128-A0": "u128_a0",
    }
    if arm not in names or seed not in SEEDS:
        raise ValueError(f"unsupported cell arm={arm!r} seed={seed!r}")
    return (
        Path(root)
        / "configs"
        / "adatad"
        / "thumos"
        / f"continuous_roi_s2_v3_{names[arm]}_seed{seed}.py"
    )


def _pipeline_step(cfg: Config, split: str, type_name: str) -> Any:
    for step in cfg.dataset[split].pipeline:
        if step.type == type_name:
            return step
    raise ValueError(f"{split} pipeline is missing {type_name}")


def validate_cell_config(path: str | Path, *, arm: str, seed: int) -> dict[str, Any]:
    path = Path(path)
    cfg = Config.fromfile(path)
    binding = cfg.continuous_roi_s2_v3_full200_compute
    if binding.protocol != PROTOCOL_ID or binding.arm != arm or int(binding.seed) != seed:
        raise ValueError("cell protocol, arm, or seed binding changed")
    if cfg.dataset.train.subset_name != "training":
        raise ValueError("formal training must use the complete training split")
    for split in ("val", "test"):
        dataset = cfg.dataset[split]
        if (
            dataset.subset_name != "validation"
            or int(dataset.window_size) != 768
            or float(dataset.window_overlap_ratio) != 0.5
            or not bool(dataset.test_mode)
        ):
            raise ValueError("formal evaluation population geometry changed")
    if (
        int(cfg.solver.train.batch_size) != 1
        or not bool(cfg.solver.ema)
        or int(cfg.workflow.max_train_iters) != EXPECTED_UPDATES_PER_EPOCH
        or int(cfg.workflow.end_epoch) != EXPECTED_EPOCHS
        or str(cfg.workflow.checkpoint_policy) != "final_only"
        or int(cfg.inference.test_epoch) != 59
    ):
        raise ValueError("formal schedule or final-EMA rule changed")
    if not bool(cfg.workflow.schedule_and_ema_on_success_only):
        raise ValueError("scheduler and EMA must advance only on successful updates")
    if not bool(cfg.workflow.fail_on_skipped_update):
        raise ValueError("a cell must fail rather than silently lose an update")

    if arm == "D160":
        step = _pipeline_step(cfg, "train", "FullFrameLetterboxView")
        expected = 160
    elif arm == "G96":
        step = _pipeline_step(cfg, "train", "FullFrameLetterboxView")
        expected = 96
    else:
        step = _pipeline_step(cfg, "train", "NativeCropSourceViews")
        if (
            int(step.global_size) != 96
            or int(step.local_size) != 128
            or bool(step.allow_local_padding)
        ):
            raise ValueError("U128-A0 fixed source-native view changed")
        custom = cfg.model.backbone.custom
        if (
            custom.wrapper_type != "native_crop_shared_videomae"
            or custom.native_crop_fusion_mode != "fixed_mean"
        ):
            raise ValueError("U128-A0 must use one shared backbone and parameter-free fusion")
        expected = None
    if expected is not None and int(step.output_size) != expected:
        raise ValueError(f"{arm} spatial resolution changed")

    forbidden = ("fit160", "gate40", "129-window", "sobol")
    lowered = path.read_text(encoding="utf-8").lower()
    if any(token in lowered for token in forbidden):
        raise ValueError("a superseded partial-data route entered the formal config")
    return {
        "arm": arm,
        "seed": seed,
        "config": str(path.resolve()),
        "work_dir": str(cfg.work_dir),
        "final_checkpoint": "checkpoint/epoch_59.pth:state_dict_ema",
    }


def validate_matrix(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    cells = [
        validate_cell_config(config_path(root, arm, seed), arm=arm, seed=seed)
        for arm in ARMS
        for seed in SEEDS
    ]
    return {
        "protocol_id": PROTOCOL_ID,
        "cells": cells,
        "cell_count": len(cells),
        "training_identities": EXPECTED_TRAINING_IDENTITIES,
        "evaluation_videos": EXPECTED_EVALUATION_VIDEOS,
        "evaluation_ordered_windows": EXPECTED_EVALUATION_WINDOWS,
        "successful_updates_per_cell": EXPECTED_TOTAL_UPDATES,
        "world_size": EXPECTED_WORLD_SIZE,
    }


def parameter_surface(cfg: Config) -> dict[str, Any]:
    """Return the parameter-bearing config after removing U128-A0 wrapper metadata."""

    model = copy.deepcopy(cfg.model.to_dict())
    custom = model["backbone"]["custom"]
    for key in list(custom):
        if key == "wrapper_type" or key.startswith("native_crop_"):
            custom.pop(key)
    optimizer = copy.deepcopy(cfg.optimizer.to_dict())
    return {"model": model, "optimizer": optimizer}


def validate_parameter_fairness(root: str | Path) -> None:
    root = Path(root)
    reference = Config.fromfile(config_path(root, "D160", 4407))
    expected = parameter_surface(reference)
    for arm in ARMS:
        cfg = Config.fromfile(config_path(root, arm, 4407))
        if parameter_surface(cfg) != expected:
            raise ValueError(f"{arm} changed the trainable parameter surface")
    u128 = Config.fromfile(config_path(root, "U128-A0", 4407))
    if u128.model.backbone.custom.native_crop_fusion_mode != "fixed_mean":
        raise ValueError("U128-A0 fusion is not parameter-free fixed_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the full-data S2-v3 matrix")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = validate_matrix(args.root)
    validate_parameter_fairness(args.root)
    receipt["parameter_fairness"] = "PASS"
    if args.output is not None:
        atomic_publish_json(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
