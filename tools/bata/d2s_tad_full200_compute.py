from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from mmengine.config import Config

from tools.bata.continuous_roi_s2_v3_full200_compute import (
    EXPECTED_EPOCHS,
    EXPECTED_EVALUATION_VIDEOS,
    EXPECTED_EVALUATION_WINDOWS,
    EXPECTED_TOTAL_UPDATES,
    EXPECTED_TRAINING_IDENTITIES,
    EXPECTED_UPDATES_PER_EPOCH,
    EXPECTED_WORLD_SIZE,
    atomic_publish_json,
    build_full_data_bundle,
    canonical_sha256,
)


D2S_PROTOCOL_ID = "ZOOMTOKEN-D2S-TAD-FULL200-COMPUTE-PARETO-3X3-v001"
D2S_ARMS = ("D160", "G96", "D2S-U128-B128")
D2S_SEEDS = (4407, 4408, 4409)


def config_path(root: str | Path, arm: str, seed: int) -> Path:
    names = {
        "D160": "d160",
        "G96": "g96",
        "D2S-U128-B128": "d2s_v3_u128_burst128",
    }
    if arm not in names or seed not in D2S_SEEDS:
        raise ValueError(f"unsupported cell arm={arm!r} seed={seed!r}")
    if arm == "D2S-U128-B128":
        filename = f"continuous_roi_{names[arm]}_seed{seed}.py"
    else:
        filename = f"continuous_roi_s2_v3_{names[arm]}_seed{seed}.py"
    return Path(root) / "configs" / "adatad" / "thumos" / filename


def validate_d2s_cell_config(path: str | Path, *, arm: str, seed: int) -> dict[str, Any]:
    path = Path(path)
    cfg = Config.fromfile(path)
    if arm == "D2S-U128-B128":
        binding = cfg.continuous_roi_d2s_v3_full200_compute
        if binding.protocol != D2S_PROTOCOL_ID or binding.arm != arm or int(binding.seed) != seed:
            raise ValueError("D2S cell protocol, arm, or seed binding changed")
        custom = cfg.model.backbone.custom
        if (
            custom.wrapper_type != "d2s_temporal_zoom_shared_videomae"
            or int(custom.burst_chunks) != 16
            or int(custom.total_chunks) != 48
            or int(custom.global_size) != 96
            or int(custom.local_size) != 128
            or str(custom.source_key) != "source"
            or bool(custom.return_feature_bundle)
        ):
            raise ValueError("D2S backbone configuration changed")
        for split in ("train", "val", "test"):
            views = [
                step
                for step in cfg.dataset[split].pipeline
                if step.type == "ContinuousRoiSourceViews"
            ]
            if len(views) != 1 or int(views[0].global_size) != 96:
                raise ValueError("D2S must receive global96 plus untouched source uint8")
            if any(
                step.type == "NativeCropSourceViews"
                for step in cfg.dataset[split].pipeline
            ):
                raise ValueError("D2S forbids pre-materializing all local crops")
        optimizer_names = [row.name for row in cfg.optimizer.backbone.custom]
        if optimizer_names != ["adapter", "proj_local", "proj_global", "gamma"]:
            raise ValueError("D2S residual parameters are not explicitly optimized")
    else:
        binding = cfg.continuous_roi_s2_v3_full200_compute
        if binding.arm != arm or int(binding.seed) != seed:
            raise ValueError("baseline cell arm or seed binding changed")

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
        int(cfg.solver.train.batch_size) != 2
        or not bool(cfg.solver.ema)
        or int(cfg.workflow.max_train_iters) != EXPECTED_UPDATES_PER_EPOCH
        or int(cfg.workflow.end_epoch) != EXPECTED_EPOCHS
        or str(cfg.workflow.checkpoint_policy) != "final_only"
        or int(cfg.inference.test_epoch) != 59
    ):
        raise ValueError("formal schedule or final-EMA rule changed")

    return {
        "arm": arm,
        "seed": seed,
        "config": str(path.resolve()),
        "work_dir": str(cfg.work_dir),
        "final_checkpoint": "checkpoint/epoch_59.pth:state_dict_ema",
    }


def validate_d2s_matrix(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    cells = [
        validate_d2s_cell_config(config_path(root, arm, seed), arm=arm, seed=seed)
        for arm in D2S_ARMS
        for seed in D2S_SEEDS
    ]
    return {
        "protocol_id": D2S_PROTOCOL_ID,
        "cells": cells,
        "cell_count": len(cells),
        "training_identities": EXPECTED_TRAINING_IDENTITIES,
        "evaluation_videos": EXPECTED_EVALUATION_VIDEOS,
        "evaluation_ordered_windows": EXPECTED_EVALUATION_WINDOWS,
        "successful_updates_per_cell": EXPECTED_TOTAL_UPDATES,
        "world_size": EXPECTED_WORLD_SIZE,
    }


def validate_d2s_parameter_fairness(root: str | Path) -> dict[str, Any]:
    """Validate baseline parity and disclose, rather than hide, candidate modules."""

    root = Path(root)
    d160 = Config.fromfile(config_path(root, "D160", 4407))
    g96 = Config.fromfile(config_path(root, "G96", 4407))
    if d160.model.to_dict() != g96.model.to_dict() or d160.optimizer.to_dict() != g96.optimizer.to_dict():
        raise ValueError("D160 and G96 baseline trainable surfaces differ")
    candidate = Config.fromfile(config_path(root, "D2S-U128-B128", 4407))
    optimizer_names = [row.name for row in candidate.optimizer.backbone.custom]
    return {
        "baseline_parameter_surface": "D160_EQUALS_G96",
        "candidate_parameter_parity_claimed": False,
        "candidate_added_trainable_modules": ["proj_local", "proj_global", "gamma"],
        "candidate_optimizer_groups": optimizer_names,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the full-data D2S-TAD matrix")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--manifest-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_args = (args.annotation, args.class_map, args.media_root, args.manifest_dir)
    if any(value is not None for value in manifest_args):
        if not all(value is not None for value in manifest_args):
            raise ValueError(
                "--annotation, --class-map, --media-root and --manifest-dir are required together"
            )
        manifest = build_full_data_bundle(
            args.annotation,
            args.class_map,
            args.media_root,
            args.manifest_dir,
            protocol_id=D2S_PROTOCOL_ID,
        )
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    receipt = validate_d2s_matrix(args.root)
    receipt["parameter_disclosure"] = validate_d2s_parameter_fairness(args.root)
    if args.output is not None:
        atomic_publish_json(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
