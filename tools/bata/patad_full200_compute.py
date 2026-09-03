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


PATAD_PROTOCOL_ID = "ZOOMTOKEN-PATAD-FULL200-COMPUTE-PARETO-3X3-v001"
PATAD_ARMS = ("D160", "G96", "PATAD-U128-B128")
SEEDS = (4407, 4408, 4409)


def config_path(root: str | Path, arm: str, seed: int) -> Path:
    root = Path(root)
    if arm == "PATAD-U128-B128":
        name = f"continuous_roi_patad_v3_u128_seed{seed}.py"
    elif arm == "G96":
        name = f"continuous_roi_s2_v3_g96_seed{seed}.py"
    elif arm == "D160":
        name = f"continuous_roi_s2_v3_d160_seed{seed}.py"
    else:
        raise ValueError(f"unknown arm: {arm}")
    return root / "configs" / "adatad" / "thumos" / name


def validate_patad_cell_config(path: str | Path, *, arm: str, seed: int) -> dict[str, Any]:
    path = Path(path)
    cfg = Config.fromfile(path)
    if arm == "PATAD-U128-B128":
        binding = cfg.continuous_roi_patad_v3_full200_compute
        if binding.protocol != PATAD_PROTOCOL_ID or binding.arm != arm or int(binding.seed) != seed:
            raise ValueError("PATAD cell protocol, arm, or seed binding changed")
        custom = cfg.model.backbone.custom
        if (
            custom.wrapper_type != "d2s_temporal_zoom_shared_videomae"
            or int(custom.burst_chunks) != 16
            or int(custom.total_chunks) != 48
            or int(custom.global_size) != 96
            or int(custom.local_size) != 128
            or str(custom.source_key) != "source"
            or not bool(custom.return_feature_bundle)
            or bool(cfg.model.backbone.backbone.with_cp)
        ):
            raise ValueError("PATAD backbone configuration changed")
        proj = cfg.model.projection
        if proj.type != "PyramidAwareAsymmetricProj" or int(proj.asymmetric_split_level) != 2:
            raise ValueError("PATAD projection configuration changed")
        for split in ("train", "val", "test"):
            views = [
                step
                for step in cfg.dataset[split].pipeline
                if step.type == "ContinuousRoiSourceViews"
            ]
            if len(views) != 1 or int(views[0].global_size) != 96:
                raise ValueError("PATAD must receive global96 plus untouched source uint8")
            if any(
                step.type == "NativeCropSourceViews"
                for step in cfg.dataset[split].pipeline
            ):
                raise ValueError("PATAD forbids pre-materializing all local crops")
        optimizer_names = [row.name for row in cfg.optimizer.backbone.custom]
        if optimizer_names != ["adapter", "proj_local", "proj_global", "gamma"]:
            raise ValueError("PATAD D2S residual parameters are not explicitly optimized")
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


def validate_patad_matrix(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    d160 = Config.fromfile(config_path(root, "D160", 4407))
    g96 = Config.fromfile(config_path(root, "G96", 4407))
    if d160.model.to_dict() != g96.model.to_dict() or d160.optimizer.to_dict() != g96.optimizer.to_dict():
        raise ValueError("D160 and G96 baseline trainable surfaces differ")
    cells = [
        validate_patad_cell_config(config_path(root, arm, seed), arm=arm, seed=seed)
        for arm in PATAD_ARMS
        for seed in SEEDS
    ]
    summary = {
        "protocol_id": PATAD_PROTOCOL_ID,
        "world_size": EXPECTED_WORLD_SIZE,
        "training_identities": EXPECTED_TRAINING_IDENTITIES,
        "successful_updates_per_cell": EXPECTED_TOTAL_UPDATES,
        "evaluation_videos": EXPECTED_EVALUATION_VIDEOS,
        "evaluation_ordered_windows": EXPECTED_EVALUATION_WINDOWS,
        "parameter_disclosure": {
            "baseline_parameter_surface": "D160_EQUALS_G96",
            "candidate_parameter_parity_claimed": False,
            "candidate_added_trainable_modules": [
                "backbone.proj_local",
                "backbone.proj_global",
                "backbone.gamma",
                "projection.q0_inj",
                "projection.q1_inj",
            ],
        },
        "cell_count": len(cells),
        "cells": cells,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PATAD matrix")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--class-map", type=Path, default=None)
    parser.add_argument("--media-root", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.annotation and args.class_map and args.media_root and args.manifest_dir:
        build_full_data_bundle(
            args.annotation,
            args.class_map,
            args.media_root,
            args.manifest_dir,
            protocol_id=PATAD_PROTOCOL_ID,
        )

    receipt = validate_patad_matrix(args.root)
    if args.output:
        atomic_publish_json(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
