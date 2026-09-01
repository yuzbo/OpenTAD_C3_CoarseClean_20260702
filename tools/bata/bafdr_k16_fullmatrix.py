# Copyright (c) OpenTAD. All rights reserved.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from mmengine.config import Config

PROTOCOL_ID = "ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001"
ARMS = (
    "D160",
    "G96",
    "U128-ALL48-A0",
    "U16-UNIFORM-A0",
    "BAFDR-K16-LATE",
    "BAFDR-K16-NOKD",
    "BAFDR-K16-FULL",
)
SEEDS = (4407, 4408, 4409)
EXPECTED_TRAINING_IDENTITIES = 200
EXPECTED_EVALUATION_VIDEOS = 211
EXPECTED_EVALUATION_WINDOWS = 792
EXPECTED_UPDATES_PER_EPOCH = 100
EXPECTED_EPOCHS = 60
EXPECTED_TOTAL_UPDATES = 6000
EXPECTED_WORLD_SIZE = 2
WINDOW_SIZE = 768
WINDOW_OVERLAP_RATIO = 0.5

ARM_CONFIG_NAMES = {
    "D160": "d160",
    "G96": "g96",
    "U128-ALL48-A0": "u128_all48_a0",
    "U16-UNIFORM-A0": "u16_uniform_a0",
    "BAFDR-K16-LATE": "late",
    "BAFDR-K16-NOKD": "nokd",
    "BAFDR-K16-FULL": "full",
}
BAFDR_WRAPPER_ARMS = {
    "U16-UNIFORM-A0",
    "BAFDR-K16-LATE",
    "BAFDR-K16-NOKD",
    "BAFDR-K16-FULL",
}
EXPECTED_PROJECTION = {
    "U16-UNIFORM-A0": "BAFDRAsymmetricProjection",
    "BAFDR-K16-LATE": "BAFDRLateProjection",
    "BAFDR-K16-NOKD": "BAFDRAsymmetricProjection",
    "BAFDR-K16-FULL": "BAFDRAsymmetricProjection",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def require_clean_commit(repo_root: str | Path, *, allow_dirty: bool = False) -> tuple[str, List[str]]:
    root = Path(repo_root)
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    tracked_status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--short", "--untracked-files=no"],
        text=True,
    ).splitlines()
    if tracked_status and not allow_dirty:
        joined = "\n".join(tracked_status[:50])
        raise RuntimeError(
            "BA-FDR formal submission requires a clean tracked worktree. "
            "Commit or stash these changes first:\n" + joined
        )
    return head, tracked_status


def get_matrix_cells(root: Path) -> List[Tuple[str, int, Path]]:
    cells = []
    for arm in ARMS:
        slug = ARM_CONFIG_NAMES[arm]
        for seed in SEEDS:
            cfg_name = f"bafdr_k16_{slug}_seed{seed}.py"
            cells.append((arm, seed, root / "configs" / "adatad" / "thumos" / cfg_name))
    return cells


def _get(mapping: Any, key: str, default: Any = None) -> Any:
    if mapping is None:
        return default
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _pipeline_for(cfg: Config, split: str) -> Sequence[Mapping[str, Any]]:
    dataset = _get(cfg, "dataset")
    split_cfg = _get(dataset, split)
    pipeline = _get(split_cfg, "pipeline")
    if pipeline is None:
        raise ValueError(f"dataset.{split}.pipeline is missing")
    return pipeline


def _find_steps(pipeline: Sequence[Mapping[str, Any]], step_type: str) -> List[Mapping[str, Any]]:
    return [step for step in pipeline if _get(step, "type") == step_type]


def validate_bafdr_pipeline(cfg: Config, *, config_path: Path, arm: str) -> Dict[str, Any]:
    split_info = {}
    for split in ("train", "val", "test"):
        pipeline = _pipeline_for(cfg, split)
        source_steps = _find_steps(pipeline, "BAFDRSourceViews")
        if len(source_steps) != 1:
            raise ValueError(f"{config_path}: {split} pipeline must contain exactly one BAFDRSourceViews")
        source_step = source_steps[0]
        if _get(source_step, "output_key") != "bafdr_inputs":
            raise ValueError(f"{config_path}: {split} BAFDRSourceViews.output_key must be bafdr_inputs")
        if int(_get(source_step, "global_size")) != 96:
            raise ValueError(f"{config_path}: {split} BAFDRSourceViews.global_size must be 96")
        if int(_get(source_step, "required_source_height")) != 180 or int(_get(source_step, "required_source_width")) != 320:
            raise ValueError(f"{config_path}: {split} BAFDRSourceViews must be frozen to 180x320 source")

        collect_steps = _find_steps(pipeline, "Collect")
        if len(collect_steps) != 1:
            raise ValueError(f"{config_path}: {split} pipeline must contain exactly one Collect")
        collect = collect_steps[0]
        if "extra_keys" in collect:
            raise ValueError(f"{config_path}: {split} Collect uses unsupported extra_keys")
        if _get(collect, "inputs") != "bafdr_inputs":
            raise ValueError(f"{config_path}: {split} Collect.inputs must be bafdr_inputs")
        keys = set(_get(collect, "keys", []))
        required_keys = {"masks", "gt_segments", "gt_labels"}
        if not required_keys.issubset(keys):
            raise ValueError(f"{config_path}: {split} Collect.keys missing {sorted(required_keys - keys)}")
        meta_keys = set(_get(collect, "meta_keys", []))
        required_meta = {"id", "fps", "duration", "video_name", "snippet_boundaries", "window_start_frame", "bafdr_geometry"}
        if not required_meta.issubset(meta_keys):
            raise ValueError(f"{config_path}: {split} Collect.meta_keys missing {sorted(required_meta - meta_keys)}")

        load_steps = _find_steps(pipeline, "LoadFrames")
        if len(load_steps) != 1:
            raise ValueError(f"{config_path}: {split} pipeline must contain exactly one LoadFrames")
        load = load_steps[0]
        if split == "train":
            if _get(load, "method") != "random_trunc" or int(_get(load, "trunc_len")) != WINDOW_SIZE:
                raise ValueError(f"{config_path}: train LoadFrames must be random_trunc/trunc_len=768")
        else:
            if _get(load, "method") != "sliding_window":
                raise ValueError(f"{config_path}: {split} LoadFrames must use sliding_window")
            if int(_get(load, "window_size")) != WINDOW_SIZE:
                raise ValueError(f"{config_path}: {split} window_size must be 768")
            if float(_get(load, "window_overlap_ratio")) != WINDOW_OVERLAP_RATIO:
                raise ValueError(f"{config_path}: {split} overlap ratio must be 0.5")

        split_info[split] = {
            "pipeline_len": len(pipeline),
            "collect_inputs": _get(collect, "inputs"),
            "source_schema": "bafdr_source_global_v1",
        }
    return split_info


def validate_bafdr_model(cfg: Config, *, config_path: Path, arm: str) -> Dict[str, Any]:
    model = _get(cfg, "model")
    backbone = _get(model, "backbone")
    custom = _get(backbone, "custom")
    if _get(custom, "wrapper_type") != "bafdr_k16_shared_videomae":
        raise ValueError(f"{config_path}: BA-FDR arms must use bafdr_k16_shared_videomae")
    expected_uniform = arm == "U16-UNIFORM-A0"
    if bool(_get(custom, "bafdr_uniform_mode")) != expected_uniform:
        raise ValueError(f"{config_path}: bafdr_uniform_mode mismatch for {arm}")
    expected_fields = {
        "bafdr_global_key": "global",
        "bafdr_source_key": "source",
        "bafdr_global_size": 96,
        "bafdr_local_size": 128,
        "bafdr_chunk_num": 48,
        "bafdr_k_chunks": 16,
        "bafdr_tubelets_per_chunk": 8,
        "bafdr_output_length": 768,
        "bafdr_return_bundle": True,
    }
    for key, expected in expected_fields.items():
        if _get(custom, key) != expected:
            raise ValueError(f"{config_path}: {key}={_get(custom, key)!r}, expected {expected!r}")

    projection = _get(model, "projection")
    projection_type = _get(projection, "type")
    if projection_type != EXPECTED_PROJECTION[arm]:
        raise ValueError(f"{config_path}: projection type {projection_type!r} does not match {arm}")
    if int(_get(projection, "in_channels")) != 384 or int(_get(projection, "out_channels")) != 512:
        raise ValueError(f"{config_path}: BA-FDR projection channels must be 384->512")
    if tuple(_get(projection, "arch")) != (2, 2, 5):
        raise ValueError(f"{config_path}: BA-FDR projection arch must be (2, 2, 5)")

    optimizer = _get(cfg, "optimizer")
    opt_backbone = _get(optimizer, "backbone")
    custom_groups = _get(opt_backbone, "custom", [])
    custom_names = {_get(group, "name") for group in custom_groups}
    required_names = {"adapter", "router", "gamma", "proj_local", "proj_global"}
    if not required_names.issubset(custom_names):
        raise ValueError(f"{config_path}: optimizer custom groups missing {sorted(required_names - custom_names)}")
    if "backbone.model" not in set(_get(opt_backbone, "exclude", [])):
        raise ValueError(f"{config_path}: optimizer must exclude frozen shared VideoMAE backbone.model")

    return {
        "wrapper_type": _get(custom, "wrapper_type"),
        "projection_type": projection_type,
        "optimizer_custom_groups": sorted(custom_names),
    }


def validate_distillation_contract(cfg: Config, *, config_path: Path, arm: str, seed: int) -> Dict[str, Any]:
    bafdr_protocol = _get(cfg, "bafdr_protocol")
    distillation = bool(_get(bafdr_protocol, "distillation", False))
    if arm == "BAFDR-K16-FULL":
        if not distillation:
            raise ValueError(f"{config_path}: FULL arm must enable distillation")
        teacher_config = _get(bafdr_protocol, "teacher_config")
        teacher_checkpoint = _get(bafdr_protocol, "teacher_checkpoint")
        if not teacher_config or not teacher_checkpoint:
            raise ValueError(f"{config_path}: FULL arm must declare teacher_config and teacher_checkpoint")
        if f"seed{seed}" not in str(teacher_config) or "d160" not in str(teacher_config):
            raise ValueError(f"{config_path}: FULL teacher_config must be same-seed D160")
        return {
            "distillation": True,
            "teacher_config": str(teacher_config),
            "teacher_checkpoint": str(teacher_checkpoint),
        }
    if distillation:
        raise ValueError(f"{config_path}: only FULL arm may enable distillation")
    return {"distillation": False}


def validate_cell_config(config_path: str | Path, expected_arm: str, expected_seed: int) -> dict[str, Any]:
    config_path = Path(config_path)
    cfg = Config.fromfile(str(config_path))
    bafdr_protocol = _get(cfg, "bafdr_protocol")
    if bafdr_protocol is None:
        raise ValueError(f"config {config_path} lacks bafdr_protocol dictionary")
    if _get(bafdr_protocol, "protocol") != PROTOCOL_ID:
        raise ValueError(f"config protocol mismatch: {_get(bafdr_protocol, 'protocol')} != {PROTOCOL_ID}")
    if _get(bafdr_protocol, "arm") != expected_arm:
        raise ValueError(f"config arm mismatch: {_get(bafdr_protocol, 'arm')} != {expected_arm}")
    if int(_get(bafdr_protocol, "seed")) != int(expected_seed):
        raise ValueError(f"config seed mismatch: {_get(bafdr_protocol, 'seed')} != {expected_seed}")

    details: Dict[str, Any] = {}
    if expected_arm in BAFDR_WRAPPER_ARMS:
        details["pipeline"] = validate_bafdr_pipeline(cfg, config_path=config_path, arm=expected_arm)
        details["model"] = validate_bafdr_model(cfg, config_path=config_path, arm=expected_arm)
        details["distillation"] = validate_distillation_contract(
            cfg,
            config_path=config_path,
            arm=expected_arm,
            seed=expected_seed,
        )

    return {
        "config_path": str(config_path),
        "arm": expected_arm,
        "seed": expected_seed,
        "config_sha256": sha256_file(config_path),
        "details": details,
    }


def receipt_for_cell(root: Path, arm: str, seed: int, work_dir_root: str | Path | None = None) -> Path:
    slug = ARM_CONFIG_NAMES[arm]
    if work_dir_root is not None:
        return Path(work_dir_root) / f"bafdr_k16_{slug}_seed{seed}" / "eval_receipt.json"
    return root / "exps" / "thumos" / "adatad" / f"bafdr_k16_{slug}_seed{seed}" / "eval_receipt.json"


def phase_receipt_for_cell(work_dir_root: str | Path, arm: str, seed: int, receipt_name: str) -> Path:
    slug = ARM_CONFIG_NAMES[arm]
    return Path(work_dir_root) / f"bafdr_k16_{slug}_seed{seed}" / receipt_name


def seal_prediction_receipts(root: Path, *, work_dir_root: str | Path, output_file: str | Path) -> Dict[str, Any]:
    cells = get_matrix_cells(root)
    sealed_cells = []
    failures = []
    for arm, seed, cfg_path in cells:
        receipt_path = phase_receipt_for_cell(work_dir_root, arm, seed, "prediction_receipt.json")
        if not receipt_path.exists():
            failures.append(f"{arm}/seed{seed}: missing prediction receipt")
            continue
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        if data.get("protocol_id") != PROTOCOL_ID:
            failures.append(f"{arm}/seed{seed}: protocol mismatch")
        if data.get("phase") != "prediction_seal" or bool(data.get("metric_opened")):
            failures.append(f"{arm}/seed{seed}: prediction receipt opened metrics")
        if int(data.get("world_size", 0)) != EXPECTED_WORLD_SIZE:
            failures.append(f"{arm}/seed{seed}: prediction world_size mismatch")
        if int(data.get("total_successful_updates", 0)) != EXPECTED_TOTAL_UPDATES:
            failures.append(f"{arm}/seed{seed}: checkpoint update mismatch")
        raw_dir = Path(data.get("raw_prediction_dir", ""))
        raw_count = len(list(raw_dir.glob("*.pkl"))) if raw_dir.exists() else 0
        if raw_count != EXPECTED_EVALUATION_WINDOWS:
            failures.append(f"{arm}/seed{seed}: raw prediction file count {raw_count}, expected {EXPECTED_EVALUATION_WINDOWS}")
        sealed_cells.append(
            {
                "arm": arm,
                "seed": seed,
                "config_path": str(cfg_path),
                "prediction_receipt": str(receipt_path),
                "raw_prediction_dir": str(raw_dir),
                "raw_prediction_file_count": raw_count,
                "checkpoint_sha256": data.get("checkpoint_sha256"),
            }
        )

    payload = {
        "schema_version": "ZOOMTOKEN-BA-FDR-K16-PREDICTION-SEAL-v001",
        "protocol_id": PROTOCOL_ID,
        "work_dir_root": str(work_dir_root),
        "expected_cells": len(cells),
        "sealed_cells": sealed_cells,
        "failures": failures,
        "status": "PREDICTIONS_SEALED_NO_METRICS_OPENED" if not failures else "FAILED",
    }
    atomic_publish_json(output_file, payload)
    if failures:
        raise RuntimeError("BA-FDR prediction seal failed: " + "; ".join(failures[:20]))
    print(f"[BA-FDR] Prediction seal written to {output_file} ({len(sealed_cells)} cells).")
    return payload


def summarize_matrix_results(
    root: Path,
    *,
    work_dir_root: str | Path | None = None,
    output_file: str | Path = "matrix_summary.json",
    require_complete: bool = False,
) -> List[Dict[str, Any]]:
    cells = get_matrix_cells(root)
    results = []
    missing = []
    print(f"\n{'ARM':<18} | {'SEED':<6} | {'mAP':<8} | {'AP@0.5':<8} | {'Updates':<8} | {'Status'}")
    print("-" * 78)
    for arm, seed, _ in cells:
        receipt_path = receipt_for_cell(root, arm, seed, work_dir_root=work_dir_root)
        if receipt_path.exists():
            data = json.loads(receipt_path.read_text(encoding="utf-8"))
            eval_res = data.get("eval_results", {})
            m_ap = eval_res.get("mAP", "N/A")
            ap50 = eval_res.get("AP@0.5", "N/A")
            updates = data.get("total_successful_updates", "N/A")
            complete = (
                data.get("protocol_id") == PROTOCOL_ID
                and data.get("phase") == "metric_opening"
                and bool(data.get("metric_opened"))
                and int(data.get("total_successful_updates", 0)) == EXPECTED_TOTAL_UPDATES
            )
            status = "COMPLETED" if complete else "INCOMPLETE_RECEIPT"
        else:
            m_ap, ap50, updates = "N/A", "N/A", "N/A"
            status = "MISSING"
            missing.append(f"{arm}/seed{seed}")
        if require_complete and status != "COMPLETED":
            missing.append(f"{arm}/seed{seed}:{status}")
        row = {
            "arm": arm,
            "seed": seed,
            "mAP": m_ap,
            "AP@0.5": ap50,
            "updates": updates,
            "status": status,
            "receipt_path": str(receipt_path),
        }
        results.append(row)
        print(f"{arm:<18} | {seed:<6} | {str(m_ap):<8} | {str(ap50):<8} | {str(updates):<8} | {status}")

    atomic_publish_json(output_file, {"protocol_id": PROTOCOL_ID, "cells": results})
    print(f"\n[BA-FDR] Summary written to {output_file}")
    if missing and require_complete:
        raise RuntimeError("BA-FDR matrix is incomplete: " + ", ".join(missing[:40]))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="BA-FDR K16 Master Orchestrator")
    parser.add_argument("--repo-root", type=str, default=".", help="path to repo root")
    parser.add_argument("--output", type=str, default="submission_receipt.json", help="output submission receipt")
    parser.add_argument("--array-idx", type=int, default=None, help="array task index 0-20 to get cell config")
    parser.add_argument("--array-cell-json", action="store_true", help="print array cell metadata as JSON")
    parser.add_argument("--summary", action="store_true", help="summarize matrix results from work_dirs")
    parser.add_argument("--seal-predictions", action="store_true", help="verify all 21 prediction receipts before metric opening")
    parser.add_argument("--work-dir-root", type=str, default=None, help="matrix work_dir root for summary receipts")
    parser.add_argument("--summary-output", type=str, default="matrix_summary.json", help="summary output JSON")
    parser.add_argument("--require-complete", action="store_true", help="fail summary if any receipt is incomplete")
    parser.add_argument("--allow-dirty", action="store_true", help="allow tracked worktree changes during local validation")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    cells = get_matrix_cells(root)

    if args.summary:
        summarize_matrix_results(
            root,
            work_dir_root=args.work_dir_root,
            output_file=args.summary_output,
            require_complete=args.require_complete,
        )
        return

    if args.seal_predictions:
        if args.work_dir_root is None:
            raise ValueError("--seal-predictions requires --work-dir-root")
        seal_prediction_receipts(root, work_dir_root=args.work_dir_root, output_file=args.output)
        return

    if args.array_idx is not None:
        if not 0 <= args.array_idx < len(cells):
            raise IndexError(f"Array index {args.array_idx} out of range (0..{len(cells)-1})")
        arm, seed, cfg_path = cells[args.array_idx]
        if args.array_cell_json:
            print(json.dumps({"array_idx": args.array_idx, "arm": arm, "seed": seed, "config_path": str(cfg_path)}))
        else:
            print(f"{cfg_path}")
        return

    head, tracked_dirty = require_clean_commit(root, allow_dirty=args.allow_dirty)
    print(f"[BA-FDR] Validating {len(cells)} configs on commit {head}...")
    matrix_cells = []
    for arm, seed, cfg_path in cells:
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config {cfg_path}")
        matrix_cells.append(validate_cell_config(cfg_path, arm, seed))

    matrix_contract = {
        "protocol_id": PROTOCOL_ID,
        "commit_sha": head,
        "tracked_dirty": tracked_dirty,
        "total_cells": len(matrix_cells),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "expected_training_identities": EXPECTED_TRAINING_IDENTITIES,
        "expected_evaluation_videos": EXPECTED_EVALUATION_VIDEOS,
        "expected_evaluation_windows": EXPECTED_EVALUATION_WINDOWS,
        "expected_epochs": EXPECTED_EPOCHS,
        "expected_updates_per_epoch": EXPECTED_UPDATES_PER_EPOCH,
        "expected_total_updates": EXPECTED_TOTAL_UPDATES,
        "expected_world_size": EXPECTED_WORLD_SIZE,
        "official_test_opened": False,
        "cells": matrix_cells,
    }
    receipt = {
        **matrix_contract,
        "matrix_contract_sha256": canonical_sha256(matrix_contract),
        "status": "VALIDATED_WITH_DIRTY_WORKTREE" if tracked_dirty else "VALIDATED_AND_READY",
    }
    atomic_publish_json(args.output, receipt)
    print(f"[BA-FDR] Master submission receipt written to {args.output} ({len(matrix_cells)} cells validated).")


if __name__ == "__main__":
    main()
