from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

from mmengine.config import Config

from tools.bata import duca_paper_training


CONFIGS = {
    "dense": "configs/adatad/thumos/duca_paper_dense_actionformer_full200.py",
    "uniform_fixed_k384": (
        "configs/adatad/thumos/duca_paper_uniform_fixed_k384_full200.py"
    ),
    "uniform_mixed_train_k384_eval": (
        "configs/adatad/thumos/duca_paper_uniform_mixed_train_k384_eval_full200.py"
    ),
    "duca_fixed_k384": (
        "configs/adatad/thumos/duca_paper_duca_fixed_k384_full200.py"
    ),
}
IMMUTABLE_FAILED_STAGE_A_COMMIT = "2df0103ec1c26ff7cff7ed15f399e78e640df211"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return duca_paper_training.canonical_sha256(value)


def _exact_checkout(repo_root: Path, expected_commit: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("DUCA paper matrix requires an exact Git commit")
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, encoding="utf-8"
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        text=True,
        encoding="utf-8",
    ).strip()
    if observed != expected_commit or status:
        raise RuntimeError("DUCA paper matrix requires a clean exact-commit checkout")


def _annotation_identity(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    database = payload.get("database", {})
    if not isinstance(database, dict):
        raise RuntimeError("THUMOS14 annotation lacks its database mapping")
    train_ids = sorted(
        str(video_id)
        for video_id, row in database.items()
        if str(row.get("subset", "")) == "training"
    )
    validation_ids = sorted(
        str(video_id)
        for video_id, row in database.items()
        if str(row.get("subset", "")) == "validation"
    )
    if (
        len(train_ids) != duca_paper_training.TRAIN_VIDEO_COUNT
        or len(validation_ids) != duca_paper_training.EVALUATION_VIDEO_COUNT
        or set(train_ids) & set(validation_ids)
        or len(database) != len(train_ids) + len(validation_ids)
    ):
        raise RuntimeError("THUMOS14 annotation is not the exact 200/211 official split")
    return {
        "training_video_count": len(train_ids),
        "validation_video_count": len(validation_ids),
        "training_video_ids_sha256": canonical_sha256(train_ids),
        "validation_video_ids_sha256": canonical_sha256(validation_ids),
    }


def build_manifest(
    *,
    repo_root: str | Path,
    expected_commit: str,
    pretrain_path: str | Path,
    annotation_path: str | Path,
    class_map_path: str | Path,
    short_window_gate_path: str | Path | None = None,
    numeric_gate_path: str | Path | None = None,
    exact211_uid_gate_path: str | Path | None = None,
    require_clean_checkout: bool = True,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    pretrain = Path(pretrain_path).expanduser().resolve()
    annotation = Path(annotation_path).expanduser().resolve()
    class_map = Path(class_map_path).expanduser().resolve()
    short_window_gate = (
        None
        if short_window_gate_path is None
        else Path(short_window_gate_path).expanduser().resolve()
    )
    numeric_gate = (
        None
        if numeric_gate_path is None
        else Path(numeric_gate_path).expanduser().resolve()
    )
    exact211_uid_gate = (
        None
        if exact211_uid_gate_path is None
        else Path(exact211_uid_gate_path).expanduser().resolve()
    )
    if require_clean_checkout:
        if expected_commit == IMMUTABLE_FAILED_STAGE_A_COMMIT:
            raise RuntimeError(
                "formal Stage-A freeze cannot reuse the immutable failed source"
            )
        _exact_checkout(repo, expected_commit)
        if (
            short_window_gate is None
            or numeric_gate is None
            or exact211_uid_gate is None
        ):
            raise RuntimeError(
                "formal Stage-A freeze requires short-window, numeric, and exact-211 UID gates"
            )
    for path, label in (
        (pretrain, "VideoMAE initialization"),
        (annotation, "THUMOS14 annotation"),
        (class_map, "THUMOS14 class map"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"DUCA paper {label} is missing: {path}")

    config_records: dict[str, Any] = {}
    for arm in duca_paper_training.ARMS:
        relative = CONFIGS[arm]
        path = repo / relative
        cfg = Config.fromfile(path)
        contract = duca_paper_training.validate_static_config(cfg)
        if contract["variant"] != arm:
            raise RuntimeError(f"DUCA paper config/arm drift: {relative}")
        config_records[arm] = {
            "path": relative,
            "sha256": sha256_file(path),
            "resolved_sha256": canonical_sha256(cfg.to_dict()),
            "evaluation_heavy_k": int(cfg.duca_paper_cell.evaluation_heavy_k),
        }

    annotation_identity = _annotation_identity(annotation)
    class_names = [
        line.strip()
        for line in class_map.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(class_names) != 20:
        raise RuntimeError("THUMOS14 class map must contain exactly twenty classes")
    prerequisite_gates: dict[str, Any] = {}
    if short_window_gate is not None:
        from tools.bata.validate_duca_paper_short_window_gate import (
            validate_gate_artifact,
        )

        gate_binding = validate_gate_artifact(
            short_window_gate,
            expected_commit=expected_commit,
        )
        prerequisite_gates["clean_linux_pytorch_code"] = {
            "schema_version": "duca_paper_clean_linux_code_gate_v3",
            "status": "passed",
            "fail_closed": True,
            "git_commit": expected_commit,
            "path": gate_binding["code_gate_path"],
            "sha256": gate_binding["code_gate_sha256"],
            "slurm_job_id": gate_binding["code_gate_slurm_job_id"],
            "claim_scope": "engineering_clean_linux_pytorch_code_only",
            "performance_evidence": False,
        }
        prerequisite_gates["real_natural_short_window_heavy_backbone"] = {
            "schema_version": (
                "duca_paper_real_short_window_heavy_backbone_gate_v1"
            ),
            "status": "passed",
            "git_commit": expected_commit,
            "path": str(short_window_gate),
            "sha256": sha256_file(short_window_gate),
            "slurm_job_id": gate_binding["slurm_job_id"],
            "claim_scope": "engineering_short_window_execution_only",
            "performance_evidence": False,
        }
    if numeric_gate is not None:
        from tools.bata.validate_duca_paper_numeric_gate import (
            validate_numeric_gate_artifact,
        )

        prerequisite_gates["production_like_learned_exactk_numeric"] = (
            validate_numeric_gate_artifact(
                numeric_gate,
                expected_commit=expected_commit,
                expected_sha256=sha256_file(numeric_gate),
            )
        )
    if exact211_uid_gate is not None:
        from tools.bata.validate_duca_paper_exact211_uid_gate import (
            validate_exact211_uid_gate_artifact,
        )

        prerequisite_gates["exact211_physical_uid_metadata"] = (
            validate_exact211_uid_gate_artifact(
                exact211_uid_gate,
                expected_commit=expected_commit,
                expected_sha256=sha256_file(exact211_uid_gate),
            )
        )
    payload = {
        "schema_version": duca_paper_training.MATRIX_SCHEMA,
        "status": "frozen",
        "git_commit": expected_commit,
        "task": "offline_temporal_action_detection",
        "detector_backend": "ActionFormer",
        "arms": list(duca_paper_training.ARMS),
        "seeds": list(duca_paper_training.SEEDS),
        "train_video_count": duca_paper_training.TRAIN_VIDEO_COUNT,
        "evaluation_video_count": duca_paper_training.EVALUATION_VIDEO_COUNT,
        "world_size": duca_paper_training.WORLD_SIZE,
        "global_batch_size": duca_paper_training.GLOBAL_BATCH_SIZE,
        "epochs": duca_paper_training.EPOCHS,
        "updates_per_epoch": duca_paper_training.UPDATES_PER_EPOCH,
        "successful_updates": duca_paper_training.SUCCESSFUL_UPDATES,
        "terminal_checkpoint_epoch": 59,
        "terminal_checkpoint_state_key": "state_dict_ema",
        "training_consumes_validation": False,
        "single_seed_claim_allowed": False,
        "partial_matrix_claim_allowed": False,
        "prerequisite_gates": prerequisite_gates,
        "budget_semantics": {
            "version": duca_paper_training.BUDGET_SEMANTICS,
            "valid_length_definition": "contiguous_true_dense_candidate_prefix",
            "execution_quantum": duca_paper_training.EXECUTION_QUANTUM,
            "effective_k_formula": (
                "min(requested_k,floor(dense_valid_len/16)*16)"
            ),
            "subquantum_policy": "fail_closed_below_one_quantum",
            "padding_or_repetition_allowed": False,
            "length_conditioned_requested_schedule": False,
            "fixed_requested_k384_evaluation_is_dynamic": False,
            "mixed_k": {
                "candidate_budgets": list(
                    duca_paper_training.MIXED_K_CANDIDATES
                ),
                "schedule_counts": list(duca_paper_training.MIXED_K_COUNTS),
                "schedule_seed": duca_paper_training.MIXED_K_SEED,
                "cycle": list(duca_paper_training.mixed_k_requested_schedule()),
                "cycle_length": len(
                    duca_paper_training.mixed_k_requested_schedule()
                ),
                "nominal_requested_mean_k": (
                    duca_paper_training.MIXED_K_NOMINAL_REQUESTED_MEAN
                ),
                "schedule_sha256": (
                    duca_paper_training.mixed_k_requested_schedule_sha256()
                ),
            },
        },
        "configs": config_records,
        "assets": {
            "pretrain_path": str(pretrain),
            "pretrain_sha256": sha256_file(pretrain),
            "annotation_path": str(annotation),
            "annotation_sha256": sha256_file(annotation),
            "class_map_path": str(class_map),
            "class_map_sha256": sha256_file(class_map),
            "class_count": len(class_names),
            **annotation_identity,
        },
        "cells": [
            {"arm": arm, "seed": seed}
            for arm in duca_paper_training.ARMS
            for seed in duca_paper_training.SEEDS
        ],
    }
    payload["scientific_contract_sha256"] = canonical_sha256(payload)
    return payload


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"DUCA paper matrix output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the full-200/exact-211 DUCA paper Stage-A matrix."
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--short-window-gate", required=True)
    parser.add_argument("--numeric-gate", required=True)
    parser.add_argument("--exact211-uid-gate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = build_manifest(
        repo_root=args.repo_root,
        expected_commit=args.expected_commit,
        pretrain_path=args.pretrain,
        annotation_path=args.annotation,
        class_map_path=args.class_map,
        short_window_gate_path=args.short_window_gate,
        numeric_gate_path=args.numeric_gate,
        exact211_uid_gate_path=args.exact211_uid_gate,
    )
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "path": str(Path(args.output).resolve()),
                "sha256": sha256_file(args.output),
                "cell_count": len(payload["cells"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
