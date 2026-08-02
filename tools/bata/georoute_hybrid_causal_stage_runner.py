#!/usr/bin/env python3
"""Run one frozen world-size-two Hybrid causal pilot arm."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_hybrid_causal_contract import (  # noqa: E402
    HYBRID_CAUSAL_CONTRACT_SCHEMA,
    HYBRID_CAUSAL_EPOCHS,
    HYBRID_CAUSAL_K,
    HYBRID_CAUSAL_P0_SUITE_SCHEMA,
    HYBRID_CAUSAL_SEED,
    HYBRID_CAUSAL_STAGE_RESULT_SCHEMA,
    HYBRID_CAUSAL_STUDY_ID,
    bind_hybrid_causal_config,
    hybrid_causal_arm_spec,
    hybrid_causal_cell_relative_path,
    validate_hybrid_causal_stage_result,
)
from tools.bata.georoute_stage_runner import (  # noqa: E402
    _atomic_write_json,
    _read_json,
    _run_logged,
    _validate_rendezvous_receipt,
    build_torchrun_prefix,
    parse_official_style_map,
)
from tools.bata.georoute_storage import storage_capacity_receipt  # noqa: E402


WORLD_SIZE = 2
BOUNDARY = Path("/data/run01/sczc063/yuzibo")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    commit = completed.stdout.strip().lower()
    if len(commit) != 40:
        raise RuntimeError("Hybrid causal stage could not resolve a full commit")
    return commit


def _self_hash_matches(payload: Mapping[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


def _read_parent_p0_suite(path: Path, *, expected_commit: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("Hybrid causal stage requires its sealed P0 suite")
    payload = _read_json(path)
    if (
        payload.get("schema_version") != HYBRID_CAUSAL_P0_SUITE_SCHEMA
        or payload.get("status") != "PASS_MECHANICAL_ONLY"
        or payload.get("study_id") != HYBRID_CAUSAL_STUDY_ID
        or payload.get("runtime_commit") != expected_commit
        or payload.get("performance_training_authorized") is not True
        or payload.get("performance_outputs_emitted") is not False
        or payload.get("official_test_opened") is not False
        or payload.get("paper_claim_allowed") is not False
        or not _self_hash_matches(payload, field="suite_sha256")
    ):
        raise RuntimeError("Hybrid causal P0 suite is invalid")
    return payload


def _finite_positive(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if (
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"Hybrid causal profile lacks finite positive {key}")
    return float(value)


def _profile(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    scope = payload.get("scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("development_only") is not True
        or scope.get("evaluator_excluded") is not True
        or scope.get("paper_grade_end_to_end_claim_allowed") is not False
        or scope.get("diagnostic_route_telemetry_inside_timed_forward") is not False
        or scope.get("separate_from_accuracy_evaluation") is not True
    ):
        raise ValueError("Hybrid causal profile is not an isolated cost replay")
    return {
        "sample_count": int(payload.get("sample_count", 0)),
        "steady_sample_count": int(payload.get("steady_sample_count", 0)),
        "loader_wait_p50_ms": _finite_positive(payload, "loader_wait_p50_ms"),
        "loader_wait_p95_ms": _finite_positive(payload, "loader_wait_p95_ms"),
        "model_and_postprocess_p50_ms": _finite_positive(
            payload, "model_and_postprocess_p50_ms"
        ),
        "model_and_postprocess_p95_ms": _finite_positive(
            payload, "model_and_postprocess_p95_ms"
        ),
        "window_wall_p50_ms": _finite_positive(payload, "window_wall_p50_ms"),
        "window_wall_p95_ms": _finite_positive(payload, "window_wall_p95_ms"),
        "peak_allocated_mb": _finite_positive(payload, "peak_allocated_mb"),
        "scope": dict(scope),
        "profile_file_sha256": sha256_file(path),
        "paper_grade_end_to_end_claim_allowed": False,
    }


def _nested(record: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _mean(records: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float | None:
    values = [
        float(value)
        for record in records
        if isinstance((value := _nested(record, path)), (int, float))
        and math.isfinite(float(value))
    ]
    return sum(values) / len(values) if values else None


def _expected_role_counts(spec: Mapping[str, Any], target_k: int) -> dict[str, int]:
    counts = {
        "context": 0,
        "roi": 0,
        "residual": 0,
        "free": 0,
        "dense": 0,
        "uniform": 0,
        "random": 0,
    }
    if str(spec["route_mode"]).startswith("structured_"):
        counts.update(
            context=int(spec["context_tokens"]),
            roi=int(spec["roi_tokens"]),
            residual=int(spec["residual_tokens"]),
        )
    else:
        role = {
            "dense": "dense",
            "uniform": "uniform",
            "random": "random",
        }[str(spec["route_mode"])]
        counts[role] = int(target_k)
    return counts


def summarize_hybrid_causal_telemetry(
    path: Path,
    *,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _read_json(path)
    records = payload.get("records")
    dataset_count = int(payload.get("dataset_count", -1))
    record_count = int(payload.get("record_count", -1))
    padding_count = int(payload.get("sampler_padding_count", -1))
    if (
        payload.get("schema_version") != "georoute_formal_development_telemetry_v1"
        or payload.get("development_only") is not True
        or payload.get("official_test_opened") is not False
        or payload.get("gt_for_route_used") is not False
        or payload.get("teacher_for_route_used") is not False
        or payload.get("oracle_used") is not False
        or payload.get("raw_prediction_cache_used") is not False
        or int(payload.get("world_size", -1)) != WORLD_SIZE
        or int(payload.get("local_batch_size", -1)) != 1
        or dataset_count <= 0
        or not isinstance(records, list)
        or len(records) != record_count
        or int(payload.get("unique_dataset_count", -1)) != dataset_count
        or padding_count != record_count - dataset_count
        or not 0 <= padding_count < WORLD_SIZE
    ):
        raise ValueError("Hybrid causal telemetry population contract failed")
    if {int(record.get("dataset_index", -1)) for record in records} != set(
        range(dataset_count)
    ):
        raise ValueError("Hybrid causal telemetry does not cover the Fit population")
    routes = [record.get("route") for record in records]
    if any(
        not isinstance(route, Mapping)
        or route.get("schema_version")
        != "georoute_diagnostic_window_telemetry_v2"
        for route in routes
    ):
        raise ValueError("Hybrid causal route telemetry schema changed")
    typed_routes: list[Mapping[str, Any]] = [dict(route) for route in routes]
    target_values = {int(route.get("target_k", -1)) for route in typed_routes}
    role_values = {
        json.dumps(route.get("role_counts", {}), sort_keys=True)
        for route in typed_routes
    }
    if len(target_values) != 1 or len(role_values) != 1:
        raise ValueError("Hybrid causal exact-K/role telemetry changed by window")
    target_k = next(iter(target_values))
    expected_target = (
        int(typed_routes[0]["item_count"])
        if spec["route_mode"] == "dense"
        else HYBRID_CAUSAL_K
    )
    observed_roles = json.loads(next(iter(role_values)))
    if target_k != expected_target or observed_roles != _expected_role_counts(
        spec, target_k
    ):
        raise ValueError("Hybrid causal telemetry differs from its frozen arm")
    population_sha256 = payload.get("population_sha256")
    if not isinstance(population_sha256, str) or len(population_sha256) != 64:
        raise ValueError("Hybrid causal telemetry population hash is invalid")

    return {
        "schema_version": "georoute_hybrid_causal_telemetry_summary_v1",
        "dataset_count": dataset_count,
        "record_count": record_count,
        "sampler_padding_count": padding_count,
        "population_sha256": population_sha256,
        "telemetry_file_sha256": sha256_file(path),
        "target_k": target_k,
        "role_counts": observed_roles,
        "unique_selected_route_hash_count": len(
            {str(route["selected_index_sha256"]) for route in typed_routes}
        ),
        "total": {
            "x_span_mean": _mean(typed_routes, ("coordinates", "x_span_mean")),
            "y_span_mean": _mean(typed_routes, ("coordinates", "y_span_mean")),
            "adjacent_jaccard_mean": _mean(
                typed_routes, ("adjacent", "jaccard_mean")
            ),
            "lineage_survival_mean": _mean(
                typed_routes, ("adjacent", "lineage_retention_mean")
            ),
        },
        "roles": {
            role: {
                "x_span_mean": _mean(
                    typed_routes, ("roles", role, "x_span_mean")
                ),
                "y_span_mean": _mean(
                    typed_routes, ("roles", role, "y_span_mean")
                ),
                "adjacent_jaccard_mean": _mean(
                    typed_routes, ("roles", role, "adjacent_jaccard_mean")
                ),
                "lineage_survival_mean": _mean(
                    typed_routes, ("roles", role, "lineage_survival_mean")
                ),
            }
            for role in ("context", "roi", "residual")
        },
        "branch_entropy": {
            role: _mean(
                typed_routes,
                ("branch_entropy", role, "conditional_entropy_mean"),
            )
            for role in ("roi", "residual")
        },
        "branch_log_probability": {
            role: _mean(
                typed_routes,
                (
                    "branch_entropy",
                    role,
                    "observed_ordered_log_probability_mean",
                ),
            )
            for role in ("roi", "residual")
        },
        "route_rng_hash_count": len(
            {
                str(route["route_rng_sha256"])
                for route in typed_routes
                if route.get("route_rng_sha256") is not None
            }
        ),
        "geometry_trajectory_changed_fraction": sum(
            1
            for route in typed_routes
            if route.get("geometry", {}).get("trajectory_changed") is True
        )
        / float(len(typed_routes)),
        "development_only": True,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }


def _dump_configs(
    *,
    args: argparse.Namespace,
    cell_root: Path,
) -> tuple[Any, dict[str, Path]]:
    config_root = args.run_root.resolve() / "control" / "bound_configs"
    config_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{HYBRID_CAUSAL_STUDY_ID}_{args.arm}_seed{HYBRID_CAUSAL_SEED}"
    paths = {
        name: config_root / f"{prefix}_{name}.py"
        for name in ("train", "accuracy", "profile")
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Hybrid causal bound config namespace already exists")

    cfg = bind_hybrid_causal_config(
        source_config_path=args.source_config,
        arm=args.arm,
        seed=HYBRID_CAUSAL_SEED,
        work_dir=cell_root / "training",
        manifest_path=args.manifest,
        development_annotation_path=args.development_annotation,
        class_map_path=args.class_map,
        development_video_root=args.development_video_root,
        pretrained_checkpoint_path=args.pretrained,
    )
    train_cfg = copy.deepcopy(cfg)
    train_cfg.model.backbone.custom.georoute_diagnostic_telemetry_enabled = False
    train_cfg.georoute_diagnostic_telemetry = dict(enabled=False)
    train_cfg.georoute_development_profile = dict(enabled=False)

    accuracy_cfg = copy.deepcopy(cfg)
    accuracy_cfg.work_dir = str((cell_root / "accuracy").resolve())
    accuracy_cfg.model.backbone.custom.georoute_diagnostic_telemetry_enabled = True
    accuracy_cfg.georoute_diagnostic_telemetry = dict(enabled=True)
    accuracy_cfg.georoute_development_profile = dict(enabled=False)

    profile_cfg = copy.deepcopy(cfg)
    profile_cfg.work_dir = str((cell_root / "profile").resolve())
    profile_cfg.model.backbone.custom.georoute_diagnostic_telemetry_enabled = False
    profile_cfg.georoute_diagnostic_telemetry = dict(enabled=False)
    profile_cfg.georoute_development_profile = dict(
        enabled=True,
        separate_from_accuracy_evaluation=True,
    )
    for name, bound in (
        ("train", train_cfg),
        ("accuracy", accuracy_cfg),
        ("profile", profile_cfg),
    ):
        bound.dump(str(paths[name]))
    return cfg, paths


def _run_torchrun(
    *,
    phase: str,
    stage: str,
    variant: str,
    seed: int,
    slurm_job_id: str,
    arguments: Sequence[str],
    log_path: Path,
    env: Mapping[str, str],
) -> dict[str, Any]:
    prefix, receipt = build_torchrun_prefix(
        phase=phase,
        slurm_job_id=slurm_job_id,
        stage=stage,
        variant=variant,
        seed=seed,
        nproc_per_node=WORLD_SIZE,
    )
    _run_logged([*prefix, *arguments], log_path=log_path, env=env)
    return receipt


def _build_result(
    *,
    args: argparse.Namespace,
    cfg: Any,
    config_paths: Mapping[str, Path],
    checkpoint: Path,
    storage_receipt_path: Path,
    prediction: Path,
    telemetry: Path,
    profile_path: Path,
    accuracy_log: Path,
    runtime_commit: str,
    rendezvous: Mapping[str, Any],
    profile_rendezvous: Mapping[str, Any],
    parent_suite_path: Path,
    parent_suite: Mapping[str, Any],
) -> dict[str, Any]:
    spec = hybrid_causal_arm_spec(args.arm)
    metrics = parse_official_style_map(
        accuracy_log.read_text(encoding="utf-8", errors="replace")
    )
    metrics["high_iou_composite"] = 0.5 * (
        float(metrics["mAP@0.6"]) + float(metrics["mAP@0.7"])
    )
    raw_profile = _read_json(profile_path)
    audit = raw_profile.get("last_georoute_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("Hybrid causal profile lacks its routing audit")
    telemetry_summary = summarize_hybrid_causal_telemetry(
        telemetry,
        spec=spec,
    )
    validated_rendezvous = _validate_rendezvous_receipt(
        rendezvous,
        stage="hybrid_causal",
        variant=args.arm,
        seed=HYBRID_CAUSAL_SEED,
        nproc_per_node=WORLD_SIZE,
    )
    if (
        profile_rendezvous.get("phase") != "test"
        or profile_rendezvous.get("stage") != "hybrid_causal_profile"
        or profile_rendezvous.get("variant") != args.arm
        or int(profile_rendezvous.get("seed", -1)) != HYBRID_CAUSAL_SEED
        or int(profile_rendezvous.get("nproc_per_node", -1)) != WORLD_SIZE
    ):
        raise ValueError("Hybrid causal profile rendezvous receipt is invalid")
    result: dict[str, Any] = {
        "schema_version": HYBRID_CAUSAL_STAGE_RESULT_SCHEMA,
        "status": "PASS_EXPLORATORY_DEVELOPMENT_ONLY",
        "study_id": HYBRID_CAUSAL_STUDY_ID,
        "experiment_schema_version": HYBRID_CAUSAL_CONTRACT_SCHEMA,
        "arm": args.arm,
        "arm_spec": spec,
        "arm_spec_sha256": canonical_sha256(spec),
        "seed": HYBRID_CAUSAL_SEED,
        "epochs": HYBRID_CAUSAL_EPOCHS,
        "token_budget": "all_valid" if spec["route_mode"] == "dense" else HYBRID_CAUSAL_K,
        "metrics": metrics,
        "profile": _profile(profile_path),
        "telemetry_summary": telemetry_summary,
        "population_sha256": telemetry_summary["population_sha256"],
        "routing_audit": dict(audit),
        "binding": dict(cfg.georoute_hybrid_causal_binding),
        "binding_sha256": str(
            cfg.georoute_hybrid_causal_binding["binding_sha256"]
        ),
        "input_receipts": {
            "source_config_sha256": str(
                cfg.georoute_hybrid_causal_binding["source_config_sha256"]
            ),
            "manifest_file_sha256": str(
                cfg.georoute_hybrid_causal_binding["manifest_file_sha256"]
            ),
            "development_annotation_sha256": str(
                cfg.georoute_hybrid_causal_binding["development_annotation"][
                    "sha256"
                ]
            ),
            "class_map_sha256": str(
                cfg.georoute_hybrid_causal_binding["class_map_sha256"]
            ),
            "pretrained_checkpoint_sha256": str(
                cfg.georoute_hybrid_causal_binding[
                    "pretrained_checkpoint_sha256"
                ]
            ),
            "development_video_root": str(
                cfg.georoute_hybrid_causal_binding["development_video_root"]
            ),
        },
        "config_receipts": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in config_paths.items()
        },
        "checkpoint_receipt": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "size_bytes": int(checkpoint.stat().st_size),
            "policy": "final_only_atomic",
        },
        "artifact_receipts": {
            "prediction": {
                "path": str(prediction.resolve()),
                "sha256": sha256_file(prediction),
            },
            "telemetry": {
                "path": str(telemetry.resolve()),
                "sha256": sha256_file(telemetry),
            },
            "profile": {
                "path": str(profile_path.resolve()),
                "sha256": sha256_file(profile_path),
            },
            "accuracy_log": {
                "path": str(accuracy_log.resolve()),
                "sha256": sha256_file(accuracy_log),
            },
        },
        "storage_receipt": _read_json(storage_receipt_path),
        "runtime_commit": runtime_commit,
        "rendezvous": validated_rendezvous,
        "profile_rendezvous": dict(profile_rendezvous),
        "parent_p0_suite": {
            "path": str(parent_suite_path.resolve()),
            "file_sha256": sha256_file(parent_suite_path),
            "suite_sha256": str(parent_suite["suite_sha256"]),
        },
        "single_seed_screen_only": True,
        "old_free_first_selector_reused": False,
        "partial_survivor_inference_allowed": False,
        "official_test_opened": False,
        "gt_for_route_used": False,
        "teacher_for_route_used": False,
        "raw_prediction_cache_used": False,
        "paper_claim_allowed": False,
    }
    result["stage_result_sha256"] = canonical_sha256(result)
    validate_hybrid_causal_stage_result(
        result,
        expected_arm=args.arm,
        expected_commit=runtime_commit,
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def _execute(args: argparse.Namespace, *, cell_root: Path) -> dict[str, Any]:
    expected_commit = str(args.expected_commit).lower()
    if _current_commit() != expected_commit:
        raise RuntimeError("Hybrid causal source differs from the bound commit")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status:
        raise RuntimeError("Hybrid causal stage requires a clean source snapshot")
    slurm_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    visible = [
        item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item
    ]
    if not slurm_job_id.isdigit() or len(visible) != WORLD_SIZE:
        raise RuntimeError("Hybrid causal stage requires two Slurm-visible GPUs")

    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()):
        raise ValueError("Hybrid causal run root leaves the write boundary")
    spec = hybrid_causal_arm_spec(args.arm)
    parent_suite_path = run_root / "control" / "hybrid_causal_p0_suite.json"
    parent_suite = _read_parent_p0_suite(
        parent_suite_path,
        expected_commit=expected_commit,
    )
    storage_profile = _read_json(
        run_root / "control" / "georoute_storage_profile.json"
    )
    storage_receipt = storage_capacity_receipt(
        run_root,
        cell_count=1,
        storage_profile=storage_profile,
        expected_commit=expected_commit,
    )

    cell_root.mkdir(parents=True, exist_ok=False)
    storage_receipt_path = cell_root / "storage_preflight.json"
    _atomic_write_json(storage_receipt_path, storage_receipt)
    cfg, config_paths = _dump_configs(args=args, cell_root=cell_root)
    inherited = dict(os.environ)
    inherited["PYTHONNOUSERSITE"] = "1"
    inherited["PYTHONDONTWRITEBYTECODE"] = "1"

    train_log = cell_root / "train.out"
    train_receipt = _run_torchrun(
        phase="train",
        stage="hybrid_causal",
        variant=args.arm,
        seed=HYBRID_CAUSAL_SEED,
        slurm_job_id=slurm_job_id,
        arguments=(
            "tools/train.py",
            str(config_paths["train"]),
            "--seed",
            str(HYBRID_CAUSAL_SEED),
            "--id",
            "0",
        ),
        log_path=train_log,
        env=inherited,
    )
    training_work = cell_root / "training" / "gpu2_id0"
    checkpoint = (
        training_work / "checkpoint" / f"epoch_{HYBRID_CAUSAL_EPOCHS - 1}.pth"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Hybrid causal final checkpoint is missing: {checkpoint}")
    payloads = sorted(checkpoint.parent.glob("*.pth"))
    temporaries = sorted(checkpoint.parent.glob("*.tmp*"))
    if payloads != [checkpoint] or temporaries:
        raise RuntimeError("Hybrid causal cell must retain exactly one final checkpoint")

    accuracy_log = cell_root / "accuracy.out"
    accuracy_receipt = _run_torchrun(
        phase="test",
        stage="hybrid_causal",
        variant=args.arm,
        seed=HYBRID_CAUSAL_SEED,
        slurm_job_id=slurm_job_id,
        arguments=(
            "tools/test.py",
            str(config_paths["accuracy"]),
            "--checkpoint",
            str(checkpoint),
            "--seed",
            str(HYBRID_CAUSAL_SEED),
            "--id",
            "0",
        ),
        log_path=accuracy_log,
        env=inherited,
    )
    accuracy_work = cell_root / "accuracy" / "gpu2_id0"
    prediction = accuracy_work / "result_detection.json"
    telemetry = accuracy_work / "georoute_diagnostic_telemetry.json"
    for path in (prediction, telemetry):
        if not path.is_file():
            raise FileNotFoundError(path)

    profile_log = cell_root / "profile.out"
    profile_receipt = _run_torchrun(
        phase="test",
        stage="hybrid_causal_profile",
        variant=args.arm,
        seed=HYBRID_CAUSAL_SEED,
        slurm_job_id=slurm_job_id,
        arguments=(
            "tools/test.py",
            str(config_paths["profile"]),
            "--checkpoint",
            str(checkpoint),
            "--seed",
            str(HYBRID_CAUSAL_SEED),
            "--id",
            "0",
        ),
        log_path=profile_log,
        env=inherited,
    )
    profile_path = (
        cell_root
        / "profile"
        / "gpu2_id0"
        / "georoute_development_profile.json"
    )
    if not profile_path.is_file():
        raise FileNotFoundError(profile_path)
    return _build_result(
        args=args,
        cfg=cfg,
        config_paths=config_paths,
        checkpoint=checkpoint,
        storage_receipt_path=storage_receipt_path,
        prediction=prediction,
        telemetry=telemetry,
        profile_path=profile_path,
        accuracy_log=accuracy_log,
        runtime_commit=expected_commit,
        rendezvous={"train": train_receipt, "test": accuracy_receipt},
        profile_rendezvous=profile_receipt,
        parent_suite_path=parent_suite_path,
        parent_suite=parent_suite,
    )


def main() -> int:
    args = _parse_args()
    hybrid_causal_arm_spec(args.arm)
    run_root = args.run_root.resolve()
    cell_root = run_root / hybrid_causal_cell_relative_path(
        arm=args.arm,
        seed=HYBRID_CAUSAL_SEED,
    )
    if cell_root.exists():
        raise FileExistsError("Hybrid causal cell exists; refusing overwrite or resume")
    try:
        result = _execute(args, cell_root=cell_root)
    except Exception as error:
        if cell_root.is_dir():
            trace = traceback.format_exc()
            failure: dict[str, Any] = {
                "schema_version": HYBRID_CAUSAL_STAGE_RESULT_SCHEMA,
                "status": "FAIL_EXPLORATORY_HYBRID_CAUSAL_ARM",
                "study_id": HYBRID_CAUSAL_STUDY_ID,
                "arm": args.arm,
                "seed": HYBRID_CAUSAL_SEED,
                "expected_runtime_commit": str(args.expected_commit).lower(),
                "observed_runtime_commit": _current_commit(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "exception_type": type(error).__name__,
                "exception_message": str(error)[:2000],
                "traceback_sha256": hashlib.sha256(
                    trace.encode("utf-8", errors="replace")
                ).hexdigest(),
                "official_test_opened": False,
                "paper_claim_allowed": False,
            }
            failure["failure_sha256"] = canonical_sha256(failure)
            _atomic_write_json(cell_root / "stage_failure.json", failure)
        raise
    _atomic_write_json(cell_root / "stage_result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
