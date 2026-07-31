#!/usr/bin/env python3
"""Seal the 15-cell GeoRoute development matrix and apply its frozen selector."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_estimator_pilot_stage_runner import (  # noqa: E402
    _pilot_profile,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_official_comparable_contract import (  # noqa: E402
    FORMAL_DEVELOPMENT_ARM_ORDER,
    FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA,
    FORMAL_DEVELOPMENT_FINALIZATION_SCHEMA,
    FORMAL_DEVELOPMENT_SEEDS,
    formal_cell_relative_path,
    read_json,
    validate_formal_checkpoint_sidecar,
    validate_protocol_manifest,
)
from tools.bata.georoute_official_development_stage_runner import (  # noqa: E402
    summarize_formal_telemetry,
    validate_formal_stage_result,
)


SELECTOR_ARMS = ("residual_st_rep_off", "residual_pl_rep_off")
CONTROL_ARMS = ("fixed_lattice", "random")
ACCURACY_KEY = "high_iou_composite"
COST_KEY = "model_and_postprocess_p50_ms"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _self_hash_matches(payload: Mapping[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or f"git {' '.join(arguments)} failed"
        )
    return completed.stdout.strip()


def _validate_artifacts(
    result: Mapping[str, Any],
    *,
    cell_root: Path,
) -> None:
    for path_field, hash_field in (
        ("config_path", "config_sha256"),
        ("prediction_path", "prediction_sha256"),
        ("test_log_path", "test_log_sha256"),
    ):
        path = Path(str(result.get(path_field, ""))).resolve()
        if (
            not path.is_file()
            or not _inside(path, cell_root)
            or sha256_file(path) != result.get(hash_field)
        ):
            raise ValueError(f"formal artifact changed: {path_field}")
    profile_path = Path(str(result.get("profile_path", ""))).resolve()
    telemetry_path = Path(str(result.get("telemetry_path", ""))).resolve()
    if (
        not profile_path.is_file()
        or not telemetry_path.is_file()
        or not _inside(profile_path, cell_root)
        or not _inside(telemetry_path, cell_root)
        or _pilot_profile(profile_path) != result.get("profile")
        or summarize_formal_telemetry(
            telemetry_path,
            arm=str(result["arm"]),
        )
        != result.get("telemetry_summary")
    ):
        raise ValueError("formal profile or telemetry receipt changed")
    checkpoint = result.get("checkpoint_receipt")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("formal stage lacks checkpoint receipt")
    checkpoint_path = Path(str(checkpoint.get("path", ""))).resolve()
    sidecar_path = Path(str(checkpoint.get("sidecar_path", ""))).resolve()
    binding = result["binding"]
    sidecar = validate_formal_checkpoint_sidecar(
        checkpoint_path,
        binding=binding,
    )
    if (
        not _inside(checkpoint_path, cell_root)
        or sidecar_path != Path(str(checkpoint_path) + ".metadata.json")
        or checkpoint.get("sha256") != sha256_file(checkpoint_path)
        or int(checkpoint.get("size_bytes", -1))
        != int(checkpoint_path.stat().st_size)
        or checkpoint.get("sidecar_file_sha256")
        != sha256_file(sidecar_path)
        or checkpoint.get("sidecar_sha256")
        != sidecar["sidecar_sha256"]
        or sorted(checkpoint_path.parent.glob("*.pth"))
        != [checkpoint_path]
        or sorted(checkpoint_path.parent.glob("*.metadata.json"))
        != [sidecar_path]
        or sorted(checkpoint_path.parent.glob("*.tmp*"))
    ):
        raise ValueError("formal checkpoint artifact receipt changed")


def _aggregate(values: Sequence[float]) -> dict[str, float]:
    normalized = [float(value) for value in values]
    if len(normalized) != len(FORMAL_DEVELOPMENT_SEEDS) or any(
        not math.isfinite(value) for value in normalized
    ):
        raise ValueError("formal aggregate requires three finite seed values")
    return {
        "mean": statistics.fmean(normalized),
        "population_sd": statistics.pstdev(normalized),
        "minimum": min(normalized),
        "maximum": max(normalized),
    }


def _selector_eligibility(
    results: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    arm: str,
) -> dict[str, Any]:
    paired: dict[str, Any] = {}
    passed = True
    for seed in FORMAL_DEVELOPMENT_SEEDS:
        treatment = results[arm][seed]
        accuracy = float(treatment["metrics"][ACCURACY_KEY])
        cost = float(treatment["profile"][COST_KEY])
        fixed_delta = accuracy - float(
            results["fixed_lattice"][seed]["metrics"][ACCURACY_KEY]
        )
        random_delta = accuracy - float(
            results["random"][seed]["metrics"][ACCURACY_KEY]
        )
        dense_cost_delta = cost - float(
            results["dense_native"][seed]["profile"][COST_KEY]
        )
        seed_passed = (
            fixed_delta > 0.0
            and random_delta > 0.0
            and dense_cost_delta < 0.0
        )
        passed = passed and seed_passed
        paired[str(seed)] = {
            "high_iou_delta_vs_fixed": fixed_delta,
            "high_iou_delta_vs_random": random_delta,
            "development_cost_delta_vs_dense_ms": dense_cost_delta,
            "passed": seed_passed,
        }
    return {
        "arm": arm,
        "paired_seed_checks": paired,
        "all_three_seeds_passed": passed,
    }


def _strict_pareto_dominates(
    results: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    treatment: str,
    control: str,
) -> dict[str, Any]:
    accuracy_deltas = [
        float(results[treatment][seed]["metrics"][ACCURACY_KEY])
        - float(results[control][seed]["metrics"][ACCURACY_KEY])
        for seed in FORMAL_DEVELOPMENT_SEEDS
    ]
    cost_deltas = [
        float(results[treatment][seed]["profile"][COST_KEY])
        - float(results[control][seed]["profile"][COST_KEY])
        for seed in FORMAL_DEVELOPMENT_SEEDS
    ]
    dominates = (
        all(delta >= 0.0 for delta in accuracy_deltas)
        and all(delta <= 0.0 for delta in cost_deltas)
        and statistics.fmean(accuracy_deltas) > 0.0
        and statistics.fmean(cost_deltas) < 0.0
    )
    return {
        "treatment": treatment,
        "control": control,
        "paired_accuracy_deltas": accuracy_deltas,
        "paired_cost_deltas_ms": cost_deltas,
        "mean_accuracy_delta": statistics.fmean(accuracy_deltas),
        "mean_cost_delta_ms": statistics.fmean(cost_deltas),
        "strict_pareto_dominates": dominates,
    }


def finalize_results(
    *,
    run_root: Path,
    expected_commit: str,
    stage_jobs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    valid: dict[str, dict[int, dict[str, Any]]] = {
        arm: {} for arm in FORMAL_DEVELOPMENT_ARM_ORDER
    }
    failures: dict[str, Any] = {}
    population_hashes = set()
    checkpoint_paths = set()
    slurm_ids = set()
    for arm in FORMAL_DEVELOPMENT_ARM_ORDER:
        for seed in FORMAL_DEVELOPMENT_SEEDS:
            cell_root = run_root / formal_cell_relative_path(
                arm=arm,
                seed=seed,
            )
            result_path = cell_root / "stage_result.json"
            failure_path = cell_root / "stage_failure.json"
            key = f"{arm}/seed{seed}"
            if result_path.is_file() and failure_path.is_file():
                failures[key] = {"status": "AMBIGUOUS_RESULT_AND_FAILURE"}
                continue
            if failure_path.is_file():
                failure = read_json(failure_path)
                failures[key] = {
                    "status": failure.get("status"),
                    "exception_type": failure.get("exception_type"),
                    "exception_message": failure.get("exception_message"),
                    "failure_path": str(failure_path),
                    "failure_file_sha256": sha256_file(failure_path),
                    "failure_self_hash_valid": _self_hash_matches(
                        failure, field="failure_sha256"
                    ),
                }
                continue
            if not result_path.is_file():
                failures[key] = {"status": "MISSING_STAGE_RESULT"}
                continue
            try:
                result = validate_formal_stage_result(
                    read_json(result_path),
                    expected_arm=arm,
                    expected_seed=seed,
                    expected_commit=expected_commit,
                )
                _validate_artifacts(result, cell_root=cell_root)
                expected_job = str(stage_jobs[arm][str(seed)])
                train_job = str(
                    result["rendezvous"]["train"]["slurm_job_id"]
                )
                test_job = str(
                    result["rendezvous"]["test"]["slurm_job_id"]
                )
                if train_job != expected_job or test_job != expected_job:
                    raise ValueError(
                        "formal stage Slurm receipt differs from deployment"
                    )
                population_hashes.add(
                    result["telemetry_summary"]["population_sha256"]
                )
                checkpoint_paths.add(
                    result["checkpoint_receipt"]["path"]
                )
                slurm_ids.add(train_job)
                valid[arm][seed] = {
                    **result,
                    "result_path": str(result_path),
                    "result_file_sha256": sha256_file(result_path),
                }
            except Exception as error:
                failures[key] = {
                    "status": "INVALID_STAGE_RESULT",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "result_path": str(result_path),
                    "result_file_sha256": sha256_file(result_path),
                }
    all_passed = (
        not failures
        and all(
            set(valid[arm]) == set(FORMAL_DEVELOPMENT_SEEDS)
            for arm in FORMAL_DEVELOPMENT_ARM_ORDER
        )
        and len(population_hashes) == 1
        and len(checkpoint_paths) == 15
        and len(slurm_ids) == 15
    )
    if not all_passed:
        finalization: dict[str, Any] = {
            "schema_version": FORMAL_DEVELOPMENT_FINALIZATION_SCHEMA,
            "status": "INCOMPLETE_OFFICIAL_COMPARABLE_DEVELOPMENT_MATRIX",
            "decision": "DEVELOPMENT_MATRIX_INCOMPLETE_NO_PERFORMANCE_INFERENCE",
            "runtime_commit": expected_commit,
            "completed_cells": sum(len(cells) for cells in valid.values()),
            "expected_cells": 15,
            "failures": failures,
            "arm_seed_results": {},
            "aggregates": {},
            "paired_contrasts": {},
            "selector_decision": {},
            "all_fifteen_cells_passed": False,
            "development_selection_inference_allowed": False,
            "sealed_official_test_protocol_implementation_authorized": False,
            "official_protocol_freeze_authorized": False,
            "official_test_open_authorized": False,
            "official_test_opened": False,
            "paper_grade_result_record_emitted": False,
            "paper_claim_allowed": False,
        }
        finalization["finalization_sha256"] = canonical_sha256(finalization)
        return finalization

    compact: dict[str, dict[str, Any]] = {}
    aggregates: dict[str, Any] = {}
    for arm in FORMAL_DEVELOPMENT_ARM_ORDER:
        compact[arm] = {}
        for seed in FORMAL_DEVELOPMENT_SEEDS:
            result = valid[arm][seed]
            compact[arm][str(seed)] = {
                "metrics": dict(result["metrics"]),
                "profile": dict(result["profile"]),
                "telemetry_summary": dict(result["telemetry_summary"]),
                "checkpoint_receipt": dict(result["checkpoint_receipt"]),
                "stage_result_sha256": result["stage_result_sha256"],
                "result_file_sha256": result["result_file_sha256"],
                "slurm_job_id": result["rendezvous"]["train"][
                    "slurm_job_id"
                ],
            }
        aggregate_metrics = {
            metric: _aggregate(
                [
                    valid[arm][seed]["metrics"][metric]
                    for seed in FORMAL_DEVELOPMENT_SEEDS
                ]
            )
            for metric in valid[arm][FORMAL_DEVELOPMENT_SEEDS[0]][
                "metrics"
            ]
        }
        aggregate_cost = {
            key: _aggregate(
                [
                    valid[arm][seed]["profile"][key]
                    for seed in FORMAL_DEVELOPMENT_SEEDS
                ]
            )
            for key in (
                "model_and_postprocess_p50_ms",
                "model_and_postprocess_p95_ms",
                "window_wall_p50_ms",
                "window_wall_p95_ms",
                "peak_allocated_mb",
            )
        }
        aggregates[arm] = {
            "metrics": aggregate_metrics,
            "development_cost": aggregate_cost,
        }
    eligibility = {
        arm: _selector_eligibility(valid, arm=arm)
        for arm in SELECTOR_ARMS
    }
    st_over_pl = _strict_pareto_dominates(
        valid,
        treatment=SELECTOR_ARMS[0],
        control=SELECTOR_ARMS[1],
    )
    pl_over_st = _strict_pareto_dominates(
        valid,
        treatment=SELECTOR_ARMS[1],
        control=SELECTOR_ARMS[0],
    )
    selected_arm = None
    if (
        eligibility[SELECTOR_ARMS[0]]["all_three_seeds_passed"]
        and st_over_pl["strict_pareto_dominates"]
        and not pl_over_st["strict_pareto_dominates"]
    ):
        selected_arm = SELECTOR_ARMS[0]
    elif (
        eligibility[SELECTOR_ARMS[1]]["all_three_seeds_passed"]
        and pl_over_st["strict_pareto_dominates"]
        and not st_over_pl["strict_pareto_dominates"]
    ):
        selected_arm = SELECTOR_ARMS[1]
    authorized = selected_arm is not None
    finalization = {
        "schema_version": FORMAL_DEVELOPMENT_FINALIZATION_SCHEMA,
        "status": "COMPLETE_OFFICIAL_COMPARABLE_DEVELOPMENT_MATRIX",
        "decision": (
            "DEVELOPMENT_METHOD_FREEZE_CANDIDATE_AUTHORIZED"
            if authorized
            else "DEVELOPMENT_SELECTION_HOLD_NO_OFFICIAL_TEST"
        ),
        "runtime_commit": expected_commit,
        "completed_cells": 15,
        "expected_cells": 15,
        "failures": {},
        "arm_seed_results": compact,
        "aggregates": aggregates,
        "paired_contrasts": {
            "selector_eligibility": eligibility,
            "st_over_pl": st_over_pl,
            "pl_over_st": pl_over_st,
        },
        "selector_decision": {
            "selected_arm": selected_arm,
            "accuracy_metric": ACCURACY_KEY,
            "development_cost_metric": COST_KEY,
            "all_seed_control_improvement_required": True,
            "strict_pareto_dominance_required": True,
            "geometry_zoom_allowed": False,
        },
        "all_fifteen_cells_passed": True,
        "cross_cell_population_consistent": True,
        "development_selection_inference_allowed": True,
        "sealed_official_test_protocol_implementation_authorized": authorized,
        "official_protocol_freeze_authorized": False,
        "official_test_open_authorized": False,
        "official_test_opened": False,
        "development_metrics_are_paper_results": False,
        "paper_grade_result_record_emitted": False,
        "paper_claim_allowed": False,
    }
    finalization["finalization_sha256"] = canonical_sha256(finalization)
    return finalization


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def _run_main(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(run_root, boundary):
        raise ValueError("formal finalizer root leaves remote boundary")
    expected_commit = str(args.expected_commit).lower()
    if _git_output("rev-parse", "HEAD").lower() != expected_commit:
        raise RuntimeError("formal finalizer source commit changed")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("formal finalizer requires a clean source snapshot")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if not slurm_job_id.isdigit():
        raise RuntimeError("formal finalizer requires Slurm")
    deployment_path = run_root / "control" / "deployment.json"
    deployment = read_json(deployment_path)
    jobs = deployment.get("jobs")
    stage_jobs = jobs.get("stage") if isinstance(jobs, Mapping) else None
    if (
        deployment.get("schema_version")
        != FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA
        or deployment.get("status")
        != "SUBMITTED_OFFICIAL_COMPARABLE_DEVELOPMENT_MATRIX"
        or deployment.get("runtime_commit") != expected_commit
        or not _self_hash_matches(deployment, field="deployment_sha256")
        or not isinstance(jobs, Mapping)
        or not isinstance(stage_jobs, Mapping)
        or str(jobs.get("finalizer", "")) != slurm_job_id
    ):
        raise RuntimeError("formal deployment receipt is invalid")
    submission_path = run_root / "control" / "finalizer_submission.json"
    submission = read_json(submission_path)
    predecessor_ids = [
        str(stage_jobs[arm][str(seed)])
        for arm in FORMAL_DEVELOPMENT_ARM_ORDER
        for seed in FORMAL_DEVELOPMENT_SEEDS
    ]
    if (
        submission.get("schema_version")
        != FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA
        or submission.get("status")
        != "SUBMITTED_DEVELOPMENT_FINALIZER_AFTERANY"
        or submission.get("runtime_commit") != expected_commit
        or submission.get("deployment_file_sha256")
        != sha256_file(deployment_path)
        or submission.get("finalizer_job_id") != slurm_job_id
        or submission.get("dependency_type") != "afterany"
        or set(submission.get("predecessor_job_ids", ()))
        != set(predecessor_ids)
        or not _self_hash_matches(submission, field="receipt_sha256")
    ):
        raise RuntimeError("formal finalizer submission receipt is invalid")
    protocol_path = run_root / "control" / "protocol_manifest.json"
    protocol = validate_protocol_manifest(read_json(protocol_path))
    if (
        sha256_file(protocol_path)
        != deployment.get("protocol_manifest_file_sha256")
        or protocol.get("protocol_sha256")
        != deployment.get("protocol_sha256")
    ):
        raise RuntimeError("formal protocol manifest changed")
    finalization = finalize_results(
        run_root=run_root,
        expected_commit=expected_commit,
        stage_jobs=stage_jobs,
    )
    finalization["deployment_path"] = str(deployment_path)
    finalization["deployment_file_sha256"] = sha256_file(deployment_path)
    finalization["finalizer_submission_path"] = str(submission_path)
    finalization["finalizer_submission_file_sha256"] = sha256_file(
        submission_path
    )
    finalization["protocol_manifest_path"] = str(protocol_path)
    finalization["protocol_manifest_file_sha256"] = sha256_file(protocol_path)
    finalization.pop("finalization_sha256")
    finalization["finalization_sha256"] = canonical_sha256(finalization)
    output = run_root / "control" / "finalization.json"
    if output.exists():
        raise FileExistsError("formal finalization already exists")
    _atomic_write_json(output, finalization)
    print(json.dumps(finalization, indent=2, sort_keys=True))
    return 0 if finalization["all_fifteen_cells_passed"] else 1


def _write_failsafe(args: argparse.Namespace, error: BaseException) -> None:
    run_root = args.run_root.resolve()
    boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(run_root, boundary):
        return
    output = run_root / "control" / "finalization.json"
    if output.exists():
        return
    trace = traceback.format_exc()
    payload: dict[str, Any] = {
        "schema_version": FORMAL_DEVELOPMENT_FINALIZATION_SCHEMA,
        "status": "FAILED_OFFICIAL_COMPARABLE_DEVELOPMENT_FINALIZER",
        "decision": "DEVELOPMENT_MATRIX_INCOMPLETE_NO_PERFORMANCE_INFERENCE",
        "expected_runtime_commit": str(args.expected_commit).lower(),
        "observed_runtime_commit": (
            _git_output("rev-parse", "HEAD").lower()
            if (ROOT / ".git").exists()
            else None
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "exception_type": type(error).__name__,
        "exception_message": str(error)[:2000],
        "traceback_sha256": __import__("hashlib").sha256(
            trace.encode("utf-8", errors="replace")
        ).hexdigest(),
        "arm_seed_results": {},
        "aggregates": {},
        "paired_contrasts": {},
        "development_selection_inference_allowed": False,
        "sealed_official_test_protocol_implementation_authorized": False,
        "official_test_open_authorized": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    payload["finalization_sha256"] = canonical_sha256(payload)
    _atomic_write_json(output, payload)


def main() -> int:
    args = _parse_args()
    try:
        return _run_main(args)
    except BaseException as error:
        try:
            _write_failsafe(args, error)
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
