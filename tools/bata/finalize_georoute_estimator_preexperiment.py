#!/usr/bin/env python3
"""Finalize the diagnostic GeoRoute D/K/M preexperiment without selection."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)


FINALIZATION_SCHEMA = "georoute_estimator_preexperiment_finalization_v1"
DEPLOYMENT_SCHEMA = "georoute_estimator_preexperiment_deployment_v1"
FROZEN_PHASE_M_VARIANTS = (
    "dense",
    "fixed",
    "fixed_geometry",
    "random",
    "free",
    "hybrid",
)
KAT_SCHEMA = "georoute_estimator_representation_kat_v1"
CENSUS_SCHEMA = "georoute_exact_index_decode_census_v1"
PHASE_M_SCHEMA = "georoute_phase_m_diagnostic_replay_v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _self_hash_matches(payload: Mapping[str, Any], field: str) -> bool:
    claimed = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return isinstance(claimed, str) and claimed == canonical_sha256(unsigned)


def _is_job_id(value: Any) -> bool:
    return isinstance(value, str) and value.isdigit()


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or f"git {' '.join(args)} failed"
        )
    return completed.stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_root = args.run_root.resolve()
    write_boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(run_root, write_boundary) or run_root == write_boundary:
        raise ValueError("preexperiment finalization left write boundary")
    finalizer_job_id = os.environ.get("SLURM_JOB_ID", "")
    if not _is_job_id(finalizer_job_id):
        raise RuntimeError("preexperiment finalizer must run inside Slurm")
    actual_commit = _git_output("rev-parse", "HEAD").lower()
    if actual_commit != args.expected_commit.lower():
        raise RuntimeError("preexperiment finalizer source mismatch")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("preexperiment finalizer requires clean source")

    deployment_path = run_root / "control" / "deployment.json"
    deployment = _read_json(deployment_path)
    if deployment.get("schema_version") != DEPLOYMENT_SCHEMA:
        raise RuntimeError("unexpected preexperiment deployment schema")
    if (
        deployment.get("status")
        != "SUBMITTED_D_K_AND_CONDITIONAL_PARALLEL_M"
        or not _self_hash_matches(deployment, "deployment_sha256")
    ):
        raise RuntimeError("invalid preexperiment deployment receipt")
    if deployment.get("runtime_commit") != actual_commit:
        raise RuntimeError("deployment/runtime commit mismatch")
    source_experiment_commit = str(
        deployment.get("source_experiment_commit", "")
    ).lower()
    if len(source_experiment_commit) != 40 or any(
        character not in "0123456789abcdef"
        for character in source_experiment_commit
    ):
        raise RuntimeError("deployment source experiment commit is invalid")
    variants = tuple(
        str(value) for value in deployment.get("phase_m_variants", [])
    )
    if variants != FROZEN_PHASE_M_VARIANTS:
        raise RuntimeError("deployment Phase M variant set is not frozen")
    source_cells = deployment.get("source_cells")
    jobs = deployment.get("jobs")
    if (
        not isinstance(source_cells, dict)
        or set(source_cells) != set(variants)
        or not isinstance(jobs, dict)
        or set(jobs) != {"kat", "decode_census", "phase_m"}
        or not _is_job_id(jobs.get("kat"))
        or not _is_job_id(jobs.get("decode_census"))
        or not isinstance(jobs.get("phase_m"), dict)
        or set(jobs["phase_m"]) != set(variants)
        or not all(_is_job_id(value) for value in jobs["phase_m"].values())
    ):
        raise RuntimeError("deployment job/source-cell binding is invalid")
    predecessor_job_ids = [
        jobs["kat"],
        jobs["decode_census"],
        *jobs["phase_m"].values(),
    ]
    if len(set(predecessor_job_ids)) != len(predecessor_job_ids):
        raise RuntimeError("deployment contains duplicate predecessor job IDs")
    expected_dependency_policy = {
        "kat_and_decode_parallel": True,
        "phase_m_after_both_pass": True,
        "phase_m_parallel": True,
        "finalizer_after_any_terminal_state": True,
    }
    if (
        deployment.get("dependency_policy") != expected_dependency_policy
        or deployment.get("training_launched") is not False
        or deployment.get("old_selector_reused") is not False
        or deployment.get("p2_p3_opened") is not False
        or deployment.get("official_test_opened") is not False
        or deployment.get("paper_claim_allowed") is not False
    ):
        raise RuntimeError("deployment policy receipt is invalid")

    finalizer_submission_path = (
        run_root / "control" / "finalizer_submission.json"
    )
    finalizer_submission = _read_json(finalizer_submission_path)
    if (
        finalizer_submission.get("schema_version") != DEPLOYMENT_SCHEMA
        or finalizer_submission.get("status")
        != "SUBMITTED_FINALIZER_AFTERANY"
        or finalizer_submission.get("runtime_commit") != actual_commit
        or finalizer_submission.get("deployment_file_sha256")
        != sha256_file(deployment_path)
        or finalizer_submission.get("finalizer_job_id") != finalizer_job_id
        or finalizer_submission.get("dependency_type") != "afterany"
        or set(finalizer_submission.get("predecessor_job_ids", []))
        != set(predecessor_job_ids)
        or not _self_hash_matches(finalizer_submission, "receipt_sha256")
    ):
        raise RuntimeError("finalizer submission receipt is invalid")

    census_path = run_root / "control" / "decode_census.json"
    kat_path = run_root / "control" / "estimator_representation_kat.json"
    census = _read_json(census_path) if census_path.is_file() else None
    kat = _read_json(kat_path) if kat_path.is_file() else None
    census_passed = bool(
        census
        and census.get("schema_version") == CENSUS_SCHEMA
        and census.get("status") == "PASS_DECODE_CENSUS"
        and census.get("runtime_commit") == actual_commit
        and census.get("source_experiment_commit")
        == source_experiment_commit
        and census.get("slurm_job_id") == jobs["decode_census"]
        and int(census.get("dataset_count", 0)) > 0
        and int(census.get("failure_count", -1)) == 0
        and int(census.get("successful_item_retrievals", -1))
        == int(census.get("expected_item_retrievals", -2))
        and census.get("official_test_opened") is False
        and census.get("gt_for_route_used") is False
        and census.get("raw_prediction_cache_used") is False
        and _self_hash_matches(census, "receipt_sha256")
    )
    kat_passed = bool(
        kat
        and kat.get("schema_version") == KAT_SCHEMA
        and kat.get("status") == "PASS_MECHANICAL_ONLY"
        and kat.get("runtime_commit") == actual_commit
        and kat.get("slurm_job_id") == jobs["kat"]
        and isinstance(kat.get("checks"), dict)
        and kat["checks"]
        and all(
            isinstance(check, dict) and check.get("passed") is True
            for check in kat["checks"].values()
        )
        and kat.get("development_metric_emitted") is False
        and kat.get("official_test_opened") is False
        and kat.get("paper_claim_allowed") is False
        and _self_hash_matches(kat, "receipt_sha256")
    )

    phase_m = {}
    all_phase_m_passed = True
    passed_population_hashes = set()
    for variant in variants:
        cell = run_root / "phase_m" / variant
        result_path = cell / "phase_m_result.json"
        failure_path = cell / "phase_m_failure.json"
        if result_path.is_file() and failure_path.is_file():
            passed = False
            phase_m[variant] = {
                "status": "AMBIGUOUS_RESULT_AND_FAILURE",
                "passed": False,
            }
        elif result_path.is_file():
            result = _read_json(result_path)
            source_artifacts = result.get("source_artifacts")
            source_cell = source_cells[variant]
            source_binding_passed = bool(
                isinstance(source_artifacts, dict)
                and source_artifacts.get("source_run_root")
                == deployment.get("source_run_root")
                and source_artifacts.get("bound_config_path")
                == source_cell.get("source_config")
                and source_artifacts.get("bound_config_sha256")
                == source_cell.get("source_config_sha256")
                and source_artifacts.get("checkpoint_path")
                == source_cell.get("source_checkpoint")
                and source_artifacts.get("checkpoint_sha256")
                == source_cell.get("source_checkpoint_sha256")
                and source_artifacts.get("prediction_path")
                == source_cell.get("source_prediction")
                and source_artifacts.get("prediction_sha256")
                == source_cell.get("source_prediction_sha256")
            )
            passed = bool(
                result.get("schema_version") == PHASE_M_SCHEMA
                and result.get("status") == "PASS_DIAGNOSTIC_ONLY"
                and result.get("variant") == variant
                and int(result.get("seed", -1)) == 3407
                and result.get("source_experiment_commit")
                == source_experiment_commit
                and result.get("runtime_commit") == actual_commit
                and result.get("slurm_job_id") == jobs["phase_m"][variant]
                and source_binding_passed
                and result.get("prediction_sha256_parity") is True
                and result.get("telemetry_population_complete") is True
                and int(result.get("dataset_count", 0)) > 0
                and int(result.get("record_count", -1))
                == int(result.get("dataset_count", -2))
                and result.get("instrumentation_only") is True
                and result.get("official_test_opened") is False
                and result.get("gt_for_route_used") is False
                and result.get("teacher_for_route_used") is False
                and result.get("raw_prediction_cache_used") is False
                and result.get("old_selector_completed") is False
                and result.get("paper_claim_allowed") is False
                and _self_hash_matches(result, "result_sha256")
            )
            if passed:
                passed_population_hashes.add(result.get("population_sha256"))
            phase_m[variant] = {
                "status": result.get("status"),
                "passed": passed,
                "result_path": str(result_path),
                "result_file_sha256": sha256_file(result_path),
                "population_sha256": result.get("population_sha256"),
            }
        elif failure_path.is_file():
            failure = _read_json(failure_path)
            passed = False
            phase_m[variant] = {
                "status": failure.get("status"),
                "passed": False,
                "failure_path": str(failure_path),
                "failure_file_sha256": sha256_file(failure_path),
                "exception_type": failure.get("exception_type"),
                "receipt_self_hash_valid": _self_hash_matches(
                    failure, "failure_sha256"
                ),
            }
        else:
            passed = False
            phase_m[variant] = {
                "status": "MISSING_DEPENDENCY_OR_ARTIFACT",
                "passed": False,
            }
        all_phase_m_passed = all_phase_m_passed and passed
    population_consistent = (
        len(passed_population_hashes) == 1
        if all_phase_m_passed
        else False
    )
    all_phase_m_passed = all_phase_m_passed and population_consistent

    if not census_passed:
        decision = "STOP_DECODE"
    elif not kat_passed:
        decision = "STOP_ESTIMATOR"
    elif not all_phase_m_passed:
        decision = "STOP_INSTRUMENTATION"
    else:
        decision = "GO_PILOT_DESIGN_ONLY"
    finalization: dict[str, Any] = {
        "schema_version": FINALIZATION_SCHEMA,
        "status": "COMPLETE_DIAGNOSTIC_PREEXPERIMENT",
        "decision": decision,
        "runtime_commit": actual_commit,
        "source_experiment_commit": source_experiment_commit,
        "deployment_path": str(deployment_path),
        "deployment_sha256": sha256_file(deployment_path),
        "finalizer_submission_path": str(finalizer_submission_path),
        "finalizer_submission_sha256": sha256_file(
            finalizer_submission_path
        ),
        "decode_census": {
            "passed": census_passed,
            "path": str(census_path),
            "file_sha256": (
                sha256_file(census_path) if census_path.is_file() else None
            ),
        },
        "estimator_representation_kat": {
            "passed": kat_passed,
            "path": str(kat_path),
            "file_sha256": (
                sha256_file(kat_path) if kat_path.is_file() else None
            ),
        },
        "phase_m": phase_m,
        "phase_m_population_consistent": population_consistent,
        "all_phase_m_passed": all_phase_m_passed,
        "training_launched": False,
        "old_selector_reused": False,
        "new_selector_emitted": False,
        "p2_p3_opened": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    finalization["finalization_sha256"] = canonical_sha256(
        finalization
    )
    output = run_root / "control" / "finalization.json"
    if output.exists():
        raise FileExistsError("preexperiment finalization already exists")
    _atomic_write_json(output, finalization)
    print(json.dumps(finalization, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
