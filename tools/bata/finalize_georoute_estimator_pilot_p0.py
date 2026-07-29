#!/usr/bin/env python3
"""Seal the six mechanical P0 leaves for the estimator pilot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.finalize_georoute_estimator_preexperiment import (  # noqa: E402
    FINALIZATION_SCHEMA as PREEXPERIMENT_FINALIZATION_SCHEMA,
)
from tools.bata.georoute_estimator_pilot_contract import (  # noqa: E402
    PILOT_ARM_ORDER,
    PILOT_DEPLOYMENT_SCHEMA,
    PILOT_K,
    PILOT_P0_FAILURE_SCHEMA,
    PILOT_P0_SUITE_SCHEMA,
    PILOT_STUDY_ID,
    pilot_arm_spec,
    validate_pilot_job_receipt,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_rendezvous_gate import (  # noqa: E402
    validate_rendezvous_gate_receipt,
)
from tools.bata.run_georoute_p0_gate import (  # noqa: E402
    validate_p0_gate_report,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
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
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


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


def _validate_preexperiment_parent(
    *,
    path: Path,
    expected_path_sha256: str,
    expected_runtime_commit: str,
    expected_source_experiment_commit: str,
    expected_finalization_sha256: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("estimator pilot parent finalization is missing")
    if sha256_file(path) != expected_path_sha256:
        raise RuntimeError("estimator pilot parent finalization file hash changed")
    payload = _read_json(path)
    if (
        payload.get("schema_version") != PREEXPERIMENT_FINALIZATION_SCHEMA
        or payload.get("status") != "COMPLETE_DIAGNOSTIC_PREEXPERIMENT"
        or payload.get("decision") != "GO_PILOT_DESIGN_ONLY"
        or payload.get("runtime_commit") != expected_runtime_commit
        or payload.get("source_experiment_commit")
        != expected_source_experiment_commit
        or payload.get("finalization_sha256")
        != expected_finalization_sha256
        or payload.get("phase_m_population_consistent") is not True
        or payload.get("all_phase_m_passed") is not True
        or payload.get("training_launched") is not False
        or payload.get("old_selector_reused") is not False
        or payload.get("p2_p3_opened") is not False
        or payload.get("official_test_opened") is not False
        or payload.get("paper_claim_allowed") is not False
        or not _self_hash_matches(payload, "finalization_sha256")
    ):
        raise RuntimeError(
            "estimator pilot parent did not authorize pilot design only"
        )
    return payload


def _storage_profile(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    runtime_commit: str,
) -> dict[str, Any]:
    measurements = [
        report.get("checkpoint_storage_measurement")
        for report in reports.values()
    ]
    if any(not isinstance(value, Mapping) for value in measurements):
        raise ValueError("estimator pilot P0 lacks storage measurements")
    return {
        "schema_version": "georoute_storage_profile_v1",
        "runtime_commit": runtime_commit,
        "checkpoint_policy": "final_only",
        "checkpoint_upper_bound_bytes": max(
            int(value["checkpoint_upper_bound_bytes"])
            for value in measurements
        ),
        "peak_checkpoint_copies_per_cell": max(
            int(value["peak_checkpoint_copies_per_cell"])
            for value in measurements
        ),
        "auxiliary_upper_bound_bytes_per_cell": max(
            int(value["auxiliary_upper_bound_bytes_per_cell"])
            for value in measurements
        ),
        "stage_fixed_overhead_bytes": max(
            int(value["stage_fixed_overhead_bytes"])
            for value in measurements
        ),
        "safety_fraction": max(
            float(value["safety_fraction"]) for value in measurements
        ),
        "safety_bytes": max(
            int(value["safety_bytes"]) for value in measurements
        ),
        "measurement_provenance": {
            arm: {
                "report_sha256": report["report_sha256"],
                "measurement_method": report[
                    "checkpoint_storage_measurement"
                ]["measurement_method"],
            }
            for arm, report in reports.items()
        },
    }


def finalize_pilot_p0(
    *,
    report_paths: Mapping[str, Path],
    expected_commit: str,
    parent_finalization_path: Path,
    parent_finalization_file_sha256: str,
    expected_parent_runtime_commit: str,
    expected_source_experiment_commit: str,
    expected_parent_finalization_sha256: str,
    expected_jobs: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if tuple(report_paths) != PILOT_ARM_ORDER:
        raise ValueError("estimator pilot P0 report set/order is not frozen")
    reports: dict[str, dict[str, Any]] = {}
    rendezvous_receipts: dict[str, dict[str, Any]] = {}
    slurm_ids = set()
    for arm, path in report_paths.items():
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        report = _read_json(path)
        validate_p0_gate_report(report)
        if (
            report.get("pilot_arm") != arm
            or report.get("runtime_commit") != expected_commit
            or int(report.get("exact_k", {}).get("target_k", -1)) != PILOT_K
        ):
            raise ValueError(f"invalid estimator pilot P0 binding for {arm}")
        spec = pilot_arm_spec(arm)
        if (
            report.get("route_mode") != spec["route_mode"]
            or report.get("estimator", {}).get("name")
            != spec["policy_estimator"]
        ):
            raise ValueError(f"estimator pilot P0 route mismatch for {arm}")
        if expected_jobs is not None and report.get("slurm_job_id") != str(
            expected_jobs[arm]
        ):
            raise ValueError(f"estimator pilot P0 Slurm binding mismatch: {arm}")
        rendezvous_path = path.with_name(f"{path.stem}.rendezvous.json")
        if rendezvous_path.is_symlink() or not rendezvous_path.is_file():
            raise FileNotFoundError(rendezvous_path)
        rendezvous = _read_json(rendezvous_path)
        validate_rendezvous_gate_receipt(
            rendezvous,
            expected_commit=expected_commit,
        )
        binding = report.get("rendezvous_isolation")
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != str(rendezvous_path.resolve())
            or binding.get("file_sha256") != sha256_file(rendezvous_path)
            or binding.get("gate_sha256") != rendezvous.get("gate_sha256")
            or binding.get("slurm_job_id") != rendezvous.get("slurm_job_id")
            or report.get("slurm_job_id") != rendezvous.get("slurm_job_id")
        ):
            raise ValueError(f"P0 report/rendezvous binding mismatch: {arm}")
        slurm_ids.add(str(report["slurm_job_id"]))
        reports[arm] = report
        rendezvous_receipts[arm] = rendezvous
    if len(slurm_ids) != len(PILOT_ARM_ORDER):
        raise ValueError("estimator pilot P0 leaves must use distinct Slurm jobs")

    parent = _validate_preexperiment_parent(
        path=parent_finalization_path,
        expected_path_sha256=parent_finalization_file_sha256,
        expected_runtime_commit=expected_parent_runtime_commit,
        expected_source_experiment_commit=expected_source_experiment_commit,
        expected_finalization_sha256=expected_parent_finalization_sha256,
    )
    storage_profile = _storage_profile(
        reports,
        runtime_commit=expected_commit,
    )
    suite: dict[str, Any] = {
        "schema_version": PILOT_P0_SUITE_SCHEMA,
        "status": "PASS_MECHANICAL_ONLY",
        "study_id": PILOT_STUDY_ID,
        "runtime_commit": expected_commit,
        "arms": list(PILOT_ARM_ORDER),
        "reports": {
            arm: {
                "path": str(report_paths[arm].resolve()),
                "file_sha256": sha256_file(report_paths[arm]),
                "report_sha256": reports[arm]["report_sha256"],
                "slurm_job_id": reports[arm]["slurm_job_id"],
            }
            for arm in PILOT_ARM_ORDER
        },
        "rendezvous_isolation": {
            "status": "PASS_CONCURRENT_RENDEZVOUS_ISOLATION",
            "receipt_count": len(rendezvous_receipts),
            "distinct_slurm_job_count": len(slurm_ids),
            "receipts": {
                arm: {
                    "path": str(
                        report_paths[arm]
                        .with_name(f"{report_paths[arm].stem}.rendezvous.json")
                        .resolve()
                    ),
                    "file_sha256": sha256_file(
                        report_paths[arm].with_name(
                            f"{report_paths[arm].stem}.rendezvous.json"
                        )
                    ),
                    "gate_sha256": rendezvous_receipts[arm]["gate_sha256"],
                    "slurm_job_id": rendezvous_receipts[arm]["slurm_job_id"],
                }
                for arm in PILOT_ARM_ORDER
            },
        },
        "preexperiment_parent": {
            "path": str(parent_finalization_path.resolve()),
            "file_sha256": parent_finalization_file_sha256,
            "finalization_sha256": parent["finalization_sha256"],
            "runtime_commit": parent["runtime_commit"],
            "source_experiment_commit": parent["source_experiment_commit"],
            "decision": parent["decision"],
            "expected_runtime_commit": expected_parent_runtime_commit,
            "expected_source_experiment_commit": expected_source_experiment_commit,
            "expected_finalization_sha256": expected_parent_finalization_sha256,
        },
        "storage_profile": storage_profile,
        "verified_properties": {
            "all_six_arm_bindings_exact": True,
            "same_k_seed_budget_and_epoch_contract": True,
            "representation_channels_explicit": True,
            "one_heavy_forward_per_subgate": True,
            "score_function_detector_gradient_paths_passed": True,
            "fixed_representation_gradient_path_passed": True,
            "final_only_storage_bound_measured": True,
        },
        "training_completed": False,
        "single_seed_exploratory": True,
        "old_selector_reused": False,
        "selector_emitted": False,
        "p2_p3_opened": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    suite["suite_sha256"] = canonical_sha256(suite)
    return suite, storage_profile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def _run_main(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(run_root, boundary) or run_root == boundary:
        raise ValueError("estimator pilot P0 finalizer left write boundary")
    expected_commit = str(args.expected_commit).lower()
    actual_commit = _git_output("rev-parse", "HEAD").lower()
    if actual_commit != expected_commit:
        raise RuntimeError("estimator pilot P0 finalizer source mismatch")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("estimator pilot P0 finalizer requires clean source")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if not slurm_job_id.isdigit():
        raise RuntimeError("estimator pilot P0 finalizer requires Slurm")
    deployment_path = run_root / "control" / "deployment.json"
    deployment = _read_json(deployment_path)
    try:
        jobs = validate_pilot_job_receipt(
            deployment.get("jobs"),
            expected_p0_finalizer=slurm_job_id,
        )
    except ValueError:
        jobs = None
    if (
        deployment.get("schema_version") != PILOT_DEPLOYMENT_SCHEMA
        or deployment.get("status") != "SUBMITTED_SIX_ARM_EXPLORATORY_PILOT"
        or deployment.get("study_id") != PILOT_STUDY_ID
        or deployment.get("runtime_commit") != expected_commit
        or tuple(deployment.get("arms", [])) != PILOT_ARM_ORDER
        or not _self_hash_matches(deployment, "deployment_sha256")
        or jobs is None
    ):
        raise RuntimeError("estimator pilot deployment receipt is invalid")
    assert jobs is not None
    report_paths = {
        arm: run_root / "p0" / f"{arm}.json"
        for arm in PILOT_ARM_ORDER
    }
    suite_path = run_root / "control" / "pilot_p0_suite.json"
    profile_path = run_root / "control" / "georoute_storage_profile.json"
    failure_path = run_root / "control" / "pilot_p0_failure.json"
    if suite_path.exists() or profile_path.exists() or failure_path.exists():
        raise FileExistsError("estimator pilot P0 finalization is already sealed")
    try:
        suite, storage_profile = finalize_pilot_p0(
            report_paths=report_paths,
            expected_commit=expected_commit,
            parent_finalization_path=Path(
                deployment["preexperiment_parent"]["path"]
            ).resolve(),
            parent_finalization_file_sha256=str(
                deployment["preexperiment_parent"]["file_sha256"]
            ),
            expected_parent_runtime_commit=str(
                deployment["preexperiment_parent"]["expected_runtime_commit"]
            ),
            expected_source_experiment_commit=str(
                deployment["preexperiment_parent"][
                    "expected_source_experiment_commit"
                ]
            ),
            expected_parent_finalization_sha256=str(
                deployment["preexperiment_parent"]["expected_finalization_sha256"]
            ),
            expected_jobs={
                arm: str(jobs["p0"][arm])
                for arm in PILOT_ARM_ORDER
            },
        )
    except Exception as error:
        failure_core: dict[str, Any] = {
            "schema_version": PILOT_P0_FAILURE_SCHEMA,
            "status": "FAIL_P0_SUITE_MECHANICAL_ONLY",
            "study_id": PILOT_STUDY_ID,
            "runtime_commit": expected_commit,
            "p0_finalizer_job_id": slurm_job_id,
            "p0_job_ids": {
                arm: str(jobs["p0"][arm])
                for arm in PILOT_ARM_ORDER
            },
            "reports": {
                arm: {
                    "path": str(path),
                    "exists": path.is_file(),
                    "file_sha256": sha256_file(path) if path.is_file() else None,
                }
                for arm, path in report_paths.items()
            },
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
            "training_authorized": False,
            "performance_inference_allowed": False,
            "official_test_opened": False,
            "p2_p3_opened": False,
            "paper_claim_allowed": False,
        }
        failure = {
            **failure_core,
            "failure_sha256": canonical_sha256(failure_core),
        }
        _atomic_write_json(failure_path, failure)
        raise
    _atomic_write_json(suite_path, suite)
    _atomic_write_json(profile_path, storage_profile)
    print(json.dumps(suite, sort_keys=True))
    return 0


def _write_failsafe_failure(
    *,
    args: argparse.Namespace,
    error: Exception,
) -> None:
    run_root = args.run_root.resolve()
    boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(run_root, boundary) or run_root == boundary:
        return
    failure_path = run_root / "control" / "pilot_p0_failure.json"
    if failure_path.exists():
        return
    try:
        observed_commit = _git_output("rev-parse", "HEAD").lower()
    except Exception:
        observed_commit = None
    failure_core: dict[str, Any] = {
        "schema_version": PILOT_P0_FAILURE_SCHEMA,
        "status": "FAIL_P0_FINALIZER_MECHANICAL_ONLY",
        "study_id": PILOT_STUDY_ID,
        "runtime_commit": str(args.expected_commit).lower(),
        "observed_runtime_commit": observed_commit,
        "p0_finalizer_job_id": os.environ.get("SLURM_JOB_ID"),
        "failure_phase": "p0_finalizer_prevalidation_or_sealing",
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": traceback.format_exc(),
        "training_authorized": False,
        "performance_inference_allowed": False,
        "official_test_opened": False,
        "p2_p3_opened": False,
        "paper_claim_allowed": False,
    }
    failure = {
        **failure_core,
        "failure_sha256": canonical_sha256(failure_core),
    }
    _atomic_write_json(failure_path, failure)


def main() -> int:
    args = _parse_args()
    try:
        return _run_main(args)
    except Exception as error:
        try:
            _write_failsafe_failure(args=args, error=error)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
