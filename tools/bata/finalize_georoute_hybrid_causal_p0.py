#!/usr/bin/env python3
"""Seal the no-performance Hybrid causal P0 suite."""

from __future__ import annotations

import argparse
import hashlib
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

from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_hybrid_causal_contract import (  # noqa: E402
    HYBRID_CAUSAL_DEPLOYMENT_SCHEMA,
    HYBRID_CAUSAL_P0_SUITE_SCHEMA,
    HYBRID_CAUSAL_SEED,
    HYBRID_CAUSAL_STUDY_ID,
)
from tools.bata.georoute_official_comparable_contract import (  # noqa: E402
    validate_world2_kat_receipt,
)
from tools.bata.georoute_stage_runner import (  # noqa: E402
    _atomic_write_json,
    _read_json,
)
from tools.bata.run_georoute_p0_gate import validate_p0_gate_report  # noqa: E402


BOUNDARY = Path("/data/run01/sczc063/yuzibo")
MAIN_ARM = "hybrid_ctx8_roi28_res28_pl_support_only"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _validate_deployment(
    path: Path,
    *,
    expected_commit: str,
    finalizer_job_id: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("Hybrid causal P0 deployment receipt is missing")
    payload = _read_json(path)
    unsigned = dict(payload)
    observed_hash = unsigned.pop("deployment_sha256", None)
    jobs = payload.get("jobs")
    dependencies = payload.get("dependencies")
    if (
        payload.get("schema_version") != HYBRID_CAUSAL_DEPLOYMENT_SCHEMA
        or payload.get("phase") != "p0"
        or payload.get("study_id") != HYBRID_CAUSAL_STUDY_ID
        or payload.get("runtime_commit") != expected_commit
        or observed_hash != canonical_sha256(unsigned)
        or not isinstance(jobs, Mapping)
        or not isinstance(dependencies, Mapping)
        or str(jobs.get("p0_finalizer")) != finalizer_job_id
        or dependencies.get("p0_finalizer", {}).get("type") != "afterany"
        or set(dependencies.get("p0_finalizer", {}).get("job_ids", []))
        != {str(jobs.get("structured_model")), str(jobs.get("world2_ddp"))}
        or payload.get("official_test_opened") is not False
        or payload.get("performance_outputs_emitted") is not False
    ):
        raise ValueError("Hybrid causal P0 deployment receipt is invalid")
    return payload


def _storage_profile(
    report: Mapping[str, Any],
    *,
    runtime_commit: str,
) -> dict[str, Any]:
    measurement = report.get("checkpoint_storage_measurement")
    if not isinstance(measurement, Mapping):
        raise ValueError("Hybrid causal P0 lacks its storage measurement")
    return {
        "schema_version": "georoute_storage_profile_v1",
        "runtime_commit": runtime_commit,
        "checkpoint_policy": "final_only",
        "checkpoint_upper_bound_bytes": int(
            measurement["checkpoint_upper_bound_bytes"]
        ),
        "peak_checkpoint_copies_per_cell": int(
            measurement["peak_checkpoint_copies_per_cell"]
        ),
        "auxiliary_upper_bound_bytes_per_cell": int(
            measurement["auxiliary_upper_bound_bytes_per_cell"]
        ),
        "stage_fixed_overhead_bytes": int(
            measurement["stage_fixed_overhead_bytes"]
        ),
        "safety_fraction": float(measurement["safety_fraction"]),
        "safety_bytes": int(measurement["safety_bytes"]),
        "measurement_provenance": {
            MAIN_ARM: {
                "report_sha256": str(report["report_sha256"]),
                "measurement_method": str(measurement["measurement_method"]),
            }
        },
    }


def finalize_hybrid_causal_p0(
    *,
    run_root: Path,
    expected_commit: str,
    finalizer_job_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deployment_path = run_root / "control" / "p0_deployment_receipt.json"
    deployment = _validate_deployment(
        deployment_path,
        expected_commit=expected_commit,
        finalizer_job_id=finalizer_job_id,
    )
    model_path = run_root / "p0" / "a7_hybrid_pl.json"
    world2_path = run_root / "control" / "world2_fp32_ddp_kat.json"
    if any(path.is_symlink() or not path.is_file() for path in (model_path, world2_path)):
        raise FileNotFoundError("Hybrid causal P0 predecessor artifact is missing")

    model = _read_json(model_path)
    validate_p0_gate_report(model)
    if (
        model.get("status") != "PASS"
        or model.get("runtime_commit") != expected_commit
        or model.get("hybrid_causal_arm") != MAIN_ARM
        or int(model.get("route_parameters", {}).get("context_tokens", -1)) != 8
        or int(model.get("route_parameters", {}).get("structured_roi_tokens", -1))
        != 28
        or int(
            model.get("route_parameters", {}).get("structured_residual_tokens", -1)
        )
        != 28
        or str(model.get("slurm_job_id"))
        != str(deployment["jobs"]["structured_model"])
    ):
        raise ValueError("Hybrid causal structured model P0 is invalid")
    world2 = _read_json(world2_path)
    validate_world2_kat_receipt(
        world2,
        expected_commit=expected_commit,
        expected_slurm_job_id=str(deployment["jobs"]["world2_ddp"]),
    )
    storage_profile = _storage_profile(model, runtime_commit=expected_commit)
    suite: dict[str, Any] = {
        "schema_version": HYBRID_CAUSAL_P0_SUITE_SCHEMA,
        "status": "PASS_MECHANICAL_ONLY",
        "study_id": HYBRID_CAUSAL_STUDY_ID,
        "runtime_commit": expected_commit,
        "seed": HYBRID_CAUSAL_SEED,
        "main_arm": MAIN_ARM,
        "model_p0": {
            "path": str(model_path.resolve()),
            "file_sha256": sha256_file(model_path),
            "report_sha256": str(model["report_sha256"]),
            "slurm_job_id": str(model["slurm_job_id"]),
        },
        "world2_ddp_kat": {
            "path": str(world2_path.resolve()),
            "file_sha256": sha256_file(world2_path),
            "kat_sha256": str(world2["kat_sha256"]),
            "slurm_job_id": str(world2["slurm_job_id"]),
        },
        "deployment_receipt": {
            "path": str(deployment_path.resolve()),
            "file_sha256": sha256_file(deployment_path),
            "deployment_sha256": str(deployment["deployment_sha256"]),
        },
        "storage_profile": storage_profile,
        "verified_properties": {
            "sequential_conditional_pl_full_graph": True,
            "context_roi_residual_exact_8_28_28": True,
            "route_private_rng_global_state_isolated": True,
            "strict_cls_reg_detector_risk": True,
            "finite_roi_and_residual_policy_gradients": True,
            "one_heavy_forward": True,
            "world2_default_fp32_ddp_reduction": True,
            "representation_support_only": True,
        },
        "performance_training_authorized": True,
        "performance_outputs_emitted": False,
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
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


def main() -> int:
    args = _parse_args()
    run_root = args.run_root.resolve()
    expected_commit = str(args.expected_commit).lower()
    suite_path = run_root / "control" / "hybrid_causal_p0_suite.json"
    profile_path = run_root / "control" / "georoute_storage_profile.json"
    failure_path = run_root / "control" / "hybrid_causal_p0_failure.json"
    if not _inside(run_root, BOUNDARY.resolve()):
        raise ValueError("Hybrid causal P0 root leaves the write boundary")
    if any(path.exists() for path in (suite_path, profile_path, failure_path)):
        raise FileExistsError("Hybrid causal P0 finalization is already sealed")
    finalizer_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    try:
        if not finalizer_job_id.isdigit():
            raise RuntimeError("Hybrid causal P0 finalizer requires Slurm")
        if _git_output("rev-parse", "HEAD").lower() != expected_commit:
            raise RuntimeError("Hybrid causal P0 finalizer commit mismatch")
        if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
            raise RuntimeError("Hybrid causal P0 finalizer requires a clean source")
        suite, storage_profile = finalize_hybrid_causal_p0(
            run_root=run_root,
            expected_commit=expected_commit,
            finalizer_job_id=finalizer_job_id,
        )
    except Exception as error:
        trace = traceback.format_exc()
        failure: dict[str, Any] = {
            "schema_version": HYBRID_CAUSAL_P0_SUITE_SCHEMA,
            "status": "FAIL_MECHANICAL_P0",
            "study_id": HYBRID_CAUSAL_STUDY_ID,
            "runtime_commit": expected_commit,
            "slurm_job_id": finalizer_job_id or None,
            "exception_type": type(error).__name__,
            "exception_message": str(error)[:2000],
            "traceback_sha256": hashlib.sha256(
                trace.encode("utf-8", errors="replace")
            ).hexdigest(),
            "performance_training_authorized": False,
            "performance_outputs_emitted": False,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }
        failure["failure_sha256"] = canonical_sha256(failure)
        _atomic_write_json(failure_path, failure)
        raise
    _atomic_write_json(suite_path, suite)
    _atomic_write_json(profile_path, storage_profile)
    print(json.dumps(suite, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
