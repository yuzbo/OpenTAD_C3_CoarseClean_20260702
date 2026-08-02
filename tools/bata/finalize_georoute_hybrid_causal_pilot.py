#!/usr/bin/env python3
"""All-terminal finalizer for the nine-arm Hybrid causal screen."""

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
    HYBRID_CAUSAL_ARM_ORDER,
    HYBRID_CAUSAL_DEPLOYMENT_SCHEMA,
    HYBRID_CAUSAL_FINALIZATION_SCHEMA,
    HYBRID_CAUSAL_P0_SUITE_SCHEMA,
    HYBRID_CAUSAL_SEED,
    HYBRID_CAUSAL_STUDY_ID,
    finalize_hybrid_causal_study,
    hybrid_causal_cell_relative_path,
)
from tools.bata.georoute_stage_runner import (  # noqa: E402
    _atomic_write_json,
    _read_json,
)


BOUNDARY = Path("/data/run01/sczc063/yuzibo")


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
        raise FileNotFoundError("Hybrid causal pilot deployment receipt is missing")
    payload = _read_json(path)
    unsigned = dict(payload)
    observed_hash = unsigned.pop("deployment_sha256", None)
    jobs = payload.get("jobs")
    stage_jobs = jobs.get("stages") if isinstance(jobs, Mapping) else None
    dependency = payload.get("dependencies", {}).get("finalizer", {})
    if (
        payload.get("schema_version") != HYBRID_CAUSAL_DEPLOYMENT_SCHEMA
        or payload.get("phase") != "pilot"
        or payload.get("status") != "SUBMITTED_NINE_ARM_EXPLORATORY_SCREEN"
        or payload.get("study_id") != HYBRID_CAUSAL_STUDY_ID
        or payload.get("runtime_commit") != expected_commit
        or tuple(payload.get("arm_order", [])) != HYBRID_CAUSAL_ARM_ORDER
        or observed_hash != canonical_sha256(unsigned)
        or not isinstance(stage_jobs, Mapping)
        or tuple(stage_jobs) != HYBRID_CAUSAL_ARM_ORDER
        or str(jobs.get("finalizer")) != finalizer_job_id
        or dependency.get("type") != "afterany"
        or set(dependency.get("job_ids", []))
        != {str(stage_jobs[arm]) for arm in HYBRID_CAUSAL_ARM_ORDER}
        or payload.get("partial_survivor_inference_allowed") is not False
        or payload.get("official_test_opened") is not False
    ):
        raise ValueError("Hybrid causal pilot deployment receipt is invalid")
    return payload


def _verify_receipt(path_text: Any, digest: Any) -> None:
    path = Path(str(path_text)).resolve()
    if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"artifact receipt mismatch: {path}")


def _validate_stage_artifacts(
    result: Mapping[str, Any],
    *,
    expected_job_id: str,
    p0_suite_path: Path,
    p0_suite_sha256: str,
) -> None:
    checkpoint = result.get("checkpoint_receipt")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("stage lacks checkpoint receipt")
    _verify_receipt(checkpoint.get("path"), checkpoint.get("sha256"))
    if int(Path(str(checkpoint["path"])).stat().st_size) != int(
        checkpoint.get("size_bytes", -1)
    ):
        raise ValueError("checkpoint size receipt mismatch")
    for group_name in ("config_receipts", "artifact_receipts"):
        group = result.get(group_name)
        if not isinstance(group, Mapping) or not group:
            raise ValueError(f"stage lacks {group_name}")
        for receipt in group.values():
            if not isinstance(receipt, Mapping):
                raise ValueError(f"invalid receipt in {group_name}")
            _verify_receipt(receipt.get("path"), receipt.get("sha256"))
    rendezvous = result.get("rendezvous")
    profile_rendezvous = result.get("profile_rendezvous")
    if (
        not isinstance(rendezvous, Mapping)
        or not isinstance(profile_rendezvous, Mapping)
        or {
            str(rendezvous.get("train", {}).get("slurm_job_id")),
            str(rendezvous.get("test", {}).get("slurm_job_id")),
            str(profile_rendezvous.get("slurm_job_id")),
        }
        != {expected_job_id}
    ):
        raise ValueError("stage rendezvous job id differs from deployment")
    parent = result.get("parent_p0_suite")
    if (
        not isinstance(parent, Mapping)
        or Path(str(parent.get("path"))).resolve() != p0_suite_path.resolve()
        or parent.get("file_sha256") != p0_suite_sha256
        or sha256_file(p0_suite_path) != p0_suite_sha256
    ):
        raise ValueError("stage P0 parent receipt mismatch")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_root = args.run_root.resolve()
    expected_commit = str(args.expected_commit).lower()
    output_path = run_root / "control" / "hybrid_causal_finalization.json"
    failure_path = run_root / "control" / "hybrid_causal_finalization_failure.json"
    if not _inside(run_root, BOUNDARY.resolve()):
        raise ValueError("Hybrid causal finalizer root leaves the write boundary")
    if output_path.exists() or failure_path.exists():
        raise FileExistsError("Hybrid causal pilot finalization is already sealed")
    finalizer_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    try:
        if not finalizer_job_id.isdigit():
            raise RuntimeError("Hybrid causal pilot finalizer requires Slurm")
        if _git_output("rev-parse", "HEAD").lower() != expected_commit:
            raise RuntimeError("Hybrid causal finalizer commit mismatch")
        if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
            raise RuntimeError("Hybrid causal finalizer requires a clean source")
        deployment_path = run_root / "control" / "pilot_deployment_receipt.json"
        deployment = _validate_deployment(
            deployment_path,
            expected_commit=expected_commit,
            finalizer_job_id=finalizer_job_id,
        )
        p0_suite_path = run_root / "control" / "hybrid_causal_p0_suite.json"
        p0_suite = _read_json(p0_suite_path)
        p0_suite_file_sha256 = sha256_file(p0_suite_path)
        if (
            p0_suite.get("schema_version") != HYBRID_CAUSAL_P0_SUITE_SCHEMA
            or p0_suite.get("status") != "PASS_MECHANICAL_ONLY"
            or p0_suite.get("runtime_commit") != expected_commit
            or p0_suite.get("performance_training_authorized") is not True
            or deployment.get("p0_suite", {}).get("file_sha256")
            != p0_suite_file_sha256
        ):
            raise ValueError("Hybrid causal finalizer P0 parent is invalid")

        results: dict[str, Mapping[str, Any]] = {}
        artifact_failures: dict[str, str] = {}
        for arm in HYBRID_CAUSAL_ARM_ORDER:
            stage_path = (
                run_root
                / hybrid_causal_cell_relative_path(
                    arm=arm,
                    seed=HYBRID_CAUSAL_SEED,
                )
                / "stage_result.json"
            )
            if not stage_path.is_file() or stage_path.is_symlink():
                artifact_failures[arm] = "MISSING_STAGE_RESULT_ARTIFACT"
                continue
            try:
                result = _read_json(stage_path)
                _validate_stage_artifacts(
                    result,
                    expected_job_id=str(deployment["jobs"]["stages"][arm]),
                    p0_suite_path=p0_suite_path,
                    p0_suite_sha256=p0_suite_file_sha256,
                )
                results[arm] = result
            except Exception as error:
                artifact_failures[arm] = (
                    f"INVALID_ARTIFACT_RECEIPT:{type(error).__name__}:{error}"
                )
        finalization = finalize_hybrid_causal_study(
            results,
            expected_commit=expected_commit,
        )
        finalization.pop("finalization_sha256", None)
        finalization.update(
            {
                "artifact_failures": artifact_failures,
                "deployment_receipt": {
                    "path": str(deployment_path.resolve()),
                    "file_sha256": sha256_file(deployment_path),
                    "deployment_sha256": str(deployment["deployment_sha256"]),
                },
                "p0_suite": {
                    "path": str(p0_suite_path.resolve()),
                    "file_sha256": p0_suite_file_sha256,
                    "suite_sha256": str(p0_suite["suite_sha256"]),
                },
                "all_terminal_finalizer_job_id": finalizer_job_id,
            }
        )
        if artifact_failures:
            finalization["status"] = "INCOMPLETE_EXPLORATORY_SCREEN"
            finalization["decision"] = "INCOMPLETE_NO_PERFORMANCE_INFERENCE"
            finalization["descriptive_contrasts"] = {}
            finalization["screen_admission_checks"] = {}
        finalization["finalization_sha256"] = canonical_sha256(finalization)
    except Exception as error:
        trace = traceback.format_exc()
        failure: dict[str, Any] = {
            "schema_version": HYBRID_CAUSAL_FINALIZATION_SCHEMA,
            "status": "FAIL_UNTRUSTED_FINALIZER_INPUT",
            "study_id": HYBRID_CAUSAL_STUDY_ID,
            "runtime_commit": expected_commit,
            "slurm_job_id": finalizer_job_id or None,
            "exception_type": type(error).__name__,
            "exception_message": str(error)[:2000],
            "traceback_sha256": hashlib.sha256(
                trace.encode("utf-8", errors="replace")
            ).hexdigest(),
            "descriptive_contrasts": {},
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }
        failure["failure_sha256"] = canonical_sha256(failure)
        _atomic_write_json(failure_path, failure)
        raise
    _atomic_write_json(output_path, finalization)
    print(json.dumps(finalization, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
