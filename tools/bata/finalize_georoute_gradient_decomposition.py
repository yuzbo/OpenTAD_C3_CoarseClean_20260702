#!/usr/bin/env python3
"""Seal the matched GeoRoute gradient decomposition without performance claims."""

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

from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_gradient_decomposition import (  # noqa: E402
    ARMS,
    COMPLETE_STATUS,
    DECISION_INCOMPLETE,
    DEPLOYMENT_SCHEMA,
    DEPLOYMENT_STATUS,
    FINALIZATION_SCHEMA,
    FINALIZER_SUBMISSION_STATUS,
    INCOMPLETE_STATUS,
    PROFILE,
    SEED,
    STAGE_PASS_STATUS,
    STAGE_SCHEMA,
    STAGE_WRAPPER_FAIL_STATUS,
    STUDY_ID,
    cell_relative_path,
    classify_pair,
    validate_job_receipt,
)
from tools.bata.georoute_gradient_decomposition_stage_runner import (  # noqa: E402
    audit_no_performance_artifacts,
    validate_stage_result,
)


BOUNDARY = Path("/data/run01/sczc063/yuzibo")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
        raise RuntimeError(
            completed.stderr.strip() or f"git {' '.join(arguments)} failed"
        )
    return completed.stdout.strip()


def _self_hash_matches(payload: Mapping[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


def _is_full_hex(value: Any, *, length: int) -> bool:
    normalized = str(value)
    return len(normalized) == length and all(
        character in "0123456789abcdef" for character in normalized
    )


def _validate_wrapper_failure(
    payload: Mapping[str, Any],
    *,
    expected_arm: str,
    expected_commit: str,
    expected_job_id: str,
) -> dict[str, Any]:
    failure = dict(payload)
    if not _self_hash_matches(failure, field="failure_sha256"):
        raise ValueError("gradient-decomposition wrapper-failure self-hash mismatch")
    if (
        failure.get("schema_version") != STAGE_SCHEMA
        or failure.get("status") != STAGE_WRAPPER_FAIL_STATUS
        or failure.get("study_id") != STUDY_ID
        or failure.get("profile") != PROFILE
        or failure.get("arm") != expected_arm
        or int(failure.get("seed", -1)) != SEED
        or failure.get("expected_runtime_commit") != expected_commit
        or not _is_full_hex(failure.get("observed_runtime_commit"), length=40)
        or str(failure.get("slurm_job_id", "")) != str(expected_job_id)
        or failure.get("checkpoint_emitted") is not False
        or failure.get("prediction_emitted") is not False
        or failure.get("evaluator_invoked") is not False
        or failure.get("official_test_opened") is not False
        or failure.get("performance_inference_allowed") is not False
        or failure.get("paper_claim_allowed") is not False
    ):
        raise ValueError("gradient-decomposition wrapper-failure contract is invalid")
    return failure


def _validate_artifact_paths(
    result: Mapping[str, Any], *, run_root: Path, cell_root: Path
) -> None:
    for path_field, hash_field in (
        ("diagnostic_receipt_path", "diagnostic_receipt_file_sha256"),
        ("bound_config_path", "bound_config_sha256"),
        ("train_log_path", "train_log_sha256"),
    ):
        path = Path(str(result.get(path_field, ""))).resolve()
        if (
            not path.is_file()
            or not _inside(path, run_root)
            or sha256_file(path) != result.get(hash_field)
        ):
            raise ValueError(
                f"gradient-decomposition artifact changed or escaped: {path_field}"
            )
    receipt_path = Path(result["diagnostic_receipt_path"]).resolve()
    if receipt_path != (cell_root / "gradient_decomposition.json").resolve():
        raise ValueError("gradient-decomposition receipt is outside its cell")
    if _read_json(receipt_path) != result["diagnostic_receipt"]:
        raise ValueError("embedded gradient receipt differs from file")
    current_audit = audit_no_performance_artifacts(cell_root)
    if any(
        current_audit[key] != result["artifact_audit"][key]
        for key in (
            "checkpoint_payload_count",
            "temporary_payload_count",
            "prediction_payload_count",
            "evaluator_output_count",
            "checkpoint_emitted",
            "prediction_emitted",
            "evaluator_invoked",
            "official_test_opened",
            "performance_inference_allowed",
            "paper_claim_allowed",
        )
    ):
        raise ValueError("gradient-decomposition artifact audit changed")


def finalize_gradient_decomposition(
    *,
    run_root: Path,
    expected_commit: str,
    expected_stage_jobs: Mapping[str, str],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    receipts: dict[str, Mapping[str, Any]] = {}
    for arm in ARMS:
        cell_root = run_root / cell_relative_path(arm)
        result_path = cell_root / "stage_result.json"
        wrapper_failure_path = (
            run_root / "control" / "stage_failures" / f"{arm}.json"
        )
        if result_path.is_file() and wrapper_failure_path.is_file():
            failures[arm] = {
                "status": "AMBIGUOUS_STAGE_RESULT_AND_WRAPPER_FAILURE",
                "stage_result_path": str(result_path),
                "wrapper_failure_path": str(wrapper_failure_path),
            }
            continue
        if wrapper_failure_path.is_file():
            try:
                failure = _validate_wrapper_failure(
                    _read_json(wrapper_failure_path),
                    expected_arm=arm,
                    expected_commit=expected_commit,
                    expected_job_id=str(expected_stage_jobs[arm]),
                )
                failures[arm] = {
                    "status": failure["status"],
                    "exception_type": failure.get("exception_type"),
                    "exception_message": failure.get("exception_message"),
                    "path": str(wrapper_failure_path),
                    "file_sha256": sha256_file(wrapper_failure_path),
                    "failure_sha256": failure["failure_sha256"],
                }
            except BaseException as error:
                failures[arm] = {
                    "status": "INVALID_WRAPPER_FAILURE",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "path": str(wrapper_failure_path),
                    "file_sha256": sha256_file(wrapper_failure_path),
                }
            continue
        if not result_path.is_file():
            failures[arm] = {
                "status": "MISSING_STAGE_RESULT",
                "cell_root": str(cell_root),
            }
            continue
        try:
            result = validate_stage_result(
                _read_json(result_path),
                expected_arm=arm,
                expected_commit=expected_commit,
                expected_job_id=str(expected_stage_jobs[arm]),
            )
            _validate_artifact_paths(
                result, run_root=run_root, cell_root=cell_root
            )
            receipt = result["diagnostic_receipt"]
            receipts[arm] = receipt
            arms[arm] = {
                "status": result["status"],
                "slurm_job_id": result["slurm_job_id"],
                "summary": dict(receipt["summary"]),
                "diagnostic_status": receipt["status"],
                "diagnostic_receipt_path": result["diagnostic_receipt_path"],
                "diagnostic_receipt_file_sha256": result[
                    "diagnostic_receipt_file_sha256"
                ],
                "stage_result_path": str(result_path),
                "stage_result_file_sha256": sha256_file(result_path),
                "stage_result_sha256": result["stage_result_sha256"],
            }
            if result["status"] != STAGE_PASS_STATUS:
                failures[arm] = {
                    "status": result["status"],
                    "diagnostic_status": receipt["status"],
                    "execution_error": result.get("execution_error"),
                }
        except BaseException as error:
            failures[arm] = {
                "status": "INVALID_STAGE_RESULT",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "path": str(result_path),
                "file_sha256": sha256_file(result_path),
            }

    classification = classify_pair(receipts)
    if failures:
        classification = {
            "decision": DECISION_INCOMPLETE,
            "repair_class_identified": False,
            "repair_class": None,
            "repair_authorized": False,
            "reason": "stage_failure_or_invalid_artifact",
        }
    all_arms_passed = (
        set(arms) == set(ARMS)
        and not failures
        and all(arms[arm]["status"] == STAGE_PASS_STATUS for arm in ARMS)
    )
    finalization: dict[str, Any] = {
        "schema_version": FINALIZATION_SCHEMA,
        "status": COMPLETE_STATUS if all_arms_passed else INCOMPLETE_STATUS,
        "decision": classification["decision"],
        "study_id": STUDY_ID,
        "profile": PROFILE,
        "runtime_commit": expected_commit,
        "seed": SEED,
        "arms": arms,
        "failures": failures,
        "all_arms_passed": all_arms_passed,
        "classification": classification,
        "repair_class_identified": bool(
            classification.get("repair_class_identified", False)
        ),
        "repair_class": classification.get("repair_class"),
        "repair_authorized": bool(
            classification.get("repair_authorized", False)
        ),
        "performance_metrics": {},
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "p2_p3_opened": False,
        "performance_inference_allowed": False,
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
    if not _inside(run_root, BOUNDARY.resolve()):
        raise ValueError("gradient-decomposition finalizer left write boundary")
    expected_commit = str(args.expected_commit).lower()
    if _git_output("rev-parse", "HEAD").lower() != expected_commit:
        raise RuntimeError("gradient-decomposition finalizer source mismatch")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("gradient-decomposition finalizer requires clean source")
    finalizer_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    if not finalizer_job_id.isdigit():
        raise RuntimeError("gradient-decomposition finalizer requires Slurm")

    deployment_path = run_root / "control" / "deployment.json"
    deployment = _read_json(deployment_path)
    jobs = validate_job_receipt(
        deployment.get("jobs"), expected_finalizer=finalizer_job_id
    )
    if (
        deployment.get("schema_version") != DEPLOYMENT_SCHEMA
        or deployment.get("status") != DEPLOYMENT_STATUS
        or deployment.get("study_id") != STUDY_ID
        or deployment.get("profile") != PROFILE
        or deployment.get("runtime_commit") != expected_commit
        or tuple(deployment.get("arms", ())) != ARMS
        or int(deployment.get("seed", -1)) != SEED
        or not _self_hash_matches(deployment, field="deployment_sha256")
    ):
        raise RuntimeError("gradient-decomposition deployment receipt is invalid")
    submission_path = run_root / "control" / "finalizer_submission.json"
    submission = _read_json(submission_path)
    if (
        submission.get("schema_version") != DEPLOYMENT_SCHEMA
        or submission.get("status") != FINALIZER_SUBMISSION_STATUS
        or submission.get("profile") != PROFILE
        or submission.get("runtime_commit") != expected_commit
        or submission.get("deployment_file_sha256")
        != sha256_file(deployment_path)
        or submission.get("finalizer_job_id") != finalizer_job_id
        or submission.get("dependency_type") != "afterany"
        or set(submission.get("predecessor_job_ids", ()))
        != set(jobs["stage"].values())
        or not _self_hash_matches(submission, field="receipt_sha256")
    ):
        raise RuntimeError(
            "gradient-decomposition finalizer submission receipt is invalid"
        )
    finalization = finalize_gradient_decomposition(
        run_root=run_root,
        expected_commit=expected_commit,
        expected_stage_jobs=jobs["stage"],
    )
    finalization["deployment_path"] = str(deployment_path)
    finalization["deployment_file_sha256"] = sha256_file(deployment_path)
    finalization["finalizer_submission_path"] = str(submission_path)
    finalization["finalizer_submission_file_sha256"] = sha256_file(
        submission_path
    )
    finalization.pop("finalization_sha256")
    finalization["finalization_sha256"] = canonical_sha256(finalization)
    output = run_root / "control" / "finalization.json"
    if output.exists():
        raise FileExistsError(
            "gradient-decomposition finalization already exists"
        )
    _atomic_write_json(output, finalization)
    print(json.dumps(finalization, sort_keys=True))
    return 0


def _write_failsafe(
    *, args: argparse.Namespace, error: BaseException
) -> None:
    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()):
        return
    output = run_root / "control" / "finalization.json"
    if output.exists():
        return
    payload: dict[str, Any] = {
        "schema_version": FINALIZATION_SCHEMA,
        "status": INCOMPLETE_STATUS,
        "decision": DECISION_INCOMPLETE,
        "study_id": STUDY_ID,
        "profile": PROFILE,
        "runtime_commit": str(args.expected_commit).lower(),
        "observed_runtime_commit": (
            _git_output("rev-parse", "HEAD").lower()
            if (ROOT / ".git").exists()
            else None
        ),
        "seed": SEED,
        "arms": {},
        "failures": {
            "finalizer": {
                "status": "FAIL_FINALIZER_PREVALIDATION_OR_SEALING",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }
        },
        "all_arms_passed": False,
        "classification": {
            "decision": DECISION_INCOMPLETE,
            "repair_class_identified": False,
            "repair_class": None,
            "repair_authorized": False,
            "reason": "finalizer_failure",
        },
        "repair_class_identified": False,
        "repair_class": None,
        "repair_authorized": False,
        "performance_metrics": {},
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "p2_p3_opened": False,
        "performance_inference_allowed": False,
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
            _write_failsafe(args=args, error=error)
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
