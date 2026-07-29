#!/usr/bin/env python3
"""Finalize the six-arm exploratory pilot without selecting or promoting."""

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

from tools.bata.georoute_estimator_pilot_contract import (  # noqa: E402
    PILOT_ARM_ORDER,
    PILOT_CONTRASTS,
    PILOT_DEPLOYMENT_SCHEMA,
    PILOT_FINALIZATION_SCHEMA,
    PILOT_P0_SUITE_SCHEMA,
    PILOT_SEED,
    PILOT_STUDY_ID,
    pilot_cell_relative_path,
)
from tools.bata.georoute_estimator_pilot_stage_runner import (  # noqa: E402
    _pilot_profile,
    _read_parent_p0_suite,
    summarize_pilot_telemetry,
    validate_pilot_stage_result,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
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


def _validate_artifact_receipts(
    result: Mapping[str, Any],
    *,
    cell_root: Path,
) -> None:
    artifact_fields = (
        ("config_path", "config_sha256"),
        ("prediction_path", "prediction_sha256"),
        ("profile_path", None),
        ("telemetry_path", None),
        ("test_log_path", "test_log_sha256"),
    )
    for path_field, hash_field in artifact_fields:
        path = Path(str(result.get(path_field, ""))).resolve()
        if not path.is_file() or not _inside(path, cell_root.parent.parent.parent):
            raise ValueError(f"estimator pilot artifact is missing or escaped: {path_field}")
        if hash_field is not None and sha256_file(path) != result.get(hash_field):
            raise ValueError(f"estimator pilot artifact hash changed: {path_field}")
        if path_field == "profile_path":
            if _pilot_profile(path) != result["profile"]:
                raise ValueError("estimator pilot profile receipt changed")
        if path_field == "telemetry_path":
            if summarize_pilot_telemetry(path) != result["telemetry_summary"]:
                raise ValueError("estimator pilot telemetry summary changed")
    checkpoint = result["checkpoint_receipt"]
    checkpoint_path = Path(str(checkpoint["path"])).resolve()
    if (
        not checkpoint_path.is_file()
        or not _inside(checkpoint_path, cell_root)
        or sha256_file(checkpoint_path) != checkpoint["sha256"]
        or int(checkpoint_path.stat().st_size) != int(checkpoint["size_bytes"])
    ):
        raise ValueError("estimator pilot final checkpoint receipt changed")
    checkpoint_payloads = sorted(checkpoint_path.parent.glob("*.pth"))
    checkpoint_temporaries = sorted(checkpoint_path.parent.glob("*.tmp*"))
    if checkpoint_payloads != [checkpoint_path] or checkpoint_temporaries:
        raise ValueError("estimator pilot final-only checkpoint policy was violated")


def _metric_delta(
    treatment: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, float]:
    return {
        key: float(treatment["metrics"][key]) - float(control["metrics"][key])
        for key in treatment["metrics"]
    }


def _cost_delta(
    treatment: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, float]:
    cost_keys = (
        "loader_wait_p50_ms",
        "loader_wait_p95_ms",
        "model_and_postprocess_p50_ms",
        "model_and_postprocess_p95_ms",
        "window_wall_p50_ms",
        "window_wall_p95_ms",
        "peak_allocated_mb",
    )
    return {
        key: float(treatment["profile"][key]) - float(control["profile"][key])
        for key in cost_keys
    }


def _compact_arm_record(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metrics": dict(result["metrics"]),
        "profile": dict(result["profile"]),
        "telemetry_summary": dict(result["telemetry_summary"]),
        "checkpoint_receipt": dict(result["checkpoint_receipt"]),
        "stage_result_sha256": result["stage_result_sha256"],
        "runtime_commit": result["runtime_commit"],
        "slurm_job_id": result["rendezvous"]["train"]["slurm_job_id"],
        "train_slurm_job_id": result["rendezvous"]["train"]["slurm_job_id"],
        "test_slurm_job_id": result["rendezvous"]["test"]["slurm_job_id"],
    }


def finalize_pilot_results(
    *,
    run_root: Path,
    expected_commit: str,
    expected_stage_jobs: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    arms: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}
    population_hashes = set()
    population_descriptor_hashes = set()
    train_slurm_ids = set()
    test_slurm_ids = set()
    rendezvous_ids = set()
    checkpoint_paths = set()
    parent_suite_hashes = set()
    source_config_hashes = set()
    manifest_hashes = set()
    annotation_hashes = set()
    class_map_hashes = set()
    pretrained_hashes = set()
    development_video_roots = set()

    for arm in PILOT_ARM_ORDER:
        cell_root = run_root / pilot_cell_relative_path(
            arm=arm,
            seed=PILOT_SEED,
        )
        result_path = cell_root / "stage_result.json"
        failure_path = cell_root / "pilot_failure.json"
        if result_path.is_file() and failure_path.is_file():
            failures[arm] = {
                "status": "AMBIGUOUS_RESULT_AND_FAILURE",
                "result_path": str(result_path),
                "failure_path": str(failure_path),
            }
            continue
        if failure_path.is_file():
            failure = _read_json(failure_path)
            failures[arm] = {
                "status": failure.get("status"),
                "exception_type": failure.get("exception_type"),
                "exception_message": failure.get("exception_message"),
                "failure_path": str(failure_path),
                "failure_file_sha256": sha256_file(failure_path),
                "failure_self_hash_valid": _self_hash_matches(
                    failure,
                    "failure_sha256",
                ),
            }
            continue
        if not result_path.is_file():
            failures[arm] = {
                "status": "MISSING_STAGE_RESULT",
                "cell_root": str(cell_root),
            }
            continue
        try:
            result = _read_json(result_path)
            validate_pilot_stage_result(
                result,
                expected_arm=arm,
                expected_commit=expected_commit,
            )
            if expected_stage_jobs is not None and result["rendezvous"]["train"][
                "slurm_job_id"
            ] != str(expected_stage_jobs[arm]):
                raise ValueError("stage-result Slurm job differs from deployment")
            _validate_artifact_receipts(result, cell_root=cell_root)
            parent_path = Path(
                str(result["parent_p0_suite"]["path"])
            ).resolve()
            canonical_parent_path = (
                run_root / "control" / "pilot_p0_suite.json"
            ).resolve()
            if (
                parent_path != canonical_parent_path
                or not parent_path.is_file()
                or result["parent_p0_suite"]["file_sha256"]
                != sha256_file(parent_path)
            ):
                raise ValueError("stage-result P0 parent file changed")
            parent_payload = _read_parent_p0_suite(
                parent_path,
                expected_commit=expected_commit,
            )
            if (
                result["parent_p0_suite"]["suite_sha256"]
                != parent_payload.get("suite_sha256")
            ):
                raise ValueError("stage-result P0 parent self-hash changed")
            binding_config = Path(result["config_path"])
            binding = result["binding"]
            if Path(str(binding["work_dir"])).resolve() != cell_root.resolve():
                raise ValueError("stage-result binding points at another work directory")
            inputs = result["input_receipts"]
            source_config_hashes.add(str(inputs["source_config_sha256"]))
            population_hashes.add(
                str(result["telemetry_summary"]["population_sha256"])
            )
            population_descriptor_hashes.add(
                str(result["telemetry_summary"]["population_descriptor_sha256"])
            )
            train_job = str(result["rendezvous"]["train"]["slurm_job_id"])
            test_job = str(result["rendezvous"]["test"]["slurm_job_id"])
            if train_job != test_job:
                raise ValueError("train and test receipts differ from their Slurm leaf")
            train_slurm_ids.add(train_job)
            test_slurm_ids.add(test_job)
            rendezvous_ids.update(
                (
                    str(result["rendezvous"]["train"]["rendezvous_id"]),
                    str(result["rendezvous"]["test"]["rendezvous_id"]),
                )
            )
            checkpoint_paths.add(str(result["checkpoint_receipt"]["path"]))
            parent_suite_hashes.add(
                str(result["parent_p0_suite"]["suite_sha256"])
            )
            arms[arm] = {
                **_compact_arm_record(result),
                "result_path": str(result_path),
                "result_file_sha256": sha256_file(result_path),
                "bound_config_path": str(binding_config),
            }
            manifest_hashes.add(str(inputs["manifest_file_sha256"]))
            annotation_hashes.add(
                str(inputs["development_annotation_sha256"])
            )
            class_map_hashes.add(str(inputs["class_map_sha256"]))
            pretrained_hashes.add(str(inputs["pretrained_checkpoint_sha256"]))
            development_video_roots.add(str(inputs["development_video_root"]))
        except Exception as error:
            failures[arm] = {
                "status": "INVALID_STAGE_RESULT",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "result_path": str(result_path),
                "result_file_sha256": sha256_file(result_path),
            }

    all_passed = set(arms) == set(PILOT_ARM_ORDER) and not failures
    cross_arm_consistent = bool(
        all_passed
        and len(population_hashes) == 1
        and len(population_descriptor_hashes) == 1
        and len(train_slurm_ids) == len(PILOT_ARM_ORDER)
        and len(test_slurm_ids) == len(PILOT_ARM_ORDER)
        and train_slurm_ids == test_slurm_ids
        and len(rendezvous_ids) == 2 * len(PILOT_ARM_ORDER)
        and len(checkpoint_paths) == len(PILOT_ARM_ORDER)
        and len(parent_suite_hashes) == 1
        and len(source_config_hashes) == 1
        and len(manifest_hashes) == 1
        and len(annotation_hashes) == 1
        and len(class_map_hashes) == 1
        and len(pretrained_hashes) == 1
        and len(development_video_roots) == 1
    )
    all_passed = all_passed and cross_arm_consistent
    contrasts: dict[str, Any] = {}
    if all_passed:
        for contrast, (treatment, control) in PILOT_CONTRASTS.items():
            contrasts[contrast] = {
                "treatment": treatment,
                "control": control,
                "metric_delta": _metric_delta(
                    arms[treatment],
                    arms[control],
                ),
                "cost_delta": _cost_delta(
                    arms[treatment],
                    arms[control],
                ),
                "interpretation": "descriptive_single_seed_only",
                "promotion_allowed": False,
            }
    if all_passed:
        status = "COMPLETE_EXPLORATORY_PILOT"
        decision = "PILOT_COMPLETE_NO_PROMOTION"
    else:
        status = "INCOMPLETE_EXPLORATORY_PILOT"
        decision = "PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE"
    finalization: dict[str, Any] = {
        "schema_version": PILOT_FINALIZATION_SCHEMA,
        "status": status,
        "decision": decision,
        "study_id": PILOT_STUDY_ID,
        "runtime_commit": expected_commit,
        "seed": PILOT_SEED,
        "arms": arms,
        "failures": failures,
        "all_six_arms_passed": all_passed,
        "cross_arm_population_and_artifact_consistent": cross_arm_consistent,
        "descriptive_contrasts": contrasts,
        "selection_rule": "descriptive_only_no_automatic_winner_or_promotion",
        "single_seed_exploratory": True,
        "confirmatory_margin_frozen": False,
        "confirmatory_seed_used": False,
        "old_selector_reused": False,
        "selector_emitted": False,
        "p2_p3_opened": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    finalization["finalization_sha256"] = canonical_sha256(finalization)
    return finalization


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_root = args.run_root.resolve()
    boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(run_root, boundary) or run_root == boundary:
        raise ValueError("estimator pilot finalizer left write boundary")
    expected_commit = str(args.expected_commit).lower()
    if _git_output("rev-parse", "HEAD").lower() != expected_commit:
        raise RuntimeError("estimator pilot finalizer source mismatch")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("estimator pilot finalizer requires clean source")
    finalizer_job_id = os.environ.get("SLURM_JOB_ID", "")
    if not finalizer_job_id.isdigit():
        raise RuntimeError("estimator pilot finalizer requires Slurm")

    deployment_path = run_root / "control" / "deployment.json"
    deployment = _read_json(deployment_path)
    jobs = deployment.get("jobs")
    if (
        deployment.get("schema_version") != PILOT_DEPLOYMENT_SCHEMA
        or deployment.get("status") != "SUBMITTED_SIX_ARM_EXPLORATORY_PILOT"
        or deployment.get("study_id") != PILOT_STUDY_ID
        or deployment.get("runtime_commit") != expected_commit
        or tuple(deployment.get("arms", [])) != PILOT_ARM_ORDER
        or not _self_hash_matches(deployment, "deployment_sha256")
        or not isinstance(jobs, Mapping)
        or not isinstance(jobs.get("stage"), Mapping)
        or tuple(jobs["stage"]) != PILOT_ARM_ORDER
    ):
        raise RuntimeError("estimator pilot deployment receipt is invalid")
    submission_path = run_root / "control" / "finalizer_submission.json"
    submission = _read_json(submission_path)
    predecessor_ids = [
        *jobs["p0"].values(),
        jobs["p0_finalizer"],
        *jobs["stage"].values(),
    ]
    if (
        submission.get("schema_version") != PILOT_DEPLOYMENT_SCHEMA
        or submission.get("status") != "SUBMITTED_EXPLORATORY_FINALIZER_AFTERANY"
        or submission.get("runtime_commit") != expected_commit
        or submission.get("deployment_file_sha256")
        != sha256_file(deployment_path)
        or submission.get("finalizer_job_id") != finalizer_job_id
        or submission.get("dependency_type") != "afterany"
        or set(submission.get("predecessor_job_ids", []))
        != set(predecessor_ids)
        or not _self_hash_matches(submission, "receipt_sha256")
    ):
        raise RuntimeError("estimator pilot finalizer submission is invalid")
    p0_suite_path = run_root / "control" / "pilot_p0_suite.json"
    if p0_suite_path.is_file():
        p0_suite = _read_json(p0_suite_path)
        if (
            p0_suite.get("schema_version") != PILOT_P0_SUITE_SCHEMA
            or p0_suite.get("status") != "PASS_MECHANICAL_ONLY"
            or p0_suite.get("runtime_commit") != expected_commit
            or not _self_hash_matches(p0_suite, "suite_sha256")
        ):
            raise RuntimeError("estimator pilot P0 suite is invalid")
    finalization = finalize_pilot_results(
        run_root=run_root,
        expected_commit=expected_commit,
        expected_stage_jobs={
            arm: str(jobs["stage"][arm])
            for arm in PILOT_ARM_ORDER
        },
    )
    finalization["deployment_path"] = str(deployment_path)
    finalization["deployment_file_sha256"] = sha256_file(deployment_path)
    finalization["finalizer_submission_path"] = str(submission_path)
    finalization["finalizer_submission_file_sha256"] = sha256_file(
        submission_path
    )
    finalization["p0_suite_path"] = (
        str(p0_suite_path) if p0_suite_path.is_file() else None
    )
    finalization["p0_suite_file_sha256"] = (
        sha256_file(p0_suite_path) if p0_suite_path.is_file() else None
    )
    finalization.pop("finalization_sha256")
    finalization["finalization_sha256"] = canonical_sha256(finalization)
    output = run_root / "control" / "pilot_finalization.json"
    if output.exists():
        raise FileExistsError("estimator pilot finalization already exists")
    _atomic_write_json(output, finalization)
    print(json.dumps(finalization, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
