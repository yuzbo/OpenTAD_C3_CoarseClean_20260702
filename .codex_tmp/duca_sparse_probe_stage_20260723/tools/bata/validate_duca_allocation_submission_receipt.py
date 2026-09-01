from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from tools.bata.export_duca_allocation_ceiling_inputs import sha256


ROLES = ("gate", "export", "diagnostics", "candidate", "completion")


def validate_submission_receipt(
    *,
    submission_json: str | Path,
    submission_token: str,
    expected_commit: str,
    suite_manifest_json: str | Path,
    suite_manifest_sha256: str,
    role: str,
    current_job_id: str | int,
    gate_json: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(submission_json).resolve()
    manifest_path = Path(suite_manifest_json).resolve()
    payload = _load(path)
    required = {
        "schema_version",
        "git_commit",
        "submission_token",
        "submission_intent_json",
        "submission_intent_sha256",
        "suite_manifest_json",
        "suite_manifest_sha256",
        "run_root",
        "target_cluster",
        "jobs_tsv",
        "jobs_tsv_sha256",
        "jobs",
    }
    if set(payload) != required:
        raise ValueError("strict allocation submission-receipt fields mismatch")
    if payload.get("schema_version") != "duca_allocation_training_suite_submission_v3":
        raise ValueError("allocation submission-receipt schema mismatch")
    if re.fullmatch(r"[0-9a-f]{64}", submission_token) is None:
        raise ValueError("allocation submission token is invalid")
    if payload.get("submission_token") != submission_token:
        raise ValueError("allocation submission token mismatch")
    if payload.get("git_commit") != expected_commit:
        raise ValueError("allocation submission commit mismatch")
    if payload.get("target_cluster") != "n16r4":
        raise ValueError("allocation submission cluster mismatch")
    if Path(str(payload.get("suite_manifest_json", ""))).resolve() != manifest_path:
        raise ValueError("allocation submission manifest path mismatch")
    if (
        payload.get("suite_manifest_sha256") != suite_manifest_sha256
        or sha256(manifest_path) != suite_manifest_sha256
    ):
        raise ValueError("allocation submission manifest hash mismatch")
    intent_path = Path(str(payload.get("submission_intent_json", ""))).resolve()
    if (
        not intent_path.is_file()
        or sha256(intent_path) != payload.get("submission_intent_sha256")
        or sha256(intent_path) != submission_token
    ):
        raise ValueError("allocation submission intent/token binding mismatch")
    intent = _load(intent_path)
    if set(intent) != {
        "schema_version",
        "git_commit",
        "target_cluster",
        "run_root",
        "manifest_sha256",
        "job_files",
        "mode",
    }:
        raise ValueError("strict allocation submission-intent fields mismatch")
    if (
        intent.get("schema_version")
        != "duca_allocation_training_suite_submission_intent_v1"
        or intent.get("git_commit") != expected_commit
        or intent.get("target_cluster") != "n16r4"
        or intent.get("manifest_sha256") != suite_manifest_sha256
        or intent.get("mode") != "submit"
    ):
        raise ValueError("allocation submission-intent binding mismatch")
    run_root = Path(str(payload.get("run_root", ""))).resolve()
    if not run_root.is_dir() or path != run_root / "submission.json":
        raise ValueError("allocation submission path/run root mismatch")
    if manifest_path != run_root / "suite_manifest.json":
        raise ValueError("allocation suite manifest is outside the run root")
    if intent_path != run_root / "submission_intent.json":
        raise ValueError("allocation submission intent is outside the run root")
    if Path(str(intent.get("run_root", ""))).resolve() != run_root:
        raise ValueError("allocation submission-intent run root mismatch")
    job_files = intent.get("job_files")
    if not isinstance(job_files, Mapping) or set(job_files) != set(ROLES):
        raise ValueError("allocation submission-intent job-file set mismatch")
    for job_role in ROLES:
        job_path = run_root / "jobs" / (
            "diagnostics.sbatch"
            if job_role == "diagnostics"
            else f"{job_role}.sbatch"
        )
        if not job_path.is_file() or sha256(job_path) != job_files[job_role]:
            raise ValueError(f"allocation generated job file changed: {job_role}")
    jobs_tsv = Path(str(payload.get("jobs_tsv", ""))).resolve()
    if (
        jobs_tsv != run_root / "jobs.tsv"
        or not jobs_tsv.is_file()
        or sha256(jobs_tsv) != payload.get("jobs_tsv_sha256")
    ):
        raise ValueError("allocation jobs.tsv binding mismatch")
    jobs = payload.get("jobs")
    if not isinstance(jobs, Mapping) or set(jobs) != set(ROLES):
        raise ValueError("allocation submission job-role set mismatch")
    normalized: dict[str, dict[str, Any]] = {}
    for job_role in ROLES:
        item = jobs[job_role]
        if not isinstance(item, Mapping) or set(item) != {
            "job_id",
            "cluster",
            "dependency",
            "job_name",
            "job_file",
        }:
            raise ValueError(f"strict allocation job receipt mismatch: {job_role}")
        job_id = str(item["job_id"])
        if re.fullmatch(r"[0-9]+", job_id) is None:
            raise ValueError(f"allocation job ID is invalid: {job_role}")
        if item.get("cluster") != "n16r4":
            raise ValueError(f"allocation job cluster mismatch: {job_role}")
        normalized[job_role] = {
            "job_id": job_id,
            "cluster": "n16r4",
            "dependency": item.get("dependency"),
            "job_name": str(item.get("job_name")),
            "job_file": str(Path(str(item.get("job_file", ""))).resolve()),
        }
    expected_dependencies = {
        "gate": None,
        "export": f"afterok:{normalized['gate']['job_id']}",
        "diagnostics": f"afterok:{normalized['export']['job_id']}",
        "candidate": f"afterok:{normalized['diagnostics']['job_id']}",
        "completion": f"afterok:{normalized['candidate']['job_id']}",
    }
    short_commit = expected_commit[:7]
    expected_job_names = {
        "gate": f"dac-gate-{short_commit}",
        "export": f"dac-export-{short_commit}",
        "diagnostics": f"dac-diag-{short_commit}",
        "candidate": f"dac-cand-{short_commit}",
        "completion": f"dac-done-{short_commit}",
    }
    for job_role, dependency in expected_dependencies.items():
        if normalized[job_role]["dependency"] != dependency:
            raise ValueError(f"allocation dependency mismatch: {job_role}")
        if normalized[job_role]["job_name"] != expected_job_names[job_role]:
            raise ValueError(f"allocation job name mismatch: {job_role}")
    with jobs_tsv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected_columns = [
        "role",
        "job_id",
        "cluster",
        "dependency",
        "job_name",
        "job_file",
    ]
    if (
        len(rows) != len(ROLES)
        or not rows
        or list(rows[0]) != expected_columns
        or [row.get("role") for row in rows] != list(ROLES)
    ):
        raise ValueError("allocation jobs.tsv structure mismatch")
    for row in rows:
        job_role = str(row["role"])
        expected_dependency = expected_dependencies[job_role] or "none"
        expected_job_path = run_root / "jobs" / (
            "diagnostics.sbatch"
            if job_role == "diagnostics"
            else f"{job_role}.sbatch"
        )
        if (
            row.get("job_id") != normalized[job_role]["job_id"]
            or row.get("cluster") != "n16r4"
            or row.get("dependency") != expected_dependency
            or row.get("job_name") != normalized[job_role]["job_name"]
            or Path(str(row.get("job_file", ""))).resolve() != expected_job_path
            or normalized[job_role]["job_file"] != str(expected_job_path)
        ):
            raise ValueError(f"allocation jobs.tsv row mismatch: {job_role}")
    if role not in ROLES:
        raise ValueError(f"unknown allocation submission role: {role}")
    if str(current_job_id) != normalized[role]["job_id"]:
        raise ValueError(f"current Slurm job does not match receipt role {role}")
    result = {
        "schema_version": "duca_allocation_submission_validation_v1",
        "validation_passed": True,
        "submission_json": str(path),
        "submission_json_sha256": sha256(path),
        "submission_token": submission_token,
        "role": role,
        "current_job_id": str(current_job_id),
        "jobs": normalized,
    }
    if gate_json is not None:
        gate = _load(Path(gate_json).resolve())
        if gate.get("gate_passed") is not True:
            raise ValueError("allocation real gate did not pass")
        if gate.get("git_commit") != expected_commit:
            raise ValueError("allocation gate commit mismatch")
        if gate.get("execution_cluster") != "n16r4":
            raise ValueError("allocation gate cluster mismatch")
        if str(gate.get("gate_job_id")) != normalized["gate"]["job_id"]:
            raise ValueError("allocation gate Job ID mismatch")
        if gate.get("submission_token") != submission_token:
            raise ValueError("allocation gate submission token mismatch")
        if gate.get("suite_manifest_json_sha256") != suite_manifest_sha256:
            raise ValueError("allocation gate suite-manifest mismatch")
        expected_gate_validation = dict(result)
        expected_gate_validation["role"] = "gate"
        expected_gate_validation["current_job_id"] = normalized["gate"]["job_id"]
        if gate.get("submission_validation") != expected_gate_validation:
            raise ValueError("allocation gate submission receipt/DAG binding mismatch")
        scheduler_validation = gate.get("scheduler_validation")
        if not isinstance(scheduler_validation, Mapping):
            raise ValueError("allocation gate scheduler validation is missing")
        scheduler_path = Path(
            str(scheduler_validation.get("scheduler_receipt_json", ""))
        ).resolve()
        if (
            not scheduler_path.is_file()
            or sha256(scheduler_path)
            != scheduler_validation.get("scheduler_receipt_json_sha256")
            or scheduler_validation.get("submission_json_sha256") != sha256(path)
            or scheduler_validation.get("submission_token") != submission_token
            or scheduler_validation.get("jobs") != normalized
        ):
            raise ValueError("allocation gate scheduler receipt binding mismatch")
        _revalidate_scheduler_snapshot_hashes(scheduler_path)
    return result


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected an object")
    return payload


def _revalidate_scheduler_snapshot_hashes(receipt_path: Path) -> None:
    post = _load(receipt_path)
    if post.get("phase") != "post_release":
        raise ValueError("allocation scheduler receipt is not post-release evidence")
    pre_path = Path(str(post.get("pre_release_snapshot_json", ""))).resolve()
    if (
        not pre_path.is_file()
        or sha256(pre_path) != post.get("pre_release_snapshot_json_sha256")
    ):
        raise ValueError("allocation pre-release scheduler snapshot changed")
    for phase, payload in (("pre_release", _load(pre_path)), ("post_release", post)):
        if payload.get("phase") != phase:
            raise ValueError("allocation scheduler snapshot phase mismatch")
        raw_jobs = payload.get("raw_jobs")
        if not isinstance(raw_jobs, Mapping) or set(raw_jobs) != set(ROLES):
            raise ValueError("allocation scheduler raw job set mismatch")
        for role in ROLES:
            item = raw_jobs[role]
            if not isinstance(item, Mapping):
                raise ValueError(f"allocation scheduler raw job is invalid: {role}")
            raw_path = Path(str(item.get("raw_snapshot", ""))).resolve()
            if (
                not raw_path.is_file()
                or sha256(raw_path) != item.get("raw_snapshot_sha256")
            ):
                raise ValueError(
                    f"allocation scheduler raw snapshot changed: {phase}/{role}"
                )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a DUCA allocation Slurm submission receipt."
    )
    parser.add_argument("--submission-json", required=True)
    parser.add_argument("--submission-token", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--suite-manifest-json", required=True)
    parser.add_argument("--suite-manifest-sha256", required=True)
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--current-job-id", required=True)
    parser.add_argument("--gate-json")
    args = parser.parse_args(argv)
    result = validate_submission_receipt(
        submission_json=args.submission_json,
        submission_token=args.submission_token,
        expected_commit=args.expected_commit,
        suite_manifest_json=args.suite_manifest_json,
        suite_manifest_sha256=args.suite_manifest_sha256,
        role=args.role,
        current_job_id=args.current_job_id,
        gate_json=args.gate_json,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
