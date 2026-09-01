from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from tools.bata.export_duca_allocation_ceiling_inputs import (
    sha256,
    write_json_exclusive,
)
from tools.bata.validate_duca_allocation_submission_receipt import (
    ROLES,
    validate_submission_receipt,
)


_SCONTROL_FIELD = re.compile(r"(?:^| )([A-Za-z][A-Za-z0-9_:]*)=")


def capture_scheduler_snapshot(
    *,
    submission_json: str | Path,
    submission_token: str,
    expected_commit: str,
    suite_manifest_json: str | Path,
    suite_manifest_sha256: str,
    phase: str,
    raw_snapshots: Mapping[str, str | Path],
    output_json: str | Path,
    pre_release_snapshot_json: str | Path | None = None,
) -> dict[str, Any]:
    if phase not in {"pre_release", "post_release"}:
        raise ValueError("scheduler snapshot phase is invalid")
    submission_path = Path(submission_json).resolve()
    submission = validate_submission_receipt(
        submission_json=submission_path,
        submission_token=submission_token,
        expected_commit=expected_commit,
        suite_manifest_json=suite_manifest_json,
        suite_manifest_sha256=suite_manifest_sha256,
        role="gate",
        current_job_id=_gate_job_id(submission_path),
    )
    if set(raw_snapshots) != set(ROLES):
        raise ValueError("scheduler snapshot must contain every DAG role")
    raw_jobs = _validate_raw_jobs(
        raw_snapshots=raw_snapshots,
        phase=phase,
        submission_validation=submission,
    )
    pre_path: Path | None = None
    pre_sha: str | None = None
    if phase == "pre_release":
        if pre_release_snapshot_json is not None:
            raise ValueError("pre-release snapshot cannot reference an earlier snapshot")
    else:
        if pre_release_snapshot_json is None:
            raise ValueError("post-release snapshot requires the pre-release snapshot")
        pre_path = Path(pre_release_snapshot_json).resolve()
        expected_pre_path = (
            Path(submission["submission_json"]).resolve().parent
            / "scheduler"
            / "pre_release.snapshot.json"
        )
        if pre_path != expected_pre_path:
            raise ValueError("pre-release scheduler snapshot path mismatch")
        pre_payload = _load(pre_path)
        _validate_snapshot_payload(
            pre_payload,
            snapshot_path=pre_path,
            expected_phase="pre_release",
            submission_validation=submission,
        )
        pre_sha = sha256(pre_path)
    result = {
        "schema_version": "duca_allocation_scheduler_snapshot_v1",
        "phase": phase,
        "execution_cluster": "n16r4",
        "submission_validation": submission,
        "raw_jobs": raw_jobs,
        "pre_release_snapshot_json": None if pre_path is None else str(pre_path),
        "pre_release_snapshot_json_sha256": pre_sha,
    }
    write_json_exclusive(output_json, result)
    return result


def validate_scheduler_receipt(
    *,
    scheduler_receipt_json: str | Path,
    submission_json: str | Path,
    submission_token: str,
    expected_commit: str,
    suite_manifest_json: str | Path,
    suite_manifest_sha256: str,
) -> dict[str, Any]:
    receipt_path = Path(scheduler_receipt_json).resolve()
    submission_path = Path(submission_json).resolve()
    if receipt_path != submission_path.parent / "scheduler_receipt.json":
        raise ValueError("scheduler receipt path is outside the submitted run root")
    submission = validate_submission_receipt(
        submission_json=submission_path,
        submission_token=submission_token,
        expected_commit=expected_commit,
        suite_manifest_json=suite_manifest_json,
        suite_manifest_sha256=suite_manifest_sha256,
        role="gate",
        current_job_id=_gate_job_id(submission_path),
    )
    payload = _load(receipt_path)
    _validate_snapshot_payload(
        payload,
        snapshot_path=receipt_path,
        expected_phase="post_release",
        submission_validation=submission,
    )
    pre_path = Path(str(payload["pre_release_snapshot_json"])).resolve()
    if sha256(pre_path) != payload["pre_release_snapshot_json_sha256"]:
        raise ValueError("pre-release scheduler snapshot hash mismatch")
    pre_payload = _load(pre_path)
    _validate_snapshot_payload(
        pre_payload,
        snapshot_path=pre_path,
        expected_phase="pre_release",
        submission_validation=submission,
    )
    return {
        "schema_version": "duca_allocation_scheduler_validation_v1",
        "validation_passed": True,
        "scheduler_receipt_json": str(receipt_path),
        "scheduler_receipt_json_sha256": sha256(receipt_path),
        "pre_release_snapshot_json": str(pre_path),
        "pre_release_snapshot_json_sha256": sha256(pre_path),
        "submission_json_sha256": sha256(submission_path),
        "submission_token": submission_token,
        "jobs": submission["jobs"],
    }


def _validate_snapshot_payload(
    payload: Mapping[str, Any],
    *,
    snapshot_path: Path,
    expected_phase: str,
    submission_validation: Mapping[str, Any],
) -> None:
    required = {
        "schema_version",
        "phase",
        "execution_cluster",
        "submission_validation",
        "raw_jobs",
        "pre_release_snapshot_json",
        "pre_release_snapshot_json_sha256",
    }
    if set(payload) != required:
        raise ValueError("strict scheduler snapshot fields mismatch")
    if payload.get("schema_version") != "duca_allocation_scheduler_snapshot_v1":
        raise ValueError("scheduler snapshot schema mismatch")
    if (
        payload.get("phase") != expected_phase
        or payload.get("execution_cluster") != "n16r4"
        or payload.get("submission_validation") != submission_validation
    ):
        raise ValueError("scheduler snapshot submission/phase binding mismatch")
    if expected_phase == "pre_release":
        if (
            payload.get("pre_release_snapshot_json") is not None
            or payload.get("pre_release_snapshot_json_sha256") is not None
        ):
            raise ValueError("pre-release snapshot has an invalid parent")
    else:
        parent = Path(str(payload.get("pre_release_snapshot_json", ""))).resolve()
        if (
            parent == snapshot_path
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(payload.get("pre_release_snapshot_json_sha256", "")),
            )
            is None
        ):
            raise ValueError("post-release scheduler parent binding is invalid")
    raw_jobs = payload.get("raw_jobs")
    if not isinstance(raw_jobs, Mapping) or set(raw_jobs) != set(ROLES):
        raise ValueError("scheduler snapshot job-role set mismatch")
    _validate_raw_jobs(
        raw_snapshots={
            role: str(raw_jobs[role]["raw_snapshot"])
            for role in ROLES
        },
        phase=expected_phase,
        submission_validation=submission_validation,
        expected_payload=raw_jobs,
    )


def _validate_raw_jobs(
    *,
    raw_snapshots: Mapping[str, str | Path],
    phase: str,
    submission_validation: Mapping[str, Any],
    expected_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    jobs = submission_validation["jobs"]
    run_root = Path(submission_validation["submission_json"]).resolve().parent
    repo_root = Path(__file__).resolve().parents[2]
    result: dict[str, Any] = {}
    for role in ROLES:
        raw_path = Path(raw_snapshots[role]).resolve()
        expected_raw_path = run_root / "scheduler" / f"{phase}.{role}.scontrol.txt"
        if raw_path != expected_raw_path or not raw_path.is_file():
            raise ValueError(f"scheduler raw snapshot path is invalid: {role}")
        raw_sha = sha256(raw_path)
        fields = parse_scontrol_record(raw_path.read_text(encoding="utf-8"))
        expected = jobs[role]
        expected_dependency = (
            "(null)"
            if expected["dependency"] is None
            else f"{expected['dependency']}(unfulfilled)"
        )
        if fields.get("JobId") != expected["job_id"]:
            raise ValueError(f"scheduler JobId mismatch: {role}")
        if fields.get("JobName") != expected["job_name"]:
            raise ValueError(f"scheduler JobName mismatch: {role}")
        if fields.get("Dependency") != expected_dependency:
            raise ValueError(f"scheduler dependency mismatch: {role}")
        if Path(str(fields.get("Command", ""))).resolve() != Path(
            expected["job_file"]
        ).resolve():
            raise ValueError(f"scheduler command path mismatch: {role}")
        if Path(str(fields.get("WorkDir", ""))).resolve() != repo_root:
            raise ValueError(f"scheduler work directory mismatch: {role}")
        if fields.get("BatchFlag") != "1":
            raise ValueError(f"scheduler batch flag mismatch: {role}")
        if "gres/gpu=1" not in str(fields.get("ReqTRES", "")):
            raise ValueError(f"scheduler GPU request mismatch: {role}")
        if phase == "pre_release":
            if (
                fields.get("JobState") != "PENDING"
                or fields.get("Reason") != "JobHeldUser"
                or fields.get("Priority") != "0"
            ):
                raise ValueError(f"scheduler pre-release hold mismatch: {role}")
        else:
            if fields.get("Reason") == "JobHeldUser":
                raise ValueError(f"scheduler post-release job remains held: {role}")
            allowed_states = {"PENDING", "RUNNING"}
            if fields.get("JobState") not in allowed_states:
                raise ValueError(f"scheduler post-release state mismatch: {role}")
            if role != "gate" and fields.get("JobState") != "PENDING":
                raise ValueError(f"scheduler child ran before the gate: {role}")
        item = {
            "raw_snapshot": str(raw_path),
            "raw_snapshot_sha256": raw_sha,
            "job_id": fields["JobId"],
            "job_name": fields["JobName"],
            "job_state": fields["JobState"],
            "reason": fields["Reason"],
            "priority": fields["Priority"],
            "dependency": fields["Dependency"],
            "command": fields["Command"],
            "work_dir": fields["WorkDir"],
            "req_tres": fields["ReqTRES"],
        }
        if expected_payload is not None and expected_payload.get(role) != item:
            raise ValueError(f"scheduler sealed snapshot changed: {role}")
        result[role] = item
    return result


def parse_scontrol_record(text: str) -> dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("scontrol snapshot must contain exactly one record")
    line = lines[0]
    matches = list(_SCONTROL_FIELD.finditer(line))
    if not matches:
        raise ValueError("scontrol snapshot contains no fields")
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        if key in fields:
            raise ValueError(f"duplicate scontrol field: {key}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        fields[key] = line[match.end() : end].strip()
    required = {
        "JobId",
        "JobName",
        "Priority",
        "JobState",
        "Reason",
        "Dependency",
        "BatchFlag",
        "ReqTRES",
        "Command",
        "WorkDir",
    }
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"scontrol snapshot is missing fields: {sorted(missing)}")
    return fields


def _gate_job_id(submission_path: Path) -> str:
    payload = _load(submission_path)
    jobs = payload.get("jobs")
    if not isinstance(jobs, Mapping) or not isinstance(jobs.get("gate"), Mapping):
        raise ValueError("submission receipt is missing the gate Job ID")
    return str(jobs["gate"].get("job_id", ""))


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected an object")
    return payload


def _raw_arguments(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        role, separator, path = value.partition("=")
        if not separator or role not in ROLES or role in result:
            raise ValueError(f"invalid scheduler raw snapshot argument: {value}")
        result[role] = Path(path).resolve()
    if set(result) != set(ROLES):
        raise ValueError("scheduler raw snapshot arguments are incomplete")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture or validate the DUCA allocation Slurm DAG receipt."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--submission-json", required=True)
    capture.add_argument("--submission-token", required=True)
    capture.add_argument("--expected-commit", required=True)
    capture.add_argument("--suite-manifest-json", required=True)
    capture.add_argument("--suite-manifest-sha256", required=True)
    capture.add_argument(
        "--phase",
        required=True,
        choices=("pre_release", "post_release"),
    )
    capture.add_argument("--raw", action="append", default=[])
    capture.add_argument("--pre-release-snapshot-json")
    capture.add_argument("--output-json", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--scheduler-receipt-json", required=True)
    validate.add_argument("--submission-json", required=True)
    validate.add_argument("--submission-token", required=True)
    validate.add_argument("--expected-commit", required=True)
    validate.add_argument("--suite-manifest-json", required=True)
    validate.add_argument("--suite-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "capture":
        result = capture_scheduler_snapshot(
            submission_json=args.submission_json,
            submission_token=args.submission_token,
            expected_commit=args.expected_commit,
            suite_manifest_json=args.suite_manifest_json,
            suite_manifest_sha256=args.suite_manifest_sha256,
            phase=args.phase,
            raw_snapshots=_raw_arguments(args.raw),
            output_json=args.output_json,
            pre_release_snapshot_json=args.pre_release_snapshot_json,
        )
    else:
        result = validate_scheduler_receipt(
            scheduler_receipt_json=args.scheduler_receipt_json,
            submission_json=args.submission_json,
            submission_token=args.submission_token,
            expected_commit=args.expected_commit,
            suite_manifest_json=args.suite_manifest_json,
            suite_manifest_sha256=args.suite_manifest_sha256,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
