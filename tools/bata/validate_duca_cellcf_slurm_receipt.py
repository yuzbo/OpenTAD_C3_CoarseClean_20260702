from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Sequence


ACTIVE_OR_SUCCESS_STATES = {
    "COMPLETED",
    "COMPLETING",
    "CONFIGURING",
    "PENDING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESIZING",
    "RUNNING",
    "SIGNALING",
    "STAGE_OUT",
    "SUSPENDED",
}

SUBMISSION_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,256}")
JOB_NAME_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,128}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
PENDING_LIKE_STATES = {
    "PENDING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
}


def _run(
    command: list[str],
    *,
    runner: Callable[..., Any] = subprocess.run,
    allow_missing_job: bool = False,
    env_overrides: dict[str, str] | None = None,
) -> str:
    run_env = None
    if env_overrides:
        run_env = os.environ.copy()
        run_env.update(env_overrides)
    result = runner(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=run_env,
    )
    if int(result.returncode) != 0:
        stderr = str(result.stderr or "").strip()
        if allow_missing_job and "invalid job id specified" in stderr.lower():
            return ""
        raise RuntimeError(f"Slurm query failed ({' '.join(command)}): {stderr}")
    return str(result.stdout or "")


def _run_bytes(
    command: list[str],
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> bytes:
    result = runner(
        command,
        text=False,
        capture_output=True,
        check=False,
    )
    stdout = result.stdout or b""
    stderr = result.stderr or b""
    if isinstance(stdout, str):
        stdout = stdout.encode("utf-8")
    if isinstance(stderr, str):
        stderr = stderr.encode("utf-8")
    if int(result.returncode) != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Slurm query failed ({' '.join(command)}): {detail}"
        )
    return bytes(stdout)


def _rows(
    output: str,
    *,
    fields: int,
    strip_fields: bool = True,
) -> list[list[str]]:
    parsed = []
    for raw in output.replace("\r", "").splitlines():
        if not raw.strip():
            continue
        values = raw.split("|")
        if len(values) != fields:
            raise ValueError(f"unexpected Slurm row: {raw!r}")
        parsed.append(
            [value.strip() for value in values]
            if strip_fields
            else values
        )
    return parsed


def _state(value: str) -> str:
    normalized = value.strip().upper().split()[0].rstrip("+")
    if normalized not in ACTIVE_OR_SUCCESS_STATES:
        raise ValueError(f"receipt points to a non-reusable Slurm state: {value}")
    return normalized


def _normalize_dependency(value: str) -> str:
    normalized = value.strip()
    if normalized.lower() in {"", "(null)", "none"}:
        return ""
    return normalized


def _canonical_submit_line(
    *,
    cluster: str,
    job_name: str,
    token: str,
    dependency: str,
    job_file: Path,
    submitted_with_hold: bool,
) -> str:
    tokens = [
        "sbatch",
        "--parsable",
    ]
    if submitted_with_hold:
        tokens.append("--hold")
    tokens.extend(
        [
            f"--clusters={cluster}",
            f"--job-name={job_name}",
            f"--comment={token}",
        ]
    )
    if dependency:
        tokens.append(f"--dependency={dependency}")
    tokens.append(str(job_file))
    return " ".join(tokens)


def _validate_submit_line(value: str, *, expected_line: str) -> None:
    if value != expected_line:
        raise ValueError("Slurm submit line does not match the canonical submission")


def _verify_scheduler_batch_script(
    *,
    job_id: int,
    cluster: str,
    expected_sha256: str,
    runner: Callable[..., Any],
) -> None:
    content = _run_bytes(
        [
            "scontrol",
            f"--clusters={cluster}",
            "write",
            "batch_script",
            str(job_id),
            "-",
        ],
        runner=runner,
    )
    observed_sha256 = hashlib.sha256(content).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "Slurm stored batch script does not match the prepared script hash"
        )


def _squeue_payload(output: str, *, cluster: str) -> dict[str, Any] | None:
    normalized = output.replace("\r", "")
    if not normalized.strip():
        return None
    json_start = normalized.find("{")
    if json_start < 0:
        raise ValueError("Slurm squeue JSON payload is missing")
    prefix = normalized[:json_start]
    prefix_lines = [line.strip() for line in prefix.splitlines() if line.strip()]
    if any(line != f"CLUSTER: {cluster}" for line in prefix_lines):
        raise ValueError("Slurm squeue JSON has an unexpected prefix")
    try:
        payload = json.loads(normalized[json_start:])
    except json.JSONDecodeError as exc:
        raise ValueError("Slurm squeue JSON payload is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("Slurm squeue JSON payload must be an object")
    errors = payload.get("errors", [])
    if not isinstance(errors, list) or errors:
        raise ValueError("Slurm squeue JSON reports an error")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Slurm squeue JSON jobs field is invalid")
    if not jobs:
        return None
    if len(jobs) != 1 or not isinstance(jobs[0], dict):
        raise ValueError("receipt resolved to multiple live Slurm jobs")
    return jobs[0]


def _live_state(value: Any) -> str:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError("Slurm live job state is ambiguous")
    states = [_state(item) for item in value]
    non_pending = [item for item in states if item not in PENDING_LIKE_STATES]
    return non_pending[0] if non_pending else "PENDING"


def _validate_live_identity(
    job: dict[str, Any],
    *,
    job_id: int,
    job_name: str,
    token: str,
    cluster: str,
    job_file: Path,
) -> str:
    if job.get("job_id") != job_id:
        raise ValueError(
            f"Slurm job id mismatch: expected {job_id}, got {job.get('job_id')}"
        )
    if job.get("name") != job_name:
        raise ValueError(
            f"Slurm job name mismatch: expected {job_name}, got {job.get('name')}"
        )
    if job.get("comment") != token:
        raise ValueError("Slurm job comment does not match the submission token")
    if job.get("cluster") != cluster:
        raise ValueError(
            f"Slurm cluster mismatch: expected {cluster}, got {job.get('cluster')}"
        )
    if job.get("command") != str(job_file):
        raise ValueError("Slurm live command does not match the bound job file")
    return _live_state(job.get("job_state"))


def _afterok_job_ids(value: str) -> frozenset[int]:
    normalized = _normalize_dependency(value)
    if not normalized:
        return frozenset()
    parts = normalized.split(":")
    if len(parts) < 2 or parts[0] != "afterok":
        raise ValueError(f"unsupported formal Slurm dependency: {normalized}")
    raw_ids = parts[1:]
    if any(re.fullmatch(r"[1-9][0-9]*", item) is None for item in raw_ids):
        raise ValueError(f"invalid formal Slurm dependency: {normalized}")
    job_ids = frozenset(int(item) for item in raw_ids)
    if len(job_ids) != len(raw_ids):
        raise ValueError(f"duplicate formal Slurm dependency: {normalized}")
    return job_ids


def _live_afterok_job_ids(value: str) -> frozenset[int]:
    normalized = _normalize_dependency(value)
    if not normalized:
        return frozenset()
    if "," not in normalized and "(" not in normalized:
        return _afterok_job_ids(normalized)

    rendered = normalized.split(",")
    job_ids: list[int] = []
    for item in rendered:
        match = re.fullmatch(
            r"afterok:([1-9][0-9]*)\(unfulfilled\)",
            item,
        )
        if match is None:
            raise ValueError(f"invalid live Slurm dependency: {normalized}")
        job_ids.append(int(match.group(1)))
    unique_ids = frozenset(job_ids)
    if len(unique_ids) != len(job_ids):
        raise ValueError(f"duplicate live Slurm dependency: {normalized}")
    return unique_ids


def _timestamp(value: str, *, label: str) -> datetime:
    normalized = value.strip()
    if normalized.lower() in {"", "unknown", "none", "n/a"}:
        raise ValueError(f"Slurm {label} timestamp is unavailable")
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}", normalized) is None:
        raise ValueError(f"invalid Slurm {label} timestamp: {value}")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid Slurm {label} timestamp: {value}") from exc


def _validate_satisfied_afterok_predecessors(
    *,
    expected_ids: frozenset[int],
    remaining_ids: frozenset[int],
    target_started: bool,
    target_start: str,
    cluster: str,
    runner: Callable[..., Any],
) -> None:
    if target_started and remaining_ids:
        raise ValueError(
            "Slurm target started while afterok dependencies remain: "
            f"{sorted(remaining_ids)}"
        )
    satisfied_ids = expected_ids - remaining_ids
    if not satisfied_ids:
        return
    output = _run(
        [
            "sacct",
            "--clusters",
            cluster,
            "--allocations",
            "--noheader",
            "--parsable2",
            "--jobs",
            ",".join(str(job_id) for job_id in sorted(satisfied_ids)),
            "--format",
            "JobIDRaw,State,ExitCode,End,Start,Cluster",
        ],
        runner=runner,
        env_overrides={"SLURM_TIME_FORMAT": "standard"},
    )
    rows = _rows(output, fields=6)
    by_job_id: dict[int, list[str]] = {}
    for row in rows:
        if re.fullmatch(r"[1-9][0-9]*", row[0]) is None:
            continue
        job_id = int(row[0])
        if job_id in satisfied_ids:
            if job_id in by_job_id:
                raise ValueError(f"afterok predecessor {job_id} is ambiguous")
            by_job_id[job_id] = row
    if set(by_job_id) != set(satisfied_ids):
        missing = sorted(set(satisfied_ids) - set(by_job_id))
        raise ValueError(f"afterok predecessor accounting is incomplete: {missing}")

    target_started_at = (
        _timestamp(target_start, label="target start")
        if target_started
        else None
    )
    for job_id, row in by_job_id.items():
        state = row[1].strip().upper().split()[0].rstrip("+")
        if state != "COMPLETED" or row[2] != "0:0":
            raise ValueError(
                f"afterok predecessor {job_id} is not successfully completed: "
                f"state={row[1]!r}, exit_code={row[2]!r}"
            )
        if row[5] != cluster:
            raise ValueError(
                f"afterok predecessor {job_id} cluster mismatch: "
                f"expected {cluster}, got {row[5]}"
            )
        predecessor_end = _timestamp(row[3], label=f"predecessor {job_id} end")
        if target_started_at is not None and predecessor_end > target_started_at:
            raise ValueError(
                f"target started before afterok predecessor {job_id} completed"
            )


def _validate_identity(
    row: list[str],
    *,
    job_id: int,
    job_name: str,
    token: str,
    cluster: str | None,
) -> str:
    expected_id = str(job_id)
    if row[0] != expected_id:
        raise ValueError(f"Slurm job id mismatch: expected {expected_id}, got {row[0]}")
    if row[1] != job_name:
        raise ValueError(f"Slurm job name mismatch: expected {job_name}, got {row[1]}")
    state = _state(row[2])
    if row[3] != token:
        raise ValueError("Slurm job comment does not match the submission token")
    if cluster is not None and row[4] != cluster:
        raise ValueError(f"Slurm cluster mismatch: expected {cluster}, got {row[4]}")
    return state


def validate_slurm_receipt(
    *,
    job_id: int,
    job_name: str,
    token: str,
    cluster: str,
    job_file: str | os.PathLike[str],
    job_file_sha256: str,
    dependency: str = "",
    require_scheduler_script: bool = False,
    submitted_with_hold: bool = False,
    require_current_user_hold: bool = False,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    if job_id <= 0:
        raise ValueError("job_id must be positive")
    if SUBMISSION_TOKEN_PATTERN.fullmatch(token) is None:
        raise ValueError("invalid Slurm submission token")
    if JOB_NAME_PATTERN.fullmatch(job_name) is None:
        raise ValueError("invalid Slurm job name")
    if re.fullmatch(r"[A-Za-z0-9._-]+", cluster) is None:
        raise ValueError("invalid Slurm cluster identity")
    if SHA256_PATTERN.fullmatch(job_file_sha256) is None:
        raise ValueError("invalid bound Slurm job file hash")
    if require_current_user_hold and not submitted_with_hold:
        raise ValueError(
            "current user-hold proof requires a submission command with --hold"
        )
    raw_job_file = Path(job_file).expanduser()
    if not raw_job_file.is_absolute():
        raise ValueError("job_file must be an absolute path")
    try:
        resolved_job_file = raw_job_file.resolve(strict=True)
    except OSError as exc:
        raise ValueError("bound Slurm job file does not exist") from exc
    if not resolved_job_file.is_file():
        raise ValueError("bound Slurm job file is not a regular file")
    resolved_job_file_text = str(resolved_job_file)
    if os.name == "posix" and re.fullmatch(
        r"/[A-Za-z0-9._/-]+",
        resolved_job_file_text,
    ) is None:
        raise ValueError("bound Slurm job file path is not scheduler-safe")
    observed_job_file_sha256 = hashlib.sha256(
        resolved_job_file.read_bytes()
    ).hexdigest()
    if observed_job_file_sha256 != job_file_sha256:
        raise ValueError("bound Slurm job file hash mismatch")
    expected_dependency = _normalize_dependency(dependency)
    expected_dependency_ids = _afterok_job_ids(expected_dependency)
    expected_submit_line = _canonical_submit_line(
        cluster=cluster,
        job_name=job_name,
        token=token,
        dependency=expected_dependency,
        job_file=resolved_job_file,
        submitted_with_hold=submitted_with_hold,
    )

    live_output = _run(
        [
            "squeue",
            "--clusters",
            cluster,
            "--jobs",
            str(job_id),
            "--json",
        ],
        runner=runner,
        allow_missing_job=True,
    )
    live_job = _squeue_payload(live_output, cluster=cluster)
    live_state: str | None = None
    live_dependency: str | None = None
    live_dependency_ids = frozenset()
    if live_job is not None:
        live_state = _validate_live_identity(
            live_job,
            job_id=job_id,
            job_name=job_name,
            token=token,
            cluster=cluster,
            job_file=resolved_job_file,
        )
        raw_live_dependency = live_job.get("dependency", "")
        if raw_live_dependency is None:
            raw_live_dependency = ""
        if not isinstance(raw_live_dependency, str):
            raise ValueError("Slurm live dependency is invalid")
        live_dependency = _normalize_dependency(raw_live_dependency)
        if live_dependency:
            live_dependency_ids = _live_afterok_job_ids(live_dependency)
            if not live_dependency_ids.issubset(expected_dependency_ids):
                raise ValueError(
                    "Slurm live dependency mismatch: "
                    f"expected a remaining subset of {expected_dependency or '(none)'}, "
                    f"got {live_dependency}"
                )
            if live_state != "PENDING":
                raise ValueError(
                    "Slurm target started while dependencies remain: "
                    f"state={live_state}, dependency={live_dependency}"
                )
    if require_current_user_hold:
        if (
            live_job is None
            or live_state != "PENDING"
            or live_job.get("hold") is not True
            or live_job.get("state_reason") != "JobHeldUser"
        ):
            raise ValueError(
                "Slurm job is not currently PENDING under JobHeldUser"
            )

    accounting_output = _run(
        [
            "sacct",
            "--clusters",
            cluster,
            "--allocations",
            "--noheader",
            "--parsable2",
            "--jobs",
            str(job_id),
            "--format",
            (
                "JobIDRaw,JobName%128,State,Comment%256,Cluster,"
                "SubmitLine,Start,End"
            ),
        ],
        runner=runner,
        env_overrides={"SLURM_TIME_FORMAT": "standard"},
    )
    accounting_rows = [
        row
        for row in _rows(accounting_output, fields=8, strip_fields=False)
        if row[0] == str(job_id)
    ]
    if len(accounting_rows) == 1:
        accounting_comment = accounting_rows[0][3]
        if accounting_comment and accounting_comment != token:
            raise ValueError("Slurm job comment does not match the submission token")
        _validate_submit_line(
            accounting_rows[0][5],
            expected_line=expected_submit_line,
        )
        state = _validate_identity(
            accounting_rows[0][:3] + [token, accounting_rows[0][4]],
            job_id=job_id,
            job_name=job_name,
            token=token,
            cluster=cluster,
        )
        _validate_satisfied_afterok_predecessors(
            expected_ids=expected_dependency_ids,
            remaining_ids=live_dependency_ids,
            target_started=(
                state != "PENDING"
                or (live_state is not None and live_state != "PENDING")
            ),
            target_start=accounting_rows[0][6],
            cluster=cluster,
            runner=runner,
        )
        payload = {
            "ok": True,
            "source": "sacct",
            "job_id": job_id,
            "job_name": job_name,
            "state": state,
            "comment": token,
            "cluster": cluster,
            "job_file": resolved_job_file_text,
            "job_file_sha256": job_file_sha256,
            "dependency": expected_dependency or None,
        }
        if require_scheduler_script:
            _verify_scheduler_batch_script(
                job_id=job_id,
                cluster=cluster,
                expected_sha256=job_file_sha256,
                runner=runner,
            )
            payload["scheduler_script_verified"] = True
        if submitted_with_hold:
            payload["submission_command_held_verified"] = True
        if require_current_user_hold:
            payload["current_user_hold_verified"] = True
        return payload
    if accounting_rows:
        raise ValueError("receipt job is ambiguous in Slurm accounting")
    if live_state is not None and live_dependency_ids == expected_dependency_ids:
        if submitted_with_hold:
            raise ValueError(
                "submission-command hold proof requires the exact accounting submit line"
            )
        if expected_dependency_ids and live_state != "PENDING":
            raise ValueError(
                "dependent Slurm target has started without accounting proof"
            )
        payload = {
            "ok": True,
            "source": "squeue",
            "job_id": job_id,
            "job_name": job_name,
            "state": live_state,
            "comment": token,
            "cluster": cluster,
            "job_file": resolved_job_file_text,
            "job_file_sha256": job_file_sha256,
            "dependency": expected_dependency or None,
        }
        if require_scheduler_script:
            _verify_scheduler_batch_script(
                job_id=job_id,
                cluster=cluster,
                expected_sha256=job_file_sha256,
                runner=runner,
            )
            payload["scheduler_script_verified"] = True
        return payload
    if live_state is not None:
        raise ValueError(
            "Slurm dependency mismatch: "
            f"expected {expected_dependency or '(none)'}, "
            f"got {live_dependency or '(none)'}"
        )
    raise ValueError("receipt job is absent in Slurm accounting")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--comment", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--job-file", required=True)
    parser.add_argument("--job-file-sha256", required=True)
    parser.add_argument("--dependency", default="")
    parser.add_argument("--require-scheduler-script", action="store_true")
    parser.add_argument("--require-submitted-with-hold", action="store_true")
    parser.add_argument("--require-current-user-hold", action="store_true")
    args = parser.parse_args(argv)
    payload = validate_slurm_receipt(
        job_id=args.job_id,
        job_name=args.job_name,
        token=args.comment,
        cluster=args.cluster,
        job_file=args.job_file,
        job_file_sha256=args.job_file_sha256,
        dependency=args.dependency,
        require_scheduler_script=args.require_scheduler_script,
        submitted_with_hold=args.require_submitted_with_hold,
        require_current_user_hold=args.require_current_user_hold,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
