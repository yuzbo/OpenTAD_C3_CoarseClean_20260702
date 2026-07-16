from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import re
import shlex
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


def _rows(output: str, *, fields: int) -> list[list[str]]:
    parsed = []
    for raw in output.replace("\r", "").splitlines():
        if not raw.strip():
            continue
        values = raw.split("|")
        if len(values) != fields:
            raise ValueError(f"unexpected Slurm row: {raw!r}")
        parsed.append([value.strip() for value in values])
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


def _submit_line_dependency(value: str) -> str:
    tokens = shlex.split(value)
    dependencies: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--dependency":
            if index + 1 >= len(tokens):
                raise ValueError("Slurm submit line has an incomplete dependency option")
            dependencies.append(tokens[index + 1])
            index += 2
            continue
        if token.startswith("--dependency="):
            dependencies.append(token.partition("=")[2])
        index += 1
    if len(dependencies) > 1:
        raise ValueError("Slurm submit line has multiple dependency options")
    return _normalize_dependency(dependencies[0] if dependencies else "")


def _submit_line_comment(value: str) -> str:
    tokens = shlex.split(value)
    comments: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--comment":
            if index + 1 >= len(tokens):
                raise ValueError("Slurm submit line has an incomplete comment option")
            comments.append(tokens[index + 1])
            index += 2
            continue
        if token.startswith("--comment="):
            comments.append(token.partition("=")[2])
        index += 1
    if len(comments) > 1:
        raise ValueError("Slurm submit line has multiple comment options")
    return comments[0].strip() if comments else ""


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
    dependency: str = "",
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    if job_id <= 0:
        raise ValueError("job_id must be positive")
    if re.fullmatch(r"[A-Za-z0-9._-]+", cluster) is None:
        raise ValueError("invalid Slurm cluster identity")
    expected_dependency = _normalize_dependency(dependency)
    expected_dependency_ids = _afterok_job_ids(expected_dependency)

    live_output = _run(
        [
            "squeue",
            "--clusters",
            cluster,
            "--noheader",
            "--jobs",
            str(job_id),
            "--format",
            "%i|%128j|%T|%256k|%E",
        ],
        runner=runner,
        allow_missing_job=True,
    )
    live_rows = _rows(live_output, fields=5)
    live_state: str | None = None
    live_dependency: str | None = None
    live_dependency_ids = frozenset()
    if live_rows:
        if len(live_rows) != 1:
            raise ValueError("receipt resolved to multiple live Slurm jobs")
        live_state = _validate_identity(
            live_rows[0][:4] + [cluster],
            job_id=job_id,
            job_name=job_name,
            token=token,
            cluster=cluster,
        )
        live_dependency = _normalize_dependency(live_rows[0][4])
        if live_dependency:
            live_dependency_ids = _afterok_job_ids(live_dependency)
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
                "SubmitLine%1024,Start,End"
            ),
        ],
        runner=runner,
        env_overrides={"SLURM_TIME_FORMAT": "standard"},
    )
    accounting_rows = [
        row for row in _rows(accounting_output, fields=8) if row[0] == str(job_id)
    ]
    if len(accounting_rows) == 1:
        accounting_comment = accounting_rows[0][3].strip()
        submit_line_comment = _submit_line_comment(accounting_rows[0][5])
        if accounting_comment and accounting_comment != token:
            raise ValueError("Slurm job comment does not match the submission token")
        if submit_line_comment and submit_line_comment != token:
            raise ValueError(
                "Slurm submit-line comment does not match the submission token"
            )
        effective_comment = accounting_comment or submit_line_comment
        state = _validate_identity(
            accounting_rows[0][:3] + [effective_comment, accounting_rows[0][4]],
            job_id=job_id,
            job_name=job_name,
            token=token,
            cluster=cluster,
        )
        accounting_dependency = _submit_line_dependency(accounting_rows[0][5])
        if accounting_dependency != expected_dependency:
            raise ValueError(
                "Slurm dependency mismatch: "
                f"expected {expected_dependency or '(none)'}, "
                f"got {accounting_dependency or '(none)'}"
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
        return {
            "ok": True,
            "source": "sacct",
            "job_id": job_id,
            "job_name": job_name,
            "state": state,
            "comment": token,
            "cluster": cluster,
            "dependency": expected_dependency or None,
        }
    if accounting_rows:
        raise ValueError("receipt job is ambiguous in Slurm accounting")
    if live_state is not None and live_dependency == expected_dependency:
        if expected_dependency_ids and live_state != "PENDING":
            raise ValueError(
                "dependent Slurm target has started without accounting proof"
            )
        return {
            "ok": True,
            "source": "squeue",
            "job_id": job_id,
            "job_name": job_name,
            "state": live_state,
            "comment": token,
            "cluster": cluster,
            "dependency": expected_dependency or None,
        }
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
    parser.add_argument("--dependency", default="")
    args = parser.parse_args(argv)
    payload = validate_slurm_receipt(
        job_id=args.job_id,
        job_name=args.job_name,
        token=args.comment,
        cluster=args.cluster,
        dependency=args.dependency,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
