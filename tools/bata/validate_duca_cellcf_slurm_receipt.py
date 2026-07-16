from __future__ import annotations

import argparse
import json
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


def _run(
    command: list[str],
    *,
    runner: Callable[..., Any] = subprocess.run,
    allow_missing_job: bool = False,
) -> str:
    result = runner(command, text=True, capture_output=True, check=False)
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
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    if job_id <= 0:
        raise ValueError("job_id must be positive")
    if re.fullmatch(r"[A-Za-z0-9._-]+", cluster) is None:
        raise ValueError("invalid Slurm cluster identity")

    live_output = _run(
        [
            "squeue",
            "--clusters",
            cluster,
            "--noheader",
            "--jobs",
            str(job_id),
            "--format",
            "%i|%128j|%T|%256k",
        ],
        runner=runner,
        allow_missing_job=True,
    )
    live_rows = _rows(live_output, fields=4)
    if live_rows:
        if len(live_rows) != 1:
            raise ValueError("receipt resolved to multiple live Slurm jobs")
        state = _validate_identity(
            live_rows[0] + [cluster],
            job_id=job_id,
            job_name=job_name,
            token=token,
            cluster=cluster,
        )
        return {
            "ok": True,
            "source": "squeue",
            "job_id": job_id,
            "job_name": job_name,
            "state": state,
            "comment": token,
            "cluster": cluster,
        }

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
            "JobIDRaw,JobName%128,State,Comment%256,Cluster",
        ],
        runner=runner,
    )
    accounting_rows = [
        row for row in _rows(accounting_output, fields=5) if row[0] == str(job_id)
    ]
    if len(accounting_rows) != 1:
        raise ValueError("receipt job is absent or ambiguous in Slurm accounting")
    state = _validate_identity(
        accounting_rows[0],
        job_id=job_id,
        job_name=job_name,
        token=token,
        cluster=cluster,
    )
    return {
        "ok": True,
        "source": "sacct",
        "job_id": job_id,
        "job_name": job_name,
        "state": state,
        "comment": token,
        "cluster": cluster,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--comment", required=True)
    parser.add_argument("--cluster", required=True)
    args = parser.parse_args(argv)
    payload = validate_slurm_receipt(
        job_id=args.job_id,
        job_name=args.job_name,
        token=args.comment,
        cluster=args.cluster,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
