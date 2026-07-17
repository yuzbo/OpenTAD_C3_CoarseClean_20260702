from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Sequence


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _run(
    command: list[str],
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> str:
    result = runner(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if int(result.returncode) != 0:
        raise RuntimeError(
            f"Slurm query failed ({' '.join(command)}): "
            f"{str(result.stderr or '').strip()}"
        )
    return str(result.stdout or "")


def _json_after_cluster_prefix(output: str, *, cluster: str) -> dict[str, Any]:
    normalized = output.replace("\r", "")
    start = normalized.find("{")
    _require(start >= 0, "Slurm JSON payload is missing")
    prefixes = [
        line.strip()
        for line in normalized[:start].splitlines()
        if line.strip()
    ]
    _require(
        all(line == f"CLUSTER: {cluster}" for line in prefixes),
        "Slurm JSON has an unexpected cluster prefix",
    )
    payload = json.loads(normalized[start:])
    _require(isinstance(payload, dict), "Slurm JSON payload is not an object")
    _require(payload.get("errors") == [], "Slurm JSON reports an error")
    _require(isinstance(payload.get("jobs"), list), "Slurm JSON jobs are invalid")
    return payload


def recover_unique_held_job_id(
    *,
    token: str,
    job_name: str,
    cluster: str,
    job_file: str | os.PathLike[str],
    user: str,
    runner: Callable[..., Any] = subprocess.run,
) -> int:
    _require(
        re.fullmatch(r"[A-Za-z0-9._-]{1,256}", token) is not None,
        "invalid submission token",
    )
    _require(
        re.fullmatch(r"[A-Za-z0-9._-]{1,128}", job_name) is not None,
        "invalid job name",
    )
    _require(
        re.fullmatch(r"[A-Za-z0-9._-]+", cluster) is not None,
        "invalid cluster",
    )
    _require(bool(user.strip()), "Slurm user is missing")
    path = Path(job_file).expanduser().resolve(strict=True)
    output = _run(
        [
            "squeue",
            "--clusters",
            cluster,
            "--user",
            user,
            "--json",
        ],
        runner=runner,
    )
    payload = _json_after_cluster_prefix(output, cluster=cluster)
    matches = [
        job
        for job in payload["jobs"]
        if isinstance(job, dict)
        and job.get("comment") == token
        and job.get("name") == job_name
        and job.get("cluster") == cluster
        and job.get("command") == str(path)
        and job.get("job_state") == ["PENDING"]
        and job.get("hold") is True
        and job.get("state_reason") == "JobHeldUser"
    ]
    _require(
        len(matches) == 1,
        "submission token did not resolve to one currently held job",
    )
    job_id = matches[0].get("job_id")
    _require(
        isinstance(job_id, int)
        and not isinstance(job_id, bool)
        and job_id > 0,
        "recovered Slurm job id is invalid",
    )
    return job_id


def verify_cancelled_job_ids(
    *,
    job_ids: Sequence[int],
    cluster: str,
    runner: Callable[..., Any] = subprocess.run,
) -> None:
    expected = tuple(int(job_id) for job_id in job_ids)
    _require(expected and all(job_id > 0 for job_id in expected), "job ids are invalid")
    _require(len(set(expected)) == len(expected), "job ids contain duplicates")
    output = _run(
        [
            "sacct",
            "-X",
            "-M",
            cluster,
            "-j",
            ",".join(str(job_id) for job_id in expected),
            "-n",
            "-P",
            "-o",
            "JobIDRaw,Cluster,State",
        ],
        runner=runner,
    )
    rows: dict[int, tuple[str, str]] = {}
    for raw in output.replace("\r", "").splitlines():
        if not raw.strip():
            continue
        fields = raw.split("|")
        _require(len(fields) == 3, f"unexpected sacct row: {raw!r}")
        if re.fullmatch(r"[1-9][0-9]*", fields[0]) is None:
            continue
        job_id = int(fields[0])
        if job_id not in expected:
            continue
        _require(job_id not in rows, f"job {job_id} is ambiguous in sacct")
        rows[job_id] = (fields[1], fields[2])
    _require(set(rows) == set(expected), "cancelled-job accounting is incomplete")
    for job_id, (observed_cluster, raw_state) in rows.items():
        state = raw_state.strip().upper().split()[0].rstrip("+")
        _require(observed_cluster == cluster, f"job {job_id} cluster mismatch")
        _require(state == "CANCELLED", f"job {job_id} is not CANCELLED")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    recover = subparsers.add_parser("recover-held-job")
    recover.add_argument("--token", required=True)
    recover.add_argument("--job-name", required=True)
    recover.add_argument("--cluster", required=True)
    recover.add_argument("--job-file", required=True)
    recover.add_argument("--user", required=True)
    cancelled = subparsers.add_parser("verify-cancelled")
    cancelled.add_argument("--job-ids", required=True)
    cancelled.add_argument("--cluster", required=True)
    args = parser.parse_args(argv)
    if args.command == "recover-held-job":
        print(
            recover_unique_held_job_id(
                token=args.token,
                job_name=args.job_name,
                cluster=args.cluster,
                job_file=args.job_file,
                user=args.user,
            )
        )
    else:
        raw_ids = args.job_ids.split(",")
        _require(
            all(re.fullmatch(r"[1-9][0-9]*", item) for item in raw_ids),
            "job ids are invalid",
        )
        verify_cancelled_job_ids(
            job_ids=[int(item) for item in raw_ids],
            cluster=args.cluster,
        )
        print("cancelled=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
