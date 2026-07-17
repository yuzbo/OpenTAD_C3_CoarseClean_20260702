from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import time
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


@dataclass(frozen=True)
class HeldJobSubmission:
    job_id: int
    job_ref: str
    raw_sbatch_response: str
    cluster: str
    dependency: str


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


def _recover_unique_job_id(
    *,
    token: str,
    job_name: str,
    cluster: str,
    job_file: str | os.PathLike[str],
    user: str,
    require_current_user_hold: bool,
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
    identity_matches = [
        job
        for job in payload["jobs"]
        if isinstance(job, dict)
        and job.get("comment") == token
        and job.get("name") == job_name
        and job.get("cluster") == cluster
        and job.get("command") == str(path)
    ]
    matches = identity_matches
    if require_current_user_hold:
        matches = [
            job
            for job in identity_matches
            if job.get("job_state") == ["PENDING"]
            and job.get("hold") is True
            and job.get("state_reason") == "JobHeldUser"
        ]
    _require(
        len(matches) == 1,
        (
            "submission token did not resolve to one currently held job"
            if require_current_user_hold
            else "submission token did not resolve to one unique job"
        ),
    )
    job_id = matches[0].get("job_id")
    _require(
        isinstance(job_id, int)
        and not isinstance(job_id, bool)
        and job_id > 0,
        "recovered Slurm job id is invalid",
    )
    return job_id


def recover_unique_held_job_id(
    *,
    token: str,
    job_name: str,
    cluster: str,
    job_file: str | os.PathLike[str],
    user: str,
    runner: Callable[..., Any] = subprocess.run,
) -> int:
    return _recover_unique_job_id(
        token=token,
        job_name=job_name,
        cluster=cluster,
        job_file=job_file,
        user=user,
        require_current_user_hold=True,
        runner=runner,
    )


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


def cancel_and_verify_job_ids(
    *,
    job_ids: Sequence[int],
    cluster: str,
    attempts: int = 10,
    delay_seconds: float = 1.0,
    runner: Callable[..., Any] = subprocess.run,
    sleeper: Callable[[float], Any] = time.sleep,
) -> None:
    ids = tuple(int(job_id) for job_id in job_ids)
    _require(ids and all(job_id > 0 for job_id in ids), "job ids are invalid")
    _require(len(set(ids)) == len(ids), "job ids contain duplicates")
    _require(attempts > 0, "cancellation verification attempts must be positive")
    result = runner(
        ["scancel", "-M", cluster, *[str(job_id) for job_id in ids]],
        text=True,
        capture_output=True,
        check=False,
    )
    if int(result.returncode) != 0:
        raise RuntimeError(
            f"Slurm cancellation failed: {str(result.stderr or '').strip()}"
        )
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            verify_cancelled_job_ids(
                job_ids=ids,
                cluster=cluster,
                runner=runner,
            )
            return
        except (RuntimeError, ValueError) as exc:
            error = exc
            if attempt + 1 < attempts:
                sleeper(delay_seconds)
    raise RuntimeError(
        "cancelled-job accounting did not converge"
    ) from error


def _recover_with_retry(
    *,
    token: str,
    job_name: str,
    cluster: str,
    job_file: Path,
    user: str,
    require_current_user_hold: bool,
    attempts: int,
    delay_seconds: float,
    runner: Callable[..., Any],
    sleeper: Callable[[float], Any],
) -> int:
    _require(
        attempts >= 0,
        "submission recovery attempts must be non-negative",
    )
    error: Exception | None = None
    attempt = 0
    while attempts == 0 or attempt < attempts:
        try:
            return _recover_unique_job_id(
                token=token,
                job_name=job_name,
                cluster=cluster,
                job_file=job_file,
                user=user,
                require_current_user_hold=require_current_user_hold,
                runner=runner,
            )
        except (RuntimeError, ValueError) as exc:
            error = exc
            attempt += 1
            if attempts == 0 or attempt < attempts:
                sleeper(delay_seconds)
    raise RuntimeError(
        "submitted Slurm job could not be recovered by its unique token"
    ) from error


def submit_held_job(
    *,
    token: str,
    job_name: str,
    cluster: str,
    job_file: str | os.PathLike[str],
    user: str,
    dependency: str = "",
    recovery_attempts: int = 10,
    recovery_delay_seconds: float = 1.0,
    runner: Callable[..., Any] = subprocess.run,
    sleeper: Callable[[float], Any] = time.sleep,
) -> HeldJobSubmission:
    path = Path(job_file).expanduser().resolve(strict=True)
    command = [
        "sbatch",
        "--parsable",
        "--hold",
        f"--clusters={cluster}",
        f"--job-name={job_name}",
        f"--comment={token}",
    ]
    if dependency:
        command.append(f"--dependency={dependency}")
    command.append(str(path))
    result = runner(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    raw = str(result.stdout or "").replace("\r", "").rstrip("\n")
    visibility_attempts = (
        0 if int(result.returncode) == 0 else recovery_attempts
    )
    recovered_id = _recover_with_retry(
        token=token,
        job_name=job_name,
        cluster=cluster,
        job_file=path,
        user=user,
        require_current_user_hold=False,
        attempts=visibility_attempts,
        delay_seconds=recovery_delay_seconds,
        runner=runner,
        sleeper=sleeper,
    )
    try:
        held_id = _recover_unique_job_id(
            token=token,
            job_name=job_name,
            cluster=cluster,
            job_file=path,
            user=user,
            require_current_user_hold=True,
            runner=runner,
        )
        _require(
            held_id == recovered_id,
            "held-job identity changed during submission reconciliation",
        )
        canonical = f"{recovered_id};{cluster}"
        _require(
            int(result.returncode) == 0,
            "sbatch returned a non-zero status after creating a held job",
        )
        _require(
            raw == canonical,
            "sbatch returned a non-canonical binding after creating a held job",
        )
    except (RuntimeError, ValueError):
        cancel_and_verify_job_ids(
            job_ids=[recovered_id],
            cluster=cluster,
            runner=runner,
            sleeper=sleeper,
        )
        raise
    return HeldJobSubmission(
        job_id=recovered_id,
        job_ref=canonical,
        raw_sbatch_response=raw,
        cluster=cluster,
        dependency=dependency or "none",
    )


def submit_held_job_pair(
    *,
    cost_token: str,
    cost_job_name: str,
    cost_job_file: str | os.PathLike[str],
    completion_token: str,
    completion_job_name: str,
    completion_job_file: str | os.PathLike[str],
    cluster: str,
    user: str,
    recovery_attempts: int = 10,
    recovery_delay_seconds: float = 1.0,
    runner: Callable[..., Any] = subprocess.run,
    sleeper: Callable[[float], Any] = time.sleep,
) -> dict[str, HeldJobSubmission]:
    submitted: list[int] = []
    try:
        cost = submit_held_job(
            token=cost_token,
            job_name=cost_job_name,
            cluster=cluster,
            job_file=cost_job_file,
            user=user,
            recovery_attempts=recovery_attempts,
            recovery_delay_seconds=recovery_delay_seconds,
            runner=runner,
            sleeper=sleeper,
        )
        submitted.append(cost.job_id)
        completion = submit_held_job(
            token=completion_token,
            job_name=completion_job_name,
            cluster=cluster,
            job_file=completion_job_file,
            user=user,
            dependency=f"afterok:{cost.job_id}",
            recovery_attempts=recovery_attempts,
            recovery_delay_seconds=recovery_delay_seconds,
            runner=runner,
            sleeper=sleeper,
        )
        submitted.append(completion.job_id)
    except (RuntimeError, ValueError):
        if submitted:
            cancel_and_verify_job_ids(
                job_ids=submitted,
                cluster=cluster,
                runner=runner,
                sleeper=sleeper,
            )
        raise
    return {"cost": cost, "completion": completion}


def fsync_artifacts(
    *,
    files: Sequence[str | os.PathLike[str]],
    directories: Sequence[str | os.PathLike[str]],
) -> None:
    _require(bool(files), "no recovery artifacts were provided for fsync")
    resolved_directories = {
        Path(directory).expanduser().resolve(strict=True)
        for directory in directories
    }
    for value in files:
        path = Path(value).expanduser().resolve(strict=True)
        _require(path.is_file(), f"recovery artifact is not a file: {path}")
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        resolved_directories.add(path.parent)
    for directory in sorted(
        resolved_directories,
        key=lambda item: (len(item.parts), str(item)),
        reverse=True,
    ):
        _require(
            directory.is_dir(),
            f"recovery fsync target is not a directory: {directory}",
        )
        descriptor = os.open(
            directory,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


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
    cancel = subparsers.add_parser("cancel-and-verify")
    cancel.add_argument("--job-ids", required=True)
    cancel.add_argument("--cluster", required=True)
    pair = subparsers.add_parser("submit-held-pair")
    pair.add_argument("--cost-token", required=True)
    pair.add_argument("--cost-job-name", required=True)
    pair.add_argument("--cost-job-file", required=True)
    pair.add_argument("--completion-token", required=True)
    pair.add_argument("--completion-job-name", required=True)
    pair.add_argument("--completion-job-file", required=True)
    pair.add_argument("--cluster", required=True)
    pair.add_argument("--user", required=True)
    sync = subparsers.add_parser("fsync-artifacts")
    sync.add_argument("--file", action="append", required=True)
    sync.add_argument("--directory", action="append", default=[])
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
    elif args.command in {"verify-cancelled", "cancel-and-verify"}:
        raw_ids = args.job_ids.split(",")
        _require(
            all(re.fullmatch(r"[1-9][0-9]*", item) for item in raw_ids),
            "job ids are invalid",
        )
        callback = (
            verify_cancelled_job_ids
            if args.command == "verify-cancelled"
            else cancel_and_verify_job_ids
        )
        callback(job_ids=[int(item) for item in raw_ids], cluster=args.cluster)
        print("cancelled=true")
    elif args.command == "submit-held-pair":
        result = submit_held_job_pair(
            cost_token=args.cost_token,
            cost_job_name=args.cost_job_name,
            cost_job_file=args.cost_job_file,
            completion_token=args.completion_token,
            completion_job_name=args.completion_job_name,
            completion_job_file=args.completion_job_file,
            cluster=args.cluster,
            user=args.user,
        )
        print(
            json.dumps(
                {key: asdict(value) for key, value in result.items()},
                sort_keys=True,
            )
        )
    else:
        fsync_artifacts(
            files=args.file,
            directories=args.directory,
        )
        print("fsync_complete=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
