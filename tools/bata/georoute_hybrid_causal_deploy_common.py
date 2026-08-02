"""Shared fail-closed Slurm helpers for the Hybrid causal study."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


RESOURCE_ARGS = {
    # N16R4 assigns memory per allocated GPU and rejects explicit --mem
    # overrides at submission time.  Keep this table limited to resources the
    # caller is allowed to request on that site.
    "model1": ("--gpus", "1", "--cpus-per-task", "5"),
    "world2": ("--gpus", "2", "--cpus-per-task", "10"),
    "stage2": ("--gpus", "2", "--cpus-per-task", "10"),
    "control": ("--gpus", "1", "--cpus-per-task", "1"),
    "cpu_control": ("--cpus-per-task", "1"),
}


def git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or f"git {' '.join(arguments)} failed"
        )
    return completed.stdout.strip()


def clean_export(value: str, *, name: str) -> str:
    if (
        not value
        or value != value.strip()
        or any(character in value for character in (",", "\n", "\r", "\x00"))
    ):
        raise ValueError(f"{name} is unsafe for sbatch --export")
    return value


def full_hex(value: str, *, length: int, name: str) -> str:
    normalized = str(value).lower()
    if (
        len(normalized) != length
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"{name} must be a full lowercase hexadecimal digest")
    return normalized


def sbatch(
    *,
    root: Path,
    name: str,
    script: Path,
    logs: Path,
    exports: Mapping[str, str],
    resource: str,
    dependency: Sequence[str] | None = None,
    dependency_type: str = "afterok",
    test_only: bool = False,
    hold: bool = False,
    kill_invalid_dependency: bool = False,
) -> str:
    if resource not in RESOURCE_ARGS:
        raise ValueError(f"unsupported Hybrid causal resource mode {resource!r}")
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        name,
        "--output",
        str(logs / f"{name}.%j.out"),
        "--error",
        str(logs / f"{name}.%j.err"),
    ]
    if test_only:
        command.append("--test-only")
    if hold and not test_only:
        command.append("--hold")
    if kill_invalid_dependency:
        command.append("--kill-on-invalid-dep=yes")
    if dependency:
        if dependency_type not in {"afterok", "afterany"}:
            raise ValueError("unsupported Hybrid causal dependency type")
        command.extend(
            [
                "--dependency",
                f"{dependency_type}:" + ":".join(map(str, dependency)),
            ]
        )
    command.extend(RESOURCE_ARGS[resource])
    command.extend(
        [
            "--export",
            ",".join(
                [
                    "ALL",
                    *(
                        f"{key}={clean_export(str(value), name=key)}"
                        for key, value in sorted(exports.items())
                    ),
                ]
            ),
            str(script),
        ]
    )
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"sbatch failed for {name}: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    if test_only:
        return "TEST_ONLY_PASS"
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"invalid sbatch job id for {name}: {completed.stdout!r}")
    return job_id


def cancel_jobs(root: Path, job_ids: Sequence[str]) -> None:
    if job_ids:
        subprocess.run(
            ["scancel", *map(str, job_ids)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )


def release_jobs(root: Path, job_ids: Sequence[str]) -> None:
    completed = subprocess.run(
        ["scontrol", "release", *map(str, job_ids)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to release held Hybrid causal jobs: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )


def require_submit_capacity(*, root: Path, additional_jobs: int) -> dict[str, int]:
    user = os.environ.get("USER")
    if not user:
        raise RuntimeError("Hybrid causal deployer cannot determine Slurm user")
    active = subprocess.run(
        ["squeue", "-h", "-u", user],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if active.returncode != 0:
        raise RuntimeError(f"cannot query squeue: {active.stderr.strip()}")
    active_count = len([line for line in active.stdout.splitlines() if line.strip()])
    association = subprocess.run(
        [
            "sacctmgr",
            "-n",
            "-P",
            "show",
            "assoc",
            "where",
            f"user={user}",
            "format=MaxSubmitJobs",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if association.returncode != 0:
        raise RuntimeError(
            "cannot query MaxSubmitJobs: " + association.stderr.strip()
        )
    limits = [
        int(line.split("|", 1)[0])
        for line in association.stdout.splitlines()
        if line.split("|", 1)[0].strip().isdigit()
    ]
    if not limits:
        raise RuntimeError("Hybrid causal deployer cannot determine MaxSubmitJobs")
    limit = min(limits)
    if active_count + int(additional_jobs) > limit:
        raise RuntimeError(
            "Hybrid causal deployer refuses partial submission: "
            f"active={active_count}, required={additional_jobs}, limit={limit}"
        )
    return {
        "active_jobs": active_count,
        "additional_jobs": int(additional_jobs),
        "max_submit_jobs": limit,
        "headroom_after_submission": limit - active_count - int(additional_jobs),
    }
