from __future__ import annotations

import re


TRAINING_PROFILES = ("exposure132", "official60")
JOB_ORDER = (
    "uniform",
    "transition_beta0",
    "cellcf",
    "aggregate",
    "cost",
    "completion",
)
JOB_SUFFIXES = {
    "uniform": "uniform",
    "transition_beta0": "transition",
    "cellcf": "cellcf",
    "aggregate": "aggregate",
    "cost": "cost",
    "completion": "completion",
}


def expected_job_name(
    training_profile: str,
    job_key: str,
    seed: int,
    git_commit: str,
) -> str:
    if training_profile not in TRAINING_PROFILES:
        raise ValueError(f"unsupported CellCF training profile: {training_profile}")
    if job_key not in JOB_SUFFIXES:
        raise ValueError(f"unsupported CellCF job key: {job_key}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("CellCF seed must be a non-negative integer")
    if re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
        raise ValueError("CellCF Git commit must be a full lowercase SHA1")
    return (
        f"cellcf-{training_profile}-{JOB_SUFFIXES[job_key]}"
        f"-s{seed}-{git_commit[:7]}"
    )


__all__ = [
    "JOB_ORDER",
    "JOB_SUFFIXES",
    "TRAINING_PROFILES",
    "expected_job_name",
]
