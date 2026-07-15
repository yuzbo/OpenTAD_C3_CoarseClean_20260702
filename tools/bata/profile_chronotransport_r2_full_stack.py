#!/usr/bin/env python3
"""Run the immutable registration-driven r2 Gate-1 full-stack profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Mapping

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.chronotransport.full_stack_profiler import (
    build_full_stack_profile_artifact,
)
from opentad.models.chronotransport.protocol import canonical_json_bytes, canonical_sha256
from opentad.models.chronotransport.registration import (
    validate_formal_random_control_lock,
    validate_pre_gate1_registration,
)
PROFILE_REQUEST_SCHEMA = "chronotransport-r2-full-stack-profile-request-v2"


def validate_profile_request(payload: Mapping[str, object]) -> dict[str, object]:
    """Accept only the full registration artifact; all execution inputs derive from it."""

    if not isinstance(payload, Mapping) or set(payload) != {"schema", "registration"}:
        raise ValueError("formal profile request fields mismatch")
    if payload["schema"] != PROFILE_REQUEST_SCHEMA:
        raise ValueError("unsupported full-stack profile request schema")
    return validate_pre_gate1_registration(payload["registration"])


def _query_gpu_identity() -> tuple[str, str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or "," not in rows[0]:
        raise RuntimeError("formal profiler requires one observable GPU UUID/driver row")
    uuid, driver = (item.strip() for item in rows[0].split(",", 1))
    if not uuid or not driver:
        raise RuntimeError("formal profiler GPU UUID/driver cannot be empty")
    return uuid, driver


def _observed_environment() -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal full-stack profiling requires exactly one visible CUDA device")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("formal full-stack profiling requires physical GPU1 visibility")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("formal full-stack profiling requires a Slurm allocation and step")
    gpu_uuid, driver = _query_gpu_identity()
    environment: dict[str, object] = {
        "gpu_model": torch.cuda.get_device_name(0),
        "gpu_uuid": gpu_uuid,
        "driver": driver,
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
        "cudnn": str(torch.backends.cudnn.version()),
        "precision": "amp_fp16",
        "batch_size": 1,
    }
    if any(not isinstance(environment[field], str) or not environment[field] for field in environment if field != "batch_size"):
        raise RuntimeError("formal profiler observed an incomplete environment identity")
    environment["environment_sha256"] = canonical_sha256(environment)
    return environment


def profile_request(
    payload: Mapping[str, object],
    repository_root: str | Path,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, object]:
    registration = validate_pre_gate1_registration(
        validate_profile_request(payload),
        repository_root=repository_root,
        context_mode="formal",
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    validate_formal_random_control_lock(registration)
    observed = _observed_environment()
    if observed != registration["environment"]:
        raise RuntimeError("observed profile environment differs from pre-Gate1 registration")
    for plan in registration["profiler"]["candidate_plan"]:
        factory_config = dict(plan["factory_config"])
        if canonical_sha256(factory_config) != plan["factory_config_sha256"]:
            raise RuntimeError("registered profile factory config changed after validation")
    return build_full_stack_profile_artifact(
        registration=registration,
        repository_root=str(repository_root),
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )


def _load_json(path: Path) -> object:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def _atomic_write(path: Path, data: bytes) -> None:
    """Publish complete bytes exactly once without replacing an existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--registration-commit", required=True)
    parser.add_argument("--registration-relpath", required=True)
    args = parser.parse_args()
    request = _load_json(args.request)
    artifact = profile_request(
        request,
        args.repository_root,
        args.registration_commit,
        args.registration_relpath,
    )
    _atomic_write(args.output, canonical_json_bytes(artifact) + b"\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
