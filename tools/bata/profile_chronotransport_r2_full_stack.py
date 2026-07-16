#!/usr/bin/env python3
"""Run the immutable registration-driven r2 Gate-1 full-stack profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping

ROOT = Path(os.path.abspath(__file__)).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.chronotransport.full_stack_profiler import (
    build_full_stack_profile_artifact,
)
from opentad.models.chronotransport.filesystem import (
    load_bound_json,
    publish_bytes_exclusive,
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
    _, value, _, _ = load_bound_json(path, label="full-stack profile request")
    return value


def _atomic_write(path: Path, data: bytes) -> None:
    """Publish complete bytes exactly once without replacing an existing artifact."""

    publish_bytes_exclusive(path, data, label="full-stack profile output")


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
