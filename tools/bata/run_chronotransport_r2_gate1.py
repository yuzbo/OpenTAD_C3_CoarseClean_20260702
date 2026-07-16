#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.path.abspath(__file__)).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.chronotransport.gate1_unlock import (
    build_gate1_unlock_artifact,
    build_gate1_unlock_artifact_for_test_only,
)
from opentad.models.chronotransport.filesystem import (
    audit_formal_python_runtime,
    load_bound_json,
    publish_bytes_exclusive,
)
from opentad.models.chronotransport.protocol import canonical_json_bytes
from opentad.models.chronotransport.registration import validate_pre_gate1_registration


def run_gate1_payload(
    payload: dict[str, object],
    repository_root: str | Path,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, object]:
    return build_gate1_unlock_artifact(
        payload,
        repository_root=str(repository_root),
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )


def run_gate1_payload_for_test_only(
    payload: dict[str, object],
) -> dict[str, object]:
    return build_gate1_unlock_artifact_for_test_only(payload)


def _load_json(path: Path) -> object:
    _, value, _, _ = load_bound_json(path, label="Gate 1 adjudication input")
    return value


def _atomic_write(path: Path, data: bytes) -> None:
    """Publish complete bytes exactly once; never replace formal evidence."""

    publish_bytes_exclusive(path, data, label="Gate 1 adjudication output")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--registration-commit", required=True)
    parser.add_argument("--registration-relpath", required=True)
    args = parser.parse_args()
    payload = _load_json(args.input)
    if not isinstance(payload, dict) or not isinstance(payload.get("registration"), dict):
        raise ValueError("formal Gate 1 input must embed the registration")
    registered = validate_pre_gate1_registration(
        payload["registration"],
        repository_root=args.repository_root,
        context_mode="formal",
        registration_commit=args.registration_commit,
        registration_relpath=args.registration_relpath,
    )
    audit_formal_python_runtime(
        repository_root=args.repository_root,
        registered_sources=registered["source_files"],
        entrypoint_relative="tools/bata/run_chronotransport_r2_gate1.py",
    )
    result = run_gate1_payload(
        payload,
        args.repository_root,
        args.registration_commit,
        args.registration_relpath,
    )
    _atomic_write(args.output, canonical_json_bytes(result) + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
