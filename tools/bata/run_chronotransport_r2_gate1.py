#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.chronotransport.gate1_unlock import (
    build_gate1_unlock_artifact,
    build_gate1_unlock_artifact_for_test_only,
)
from opentad.models.chronotransport.protocol import canonical_json_bytes


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
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def _atomic_write(path: Path, data: bytes) -> None:
    """Publish complete bytes exactly once; never replace formal evidence."""

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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--registration-commit", required=True)
    parser.add_argument("--registration-relpath", required=True)
    args = parser.parse_args()
    payload = _load_json(args.input)
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
