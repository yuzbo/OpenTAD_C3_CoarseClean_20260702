#!/usr/bin/env python3
"""Generate r2 registration from a clean detached repository and real inputs."""

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

from opentad.models.chronotransport.protocol import canonical_json_bytes
from opentad.models.chronotransport.registration import (
    build_pre_gate1_registration_from_context,
)


def _load_template(path: Path) -> dict[str, object]:
    def reject(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)
    if not isinstance(value, dict):
        raise TypeError("registration identity template must be an object")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    """Publish canonical registration bytes exactly once."""

    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError("registration output path must not traverse a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError("registration output path must not traverse a symlink")
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-template", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--config-identity", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-registry-id", required=True)
    parser.add_argument("--checkpoint-uri", required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint-provider-receipt", type=Path, required=True)
    parser.add_argument("--content-store-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registration = build_pre_gate1_registration_from_context(
        _load_template(args.identity_template),
        repository_root=args.repository_root,
        manifest_path=args.manifest,
        registry_path=args.registry,
        config_identity_path=args.config_identity,
        checkpoint_source=args.checkpoint,
        checkpoint_registry_id=args.checkpoint_registry_id,
        checkpoint_authenticated_uri=args.checkpoint_uri,
        checkpoint_receipt_path=args.checkpoint_receipt,
        checkpoint_provider_receipt_path=args.checkpoint_provider_receipt,
        content_store_root=args.content_store_root,
        data_root=args.data_root,
    )
    _atomic_write(args.output, canonical_json_bytes(registration) + b"\n")
    print(registration["registration_sha256"])


if __name__ == "__main__":
    main()
