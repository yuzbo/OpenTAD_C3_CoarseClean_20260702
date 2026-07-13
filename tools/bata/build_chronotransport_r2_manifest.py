#!/usr/bin/env python3
"""Build the immutable label-free ChronoTransport r2 window manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.chronotransport.protocol import (
    build_r2_manifest,
    manifest_exact_bytes,
    validate_r2_manifest,
)


def _load_json(path: Path) -> Any:
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicate_keys)


def load_manifest_file(
    manifest_path: str | Path,
    *,
    registry_path: str | Path,
    config_identity_path: str | Path,
) -> dict[str, Any]:
    """Load a formal manifest only when its raw bytes are exactly canonical."""

    manifest_path = Path(manifest_path)
    raw = manifest_path.read_bytes()
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise TypeError("manifest file root must be a JSON object")
    if raw != manifest_exact_bytes(manifest):
        raise ValueError("manifest file is not encoded as exact canonical bytes")
    exact_digest = hashlib.sha256(raw).hexdigest()
    sidecar_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    try:
        sidecar_bytes = sidecar_path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError("manifest SHA-256 sidecar is missing") from exc
    if sidecar_bytes != (exact_digest + "\n").encode("ascii"):
        raise ValueError("manifest SHA-256 sidecar does not match exact bytes")
    registry = _load_json(Path(registry_path))
    config_identity = _load_json(Path(config_identity_path))
    return validate_r2_manifest(
        manifest,
        registry=registry,
        config_identity=config_identity,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_manifest_file(
    registry_path: str | Path,
    config_identity_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    registry_path = Path(registry_path)
    config_identity_path = Path(config_identity_path)
    output_path = Path(output_path)
    registry = _load_json(registry_path)
    config_identity = _load_json(config_identity_path)
    manifest = build_r2_manifest(registry, config_identity)
    validate_r2_manifest(manifest, registry=registry, config_identity=config_identity)
    exact_bytes = manifest_exact_bytes(manifest)
    exact_bytes_sha256 = hashlib.sha256(exact_bytes).hexdigest()
    sidecar_path = output_path.with_suffix(output_path.suffix + ".sha256")
    _atomic_write(output_path, exact_bytes)
    _atomic_write(sidecar_path, (exact_bytes_sha256 + "\n").encode("ascii"))
    persisted = _load_json(output_path)
    validate_r2_manifest(persisted, registry=registry, config_identity=config_identity)
    if output_path.read_bytes() != exact_bytes:
        raise RuntimeError("persisted manifest bytes differ from canonical exact bytes")
    return {
        "schema": "chronotransport-r2-manifest-build-report-v1",
        "protocol": manifest["protocol"],
        "manifest_path": str(output_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "exact_bytes_sha256": exact_bytes_sha256,
        "sha256_sidecar": str(sidecar_path),
        "population_size": manifest["population_size"],
        "split_hashes": manifest["split_hashes"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--config-identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_manifest_file(args.registry, args.config_identity, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
