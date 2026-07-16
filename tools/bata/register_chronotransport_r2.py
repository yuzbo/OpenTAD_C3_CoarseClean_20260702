#!/usr/bin/env python3
"""Generate r2 registration from a clean detached repository and real inputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(os.path.abspath(__file__)).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.chronotransport.protocol import canonical_json_bytes
from opentad.models.chronotransport.filesystem import (
    audit_formal_python_runtime,
    load_bound_json,
    publish_bytes_exclusive,
)
from opentad.models.chronotransport.registration import (
    build_pre_gate1_registration_from_context,
)


def _load_template(path: Path) -> dict[str, object]:
    _, value, _, _ = load_bound_json(path, label="registration identity template")
    if not isinstance(value, dict):
        raise TypeError("registration identity template must be an object")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    """Publish canonical registration bytes exactly once."""

    publish_bytes_exclusive(path, payload, label="canonical registration output")


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
    audit_formal_python_runtime(
        repository_root=args.repository_root,
        registered_sources=registration["source_files"],
        entrypoint_relative="tools/bata/register_chronotransport_r2.py",
    )
    _atomic_write(args.output, canonical_json_bytes(registration) + b"\n")
    print(registration["registration_sha256"])


if __name__ == "__main__":
    main()
