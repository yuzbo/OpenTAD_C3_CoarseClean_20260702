from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any, Mapping


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def with_content_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    output.pop("content_sha256", None)
    output["content_sha256"] = canonical_sha256(output)
    return output


def verify_content_sha256(payload: Mapping[str, Any]) -> None:
    unsigned = dict(payload)
    embedded = unsigned.pop("content_sha256", None)
    if embedded != canonical_sha256(unsigned):
        raise ValueError("evidence content_sha256 is invalid")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        # Directory handles are not generally openable on Windows.  The file
        # and atomic hard-link publication remain fully enforced there.
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_exclusive_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Publish finite JSON atomically without ever replacing an existing file."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = target.parent / (
        f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    descriptor = os.open(
        str(temporary),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(str(temporary), str(target))
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target
