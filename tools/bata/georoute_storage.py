"""Fail-closed aggregate storage accounting for GeoRoute experiment cells."""

from __future__ import annotations

import os
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping


GEOROUTE_STORAGE_SCHEMA = "georoute_storage_preflight_v1"
GIB = 1024**3
DEFAULT_PERSISTENT_BYTES_PER_CELL = 4 * GIB
DEFAULT_ATOMIC_TEMP_BYTES_PER_CELL = 2 * GIB
DEFAULT_RESERVE_BYTES = 32 * GIB
GEOROUTE_STORAGE_PROFILE_SCHEMA = "georoute_storage_profile_v1"


def _positive_env_bytes(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(int(default))))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _existing_anchor(path: Path) -> Path:
    anchor = path.resolve()
    while not anchor.exists():
        if anchor.parent == anchor:
            raise FileNotFoundError(path)
        anchor = anchor.parent
    return anchor


def storage_capacity_receipt(
    path: str | Path,
    *,
    cell_count: int,
    storage_profile: Mapping[str, Any] | None = None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    """Measure free bytes and reject an unsafe concurrent artifact peak."""

    if int(cell_count) <= 0:
        raise ValueError("GeoRoute storage preflight cell_count must be positive")
    target = Path(path).resolve()
    anchor = _existing_anchor(target)
    profile_sha256 = None
    if storage_profile is None:
        persistent = _positive_env_bytes(
            "GEOROUTE_STORAGE_PERSISTENT_BYTES_PER_CELL",
            DEFAULT_PERSISTENT_BYTES_PER_CELL,
        )
        atomic_temp = _positive_env_bytes(
            "GEOROUTE_STORAGE_ATOMIC_TEMP_BYTES_PER_CELL",
            DEFAULT_ATOMIC_TEMP_BYTES_PER_CELL,
        )
        reserve = _positive_env_bytes(
            "GEOROUTE_STORAGE_RESERVE_BYTES",
            DEFAULT_RESERVE_BYTES,
        )
        stage_fixed_overhead = 0
        profile_source = "conservative_pre_p0_defaults"
    else:
        profile = dict(storage_profile)
        if profile.get("schema_version") != GEOROUTE_STORAGE_PROFILE_SCHEMA:
            raise ValueError("unexpected GeoRoute storage profile schema")
        if expected_commit is not None and profile.get("runtime_commit") != str(
            expected_commit
        ).lower():
            raise ValueError("GeoRoute storage profile commit mismatch")
        checkpoint = int(profile.get("checkpoint_upper_bound_bytes", 0))
        copies = int(profile.get("peak_checkpoint_copies_per_cell", 0))
        auxiliary = int(
            profile.get("auxiliary_upper_bound_bytes_per_cell", 0)
        )
        stage_fixed_overhead = int(
            profile.get("stage_fixed_overhead_bytes", 0)
        )
        safety_fraction = float(profile.get("safety_fraction", 0.0))
        safety_bytes = int(profile.get("safety_bytes", 0))
        if (
            checkpoint <= 0
            or copies <= 0
            or auxiliary <= 0
            or stage_fixed_overhead < 0
            or safety_fraction < 0.0
            or safety_bytes <= 0
        ):
            raise ValueError("GeoRoute storage profile contains non-positive bounds")
        persistent = checkpoint + auxiliary
        atomic_temp = checkpoint * copies
        unsafe_peak = (
            int(cell_count) * (persistent + atomic_temp)
            + stage_fixed_overhead
        )
        reserve = max(safety_bytes, int(unsafe_peak * safety_fraction))
        profile_source = "same_commit_p0_measurement_bound"
        profile_sha256 = hashlib.sha256(
            json.dumps(
                profile,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
    if hasattr(os, "statvfs"):
        stat = os.statvfs(anchor)
        free_bytes = int(stat.f_bavail) * int(stat.f_frsize)
    else:
        free_bytes = int(shutil.disk_usage(anchor).free)
    required_bytes = int(cell_count) * (
        persistent + atomic_temp
    ) + stage_fixed_overhead + reserve
    receipt = {
        "schema_version": GEOROUTE_STORAGE_SCHEMA,
        "target_path": str(target),
        "measured_anchor": str(anchor),
        "cell_count": int(cell_count),
        "persistent_bytes_per_cell": persistent,
        "atomic_temp_bytes_per_cell": atomic_temp,
        "reserve_bytes": reserve,
        "stage_fixed_overhead_bytes": stage_fixed_overhead,
        "storage_profile_source": profile_source,
        "storage_profile_sha256": profile_sha256,
        "required_free_bytes": required_bytes,
        "observed_free_bytes": free_bytes,
        "headroom_after_required_bytes": free_bytes - required_bytes,
        "atomic_publish_peak_included": True,
        "status": "PASS_STORAGE_PREFLIGHT"
        if free_bytes >= required_bytes
        else "FAIL_STORAGE_PREFLIGHT",
    }
    if free_bytes < required_bytes:
        raise RuntimeError(
            "GeoRoute aggregate storage preflight failed: "
            f"free={free_bytes}, required={required_bytes}, "
            f"cells={cell_count}, anchor={anchor}"
        )
    return receipt
