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
MIB = 1024**2
DEFAULT_PERSISTENT_BYTES_PER_CELL = 4 * GIB
DEFAULT_ATOMIC_TEMP_BYTES_PER_CELL = 2 * GIB
DEFAULT_RESERVE_BYTES = 32 * GIB
GEOROUTE_STORAGE_PROFILE_SCHEMA = "georoute_storage_profile_v1"
GEOROUTE_NO_ARTIFACT_STORAGE_SCHEMA = (
    "georoute_no_artifact_storage_preflight_v1"
)
NO_ARTIFACT_BYTES_PER_LEAF = 512 * MIB
NO_ARTIFACT_FIXED_OVERHEAD_BYTES = 1 * GIB
NO_ARTIFACT_RESERVE_BYTES = 24 * GIB
NO_ARTIFACT_FORBIDDEN_OUTPUTS = (
    "checkpoint",
    "prediction",
    "metric",
    "evaluator",
    "official_test",
)


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


def no_artifact_storage_capacity_receipt(
    path: str | Path,
    *,
    leaf_count: int,
) -> dict[str, Any]:
    """Guard a preflight whose contract forbids every large model artifact.

    Unlike a training cell, an official-comparability preflight may not write a
    checkpoint, prediction, metric, evaluator output, or official-test output.
    Reusing the training-cell estimate would reserve atomic checkpoint copies
    that the protocol makes impossible.  This fixed, non-environment-tunable
    profile still holds 512 MiB per leaf, 1 GiB of shared overhead, and a
    24-GiB filesystem reserve.
    """

    if int(leaf_count) <= 0:
        raise ValueError("GeoRoute no-artifact leaf_count must be positive")
    target = Path(path).resolve()
    anchor = _existing_anchor(target)
    if hasattr(os, "statvfs"):
        stat = os.statvfs(anchor)
        free_bytes = int(stat.f_bavail) * int(stat.f_frsize)
    else:
        free_bytes = int(shutil.disk_usage(anchor).free)
    required_bytes = (
        int(leaf_count) * NO_ARTIFACT_BYTES_PER_LEAF
        + NO_ARTIFACT_FIXED_OVERHEAD_BYTES
        + NO_ARTIFACT_RESERVE_BYTES
    )
    receipt = {
        "schema_version": GEOROUTE_NO_ARTIFACT_STORAGE_SCHEMA,
        "target_path": str(target),
        "measured_anchor": str(anchor),
        "leaf_count": int(leaf_count),
        "artifact_policy": "NO_LARGE_ARTIFACTS_ALLOWED",
        "forbidden_outputs": list(NO_ARTIFACT_FORBIDDEN_OUTPUTS),
        "bytes_per_leaf": NO_ARTIFACT_BYTES_PER_LEAF,
        "fixed_overhead_bytes": NO_ARTIFACT_FIXED_OVERHEAD_BYTES,
        "reserve_bytes": NO_ARTIFACT_RESERVE_BYTES,
        "required_free_bytes": required_bytes,
        "observed_free_bytes": free_bytes,
        "headroom_after_required_bytes": free_bytes - required_bytes,
        "atomic_publish_peak_included": False,
        "status": (
            "PASS_STORAGE_PREFLIGHT"
            if free_bytes >= required_bytes
            else "FAIL_STORAGE_PREFLIGHT"
        ),
    }
    if free_bytes < required_bytes:
        raise RuntimeError(
            "GeoRoute no-artifact storage preflight failed: "
            f"free={free_bytes}, required={required_bytes}, "
            f"leaves={leaf_count}, anchor={anchor}"
        )
    return receipt
