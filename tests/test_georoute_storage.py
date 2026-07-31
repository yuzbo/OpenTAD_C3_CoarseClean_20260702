from __future__ import annotations

from pathlib import Path

import pytest

from tools.bata.georoute_storage import (
    GIB,
    GEOROUTE_NO_ARTIFACT_STORAGE_SCHEMA,
    no_artifact_storage_capacity_receipt,
    storage_capacity_receipt,
)


def _profile(commit: str) -> dict:
    return {
        "schema_version": "georoute_storage_profile_v1",
        "runtime_commit": commit,
        "checkpoint_policy": "final_only",
        "checkpoint_upper_bound_bytes": 4096,
        "peak_checkpoint_copies_per_cell": 1,
        "auxiliary_upper_bound_bytes_per_cell": 2048,
        "stage_fixed_overhead_bytes": 1024,
        "safety_fraction": 0.25,
        "safety_bytes": 1024,
        "measurement_provenance": {"unit": True},
    }


def test_storage_preflight_binds_same_commit_profile_and_atomic_peak(tmp_path: Path):
    commit = "a" * 40
    receipt = storage_capacity_receipt(
        tmp_path,
        cell_count=7,
        storage_profile=_profile(commit),
        expected_commit=commit,
    )
    assert receipt["status"] == "PASS_STORAGE_PREFLIGHT"
    assert receipt["cell_count"] == 7
    assert receipt["atomic_publish_peak_included"] is True
    assert receipt["storage_profile_source"] == "same_commit_p0_measurement_bound"
    with pytest.raises(ValueError, match="commit mismatch"):
        storage_capacity_receipt(
            tmp_path,
            cell_count=7,
            storage_profile=_profile(commit),
            expected_commit="b" * 40,
        )


def test_no_artifact_preflight_does_not_reserve_impossible_checkpoints(
    tmp_path: Path,
):
    receipt = no_artifact_storage_capacity_receipt(tmp_path, leaf_count=2)
    assert receipt["schema_version"] == GEOROUTE_NO_ARTIFACT_STORAGE_SCHEMA
    assert receipt["status"] == "PASS_STORAGE_PREFLIGHT"
    assert receipt["artifact_policy"] == "NO_LARGE_ARTIFACTS_ALLOWED"
    assert receipt["required_free_bytes"] == 26 * GIB
    assert receipt["atomic_publish_peak_included"] is False
    assert set(receipt["forbidden_outputs"]) == {
        "checkpoint",
        "prediction",
        "metric",
        "evaluator",
        "official_test",
    }
    with pytest.raises(ValueError, match="leaf_count"):
        no_artifact_storage_capacity_receipt(tmp_path, leaf_count=0)
