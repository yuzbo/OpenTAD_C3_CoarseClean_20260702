from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata.aggregate_duca_protected_physical_official60 import (
    aggregate_official60,
)
from tools.bata.duca_protected_physical_training import canonical_sha256


COMMIT = "a" * 40
P0_SHA256 = "b" * 64
AUTH_SHA256 = "c" * 64
VARIANTS = (
    "exact_uniform",
    "transition_no_bridge",
    "protected_e2e",
    "protected_e2e_rho001",
)


def _write_evidence(
    root: Path,
    variant: str,
    average_map: float,
    *,
    commit: str = COMMIT,
    updates: int = 6000,
) -> Path:
    payload = {
        "schema": "duca_protected_physical_post_run_evidence_v1",
        "ok": True,
        "paper_claim_allowed": False,
        "task": "offline_temporal_action_detection",
        "variant": variant,
        "git_commit": commit,
        "seed": 3407,
        "protocol_manifest_sha256": P0_SHA256,
        "authorization_sha256": AUTH_SHA256,
        "successful_optimizer_updates": updates,
        "checkpoint_epoch": 59,
        "checkpoint_state_key": "state_dict_ema",
        "checkpoint_path": str(root / variant / "epoch_59.pth"),
        "checkpoint_sha256": canonical_sha256([variant, "checkpoint"]),
        "prediction_path": str(root / variant / "predictions.pkl"),
        "prediction_sha256": canonical_sha256([variant, "prediction"]),
        "metric_recomputation": {"performed": True},
        "metrics": {
            "average_mAP": average_map,
            "mAP@0.3": average_map + 10.0,
            "mAP@0.4": average_map + 5.0,
            "mAP@0.5": average_map,
            "mAP@0.6": average_map - 5.0,
            "mAP@0.7": average_map - 10.0,
        },
        "non_finite_collapse": False,
        "artifact_chain_sha256": canonical_sha256([variant, "artifact"]),
    }
    path = root / f"{variant}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _suite_paths(root: Path) -> list[Path]:
    maps = {
        "exact_uniform": 65.0,
        "transition_no_bridge": 65.2,
        "protected_e2e": 65.6,
        "protected_e2e_rho001": 65.5,
    }
    return [
        _write_evidence(root, variant, maps[variant])
        for variant in VARIANTS
    ]


def test_official60_aggregate_reports_preregistered_comparisons(
    tmp_path: Path,
) -> None:
    result = aggregate_official60(
        expected_commit=COMMIT,
        protocol_manifest_sha256=P0_SHA256,
        authorization_sha256=AUTH_SHA256,
        evidence_paths=_suite_paths(tmp_path),
    )

    assert result["ok"] is True
    assert result["paper_claim_allowed"] is False
    assert result["decision"]["best_learned_variant"] == "protected_e2e"
    assert result["decision"]["strictly_above_65"] is True
    assert result["decision"]["strictly_above_matched_uniform"] is True
    assert result["comparisons"]["protected_minus_transition"] == pytest.approx(
        0.4
    )


def test_official60_aggregate_rejects_commit_drift(tmp_path: Path) -> None:
    paths = _suite_paths(tmp_path)
    _write_evidence(
        tmp_path,
        "protected_e2e",
        65.6,
        commit="9" * 40,
    )

    with pytest.raises(ValueError, match="commit"):
        aggregate_official60(
            expected_commit=COMMIT,
            protocol_manifest_sha256=P0_SHA256,
            authorization_sha256=AUTH_SHA256,
            evidence_paths=paths,
        )


def test_official60_aggregate_rejects_unmatched_updates(tmp_path: Path) -> None:
    paths = _suite_paths(tmp_path)
    paths[2] = _write_evidence(
        tmp_path,
        "protected_e2e",
        65.6,
        updates=5999,
    )

    with pytest.raises(ValueError, match="exposure"):
        aggregate_official60(
            expected_commit=COMMIT,
            protocol_manifest_sha256=P0_SHA256,
            authorization_sha256=AUTH_SHA256,
            evidence_paths=paths,
        )
