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
UNI_COMPANION_VARIANTS = (
    "exact_uniform",
    "protected_e2e",
    "protected_e2e_bridge025",
    "protected_e2e_uni_companion",
)
HOMOTOPY_VARIANTS = (
    "exact_uniform",
    "protected_e2e",
    "protected_e2e_bridge025",
    "protected_e2e_homotopy025",
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
    }
    payload["artifact_chain_sha256"] = canonical_sha256(payload)
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
    return [_write_evidence(root, variant, maps[variant]) for variant in VARIANTS]


def _uni_companion_suite_paths(root: Path) -> list[Path]:
    maps = {
        "exact_uniform": 65.0,
        "protected_e2e": 64.6,
        "protected_e2e_bridge025": 65.4,
        "protected_e2e_uni_companion": 65.8,
    }
    return [
        _write_evidence(root, variant, maps[variant])
        for variant in UNI_COMPANION_VARIANTS
    ]


def _homotopy_suite_paths(
    root: Path,
    *,
    homotopy_map: float = 65.7,
) -> list[Path]:
    maps = {
        "exact_uniform": 65.0,
        "protected_e2e": 64.6,
        "protected_e2e_bridge025": 65.4,
        "protected_e2e_homotopy025": homotopy_map,
    }
    return [
        _write_evidence(root, variant, maps[variant])
        for variant in HOMOTOPY_VARIANTS
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
    assert result["comparisons"]["protected_minus_transition"] == pytest.approx(0.4)


def test_official60_aggregate_reports_uni_companion_optimization(
    tmp_path: Path,
) -> None:
    result = aggregate_official60(
        expected_commit=COMMIT,
        protocol_manifest_sha256=P0_SHA256,
        authorization_sha256=AUTH_SHA256,
        evidence_paths=_uni_companion_suite_paths(tmp_path),
    )

    assert result["suite_kind"] == "uni_companion_optimization"
    assert result["decision"]["best_learned_variant"] == "protected_e2e_uni_companion"
    assert result["decision"]["strictly_above_65"] is True
    assert result["decision"]["strictly_above_matched_uniform"] is True
    assert result["decision"]["bridge025_improves_protected"] is True
    assert result["decision"]["uni_companion_improves_bridge025"] is True
    assert result["comparisons"]["bridge025_minus_protected"] == pytest.approx(0.8)
    assert result["comparisons"]["uni_companion_minus_bridge025"] == pytest.approx(0.4)


def test_official60_aggregate_reports_homotopy_optimization(
    tmp_path: Path,
) -> None:
    result = aggregate_official60(
        expected_commit=COMMIT,
        protocol_manifest_sha256=P0_SHA256,
        authorization_sha256=AUTH_SHA256,
        evidence_paths=_homotopy_suite_paths(tmp_path),
    )

    assert result["suite_kind"] == "homotopy_optimization"
    assert result["decision"]["best_learned_variant"] == (
        "protected_e2e_homotopy025"
    )
    assert result["decision"]["homotopy_average_mAP"] == pytest.approx(65.7)
    assert result["decision"]["homotopy_strictly_above_65"] is True
    assert result["decision"]["homotopy_improves_bridge025"] is True
    assert result["decision"]["homotopy_improves_uniform"] is True
    assert result["comparisons"]["homotopy_minus_bridge025"] == pytest.approx(0.3)
    assert result["comparisons"]["homotopy_minus_uniform"] == pytest.approx(0.7)


def test_official60_homotopy_threshold_is_strict(tmp_path: Path) -> None:
    result = aggregate_official60(
        expected_commit=COMMIT,
        protocol_manifest_sha256=P0_SHA256,
        authorization_sha256=AUTH_SHA256,
        evidence_paths=_homotopy_suite_paths(tmp_path, homotopy_map=65.0),
    )

    assert result["decision"]["homotopy_strictly_above_65"] is False


def test_official60_submitter_requires_homotopy_authorization() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "submit_duca_protected_physical_official60_suite.sh"
    )
    script = script_path.read_text(encoding="utf-8")
    authorization_start = script.index(
        '    sys.argv[5] == "homotopy_optimization"'
    )
    authorization_end = script.index("\nPY", authorization_start)
    authorization_block = script[authorization_start:authorization_end]
    suite_start = script.index("  homotopy_optimization)")
    suite_end = script.index("    ;;", suite_start)
    suite_block = script[suite_start:suite_end]

    assert '"official60_four_arm_training"' in script
    assert '"official60_uni_companion_training"' in authorization_block
    assert '"official60_homotopy_training"' in authorization_block
    assert "exact_uniform" in suite_block
    assert "protected_e2e" in suite_block
    assert "protected_e2e_bridge025" in suite_block
    assert "protected_e2e_homotopy025" in suite_block
    assert "protected_e2e_uni_companion" not in suite_block


def test_official60_submitter_closes_completion_heredoc() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "submit_duca_protected_physical_official60_suite.sh"
    )
    lines = script_path.read_text(encoding="utf-8").splitlines()
    delimiter_index = lines.index("EOF", lines.index('cat > "${COMPLETE_JOB}" <<EOF'))

    assert lines[delimiter_index - 1].endswith("\\\\")


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


def test_official60_aggregate_rejects_tampered_artifact_chain(
    tmp_path: Path,
) -> None:
    paths = _homotopy_suite_paths(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["metrics"]["average_mAP"] = 99.0
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact-chain hash mismatch"):
        aggregate_official60(
            expected_commit=COMMIT,
            protocol_manifest_sha256=P0_SHA256,
            authorization_sha256=AUTH_SHA256,
            evidence_paths=paths,
        )
