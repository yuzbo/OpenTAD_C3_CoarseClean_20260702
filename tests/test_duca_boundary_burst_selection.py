from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import analyze_duca_selection_quality as quality
from tools.bata.select_duca_boundary_burst_candidates import (
    _effective_budget_contract_verified,
    _read_candidate,
    _ranking_key,
    validate_p0_real_gate,
)
from tools.bata.select_duca_frontend_checkpoint import sha256_file


def _contract_summary() -> dict:
    return {
        "sample_count": 3,
        "protocol": {
            "budget_matched": True,
            "valid_length_matched": True,
            "max_hole_matched": True,
            "sampling_contract_evidence": {
                "sample_count": 3,
                "requested_budget_min": 384,
                "requested_budget_max": 384,
                "effective_budget_min": 300,
                "effective_budget_max": 384,
                "selected_count_min": 300,
                "selected_count_max": 384,
                "budget_violation_count": 0,
                "requested_max_unselected_hole_min": 2,
                "requested_max_unselected_hole_max": 2,
                "observed_max_unselected_hole_min": 0,
                "observed_max_unselected_hole_max": 2,
                "max_hole_violation_count": 0,
            },
        },
    }


def test_effective_budget_contract_accepts_short_valid_windows() -> None:
    summary = _contract_summary()
    assert _effective_budget_contract_verified(summary)
    summary["protocol"]["sampling_contract_evidence"]["budget_violation_count"] = 1
    assert not _effective_budget_contract_verified(summary)
    assert not _effective_budget_contract_verified({})


def test_effective_budget_contract_rejects_self_asserted_boole_without_rows() -> None:
    summary = {
        "sample_count": 3,
        "protocol": {
            "budget_matched": True,
            "valid_length_matched": True,
            "max_hole_matched": True,
        },
    }
    assert not _effective_budget_contract_verified(summary)


def _candidate(variant: str, quota_gain: float, epoch: int) -> dict:
    return {
        "variant": variant,
        "epoch_one_based": epoch,
        "metrics": {
            "both_endpoints_quota_recall_gain": quota_gain,
            "endpoint_quota_recall_gain": quota_gain,
            "endpoint_bilateral_recall_gain": quota_gain,
            "boundary_recall_r0_gain": 0.0,
            "uniform_minus_learned_endpoint_distance": 0.0,
            "policy_transition_auroc_r0": 0.5,
        },
    }


def test_checkpoint_selection_uses_earliest_passing_epoch_not_holdout_best() -> None:
    weak_early = _candidate("burst_r2q3", 0.05, 5)
    strong_late = _candidate("burst_r2q3", 0.20, 20)

    assert sorted([strong_late, weak_early], key=_ranking_key)[0] is weak_early


def test_checkpoint_selection_uses_epoch_for_all_mechanisms() -> None:
    early = _candidate("burst_r4q5", 0.20, 10)
    late = _candidate("burst_r4q5", 0.20, 20)

    assert sorted([late, early], key=_ranking_key)[0] is early


def _record(sample_id: str, learned_positions: list[int]) -> dict:
    return {
        "schema_version": quality.RECORD_SCHEMA_VERSION,
        "sample_id": sample_id,
        "video_id": sample_id.split("|")[0],
        "valid_len": 8,
        "requested_budget": 4,
        "effective_budget": 4,
        "budget": 4,
        "max_hole_contract": {"requested_max_unselected_hole": 2},
        "gt_segments": [[2.0, 6.0]],
        "gt_boundary_validity": [[True, True]],
        "p_action": [0.05, 0.10, 0.80, 0.90, 0.85, 0.75, 0.10, 0.05],
        "transition_policy_scores": [0.0, 0.2, 1.0, 0.1, 0.1, 0.9, 0.2, 0.0],
        "abs_delta_p_action": [0.0, 0.05, 0.70, 0.10, 0.05, 0.65, 0.10, 0.05],
        "raw_transition_scores": [0.0, 0.1, 0.8, 0.2, 0.1, 0.7, 0.1, 0.0],
        "selected_positions": learned_positions,
    }


def _write_candidate(tmp_path: Path, variant: str) -> dict:
    checkpoint = tmp_path / f"{variant}.pth"
    records = tmp_path / f"{variant}.jsonl"
    output_dir = tmp_path / f"{variant}.quality"
    summary_path = output_dir / "selection_quality_summary.json"
    checkpoint.write_bytes(b"checkpoint")
    records.write_text(
        json.dumps(_record(f"{variant}|0", [1, 2, 5, 7])) + "\n",
        encoding="utf-8",
    )
    # Generate the fixture with the exact production analyzer invocation.
    quality.analyze_jsonl(
        records_jsonl=records,
        output_dir=output_dir,
        bootstrap_samples=2000,
        random_seed=3407,
        representative_per_stratum=2,
    )
    return {
        "epoch_one_based": 5,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "records_path": str(records),
        "records_sha256": sha256_file(records),
    }


def test_candidate_recomputes_summary_with_production_analyzer(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path, "gaussian_matched")

    result = _read_candidate(candidate, "gaussian_matched")

    assert result["records_sha256"] == candidate["records_sha256"]
    assert result["summary_sha256"] == candidate["summary_sha256"]


def test_candidate_reanalysis_rejects_empty_records(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path, "gaussian_matched")
    records = Path(candidate["records_path"])
    records.write_text("", encoding="utf-8")
    candidate["records_sha256"] = sha256_file(records)

    with pytest.raises(RuntimeError, match="records JSONL is empty"):
        _read_candidate(candidate, "gaussian_matched")


def test_candidate_reanalysis_rejects_summary_from_unrelated_records(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path, "burst_r2q3")
    records = Path(candidate["records_path"])
    with records.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_record("other|0", [0, 2, 5, 7])) + "\n")
    candidate["records_sha256"] = sha256_file(records)

    with pytest.raises(RuntimeError, match="disagrees with production reanalysis"):
        _read_candidate(candidate, "burst_r2q3")


def test_candidate_reanalysis_rejects_records_identity_drift(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path, "burst_r4q5")
    summary = Path(candidate["summary_path"])
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["records_jsonl"] = str(tmp_path / "unrelated.jsonl")
    summary.write_text(json.dumps(payload), encoding="utf-8")
    candidate["summary_sha256"] = sha256_file(summary)

    with pytest.raises(RuntimeError, match="records identity drift"):
        _read_candidate(candidate, "burst_r4q5")


def _write_p0_real_gate(path: Path, commit: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "duca_frontend_p0_real_cuda_gate_v1",
                "ok": True,
                "fail_closed": True,
                "git_binding": {"git_commit": commit},
                "final_git_binding": {"git_commit": commit},
            }
        ),
        encoding="utf-8",
    )


def test_p0_real_gate_identity_is_path_hash_schema_commit_and_ok_bound(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    gate = tmp_path / "p0_real_gate.json"
    _write_p0_real_gate(gate, commit)

    binding = validate_p0_real_gate(
        gate_path=gate,
        gate_sha256=sha256_file(gate),
        expected_commit=commit,
    )

    assert binding == {
        "path": str(gate.resolve()),
        "sha256": sha256_file(gate),
        "schema": "duca_frontend_p0_real_cuda_gate_v1",
        "git_commit": commit,
        "ok": True,
    }
    with pytest.raises(RuntimeError, match="path/hash drift"):
        validate_p0_real_gate(
            gate_path=gate,
            gate_sha256="0" * 64,
            expected_commit=commit,
        )
    _write_p0_real_gate(gate, "b" * 40)
    with pytest.raises(RuntimeError, match="contract drift"):
        validate_p0_real_gate(
            gate_path=gate,
            gate_sha256=sha256_file(gate),
            expected_commit=commit,
        )
    for field, value in (
        ("schema", "unexpected_schema"),
        ("ok", False),
        ("fail_closed", False),
    ):
        _write_p0_real_gate(gate, commit)
        payload = json.loads(gate.read_text(encoding="utf-8"))
        payload[field] = value
        gate.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeError, match="contract drift"):
            validate_p0_real_gate(
                gate_path=gate,
                gate_sha256=sha256_file(gate),
                expected_commit=commit,
            )
