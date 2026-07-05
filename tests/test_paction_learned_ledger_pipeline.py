from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.bata import run_paction_learned_policy_ledger_pipeline as ledger_pipeline
from tools.bata import train_paction_acquisition_policy as train_policy
from tools.bata import validate_paction_learned_policy_ledger as validate_ledger


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sample_row(sample_id: str, p_action: list[float], action_target: list[float], boundaries: list[int]) -> dict:
    return {
        "sample_id": sample_id,
        "dense_len": len(p_action),
        "valid_len": len(p_action),
        "frame_signals": {"p_action": p_action},
        "p_action": p_action,
        "action_target": action_target,
        "gt_boundaries": boundaries,
        "strategy_selected_positions": {"delta_p_action": [1, 4]},
    }


@pytest.mark.skipif(
    os.name == "nt",
    reason="torch training smoke tests run on the Linux/remote training environment",
)
def test_learned_policy_pipeline_generates_fixed_and_dynamic_strict_ledgers(tmp_path: Path) -> None:
    sample_jsonl = tmp_path / "samples.jsonl"
    checkpoint = tmp_path / "policy.pth"
    rows = [
        _sample_row("video_test_0001|0", [0.05, 0.70, 0.92, 0.20, 0.80, 0.10], [0, 1, 1, 0, 1, 0], [1, 4]),
        _sample_row("video_test_0002|0", [0.10, 0.25, 0.88, 0.91, 0.35, 0.05], [0, 0, 1, 1, 0, 0], [2, 3]),
    ]
    _write_jsonl(sample_jsonl, rows)
    train_policy.run_training(
        sample_jsonl,
        out_dir=tmp_path / "train_out",
        checkpoint_path=checkpoint,
        epochs=1,
        batch_size=2,
        hidden_dim=8,
        num_layers=1,
        fixed_budget=3,
        dynamic_budget_buckets=[2, 3, 4],
        device="cpu",
        seed=0,
    )

    summary = ledger_pipeline.run_pipeline(
        input_jsonl=sample_jsonl,
        checkpoint_path=checkpoint,
        out_dir=tmp_path / "ledgers",
        fixed_budgets=[2, 4],
        dynamic_target_len=4,
        dynamic_budget_buckets=[2, 3, 4],
        device="cpu",
        deploy_selection_ledger=True,
    )

    assert summary["decision"] == "C3_PACTION_LEARNED_POLICY_LEDGER_PIPELINE_READY"
    assert sorted(summary["ledgers"]) == ["learned_dynamic", "learned_fixed_2", "learned_fixed_4"]
    for name, item in summary["ledgers"].items():
        ledger_path = Path(item["ledger_jsonl"])
        sample_path = Path(item["sample_jsonl"])
        assert ledger_path.is_file(), name
        assert sample_path.is_file(), name
        sample_rows = _read_jsonl(sample_path)
        assert all("action_target" not in row for row in sample_rows)
        assert all("gt_boundaries" not in row for row in sample_rows)
        assert item["validation_summary"]["metric_sample_jsonl"] == str(sample_jsonl)
        ledger_rows = _read_jsonl(ledger_path)
        assert ledger_rows
        assert all(row["deploy_selection_ledger"] is True for row in ledger_rows)
        assert all(row["diagnostics"]["uniform_visible_fill_count"] == 0 for row in ledger_rows)
        assert item["validation_summary"]["uses_uniform_fill"] is False
        assert "max_gap" in item["validation_summary"]
        assert "p95_gap" in item["validation_summary"]
        assert "boundary_support_r1" in item["validation_summary"]
        assert "action_positive_coverage" in item["validation_summary"]


def test_learned_policy_ledger_validator_rejects_uniform_fill_and_forbidden_flags(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(
        samples,
        [_sample_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], [1, 3])],
    )
    _write_jsonl(
        ledger,
        [
            {
                "schema_version": "pc_ot_mras_frontend_value_transport_ledger_v0",
                "sample_id": "video_test_0001|0",
                "selected_positions_unit": "local_dense_index",
                "selected_positions": [1, 3],
                "target_len": 2,
                "selected_count": 2,
                "valid_len": 4,
                "dense_len": 4,
                "deploy_selection_ledger": True,
                "diagnostic_only": False,
                "uses_gt": False,
                "uses_teacher": False,
                "uses_oracle": False,
                "uses_cache": False,
                "uses_raw_prediction": False,
                "uses_checkpoint": False,
                "diagnostics": {"uniform_visible_fill_count": 1, "source_strategy": "learned_paction_gap_loss_value"},
            }
        ],
    )

    with pytest.raises(ValueError, match="uniform_visible_fill_count"):
        validate_ledger.validate_ledger(
            sample_jsonl=samples,
            ledger_jsonl=ledger,
            strategy="learned_paction_gap_loss_value",
            expected_target_len=2,
            require_selected_count=2,
            require_deployable=True,
        )


def test_learned_policy_ledger_validator_rejects_bootstrap_policy_source(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    checkpoint = tmp_path / "policy.pth"
    checkpoint.write_bytes(b"fake checkpoint bytes")
    _write_jsonl(
        samples,
        [
            {
                **_sample_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], [1, 3]),
                "paction_policy": {
                    "source": "bootstrap_paction_gap_loss_surrogate_policy",
                    "checkpoint_path": str(checkpoint),
                    "uses_uniform_fill": False,
                    "uses_uniform_scaffold": False,
                },
            }
        ],
    )
    _write_jsonl(
        ledger,
        [
            {
                "schema_version": "pc_ot_mras_frontend_value_transport_ledger_v0",
                "sample_id": "video_test_0001|0",
                "selected_positions_unit": "local_dense_index",
                "selected_positions": [1, 3],
                "target_len": 2,
                "selected_count": 2,
                "valid_len": 4,
                "dense_len": 4,
                "deploy_selection_ledger": True,
                "diagnostic_only": False,
                "policy_source": "learned_paction_gap_loss_policy_checkpoint",
                "policy_checkpoint_path": str(checkpoint),
                "policy_checkpoint_sha256": "2f05d4b689d2705f6f780e9304eb440953164c179f7f42c7eac98227cd3fc60c",
                "uses_gt": False,
                "uses_teacher": False,
                "uses_oracle": False,
                "uses_cache": False,
                "uses_raw_prediction": False,
                "uses_checkpoint": False,
                "diagnostics": {
                    "uniform_visible_fill_count": 0,
                    "source_strategy": "learned_paction_gap_loss_value",
                    "policy_source": "learned_paction_gap_loss_policy_checkpoint",
                    "policy_checkpoint_path": str(checkpoint),
                    "policy_checkpoint_sha256": "2f05d4b689d2705f6f780e9304eb440953164c179f7f42c7eac98227cd3fc60c",
                },
            }
        ],
    )

    with pytest.raises(ValueError, match="paction_policy.source"):
        validate_ledger.validate_ledger(
            sample_jsonl=samples,
            ledger_jsonl=ledger,
            strategy="learned_paction_gap_loss_value",
            expected_target_len=2,
            require_selected_count=2,
            require_deployable=True,
            require_policy_source="learned_paction_gap_loss_policy_checkpoint",
            require_checkpoint_path=checkpoint,
        )
