from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import apply_paction_acquisition_policy as apply_policy
from tools.bata import convert_lowres_probe_samples_to_value_transport_ledger as convert_ledger


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_policy_application_enriches_probe_samples_with_learned_strategies(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "samples.policy.jsonl"
    summary_json = tmp_path / "summary.json"
    input_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "video_test_0001|0",
                "dense_len": 4,
                "valid_len": 4,
                "strategy_selected_positions": {"delta_p_action": [1, 3]},
                "frame_signals": {"p_action": [0.10, 0.90, 0.20, 0.80]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = apply_policy.run_policy_application(
        input_jsonl,
        output_jsonl,
        summary_json=summary_json,
        fixed_budget=2,
        dynamic_budget_buckets=[1, 2, 4],
    )
    rows = _read_jsonl(output_jsonl)

    assert summary["decision"] == "C3_PACTION_POLICY_APPLICATION_READY"
    assert summary["row_count"] == 1
    assert summary["fixed_budget"] == 2
    assert summary["dynamic_budget_buckets"] == [1, 2, 4]
    assert summary_json.is_file()
    assert len(rows) == 1
    strategies = rows[0]["strategy_selected_positions"]
    assert "delta_p_action" in strategies
    assert "learned_paction_gap_loss_value" in strategies
    assert "learned_paction_gap_loss_dynamic_budget" in strategies
    assert len(strategies["learned_paction_gap_loss_value"]) == 2
    assert rows[0]["paction_policy"]["source"] == "bootstrap_paction_gap_loss_surrogate_policy"
    assert rows[0]["paction_policy"]["uses_uniform_fill"] is False


def test_policy_application_rejects_rows_without_paction_signal(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "samples.policy.jsonl"
    input_jsonl.write_text(json.dumps({"sample_id": "video_test_0001|0", "dense_len": 4}) + "\n", encoding="utf-8")

    try:
        apply_policy.run_policy_application(input_jsonl, output_jsonl)
    except ValueError as exc:
        assert "p_action" in str(exc)
    else:  # pragma: no cover - failure branch
        raise AssertionError("run_policy_application should reject rows without p_action")


def test_policy_application_can_strip_deploy_invisible_payload(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "samples.policy.jsonl"
    input_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "video_test_0001|0",
                "dense_len": 4,
                "valid_len": 4,
                "frame_signals": {"p_action": [0.10, 0.90, 0.20, 0.80]},
                "p_action": [0.10, 0.90, 0.20, 0.80],
                "action_target": [0, 1, 0, 1],
                "uses_gt_for_diagnostics": True,
                "gt_boundaries": [1, 3],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = apply_policy.run_policy_application(
        input_jsonl,
        output_jsonl,
        fixed_budget=2,
        strip_deploy_invisible_payload=True,
    )
    rows = _read_jsonl(output_jsonl)

    assert summary["strip_deploy_invisible_payload"] is True
    assert "strategy_selected_positions" in rows[0]
    assert "paction_policy" in rows[0]
    assert rows[0]["deploy_invisible_payload_stripped"] is True
    assert "frame_signals" not in rows[0]
    assert "p_action" not in rows[0]
    assert "action_target" not in rows[0]
    assert "uses_gt_for_diagnostics" not in rows[0]
    assert "gt_boundaries" not in rows[0]


def test_policy_application_matches_short_valid_ratio_fixed_budget(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "samples.jsonl"
    policy_jsonl = tmp_path / "samples.policy.jsonl"
    ledger_jsonl = tmp_path / "value_transport_ledger.jsonl"
    input_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "video_test_0001|0",
                "dense_len": 8,
                "valid_len": 5,
                "strategy_selected_positions": {"delta_p_action": [1, 3]},
                "frame_signals": {
                    "p_action": [0.10, 0.90, 0.20, 0.80, 0.70, 0.60, 0.50, 0.40],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    apply_policy.run_policy_application(input_jsonl, policy_jsonl, fixed_budget=4)
    rows = _read_jsonl(policy_jsonl)

    selected = rows[0]["strategy_selected_positions"]["learned_paction_gap_loss_value"]
    assert len(selected) == 3
    assert rows[0]["paction_policy"]["fixed_budget"] == 3

    summary = convert_ledger.run_conversion(
        policy_jsonl,
        ledger_jsonl,
        strategy="learned_paction_gap_loss_value",
        target_len=4,
        require_selected_count=4,
        allow_short_valid_ratio_count=True,
        deploy_selection_ledger=False,
    )

    assert summary["min_selected_count"] == 3
    assert summary["max_selected_count"] == 3


def test_bootstrap_policy_output_cannot_be_converted_to_deploy_ledger(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "samples.jsonl"
    policy_jsonl = tmp_path / "samples.policy.jsonl"
    ledger_jsonl = tmp_path / "value_transport_ledger.jsonl"
    input_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "video_test_0001|0",
                "dense_len": 4,
                "valid_len": 4,
                "frame_signals": {"p_action": [0.10, 0.90, 0.20, 0.80]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    apply_policy.run_policy_application(input_jsonl, policy_jsonl, fixed_budget=2)

    with pytest.raises(ValueError, match="checkpoint policy source"):
        convert_ledger.run_conversion(
            policy_jsonl,
            ledger_jsonl,
            strategy="learned_paction_gap_loss_value",
            target_len=2,
            require_selected_count=2,
            deploy_selection_ledger=True,
        )
