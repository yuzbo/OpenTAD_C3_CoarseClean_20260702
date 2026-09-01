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
        allow_bootstrap_for_tests=True,
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
    provenance = rows[0]["paction_policy"]["p_action_provenance"]
    assert provenance["uses_prediction_cache"] is False
    assert provenance["prediction_uses_gt"] is False


def test_policy_application_requires_checkpoint_by_default(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "samples.policy.jsonl"
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

    with pytest.raises(ValueError, match="checkpoint_path is required"):
        apply_policy.run_policy_application(input_jsonl, output_jsonl, fixed_budget=2)
    assert not output_jsonl.exists()


@pytest.mark.parametrize("flag", ["uses_gt", "uses_gt_for_diagnostics", "diagnostic_only", "training_only"])
def test_policy_application_rejects_leak_flags_even_in_explicit_bootstrap_mode(tmp_path: Path, flag: str) -> None:
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "samples.policy.jsonl"
    row = {
        "sample_id": "video_test_0001|0",
        "dense_len": 4,
        "valid_len": 4,
        "frame_signals": {"p_action": [0.10, 0.90, 0.20, 0.80]},
        flag: True,
    }
    input_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=flag):
        apply_policy.run_policy_application(
            input_jsonl,
            output_jsonl,
            fixed_budget=2,
            allow_bootstrap_for_tests=True,
        )


def test_policy_application_rejects_rows_without_paction_signal(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "samples.policy.jsonl"
    input_jsonl.write_text(json.dumps({"sample_id": "video_test_0001|0", "dense_len": 4}) + "\n", encoding="utf-8")

    try:
        apply_policy.run_policy_application(input_jsonl, output_jsonl, allow_bootstrap_for_tests=True)
    except ValueError as exc:
        assert "p_action" in str(exc)
    else:  # pragma: no cover - failure branch
        raise AssertionError("run_policy_application should reject rows without p_action")


def _paction_positive_provenance() -> dict:
    return {
        "p_action_source": "lowres_action_probe",
        "probe_model": "mobilenetv3_64px",
        "no_gt_generation": True,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "prediction_uses_gt": False,
    }


def test_policy_application_strict_source_rejects_gt_payload_before_strip(tmp_path: Path) -> None:
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
                "paction_positive_provenance": _paction_positive_provenance(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden strict deploy p_action source"):
        apply_policy.run_policy_application(
            input_jsonl,
            output_jsonl,
            fixed_budget=2,
            strip_deploy_invisible_payload=True,
            strict_deploy_source=True,
            allow_bootstrap_for_tests=True,
        )


def test_policy_application_strict_source_requires_verifiable_provenance(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "samples.policy.jsonl"
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

    with pytest.raises(ValueError, match="p_action positive provenance"):
        apply_policy.run_policy_application(
            input_jsonl,
            output_jsonl,
            fixed_budget=2,
            strict_deploy_source=True,
            allow_bootstrap_for_tests=True,
        )


def test_policy_application_strict_source_accepts_provenance_and_strips_signals(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "samples.policy.jsonl"
    input_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "video_test_0001|0",
                "dense_len": 4,
                "valid_len": 4,
                "frame_signals": {"p_action": [0.10, 0.90, 0.20, 0.80]},
                "paction_positive_provenance": _paction_positive_provenance(),
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
        strict_deploy_source=True,
        allow_bootstrap_for_tests=True,
    )
    rows = _read_jsonl(output_jsonl)

    assert summary["strip_deploy_invisible_payload"] is True
    assert summary["strict_deploy_source"] is True
    assert "strategy_selected_positions" in rows[0]
    assert "paction_policy" in rows[0]
    assert rows[0]["paction_policy"]["p_action_provenance"]["probe_model"] == "mobilenetv3_64px"
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

    apply_policy.run_policy_application(input_jsonl, policy_jsonl, fixed_budget=4, allow_bootstrap_for_tests=True)
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
    apply_policy.run_policy_application(input_jsonl, policy_jsonl, fixed_budget=2, allow_bootstrap_for_tests=True)

    with pytest.raises(ValueError, match="checkpoint policy source"):
        convert_ledger.run_conversion(
            policy_jsonl,
            ledger_jsonl,
            strategy="learned_paction_gap_loss_value",
            target_len=2,
            require_selected_count=2,
            deploy_selection_ledger=True,
        )
