from __future__ import annotations

import json

import pytest

from tools.bata.finalize_duca_rime_inference_ledger import finalize_ledger


def _row(video, start=0):
    return {
        "schema_version": "duca_rime_inference_ledger_v1",
        "video_id": video,
        "window_start_frame": start,
        "arm": "rime_full",
        "candidate_budgets": [2, 4],
        "requested_k": 4,
        "effective_k": 4,
        "unique_k": 4,
        "backbone_input_k": 4,
        "padded_k": 4,
        "risk_fallback": False,
        "cost_unit": "heavy_rgb_frames",
        "dense_valid_len": 4,
        "selected_dense_indices": [0, 1, 2, 3],
        "max_gap_seconds_cap": 1.0,
        "observed_max_gap_seconds": 0.75,
        "budget_protocol_sha256": "a" * 64,
        "provenance": {
            "uses_gt": False,
            "uses_teacher": False,
            "uses_prediction_cache": False,
            "uses_test_batch_composition": False,
            "raw_predictions_stored": False,
        },
    }


def test_finalize_rime_inference_ledger_is_sorted_unique_and_no_padding(tmp_path):
    shard = tmp_path / "rank0.jsonl"
    shard.write_text(
        json.dumps(_row("v1")) + "\n" + json.dumps(_row("v0")) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "ledger.jsonl"
    result = finalize_ledger(
        shards=[shard],
        output_jsonl=output,
        expected_arm="rime_full",
        expected_protocol_sha256="a" * 64,
    )
    assert result["record_count"] == 2
    assert result["no_padding_ledger"] is True
    assert result["max_observed_gap_seconds"] == pytest.approx(0.75)
    assert json.loads(output.read_text(encoding="utf-8").splitlines()[0])["video_id"] == "v0"


def test_finalize_rime_inference_ledger_rejects_padding(tmp_path):
    row = _row("v0")
    row["padded_k"] = 8
    shard = tmp_path / "rank0.jsonl"
    shard.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no-padding"):
        finalize_ledger(
            shards=[shard],
            output_jsonl=tmp_path / "ledger.jsonl",
            expected_arm="rime_full",
        )


def test_finalize_rime_inference_ledger_rejects_out_of_range_position(tmp_path):
    row = _row("v0")
    row["selected_dense_indices"][-1] = row["dense_valid_len"]
    shard = tmp_path / "rank0.jsonl"
    shard.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no-padding"):
        finalize_ledger(
            shards=[shard],
            output_jsonl=tmp_path / "ledger.jsonl",
            expected_arm="rime_full",
        )


def test_finalize_stage0_ledger_seals_explicit_short_window_budget_truth(tmp_path):
    row = _row("v0")
    row.update(
        {
            "arm": "exact_uniform",
            "candidate_budgets": [512],
            "requested_k": 512,
            "effective_k": 224,
            "unique_k": 224,
            "backbone_input_k": 224,
            "padded_k": 224,
            "dense_valid_len": 231,
            "selected_dense_indices": list(range(224)),
            "raw_budget": 512,
            "reachable_budget": 224,
            "realized_budget": 224,
            "projection_unused_budget": 288,
            "solver_unused_budget": 0,
            "budget_scope": "window_fixed_request",
            "claim_scope": "stage0_engineering_window_execution",
        }
    )
    shard = tmp_path / "rank0.jsonl"
    shard.write_text(json.dumps(row) + "\n", encoding="utf-8")
    summary = finalize_ledger(
        shards=[shard],
        output_jsonl=tmp_path / "ledger.jsonl",
        expected_arm="exact_uniform",
        expected_protocol_sha256="a" * 64,
        require_explicit_budget_truth=True,
    )
    assert summary["explicit_budget_truth"] is True
    assert summary["raw_budget_total"] == 512
    assert summary["reachable_budget_total"] == 224
    assert summary["realized_budget_total"] == 224
    assert summary["projection_unused_budget_total"] == 288
    assert summary["solver_unused_budget_total"] == 0


def test_finalize_stage0_ledger_rejects_inconsistent_explicit_budget_truth(tmp_path):
    row = _row("v0")
    row.update(
        {
            "arm": "exact_uniform",
            "raw_budget": 4,
            "reachable_budget": 4,
            "realized_budget": 4,
            "projection_unused_budget": 1,
            "solver_unused_budget": 0,
            "budget_scope": "window_fixed_request",
            "claim_scope": "stage0_engineering_window_execution",
        }
    )
    shard = tmp_path / "rank0.jsonl"
    shard.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="budget truth is inconsistent"):
        finalize_ledger(
            shards=[shard],
            output_jsonl=tmp_path / "ledger.jsonl",
            expected_arm="exact_uniform",
            expected_protocol_sha256="a" * 64,
            require_explicit_budget_truth=True,
        )


def test_finalize_stage0_ledger_requires_hash_and_explicit_fields(tmp_path):
    row = _row("v0")
    row["arm"] = "exact_uniform"
    shard = tmp_path / "rank0.jsonl"
    shard.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected protocol SHA-256"):
        finalize_ledger(
            shards=[shard],
            output_jsonl=tmp_path / "ledger.jsonl",
            expected_arm="exact_uniform",
            require_explicit_budget_truth=True,
        )
    with pytest.raises(ValueError, match="explicit budget truth is required"):
        finalize_ledger(
            shards=[shard],
            output_jsonl=tmp_path / "ledger.jsonl",
            expected_arm="exact_uniform",
            expected_protocol_sha256="a" * 64,
            require_explicit_budget_truth=True,
        )


def test_finalize_hrime_stage1_oracle_ledger_is_explicitly_non_deployable(tmp_path):
    row = _row("v0")
    row["arm"] = "hrime_stage1_learned_positions"
    row.update(
        {
            "raw_budget": 4,
            "reachable_budget": 4,
            "realized_budget": 4,
            "projection_unused_budget": 0,
            "solver_unused_budget": 0,
            "budget_scope": "video_exact_total_window_assignment",
            "claim_scope": (
                "stage1_development_oracle_execution_not_deployable"
            ),
        }
    )
    row["decision_provenance"] = {
        "role": "hrime_stage1_joint_oracle",
        "oracle_only": True,
        "deployment_candidate": False,
        "uses_official_final": False,
        "uses_gt": True,
        "uses_teacher": False,
        "uses_prediction_cache": False,
        "uses_test_batch_composition": False,
        "position_policy": "frozen_rime_selector",
        "assignment_sha256": "b" * 64,
    }
    row["provenance"]["uses_gt"] = True
    shard = tmp_path / "rank0.jsonl"
    shard.write_text(json.dumps(row) + "\n", encoding="utf-8")
    summary = finalize_ledger(
        shards=[shard],
        output_jsonl=tmp_path / "ledger.jsonl",
        expected_arm="hrime_stage1_learned_positions",
        expected_protocol_sha256="a" * 64,
        require_explicit_budget_truth=True,
        allow_oracle_only=True,
        expected_decision_role="hrime_stage1_joint_oracle",
    )
    assert summary["oracle_only"] is True
    assert summary["deployment_candidate"] is False
    assert summary["development_gt_used_for_oracle_assignment"] is True
    assert summary["official_final_labels_used_for_decision"] is False
    assert summary["explicit_budget_truth"] is True
    assert summary["budget_scope"] == "video_exact_total_window_assignment"


def test_finalize_hrime_stage1_oracle_ledger_fails_without_explicit_permission(
    tmp_path,
):
    row = _row("v0")
    row["arm"] = "hrime_stage1_learned_positions"
    row["decision_provenance"] = {
        "role": "hrime_stage1_joint_oracle",
        "oracle_only": True,
        "deployment_candidate": False,
        "uses_official_final": False,
        "uses_gt": True,
    }
    row["provenance"]["uses_gt"] = True
    shard = tmp_path / "rank0.jsonl"
    shard.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contaminated or mismatched"):
        finalize_ledger(
            shards=[shard],
            output_jsonl=tmp_path / "ledger.jsonl",
            expected_arm="hrime_stage1_learned_positions",
        )
