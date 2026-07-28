from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.bata.freeze_hrime_stage1_preregistration import (
    build_preregistration_payload,
)
from tools.bata.hrime_stage1_oracle import (
    EXECUTION_RECEIPT_SCHEMA,
    PREREGISTRATION_SCHEMA,
    STAGE1_ALLOCATION_CONTRACT,
    STAGE1_EVALUATION_CONTRACT,
    STRATEGY_CONTRACTS,
    WINDOW_OPTION_SCHEMA,
    build_stage1_plan,
    canonical_sha256,
    finalize_stage1_oracle,
    surrogate_audit,
    validate_preregistration,
    validate_window_option_rows,
)


def _write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preregistration(*, anchors=(384,)) -> dict:
    payload = {
        "schema_version": PREREGISTRATION_SCHEMA,
        "status": "frozen",
        "task": "offline_temporal_action_detection",
        "git_commit": "1" * 40,
        "split_manifest_sha256": "2" * 64,
        "split_assignment_sha256": "3" * 64,
        "development_role": "certification_development",
        "uses_official_final": False,
        "official_final_used_for_selection": False,
        "candidate_budgets": [192, 256, 384, 512],
        "anchor_nominal_budgets": list(anchors),
        "allocation_contract": STAGE1_ALLOCATION_CONTRACT,
        "evaluation_contract": STAGE1_EVALUATION_CONTRACT,
        "strategy_contract_sha256": canonical_sha256(STRATEGY_CONTRACTS),
        "oracle_risk_weight": 0.0,
        "primary_endpoint": {
            "metric": "avg_map",
            "direction": "higher",
            "alpha": 0.05,
            "min_mean_delta": 0.10,
            "min_lcb_delta": 0.05,
            "noninferiority_margin": 0.0,
        },
        "bootstrap": {"unit": "video", "samples": 200, "seed": 3407},
        "multiplicity": {
            "method": "intersection_union_single_primary_with_guardrails",
            "family": [
                f"k{anchor}:joint_oracle_vs_{comparator}:avg_map"
                for anchor in anchors
                for comparator in (
                    "uniform_same_total",
                    "independent_exact_total",
                )
            ],
        },
        "guardrails": [],
        "surrogate_audit": {
            "min_spearman": -1.0,
            "min_sign_agreement": 0.0,
            "max_worst_rank_error": 1.0,
            "error_normalization": "fractional_midrank",
        },
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def _row(video: str, start: int, *, flip_oracle: bool = False) -> dict:
    budgets = [192, 256, 384, 512]
    if start == 0:
        oracle = [0.0, 0.1, 0.2, 1.0]
    else:
        oracle = [0.0, 0.8, 0.2, 0.1]
    if flip_oracle:
        oracle = list(reversed(oracle))
    row = {
        "schema_version": WINDOW_OPTION_SCHEMA,
        "status": "measured",
        "video_id": video,
        "window_start_frame": start,
        "valid_length": 768,
        "split_role": "certification_development",
        "split_assignment_sha256": "3" * 64,
        "candidate_budgets": budgets,
        "options": [
            {
                "effective_k": budget,
                "nominal_budgets": [budget],
                "predicted_utility": float(-abs(budget - 384)),
                "predicted_risk": 0.0,
                "oracle_utility": float(utility),
                "oracle_risk": 0.0,
            }
            for budget, utility in zip(budgets, oracle)
        ],
        "source_identity_sha256": "4" * 64,
        "provenance": {
            "uses_official_final": False,
            "uses_gt_for_oracle_utility": True,
            "uses_gt_for_predicted_utility": False,
            "uses_teacher": False,
            "uses_prediction_cache": False,
            "oracle_only": True,
            "deployment_candidate": False,
        },
    }
    row["record_sha256"] = canonical_sha256(row)
    return row


def _rows() -> list[dict]:
    return [
        _row("video_a", 0),
        _row("video_a", 576),
        _row("video_b", 0, flip_oracle=True),
        _row("video_b", 576, flip_oracle=True),
    ]


def test_stage1_preregistration_requires_internal_content_hash():
    payload = _preregistration()
    validated = validate_preregistration(payload)
    assert validated["candidate_budgets"] == (192, 256, 384, 512)
    payload["primary_endpoint"]["min_mean_delta"] = 999.0
    with pytest.raises(ValueError, match="content hash drift"):
        validate_preregistration(payload)


def test_preregistration_builder_freezes_numeric_and_evaluator_contracts():
    seed = _preregistration()
    payload = build_preregistration_payload(
        git_commit=seed["git_commit"],
        split_manifest_sha256=seed["split_manifest_sha256"],
        split_assignment_sha256=seed["split_assignment_sha256"],
        anchor_nominal_budgets=seed["anchor_nominal_budgets"],
        oracle_risk_weight=seed["oracle_risk_weight"],
        primary_endpoint=seed["primary_endpoint"],
        bootstrap=seed["bootstrap"],
        multiplicity=seed["multiplicity"],
        guardrails=seed["guardrails"],
        surrogate_audit=seed["surrogate_audit"],
    )
    validated = validate_preregistration(payload)
    assert validated["allocation_contract"]["score_dtype"] == "int64"
    assert validated["allocation_contract"]["score_rounding"] == "ROUND_HALF_EVEN"
    assert (
        validated["evaluation_contract"]["pipeline"]
        == "full_detector_window_merge_nms"
    )
    payload["allocation_contract"]["score_scale"] = 1
    payload["content_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="numeric contract drift"):
        validate_preregistration(payload)


def test_preregistration_rejects_negative_oracle_risk_weight_at_freeze_and_validation():
    seed = _preregistration()
    with pytest.raises(ValueError, match="must be non-negative"):
        build_preregistration_payload(
            git_commit=seed["git_commit"],
            split_manifest_sha256=seed["split_manifest_sha256"],
            split_assignment_sha256=seed["split_assignment_sha256"],
            anchor_nominal_budgets=seed["anchor_nominal_budgets"],
            oracle_risk_weight=-0.01,
            primary_endpoint=seed["primary_endpoint"],
            bootstrap=seed["bootstrap"],
            multiplicity=seed["multiplicity"],
            guardrails=seed["guardrails"],
            surrogate_audit=seed["surrogate_audit"],
        )
    seed["oracle_risk_weight"] = -0.01
    seed["content_sha256"] = canonical_sha256(
        {key: value for key, value in seed.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="oracle risk weight is invalid"):
        validate_preregistration(seed)


def test_window_option_rows_are_video_complete_and_hash_bound():
    rows = _rows()
    validated = validate_window_option_rows(
        rows,
        expected_videos=("video_a", "video_b"),
        expected_split_role="certification_development",
        expected_split_assignment_sha256="3" * 64,
        candidate_budgets=(192, 256, 384, 512),
    )
    assert [(row["video_id"], row["window_start_frame"]) for row in validated] == [
        ("video_a", 0),
        ("video_a", 576),
        ("video_b", 0),
        ("video_b", 576),
    ]
    rows[0]["options"][0]["oracle_utility"] = 123.0
    with pytest.raises(ValueError, match="record hash drift"):
        validate_window_option_rows(
            rows,
            expected_videos=("video_a", "video_b"),
            expected_split_role="certification_development",
            expected_split_assignment_sha256="3" * 64,
            candidate_budgets=(192, 256, 384, 512),
        )


def test_stage1_plan_emits_exact_same_total_replays_and_oracle_provenance(tmp_path: Path):
    prereg = _preregistration()
    rows = validate_window_option_rows(
        _rows(),
        expected_videos=("video_a", "video_b"),
        expected_split_role="certification_development",
        expected_split_assignment_sha256="3" * 64,
        candidate_budgets=(192, 256, 384, 512),
    )
    result = build_stage1_plan(
        rows=rows,
        window_options_sha256="7" * 64,
        preregistration=prereg,
        preregistration_sha256="5" * 64,
        budget_protocol_sha256="6" * 64,
        output_root=tmp_path / "plan",
    )
    plan = result["payload"]
    assert plan["authorizes_stage2_training"] is False
    assert plan["video_count"] == 2
    for video_plan in plan["video_plans"]:
        assert video_plan["target_total_effective_k"] == 768
        for strategy in STRATEGY_CONTRACTS:
            assert sum(video_plan["strategies"][strategy]["assignment"]) == 768
    for strategy, contract in STRATEGY_CONTRACTS.items():
        replay = Path(plan["replay_artifacts"][strategy]["384"]["path"])
        rows = [
            json.loads(line)
            for line in replay.read_text(encoding="utf-8").splitlines()
            if line
        ]
        assert len(rows) == 4
        assert {row["provenance"]["uses_gt"] for row in rows} == {
            contract["uses_gt"]
        }
        assert all(row["provenance"]["oracle_only"] for row in rows)
        assert all(not row["provenance"]["deployment_candidate"] for row in rows)


def test_stage1_plan_rejects_a_globally_degenerate_shuffled_null(tmp_path: Path):
    prereg = _preregistration()
    rows = _rows()
    for row in rows:
        for option in row["options"]:
            option["oracle_utility"] = -abs(int(option["effective_k"]) - 384)
        row["record_sha256"] = canonical_sha256(
            {key: value for key, value in row.items() if key != "record_sha256"}
        )
    rows = validate_window_option_rows(
        rows,
        expected_videos=("video_a", "video_b"),
        expected_split_role="certification_development",
        expected_split_assignment_sha256="3" * 64,
        candidate_budgets=(192, 256, 384, 512),
    )
    with pytest.raises(ValueError, match="shuffled-null cell is degenerate"):
        build_stage1_plan(
            rows=rows,
            window_options_sha256="7" * 64,
            preregistration=prereg,
            preregistration_sha256="5" * 64,
            budget_protocol_sha256="6" * 64,
            output_root=tmp_path / "degenerate-plan",
        )
    assert not (tmp_path / "degenerate-plan").exists()


def test_surrogate_audit_reports_rank_sign_and_worst_error():
    result = surrogate_audit([0.1, 0.4, -0.2], [0.2, 0.5, -0.1])
    assert result["spearman"] == pytest.approx(1.0)
    assert result["sign_agreement"] == pytest.approx(1.0)
    assert result["worst_rank_error"] == pytest.approx(0.0)
    zero_mismatch = surrogate_audit([0.0, 1.0], [-1.0, 1.0])
    assert zero_mismatch["sign_agreement"] == pytest.approx(0.5)


def test_complete_execution_matrix_alone_can_authorize_stage2(tmp_path: Path):
    prereg = _preregistration()
    prereg_path = tmp_path / "prereg.json"
    prereg_sha = _write_json(prereg_path, prereg)
    rows = validate_window_option_rows(
        _rows(),
        expected_videos=("video_a", "video_b"),
        expected_split_role="certification_development",
        expected_split_assignment_sha256="3" * 64,
        candidate_budgets=(192, 256, 384, 512),
    )
    plan_result = build_stage1_plan(
        rows=rows,
        window_options_sha256="7" * 64,
        preregistration=prereg,
        preregistration_sha256=prereg_sha,
        budget_protocol_sha256="6" * 64,
        output_root=tmp_path / "plan",
    )
    plan_path = Path(plan_result["path"])
    plan_sha = plan_result["sha256"]
    metric_values = {
        "uniform_same_total": {"video_a": 0.50, "video_b": 0.50},
        "independent_exact_total": {"video_a": 0.55, "video_b": 0.55},
        "joint_oracle": {"video_a": 0.75, "video_b": 0.75},
        "joint_same_k_uniform_positions": {"video_a": 0.60, "video_b": 0.60},
        "shuffled_null": {"video_a": 0.40, "video_b": 0.40},
    }
    bindings = []
    for strategy in STRATEGY_CONTRACTS:
        metrics_path = tmp_path / f"{strategy}.metrics.json"
        metrics_sha = _write_json(
            metrics_path,
            {"video_metrics": {"avg_map": metric_values[strategy]}},
        )
        receipt = {
            "schema_version": EXECUTION_RECEIPT_SCHEMA,
            "status": "passed",
            "strategy": strategy,
            "anchor_nominal_budget": 384,
            "plan_manifest_sha256": plan_sha,
            "uses_official_final": False,
            "oracle_only": True,
            "deployment_candidate": False,
            "position_policy": STRATEGY_CONTRACTS[strategy]["position_policy"],
            "uses_gt_at_decision": STRATEGY_CONTRACTS[strategy]["uses_gt"],
            "checkpoint_sha256": "7" * 64,
            "detector_backend": "ActionFormer",
            "evaluation_seed": 3407,
            "annotation_sha256": "8" * 64,
            "nms_contract_sha256": "9" * 64,
            "post_processing_pipeline_identity_sha256": "b" * 64,
            "official_evaluator_source_sha256": "a" * 64,
            "localization_metrics_path": str(metrics_path),
            "localization_metrics_sha256": metrics_sha,
        }
        receipt["content_sha256"] = canonical_sha256(receipt)
        receipt_path = tmp_path / f"{strategy}.receipt.json"
        receipt_sha = _write_json(receipt_path, receipt)
        bindings.append((strategy, 384, str(receipt_path), receipt_sha))
    result = finalize_stage1_oracle(
        plan_manifest=plan_path,
        plan_manifest_sha256=plan_sha,
        preregistration=prereg_path,
        preregistration_sha256=prereg_sha,
        execution_receipts=bindings,
        output_receipt=tmp_path / "oracle.receipt.json",
    )
    assert result["payload"]["status"] == "passed"
    assert result["payload"]["authorizes_stage2_training"] is True
    assert result["payload"]["gate_status"] == {
        "primary": True,
        "guardrails": True,
        "surrogate": True,
    }


def test_incomplete_execution_matrix_fails_closed(tmp_path: Path):
    prereg = _preregistration()
    prereg_path = tmp_path / "prereg.json"
    prereg_sha = _write_json(prereg_path, prereg)
    rows = validate_window_option_rows(
        _rows(),
        expected_videos=("video_a", "video_b"),
        expected_split_role="certification_development",
        expected_split_assignment_sha256="3" * 64,
        candidate_budgets=(192, 256, 384, 512),
    )
    plan_result = build_stage1_plan(
        rows=rows,
        window_options_sha256="7" * 64,
        preregistration=prereg,
        preregistration_sha256=prereg_sha,
        budget_protocol_sha256="6" * 64,
        output_root=tmp_path / "plan",
    )
    with pytest.raises(ValueError, match="matrix is incomplete"):
        finalize_stage1_oracle(
            plan_manifest=plan_result["path"],
            plan_manifest_sha256=plan_result["sha256"],
            preregistration=prereg_path,
            preregistration_sha256=prereg_sha,
            execution_receipts=[],
            output_receipt=tmp_path / "oracle.receipt.json",
        )
