from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import detector_aware_acquisition_policy as detector_policy
from tools.bata import detector_teacher_utility
from tools.bata import validate_stage4_detector_aware_truetime_curriculum as stage4


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _generator_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "teacher_generator.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "c3_detector_teacher_utility_generator_manifest_v1",
                "decision": "C3_DETECTOR_TEACHER_UTILITY_GENERATOR_MANIFEST_READY",
                "teacher_signal_source": "adatad_dense_teacher",
                "generator_source": "dense_detector_forward_train",
                "split_scope": "train_only",
                "input_split": "training",
                "uses_evaluator_outputs": False,
                "uses_raw_prediction": False,
                "uses_prediction_cache": False,
                "load_from_raw_predictions": False,
                "uses_val_or_test_gt_for_selection": False,
                "uses_gt_for_selection": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _stage2_teacher_evidence(tmp_path: Path) -> dict:
    dense_points = tmp_path / "teacher_dense_points.jsonl"
    base_samples = tmp_path / "train_paction_samples.jsonl"
    output_jsonl = tmp_path / "samples_with_teacher_utility.jsonl"
    summary_json = tmp_path / "teacher_utility_export.summary.json"
    checkpoint = tmp_path / "teacher.pth"
    config = tmp_path / "teacher.py"
    generator_manifest = _generator_manifest(tmp_path)
    checkpoint.write_bytes(b"stage2 teacher checkpoint")
    config.write_text("model = dict(type='ActionFormer')\n", encoding="utf-8")
    _write_jsonl(
        dense_points,
        [
            {
                "sample_id": "video_test_0001|0",
                "split": "training",
                "dense_len": 4,
                "valid_len": 4,
                "teacher_dense_points": [{"point_index": 1, "proposal_score": 0.8}],
                "teacher_signal_source": "adatad_dense_teacher",
                "teacher_axis": "dense_frame_index",
                "fps": 25.0,
                "snippet_stride": 4,
                "window_start_frame": 0,
                "window_size": 768,
            }
        ],
    )
    _write_jsonl(
        base_samples,
        [
            {
                "sample_id": "video_test_0001|0",
                "split": "training",
                "dense_len": 4,
                "valid_len": 4,
                "frame_signals": {"p_action": [0.1, 0.8, 0.2, 0.7]},
            }
        ],
    )
    detector_teacher_utility.run_export(
        dense_points,
        output_jsonl,
        summary_json=summary_json,
        base_samples_jsonl=base_samples,
        teacher_checkpoint_path=checkpoint,
        teacher_config_path=config,
        generator_manifest_json=generator_manifest,
        expected_split="training",
    )
    return detector_teacher_utility.validate_teacher_utility_export_evidence(
        summary_json,
        output_jsonl=output_jsonl,
        require_paction=True,
    )


def _stage2_policy_evidence() -> dict:
    return {
        "decision": "C3_DETECTOR_AWARE_POLICY_TRAIN_READY",
        "policy_family": "detector_aware_offline_selector",
        "stage_label": detector_policy.STAGE_LABEL,
        "policy_source": detector_policy.DETECTOR_AWARE_CHECKPOINT_POLICY_SOURCE,
        "teacher_target_scope": "train_only",
        "checkpoint_sha256": "a" * 64,
        "train_jsonl_sha256": "b" * 64,
        "utility_semantics": "signed_detector_utility_v1",
        "signed_utility_supported": True,
        "dynamic_gain_calibration": {
            "score_semantics": "calibrated_marginal_gain",
            "calibration_scope": "cross_video_comparable",
            "target_source": "abs_signed_detector_utility",
        },
        "uses_gt_for_selection": False,
        "uses_val_or_test_gt_for_selection": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "load_from_raw_predictions": False,
        "uses_uniform_fill": False,
        "uses_uniform_scaffold": False,
        "end_to_end": False,
    }


def _stage3_proof() -> dict:
    return {
        "route_variant": stage4.STAGE3_ROUTE,
        "geometry_roundtrip_passed": True,
        "prediction_inverse_map_passed": True,
        "selected_input_st_gradient_passed": True,
        "selected_input_selector_grad_norm": 0.25,
        "detector_loss_selector_grad_passed": True,
        "detector_loss_selector_grad_norm": 0.25,
        "selector_grad_nonzero": True,
        "loss_keys": ["loss_cls", "loss_reg"],
        "proof_source": "registered_detector_forward_train_cost_backward",
        "actionformer_proof_source": "opentad_actionformer_forward_train_cost_backward",
        "actionformer_detector_loss_selector_grad_passed": True,
        "actionformer_detector_loss_selector_grad_norm": 0.31,
        "actionformer_loss_keys": ["cls_loss", "reg_loss"],
        "actionformer_selected_axis_smoke": True,
        "sparse_distill_adapter_ready": True,
        "sparse_distill_claim_allowed": False,
        "sparse_distill_map_claim_allowed": False,
        "sparse_distill_proof_source": "fail_closed_sparse_detector_distillation_adapter",
        "uses_gt_for_selection": False,
        "uses_val_or_test_gt_for_selection": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "load_from_raw_predictions": False,
        "uses_uniform_fill": False,
        "uses_uniform_scaffold": False,
    }


def _ledger_summary(strategy: str) -> dict:
    dynamic = strategy == detector_policy.DETECTOR_AWARE_DYNAMIC_STRATEGY
    return {
        "schema_version": "c3_detector_aware_policy_ledger_validation_v1",
        "decision": "C3_DETECTOR_AWARE_POLICY_LEDGER_VALIDATION_PASS",
        "stage_label": detector_policy.STAGE_LABEL,
        "strategy": strategy,
        "split": "train",
        "required_policy_source": detector_policy.DETECTOR_AWARE_CHECKPOINT_POLICY_SOURCE,
        "min_selected_count": 2 if dynamic else 4,
        "max_selected_count": 4,
        "mean_selected_count": 3.0 if dynamic else 4.0,
        "max_gap": 2,
        "p95_gap": 2.0,
        "max_unselected_hole": 2,
        "p95_unselected_hole": 2.0,
        "max_uniform_similarity": 0.5,
        "boundary_support_r1": 0.75,
        "action_positive_coverage": 0.8,
        "dynamic_gain_calibration": {
            "score_semantics": "calibrated_marginal_gain",
            "calibration_scope": "cross_video_comparable",
            "target_source": "abs_signed_detector_utility",
        },
        "dynamic_budget_iqr": 1.0 if dynamic else 0.0,
        "dynamic_budget_entropy": 1.0 if dynamic else 0.0,
        "total_uniform_visible_fill_count": 0,
        "uses_uniform_fill": False,
        "uses_uniform_scaffold": False,
        "map_claim_allowed": False,
        "adatad_map": None,
        "end_to_end": False,
    }


def _ledger_summaries() -> list[dict]:
    out = []
    for split in ("train", "val", "test"):
        for strategy in (
            detector_policy.DETECTOR_AWARE_FIXED_384_STRATEGY,
            detector_policy.DETECTOR_AWARE_FIXED_768_STRATEGY,
            detector_policy.DETECTOR_AWARE_DYNAMIC_STRATEGY,
        ):
            summary = _ledger_summary(strategy)
            summary["split"] = split
            out.append(summary)
    return out


def test_stage4_curriculum_evidence_accepts_stage2_and_stage3_artifacts(tmp_path: Path) -> None:
    evidence = stage4.build_evidence(
        stage2_teacher_evidence=_stage2_teacher_evidence(tmp_path),
        stage2_policy_evidence=_stage2_policy_evidence(),
        stage2_ledger_validation_summaries=_ledger_summaries(),
        stage3_proof=_stage3_proof(),
    )

    validated = stage4.validate_evidence(evidence)

    assert validated["decision"] == "C3_STAGE4_CURRICULUM_EVIDENCE_PASS"
    assert validated["required_phases"] == stage4.REQUIRED_PHASES
    assert validated["claim_locks"]["paper_claim_allowed"] is False
    assert validated["curriculum_contract"]["bilevel_fulltrain_requires_sparse_detector_map"] is True
    assert validated["stage_status"]["stage4_curriculum_bilevel"] == "planned_gate_only_no_map_claim"


def test_stage4_curriculum_evidence_fails_closed_on_claim_or_policy_source(tmp_path: Path) -> None:
    evidence = stage4.build_evidence(
        stage2_teacher_evidence=_stage2_teacher_evidence(tmp_path),
        stage2_policy_evidence=_stage2_policy_evidence(),
        stage2_ledger_validation_summaries=_ledger_summaries(),
        stage3_proof=_stage3_proof(),
    )
    evidence["claim_locks"]["map_claim_allowed"] = True
    with pytest.raises(ValueError, match="map_claim_allowed"):
        stage4.validate_evidence(evidence)

    evidence = stage4.build_evidence(
        stage2_teacher_evidence=_stage2_teacher_evidence(tmp_path),
        stage2_policy_evidence={**_stage2_policy_evidence(), "policy_source": "bootstrap_detector_aware_surrogate_policy"},
        stage2_ledger_validation_summaries=_ledger_summaries(),
        stage3_proof=_stage3_proof(),
    )
    with pytest.raises(ValueError, match="learned_detector_aware_policy_checkpoint"):
        stage4.validate_evidence(evidence)


def test_stage4_curriculum_evidence_rejects_placeholders_and_bad_sha(tmp_path: Path) -> None:
    teacher = _stage2_teacher_evidence(tmp_path)
    teacher["teacher_checkpoint_path"] = "REPLACE_WITH_DENSE_TEACHER_CHECKPOINT"
    evidence = stage4.build_evidence(
        stage2_teacher_evidence=teacher,
        stage2_policy_evidence=_stage2_policy_evidence(),
        stage2_ledger_validation_summaries=_ledger_summaries(),
        stage3_proof=_stage3_proof(),
    )
    with pytest.raises(ValueError, match="placeholder"):
        stage4.validate_evidence(evidence)

    teacher = _stage2_teacher_evidence(tmp_path)
    teacher["teacher_checkpoint_sha256"] = "not-a-sha"
    evidence = stage4.build_evidence(
        stage2_teacher_evidence=teacher,
        stage2_policy_evidence=_stage2_policy_evidence(),
        stage2_ledger_validation_summaries=_ledger_summaries(),
        stage3_proof=_stage3_proof(),
    )
    with pytest.raises(ValueError, match="sha256"):
        stage4.validate_evidence(evidence)


def test_stage4_curriculum_evidence_requires_generator_manifest(tmp_path: Path) -> None:
    teacher = _stage2_teacher_evidence(tmp_path)
    teacher.pop("generator_manifest_json")
    evidence = stage4.build_evidence(
        stage2_teacher_evidence=teacher,
        stage2_policy_evidence=_stage2_policy_evidence(),
        stage2_ledger_validation_summaries=_ledger_summaries(),
        stage3_proof=_stage3_proof(),
    )
    with pytest.raises(ValueError, match="generator_manifest"):
        stage4.validate_evidence(evidence)


def test_stage4_curriculum_evidence_requires_actionformer_detector_grad(tmp_path: Path) -> None:
    proof = _stage3_proof()
    proof["actionformer_detector_loss_selector_grad_passed"] = False
    evidence = stage4.build_evidence(
        stage2_teacher_evidence=_stage2_teacher_evidence(tmp_path),
        stage2_policy_evidence=_stage2_policy_evidence(),
        stage2_ledger_validation_summaries=_ledger_summaries(),
        stage3_proof=proof,
    )

    with pytest.raises(ValueError, match="ActionFormer detector-loss proof"):
        stage4.validate_evidence(evidence)


def test_stage4_curriculum_evidence_requires_signed_calibrated_stage2_contract(tmp_path: Path) -> None:
    policy = _stage2_policy_evidence()
    policy.pop("dynamic_gain_calibration")
    evidence = stage4.build_evidence(
        stage2_teacher_evidence=_stage2_teacher_evidence(tmp_path),
        stage2_policy_evidence=policy,
        stage2_ledger_validation_summaries=_ledger_summaries(),
        stage3_proof=_stage3_proof(),
    )

    with pytest.raises(ValueError, match="dynamic_gain_calibration"):
        stage4.validate_evidence(evidence)

    ledger = _ledger_summary(detector_policy.DETECTOR_AWARE_DYNAMIC_STRATEGY)
    ledger.pop("dynamic_gain_calibration")
    evidence = stage4.build_evidence(
        stage2_teacher_evidence=_stage2_teacher_evidence(tmp_path),
        stage2_policy_evidence=_stage2_policy_evidence(),
        stage2_ledger_validation_summaries=[
            _ledger_summary(detector_policy.DETECTOR_AWARE_FIXED_384_STRATEGY),
            _ledger_summary(detector_policy.DETECTOR_AWARE_FIXED_768_STRATEGY),
            ledger,
        ],
        stage3_proof=_stage3_proof(),
    )

    with pytest.raises(ValueError, match="dynamic_gain_calibration"):
        stage4.validate_evidence(evidence)


def test_stage4_curriculum_evidence_requires_sparse_distill_fail_closed_proof(tmp_path: Path) -> None:
    proof = _stage3_proof()
    proof["sparse_distill_claim_allowed"] = True
    evidence = stage4.build_evidence(
        stage2_teacher_evidence=_stage2_teacher_evidence(tmp_path),
        stage2_policy_evidence=_stage2_policy_evidence(),
        stage2_ledger_validation_summaries=_ledger_summaries(),
        stage3_proof=proof,
    )

    with pytest.raises(ValueError, match="sparse distill claim"):
        stage4.validate_evidence(evidence)

    proof = _stage3_proof()
    proof.pop("sparse_distill_adapter_ready")
    evidence = stage4.build_evidence(
        stage2_teacher_evidence=_stage2_teacher_evidence(tmp_path),
        stage2_policy_evidence=_stage2_policy_evidence(),
        stage2_ledger_validation_summaries=_ledger_summaries(),
        stage3_proof=proof,
    )

    with pytest.raises(ValueError, match="sparse distill adapter"):
        stage4.validate_evidence(evidence)


def test_stage4_curriculum_evidence_rejects_missing_or_collapsed_dynamic_ledger(tmp_path: Path) -> None:
    evidence = stage4.build_evidence(
        stage2_teacher_evidence=_stage2_teacher_evidence(tmp_path),
        stage2_policy_evidence=_stage2_policy_evidence(),
        stage2_ledger_validation_summaries=[
            item for item in _ledger_summaries() if item["strategy"] != detector_policy.DETECTOR_AWARE_DYNAMIC_STRATEGY
        ],
        stage3_proof=_stage3_proof(),
    )
    with pytest.raises(ValueError, match="missing variants"):
        stage4.validate_evidence(evidence)

    evidence = stage4.build_evidence(
        stage2_teacher_evidence=_stage2_teacher_evidence(tmp_path),
        stage2_policy_evidence=_stage2_policy_evidence(),
        stage2_ledger_validation_summaries=[
            _ledger_summary(detector_policy.DETECTOR_AWARE_FIXED_384_STRATEGY),
            _ledger_summary(detector_policy.DETECTOR_AWARE_FIXED_768_STRATEGY),
            _ledger_summary(detector_policy.DETECTOR_AWARE_DYNAMIC_STRATEGY),
        ],
        stage3_proof=_stage3_proof(),
    )
    with pytest.raises(ValueError, match="split coverage"):
        stage4.validate_evidence(evidence)

    collapsed_ledgers = _ledger_summaries()
    for item in collapsed_ledgers:
        if item["strategy"] == detector_policy.DETECTOR_AWARE_DYNAMIC_STRATEGY:
            item["min_selected_count"] = 4
            item["max_selected_count"] = 4
            item["dynamic_budget_iqr"] = 0.0
            item["dynamic_budget_entropy"] = 0.0
    evidence = stage4.build_evidence(
        stage2_teacher_evidence=_stage2_teacher_evidence(tmp_path),
        stage2_policy_evidence=_stage2_policy_evidence(),
        stage2_ledger_validation_summaries=collapsed_ledgers,
        stage3_proof=_stage3_proof(),
    )
    with pytest.raises(ValueError, match="dynamic ledger selected_count collapsed"):
        stage4.validate_evidence(evidence)
