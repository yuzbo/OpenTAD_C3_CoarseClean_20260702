from __future__ import annotations

import copy
from pathlib import Path

from tools.bata.georoute_experiment_contract import canonical_sha256
from tools.bata.georoute_hybrid_causal_contract import (
    HYBRID_CAUSAL_ARM_ORDER,
    HYBRID_CAUSAL_ARM_SPECS,
    HYBRID_CAUSAL_CONTRACT_SCHEMA,
    HYBRID_CAUSAL_EPOCHS,
    HYBRID_CAUSAL_SEED,
    HYBRID_CAUSAL_STAGE_RESULT_SCHEMA,
    HYBRID_CAUSAL_STUDY_ID,
    finalize_hybrid_causal_study,
    hybrid_causal_arm_spec,
    validate_frozen_hybrid_causal_contract,
    validate_hybrid_causal_stage_result,
)


def _stage_result(arm: str, *, h: float, m7: float, p50: float):
    spec = hybrid_causal_arm_spec(arm)
    item_count = 220
    target_k = item_count if spec["route_mode"] == "dense" else 64
    role_counts = {
        "context": 0,
        "roi": 0,
        "residual": 0,
        "free": 0,
        "dense": 0,
        "uniform": 0,
        "random": 0,
    }
    if spec["route_mode"].startswith("structured_"):
        role_counts.update(
            context=spec["context_tokens"],
            roi=spec["roi_tokens"],
            residual=spec["residual_tokens"],
        )
    else:
        role_counts[spec["route_mode"]] = target_k
    binding = {
        "schema_version": HYBRID_CAUSAL_CONTRACT_SCHEMA,
        "world_size": 2,
        "local_batch": 1,
        "fp16_compress": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    metrics = {
        "average_mAP": h + 2.0,
        "mAP@0.3": h + 5.0,
        "mAP@0.4": h + 4.0,
        "mAP@0.5": h + 3.0,
        "mAP@0.6": 2.0 * h - m7,
        "mAP@0.7": m7,
        "high_iou_composite": h,
    }
    result = {
        "schema_version": HYBRID_CAUSAL_STAGE_RESULT_SCHEMA,
        "status": "PASS_EXPLORATORY_DEVELOPMENT_ONLY",
        "study_id": HYBRID_CAUSAL_STUDY_ID,
        "arm": arm,
        "arm_spec": spec,
        "arm_spec_sha256": canonical_sha256(spec),
        "seed": HYBRID_CAUSAL_SEED,
        "epochs": HYBRID_CAUSAL_EPOCHS,
        "population_sha256": "p" * 64,
        "metrics": metrics,
        "profile": {
            "model_and_postprocess_p50_ms": p50,
            "scope": {
                "diagnostic_route_telemetry_inside_timed_forward": False,
                "separate_from_accuracy_evaluation": True,
            },
        },
        "telemetry_summary": {
            "population_sha256": "p" * 64,
            "official_test_opened": False,
            "paper_claim_allowed": False,
            "role_counts": role_counts,
            "target_k": target_k,
        },
        "binding": binding,
        "binding_sha256": binding["binding_sha256"],
        "routing_audit": {
            "item_count": item_count,
            "target_k": target_k,
            "selected_unique_count_min": target_k,
            "selected_unique_count_max": target_k,
            "selected_duplicate_count": 0,
            "heavy_backbone_forward_count": 1,
            "absolute_position_enabled": True,
            "absolute_coordinates_enabled": False,
            "roi_relative_coordinates_enabled": False,
            "geometry_projection_enabled": False,
            "pooling_mode": "uniform_selected",
            "route_mode": spec["route_mode"],
            "policy_estimator": spec["policy_estimator"],
            "role_counts": role_counts,
            "diagnostic_telemetry_enabled": False,
            "routing_schema": (
                "georoute_fixed_quota_structured_routing_v1"
                if spec["route_mode"].startswith("structured_")
                else "georoute_native_routing_v2"
            ),
            "route_rng": (
                {
                    "schema_version": "georoute_route_private_rng_v1",
                    "global_rng_consumed": False,
                }
                if spec["route_mode"].startswith("structured_")
                else {}
            ),
            "geometry_temporal_shift_tubelets": spec[
                "geometry_temporal_shift_tubelets"
            ],
            "uses_gt_for_route": False,
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_test_evidence": False,
        },
        "runtime_commit": "c" * 40,
        "official_test_opened": False,
        "partial_survivor_inference_allowed": False,
        "paper_claim_allowed": False,
    }
    result["stage_result_sha256"] = canonical_sha256(result)
    return result


def _passing_matrix():
    values = {
        "dense_native": (10.3, 8.8, 12.0),
        "fixed_lattice_k64": (9.2, 7.8, 8.0),
        "random_lattice_k64": (9.0, 7.6, 8.1),
        "residual_pl_k64_support_only": (9.4, 7.9, 8.3),
        "context8_residual56_pl_support_only": (9.7, 8.1, 8.4),
        "context8_roi56_pl_support_only": (9.8, 8.2, 8.5),
        "hybrid_ctx8_roi28_res28_st_support_only": (9.9, 8.25, 8.6),
        "hybrid_ctx8_roi28_res28_pl_support_only": (10.1, 8.5, 8.7),
        "hybrid_ctx8_roi28_res28_pl_geometry_shift127": (9.6, 8.0, 8.7),
    }
    return {
        arm: _stage_result(arm, h=h, m7=m7, p50=p50)
        for arm, (h, m7, p50) in values.items()
    }


def test_frozen_nine_arm_contract_has_one_support_intervention_per_arm():
    validate_frozen_hybrid_causal_contract()
    assert HYBRID_CAUSAL_ARM_ORDER == tuple(HYBRID_CAUSAL_ARM_SPECS)
    assert len(HYBRID_CAUSAL_ARM_ORDER) == 9
    assert [HYBRID_CAUSAL_ARM_SPECS[arm]["arm_id"] for arm in HYBRID_CAUSAL_ARM_ORDER] == [
        f"A{index}" for index in range(9)
    ]
    for spec in HYBRID_CAUSAL_ARM_SPECS.values():
        assert spec["absolute_position_enabled"] is True
        assert spec["absolute_coordinates_enabled"] is False
        assert spec["roi_relative_coordinates_enabled"] is False
        assert spec["geometry_projection_enabled"] is False
        assert spec["pooling_mode"] == "uniform_selected"


def test_stage_validator_rejects_unknown_or_duplicate_route_support():
    valid = _stage_result("hybrid_ctx8_roi28_res28_pl_support_only", h=10.1, m7=8.5, p50=8.7)
    validate_hybrid_causal_stage_result(
        valid,
        expected_arm="hybrid_ctx8_roi28_res28_pl_support_only",
        expected_commit="c" * 40,
    )
    broken = copy.deepcopy(valid)
    broken["routing_audit"]["selected_duplicate_count"] = 1
    broken.pop("stage_result_sha256")
    broken["stage_result_sha256"] = canonical_sha256(broken)
    try:
        validate_hybrid_causal_stage_result(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate structured support must fail closed")


def test_incomplete_finalizer_emits_no_contrast_or_performance_inference():
    matrix = _passing_matrix()
    matrix.pop("context8_roi56_pl_support_only")
    finalization = finalize_hybrid_causal_study(
        matrix,
        expected_commit="c" * 40,
    )
    assert finalization["decision"] == "INCOMPLETE_NO_PERFORMANCE_INFERENCE"
    assert finalization["descriptive_contrasts"] == {}
    assert finalization["paper_claim_allowed"] is False


def test_all_nine_passing_screen_only_admits_a_new_confirmatory_freeze():
    finalization = finalize_hybrid_causal_study(
        _passing_matrix(),
        expected_commit="c" * 40,
    )
    assert finalization["decision"] == "ADMIT_SEPARATELY_FROZEN_CONFIRMATORY_STUDY"
    assert all(finalization["screen_admission_checks"].values())
    assert finalization["multiple_comparison_adjusted_claim_allowed"] is False
    assert finalization["official_test_opened"] is False


def test_geometry_shift_tie_is_mechanism_ambiguous_not_a_hybrid_win():
    matrix = _passing_matrix()
    main = matrix["hybrid_ctx8_roi28_res28_pl_support_only"]
    shifted = matrix["hybrid_ctx8_roi28_res28_pl_geometry_shift127"]
    shifted["metrics"]["high_iou_composite"] = main["metrics"]["high_iou_composite"]
    shifted["metrics"]["mAP@0.6"] = main["metrics"]["mAP@0.6"]
    shifted["metrics"]["mAP@0.7"] = main["metrics"]["mAP@0.7"]
    shifted.pop("stage_result_sha256")
    shifted["stage_result_sha256"] = canonical_sha256(shifted)
    finalization = finalize_hybrid_causal_study(
        matrix,
        expected_commit="c" * 40,
    )
    assert finalization["decision"] == "HOLD_MECHANISM_AMBIGUOUS"
    assert finalization["screen_admission_checks"]["aligned_gt_geometry_shift"] is False


def test_pilot_base_explicitly_disables_all_representation_and_fp16_compression():
    source = (
        Path(__file__).parents[1]
        / "configs"
        / "adatad"
        / "thumos"
        / "georoute_hybrid_causal_pilot_base.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "georoute_absolute_position_enabled=True",
        "georoute_absolute_coordinates_enabled=False",
        "georoute_roi_relative_coordinates_enabled=False",
        "georoute_geometry_projection_enabled=False",
        'georoute_pooling_mode="uniform_selected"',
        "georoute_policy_temperature=0.7",
        'georoute_score_function_temporal_reduction="mean"',
        "fp16_compress=False",
        "batch_size=2",
        "official_test_open_allowed=False",
    ):
        assert literal in source
