from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from mmengine.config import Config

from opentad.models.backbones.georoute_routing import (
    select_dynamic_global_exact_budget,
)
from opentad.models.backbones.georoute_wrapper import GeoRouteBackboneWrapper
from tools.bata.analyze_georoute_dynamic_role_calibration import (
    summarize_dynamic_role_calibration_telemetry,
)
from tools.bata.analyze_georoute_categorical_role_invariance import (
    summarize_categorical_role_invariance_payloads,
)
from tools.bata.run_georoute_phase_m_replay import (
    _build_replay_test_arguments,
    _configure_replay_instrumentation,
    _validate_replay_telemetry_header,
)
from tools.bata.run_georoute_role_instrumentation_pair import (
    _build_pair_test_arguments,
    _configure_pair_mode,
    _validate_formal_telemetry,
    compare_prediction_artifacts,
)
from tools.bata.run_georoute_role_instrumentation_triplet import (
    MODE_SPECIFICATIONS,
    classify_triplet_comparisons,
)
from tools.bata.run_georoute_residual_centering_probe import (
    _build_probe_test_arguments,
    _configure_probe_mode,
    classify_residual_centering_role_gate,
    summarize_residual_centering_branch_payload,
)


def _telemetry_payload() -> dict:
    q_base = torch.tensor([[[4.0, 3.0], [2.0, 1.0]]])
    delta_roi = torch.full_like(q_base, -1.0)
    delta_residual = torch.full_like(q_base, 2.0)
    valid = torch.ones_like(q_base, dtype=torch.bool)
    route = select_dynamic_global_exact_budget(
        q_base=q_base,
        delta_roi=delta_roi,
        delta_residual=delta_residual,
        window_budget=2,
        training=False,
        estimator="none",
        temperature=0.5,
        valid_mask=valid,
    )
    calibration = GeoRouteBackboneWrapper._dynamic_policy_calibration_telemetry(
        route=route,
        q_base=q_base,
        delta_roi=delta_roi,
        delta_residual=delta_residual,
        valid_patch_mask=valid,
    )
    return {
        "schema_version": "georoute_diagnostic_telemetry_v1",
        "development_only": True,
        "official_test_opened": False,
        "gt_for_route_used": False,
        "teacher_for_route_used": False,
        "oracle_used": False,
        "raw_prediction_cache_used": False,
        "dataset_count": 1,
        "record_count": 1,
        "unique_dataset_count": 1,
        "sampler_padding_count": 0,
        "population_sha256": "a" * 64,
        "records": [
            {
                "dataset_index": 0,
                "video_id": "development-only",
                "route": {
                    "schema_version": (
                        "georoute_dynamic_diagnostic_window_telemetry_v1"
                    ),
                    "measurement_scope": (
                        "accuracy_replay_only_excluded_from_timed_cost"
                    ),
                    "roles": {"aggregate_counts": calibration["selected_role_counts"]},
                    "policy_calibration": calibration,
                },
            }
        ],
    }


@pytest.mark.parametrize("role_calibration_enabled", [False, True])
def test_phase_m_replay_instrumentation_preserves_route_configuration(
    tmp_path: Path,
    role_calibration_enabled: bool,
):
    cfg = Config(
        dict(
            model=dict(
                backbone=dict(
                    custom=dict(
                        georoute_route_mode="dynamic_scnr",
                        georoute_window_token_budget=24576,
                        georoute_diagnostic_telemetry_enabled=False,
                    )
                )
            ),
            georoute_diagnostic_telemetry=dict(enabled=False),
            georoute_development_profile=dict(enabled=False),
            post_processing=dict(save_dict=False),
            inference=dict(
                load_from_raw_predictions=False,
                save_raw_prediction=False,
            ),
        )
    )

    _configure_replay_instrumentation(
        cfg,
        replay_work=tmp_path / "replay",
        role_calibration_telemetry_enabled=role_calibration_enabled,
    )

    custom = cfg.model.backbone.custom
    assert cfg.work_dir == str(tmp_path / "replay" / "gpu1_id0")
    assert custom.georoute_route_mode == "dynamic_scnr"
    assert custom.georoute_window_token_budget == 24576
    assert custom.georoute_diagnostic_telemetry_enabled is True
    assert (
        custom.georoute_role_calibration_telemetry_enabled is role_calibration_enabled
    )
    assert cfg.georoute_diagnostic_telemetry.enabled is True
    assert cfg.georoute_development_profile.enabled is (not role_calibration_enabled)
    assert cfg.post_processing.save_dict is True
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False


@pytest.mark.parametrize(
    ("role_calibration_enabled", "expects_no_eval"),
    [(False, True), (True, False)],
)
def test_phase_m_role_replay_preserves_frozen_m2_evaluation_contract(
    role_calibration_enabled: bool,
    expects_no_eval: bool,
):
    arguments = _build_replay_test_arguments(
        command_prefix=["python", "-m", "torch.distributed.run"],
        bound_config=Path("bound.py"),
        checkpoint=Path("epoch_59.pth"),
        seed=3407,
        role_calibration_telemetry_enabled=role_calibration_enabled,
    )

    assert ("--not_eval" in arguments) is expects_no_eval
    expected_tail = ["--id", "0"] if role_calibration_enabled else ["0", "--not_eval"]
    assert arguments[-2:] == expected_tail


@pytest.mark.parametrize(
    ("schema", "role_calibration_enabled"),
    [
        ("georoute_diagnostic_telemetry_v1", False),
        ("georoute_formal_development_telemetry_v1", True),
    ],
)
def test_phase_m_replay_requires_mode_matched_no_leak_telemetry_schema(
    schema: str,
    role_calibration_enabled: bool,
):
    payload = _telemetry_payload()
    payload["schema_version"] = schema

    _validate_replay_telemetry_header(
        payload,
        role_calibration_telemetry_enabled=role_calibration_enabled,
    )
    with pytest.raises(RuntimeError, match="no-leak schema"):
        _validate_replay_telemetry_header(
            payload,
            role_calibration_telemetry_enabled=not role_calibration_enabled,
        )


@pytest.mark.parametrize("role_calibration_enabled", [False, True])
def test_role_instrumentation_pair_has_identical_accuracy_path(
    tmp_path: Path,
    role_calibration_enabled: bool,
):
    cfg = Config(
        dict(
            model=dict(
                backbone=dict(
                    custom=dict(
                        georoute_route_mode="dynamic_scnr",
                        georoute_window_token_budget=24576,
                        georoute_diagnostic_telemetry_enabled=True,
                    )
                )
            ),
            georoute_diagnostic_telemetry=dict(enabled=True),
            georoute_development_profile=dict(enabled=False),
            post_processing=dict(save_dict=True),
            inference=dict(
                load_from_raw_predictions=False,
                save_raw_prediction=False,
            ),
        )
    )
    binding = {"schema_version": "pair", "pair_mode": "test"}
    _configure_pair_mode(
        cfg,
        work_dir=tmp_path / "gpu1_id0",
        role_calibration_enabled=role_calibration_enabled,
        binding=binding,
    )
    arguments = _build_pair_test_arguments(
        command_prefix=["python", "-m", "torch.distributed.run"],
        bound_config=Path("bound.py"),
        checkpoint=Path("epoch_59.pth"),
        seed=3407,
    )

    assert "--not_eval" not in arguments
    assert cfg.georoute_development_profile.enabled is False
    assert (
        cfg.model.backbone.custom.georoute_role_calibration_telemetry_enabled
        is role_calibration_enabled
    )
    assert dict(cfg.georoute_phase_m_binding) == binding
    assert arguments[-2:] == ["--id", "0"]


def test_residual_centering_probe_is_opt_in_and_forces_no_eval(tmp_path: Path):
    cfg = Config(
        dict(
            model=dict(
                backbone=dict(
                    custom=dict(
                        georoute_route_mode="dynamic_scnr",
                        georoute_branch_calibration_mode="none",
                        georoute_diagnostic_telemetry_enabled=True,
                    )
                )
            ),
            georoute_diagnostic_telemetry=dict(enabled=True),
            georoute_development_profile=dict(enabled=False),
            post_processing=dict(save_dict=True),
            inference=dict(
                load_from_raw_predictions=False,
                save_raw_prediction=False,
            ),
        )
    )
    binding = {"schema_version": "probe", "probe_mode": "centered_a"}

    _configure_probe_mode(
        cfg,
        work_dir=tmp_path / "gpu1_id0",
        binding=binding,
    )
    arguments = _build_probe_test_arguments(
        command_prefix=["python", "-m", "torch.distributed.run"],
        bound_config=Path("bound.py"),
        checkpoint=Path("epoch_59.pth"),
        seed=3407,
    )

    custom = cfg.model.backbone.custom
    assert custom.georoute_branch_calibration_mode == "residual_window_center"
    assert custom.georoute_role_calibration_telemetry_enabled is True
    assert cfg.georoute_development_profile.enabled is False
    assert dict(cfg.georoute_phase_m_binding) == binding
    assert arguments[-3:] == ["--id", "0", "--not_eval"]
    assert arguments.count("--not_eval") == 1


def _residual_centering_payload() -> dict:
    payload = _telemetry_payload()
    route = payload["records"][0]["route"]
    route.update(tubelet_count=2, item_count=2)
    route["branch_calibration"] = {
        "schema_version": "scnr_dynamic_branch_calibration_window_v1",
        "mode": "residual_window_center",
        "target": "delta_residual",
        "scope": "complete_window_all_valid_candidates",
        "valid_candidate_count": 4,
        "residual_valid_mean_before": 2.0,
        "residual_valid_mean_after": 1e-7,
        "changes_q_base": False,
        "changes_delta_roi": False,
        "changes_context_zero_modifier": False,
        "changes_budget_or_role_quota": False,
        "mean_detached": False,
    }
    return payload


def test_residual_centering_probe_validates_transform_receipt():
    payload = _residual_centering_payload()
    summary = summarize_residual_centering_branch_payload(payload)

    assert summary["transform_receipts_valid"] is True
    assert summary["residual_valid_mean_after_max_abs"] == pytest.approx(1e-7)
    payload["records"][0]["route"]["branch_calibration"][
        "residual_valid_mean_after"
    ] = 1e-2
    with pytest.raises(ValueError, match="branch receipt is invalid"):
        summarize_residual_centering_branch_payload(payload)


@pytest.mark.parametrize(
    ("valid_counts", "selected_counts", "passed"),
    [
        (
            {"context": 1, "roi": 2, "residual": 7},
            {"context": 0, "roi": 1, "residual": 4},
            True,
        ),
        (
            {"context": 0, "roi": 2, "residual": 8},
            {"context": 0, "roi": 1, "residual": 4},
            False,
        ),
        (
            {"context": 1, "roi": 2, "residual": 7},
            {"context": 0, "roi": 0, "residual": 5},
            False,
        ),
    ],
)
def test_residual_centering_probe_applies_fail_closed_structural_gate(
    valid_counts: dict,
    selected_counts: dict,
    passed: bool,
):
    summary = {
        "roles": {
            "valid": {"counts": valid_counts},
            "selected": {"counts": selected_counts},
        }
    }

    gate = classify_residual_centering_role_gate(summary)

    assert gate["passed"] is passed
    assert gate["performance_claim_allowed"] is False
    assert (gate["status"].startswith("PASS_")) is passed


@pytest.mark.parametrize("role_calibration_enabled", [False, True])
def test_role_instrumentation_pair_requires_formal_complete_population(
    role_calibration_enabled: bool,
):
    binding = {"schema_version": "pair", "pair_mode": "test"}
    payload = _telemetry_payload()
    payload["schema_version"] = "georoute_formal_development_telemetry_v1"
    payload["phase_m_binding"] = binding
    payload["records"][0]["route"].pop("policy_calibration", None)
    if role_calibration_enabled:
        payload["records"][0]["route"]["policy_calibration"] = {}

    receipt = _validate_formal_telemetry(
        payload,
        expected_binding=binding,
        expected_population_sha256="a" * 64,
        expected_dataset_count=1,
        role_calibration_enabled=role_calibration_enabled,
    )

    assert receipt["role_calibration_records_present"] is role_calibration_enabled
    payload["sampler_padding_count"] = 1
    with pytest.raises(RuntimeError, match="population parity"):
        _validate_formal_telemetry(
            payload,
            expected_binding=binding,
            expected_population_sha256="a" * 64,
            expected_dataset_count=1,
            role_calibration_enabled=role_calibration_enabled,
        )


def test_role_instrumentation_pair_compares_predictions_without_metrics(
    tmp_path: Path,
):
    left = tmp_path / "left.json"
    equal = tmp_path / "equal.json"
    changed = tmp_path / "changed.json"
    payload = {
        "results": {"video": [{"segment": [1.0, 2.0], "label": "action", "score": 0.5}]}
    }
    left.write_text(json.dumps(payload), encoding="utf-8")
    equal.write_text(json.dumps(payload), encoding="utf-8")
    changed_payload = copy.deepcopy(payload)
    changed_payload["results"]["video"][0]["score"] = 0.4
    changed.write_text(json.dumps(changed_payload), encoding="utf-8")

    exact = compare_prediction_artifacts(left, equal)
    drift = compare_prediction_artifacts(left, changed)

    assert exact["raw_sha256_parity"] is True
    assert exact["exact_candidate_identity_overlap"] == 1
    assert drift["raw_sha256_parity"] is False
    assert drift["json_semantic_parity"] is False
    assert drift["exact_candidate_identity_overlap"] == 1


def _triplet_comparisons(
    *,
    control_parity: bool,
    treatment_parity: bool,
    source_parity: bool,
) -> dict:
    def comparison(parity: bool) -> dict:
        return {"raw_sha256_parity": parity}

    return {
        "source_vs_role_off_a": comparison(source_parity),
        "source_vs_role_off_b": comparison(source_parity),
        "source_vs_role_on": comparison(source_parity),
        "role_off_a_vs_role_off_b": comparison(control_parity),
        "role_off_a_vs_role_on": comparison(treatment_parity),
        "role_off_b_vs_role_on": comparison(treatment_parity),
    }


@pytest.mark.parametrize(
    (
        "control_parity",
        "treatment_parity",
        "source_parity",
        "expected_status",
        "analysis_allowed",
    ),
    [
        (
            False,
            False,
            False,
            "FAIL_STRICT_BASELINE_REPLAY_NONDETERMINISM",
            False,
        ),
        (
            True,
            False,
            False,
            "FAIL_STRICT_ROLE_INSTRUMENTATION_NONNEUTRAL",
            False,
        ),
        (
            True,
            True,
            False,
            "PASS_STRICT_TRIPLET_NEUTRALITY_SOURCE_REPLAY_DRIFT_DIAGNOSTIC_ONLY",
            False,
        ),
        (
            True,
            True,
            True,
            "PASS_STRICT_TRIPLET_NEUTRALITY_AND_SOURCE_RAW_PARITY_DIAGNOSTIC_ONLY",
            False,
        ),
    ],
)
def test_role_instrumentation_triplet_has_fail_closed_causal_verdict(
    control_parity: bool,
    treatment_parity: bool,
    source_parity: bool,
    expected_status: str,
    analysis_allowed: bool,
):
    verdict = classify_triplet_comparisons(
        _triplet_comparisons(
            control_parity=control_parity,
            treatment_parity=treatment_parity,
            source_parity=source_parity,
        )
    )

    assert verdict["status"] == expected_status
    assert (
        verdict["role_calibration_analysis_allowed_under_frozen_contract"]
        is analysis_allowed
    )


def test_role_instrumentation_triplet_repeats_true_off_before_treatment():
    assert MODE_SPECIFICATIONS == (
        ("role_off_a", False, "role_off"),
        ("role_off_b", False, "role_off"),
        ("role_on", True, "role_on"),
    )


def _categorical_invariance_payloads() -> dict:
    legacy_on = _telemetry_payload()
    legacy_on["schema_version"] = "georoute_formal_development_telemetry_v1"
    legacy_on["records"][0]["route"]["geometry"] = {"values": [1.0]}
    strict_on = copy.deepcopy(legacy_on)
    strict_on["records"][0]["route"]["geometry"] = {"values": [2.0]}
    strict_on["records"][0]["route"]["policy_calibration"]["fields"] = {
        "strict_backend_continuous_fields": True
    }
    payloads = {
        "legacy_role_on": legacy_on,
        "strict_role_on": strict_on,
    }
    for prefix in ("legacy", "strict"):
        for suffix in ("role_off_a", "role_off_b"):
            payload = copy.deepcopy(payloads[f"{prefix}_role_on"])
            payload["records"][0]["route"].pop("policy_calibration")
            payloads[f"{prefix}_{suffix}"] = payload
    return payloads


def test_categorical_role_invariance_excludes_backend_sensitive_fields():
    summary = summarize_categorical_role_invariance_payloads(
        _categorical_invariance_payloads()
    )

    assert summary["roles"]["selected"]["counts"] == {
        "context": 0,
        "roi": 0,
        "residual": 2,
    }
    assert summary["categorical_invariance"]["geometry_payload_parity"] is False
    assert summary["categorical_invariance"]["continuous_policy_fields_parity"] is False
    assert (
        summary["interpretation_boundary"]["categorical_role_analysis_allowed"] is True
    )
    assert (
        summary["interpretation_boundary"][
            "continuous_score_calibration_analysis_allowed"
        ]
        is False
    )


def test_categorical_role_invariance_rejects_a_role_change():
    payloads = _categorical_invariance_payloads()
    strict = payloads["strict_role_on"]["records"][0]["route"]
    strict["roles"]["aggregate_counts"] = {
        "context": 1,
        "roi": 0,
        "residual": 1,
    }
    strict["policy_calibration"]["selected_role_counts"] = {
        "context": 1,
        "roi": 0,
        "residual": 1,
    }
    strict["policy_calibration"]["selected_role_fractions"] = {
        "context": 0.5,
        "roi": 0.0,
        "residual": 0.5,
    }

    with pytest.raises(ValueError, match="categorical route payload changed"):
        summarize_categorical_role_invariance_payloads(payloads)


def test_role_calibration_summary_detects_observed_collapse_without_quota(
    tmp_path: Path,
):
    path = tmp_path / "telemetry.json"
    path.write_text(
        json.dumps(_telemetry_payload(), sort_keys=True),
        encoding="utf-8",
    )

    summary = summarize_dynamic_role_calibration_telemetry(path)

    assert summary["roles"]["valid"]["counts"] == {
        "context": 0,
        "roi": 0,
        "residual": 4,
    }
    assert summary["roles"]["selected"]["counts"] == {
        "context": 0,
        "roi": 0,
        "residual": 2,
    }
    assert summary["roles"]["unselected"]["counts"] == {
        "context": 0,
        "roi": 0,
        "residual": 2,
    }
    assert summary["roles"]["windows_missing_role"] == {
        "context": 1,
        "roi": 1,
        "residual": 0,
    }
    assert summary["roles"]["dominant_window_counts"] == {"residual": 1}
    assert summary["roles"]["role_balance_enforced"] is False
    residual = summary["fields"]["delta_residual"]["valid"]
    assert residual["weighted_mean"] == pytest.approx(2.0)
    assert residual["positive_count"] == 4
    assert summary["interpretation_boundary"]["floor_selection_allowed"] is False


def test_role_calibration_summary_rejects_a_quota_claim(tmp_path: Path):
    payload = copy.deepcopy(_telemetry_payload())
    payload["records"][0]["route"]["policy_calibration"][
        "role_target_fractions_used"
    ] = True
    path = tmp_path / "telemetry.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="window schema changed"):
        summarize_dynamic_role_calibration_telemetry(path)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "selected_role_fractions",
            {"context": 0.5, "roi": 0.0, "residual": 0.5},
            "disagrees",
        ),
        (
            "selected_over_valid_role_fraction_ratio",
            {"context": None, "roi": None, "residual": 0.5},
            "inconsistent",
        ),
    ],
)
def test_role_calibration_summary_rejects_inconsistent_derived_statistics(
    tmp_path: Path,
    field: str,
    replacement: dict,
    message: str,
):
    payload = copy.deepcopy(_telemetry_payload())
    payload["records"][0]["route"]["policy_calibration"][field] = replacement
    path = tmp_path / "telemetry.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        summarize_dynamic_role_calibration_telemetry(path)


def test_role_calibration_summary_rejects_coerced_dataset_index(tmp_path: Path):
    payload = copy.deepcopy(_telemetry_payload())
    payload["records"][0]["dataset_index"] = "0"
    path = tmp_path / "telemetry.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset indices"):
        summarize_dynamic_role_calibration_telemetry(path)
