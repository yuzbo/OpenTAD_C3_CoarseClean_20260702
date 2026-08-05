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
from tools.bata.run_georoute_phase_m_replay import (
    _build_replay_test_arguments,
    _configure_replay_instrumentation,
    _validate_replay_telemetry_header,
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
                    "roles": {
                        "aggregate_counts": calibration["selected_role_counts"]
                    },
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
        custom.georoute_role_calibration_telemetry_enabled
        is role_calibration_enabled
    )
    assert cfg.georoute_diagnostic_telemetry.enabled is True
    assert (
        cfg.georoute_development_profile.enabled
        is (not role_calibration_enabled)
    )
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
    expected_tail = (
        ["--id", "0"]
        if role_calibration_enabled
        else ["0", "--not_eval"]
    )
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
        ("selected_role_fractions", {"context": 0.5, "roi": 0.0, "residual": 0.5}, "disagrees"),
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
