import json
import hashlib

import pytest

import tools.bata.finalize_duca_h65_singleclock_terminal as finalizer_module
from tools.bata.finalize_duca_h65_singleclock_terminal import (
    _identity_equal,
    _twin_execution_contract_ok,
    finalize,
)
from tools.bata.duca_p0_evaluation import (
    canonical_sha256,
    official_evaluator_identity,
)


def _bootstrap(
    names,
    points,
    *,
    metrics_by_family=None,
    baseline_family=None,
    namespace=None,
    nonce="test-singleclock-bootstrap",
):
    sampled = {}
    point_estimates = {}
    for name in names:
        sampled[name] = {}
        point_estimates[name] = {}
        for metric, value in points[name].items():
            sampled[name][metric] = [value] * 10000
            point_estimates[name][metric] = value
    payload = {
        "samples": 10000,
        "lower_rank": 250,
        "upper_rank": 9750,
        "sampled_metrics": sampled,
        "point_estimates": point_estimates,
    }
    if metrics_by_family is not None:
        evaluations = {
            family: json.loads(path.read_text(encoding="utf-8"))
            for family, path in metrics_by_family.items()
        }
        first = evaluations[names[0]]
        payload.update(
            {
                "schema_version": "duca_h65_official_pcg64_video_bootstrap_v1",
                "official_evaluator_reexecuted_per_resample": True,
                "paired_video_cluster_bootstrap": True,
                "rng": "numpy.random.PCG64",
                "nonce": nonce,
                "namespace": namespace,
                "interval_rank_convention": "one_based_order_statistics",
                "baseline_family": baseline_family,
                "family_order": list(names),
                "prediction_paths": {
                    family: evaluations[family]["prediction_path"]
                    for family in names
                },
                "prediction_sha256": {
                    family: evaluations[family]["prediction_sha256"]
                    for family in names
                },
                "evaluation_config": first["evaluation_config"],
                "evaluation_config_sha256": first[
                    "evaluation_config_sha256"
                ],
                "evaluator": first["evaluator"],
            }
        )
    return payload


def _identity(path):
    path.write_text(
        json.dumps(
            {
                "schema_version": "duca_h65_single_clock_selected_input_identity_v2",
                "sample_count": 1,
                "total_input_exposure_count": 1,
                "unique_physical_window_count": 1,
                "duplicate_exposure_count": 0,
                "duplicate_samples": [],
                "records": [
                    {
                        "sample_id": "v|window_start_frame=0",
                        "video_name": "v",
                        "window_start_frame": 0,
                        "selected_valid_len": 384,
                        "dense_valid_len": 768,
                        "selected_positions": list(range(384)),
                        "selected_rgb_sha256": "r",
                        "videomae_input_sha256": "v",
                        "selected_positions_sha256": "p",
                        "selected_mask_sha256": "m",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(path, checkpoint, state_key, values):
    annotation = path.parent / "annotation.json"
    class_map = path.parent / "class_map.txt"
    prediction = path.with_name(f"{path.stem}_predictions.json")
    annotation.write_text(
        json.dumps(
            {
                "database": {
                    "v": {"subset": "validation", "annotations": []}
                }
            }
        ),
        encoding="utf-8",
    )
    class_map.write_text("0 action\n", encoding="utf-8")
    prediction.write_text(
        json.dumps(
            {
                "results": {
                    "v": [
                        {"segment": [0.0, 1.0], "label": "action", "score": 1.0}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    evaluation_config = {
        "type": "mAP",
        "ground_truth_filename": str(annotation.resolve()),
        "subset": "validation",
        "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        "top_k": None,
        "blocked_videos": None,
        "thread": 16,
    }
    payload = {
        "schema_version": "duca_protected_physical_terminal_evaluation_v1",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_epoch": 59,
        "checkpoint_state_key": state_key,
        "prediction_path": str(prediction.resolve()),
        "prediction_sha256": _sha256(prediction),
        "metrics": values,
        "result_count": 1,
        "video_count": 1,
        "evaluation_annotation_path": str(annotation.resolve()),
        "evaluation_annotation_sha256": _sha256(annotation),
        "evaluation_class_map_path": str(class_map.resolve()),
        "evaluation_class_map_sha256": _sha256(class_map),
        "evaluation_config": evaluation_config,
        "evaluation_config_sha256": canonical_sha256(evaluation_config),
        "evaluator": official_evaluator_identity(),
    }
    payload["evaluation_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _configs(tmp_path):
    root = tmp_path / "configs" / "adatad" / "thumos"
    root.mkdir(parents=True)
    on = root / "duca_h65_first_singleclock_cycle4.py"
    zero = root / "duca_h65_first_singleclock_cycle4_gate_zero.py"
    off = root / "duca_sampling_rate_curriculum_stage2_joint384.py"
    on.write_text("single_clock_gate_zero = False\n", encoding="utf-8")
    zero.write_text("single_clock_gate_zero = True\n", encoding="utf-8")
    off.write_text("single_clock = False\n", encoding="utf-8")
    return on, zero, off


def _config_row(path, gate_zero):
    return {
        "config_path": str(path),
        "config_sha256": _sha256(path),
        "single_clock_gate_zero": gate_zero,
    }


def _family_row(metrics, config, gate_zero, identity=None):
    row = {
        "metrics_path": str(metrics),
        "metrics_sha256": _sha256(metrics),
        **_config_row(config, gate_zero),
    }
    if identity is not None:
        row["selected_input_identity_path"] = str(identity)
        row["selected_input_identity_sha256"] = _sha256(identity)
    return row


def test_twin_execution_contract_rejects_a_mislabeled_gate_zero(tmp_path):
    on, zero, _ = _configs(tmp_path)
    assert _twin_execution_contract_ok(
        _config_row(on, False), _config_row(zero, True)
    )
    assert not _twin_execution_contract_ok(
        _config_row(on, False), _config_row(zero, False)
    )


def _finalizer_fixture(tmp_path):
    on_config, zero_config, off_config = _configs(tmp_path)
    clock_checkpoint = tmp_path / "clock_epoch_59.pth"
    off_checkpoint = tmp_path / "off_epoch_59.pth"
    stage1_checkpoint = tmp_path / "stage1_epoch_29.pth"
    clock_checkpoint.write_bytes(b"clock")
    off_checkpoint.write_bytes(b"off")
    stage1_checkpoint.write_bytes(b"stage1")
    final_on = tmp_path / "final_on.json"
    final_zero = tmp_path / "final_zero.json"
    ema_on = tmp_path / "ema_on.json"
    ema_zero = tmp_path / "ema_zero.json"
    for path in (final_on, final_zero, ema_on, ema_zero):
        _identity(path)
    metrics = ("average_mAP", "mAP@0.6", "mAP@0.7")
    final_points = {
        "final_on": dict.fromkeys(metrics, 0.66),
        "final_gate_zero": dict.fromkeys(metrics, 0.65),
        "h65_off_final": dict.fromkeys(metrics, 0.65),
    }
    ema_points = {
        "ema_on": dict.fromkeys(metrics, 0.658),
        "ema_gate_zero": dict.fromkeys(metrics, 0.68),
        "h65_off_ema": dict.fromkeys(metrics, 0.66),
    }
    old_points = {
        "truetime": dict.fromkeys(metrics, 0.62),
        "rankpack": dict.fromkeys(metrics, 0.61),
    }
    final_on_metrics = tmp_path / "final_on_metrics.json"
    final_zero_metrics = tmp_path / "final_zero_metrics.json"
    ema_on_metrics = tmp_path / "ema_on_metrics.json"
    ema_zero_metrics = tmp_path / "ema_zero_metrics.json"
    off_final_metrics = tmp_path / "off_final_metrics.json"
    off_ema_metrics = tmp_path / "off_ema_metrics.json"
    _metrics(final_on_metrics, clock_checkpoint, "state_dict", final_points["final_on"])
    _metrics(final_zero_metrics, clock_checkpoint, "state_dict", final_points["final_gate_zero"])
    _metrics(ema_on_metrics, clock_checkpoint, "state_dict_ema", ema_points["ema_on"])
    _metrics(ema_zero_metrics, clock_checkpoint, "state_dict_ema", ema_points["ema_gate_zero"])
    _metrics(off_final_metrics, off_checkpoint, "state_dict", final_points["h65_off_final"])
    _metrics(off_ema_metrics, off_checkpoint, "state_dict_ema", ema_points["h65_off_ema"])
    eval_commit = "e" * 40
    receipt = {
        "schema_version": "duca_h65_singleclock_terminal_eval_receipt_v1",
        "git_commit": eval_commit,
        "clock_checkpoint": str(clock_checkpoint.resolve()),
        "clock_checkpoint_sha256": _sha256(clock_checkpoint),
        "h65_off_checkpoint": str(off_checkpoint.resolve()),
        "h65_off_checkpoint_sha256": _sha256(off_checkpoint),
        "stage1_checkpoint": str(stage1_checkpoint.resolve()),
        "stage1_checkpoint_sha256": _sha256(stage1_checkpoint),
        "families": {
            "final_on": _family_row(final_on_metrics, on_config, False, final_on),
            "final_gate_zero": _family_row(final_zero_metrics, zero_config, True, final_zero),
            "ema_on": _family_row(ema_on_metrics, on_config, False, ema_on),
            "ema_gate_zero": _family_row(ema_zero_metrics, zero_config, True, ema_zero),
            "h65_off_final": _family_row(off_final_metrics, off_config, None),
            "h65_off_ema": _family_row(off_ema_metrics, off_config, None),
        }
    }
    stage1_sha = _sha256(stage1_checkpoint)
    clock = {
        "family": "clock_on", "checkpoint_epoch": 59,
        "successful_optimizer_updates": 6000, "scheduler_last_epoch": 6000,
        "stage1_checkpoint_sha256": stage1_sha, "stage1_checkpoint_epoch": 29,
        "single_clock_values": {"state_dict": {"clock": 0.1}, "state_dict_ema": {"clock": 0.1}},
        "recovery_state_complete": True,
        "recovery_protocol_deviation": [],
    }
    off = dict(
        clock,
        family="h65_off",
        single_clock_values={
            "state_dict": {"registered_clock": 0.0},
            "state_dict_ema": {"registered_clock": 0.0},
        },
    )
    replay_hash = "a" * 64
    off_evaluation = json.loads(off_ema_metrics.read_text(encoding="utf-8"))
    h65_replay_identity = {
        "schema_version": "duca_h65_replay_five_boundary_identity_v1",
        "checkpoint_sha256": _sha256(off_checkpoint),
        "five_boundaries": {
            key: {
                "reference_sha256": replay_hash,
                "replay_sha256": replay_hash,
                "bit_identical": True,
            }
            for key in (
                "selected_integer_indices",
                "gathered_rgb_tensor",
                "videomae_input_tensor",
                "detector_raw_selected_q",
                "canonical_official_evaluator_json",
            )
        },
        "bindings": {
            "config_sha256": _sha256(off_config),
            "annotation_sha256": off_evaluation["evaluation_annotation_sha256"],
            "class_map_sha256": off_evaluation["evaluation_class_map_sha256"],
            "evaluator_sha256": off_evaluation["evaluator"]["source_sha256"],
            "evaluation_config_sha256": off_evaluation["evaluation_config_sha256"],
        },
    }
    nominal_uniform_identity = {
        "schema_version": "duca_h65_singleclock_nominal_uniform_bit_identity_v1",
        "checkpoint_sha256": _sha256(clock_checkpoint),
        "canonical_uniform_positions_exact": True,
        "relative_clock_residual_bit_zero": True,
        "relative_bias_bit_zero": True,
        "first_temporal_mixing": {
            "singleclock_sha256": replay_hash,
            "gate_zero_sha256": replay_hash,
            "bit_identical": True,
        },
        "backbone_output": {
            "singleclock_sha256": replay_hash,
            "gate_zero_sha256": replay_hash,
            "bit_identical": True,
        },
    }
    return {
        "receipt": receipt,
        "clock_audit": clock,
        "off_audit": off,
        "final_bootstrap": _bootstrap(
            tuple(final_points),
            final_points,
            metrics_by_family={
                "final_on": final_on_metrics,
                "final_gate_zero": final_zero_metrics,
                "h65_off_final": off_final_metrics,
            },
            baseline_family="final_gate_zero",
            namespace="SINGLECLOCK_FINAL_PAIRED_VIDEO_BOOTSTRAP_V1",
        ),
        "ema_bootstrap": _bootstrap(
            tuple(ema_points),
            ema_points,
            metrics_by_family={
                "ema_on": ema_on_metrics,
                "ema_gate_zero": ema_zero_metrics,
                "h65_off_ema": off_ema_metrics,
            },
            baseline_family="ema_gate_zero",
            namespace="SINGLECLOCK_EMA_PAIRED_VIDEO_BOOTSTRAP_V1",
        ),
        "h65_replay_identity": h65_replay_identity,
        "nominal_uniform_identity": nominal_uniform_identity,
        "old_pair_bootstrap": _bootstrap(tuple(old_points), old_points),
        "strata": {
            "schema_version": "duca_h65_singleclock_strata_v1",
            "primary_checkpoint_state_key": "state_dict_ema",
            "short_action_delta_pp": 0.1,
            "distortion_interaction_point_pp": 0.2,
        },
        "cost": {
            "schema_version": "duca_h65_singleclock_cost_pair_v1",
            "median_latency_ratio_on_over_gate_zero": 1.0,
            "p90_latency_ratio_on_over_gate_zero": 1.0,
            "peak_memory_ratio_on_over_gate_zero": 1.0,
        },
        "stage1_average_map": 0.594231,
        "expected_eval_commit": eval_commit,
    }


def test_finalizer_accepts_inclusive_minus_point_two_pp_gate(tmp_path):
    result = finalize(**_finalizer_fixture(tmp_path))
    assert result["decision_token"] == "PASS_UNIT1_SINGLECLOCK_GATE"
    assert result["evidence_status"] == "VALID"
    assert all(
        result["primary_metrics"][metric]["point_gate_pass"]
        for metric in ("average_mAP", "mAP@0.6", "mAP@0.7")
    )
    assert result["primary_metrics"]["average_mAP"]["point_delta_pp_decimal"] == "-0.200"
    assert result["boundary_gate"]["status"] == "NOT_EVALUABLE_PREEXISTING_ARTIFACT_GAP"
    assert result["boundary_gate"]["used_for_decision"] is False
    assert result["paper_claim_admissible"] is False
    assert result["unit2_query_builder_eligible"] is False


def test_finalizer_rejects_metrics_changed_after_receipt(tmp_path):
    kwargs = _finalizer_fixture(tmp_path)
    metrics_path = kwargs["receipt"]["families"]["ema_gate_zero"]["metrics_path"]
    with open(metrics_path, "a", encoding="utf-8") as stream:
        stream.write("\n")
    result = finalize(**kwargs)
    assert result["evidence_status"] == "INVALID"
    assert result["decision_token"] is None
    assert result["first_failure"] == "INVALID_CHECKPOINT_CONFIG_EVALUATOR_BINDING"


def test_finalizer_rejects_self_consistent_but_nonofficial_evaluator(tmp_path):
    kwargs = _finalizer_fixture(tmp_path)
    row = kwargs["receipt"]["families"]["ema_on"]
    path = row["metrics_path"]
    payload = json.loads(open(path, encoding="utf-8").read())
    payload["evaluator"] = {
        "module": "opentad.evaluations.mAP",
        "class_name": "mAP",
        "source_path": payload["evaluator"]["source_path"],
        "source_sha256": "f" * 64,
    }
    payload.pop("evaluation_sha256")
    payload["evaluation_sha256"] = canonical_sha256(payload)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream)
    with open(path, "rb") as stream:
        row["metrics_sha256"] = hashlib.sha256(stream.read()).hexdigest()
    result = finalize(**kwargs)
    assert result["evidence_status"] == "INVALID"
    assert result["decision_token"] is None
    assert result["first_failure"] == "INVALID_CHECKPOINT_CONFIG_EVALUATOR_BINDING"


def test_finalizer_rejects_bootstrap_not_bound_to_terminal_prediction(tmp_path):
    kwargs = _finalizer_fixture(tmp_path)
    kwargs["ema_bootstrap"]["prediction_sha256"]["ema_on"] = "f" * 64
    result = finalize(**kwargs)
    assert result["evidence_status"] == "INVALID"
    assert result["decision_token"] is None
    assert result["first_failure"] == "INVALID_BOOTSTRAP_EVIDENCE_BINDING"


def test_recovery_state_and_old_diagnostics_do_not_change_unit1_decision(tmp_path):
    kwargs = _finalizer_fixture(tmp_path)
    kwargs["off_audit"]["recovery_state_complete"] = False
    kwargs["off_audit"]["recovery_protocol_deviation"] = [
        "rng_state",
        "data_loader_state",
    ]
    result = finalize(**kwargs)
    assert result["decision_token"] == "PASS_UNIT1_SINGLECLOCK_GATE"
    assert result["diagnostics"]["h65_off_recovery_contract_pass"] is False
    assert result["diagnostics"]["h65_off_recovery_protocol_deviation"] == [
        "rng_state",
        "data_loader_state",
    ]
    assert result["paper_claim_admissible"] is False


def test_clock_recovery_gap_is_diagnostic_only(tmp_path):
    kwargs = _finalizer_fixture(tmp_path)
    kwargs["clock_audit"]["recovery_state_complete"] = False
    kwargs["clock_audit"]["recovery_protocol_deviation"] = [
        "rng_state",
        "data_loader_state",
    ]
    result = finalize(**kwargs)
    assert result["decision_token"] == "PASS_UNIT1_SINGLECLOCK_GATE"
    assert result["diagnostics"]["clock_recovery_contract_pass"] is False
    assert result["paper_claim_admissible"] is False


def _rewrite_terminal_metric(kwargs, family, metric, value):
    row = kwargs["receipt"]["families"][family]
    path = row["metrics_path"]
    payload = json.loads(open(path, encoding="utf-8").read())
    payload["metrics"][metric] = value
    payload.pop("evaluation_sha256", None)
    payload["evaluation_sha256"] = canonical_sha256(payload)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream)
    with open(path, "rb") as stream:
        row["metrics_sha256"] = hashlib.sha256(stream.read()).hexdigest()


def _set_ema_on_metric(kwargs, metric, value, *, sampled_value=None):
    _rewrite_terminal_metric(kwargs, "ema_on", metric, value)
    kwargs["ema_bootstrap"]["point_estimates"]["ema_on"][metric] = value
    kwargs["ema_bootstrap"]["sampled_metrics"]["ema_on"][metric] = [
        value if sampled_value is None else sampled_value
    ] * 10000


@pytest.mark.parametrize("metric", ["average_mAP", "mAP@0.6", "mAP@0.7"])
def test_any_primary_metric_below_minus_point_two_pp_kills(tmp_path, metric):
    kwargs = _finalizer_fixture(tmp_path)
    _set_ema_on_metric(kwargs, metric, "0.657999")
    result = finalize(**kwargs)
    assert result["decision_token"] == "KILL_SINGLECLOCK_REPRESENTATION"
    assert result["first_failure"] == f"PRIMARY_NONINFERIORITY_FAILURE:{metric}"


def test_bootstrap_ci_is_report_only(tmp_path):
    kwargs = _finalizer_fixture(tmp_path)
    for metric in ("average_mAP", "mAP@0.6", "mAP@0.7"):
        _set_ema_on_metric(kwargs, metric, "0.659", sampled_value="0.650")
    result = finalize(**kwargs)
    assert result["decision_token"] == "PASS_UNIT1_SINGLECLOCK_GATE"
    assert result["primary_metrics"]["average_mAP"]["ci_lower_pp_report_only"] < -0.20


def test_same_checkpoint_gate_zero_loss_does_not_enter_primary_gate(tmp_path):
    kwargs = _finalizer_fixture(tmp_path)
    for metric in ("average_mAP", "mAP@0.6", "mAP@0.7"):
        _rewrite_terminal_metric(kwargs, "ema_gate_zero", metric, 0.70)
        kwargs["ema_bootstrap"]["point_estimates"]["ema_gate_zero"][metric] = 0.70
        kwargs["ema_bootstrap"]["sampled_metrics"]["ema_gate_zero"][metric] = [0.70] * 10000
    result = finalize(**kwargs)
    assert result["decision_token"] == "PASS_UNIT1_SINGLECLOCK_GATE"


def test_h65_replay_identity_failure_is_invalid_not_scientific_kill(tmp_path):
    kwargs = _finalizer_fixture(tmp_path)
    kwargs["h65_replay_identity"]["five_boundaries"]["gathered_rgb_tensor"]["bit_identical"] = False
    result = finalize(**kwargs)
    assert result["evidence_status"] == "INVALID"
    assert result["decision_token"] is None
    assert result["first_failure"] == "INVALID_H65_REPLAY_IDENTITY"


def test_nominal_uniform_bit_identity_failure_is_scientific_kill(tmp_path):
    kwargs = _finalizer_fixture(tmp_path)
    kwargs["nominal_uniform_identity"]["backbone_output"]["bit_identical"] = False
    result = finalize(**kwargs)
    assert result["evidence_status"] == "VALID"
    assert result["decision_token"] == "KILL_SINGLECLOCK_REPRESENTATION"
    assert result["first_failure"] == "NOMINAL_UNIFORM_BIT_IDENTITY_FAILURE"


def test_cost_is_report_only(tmp_path):
    kwargs = _finalizer_fixture(tmp_path)
    kwargs["cost"].update(
        median_latency_ratio_on_over_gate_zero=1.5,
        p90_latency_ratio_on_over_gate_zero=1.8,
        peak_memory_ratio_on_over_gate_zero=1.3,
    )
    result = finalize(**kwargs)
    assert result["decision_token"] == "PASS_UNIT1_SINGLECLOCK_GATE"
    assert result["cost"]["decision_role"] == "report_only"


def test_evaluable_boundary_positive_delta_kills(tmp_path, monkeypatch):
    monkeypatch.setattr(
        finalizer_module,
        "_evaluable_boundary_binding_ok",
        lambda *args, **kwargs: True,
    )
    kwargs = _finalizer_fixture(tmp_path)
    kwargs["boundary"] = {
        "schema_version": "duca_h65_singleclock_boundary_gate_v1",
        "status": "EVALUABLE",
        "comparison": "ema_on_minus_h65_off_ema",
        "high_gapcv_delta_point": 1e-12,
        "high_boundary_density_delta_point": -0.01,
        "bootstrap_samples": 10000,
        "bootstrap_cluster": "whole_video",
        "ci_role": "report_only",
    }
    result = finalize(**kwargs)
    assert result["decision_token"] == "KILL_SINGLECLOCK_REPRESENTATION"
    assert result["first_failure"] == "BOUNDARY_RISK_FAILURE"


def test_evaluable_boundary_zero_is_inclusive_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(
        finalizer_module,
        "_evaluable_boundary_binding_ok",
        lambda *args, **kwargs: True,
    )
    kwargs = _finalizer_fixture(tmp_path)
    kwargs["boundary"] = {
        "schema_version": "duca_h65_singleclock_boundary_gate_v1",
        "status": "EVALUABLE",
        "comparison": "ema_on_minus_h65_off_ema",
        "high_gapcv_delta_point": 0.0,
        "high_boundary_density_delta_point": -0.01,
        "bootstrap_samples": 10000,
        "bootstrap_cluster": "whole_video",
        "ci_role": "report_only",
    }
    result = finalize(**kwargs)
    assert result["decision_token"] == "PASS_UNIT1_SINGLECLOCK_GATE"
    assert result["boundary_gate"]["used_for_decision"] is True


def test_evaluable_boundary_without_provenance_is_invalid(tmp_path):
    kwargs = _finalizer_fixture(tmp_path)
    kwargs["boundary"] = {
        "schema_version": "duca_h65_singleclock_boundary_gate_v1",
        "status": "EVALUABLE",
        "comparison": "ema_on_minus_h65_off_ema",
        "high_gapcv_delta_point": -0.01,
        "high_boundary_density_delta_point": -0.01,
        "bootstrap_samples": 10000,
        "bootstrap_cluster": "whole_video",
        "ci_role": "report_only",
    }
    result = finalize(**kwargs)
    assert result["evidence_status"] == "INVALID"
    assert result["decision_token"] is None
    assert result["first_failure"] == "INVALID_BOUNDARY_EVIDENCE_BINDING"


def test_identity_accounting_mismatch_rejected():
    from tools.bata.finalize_duca_h65_singleclock_terminal import _identity_equal

    base = {"schema_version": "duca_h65_single_clock_selected_input_identity_v2", "sample_count": 1, "records": [_record()], "total_input_exposure_count": 2,
            "unique_physical_window_count": 1, "duplicate_exposure_count": 1,
            "duplicate_samples": [{"sample_id": "v|window_start_frame=0", "duplicate_exposure_count": 1}]}
    changed = dict(base)
    changed["duplicate_exposure_count"] = 0
    assert not _identity_equal(base, changed)


def test_identity_accounting_requires_explicit_fields_and_consistency():
    from tools.bata.finalize_duca_h65_singleclock_terminal import _identity_equal

    valid = {
        "schema_version": "duca_h65_single_clock_selected_input_identity_v2",
        "sample_count": 2,
        "total_input_exposure_count": 3,
        "unique_physical_window_count": 2,
        "duplicate_exposure_count": 1,
        "duplicate_samples": [{"sample_id": "v|window_start_frame=0", "duplicate_exposure_count": 1}],
        "records": [_record(), _record("v|window_start_frame=1", start=1)],
    }
    assert _identity_equal(valid, dict(valid))
    missing = dict(valid)
    del missing["duplicate_samples"]
    assert not _identity_equal(valid, missing)
    malformed = dict(valid, total_input_exposure_count=2)
    assert not _identity_equal(valid, malformed)
    malformed = dict(valid, duplicate_samples=[{"sample_id": "", "duplicate_exposure_count": 1}])
    assert not _identity_equal(valid, malformed)


def _record(sample_id="v|window_start_frame=0", video_name="v", start=0, positions=None):
    positions = list(range(2)) if positions is None else positions
    return {
        "sample_id": sample_id, "video_name": video_name, "window_start_frame": start,
        "selected_valid_len": len(positions), "dense_valid_len": 4,
        "selected_positions": positions, "selected_rgb_sha256": "r",
        "videomae_input_sha256": "v",
        "selected_positions_sha256": "p", "selected_mask_sha256": "m",
    }


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(records=[]),
    lambda p: p["records"].append(_record("v|window_start_frame=2", start=2)),
    lambda p: p["records"].reverse(),
    lambda p: p["records"][0].update(sample_id="wrong"),
    lambda p: p["records"][0].update(selected_positions=[0, 0]),
    lambda p: p["records"][0].update(videomae_input_sha256=""),
])
def test_identity_records_validator_rejects_malformed_payload(mutate):
    payload = {"schema_version": "duca_h65_single_clock_selected_input_identity_v2", "sample_count": 2, "total_input_exposure_count": 2,
               "unique_physical_window_count": 2, "duplicate_exposure_count": 0,
               "duplicate_samples": [],
               "records": [_record(), _record("v|window_start_frame=1", start=1)]}
    mutate(payload)
    assert not _identity_equal(payload, payload)


def test_identity_records_accept_valid_duplicate_payload():
    payload = {"schema_version": "duca_h65_single_clock_selected_input_identity_v2", "sample_count": 1, "total_input_exposure_count": 2,
               "unique_physical_window_count": 1, "duplicate_exposure_count": 1,
               "duplicate_samples": [{"sample_id": "v|window_start_frame=0", "duplicate_exposure_count": 1}],
               "records": [_record()]}
    assert _identity_equal(payload, dict(payload))


def test_identity_records_reject_different_videomae_input_tensor_hash():
    payload = {"schema_version": "duca_h65_single_clock_selected_input_identity_v2",
               "sample_count": 1, "total_input_exposure_count": 1,
               "unique_physical_window_count": 1, "duplicate_exposure_count": 0,
               "duplicate_samples": [], "records": [_record()]}
    changed = json.loads(json.dumps(payload))
    changed["records"][0]["videomae_input_sha256"] = "different"
    assert not _identity_equal(payload, changed)


def test_identity_records_validator_rejects_position_at_dense_boundary():
    payload = {"schema_version": "duca_h65_single_clock_selected_input_identity_v2", "sample_count": 1, "total_input_exposure_count": 1,
               "unique_physical_window_count": 1, "duplicate_exposure_count": 0,
               "duplicate_samples": [], "records": [_record(positions=[0, 4])]}
    assert not _identity_equal(payload, payload)


def test_identity_records_validator_accepts_last_dense_position():
    payload = {"schema_version": "duca_h65_single_clock_selected_input_identity_v2", "sample_count": 1, "total_input_exposure_count": 1,
               "unique_physical_window_count": 1, "duplicate_exposure_count": 0,
               "duplicate_samples": [], "records": [_record(positions=[0, 3])]}
    assert _identity_equal(payload, payload)
