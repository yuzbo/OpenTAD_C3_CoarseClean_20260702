import json
import hashlib

import pytest

from tools.bata.finalize_duca_h65_singleclock_terminal import (
    _twin_execution_contract_ok,
    finalize,
)


def _bootstrap(names, points):
    sampled = {}
    point_estimates = {}
    for name in names:
        sampled[name] = {}
        point_estimates[name] = {}
        for metric, value in points[name].items():
            sampled[name][metric] = [value] * 10000
            point_estimates[name][metric] = value
    return {
        "samples": 10000,
        "lower_rank": 250,
        "upper_rank": 9750,
        "sampled_metrics": sampled,
        "point_estimates": point_estimates,
    }


def _identity(path):
    path.write_text(
        json.dumps(
            {
                "sample_count": 1,
                "records": [
                    {
                        "sample_id": "v|window_start_frame=0",
                        "video_name": "v",
                        "window_start_frame": 0,
                        "selected_valid_len": 384,
                        "dense_valid_len": 768,
                        "selected_positions": list(range(384)),
                        "selected_rgb_sha256": "r",
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


def _metrics(path, checkpoint, state_key):
    path.write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_epoch": 59,
                "checkpoint_state_key": state_key,
            }
        ),
        encoding="utf-8",
    )


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
    final_on_metrics = tmp_path / "final_on_metrics.json"
    final_zero_metrics = tmp_path / "final_zero_metrics.json"
    ema_on_metrics = tmp_path / "ema_on_metrics.json"
    ema_zero_metrics = tmp_path / "ema_zero_metrics.json"
    off_final_metrics = tmp_path / "off_final_metrics.json"
    off_ema_metrics = tmp_path / "off_ema_metrics.json"
    _metrics(final_on_metrics, clock_checkpoint, "state_dict")
    _metrics(final_zero_metrics, clock_checkpoint, "state_dict")
    _metrics(ema_on_metrics, clock_checkpoint, "state_dict_ema")
    _metrics(ema_zero_metrics, clock_checkpoint, "state_dict_ema")
    _metrics(off_final_metrics, off_checkpoint, "state_dict")
    _metrics(off_ema_metrics, off_checkpoint, "state_dict_ema")
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
        single_clock_values={"state_dict": {}, "state_dict_ema": {}},
    )
    metrics = ("average_mAP", "mAP@0.6", "mAP@0.7")
    final_points = {
        "final_on": dict.fromkeys(metrics, 0.66),
        "final_gate_zero": dict.fromkeys(metrics, 0.65),
        "h65_off_final": dict.fromkeys(metrics, 0.65),
    }
    ema_points = {
        "ema_on": dict.fromkeys(metrics, 0.67),
        "ema_gate_zero": dict.fromkeys(metrics, 0.66),
        "h65_off_ema": dict.fromkeys(metrics, 0.66),
    }
    old_points = {
        "truetime": dict.fromkeys(metrics, 0.62),
        "rankpack": dict.fromkeys(metrics, 0.61),
    }
    return {
        "receipt": receipt,
        "clock_audit": clock,
        "off_audit": off,
        "final_bootstrap": _bootstrap(tuple(final_points), final_points),
        "ema_bootstrap": _bootstrap(tuple(ema_points), ema_points),
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


def test_finalizer_accepts_frozen_positive_gate(tmp_path):
    result = finalize(**_finalizer_fixture(tmp_path))
    assert result["decision"] == "CONTINUE_TO_REPLICATION"
    assert result["identity_gate_pass"] is True
    assert result["twin_execution_contract_pass"] is True
    assert result["family_execution_contract_pass"] is True
    assert result["clock_recovery_contract_pass"] is True
    assert result["h65_off_recovery_contract_pass"] is True
    assert result["paper_claim_admissible"] is True
    assert result["bridge_authorized"] is False


def test_finalizer_rejects_metrics_changed_after_receipt(tmp_path):
    kwargs = _finalizer_fixture(tmp_path)
    metrics_path = kwargs["receipt"]["families"]["ema_gate_zero"]["metrics_path"]
    with open(metrics_path, "a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="metrics hash mismatch"):
        finalize(**kwargs)


def test_finalizer_keeps_positive_mechanism_diagnostic_but_blocks_replication_for_legacy_off_recovery_gap(tmp_path):
    kwargs = _finalizer_fixture(tmp_path)
    kwargs["off_audit"]["recovery_state_complete"] = False
    kwargs["off_audit"]["recovery_protocol_deviation"] = [
        "rng_state",
        "data_loader_state",
    ]
    result = finalize(**kwargs)
    assert result["decision"] == "REVISE_WITHOUT_MORE_TIME_MODULES"
    assert result["checkpoint_audit_gate_pass"] is True
    assert result["h65_off_recovery_contract_pass"] is False
    assert result["h65_off_recovery_protocol_deviation"] == [
        "rng_state",
        "data_loader_state",
    ]
    assert result["paper_claim_admissible"] is False


def test_finalizer_hard_fails_when_clock_recovery_state_is_incomplete(tmp_path):
    kwargs = _finalizer_fixture(tmp_path)
    kwargs["clock_audit"]["recovery_state_complete"] = False
    kwargs["clock_audit"]["recovery_protocol_deviation"] = [
        "rng_state",
        "data_loader_state",
    ]
    result = finalize(**kwargs)
    assert result["decision"] == "PIVOT_TO_ACQUISITION_OR_TRAINING_MATURITY"
    assert result["checkpoint_audit_gate_pass"] is True
    assert result["clock_recovery_contract_pass"] is False
    assert result["paper_claim_admissible"] is False
