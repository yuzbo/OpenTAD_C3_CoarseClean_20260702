import json
import hashlib

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


def _metrics(path, state_key):
    path.write_text(
        json.dumps(
            {
                "checkpoint_path": "epoch_59.pth",
                "checkpoint_sha256": "c" * 64,
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
    on.write_text("single_clock_gate_zero = False\n", encoding="utf-8")
    zero.write_text("single_clock_gate_zero = True\n", encoding="utf-8")
    return on, zero


def _config_row(path, gate_zero):
    return {
        "config_path": str(path),
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "single_clock_gate_zero": gate_zero,
    }


def test_twin_execution_contract_rejects_a_mislabeled_gate_zero(tmp_path):
    on, zero = _configs(tmp_path)
    assert _twin_execution_contract_ok(
        _config_row(on, False), _config_row(zero, True)
    )
    assert not _twin_execution_contract_ok(
        _config_row(on, False), _config_row(zero, False)
    )


def test_finalizer_accepts_frozen_positive_gate(tmp_path):
    on_config, zero_config = _configs(tmp_path)
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
    _metrics(final_on_metrics, "state_dict")
    _metrics(final_zero_metrics, "state_dict")
    _metrics(ema_on_metrics, "state_dict_ema")
    _metrics(ema_zero_metrics, "state_dict_ema")
    receipt = {
        "families": {
            "final_on": {"selected_input_identity_path": str(final_on), "metrics_path": str(final_on_metrics), **_config_row(on_config, False)},
            "final_gate_zero": {"selected_input_identity_path": str(final_zero), "metrics_path": str(final_zero_metrics), **_config_row(zero_config, True)},
            "ema_on": {"selected_input_identity_path": str(ema_on), "metrics_path": str(ema_on_metrics), **_config_row(on_config, False)},
            "ema_gate_zero": {"selected_input_identity_path": str(ema_zero), "metrics_path": str(ema_zero_metrics), **_config_row(zero_config, True)},
            "h65_off_final": {},
            "h65_off_ema": {},
        }
    }
    clock = {
        "family": "clock_on", "checkpoint_epoch": 59,
        "successful_optimizer_updates": 6000, "scheduler_last_epoch": 6000,
        "stage1_checkpoint_sha256": "s", "stage1_checkpoint_epoch": 29,
        "single_clock_values": {"state_dict": {"clock": 0.1}, "state_dict_ema": {"clock": 0.1}},
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
    result = finalize(
        receipt=receipt,
        clock_audit=clock,
        off_audit=off,
        final_bootstrap=_bootstrap(tuple(final_points), final_points),
        ema_bootstrap=_bootstrap(tuple(ema_points), ema_points),
        old_pair_bootstrap=_bootstrap(tuple(old_points), old_points),
        strata={
            "schema_version": "duca_h65_singleclock_strata_v1",
            "primary_checkpoint_state_key": "state_dict_ema",
            "short_action_delta_pp": 0.1,
            "distortion_interaction_point_pp": 0.2,
        },
        cost={
            "schema_version": "duca_h65_singleclock_cost_pair_v1",
            "median_latency_ratio_on_over_gate_zero": 1.0,
            "p90_latency_ratio_on_over_gate_zero": 1.0,
            "peak_memory_ratio_on_over_gate_zero": 1.0,
        },
        stage1_average_map=0.594231,
    )
    assert result["decision"] == "CONTINUE_TO_REPLICATION"
    assert result["identity_gate_pass"] is True
    assert result["twin_execution_contract_pass"] is True
    assert result["bridge_authorized"] is False
