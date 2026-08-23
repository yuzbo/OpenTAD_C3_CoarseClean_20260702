from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tools.bata.duca_p0_training import atomic_write_json


METRICS = ("average_mAP", "mAP@0.6", "mAP@0.7")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"expected a JSON object: {path}")
    return payload


def _metric(metrics: Mapping[str, Any], key: str) -> float:
    aliases = {
        "average_mAP": ("average_mAP", "Avg-mAP", "avg_mAP"),
        "mAP@0.6": ("mAP@0.6", "0.6"),
        "mAP@0.7": ("mAP@0.7", "0.7"),
    }
    for alias in aliases[key]:
        if alias in metrics:
            value = float(metrics[alias])
            _require(np.isfinite(value), f"non-finite metric {key}")
            return value
    raise ValueError(f"missing metric {key}")


def _exact_interval(values: np.ndarray) -> tuple[float, float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    _require(ordered.shape == (10000,), "terminal bootstrap must contain exactly 10,000 draws")
    return float(ordered[249]), float(ordered[9749])


def _paired_delta(
    bootstrap: Mapping[str, Any], lhs: str, rhs: str, metric: str
) -> dict[str, float]:
    _require(bootstrap.get("samples") == 10000, "bootstrap sample count must be 10,000")
    _require(
        (bootstrap.get("lower_rank"), bootstrap.get("upper_rank")) == (250, 9750),
        "bootstrap interval ranks must be 250/9750",
    )
    sampled = bootstrap.get("sampled_metrics")
    points = bootstrap.get("point_estimates")
    _require(isinstance(sampled, Mapping) and isinstance(points, Mapping), "bootstrap evidence is incomplete")
    lhs_draws = np.asarray(sampled[lhs][metric], dtype=np.float64)
    rhs_draws = np.asarray(sampled[rhs][metric], dtype=np.float64)
    _require(lhs_draws.shape == rhs_draws.shape == (10000,), "paired bootstrap families do not align")
    delta = lhs_draws - rhs_draws
    lower, upper = _exact_interval(delta)
    point = _metric(points[lhs], metric) - _metric(points[rhs], metric)
    return {
        "point": float(point),
        "ci_lower_exact_rank": lower,
        "ci_upper_exact_rank": upper,
        "point_pp": float(point * 100.0),
        "ci_lower_pp": float(lower * 100.0),
        "ci_upper_pp": float(upper * 100.0),
    }


def _identity_equal(on: Mapping[str, Any], zero: Mapping[str, Any]) -> bool:
    if on.get("sample_count") != zero.get("sample_count"):
        return False
    on_records = on.get("records")
    zero_records = zero.get("records")
    if not isinstance(on_records, list) or not isinstance(zero_records, list):
        return False
    keys = (
        "sample_id",
        "video_name",
        "window_start_frame",
        "selected_valid_len",
        "dense_valid_len",
        "selected_positions",
        "selected_rgb_sha256",
        "selected_positions_sha256",
        "selected_mask_sha256",
    )
    return [tuple(row.get(key) if key != "selected_positions" else tuple(row.get(key, ())) for key in keys) for row in on_records] == [
        tuple(row.get(key) if key != "selected_positions" else tuple(row.get(key, ())) for key in keys)
        for row in zero_records
    ]


def _config_hash_ok(row: Mapping[str, Any], expected_suffix: str) -> bool:
    config_path = Path(str(row.get("config_path", "")))
    expected_hash = str(row.get("config_sha256", ""))
    if not _file_hash_matches(config_path, expected_hash):
        return False
    if config_path.as_posix().endswith(expected_suffix) is False:
        return False
    return True


def _file_hash_matches(path: str | Path, expected_hash: str) -> bool:
    artifact = Path(path)
    expected = str(expected_hash)
    return bool(
        artifact.is_file()
        and len(expected) == 64
        and hashlib.sha256(artifact.read_bytes()).hexdigest() == expected
    )


def _same_path(lhs: str | Path, rhs: str | Path) -> bool:
    return Path(lhs).resolve() == Path(rhs).resolve()


def _family_execution_contract_ok(
    row: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    expected_config_suffix: str,
    expected_gate_zero: bool | None,
    expected_checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    expected_state_key: str,
    require_identity: bool,
) -> bool:
    if row.get("single_clock_gate_zero") is not expected_gate_zero:
        return False
    if not _config_hash_ok(row, expected_config_suffix):
        return False
    if not _same_path(metrics.get("checkpoint_path", ""), expected_checkpoint_path):
        return False
    if metrics.get("checkpoint_sha256") != expected_checkpoint_sha256:
        return False
    if metrics.get("checkpoint_epoch") != 59:
        return False
    if metrics.get("checkpoint_state_key") != expected_state_key:
        return False
    identity_path = row.get("selected_input_identity_path")
    identity_hash = row.get("selected_input_identity_sha256")
    if require_identity:
        return _file_hash_matches(identity_path or "", identity_hash or "")
    return identity_path is None and identity_hash is None


def _twin_execution_contract_ok(
    on_row: Mapping[str, Any], zero_row: Mapping[str, Any]
) -> bool:
    on_suffix = "configs/adatad/thumos/duca_h65_first_singleclock_cycle4.py"
    zero_suffix = (
        "configs/adatad/thumos/"
        "duca_h65_first_singleclock_cycle4_gate_zero.py"
    )
    return bool(
        on_row.get("single_clock_gate_zero") is False
        and zero_row.get("single_clock_gate_zero") is True
        and _config_hash_ok(on_row, on_suffix)
        and _config_hash_ok(zero_row, zero_suffix)
        and on_row.get("config_sha256") != zero_row.get("config_sha256")
    )


def _audit_ok(clock: Mapping[str, Any], off: Mapping[str, Any]) -> bool:
    clock_values = clock.get("single_clock_values")
    off_values = off.get("single_clock_values")
    scalar_ok = (
        isinstance(clock_values, Mapping)
        and isinstance(off_values, Mapping)
        and set(clock_values) == {"state_dict", "state_dict_ema"}
        and set(off_values) == {"state_dict", "state_dict_ema"}
        and all(
            isinstance(clock_values[key], Mapping)
            and len(clock_values[key]) == 1
            and np.isfinite(float(next(iter(clock_values[key].values()))))
            and float(next(iter(clock_values[key].values()))) != 0.0
            and not off_values[key]
            for key in ("state_dict", "state_dict_ema")
        )
    )
    common = (
        clock.get("checkpoint_epoch") == off.get("checkpoint_epoch") == 59
        and clock.get("successful_optimizer_updates") == off.get("successful_optimizer_updates") == 6000
        and clock.get("scheduler_last_epoch") == off.get("scheduler_last_epoch") == 6000
        and clock.get("stage1_checkpoint_sha256") == off.get("stage1_checkpoint_sha256")
        and clock.get("stage1_checkpoint_epoch") == off.get("stage1_checkpoint_epoch") == 29
    )
    return bool(common and scalar_ok and clock.get("family") == "clock_on" and off.get("family") == "h65_off")


def finalize(
    *,
    receipt: Mapping[str, Any],
    clock_audit: Mapping[str, Any],
    off_audit: Mapping[str, Any],
    final_bootstrap: Mapping[str, Any],
    ema_bootstrap: Mapping[str, Any],
    old_pair_bootstrap: Mapping[str, Any],
    strata: Mapping[str, Any],
    cost: Mapping[str, Any],
    stage1_average_map: float,
    expected_eval_commit: str,
) -> dict[str, Any]:
    _require(
        receipt.get("schema_version")
        == "duca_h65_singleclock_terminal_eval_receipt_v1",
        "terminal receipt schema mismatch",
    )
    _require(
        receipt.get("git_commit") == expected_eval_commit
        and len(str(expected_eval_commit)) == 40,
        "terminal evaluation commit mismatch",
    )
    families = receipt.get("families")
    required = {
        "final_on",
        "final_gate_zero",
        "ema_on",
        "ema_gate_zero",
        "h65_off_final",
        "h65_off_ema",
    }
    _require(isinstance(families, Mapping) and set(families) == required, "terminal receipt families differ from the frozen six readouts")

    clock_checkpoint = receipt.get("clock_checkpoint", "")
    clock_checkpoint_sha256 = receipt.get("clock_checkpoint_sha256", "")
    off_checkpoint = receipt.get("h65_off_checkpoint", "")
    off_checkpoint_sha256 = receipt.get("h65_off_checkpoint_sha256", "")
    stage1_checkpoint = receipt.get("stage1_checkpoint", "")
    stage1_checkpoint_sha256 = receipt.get("stage1_checkpoint_sha256", "")
    _require(
        _file_hash_matches(clock_checkpoint, clock_checkpoint_sha256),
        "SingleClock checkpoint binding failed",
    )
    _require(
        _file_hash_matches(off_checkpoint, off_checkpoint_sha256),
        "H65 OFF checkpoint binding failed",
    )
    _require(
        _file_hash_matches(stage1_checkpoint, stage1_checkpoint_sha256),
        "Stage-1 checkpoint binding failed",
    )

    clock_suffix = "configs/adatad/thumos/duca_h65_first_singleclock_cycle4.py"
    zero_suffix = (
        "configs/adatad/thumos/"
        "duca_h65_first_singleclock_cycle4_gate_zero.py"
    )
    off_suffix = (
        "configs/adatad/thumos/"
        "duca_sampling_rate_curriculum_stage2_joint384.py"
    )
    family_specs = {
        "final_on": (clock_suffix, False, clock_checkpoint, clock_checkpoint_sha256, "state_dict", True),
        "final_gate_zero": (zero_suffix, True, clock_checkpoint, clock_checkpoint_sha256, "state_dict", True),
        "ema_on": (clock_suffix, False, clock_checkpoint, clock_checkpoint_sha256, "state_dict_ema", True),
        "ema_gate_zero": (zero_suffix, True, clock_checkpoint, clock_checkpoint_sha256, "state_dict_ema", True),
        "h65_off_final": (off_suffix, None, off_checkpoint, off_checkpoint_sha256, "state_dict", False),
        "h65_off_ema": (off_suffix, None, off_checkpoint, off_checkpoint_sha256, "state_dict_ema", False),
    }
    loaded_metrics = {}
    execution_contract = {}
    for family, spec in family_specs.items():
        row = families[family]
        _require(isinstance(row, Mapping), f"terminal family row is invalid: {family}")
        metrics_path = row.get("metrics_path", "")
        _require(
            _file_hash_matches(metrics_path, row.get("metrics_sha256", "")),
            f"terminal metrics hash mismatch: {family}",
        )
        metrics = _load(metrics_path)
        loaded_metrics[family] = metrics
        execution_contract[family] = _family_execution_contract_ok(
            row,
            metrics,
            expected_config_suffix=spec[0],
            expected_gate_zero=spec[1],
            expected_checkpoint_path=spec[2],
            expected_checkpoint_sha256=spec[3],
            expected_state_key=spec[4],
            require_identity=spec[5],
        )
    _require(all(execution_contract.values()), "terminal family execution binding failed")

    identity = {}
    twin_execution_contract = {}
    for prefix in ("final", "ema"):
        on_row = families[f"{prefix}_on"]
        zero_row = families[f"{prefix}_gate_zero"]
        twin_execution_contract[prefix] = _twin_execution_contract_ok(
            on_row, zero_row
        )
        on_metrics = loaded_metrics[f"{prefix}_on"]
        zero_metrics = loaded_metrics[f"{prefix}_gate_zero"]
        for key in ("checkpoint_path", "checkpoint_sha256", "checkpoint_epoch", "checkpoint_state_key"):
            _require(on_metrics.get(key) == zero_metrics.get(key), f"{prefix} twin differs on {key}")
        on = _load(on_row["selected_input_identity_path"])
        zero = _load(zero_row["selected_input_identity_path"])
        identity[prefix] = _identity_equal(on, zero)

    family_names = {
        "final": (final_bootstrap, "final_on", "final_gate_zero", "h65_off_final"),
        "ema": (ema_bootstrap, "ema_on", "ema_gate_zero", "h65_off_ema"),
    }
    estimates: dict[str, Any] = {}
    for prefix, (artifact, on_name, zero_name, off_name) in family_names.items():
        estimates[prefix] = {
            "single_clock": {
                metric: _paired_delta(artifact, on_name, zero_name, metric)
                for metric in METRICS
            },
            "external": {
                metric: _paired_delta(artifact, on_name, off_name, metric)
                for metric in METRICS
            },
            "coadaptation": {
                metric: _paired_delta(artifact, zero_name, off_name, metric)
                for metric in ("average_mAP", "mAP@0.7")
            },
        }

    old = {
        metric: _paired_delta(old_pair_bootstrap, "truetime", "rankpack", metric)
        for metric in METRICS
    }
    old_pair_partial_representation_evidence = (
        old["average_mAP"]["ci_lower_pp"] > 0.0
        and old["mAP@0.6"]["ci_lower_pp"] > 0.0
    )
    old_pair_no_explicit_harm = (
        old["average_mAP"]["ci_upper_pp"] > 0.0
        and old["mAP@0.7"]["ci_upper_pp"] >= -0.20
    )

    _require(strata.get("schema_version") == "duca_h65_singleclock_strata_v1", "strata artifact schema mismatch")
    _require(
        strata.get("primary_checkpoint_state_key") == "state_dict_ema",
        "strata evidence must use the frozen EMA checkpoint state",
    )
    _require(cost.get("schema_version") == "duca_h65_singleclock_cost_pair_v1", "cost artifact schema mismatch")
    cost_pass = (
        float(cost["median_latency_ratio_on_over_gate_zero"]) <= 1.01
        and float(cost["p90_latency_ratio_on_over_gate_zero"]) <= 1.02
        and float(cost["peak_memory_ratio_on_over_gate_zero"]) <= 1.02
    )
    strata_pass = (
        float(strata["short_action_delta_pp"]) >= -0.50
        and float(strata["distortion_interaction_point_pp"]) > 0.0
    )

    final_sc = estimates["final"]["single_clock"]
    ema_sc = estimates["ema"]["single_clock"]
    ema_ext = estimates["ema"]["external"]
    ema_co = estimates["ema"]["coadaptation"]
    final_off_avg = _metric(final_bootstrap["point_estimates"]["h65_off_final"], "average_mAP")
    ema_off_avg = _metric(ema_bootstrap["point_estimates"]["h65_off_ema"], "average_mAP")
    baseline_mature = (ema_off_avg - float(stage1_average_map)) * 100.0 >= 0.50

    direction_consistent = all(
        final_sc[metric]["point"] * ema_sc[metric]["point"] >= 0.0
        for metric in METRICS
    )
    main_pass = (
        ema_sc["average_mAP"]["point_pp"] >= 0.50
        and ema_sc["average_mAP"]["ci_lower_pp"] > 0.0
        and ema_sc["mAP@0.6"]["point_pp"] >= 0.0
        and ema_sc["mAP@0.7"]["point_pp"] >= 0.0
        and ema_sc["mAP@0.7"]["ci_lower_pp"] > -0.20
        and ema_ext["average_mAP"]["point_pp"] >= 0.50
        and ema_ext["mAP@0.7"]["point_pp"] >= 0.0
        and all(
            row["ci_lower_pp"] >= -0.20 and row["ci_upper_pp"] <= 0.20
            for row in ema_co.values()
        )
    )
    hard_fail = (
        not old_pair_no_explicit_harm
        or not baseline_mature
        or not all(identity.values())
        or not all(twin_execution_contract.values())
        or not _audit_ok(clock_audit, off_audit)
        or not cost_pass
        or any(
            sc["average_mAP"]["point_pp"] <= 0.0
            or sc["average_mAP"]["ci_upper_pp"] <= 0.0
            or sc["mAP@0.7"]["point_pp"] <= -0.50
            or sc["mAP@0.7"]["ci_upper_pp"] < -0.20
            for sc in (final_sc, ema_sc)
        )
    )
    if hard_fail:
        decision = "PIVOT_TO_ACQUISITION_OR_TRAINING_MATURITY"
    elif main_pass and strata_pass and direction_consistent:
        decision = "CONTINUE_TO_REPLICATION"
    else:
        decision = "REVISE_WITHOUT_MORE_TIME_MODULES"

    return {
        "schema_version": "duca_h65_singleclock_terminal_adjudication_v1",
        "decision": decision,
        "identity_gate_pass": all(identity.values()),
        "twin_execution_contract_pass": all(twin_execution_contract.values()),
        "family_execution_contract_pass": all(execution_contract.values()),
        "checkpoint_audit_gate_pass": _audit_ok(clock_audit, off_audit),
        "old_pair_representation_gate_pass": old_pair_no_explicit_harm,
        "old_pair_partial_representation_evidence": old_pair_partial_representation_evidence,
        "h65_off_training_maturity_gate_pass": baseline_mature,
        "cost_gate_pass": cost_pass,
        "strata_gate_pass": strata_pass,
        "final_ema_direction_consistent": direction_consistent,
        "identity": identity,
        "old_pair": old,
        "estimates": estimates,
        "cost": dict(cost),
        "strata": dict(strata),
        "stage1_average_map": float(stage1_average_map),
        "h65_off_final_average_map": final_off_avg,
        "h65_off_ema_average_map": ema_off_avg,
        "claim_boundary": "single_seed_representation_gate_only",
        "bridge_authorized": False,
        "dynamic_k_authorized": False,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Finalize the frozen H65 SingleClock terminal gate")
    parser.add_argument("--terminal-receipt", required=True)
    parser.add_argument("--clock-audit", required=True)
    parser.add_argument("--off-audit", required=True)
    parser.add_argument("--final-bootstrap", required=True)
    parser.add_argument("--ema-bootstrap", required=True)
    parser.add_argument("--old-pair-bootstrap", required=True)
    parser.add_argument("--strata", required=True)
    parser.add_argument("--cost", required=True)
    parser.add_argument("--stage1-average-map", type=float, default=0.594231)
    parser.add_argument("--expected-eval-commit", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = finalize(
        receipt=_load(args.terminal_receipt),
        clock_audit=_load(args.clock_audit),
        off_audit=_load(args.off_audit),
        final_bootstrap=_load(args.final_bootstrap),
        ema_bootstrap=_load(args.ema_bootstrap),
        old_pair_bootstrap=_load(args.old_pair_bootstrap),
        strata=_load(args.strata),
        cost=_load(args.cost),
        stage1_average_map=args.stage1_average_map,
        expected_eval_commit=args.expected_eval_commit,
    )
    atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
