from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tools.bata.duca_p0_training import atomic_write_json


METRICS = ("average_mAP", "mAP@0.6", "mAP@0.7")
NONINFERIORITY_MARGIN_PP = Decimal("-0.20")
VALID_DECISIONS = {
    "PASS_UNIT1_SINGLECLOCK_GATE",
    "KILL_SINGLECLOCK_REPRESENTATION",
}


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


def _metric_decimal(metrics: Mapping[str, Any], key: str) -> Decimal:
    aliases = {
        "average_mAP": ("average_mAP", "Avg-mAP", "avg_mAP"),
        "mAP@0.6": ("mAP@0.6", "0.6"),
        "mAP@0.7": ("mAP@0.7", "0.7"),
    }
    for alias in aliases[key]:
        if alias in metrics:
            value = Decimal(str(metrics[alias]))
            _require(value.is_finite(), f"non-finite metric {key}")
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
    def valid_records(payload: Mapping[str, Any]) -> bool:
        records = payload.get("records")
        sample_count = payload.get("sample_count")
        if not isinstance(records, list) or isinstance(sample_count, bool) or not isinstance(sample_count, int):
            return False
        if len(records) != sample_count or sample_count <= 0:
            return False
        required = ("sample_id", "video_name", "window_start_frame", "selected_valid_len", "dense_valid_len", "selected_positions", "selected_rgb_sha256", "videomae_input_sha256", "selected_positions_sha256", "selected_mask_sha256")
        ids = []
        for row in records:
            if not isinstance(row, Mapping) or any(key not in row for key in required):
                return False
            if not isinstance(row["sample_id"], str) or not row["sample_id"] or not isinstance(row["video_name"], str) or not row["video_name"]:
                return False
            if any(not isinstance(row[key], str) or not row[key] for key in ("selected_rgb_sha256", "videomae_input_sha256", "selected_positions_sha256", "selected_mask_sha256")):
                return False
            fields = ("window_start_frame", "selected_valid_len", "dense_valid_len")
            if any(isinstance(row[key], bool) or not isinstance(row[key], int) for key in fields):
                return False
            if row["window_start_frame"] < 0 or row["selected_valid_len"] <= 0 or row["dense_valid_len"] <= 0 or row["selected_valid_len"] > row["dense_valid_len"]:
                return False
            positions = row["selected_positions"]
            if not isinstance(positions, list) or len(positions) != row["selected_valid_len"] or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= row["dense_valid_len"] for value in positions) or positions != sorted(set(positions)):
                return False
            if row["sample_id"] != f"{row['video_name']}|window_start_frame={row['window_start_frame']}":
                return False
            ids.append(row["sample_id"])
        return len(set(ids)) == len(ids) and ids == sorted(ids)

    def valid_accounting(payload: Mapping[str, Any]) -> bool:
        if payload.get("schema_version") != "duca_h65_single_clock_selected_input_identity_v2":
            return False
        required = ("sample_count", "total_input_exposure_count", "unique_physical_window_count", "duplicate_exposure_count", "duplicate_samples")
        if any(key not in payload for key in required):
            return False
        integers = tuple(payload[key] for key in required[:4])
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in integers):
            return False
        unique = payload["unique_physical_window_count"]
        if payload["sample_count"] != unique:
            return False
        duplicates = payload["duplicate_exposure_count"]
        if payload["total_input_exposure_count"] != unique + duplicates:
            return False
        if not valid_records(payload):
            return False
        rows = payload["duplicate_samples"]
        if not isinstance(rows, list):
            return False
        normalized = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"sample_id", "duplicate_exposure_count"}:
                return False
            sample_id = row["sample_id"]
            count = row["duplicate_exposure_count"]
            if not isinstance(sample_id, str) or not sample_id:
                return False
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                return False
            normalized.append((sample_id, count))
        if len({sample_id for sample_id, _ in normalized}) != len(normalized):
            return False
        if normalized != sorted(normalized):
            return False
        if sum(count for _, count in normalized) != duplicates:
            return False
        if any(sample_id not in {row["sample_id"] for row in payload["records"]} for sample_id, _ in normalized):
            return False
        return duplicates != 0 or not normalized

    if not valid_accounting(on) or not valid_accounting(zero):
        return False
    if on["sample_count"] != zero["sample_count"]:
        return False
    accounting_keys = (
        "total_input_exposure_count",
        "unique_physical_window_count",
        "duplicate_exposure_count",
        "duplicate_samples",
    )
    if any(on[key] != zero[key] for key in accounting_keys):
        return False
    on_records = on["records"]
    zero_records = zero["records"]
    keys = (
        "sample_id",
        "video_name",
        "window_start_frame",
        "selected_valid_len",
        "dense_valid_len",
        "selected_positions",
        "selected_rgb_sha256",
        "videomae_input_sha256",
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
            and isinstance(off_values[key], Mapping)
            and len(off_values[key]) <= 1
            and all(
                np.isfinite(float(value)) and float(value) == 0.0
                for value in off_values[key].values()
            )
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


def _sha256_text(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _h65_replay_identity_ok(
    payload: Mapping[str, Any],
    *,
    expected_checkpoint_sha256: str,
    expected_bindings: Mapping[str, str],
) -> bool:
    if payload.get("schema_version") != "duca_h65_replay_five_boundary_identity_v1":
        return False
    if payload.get("checkpoint_sha256") != expected_checkpoint_sha256:
        return False
    boundaries = payload.get("five_boundaries")
    required = {
        "selected_integer_indices",
        "gathered_rgb_tensor",
        "videomae_input_tensor",
        "detector_raw_selected_q",
        "canonical_official_evaluator_json",
    }
    if not isinstance(boundaries, Mapping) or set(boundaries) != required:
        return False
    for key in sorted(required):
        row = boundaries[key]
        if not isinstance(row, Mapping) or row.get("bit_identical") is not True:
            return False
        reference = row.get("reference_sha256")
        replay = row.get("replay_sha256")
        if not _sha256_text(reference) or reference != replay:
            return False
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        return False
    return all(
        _sha256_text(expected_bindings.get(key))
        and bindings.get(key) == expected_bindings.get(key)
        for key in expected_bindings
    )


def _nominal_uniform_identity_ok(
    payload: Mapping[str, Any], *, expected_checkpoint_sha256: str
) -> bool:
    if payload.get("schema_version") != "duca_h65_singleclock_nominal_uniform_bit_identity_v1":
        return False
    if payload.get("checkpoint_sha256") != expected_checkpoint_sha256:
        return False
    if payload.get("canonical_uniform_positions_exact") is not True:
        return False
    if payload.get("relative_clock_residual_bit_zero") is not True:
        return False
    if payload.get("relative_bias_bit_zero") is not True:
        return False
    for key in ("first_temporal_mixing", "backbone_output"):
        row = payload.get(key)
        if not isinstance(row, Mapping) or row.get("bit_identical") is not True:
            return False
        on_hash = row.get("singleclock_sha256")
        zero_hash = row.get("gate_zero_sha256")
        if not _sha256_text(on_hash) or on_hash != zero_hash:
            return False
    return True


def _invalid_result(
    *, first_failure: str, diagnostics: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "schema_version": "duca_h65_singleclock_unit1_terminal_gate_v2",
        "evidence_status": "INVALID",
        "decision_token": None,
        "primary_checkpoint_state_key": "state_dict_ema",
        "primary_comparison": "ema_on_minus_h65_off_ema",
        "diagnostics": dict(diagnostics or {}),
        "claim_boundary": "single_seed_unit1_development_representation_gate_only",
        "paper_claim_admissible": False,
        "unit2_query_builder_eligible": False,
        "dynamic_k_authorized": False,
        "first_failure": first_failure,
    }


def _boundary_gate(
    boundary: Mapping[str, Any] | None,
    *,
    missing_artifacts: tuple[str, ...],
) -> dict[str, Any]:
    if boundary is None:
        missing = list(dict.fromkeys(str(item) for item in missing_artifacts if str(item)))
        if not missing:
            missing = ["boundary_gate_artifact_not_supplied"]
        return {
            "status": "NOT_EVALUABLE_PREEXISTING_ARTIFACT_GAP",
            "used_for_decision": False,
            "comparison": "ema_on_minus_h65_off_ema",
            "missing_artifacts": missing,
            "boundary_mechanism_claim_supported": False,
            "bootstrap_samples": None,
            "bootstrap_cluster": "whole_video",
            "ci_role": "report_only",
        }
    _require(
        boundary.get("schema_version") == "duca_h65_singleclock_boundary_gate_v1",
        "boundary gate artifact schema mismatch",
    )
    status = boundary.get("status")
    if status == "NOT_EVALUABLE_PREEXISTING_ARTIFACT_GAP":
        missing = boundary.get("missing_artifacts")
        _require(isinstance(missing, list) and missing, "boundary artifact gap must be explicit")
        return {
            **dict(boundary),
            "used_for_decision": False,
            "boundary_mechanism_claim_supported": False,
        }
    _require(status == "EVALUABLE", "unknown boundary gate status")
    _require(
        boundary.get("comparison") == "ema_on_minus_h65_off_ema",
        "boundary comparison is not the frozen primary pair",
    )
    _require(
        boundary.get("bootstrap_samples") == 10000
        and boundary.get("bootstrap_cluster") == "whole_video"
        and boundary.get("ci_role") == "report_only",
        "boundary bootstrap contract mismatch",
    )
    high_gap = float(boundary["high_gapcv_delta_point"])
    high_density = float(boundary["high_boundary_density_delta_point"])
    _require(np.isfinite(high_gap) and np.isfinite(high_density), "boundary delta is non-finite")
    return {
        **dict(boundary),
        "used_for_decision": True,
        "high_gapcv_pass": high_gap <= 0.0,
        "high_boundary_density_pass": high_density <= 0.0,
        "boundary_mechanism_claim_supported": high_gap <= 0.0 and high_density <= 0.0,
    }


def finalize(
    *,
    receipt: Mapping[str, Any],
    clock_audit: Mapping[str, Any],
    off_audit: Mapping[str, Any],
    final_bootstrap: Mapping[str, Any],
    ema_bootstrap: Mapping[str, Any],
    h65_replay_identity: Mapping[str, Any],
    nominal_uniform_identity: Mapping[str, Any],
    expected_eval_commit: str,
    boundary: Mapping[str, Any] | None = None,
    boundary_missing_artifacts: tuple[str, ...] = (),
    old_pair_bootstrap: Mapping[str, Any] | None = None,
    strata: Mapping[str, Any] | None = None,
    cost: Mapping[str, Any] | None = None,
    stage1_average_map: float | None = None,
    implementation_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the accepted Unit-1 SingleClock terminal gate.

    Only the EMA ON-minus-H65-OFF point deltas, nominal-uniform bit identity,
    and an evaluable boundary gate may produce a scientific PASS/KILL.  All
    other readouts are diagnostics.  Evidence-binding failures return an
    INVALID artifact with no scientific decision token.
    """

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
    _require(
        isinstance(families, Mapping) and set(families) == required,
        "terminal receipt families differ from the frozen six readouts",
    )

    clock_checkpoint = receipt.get("clock_checkpoint", "")
    clock_checkpoint_sha256 = receipt.get("clock_checkpoint_sha256", "")
    off_checkpoint = receipt.get("h65_off_checkpoint", "")
    off_checkpoint_sha256 = receipt.get("h65_off_checkpoint_sha256", "")
    stage1_checkpoint = receipt.get("stage1_checkpoint", "")
    stage1_checkpoint_sha256 = receipt.get("stage1_checkpoint_sha256", "")
    for label, path, digest in (
        ("SingleClock", clock_checkpoint, clock_checkpoint_sha256),
        ("H65 OFF", off_checkpoint, off_checkpoint_sha256),
        ("Stage-1", stage1_checkpoint, stage1_checkpoint_sha256),
    ):
        if not _file_hash_matches(path, digest):
            return _invalid_result(
                first_failure="INVALID_CHECKPOINT_CONFIG_EVALUATOR_BINDING",
                diagnostics={"failed_checkpoint_binding": label},
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
    loaded_metrics: dict[str, Any] = {}
    execution_contract: dict[str, bool] = {}
    for family, spec in family_specs.items():
        row = families[family]
        if not isinstance(row, Mapping):
            return _invalid_result(
                first_failure="INVALID_CHECKPOINT_CONFIG_EVALUATOR_BINDING",
                diagnostics={"invalid_family_row": family},
            )
        metrics_path = row.get("metrics_path", "")
        if not _file_hash_matches(metrics_path, row.get("metrics_sha256", "")):
            return _invalid_result(
                first_failure="INVALID_CHECKPOINT_CONFIG_EVALUATOR_BINDING",
                diagnostics={"metrics_hash_mismatch": family},
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
    if not all(execution_contract.values()):
        return _invalid_result(
            first_failure="INVALID_CHECKPOINT_CONFIG_EVALUATOR_BINDING",
            diagnostics={"family_execution_contract": execution_contract},
        )

    selected_input_identity: dict[str, bool] = {}
    twin_execution_contract: dict[str, bool] = {}
    for prefix in ("final", "ema"):
        on_row = families[f"{prefix}_on"]
        zero_row = families[f"{prefix}_gate_zero"]
        twin_execution_contract[prefix] = _twin_execution_contract_ok(
            on_row, zero_row
        )
        on_metrics = loaded_metrics[f"{prefix}_on"]
        zero_metrics = loaded_metrics[f"{prefix}_gate_zero"]
        for key in (
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_epoch",
            "checkpoint_state_key",
        ):
            if on_metrics.get(key) != zero_metrics.get(key):
                return _invalid_result(
                    first_failure="INVALID_SAME_CHECKPOINT_GATE_ZERO_EXECUTION",
                    diagnostics={"twin": prefix, "mismatched_field": key},
                )
        on = _load(on_row["selected_input_identity_path"])
        zero = _load(zero_row["selected_input_identity_path"])
        selected_input_identity[prefix] = _identity_equal(on, zero)
    if not all(twin_execution_contract.values()) or not all(
        selected_input_identity.values()
    ):
        return _invalid_result(
            first_failure="INVALID_SAME_CHECKPOINT_GATE_ZERO_EXECUTION",
            diagnostics={
                "selected_input_identity": selected_input_identity,
                "twin_execution_contract": twin_execution_contract,
            },
        )

    h65_replay_pass = _h65_replay_identity_ok(
        h65_replay_identity,
        expected_checkpoint_sha256=str(off_checkpoint_sha256),
        expected_bindings={
            "config_sha256": str(families["h65_off_ema"]["config_sha256"]),
            "annotation_sha256": str(
                loaded_metrics["h65_off_ema"].get(
                    "evaluation_annotation_sha256", ""
                )
            ),
            "class_map_sha256": str(
                loaded_metrics["h65_off_ema"].get(
                    "evaluation_class_map_sha256", ""
                )
            ),
            "evaluator_sha256": str(
                (
                    loaded_metrics["h65_off_ema"].get("evaluator") or {}
                ).get("source_sha256", "")
            ),
            "evaluation_config_sha256": str(
                loaded_metrics["h65_off_ema"].get(
                    "evaluation_config_sha256", ""
                )
            ),
        },
    )
    if not h65_replay_pass:
        return _invalid_result(
            first_failure="INVALID_H65_REPLAY_IDENTITY",
            diagnostics={"h65_replay_five_boundary_pass": False},
        )
    nominal_uniform_pass = _nominal_uniform_identity_ok(
        nominal_uniform_identity,
        expected_checkpoint_sha256=str(clock_checkpoint_sha256),
    )

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

    old = None
    if old_pair_bootstrap is not None:
        old = {
            metric: _paired_delta(
                old_pair_bootstrap, "truetime", "rankpack", metric
            )
            for metric in METRICS
        }

    ema_external = estimates["ema"]["external"]
    point_estimates = ema_bootstrap["point_estimates"]
    primary_metrics: dict[str, Any] = {}
    metric_pass = True
    for metric in METRICS:
        delta_pp = (
            _metric_decimal(point_estimates["ema_on"], metric)
            - _metric_decimal(point_estimates["h65_off_ema"], metric)
        ) * Decimal("100")
        passed = delta_pp >= NONINFERIORITY_MARGIN_PP
        metric_pass = metric_pass and passed
        primary_metrics[metric] = {
            "point_delta_pp": float(delta_pp),
            "point_delta_pp_decimal": format(delta_pp, "f"),
            "point_gate_pass": passed,
            "ci_lower_pp_report_only": ema_external[metric]["ci_lower_pp"],
            "ci_upper_pp_report_only": ema_external[metric]["ci_upper_pp"],
        }

    boundary_gate = _boundary_gate(
        boundary,
        missing_artifacts=boundary_missing_artifacts,
    )
    boundary_evaluable = boundary_gate["status"] == "EVALUABLE"
    boundary_pass = bool(
        not boundary_evaluable
        or (
            boundary_gate["high_gapcv_pass"]
            and boundary_gate["high_boundary_density_pass"]
        )
    )
    kill = bool(
        not nominal_uniform_pass
        or not metric_pass
        or (boundary_evaluable and not boundary_pass)
    )
    decision = (
        "KILL_SINGLECLOCK_REPRESENTATION"
        if kill
        else "PASS_UNIT1_SINGLECLOCK_GATE"
    )
    _require(decision in VALID_DECISIONS, "illegal Unit-1 decision token")

    cost_report: dict[str, Any] = {
        "decision_role": "report_only",
        "status": "NOT_AVAILABLE" if cost is None else "AVAILABLE",
    }
    if cost is not None:
        cost_report["artifact"] = dict(cost)
    review_pass = bool(
        implementation_review is not None
        and implementation_review.get("schema_version")
        == "duca_h65_singleclock_unit1_gate_implementation_review_v1"
        and implementation_review.get("verdict")
        == "UNIT1_GATE_IMPLEMENTATION_PASS"
        and implementation_review.get("focused_tests_pass") is True
    )
    eligibility_blockers = []
    if decision != "PASS_UNIT1_SINGLECLOCK_GATE":
        eligibility_blockers.append("unit1_gate_not_passed")
    if not review_pass:
        eligibility_blockers.append("independent_implementation_review_not_passed")

    diagnostics = {
        "final_on_vs_h65_off": estimates["final"]["external"],
        "ema_on_vs_same_checkpoint_gatezero": estimates["ema"]["single_clock"],
        "final_on_vs_same_checkpoint_gatezero": estimates["final"]["single_clock"],
        "gatezero_vs_h65_off_coadaptation": estimates["ema"]["coadaptation"],
        "old_rankpack_truetime": old,
        "legacy_strata": None if strata is None else dict(strata),
        "checkpoint_audit_gate_pass": _audit_ok(clock_audit, off_audit),
        "clock_recovery_contract_pass": bool(
            clock_audit.get("recovery_state_complete")
        ),
        "h65_off_recovery_contract_pass": bool(
            off_audit.get("recovery_state_complete")
        ),
        "h65_off_recovery_protocol_deviation": list(
            off_audit.get("recovery_protocol_deviation", ())
        ),
        "stage1_average_map": (
            None if stage1_average_map is None else float(stage1_average_map)
        ),
        "h65_off_final_average_map": _metric(
            final_bootstrap["point_estimates"]["h65_off_final"],
            "average_mAP",
        ),
        "h65_off_ema_average_map": _metric(
            ema_bootstrap["point_estimates"]["h65_off_ema"],
            "average_mAP",
        ),
    }

    first_failure = None
    if not nominal_uniform_pass:
        first_failure = "NOMINAL_UNIFORM_BIT_IDENTITY_FAILURE"
    elif not metric_pass:
        first_failure = next(
            f"PRIMARY_NONINFERIORITY_FAILURE:{metric}"
            for metric in METRICS
            if not primary_metrics[metric]["point_gate_pass"]
        )
    elif boundary_evaluable and not boundary_pass:
        first_failure = "BOUNDARY_RISK_FAILURE"

    return {
        "schema_version": "duca_h65_singleclock_unit1_terminal_gate_v2",
        "evidence_status": "VALID",
        "decision_token": decision,
        "primary_checkpoint_state_key": "state_dict_ema",
        "primary_comparison": "ema_on_minus_h65_off_ema",
        "identity": {
            "h65_replay_five_boundary_pass": h65_replay_pass,
            "same_checkpoint_gatezero_execution_pass": True,
            "nominal_uniform_backbone_bit_identical": nominal_uniform_pass,
        },
        "thresholds_pp": {
            **{
                metric: float(NONINFERIORITY_MARGIN_PP)
                for metric in METRICS
            },
            "comparison": "inclusive_point_estimate",
        },
        "primary_metrics": primary_metrics,
        "boundary_gate": boundary_gate,
        "diagnostics": diagnostics,
        "cost": cost_report,
        "claim_boundary": "single_seed_unit1_development_representation_gate_only",
        "paper_claim_admissible": False,
        "unit2_query_builder_eligible": bool(
            decision == "PASS_UNIT1_SINGLECLOCK_GATE" and review_pass
        ),
        "unit2_eligibility_blockers": eligibility_blockers,
        "dynamic_k_authorized": False,
        "first_failure": first_failure,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Finalize the frozen H65 SingleClock terminal gate")
    parser.add_argument("--terminal-receipt", required=True)
    parser.add_argument("--clock-audit", required=True)
    parser.add_argument("--off-audit", required=True)
    parser.add_argument("--final-bootstrap", required=True)
    parser.add_argument("--ema-bootstrap", required=True)
    parser.add_argument("--h65-replay-identity", required=True)
    parser.add_argument("--nominal-uniform-identity", required=True)
    parser.add_argument("--boundary")
    parser.add_argument("--boundary-missing-artifact", action="append", default=[])
    parser.add_argument("--old-pair-bootstrap")
    parser.add_argument("--strata")
    parser.add_argument("--cost")
    parser.add_argument("--implementation-review")
    parser.add_argument("--stage1-average-map", type=float)
    parser.add_argument("--expected-eval-commit", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        payload = finalize(
            receipt=_load(args.terminal_receipt),
            clock_audit=_load(args.clock_audit),
            off_audit=_load(args.off_audit),
            final_bootstrap=_load(args.final_bootstrap),
            ema_bootstrap=_load(args.ema_bootstrap),
            h65_replay_identity=_load(args.h65_replay_identity),
            nominal_uniform_identity=_load(args.nominal_uniform_identity),
            boundary=None if args.boundary is None else _load(args.boundary),
            boundary_missing_artifacts=tuple(args.boundary_missing_artifact),
            old_pair_bootstrap=(
                None
                if args.old_pair_bootstrap is None
                else _load(args.old_pair_bootstrap)
            ),
            strata=None if args.strata is None else _load(args.strata),
            cost=None if args.cost is None else _load(args.cost),
            stage1_average_map=args.stage1_average_map,
            implementation_review=(
                None
                if args.implementation_review is None
                else _load(args.implementation_review)
            ),
            expected_eval_commit=args.expected_eval_commit,
        )
    except (KeyError, TypeError, ValueError) as error:
        payload = _invalid_result(
            first_failure=f"INVALID_EVIDENCE_BINDING:{error}",
        )
    atomic_write_json(args.output, payload)
    if payload.get("evidence_status") == "INVALID":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
