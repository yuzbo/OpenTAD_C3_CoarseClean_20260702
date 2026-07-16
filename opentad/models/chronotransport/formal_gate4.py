"""Immutable evidence and adjudication contracts for formal Gate 4."""

from __future__ import annotations

import copy
import hashlib
import math
from statistics import median
from typing import Any, Mapping, Sequence

from .actions import ChronoAction
from .environment import validate_observed_environment
from .gate1_unlock import GATE1_UNLOCK_SCHEMA
from .gate4 import ARMS, ARM_ORDERS, SEEDS, _adjudicate_gate4_statistics
from .gate4_population import validate_gate4_population_artifact
from .gates23 import SCHEDULER_EPSILON
from .post_stage_c import validate_post_stage_c_gate3_unlock
from .protocol import R2_PROTOCOL_ID, canonical_sha256
from .registration import (
    claim_flags,
    validate_formal_gate1_context,
    validate_pre_gate1_registration,
)
from .scheduler import R2_NON_DENSE_NAMES


GATE4_TIMING_EVIDENCE_SCHEMA = "chronotransport-r2-gate4-timing-evidence-v1"
GATE4_METRIC_EVIDENCE_SCHEMA = "chronotransport-r2-gate4-metric-evidence-envelope-v1"
GATE4_REGRET_EVIDENCE_SCHEMA = "chronotransport-r2-gate4-regret-evidence-v1"
GATE4_SEED_SHARD_SCHEMA = "chronotransport-r2-gate4-seed-shard-v1"
GATE4_FORMAL_REPORT_SCHEMA = "chronotransport-r2-gate4-formal-report-v1"
GATE4_TERMINAL_SCHEMA = "chronotransport-r2-gate4-terminal-v1"
GATE4_PRODUCER_IDENTITY = (
    "tools.bata.chronotransport_r2_gate4_factory:build_formal_gate4_evidence"
)
GATE4_ENERGY_ARM_ORDER_BY_SEED = {
    seed: ARM_ORDERS[index] for index, seed in enumerate(SEEDS)
}

_COMMON_FIELDS = {
    "protocol",
    "registration_sha256",
    "registration_commit",
    "population_artifact_sha256",
    "post_stage_c_gate3_unlock_sha256",
    "stage_c_bindings",
    "scheduler_contract",
    "seed_shard_artifact_sha256_by_seed",
    "observed_environment_by_seed",
    "producer_identity",
}
_TIMING_FIELDS = {
    "schema",
    *_COMMON_FIELDS,
    "row_count",
    "row_order_sha256",
    "rows",
    "execution_audit_count",
    "execution_audit_order_sha256",
    "execution_audit",
    "artifact_sha256",
}
_METRIC_FIELDS = {
    "schema",
    *_COMMON_FIELDS,
    "metric_evidence",
    "metric_evidence_sha256",
    "artifact_sha256",
}
_REGRET_FIELDS = {
    "schema",
    *_COMMON_FIELDS,
    "row_count",
    "row_order_sha256",
    "rows",
    "artifact_sha256",
}
_SEED_SHARD_FIELDS = {
    "schema",
    "protocol",
    "seed",
    "registration_sha256",
    "registration_commit",
    "population_artifact_sha256",
    "post_stage_c_gate3_unlock_sha256",
    "stage_c_binding",
    "scheduler_contract",
    "observed_environment",
    "power_sampling_hz",
    "power_samples",
    "power_trace_sha256",
    "energy_arm_order",
    "energy_blocks",
    "energy_block_order_sha256",
    "timing_rows",
    "execution_audit",
    "predictions",
    "regret_rows",
    "artifact_sha256",
}
_ENERGY_BLOCK_FIELDS = {
    "arm",
    "invocation_count",
    "invocation_order_sha256",
    "start_ms",
    "end_ms",
    "duration_ms",
    "energy_j",
    "post_nms_prediction_sha256",
}


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be full SHA-1")
    return value


def _copy_rows(rows: Sequence[Mapping[str, Any]], label: str) -> list[dict[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError(f"{label} rows must be a sequence")
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"{label} row {index} must be a mapping")
        result.append(copy.deepcopy(dict(row)))
    return result


def _validate_power_trace_cadence(samples: Sequence[Mapping[str, Any]]) -> None:
    gaps = [
        float(right["offset_ms"]) - float(left["offset_ms"])
        for left, right in zip(samples, samples[1:])
    ]
    if not gaps or max(gaps) > 250.0 or median(gaps) > 125.0:
        raise ValueError("formal Gate4 NVML trace missed the registered 10-Hz cadence")


def _single_invocation_action_sha256(batched_actions: Sequence[Any]) -> str:
    """Normalize batch-size-one runtime actions to the registered 2-D payload."""

    if (
        not isinstance(batched_actions, Sequence)
        or isinstance(batched_actions, (str, bytes))
        or len(batched_actions) != 1
    ):
        raise ValueError("formal Gate4 action evidence requires batch size one")
    actions = batched_actions[0]
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        raise TypeError("formal Gate4 action payload must be a sequence")
    return canonical_sha256(actions)


def _stage_c_bindings(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != {str(seed) for seed in SEEDS}:
        raise ValueError("formal Gate4 requires exactly three Stage-C bindings")
    result = {}
    expected = {
        "completion_artifact_sha256",
        "checkpoint_file_sha256",
        "checkpoint_provenance_sha256",
        "predictor_canonical_sha256",
        "fit_baseline_payload_sha256",
    }
    for seed in SEEDS:
        row = value[str(seed)]
        if not isinstance(row, Mapping) or set(row) != expected:
            raise ValueError(f"formal Gate4 Stage-C binding {seed} fields mismatch")
        result[str(seed)] = {
            field: _sha(row[field], f"formal Gate4 Stage-C {seed}/{field}")
            for field in sorted(expected)
        }
    return result


def _scheduler_contract(value: Any) -> dict[str, Any]:
    fields = {
        "budget",
        "epsilon",
        "calibration_frozen_static",
        "q_conf_by_seed",
        "gate1_unlock_artifact_sha256",
        "calibration_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("formal Gate4 scheduler contract fields mismatch")
    for field in ("budget", "epsilon"):
        number = value[field]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) <= 0.0
        ):
            raise ValueError(f"formal Gate4 scheduler {field} must be positive")
    if float(value["epsilon"]) != float(SCHEDULER_EPSILON):
        raise ValueError("formal Gate4 scheduler epsilon differs from r2")
    static = value["calibration_frozen_static"]
    if static not in R2_NON_DENSE_NAMES:
        raise ValueError("formal Gate4 static comparator is outside r2 library")
    q_conf = value["q_conf_by_seed"]
    if not isinstance(q_conf, Mapping) or set(q_conf) != {str(seed) for seed in SEEDS}:
        raise ValueError("formal Gate4 q_conf requires exactly three seeds")
    normalized_q = {}
    for seed in SEEDS:
        number = q_conf[str(seed)]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) < 0.0
        ):
            raise ValueError("formal Gate4 q_conf must be finite/non-negative")
        normalized_q[str(seed)] = float(number)
    return {
        "budget": float(value["budget"]),
        "epsilon": float(value["epsilon"]),
        "calibration_frozen_static": static,
        "q_conf_by_seed": normalized_q,
        "gate1_unlock_artifact_sha256": _sha(
            value["gate1_unlock_artifact_sha256"], "Gate1 unlock"
        ),
        "calibration_sha256": _sha(
            value["calibration_sha256"], "post-Stage-C calibration"
        ),
    }


def _common(
    *,
    registration_sha256: str,
    registration_commit: str,
    population_artifact_sha256: str,
    post_stage_c_gate3_unlock_sha256: str,
    stage_c_bindings: Mapping[str, Mapping[str, str]],
    scheduler_contract: Mapping[str, Any],
    seed_shard_artifact_sha256_by_seed: Mapping[str, str],
    observed_environment_by_seed: Mapping[str, Mapping[str, Any]],
    producer_identity: str = GATE4_PRODUCER_IDENTITY,
) -> dict[str, Any]:
    if producer_identity != GATE4_PRODUCER_IDENTITY:
        raise ValueError("formal Gate4 producer identity mismatch")
    if not isinstance(seed_shard_artifact_sha256_by_seed, Mapping) or set(
        seed_shard_artifact_sha256_by_seed
    ) != {str(seed) for seed in SEEDS}:
        raise ValueError("formal Gate4 requires exactly three seed shard hashes")
    shard_hashes = {
        str(seed): _sha(
            seed_shard_artifact_sha256_by_seed[str(seed)],
            f"formal Gate4 seed shard {seed}",
        )
        for seed in SEEDS
    }
    if not isinstance(observed_environment_by_seed, Mapping) or set(
        observed_environment_by_seed
    ) != {str(seed) for seed in SEEDS}:
        raise TypeError("formal Gate4 requires one observed environment per seed")
    observed = {}
    for seed in SEEDS:
        row = observed_environment_by_seed[str(seed)]
        if not isinstance(row, Mapping):
            raise TypeError("formal Gate4 observed environment must be a mapping")
        observed[str(seed)] = copy.deepcopy(dict(row))
        _sha(
            observed[str(seed)].get("observed_environment_sha256"),
            f"observed environment {seed}",
        )
    return {
        "protocol": R2_PROTOCOL_ID,
        "registration_sha256": _sha(registration_sha256, "registration"),
        "registration_commit": _commit(registration_commit, "registration commit"),
        "population_artifact_sha256": _sha(
            population_artifact_sha256, "Gate4 population"
        ),
        "post_stage_c_gate3_unlock_sha256": _sha(
            post_stage_c_gate3_unlock_sha256, "post-Stage-C Gate3 unlock"
        ),
        "stage_c_bindings": _stage_c_bindings(stage_c_bindings),
        "scheduler_contract": _scheduler_contract(scheduler_contract),
        "seed_shard_artifact_sha256_by_seed": shard_hashes,
        "observed_environment_by_seed": observed,
        "producer_identity": producer_identity,
    }


def build_gate4_timing_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    execution_audit: Sequence[Mapping[str, Any]],
    **identity: Any,
) -> dict[str, Any]:
    copied = _copy_rows(rows, "formal Gate4 timing")
    audit = _copy_rows(execution_audit, "formal Gate4 execution audit")
    artifact = {
        "schema": GATE4_TIMING_EVIDENCE_SCHEMA,
        **_common(**identity),
        "row_count": len(copied),
        "row_order_sha256": canonical_sha256(copied),
        "rows": copied,
        "execution_audit_count": len(audit),
        "execution_audit_order_sha256": canonical_sha256(audit),
        "execution_audit": audit,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def build_gate4_metric_evidence(
    metric_evidence: Mapping[str, Any], **identity: Any
) -> dict[str, Any]:
    if not isinstance(metric_evidence, Mapping):
        raise TypeError("formal Gate4 metric evidence must be a mapping")
    metric = copy.deepcopy(dict(metric_evidence))
    artifact = {
        "schema": GATE4_METRIC_EVIDENCE_SCHEMA,
        **_common(**identity),
        "metric_evidence": metric,
        "metric_evidence_sha256": canonical_sha256(metric),
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def build_gate4_regret_evidence(
    rows: Sequence[Mapping[str, Any]], **identity: Any
) -> dict[str, Any]:
    copied = _copy_rows(rows, "formal Gate4 regret")
    artifact = {
        "schema": GATE4_REGRET_EVIDENCE_SCHEMA,
        **_common(**identity),
        "row_count": len(copied),
        "row_order_sha256": canonical_sha256(copied),
        "rows": copied,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def _identity_from_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "registration_sha256": value["registration_sha256"],
        "registration_commit": value["registration_commit"],
        "population_artifact_sha256": value["population_artifact_sha256"],
        "post_stage_c_gate3_unlock_sha256": value[
            "post_stage_c_gate3_unlock_sha256"
        ],
        "stage_c_bindings": value["stage_c_bindings"],
        "scheduler_contract": value["scheduler_contract"],
        "seed_shard_artifact_sha256_by_seed": value[
            "seed_shard_artifact_sha256_by_seed"
        ],
        "observed_environment_by_seed": value["observed_environment_by_seed"],
        "producer_identity": value["producer_identity"],
    }


def build_gate4_seed_shard(
    *,
    seed: int,
    registration_sha256: str,
    registration_commit: str,
    population_artifact_sha256: str,
    post_stage_c_gate3_unlock_sha256: str,
    stage_c_binding: Mapping[str, str],
    scheduler_contract: Mapping[str, Any],
    observed_environment: Mapping[str, Any],
    power_sampling_hz: float,
    power_samples: Sequence[Mapping[str, Any]],
    energy_arm_order: Sequence[str],
    energy_blocks: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    execution_audit: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    regret_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if seed not in SEEDS or isinstance(seed, bool):
        raise ValueError("formal Gate4 seed shard seed mismatch")
    bindings = _stage_c_bindings(
        {str(item): stage_c_binding for item in SEEDS}
    )[str(seed)]
    if not isinstance(observed_environment, Mapping):
        raise TypeError("formal Gate4 seed environment must be a mapping")
    observed = copy.deepcopy(dict(observed_environment))
    _sha(observed.get("observed_environment_sha256"), "seed observed environment")
    if not isinstance(predictions, Mapping) or set(predictions) != set(ARMS):
        raise ValueError("formal Gate4 seed predictions require D/C/S arms")
    prediction_rows = {
        arm: _copy_rows(predictions[arm], f"formal Gate4 {arm} predictions")
        for arm in ARMS
    }
    if (
        isinstance(power_sampling_hz, bool)
        or not isinstance(power_sampling_hz, (int, float))
        or float(power_sampling_hz) != 10.0
    ):
        raise ValueError("formal Gate4 power sampling must equal 10 Hz")
    samples = _copy_rows(power_samples, "formal Gate4 power")
    if len(samples) < 2:
        raise ValueError("formal Gate4 power trace requires at least two samples")
    previous = -1.0
    for sample in samples:
        if set(sample) != {"offset_ms", "power_w"}:
            raise ValueError("formal Gate4 power sample fields mismatch")
        offset = sample["offset_ms"]
        power = sample["power_w"]
        if (
            isinstance(offset, bool)
            or not isinstance(offset, (int, float))
            or not math.isfinite(float(offset))
            or float(offset) < 0.0
            or float(offset) <= previous
            or isinstance(power, bool)
            or not isinstance(power, (int, float))
            or not math.isfinite(float(power))
            or float(power) < 0.0
        ):
            raise ValueError("formal Gate4 power samples must be ordered finite values")
        previous = float(offset)
    _validate_power_trace_cadence(samples)
    expected_energy_order = list(GATE4_ENERGY_ARM_ORDER_BY_SEED[seed])
    if (
        not isinstance(energy_arm_order, Sequence)
        or isinstance(energy_arm_order, (str, bytes))
        or list(energy_arm_order) != expected_energy_order
    ):
        raise ValueError("formal Gate4 energy arm order differs from the frozen Latin order")
    raw_energy = _copy_rows(energy_blocks, "formal Gate4 energy block")
    if len(raw_energy) != len(ARMS):
        raise ValueError("formal Gate4 requires exactly three long energy blocks")
    normalized_energy = []
    previous_end = -1.0
    for index, raw in enumerate(raw_energy):
        if set(raw) != _ENERGY_BLOCK_FIELDS:
            raise ValueError("formal Gate4 energy block fields mismatch")
        arm = raw["arm"]
        if arm != expected_energy_order[index]:
            raise ValueError("formal Gate4 energy block arm order mismatch")
        count = raw["invocation_count"]
        start = raw["start_ms"]
        end = raw["end_ms"]
        if (
            type(count) is not int
            or count < 200
            or isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(start) < 0.0
            or float(start) < previous_end
            or float(end) <= float(start)
            or float(end) - float(start) < 1000.0
        ):
            raise ValueError("formal Gate4 energy block must be a long ordered interval")
        inside = [
            sample
            for sample in samples
            if float(start) <= float(sample["offset_ms"]) <= float(end)
        ]
        if len(inside) < 10:
            raise ValueError("formal Gate4 energy block has insufficient 10-Hz support")
        duration = float(end) - float(start)
        energy = integrate_power_trace_j(samples, float(start), float(end))
        normalized = {
            "arm": arm,
            "invocation_count": count,
            "invocation_order_sha256": _sha(
                raw["invocation_order_sha256"], "formal Gate4 energy invocation order"
            ),
            "start_ms": float(start),
            "end_ms": float(end),
            "duration_ms": duration,
            "energy_j": energy,
            "post_nms_prediction_sha256": _sha(
                raw["post_nms_prediction_sha256"],
                "formal Gate4 energy post-NMS predictions",
            ),
        }
        if (
            not math.isclose(float(raw["duration_ms"]), duration, rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(float(raw["energy_j"]), energy, rel_tol=1e-9, abs_tol=1e-9)
        ):
            raise ValueError("formal Gate4 energy block summary differs from trace integration")
        normalized_energy.append(normalized)
        previous_end = float(end)
    artifact = {
        "schema": GATE4_SEED_SHARD_SCHEMA,
        "protocol": R2_PROTOCOL_ID,
        "seed": seed,
        "registration_sha256": _sha(registration_sha256, "registration"),
        "registration_commit": _commit(registration_commit, "registration commit"),
        "population_artifact_sha256": _sha(
            population_artifact_sha256, "Gate4 population"
        ),
        "post_stage_c_gate3_unlock_sha256": _sha(
            post_stage_c_gate3_unlock_sha256, "post-Stage-C Gate3 unlock"
        ),
        "stage_c_binding": bindings,
        "scheduler_contract": _scheduler_contract(scheduler_contract),
        "observed_environment": observed,
        "power_sampling_hz": 10.0,
        "power_samples": samples,
        "power_trace_sha256": canonical_sha256(samples),
        "energy_arm_order": expected_energy_order,
        "energy_blocks": normalized_energy,
        "energy_block_order_sha256": canonical_sha256(normalized_energy),
        "timing_rows": _copy_rows(timing_rows, "formal Gate4 seed timing"),
        "execution_audit": _copy_rows(
            execution_audit, "formal Gate4 seed execution audit"
        ),
        "predictions": prediction_rows,
        "regret_rows": _copy_rows(regret_rows, "formal Gate4 seed regret"),
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def validate_gate4_seed_shard(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SEED_SHARD_FIELDS:
        raise ValueError("formal Gate4 seed shard fields mismatch")
    if value["schema"] != GATE4_SEED_SHARD_SCHEMA or value["protocol"] != R2_PROTOCOL_ID:
        raise ValueError("formal Gate4 seed shard schema/protocol mismatch")
    rebuilt = build_gate4_seed_shard(
        seed=value["seed"],
        registration_sha256=value["registration_sha256"],
        registration_commit=value["registration_commit"],
        population_artifact_sha256=value["population_artifact_sha256"],
        post_stage_c_gate3_unlock_sha256=value[
            "post_stage_c_gate3_unlock_sha256"
        ],
        stage_c_binding=value["stage_c_binding"],
        scheduler_contract=value["scheduler_contract"],
        observed_environment=value["observed_environment"],
        power_sampling_hz=value["power_sampling_hz"],
        power_samples=value["power_samples"],
        energy_arm_order=value["energy_arm_order"],
        energy_blocks=value["energy_blocks"],
        timing_rows=value["timing_rows"],
        execution_audit=value["execution_audit"],
        predictions=value["predictions"],
        regret_rows=value["regret_rows"],
    )
    if rebuilt != dict(value):
        raise ValueError("formal Gate4 seed shard differs from exact recomputation")
    return rebuilt


def validate_gate4_timing_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TIMING_FIELDS:
        raise ValueError("formal Gate4 timing evidence fields mismatch")
    if value["schema"] != GATE4_TIMING_EVIDENCE_SCHEMA:
        raise ValueError("formal Gate4 timing evidence schema mismatch")
    rebuilt = build_gate4_timing_evidence(
        value["rows"],
        execution_audit=value["execution_audit"],
        **_identity_from_artifact(value),
    )
    if (
        value["row_count"] != len(value["rows"])
        or value["execution_audit_count"] != len(value["execution_audit"])
        or rebuilt != dict(value)
    ):
        raise ValueError("formal Gate4 timing evidence differs from exact recomputation")
    return rebuilt


def validate_gate4_metric_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _METRIC_FIELDS:
        raise ValueError("formal Gate4 metric evidence fields mismatch")
    if value["schema"] != GATE4_METRIC_EVIDENCE_SCHEMA:
        raise ValueError("formal Gate4 metric evidence schema mismatch")
    rebuilt = build_gate4_metric_evidence(
        value["metric_evidence"], **_identity_from_artifact(value)
    )
    if rebuilt != dict(value):
        raise ValueError("formal Gate4 metric evidence differs from exact recomputation")
    return rebuilt


def validate_gate4_regret_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _REGRET_FIELDS:
        raise ValueError("formal Gate4 regret evidence fields mismatch")
    if value["schema"] != GATE4_REGRET_EVIDENCE_SCHEMA:
        raise ValueError("formal Gate4 regret evidence schema mismatch")
    rebuilt = build_gate4_regret_evidence(value["rows"], **_identity_from_artifact(value))
    if value["row_count"] != len(value["rows"]) or rebuilt != dict(value):
        raise ValueError("formal Gate4 regret evidence differs from exact recomputation")
    return rebuilt


def _validate_evidence_chain(
    timing: Mapping[str, Any],
    metric: Mapping[str, Any],
    regret: Mapping[str, Any],
    *,
    registration: Mapping[str, Any],
    population: Mapping[str, Any],
    post_stage_c_unlock: Mapping[str, Any],
    post_stage_c_report: Mapping[str, Any],
    post_stage_c_replay: Mapping[str, Any],
    gate1_unlock: Mapping[str, Any],
    seed_shards: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    registered = validate_pre_gate1_registration(registration)
    official = validate_gate4_population_artifact(population)
    if official != registered["gate4_population"]["artifact"]:
        raise ValueError("formal Gate4 population differs from registration")
    unlock = validate_post_stage_c_gate3_unlock(
        post_stage_c_unlock,
        report=post_stage_c_report,
        replay=post_stage_c_replay,
    )
    if not isinstance(gate1_unlock, Mapping):
        raise TypeError("formal Gate4 Gate1 unlock must be a mapping")
    gate1_unsigned = dict(gate1_unlock)
    gate1_artifact_sha256 = gate1_unsigned.pop("artifact_sha256", None)
    gate1_result = gate1_unlock.get("gate1_result")
    if (
        gate1_unlock.get("schema") != GATE1_UNLOCK_SCHEMA
        or gate1_unlock.get("status") != "PASS"
        or gate1_unlock.get("oracle_headroom") is not True
        or gate1_artifact_sha256 != canonical_sha256(gate1_unsigned)
        or gate1_unlock.get("registration_sha256")
        != registered["registration_sha256"]
        or not isinstance(gate1_result, Mapping)
    ):
        raise ValueError("formal Gate4 requires an exact PASS Gate1 unlock")
    scheduler = _scheduler_contract(
        {
            "budget": gate1_result.get("budget"),
            "epsilon": SCHEDULER_EPSILON,
            "calibration_frozen_static": gate1_result.get(
                "calibration_frozen_static"
            ),
            "q_conf_by_seed": unlock.get("q_conf_by_seed"),
            "gate1_unlock_artifact_sha256": gate1_artifact_sha256,
            "calibration_sha256": unlock["post_stage_c_gate3_report_sha256"],
        }
    )
    if not isinstance(seed_shards, Mapping) or set(seed_shards) != {
        str(seed) for seed in SEEDS
    }:
        raise ValueError("formal Gate4 requires exactly three seed shards")
    shards = {
        str(seed): validate_gate4_seed_shard(seed_shards[str(seed)])
        for seed in SEEDS
    }
    shard_hashes = {
        str(seed): shards[str(seed)]["artifact_sha256"] for seed in SEEDS
    }
    for seed in SEEDS:
        shard = shards[str(seed)]
        if (
            shard["seed"] != seed
            or shard["registration_sha256"] != registered["registration_sha256"]
            or shard["registration_commit"] != unlock["registration_commit"]
            or shard["population_artifact_sha256"] != official["artifact_sha256"]
            or shard["post_stage_c_gate3_unlock_sha256"]
            != unlock["artifact_sha256"]
            or shard["stage_c_binding"] != unlock["stage_c_bindings"][str(seed)]
            or shard["scheduler_contract"] != scheduler
        ):
            raise ValueError(f"formal Gate4 seed shard {seed} chain mismatch")
        validate_observed_environment(
            shard["observed_environment"],
            required_environment=registered["environment"],
        )
    timing = validate_gate4_timing_evidence(timing)
    metric = validate_gate4_metric_evidence(metric)
    regret = validate_gate4_regret_evidence(regret)
    identities = {
        "protocol": R2_PROTOCOL_ID,
        "registration_sha256": registered["registration_sha256"],
        "registration_commit": unlock["registration_commit"],
        "population_artifact_sha256": official["artifact_sha256"],
        "post_stage_c_gate3_unlock_sha256": unlock["artifact_sha256"],
        "stage_c_bindings": unlock["stage_c_bindings"],
        "scheduler_contract": scheduler,
        "seed_shard_artifact_sha256_by_seed": shard_hashes,
    }
    for evidence in (timing, metric, regret):
        for field, expected in identities.items():
            if evidence[field] != expected:
                raise ValueError(f"formal Gate4 evidence chain mismatch: {field}")
        for seed in SEEDS:
            validate_observed_environment(
                evidence["observed_environment_by_seed"][str(seed)],
                required_environment=registered["environment"],
            )
    environments = [
        item["observed_environment_by_seed"] for item in (timing, metric, regret)
    ]
    if environments[1:] != environments[:-1]:
        raise ValueError(
            "formal Gate4 evidence envelopes must share the same per-seed environments"
        )
    expected_timing_rows = [
        row for seed in SEEDS for row in shards[str(seed)]["timing_rows"]
    ]
    expected_audit = [
        row for seed in SEEDS for row in shards[str(seed)]["execution_audit"]
    ]
    expected_regret = [
        row for seed in SEEDS for row in shards[str(seed)]["regret_rows"]
    ]
    expected_environments = {
        str(seed): shards[str(seed)]["observed_environment"] for seed in SEEDS
    }
    expected_predictions = {
        str(seed): shards[str(seed)]["predictions"] for seed in SEEDS
    }
    if (
        timing["rows"] != expected_timing_rows
        or timing["execution_audit"] != expected_audit
        or regret["rows"] != expected_regret
        or timing["observed_environment_by_seed"] != expected_environments
        or metric["metric_evidence"].get("predictions") != expected_predictions
    ):
        raise ValueError("formal Gate4 final evidence differs from immutable seed shards")
    energy_invocations = [row["invocation_id"] for row in official["unique_invocations"]]
    energy_invocation_order_sha256 = canonical_sha256(energy_invocations)
    for seed in SEEDS:
        shard = shards[str(seed)]
        samples = shard["power_samples"]
        _validate_power_trace_cadence(samples)
        energy_by_arm = {row["arm"]: row for row in shard["energy_blocks"]}
        for arm in ARMS:
            block = energy_by_arm[arm]
            if (
                block["invocation_count"] != len(energy_invocations)
                or block["invocation_order_sha256"]
                != energy_invocation_order_sha256
                or block["post_nms_prediction_sha256"]
                != canonical_sha256(shard["predictions"][arm])
            ):
                raise ValueError(
                    "formal Gate4 long energy block population/prediction mismatch"
                )
            for timing_row in shard["timing_rows"]:
                actual_energy = timing_row["arms"][arm]["nvml_energy_j"]
                if not math.isclose(
                    float(actual_energy),
                    float(block["energy_j"]),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        "formal Gate4 timing diagnostic differs from long-block energy"
                    )
    return timing, metric, regret


def integrate_power_trace_j(
    samples: Sequence[Mapping[str, Any]], start_ms: float, end_ms: float
) -> float:
    """Integrate one long arm block from the immutable 10-Hz NVML trace."""

    points = [(float(row["offset_ms"]), float(row["power_w"])) for row in samples]
    start = float(start_ms)
    end = float(end_ms)
    if (
        len(points) < 2
        or not math.isfinite(start)
        or not math.isfinite(end)
        or end <= start
        or start < points[0][0]
        or end > points[-1][0]
    ):
        raise ValueError("formal Gate4 power interval lies outside sampled trace")

    def interpolate(moment: float) -> float:
        for index in range(1, len(points)):
            left_t, left_p = points[index - 1]
            right_t, right_p = points[index]
            if moment <= right_t:
                if right_t == left_t:
                    return right_p
                weight = (moment - left_t) / (right_t - left_t)
                return left_p + weight * (right_p - left_p)
        return points[-1][1]

    clipped = [(start, interpolate(start))]
    clipped.extend((time_ms, power) for time_ms, power in points if start < time_ms < end)
    clipped.append((end, interpolate(end)))
    energy = 0.0
    for (left_t, left_p), (right_t, right_p) in zip(clipped, clipped[1:]):
        energy += 0.5 * (left_p + right_p) * (right_t - left_t) / 1000.0
    if not math.isfinite(energy) or energy < 0.0:
        raise ValueError("formal Gate4 integrated energy is invalid")
    return energy


def _validate_population_rows(
    timing: Mapping[str, Any],
    metric: Mapping[str, Any],
    regret: Mapping[str, Any],
    *,
    population: Mapping[str, Any],
    registration: Mapping[str, Any],
) -> None:
    blocks = population["timing_blocks"]
    timing_rows = timing["rows"]
    expected_timing = []
    for seed in SEEDS:
        expected_timing.extend(
            (
                seed,
                row["official_video_id"],
                row["invocation_id"],
                row["repetition_id"],
                row["invocation_order_index"],
                tuple(row["arm_order"]),
            )
            for row in blocks
        )
    actual_timing = [
        (
            row.get("seed"),
            row.get("official_video_id"),
            row.get("invocation_id"),
            row.get("repetition_id"),
            row.get("invocation_order_index"),
            tuple(row.get("arm_order", ())),
        )
        for row in timing_rows
    ]
    if actual_timing != expected_timing:
        raise ValueError("formal Gate4 timing rows differ from registered block order")
    expected_audit = [
        (seed, block["invocation_id"], block["repetition_id"], arm)
        for seed in SEEDS
        for block in blocks
        for arm in block["arm_order"]
    ]
    actual_audit = [
        (
            row.get("seed"),
            row.get("invocation_id"),
            row.get("repetition_id"),
            row.get("arm"),
        )
        for row in timing["execution_audit"]
    ]
    if actual_audit != expected_audit:
        raise ValueError("formal Gate4 execution audit differs from D/C/S block order")
    scheduler = timing["scheduler_contract"]
    library = registration["candidate_library"]
    candidates = {row["name"]: row for row in library["candidates"]}
    layer_depths = [int(end) - int(start) for start, end in library["layer_groups"]]

    def expected_runtime_counts(candidate: Mapping[str, Any]) -> dict[str, int]:
        counts = {action.name.lower(): 0 for action in ChronoAction}
        for action_row in candidate["actions"]:
            for group_index, action_value in enumerate(action_row):
                action = ChronoAction(int(action_value))
                counts[action.name.lower()] += layer_depths[group_index]
        return counts

    for row in timing["execution_audit"]:
        required = {
            "seed",
            "invocation_id",
            "repetition_id",
            "arm",
            "selected_schedule",
            "requested_action_sha256",
            "executed_action_sha256",
            "recompute_rows",
            "transport_rows",
            "hold_rows",
            "schedule_repair_count",
            "runtime_fail_closed_repairs",
            "whole_window_dense_fallback",
            "upper_risk",
            "estimated_cost",
            "registered_gate3_calibration_sha256",
            "registered_q_conf",
            "registered_budget",
            "evidence_valid",
            "fail_closed",
        }
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("formal Gate4 execution audit fields mismatch")
        arm = row["arm"]
        seed = row["seed"]
        upper = row["upper_risk"]
        cost = row["estimated_cost"]
        candidate = candidates.get(row["selected_schedule"])
        if candidate is None:
            raise ValueError("formal Gate4 selected schedule is outside registration")
        expected_counts = expected_runtime_counts(candidate)
        if (
            arm not in ARMS
            or seed not in SEEDS
            or row["schedule_repair_count"] != 0
            or row["runtime_fail_closed_repairs"] != 0
            or row["whole_window_dense_fallback"] is not False
            or row["evidence_valid"] is not True
            or row["requested_action_sha256"] != candidate["action_sha256"]
            or row["executed_action_sha256"] != candidate["action_sha256"]
            or row["recompute_rows"] != expected_counts["recompute"]
            or row["transport_rows"] != expected_counts["transport"]
            or row["hold_rows"] != expected_counts["hold"]
            or isinstance(upper, bool)
            or not isinstance(upper, (int, float))
            or not math.isfinite(float(upper))
            or float(upper) < 0.0
            or isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or float(cost) < 0.0
        ):
            raise ValueError(
                "formal Gate4 runtime audit differs from the registered executed schedule"
            )
        if arm == "dense":
            if (
                row["selected_schedule"] != "dense"
                or row["fail_closed"] is not False
                or row["transport_rows"] != 0
                or row["hold_rows"] != 0
                or row["recompute_rows"] <= 0
            ):
                raise ValueError("formal Gate4 matched-dense audit is not dense")
        elif arm == "static":
            if (
                row["selected_schedule"] != scheduler["calibration_frozen_static"]
                or row["fail_closed"] is not False
            ):
                raise ValueError("formal Gate4 static schedule differs from Gate1")
        else:
            dense_safety_fallback = (
                row["selected_schedule"] == "dense" and row["fail_closed"] is True
            )
            if (
                row["registered_q_conf"] != scheduler["q_conf_by_seed"][str(seed)]
                or row["registered_budget"] != scheduler["budget"]
                or (not dense_safety_fallback and float(upper) > scheduler["epsilon"])
                or (not dense_safety_fallback and float(cost) > scheduler["budget"])
                or row["registered_gate3_calibration_sha256"]
                != scheduler["calibration_sha256"]
                or (
                    row["selected_schedule"] != "dense"
                    and row["fail_closed"] is not False
                )
            ):
                raise ValueError("formal Gate4 learned scheduler violated Gate3 bounds")
    raw_metric = metric["metric_evidence"]
    if (
        raw_metric.get("official_video_ids") != population["official_video_ids"]
        or raw_metric.get("ground_truth") != population["ground_truth"]
        or raw_metric.get("fit_duration_quartile_thresholds")
        != population["fit_duration_quartile_thresholds"]
    ):
        raise ValueError("formal Gate4 metric population differs from registration")
    invocations = population["unique_invocations"]
    expected_regret = [
        (seed, row["official_video_id"], row["invocation_id"])
        for seed in SEEDS
        for row in invocations
    ]
    actual_regret = [
        (row.get("seed"), row.get("official_video_id"), row.get("invocation_id"))
        for row in regret["rows"]
    ]
    if actual_regret != expected_regret:
        raise ValueError("formal Gate4 regret rows differ from registered invocation order")


def adjudicate_formal_gate4(
    *,
    timing_evidence: Mapping[str, Any],
    metric_evidence: Mapping[str, Any],
    regret_evidence: Mapping[str, Any],
    registration: Mapping[str, Any],
    population: Mapping[str, Any],
    post_stage_c_unlock: Mapping[str, Any],
    post_stage_c_report: Mapping[str, Any],
    post_stage_c_replay: Mapping[str, Any],
    gate1_unlock: Mapping[str, Any],
    seed_shards: Mapping[str, Mapping[str, Any]],
    repository_root: str,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, Any]:
    """Adjudicate only exact repository-produced formal evidence."""

    registered = validate_formal_gate1_context(
        registration,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    timing, metric, regret = _validate_evidence_chain(
        timing_evidence,
        metric_evidence,
        regret_evidence,
        registration=registered,
        population=population,
        post_stage_c_unlock=post_stage_c_unlock,
        post_stage_c_report=post_stage_c_report,
        post_stage_c_replay=post_stage_c_replay,
        gate1_unlock=gate1_unlock,
        seed_shards=seed_shards,
    )
    official = validate_gate4_population_artifact(population)
    _validate_population_rows(
        timing,
        metric,
        regret,
        population=official,
        registration=registered,
    )
    report = _adjudicate_gate4_statistics(
        timing_rows=timing["rows"],
        metric_evidence=metric["metric_evidence"],
        regret_rows=regret["rows"],
        bootstrap_samples=5000,
        bootstrap_seed=20260711,
        report_schema=GATE4_FORMAL_REPORT_SCHEMA,
        evidence_scope="registered_official_full_video_sliding_window",
        formal_evidence=True,
    )
    report.pop("artifact_sha256")
    report["evidence"] = {
        "registration_sha256": timing["registration_sha256"],
        "registration_commit": timing["registration_commit"],
        "population_artifact_sha256": timing["population_artifact_sha256"],
        "post_stage_c_gate3_unlock_sha256": timing[
            "post_stage_c_gate3_unlock_sha256"
        ],
        "stage_c_bindings_sha256": canonical_sha256(timing["stage_c_bindings"]),
        "observed_environment_sha256_by_seed": {
            seed: row["observed_environment_sha256"]
            for seed, row in timing["observed_environment_by_seed"].items()
        },
        "energy_block_order_sha256_by_seed": {
            str(seed): seed_shards[str(seed)]["energy_block_order_sha256"]
            for seed in SEEDS
        },
        "timing_evidence_sha256": timing["artifact_sha256"],
        "metric_evidence_sha256": metric["artifact_sha256"],
        "regret_evidence_sha256": regret["artifact_sha256"],
    }
    passed = report["status"] == "PASS"
    report["claim_flags"] = claim_flags(
        gate1=True,
        gate2=True,
        gate3=True,
        gate4=passed,
    )
    report["artifact_sha256"] = canonical_sha256(report)
    return report


def validate_formal_gate4_report(report: Mapping[str, Any], **evidence: Any) -> dict[str, Any]:
    expected = adjudicate_formal_gate4(**evidence)
    if not isinstance(report, Mapping) or dict(report) != expected:
        raise ValueError("formal Gate4 report differs from exact recomputation")
    return expected


def build_gate4_terminal(
    *,
    report: Mapping[str, Any],
    timing_path: str,
    timing_file_sha256: str,
    metric_path: str,
    metric_file_sha256: str,
    regret_path: str,
    regret_file_sha256: str,
    report_path: str,
    report_file_sha256: str,
) -> dict[str, Any]:
    if report.get("schema") != GATE4_FORMAL_REPORT_SCHEMA:
        raise ValueError("Gate4 terminal requires a formal report")
    unsigned = dict(report)
    artifact_sha256 = unsigned.pop("artifact_sha256", None)
    if artifact_sha256 != canonical_sha256(unsigned):
        raise ValueError("Gate4 formal report artifact hash mismatch")
    references = {}
    for name, path, digest, evidence_sha in (
        (
            "timing",
            timing_path,
            timing_file_sha256,
            report["evidence"]["timing_evidence_sha256"],
        ),
        (
            "metric",
            metric_path,
            metric_file_sha256,
            report["evidence"]["metric_evidence_sha256"],
        ),
        (
            "regret",
            regret_path,
            regret_file_sha256,
            report["evidence"]["regret_evidence_sha256"],
        ),
        ("report", report_path, report_file_sha256, artifact_sha256),
    ):
        if not isinstance(path, str) or not path:
            raise ValueError(f"Gate4 terminal {name} path must be non-empty")
        references[name] = {
            "path": path,
            "file_sha256": _sha(digest, f"Gate4 terminal {name} file"),
            "artifact_sha256": _sha(evidence_sha, f"Gate4 terminal {name} artifact"),
        }
    terminal = {
        "schema": GATE4_TERMINAL_SCHEMA,
        "protocol": R2_PROTOCOL_ID,
        "status": "SUCCESS" if report["status"] == "PASS" else "FAIL",
        "registration_sha256": report["evidence"]["registration_sha256"],
        "registration_commit": report["evidence"]["registration_commit"],
        "references": references,
        "claim_flags": dict(report["claim_flags"]),
    }
    terminal["artifact_sha256"] = canonical_sha256(terminal)
    return terminal


def exact_json_file_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "GATE4_ENERGY_ARM_ORDER_BY_SEED",
    "GATE4_FORMAL_REPORT_SCHEMA",
    "GATE4_METRIC_EVIDENCE_SCHEMA",
    "GATE4_PRODUCER_IDENTITY",
    "GATE4_REGRET_EVIDENCE_SCHEMA",
    "GATE4_SEED_SHARD_SCHEMA",
    "GATE4_TERMINAL_SCHEMA",
    "GATE4_TIMING_EVIDENCE_SCHEMA",
    "adjudicate_formal_gate4",
    "build_gate4_metric_evidence",
    "build_gate4_regret_evidence",
    "build_gate4_seed_shard",
    "build_gate4_terminal",
    "build_gate4_timing_evidence",
    "integrate_power_trace_j",
    "validate_formal_gate4_report",
    "validate_gate4_metric_evidence",
    "validate_gate4_regret_evidence",
    "validate_gate4_seed_shard",
    "validate_gate4_timing_evidence",
]
