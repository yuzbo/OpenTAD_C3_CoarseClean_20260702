from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn

from .environment import (
    OBSERVED_PROVENANCE_FIELDS,
    REQUIRED_ENVIRONMENT_SCHEMA,
    observed_environment_from_provenance,
)
from .losses import R2StageBLosses, compose_r2_stage_b_loss
from .protocol import validate_stage_b_exposure_artifact
from .replay import validate_compact_record
from .scheduler import R2_NON_DENSE_NAMES


SPLIT_NAMES = ("fit", "calibration", "evaluation")


def _stable_hash(lines: Iterable[str]) -> str:
    payload = "".join(f"{line}\n" for line in sorted(map(str, lines))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_split_manifest(
    video_ids: Iterable[str],
    *,
    seed: int,
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
) -> dict[str, object]:
    ids = sorted(set(map(str, video_ids)))
    if len(ids) < len(SPLIT_NAMES):
        raise ValueError("formal Stage B requires at least three unique video ids")
    ratios = tuple(float(value) for value in ratios)
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError("split ratios must contain fit/calibration/evaluation values")
    if any((not math.isfinite(value)) or value <= 0.0 for value in ratios):
        raise ValueError("split ratios must be finite and positive")
    total_ratio = sum(ratios)
    normalized = tuple(value / total_ratio for value in ratios)

    raw_counts = [len(ids) * value for value in normalized]
    counts = [max(1, int(math.floor(value))) for value in raw_counts]
    while sum(counts) > len(ids):
        candidates = [index for index, value in enumerate(counts) if value > 1]
        if not candidates:
            raise ValueError("unable to allocate non-empty formal Stage-B splits")
        index = max(candidates, key=lambda item: counts[item] - raw_counts[item])
        counts[index] -= 1
    while sum(counts) < len(ids):
        index = max(
            range(len(counts)),
            key=lambda item: raw_counts[item] - counts[item],
        )
        counts[index] += 1

    ordered = sorted(
        ids,
        key=lambda video_id: hashlib.sha256(
            f"{int(seed)}\0{video_id}".encode("utf-8")
        ).hexdigest(),
    )
    splits: dict[str, list[str]] = {}
    cursor = 0
    for name, count in zip(SPLIT_NAMES, counts):
        splits[name] = sorted(ordered[cursor : cursor + count])
        cursor += count
    split_hashes = {name: _stable_hash(values) for name, values in splits.items()}
    manifest: dict[str, object] = {
        "schema_version": "chronotransport_stage_b_split_v1",
        "seed": int(seed),
        "ratios": dict(zip(SPLIT_NAMES, normalized)),
        "video_count": len(ids),
        "splits": splits,
        "split_hashes": split_hashes,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest


def validate_split_manifest(
    manifest: Mapping[str, object], *, expected_video_ids: Iterable[str] | None = None
) -> dict[str, object]:
    if manifest.get("schema_version") != "chronotransport_stage_b_split_v1":
        raise ValueError("unsupported formal Stage-B split manifest schema")
    raw_splits = manifest.get("splits")
    if not isinstance(raw_splits, Mapping):
        raise ValueError("formal Stage-B split manifest requires splits")
    splits = {name: list(map(str, raw_splits.get(name, ()))) for name in SPLIT_NAMES}
    if any(not values for values in splits.values()):
        raise ValueError("formal Stage-B splits must be non-empty")
    flattened = [value for name in SPLIT_NAMES for value in splits[name]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("formal Stage-B split ids must be disjoint")
    if expected_video_ids is not None and set(flattened) != set(map(str, expected_video_ids)):
        raise ValueError("formal Stage-B split manifest does not cover the dataset exactly")
    hashes = {name: _stable_hash(values) for name, values in splits.items()}
    if dict(manifest.get("split_hashes", {})) != hashes:
        raise ValueError("formal Stage-B split hashes do not match split ids")
    return {**dict(manifest), "splits": splits, "split_hashes": hashes}


def select_schedule_for_step(step: int, candidates: Sequence[str]) -> str:
    step = int(step)
    names = tuple(str(name) for name in candidates)
    if step <= 0:
        raise ValueError("training step must be positive")
    if not names or len(names) != len(set(names)):
        raise ValueError("training candidates must be non-empty and unique")
    return names[(step - 1) % len(names)]


def calibrate_stage_b_records(
    records: Sequence[Mapping[str, object]], *, coverage: float
) -> dict[str, object]:
    coverage = float(coverage)
    if not 0.0 < coverage < 1.0:
        raise ValueError("calibration coverage must lie in (0, 1)")
    residuals = []
    normalized = []
    for record in records:
        prediction = float(record["predicted_risk"])
        target = float(record["regret"])
        if not math.isfinite(prediction) or not math.isfinite(target):
            raise ValueError("calibration prediction and regret must be finite")
        normalized.append((prediction, target))
        residuals.append(max(target - prediction, 0.0))
    if not residuals:
        raise ValueError("calibration records must be non-empty")
    residuals.sort()
    rank = min(len(residuals), int(math.ceil((len(residuals) + 1) * coverage)))
    offset = residuals[rank - 1]
    empirical = sum(prediction + offset >= target for prediction, target in normalized) / len(
        normalized
    )
    return {
        "records": len(normalized),
        "target_coverage": coverage,
        "offset": float(offset),
        "coverage": float(empirical),
    }


def compact_stage_b_record(
    *,
    sample_id: str,
    split: str,
    schedule: str,
    predicted_risk: float,
    upper_risk: float,
    regret: float,
    feature_mse: float,
    dense_loss: float,
    counterfactual_loss: float,
    cost: Mapping[str, object],
) -> dict[str, object]:
    record = {
        "sample_id": str(sample_id),
        "split": str(split),
        "schedule": str(schedule),
        "signals": {
            "predicted_risk": float(predicted_risk),
            "upper_risk": float(upper_risk),
        },
        "pooled_targets": {
            "feature_mse": float(feature_mse),
            "dense_loss": float(dense_loss),
            "counterfactual_loss": float(counterfactual_loss),
        },
        "cost": dict(cost),
        "regret": float(regret),
    }
    numeric = (
        predicted_risk,
        upper_risk,
        regret,
        feature_mse,
        dense_loss,
        counterfactual_loss,
    )
    if any(not math.isfinite(float(value)) for value in numeric):
        raise ValueError("formal Stage-B compact record values must be finite")
    return validate_compact_record(record)


def save_calibrated_stage_b_checkpoint(
    source: Path | str,
    output: Path | str,
    *,
    calibration_offset: float,
    split_hashes: Mapping[str, str],
    p3_gate_status: str,
) -> None:
    source = Path(source)
    output = Path(output)
    calibration_offset = float(calibration_offset)
    if not math.isfinite(calibration_offset) or calibration_offset < 0.0:
        raise ValueError("calibration offset must be finite and non-negative")
    if str(p3_gate_status) not in {"PASS", "FAIL"}:
        raise ValueError("P3 gate status must be PASS or FAIL")
    checkpoint = torch.load(source, map_location="cpu")
    for state_key in ("state_dict", "state_dict_ema"):
        state = checkpoint.get(state_key)
        if not isinstance(state, Mapping):
            raise ValueError(f"training checkpoint requires {state_key}")
        matched = 0
        for name, value in state.items():
            if str(name).endswith(
                (
                    "risk_predictor.calibration_offset",
                    "scheduler.predictor.calibration_offset",
                )
            ):
                state[name] = torch.as_tensor(
                    calibration_offset,
                    dtype=value.dtype,
                    device=value.device,
                ).reshape(value.shape)
                matched += 1
        if matched == 0:
            raise ValueError(f"{state_key} has no ChronoTransport calibration offset")
    meta = dict(checkpoint.get("meta", {}))
    meta.update(
        chronotransport_stage="B",
        calibration_ready=True,
        measured_cost_ready=False,
        split_hashes=dict(split_hashes),
        calibration_offset=calibration_offset,
        p3_gate_status=str(p3_gate_status),
        deploy_claim_allowed=False,
        metric_claim_allowed=False,
        latency_claim_allowed=False,
        paper_claim_allowed=False,
    )
    checkpoint["meta"] = meta
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def _pearson(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or len(first) < 2:
        return 0.0
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    numerator = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first, second)
    )
    first_scale = math.sqrt(sum((value - first_mean) ** 2 for value in first))
    second_scale = math.sqrt(sum((value - second_mean) ** 2 for value in second))
    if first_scale == 0.0 or second_scale == 0.0:
        return 0.0
    return numerator / (first_scale * second_scale)


def spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    return _pearson(_average_ranks(first), _average_ranks(second))


def _bootstrap_mean_ci(
    values: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float]:
    values = tuple(float(value) for value in values)
    samples = int(samples)
    if not values or samples <= 0:
        raise ValueError("bootstrap requires non-empty values and positive samples")
    rng = random.Random(int(seed))
    means = []
    for _ in range(samples):
        means.append(sum(rng.choice(values) for _ in values) / len(values))
    means.sort()
    lower = means[int(math.floor(0.025 * (samples - 1)))]
    upper = means[int(math.ceil(0.975 * (samples - 1)))]
    return float(lower), float(upper)


def summarize_stage_b_evaluation(
    records: Sequence[Mapping[str, object]],
    *,
    coverage_target: float,
    min_spearman: float,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 3407,
) -> dict[str, object]:
    if not records:
        raise ValueError("formal Stage-B evaluation records must be non-empty")
    coverage_target = float(coverage_target)
    min_spearman = float(min_spearman)
    if not 0.0 < coverage_target < 1.0:
        raise ValueError("coverage target must lie in (0, 1)")
    required = {
        "sample_id",
        "schedule",
        "predicted_risk",
        "upper_risk",
        "regret",
        "feature_mse",
    }
    normalized = []
    for record in records:
        missing = required - set(record)
        if missing:
            raise ValueError(f"formal Stage-B evaluation record missing {sorted(missing)}")
        row = dict(record)
        for key in ("predicted_risk", "upper_risk", "regret", "feature_mse"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError("formal Stage-B evaluation values must be finite")
        normalized.append(row)

    coverage = sum(row["upper_risk"] >= row["regret"] for row in normalized) / len(normalized)
    correlation = spearman_correlation(
        [row["predicted_risk"] for row in normalized],
        [row["regret"] for row in normalized],
    )
    by_key = {(str(row["sample_id"]), str(row["schedule"])): row for row in normalized}
    transport_ids = {
        sample_id
        for sample_id, schedule in by_key
        if schedule == "periodic2_transport"
    }
    hold_ids = {
        sample_id for sample_id, schedule in by_key if schedule == "periodic2_hold"
    }
    paired_ids = sorted(transport_ids & hold_ids)
    if not paired_ids:
        raise ValueError("formal Stage-B gate requires paired periodic2 transport/hold records")
    regret_improvement = [
        by_key[(sample_id, "periodic2_hold")]["regret"]
        - by_key[(sample_id, "periodic2_transport")]["regret"]
        for sample_id in paired_ids
    ]
    feature_improvement = [
        by_key[(sample_id, "periodic2_hold")]["feature_mse"]
        - by_key[(sample_id, "periodic2_transport")]["feature_mse"]
        for sample_id in paired_ids
    ]
    regret_ci = _bootstrap_mean_ci(
        regret_improvement, samples=bootstrap_samples, seed=bootstrap_seed
    )
    feature_ci = _bootstrap_mean_ci(
        feature_improvement, samples=bootstrap_samples, seed=bootstrap_seed + 1
    )
    gates = {
        "coverage": coverage >= coverage_target,
        "risk_regret_spearman": correlation >= min_spearman,
        "transport_regret": regret_ci[0] > 0.0,
        "transport_feature": feature_ci[0] > 0.0,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "records": len(normalized),
        "coverage": float(coverage),
        "coverage_target": coverage_target,
        "risk_regret_spearman": float(correlation),
        "min_spearman": min_spearman,
        "transport_vs_hold": {
            "paired_samples": len(paired_ids),
            "mean_regret_improvement": sum(regret_improvement) / len(regret_improvement),
            "regret_improvement_ci95": list(regret_ci),
            "mean_feature_improvement": sum(feature_improvement) / len(feature_improvement),
            "feature_improvement_ci95": list(feature_ci),
        },
        "gates": gates,
    }


# CT-P3R-3S-r2 formal Stage B.  Legacy helpers above remain import-compatible,
# but this is the only implementation allowed for the frozen r2 protocol.
_R2_STAGE_B_CHECKPOINT_SCHEMA = "chronotransport-r2-stage-b-checkpoint-v1"
_R2_STAGE_B_LEDGER_SCHEMA = "chronotransport-r2-stage-b-ledger-row-v1"
_R2_STAGE_B_PHASE_COMPLETION_SCHEMA = (
    "chronotransport-r2-stage-b-phase-completion-v1"
)
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REGISTERED_PROVENANCE_KEYS = {
    "registration_sha256",
    "registration_commit",
    "spec_commit",
    "spec_sha256",
    "implementation_commit",
    "source_files_sha256",
    "upstream_commits_sha256",
    "split_hashes_sha256",
    "action_library_sha256",
    "environment_sha256",
    "cost_plan_sha256",
    "gate1_unlock_payload_sha256",
    "gate1_unlock_file_sha256",
    "gate1_status",
} | set(OBSERVED_PROVENANCE_FIELDS)
_STAGE_B_COUNTER_KEYS = {
    "attempted_optimizer_updates",
    "successful_optimizer_updates",
    "nonfinite_or_skipped_updates",
    "amp_skips",
    "ema_updates",
    "lr_scheduler_updates",
    "infrastructure_resume_events",
}
_STAGE_B_CHECKPOINT_KEYS = {
    "schema",
    "state_dict",
    "state_dict_ema",
    "optimizer",
    "lr_scheduler",
    "ema",
    "rng_state",
    "ledger_rows",
    "meta",
}
_STAGE_B_META_KEYS = {
    "protocol",
    "chronotransport_stage",
    "status",
    "seed",
    "successful_update_cursor",
    "counters",
    "ledger_sha256",
    "dense_checkpoint_path",
    "dense_checkpoint_sha256",
    "dense_checkpoint_state_key",
    "dense_checkpoint_state_sha256",
    "dense_checkpoint_bytes",
    "dense_checkpoint_top_level_keys_sha256",
    "manifest_sha256",
    "library_sha256",
    "config_sha256",
    "exposure_artifact_sha256",
    "transport_path",
    "risk_predictor_path",
    "scheduler_predictor_alias",
    "scheduler_predictor_paths",
    "parameter_aliases",
    "state_dict_sha256",
    "state_dict_ema_sha256",
    "ema_state_sha256",
    "registered_provenance",
    "registered_provenance_sha256",
    "optimizer",
    "loss",
    "ema_decay",
    "gradient_clip_norm",
    "amp_enabled",
    "calibration_ready",
    "measured_cost_ready",
    "deploy_claim_allowed",
    "paper_claim_allowed",
}
_STAGE_B_LEDGER_KEYS = {
    "schema",
    "seed",
    "successful_update",
    "canonical_window_index",
    "video_id",
    "window_id",
    "candidate_index",
    "candidate_name",
    "dense_checkpoint_path",
    "dense_checkpoint_sha256",
    "dense_checkpoint_state_key",
    "dense_checkpoint_state_sha256",
    "dense_checkpoint_bytes",
    "dense_checkpoint_top_level_keys_sha256",
    "manifest_sha256",
    "library_sha256",
    "config_sha256",
    "exposure_artifact_sha256",
    "materialized_window_sha256",
    "counterfactual_window_sha256",
    "augmentation_sha256",
    "requested_action_sha256",
    "executed_action_sha256",
    "registered_provenance_sha256",
    "loss_detector",
    "loss_transport",
    "loss_risk",
    "loss_total",
    "row_sha256",
}


def _require_sha256(value: str, *, field: str) -> str:
    value = str(value)
    if not _LOWER_SHA256.fullmatch(value):
        raise ValueError(f"{field} must be one lowercase SHA-256 digest")
    return value


def _validate_candidate_action_sha256_by_name(
    value: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(R2_NON_DENSE_NAMES):
        raise ValueError(
            "registered candidate action hashes must follow the exact 16-candidate order"
        )
    return {
        name: _require_sha256(
            value[name], field=f"registered action SHA-256 for {name}"
        )
        for name in R2_NON_DENSE_NAMES
    }


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_exact_bytes(value: Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return tensor.view(torch.uint8).numpy().tobytes(order="C")


def _state_dict_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(b"CT-P3R-3S-r2-state-dict-v1\0")
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError("checkpoint state_dict must contain only string-to-tensor entries")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(_canonical_json_bytes(list(value.shape)))
        digest.update(b"\0")
        digest.update(_tensor_exact_bytes(value))
        digest.update(b"\0")
    return digest.hexdigest()


def _strip_uniform_ddp_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    keys = tuple(state)
    if any(not isinstance(key, str) for key in keys):
        raise TypeError("checkpoint state_dict keys must be strings")
    prefixed = tuple(key.startswith("module.") for key in keys)
    if any(prefixed) and not all(prefixed):
        raise ValueError("checkpoint state_dict mixes DDP-prefixed and unprefixed keys")
    if keys and all(prefixed):
        return {key[7:]: value for key, value in state.items()}
    return dict(state)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validate_registered_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _REGISTERED_PROVENANCE_KEYS:
        raise ValueError("Stage-B registered provenance fields mismatch")
    normalized = dict(value)
    for field in (
        "registration_sha256",
        "spec_sha256",
        "source_files_sha256",
        "upstream_commits_sha256",
        "split_hashes_sha256",
        "action_library_sha256",
        "environment_sha256",
        "cost_plan_sha256",
        "gate1_unlock_payload_sha256",
        "gate1_unlock_file_sha256",
        "required_environment_sha256",
        "allocation_identity_sha256",
        "observed_environment_sha256",
    ):
        _require_sha256(str(normalized[field]), field=f"registered provenance {field}")
    for field in ("spec_commit", "implementation_commit", "registration_commit"):
        if not isinstance(normalized[field], str) or not _FULL_COMMIT.fullmatch(
            normalized[field]
        ):
            raise ValueError(f"registered provenance {field} must be a full commit")
    if normalized["gate1_status"] != "PASS":
        raise ValueError("registered provenance requires Gate-1 PASS unlock")
    required_environment = {
        "schema": REQUIRED_ENVIRONMENT_SCHEMA,
        "gpu_model": normalized["gpu_model"],
        "driver": normalized["driver"],
        "cuda": normalized["cuda"],
        "pytorch": normalized["pytorch"],
        "cudnn": normalized["cudnn"],
        "precision": normalized["precision"],
        "batch_size": normalized["batch_size"],
        "environment_sha256": normalized["environment_sha256"],
    }
    observed_environment_from_provenance(
        normalized,
        required_environment=required_environment,
    )
    return normalized


@dataclass
class StageBReplayOutput:
    counterfactual_task_loss: Tensor
    counterfactual_features: Tensor
    dense_features: Tensor
    predicted_quantile: Tensor
    regret_target: Tensor
    materialized_window_sha256: str
    counterfactual_window_sha256: str
    augmentation_sha256: str
    requested_action_sha256: str
    executed_action_sha256: str
    amp_skipped: bool = False


class FormalStageBInvalid(RuntimeError):
    def __init__(
        self,
        reason: str,
        *,
        counters: Mapping[str, int],
        successful_update_cursor: int,
    ) -> None:
        super().__init__(f"INVALID_IMPLEMENTATION: {reason}")
        self.counters = dict(counters)
        self.successful_update_cursor = int(successful_update_cursor)


class StageBUpdateState:
    """Successful-update indexed r2 exposure cursor.

    Failed/retried attempts are audit events only and never consume the next
    window or candidate.  Formal completion separately rejects every skip.
    """

    def __init__(
        self,
        *,
        seed: int,
        exposure_artifact: Mapping[str, Any],
        candidate_names: Sequence[str],
        successful_update_cursor: int = 0,
        counters: Mapping[str, int] | None = None,
    ) -> None:
        if type(seed) is not int:
            raise ValueError("formal Stage B seed must be an integer")
        self.seed = seed
        if self.seed not in (3407, 3408, 3409):
            raise ValueError("formal Stage B seed must be 3407, 3408, or 3409")
        self.candidate_names = tuple(map(str, candidate_names))
        if self.candidate_names != R2_NON_DENSE_NAMES:
            raise ValueError("formal Stage B requires the frozen 16-candidate order")
        matrices = exposure_artifact.get("matrices")
        if not isinstance(matrices, Mapping) or str(self.seed) not in matrices:
            raise ValueError("Stage-B exposure artifact does not contain the run seed")
        self.rows = tuple(dict(row) for row in matrices[str(self.seed)])
        if len(self.rows) != 140:
            raise ValueError("formal Stage B requires exactly 140 exposure rows")
        if type(successful_update_cursor) is not int:
            raise ValueError("Stage-B successful-update cursor must be an integer")
        self.successful_update_cursor = successful_update_cursor
        if not 0 <= self.successful_update_cursor <= 140:
            raise ValueError("Stage-B successful-update cursor is out of range")
        defaults = {
            "attempted_optimizer_updates": 0,
            "successful_optimizer_updates": 0,
            "nonfinite_or_skipped_updates": 0,
            "amp_skips": 0,
            "ema_updates": 0,
            "lr_scheduler_updates": 0,
            "infrastructure_resume_events": 0,
        }
        if counters is not None:
            if not isinstance(counters, Mapping) or set(counters) != set(defaults):
                raise ValueError("Stage-B counters must have the exact frozen key set")
            for name, value in counters.items():
                if type(value) is not int or int(value) < 0:
                    raise ValueError(f"Stage-B counter {name} must be a non-negative integer")
                defaults[name] = int(value)
        self.counters = defaults
        if self.counters["successful_optimizer_updates"] != self.successful_update_cursor:
            raise ValueError("Stage-B success counter/cursor mismatch")

    def current(self) -> dict[str, Any]:
        if self.successful_update_cursor >= 140:
            raise StopIteration("Stage B already has 140 successful updates")
        row = dict(self.rows[self.successful_update_cursor])
        candidate_index = int(row["candidate"])
        row["candidate_index"] = candidate_index
        row["candidate_name"] = self.candidate_names[candidate_index]
        return row

    def record_retry(self, reason: str) -> None:
        if not str(reason):
            raise ValueError("retry reason must be non-empty")
        self.counters["infrastructure_resume_events"] += 1

    def record_skip(self, reason: str, *, amp_skip: bool = False) -> None:
        if not str(reason):
            raise ValueError("skip reason must be non-empty")
        self.counters["attempted_optimizer_updates"] += 1
        self.counters["nonfinite_or_skipped_updates"] += 1
        if amp_skip:
            self.counters["amp_skips"] += 1

    def record_success(self, *, ledger_row_sha256: str) -> None:
        _require_sha256(ledger_row_sha256, field="ledger row SHA-256")
        self.counters["attempted_optimizer_updates"] += 1
        self.counters["successful_optimizer_updates"] += 1
        self.counters["ema_updates"] += 1
        self.counters["lr_scheduler_updates"] += 1
        self.successful_update_cursor += 1

    def validate_complete(self) -> None:
        required = {
            "attempted_optimizer_updates": 140,
            "successful_optimizer_updates": 140,
            "nonfinite_or_skipped_updates": 0,
            "amp_skips": 0,
            "ema_updates": 140,
            "lr_scheduler_updates": 140,
        }
        actual = {name: self.counters[name] for name in required}
        if self.successful_update_cursor != 140 or actual != required:
            raise FormalStageBInvalid(
                f"formal completion counters mismatch: expected={required}, actual={actual}",
                counters=self.counters,
                successful_update_cursor=self.successful_update_cursor,
            )


def _named_modules_with_aliases(model: nn.Module) -> list[tuple[str, nn.Module]]:
    try:
        return list(model.named_modules(remove_duplicate=False))
    except TypeError:  # pragma: no cover - compatibility with old torch only
        return list(model.named_modules())


def _named_parameters_with_aliases(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    try:
        return list(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - compatibility with old torch only
        return list(model.named_parameters())


def _stage_b_parameter_identity(model: nn.Module) -> dict[str, Any]:
    modules = _named_modules_with_aliases(model)
    transports = [
        (name, module)
        for name, module in modules
        if name.endswith("chronotransport.transport")
    ]
    risks = [
        (name, module)
        for name, module in modules
        if name.endswith("chronotransport.risk_predictor")
    ]
    if len(transports) != 1 or len(risks) != 1:
        raise ValueError("formal Stage B requires exactly one transport and one risk_predictor")
    transport_path, transport = transports[0]
    risk_path, risk = risks[0]
    scheduler_predictors = [
        (name, module)
        for name, module in modules
        if name.endswith("chronotransport.scheduler.predictor")
    ]
    if scheduler_predictors and any(module is not risk for _, module in scheduler_predictors):
        raise ValueError("scheduler.predictor must alias the canonical risk_predictor")

    transport_params = tuple(transport.parameters())
    risk_params = tuple(risk.parameters())
    transport_ids = {id(parameter) for parameter in transport_params}
    risk_ids = {id(parameter) for parameter in risk_params}
    if not transport_params or not risk_params or transport_ids & risk_ids:
        raise ValueError("Stage-B transport/risk parameters must be non-empty and disjoint")
    all_named = _named_parameters_with_aliases(model)
    aliases: dict[int, list[str]] = {}
    for name, parameter in all_named:
        aliases.setdefault(id(parameter), []).append(name)
    canonical_names = {
        id(parameter): sorted(aliases[id(parameter)])[0]
        for parameter in (*transport_params, *risk_params)
    }
    return {
        "transport_path": transport_path,
        "risk_predictor_path": risk_path,
        "scheduler_predictor_alias": bool(scheduler_predictors),
        "scheduler_predictor_paths": sorted(name for name, _ in scheduler_predictors),
        "transport_parameters": transport_params,
        "risk_parameters": risk_params,
        "trainable_parameters": transport_params + risk_params,
        "trainable_ids": transport_ids | risk_ids,
        "canonical_names": canonical_names,
        "parameter_aliases": {
            canonical_names[parameter_id]: sorted(aliases[parameter_id])
            for parameter_id in sorted(canonical_names, key=lambda item: canonical_names[item])
        },
    }


def logical_risk_predictor_state_sha256(
    model: nn.Module, state_dict_ema: Mapping[str, Tensor]
) -> str:
    """Hash each logical risk-predictor parameter once after alias validation."""

    identity = _stage_b_parameter_identity(model)
    canonical_state: dict[str, Tensor] = {}
    for parameter in identity["risk_parameters"]:
        canonical_name = identity["canonical_names"][id(parameter)]
        aliases = identity["parameter_aliases"][canonical_name]
        if any(alias not in state_dict_ema for alias in aliases):
            raise ValueError(
                f"risk predictor EMA aliases are incomplete for {canonical_name!r}"
            )
        values = [state_dict_ema[alias] for alias in aliases]
        first = values[0]
        if not isinstance(first, Tensor) or any(
            not isinstance(value, Tensor)
            or value.dtype != first.dtype
            or tuple(value.shape) != tuple(first.shape)
            or _tensor_exact_bytes(value) != _tensor_exact_bytes(first)
            for value in values[1:]
        ):
            raise ValueError(
                f"risk predictor EMA aliases conflict for {canonical_name!r}"
            )
        canonical_state[canonical_name] = first
    if not canonical_state:
        raise ValueError("Stage-B checkpoint has no logical risk predictor state")
    return _state_dict_sha256(canonical_state)


def _load_registered_dense_checkpoint(
    model: nn.Module,
    *,
    checkpoint_path: Path,
    expected_sha256: str,
    use_ema: bool,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    if type(use_ema) is not bool:
        raise TypeError("dense checkpoint EMA selection must be a boolean config decision")
    checkpoint_path, checkpoint, checkpoint_bytes = _load_torch_regular_file(
        checkpoint_path, label="registered dense checkpoint"
    )
    actual_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("dense checkpoint SHA-256 mismatch")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("registered dense checkpoint root must be a mapping")
    state_key = "state_dict_ema" if use_ema else "state_dict"
    if state_key not in checkpoint or not isinstance(checkpoint[state_key], Mapping):
        raise ValueError(f"registered dense checkpoint requires mapping {state_key!r}")
    state = _strip_uniform_ddp_prefix(checkpoint[state_key])
    for canonical_name, aliases in identity["parameter_aliases"].items():
        present = [state[name] for name in aliases if name in state]
        if len(present) > 1:
            first = present[0]
            if any(
                first.dtype != value.dtype
                or tuple(first.shape) != tuple(value.shape)
                or not torch.equal(first, value)
                for value in present[1:]
            ):
                raise ValueError(
                    f"registered dense checkpoint has conflicting alias tensors for {canonical_name}"
                )
    state_sha256 = _state_dict_sha256(state)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("registered dense checkpoint does not strictly match the r2 model") from error
    loaded_state = model.state_dict()
    loaded_sha256 = _state_dict_sha256(loaded_state)
    if loaded_sha256 != state_sha256:
        raise ValueError("strictly loaded model state identity differs from checkpoint state identity")
    return {
        "dense_checkpoint_state_key": state_key,
        "dense_checkpoint_state_sha256": state_sha256,
        "dense_checkpoint_bytes": len(checkpoint_bytes),
        "dense_checkpoint_top_level_keys_sha256": _canonical_json_sha256(
            sorted(map(str, checkpoint))
        ),
    }


def _capture_registered_dense_frozen_state(
    model: nn.Module, identity: Mapping[str, Any]
) -> dict[str, Any]:
    state = model.state_dict()
    trainable_aliases = {
        alias
        for aliases in identity["parameter_aliases"].values()
        for alias in aliases
    }
    if not trainable_aliases.issubset(state):
        raise ValueError("registered dense state omits a trainable parameter alias")
    frozen = {
        name: value.detach().cpu().clone()
        for name, value in state.items()
        if name not in trainable_aliases
    }
    if not frozen:
        raise ValueError("registered dense model has no frozen parameter/buffer state")
    baseline = {
        "all_state_keys": tuple(sorted(state)),
        "frozen_state": frozen,
    }
    _validate_registered_dense_frozen_state(
        state,
        registered_dense_frozen_state=baseline,
        label="registered dense baseline",
    )
    return baseline


def _validate_registered_dense_frozen_state(
    state: Mapping[str, Any],
    *,
    registered_dense_frozen_state: Mapping[str, Any],
    label: str,
) -> None:
    if not isinstance(state, Mapping):
        raise ValueError(f"{label} must be a state mapping")
    expected_keys = tuple(registered_dense_frozen_state["all_state_keys"])
    if tuple(sorted(state)) != expected_keys:
        raise ValueError(f"{label} state keys differ from registered dense")
    frozen = registered_dense_frozen_state["frozen_state"]
    for name, expected in frozen.items():
        actual = state.get(name)
        if not isinstance(actual, Tensor):
            raise ValueError(f"{label} frozen state {name!r} is missing or non-tensor")
        if actual.dtype != expected.dtype or tuple(actual.shape) != tuple(expected.shape):
            raise ValueError(
                f"{label} frozen state {name!r} dtype/shape differs from registered dense"
            )
        if (actual.is_floating_point() or actual.is_complex()) and not torch.isfinite(
            actual
        ).all():
            raise ValueError(f"{label} frozen state {name!r} is non-finite")
        if _tensor_exact_bytes(actual) != _tensor_exact_bytes(expected):
            raise ValueError(
                f"{label} frozen state {name!r} bytes differ from registered dense"
            )


def _configure_stage_b_model(model: nn.Module, identity: Mapping[str, Any]) -> None:
    trainable_ids = set(identity["trainable_ids"])
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in trainable_ids
    model.eval()
    modules = dict(_named_modules_with_aliases(model))
    modules[str(identity["transport_path"])].train()
    modules[str(identity["risk_predictor_path"])].train()
    actual = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if actual != trainable_ids:
        raise RuntimeError("formal Stage B trainable parameter identity mismatch")


def _floating_tensors(value: Any) -> Iterable[Tensor]:
    if isinstance(value, Tensor):
        if value.is_floating_point():
            yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _floating_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _floating_tensors(item)
    elif hasattr(value, "__dataclass_fields__"):
        yield from _floating_tensors(vars(value))


def _assert_fp32_tensors(value: Any, *, label: str) -> None:
    wrong = sorted({str(tensor.dtype) for tensor in _floating_tensors(value) if tensor.dtype != torch.float32})
    if wrong:
        raise ValueError(f"{label} must be strictly FP32; observed {wrong}")


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("resume checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _ema_initialize(identity: Mapping[str, Any]) -> dict[str, Tensor]:
    return {
        identity["canonical_names"][id(parameter)]: parameter.detach().cpu().clone()
        for parameter in identity["trainable_parameters"]
    }


def _ema_update(
    ema: dict[str, Tensor], identity: Mapping[str, Any], *, decay: float = 0.999
) -> None:
    with torch.no_grad():
        for parameter in identity["trainable_parameters"]:
            name = identity["canonical_names"][id(parameter)]
            current = parameter.detach().cpu()
            ema[name].mul_(decay).add_(current, alpha=1.0 - decay)


def _ema_state_dict(
    model: nn.Module, ema: Mapping[str, Tensor], identity: Mapping[str, Any]
) -> dict[str, Tensor]:
    result = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    aliases = identity["parameter_aliases"]
    for canonical_name, alias_names in aliases.items():
        for alias_name in alias_names:
            if alias_name in result:
                result[alias_name] = ema[canonical_name].clone()
    return result


def _path_without_symlink_components(
    path: Path | str, *, label: str, allow_missing: bool = False
) -> Path:
    """Return an absolute lexical path after checking every existing component."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if allow_missing:
                break
            raise FileNotFoundError(current) from None
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} contains a symlink component: {current}")
    return absolute


def _read_regular_file_bytes(path: Path | str, *, label: str) -> tuple[Path, bytes]:
    """Read one exact regular inode without following a symlink or replacement."""

    exact = _path_without_symlink_components(path, label=label)
    before = os.lstat(exact)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(exact, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise RuntimeError(f"{label} changed identity while being read")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = os.lstat(exact)
        if after.st_dev != opened.st_dev or after.st_ino != opened.st_ino:
            raise RuntimeError(f"{label} changed identity while being read")
    finally:
        os.close(descriptor)
    return exact, payload


def _load_torch_regular_file(
    path: Path | str, *, label: str
) -> tuple[Path, Any, bytes]:
    exact, payload = _read_regular_file_bytes(path, label=label)
    try:
        checkpoint = torch.load(io.BytesIO(payload), map_location="cpu")
    except Exception as error:
        raise ValueError(f"{label} is not a valid torch checkpoint") from error
    return exact, checkpoint, payload


def _atomic_torch_save(payload: Mapping[str, Any], output: Path) -> None:
    output = _path_without_symlink_components(
        output, label="Stage-B checkpoint path", allow_missing=True
    )
    parent = _path_without_symlink_components(
        output.parent, label="Stage-B checkpoint parent", allow_missing=True
    )
    parent.mkdir(parents=True, exist_ok=True)
    parent = _path_without_symlink_components(
        parent, label="Stage-B checkpoint parent"
    )
    output = _path_without_symlink_components(
        output, label="Stage-B checkpoint path", allow_missing=True
    )
    handle, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        _fsync_directory(parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _reuse_exact_regular_file(path: Path, payload: bytes, *, label: str) -> bool:
    """Accept an interrupted publication only when the existing inode is exact."""

    try:
        before = os.lstat(path)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return False
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise RuntimeError(f"{label} changed identity while being verified")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            existing = stream.read()
        after = os.lstat(path)
        if after.st_dev != opened.st_dev or after.st_ino != opened.st_ino:
            raise RuntimeError(f"{label} changed identity while being verified")
    finally:
        os.close(descriptor)
    if existing != payload:
        raise FileExistsError(f"{label} already exists with different bytes: {path}")
    return True


def _atomic_write_ledger(rows: Sequence[Mapping[str, Any]], output: Path) -> str:
    payload = b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)
    digest = hashlib.sha256(payload).hexdigest()
    output = _path_without_symlink_components(
        output, label="Stage-B ledger path", allow_missing=True
    )
    parent = _path_without_symlink_components(
        output.parent, label="Stage-B ledger parent", allow_missing=True
    )
    parent.mkdir(parents=True, exist_ok=True)
    parent = _path_without_symlink_components(parent, label="Stage-B ledger parent")
    output = _path_without_symlink_components(
        output, label="Stage-B ledger path", allow_missing=True
    )
    if _reuse_exact_regular_file(output, payload, label="Stage-B ledger"):
        return digest
    handle, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            if not _reuse_exact_regular_file(output, payload, label="Stage-B ledger"):
                raise error
        else:
            _fsync_directory(parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return digest


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: Mapping[str, Tensor],
    identity: Mapping[str, Any],
    state: StageBUpdateState,
    ledger_rows: Sequence[Mapping[str, Any]],
    ledger_sha256: str,
    provenance: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    state_dict = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    state_dict_ema = _ema_state_dict(model, ema, identity)
    ema_state = {name: value.clone() for name, value in ema.items()}
    return {
        "schema": _R2_STAGE_B_CHECKPOINT_SCHEMA,
        "state_dict": state_dict,
        "state_dict_ema": state_dict_ema,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "ema": ema_state,
        "rng_state": _capture_rng_state(),
        "ledger_rows": list(ledger_rows),
        "meta": {
            "protocol": "CT-P3R-3S-r2",
            "chronotransport_stage": "B",
            "status": str(status),
            "seed": state.seed,
            "successful_update_cursor": state.successful_update_cursor,
            "counters": dict(state.counters),
            "ledger_sha256": ledger_sha256,
            "dense_checkpoint_path": provenance["dense_checkpoint_path"],
            "dense_checkpoint_sha256": provenance["dense_checkpoint_sha256"],
            "dense_checkpoint_state_key": provenance["dense_checkpoint_state_key"],
            "dense_checkpoint_state_sha256": provenance["dense_checkpoint_state_sha256"],
            "dense_checkpoint_bytes": provenance["dense_checkpoint_bytes"],
            "dense_checkpoint_top_level_keys_sha256": provenance[
                "dense_checkpoint_top_level_keys_sha256"
            ],
            "manifest_sha256": provenance["manifest_sha256"],
            "library_sha256": provenance["library_sha256"],
            "config_sha256": provenance["config_sha256"],
            "exposure_artifact_sha256": provenance["exposure_artifact_sha256"],
            "transport_path": identity["transport_path"],
            "risk_predictor_path": identity["risk_predictor_path"],
            "scheduler_predictor_alias": identity["scheduler_predictor_alias"],
            "scheduler_predictor_paths": identity["scheduler_predictor_paths"],
            "parameter_aliases": identity["parameter_aliases"],
            "state_dict_sha256": _state_dict_sha256(state_dict),
            "state_dict_ema_sha256": _state_dict_sha256(state_dict_ema),
            "ema_state_sha256": _state_dict_sha256(ema_state),
            "registered_provenance": provenance["registered_provenance"],
            "registered_provenance_sha256": provenance[
                "registered_provenance_sha256"
            ],
            "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 0.0},
            "loss": {
                "lambda_transport": 0.1,
                "lambda_risk": 0.1,
                "risk_quantile": 0.9,
                "feature_loss": "elementwise_mean_mse",
            },
            "ema_decay": 0.999,
            "gradient_clip_norm": 1.0,
            "amp_enabled": False,
            "calibration_ready": False,
            "measured_cost_ready": False,
            "deploy_claim_allowed": False,
            "paper_claim_allowed": False,
        },
    }


def _validate_resume_metadata(
    checkpoint: Mapping[str, Any],
    *,
    seed: int,
    identity: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    if set(checkpoint) != _STAGE_B_CHECKPOINT_KEYS:
        raise ValueError("r2 Stage-B resume checkpoint must have the exact frozen key set")
    if checkpoint.get("schema") != _R2_STAGE_B_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported r2 Stage-B resume checkpoint schema")
    meta = checkpoint.get("meta")
    if not isinstance(meta, Mapping) or set(meta) != _STAGE_B_META_KEYS:
        raise ValueError("r2 Stage-B resume metadata must have the exact frozen key set")
    if meta["protocol"] != "CT-P3R-3S-r2" or meta["chronotransport_stage"] != "B":
        raise ValueError("r2 Stage-B resume protocol/stage mismatch")
    if meta["status"] not in {
        "INCOMPLETE_INFRASTRUCTURE_CHECKPOINT",
        "TRAINING_COMPLETE_BASELINE_PENDING",
    }:
        raise ValueError("r2 Stage-B resume status mismatch")
    if type(meta["seed"]) is not int or meta["seed"] != seed:
        raise ValueError("r2 Stage-B resume seed mismatch")
    cursor = meta["successful_update_cursor"]
    if type(cursor) is not int or not 0 <= cursor <= 140:
        raise ValueError("r2 Stage-B resume cursor must be an integer in [0, 140]")
    counters = meta["counters"]
    if not isinstance(counters, Mapping) or set(counters) != _STAGE_B_COUNTER_KEYS:
        raise ValueError("r2 Stage-B resume counters must have the exact frozen key set")
    if any(type(value) is not int or value < 0 for value in counters.values()):
        raise ValueError("r2 Stage-B resume counters must be non-negative integers")
    if counters["successful_optimizer_updates"] != cursor:
        raise ValueError("r2 Stage-B resume success counter/cursor mismatch")
    if meta["status"] == "TRAINING_COMPLETE_BASELINE_PENDING" and cursor != 140:
        raise ValueError("training-complete r2 Stage-B checkpoint must have cursor 140")
    if meta["status"] != "TRAINING_COMPLETE_BASELINE_PENDING" and cursor >= 140:
        raise ValueError("incomplete r2 Stage-B checkpoint must have cursor below 140")
    for field in (
        "dense_checkpoint_path",
        "dense_checkpoint_sha256",
        "dense_checkpoint_state_key",
        "dense_checkpoint_state_sha256",
        "dense_checkpoint_bytes",
        "dense_checkpoint_top_level_keys_sha256",
        "manifest_sha256",
        "library_sha256",
        "config_sha256",
        "exposure_artifact_sha256",
    ):
        if meta[field] != provenance[field]:
            raise ValueError(f"r2 Stage-B resume {field} mismatch")
    _require_sha256(meta["ledger_sha256"], field="resume ledger SHA-256")
    if meta["transport_path"] != identity["transport_path"]:
        raise ValueError("r2 Stage-B resume transport identity mismatch")
    if meta["risk_predictor_path"] != identity["risk_predictor_path"]:
        raise ValueError("r2 Stage-B resume predictor identity mismatch")
    if type(meta["scheduler_predictor_alias"]) is not bool or meta[
        "scheduler_predictor_alias"
    ] != bool(identity["scheduler_predictor_alias"]):
        raise ValueError("r2 Stage-B resume predictor alias mismatch")
    if meta["scheduler_predictor_paths"] != identity["scheduler_predictor_paths"]:
        raise ValueError("r2 Stage-B resume predictor alias paths mismatch")
    if meta["parameter_aliases"] != identity["parameter_aliases"]:
        raise ValueError("r2 Stage-B resume parameter aliases mismatch")
    if (
        meta["registered_provenance"] != provenance["registered_provenance"]
        or meta["registered_provenance_sha256"]
        != provenance["registered_provenance_sha256"]
    ):
        raise ValueError("r2 Stage-B resume registered provenance mismatch")
    for field, state_key in (
        ("state_dict_sha256", "state_dict"),
        ("state_dict_ema_sha256", "state_dict_ema"),
        ("ema_state_sha256", "ema"),
    ):
        if meta[field] != _state_dict_sha256(checkpoint[state_key]):
            label = "EMA state SHA-256" if field == "ema_state_sha256" else field
            raise ValueError(f"r2 Stage-B resume {label} mismatch")
    if meta["optimizer"] != {"name": "AdamW", "lr": 1e-4, "weight_decay": 0.0}:
        raise ValueError("r2 Stage-B resume optimizer protocol mismatch")
    if meta["loss"] != {
        "lambda_transport": 0.1,
        "lambda_risk": 0.1,
        "risk_quantile": 0.9,
        "feature_loss": "elementwise_mean_mse",
    }:
        raise ValueError("r2 Stage-B resume loss protocol mismatch")
    expected_scalars = {
        "ema_decay": 0.999,
        "gradient_clip_norm": 1.0,
        "amp_enabled": False,
        "calibration_ready": False,
        "measured_cost_ready": False,
        "deploy_claim_allowed": False,
        "paper_claim_allowed": False,
    }
    for field, expected in expected_scalars.items():
        if type(meta[field]) is not type(expected) or meta[field] != expected:
            raise ValueError(f"r2 Stage-B resume {field} protocol mismatch")
    return meta


def _validate_resume_training_state(
    checkpoint: Mapping[str, Any],
    *,
    cursor: int,
    identity: Mapping[str, Any],
    registered_dense_frozen_state: Mapping[str, Any],
) -> None:
    optimizer = checkpoint.get("optimizer")
    if not isinstance(optimizer, Mapping) or set(optimizer) != {"state", "param_groups"}:
        raise ValueError("r2 Stage-B resume optimizer structure mismatch")
    groups = optimizer["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], Mapping):
        raise ValueError("r2 Stage-B resume optimizer must contain one parameter group")
    group = groups[0]
    if (
        type(group.get("lr")) is not float
        or group["lr"] != 1e-4
        or type(group.get("weight_decay")) is not float
        or group["weight_decay"] != 0.0
        or not isinstance(group.get("params"), list)
        or len(group["params"]) != len(identity["trainable_parameters"])
    ):
        raise ValueError("r2 Stage-B resume optimizer LR/WD/parameter count mismatch")
    state = optimizer["state"]
    if not isinstance(state, Mapping):
        raise ValueError("r2 Stage-B resume optimizer state must be a mapping")
    if len(set(group["params"])) != len(group["params"]) or set(state) != set(
        group["params"]
    ):
        raise ValueError("r2 Stage-B resume optimizer state/parameter mapping mismatch")
    for parameter_state in state.values():
        if not isinstance(parameter_state, Mapping):
            raise ValueError("r2 Stage-B resume optimizer parameter state mismatch")
        step = parameter_state.get("step")
        if isinstance(step, Tensor):
            if step.numel() != 1 or not torch.isfinite(step).all():
                raise ValueError("r2 Stage-B resume optimizer step must be finite scalar")
            step_value = float(step.item())
        elif type(step) is int:
            step_value = float(step)
        else:
            raise ValueError("r2 Stage-B resume optimizer step structure mismatch")
        if step_value != float(cursor):
            raise ValueError("r2 Stage-B resume optimizer step/cursor mismatch")
        for value in parameter_state.values():
            if isinstance(value, Tensor) and not torch.isfinite(value).all():
                raise ValueError("r2 Stage-B resume optimizer contains non-finite state")
    scheduler = checkpoint.get("lr_scheduler")
    if not isinstance(scheduler, Mapping):
        raise ValueError("r2 Stage-B resume scheduler structure mismatch")
    if type(scheduler.get("last_epoch")) is not int or scheduler["last_epoch"] != cursor:
        raise ValueError("r2 Stage-B resume scheduler step/cursor mismatch")
    if "_step_count" in scheduler and (
        type(scheduler["_step_count"]) is not int
        or scheduler["_step_count"] != cursor + 1
    ):
        raise ValueError("r2 Stage-B resume scheduler internal step mismatch")
    for state_key in ("state_dict", "state_dict_ema"):
        model_state = checkpoint.get(state_key)
        if not isinstance(model_state, Mapping):
            raise ValueError(f"r2 Stage-B resume {state_key} structure mismatch")
        _validate_registered_dense_frozen_state(
            model_state,
            registered_dense_frozen_state=registered_dense_frozen_state,
            label=f"r2 Stage-B resume {state_key}",
        )
        for canonical_name, aliases in identity["parameter_aliases"].items():
            if any(name not in model_state for name in aliases):
                raise ValueError(
                    f"r2 Stage-B resume {state_key} alias keys are incomplete for {canonical_name}"
                )
            values = [model_state[name] for name in aliases]
            first = values[0]
            if not isinstance(first, Tensor) or any(
                not isinstance(value, Tensor)
                or value.dtype != first.dtype
                or tuple(value.shape) != tuple(first.shape)
                or not torch.equal(first, value)
                for value in values[1:]
            ):
                raise ValueError(
                    f"r2 Stage-B resume {state_key} has conflicting alias state for {canonical_name}"
                )
            if first.dtype != torch.float32 or not torch.isfinite(first).all():
                raise ValueError(
                    f"r2 Stage-B resume {state_key} trainable state must be finite FP32"
                )
    ema = checkpoint.get("ema")
    expected_names = set(identity["canonical_names"].values())
    if not isinstance(ema, Mapping) or set(ema) != expected_names:
        raise ValueError("r2 Stage-B resume EMA key structure mismatch")
    for value in ema.values():
        if not isinstance(value, Tensor) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("r2 Stage-B resume EMA tensors must be finite FP32")
    state_dict_ema = checkpoint["state_dict_ema"]
    for canonical_name, aliases in identity["parameter_aliases"].items():
        shadow = ema[canonical_name]
        for alias_name in aliases:
            alias_value = state_dict_ema[alias_name]
            if (
                alias_value.dtype != shadow.dtype
                or tuple(alias_value.shape) != tuple(shadow.shape)
                or _tensor_exact_bytes(alias_value) != _tensor_exact_bytes(shadow)
            ):
                raise ValueError(
                    "r2 Stage-B resume EMA map differs from state_dict_ema alias "
                    f"{alias_name!r} for {canonical_name!r}"
                )
    rng = checkpoint.get("rng_state")
    if not isinstance(rng, Mapping) or set(rng) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise ValueError("r2 Stage-B resume RNG structure mismatch")
    if not isinstance(rng["python"], tuple) or not isinstance(rng["numpy"], tuple):
        raise ValueError("r2 Stage-B resume RNG Python/NumPy state mismatch")
    if not isinstance(rng["torch_cpu"], Tensor) or rng["torch_cpu"].dtype != torch.uint8:
        raise ValueError("r2 Stage-B resume RNG Torch CPU state mismatch")
    cuda_state = rng["torch_cuda"]
    if cuda_state is not None and (
        not isinstance(cuda_state, list)
        or any(not isinstance(value, Tensor) or value.dtype != torch.uint8 for value in cuda_state)
    ):
        raise ValueError("r2 Stage-B resume RNG CUDA state mismatch")


def _validate_resume_ledger(
    rows: Any,
    *,
    cursor: int,
    seed: int,
    batches: Sequence[Mapping[str, Any]],
    exposure_artifact: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != cursor:
        raise ValueError("r2 Stage-B resume cursor/ledger length mismatch")
    expected_rows = exposure_artifact["matrices"][str(seed)]
    normalized: list[dict[str, Any]] = []
    for update, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping) or set(raw_row) != _STAGE_B_LEDGER_KEYS:
            raise ValueError("r2 Stage-B resume ledger row must have the exact frozen key set")
        row = dict(raw_row)
        for field in (
            "seed",
            "successful_update",
            "canonical_window_index",
            "candidate_index",
        ):
            if type(row[field]) is not int:
                raise ValueError(f"r2 Stage-B resume ledger row {field} must be an integer")
        expected = expected_rows[update]
        if (
            row["schema"] != _R2_STAGE_B_LEDGER_SCHEMA
            or row["seed"] != seed
            or row["successful_update"] != update
            or row["canonical_window_index"] != update
            or row["window_id"] != expected["window_id"]
            or row["candidate_index"] != expected["candidate"]
            or row["candidate_name"] != R2_NON_DENSE_NAMES[expected["candidate"]]
            or row["video_id"] != _batch_video_id(batches, update)
        ):
            raise ValueError("r2 Stage-B resume ledger row exposure binding mismatch")
        for field in (
            "dense_checkpoint_path",
            "dense_checkpoint_sha256",
            "dense_checkpoint_state_key",
            "dense_checkpoint_state_sha256",
            "dense_checkpoint_bytes",
            "dense_checkpoint_top_level_keys_sha256",
            "registered_provenance_sha256",
            "manifest_sha256",
            "library_sha256",
            "config_sha256",
            "exposure_artifact_sha256",
        ):
            if row[field] != provenance[field]:
                raise ValueError(f"r2 Stage-B resume ledger row {field} mismatch")
        for field in (
            "materialized_window_sha256",
            "counterfactual_window_sha256",
            "augmentation_sha256",
            "requested_action_sha256",
            "executed_action_sha256",
            "row_sha256",
        ):
            _require_sha256(row[field], field=f"resume ledger row {field}")
        if row["materialized_window_sha256"] != row["counterfactual_window_sha256"]:
            raise ValueError("r2 Stage-B resume ledger row paired materialization mismatch")
        if row["requested_action_sha256"] != row["executed_action_sha256"]:
            raise ValueError("r2 Stage-B resume ledger row action hash mismatch")
        for field in ("loss_detector", "loss_transport", "loss_risk", "loss_total"):
            if isinstance(row[field], bool) or not isinstance(row[field], (int, float)):
                raise ValueError(f"r2 Stage-B resume ledger row {field} must be numeric")
            if not math.isfinite(float(row[field])):
                raise ValueError(f"r2 Stage-B resume ledger row {field} must be finite")
        unsigned = dict(row)
        row_sha256 = unsigned.pop("row_sha256")
        if row_sha256 != _canonical_json_sha256(unsigned):
            raise ValueError("r2 Stage-B resume ledger row digest mismatch")
        normalized.append(row)
    return normalized


def _batch_window_ids(batches: Sequence[Mapping[str, Any]]) -> list[str]:
    declared = getattr(batches, "window_ids", None)
    if declared is not None:
        return list(map(str, declared))
    return [str(batch.get("window_id", "")) for batch in batches]


def _batch_video_id(batches: Sequence[Mapping[str, Any]], index: int) -> str:
    declared = getattr(batches, "video_ids", None)
    if declared is not None:
        return str(declared[index])
    return str(batches[index].get("video_id", ""))


def run_r2_stage_b_training(
    *,
    model: nn.Module,
    batches: Sequence[Mapping[str, Any]],
    replay_step: Callable[[nn.Module, Mapping[str, Any], str], StageBReplayOutput],
    seed: int,
    exposure_artifact: Mapping[str, Any],
    dense_checkpoint_path: Path | str,
    dense_checkpoint_sha256: str,
    dense_checkpoint_use_ema: bool,
    manifest_sha256: str,
    library_sha256: str,
    config_sha256: str,
    output_checkpoint: Path | str,
    ledger_path: Path | str,
    registered_provenance: Mapping[str, Any],
    resume_from: Path | str | None = None,
    stop_after_successful: int | None = None,
    checkpoint_frequency: int = 1,
    preflight: Callable[[nn.Module], None] | None = None,
) -> dict[str, Any]:
    """Run the fixed single-process, FP32, 140-success r2 Stage-B protocol."""

    if type(seed) is not int or seed not in (3407, 3408, 3409):
        raise ValueError("formal Stage B seed must be the integer 3407, 3408, or 3409")
    if len(batches) != 140:
        raise ValueError("formal Stage B requires exactly 140 materialized fit batches")
    fit_window_ids = _batch_window_ids(batches)
    if any(not window_id for window_id in fit_window_ids):
        raise ValueError("every formal Stage-B batch requires a non-empty window_id")
    if any(not _batch_video_id(batches, index) for index in range(140)):
        raise ValueError("every formal Stage-B batch requires a non-empty video_id")
    validated_exposure = validate_stage_b_exposure_artifact(
        exposure_artifact, fit_window_ids=fit_window_ids
    )
    dense_checkpoint_path = _path_without_symlink_components(
        dense_checkpoint_path, label="Stage-B dense checkpoint"
    )
    if not stat.S_ISREG(os.lstat(dense_checkpoint_path).st_mode):
        raise ValueError("Stage-B dense checkpoint must be a regular file")
    expected_dense_sha = _require_sha256(
        dense_checkpoint_sha256, field="dense checkpoint SHA-256"
    )
    actual_dense_sha = _file_sha256(dense_checkpoint_path)
    normalized_registered_provenance = _validate_registered_provenance(
        registered_provenance
    )
    provenance: dict[str, Any] = {
        "dense_checkpoint_path": str(dense_checkpoint_path),
        "dense_checkpoint_sha256": actual_dense_sha,
        "manifest_sha256": _require_sha256(manifest_sha256, field="manifest SHA-256"),
        "library_sha256": _require_sha256(library_sha256, field="library SHA-256"),
        "config_sha256": _require_sha256(config_sha256, field="config SHA-256"),
        "exposure_artifact_sha256": _require_sha256(
            str(validated_exposure["artifact_sha256"]),
            field="exposure artifact SHA-256",
        ),
        "registered_provenance": normalized_registered_provenance,
        "registered_provenance_sha256": _canonical_json_sha256(
            normalized_registered_provenance
        ),
    }
    output_checkpoint = Path(output_checkpoint)
    ledger_path = Path(ledger_path)
    if type(checkpoint_frequency) is not int or checkpoint_frequency < 0:
        raise ValueError("checkpoint_frequency must be a non-negative integer")
    identity = _stage_b_parameter_identity(model)
    dense_state_identity = _load_registered_dense_checkpoint(
        model,
        checkpoint_path=dense_checkpoint_path,
        expected_sha256=expected_dense_sha,
        use_ema=dense_checkpoint_use_ema,
        identity=identity,
    )
    provenance.update(dense_state_identity)
    registered_dense_frozen_state = _capture_registered_dense_frozen_state(
        model, identity
    )
    _configure_stage_b_model(model, identity)
    optimizer = torch.optim.AdamW(
        identity["trainable_parameters"], lr=1e-4, weight_decay=0.0
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    state = StageBUpdateState(
        seed=seed,
        exposure_artifact=validated_exposure,
        candidate_names=R2_NON_DENSE_NAMES,
    )
    ema = _ema_initialize(identity)
    ledger_rows: list[dict[str, Any]] = []

    if resume_from is not None:
        resume_path = Path(resume_from)
        checkpoint = torch.load(resume_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("r2 Stage-B resume checkpoint must be a mapping")
        meta = _validate_resume_metadata(
            checkpoint,
            seed=seed,
            identity=identity,
            provenance=provenance,
        )
        _validate_resume_training_state(
            checkpoint,
            cursor=meta["successful_update_cursor"],
            identity=identity,
            registered_dense_frozen_state=registered_dense_frozen_state,
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        ema = {name: value.clone() for name, value in checkpoint["ema"].items()}
        cursor = meta["successful_update_cursor"]
        ledger_rows = _validate_resume_ledger(
            checkpoint["ledger_rows"],
            cursor=cursor,
            seed=seed,
            batches=batches,
            exposure_artifact=validated_exposure,
            provenance=provenance,
        )
        actual_ledger_sha = _canonical_ledger_sha256(ledger_rows)
        if actual_ledger_sha != meta["ledger_sha256"]:
            raise ValueError("r2 Stage-B resume ledger digest mismatch")
        external_ledger = resume_path.with_suffix(".jsonl")
        if not external_ledger.is_file():
            raise ValueError("r2 Stage-B resume external ledger is missing")
        expected_external = b"".join(
            _canonical_json_bytes(row) + b"\n" for row in ledger_rows
        )
        if external_ledger.read_bytes() != expected_external:
            raise ValueError("r2 Stage-B resume external ledger differs from checkpoint prefix")
        state = StageBUpdateState(
            seed=seed,
            exposure_artifact=validated_exposure,
            candidate_names=R2_NON_DENSE_NAMES,
            successful_update_cursor=cursor,
            counters=meta.get("counters"),
        )
        state.record_retry("infrastructure-resume")
        _restore_rng_state(checkpoint["rng_state"])

    if preflight is not None:
        if not callable(preflight):
            raise TypeError("Stage-B preflight must be callable")
        preflight(model)

    if stop_after_successful is not None:
        if type(stop_after_successful) is not int or not (
            state.successful_update_cursor < stop_after_successful <= 140
        ):
            raise ValueError("stop_after_successful must be after the current cursor and at most 140")
        target_cursor = int(stop_after_successful)
    else:
        target_cursor = 140

    while state.successful_update_cursor < target_cursor:
        exposure = state.current()
        update = state.successful_update_cursor
        batch = batches[update]
        if str(batch.get("window_id")) != str(exposure["window_id"]):
            raise FormalStageBInvalid(
                "batch/exposure window mismatch",
                counters=state.counters,
                successful_update_cursor=state.successful_update_cursor,
            )
        optimizer.zero_grad(set_to_none=True)
        try:
            _assert_fp32_tensors(batch, label="Stage-B materialized batch")
            _assert_fp32_tensors(
                identity["trainable_parameters"], label="Stage-B trainable parameters"
            )
        except ValueError as error:
            state.record_skip("non-FP32 input or parameter")
            raise FormalStageBInvalid(
                str(error),
                counters=state.counters,
                successful_update_cursor=state.successful_update_cursor,
            ) from error
        if torch.is_autocast_enabled():
            state.record_skip("ambient autocast enabled", amp_skip=True)
            raise FormalStageBInvalid(
                "Stage B requires autocast disabled",
                counters=state.counters,
                successful_update_cursor=state.successful_update_cursor,
            )
        device = next(iter(identity["trainable_parameters"])).device
        with torch.autocast(device_type=device.type, enabled=False):
            output = replay_step(model, batch, str(exposure["candidate_name"]))
        if not isinstance(output, StageBReplayOutput):
            raise TypeError("formal Stage-B replay_step must return StageBReplayOutput")
        try:
            _assert_fp32_tensors(output, label="Stage-B forward outputs")
        except ValueError as error:
            state.record_skip("non-FP32 forward output")
            raise FormalStageBInvalid(
                str(error),
                counters=state.counters,
                successful_update_cursor=state.successful_update_cursor,
            ) from error
        try:
            materialized_hash = _require_sha256(
                output.materialized_window_sha256,
                field="materialized window SHA-256",
            )
            counterfactual_hash = _require_sha256(
                output.counterfactual_window_sha256,
                field="counterfactual window SHA-256",
            )
            augmentation_hash = _require_sha256(
                output.augmentation_sha256,
                field="augmentation SHA-256",
            )
            requested_action_hash = _require_sha256(
                output.requested_action_sha256,
                field="requested action SHA-256",
            )
            executed_action_hash = _require_sha256(
                output.executed_action_sha256,
                field="executed action SHA-256",
            )
        except ValueError as error:
            state.record_skip("invalid replay hash")
            raise FormalStageBInvalid(
                str(error),
                counters=state.counters,
                successful_update_cursor=state.successful_update_cursor,
            ) from error
        if materialized_hash != counterfactual_hash:
            state.record_skip("paired materialization mismatch")
            raise FormalStageBInvalid(
                "dense/counterfactual materialized window hashes differ",
                counters=state.counters,
                successful_update_cursor=state.successful_update_cursor,
            )
        if requested_action_hash != executed_action_hash:
            state.record_skip("requested/executed action mismatch")
            raise FormalStageBInvalid(
                "formal Stage B forbids requested/executed action divergence",
                counters=state.counters,
                successful_update_cursor=state.successful_update_cursor,
            )
        expected_augmentation = batch.get("augmentation_sha256")
        if expected_augmentation is not None and str(expected_augmentation) != augmentation_hash:
            state.record_skip("augmentation mismatch")
            raise FormalStageBInvalid(
                "replay augmentation hash does not match the materialized batch",
                counters=state.counters,
                successful_update_cursor=state.successful_update_cursor,
            )
        if type(output.amp_skipped) is not bool or output.amp_skipped:
            state.record_skip("AMP skip", amp_skip=True)
            raise FormalStageBInvalid(
                "Stage B forbids autocast/GradScaler and AMP skips",
                counters=state.counters,
                successful_update_cursor=state.successful_update_cursor,
            )
        losses: R2StageBLosses = compose_r2_stage_b_loss(
            counterfactual_task_loss=output.counterfactual_task_loss,
            counterfactual_features=output.counterfactual_features,
            dense_features=output.dense_features,
            predicted_quantile=output.predicted_quantile,
            regret_target=output.regret_target,
        )
        loss_values = torch.stack(
            [losses.total, losses.detector, losses.transport, losses.risk]
        )
        if not torch.isfinite(loss_values).all():
            state.record_skip("nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            raise FormalStageBInvalid(
                "non-finite Stage-B loss",
                counters=state.counters,
                successful_update_cursor=state.successful_update_cursor,
            )
        losses.total.backward()
        gradients = [
            parameter.grad
            for parameter in identity["trainable_parameters"]
            if parameter.grad is not None
        ]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            state.record_skip("nonfinite or absent gradient")
            optimizer.zero_grad(set_to_none=True)
            raise FormalStageBInvalid(
                "non-finite or absent Stage-B gradient",
                counters=state.counters,
                successful_update_cursor=state.successful_update_cursor,
            )
        clip_norm = torch.nn.utils.clip_grad_norm_(
            identity["trainable_parameters"], max_norm=1.0
        )
        if not torch.isfinite(torch.as_tensor(clip_norm)).all():
            state.record_skip("nonfinite gradient clip norm")
            optimizer.zero_grad(set_to_none=True)
            raise FormalStageBInvalid(
                "non-finite Stage-B pre-clip gradient norm",
                counters=state.counters,
                successful_update_cursor=state.successful_update_cursor,
            )
        optimizer.step()
        lr_scheduler.step()
        _ema_update(ema, identity, decay=0.999)
        if any(
            not torch.isfinite(parameter.detach()).all()
            for parameter in identity["trainable_parameters"]
        ) or any(not torch.isfinite(value).all() for value in ema.values()):
            state.record_skip("nonfinite post-update state")
            raise FormalStageBInvalid(
                "non-finite Stage-B transport/risk/EMA state after optimizer update",
                counters=state.counters,
                successful_update_cursor=state.successful_update_cursor,
            )
        expected_scheduler_epoch = update + 1
        if int(lr_scheduler.last_epoch) != expected_scheduler_epoch:
            state.record_skip("LR scheduler step mismatch")
            raise FormalStageBInvalid(
                "Stage-B LR scheduler did not advance exactly once",
                counters=state.counters,
                successful_update_cursor=state.successful_update_cursor,
            )
        try:
            _validate_registered_dense_frozen_state(
                model.state_dict(),
                registered_dense_frozen_state=registered_dense_frozen_state,
                label=f"Stage-B successful step {update} terminal",
            )
        except ValueError as error:
            state.record_skip("registered dense frozen state mutation")
            raise FormalStageBInvalid(
                str(error),
                counters=state.counters,
                successful_update_cursor=state.successful_update_cursor,
            ) from error

        row: dict[str, Any] = {
            "schema": _R2_STAGE_B_LEDGER_SCHEMA,
            "seed": seed,
            "successful_update": update,
            "canonical_window_index": int(exposure["canonical_window_index"]),
            "video_id": str(batch.get("video_id", "")),
            "window_id": str(exposure["window_id"]),
            "candidate_index": int(exposure["candidate_index"]),
            "candidate_name": str(exposure["candidate_name"]),
            "dense_checkpoint_path": str(dense_checkpoint_path),
            "dense_checkpoint_sha256": actual_dense_sha,
            "dense_checkpoint_state_key": provenance["dense_checkpoint_state_key"],
            "dense_checkpoint_state_sha256": provenance[
                "dense_checkpoint_state_sha256"
            ],
            "dense_checkpoint_bytes": provenance["dense_checkpoint_bytes"],
            "dense_checkpoint_top_level_keys_sha256": provenance[
                "dense_checkpoint_top_level_keys_sha256"
            ],
            "manifest_sha256": provenance["manifest_sha256"],
            "library_sha256": provenance["library_sha256"],
            "config_sha256": provenance["config_sha256"],
            "exposure_artifact_sha256": provenance["exposure_artifact_sha256"],
            "materialized_window_sha256": materialized_hash,
            "counterfactual_window_sha256": counterfactual_hash,
            "augmentation_sha256": augmentation_hash,
            "requested_action_sha256": requested_action_hash,
            "executed_action_sha256": executed_action_hash,
            "registered_provenance_sha256": provenance[
                "registered_provenance_sha256"
            ],
            "loss_detector": float(losses.detector.detach().cpu()),
            "loss_transport": float(losses.transport.detach().cpu()),
            "loss_risk": float(losses.risk.detach().cpu()),
            "loss_total": float(losses.total.detach().cpu()),
        }
        row["row_sha256"] = _canonical_json_sha256(row)
        ledger_rows.append(row)
        state.record_success(ledger_row_sha256=row["row_sha256"])
        if checkpoint_frequency and state.successful_update_cursor % checkpoint_frequency == 0:
            prefix_checkpoint = output_checkpoint.with_name(
                f"{output_checkpoint.stem}.step{state.successful_update_cursor}{output_checkpoint.suffix}"
            )
            prefix_ledger = prefix_checkpoint.with_suffix(".jsonl")
            prefix_ledger_sha256 = _atomic_write_ledger(ledger_rows, prefix_ledger)
            prefix_payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                ema=ema,
                identity=identity,
                state=state,
                ledger_rows=ledger_rows,
                ledger_sha256=prefix_ledger_sha256,
                provenance=provenance,
                status=(
                    "TRAINING_COMPLETE_BASELINE_PENDING"
                    if state.successful_update_cursor == 140
                    else "INCOMPLETE_INFRASTRUCTURE_CHECKPOINT"
                ),
            )
            _atomic_torch_save(prefix_payload, prefix_checkpoint)

    try:
        _validate_registered_dense_frozen_state(
            model.state_dict(),
            registered_dense_frozen_state=registered_dense_frozen_state,
            label="Stage-B terminal",
        )
    except ValueError as error:
        raise FormalStageBInvalid(
            str(error),
            counters=state.counters,
            successful_update_cursor=state.successful_update_cursor,
        ) from error

    ledger_sha256 = _atomic_write_ledger(ledger_rows, ledger_path)
    status = (
        "TRAINING_COMPLETE_BASELINE_PENDING"
        if state.successful_update_cursor == 140
        else "INCOMPLETE_INFRASTRUCTURE_CHECKPOINT"
    )
    if status == "TRAINING_COMPLETE_BASELINE_PENDING":
        state.validate_complete()
    checkpoint_payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema=ema,
        identity=identity,
        state=state,
        ledger_rows=ledger_rows,
        ledger_sha256=ledger_sha256,
        provenance=provenance,
        status=status,
    )
    _atomic_torch_save(checkpoint_payload, output_checkpoint)
    return {
        "status": status,
        "seed": seed,
        "counters": dict(state.counters),
        "successful_update_cursor": state.successful_update_cursor,
        "checkpoint_path": str(output_checkpoint),
        "checkpoint_sha256": _file_sha256(output_checkpoint),
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger_sha256,
        **provenance,
    }


def _canonical_ledger_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)
    return hashlib.sha256(payload).hexdigest()


def build_fit_schedule_constant_artifact(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    fit_window_ids: Sequence[str],
    candidate_action_sha256_by_name: Mapping[str, str],
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    """Freeze the r2 fit-only 140x16 replay and rank-127 constants."""

    if type(seed) is not int or seed not in (3407, 3408, 3409):
        raise ValueError("fit-only baseline seed must be 3407, 3408, or 3409")
    windows = list(map(str, fit_window_ids))
    if len(windows) != 140 or len(set(windows)) != 140:
        raise ValueError("fit-only baseline requires exactly 140 unique fit windows")
    registered_actions = _validate_candidate_action_sha256_by_name(
        candidate_action_sha256_by_name
    )
    provenance_fields = {
        "registration_sha256",
        "manifest_sha256",
        "library_sha256",
        "trained_checkpoint_sha256",
        "predictor_state_sha256",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != provenance_fields:
        raise ValueError("fit-only baseline provenance fields mismatch")
    validated_provenance = {
        field: _require_sha256(provenance[field], field=field)
        for field in sorted(provenance_fields)
    }
    expected_count = 140 * len(R2_NON_DENSE_NAMES)
    if not isinstance(rows, Sequence) or len(rows) != expected_count:
        raise ValueError("fit-only baseline requires exactly 140x16 replay rows")
    required = {
        "seed",
        "window_id",
        "candidate_index",
        "schedule",
        "regret",
        "materialized_window_sha256",
        "augmentation_sha256",
        "requested_action_sha256",
        "executed_action_sha256",
    }
    normalized: list[dict[str, Any]] = []
    by_schedule: dict[str, list[float]] = {name: [] for name in R2_NON_DENSE_NAMES}
    replay_keys = []
    for ordinal, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("fit-only baseline replay row fields mismatch")
        row = dict(raw)
        window_index, candidate_index = divmod(ordinal, 16)
        schedule = R2_NON_DENSE_NAMES[candidate_index]
        if (
            type(row["seed"]) is not int
            or row["seed"] != seed
            or type(row["candidate_index"]) is not int
            or row["candidate_index"] != candidate_index
            or row["window_id"] != windows[window_index]
            or row["schedule"] != schedule
        ):
            raise ValueError("fit-only baseline replay row canonical order/binding mismatch")
        if isinstance(row["regret"], bool) or not isinstance(row["regret"], (int, float)):
            raise TypeError("fit-only baseline regret must be numeric")
        regret = float(row["regret"])
        if not math.isfinite(regret) or regret < 0.0:
            raise ValueError("fit-only baseline regret must be finite and non-negative")
        row["regret"] = regret
        for field in (
            "materialized_window_sha256",
            "augmentation_sha256",
            "requested_action_sha256",
            "executed_action_sha256",
        ):
            row[field] = _require_sha256(row[field], field=f"baseline row {field}")
        if row["requested_action_sha256"] != row["executed_action_sha256"]:
            raise ValueError("fit-only baseline forbids requested/executed action divergence")
        if row["requested_action_sha256"] != registered_actions[schedule]:
            raise ValueError(
                f"fit-only baseline action for {schedule} differs from registered action"
            )
        row["row_sha256"] = _canonical_json_sha256(row)
        normalized.append(row)
        by_schedule[schedule].append(regret)
        replay_keys.append(
            {
                "seed": seed,
                "window_id": row["window_id"],
                "schedule": schedule,
                "candidate_index": candidate_index,
                "row_sha256": row["row_sha256"],
            }
        )
    constants = {}
    for schedule in R2_NON_DENSE_NAMES:
        targets = sorted(by_schedule[schedule])
        if len(targets) != 140:
            raise ValueError("every fit-only baseline schedule requires 140 targets")
        constants[schedule] = targets[126]
    artifact: dict[str, Any] = {
        "schema": "chronotransport-r2-fit-schedule-constant-v1",
        "protocol": "CT-P3R-3S-r2",
        "seed": seed,
        "candidate_order": list(R2_NON_DENSE_NAMES),
        "candidate_action_sha256_by_name": registered_actions,
        "fit_window_ids": windows,
        "fit_window_order_sha256": _canonical_json_sha256(windows),
        "row_count": expected_count,
        "rows": normalized,
        "fit_replay_key_sha256": _canonical_json_sha256(replay_keys),
        "quantile": 0.9,
        "order_statistic_rank": 127,
        "schedule_constants": constants,
        "provenance": validated_provenance,
    }
    artifact["artifact_sha256"] = _canonical_json_sha256(artifact)
    return artifact


def _load_exact_canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    _, raw = _read_regular_file_bytes(path, label=label)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object")
    if raw != _canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{label} bytes must be exact canonical JSON plus one newline")
    return value


def _load_exact_canonical_ledger(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    _, raw = _read_regular_file_bytes(path, label="Stage-B ledger")
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("Stage-B ledger must be non-empty and newline terminated")
    rows: list[dict[str, Any]] = []
    rebuilt = bytearray()
    for ordinal, line in enumerate(raw.splitlines()):
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Stage-B ledger row {ordinal} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"Stage-B ledger row {ordinal} must be an object")
        encoded = _canonical_json_bytes(row)
        if encoded != line:
            raise ValueError(f"Stage-B ledger row {ordinal} is not canonical JSON")
        rows.append(row)
        rebuilt.extend(encoded + b"\n")
    if bytes(rebuilt) != raw:
        raise ValueError("Stage-B ledger exact bytes mismatch")
    return rows, raw


def _validate_fit_schedule_constant_artifact(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema",
        "protocol",
        "seed",
        "candidate_order",
        "candidate_action_sha256_by_name",
        "fit_window_ids",
        "fit_window_order_sha256",
        "row_count",
        "rows",
        "fit_replay_key_sha256",
        "quantile",
        "order_statistic_rank",
        "schedule_constants",
        "provenance",
        "artifact_sha256",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != required:
        raise ValueError("fit-only baseline artifact fields mismatch")
    unsigned = dict(artifact)
    artifact_sha256 = unsigned.pop("artifact_sha256")
    if artifact_sha256 != _canonical_json_sha256(unsigned):
        raise ValueError("fit-only baseline payload SHA-256 mismatch")
    source_rows = []
    for row in artifact["rows"]:
        if not isinstance(row, Mapping) or "row_sha256" not in row:
            raise ValueError("fit-only baseline canonical row digest is missing")
        unsigned_row = dict(row)
        row_sha256 = unsigned_row.pop("row_sha256")
        if row_sha256 != _canonical_json_sha256(unsigned_row):
            raise ValueError("fit-only baseline canonical row digest mismatch")
        source_rows.append(unsigned_row)
    rebuilt = build_fit_schedule_constant_artifact(
        source_rows,
        seed=artifact["seed"],
        fit_window_ids=artifact["fit_window_ids"],
        candidate_action_sha256_by_name=artifact[
            "candidate_action_sha256_by_name"
        ],
        provenance=artifact["provenance"],
    )
    if dict(artifact) != rebuilt:
        raise ValueError("fit-only baseline differs from its canonical 140x16 rebuild")
    return rebuilt


def _phase_completion_payload(
    *,
    registration_sha256: str,
    registration_commit: str,
    seed: int,
    model: nn.Module,
    batches: Sequence[Mapping[str, Any]],
    exposure_artifact: Mapping[str, Any],
    dense_checkpoint_path: Path | str,
    dense_checkpoint_sha256: str,
    dense_checkpoint_use_ema: bool,
    registered_provenance: Mapping[str, Any],
    checkpoint_path: Path | str,
    ledger_path: Path | str,
    fit_baseline_path: Path | str,
    candidate_action_sha256_by_name: Mapping[str, str],
    manifest_sha256: str,
    library_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    registration_sha256 = _require_sha256(
        registration_sha256, field="phase marker registration SHA-256"
    )
    if not _FULL_COMMIT.fullmatch(str(registration_commit)):
        raise ValueError("phase marker registration commit R must be one full commit")
    if type(seed) is not int or seed not in (3407, 3408, 3409):
        raise ValueError("phase marker seed must be 3407, 3408, or 3409")
    manifest_sha256 = _require_sha256(
        manifest_sha256, field="phase marker manifest SHA-256"
    )
    library_sha256 = _require_sha256(
        library_sha256, field="phase marker library SHA-256"
    )
    config_sha256 = _require_sha256(
        config_sha256, field="phase marker config SHA-256"
    )
    registered_actions = _validate_candidate_action_sha256_by_name(
        candidate_action_sha256_by_name
    )
    if len(batches) != 140:
        raise ValueError("phase marker requires exactly 140 materialized fit batches")
    fit_window_ids = _batch_window_ids(batches)
    if any(not window_id for window_id in fit_window_ids):
        raise ValueError("phase marker batches require non-empty window IDs")
    if any(not _batch_video_id(batches, index) for index in range(140)):
        raise ValueError("phase marker batches require non-empty video IDs")
    validated_exposure = validate_stage_b_exposure_artifact(
        exposure_artifact, fit_window_ids=fit_window_ids
    )
    dense_checkpoint_path = _path_without_symlink_components(
        dense_checkpoint_path, label="phase marker dense checkpoint"
    )
    if not stat.S_ISREG(os.lstat(dense_checkpoint_path).st_mode):
        raise ValueError("phase marker dense checkpoint must be a regular file")
    dense_checkpoint_sha256 = _require_sha256(
        dense_checkpoint_sha256, field="phase marker dense checkpoint SHA-256"
    )
    normalized_registered_provenance = _validate_registered_provenance(
        registered_provenance
    )
    if (
        normalized_registered_provenance["registration_sha256"]
        != registration_sha256
        or normalized_registered_provenance["registration_commit"]
        != str(registration_commit)
    ):
        raise ValueError("phase marker registered provenance differs from registration R")
    checkpoint_path = _path_without_symlink_components(
        checkpoint_path, label="phase marker trained checkpoint"
    )
    ledger_path = _path_without_symlink_components(
        ledger_path, label="phase marker ledger"
    )
    fit_baseline_path = _path_without_symlink_components(
        fit_baseline_path, label="phase marker fit-only baseline"
    )
    if not stat.S_ISREG(os.lstat(checkpoint_path).st_mode):
        raise ValueError("phase marker trained checkpoint must be a regular file")
    checkpoint_path, checkpoint, checkpoint_payload = _load_torch_regular_file(
        checkpoint_path, label="Stage-B trained checkpoint"
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
    checkpoint_bytes = len(checkpoint_payload)
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != _STAGE_B_CHECKPOINT_KEYS:
        raise ValueError("Stage-B trained checkpoint fields mismatch")

    identity = _stage_b_parameter_identity(model)
    dense_state_identity = _load_registered_dense_checkpoint(
        model,
        checkpoint_path=dense_checkpoint_path,
        expected_sha256=dense_checkpoint_sha256,
        use_ema=dense_checkpoint_use_ema,
        identity=identity,
    )
    registered_dense_frozen_state = _capture_registered_dense_frozen_state(
        model, identity
    )
    provenance: dict[str, Any] = {
        "dense_checkpoint_path": str(dense_checkpoint_path),
        "dense_checkpoint_sha256": dense_checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "library_sha256": library_sha256,
        "config_sha256": config_sha256,
        "exposure_artifact_sha256": _require_sha256(
            str(validated_exposure["artifact_sha256"]),
            field="phase marker exposure artifact SHA-256",
        ),
        "registered_provenance": normalized_registered_provenance,
        "registered_provenance_sha256": _canonical_json_sha256(
            normalized_registered_provenance
        ),
        **dense_state_identity,
    }
    meta = _validate_resume_metadata(
        checkpoint,
        seed=seed,
        identity=identity,
        provenance=provenance,
    )
    _validate_resume_training_state(
        checkpoint,
        cursor=meta["successful_update_cursor"],
        identity=identity,
        registered_dense_frozen_state=registered_dense_frozen_state,
    )
    if (
        meta["status"] != "TRAINING_COMPLETE_BASELINE_PENDING"
        or meta["successful_update_cursor"] != 140
        or meta["seed"] != seed
        or meta["calibration_ready"] is not False
        or meta["manifest_sha256"] != manifest_sha256
        or meta["library_sha256"] != library_sha256
        or meta["config_sha256"] != config_sha256
    ):
        raise ValueError("Stage-B trained checkpoint is not a complete baseline-pending run")
    state_dict_ema = checkpoint.get("state_dict_ema")
    if not isinstance(state_dict_ema, Mapping):
        raise ValueError("Stage-B checkpoint state_dict_ema is missing")
    state_dict_ema_sha256 = _state_dict_sha256(state_dict_ema)
    if state_dict_ema_sha256 != meta["state_dict_ema_sha256"]:
        raise ValueError("Stage-B checkpoint EMA state SHA-256 mismatch")
    predictor_sha256 = logical_risk_predictor_state_sha256(model, state_dict_ema)

    checkpoint_ledger_rows = _validate_resume_ledger(
        checkpoint["ledger_rows"],
        cursor=meta["successful_update_cursor"],
        seed=seed,
        batches=batches,
        exposure_artifact=validated_exposure,
        provenance=provenance,
    )
    checkpoint_ledger_bytes = b"".join(
        _canonical_json_bytes(row) + b"\n" for row in checkpoint_ledger_rows
    )
    checkpoint_ledger_sha256 = hashlib.sha256(checkpoint_ledger_bytes).hexdigest()
    if checkpoint_ledger_sha256 != meta["ledger_sha256"]:
        raise ValueError("Stage-B checkpoint ledger digest differs from metadata")
    ledger_rows, ledger_bytes_payload = _load_exact_canonical_ledger(ledger_path)
    ledger_exact_sha256 = hashlib.sha256(ledger_bytes_payload).hexdigest()
    if (
        len(ledger_rows) != 140
        or ledger_rows != checkpoint_ledger_rows
        or ledger_bytes_payload != checkpoint_ledger_bytes
        or ledger_exact_sha256 != checkpoint_ledger_sha256
    ):
        raise ValueError(
            "Stage-B external JSONL differs from checkpoint ledger rows/bytes/digest"
        )

    baseline = _validate_fit_schedule_constant_artifact(
        _load_exact_canonical_json(fit_baseline_path, label="fit-only baseline")
    )
    baseline_bytes = _canonical_json_bytes(baseline) + b"\n"
    provenance = baseline["provenance"]
    if (
        baseline["seed"] != seed
        or baseline["row_count"] != 140 * len(R2_NON_DENSE_NAMES)
        or baseline["candidate_order"] != list(R2_NON_DENSE_NAMES)
        or baseline["candidate_action_sha256_by_name"] != registered_actions
        or baseline["fit_window_ids"] != fit_window_ids
        or baseline["fit_window_order_sha256"]
        != _canonical_json_sha256(fit_window_ids)
        or provenance["registration_sha256"] != registration_sha256
        or provenance["manifest_sha256"] != manifest_sha256
        or provenance["library_sha256"] != library_sha256
        or provenance["trained_checkpoint_sha256"] != checkpoint_sha256
        or provenance["predictor_state_sha256"] != predictor_sha256
    ):
        raise ValueError(
            "fit-only baseline fit windows, registered actions, or provenance differ "
            "from the Stage-B run"
        )

    return {
        "schema": _R2_STAGE_B_PHASE_COMPLETION_SCHEMA,
        "protocol": "CT-P3R-3S-r2",
        "status": "PHASE_COMPLETE",
        "registration_sha256": registration_sha256,
        "registration_commit": str(registration_commit),
        "seed": seed,
        "manifest_sha256": manifest_sha256,
        "library_sha256": library_sha256,
        "config_sha256": config_sha256,
        "candidate_order": list(R2_NON_DENSE_NAMES),
        "trained_checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_bytes,
            "exact_bytes_sha256": checkpoint_sha256,
            "state_dict_ema_sha256": state_dict_ema_sha256,
            "predictor_canonical_sha256": predictor_sha256,
        },
        "ledger": {
            "path": str(ledger_path),
            "bytes": len(ledger_bytes_payload),
            "exact_bytes_sha256": ledger_exact_sha256,
            "canonical_rows_sha256": _canonical_json_sha256(ledger_rows),
            "row_count": len(ledger_rows),
        },
        "fit_baseline": {
            "path": str(fit_baseline_path),
            "bytes": len(baseline_bytes),
            "exact_bytes_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
            "payload_sha256": baseline["artifact_sha256"],
            "row_count": baseline["row_count"],
            "fit_window_order_sha256": baseline["fit_window_order_sha256"],
            "fit_replay_key_sha256": baseline["fit_replay_key_sha256"],
        },
    }


def build_r2_stage_b_phase_completion_marker(**kwargs: Any) -> dict[str, Any]:
    """Build but do not write the only artifact allowed to claim Stage-B completion."""

    marker = _phase_completion_payload(**kwargs)
    marker["artifact_sha256"] = _canonical_json_sha256(marker)
    return marker


def validate_r2_stage_b_phase_completion_marker(
    marker: Mapping[str, Any] | Path | str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Revalidate the marker and every bound artifact before calibration/evaluation."""

    if isinstance(marker, (str, Path)):
        marker = _load_exact_canonical_json(
            Path(marker), label="Stage-B phase-completion marker"
        )
    if not isinstance(marker, Mapping):
        raise TypeError("Stage-B phase-completion marker must be a mapping or path")
    expected = build_r2_stage_b_phase_completion_marker(**kwargs)
    if set(marker) != set(expected):
        raise ValueError("Stage-B phase-completion marker fields mismatch")
    unsigned = dict(marker)
    claimed_sha256 = unsigned.pop("artifact_sha256")
    if claimed_sha256 != _canonical_json_sha256(unsigned):
        raise ValueError("Stage-B phase-completion marker artifact SHA-256 mismatch")
    if dict(marker) != expected:
        raise ValueError("Stage-B phase-completion marker differs from bound artifacts")
    return expected
