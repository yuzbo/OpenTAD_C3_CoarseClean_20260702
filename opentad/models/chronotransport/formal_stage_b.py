from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import torch

from .replay import validate_compact_record


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
