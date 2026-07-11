"""Pure, schema-first adjudicators for the ChronoTransport r2 hard gates."""

from __future__ import annotations

import math
import random
from typing import Mapping, Sequence


HOLD_TIME = ("periodic2_hold", "periodic4_hold", "periodic8_hold")
HOLD_LAYER = ("layer_only_early_recompute_hold", "layer_only_late_recompute_hold")


def _finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires non-empty values")
    return float(sum(values) / len(values))


def _percentile_ci(values: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(map(float, values))
    if not ordered:
        raise ValueError("percentile CI requires values")
    last = len(ordered) - 1
    return ordered[int(math.floor(0.025 * last))], ordered[int(math.ceil(0.975 * last))]


def _normalize_vectors(
    records: Mapping[str, Mapping[str, float]],
    names: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, list[float]]]:
    windows = tuple(sorted(map(str, records)))
    if not windows:
        raise ValueError("gate records require at least one window")
    vectors = {name: [] for name in names}
    for window in windows:
        row = records[window]
        for name in names:
            if name not in row:
                raise ValueError(f"window {window} is missing candidate {name}")
            vectors[name].append(_finite(row[name], f"regret[{window},{name}]"))
    return windows, vectors


def _argmin_mean(names: Sequence[str], vectors: Mapping[str, Sequence[float]]) -> str:
    if not names:
        raise ValueError("candidate selection requires non-empty names")
    return min(names, key=lambda name: (_mean(vectors[name]), tuple(names).index(name)))


def gate1_oracle_headroom(
    *,
    calibration: Mapping[str, Mapping[str, float]],
    evaluation: Mapping[str, Mapping[str, float]],
    candidate_cost_p50: Mapping[str, float],
    dense_cost_p50: float,
    budget: float,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20260711,
) -> dict[str, object]:
    """Adjudicate the frozen equal-cost HOLD-library oracle headroom gate."""

    dense_cost = _finite(dense_cost_p50, "dense cost")
    budget = _finite(budget, "budget")
    if dense_cost <= 0 or budget <= 0:
        raise ValueError("costs and budget must be positive")
    costs = {str(name): _finite(value, f"cost[{name}]") for name, value in candidate_cost_p50.items()}
    hold_names = tuple(
        name
        for name in costs
        if (name.endswith("_hold") or name == "hold_only") and costs[name] <= budget
    )
    for required in (*HOLD_TIME, *HOLD_LAYER):
        if required not in hold_names:
            raise ValueError(f"Gate 1 budget lacks required HOLD candidate: {required}")
    controls = tuple(
        name for name in costs if name.startswith(("motion_topk_p", "random_p")) and costs[name] <= budget
    )
    required_controls = tuple(
        f"{prefix}_p{period}"
        for prefix in ("motion_topk", "random")
        for period in (2, 4, 8)
    )
    missing_controls = [name for name in required_controls if name not in controls]
    if missing_controls:
        raise ValueError(f"Gate 1 requires cost-feasible comparators: {missing_controls}")

    all_names = tuple(dict.fromkeys((*hold_names, *controls)))
    _, calibration_vectors = _normalize_vectors(calibration, hold_names)
    windows, evaluation_vectors = _normalize_vectors(evaluation, all_names)
    calibration_static = _argmin_mean(hold_names, calibration_vectors)

    def replicate(indices: Sequence[int]) -> tuple[float, str, str, dict[str, float]]:
        sampled = {
            name: [evaluation_vectors[name][index] for index in indices]
            for name in all_names
        }
        evaluation_static = _argmin_mean(hold_names, sampled)
        time_oracle = [min(sampled[name][pos] for name in HOLD_TIME) for pos in range(len(indices))]
        layer_oracle = [min(sampled[name][pos] for name in HOLD_LAYER) for pos in range(len(indices))]
        joint_oracle = [min(sampled[name][pos] for name in hold_names) for pos in range(len(indices))]
        comparators: dict[str, list[float]] = {
            f"calibration_static:{calibration_static}": sampled[calibration_static],
            "time_only_oracle": time_oracle,
            "layer_only_oracle": layer_oracle,
            f"evaluation_static:{evaluation_static}": sampled[evaluation_static],
        }
        comparators.update({name: sampled[name] for name in controls})
        strongest = min(comparators, key=lambda name: (_mean(comparators[name]), name))
        means = {name: _mean(vector) for name, vector in comparators.items()}
        return (
            _mean([left - right for left, right in zip(comparators[strongest], joint_oracle)]),
            strongest,
            evaluation_static,
            {**means, "joint_oracle": _mean(joint_oracle)},
        )

    full_indices = list(range(len(windows)))
    improvement, strongest, evaluation_static, means = replicate(full_indices)
    strongest_mean = means[strongest]
    relative = float("nan") if strongest_mean <= 1e-12 else improvement / strongest_mean
    rng = random.Random(int(bootstrap_seed))
    bootstrap = []
    for _ in range(int(bootstrap_samples)):
        indices = [rng.randrange(len(windows)) for _ in windows]
        bootstrap.append(replicate(indices)[0])
    ci = _percentile_ci(bootstrap)
    saving = 1.0 - budget / dense_cost
    hard = {
        "relative_reduction_ge_10pct": strongest_mean > 1e-12 and relative >= 0.10,
        "paired_bootstrap_ci_lower_gt_0": ci[0] > 0.0,
        "budget_saving_ge_20pct": saving >= 0.20,
    }
    return {
        "schema": "chronotransport-r2-gate1-v1",
        "status": "PASS" if all(hard.values()) else "FAIL",
        "oracle_headroom": bool(all(hard.values())),
        "windows": len(windows),
        "candidate_set_size": len(hold_names),
        "feasible_hold_names": list(hold_names),
        "time_oracle_set_size": len(HOLD_TIME),
        "layer_oracle_set_size": len(HOLD_LAYER),
        "joint_oracle_set_size": len(hold_names),
        "calibration_frozen_static": calibration_static,
        "evaluation_best_static": evaluation_static,
        "strongest_comparator": strongest,
        "mean_regret": means,
        "absolute_improvement": improvement,
        "relative_reduction": relative,
        "bootstrap_ci95": list(ci),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "dense_cost_p50": dense_cost,
        "budget": budget,
        "budget_saving": saving,
        "hard_conditions": hard,
    }


def gate2_matched_transport(
    rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20260711,
) -> dict[str, object]:
    """Adjudicate matched TRANSPORT vs HOLD with window/seed hierarchy."""

    normalized = []
    for row in rows:
        seed = int(row["seed"])
        window = str(row["window_id"])
        period = int(row["period"])
        if period not in (2, 4, 8):
            raise ValueError("Gate 2 period must be 2, 4, or 8")
        normalized.append(
            {
                "seed": seed,
                "window": window,
                "period": period,
                "hold_regret": _finite(row["hold_regret"], "hold regret"),
                "transport_regret": _finite(row["transport_regret"], "transport regret"),
                "hold_mse": _finite(row["hold_mse"], "hold mse"),
                "transport_mse": _finite(row["transport_mse"], "transport mse"),
            }
        )
    windows = sorted({row["window"] for row in normalized})
    seeds = sorted({row["seed"] for row in normalized})
    expected = {(window, seed, period) for window in windows for seed in seeds for period in (2, 4, 8)}
    actual = {(row["window"], row["seed"], row["period"]) for row in normalized}
    if actual != expected or len(actual) != len(normalized):
        raise ValueError("Gate 2 requires one complete window×seed×period vector")
    by_key = {(row["window"], row["seed"], row["period"]): row for row in normalized}

    def means(sampled_windows: Sequence[str], sampled_seeds: Sequence[int]) -> tuple[float, float, float]:
        selected = [by_key[(window, seed, period)] for window in sampled_windows for seed in sampled_seeds for period in (2, 4, 8)]
        detector = _mean([row["hold_regret"] - row["transport_regret"] for row in selected])
        feature = _mean([row["hold_mse"] - row["transport_mse"] for row in selected])
        hold = _mean([row["hold_regret"] for row in selected])
        return detector, feature, hold

    detector, feature, hold = means(windows, seeds)
    rng = random.Random(int(bootstrap_seed))
    detector_boot, feature_boot = [], []
    for _ in range(int(bootstrap_samples)):
        sampled_windows = [rng.choice(windows) for _ in windows]
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        det, feat, _ = means(sampled_windows, sampled_seeds)
        detector_boot.append(det)
        feature_boot.append(feat)
    per_seed = {}
    for seed in seeds:
        det, feat, _ = means(windows, [seed])
        per_seed[str(seed)] = {"detector_improvement": det, "feature_improvement": feat}
    relative = float("nan") if hold <= 1e-12 else detector / hold
    detector_ci = _percentile_ci(detector_boot)
    feature_ci = _percentile_ci(feature_boot)
    hard = {
        "relative_reduction_ge_5pct": hold > 1e-12 and relative >= 0.05,
        "detector_ci_lower_gt_0": detector_ci[0] > 0.0,
        "feature_ci_lower_gt_0": feature_ci[0] > 0.0,
        "each_seed_nonnegative": all(
            value["detector_improvement"] >= 0 and value["feature_improvement"] >= 0
            for value in per_seed.values()
        ),
    }
    return {
        "schema": "chronotransport-r2-gate2-v1",
        "status": "PASS" if all(hard.values()) else "FAIL",
        "mechanism": bool(all(hard.values())),
        "detector_improvement": detector,
        "feature_improvement": feature,
        "detector_relative_reduction": relative,
        "detector_ci95": list(detector_ci),
        "feature_ci95": list(feature_ci),
        "per_seed": per_seed,
        "hard_conditions": hard,
    }
