from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist, mean, pstdev
from typing import Any, Iterable, Mapping, Sequence


SUMMARY_SCHEMA = "duca_rime_causal_gate_summary_v1"
PROTOCOL_SCHEMA = "duca_rime_budget_protocol_v1"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    rows = [
        json.loads(line)
        for line in resolved.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL must contain nonempty object records: {resolved}")
    return rows


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite a different gate artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires nonempty values")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(quantile)
    low, high = int(math.floor(rank)), int(math.ceil(rank))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _video_macro(
    values: Sequence[float],
    videos: Sequence[str],
) -> tuple[list[str], list[float]]:
    if len(values) != len(videos):
        raise ValueError("values and videos must align")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, video in zip(values, videos):
        if math.isfinite(float(value)):
            grouped[str(video)].append(float(value))
    ids = sorted(grouped)
    return ids, [mean(grouped[video]) for video in ids]


def cluster_bootstrap(
    values: Sequence[float],
    videos: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    ids, macro = _video_macro(values, videos)
    if not ids:
        raise ValueError("cluster bootstrap has no finite video values")
    rng = random.Random(int(seed))
    draws = [
        mean(rng.choice(macro) for _ in macro)
        for _sample in range(max(1, int(samples)))
    ]
    return {
        "mean": mean(macro),
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
        "video_count": len(ids),
        "bootstrap_samples": max(1, int(samples)),
    }


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        stop = cursor + 1
        while stop < len(order) and float(values[order[stop]]) == float(
            values[order[cursor]]
        ):
            stop += 1
        rank = 0.5 * (cursor + 1 + stop)
        for offset in range(cursor, stop):
            ranks[order[offset]] = rank
        cursor = stop
    return ranks


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        raise ValueError("Spearman inputs must align and contain at least two values")
    rank_a, rank_b = _rankdata(values_a), _rankdata(values_b)
    mean_a, mean_b = mean(rank_a), mean(rank_b)
    numerator = sum(
        (left - mean_a) * (right - mean_b)
        for left, right in zip(rank_a, rank_b)
    )
    denominator = math.sqrt(
        sum((value - mean_a) ** 2 for value in rank_a)
        * sum((value - mean_b) ** 2 for value in rank_b)
    )
    return 0.0 if denominator <= 0.0 else numerator / denominator


def phase0_variance(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary_metric: str,
    alpha: float,
    power: float,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("schema_version") != "duca_rime_phase0_measurement_v1":
            raise ValueError("unsupported Phase-0 measurement schema")
        if str(row.get("metric_name")) != str(primary_metric):
            continue
        value = float(row["value"])
        if not math.isfinite(value):
            raise ValueError("Phase-0 measurements must be finite")
        grouped[str(row["video_id"])].append(value)
    if len(grouped) < 3 or any(len(values) < 2 for values in grouped.values()):
        raise ValueError("ICC/MDE requires >=3 videos and >=2 replicates per video")
    counts = [len(values) for values in grouped.values()]
    total = sum(counts)
    grand = sum(sum(values) for values in grouped.values()) / total
    between = sum(
        len(values) * (mean(values) - grand) ** 2 for values in grouped.values()
    ) / (len(grouped) - 1)
    within = sum(
        sum((value - mean(values)) ** 2 for value in values)
        for values in grouped.values()
    ) / (total - len(grouped))
    n0 = (
        total - sum(count * count for count in counts) / float(total)
    ) / (len(grouped) - 1)
    denominator = between + (n0 - 1.0) * within
    icc = 0.0 if denominator <= 0.0 else (between - within) / denominator
    video_means = [mean(values) for values in grouped.values()]
    standard_deviation = pstdev(video_means)
    z_alpha = NormalDist().inv_cdf(1.0 - float(alpha) / 2.0)
    z_power = NormalDist().inv_cdf(float(power))
    mde = (z_alpha + z_power) * standard_deviation / math.sqrt(len(video_means))
    rank_threshold = min(
        0.50,
        max(0.05, (z_alpha + z_power) / math.sqrt(max(4, len(video_means) - 3))),
    )
    return {
        "schema_version": SUMMARY_SCHEMA,
        "stage": "phase0_variance_power",
        "primary_metric": str(primary_metric),
        "video_count": len(video_means),
        "replicate_count": total,
        "icc_1_1": icc,
        "between_video_mean_square": between,
        "within_video_mean_square": within,
        "video_mean_std": standard_deviation,
        "alpha": float(alpha),
        "power": float(power),
        "paired_video_mde": mde,
        "rule_derived_thresholds": {
            "min_o1_headroom": mde,
            "max_o2_decoder_regret": mde,
            "min_o3_spearman": rank_threshold,
        },
        "gate_pass": True,
        "claim_scope": "threshold_design_only_no_model_result",
    }


def _oracle_assignment(
    videos: Sequence[str],
    budgets: Sequence[int],
    costs: Mapping[int, int],
    scores: Mapping[tuple[str, int], float],
    *,
    target_mean_cost: float,
) -> dict[str, int]:
    maximum_total = int(math.floor(float(target_mean_cost) * len(videos) + 1.0e-9))
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for video in videos:
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (value, path) in states.items():
            for budget in budgets:
                total = used + int(costs[budget])
                if total > maximum_total:
                    continue
                candidate = (value + float(scores[(video, budget)]), path + (budget,))
                previous = next_states.get(total)
                if previous is None or candidate[0] > previous[0] + 1.0e-12 or (
                    abs(candidate[0] - previous[0]) <= 1.0e-12
                    and candidate[1] < previous[1]
                ):
                    next_states[total] = candidate
        if not next_states:
            raise RuntimeError("no feasible per-video Oracle assignment at target cost")
        states = next_states
    _cost, (_score, path) = max(
        states.items(),
        key=lambda item: (item[1][0], item[0], tuple(-value for value in item[1][1])),
    )
    return dict(zip(videos, path))


def analyze_o1(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_mean_cost: float,
    min_headroom: float,
    bootstrap_samples: int,
    shuffles: int,
    seed: int,
) -> dict[str, Any]:
    panel: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("schema_version") != "duca_rime_o1_budget_panel_v1":
            raise ValueError("unsupported O1 schema")
        video, budget = str(row["video_id"]), int(row["budget"])
        if (video, budget) in panel:
            raise ValueError("duplicate O1 video/budget row")
        if not math.isfinite(float(row["score"])) or not math.isfinite(float(row["cost"])):
            raise ValueError("O1 score/cost must be finite")
        panel[(video, budget)] = row
    videos = sorted({video for video, _budget in panel})
    budgets = sorted({budget for _video, budget in panel})
    if len(videos) < 3 or len(budgets) < 2:
        raise ValueError("O1 requires >=3 videos and >=2 budgets")
    if set(panel) != {(video, budget) for video in videos for budget in budgets}:
        raise ValueError("O1 budget panel must be rectangular")
    cost_by_budget = {}
    for budget in budgets:
        values = {float(panel[(video, budget)]["cost"]) for video in videos}
        if len(values) != 1:
            raise ValueError("O1 measured heavy-frame cost must be fixed for each K")
        value = values.pop()
        if abs(value - round(value)) > 1.0e-9:
            raise ValueError("O1 Oracle currently requires integer heavy-frame costs")
        cost_by_budget[budget] = int(round(value))
    score = {
        key: float(row["score"])
        for key, row in panel.items()
    }
    fixed_means = {
        budget: mean(score[(video, budget)] for video in videos)
        for budget in budgets
        if cost_by_budget[budget] <= float(target_mean_cost) + 1.0e-9
    }
    if not fixed_means:
        raise ValueError("O1 target is below every fixed budget")
    best_fixed = max(fixed_means, key=lambda budget: (fixed_means[budget], -budget))
    assignment = _oracle_assignment(
        videos,
        budgets,
        cost_by_budget,
        score,
        target_mean_cost=target_mean_cost,
    )
    realized_cost = mean(cost_by_budget[assignment[video]] for video in videos)
    if realized_cost > float(target_mean_cost) + 1.0e-9:
        raise RuntimeError("O1 Oracle exceeded the mean-cost contract")
    fixed_video = [score[(video, best_fixed)] for video in videos]
    oracle_video = [score[(video, assignment[video])] for video in videos]
    headroom = [
        oracle - fixed for oracle, fixed in zip(oracle_video, fixed_video)
    ]
    rng = random.Random(int(seed))
    shuffled_video_sums = [0.0] * len(videos)
    assigned_budgets = [assignment[video] for video in videos]
    shuffle_means = []
    for _ in range(max(1, int(shuffles))):
        shuffled = list(assigned_budgets)
        rng.shuffle(shuffled)
        values = [score[(video, budget)] for video, budget in zip(videos, shuffled)]
        shuffle_means.append(mean(values))
        for index, value in enumerate(values):
            shuffled_video_sums[index] += value
    shuffled_video = [
        value / max(1, int(shuffles)) for value in shuffled_video_sums
    ]
    oracle_vs_shuffle = [
        oracle - shuffled for oracle, shuffled in zip(oracle_video, shuffled_video)
    ]
    headroom_ci = cluster_bootstrap(
        headroom,
        videos,
        samples=bootstrap_samples,
        seed=seed + 1,
    )
    shuffle_ci = cluster_bootstrap(
        oracle_vs_shuffle,
        videos,
        samples=bootstrap_samples,
        seed=seed + 2,
    )
    gate = (
        headroom_ci["ci95_low"] > float(min_headroom)
        and shuffle_ci["ci95_low"] > 0.0
    )
    return {
        "schema_version": SUMMARY_SCHEMA,
        "stage": "o1_dynamic_budget_headroom",
        "video_count": len(videos),
        "budgets": budgets,
        "cost_by_budget": cost_by_budget,
        "target_mean_cost": float(target_mean_cost),
        "realized_oracle_mean_cost": realized_cost,
        "best_fixed_budget": best_fixed,
        "best_fixed_score": fixed_means[best_fixed],
        "oracle_score": mean(oracle_video),
        "oracle_assignment": assignment,
        "oracle_minus_best_fixed": headroom_ci,
        "oracle_minus_histogram_shuffle": shuffle_ci,
        "shuffle_score_distribution": {
            "mean": mean(shuffle_means),
            "p025": _percentile(shuffle_means, 0.025),
            "p975": _percentile(shuffle_means, 0.975),
            "repetitions": max(1, int(shuffles)),
        },
        "threshold": {"min_headroom": float(min_headroom)},
        "gate_pass": gate,
        "stop_if_failed": "do_not_train_dynamic_budget_controller",
    }


def analyze_o2(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_family: str,
    max_regret: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    panel = {}
    for row in rows:
        if row.get("schema_version") != "duca_rime_o2_decoder_panel_v1":
            raise ValueError("unsupported O2 schema")
        key = (str(row["video_id"]), int(row["budget"]), str(row["family"]))
        if key in panel:
            raise ValueError("duplicate O2 video/budget/family row")
        if "selection_keys" in row:
            positions = [str(value) for value in row["selection_keys"]]
            if (
                positions != sorted(set(positions))
                or not positions
                or row.get("exact_k_all_windows") is not True
            ):
                raise ValueError(
                    "O2 selection manifest must be ordered, unique, and exact-K"
                )
        else:
            positions = [int(value) for value in row["selected_positions"]]
            if positions != sorted(set(positions)) or len(positions) != int(
                row["budget"]
            ):
                raise ValueError("O2 selected positions must be ordered unique exact-K")
        if bool(row.get("max_gap_violation", False)):
            raise ValueError("O2 row violates the physical max-gap contract")
        panel[key] = row
    videos = sorted({key[0] for key in panel})
    budgets = sorted({key[1] for key in panel})
    families = sorted({key[2] for key in panel})
    if "independent" not in families or selected_family not in families:
        raise ValueError("O2 requires independent and the predeclared selected family")
    required = {
        (video, budget, family)
        for video in videos
        for budget in budgets
        for family in families
    }
    if set(panel) != required:
        raise ValueError("O2 decoder panel must be rectangular")
    video_regret = []
    video_overlap = []
    for video in videos:
        regrets = []
        overlaps = []
        previous = None
        for budget in budgets:
            independent = float(panel[(video, budget, "independent")]["score"])
            selected = float(panel[(video, budget, selected_family)]["score"])
            regrets.append(independent - selected)
            selected_row = panel[(video, budget, selected_family)]
            positions = set(
                str(value)
                for value in selected_row.get(
                    "selection_keys",
                    selected_row.get("selected_positions", ()),
                )
            )
            if previous is not None:
                overlaps.append(len(previous & positions) / float(len(previous)))
            previous = positions
        video_regret.append(mean(regrets))
        video_overlap.append(mean(overlaps) if overlaps else 1.0)
    regret_ci = cluster_bootstrap(
        video_regret,
        videos,
        samples=bootstrap_samples,
        seed=seed,
    )
    overlap_ci = cluster_bootstrap(
        video_overlap,
        videos,
        samples=bootstrap_samples,
        seed=seed + 1,
    )
    gate = regret_ci["ci95_high"] <= float(max_regret)
    return {
        "schema_version": SUMMARY_SCHEMA,
        "stage": "o2_decoder_family_regret",
        "video_count": len(videos),
        "budgets": budgets,
        "families": families,
        "selected_family": str(selected_family),
        "independent_minus_selected_regret": regret_ci,
        "consecutive_budget_overlap": overlap_ci,
        "threshold": {"max_regret": float(max_regret)},
        "gate_pass": gate,
        "stop_if_failed": f"drop_decoder_family_{selected_family}",
    }


def analyze_o3(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_spearman: float,
    null_margin: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row.get("schema_version") != "duca_rime_o3_rank_record_v1":
            raise ValueError("unsupported O3 schema")
        provenance = row.get("provenance")
        video = str(row["video_id"])
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("cross_fitted") is not True
            or provenance.get("uses_validation_or_test") is not False
            or video not in set(str(value) for value in provenance.get("eval_video_ids", ()))
            or video in set(str(value) for value in provenance.get("fit_video_ids", ()))
        ):
            raise ValueError("O3 requires explicit leakage-free cross-fit provenance")
        family = str(row["score_family"])
        predicted, actual = float(row["predicted_gain"]), float(row["actual_gain"])
        if not math.isfinite(predicted) or not math.isfinite(actual):
            raise ValueError("O3 gains must be finite")
        grouped[(family, video)].append((predicted, actual))
    families = sorted({family for family, _video in grouped})
    videos = sorted({video for _family, video in grouped})
    if "learned" not in families or len(videos) < 3:
        raise ValueError("O3 requires learned scores and >=3 evaluation videos")
    summaries = {}
    for family in families:
        correlations, directions, top_actual, regrets, correlation_videos = [], [], [], [], []
        for video in videos:
            pairs = grouped.get((family, video), ())
            if len(pairs) < 3:
                continue
            predicted = [pair[0] for pair in pairs]
            actual = [pair[1] for pair in pairs]
            correlations.append(spearman(predicted, actual))
            directions.append(
                mean(
                    (left >= 0.0) == (right >= 0.0)
                    for left, right in zip(predicted, actual)
                )
            )
            take = max(1, int(math.ceil(0.10 * len(pairs))))
            ranked = sorted(range(len(pairs)), key=lambda index: (-predicted[index], index))
            top_actual.append(mean(actual[index] for index in ranked[:take]))
            predicted_best = max(range(len(pairs)), key=lambda index: (predicted[index], -index))
            actual_best = max(actual)
            scale = max(actual) - min(actual)
            regrets.append(
                0.0
                if scale <= 1.0e-12
                else (actual_best - actual[predicted_best]) / scale
            )
            correlation_videos.append(video)
        if len(correlations) < 3:
            raise ValueError(f"O3 family {family} has fewer than three valid videos")
        summaries[family] = {
            "spearman": cluster_bootstrap(
                correlations,
                correlation_videos,
                samples=bootstrap_samples,
                seed=seed + len(summaries) * 10,
            ),
            "direction_accuracy": cluster_bootstrap(
                directions,
                correlation_videos,
                samples=bootstrap_samples,
                seed=seed + len(summaries) * 10 + 1,
            ),
            "top10_actual_gain": cluster_bootstrap(
                top_actual,
                correlation_videos,
                samples=bootstrap_samples,
                seed=seed + len(summaries) * 10 + 2,
            ),
            "normalized_regret": cluster_bootstrap(
                regrets,
                correlation_videos,
                samples=bootstrap_samples,
                seed=seed + len(summaries) * 10 + 3,
            ),
        }
    null_families = [family for family in families if family != "learned"]
    best_null = max(
        (
            summaries[family]["spearman"]["ci95_high"]
            for family in null_families
        ),
        default=0.0,
    )
    learned = summaries["learned"]["spearman"]
    gate = (
        learned["ci95_low"] > float(min_spearman)
        and learned["ci95_low"] > best_null + float(null_margin)
    )
    return {
        "schema_version": SUMMARY_SCHEMA,
        "stage": "o3_cross_fitted_hard_utility_rank",
        "families": summaries,
        "best_null_spearman_ci95_high": best_null,
        "threshold": {
            "min_spearman": float(min_spearman),
            "null_margin": float(null_margin),
        },
        "gate_pass": gate,
        "stop_if_failed": "drop_hard_utility_head_and_long_training",
    }


def _risk_metrics(labels: Sequence[int], scores: Sequence[float], bins: int) -> dict[str, float]:
    if len(labels) != len(scores) or not labels:
        raise ValueError("risk labels/scores must be nonempty and aligned")
    if int(bins) < 1:
        raise ValueError("risk calibration bins must be at least one")
    brier = mean((float(score) - int(label)) ** 2 for score, label in zip(scores, labels))
    ece = 0.0
    for index in range(int(bins)):
        low, high = index / bins, (index + 1) / bins
        members = [
            item
            for item, score in enumerate(scores)
            if score >= low and (score < high or (index == bins - 1 and score <= high))
        ]
        if members:
            ece += len(members) / len(scores) * abs(
                mean(scores[item] for item in members)
                - mean(labels[item] for item in members)
            )
    return {
        "brier": brier,
        "ece": ece,
        "sharpness_variance": mean((score - mean(scores)) ** 2 for score in scores),
        "prevalence": mean(labels),
    }


def analyze_o4(
    rows: Sequence[Mapping[str, Any]],
    *,
    risk_threshold: float,
    max_brier: float,
    max_ece: float,
    min_coverage: float,
    max_low_risk_failure: float,
    calibration_bins: int,
) -> dict[str, Any]:
    labels, scores, videos = [], [], []
    accepted, fallback = [], []
    for row in rows:
        if row.get("schema_version") != "duca_rime_o4_risk_record_v1":
            raise ValueError("unsupported O4 schema")
        provenance = row.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("fit_split") not in {"train", "training", "train_only"}
            or provenance.get("uses_validation_or_test") is not False
        ):
            raise ValueError("O4 risk records require train-only calibration provenance")
        predicted = float(row["predicted_risk"])
        observed = int(row["observed_pair_failure"])
        if not 0.0 <= predicted <= 1.0 or observed not in {0, 1}:
            raise ValueError("O4 risk score/label is invalid")
        requested = int(row["requested_k"])
        effective = int(row["effective_k"])
        unique = int(row["unique_k"])
        backbone = int(row["backbone_input_k"])
        padded = int(row["padded_k"])
        if not requested >= effective == unique == backbone == padded > 0:
            raise ValueError("O4 contains a pad-to-Kmax or inconsistent cost ledger")
        labels.append(observed)
        scores.append(predicted)
        videos.append(str(row["video_id"]))
        accepted.append(predicted <= float(risk_threshold))
        fallback.append(bool(row["risk_fallback"]))
    metrics = _risk_metrics(labels, scores, calibration_bins)
    accepted_labels = [
        label for label, keep in zip(labels, accepted) if keep
    ]
    coverage = sum(accepted) / len(accepted)
    low_risk_failure = (
        1.0 if not accepted_labels else mean(accepted_labels)
    )
    fallback_rate = sum(fallback) / len(fallback)
    gate = (
        metrics["brier"] <= float(max_brier)
        and metrics["ece"] <= float(max_ece)
        and coverage >= float(min_coverage)
        and low_risk_failure <= float(max_low_risk_failure)
    )
    return {
        "schema_version": SUMMARY_SCHEMA,
        "stage": "o4_pair_risk_calibration",
        "record_count": len(rows),
        "video_count": len(set(videos)),
        "risk_threshold": float(risk_threshold),
        "metrics": metrics,
        "low_risk_coverage": coverage,
        "low_risk_observed_failure": low_risk_failure,
        "risk_infeasible_fallback_rate": fallback_rate,
        "no_padding_ledger": True,
        "threshold": {
            "max_brier": float(max_brier),
            "max_ece": float(max_ece),
            "min_coverage": float(min_coverage),
            "max_low_risk_failure": float(max_low_risk_failure),
        },
        "gate_pass": gate,
        "stop_if_failed": "drop_pair_risk_contribution",
    }


def _choose_budget(
    utility: Sequence[float],
    risk: Sequence[float],
    costs: Sequence[float],
    *,
    price: float,
    risk_weight: float,
    risk_threshold: float,
) -> int:
    feasible = [value <= risk_threshold for value in risk]
    feasible[-1] = True
    if not any(feasible[:-1]):
        return len(costs) - 1
    normalized = [value / costs[-1] for value in costs]
    scores = [
        utility[index] - risk_weight * risk[index] - price * normalized[index]
        if feasible[index]
        else float("-inf")
        for index in range(len(costs))
    ]
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def freeze_protocol(
    *,
    summaries: Sequence[str | Path],
    calibration_jsonl: str | Path,
    output: str | Path,
    candidate_budgets: Sequence[int],
    candidate_costs: Sequence[float],
    target_mean_cost: float,
    risk_weight: float,
    risk_threshold: float,
    decoder_family: str,
    weak_overlap_fraction: float,
) -> dict[str, Any]:
    evidence = [json.loads(Path(path).read_text(encoding="utf-8")) for path in summaries]
    stages = {str(row.get("stage")): row for row in evidence}
    required = {
        "o1_dynamic_budget_headroom",
        "o2_decoder_family_regret",
        "o3_cross_fitted_hard_utility_rank",
        "o4_pair_risk_calibration",
    }
    if not required <= set(stages):
        raise ValueError("protocol freeze is missing one or more O1-O4 summaries")
    failed = [stage for stage in required if stages[stage].get("gate_pass") is not True]
    if failed:
        raise RuntimeError(f"RIME protocol cannot freeze because gates failed: {failed}")
    if stages["o2_decoder_family_regret"].get("selected_family") != decoder_family:
        raise ValueError("frozen decoder family disagrees with O2 evidence")
    budgets = tuple(int(value) for value in candidate_budgets)
    costs = tuple(float(value) for value in candidate_costs)
    if (
        len(budgets) < 2
        or tuple(sorted(set(budgets))) != budgets
        or budgets[0] <= 0
        or len(costs) != len(budgets)
        or any(not math.isfinite(value) or value <= 0.0 for value in costs)
        or any(right <= left for left, right in zip(costs[:-1], costs[1:]))
        or not costs[0] <= float(target_mean_cost) <= costs[-1]
        or not math.isfinite(float(risk_weight))
        or float(risk_weight) < 0.0
        or not 0.0 <= float(risk_threshold) <= 1.0
    ):
        raise ValueError("protocol candidate budgets/costs are invalid")
    calibration = _read_jsonl(calibration_jsonl)
    curves = []
    for row in calibration:
        if row.get("schema_version") != "duca_rime_price_calibration_v1":
            raise ValueError("unsupported RIME price calibration schema")
        provenance = row.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("fit_split") not in {"train", "training", "train_only"}
            or provenance.get("uses_validation_or_test") is not False
        ):
            raise ValueError("price calibration must be train-only")
        utility = tuple(float(value) for value in row["predicted_utility"])
        risk = tuple(float(value) for value in row["risk_upper"])
        if len(utility) != len(budgets) or len(risk) != len(budgets):
            raise ValueError("price calibration curves must align with candidate budgets")
        curves.append((utility, risk))

    def realized(price: float) -> tuple[float, list[int]]:
        selected = [
            _choose_budget(
                utility,
                risk,
                costs,
                price=price,
                risk_weight=risk_weight,
                risk_threshold=risk_threshold,
            )
            for utility, risk in curves
        ]
        return mean(costs[index] for index in selected), selected

    low, high = 0.0, 1.0
    high_mean, _ = realized(high)
    while high_mean > float(target_mean_cost) and high < 1.0e6:
        high *= 2.0
        high_mean, _ = realized(high)
    if high_mean > float(target_mean_cost):
        raise RuntimeError("frozen price cannot attain the requested mean cost")
    for _ in range(80):
        middle = 0.5 * (low + high)
        middle_mean, _ = realized(middle)
        if middle_mean <= float(target_mean_cost):
            high = middle
        else:
            low = middle
    realized_mean, selected = realized(high)
    payload = {
        "schema_version": PROTOCOL_SCHEMA,
        "fit_split": "train_only",
        "uses_validation_or_test_labels": False,
        "candidate_budgets": list(budgets),
        "candidate_costs": list(costs),
        "target_mean_cost": float(target_mean_cost),
        "realized_calibration_mean_cost": realized_mean,
        "frozen_price": high,
        "risk_weight": float(risk_weight),
        "risk_threshold": float(risk_threshold),
        "decoder_family": str(decoder_family),
        "weak_overlap_fraction": float(weak_overlap_fraction),
        "calibration_video_count": len(curves),
        "calibration_selected_indices": selected,
        "evidence_summaries": [
            {
                "path": str(Path(path).expanduser().resolve()),
                "sha256": _sha256_file(path),
                "stage": json.loads(Path(path).read_text(encoding="utf-8"))["stage"],
            }
            for path in summaries
        ],
        "calibration_jsonl": str(Path(calibration_jsonl).expanduser().resolve()),
        "calibration_jsonl_sha256": _sha256_file(calibration_jsonl),
        "gate_pass": True,
    }
    _write_json(output, payload)
    payload["output_path"] = str(Path(output).expanduser().resolve())
    payload["output_sha256"] = _sha256_file(output)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DUCA-RIME Phase 0/2 causal gates")
    sub = parser.add_subparsers(dest="command", required=True)

    phase0 = sub.add_parser("phase0")
    phase0.add_argument("--records-jsonl", required=True)
    phase0.add_argument("--output", required=True)
    phase0.add_argument("--primary-metric", default="avg_map")
    phase0.add_argument("--alpha", type=float, default=0.05)
    phase0.add_argument("--power", type=float, default=0.80)

    o1 = sub.add_parser("o1")
    o1.add_argument("--records-jsonl", required=True)
    o1.add_argument("--output", required=True)
    o1.add_argument("--target-mean-cost", type=float, required=True)
    o1.add_argument("--min-headroom", type=float, required=True)
    o1.add_argument("--bootstrap-samples", type=int, default=2000)
    o1.add_argument("--shuffles", type=int, default=2000)
    o1.add_argument("--seed", type=int, default=3407)

    o2 = sub.add_parser("o2")
    o2.add_argument("--records-jsonl", required=True)
    o2.add_argument("--output", required=True)
    o2.add_argument("--selected-family", required=True)
    o2.add_argument("--max-regret", type=float, required=True)
    o2.add_argument("--bootstrap-samples", type=int, default=2000)
    o2.add_argument("--seed", type=int, default=3407)

    o3 = sub.add_parser("o3")
    o3.add_argument("--records-jsonl", required=True)
    o3.add_argument("--output", required=True)
    o3.add_argument("--min-spearman", type=float, required=True)
    o3.add_argument("--null-margin", type=float, default=0.0)
    o3.add_argument("--bootstrap-samples", type=int, default=2000)
    o3.add_argument("--seed", type=int, default=3407)

    o4 = sub.add_parser("o4")
    o4.add_argument("--records-jsonl", required=True)
    o4.add_argument("--output", required=True)
    o4.add_argument("--risk-threshold", type=float, required=True)
    o4.add_argument("--max-brier", type=float, required=True)
    o4.add_argument("--max-ece", type=float, required=True)
    o4.add_argument("--min-coverage", type=float, required=True)
    o4.add_argument("--max-low-risk-failure", type=float, required=True)
    o4.add_argument("--calibration-bins", type=int, default=10)

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--summary", action="append", required=True)
    freeze.add_argument("--calibration-jsonl", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--candidate-budgets", nargs="+", type=int, required=True)
    freeze.add_argument("--candidate-costs", nargs="+", type=float, required=True)
    freeze.add_argument("--target-mean-cost", type=float, required=True)
    freeze.add_argument("--risk-weight", type=float, required=True)
    freeze.add_argument("--risk-threshold", type=float, required=True)
    freeze.add_argument("--decoder-family", required=True)
    freeze.add_argument("--weak-overlap-fraction", type=float, default=0.50)

    args = parser.parse_args(argv)
    if args.command == "phase0":
        result = phase0_variance(
            _read_jsonl(args.records_jsonl),
            primary_metric=args.primary_metric,
            alpha=args.alpha,
            power=args.power,
        )
        _write_json(args.output, result)
    elif args.command == "o1":
        result = analyze_o1(
            _read_jsonl(args.records_jsonl),
            target_mean_cost=args.target_mean_cost,
            min_headroom=args.min_headroom,
            bootstrap_samples=args.bootstrap_samples,
            shuffles=args.shuffles,
            seed=args.seed,
        )
        _write_json(args.output, result)
    elif args.command == "o2":
        result = analyze_o2(
            _read_jsonl(args.records_jsonl),
            selected_family=args.selected_family,
            max_regret=args.max_regret,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        _write_json(args.output, result)
    elif args.command == "o3":
        result = analyze_o3(
            _read_jsonl(args.records_jsonl),
            min_spearman=args.min_spearman,
            null_margin=args.null_margin,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        _write_json(args.output, result)
    elif args.command == "o4":
        result = analyze_o4(
            _read_jsonl(args.records_jsonl),
            risk_threshold=args.risk_threshold,
            max_brier=args.max_brier,
            max_ece=args.max_ece,
            min_coverage=args.min_coverage,
            max_low_risk_failure=args.max_low_risk_failure,
            calibration_bins=args.calibration_bins,
        )
        _write_json(args.output, result)
    else:
        result = freeze_protocol(
            summaries=args.summary,
            calibration_jsonl=args.calibration_jsonl,
            output=args.output,
            candidate_budgets=args.candidate_budgets,
            candidate_costs=args.candidate_costs,
            target_mean_cost=args.target_mean_cost,
            risk_weight=args.risk_weight,
            risk_threshold=args.risk_threshold,
            decoder_family=args.decoder_family,
            weak_overlap_fraction=args.weak_overlap_fraction,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
