from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


RECORD_SCHEMA_VERSION = "duca_selection_quality_record_v2"
SUMMARY_SCHEMA_VERSION = "duca_selection_quality_summary_v2"
RADII = (0, 1, 2, 4, 8)
SELECTION_METHODS = (
    "learned",
    "uniform",
    "stratified_random",
    "utility_topk_diagnostic",
    "pure_delta_topk_diagnostic",
    "raw_transition_topk_diagnostic",
)


def _finite(values: Iterable[float | int | None]) -> list[float]:
    out = []
    for value in values:
        if value is not None and math.isfinite(float(value)):
            out.append(float(value))
    return out


def _mean(values: Iterable[float | int | None]) -> float | None:
    finite = _finite(values)
    return None if not finite else mean(finite)


def _percentile(values: Iterable[float | int | None], q: float) -> float | None:
    finite = sorted(_finite(values))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    rank = (len(finite) - 1) * float(q)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return finite[low]
    return finite[low] + (finite[high] - finite[low]) * (rank - low)


def _round(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, digits)
    if isinstance(value, Mapping):
        return {str(key): _round(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(item, digits) for item in value]
    return value


def exact_uniform_positions(valid_len: int, budget: int) -> list[int]:
    valid_len = max(0, int(valid_len))
    budget = min(max(0, int(budget)), valid_len)
    if budget <= 0:
        return []
    if budget == 1:
        return [0]
    positions = [int(round(idx * (valid_len - 1) / float(budget - 1))) for idx in range(budget)]
    if len(set(positions)) != budget:
        raise ValueError("round-linspace reference unexpectedly produced duplicate positions")
    return positions


def stratified_random_positions(valid_len: int, budget: int, seed: int) -> list[int]:
    valid_len = max(0, int(valid_len))
    budget = min(max(0, int(budget)), valid_len)
    if budget <= 0:
        return []
    rng = random.Random(int(seed))
    available = set(range(valid_len))
    selected: list[int] = []
    for slot in range(budget):
        low = int(math.floor(slot * valid_len / float(budget)))
        high = int(math.floor((slot + 1) * valid_len / float(budget))) - 1
        candidates = [idx for idx in range(low, max(low, high) + 1) if idx in available]
        if not candidates:
            candidates = sorted(available)
        choice = rng.choice(candidates)
        selected.append(choice)
        available.remove(choice)
    return sorted(selected)


def _rank_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(int(label) == 1 for label in labels)
    negatives = len(labels) - positives
    if positives <= 0 or negatives <= 0:
        return None
    order = sorted(range(len(scores)), key=lambda idx: float(scores[idx]))
    rank_sum = 0.0
    cursor = 0
    while cursor < len(order):
        stop = cursor + 1
        while stop < len(order) and float(scores[order[stop]]) == float(scores[order[cursor]]):
            stop += 1
        average_rank = 0.5 * ((cursor + 1) + stop)
        rank_sum += average_rank * sum(int(labels[order[idx]]) == 1 for idx in range(cursor, stop))
        cursor = stop
    return (rank_sum - positives * (positives + 1) / 2.0) / float(positives * negatives)


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(int(label) == 1 for label in labels)
    if positives <= 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda pair: float(pair[0]), reverse=True)
    hits = 0
    precision_sum = 0.0
    for rank, (_score, label) in enumerate(ranked, start=1):
        if int(label) == 1:
            hits += 1
            precision_sum += hits / float(rank)
    return precision_sum / float(positives)


def binary_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    calibration_bins: int = 10,
    threshold: float = 0.5,
    calibrated: bool = True,
) -> dict[str, Any]:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have identical length")
    pairs = [
        (int(label), float(score))
        for label, score in zip(labels, scores)
        if int(label) in {0, 1} and math.isfinite(float(score))
    ]
    if not pairs:
        return {
            "count": 0,
            "positive_count": 0,
            "prevalence": None,
            "auroc": None,
            "auprc": None,
            "auprc_lift": None,
            "brier": None,
            "ece": None,
            "balanced_accuracy_at_0_5": None,
            "f1_at_0_5": None,
        }
    clean_labels = [item[0] for item in pairs]
    clean_scores = [item[1] for item in pairs]
    count = len(pairs)
    positives = sum(clean_labels)
    prevalence = positives / float(count)
    auprc = _average_precision(clean_labels, clean_scores)
    probability_scores = [min(1.0, max(0.0, score)) for score in clean_scores] if calibrated else []
    predictions = [int(score >= float(threshold)) for score in probability_scores]
    tp = sum(pred == 1 and label == 1 for pred, label in zip(predictions, clean_labels)) if calibrated else 0
    tn = sum(pred == 0 and label == 0 for pred, label in zip(predictions, clean_labels)) if calibrated else 0
    fp = sum(pred == 1 and label == 0 for pred, label in zip(predictions, clean_labels)) if calibrated else 0
    fn = sum(pred == 0 and label == 1 for pred, label in zip(predictions, clean_labels)) if calibrated else 0
    tpr = None if not calibrated or tp + fn == 0 else tp / float(tp + fn)
    tnr = None if not calibrated or tn + fp == 0 else tn / float(tn + fp)
    balanced = None if tpr is None or tnr is None else 0.5 * (tpr + tnr)
    f1 = None if not calibrated or 2 * tp + fp + fn == 0 else 2 * tp / float(2 * tp + fp + fn)
    brier = None if not calibrated else mean((score - label) ** 2 for score, label in zip(probability_scores, clean_labels))
    ece = 0.0
    bins = max(1, int(calibration_bins))
    for idx in range(bins if calibrated else 0):
        low = idx / float(bins)
        high = (idx + 1) / float(bins)
        members = [
            pos
            for pos, score in enumerate(probability_scores)
            if score >= low and (score < high or (idx == bins - 1 and score <= high))
        ]
        if not members:
            continue
        confidence = mean(probability_scores[pos] for pos in members)
        accuracy = mean(clean_labels[pos] for pos in members)
        ece += len(members) / float(count) * abs(confidence - accuracy)
    return {
        "count": count,
        "positive_count": positives,
        "prevalence": prevalence,
        "auroc": _rank_auroc(clean_labels, clean_scores),
        "auprc": auprc,
        "auprc_lift": None if auprc is None or prevalence <= 0.0 else auprc / prevalence,
        "brier": brier,
        "ece": ece if calibrated else None,
        "balanced_accuracy_at_0_5": balanced,
        "f1_at_0_5": f1,
    }


def _validated_segments(record: Mapping[str, Any], valid_len: int) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for raw in record.get("gt_segments", []):
        if not isinstance(raw, Sequence) or len(raw) != 2:
            raise ValueError(f"{record.get('sample_id')}: each gt segment must be [start,end]")
        start, end = sorted((float(raw[0]), float(raw[1])))
        # Segment coordinates are half-open; an end coordinate equal to T is valid.
        start = min(float(valid_len), max(0.0, start)) if valid_len else 0.0
        end = min(float(valid_len), max(0.0, end)) if valid_len else 0.0
        if end > start:
            out.append((start, end))
    return out


def _action_labels(valid_len: int, segments: Sequence[tuple[float, float]]) -> list[int]:
    return [int(any(float(idx) >= start and float(idx) < end for start, end in segments)) for idx in range(valid_len)]


def _boundaries(valid_len: int, segments: Sequence[tuple[float, float]]) -> list[float]:
    values: list[float] = []
    for start, end in segments:
        values.extend((max(0.0, min(valid_len - 1.0, start)), max(0.0, min(valid_len - 1.0, end))))
    return values


def _boundary_labels(valid_len: int, boundaries: Sequence[float], radius: int) -> list[int]:
    return [int(any(abs(float(idx) - boundary) <= int(radius) for boundary in boundaries)) for idx in range(valid_len)]


def _max_unselected_hole(valid_len: int, positions: Sequence[int]) -> int:
    selected = sorted(set(int(item) for item in positions if 0 <= int(item) < valid_len))
    if not selected:
        return valid_len
    holes = [selected[0], valid_len - 1 - selected[-1]]
    holes.extend(max(0, right - left - 1) for left, right in zip(selected, selected[1:]))
    return max(holes, default=0)


def _topk_positions(scores: Sequence[float], budget: int) -> list[int]:
    ranked = sorted(range(len(scores)), key=lambda idx: (-float(scores[idx]), idx))
    return sorted(ranked[: min(max(0, int(budget)), len(ranked))])


def _selection_metrics(
    *,
    valid_len: int,
    positions: Sequence[int],
    segments: Sequence[tuple[float, float]],
    boundaries: Sequence[float],
) -> dict[str, Any]:
    selected = sorted(set(int(item) for item in positions))
    if any(item < 0 or item >= valid_len for item in selected):
        raise ValueError("selected positions must lie inside the valid prefix")
    boundary_recall: dict[str, float | None] = {}
    boundary_precision: dict[str, float | None] = {}
    both_endpoint_coverage: dict[str, float | None] = {}
    for radius in RADII:
        boundary_recall[f"r{radius}"] = (
            None
            if not boundaries
            else mean(any(abs(float(pos) - boundary) <= radius for pos in selected) for boundary in boundaries)
        )
        boundary_precision[f"r{radius}"] = (
            None
            if not selected or not boundaries
            else mean(any(abs(float(pos) - boundary) <= radius for boundary in boundaries) for pos in selected)
        )
        both_endpoint_coverage[f"r{radius}"] = (
            None
            if not segments
            else mean(
                any(
                    abs(float(pos) - max(0.0, min(valid_len - 1.0, start))) <= radius
                    for pos in selected
                )
                and any(
                    abs(float(pos) - max(0.0, min(valid_len - 1.0, end))) <= radius
                    for pos in selected
                )
                for start, end in segments
            )
        )
    nearest = [min(abs(float(pos) - boundary) for pos in selected) for boundary in boundaries] if selected else []
    action_selected = [
        any(float(pos) >= start and float(pos) < end for start, end in segments)
        for pos in selected
    ]
    action_prevalence = _mean(_action_labels(valid_len, segments))
    action_fraction = _mean(action_selected)
    gaps = [right - left for left, right in zip(selected, selected[1:])]
    return {
        "selected_positions": selected,
        "selected_count": len(selected),
        "boundary_recall": boundary_recall,
        "boundary_precision": boundary_precision,
        "both_endpoint_coverage": both_endpoint_coverage,
        "mean_endpoint_distance": _mean(nearest),
        "median_endpoint_distance": _percentile(nearest, 0.5),
        "p90_endpoint_distance": _percentile(nearest, 0.9),
        "max_unselected_hole": _max_unselected_hole(valid_len, selected),
        "mean_selected_gap": _mean(gaps),
        "p95_selected_gap": _percentile(gaps, 0.95),
        "action_selected_fraction": action_fraction,
        "action_prevalence": action_prevalence,
        "action_enrichment": (
            None
            if action_fraction is None or action_prevalence in {None, 0.0}
            else action_fraction / float(action_prevalence)
        ),
    }


def _score_vector(record: Mapping[str, Any], key: str, valid_len: int) -> list[float]:
    values = record.get(key)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{record.get('sample_id')}: missing score vector {key}")
    if len(values) < valid_len:
        raise ValueError(f"{record.get('sample_id')}: {key} is shorter than valid_len")
    return [float(item) for item in values[:valid_len]]


def analyze_record(record: Mapping[str, Any], *, random_seed: int = 0) -> dict[str, Any]:
    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise ValueError(f"unsupported record schema {record.get('schema_version')!r}")
    sample_id = str(record.get("sample_id") or "")
    if not sample_id:
        raise ValueError("record is missing sample_id")
    valid_len = int(record.get("valid_len", 0))
    budget = min(int(record.get("budget", 0)), valid_len)
    if valid_len <= 0 or budget <= 0:
        raise ValueError(f"{sample_id}: valid_len and budget must be positive")
    segments = _validated_segments(record, valid_len)
    boundaries = _boundaries(valid_len, segments)
    action_labels = _action_labels(valid_len, segments)
    p_action = _score_vector(record, "p_action", valid_len)
    policy_scores = _score_vector(record, "transition_policy_scores", valid_len)
    raw_scores = _score_vector(record, "raw_transition_scores", valid_len)
    pure_delta_scores = _score_vector(record, "abs_delta_p_action", valid_len)
    selected = [int(item) for item in record.get("selected_positions", []) if int(item) >= 0]
    if len(selected) != budget:
        raise ValueError(f"{sample_id}: selected_count={len(selected)} does not match budget={budget}")
    methods = {
        "learned": selected,
        "uniform": exact_uniform_positions(valid_len, budget),
        "stratified_random": stratified_random_positions(valid_len, budget, seed=random_seed + sum(map(ord, sample_id))),
        "utility_topk_diagnostic": _topk_positions(policy_scores, budget),
        "pure_delta_topk_diagnostic": _topk_positions(pure_delta_scores, budget),
        "raw_transition_topk_diagnostic": _topk_positions(raw_scores, budget),
    }
    selection = {
        name: _selection_metrics(
            valid_len=valid_len,
            positions=positions,
            segments=segments,
            boundaries=boundaries,
        )
        for name, positions in methods.items()
    }
    transition: dict[str, Any] = {}
    for radius in RADII:
        labels = _boundary_labels(valid_len, boundaries, radius)
        transition[f"r{radius}"] = {
            "policy": binary_metrics(labels, policy_scores, calibrated=False),
            "pure_abs_delta_p_action": binary_metrics(labels, pure_delta_scores, calibrated=False),
            "raw_actionness_transition": binary_metrics(labels, raw_scores, calibrated=False),
        }
    learned_distance = selection["learned"]["mean_endpoint_distance"]
    uniform_distance = selection["uniform"]["mean_endpoint_distance"]
    gain = (
        0.0
        if learned_distance is None or uniform_distance is None
        else float(uniform_distance) - float(learned_distance)
    )
    return {
        "sample_id": sample_id,
        "video_id": str(record.get("video_id") or sample_id.split("|")[0]),
        "valid_len": valid_len,
        "budget": budget,
        "gt_segment_count": len(segments),
        "gt_boundary_count": len(boundaries),
        "action_labels": action_labels,
        "p_action": p_action,
        "transition_policy_scores": policy_scores,
        "abs_delta_p_action": pure_delta_scores,
        "raw_transition_scores": raw_scores,
        "gt_segments": [[start, end] for start, end in segments],
        "coarse": binary_metrics(action_labels, p_action),
        "transition": transition,
        "selection": selection,
        "selection_gain_vs_uniform": gain,
        "source": dict(record.get("source", {})),
    }


def choose_representative_samples(rows: Sequence[Mapping[str, Any]], *, per_stratum: int = 2) -> list[dict[str, Any]]:
    eligible = [row for row in rows if int(row.get("gt_boundary_count", 1)) > 0]
    if not eligible or per_stratum <= 0:
        return []
    ordered = sorted(eligible, key=lambda row: (float(row["selection_gain_vs_uniform"]), str(row["sample_id"])))
    count = min(int(per_stratum), len(ordered))
    groups = [
        ("best", list(reversed(ordered[-count:]))),
        (
            "median",
            sorted(
                ordered,
                key=lambda row: (abs(float(row["selection_gain_vs_uniform"]) - median(float(item["selection_gain_vs_uniform"]) for item in ordered)), str(row["sample_id"])),
            )[:count],
        ),
        ("worst", ordered[:count]),
    ]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for stratum, candidates in groups:
        for candidate in candidates:
            sample_id = str(candidate["sample_id"])
            if sample_id in seen and len(eligible) >= count * 3:
                continue
            row = dict(candidate)
            row["sample_stratum"] = stratum
            out.append(row)
            seen.add(sample_id)
    return out


def _bootstrap_ci(
    values: Sequence[float | int | None],
    *,
    samples: int,
    seed: int,
    clusters: Sequence[str] | None = None,
) -> dict[str, float | int | None]:
    if clusters is not None and len(clusters) != len(values):
        raise ValueError("bootstrap clusters must align with values")
    finite_pairs = [
        (str(clusters[idx]) if clusters is not None else str(idx), float(value))
        for idx, value in enumerate(values)
        if value is not None and math.isfinite(float(value))
    ]
    finite = [value for _cluster, value in finite_pairs]
    if not finite:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0, "cluster_n": 0}
    rng = random.Random(int(seed))
    grouped: dict[str, list[float]] = {}
    for cluster, value in finite_pairs:
        grouped.setdefault(cluster, []).append(value)
    cluster_ids = sorted(grouped)
    estimates = []
    for _ in range(max(1, int(samples))):
        sampled = [rng.choice(cluster_ids) for _idx in range(len(cluster_ids))]
        draw = [value for cluster in sampled for value in grouped[cluster]]
        estimates.append(mean(draw))
    return {
        "mean": mean(finite),
        "ci95_low": _percentile(estimates, 0.025),
        "ci95_high": _percentile(estimates, 0.975),
        "n": len(finite),
        "cluster_n": len(cluster_ids),
    }


def _flatten_sample_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sample_id": row["sample_id"],
        "video_id": row["video_id"],
        "valid_len": row["valid_len"],
        "budget": row["budget"],
        "gt_segment_count": row["gt_segment_count"],
        "coarse_auroc": row["coarse"]["auroc"],
        "coarse_auprc": row["coarse"]["auprc"],
        "coarse_auprc_lift": row["coarse"]["auprc_lift"],
        "coarse_brier": row["coarse"]["brier"],
        "coarse_ece": row["coarse"]["ece"],
        "transition_policy_auprc_r0": row["transition"]["r0"]["policy"]["auprc"],
        "transition_policy_auprc_r4": row["transition"]["r4"]["policy"]["auprc"],
        "transition_pure_delta_auprc_r0": row["transition"]["r0"]["pure_abs_delta_p_action"]["auprc"],
        "transition_pure_delta_auprc_r4": row["transition"]["r4"]["pure_abs_delta_p_action"]["auprc"],
        "transition_compound_auprc_r0": row["transition"]["r0"]["raw_actionness_transition"]["auprc"],
        "transition_compound_auprc_r4": row["transition"]["r4"]["raw_actionness_transition"]["auprc"],
        "selection_gain_vs_uniform": row["selection_gain_vs_uniform"],
    }
    for method in ("learned", "uniform", "stratified_random"):
        metrics = row["selection"][method]
        out[f"{method}_mean_endpoint_distance"] = metrics["mean_endpoint_distance"]
        out[f"{method}_max_unselected_hole"] = metrics["max_unselected_hole"]
        out[f"{method}_action_enrichment"] = metrics["action_enrichment"]
        for radius in RADII:
            out[f"{method}_boundary_recall_r{radius}"] = metrics["boundary_recall"][f"r{radius}"]
            out[f"{method}_both_endpoint_r{radius}"] = metrics["both_endpoint_coverage"][f"r{radius}"]
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_curve(values: Sequence[float]) -> list[float]:
    finite = _finite(values)
    if not finite:
        return [0.0 for _ in values]
    low, high = min(finite), max(finite)
    if high <= low:
        return [0.5 for _ in values]
    return [(float(item) - low) / (high - low) for item in values]


def _plot_outputs(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], out: Path, representatives: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})
    action_scores = [score for row in rows for score, label in zip(row["p_action"], row["action_labels"]) if label == 1]
    background_scores = [score for row in rows for score, label in zip(row["p_action"], row["action_labels"]) if label == 0]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    ax = axes[0, 0]
    ax.hist(background_scores, bins=20, range=(0, 1), density=True, alpha=0.65, label="background", color="#4C78A8")
    ax.hist(action_scores, bins=20, range=(0, 1), density=True, alpha=0.65, label="action", color="#F58518")
    ax.set(xlabel="p(action)", ylabel="density")
    ax.legend(frameon=False)
    ax.text(-0.14, 1.05, "A", transform=ax.transAxes, fontweight="bold")

    ax = axes[0, 1]
    all_scores = [score for row in rows for score in row["p_action"]]
    all_labels = [label for row in rows for label in row["action_labels"]]
    xs, ys, ns = [], [], []
    for idx in range(10):
        members = [pos for pos, score in enumerate(all_scores) if idx / 10.0 <= score < (idx + 1) / 10.0 or (idx == 9 and score == 1.0)]
        if members:
            xs.append(mean(all_scores[pos] for pos in members))
            ys.append(mean(all_labels[pos] for pos in members))
            ns.append(len(members))
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    ax.scatter(xs, ys, s=[max(12, 80 * n / max(ns)) for n in ns], color="#54A24B", edgecolor="white", linewidth=0.4)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="confidence", ylabel="empirical action rate")
    ax.text(-0.14, 1.05, "B", transform=ax.transAxes, fontweight="bold")

    ax = axes[1, 0]
    radii = list(RADII)
    policy = [summary["transition"][f"r{radius}"]["policy"]["auprc"] for radius in radii]
    pure_delta = [summary["transition"][f"r{radius}"]["pure_abs_delta_p_action"]["auprc"] for radius in radii]
    raw = [summary["transition"][f"r{radius}"]["raw_actionness_transition"]["auprc"] for radius in radii]
    ax.plot(radii, policy, marker="o", label="learned transition utility", color="#E45756")
    ax.plot(radii, pure_delta, marker="^", label="pure |delta p(action)|", color="#4C78A8")
    ax.plot(radii, raw, marker="s", label="compound transition proxy", color="#72B7B2")
    ax.set(xlabel="GT boundary radius", ylabel="AUPRC", xticks=radii, ylim=(0, 1))
    ax.legend(frameon=False)
    ax.text(-0.14, 1.05, "C", transform=ax.transAxes, fontweight="bold")

    ax = axes[1, 1]
    methods = ["learned", "uniform", "stratified_random"]
    labels = ["learned", "uniform", "strat. random"]
    colors = ["#E45756", "#4C78A8", "#B279A2"]
    x = list(range(len(RADII)))
    width = 0.24
    for offset, method in enumerate(methods):
        values = [summary["selection"][method]["boundary_recall"][f"r{radius}"]["mean"] for radius in RADII]
        ax.bar([item + (offset - 1) * width for item in x], values, width=width, color=colors[offset], label=labels[offset])
    ax.set(xlabel="GT boundary radius", ylabel="endpoint recall", xticks=x, xticklabels=[str(r) for r in RADII], ylim=(0, 1.05))
    ax.legend(frameon=False, ncol=1)
    ax.text(-0.14, 1.05, "D", transform=ax.transAxes, fontweight="bold")
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"selection_quality_overview.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if not representatives:
        return
    fig, axes = plt.subplots(len(representatives), 1, figsize=(9.0, max(2.1, 1.65 * len(representatives))), squeeze=False, constrained_layout=True)
    for ax, row in zip(axes[:, 0], representatives):
        x = list(range(int(row["valid_len"])))
        ax.plot(x, row["p_action"], color="#F58518", linewidth=1.1, label="p(action)")
        ax.plot(x, _normalize_curve(row["transition_policy_scores"]), color="#E45756", linewidth=1.0, label="transition utility")
        for start, end in row["gt_segments"]:
            ax.axvspan(start, end, color="#54A24B", alpha=0.14, linewidth=0)
        learned = row["selection"]["learned"]["selected_positions"]
        uniform = row["selection"]["uniform"]["selected_positions"]
        ax.scatter(learned, [-0.08] * len(learned), marker="|", s=18, color="#E45756", linewidth=0.8, label="learned")
        ax.scatter(uniform, [-0.18] * len(uniform), marker="|", s=18, color="#4C78A8", linewidth=0.8, label="uniform")
        ax.set(xlim=(0, max(1, row["valid_len"] - 1)), ylim=(-0.25, 1.02), ylabel=row["sample_stratum"])
        ax.text(0.005, 0.95, f"{row['sample_id']}  gain={row['selection_gain_vs_uniform']:+.3f}", transform=ax.transAxes, va="top", fontsize=7)
    axes[-1, 0].set_xlabel("dense temporal index")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:4], labels[:4], loc="upper center", ncol=4, frameon=False)
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"selection_quality_samples.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_jsonl(
    *,
    records_jsonl: str | Path,
    output_dir: str | Path,
    bootstrap_samples: int = 2000,
    random_seed: int = 0,
    representative_per_stratum: int = 2,
) -> dict[str, Any]:
    records_path = Path(records_jsonl).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [analyze_record(record, random_seed=random_seed) for record in records]
    if not rows:
        raise ValueError("records JSONL is empty")
    pooled_action_labels = [label for row in rows for label in row["action_labels"]]
    pooled_action_scores = [score for row in rows for score in row["p_action"]]
    video_clusters = [str(row["video_id"]) for row in rows]
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "records_jsonl": str(records_path),
        "sample_count": len(rows),
        "video_count": len({str(row["video_id"]) for row in rows}),
        "frame_observation_count": len(pooled_action_labels),
        "coarse": {
            "pooled": binary_metrics(pooled_action_labels, pooled_action_scores),
            "macro_auroc": _bootstrap_ci([row["coarse"]["auroc"] for row in rows], samples=bootstrap_samples, seed=random_seed + 1, clusters=video_clusters),
            "macro_auprc": _bootstrap_ci([row["coarse"]["auprc"] for row in rows], samples=bootstrap_samples, seed=random_seed + 2, clusters=video_clusters),
            "macro_auprc_lift": _bootstrap_ci([row["coarse"]["auprc_lift"] for row in rows], samples=bootstrap_samples, seed=random_seed + 3, clusters=video_clusters),
        },
        "transition": {},
        "selection": {},
        "comparison": {},
        "protocol": {
            "task": "offline_tad_diagnostic",
            "gt_role": "evaluation_only_not_selection",
            "uniform_reference": "round_linspace_endpoints",
            "budget_matched": True,
            "valid_length_matched": True,
            "bootstrap_unit": "validation_video_cluster",
            "random_baseline": "one_sample_per_temporal_stratum",
        },
    }
    for radius in RADII:
        labels = [label for row in rows for label in _boundary_labels(row["valid_len"], _boundaries(row["valid_len"], [tuple(seg) for seg in row["gt_segments"]]), radius)]
        policy = [score for row in rows for score in row["transition_policy_scores"]]
        pure_delta = [score for row in rows for score in row["abs_delta_p_action"]]
        raw = [score for row in rows for score in row["raw_transition_scores"]]
        summary["transition"][f"r{radius}"] = {
            "policy": binary_metrics(labels, policy, calibrated=False),
            "pure_abs_delta_p_action": binary_metrics(labels, pure_delta, calibrated=False),
            "raw_actionness_transition": binary_metrics(labels, raw, calibrated=False),
            "macro_policy_auprc": _bootstrap_ci(
                [row["transition"][f"r{radius}"]["policy"]["auprc"] for row in rows],
                samples=bootstrap_samples,
                seed=random_seed + 10 + radius,
                clusters=video_clusters,
            ),
        }
    for method in SELECTION_METHODS:
        method_summary: dict[str, Any] = {
            "selected_count": _bootstrap_ci([row["selection"][method]["selected_count"] for row in rows], samples=bootstrap_samples, seed=random_seed + 20, clusters=video_clusters),
            "mean_endpoint_distance": _bootstrap_ci([row["selection"][method]["mean_endpoint_distance"] for row in rows], samples=bootstrap_samples, seed=random_seed + 21, clusters=video_clusters),
            "max_unselected_hole": _bootstrap_ci([row["selection"][method]["max_unselected_hole"] for row in rows], samples=bootstrap_samples, seed=random_seed + 22, clusters=video_clusters),
            "action_enrichment": _bootstrap_ci([row["selection"][method]["action_enrichment"] for row in rows], samples=bootstrap_samples, seed=random_seed + 23, clusters=video_clusters),
            "boundary_recall": {},
            "boundary_precision": {},
            "both_endpoint_coverage": {},
            "pooled": {"boundary_recall": {}, "both_endpoint_coverage": {}},
            "diagnostic_only": method.endswith("_diagnostic"),
        }
        for radius in RADII:
            for field in ("boundary_recall", "boundary_precision", "both_endpoint_coverage"):
                method_summary[field][f"r{radius}"] = _bootstrap_ci(
                    [row["selection"][method][field][f"r{radius}"] for row in rows],
                    samples=bootstrap_samples,
                    seed=random_seed + 30 + radius,
                    clusters=video_clusters,
                )
            boundary_hits = 0
            boundary_total = 0
            both_hits = 0
            instance_total = 0
            for row in rows:
                positions = row["selection"][method]["selected_positions"]
                segments = [tuple(segment) for segment in row["gt_segments"]]
                boundaries = _boundaries(row["valid_len"], segments)
                boundary_hits += sum(
                    any(abs(float(position) - boundary) <= radius for position in positions)
                    for boundary in boundaries
                )
                boundary_total += len(boundaries)
                both_hits += sum(
                    any(abs(float(position) - start) <= radius for position in positions)
                    and any(abs(float(position) - min(float(row["valid_len"] - 1), end)) <= radius for position in positions)
                    for start, end in segments
                )
                instance_total += len(segments)
            method_summary["pooled"]["boundary_recall"][f"r{radius}"] = (
                None if boundary_total == 0 else boundary_hits / float(boundary_total)
            )
            method_summary["pooled"]["both_endpoint_coverage"][f"r{radius}"] = (
                None if instance_total == 0 else both_hits / float(instance_total)
            )
        summary["selection"][method] = method_summary
    summary["comparison"] = {
        "learned_minus_uniform_boundary_recall_r0": (
            summary["selection"]["learned"]["boundary_recall"]["r0"]["mean"]
            - summary["selection"]["uniform"]["boundary_recall"]["r0"]["mean"]
        ),
        "learned_minus_uniform_boundary_recall_r1": (
            summary["selection"]["learned"]["boundary_recall"]["r1"]["mean"]
            - summary["selection"]["uniform"]["boundary_recall"]["r1"]["mean"]
        ),
        "uniform_minus_learned_endpoint_distance": _mean([row["selection_gain_vs_uniform"] for row in rows]),
        "learned_minus_uniform_max_hole": (
            summary["selection"]["learned"]["max_unselected_hole"]["mean"]
            - summary["selection"]["uniform"]["max_unselected_hole"]["mean"]
        ),
        "paired_uniform_minus_learned_endpoint_distance": _bootstrap_ci(
            [row["selection_gain_vs_uniform"] for row in rows],
            samples=bootstrap_samples,
            seed=random_seed + 70,
            clusters=video_clusters,
        ),
        "paired_learned_minus_uniform_boundary_recall_r0": _bootstrap_ci(
            [
                row["selection"]["learned"]["boundary_recall"]["r0"]
                - row["selection"]["uniform"]["boundary_recall"]["r0"]
                if row["selection"]["learned"]["boundary_recall"]["r0"] is not None
                and row["selection"]["uniform"]["boundary_recall"]["r0"] is not None
                else None
                for row in rows
            ],
            samples=bootstrap_samples,
            seed=random_seed + 71,
            clusters=video_clusters,
        ),
        "paired_learned_minus_uniform_boundary_recall_r1": _bootstrap_ci(
            [
                row["selection"]["learned"]["boundary_recall"]["r1"]
                - row["selection"]["uniform"]["boundary_recall"]["r1"]
                if row["selection"]["learned"]["boundary_recall"]["r1"] is not None
                and row["selection"]["uniform"]["boundary_recall"]["r1"] is not None
                else None
                for row in rows
            ],
            samples=bootstrap_samples,
            seed=random_seed + 72,
            clusters=video_clusters,
        ),
    }
    representatives = choose_representative_samples(rows, per_stratum=representative_per_stratum)
    summary["representative_samples"] = [
        {"sample_id": row["sample_id"], "stratum": row["sample_stratum"], "selection_gain_vs_uniform": row["selection_gain_vs_uniform"]}
        for row in representatives
    ]
    summary = _round(summary)
    (out / "selection_quality_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out / "selection_quality_per_sample.csv", [_round(_flatten_sample_row(row)) for row in rows])
    with (out / "selection_quality_analyzed.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_round(row), sort_keys=True) + "\n")
    _plot_outputs(rows, summary, out, representatives)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze DUCA coarse classification, transition utility, and frame selection quality.")
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--representative-per-stratum", type=int, default=2)
    args = parser.parse_args(argv)
    summary = analyze_jsonl(
        records_jsonl=args.records_jsonl,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
        representative_per_stratum=args.representative_per_stratum,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
