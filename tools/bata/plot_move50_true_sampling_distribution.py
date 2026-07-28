from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SAMPLE_JSON = (
    Path("analysis_outputs")
    / "lattice_budgeted_radius_fb7419c_20260708_150121"
    / "extracted_video_test_0000188_6144.source.json"
)
DEFAULT_OUTPUT_DIR = Path("analysis_outputs") / "move50_true_sampling_distribution_20260709"


def _scale01(values: Sequence[Any], length: int) -> np.ndarray:
    arr = np.asarray(list(values)[:length], dtype=float)
    if arr.size < length:
        arr = np.pad(arr, (0, length - arr.size), mode="constant")
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(length, dtype=float)
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if hi <= lo:
        return np.zeros(length, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _uniform_reference_positions(*, valid_len: int, selected_count: int) -> list[int]:
    valid_len = int(valid_len)
    selected_count = int(selected_count)
    if valid_len <= 0 or selected_count <= 0:
        return []
    if selected_count >= valid_len:
        return list(range(valid_len))
    step = float(valid_len) / float(selected_count)
    positions = [int(round(float(idx) * step)) for idx in range(selected_count)]
    positions = [max(0, min(valid_len - 1, item)) for item in positions]
    if len(set(positions)) == len(positions):
        return sorted(positions)
    out: list[int] = []
    used: set[int] = set()
    for item in positions:
        candidate = int(item)
        while candidate in used and candidate + 1 < valid_len:
            candidate += 1
        while candidate in used and candidate - 1 >= 0:
            candidate -= 1
        if candidate not in used:
            used.add(candidate)
            out.append(candidate)
    return sorted(out)


def _spans_from_binary(values: Sequence[Any]) -> list[tuple[int, int]]:
    arr = [float(item) > 0.5 for item in values]
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(arr):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(arr) - 1))
    return spans


def _boundaries_from_spans(spans: Sequence[tuple[int, int]], length: int) -> list[int]:
    boundaries: set[int] = set()
    for start, end in spans:
        if 0 <= start < length:
            boundaries.add(start)
        if 0 <= end < length:
            boundaries.add(end)
    return sorted(boundaries)


def _resolve_sample(payload: Mapping[str, Any], *, model: str | None, sample_id: str | None) -> Mapping[str, Any]:
    if "models" not in payload:
        if not isinstance(payload, Mapping):
            raise ValueError("sample JSON must contain an object")
        if sample_id is not None and str(payload.get("sample_id") or "") != sample_id:
            raise ValueError(f"sample JSON contains {payload.get('sample_id')}, not requested sample_id={sample_id}")
        return payload

    models = payload.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ValueError("nested sample JSON has no models")
    resolved_model = model if model in models else str(next(iter(models)))
    model_payload = models[resolved_model]
    if not isinstance(model_payload, Mapping):
        raise ValueError(f"model payload must be an object: {resolved_model}")
    samples = model_payload.get("samples")
    if not isinstance(samples, Mapping) or not samples:
        raise ValueError(f"model has no samples: {resolved_model}")
    resolved_sample_id = sample_id if sample_id in samples else str(next(iter(samples)))
    sample = samples[resolved_sample_id]
    if not isinstance(sample, Mapping):
        raise ValueError(f"sample payload must be an object: {resolved_sample_id}")
    return sample


def _mean_nearest_distance(source: Sequence[int], target: Sequence[int]) -> float | None:
    if not source or not target:
        return None
    target_arr = np.asarray(list(target), dtype=float)
    return float(np.mean([np.min(np.abs(target_arr - float(item))) for item in source]))


def _mean_rank_aligned_abs_distance(left: Sequence[int], right: Sequence[int]) -> float | None:
    if not left or not right:
        return None
    count = min(len(left), len(right))
    if count <= 0:
        return None
    left_arr = np.asarray(sorted(left)[:count], dtype=float)
    right_arr = np.asarray(sorted(right)[:count], dtype=float)
    return float(np.mean(np.abs(left_arr - right_arr)))


def _selection_metrics(valid_len: int, selected: Sequence[int], reference: Sequence[int]) -> dict[str, Any]:
    selected_set = set(int(item) for item in selected)
    reference_set = set(int(item) for item in reference)
    if not selected_set and not reference_set:
        jaccard = 1.0
    elif not selected_set or not reference_set:
        jaccard = 0.0
    else:
        jaccard = len(selected_set & reference_set) / len(selected_set | reference_set)
    overlap_count = len(selected_set & reference_set)
    return {
        "valid_len": int(valid_len),
        "count": len(selected_set),
        "reference_count": len(reference_set),
        "overlap_count": overlap_count,
        "overlap_fraction": float(overlap_count / max(1, len(selected_set))),
        "jaccard": float(jaccard),
        "mean_nearest_distance": _mean_nearest_distance(selected, reference),
        "mean_rank_aligned_abs_distance": _mean_rank_aligned_abs_distance(selected, reference),
    }


def _max_unselected_hole(valid_len: int, selected: Sequence[int]) -> int:
    selected_set = set(int(item) for item in selected)
    best = 0
    current = 0
    for idx in range(valid_len):
        if idx in selected_set:
            best = max(best, current)
            current = 0
        else:
            current += 1
    return max(best, current)


def _move_lattice_centers(
    *,
    score: np.ndarray,
    valid_len: int,
    center_budget: int,
    move_ratio: float,
    local_radius: int,
) -> tuple[list[int], dict[str, Any]]:
    base = _uniform_reference_positions(valid_len=valid_len, selected_count=center_budget)
    replaceable_budget = int(round(len(base) * float(move_ratio)))
    candidates: list[tuple[float, int, int, float]] = []
    for center in base:
        lo = max(0, center - local_radius)
        hi = min(valid_len - 1, center + local_radius)
        best = max(range(lo, hi + 1), key=lambda pos: (float(score[pos]), -abs(pos - center), -pos))
        gain = float(score[best] - score[center])
        candidates.append((gain, center, best, abs(float(best - center))))
    candidates.sort(key=lambda item: (item[0], -item[3]), reverse=True)

    selected = list(base)
    used = set(selected)
    replacements: list[dict[str, Any]] = []
    index_by_center = {center: idx for idx, center in enumerate(selected)}
    for gain, old, new, distance in candidates:
        if len(replacements) >= replaceable_budget:
            break
        if gain <= 0 or new == old or new in used:
            continue
        idx = index_by_center.get(old)
        if idx is None:
            continue
        selected[idx] = new
        used.remove(old)
        used.add(new)
        replacements.append({"from": int(old), "to": int(new), "gain": float(gain), "distance": float(distance)})

    diagnostics = {
        "base_center_count": len(base),
        "move_ratio": float(move_ratio),
        "replaceable_budget": replaceable_budget,
        "replaced_count": len(replacements),
        "replacement_distance_max": max((item["distance"] for item in replacements), default=0.0),
        "replacement_distance_mean": float(np.mean([item["distance"] for item in replacements])) if replacements else 0.0,
        "replacements": replacements,
    }
    return sorted(selected), diagnostics


def _adaptive_radius(
    *,
    p_action: np.ndarray,
    p_change: np.ndarray,
    boundary: np.ndarray,
    entropy: np.ndarray,
    centers: Sequence[int],
) -> list[float]:
    radius_float: list[float] = []
    for center in centers:
        uncertainty = 1.0 - abs(float(p_action[center]) - 0.5) * 2.0
        uncertainty = max(0.0, min(1.0, uncertainty))
        value = 1.2 + 10.8 * (
            0.35 * float(entropy[center])
            + 0.25 * float(boundary[center])
            + 0.20 * float(p_change[center])
            + 0.20 * uncertainty
        )
        radius_float.append(float(np.clip(value, 1.0, 12.0)))
    return radius_float


def _expanded_positions(
    *,
    centers: Sequence[int],
    radii: Sequence[float],
    score: np.ndarray,
    valid_len: int,
    expanded_budget: int,
) -> list[int]:
    selected = set(int(item) for item in centers)
    candidate_union = set(selected)
    for center, radius in zip(centers, radii):
        r = int(round(float(radius)))
        for pos in range(max(0, center - r), min(valid_len - 1, center + r) + 1):
            candidate_union.add(pos)
    ranked_candidates = sorted(candidate_union - selected, key=lambda pos: (float(score[pos]), -abs(pos)), reverse=True)
    for pos in ranked_candidates:
        if len(selected) >= expanded_budget:
            break
        selected.add(int(pos))
    if len(selected) < expanded_budget:
        global_ranked = sorted(set(range(valid_len)) - selected, key=lambda pos: float(score[pos]), reverse=True)
        for pos in global_ranked:
            if len(selected) >= expanded_budget:
                break
            selected.add(int(pos))
    return sorted(selected)


def _plot_distribution(
    *,
    output_path: Path,
    sample_id: str,
    valid_len: int,
    action_target: np.ndarray,
    p_action: np.ndarray,
    p_change: np.ndarray,
    boundary: np.ndarray,
    score: np.ndarray,
    centers: Sequence[int],
    expanded: Sequence[int],
    uniform: Sequence[int],
    metrics: Mapping[str, Any],
) -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )
    spans = _spans_from_binary(action_target)
    boundaries = _boundaries_from_spans(spans, valid_len)
    x = np.arange(valid_len)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 8.4),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.35, 1.4], "hspace": 0.12},
    )
    signal_ax, rug_ax, density_ax = axes

    for ax in axes:
        for start, end in spans:
            ax.axvspan(start, end, color="#ef4444", alpha=0.09, linewidth=0)
        for boundary_pos in boundaries:
            ax.axvline(boundary_pos, color="#ef4444", alpha=0.26, linewidth=0.8)

    signal_ax.plot(x, p_action, color="#2563eb", linewidth=1.8, label="动作概率")
    signal_ax.plot(x, p_change, color="#0f766e", linewidth=1.1, label="变化信号")
    signal_ax.plot(x, boundary, color="#7c3aed", linewidth=1.1, label="边界信号")
    signal_ax.plot(x, score, color="#f97316", linewidth=1.1, alpha=0.8, label="move50采样分数")
    signal_ax.scatter(
        list(centers),
        [score[pos] for pos in centers],
        s=14,
        color="#f97316",
        edgecolor="white",
        linewidth=0.25,
        zorder=5,
        label="move50中心",
    )
    signal_ax.set_ylabel("归一化信号")
    signal_ax.set_ylim(-0.04, 1.04)
    signal_ax.grid(axis="y", alpha=0.18)
    signal_ax.legend(frameon=False, loc="upper right", ncol=5)

    rug_rows = [
        (f"均匀{len(uniform)}", uniform, "#64748b", 0.82, 0.85),
        (f"move50中心{len(centers)}", centers, "#2563eb", 0.50, 1.1),
        (f"move50展开{len(expanded)}", expanded, "#f97316", 0.18, 1.0),
    ]
    for label, positions, color, y, width in rug_rows:
        rug_ax.vlines(positions, y - 0.10, y + 0.10, color=color, linewidth=width, alpha=0.88)
        rug_ax.scatter(positions, [y] * len(positions), s=8, color=color, alpha=0.88)
    rug_ax.set_yticks([row[3] for row in rug_rows], [row[0] for row in rug_rows])
    rug_ax.set_ylim(0.02, 0.98)
    rug_ax.grid(axis="x", alpha=0.12)
    rug_ax.spines[["top", "right", "left"]].set_visible(False)
    rug_ax.tick_params(axis="y", length=0)

    bins = np.linspace(0, valid_len, 33)
    uniform_hist, _ = np.histogram(uniform, bins=bins)
    expanded_hist, _ = np.histogram(expanded, bins=bins)
    centers_x = (bins[:-1] + bins[1:]) / 2.0
    width = (bins[1] - bins[0]) * 0.36
    density_ax.bar(centers_x - width / 2, uniform_hist, width=width, color="#94a3b8", alpha=0.68, label=f"均匀{len(uniform)}")
    density_ax.bar(centers_x + width / 2, expanded_hist, width=width, color="#f97316", alpha=0.78, label=f"move50展开{len(expanded)}")
    density_ax.set_ylabel("每段采样数")
    density_ax.set_xlabel("dense-window snippet index")
    density_ax.grid(axis="y", alpha=0.16)
    density_ax.legend(frameon=False, loc="upper right", ncol=2)

    metric_text = (
        f"sample={sample_id}  centers={len(centers)}  expanded={len(expanded)}  "
        f"Jaccard={metrics['expanded_vs_uniform']['jaccard']:.3f}  "
        f"overlap={metrics['expanded_vs_uniform']['overlap_fraction']:.3f}  "
        f"max hole={metrics['expanded_max_hole']}"
    )
    fig.text(0.01, 0.985, "move50样本真采样分布图", ha="left", va="top", fontsize=18, weight="bold")
    fig.text(0.01, 0.952, metric_text, ha="left", va="top", fontsize=12, color="#334155")
    fig.text(
        0.01,
        0.018,
        "说明：move50为本地重解码诊断变体；GT只用于可视化评估，不参与采样分数生成。",
        ha="left",
        va="bottom",
        fontsize=11,
        color="#64748b",
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.91, bottom=0.09)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def build_move50_distribution(
    *,
    sample_json: str | Path,
    output_dir: str | Path,
    prefix: str,
    model: str | None,
    sample_id: str | None,
    center_budget: int,
    expanded_budget: int,
    move_ratio: float,
    local_radius: int,
) -> dict[str, Any]:
    sample_path = Path(sample_json)
    sample = _resolve_sample(json.loads(sample_path.read_text(encoding="utf-8")), model=model, sample_id=sample_id)
    valid_len = int(sample.get("valid_len") or sample.get("dense_len") or len(sample.get("p_action", [])))
    if valid_len <= 0:
        raise ValueError(f"{sample_path}: valid_len must be positive")
    sample_id = str(sample.get("sample_id") or "sample")

    p_action = _scale01(sample.get("p_action") or sample.get("frame_signals", {}).get("p_action") or [], valid_len)
    p_change = _scale01(sample.get("p_change") or sample.get("frame_signals", {}).get("p_change") or [], valid_len)
    boundary = _scale01(sample.get("boundary_score") or sample.get("frame_signals", {}).get("boundary_score") or [], valid_len)
    entropy = _scale01(sample.get("entropy") or sample.get("frame_signals", {}).get("entropy") or [], valid_len)
    action_target = np.asarray(list(sample.get("action_target") or [0] * valid_len)[:valid_len], dtype=float)
    if action_target.size < valid_len:
        action_target = np.pad(action_target, (0, valid_len - action_target.size), mode="constant")

    score = 0.55 * p_action + 0.25 * p_change + 0.20 * boundary
    if score.max() > score.min():
        score = (score - score.min()) / (score.max() - score.min())

    centers, replacement_diag = _move_lattice_centers(
        score=score,
        valid_len=valid_len,
        center_budget=center_budget,
        move_ratio=move_ratio,
        local_radius=local_radius,
    )
    radii = _adaptive_radius(
        p_action=p_action,
        p_change=p_change,
        boundary=boundary,
        entropy=entropy,
        centers=centers,
    )
    expanded = _expanded_positions(
        centers=centers,
        radii=radii,
        score=score,
        valid_len=valid_len,
        expanded_budget=min(expanded_budget, valid_len),
    )
    uniform_centers = _uniform_reference_positions(valid_len=valid_len, selected_count=len(centers))
    uniform_expanded = _uniform_reference_positions(valid_len=valid_len, selected_count=len(expanded))

    metrics = {
        "center_vs_uniform": _selection_metrics(valid_len, centers, uniform_centers),
        "expanded_vs_uniform": _selection_metrics(valid_len, expanded, uniform_expanded),
        "center_max_hole": _max_unselected_hole(valid_len, centers),
        "expanded_max_hole": _max_unselected_hole(valid_len, expanded),
        "radius": {
            "min": float(np.min(radii)) if radii else 0.0,
            "mean": float(np.mean(radii)) if radii else 0.0,
            "p50": float(np.percentile(radii, 50)) if radii else 0.0,
            "p95": float(np.percentile(radii, 95)) if radii else 0.0,
            "max": float(np.max(radii)) if radii else 0.0,
        },
        "replacement": replacement_diag,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figure_path = output_path / f"{prefix}_move50_true_sampling_distribution_{sample_id.replace('|', '_')}.png"
    _plot_distribution(
        output_path=figure_path,
        sample_id=sample_id,
        valid_len=valid_len,
        action_target=action_target,
        p_action=p_action,
        p_change=p_change,
        boundary=boundary,
        score=score,
        centers=centers,
        expanded=expanded,
        uniform=uniform_expanded,
        metrics=metrics,
    )

    manifest = {
        "schema_version": "move50_true_sampling_distribution_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_sample_json": str(sample_path),
        "sample_id": sample_id,
        "valid_len": valid_len,
        "variant": f"paction_lattice_radius_score_only_move{int(round(move_ratio * 100))}",
        "diagnostic_note": "Local re-decode from deploy-visible source signals; not a remote full detector result.",
        "score_formula": "0.55*scaled_p_action + 0.25*scaled_p_change + 0.20*scaled_boundary_score",
        "center_budget": len(centers),
        "expanded_budget": len(expanded),
        "requested_center_budget": int(center_budget),
        "requested_expanded_budget": int(expanded_budget),
        "move_ratio": float(move_ratio),
        "local_radius": int(local_radius),
        "centers": centers,
        "expanded_positions": expanded,
        "uniform_expanded_reference": uniform_expanded,
        "metrics": metrics,
        "figures": [
            {"kind": "move50_true_sampling_distribution", "path": str(figure_path), "bytes": figure_path.stat().st_size},
            {"kind": "move50_true_sampling_distribution_pdf", "path": str(figure_path.with_suffix(".pdf")), "bytes": figure_path.with_suffix(".pdf").stat().st_size},
        ],
    }
    manifest_path = output_path / f"{prefix}_move50_true_sampling_distribution_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw a complete move50 true sampling distribution diagnostic for one dense sample.")
    parser.add_argument("--sample-json", default=str(DEFAULT_SAMPLE_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default="move50")
    parser.add_argument("--model", default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--center-budget", type=int, default=192)
    parser.add_argument("--expanded-budget", type=int, default=384)
    parser.add_argument("--move-ratio", type=float, default=0.50)
    parser.add_argument("--local-radius", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    manifest = build_move50_distribution(
        sample_json=args.sample_json,
        output_dir=args.output_dir,
        prefix=args.prefix,
        model=args.model,
        sample_id=args.sample_id,
        center_budget=args.center_budget,
        expanded_budget=args.expanded_budget,
        move_ratio=args.move_ratio,
        local_radius=args.local_radius,
    )
    print(json.dumps({k: manifest[k] for k in ("sample_id", "variant", "center_budget", "expanded_budget", "figures")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
