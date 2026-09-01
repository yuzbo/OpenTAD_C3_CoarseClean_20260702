from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bata import plot_move50_true_sampling_distribution as move_sampler


DEFAULT_LEDGER_DIR = Path("analysis_outputs") / "lattice_budgeted_radius_fb7419c_20260708_150121"
DEFAULT_SAMPLE_JSON = (
    Path("analysis_outputs")
    / "c3_completed_coarse_selection_oracle_style_20260704"
    / "completed_selection_oracle_style_data.json"
)
DEFAULT_SAMPLE_MODEL = "temporal_tcn_lite"
DEFAULT_SAMPLE_ID = "video_test_0000006|0"
DEFAULT_OUTPUT_DIR = Path("analysis_outputs") / "move25_move50_training_performance_curves_20260709"


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def _uniform_positions(valid_len: int, count: int) -> list[int]:
    return move_sampler._uniform_reference_positions(valid_len=int(valid_len), selected_count=int(count))


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(int(item) for item in left)
    right_set = set(int(item) for item in right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return float(len(left_set & right_set) / len(left_set | right_set))


def _overlap_fraction(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(int(item) for item in left)
    if not left_set:
        return 0.0
    return float(len(left_set & set(int(item) for item in right)) / len(left_set))


def _mean_nearest_distance(selected: Sequence[int], reference: Sequence[int]) -> float:
    if not selected or not reference:
        return 0.0
    ref = np.asarray(reference, dtype=float)
    return float(np.mean([np.min(np.abs(ref - float(pos))) for pos in selected]))


def _max_hole(valid_len: int, selected: Sequence[int]) -> int:
    return int(move_sampler._max_unselected_hole(int(valid_len), selected))


def _ecdf(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x = np.asarray(sorted(float(item) for item in values), dtype=float)
    y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
    return x, y


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _ledger_rows_to_metrics(rows: Sequence[Mapping[str, Any]], *, split: str) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for row in rows:
        valid_len = int(row.get("valid_len") or row.get("dense_len") or 0)
        centers = [int(item) for item in row.get("selected_positions") or []]
        expanded = [int(item) for item in row.get("expanded_selected_positions") or centers]
        center_ref = _uniform_positions(valid_len, len(centers))
        expanded_ref = _uniform_positions(valid_len, len(expanded))
        radius_float = [float(item) for item in row.get("context_radius_float_by_position") or []]
        metrics.append(
            {
                "split": split,
                "sample_id": str(row.get("sample_id") or ""),
                "valid_len": valid_len,
                "center_count": len(centers),
                "expanded_count": len(expanded),
                "center_fraction": float(len(centers) / max(1, valid_len)),
                "expanded_fraction": float(len(expanded) / max(1, valid_len)),
                "center_jaccard_uniform": _jaccard(centers, center_ref),
                "expanded_jaccard_uniform": _jaccard(expanded, expanded_ref),
                "expanded_overlap_uniform": _overlap_fraction(expanded, expanded_ref),
                "expanded_nearest_uniform": _mean_nearest_distance(expanded, expanded_ref),
                "center_max_hole": _max_hole(valid_len, centers),
                "expanded_max_hole": _max_hole(valid_len, expanded),
                "radius_mean": _safe_mean(radius_float),
                "radius_p95": _safe_percentile(radius_float, 95),
                "policy": str(row.get("policy") or ""),
                "route_variant": str(row.get("route_variant") or ""),
            }
        )
    return metrics


def _read_split_summary_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            converted: dict[str, Any] = {}
            for key, value in row.items():
                if key == "split":
                    converted[key] = value
                else:
                    try:
                        converted[key] = float(value)
                    except (TypeError, ValueError):
                        converted[key] = value
            rows.append(converted)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _plot_move25_split_distribution(
    *,
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    split_summary_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    _set_style()
    colors = {"train": "#2563eb", "val": "#f97316", "test": "#059669"}
    labels = {"train": "训练集", "val": "验证集", "test": "测试集"}

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0))
    ax_jaccard, ax_hole, ax_radius, ax_summary = axes.flat

    summary: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        color = colors.get(split, "#475569")
        label = labels.get(split, split)
        jaccard = [float(row["expanded_jaccard_uniform"]) for row in rows]
        holes = [float(row["expanded_max_hole"]) for row in rows]
        radius = [float(row["radius_mean"]) for row in rows]
        overlap = [float(row["expanded_overlap_uniform"]) for row in rows]

        x, y = _ecdf(jaccard)
        ax_jaccard.plot(x, y, color=color, linewidth=2.0, label=f"{label} n={len(rows)}")
        x_hole, y_hole = _ecdf(holes)
        ax_hole.step(x_hole, y_hole, where="post", color=color, linewidth=2.0, label=label)
        x_radius, y_radius = _ecdf(radius)
        ax_radius.plot(x_radius, y_radius, color=color, linewidth=2.0, label=label)

        summary[split] = {
            "rows": len(rows),
            "expanded_jaccard_uniform_mean": _safe_mean(jaccard),
            "expanded_jaccard_uniform_p50": _safe_percentile(jaccard, 50),
            "expanded_overlap_uniform_mean": _safe_mean(overlap),
            "expanded_max_hole_p95": _safe_percentile(holes, 95),
            "radius_mean": _safe_mean(radius),
        }

    ax_jaccard.set_xlabel("与均匀384的 Jaccard")
    ax_jaccard.set_ylabel("累计比例")
    ax_jaccard.grid(axis="both", alpha=0.18)
    ax_jaccard.legend(frameon=False, loc="lower right")

    ax_hole.set_xlabel("最大未采样空洞")
    ax_hole.set_ylabel("累计比例")
    ax_hole.grid(axis="both", alpha=0.18)
    ax_hole.legend(frameon=False, loc="lower right")

    ax_radius.set_xlabel("平均上下文半径")
    ax_radius.set_ylabel("累计比例")
    ax_radius.grid(axis="both", alpha=0.18)
    ax_radius.legend(frameon=False, loc="lower right")

    if split_summary_rows:
        split_order = [row["split"] for row in split_summary_rows]
        x = np.arange(len(split_order), dtype=float)
        width = 0.17
        metrics = [
            ("expanded_uniform_jaccard_mean", "采样形状差异"),
            ("expanded_uniform_overlap_mean", "均匀重合比例"),
            ("boundary_support_r4", "边界覆盖 r4"),
            ("action_positive_coverage", "动作覆盖"),
        ]
        palette = ["#2563eb", "#f97316", "#7c3aed", "#059669"]
        for offset, (key, label) in enumerate(metrics):
            values = [float(row.get(key, 0.0)) for row in split_summary_rows]
            ax_summary.bar(x + (offset - 1.5) * width, values, width=width, color=palette[offset], label=label)
        ax_summary.set_xticks(x, [labels.get(item, item) for item in split_order])
        ax_summary.set_ylim(0.0, 0.55)
        ax_summary.set_ylabel("均值")
        ax_summary.grid(axis="y", alpha=0.18)
        ax_summary.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    else:
        ax_summary.axis("off")

    fig.text(0.01, 0.985, "move25 全量 ledger：训练/验证/测试性能分布", fontsize=16, weight="bold", va="top")
    fig.text(
        0.01,
        0.952,
        "读取本地 train/val/test ledger；指标反映采样形状、空洞、上下文半径和诊断覆盖，不代表最终 TAD mAP。",
        fontsize=11,
        color="#475569",
        va="top",
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.08, hspace=0.36, wspace=0.22)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    return summary


def _spans_from_binary(values: Sequence[float]) -> list[tuple[int, int]]:
    return move_sampler._spans_from_binary(values)


def _boundary_positions(spans: Sequence[tuple[int, int]], valid_len: int) -> list[int]:
    return move_sampler._boundaries_from_spans(spans, valid_len)


def _coverage_metrics(
    *,
    valid_len: int,
    action_target: Sequence[float],
    selected: Sequence[int],
    boundary_radius: int = 4,
) -> dict[str, float]:
    selected_set = set(int(item) for item in selected)
    action_positions = {idx for idx, value in enumerate(action_target[:valid_len]) if float(value) > 0.5}
    spans = _spans_from_binary(action_target[:valid_len])
    boundaries = _boundary_positions(spans, valid_len)
    if action_positions:
        action_coverage = len(action_positions & selected_set) / float(len(action_positions))
    else:
        action_coverage = 0.0
    if boundaries:
        covered = 0
        for boundary in boundaries:
            if any(abs(int(pos) - int(boundary)) <= boundary_radius for pos in selected_set):
                covered += 1
        boundary_coverage = covered / float(len(boundaries))
    else:
        boundary_coverage = 0.0
    touched_spans = 0
    for start, end in spans:
        if any(start <= int(pos) <= end for pos in selected_set):
            touched_spans += 1
    segment_touch = touched_spans / float(len(spans)) if spans else 0.0
    return {
        "action_coverage": float(action_coverage),
        "boundary_r4": float(boundary_coverage),
        "segment_touch": float(segment_touch),
    }


def _build_sample_variant(
    *,
    sample: Mapping[str, Any],
    source_sample_json: Path,
    move_ratio: float,
    center_budget: int,
    expanded_budget: int,
    local_radius: int,
) -> dict[str, Any]:
    valid_len = int(sample.get("valid_len") or sample.get("dense_len") or len(sample.get("p_action", [])))
    p_action = move_sampler._scale01(sample.get("p_action") or sample.get("frame_signals", {}).get("p_action") or [], valid_len)
    p_change = move_sampler._scale01(sample.get("p_change") or sample.get("frame_signals", {}).get("p_change") or [], valid_len)
    boundary = move_sampler._scale01(sample.get("boundary_score") or sample.get("frame_signals", {}).get("boundary_score") or [], valid_len)
    entropy = move_sampler._scale01(sample.get("entropy") or sample.get("frame_signals", {}).get("entropy") or [], valid_len)
    action_target = np.asarray(list(sample.get("action_target") or [0] * valid_len)[:valid_len], dtype=float)
    if action_target.size < valid_len:
        action_target = np.pad(action_target, (0, valid_len - action_target.size), mode="constant")

    score = 0.55 * p_action + 0.25 * p_change + 0.20 * boundary
    if float(score.max()) > float(score.min()):
        score = (score - score.min()) / (score.max() - score.min())

    centers, replacement = move_sampler._move_lattice_centers(
        score=score,
        valid_len=valid_len,
        center_budget=min(center_budget, valid_len),
        move_ratio=move_ratio,
        local_radius=local_radius,
    )
    radii = move_sampler._adaptive_radius(
        p_action=p_action,
        p_change=p_change,
        boundary=boundary,
        entropy=entropy,
        centers=centers,
    )
    expanded = move_sampler._expanded_positions(
        centers=centers,
        radii=radii,
        score=score,
        valid_len=valid_len,
        expanded_budget=min(expanded_budget, valid_len),
    )
    uniform_expanded = _uniform_positions(valid_len, len(expanded))
    coverage = _coverage_metrics(valid_len=valid_len, action_target=action_target.tolist(), selected=expanded)
    metrics = {
        "move_ratio": float(move_ratio),
        "valid_len": valid_len,
        "center_count": len(centers),
        "expanded_count": len(expanded),
        "expanded_jaccard_uniform": _jaccard(expanded, uniform_expanded),
        "expanded_overlap_uniform": _overlap_fraction(expanded, uniform_expanded),
        "expanded_nearest_uniform": _mean_nearest_distance(expanded, uniform_expanded),
        "expanded_max_hole": _max_hole(valid_len, expanded),
        "radius_mean": _safe_mean([float(item) for item in radii]),
        "radius_p95": _safe_percentile([float(item) for item in radii], 95),
        "replaced_count": int(replacement.get("replaced_count", 0)),
        "replaceable_budget": int(replacement.get("replaceable_budget", 0)),
        **coverage,
    }
    return {
        "source_sample_json": str(source_sample_json),
        "sample_id": str(sample.get("sample_id") or "sample"),
        "valid_len": valid_len,
        "p_action": p_action,
        "p_change": p_change,
        "boundary": boundary,
        "score": score,
        "action_target": action_target,
        "centers": centers,
        "expanded": expanded,
        "uniform_expanded": uniform_expanded,
        "metrics": metrics,
    }


def _plot_same_window_move25_move50(
    *,
    variants: Mapping[str, Mapping[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    _set_style()
    first = next(iter(variants.values()))
    valid_len = int(first["valid_len"])
    sample_id = str(first["sample_id"])
    x = np.arange(valid_len)
    action_target = np.asarray(first["action_target"], dtype=float)
    spans = _spans_from_binary(action_target.tolist())
    boundaries = _boundary_positions(spans, valid_len)

    fig = plt.figure(figsize=(14.0, 9.2))
    grid = fig.add_gridspec(
        4,
        1,
        height_ratios=[2.8, 1.2, 1.45, 1.55],
        hspace=0.15,
    )
    ax_signal = fig.add_subplot(grid[0, 0])
    ax_rug = fig.add_subplot(grid[1, 0], sharex=ax_signal)
    ax_hist = fig.add_subplot(grid[2, 0], sharex=ax_signal)
    ax_metric = fig.add_subplot(grid[3, 0])
    axes = [ax_signal, ax_rug, ax_hist, ax_metric]

    for ax in axes[:3]:
        for start, end in spans:
            ax.axvspan(start, end, color="#ef4444", alpha=0.08, linewidth=0)
        for boundary in boundaries:
            ax.axvline(boundary, color="#ef4444", alpha=0.24, linewidth=0.8)

    ax_signal.plot(x, first["p_action"], color="#2563eb", linewidth=1.9, label="动作支持分数")
    ax_signal.plot(x, first["score"], color="#111827", linewidth=1.2, alpha=0.78, label="综合采样分数")
    ax_signal.plot(x, first["p_change"], color="#0f766e", linewidth=1.0, alpha=0.75, label="变化信号")
    for name, color in [("move25", "#f97316"), ("move50", "#7c3aed")]:
        variant = variants[name]
        centers = list(variant["centers"])
        ax_signal.scatter(
            centers,
            [float(variant["score"][pos]) for pos in centers],
            s=11,
            color=color,
            alpha=0.78,
            label=f"{name}中心",
            zorder=4,
        )
    ax_signal.set_ylabel("归一化信号")
    ax_signal.set_ylim(-0.04, 1.04)
    ax_signal.grid(axis="y", alpha=0.18)
    ax_signal.legend(frameon=False, ncol=5, loc="upper right")

    rows = [
        ("均匀384", list(first["uniform_expanded"]), "#94a3b8", 0.86),
        ("move25中心", list(variants["move25"]["centers"]), "#fb923c", 0.64),
        ("move25展开", list(variants["move25"]["expanded"]), "#f97316", 0.42),
        ("move50中心", list(variants["move50"]["centers"]), "#a78bfa", 0.22),
        ("move50展开", list(variants["move50"]["expanded"]), "#7c3aed", 0.06),
    ]
    for label, positions, color, y_pos in rows:
        ax_rug.vlines(positions, y_pos - 0.045, y_pos + 0.045, color=color, linewidth=0.95, alpha=0.9)
    ax_rug.set_yticks([row[3] for row in rows], [row[0] for row in rows])
    ax_rug.set_ylim(-0.02, 0.94)
    ax_rug.spines[["left", "top", "right"]].set_visible(False)
    ax_rug.tick_params(axis="y", length=0)
    ax_rug.grid(axis="x", alpha=0.10)

    bins = np.linspace(0, valid_len, 33)
    centers_x = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.25
    uniform_hist, _ = np.histogram(first["uniform_expanded"], bins=bins)
    move25_hist, _ = np.histogram(variants["move25"]["expanded"], bins=bins)
    move50_hist, _ = np.histogram(variants["move50"]["expanded"], bins=bins)
    ax_hist.bar(centers_x - width, uniform_hist, width=width, color="#94a3b8", alpha=0.62, label="均匀384")
    ax_hist.bar(centers_x, move25_hist, width=width, color="#f97316", alpha=0.72, label="move25展开")
    ax_hist.bar(centers_x + width, move50_hist, width=width, color="#7c3aed", alpha=0.70, label="move50展开")
    ax_hist.set_ylabel("每段采样数")
    ax_hist.set_xlabel("dense-window snippet index")
    ax_hist.grid(axis="y", alpha=0.16)
    ax_hist.legend(frameon=False, ncol=3, loc="upper right")

    metric_keys = [
        ("expanded_jaccard_uniform", "形状差异"),
        ("expanded_overlap_uniform", "均匀重合"),
        ("action_coverage", "动作覆盖"),
        ("boundary_r4", "边界覆盖"),
    ]
    x_metric = np.arange(len(metric_keys), dtype=float)
    bar_width = 0.34
    for idx, (name, color) in enumerate([("move25", "#f97316"), ("move50", "#7c3aed")]):
        values = [float(variants[name]["metrics"][key]) for key, _ in metric_keys]
        ax_metric.bar(x_metric + (idx - 0.5) * bar_width, values, width=bar_width, color=color, alpha=0.82, label=name)
        for xpos, value in zip(x_metric + (idx - 0.5) * bar_width, values):
            ax_metric.text(xpos, value + 0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax_metric.set_xticks(x_metric, [label for _, label in metric_keys])
    ax_metric.set_ylim(0.0, 1.12)
    ax_metric.set_ylabel("比例")
    ax_metric.grid(axis="y", alpha=0.16)

    m25 = variants["move25"]["metrics"]
    m50 = variants["move50"]["metrics"]
    note = (
        f"sample={sample_id}  K={m25['expanded_count']}  "
        f"move25: max hole={m25['expanded_max_hole']}, replaced={m25['replaced_count']}  "
        f"move50: max hole={m50['expanded_max_hole']}, replaced={m50['replaced_count']}"
    )
    fig.text(0.01, 0.985, "同一窗口诊断：move25 与 move50 采样分布曲线", fontsize=16, weight="bold", va="top")
    fig.text(0.01, 0.955, note, fontsize=11, color="#475569", va="top")
    fig.text(
        0.01,
        0.018,
        "红色浅区为真值动作区间，仅用于可视化评估；采样分数由 deploy-visible 信号重解码得到，不使用真值生成。",
        fontsize=10.5,
        color="#64748b",
        va="bottom",
    )
    fig.subplots_adjust(left=0.095, right=0.99, top=0.90, bottom=0.09)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    return {name: dict(variant["metrics"]) for name, variant in variants.items()}


def build_curves(
    *,
    ledger_dir: Path,
    sample_json: Path,
    sample_model: str | None,
    sample_id: str | None,
    output_dir: Path,
    center_budget: int,
    expanded_budget: int,
    local_radius: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        ledger_path = ledger_dir / f"{split}.ledger.jsonl"
        rows = _read_jsonl(ledger_path)
        metrics = _ledger_rows_to_metrics(rows, split=split)
        split_rows[split] = metrics
        all_rows.extend(metrics)

    split_summary_csv = ledger_dir / "lattice_selection_distribution_summary.csv"
    split_summary_rows = _read_split_summary_csv(split_summary_csv)

    row_metrics_csv = output_dir / "move25_ledger_row_metrics.csv"
    _write_csv(row_metrics_csv, all_rows)

    split_distribution_path = output_dir / "move25_full_split_distribution_curves.png"
    move25_split_summary = _plot_move25_split_distribution(
        rows_by_split=split_rows,
        split_summary_rows=split_summary_rows,
        output_path=split_distribution_path,
    )

    sample = move_sampler._resolve_sample(_read_json(sample_json), model=sample_model, sample_id=sample_id)
    variants = {
        "move25": _build_sample_variant(
            sample=sample,
            source_sample_json=sample_json,
            move_ratio=0.25,
            center_budget=center_budget,
            expanded_budget=expanded_budget,
            local_radius=local_radius,
        ),
        "move50": _build_sample_variant(
            sample=sample,
            source_sample_json=sample_json,
            move_ratio=0.50,
            center_budget=center_budget,
            expanded_budget=expanded_budget,
            local_radius=local_radius,
        ),
    }
    same_window_path = output_dir / "move25_move50_same_window_distribution_curves.png"
    same_window_summary = _plot_same_window_move25_move50(variants=variants, output_path=same_window_path)

    same_window_metrics = []
    for name, variant in variants.items():
        row = {"variant": name, "sample_id": variant["sample_id"], **variant["metrics"]}
        same_window_metrics.append(row)
    same_window_csv = output_dir / "move25_move50_same_window_metrics.csv"
    _write_csv(same_window_csv, same_window_metrics)

    manifest = {
        "schema_version": "move25_move50_training_performance_curves_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ledger_dir": str(ledger_dir),
        "source_sample_json": str(sample_json),
        "source_sample_model": sample_model,
        "source_sample_id": sample_id,
        "evidence_scope": {
            "move25": "full local train/val/test ledger plus same-window diagnostic",
            "move50": "same-window diagnostic re-decode from local source sample; no full local train/val/test ledger was found",
            "training_history": "no local epoch-level move25/move50 training history was found; loss-vs-epoch curves were not fabricated",
        },
        "outputs": {
            "move25_split_distribution_png": str(split_distribution_path),
            "move25_split_distribution_pdf": str(split_distribution_path.with_suffix(".pdf")),
            "same_window_png": str(same_window_path),
            "same_window_pdf": str(same_window_path.with_suffix(".pdf")),
            "move25_row_metrics_csv": str(row_metrics_csv),
            "same_window_metrics_csv": str(same_window_csv),
        },
        "move25_split_summary": move25_split_summary,
        "same_window_summary": same_window_summary,
    }
    manifest_path = output_dir / "move25_move50_training_performance_curves_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot move25/move50 training and performance distribution diagnostics.")
    parser.add_argument("--ledger-dir", default=str(DEFAULT_LEDGER_DIR))
    parser.add_argument("--sample-json", default=str(DEFAULT_SAMPLE_JSON))
    parser.add_argument("--sample-model", default=DEFAULT_SAMPLE_MODEL)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--center-budget", type=int, default=192)
    parser.add_argument("--expanded-budget", type=int, default=384)
    parser.add_argument("--local-radius", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    manifest = build_curves(
        ledger_dir=Path(args.ledger_dir),
        sample_json=Path(args.sample_json),
        sample_model=args.sample_model,
        sample_id=args.sample_id,
        output_dir=Path(args.output_dir),
        center_budget=int(args.center_budget),
        expanded_budget=int(args.expanded_budget),
        local_radius=int(args.local_radius),
    )
    print(json.dumps(manifest["outputs"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
