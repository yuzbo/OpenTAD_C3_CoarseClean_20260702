#!/usr/bin/env python3
"""Render GeoRoute-AdaTAD paper plots from validated records and analysis JSON.

Nothing is hard-coded as a result.  The caller supplies an output directory;
generated PDFs/PNGs are intentionally not repository artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from georoute_result_schema import (
    GeoRouteResultSchemaError,
    canonical_json_sha256,
    load_records,
    validate_records,
)


COLORS = {
    "dense_native": "#4C78A8",
    "fixed_lattice": "#9C755F",
    "random": "#BAB0AC",
    "free_token_select": "#E45756",
    "roi_only": "#54A24B",
    "roi_residual": "#F58518",
    "tome": "#72B7B2",
    "amod": "#B279A2",
    "roi_residual_amod": "#FF9DA6",
}
MARKERS = {
    "dense_native": "s",
    "fixed_lattice": "o",
    "random": "x",
    "free_token_select": "^",
    "roi_only": "D",
    "roi_residual": "P",
    "tome": "v",
    "amod": "h",
    "roi_residual_amod": "*",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _group_rows(records: Iterable[Mapping[str, Any]]) -> Dict[tuple, List[Mapping[str, Any]]]:
    grouped: Dict[tuple, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["variant"],
            record["budget"]["tokens_per_tubelet"],
            record["stage"],
            record["dataset"],
            record["detector"],
        )
        grouped[key].append(record)
    return grouped


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values)


def _sem(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    variance = sum((float(value) - mean) ** 2 for value in values) / (len(values) - 1)
    return (variance / len(values)) ** 0.5


def _save(fig: plt.Figure, output_dir: Path, name: str, fmt: str) -> Path:
    output = output_dir / f"{name}.{fmt}"
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_pareto(records: Sequence[Mapping[str, Any]], output_dir: Path, fmt: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.3, 3.5))
    for key, rows in _group_rows(records).items():
        variant, budget, _stage, _dataset, _detector = key
        xs = [float(row["cost"]["end_to_end_p50_ms"]) for row in rows]
        ys = [float(row["metrics"]["average_map"]) for row in rows]
        ax.scatter(
            xs,
            ys,
            color=COLORS[variant],
            marker=MARKERS[variant],
            s=48,
            alpha=0.82,
            label=f"{variant} (K={budget})",
        )
        ax.errorbar(
            _mean(xs),
            _mean(ys),
            xerr=_sem(xs),
            yerr=_sem(ys),
            color=COLORS[variant],
            fmt="none",
            capsize=2,
            alpha=0.9,
        )
    ax.set_xlabel("End-to-end latency per window (ms, p50)")
    ax.set_ylabel("Average mAP")
    ax.legend(frameon=False, loc="best")
    return _save(fig, output_dir, "georoute_accuracy_cost_pareto", fmt)


def plot_high_iou(records: Sequence[Mapping[str, Any]], output_dir: Path, fmt: str) -> Path:
    grouped = _group_rows(records)
    labels, values_06, values_07 = [], [], []
    for key, rows in sorted(grouped.items()):
        variant, budget, _stage, _dataset, _detector = key
        labels.append(f"{variant}\nK={budget}")
        values_06.append(_mean([float(row["metrics"]["map_by_tiou"]["0.6"]) for row in rows]))
        values_07.append(_mean([float(row["metrics"]["map_by_tiou"]["0.7"]) for row in rows]))
    fig, ax = plt.subplots(figsize=(max(5.5, 0.78 * len(labels)), 3.5))
    positions = list(range(len(labels)))
    ax.bar([x - 0.19 for x in positions], values_06, width=0.38, label="mAP@0.6", color="#4C78A8")
    ax.bar([x + 0.19 for x in positions], values_07, width=0.38, label="mAP@0.7", color="#F58518")
    ax.set_xticks(positions, labels, rotation=20, ha="right")
    ax.set_ylabel("mAP")
    ax.legend(frameon=False)
    return _save(fig, output_dir, "georoute_high_iou", fmt)


def plot_budget_curve(records: Sequence[Mapping[str, Any]], output_dir: Path, fmt: str) -> Path:
    by_variant: Dict[str, List[tuple]] = defaultdict(list)
    for key, rows in _group_rows(records).items():
        variant, budget, _stage, _dataset, _detector = key
        high_iou = _mean(
            [
                (float(row["metrics"]["map_by_tiou"]["0.6"]) + float(row["metrics"]["map_by_tiou"]["0.7"])) / 2.0
                for row in rows
            ]
        )
        by_variant[variant].append((budget, high_iou))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for variant, points in sorted(by_variant.items()):
        points = sorted(points)
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker=MARKERS[variant],
            color=COLORS[variant],
            label=variant,
        )
    ax.set_xlabel("Exact native tokens per tubelet (K)")
    ax.set_ylabel("Mean mAP@{0.6, 0.7}")
    ax.legend(frameon=False, loc="best")
    return _save(fig, output_dir, "georoute_budget_curve", fmt)


def plot_mechanism_ablation(records: Sequence[Mapping[str, Any]], output_dir: Path, fmt: str) -> Path:
    grouped = _group_rows(records)
    rows = []
    for key, group in sorted(grouped.items()):
        variant, budget, _stage, _dataset, _detector = key
        rows.append(
            (
                f"{variant}\nK={budget}",
                _mean([float(item["metrics"]["average_map"]) for item in group]),
                _mean([float(item["cost"]["gross_gpu_energy_j"]) for item in group]),
                COLORS[variant],
            )
        )
    fig, axes = plt.subplots(1, 2, figsize=(max(7.2, 0.82 * len(rows) * 2), 3.4))
    positions = list(range(len(rows)))
    labels = [row[0] for row in rows]
    colors = [row[3] for row in rows]
    axes[0].bar(positions, [row[1] for row in rows], color=colors)
    axes[0].set_ylabel("Average mAP")
    axes[1].bar(positions, [row[2] for row in rows], color=colors)
    axes[1].set_ylabel("Gross GPU energy per window (J)")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=20, ha="right")
    return _save(fig, output_dir, "georoute_mechanism_ablation", fmt)


def plot_route_diagnostics(
    records: Sequence[Mapping[str, Any]], output_dir: Path, fmt: str
) -> Optional[Path]:
    required = ("roi_area_fraction", "roi_center_velocity", "residual_token_fraction")
    rows = [record for record in records if all(name in record["diagnostics"] for name in required)]
    if not rows:
        return None
    by_variant: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1))
    for axis, name, label in zip(
        axes,
        required,
        ("ROI area / frame", "ROI centre velocity", "Residual allocation fraction"),
    ):
        variants = sorted(by_variant)
        values = [_mean([float(row["diagnostics"][name]) for row in by_variant[variant]]) for variant in variants]
        axis.bar(range(len(variants)), values, color=[COLORS[variant] for variant in variants])
        axis.set_xticks(range(len(variants)), variants, rotation=24, ha="right")
        axis.set_ylabel(label)
    return _save(fig, output_dir, "georoute_route_diagnostics", fmt)


def plot_training_stability(
    records: Sequence[Mapping[str, Any]], output_dir: Path, fmt: str
) -> Optional[Path]:
    rows = [record for record in records if isinstance(record["diagnostics"].get("training_curve"), list)]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for record in rows:
        curve = record["diagnostics"]["training_curve"]
        if not curve:
            continue
        steps = [float(point["step"]) for point in curve]
        losses = [float(point["loss"]) for point in curve]
        label = f"{record['variant']} seed={record['seed']}"
        ax.plot(steps, losses, color=COLORS[record["variant"]], alpha=0.55, label=label)
    ax.set_xlabel("Successful optimizer update")
    ax.set_ylabel("Detector training loss")
    ax.legend(frameon=False, loc="best", ncol=2)
    return _save(fig, output_dir, "georoute_training_stability", fmt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--format", choices=("pdf", "png"), default="pdf")
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    try:
        records = validate_records(
            load_records(args.records), development_only=args.development_only
        )
        analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    except (GeoRouteResultSchemaError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if analysis.get("input_records_sha256") is None:
        parser.error("analysis JSON has no input_records_sha256")
    if analysis["input_records_sha256"] != canonical_json_sha256(records):
        parser.error(
            "analysis JSON does not bind the supplied validated result records; "
            "rerun analyze_georoute_results.py"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    outputs = [
        plot_pareto(records, args.output_dir, args.format),
        plot_high_iou(records, args.output_dir, args.format),
        plot_budget_curve(records, args.output_dir, args.format),
        plot_mechanism_ablation(records, args.output_dir, args.format),
    ]
    optional = (
        plot_route_diagnostics(records, args.output_dir, args.format),
        plot_training_stability(records, args.output_dir, args.format),
    )
    outputs.extend(path for path in optional if path is not None)
    print(json.dumps({"written": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
