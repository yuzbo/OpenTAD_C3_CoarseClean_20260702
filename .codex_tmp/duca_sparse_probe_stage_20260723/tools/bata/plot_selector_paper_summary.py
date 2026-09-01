from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from tools.bata.plot_selector_geometry import require_matplotlib
except ImportError:
    from plot_selector_geometry import require_matplotlib  # type: ignore


SCHEMA_VERSION = "selector_paper_summary_plots_v1"
READY = "SELECTOR_PAPER_SUMMARY_PLOTS_READY"


METHOD_LABELS = {
    "gas_vt_fixed_384": "GAS-VT",
    "paction_learned_fixed_384": "PAction",
    "lattice_move50": "Lattice-50",
    "lattice_move75": "Lattice-75",
}

LABEL_OFFSETS = {
    "gas_vt_fixed_384": (8, 5),
    "paction_learned_fixed_384": (8, 24),
    "lattice_move50": (8, -18),
    "lattice_move75": (8, -36),
}


METRIC_ALIASES = {
    "average_mAP": ("average_mAP", "avg_map", "mAP", "map"),
    "both_endpoint_r16": (
        "endpoint_both_coverage_r16_mean",
        "endpoint_both_coverage_r16",
        "both_endpoint_coverage_mean",
        "boundary_recall_both",
    ),
    "both_endpoint_r1": (
        "endpoint_both_coverage_r1_mean",
        "endpoint_both_coverage_r1",
        "both_endpoint_coverage_r1",
    ),
    "boundary_r1": ("boundary_recall_r1_mean", "boundary_recall_r1", "boundary_support_r1"),
    "boundary_r16": ("boundary_recall_r16_mean", "boundary_recall_r16", "boundary_support_r16"),
    "p95_hole": ("p95_unselected_hole_mean", "p95_unselected_hole", "p95_hole"),
    "max_hole": ("max_unselected_hole_mean", "max_unselected_hole", "max_hole"),
    "boundary_share": ("boundary_band_selected_ratio_mean", "boundary_region_share", "boundary_share"),
    "action_interior_share": ("action_interior_selected_ratio_mean", "action_region_share", "action_share"),
    "background_share": ("background_selected_ratio_mean", "background_region_share", "background_share"),
    "selected_count": ("selected_count_mean", "selected_count", "n_selected"),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def _first_float(row: Mapping[str, str], metric: str) -> float | None:
    for key in METRIC_ALIASES[metric]:
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _method(row: Mapping[str, str]) -> str:
    return str(row.get("method") or row.get("strategy") or row.get("selector") or "unknown").strip()


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _format_value(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "NA"
    if abs(value) >= 100:
        return f"{value:.0f}{suffix}"
    if abs(value) >= 10:
        return f"{value:.1f}{suffix}"
    return f"{value:.2f}{suffix}"


def _load_rows(analysis_root: Path) -> list[dict[str, str]]:
    table_rows = _read_csv(analysis_root / "tables" / "table1_map_vs_geometry.csv")
    if table_rows:
        return table_rows
    return _read_csv(analysis_root / "geometry" / "method_summary.csv")


def _filter_rows(rows: Sequence[dict[str, str]], methods: Sequence[str] | None) -> list[dict[str, str]]:
    if not methods:
        return list(rows)
    wanted = set(methods)
    return [row for row in rows if _method(row) in wanted]


def _save(fig: Any, out_dir: Path, stem: str, formats: Sequence[str]) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        path = out_dir / f"{stem}.{fmt}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if fmt != "pdf":
            kwargs["dpi"] = 220
        else:
            kwargs["metadata"] = {"Creator": "OpenTAD selector paper summary plots", "CreationDate": None, "ModDate": None}
        fig.savefig(path, **kwargs)
        written.append(path.name)
    return written


def _plot_gap_boundary_quadrant(plt: Any, rows: Sequence[dict[str, str]], out_dir: Path, formats: Sequence[str]) -> list[str]:
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    points: list[tuple[str, float, float, float | None, float]] = []
    for row in rows:
        method = _method(row)
        hole = _first_float(row, "p95_hole")
        coverage = _first_float(row, "both_endpoint_r16") or _first_float(row, "boundary_r16")
        if hole is None or coverage is None:
            continue
        map_value = _first_float(row, "average_mAP")
        marker_size = 70.0 if map_value is None else 45.0 + max(0.0, map_value)
        points.append((method, max(hole, 1e-3), coverage, map_value, marker_size))
    if not points:
        raise ValueError("no rows contain both p95 hole and endpoint/boundary coverage metrics")

    for method, hole, coverage, map_value, marker_size in points:
        ax.scatter(hole, coverage, s=marker_size, alpha=0.82)
    y_values = [point[2] for point in points]
    y_min, y_max = min(y_values), max(y_values)
    y_margin = max(0.02, (y_max - y_min) * 0.35)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    for method, hole, coverage, map_value, _ in points:
        label = _label(method)
        if map_value is not None:
            label += f"\n{map_value:.1f} mAP"
        ax.annotate(
            label,
            (hole, coverage),
            xytext=LABEL_OFFSETS.get(method, (8, 8)),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.72},
        )
    ax.set_xscale("log")
    ax.set_xlabel("p95 unselected hole (lower is better, log scale)")
    ax.set_ylabel("Both-endpoint coverage @ r16")
    ax.grid(True, which="both", alpha=0.22)
    ax.axvline(10.0, color="0.55", linestyle="--", linewidth=1.0)
    ax.text(10.6, ax.get_ylim()[0] + 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "hole=10", fontsize=8, color="0.35")
    fig.tight_layout()
    written = _save(fig, out_dir, "paper_gap_boundary_quadrant", formats)
    plt.close(fig)
    return written


def _plot_delta_vs_baseline(
    plt: Any,
    rows: Sequence[dict[str, str]],
    out_dir: Path,
    formats: Sequence[str],
    baseline_method: str,
) -> list[str]:
    baseline = next((row for row in rows if _method(row) == baseline_method), None)
    if baseline is None:
        raise ValueError(f"baseline method {baseline_method!r} is missing")
    base_map = _first_float(baseline, "average_mAP")
    base_cov = _first_float(baseline, "both_endpoint_r16") or _first_float(baseline, "boundary_r16")
    base_hole = _first_float(baseline, "p95_hole")
    compared = [row for row in rows if _method(row) != baseline_method]
    if not compared:
        raise ValueError("need at least one non-baseline method")

    labels = [_label(_method(row)) for row in compared]
    map_delta = [None if base_map is None or _first_float(row, "average_mAP") is None else _first_float(row, "average_mAP") - base_map for row in compared]
    cov_delta = [None if base_cov is None or (_first_float(row, "both_endpoint_r16") or _first_float(row, "boundary_r16")) is None else ((_first_float(row, "both_endpoint_r16") or _first_float(row, "boundary_r16")) - base_cov) * 100.0 for row in compared]
    hole_reduction = [None if base_hole is None or _first_float(row, "p95_hole") in (None, 0.0) else base_hole / max(_first_float(row, "p95_hole") or 1e-12, 1e-12) for row in compared]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharex=False)
    panels = [
        (axes[0], "Delta mAP", map_delta, "mAP points"),
        (axes[1], "Delta endpoint coverage", cov_delta, "percentage points"),
        (axes[2], "p95 hole reduction", hole_reduction, "x vs GAS-VT"),
    ]
    for ax, title, values, ylabel in panels:
        numeric = [0.0 if value is None else value for value in values]
        bars = ax.bar(labels, numeric, color=["#4C78A8" if value is not None else "#D0D0D0" for value in values])
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.set_title(title, fontsize=9, pad=8)
        ax.tick_params(axis="x", rotation=20)
        valid_values = [value for value in numeric if math.isfinite(value)]
        if valid_values:
            top = max(valid_values)
            bottom = min(0.0, min(valid_values))
            span = max(1.0, top - bottom)
            ax.set_ylim(bottom - 0.05 * span, top + 0.20 * span)
        for bar, value in zip(bars, values):
            text = "NA" if value is None else _format_value(value)
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            offset = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0] or 1.0)
            ax.text(bar.get_x() + bar.get_width() / 2.0, y + (offset if y >= 0 else -offset), text, ha="center", va=va, fontsize=8)
    fig.tight_layout()
    written = _save(fig, out_dir, "paper_delta_vs_gasvt", formats)
    plt.close(fig)
    return written


def _score(value: float | None, lower: float, upper: float, higher_is_better: bool = True) -> float:
    if value is None or upper <= lower:
        return 0.0
    clipped = min(max(value, lower), upper)
    norm = (clipped - lower) / (upper - lower)
    return norm if higher_is_better else 1.0 - norm


def _plot_scorecard(plt: Any, rows: Sequence[dict[str, str]], out_dir: Path, formats: Sequence[str]) -> list[str]:
    metrics = [
        ("average_mAP", "mAP", True),
        ("both_endpoint_r16", "Both endpoint\n@r16", True),
        ("boundary_r1", "Boundary\n@r1", True),
        ("p95_hole", "p95 hole\n(low good)", False),
        ("background_share", "Background\nshare", False),
    ]
    values_by_metric: dict[str, list[float]] = {}
    for metric, _, _ in metrics:
        vals = [_first_float(row, metric) for row in rows]
        values_by_metric[metric] = [value for value in vals if value is not None]
    matrix: list[list[float]] = []
    labels: list[str] = []
    annotations: list[list[str]] = []
    for row in rows:
        labels.append(_label(_method(row)))
        row_scores: list[float] = []
        row_ann: list[str] = []
        for metric, _, higher in metrics:
            value = _first_float(row, metric)
            vals = values_by_metric.get(metric, [])
            if vals:
                lower, upper = min(vals), max(vals)
            else:
                lower, upper = 0.0, 1.0
            row_scores.append(_score(value, lower, upper, higher_is_better=higher))
            row_ann.append(_format_value(value))
        matrix.append(row_scores)
        annotations.append(row_ann)
    if not matrix:
        raise ValueError("no method rows available for scorecard")

    fig, ax = plt.subplots(figsize=(7.0, max(2.5, 0.6 * len(labels) + 1.6)))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="YlGnBu", aspect="auto")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xticks(range(len(metrics)), labels=[label for _, label, _ in metrics])
    ax.tick_params(axis="x", rotation=0)
    for row_idx, row_ann in enumerate(annotations):
        for col_idx, text in enumerate(row_ann):
            color = "white" if matrix[row_idx][col_idx] > 0.62 else "black"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("within-metric score")
    fig.tight_layout()
    written = _save(fig, out_dir, "paper_selector_scorecard", formats)
    plt.close(fig)
    return written


def plot_paper_summary(
    analysis_root: Path,
    out_dir: Path,
    methods: Sequence[str] | None = None,
    baseline_method: str = "gas_vt_fixed_384",
    formats: Sequence[str] = ("pdf", "png"),
) -> dict[str, Any]:
    plt = require_matplotlib()
    rows = _filter_rows(_load_rows(analysis_root), methods)
    if not rows:
        raise ValueError(f"no method rows found under {analysis_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    skipped: dict[str, str] = {}
    plotters = {
        "paper_gap_boundary_quadrant": lambda: _plot_gap_boundary_quadrant(plt, rows, out_dir, formats),
        "paper_delta_vs_gasvt": lambda: _plot_delta_vs_baseline(plt, rows, out_dir, formats, baseline_method),
        "paper_selector_scorecard": lambda: _plot_scorecard(plt, rows, out_dir, formats),
    }
    for name, plotter in plotters.items():
        try:
            generated.extend(plotter())
        except ValueError as exc:
            skipped[name] = str(exc)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "decision": READY,
        "analysis_root": str(analysis_root),
        "baseline_method": baseline_method,
        "formats": list(formats),
        "methods": [_method(row) for row in rows],
        "generated": generated,
        "skipped": skipped,
    }
    _write_json(out_dir / "plot_selector_paper_summary_manifest.json", manifest)
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate direct paper-facing selector geometry summary plots.")
    parser.add_argument("--analysis-root", required=True, type=Path, help="Root containing geometry/ and tables/ outputs.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--methods", nargs="*", default=None, help="Comma-separated or space-separated method names.")
    parser.add_argument("--baseline-method", default="gas_vt_fixed_384")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], help="Output formats, e.g. pdf png.")
    return parser


def _normalize_methods(methods: Sequence[str] | None) -> list[str] | None:
    if not methods:
        return None
    parsed: list[str] = []
    for item in methods:
        parsed.extend(part.strip() for part in str(item).split(","))
    parsed = [item for item in parsed if item]
    return parsed or None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    plot_paper_summary(
        analysis_root=args.analysis_root,
        out_dir=args.out_dir,
        methods=_normalize_methods(args.methods),
        baseline_method=args.baseline_method,
        formats=args.formats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
