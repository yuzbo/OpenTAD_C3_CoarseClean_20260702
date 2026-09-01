from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


DISTANCE_COLUMNS = (
    "boundary_distance",
    "boundary_distance_frames",
    "nearest_boundary_distance",
    "nearest_boundary_distance_frames",
    "distance_to_boundary",
    "distance_to_nearest_boundary",
    "min_boundary_distance",
)
FRAME_COLUMNS = ("frame_idx", "frame_index", "frame", "dense_frame_idx", "source_frame_idx", "t")
METHOD_COLUMNS = ("method", "strategy", "selector", "selector_method")
REGION_COLUMNS = ("region", "action_region", "gt_region", "frame_region")
HOLE_COLUMNS = ("hole_length", "hole_size", "gap_length", "unselected_hole", "nearest_hole")
COUNT_COLUMNS = ("selected_count", "count", "n_selected", "num_selected", "frames", "num_frames")
SHARE_COLUMNS = ("selected_share", "share", "fraction", "ratio", "region_share")
VIDEO_COLUMNS = ("video_id", "video", "video_name")


def require_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for selector geometry plots. Install it with `pip install matplotlib` "
            "or run in an environment that already provides matplotlib."
        ) from exc
    return plt


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_methods(methods: Sequence[str] | str | None) -> list[str] | None:
    if methods is None:
        return None
    if isinstance(methods, str):
        raw_items = [methods]
    else:
        raw_items = list(methods)
    parsed: list[str] = []
    for item in raw_items:
        parsed.extend(part.strip() for part in str(item).split(","))
    parsed = [item for item in parsed if item]
    return parsed or None


def _columns(rows: Sequence[dict[str, str]]) -> dict[str, str]:
    if not rows:
        return {}
    out: dict[str, str] = {}
    for key in rows[0].keys():
        out[key.lower()] = key
    return out


def find_column(rows: Sequence[dict[str, str]], candidates: Iterable[str]) -> str | None:
    available = _columns(rows)
    for candidate in candidates:
        key = available.get(candidate.lower())
        if key is not None:
            return key
    return None


def find_matching_columns(rows: Sequence[dict[str, str]], patterns: Sequence[str]) -> list[str]:
    if not rows:
        return []
    regexes = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    return [key for key in rows[0].keys() if all(regex.search(key) for regex in regexes)]


def as_float(value: object) -> float | None:
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


def row_method(row: dict[str, str], default: str = "default") -> str:
    for column in METHOD_COLUMNS:
        if column in row and str(row[column]).strip():
            return str(row[column]).strip()
    return default


def filter_methods(rows: Sequence[dict[str, str]], methods: Sequence[str] | None) -> list[dict[str, str]]:
    if not methods:
        return list(rows)
    wanted = set(methods)
    return [row for row in rows if row_method(row) in wanted]


def save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    metadata = {"Creator": "OpenTAD selector geometry plotting", "CreationDate": None, "ModDate": None}
    if suffix == ".pdf":
        fig.savefig(path, bbox_inches="tight", metadata=metadata)
    else:
        fig.savefig(path, bbox_inches="tight", dpi=160)


def _plot_boundary_distance_cdf(
    plt: Any, selected_rows: Sequence[dict[str, str]], methods: Sequence[str] | None, out_path: Path
) -> str | None:
    rows = filter_methods(selected_rows, methods)
    distance_col = find_column(rows, DISTANCE_COLUMNS)
    if distance_col is None:
        return "selected_frame_metrics.csv is missing a boundary distance column"
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = as_float(row.get(distance_col))
        if value is not None:
            grouped[row_method(row)].append(value)
    grouped = {key: sorted(values) for key, values in grouped.items() if values}
    if not grouped:
        return f"{distance_col} has no numeric values"

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for method in sorted(grouped):
        values = grouped[method]
        y = [(idx + 1) / len(values) for idx in range(len(values))]
        ax.step(values, y, where="post", label=method)
    ax.set_xlabel(distance_col)
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    return None


def _region_counts_from_action_summary(
    action_rows: Sequence[dict[str, str]], methods: Sequence[str] | None
) -> dict[str, dict[str, float]]:
    rows = filter_methods(action_rows, methods)
    region_col = find_column(rows, REGION_COLUMNS)
    if region_col is None:
        return {}
    share_col = find_column(rows, SHARE_COLUMNS)
    count_col = find_column(rows, COUNT_COLUMNS)
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        region = str(row.get(region_col, "")).strip()
        if not region:
            continue
        value = as_float(row.get(share_col)) if share_col else None
        if value is None:
            value = as_float(row.get(count_col)) if count_col else None
        if value is not None:
            grouped[row_method(row)][region] += value
    return {method: dict(values) for method, values in grouped.items()}


def _region_counts_from_selected(selected_rows: Sequence[dict[str, str]], methods: Sequence[str] | None) -> dict[str, dict[str, float]]:
    rows = filter_methods(selected_rows, methods)
    region_col = find_column(rows, REGION_COLUMNS)
    if region_col is None:
        return {}
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        region = str(row.get(region_col, "")).strip()
        if region:
            grouped[row_method(row)][region] += 1.0
    return {method: dict(values) for method, values in grouped.items()}


def _plot_region_share_stacked_bar(
    plt: Any,
    selected_rows: Sequence[dict[str, str]],
    action_rows: Sequence[dict[str, str]],
    methods: Sequence[str] | None,
    out_path: Path,
) -> str | None:
    grouped = _region_counts_from_action_summary(action_rows, methods) or _region_counts_from_selected(selected_rows, methods)
    grouped = {method: values for method, values in grouped.items() if values}
    if not grouped:
        return "no region/count or region/share fields were found"
    regions = sorted({region for values in grouped.values() for region in values})
    method_names = sorted(grouped)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    bottoms = [0.0 for _ in method_names]
    for region in regions:
        shares: list[float] = []
        for idx, method in enumerate(method_names):
            total = sum(grouped[method].values())
            value = grouped[method].get(region, 0.0)
            share = value / total if total > 1.000001 else value
            shares.append(share)
            bottoms[idx] += 0.0
        ax.bar(method_names, shares, bottom=bottoms, label=region)
        bottoms = [bottom + share for bottom, share in zip(bottoms, shares)]
    ax.set_ylabel("Selected share")
    ax.set_ylim(0.0, max(1.0, max(bottoms) * 1.05))
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    return None


def _radius_from_column(name: str) -> float | None:
    lower = name.lower()
    match = re.search(r"(?:radius|r|@)[_-]?(\d+(?:\.\d+)?)", lower)
    if match is None:
        return None
    return float(match.group(1))


def _plot_endpoint_coverage_by_radius(
    plt: Any, method_rows: Sequence[dict[str, str]], methods: Sequence[str] | None, out_path: Path
) -> str | None:
    rows = filter_methods(method_rows, methods)
    coverage_cols = [
        col
        for col in find_matching_columns(rows, (r"(endpoint|boundary)", r"(coverage|support)"))
        if _radius_from_column(col) is not None
    ]
    if not rows or not coverage_cols:
        return "method_summary.csv is missing endpoint/boundary coverage-by-radius columns"
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        method = row_method(row)
        for col in coverage_cols:
            radius = _radius_from_column(col)
            value = as_float(row.get(col))
            if radius is not None and value is not None:
                grouped[method].append((radius, value))
    grouped = {method: sorted(values) for method, values in grouped.items() if values}
    if not grouped:
        return "coverage-by-radius columns have no numeric values"

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for method in sorted(grouped):
        by_radius: dict[float, list[float]] = defaultdict(list)
        for radius, value in grouped[method]:
            by_radius[radius].append(value)
        radii = sorted(by_radius)
        values = [sum(by_radius[radius]) / len(by_radius[radius]) for radius in radii]
        ax.plot(radii, values, marker="o", label=method)
    ax.set_xlabel("Boundary radius")
    ax.set_ylabel("Endpoint coverage")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    return None


def _plot_holes_by_region_boxplot(
    plt: Any, selected_rows: Sequence[dict[str, str]], methods: Sequence[str] | None, out_path: Path
) -> str | None:
    rows = filter_methods(selected_rows, methods)
    region_col = find_column(rows, REGION_COLUMNS)
    hole_col = find_column(rows, HOLE_COLUMNS)
    if region_col is None or hole_col is None:
        return "selected_frame_metrics.csv is missing region or hole-length columns"
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        region = str(row.get(region_col, "")).strip()
        value = as_float(row.get(hole_col))
        if region and value is not None:
            grouped[f"{row_method(row)}\n{region}"].append(value)
    labels = sorted(key for key, values in grouped.items() if values)
    if not labels:
        return f"{hole_col} has no numeric values"

    fig, ax = plt.subplots(figsize=(max(6.4, len(labels) * 0.85), 4.2))
    ax.boxplot([grouped[label] for label in labels], labels=labels, showfliers=False)
    ax.set_ylabel(hole_col)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    return None


def plot_geometry_suite(geometry_dir: Path, out_dir: Path, methods: Sequence[str] | None = None) -> dict[str, Any]:
    plt = require_matplotlib()
    selected_rows = read_csv_rows(geometry_dir / "selected_frame_metrics.csv")
    method_rows = read_csv_rows(geometry_dir / "method_summary.csv")
    action_rows = read_csv_rows(geometry_dir / "action_summary.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    method_filter = list(methods) if methods else None

    plots = {
        "boundary_distance_cdf.pdf": lambda path: _plot_boundary_distance_cdf(
            plt, selected_rows, method_filter, path
        ),
        "region_share_stacked_bar.pdf": lambda path: _plot_region_share_stacked_bar(
            plt, selected_rows, action_rows, method_filter, path
        ),
        "endpoint_coverage_by_radius.pdf": lambda path: _plot_endpoint_coverage_by_radius(
            plt, method_rows, method_filter, path
        ),
        "holes_by_region_boxplot.pdf": lambda path: _plot_holes_by_region_boxplot(
            plt, selected_rows, method_filter, path
        ),
    }

    generated: list[str] = []
    skipped: dict[str, str] = {}
    for filename, plotter in plots.items():
        reason = plotter(out_dir / filename)
        if reason is None:
            generated.append(filename)
        else:
            skipped[filename] = reason

    manifest = {
        "schema_version": "selector_geometry_plot_manifest_v1",
        "geometry_dir": str(geometry_dir),
        "methods": method_filter,
        "generated": generated,
        "skipped": skipped,
        "inputs": {
            "selected_frame_metrics.csv": bool(selected_rows),
            "method_summary.csv": bool(method_rows),
            "action_summary.csv": bool(action_rows),
        },
    }
    write_json(out_dir / "plot_selector_geometry_manifest.json", manifest)
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot selector geometry analysis figures from analyzer CSV outputs.")
    parser.add_argument("--geometry-dir", required=True, type=Path)
    parser.add_argument("--methods", nargs="*", default=None, help="Comma-separated or space-separated method names.")
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    plot_geometry_suite(args.geometry_dir, args.out_dir, methods=normalize_methods(args.methods))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
