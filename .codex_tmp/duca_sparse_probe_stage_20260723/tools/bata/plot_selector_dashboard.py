from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

try:
    from tools.bata.plot_selector_geometry import (
        FRAME_COLUMNS,
        VIDEO_COLUMNS,
        as_float,
        filter_methods,
        find_column,
        normalize_methods,
        read_csv_rows,
        require_matplotlib,
        row_method,
        save_figure,
    )
except ImportError:
    from plot_selector_geometry import (  # type: ignore
        FRAME_COLUMNS,
        VIDEO_COLUMNS,
        as_float,
        filter_methods,
        find_column,
        normalize_methods,
        read_csv_rows,
        require_matplotlib,
        row_method,
        save_figure,
    )


PREFERRED_METRIC_COLUMNS = (
    "selected_count",
    "endpoint_coverage_r1",
    "endpoint_coverage_r2",
    "endpoint_coverage_r4",
    "boundary_support_r1",
    "boundary_support_r2",
    "action_coverage",
    "mean_boundary_distance",
    "max_hole",
    "p95_hole",
)


def _video_rows(rows: Sequence[dict[str, str]], video_id: str) -> list[dict[str, str]]:
    video_col = find_column(rows, VIDEO_COLUMNS)
    if video_col is None:
        return []
    return [row for row in rows if str(row.get(video_col, "")).strip() == video_id]


def _metric_text(row: dict[str, str] | None) -> str:
    if row is None:
        return "video_summary.csv row missing"
    items: list[str] = []
    for column in PREFERRED_METRIC_COLUMNS:
        if column in row and str(row[column]).strip():
            items.append(f"{column}: {row[column]}")
    if not items:
        for key, value in row.items():
            if key.lower() in {"video_id", "video", "video_name", "method", "strategy", "selector", "selector_method"}:
                continue
            if as_float(value) is not None:
                items.append(f"{key}: {value}")
            if len(items) >= 6:
                break
    return "\n".join(items[:6]) if items else "no numeric summary metrics"


def plot_dashboard(
    geometry_dir: Path,
    out_path: Path,
    video_id: str,
    methods: Sequence[str] | None = None,
) -> None:
    plt = require_matplotlib()
    selected_rows = filter_methods(_video_rows(read_csv_rows(geometry_dir / "selected_frame_metrics.csv"), video_id), methods)
    summary_rows = filter_methods(_video_rows(read_csv_rows(geometry_dir / "video_summary.csv"), video_id), methods)
    method_names = list(methods or sorted({row_method(row) for row in [*summary_rows, *selected_rows]}))
    if not method_names:
        raise ValueError(f"no rows found for video_id={video_id!r}")

    frame_col = find_column(selected_rows, FRAME_COLUMNS)
    summary_by_method = {row_method(row): row for row in summary_rows}
    frames_by_method: dict[str, list[float]] = {method: [] for method in method_names}
    for row in selected_rows:
        frame = as_float(row.get(frame_col)) if frame_col else None
        if frame is not None:
            frames_by_method.setdefault(row_method(row), []).append(frame)
    all_frames = [frame for frames in frames_by_method.values() for frame in frames]
    xmin = min(all_frames) if all_frames else 0.0
    xmax = max(all_frames) if all_frames else 1.0
    if xmax <= xmin:
        xmax = xmin + 1.0

    height = max(2.8, 1.25 * len(method_names))
    fig, axes = plt.subplots(len(method_names), 1, figsize=(10.0, height), sharex=True)
    if len(method_names) == 1:
        axes = [axes]

    for ax, method in zip(axes, method_names):
        frames = sorted(frames_by_method.get(method, []))
        if frames:
            ax.eventplot(frames, orientation="horizontal", lineoffsets=0.0, linelengths=0.7, linewidths=1.4)
        ax.set_yticks([])
        ax.set_ylabel(method, rotation=0, ha="right", va="center", labelpad=52)
        ax.set_xlim(xmin - 0.03 * (xmax - xmin), xmax + 0.03 * (xmax - xmin))
        ax.grid(True, axis="x", alpha=0.22)
        ax.text(
            1.005,
            0.5,
            _metric_text(summary_by_method.get(method)),
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=8,
            family="monospace",
        )

    axes[-1].set_xlabel(frame_col or "frame")
    fig.suptitle(video_id, y=0.98)
    fig.subplots_adjust(right=0.72, hspace=0.36)
    save_figure(fig, out_path)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a single-video selector geometry dashboard.")
    parser.add_argument("--geometry-dir", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--methods", nargs="*", default=None, help="Comma-separated or space-separated method names.")
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    plot_dashboard(
        geometry_dir=args.geometry_dir,
        out_path=args.out,
        video_id=args.video_id,
        methods=normalize_methods(args.methods),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
