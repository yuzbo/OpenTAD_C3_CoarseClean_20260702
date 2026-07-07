from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

try:
    from tools.bata.plot_selector_geometry import (
        FRAME_COLUMNS,
        METHOD_COLUMNS,
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
        METHOD_COLUMNS,
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


PACTION_COLUMNS = ("p_action", "action_probability", "prob_action", "score", "selector_score")
GT_START_COLUMNS = ("gt_start_frame", "segment_start_frame", "action_start_frame", "gt_start", "segment_start")
GT_END_COLUMNS = ("gt_end_frame", "segment_end_frame", "action_end_frame", "gt_end", "segment_end")
BOUNDARY_RADIUS_COLUMNS = ("boundary_radius", "boundary_band_radius", "endpoint_radius")


def _video_rows(rows: Sequence[dict[str, str]], video_id: str) -> list[dict[str, str]]:
    video_col = find_column(rows, VIDEO_COLUMNS)
    if video_col is None:
        return []
    return [row for row in rows if str(row.get(video_col, "")).strip() == video_id]


def _numeric_frame(row: dict[str, str], frame_col: str | None) -> float | None:
    return as_float(row.get(frame_col)) if frame_col else None


def _collect_segments(rows: Sequence[dict[str, str]]) -> list[tuple[float, float]]:
    start_col = find_column(rows, GT_START_COLUMNS)
    end_col = find_column(rows, GT_END_COLUMNS)
    if start_col is None or end_col is None:
        return []
    segments = set()
    for row in rows:
        start = as_float(row.get(start_col))
        end = as_float(row.get(end_col))
        if start is None or end is None or end < start:
            continue
        segments.add((start, end))
    return sorted(segments)


def plot_timeline(
    geometry_dir: Path,
    out_path: Path,
    video_id: str,
    methods: Sequence[str] | None = None,
) -> None:
    plt = require_matplotlib()
    selected_rows = filter_methods(_video_rows(read_csv_rows(geometry_dir / "selected_frame_metrics.csv"), video_id), methods)
    frame_rows = _video_rows(read_csv_rows(geometry_dir / "frame_metrics.csv"), video_id)
    frame_col = find_column(selected_rows, FRAME_COLUMNS) or find_column(frame_rows, FRAME_COLUMNS)
    paction_col = find_column(frame_rows, PACTION_COLUMNS)
    if not selected_rows and not frame_rows:
        raise ValueError(f"no rows found for video_id={video_id!r}")

    method_names = list(methods or sorted({row_method(row) for row in selected_rows}) or ["default"])
    fig, ax = plt.subplots(figsize=(10.0, 3.8))

    segments = _collect_segments(selected_rows) or _collect_segments(frame_rows)
    radius_col = find_column(selected_rows, BOUNDARY_RADIUS_COLUMNS) or find_column(frame_rows, BOUNDARY_RADIUS_COLUMNS)
    radius = 1.0
    if radius_col is not None:
        numeric_radii = [as_float(row.get(radius_col)) for row in [*selected_rows, *frame_rows]]
        numeric_radii = [value for value in numeric_radii if value is not None]
        if numeric_radii:
            radius = max(numeric_radii)
    for start, end in segments:
        ax.axvspan(start, end, color="#d7eadf", alpha=0.45, linewidth=0)
        for boundary in (start, end):
            ax.axvspan(boundary - radius, boundary + radius, color="#f4b6b2", alpha=0.35, linewidth=0)

    y_by_method = {method: idx + 1 for idx, method in enumerate(method_names)}
    grouped_frames: dict[str, list[float]] = defaultdict(list)
    for row in selected_rows:
        frame = _numeric_frame(row, frame_col)
        if frame is not None:
            grouped_frames[row_method(row)].append(frame)
    for method in method_names:
        frames = grouped_frames.get(method, [])
        if not frames:
            continue
        y = y_by_method[method]
        ax.vlines(frames, y - 0.34, y + 0.34, linewidth=1.2, label=f"{method} selected")

    if frame_rows and frame_col is not None and paction_col is not None:
        method_col = find_column(frame_rows, METHOD_COLUMNS)
        if method_col is None:
            points = sorted(
                (frame, value)
                for row in frame_rows
                if (frame := _numeric_frame(row, frame_col)) is not None
                and (value := as_float(row.get(paction_col))) is not None
            )
            if points:
                xs, ys = zip(*points)
                ax.plot(xs, [float(y) * 0.8 for y in ys], color="#333333", linewidth=1.5, label=paction_col)
        else:
            for method in method_names:
                points = sorted(
                    (frame, value)
                    for row in frame_rows
                    if str(row.get(method_col, "")).strip() == method
                    and (frame := _numeric_frame(row, frame_col)) is not None
                    and (value := as_float(row.get(paction_col))) is not None
                )
                if points:
                    xs, ys = zip(*points)
                    ax.plot(xs, [float(y) * 0.8 for y in ys], linewidth=1.5, label=f"{method} {paction_col}")

    ax.set_title(video_id)
    ax.set_xlabel(frame_col or "frame")
    ax.set_yticks([y_by_method[method] for method in method_names])
    ax.set_yticklabels(method_names)
    ax.set_ylim(0.0, max(y_by_method.values()) + 0.8)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot one selector timeline from analyzer CSV outputs.")
    parser.add_argument("--geometry-dir", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--methods", nargs="*", default=None, help="Comma-separated or space-separated method names.")
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    plot_timeline(
        geometry_dir=args.geometry_dir,
        out_path=args.out,
        video_id=args.video_id,
        methods=normalize_methods(args.methods),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
