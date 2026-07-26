from __future__ import annotations

import csv
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from tools.bata import plot_selector_dashboard
from tools.bata import plot_selector_geometry
from tools.bata import plot_selector_timeline


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_toy_geometry_dir(tmp_path: Path) -> Path:
    geometry_dir = tmp_path / "geometry"
    selected_rows = [
        {
            "video_id": "video_a",
            "method": "fixed384",
            "frame_idx": 0,
            "selected": 1,
            "region": "background",
            "boundary_distance": 4,
            "hole_length": 2,
            "gt_start_frame": 10,
            "gt_end_frame": 22,
        },
        {
            "video_id": "video_a",
            "method": "fixed384",
            "frame_idx": 12,
            "selected": 1,
            "region": "action",
            "boundary_distance": 1,
            "hole_length": 1,
            "gt_start_frame": 10,
            "gt_end_frame": 22,
        },
        {
            "video_id": "video_a",
            "method": "learned",
            "frame_idx": 18,
            "selected": 1,
            "region": "action",
            "boundary_distance": 2,
            "hole_length": 3,
            "gt_start_frame": 10,
            "gt_end_frame": 22,
        },
        {
            "video_id": "video_b",
            "method": "learned",
            "frame_idx": 30,
            "selected": 1,
            "region": "boundary",
            "boundary_distance": 0,
            "hole_length": 1,
            "gt_start_frame": 28,
            "gt_end_frame": 40,
        },
    ]
    _write_csv(geometry_dir / "selected_frame_metrics.csv", selected_rows)
    _write_csv(
        geometry_dir / "method_summary.csv",
        [
            {
                "method": "fixed384",
                "selected_count": 2,
                "endpoint_coverage_r1": 0.50,
                "endpoint_coverage_r2": 0.75,
                "endpoint_coverage_r4": 1.00,
            },
            {
                "method": "learned",
                "selected_count": 2,
                "endpoint_coverage_r1": 0.75,
                "endpoint_coverage_r2": 1.00,
                "endpoint_coverage_r4": 1.00,
            },
        ],
    )
    _write_csv(
        geometry_dir / "action_summary.csv",
        [
            {"method": "fixed384", "region": "background", "selected_count": 1},
            {"method": "fixed384", "region": "action", "selected_count": 1},
            {"method": "learned", "region": "action", "selected_count": 1},
            {"method": "learned", "region": "boundary", "selected_count": 1},
        ],
    )
    _write_csv(
        geometry_dir / "video_summary.csv",
        [
            {"video_id": "video_a", "method": "fixed384", "selected_count": 2, "boundary_support_r1": 0.5},
            {"video_id": "video_a", "method": "learned", "selected_count": 1, "boundary_support_r1": 1.0},
        ],
    )
    _write_csv(
        geometry_dir / "frame_metrics.csv",
        [
            {"video_id": "video_a", "frame_idx": 0, "p_action": 0.10},
            {"video_id": "video_a", "frame_idx": 12, "p_action": 0.90},
            {"video_id": "video_a", "frame_idx": 18, "p_action": 0.80},
            {"video_id": "video_a", "frame_idx": 24, "p_action": 0.20},
        ],
    )
    return geometry_dir


def test_plot_selector_geometry_suite_writes_nonempty_figures_and_manifest(tmp_path: Path) -> None:
    geometry_dir = _make_toy_geometry_dir(tmp_path)
    out_dir = tmp_path / "geometry_plots"

    written = plot_selector_geometry.main(
        ["--geometry-dir", str(geometry_dir), "--methods", "fixed384,learned", "--out-dir", str(out_dir)]
    )

    expected = [
        "boundary_distance_cdf.pdf",
        "region_share_stacked_bar.pdf",
        "endpoint_coverage_by_radius.pdf",
        "holes_by_region_boxplot.pdf",
    ]
    assert written == 0
    for name in expected:
        path = out_dir / name
        assert path.is_file()
        assert path.stat().st_size > 0
    manifest = out_dir / "plot_selector_geometry_manifest.json"
    assert manifest.is_file()
    assert "generated" in manifest.read_text(encoding="utf-8")


def test_plot_selector_timeline_and_dashboard_write_nonempty_figures(tmp_path: Path) -> None:
    geometry_dir = _make_toy_geometry_dir(tmp_path)
    timeline_out = tmp_path / "timeline.png"
    dashboard_out = tmp_path / "dashboard.png"

    assert (
        plot_selector_timeline.main(
            [
                "--geometry-dir",
                str(geometry_dir),
                "--video-id",
                "video_a",
                "--methods",
                "fixed384,learned",
                "--out",
                str(timeline_out),
            ]
        )
        == 0
    )
    assert (
        plot_selector_dashboard.main(
            [
                "--geometry-dir",
                str(geometry_dir),
                "--video-id",
                "video_a",
                "--methods",
                "fixed384,learned",
                "--out",
                str(dashboard_out),
            ]
        )
        == 0
    )
    assert timeline_out.is_file()
    assert timeline_out.stat().st_size > 0
    assert dashboard_out.is_file()
    assert dashboard_out.stat().st_size > 0
