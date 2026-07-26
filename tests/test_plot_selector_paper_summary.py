from __future__ import annotations

import csv
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from tools.bata import plot_selector_paper_summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_plot_selector_paper_summary_writes_direct_comparison_figures(tmp_path: Path) -> None:
    analysis_root = tmp_path / "analysis"
    _write_csv(
        analysis_root / "tables" / "table1_map_vs_geometry.csv",
        [
            {
                "method": "gas_vt_fixed_384",
                "average_mAP": 44.9,
                "endpoint_both_coverage_r16_mean": 0.17,
                "endpoint_both_coverage_r1_mean": 0.12,
                "boundary_recall_r1_mean": 0.23,
                "p95_unselected_hole_mean": 92.5,
                "background_selected_ratio_mean": 0.58,
            },
            {
                "method": "paction_learned_fixed_384",
                "average_mAP": 59.1,
                "endpoint_both_coverage_r16_mean": 0.26,
                "endpoint_both_coverage_r1_mean": 0.25,
                "boundary_recall_r1_mean": 0.42,
                "p95_unselected_hole_mean": 2.2,
                "background_selected_ratio_mean": 0.59,
            },
            {
                "method": "lattice_move50",
                "average_mAP": "",
                "endpoint_both_coverage_r16_mean": 0.27,
                "endpoint_both_coverage_r1_mean": 0.26,
                "boundary_recall_r1_mean": 0.43,
                "p95_unselected_hole_mean": 2.0,
                "background_selected_ratio_mean": 0.61,
            },
        ],
    )
    out_dir = tmp_path / "paper_figures"

    manifest = plot_selector_paper_summary.plot_paper_summary(
        analysis_root=analysis_root,
        out_dir=out_dir,
        formats=("pdf",),
    )

    expected = {
        "paper_gap_boundary_quadrant.pdf",
        "paper_delta_vs_gasvt.pdf",
        "paper_selector_scorecard.pdf",
    }
    assert set(manifest["generated"]) == expected
    assert manifest["skipped"] == {}
    for name in expected:
        path = out_dir / name
        assert path.is_file()
        assert path.stat().st_size > 0
