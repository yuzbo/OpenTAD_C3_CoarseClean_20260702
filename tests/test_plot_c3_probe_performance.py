from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _assert_png(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 1000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_renders_coarse_classifier_and_indirect_selection_figures(tmp_path: Path) -> None:
    from tools.bata.plot_c3_probe_performance import render_performance_figures

    candidate_csv = tmp_path / "candidate_summary.csv"
    _write_csv(
        candidate_csv,
        [
            {
                "model": "temporal_tcn_lite",
                "ap": 0.43,
                "auc": 0.65,
                "best_f1": 0.52,
                "action_bg_gap": 0.072,
                "change_lift": 0.013,
                "top10_r4": 0.62,
                "top20_r4": 0.81,
                "candidate_score": 0.59,
            },
            {
                "model": "temporal_tcn_motion",
                "ap": 0.44,
                "auc": 0.66,
                "best_f1": 0.51,
                "action_bg_gap": 0.090,
                "change_lift": 0.012,
                "top10_r4": 0.60,
                "top20_r4": 0.80,
                "candidate_score": 0.58,
            },
        ],
    )

    selection_csv = tmp_path / "selection_stats.csv"
    _write_csv(
        selection_csv,
        [
            {
                "model": "temporal_tcn_lite",
                "strategy": "delta_p_action",
                "probe_ap": 0.43,
                "probe_roc_auc": 0.65,
                "boundary_support_r1_global": 0.21,
                "action_coverage_global": 0.52,
                "p95_window_max_empty_gap": 67.3,
                "mean_valid_len": 732.0,
                "max_empty_gap_global": 184,
                "mean_selected": 366.1,
                "windows": 515,
            },
            {
                "model": "temporal_tcn_lite",
                "strategy": "topk_action_logit",
                "probe_ap": 0.43,
                "probe_roc_auc": 0.65,
                "boundary_support_r1_global": 0.16,
                "action_coverage_global": 0.61,
                "p95_window_max_empty_gap": 210.0,
                "mean_valid_len": 732.0,
                "max_empty_gap_global": 355,
                "mean_selected": 366.1,
                "windows": 515,
            },
            {
                "model": "temporal_tcn_motion",
                "strategy": "delta_p_action",
                "probe_ap": 0.44,
                "probe_roc_auc": 0.66,
                "boundary_support_r1_global": 0.22,
                "action_coverage_global": 0.50,
                "p95_window_max_empty_gap": 70.6,
                "mean_valid_len": 732.0,
                "max_empty_gap_global": 179,
                "mean_selected": 366.1,
                "windows": 515,
            },
            {
                "model": "temporal_tcn_motion",
                "strategy": "topk_action_logit",
                "probe_ap": 0.44,
                "probe_roc_auc": 0.66,
                "boundary_support_r1_global": 0.15,
                "action_coverage_global": 0.59,
                "p95_window_max_empty_gap": 205.0,
                "mean_valid_len": 732.0,
                "max_empty_gap_global": 340,
                "mean_selected": 366.1,
                "windows": 515,
            },
        ],
    )

    sample_json = tmp_path / "samples.json"
    sample_json.write_text(
        json.dumps(
            {
                "models": {
                    "temporal_tcn_lite": {
                        "samples": {
                            "video_test_0001|0": {
                                "valid_len": 8,
                                "p_action": [0.05, 0.20, 0.75, 0.82, 0.30, 0.10, 0.65, 0.72],
                                "p_change": [0.0, 0.15, 0.55, 0.07, 0.52, 0.20, 0.55, 0.07],
                                "boundary_score": [0.0, 0.25, 0.90, 0.30, 0.85, 0.35, 0.70, 0.20],
                                "segments": [[2, 4], [6, 7]],
                                "boundaries": [2, 4, 6, 7],
                                "selected_delta_p_action": [0, 3, 6],
                                "boundary_support_r1": 1.0,
                                "action_coverage": 0.75,
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "figures"
    manifest = render_performance_figures(
        candidate_csv=candidate_csv,
        selection_csv=selection_csv,
        sample_json=sample_json,
        output_dir=output_dir,
        prefix="fixture",
    )

    assert manifest["figure_count"] == 5
    assert manifest["source_files"]["candidate_csv"] == str(candidate_csv)
    assert manifest["source_files"]["selection_csv"] == str(selection_csv)
    comparison = manifest["uniform_delta_comparison"]
    assert comparison["valid_len"] == 8
    assert comparison["selected_count"] == 3
    assert comparison["delta_label"] == "delta_p_action_3"
    assert comparison["uniform_label"] == "uniform_3"
    assert comparison["jaccard"] == pytest.approx(0.5)
    assert comparison["overlap_fraction"] == pytest.approx(2 / 3)
    assert comparison["mean_delta_to_uniform_nearest_distance"] == pytest.approx(1 / 3)
    assert comparison["mean_uniform_to_delta_nearest_distance"] == pytest.approx(1 / 3)
    assert comparison["mean_rank_aligned_abs_distance"] == pytest.approx(1 / 3)
    assert {figure["kind"] for figure in manifest["figures"]} == {
        "coarse_classifier_metrics",
        "indirect_selection",
        "strategy_tradeoff",
        "sample_timeline",
        "uniform_vs_delta_rug",
    }
    for figure in manifest["figures"]:
        _assert_png(Path(figure["path"]))
    manifest_path = output_dir / "fixture_manifest.json"
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved_manifest["figure_count"] == 5
    assert saved_manifest["uniform_delta_comparison"]["jaccard"] == pytest.approx(0.5)


def test_renders_same_window_selection_rug_from_source_sample_and_ledger(tmp_path: Path) -> None:
    from tools.bata.plot_c3_probe_performance import render_same_window_selection_rug

    sample_json = tmp_path / "source_sample.json"
    sample_json.write_text(
        json.dumps(
            {
                "sample_id": "video_test_0001|0",
                "valid_len": 8,
                "p_action": [0.05, 0.20, 0.75, 0.82, 0.30, 0.10, 0.65, 0.72],
                "frame_signals": {
                    "p_change": [0.0, 0.15, 0.55, 0.07, 0.52, 0.20, 0.55, 0.07],
                    "boundary_score": [0.0, 0.25, 0.90, 0.30, 0.85, 0.35, 0.70, 0.20],
                },
                "action_target": [0, 0, 1, 1, 0, 0, 1, 0],
            }
        ),
        encoding="utf-8",
    )
    ledger_jsonl = tmp_path / "test.ledger.jsonl"
    ledger_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "video_test_0001|0",
                "target_len": 8,
                "expanded_selected_positions": [0, 3, 6],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "figures"
    manifest = render_same_window_selection_rug(
        sample_json=sample_json,
        ledger_jsonl=ledger_jsonl,
        ledger_selection_key="expanded_selected_positions",
        selection_key="lattice_expanded_positions",
        selection_label="lattice_expanded_3",
        model="budgeted_adaptive_radius_lattice",
        output_dir=output_dir,
        prefix="lattice",
    )

    assert manifest["figure_count"] == 1
    assert manifest["selection_label"] == "lattice_expanded_3"
    comparison = manifest["uniform_selection_comparison"]
    assert comparison["selected_count"] == 3
    assert comparison["uniform_label"] == "uniform_3"
    assert comparison["jaccard"] == pytest.approx(0.5)
    assert comparison["overlap_fraction"] == pytest.approx(2 / 3)
    assert comparison["mean_selection_to_uniform_nearest_distance"] == pytest.approx(1 / 3)
    assert comparison["mean_uniform_to_selection_nearest_distance"] == pytest.approx(1 / 3)
    assert comparison["mean_rank_aligned_abs_distance"] == pytest.approx(1 / 3)
    assert manifest["figures"][0]["kind"] == "same_window_selection_rug"
    _assert_png(Path(manifest["figures"][0]["path"]))
