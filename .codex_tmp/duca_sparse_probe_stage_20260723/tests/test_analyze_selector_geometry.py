from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.bata import analyze_selector_geometry as geom


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_selector_geometry_analysis_outputs_boundary_region_and_endpoint_tables(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    uniform = tmp_path / "uniform.jsonl"
    paction = tmp_path / "paction.jsonl"
    out_dir = tmp_path / "geometry"
    sample_id = "video_validation_0001|0"
    _write_jsonl(
        samples,
        [
            {
                "sample_id": sample_id,
                "split": "validation",
                "valid_len": 12,
                "dense_len": 12,
                "gt_segments": [[3, 8]],
                "frame_signals": {"p_action": [0.05, 0.1, 0.2, 0.9, 0.6, 0.4, 0.3, 0.85, 0.2, 0.1, 0.05, 0.02]},
            }
        ],
    )
    _write_jsonl(
        uniform,
        [
            {
                "sample_id": sample_id,
                "valid_len": 12,
                "dense_len": 12,
                "selected_count": 4,
                "selected_positions": [0, 3, 6, 9],
            }
        ],
    )
    _write_jsonl(
        paction,
        [
            {
                "sample_id": sample_id,
                "valid_len": 12,
                "dense_len": 12,
                "selected_count": 4,
                "selected_positions": [3, 5, 7, 8],
            }
        ],
    )

    manifest = geom.analyze_geometry(
        selector_ledgers={"uniform_384": uniform, "paction_learned_fixed_384": paction},
        sample_jsonls={"*": samples},
        out_dir=out_dir,
        run_tag="toy_geometry",
        split="validation",
        boundary_band_radius=1,
        radii_frames=[1, 2],
    )

    assert manifest["decision"] == "C3_SELECTOR_GEOMETRY_ANALYSIS_READY"
    assert manifest["coordinate_contract"]["gt_segment_convention"] == "half_open_[start,end)"
    selected_rows = _read_csv(out_dir / "selected_frame_metrics.csv")
    assert {row["method"] for row in selected_rows} == {"uniform_384", "paction_learned_fixed_384"}
    paction_rows = [row for row in selected_rows if row["method"] == "paction_learned_fixed_384"]
    assert [row["region"] for row in paction_rows].count("boundary_band") == 3
    assert any(row["nearest_boundary_distance_frame"] == "0" for row in paction_rows)

    video_rows = _read_csv(out_dir / "video_summary.csv")
    paction_video = next(row for row in video_rows if row["method"] == "paction_learned_fixed_384")
    assert paction_video["selected_count"] == "4"
    assert float(paction_video["boundary_band_selected_ratio"]) == 0.75
    assert float(paction_video["boundary_recall_r1"]) == 1.0

    action_rows = _read_csv(out_dir / "action_summary.csv")
    paction_r1 = next(row for row in action_rows if row["method"] == "paction_learned_fixed_384" and row["radius_frame"] == "1")
    assert paction_r1["start_hit"] == "1"
    assert paction_r1["end_hit"] == "1"
    assert paction_r1["both_endpoint_hit"] == "1"

    holes = _read_csv(out_dir / "holes_by_region.csv")
    assert {row["region"] for row in holes} >= {"whole_video", "boundary_band", "action_interior", "background"}
    calibration = _read_csv(out_dir / "paction_calibration.csv")
    assert calibration
    method_summary = _read_csv(out_dir / "method_summary.csv")
    assert {row["method"] for row in method_summary} == {"uniform_384", "paction_learned_fixed_384"}


def test_selector_geometry_cli_accepts_named_ledgers_and_common_samples(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    out_dir = tmp_path / "geometry_cli"
    _write_jsonl(samples, [{"sample_id": "v|0", "valid_len": 5, "gt_segments": [[1, 4]], "p_action": [0.1, 0.8, 0.2, 0.7, 0.1]}])
    _write_jsonl(ledger, [{"sample_id": "v|0", "valid_len": 5, "selected_count": 2, "selected_positions": [1, 3]}])

    assert geom.main(
        [
            "--selector-ledger",
            f"method_a={ledger}",
            "--common-sample-jsonl",
            str(samples),
            "--run-tag",
            "cli",
            "--out-dir",
            str(out_dir),
        ]
    ) == 0
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "method_summary.csv").exists()
