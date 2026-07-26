from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.bata import export_selector_paper_tables as paper_tables
from tools.bata import generate_selector_failure_gallery as failure_gallery
from tools.bata import validate_selector_geometry_metrics as validator


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _toy_geometry_dir(tmp_path: Path) -> Path:
    geometry_dir = tmp_path / "geometry"
    geometry_dir.mkdir()
    (geometry_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "selector_geometry_v1",
                "coordinate_contract": {
                    "selected_positions_unit": "local_dense_index",
                    "valid_range": "[0, valid_len)",
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        geometry_dir / "method_summary.csv",
        [
            {
                "method": "uniform384",
                "selected_count_min": 384,
                "selected_count_max": 384,
                "selected_count_mean": 384,
                "boundary_recall_both": 0.80,
                "p95_gap": 3,
                "action_region_share": 0.40,
                "background_region_share": 0.60,
                "stage": "stage2",
            },
            {
                "method": "gasvt384",
                "selected_count_min": 384,
                "selected_count_max": 384,
                "selected_count_mean": 384,
                "boundary_recall_both": 0.65,
                "p95_gap": 9,
                "action_region_share": 0.55,
                "background_region_share": 0.45,
                "stage": "stage3",
            },
        ],
    )
    _write_csv(
        geometry_dir / "video_summary.csv",
        [
            {"method": "uniform384", "video_id": "v1", "p95_gap": 3, "boundary_recall_both": 0.80},
            {"method": "gasvt384", "video_id": "v1", "p95_gap": 10, "boundary_recall_both": 0.55},
            {"method": "gasvt384", "video_id": "v2", "p95_gap": 6, "boundary_recall_both": 0.90},
        ],
    )
    _write_csv(
        geometry_dir / "action_summary.csv",
        [
            {
                "method": "gasvt384",
                "video_id": "v1",
                "action_id": "a1",
                "start_endpoint_coverage": 1,
                "end_endpoint_coverage": 0,
                "both_endpoint_coverage": 0,
                "boundary_recall": 0.20,
            },
            {
                "method": "gasvt384",
                "video_id": "v2",
                "action_id": "a2",
                "start_endpoint_coverage": 1,
                "end_endpoint_coverage": 1,
                "both_endpoint_coverage": 1,
                "boundary_recall": 0.90,
            },
        ],
    )
    _write_csv(
        geometry_dir / "selected_frame_metrics.csv",
        [
            {
                "method": "gasvt384",
                "video_id": "v1",
                "frame_index": 0,
                "selected": 1,
                "invalid_selected": 0,
                "is_padding": 0,
                "selected_count": 384,
                "required_selected_count": 384,
            },
            {
                "method": "uniform384",
                "video_id": "v1",
                "frame_index": 1,
                "selected": 1,
                "invalid_selected": 0,
                "is_padding": 0,
                "selected_count": 384,
                "required_selected_count": 384,
            },
        ],
    )
    _write_csv(
        geometry_dir / "holes_by_region.csv",
        [
            {"method": "gasvt384", "region": "action", "hole_count": 2, "frame_count": 20},
            {"method": "gasvt384", "region": "background", "hole_count": 1, "frame_count": 80},
            {"method": "uniform384", "region": "action", "hole_count": 1, "frame_count": 20},
        ],
    )
    return geometry_dir


def test_export_selector_paper_tables_writes_four_tables_with_optional_map(tmp_path: Path) -> None:
    geometry_dir = _toy_geometry_dir(tmp_path)
    map_json = tmp_path / "map.json"
    map_json.write_text(
        json.dumps({"methods": {"gasvt384": {"average_mAP": 41.2}, "uniform384": {"average_mAP": 39.8}}}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "paper"

    summary = paper_tables.export_tables(geometry_dir=geometry_dir, output_dir=out_dir, map_json=map_json)

    assert summary["decision"] == "SELECTOR_GEOMETRY_PAPER_TABLES_READY"
    for name in (
        "table1_map_vs_geometry.csv",
        "table2_boundary_recall.csv",
        "table3_region_share.csv",
        "table4_stage2_stage3_detector_aware.csv",
    ):
        assert (out_dir / name).is_file()
    table1 = _read_csv(out_dir / "table1_map_vs_geometry.csv")
    assert {row["method"] for row in table1} == {"gasvt384", "uniform384"}
    assert next(row for row in table1 if row["method"] == "gasvt384")["average_mAP"] == "41.2"
    table2 = _read_csv(out_dir / "table2_boundary_recall.csv")
    assert table2[0]["start_endpoint_coverage_mean"]
    table3 = _read_csv(out_dir / "table3_region_share.csv")
    assert {"method", "region", "hole_share", "frame_share"}.issubset(table3[0])


def test_export_selector_paper_tables_still_writes_geometry_tables_without_map(tmp_path: Path) -> None:
    geometry_dir = _toy_geometry_dir(tmp_path)
    out_dir = tmp_path / "paper_no_map"

    paper_tables.export_tables(geometry_dir=geometry_dir, output_dir=out_dir)

    table1 = _read_csv(out_dir / "table1_map_vs_geometry.csv")
    assert "average_mAP" in table1[0]
    assert all(row["average_mAP"] == "" for row in table1)


def test_validate_selector_geometry_metrics_enforces_contract_and_selected_rows(tmp_path: Path) -> None:
    geometry_dir = _toy_geometry_dir(tmp_path)

    report = validator.validate_geometry_dir(
        geometry_dir,
        require_coordinate_contract=True,
        require_no_padding_selected=True,
    )

    assert report["decision"] == "SELECTOR_GEOMETRY_VALIDATION_PASS"
    assert report["coordinate_contract_present"] is True
    assert report["selected_frame_metrics"]["invalid_selected_count"] == 0
    assert report["action_summary"]["has_both_endpoint_coverage"] is True


def test_validate_selector_geometry_metrics_rejects_invalid_or_padding_selected(tmp_path: Path) -> None:
    geometry_dir = _toy_geometry_dir(tmp_path)
    _write_csv(
        geometry_dir / "selected_frame_metrics.csv",
        [
            {
                "method": "gasvt384",
                "video_id": "v1",
                "frame_index": 99,
                "selected": 1,
                "invalid_selected": 1,
                "is_padding": 1,
                "selected_count": 384,
                "required_selected_count": 384,
            }
        ],
    )

    with pytest.raises(AssertionError, match="invalid selected"):
        validator.validate_geometry_dir(geometry_dir, require_no_padding_selected=True)


def test_generate_selector_failure_gallery_outputs_jsonl_and_index(tmp_path: Path) -> None:
    geometry_dir = _toy_geometry_dir(tmp_path)
    out_dir = tmp_path / "failures"

    summary = failure_gallery.generate_gallery(
        geometry_dir=geometry_dir,
        output_dir=out_dir,
        criteria=["low_boundary_recall", "high_p95_gap", "method_gap_vs_baseline"],
        top_k=1,
        baseline_method="uniform384",
    )

    assert summary["decision"] == "SELECTOR_FAILURE_GALLERY_READY"
    failures = _read_jsonl(out_dir / "failures.jsonl")
    assert {row["criterion"] for row in failures} == {
        "low_boundary_recall",
        "high_p95_gap",
        "method_gap_vs_baseline",
    }
    assert next(row for row in failures if row["criterion"] == "low_boundary_recall")["action_id"] == "a1"
    assert next(row for row in failures if row["criterion"] == "method_gap_vs_baseline")["baseline_method"] == "uniform384"
    index = _read_csv(out_dir / "gallery_index.csv")
    assert {"criterion", "method", "video_id", "score", "rank"}.issubset(index[0])
