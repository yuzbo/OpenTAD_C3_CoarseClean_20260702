from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "selector_geometry_paper_tables_v1"
READY = "SELECTOR_GEOMETRY_PAPER_TABLES_READY"


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row.get(key)) for key in fieldnames})


def _format_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return value


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row.get(name) not in (None, ""):
            return row.get(name)
    return None


def _method(row: Mapping[str, Any]) -> str:
    value = _first_present(row, ("method", "selector", "strategy", "variant", "name"))
    return "" if value is None else str(value)


def _load_map_by_method(map_json: str | Path | None) -> dict[str, dict[str, Any]]:
    if map_json is None:
        return {}
    payload = _read_json(map_json)
    source: Any = payload.get("methods", payload.get("method_metrics", payload.get("results", payload)))
    out: dict[str, dict[str, Any]] = {}
    if isinstance(source, Mapping):
        for name, metrics in source.items():
            if isinstance(metrics, Mapping):
                out[str(name)] = dict(metrics)
            else:
                out[str(name)] = {"average_mAP": metrics}
    elif isinstance(source, list):
        for item in source:
            if isinstance(item, Mapping):
                method = _method(item)
                if method:
                    out[method] = dict(item)
    return out


def _map_value(map_by_method: Mapping[str, Mapping[str, Any]], method: str) -> Any:
    metrics = map_by_method.get(method, {})
    return _first_present(metrics, ("average_mAP", "avg_mAP", "mAP", "mean_mAP", "best_average_mAP"))


def _mean(rows: Sequence[Mapping[str, Any]], aliases: Sequence[str]) -> float | None:
    values = [_finite_float(_first_present(row, aliases)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _count_rows_by_method(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        method = _method(row)
        if method:
            grouped.setdefault(method, []).append(row)
    return grouped


def _method_summary_table(method_rows: Sequence[Mapping[str, Any]], map_by_method: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sorted(method_rows, key=_method):
        method = _method(row)
        out = {"method": method, "average_mAP": _map_value(map_by_method, method)}
        for key in sorted(row):
            if key not in {"method", "selector", "strategy", "variant", "name"}:
                out[key] = row.get(key)
        rows.append(out)
    return rows


def _boundary_recall_table(action_rows: Sequence[Mapping[str, Any]], method_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = _count_rows_by_method(action_rows)
    if not grouped:
        grouped = _count_rows_by_method(method_rows)
    table: list[dict[str, Any]] = []
    for method in sorted(grouped):
        rows = grouped[method]
        table.append(
            {
                "method": method,
                "action_count": len(rows),
                "start_endpoint_coverage_mean": _mean(rows, ("start_endpoint_coverage", "start_coverage", "start_covered")),
                "end_endpoint_coverage_mean": _mean(rows, ("end_endpoint_coverage", "end_coverage", "end_covered")),
                "both_endpoint_coverage_mean": _mean(
                    rows,
                    ("both_endpoint_coverage", "both_endpoint_covered", "both_coverage", "boundary_recall_both"),
                ),
                "boundary_recall_mean": _mean(rows, ("boundary_recall", "boundary_support", "boundary_support_r1")),
            }
        )
    return table


def _region_share_table(hole_rows: Sequence[Mapping[str, Any]], method_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = _count_rows_by_method(hole_rows)
    table: list[dict[str, Any]] = []
    for method in sorted(grouped):
        rows = grouped[method]
        total_holes = sum(_finite_float(_first_present(row, ("hole_count", "holes", "unselected_hole_count"))) or 0.0 for row in rows)
        total_frames = sum(_finite_float(_first_present(row, ("frame_count", "frames", "region_frames"))) or 0.0 for row in rows)
        for row in sorted(rows, key=lambda item: str(item.get("region", ""))):
            hole_count = _finite_float(_first_present(row, ("hole_count", "holes", "unselected_hole_count"))) or 0.0
            frame_count = _finite_float(_first_present(row, ("frame_count", "frames", "region_frames"))) or 0.0
            table.append(
                {
                    "method": method,
                    "region": row.get("region", ""),
                    "hole_count": hole_count,
                    "frame_count": frame_count,
                    "hole_share": None if total_holes <= 0.0 else hole_count / total_holes,
                    "frame_share": None if total_frames <= 0.0 else frame_count / total_frames,
                }
            )
    if table:
        return table
    for row in method_rows:
        method = _method(row)
        for key, value in row.items():
            if key.endswith("_region_share"):
                table.append({"method": method, "region": key[: -len("_region_share")], "hole_share": "", "frame_share": value})
    return table


def _stage_detector_table(
    method_rows: Sequence[Mapping[str, Any]], map_by_method: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in sorted(method_rows, key=_method):
        method = _method(row)
        table.append(
            {
                "method": method,
                "stage": _first_present(row, ("stage", "selector_stage", "route_stage")),
                "average_mAP": _map_value(map_by_method, method),
                "selected_count_mean": _first_present(row, ("selected_count_mean", "mean_selected_count")),
                "selected_count_min": _first_present(row, ("selected_count_min", "min_selected_count")),
                "selected_count_max": _first_present(row, ("selected_count_max", "max_selected_count")),
                "p95_gap": _first_present(row, ("p95_gap", "p95_unselected_hole")),
                "boundary_recall_both": _first_present(
                    row, ("boundary_recall_both", "both_endpoint_coverage", "boundary_bracket_support_r1")
                ),
                "detector_aware_metric": _first_present(
                    row,
                    (
                        "detector_aware_score",
                        "teacher_utility_mean",
                        "responsibility_utility_mean",
                        "action_region_share",
                    ),
                ),
            }
        )
    return table


def export_tables(
    *,
    geometry_dir: str | Path,
    output_dir: str | Path,
    map_json: str | Path | None = None,
) -> dict[str, Any]:
    geometry_path = Path(geometry_dir).expanduser()
    out_path = Path(output_dir).expanduser()
    method_rows = _read_csv(geometry_path / "method_summary.csv")
    action_rows = _read_csv(geometry_path / "action_summary.csv")
    hole_rows = _read_csv(geometry_path / "holes_by_region.csv")
    map_by_method = _load_map_by_method(map_json)

    table1 = _method_summary_table(method_rows, map_by_method)
    table2 = _boundary_recall_table(action_rows, method_rows)
    table3 = _region_share_table(hole_rows, method_rows)
    table4 = _stage_detector_table(method_rows, map_by_method)

    outputs = {
        "table1_map_vs_geometry.csv": (
            table1,
            ["method", "average_mAP"]
            + sorted({key for row in table1 for key in row if key not in {"method", "average_mAP"}}),
        ),
        "table2_boundary_recall.csv": (
            table2,
            [
                "method",
                "action_count",
                "start_endpoint_coverage_mean",
                "end_endpoint_coverage_mean",
                "both_endpoint_coverage_mean",
                "boundary_recall_mean",
            ],
        ),
        "table3_region_share.csv": (
            table3,
            ["method", "region", "hole_count", "frame_count", "hole_share", "frame_share"],
        ),
        "table4_stage2_stage3_detector_aware.csv": (
            table4,
            [
                "method",
                "stage",
                "average_mAP",
                "selected_count_mean",
                "selected_count_min",
                "selected_count_max",
                "p95_gap",
                "boundary_recall_both",
                "detector_aware_metric",
            ],
        ),
    }
    for filename, (rows, fieldnames) in outputs.items():
        _write_csv(out_path / filename, rows, fieldnames)

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": READY,
        "geometry_dir": str(geometry_path),
        "output_dir": str(out_path),
        "map_json": None if map_json is None else str(map_json),
        "row_counts": {name: len(rows) for name, (rows, _) in outputs.items()},
        "outputs": {name: str(out_path / name) for name in outputs},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export selector geometry analyzer outputs into paper-ready CSV tables.")
    parser.add_argument("--geometry-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--map-json", help="Optional detector mAP JSON keyed by method.")
    args = parser.parse_args(argv)
    summary = export_tables(geometry_dir=args.geometry_dir, output_dir=args.output_dir, map_json=args.map_json)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
