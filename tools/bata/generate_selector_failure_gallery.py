from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "selector_failure_gallery_v1"
READY = "SELECTOR_FAILURE_GALLERY_READY"
SUPPORTED_CRITERIA = ("low_boundary_recall", "high_p95_gap", "method_gap_vs_baseline")


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row.get(key)) for key in fieldnames})


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")


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


def _first_present(row: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for name in aliases:
        if name in row and row.get(name) not in (None, ""):
            return row.get(name)
    return None


def _method(row: Mapping[str, Any]) -> str:
    value = _first_present(row, ("method", "selector", "strategy", "variant", "name"))
    return "" if value is None else str(value)


def _video_id(row: Mapping[str, Any]) -> str:
    value = _first_present(row, ("video_id", "video_name", "video", "sample_id"))
    return "" if value is None else str(value)


def _action_id(row: Mapping[str, Any]) -> str:
    value = _first_present(row, ("action_id", "segment_id", "instance_id", "annotation_id"))
    return "" if value is None else str(value)


def _base_failure(
    *,
    criterion: str,
    rank: int,
    row: Mapping[str, Any],
    score: float,
    metric_name: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "rank": int(rank),
        "method": _method(row),
        "video_id": _video_id(row),
        "action_id": _action_id(row),
        "score": score,
        "metric_name": metric_name,
        "metric_value": score,
        "source_row": dict(row),
    }


def _low_boundary_recall(action_rows: Sequence[Mapping[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    scored: list[tuple[float, Mapping[str, Any]]] = []
    for row in action_rows:
        value = _finite_float(
            _first_present(row, ("boundary_recall", "both_endpoint_coverage", "boundary_recall_both", "boundary_support"))
        )
        if value is not None:
            scored.append((value, row))
    scored.sort(key=lambda item: item[0])
    return [
        _base_failure(criterion="low_boundary_recall", rank=rank, row=row, score=value, metric_name="boundary_recall")
        for rank, (value, row) in enumerate(scored[:top_k], start=1)
    ]


def _high_p95_gap(video_rows: Sequence[Mapping[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    scored: list[tuple[float, Mapping[str, Any]]] = []
    for row in video_rows:
        value = _finite_float(_first_present(row, ("p95_gap", "p95_unselected_hole", "max_gap")))
        if value is not None:
            scored.append((value, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        _base_failure(criterion="high_p95_gap", rank=rank, row=row, score=value, metric_name="p95_gap")
        for rank, (value, row) in enumerate(scored[:top_k], start=1)
    ]


def _method_gap_vs_baseline(
    video_rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    baseline_method: str | None,
) -> list[dict[str, Any]]:
    if not baseline_method:
        return []
    baseline_by_video: dict[str, Mapping[str, Any]] = {
        _video_id(row): row for row in video_rows if _method(row) == baseline_method and _video_id(row)
    }
    scored: list[tuple[float, Mapping[str, Any], Mapping[str, Any], float, float]] = []
    for row in video_rows:
        method = _method(row)
        video_id = _video_id(row)
        if not video_id or method == baseline_method:
            continue
        baseline = baseline_by_video.get(video_id)
        if baseline is None:
            continue
        method_value = _finite_float(
            _first_present(row, ("boundary_recall_both", "both_endpoint_coverage", "boundary_recall", "action_coverage"))
        )
        baseline_value = _finite_float(
            _first_present(
                baseline,
                ("boundary_recall_both", "both_endpoint_coverage", "boundary_recall", "action_coverage"),
            )
        )
        if method_value is None or baseline_value is None:
            continue
        scored.append((baseline_value - method_value, row, baseline, method_value, baseline_value))
    scored.sort(key=lambda item: item[0], reverse=True)
    failures: list[dict[str, Any]] = []
    for rank, (gap, row, baseline, method_value, baseline_value) in enumerate(scored[:top_k], start=1):
        failure = _base_failure(
            criterion="method_gap_vs_baseline",
            rank=rank,
            row=row,
            score=gap,
            metric_name="baseline_minus_method_boundary_recall",
        )
        failure.update(
            {
                "baseline_method": baseline_method,
                "baseline_metric_value": baseline_value,
                "method_metric_value": method_value,
                "baseline_source_row": dict(baseline),
            }
        )
        failures.append(failure)
    return failures


def _failure_rows_for_criterion(
    criterion: str,
    *,
    video_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    top_k: int,
    baseline_method: str | None,
) -> list[dict[str, Any]]:
    if criterion == "low_boundary_recall":
        return _low_boundary_recall(action_rows, top_k=top_k)
    if criterion == "high_p95_gap":
        return _high_p95_gap(video_rows, top_k=top_k)
    if criterion == "method_gap_vs_baseline":
        return _method_gap_vs_baseline(video_rows, top_k=top_k, baseline_method=baseline_method)
    raise ValueError(f"unsupported criterion: {criterion}")


def generate_gallery(
    *,
    geometry_dir: str | Path,
    output_dir: str | Path,
    criteria: Sequence[str] = SUPPORTED_CRITERIA,
    top_k: int = 10,
    baseline_method: str | None = None,
) -> dict[str, Any]:
    geometry_path = Path(geometry_dir).expanduser()
    out_path = Path(output_dir).expanduser()
    video_rows = _read_csv(geometry_path / "video_summary.csv")
    action_rows = _read_csv(geometry_path / "action_summary.csv")
    failures: list[dict[str, Any]] = []
    for criterion in criteria:
        failures.extend(
            _failure_rows_for_criterion(
                criterion,
                video_rows=video_rows,
                action_rows=action_rows,
                top_k=int(top_k),
                baseline_method=baseline_method,
            )
        )

    _write_jsonl(out_path / "failures.jsonl", failures)
    index_rows = [
        {
            "criterion": row.get("criterion"),
            "rank": row.get("rank"),
            "method": row.get("method"),
            "video_id": row.get("video_id"),
            "action_id": row.get("action_id"),
            "score": row.get("score"),
            "metric_name": row.get("metric_name"),
            "baseline_method": row.get("baseline_method"),
        }
        for row in failures
    ]
    _write_csv(
        out_path / "gallery_index.csv",
        index_rows,
        ["criterion", "rank", "method", "video_id", "action_id", "score", "metric_name", "baseline_method"],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": READY,
        "geometry_dir": str(geometry_path),
        "output_dir": str(out_path),
        "criteria": list(criteria),
        "top_k": int(top_k),
        "baseline_method": baseline_method,
        "failure_count": len(failures),
        "outputs": {
            "failures_jsonl": str(out_path / "failures.jsonl"),
            "gallery_index_csv": str(out_path / "gallery_index.csv"),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate selector geometry failure-gallery indexes from analyzer outputs.")
    parser.add_argument("--geometry-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--criteria", action="append", choices=SUPPORTED_CRITERIA)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--baseline-method")
    args = parser.parse_args(argv)
    criteria = args.criteria or list(SUPPORTED_CRITERIA)
    summary = generate_gallery(
        geometry_dir=args.geometry_dir,
        output_dir=args.output_dir,
        criteria=criteria,
        top_k=args.top_k,
        baseline_method=args.baseline_method,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
