from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "c3_selector_geometry_analysis_v1"
READY = "C3_SELECTOR_GEOMETRY_ANALYSIS_READY"
REGIONS = ("boundary_band", "action_interior", "background", "invalid")


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    label: str = ""
    instance_id: str = ""

    @property
    def length(self) -> int:
        return max(0, int(self.end) - int(self.start))


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL has no rows: {path}")
    return rows


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"expected NAME=PATH, got {raw!r}")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty name in {raw!r}")
    return name, Path(path).expanduser()


def _sample_map(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}: sample row missing sample_id")
        if sample_id in out:
            raise ValueError(f"{path}: duplicate sample_id {sample_id}")
        out[sample_id] = row
    return out


def _video_id(sample_id: str) -> str:
    return str(sample_id).split("|", 1)[0]


def _as_float(value: Any, *, default: float | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _segment_from_any(raw: Any, index: int) -> Segment | None:
    label = ""
    instance_id = f"gt_{index}"
    if isinstance(raw, Mapping):
        segment = raw.get("segment", raw.get("segments", raw.get("range")))
        label = str(raw.get("label", raw.get("class", raw.get("class_name", ""))))
        instance_id = str(raw.get("instance_id", raw.get("id", instance_id)))
        raw = segment
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or len(raw) < 2:
        return None
    start = _as_float(raw[0])
    end = _as_float(raw[1])
    if start is None or end is None:
        return None
    start_i = int(math.floor(start))
    end_i = int(math.ceil(end))
    if end_i <= start_i:
        return None
    return Segment(start=start_i, end=end_i, label=label, instance_id=instance_id)


def _segments_from_sample(row: Mapping[str, Any]) -> list[Segment]:
    raw = row.get("gt_segments", row.get("segments", row.get("annotations")))
    if not isinstance(raw, list):
        return []
    out: list[Segment] = []
    for idx, item in enumerate(raw):
        segment = _segment_from_any(item, idx)
        if segment is not None:
            out.append(segment)
    return out


def _annotation_segments(annotation: Mapping[str, Any]) -> dict[str, list[Segment]]:
    if not annotation:
        return {}
    database = annotation.get("database") if isinstance(annotation.get("database"), Mapping) else annotation
    if not isinstance(database, Mapping):
        return {}
    out: dict[str, list[Segment]] = {}
    for video_id, payload in database.items():
        annotations = payload.get("annotations") if isinstance(payload, Mapping) else None
        if not isinstance(annotations, list):
            continue
        segments: list[Segment] = []
        for idx, item in enumerate(annotations):
            segment = _segment_from_any(item, idx)
            if segment is not None:
                segments.append(segment)
        out[str(video_id)] = segments
    return out


def _p_action_values(row: Mapping[str, Any] | None) -> list[float]:
    if row is None:
        return []
    frame_signals = row.get("frame_signals")
    raw = frame_signals.get("p_action") if isinstance(frame_signals, Mapping) else None
    if raw is None:
        raw = row.get("p_action")
    if not isinstance(raw, list):
        return []
    out: list[float] = []
    for item in raw:
        value = _as_float(item)
        if value is None:
            return []
        out.append(value)
    return out


def _positions(value: Any, *, context: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: selected_positions must be a list")
    out = [int(item) for item in value]
    if out != sorted(out):
        raise ValueError(f"{context}: selected_positions must be sorted")
    if len(set(out)) != len(out):
        raise ValueError(f"{context}: selected_positions must be unique")
    if any(item < 0 for item in out):
        raise ValueError(f"{context}: selected_positions must be non-negative")
    return out


def _valid_len(ledger_row: Mapping[str, Any], sample_row: Mapping[str, Any] | None, p_action: Sequence[float]) -> int:
    for row in (ledger_row, sample_row or {}):
        for key in ("valid_len", "dense_valid_len", "selected_valid_len", "dense_len"):
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(0, int(value))
    return len(p_action)


def boundary_frames(segments: Sequence[Segment]) -> list[int]:
    out: list[int] = []
    for segment in segments:
        if segment.length <= 0:
            continue
        out.append(int(segment.start))
        out.append(int(segment.end) - 1)
    return out


def nearest_boundary_distance(position: int, segments: Sequence[Segment]) -> int | None:
    boundaries = boundary_frames(segments)
    if not boundaries:
        return None
    return min(abs(int(position) - int(boundary)) for boundary in boundaries)


def inside_action(position: int, segments: Sequence[Segment]) -> bool:
    return any(int(segment.start) <= int(position) < int(segment.end) for segment in segments)


def normalized_action_time(position: int, segments: Sequence[Segment]) -> float | None:
    hits = [segment for segment in segments if int(segment.start) <= int(position) < int(segment.end) and segment.length > 0]
    if not hits:
        return None
    segment = min(hits, key=lambda item: item.length)
    return (float(position) - float(segment.start)) / max(1.0, float(segment.length))


def classify_region(position: int, *, valid_len: int, segments: Sequence[Segment], boundary_radius: int) -> str:
    if int(position) < 0 or int(position) >= int(valid_len):
        return "invalid"
    dist = nearest_boundary_distance(int(position), segments)
    if dist is not None and dist <= int(boundary_radius):
        return "boundary_band"
    if inside_action(int(position), segments):
        return "action_interior"
    return "background"


def _mask_for_region(valid_len: int, segments: Sequence[Segment], boundary_radius: int, region: str) -> list[bool]:
    return [
        classify_region(idx, valid_len=valid_len, segments=segments, boundary_radius=boundary_radius) == region
        for idx in range(max(0, int(valid_len)))
    ]


def _holes_for_mask(selected: set[int], mask: Sequence[bool]) -> list[int]:
    holes: list[int] = []
    run = 0
    for idx, keep in enumerate(mask):
        if not keep:
            if run:
                holes.append(run)
                run = 0
            continue
        if idx in selected:
            if run:
                holes.append(run)
                run = 0
        else:
            run += 1
    if run:
        holes.append(run)
    return holes


def _percentile(values: Sequence[float], q: float) -> float | None:
    finite = sorted(float(item) for item in values if math.isfinite(float(item)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    rank = (len(finite) - 1) * float(q)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return finite[int(rank)]
    return finite[low] + (finite[high] - finite[low]) * (rank - low)


def _mean(values: Sequence[float]) -> float | None:
    finite = [float(item) for item in values if math.isfinite(float(item))]
    return None if not finite else sum(finite) / float(len(finite))


def _summary_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 6)
    return value


def analyze_geometry(
    *,
    selector_ledgers: Mapping[str, str | Path],
    sample_jsonls: Mapping[str, str | Path],
    annotation_json: str | Path | None = None,
    out_dir: str | Path,
    run_tag: str,
    split: str,
    coordinate_system: str = "dense_frame_index",
    gt_convention: str = "half_open_[start,end)",
    frame_stride_seconds: float = 1.0,
    boundary_band_radius: int = 4,
    radii_frames: Sequence[int] = (1, 2, 4, 8, 16),
) -> dict[str, Any]:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    annotations = _annotation_segments(_read_json(annotation_json)) if annotation_json is not None else {}
    sample_maps = {method: _sample_map(path) for method, path in sample_jsonls.items()}
    common_sample_map = sample_maps.get("*", {})
    selected_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    hole_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    radii = sorted({int(radius) for radius in radii_frames})
    if not radii:
        radii = [int(boundary_band_radius)]

    for method, ledger_path in selector_ledgers.items():
        method_samples = sample_maps.get(method, common_sample_map)
        for line_no, ledger_row in enumerate(_read_jsonl(ledger_path), start=1):
            sample_id = ledger_row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{ledger_path}:{line_no}: missing sample_id")
            sample_row = method_samples.get(sample_id)
            p_action = _p_action_values(sample_row)
            valid_len = _valid_len(ledger_row, sample_row, p_action)
            selected = _positions(ledger_row.get("selected_positions"), context=f"{ledger_path}:{line_no}")
            video_id = str(ledger_row.get("video_id") or _video_id(sample_id))
            segments = _segments_from_sample(sample_row or {}) or annotations.get(video_id, [])
            boundaries = boundary_frames(segments)
            selected_set = {int(item) for item in selected}
            target_len = int(ledger_row.get("target_len") or ledger_row.get("required_selected_count") or len(selected))
            region_counts = {region: 0 for region in REGIONS}
            distances: list[float] = []
            normalized_positions: list[float] = []
            p_values_selected: list[float] = []

            for frame_idx in range(max(0, int(valid_len))):
                frame_region = classify_region(
                    frame_idx,
                    valid_len=valid_len,
                    segments=segments,
                    boundary_radius=boundary_band_radius,
                )
                frame_rows.append(
                    {
                        "run_tag": run_tag,
                        "split": split,
                        "method": method,
                        "sample_id": sample_id,
                        "video_id": video_id,
                        "frame_index": frame_idx,
                        "region": frame_region,
                        "inside_action": int(inside_action(frame_idx, segments)),
                        "within_boundary_band": int(frame_region == "boundary_band"),
                        "p_action": _summary_value(p_action[frame_idx] if frame_idx < len(p_action) else None),
                    }
                )

            for position in selected:
                dist = nearest_boundary_distance(position, segments)
                region = classify_region(
                    position,
                    valid_len=valid_len,
                    segments=segments,
                    boundary_radius=boundary_band_radius,
                )
                region_counts[region] += 1
                if dist is not None:
                    distances.append(float(dist))
                norm = normalized_action_time(position, segments)
                if norm is not None:
                    normalized_positions.append(float(norm))
                p_value = p_action[position] if 0 <= position < len(p_action) else None
                if p_value is not None:
                    p_values_selected.append(float(p_value))
                selected_rows.append(
                    {
                        "run_tag": run_tag,
                        "split": split,
                        "method": method,
                        "sample_id": sample_id,
                        "video_id": video_id,
                        "selected_position": int(position),
                        "selected_index": selected.index(position),
                        "selected_count": len(selected),
                        "target_len": target_len,
                        "valid_len": int(valid_len),
                        "nearest_boundary_distance_frame": _summary_value(dist),
                        "nearest_boundary_distance_second": _summary_value(None if dist is None else dist * frame_stride_seconds),
                        "region": region,
                        "inside_action": int(inside_action(position, segments)),
                        "within_boundary_band": int(region == "boundary_band"),
                        "normalized_action_time": _summary_value(norm),
                        "p_action": _summary_value(p_value),
                        "invalid_selected": int(region == "invalid"),
                    }
                )

            for region in ("whole_video", "boundary_band", "action_interior", "background"):
                mask = [True] * max(0, int(valid_len)) if region == "whole_video" else _mask_for_region(valid_len, segments, boundary_band_radius, region)
                holes = _holes_for_mask(selected_set, mask)
                hole_rows.append(
                    {
                        "run_tag": run_tag,
                        "split": split,
                        "method": method,
                        "sample_id": sample_id,
                        "video_id": video_id,
                        "region": region,
                        "num_holes": len(holes),
                        "max_unselected_hole": max(holes) if holes else 0,
                        "p95_unselected_hole": _summary_value(_percentile([float(item) for item in holes], 0.95) or 0.0),
                        "mean_unselected_hole": _summary_value(_mean([float(item) for item in holes]) or 0.0),
                    }
                )

            for action_idx, segment in enumerate(segments):
                if segment.length <= 0:
                    continue
                start_boundary = int(segment.start)
                end_boundary = int(segment.end) - 1
                for radius in radii:
                    start_hit = any(abs(pos - start_boundary) <= radius for pos in selected_set)
                    end_hit = any(abs(pos - end_boundary) <= radius for pos in selected_set)
                    action_rows.append(
                        {
                            "run_tag": run_tag,
                            "split": split,
                            "method": method,
                            "sample_id": sample_id,
                            "video_id": video_id,
                            "instance_id": segment.instance_id or f"gt_{action_idx}",
                            "label": segment.label,
                            "action_start": int(segment.start),
                            "action_end": int(segment.end),
                            "action_len": int(segment.length),
                            "radius_frame": int(radius),
                            "start_hit": int(start_hit),
                            "end_hit": int(end_hit),
                            "both_endpoint_hit": int(start_hit and end_hit),
                            "start_endpoint_coverage": int(start_hit),
                            "end_endpoint_coverage": int(end_hit),
                            "both_endpoint_coverage": int(start_hit and end_hit),
                            "any_endpoint_hit": int(start_hit or end_hit),
                            "any_selected_inside_action": int(any(segment.start <= pos < segment.end for pos in selected_set)),
                            "interior_selected_count": sum(1 for pos in selected_set if segment.start < pos < segment.end - 1),
                        }
                    )

            if p_action and boundaries:
                for quantile in (0.5, 0.7, 0.8, 0.9, 0.95):
                    threshold = _percentile(p_action, quantile)
                    if threshold is None:
                        continue
                    kept = [idx for idx, value in enumerate(p_action[:valid_len]) if float(value) >= float(threshold)]
                    kept_dist = [nearest_boundary_distance(idx, segments) for idx in kept]
                    kept_dist_f = [float(item) for item in kept_dist if item is not None]
                    calibration_rows.append(
                        {
                            "run_tag": run_tag,
                            "split": split,
                            "method": method,
                            "sample_id": sample_id,
                            "video_id": video_id,
                            "score_field": "p_action",
                            "score_quantile": quantile,
                            "threshold": _summary_value(threshold),
                            "num_positions": len(kept),
                            "mean_distance_frame": _summary_value(_mean(kept_dist_f)),
                            "median_distance_frame": _summary_value(_percentile(kept_dist_f, 0.5)),
                            "p90_distance_frame": _summary_value(_percentile(kept_dist_f, 0.9)),
                            "frac_within_4_frames": _summary_value(_mean([1.0 if item <= 4 else 0.0 for item in kept_dist_f])),
                            "frac_within_8_frames": _summary_value(_mean([1.0 if item <= 8 else 0.0 for item in kept_dist_f])),
                        }
                    )

            selected_count = len(selected)
            denom = float(selected_count) if selected_count else 1.0
            boundary_hits_by_radius = {
                radius: sum(1 for boundary in boundaries if any(abs(pos - boundary) <= radius for pos in selected_set))
                for radius in radii
            }
            video_row: dict[str, Any] = {
                "run_tag": run_tag,
                "split": split,
                "method": method,
                "sample_id": sample_id,
                "video_id": video_id,
                "selected_count": selected_count,
                "valid_len": int(valid_len),
                "gt_action_count": len(segments),
                "gt_boundary_count": len(boundaries),
                "boundary_band_selected_count": region_counts["boundary_band"],
                "action_interior_selected_count": region_counts["action_interior"],
                "background_selected_count": region_counts["background"],
                "invalid_selected_count": region_counts["invalid"],
                "boundary_band_selected_ratio": _summary_value(region_counts["boundary_band"] / denom),
                "action_interior_selected_ratio": _summary_value(region_counts["action_interior"] / denom),
                "background_selected_ratio": _summary_value(region_counts["background"] / denom),
                "invalid_selected_ratio": _summary_value(region_counts["invalid"] / denom),
                "median_boundary_distance": _summary_value(_percentile(distances, 0.5)),
                "p90_boundary_distance": _summary_value(_percentile(distances, 0.9)),
                "p95_boundary_distance": _summary_value(_percentile(distances, 0.95)),
                "mean_selected_p_action": _summary_value(_mean(p_values_selected)),
                "mean_normalized_action_time": _summary_value(_mean(normalized_positions)),
            }
            for radius in radii:
                video_row[f"boundary_recall_r{radius}"] = _summary_value(
                    None if not boundaries else boundary_hits_by_radius[radius] / float(len(boundaries))
                )
            video_rows.append(video_row)

    method_rows: list[dict[str, Any]] = []
    for method in selector_ledgers:
        method_videos = [row for row in video_rows if row["method"] == method]
        method_selected = [row for row in selected_rows if row["method"] == method]
        method_actions = [row for row in action_rows if row["method"] == method]
        method_holes = [row for row in hole_rows if row["method"] == method and row["region"] == "whole_video"]
        distances = [
            float(row["nearest_boundary_distance_frame"])
            for row in method_selected
            if row.get("nearest_boundary_distance_frame") != ""
        ]
        method_row: dict[str, Any] = {
            "run_tag": run_tag,
            "split": split,
            "method": method,
            "video_count": len(method_videos),
            "selected_count_mean": _summary_value(_mean([float(row["selected_count"]) for row in method_videos])),
            "boundary_band_selected_ratio_mean": _summary_value(_mean([float(row["boundary_band_selected_ratio"]) for row in method_videos if row["boundary_band_selected_ratio"] != ""])),
            "action_interior_selected_ratio_mean": _summary_value(_mean([float(row["action_interior_selected_ratio"]) for row in method_videos if row["action_interior_selected_ratio"] != ""])),
            "background_selected_ratio_mean": _summary_value(_mean([float(row["background_selected_ratio"]) for row in method_videos if row["background_selected_ratio"] != ""])),
            "invalid_selected_count_total": sum(int(row["invalid_selected_count"]) for row in method_videos),
            "median_boundary_distance": _summary_value(_percentile(distances, 0.5)),
            "p90_boundary_distance": _summary_value(_percentile(distances, 0.9)),
            "p95_boundary_distance": _summary_value(_percentile(distances, 0.95)),
            "max_unselected_hole_mean": _summary_value(_mean([float(row["max_unselected_hole"]) for row in method_holes])),
            "p95_unselected_hole_mean": _summary_value(_mean([float(row["p95_unselected_hole"]) for row in method_holes])),
        }
        for radius in radii:
            method_row[f"boundary_recall_r{radius}_mean"] = _summary_value(
                _mean([float(row[f"boundary_recall_r{radius}"]) for row in method_videos if row.get(f"boundary_recall_r{radius}") != ""])
            )
            radius_actions = [row for row in method_actions if int(row["radius_frame"]) == int(radius)]
            method_row[f"endpoint_both_coverage_r{radius}_mean"] = _summary_value(
                _mean([float(row["both_endpoint_hit"]) for row in radius_actions])
            )
        method_rows.append(method_row)

    selected_fields = [
        "run_tag", "split", "method", "sample_id", "video_id", "selected_position", "selected_index", "valid_len",
        "selected_count", "target_len", "nearest_boundary_distance_frame", "nearest_boundary_distance_second", "region", "inside_action",
        "within_boundary_band", "normalized_action_time", "p_action", "invalid_selected",
    ]
    frame_fields = ["run_tag", "split", "method", "sample_id", "video_id", "frame_index", "region", "inside_action", "within_boundary_band", "p_action"]
    video_fields = [
        "run_tag", "split", "method", "sample_id", "video_id", "selected_count", "valid_len", "gt_action_count",
        "gt_boundary_count", "boundary_band_selected_count", "action_interior_selected_count", "background_selected_count",
        "invalid_selected_count", "boundary_band_selected_ratio", "action_interior_selected_ratio",
        "background_selected_ratio", "invalid_selected_ratio", "median_boundary_distance", "p90_boundary_distance",
        "p95_boundary_distance", "mean_selected_p_action", "mean_normalized_action_time",
        *[f"boundary_recall_r{radius}" for radius in radii],
    ]
    action_fields = [
        "run_tag", "split", "method", "sample_id", "video_id", "instance_id", "label", "action_start", "action_end",
        "action_len", "radius_frame", "start_hit", "end_hit", "both_endpoint_hit", "any_endpoint_hit",
        "start_endpoint_coverage", "end_endpoint_coverage", "both_endpoint_coverage", "any_selected_inside_action",
        "interior_selected_count",
    ]
    hole_fields = ["run_tag", "split", "method", "sample_id", "video_id", "region", "num_holes", "max_unselected_hole", "p95_unselected_hole", "mean_unselected_hole"]
    calibration_fields = [
        "run_tag", "split", "method", "sample_id", "video_id", "score_field", "score_quantile", "threshold",
        "num_positions", "mean_distance_frame", "median_distance_frame", "p90_distance_frame", "frac_within_4_frames",
        "frac_within_8_frames",
    ]
    method_fields = [
        "run_tag", "split", "method", "video_count", "selected_count_mean", "boundary_band_selected_ratio_mean",
        "action_interior_selected_ratio_mean", "background_selected_ratio_mean", "invalid_selected_count_total",
        "median_boundary_distance", "p90_boundary_distance", "p95_boundary_distance", "max_unselected_hole_mean",
        "p95_unselected_hole_mean",
        *[f"boundary_recall_r{radius}_mean" for radius in radii],
        *[f"endpoint_both_coverage_r{radius}_mean" for radius in radii],
    ]
    _write_csv(out / "selected_frame_metrics.csv", selected_rows, selected_fields)
    _write_csv(out / "frame_metrics.csv", frame_rows, frame_fields)
    _write_csv(out / "video_summary.csv", video_rows, video_fields)
    _write_csv(out / "action_summary.csv", action_rows, action_fields)
    _write_csv(out / "holes_by_region.csv", hole_rows, hole_fields)
    _write_csv(out / "paction_calibration.csv", calibration_rows, calibration_fields)
    _write_csv(out / "method_summary.csv", method_rows, method_fields)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "decision": READY,
        "run_tag": run_tag,
        "split": split,
        "methods": sorted(selector_ledgers),
        "row_counts": {
            "selected_frame_metrics": len(selected_rows),
            "frame_metrics": len(frame_rows),
            "video_summary": len(video_rows),
            "action_summary": len(action_rows),
            "holes_by_region": len(hole_rows),
            "paction_calibration": len(calibration_rows),
            "method_summary": len(method_rows),
        },
        "coordinate_contract": {
            "coordinate_system": coordinate_system,
            "axis": coordinate_system,
            "gt_segment_convention": gt_convention,
            "selected_positions_unit": "local_dense_index",
            "selected_positions_are_centers": True,
            "frame_stride_seconds": float(frame_stride_seconds),
            "boundary_band_radius": int(boundary_band_radius),
            "radii_frames": radii,
        },
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze sparse selector geometry under matched dense-frame coordinates.")
    parser.add_argument("--selector-ledger", action="append", required=True, metavar="METHOD=PATH")
    parser.add_argument("--sample-jsonl", action="append", default=[], metavar="METHOD=PATH")
    parser.add_argument("--common-sample-jsonl", default=None)
    parser.add_argument("--annotation-json", default=None)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--coordinate-system", default="dense_frame_index")
    parser.add_argument("--gt-convention", default="half_open_[start,end)")
    parser.add_argument("--frame-stride-seconds", type=float, default=1.0)
    parser.add_argument("--boundary-band-radius", type=int, default=4)
    parser.add_argument("--radii-frames", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    selector_ledgers = dict(_parse_named_path(item) for item in args.selector_ledger)
    sample_jsonls = dict(_parse_named_path(item) for item in args.sample_jsonl)
    if args.common_sample_jsonl:
        sample_jsonls["*"] = Path(args.common_sample_jsonl).expanduser()
    analyze_geometry(
        selector_ledgers=selector_ledgers,
        sample_jsonls=sample_jsonls,
        annotation_json=args.annotation_json,
        out_dir=args.out_dir,
        run_tag=args.run_tag,
        split=args.split,
        coordinate_system=args.coordinate_system,
        gt_convention=args.gt_convention,
        frame_stride_seconds=float(args.frame_stride_seconds),
        boundary_band_radius=int(args.boundary_band_radius),
        radii_frames=args.radii_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
