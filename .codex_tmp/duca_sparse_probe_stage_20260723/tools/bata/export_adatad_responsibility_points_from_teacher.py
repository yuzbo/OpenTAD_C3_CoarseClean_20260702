from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROW_SCHEMA_VERSION = "c3_adatad_responsibility_points_from_teacher_row_v1"
SUMMARY_SCHEMA_VERSION = "c3_adatad_responsibility_points_from_teacher_export_v1"
MANIFEST_SCHEMA_VERSION = "c3_adatad_responsibility_points_from_teacher_manifest_v1"
READY = "ADATAD_RESPONSIBILITY_POINTS_FROM_TEACHER_READY"
UTILITY_SOURCE_TYPE = "point_loss_gradient_responsibility_v1"
TRAIN_SPLITS = {"train", "training"}


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


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, *, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ValueError("numeric value is required")
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        if default is None:
            raise ValueError(f"numeric value expected, got {value!r}") from exc
        return float(default)
    if not math.isfinite(out):
        if default is None:
            raise ValueError(f"finite numeric value expected, got {value!r}")
        return float(default)
    return out


def _int_value(value: Any, *, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError("integer value is required")
        return int(default)
    try:
        return int(round(float(value)))
    except (TypeError, ValueError) as exc:
        if default is None:
            raise ValueError(f"integer value expected, got {value!r}") from exc
        return int(default)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _split(row: Mapping[str, Any]) -> str | None:
    value = row.get("split") or row.get("subset") or row.get("subset_name")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def _require_train(row: Mapping[str, Any], *, context: str) -> None:
    if _split(row) not in TRAIN_SPLITS:
        raise ValueError(f"{context}: split must be train/training")


def _samples_by_id(path: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line_no, row in enumerate(_read_jsonl(path), start=1):
        _require_train(row, context=f"{path}:{line_no}")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}:{line_no}: sample_id is required")
        if sample_id in out:
            raise ValueError(f"{path}:{line_no}: duplicate sample_id {sample_id}")
        out[sample_id] = row
    return out


def _segments(row: Mapping[str, Any], *, dense_len: int) -> list[tuple[float, float]]:
    raw = row.get("gt_segments") or row.get("segments") or []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError(f"{row.get('sample_id')}: gt_segments/segments must be a list")
    segments: list[tuple[float, float]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes, bytearray)) or len(item) < 2:
            raise ValueError(f"{row.get('sample_id')}: segment {idx} must be [start, end]")
        start = _finite_float(item[0])
        end = _finite_float(item[1])
        start = max(0.0, min(float(dense_len), start))
        end = max(0.0, min(float(dense_len), end))
        if end > start:
            segments.append((start, end))
    return segments


def _tiou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return 0.0 if union <= 0.0 else inter / union


def _boundary_role(center: int, segment: tuple[float, float] | None, *, radius: int) -> str:
    if segment is None:
        return "background"
    start, end = segment
    near_start = abs(float(center) - start) <= float(radius)
    near_end = abs(float(center) - end) <= float(radius)
    if near_start and near_end:
        return "full_segment"
    if near_start:
        return "start_boundary"
    if near_end:
        return "end_boundary"
    if start < float(center) < end:
        return "action_interior"
    return "context"


def _point_to_responsibility(
    point: Mapping[str, Any],
    *,
    dense_len: int,
    gt_segments: Sequence[tuple[float, float]],
    point_index: int,
    min_positive_iou: float,
    boundary_radius: int,
) -> dict[str, Any] | None:
    center = _int_value(point.get("point_index"), default=point_index)
    if center < 0 or center >= dense_len:
        return None
    score = _clip01(_finite_float(point.get("proposal_score") or point.get("classification_score"), default=0.0))
    segment_start = _finite_float(point.get("segment_start"), default=float(center))
    segment_end = _finite_float(point.get("segment_end"), default=float(center + 1))
    if segment_end < segment_start:
        segment_start, segment_end = segment_end, segment_start
    support_start = max(0, min(dense_len - 1, int(math.floor(segment_start))))
    support_end = max(0, min(dense_len - 1, int(math.ceil(segment_end))))
    if support_end < support_start:
        support_start = support_end = max(0, min(dense_len - 1, center))

    best_iou = 0.0
    best_gt_index: int | None = None
    for gt_index, (gt_start, gt_end) in enumerate(gt_segments):
        iou = _tiou(segment_start, segment_end, gt_start, gt_end)
        if iou > best_iou:
            best_iou = iou
            best_gt_index = gt_index

    matched = best_gt_index is not None and best_iou >= float(min_positive_iou)
    matched_segment = None if best_gt_index is None else gt_segments[best_gt_index]
    positive_gain = score * best_iou if matched else 0.0
    negative_risk = score * (1.0 - best_iou) if not matched else score * max(0.0, 0.5 - best_iou)
    cls_loss = (1.0 - score) if matched else score
    reg_loss = (1.0 - best_iou) if matched else best_iou
    quality_loss = abs(score - best_iou)
    grad_norm = max(1.0e-6, cls_loss + reg_loss + quality_loss)

    return {
        "true_time_center": center,
        "support_start": support_start,
        "support_end": support_end,
        "utility_source_type": UTILITY_SOURCE_TYPE,
        "positive_gain": _clip01(positive_gain),
        "negative_risk": _clip01(negative_risk),
        "cls_loss": _clip01(cls_loss),
        "reg_loss": _clip01(reg_loss),
        "quality_loss": _clip01(quality_loss),
        "grad_norm": float(grad_norm),
        "boundary_role": _boundary_role(center, matched_segment, radius=boundary_radius),
        "assigned_gt_id": None if best_gt_index is None else f"gt-{best_gt_index}",
        "teacher_point_index": point.get("point_index"),
        "teacher_proposal_score": score,
        "teacher_best_tiou": float(best_iou),
    }


def convert_teacher_points_to_responsibility(
    teacher_points_jsonl: str | Path,
    base_samples_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    manifest_json: str | Path | None = None,
    summary_json: str | Path | None = None,
    min_positive_iou: float = 0.3,
    boundary_radius: int = 4,
    max_points_per_sample: int = 512,
) -> dict[str, Any]:
    base = _samples_by_id(base_samples_jsonl)
    rows: list[dict[str, Any]] = []
    missing_base: list[str] = []
    zero_point_rows = 0
    for line_no, row in enumerate(_read_jsonl(teacher_points_jsonl), start=1):
        _require_train(row, context=f"{teacher_points_jsonl}:{line_no}")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{teacher_points_jsonl}:{line_no}: sample_id is required")
        base_row = base.get(sample_id)
        if base_row is None:
            missing_base.append(sample_id)
            continue
        dense_len = _int_value(row.get("dense_len") or base_row.get("dense_len"))
        if dense_len <= 0:
            raise ValueError(f"{sample_id}: dense_len must be positive")
        gt_segments = _segments(base_row, dense_len=dense_len)
        teacher_points = row.get("teacher_dense_points")
        if not isinstance(teacher_points, Sequence) or isinstance(teacher_points, (str, bytes, bytearray)):
            raise ValueError(f"{sample_id}: teacher_dense_points must be a list")
        ordered_points = list(teacher_points)[: max(0, int(max_points_per_sample))]
        responsibility_points = [
            converted
            for point_index, point in enumerate(ordered_points)
            if isinstance(point, Mapping)
            for converted in [
                _point_to_responsibility(
                    point,
                    dense_len=dense_len,
                    gt_segments=gt_segments,
                    point_index=point_index,
                    min_positive_iou=float(min_positive_iou),
                    boundary_radius=int(boundary_radius),
                )
            ]
            if converted is not None
        ]
        if not responsibility_points:
            zero_point_rows += 1
            # Keep one explicit low-utility point so downstream validators can
            # still account for the sample instead of silently dropping it.
            responsibility_points = [
                {
                    "true_time_center": 0,
                    "support_start": 0,
                    "support_end": 0,
                    "utility_source_type": UTILITY_SOURCE_TYPE,
                    "positive_gain": 0.0,
                    "negative_risk": 0.01,
                    "cls_loss": 0.01,
                    "reg_loss": 0.0,
                    "quality_loss": 0.01,
                    "grad_norm": 0.02,
                    "boundary_role": "background",
                    "assigned_gt_id": None,
                    "teacher_point_index": None,
                    "teacher_proposal_score": 0.0,
                    "teacher_best_tiou": 0.0,
                }
            ]
        rows.append(
            {
                "schema_version": ROW_SCHEMA_VERSION,
                "sample_id": sample_id,
                "split": "training",
                "dense_len": dense_len,
                "valid_len": _int_value(row.get("valid_len") or base_row.get("valid_len"), default=dense_len),
                "points": responsibility_points,
                "source_teacher_points_jsonl": str(Path(teacher_points_jsonl).expanduser()),
                "source_base_samples_jsonl": str(Path(base_samples_jsonl).expanduser()),
                "utility_source_type": UTILITY_SOURCE_TYPE,
                "training_only": True,
                "uses_gt_for_selection": False,
                "uses_val_or_test_gt_for_selection": False,
                "uses_teacher_at_deploy": False,
                "uses_prediction_cache_at_deploy": False,
                "uses_prediction_cache": False,
                "uses_raw_prediction": False,
                "load_from_raw_predictions": False,
                "teacher_train_only_forward": True,
                "proposal_score_surrogate_utility": False,
                "point_responsibility_utility": True,
            }
        )

    if missing_base:
        preview = ", ".join(missing_base[:5])
        raise ValueError(f"missing base samples for {len(missing_base)} teacher rows; first: {preview}")
    if not rows:
        raise ValueError("no responsibility rows produced")
    _write_jsonl(output_jsonl, rows)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "decision": READY,
        "split": "train",
        "utility_source_type": UTILITY_SOURCE_TYPE,
        "utility_construction": "train_only_dense_teacher_predictions_with_train_gt_tiou_point_loss_proxy",
        "teacher_points_jsonl": str(Path(teacher_points_jsonl).expanduser()),
        "teacher_points_jsonl_sha256": _sha256_file(teacher_points_jsonl),
        "base_samples_jsonl": str(Path(base_samples_jsonl).expanduser()),
        "base_samples_jsonl_sha256": _sha256_file(base_samples_jsonl),
        "output_jsonl": str(Path(output_jsonl).expanduser()),
        "output_jsonl_sha256": _sha256_file(output_jsonl),
        "row_count": len(rows),
        "zero_point_rows": zero_point_rows,
        "min_positive_iou": float(min_positive_iou),
        "boundary_radius": int(boundary_radius),
        "max_points_per_sample": int(max_points_per_sample),
        "uses_gt_for_selection": False,
        "uses_val_or_test_gt_for_selection": False,
        "uses_teacher_at_deploy": False,
        "uses_prediction_cache_at_deploy": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "load_from_raw_predictions": False,
        "training_only": True,
        "end_to_end": False,
    }
    if manifest_json is not None:
        _write_json(manifest_json, manifest)
    if summary_json is not None:
        _write_json(summary_json, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert train-only dense AdaTAD teacher points into point-responsibility source JSONL."
    )
    parser.add_argument("--teacher-points-jsonl", required=True)
    parser.add_argument("--base-samples-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--manifest-json")
    parser.add_argument("--summary-json")
    parser.add_argument("--min-positive-iou", type=float, default=0.3)
    parser.add_argument("--boundary-radius", type=int, default=4)
    parser.add_argument("--max-points-per-sample", type=int, default=512)
    args = parser.parse_args(argv)
    summary = convert_teacher_points_to_responsibility(
        args.teacher_points_jsonl,
        args.base_samples_jsonl,
        args.output_jsonl,
        manifest_json=args.manifest_json,
        summary_json=args.summary_json,
        min_positive_iou=args.min_positive_iou,
        boundary_radius=args.boundary_radius,
        max_points_per_sample=args.max_points_per_sample,
    )
    print(summary["decision"], flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
