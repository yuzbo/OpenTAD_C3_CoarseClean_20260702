from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROW_SCHEMA_VERSION = "c3_adatad_responsibility_utility_row_v1"
SUMMARY_SCHEMA_VERSION = "c3_adatad_responsibility_utility_export_v1"
READY = "ADATAD_RESPONSIBILITY_UTILITY_EXPORT_READY"
UTILITY_SOURCE_TYPE = "point_loss_gradient_responsibility_v1"
UTILITY_SEMANTICS = "signed_point_responsibility_utility_v1"
FORBIDDEN_TRUE_FLAGS = (
    "uses_val_or_test_gt_for_selection",
    "uses_gt_for_selection",
    "uses_teacher_at_deploy",
    "uses_prediction_cache_at_deploy",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "load_from_raw_predictions",
)
TRAIN_SPLITS = {"train", "training"}


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON manifest must be an object: {path}")
    return payload


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


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _row_split(row: Mapping[str, Any]) -> str | None:
    for key in ("split", "subset", "subset_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _require_train_split(row: Mapping[str, Any], *, context: str) -> None:
    split = _row_split(row)
    if split not in TRAIN_SPLITS:
        raise ValueError(f"{context}: split must be train/training")


def _finite_float(value: Any, *, key: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{key} is required")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not math.isfinite(out):
        raise ValueError(f"{key} must be finite")
    return out


def _int_value(value: Any, *, key: str) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{key} is required")
    try:
        out = int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be integer-like") from exc
    return out


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _validate_manifest(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(manifest or {})
    split = str(payload.get("split", "")).strip().lower()
    if split != "train":
        raise ValueError("responsibility utility manifest requires split=train")
    for key in FORBIDDEN_TRUE_FLAGS:
        if _is_true(payload.get(key, False)):
            raise ValueError(f"responsibility utility manifest requires {key}=False")
    return {
        "split": "train",
        "uses_val_or_test_gt_for_selection": False,
        "uses_gt_for_selection": False,
        "uses_teacher_at_deploy": False,
        "uses_prediction_cache_at_deploy": False,
        "training_only": True,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "load_from_raw_predictions": False,
    }


def _base_samples_by_id(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    samples: dict[str, dict[str, Any]] = {}
    for line_no, row in enumerate(_read_jsonl(path), start=1):
        _require_train_split(row, context=f"base_samples_jsonl:{line_no}")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}:{line_no}: sample_id is required")
        action_target = row.get("action_target")
        if not isinstance(action_target, Sequence) or isinstance(action_target, (str, bytes, bytearray)):
            raise ValueError(f"{path}:{line_no}: action_target is required in base_samples_jsonl for Stage2 training")
        if sample_id in samples:
            raise ValueError(f"{path}:{line_no}: duplicate sample_id {sample_id}")
        samples[sample_id] = dict(row)
    return samples


def _apply_responsibility_points(row: Mapping[str, Any], *, line_no: int) -> dict[str, Any]:
    _require_train_split(row, context=f"source row {line_no}")
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"source row {line_no}: sample_id is required")
    dense_len = _int_value(row.get("dense_len"), key="dense_len")
    if dense_len <= 0:
        raise ValueError(f"{sample_id}: dense_len must be positive")
    points = row.get("points")
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes, bytearray)) or not points:
        raise ValueError(f"{sample_id}: points must be a non-empty list")

    positive = [0.0] * dense_len
    negative = [0.0] * dense_len
    normalized_points: list[dict[str, Any]] = []
    for point_index, point in enumerate(points):
        if not isinstance(point, Mapping):
            raise ValueError(f"{sample_id}: point {point_index} must be an object")
        source_type = point.get("utility_source_type")
        if source_type != UTILITY_SOURCE_TYPE:
            raise ValueError(f"{sample_id}: unsupported utility_source_type={source_type!r}; proposal_score_surrogate is not responsibility utility")
        support_start = _int_value(point.get("support_start"), key="support_start")
        support_end = _int_value(point.get("support_end"), key="support_end")
        if support_start < 0 or support_end < support_start or support_end >= dense_len:
            raise ValueError(f"{sample_id}: support_start/support_end out of dense_len bounds")
        pos = _clip01(_finite_float(point.get("positive_gain"), key="positive_gain"))
        neg = _clip01(_finite_float(point.get("negative_risk"), key="negative_risk"))
        for frame_index in range(support_start, support_end + 1):
            positive[frame_index] = max(positive[frame_index], pos)
            negative[frame_index] = max(negative[frame_index], neg)
        normalized_points.append(
            {
                "true_time_center": _int_value(point.get("true_time_center"), key="true_time_center"),
                "support_start": support_start,
                "support_end": support_end,
                "utility_source_type": UTILITY_SOURCE_TYPE,
                "positive_gain": pos,
                "negative_risk": neg,
                "cls_loss": _clip01(_finite_float(point.get("cls_loss", 0.0), key="cls_loss")),
                "reg_loss": _clip01(_finite_float(point.get("reg_loss", 0.0), key="reg_loss")),
                "quality_loss": _clip01(_finite_float(point.get("quality_loss", 0.0), key="quality_loss")),
                "grad_norm": max(0.0, _finite_float(point.get("grad_norm", 0.0), key="grad_norm")),
                "boundary_role": str(point.get("boundary_role", "")),
                "assigned_gt_id": point.get("assigned_gt_id"),
            }
        )

    signed = [_clip01(pos) - _clip01(neg) for pos, neg in zip(positive, negative, strict=True)]
    provenance = {
        "split_scope": "train_only",
        "utility_source_type": UTILITY_SOURCE_TYPE,
        "utility_semantics": UTILITY_SEMANTICS,
        "point_responsibility_utility": True,
        "proposal_score_surrogate_utility": False,
        "uses_val_or_test_gt_for_selection": False,
        "uses_gt_for_selection": False,
        "uses_teacher_at_deploy": False,
        "uses_prediction_cache_at_deploy": False,
        "training_only": True,
    }
    teacher_utility = {
        "utility_semantics": UTILITY_SEMANTICS,
        "utility_source_type": UTILITY_SOURCE_TYPE,
        "frame_utility": [max(0.0, value) for value in signed],
        "signed_frame_utility": signed,
        "positive_observation_gain": positive,
        "negative_observation_risk": negative,
        "provenance": dict(provenance),
    }
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "sample_id": sample_id,
        "split": "training",
        "dense_len": dense_len,
        "positive_observation_gain": positive,
        "negative_observation_risk": negative,
        "signed_frame_utility": signed,
        "frame_utility": [max(0.0, value) for value in signed],
        "teacher_utility": teacher_utility,
        "teacher_utility_provenance": provenance,
        "utility_semantics": UTILITY_SEMANTICS,
        "utility_source_type": UTILITY_SOURCE_TYPE,
        "responsibility_points": normalized_points,
        "uses_val_or_test_gt_for_selection": False,
        "uses_gt_for_selection": False,
        "uses_teacher_at_deploy": False,
        "uses_prediction_cache_at_deploy": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "load_from_raw_predictions": False,
        "training_only": True,
        "end_to_end": False,
    }


def run_export(
    source_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    summary_json: str | Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    base_samples_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    manifest_payload = _validate_manifest(manifest)
    source_path = Path(source_jsonl).expanduser()
    out_path = Path(output_jsonl).expanduser()
    base_samples = _base_samples_by_id(base_samples_jsonl)
    rows = []
    for line_no, row in enumerate(_read_jsonl(source_path), start=1):
        utility_row = _apply_responsibility_points(row, line_no=line_no)
        if base_samples:
            sample_id = utility_row["sample_id"]
            if sample_id not in base_samples:
                raise ValueError(f"{source_path}:{line_no}: sample_id {sample_id} missing from base_samples_jsonl")
            merged = dict(base_samples[sample_id])
            if int(merged.get("dense_len", utility_row["dense_len"])) != int(utility_row["dense_len"]):
                raise ValueError(f"{sample_id}: dense_len mismatch between responsibility points and base sample")
            merged.update(utility_row)
            merged.pop("training_only", None)
            rows.append(merged)
        else:
            rows.append(utility_row)
    _write_jsonl(out_path, rows)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "utility_semantics": UTILITY_SEMANTICS,
        "utility_source_type": UTILITY_SOURCE_TYPE,
        "split_scope": "train_only",
        "source_jsonl": str(source_path),
        "source_jsonl_sha256": _sha256_file(source_path),
        "base_samples_jsonl": None if base_samples_jsonl is None else str(Path(base_samples_jsonl).expanduser()),
        "base_samples_jsonl_sha256": None if base_samples_jsonl is None else _sha256_file(base_samples_jsonl),
        "output_jsonl": str(out_path),
        "output_jsonl_sha256": _sha256_file(out_path),
        "row_count": len(rows),
        "min_dense_len": min(int(row["dense_len"]) for row in rows),
        "max_dense_len": max(int(row["dense_len"]) for row in rows),
        **manifest_payload,
        "training_only": True,
        "end_to_end": False,
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export train-only AdaTAD point-responsibility utility on dense frame axis.")
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--manifest-json")
    parser.add_argument("--base-samples-jsonl")
    args = parser.parse_args(argv)
    manifest = _read_json(args.manifest_json) if args.manifest_json else None
    summary = run_export(
        args.source_jsonl,
        args.output_jsonl,
        summary_json=args.summary_json,
        manifest=manifest,
        base_samples_jsonl=args.base_samples_jsonl,
    )
    print(summary["decision"], flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
