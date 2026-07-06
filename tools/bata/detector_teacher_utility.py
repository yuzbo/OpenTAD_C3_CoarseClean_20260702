from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


ROW_SCHEMA_VERSION = "c3_detector_teacher_utility_row_v1"
SUMMARY_SCHEMA_VERSION = "c3_detector_teacher_utility_export_v1"
READY = "C3_DETECTOR_TEACHER_UTILITY_EXPORT_READY"
STAGE_LABEL = "Stage-2 detector-aware offline selector"
FORBIDDEN_TRUE_FLAGS = (
    "uses_gt",
    "uses_gt_for_selection",
    "uses_val_gt",
    "uses_test_gt",
    "uses_oracle",
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "prediction_uses_gt",
)
SPLIT_ALIASES = {
    "training": {"train", "training"},
    "train": {"train", "training"},
    "validation": {"val", "valid", "validation"},
    "val": {"val", "valid", "validation"},
    "test": {"test", "testing"},
}


JSONL_SCHEMA = {
    "schema_version": ROW_SCHEMA_VERSION,
    "required_keys": ["sample_id", "split", "dense_len", "valid_len", "frame_utility", "teacher_utility_provenance"],
    "frame_utility": "float list on local dense frame axis; padded positions >= valid_len are zero",
    "npz": "arrays sample_ids(str), splits(str), dense_lens(int64), valid_lens(int64), frame_utility(float32 padded)",
}


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
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _split(row: Mapping[str, Any]) -> str | None:
    for key in ("split", "subset", "subset_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _split_matches(actual: str | None, expected: str | None) -> bool:
    if expected is None:
        return True
    expected_key = str(expected).strip().lower()
    return actual in SPLIT_ALIASES.get(expected_key, {expected_key})


def _finite01(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return max(0.0, min(1.0, out))


def _point_index(point: Mapping[str, Any]) -> int | None:
    for key in ("point_index", "frame_index", "dense_index", "t"):
        if key in point:
            try:
                return int(round(float(point[key])))
            except (TypeError, ValueError):
                return None
    return None


def _point_utility(point: Mapping[str, Any]) -> float:
    if "utility" in point:
        return _finite01(point["utility"])
    if "proposal_score" in point:
        return _finite01(point["proposal_score"])
    cls = _finite01(point.get("classification_score", point.get("score", 0.0)))
    loc = _finite01(point.get("localization_quality", point.get("regression_quality", 1.0)))
    ctr = _finite01(point.get("centerness", 1.0))
    return cls * loc * ctr


def _normalize(values: Sequence[float], *, valid_len: int) -> list[float]:
    out = [float(item) for item in values]
    valid = out[: max(0, int(valid_len))]
    max_value = max(valid) if valid else 0.0
    if max_value > 0.0:
        out = [value / max_value for value in out]
    for idx in range(max(0, int(valid_len)), len(out)):
        out[idx] = 0.0
    return [max(0.0, min(1.0, float(item))) for item in out]


def map_dense_points_to_frame_utility(
    dense_points: Sequence[Mapping[str, Any]],
    *,
    dense_len: int,
    valid_len: int | None = None,
    spread_radius: int = 0,
) -> list[float]:
    """Map AdaTAD dense point/proposal signals to a dense-frame utility target.

    The mapping is train-only target construction. It consumes dense teacher outputs
    such as point classification score, localization/regression quality, centerness,
    or proposal_score, and never consumes validation/test GT.
    """
    dense_len = max(0, int(dense_len))
    valid_len = dense_len if valid_len is None else max(0, min(int(valid_len), dense_len))
    values = [0.0 for _ in range(dense_len)]
    radius = max(0, int(spread_radius))
    for point in dense_points:
        if not isinstance(point, Mapping):
            continue
        center = _point_index(point)
        if center is None or center < 0 or center >= valid_len:
            continue
        utility = _point_utility(point)
        for pos in range(max(0, center - radius), min(valid_len, center + radius + 1)):
            distance = abs(pos - center)
            weight = 1.0 if radius <= 0 else max(0.0, 1.0 - (0.5 * distance / float(radius + 1)))
            values[pos] = max(values[pos], utility * weight)
    return _normalize(values, valid_len=valid_len)


def _dense_points(row: Mapping[str, Any], *, line_no: int) -> list[Mapping[str, Any]]:
    for key in ("teacher_dense_points", "dense_teacher_points", "dense_points", "teacher_points"):
        value = row.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    raise ValueError(f"line {line_no}: teacher_dense_points are required")


def _validate_source_row(row: Mapping[str, Any], *, line_no: int, expected_split: str | None) -> None:
    actual = _split(row)
    if expected_split is not None and not _split_matches(actual, expected_split):
        raise ValueError(f"line {line_no}: expected split {expected_split}, got {actual or '<missing>'}")
    for key in FORBIDDEN_TRUE_FLAGS:
        if _is_true(row.get(key, False)):
            raise ValueError(f"line {line_no}: forbidden teacher source flag {key}=true")


def teacher_utility_row_from_dense_teacher(
    row: Mapping[str, Any],
    *,
    line_no: int = 1,
    expected_split: str | None = "training",
    spread_radius: int = 1,
) -> dict[str, Any]:
    _validate_source_row(row, line_no=line_no, expected_split=expected_split)
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"line {line_no}: sample_id is required")
    dense_len = int(row.get("dense_len") or row.get("valid_len") or 0)
    valid_len = int(row.get("valid_len") or dense_len)
    if dense_len <= 0 or valid_len <= 0 or valid_len > dense_len:
        raise ValueError(f"line {line_no}: dense_len/valid_len must be positive and consistent")
    frame_utility = map_dense_points_to_frame_utility(
        _dense_points(row, line_no=line_no),
        dense_len=dense_len,
        valid_len=valid_len,
        spread_radius=spread_radius,
    )
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "stage_label": STAGE_LABEL,
        "sample_id": sample_id,
        "split": _split(row) or "training",
        "dense_len": int(dense_len),
        "valid_len": int(valid_len),
        "frame_utility": frame_utility,
        "teacher_utility": {"frame_utility": frame_utility},
        "teacher_utility_provenance": {
            "teacher_signal_source": str(row.get("teacher_signal_source") or "adatad_dense_teacher"),
            "teacher_checkpoint_path": row.get("teacher_checkpoint_path"),
            "teacher_checkpoint_sha256": row.get("teacher_checkpoint_sha256"),
            "split_scope": "train_only",
            "uses_val_or_test_gt_for_selection": False,
            "uses_gt_for_selection": False,
            "export_schema": ROW_SCHEMA_VERSION,
        },
        "uses_gt": False,
        "uses_teacher": True,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "prediction_uses_gt": False,
        "training_only": True,
        "end_to_end": False,
    }


def _sample_map(rows: Sequence[Mapping[str, Any]], *, source_name: str) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for line_no, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{source_name}:{line_no}: sample_id is required")
        if sample_id in out:
            raise ValueError(f"{source_name}:{line_no}: duplicate sample_id {sample_id}")
        out[sample_id] = row
    return out


def _has_paction(row: Mapping[str, Any]) -> bool:
    frame_signals = row.get("frame_signals")
    return (
        isinstance(frame_signals, Mapping)
        and isinstance(frame_signals.get("p_action"), list)
        or isinstance(row.get("p_action"), list)
    )


def _merge_teacher_utility_into_base(
    *,
    utility_rows: Sequence[Mapping[str, Any]],
    base_samples_jsonl: str | Path,
    expected_split: str | None,
) -> list[dict[str, Any]]:
    utility_by_id = _sample_map(utility_rows, source_name="teacher_utility_rows")
    merged_rows: list[dict[str, Any]] = []
    for line_no, base in enumerate(_read_jsonl(base_samples_jsonl), start=1):
        _validate_source_row(base, line_no=line_no, expected_split=expected_split)
        sample_id = str(base["sample_id"])
        if sample_id not in utility_by_id:
            raise ValueError(f"{base_samples_jsonl}:{line_no}: missing teacher utility for sample_id {sample_id}")
        utility = utility_by_id[sample_id]
        base_dense_len = int(base.get("dense_len") or base.get("valid_len") or 0)
        base_valid_len = int(base.get("valid_len") or base_dense_len)
        if int(utility["dense_len"]) != base_dense_len or int(utility["valid_len"]) != base_valid_len:
            raise ValueError(f"{base_samples_jsonl}:{line_no}: dense_len/valid_len mismatch for sample_id {sample_id}")
        if not _has_paction(base):
            raise ValueError(f"{base_samples_jsonl}:{line_no}: base sample must include p_action for selector training")
        merged = dict(base)
        merged["schema_version"] = ROW_SCHEMA_VERSION
        merged["stage_label"] = STAGE_LABEL
        merged["frame_utility"] = list(utility["frame_utility"])
        merged["teacher_utility"] = dict(utility["teacher_utility"])
        merged["teacher_utility_provenance"] = dict(utility["teacher_utility_provenance"])
        merged["uses_teacher"] = True
        merged["training_only"] = True
        merged["end_to_end"] = False
        merged_rows.append(merged)
    if set(utility_by_id) != {str(row["sample_id"]) for row in merged_rows}:
        extra = sorted(set(utility_by_id) - {str(row["sample_id"]) for row in merged_rows})
        if extra:
            raise ValueError(f"teacher utility rows have no matching base sample: {extra[:3]}")
    return merged_rows


def _write_npz(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import numpy as np

    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_len = max(int(row["dense_len"]) for row in rows)
    utility = np.zeros((len(rows), max_len), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        values = [float(item) for item in row["frame_utility"]]
        utility[row_idx, : len(values)] = np.asarray(values, dtype=np.float32)
    np.savez_compressed(
        out_path,
        sample_ids=np.asarray([str(row["sample_id"]) for row in rows]),
        splits=np.asarray([str(row["split"]) for row in rows]),
        dense_lens=np.asarray([int(row["dense_len"]) for row in rows], dtype=np.int64),
        valid_lens=np.asarray([int(row["valid_len"]) for row in rows], dtype=np.int64),
        frame_utility=utility,
        schema_version=np.asarray([ROW_SCHEMA_VERSION]),
    )


def run_export(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    summary_json: str | Path | None = None,
    output_npz: str | Path | None = None,
    base_samples_jsonl: str | Path | None = None,
    teacher_checkpoint_path: str | Path | None = None,
    teacher_config_path: str | Path | None = None,
    expected_split: str | None = "training",
    spread_radius: int = 1,
) -> dict[str, Any]:
    source_rows = _read_jsonl(input_jsonl)
    out_rows = [
        teacher_utility_row_from_dense_teacher(
            row,
            line_no=line_no,
            expected_split=expected_split,
            spread_radius=int(spread_radius),
        )
        for line_no, row in enumerate(source_rows, start=1)
    ]
    if base_samples_jsonl is not None:
        out_rows = _merge_teacher_utility_into_base(
            utility_rows=out_rows,
            base_samples_jsonl=base_samples_jsonl,
            expected_split=expected_split,
        )
    _write_jsonl(output_jsonl, out_rows)
    if output_npz is not None:
        _write_npz(output_npz, out_rows)
    checkpoint_sha256 = None if teacher_checkpoint_path is None else _sha256_file(teacher_checkpoint_path)
    config_sha256 = None if teacher_config_path is None else _sha256_file(teacher_config_path)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "stage_label": STAGE_LABEL,
        "route_label": "DIVERGENT_INNOVATION_DETECTOR_AWARE_UTILITY_DO_NOT_MERGE_WITH_C3",
        "teacher_signal_source": "adatad_dense_teacher",
        "split_scope": "train_only",
        "teacher_checkpoint_path": None if teacher_checkpoint_path is None else str(teacher_checkpoint_path),
        "teacher_checkpoint_sha256": checkpoint_sha256,
        "teacher_config_path": None if teacher_config_path is None else str(teacher_config_path),
        "teacher_config_sha256": config_sha256,
        "input_jsonl": str(input_jsonl),
        "base_samples_jsonl": None if base_samples_jsonl is None else str(base_samples_jsonl),
        "output_jsonl": str(output_jsonl),
        "output_npz": None if output_npz is None else str(output_npz),
        "row_count": len(out_rows),
        "jsonl_row_schema": JSONL_SCHEMA,
        "expected_split": expected_split,
        "split_scope": "train_only",
        "uses_val_or_test_gt_for_selection": False,
        "uses_gt_for_selection": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "load_from_raw_predictions": False,
        "end_to_end": False,
        "input_jsonl_sha256": _sha256_file(input_jsonl),
        "base_samples_jsonl_sha256": None if base_samples_jsonl is None else _sha256_file(base_samples_jsonl),
        "output_jsonl_sha256": _sha256_file(output_jsonl),
        "git_sha": _git_sha(),
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    return payload


def validate_teacher_utility_export_evidence(
    summary_json: str | Path,
    *,
    output_jsonl: str | Path | None = None,
    require_paction: bool = False,
) -> dict[str, Any]:
    summary = _read_json(summary_json)
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError("teacher utility evidence schema_version mismatch")
    if summary.get("decision") != READY:
        raise ValueError("teacher utility export decision is not ready")
    if summary.get("teacher_signal_source") != "adatad_dense_teacher":
        raise ValueError("teacher_signal_source must be adatad_dense_teacher")
    if summary.get("split_scope") != "train_only":
        raise ValueError("split_scope must be train_only")
    for key in ("uses_val_or_test_gt_for_selection", "uses_gt_for_selection", "uses_prediction_cache", "uses_raw_prediction", "load_from_raw_predictions", "end_to_end"):
        if summary.get(key) is not False:
            raise ValueError(f"teacher utility evidence must set {key}=false")
    for key in ("teacher_checkpoint_path", "teacher_checkpoint_sha256", "teacher_config_path", "teacher_config_sha256"):
        if not isinstance(summary.get(key), str) or not summary[key]:
            raise ValueError(f"teacher utility evidence requires {key}")
    output_path = Path(output_jsonl or summary.get("output_jsonl", "")).expanduser()
    if not output_path.is_file():
        raise ValueError(f"teacher utility output_jsonl missing: {output_path}")
    actual_output_sha = _sha256_file(output_path)
    if summary.get("output_jsonl_sha256") != actual_output_sha:
        raise ValueError("output_jsonl_sha256 mismatch")
    rows = _read_jsonl(output_path)
    if int(summary.get("row_count", -1)) != len(rows):
        raise ValueError("teacher utility row_count mismatch")
    for line_no, row in enumerate(rows, start=1):
        if row.get("schema_version") != ROW_SCHEMA_VERSION:
            raise ValueError(f"{output_path}:{line_no}: teacher utility row schema mismatch")
        if not _split_matches(_split(row), "training"):
            raise ValueError(f"{output_path}:{line_no}: teacher utility rows must be training split")
        provenance = row.get("teacher_utility_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{output_path}:{line_no}: missing teacher_utility_provenance")
        if provenance.get("teacher_signal_source") != "adatad_dense_teacher":
            raise ValueError(f"{output_path}:{line_no}: teacher_signal_source must be adatad_dense_teacher")
        if provenance.get("split_scope") != "train_only":
            raise ValueError(f"{output_path}:{line_no}: split_scope must be train_only")
        if row.get("uses_teacher") is not True or row.get("training_only") is not True:
            raise ValueError(f"{output_path}:{line_no}: teacher utility rows must be training_only teacher artifacts")
        for key in FORBIDDEN_TRUE_FLAGS:
            if _is_true(row.get(key, False)):
                raise ValueError(f"{output_path}:{line_no}: forbidden teacher utility row flag {key}=true")
        if require_paction and not _has_paction(row):
            raise ValueError(f"{output_path}:{line_no}: p_action is required for selector training evidence")
    evidence = dict(summary)
    evidence["decision"] = "C3_DETECTOR_TEACHER_UTILITY_EVIDENCE_PASS"
    evidence["summary_json"] = str(summary_json)
    evidence["validated_output_jsonl"] = str(output_path)
    evidence["validated_output_jsonl_sha256"] = actual_output_sha
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export train-only AdaTAD dense teacher utility to JSONL/NPZ.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--output-npz")
    parser.add_argument("--base-samples-jsonl")
    parser.add_argument("--teacher-checkpoint-path")
    parser.add_argument("--teacher-config-path")
    parser.add_argument("--expected-split", default="training")
    parser.add_argument("--spread-radius", type=int, default=1)
    args = parser.parse_args(argv)
    summary = run_export(
        args.input_jsonl,
        args.output_jsonl,
        summary_json=args.summary_json,
        output_npz=args.output_npz,
        base_samples_jsonl=args.base_samples_jsonl,
        teacher_checkpoint_path=args.teacher_checkpoint_path,
        teacher_config_path=args.teacher_config_path,
        expected_split=args.expected_split,
        spread_radius=int(args.spread_radius),
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
