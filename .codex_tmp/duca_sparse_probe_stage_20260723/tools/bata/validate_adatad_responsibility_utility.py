from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.bata import export_adatad_responsibility_utility as exporter


VALIDATION_READY = "ADATAD_RESPONSIBILITY_UTILITY_VALIDATION_PASS"
FORBIDDEN_TRUE_FLAGS = (
    "uses_val_or_test_gt_for_selection",
    "uses_gt_for_selection",
    "uses_teacher_at_deploy",
    "uses_prediction_cache_at_deploy",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "load_from_raw_predictions",
    "end_to_end",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence must be an object: {path}")
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


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_sha256(payload: Mapping[str, Any], key: str, *, context: str) -> None:
    value = payload.get(key)
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None, f"{context}: {key} must be sha256")


def _require_false_flags(payload: Mapping[str, Any], *, context: str) -> None:
    for key in FORBIDDEN_TRUE_FLAGS:
        if _is_true(payload.get(key, False)):
            raise ValueError(f"{context}: forbidden flag {key}=true")


def _require_float_list(payload: Mapping[str, Any], key: str, length: int, *, context: str) -> list[float]:
    values = payload.get(key)
    _require(isinstance(values, list), f"{context}: {key} must be a list")
    _require(len(values) == length, f"{context}: {key} length must equal dense_len")
    out: list[float] = []
    for index, value in enumerate(values):
        _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{context}: {key}[{index}] must be numeric")
        numeric = float(value)
        _require(math.isfinite(numeric), f"{context}: {key}[{index}] must be finite")
        out.append(numeric)
    return out


def _validate_row(row: Mapping[str, Any], *, line_no: int) -> None:
    context = f"responsibility utility row {line_no}"
    _require(row.get("schema_version") == exporter.ROW_SCHEMA_VERSION, f"{context}: schema_version mismatch")
    _require(row.get("utility_semantics") == exporter.UTILITY_SEMANTICS, f"{context}: utility_semantics mismatch")
    _require(row.get("utility_source_type") == exporter.UTILITY_SOURCE_TYPE, f"{context}: utility_source_type mismatch")
    _require(isinstance(row.get("sample_id"), str) and bool(str(row["sample_id"])), f"{context}: sample_id is required")
    dense_len = row.get("dense_len")
    _require(isinstance(dense_len, int) and dense_len > 0, f"{context}: dense_len must be positive int")
    _require(row.get("split") in {"train", "training"}, f"{context}: split must be train/training")
    positive = _require_float_list(row, "positive_observation_gain", dense_len, context=context)
    negative = _require_float_list(row, "negative_observation_risk", dense_len, context=context)
    signed = _require_float_list(row, "signed_frame_utility", dense_len, context=context)
    for index, (pos, neg, value) in enumerate(zip(positive, negative, signed, strict=True)):
        _require(0.0 <= pos <= 1.0, f"{context}: positive_observation_gain[{index}] outside [0,1]")
        _require(0.0 <= neg <= 1.0, f"{context}: negative_observation_risk[{index}] outside [0,1]")
        _require(abs(value - (pos - neg)) <= 1e-6, f"{context}: signed_frame_utility[{index}] must equal positive-negative")
    points = row.get("responsibility_points")
    _require(isinstance(points, list) and bool(points), f"{context}: responsibility_points is required")
    for point_index, point in enumerate(points):
        _require(isinstance(point, Mapping), f"{context}: responsibility_points[{point_index}] must be object")
        _require(point.get("utility_source_type") == exporter.UTILITY_SOURCE_TYPE, f"{context}: point utility_source_type mismatch")
    _require_false_flags(row, context=context)
    provenance = row.get("teacher_utility_provenance")
    if not isinstance(provenance, Mapping):
        provenance = (row.get("teacher_utility") or {}).get("provenance") if isinstance(row.get("teacher_utility"), Mapping) else None
    _require(
        row.get("training_only") is True or (isinstance(provenance, Mapping) and provenance.get("training_only") is True),
        f"{context}: training_only must be true in row or teacher_utility_provenance",
    )


def validate_responsibility_utility_export(
    summary_json: str | Path,
    *,
    output_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    summary_path = Path(summary_json).expanduser()
    summary = _read_json(summary_path)
    _require(summary.get("schema_version") == exporter.SUMMARY_SCHEMA_VERSION, "summary schema_version mismatch")
    _require(summary.get("decision") == exporter.READY, "summary decision mismatch")
    _require(summary.get("utility_semantics") == exporter.UTILITY_SEMANTICS, "summary utility_semantics mismatch")
    _require(summary.get("utility_source_type") == exporter.UTILITY_SOURCE_TYPE, "summary utility_source_type mismatch")
    _require(summary.get("split_scope") == "train_only", "summary split_scope must be train_only")
    _require_false_flags(summary, context="responsibility utility summary")
    _require(summary.get("training_only") is True, "summary training_only must be true")
    _require_sha256(summary, "source_jsonl_sha256", context="responsibility utility summary")
    _require_sha256(summary, "output_jsonl_sha256", context="responsibility utility summary")

    output_path = Path(output_jsonl or summary.get("output_jsonl", "")).expanduser()
    _require(output_path.is_file(), f"output_jsonl missing: {output_path}")
    actual_output_sha = _sha256_file(output_path)
    _require(actual_output_sha == str(summary["output_jsonl_sha256"]).lower(), "output_jsonl_sha256 mismatch")
    rows = _read_jsonl(output_path)
    _require(len(rows) == int(summary.get("row_count", -1)), "row_count mismatch")
    for line_no, row in enumerate(rows, start=1):
        _validate_row(row, line_no=line_no)

    evidence = dict(summary)
    evidence["decision"] = VALIDATION_READY
    evidence["summary_json"] = str(summary_path)
    evidence["summary_json_sha256"] = _sha256_file(summary_path)
    evidence["validated_output_jsonl"] = str(output_path)
    evidence["validated_output_jsonl_sha256"] = actual_output_sha
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate train-only AdaTAD point-responsibility utility export.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-jsonl")
    parser.add_argument("--write-evidence-json")
    args = parser.parse_args(argv)
    evidence = validate_responsibility_utility_export(args.summary_json, output_jsonl=args.output_jsonl)
    if args.write_evidence_json:
        out = Path(args.write_evidence_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(VALIDATION_READY, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
