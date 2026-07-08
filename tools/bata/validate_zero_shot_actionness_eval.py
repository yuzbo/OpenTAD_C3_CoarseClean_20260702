from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata import eval_zero_shot_actionness as actionness


SUMMARY_SCHEMA_VERSION = "zero_shot_actionness_eval_validation_v1"
READY = "ZERO_SHOT_ACTIONNESS_EVAL_VALIDATION_PASS"


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
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


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_true(value: Any) -> bool:
    return actionness._is_true(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_bool_or_none(value: Any, *, name: str) -> None:
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{name} must be bool or null")


def _validate_row(row: Mapping[str, Any], *, source_name: str) -> None:
    required = (
        "video_id",
        "window_id",
        "time_index",
        "original_time",
        "p_action",
        "logit",
        "valid",
        "source_name",
        "source_provenance",
        "prompt_hash",
        "checkpoint_hash",
        "thumos_trained",
        "uses_labels",
        "uses_teacher",
        "calibration_split",
    )
    for key in required:
        if key not in row:
            raise ValueError(f"{source_name}: missing required field {key}")
    if row.get("schema_version") not in {None, actionness.OUTPUT_SCHEMA_VERSION}:
        raise ValueError(f"{source_name}: schema_version mismatch")
    if not isinstance(row.get("video_id"), str) or not row["video_id"]:
        raise ValueError(f"{source_name}: video_id must be non-empty")
    if not isinstance(row.get("window_id"), str) or not row["window_id"]:
        raise ValueError(f"{source_name}: window_id must be non-empty")
    if isinstance(row.get("time_index"), bool) or not isinstance(row.get("time_index"), int):
        raise ValueError(f"{source_name}: time_index must be an integer")
    for key in ("original_time", "p_action", "logit"):
        if isinstance(row.get(key), bool) or not isinstance(row.get(key), (int, float)):
            raise ValueError(f"{source_name}: {key} must be numeric")
    if not 0.0 <= float(row["p_action"]) <= 1.0:
        raise ValueError(f"{source_name}: p_action must lie in [0,1]")
    if not isinstance(row.get("valid"), bool):
        raise ValueError(f"{source_name}: valid must be bool")
    if not isinstance(row.get("source_name"), str) or not row["source_name"]:
        raise ValueError(f"{source_name}: source_name must be non-empty")
    provenance = row.get("source_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{source_name}: source_provenance must be an object")
    if provenance.get("source_name") != row.get("source_name"):
        raise ValueError(f"{source_name}: source_provenance.source_name mismatch")
    for key in ("uses_labels", "uses_teacher"):
        if row.get(key) is not False:
            raise ValueError(f"{source_name}: {key} must be false")
        if provenance.get(key) is not False:
            raise ValueError(f"{source_name}: source_provenance.{key} must be false")
    for key in ("uses_gt", "uses_prediction_cache", "uses_raw_prediction"):
        if provenance.get(key) is not False:
            raise ValueError(f"{source_name}: source_provenance.{key} must be false")
    _validate_bool_or_none(row.get("thumos_trained"), name=f"{source_name}: thumos_trained")
    _validate_bool_or_none(provenance.get("thumos_trained"), name=f"{source_name}: source_provenance.thumos_trained")
    for key in actionness.FORBIDDEN_TRUE_FLAGS:
        if _is_true(row.get(key, False)):
            raise ValueError(f"{source_name}: forbidden flag {key}=true")
        if _is_true(provenance.get(key, False)):
            raise ValueError(f"{source_name}: forbidden source_provenance flag {key}=true")
    actionness._reject_source_leakage(
        {"source_provenance": provenance, "source_payload": row.get("source_payload", {})},
        source_name=source_name,
    )


def validate_eval(
    *,
    actionness_jsonl: str | Path,
    summary_json: str | Path | None = None,
    validation_json: str | Path | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(actionness_jsonl)
    seen: set[tuple[str, str]] = set()
    for line_no, row in enumerate(rows, start=1):
        _validate_row(row, source_name=f"{actionness_jsonl}:{line_no}")
        key = (str(row["video_id"]), str(row["window_id"]))
        if key in seen:
            raise ValueError(f"{actionness_jsonl}:{line_no}: duplicate video/window key {key}")
        seen.add(key)
    summary = _read_json(summary_json) if summary_json is not None else {}
    if summary:
        expected = summary.get("row_count")
        if expected is not None and int(expected) != len(rows):
            raise ValueError(f"{summary_json}: row_count mismatch")
        if "source_provenance" in summary:
            actionness._reject_source_leakage(
                {"source_provenance": summary["source_provenance"]},
                source_name=f"{summary_json}:source_provenance",
            )
        if summary.get("source_scoring_reads_gt") is True:
            raise ValueError(f"{summary_json}: source_scoring_reads_gt must not be true")
    result = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "actionness_jsonl": str(actionness_jsonl),
        "summary_json": None if summary_json is None else str(summary_json),
        "row_count": len(rows),
        "required_schema_fields_checked": True,
        "no_leak_scan_passed": True,
        "threshold_free_metrics_required": True,
    }
    if validation_json is not None:
        _write_json(validation_json, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate zero-shot actionness eval JSONL provenance and schema.")
    parser.add_argument("--actionness-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--validation-json")
    args = parser.parse_args(argv)
    result = validate_eval(
        actionness_jsonl=args.actionness_jsonl,
        summary_json=args.summary_json,
        validation_json=args.validation_json,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
