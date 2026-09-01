from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata import eval_zero_shot_actionness as actionness_eval
from tools.bata import run_zero_shot_actionness_selection_eval as selection_eval


SUMMARY_SCHEMA_VERSION = "zero_shot_actionness_selection_validation_v1"
READY = "ZERO_SHOT_ACTIONNESS_SELECTION_VALIDATION_PASS"


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
    return actionness_eval._is_true(value)


def _positions(row: Mapping[str, Any], *, source_name: str) -> list[int]:
    raw = row.get("selected_positions")
    if not isinstance(raw, list):
        raise ValueError(f"{source_name}: selected_positions must be a list")
    selected = [int(item) for item in raw]
    if selected != sorted(selected):
        raise ValueError(f"{source_name}: selected_positions must be sorted")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{source_name}: selected_positions must be unique")
    if any(item < 0 for item in selected):
        raise ValueError(f"{source_name}: selected_positions must be non-negative")
    return selected


def _validate_row(row: Mapping[str, Any], *, source_name: str) -> None:
    required = (
        "video_id",
        "baseline",
        "valid_len",
        "budget",
        "selected_positions",
        "selected_count",
        "ledger_role",
        "diagnostic_only",
        "uses_gt_for_selection",
        "uses_labels",
        "uses_teacher",
        "uses_raw_prediction",
        "budget_violation",
    )
    for key in required:
        if key not in row:
            raise ValueError(f"{source_name}: missing required field {key}")
    if row.get("schema_version") not in {None, selection_eval.AUDIT_SCHEMA_VERSION}:
        raise ValueError(f"{source_name}: schema_version mismatch")
    if row.get("ledger_role") != selection_eval.LEDGER_ROLE:
        raise ValueError(f"{source_name}: ledger_role must be {selection_eval.LEDGER_ROLE}")
    if not isinstance(row.get("video_id"), str) or not row["video_id"]:
        raise ValueError(f"{source_name}: video_id must be non-empty")
    baseline = str(row.get("baseline"))
    if baseline not in selection_eval.DEFAULT_BASELINES:
        raise ValueError(f"{source_name}: unknown baseline {baseline}")
    selected = _positions(row, source_name=source_name)
    valid_len = int(row.get("valid_len"))
    budget = int(row.get("budget"))
    if valid_len <= 0 or budget <= 0:
        raise ValueError(f"{source_name}: valid_len and budget must be positive")
    if selected and selected[-1] >= valid_len:
        raise ValueError(f"{source_name}: selected_positions exceed valid_len")
    if int(row.get("selected_count")) != len(selected):
        raise ValueError(f"{source_name}: selected_count mismatch")
    if len(selected) > budget or bool(row.get("budget_violation")):
        raise ValueError(f"{source_name}: selected positions violate budget")
    for key in selection_eval.FORBIDDEN_TRUE_FLAGS:
        if _is_true(row.get(key, False)):
            raise ValueError(f"{source_name}: forbidden flag {key}=true")
    if row.get("uses_labels") is not False:
        raise ValueError(f"{source_name}: uses_labels must be false")
    if baseline == "oracle-actionness":
        if row.get("diagnostic_only") is not True or row.get("uses_gt_for_selection") is not True:
            raise ValueError(f"{source_name}: oracle-actionness must be diagnostic_only with uses_gt_for_selection=true")
    else:
        if row.get("diagnostic_only") is not False:
            raise ValueError(f"{source_name}: deployable baseline {baseline} must not be diagnostic_only")
        if row.get("uses_gt_for_selection") is not False:
            raise ValueError(f"{source_name}: deployable baseline {baseline} must not use GT for selection")
    actionness_eval._reject_source_leakage(
        {
            "source_payload": row.get("source_payload", {}),
            "source_provenance": row.get("source_provenance", {}),
        },
        source_name=source_name,
    )
    selection_eval.validate_sparse_temporal_grid_row(row)


def validate_selection_eval(
    *,
    audit_jsonl: str | Path,
    summary_json: str | Path | None = None,
    validation_json: str | Path | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(audit_jsonl)
    seen: set[tuple[str, str]] = set()
    for line_no, row in enumerate(rows, start=1):
        _validate_row(row, source_name=f"{audit_jsonl}:{line_no}")
        key = (str(row["baseline"]), str(row["video_id"]))
        if key in seen:
            raise ValueError(f"{audit_jsonl}:{line_no}: duplicate baseline/video row {key}")
        seen.add(key)
    summary = _read_json(summary_json) if summary_json is not None else {}
    if summary:
        expected = summary.get("row_count")
        if expected is not None and int(expected) != len(rows):
            raise ValueError(f"{summary_json}: row_count mismatch")
        deployable = summary.get("deployable_claim_baselines", [])
        if "oracle-actionness" in deployable:
            raise ValueError(f"{summary_json}: oracle-actionness must not enter deployable_claim_baselines")
        if summary.get("oracle_is_diagnostic_only") is False:
            raise ValueError(f"{summary_json}: oracle_is_diagnostic_only must not be false")
    result = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "audit_jsonl": str(audit_jsonl),
        "summary_json": None if summary_json is None else str(summary_json),
        "row_count": len(rows),
        "ledger_role": selection_eval.LEDGER_ROLE,
        "oracle_diagnostic_only_verified": True,
        "sparse_grid_validation_passed": True,
        "no_leak_scan_passed": True,
    }
    if validation_json is not None:
        _write_json(validation_json, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate zero-shot actionness selection audit artifacts.")
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--validation-json")
    args = parser.parse_args(argv)
    result = validate_selection_eval(
        audit_jsonl=args.audit_jsonl,
        summary_json=args.summary_json,
        validation_json=args.validation_json,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
