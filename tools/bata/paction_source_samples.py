from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SELECTION_RELEVANT_KEYS = (
    "sample_id",
    "video_name",
    "window_start_frame",
    "dense_len",
    "valid_len",
    "frame_signals",
    "p_action",
    "strategy_selected_positions",
    "action_target",
    "gt_boundaries",
    "gt_segments",
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: sample row must be a JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_row_signature(row: Mapping[str, Any]) -> str:
    payload = {key: row.get(key) for key in SELECTION_RELEVANT_KEYS if key in row}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonicalize_unique_sample_jsonl(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    report_json: str | Path | None = None,
    split: str = "",
    allow_identical_drop: bool = True,
) -> dict[str, Any]:
    rows = _read_jsonl(input_jsonl)
    seen: dict[str, tuple[str, int]] = {}
    out_rows: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    for line_no, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{input_jsonl}:{line_no}: sample row is missing sample_id")
        signature = canonical_row_signature(row)
        if sample_id in seen:
            old_signature, old_line = seen[sample_id]
            item = {
                "sample_id": sample_id,
                "first_line": int(old_line),
                "duplicate_line": int(line_no),
                "identical": bool(signature == old_signature),
            }
            duplicates.append(item)
            if signature == old_signature and allow_identical_drop:
                continue
            conflicts.append(item)
            raise ValueError(
                f"{input_jsonl}:{line_no}: conflicting duplicate sample_id={sample_id}; "
                f"first_line={old_line}"
            )
        seen[sample_id] = (signature, line_no)
        out_rows.append(dict(row))

    _write_jsonl(output_jsonl, out_rows)
    report = {
        "schema_version": "c3_paction_source_sample_canonicalization_v1",
        "split": str(split),
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "input_rows": len(rows),
        "output_rows": len(out_rows),
        "unique_sample_ids": len(seen),
        "duplicate_count": len(duplicates),
        "conflicting_duplicate_count": len(conflicts),
        "duplicates": duplicates,
    }
    if report_json is not None:
        _write_json(report_json, report)
    return report
