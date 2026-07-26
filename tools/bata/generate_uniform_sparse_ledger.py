from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


OUTPUT_SCHEMA_VERSION = "c3_uniform_sparse_ledger_v1"
SUMMARY_SCHEMA_VERSION = "c3_uniform_sparse_ledger_summary_v1"
READY = "C3_UNIFORM_EXACT_LEDGER_READY"
NO_GO = "C3_UNIFORM_EXACT_LEDGER_NO_GO"
SELECTION_FAMILY = "uniform_exact"
FORBIDDEN_TRUE_FLAGS = (
    "uses_gt",
    "uses_teacher",
    "uses_oracle",
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "uses_checkpoint",
    "prediction_uses_gt",
    "training_only",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _dedupe_identical_sample_ids(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduped: list[dict[str, Any]] = []
    seen: dict[str, Mapping[str, Any]] = {}
    duplicate_count = 0
    for row in rows:
        row_dict = dict(row)
        sample_id = row_dict.get("sample_id")
        if isinstance(sample_id, str) and sample_id:
            previous = seen.get(sample_id)
            if previous is not None:
                if dict(previous) != row_dict:
                    raise ValueError(f"duplicate sample_id has non-identical source rows: {sample_id}")
                duplicate_count += 1
                continue
            seen[sample_id] = row_dict
        deduped.append(row_dict)
    return deduped, duplicate_count


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text and (text.isdigit() or (text[0] in "+-" and text[1:].isdigit())):
            return int(text)
    raise ValueError(f"{name} must be an integer")


def _optional_int(row: Mapping[str, Any], key: str) -> int | None:
    if key not in row or row[key] is None:
        return None
    return _strict_int(row[key], name=key)


def _valid_mask_positions(mask: Any, *, dense_len: int | None, line_no: int) -> list[int]:
    if not isinstance(mask, list):
        raise ValueError(f"line {line_no}: valid_mask must be a JSON list")
    if dense_len is not None and len(mask) != int(dense_len):
        raise ValueError(f"line {line_no}: valid_mask length must equal dense_len")
    positions: list[int] = []
    seen_false = False
    for idx, item in enumerate(mask):
        value = _is_true(item)
        if value and seen_false:
            raise ValueError(f"line {line_no}: valid_mask must be a contiguous true prefix")
        if value:
            positions.append(idx)
        else:
            seen_false = True
    if not positions:
        raise ValueError(f"line {line_no}: valid_mask has no valid positions")
    return positions


def _row_valid_positions(row: Mapping[str, Any], *, line_no: int) -> tuple[int, int, list[int]]:
    dense_len = _optional_int(row, "dense_len")
    valid_len = _optional_int(row, "valid_len")
    valid_positions: list[int] | None = None
    if "valid_mask" in row and row["valid_mask"] is not None:
        valid_positions = _valid_mask_positions(row["valid_mask"], dense_len=dense_len, line_no=line_no)
        if dense_len is None:
            dense_len = len(row["valid_mask"])
        if valid_len is not None and valid_len != len(valid_positions):
            raise ValueError(f"line {line_no}: valid_len does not match valid_mask true count")
        valid_len = len(valid_positions)
    if valid_len is None:
        valid_len = dense_len
    if valid_len is None or valid_len <= 0:
        raise ValueError(f"line {line_no}: valid_len or valid_mask must define a positive valid span")
    if dense_len is None:
        dense_len = valid_len
    if dense_len <= 0:
        raise ValueError(f"line {line_no}: dense_len must be positive")
    if valid_len > dense_len:
        raise ValueError(f"line {line_no}: valid_len cannot exceed dense_len")
    if valid_positions is None:
        valid_positions = list(range(valid_len))
    return int(dense_len), int(valid_len), valid_positions


def uniform_positions(*, valid_len: int, target_len: int) -> list[int]:
    valid_len = int(valid_len)
    target_len = int(target_len)
    if valid_len <= 0:
        raise ValueError("valid_len must be positive")
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    count = min(valid_len, target_len)
    if count == valid_len:
        return list(range(valid_len))
    if count == 1:
        return [0]
    return [int(round(rank * (valid_len - 1) / float(count - 1))) for rank in range(count)]


def _select_from_valid_positions(valid_positions: Sequence[int], *, target_len: int) -> list[int]:
    ranks = uniform_positions(valid_len=len(valid_positions), target_len=int(target_len))
    return [int(valid_positions[rank]) for rank in ranks]


def _count_histogram(values: Sequence[int]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: int(item[0])))


def _validate_selected_positions(
    selected: Sequence[int],
    *,
    valid_positions: Sequence[int],
    line_no: int,
) -> None:
    selected_list = [int(item) for item in selected]
    if selected_list != sorted(selected_list):
        raise ValueError(f"line {line_no}: selected_positions must be sorted")
    if len(set(selected_list)) != len(selected_list):
        raise ValueError(f"line {line_no}: selected_positions must be unique")
    valid_set = set(int(item) for item in valid_positions)
    if any(item not in valid_set for item in selected_list):
        raise ValueError(f"line {line_no}: selected_positions must stay within valid range")


def source_row_to_ledger_row(
    row: Mapping[str, Any],
    *,
    line_no: int,
    target_len: int,
    allow_short_valid: bool = False,
) -> dict[str, Any]:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"line {line_no}: sample_id must be a non-empty string")
    for key in FORBIDDEN_TRUE_FLAGS:
        if _is_true(row.get(key, False)):
            raise ValueError(f"line {line_no}: forbidden source flag {key}=true")
    dense_len, valid_len, valid_positions = _row_valid_positions(row, line_no=line_no)
    if valid_len < int(target_len) and not allow_short_valid:
        raise ValueError(
            f"line {line_no}: short valid_len={valid_len} is below target_len={int(target_len)}; "
            "use --allow-short-valid only for explicit non-paper-main diagnostics"
        )
    selected = _select_from_valid_positions(valid_positions, target_len=int(target_len))
    _validate_selected_positions(selected, valid_positions=valid_positions, line_no=line_no)
    ledger_row: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "sample_id": sample_id,
        "dense_len": int(dense_len),
        "valid_len": int(valid_len),
        "selected_positions_unit": "local_dense_index",
        "selected_positions": selected,
        "selected_count": len(selected),
        "target_len": int(target_len),
        "selection_family": SELECTION_FAMILY,
        "uses_uniform_scaffold": True,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_prediction_cache": False,
        "deploy_selection_ledger": True,
    }
    if isinstance(row.get("video_name"), str) and row["video_name"]:
        ledger_row["video_name"] = row["video_name"]
    return ledger_row


def run_generation(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    summary_json: str | Path | None = None,
    target_len: int = 384,
    allow_short_valid: bool = False,
    allow_duplicate_identical_sample_id: bool = False,
) -> dict[str, Any]:
    target_len = _strict_int(target_len, name="target_len")
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    source_rows = _read_jsonl(input_jsonl)
    source_row_count = len(source_rows)
    duplicate_identical_count = 0
    if allow_duplicate_identical_sample_id:
        source_rows, duplicate_identical_count = _dedupe_identical_sample_ids(source_rows)
    ledger_rows = [
        source_row_to_ledger_row(
            row,
            line_no=line_no,
            target_len=target_len,
            allow_short_valid=bool(allow_short_valid),
        )
        for line_no, row in enumerate(source_rows, start=1)
    ]
    sample_ids = [str(row["sample_id"]) for row in ledger_rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("uniform sparse ledger has duplicate sample_id")
    _write_jsonl(output_jsonl, ledger_rows)
    selected_counts = [int(row["selected_count"]) for row in ledger_rows]
    short_valid_count = sum(1 for row in ledger_rows if int(row["valid_len"]) < target_len)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "source_row_count": int(source_row_count),
        "row_count": len(ledger_rows),
        "target_len": int(target_len),
        "selection_family": SELECTION_FAMILY,
        "allow_short_valid": bool(allow_short_valid),
        "allow_duplicate_identical_sample_id": bool(allow_duplicate_identical_sample_id),
        "duplicate_identical_sample_id_count": int(duplicate_identical_count),
        "short_valid_count": int(short_valid_count),
        "selected_count_histogram": _count_histogram(selected_counts),
        "min_selected_count": min(selected_counts),
        "max_selected_count": max(selected_counts),
        "uses_uniform_scaffold": True,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_prediction_cache": False,
        "deploy_selection_ledger": True,
        "source_sha256": sha256_file(input_jsonl),
        "ledger_sha256": sha256_file(output_jsonl),
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate strict exact uniform sparse C3 value-transport ledger rows.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--target-len", type=int, default=384)
    parser.add_argument("--allow-short-valid", action="store_true")
    parser.add_argument("--allow-duplicate-identical-sample-id", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = run_generation(
            args.input_jsonl,
            args.output_jsonl,
            summary_json=args.summary_json,
            target_len=int(args.target_len),
            allow_short_valid=bool(args.allow_short_valid),
            allow_duplicate_identical_sample_id=bool(args.allow_duplicate_identical_sample_id),
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps({"schema_version": SUMMARY_SCHEMA_VERSION, "decision": NO_GO, "error": str(exc)}), flush=True)
        return 1
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
