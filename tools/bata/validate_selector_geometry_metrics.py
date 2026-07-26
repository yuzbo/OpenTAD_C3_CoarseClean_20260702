from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "selector_geometry_metrics_validation_v1"
PASS = "SELECTOR_GEOMETRY_VALIDATION_PASS"
FAIL = "SELECTOR_GEOMETRY_VALIDATION_FAIL"


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise ValueError(f"required CSV does not exist: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _falsey(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    text = str(value).strip().lower()
    return text in {"0", "false", "no", "n", "f"}


def _has_any_field(fieldnames: set[str], aliases: Sequence[str]) -> bool:
    return any(name in fieldnames for name in aliases)


def _first_present(row: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for name in aliases:
        if name in row and row.get(name) not in (None, ""):
            return row.get(name)
    return None


def _validate_selected_frame_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_no_padding_selected: bool,
) -> dict[str, Any]:
    fieldnames = {key for row in rows for key in row}
    if not rows:
        raise AssertionError("selected_frame_metrics has no rows")
    if not _has_any_field(fieldnames, ("selected_count", "num_selected", "selected_frames")):
        raise AssertionError("selected_frame_metrics missing required selected_count field")
    if not _has_any_field(
        fieldnames,
        ("required_selected_count", "require_selected_count", "target_len", "expected_selected_count"),
    ):
        raise AssertionError("selected_frame_metrics missing required selected_count target field")

    invalid_rows: list[dict[str, Any]] = []
    padding_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        selected_value = _first_present(row, ("selected", "is_selected", "selected_flag"))
        selected = True if selected_value is None else _truthy(selected_value)
        invalid_selected = _truthy(_first_present(row, ("invalid_selected", "selected_invalid", "out_of_range_selected")))
        valid_value = _first_present(row, ("is_valid", "valid", "within_valid_range"))
        selected_but_invalid_valid_flag = selected and valid_value is not None and _falsey(valid_value)
        if selected and (invalid_selected or selected_but_invalid_valid_flag):
            invalid_rows.append({"row_index": index, **dict(row)})
        if selected and _truthy(_first_present(row, ("is_padding", "padding", "padding_selected"))):
            padding_rows.append({"row_index": index, **dict(row)})

    if invalid_rows:
        raise AssertionError(f"selected_frame_metrics contains invalid selected rows: {len(invalid_rows)} invalid selected")
    if require_no_padding_selected and padding_rows:
        raise AssertionError(f"selected_frame_metrics contains padding selected rows: {len(padding_rows)}")
    return {
        "row_count": len(rows),
        "has_selected_count": True,
        "has_required_selected_count": True,
        "invalid_selected_count": len(invalid_rows),
        "padding_selected_count": len(padding_rows),
    }


def _validate_action_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fieldnames = {key for row in rows for key in row}
    if not rows:
        raise AssertionError("action_summary has no rows")
    has_start = _has_any_field(fieldnames, ("start_endpoint_coverage", "start_coverage", "start_covered"))
    has_end = _has_any_field(fieldnames, ("end_endpoint_coverage", "end_coverage", "end_covered"))
    has_both = _has_any_field(
        fieldnames,
        ("both_endpoint_coverage", "both_endpoint_covered", "both_coverage", "boundary_recall_both"),
    )
    if not (has_start and has_end and has_both):
        raise AssertionError("action_summary must contain start/end/both endpoint coverage fields")
    return {
        "row_count": len(rows),
        "has_start_endpoint_coverage": has_start,
        "has_end_endpoint_coverage": has_end,
        "has_both_endpoint_coverage": has_both,
    }


def validate_geometry_dir(
    geometry_dir: str | Path,
    *,
    require_no_padding_selected: bool = False,
    require_coordinate_contract: bool = False,
) -> dict[str, Any]:
    geometry_path = Path(geometry_dir).expanduser()
    manifest = _read_json(geometry_path / "manifest.json")
    coordinate_contract = manifest.get("coordinate_contract")
    coordinate_contract_present = isinstance(coordinate_contract, Mapping) and bool(coordinate_contract)
    if require_coordinate_contract and not coordinate_contract_present:
        raise AssertionError("manifest missing coordinate_contract")

    selected_report = _validate_selected_frame_metrics(
        _read_csv(geometry_path / "selected_frame_metrics.csv"),
        require_no_padding_selected=require_no_padding_selected,
    )
    action_report = _validate_action_summary(_read_csv(geometry_path / "action_summary.csv"))
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": PASS,
        "geometry_dir": str(geometry_path),
        "coordinate_contract_present": coordinate_contract_present,
        "coordinate_contract": coordinate_contract if coordinate_contract_present else None,
        "selected_frame_metrics": selected_report,
        "action_summary": action_report,
        "requirements": {
            "require_no_padding_selected": bool(require_no_padding_selected),
            "require_coordinate_contract": bool(require_coordinate_contract),
        },
    }


def _failure_report(error: BaseException, geometry_dir: str | Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": FAIL,
        "geometry_dir": str(geometry_dir),
        "error": str(error),
        "error_type": type(error).__name__,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate selector geometry analyzer CSV/JSON metrics.")
    parser.add_argument("--geometry-dir", required=True)
    parser.add_argument("--require-no-padding-selected", action="store_true")
    parser.add_argument("--require-coordinate-contract", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        report = validate_geometry_dir(
            args.geometry_dir,
            require_no_padding_selected=args.require_no_padding_selected,
            require_coordinate_contract=args.require_coordinate_contract,
        )
        exit_code = 0
    except (AssertionError, ValueError, OSError, json.JSONDecodeError) as exc:
        report = _failure_report(exc, args.geometry_dir)
        exit_code = 1
    if args.output_json:
        _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
