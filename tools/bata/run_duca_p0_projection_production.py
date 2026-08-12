from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from opentad.models.selectors.pc_ot_mras_prebackbone_frame_selector import (
    DUCAProjectionError,
    project_duca_fixed_targets_v001,
)


SCHEMA_VERSION = "duca_p0_projection_production_v001"
FIELD_ORDER = ("T", "K", "Q", "u", "a")
FIXED_Q = 1_048_576


def _resolve_paths(input_jsonl: str | Path, output_jsonl: str | Path) -> tuple[Path, Path]:
    input_path = Path(input_jsonl).expanduser().resolve(strict=True)
    output_path = Path(output_jsonl).expanduser().resolve(strict=False)
    if not input_path.is_file():
        raise ValueError(f"input JSONL is not a file: {input_path}")
    if output_path == input_path:
        raise ValueError("production output must be distinct from the sealed input")

    sealed_root = input_path.parent.parent if input_path.parent.name == "materialized" else input_path.parent
    if output_path == sealed_root or sealed_root in output_path.parents:
        raise ValueError("production output must be outside the sealed Evaluator package")
    if not output_path.parent.is_dir():
        raise ValueError(f"Builder output directory does not exist: {output_path.parent}")
    if output_path.exists():
        raise FileExistsError(f"production output already exists: {output_path}")
    return input_path, output_path


def _read_sealed_rows(input_path: Path) -> list[tuple[int, str, dict[str, Any]]]:
    rows: list[tuple[int, str, dict[str, Any]]] = []
    with input_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.endswith(b"\n") or raw_line.endswith(b"\r\n"):
                raise ValueError(f"line {line_number}: sealed JSONL must use LF termination")
            raw_json = raw_line[:-1]
            if not raw_json:
                raise ValueError(f"line {line_number}: blank records are forbidden")
            input_json = raw_json.decode("utf-8")
            record = json.loads(input_json)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: sealed record must be a JSON object")
            if tuple(record) != FIELD_ORDER:
                raise ValueError(
                    f"line {line_number}: sealed fields must be ordered exactly as {FIELD_ORDER}"
                )
            if type(record["Q"]) is not int or record["Q"] != FIXED_Q:
                raise ValueError(f"line {line_number}: Q must equal {FIXED_Q}")
            if type(record["T"]) is not int or type(record["K"]) is not int:
                raise ValueError(f"line {line_number}: T and K must be integers")
            if not isinstance(record["u"], list) or not isinstance(record["a"], list):
                raise ValueError(f"line {line_number}: u and a must be integer-array inputs")
            rows.append((line_number, input_json, record))
    if not rows:
        raise ValueError("sealed input contains no records")
    return rows


def _project_row(line_number: int, input_json: str, record: dict[str, Any]) -> dict[str, Any]:
    try:
        certificate = project_duca_fixed_targets_v001(
            record["T"],
            record["K"],
            record["u"],
            record["a"],
        )
    except DUCAProjectionError as exc:
        if not exc.code:
            raise RuntimeError(
                f"line {line_number}: production projector returned an untyped failure"
            ) from exc
        return {
            "schema_version": SCHEMA_VERSION,
            "line_number": line_number,
            "input_json": input_json,
            "status": exc.code,
            "p": None,
            "E2": None,
            "E_infinity": None,
            "E1": None,
            "U1": None,
            "candidate_order": "ascending",
            "scope_deviation": "none",
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "line_number": line_number,
        "input_json": input_json,
        "status": "OK",
        "p": list(certificate.positions),
        "E2": certificate.e2,
        "E_infinity": certificate.e_infinity,
        "E1": certificate.e1,
        "U1": certificate.u1,
        "candidate_order": "ascending",
        "scope_deviation": "none",
    }


def _publish_jsonl_atomically(output_path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def run_production_projection(input_jsonl: str | Path, output_jsonl: str | Path) -> None:
    input_path, output_path = _resolve_paths(input_jsonl, output_jsonl)
    sealed_rows = _read_sealed_rows(input_path)
    output_rows = [
        _project_row(line_number, input_json, record)
        for line_number, input_json, record in sealed_rows
    ]
    _publish_jsonl_atomically(output_path, output_rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen DUCA P0 production projector on sealed JSONL records."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args(argv)
    run_production_projection(args.input_jsonl, args.output_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
