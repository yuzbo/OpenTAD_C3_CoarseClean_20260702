from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({str(key) for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def mean(values: Iterable[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not finite else sum(finite) / len(finite)


def max_unselected_hole(valid_len: int, positions: Sequence[int]) -> int:
    selected = sorted(set(int(value) for value in positions))
    if any(value < 0 or value >= valid_len for value in selected):
        raise ValueError("selected positions outside valid prefix")
    if not selected:
        return int(valid_len)
    holes = [selected[0], valid_len - 1 - selected[-1]]
    holes.extend(right - left - 1 for left, right in zip(selected, selected[1:]))
    return max(holes, default=0)


def validate_selection(valid_len: int, budget: int, max_hole: int, positions: Sequence[int]) -> tuple[bool, str]:
    selected = sorted(set(int(value) for value in positions))
    if len(selected) != min(int(budget), int(valid_len)):
        return False, "budget"
    if any(value < 0 or value >= valid_len for value in selected):
        return False, "range"
    if max_unselected_hole(valid_len, selected) > int(max_hole):
        return False, "max_hole"
    return True, "ok"
