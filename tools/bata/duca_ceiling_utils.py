from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SHA256_KEYS = (
    "config_sha256",
    "checkpoint_sha256",
    "data_sha256",
    "geometry_sha256",
    "gt_sha256",
)


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


def require_finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def validate_provenance(value: Any, *, context: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: provenance object is required")
    commit = value.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError(f"{context}: provenance git_commit must be a full commit")
    result = {"git_commit": commit}
    for key in SHA256_KEYS:
        digest = value.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ValueError(f"{context}: provenance {key} must be SHA-256")
        result[key] = digest.lower()
    evaluator = value.get("evaluator_identity")
    if not isinstance(evaluator, str) or not evaluator:
        raise ValueError(f"{context}: provenance evaluator_identity is required")
    result["evaluator_identity"] = evaluator
    return result
