from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CostLookupKey:
    hardware: str
    precision: str
    batch_size: int
    candidate_schedule: str
    selected_rows_per_group: tuple[int, ...]

    def encode(self) -> str:
        rows = ",".join(str(int(value)) for value in self.selected_rows_per_group)
        return f"{self.hardware}|{self.precision}|{int(self.batch_size)}|{self.candidate_schedule}|{rows}"


class ScheduleCostLookup:
    schema_version = "chronotransport_schedule_cost_v1"

    def __init__(self, entries: Mapping[str, Mapping[str, float]]) -> None:
        self.entries = {str(key): dict(value) for key, value in entries.items()}
        for value in self.entries.values():
            for statistic in ("p50", "p95"):
                number = float(value[statistic])
                if not math.isfinite(number) or number < 0:
                    raise ValueError("p50/p95 costs must be finite and non-negative")

    @classmethod
    def from_json(cls, path: str | Path) -> "ScheduleCostLookup":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != cls.schema_version:
            raise ValueError("unsupported schedule cost schema")
        return cls(payload["entries"])

    def get(self, key: CostLookupKey, statistic: str = "p50") -> float:
        if statistic not in {"p50", "p95"}:
            raise ValueError("statistic must be p50 or p95")
        encoded = key.encode()
        if encoded not in self.entries:
            raise KeyError(f"missing measured schedule cost: {encoded}")
        return float(self.entries[encoded][statistic])

    @classmethod
    def payload(cls, entries: Sequence[tuple[CostLookupKey, float, float]]) -> dict:
        return {
            "schema_version": cls.schema_version,
            "entries": {
                key.encode(): {"p50": float(p50), "p95": float(p95)}
                for key, p50, p95 in entries
            },
        }
