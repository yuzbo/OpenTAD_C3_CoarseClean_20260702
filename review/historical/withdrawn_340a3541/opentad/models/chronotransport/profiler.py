from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Iterator, Mapping

import torch


REQUIRED_STAGE_FIELDS = (
    "data_decode",
    "preprocess",
    "h2d",
    "innovation",
    "scheduler",
    "recompute",
    "transport",
    "cache_movement",
    "neck_head",
    "postprocess",
)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class ChronoProfiler:
    def __init__(self, *, sync_cuda: bool = True) -> None:
        self.sync_cuda = bool(sync_cuda)
        self._latency_ms: dict[str, list[float]] = defaultdict(list)
        self._action_counts: dict[str, int] = defaultdict(int)
        self._metadata: dict[str, object] = {}

    def _sync(self) -> None:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self._sync()
        start = perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.record(name, (perf_counter() - start) * 1000.0)

    def record(self, name: str, latency_ms: float) -> None:
        value = float(latency_ms)
        if value < 0.0:
            raise ValueError("latency must be non-negative")
        self._latency_ms[str(name)].append(value)

    def record_actions(self, **counts: int) -> None:
        for name, count in counts.items():
            count = int(count)
            if count < 0:
                raise ValueError("action counts must be non-negative")
            self._action_counts[str(name)] += count

    def update_metadata(self, **metadata: object) -> None:
        self._metadata.update(metadata)

    def summary(self, *, fill_missing: bool = True) -> dict[str, object]:
        names = set(self._latency_ms)
        if fill_missing:
            names.update(REQUIRED_STAGE_FIELDS)
        latency = {}
        for name in sorted(names):
            values = list(self._latency_ms.get(name, ()))
            latency[name] = {
                "count": len(values),
                "total": float(sum(values)),
                "p50": float(_percentile(values, 0.50)),
                "p95": float(_percentile(values, 0.95)),
            }
        peak_gpu = None
        if torch.cuda.is_available():
            peak_gpu = int(torch.cuda.max_memory_allocated())
        return {
            "schema_version": "chronotransport_profile_v1",
            "latency_ms": latency,
            "action_counts": dict(self._action_counts),
            "peak_gpu_memory_bytes": peak_gpu,
            "metadata": dict(self._metadata),
        }

    @staticmethod
    def validate_summary(summary: Mapping[str, object]) -> None:
        latency = summary.get("latency_ms")
        if not isinstance(latency, Mapping):
            raise ValueError("profile summary must contain latency_ms mapping")
        missing = [name for name in REQUIRED_STAGE_FIELDS if name not in latency]
        if missing:
            raise ValueError(f"profile summary missing required fields: {missing}")
