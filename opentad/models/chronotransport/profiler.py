from __future__ import annotations

import math
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
    "dense_adatad_adapter",
    "neck_head",
    "postprocess",
)
PROFILE_SCHEMA_VERSION = "chronotransport_profile_v2"


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class ChronoProfiler:
    def __init__(
        self,
        *,
        sync_cuda: bool = True,
        deferred_cuda_events: bool = False,
    ) -> None:
        self.sync_cuda = bool(sync_cuda)
        self.deferred_cuda_events = bool(deferred_cuda_events)
        if self.sync_cuda and self.deferred_cuda_events:
            raise ValueError("profiler sync and deferred CUDA modes are mutually exclusive")
        self._latency_ms: dict[str, list[float]] = defaultdict(list)
        self._cuda_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._action_counts: dict[str, int] = defaultdict(int)
        self._metadata: dict[str, object] = {}

    @property
    def has_pending_cuda_events(self) -> bool:
        """Whether deferred CUDA measurements still need an outer-boundary flush."""

        return bool(self._cuda_events)

    def flush_deferred_cuda_events(self, *, synchronize: bool) -> None:
        """Materialize deferred CUDA durations after the caller's timing sync.

        Formal full-stack timing synchronizes once at the outer invocation
        boundary.  Requiring that caller to flush with ``synchronize=False``
        keeps diagnostic stage timing from inserting a hidden mid-forward
        synchronization into the primary latency sample.
        """

        if type(synchronize) is not bool:
            raise TypeError("deferred CUDA synchronize flag must be boolean")
        if not self._cuda_events:
            return
        if synchronize:
            self._cuda_events[-1][2].synchronize()
        events = self._cuda_events
        self._cuda_events = []
        for name, start, end in events:
            self.record(name, float(start.elapsed_time(end)))

    def _sync(self) -> None:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self.deferred_cuda_events and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            try:
                yield
            finally:
                end_event.record()
                self._cuda_events.append((str(name), start_event, end_event))
            return
        self._sync()
        start = perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.record(name, (perf_counter() - start) * 1000.0)

    def record(self, name: str, latency_ms: float) -> None:
        value = float(latency_ms)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("latency must be finite and non-negative")
        self._latency_ms[str(name)].append(value)

    def record_actions(self, **counts: int) -> None:
        for name, count in counts.items():
            count = int(count)
            if count < 0:
                raise ValueError("action counts must be non-negative")
            self._action_counts[str(name)] += count

    def update_metadata(self, **metadata: object) -> None:
        self._metadata.update(metadata)

    def summary(
        self,
        *,
        fill_missing: bool = True,
        flush_deferred: bool = True,
        synchronize_deferred: bool = True,
    ) -> dict[str, object]:
        if type(flush_deferred) is not bool or type(synchronize_deferred) is not bool:
            raise TypeError("profiler deferred-summary flags must be boolean")
        if flush_deferred:
            self.flush_deferred_cuda_events(synchronize=synchronize_deferred)
        elif self._cuda_events:
            raise RuntimeError(
                "deferred CUDA events must be flushed at the outer timing boundary"
            )
        names = set(self._latency_ms)
        if fill_missing:
            names.update(REQUIRED_STAGE_FIELDS)
        latency = {}
        for name in sorted(names):
            values = list(self._latency_ms.get(name, ()))
            latency[name] = {
                "count": len(values),
                "total": float(sum(values)),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
            }
        peak_gpu = None
        if torch.cuda.is_available():
            peak_gpu = int(torch.cuda.max_memory_allocated())
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "latency_ms": latency,
            "action_counts": dict(self._action_counts),
            "peak_gpu_memory_bytes": peak_gpu,
            "metadata": dict(self._metadata),
        }

    @staticmethod
    def validate_summary(summary: Mapping[str, object]) -> None:
        if summary.get("schema_version") != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"profile summary schema must equal {PROFILE_SCHEMA_VERSION}"
            )
        latency = summary.get("latency_ms")
        if not isinstance(latency, Mapping):
            raise ValueError("profile summary must contain latency_ms mapping")
        required = (*REQUIRED_STAGE_FIELDS, "total_ms")
        missing = [name for name in required if name not in latency]
        if missing:
            raise ValueError(f"profile summary missing required direct samples: {missing}")
        for name in required:
            measurement = latency[name]
            if not isinstance(measurement, Mapping):
                raise ValueError(f"profile field {name} must contain direct samples")
            count = measurement.get("count")
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError(f"profile field {name} must contain direct samples")
            for statistic in ("total", "p50", "p95"):
                value = measurement.get(statistic)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(f"profile field {name}.{statistic} must be measured")
                if not math.isfinite(float(value)) or float(value) < 0.0:
                    raise ValueError(f"profile field {name}.{statistic} must be finite/non-negative")
