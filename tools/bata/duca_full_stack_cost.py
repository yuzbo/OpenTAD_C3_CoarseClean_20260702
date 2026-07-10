from __future__ import annotations

import csv
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


OFFLINE_FULL_WINDOW_PROTOCOL = "offline_full_window_runtime_selection"
PROFILE_SCHEMA_VERSION = "duca-full-stack-cost-v1"

_REQUIRED_TOP_LEVEL_STAGES = (
    "input_pipeline_serial_ms",
    "h2d_ms",
    "model_forward_ms",
    "postprocess_ms",
)

_MODEL_CHILD_STAGES = (
    "frame_selector_total_ms",
    "backbone_wrapper_total_ms",
    "projection_ms",
    "neck_ms",
    "head_ms",
)

_REQUIRED_NESTED_STAGES = _MODEL_CHILD_STAGES + (
    "coarse_probe_ms",
    "heavy_backbone_ms",
)


class StageRecorder:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        synchronize: Callable[[], None] | None = None,
    ) -> None:
        self._clock = clock
        self._synchronize = synchronize or (lambda: None)
        self._sample: dict[str, float] | None = None
        self._open_measurements = 0

    def begin_sample(self) -> None:
        if self._sample is not None:
            raise RuntimeError("a StageRecorder sample is already active")
        self._sample = {}
        self._open_measurements = 0

    def end_sample(self) -> dict[str, float]:
        if self._sample is None:
            raise RuntimeError("begin_sample must be called before end_sample")
        if self._open_measurements:
            raise RuntimeError("cannot end a StageRecorder sample with open measurements")
        sample = dict(self._sample)
        self._sample = None
        return sample

    def record_value(self, name: str, value: float, *, accumulate: bool = False) -> None:
        if self._sample is None:
            raise RuntimeError("begin_sample must be called before recording a stage")
        checked = _finite_nonnegative(value, name=name)
        if accumulate:
            self._sample[name] = self._sample.get(name, 0.0) + checked
        else:
            self._sample[name] = checked

    def start_stage(self) -> float:
        if self._sample is None:
            raise RuntimeError("begin_sample must be called before measuring a stage")
        self._synchronize()
        self._open_measurements += 1
        return float(self._clock())

    def stop_stage(self, name: str, started_at: float) -> float:
        if self._sample is None:
            raise RuntimeError("begin_sample must be called before measuring a stage")
        if self._open_measurements <= 0:
            raise RuntimeError("stop_stage called without a matching start_stage")
        self._synchronize()
        elapsed_ms = max(0.0, (float(self._clock()) - float(started_at)) * 1000.0)
        self._open_measurements -= 1
        self.record_value(name, elapsed_ms, accumulate=True)
        return elapsed_ms

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started_at = self.start_stage()
        try:
            yield
        finally:
            self.stop_stage(name, started_at)


class ModuleStageHooks:
    def __init__(self, recorder: StageRecorder) -> None:
        self.recorder = recorder
        self._handles: list[Any] = []
        self._starts: dict[tuple[int, str], list[float]] = {}

    def register(self, name: str, module: Any) -> None:
        if module is None:
            raise ValueError(f"cannot register {name} on a missing module")
        if not hasattr(module, "register_forward_pre_hook") or not hasattr(module, "register_forward_hook"):
            raise TypeError(f"module for {name} does not support forward hooks")
        key = (id(module), str(name))

        def before(_module: Any, _inputs: Any) -> None:
            self._starts.setdefault(key, []).append(self.recorder.start_stage())

        def after(_module: Any, _inputs: Any, _output: Any) -> None:
            starts = self._starts.get(key)
            if not starts:
                raise RuntimeError(f"forward hook for {name} is missing its start timestamp")
            self.recorder.stop_stage(str(name), starts.pop())

        self._handles.append(module.register_forward_pre_hook(before))
        self._handles.append(module.register_forward_hook(after))

    def close(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._starts.clear()


class MethodStageHooks:
    def __init__(self, recorder: StageRecorder) -> None:
        self.recorder = recorder
        self._originals: list[tuple[Any, str, Any]] = []

    def register(self, name: str, target: Any, method_name: str) -> None:
        original = getattr(target, method_name, None)
        if original is None or not callable(original):
            raise ValueError(f"cannot profile missing method {method_name} for {name}")

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with self.recorder.measure(name):
                return original(*args, **kwargs)

        self._originals.append((target, method_name, original))
        setattr(target, method_name, wrapped)

    def close(self) -> None:
        for target, method_name, original in reversed(self._originals):
            setattr(target, method_name, original)
        self._originals.clear()


def _finite_nonnegative(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty value sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = probability * float(len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - float(lower)
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty value sequence")
    checked = [_finite_nonnegative(item, name="stage sample") for item in values]
    return {
        "count": len(checked),
        "mean": sum(checked) / float(len(checked)),
        "p50": _quantile(checked, 0.50),
        "p95": _quantile(checked, 0.95),
        "min": min(checked),
        "max": max(checked),
    }


def _validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(metadata)
    required = (
        "method",
        "protocol",
        "hardware_fingerprint",
        "host_fingerprint",
        "software_fingerprint",
        "config_commit",
        "tracked_tree_clean",
        "dataset_fingerprint",
        "inference_fingerprint",
        "detector_stack_fingerprint",
        "batch_size",
        "loader_workers",
        "warmup_samples",
        "amp",
        "uses_ema",
        "random_init",
        "power_sampling_enabled",
        "power_interval_ms",
        "power_gpu_id",
    )
    for key in required:
        if key not in out:
            raise ValueError(f"profile metadata requires {key}")
    for key in (
        "method",
        "protocol",
        "hardware_fingerprint",
        "host_fingerprint",
        "software_fingerprint",
        "config_commit",
        "dataset_fingerprint",
        "inference_fingerprint",
        "detector_stack_fingerprint",
    ):
        if not str(out.get(key, "")).strip():
            raise ValueError(f"profile metadata requires {key}")
    if out["protocol"] != OFFLINE_FULL_WINDOW_PROTOCOL:
        raise ValueError(
            "DUCA cost profiles must use the offline full-window protocol; "
            f"got {out['protocol']!r}"
        )
    return out


def _derived_sample(sample: Mapping[str, Any], *, index: int) -> dict[str, float]:
    checked: dict[str, float] = {}
    for key in _REQUIRED_TOP_LEVEL_STAGES + _REQUIRED_NESTED_STAGES:
        if key not in sample:
            raise ValueError(f"profile sample {index} is missing {key}")
        checked[key] = _finite_nonnegative(sample[key], name=f"sample[{index}].{key}")

    model_children = sum(checked[key] for key in _MODEL_CHILD_STAGES)
    model_total = checked["model_forward_ms"]
    tolerance = max(1.0e-6, model_total * 1.0e-4)
    if model_children > model_total + tolerance:
        breakdown = ", ".join(f"{key}={checked[key]:.6f}" for key in _MODEL_CHILD_STAGES)
        raise ValueError(
            f"profile sample {index} child stages exceed model_forward_ms: "
            f"children={model_children:.6f}, model_forward_ms={model_total:.6f}; {breakdown}"
        )

    selector_total = checked["frame_selector_total_ms"]
    coarse_probe = checked["coarse_probe_ms"]
    if coarse_probe > selector_total + tolerance:
        raise ValueError(f"profile sample {index} coarse_probe_ms exceeds frame_selector_total_ms")

    backbone_total = checked["backbone_wrapper_total_ms"]
    heavy_backbone = checked["heavy_backbone_ms"]
    if heavy_backbone > backbone_total + tolerance:
        raise ValueError(f"profile sample {index} heavy_backbone_ms exceeds backbone_wrapper_total_ms")

    derived = dict(checked)
    derived["end_to_end_serial_ms"] = sum(checked[key] for key in _REQUIRED_TOP_LEVEL_STAGES)
    derived["selector_policy_ms"] = max(0.0, selector_total - coarse_probe)
    derived["backbone_wrapper_overhead_ms"] = max(0.0, backbone_total - heavy_backbone)
    derived["model_unattributed_ms"] = max(0.0, model_total - model_children)
    return derived


def build_profile_summary(
    samples: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one measured sample is required")
    meta = _validate_metadata(metadata)
    derived_samples = [_derived_sample(sample, index=index) for index, sample in enumerate(samples)]

    stage_names = sorted(derived_samples[0])
    stages = {
        name: _summary([sample[name] for sample in derived_samples])
        for name in stage_names
    }

    selected_counts = []
    peak_memory = []
    energy = []
    for index, sample in enumerate(samples):
        if "selected_count" not in sample:
            raise ValueError(f"profile sample {index} is missing selected_count")
        selected_counts.append(_finite_nonnegative(sample["selected_count"], name="selected_count"))
        if sample.get("peak_gpu_memory_mb") is not None:
            peak_memory.append(_finite_nonnegative(sample["peak_gpu_memory_mb"], name="peak_gpu_memory_mb"))
        if sample.get("gpu_energy_j") is not None:
            energy.append(_finite_nonnegative(sample["gpu_energy_j"], name="gpu_energy_j"))

    report = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        **meta,
        "sample_count": len(samples),
        "stage_semantics": {
            "input_pipeline_serial_ms": "dataset read, decode, transforms, and collate with zero loader workers",
            "h2d_ms": "recursive tensor transfer to the measured accelerator",
            "model_forward_ms": "raw detector forward without post-processing",
            "postprocess_ms": "detector post-processing, including NMS when configured",
            "coarse_probe_ms": "nested inside frame_selector_total_ms",
            "heavy_backbone_ms": "nested inside backbone_wrapper_total_ms",
        },
        "stages": stages,
        "selected_count": _summary(selected_counts),
        "resources": {
            "peak_gpu_memory_mb": None if not peak_memory else _summary(peak_memory),
        },
        "energy": {
            "gpu_energy_j": None if not energy else _summary(energy),
            "measurement_scope": "h2d_plus_model_forward_plus_postprocess",
        },
        "claims": {
            "full_stack_latency_measured": True,
            "offline_full_window": True,
            "decoder_and_preprocess_included": True,
            "decoder_and_preprocess_separated": False,
            "estimated_flops_used_as_latency": False,
        },
        "raw_samples": [dict(sample) for sample in samples],
    }
    return report


def _require_comparable(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    if bool(baseline.get("random_init")) or bool(candidate.get("random_init")):
        raise ValueError("random_init profiles are smoke diagnostics and cannot be paper-comparable")
    if not bool(baseline.get("tracked_tree_clean")) or not bool(candidate.get("tracked_tree_clean")):
        raise ValueError("tracked_tree_clean must be true for paper-comparable profiles")
    for key in (
        "schema_version",
        "protocol",
        "hardware_fingerprint",
        "host_fingerprint",
        "software_fingerprint",
        "config_commit",
        "dataset_fingerprint",
        "inference_fingerprint",
        "detector_stack_fingerprint",
        "batch_size",
        "loader_workers",
        "warmup_samples",
        "amp",
        "uses_ema",
        "power_sampling_enabled",
        "power_interval_ms",
        "power_gpu_id",
        "sample_count",
    ):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(
                f"cost profiles have incompatible {key}: "
                f"baseline={baseline.get(key)!r}, candidate={candidate.get(key)!r}"
            )


def _stage_p50(report: Mapping[str, Any], name: str) -> float:
    try:
        return float(report["stages"][name]["p50"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"profile is missing stages.{name}.p50") from exc


def _stage_p95(report: Mapping[str, Any], name: str) -> float:
    try:
        return float(report["stages"][name]["p95"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"profile is missing stages.{name}.p95") from exc


def _optional_summary_value(report: Mapping[str, Any], path: Sequence[str], statistic: str = "p50") -> float | None:
    value: Any = report
    try:
        for key in path:
            value = value[key]
        if value is None:
            return None
        return float(value[statistic])
    except (KeyError, TypeError, ValueError):
        return None


def compare_profile_summaries(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    _require_comparable(baseline, candidate)
    baseline_e2e = _stage_p50(baseline, "end_to_end_serial_ms")
    candidate_e2e = _stage_p50(candidate, "end_to_end_serial_ms")
    baseline_e2e_p95 = _stage_p95(baseline, "end_to_end_serial_ms")
    candidate_e2e_p95 = _stage_p95(candidate, "end_to_end_serial_ms")
    if baseline_e2e <= 0.0 or candidate_e2e <= 0.0:
        raise ValueError("end-to-end p50 latency must be positive")

    baseline_heavy = _stage_p50(baseline, "heavy_backbone_ms")
    candidate_heavy = _stage_p50(candidate, "heavy_backbone_ms")
    heavy_saving = baseline_heavy - candidate_heavy
    candidate_frontend = _stage_p50(candidate, "frame_selector_total_ms")
    frontend_fraction = math.inf if heavy_saving <= 0.0 else candidate_frontend / heavy_saving
    saving_fraction = (baseline_e2e - candidate_e2e) / baseline_e2e
    p95_saving_fraction = (baseline_e2e_p95 - candidate_e2e_p95) / baseline_e2e_p95
    cost_gates = {
        "end_to_end_saving_at_least_15pct": saving_fraction >= 0.15,
        "p95_end_to_end_saving_at_least_15pct": p95_saving_fraction >= 0.15,
        "frontend_consumes_at_most_40pct_of_backbone_saving": frontend_fraction <= 0.40,
        "heavy_backbone_saving_positive": heavy_saving > 0.0,
    }
    cost_gates["all_cost_gates_pass"] = all(cost_gates.values())

    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "comparable": True,
        "protocol": baseline["protocol"],
        "hardware_fingerprint": baseline["hardware_fingerprint"],
        "baseline_method": baseline.get("method"),
        "candidate_method": candidate.get("method"),
        "end_to_end_serial": {
            "baseline_ms": baseline_e2e,
            "candidate_ms": candidate_e2e,
            "latency_saving_ms": baseline_e2e - candidate_e2e,
            "latency_saving_fraction": saving_fraction,
            "speedup": baseline_e2e / candidate_e2e,
            "baseline_p95_ms": baseline_e2e_p95,
            "candidate_p95_ms": candidate_e2e_p95,
            "p95_latency_saving_ms": baseline_e2e_p95 - candidate_e2e_p95,
            "p95_latency_saving_fraction": p95_saving_fraction,
            "p95_speedup": baseline_e2e_p95 / candidate_e2e_p95,
        },
        "heavy_backbone": {
            "baseline_ms": baseline_heavy,
            "candidate_ms": candidate_heavy,
            "latency_saving_ms": heavy_saving,
        },
        "frontend_overhead": {
            "candidate_ms": candidate_frontend,
            "fraction_of_heavy_backbone_saving": frontend_fraction,
        },
        "resources": {
            "baseline_peak_gpu_memory_mb": _optional_summary_value(
                baseline, ("resources", "peak_gpu_memory_mb")
            ),
            "candidate_peak_gpu_memory_mb": _optional_summary_value(
                candidate, ("resources", "peak_gpu_memory_mb")
            ),
            "baseline_gpu_energy_j": _optional_summary_value(baseline, ("energy", "gpu_energy_j")),
            "candidate_gpu_energy_j": _optional_summary_value(candidate, ("energy", "gpu_energy_j")),
        },
        "selected_count_p50": _optional_summary_value(candidate, ("selected_count",)),
        "gates": cost_gates,
    }


def build_cost_matrix(
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one candidate cost profile is required")
    comparisons = [compare_profile_summaries(baseline, candidate) for candidate in candidates]
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "protocol": baseline.get("protocol"),
        "hardware_fingerprint": baseline.get("hardware_fingerprint"),
        "baseline_method": baseline.get("method"),
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }


def write_cost_matrix_artifacts(matrix: Mapping[str, Any], output_prefix: str | Path) -> dict[str, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    tsv_path = prefix.with_suffix(".tsv")
    json_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = (
        "candidate_method",
        "selected_count_p50",
        "e2e_p50_ms",
        "e2e_p95_ms",
        "p50_saving_fraction",
        "p95_saving_fraction",
        "p50_speedup",
        "heavy_backbone_saving_ms",
        "frontend_ms",
        "frontend_fraction_of_backbone_saving",
        "peak_gpu_memory_mb",
        "gpu_energy_j",
        "all_cost_gates_pass",
    )
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for comparison in matrix.get("comparisons", []):
            end_to_end = comparison["end_to_end_serial"]
            frontend = comparison["frontend_overhead"]
            resources = comparison["resources"]
            writer.writerow(
                {
                    "candidate_method": comparison.get("candidate_method"),
                    "selected_count_p50": comparison.get("selected_count_p50"),
                    "e2e_p50_ms": end_to_end["candidate_ms"],
                    "e2e_p95_ms": end_to_end["candidate_p95_ms"],
                    "p50_saving_fraction": end_to_end["latency_saving_fraction"],
                    "p95_saving_fraction": end_to_end["p95_latency_saving_fraction"],
                    "p50_speedup": end_to_end["speedup"],
                    "heavy_backbone_saving_ms": comparison["heavy_backbone"]["latency_saving_ms"],
                    "frontend_ms": frontend["candidate_ms"],
                    "frontend_fraction_of_backbone_saving": frontend["fraction_of_heavy_backbone_saving"],
                    "peak_gpu_memory_mb": resources["candidate_peak_gpu_memory_mb"],
                    "gpu_energy_j": resources["candidate_gpu_energy_j"],
                    "all_cost_gates_pass": comparison["gates"]["all_cost_gates_pass"],
                }
            )
    return {"json": json_path, "tsv": tsv_path}


def _interpolate_power(samples: Sequence[tuple[float, float]], timestamp: float) -> float:
    if timestamp <= samples[0][0]:
        return samples[0][1]
    if timestamp >= samples[-1][0]:
        return samples[-1][1]
    for left, right in zip(samples[:-1], samples[1:]):
        if left[0] <= timestamp <= right[0]:
            span = right[0] - left[0]
            if span <= 0.0:
                return right[1]
            weight = (timestamp - left[0]) / span
            return left[1] * (1.0 - weight) + right[1] * weight
    return samples[-1][1]


def integrate_power_samples(
    samples: Iterable[tuple[float, float]],
    *,
    start_time_s: float,
    end_time_s: float,
) -> dict[str, float | int]:
    start = float(start_time_s)
    end = float(end_time_s)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("power integration requires finite end_time_s > start_time_s")

    checked = sorted(
        (
            float(timestamp),
            _finite_nonnegative(power, name="power_w"),
        )
        for timestamp, power in samples
        if math.isfinite(float(timestamp))
    )
    if not checked:
        raise ValueError("at least one power sample is required")
    if len(checked) < 2 or checked[0][0] > start or checked[-1][0] < end:
        raise ValueError("power samples must bracket the complete measurement window")

    interior = [(timestamp, power) for timestamp, power in checked if start < timestamp < end]
    clipped = [
        (start, _interpolate_power(checked, start)),
        *interior,
        (end, _interpolate_power(checked, end)),
    ]
    energy_j = 0.0
    for left, right in zip(clipped[:-1], clipped[1:]):
        energy_j += 0.5 * (left[1] + right[1]) * (right[0] - left[0])
    duration = end - start
    return {
        "sample_count": len(checked),
        "interior_sample_count": len(interior),
        "window_bracketed": True,
        "duration_s": duration,
        "average_power_w": energy_j / duration,
        "energy_j": energy_j,
    }


def write_profile_artifacts(report: Mapping[str, Any], output_prefix: str | Path) -> dict[str, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".summary.json")
    tsv_path = prefix.with_suffix(".summary.tsv")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("stage", "count", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms"))
        for stage, values in sorted(report.get("stages", {}).items()):
            writer.writerow(
                (
                    stage,
                    values["count"],
                    f"{float(values['mean']):.6f}",
                    f"{float(values['p50']):.6f}",
                    f"{float(values['p95']):.6f}",
                    f"{float(values['min']):.6f}",
                    f"{float(values['max']):.6f}",
                )
            )
    return {"json": json_path, "tsv": tsv_path}


__all__ = [
    "OFFLINE_FULL_WINDOW_PROTOCOL",
    "PROFILE_SCHEMA_VERSION",
    "MethodStageHooks",
    "ModuleStageHooks",
    "StageRecorder",
    "build_cost_matrix",
    "build_profile_summary",
    "compare_profile_summaries",
    "integrate_power_samples",
    "write_cost_matrix_artifacts",
    "write_profile_artifacts",
]
