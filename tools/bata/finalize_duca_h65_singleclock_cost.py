from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.bata.duca_full_stack_cost import validate_and_rebuild_profile_summary
from tools.bata.duca_p0_training import atomic_write_json


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"expected a JSON object: {path}")
    validate_and_rebuild_profile_summary(payload)
    return payload


def _e2e_samples(report: Mapping[str, Any]) -> np.ndarray:
    values = []
    for index, row in enumerate(report["raw_samples"]):
        try:
            value = sum(float(row[key]) for key in (
                "input_pipeline_serial_ms", "h2d_ms", "model_forward_ms", "postprocess_ms"
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"profile sample {index} lacks full-stack latency") from exc
        _require(np.isfinite(value) and value > 0.0, "full-stack latency must be finite and positive")
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def _peak_memory_samples(report: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(
        [float(row["peak_gpu_memory_mb"]) for row in report["raw_samples"]],
        dtype=np.float64,
    )
    _require(np.all(np.isfinite(values)) and np.all(values > 0.0), "peak-memory samples must be finite and positive")
    return values


def _check_pair(on: Mapping[str, Any], zero: Mapping[str, Any], repeat: int) -> None:
    for report, role in ((on, "on"), (zero, "gate_zero")):
        _require(report.get("profile_repeat_index") == repeat, f"{role} repeat index drift")
        _require(report.get("complete_official_workload") is True, f"{role} is not a complete official workload")
        _require(report.get("sample_count") == report.get("full_workload_batch_count"), f"{role} workload is incomplete")
        _require(report.get("uses_ema") is True, f"{role} must use state_dict_ema")
        _require(report.get("checkpoint_epoch") == 59, f"{role} must use epoch_59")
        _require(report.get("checkpoint_state_key") == "state_dict_ema", f"{role} checkpoint state drift")
        _require(report.get("warmup_samples") == 50, f"{role} must use 50 warmup windows")
        _require(report.get("batch_size") == 1 and report.get("loader_workers") == 0, f"{role} serial workload drift")
        _require(report.get("tracked_tree_clean") is True, f"{role} evidence tree is dirty")
    for key in (
        "profile_session_id", "profile_pair_id", "hardware_fingerprint", "host_fingerprint",
        "software_fingerprint", "dataset_fingerprint", "source_dataset_fingerprint",
        "inference_fingerprint", "detector_stack_fingerprint",
        "gate_zero_normalized_config_fingerprint", "sample_count", "full_workload_batch_count", "amp",
        "power_sampling_enabled", "power_interval_ms", "power_gpu_id", "checkpoint_path",
        "checkpoint_sha256", "checkpoint_epoch", "checkpoint_state_key",
    ):
        _require(on.get(key) == zero.get(key), f"cost pair differs on {key}")
    _require(on.get("single_clock_gate_zero") is False, "ON cost profile did not enable the learned Clock")
    _require(zero.get("single_clock_gate_zero") is True, "gate-zero cost profile did not zero the Clock")
    _require(
        {int(on.get("profile_order_position", 0)), int(zero.get("profile_order_position", 0))} == {1, 2},
        "paired order positions must be 1 and 2",
    )


def finalize_cost(on_reports: Sequence[Mapping[str, Any]], zero_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(on_reports) == len(zero_reports) == 3, "cost evidence requires exactly three ON/OFF workload repeats")
    on_totals = []
    zero_totals = []
    on_windows = []
    zero_windows = []
    on_memory = []
    zero_memory = []
    for repeat, (on, zero) in enumerate(zip(on_reports, zero_reports), start=1):
        _check_pair(on, zero, repeat)
        on_e2e = _e2e_samples(on)
        zero_e2e = _e2e_samples(zero)
        on_totals.append(float(on_e2e.sum()))
        zero_totals.append(float(zero_e2e.sum()))
        on_windows.append(on_e2e)
        zero_windows.append(zero_e2e)
        on_memory.append(_peak_memory_samples(on))
        zero_memory.append(_peak_memory_samples(zero))

    median_on = float(np.median(on_totals))
    median_zero = float(np.median(zero_totals))
    run_p90_on = [float(np.percentile(row, 90, method="linear")) for row in on_windows]
    run_p90_zero = [float(np.percentile(row, 90, method="linear")) for row in zero_windows]
    p90_on = float(np.median(run_p90_on))
    p90_zero = float(np.median(run_p90_zero))
    peak_on = float(np.max(np.concatenate(on_memory)))
    peak_zero = float(np.max(np.concatenate(zero_memory)))
    _require(median_zero > 0.0 and p90_zero > 0.0 and peak_zero > 0.0, "gate-zero cost denominators must be positive")
    return {
        "schema_version": "duca_h65_singleclock_cost_pair_v1",
        "protocol": "same_node_interleaved_three_complete_official_workloads",
        "complete_workload_repeats": 3,
        "warmup_windows_per_repeat": 50,
        "on_complete_run_total_ms": on_totals,
        "gate_zero_complete_run_total_ms": zero_totals,
        "median_complete_run_latency_ms_on": median_on,
        "median_complete_run_latency_ms_gate_zero": median_zero,
        "median_latency_ratio_on_over_gate_zero": median_on / median_zero,
        "per_run_window_p90_ms_on": run_p90_on,
        "per_run_window_p90_ms_gate_zero": run_p90_zero,
        "median_run_window_p90_ms_on": p90_on,
        "median_run_window_p90_ms_gate_zero": p90_zero,
        "p90_latency_ratio_on_over_gate_zero": p90_on / p90_zero,
        "peak_gpu_memory_mb_on": peak_on,
        "peak_gpu_memory_mb_gate_zero": peak_zero,
        "peak_memory_ratio_on_over_gate_zero": peak_on / peak_zero,
        "primary_checkpoint_state_key": "state_dict_ema",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Finalize the frozen SingleClock ON/gate-zero cost gate")
    parser.add_argument("--on-summary", action="append", required=True)
    parser.add_argument("--gate-zero-summary", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atomic_write_json(
        args.output,
        finalize_cost([_load(path) for path in args.on_summary], [_load(path) for path in args.gate_zero_summary]),
    )


if __name__ == "__main__":
    main()
