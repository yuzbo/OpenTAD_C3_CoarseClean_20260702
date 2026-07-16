from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "duca_cellcf_cost_pair_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    _require(resolved.is_file(), f"cost profile is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"cost profile is not an object: {resolved}")
    payload["_path"] = str(resolved)
    return payload


def _stage(report: Mapping[str, Any], name: str, statistic: str = "p50") -> float:
    try:
        return float(report["stages"][name][statistic])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"profile is missing stages.{name}.{statistic}") from exc


def _validate_group(reports: list[dict[str, Any]], method: str) -> None:
    _require(len(reports) >= 3, f"{method} requires at least three fresh-process repeats")
    reference = reports[0]
    for report in reports:
        _require(report.get("method") == method, f"unexpected method in {report['_path']}")
        _require(int(report.get("sample_count", 0)) >= 500, f"{method} requires at least 500 real windows per repeat")
        _require(report.get("tracked_tree_clean") is True, f"{method} profile used a dirty tree")
        _require(report.get("random_init") is False and report.get("uses_ema") is True, f"{method} must use terminal EMA weights")
        for key in (
            "protocol",
            "hardware_fingerprint",
            "host_fingerprint",
            "software_fingerprint",
            "config_commit",
            "source_dataset_fingerprint",
            "inference_fingerprint",
            "detector_stack_fingerprint",
            "batch_size",
            "loader_workers",
            "amp",
            "checkpoint_sha256",
        ):
            _require(report.get(key) == reference.get(key), f"{method} repeat drifted on {key}")


def summarize(cellcf_paths: list[str], bare_paths: list[str]) -> dict[str, Any]:
    cellcf = [_load(path) for path in cellcf_paths]
    bare = [_load(path) for path in bare_paths]
    _validate_group(cellcf, "cellcf-fixed384")
    _validate_group(bare, "bare-uniform384")
    for key in (
        "protocol",
        "hardware_fingerprint",
        "host_fingerprint",
        "software_fingerprint",
        "config_commit",
        "source_dataset_fingerprint",
        "inference_fingerprint",
        "detector_stack_fingerprint",
        "batch_size",
        "loader_workers",
        "amp",
        "checkpoint_sha256",
    ):
        _require(cellcf[0].get(key) == bare[0].get(key), f"paired profiles differ on {key}")
    _require(all(_stage(report, "frame_selector_total_ms") > 0.0 for report in cellcf), "CellCF selector was not measured")
    _require(all(_stage(report, "coarse_probe_ms") > 0.0 for report in cellcf), "CellCF coarse probe was not measured")
    _require(all(_stage(report, "frame_selector_total_ms") == 0.0 for report in bare), "bare uniform unexpectedly built a selector")
    _require(all(_stage(report, "coarse_probe_ms") == 0.0 for report in bare), "bare uniform unexpectedly built a probe")

    def run_medians(reports: list[dict[str, Any]], stage: str) -> list[float]:
        return [_stage(report, stage) for report in reports]

    cellcf_e2e = run_medians(cellcf, "end_to_end_serial_ms")
    bare_e2e = run_medians(bare, "end_to_end_serial_ms")
    cellcf_median = statistics.median(cellcf_e2e)
    bare_median = statistics.median(bare_e2e)
    return {
        "schema": SCHEMA,
        "ok": True,
        "task": "offline_temporal_action_detection",
        "config_commit": cellcf[0]["config_commit"],
        "hardware_fingerprint": cellcf[0]["hardware_fingerprint"],
        "checkpoint_sha256": cellcf[0]["checkpoint_sha256"],
        "repeats_per_method": min(len(cellcf), len(bare)),
        "samples_per_repeat": min(
            min(int(report["sample_count"]) for report in cellcf),
            min(int(report["sample_count"]) for report in bare),
        ),
        "cellcf": {
            "end_to_end_p50_ms_by_repeat": cellcf_e2e,
            "end_to_end_p50_ms_median": cellcf_median,
            "selector_p50_ms_median": statistics.median(run_medians(cellcf, "frame_selector_total_ms")),
            "coarse_probe_p50_ms_median": statistics.median(run_medians(cellcf, "coarse_probe_ms")),
        },
        "bare_uniform": {
            "end_to_end_p50_ms_by_repeat": bare_e2e,
            "end_to_end_p50_ms_median": bare_median,
        },
        "frontend_overhead": {
            "end_to_end_ms": cellcf_median - bare_median,
            "fraction_of_bare_uniform": (cellcf_median - bare_median) / bare_median,
            "cellcf_to_bare_ratio": cellcf_median / bare_median,
        },
        "claim_scope": "frontend_overhead_only_not_dense_compute_saving",
        "dense_baseline_still_required": True,
        "cellcf_profile_paths": [report["_path"] for report in cellcf],
        "bare_uniform_profile_paths": [report["_path"] for report in bare],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cellcf", action="append", required=True)
    parser.add_argument("--bare-uniform", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    try:
        payload = summarize(args.cellcf, args.bare_uniform)
        code = 0
    except Exception as exc:
        payload = {"schema": SCHEMA, "ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        code = 1
    output = json.dumps(payload, indent=2, sort_keys=True)
    print(output)
    Path(args.output_json).write_text(output + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
