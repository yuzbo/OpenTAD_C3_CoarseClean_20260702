from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.bata.bootstrap_duca_h65_official_map import (
    _METRIC_KEYS,
    exact_interval,
)
from tools.bata.duca_p0_evaluation import recompute_official_map, sha256_file
from tools.bata.duca_p0_training import atomic_write_json


_IDENTITY_KEYS = (
    "official_evaluator_reexecuted_per_resample",
    "paired_video_cluster_bootstrap",
    "rng",
    "nonce",
    "namespace",
    "seed_uint64",
    "seed_sha256",
    "samples",
    "interval_rank_convention",
    "lower_rank",
    "upper_rank",
    "baseline_family",
    "family_order",
    "video_ids",
    "prediction_paths",
    "prediction_sha256",
    "evaluation_config",
    "evaluation_config_sha256",
    "evaluator",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def merge_bootstrap_shard_payloads(
    shards: Sequence[Mapping[str, Any]],
    *,
    point_estimates: Mapping[str, Any],
    shard_paths: Sequence[str] | None = None,
    shard_sha256: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Merge contiguous deterministic shards without changing draw order."""
    _require(bool(shards), "at least one bootstrap shard is required")
    ordered = sorted(shards, key=lambda row: int(row.get("sample_start", -1)))
    reference = ordered[0]
    for row in ordered:
        _require(
            row.get("schema_version")
            == "duca_h65_official_pcg64_video_bootstrap_shard_v1",
            "bootstrap shard schema mismatch",
        )
        for key in _IDENTITY_KEYS:
            _require(row.get(key) == reference.get(key), f"bootstrap shard identity differs: {key}")

    samples = int(reference["samples"])
    cursor = 0
    families = tuple(reference["family_order"])
    sampled = {
        family: {metric: [] for metric in _METRIC_KEYS} for family in families
    }
    shard_execution = []
    for row in ordered:
        start = int(row["sample_start"])
        stop = int(row["sample_stop"])
        _require(start == cursor and start < stop <= samples, "bootstrap shard coverage is not contiguous")
        _require(int(row["shard_samples"]) == stop - start, "bootstrap shard length metadata mismatch")
        shard_metrics = row.get("sampled_metrics")
        _require(isinstance(shard_metrics, Mapping), "bootstrap shard metrics are missing")
        for family in families:
            _require(set(shard_metrics[family]) == set(_METRIC_KEYS), "bootstrap shard metric keys differ")
            for metric in _METRIC_KEYS:
                values = list(shard_metrics[family][metric])
                _require(len(values) == stop - start, "bootstrap shard metric vector length mismatch")
                sampled[family][metric].extend(float(value) for value in values)
        shard_execution.append(dict(row.get("execution") or {}))
        cursor = stop
    _require(cursor == samples, "bootstrap shards do not cover all frozen draws")

    baseline = str(reference["baseline_family"])
    _require(baseline in families and set(point_estimates) == set(families), "bootstrap family binding failed")
    comparisons: dict[str, Any] = {}
    for family in families:
        if family == baseline:
            continue
        metric_rows = {}
        for metric in _METRIC_KEYS:
            delta = np.asarray(sampled[family][metric], dtype=np.float64) - np.asarray(
                sampled[baseline][metric], dtype=np.float64
            )
            lower, upper = exact_interval(
                delta,
                lower_rank=int(reference["lower_rank"]),
                upper_rank=int(reference["upper_rank"]),
            )
            metric_rows[metric] = {
                "delta_samples": [float(value) for value in delta],
                "delta_mean": float(delta.mean()),
                "ci_lower_exact_rank": lower,
                "ci_upper_exact_rank": upper,
            }
        comparisons[family] = metric_rows

    payload = {
        key: reference[key]
        for key in _IDENTITY_KEYS
        if key != "official_evaluator_reexecuted_per_resample"
    }
    return {
        "schema_version": "duca_h65_official_pcg64_video_bootstrap_v1",
        "official_evaluator_reexecuted_per_resample": True,
        **payload,
        "execution": {
            "engine": "sharded_official_compute_average_precision_detection_in_memory_v1",
            "result_order": "sample_start_then_executor_map_input_order",
            "shard_count": len(ordered),
            "shards": shard_execution,
            "shard_paths": list(shard_paths or ()),
            "shard_sha256": list(shard_sha256 or ()),
        },
        "point_estimates": dict(point_estimates),
        "sampled_metrics": sampled,
        "comparisons": comparisons,
    }


def merge_bootstrap_shards(paths: Sequence[str | Path]) -> dict[str, Any]:
    resolved = [Path(path).resolve() for path in paths]
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in resolved]
    reference = shards[0]
    point_estimates = {
        family: recompute_official_map(
            reference["prediction_paths"][family],
            reference["evaluation_config"],
            expected_subset="validation",
        )["metrics"]
        for family in reference["family_order"]
    }
    return merge_bootstrap_shard_payloads(
        shards,
        point_estimates=point_estimates,
        shard_paths=[str(path) for path in resolved],
        shard_sha256=[sha256_file(path) for path in resolved],
    )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Merge exact contiguous DUCA H65 bootstrap shards"
    )
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    atomic_write_json(args.output, merge_bootstrap_shards(args.shard))


if __name__ == "__main__":
    main()
