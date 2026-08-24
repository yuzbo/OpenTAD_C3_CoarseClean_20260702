from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from tools.bata.duca_p0_evaluation import (
    EXPECTED_TIOU_THRESHOLDS,
    _metrics_from_evaluator,
    canonical_sha256,
    evaluation_video_ids,
    normalize_evaluation_config,
    official_evaluator_identity,
    prediction_results,
    recompute_official_map,
    sha256_file,
)
from tools.bata.duca_p0_training import atomic_write_json


_WORKER_STATE: dict[str, Any] | None = None
_WORKER_DIRECTORY: tempfile.TemporaryDirectory[str] | None = None
_METRIC_KEYS = ("average_mAP", "mAP@0.3", "mAP@0.4", "mAP@0.5", "mAP@0.6", "mAP@0.7")


def seed_from_nonce(nonce: str, namespace: str) -> tuple[int, str]:
    nonce = str(nonce).strip()
    namespace = str(namespace).strip()
    if not nonce or not namespace:
        raise ValueError("bootstrap nonce and namespace must be nonempty")
    digest = hashlib.sha256(f"{nonce}\n{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False), digest.hex()


def exact_interval(values: list[float] | np.ndarray, *, lower_rank: int, upper_rank: int) -> tuple[float, float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.ndim != 1 or ordered.size == 0:
        raise ValueError("exact interval requires a nonempty vector")
    lower_rank = int(lower_rank)
    upper_rank = int(upper_rank)
    if not 1 <= lower_rank <= upper_rank <= ordered.size:
        raise ValueError("exact interval ranks lie outside the bootstrap sample")
    return float(ordered[lower_rank - 1]), float(ordered[upper_rank - 1])


def _evaluate_draw(
    draw: tuple[str, ...],
    *,
    families: tuple[str, ...],
    database: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    evaluation_config: Mapping[str, Any],
    ground_truth_path: Path,
) -> tuple[tuple[float, ...], ...]:
    from opentad.evaluations.mAP import mAP

    synthetic_database: dict[str, Any] = {}
    synthetic_predictions = {family: {} for family in families}
    for draw_index, video_id in enumerate(draw):
        synthetic_id = f"bootstrap_{draw_index:05d}_{video_id}"
        synthetic_database[synthetic_id] = dict(database[video_id])
        for family in families:
            synthetic_predictions[family][synthetic_id] = [
                dict(item) for item in predictions[family].get(video_id, [])
            ]
    ground_truth_path.write_text(
        json.dumps({"database": synthetic_database}, sort_keys=True), encoding="utf-8"
    )
    kwargs = dict(evaluation_config)
    kwargs.pop("type")
    kwargs["ground_truth_filename"] = str(ground_truth_path)
    kwargs["blocked_videos"] = None
    rows = []
    for family in families:
        evaluator = mAP(prediction_filename={"results": synthetic_predictions[family]}, **kwargs)
        metrics = _metrics_from_evaluator(evaluator)
        rows.append(tuple(float(metrics[key]) for key in _METRIC_KEYS))
    return tuple(rows)


def _initialize_worker(
    families: tuple[str, ...],
    database: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    evaluation_config: Mapping[str, Any],
) -> None:
    global _WORKER_DIRECTORY, _WORKER_STATE
    _WORKER_DIRECTORY = tempfile.TemporaryDirectory(prefix="duca-h65-bootstrap-worker-")
    _WORKER_STATE = {
        "families": families,
        "database": database,
        "predictions": predictions,
        "evaluation_config": evaluation_config,
        "ground_truth_path": Path(_WORKER_DIRECTORY.name) / "ground_truth.json",
    }


def _evaluate_draw_in_worker(draw: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    if _WORKER_STATE is None:
        raise RuntimeError("H65 bootstrap worker was not initialized")
    return _evaluate_draw(draw, **_WORKER_STATE)


def bootstrap_h65_official_map(
    prediction_paths: Mapping[str, str | Path],
    evaluation_config: Any,
    *,
    baseline_family: str,
    nonce: str,
    namespace: str,
    samples: int = 10000,
    lower_rank: int = 250,
    upper_rank: int = 9750,
    workers: int = 1,
    chunksize: int = 1,
) -> dict[str, Any]:
    families = tuple(str(key) for key in prediction_paths)
    if baseline_family not in families or len(families) < 2:
        raise ValueError("bootstrap requires a baseline and at least one comparison")
    samples = int(samples)
    if samples != 10000:
        raise ValueError("H65 terminal bootstrap is frozen at exactly 10,000 resamples")
    if (int(lower_rank), int(upper_rank)) != (250, 9750):
        raise ValueError("H65 terminal bootstrap interval is frozen at ranks 250/9750")
    workers = int(workers)
    if workers < 1 or workers > 64:
        raise ValueError("bootstrap workers must lie in [1,64]")
    chunksize = int(chunksize)
    if chunksize < 1:
        raise ValueError("bootstrap chunksize must be positive")

    cfg = normalize_evaluation_config(evaluation_config, expected_subset="validation")
    evaluator_thread = int(cfg["thread"])
    if evaluator_thread < 1:
        raise ValueError("official evaluator thread count must be positive")
    if workers > 1 and evaluator_thread != 1:
        raise ValueError(
            "parallel bootstrap workers require evaluator thread=1 to avoid nested process oversubscription"
        )
    expected = evaluation_video_ids(cfg, expected_subset="validation")
    annotation = json.loads(Path(cfg["ground_truth_filename"]).read_text(encoding="utf-8"))
    database = annotation.get("database")
    if not isinstance(database, Mapping) or any(video not in database for video in expected):
        raise ValueError("bootstrap annotation does not cover the validation videos")
    predictions = {family: prediction_results(path) for family, path in prediction_paths.items()}
    expected_set = set(expected)
    for family, rows in predictions.items():
        extras = set(rows) - expected_set
        if extras:
            raise ValueError(f"{family} prediction contains out-of-scope videos: {sorted(extras)[:4]}")

    seed, seed_sha256 = seed_from_nonce(nonce, namespace)
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.integers(0, len(expected), size=(samples, len(expected)), dtype=np.int32)
    draws = [tuple(expected[int(index)] for index in row) for row in indices]
    sampled = {
        family: {metric: [] for metric in _METRIC_KEYS}
        for family in families
    }

    if workers == 1:
        with tempfile.TemporaryDirectory(prefix="duca-h65-bootstrap-") as directory:
            kwargs = {
                "families": families,
                "database": database,
                "predictions": predictions,
                "evaluation_config": cfg,
                "ground_truth_path": Path(directory) / "ground_truth.json",
            }
            iterator = (_evaluate_draw(draw, **kwargs) for draw in draws)
            _collect(iterator, sampled, families, samples)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(families, database, predictions, cfg),
        ) as executor:
            iterator = executor.map(
                _evaluate_draw_in_worker,
                draws,
                chunksize=chunksize,
            )
            _collect(iterator, sampled, families, samples)

    comparisons: dict[str, Any] = {}
    for family in families:
        if family == baseline_family:
            continue
        metric_rows = {}
        for metric in _METRIC_KEYS:
            delta = np.asarray(sampled[family][metric]) - np.asarray(sampled[baseline_family][metric])
            lower, upper = exact_interval(delta, lower_rank=lower_rank, upper_rank=upper_rank)
            metric_rows[metric] = {
                "delta_samples": [float(value) for value in delta],
                "delta_mean": float(delta.mean()),
                "ci_lower_exact_rank": lower,
                "ci_upper_exact_rank": upper,
            }
        comparisons[family] = metric_rows

    point_estimates = {
        family: recompute_official_map(path, cfg, expected_subset="validation")["metrics"]
        for family, path in prediction_paths.items()
    }
    return {
        "schema_version": "duca_h65_official_pcg64_video_bootstrap_v1",
        "official_evaluator_reexecuted_per_resample": True,
        "paired_video_cluster_bootstrap": True,
        "rng": "numpy.random.PCG64",
        "nonce": nonce,
        "namespace": namespace,
        "seed_uint64": seed,
        "seed_sha256": seed_sha256,
        "samples": samples,
        "interval_rank_convention": "one_based_order_statistics",
        "lower_rank": int(lower_rank),
        "upper_rank": int(upper_rank),
        "baseline_family": baseline_family,
        "family_order": list(families),
        "video_ids": list(expected),
        "prediction_paths": {key: str(Path(value).resolve()) for key, value in prediction_paths.items()},
        "prediction_sha256": {key: sha256_file(value) for key, value in prediction_paths.items()},
        "evaluation_config": cfg,
        "evaluation_config_sha256": canonical_sha256(cfg),
        "execution": {
            "workers": workers,
            "evaluator_thread": evaluator_thread,
            "chunksize": chunksize,
            "result_order": "executor_map_input_order",
        },
        "evaluator": official_evaluator_identity(),
        "point_estimates": point_estimates,
        "sampled_metrics": sampled,
        "comparisons": comparisons,
    }


def _collect(iterator, sampled, families, samples):
    for sample_index, rows in enumerate(iterator, start=1):
        for family, values in zip(families, rows):
            for metric, value in zip(_METRIC_KEYS, values):
                sampled[family][metric].append(float(value))
        if sample_index % 1000 == 0 or sample_index == samples:
            print(f"[DUCA_H65_BOOTSTRAP] {sample_index}/{samples}", flush=True)


def _parse_prediction(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be FAMILY=PATH")
    family, path = value.split("=", 1)
    if not family or not path:
        raise argparse.ArgumentTypeError("prediction must be FAMILY=PATH")
    return family, path


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Exact H65 paired official-mAP bootstrap")
    parser.add_argument("--prediction", action="append", type=_parse_prediction, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--evaluator-thread", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    prediction_paths = dict(args.prediction)
    if len(prediction_paths) != len(args.prediction):
        raise ValueError("prediction family names must be unique")
    evaluation_config = {
        "type": "mAP",
        "ground_truth_filename": args.annotation,
        "subset": "validation",
        "tiou_thresholds": EXPECTED_TIOU_THRESHOLDS,
        "top_k": None,
        "blocked_videos": None,
        "thread": args.evaluator_thread,
    }
    payload = bootstrap_h65_official_map(
        prediction_paths,
        evaluation_config,
        baseline_family=args.baseline,
        nonce=args.nonce,
        namespace=args.namespace,
        workers=args.workers,
        chunksize=args.chunksize,
    )
    atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
