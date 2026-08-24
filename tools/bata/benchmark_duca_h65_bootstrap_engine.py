from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from tools.bata.bootstrap_duca_h65_official_map import (
    _evaluate_draw,
    _evaluate_draw_in_memory,
    seed_from_nonce,
)
from tools.bata.duca_p0_evaluation import (
    EXPECTED_TIOU_THRESHOLDS,
    canonical_sha256,
    evaluation_video_ids,
    normalize_evaluation_config,
    prediction_results,
    sha256_file,
)
from tools.bata.duca_p0_training import atomic_write_json


def benchmark_bootstrap_engines(
    *,
    annotation_path: str | Path,
    prediction_paths: dict[str, str | Path],
    nonce: str,
    namespace: str,
    draws: int,
) -> dict:
    draws = int(draws)
    if draws < 1 or draws > 32:
        raise ValueError("benchmark draw count must lie in [1,32]")
    config = normalize_evaluation_config(
        {
            "type": "mAP",
            "ground_truth_filename": str(annotation_path),
            "subset": "validation",
            "tiou_thresholds": EXPECTED_TIOU_THRESHOLDS,
            "top_k": None,
            "blocked_videos": None,
            "thread": 16,
        },
        expected_subset="validation",
    )
    video_ids = evaluation_video_ids(config, expected_subset="validation")
    annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    database = annotation["database"]
    families = tuple(prediction_paths)
    predictions = {
        family: prediction_results(path) for family, path in prediction_paths.items()
    }
    seed, seed_sha256 = seed_from_nonce(nonce, namespace)
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.integers(
        0, len(video_ids), size=(draws, len(video_ids)), dtype=np.int32
    )
    sampled_draws = [
        tuple(video_ids[int(index)] for index in row) for row in indices
    ]

    legacy_rows = []
    with tempfile.TemporaryDirectory(prefix="duca-h65-bootstrap-benchmark-") as directory:
        ground_truth_path = Path(directory) / "ground_truth.json"
        legacy_started = time.perf_counter()
        for draw in sampled_draws:
            legacy_rows.append(
                _evaluate_draw(
                    draw,
                    families=families,
                    database=database,
                    predictions=predictions,
                    evaluation_config=config,
                    ground_truth_path=ground_truth_path,
                )
            )
        legacy_seconds = time.perf_counter() - legacy_started

    memory_started = time.perf_counter()
    memory_rows = [
        _evaluate_draw_in_memory(
            draw,
            families=families,
            database=database,
            predictions=predictions,
            evaluation_config=config,
        )
        for draw in sampled_draws
    ]
    memory_seconds = time.perf_counter() - memory_started
    exact_equal = memory_rows == legacy_rows
    if not exact_equal:
        raise AssertionError("in-memory AP core differs from legacy JSON evaluator path")

    return {
        "schema_version": "duca_h65_bootstrap_engine_benchmark_v1",
        "evidence_class": "EXECUTION_EQUIVALENCE_AND_RUNTIME_ONLY",
        "draws": draws,
        "videos_per_draw": len(video_ids),
        "families": list(families),
        "nonce": nonce,
        "namespace": namespace,
        "seed_uint64": seed,
        "seed_sha256": seed_sha256,
        "annotation_path": str(Path(annotation_path).resolve()),
        "annotation_sha256": sha256_file(annotation_path),
        "prediction_paths": {
            family: str(Path(path).resolve()) for family, path in prediction_paths.items()
        },
        "prediction_sha256": {
            family: sha256_file(path) for family, path in prediction_paths.items()
        },
        "evaluation_config": config,
        "evaluation_config_sha256": canonical_sha256(config),
        "exact_rows_equal": exact_equal,
        "legacy_json_seconds": legacy_seconds,
        "in_memory_seconds": memory_seconds,
        "speedup": legacy_seconds / memory_seconds,
        "legacy_seconds_per_draw": legacy_seconds / draws,
        "in_memory_seconds_per_draw": memory_seconds / draws,
        "science_result_allowed": False,
    }


def _parse_prediction(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be FAMILY=PATH")
    family, path = value.split("=", 1)
    if not family or not path:
        raise argparse.ArgumentTypeError("prediction must be FAMILY=PATH")
    return family, path


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Bounded real-input benchmark for the DUCA H65 bootstrap engines"
    )
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--prediction", action="append", type=_parse_prediction, required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    prediction_paths = dict(args.prediction)
    if len(prediction_paths) != len(args.prediction):
        raise ValueError("prediction family names must be unique")
    payload = benchmark_bootstrap_engines(
        annotation_path=args.annotation,
        prediction_paths=prediction_paths,
        nonce=args.nonce,
        namespace=args.namespace,
        draws=args.draws,
    )
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
