import json
from pathlib import Path

import numpy as np
import pytest

from tools.bata.bootstrap_duca_h65_official_map import (
    _evaluate_draw,
    bootstrap_h65_official_map,
    exact_interval,
    parse_args,
    seed_from_nonce,
)


LAUNCHER = "scripts/run_duca_h65_old_pair_bootstrap_n16r4.sbatch"
TERMINAL_LAUNCHER = "scripts/run_duca_h65_singleclock_terminal_eval_n16r4.sbatch"


def test_pcg64_seed_is_namespaced_deterministic_and_case_sensitive():
    first, first_digest = seed_from_nonce("nonce", "final-ema-on-vs-gate-zero")
    second, second_digest = seed_from_nonce("nonce", "final-ema-on-vs-gate-zero")
    changed, changed_digest = seed_from_nonce("nonce", "final-on-vs-gate-zero")
    assert first == second
    assert first_digest == second_digest
    assert first != changed
    assert first_digest != changed_digest
    assert first == int.from_bytes(
        __import__("hashlib").sha256(b"nonce\nfinal-ema-on-vs-gate-zero").digest()[:8],
        byteorder="big",
        signed=False,
    )
    assert np.random.Generator(np.random.PCG64(first)).integers(0, 100, 8).tolist() == np.random.Generator(
        np.random.PCG64(second)
    ).integers(0, 100, 8).tolist()


def test_exact_interval_uses_frozen_one_based_order_statistics():
    values = list(range(10000, 0, -1))
    lower, upper = exact_interval(values, lower_rank=250, upper_rank=9750)
    assert lower == 250.0
    assert upper == 9750.0
    with pytest.raises(ValueError, match="outside"):
        exact_interval(values, lower_rank=0, upper_rank=9750)


def test_old_pair_launcher_freezes_exact_predictions_and_statistical_nonce():
    text = Path(LAUNCHER).read_text()
    assert "rankpack_k384/run/terminal_eval/gpu1_id0/result_detection.json" in text
    assert "truetime_k384/run/terminal_eval/gpu1_id0/result_detection.json" in text
    assert "DUCA-H65-60-TRUETIME-BRIDGE-DIRECT-v001-20260823" in text
    assert "PAIRED_VIDEO_BOOTSTRAP_V1" in text
    assert "--workers" in text
    assert "--evaluator-thread 1" in text
    assert "--chunksize 1" in text
    assert "input_identity.json" in text


def test_singleclock_launchers_disable_nested_evaluator_parallelism():
    text = Path(TERMINAL_LAUNCHER).read_text()
    assert text.count("--evaluator-thread 1") == 3
    assert text.count("--chunksize 1") == 3


def test_cli_defaults_to_single_thread_evaluator_and_unit_chunks():
    args = parse_args(
        [
            "--prediction", "baseline=/tmp/baseline.json",
            "--prediction", "candidate=/tmp/candidate.json",
            "--baseline", "baseline",
            "--annotation", "/tmp/annotation.json",
            "--nonce", "nonce",
            "--namespace", "namespace",
            "--output", "/tmp/output.json",
        ]
    )
    assert args.workers == 1
    assert args.evaluator_thread == 1
    assert args.chunksize == 1


def test_official_map_metrics_are_identical_for_one_or_sixteen_evaluator_processes(tmp_path):
    database = {
        "video_a": {
            "subset": "validation",
            "duration": 10.0,
            "annotations": [{"segment": [1.0, 3.0], "label": "Action"}],
        },
        "video_b": {
            "subset": "validation",
            "duration": 10.0,
            "annotations": [{"segment": [4.0, 6.0], "label": "Action"}],
        },
    }
    predictions = {
        "baseline": {
            "video_a": [{"segment": [1.0, 3.0], "label": "Action", "score": 0.9}],
            "video_b": [{"segment": [0.0, 1.0], "label": "Action", "score": 0.8}],
        },
        "candidate": {
            "video_a": [{"segment": [1.0, 3.0], "label": "Action", "score": 0.9}],
            "video_b": [{"segment": [4.0, 6.0], "label": "Action", "score": 0.8}],
        },
    }
    base_config = {
        "type": "mAP",
        "ground_truth_filename": "unused",
        "subset": "validation",
        "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        "top_k": None,
        "blocked_videos": None,
    }
    rows = []
    for thread in (1, 16):
        config = {**base_config, "thread": thread}
        rows.append(
            _evaluate_draw(
                ("video_a", "video_b"),
                families=("baseline", "candidate"),
                database=database,
                predictions=predictions,
                evaluation_config=config,
                ground_truth_path=tmp_path / f"ground_truth_thread_{thread}.json",
            )
        )
    assert rows[0] == rows[1]


def test_parallel_bootstrap_rejects_nested_evaluator_processes(tmp_path):
    annotation = tmp_path / "annotation.json"
    prediction_a = tmp_path / "a.json"
    prediction_b = tmp_path / "b.json"
    annotation.write_text(
        json.dumps(
            {
                "database": {
                    "video": {
                        "subset": "validation",
                        "duration": 10.0,
                        "annotations": [{"segment": [1.0, 2.0], "label": "Action"}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "results": {
            "video": [{"segment": [1.0, 2.0], "label": "Action", "score": 1.0}]
        }
    }
    prediction_a.write_text(json.dumps(payload), encoding="utf-8")
    prediction_b.write_text(json.dumps(payload), encoding="utf-8")
    evaluation_config = {
        "type": "mAP",
        "ground_truth_filename": str(annotation),
        "subset": "validation",
        "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        "top_k": None,
        "blocked_videos": None,
        "thread": 16,
    }
    with pytest.raises(ValueError, match="nested process oversubscription"):
        bootstrap_h65_official_map(
            {"baseline": prediction_a, "candidate": prediction_b},
            evaluation_config,
            baseline_family="baseline",
            nonce="nonce",
            namespace="namespace",
            workers=8,
        )
