from pathlib import Path

import numpy as np
import pytest

from tools.bata.bootstrap_duca_h65_official_map import (
    _evaluate_draw,
    _evaluate_draw_in_memory,
    bootstrap_h65_official_map,
    exact_interval,
    parse_args,
    resolve_sample_range,
    seed_from_nonce,
)
from tools.bata.merge_duca_h65_bootstrap_shards import (
    merge_bootstrap_shard_payloads,
)


LAUNCHER = "scripts/run_duca_h65_old_pair_bootstrap_n16r4.sbatch"
TERMINAL_LAUNCHER = "scripts/run_duca_h65_singleclock_terminal_eval_n16r4.sbatch"
SHARD_LAUNCHER = "scripts/run_duca_h65_old_pair_bootstrap_shard_n16r4.sbatch"
MERGE_LAUNCHER = "scripts/run_duca_h65_old_pair_bootstrap_merge_n16r4.sbatch"
SHARDED_SUBMITTER = "scripts/submit_duca_h65_old_pair_bootstrap_sharded_n16r4.sh"


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
    assert "--evaluator-thread 16" in text
    assert "--chunksize 1" in text
    assert "input_identity.json" in text


def test_singleclock_launchers_freeze_official_evaluator_metadata():
    text = Path(TERMINAL_LAUNCHER).read_text()
    assert text.count("--evaluator-thread 16") == 3
    assert text.count("--chunksize 1") == 3


def test_sharded_launcher_freezes_sixteen_ranges_and_afterok_merge():
    shard = Path(SHARD_LAUNCHER).read_text()
    merge = Path(MERGE_LAUNCHER).read_text()
    submitter = Path(SHARDED_SUBMITTER).read_text()
    assert "#SBATCH --array=0-15" in shard
    assert "SHARD_COUNT=16" in shard
    assert "DUCA_BOOTSTRAP_SHARD_COUNT" not in shard
    assert "SHARD_COUNT=16" in merge
    assert "DUCA_BOOTSTRAP_SHARD_COUNT" not in merge
    assert 'afterok:$array_job' in submitter
    assert submitter.count("--export=ALL") == 2
    assert "run_duca_h65_old_pair_bootstrap_shard_n16r4.sbatch" in submitter
    assert "run_duca_h65_old_pair_bootstrap_merge_n16r4.sbatch" in submitter
    assert "old-pair input identity schema mismatch" in merge
    assert "rankpack prediction identity mismatch" in merge
    assert "truetime prediction identity mismatch" in merge
    assert "official evaluator identity mismatch" in merge


def test_cli_preserves_official_evaluator_metadata_and_unit_chunks():
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
    assert args.evaluator_thread == 16
    assert args.chunksize == 1
    assert args.sample_start == 0
    assert args.sample_stop is None


def test_frozen_draw_matrix_can_only_be_split_into_nonempty_half_open_shards():
    assert resolve_sample_range(10000) == (0, 10000)
    assert resolve_sample_range(10000, 625, 1250) == (625, 1250)
    for start, stop in ((-1, 10), (10, 10), (10, 10001)):
        with pytest.raises(ValueError, match="0 <= start < stop <= samples"):
            resolve_sample_range(10000, start, stop)


def test_bootstrap_shard_merge_preserves_draw_order_and_exact_rank_contract():
    families = ["baseline", "candidate"]
    metrics = ["average_mAP", "mAP@0.3", "mAP@0.4", "mAP@0.5", "mAP@0.6", "mAP@0.7"]
    common = {
        "schema_version": "duca_h65_official_pcg64_video_bootstrap_shard_v1",
        "official_evaluator_reexecuted_per_resample": True,
        "paired_video_cluster_bootstrap": True,
        "rng": "numpy.random.PCG64",
        "nonce": "nonce",
        "namespace": "namespace",
        "seed_uint64": 7,
        "seed_sha256": "a" * 64,
        "samples": 10000,
        "interval_rank_convention": "one_based_order_statistics",
        "lower_rank": 250,
        "upper_rank": 9750,
        "baseline_family": "baseline",
        "family_order": families,
        "video_ids": ["video_a", "video_b"],
        "prediction_paths": {"baseline": "/b.json", "candidate": "/c.json"},
        "prediction_sha256": {"baseline": "b" * 64, "candidate": "c" * 64},
        "evaluation_config": {"subset": "validation"},
        "evaluation_config_sha256": "d" * 64,
        "evaluator": {"source_sha256": "e" * 64},
        "execution": {"workers": 8},
    }

    def shard(start, stop):
        values = np.arange(start, stop, dtype=np.float64) / 10000.0
        sampled = {
            "baseline": {metric: values.tolist() for metric in metrics},
            "candidate": {metric: (values + 0.01).tolist() for metric in metrics},
        }
        return {
            **common,
            "sample_start": start,
            "sample_stop": stop,
            "shard_samples": stop - start,
            "sampled_metrics": sampled,
        }

    merged = merge_bootstrap_shard_payloads(
        [shard(5000, 10000), shard(0, 5000)],
        point_estimates={"baseline": {"average_mAP": 0.5}, "candidate": {"average_mAP": 0.51}},
    )
    assert merged["samples"] == 10000
    assert len(merged["sampled_metrics"]["candidate"]["mAP@0.7"]) == 10000
    delta = merged["comparisons"]["candidate"]["average_mAP"]
    assert delta["delta_mean"] == pytest.approx(0.01)
    assert delta["ci_lower_exact_rank"] == pytest.approx(0.01)
    assert delta["ci_upper_exact_rank"] == pytest.approx(0.01)


def test_bootstrap_shard_merge_rejects_gaps_before_computing_statistics():
    base = {
        "schema_version": "duca_h65_official_pcg64_video_bootstrap_shard_v1",
        "official_evaluator_reexecuted_per_resample": True,
        "paired_video_cluster_bootstrap": True,
        "rng": "numpy.random.PCG64",
        "nonce": "n",
        "namespace": "s",
        "seed_uint64": 1,
        "seed_sha256": "a" * 64,
        "samples": 10000,
        "interval_rank_convention": "one_based_order_statistics",
        "lower_rank": 250,
        "upper_rank": 9750,
        "baseline_family": "baseline",
        "family_order": ["baseline", "candidate"],
        "video_ids": ["v"],
        "prediction_paths": {"baseline": "/b", "candidate": "/c"},
        "prediction_sha256": {"baseline": "b" * 64, "candidate": "c" * 64},
        "evaluation_config": {},
        "evaluation_config_sha256": "d" * 64,
        "evaluator": {},
        "sample_start": 1,
        "sample_stop": 10000,
        "shard_samples": 9999,
        "sampled_metrics": {
            family: {metric: [0.0] * 9999 for metric in ("average_mAP", "mAP@0.3", "mAP@0.4", "mAP@0.5", "mAP@0.6", "mAP@0.7")}
            for family in ("baseline", "candidate")
        },
    }
    with pytest.raises(ValueError, match="coverage is not contiguous"):
        merge_bootstrap_shard_payloads(
            [base], point_estimates={"baseline": {}, "candidate": {}}
        )


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


def test_in_memory_official_ap_is_exactly_equal_to_legacy_json_path(tmp_path):
    database = {
        "video_a": {
            "subset": "validation",
            "duration": 10.0,
            "annotations": [
                {"segment": [1.0, 3.0], "label": "Action"},
                {"segment": [1.0, 3.0], "label": "Action"},
            ],
        },
        "video_b": {
            "subset": "validation",
            "duration": 10.0,
            "annotations": [{"segment": [4.0, 6.0], "label": "Other"}],
        },
    }
    predictions = {
        "baseline": {
            "video_a": [
                {"segment": [1.0, 3.0], "label": "Action", "score": 0.8},
                {"segment": [0.0, 1.0], "label": "Action", "score": 0.8},
            ],
            "video_b": [{"segment": [4.0, 6.0], "label": "Other", "score": 0.9}],
        },
        "candidate": {
            "video_a": [{"segment": [1.0, 3.0], "label": "Action", "score": 0.8}],
            "video_b": [
                {"segment": [4.0, 6.0], "label": "Other", "score": 0.9},
                {"segment": [2.0, 3.0], "label": "Unknown", "score": 0.7},
            ],
        },
    }
    config = {
        "type": "mAP",
        "ground_truth_filename": "unused",
        "subset": "validation",
        "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        "top_k": None,
        "blocked_videos": None,
        "thread": 16,
    }
    draws = (
        ("video_b", "video_a", "video_a"),
        ("video_a", "video_b", "video_b"),
    )
    for draw_index, draw in enumerate(draws):
        legacy = _evaluate_draw(
            draw,
            families=("baseline", "candidate"),
            database=database,
            predictions=predictions,
            evaluation_config=config,
            ground_truth_path=tmp_path / f"legacy_ground_truth_{draw_index}.json",
        )
        in_memory = _evaluate_draw_in_memory(
            draw,
            families=("baseline", "candidate"),
            database=database,
            predictions=predictions,
            evaluation_config=config,
        )
        assert in_memory == legacy
