from __future__ import annotations

import random
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from opentad.evaluations.mAP import compute_average_precision_detection
from opentad.models.chronotransport import formal_gate4, gate4 as gate4_statistics
from opentad.models.chronotransport.formal_gate4 import (
    GATE4_ENERGY_ARM_ORDER_BY_SEED,
    _single_invocation_action_sha256,
    _validate_population_rows,
    adjudicate_formal_gate4,
    build_gate4_seed_shard,
    validate_gate4_seed_shard,
)
from opentad.models.chronotransport.actions import LayerGroup
from opentad.models.chronotransport.gate4 import (
    _bootstrap_map_pair,
    _map_at,
    _metric_bootstrap_distributions,
    _prepare_bootstrap_map_cache,
    _rows_by_source,
)
from opentad.models.chronotransport.gates23 import SCHEDULER_EPSILON
from opentad.models.chronotransport.profiler import ChronoProfiler
from opentad.models.chronotransport.scheduler import ScheduleLibrary
from tools.bata.chronotransport_r2_gate4_factory import (
    _OfficialPopulationPipeline,
    _reject_config_overrides,
    build_formal_gate4_seed_shard,
    precheck_formal_gate4_seed,
)


def _sha(character: str) -> str:
    return character * 64


def _energy_samples() -> list[dict[str, float]]:
    return [
        {"offset_ms": float(offset), "power_w": 100.0}
        for offset in range(0, 3401, 100)
    ]


def _energy_blocks(*, short_first: bool = False) -> list[dict[str, object]]:
    intervals = [(100.0, 1100.0), (1200.0, 2200.0), (2300.0, 3300.0)]
    if short_first:
        intervals[0] = (100.0, 900.0)
    rows = []
    for index, (arm, interval) in enumerate(
        zip(GATE4_ENERGY_ARM_ORDER_BY_SEED[3407], intervals)
    ):
        start, end = interval
        rows.append(
            {
                "arm": arm,
                "invocation_count": 200,
                "invocation_order_sha256": _sha("1"),
                "start_ms": start,
                "end_ms": end,
                "duration_ms": end - start,
                "energy_j": (end - start) / 10.0,
                "post_nms_prediction_sha256": _sha(str(index + 2)),
            }
        )
    return rows


def _seed_shard_kwargs() -> dict[str, object]:
    stage_c_binding = {
        "completion_artifact_sha256": _sha("a"),
        "checkpoint_file_sha256": _sha("b"),
        "checkpoint_provenance_sha256": _sha("c"),
        "predictor_canonical_sha256": _sha("d"),
        "fit_baseline_payload_sha256": _sha("e"),
    }
    scheduler = {
        "budget": 10.0,
        "epsilon": float(SCHEDULER_EPSILON),
        "calibration_frozen_static": "periodic4_hold",
        "q_conf_by_seed": {"3407": 0.1, "3408": 0.2, "3409": 0.3},
        "gate1_unlock_artifact_sha256": _sha("f"),
        "calibration_sha256": _sha("0"),
    }
    return {
        "seed": 3407,
        "registration_sha256": _sha("1"),
        "registration_commit": "2" * 40,
        "population_artifact_sha256": _sha("3"),
        "post_stage_c_gate3_unlock_sha256": _sha("4"),
        "stage_c_binding": stage_c_binding,
        "scheduler_contract": scheduler,
        "observed_environment": {"observed_environment_sha256": _sha("5")},
        "power_sampling_hz": 10.0,
        "power_samples": _energy_samples(),
        "energy_arm_order": list(GATE4_ENERGY_ARM_ORDER_BY_SEED[3407]),
        "energy_blocks": _energy_blocks(),
        "timing_rows": [],
        "execution_audit": [],
        "predictions": {"dense": [], "chronotransport": [], "static": []},
        "regret_rows": [],
    }


def test_seed_shard_binds_only_long_population_energy_blocks() -> None:
    shard = build_gate4_seed_shard(**_seed_shard_kwargs())
    assert validate_gate4_seed_shard(shard) == shard
    assert [row["arm"] for row in shard["energy_blocks"]] == list(
        GATE4_ENERGY_ARM_ORDER_BY_SEED[3407]
    )
    assert [row["energy_j"] for row in shard["energy_blocks"]] == [
        100.0,
        100.0,
        100.0,
    ]

    tampered = deepcopy(shard)
    tampered["energy_blocks"][0]["energy_j"] = 1.0
    with pytest.raises(ValueError, match="trace integration"):
        validate_gate4_seed_shard(tampered)


def test_seed_shard_rejects_single_inference_energy_semantics() -> None:
    kwargs = _seed_shard_kwargs()
    kwargs["energy_blocks"] = _energy_blocks(short_first=True)
    with pytest.raises(ValueError, match="long ordered interval"):
        build_gate4_seed_shard(**kwargs)


def test_seed_shard_rejects_persistently_sub_10hz_power_trace() -> None:
    kwargs = _seed_shard_kwargs()
    kwargs["power_samples"] = [
        {"offset_ms": float(offset), "power_w": 100.0}
        for offset in range(0, 3401, 200)
    ]
    with pytest.raises(ValueError, match="10-Hz cadence"):
        build_gate4_seed_shard(**kwargs)


def test_gate4_runtime_action_hash_normalizes_only_batch_size_one() -> None:
    actions = ScheduleLibrary.r2(
        layer_groups=(LayerGroup(0, 4), LayerGroup(4, 8), LayerGroup(8, 12))
    ).canonical_payload()["candidates"][0]
    assert _single_invocation_action_sha256([actions["actions"]]) == actions[
        "action_sha256"
    ]
    with pytest.raises(ValueError, match="batch size one"):
        _single_invocation_action_sha256([actions["actions"], actions["actions"]])


def test_formal_adjudicator_revalidates_clean_registration_context(monkeypatch) -> None:
    def reject_context(*_args, **_kwargs):
        raise RuntimeError("FORMAL_CONTEXT_REVALIDATED")

    monkeypatch.setattr(formal_gate4, "validate_formal_gate1_context", reject_context)
    with pytest.raises(RuntimeError, match="FORMAL_CONTEXT_REVALIDATED"):
        adjudicate_formal_gate4(
            timing_evidence={},
            metric_evidence={},
            regret_evidence={},
            registration={},
            population={},
            post_stage_c_unlock={},
            post_stage_c_report={},
            post_stage_c_replay={},
            gate1_unlock={},
            seed_shards={},
            repository_root="/registered/repository",
            registration_commit="a" * 40,
            registration_relpath="registration.json",
        )


def test_gate4_precheck_and_seed_producer_share_config_override_lock(monkeypatch) -> None:
    monkeypatch.setenv("CHRONOTRANSPORT_MODE", "dense")
    with pytest.raises(RuntimeError, match="forbids config overrides"):
        _reject_config_overrides()
    kwargs = {
        "registration_path": None,
        "registration": {},
        "registration_commit": "",
        "registration_relpath": "",
        "gate1_unlock": {},
        "gate1_unlock_path": None,
        "pre_stage_c_gates23_replay_path": None,
        "pre_stage_c_gates23_report_path": None,
        "phase_marker_paths": {},
        "post_stage_c_replay": {},
        "post_stage_c_report": {},
        "post_stage_c_unlock": {},
        "seed": 3407,
    }
    for producer in (precheck_formal_gate4_seed, build_formal_gate4_seed_shard):
        with pytest.raises(RuntimeError, match="forbids config overrides"):
            producer(**kwargs)


def test_gate4_materialization_rechecks_the_same_media_descriptor_after_decode() -> None:
    class MutableDescriptor:
        path = "/registered/media/video.mp4"
        proc_path = "/proc/self/fd/17"

        def __init__(self) -> None:
            self.calls = 0
            self.changed = False

        def assert_stable(self) -> None:
            self.calls += 1
            if self.changed:
                raise RuntimeError("media changed during decode")

    media = MutableDescriptor()
    pipeline = _OfficialPopulationPipeline.__new__(_OfficialPopulationPipeline)
    pipeline.invocations = {
        "video/window-0": {
            "official_video_id": "video",
            "sampled_frame_indices": [0, 1],
            "valid_count": 2,
        }
    }
    pipeline.records = {"video": {"frame": 10, "duration": 1.0}}
    pipeline.population = {"dataset_contract": {"feature_stride": 1}}
    pipeline.ground_truth_by_video = {"video": []}
    pipeline._media = {"video": media}

    def decode(sample):
        media.changed = True
        return {
            **sample,
            "total_frames": 10,
            "avg_fps": 10.0,
            "frame_inds": np.asarray([0, 1]),
        }

    pipeline.decode_pipeline = decode
    pipeline.test_preprocess = lambda sample: {
        **sample,
        "masks": np.asarray([True, True] + [False] * 766),
    }
    pipeline.regret_preprocess = pipeline.test_preprocess

    with pytest.raises(RuntimeError, match="changed during decode"):
        pipeline.materialize("video/window-0", regret=False)
    assert media.calls == 2


def test_deferred_cuda_profiler_never_synchronizes_inside_summary(monkeypatch) -> None:
    events = []

    class FakeEvent:
        def __init__(self, *, enable_timing: bool):
            assert enable_timing is True
            self.synchronize_count = 0
            events.append(self)

        def record(self) -> None:
            return None

        def synchronize(self) -> None:
            self.synchronize_count += 1

        def elapsed_time(self, _other) -> float:
            return 2.5

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", FakeEvent)
    profiler = ChronoProfiler(sync_cuda=False, deferred_cuda_events=True)
    with profiler.stage("scheduler"):
        pass

    with pytest.raises(RuntimeError, match="outer timing boundary"):
        profiler.summary(flush_deferred=False)
    assert sum(event.synchronize_count for event in events) == 0

    profiler.flush_deferred_cuda_events(synchronize=False)
    summary = profiler.summary(fill_missing=False)
    assert summary["latency_ms"]["scheduler"]["total"] == 2.5
    assert sum(event.synchronize_count for event in events) == 0


def _population_audit_fixture():
    library = ScheduleLibrary.r2(
        layer_groups=(LayerGroup(0, 4), LayerGroup(4, 8), LayerGroup(8, 12))
    ).canonical_payload()
    candidates = {row["name"]: row for row in library["candidates"]}
    scheduler = {
        "budget": 10.0,
        "epsilon": float(SCHEDULER_EPSILON),
        "calibration_frozen_static": "periodic4_hold",
        "q_conf_by_seed": {"3407": 0.1, "3408": 0.2, "3409": 0.3},
        "gate1_unlock_artifact_sha256": _sha("a"),
        "calibration_sha256": _sha("b"),
    }
    block = {
        "official_video_id": "video",
        "invocation_id": "video/window-0",
        "repetition_id": 0,
        "invocation_order_index": 0,
        "arm_order": ["dense", "chronotransport", "static"],
    }
    population = {
        "timing_blocks": [block],
        "official_video_ids": ["video"],
        "ground_truth": [
            {"official_video_id": "video", "label": "action", "segment": [1.0, 2.0]}
        ],
        "fit_duration_quartile_thresholds": [1.0, 2.0, 3.0],
        "unique_invocations": [
            {"official_video_id": "video", "invocation_id": "video/window-0"}
        ],
    }

    def counts(name: str) -> dict[str, int]:
        actions = candidates[name]["actions"]
        return {
            "recompute_rows": 4 * sum(value == 0 for row in actions for value in row),
            "transport_rows": 4 * sum(value == 1 for row in actions for value in row),
            "hold_rows": 4 * sum(value == 2 for row in actions for value in row),
        }

    audits = []
    for seed in (3407, 3408, 3409):
        for arm, selected, fail_closed in (
            ("dense", "dense", False),
            ("chronotransport", "dense", True),
            ("static", "periodic4_hold", False),
        ):
            candidate = candidates[selected]
            audits.append(
                {
                    "seed": seed,
                    "invocation_id": "video/window-0",
                    "repetition_id": 0,
                    "arm": arm,
                    "selected_schedule": selected,
                    "requested_action_sha256": candidate["action_sha256"],
                    "executed_action_sha256": candidate["action_sha256"],
                    **counts(selected),
                    "schedule_repair_count": 0,
                    "runtime_fail_closed_repairs": 0,
                    "whole_window_dense_fallback": False,
                    "upper_risk": 0.0,
                    "estimated_cost": 10.0,
                    "registered_gate3_calibration_sha256": scheduler[
                        "calibration_sha256"
                    ],
                    "registered_q_conf": scheduler["q_conf_by_seed"][str(seed)],
                    "registered_budget": scheduler["budget"],
                    "evidence_valid": True,
                    "fail_closed": fail_closed,
                }
            )
    timing = {
        "scheduler_contract": scheduler,
        "rows": [
            {"seed": seed, **block}
            for seed in (3407, 3408, 3409)
        ],
        "execution_audit": audits,
    }
    metric = {
        "metric_evidence": {
            "official_video_ids": population["official_video_ids"],
            "ground_truth": population["ground_truth"],
            "fit_duration_quartile_thresholds": population[
                "fit_duration_quartile_thresholds"
            ],
        }
    }
    regret = {
        "rows": [
            {
                "seed": seed,
                "official_video_id": "video",
                "invocation_id": "video/window-0",
            }
            for seed in (3407, 3408, 3409)
        ]
    }
    return timing, metric, regret, population, {"candidate_library": library}


def test_formal_population_audit_binds_executed_actions_to_registration() -> None:
    timing, metric, regret, population, registration = _population_audit_fixture()
    _validate_population_rows(
        timing,
        metric,
        regret,
        population=population,
        registration=registration,
    )

    tampered = deepcopy(timing)
    tampered["execution_audit"][0]["executed_action_sha256"] = _sha("f")
    with pytest.raises(ValueError, match="registered executed schedule"):
        _validate_population_rows(
            tampered,
            metric,
            regret,
            population=population,
            registration=registration,
        )

    nonfinite = deepcopy(timing)
    nonfinite["execution_audit"][1]["upper_risk"] = float("nan")
    with pytest.raises(ValueError, match="registered executed schedule"):
        _validate_population_rows(
            nonfinite,
            metric,
            regret,
            population=population,
            registration=registration,
        )


def test_numpy_gate4_map_matches_official_opentad_for_ties_and_resampling(
    monkeypatch,
) -> None:
    ground_truth = [
        ("v0", "a", 0.0, 1.0),
        ("v0", "b", 2.0, 3.5),
        ("v1", "a", 0.5, 1.8),
        ("v1", "b", 3.0, 4.0),
        ("v2", "a", 1.0, 2.0),
        ("v2", "b", 4.0, 5.0),
    ]
    predictions = [
        ("v0", "a", 0.0, 1.0, 0.5),
        ("v0", "a", 1.2, 1.9, 0.5),
        ("v0", "b", 2.1, 3.4, 0.8),
        ("v1", "a", 0.4, 1.7, 0.8),
        ("v1", "b", 2.8, 4.1, 0.5),
        ("v2", "a", 0.9, 2.1, 0.5),
        ("v2", "b", 5.2, 6.0, 0.8),
    ]
    sampled = ["v1", "v0", "v1", "v2"]

    synthetic_gt = []
    synthetic_predictions = []
    for position, source in enumerate(sampled):
        synthetic = f"boot/{position}/{source}"
        synthetic_gt.extend(
            (synthetic, *row[1:]) for row in ground_truth if row[0] == source
        )
        synthetic_predictions.extend(
            (synthetic, *row[1:]) for row in predictions if row[0] == source
        )
    expected_by_threshold = {}
    for threshold in (0.3, 0.7):
        aps = []
        for label in sorted({row[1] for row in synthetic_gt}):
            class_gt = [row for row in synthetic_gt if row[1] == label]
            class_predictions = [
                row for row in synthetic_predictions if row[1] == label
            ]
            gt_frame = pd.DataFrame(
                [
                    {"video-id": row[0], "t-start": row[2], "t-end": row[3]}
                    for row in class_gt
                ]
            )
            prediction_frame = pd.DataFrame(
                [
                    {
                        "video-id": row[0],
                        "t-start": row[2],
                        "t-end": row[3],
                        "score": row[4],
                    }
                    for row in class_predictions
                ]
            )
            aps.append(
                float(
                    compute_average_precision_detection(
                        gt_frame,
                        prediction_frame,
                        tiou_thresholds=np.asarray([threshold], dtype=np.float64),
                    )[0]
                )
            )
        expected_by_threshold[threshold] = float(np.mean(aps))

    for threshold, expected in expected_by_threshold.items():
        assert _map_at(
            ground_truth,
            predictions,
            sampled,
            tiou_threshold=threshold,
        ) == pytest.approx(expected, rel=0.0, abs=1e-15)

    cache = _prepare_bootstrap_map_cache(
        _rows_by_source(ground_truth),
        _rows_by_source(predictions),
        ["v0", "v1", "v2"],
        tiou_threshold=0.7,
        q1_threshold=1.1,
    )
    sampled_layouts = [
        sampled,
        ["v0", "v0", "v1"],
        ["v2", "v0", "v0", "v1", "v2"],
    ]
    expected_cached = [
        (
            _map_at(
                ground_truth,
                predictions,
                layout,
                tiou_threshold=0.7,
            ),
            _map_at(
                ground_truth,
                predictions,
                layout,
                tiou_threshold=0.7,
                q1_threshold=1.1,
            ),
        )
        for layout in sampled_layouts
    ]

    def reject_slow_fallback(*_args, **_kwargs):
        raise AssertionError("same-video ties must stay on the exact cached path")

    monkeypatch.setattr(
        gate4_statistics, "_official_ap_numpy", reject_slow_fallback
    )
    for index, (layout, expected) in enumerate(zip(sampled_layouts, expected_cached)):
        fast_full, fast_q1 = _bootstrap_map_pair(
            cache, layout, synthetic_prefix=f"boot/{index}"
        )
        assert fast_full == pytest.approx(expected[0], rel=0.0, abs=1e-15)
        assert fast_q1 == pytest.approx(expected[1], rel=0.0, abs=1e-15)
    assert expected_cached[0][0] == pytest.approx(
        expected_by_threshold[0.7], rel=0.0, abs=1e-15
    )


def test_gate4_bootstrap_cache_preserves_cross_video_score_ties(monkeypatch) -> None:
    ground_truth = [
        ("v0", "a", 0.0, 1.0),
        ("v1", "a", 0.0, 1.0),
    ]
    predictions = [
        ("v0", "a", 0.0, 1.0, 0.9),
        ("v0", "a", 2.0, 3.0, 0.4),
        ("v1", "a", 2.0, 3.0, 0.9),
        ("v1", "a", 0.0, 1.0, 0.3),
    ]
    sampled = ["v1", "v0", "v1"]
    expected = _map_at(
        ground_truth, predictions, sampled, tiou_threshold=0.7
    )
    cache = _prepare_bootstrap_map_cache(
        _rows_by_source(ground_truth),
        _rows_by_source(predictions),
        ["v0", "v1"],
        tiou_threshold=0.7,
        q1_threshold=1.1,
    )

    def reject_slow_fallback(*_args, **_kwargs):
        raise AssertionError("cross-video ties must stay on the exact cached path")

    monkeypatch.setattr(
        gate4_statistics, "_official_ap_numpy", reject_slow_fallback
    )
    actual, actual_q1 = _bootstrap_map_pair(
        cache, sampled, synthetic_prefix="boot"
    )
    assert actual == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert actual_q1 == pytest.approx(expected, rel=0.0, abs=1e-15)


def test_gate4_cached_bootstrap_preserves_frozen_rng_and_seed_draws(monkeypatch) -> None:
    videos = ["v0", "v1", "v2"]
    ground_truth = [
        ("v0", "a", 0.0, 1.0),
        ("v1", "a", 0.0, 1.0),
        ("v2", "a", 0.0, 1.5),
        ("v2", "a", 2.0, 3.0),
    ]
    predictions = {}
    caches = {}
    for seed_index, seed in enumerate((3407, 3408, 3409)):
        dense = [
            (video, "a", 0.0, 1.0, 0.9 - 0.01 * index)
            for index, video in enumerate(videos)
        ]
        chronotransport = [
            (
                video,
                "a",
                0.0 if not (seed == 3409 and index == 2) else 2.0,
                1.0 if not (seed == 3409 and index == 2) else 3.0,
                0.8 - 0.01 * index - 0.001 * seed_index,
            )
            for index, video in enumerate(videos)
        ]
        predictions[seed] = {"dense": dense, "chronotransport": chronotransport}
        caches[seed] = {
            arm: _prepare_bootstrap_map_cache(
                _rows_by_source(ground_truth),
                _rows_by_source(rows),
                videos,
                tiou_threshold=0.7,
                q1_threshold=1.1,
            )
            for arm, rows in predictions[seed].items()
        }

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    actual, actual_q1 = _metric_bootstrap_distributions(
        caches,
        videos,
        bootstrap_samples=17,
        bootstrap_seed=20260711,
    )
    rng = random.Random(20260711 ^ 0x7C31)
    expected, expected_q1 = [], []
    for _ in range(17):
        sampled_videos = [rng.choice(videos) for _ in videos]
        sampled_seeds = [
            rng.choice((3407, 3408, 3409)) for _ in (3407, 3408, 3409)
        ]
        dense = np.mean(
            [
                _map_at(
                    ground_truth,
                    predictions[seed]["dense"],
                    sampled_videos,
                    tiou_threshold=0.7,
                )
                for seed in sampled_seeds
            ]
        )
        ct = np.mean(
            [
                _map_at(
                    ground_truth,
                    predictions[seed]["chronotransport"],
                    sampled_videos,
                    tiou_threshold=0.7,
                )
                for seed in sampled_seeds
            ]
        )
        dense_q1 = np.mean(
            [
                _map_at(
                    ground_truth,
                    predictions[seed]["dense"],
                    sampled_videos,
                    tiou_threshold=0.7,
                    q1_threshold=1.1,
                )
                for seed in sampled_seeds
            ]
        )
        ct_q1 = np.mean(
            [
                _map_at(
                    ground_truth,
                    predictions[seed]["chronotransport"],
                    sampled_videos,
                    tiou_threshold=0.7,
                    q1_threshold=1.1,
                )
                for seed in sampled_seeds
            ]
        )
        expected.append(100.0 * float(dense - ct))
        expected_q1.append(100.0 * float(dense_q1 - ct_q1))
    assert actual == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert actual_q1 == pytest.approx(expected_q1, rel=0.0, abs=1e-15)
