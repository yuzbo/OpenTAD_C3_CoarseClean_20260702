from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from opentad.evaluations.mAP import compute_average_precision_detection
from opentad.models.chronotransport.gate4 import (
    _map_at,
    adjudicate_gate4,
    validate_gate4_report,
)
from opentad.models.chronotransport.gate4_population import (
    build_gate4_population_artifact,
    gate4_population_exact_bytes,
    validate_gate4_population_artifact,
)
from opentad.models.chronotransport.protocol import canonical_sha256


SEEDS = (3407, 3408, 3409)
ORDERS = (
    ("dense", "chronotransport", "static"),
    ("chronotransport", "static", "dense"),
    ("static", "dense", "chronotransport"),
    ("static", "chronotransport", "dense"),
    ("chronotransport", "dense", "static"),
    ("dense", "static", "chronotransport"),
)


def _arm(total, heavy, *, innovation=0.0, scheduler=0.0, transport=0.0, cache=0.0):
    return {
        "total_ms": float(total),
        "peak_gpu_memory_bytes": 1_000_000,
        "nvml_energy_j": 2.0,
        "stage_ms": {
            "decode_ms": 0.1,
            "preprocess_ms": 0.1,
            "h2d_ms": 0.1,
            "patch_embed_ms": 0.2,
            "heavy_ms": float(heavy),
            "innovation_ms": float(innovation),
            "scheduler_ms": float(scheduler),
            "transport_ms": float(transport),
            "cache_movement_ms": float(cache),
            "adapter_ms": 0.2,
            "head_ms": 0.1,
            "postprocess_ms": 0.1,
        },
    }


def _timing_rows():
    rows = []
    for seed in SEEDS:
        ordinal = 0
        for video in range(20):
            for invocation in range(12):
                jitter = ((video * 12 + invocation) % 7) / 100.0
                invocation_id = f"video-{video:02d}/window-{invocation:02d}"
                rows.append(
                    {
                        "seed": seed,
                        "official_video_id": f"video-{video:02d}",
                        "invocation_id": invocation_id,
                        "repetition_id": 0,
                        "invocation_order_index": ordinal,
                        "arm_order": list(ORDERS[ordinal % 6]),
                        "arms": {
                            "dense": _arm(10.0 + jitter, 8.0 + jitter),
                            "chronotransport": _arm(
                                7.0 + jitter,
                                4.0 + jitter,
                                innovation=0.25,
                                scheduler=0.25,
                                transport=0.25,
                                cache=0.25,
                            ),
                            "static": _arm(8.0 + jitter, 5.0 + jitter),
                        },
                    }
                )
                ordinal += 1
    return rows


def _metric_evidence():
    ground_truth = []
    predictions = {str(seed): {} for seed in SEEDS}
    for video in range(20):
        ground_truth.append(
            {
                "official_video_id": f"video-{video:02d}",
                "label": "action",
                "segment": [1.0, 2.0],
            }
        )
    for seed in SEEDS:
        for arm in ("dense", "chronotransport", "static"):
            predictions[str(seed)][arm] = [
                {
                    "official_video_id": f"video-{video:02d}",
                    "label": "action",
                    "segment": [1.0, 2.0],
                    "score": 0.99 - video / 1000.0,
                }
                for video in range(20)
            ]
    return {
        "schema": "chronotransport-r2-gate4-metric-evidence-v1",
        "official_video_ids": [f"video-{video:02d}" for video in range(20)],
        "fit_duration_quartile_thresholds": [1.5, 2.5, 3.5],
        "ground_truth": ground_truth,
        "predictions": predictions,
    }


def _regret_rows():
    rows = []
    for seed in SEEDS:
        for video in range(20):
            for invocation in range(12):
                rows.append(
                    {
                        "seed": seed,
                        "official_video_id": f"video-{video:02d}",
                        "invocation_id": f"video-{video:02d}/window-{invocation:02d}",
                        "dense_detector_loss": 1.0,
                        "chronotransport_detector_loss": 1.1,
                        "static_detector_loss": 1.4,
                    }
                )
    return rows


def _gate4_population():
    videos = [
        {
            "official_video_id": f"video-{index:02d}",
            "media_path": f"video-{index:02d}.mp4",
            "media_bytes": 1000 + index,
            "media_sha256": canonical_sha256(["media", index]),
            "frame": 16_000,
            "duration": 100.0,
        }
        for index in range(20)
    ]
    contract = {
        "dataset_type": "ThumosSlidingDataset",
        "subset": "validation",
        "test_mode": True,
        "feature_stride": 4,
        "sample_stride": 1,
        "offset_frames": 0,
        "window_size": 768,
        "window_overlap_ratio": 0.5,
        "scale_factor": 1,
        "test_pipeline_sha256": canonical_sha256("test-pipeline"),
        "regret_pipeline_sha256": canonical_sha256("regret-pipeline"),
        "inference_sha256": canonical_sha256("inference"),
        "post_processing_sha256": canonical_sha256("post-processing"),
        "evaluation_sha256": canonical_sha256("evaluation"),
    }
    return build_gate4_population_artifact(
        config_sources_sha256={
            "configs/adatad/thumos/c3_chronotransport_r2_stage_c.py": canonical_sha256(
                "config"
            )
        },
        annotation={"path": "/registered/thumos.json", "sha256": canonical_sha256("ann")},
        class_map={"path": "/registered/classes.txt", "sha256": canonical_sha256("classes")},
        data_root="/registered/thumos/videos",
        dataset_contract=contract,
        videos=videos,
        ground_truth=[
            {
                "official_video_id": row["official_video_id"],
                "label": "action",
                "segment": [1.0, 2.0],
            }
            for row in videos[:-1]
        ],
        fit_manifest_sha256=canonical_sha256("fit-manifest"),
        fit_duration_quartile_thresholds=[1.5, 2.5, 3.5],
    )


def _run(**kwargs):
    return adjudicate_gate4(
        timing_rows=kwargs.pop("timing_rows", _timing_rows()),
        metric_evidence=kwargs.pop("metric_evidence", _metric_evidence()),
        regret_rows=kwargs.pop("regret_rows", _regret_rows()),
        bootstrap_samples=kwargs.pop("bootstrap_samples", 100),
        bootstrap_seed=20260711,
        formal=False,
        **kwargs,
    )


def test_gate4_passes_all_six_hard_conditions_and_keeps_claims_locked():
    result = _run()
    assert result["schema"] == "chronotransport-r2-gate4-test-only-v1"
    assert result["evidence_scope"] == "test_only_unregistered_raw_mappings"
    assert result["formal_evidence"] is False
    assert result["status"] == "PASS"
    assert result["mechanism"] is True
    assert all(result["hard_conditions"].values())
    assert result["latency"]["saving_lcb95"] >= 0.15
    assert result["metrics"]["map07_drop_ucb95_points"] <= 1.5
    assert result["metrics"]["short_q1_drop_ucb95_points"] <= 1.5
    assert result["cost_decomposition"]["median_heavy_saving_ms"] > 0
    assert result["cost_decomposition"]["median_margin_lcb95_ms"] > 0
    assert result["latency"]["ct_minus_static_ucb95_ms"] <= 0
    assert result["regret"]["ct_over_static_improvement_ci95"][0] > 0
    assert result["diagnostics"]["p95_ms"]["dense"] > 0
    assert result["diagnostics"]["throughput_per_second"]["chronotransport"] > 0
    assert result["diagnostics"]["peak_gpu_memory_bytes"]["dense"] == 1_000_000
    assert result["diagnostics"]["median_nvml_block_energy_j"]["dense"] == 2.0
    assert result["diagnostics"]["total_ms_distribution"]["dense"]["count"] == 720
    assert len(
        result["diagnostics"]["total_ms_distribution"]["dense"]["sample_sha256"]
    ) == 64
    assert set(result["diagnostics"]["map_by_tiou"]["dense"]) == {
        "0.3",
        "0.4",
        "0.5",
        "0.6",
        "0.7",
    }
    assert len(result["diagnostics"]["duration_quartile_map07"]["dense"]) == 4
    assert "heavy_ms" in result["diagnostics"]["median_stage_ms"]["chronotransport"]
    assert result["deploy_claim_allowed"] is False
    assert result["paper_claim_allowed"] is False


def test_gate4_population_recomputes_exact_invocations_and_timing_padding():
    population = _gate4_population()
    assert validate_gate4_population_artifact(population) == population
    assert population["official_video_ids"] == [
        f"video-{index:02d}" for index in range(20)
    ]
    assert population["unique_invocation_count"] == 200
    assert population["timing_block_count"] == 204
    assert population["timing_block_count"] % 6 == 0
    assert [row["invocation_order_index"] for row in population["timing_blocks"]] == list(
        range(204)
    )
    assert population["timing_blocks"][0]["arm_order"] == list(ORDERS[0])
    assert population["timing_blocks"][5]["arm_order"] == list(ORDERS[5])
    assert gate4_population_exact_bytes(population).endswith(b"\n")


def test_gate4_population_rejects_media_or_invocation_tamper():
    population = _gate4_population()
    population["videos"][0]["frame"] += 4
    with pytest.raises(ValueError, match="recomputation"):
        validate_gate4_population_artifact(population)


def test_gate4_metric_population_may_include_video_without_ground_truth():
    metrics = _metric_evidence()
    metrics["ground_truth"] = metrics["ground_truth"][:-1]
    result = _run(metric_evidence=metrics)
    assert result["metrics"]["official_video_count"] == 20


def test_gate4_timing_repetitions_never_become_metric_or_regret_samples():
    result = _run()
    assert result["timing"]["matched_rows_per_seed"] == 240
    assert result["metrics"]["official_video_count"] == 20
    assert result["regret"]["unique_invocations_per_seed"] == 240
    assert result["metrics"]["bootstrap_unit"] == "official_video_then_seed"
    assert result["regret"]["bootstrap_unit"] == "official_video_then_seed"


@pytest.mark.parametrize("tamper", ["missing_arm", "wrong_order", "duplicate_block", "nonfinite"])
def test_gate4_rejects_incomplete_or_noncanonical_matched_timing(tamper):
    rows = _timing_rows()
    if tamper == "missing_arm":
        rows[0]["arms"].pop("static")
    elif tamper == "wrong_order":
        rows[1]["arm_order"] = list(ORDERS[0])
    elif tamper == "duplicate_block":
        rows[1]["invocation_id"] = rows[0]["invocation_id"]
    else:
        rows[0]["arms"]["chronotransport"]["total_ms"] = float("nan")
    with pytest.raises((TypeError, ValueError), match="timing|arm|order|finite|duplicate"):
        _run(timing_rows=rows)


def test_gate4_requires_same_complete_official_population_for_all_seeds():
    rows = _timing_rows()
    rows = [
        row
        for row in rows
        if not (row["seed"] == 3409 and row["invocation_order_index"] == 239)
    ]
    with pytest.raises(ValueError, match="population|240|multiple|same"):
        _run(timing_rows=rows)


def test_gate4_rejects_duplicate_metric_or_regret_samples():
    metrics = _metric_evidence()
    metrics["ground_truth"].append(copy.deepcopy(metrics["ground_truth"][0]))
    with pytest.raises(ValueError, match="ground truth|duplicate"):
        _run(metric_evidence=metrics)

    regret = _regret_rows()
    regret.append(copy.deepcopy(regret[0]))
    with pytest.raises(ValueError, match="regret|duplicate"):
        _run(regret_rows=regret)


def test_gate4_per_seed_fail_closed_prevents_pooled_success():
    rows = _timing_rows()
    for row in rows:
        if row["seed"] == 3409:
            row["arms"]["chronotransport"]["total_ms"] = 9.0
    result = _run(timing_rows=rows)
    assert result["status"] == "FAIL"
    assert result["hard_conditions"]["every_seed_within_thresholds"] is False
    assert result["per_seed"]["3409"]["latency_saving"] < 0.15


def test_gate4_metric_bootstrap_rebuilds_each_seed_without_cross_seed_nms():
    metrics = _metric_evidence()
    # One seed has no CT detections. The point and bootstrap evidence must expose
    # it; predictions from the two good seeds cannot be pooled to hide it.
    metrics["predictions"]["3409"]["chronotransport"] = []
    result = _run(metric_evidence=metrics)
    assert result["per_seed"]["3409"]["map07_drop_points"] > 1.5
    assert result["hard_conditions"]["every_seed_within_thresholds"] is False


def test_gate4_formal_mode_fixes_bootstrap_count_and_seed():
    with pytest.raises(RuntimeError, match="registered evidence producer"):
        adjudicate_gate4(
            timing_rows=[],
            metric_evidence={},
            regret_rows=[],
            bootstrap_samples=5000,
            bootstrap_seed=20260711,
            formal=True,
        )

    with pytest.raises(ValueError, match="5000"):
        adjudicate_gate4(
            timing_rows=_timing_rows(),
            metric_evidence=_metric_evidence(),
            regret_rows=_regret_rows(),
            bootstrap_samples=100,
            bootstrap_seed=20260711,
            formal=True,
        )


def test_gate4_report_validator_recomputes_all_raw_evidence_and_rejects_tamper():
    timing = _timing_rows()
    metrics = _metric_evidence()
    regret = _regret_rows()
    report = adjudicate_gate4(
        timing_rows=timing,
        metric_evidence=metrics,
        regret_rows=regret,
        bootstrap_samples=50,
        bootstrap_seed=20260711,
        formal=False,
    )
    assert validate_gate4_report(
        report,
        timing_rows=timing,
        metric_evidence=metrics,
        regret_rows=regret,
        bootstrap_samples=50,
        bootstrap_seed=20260711,
        formal=False,
    ) == report
    report["latency"]["saving"] = 0.99
    with pytest.raises(ValueError, match="Gate-4 report"):
        validate_gate4_report(
            report,
            timing_rows=timing,
            metric_evidence=metrics,
            regret_rows=regret,
            bootstrap_samples=50,
            bootstrap_seed=20260711,
            formal=False,
        )
    forged_payload = dict(report)
    forged_payload.pop("artifact_sha256")
    report["artifact_sha256"] = canonical_sha256(forged_payload)
    with pytest.raises(ValueError, match="recomputed raw evidence"):
        validate_gate4_report(
            report,
            timing_rows=timing,
            metric_evidence=metrics,
            regret_rows=regret,
            bootstrap_samples=50,
            bootstrap_seed=20260711,
            formal=False,
        )
    with pytest.raises(ValueError, match="20260711"):
        adjudicate_gate4(
            timing_rows=_timing_rows(),
            metric_evidence=_metric_evidence(),
            regret_rows=_regret_rows(),
            bootstrap_samples=5000,
            bootstrap_seed=1,
            formal=True,
        )


def test_gate4_map_uses_official_opentad_equal_score_semantics():
    ground_truth = [("video", "action", 0.0, 1.0)]
    predictions = [
        ("video", "action", 0.0, 1.0, 0.5),
        ("video", "action", 2.0, 3.0, 0.5),
    ]
    official_gt = pd.DataFrame(
        [{"video-id": "video", "t-start": 0.0, "t-end": 1.0}]
    )
    official_predictions = pd.DataFrame(
        [
            {"video-id": "video", "t-start": 0.0, "t-end": 1.0, "score": 0.5},
            {"video-id": "video", "t-start": 2.0, "t-end": 3.0, "score": 0.5},
        ]
    )
    official = float(
        compute_average_precision_detection(
            official_gt,
            official_predictions,
            tiou_thresholds=np.asarray([0.7], dtype=np.float64),
        )[0]
    )
    actual = _map_at(
        ground_truth,
        predictions,
        ["video"],
        tiou_threshold=0.7,
    )
    assert actual == official
