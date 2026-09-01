import json
import math

import pytest

from tools.bata.continuous_roi_s2_v3_full200_compute import (
    canonical_sha256,
    sha256_file,
)
from tools.bata.continuous_roi_s2_v3_full200_compute_eval import (
    GroundTruth,
    Prediction,
    begin_single_gt_open,
    bootstrap_draws,
    bootstrap_occurrences,
    build_prediction_bundle_payload,
    build_prediction_seal,
    evaluate_slot_metrics,
    full_class_map_vector,
    normalized_boundary_medians,
    short_q1_recall,
    simultaneous_bounds,
    VideoOccurrence,
)
from tools.bata.continuous_roi_s2_v3_full200_compute_infer import (
    build_checkpoint_seal,
    load_checkpoint_seal,
)


def _prediction(
    *,
    video="v0",
    window=0,
    proposal=0,
    class_index=0,
    score=1.0,
    start=0.0,
    end=10.0,
):
    return Prediction(
        video_id=video,
        source_window_ordinal=window,
        raw_proposal_ordinal=proposal,
        class_index=class_index,
        score=score,
        start=start,
        end=end,
    )


def test_short_recall_uses_global_all_class_top100_and_stable_ties():
    ground_truth = [GroundTruth("v0", 0, 1, 0.0, 10.0)]
    predictions = [
        _prediction(proposal=index, class_index=0, score=2.0, start=20.0, end=30.0)
        for index in range(100)
    ]
    predictions.append(
        _prediction(proposal=100, class_index=1, score=1.0, start=0.0, end=10.0)
    )
    matched, total, recall = short_q1_recall(
        ground_truth, predictions, q1=10.0
    )
    assert (matched, total, recall) == (0, 1, 0.0)

    predictions = [
        _prediction(proposal=1, class_index=1, score=1.0, start=0.0, end=10.0),
        _prediction(proposal=0, class_index=1, score=1.0, start=0.0, end=10.0),
    ]
    assert [row.raw_proposal_ordinal for row in sorted(predictions, key=lambda row: row.order_key)] == [0, 1]
    assert short_q1_recall(ground_truth, predictions, q1=10.0) == (1, 1, 1.0)


def test_boundary_metric_is_unclipped_median_on_matches():
    ground_truth = [
        GroundTruth("v0", 0, 0, 0.0, 10.0),
        GroundTruth("v0", 1, 0, 20.0, 30.0),
    ]
    predictions = [
        _prediction(proposal=0, start=1.0, end=12.0),
        _prediction(proposal=1, start=18.0, end=31.0),
    ]
    start, end = normalized_boundary_medians(ground_truth, predictions)
    assert start == pytest.approx(0.15)
    assert end == pytest.approx(0.15)
    assert normalized_boundary_medians(ground_truth, []) == (math.inf, math.inf)


def test_exact_bootstrap_draws_and_order_statistics():
    videos = [f"v{index:03d}" for index in range(211)]
    draw = bootstrap_draws(0, videos)
    assert draw["seed_draws"] == [4408, 4409, 4407]
    assert [rows[:3] for rows in draw["video_index_draws"]] == [
        [78, 83, 79],
        [120, 25, 73],
        [94, 57, 1],
    ]
    lcb, ucb = simultaneous_bounds(list(range(20_000)), list(range(20_000)))
    assert lcb == 199
    assert ucb == 19_800


def test_bootstrap_occurrences_are_unique_and_preserve_original_video():
    videos = [f"v{index:03d}" for index in range(211)]
    draw = bootstrap_draws(0, videos)
    rows = bootstrap_occurrences(draw, videos, seed_slot=0)
    assert len(rows) == 211
    assert len({row.synthetic_video_id for row in rows}) == 211
    assert rows[0].original_video_id == "v078"
    assert rows[0].synthetic_video_id.endswith("videoslot/000/origvideo/v078")


def test_fixed_class_denominator_and_slot_metrics():
    ground_truth = [GroundTruth("v0", 0, 0, 0.0, 10.0)]
    predictions = [_prediction(start=0.0, end=10.0)]
    occurrences = [VideoOccurrence("v0", "v0")]
    vector = full_class_map_vector(
        ground_truth,
        predictions,
        occurrences=occurrences,
        class_count=2,
    )
    assert vector == pytest.approx((50.0,) * 5)
    metrics = evaluate_slot_metrics(
        ground_truth,
        predictions,
        occurrences=occurrences,
        class_count=1,
        q1=10.0,
    )
    assert metrics.average_map_pp == pytest.approx(100.0)
    assert metrics.map_at_0_7_pp == pytest.approx(100.0)
    assert metrics.short_q1_recall == pytest.approx(1.0)
    assert metrics.normalized_start_error_median == pytest.approx(0.0)
    assert metrics.normalized_end_error_median == pytest.approx(0.0)


def test_prediction_seal_is_9_of_9_and_gt_open_is_irreversible(tmp_path):
    videos = [f"v{index:03d}" for index in range(211)]
    class_map = ["Action"]
    prediction_paths = {}
    for arm in ("D160", "G96", "U128-A0"):
        prediction_paths[arm] = {}
        for seed in (4407, 4408, 4409):
            path = tmp_path / f"{arm}-{seed}.json"
            payload = build_prediction_bundle_payload(
                arm=arm,
                seed=seed,
                population_manifest_sha256="population",
                video_order=videos,
                results={video_id: [] for video_id in videos},
                class_map=class_map,
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            prediction_paths[arm][seed] = path
    seal_path = tmp_path / "prediction_seal.json"
    seal = build_prediction_seal(
        prediction_paths,
        checkpoint_seal_sha256="checkpoint-seal",
        population_manifest_sha256="population",
        expected_video_order=videos,
        class_map=class_map,
        output_path=seal_path,
    )
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps({"database": {}}), encoding="utf-8")
    marker = tmp_path / "gt_open_started.json"
    begin_single_gt_open(
        marker_path=marker,
        annotation_path=annotation,
        prediction_seal_path=seal_path,
        expected_prediction_seal_sha256=sha256_file(seal_path),
    )
    assert seal["row_count"] == 9
    with pytest.raises(FileExistsError):
        begin_single_gt_open(
            marker_path=marker,
            annotation_path=annotation,
            prediction_seal_path=seal_path,
            expected_prediction_seal_sha256=sha256_file(seal_path),
        )


def test_checkpoint_seal_requires_nine_complete_full_training_receipts(tmp_path):
    cells = []
    for arm in ("D160", "G96", "U128-A0"):
        for seed in (4407, 4408, 4409):
            checkpoint = tmp_path / f"{arm}-{seed}.pth"
            checkpoint.write_bytes(f"{arm}-{seed}".encode())
            config = tmp_path / f"{arm}-{seed}.py"
            config.write_text("seed = 1\n", encoding="utf-8")
            terminal = tmp_path / f"{arm}-{seed}.terminal.json"
            terminal_payload = {
                "complete": True,
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_state": "epoch_59_state_dict_ema_update_6000",
            }
            terminal_payload["receipt_sha256"] = canonical_sha256(terminal_payload)
            terminal.write_text(json.dumps(terminal_payload), encoding="utf-8")
            cells.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "checkpoint_path": str(checkpoint),
                    "config_path": str(config),
                    "training_terminal_receipt_path": str(terminal),
                }
            )
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({"cells": cells}), encoding="utf-8")
    seal_path = tmp_path / "checkpoint_seal.json"
    seal = build_checkpoint_seal(
        matrix_path=matrix,
        population_manifest_sha256="population",
        expected_commit="candidate",
        output_path=seal_path,
    )
    assert seal["row_count"] == 9
    assert load_checkpoint_seal(
        seal_path,
        expected_commit="candidate",
        expected_population_manifest_sha256="population",
    )["seal_sha256"] == seal["seal_sha256"]
