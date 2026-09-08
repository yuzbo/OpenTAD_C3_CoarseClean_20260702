import json
import math

import pytest

from tools.bata import continuous_roi_s2_v3_full200_compute_eval as evaluation
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
    PROTOCOL_ID,
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


@pytest.mark.parametrize("entry", ["direct", "cli"])
def test_prediction_seal_is_9_of_9_and_gt_open_is_irreversible(tmp_path, monkeypatch, entry):
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
    if entry == "direct":
        begin_single_gt_open(
            marker_path=marker,
            annotation_path=annotation,
            prediction_seal_path=seal_path,
            expected_prediction_seal_sha256=sha256_file(seal_path),
        )
    else:
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({
            "evaluation": {"video_order": videos},
            "class_map": {"classes": class_map},
        }), encoding="utf-8")
        checkpoint_seal = tmp_path / "checkpoint_seal.json"
        checkpoint_seal.write_text("{}", encoding="utf-8")

        def stop_before_gt_loading(**kwargs):
            raise RuntimeError("test_barrier_reached")

        monkeypatch.setattr(evaluation, "load_ground_truth_after_single_open", stop_before_gt_loading)
        assert seal["seal_sha256"] != sha256_file(seal_path)
        with pytest.raises(RuntimeError, match="test_barrier_reached"):
            evaluation.main([
                "evaluate-matrix", "--prediction-seal", str(seal_path),
                "--checkpoint-seal", str(checkpoint_seal),
                "--manifest", str(manifest), "--annotation", str(annotation),
                "--marker-path", str(marker), "--output-dir", str(tmp_path / "output"),
            ])
    assert json.loads(marker.read_text(encoding="utf-8"))["prediction_seal_sha256"] == sha256_file(seal_path)
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
                "protocol_id": PROTOCOL_ID,
                "arm": arm,
                "seed": seed,
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


@pytest.mark.parametrize("gt_segments,prediction_rows,expected_ap", [
    ([(0.0, 1.0)], [(0.0, 1.0, 0.5), (2.0, 3.0, 0.5)], 50.0),
    ([(0.0, 2.0), (1.0, 3.0)], [(0.0, 3.0, 0.9), (0.0, 2.0, 0.8)], 100.0),
    ([(0.0, 1.0)], [(0.0, 0.0, 0.9), (0.0, 1.0, 0.8)], 50.0),
    ([(0.0, 1.0)], [(10.0, 10.0, 0.9), (0.0, 1.0, 0.8)], 50.0),
])
def test_map_matches_official_ties_and_zero_duration_false_positives(gt_segments, prediction_rows, expected_ap):
    import pandas as pd
    from opentad.evaluations.mAP import compute_average_precision_detection

    ground_truth = [GroundTruth("v", index, 0, start, end)
                    for index, (start, end) in enumerate(gt_segments)]
    predictions = [Prediction("v", 0, index, 0, score, start, end)
                   for index, (start, end, score) in enumerate(prediction_rows)]
    local = full_class_map_vector(
        ground_truth, predictions, occurrences=[VideoOccurrence("v", "v")],
        class_count=1, tiou_thresholds=[0.5],
    )
    official = compute_average_precision_detection(
        pd.DataFrame([{"video-id": "v", "t-start": s, "t-end": e, "label": 0}
                      for s, e in gt_segments]),
        pd.DataFrame([{"video-id": "v", "t-start": s, "t-end": e, "score": score, "label": 0}
                      for s, e, score in prediction_rows]), tiou_thresholds=[0.5],
    )
    assert local[0] == pytest.approx(expected_ap, abs=1e-12)
    assert local[0] == pytest.approx(100.0 * official[0], abs=1e-12)


@pytest.mark.parametrize("values", [
    {"score": math.nan}, {"score": math.inf}, {"start": math.nan},
    {"end": math.inf}, {"start": 2.0, "end": 1.0},
])
def test_nonfinite_or_reversed_predictions_still_fail_with_identity(values):
    with pytest.raises(ValueError, match="uid=.*v0"):
        _prediction(**values)


def test_zero_duration_ground_truth_remains_invalid():
    with pytest.raises(ValueError, match="positive duration"):
        GroundTruth("v0", 0, 0, 1.0, 1.0)
