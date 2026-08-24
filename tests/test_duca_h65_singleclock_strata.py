import json

from tools.bata.analyze_duca_h65_singleclock_strata import (
    _match_boundary_errors,
    _record_distortion,
    _uniform_positions,
    _video_annotations,
    evaluate_boundary_risk_strata,
    freeze_boundary_risk_strata,
    freeze_training_strata,
)


def test_uniform_distortion_is_exactly_zero():
    positions = _uniform_positions(768, 384)
    record = {
        "selected_positions": positions.tolist(),
        "selected_valid_len": 384,
        "dense_valid_len": 768,
    }
    assert _record_distortion(record) == 0.0


def test_partial_trailing_tubelet_does_not_create_clock_support():
    record = {
        "selected_positions": [0, 4, 9],
        "selected_valid_len": 3,
        "dense_valid_len": 10,
    }
    assert _record_distortion(record) == 0.0


def test_freeze_uses_training_only_q1_and_video_distortion(tmp_path):
    identity = {
        "records": [
            {
                "video_name": f"train_{index}",
                "selected_positions": _uniform_positions(8, 4).tolist(),
                "selected_valid_len": 4,
                "dense_valid_len": 8,
            }
            for index in range(4)
        ]
    }
    annotation = {
        "database": {
            **{
                f"train_{index}": {
                    "subset": "training",
                    "annotations": [{"segment": [0.0, float(index + 1)]}],
                }
                for index in range(4)
            },
            "validation_0": {
                "subset": "validation",
                "annotations": [{"segment": [0.0, 100.0]}],
            },
        }
    }
    identity_path = tmp_path / "identity.json"
    annotation_path = tmp_path / "annotation.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    frozen = freeze_training_strata(
        training_identity_path=identity_path,
        annotation_path=annotation_path,
    )
    assert frozen["short_action_duration_q25_seconds"] == 1.75
    assert frozen["distortion_q75"] == 0.0
    assert frozen["validation_or_test_used"] is False


def _identity_rows(video_names):
    return {
        "records": [
            {
                "sample_id": f"{video}|window_start_frame=0",
                "video_name": video,
                "selected_positions": [0, 1, 4, 5],
                "selected_valid_len": 4,
                "dense_valid_len": 8,
            }
            for video in video_names
        ]
    }


def _ledger(video_names, subset):
    return {
        "schema_version": "duca_h65_physical_window_ledger_v1",
        "subset": subset,
        "records": [
            {
                "sample_id": f"{video}|window_start_frame=0",
                "video_name": video,
                "valid_start_seconds": 0.0,
                "valid_end_seconds": 10.0,
                "is_final_valid_window": True,
            }
            for video in video_names
        ],
    }


def test_boundary_matching_penalizes_unmatched_gt_and_uses_physical_seconds():
    annotations = {
        "v": [
            {
                "label": "Action",
                "start": 2.0,
                "end": 4.0,
                "canonical_occurrence_index": 0,
            },
            {
                "label": "Action",
                "start": 6.0,
                "end": 8.0,
                "canonical_occurrence_index": 1,
            },
        ]
    }
    errors = _match_boundary_errors(
        {
            "v": [
                {"segment": [2.5, 4.0], "label": "Action", "score": 0.9},
            ]
        },
        annotations,
    )
    assert errors["v"][0] == 0.125
    assert errors["v"][1] == 1.0


def test_boundary_gt_uses_official_duplicate_removal_tolerance():
    annotation = {
        "database": {
            "v": {
                "subset": "validation",
                "annotations": [
                    {"segment": [2.0, 4.0], "label": "Action"},
                    {"segment": [2.0005, 4.0005], "label": "Action"},
                    {"segment": [2.0, 4.0], "label": "Other"},
                ],
            }
        }
    }
    rows = _video_annotations(annotation, subset="validation")["v"]
    assert [(row["label"], row["start"], row["end"]) for row in rows] == [
        ("Action", 2.0, 4.0),
        ("Other", 2.0, 4.0),
    ]


def test_boundary_risk_freeze_and_evaluation_are_training_only_and_lower_is_better(tmp_path):
    training_videos = [f"train_{index}" for index in range(4)]
    validation_videos = ["validation_0", "validation_1"]
    annotation = {
        "database": {
            **{
                video: {
                    "subset": "training",
                    "annotations": [{"segment": [2.0, 4.0], "label": "Action"}],
                }
                for video in training_videos
            },
            **{
                video: {
                    "subset": "validation",
                    "annotations": [{"segment": [2.0, 4.0], "label": "Action"}],
                }
                for video in validation_videos
            },
        }
    }
    paths = {}
    payloads = {
        "training_identity": _identity_rows(training_videos),
        "training_ledger": _ledger(training_videos, "training"),
        "validation_identity": _identity_rows(validation_videos),
        "validation_ledger": _ledger(validation_videos, "validation"),
        "annotation": annotation,
        "on_prediction": {
            "results": {
                video: [{"segment": [2.0, 4.0], "label": "Action", "score": 0.9}]
                for video in validation_videos
            }
        },
        "off_prediction": {
            "results": {
                video: [{"segment": [2.5, 4.0], "label": "Action", "score": 0.9}]
                for video in validation_videos
            }
        },
    }
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    frozen = freeze_boundary_risk_strata(
        training_identity_path=paths["training_identity"],
        training_window_ledger_path=paths["training_ledger"],
        annotation_path=paths["annotation"],
    )
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    result = evaluate_boundary_risk_strata(
        frozen_path=frozen_path,
        validation_identity_path=paths["validation_identity"],
        validation_window_ledger_path=paths["validation_ledger"],
        annotation_path=paths["annotation"],
        on_prediction_path=paths["on_prediction"],
        off_prediction_path=paths["off_prediction"],
        nonce="test-boundary-risk",
    )
    assert frozen["validation_or_test_used"] is False
    assert result["status"] == "EVALUABLE"
    assert result["validation_or_test_used_for_cutpoints"] is False
    assert result["high_gapcv_delta_point"] < 0.0
    assert result["high_boundary_density_delta_point"] < 0.0
    assert result["high_gapcv_pass"] is True
    assert result["high_boundary_density_pass"] is True
