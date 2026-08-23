import json

from tools.bata.analyze_duca_h65_singleclock_strata import (
    _record_distortion,
    _uniform_positions,
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
