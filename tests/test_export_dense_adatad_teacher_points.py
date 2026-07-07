from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import detector_teacher_utility
from tools.bata import export_dense_adatad_teacher_points as exporter


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_source_sample_map_requires_unique_strict_window_ids(tmp_path: Path) -> None:
    source = tmp_path / "samples.jsonl"
    _write_jsonl(
        source,
        [
            {
                "sample_id": "video_validation_0000001|0",
                "split": "training",
                "dense_len": 768,
                "valid_len": 120,
                "frame_signals": {"p_action": [0.1, 0.2]},
            }
        ],
    )

    samples = exporter.read_source_samples(source)

    assert samples["video_validation_0000001|0"].video_name == "video_validation_0000001"
    assert samples["video_validation_0000001|0"].window_start_frame == 0
    assert samples["video_validation_0000001|0"].valid_len == 120

    _write_jsonl(
        source,
        [
            {"sample_id": "video_validation_0000001|0", "dense_len": 4, "valid_len": 4},
            {"sample_id": "video_validation_0000001|0", "dense_len": 4, "valid_len": 4},
        ],
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        exporter.read_source_samples(source)


def test_dense_teacher_row_from_predictions_is_train_only_and_manifest_compatible(tmp_path: Path) -> None:
    checkpoint = tmp_path / "teacher.pth"
    config = tmp_path / "teacher.py"
    checkpoint.write_bytes(b"teacher checkpoint")
    config.write_text("model = dict(type='AdaTAD')\n", encoding="utf-8")
    sample = exporter.SourceSample(
        sample_id="video_validation_0000001|0",
        video_name="video_validation_0000001",
        window_start_frame=0,
        dense_len=6,
        valid_len=5,
        row={
            "sample_id": "video_validation_0000001|0",
            "dense_len": 6,
            "valid_len": 5,
            "fps": 25.0,
            "snippet_stride": 4,
            "window_size": 768,
        },
    )

    row = exporter.dense_teacher_row_from_predictions(
        sample=sample,
        meta={"video_name": "video_validation_0000001", "window_start_frame": 0, "fps": 25.0, "snippet_stride": 4},
        proposals=[[0.2, 1.8], [3.2, 4.7], [9.0, 10.0]],
        scores=[[0.2, 0.8], {"classification_score": 0.5}, 0.9],
        topk=2,
        teacher_checkpoint_path=str(checkpoint),
        teacher_checkpoint_sha256=detector_teacher_utility._sha256_file(checkpoint),
        teacher_config_path=str(config),
        teacher_config_sha256=detector_teacher_utility._sha256_file(config),
    )

    assert row["schema_version"] == "c3_dense_adatad_teacher_points_row_v1"
    assert row["split"] == "training"
    assert row["teacher_signal_source"] == "adatad_dense_teacher"
    assert row["uses_evaluator_outputs"] is False
    assert row["uses_raw_prediction"] is False
    assert row["uses_prediction_cache"] is False
    assert row["uses_val_or_test_gt_for_selection"] is False
    assert row["uses_gt_for_selection"] is False
    assert row["training_only"] is True
    assert row["end_to_end"] is False
    assert [point["point_index"] for point in row["teacher_dense_points"]] == [1, 4]
    assert row["teacher_utility_provenance"]["generator_source"] == "dense_detector_forward_train"


def test_stage2_launcher_requires_and_passes_generator_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    stage2 = (root / "scripts" / "run_c3_detector_aware_selector_adatad_full_train_gpu1.sh").read_text(
        encoding="utf-8"
    )
    precheck = (root / "scripts" / "run_duca_stage2_detector_aware_precheck_gpu1.sh").read_text(encoding="utf-8")

    assert "export_dense_adatad_teacher_points.py" in stage2
    assert "C3_DETECTOR_AWARE_TEACHER_GENERATOR_MANIFEST_JSON" in stage2
    assert "--generator-manifest-json" in stage2
    assert "--require-stage2-generator-manifest" in precheck

