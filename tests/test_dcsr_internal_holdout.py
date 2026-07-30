import hashlib
import json

import numpy as np
import pytest

from libs.datasets.thumos14 import THUMOS14Dataset
from libs.utils import ANETdetection
from tools.build_dcsr_internal_holdout import build_manifest


def _write_annotation(path):
    database = {}
    label_sets = [
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1),
        (0, 2),
        (1, 2),
    ]
    for index, labels in enumerate(label_sets):
        video_id = "validation_{:02d}".format(index)
        database[video_id] = {
            "subset": "validation",
            "fps": 30.0,
            "duration": 10.0,
            "annotations": [
                {
                    "segment": [1.0 + label, 2.0 + label],
                    "label": "class_{:d}".format(label),
                    "label_id": label,
                }
                for label in labels
            ],
        }
    database["test_ignored"] = {
        "subset": "test",
        "fps": 30.0,
        "duration": 10.0,
        "annotations": [
            {
                "segment": [1.0, 2.0],
                "label": "class_0",
                "label_id": 0,
            }
        ],
    }
    path.write_text(
        json.dumps({"database": database}),
        encoding="utf-8",
    )
    return database


def _annotation_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_internal_holdout_builder_never_reads_test_and_covers_classes(tmp_path):
    annotation_path = tmp_path / "annotations.json"
    feature_folder = tmp_path / "features"
    feature_folder.mkdir()
    database = _write_annotation(annotation_path)
    for video_id, record in database.items():
        if record["subset"] == "validation":
            np.save(
                feature_folder / (video_id + ".npy"),
                np.zeros((4, 4), dtype=np.float32),
            )

    manifest = build_manifest(
        str(annotation_path),
        str(feature_folder),
        ".npy",
        seed=20260730,
        holdout_numerator=1,
        holdout_denominator=3,
    )

    assert manifest["test_annotations_used"] is False
    assert manifest["test_records_selected"] is False
    assert manifest["source_split"] == "validation"
    assert "test_ignored" not in manifest["train_video_ids"]
    assert "test_ignored" not in manifest["holdout_video_ids"]
    assert set(manifest["train_video_ids"]).isdisjoint(
        manifest["holdout_video_ids"]
    )
    assert manifest["train_all_class_coverage"] is True
    assert manifest["holdout_all_class_coverage"] is True
    assert manifest["all_class_ids"] == [0, 1, 2]


def test_thumos_manifest_filter_and_evaluator_share_exact_holdout(tmp_path):
    annotation_path = tmp_path / "annotations.json"
    feature_folder = tmp_path / "features"
    feature_folder.mkdir()
    database = _write_annotation(annotation_path)
    for video_id, record in database.items():
        if record["subset"] == "validation":
            np.save(
                feature_folder / (video_id + ".npy"),
                np.zeros((4, 4), dtype=np.float32),
            )
    manifest = build_manifest(
        str(annotation_path),
        str(feature_folder),
        ".npy",
        seed=20260730,
        holdout_numerator=1,
        holdout_denominator=3,
    )
    manifest_path = tmp_path / "holdout.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    dataset = THUMOS14Dataset(
        is_training=False,
        split=["validation"],
        feat_folder=str(feature_folder),
        json_file=str(annotation_path),
        feat_stride=4,
        num_frames=16,
        default_fps=None,
        downsample_rate=1,
        max_seq_len=16,
        trunc_thresh=0.5,
        crop_ratio=None,
        input_dim=4,
        num_classes=3,
        file_prefix=None,
        file_ext=".npy",
        force_upsampling=False,
        video_id_manifest=str(manifest_path),
        video_id_manifest_sha256=manifest_sha256,
        video_id_manifest_subset="holdout",
    )
    expected_ids = tuple(sorted(manifest["holdout_video_ids"]))
    assert dataset.evaluation_video_ids == expected_ids
    assert tuple(sorted(item["id"] for item in dataset.data_list)) == expected_ids

    evaluator = ANETdetection(
        str(annotation_path),
        split="validation",
        tiou_thresholds=np.linspace(0.3, 0.7, 5),
        video_ids=dataset.evaluation_video_ids,
        num_workers=1,
    )
    assert frozenset(
        evaluator.ground_truth["video-id"].unique()
    ) == frozenset(expected_ids)
    with pytest.raises(ValueError, match="escaped internal holdout"):
        evaluator.evaluate(
            {
                "video-id": np.asarray(["validation_00", "test_ignored"]),
                "t-start": np.asarray([1.0, 1.0]),
                "t-end": np.asarray([2.0, 2.0]),
                "label": np.asarray([0, 0]),
                "score": np.asarray([0.5, 0.5]),
            },
            verbose=False,
        )


def test_manifest_annotation_hash_is_bound(tmp_path):
    annotation_path = tmp_path / "annotations.json"
    feature_folder = tmp_path / "features"
    feature_folder.mkdir()
    database = _write_annotation(annotation_path)
    for video_id, record in database.items():
        if record["subset"] == "validation":
            np.save(
                feature_folder / (video_id + ".npy"),
                np.zeros((4, 4), dtype=np.float32),
            )
    manifest = build_manifest(
        str(annotation_path),
        str(feature_folder),
        ".npy",
        seed=20260730,
        holdout_numerator=1,
        holdout_denominator=3,
    )
    assert manifest["source_annotation_sha256"] == _annotation_sha256(
        annotation_path
    )
