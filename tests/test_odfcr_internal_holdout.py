import hashlib
import json

import numpy as np
import pytest

from libs.datasets.thumos14 import THUMOS14Dataset
from tools.build_dcsr_internal_holdout import build_manifest as build_v1
from tools.build_odfcr_internal_holdout_v2 import build_manifest as build_v2
from tools.evaluate_odfcr_internal_predictions import (
    _load_and_validate_manifest,
)


def _write_official_shape_fixture(annotation_path, feature_folder):
    database = {}
    for index in range(200):
        video_id = "video_validation_{:07d}".format(index)
        labels = (index % 20, (index + 7) % 20)
        database[video_id] = {
            "subset": "validation",
            "fps": 30.0,
            "duration": 10.0,
            "annotations": [
                {
                    "segment": [1.0, 2.0],
                    "label": "class_{:02d}".format(label),
                    "label_id": label,
                }
                for label in labels
            ],
        }
        np.save(
            feature_folder / (video_id + ".npy"),
            np.zeros((4, 4), dtype=np.float32),
        )
    database["video_test_forbidden"] = {
        "subset": "test",
        "fps": 30.0,
        "duration": 10.0,
        "annotations": [
            {
                "segment": [1.0, 2.0],
                "label": "class_00",
                "label_id": 0,
            }
        ],
    }
    annotation_path.write_text(
        json.dumps({"database": database}),
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_manifests(tmp_path):
    annotation_path = tmp_path / "annotations.json"
    feature_folder = tmp_path / "features"
    feature_folder.mkdir()
    _write_official_shape_fixture(annotation_path, feature_folder)
    previous = build_v1(
        str(annotation_path),
        str(feature_folder),
        ".npy",
        seed=20260730,
        holdout_numerator=1,
        holdout_denominator=5,
    )
    previous_path = tmp_path / "holdout_v1.json"
    previous_path.write_text(
        json.dumps(previous, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    current = build_v2(
        str(previous_path),
        str(annotation_path),
        str(feature_folder),
        ".npy",
        seed=2026073100,
    )
    current_path = tmp_path / "holdout_v2.json"
    current_path.write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        annotation_path,
        feature_folder,
        previous_path,
        previous,
        current_path,
        current,
    )


def test_odfcr_holdout_v2_uses_only_old_train_and_is_disjoint(tmp_path):
    (
        _,
        _,
        previous_path,
        previous,
        _,
        current,
    ) = _build_manifests(tmp_path)

    assert current["schema_version"] == (
        "actionformer_odfcr_internal_holdout_v2"
    )
    assert current["previous_manifest_sha256"] == _sha256(previous_path)
    assert current["candidate_pool_video_ids"] == sorted(
        previous["train_video_ids"]
    )
    assert set(current["holdout_video_ids"]) <= set(
        previous["train_video_ids"]
    )
    assert set(current["holdout_video_ids"]).isdisjoint(
        previous["holdout_video_ids"]
    )
    assert len(current["holdout_video_ids"]) == 40
    assert len(current["train_video_ids"]) == 160
    assert set(current["train_video_ids"]).isdisjoint(
        current["holdout_video_ids"]
    )
    assert set(current["train_video_ids"]) | set(
        current["holdout_video_ids"]
    ) == set(previous["train_video_ids"]) | set(
        previous["holdout_video_ids"]
    )
    assert current["all_class_ids"] == list(range(20))
    assert "video_test_forbidden" not in current["train_video_ids"]
    assert "video_test_forbidden" not in current["holdout_video_ids"]
    assert current["paper_performance_row_allowed"] is False
    assert current["official_test_authorized"] is False


@pytest.mark.parametrize("subset", ["train", "holdout"])
def test_odfcr_holdout_v2_dataset_filter_is_exact(tmp_path, subset):
    (
        annotation_path,
        feature_folder,
        _,
        _,
        current_path,
        current,
    ) = _build_manifests(tmp_path)
    dataset = THUMOS14Dataset(
        is_training=(subset == "train"),
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
        num_classes=20,
        file_prefix=None,
        file_ext=".npy",
        force_upsampling=False,
        video_id_manifest=str(current_path),
        video_id_manifest_sha256=_sha256(current_path),
        video_id_manifest_subset=subset,
    )
    expected = tuple(sorted(current[subset + "_video_ids"]))
    assert dataset.evaluation_video_ids == expected
    assert tuple(sorted(item["id"] for item in dataset.data_list)) == expected


def test_odfcr_evaluator_revalidates_previous_manifest_membership(tmp_path):
    (
        annotation_path,
        _,
        previous_path,
        _,
        current_path,
        current,
    ) = _build_manifests(tmp_path)
    _, holdout_ids = _load_and_validate_manifest(
        str(current_path),
        str(previous_path),
        str(annotation_path),
        _sha256(annotation_path),
    )
    assert holdout_ids == frozenset(current["holdout_video_ids"])

    corrupted = dict(current)
    corrupted["holdout_video_ids"] = list(current["holdout_video_ids"])
    corrupted["holdout_video_ids"][0] = current[
        "previous_holdout_video_ids"
    ][0]
    corrupted_path = tmp_path / "corrupted_v2.json"
    corrupted_path.write_text(
        json.dumps(corrupted),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="holdout-v2"):
        _load_and_validate_manifest(
            str(corrupted_path),
            str(previous_path),
            str(annotation_path),
            _sha256(annotation_path),
        )


@pytest.mark.parametrize(
    "field,mutator",
    [
        (
            "holdout_video_ids",
            lambda current: [
                current["holdout_video_ids"][0]
            ]
            + current["holdout_video_ids"][:-1],
        ),
        ("candidate_pool_is_previous_train_only", lambda current: False),
        ("holdout_all_class_coverage", lambda current: False),
    ],
)
def test_odfcr_evaluator_rejects_manifest_semantic_drift(
    tmp_path, field, mutator
):
    (
        annotation_path,
        _,
        previous_path,
        _,
        _,
        current,
    ) = _build_manifests(tmp_path)
    corrupted = dict(current)
    corrupted[field] = mutator(current)
    corrupted_path = tmp_path / ("corrupted_" + field + ".json")
    corrupted_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="holdout-v2"):
        _load_and_validate_manifest(
            str(corrupted_path),
            str(previous_path),
            str(annotation_path),
            _sha256(annotation_path),
        )


def test_odfcr_builder_rejects_previous_manifest_count_drift(tmp_path):
    (
        annotation_path,
        feature_folder,
        previous_path,
        previous,
        _,
        _,
    ) = _build_manifests(tmp_path)
    previous["train_video_ids"] = previous["train_video_ids"][:-1]
    previous["train_video_count"] = 159
    previous_path.write_text(json.dumps(previous), encoding="utf-8")
    with pytest.raises(ValueError, match="160/40"):
        build_v2(
            str(previous_path),
            str(annotation_path),
            str(feature_folder),
            ".npy",
            seed=2026073100,
        )
