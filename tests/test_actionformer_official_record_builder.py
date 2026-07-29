import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "bata" / "build_actionformer_official_record.py"
SPEC = importlib.util.spec_from_file_location("official_record_builder", MODULE_PATH)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _raw_predictions():
    return {
        "video-id": ["video_test_0000001", "video_test_0000001"],
        "t-start": np.asarray([1.0, 2.0], dtype=np.float32),
        "t-end": np.asarray([2.0, 3.0], dtype=np.float32),
        "label": np.asarray([0, 1], dtype=np.int64),
        "score": np.asarray([0.9, 0.8], dtype=np.float32),
    }


def test_raw_prediction_pickle_contract(tmp_path):
    path = tmp_path / "eval_results.pkl"
    with path.open("wb") as handle:
        pickle.dump(_raw_predictions(), handle)
    normalized, count = builder.load_and_validate_raw_predictions(path)
    assert count == 2
    assert normalized["label"].dtype == np.int64
    assert normalized["t-start"].dtype == np.float64

    payload = _raw_predictions()
    payload["score"][0] = np.nan
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(builder.protocol.ProtocolError, match="NaN/Inf"):
        builder.load_and_validate_raw_predictions(path)


def test_annotation_class_map_and_pinned_release_split_counts(tmp_path):
    database = {}
    for index in range(200):
        database[f"video_validation_{index:07d}"] = {
            "subset": "Validation",
            "annotations": [
                {
                    "label": builder.protocol.OFFICIAL_THUMOS_CLASS_NAMES[
                        index % 20
                    ],
                    "label_id": index % 20,
                    "segment": [1.0, 2.0],
                }
            ],
        }
    for index in range(212):
        database[f"video_test_{index:07d}"] = {
            "subset": "Test",
            "annotations": [
                {
                    "label": builder.protocol.OFFICIAL_THUMOS_CLASS_NAMES[
                        index % 20
                    ],
                    "label_id": index % 20,
                    "segment": [1.0, 2.0],
                }
            ],
        }
    annotation = tmp_path / "thumos14.json"
    annotation.write_text(json.dumps({"database": database}), encoding="utf-8")
    class_map, split_counts, videos = builder.parse_annotation(annotation)
    assert split_counts == {"test": 212, "validation": 200}
    assert len(class_map["labels"]) == 20
    assert len(videos) == 412
    assert {subset for _, subset in videos} == {"test", "validation"}


def test_feature_manifest_checks_full_inventory_and_finiteness(
    tmp_path, monkeypatch
):
    feature_dir = tmp_path / "i3d_features"
    feature_dir.mkdir()
    np.save(feature_dir / "video_a.npy", np.ones((3, 2048), dtype=np.float32))
    np.save(
        feature_dir / "video_extra.npy",
        np.ones((2, 2048), dtype=np.float32),
    )
    monkeypatch.setattr(builder.protocol, "OFFICIAL_FEATURE_INVENTORY_VIDEO_COUNT", 2)
    monkeypatch.setattr(
        builder.protocol,
        "OFFICIAL_FEATURE_ONLY_UNANNOTATED_VIDEOS",
        ("video_extra",),
    )
    monkeypatch.setattr(builder.protocol, "OFFICIAL_EVALUATED_VIDEO_COUNT", 1)
    manifest = builder.build_feature_manifest(
        feature_dir,
        [("video_a", "test")],
    )
    assert manifest["feature_inventory_video_count"] == 2
    assert manifest["annotation_feature_backed_video_count"] == 1
    assert manifest["evaluated_feature_backed_video_count"] == 1
    assert manifest["feature_only_unannotated_videos"] == ["video_extra"]
    assert manifest["evaluated_video_ids"] == ["video_a"]
    assert manifest["features"][0]["shape"] == [3, 2048]

    invalid_dir = tmp_path / "invalid_i3d_features"
    invalid_dir.mkdir()
    invalid = np.ones((2, 2048), dtype=np.float32)
    invalid[0, 10] = np.inf
    np.save(invalid_dir / "video_b.npy", invalid)
    monkeypatch.setattr(builder.protocol, "OFFICIAL_FEATURE_INVENTORY_VIDEO_COUNT", 1)
    monkeypatch.setattr(
        builder.protocol,
        "OFFICIAL_FEATURE_ONLY_UNANNOTATED_VIDEOS",
        (),
    )
    with pytest.raises(builder.protocol.ProtocolError, match="NaN/Inf"):
        builder.build_feature_manifest(
            invalid_dir,
            [("video_b", "test")],
        )


def test_pinned_evaluator_manifest_matches_exact_official_clone():
    repo = ROOT.parent / "official_actionformer_release"
    if not repo.is_dir():
        pytest.skip("pinned official ActionFormer clone is not present")
    repo, commit, tree = builder.verify_official_source(repo)
    manifest, fingerprint = builder.build_evaluator_manifest(repo, commit, tree)
    assert manifest["files"] == builder.protocol.OFFICIAL_EVALUATOR_FILES
    assert fingerprint == builder.protocol.OFFICIAL_EVALUATOR_FINGERPRINT_SHA256


def test_official_effective_config_and_released_seed_are_pinned():
    repo = ROOT.parent / "official_actionformer_release"
    if not repo.is_dir():
        pytest.skip("pinned official ActionFormer clone is not present")
    config = builder.load_effective_config(repo)

    assert config["init_rand_seed"] == 1234567891
    assert config["init_rand_seed"] == builder.protocol.OFFICIAL_TRAINING_SEED
    assert builder.protocol.canonical_sha256(config) == (
        builder.protocol.OFFICIAL_EFFECTIVE_CONFIG_SHA256
    )


def test_train_log_effective_config_binding_rejects_wrong_seed():
    config = {
        "train_split": ["validation"],
        "val_split": ["test"],
        "init_rand_seed": 1234567891,
    }
    parsed = builder.protocol.parse_actionformer_train_log_config(
        repr(config) + "\nUsing model EMA ...\n"
    )
    assert parsed == config
    config["init_rand_seed"] = 0
    assert builder.protocol.canonical_sha256(config) != (
        builder.protocol.OFFICIAL_EFFECTIVE_CONFIG_SHA256
    )


def test_official_source_hashes_pin_git_lf_bytes_not_windows_checkout_bytes():
    assert builder.protocol.OFFICIAL_CONFIG_SHA256 == (
        "c0ac0df560cd564941b56cd9391ad0bd5cea386d2e4b6cf9fc8ffcab821955cd"
    )
    assert builder.protocol.OFFICIAL_README_SHA256 == (
        "f0431584b4df0702fa08f961fb0038e1277f41c12b7df47b7d2bfed47e59af23"
    )
    assert builder.protocol.OFFICIAL_EVALUATOR_FINGERPRINT_SHA256 == (
        "1d18fbb07a774422a1594946dcf2c59a741c5de3a55d42fa029636ffc43c30b6"
    )


def test_environment_manifest_has_stable_comparability_fingerprint():
    manifest, fingerprint = builder.environment_manifest()
    assert manifest["schema_version"] == builder.protocol.ENVIRONMENT_MANIFEST_SCHEMA
    assert fingerprint == builder.protocol.canonical_sha256(
        manifest["comparability"]
    )


def test_official_eval_log_rounding_is_bounded_without_weakening_exact_metrics():
    logged = builder.protocol.parse_actionformer_eval_log(
        "\n".join(
            (
                "|tIoU = 0.30: mAP = 82.13 (%)",
                "|tIoU = 0.40: mAP = 77.81 (%)",
                "|tIoU = 0.50: mAP = 70.95 (%)",
                "|tIoU = 0.60: mAP = 59.40 (%)",
                "|tIoU = 0.70: mAP = 43.87 (%)",
                "Average mAP: 66.83 (%)",
            )
        )
    )
    recomputed = {
        key: value
        for key, value in logged.items()
        if key != "average_mAP"
    }
    recomputed["average_mAP"] = sum(recomputed.values()) / 5.0

    maximum = builder.protocol._assert_metrics_close(
        logged,
        recomputed,
        atol=5.1e-5,
        label="official_log_vs_independent_recompute",
        left_is_logged=True,
    )

    assert maximum == pytest.approx(2.0e-5)
    with pytest.raises(
        builder.protocol.ProtocolError,
        match="average_mAP is not the five-threshold mean",
    ):
        builder.protocol._validate_metrics(logged, name="exact_metrics")


def test_official_eval_log_rounding_does_not_hide_inconsistent_average():
    logged = {
        "mAP@0.3": 0.8213,
        "mAP@0.4": 0.7781,
        "mAP@0.5": 0.7095,
        "mAP@0.6": 0.5940,
        "mAP@0.7": 0.4387,
        "average_mAP": 0.6700,
    }
    exact = dict(logged)
    exact["average_mAP"] = sum(
        exact[f"mAP@{threshold:.1f}"]
        for threshold in (0.3, 0.4, 0.5, 0.6, 0.7)
    ) / 5.0

    with pytest.raises(
        builder.protocol.ProtocolError,
        match="average_mAP is not the five-threshold mean",
    ):
        builder.protocol._assert_metrics_close(
            logged,
            exact,
            atol=5.1e-5,
            label="official_log_vs_independent_recompute",
            left_is_logged=True,
        )
