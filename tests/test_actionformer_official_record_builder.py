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


def test_annotation_class_map_and_official_test_count(tmp_path):
    database = {}
    for index in range(213):
        database[f"video_test_{index:07d}"] = {
            "subset": "test",
            "annotations": [
                {
                    "label": f"class-{index % 20}",
                    "label_id": index % 20,
                    "segment": [1.0, 2.0],
                }
            ],
        }
    annotation = tmp_path / "thumos14.json"
    annotation.write_text(json.dumps({"database": database}), encoding="utf-8")
    class_map, split_counts, videos = builder.parse_annotation(annotation)
    assert split_counts == {"test": 213}
    assert len(class_map["labels"]) == 20
    assert len(videos) == 213


def test_feature_manifest_checks_shape_and_finiteness(tmp_path):
    feature_dir = tmp_path / "i3d_features"
    feature_dir.mkdir()
    np.save(feature_dir / "video_a.npy", np.ones((3, 2048), dtype=np.float32))
    manifest = builder.build_feature_manifest(
        feature_dir,
        [("video_a", "test")],
    )
    assert manifest["feature_count"] == 1
    assert manifest["features"][0]["shape"] == [3, 2048]

    invalid = np.ones((2, 2048), dtype=np.float32)
    invalid[0, 10] = np.inf
    np.save(feature_dir / "video_b.npy", invalid)
    with pytest.raises(builder.protocol.ProtocolError, match="NaN/Inf"):
        builder.build_feature_manifest(
            feature_dir,
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


def test_environment_manifest_has_stable_comparability_fingerprint():
    manifest, fingerprint = builder.environment_manifest()
    assert manifest["schema_version"] == builder.protocol.ENVIRONMENT_MANIFEST_SCHEMA
    assert fingerprint == builder.protocol.canonical_sha256(
        manifest["comparability"]
    )
