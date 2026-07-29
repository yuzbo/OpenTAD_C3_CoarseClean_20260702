import copy
import hashlib
import importlib.util
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "bata" / "validate_actionformer_thumos_comparability.py"
SPEC = importlib.util.spec_from_file_location("comparability", MODULE_PATH)
comparability = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparability)


OFFICIAL_METRICS = {
    "average_mAP": 0.6683,
    "mAP@0.3": 0.8213,
    "mAP@0.4": 0.7780,
    "mAP@0.5": 0.7095,
    "mAP@0.6": 0.5940,
    "mAP@0.7": 0.4387,
}

SOURCE_CONFIG_LOADER = b"""\
import yaml

DEFAULTS = {}

def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)
"""


def _base_effective_config():
    return {
        "train_split": ["validation"],
        "val_split": ["test"],
        "init_rand_seed": comparability.OFFICIAL_TRAINING_SEED,
        "model": {"fpn_start_level": 0},
        "dataset": {
            "input_dim": 2048,
            "num_frames": 16,
            "feat_stride": 4,
            "max_seq_len": 2304,
        },
        "opt": {
            "type": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.05,
            "epochs": 30,
        },
        "loader": {"batch_size": 2},
        "test_cfg": {
            "pre_nms_thresh": 0.001,
            "pre_nms_topk": 2000,
            "nms_method": "soft",
            "nms_sigma": 0.5,
            "iou_threshold": 0.1,
            "min_score": 0.001,
            "max_seg_num": 200,
            "multiclass_nms": True,
            "voting_thresh": 0.7,
        },
    }


def _config_bytes(config):
    return (
        json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def _write_json(path, payload):
    return _write(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _receipt(path, *, include_md5=False):
    receipt = {
        "path": str(path),
        "sha256": comparability.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if include_md5:
        receipt["md5"] = comparability.md5_file(path)
    return receipt


def _eval_log(metrics):
    lines = []
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        lines.append(
            f"|tIoU = {threshold:.2f}: mAP = "
            f"{metrics[f'mAP@{threshold:.1f}'] * 100.0:.2f} (%)"
        )
    lines.append(f"Average mAP: {metrics['average_mAP'] * 100.0:.2f} (%)")
    return "\n".join(lines) + "\n"


def _patch_official_constants(monkeypatch, fixture):
    monkeypatch.setattr(
        comparability,
        "OFFICIAL_CONFIG_SHA256",
        fixture["config_sha256"],
    )
    monkeypatch.setattr(
        comparability,
        "OFFICIAL_README_SHA256",
        fixture["readme_sha256"],
    )
    monkeypatch.setattr(
        comparability,
        "OFFICIAL_EFFECTIVE_CONFIG_SHA256",
        fixture["effective_config_sha256"],
    )
    monkeypatch.setattr(
        comparability,
        "OFFICIAL_TRAIN_LOG_RAW_EFFECTIVE_CONFIG_SHA256",
        fixture["train_log_raw_effective_config_sha256"],
    )
    monkeypatch.setattr(
        comparability,
        "OFFICIAL_THUMOS_ARCHIVE_MD5",
        fixture["archive_md5"],
    )
    monkeypatch.setattr(
        comparability,
        "OFFICIAL_EVALUATOR_FILES",
        fixture["evaluator_files"],
    )
    monkeypatch.setattr(
        comparability,
        "OFFICIAL_EVALUATOR_FINGERPRINT_SHA256",
        fixture["evaluator_fingerprint"],
    )
    monkeypatch.setattr(
        comparability,
        "OFFICIAL_NOMINAL_SPLIT_COUNTS",
        fixture["nominal_split_counts"],
    )
    monkeypatch.setattr(
        comparability,
        "OFFICIAL_ANNOTATION_SPLIT_COUNTS",
        fixture["annotation_split_counts"],
    )
    monkeypatch.setattr(
        comparability,
        "OFFICIAL_ANNOTATION_DATABASE_VIDEO_COUNT",
        fixture["annotation_database_video_count"],
    )
    monkeypatch.setattr(
        comparability,
        "OFFICIAL_EVALUATED_VIDEO_COUNT",
        fixture["evaluated_video_count"],
    )
    monkeypatch.setattr(
        comparability,
        "OFFICIAL_FEATURE_INVENTORY_VIDEO_COUNT",
        fixture["feature_inventory_video_count"],
    )
    monkeypatch.setattr(
        comparability,
        "OFFICIAL_FEATURE_ONLY_UNANNOTATED_VIDEOS",
        tuple(fixture["feature_only_unannotated_videos"]),
    )
    expected_updates = {
        "source.config_sha256": fixture["config_sha256"],
        "source.readme_sha256": fixture["readme_sha256"],
        "model.effective_config_sha256": fixture["effective_config_sha256"],
        "dataset.data_archive_md5": fixture["archive_md5"],
        "dataset.nominal_split_counts": fixture["nominal_split_counts"],
        "dataset.annotation_split_counts": fixture["annotation_split_counts"],
        "dataset.annotation_database_video_count": fixture[
            "annotation_database_video_count"
        ],
        "dataset.evaluated_video_count": fixture["evaluated_video_count"],
        "dataset.blocked_videos": [],
        "dataset.feature_only_unannotated_videos": fixture[
            "feature_only_unannotated_videos"
        ],
        "input.feature_inventory_video_count": fixture[
            "feature_inventory_video_count"
        ],
        "input.annotation_feature_backed_video_count": fixture[
            "annotation_feature_backed_video_count"
        ],
        "input.evaluated_feature_backed_video_count": fixture[
            "evaluated_video_count"
        ],
        "input.missing_annotated_feature_videos": [],
        "evaluation.evaluator_sha256": fixture["evaluator_fingerprint"],
        "result.prediction_video_count": fixture["evaluated_video_count"],
    }
    for path, value in expected_updates.items():
        monkeypatch.setitem(
            comparability.OFFICIAL_ACTIONFORMER_EXPECTED,
            path,
            value,
        )


def _record(
    tmp_path,
    *,
    record_id="official-actionformer-main",
    stratum="official_reproduction",
    metrics=None,
    config_bytes=b"official-config\n",
):
    metrics = copy.deepcopy(metrics or OFFICIAL_METRICS)
    root = tmp_path / record_id
    root.mkdir(parents=True)

    config = _write(root / "configs" / "thumos_i3d.yaml", config_bytes)
    readme = _write(root / "README.md", b"official-readme\n")
    archive = _write(root / "thumos.tar.gz", b"official-thumos-archive\n")
    annotation_database = {
        "video_test_0000001": {
            "subset": "Test",
            "annotations": [
                {
                    "label_id": label_id,
                    "label": label,
                    "segment": [1.0, 2.0],
                }
                for label_id, label in enumerate(
                    comparability.OFFICIAL_THUMOS_CLASS_NAMES
                )
            ],
        },
        "video_validation_0000001": {
            "subset": "Validation",
            "annotations": [],
        },
    }
    annotation = _write_json(
        root / "thumos14.json",
        {"database": annotation_database},
    )
    class_map = _write_json(
        root / "OFFICIAL_CLASS_MAP.json",
        comparability._official_class_map_payload(),
    )
    feature_root = tmp_path / "shared_i3d_features"
    feature_ids_and_subsets = (
        ("video_test_0000001", "test"),
        ("video_test_0001292", None),
        ("video_validation_0000001", "validation"),
    )
    feature_entries = []
    for video_id, subset in feature_ids_and_subsets:
        feature_root.mkdir(parents=True, exist_ok=True)
        feature_path = feature_root / f"{video_id}.npy"
        np.save(feature_path, np.ones((1, 2048), dtype=np.float32))
        feature_entries.append(
            {
                "video_id": video_id,
                "annotation_subset": subset,
                "file": feature_path.name,
                "sha256": comparability.sha256_file(feature_path),
                "size_bytes": feature_path.stat().st_size,
                "dtype": "float32",
                "shape": [1, 2048],
            }
        )
    annotation_video_ids = sorted(annotation_database)
    evaluated_video_ids = ["video_test_0000001"]
    observation_manifest = _write_json(
        root / "OFFICIAL_FEATURE_MANIFEST.json",
        {
            "schema_version": comparability.OBSERVATION_MANIFEST_SCHEMA,
            "feature_family": "two_stream_i3d_kinetics",
            "feature_root": str(feature_root),
            "feature_inventory_video_count": 3,
            "annotation_feature_backed_video_count": 2,
            "evaluated_feature_backed_video_count": 1,
            "missing_annotated_feature_videos": [],
            "feature_only_unannotated_videos": ["video_test_0001292"],
            "annotation_video_ids_sha256": comparability.canonical_sha256(
                annotation_video_ids
            ),
            "evaluated_video_ids": evaluated_video_ids,
            "evaluated_video_ids_sha256": comparability.canonical_sha256(
                evaluated_video_ids
            ),
            "features": feature_entries,
        },
    )

    source_root = root / "official_source"
    evaluator_file = _write(source_root / "eval.py", b"official-evaluator\n")
    evaluator_files = {"eval.py": comparability.sha256_file(evaluator_file)}
    evaluator_fingerprint = comparability.canonical_sha256(
        {"files": evaluator_files}
    )
    evaluator_manifest = _write_json(
        root / "OFFICIAL_EVALUATOR_MANIFEST.json",
        {
            "schema_version": comparability.EVALUATOR_MANIFEST_SCHEMA,
            "source_root": str(source_root),
            "repository_url": comparability.OFFICIAL_REPOSITORY_URL,
            "commit": comparability.OFFICIAL_COMMIT,
            "tree": comparability.OFFICIAL_TREE,
            "clean": True,
            "files": evaluator_files,
        },
    )
    environment_payload = {
        "python": "test-python",
        "numpy": "test-numpy",
        "torch": "test-torch",
        "cuda": "test-cuda",
    }
    environment_manifest = _write_json(
        root / "OFFICIAL_ENVIRONMENT_MANIFEST.json",
        {
            "schema_version": comparability.ENVIRONMENT_MANIFEST_SCHEMA,
            "comparability": environment_payload,
            "execution": {"slurm_job_id": "unit-test"},
        },
    )
    environment_fingerprint = comparability.canonical_sha256(environment_payload)

    raw_predictions = root / "eval_results.pkl"
    with raw_predictions.open("wb") as handle:
        pickle.dump(
            {
                "video-id": ["video_test_0000001"],
                "t-start": np.asarray([1.0], dtype=np.float32),
                "t-end": np.asarray([2.0], dtype=np.float32),
                "label": np.asarray([0], dtype=np.int64),
                "score": np.asarray([0.9], dtype=np.float32),
            },
            handle,
        )
    checkpoint = _write(root / "epoch_034.pth.tar", b"checkpoint\n")
    effective_config = _base_effective_config()
    effective_config_sha = comparability.canonical_sha256(effective_config)
    train_log_config = copy.deepcopy(effective_config)
    if stratum == "official_reproduction":
        del train_log_config["model"]["fpn_start_level"]
    normalized_train_log_config, train_log_normalization = (
        comparability.normalize_actionformer_train_log_config(train_log_config)
    )
    assert normalized_train_log_config == effective_config
    train_log = _write(
        root / "train.log",
        repr(train_log_config) + "\nUsing model EMA ...\n",
    )
    eval_log = _write(root / "eval.log", _eval_log(metrics))

    archive_sha = comparability.sha256_file(archive)
    archive_md5 = comparability.md5_file(archive)
    annotation_sha = comparability.sha256_file(annotation)
    class_map_sha = comparability.sha256_file(class_map)
    observation_sha = comparability.sha256_file(observation_manifest)
    data_manifest = _write_json(
        root / "OFFICIAL_DATA_MANIFEST.json",
        {
            "schema_version": comparability.DATA_MANIFEST_SCHEMA,
            "archive_sha256": archive_sha,
            "archive_md5": archive_md5,
            "annotation_sha256": annotation_sha,
            "class_map_sha256": class_map_sha,
            "feature_manifest_sha256": observation_sha,
            "nominal_split_counts": {"test": 2, "validation": 1},
            "annotation_split_counts": {"test": 1, "validation": 1},
            "annotation_database_video_count": 2,
            "evaluated_video_count": 1,
            "evaluated_video_ids_sha256": comparability.canonical_sha256(
                evaluated_video_ids
            ),
            "blocked_videos": [],
            "feature_only_unannotated_videos": ["video_test_0001292"],
        },
    )

    raw_sha = comparability.sha256_file(raw_predictions)
    eval_log_sha = comparability.sha256_file(eval_log)
    evaluator_manifest_sha = comparability.sha256_file(evaluator_manifest)
    metric_attestation = _write_json(
        root / "OFFICIAL_METRIC_ATTESTATION.json",
        {
            "schema_version": comparability.METRIC_ATTESTATION_SCHEMA,
            "validation_pass": True,
            "raw_predictions_sha256": raw_sha,
            "eval_log_sha256": eval_log_sha,
            "evaluator_fingerprint_sha256": evaluator_fingerprint,
            "evaluator_manifest_sha256": evaluator_manifest_sha,
            "annotation_sha256": annotation_sha,
            "prediction_count": 1,
            "prediction_video_count": 1,
            "prediction_video_ids_sha256": comparability.canonical_sha256(
                evaluated_video_ids
            ),
            "evaluated_video_count": 1,
            "evaluated_video_ids_sha256": comparability.canonical_sha256(
                evaluated_video_ids
            ),
            "prediction_videos_within_evaluated_set": True,
            "logged_metrics": metrics,
            "recomputed_metrics": metrics,
            "max_abs_delta": 0.0,
        },
    )
    environment_manifest_sha = comparability.sha256_file(environment_manifest)
    metric_attestation_sha = comparability.sha256_file(metric_attestation)
    run_manifest = _write_json(
        root / "OFFICIAL_RUN_MANIFEST.json",
        {
            "schema_version": comparability.RUN_MANIFEST_SCHEMA,
            "source_commit": comparability.OFFICIAL_COMMIT,
            "source_tree": comparability.OFFICIAL_TREE,
            "config_sha256": comparability.sha256_file(config),
            "effective_config_sha256": effective_config_sha,
            "train_log_effective_config_sha256": effective_config_sha,
            "train_log_raw_effective_config_sha256": comparability.canonical_sha256(
                train_log_config
            ),
            "train_log_normalized_effective_config_sha256": (
                effective_config_sha
            ),
            "train_log_normalization": train_log_normalization,
            "train_log_normalization_sha256": comparability.canonical_sha256(
                train_log_normalization
            ),
            "data_manifest_sha256": comparability.sha256_file(data_manifest),
            "checkpoint_sha256": comparability.sha256_file(checkpoint),
            "raw_predictions_sha256": raw_sha,
            "train_log_sha256": comparability.sha256_file(train_log),
            "eval_log_sha256": eval_log_sha,
            "evaluator_manifest_sha256": evaluator_manifest_sha,
            "metric_attestation_sha256": metric_attestation_sha,
            "environment_manifest_sha256": environment_manifest_sha,
            "training_command": comparability.OFFICIAL_ACTIONFORMER_EXPECTED[
                "training.command"
            ],
            "evaluation_command": comparability.OFFICIAL_ACTIONFORMER_EXPECTED[
                "evaluation.command"
            ],
        },
    )

    paths = {
        "config": config,
        "readme": readme,
        "data_archive": archive,
        "data_manifest": data_manifest,
        "annotation": annotation,
        "class_map": class_map,
        "observation_manifest": observation_manifest,
        "checkpoint": checkpoint,
        "raw_predictions": raw_predictions,
        "train_log": train_log,
        "eval_log": eval_log,
        "evaluator_manifest": evaluator_manifest,
        "environment_manifest": environment_manifest,
        "metric_attestation": metric_attestation,
        "run_manifest": run_manifest,
    }
    receipts = {
        name: _receipt(path, include_md5=(name == "data_archive"))
        for name, path in paths.items()
    }
    record = {
        "schema_version": comparability.RECORD_SCHEMA,
        "record_id": record_id,
        "evidence_stratum": stratum,
        "protocol_family": "official_actionformer_i3d_stride4_v1",
        "source": {
            "repository_url": comparability.OFFICIAL_REPOSITORY_URL,
            "commit": comparability.OFFICIAL_COMMIT,
            "tree": comparability.OFFICIAL_TREE,
            "config_path": "configs/thumos_i3d.yaml",
            "config_sha256": receipts["config"]["sha256"],
            "readme_sha256": receipts["readme"]["sha256"],
            "implementation": "official_actionformer_release",
        },
        "dataset": {
            "name": "THUMOS14",
            "train_split": "validation",
            "eval_split": "test",
            "annotation_sha256": receipts["annotation"]["sha256"],
            "class_map_sha256": receipts["class_map"]["sha256"],
            "data_manifest_sha256": receipts["data_manifest"]["sha256"],
            "data_archive_md5": receipts["data_archive"]["md5"],
            "data_archive_sha256": receipts["data_archive"]["sha256"],
            "num_classes": 20,
            "nominal_split_counts": {"test": 2, "validation": 1},
            "annotation_split_counts": {"test": 1, "validation": 1},
            "annotation_database_video_count": 2,
            "evaluated_video_count": 1,
            "evaluated_video_ids_sha256": comparability.canonical_sha256(
                evaluated_video_ids
            ),
            "blocked_videos": [],
            "feature_only_unannotated_videos": ["video_test_0001292"],
        },
        "input": {
            "feature_family": "two_stream_i3d_kinetics",
            "feature_provenance_sha256": receipts[
                "observation_manifest"
            ]["sha256"],
            "feature_inventory_video_count": 3,
            "annotation_feature_backed_video_count": 2,
            "evaluated_feature_backed_video_count": 1,
            "missing_annotated_feature_videos": [],
            "input_dim": 2048,
            "clip_frames": 16,
            "frame_stride": 4,
            "seconds_per_feature": 4.0 / 30.0,
            "max_seq_len": 2304,
            "observation_budget": None,
            "observation_manifest_sha256": receipts[
                "observation_manifest"
            ]["sha256"],
            "selection_policy": "dense_all_i3d_features",
        },
        "model": {
            "detector": "ActionFormer",
            "head": "ActionFormerHead",
            "projection": "ActionFormerIdentityFPN",
            "query_geometry": "uniform_i3d_feature_grid",
            "effective_config_sha256": effective_config_sha,
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.05,
            "epochs": 30,
            "batch_size": 2,
            "seed": comparability.OFFICIAL_TRAINING_SEED,
            "amp": False,
            "ema": True,
            "checkpoint_rule": "ema_epoch_034",
            "command": (
                "python ./train.py ./configs/thumos_i3d.yaml --output reproduce"
            ),
        },
        "post_processing": {
            "pre_nms_thresh": 0.001,
            "pre_nms_topk": 2000,
            "use_soft_nms": True,
            "sigma": 0.5,
            "nms_iou_threshold": 0.1,
            "nms_min_score": 0.001,
            "max_seg_num": 200,
            "multiclass": True,
            "voting_thresh": 0.7,
            "score_fusion": False,
            "round_before_cross_window_nms": False,
            "round_after_cross_window_nms": False,
        },
        "evaluation": {
            "evaluator": "official_actionformer_ANETdetection",
            "evaluator_sha256": evaluator_fingerprint,
            "evaluator_manifest_sha256": receipts[
                "evaluator_manifest"
            ]["sha256"],
            "command": (
                "python ./eval.py ./configs/thumos_i3d.yaml <checkpoint>"
            ),
            "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        },
        "integrity": {
            "sampling_uses_gt": False,
            "inference_uses_test_labels": False,
            "inference_uses_teacher": False,
            "inference_uses_prediction_cache": False,
        },
        "environment": {
            "manifest_sha256": receipts["environment_manifest"]["sha256"],
            "comparability_fingerprint_sha256": environment_fingerprint,
        },
        "result": {
            "artifact_path": str(raw_predictions),
            "artifact_sha256": raw_sha,
            "raw_predictions_sha256": raw_sha,
            "checkpoint_sha256": receipts["checkpoint"]["sha256"],
            "eval_log_sha256": eval_log_sha,
            "metric_attestation_sha256": metric_attestation_sha,
            "run_manifest_sha256": receipts["run_manifest"]["sha256"],
            "metrics_source": (
                "official_actionformer_eval_log_and_independent_recompute"
            ),
            "prediction_count": 1,
            "prediction_video_count": 1,
            "prediction_video_ids_sha256": comparability.canonical_sha256(
                evaluated_video_ids
            ),
            "metrics": metrics,
        },
        "receipts": receipts,
    }
    fixture = {
        "config_sha256": receipts["config"]["sha256"],
        "readme_sha256": receipts["readme"]["sha256"],
        "effective_config_sha256": effective_config_sha,
        "train_log_raw_effective_config_sha256": comparability.canonical_sha256(
            train_log_config
        ),
        "archive_md5": receipts["data_archive"]["md5"],
        "evaluator_files": evaluator_files,
        "evaluator_fingerprint": evaluator_fingerprint,
        "nominal_split_counts": {"test": 2, "validation": 1},
        "annotation_split_counts": {"test": 1, "validation": 1},
        "annotation_database_video_count": 2,
        "evaluated_video_count": 1,
        "feature_inventory_video_count": 3,
        "annotation_feature_backed_video_count": 2,
        "feature_only_unannotated_videos": ["video_test_0001292"],
    }
    return record, fixture


def _git(repo, *arguments):
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _matched_head_source_repo(tmp_path):
    upstream = tmp_path / "matched_upstream.git"
    origin = tmp_path / "matched_origin.git"
    _git(tmp_path, "init", "--bare", str(upstream))
    _git(tmp_path, "init", "--bare", str(origin))
    repo = tmp_path / "matched_source_repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Unit Test")
    _git(repo, "config", "user.email", "unit@test.invalid")
    base_url = upstream.resolve().as_uri()
    candidate_url = origin.resolve().as_uri()
    _git(repo, "remote", "add", "upstream", base_url)
    _git(repo, "remote", "add", "origin", candidate_url)
    base_config = _base_effective_config()
    candidate_config = copy.deepcopy(base_config)
    candidate_config["model_intervention"] = {
        "head": "SupportDecoupledPhysicalQueryHead",
        "projection": "PhysTimeMeasureProjection",
    }
    base_config_bytes = _config_bytes(base_config)
    candidate_config_bytes = _config_bytes(candidate_config)
    _write(repo / "README.md", b"official-readme\n")
    _write(repo / "eval.py", b"official-evaluator\n")
    _write(repo / "train.py", b"official-trainer\n")
    _write(repo / "configs" / "thumos_i3d.yaml", base_config_bytes)
    _write(repo / "libs" / "core" / "config.py", SOURCE_CONFIG_LOADER)
    _write(repo / "libs" / "modeling" / "meta_archs.py", b"HEAD = 'official'\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "official base")
    base_commit = _git(repo, "rev-parse", "HEAD")
    base_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    base_ref = "refs/heads/main"
    candidate_ref = "refs/heads/candidate"
    _git(repo, "push", "upstream", f"{base_commit}:{base_ref}")

    _write(
        repo / "configs" / "thumos_i3d_random_k384_sdpq.yaml",
        candidate_config_bytes,
    )
    _write(
        repo / "libs" / "modeling" / "meta_archs.py",
        b"from .sdpq import SupportDecoupledPhysicalQueryHead\n",
    )
    _write(
        repo / "libs" / "modeling" / "sdpq.py",
        b"class SupportDecoupledPhysicalQueryHead:\n    pass\n",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "matched head")
    candidate_commit = _git(repo, "rev-parse", "HEAD")
    candidate_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    _git(repo, "push", "origin", f"{candidate_commit}:{candidate_ref}")
    return {
        "repo": repo,
        "base_url": base_url,
        "candidate_url": candidate_url,
        "base_commit": base_commit,
        "base_tree": base_tree,
        "candidate_commit": candidate_commit,
        "candidate_tree": candidate_tree,
        "base_ref": base_ref,
        "candidate_ref": candidate_ref,
        "base_config_bytes": base_config_bytes,
        "candidate_config_bytes": candidate_config_bytes,
    }


def _patch_official_source_identity(monkeypatch, source):
    monkeypatch.setattr(
        comparability,
        "OFFICIAL_REPOSITORY_URL",
        source["base_url"],
    )
    monkeypatch.setattr(comparability, "OFFICIAL_COMMIT", source["base_commit"])
    monkeypatch.setattr(comparability, "OFFICIAL_TREE", source["base_tree"])
    monkeypatch.setitem(
        comparability.OFFICIAL_ACTIONFORMER_EXPECTED,
        "source.repository_url",
        source["base_url"],
    )
    monkeypatch.setitem(
        comparability.OFFICIAL_ACTIONFORMER_EXPECTED,
        "source.commit",
        source["base_commit"],
    )
    monkeypatch.setitem(
        comparability.OFFICIAL_ACTIONFORMER_EXPECTED,
        "source.tree",
        source["base_tree"],
    )


def _refresh_candidate_manifests(candidate, source, attestation):
    candidate["source"].update(
        {
            "repository_url": source["candidate_url"],
            "commit": source["candidate_commit"],
            "tree": source["candidate_tree"],
            "config_path": "configs/thumos_i3d_random_k384_sdpq.yaml",
            "implementation": "official_actionformer_plus_sdpq",
        }
    )
    assert (
        candidate["source"]["config_sha256"]
        == attestation["candidate"]["config_blob_sha256"]
    )

    evaluator_path = Path(candidate["receipts"]["evaluator_manifest"]["path"])
    evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
    evaluator.update(
        {
            "source_root": str(source["repo"]),
            "repository_url": source["candidate_url"],
            "commit": source["candidate_commit"],
            "tree": source["candidate_tree"],
        }
    )
    _write_json(evaluator_path, evaluator)
    candidate["receipts"]["evaluator_manifest"] = _receipt(evaluator_path)
    evaluator_sha = candidate["receipts"]["evaluator_manifest"]["sha256"]
    candidate["evaluation"]["evaluator_manifest_sha256"] = evaluator_sha

    attestation_path = (
        Path(candidate["receipts"]["config"]["path"]).parents[1]
        / "SOURCE_DIFF_ATTESTATION.json"
    )
    _write_json(attestation_path, attestation)
    candidate["receipts"]["source_diff_attestation"] = _receipt(attestation_path)
    candidate["result"]["source_diff_attestation_sha256"] = candidate["receipts"][
        "source_diff_attestation"
    ]["sha256"]

    metric_path = Path(candidate["receipts"]["metric_attestation"]["path"])
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    metric["evaluator_manifest_sha256"] = evaluator_sha
    _write_json(metric_path, metric)
    candidate["receipts"]["metric_attestation"] = _receipt(metric_path)
    candidate["result"]["metric_attestation_sha256"] = candidate["receipts"][
        "metric_attestation"
    ]["sha256"]

    train_log_path = Path(candidate["receipts"]["train_log"]["path"])
    raw_train_config = comparability.parse_actionformer_train_log_config(
        train_log_path.read_text(encoding="utf-8")
    )
    normalized_train_config, train_log_normalization = (
        comparability.normalize_actionformer_train_log_config(raw_train_config)
    )
    assert (
        comparability.canonical_sha256(normalized_train_config)
        == candidate["model"]["effective_config_sha256"]
    )

    run_path = Path(candidate["receipts"]["run_manifest"]["path"])
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run.update(
        {
            "source_commit": source["candidate_commit"],
            "source_tree": source["candidate_tree"],
            "config_sha256": candidate["source"]["config_sha256"],
            "effective_config_sha256": candidate["model"][
                "effective_config_sha256"
            ],
            "train_log_effective_config_sha256": candidate["model"][
                "effective_config_sha256"
            ],
            "train_log_raw_effective_config_sha256": (
                comparability.canonical_sha256(raw_train_config)
            ),
            "train_log_normalized_effective_config_sha256": candidate["model"][
                "effective_config_sha256"
            ],
            "train_log_normalization": train_log_normalization,
            "train_log_normalization_sha256": comparability.canonical_sha256(
                train_log_normalization
            ),
            "train_log_sha256": candidate["receipts"]["train_log"]["sha256"],
            "evaluator_manifest_sha256": evaluator_sha,
            "metric_attestation_sha256": candidate["result"][
                "metric_attestation_sha256"
            ],
        }
    )
    _write_json(run_path, run)
    candidate["receipts"]["run_manifest"] = _receipt(run_path)
    candidate["result"]["run_manifest_sha256"] = candidate["receipts"][
        "run_manifest"
    ]["sha256"]


def _matched_head_records(tmp_path, monkeypatch):
    source = _matched_head_source_repo(tmp_path)
    _patch_official_source_identity(monkeypatch, source)
    reference, fixture = _record(
        tmp_path,
        record_id="official-reference",
        config_bytes=source["base_config_bytes"],
    )
    _patch_official_constants(monkeypatch, fixture)
    candidate, _ = _record(
        tmp_path,
        record_id="matched-head",
        stratum="matched_method_control",
        metrics={
            "average_mAP": 0.60,
            "mAP@0.3": 0.75,
            "mAP@0.4": 0.70,
            "mAP@0.5": 0.62,
            "mAP@0.6": 0.52,
            "mAP@0.7": 0.41,
        },
        config_bytes=source["candidate_config_bytes"],
    )
    candidate["model"]["head"] = "SupportDecoupledPhysicalQueryHead"
    candidate["model"]["projection"] = "PhysTimeMeasureProjection"
    train_log_path = Path(candidate["receipts"]["train_log"]["path"])
    train_config = comparability.parse_actionformer_train_log_config(
        train_log_path.read_text(encoding="utf-8")
    )
    train_config["model_intervention"] = {
        "head": "SupportDecoupledPhysicalQueryHead",
        "projection": "PhysTimeMeasureProjection",
    }
    _write(
        train_log_path,
        repr(train_config) + "\nUsing model EMA ...\n",
    )
    candidate["receipts"]["train_log"] = _receipt(train_log_path)
    normalized_train_config, _ = (
        comparability.normalize_actionformer_train_log_config(train_config)
    )
    candidate["model"]["effective_config_sha256"] = (
        comparability.canonical_sha256(normalized_train_config)
    )
    attestation = comparability.source_diff.collect_attestation(
        repository=source["repo"],
        base_commit=source["base_commit"],
        candidate_commit=source["candidate_commit"],
        base_repository_url=source["base_url"],
        candidate_repository_url=source["candidate_url"],
        base_remote="upstream",
        candidate_remote="origin",
        base_remote_ref=source["base_ref"],
        candidate_remote_ref=source["candidate_ref"],
        base_config_path="configs/thumos_i3d.yaml",
        candidate_config_path="configs/thumos_i3d_random_k384_sdpq.yaml",
        intervention="head_projection",
    )
    _refresh_candidate_manifests(candidate, source, attestation)
    return reference, candidate, source


def test_official_record_is_main_table_eligible(tmp_path, monkeypatch):
    record, fixture = _record(tmp_path)
    _patch_official_constants(monkeypatch, fixture)
    verdict = comparability.classify(record)
    assert verdict["main_table_eligible"] is True
    assert verdict["artifacts_verified"] is True
    assert verdict["official_actionformer_protocol_match"] is True
    assert verdict["claim_boundary"] == "paper_main_table"


def test_train_log_normalization_attestation_tamper_fails_closed(
    tmp_path,
    monkeypatch,
):
    record, fixture = _record(tmp_path)
    _patch_official_constants(monkeypatch, fixture)
    run_path = Path(record["receipts"]["run_manifest"]["path"])
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["train_log_normalization"]["applied_defaults"][0]["value"] = 1
    run["train_log_normalization_sha256"] = comparability.canonical_sha256(
        run["train_log_normalization"]
    )
    _write_json(run_path, run)
    record["receipts"]["run_manifest"] = _receipt(run_path)
    record["result"]["run_manifest_sha256"] = record["receipts"]["run_manifest"][
        "sha256"
    ]

    with pytest.raises(
        comparability.ProtocolError,
        match="train_log_normalization_sha256",
    ):
        comparability.classify(record)


def test_official_train_log_raw_config_identity_fails_closed(
    tmp_path,
    monkeypatch,
):
    record, fixture = _record(tmp_path)
    _patch_official_constants(monkeypatch, fixture)
    train_log_path = Path(record["receipts"]["train_log"]["path"])
    raw_config = comparability.parse_actionformer_train_log_config(
        train_log_path.read_text(encoding="utf-8")
    )
    raw_config["model"]["fpn_start_level"] = 0
    _write(
        train_log_path,
        repr(raw_config) + "\nUsing model EMA ...\n",
    )
    record["receipts"]["train_log"] = _receipt(train_log_path)
    normalized, normalization = (
        comparability.normalize_actionformer_train_log_config(raw_config)
    )
    assert (
        comparability.canonical_sha256(normalized)
        == record["model"]["effective_config_sha256"]
    )

    run_path = Path(record["receipts"]["run_manifest"]["path"])
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run.update(
        {
            "train_log_effective_config_sha256": record["model"][
                "effective_config_sha256"
            ],
            "train_log_raw_effective_config_sha256": (
                normalization["raw_effective_config_sha256"]
            ),
            "train_log_normalized_effective_config_sha256": (
                normalization["normalized_effective_config_sha256"]
            ),
            "train_log_normalization": normalization,
            "train_log_normalization_sha256": comparability.canonical_sha256(
                normalization
            ),
            "train_log_sha256": record["receipts"]["train_log"]["sha256"],
        }
    )
    _write_json(run_path, run)
    record["receipts"]["run_manifest"] = _receipt(run_path)
    record["result"]["run_manifest_sha256"] = record["receipts"]["run_manifest"][
        "sha256"
    ]

    with pytest.raises(
        comparability.ProtocolError,
        match="official released train-log raw effective-config SHA-256 mismatch",
    ):
        comparability.classify(record)


def test_legacy_v3_official_record_remains_readable(tmp_path, monkeypatch):
    record, fixture = _record(tmp_path, record_id="legacy-v3-official")
    _patch_official_constants(monkeypatch, fixture)
    record["schema_version"] = comparability.LEGACY_OFFICIAL_RECORD_SCHEMA

    verdict = comparability.classify(record)

    assert verdict["main_table_eligible"] is True
    assert verdict["official_actionformer_protocol_match"] is True


def test_non_official_protocol_stays_out_of_main_table(tmp_path, monkeypatch):
    record, fixture = _record(tmp_path)
    _patch_official_constants(monkeypatch, fixture)
    record["training"]["epochs"] = 60
    mismatches = comparability.official_expectation_mismatches(record)
    assert any(
        item["path"] == "training.epochs"
        for item in mismatches
    )
    with pytest.raises(
        comparability.ProtocolError,
        match="effective-config binding mismatch: training.epochs",
    ):
        comparability.classify(record)


def test_matched_control_requires_source_diff_attestation(tmp_path, monkeypatch):
    reference, fixture = _record(tmp_path, record_id="official-reference")
    _patch_official_constants(monkeypatch, fixture)
    candidate, _ = _record(
        tmp_path,
        record_id="matched-head",
        stratum="matched_method_control",
        metrics={
            "average_mAP": 0.60,
            "mAP@0.3": 0.75,
            "mAP@0.4": 0.70,
            "mAP@0.5": 0.62,
            "mAP@0.6": 0.52,
            "mAP@0.7": 0.41,
        },
        config_bytes=b"matched-head-config\n",
    )
    candidate["model"]["head"] = "SupportDecoupledPhysicalQueryHead"
    candidate["model"]["projection"] = "PhysTimeMeasureProjection"
    candidate["model"]["effective_config_sha256"] = candidate["source"][
        "config_sha256"
    ]

    with pytest.raises(comparability.ProtocolError, match="receipt set"):
        comparability.classify(
            candidate,
            reference=reference,
            intervention="head_projection",
        )

    candidate["post_processing"]["max_seg_num"] = 2000
    mismatches = comparability.compare_records(
        reference,
        candidate,
        "head_projection",
    )
    assert any(
        item["path"] == "post_processing.max_seg_num"
        for item in mismatches
    )


def test_live_source_diff_can_unlock_matched_main_table(tmp_path, monkeypatch):
    reference, candidate, _ = _matched_head_records(tmp_path, monkeypatch)

    verdict = comparability.classify(
        candidate,
        reference=reference,
        intervention="head_projection",
    )

    assert verdict["main_table_eligible"] is True
    assert verdict["matched_delta_allowed"] is True
    assert verdict["source_diff_attestation_verified"] is True
    assert verdict["claim_boundary"] == "paper_main_table"
    assert verdict["pair_mismatches"] == []


def test_source_diff_tamper_fails_closed_after_receipt_rebinding(
    tmp_path,
    monkeypatch,
):
    reference, candidate, _ = _matched_head_records(tmp_path, monkeypatch)
    attestation_path = Path(
        candidate["receipts"]["source_diff_attestation"]["path"]
    )
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["diff"]["binary_sha256"] = "0" * 64
    _write_json(attestation_path, attestation)
    candidate["receipts"]["source_diff_attestation"] = _receipt(attestation_path)
    candidate["result"]["source_diff_attestation_sha256"] = candidate["receipts"][
        "source_diff_attestation"
    ]["sha256"]

    with pytest.raises(comparability.ProtocolError, match="source-diff attestation"):
        comparability.classify(
            candidate,
            reference=reference,
            intervention="head_projection",
        )


def test_source_diff_intervention_and_base_are_bound(tmp_path, monkeypatch):
    reference, candidate, _ = _matched_head_records(tmp_path, monkeypatch)
    attestation = comparability._verify_source_diff_attestation(
        candidate,
        candidate["receipts"],
    )

    with pytest.raises(comparability.ProtocolError, match="intervention"):
        comparability._bind_source_diff_to_reference(
            attestation,
            reference,
            "selection_budget",
        )

    changed_reference = copy.deepcopy(reference)
    changed_reference["source"]["config_path"] = "configs/not_the_base.yaml"
    with pytest.raises(comparability.ProtocolError, match="base binding"):
        comparability._bind_source_diff_to_reference(
            attestation,
            changed_reference,
            "head_projection",
        )


def test_official_record_rejects_extra_source_diff_receipt(tmp_path, monkeypatch):
    record, fixture = _record(tmp_path)
    _patch_official_constants(monkeypatch, fixture)
    path = _write_json(
        tmp_path / "unexpected_source_diff.json",
        {"schema_version": "unexpected"},
    )
    record["receipts"]["source_diff_attestation"] = _receipt(path)

    with pytest.raises(comparability.ProtocolError, match="receipt set"):
        comparability.validate_record(record)


def test_arbitrary_difference_prefix_is_rejected(tmp_path, monkeypatch):
    reference, fixture = _record(tmp_path, record_id="official-reference")
    _patch_official_constants(monkeypatch, fixture)
    candidate, _ = _record(
        tmp_path,
        record_id="matched",
        stratum="matched_method_control",
    )
    with pytest.raises(comparability.ProtocolError, match="unknown matched intervention"):
        comparability.compare_records(reference, candidate, "result")


def test_non_official_reference_cannot_anchor_matched_delta(tmp_path, monkeypatch):
    reference, candidate, _ = _matched_head_records(tmp_path, monkeypatch)
    reference["evidence_stratum"] = "external_reference_only"
    verdict = comparability.classify(
        candidate,
        reference=reference,
        intervention="head_projection",
    )
    assert verdict["main_table_eligible"] is False
    assert any("official reproduction" in reason for reason in verdict["ineligibility_reasons"])


def test_nonofficial_protocol_reference_cannot_anchor_matched_delta(
    tmp_path,
    monkeypatch,
):
    reference, candidate, _ = _matched_head_records(tmp_path, monkeypatch)
    reference["source"]["implementation"] = "nonofficial_actionformer_wrapper"

    verdict = comparability.classify(
        candidate,
        reference=reference,
        intervention="head_projection",
    )

    assert verdict["main_table_eligible"] is False
    assert verdict["matched_delta_allowed"] is False
    assert verdict["reference_official_protocol_mismatches"]
    assert any(
        "reference does not match" in reason
        for reason in verdict["ineligibility_reasons"]
    )


def test_eval_log_metric_binding_fails_closed(tmp_path, monkeypatch):
    record, fixture = _record(tmp_path)
    _patch_official_constants(monkeypatch, fixture)
    eval_log = Path(record["receipts"]["eval_log"]["path"])
    eval_log.write_text(_eval_log({**OFFICIAL_METRICS, "mAP@0.7": 0.4287, "average_mAP": 0.6663}))
    record["receipts"]["eval_log"] = _receipt(eval_log)
    record["result"]["eval_log_sha256"] = record["receipts"]["eval_log"]["sha256"]
    with pytest.raises(comparability.ProtocolError, match="attestation"):
        comparability.validate_record(record)


def test_readme_and_evaluator_bytes_are_live_verified(tmp_path, monkeypatch):
    record, fixture = _record(tmp_path)
    _patch_official_constants(monkeypatch, fixture)
    Path(record["receipts"]["readme"]["path"]).write_text("tampered\n")
    with pytest.raises(comparability.ProtocolError, match="receipt SHA-256"):
        comparability.validate_record(record)

    record, fixture = _record(tmp_path, record_id="fresh-official")
    _patch_official_constants(monkeypatch, fixture)
    manifest = json.loads(
        Path(record["receipts"]["evaluator_manifest"]["path"]).read_text()
    )
    source_file = Path(manifest["source_root"]) / "eval.py"
    source_file.write_text("tampered evaluator\n")
    with pytest.raises(comparability.ProtocolError, match="evaluator source file"):
        comparability.validate_record(record)


def test_class_map_and_annotation_labels_are_semantically_pinned(tmp_path):
    record, _ = _record(tmp_path)
    class_map_path = Path(record["receipts"]["class_map"]["path"])
    payload = json.loads(class_map_path.read_text(encoding="utf-8"))
    payload["labels"][0]["label"] = "NotBaseballPitch"
    _write_json(class_map_path, payload)

    with pytest.raises(comparability.ProtocolError, match="class map"):
        comparability._annotation_video_sets(
            record["receipts"]["annotation"]["path"],
            class_map_path,
        )


def test_feature_manifest_reloads_real_npy_content(tmp_path):
    record, _ = _record(tmp_path)
    manifest_path = Path(record["receipts"]["observation_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_root = Path(manifest["feature_root"])
    target = feature_root / "video_test_0001292.npy"
    np.save(target, np.ones((1, 1024), dtype=np.float32))
    entry = next(
        item
        for item in manifest["features"]
        if item["video_id"] == "video_test_0001292"
    )
    entry["sha256"] = comparability.sha256_file(target)
    entry["size_bytes"] = target.stat().st_size
    entry["shape"] = [1, 1024]
    _write_json(manifest_path, manifest)
    annotation_sets, _ = comparability._annotation_video_sets(
        record["receipts"]["annotation"]["path"],
        record["receipts"]["class_map"]["path"],
    )

    with pytest.raises(comparability.ProtocolError, match="T x 2048"):
        comparability._verify_feature_manifest(
            record,
            record["receipts"],
            annotation_sets,
        )


def test_unmanifested_feature_inventory_fails_closed(tmp_path, monkeypatch):
    record, fixture = _record(tmp_path)
    _patch_official_constants(monkeypatch, fixture)
    manifest = json.loads(
        Path(record["receipts"]["observation_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    np.save(
        Path(manifest["feature_root"]) / "video_test_rogue.npy",
        np.ones((1, 2048), dtype=np.float32),
    )

    with pytest.raises(comparability.ProtocolError, match="full .npy inventory"):
        comparability.validate_record(record)


def test_raw_prediction_video_ids_must_be_evaluated(tmp_path):
    record, _ = _record(tmp_path)
    raw_path = Path(record["receipts"]["raw_predictions"]["path"])
    with raw_path.open("wb") as handle:
        pickle.dump(
            {
                "video-id": ["video_test_not_evaluated"],
                "t-start": np.asarray([1.0], dtype=np.float32),
                "t-end": np.asarray([2.0], dtype=np.float32),
                "label": np.asarray([0], dtype=np.int64),
                "score": np.asarray([0.9], dtype=np.float32),
            },
            handle,
        )

    with pytest.raises(comparability.ProtocolError, match="outside the evaluated set"):
        comparability._verify_raw_prediction_identity(
            record,
            record["receipts"],
            ["video_test_0000001"],
        )


def test_official_raw_predictions_must_cover_exact_evaluated_set(tmp_path):
    record, _ = _record(tmp_path)
    raw_path = Path(record["receipts"]["raw_predictions"]["path"])
    with raw_path.open("wb") as handle:
        pickle.dump(
            {
                "video-id": [],
                "t-start": np.asarray([], dtype=np.float32),
                "t-end": np.asarray([], dtype=np.float32),
                "label": np.asarray([], dtype=np.int64),
                "score": np.asarray([], dtype=np.float32),
            },
            handle,
        )

    with pytest.raises(comparability.ProtocolError, match="exact"):
        comparability._verify_raw_prediction_identity(
            record,
            record["receipts"],
            ["video_test_0000001"],
        )


def test_forbidden_test_information_fails_closed(tmp_path, monkeypatch):
    record, fixture = _record(tmp_path)
    _patch_official_constants(monkeypatch, fixture)
    record["integrity"]["inference_uses_test_labels"] = True
    with pytest.raises(comparability.ProtocolError, match="forbidden"):
        comparability.validate_record(record)


def test_historical_reference_is_never_a_matched_delta(tmp_path, monkeypatch):
    record, fixture = _record(
        tmp_path,
        record_id="legacy-63-61",
        stratum="external_reference_only",
    )
    _patch_official_constants(monkeypatch, fixture)
    verdict = comparability.classify(record)
    assert verdict["main_table_eligible"] is False
    assert verdict["matched_delta_allowed"] is False
    assert verdict["claim_boundary"] == "external_reference_only"
