#!/usr/bin/env python3
"""Build a paper-eligible record from an exact official ActionFormer run.

This command is intentionally fail-closed.  It verifies the pinned official
source snapshot and THUMOS archive, hashes every I3D feature, validates the raw
prediction pickle, independently invokes the pinned official evaluator, and
only then asks the comparability gate to issue a main-table verdict.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import pickle
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


PROTOCOL_PATH = Path(__file__).with_name(
    "validate_actionformer_thumos_comparability.py"
)
SPEC = importlib.util.spec_from_file_location(
    "actionformer_comparability_protocol",
    PROTOCOL_PATH,
)
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)


def _run_git(repo, *arguments):
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_blob_sha256(repo, relative):
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def verify_official_source(repo):
    repo = Path(repo).resolve()
    if not (repo / ".git").exists():
        raise protocol.ProtocolError(f"official source is not a Git repository: {repo}")
    commit = _run_git(repo, "rev-parse", "HEAD")
    tree = _run_git(repo, "rev-parse", "HEAD^{tree}")
    cleanliness = _run_git(repo, "status", "--porcelain")
    if commit != protocol.OFFICIAL_COMMIT:
        raise protocol.ProtocolError(
            f"official source commit mismatch: {commit} != {protocol.OFFICIAL_COMMIT}"
        )
    if tree != protocol.OFFICIAL_TREE:
        raise protocol.ProtocolError(
            f"official source tree mismatch: {tree} != {protocol.OFFICIAL_TREE}"
        )
    if cleanliness:
        raise protocol.ProtocolError("official source worktree is not clean")
    config = repo / "configs" / "thumos_i3d.yaml"
    readme = repo / "README.md"
    if _git_blob_sha256(repo, "configs/thumos_i3d.yaml") != (
        protocol.OFFICIAL_CONFIG_SHA256
    ):
        raise protocol.ProtocolError("official ActionFormer config hash mismatch")
    if _git_blob_sha256(repo, "README.md") != protocol.OFFICIAL_README_SHA256:
        raise protocol.ProtocolError("official ActionFormer README hash mismatch")
    return repo, commit, tree


def load_effective_config(repo, relative="configs/thumos_i3d.yaml"):
    previous_path = list(sys.path)
    previous_modules = {
        key: value
        for key, value in sys.modules.items()
        if key == "libs" or key.startswith("libs.")
    }
    for key in list(previous_modules):
        del sys.modules[key]
    sys.path.insert(0, str(repo))
    try:
        config_module = importlib.import_module("libs.core.config")
        config = config_module.load_config(
            str(repo / relative),
            defaults=copy.deepcopy(config_module.DEFAULTS),
        )
    except Exception as error:
        raise protocol.ProtocolError(
            f"cannot load pinned official effective config: {error}"
        ) from error
    finally:
        sys.path[:] = previous_path
        for key in list(sys.modules):
            if key == "libs" or key.startswith("libs."):
                del sys.modules[key]
        sys.modules.update(previous_modules)
    if not isinstance(config, dict) or not config:
        raise protocol.ProtocolError("pinned official effective config is empty")
    return config


def build_evaluator_manifest(repo, commit, tree):
    files = {}
    for relative, expected_sha in sorted(protocol.OFFICIAL_EVALUATOR_FILES.items()):
        path = repo / relative
        if not path.is_file():
            raise protocol.ProtocolError(f"missing official evaluator file: {relative}")
        observed = _git_blob_sha256(repo, relative)
        if observed != expected_sha:
            raise protocol.ProtocolError(
                f"official evaluator file hash mismatch: {relative}"
            )
        files[relative] = observed
    fingerprint = protocol.canonical_sha256({"files": files})
    if fingerprint != protocol.OFFICIAL_EVALUATOR_FINGERPRINT_SHA256:
        raise protocol.ProtocolError("official evaluator fingerprint mismatch")
    return {
        "schema_version": protocol.EVALUATOR_MANIFEST_SCHEMA,
        "source_root": str(repo),
        "repository_url": protocol.OFFICIAL_REPOSITORY_URL,
        "commit": commit,
        "tree": tree,
        "clean": True,
        "files": files,
    }, fingerprint


def parse_annotation(annotation_path):
    payload = protocol.load_json(annotation_path)
    database = payload.get("database")
    if not isinstance(database, dict) or not database:
        raise protocol.ProtocolError("THUMOS annotation database is empty")
    labels_by_id = {}
    split_counts = {}
    videos = []
    for video_id, record in sorted(database.items()):
        if not isinstance(record, dict):
            raise protocol.ProtocolError(f"invalid annotation record: {video_id}")
        raw_subset = record.get("subset")
        if not isinstance(raw_subset, str):
            raise protocol.ProtocolError(f"missing subset for video: {video_id}")
        subset = raw_subset.casefold()
        if subset not in {"validation", "test"}:
            raise protocol.ProtocolError(
                f"unexpected THUMOS annotation subset: {video_id}: {raw_subset}"
            )
        split_counts[subset] = split_counts.get(subset, 0) + 1
        videos.append((video_id, subset))
        actions = record.get("annotations", [])
        if not isinstance(actions, list):
            raise protocol.ProtocolError(
                f"invalid annotation list in video: {video_id}"
            )
        for action in actions:
            if not isinstance(action, dict):
                raise protocol.ProtocolError(
                    f"invalid action annotation in video: {video_id}"
                )
            label = action.get("label")
            label_id = action.get("label_id")
            if type(label_id) is not int or not isinstance(label, str):
                raise protocol.ProtocolError(
                    f"invalid class mapping in video: {video_id}"
                )
            previous = labels_by_id.setdefault(label_id, label)
            if previous != label:
                raise protocol.ProtocolError(
                    f"conflicting label name for class {label_id}"
                )
    if sorted(labels_by_id) != list(range(20)):
        raise protocol.ProtocolError("THUMOS annotation does not define label IDs 0..19")
    expected_labels = {
        label_id: label
        for label_id, label in enumerate(protocol.OFFICIAL_THUMOS_CLASS_NAMES)
    }
    if labels_by_id != expected_labels:
        raise protocol.ProtocolError(
            "pinned official THUMOS label-ID mapping mismatch"
        )
    if split_counts != protocol.OFFICIAL_ANNOTATION_SPLIT_COUNTS:
        raise protocol.ProtocolError(
            "pinned official THUMOS annotation split mismatch: "
            f"{split_counts} != {protocol.OFFICIAL_ANNOTATION_SPLIT_COUNTS}"
        )
    if len(database) != protocol.OFFICIAL_ANNOTATION_DATABASE_VIDEO_COUNT:
        raise protocol.ProtocolError(
            "pinned official THUMOS annotation database count mismatch: "
            f"{len(database)} != {protocol.OFFICIAL_ANNOTATION_DATABASE_VIDEO_COUNT}"
        )
    class_map = {
        "schema_version": "actionformer_thumos_class_map_v1",
        "labels": [
            {"label_id": label_id, "label": labels_by_id[label_id]}
            for label_id in range(20)
        ],
    }
    return class_map, split_counts, videos


def _array_is_finite(array, chunk_rows=4096):
    for start in range(0, int(array.shape[0]), chunk_rows):
        stop = min(start + chunk_rows, int(array.shape[0]))
        if not np.isfinite(np.asarray(array[start:stop])).all():
            return False
    return True


def build_feature_manifest(feature_dir, videos):
    feature_dir = Path(feature_dir).resolve()
    if not feature_dir.is_dir():
        raise protocol.ProtocolError(f"I3D feature directory is missing: {feature_dir}")
    annotation_subsets = dict(videos)
    annotation_ids = set(annotation_subsets)
    feature_paths = sorted(feature_dir.glob("*.npy"), key=lambda path: path.name)
    feature_ids = [path.stem for path in feature_paths]
    if len(set(feature_ids)) != len(feature_ids):
        raise protocol.ProtocolError("official feature inventory contains duplicate video IDs")
    feature_id_set = set(feature_ids)
    missing = sorted(annotation_ids - feature_id_set)
    feature_only = sorted(feature_id_set - annotation_ids)
    expected_feature_only = sorted(
        protocol.OFFICIAL_FEATURE_ONLY_UNANNOTATED_VIDEOS
    )
    if missing:
        raise protocol.ProtocolError(
            f"official feature set is incomplete: {len(missing)} annotated videos missing"
        )
    if feature_only != expected_feature_only:
        raise protocol.ProtocolError(
            "pinned official feature-only video set mismatch: "
            f"{feature_only} != {expected_feature_only}"
        )
    if len(feature_paths) != protocol.OFFICIAL_FEATURE_INVENTORY_VIDEO_COUNT:
        raise protocol.ProtocolError(
            "pinned official feature inventory count mismatch: "
            f"{len(feature_paths)} != {protocol.OFFICIAL_FEATURE_INVENTORY_VIDEO_COUNT}"
        )

    entries = []
    for path in feature_paths:
        video_id = path.stem
        subset = annotation_subsets.get(video_id)
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as error:
            raise protocol.ProtocolError(
                f"cannot load I3D feature: {video_id}: {error}"
            ) from error
        if array.ndim != 2 or int(array.shape[1]) != 2048:
            raise protocol.ProtocolError(
                f"I3D feature must be T x 2048: {video_id}: {array.shape}"
            )
        if int(array.shape[0]) <= 0:
            raise protocol.ProtocolError(f"I3D feature is empty: {video_id}")
        if not _array_is_finite(array):
            raise protocol.ProtocolError(f"I3D feature contains NaN/Inf: {video_id}")
        entries.append(
            {
                "video_id": video_id,
                "annotation_subset": subset,
                "file": f"{video_id}.npy",
                "sha256": protocol.sha256_file(path),
                "size_bytes": path.stat().st_size,
                "dtype": str(array.dtype),
                "shape": [int(value) for value in array.shape],
            }
        )
    evaluated_video_ids = sorted(
        video_id
        for video_id, subset in videos
        if subset == "test" and video_id in feature_id_set
    )
    annotation_feature_backed_ids = sorted(annotation_ids & feature_id_set)
    if len(evaluated_video_ids) != protocol.OFFICIAL_EVALUATED_VIDEO_COUNT:
        raise protocol.ProtocolError(
            "pinned official evaluated-video count mismatch: "
            f"{len(evaluated_video_ids)} != {protocol.OFFICIAL_EVALUATED_VIDEO_COUNT}"
        )
    return {
        "schema_version": protocol.OBSERVATION_MANIFEST_SCHEMA,
        "feature_family": "two_stream_i3d_kinetics",
        "feature_root": str(feature_dir),
        "feature_inventory_video_count": len(entries),
        "annotation_feature_backed_video_count": len(
            annotation_feature_backed_ids
        ),
        "evaluated_feature_backed_video_count": len(evaluated_video_ids),
        "missing_annotated_feature_videos": missing,
        "feature_only_unannotated_videos": feature_only,
        "annotation_video_ids_sha256": protocol.canonical_sha256(
            sorted(annotation_ids)
        ),
        "evaluated_video_ids": evaluated_video_ids,
        "evaluated_video_ids_sha256": protocol.canonical_sha256(
            evaluated_video_ids
        ),
        "features": entries,
    }


def load_and_validate_raw_predictions(path):
    path = Path(path).resolve()
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:
        raise protocol.ProtocolError(
            f"cannot load official eval_results.pkl: {error}"
        ) from error
    required = ("video-id", "t-start", "t-end", "label", "score")
    if not isinstance(payload, dict) or set(payload) != set(required):
        raise protocol.ProtocolError(
            "official raw prediction pickle has an unexpected key set"
        )
    arrays = {
        "t-start": np.asarray(payload["t-start"]),
        "t-end": np.asarray(payload["t-end"]),
        "label": np.asarray(payload["label"]),
        "score": np.asarray(payload["score"]),
    }
    video_ids = list(payload["video-id"])
    counts = {len(video_ids)} | {int(array.size) for array in arrays.values()}
    if len(counts) != 1:
        raise protocol.ProtocolError("official raw prediction arrays have unequal lengths")
    count = counts.pop()
    for key in ("t-start", "t-end", "score"):
        array = np.asarray(arrays[key], dtype=np.float64).reshape(-1)
        if not np.isfinite(array).all():
            raise protocol.ProtocolError(f"raw predictions contain NaN/Inf: {key}")
        arrays[key] = array
    raw_labels = np.asarray(arrays["label"]).reshape(-1)
    labels = raw_labels.astype(np.int64)
    if not np.array_equal(raw_labels, labels):
        raise protocol.ProtocolError("raw prediction labels are not integers")
    if count and (int(labels.min()) < 0 or int(labels.max()) >= 20):
        raise protocol.ProtocolError("raw prediction labels are outside 0..19")
    arrays["label"] = labels
    if np.any(arrays["t-end"] <= arrays["t-start"]):
        raise protocol.ProtocolError("raw predictions contain non-positive segments")
    if np.any(arrays["score"] < 0.0):
        raise protocol.ProtocolError("raw predictions contain negative scores")
    if any(not isinstance(video_id, str) or not video_id for video_id in video_ids):
        raise protocol.ProtocolError("raw predictions contain invalid video IDs")
    normalized = {
        "video-id": video_ids,
        "t-start": arrays["t-start"],
        "t-end": arrays["t-end"],
        "label": arrays["label"],
        "score": arrays["score"],
    }
    return normalized, count


def recompute_official_metrics(repo, annotation_path, raw_predictions):
    previous_path = list(sys.path)
    previous_modules = {
        key: value
        for key, value in sys.modules.items()
        if key == "libs" or key.startswith("libs.")
    }
    for key in list(previous_modules):
        del sys.modules[key]
    sys.path.insert(0, str(repo))
    try:
        metrics_module = importlib.import_module("libs.utils.metrics")
        evaluator = metrics_module.ANETdetection(
            str(annotation_path),
            split="test",
            tiou_thresholds=np.linspace(0.3, 0.7, 5),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            mAP, average_mAP, _ = evaluator.evaluate(raw_predictions, verbose=True)
    except Exception as error:
        raise protocol.ProtocolError(
            f"pinned official evaluator recomputation failed: {error}"
        ) from error
    finally:
        sys.path[:] = previous_path
        for key in list(sys.modules):
            if key == "libs" or key.startswith("libs."):
                del sys.modules[key]
        sys.modules.update(previous_modules)
    values = np.asarray(mAP, dtype=np.float64).reshape(-1)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise protocol.ProtocolError("official evaluator returned invalid mAP values")
    metrics = {
        f"mAP@{threshold:.1f}": float(value)
        for threshold, value in zip((0.3, 0.4, 0.5, 0.6, 0.7), values)
    }
    metrics["average_mAP"] = float(average_mAP)
    protocol._validate_metrics(metrics, name="independent_official_metrics")
    return metrics


def environment_manifest():
    package_names = (
        "joblib",
        "numpy",
        "pandas",
        "PyYAML",
        "scipy",
        "torch",
        "torchvision",
    )
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    probe_source = r"""
import json
import torch
payload = {
    "probe_pass": True,
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cudnn_version": (
        None if not torch.backends.cudnn.is_available()
        else torch.backends.cudnn.version()
    ),
    "cuda_available": bool(torch.cuda.is_available()),
    "gpu_name": (
        None if not torch.cuda.is_available()
        else torch.cuda.get_device_name(0)
    ),
}
print(json.dumps(payload, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe_source],
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        try:
            torch_payload = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise protocol.ProtocolError(
                "torch environment probe returned invalid JSON"
            ) from error
    else:
        torch_payload = {
            "probe_pass": False,
            "returncode": completed.returncode,
            "stderr_sha256": hashlib.sha256(completed.stderr or b"").hexdigest(),
        }
    comparability = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_runtime": torch_payload,
    }
    execution = {
        key: os.environ.get(key)
        for key in (
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_CLUSTER_NAME",
            "SLURMD_NODENAME",
            "CUDA_VISIBLE_DEVICES",
        )
    }
    return {
        "schema_version": protocol.ENVIRONMENT_MANIFEST_SCHEMA,
        "comparability": comparability,
        "execution": execution,
    }, protocol.canonical_sha256(comparability)


def _write_json(path, payload):
    protocol.atomic_write_json(path, payload)
    return Path(path).resolve()


def _receipt(path, *, include_md5=False):
    path = Path(path).resolve()
    receipt = {
        "path": str(path),
        "sha256": protocol.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if include_md5:
        receipt["md5"] = protocol.md5_file(path)
    return receipt


def build_record(args):
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise protocol.ProtocolError(
            f"official record output directory already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    repo, commit, tree = verify_official_source(args.official_repo)
    archive = Path(args.data_archive).resolve()
    annotation = Path(args.annotation).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    raw_predictions_path = Path(args.raw_predictions).resolve()
    train_log = Path(args.train_log).resolve()
    eval_log = Path(args.eval_log).resolve()
    for name, path in {
        "data_archive": archive,
        "annotation": annotation,
        "checkpoint": checkpoint,
        "raw_predictions": raw_predictions_path,
        "train_log": train_log,
        "eval_log": eval_log,
    }.items():
        if not path.is_file():
            raise protocol.ProtocolError(f"missing official run artifact: {name}: {path}")
    if protocol.md5_file(archive) != protocol.OFFICIAL_THUMOS_ARCHIVE_MD5:
        raise protocol.ProtocolError("THUMOS archive MD5 is not the official release")
    effective_config = load_effective_config(repo)
    train_log_config = protocol.parse_actionformer_train_log_config(
        train_log.read_text(encoding="utf-8", errors="strict")
    )
    effective_config_sha = protocol.canonical_sha256(effective_config)
    train_log_config_sha = protocol.canonical_sha256(train_log_config)
    if effective_config_sha != train_log_config_sha:
        raise protocol.ProtocolError(
            "released train log effective config differs from the pinned source config"
        )
    if effective_config_sha != protocol.OFFICIAL_EFFECTIVE_CONFIG_SHA256:
        raise protocol.ProtocolError(
            "pinned official effective-config SHA-256 mismatch"
        )
    if (
        effective_config.get("init_rand_seed")
        != protocol.OFFICIAL_TRAINING_SEED
    ):
        raise protocol.ProtocolError("pinned official training seed mismatch")

    class_map, split_counts, videos = parse_annotation(annotation)
    feature_manifest = build_feature_manifest(args.feature_dir, videos)
    evaluator_payload, evaluator_fingerprint = build_evaluator_manifest(
        repo, commit, tree
    )
    environment_payload, environment_fingerprint = environment_manifest()
    if environment_payload["comparability"]["torch_runtime"].get("probe_pass") is not True:
        raise protocol.ProtocolError("torch/CUDA environment probe failed")

    class_map_path = _write_json(output_dir / "OFFICIAL_CLASS_MAP.json", class_map)
    feature_manifest_path = _write_json(
        output_dir / "OFFICIAL_FEATURE_MANIFEST.json",
        feature_manifest,
    )
    evaluator_manifest_path = _write_json(
        output_dir / "OFFICIAL_EVALUATOR_MANIFEST.json",
        evaluator_payload,
    )
    environment_manifest_path = _write_json(
        output_dir / "OFFICIAL_ENVIRONMENT_MANIFEST.json",
        environment_payload,
    )

    archive_sha = protocol.sha256_file(archive)
    archive_md5 = protocol.md5_file(archive)
    annotation_sha = protocol.sha256_file(annotation)
    class_map_sha = protocol.sha256_file(class_map_path)
    feature_manifest_sha = protocol.sha256_file(feature_manifest_path)
    evaluated_video_ids = feature_manifest["evaluated_video_ids"]
    evaluated_video_ids_sha = feature_manifest["evaluated_video_ids_sha256"]
    data_manifest = {
        "schema_version": protocol.DATA_MANIFEST_SCHEMA,
        "archive_sha256": archive_sha,
        "archive_md5": archive_md5,
        "annotation_sha256": annotation_sha,
        "class_map_sha256": class_map_sha,
        "feature_manifest_sha256": feature_manifest_sha,
        "nominal_split_counts": protocol.OFFICIAL_NOMINAL_SPLIT_COUNTS,
        "annotation_split_counts": split_counts,
        "annotation_database_video_count": len(videos),
        "evaluated_video_count": len(evaluated_video_ids),
        "evaluated_video_ids_sha256": evaluated_video_ids_sha,
        "blocked_videos": [],
        "feature_only_unannotated_videos": feature_manifest[
            "feature_only_unannotated_videos"
        ],
    }
    data_manifest_path = _write_json(
        output_dir / "OFFICIAL_DATA_MANIFEST.json",
        data_manifest,
    )

    raw_predictions, prediction_count = load_and_validate_raw_predictions(
        raw_predictions_path
    )
    prediction_video_ids = sorted(set(raw_predictions["video-id"]))
    unexpected_prediction_videos = sorted(
        set(prediction_video_ids) - set(evaluated_video_ids)
    )
    if unexpected_prediction_videos:
        raise protocol.ProtocolError(
            "raw predictions contain videos outside the feature-backed official "
            f"test set: {unexpected_prediction_videos[:8]}"
        )
    missing_prediction_videos = sorted(
        set(evaluated_video_ids) - set(prediction_video_ids)
    )
    if missing_prediction_videos:
        raise protocol.ProtocolError(
            "pinned official reproduction produced no raw predictions for "
            f"{len(missing_prediction_videos)} evaluated videos"
        )
    logged_metrics = protocol.parse_actionformer_eval_log(
        eval_log.read_text(encoding="utf-8", errors="strict")
    )
    recomputed_metrics = recompute_official_metrics(
        repo,
        annotation,
        raw_predictions,
    )
    maximum_delta = protocol._assert_metrics_close(
        logged_metrics,
        recomputed_metrics,
        atol=5.1e-5,
        label="official_log_vs_independent_recompute",
        left_is_logged=True,
    )
    evaluator_manifest_sha = protocol.sha256_file(evaluator_manifest_path)
    raw_predictions_sha = protocol.sha256_file(raw_predictions_path)
    eval_log_sha = protocol.sha256_file(eval_log)
    metric_attestation = {
        "schema_version": protocol.METRIC_ATTESTATION_SCHEMA,
        "validation_pass": True,
        "raw_predictions_sha256": raw_predictions_sha,
        "eval_log_sha256": eval_log_sha,
        "evaluator_fingerprint_sha256": evaluator_fingerprint,
        "evaluator_manifest_sha256": evaluator_manifest_sha,
        "annotation_sha256": annotation_sha,
        "prediction_count": prediction_count,
        "prediction_video_count": len(prediction_video_ids),
        "prediction_video_ids_sha256": protocol.canonical_sha256(
            prediction_video_ids
        ),
        "evaluated_video_count": len(evaluated_video_ids),
        "evaluated_video_ids_sha256": evaluated_video_ids_sha,
        "prediction_videos_within_evaluated_set": True,
        "logged_metrics": logged_metrics,
        "recomputed_metrics": recomputed_metrics,
        "max_abs_delta": maximum_delta,
    }
    metric_attestation_path = _write_json(
        output_dir / "OFFICIAL_METRIC_ATTESTATION.json",
        metric_attestation,
    )
    metric_attestation_sha = protocol.sha256_file(metric_attestation_path)

    run_manifest = {
        "schema_version": protocol.RUN_MANIFEST_SCHEMA,
        "source_commit": commit,
        "source_tree": tree,
        "config_sha256": protocol.sha256_file(
            repo / "configs" / "thumos_i3d.yaml"
        ),
        "effective_config_sha256": effective_config_sha,
        "train_log_effective_config_sha256": train_log_config_sha,
        "data_manifest_sha256": protocol.sha256_file(data_manifest_path),
        "checkpoint_sha256": protocol.sha256_file(checkpoint),
        "raw_predictions_sha256": raw_predictions_sha,
        "train_log_sha256": protocol.sha256_file(train_log),
        "eval_log_sha256": eval_log_sha,
        "evaluator_manifest_sha256": evaluator_manifest_sha,
        "metric_attestation_sha256": metric_attestation_sha,
        "environment_manifest_sha256": protocol.sha256_file(
            environment_manifest_path
        ),
        "training_command": protocol.OFFICIAL_ACTIONFORMER_EXPECTED[
            "training.command"
        ],
        "evaluation_command": protocol.OFFICIAL_ACTIONFORMER_EXPECTED[
            "evaluation.command"
        ],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    run_manifest_path = _write_json(
        output_dir / "OFFICIAL_RUN_MANIFEST.json",
        run_manifest,
    )

    paths = {
        "config": repo / "configs" / "thumos_i3d.yaml",
        "readme": repo / "README.md",
        "data_archive": archive,
        "data_manifest": data_manifest_path,
        "annotation": annotation,
        "class_map": class_map_path,
        "observation_manifest": feature_manifest_path,
        "checkpoint": checkpoint,
        "raw_predictions": raw_predictions_path,
        "train_log": train_log,
        "eval_log": eval_log,
        "evaluator_manifest": evaluator_manifest_path,
        "environment_manifest": environment_manifest_path,
        "metric_attestation": metric_attestation_path,
        "run_manifest": run_manifest_path,
    }
    receipts = {
        name: _receipt(path, include_md5=(name == "data_archive"))
        for name, path in paths.items()
    }
    record = {
        "schema_version": protocol.RECORD_SCHEMA,
        "record_id": args.record_id,
        "evidence_stratum": "official_reproduction",
        "protocol_family": "official_actionformer_i3d_stride4_v1",
        "source": {
            "repository_url": protocol.OFFICIAL_REPOSITORY_URL,
            "commit": commit,
            "tree": tree,
            "config_path": "configs/thumos_i3d.yaml",
            "config_sha256": receipts["config"]["sha256"],
            "readme_sha256": receipts["readme"]["sha256"],
            "implementation": "official_actionformer_release",
        },
        "dataset": {
            "name": "THUMOS14",
            "train_split": "validation",
            "eval_split": "test",
            "annotation_sha256": annotation_sha,
            "class_map_sha256": class_map_sha,
            "data_manifest_sha256": receipts["data_manifest"]["sha256"],
            "data_archive_md5": archive_md5,
            "data_archive_sha256": archive_sha,
            "num_classes": 20,
            "nominal_split_counts": protocol.OFFICIAL_NOMINAL_SPLIT_COUNTS,
            "annotation_split_counts": split_counts,
            "annotation_database_video_count": len(videos),
            "evaluated_video_count": len(evaluated_video_ids),
            "evaluated_video_ids_sha256": evaluated_video_ids_sha,
            "blocked_videos": [],
            "feature_only_unannotated_videos": feature_manifest[
                "feature_only_unannotated_videos"
            ],
        },
        "input": {
            "feature_family": "two_stream_i3d_kinetics",
            "feature_provenance_sha256": feature_manifest_sha,
            "feature_inventory_video_count": feature_manifest[
                "feature_inventory_video_count"
            ],
            "annotation_feature_backed_video_count": feature_manifest[
                "annotation_feature_backed_video_count"
            ],
            "evaluated_feature_backed_video_count": feature_manifest[
                "evaluated_feature_backed_video_count"
            ],
            "missing_annotated_feature_videos": feature_manifest[
                "missing_annotated_feature_videos"
            ],
            "input_dim": 2048,
            "clip_frames": 16,
            "frame_stride": 4,
            "seconds_per_feature": 4.0 / 30.0,
            "max_seq_len": 2304,
            "observation_budget": None,
            "observation_manifest_sha256": feature_manifest_sha,
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
            "seed": effective_config["init_rand_seed"],
            "amp": False,
            "ema": True,
            "checkpoint_rule": args.checkpoint_rule,
            "command": protocol.OFFICIAL_ACTIONFORMER_EXPECTED[
                "training.command"
            ],
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
            "evaluator_manifest_sha256": evaluator_manifest_sha,
            "command": protocol.OFFICIAL_ACTIONFORMER_EXPECTED[
                "evaluation.command"
            ],
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
            "artifact_path": str(raw_predictions_path),
            "artifact_sha256": raw_predictions_sha,
            "raw_predictions_sha256": raw_predictions_sha,
            "checkpoint_sha256": receipts["checkpoint"]["sha256"],
            "eval_log_sha256": eval_log_sha,
            "metric_attestation_sha256": metric_attestation_sha,
            "run_manifest_sha256": receipts["run_manifest"]["sha256"],
            "metrics_source": (
                "official_actionformer_eval_log_and_independent_recompute"
            ),
            "prediction_count": prediction_count,
            "prediction_video_count": len(prediction_video_ids),
            "prediction_video_ids_sha256": protocol.canonical_sha256(
                prediction_video_ids
            ),
            "metrics": recomputed_metrics,
        },
        "receipts": receipts,
    }
    record_path = _write_json(output_dir / "OFFICIAL_PROTOCOL_RECORD.json", record)
    verdict = protocol.classify(record)
    if not verdict["main_table_eligible"]:
        raise protocol.ProtocolError(
            "official record failed main-table classification: "
            f"{verdict['ineligibility_reasons']}"
        )
    verdict_path = _write_json(
        output_dir / "OFFICIAL_COMPARABILITY_VERDICT.json",
        verdict,
    )
    completion = {
        "schema_version": "actionformer_official_record_completion_v1",
        "validation_pass": True,
        "main_table_eligible": True,
        "record_path": str(record_path),
        "record_sha256": protocol.sha256_file(record_path),
        "verdict_path": str(verdict_path),
        "verdict_sha256": protocol.sha256_file(verdict_path),
        "record_canonical_sha256": protocol.canonical_sha256(record),
        "metrics": recomputed_metrics,
        "prediction_count": prediction_count,
    }
    _write_json(output_dir / "OFFICIAL_RECORD_COMPLETE.json", completion)
    return completion


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a pinned official ActionFormer THUMOS14 record."
    )
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--data-archive", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw-predictions", required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--eval-log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--checkpoint-rule", default="ema_epoch_034")
    return parser.parse_args()


def main():
    completion = build_record(parse_args())
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
