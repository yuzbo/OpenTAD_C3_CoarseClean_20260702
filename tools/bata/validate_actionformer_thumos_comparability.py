#!/usr/bin/env python3
"""Fail-closed protocol classification for ActionFormer/THUMOS14 results.

Main-table eligibility is intentionally unavailable without live artifact
verification.  A matched result must be anchored to a verified official
reproduction and may differ only through one named intervention bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path


RECORD_SCHEMA = "actionformer_thumos_protocol_record_v2"
OUTPUT_SCHEMA = "actionformer_thumos_comparability_v2"
METRIC_ATTESTATION_SCHEMA = "actionformer_official_metric_attestation_v1"
EVALUATOR_MANIFEST_SCHEMA = "actionformer_official_evaluator_manifest_v1"
ENVIRONMENT_MANIFEST_SCHEMA = "actionformer_environment_manifest_v1"
RUN_MANIFEST_SCHEMA = "actionformer_official_run_manifest_v1"
DATA_MANIFEST_SCHEMA = "actionformer_official_data_manifest_v1"
OBSERVATION_MANIFEST_SCHEMA = "actionformer_official_feature_manifest_v1"

EVIDENCE_STRATA = {
    "official_reproduction",
    "matched_method_control",
    "external_reference_only",
    "diagnostic_only",
}
MAIN_TABLE_STRATA = {"official_reproduction", "matched_method_control"}
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX_SHA1 = re.compile(r"^[0-9a-f]{40}$")
HEX_MD5 = re.compile(r"^[0-9a-f]{32}$")

OFFICIAL_REPOSITORY_URL = "https://github.com/happyharrycn/actionformer_release"
OFFICIAL_COMMIT = "61ea7eb9308a568b0cf45e3804830836e30061de"
OFFICIAL_TREE = "7b06c5261ba244788c942a0d73e304581bc35154"
OFFICIAL_CONFIG_SHA256 = (
    "73f8aeaf7deef93aba57259badd4c454990ec1e0ce6eaa7c3434db44baaeeaf0"
)
OFFICIAL_README_SHA256 = (
    "bdee4eb088a74e190935097742c7dbfaf254eb912f79729dccd73b9b36b33db8"
)
OFFICIAL_THUMOS_ARCHIVE_MD5 = "375f76ffbf7447af1035e694971ec9b2"
OFFICIAL_EVALUATOR_FILES = {
    "eval.py": "525d859ff0ae9dfcee3c91b3fd96227cbd67d0774f4ed062f196a1b888fafcc4",
    "libs/core/config.py": (
        "6bdee9397f033b02494f416defc2adb4499780e6fb48cd80e7c110add6cb0615"
    ),
    "libs/datasets/thumos14.py": (
        "614077a884a51da496c753222879d2408968d13b01598d3dd19f9b368b680c5a"
    ),
    "libs/utils/metrics.py": (
        "b937a20f8ee06d43669eef57d12a01708d68eed6937bfd9074fd764a7551a535"
    ),
    "libs/utils/nms.py": (
        "b3234a72126cf82ace87b1653d85438a425d6c5a3947f7f68f9ad61e7c83ba42"
    ),
    "libs/utils/train_utils.py": (
        "a05ddd1c9493f3190833e0b12fb673688cecebc61beb46c9d1aa643364e61e1e"
    ),
}
OFFICIAL_EVALUATOR_FINGERPRINT_SHA256 = (
    "63ece4fbeeacdcfe21664bc6bcd082c115e779fa1451b9b69e87be8e0be1f644"
)

RECEIPT_NAMES = (
    "config",
    "readme",
    "data_archive",
    "data_manifest",
    "annotation",
    "class_map",
    "observation_manifest",
    "checkpoint",
    "raw_predictions",
    "train_log",
    "eval_log",
    "evaluator_manifest",
    "environment_manifest",
    "metric_attestation",
    "run_manifest",
)

REQUIRED_PATHS = (
    "schema_version",
    "record_id",
    "evidence_stratum",
    "protocol_family",
    "source.repository_url",
    "source.commit",
    "source.tree",
    "source.config_path",
    "source.config_sha256",
    "source.readme_sha256",
    "source.implementation",
    "dataset.name",
    "dataset.train_split",
    "dataset.eval_split",
    "dataset.annotation_sha256",
    "dataset.class_map_sha256",
    "dataset.data_manifest_sha256",
    "dataset.data_archive_md5",
    "dataset.data_archive_sha256",
    "dataset.num_classes",
    "dataset.eval_video_count",
    "dataset.blocked_videos",
    "input.feature_family",
    "input.feature_provenance_sha256",
    "input.input_dim",
    "input.clip_frames",
    "input.frame_stride",
    "input.seconds_per_feature",
    "input.max_seq_len",
    "input.observation_budget",
    "input.observation_manifest_sha256",
    "input.selection_policy",
    "model.detector",
    "model.head",
    "model.projection",
    "model.query_geometry",
    "model.effective_config_sha256",
    "training.optimizer",
    "training.learning_rate",
    "training.weight_decay",
    "training.epochs",
    "training.batch_size",
    "training.seed",
    "training.amp",
    "training.ema",
    "training.checkpoint_rule",
    "training.command",
    "post_processing.pre_nms_thresh",
    "post_processing.pre_nms_topk",
    "post_processing.use_soft_nms",
    "post_processing.sigma",
    "post_processing.nms_iou_threshold",
    "post_processing.nms_min_score",
    "post_processing.max_seg_num",
    "post_processing.multiclass",
    "post_processing.voting_thresh",
    "post_processing.score_fusion",
    "post_processing.round_before_cross_window_nms",
    "post_processing.round_after_cross_window_nms",
    "evaluation.evaluator",
    "evaluation.evaluator_sha256",
    "evaluation.evaluator_manifest_sha256",
    "evaluation.command",
    "evaluation.tiou_thresholds",
    "integrity.sampling_uses_gt",
    "integrity.inference_uses_test_labels",
    "integrity.inference_uses_teacher",
    "integrity.inference_uses_prediction_cache",
    "environment.manifest_sha256",
    "environment.comparability_fingerprint_sha256",
    "result.artifact_path",
    "result.artifact_sha256",
    "result.raw_predictions_sha256",
    "result.checkpoint_sha256",
    "result.eval_log_sha256",
    "result.metric_attestation_sha256",
    "result.run_manifest_sha256",
    "result.metrics_source",
    "result.prediction_count",
    "result.metrics",
)
SHA_PATHS = {path for path in REQUIRED_PATHS if path.endswith("_sha256")}

# These are result identities, not protected protocol inputs.  Each record is
# still independently verified against its receipts and metric attestation.
PAIR_OUTPUT_EXEMPT_PREFIXES = (
    "record_id",
    "evidence_stratum",
    "result.artifact_path",
    "result.artifact_sha256",
    "result.raw_predictions_sha256",
    "result.checkpoint_sha256",
    "result.eval_log_sha256",
    "result.metric_attestation_sha256",
    "result.run_manifest_sha256",
    "result.prediction_count",
    "result.metrics",
    "receipts.checkpoint",
    "receipts.raw_predictions",
    "receipts.train_log",
    "receipts.eval_log",
    "receipts.metric_attestation",
    "receipts.run_manifest",
    "receipts.environment_manifest",
    "receipts.evaluator_manifest",
    "environment.manifest_sha256",
    "evaluation.evaluator_manifest_sha256",
)

# No arbitrary path prefixes are accepted.  A candidate declares exactly one
# intervention, and the validator derives the only legal method-side changes.
INTERVENTION_ALLOWED_PATHS = {
    "selection_budget": {
        "source.repository_url",
        "source.commit",
        "source.tree",
        "source.config_sha256",
        "source.implementation",
        "receipts.config",
        "input.feature_provenance_sha256",
        "input.observation_budget",
        "input.observation_manifest_sha256",
        "input.selection_policy",
        "receipts.observation_manifest",
        "model.effective_config_sha256",
    },
    "head_projection": {
        "source.repository_url",
        "source.commit",
        "source.tree",
        "source.config_sha256",
        "source.implementation",
        "receipts.config",
        "model.head",
        "model.projection",
        "model.effective_config_sha256",
    },
    "coordinate_geometry": {
        "source.repository_url",
        "source.commit",
        "source.tree",
        "source.config_sha256",
        "source.implementation",
        "receipts.config",
        "model.query_geometry",
        "model.effective_config_sha256",
    },
}

OFFICIAL_ACTIONFORMER_EXPECTED = {
    "source.repository_url": OFFICIAL_REPOSITORY_URL,
    "source.commit": OFFICIAL_COMMIT,
    "source.tree": OFFICIAL_TREE,
    "source.config_path": "configs/thumos_i3d.yaml",
    "source.config_sha256": OFFICIAL_CONFIG_SHA256,
    "source.readme_sha256": OFFICIAL_README_SHA256,
    "source.implementation": "official_actionformer_release",
    "dataset.name": "THUMOS14",
    "dataset.train_split": "validation",
    "dataset.eval_split": "test",
    "dataset.num_classes": 20,
    "dataset.eval_video_count": 213,
    "dataset.data_archive_md5": OFFICIAL_THUMOS_ARCHIVE_MD5,
    "input.feature_family": "two_stream_i3d_kinetics",
    "input.input_dim": 2048,
    "input.clip_frames": 16,
    "input.frame_stride": 4,
    "input.max_seq_len": 2304,
    "input.observation_budget": None,
    "input.selection_policy": "dense_all_i3d_features",
    "model.detector": "ActionFormer",
    "model.head": "ActionFormerHead",
    "model.projection": "ActionFormerIdentityFPN",
    "model.query_geometry": "uniform_i3d_feature_grid",
    "training.optimizer": "AdamW",
    "training.learning_rate": 0.0001,
    "training.weight_decay": 0.05,
    "training.epochs": 30,
    "training.batch_size": 2,
    "training.seed": 0,
    "training.amp": False,
    "training.ema": True,
    "training.command": (
        "python ./train.py ./configs/thumos_i3d.yaml --output reproduce"
    ),
    "post_processing.pre_nms_thresh": 0.001,
    "post_processing.pre_nms_topk": 2000,
    "post_processing.use_soft_nms": True,
    "post_processing.sigma": 0.5,
    "post_processing.nms_iou_threshold": 0.1,
    "post_processing.nms_min_score": 0.001,
    "post_processing.max_seg_num": 200,
    "post_processing.multiclass": True,
    "post_processing.voting_thresh": 0.7,
    "post_processing.score_fusion": False,
    "post_processing.round_before_cross_window_nms": False,
    "post_processing.round_after_cross_window_nms": False,
    "evaluation.evaluator": "official_actionformer_ANETdetection",
    "evaluation.evaluator_sha256": OFFICIAL_EVALUATOR_FINGERPRINT_SHA256,
    "evaluation.command": (
        "python ./eval.py ./configs/thumos_i3d.yaml <checkpoint>"
    ),
    "evaluation.tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
    "result.metrics_source": (
        "official_actionformer_eval_log_and_independent_recompute"
    ),
}


class ProtocolError(ValueError):
    """Raised when a protocol record violates the evidence contract."""


def _get_path(payload, dotted_path):
    value = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ProtocolError(f"missing required field: {dotted_path}")
        value = value[part]
    return value


def _flatten(payload, prefix=""):
    flattened = {}
    if isinstance(payload, dict):
        for key in sorted(payload):
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(payload[key], child))
    else:
        flattened[prefix] = payload
    return flattened


def _is_empty(value):
    return value is None or value == "" or value == [] or value == {}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path):
    def reject_constant(value):
        raise ProtocolError(f"JSON contains non-finite numeric token: {value}")

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_actionformer_eval_log(text):
    threshold_pattern = re.compile(
        r"\|tIoU\s*=\s*(0\.[3-7]0):\s*mAP\s*=\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*\(%\)"
    )
    average_pattern = re.compile(
        r"Average mAP:\s*([0-9]+(?:\.[0-9]+)?)\s*\(%\)"
    )
    metrics = {}
    for threshold, percentage in threshold_pattern.findall(text):
        key = f"mAP@{float(threshold):.1f}"
        value = float(percentage) / 100.0
        if key in metrics and not math.isclose(metrics[key], value, abs_tol=1.0e-12):
            raise ProtocolError(f"eval log contains conflicting values for {key}")
        metrics[key] = value
    averages = [float(value) / 100.0 for value in average_pattern.findall(text)]
    if not averages:
        raise ProtocolError("eval log is missing Average mAP")
    if max(averages) - min(averages) > 1.0e-12:
        raise ProtocolError("eval log contains conflicting Average mAP values")
    metrics["average_mAP"] = averages[0]
    required = {"average_mAP"} | {f"mAP@{value:.1f}" for value in (0.3, 0.4, 0.5, 0.6, 0.7)}
    if set(metrics) != required:
        raise ProtocolError("eval log does not contain exactly the THUMOS14 mAP set")
    return metrics


def _validate_metrics(metrics, *, name):
    required = {"average_mAP"} | {
        f"mAP@{value:.1f}" for value in (0.3, 0.4, 0.5, 0.6, 0.7)
    }
    if not isinstance(metrics, dict) or set(metrics) != required:
        raise ProtocolError(f"{name} does not contain exactly the THUMOS14 mAP set")
    normalized = {}
    for key in sorted(required):
        if isinstance(metrics[key], bool):
            raise ProtocolError(f"invalid metric type: {name}.{key}")
        value = float(metrics[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ProtocolError(f"invalid metric value: {name}.{key}")
        normalized[key] = value
    mean = sum(
        normalized[f"mAP@{value:.1f}"] for value in (0.3, 0.4, 0.5, 0.6, 0.7)
    ) / 5.0
    if abs(mean - normalized["average_mAP"]) > 1.0e-10:
        raise ProtocolError(f"{name}.average_mAP is not the five-threshold mean")
    return normalized


def _assert_metrics_close(left, right, *, atol, label):
    left = _validate_metrics(left, name=f"{label}.left")
    right = _validate_metrics(right, name=f"{label}.right")
    maximum = max(abs(left[key] - right[key]) for key in left)
    if maximum > atol:
        raise ProtocolError(f"{label} metric mismatch: max_abs_delta={maximum}")
    return maximum


def _verify_receipts(record):
    receipts = record.get("receipts")
    if not isinstance(receipts, dict) or set(receipts) != set(RECEIPT_NAMES):
        raise ProtocolError("receipt set is incomplete or contains unknown entries")
    for name in RECEIPT_NAMES:
        receipt = receipts[name]
        if not isinstance(receipt, dict):
            raise ProtocolError(f"receipt is not an object: {name}")
        path = Path(receipt.get("path", "")).resolve()
        if not path.is_file():
            raise ProtocolError(f"missing receipt artifact: {name}: {path}")
        observed_sha = sha256_file(path)
        if receipt.get("sha256") != observed_sha:
            raise ProtocolError(f"receipt SHA-256 mismatch: {name}")
        size = receipt.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int):
            raise ProtocolError(f"receipt size is not an integer: {name}")
        if size != path.stat().st_size:
            raise ProtocolError(f"receipt size mismatch: {name}")
    archive_receipt = receipts["data_archive"]
    observed_md5 = md5_file(archive_receipt["path"])
    if archive_receipt.get("md5") != observed_md5:
        raise ProtocolError("dataset archive MD5 receipt mismatch")

    bindings = {
        "source.config_sha256": receipts["config"]["sha256"],
        "source.readme_sha256": receipts["readme"]["sha256"],
        "dataset.data_archive_sha256": receipts["data_archive"]["sha256"],
        "dataset.data_archive_md5": receipts["data_archive"]["md5"],
        "dataset.data_manifest_sha256": receipts["data_manifest"]["sha256"],
        "dataset.annotation_sha256": receipts["annotation"]["sha256"],
        "dataset.class_map_sha256": receipts["class_map"]["sha256"],
        "input.feature_provenance_sha256": receipts[
            "observation_manifest"
        ]["sha256"],
        "input.observation_manifest_sha256": receipts[
            "observation_manifest"
        ]["sha256"],
        "evaluation.evaluator_manifest_sha256": receipts[
            "evaluator_manifest"
        ]["sha256"],
        "environment.manifest_sha256": receipts[
            "environment_manifest"
        ]["sha256"],
        "result.artifact_sha256": receipts["raw_predictions"]["sha256"],
        "result.raw_predictions_sha256": receipts[
            "raw_predictions"
        ]["sha256"],
        "result.checkpoint_sha256": receipts["checkpoint"]["sha256"],
        "result.eval_log_sha256": receipts["eval_log"]["sha256"],
        "result.metric_attestation_sha256": receipts[
            "metric_attestation"
        ]["sha256"],
        "result.run_manifest_sha256": receipts["run_manifest"]["sha256"],
    }
    for dotted_path, expected in bindings.items():
        if _get_path(record, dotted_path) != expected:
            raise ProtocolError(f"receipt binding mismatch: {dotted_path}")
    if (
        Path(record["result"]["artifact_path"]).resolve()
        != Path(receipts["raw_predictions"]["path"]).resolve()
    ):
        raise ProtocolError("result artifact path differs from raw prediction receipt")
    return receipts


def _verify_evaluator_manifest(record, receipts):
    manifest = load_json(receipts["evaluator_manifest"]["path"])
    if manifest.get("schema_version") != EVALUATOR_MANIFEST_SCHEMA:
        raise ProtocolError("unsupported evaluator manifest schema")
    source_root = Path(manifest.get("source_root", "")).resolve()
    if not source_root.is_dir():
        raise ProtocolError("evaluator manifest source root is missing")
    if manifest.get("commit") != record["source"]["commit"]:
        raise ProtocolError("evaluator manifest commit mismatch")
    if manifest.get("tree") != record["source"]["tree"]:
        raise ProtocolError("evaluator manifest tree mismatch")
    if manifest.get("clean") is not True:
        raise ProtocolError("evaluator source snapshot is not clean")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ProtocolError("evaluator manifest file map is empty")
    for relative, expected_sha in sorted(files.items()):
        if not isinstance(relative, str) or relative.startswith(("/", "\\")):
            raise ProtocolError("evaluator manifest contains an absolute file path")
        path = (source_root / relative).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as error:
            raise ProtocolError("evaluator manifest path escapes source root") from error
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ProtocolError(f"evaluator source file mismatch: {relative}")
    fingerprint_payload = {
        "files": files,
    }
    fingerprint = canonical_sha256(fingerprint_payload)
    if record["evaluation"]["evaluator_sha256"] != fingerprint:
        raise ProtocolError("evaluator fingerprint mismatch")
    return manifest


def _verify_manifests_and_attestation(record, receipts):
    data_manifest = load_json(receipts["data_manifest"]["path"])
    if data_manifest.get("schema_version") != DATA_MANIFEST_SCHEMA:
        raise ProtocolError("unsupported data manifest schema")
    data_bindings = {
        "archive_sha256": record["dataset"]["data_archive_sha256"],
        "archive_md5": record["dataset"]["data_archive_md5"],
        "annotation_sha256": record["dataset"]["annotation_sha256"],
        "class_map_sha256": record["dataset"]["class_map_sha256"],
        "feature_manifest_sha256": record["input"][
            "observation_manifest_sha256"
        ],
        "eval_video_count": record["dataset"]["eval_video_count"],
    }
    for key, expected in data_bindings.items():
        if data_manifest.get(key) != expected:
            raise ProtocolError(f"data manifest binding mismatch: {key}")

    observation_manifest = load_json(receipts["observation_manifest"]["path"])
    if observation_manifest.get("schema_version") != OBSERVATION_MANIFEST_SCHEMA:
        raise ProtocolError("unsupported feature manifest schema")
    if observation_manifest.get("missing_videos") != record["dataset"]["blocked_videos"]:
        raise ProtocolError("feature manifest missing-video list mismatch")
    if observation_manifest.get("feature_family") != record["input"]["feature_family"]:
        raise ProtocolError("feature manifest family mismatch")

    environment_manifest = load_json(receipts["environment_manifest"]["path"])
    if environment_manifest.get("schema_version") != ENVIRONMENT_MANIFEST_SCHEMA:
        raise ProtocolError("unsupported environment manifest schema")
    comparability = environment_manifest.get("comparability")
    if not isinstance(comparability, dict) or not comparability:
        raise ProtocolError("environment comparability payload is empty")
    if (
        canonical_sha256(comparability)
        != record["environment"]["comparability_fingerprint_sha256"]
    ):
        raise ProtocolError("environment comparability fingerprint mismatch")

    logged_metrics = parse_actionformer_eval_log(
        Path(receipts["eval_log"]["path"]).read_text(
            encoding="utf-8", errors="strict"
        )
    )
    attestation = load_json(receipts["metric_attestation"]["path"])
    if attestation.get("schema_version") != METRIC_ATTESTATION_SCHEMA:
        raise ProtocolError("unsupported metric attestation schema")
    if attestation.get("validation_pass") is not True:
        raise ProtocolError("metric attestation did not pass")
    attestation_bindings = {
        "raw_predictions_sha256": record["result"]["raw_predictions_sha256"],
        "eval_log_sha256": record["result"]["eval_log_sha256"],
        "evaluator_fingerprint_sha256": record["evaluation"]["evaluator_sha256"],
        "evaluator_manifest_sha256": record["evaluation"][
            "evaluator_manifest_sha256"
        ],
        "annotation_sha256": record["dataset"]["annotation_sha256"],
        "prediction_count": record["result"]["prediction_count"],
    }
    for key, expected in attestation_bindings.items():
        if attestation.get(key) != expected:
            raise ProtocolError(f"metric attestation binding mismatch: {key}")
    _assert_metrics_close(
        attestation.get("logged_metrics"),
        logged_metrics,
        atol=1.0e-12,
        label="attestation_vs_eval_log",
    )
    _assert_metrics_close(
        attestation.get("recomputed_metrics"),
        record["result"]["metrics"],
        atol=1.0e-12,
        label="attestation_vs_record",
    )
    maximum = _assert_metrics_close(
        attestation.get("logged_metrics"),
        attestation.get("recomputed_metrics"),
        atol=5.1e-5,
        label="official_log_vs_independent_recompute",
    )
    if abs(float(attestation.get("max_abs_delta", math.inf)) - maximum) > 1.0e-12:
        raise ProtocolError("metric attestation max_abs_delta mismatch")

    run_manifest = load_json(receipts["run_manifest"]["path"])
    if run_manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise ProtocolError("unsupported run manifest schema")
    run_bindings = {
        "source_commit": record["source"]["commit"],
        "source_tree": record["source"]["tree"],
        "config_sha256": record["source"]["config_sha256"],
        "data_manifest_sha256": record["dataset"]["data_manifest_sha256"],
        "checkpoint_sha256": record["result"]["checkpoint_sha256"],
        "raw_predictions_sha256": record["result"]["raw_predictions_sha256"],
        "train_log_sha256": receipts["train_log"]["sha256"],
        "eval_log_sha256": record["result"]["eval_log_sha256"],
        "evaluator_manifest_sha256": record["evaluation"][
            "evaluator_manifest_sha256"
        ],
        "metric_attestation_sha256": record["result"][
            "metric_attestation_sha256"
        ],
        "environment_manifest_sha256": record["environment"]["manifest_sha256"],
    }
    for key, expected in run_bindings.items():
        if run_manifest.get(key) != expected:
            raise ProtocolError(f"run manifest binding mismatch: {key}")


def _verify_strict_types(record):
    string_paths = (
        "record_id",
        "evidence_stratum",
        "protocol_family",
        "source.repository_url",
        "source.config_path",
        "source.implementation",
        "input.feature_family",
        "input.selection_policy",
        "model.detector",
        "model.head",
        "model.projection",
        "model.query_geometry",
        "training.optimizer",
        "training.checkpoint_rule",
        "training.command",
        "evaluation.evaluator",
        "evaluation.command",
        "result.artifact_path",
        "result.metrics_source",
    )
    for path in string_paths:
        if not isinstance(_get_path(record, path), str):
            raise ProtocolError(f"field must be a string: {path}")
    bool_paths = (
        "training.amp",
        "training.ema",
        "post_processing.use_soft_nms",
        "post_processing.multiclass",
        "post_processing.score_fusion",
        "post_processing.round_before_cross_window_nms",
        "post_processing.round_after_cross_window_nms",
        "integrity.sampling_uses_gt",
        "integrity.inference_uses_test_labels",
        "integrity.inference_uses_teacher",
        "integrity.inference_uses_prediction_cache",
    )
    for path in bool_paths:
        if type(_get_path(record, path)) is not bool:
            raise ProtocolError(f"field must be a boolean: {path}")
    positive_int_paths = (
        "dataset.num_classes",
        "dataset.eval_video_count",
        "input.input_dim",
        "input.clip_frames",
        "input.frame_stride",
        "input.max_seq_len",
        "training.epochs",
        "training.batch_size",
        "post_processing.pre_nms_topk",
        "post_processing.max_seg_num",
    )
    for path in positive_int_paths:
        value = _get_path(record, path)
        if type(value) is not int or value <= 0:
            raise ProtocolError(f"field must be a positive integer: {path}")
    prediction_count = record["result"]["prediction_count"]
    if type(prediction_count) is not int or prediction_count < 0:
        raise ProtocolError("result.prediction_count must be a non-negative integer")
    budget = record["input"]["observation_budget"]
    if budget is not None and (type(budget) is not int or budget <= 0):
        raise ProtocolError("input.observation_budget must be null or a positive integer")
    if not isinstance(record["dataset"]["blocked_videos"], list):
        raise ProtocolError("dataset.blocked_videos must be an array")


def validate_record(record):
    """Validate a record and all live receipt artifacts.

    There is deliberately no schema-only or skip-hash mode.  Such records can
    be discussed as templates, but cannot be classified by this main-table
    gate.
    """

    if not isinstance(record, dict):
        raise ProtocolError("protocol record must be a JSON object")
    for path in REQUIRED_PATHS:
        value = _get_path(record, path)
        if _is_empty(value) and path not in {
            "dataset.blocked_videos",
            "input.observation_budget",
        }:
            raise ProtocolError(f"required field is empty: {path}")
    if record["schema_version"] != RECORD_SCHEMA:
        raise ProtocolError("unsupported protocol record schema")
    if record["evidence_stratum"] not in EVIDENCE_STRATA:
        raise ProtocolError("unknown evidence stratum")
    _verify_strict_types(record)
    for path in SHA_PATHS:
        value = _get_path(record, path)
        if not isinstance(value, str) or not HEX_SHA256.fullmatch(value):
            raise ProtocolError(f"invalid SHA-256 field: {path}")
    if not HEX_SHA1.fullmatch(record["source"]["commit"]):
        raise ProtocolError("source commit is not a full Git object ID")
    if not HEX_SHA1.fullmatch(record["source"]["tree"]):
        raise ProtocolError("source tree is not a full Git object ID")
    if not HEX_MD5.fullmatch(record["dataset"]["data_archive_md5"]):
        raise ProtocolError("dataset archive MD5 is invalid")

    forbidden = {
        key: record["integrity"][key]
        for key in (
            "sampling_uses_gt",
            "inference_uses_test_labels",
            "inference_uses_teacher",
            "inference_uses_prediction_cache",
        )
    }
    active = sorted(key for key, value in forbidden.items() if value is not False)
    if active:
        raise ProtocolError(f"forbidden inference information is active: {active}")
    if record["evaluation"]["tiou_thresholds"] != [0.3, 0.4, 0.5, 0.6, 0.7]:
        raise ProtocolError("THUMOS14 tIoU thresholds must be 0.3:0.1:0.7")
    _validate_metrics(record["result"]["metrics"], name="result.metrics")

    receipts = _verify_receipts(record)
    evaluator_manifest = _verify_evaluator_manifest(record, receipts)
    _verify_manifests_and_attestation(record, receipts)

    if record["evidence_stratum"] in MAIN_TABLE_STRATA:
        if record["dataset"]["data_archive_md5"] != OFFICIAL_THUMOS_ARCHIVE_MD5:
            raise ProtocolError("main-table THUMOS archive is not the official archive")
        if record["source"]["readme_sha256"] != OFFICIAL_README_SHA256:
            raise ProtocolError("main-table source is not based on the pinned README")
        evaluator_files = evaluator_manifest["files"]
        if evaluator_files != OFFICIAL_EVALUATOR_FILES:
            raise ProtocolError("main-table evaluator implementation is not pinned")
        if (
            record["evaluation"]["evaluator_sha256"]
            != OFFICIAL_EVALUATOR_FINGERPRINT_SHA256
        ):
            raise ProtocolError("main-table evaluator fingerprint is not official")
    if record["evidence_stratum"] == "official_reproduction":
        if receipts["config"]["sha256"] != OFFICIAL_CONFIG_SHA256:
            raise ProtocolError("official reproduction config bytes are not pinned")
        if receipts["readme"]["sha256"] != OFFICIAL_README_SHA256:
            raise ProtocolError("official reproduction README bytes are not pinned")
        if receipts["data_archive"].get("md5") != OFFICIAL_THUMOS_ARCHIVE_MD5:
            raise ProtocolError("official reproduction archive bytes are not pinned")
    return record


def official_expectation_mismatches(record):
    mismatches = []
    for path, expected in OFFICIAL_ACTIONFORMER_EXPECTED.items():
        observed = _get_path(record, path)
        if observed != expected:
            mismatches.append(
                {"path": path, "reference": expected, "candidate": observed}
            )
    seconds = float(_get_path(record, "input.seconds_per_feature"))
    expected_seconds = 4.0 / 30.0
    if abs(seconds - expected_seconds) > 1.0e-6:
        mismatches.append(
            {
                "path": "input.seconds_per_feature",
                "reference": expected_seconds,
                "candidate": seconds,
            }
        )
    return mismatches


def _path_matches_prefix(path, prefixes):
    return any(path == prefix or path.startswith(f"{prefix}.") for prefix in prefixes)


def compare_records(reference, candidate, intervention):
    if intervention not in INTERVENTION_ALLOWED_PATHS:
        raise ProtocolError(f"unknown matched intervention: {intervention}")
    allowed = INTERVENTION_ALLOWED_PATHS[intervention]
    reference_flat = _flatten(reference)
    candidate_flat = _flatten(candidate)
    paths = sorted(set(reference_flat) | set(candidate_flat))
    mismatches = []
    for path in paths:
        if path.startswith("receipts.") and path.endswith(".path"):
            continue
        if _path_matches_prefix(path, PAIR_OUTPUT_EXEMPT_PREFIXES):
            continue
        if _path_matches_prefix(path, allowed):
            continue
        reference_value = reference_flat.get(path, "<missing>")
        candidate_value = candidate_flat.get(path, "<missing>")
        if reference_value != candidate_value:
            mismatches.append(
                {
                    "path": path,
                    "reference": reference_value,
                    "candidate": candidate_value,
                }
            )
    return mismatches


def classify(record, *, reference=None, intervention=None):
    # Re-validate here so callers cannot bypass live receipt checks by calling
    # classify() directly on a schema-shaped dictionary.
    record = validate_record(record)
    if reference is not None:
        reference = validate_record(reference)

    stratum = record["evidence_stratum"]
    official_mismatches = official_expectation_mismatches(record)
    pair_mismatches = []
    reasons = []

    if stratum == "official_reproduction":
        if intervention is not None:
            raise ProtocolError("official reproduction cannot declare an intervention")
        if official_mismatches:
            reasons.append("candidate differs from the official ActionFormer protocol")
        if reference is not None:
            pair_mismatches = compare_records(
                reference, record, "coordinate_geometry"
            )
            if pair_mismatches:
                reasons.append("candidate differs from the sealed official reference")
    elif stratum == "matched_method_control":
        if reference is None:
            reasons.append("matched control requires a reference record")
        elif reference["evidence_stratum"] != "official_reproduction":
            reasons.append("matched control must be anchored to an official reproduction")
        elif intervention is None:
            reasons.append("matched control requires one named intervention")
        else:
            if record["protocol_family"] != reference["protocol_family"]:
                reasons.append("matched records use different protocol families")
            pair_mismatches = compare_records(reference, record, intervention)
            if pair_mismatches:
                reasons.append("protected matched-protocol fields differ")
    elif stratum == "external_reference_only":
        reasons.append("external/historical reference cannot define a matched delta")
    elif stratum == "diagnostic_only":
        reasons.append("mechanistic diagnostic cannot be a benchmark row")

    main_table_eligible = stratum in MAIN_TABLE_STRATA and not reasons
    return {
        "schema_version": OUTPUT_SCHEMA,
        "validation_pass": True,
        "artifacts_verified": True,
        "record_id": record["record_id"],
        "evidence_stratum": stratum,
        "protocol_family": record["protocol_family"],
        "main_table_eligible": main_table_eligible,
        "matched_delta_allowed": (
            stratum == "matched_method_control" and main_table_eligible
        ),
        "official_actionformer_protocol_match": not official_mismatches,
        "official_protocol_mismatches": official_mismatches,
        "pair_mismatches": pair_mismatches,
        "intervention": intervention,
        "allowed_method_differences": (
            []
            if intervention is None
            else sorted(INTERVENTION_ALLOWED_PATHS[intervention])
        ),
        "claim_boundary": (
            "paper_main_table"
            if main_table_eligible
            else (
                "external_reference_only"
                if stratum == "external_reference_only"
                else "diagnostic_only"
            )
        ),
        "ineligibility_reasons": reasons,
        "record_canonical_sha256": canonical_sha256(record),
        "reference_record_canonical_sha256": (
            None if reference is None else canonical_sha256(reference)
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify ActionFormer/THUMOS14 result comparability."
    )
    parser.add_argument("--record", required=True)
    parser.add_argument("--reference")
    parser.add_argument(
        "--intervention",
        choices=sorted(INTERVENTION_ALLOWED_PATHS),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-main-table", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    record = load_json(args.record)
    reference = load_json(args.reference) if args.reference else None
    verdict = classify(
        record,
        reference=reference,
        intervention=args.intervention,
    )
    atomic_write_json(args.output, verdict)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    if args.require_main_table and not verdict["main_table_eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
