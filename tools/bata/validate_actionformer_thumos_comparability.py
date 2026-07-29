#!/usr/bin/env python3
"""Fail-closed protocol classification for ActionFormer/THUMOS14 results.

Main-table eligibility is intentionally unavailable without live artifact
verification.  A matched result must be anchored to a verified official
reproduction and may differ only through one named intervention bundle.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import pickle
import re
from pathlib import Path

import numpy as np

try:
    from tools.bata import actionformer_source_diff as source_diff
except ModuleNotFoundError:
    import actionformer_source_diff as source_diff


RECORD_SCHEMA = "actionformer_thumos_protocol_record_v4"
LEGACY_OFFICIAL_RECORD_SCHEMA = "actionformer_thumos_protocol_record_v3"
OUTPUT_SCHEMA = "actionformer_thumos_comparability_v4"
METRIC_ATTESTATION_SCHEMA = "actionformer_official_metric_attestation_v1"
EVALUATOR_MANIFEST_SCHEMA = "actionformer_official_evaluator_manifest_v1"
ENVIRONMENT_MANIFEST_SCHEMA = "actionformer_environment_manifest_v1"
RUN_MANIFEST_SCHEMA = "actionformer_official_run_manifest_v2"
TRAIN_LOG_NORMALIZATION_SCHEMA = (
    "actionformer_train_log_effective_config_normalization_v1"
)
DATA_MANIFEST_SCHEMA = "actionformer_official_data_manifest_v2"
OBSERVATION_MANIFEST_SCHEMA = "actionformer_official_feature_manifest_v2"
EXACT_METRIC_MEAN_ATOL = 1.0e-10
# The pinned evaluator prints every threshold and the aggregate independently
# to two decimal percentage points.  In normalized units each displayed value
# therefore has up to 5e-5 quantization error; comparing the displayed
# aggregate with the mean of five independently displayed thresholds can
# accumulate up to 1e-4.  Keep a small floating-point margin, but use this only
# for values parsed from the official text log.
LOGGED_METRIC_MEAN_ATOL = 1.01e-4

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
# These are SHA-256 values of the canonical LF bytes stored in the pinned Git
# tree.  Do not regenerate them from a Windows CRLF checkout.
OFFICIAL_CONFIG_SHA256 = (
    "c0ac0df560cd564941b56cd9391ad0bd5cea386d2e4b6cf9fc8ffcab821955cd"
)
OFFICIAL_README_SHA256 = (
    "f0431584b4df0702fa08f961fb0038e1277f41c12b7df47b7d2bfed47e59af23"
)
OFFICIAL_EFFECTIVE_CONFIG_SHA256 = (
    "835cf30fbcfd27bd6af8885fff002813c8596e2948fce3adf29e3716f316dde4"
)
OFFICIAL_TRAIN_LOG_RAW_EFFECTIVE_CONFIG_SHA256 = (
    "ad426e1a25be48423e21f854bbc6d815c6063388811350ad5fada5ac8933d3a7"
)
OFFICIAL_TRAINING_SEED = 1234567891
OFFICIAL_THUMOS_ARCHIVE_MD5 = "375f76ffbf7447af1035e694971ec9b2"
OFFICIAL_EVALUATOR_FILES = {
    "eval.py": "adf7babd04f78ca8ef232a1ceb23323df25887220f2677913e04a5372d34b158",
    "libs/core/config.py": (
        "014f1000ac09eb1687d2e6b59bdf9f0afa1dc0a2daed909ee988808929723bc8"
    ),
    "libs/datasets/thumos14.py": (
        "ed4bf6767311d00dbf05c5ca381f206d4db40a8efeefe53cf3baba6926a4286b"
    ),
    "libs/utils/metrics.py": (
        "e73afc6afe960aafdb7d607a01eeef07afb5a15a2c9a7a0d1217546adf889480"
    ),
    "libs/utils/nms.py": (
        "b1346a66cb2e5374afcdac9c4adb4889eb1b8ea2343942c107483dadb3450068"
    ),
    "libs/utils/train_utils.py": (
        "60d1cdf0953786b071c0d6a6eecb68a87f546553a7fa7191c5ea79c0992138c6"
    ),
}
OFFICIAL_EVALUATOR_FINGERPRINT_SHA256 = (
    "1d18fbb07a774422a1594946dcf2c59a741c5de3a55d42fa029636ffc43c30b6"
)
OFFICIAL_NOMINAL_SPLIT_COUNTS = {"test": 213, "validation": 200}
OFFICIAL_ANNOTATION_SPLIT_COUNTS = {"test": 212, "validation": 200}
OFFICIAL_ANNOTATION_DATABASE_VIDEO_COUNT = 412
OFFICIAL_EVALUATED_VIDEO_COUNT = 212
OFFICIAL_FEATURE_INVENTORY_VIDEO_COUNT = 413
OFFICIAL_FEATURE_ONLY_UNANNOTATED_VIDEOS = ("video_test_0001292",)
OFFICIAL_THUMOS_CLASS_NAMES = (
    "BaseballPitch",
    "BasketballDunk",
    "Billiards",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "Diving",
    "FrisbeeCatch",
    "GolfSwing",
    "HammerThrow",
    "HighJump",
    "JavelinThrow",
    "LongJump",
    "PoleVault",
    "Shotput",
    "SoccerPenalty",
    "TennisSwing",
    "ThrowDiscus",
    "VolleyballSpiking",
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
SOURCE_DIFF_RECEIPT_NAME = "source_diff_attestation"

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
    "dataset.nominal_split_counts",
    "dataset.annotation_split_counts",
    "dataset.annotation_database_video_count",
    "dataset.evaluated_video_count",
    "dataset.evaluated_video_ids_sha256",
    "dataset.blocked_videos",
    "dataset.feature_only_unannotated_videos",
    "input.feature_family",
    "input.feature_provenance_sha256",
    "input.feature_inventory_video_count",
    "input.annotation_feature_backed_video_count",
    "input.evaluated_feature_backed_video_count",
    "input.missing_annotated_feature_videos",
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
    "result.prediction_video_count",
    "result.prediction_video_ids_sha256",
    "result.metrics",
)
SHA_PATHS = {path for path in REQUIRED_PATHS if path.endswith("_sha256")}

# These are result identities, not protected protocol inputs.  Each record is
# still independently verified against its receipts and metric attestation.
PAIR_OUTPUT_EXEMPT_PREFIXES = (
    "schema_version",
    "record_id",
    "evidence_stratum",
    "result.artifact_path",
    "result.artifact_sha256",
    "result.raw_predictions_sha256",
    "result.checkpoint_sha256",
    "result.eval_log_sha256",
    "result.metric_attestation_sha256",
    "result.run_manifest_sha256",
    "result.source_diff_attestation_sha256",
    "result.prediction_count",
    "result.prediction_video_count",
    "result.prediction_video_ids_sha256",
    "result.metrics",
    "receipts.checkpoint",
    "receipts.raw_predictions",
    "receipts.train_log",
    "receipts.eval_log",
    "receipts.metric_attestation",
    "receipts.run_manifest",
    "receipts.source_diff_attestation",
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
        "source.config_path",
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
        "source.config_path",
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
        "source.config_path",
        "source.config_sha256",
        "source.implementation",
        "receipts.config",
        "model.query_geometry",
        "model.effective_config_sha256",
    },
}
INTERVENTION_ALLOWED_PATHS["sparsehead_method"] = (
    INTERVENTION_ALLOWED_PATHS["selection_budget"]
    | INTERVENTION_ALLOWED_PATHS["head_projection"]
    | INTERVENTION_ALLOWED_PATHS["coordinate_geometry"]
)

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
    "dataset.nominal_split_counts": OFFICIAL_NOMINAL_SPLIT_COUNTS,
    "dataset.annotation_split_counts": OFFICIAL_ANNOTATION_SPLIT_COUNTS,
    "dataset.annotation_database_video_count": OFFICIAL_ANNOTATION_DATABASE_VIDEO_COUNT,
    "dataset.evaluated_video_count": OFFICIAL_EVALUATED_VIDEO_COUNT,
    "dataset.blocked_videos": [],
    "dataset.feature_only_unannotated_videos": list(
        OFFICIAL_FEATURE_ONLY_UNANNOTATED_VIDEOS
    ),
    "dataset.data_archive_md5": OFFICIAL_THUMOS_ARCHIVE_MD5,
    "input.feature_family": "two_stream_i3d_kinetics",
    "input.feature_inventory_video_count": OFFICIAL_FEATURE_INVENTORY_VIDEO_COUNT,
    "input.annotation_feature_backed_video_count": (
        OFFICIAL_ANNOTATION_DATABASE_VIDEO_COUNT
    ),
    "input.evaluated_feature_backed_video_count": OFFICIAL_EVALUATED_VIDEO_COUNT,
    "input.missing_annotated_feature_videos": [],
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
    "model.effective_config_sha256": OFFICIAL_EFFECTIVE_CONFIG_SHA256,
    "training.optimizer": "AdamW",
    "training.learning_rate": 0.0001,
    "training.weight_decay": 0.05,
    "training.epochs": 30,
    "training.batch_size": 2,
    "training.seed": OFFICIAL_TRAINING_SEED,
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
    "result.prediction_video_count": OFFICIAL_EVALUATED_VIDEO_COUNT,
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


def parse_actionformer_train_log_config(text):
    marker = "\nUsing model EMA"
    if marker not in text:
        raise ProtocolError("train log is missing the effective-config/EMA boundary")
    config_text = text.split(marker, 1)[0].strip()
    try:
        config = ast.literal_eval(config_text)
    except (SyntaxError, ValueError) as error:
        raise ProtocolError("train log effective config is not a Python literal") from error
    if not isinstance(config, dict) or not config:
        raise ProtocolError("train log effective config is empty")
    if "\nUsing model EMA" not in text:
        raise ProtocolError("train log does not attest EMA training")
    return config


def normalize_actionformer_train_log_config(config):
    """Canonicalize the one documented upstream-default omission.

    The released THUMOS train log predates serialization of
    ``model.fpn_start_level``.  The pinned official runtime fills the missing
    key from ``libs/core/config.py`` with the integer value zero.  No other
    missing key, alternate value, type coercion, or ignored field is allowed;
    exact canonical equality with the live source-expanded config is checked
    by the caller.
    """

    if not isinstance(config, dict) or not config:
        raise ProtocolError("train log effective config is empty")
    raw_config = copy.deepcopy(config)
    normalized = copy.deepcopy(config)
    model = normalized.get("model")
    if not isinstance(model, dict):
        raise ProtocolError(
            "train log effective config is missing the model object"
        )
    applied_defaults = []
    if "fpn_start_level" not in model:
        model["fpn_start_level"] = 0
        applied_defaults.append(
            {
                "path": "model.fpn_start_level",
                "value": 0,
                "source": {
                    "repository_url": OFFICIAL_REPOSITORY_URL,
                    "commit": OFFICIAL_COMMIT,
                    "file": "libs/core/config.py",
                },
            }
        )
    else:
        value = model["fpn_start_level"]
        if type(value) is not int or value != 0:
            raise ProtocolError(
                "train log model.fpn_start_level must be the exact integer "
                "official default 0"
            )
    attestation = {
        "schema_version": TRAIN_LOG_NORMALIZATION_SCHEMA,
        "validation_pass": True,
        "policy": "inject_only_missing_official_model_fpn_start_level_zero",
        "raw_effective_config_sha256": canonical_sha256(raw_config),
        "normalized_effective_config_sha256": canonical_sha256(normalized),
        "applied_defaults": applied_defaults,
    }
    return normalized, attestation


def _verify_logged_effective_config(record, config):
    try:
        bindings = {
            "input.input_dim": config["dataset"]["input_dim"],
            "input.clip_frames": config["dataset"]["num_frames"],
            "input.frame_stride": config["dataset"]["feat_stride"],
            "input.max_seq_len": config["dataset"]["max_seq_len"],
            "training.optimizer": config["opt"]["type"],
            "training.learning_rate": config["opt"]["learning_rate"],
            "training.weight_decay": config["opt"]["weight_decay"],
            "training.epochs": config["opt"]["epochs"],
            "training.batch_size": config["loader"]["batch_size"],
            "training.seed": config["init_rand_seed"],
            "post_processing.pre_nms_thresh": config["test_cfg"][
                "pre_nms_thresh"
            ],
            "post_processing.pre_nms_topk": config["test_cfg"]["pre_nms_topk"],
            "post_processing.sigma": config["test_cfg"]["nms_sigma"],
            "post_processing.nms_iou_threshold": config["test_cfg"][
                "iou_threshold"
            ],
            "post_processing.nms_min_score": config["test_cfg"]["min_score"],
            "post_processing.max_seg_num": config["test_cfg"]["max_seg_num"],
            "post_processing.multiclass": config["test_cfg"][
                "multiclass_nms"
            ],
            "post_processing.voting_thresh": config["test_cfg"][
                "voting_thresh"
            ],
        }
    except (KeyError, TypeError) as error:
        raise ProtocolError(
            "train log effective config is missing a protected protocol field"
        ) from error
    split_bindings = {
        "dataset.train_split": config["train_split"],
        "dataset.eval_split": config["val_split"],
    }
    for dotted_path, observed in split_bindings.items():
        if observed != [_get_path(record, dotted_path)]:
            raise ProtocolError(
                f"train log effective-config binding mismatch: {dotted_path}"
            )
    for dotted_path, expected in bindings.items():
        if _get_path(record, dotted_path) != expected:
            raise ProtocolError(
                f"train log effective-config binding mismatch: {dotted_path}"
            )
    nms_method = config["test_cfg"].get("nms_method")
    if record["post_processing"]["use_soft_nms"] != (nms_method == "soft"):
        raise ProtocolError(
            "train log effective-config binding mismatch: "
            "post_processing.use_soft_nms"
        )
    effective_sha = canonical_sha256(config)
    if record["model"]["effective_config_sha256"] != effective_sha:
        raise ProtocolError("effective config SHA-256 differs from the train log")
    return effective_sha


def _validate_metrics(metrics, *, name, mean_atol=EXACT_METRIC_MEAN_ATOL):
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
    if abs(mean - normalized["average_mAP"]) > mean_atol:
        raise ProtocolError(f"{name}.average_mAP is not the five-threshold mean")
    return normalized


def _assert_metrics_close(
    left,
    right,
    *,
    atol,
    label,
    left_is_logged=False,
    right_is_logged=False,
):
    left = _validate_metrics(
        left,
        name=f"{label}.left",
        mean_atol=(
            LOGGED_METRIC_MEAN_ATOL
            if left_is_logged
            else EXACT_METRIC_MEAN_ATOL
        ),
    )
    right = _validate_metrics(
        right,
        name=f"{label}.right",
        mean_atol=(
            LOGGED_METRIC_MEAN_ATOL
            if right_is_logged
            else EXACT_METRIC_MEAN_ATOL
        ),
    )
    maximum = max(abs(left[key] - right[key]) for key in left)
    if maximum > atol:
        raise ProtocolError(f"{label} metric mismatch: max_abs_delta={maximum}")
    return maximum


def _verify_receipts(record):
    receipts = record.get("receipts")
    expected_names = set(RECEIPT_NAMES)
    if record.get("evidence_stratum") == "matched_method_control":
        expected_names.add(SOURCE_DIFF_RECEIPT_NAME)
    if not isinstance(receipts, dict) or set(receipts) != expected_names:
        raise ProtocolError("receipt set is incomplete or contains unknown entries")
    for name in sorted(expected_names):
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
    if record.get("evidence_stratum") == "matched_method_control":
        source_diff_sha = _get_path(
            record,
            "result.source_diff_attestation_sha256",
        )
        if source_diff_sha != receipts[SOURCE_DIFF_RECEIPT_NAME]["sha256"]:
            raise ProtocolError(
                "receipt binding mismatch: result.source_diff_attestation_sha256"
            )
    if (
        Path(record["result"]["artifact_path"]).resolve()
        != Path(receipts["raw_predictions"]["path"]).resolve()
    ):
        raise ProtocolError("result artifact path differs from raw prediction receipt")
    return receipts


def _verify_source_diff_attestation(record, receipts):
    if record["evidence_stratum"] != "matched_method_control":
        return None
    source_diff_sha = _get_path(record, "result.source_diff_attestation_sha256")
    if not isinstance(source_diff_sha, str) or not HEX_SHA256.fullmatch(source_diff_sha):
        raise ProtocolError(
            "invalid SHA-256 field: result.source_diff_attestation_sha256"
        )
    try:
        attestation = source_diff.validate_attestation_live(
            load_json(receipts[SOURCE_DIFF_RECEIPT_NAME]["path"])
        )
    except source_diff.SourceDiffError as error:
        raise ProtocolError(f"source-diff attestation failed: {error}") from error
    candidate = attestation["candidate"]
    candidate_bindings = {
        "repository_url": record["source"]["repository_url"],
        "commit": record["source"]["commit"],
        "tree": record["source"]["tree"],
        "config_path": record["source"]["config_path"],
        "config_blob_sha256": record["source"]["config_sha256"],
        "effective_config_sha256": record["model"][
            "effective_config_sha256"
        ],
    }
    for key, expected in candidate_bindings.items():
        if candidate.get(key) != expected:
            raise ProtocolError(
                f"source-diff candidate binding mismatch: {key}"
            )
    intervention = attestation.get("intervention")
    expected_allowed_paths = source_diff.SOURCE_INTERVENTION_ALLOWED_PATHS.get(
        intervention
    )
    if expected_allowed_paths is None:
        raise ProtocolError("source-diff attestation names an unknown intervention")
    if attestation["policy"].get("allowed_paths") != sorted(expected_allowed_paths):
        raise ProtocolError("source-diff source-path allowlist mismatch")
    expected_effective_paths = source_diff.EFFECTIVE_CONFIG_ALLOWED_PATHS.get(
        intervention
    )
    if expected_effective_paths is None:
        raise ProtocolError(
            "source-diff attestation names an unknown effective-config intervention"
        )
    if attestation["policy"].get("allowed_effective_config_paths") != sorted(
        expected_effective_paths
    ):
        raise ProtocolError("source-diff effective-config allowlist mismatch")
    effective_config = attestation.get("effective_config")
    if not isinstance(effective_config, dict):
        raise ProtocolError("source-diff effective-config proof is missing")
    if effective_config.get("candidate_sha256") != record["model"][
        "effective_config_sha256"
    ]:
        raise ProtocolError(
            "source-diff candidate effective-config SHA-256 mismatch"
        )
    return attestation


def _bind_source_diff_to_reference(attestation, reference, intervention):
    if attestation is None:
        raise ProtocolError("matched control is missing a source-diff attestation")
    if attestation.get("intervention") != intervention:
        raise ProtocolError("source-diff intervention does not match the classifier")
    base = attestation["base"]
    reference_bindings = {
        "repository_url": reference["source"]["repository_url"],
        "commit": reference["source"]["commit"],
        "tree": reference["source"]["tree"],
        "config_path": reference["source"]["config_path"],
        "config_blob_sha256": reference["source"]["config_sha256"],
        "effective_config_sha256": reference["model"][
            "effective_config_sha256"
        ],
    }
    for key, expected in reference_bindings.items():
        if base.get(key) != expected:
            raise ProtocolError(f"source-diff base binding mismatch: {key}")
    return True


def _verify_evaluator_manifest(record, receipts):
    manifest = load_json(receipts["evaluator_manifest"]["path"])
    if manifest.get("schema_version") != EVALUATOR_MANIFEST_SCHEMA:
        raise ProtocolError("unsupported evaluator manifest schema")
    source_root = Path(manifest.get("source_root", "")).resolve()
    if not source_root.is_dir():
        raise ProtocolError("evaluator manifest source root is missing")
    if manifest.get("repository_url") != record["source"]["repository_url"]:
        raise ProtocolError("evaluator manifest repository URL mismatch")
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


def _official_class_map_payload():
    return {
        "schema_version": "actionformer_thumos_class_map_v1",
        "labels": [
            {"label_id": label_id, "label": label}
            for label_id, label in enumerate(OFFICIAL_THUMOS_CLASS_NAMES)
        ],
    }


def _annotation_video_sets(annotation_path, class_map_path):
    class_map = load_json(class_map_path)
    if class_map != _official_class_map_payload():
        raise ProtocolError("THUMOS class map differs from the pinned official mapping")
    payload = load_json(annotation_path)
    database = payload.get("database")
    if not isinstance(database, dict) or not database:
        raise ProtocolError("THUMOS annotation database is empty")
    split_video_ids = {"test": set(), "validation": set()}
    observed_label_ids = set()
    for video_id, entry in database.items():
        if not isinstance(video_id, str) or not video_id:
            raise ProtocolError("THUMOS annotation contains an invalid video ID")
        if not isinstance(entry, dict) or not isinstance(entry.get("subset"), str):
            raise ProtocolError(f"THUMOS annotation subset is invalid: {video_id}")
        subset = entry["subset"].casefold()
        if subset not in split_video_ids:
            raise ProtocolError(
                f"THUMOS annotation contains an unexpected subset: {video_id}"
            )
        split_video_ids[subset].add(video_id)
        annotations = entry.get("annotations", [])
        if not isinstance(annotations, list):
            raise ProtocolError(f"THUMOS annotations are invalid: {video_id}")
        for action in annotations:
            if not isinstance(action, dict):
                raise ProtocolError(f"THUMOS action is invalid: {video_id}")
            label_id = action.get("label_id")
            label = action.get("label")
            if (
                type(label_id) is not int
                or not 0 <= label_id < len(OFFICIAL_THUMOS_CLASS_NAMES)
                or label != OFFICIAL_THUMOS_CLASS_NAMES[label_id]
            ):
                raise ProtocolError(
                    f"THUMOS annotation label mapping mismatch: {video_id}"
                )
            observed_label_ids.add(label_id)
    split_counts = {
        subset: len(video_ids)
        for subset, video_ids in sorted(split_video_ids.items())
    }
    if sum(split_counts.values()) != len(database):
        raise ProtocolError("THUMOS annotation video IDs are not unique")
    if observed_label_ids != set(range(len(OFFICIAL_THUMOS_CLASS_NAMES))):
        raise ProtocolError("THUMOS annotation does not cover all official classes")
    return split_video_ids, split_counts


def _verify_feature_manifest(record, receipts, annotation_sets):
    manifest = load_json(receipts["observation_manifest"]["path"])
    if manifest.get("schema_version") != OBSERVATION_MANIFEST_SCHEMA:
        raise ProtocolError("unsupported feature manifest schema")
    if manifest.get("feature_family") != record["input"]["feature_family"]:
        raise ProtocolError("feature manifest family mismatch")
    feature_root = Path(manifest.get("feature_root", "")).resolve()
    if not feature_root.is_dir():
        raise ProtocolError("feature manifest root is missing")
    entries = manifest.get("features")
    if not isinstance(entries, list) or not entries:
        raise ProtocolError("feature manifest entries are empty")

    annotation_ids = set().union(*annotation_sets.values())
    feature_ids = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ProtocolError("feature manifest entry is not an object")
        video_id = entry.get("video_id")
        relative = entry.get("file")
        if not isinstance(video_id, str) or not video_id:
            raise ProtocolError("feature manifest contains an invalid video ID")
        if video_id in feature_ids:
            raise ProtocolError(f"duplicate feature video ID: {video_id}")
        feature_ids.add(video_id)
        if relative != f"{video_id}.npy":
            raise ProtocolError(f"feature filename does not match video ID: {video_id}")
        path = (feature_root / relative).resolve()
        try:
            path.relative_to(feature_root)
        except ValueError as error:
            raise ProtocolError("feature manifest path escapes feature root") from error
        if not path.is_file():
            raise ProtocolError(f"feature file is missing: {video_id}")
        if entry.get("sha256") != sha256_file(path):
            raise ProtocolError(f"feature SHA-256 mismatch: {video_id}")
        size = entry.get("size_bytes")
        if type(size) is not int or size != path.stat().st_size:
            raise ProtocolError(f"feature size mismatch: {video_id}")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as error:
            raise ProtocolError(
                f"cannot load pinned official feature: {video_id}: {error}"
            ) from error
        if (
            array.ndim != 2
            or int(array.shape[0]) <= 0
            or int(array.shape[1]) != 2048
        ):
            raise ProtocolError(
                f"official feature must be non-empty T x 2048: {video_id}"
            )
        if entry.get("dtype") != str(array.dtype) or entry.get("shape") != [
            int(value) for value in array.shape
        ]:
            raise ProtocolError(f"feature dtype/shape receipt mismatch: {video_id}")
        for start in range(0, int(array.shape[0]), 4096):
            if not np.isfinite(np.asarray(array[start : start + 4096])).all():
                raise ProtocolError(f"official feature contains NaN/Inf: {video_id}")
        expected_subset = next(
            (
                subset
                for subset, video_ids in annotation_sets.items()
                if video_id in video_ids
            ),
            None,
        )
        if entry.get("annotation_subset") != expected_subset:
            raise ProtocolError(f"feature annotation-subset mismatch: {video_id}")

    root_feature_ids = {
        path.stem for path in feature_root.glob("*.npy") if path.is_file()
    }
    if root_feature_ids != feature_ids:
        raise ProtocolError("feature manifest does not cover the full .npy inventory")

    missing_ids = sorted(annotation_ids - feature_ids)
    feature_only_ids = sorted(feature_ids - annotation_ids)
    evaluated_ids = sorted(annotation_sets["test"] & feature_ids)
    annotation_feature_backed_ids = sorted(annotation_ids & feature_ids)
    expected_manifest = {
        "feature_inventory_video_count": len(feature_ids),
        "annotation_feature_backed_video_count": len(
            annotation_feature_backed_ids
        ),
        "evaluated_feature_backed_video_count": len(evaluated_ids),
        "missing_annotated_feature_videos": missing_ids,
        "feature_only_unannotated_videos": feature_only_ids,
        "annotation_video_ids_sha256": canonical_sha256(sorted(annotation_ids)),
        "evaluated_video_ids": evaluated_ids,
        "evaluated_video_ids_sha256": canonical_sha256(evaluated_ids),
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ProtocolError(f"feature manifest set binding mismatch: {key}")

    record_bindings = {
        "feature_inventory_video_count": record["input"][
            "feature_inventory_video_count"
        ],
        "annotation_feature_backed_video_count": record["input"][
            "annotation_feature_backed_video_count"
        ],
        "evaluated_feature_backed_video_count": record["input"][
            "evaluated_feature_backed_video_count"
        ],
        "missing_annotated_feature_videos": record["input"][
            "missing_annotated_feature_videos"
        ],
        "feature_only_unannotated_videos": record["dataset"][
            "feature_only_unannotated_videos"
        ],
        "evaluated_video_ids_sha256": record["dataset"][
            "evaluated_video_ids_sha256"
        ],
    }
    for key, expected in record_bindings.items():
        if manifest.get(key) != expected:
            raise ProtocolError(f"feature manifest record binding mismatch: {key}")
    if len(evaluated_ids) != record["dataset"]["evaluated_video_count"]:
        raise ProtocolError("feature-backed evaluated-video count mismatch")
    return manifest


def _verify_raw_prediction_identity(record, receipts, evaluated_video_ids):
    try:
        with Path(receipts["raw_predictions"]["path"]).open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:
        raise ProtocolError(f"cannot load raw prediction artifact: {error}") from error
    required = {"video-id", "t-start", "t-end", "label", "score"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ProtocolError("raw prediction artifact has an unexpected key set")
    video_ids = list(payload["video-id"])
    arrays = {
        key: np.asarray(payload[key]).reshape(-1)
        for key in ("t-start", "t-end", "label", "score")
    }
    counts = {len(video_ids)} | {int(array.size) for array in arrays.values()}
    if len(counts) != 1:
        raise ProtocolError("raw prediction arrays have unequal lengths")
    if any(not isinstance(video_id, str) or not video_id for video_id in video_ids):
        raise ProtocolError("raw predictions contain invalid video IDs")
    for key in ("t-start", "t-end", "score"):
        if not np.isfinite(np.asarray(arrays[key], dtype=np.float64)).all():
            raise ProtocolError(f"raw predictions contain NaN/Inf: {key}")
    labels = np.asarray(arrays["label"])
    integer_labels = labels.astype(np.int64)
    if not np.array_equal(labels, integer_labels):
        raise ProtocolError("raw prediction labels are not integers")
    if integer_labels.size and (
        int(integer_labels.min()) < 0
        or int(integer_labels.max()) >= len(OFFICIAL_THUMOS_CLASS_NAMES)
    ):
        raise ProtocolError("raw prediction labels are outside the official class range")
    starts = np.asarray(arrays["t-start"], dtype=np.float64)
    ends = np.asarray(arrays["t-end"], dtype=np.float64)
    scores = np.asarray(arrays["score"], dtype=np.float64)
    if np.any(ends <= starts):
        raise ProtocolError("raw predictions contain non-positive segments")
    if np.any(scores < 0.0):
        raise ProtocolError("raw predictions contain negative scores")

    prediction_video_ids = sorted(set(video_ids))
    if not set(prediction_video_ids).issubset(set(evaluated_video_ids)):
        raise ProtocolError("raw predictions contain videos outside the evaluated set")
    if (
        record["evidence_stratum"] == "official_reproduction"
        and prediction_video_ids != sorted(evaluated_video_ids)
    ):
        raise ProtocolError(
            "official reproduction raw predictions do not cover the exact "
            "feature-backed evaluated-video set"
        )
    bindings = {
        "prediction_count": counts.pop(),
        "prediction_video_count": len(prediction_video_ids),
        "prediction_video_ids_sha256": canonical_sha256(prediction_video_ids),
    }
    for key, expected in bindings.items():
        if record["result"].get(key) != expected:
            raise ProtocolError(f"raw prediction identity binding mismatch: {key}")
    return prediction_video_ids


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
        "nominal_split_counts": record["dataset"]["nominal_split_counts"],
        "annotation_split_counts": record["dataset"]["annotation_split_counts"],
        "annotation_database_video_count": record["dataset"][
            "annotation_database_video_count"
        ],
        "evaluated_video_count": record["dataset"]["evaluated_video_count"],
        "evaluated_video_ids_sha256": record["dataset"][
            "evaluated_video_ids_sha256"
        ],
        "blocked_videos": record["dataset"]["blocked_videos"],
        "feature_only_unannotated_videos": record["dataset"][
            "feature_only_unannotated_videos"
        ],
    }
    for key, expected in data_bindings.items():
        if data_manifest.get(key) != expected:
            raise ProtocolError(f"data manifest binding mismatch: {key}")

    annotation_sets, annotation_split_counts = _annotation_video_sets(
        receipts["annotation"]["path"],
        receipts["class_map"]["path"],
    )
    if annotation_split_counts != record["dataset"]["annotation_split_counts"]:
        raise ProtocolError("annotation split-count binding mismatch")
    if sum(annotation_split_counts.values()) != record["dataset"][
        "annotation_database_video_count"
    ]:
        raise ProtocolError("annotation database-count binding mismatch")
    observation_manifest = _verify_feature_manifest(
        record, receipts, annotation_sets
    )
    prediction_video_ids = _verify_raw_prediction_identity(
        record,
        receipts,
        observation_manifest["evaluated_video_ids"],
    )

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

    train_log_text = Path(receipts["train_log"]["path"]).read_text(
        encoding="utf-8",
        errors="strict",
    )
    raw_logged_config = parse_actionformer_train_log_config(train_log_text)
    logged_config, train_log_normalization = (
        normalize_actionformer_train_log_config(raw_logged_config)
    )
    _verify_logged_effective_config(record, logged_config)
    if record["training"]["ema"] is not True:
        raise ProtocolError("ActionFormer paper-matched training must use EMA")

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
        "prediction_video_count": record["result"]["prediction_video_count"],
        "prediction_video_ids_sha256": record["result"][
            "prediction_video_ids_sha256"
        ],
        "evaluated_video_count": record["dataset"]["evaluated_video_count"],
        "evaluated_video_ids_sha256": record["dataset"][
            "evaluated_video_ids_sha256"
        ],
        "prediction_videos_within_evaluated_set": True,
    }
    for key, expected in attestation_bindings.items():
        if attestation.get(key) != expected:
            raise ProtocolError(f"metric attestation binding mismatch: {key}")
    if attestation.get("prediction_video_ids_sha256") != canonical_sha256(
        prediction_video_ids
    ):
        raise ProtocolError("metric attestation prediction-video identity mismatch")
    _assert_metrics_close(
        attestation.get("logged_metrics"),
        logged_metrics,
        atol=1.0e-12,
        label="attestation_vs_eval_log",
        left_is_logged=True,
        right_is_logged=True,
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
        left_is_logged=True,
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
        "effective_config_sha256": record["model"][
            "effective_config_sha256"
        ],
        "train_log_effective_config_sha256": record["model"][
            "effective_config_sha256"
        ],
        "train_log_raw_effective_config_sha256": train_log_normalization[
            "raw_effective_config_sha256"
        ],
        "train_log_normalized_effective_config_sha256": train_log_normalization[
            "normalized_effective_config_sha256"
        ],
        "train_log_normalization_sha256": canonical_sha256(
            train_log_normalization
        ),
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
        "training_command": record["training"]["command"],
        "evaluation_command": record["evaluation"]["command"],
    }
    for key, expected in run_bindings.items():
        if run_manifest.get(key) != expected:
            raise ProtocolError(f"run manifest binding mismatch: {key}")
    if run_manifest.get("train_log_normalization") != train_log_normalization:
        raise ProtocolError("run manifest train-log normalization mismatch")
    if record["evidence_stratum"] == "official_reproduction":
        if (
            train_log_normalization["raw_effective_config_sha256"]
            != OFFICIAL_TRAIN_LOG_RAW_EFFECTIVE_CONFIG_SHA256
        ):
            raise ProtocolError(
                "official released train-log raw effective-config SHA-256 mismatch"
            )
        applied_defaults = train_log_normalization["applied_defaults"]
        if (
            len(applied_defaults) != 1
            or applied_defaults[0].get("path") != "model.fpn_start_level"
            or type(applied_defaults[0].get("value")) is not int
            or applied_defaults[0]["value"] != 0
        ):
            raise ProtocolError(
                "official released train log must attest only the upstream "
                "model.fpn_start_level=0 default"
            )


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
        "dataset.annotation_database_video_count",
        "dataset.evaluated_video_count",
        "input.feature_inventory_video_count",
        "input.annotation_feature_backed_video_count",
        "input.evaluated_feature_backed_video_count",
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
    prediction_video_count = record["result"]["prediction_video_count"]
    if type(prediction_video_count) is not int or prediction_video_count < 0:
        raise ProtocolError(
            "result.prediction_video_count must be a non-negative integer"
        )
    if prediction_video_count > record["dataset"]["evaluated_video_count"]:
        raise ProtocolError(
            "result.prediction_video_count exceeds the evaluated-video count"
        )
    seed = record["training"]["seed"]
    if type(seed) is not int or seed < 0:
        raise ProtocolError("training.seed must be a non-negative integer")
    numeric_ranges = {
        "input.seconds_per_feature": (0.0, math.inf, False),
        "training.learning_rate": (0.0, math.inf, False),
        "training.weight_decay": (0.0, math.inf, True),
        "post_processing.pre_nms_thresh": (0.0, math.inf, True),
        "post_processing.sigma": (0.0, math.inf, False),
        "post_processing.nms_iou_threshold": (0.0, 1.0, True),
        "post_processing.nms_min_score": (0.0, math.inf, True),
        "post_processing.voting_thresh": (0.0, 1.0, True),
    }
    for path, (lower, upper, lower_inclusive) in numeric_ranges.items():
        value = _get_path(record, path)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProtocolError(f"field must be numeric: {path}")
        value = float(value)
        lower_ok = value >= lower if lower_inclusive else value > lower
        if not math.isfinite(value) or not lower_ok or value > upper:
            raise ProtocolError(f"numeric field is outside its valid range: {path}")
    budget = record["input"]["observation_budget"]
    if budget is not None and (type(budget) is not int or budget <= 0):
        raise ProtocolError("input.observation_budget must be null or a positive integer")
    for path in (
        "dataset.blocked_videos",
        "dataset.feature_only_unannotated_videos",
        "input.missing_annotated_feature_videos",
    ):
        values = _get_path(record, path)
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
            or values != sorted(set(values))
        ):
            raise ProtocolError(f"field must be a sorted unique string array: {path}")
    for path in (
        "dataset.nominal_split_counts",
        "dataset.annotation_split_counts",
    ):
        counts = _get_path(record, path)
        if not isinstance(counts, dict) or set(counts) != {"test", "validation"}:
            raise ProtocolError(f"field must contain test/validation counts: {path}")
        if any(type(value) is not int or value <= 0 for value in counts.values()):
            raise ProtocolError(f"field contains a non-positive count: {path}")


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
            "input.missing_annotated_feature_videos",
            "input.observation_budget",
        }:
            raise ProtocolError(f"required field is empty: {path}")
    accepted_schemas = {RECORD_SCHEMA}
    if record["evidence_stratum"] != "matched_method_control":
        accepted_schemas.add(LEGACY_OFFICIAL_RECORD_SCHEMA)
    if record["schema_version"] not in accepted_schemas:
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
    source_diff_attestation = _verify_source_diff_attestation(record, receipts)
    evaluator_manifest = _verify_evaluator_manifest(record, receipts)
    if source_diff_attestation is not None and (
        Path(evaluator_manifest["source_root"]).resolve()
        != Path(source_diff_attestation["repository_root"]).resolve()
    ):
        raise ProtocolError(
            "matched evaluator source root differs from the source-diff repository"
        )
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
    reference_official_mismatches = (
        [] if reference is None else official_expectation_mismatches(reference)
    )
    pair_mismatches = []
    reasons = []
    source_diff_attestation_verified = False

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
        elif reference_official_mismatches:
            reasons.append(
                "matched control reference does not match the official ActionFormer protocol"
            )
        elif intervention is None:
            reasons.append("matched control requires one named intervention")
        else:
            if record["protocol_family"] != reference["protocol_family"]:
                reasons.append("matched records use different protocol families")
            pair_mismatches = compare_records(reference, record, intervention)
            if pair_mismatches:
                reasons.append("protected matched-protocol fields differ")
            source_diff_attestation_verified = _bind_source_diff_to_reference(
                _verify_source_diff_attestation(record, record["receipts"]),
                reference,
                intervention,
            )
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
        "reference_official_protocol_mismatches": reference_official_mismatches,
        "pair_mismatches": pair_mismatches,
        "intervention": intervention,
        "source_diff_attestation_verified": source_diff_attestation_verified,
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
