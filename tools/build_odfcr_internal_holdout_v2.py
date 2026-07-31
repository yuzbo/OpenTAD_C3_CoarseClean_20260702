#!/usr/bin/env python
"""Freeze the ODF-CR validation-only holdout without reusing the old holdout."""

import argparse
import hashlib
import json
import os


SCHEMA_VERSION = "actionformer_odfcr_internal_holdout_v2"
PREVIOUS_SCHEMA_VERSION = "actionformer_dcsr_internal_holdout_v1"
EXPECTED_VALIDATION_COUNT = 200
EXPECTED_PREVIOUS_TRAIN_COUNT = 160
EXPECTED_PREVIOUS_HOLDOUT_COUNT = 40
EXPECTED_HOLDOUT_COUNT = 40
EXPECTED_CLASS_COUNT = 20


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fid:
        for chunk in iter(lambda: fid.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(seed, video_id):
    return hashlib.sha256(
        "{:d}|{:s}".format(seed, video_id).encode("utf-8")
    ).hexdigest()


def _labels(record):
    return frozenset(
        int(annotation["label_id"])
        for annotation in record.get("annotations", [])
    )


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fid:
        return json.load(fid)


def _validate_previous_manifest(previous, previous_sha256, annotation_sha256):
    if previous.get("schema_version") != PREVIOUS_SCHEMA_VERSION:
        raise ValueError("previous manifest schema is not DCSR holdout v1")
    if (
        previous.get("source_split") != "validation"
        or previous.get("test_annotations_used") is not False
        or previous.get("test_records_selected") is not False
        or previous.get("source_annotation_sha256") != annotation_sha256
    ):
        raise ValueError("previous manifest source contract mismatch")
    previous_train = previous.get("train_video_ids")
    previous_holdout = previous.get("holdout_video_ids")
    if (
        not isinstance(previous_train, list)
        or not isinstance(previous_holdout, list)
        or len(previous_train) != EXPECTED_PREVIOUS_TRAIN_COUNT
        or len(previous_holdout) != EXPECTED_PREVIOUS_HOLDOUT_COUNT
        or previous.get("train_video_count")
        != EXPECTED_PREVIOUS_TRAIN_COUNT
        or previous.get("holdout_video_count")
        != EXPECTED_PREVIOUS_HOLDOUT_COUNT
    ):
        raise ValueError("previous manifest must contain an exact 160/40 split")
    if (
        len(previous_train) != len(set(previous_train))
        or len(previous_holdout) != len(set(previous_holdout))
        or set(previous_train) & set(previous_holdout)
    ):
        raise ValueError("previous manifest IDs are duplicate or overlapping")
    if len(previous_sha256) != 64:
        raise ValueError("previous manifest SHA-256 is malformed")
    return frozenset(previous_train), frozenset(previous_holdout)


def _validation_records(annotation):
    database = annotation.get("database")
    if not isinstance(database, dict):
        raise ValueError("annotation database must be a dictionary")
    validation = {}
    for video_id, record in database.items():
        if not isinstance(video_id, str) or not isinstance(record, dict):
            raise ValueError("annotation database record is malformed")
        subset = record.get("subset")
        if not isinstance(subset, str):
            raise ValueError("annotation record is missing a string subset")
        if subset.lower() == "validation":
            validation[video_id] = record
    if len(validation) != EXPECTED_VALIDATION_COUNT:
        raise ValueError("ODF-CR requires exactly 200 validation videos")
    return validation


def validate_manifest_contract(
    manifest,
    previous,
    previous_sha256,
    annotation,
    annotation_sha256,
):
    """Validate every set-membership and class-coverage claim in holdout-v2."""
    previous_train, previous_holdout = _validate_previous_manifest(
        previous, previous_sha256, annotation_sha256
    )
    validation = _validation_records(annotation)
    validation_ids = frozenset(validation)
    if previous_train | previous_holdout != validation_ids:
        raise ValueError(
            "previous manifest IDs do not equal the official validation set"
        )

    train_ids = manifest.get("train_video_ids")
    holdout_ids = manifest.get("holdout_video_ids")
    candidate_ids = manifest.get("candidate_pool_video_ids")
    if (
        not isinstance(train_ids, list)
        or not isinstance(holdout_ids, list)
        or not isinstance(candidate_ids, list)
        or len(train_ids) != 160
        or len(holdout_ids) != EXPECTED_HOLDOUT_COUNT
        or len(candidate_ids) != EXPECTED_PREVIOUS_TRAIN_COUNT
        or len(train_ids) != len(set(train_ids))
        or len(holdout_ids) != len(set(holdout_ids))
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise ValueError("holdout-v2 ID lists must be exact and duplicate-free")
    train = frozenset(train_ids)
    holdout = frozenset(holdout_ids)
    candidate = frozenset(candidate_ids)
    all_classes = frozenset(
        label for record in validation.values() for label in _labels(record)
    )
    train_classes = frozenset(
        label for video_id in train for label in _labels(validation[video_id])
    )
    holdout_classes = frozenset(
        label for video_id in holdout for label in _labels(validation[video_id])
    )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("source_split") != "validation"
        or manifest.get("source_annotation_sha256") != annotation_sha256
        or manifest.get("previous_manifest_sha256") != previous_sha256
        or manifest.get("previous_manifest_schema_version")
        != PREVIOUS_SCHEMA_VERSION
        or manifest.get("previous_train_video_ids")
        != sorted(previous_train)
        or manifest.get("previous_holdout_video_ids")
        != sorted(previous_holdout)
        or manifest.get("train_video_count") != 160
        or manifest.get("holdout_video_count") != EXPECTED_HOLDOUT_COUNT
        or manifest.get("candidate_pool_is_previous_train_only") is not True
        or manifest.get("new_holdout_disjoint_previous_holdout") is not True
        or manifest.get("disjoint") is not True
        or manifest.get("train_all_class_coverage") is not True
        or manifest.get("holdout_all_class_coverage") is not True
        or manifest.get("test_annotations_used") is not False
        or manifest.get("test_records_selected") is not False
        or manifest.get("predictions_read") is not False
        or manifest.get("metrics_read") is not False
        or manifest.get("checkpoint_read") is not False
        or manifest.get("paper_performance_row_allowed") is not False
        or manifest.get("official_test_authorized") is not False
        or candidate != previous_train
        or not holdout <= previous_train
        or holdout & previous_holdout
        or train & holdout
        or train | holdout != validation_ids
        or train != validation_ids - holdout
        or len(all_classes) != EXPECTED_CLASS_COUNT
        or manifest.get("all_class_ids") != sorted(all_classes)
        or train_classes != all_classes
        or holdout_classes != all_classes
    ):
        raise ValueError("invalid ODF-CR holdout-v2 contract")
    return train, holdout, validation


def _choose_holdout(validation, candidate_ids, seed):
    all_classes = frozenset(
        label for record in validation.values() for label in _labels(record)
    )
    if len(all_classes) != EXPECTED_CLASS_COUNT:
        raise ValueError("ODF-CR requires all 20 THUMOS classes")
    candidate_classes = frozenset(
        label for video_id in candidate_ids for label in _labels(
            validation[video_id]
        )
    )
    if candidate_classes != all_classes:
        raise ValueError("previous train-160 cannot cover holdout-v2 classes")

    ordered_ids = sorted(
        candidate_ids,
        key=lambda video_id: (_rank(seed, video_id), video_id),
    )
    train_class_remaining = {
        label: sum(
            label in _labels(record) for record in validation.values()
        )
        for label in all_classes
    }
    holdout = []
    uncovered = set(all_classes)
    while uncovered:
        candidates = []
        for video_id in ordered_ids:
            if video_id in holdout:
                continue
            labels = _labels(validation[video_id])
            newly_covered = labels & uncovered
            if not newly_covered:
                continue
            if any(train_class_remaining[label] <= 1 for label in labels):
                continue
            candidates.append(
                (-len(newly_covered), _rank(seed, video_id), video_id)
            )
        if not candidates:
            raise ValueError("holdout-v2 cannot cover every THUMOS class")
        _, _, chosen = min(candidates)
        holdout.append(chosen)
        for label in _labels(validation[chosen]):
            train_class_remaining[label] -= 1
        uncovered -= _labels(validation[chosen])

    for video_id in ordered_ids:
        if len(holdout) >= EXPECTED_HOLDOUT_COUNT:
            break
        if video_id in holdout:
            continue
        labels = _labels(validation[video_id])
        if any(train_class_remaining[label] <= 1 for label in labels):
            continue
        holdout.append(video_id)
        for label in labels:
            train_class_remaining[label] -= 1
    if len(holdout) != EXPECTED_HOLDOUT_COUNT:
        raise ValueError("failed to fill the exact 40-video holdout-v2")
    return frozenset(holdout), all_classes


def build_manifest(
    previous_manifest_path,
    annotation_path,
    feature_folder,
    feature_extension,
    seed,
):
    annotation_sha256 = _sha256_file(annotation_path)
    previous_sha256 = _sha256_file(previous_manifest_path)
    previous = _read_json(previous_manifest_path)
    previous_train, previous_holdout = _validate_previous_manifest(
        previous, previous_sha256, annotation_sha256
    )
    annotation = _read_json(annotation_path)
    validation = _validation_records(annotation)
    for video_id in validation:
        feature_path = os.path.join(
            feature_folder, video_id + feature_extension
        )
        if not os.path.isfile(feature_path):
            raise ValueError(
                "official validation feature is missing: {:s}".format(
                    feature_path
                )
            )
    if previous_train | previous_holdout != frozenset(validation):
        raise ValueError(
            "previous manifest IDs do not equal the official validation set"
        )

    holdout, all_classes = _choose_holdout(
        validation, previous_train, seed
    )
    train = frozenset(validation) - holdout
    if holdout & previous_holdout:
        raise RuntimeError("holdout-v2 overlaps the previously observed holdout")
    if len(train) != 160 or len(holdout) != 40 or train & holdout:
        raise RuntimeError("holdout-v2 160/40 disjointness contract failed")
    train_classes = frozenset(
        label for video_id in train for label in _labels(validation[video_id])
    )
    holdout_classes = frozenset(
        label
        for video_id in holdout
        for label in _labels(validation[video_id])
    )
    if train_classes != all_classes or holdout_classes != all_classes:
        raise RuntimeError("holdout-v2 class coverage contract failed")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_annotation_path": os.path.realpath(annotation_path),
        "source_annotation_sha256": annotation_sha256,
        "feature_folder": os.path.realpath(feature_folder),
        "feature_extension": feature_extension,
        "source_split": "validation",
        "test_annotations_used": False,
        "test_records_selected": False,
        "predictions_read": False,
        "metrics_read": False,
        "checkpoint_read": False,
        "paper_performance_row_allowed": False,
        "official_test_authorized": False,
        "selection_algorithm": (
            "old_train160_sha256_rank_class_coverage_then_fill_v2"
        ),
        "seed": seed,
        "previous_manifest_path": os.path.realpath(previous_manifest_path),
        "previous_manifest_sha256": previous_sha256,
        "previous_manifest_schema_version": PREVIOUS_SCHEMA_VERSION,
        "previous_train_video_ids": sorted(previous_train),
        "previous_holdout_video_ids": sorted(previous_holdout),
        "candidate_pool_video_ids": sorted(previous_train),
        "candidate_pool_is_previous_train_only": True,
        "new_holdout_disjoint_previous_holdout": True,
        "all_class_ids": sorted(all_classes),
        "train_video_ids": sorted(train),
        "holdout_video_ids": sorted(holdout),
        "train_video_count": len(train),
        "holdout_video_count": len(holdout),
        "disjoint": True,
        "train_all_class_coverage": True,
        "holdout_all_class_coverage": True,
    }
    validated_train, validated_holdout, _ = validate_manifest_contract(
        manifest,
        previous,
        previous_sha256,
        annotation,
        annotation_sha256,
    )
    if validated_train != train or validated_holdout != holdout:
        raise RuntimeError("holdout-v2 self-validation changed selected IDs")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-manifest", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--feature-folder", required=True)
    parser.add_argument("--feature-extension", default=".npy")
    parser.add_argument("--seed", type=int, default=2026073100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError("refusing to overwrite holdout-v2 manifest")
    manifest = build_manifest(
        args.previous_manifest,
        args.annotation,
        args.feature_folder,
        args.feature_extension,
        args.seed,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary_path = args.output + ".tmp"
    with open(temporary_path, "x", encoding="utf-8") as fid:
        json.dump(manifest, fid, indent=2, sort_keys=True)
        fid.write("\n")
    os.replace(temporary_path, args.output)
    print(
        json.dumps(
            {
                "output": os.path.realpath(args.output),
                "sha256": _sha256_file(args.output),
                "previous_manifest_sha256": (
                    manifest["previous_manifest_sha256"]
                ),
                "train_video_count": manifest["train_video_count"],
                "holdout_video_count": manifest["holdout_video_count"],
                "validation_only": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
