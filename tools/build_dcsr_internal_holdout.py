#!/usr/bin/env python
"""Freeze a deterministic, class-covered THUMOS validation holdout.

This tool only reads official ``validation`` annotations and feature-file
presence.  It never reads the THUMOS ``test`` split, predictions, checkpoints,
or metrics.
"""

import argparse
import hashlib
import json
import os


SCHEMA_VERSION = "actionformer_dcsr_internal_holdout_v1"


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


def build_manifest(
    annotation_path,
    feature_folder,
    feature_extension,
    seed,
    holdout_numerator,
    holdout_denominator,
):
    with open(annotation_path, "r", encoding="utf-8") as fid:
        annotation = json.load(fid)
    database = annotation["database"]
    validation = {}
    for video_id, record in database.items():
        if record["subset"].lower() != "validation":
            continue
        feature_path = os.path.join(
            feature_folder, video_id + feature_extension
        )
        if not os.path.isfile(feature_path):
            raise ValueError(
                "official validation feature is missing: {:s}".format(
                    feature_path
                )
            )
        validation[video_id] = record
    if not validation:
        raise ValueError("official validation split is empty")

    all_classes = frozenset(
        label
        for record in validation.values()
        for label in _labels(record)
    )
    class_counts = {
        label: sum(
            label in _labels(record) for record in validation.values()
        )
        for label in all_classes
    }
    if any(count < 2 for count in class_counts.values()):
        raise ValueError(
            "cannot create disjoint class-covered train/holdout subsets"
        )

    ordered_ids = sorted(
        validation,
        key=lambda video_id: (_rank(seed, video_id), video_id),
    )
    target_holdout = max(
        1,
        len(ordered_ids) * holdout_numerator // holdout_denominator,
    )
    if target_holdout >= len(ordered_ids):
        raise ValueError("holdout would consume the full validation split")

    holdout = []
    train_class_remaining = dict(class_counts)
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
                (
                    -len(newly_covered),
                    _rank(seed, video_id),
                    video_id,
                )
            )
        if not candidates:
            raise ValueError(
                "deterministic holdout cannot cover every class"
            )
        _, _, chosen = min(candidates)
        holdout.append(chosen)
        for label in _labels(validation[chosen]):
            train_class_remaining[label] -= 1
        uncovered -= _labels(validation[chosen])

    for video_id in ordered_ids:
        if len(holdout) >= target_holdout:
            break
        if video_id in holdout:
            continue
        labels = _labels(validation[video_id])
        if any(train_class_remaining[label] <= 1 for label in labels):
            continue
        holdout.append(video_id)
        for label in labels:
            train_class_remaining[label] -= 1
    if len(holdout) != target_holdout:
        raise ValueError("failed to fill deterministic holdout budget")

    holdout_set = frozenset(holdout)
    train_ids = sorted(set(ordered_ids) - holdout_set)
    holdout_ids = sorted(holdout_set)
    train_classes = frozenset(
        label for video_id in train_ids for label in _labels(validation[video_id])
    )
    holdout_classes = frozenset(
        label
        for video_id in holdout_ids
        for label in _labels(validation[video_id])
    )
    if train_classes != all_classes or holdout_classes != all_classes:
        raise RuntimeError("class coverage contract failed")
    if set(train_ids) & set(holdout_ids):
        raise RuntimeError("internal train/holdout overlap")

    return {
        "schema_version": SCHEMA_VERSION,
        "source_annotation_path": os.path.realpath(annotation_path),
        "source_annotation_sha256": _sha256_file(annotation_path),
        "feature_folder": os.path.realpath(feature_folder),
        "feature_extension": feature_extension,
        "source_split": "validation",
        "test_annotations_used": False,
        "test_records_selected": False,
        "predictions_read": False,
        "metrics_read": False,
        "checkpoint_read": False,
        "selection_algorithm": (
            "sha256_rank_class_coverage_then_fill_v1"
        ),
        "seed": seed,
        "holdout_fraction": {
            "numerator": holdout_numerator,
            "denominator": holdout_denominator,
        },
        "all_class_ids": sorted(all_classes),
        "train_video_ids": train_ids,
        "holdout_video_ids": holdout_ids,
        "train_video_count": len(train_ids),
        "holdout_video_count": len(holdout_ids),
        "disjoint": True,
        "train_all_class_coverage": True,
        "holdout_all_class_coverage": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--feature-folder", required=True)
    parser.add_argument("--feature-extension", default=".npy")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--holdout-numerator", type=int, default=1)
    parser.add_argument("--holdout-denominator", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        args.holdout_numerator <= 0
        or args.holdout_denominator <= 1
        or args.holdout_numerator >= args.holdout_denominator
    ):
        raise ValueError("holdout fraction must lie strictly between 0 and 1")
    if os.path.exists(args.output):
        raise FileExistsError("refusing to overwrite holdout manifest")

    manifest = build_manifest(
        args.annotation,
        args.feature_folder,
        args.feature_extension,
        args.seed,
        args.holdout_numerator,
        args.holdout_denominator,
    )
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
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
                "train_video_count": manifest["train_video_count"],
                "holdout_video_count": manifest["holdout_video_count"],
                "validation_only": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
