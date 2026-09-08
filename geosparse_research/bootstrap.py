"""Paired video-cluster bootstrap of fixed-checkpoint official detection mAP.

Inputs are the dataframes produced by the official evaluator's loaders (which
deduplicate annotations and map class names). `video_ids` must also include
background and empty-prediction videos from the evaluation split.
"""
import numpy as np
import pandas as pd

from opentad.evaluations.mAP import compute_average_precision_detection


def resample_video_rows(table, sampled_videos):
    """Duplicate complete video clusters, giving each occurrence its own ID."""
    groups = {key: value for key, value in table.groupby("video-id", sort=False)}
    pieces = []
    for occurrence, video in enumerate(sampled_videos):
        if video in groups:
            item = groups[video].copy()
            item["video-id"] = str(occurrence)
            pieces.append(item)
    return pd.concat(pieces, ignore_index=True) if pieces else table.iloc[:0].copy()


def dataset_map(ground_truth, prediction, class_ids, tiou_thresholds):
    """Use official per-class AP and official class/threshold averaging."""
    present = set(ground_truth["label"])
    missing = [label for label in class_ids if label not in present]
    if missing:
        return dict(defined=False, missing_classes=missing)
    ap = np.stack([
        compute_average_precision_detection(
            ground_truth[ground_truth["label"] == label].reset_index(drop=True),
            prediction[prediction["label"] == label].reset_index(drop=True),
            tiou_thresholds=np.asarray(tiou_thresholds))
        for label in class_ids], axis=1)
    return dict(defined=True, missing_classes=[], ap=ap.tolist(),
                map_by_tiou=ap.mean(axis=1).tolist(), average_map=float(ap.mean()))


def paired_video_map_bootstrap(ground_truth, prediction_left, prediction_right, video_ids,
                               *, tiou_thresholds=(.3, .4, .5, .6, .7), draws=2000, seed=0,
                               confidence=.95):
    """Measure right-minus-left uncertainty for two *fixed* sets of weights.

    Values are fractions, differences are fraction-point differences. This
    percentile interval neither measures training-seed variance nor removes
    test-best checkpoint-selection bias. Draws missing an original GT class
    remain in the ledger as undefined; no redraw or changed class average.
    """
    videos = list(video_ids)
    if not videos or len(videos) != len(set(videos)):
        raise ValueError("video_ids must be a nonempty unique full split")
    if draws < 1 or not 0 < confidence < 1:
        raise ValueError("draws and confidence must be positive and valid")
    required = {"video-id", "t-start", "t-end", "label"}
    for table, prediction in ((ground_truth, False), (prediction_left, True), (prediction_right, True)):
        if not (required | ({"score"} if prediction else set())) <= set(table.columns):
            raise ValueError("use dataframes from the official annotation/prediction loaders")
        if not set(table["video-id"]) <= set(videos):
            raise ValueError("table contains a video outside the declared evaluation split")
    classes = sorted(ground_truth["label"].unique().tolist())
    if not classes:
        raise ValueError("official detection mAP requires ground-truth instances")
    thresholds = np.asarray(tiou_thresholds, dtype=float)
    if thresholds.ndim != 1 or not len(thresholds) or not ((thresholds > 0) & (thresholds <= 1)).all():
        raise ValueError("invalid tIoU thresholds")
    left = dataset_map(ground_truth, prediction_left, classes, thresholds)
    right = dataset_map(ground_truth, prediction_right, classes, thresholds)
    rng = np.random.default_rng(seed)
    ledger, delta_samples = [], []
    for draw in range(draws):
        sampled = [videos[index] for index in rng.integers(0, len(videos), size=len(videos))]
        gt = resample_video_rows(ground_truth, sampled)
        lmap = dataset_map(gt, resample_video_rows(prediction_left, sampled), classes, thresholds)
        row = dict(draw=draw, sampled_videos=sampled, defined=lmap["defined"], missing_classes=lmap["missing_classes"])
        if lmap["defined"]:
            rmap = dataset_map(gt, resample_video_rows(prediction_right, sampled), classes, thresholds)
            delta = np.asarray(rmap["map_by_tiou"]) - lmap["map_by_tiou"]
            row.update(left=lmap["average_map"], right=rmap["average_map"],
                       delta=float(delta.mean()), delta_by_tiou=delta.tolist())
            delta_samples.append(np.r_[delta, delta.mean()])
        ledger.append(row)
    interval = None
    if delta_samples:
        alpha = (1 - confidence) / 2
        bounds = np.quantile(delta_samples, [alpha, 1 - alpha], axis=0)
        interval = dict(by_tiou=bounds[:, :-1].T.tolist(), average_map=bounds[:, -1].tolist())
    return dict(unit="fraction", difference="right_minus_left", seed=seed, draws=draws,
                valid_draws=len(delta_samples), undefined_draws=draws - len(delta_samples),
                interval_method="paired_video_percentile", confidence=confidence,
                interval_condition="all original GT classes represented in draw",
                uncertainty="evaluation-video sampling conditional on fixed weights; not seed variance or selection correction",
                video_count=len(videos), class_ids=classes, tiou_thresholds=thresholds.tolist(),
                left=left, right=right, delta=right["average_map"] - left["average_map"],
                interval=interval, draws_ledger=ledger)
