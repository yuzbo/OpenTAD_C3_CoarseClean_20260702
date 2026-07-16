from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.spatial_zoom_s1_contract import (  # noqa: E402
    S1_BOOTSTRAP_REPLICATES,
    S1_BOOTSTRAP_SEED,
    S1_CHECKPOINT_RULE,
    S1_PROFILE_ORDER_SEED,
    atomic_publish_json,
    S1_RESOLUTIONS,
    S1_TRAINING_SEEDS,
    build_s1_profile_order,
    canonical_sha256,
    sha256_file,
    validate_s1_manifest,
)
from tools.bata.spatial_zoom_s1_profile_recovery import (  # noqa: E402
    load_profile_recovery_certificate,
)

Segment = tuple[float, float]
Prediction = tuple[float, float, float]
S1_FORMAL_REPORT_SCHEMA = "spatial_zoom_s1_formal_result_report_v3"


@dataclass(frozen=True)
class DetectionCorpus:
    gt: Mapping[str, Mapping[str, Sequence[Segment]]]
    predictions: Mapping[str, Mapping[str, Sequence[Prediction]]]
    video_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.video_ids or len(self.video_ids) != len(set(self.video_ids)):
            raise ValueError("DetectionCorpus requires unique, non-empty video_ids")
        video_set = set(self.video_ids)
        if not self.gt:
            raise ValueError("DetectionCorpus requires at least one ground-truth class")
        for label, by_video in self.gt.items():
            if not str(label) or not set(by_video).issubset(video_set):
                raise ValueError("ground truth contains an invalid class or video id")
            if not any(by_video.values()):
                raise ValueError(f"ground-truth class {label!r} has no segments")
            for segments in by_video.values():
                for start, end in segments:
                    if not all(
                        math.isfinite(float(value)) for value in (start, end)
                    ) or float(end) <= float(start):
                        raise ValueError("ground truth contains an invalid segment")
        for label, by_video in self.predictions.items():
            if label not in self.gt:
                raise ValueError("predictions contain a class outside ground truth")
            if not set(by_video).issubset(video_set):
                raise ValueError("predictions contain a video outside the corpus")
            for rows in by_video.values():
                for score, start, end in rows:
                    # OpenTAD keeps zero-length proposals as zero-IoU false positives.
                    if not all(
                        math.isfinite(float(value)) for value in (score, start, end)
                    ) or float(end) < float(start):
                        raise ValueError(
                            "predictions contain a non-finite or invalid row"
                        )

    @classmethod
    def from_files(
        cls,
        *,
        ground_truth_path: str | Path,
        prediction_path: str | Path,
        subset: str,
        video_ids: Sequence[str],
    ) -> "DetectionCorpus":
        selected = tuple(sorted(set(map(str, video_ids))))
        selected_set = set(selected)
        gt_payload = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
        prediction_payload = json.loads(
            Path(prediction_path).read_text(encoding="utf-8")
        )
        database = gt_payload.get("database")
        results = prediction_payload.get("results")
        if not isinstance(database, Mapping) or not isinstance(results, Mapping):
            raise ValueError(
                "S1 expects official THUMOS ground truth and result_detection JSON"
            )
        gt: dict[str, dict[str, list[Segment]]] = {}
        for video_id in selected:
            video = database.get(video_id)
            if not isinstance(video, Mapping) or str(video.get("subset")) != str(
                subset
            ):
                raise ValueError(
                    f"video {video_id!r} is missing from ground-truth subset {subset!r}"
                )
            seen = set()
            for annotation in video.get("annotations", []):
                label = str(annotation.get("label", ""))
                segment = annotation.get("segment", ())
                if label == "Ambiguous" or len(segment) != 2:
                    continue
                start, end = float(segment[0]), float(segment[1])
                key = (label, start, end)
                if end <= start or key in seen:
                    continue
                seen.add(key)
                gt.setdefault(label, {}).setdefault(video_id, []).append((start, end))
        predictions: dict[str, dict[str, list[Prediction]]] = {}
        labels = set(gt)
        unexpected_videos = sorted(set(map(str, results)) - selected_set)
        if unexpected_videos:
            raise ValueError(
                "prediction file contains videos outside the frozen corpus"
            )
        for video_id, rows in results.items():
            video_id = str(video_id)
            if not isinstance(rows, list):
                raise ValueError("prediction rows must be a list")
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError("prediction row must be an object")
                label = str(row.get("label", ""))
                segment = row.get("segment", ())
                if label not in labels:
                    raise ValueError(
                        f"prediction label {label!r} is outside ground truth"
                    )
                if len(segment) != 2:
                    raise ValueError("prediction segment must have two endpoints")
                start, end = float(segment[0]), float(segment[1])
                score = float(row.get("score", 0.0))
                if (
                    not all(math.isfinite(value) for value in (start, end, score))
                    or end < start
                ):
                    raise ValueError(
                        "prediction contains a non-finite or invalid segment"
                    )
                predictions.setdefault(label, {}).setdefault(video_id, []).append(
                    (score, start, end)
                )
        return cls(gt=gt, predictions=predictions, video_ids=selected)


def _segment_iou(segment: Segment, candidates: Sequence[Segment]) -> np.ndarray:
    if not candidates:
        return np.empty((0,), dtype=np.float64)
    candidate_array = np.asarray(candidates, dtype=np.float64)
    tt1 = np.maximum(float(segment[0]), candidate_array[:, 0])
    tt2 = np.minimum(float(segment[1]), candidate_array[:, 1])
    intersection = np.maximum(0.0, tt2 - tt1)
    union = (
        (float(segment[1]) - float(segment[0]))
        + (candidate_array[:, 1] - candidate_array[:, 0])
        - intersection
    )
    return intersection / np.maximum(union, np.finfo(np.float64).eps)


def _interpolated_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    mprec = np.hstack(([0.0], precision, [0.0]))
    mrec = np.hstack(([0.0], recall, [1.0]))
    for index in range(len(mprec) - 2, -1, -1):
        mprec[index] = max(mprec[index], mprec[index + 1])
    changed = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[changed] - mrec[changed - 1]) * mprec[changed]))


def _in_duration_group(duration: float, bounds: tuple[float, float] | None) -> bool:
    if bounds is None:
        return True
    lower, upper = bounds
    return duration > lower and duration <= upper


def _class_ap(
    *,
    gt_by_video: Mapping[str, Sequence[Segment]],
    pred_by_video: Mapping[str, Sequence[Prediction]],
    video_sample: Sequence[str],
    tiou_thresholds: Sequence[float],
    duration_bounds: tuple[float, float] | None,
    video_weights: Mapping[str, float] | None = None,
) -> np.ndarray | None:
    gt_by_cluster: dict[int, list[Segment]] = {}
    predictions: list[tuple[float, int, Segment, float]] = []
    cluster_weights: dict[int, float] = {}
    if video_weights is not None:
        sampled_ids = tuple(map(str, video_sample))
        if len(sampled_ids) != len(set(sampled_ids)):
            raise ValueError(
                "Bayesian video weighting requires one unique cluster per video"
            )
        if set(sampled_ids) != set(map(str, video_weights)):
            raise ValueError(
                "Bayesian video weights must match the complete sampled video set"
            )
    for cluster_id, video_id in enumerate(video_sample):
        weight = 1.0 if video_weights is None else float(video_weights[str(video_id)])
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("Bayesian cluster weights must be finite and positive")
        cluster_weights[cluster_id] = weight
        gt_segments = [
            (float(start), float(end))
            for start, end in gt_by_video.get(str(video_id), ())
            if _in_duration_group(float(end) - float(start), duration_bounds)
        ]
        gt_by_cluster[cluster_id] = gt_segments
        for score, start, end in pred_by_video.get(str(video_id), ()):
            predictions.append(
                (float(score), cluster_id, (float(start), float(end)), weight)
            )
    npos = sum(
        cluster_weights[cluster_id] * len(segments)
        for cluster_id, segments in gt_by_cluster.items()
    )
    if npos == 0:
        return None
    if predictions:
        order = np.asarray([row[0] for row in predictions]).argsort()[::-1]
        predictions = [predictions[int(index)] for index in order]
    thresholds = np.asarray(tuple(tiou_thresholds), dtype=np.float64)
    true_positive = np.zeros((len(thresholds), len(predictions)), dtype=np.float64)
    false_positive = np.zeros_like(true_positive)
    locks = {
        cluster_id: np.zeros((len(thresholds), len(segments)), dtype=np.bool_)
        for cluster_id, segments in gt_by_cluster.items()
    }
    for prediction_index, (_, cluster_id, segment, weight) in enumerate(predictions):
        gt_segments = gt_by_cluster[cluster_id]
        if not gt_segments:
            false_positive[:, prediction_index] = weight
            continue
        overlaps = _segment_iou(segment, gt_segments)
        order = np.argsort(overlaps)[::-1]
        for threshold_index, threshold in enumerate(thresholds):
            matched = False
            for gt_index in order:
                if overlaps[gt_index] < threshold:
                    break
                if not locks[cluster_id][threshold_index, gt_index]:
                    locks[cluster_id][threshold_index, gt_index] = True
                    true_positive[threshold_index, prediction_index] = weight
                    matched = True
                    break
            if not matched:
                false_positive[threshold_index, prediction_index] = weight
    ap = np.zeros(len(thresholds), dtype=np.float64)
    for threshold_index in range(len(thresholds)):
        tp = np.cumsum(true_positive[threshold_index])
        fp = np.cumsum(false_positive[threshold_index])
        recall = tp / float(npos)
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap[threshold_index] = _interpolated_ap(precision, recall)
    return ap


def _map_vector(
    corpus: DetectionCorpus,
    *,
    video_sample: Sequence[str],
    tiou_thresholds: Sequence[float],
    duration_bounds: tuple[float, float] | None = None,
    required_labels: Sequence[str] | None = None,
    video_weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    class_ap = []
    labels = tuple(sorted(corpus.gt) if required_labels is None else required_labels)
    for label in labels:
        result = _class_ap(
            gt_by_video=corpus.gt[label],
            pred_by_video=corpus.predictions.get(label, {}),
            video_sample=video_sample,
            tiou_thresholds=tiou_thresholds,
            duration_bounds=duration_bounds,
            video_weights=video_weights,
        )
        if result is None:
            raise ValueError(
                f"sample has no required duration support for class {label}"
            )
        class_ap.append(result)
    if not class_ap:
        raise ValueError("sample contains no ground-truth class support")
    return np.mean(np.stack(class_ap, axis=0), axis=0) * 100.0


def _duration_supported_labels(
    corpus: DetectionCorpus, bounds: tuple[float, float] | None
) -> tuple[str, ...]:
    labels = []
    for label in sorted(corpus.gt):
        if any(
            _in_duration_group(float(end) - float(start), bounds)
            for segments in corpus.gt[label].values()
            for start, end in segments
        ):
            labels.append(label)
    if not labels:
        raise ValueError("S1 duration group has no ground-truth class support")
    return tuple(labels)


def assert_official_evaluator_parity(
    corpus: DetectionCorpus,
    *,
    tiou_thresholds: Sequence[float],
    duration_bounds: tuple[float, float] | None = None,
) -> None:
    import pandas as pd

    from opentad.evaluations.mAP import compute_average_precision_detection

    thresholds = np.asarray(tuple(map(float, tiou_thresholds)), dtype=np.float64)
    labels = _duration_supported_labels(corpus, duration_bounds)
    for label in labels:
        ground_truth_rows = []
        prediction_rows = []
        for video_id in corpus.video_ids:
            for start, end in corpus.gt[label].get(video_id, ()):
                if _in_duration_group(float(end) - float(start), duration_bounds):
                    ground_truth_rows.append(
                        {
                            "video-id": video_id,
                            "t-start": float(start),
                            "t-end": float(end),
                        }
                    )
            for score, start, end in corpus.predictions.get(label, {}).get(
                video_id, ()
            ):
                prediction_rows.append(
                    {
                        "video-id": video_id,
                        "t-start": float(start),
                        "t-end": float(end),
                        "score": float(score),
                    }
                )
        official = compute_average_precision_detection(
            pd.DataFrame(ground_truth_rows, columns=("video-id", "t-start", "t-end")),
            pd.DataFrame(
                prediction_rows,
                columns=("video-id", "t-start", "t-end", "score"),
            ),
            tiou_thresholds=thresholds,
        )
        local = _class_ap(
            gt_by_video=corpus.gt[label],
            pred_by_video=corpus.predictions.get(label, {}),
            video_sample=corpus.video_ids,
            tiou_thresholds=thresholds,
            duration_bounds=duration_bounds,
        )
        if local is None or not np.allclose(local, official, rtol=0.0, atol=1e-12):
            raise AssertionError(
                f"S1 AP implementation diverges from official evaluator for class {label}"
            )


def _boundary_error(
    corpus: DetectionCorpus, *, video_sample: Sequence[str], match_tiou: float = 0.5
) -> dict[str, float | int | None]:
    start_errors: list[float] = []
    end_errors: list[float] = []
    gt_count = 0
    for label in sorted(corpus.gt):
        for cluster_id, video_id in enumerate(video_sample):
            del cluster_id
            gt_segments = list(corpus.gt[label].get(str(video_id), ()))
            gt_count += len(gt_segments)
            locked = np.zeros(len(gt_segments), dtype=np.bool_)
            predictions = sorted(
                corpus.predictions.get(label, {}).get(str(video_id), ()),
                key=lambda row: -float(row[0]),
            )
            for _, start, end in predictions:
                overlaps = _segment_iou((float(start), float(end)), gt_segments)
                if not len(overlaps):
                    continue
                for gt_index in np.argsort(overlaps)[::-1]:
                    if overlaps[gt_index] < match_tiou:
                        break
                    if not locked[gt_index]:
                        locked[gt_index] = True
                        gt_start, gt_end = gt_segments[gt_index]
                        start_errors.append(abs(float(start) - float(gt_start)))
                        end_errors.append(abs(float(end) - float(gt_end)))
                        break
    matched = len(start_errors)
    return {
        "match_tiou": float(match_tiou),
        "matched_gt": matched,
        "total_gt": gt_count,
        "matched_recall": 0.0 if gt_count == 0 else matched / gt_count,
        "start_mae_seconds": None if not start_errors else float(np.mean(start_errors)),
        "end_mae_seconds": None if not end_errors else float(np.mean(end_errors)),
    }


def evaluate_corpus(
    corpus: DetectionCorpus,
    *,
    video_sample: Sequence[str],
    tiou_thresholds: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
    duration_quartiles: Sequence[float],
) -> dict[str, Any]:
    thresholds = tuple(float(value) for value in tiou_thresholds)
    q1, q2, q3 = (float(value) for value in duration_quartiles)
    if not 0.0 < q1 <= q2 <= q3:
        raise ValueError("duration quartiles must be positive and ordered")
    overall_labels = _duration_supported_labels(corpus, None)
    overall = _map_vector(
        corpus,
        video_sample=video_sample,
        tiou_thresholds=thresholds,
        required_labels=overall_labels,
    )
    duration_bounds = {
        "short": (0.0, q1),
        "medium": (q1, q3),
        "long": (q3, math.inf),
    }
    duration_map = {}
    for name, bounds in duration_bounds.items():
        labels = _duration_supported_labels(corpus, bounds)
        values = _map_vector(
            corpus,
            video_sample=video_sample,
            tiou_thresholds=thresholds,
            duration_bounds=bounds,
            required_labels=labels,
        )
        duration_map[name] = {
            f"{threshold:.1f}": float(value)
            for threshold, value in zip(thresholds, values)
        }
    high_indices = [
        index for index, threshold in enumerate(thresholds) if threshold >= 0.6
    ]
    if not high_indices:
        high_indices = [len(thresholds) - 1]
    return {
        "average_map": float(np.mean(overall)),
        "map_at": {
            f"{threshold:.1f}": float(value)
            for threshold, value in zip(thresholds, overall)
        },
        "high_tiou_headroom": float(np.mean(overall[high_indices])),
        "duration_map": duration_map,
        "duration_protocol": (
            "GT-conditioned frozen fit-duration population; all predictions remain "
            "eligible false positives"
        ),
        "boundary_error": _boundary_error(corpus, video_sample=video_sample),
    }


def _metric_vector(
    metrics: Mapping[str, Any], thresholds: Sequence[float]
) -> dict[str, float]:
    vector = {
        "average_map": float(metrics["average_map"]),
        "high_tiou_headroom": float(metrics["high_tiou_headroom"]),
    }
    short = metrics["duration_map"]["short"].get("0.7")
    vector["short_map_0.7"] = float("nan") if short is None else float(short)
    for threshold in thresholds:
        vector[f"map_{threshold:.1f}"] = float(metrics["map_at"][f"{threshold:.1f}"])
    return vector


def _bootstrap_metric_vector(
    corpus: DetectionCorpus,
    *,
    video_sample: Sequence[str],
    tiou_thresholds: Sequence[float],
    duration_quartiles: Sequence[float],
    video_weights: Mapping[str, float],
) -> dict[str, float]:
    """Recompute only the preregistered bootstrap metrics from raw detections."""
    thresholds = tuple(float(value) for value in tiou_thresholds)
    overall_labels = _duration_supported_labels(corpus, None)
    overall = _map_vector(
        corpus,
        video_sample=video_sample,
        tiou_thresholds=thresholds,
        required_labels=overall_labels,
        video_weights=video_weights,
    )
    high_indices = [
        index for index, threshold in enumerate(thresholds) if threshold >= 0.6
    ]
    if not high_indices:
        high_indices = [len(thresholds) - 1]
    q1 = float(duration_quartiles[0])
    short_bounds = (0.0, q1)
    short_labels = _duration_supported_labels(corpus, short_bounds)
    short_map_0_7 = float(
        _map_vector(
            corpus,
            video_sample=video_sample,
            tiou_thresholds=(0.7,),
            duration_bounds=short_bounds,
            required_labels=short_labels,
            video_weights=video_weights,
        )[0]
    )
    vector = {
        "average_map": float(np.mean(overall)),
        "high_tiou_headroom": float(np.mean(overall[high_indices])),
        "short_map_0.7": short_map_0_7,
    }
    vector.update(
        {
            f"map_{threshold:.1f}": float(value)
            for threshold, value in zip(thresholds, overall)
        }
    )
    return vector


def _paired_bayesian_weights(
    corpus: DetectionCorpus,
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    if int(replicates) <= 1:
        raise ValueError("S1 Bayesian bootstrap requires at least two replicates")
    rng = np.random.default_rng(int(seed))
    weights = rng.exponential(scale=1.0, size=(int(replicates), len(corpus.video_ids)))
    weights /= np.mean(weights, axis=1, keepdims=True)
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise RuntimeError("S1 Bayesian bootstrap generated invalid video weights")
    return weights


def _mean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    boundary_rows = [row["boundary_error"] for row in rows]

    def optional_mean(key: str) -> float | None:
        values = [row[key] for row in boundary_rows if row[key] is not None]
        return None if not values else float(np.mean(values))

    return {
        "average_map": float(np.mean([row["average_map"] for row in rows])),
        "high_tiou_headroom": float(
            np.mean([row["high_tiou_headroom"] for row in rows])
        ),
        "map_at": {
            key: float(np.mean([row["map_at"][key] for row in rows]))
            for key in rows[0]["map_at"]
        },
        "duration_map": {
            group: {
                key: (
                    None
                    if not [
                        row["duration_map"][group][key]
                        for row in rows
                        if row["duration_map"][group][key] is not None
                    ]
                    else float(
                        np.mean(
                            [
                                row["duration_map"][group][key]
                                for row in rows
                                if row["duration_map"][group][key] is not None
                            ]
                        )
                    )
                )
                for key in rows[0]["duration_map"][group]
            }
            for group in ("short", "medium", "long")
        },
        "boundary_error": {
            "match_tiou": float(boundary_rows[0]["match_tiou"]),
            "matched_gt_mean": float(
                np.mean([row["matched_gt"] for row in boundary_rows])
            ),
            "total_gt_mean": float(np.mean([row["total_gt"] for row in boundary_rows])),
            "matched_recall_mean": float(
                np.mean([row["matched_recall"] for row in boundary_rows])
            ),
            "start_mae_seconds_mean": optional_mean("start_mae_seconds"),
            "end_mae_seconds_mean": optional_mean("end_mae_seconds"),
        },
    }


def _simultaneous_max_t_lower_bounds(
    observed: np.ndarray,
    bootstrap: np.ndarray,
    *,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return one-sided simultaneous lower bounds for correlated estimates."""

    observed = np.asarray(observed, dtype=np.float64)
    bootstrap = np.asarray(bootstrap, dtype=np.float64)
    if (
        observed.ndim != 1
        or bootstrap.ndim != 2
        or bootstrap.shape[1] != observed.shape[0]
    ):
        raise ValueError("max-T inputs must be [candidate] and [replicate,candidate]")
    if (
        not 0.0 < float(confidence) < 1.0
        or not np.isfinite(observed).all()
        or not np.isfinite(bootstrap).all()
    ):
        raise ValueError("max-T inputs and confidence must be finite and valid")
    standard_error = np.std(bootstrap, axis=0, ddof=1)
    positive = standard_error > np.finfo(np.float64).eps
    if np.any(positive):
        pivots = np.zeros_like(bootstrap)
        # A lower confidence bound uses the upper tail of (theta* - theta_hat) / SE.
        pivots[:, positive] = (
            bootstrap[:, positive] - observed[positive]
        ) / standard_error[positive]
        critical = max(
            0.0, float(np.quantile(np.max(pivots, axis=1), float(confidence)))
        )
    else:
        critical = 0.0
    return observed - critical * standard_error, standard_error, critical


def _assert_global_profile_matrix_comparability(
    profiles: Mapping[tuple[int, int], Mapping[str, Any]],
) -> None:
    """Require every profile in the 3x3 matrix to share one cost protocol."""

    globally_fixed = (
        "protocol",
        "protocol_fingerprint",
        "manifest_sha256",
        "hardware_identity",
        "hardware_fingerprint",
        "software_identity",
        "software_fingerprint",
        "config_commit",
        "experiment_namespace",
        "canonical_experiment_root",
        "pretrained_checkpoint_sha256",
        "batch_size",
        "loader_workers",
        "warmup_samples",
        "amp",
        "power_sampling_enabled",
        "formal_profile",
        "split",
        "sample_count",
        "sample_manifest_sha256",
        "physical_window_manifest_sha256",
        "loader_exposure_count",
        "physical_window_count",
        "duplicate_physical_window_exposure_count",
        "max_physical_window_multiplicity",
        "test_open_certificate_sha256",
        "precheck_file_sha256",
        "precheck_sha256",
        "power_gpu_id",
        "power_interval_ms",
        "world_size",
        "execution_wrapper",
        "result_finalizer",
        "profile_order_seed",
        "profile_order_sha256",
        "profile_code_commit",
        "profile_recovery_certificate_path",
        "profile_recovery_certificate_file_sha256",
        "profile_recovery_certificate_sha256",
        "profile_recovery_campaign_id",
    )
    for key in globally_fixed:
        values = {
            json.dumps(profile.get(key), sort_keys=True)
            for profile in profiles.values()
        }
        if len(values) != 1:
            raise ValueError(f"S1 profile matrix spans incompatible {key}")
    ordinals = {
        int(profile.get("profile_order_ordinal", -1)) for profile in profiles.values()
    }
    if ordinals != set(range(9)):
        raise ValueError("S1 profile matrix does not cover the frozen order ordinals")


def aggregate_s1_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    duration_quartiles: Sequence[float],
    tiou_thresholds: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 3407001,
    require_three_seeds: bool = True,
    expected_class_count: int | None = None,
) -> dict[str, Any]:
    thresholds = tuple(float(value) for value in tiou_thresholds)
    by_key: dict[tuple[int, int], DetectionCorpus] = {}
    profiles: dict[tuple[int, int], Mapping[str, Any]] = {}
    for run in runs:
        resolution = int(run["resolution"])
        seed = int(run["seed"])
        corpus = run.get("corpus")
        if not isinstance(corpus, DetectionCorpus):
            raise ValueError(
                "aggregate_s1_runs requires resolved DetectionCorpus objects"
            )
        key = (resolution, seed)
        if key in by_key:
            raise ValueError(f"duplicate S1 run {key}")
        by_key[key] = corpus
        profile = run.get("profile")
        if profile is not None:
            if not isinstance(profile, Mapping):
                raise ValueError("S1 run profile must be a resolved mapping")
            profiles[key] = profile
        if expected_class_count is not None and len(corpus.gt) != int(
            expected_class_count
        ):
            raise ValueError(
                f"formal S1 evaluation requires exactly {expected_class_count} classes"
            )
    seeds = tuple(sorted({seed for _, seed in by_key}))
    expected_seeds = S1_TRAINING_SEEDS if require_three_seeds else seeds
    if seeds != tuple(expected_seeds):
        raise ValueError(f"S1 seed matrix must be {tuple(expected_seeds)}")
    expected_keys = {
        (resolution, seed) for resolution in S1_RESOLUTIONS for seed in seeds
    }
    if set(by_key) != expected_keys:
        raise ValueError(
            "S1 result matrix must contain dense160/224/256 for every seed"
        )
    if expected_class_count is not None and set(profiles) != expected_keys:
        raise ValueError("formal S1 result gate requires a profile for every run")
    for seed in seeds:
        baseline = by_key[(160, seed)]
        for resolution in (224, 256):
            candidate = by_key[(resolution, seed)]
            if candidate.video_ids != baseline.video_ids or candidate.gt != baseline.gt:
                raise ValueError(
                    "S1 paired runs must share identical videos and ground truth"
                )

    observed: dict[tuple[int, int], dict[str, Any]] = {}
    for key, corpus in by_key.items():
        if expected_class_count is not None:
            assert_official_evaluator_parity(
                corpus,
                tiou_thresholds=thresholds,
            )
            assert_official_evaluator_parity(
                corpus,
                tiou_thresholds=(0.7,),
                duration_bounds=(0.0, float(duration_quartiles[0])),
            )
        observed[key] = evaluate_corpus(
            corpus,
            video_sample=corpus.video_ids,
            tiou_thresholds=thresholds,
            duration_quartiles=duration_quartiles,
        )
    metric_names = (
        "average_map",
        "high_tiou_headroom",
        "short_map_0.7",
        *(f"map_{threshold:.1f}" for threshold in thresholds),
    )
    boot_deltas: dict[int, dict[int, dict[str, list[float]]]] = {
        resolution: {seed: {name: [] for name in metric_names} for seed in seeds}
        for resolution in (224, 256)
    }
    for seed_index, seed in enumerate(seeds):
        baseline_corpus = by_key[(160, seed)]
        weight_matrix = _paired_bayesian_weights(
            baseline_corpus,
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed) + seed_index,
        )
        for weights in weight_matrix:
            video_weights = {
                video_id: float(weight)
                for video_id, weight in zip(baseline_corpus.video_ids, weights)
            }
            baseline_vector = _bootstrap_metric_vector(
                baseline_corpus,
                video_sample=baseline_corpus.video_ids,
                tiou_thresholds=thresholds,
                duration_quartiles=duration_quartiles,
                video_weights=video_weights,
            )
            for resolution in (224, 256):
                candidate_vector = _bootstrap_metric_vector(
                    by_key[(resolution, seed)],
                    video_sample=baseline_corpus.video_ids,
                    tiou_thresholds=thresholds,
                    duration_quartiles=duration_quartiles,
                    video_weights=video_weights,
                )
                for name in metric_names:
                    boot_deltas[resolution][seed][name].append(
                        candidate_vector[name] - baseline_vector[name]
                    )

    hierarchy_rng = np.random.default_rng(int(bootstrap_seed) + 10_000)
    seed_draws = hierarchy_rng.integers(
        0, len(seeds), size=(int(bootstrap_replicates), len(seeds))
    )
    video_draws = hierarchy_rng.integers(
        0,
        int(bootstrap_replicates),
        size=(int(bootstrap_replicates), len(seeds)),
    )
    pooled_boot: dict[int, dict[str, np.ndarray]] = {}
    for resolution in (224, 256):
        pooled_boot[resolution] = {}
        for name in metric_names:
            stacked = np.stack(
                [
                    np.asarray(boot_deltas[resolution][seed][name], dtype=np.float64)
                    for seed in seeds
                ],
                axis=0,
            )
            draws = stacked[seed_draws, video_draws]
            if not np.isfinite(draws).all():
                raise ValueError("S1 bootstrap produced non-finite metric values")
            pooled_boot[resolution][name] = np.mean(draws, axis=1)

    observed_delta: dict[int, dict[str, float]] = {}
    per_seed_delta: dict[int, dict[int, dict[str, float]]] = {224: {}, 256: {}}
    for resolution in (224, 256):
        for seed in seeds:
            candidate = _metric_vector(observed[(resolution, seed)], thresholds)
            baseline = _metric_vector(observed[(160, seed)], thresholds)
            per_seed_delta[resolution][seed] = {
                name: candidate[name] - baseline[name] for name in metric_names
            }
        observed_delta[resolution] = {
            name: float(
                np.mean([per_seed_delta[resolution][seed][name] for seed in seeds])
            )
            for name in metric_names
        }

    observed_h = np.asarray(
        [
            observed_delta[224]["high_tiou_headroom"],
            observed_delta[256]["high_tiou_headroom"],
        ],
        dtype=np.float64,
    )
    boot_h = np.stack(
        [
            pooled_boot[224]["high_tiou_headroom"],
            pooled_boot[256]["high_tiou_headroom"],
        ],
        axis=1,
    )
    simultaneous_lower, standard_error, critical = _simultaneous_max_t_lower_bounds(
        observed_h, boot_h
    )

    resolution_report: dict[str, Any] = {}
    for index, resolution in enumerate((224, 256)):
        mean_metrics = _mean_metrics([observed[(resolution, seed)] for seed in seeds])
        delta = observed_delta[resolution]
        ci = {
            name: {
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
            }
            for name, values in pooled_boot[resolution].items()
        }
        high_positive_seeds = sum(
            per_seed_delta[resolution][seed].get("map_0.6", 0.0) > 0.0
            and per_seed_delta[resolution][seed].get("map_0.7", 0.0) > 0.0
            for seed in seeds
        )
        stable_regressions = {
            f"{threshold:.1f}": sum(
                per_seed_delta[resolution][seed][f"map_{threshold:.1f}"] < -0.5
                for seed in seeds
            )
            for threshold in thresholds
        }
        conditions = {
            "delta_high_tiou_at_least_1_point": delta["high_tiou_headroom"] >= 1.0,
            "simultaneous_one_sided_95lcb_positive": float(simultaneous_lower[index])
            > 0.0,
            "delta_average_map_at_least_0_5": delta["average_map"] >= 0.5,
            "delta_short_q1_map_0_7_at_least_0_5": delta["short_map_0.7"] >= 0.5,
            "map_0_6_and_0_7_positive_in_two_of_three_seeds": high_positive_seeds >= 2,
            "no_stable_tiou_regression_over_0_5": all(
                count < 2 for count in stable_regressions.values()
            ),
        }
        conditions["all_conditions"] = all(conditions.values())
        resolution_report[str(resolution)] = {
            "metrics_mean": mean_metrics,
            "metrics_per_seed": {
                str(seed): observed[(resolution, seed)] for seed in seeds
            },
            "delta_vs_dense160": delta,
            "paired_95ci": ci,
            "per_seed_delta": {
                str(seed): per_seed_delta[resolution][seed] for seed in seeds
            },
            "stable_regression_seed_counts": stable_regressions,
            "simultaneous_high_tiou_one_sided_95lcb": float(simultaneous_lower[index]),
            "gate": conditions,
        }
    cost_summary: dict[str, Any] = {}
    if profiles:
        from tools.bata.spatial_zoom_s1_cost import compare_resolution_profiles

        if set(profiles) != expected_keys:
            raise ValueError(
                "S1 cost aggregation requires the complete 3x3 profile matrix"
            )
        _assert_global_profile_matrix_comparability(profiles)
        for seed in seeds:
            for resolution in (224, 256):
                compare_resolution_profiles(
                    profiles[(160, seed)], profiles[(resolution, seed)]
                )
        for resolution in S1_RESOLUTIONS:
            rows = [profiles[(resolution, seed)] for seed in seeds]
            cost_summary[str(resolution)] = {
                "end_to_end_p50_ms": float(
                    np.mean(
                        [row["stages"]["end_to_end_serial_ms"]["p50"] for row in rows]
                    )
                ),
                "end_to_end_p95_ms": float(
                    np.mean(
                        [row["stages"]["end_to_end_serial_ms"]["p95"] for row in rows]
                    )
                ),
                "peak_gpu_allocated_mb": float(
                    np.mean(
                        [
                            row["resources"]["peak_gpu_allocated_mb"]["max"]
                            for row in rows
                        ]
                    )
                ),
                "energy_j_per_window": float(
                    np.mean([row["resources"]["gpu_energy_j"]["mean"] for row in rows])
                ),
            }
    for row in resolution_report.values():
        # Accuracy establishes S1 headroom. Cost only chooses among candidates
        # that have independently passed every preregistered accuracy gate.
        row["gate"]["eligible_for_resolution_freeze"] = bool(
            row["gate"]["all_conditions"]
        )
    route_go = any(
        row["gate"]["eligible_for_resolution_freeze"]
        for row in resolution_report.values()
    )
    eligible_resolutions = [
        int(resolution)
        for resolution, row in resolution_report.items()
        if row["gate"]["eligible_for_resolution_freeze"]
    ]
    selected_resolution = None
    if eligible_resolutions:
        if cost_summary:
            selected_resolution = min(
                eligible_resolutions,
                key=lambda resolution: (
                    cost_summary[str(resolution)]["end_to_end_p50_ms"],
                    -observed_delta[resolution]["high_tiou_headroom"],
                    resolution,
                ),
            )
        else:
            selected_resolution = min(eligible_resolutions)
    return {
        "schema_version": "spatial_zoom_s1_result_gate_v3",
        "status": "GO" if route_go else "KILL",
        "baseline_dense160": _mean_metrics([observed[(160, seed)] for seed in seeds]),
        "baseline_dense160_per_seed": {
            str(seed): observed[(160, seed)] for seed in seeds
        },
        "resolutions": resolution_report,
        "cost_summary": cost_summary or None,
        "resolution_decision": {
            "selected_resolution": selected_resolution,
            "eligible_resolutions": eligible_resolutions,
            "rule": (
                "pass all preregistered accuracy gates independently; then freeze the "
                "passing candidate with the lowest mean end-to-end p50 latency"
            ),
            "formal_cost_used": bool(cost_summary),
        },
        "bootstrap": {
            "unit": "paired_bayesian_video_cluster",
            "paired": True,
            "recomputes_full_class_ap": True,
            "positive_video_weights": True,
            "support_rejection": False,
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "hierarchical_pooling": (
                "resample training seeds with replacement, then draw one paired "
                "Bayesian video-weight replicate within each sampled seed"
            ),
            "inferential_target": (
                "Bayesian bootstrap over the empirical video-cluster distribution "
                "with fixed class support and weighted AP"
            ),
        },
        "simultaneous_max_t": {
            "metric": "high_tiou_headroom",
            "direction": "one_sided_95_percent_lower_confidence_bound",
            "candidates": [224, 256],
            "critical_value": critical,
            "lower_bounds": {
                "224": float(simultaneous_lower[0]),
                "256": float(simultaneous_lower[1]),
            },
        },
        "metric_units": "absolute mAP percentage points",
        "duration_protocol": (
            "quartiles frozen from fit GT; duration groups are GT-conditioned and "
            "retain all predictions as possible false positives"
        ),
        "official_evaluator_parity_checked": expected_class_count is not None,
        "claim_allowed": False,
    }


def seal_s1_result_report(
    core_report: Mapping[str, Any],
    *,
    source_descriptors: Sequence[Mapping[str, Any]],
    global_identity: Mapping[str, Any],
) -> dict[str, Any]:
    sources = sorted(
        (dict(row) for row in source_descriptors),
        key=lambda row: (int(row["resolution"]), int(row["seed"])),
    )
    if (
        len(sources) != 9
        or len({(int(row["resolution"]), int(row["seed"])) for row in sources}) != 9
    ):
        raise ValueError("formal S1 report requires nine unique descriptor identities")
    report = json.loads(json.dumps(dict(core_report)))
    report["formal_report_schema"] = S1_FORMAL_REPORT_SCHEMA
    report["source_descriptors"] = sources
    report["global_identity"] = dict(global_identity)
    report["report_sha256"] = canonical_sha256(report)
    return report


def validate_s1_result_report_envelope(
    report: Mapping[str, Any],
    *,
    expected_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checked = json.loads(json.dumps(dict(report)))
    report_hash = checked.pop("report_sha256", None)
    if not report_hash or canonical_sha256(checked) != report_hash:
        raise ValueError("S1 formal result report self-hash mismatch")
    checked["report_sha256"] = report_hash
    if checked.get("formal_report_schema") != S1_FORMAL_REPORT_SCHEMA:
        raise ValueError("unsupported S1 formal result report schema")
    sources = checked.get("source_descriptors")
    if not isinstance(sources, list) or len(sources) != 9:
        raise ValueError("S1 formal result report has no complete source matrix")
    if expected_report is not None and checked != dict(expected_report):
        raise ValueError("S1 formal result report differs from deterministic rebuild")
    return checked


def _load_run_descriptor(
    path: Path, *, manifest: Mapping[str, Any], ground_truth_path: Path
) -> dict[str, Any]:
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    descriptor_hash = descriptor.pop("descriptor_sha256", None)
    if not descriptor_hash or canonical_sha256(descriptor) != descriptor_hash:
        raise ValueError(f"S1 run descriptor self-hash mismatch: {path}")
    descriptor["descriptor_sha256"] = descriptor_hash
    required = (
        "schema_version",
        "resolution",
        "seed",
        "manifest_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "checkpoint_selection_rule",
        "checkpoint_selection_path",
        "checkpoint_selection_sha256",
        "checkpoint_selection_internal_sha256",
        "test_evidence_path",
        "test_evidence_file_sha256",
        "test_evidence_sha256",
        "test_open_certificate_path",
        "test_open_certificate_file_sha256",
        "test_open_certificate_sha256",
        "test_open_marker_path",
        "test_open_marker_file_sha256",
        "test_open_marker_sha256",
        "prediction_path",
        "prediction_sha256",
        "profile_summary_path",
        "profile_summary_sha256",
        "profile_summary_internal_sha256",
        "profile_samples_path",
        "profile_samples_sha256",
        "profile_power_path",
        "profile_power_sha256",
        "ground_truth_path",
        "ground_truth_sha256",
        "config_path",
        "resolved_config_sha256",
        "code_commit",
        "profile_code_commit",
        "experiment_namespace",
        "canonical_experiment_root",
        "precheck_file_sha256",
        "precheck_sha256",
        "pretrained_checkpoint_sha256",
        "profile_attempt_marker_path",
        "profile_attempt_marker_file_sha256",
        "profile_attempt_marker_sha256",
        "profile_order_seed",
        "profile_order_sha256",
        "profile_order_ordinal",
        "profile_recovery_certificate_path",
        "profile_recovery_certificate_file_sha256",
        "profile_recovery_certificate_sha256",
        "profile_recovery_campaign_id",
    )
    missing = [key for key in required if key not in descriptor]
    if missing:
        raise ValueError(f"S1 run descriptor {path} is missing {missing}")
    if descriptor["schema_version"] != "spatial_zoom_s1_run_v5":
        raise ValueError("unsupported S1 run descriptor schema")
    if descriptor["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("S1 run descriptor uses a different manifest")
    if descriptor["checkpoint_selection_rule"] != S1_CHECKPOINT_RULE:
        raise ValueError("S1 run descriptor changed checkpoint selection")
    artifacts = (
        ("checkpoint_path", "checkpoint_sha256"),
        ("checkpoint_selection_path", "checkpoint_selection_sha256"),
        ("test_evidence_path", "test_evidence_file_sha256"),
        ("test_open_certificate_path", "test_open_certificate_file_sha256"),
        ("test_open_marker_path", "test_open_marker_file_sha256"),
        ("prediction_path", "prediction_sha256"),
        ("profile_summary_path", "profile_summary_sha256"),
        ("profile_samples_path", "profile_samples_sha256"),
        ("profile_power_path", "profile_power_sha256"),
        (
            "profile_attempt_marker_path",
            "profile_attempt_marker_file_sha256",
        ),
        (
            "profile_recovery_certificate_path",
            "profile_recovery_certificate_file_sha256",
        ),
        ("ground_truth_path", "ground_truth_sha256"),
    )
    for path_key, hash_key in artifacts:
        artifact = Path(descriptor[path_key])
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        if sha256_file(artifact) != descriptor[hash_key]:
            raise ValueError(f"S1 descriptor artifact hash mismatch: {artifact}")
    if sha256_file(ground_truth_path) != descriptor["ground_truth_sha256"]:
        raise ValueError("S1 analyzer ground truth differs from the run descriptor")
    cfg = Config.fromfile(descriptor["config_path"])
    if canonical_sha256(cfg.to_dict()) != descriptor["resolved_config_sha256"]:
        raise ValueError("S1 bound config hash mismatch")
    from tools.bata.select_spatial_zoom_s1_checkpoint import (
        validate_checkpoint_selection,
    )
    from tools.bata.spatial_zoom_s1_cost import (
        S1_PROFILE_SCHEMA,
        validate_profile_summary,
    )
    from tools.bata.spatial_zoom_s1_evidence import validate_s1_test_evidence
    from tools.bata.spatial_zoom_s1_training import validate_bound_s1_training_config
    from tools.bata.profile_spatial_zoom_s1 import validate_profile_attempt_marker

    binding = validate_bound_s1_training_config(cfg, seed=int(descriptor["seed"]))
    if not binding["formal_precheck_verified"]:
        raise RuntimeError("formal S1 analysis requires the bound full precheck")
    if descriptor["code_commit"] != binding["code_commit"]:
        raise ValueError("S1 run descriptor Git commit mismatch")
    recovery = load_profile_recovery_certificate(
        descriptor["profile_recovery_certificate_path"],
        binding=binding,
        verify_checkout=True,
    )
    recovery_expected = {
        "profile_code_commit": recovery["profile_code_commit"],
        "profile_recovery_certificate_sha256": recovery["certificate_sha256"],
        "profile_recovery_campaign_id": recovery["campaign_id"],
    }
    for key, value in recovery_expected.items():
        if descriptor.get(key) != value:
            raise ValueError(f"S1 run descriptor {key} mismatch")
    profile_order = build_s1_profile_order()
    profile_order_entry = next(
        row
        for row in profile_order
        if int(row["resolution"]) == int(descriptor["resolution"])
        and int(row["seed"]) == int(descriptor["seed"])
    )
    expected_profile_order = {
        "profile_order_seed": S1_PROFILE_ORDER_SEED,
        "profile_order_sha256": canonical_sha256(profile_order),
        "profile_order_ordinal": int(profile_order_entry["ordinal"]),
    }
    for key, value in expected_profile_order.items():
        if descriptor.get(key) != value:
            raise ValueError(f"S1 run descriptor {key} mismatch")
    for key in (
        "precheck_file_sha256",
        "precheck_sha256",
        "pretrained_checkpoint_sha256",
        "experiment_namespace",
        "canonical_experiment_root",
    ):
        if descriptor[key] != binding[key]:
            raise ValueError(f"S1 run descriptor {key} mismatch")
    selection = validate_checkpoint_selection(
        json.loads(
            Path(descriptor["checkpoint_selection_path"]).read_text(encoding="utf-8")
        ),
        config=cfg,
        seed=int(descriptor["seed"]),
        manifest=manifest,
        checkpoint_path=descriptor["checkpoint_path"],
        protocol_fingerprint=binding["protocol_fingerprint"],
    )
    test_evidence = validate_s1_test_evidence(
        json.loads(Path(descriptor["test_evidence_path"]).read_text(encoding="utf-8")),
        cfg=cfg,
        seed=int(descriptor["seed"]),
    )
    if (
        selection["selection_sha256"]
        != descriptor["checkpoint_selection_internal_sha256"]
    ):
        raise ValueError("S1 selection internal identity mismatch")
    if test_evidence["evidence_sha256"] != descriptor["test_evidence_sha256"]:
        raise ValueError("S1 test evidence identity mismatch")
    if (
        test_evidence["test_open_certificate_sha256"]
        != descriptor["test_open_certificate_sha256"]
    ):
        raise ValueError("S1 test-open certificate identity mismatch")
    if (
        test_evidence["test_open_marker_sha256"]
        != descriptor["test_open_marker_sha256"]
    ):
        raise ValueError("S1 test-open marker identity mismatch")
    profile_attempt_marker = validate_profile_attempt_marker(
        descriptor["profile_attempt_marker_path"]
    )
    if (
        profile_attempt_marker["marker_sha256"]
        != descriptor["profile_attempt_marker_sha256"]
    ):
        raise ValueError("S1 profile-attempt marker identity mismatch")
    profile = validate_profile_summary(
        json.loads(Path(descriptor["profile_summary_path"]).read_text(encoding="utf-8"))
    )
    if profile.get("schema_version") != S1_PROFILE_SCHEMA:
        raise ValueError("S1 descriptor profile schema mismatch")
    if profile["profile_sha256"] != descriptor["profile_summary_internal_sha256"]:
        raise ValueError("S1 profile internal identity mismatch")
    if (
        profile["sample_trace_file_sha256"] != descriptor["profile_samples_sha256"]
        or profile["power_trace_file_sha256"] != descriptor["profile_power_sha256"]
    ):
        raise ValueError("S1 descriptor profile trace identity mismatch")
    profile_expected = {
        "resolution": int(descriptor["resolution"]),
        "seed": int(descriptor["seed"]),
        "split": "test",
        "checkpoint_sha256": descriptor["checkpoint_sha256"],
        "checkpoint_epoch": int(descriptor["checkpoint_epoch"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "test_open_certificate_sha256": descriptor["test_open_certificate_sha256"],
        "test_evidence_sha256": descriptor["test_evidence_sha256"],
        "test_open_marker_sha256": descriptor["test_open_marker_sha256"],
        "config_commit": binding["code_commit"],
        "profile_code_commit": recovery["profile_code_commit"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "profile_attempt_marker_path": str(
            Path(descriptor["profile_attempt_marker_path"]).resolve()
        ),
        "profile_attempt_marker_file_sha256": descriptor[
            "profile_attempt_marker_file_sha256"
        ],
        "profile_attempt_marker_sha256": descriptor["profile_attempt_marker_sha256"],
        "profile_recovery_certificate_path": str(
            Path(descriptor["profile_recovery_certificate_path"]).resolve()
        ),
        "profile_recovery_certificate_file_sha256": descriptor[
            "profile_recovery_certificate_file_sha256"
        ],
        "profile_recovery_certificate_sha256": recovery["certificate_sha256"],
        "profile_recovery_campaign_id": recovery["campaign_id"],
        **expected_profile_order,
        "trained_checkpoint": True,
    }
    for key, expected in profile_expected.items():
        if profile.get(key) != expected:
            raise ValueError(f"S1 profile {key} mismatch in analyzer")
    profile_path = Path(descriptor["profile_summary_path"]).resolve()
    if not profile_path.name.endswith(".summary.json"):
        raise ValueError("formal S1 profile summary must end in .summary.json")
    marker_expected = {
        "resolution": int(descriptor["resolution"]),
        "seed": int(descriptor["seed"]),
        "bound_config_sha256": descriptor["resolved_config_sha256"],
        "code_commit": binding["code_commit"],
        "profile_code_commit": recovery["profile_code_commit"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": descriptor["checkpoint_sha256"],
        "test_open_certificate_sha256": descriptor["test_open_certificate_sha256"],
        "test_evidence_sha256": descriptor["test_evidence_sha256"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "hardware_fingerprint": profile["hardware_fingerprint"],
        "software_fingerprint": profile["software_fingerprint"],
        **expected_profile_order,
        "canonical_output_prefix": str(
            profile_path.with_name(profile_path.name[: -len(".summary.json")])
        ),
        "profile_recovery_certificate_path": str(
            Path(descriptor["profile_recovery_certificate_path"]).resolve()
        ),
        "profile_recovery_certificate_file_sha256": descriptor[
            "profile_recovery_certificate_file_sha256"
        ],
        "profile_recovery_certificate_sha256": recovery["certificate_sha256"],
        "profile_recovery_campaign_id": recovery["campaign_id"],
    }
    for key, expected in marker_expected.items():
        if profile_attempt_marker.get(key) != expected:
            raise ValueError(f"S1 profile-attempt marker {key} mismatch in analyzer")
    prediction = Path(descriptor["prediction_path"])
    descriptor["corpus"] = DetectionCorpus.from_files(
        ground_truth_path=ground_truth_path,
        prediction_path=prediction,
        subset=manifest["annotation_subsets"]["sealed_test"],
        video_ids=manifest["splits"]["test"],
    )
    if len(descriptor["corpus"].gt) != 20:
        raise ValueError("formal S1 THUMOS evaluation requires exactly 20 classes")
    descriptor["profile"] = profile
    return descriptor


def _single_run_value(runs: Sequence[Mapping[str, Any]], key: str) -> Any:
    values = {json.dumps(run.get(key), sort_keys=True) for run in runs}
    if len(values) != 1:
        raise ValueError(f"formal S1 descriptor matrix spans incompatible {key}")
    return runs[0][key]


def build_formal_s1_result_report(
    *,
    manifest_path: str | Path,
    ground_truth_path: str | Path,
    descriptor_paths: Sequence[str | Path],
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    ground_truth_path = Path(ground_truth_path).resolve()
    resolved_descriptors = [Path(path).resolve() for path in descriptor_paths]
    manifest = validate_s1_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        annotation_path=ground_truth_path,
    )
    runs = [
        _load_run_descriptor(
            path, manifest=manifest, ground_truth_path=ground_truth_path
        )
        for path in resolved_descriptors
    ]
    code_commit = _single_run_value(runs, "code_commit")
    profile_code_commit = _single_run_value(runs, "profile_code_commit")
    if profile_code_commit == code_commit:
        raise ValueError("S1 recovery analysis did not separate training/profile code")
    _single_run_value(runs, "profile_recovery_certificate_sha256")
    _single_run_value(runs, "profile_recovery_campaign_id")
    quartiles = manifest["duration_quartiles_seconds"]
    core_report = aggregate_s1_runs(
        runs,
        duration_quartiles=(quartiles["q1"], quartiles["q2"], quartiles["q3"]),
        bootstrap_replicates=S1_BOOTSTRAP_REPLICATES,
        bootstrap_seed=S1_BOOTSTRAP_SEED,
        require_three_seeds=True,
        expected_class_count=20,
    )
    sources = [
        {
            "resolution": int(run["resolution"]),
            "seed": int(run["seed"]),
            "descriptor_path": str(path),
            "descriptor_file_sha256": sha256_file(path),
            "descriptor_sha256": run["descriptor_sha256"],
        }
        for path, run in zip(resolved_descriptors, runs)
    ]
    global_identity = {
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "ground_truth_path": str(ground_truth_path),
        "ground_truth_sha256": sha256_file(ground_truth_path),
        "code_commit": code_commit,
        "profile_code_commit": profile_code_commit,
        "profile_recovery_certificate_sha256": _single_run_value(
            runs, "profile_recovery_certificate_sha256"
        ),
        "profile_recovery_campaign_id": _single_run_value(
            runs, "profile_recovery_campaign_id"
        ),
        "experiment_namespace": _single_run_value(runs, "experiment_namespace"),
        "canonical_experiment_root": _single_run_value(
            runs, "canonical_experiment_root"
        ),
        "protocol_fingerprint": _single_run_value(runs, "protocol_fingerprint"),
        "precheck_file_sha256": _single_run_value(runs, "precheck_file_sha256"),
        "precheck_sha256": _single_run_value(runs, "precheck_sha256"),
        "pretrained_checkpoint_sha256": _single_run_value(
            runs, "pretrained_checkpoint_sha256"
        ),
        "test_open_certificate_sha256": _single_run_value(
            runs, "test_open_certificate_sha256"
        ),
        "profile_order_seed": _single_run_value(runs, "profile_order_seed"),
        "profile_order_sha256": _single_run_value(runs, "profile_order_sha256"),
        "source_descriptor_matrix_sha256": canonical_sha256(
            sorted(sources, key=lambda row: (row["resolution"], row["seed"]))
        ),
    }
    return seal_s1_result_report(
        core_report,
        source_descriptors=sources,
        global_identity=global_identity,
    )


def validate_formal_s1_result_report(
    report: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    ground_truth_path: str | Path,
    descriptor_paths: Sequence[str | Path],
) -> dict[str, Any]:
    checked = validate_s1_result_report_envelope(report)
    expected = build_formal_s1_result_report(
        manifest_path=manifest_path,
        ground_truth_path=ground_truth_path,
        descriptor_paths=descriptor_paths,
    )
    return validate_s1_result_report_envelope(checked, expected_report=expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute and gate matched S1 results"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-report", type=Path)
    args = parser.parse_args(argv)
    if (args.output is None) == (args.validate_report is None):
        parser.error("choose exactly one of --output or --validate-report")
    if args.validate_report is not None:
        checked = validate_formal_s1_result_report(
            json.loads(args.validate_report.read_text(encoding="utf-8")),
            manifest_path=args.manifest,
            ground_truth_path=args.ground_truth,
            descriptor_paths=args.run,
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "report": str(args.validate_report.resolve()),
                    "report_sha256": checked["report_sha256"],
                },
                indent=2,
            )
        )
        return 0
    report = build_formal_s1_result_report(
        manifest_path=args.manifest,
        ground_truth_path=args.ground_truth,
        descriptor_paths=args.run,
    )
    expected_output = (
        Path(report["global_identity"]["canonical_experiment_root"])
        / "final"
        / "s1_go_kill_report.json"
    ).resolve()
    if args.output.resolve() != expected_output:
        raise ValueError(f"formal S1 report path must be canonical: {expected_output}")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an S1 GO/KILL report")
    atomic_publish_json(args.output, report)
    print(
        json.dumps({"status": report["status"], "output": str(args.output)}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = [
    "S1_FORMAL_REPORT_SCHEMA",
    "DetectionCorpus",
    "aggregate_s1_runs",
    "assert_official_evaluator_parity",
    "build_formal_s1_result_report",
    "evaluate_corpus",
    "seal_s1_result_report",
    "validate_formal_s1_result_report",
    "validate_s1_result_report_envelope",
]
