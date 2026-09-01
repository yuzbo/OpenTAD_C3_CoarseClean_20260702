from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from tools.bata.continuous_roi_s2_v3_full200_compute import (
    ARMS,
    PROTOCOL_ID,
    SEEDS,
    atomic_publish_json,
    canonical_json_bytes,
    canonical_sha256,
    sha256_file,
)


BOOTSTRAP_DOMAIN = "ZT_S2V3_HIERARCHICAL_BOOTSTRAP_V1"
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_MASTER_SEED = 20_260_720
PRIMARY_LCB_INDEX = 199
RATIO_UCB_INDEX = 19_800
TIOU_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
PREDICTION_BUNDLE_SCHEMA = "s2_v3_full200_complete_prediction_v1"


@dataclass(frozen=True)
class GroundTruth:
    video_id: str
    annotation_ordinal: int
    class_index: int
    start: float
    end: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.start)
            or not math.isfinite(self.end)
            or self.end <= self.start
        ):
            raise ValueError("foreground ground truth must have finite positive duration")

    @property
    def uid(self) -> tuple[str, int]:
        return self.video_id, self.annotation_ordinal

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Prediction:
    video_id: str
    source_window_ordinal: int
    raw_proposal_ordinal: int
    class_index: int
    score: float
    start: float
    end: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.score)
            or not math.isfinite(self.start)
            or not math.isfinite(self.end)
            or self.end <= self.start
        ):
            raise ValueError("prediction must have finite score and positive duration")

    @property
    def uid(self) -> tuple[str, int, int, int]:
        return (
            self.video_id,
            self.source_window_ordinal,
            self.raw_proposal_ordinal,
            self.class_index,
        )

    @property
    def order_key(self) -> tuple[Any, ...]:
        return (-self.score, self.class_index, self.start, self.end, self.uid)


@dataclass(frozen=True)
class VideoOccurrence:
    synthetic_video_id: str
    original_video_id: str


@dataclass(frozen=True)
class SlotMetrics:
    average_map_pp: float
    map_at_0_7_pp: float
    short_q1_recall: float
    normalized_start_error_median: float
    normalized_end_error_median: float


@dataclass(frozen=True)
class PredictionBundle:
    arm: str
    seed: int
    population_manifest_sha256: str
    video_order: tuple[str, ...]
    predictions_by_video: Mapping[str, tuple[Prediction, ...]]
    bundle_sha256: str

    @property
    def predictions(self) -> tuple[Prediction, ...]:
        return tuple(
            row
            for video_id in self.video_order
            for row in self.predictions_by_video[video_id]
        )


def temporal_iou(prediction: Prediction, target: GroundTruth) -> float:
    intersection = max(
        0.0, min(prediction.end, target.end) - max(prediction.start, target.start)
    )
    union = max(prediction.end, target.end) - min(prediction.start, target.start)
    if union <= 0.0:
        raise ValueError("tIoU union must be positive")
    return intersection / union


def _match_predictions(
    predictions: Sequence[Prediction],
    ground_truth: Sequence[GroundTruth],
    *,
    tiou_threshold: float,
) -> list[tuple[Prediction, GroundTruth]]:
    unmatched = set(range(len(ground_truth)))
    matches: list[tuple[Prediction, GroundTruth]] = []
    for prediction in sorted(predictions, key=lambda item: item.order_key):
        candidates: list[tuple[float, float, float, tuple[str, int], int]] = []
        for index in unmatched:
            target = ground_truth[index]
            if (
                target.video_id != prediction.video_id
                or target.class_index != prediction.class_index
            ):
                continue
            overlap = temporal_iou(prediction, target)
            if overlap >= tiou_threshold:
                candidates.append(
                    (-overlap, target.start, target.end, target.uid, index)
                )
        if candidates:
            _, _, _, _, index = min(candidates)
            unmatched.remove(index)
            matches.append((prediction, ground_truth[index]))
    return matches


def _interpolated_ap(precision: Sequence[float], recall: Sequence[float]) -> float:
    mprec = [0.0, *map(float, precision), 0.0]
    mrec = [0.0, *map(float, recall), 1.0]
    for index in range(len(mprec) - 2, -1, -1):
        mprec[index] = max(mprec[index], mprec[index + 1])
    return sum(
        (mrec[index] - mrec[index - 1]) * mprec[index]
        for index in range(1, len(mrec))
        if mrec[index] != mrec[index - 1]
    )


def _by_video(rows: Iterable[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        grouped.setdefault(str(row.video_id), []).append(row)
    return grouped


def full_class_map_vector(
    ground_truth: Sequence[GroundTruth],
    predictions: Sequence[Prediction],
    *,
    occurrences: Sequence[VideoOccurrence],
    class_count: int,
    tiou_thresholds: Sequence[float] = TIOU_THRESHOLDS,
) -> tuple[float, ...]:
    """Recompute official-style class AP on unique bootstrap occurrences."""

    if class_count <= 0:
        raise ValueError("class_count must be positive")
    thresholds = tuple(map(float, tiou_thresholds))
    if not thresholds or any(not 0.0 < value <= 1.0 for value in thresholds):
        raise ValueError("tIoU thresholds must be in (0,1]")
    if not occurrences or len({row.synthetic_video_id for row in occurrences}) != len(
        occurrences
    ):
        raise ValueError("each sampled occurrence needs a unique synthetic identity")
    gt_by_video = _by_video(ground_truth)
    pred_by_video = _by_video(predictions)
    class_vectors: list[list[float]] = []
    for class_index in range(class_count):
        gt_by_cluster: dict[str, list[GroundTruth]] = {}
        prediction_rows: list[tuple[Prediction, str]] = []
        for occurrence in occurrences:
            targets = [
                row
                for row in gt_by_video.get(occurrence.original_video_id, ())
                if row.class_index == class_index
            ]
            gt_by_cluster[occurrence.synthetic_video_id] = targets
            prediction_rows.extend(
                (row, occurrence.synthetic_video_id)
                for row in pred_by_video.get(occurrence.original_video_id, ())
                if row.class_index == class_index
            )
        npos = sum(map(len, gt_by_cluster.values()))
        if npos == 0:
            class_vectors.append([0.0] * len(thresholds))
            continue
        prediction_rows.sort(
            key=lambda item: (-item[0].score, item[1], item[0].uid)
        )
        locks = {
            cluster_id: [set() for _ in thresholds]
            for cluster_id in gt_by_cluster
        }
        true_positive = [[0.0] * len(prediction_rows) for _ in thresholds]
        false_positive = [[0.0] * len(prediction_rows) for _ in thresholds]
        for prediction_index, (prediction, cluster_id) in enumerate(prediction_rows):
            targets = gt_by_cluster[cluster_id]
            ranked = sorted(
                (
                    (-temporal_iou(prediction, target), target.start, target.end, target.uid, index)
                    for index, target in enumerate(targets)
                ),
                key=lambda row: row,
            )
            for threshold_index, threshold in enumerate(thresholds):
                for negative_overlap, _, _, _, target_index in ranked:
                    overlap = -negative_overlap
                    if overlap < threshold:
                        break
                    if target_index not in locks[cluster_id][threshold_index]:
                        locks[cluster_id][threshold_index].add(target_index)
                        true_positive[threshold_index][prediction_index] = 1.0
                        break
                if true_positive[threshold_index][prediction_index] == 0.0:
                    false_positive[threshold_index][prediction_index] = 1.0
        ap = []
        for threshold_index in range(len(thresholds)):
            cumulative_tp: list[float] = []
            cumulative_fp: list[float] = []
            tp_total = fp_total = 0.0
            for tp_value, fp_value in zip(
                true_positive[threshold_index], false_positive[threshold_index]
            ):
                tp_total += tp_value
                fp_total += fp_value
                cumulative_tp.append(tp_total)
                cumulative_fp.append(fp_total)
            recall = [value / npos for value in cumulative_tp]
            precision = [
                tp / max(tp + fp, float.fromhex("0x1.0000000000000p-1022"))
                for tp, fp in zip(cumulative_tp, cumulative_fp)
            ]
            ap.append(_interpolated_ap(precision, recall))
        class_vectors.append(ap)
    return tuple(
        100.0 * sum(row[index] for row in class_vectors) / class_count
        for index in range(len(thresholds))
    )


def _sample_short_recall(
    ground_truth: Sequence[GroundTruth],
    predictions: Sequence[Prediction],
    *,
    occurrences: Sequence[VideoOccurrence],
    q1: float,
) -> float:
    gt_by_video = _by_video(ground_truth)
    pred_by_video = _by_video(predictions)
    matched_total = target_total = 0
    for occurrence in occurrences:
        targets = [
            row
            for row in gt_by_video.get(occurrence.original_video_id, ())
            if row.duration <= q1
        ]
        target_total += len(targets)
        top100 = sorted(
            pred_by_video.get(occurrence.original_video_id, ()),
            key=lambda row: row.order_key,
        )[:100]
        matched_total += len(_match_predictions(top100, targets, tiou_threshold=0.70))
    return 0.0 if target_total == 0 else matched_total / target_total


def _sample_boundary_medians(
    ground_truth: Sequence[GroundTruth],
    predictions: Sequence[Prediction],
    *,
    occurrences: Sequence[VideoOccurrence],
) -> tuple[float, float]:
    gt_by_video = _by_video(ground_truth)
    pred_by_video = _by_video(predictions)
    start_errors: list[float] = []
    end_errors: list[float] = []
    for occurrence in occurrences:
        matches = _match_predictions(
            pred_by_video.get(occurrence.original_video_id, ()),
            gt_by_video.get(occurrence.original_video_id, ()),
            tiou_threshold=0.50,
        )
        start_errors.extend(
            abs(prediction.start - target.start) / target.duration
            for prediction, target in matches
        )
        end_errors.extend(
            abs(prediction.end - target.end) / target.duration
            for prediction, target in matches
        )
    if not start_errors:
        return math.inf, math.inf
    return float(median(start_errors)), float(median(end_errors))


def evaluate_slot_metrics(
    ground_truth: Sequence[GroundTruth],
    predictions: Sequence[Prediction],
    *,
    occurrences: Sequence[VideoOccurrence],
    class_count: int,
    q1: float,
) -> SlotMetrics:
    map_vector = full_class_map_vector(
        ground_truth,
        predictions,
        occurrences=occurrences,
        class_count=class_count,
    )
    start_error, end_error = _sample_boundary_medians(
        ground_truth, predictions, occurrences=occurrences
    )
    return SlotMetrics(
        average_map_pp=sum(map_vector) / len(map_vector),
        map_at_0_7_pp=map_vector[-1],
        short_q1_recall=_sample_short_recall(
            ground_truth, predictions, occurrences=occurrences, q1=q1
        ),
        normalized_start_error_median=start_error,
        normalized_end_error_median=end_error,
    )


def short_q1_recall(
    ground_truth: Sequence[GroundTruth],
    predictions: Sequence[Prediction],
    *,
    q1: float,
) -> tuple[int, int, float]:
    if not math.isfinite(q1) or q1 <= 0.0:
        raise ValueError("short-Q1 scalar must be finite and positive")
    short_gt = [target for target in ground_truth if target.duration <= q1]
    if not short_gt:
        return 0, 0, 0.0
    top100: list[Prediction] = []
    video_ids = sorted({target.video_id for target in ground_truth} | {item.video_id for item in predictions})
    for video_id in video_ids:
        rows = sorted(
            (item for item in predictions if item.video_id == video_id),
            key=lambda item: item.order_key,
        )
        top100.extend(rows[:100])
    matches = _match_predictions(top100, short_gt, tiou_threshold=0.70)
    return len(matches), len(short_gt), len(matches) / len(short_gt)


def normalized_boundary_medians(
    ground_truth: Sequence[GroundTruth],
    predictions: Sequence[Prediction],
) -> tuple[float, float]:
    matches = _match_predictions(predictions, ground_truth, tiou_threshold=0.50)
    if not matches:
        return math.inf, math.inf
    start_errors = [
        abs(prediction.start - target.start) / target.duration
        for prediction, target in matches
    ]
    end_errors = [
        abs(prediction.end - target.end) / target.duration
        for prediction, target in matches
    ]
    return float(median(start_errors)), float(median(end_errors))


def prediction_rows_to_objects(
    video_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    class_map: Sequence[str] | None = None,
) -> list[Prediction]:
    predictions: list[Prediction] = []
    seen_uids = set()
    for row in rows:
        uid = row.get("prediction_uid")
        if not isinstance(uid, list) or len(uid) != 4 or str(uid[0]) != video_id:
            raise ValueError("prediction_uid must be [video_id,window,proposal,class]")
        prediction = Prediction(
            video_id=video_id,
            source_window_ordinal=int(uid[1]),
            raw_proposal_ordinal=int(uid[2]),
            class_index=int(uid[3]),
            score=float(row["score"]),
            start=float(row["segment"][0]),
            end=float(row["segment"][1]),
        )
        if class_map is not None:
            if not 0 <= prediction.class_index < len(class_map):
                raise ValueError("prediction class index is outside the frozen class map")
            if row.get("label") != class_map[prediction.class_index]:
                raise ValueError("prediction label differs from its frozen class index")
        if prediction.uid in seen_uids:
            raise ValueError("duplicate prediction_uid")
        seen_uids.add(prediction.uid)
        predictions.append(prediction)
    return predictions


def build_prediction_bundle_payload(
    *,
    arm: str,
    seed: int,
    population_manifest_sha256: str,
    video_order: Sequence[str],
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    class_map: Sequence[str],
) -> dict[str, Any]:
    videos = tuple(map(str, video_order))
    if len(videos) != 211 or len(set(videos)) != 211:
        raise ValueError("prediction bundle requires the complete 211-video population")
    if set(results) != set(videos):
        raise ValueError("prediction results must include every video, including empty videos")
    prediction_count = 0
    normalized_results: dict[str, list[Mapping[str, Any]]] = {}
    for video_id in videos:
        rows = list(results[video_id])
        prediction_rows_to_objects(video_id, rows, class_map=class_map)
        normalized_results[video_id] = rows
        prediction_count += len(rows)
    payload: dict[str, Any] = {
        "schema_version": PREDICTION_BUNDLE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "arm": str(arm),
        "seed": int(seed),
        "population_manifest_sha256": str(population_manifest_sha256),
        "video_order": list(videos),
        "video_count": len(videos),
        "prediction_count": prediction_count,
        "results": normalized_results,
    }
    if payload["arm"] not in ARMS or payload["seed"] not in SEEDS:
        raise ValueError("prediction bundle arm or seed is outside the frozen matrix")
    payload["bundle_sha256"] = canonical_sha256(payload)
    return payload


def load_complete_prediction_bundle(
    path: str | Path,
    *,
    expected_arm: str,
    expected_seed: int,
    expected_population_manifest_sha256: str,
    expected_video_order: Sequence[str],
    class_map: Sequence[str],
) -> PredictionBundle:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    checked = dict(payload)
    digest = checked.pop("bundle_sha256", None)
    if not isinstance(digest, str) or canonical_sha256(checked) != digest:
        raise ValueError("prediction bundle self-hash mismatch")
    videos = tuple(map(str, expected_video_order))
    if (
        payload.get("schema_version") != PREDICTION_BUNDLE_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("arm") != expected_arm
        or int(payload.get("seed", -1)) != int(expected_seed)
        or payload.get("population_manifest_sha256")
        != expected_population_manifest_sha256
        or tuple(map(str, payload.get("video_order", ()))) != videos
        or int(payload.get("video_count", -1)) != len(videos)
    ):
        raise ValueError("prediction bundle identity differs from the frozen cell")
    results = payload.get("results")
    if not isinstance(results, Mapping) or set(results) != set(videos):
        raise ValueError("prediction bundle omits or adds evaluation videos")
    by_video: dict[str, tuple[Prediction, ...]] = {}
    for video_id in videos:
        rows = results[video_id]
        if not isinstance(rows, list):
            raise ValueError("each prediction result entry must be a list")
        by_video[video_id] = tuple(
            prediction_rows_to_objects(video_id, rows, class_map=class_map)
        )
    if sum(map(len, by_video.values())) != int(payload.get("prediction_count", -1)):
        raise ValueError("prediction bundle count differs from its rows")
    return PredictionBundle(
        arm=expected_arm,
        seed=int(expected_seed),
        population_manifest_sha256=expected_population_manifest_sha256,
        video_order=videos,
        predictions_by_video=by_video,
        bundle_sha256=digest,
    )


def sha_counter_index(*, key: str, n: int, rejection_counter: int = 0) -> int:
    if n <= 0:
        raise ValueError("counter sampler range must be positive")
    counter = int(rejection_counter)
    while True:
        payload = "\0".join(
            (BOOTSTRAP_DOMAIN, str(BOOTSTRAP_MASTER_SEED), key, str(counter))
        ).encode("ascii")
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        limit = ((1 << 64) // n) * n
        if value < limit:
            return value % n
        counter += 1


def bootstrap_draws(
    replicate: int, video_cluster_order: Sequence[str]
) -> dict[str, Any]:
    if replicate < 0 or replicate >= BOOTSTRAP_REPLICATES:
        raise ValueError("bootstrap replicate ordinal is outside the frozen range")
    if len(video_cluster_order) != 211 or list(video_cluster_order) != sorted(
        video_cluster_order
    ):
        raise ValueError("bootstrap requires 211 UTF-8 lexicographically ordered videos")
    seed_draws = []
    video_draws = []
    for seed_slot in range(3):
        seed_draws.append(
            SEEDS[sha_counter_index(key=f"seed/{replicate}/{seed_slot}", n=3)]
        )
        video_draws.append(
            [
                sha_counter_index(
                    key=f"video/{replicate}/{seed_slot}/{video_slot}", n=211
                )
                for video_slot in range(211)
            ]
        )
    return {
        "replicate": replicate,
        "seed_draws": seed_draws,
        "video_index_draws": video_draws,
    }


def bootstrap_occurrences(
    draw: Mapping[str, Any],
    video_cluster_order: Sequence[str],
    *,
    seed_slot: int,
) -> tuple[VideoOccurrence, ...]:
    replicate = int(draw["replicate"])
    seed = int(draw["seed_draws"][seed_slot])
    indices = list(map(int, draw["video_index_draws"][seed_slot]))
    if len(indices) != 211 or any(not 0 <= index < 211 for index in indices):
        raise ValueError("bootstrap video draw is not a complete 211-cluster sample")
    return tuple(
        VideoOccurrence(
            synthetic_video_id=(
                f"boot/{replicate:05d}/seedslot/{seed_slot}/origseed/{seed}/"
                f"videoslot/{video_slot:03d}/origvideo/{video_cluster_order[index]}"
            ),
            original_video_id=str(video_cluster_order[index]),
        )
        for video_slot, index in enumerate(indices)
    )


def _mean_slot_metrics(rows: Sequence[SlotMetrics]) -> dict[str, float]:
    if len(rows) != 3:
        raise ValueError("arm aggregation requires exactly three synthetic seed slots")
    keys = (
        "average_map_pp",
        "map_at_0_7_pp",
        "short_q1_recall",
        "normalized_start_error_median",
        "normalized_end_error_median",
    )
    return {
        key: sum(float(getattr(row, key)) for row in rows) / 3.0 for key in keys
    }


def _ratio(candidate: float, reference: float) -> float:
    if not math.isfinite(reference) or reference <= 0.0:
        return math.inf
    if not math.isfinite(candidate):
        return math.inf
    return candidate / reference


def _atomic_jsonl_handles(paths: Sequence[Path]) -> tuple[list[Any], list[Path]]:
    handles = []
    temporaries = []
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporaries.append(temporary)
        handles.append(temporary.open("xb"))
    return handles, temporaries


def _publish_jsonl_handles(
    handles: Sequence[Any], temporaries: Sequence[Path], targets: Sequence[Path]
) -> None:
    for handle in handles:
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
    published: list[Path] = []
    try:
        for temporary, target in zip(temporaries, targets):
            os.link(temporary, target)
            published.append(target)
    except Exception:
        for target in published:
            target.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)


def _write_jsonl_row(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(canonical_json_bytes(_json_safe(row)) + b"\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("NaN is forbidden in a formal result artifact")
        if math.isinf(value):
            return "+Infinity" if value > 0.0 else "-Infinity"
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run_hierarchical_bootstrap(
    *,
    ground_truth: Sequence[GroundTruth],
    predictions: Mapping[str, Mapping[int, Sequence[Prediction]]],
    video_cluster_order: Sequence[str],
    class_count: int,
    q1: float,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run and atomically seal the exact 20,000-replicate nested bootstrap."""

    if set(predictions) != set(ARMS):
        raise ValueError("bootstrap prediction matrix must contain three frozen arms")
    for arm in ARMS:
        if set(predictions[arm]) != set(SEEDS):
            raise ValueError(f"bootstrap prediction matrix for {arm} is not 3/3")
    videos = tuple(map(str, video_cluster_order))
    if len(videos) != 211 or tuple(sorted(videos)) != videos or len(set(videos)) != 211:
        raise ValueError("bootstrap video cluster order must be 211 sorted unique IDs")
    if not any(target.duration <= q1 for target in ground_truth):
        raise ValueError("complete population has no Short-Q1 ground truth")
    expected_videos = set(videos)
    if {target.video_id for target in ground_truth} - expected_videos:
        raise ValueError("ground truth contains an out-of-population video")
    for arm in ARMS:
        for seed in SEEDS:
            if {row.video_id for row in predictions[arm][seed]} - expected_videos:
                raise ValueError("prediction contains an out-of-population video")

    full_occurrences = tuple(VideoOccurrence(video_id, video_id) for video_id in videos)
    point_per_seed: dict[str, dict[int, SlotMetrics]] = {
        arm: {
            seed: evaluate_slot_metrics(
                ground_truth,
                predictions[arm][seed],
                occurrences=full_occurrences,
                class_count=class_count,
                q1=q1,
            )
            for seed in SEEDS
        }
        for arm in ARMS
    }
    for seed in SEEDS:
        reference = point_per_seed["D160"][seed]
        if (
            not math.isfinite(reference.normalized_start_error_median)
            or reference.normalized_start_error_median <= 0.0
            or not math.isfinite(reference.normalized_end_error_median)
            or reference.normalized_end_error_median <= 0.0
        ):
            raise ValueError("D160 point boundary denominator is not positive and finite")
    point = {
        arm: _mean_slot_metrics([point_per_seed[arm][seed] for seed in SEEDS])
        for arm in ARMS
    }

    output_dir = Path(output_dir).resolve()
    targets = [
        output_dir / "bootstrap_draws.jsonl",
        output_dir / "bootstrap_replicates_primary.jsonl",
        output_dir / "bootstrap_replicates_g96_dominance.jsonl",
    ]
    handles, temporaries = _atomic_jsonl_handles(targets)
    primary_replicates = {name: [] for name in ("average_map", "map_at_0_7", "short", "start", "end")}
    dominance_replicates = {name: [] for name in primary_replicates}
    try:
        for replicate in range(BOOTSTRAP_REPLICATES):
            draw = bootstrap_draws(replicate, videos)
            _write_jsonl_row(handles[0], draw)
            arm_slots: dict[str, list[SlotMetrics]] = {arm: [] for arm in ARMS}
            for seed_slot, seed in enumerate(draw["seed_draws"]):
                occurrences = bootstrap_occurrences(
                    draw, videos, seed_slot=seed_slot
                )
                for arm in ARMS:
                    arm_slots[arm].append(
                        evaluate_slot_metrics(
                            ground_truth,
                            predictions[arm][int(seed)],
                            occurrences=occurrences,
                            class_count=class_count,
                            q1=q1,
                        )
                    )
            arm_metrics = {
                arm: _mean_slot_metrics(arm_slots[arm]) for arm in ARMS
            }
            primary = {
                "replicate": replicate,
                "average_map_difference_pp": arm_metrics["U128-A0"]["average_map_pp"]
                - arm_metrics["D160"]["average_map_pp"],
                "map_at_0_7_difference_pp": arm_metrics["U128-A0"]["map_at_0_7_pp"]
                - arm_metrics["D160"]["map_at_0_7_pp"],
                "short_q1_recall_difference": arm_metrics["U128-A0"]["short_q1_recall"]
                - arm_metrics["D160"]["short_q1_recall"],
                "start_error_ratio": _ratio(
                    arm_metrics["U128-A0"]["normalized_start_error_median"],
                    arm_metrics["D160"]["normalized_start_error_median"],
                ),
                "end_error_ratio": _ratio(
                    arm_metrics["U128-A0"]["normalized_end_error_median"],
                    arm_metrics["D160"]["normalized_end_error_median"],
                ),
            }
            dominance = {
                "replicate": replicate,
                "average_map_difference_pp": arm_metrics["G96"]["average_map_pp"]
                - arm_metrics["U128-A0"]["average_map_pp"],
                "map_at_0_7_difference_pp": arm_metrics["G96"]["map_at_0_7_pp"]
                - arm_metrics["U128-A0"]["map_at_0_7_pp"],
                "short_q1_recall_difference": arm_metrics["G96"]["short_q1_recall"]
                - arm_metrics["U128-A0"]["short_q1_recall"],
                "start_error_ratio": _ratio(
                    arm_metrics["G96"]["normalized_start_error_median"],
                    arm_metrics["U128-A0"]["normalized_start_error_median"],
                ),
                "end_error_ratio": _ratio(
                    arm_metrics["G96"]["normalized_end_error_median"],
                    arm_metrics["U128-A0"]["normalized_end_error_median"],
                ),
            }
            _write_jsonl_row(handles[1], primary)
            _write_jsonl_row(handles[2], dominance)
            for store, row in (
                (primary_replicates, primary),
                (dominance_replicates, dominance),
            ):
                store["average_map"].append(row["average_map_difference_pp"])
                store["map_at_0_7"].append(row["map_at_0_7_difference_pp"])
                store["short"].append(row["short_q1_recall_difference"])
                store["start"].append(row["start_error_ratio"])
                store["end"].append(row["end_error_ratio"])
        _publish_jsonl_handles(handles, temporaries, targets)
    except Exception:
        for handle in handles:
            if not handle.closed:
                handle.close()
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
        raise

    def bounds(rows: Mapping[str, Sequence[float]]) -> dict[str, float]:
        return {
            "average_map_difference_lcb_pp": sorted(rows["average_map"])[PRIMARY_LCB_INDEX],
            "map_at_0_7_difference_lcb_pp": sorted(rows["map_at_0_7"])[PRIMARY_LCB_INDEX],
            "short_q1_recall_difference_lcb": sorted(rows["short"])[PRIMARY_LCB_INDEX],
            "start_error_ratio_ucb": sorted(rows["start"])[RATIO_UCB_INDEX],
            "end_error_ratio_ucb": sorted(rows["end"])[RATIO_UCB_INDEX],
        }

    bounds_payload: dict[str, Any] = {
        "schema_version": "s2_v3_full200_bootstrap_bounds_v1",
        "protocol_id": PROTOCOL_ID,
        "point_per_seed": {
            arm: {str(seed): point_per_seed[arm][seed].__dict__ for seed in SEEDS}
            for arm in ARMS
        },
        "point_arm_mean": point,
        "primary": bounds(primary_replicates),
        "g96_dominance": bounds(dominance_replicates),
        "replicates": BOOTSTRAP_REPLICATES,
        "master_seed": BOOTSTRAP_MASTER_SEED,
        "draws_file_sha256": sha256_file(targets[0]),
        "primary_replicates_file_sha256": sha256_file(targets[1]),
        "g96_dominance_replicates_file_sha256": sha256_file(targets[2]),
    }
    bounds_payload = _json_safe(bounds_payload)
    bounds_payload["bounds_sha256"] = canonical_sha256(bounds_payload)
    atomic_publish_json(output_dir / "bootstrap_bounds.json", bounds_payload)
    schema = {
        "schema_version": "s2_v3_full200_bootstrap_schema_v1",
        "protocol_id": PROTOCOL_ID,
        "rng_domain": BOOTSTRAP_DOMAIN,
        "replicates": BOOTSTRAP_REPLICATES,
        "master_seed": BOOTSTRAP_MASTER_SEED,
        "seed_order": list(SEEDS),
        "video_cluster_order": list(videos),
        "video_cluster_order_sha256": canonical_sha256(videos),
        "difference_lcb_index_zero_based": PRIMARY_LCB_INDEX,
        "ratio_ucb_index_zero_based": RATIO_UCB_INDEX,
        "interpolation": "none",
    }
    schema["schema_sha256"] = canonical_sha256(schema)
    atomic_publish_json(output_dir / "bootstrap_schema.json", schema)
    return bounds_payload


def simultaneous_bounds(
    difference_replicates: Sequence[float], ratio_replicates: Sequence[float]
) -> tuple[float, float]:
    if len(difference_replicates) != BOOTSTRAP_REPLICATES or len(
        ratio_replicates
    ) != BOOTSTRAP_REPLICATES:
        raise ValueError("simultaneous bounds require exactly 20,000 replicates")
    if any(math.isnan(float(value)) for value in difference_replicates):
        raise ValueError("difference bootstrap contains NaN")
    if any(math.isnan(float(value)) for value in ratio_replicates):
        raise ValueError("ratio bootstrap contains NaN")
    difference = sorted(map(float, difference_replicates))
    ratio = sorted(map(float, ratio_replicates))
    return difference[PRIMARY_LCB_INDEX], ratio[RATIO_UCB_INDEX]


def build_prediction_seal(
    prediction_paths: Mapping[str, Mapping[int, str | Path]],
    *,
    checkpoint_seal_sha256: str,
    population_manifest_sha256: str,
    expected_video_order: Sequence[str],
    class_map: Sequence[str],
    output_path: str | Path,
) -> dict[str, Any]:
    if set(prediction_paths) != set(ARMS):
        raise ValueError("prediction seal requires exactly the three frozen arms")
    rows = []
    for arm in ARMS:
        if set(prediction_paths[arm]) != set(SEEDS):
            raise ValueError(f"prediction seal for {arm} is not 3/3 complete")
        for seed in SEEDS:
            path = Path(prediction_paths[arm][seed]).resolve()
            if not path.is_file():
                raise ValueError(f"missing prediction file: {path}")
            bundle = load_complete_prediction_bundle(
                path,
                expected_arm=arm,
                expected_seed=seed,
                expected_population_manifest_sha256=population_manifest_sha256,
                expected_video_order=expected_video_order,
                class_map=class_map,
            )
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "path": path.as_posix(),
                    "sha256": sha256_file(path),
                    "bundle_sha256": bundle.bundle_sha256,
                    "video_count": len(bundle.video_order),
                    "prediction_count": len(bundle.predictions),
                }
            )
    seal: dict[str, Any] = {
        "schema_version": "s2_v3_full200_prediction_seal_v1",
        "protocol_id": PROTOCOL_ID,
        "checkpoint_seal_sha256": checkpoint_seal_sha256,
        "population_manifest_sha256": population_manifest_sha256,
        "video_order_sha256": canonical_sha256(tuple(map(str, expected_video_order))),
        "class_map_sha256": canonical_sha256(tuple(map(str, class_map))),
        "rows": rows,
        "row_count": len(rows),
    }
    seal["seal_sha256"] = canonical_sha256(seal)
    atomic_publish_json(output_path, seal)
    return seal


def _validated_gt_open_marker(
    *, marker_path: str | Path, annotation_path: str | Path
) -> dict[str, Any]:
    marker = json.loads(Path(marker_path).read_text(encoding="utf-8"))
    checked = dict(marker)
    digest = checked.pop("marker_sha256", None)
    annotation = Path(annotation_path).resolve()
    if not isinstance(digest, str) or canonical_sha256(checked) != digest:
        raise ValueError("GT-opening marker self-hash mismatch")
    if (
        marker.get("schema_version") != "s2_v3_single_gt_open_v1"
        or marker.get("protocol_id") != PROTOCOL_ID
        or marker.get("consumer") != "task_local_final_evaluator"
        or marker.get("annotation_path") != annotation.as_posix()
        or marker.get("annotation_sha256") != sha256_file(annotation)
    ):
        raise ValueError("GT-opening marker does not bind the frozen annotation")
    return marker


def load_ground_truth_after_single_open(
    *,
    marker_path: str | Path,
    annotation_path: str | Path,
    expected_video_order: Sequence[str],
    class_map: Sequence[str],
) -> tuple[GroundTruth, ...]:
    """Read metric-bearing GT only after the irreversible marker exists."""

    _validated_gt_open_marker(
        marker_path=marker_path, annotation_path=annotation_path
    )
    payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    database = payload.get("database")
    if not isinstance(database, Mapping):
        raise ValueError("metric-bearing annotation has no database mapping")
    expected = tuple(map(str, expected_video_order))
    validation = {
        str(video_id): row
        for video_id, row in database.items()
        if str(row.get("subset")) == "validation"
    }
    if set(validation) != set(expected) or len(expected) != 211:
        raise ValueError("GT opening is not the complete frozen 211-video population")
    class_index = {str(label): index for index, label in enumerate(class_map)}
    if len(class_index) != len(class_map) or not class_index:
        raise ValueError("frozen class map is empty or non-unique")
    rows: list[GroundTruth] = []
    for video_id in expected:
        annotations = validation[video_id].get("annotations")
        if not isinstance(annotations, list):
            raise ValueError(f"{video_id} has no metric-bearing annotations")
        retained: list[tuple[str, float, float]] = []
        for annotation_ordinal, annotation in enumerate(annotations):
            label = str(annotation.get("label"))
            segment = annotation.get("segment")
            if not isinstance(segment, list) or len(segment) != 2:
                raise ValueError("metric-bearing annotation segment is invalid")
            start, end = map(float, segment)
            if end - start <= 0.0:
                continue
            if any(
                previous_label == label
                and abs(previous_start - start) <= 1e-3
                and abs(previous_end - end) <= 1e-3
                for previous_label, previous_start, previous_end in retained
            ):
                continue
            retained.append((label, start, end))
            if label not in class_index:
                continue
            rows.append(
                GroundTruth(
                    video_id=video_id,
                    annotation_ordinal=annotation_ordinal,
                    class_index=class_index[label],
                    start=start,
                    end=end,
                )
            )
    if not rows:
        raise ValueError("complete evaluation population has no foreground GT")
    return tuple(rows)


def assert_official_point_evaluator_parity(
    *,
    marker_path: str | Path,
    annotation_path: str | Path,
    bundle: PredictionBundle,
    class_map: Sequence[str],
    atol: float = 1e-12,
) -> dict[str, float]:
    """Cross-check task-local point mAP against the unchanged official evaluator."""

    ground_truth = load_ground_truth_after_single_open(
        marker_path=marker_path,
        annotation_path=annotation_path,
        expected_video_order=bundle.video_order,
        class_map=class_map,
    )
    occurrences = tuple(VideoOccurrence(video_id, video_id) for video_id in bundle.video_order)
    local = full_class_map_vector(
        ground_truth,
        bundle.predictions,
        occurrences=occurrences,
        class_count=len(class_map),
    )
    official_predictions = {
        "results": {
            video_id: [
                {
                    "label": class_map[row.class_index],
                    "segment": [row.start, row.end],
                    "score": row.score,
                }
                for row in bundle.predictions_by_video[video_id]
            ]
            for video_id in bundle.video_order
        }
    }
    from opentad.evaluations.mAP import mAP

    evaluator = mAP(
        ground_truth_filename=str(Path(annotation_path).resolve()),
        prediction_filename=official_predictions,
        subset="validation",
        tiou_thresholds=TIOU_THRESHOLDS,
        thread=1,
    )
    if set(evaluator.activity_index) != set(map(str, class_map)):
        raise ValueError("official evaluator class population differs from the frozen map")
    official = evaluator.evaluate()
    official_vector = tuple(
        100.0 * float(official[f"mAP@{threshold}"])
        for threshold in TIOU_THRESHOLDS
    )
    if any(abs(left - right) > atol for left, right in zip(local, official_vector)):
        raise ValueError(
            f"task-local mAP differs from official evaluator: local={local} official={official_vector}"
        )
    return {
        "average_mAP": float(official["average_mAP"]),
        **{
            f"mAP@{threshold}": float(official[f"mAP@{threshold}"])
            for threshold in TIOU_THRESHOLDS
        },
    }


def begin_single_gt_open(
    *,
    marker_path: str | Path,
    annotation_path: str | Path,
    prediction_seal_path: str | Path,
    expected_prediction_seal_sha256: str,
) -> dict[str, Any]:
    """Create the irreversible marker before the evaluator can read GT bytes."""

    marker_path = Path(marker_path)
    if marker_path.exists():
        raise FileExistsError("metric-bearing GT was already opened")
    prediction_seal_path = Path(prediction_seal_path).resolve()
    if sha256_file(prediction_seal_path) != expected_prediction_seal_sha256:
        raise ValueError("prediction seal hash mismatch")
    seal = json.loads(prediction_seal_path.read_text(encoding="utf-8"))
    if seal.get("row_count") != 9 or len(seal.get("rows", [])) != 9:
        raise ValueError("GT opening requires a complete 9/9 prediction seal")
    marker: dict[str, Any] = {
        "schema_version": "s2_v3_single_gt_open_v1",
        "protocol_id": PROTOCOL_ID,
        "consumer": "task_local_final_evaluator",
        "annotation_path": Path(annotation_path).resolve().as_posix(),
        "annotation_sha256": sha256_file(annotation_path),
        "prediction_seal_path": prediction_seal_path.as_posix(),
        "prediction_seal_sha256": expected_prediction_seal_sha256,
    }
    marker["marker_sha256"] = canonical_sha256(marker)
    atomic_publish_json(marker_path, marker)
    return marker


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot evaluator and prediction sealer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal-predictions")
    seal.add_argument("--prediction-dir", type=Path, required=True)
    seal.add_argument("--checkpoint-seal", type=Path, required=True)
    seal.add_argument("--manifest", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate-matrix")
    evaluate.add_argument("--prediction-seal", type=Path, required=True)
    evaluate.add_argument("--checkpoint-seal", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--annotation", type=Path, required=True)
    evaluate.add_argument("--compute-comparison", type=Path)
    evaluate.add_argument("--marker-path", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "seal-predictions":
        checkpoint_seal = json.loads(Path(args.checkpoint_seal).read_text(encoding="utf-8"))
        checkpoint_seal_sha = checkpoint_seal.get("seal_sha256", sha256_file(args.checkpoint_seal))
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        manifest_sha = manifest.get("manifest_sha256", sha256_file(args.manifest))
        prediction_paths: dict[str, dict[int, Path]] = {}
        for arm in ARMS:
            prediction_paths[arm] = {}
            for seed in SEEDS:
                path = args.prediction_dir / f"prediction_{arm}_seed{seed}.json"
                prediction_paths[arm][seed] = path
        seal = build_prediction_seal(
            prediction_paths,
            checkpoint_seal_sha256=checkpoint_seal_sha,
            population_manifest_sha256=manifest_sha,
            expected_video_order=manifest["evaluation"]["video_order"],
            class_map=manifest["class_map"]["classes"],
            output_path=args.output,
        )
        print(json.dumps({"status": "PASS", "seal_sha256": seal["seal_sha256"]}, sort_keys=True))
        return 0
    elif args.command == "evaluate-matrix":
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        manifest_sha = manifest.get("manifest_sha256", sha256_file(args.manifest))
        prediction_seal = json.loads(Path(args.prediction_seal).read_text(encoding="utf-8"))
        prediction_seal_sha = prediction_seal.get("seal_sha256", sha256_file(args.prediction_seal))
        begin_single_gt_open(
            marker_path=args.marker_path,
            annotation_path=args.annotation,
            prediction_seal_path=args.prediction_seal,
            expected_prediction_seal_sha256=prediction_seal_sha,
        )
        gt = load_ground_truth_after_single_open(
            marker_path=args.marker_path,
            annotation_path=args.annotation,
            expected_video_order=manifest["evaluation"]["video_order"],
            class_map=manifest["class_map"]["classes"],
        )
        q1 = float.fromhex(manifest["short_q1"]["q1_float64_hex"])
        class_map = manifest["class_map"]["classes"]
        predictions: dict[str, dict[int, list[Prediction]]] = {arm: {} for arm in ARMS}
        for row in prediction_seal["rows"]:
            arm = row["arm"]
            seed = int(row["seed"])
            bundle = load_complete_prediction_bundle(
                row["path"],
                expected_arm=arm,
                expected_seed=seed,
                expected_population_manifest_sha256=manifest_sha,
                expected_video_order=manifest["evaluation"]["video_order"],
                class_map=class_map,
            )
            assert_official_point_evaluator_parity(
                marker_path=args.marker_path,
                annotation_path=args.annotation,
                bundle=bundle,
                class_map=class_map,
            )
            predictions[arm][seed] = list(bundle.predictions)
        bounds_result = run_hierarchical_bootstrap(
            ground_truth=gt,
            predictions=predictions,
            video_cluster_order=manifest["evaluation"]["video_cluster_order"],
            class_count=len(class_map),
            q1=q1,
            output_dir=args.output_dir,
        )
        primary_bounds = bounds_result["primary"]
        g96_bounds = bounds_result["g96_dominance"]
        # Gate checks:
        gates = {
            "avg_map_difference_lcb_ge_neg_1_00": primary_bounds["average_map_difference_lcb_pp"] >= -1.00,
            "map_at_0_7_difference_lcb_ge_neg_1_50": primary_bounds["map_at_0_7_difference_lcb_pp"] >= -1.50,
            "short_q1_recall_difference_lcb_ge_neg_0_03": primary_bounds["short_q1_recall_difference_lcb"] >= -0.03,
            "start_error_ratio_ucb_le_1_15": primary_bounds["start_error_ratio_ucb"] <= 1.15,
            "end_error_ratio_ucb_le_1_15": primary_bounds["end_error_ratio_ucb"] <= 1.15,
        }
        # Check seed floors
        seed_deltas = {}
        for seed in SEEDS:
            u128_map = bounds_result["point_per_seed"]["U128-A0"][str(seed)]["average_map_pp"]
            d160_map = bounds_result["point_per_seed"]["D160"][str(seed)]["average_map_pp"]
            delta = u128_map - d160_map
            seed_deltas[str(seed)] = delta
            gates[f"seed_{seed}_avg_map_delta_ge_neg_3_00"] = delta >= -3.00
        compute_gate = None
        if args.compute_comparison is not None and args.compute_comparison.is_file():
            comp = json.loads(args.compute_comparison.read_text(encoding="utf-8"))
            compute_gate = comp.get("primary_exact_10u_le_9d", False)
            gates["compute_c_exec_u128_a0_over_d160_le_0_90"] = compute_gate
        g96_dominates = (
            g96_bounds["average_map_difference_lcb_pp"] >= 0.0
            and g96_bounds["map_at_0_7_difference_lcb_pp"] >= 0.0
            and g96_bounds["short_q1_recall_difference_lcb"] >= 0.0
            and g96_bounds["start_error_ratio_ucb"] <= 1.0
            and g96_bounds["end_error_ratio_ucb"] <= 1.0
        )
        all_passed = all(gates.values())
        if all_passed:
            verdict = "REVISE_TO_G96_CONTROL_ONLY" if g96_dominates else "CONTINUE_S2_V3_A0"
        else:
            verdict = "STOP_S2_V3_A0_EXACT_ROUTE"
        evaluation_summary = {
            "schema_version": "s2_v3_admission_evaluation_v1",
            "protocol_id": PROTOCOL_ID,
            "verdict": verdict,
            "all_gates_passed": all_passed,
            "g96_strictly_dominates": g96_dominates,
            "gates": gates,
            "seed_deltas": seed_deltas,
            "bootstrap_bounds": bounds_result,
        }
        evaluation_summary["evaluation_sha256"] = canonical_sha256(evaluation_summary)
        atomic_publish_json(args.output_dir / "admission_evaluation.json", evaluation_summary)
        print(json.dumps({"status": "PASS", "verdict": verdict, "evaluation_sha256": evaluation_summary["evaluation_sha256"]}, sort_keys=True))
        return 0
    return 1


__all__ = [
    "BOOTSTRAP_MASTER_SEED",
    "BOOTSTRAP_REPLICATES",
    "GroundTruth",
    "Prediction",
    "PredictionBundle",
    "SlotMetrics",
    "VideoOccurrence",
    "begin_single_gt_open",
    "assert_official_point_evaluator_parity",
    "bootstrap_draws",
    "bootstrap_occurrences",
    "build_prediction_seal",
    "build_prediction_bundle_payload",
    "evaluate_slot_metrics",
    "full_class_map_vector",
    "load_complete_prediction_bundle",
    "load_ground_truth_after_single_open",
    "normalized_boundary_medians",
    "prediction_rows_to_objects",
    "sha_counter_index",
    "short_q1_recall",
    "simultaneous_bounds",
    "temporal_iou",
    "run_hierarchical_bootstrap",
]


if __name__ == "__main__":
    raise SystemExit(main())
