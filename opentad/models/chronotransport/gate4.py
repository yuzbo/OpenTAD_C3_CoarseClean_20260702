from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from opentad.evaluations.mAP import compute_average_precision_detection

from .protocol import canonical_sha256


SEEDS = (3407, 3408, 3409)
ARMS = ("dense", "chronotransport", "static")
ARM_ORDERS = (
    ("dense", "chronotransport", "static"),
    ("chronotransport", "static", "dense"),
    ("static", "dense", "chronotransport"),
    ("static", "chronotransport", "dense"),
    ("chronotransport", "dense", "static"),
    ("dense", "static", "chronotransport"),
)
_TIMING_FIELDS = {
    "seed",
    "official_video_id",
    "invocation_id",
    "repetition_id",
    "invocation_order_index",
    "arm_order",
    "arms",
}
_ARM_FIELDS = {
    "total_ms",
    "peak_gpu_memory_bytes",
    "nvml_energy_j",
    "stage_ms",
}
_STAGE_FIELDS = {
    "decode_ms",
    "preprocess_ms",
    "h2d_ms",
    "patch_embed_ms",
    "heavy_ms",
    "innovation_ms",
    "scheduler_ms",
    "transport_ms",
    "cache_movement_ms",
    "adapter_ms",
    "head_ms",
    "postprocess_ms",
}
_REGRET_FIELDS = {
    "seed",
    "official_video_id",
    "invocation_id",
    "dense_detector_loss",
    "chronotransport_detector_loss",
    "static_detector_loss",
}


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a numeric scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        qualifier = "positive " if positive else "non-negative "
        raise ValueError(f"{label} must be one finite {qualifier}scalar")
    return result


def _integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty canonical string")
    return value


def _percentile(values: Sequence[float], q: float) -> float:
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("bootstrap distribution must be non-empty and finite")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _p50(values: Sequence[float]) -> float:
    return _percentile(values, 50.0)


def _normalize_timing(rows: Sequence[Mapping[str, Any]]):
    if not isinstance(rows, Sequence):
        raise TypeError("Gate-4 timing rows must be a sequence")
    by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in SEEDS}
    for ordinal, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != _TIMING_FIELDS:
            raise ValueError(f"Gate-4 timing row {ordinal} fields mismatch")
        seed = _integer(raw["seed"], "timing seed")
        if seed not in by_seed:
            raise ValueError("Gate-4 timing seed must be 3407, 3408, or 3409")
        arm_order = raw["arm_order"]
        if not isinstance(arm_order, list) or tuple(arm_order) not in ARM_ORDERS:
            raise ValueError("Gate-4 timing arm order is invalid")
        arms = raw["arms"]
        if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
            raise ValueError("Gate-4 timing arms must be the complete D/C/S set")
        normalized_arms = {}
        for arm in ARMS:
            payload = arms[arm]
            if not isinstance(payload, Mapping) or set(payload) != _ARM_FIELDS:
                raise ValueError(f"Gate-4 timing arm {arm} fields mismatch")
            stages = payload["stage_ms"]
            if not isinstance(stages, Mapping) or set(stages) != _STAGE_FIELDS:
                raise ValueError(f"Gate-4 timing arm {arm} stage fields mismatch")
            memory = _integer(
                payload["peak_gpu_memory_bytes"], f"timing {arm} peak GPU memory"
            )
            if memory < 0:
                raise ValueError("Gate-4 peak GPU memory must be non-negative")
            normalized_arms[arm] = {
                "total_ms": _finite(
                    payload["total_ms"], f"timing {arm} total_ms", positive=True
                ),
                "peak_gpu_memory_bytes": memory,
                "nvml_energy_j": _finite(
                    payload["nvml_energy_j"], f"timing {arm} NVML energy"
                ),
                "stage_ms": {
                    field: _finite(stages[field], f"timing {arm} {field}")
                    for field in sorted(_STAGE_FIELDS)
                },
            }
        by_seed[seed].append(
            {
                "seed": seed,
                "official_video_id": _text(
                    raw["official_video_id"], "timing official video ID"
                ),
                "invocation_id": _text(raw["invocation_id"], "timing invocation ID"),
                "repetition_id": _integer(raw["repetition_id"], "timing repetition ID"),
                "invocation_order_index": _integer(
                    raw["invocation_order_index"], "timing invocation order index"
                ),
                "arm_order": tuple(arm_order),
                "arms": normalized_arms,
            }
        )

    reference_keys = None
    for seed in SEEDS:
        seed_rows = sorted(by_seed[seed], key=lambda row: row["invocation_order_index"])
        count = len(seed_rows)
        if count < 200 or count % 6 != 0:
            raise ValueError(
                "Gate-4 timing population per seed must contain at least 200 rows and be a multiple of six"
            )
        if [row["invocation_order_index"] for row in seed_rows] != list(range(count)):
            raise ValueError("Gate-4 timing order indices must be contiguous and unique")
        keys = []
        seen = set()
        for index, row in enumerate(seed_rows):
            if row["arm_order"] != ARM_ORDERS[index % 6]:
                raise ValueError("Gate-4 timing six-order crossover sequence mismatch")
            key = (
                row["official_video_id"],
                row["invocation_id"],
                row["repetition_id"],
            )
            if key in seen:
                raise ValueError("Gate-4 timing contains a duplicate matched invocation block")
            seen.add(key)
            keys.append(key)
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError("Gate-4 timing seeds must use the same complete matched population")
        by_seed[seed] = seed_rows
    return by_seed


def _timing_statistics(by_seed, *, bootstrap_samples: int, bootstrap_seed: int):
    videos = sorted({row["official_video_id"] for row in by_seed[SEEDS[0]]})
    by_seed_video = {
        seed: {
            video: [row for row in by_seed[seed] if row["official_video_id"] == video]
            for video in videos
        }
        for seed in SEEDS
    }

    def summarize(rows):
        totals = {arm: [row["arms"][arm]["total_ms"] for row in rows] for arm in ARMS}
        p50 = {arm: _p50(values) for arm, values in totals.items()}
        if p50["dense"] <= 0.0:
            raise ValueError("Gate-4 dense p50 must be positive")
        heavy = [
            row["arms"]["dense"]["stage_ms"]["heavy_ms"]
            - row["arms"]["chronotransport"]["stage_ms"]["heavy_ms"]
            for row in rows
        ]
        margins = []
        for row, saving in zip(rows, heavy):
            ct = row["arms"]["chronotransport"]
            overhead = sum(
                ct["stage_ms"][field]
                for field in (
                    "innovation_ms",
                    "scheduler_ms",
                    "transport_ms",
                    "cache_movement_ms",
                )
            )
            margins.append(0.40 * saving - overhead)
        return {
            "p50": p50,
            "latency_saving": (p50["dense"] - p50["chronotransport"]) / p50["dense"],
            "ct_minus_static_ms": p50["chronotransport"] - p50["static"],
            "median_heavy_saving_ms": _p50(heavy),
            "median_margin_ms": _p50(margins),
        }

    pooled_rows = [row for seed in SEEDS for row in by_seed[seed]]
    point = summarize(pooled_rows)
    per_seed = {str(seed): summarize(by_seed[seed]) for seed in SEEDS}
    diagnostics = {
        "p95_ms": {
            arm: _percentile(
                [row["arms"][arm]["total_ms"] for row in pooled_rows], 95.0
            )
            for arm in ARMS
        },
        "peak_gpu_memory_bytes": {
            arm: max(row["arms"][arm]["peak_gpu_memory_bytes"] for row in pooled_rows)
            for arm in ARMS
        },
        "median_nvml_block_energy_j": {
            arm: _p50([row["arms"][arm]["nvml_energy_j"] for row in pooled_rows])
            for arm in ARMS
        },
        "median_stage_ms": {
            arm: {
                field: _p50(
                    [row["arms"][arm]["stage_ms"][field] for row in pooled_rows]
                )
                for field in sorted(_STAGE_FIELDS)
            }
            for arm in ARMS
        },
        "total_ms_distribution": {},
    }
    for arm in ARMS:
        samples = [row["arms"][arm]["total_ms"] for row in pooled_rows]
        diagnostics["total_ms_distribution"][arm] = {
            "count": len(samples),
            "min": min(samples),
            "max": max(samples),
            "p50": _p50(samples),
            "p95": _percentile(samples, 95.0),
            "sample_sha256": canonical_sha256(samples),
        }
    diagnostics["throughput_per_second"] = {
        arm: 1000.0 / point["p50"][arm] for arm in ARMS
    }
    rng = random.Random(int(bootstrap_seed) ^ 0x4A17)
    saving_boot, static_boot, margin_boot = [], [], []
    for _ in range(bootstrap_samples):
        sampled_videos = [rng.choice(videos) for _ in videos]
        sampled_seeds = [rng.choice(SEEDS) for _ in SEEDS]
        sampled_rows = []
        for video in sampled_videos:
            for seed in sampled_seeds:
                blocks = by_seed_video[seed][video]
                sampled_rows.extend(rng.choice(blocks) for _ in blocks)
        value = summarize(sampled_rows)
        saving_boot.append(value["latency_saving"])
        static_boot.append(value["ct_minus_static_ms"])
        margin_boot.append(value["median_margin_ms"])
    return {
        "videos": videos,
        "point": point,
        "per_seed": per_seed,
        "saving_lcb95": _percentile(saving_boot, 5.0),
        "ct_minus_static_ucb95_ms": _percentile(static_boot, 95.0),
        "margin_lcb95_ms": _percentile(margin_boot, 5.0),
        "diagnostics": diagnostics,
    }


def _segment(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be a two-value list")
    start = _finite(value[0], f"{label} start")
    end = _finite(value[1], f"{label} end")
    if end <= start:
        raise ValueError(f"{label} must have positive duration")
    return start, end


def _normalize_metrics(value: Mapping[str, Any], *, expected_videos: Sequence[str]):
    expected = {
        "schema",
        "fit_duration_quartile_thresholds",
        "ground_truth",
        "predictions",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Gate-4 metric evidence fields mismatch")
    if value["schema"] != "chronotransport-r2-gate4-metric-evidence-v1":
        raise ValueError("Gate-4 metric evidence schema mismatch")
    quartiles = value["fit_duration_quartile_thresholds"]
    if not isinstance(quartiles, list) or len(quartiles) != 3:
        raise ValueError("Gate-4 fit duration quartiles require Q1/Q2/Q3")
    quartiles = tuple(
        _finite(item, f"fit duration Q{index}", positive=True)
        for index, item in enumerate(quartiles, 1)
    )
    if not quartiles[0] < quartiles[1] < quartiles[2]:
        raise ValueError("Gate-4 fit duration quartiles must be strictly increasing")
    gt_rows = []
    seen_gt = set()
    for index, raw in enumerate(value["ground_truth"]):
        if not isinstance(raw, Mapping) or set(raw) != {
            "official_video_id",
            "label",
            "segment",
        }:
            raise ValueError(f"Gate-4 ground truth row {index} fields mismatch")
        row = (
            _text(raw["official_video_id"], "metric video ID"),
            _text(raw["label"], "metric label"),
            *_segment(raw["segment"], "ground truth segment"),
        )
        if row in seen_gt:
            raise ValueError("Gate-4 ground truth contains a duplicate instance")
        seen_gt.add(row)
        gt_rows.append(row)
    videos = sorted({row[0] for row in gt_rows})
    if videos != list(expected_videos):
        raise ValueError("Gate-4 metric ground truth differs from official timing videos")
    labels = {row[1] for row in gt_rows}
    predictions = value["predictions"]
    if not isinstance(predictions, Mapping) or set(predictions) != {str(seed) for seed in SEEDS}:
        raise ValueError("Gate-4 metric predictions require exactly three seeds")
    normalized_predictions = {}
    for seed in SEEDS:
        arms = predictions[str(seed)]
        if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
            raise ValueError("Gate-4 metric predictions require D/C/S arms")
        normalized_predictions[seed] = {}
        for arm in ARMS:
            rows, seen = [], set()
            if not isinstance(arms[arm], list):
                raise TypeError("Gate-4 arm predictions must be a list")
            for index, raw in enumerate(arms[arm]):
                if not isinstance(raw, Mapping) or set(raw) != {
                    "official_video_id",
                    "label",
                    "segment",
                    "score",
                }:
                    raise ValueError(f"Gate-4 prediction row {index} fields mismatch")
                video = _text(raw["official_video_id"], "prediction video ID")
                label = _text(raw["label"], "prediction label")
                if video not in videos or label not in labels:
                    raise ValueError("Gate-4 prediction is outside the official GT population")
                start, end = _segment(raw["segment"], "prediction segment")
                score = _finite(raw["score"], "prediction score")
                row = (video, label, start, end, score)
                if row in seen:
                    raise ValueError("Gate-4 predictions contain a duplicate row")
                seen.add(row)
                rows.append(row)
            normalized_predictions[seed][arm] = rows
    return quartiles, gt_rows, normalized_predictions


def _map_at(
    gt_rows,
    prediction_rows,
    sampled_videos: Sequence[str],
    *,
    tiou_threshold: float = 0.7,
    q1_threshold: float | None = None,
    duration_lower: float | None = None,
) -> float:
    gt_by_source = defaultdict(list)
    pred_by_source = defaultdict(list)
    for row in gt_rows:
        duration = row[3] - row[2]
        if (duration_lower is None or duration > duration_lower) and (
            q1_threshold is None or duration <= q1_threshold
        ):
            gt_by_source[row[0]].append(row)
    for row in prediction_rows:
        pred_by_source[row[0]].append(row)
    synthetic_gt, synthetic_pred = [], []
    for position, source in enumerate(sampled_videos):
        synthetic = f"boot/{position}/{source}"
        synthetic_gt.extend((synthetic, *row[1:]) for row in gt_by_source[source])
        synthetic_pred.extend((synthetic, *row[1:]) for row in pred_by_source[source])
    labels = sorted({row[1] for row in synthetic_gt})
    if not labels:
        raise ValueError("Gate-4 metric bootstrap has no ground truth")
    aps = []
    for label in labels:
        class_gt = [row for row in synthetic_gt if row[1] == label]
        class_pred = [row for row in synthetic_pred if row[1] == label]
        official_gt = pd.DataFrame(
            [
                {"video-id": row[0], "t-start": row[2], "t-end": row[3]}
                for row in class_gt
            ],
            columns=("video-id", "t-start", "t-end"),
        )
        official_predictions = pd.DataFrame(
            [
                {
                    "video-id": row[0],
                    "t-start": row[2],
                    "t-end": row[3],
                    "score": row[4],
                }
                for row in class_pred
            ],
            columns=("video-id", "t-start", "t-end", "score"),
        )
        aps.append(
            float(
                compute_average_precision_detection(
                    official_gt,
                    official_predictions,
                    tiou_thresholds=np.asarray([tiou_threshold], dtype=np.float64),
                )[0]
            )
        )
    return float(np.mean(aps))


def _metric_statistics(
    quartiles,
    gt_rows,
    predictions,
    videos,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
):
    q1 = quartiles[0]
    per_seed = {}
    for seed in SEEDS:
        dense = _map_at(gt_rows, predictions[seed]["dense"], videos)
        ct = _map_at(gt_rows, predictions[seed]["chronotransport"], videos)
        dense_q1 = _map_at(gt_rows, predictions[seed]["dense"], videos, q1_threshold=q1)
        ct_q1 = _map_at(
            gt_rows, predictions[seed]["chronotransport"], videos, q1_threshold=q1
        )
        per_seed[str(seed)] = {
            "dense_map07": dense,
            "chronotransport_map07": ct,
            "map07_drop_points": 100.0 * (dense - ct),
            "dense_short_q1_map07": dense_q1,
            "chronotransport_short_q1_map07": ct_q1,
            "short_q1_drop_points": 100.0 * (dense_q1 - ct_q1),
        }
    point_drop = float(np.mean([row["map07_drop_points"] for row in per_seed.values()]))
    point_q1 = float(np.mean([row["short_q1_drop_points"] for row in per_seed.values()]))
    rng = random.Random(int(bootstrap_seed) ^ 0x7C31)
    drops, q1_drops = [], []
    for _ in range(bootstrap_samples):
        sampled_videos = [rng.choice(videos) for _ in videos]
        sampled_seeds = [rng.choice(SEEDS) for _ in SEEDS]
        dense = np.mean(
            [_map_at(gt_rows, predictions[seed]["dense"], sampled_videos) for seed in sampled_seeds]
        )
        ct = np.mean(
            [
                _map_at(gt_rows, predictions[seed]["chronotransport"], sampled_videos)
                for seed in sampled_seeds
            ]
        )
        dense_q1 = np.mean(
            [
                _map_at(
                    gt_rows,
                    predictions[seed]["dense"],
                    sampled_videos,
                    q1_threshold=q1,
                )
                for seed in sampled_seeds
            ]
        )
        ct_q1 = np.mean(
            [
                _map_at(
                    gt_rows,
                    predictions[seed]["chronotransport"],
                    sampled_videos,
                    q1_threshold=q1,
                )
                for seed in sampled_seeds
            ]
        )
        drops.append(100.0 * float(dense - ct))
        q1_drops.append(100.0 * float(dense_q1 - ct_q1))
    map_by_tiou = {}
    average_map = {}
    duration_quartile_map07 = {}
    bounds = (
        (None, quartiles[0]),
        (quartiles[0], quartiles[1]),
        (quartiles[1], quartiles[2]),
        (quartiles[2], None),
    )
    for arm in ARMS:
        by_threshold = {}
        for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
            by_threshold[f"{threshold:.1f}"] = float(
                np.mean(
                    [
                        _map_at(
                            gt_rows,
                            predictions[seed][arm],
                            videos,
                            tiou_threshold=threshold,
                        )
                        for seed in SEEDS
                    ]
                )
            )
        map_by_tiou[arm] = by_threshold
        average_map[arm] = float(np.mean(list(by_threshold.values())))
        quartile_values = []
        for lower, upper in bounds:
            try:
                values = [
                    _map_at(
                        gt_rows,
                        predictions[seed][arm],
                        videos,
                        tiou_threshold=0.7,
                        q1_threshold=upper,
                        duration_lower=lower,
                    )
                    for seed in SEEDS
                ]
            except ValueError as error:
                if "no ground truth" not in str(error):
                    raise
                quartile_values.append(None)
            else:
                quartile_values.append(float(np.mean(values)))
        duration_quartile_map07[arm] = quartile_values
    return {
        "per_seed": per_seed,
        "point_drop": point_drop,
        "point_q1_drop": point_q1,
        "drop_ucb95": _percentile(drops, 95.0),
        "q1_drop_ucb95": _percentile(q1_drops, 95.0),
        "map_by_tiou": map_by_tiou,
        "average_map": average_map,
        "duration_quartile_map07": duration_quartile_map07,
    }


def _normalize_regret(rows, *, timing_by_seed):
    by_key = {}
    timing_invocations = {
        seed: {
            (row["official_video_id"], row["invocation_id"])
            for row in timing_by_seed[seed]
        }
        for seed in SEEDS
    }
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != _REGRET_FIELDS:
            raise ValueError(f"Gate-4 regret row {index} fields mismatch")
        seed = _integer(raw["seed"], "regret seed")
        if seed not in SEEDS:
            raise ValueError("Gate-4 regret seed mismatch")
        key = (
            seed,
            _text(raw["official_video_id"], "regret video ID"),
            _text(raw["invocation_id"], "regret invocation ID"),
        )
        if key in by_key:
            raise ValueError("Gate-4 regret contains a duplicate invocation")
        by_key[key] = {
            "dense": _finite(raw["dense_detector_loss"], "dense detector loss"),
            "ct": _finite(
                raw["chronotransport_detector_loss"], "ChronoTransport detector loss"
            ),
            "static": _finite(raw["static_detector_loss"], "static detector loss"),
        }
    for seed in SEEDS:
        actual = {(video, invocation) for s, video, invocation in by_key if s == seed}
        if actual != timing_invocations[seed]:
            raise ValueError(
                "Gate-4 regret population must contain every unique official invocation once"
            )
    return by_key


def _regret_statistics(by_key, videos, *, bootstrap_samples: int, bootstrap_seed: int):
    invocations = defaultdict(list)
    for seed, video, invocation in by_key:
        invocations[(seed, video)].append(invocation)

    def mean_improvement(sampled_videos, sampled_seeds):
        values = []
        for video in sampled_videos:
            for seed in sampled_seeds:
                for invocation in invocations[(seed, video)]:
                    row = by_key[(seed, video, invocation)]
                    r_ct = max(row["ct"] - row["dense"], 0.0)
                    r_static = max(row["static"] - row["dense"], 0.0)
                    values.append(r_static - r_ct)
        return float(np.mean(values))

    point = mean_improvement(videos, SEEDS)
    per_seed = {str(seed): mean_improvement(videos, [seed]) for seed in SEEDS}
    rng = random.Random(int(bootstrap_seed) ^ 0x19D3)
    boot = []
    for _ in range(bootstrap_samples):
        sampled_videos = [rng.choice(videos) for _ in videos]
        sampled_seeds = [rng.choice(SEEDS) for _ in SEEDS]
        boot.append(mean_improvement(sampled_videos, sampled_seeds))
    return point, per_seed, (_percentile(boot, 2.5), _percentile(boot, 97.5))


def adjudicate_gate4(
    *,
    timing_rows: Sequence[Mapping[str, Any]],
    metric_evidence: Mapping[str, Any],
    regret_rows: Sequence[Mapping[str, Any]],
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20260711,
    formal: bool = True,
) -> dict[str, Any]:
    """Exercise Gate-4 statistics on unregistered, test-only raw evidence.

    A real Gate-4 artifact must be minted by a repository-owned evidence
    producer that binds the official invocation population, model/checkpoint
    identities, post-Stage-C Gate-3 unlock, clean detached registration R, and
    immutable timing/metric/regret artifacts.  That production workflow does
    not exist yet, so ``formal=True`` intentionally fails closed instead of
    allowing caller-supplied mappings to impersonate formal evidence.
    """

    if type(formal) is not bool:
        raise TypeError("Gate-4 formal flag must be boolean")
    bootstrap_samples = _integer(bootstrap_samples, "Gate-4 bootstrap samples")
    bootstrap_seed = _integer(bootstrap_seed, "Gate-4 bootstrap seed")
    if formal and bootstrap_samples != 5000:
        raise ValueError("formal Gate 4 requires exactly 5000 bootstrap samples")
    if formal and bootstrap_seed != 20260711:
        raise ValueError("formal Gate 4 requires bootstrap seed 20260711")
    if formal:
        raise RuntimeError(
            "formal Gate 4 requires a registered evidence producer; "
            "caller-supplied raw mappings are test-only"
        )
    if bootstrap_samples <= 0:
        raise ValueError("Gate-4 bootstrap samples must be positive")

    timing = _normalize_timing(timing_rows)
    timing_stats = _timing_statistics(
        timing, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed
    )
    quartiles, gt, predictions = _normalize_metrics(
        metric_evidence, expected_videos=timing_stats["videos"]
    )
    metric_stats = _metric_statistics(
        quartiles,
        gt,
        predictions,
        timing_stats["videos"],
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    regret = _normalize_regret(regret_rows, timing_by_seed=timing)
    regret_point, regret_per_seed, regret_ci = _regret_statistics(
        regret,
        timing_stats["videos"],
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )

    per_seed = {}
    for seed in SEEDS:
        latency = timing_stats["per_seed"][str(seed)]
        metrics = metric_stats["per_seed"][str(seed)]
        per_seed[str(seed)] = {
            "latency_saving": latency["latency_saving"],
            "map07_drop_points": metrics["map07_drop_points"],
            "short_q1_drop_points": metrics["short_q1_drop_points"],
            "median_margin_ms": latency["median_margin_ms"],
            "ct_minus_static_ms": latency["ct_minus_static_ms"],
            "ct_over_static_regret_improvement": regret_per_seed[str(seed)],
        }
    every_seed = all(
        row["latency_saving"] >= 0.15
        and row["map07_drop_points"] <= 1.5
        and row["short_q1_drop_points"] <= 1.5
        and row["median_margin_ms"] > 0.0
        and row["ct_minus_static_ms"] <= 0.0
        for row in per_seed.values()
    )
    hard = {
        "latency_saving_lcb_ge_0p15": timing_stats["saving_lcb95"] >= 0.15,
        "map07_drop_ucb_le_1p5_points": metric_stats["drop_ucb95"] <= 1.5,
        "short_q1_drop_ucb_le_1p5_points": metric_stats["q1_drop_ucb95"] <= 1.5,
        "median_heavy_saving_gt_0": timing_stats["point"]["median_heavy_saving_ms"]
        > 0.0,
        "median_margin_lcb_gt_0": timing_stats["margin_lcb95_ms"] > 0.0,
        "ct_minus_static_ucb_le_0": timing_stats["ct_minus_static_ucb95_ms"] <= 0.0,
        "ct_over_static_regret_ci_lower_gt_0": regret_ci[0] > 0.0,
        "every_seed_within_thresholds": every_seed,
    }
    result = {
        "schema": "chronotransport-r2-gate4-test-only-v1",
        "protocol": "CT-P3R-3S-r2",
        "evidence_scope": "test_only_unregistered_raw_mappings",
        "formal_evidence": False,
        "status": "PASS" if all(hard.values()) else "FAIL",
        "mechanism": bool(all(hard.values())),
        "bootstrap": {"samples": bootstrap_samples, "seed": bootstrap_seed},
        "timing": {
            "matched_rows_per_seed": len(timing[SEEDS[0]]),
            "official_video_count": len(timing_stats["videos"]),
            "six_order_crossover": True,
            "bootstrap_unit": "official_video_then_matched_block_then_seed",
        },
        "latency": {
            "dense_p50_ms": timing_stats["point"]["p50"]["dense"],
            "chronotransport_p50_ms": timing_stats["point"]["p50"][
                "chronotransport"
            ],
            "static_p50_ms": timing_stats["point"]["p50"]["static"],
            "saving": timing_stats["point"]["latency_saving"],
            "saving_lcb95": timing_stats["saving_lcb95"],
            "ct_minus_static_ms": timing_stats["point"]["ct_minus_static_ms"],
            "ct_minus_static_ucb95_ms": timing_stats["ct_minus_static_ucb95_ms"],
        },
        "metrics": {
            "official_video_count": len(timing_stats["videos"]),
            "bootstrap_unit": "official_video_then_seed",
            "map07_drop_points": metric_stats["point_drop"],
            "map07_drop_ucb95_points": metric_stats["drop_ucb95"],
            "short_q1_drop_points": metric_stats["point_q1_drop"],
            "short_q1_drop_ucb95_points": metric_stats["q1_drop_ucb95"],
            "fit_duration_quartile_thresholds": list(quartiles),
        },
        "cost_decomposition": {
            "median_heavy_saving_ms": timing_stats["point"][
                "median_heavy_saving_ms"
            ],
            "median_margin_ms": timing_stats["point"]["median_margin_ms"],
            "median_margin_lcb95_ms": timing_stats["margin_lcb95_ms"],
        },
        "regret": {
            "unique_invocations_per_seed": len(regret) // len(SEEDS),
            "bootstrap_unit": "official_video_then_seed",
            "ct_over_static_improvement": regret_point,
            "ct_over_static_improvement_ci95": list(regret_ci),
        },
        "diagnostics": {
            **timing_stats["diagnostics"],
            "map_by_tiou": metric_stats["map_by_tiou"],
            "average_map": metric_stats["average_map"],
            "duration_quartile_map07": metric_stats["duration_quartile_map07"],
            "energy_semantics": "10Hz_NVML_long_timed_block_trapezoidal_only",
        },
        "per_seed": per_seed,
        "hard_conditions": hard,
        "deploy_claim_allowed": False,
        "paper_claim_allowed": False,
    }
    result["artifact_sha256"] = canonical_sha256(result)
    return result


def validate_gate4_report(
    report: Mapping[str, Any],
    *,
    timing_rows: Sequence[Mapping[str, Any]],
    metric_evidence: Mapping[str, Any],
    regret_rows: Sequence[Mapping[str, Any]],
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20260711,
    formal: bool = True,
) -> dict[str, Any]:
    """Recompute the test-only Gate-4 statistic artifact and reject tamper.

    ``formal=True`` is intentionally locked for the same reason as
    :func:`adjudicate_gate4`.
    """

    expected = adjudicate_gate4(
        timing_rows=timing_rows,
        metric_evidence=metric_evidence,
        regret_rows=regret_rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        formal=formal,
    )
    if not isinstance(report, Mapping):
        raise TypeError("Gate-4 report must be a mapping")

    candidate = dict(report)
    claimed_sha256 = candidate.pop("artifact_sha256", None)
    if claimed_sha256 != canonical_sha256(candidate):
        raise ValueError("Gate-4 report artifact hash does not match its payload")
    if dict(report) != expected:
        raise ValueError("Gate-4 report does not match recomputed raw evidence")
    return expected


__all__ = ["adjudicate_gate4", "validate_gate4_report"]
