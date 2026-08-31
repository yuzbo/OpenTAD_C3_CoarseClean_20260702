"""One-time official held-out evaluation for the frozen H65 exposure study."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.evaluations import build_evaluator
from opentad.evaluations.mAP import segment_iou
from tools.bata.prepare_duca_h65_system_multibudget_exposure import sha256_file


TIOU_THRESHOLDS = np.asarray((0.3, 0.4, 0.5, 0.6, 0.7), dtype=np.float64)
SEEDS = (3407, 3408, 3409)
VIEWS = ("control_k384", "candidate_k384", "candidate_mixed")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _order_interval(values: np.ndarray) -> dict[str, float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.shape != (10_000,):
        raise ValueError("the frozen interval requires exactly 10,000 replicates")
    return {
        "lower_pp": float(ordered[249]),
        "upper_pp": float(ordered[9749]),
        "positive_fraction": float(np.mean(ordered > 0.0)),
        "order_statistics_one_based": [250, 9750],
    }


def _counts_from_indices(indices: np.ndarray, video_count: int) -> np.ndarray:
    if indices.shape != (10_000, video_count):
        raise ValueError("bootstrap index shape differs from the frozen population")
    counts = np.zeros((indices.shape[0], video_count), dtype=np.int16)
    rows = np.repeat(np.arange(indices.shape[0], dtype=np.int64), video_count)
    np.add.at(counts, (rows, indices.reshape(-1).astype(np.int64)), 1)
    if not np.all(counts.sum(axis=1) == video_count):
        raise ValueError("bootstrap count reconstruction changed the sample size")
    return counts


class OfficialAPBootstrapArm:
    """Point-parity bootstrap accelerator for the unchanged OpenTAD AP rule."""

    def __init__(self, evaluator, video_ids: list[str]):
        self.evaluator = evaluator
        self.video_ids = tuple(video_ids)
        self.video_index = {video_id: index for index, video_id in enumerate(video_ids)}
        self.class_records = []
        for class_index in evaluator.activity_index.values():
            ground_truth = evaluator.ground_truth[
                evaluator.ground_truth["label"] == class_index
            ].reset_index(drop=True)
            prediction = evaluator.prediction[
                evaluator.prediction["label"] == class_index
            ].reset_index(drop=True)
            self.class_records.append(
                self._prepare_class(ground_truth, prediction, len(video_ids))
            )

    def _prepare_class(self, ground_truth, prediction, video_count):
        gt_segments = {}
        gt_counts = np.zeros(video_count, dtype=np.int32)
        for video_id, group in ground_truth.groupby("video-id", sort=False):
            if video_id not in self.video_index:
                raise ValueError("official ground truth leaves the frozen video population")
            values = group[["t-start", "t-end"]].to_numpy(dtype=np.float64)
            gt_segments[str(video_id)] = values
            gt_counts[self.video_index[str(video_id)]] = int(len(values))
        sort_index = prediction["score"].to_numpy().argsort()[::-1]
        prediction = prediction.loc[sort_index].reset_index(drop=True)
        prediction_video = np.asarray(
            [self.video_index[str(value)] for value in prediction["video-id"]],
            dtype=np.int16,
        )
        true_positive = np.zeros(
            (len(TIOU_THRESHOLDS), len(prediction)), dtype=np.bool_
        )
        locks = {
            video_id: np.zeros((len(TIOU_THRESHOLDS), len(values)), dtype=np.bool_)
            for video_id, values in gt_segments.items()
        }
        for prediction_index, row in prediction.iterrows():
            video_id = str(row["video-id"])
            values = gt_segments.get(video_id)
            if values is None or len(values) == 0:
                continue
            overlaps = segment_iou(
                np.asarray((float(row["t-start"]), float(row["t-end"])), dtype=np.float64),
                values,
            )
            order = overlaps.argsort()[::-1]
            for threshold_index, threshold in enumerate(TIOU_THRESHOLDS):
                for gt_index in order:
                    if overlaps[gt_index] < threshold:
                        break
                    if locks[video_id][threshold_index, gt_index]:
                        continue
                    locks[video_id][threshold_index, gt_index] = True
                    true_positive[threshold_index, prediction_index] = True
                    break
        return {
            "gt_counts": gt_counts,
            "prediction_video": prediction_video,
            "true_positive": true_positive,
        }

    @staticmethod
    def _class_ap_for_counts(record, counts, *, chunk_size=32):
        gt_counts = record["gt_counts"].astype(np.int64, copy=False)
        prediction_video = record["prediction_video"].astype(np.int64, copy=False)
        true_positive = record["true_positive"]
        replicate_count = int(counts.shape[0])
        result = np.zeros((replicate_count, len(TIOU_THRESHOLDS)), dtype=np.float64)
        positive_count = counts.astype(np.int64) @ gt_counts
        for start in range(0, replicate_count, int(chunk_size)):
            stop = min(start + int(chunk_size), replicate_count)
            sample_counts = counts[start:stop].astype(np.int64, copy=False)
            if prediction_video.size:
                weights = sample_counts[:, prediction_video]
                cumulative_predictions = np.cumsum(weights, axis=1, dtype=np.int64)
            else:
                weights = np.zeros((stop - start, 0), dtype=np.int64)
                cumulative_predictions = weights
            for threshold_index in range(len(TIOU_THRESHOLDS)):
                tp = true_positive[threshold_index]
                events = np.flatnonzero(tp)
                if events.size == 0:
                    continue
                last_event = int(events[-1])
                events = events[events <= last_event]
                local_weights = weights[:, : last_event + 1]
                weighted_tp = local_weights * tp[: last_event + 1][None, :]
                cumulative_tp = np.cumsum(weighted_tp, axis=1, dtype=np.int64)
                event_weight = local_weights[:, events]
                max_weight = int(event_weight.max(initial=0))
                if max_weight == 0:
                    continue
                tp_before = cumulative_tp[:, events] - event_weight
                prediction_before = cumulative_predictions[:, events] - event_weight
                copies = np.arange(1, max_weight + 1, dtype=np.int64)[None, None, :]
                valid = copies <= event_weight[:, :, None]
                precision = (tp_before[:, :, None] + copies) / np.maximum(
                    prediction_before[:, :, None] + copies, 1
                )
                flattened_precision = np.where(valid, precision, -np.inf).reshape(
                    stop - start, -1
                )
                flattened_valid = valid.reshape(stop - start, -1)
                envelope = np.maximum.accumulate(
                    flattened_precision[:, ::-1], axis=1
                )[:, ::-1]
                envelope[~np.isfinite(envelope)] = 0.0
                numerator = np.where(flattened_valid, envelope, 0.0).sum(axis=1)
                denominator = positive_count[start:stop]
                active = denominator > 0
                values = np.zeros(stop - start, dtype=np.float64)
                values[active] = numerator[active] / denominator[active]
                result[start:stop, threshold_index] = values
        return result, positive_count

    def metrics_for_counts(self, counts: np.ndarray) -> np.ndarray:
        sums = np.zeros((counts.shape[0], len(TIOU_THRESHOLDS)), dtype=np.float64)
        for record in self.class_records:
            values, positive_count = self._class_ap_for_counts(record, counts)
            if np.any(positive_count <= 0):
                raise RuntimeError(
                    "a whole-video bootstrap replicate omits an official activity class"
                )
            sums += values
        if not self.class_records:
            raise RuntimeError("official evaluator contains no activity classes")
        return 100.0 * sums / len(self.class_records)


def _load_seals(
    paths: list[Path],
    expected_commit: str,
    *,
    calibration_sha256: str,
    held_out_ids_sha256: str,
    inference_annotation_sha256: str,
):
    records = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "duca_h65_system_multibudget_prediction_seal_v1":
            raise SystemExit(f"invalid prediction seal schema: {path}")
        if payload.get("git_commit") != expected_commit:
            raise SystemExit("prediction seals do not share the frozen commit")
        if (
            payload.get("calibration_sha256") != calibration_sha256
            or payload.get("held_out_video_ids_sha256") != held_out_ids_sha256
            or payload.get("held_out_inference_annotation_sha256")
            != inference_annotation_sha256
        ):
            raise SystemExit("prediction seals do not share the frozen held-out inputs")
        seed = int(payload["seed"])
        view = str(payload["budget_view"])
        key = (seed, view)
        if seed not in SEEDS or view not in VIEWS or key in records:
            raise SystemExit("prediction seal matrix contains a duplicate or unexpected entry")
        expected_arm = "control" if view == "control_k384" else "candidate"
        if payload.get("arm") != expected_arm:
            raise SystemExit("prediction seal arm and budget view disagree")
        prediction = Path(payload["prediction_path"])
        cost = Path(payload["execution_cost_path"])
        if (
            not prediction.is_file()
            or sha256_file(prediction) != payload["prediction_sha256"]
            or not cost.is_file()
            or sha256_file(cost) != payload["execution_cost_sha256"]
        ):
            raise SystemExit("sealed prediction or cost SHA256 mismatch")
        records[key] = {"seal": payload, "prediction": prediction, "cost": cost}
    expected = {(seed, view) for seed in SEEDS for view in VIEWS}
    if set(records) != expected:
        raise SystemExit("one-time evaluation requires all nine sealed prediction views")
    return records


def _initialize_official_evaluator(annotation: Path, prediction: Path):
    prediction_payload = json.loads(prediction.read_text(encoding="utf-8"))
    return build_evaluator(
        dict(
            type="mAP",
            ground_truth_filename=str(annotation),
            prediction_filename=prediction_payload,
            subset="validation",
            tiou_thresholds=TIOU_THRESHOLDS.tolist(),
            thread=16,
        )
    )


def _official_point(evaluator, prediction: Path):
    prediction_payload = json.loads(prediction.read_text(encoding="utf-8"))
    evaluator.prediction = evaluator._import_prediction(prediction_payload)
    metrics = evaluator.evaluate()
    return {key: float(value) for key, value in metrics.items()}


def _metric_vector(metrics: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [100.0 * metrics[f"mAP@{threshold:.1f}"] for threshold in TIOU_THRESHOLDS],
        dtype=np.float64,
    )


def _seed_summary(points, view):
    vectors = np.stack([points[(seed, view)] for seed in SEEDS], axis=0)
    return {
        "per_seed_percent": {
            str(seed): {
                "average_mAP": float(np.mean(vectors[index])),
                **{
                    f"mAP@{threshold:.1f}": float(vectors[index, threshold_index])
                    for threshold_index, threshold in enumerate(TIOU_THRESHOLDS)
                },
            }
            for index, seed in enumerate(SEEDS)
        },
        "three_seed_mean_percent": {
            "average_mAP": float(np.mean(vectors)),
            **{
                f"mAP@{threshold:.1f}": float(np.mean(vectors[:, threshold_index]))
                for threshold_index, threshold in enumerate(TIOU_THRESHOLDS)
            },
        },
        "three_seed_std_percent": {
            "average_mAP": float(np.std(vectors.mean(axis=1), ddof=1)),
            **{
                f"mAP@{threshold:.1f}": float(np.std(vectors[:, threshold_index], ddof=1))
                for threshold_index, threshold in enumerate(TIOU_THRESHOLDS)
            },
        },
    }


def evaluate(args) -> dict[str, Any]:
    annotation = args.annotation.expanduser().resolve()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if sha256_file(annotation) != calibration.get("annotation_sha256"):
        raise SystemExit("official held-out annotation differs from PRE_RUN")
    ids = [
        line for line in args.held_out_ids.read_text(encoding="utf-8").splitlines() if line
    ]
    if len(ids) != 211 or ids != sorted(ids):
        raise SystemExit("held-out ID file must be the frozen sorted 211-video population")
    if sha256_file(args.held_out_ids) != calibration.get("held_out_id_sha256"):
        raise SystemExit("held-out ID artifact differs from PRE_RUN")
    indices = np.load(args.bootstrap_indices, allow_pickle=False)
    if indices.dtype != np.uint16 or indices.shape != (10_000, 211):
        raise SystemExit("shared bootstrap indices must be uint16[10000,211]")
    if sha256_file(args.bootstrap_indices) != calibration["paired_bootstrap_indices"]["sha256"]:
        raise SystemExit("bootstrap indices differ from PRE_RUN")
    records = _load_seals(
        args.seal,
        args.expected_commit,
        calibration_sha256=sha256_file(args.calibration),
        held_out_ids_sha256=sha256_file(args.held_out_ids),
        inference_annotation_sha256=calibration[
            "held_out_inference_annotation"
        ]["sha256"],
    )
    counts = _counts_from_indices(indices, len(ids))
    first_prediction = records[(SEEDS[0], VIEWS[0])]["prediction"]
    evaluator = _initialize_official_evaluator(annotation, first_prediction)
    source_path = Path(inspect.getsourcefile(evaluator.__class__)).resolve()
    evaluator_source = {
        "module": evaluator.__class__.__module__,
        "class_name": evaluator.__class__.__qualname__,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
    }
    point_vectors = {}
    replicate_vectors = {}
    point_metrics = {}
    for seed in SEEDS:
        for view in VIEWS:
            key = (seed, view)
            metrics = _official_point(evaluator, records[key]["prediction"])
            point_metrics[key] = metrics
            point_vectors[key] = _metric_vector(metrics)
            bootstrap_arm = OfficialAPBootstrapArm(evaluator, ids)
            parity = bootstrap_arm.metrics_for_counts(
                np.ones((1, len(ids)), dtype=np.int16)
            )[0]
            if not np.allclose(parity, point_vectors[key], rtol=0.0, atol=1.0e-6):
                raise RuntimeError(f"bootstrap implementation lacks official point parity for {key}")
            replicate_vectors[key] = bootstrap_arm.metrics_for_counts(counts)

    deltas = {}
    for name, candidate_view in (
        ("candidate_k384_minus_control_k384", "candidate_k384"),
        ("candidate_mixed_minus_control_k384", "candidate_mixed"),
    ):
        per_seed_point = np.stack(
            [
                point_vectors[(seed, candidate_view)]
                - point_vectors[(seed, "control_k384")]
                for seed in SEEDS
            ],
            axis=0,
        )
        replicate_delta = np.mean(
            np.stack(
                [
                    replicate_vectors[(seed, candidate_view)]
                    - replicate_vectors[(seed, "control_k384")]
                    for seed in SEEDS
                ],
                axis=0,
            ),
            axis=0,
        )
        replicate_avg = replicate_delta.mean(axis=1)
        deltas[name] = {
            "per_seed_point_delta_pp": {
                str(seed): {
                    "average_mAP": float(per_seed_point[index].mean()),
                    **{
                        f"mAP@{threshold:.1f}": float(
                            per_seed_point[index, threshold_index]
                        )
                        for threshold_index, threshold in enumerate(TIOU_THRESHOLDS)
                    },
                }
                for index, seed in enumerate(SEEDS)
            },
            "three_seed_mean_point_delta_pp": {
                "average_mAP": float(per_seed_point.mean()),
                **{
                    f"mAP@{threshold:.1f}": float(
                        per_seed_point[:, threshold_index].mean()
                    )
                    for threshold_index, threshold in enumerate(TIOU_THRESHOLDS)
                },
            },
            "paired_whole_video_bootstrap": {
                "replicates": 10_000,
                "shared_indices_sha256": sha256_file(args.bootstrap_indices),
                "average_mAP": _order_interval(replicate_avg),
                "mAP@0.7": _order_interval(replicate_delta[:, -1]),
            },
        }

    costs = {}
    for key, record in records.items():
        cost = json.loads(record["cost"].read_text(encoding="utf-8"))
        costs[f"{key[0]}:{key[1]}"] = {
            field: cost[field]
            for field in (
                "total_actual_observations",
                "total_execution_slots",
                "measurement_scope",
                "per_video_wall_ms_p50",
                "per_video_wall_ms_p95",
                "per_video_component_ms",
                "full_population_wall_ms",
                "unattributed_framework_wall_ms",
                "peak_cuda_memory_mb",
            )
        }
    k384 = deltas["candidate_k384_minus_control_k384"]["three_seed_mean_point_delta_pp"]
    mixed = deltas["candidate_mixed_minus_control_k384"]
    mixed_point = mixed["three_seed_mean_point_delta_pp"]
    mixed_interval = mixed["paired_whole_video_bootstrap"]
    mixed_cost_ok = all(
        int(costs[f"{seed}:candidate_mixed"]["total_actual_observations"])
        <= int(costs[f"{seed}:control_k384"]["total_actual_observations"])
        for seed in SEEDS
    )
    k384_gate = k384["average_mAP"] >= -0.2 and k384["mAP@0.7"] >= -0.2
    mixed_gate = (
        mixed_point["average_mAP"] >= 0.8
        and mixed_point["mAP@0.7"] >= 1.0
        and mixed_interval["average_mAP"]["lower_pp"] > 0.0
        and mixed_interval["mAP@0.7"]["lower_pp"] > 0.0
        and mixed_cost_ok
    )
    report = {
        "schema_version": "duca_h65_system_multibudget_one_time_evaluation_v1",
        "git_commit": args.expected_commit,
        "held_out_population": {
            "subset": "validation",
            "video_count": 211,
            "annotation_path": str(annotation),
            "annotation_sha256": sha256_file(annotation),
        },
        "official_evaluator": evaluator_source,
        "views": {view: _seed_summary(point_vectors, view) for view in VIEWS},
        "contrasts": deltas,
        "costs": costs,
        "gates": {
            "k384_safety_pass": bool(k384_gate),
            "mixed_primary_pass": bool(mixed_gate),
            "actual_observation_cost_pass": bool(mixed_cost_ok),
            "all_pass": bool(k384_gate and mixed_gate),
            "failure_means_stop_h65_multibudget_exposure_hypothesis": True,
        },
        "held_out_opened_after_all_predictions_sealed": True,
        "held_out_label_parse_count": 1,
        "predictions_regenerated_after_opening": False,
    }
    _atomic_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--held-out-ids", type=Path, required=True)
    parser.add_argument("--bootstrap-indices", type=Path, required=True)
    parser.add_argument("--seal", type=Path, action="append", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.calibration = args.calibration.expanduser().resolve()
    args.held_out_ids = args.held_out_ids.expanduser().resolve()
    args.bootstrap_indices = args.bootstrap_indices.expanduser().resolve()
    args.seal = [path.expanduser().resolve() for path in args.seal]
    args.output = args.output.expanduser().resolve()
    report = evaluate(args)
    print(
        "ONE_TIME_EVALUATION_COMPLETE "
        f"all_pass={report['gates']['all_pass']} output={args.output}"
    )


if __name__ == "__main__":
    main()
