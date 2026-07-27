from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from opentad.evaluations.mAP import mAP
from tools.bata.create_duca_rime_splits import TRAIN_ROLES, validate_rime_splits
from tools.bata.duca_p0_evaluation import (
    _metrics_from_evaluator,
    normalize_evaluation_config,
    prediction_results,
)
from tools.bata.duca_rime_training import (
    PHASE2_BASELINE_CHECKPOINT_COMPATIBILITY_MODE,
    PHASE2_BASELINE_IGNORED_UNEXPECTED_KEYS,
)


SCHEMA = "duca_rime_localization_metrics_v1"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _tiou(left: Sequence[float], right: Sequence[float]) -> float:
    start = max(float(left[0]), float(right[0]))
    end = min(float(left[1]), float(right[1]))
    intersection = max(0.0, end - start)
    union = max(float(left[1]), float(right[1])) - min(
        float(left[0]),
        float(right[0]),
    )
    return 0.0 if union <= 0.0 else intersection / union


def _official_metrics(
    *,
    database: Mapping[str, Any],
    predictions: Mapping[str, list[dict[str, Any]]],
    subset: str,
    ground_truth_path: Path,
) -> dict[str, float]:
    annotation_count = sum(
        len(row.get("annotations", ()))
        for row in database.values()
        if isinstance(row, Mapping)
    )
    if annotation_count == 0:
        return {
            "average_mAP": 0.0,
            **{
                f"mAP@{threshold:.1f}": 0.0
                for threshold in (0.3, 0.4, 0.5, 0.6, 0.7)
            },
        }
    ground_truth_path.write_text(
        json.dumps({"database": database}, sort_keys=True),
        encoding="utf-8",
    )
    evaluator = mAP(
        ground_truth_filename=str(ground_truth_path),
        prediction_filename={"results": dict(predictions)},
        subset=str(subset),
        tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
        top_k=None,
        blocked_videos=None,
        thread=1,
    )
    return _metrics_from_evaluator(evaluator)


def _duration_group(
    annotation: Mapping[str, Any],
    *,
    short_max_seconds: float,
    medium_max_seconds: float,
) -> str:
    segment = annotation["segment"]
    duration = float(segment[1]) - float(segment[0])
    if duration <= float(short_max_seconds):
        return "short"
    if duration <= float(medium_max_seconds):
        return "medium"
    return "long"


def _matching_metrics(
    annotations: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[float, float, int]:
    pair_support = []
    boundary_error = []
    for annotation in annotations:
        segment = annotation["segment"]
        label = str(annotation["label"])
        candidates = [
            row
            for row in predictions
            if str(row.get("label")) == label
            and isinstance(row.get("segment"), (list, tuple))
            and len(row["segment"]) == 2
        ]
        best = max(
            candidates,
            key=lambda row: (
                _tiou(segment, row["segment"]),
                float(row.get("score", 0.0)),
            ),
            default=None,
        )
        best_iou = 0.0 if best is None else _tiou(segment, best["segment"])
        pair_support.append(float(best_iou >= 0.7))
        duration = max(1.0e-12, float(segment[1]) - float(segment[0]))
        boundary_error.append(
            1.0
            if best is None
            else (
                abs(float(best["segment"][0]) - float(segment[0]))
                + abs(float(best["segment"][1]) - float(segment[1]))
            )
            / (2.0 * duration)
        )
    if not annotations:
        return 0.0, 0.0, 0
    return mean(pair_support), mean(boundary_error), len(annotations)


def evaluate_predictions(
    *,
    terminal_evaluation: str | Path,
    split_manifest: str | Path,
    split_manifest_sha256: str,
    phase: int,
    short_max_seconds: float,
    medium_max_seconds: float,
    output: str | Path,
    split_role: str | None = None,
) -> dict[str, Any]:
    if (
        int(phase) not in {2, 3, 4}
        or not 0.0 < float(short_max_seconds) < float(medium_max_seconds)
    ):
        raise ValueError("invalid RIME evaluation phase or duration thresholds")
    split_validation = validate_rime_splits(
        split_manifest,
        expected_sha256=split_manifest_sha256,
    )
    split = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    evaluation_path = Path(terminal_evaluation).expanduser().resolve()
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    terminal_schema = evaluation.get("schema_version")
    supported_schema = terminal_schema == "duca_rime_terminal_evaluation_v1"
    if (
        int(phase) == 2
        and terminal_schema
        == "duca_rime_phase2_baseline_terminal_evaluation_v1"
    ):
        baseline_contract = evaluation.get("baseline_contract")
        checkpoint_compatibility = evaluation.get("checkpoint_compatibility")
        supported_schema = (
            isinstance(baseline_contract, Mapping)
            and int(baseline_contract.get("phase", -1)) == 2
            and baseline_contract.get("uses_official_final") is False
            and baseline_contract.get("padded_to_kmax") is False
            and isinstance(checkpoint_compatibility, Mapping)
            and checkpoint_compatibility.get("mode")
            == PHASE2_BASELINE_CHECKPOINT_COMPATIBILITY_MODE
            and checkpoint_compatibility.get("missing_keys") == []
            and checkpoint_compatibility.get("ignored_unexpected_keys")
            == sorted(PHASE2_BASELINE_IGNORED_UNEXPECTED_KEYS)
            and evaluation.get("training_identity") is None
        )
    if (
        not supported_schema
        or evaluation.get("task") != "offline_temporal_action_detection"
        or evaluation.get("runtime_gt_input_to_selector") is not False
        or evaluation.get("padded_to_kmax") is not False
    ):
        raise ValueError("invalid RIME terminal evaluation")
    expected_subset = "training" if int(phase) in {2, 3} else "validation"
    cfg = normalize_evaluation_config(
        evaluation.get("evaluation_config"),
        expected_subset=expected_subset,
    )
    if int(phase) in {2, 3}:
        resolved_role = (
            "certification_development"
            if int(phase) == 3
            else str(split_role or "")
        )
        if resolved_role not in TRAIN_ROLES:
            raise ValueError("Phase-2 evaluation requires a registered train role")
        role = split["train_roles"][resolved_role]
        expected_videos = tuple(str(value) for value in role["videos"])
        if (
            cfg["blocked_videos"] is None
            or Path(cfg["blocked_videos"]).resolve()
            != Path(role["block_list_path"]).resolve()
            or _sha256_file(cfg["blocked_videos"]) != role["block_list_sha256"]
        ):
            raise ValueError(
                f"Phase-{phase} evaluator is not bound to the requested train role"
            )
        resolved_split_role = resolved_role
    else:
        final = split["official_final_evaluation"]
        expected_videos = tuple(str(value) for value in final["videos"])
        if cfg["blocked_videos"] is not None:
            raise ValueError("Phase-4 evaluator does not cover the full official final set")
        resolved_split_role = "official_final_evaluation"
    prediction_path = Path(evaluation["prediction_path"]).resolve()
    if (
        not prediction_path.is_file()
        or _sha256_file(prediction_path) != evaluation.get("prediction_sha256")
    ):
        raise ValueError("RIME terminal prediction artifact drifted")
    predictions = prediction_results(prediction_path)
    extras = set(predictions) - set(expected_videos)
    if extras:
        raise ValueError(f"RIME predictions contain out-of-scope videos: {sorted(extras)[:4]}")
    annotation_path = Path(cfg["ground_truth_filename"]).resolve()
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    database = annotation.get("database")
    if not isinstance(database, Mapping) or any(
        video not in database for video in expected_videos
    ):
        raise ValueError("RIME annotation does not cover the evaluation videos")

    video_metrics = {
        metric: {}
        for metric in (
            "avg_map",
            "map_0.6",
            "map_0.7",
            "short_map",
            "medium_map",
            "long_map",
            "pair_support",
            "boundary_error",
        )
    }
    duration_support = {}
    with tempfile.TemporaryDirectory(prefix="duca-rime-metrics-") as directory:
        ground_truth_path = Path(directory) / "ground_truth.json"
        for video in expected_videos:
            row = dict(database[video])
            annotations = [
                dict(value) for value in row.get("annotations", ())
            ]
            if not annotations:
                raise ValueError(f"RIME evaluation video has no annotations: {video}")
            video_predictions = [dict(value) for value in predictions.get(video, ())]
            full = _official_metrics(
                database={video: row},
                predictions={video: video_predictions},
                subset=expected_subset,
                ground_truth_path=ground_truth_path,
            )
            video_metrics["avg_map"][video] = float(full["average_mAP"])
            video_metrics["map_0.6"][video] = float(full["mAP@0.6"])
            video_metrics["map_0.7"][video] = float(full["mAP@0.7"])
            support = {}
            for group in ("short", "medium", "long"):
                grouped_annotations = [
                    value
                    for value in annotations
                    if _duration_group(
                        value,
                        short_max_seconds=short_max_seconds,
                        medium_max_seconds=medium_max_seconds,
                    )
                    == group
                ]
                grouped_row = dict(row)
                grouped_row["annotations"] = grouped_annotations
                grouped = _official_metrics(
                    database={video: grouped_row},
                    predictions={video: video_predictions},
                    subset=expected_subset,
                    ground_truth_path=ground_truth_path,
                )
                video_metrics[f"{group}_map"][video] = float(
                    grouped["average_mAP"]
                )
                support[group] = len(grouped_annotations)
            pair, boundary, count = _matching_metrics(
                annotations,
                video_predictions,
            )
            video_metrics["pair_support"][video] = pair
            video_metrics["boundary_error"][video] = boundary
            duration_support[video] = {**support, "all": count}
    payload = {
        "schema_version": SCHEMA,
        "phase": int(phase),
        "terminal_schema_version": str(terminal_schema),
        "git_commit": evaluation["git_commit"],
        "variant": evaluation["variant"],
        "seed": int(evaluation["seed"]),
        "detector_backend": str(evaluation["detector_backend"]),
        "target_mean_cost": float(evaluation["target_mean_cost"]),
        "padded_to_kmax": False,
        "split_role": resolved_split_role,
        "split_manifest_path": str(Path(split_manifest).resolve()),
        "split_manifest_sha256": split_validation["manifest_sha256"],
        "split_assignment_sha256": split_validation["assignment_sha256"],
        "evaluation_video_ids": list(expected_videos),
        "terminal_evaluation_path": str(evaluation_path),
        "terminal_evaluation_sha256": _sha256_file(evaluation_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": _sha256_file(prediction_path),
        "annotation_path": str(annotation_path),
        "annotation_sha256": _sha256_file(annotation_path),
        "duration_thresholds_seconds": {
            "short_max": float(short_max_seconds),
            "medium_max": float(medium_max_seconds),
        },
        "duration_support": duration_support,
        "video_metrics": video_metrics,
        "official_evaluator_used_for_map_metrics": True,
        "pair_support_tiou_threshold": 0.7,
        "boundary_error_normalization": "mean_absolute_endpoint_error_over_gt_duration",
        "uses_official_final": int(phase) == 4,
        "official_final_used_for_training_or_selection": False,
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    target = Path(output).expanduser().resolve()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite different RIME metrics: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return {"path": str(target), "sha256": _sha256_file(target), "payload": payload}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build per-video DUCA-RIME localization evidence."
    )
    parser.add_argument("--terminal-evaluation", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--phase", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--split-role", choices=TRAIN_ROLES)
    parser.add_argument("--short-max-seconds", type=float, required=True)
    parser.add_argument("--medium-max-seconds", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = evaluate_predictions(
        terminal_evaluation=args.terminal_evaluation,
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_manifest_sha256,
        phase=args.phase,
        short_max_seconds=args.short_max_seconds,
        medium_max_seconds=args.medium_max_seconds,
        output=args.output,
        split_role=args.split_role,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
