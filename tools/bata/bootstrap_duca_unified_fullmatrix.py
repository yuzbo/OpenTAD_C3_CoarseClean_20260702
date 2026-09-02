from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mmengine.config import Config

from opentad.evaluations.builder import remove_duplicate_annotations
from opentad.evaluations.mAP import compute_average_precision_detection


PRIMARY_CANDIDATE = "A11"
PRIMARY_CONTROL = "A10"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _work_dir(row: dict[str, Any], project_dir: Path) -> Path:
    work = Path(str(row["work_dir"]))
    if not work.is_absolute():
        work = project_dir / work
    return work / "gpu1_id0"


def _config_path(row: dict[str, Any], project_dir: Path) -> Path:
    path = Path(str(row["config_path"]))
    if not path.is_absolute():
        path = project_dir / path
    return path


def _activity_index(gt_payload: dict[str, Any], subset: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for video in gt_payload["database"].values():
        if video.get("subset") != subset:
            continue
        for ann in remove_duplicate_annotations(video.get("annotations", [])):
            label = ann["label"]
            if label not in out:
                out[label] = len(out)
    return out


def _ground_truth_by_video(
    gt_payload: dict[str, Any],
    *,
    subset: str,
    activity_index: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for video_id, video in gt_payload["database"].items():
        if video.get("subset") != subset:
            continue
        rows = []
        for ann in remove_duplicate_annotations(video.get("annotations", [])):
            rows.append(
                {
                    "t-start": float(ann["segment"][0]),
                    "t-end": float(ann["segment"][1]),
                    "label": int(activity_index[ann["label"]]),
                }
            )
        if rows:
            out[str(video_id)] = rows
    return out


def _predictions_by_video(
    pred_payload: dict[str, Any],
    *,
    activity_index: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for video_id, preds in pred_payload.get("results", {}).items():
        rows = []
        for pred in preds:
            label = pred.get("label")
            label_idx = activity_index.get(label, len(activity_index))
            rows.append(
                {
                    "t-start": float(pred["segment"][0]),
                    "t-end": float(pred["segment"][1]),
                    "label": int(label_idx),
                    "score": float(pred["score"]),
                }
            )
        out[str(video_id)] = rows
    return out


def _bootstrap_frames(
    sample_video_ids: list[str],
    gt_by_video: dict[str, list[dict[str, Any]]],
    pred_by_video: dict[str, list[dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_rows = []
    pred_rows = []
    for sample_idx, video_id in enumerate(sample_video_ids):
        sampled_id = f"{sample_idx:04d}:{video_id}"
        for row in gt_by_video.get(video_id, []):
            gt_rows.append({"video-id": sampled_id, **row})
        for row in pred_by_video.get(video_id, []):
            pred_rows.append({"video-id": sampled_id, **row})
    gt = pd.DataFrame(gt_rows, columns=["video-id", "t-start", "t-end", "label"])
    pred = pd.DataFrame(pred_rows, columns=["video-id", "t-start", "t-end", "label", "score"])
    return gt, pred


def _evaluate_average_map(
    sample_video_ids: list[str],
    gt_by_video: dict[str, list[dict[str, Any]]],
    pred_by_video: dict[str, list[dict[str, Any]]],
    *,
    class_count: int,
    tiou_thresholds: list[float],
) -> dict[str, float]:
    gt, pred = _bootstrap_frames(sample_video_ids, gt_by_video, pred_by_video)
    ap = np.zeros((len(tiou_thresholds), class_count), dtype=np.float64)
    if gt.empty:
        return {"average_mAP": 0.0, **{f"mAP@{thr}": 0.0 for thr in tiou_thresholds}}
    for class_idx in range(class_count):
        gt_idx = gt["label"] == class_idx
        pred_idx = pred["label"] == class_idx if not pred.empty else []
        class_gt = gt.loc[gt_idx].reset_index(drop=True)
        class_pred = pred.loc[pred_idx].reset_index(drop=True) if not pred.empty else pred
        if class_gt.empty:
            continue
        ap[:, class_idx] = compute_average_precision_detection(
            class_gt,
            class_pred,
            tiou_thresholds=np.asarray(tiou_thresholds, dtype=np.float64),
        )
    maps = ap.mean(axis=1)
    payload = {"average_mAP": float(maps.mean())}
    payload.update({f"mAP@{thr}": float(value) for thr, value in zip(tiou_thresholds, maps)})
    return payload


def _row_for(matrix: dict[str, Any], arm_id: str, seed: int) -> dict[str, Any] | None:
    for row in matrix["rows"]:
        if row["phase"] == "confirmation" and row["arm_id"] == arm_id and int(row["seed"]) == int(seed):
            return row
    return None


def _load_result(row: dict[str, Any], project_dir: Path) -> dict[str, Any] | None:
    path = _work_dir(row, project_dir) / "result_detection.json"
    if not path.is_file():
        return None
    return _read_json(path)


def build_bootstrap(
    matrix_path: Path,
    *,
    run_root: Path,
    shard: int,
    num_shards: int,
    draws: int,
) -> dict[str, Any]:
    matrix = _read_json(matrix_path)
    project_dir = matrix_path.resolve().parents[2]
    first_config = Config.fromfile(str(_config_path(matrix["rows"][0], project_dir)))
    gt_path = Path(str(first_config.evaluation.ground_truth_filename))
    if not gt_path.is_absolute():
        gt_path = project_dir / gt_path
    gt_payload = _read_json(gt_path)
    subset = str(first_config.evaluation.subset)
    tiou_thresholds = [float(value) for value in first_config.evaluation.tiou_thresholds]
    activity_index = _activity_index(gt_payload, subset)
    gt_by_video = _ground_truth_by_video(gt_payload, subset=subset, activity_index=activity_index)
    video_ids = sorted(gt_by_video)
    if not video_ids:
        raise ValueError("no validation videos found for bootstrap")

    confirmation_seeds = sorted(
        {
            int(row["seed"])
            for row in matrix["rows"]
            if row["phase"] == "confirmation" and row["arm_id"] in {PRIMARY_CANDIDATE, PRIMARY_CONTROL}
        }
    )
    required_results = {}
    missing = []
    for seed in confirmation_seeds:
        for arm_id in (PRIMARY_CANDIDATE, PRIMARY_CONTROL):
            row = _row_for(matrix, arm_id, seed)
            if row is None:
                missing.append(f"{arm_id}/seed{seed}:matrix_row")
                continue
            payload = _load_result(row, project_dir)
            if payload is None:
                missing.append(f"{arm_id}/seed{seed}:result_detection")
                continue
            required_results[(arm_id, seed)] = _predictions_by_video(payload, activity_index=activity_index)
    if missing:
        return {
            "schema_version": "duca_unified_primary_bootstrap_shard_v1",
            "matrix_id": matrix["matrix_id"],
            "run_root": str(run_root),
            "shard": int(shard),
            "num_shards": int(num_shards),
            "complete": False,
            "missing": missing,
            "deltas": [],
        }

    start = int(math.floor(int(draws) * int(shard) / int(num_shards)))
    end = int(math.floor(int(draws) * (int(shard) + 1) / int(num_shards)))
    rng = np.random.default_rng(20260902 + int(shard))
    deltas = []
    class_count = len(activity_index)
    for draw_idx in range(start, end):
        sampled = [video_ids[int(idx)] for idx in rng.integers(0, len(video_ids), size=len(video_ids))]
        seed_deltas = []
        for seed in confirmation_seeds:
            candidate = _evaluate_average_map(
                sampled,
                gt_by_video,
                required_results[(PRIMARY_CANDIDATE, seed)],
                class_count=class_count,
                tiou_thresholds=tiou_thresholds,
            )
            control = _evaluate_average_map(
                sampled,
                gt_by_video,
                required_results[(PRIMARY_CONTROL, seed)],
                class_count=class_count,
                tiou_thresholds=tiou_thresholds,
            )
            seed_deltas.append(float(candidate["average_mAP"] - control["average_mAP"]))
        deltas.append(
            {
                "draw": int(draw_idx),
                "mean_delta_average_mAP": float(sum(seed_deltas) / len(seed_deltas)),
                "mean_delta_average_mAP_pp": float(100.0 * sum(seed_deltas) / len(seed_deltas)),
                "seed_deltas": seed_deltas,
            }
        )
    return {
        "schema_version": "duca_unified_primary_bootstrap_shard_v1",
        "matrix_id": matrix["matrix_id"],
        "run_root": str(run_root),
        "shard": int(shard),
        "num_shards": int(num_shards),
        "draws_requested_total": int(draws),
        "draw_start": start,
        "draw_end": end,
        "draw_count": len(deltas),
        "complete": True,
        "contrast": f"{PRIMARY_CANDIDATE}-{PRIMARY_CONTROL}",
        "confirmation_seeds": confirmation_seeds,
        "video_cluster_count": len(video_ids),
        "tiou_thresholds": tiou_thresholds,
        "deltas": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Shard DUCA unified full-matrix paired video-cluster bootstrap")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_bootstrap(
        args.matrix,
        run_root=args.run_root,
        shard=args.shard,
        num_shards=args.num_shards,
        draws=args.draws,
    )
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
