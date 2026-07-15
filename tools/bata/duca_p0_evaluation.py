from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

EXPECTED_TIOU_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
EXPECTED_EVALUATOR_MODULE = "opentad.evaluations.mAP"
EXPECTED_EVALUATOR_CLASS = "mAP"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError("evaluation config must be a mapping")
    return {str(key): item for key, item in value.items()}


def normalize_evaluation_config(value: Any) -> dict[str, Any]:
    cfg = _plain_mapping(value)
    allowed = {
        "type",
        "ground_truth_filename",
        "subset",
        "tiou_thresholds",
        "top_k",
        "blocked_videos",
        "thread",
    }
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise ValueError(f"formal DUCA evaluation has unknown fields: {unknown}")
    if cfg.get("type") != EXPECTED_EVALUATOR_CLASS:
        raise ValueError("formal DUCA evaluation must use OpenTAD mAP")
    if str(cfg.get("subset")) != "validation":
        raise ValueError("formal DUCA evaluation must use the THUMOS validation subset")
    thresholds = [float(item) for item in cfg.get("tiou_thresholds", [])]
    if thresholds != EXPECTED_TIOU_THRESHOLDS:
        raise ValueError("formal DUCA evaluation tIoU thresholds are not frozen")
    annotation = Path(str(cfg.get("ground_truth_filename", ""))).expanduser().resolve()
    if not annotation.is_file():
        raise ValueError(f"formal DUCA evaluation annotation is missing: {annotation}")
    top_k = cfg.get("top_k")
    if top_k is not None:
        raise ValueError("formal DUCA P0 primary evaluation does not use top-k recall")
    blocked = cfg.get("blocked_videos")
    blocked_path = None
    if blocked is not None:
        blocked_path = Path(str(blocked)).expanduser().resolve()
        if not blocked_path.is_file():
            raise ValueError(f"formal DUCA blocked-video file is missing: {blocked_path}")
    return {
        "type": EXPECTED_EVALUATOR_CLASS,
        "ground_truth_filename": str(annotation),
        "subset": "validation",
        "tiou_thresholds": EXPECTED_TIOU_THRESHOLDS,
        "top_k": None,
        "blocked_videos": None if blocked_path is None else str(blocked_path),
        "thread": int(cfg.get("thread", 16)),
    }


def evaluation_config_sha256(value: Any) -> str:
    return canonical_sha256(normalize_evaluation_config(value))


def official_evaluator_identity() -> dict[str, str]:
    from opentad.evaluations.mAP import mAP

    source = inspect.getsourcefile(mAP)
    if source is None:
        raise ValueError("cannot locate the OpenTAD mAP evaluator source")
    source_path = Path(source).resolve()
    expected_path = (
        Path(__file__).resolve().parents[2] / "opentad/evaluations/mAP.py"
    ).resolve()
    if source_path != expected_path:
        raise ValueError(
            f"unexpected mAP evaluator source: expected {expected_path}, got {source_path}"
        )
    return {
        "module": mAP.__module__,
        "class_name": mAP.__qualname__,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
    }


def prediction_counts(path: str | Path) -> tuple[int, int]:
    prediction_path = Path(path).resolve()
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("prediction artifact has no results mapping")
    count = 0
    for video_id, proposals in results.items():
        if not isinstance(video_id, str) or not isinstance(proposals, list):
            raise ValueError("prediction artifact has an invalid result entry")
        count += len(proposals)
    if not results or count <= 0:
        raise ValueError("formal DUCA prediction artifact is empty")
    return count, len(results)


def recompute_official_map(
    prediction_path: str | Path,
    evaluation_config: Any,
) -> dict[str, Any]:
    import numpy as np

    from opentad.evaluations.mAP import (
        compute_average_precision_detection,
        mAP,
    )

    cfg = normalize_evaluation_config(evaluation_config)
    kwargs = dict(cfg)
    kwargs.pop("type")
    # Recompute in-process with the exact official import/parsing and AP
    # function. This avoids accepting metrics copied into a JSON sidecar and
    # avoids multiprocessing-dependent ordering in the evidence finalizer.
    evaluator = mAP(prediction_filename=str(Path(prediction_path).resolve()), **kwargs)
    class_ids = list(evaluator.activity_index.values())
    if not class_ids:
        raise ValueError("formal DUCA annotation contains no evaluation classes")
    thresholds = np.asarray(evaluator.tiou_thresholds, dtype=np.float64)
    ap = np.zeros((len(thresholds), len(class_ids)), dtype=np.float64)
    for cidx in class_ids:
        gt_idx = evaluator.ground_truth["label"] == cidx
        pred_idx = evaluator.prediction["label"] == cidx
        ap[:, cidx] = compute_average_precision_detection(
            evaluator.ground_truth.loc[gt_idx].reset_index(drop=True),
            evaluator.prediction.loc[pred_idx].reset_index(drop=True),
            tiou_thresholds=thresholds,
        )
    maps = ap.mean(axis=1)
    metrics = {"average_mAP": float(maps.mean())}
    for threshold, value in zip(thresholds, maps):
        metrics[f"mAP@{float(threshold)}"] = float(value)
    result_count, video_count = prediction_counts(prediction_path)
    return {
        "metrics": metrics,
        "result_count": result_count,
        "video_count": video_count,
        "evaluator": official_evaluator_identity(),
        "evaluation_config": cfg,
        "evaluation_config_sha256": canonical_sha256(cfg),
    }


__all__ = [
    "EXPECTED_EVALUATOR_CLASS",
    "EXPECTED_EVALUATOR_MODULE",
    "EXPECTED_TIOU_THRESHOLDS",
    "evaluation_config_sha256",
    "normalize_evaluation_config",
    "official_evaluator_identity",
    "prediction_counts",
    "recompute_official_map",
]
