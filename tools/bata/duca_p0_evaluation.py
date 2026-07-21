from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path
import random
import tempfile
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


def normalize_evaluation_config(
    value: Any,
    *,
    expected_subset: str = "validation",
) -> dict[str, Any]:
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
    expected_subset = str(expected_subset)
    if expected_subset not in {"training", "validation", "test"}:
        raise ValueError(f"unsupported formal DUCA evaluation subset: {expected_subset}")
    if str(cfg.get("subset")) != expected_subset:
        raise ValueError(
            "formal DUCA evaluation subset mismatch: "
            f"expected {expected_subset}, got {cfg.get('subset')}"
        )
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
        "subset": expected_subset,
        "tiou_thresholds": EXPECTED_TIOU_THRESHOLDS,
        "top_k": None,
        "blocked_videos": None if blocked_path is None else str(blocked_path),
        "thread": int(cfg.get("thread", 16)),
    }


def evaluation_config_sha256(
    value: Any,
    *,
    expected_subset: str = "validation",
) -> str:
    return canonical_sha256(
        normalize_evaluation_config(value, expected_subset=expected_subset)
    )


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


def prediction_results(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    prediction_path = Path(path).resolve()
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("prediction artifact has no results mapping")
    normalized: dict[str, list[dict[str, Any]]] = {}
    for video_id, proposals in results.items():
        if not isinstance(video_id, str) or not isinstance(proposals, list):
            raise ValueError("prediction artifact has an invalid result entry")
        normalized[video_id] = [dict(item) for item in proposals]
    return normalized


def evaluation_video_ids(
    evaluation_config: Any,
    *,
    expected_subset: str,
) -> tuple[str, ...]:
    cfg = normalize_evaluation_config(
        evaluation_config,
        expected_subset=expected_subset,
    )
    annotation = json.loads(
        Path(cfg["ground_truth_filename"]).read_text(encoding="utf-8")
    )
    database = annotation.get("database")
    if not isinstance(database, Mapping):
        raise ValueError("formal DUCA annotation has no database mapping")
    blocked: set[str] = set()
    if cfg["blocked_videos"] is not None:
        raw_blocked = json.loads(
            Path(cfg["blocked_videos"]).read_text(encoding="utf-8")
        )
        if not isinstance(raw_blocked, list) or any(
            not isinstance(item, str) for item in raw_blocked
        ):
            raise ValueError("formal DUCA blocked-video artifact must be a string list")
        blocked = set(raw_blocked)
    videos = tuple(
        sorted(
            str(video_id)
            for video_id, row in database.items()
            if isinstance(row, Mapping)
            and str(row.get("subset")) == expected_subset
            and str(video_id) not in blocked
        )
    )
    if not videos:
        raise ValueError("formal DUCA evaluation target set is empty")
    return videos


def _metrics_from_evaluator(evaluator: Any) -> dict[str, float]:
    import numpy as np

    from opentad.evaluations.mAP import compute_average_precision_detection

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
    return metrics


def recompute_official_map(
    prediction_path: str | Path,
    evaluation_config: Any,
    *,
    expected_subset: str = "validation",
) -> dict[str, Any]:
    from opentad.evaluations.mAP import mAP

    cfg = normalize_evaluation_config(
        evaluation_config,
        expected_subset=expected_subset,
    )
    kwargs = dict(cfg)
    kwargs.pop("type")
    # Recompute in-process with the exact official import/parsing and AP
    # function. This avoids accepting metrics copied into a JSON sidecar and
    # avoids multiprocessing-dependent ordering in the evidence finalizer.
    evaluator = mAP(prediction_filename=str(Path(prediction_path).resolve()), **kwargs)
    metrics = _metrics_from_evaluator(evaluator)
    result_count, video_count = prediction_counts(prediction_path)
    return {
        "metrics": metrics,
        "result_count": result_count,
        "video_count": video_count,
        "evaluator": official_evaluator_identity(),
        "evaluation_config": cfg,
        "evaluation_config_sha256": canonical_sha256(cfg),
    }


def bootstrap_official_map_differences(
    prediction_paths: Mapping[str, str | Path],
    evaluation_config: Any,
    *,
    baseline_family: str,
    expected_video_ids: list[str] | tuple[str, ...],
    expected_subset: str,
    samples: int = 1000,
    seed: int = 3407,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired video-cluster bootstrap that reruns the official evaluator.

    Duplicate bootstrap draws receive synthetic video identities, preserving
    their multiplicity instead of collapsing repeated videos in a dictionary.
    """

    import numpy as np

    from opentad.evaluations.mAP import mAP

    families = tuple(str(key) for key in prediction_paths)
    if baseline_family not in families or len(families) < 2:
        raise ValueError("bootstrap requires a baseline and at least one comparison")
    if int(samples) < 100:
        raise ValueError("formal DUCA bootstrap requires at least 100 samples")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("bootstrap confidence must lie in (0,1)")
    expected = tuple(str(value) for value in expected_video_ids)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("bootstrap video identities must be nonempty and unique")

    cfg = normalize_evaluation_config(
        evaluation_config,
        expected_subset=expected_subset,
    )
    if evaluation_video_ids(cfg, expected_subset=expected_subset) != tuple(
        sorted(expected)
    ):
        raise ValueError("bootstrap video set differs from the evaluator target set")
    annotation = json.loads(
        Path(cfg["ground_truth_filename"]).read_text(encoding="utf-8")
    )
    database = annotation.get("database")
    if not isinstance(database, Mapping) or any(video not in database for video in expected):
        raise ValueError("bootstrap annotation does not cover the expected videos")
    predictions = {
        family: prediction_results(path)
        for family, path in prediction_paths.items()
    }
    expected_set = set(expected)
    for family, rows in predictions.items():
        extras = set(rows) - expected_set
        if extras:
            raise ValueError(f"{family} prediction contains out-of-scope videos: {sorted(extras)[:4]}")

    rng = random.Random(int(seed))
    sampled_maps = {family: [] for family in families}
    alpha = (1.0 - float(confidence)) / 2.0
    with tempfile.TemporaryDirectory(prefix="duca-r0-bootstrap-") as directory:
        ground_truth_path = Path(directory) / "ground_truth.json"
        for _ in range(int(samples)):
            draw = [expected[rng.randrange(len(expected))] for _ in expected]
            synthetic_database: dict[str, Any] = {}
            synthetic_predictions = {family: {} for family in families}
            for draw_index, video_id in enumerate(draw):
                synthetic_id = f"bootstrap_{draw_index:05d}_{video_id}"
                synthetic_database[synthetic_id] = dict(database[video_id])
                for family in families:
                    synthetic_predictions[family][synthetic_id] = [
                        dict(item) for item in predictions[family].get(video_id, [])
                    ]
            ground_truth_path.write_text(
                json.dumps({"database": synthetic_database}, sort_keys=True),
                encoding="utf-8",
            )
            kwargs = dict(cfg)
            kwargs.pop("type")
            kwargs["ground_truth_filename"] = str(ground_truth_path)
            kwargs["blocked_videos"] = None
            for family in families:
                evaluator = mAP(
                    prediction_filename={"results": synthetic_predictions[family]},
                    **kwargs,
                )
                sampled_maps[family].append(
                    _metrics_from_evaluator(evaluator)["average_mAP"]
                )

    baseline = np.asarray(sampled_maps[baseline_family], dtype=np.float64)
    comparisons: dict[str, Any] = {}
    for family in families:
        if family == baseline_family:
            continue
        delta = np.asarray(sampled_maps[family], dtype=np.float64) - baseline
        comparisons[family] = {
            "headroom_samples": [float(value) for value in delta],
            "headroom_mean": float(delta.mean()),
            "headroom_ci_lower": float(np.quantile(delta, alpha)),
            "headroom_ci_upper": float(np.quantile(delta, 1.0 - alpha)),
        }
    return {
        "schema": "duca_r0_official_video_bootstrap_v1",
        "official_evaluator_reexecuted_per_resample": True,
        "paired_video_cluster_bootstrap": True,
        "baseline_family": baseline_family,
        "family_order": list(families),
        "video_ids": list(expected),
        "samples": int(samples),
        "seed": int(seed),
        "confidence": float(confidence),
        "evaluation_config": cfg,
        "evaluation_config_sha256": canonical_sha256(cfg),
        "sampled_average_mAP": sampled_maps,
        "comparisons": comparisons,
    }


__all__ = [
    "EXPECTED_EVALUATOR_CLASS",
    "EXPECTED_EVALUATOR_MODULE",
    "EXPECTED_TIOU_THRESHOLDS",
    "evaluation_config_sha256",
    "evaluation_video_ids",
    "normalize_evaluation_config",
    "official_evaluator_identity",
    "prediction_counts",
    "prediction_results",
    "bootstrap_official_map_differences",
    "recompute_official_map",
]
