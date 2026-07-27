from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, is_dataclass
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

_BOOTSTRAP_WORKER_STATE: dict[str, Any] | None = None
_BOOTSTRAP_WORKER_DIRECTORY: tempfile.TemporaryDirectory[str] | None = None


def canonical_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return canonical_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): canonical_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        canonical_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
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


def _evaluate_bootstrap_draw(
    draw: tuple[str, ...],
    *,
    families: tuple[str, ...],
    database: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    cfg: Mapping[str, Any],
    ground_truth_path: Path,
    metric_names: tuple[str, ...] = ("average_mAP",),
) -> tuple[float, ...]:
    from opentad.evaluations.mAP import mAP

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
    values = []
    for family in families:
        evaluator = mAP(
            prediction_filename={"results": synthetic_predictions[family]},
            **kwargs,
        )
        metrics = _metrics_from_evaluator(evaluator)
        for metric_name in metric_names:
            if metric_name not in metrics:
                raise ValueError(
                    f"official evaluator did not emit bootstrap metric {metric_name}"
                )
            values.append(float(metrics[metric_name]))
    return tuple(values)


def _initialize_bootstrap_worker(
    families: tuple[str, ...],
    database: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    cfg: Mapping[str, Any],
    metric_names: tuple[str, ...] = ("average_mAP",),
) -> None:
    global _BOOTSTRAP_WORKER_DIRECTORY, _BOOTSTRAP_WORKER_STATE
    _BOOTSTRAP_WORKER_DIRECTORY = tempfile.TemporaryDirectory(
        prefix="duca-r0-bootstrap-worker-"
    )
    _BOOTSTRAP_WORKER_STATE = {
        "families": families,
        "database": database,
        "predictions": predictions,
        "cfg": cfg,
        "metric_names": metric_names,
        "ground_truth_path": Path(_BOOTSTRAP_WORKER_DIRECTORY.name)
        / "ground_truth.json",
    }


def _evaluate_bootstrap_draw_in_worker(draw: tuple[str, ...]) -> tuple[float, ...]:
    if _BOOTSTRAP_WORKER_STATE is None:
        raise RuntimeError("DUCA bootstrap worker was not initialized")
    return _evaluate_bootstrap_draw(draw, **_BOOTSTRAP_WORKER_STATE)


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
    workers: int = 1,
) -> dict[str, Any]:
    """Paired video-cluster bootstrap that reruns the official evaluator.

    Duplicate bootstrap draws receive synthetic video identities, preserving
    their multiplicity instead of collapsing repeated videos in a dictionary.
    """

    import numpy as np

    families = tuple(str(key) for key in prediction_paths)
    if baseline_family not in families or len(families) < 2:
        raise ValueError("bootstrap requires a baseline and at least one comparison")
    if int(samples) < 100:
        raise ValueError("formal DUCA bootstrap requires at least 100 samples")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("bootstrap confidence must lie in (0,1)")
    workers = int(workers)
    if workers < 1 or workers > 64:
        raise ValueError("formal DUCA bootstrap workers must lie in [1,64]")
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
    draws = [
        tuple(expected[rng.randrange(len(expected))] for _ in expected)
        for _ in range(int(samples))
    ]
    sampled_maps = {family: [] for family in families}
    alpha = (1.0 - float(confidence)) / 2.0
    if workers == 1:
        with tempfile.TemporaryDirectory(prefix="duca-r0-bootstrap-") as directory:
            ground_truth_path = Path(directory) / "ground_truth.json"
            results = (
                _evaluate_bootstrap_draw(
                    draw,
                    families=families,
                    database=database,
                    predictions=predictions,
                    cfg=cfg,
                    ground_truth_path=ground_truth_path,
                )
                for draw in draws
            )
            for sample_index, values in enumerate(results, start=1):
                for family, value in zip(families, values):
                    sampled_maps[family].append(value)
                if sample_index % max(1, len(draws) // 10) == 0 or sample_index == len(
                    draws
                ):
                    print(
                        f"[DUCA_R0_BOOTSTRAP] {sample_index}/{len(draws)} "
                        f"workers={workers}",
                        flush=True,
                    )
    else:
        chunk_size = max(1, len(draws) // (workers * 8))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_bootstrap_worker,
            initargs=(families, database, predictions, cfg),
        ) as executor:
            results = executor.map(
                _evaluate_bootstrap_draw_in_worker,
                draws,
                chunksize=chunk_size,
            )
            for sample_index, values in enumerate(results, start=1):
                for family, value in zip(families, values):
                    sampled_maps[family].append(value)
                if sample_index % max(1, len(draws) // 10) == 0 or sample_index == len(
                    draws
                ):
                    print(
                        f"[DUCA_R0_BOOTSTRAP] {sample_index}/{len(draws)} "
                        f"workers={workers}",
                        flush=True,
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


def bootstrap_official_metric_differences(
    prediction_paths: Mapping[str, str | Path],
    evaluation_config: Any,
    *,
    baseline_family: str,
    expected_video_ids: list[str] | tuple[str, ...],
    expected_subset: str,
    metric_names: tuple[str, ...] = ("average_mAP", "mAP@0.7"),
    samples: int = 1000,
    seed: int = 3407,
    confidence: float = 0.95,
    workers: int = 1,
) -> dict[str, Any]:
    """Paired bootstrap for multiple metrics, rerunning the official evaluator."""

    import numpy as np

    families = tuple(str(key) for key in prediction_paths)
    metrics = tuple(str(value) for value in metric_names)
    if baseline_family not in families or len(families) < 2:
        raise ValueError("bootstrap requires a baseline and at least one comparison")
    if (
        not metrics
        or len(metrics) != len(set(metrics))
        or any(
            metric not in {"average_mAP", *(
                f"mAP@{float(value)}" for value in EXPECTED_TIOU_THRESHOLDS
            )}
            for metric in metrics
        )
    ):
        raise ValueError("unsupported or duplicated official bootstrap metric")
    if int(samples) < 100:
        raise ValueError("formal DUCA bootstrap requires at least 100 samples")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("bootstrap confidence must lie in (0,1)")
    workers = int(workers)
    if workers < 1 or workers > 64:
        raise ValueError("formal DUCA bootstrap workers must lie in [1,64]")
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
    if not isinstance(database, Mapping) or any(
        video not in database for video in expected
    ):
        raise ValueError("bootstrap annotation does not cover the expected videos")
    predictions = {
        family: prediction_results(path)
        for family, path in prediction_paths.items()
    }
    expected_set = set(expected)
    for family, rows in predictions.items():
        extras = set(rows) - expected_set
        if extras:
            raise ValueError(
                f"{family} prediction contains out-of-scope videos: {sorted(extras)[:4]}"
            )

    rng = random.Random(int(seed))
    draws = [
        tuple(expected[rng.randrange(len(expected))] for _ in expected)
        for _ in range(int(samples))
    ]
    sampled = {
        family: {metric: [] for metric in metrics}
        for family in families
    }

    def absorb(values: tuple[float, ...]) -> None:
        cursor = 0
        for family in families:
            for metric in metrics:
                sampled[family][metric].append(float(values[cursor]))
                cursor += 1

    if workers == 1:
        with tempfile.TemporaryDirectory(prefix="duca-rime-bootstrap-") as directory:
            ground_truth_path = Path(directory) / "ground_truth.json"
            for draw in draws:
                absorb(
                    _evaluate_bootstrap_draw(
                        draw,
                        families=families,
                        database=database,
                        predictions=predictions,
                        cfg=cfg,
                        ground_truth_path=ground_truth_path,
                        metric_names=metrics,
                    )
                )
    else:
        chunk_size = max(1, len(draws) // (workers * 8))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_bootstrap_worker,
            initargs=(families, database, predictions, cfg, metrics),
        ) as executor:
            for values in executor.map(
                _evaluate_bootstrap_draw_in_worker,
                draws,
                chunksize=chunk_size,
            ):
                absorb(values)

    alpha = (1.0 - float(confidence)) / 2.0
    aliases = {"average_mAP": "avg_map", "mAP@0.7": "map_0.7"}
    comparisons: dict[str, Any] = {}
    for family in families:
        if family == baseline_family:
            continue
        comparisons[family] = {}
        for metric in metrics:
            left = np.asarray(sampled[family][metric], dtype=np.float64)
            right = np.asarray(sampled[baseline_family][metric], dtype=np.float64)
            delta = left - right
            comparisons[family][aliases.get(metric, metric)] = {
                "mean": float(delta.mean()),
                "ci95_low": float(np.quantile(delta, alpha)),
                "ci95_high": float(np.quantile(delta, 1.0 - alpha)),
            }
    return {
        "schema_version": "duca_rime_official_metric_bootstrap_v1",
        "official_evaluator_reexecuted_per_resample": True,
        "paired_video_cluster_bootstrap": True,
        "baseline_family": baseline_family,
        "family_order": list(families),
        "official_metric_names": list(metrics),
        "video_ids": list(expected),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
        "confidence": float(confidence),
        "evaluation_config": cfg,
        "evaluation_config_sha256": canonical_sha256(cfg),
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
    "bootstrap_official_metric_differences",
    "bootstrap_official_map_differences",
    "recompute_official_map",
]
