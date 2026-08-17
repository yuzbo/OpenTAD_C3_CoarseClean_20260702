#!/usr/bin/env python3
"""Fail-closed finalizer for legacy GeoRoute and the ZoomToken P1 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_estimator_pilot_stage_runner import (  # noqa: E402
    _pilot_profile,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_official_comparable_contract import (  # noqa: E402
    FORMAL_DEVELOPMENT_ARM_ORDER,
    FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA,
    FORMAL_DEVELOPMENT_FINALIZATION_SCHEMA,
    FORMAL_DEVELOPMENT_SEEDS,
    P1_DEVELOPMENT_SEED,
    P1_FIRST_SCREEN_ARM_ORDER,
    P1_MATCHED_RUNNER_ARM_ORDER,
    formal_cell_relative_path,
    read_json,
    validate_p1_shared_official_baseline_receipt,
    validate_protocol_manifest,
)
from tools.bata.georoute_official_development_stage_runner import (  # noqa: E402
    _p1_cell_relative_path,
    summarize_formal_telemetry,
    validate_formal_checkpoint_population,
    validate_formal_stage_result,
    validate_p1_deployment_shape,
)
from tools.bata.zoomtoken_scnr_steady_cost_contract_v001 import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    P1_COST_LEAF_SPECS,
    P1_COST_RATIO_LIMIT,
    P1_STUDY_ID,
    VIDEO_CLUSTERS,
    analyze_p1_cost_leaves,
    p1_cost_leaf_relative_path,
    read_jsonl_objects,
    validate_p1_cost_leaf_receipt,
    validate_p1_cost_rows,
    validate_p1_cost_warmup_rows,
)


SELECTOR_ARMS = ("residual_st_rep_off", "residual_pl_rep_off")
CONTROL_ARMS = ("fixed_lattice", "random")
ACCURACY_KEY = "high_iou_composite"
COST_KEY = "model_and_postprocess_p50_ms"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _self_hash_matches(payload: Mapping[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or f"git {' '.join(arguments)} failed"
        )
    return completed.stdout.strip()


def _validate_artifacts(
    result: Mapping[str, Any],
    *,
    cell_root: Path,
    run_root: Path,
) -> None:
    for path_field, hash_field in (
        ("prediction_path", "prediction_sha256"),
        ("test_log_path", "test_log_sha256"),
    ):
        path = Path(str(result.get(path_field, ""))).resolve()
        if (
            not path.is_file()
            or not _inside(path, cell_root)
            or sha256_file(path) != result.get(hash_field)
        ):
            raise ValueError(f"formal artifact changed: {path_field}")
    config_path = Path(str(result.get("config_path", ""))).resolve()
    expected_config_path = (
        run_root
        / "control"
        / "bound_configs"
        / f"{result['arm']}_seed{int(result['seed'])}.py"
    ).resolve()
    if (
        config_path != expected_config_path
        or not config_path.is_file()
        or sha256_file(config_path) != result.get("config_sha256")
    ):
        raise ValueError("formal artifact changed: config_path")
    profile_path = Path(str(result.get("profile_path", ""))).resolve()
    telemetry_path = Path(str(result.get("telemetry_path", ""))).resolve()
    if (
        not profile_path.is_file()
        or not telemetry_path.is_file()
        or not _inside(profile_path, cell_root)
        or not _inside(telemetry_path, cell_root)
        or _pilot_profile(profile_path) != result.get("profile")
        or summarize_formal_telemetry(
            telemetry_path,
            arm=str(result["arm"]),
        )
        != result.get("telemetry_summary")
    ):
        raise ValueError("formal profile or telemetry receipt changed")
    checkpoint = result.get("checkpoint_receipt")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("formal stage lacks checkpoint receipt")
    checkpoint_path = Path(str(checkpoint.get("path", ""))).resolve()
    sidecar_path = Path(str(checkpoint.get("sidecar_path", ""))).resolve()
    binding = result["binding"]
    recovery_receipts = validate_formal_checkpoint_population(
        checkpoint_path,
        binding=binding,
    )
    sidecar = read_json(sidecar_path)
    if (
        not _inside(checkpoint_path, cell_root)
        or sidecar_path != Path(str(checkpoint_path) + ".metadata.json")
        or checkpoint.get("sha256") != sha256_file(checkpoint_path)
        or int(checkpoint.get("size_bytes", -1))
        != int(checkpoint_path.stat().st_size)
        or checkpoint.get("sidecar_file_sha256")
        != sha256_file(sidecar_path)
        or checkpoint.get("sidecar_sha256")
        != sidecar["sidecar_sha256"]
        or result.get("recovery_checkpoint_receipts") != recovery_receipts
    ):
        raise ValueError("formal checkpoint artifact receipt changed")


def _aggregate(values: Sequence[float]) -> dict[str, float]:
    normalized = [float(value) for value in values]
    if len(normalized) != len(FORMAL_DEVELOPMENT_SEEDS) or any(
        not math.isfinite(value) for value in normalized
    ):
        raise ValueError("formal aggregate requires three finite seed values")
    return {
        "mean": statistics.fmean(normalized),
        "population_sd": statistics.pstdev(normalized),
        "minimum": min(normalized),
        "maximum": max(normalized),
    }


P1_GT_BINS = (
    "HIT_070",
    "START_LIMITED",
    "END_LIMITED",
    "EITHER_ENDPOINT_RESCUES",
    "JOINT_BOUNDARY_LIMITED",
    "CLASS_CONFUSION",
    "MISS_OR_SEVERE_LOCALIZATION",
)
P1_UNMATCHED_PREDICTION_BINS = (
    "DUPLICATE_FP",
    "CLASS_CONFUSION_FP",
    "OTHER_FP",
)


def _finite_interval(value: Any, *, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} interval is malformed")
    start, end = float(value[0]), float(value[1])
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError(f"{label} interval is nonfinite or nonpositive")
    return start, end


def _p1_tiou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    intersection = max(
        0.0,
        min(float(left["end"]), float(right["end"]))
        - max(float(left["start"]), float(right["start"])),
    )
    union = (
        float(left["end"])
        - float(left["start"])
        + float(right["end"])
        - float(right["start"])
        - intersection
    )
    if union <= 0.0:
        raise ValueError("P1 tIoU received a nonpositive union")
    return intersection / union


def _p1_endpoint_corrected_tiou(
    prediction: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    *,
    endpoint: str,
) -> float:
    corrected = dict(prediction)
    corrected[endpoint] = float(ground_truth[endpoint])
    if float(corrected["end"]) <= float(corrected["start"]):
        return 0.0
    return _p1_tiou(corrected, ground_truth)


def _match_p1_video(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen class-consistent score order and inclusive thresholds."""

    gt_rows = [dict(row) for row in ground_truth]
    pred_rows = [dict(row) for row in predictions]
    gt_ids = [str(row.get("id", "")) for row in gt_rows]
    pred_ids = [str(row.get("id", "")) for row in pred_rows]
    if (
        any(not value for value in gt_ids + pred_ids)
        or len(set(gt_ids)) != len(gt_ids)
        or len(set(pred_ids)) != len(pred_ids)
    ):
        raise ValueError("P1 diagnostic IDs are malformed or duplicate")
    matches: list[dict[str, Any]] = []
    matched_gt: set[str] = set()
    matched_pred: set[str] = set()
    for label in sorted({str(row["label"]) for row in pred_rows}):
        ordered_predictions = sorted(
            (row for row in pred_rows if str(row["label"]) == label),
            key=lambda row: (
                -float(row["score"]),
                float(row["start"]),
                float(row["end"]),
                str(row["id"]),
            ),
        )
        for prediction in ordered_predictions:
            candidates = [
                (ground, _p1_tiou(prediction, ground))
                for ground in gt_rows
                if str(ground["label"]) == label
                and str(ground["id"]) not in matched_gt
            ]
            candidates = [pair for pair in candidates if pair[1] >= 0.50]
            if not candidates:
                continue
            ground, overlap = min(
                candidates,
                key=lambda pair: (
                    -pair[1],
                    float(pair[0]["start"]),
                    float(pair[0]["end"]),
                    str(pair[0]["id"]),
                ),
            )
            matched_gt.add(str(ground["id"]))
            matched_pred.add(str(prediction["id"]))
            matches.append(
                {"ground_truth": ground, "prediction": prediction, "tiou": overlap}
            )

    gt_bins = {name: 0 for name in P1_GT_BINS}
    pred_bins = {name: 0 for name in P1_UNMATCHED_PREDICTION_BINS}
    match_by_gt = {str(row["ground_truth"]["id"]): row for row in matches}
    boundary_rows: list[dict[str, float]] = []
    short_gt = 0
    short_hit = 0
    for ground in gt_rows:
        duration = float(ground["end"]) - float(ground["start"])
        is_short = duration <= 5.0
        short_gt += int(is_short)
        match = match_by_gt.get(str(ground["id"]))
        if match is not None:
            prediction = match["prediction"]
            overlap = float(match["tiou"])
            signed_start = (float(prediction["start"]) - float(ground["start"])) / duration
            signed_end = (float(prediction["end"]) - float(ground["end"])) / duration
            boundary_rows.append(
                {
                    "signed_start": signed_start,
                    "signed_end": signed_end,
                    "absolute_start": abs(signed_start),
                    "absolute_end": abs(signed_end),
                }
            )
            if overlap >= 0.70:
                gt_bin = "HIT_070"
            else:
                start_rescues = (
                    _p1_endpoint_corrected_tiou(
                        prediction, ground, endpoint="start"
                    )
                    >= 0.70
                )
                end_rescues = (
                    _p1_endpoint_corrected_tiou(prediction, ground, endpoint="end")
                    >= 0.70
                )
                if start_rescues and end_rescues:
                    gt_bin = "EITHER_ENDPOINT_RESCUES"
                elif start_rescues:
                    gt_bin = "START_LIMITED"
                elif end_rescues:
                    gt_bin = "END_LIMITED"
                else:
                    gt_bin = "JOINT_BOUNDARY_LIMITED"
        elif any(
            str(prediction["label"]) != str(ground["label"])
            and _p1_tiou(prediction, ground) >= 0.50
            for prediction in pred_rows
        ):
            gt_bin = "CLASS_CONFUSION"
        else:
            gt_bin = "MISS_OR_SEVERE_LOCALIZATION"
        gt_bins[gt_bin] += 1
        short_hit += int(is_short and gt_bin == "HIT_070")

    for prediction in pred_rows:
        if str(prediction["id"]) in matched_pred:
            continue
        duplicate = any(
            str(prediction["label"]) == str(ground["label"])
            and _p1_tiou(prediction, ground) >= 0.50
            for ground in gt_rows
        )
        confusion = any(
            str(prediction["label"]) != str(ground["label"])
            and _p1_tiou(prediction, ground) >= 0.50
            for ground in gt_rows
        )
        pred_bin = (
            "DUPLICATE_FP"
            if duplicate
            else "CLASS_CONFUSION_FP"
            if confusion
            else "OTHER_FP"
        )
        pred_bins[pred_bin] += 1
    return {
        "gt_count": len(gt_rows),
        "prediction_count": len(pred_rows),
        "matched_count": len(matches),
        "unmatched_gt_count": len(gt_rows) - len(matches),
        "unmatched_prediction_count": len(pred_rows) - len(matches),
        "gt_bins": gt_bins,
        "unmatched_prediction_bins": pred_bins,
        "short_gt_count": short_gt,
        "short_hit_070_count": short_hit,
        "boundary_rows": boundary_rows,
    }


def _p1_prediction_rows(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    results = payload.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("P1 prediction payload lacks results")
    normalized: dict[str, list[dict[str, Any]]] = {}
    for video_id, raw_rows in results.items():
        if not isinstance(video_id, str) or not video_id or not isinstance(raw_rows, list):
            raise ValueError("P1 prediction video identity is malformed")
        rows = []
        for ordinal, raw in enumerate(raw_rows):
            if not isinstance(raw, Mapping):
                raise ValueError("P1 prediction row is malformed")
            start, end = _finite_interval(raw.get("segment"), label="prediction")
            score = float(raw.get("score", float("nan")))
            label = raw.get("label")
            prediction_id = raw.get("id", f"PRED:{video_id}:{ordinal}")
            if (
                not isinstance(label, str)
                or not label
                or not isinstance(prediction_id, str)
                or not prediction_id
                or not math.isfinite(score)
            ):
                raise ValueError("P1 prediction ID, class, or score is malformed")
            rows.append(
                {
                    "id": prediction_id,
                    "label": label,
                    "start": start,
                    "end": end,
                    "score": score,
                }
            )
        normalized[video_id] = rows
    return normalized


def _p1_ground_truth_rows(
    annotation: Mapping[str, Any], *, video_ids: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    database = annotation.get("database")
    if not isinstance(database, Mapping):
        raise ValueError("P1 annotation lacks its source-video database")
    normalized: dict[str, list[dict[str, Any]]] = {}
    for video_id in video_ids:
        record = database.get(video_id)
        raw_rows = record.get("annotations") if isinstance(record, Mapping) else None
        if not isinstance(raw_rows, list):
            raise ValueError("P1 source video lacks official GT annotations")
        deduplicated: dict[tuple[str, float, float], dict[str, Any]] = {}
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                raise ValueError("P1 GT row is malformed")
            start, end = _finite_interval(raw.get("segment"), label="ground truth")
            label = raw.get("label")
            if not isinstance(label, str) or not label:
                raise ValueError("P1 GT class identity is malformed")
            key = (label, start, end)
            explicit_id = raw.get("id")
            if explicit_id is not None and (
                not isinstance(explicit_id, str) or not explicit_id
            ):
                raise ValueError("P1 GT canonical ID is malformed")
            deduplicated.setdefault(
                key,
                {
                    "id": explicit_id
                    or f"GT:{video_id}:{label}:{start.hex()}:{end.hex()}",
                    "label": label,
                    "start": start,
                    "end": end,
                },
            )
        rows = sorted(
            deduplicated.values(),
            key=lambda row: (row["start"], row["end"], row["label"], row["id"]),
        )
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("P1 GT canonical IDs are not unique")
        normalized[video_id] = rows
    return normalized


def _summarize_p1_video_rows(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    gt_count = sum(int(row["gt_count"]) for row in rows.values())
    matched_count = sum(int(row["matched_count"]) for row in rows.values())
    boundary = [item for row in rows.values() for item in row["boundary_rows"]]
    gt_bins = {
        name: sum(int(row["gt_bins"][name]) for row in rows.values())
        for name in P1_GT_BINS
    }
    prediction_bins = {
        name: sum(
            int(row["unmatched_prediction_bins"][name]) for row in rows.values()
        )
        for name in P1_UNMATCHED_PREDICTION_BINS
    }
    short_gt = sum(int(row["short_gt_count"]) for row in rows.values())
    short_hit = sum(int(row["short_hit_070_count"]) for row in rows.values())
    return {
        "video_cluster_count": len(rows),
        "gt_count": gt_count,
        "matched_count": matched_count,
        "unmatched_gt_count": sum(
            int(row["unmatched_gt_count"]) for row in rows.values()
        ),
        "unmatched_prediction_count": sum(
            int(row["unmatched_prediction_count"]) for row in rows.values()
        ),
        "gt_bins": gt_bins,
        "gt_bin_micro_rates": {
            name: count / gt_count if gt_count else None for name, count in gt_bins.items()
        },
        "unmatched_prediction_bins": prediction_bins,
        "short_actions": {
            "definition_seconds": "0 < end-start <= 5.0",
            "gt_count": short_gt,
            "hit_070_count": short_hit,
            "tp_070_recall": short_hit / short_gt if short_gt else None,
            "report_only": True,
        },
        "boundary": {
            "matched_count": matched_count,
            "mean_absolute_start_error_over_gt_duration": (
                statistics.fmean(item["absolute_start"] for item in boundary)
                if boundary
                else None
            ),
            "mean_absolute_end_error_over_gt_duration": (
                statistics.fmean(item["absolute_end"] for item in boundary)
                if boundary
                else None
            ),
            "median_signed_start_offset_over_gt_duration": (
                statistics.median(item["signed_start"] for item in boundary)
                if boundary
                else None
            ),
            "median_signed_end_offset_over_gt_duration": (
                statistics.median(item["signed_end"] for item in boundary)
                if boundary
                else None
            ),
            "zero_match_errors": "NA" if not boundary else None,
            "unmatched_not_imputed": True,
            "report_only": True,
        },
        "high_iou_decomposition_report_only": True,
    }


def evaluate_p1_report_only_diagnostics(
    annotation: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    expected_video_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    predictions = _p1_prediction_rows(prediction)
    if expected_video_ids is not None:
        video_ids = tuple(sorted(set(map(str, expected_video_ids))))
        if not set(predictions) <= set(video_ids):
            raise ValueError("P1 prediction payload contains a non-Gate video")
        for video_id in video_ids:
            predictions.setdefault(video_id, [])
    else:
        video_ids = tuple(sorted(predictions))
    if len(video_ids) != VIDEO_CLUSTERS:
        raise ValueError("P1 diagnostics require exactly 40 source-video clusters")
    ground_truth = _p1_ground_truth_rows(annotation, video_ids=video_ids)
    video_rows = {
        video_id: _match_p1_video(ground_truth[video_id], predictions[video_id])
        for video_id in video_ids
    }
    return {
        "schema_version": "zoomtoken_p1_report_only_diagnostics_v001",
        "matching": {
            "prediction_order": "descending_score_then_start_end_canonical_id",
            "same_class_greedy_one_to_one": True,
            "tiou_threshold": 0.50,
            "threshold_inclusive": True,
            "gt_tie_break": "start_then_end_then_canonical_id",
        },
        "summary": _summarize_p1_video_rows(video_rows),
        "video_rows": video_rows,
        "report_only": True,
        "affects_p1_gate": False,
    }


def _p1_diagnostic_vector(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    keyed = {str(index): row for index, row in enumerate(rows)}
    summary = _summarize_p1_video_rows(keyed)
    vector: dict[str, float | None] = {
        f"gt_bin_rate/{name}": summary["gt_bin_micro_rates"][name]
        for name in P1_GT_BINS
    }
    vector["short_tp_070_recall"] = summary["short_actions"]["tp_070_recall"]
    vector["mean_absolute_start_error"] = summary["boundary"][
        "mean_absolute_start_error_over_gt_duration"
    ]
    vector["mean_absolute_end_error"] = summary["boundary"][
        "mean_absolute_end_error_over_gt_duration"
    ]
    return vector


def paired_p1_diagnostic_bootstrap(
    q_diagnostics: Mapping[str, Any],
    comparator_diagnostics: Mapping[str, Any],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    q_rows = q_diagnostics.get("video_rows")
    comparator_rows = comparator_diagnostics.get("video_rows")
    if (
        not isinstance(q_rows, Mapping)
        or not isinstance(comparator_rows, Mapping)
        or set(q_rows) != set(comparator_rows)
        or len(q_rows) != VIDEO_CLUSTERS
        or int(bootstrap_replicates) <= 0
    ):
        raise ValueError("P1 diagnostic bootstrap population is invalid")
    videos = tuple(sorted(q_rows))
    point_q = _p1_diagnostic_vector([q_rows[video] for video in videos])
    point_comparator = _p1_diagnostic_vector(
        [comparator_rows[video] for video in videos]
    )
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draws: dict[str, list[float]] = {name: [] for name in point_q}
    for _replicate in range(int(bootstrap_replicates)):
        sampled = rng.integers(0, len(videos), size=VIDEO_CLUSTERS)
        q_vector = _p1_diagnostic_vector(
            [q_rows[videos[int(index)]] for index in sampled]
        )
        comparator_vector = _p1_diagnostic_vector(
            [comparator_rows[videos[int(index)]] for index in sampled]
        )
        for name in draws:
            if q_vector[name] is not None and comparator_vector[name] is not None:
                draws[name].append(float(q_vector[name]) - float(comparator_vector[name]))
    differences: dict[str, Any] = {}
    for name, values in draws.items():
        point = (
            None
            if point_q[name] is None or point_comparator[name] is None
            else float(point_q[name]) - float(point_comparator[name])
        )
        interval = (
            None
            if not values
            else [
                float(value)
                for value in np.quantile(values, [0.025, 0.975], method="linear")
            ]
        )
        differences[name] = {
            "q_minus_comparator": point,
            "paired_percentile_95_ci": interval,
        }
    return {
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "video_cluster_count": VIDEO_CLUSTERS,
        "differences": differences,
        "report_only": True,
        "affects_p1_gate": False,
    }


def _selector_eligibility(
    results: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    arm: str,
) -> dict[str, Any]:
    paired: dict[str, Any] = {}
    passed = True
    for seed in FORMAL_DEVELOPMENT_SEEDS:
        treatment = results[arm][seed]
        accuracy = float(treatment["metrics"][ACCURACY_KEY])
        cost = float(treatment["profile"][COST_KEY])
        fixed_delta = accuracy - float(
            results["fixed_lattice"][seed]["metrics"][ACCURACY_KEY]
        )
        random_delta = accuracy - float(
            results["random"][seed]["metrics"][ACCURACY_KEY]
        )
        dense_cost_delta = cost - float(
            results["dense_native"][seed]["profile"][COST_KEY]
        )
        seed_passed = (
            fixed_delta > 0.0
            and random_delta > 0.0
            and dense_cost_delta < 0.0
        )
        passed = passed and seed_passed
        paired[str(seed)] = {
            "high_iou_delta_vs_fixed": fixed_delta,
            "high_iou_delta_vs_random": random_delta,
            "development_cost_delta_vs_dense_ms": dense_cost_delta,
            "passed": seed_passed,
        }
    return {
        "arm": arm,
        "paired_seed_checks": paired,
        "all_three_seeds_passed": passed,
    }


def _strict_pareto_dominates(
    results: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    treatment: str,
    control: str,
) -> dict[str, Any]:
    accuracy_deltas = [
        float(results[treatment][seed]["metrics"][ACCURACY_KEY])
        - float(results[control][seed]["metrics"][ACCURACY_KEY])
        for seed in FORMAL_DEVELOPMENT_SEEDS
    ]
    cost_deltas = [
        float(results[treatment][seed]["profile"][COST_KEY])
        - float(results[control][seed]["profile"][COST_KEY])
        for seed in FORMAL_DEVELOPMENT_SEEDS
    ]
    dominates = (
        all(delta >= 0.0 for delta in accuracy_deltas)
        and all(delta <= 0.0 for delta in cost_deltas)
        and statistics.fmean(accuracy_deltas) > 0.0
        and statistics.fmean(cost_deltas) < 0.0
    )
    return {
        "treatment": treatment,
        "control": control,
        "paired_accuracy_deltas": accuracy_deltas,
        "paired_cost_deltas_ms": cost_deltas,
        "mean_accuracy_delta": statistics.fmean(accuracy_deltas),
        "mean_cost_delta_ms": statistics.fmean(cost_deltas),
        "strict_pareto_dominates": dominates,
    }


def finalize_results(
    *,
    run_root: Path,
    expected_commit: str,
    stage_jobs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    valid: dict[str, dict[int, dict[str, Any]]] = {
        arm: {} for arm in FORMAL_DEVELOPMENT_ARM_ORDER
    }
    failures: dict[str, Any] = {}
    population_hashes = set()
    checkpoint_paths = set()
    slurm_ids = set()
    for arm in FORMAL_DEVELOPMENT_ARM_ORDER:
        for seed in FORMAL_DEVELOPMENT_SEEDS:
            cell_root = run_root / formal_cell_relative_path(
                arm=arm,
                seed=seed,
            )
            result_path = cell_root / "stage_result.json"
            failure_path = cell_root / "stage_failure.json"
            key = f"{arm}/seed{seed}"
            if result_path.is_file() and failure_path.is_file():
                failures[key] = {"status": "AMBIGUOUS_RESULT_AND_FAILURE"}
                continue
            if failure_path.is_file():
                failure = read_json(failure_path)
                failures[key] = {
                    "status": failure.get("status"),
                    "exception_type": failure.get("exception_type"),
                    "exception_message": failure.get("exception_message"),
                    "failure_path": str(failure_path),
                    "failure_file_sha256": sha256_file(failure_path),
                    "failure_self_hash_valid": _self_hash_matches(
                        failure, field="failure_sha256"
                    ),
                }
                continue
            if not result_path.is_file():
                failures[key] = {"status": "MISSING_STAGE_RESULT"}
                continue
            try:
                result = validate_formal_stage_result(
                    read_json(result_path),
                    expected_arm=arm,
                    expected_seed=seed,
                    expected_commit=expected_commit,
                )
                _validate_artifacts(result, cell_root=cell_root, run_root=run_root)
                expected_job = str(stage_jobs[arm][str(seed)])
                train_job = str(
                    result["rendezvous"]["train"]["slurm_job_id"]
                )
                test_job = str(
                    result["rendezvous"]["test"]["slurm_job_id"]
                )
                if train_job != expected_job or test_job != expected_job:
                    raise ValueError(
                        "formal stage Slurm receipt differs from deployment"
                    )
                population_hashes.add(
                    result["telemetry_summary"]["population_sha256"]
                )
                checkpoint_paths.add(
                    result["checkpoint_receipt"]["path"]
                )
                slurm_ids.add(train_job)
                valid[arm][seed] = {
                    **result,
                    "result_path": str(result_path),
                    "result_file_sha256": sha256_file(result_path),
                }
            except Exception as error:
                failures[key] = {
                    "status": "INVALID_STAGE_RESULT",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "result_path": str(result_path),
                    "result_file_sha256": sha256_file(result_path),
                }
    all_passed = (
        not failures
        and all(
            set(valid[arm]) == set(FORMAL_DEVELOPMENT_SEEDS)
            for arm in FORMAL_DEVELOPMENT_ARM_ORDER
        )
        and len(population_hashes) == 1
        and len(checkpoint_paths) == 15
        and len(slurm_ids) == 15
    )
    if not all_passed:
        finalization: dict[str, Any] = {
            "schema_version": FORMAL_DEVELOPMENT_FINALIZATION_SCHEMA,
            "status": "INCOMPLETE_OFFICIAL_COMPARABLE_DEVELOPMENT_MATRIX",
            "decision": "DEVELOPMENT_MATRIX_INCOMPLETE_NO_PERFORMANCE_INFERENCE",
            "runtime_commit": expected_commit,
            "completed_cells": sum(len(cells) for cells in valid.values()),
            "expected_cells": 15,
            "failures": failures,
            "arm_seed_results": {},
            "aggregates": {},
            "paired_contrasts": {},
            "selector_decision": {},
            "all_fifteen_cells_passed": False,
            "development_selection_inference_allowed": False,
            "sealed_official_test_protocol_implementation_authorized": False,
            "official_protocol_freeze_authorized": False,
            "official_test_open_authorized": False,
            "official_test_opened": False,
            "paper_grade_result_record_emitted": False,
            "paper_claim_allowed": False,
        }
        finalization["finalization_sha256"] = canonical_sha256(finalization)
        return finalization

    compact: dict[str, dict[str, Any]] = {}
    aggregates: dict[str, Any] = {}
    for arm in FORMAL_DEVELOPMENT_ARM_ORDER:
        compact[arm] = {}
        for seed in FORMAL_DEVELOPMENT_SEEDS:
            result = valid[arm][seed]
            compact[arm][str(seed)] = {
                "metrics": dict(result["metrics"]),
                "profile": dict(result["profile"]),
                "telemetry_summary": dict(result["telemetry_summary"]),
                "checkpoint_receipt": dict(result["checkpoint_receipt"]),
                "stage_result_sha256": result["stage_result_sha256"],
                "result_file_sha256": result["result_file_sha256"],
                "slurm_job_id": result["rendezvous"]["train"][
                    "slurm_job_id"
                ],
            }
        aggregate_metrics = {
            metric: _aggregate(
                [
                    valid[arm][seed]["metrics"][metric]
                    for seed in FORMAL_DEVELOPMENT_SEEDS
                ]
            )
            for metric in valid[arm][FORMAL_DEVELOPMENT_SEEDS[0]][
                "metrics"
            ]
        }
        aggregate_cost = {
            key: _aggregate(
                [
                    valid[arm][seed]["profile"][key]
                    for seed in FORMAL_DEVELOPMENT_SEEDS
                ]
            )
            for key in (
                "model_and_postprocess_p50_ms",
                "model_and_postprocess_p95_ms",
                "window_wall_p50_ms",
                "window_wall_p95_ms",
                "peak_allocated_mb",
            )
        }
        aggregates[arm] = {
            "metrics": aggregate_metrics,
            "development_cost": aggregate_cost,
        }
    eligibility = {
        arm: _selector_eligibility(valid, arm=arm)
        for arm in SELECTOR_ARMS
    }
    st_over_pl = _strict_pareto_dominates(
        valid,
        treatment=SELECTOR_ARMS[0],
        control=SELECTOR_ARMS[1],
    )
    pl_over_st = _strict_pareto_dominates(
        valid,
        treatment=SELECTOR_ARMS[1],
        control=SELECTOR_ARMS[0],
    )
    selected_arm = None
    if (
        eligibility[SELECTOR_ARMS[0]]["all_three_seeds_passed"]
        and st_over_pl["strict_pareto_dominates"]
        and not pl_over_st["strict_pareto_dominates"]
    ):
        selected_arm = SELECTOR_ARMS[0]
    elif (
        eligibility[SELECTOR_ARMS[1]]["all_three_seeds_passed"]
        and pl_over_st["strict_pareto_dominates"]
        and not st_over_pl["strict_pareto_dominates"]
    ):
        selected_arm = SELECTOR_ARMS[1]
    authorized = selected_arm is not None
    finalization = {
        "schema_version": FORMAL_DEVELOPMENT_FINALIZATION_SCHEMA,
        "status": "COMPLETE_OFFICIAL_COMPARABLE_DEVELOPMENT_MATRIX",
        "decision": (
            "DEVELOPMENT_METHOD_FREEZE_CANDIDATE_AUTHORIZED"
            if authorized
            else "DEVELOPMENT_SELECTION_HOLD_NO_OFFICIAL_TEST"
        ),
        "runtime_commit": expected_commit,
        "completed_cells": 15,
        "expected_cells": 15,
        "failures": {},
        "arm_seed_results": compact,
        "aggregates": aggregates,
        "paired_contrasts": {
            "selector_eligibility": eligibility,
            "st_over_pl": st_over_pl,
            "pl_over_st": pl_over_st,
        },
        "selector_decision": {
            "selected_arm": selected_arm,
            "accuracy_metric": ACCURACY_KEY,
            "development_cost_metric": COST_KEY,
            "all_seed_control_improvement_required": True,
            "strict_pareto_dominance_required": True,
            "geometry_zoom_allowed": False,
        },
        "all_fifteen_cells_passed": True,
        "cross_cell_population_consistent": True,
        "development_selection_inference_allowed": True,
        "sealed_official_test_protocol_implementation_authorized": authorized,
        "official_protocol_freeze_authorized": False,
        "official_test_open_authorized": False,
        "official_test_opened": False,
        "development_metrics_are_paper_results": False,
        "paper_grade_result_record_emitted": False,
        "paper_claim_allowed": False,
    }
    finalization["finalization_sha256"] = canonical_sha256(finalization)
    return finalization


def _scheduler_states(job_ids: Sequence[str]) -> dict[str, dict[str, str]]:
    expected = tuple(str(job_id) for job_id in job_ids)
    if any(not job_id.isdigit() for job_id in expected) or len(set(expected)) != len(
        expected
    ):
        raise ValueError("P1 scheduler job population is malformed")
    completed = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(expected),
            "--format=JobIDRaw,State,ExitCode",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "sacct P1 terminal query failed")
    rows: dict[str, dict[str, str]] = {}
    for line in completed.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 3 or fields[0] not in expected or fields[0] in rows:
            continue
        rows[fields[0]] = {"state": fields[1], "exit_code": fields[2]}
    if set(rows) != set(expected):
        raise ValueError("P1 scheduler query omitted a predecessor job")
    return rows


def finalize_p1_results(
    *,
    run_root: Path,
    expected_commit: str,
    deployment: Mapping[str, Any],
    scheduler_states: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    jobs = deployment.get("jobs")
    stage_jobs = jobs.get("stage") if isinstance(jobs, Mapping) else None
    cost_jobs = jobs.get("cost") if isinstance(jobs, Mapping) else None
    preflight_job = str(jobs.get("runtime_preflight", "")) if isinstance(jobs, Mapping) else ""
    if not isinstance(stage_jobs, Mapping) or not isinstance(cost_jobs, Mapping):
        raise ValueError("P1 deployment lacks stage/cost jobs")
    expected_jobs = [
        preflight_job,
        *(
            str(stage_jobs[arm][str(P1_DEVELOPMENT_SEED)])
            for arm in P1_MATCHED_RUNNER_ARM_ORDER
        ),
        *(str(cost_jobs[leaf_id]) for leaf_id in P1_COST_LEAF_SPECS),
    ]
    failures: dict[str, Any] = {}
    for job_id in expected_jobs:
        row = scheduler_states.get(job_id)
        if (
            not isinstance(row, Mapping)
            or row.get("state") != "COMPLETED"
            or row.get("exit_code") != "0:0"
        ):
            failures[f"scheduler/{job_id}"] = {
                "status": "NONTERMINAL_OR_NONZERO_PREDECESSOR",
                "scheduler": dict(row) if isinstance(row, Mapping) else None,
            }

    valid: dict[str, dict[str, Any]] = {}
    population_hashes: set[str] = set()
    checkpoint_paths: set[str] = set()
    stage_slurm_ids: set[str] = set()
    runtime_fingerprints: set[str] = set()
    shared = deployment.get("shared_official_baseline")
    try:
        if not isinstance(shared, Mapping):
            raise ValueError("P1 deployment lacks the shared official DO receipt")
        shared_path = Path(str(shared["receipt_path"])).resolve()
        shared_receipt = validate_p1_shared_official_baseline_receipt(shared_path)
        if (
            sha256_file(shared_path) != shared.get("receipt_file_sha256")
            or shared_receipt.get("receipt_sha256") != shared.get("receipt_sha256")
        ):
            raise ValueError("shared official DO receipt changed")
        checkpoint_paths.add(str(shared_receipt["checkpoint"]["path"]))
        valid["DO"] = {
            "external_shared_official_baseline": True,
            "metrics": dict(shared_receipt["metrics"]),
            "shared_receipt_path": str(shared_path),
            "shared_receipt_file_sha256": sha256_file(shared_path),
            "shared_receipt_sha256": shared_receipt["receipt_sha256"],
            "checkpoint_receipt": dict(shared_receipt["checkpoint"]),
            "report_only": True,
        }
    except Exception as error:
        failures["accuracy/DO"] = {
            "status": "INVALID_OR_MISSING_SHARED_OFFICIAL_BASELINE",
            "exception_type": type(error).__name__,
            "exception_message": str(error),
        }

    for arm in P1_MATCHED_RUNNER_ARM_ORDER:
        cell_root = run_root / _p1_cell_relative_path(
            arm=arm, seed=P1_DEVELOPMENT_SEED
        )
        result_path = cell_root / "stage_result.json"
        failure_path = cell_root / "stage_failure.json"
        key = f"accuracy/{arm}"
        if result_path.is_file() and failure_path.is_file():
            failures[key] = {"status": "AMBIGUOUS_RESULT_AND_FAILURE"}
            continue
        if failure_path.is_file():
            failures[key] = {
                "status": "FAILED_STAGE",
                "failure_path": str(failure_path),
                "failure_file_sha256": sha256_file(failure_path),
            }
            continue
        if not result_path.is_file():
            failures[key] = {"status": "MISSING_STAGE_RESULT"}
            continue
        try:
            result = validate_formal_stage_result(
                read_json(result_path),
                expected_arm=arm,
                expected_seed=P1_DEVELOPMENT_SEED,
                expected_commit=expected_commit,
            )
            _validate_artifacts(result, cell_root=cell_root, run_root=run_root)
            expected_job = str(stage_jobs[arm][str(P1_DEVELOPMENT_SEED)])
            if (
                str(result["rendezvous"]["train"]["slurm_job_id"]) != expected_job
                or str(result["rendezvous"]["test"]["slurm_job_id"]) != expected_job
            ):
                raise ValueError("P1 accuracy Slurm identity differs from deployment")
            runtime = result["runtime_attestation"]
            for path_field, hash_field in (
                ("preflight_path", "preflight_file_sha256"),
                ("leaf_path", "leaf_file_sha256"),
            ):
                path = Path(str(runtime[path_field])).resolve()
                if not path.is_file() or sha256_file(path) != runtime[hash_field]:
                    raise ValueError("P1 accuracy runtime attestation changed")
            population_hashes.add(
                result["telemetry_summary"]["physical_population_sha256"]
            )
            checkpoint_paths.add(result["checkpoint_receipt"]["path"])
            stage_slurm_ids.add(expected_job)
            runtime_fingerprints.add(runtime["runtime_class_fingerprint"])
            valid[arm] = {
                **result,
                "result_path": str(result_path),
                "result_file_sha256": sha256_file(result_path),
            }
        except Exception as error:
            failures[key] = {
                "status": "INVALID_STAGE_RESULT",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            }

    cost_rows: dict[str, list[dict[str, Any]]] = {}
    cost_receipts: dict[str, dict[str, Any]] = {}
    cost_population_hashes: set[str] = set()
    for leaf_id in P1_COST_LEAF_SPECS:
        leaf_root = run_root / p1_cost_leaf_relative_path(leaf_id)
        receipt_path = leaf_root / "receipt.json"
        failure_path = leaf_root / "cost_failure.json"
        key = f"cost/{leaf_id}"
        if receipt_path.is_file() and failure_path.is_file():
            failures[key] = {"status": "AMBIGUOUS_RESULT_AND_FAILURE"}
            continue
        if failure_path.is_file():
            failures[key] = {
                "status": "FAILED_COST_LEAF",
                "failure_path": str(failure_path),
                "failure_file_sha256": sha256_file(failure_path),
            }
            continue
        if not receipt_path.is_file():
            failures[key] = {"status": "MISSING_COST_LEAF"}
            continue
        try:
            receipt = validate_p1_cost_leaf_receipt(
                read_json(receipt_path),
                expected_leaf_id=leaf_id,
                expected_job_id=str(cost_jobs[leaf_id]),
            )
            artifacts = receipt["artifacts"]
            for artifact in artifacts.values():
                path = Path(str(artifact["path"])).resolve()
                if (
                    not path.is_file()
                    or not _inside(path, leaf_root)
                    or sha256_file(path) != artifact["sha256"]
                ):
                    raise ValueError("P1 cost artifact changed")
            rows = read_jsonl_objects(
                artifacts["measured_samples"]["path"],
                label=f"P1 cost {leaf_id} samples",
            )
            checked_rows = validate_p1_cost_rows(rows, leaf_id=leaf_id)
            warmup_rows = read_jsonl_objects(
                artifacts["warmup_identities"]["path"],
                label=f"P1 cost {leaf_id} warmup identities",
            )
            validate_p1_cost_warmup_rows(warmup_rows, leaf_id=leaf_id)
            runtime = receipt.get("runtime_attestation")
            if (
                not isinstance(runtime, Mapping)
                or runtime.get("runtime_class_fingerprint") not in runtime_fingerprints
            ):
                raise ValueError("P1 cost runtime class differs from accuracy/preflight")
            cost_population_hashes.add(str(receipt["population_sha256"]))
            cost_rows[leaf_id] = checked_rows
            cost_receipts[leaf_id] = {
                "receipt_path": str(receipt_path),
                "receipt_file_sha256": sha256_file(receipt_path),
                "receipt_sha256": receipt["receipt_sha256"],
                "slurm_job_id": receipt["slurm_job_id"],
            }
        except Exception as error:
            failures[key] = {
                "status": "INVALID_COST_LEAF",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            }

    complete_shape = (
        not failures
        and set(valid) == set(P1_FIRST_SCREEN_ARM_ORDER)
        and set(cost_rows) == set(P1_COST_LEAF_SPECS)
        and len(population_hashes) == 1
        and len(cost_population_hashes) == 1
        and population_hashes == cost_population_hashes
        and len(checkpoint_paths) == 5
        and len(stage_slurm_ids) == 4
        and len(runtime_fingerprints) == 1
    )
    if not complete_shape:
        finalization: dict[str, Any] = {
            "schema_version": "zoomtoken_p1_finalization_v001",
            "study_id": P1_STUDY_ID,
            "status": "INVALID_P1_MATRIX",
            "decision": "NO_SURVIVOR_INVALID_P1",
            "runtime_commit": expected_commit,
            "completed_accuracy_cells": len(valid),
            "expected_accuracy_cells": 5,
            "completed_cost_leaves": len(cost_rows),
            "expected_cost_leaves": 8,
            "failures": failures,
            "partial_arm_conclusion_allowed": False,
            "accuracy_gate": {},
            "cost_gate": {},
            "report_only_diagnostics": {},
            "q_survives_p1": False,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }
        finalization["finalization_sha256"] = canonical_sha256(finalization)
        return finalization

    annotation_receipt = deployment.get("input_receipts", {}).get(
        "GEOROUTE_DEVELOPMENT_ANNOTATION"
    )
    try:
        if not isinstance(annotation_receipt, Mapping):
            raise ValueError("P1 deployment lacks its annotation receipt")
        annotation_path = Path(str(annotation_receipt["path"])).resolve()
        if (
            not annotation_path.is_file()
            or sha256_file(annotation_path) != annotation_receipt.get("sha256")
        ):
            raise ValueError("P1 development annotation changed")
        annotation = read_json(annotation_path)
        diagnostics: dict[str, dict[str, Any]] = {}
        q_telemetry = read_json(valid["Q"]["telemetry_path"])
        telemetry_records = q_telemetry.get("records")
        if not isinstance(telemetry_records, list):
            raise ValueError("P1 Q telemetry lacks source-video identities")
        expected_video_ids = tuple(
            sorted(
                {
                    str(record.get("video_id", ""))
                    for record in telemetry_records
                    if isinstance(record, Mapping)
                }
            )
        )
        if len(expected_video_ids) != VIDEO_CLUSTERS or any(
            not video_id for video_id in expected_video_ids
        ):
            raise ValueError("P1 telemetry does not bind the frozen 40 videos")
        for arm in P1_MATCHED_RUNNER_ARM_ORDER:
            prediction = read_json(valid[arm]["prediction_path"])
            arm_diagnostics = evaluate_p1_report_only_diagnostics(
                annotation,
                prediction,
                expected_video_ids=expected_video_ids,
            )
            diagnostics[arm] = arm_diagnostics
        diagnostics["DO"] = {
            "source": "shared_official_baseline_final_receipt",
            "report_only": True,
            "matched_40_video_diagnostics_available": False,
            "reason": "shared official receipt uses its frozen official population",
            "shared_receipt_sha256": valid["DO"]["shared_receipt_sha256"],
        }
        diagnostic_differences = {
            comparator: paired_p1_diagnostic_bootstrap(
                diagnostics["Q"], diagnostics[comparator]
            )
            for comparator in ("DN", "U", "R")
        }
        diagnostic_differences["DO"] = {
            "report_only": True,
            "available": False,
            "reason": "no cross-population diagnostic contrast",
        }
        cost_analysis = analyze_p1_cost_leaves(cost_rows)
    except Exception as error:
        finalization = {
            "schema_version": "zoomtoken_p1_finalization_v001",
            "study_id": P1_STUDY_ID,
            "status": "INVALID_P1_MATRIX",
            "decision": "NO_SURVIVOR_INVALID_P1",
            "runtime_commit": expected_commit,
            "completed_accuracy_cells": 5,
            "expected_accuracy_cells": 5,
            "completed_cost_leaves": 8,
            "expected_cost_leaves": 8,
            "failures": {
                "final_analysis": {
                    "status": "MALFORMED_OR_CONTAMINATED_ANALYSIS",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }
            },
            "partial_arm_conclusion_allowed": False,
            "accuracy_gate": {},
            "cost_gate": {},
            "report_only_diagnostics": {},
            "q_survives_p1": False,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }
        finalization["finalization_sha256"] = canonical_sha256(finalization)
        return finalization

    q_map_070 = float(valid["Q"]["metrics"]["mAP@0.7"])
    accuracy_gate = {
        "metric": "official_mAP@0.7",
        "q": q_map_070,
        "u": float(valid["U"]["metrics"]["mAP@0.7"]),
        "r": float(valid["R"]["metrics"]["mAP@0.7"]),
    }
    accuracy_gate["q_strictly_beats_u"] = q_map_070 > accuracy_gate["u"]
    accuracy_gate["q_strictly_beats_r"] = q_map_070 > accuracy_gate["r"]
    accuracy_gate["passed"] = (
        accuracy_gate["q_strictly_beats_u"]
        and accuracy_gate["q_strictly_beats_r"]
    )
    cost_gate = {
        "dense_denominator": "DN",
        "limit": P1_COST_RATIO_LIMIT,
        "equality_passes": True,
        "tolerance": 0.0,
        "analysis": cost_analysis,
        "passed": cost_analysis["q_over_dn_cost_gate_passed"],
        "do_mandatory_report_only": True,
    }
    survives = bool(accuracy_gate["passed"] and cost_gate["passed"])
    compact = {
        arm: {
            "metrics": dict(valid[arm]["metrics"]),
            "stage_result_sha256": valid[arm]["stage_result_sha256"],
            "result_file_sha256": valid[arm]["result_file_sha256"],
            "slurm_job_id": valid[arm]["rendezvous"]["train"]["slurm_job_id"],
        }
        for arm in P1_MATCHED_RUNNER_ARM_ORDER
    }
    compact["DO"] = {
        "metrics": dict(valid["DO"]["metrics"]),
        "report_only": True,
        "shared_receipt_path": valid["DO"]["shared_receipt_path"],
        "shared_receipt_file_sha256": valid["DO"]["shared_receipt_file_sha256"],
        "shared_receipt_sha256": valid["DO"]["shared_receipt_sha256"],
        "scheduled_by_p1": False,
    }
    finalization = {
        "schema_version": "zoomtoken_p1_finalization_v001",
        "study_id": P1_STUDY_ID,
        "status": "COMPLETE_VALID_P1_MATRIX",
        "decision": "Q_CORE_P1_SURVIVES" if survives else "STOP_Q_CORE_P1",
        "runtime_commit": expected_commit,
        "completed_accuracy_cells": 5,
        "expected_accuracy_cells": 5,
        "completed_cost_leaves": 8,
        "expected_cost_leaves": 8,
        "failures": {},
        "arm_results": compact,
        "cost_leaf_receipts": cost_receipts,
        "accuracy_gate": accuracy_gate,
        "cost_gate": cost_gate,
        "report_only_diagnostics": {
            "per_arm": diagnostics,
            "paired_q_minus_comparator": diagnostic_differences,
            "short_actions_affect_gate": False,
            "boundary_diagnostics_affect_gate": False,
            "high_iou_decomposition_affects_gate": False,
        },
        "partial_arm_conclusion_allowed": False,
        "q_survives_p1": survives,
        "conditional_g_n_f_open": survives,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    finalization["finalization_sha256"] = canonical_sha256(finalization)
    return finalization


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def _run_main(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(run_root, boundary):
        raise ValueError("formal finalizer root leaves remote boundary")
    expected_commit = str(args.expected_commit).lower()
    if _git_output("rev-parse", "HEAD").lower() != expected_commit:
        raise RuntimeError("formal finalizer source commit changed")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("formal finalizer requires a clean source snapshot")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if not slurm_job_id.isdigit():
        raise RuntimeError("formal finalizer requires Slurm")
    deployment_path = run_root / "control" / "deployment.json"
    deployment = read_json(deployment_path)
    jobs = deployment.get("jobs")
    stage_jobs = jobs.get("stage") if isinstance(jobs, Mapping) else None
    p1_mode = deployment.get("mode") == "p1"
    if p1_mode and np.__version__ != "1.23.5":
        raise RuntimeError("P1 bootstrap requires frozen NumPy 1.23.5")
    if p1_mode:
        validate_p1_deployment_shape(deployment)
    expected_status = (
        "SUBMITTED_ZOOMTOKEN_P1_DNURQ_MATRIX"
        if p1_mode
        else "SUBMITTED_OFFICIAL_COMPARABLE_DEVELOPMENT_MATRIX"
    )
    if (
        deployment.get("schema_version")
        != FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA
        or deployment.get("status")
        != expected_status
        or deployment.get("runtime_commit") != expected_commit
        or not _self_hash_matches(deployment, field="deployment_sha256")
        or not isinstance(jobs, Mapping)
        or not isinstance(stage_jobs, Mapping)
        or str(jobs.get("finalizer", "")) != slurm_job_id
    ):
        raise RuntimeError("formal deployment receipt is invalid")
    submission_path = run_root / "control" / "finalizer_submission.json"
    submission = read_json(submission_path)
    predecessor_ids = (
        [
            str(jobs["runtime_preflight"]),
            *(
                str(stage_jobs[arm][str(P1_DEVELOPMENT_SEED)])
                for arm in P1_MATCHED_RUNNER_ARM_ORDER
            ),
            *(
                str(jobs["cost"][leaf_id])
                for leaf_id in P1_COST_LEAF_SPECS
            ),
        ]
        if p1_mode
        else [
            str(stage_jobs[arm][str(seed)])
            for arm in FORMAL_DEVELOPMENT_ARM_ORDER
            for seed in FORMAL_DEVELOPMENT_SEEDS
        ]
    )
    expected_submission_status = (
        "SUBMITTED_P1_FINALIZER_AFTERANY"
        if p1_mode
        else "SUBMITTED_DEVELOPMENT_FINALIZER_AFTERANY"
    )
    submitted_predecessors = tuple(
        map(str, submission.get("predecessor_job_ids", ()))
    )
    predecessor_binding_valid = (
        submitted_predecessors == tuple(predecessor_ids)
        if p1_mode
        else set(submitted_predecessors) == set(predecessor_ids)
    )
    if (
        submission.get("schema_version")
        != FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA
        or submission.get("status")
        != expected_submission_status
        or submission.get("runtime_commit") != expected_commit
        or submission.get("deployment_file_sha256")
        != sha256_file(deployment_path)
        or submission.get("finalizer_job_id") != slurm_job_id
        or submission.get("dependency_type") != "afterany"
        or not predecessor_binding_valid
        or not _self_hash_matches(submission, field="receipt_sha256")
    ):
        raise RuntimeError("formal finalizer submission receipt is invalid")
    protocol_path = run_root / "control" / "protocol_manifest.json"
    protocol = validate_protocol_manifest(read_json(protocol_path))
    if (
        sha256_file(protocol_path)
        != deployment.get("protocol_manifest_file_sha256")
        or protocol.get("protocol_sha256")
        != deployment.get("protocol_sha256")
    ):
        raise RuntimeError("formal protocol manifest changed")
    finalization = (
        finalize_p1_results(
            run_root=run_root,
            expected_commit=expected_commit,
            deployment=deployment,
            scheduler_states=_scheduler_states(predecessor_ids),
        )
        if p1_mode
        else finalize_results(
            run_root=run_root,
            expected_commit=expected_commit,
            stage_jobs=stage_jobs,
        )
    )
    finalization["deployment_path"] = str(deployment_path)
    finalization["deployment_file_sha256"] = sha256_file(deployment_path)
    finalization["finalizer_submission_path"] = str(submission_path)
    finalization["finalizer_submission_file_sha256"] = sha256_file(
        submission_path
    )
    finalization["protocol_manifest_path"] = str(protocol_path)
    finalization["protocol_manifest_file_sha256"] = sha256_file(protocol_path)
    finalization.pop("finalization_sha256")
    finalization["finalization_sha256"] = canonical_sha256(finalization)
    output = run_root / "control" / "finalization.json"
    if output.exists():
        raise FileExistsError("formal finalization already exists")
    _atomic_write_json(output, finalization)
    print(json.dumps(finalization, indent=2, sort_keys=True))
    if p1_mode:
        return 1 if finalization["decision"] == "NO_SURVIVOR_INVALID_P1" else 0
    return 0 if finalization["all_fifteen_cells_passed"] else 1


def _write_failsafe(args: argparse.Namespace, error: BaseException) -> None:
    run_root = args.run_root.resolve()
    boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(run_root, boundary):
        return
    output = run_root / "control" / "finalization.json"
    if output.exists():
        return
    trace = traceback.format_exc()
    deployment_path = run_root / "control" / "deployment.json"
    p1_mode = False
    if deployment_path.is_file():
        try:
            p1_mode = read_json(deployment_path).get("mode") == "p1"
        except Exception:
            p1_mode = False
    payload: dict[str, Any] = {
        "schema_version": (
            "zoomtoken_p1_finalization_v001"
            if p1_mode
            else FORMAL_DEVELOPMENT_FINALIZATION_SCHEMA
        ),
        "status": (
            "INVALID_P1_MATRIX"
            if p1_mode
            else "FAILED_OFFICIAL_COMPARABLE_DEVELOPMENT_FINALIZER"
        ),
        "decision": (
            "NO_SURVIVOR_INVALID_P1"
            if p1_mode
            else "DEVELOPMENT_MATRIX_INCOMPLETE_NO_PERFORMANCE_INFERENCE"
        ),
        "expected_runtime_commit": str(args.expected_commit).lower(),
        "observed_runtime_commit": (
            _git_output("rev-parse", "HEAD").lower()
            if (ROOT / ".git").exists()
            else None
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "exception_type": type(error).__name__,
        "exception_message": str(error)[:2000],
        "traceback_sha256": __import__("hashlib").sha256(
            trace.encode("utf-8", errors="replace")
        ).hexdigest(),
        "arm_seed_results": {},
        "aggregates": {},
        "paired_contrasts": {},
        "development_selection_inference_allowed": False,
        "sealed_official_test_protocol_implementation_authorized": False,
        "official_test_open_authorized": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    if p1_mode:
        payload.update(
            {
                "study_id": P1_STUDY_ID,
                "partial_arm_conclusion_allowed": False,
                "q_survives_p1": False,
            }
        )
    payload["finalization_sha256"] = canonical_sha256(payload)
    _atomic_write_json(output, payload)


def main() -> int:
    args = _parse_args()
    try:
        return _run_main(args)
    except BaseException as error:
        try:
            _write_failsafe(args, error)
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
