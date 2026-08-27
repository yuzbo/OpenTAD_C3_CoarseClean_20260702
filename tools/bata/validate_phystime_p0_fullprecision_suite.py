#!/usr/bin/env python3
"""Validate and aggregate the complete four-job PhysTime P0 replay suite."""

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict, deque
from pathlib import Path


EXPECTED_RUNS = {
    "selected_online": ("selected_axis", "online"),
    "selected_ema": ("selected_axis", "ema"),
    "physical_online": ("physical_metric", "online"),
    "physical_ema": ("physical_metric", "ema"),
}
EXPECTED_MODES = (
    "legacy_unfiltered",
    "legacy_filtered",
    "fullprecision_unfiltered",
    "fullprecision_filtered",
)
EXPECTED_WITHIN_COMPARISONS = {
    "rounding_effect_unfiltered": (
        "fullprecision_unfiltered",
        "legacy_unfiltered",
    ),
    "rounding_effect_filtered": (
        "fullprecision_filtered",
        "legacy_filtered",
    ),
    "validity_filter_effect_legacy": (
        "legacy_filtered",
        "legacy_unfiltered",
    ),
    "validity_filter_effect_fullprecision": (
        "fullprecision_filtered",
        "fullprecision_unfiltered",
    ),
}
METRIC_EPSILON = 1.0e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate all four P0 replay completions and aggregate diagnostics."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path, description):
    path = Path(path)
    require(path.is_file(), f"{description} is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_completion_run_dir(completion, expected_run_dir, description):
    observed = Path(completion.get("run_dir", "")).resolve()
    expected = Path(expected_run_dir).resolve()
    require(
        observed == expected,
        f"{description} completion run_dir mismatch: "
        f"observed={observed}, expected={expected}",
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finite_metrics(payload):
    metrics = {key: float(value) for key, value in payload.items()}
    require(metrics, "metric dictionary is empty")
    require(
        all(math.isfinite(value) for value in metrics.values()),
        "metric dictionary contains a non-finite value",
    )
    return metrics


def metric_delta(lhs, rhs):
    lhs = finite_metrics(lhs)
    rhs = finite_metrics(rhs)
    require(lhs.keys() == rhs.keys(), "metric keys differ across P0 suite arms")
    return {
        "fraction": {
            key: lhs[key] - rhs[key] for key in sorted(lhs)
        },
        "percentage_points": {
            key: 100.0 * (lhs[key] - rhs[key]) for key in sorted(lhs)
        },
    }


def require_metric_match(observed, expected, description):
    observed = finite_metrics(observed)
    expected = finite_metrics(expected)
    require(
        observed.keys() == expected.keys(),
        f"{description} metric keys differ",
    )
    require(
        all(
            abs(observed[key] - expected[key]) <= METRIC_EPSILON
            for key in observed
        ),
        f"{description} metric values differ",
    )


def within_run_delta_report(mode_metrics, prediction_counts):
    report = {}
    for name, (lhs_name, rhs_name) in EXPECTED_WITHIN_COMPARISONS.items():
        delta = metric_delta(
            mode_metrics[lhs_name],
            mode_metrics[rhs_name],
        )
        report[name] = {
            "status": "comparable",
            "lhs": lhs_name,
            "rhs": rhs_name,
            "metric_delta_fraction": delta["fraction"],
            "metric_delta_percentage_points": delta["percentage_points"],
            "prediction_count_delta": (
                prediction_counts[lhs_name]
                - prediction_counts[rhs_name]
            ),
        }
    return report


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values):
    values = [float(value) for value in values]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def flatten_predictions(prediction_payload):
    flattened = []
    for video_name, detections in prediction_payload["results"].items():
        for rank, detection in enumerate(detections):
            start, end = (float(value) for value in detection["segment"])
            score = float(detection["score"])
            require(
                math.isfinite(start)
                and math.isfinite(end)
                and math.isfinite(score)
                and end > start,
                "suite diagnostics received an invalid prediction",
            )
            flattened.append(
                {
                    "video": video_name,
                    "label": detection["label"],
                    "start": start,
                    "end": end,
                    "score": score,
                    "rank": rank,
                }
            )
    return flattened


def rounded_identity(item):
    return (
        item["video"],
        item["label"],
        round(item["start"], 2),
        round(item["end"], 2),
    )


def _strict_rounded_identity_match(lhs, rhs):
    rhs_queues = defaultdict(deque)
    for item in sorted(
        rhs,
        key=lambda value: (
            rounded_identity(value),
            -value["score"],
            value["rank"],
        ),
    ):
        rhs_queues[rounded_identity(item)].append(item)

    matched = []
    lhs_unmatched = []
    for item in sorted(
        lhs,
        key=lambda value: (
            rounded_identity(value),
            -value["score"],
            value["rank"],
        ),
    ):
        queue = rhs_queues[rounded_identity(item)]
        if queue:
            matched.append((item, queue.popleft()))
        else:
            lhs_unmatched.append(item)
    rhs_unmatched = [
        item for queue in rhs_queues.values() for item in queue
    ]
    return {
        "identity_contract": (
            "video+label+segment rounded to 2 decimals; duplicate identities "
            "paired by descending score then original rank"
        ),
        "matched_prediction_count": len(matched),
        "lhs_only_count": len(lhs_unmatched),
        "rhs_only_count": len(rhs_unmatched),
    }


def _greedy_iou_match(lhs, rhs, minimum_iou=0.5):
    lhs_groups = defaultdict(list)
    rhs_groups = defaultdict(list)
    for index, item in enumerate(lhs):
        lhs_groups[(item["video"], item["label"])].append((index, item))
    for index, item in enumerate(rhs):
        rhs_groups[(item["video"], item["label"])].append((index, item))
    candidates = []
    shared_groups = set(lhs_groups).intersection(rhs_groups)
    for key in sorted(shared_groups, key=lambda value: (value[0], str(value[1]))):
        for lhs_index, lhs_item in lhs_groups[key]:
            for rhs_index, rhs_item in rhs_groups[key]:
                iou = segment_iou(
                    lhs_item["start"],
                    lhs_item["end"],
                    rhs_item["start"],
                    rhs_item["end"],
                )
                if iou >= minimum_iou:
                    candidates.append(
                        (
                            -iou,
                            abs(lhs_item["score"] - rhs_item["score"]),
                            lhs_item["rank"],
                            rhs_item["rank"],
                            lhs_index,
                            rhs_index,
                        )
                    )
    lhs_used = set()
    rhs_used = set()
    matched = []
    for negative_iou, _, _, _, lhs_index, rhs_index in sorted(candidates):
        if lhs_index in lhs_used or rhs_index in rhs_used:
            continue
        lhs_used.add(lhs_index)
        rhs_used.add(rhs_index)
        matched.append((lhs[lhs_index], rhs[rhs_index], -negative_iou))
    lhs_unmatched = [item for index, item in enumerate(lhs) if index not in lhs_used]
    rhs_unmatched = [item for index, item in enumerate(rhs) if index not in rhs_used]
    return matched, lhs_unmatched, rhs_unmatched


def compare_prediction_decisions(lhs_payload, rhs_payload):
    lhs = flatten_predictions(lhs_payload)
    rhs = flatten_predictions(rhs_payload)
    strict_identity = _strict_rounded_identity_match(lhs, rhs)
    matched, lhs_unmatched, rhs_unmatched = _greedy_iou_match(lhs, rhs)
    start_displacement = [
        abs(lhs_item["start"] - rhs_item["start"])
        for lhs_item, rhs_item, _ in matched
    ]
    end_displacement = [
        abs(lhs_item["end"] - rhs_item["end"])
        for lhs_item, rhs_item, _ in matched
    ]
    score_displacement = [
        abs(lhs_item["score"] - rhs_item["score"])
        for lhs_item, rhs_item, _ in matched
    ]
    rank_displacement = [
        abs(lhs_item["rank"] - rhs_item["rank"])
        for lhs_item, rhs_item, _ in matched
    ]
    matched_ious = [iou for _, _, iou in matched]
    return {
        "identity_contract": (
            "Primary suppression/boundary diagnostic uses deterministic "
            "one-to-one same-video same-class IoU matching at IoU>=0.5. "
            "The prior strict rounded identity diagnostic is retained "
            "separately."
        ),
        "strict_rounded_identity": strict_identity,
        "iou_match_threshold": 0.5,
        "lhs_prediction_count": len(lhs),
        "rhs_prediction_count": len(rhs),
        "matched_prediction_count": len(matched),
        "lhs_only_count": len(lhs_unmatched),
        "rhs_only_count": len(rhs_unmatched),
        "matched_fraction_of_lhs": (
            len(matched) / len(lhs) if lhs else 1.0
        ),
        "matched_fraction_of_rhs": (
            len(matched) / len(rhs) if rhs else 1.0
        ),
        "matched_segment_iou": summarize(matched_ious),
        "boundary_start_abs_seconds": summarize(start_displacement),
        "boundary_end_abs_seconds": summarize(end_displacement),
        "score_abs_delta": summarize(score_displacement),
        "rank_abs_delta": summarize(rank_displacement),
    }


def remove_duplicate_annotations(annotations, tolerance=1.0e-3):
    valid = []
    for annotation in annotations:
        start, end = (float(value) for value in annotation["segment"])
        label = annotation["label"]
        if end - start <= 0:
            continue
        duplicate = any(
            abs(start - item["start"]) <= tolerance
            and abs(end - item["end"]) <= tolerance
            and label == item["label"]
            for item in valid
        )
        if not duplicate:
            valid.append(
                {
                    "start": start,
                    "end": end,
                    "label": label,
                }
            )
    return valid


def load_ground_truth(gate):
    contracts = {
        arm: gate["runtime_configs"][arm]["evaluation_contract"]
        for arm in ("selected_axis", "physical_metric")
    }
    require(
        contracts["selected_axis"] == contracts["physical_metric"],
        "P0 arms do not share one evaluation contract",
    )
    contract = contracts["selected_axis"]
    blocked = set()
    if contract["blocked_videos"] is not None:
        blocked = set(
            read_json(contract["blocked_videos"], "blocked video list")
        )
    payload = read_json(
        contract["ground_truth_filename"],
        "P0 ground truth",
    )
    ground_truth = []
    for video_name, video in payload["database"].items():
        if (
            video["subset"] != contract["subset"]
            or video_name in blocked
        ):
            continue
        for annotation in remove_duplicate_annotations(video["annotations"]):
            ground_truth.append(
                {
                    "video": video_name,
                    **annotation,
                    "duration": annotation["end"] - annotation["start"],
                }
            )
    require(ground_truth, "P0 ground truth subset is empty")
    return ground_truth, contract


def segment_iou(lhs_start, lhs_end, rhs_start, rhs_end):
    intersection = max(
        0.0,
        min(lhs_end, rhs_end) - max(lhs_start, rhs_start),
    )
    union = (lhs_end - lhs_start) + (rhs_end - rhs_start) - intersection
    return intersection / union if union > 0 else 0.0


def proposal_recall_diagnostics(prediction_payload, ground_truth):
    predictions = defaultdict(list)
    for item in flatten_predictions(prediction_payload):
        predictions[(item["video"], item["label"])].append(item)
    durations = [item["duration"] for item in ground_truth]
    short_boundary = percentile(durations, 0.25)
    long_boundary = percentile(durations, 0.75)
    strata = {
        "all": [],
        "short_q1": [],
        "middle_q2_q3": [],
        "long_q4": [],
    }
    for item in ground_truth:
        candidates = predictions[(item["video"], item["label"])]
        best_iou = max(
            (
                segment_iou(
                    item["start"],
                    item["end"],
                    candidate["start"],
                    candidate["end"],
                )
                for candidate in candidates
            ),
            default=0.0,
        )
        strata["all"].append(best_iou)
        if item["duration"] <= short_boundary:
            strata["short_q1"].append(best_iou)
        elif item["duration"] > long_boundary:
            strata["long_q4"].append(best_iou)
        else:
            strata["middle_q2_q3"].append(best_iou)
    return {
        "duration_strata_seconds": {
            "short_max_q25": short_boundary,
            "long_min_above_q75": long_boundary,
        },
        "strata": {
            name: {
                "ground_truth_count": len(values),
                "mean_best_iou": (
                    sum(values) / len(values) if values else None
                ),
                "proposal_recall": {
                    str(threshold): (
                        sum(value >= threshold for value in values)
                        / len(values)
                        if values
                        else None
                    )
                    for threshold in (0.5, 0.7, 0.9)
                },
            }
            for name, values in strata.items()
        },
    }


def validate_suite(run_root):
    run_root = Path(run_root).resolve()
    deployment_path = run_root / "deployment_summary.json"
    deployment = read_json(deployment_path, "P0 deployment summary")
    require(
        deployment.get("schema_version")
        == "phystime_p0_fullprecision_deployment_v1",
        "P0 deployment summary schema mismatch",
    )
    require(
        deployment.get("new_training") is False
        and deployment.get("frozen_epoch") == 59,
        "P0 deployment is not a frozen epoch-59 replay",
    )

    completions = {}
    manifests = {}
    predictions = {}
    verified_mode_metrics = {}
    prediction_counts = {}
    completion_artifacts = {}
    shared_identity = None
    shared_gate_path = None
    for variant, (expected_arm, expected_weights) in EXPECTED_RUNS.items():
        variant_dir = run_root / variant
        completion_path = variant_dir / "P0_COMPLETE.json"
        completion = read_json(
            completion_path,
            f"{variant} completion",
        )
        require(
            completion.get("schema_version")
            == "phystime_p0_fullprecision_completion_v2"
            and completion.get("validation_pass") is True,
            f"{variant} independent completion did not pass",
        )
        require(
            completion.get("arm") == expected_arm
            and completion.get("weights_source") == expected_weights,
            f"{variant} arm/weight identity mismatch",
        )
        require(
            completion.get("new_training") is False
            and completion.get("evaluation_epoch") == 59,
            f"{variant} is not a frozen epoch-59 replay",
        )
        validate_completion_run_dir(completion, variant_dir, variant)
        identity = (
            completion["runtime_commit"],
            completion["runtime_tree"],
            completion["source_commit"],
            completion["source_tree"],
        )
        if shared_identity is None:
            shared_identity = identity
        require(identity == shared_identity, "P0 suite snapshot identity mismatch")
        require(
            completion["runtime_commit"] == deployment["runtime_commit"]
            and completion["runtime_tree"] == deployment["runtime_tree"]
            and completion["source_commit"] == deployment["source_commit"]
            and completion["source_tree"] == deployment["source_tree"],
            f"{variant} identity differs from deployment summary",
        )
        for artifact_name, artifact_record in completion["artifacts"].items():
            require(
                sha256_file(artifact_record["path"])
                == artifact_record["sha256"],
                f"{variant} {artifact_name} hash mismatch",
            )
        manifest = read_json(
            completion["artifacts"]["manifest"]["path"],
            f"{variant} run manifest",
        )
        require(
            manifest.get("arm") == expected_arm
            and manifest.get("weights_source") == expected_weights
            and manifest.get("evaluation_epoch") == 59
            and manifest.get("new_training") is False,
            f"{variant} manifest identity/frozen contract mismatch",
        )
        gate_path = Path(manifest["gate"]).resolve()
        if shared_gate_path is None:
            shared_gate_path = gate_path
        require(gate_path == shared_gate_path, "P0 suite gate path mismatch")
        mode_predictions = {}
        variant_metrics = {}
        variant_counts = {}
        for mode in EXPECTED_MODES:
            records = completion["mode_artifacts"][mode]
            for record_name in ("report", "predictions", "metrics", "audit"):
                record = records[record_name]
                require(
                    sha256_file(record["path"]) == record["sha256"],
                    f"{variant}/{mode}/{record_name} hash mismatch",
                )
            report_payload = read_json(
                records["report"]["path"],
                f"{variant}/{mode} report",
            )
            mode_predictions[mode] = read_json(
                records["predictions"]["path"],
                f"{variant}/{mode} predictions",
            )
            metric_payload = read_json(
                records["metrics"]["path"],
                f"{variant}/{mode} metrics",
            )
            audit_payload = read_json(
                records["audit"]["path"],
                f"{variant}/{mode} audit",
            )
            require(
                int(metric_payload.get("evaluation_epoch", -1)) == 59,
                f"{variant}/{mode} metric epoch mismatch",
            )
            require(
                int(
                    mode_predictions[mode].get("evaluation_epoch", -1)
                )
                == 59,
                f"{variant}/{mode} prediction epoch mismatch",
            )
            require(
                report_payload.get("schema_version")
                == "phystime_p0_replay_mode_v1"
                and report_payload.get("mode") == mode
                and report_payload.get("status") == "completed"
                and report_payload.get("metrics") == metric_payload,
                f"{variant}/{mode} report/metrics contract mismatch",
            )
            require(
                audit_payload.get("schema_version")
                == "opentad_cross_window_nms_audit_v1"
                and audit_payload.get("aggregate")
                == records["audit_aggregate"],
                f"{variant}/{mode} audit/completion contract mismatch",
            )
            variant_metrics[mode] = {
                key: float(value)
                for key, value in metric_payload.items()
                if key != "evaluation_epoch"
            }
            require_metric_match(
                completion["mode_metrics"][mode],
                variant_metrics[mode],
                f"{variant}/{mode} completion/artifact",
            )
            variant_counts[mode] = sum(
                len(items)
                for items in mode_predictions[mode]["results"].values()
            )
            require(
                variant_counts[mode] == records["prediction_count"],
                f"{variant}/{mode} prediction count mismatch",
            )
            require(
                variant_counts[mode]
                == int(report_payload["prediction_count"]),
                f"{variant}/{mode} report prediction count mismatch",
            )
        independent_within_delta = within_run_delta_report(
            variant_metrics,
            variant_counts,
        )
        require(
            json.dumps(
                independent_within_delta,
                sort_keys=True,
                separators=(",", ":"),
            )
            == json.dumps(
                completion["delta_report"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            f"{variant} within-run delta report mismatch",
        )
        completions[variant] = completion
        manifests[variant] = manifest
        predictions[variant] = mode_predictions
        verified_mode_metrics[variant] = variant_metrics
        prediction_counts[variant] = variant_counts
        completion_artifacts[variant] = {
            "path": str(completion_path.resolve()),
            "sha256": sha256_file(completion_path),
            "size_bytes": completion_path.stat().st_size,
        }

    gate = read_json(shared_gate_path, "shared P0 gate")
    require(gate.get("gate_pass") is True, "shared P0 gate did not pass")
    ground_truth, evaluation_contract = load_ground_truth(gate)

    cross_arm_metric_deltas = {}
    cross_arm_decisions = {}
    for weights in ("online", "ema"):
        selected_variant = f"selected_{weights}"
        physical_variant = f"physical_{weights}"
        cross_arm_metric_deltas[weights] = {}
        cross_arm_decisions[weights] = {}
        for mode in EXPECTED_MODES:
            cross_arm_metric_deltas[weights][mode] = metric_delta(
                verified_mode_metrics[physical_variant][mode],
                verified_mode_metrics[selected_variant][mode],
            )
            cross_arm_decisions[weights][mode] = compare_prediction_decisions(
                predictions[physical_variant][mode],
                predictions[selected_variant][mode],
            )

    weight_source_metric_deltas = {}
    weight_source_decisions = {}
    for arm in ("selected", "physical"):
        online_variant = f"{arm}_online"
        ema_variant = f"{arm}_ema"
        weight_source_metric_deltas[arm] = {}
        weight_source_decisions[arm] = {}
        for mode in EXPECTED_MODES:
            weight_source_metric_deltas[arm][mode] = metric_delta(
                verified_mode_metrics[ema_variant][mode],
                verified_mode_metrics[online_variant][mode],
            )
            weight_source_decisions[arm][mode] = compare_prediction_decisions(
                predictions[ema_variant][mode],
                predictions[online_variant][mode],
            )

    within_run_decisions = {}
    for variant in EXPECTED_RUNS:
        within_run_decisions[variant] = {
            "rounding_effect_unfiltered": compare_prediction_decisions(
                predictions[variant]["fullprecision_unfiltered"],
                predictions[variant]["legacy_unfiltered"],
            ),
            "rounding_effect_filtered": compare_prediction_decisions(
                predictions[variant]["fullprecision_filtered"],
                predictions[variant]["legacy_filtered"],
            ),
            "validity_filter_effect_legacy": compare_prediction_decisions(
                predictions[variant]["legacy_filtered"],
                predictions[variant]["legacy_unfiltered"],
            ),
            "validity_filter_effect_fullprecision": (
                compare_prediction_decisions(
                    predictions[variant]["fullprecision_filtered"],
                    predictions[variant]["fullprecision_unfiltered"],
                )
            ),
        }

    proposal_recall = {
        variant: {
            mode: proposal_recall_diagnostics(
                predictions[variant][mode],
                ground_truth,
            )
            for mode in EXPECTED_MODES
        }
        for variant in EXPECTED_RUNS
    }
    return {
        "schema_version": "phystime_p0_fullprecision_suite_completion_v1",
        "validation_pass": True,
        "status": "diagnostic_completed",
        "completed_at_unix": time.time(),
        "run_root": str(run_root),
        "new_training": False,
        "frozen_epoch": 59,
        "runtime_commit": shared_identity[0],
        "runtime_tree": shared_identity[1],
        "source_commit": shared_identity[2],
        "source_tree": shared_identity[3],
        "deployment_summary": {
            "path": str(deployment_path.resolve()),
            "sha256": sha256_file(deployment_path),
        },
        "gate": {
            "path": str(shared_gate_path),
            "sha256": sha256_file(shared_gate_path),
        },
        "evaluation_contract": evaluation_contract,
        "ground_truth_count": len(ground_truth),
        "completion_artifacts": completion_artifacts,
        "mode_metrics": {
            variant: verified_mode_metrics[variant]
            for variant in EXPECTED_RUNS
        },
        "within_run_metric_deltas": {
            variant: within_run_delta_report(
                verified_mode_metrics[variant],
                prediction_counts[variant],
            )
            for variant in EXPECTED_RUNS
        },
        "cross_arm_physical_minus_selected": cross_arm_metric_deltas,
        "weight_source_ema_minus_online": weight_source_metric_deltas,
        "within_run_decision_diagnostics": within_run_decisions,
        "cross_arm_decision_diagnostics": cross_arm_decisions,
        "weight_source_decision_diagnostics": weight_source_decisions,
        "proposal_recall_by_duration_and_iou": proposal_recall,
        "claim_boundary": (
            "This artifact closes the frozen post-processing diagnostic only; "
            "it does not establish multi-seed, cross-dataset, cost, or "
            "paper-ready evidence."
        ),
    }


def main():
    args = parse_args()
    completion = validate_suite(args.run_root)
    atomic_write_json(args.output, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
