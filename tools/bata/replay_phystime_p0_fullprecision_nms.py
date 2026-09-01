#!/usr/bin/env python3
"""Replay cross-window NMS policies from one frozen full-precision prediction artifact."""

import argparse
import copy
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

from mmengine.config import Config

from opentad.cores.test_engine import InvalidProposalError, apply_sliding_window_nms
from opentad.evaluations import build_evaluator


MODE_SPECS = {
    "legacy_unfiltered": {
        "round_before_cross_window_nms": True,
        "round_after_cross_window_nms": True,
        "filter_invalid_proposals": False,
    },
    "legacy_filtered": {
        "round_before_cross_window_nms": True,
        "round_after_cross_window_nms": True,
        "filter_invalid_proposals": True,
    },
    "fullprecision_unfiltered": {
        "round_before_cross_window_nms": False,
        "round_after_cross_window_nms": False,
        "filter_invalid_proposals": False,
    },
    "fullprecision_filtered": {
        "round_before_cross_window_nms": False,
        "round_after_cross_window_nms": False,
        "filter_invalid_proposals": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the PhysTime P0 2x2 precision/validity replay."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--pre-cross-window", required=True)
    parser.add_argument("--direct-result", required=True)
    parser.add_argument("--direct-metrics", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-completion", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--arm",
        required=True,
        choices=("selected_axis", "physical_metric"),
    )
    parser.add_argument(
        "--weights-source",
        required=True,
        choices=("online", "ema"),
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--expected-runtime-commit")
    parser.add_argument("--expected-runtime-tree")
    parser.add_argument("--evaluation-epoch", type=int, default=59)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def load_pre_cross_window_artifact(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != "opentad_pre_cross_window_detections_v1":
        raise ValueError("unsupported pre-cross-window artifact schema")
    if payload.get("artifact_kind") != "pre_cross_window_nms_full_precision_detections":
        raise ValueError("artifact is not a full-precision pre-cross-window prediction set")
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError("pre-cross-window artifact results must be a dictionary")
    return payload


def run_replay(
    *,
    cfg,
    pre_cross_window_payload,
    output_dir,
    evaluation_epoch,
):
    output_dir = Path(output_dir)
    results = pre_cross_window_payload["results"]
    mode_reports = {}
    for mode_name, mode_spec in MODE_SPECS.items():
        mode_dir = output_dir / "modes" / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        mode_cfg = copy.deepcopy(cfg.post_processing)
        mode_cfg.sliding_window = True
        for key, value in mode_spec.items():
            mode_cfg[key] = value

        try:
            merged, audit = apply_sliding_window_nms(
                results,
                mode_cfg,
                return_audit=True,
            )
        except InvalidProposalError as error:
            blocked_audit_path = mode_dir / "post_processing_audit.json"
            atomic_write_json(blocked_audit_path, error.audit)
            blocked_report = {
                "schema_version": "phystime_p0_replay_mode_v1",
                "mode": mode_name,
                "status": "blocked_invalid_unfiltered_input",
                "policy": mode_spec,
                "error": str(error),
                "audit_sha256": sha256_file(blocked_audit_path),
                "audit_summary": {
                    "policy": error.audit["policy"],
                    "aggregate": error.audit["aggregate"],
                },
            }
            atomic_write_json(mode_dir / "mode_report.json", blocked_report)
            mode_reports[mode_name] = blocked_report
            continue

        prediction_payload = {
            "results": merged,
            "evaluation_epoch": evaluation_epoch,
        }
        metrics = evaluate_predictions(cfg, prediction_payload)
        metrics_payload = dict(metrics)
        metrics_payload["evaluation_epoch"] = evaluation_epoch
        prediction_path = mode_dir / "result_detection.json"
        metrics_path = mode_dir / "evaluation_metrics.json"
        audit_path = mode_dir / "post_processing_audit.json"
        atomic_write_json(prediction_path, prediction_payload)
        atomic_write_json(metrics_path, metrics_payload)
        atomic_write_json(audit_path, audit)
        report = {
            "schema_version": "phystime_p0_replay_mode_v1",
            "mode": mode_name,
            "status": "completed",
            "policy": mode_spec,
            "metrics": metrics_payload,
            "prediction_count": sum(len(items) for items in merged.values()),
            "prediction_sha256": sha256_file(prediction_path),
            "canonical_results_sha256": canonical_sha256(merged),
            "metrics_sha256": sha256_file(metrics_path),
            "audit_sha256": sha256_file(audit_path),
            "audit_summary": {
                "policy": audit["policy"],
                "aggregate": audit["aggregate"],
            },
        }
        atomic_write_json(mode_dir / "mode_report.json", report)
        mode_reports[mode_name] = report
    return mode_reports


def evaluate_predictions(cfg, prediction_payload):
    evaluation_cfg = copy.deepcopy(cfg.evaluation)
    evaluator = build_evaluator(
        dict(prediction_filename=prediction_payload, **evaluation_cfg)
    )
    return evaluator.evaluate()


def compare_direct_fullprecision_filtered(
    direct_result,
    direct_metrics,
    mode_report,
):
    if mode_report.get("status") != "completed":
        return {
            "match": False,
            "reason": "fullprecision_filtered replay did not complete",
        }
    direct_results = direct_result.get("results")
    replay_results_sha = mode_report.get("canonical_results_sha256")
    direct_results_sha = canonical_sha256(direct_results)
    direct_metric_values = {
        key: float(value)
        for key, value in direct_metrics.items()
        if key != "evaluation_epoch"
    }
    replay_metric_values = {
        key: float(value)
        for key, value in mode_report["metrics"].items()
        if key != "evaluation_epoch"
    }
    metric_deltas = {
        key: direct_metric_values.get(key, float("nan"))
        - replay_metric_values.get(key, float("nan"))
        for key in sorted(set(direct_metric_values) | set(replay_metric_values))
    }
    metrics_match = (
        direct_metric_values.keys() == replay_metric_values.keys()
        and all(abs(delta) <= 1.0e-12 for delta in metric_deltas.values())
    )
    predictions_match = direct_results_sha == replay_results_sha
    return {
        "match": predictions_match and metrics_match,
        "predictions_match": predictions_match,
        "metrics_match": metrics_match,
        "direct_canonical_results_sha256": direct_results_sha,
        "replay_canonical_results_sha256": replay_results_sha,
        "metric_deltas": metric_deltas,
    }


def compare_source_legacy_ema(
    *,
    source_completion,
    mode_report,
    replay_output_dir,
    weights_source,
):
    if weights_source != "ema":
        return {
            "status": "not_applicable_online_weights",
            "match": True,
        }
    if mode_report.get("status") != "completed":
        return {
            "status": "legacy_replay_incomplete",
            "match": False,
        }
    source_prediction_path = Path(
        source_completion["artifacts"]["predictions"]["path"]
    )
    source_prediction = json.loads(
        source_prediction_path.read_text(encoding="utf-8")
    )
    replay_prediction_path = (
        Path(replay_output_dir)
        / "modes"
        / "legacy_unfiltered"
        / "result_detection.json"
    )
    replay_prediction = json.loads(
        replay_prediction_path.read_text(encoding="utf-8")
    )
    source_results_sha = canonical_sha256(source_prediction["results"])
    replay_results_sha = canonical_sha256(replay_prediction["results"])
    source_metrics = {
        key: float(value)
        for key, value in source_completion["metrics"].items()
    }
    replay_metrics = {
        key: float(value)
        for key, value in mode_report["metrics"].items()
        if key != "evaluation_epoch"
    }
    metric_deltas = {
        key: replay_metrics.get(key, float("nan"))
        - source_metrics.get(key, float("nan"))
        for key in sorted(set(source_metrics) | set(replay_metrics))
    }
    metrics_match = (
        source_metrics.keys() == replay_metrics.keys()
        and all(abs(delta) <= 1.0e-12 for delta in metric_deltas.values())
    )
    predictions_match = source_results_sha == replay_results_sha
    return {
        "status": "compared",
        "match": predictions_match and metrics_match,
        "predictions_match": predictions_match,
        "metrics_match": metrics_match,
        "source_prediction_path": str(source_prediction_path.resolve()),
        "source_prediction_sha256": sha256_file(source_prediction_path),
        "source_canonical_results_sha256": source_results_sha,
        "replay_canonical_results_sha256": replay_results_sha,
        "metric_deltas": metric_deltas,
    }


def build_delta_report(mode_reports):
    comparisons = {
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
    report = {}
    for name, (lhs_name, rhs_name) in comparisons.items():
        lhs = mode_reports[lhs_name]
        rhs = mode_reports[rhs_name]
        if lhs.get("status") != "completed" or rhs.get("status") != "completed":
            report[name] = {
                "status": "not_comparable",
                "lhs": lhs_name,
                "rhs": rhs_name,
                "lhs_status": lhs.get("status"),
                "rhs_status": rhs.get("status"),
            }
            continue
        metric_keys = sorted(
            set(lhs["metrics"]).intersection(rhs["metrics"])
            - {"evaluation_epoch"}
        )
        report[name] = {
            "status": "comparable",
            "lhs": lhs_name,
            "rhs": rhs_name,
            "metric_delta_fraction": {
                key: float(lhs["metrics"][key]) - float(rhs["metrics"][key])
                for key in metric_keys
            },
            "metric_delta_percentage_points": {
                key: 100.0
                * (float(lhs["metrics"][key]) - float(rhs["metrics"][key]))
                for key in metric_keys
            },
            "prediction_count_delta": lhs["prediction_count"]
            - rhs["prediction_count"],
        }
    return report


def validate_source_artifacts(args):
    source_completion = json.loads(
        Path(args.source_completion).read_text(encoding="utf-8")
    )
    if source_completion.get("validation_pass") is not True:
        raise ValueError("source full60 completion did not pass validation")
    source_manifest = json.loads(
        Path(args.source_manifest).read_text(encoding="utf-8")
    )
    if source_manifest.get("commit") != args.source_commit:
        raise ValueError("source manifest commit mismatch")
    if source_manifest.get("git_tree") != args.source_tree:
        raise ValueError("source manifest tree mismatch")
    if source_manifest.get("variant") != args.arm:
        raise ValueError("source manifest arm mismatch")
    if int(source_manifest.get("final_epoch", -1)) != args.evaluation_epoch:
        raise ValueError("source manifest final epoch mismatch")
    return source_completion, source_manifest


def current_git_identity():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()
    return commit, tree


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    runtime_commit, runtime_tree = current_git_identity()
    if (
        args.expected_runtime_commit is not None
        and runtime_commit != args.expected_runtime_commit
    ):
        raise SystemExit("runtime commit differs from the submitted P0 snapshot")
    if (
        args.expected_runtime_tree is not None
        and runtime_tree != args.expected_runtime_tree
    ):
        raise SystemExit("runtime tree differs from the submitted P0 snapshot")

    source_completion, source_manifest = validate_source_artifacts(args)
    cfg = Config.fromfile(args.config, lazy_import=False)
    pre_cross_window_payload = load_pre_cross_window_artifact(
        args.pre_cross_window
    )
    if pre_cross_window_payload.get("git_commit") not in (None, runtime_commit):
        raise SystemExit("pre-cross-window artifact runtime commit mismatch")
    if pre_cross_window_payload.get("git_tree") not in (None, runtime_tree):
        raise SystemExit("pre-cross-window artifact runtime tree mismatch")

    direct_result = json.loads(Path(args.direct_result).read_text(encoding="utf-8"))
    direct_metrics = json.loads(
        Path(args.direct_metrics).read_text(encoding="utf-8")
    )
    mode_reports = run_replay(
        cfg=cfg,
        pre_cross_window_payload=pre_cross_window_payload,
        output_dir=output_dir,
        evaluation_epoch=args.evaluation_epoch,
    )
    direct_equivalence = compare_direct_fullprecision_filtered(
        direct_result,
        direct_metrics,
        mode_reports["fullprecision_filtered"],
    )
    source_legacy_equivalence = compare_source_legacy_ema(
        source_completion=source_completion,
        mode_report=mode_reports["legacy_unfiltered"],
        replay_output_dir=output_dir,
        weights_source=args.weights_source,
    )
    delta_report = build_delta_report(mode_reports)
    all_modes_completed = all(
        report.get("status") == "completed"
        for report in mode_reports.values()
    )
    validation_pass = (
        all_modes_completed
        and direct_equivalence["match"]
        and source_legacy_equivalence["match"]
    )

    completion = {
        "schema_version": "phystime_p0_fullprecision_nms_replay_v1",
        "validation_pass": validation_pass,
        "status": "completed" if validation_pass else "incomplete_or_failed",
        "arm": args.arm,
        "weights_source": args.weights_source,
        "evaluation_epoch": args.evaluation_epoch,
        "runtime": {
            "commit": runtime_commit,
            "git_tree": runtime_tree,
            "config": str(Path(args.config).resolve()),
            "config_sha256": canonical_sha256(cfg.to_dict()),
        },
        "source_full60": {
            "commit": args.source_commit,
            "git_tree": args.source_tree,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "completion": str(Path(args.source_completion).resolve()),
            "completion_sha256": sha256_file(args.source_completion),
            "manifest": str(Path(args.source_manifest).resolve()),
            "manifest_sha256": sha256_file(args.source_manifest),
            "source_metrics": source_completion.get("metrics"),
            "source_effective_config_sha256": source_manifest.get(
                "effective_config_sha256"
            ),
        },
        "inference_artifacts": {
            "pre_cross_window": str(Path(args.pre_cross_window).resolve()),
            "pre_cross_window_sha256": sha256_file(args.pre_cross_window),
            "direct_result": str(Path(args.direct_result).resolve()),
            "direct_result_sha256": sha256_file(args.direct_result),
            "direct_metrics": str(Path(args.direct_metrics).resolve()),
            "direct_metrics_sha256": sha256_file(args.direct_metrics),
        },
        "mode_reports": mode_reports,
        "direct_fullprecision_filtered_equivalence": direct_equivalence,
        "source_legacy_ema_equivalence": source_legacy_equivalence,
        "delta_report": delta_report,
        "all_modes_completed": all_modes_completed,
    }
    atomic_write_json(output_dir / "P0_REPLAY_COMPLETE.json", completion)
    if not validation_pass:
        raise SystemExit(
            "P0 replay completion contract failed; inspect P0_REPLAY_COMPLETE.json"
        )
    print(json.dumps(completion, indent=2, default=_json_default))


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


if __name__ == "__main__":
    main()
