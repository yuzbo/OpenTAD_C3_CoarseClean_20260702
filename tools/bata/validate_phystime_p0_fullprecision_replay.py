#!/usr/bin/env python3
"""Independent artifact validator for one PhysTime P0 arm/weight replay."""

import argparse
import gzip
import hashlib
import json
import math
import os
import time
from pathlib import Path

from mmengine.config import Config

EXPECTED_EPOCH = 59
EXPECTED_MODE_SPECS = {
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
EXPECTED_COMPARISONS = {
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate one completed PhysTime P0 replay."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path, description):
    path = Path(path)
    require(path.is_file(), f"{description} is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_gzip_json(path, description):
    path = Path(path)
    require(path.is_file(), f"{description} is missing: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


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
        default=lambda value: value.item(),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact(path, started_at):
    path = Path(path).resolve()
    require(path.is_file(), f"artifact is missing: {path}")
    stat = path.stat()
    require(
        stat.st_mtime + 1.0 >= started_at,
        f"artifact predates this replay: {path}",
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": stat.st_size,
        "mtime_unix": stat.st_mtime,
    }


def metric_values(payload):
    metrics = {
        key: float(value)
        for key, value in payload.items()
        if key != "evaluation_epoch"
    }
    require(metrics, "metric payload is empty")
    require(
        all(math.isfinite(value) for value in metrics.values()),
        "metric payload contains non-finite values",
    )
    return metrics


def validate_evaluation_epoch(payload, description):
    require(
        int(payload.get("evaluation_epoch", -1)) == EXPECTED_EPOCH,
        f"{description} epoch mismatch",
    )


def prediction_count(payload, description):
    results = payload.get("results")
    require(isinstance(results, dict), f"{description} results are not a dictionary")
    require(
        all(isinstance(items, list) for items in results.values()),
        f"{description} contains a non-list video prediction set",
    )
    return sum(len(items) for items in results.values())


def validate_audit_count_conservation(
    audit,
    input_payload,
    output_payload,
    description,
):
    require(
        audit.get("schema_version") == "opentad_cross_window_nms_audit_v1",
        f"{description} audit schema mismatch",
    )
    require(audit.get("nms_applied") is True, f"{description} did not apply NMS")
    input_results = input_payload.get("results")
    output_results = output_payload.get("results")
    require(
        isinstance(input_results, dict) and isinstance(output_results, dict),
        f"{description} prediction payload is malformed",
    )
    videos = audit.get("videos")
    aggregate = audit.get("aggregate")
    require(
        isinstance(videos, dict) and isinstance(aggregate, dict),
        f"{description} audit does not contain video/aggregate dictionaries",
    )
    require(
        set(videos) == set(input_results),
        f"{description} audit/input video sets differ",
    )
    require(
        set(output_results).issubset(input_results),
        f"{description} output contains a video absent from the input",
    )
    fields_to_sum = (
        "input_detections",
        "valid_detections",
        "invalid_detections",
        "filtered_detections",
        "raw_invalid_detections",
        "raw_filtered_detections",
        "effective_input_detections",
        "effective_invalid_detections",
        "effective_filtered_detections",
        "rounding_induced_invalid_detections",
        "kept_for_nms",
        "post_nms_detections",
        "post_nms_invalid_detections",
        "post_nms_filtered_detections",
        "pre_nms_rounding_changed_segment_values",
        "pre_nms_rounding_changed_scores",
    )
    summed = {key: 0 for key in fields_to_sum}
    videos_with_invalid = 0
    filter_invalid = bool(audit["policy"]["filter_invalid_proposals"])
    for video_name, input_items in input_results.items():
        video_audit = videos[video_name]
        raw = video_audit["raw_validation"]
        effective = video_audit["effective_validation"]
        output = video_audit["output_validation"]
        output_items = output_results.get(video_name, [])
        require(
            raw["input_detections"] == len(input_items)
            and raw["valid_detections"] + raw["invalid_detections"]
            == raw["input_detections"],
            f"{description}/{video_name} raw count conservation failed",
        )
        require(
            effective["input_detections"] == raw["valid_detections"]
            and effective["valid_detections"]
            + effective["invalid_detections"]
            == effective["input_detections"],
            f"{description}/{video_name} effective count conservation failed",
        )
        require(
            video_audit["input_detections"] == len(input_items)
            and video_audit["valid_detections"]
            == effective["valid_detections"]
            and video_audit["invalid_detections"]
            == raw["invalid_detections"] + effective["invalid_detections"]
            and video_audit["rounding_induced_invalid_detections"]
            == effective["invalid_detections"]
            and video_audit["kept_for_nms"] == effective["valid_detections"],
            f"{description}/{video_name} pre-NMS audit counts differ",
        )
        require(
            video_audit["filtered_detections"]
            == (video_audit["invalid_detections"] if filter_invalid else 0)
            and video_audit["raw_filtered_detections"]
            == (raw["invalid_detections"] if filter_invalid else 0)
            and video_audit["effective_filtered_detections"]
            == (effective["invalid_detections"] if filter_invalid else 0),
            f"{description}/{video_name} pre-NMS filter counts differ",
        )
        if output is None:
            require(
                not output_items
                and video_audit["kept_for_nms"] == 0
                and video_audit["post_nms_detections"] == 0
                and video_audit["post_nms_invalid_detections"] == 0
                and video_audit["post_nms_filtered_detections"] == 0,
                f"{description}/{video_name} empty-output audit differs",
            )
        else:
            require(
                output["valid_detections"] == len(output_items)
                and output["valid_detections"] + output["invalid_detections"]
                == output["input_detections"]
                and video_audit["post_nms_detections"] == len(output_items)
                and video_audit["post_nms_invalid_detections"]
                == output["invalid_detections"]
                and video_audit["post_nms_filtered_detections"]
                == (output["invalid_detections"] if filter_invalid else 0),
                f"{description}/{video_name} post-NMS count conservation failed",
            )
        if video_audit["invalid_detections"] > 0:
            videos_with_invalid += 1
        for key in fields_to_sum:
            summed[key] += int(video_audit[key])
    require(
        int(aggregate["videos"]) == len(input_results),
        f"{description} aggregate video count differs",
    )
    require(
        int(aggregate["videos_with_invalid_detections"])
        == videos_with_invalid,
        f"{description} aggregate invalid-video count differs",
    )
    for key, expected in summed.items():
        require(
            int(aggregate[key]) == expected,
            f"{description} aggregate {key} differs: "
            f"observed={aggregate[key]}, expected={expected}",
        )
    reason_sources = {
        "invalid_reason_counts": "invalid_reason_counts",
        "raw_invalid_reason_counts": "raw_validation",
        "effective_invalid_reason_counts": "effective_validation",
    }
    for aggregate_key, video_key in reason_sources.items():
        expected_reasons = {
            reason: 0 for reason in aggregate[aggregate_key]
        }
        for video_audit in videos.values():
            source = video_audit[video_key]
            if video_key in {"raw_validation", "effective_validation"}:
                source = source["invalid_reason_counts"]
            for reason, count in source.items():
                expected_reasons[reason] += int(count)
        require(
            aggregate[aggregate_key] == expected_reasons,
            f"{description} aggregate {aggregate_key} differs",
        )
    require(
        int(aggregate["input_detections"])
        == prediction_count(input_payload, f"{description} input"),
        f"{description} aggregate input count differs from pre-cross artifact",
    )
    require(
        int(aggregate["post_nms_detections"])
        == prediction_count(output_payload, f"{description} output"),
        f"{description} aggregate output count differs from result artifact",
    )


def recompute_metrics(cfg, prediction_payload):
    from opentad.evaluations import build_evaluator

    evaluator = build_evaluator(
        dict(prediction_filename=prediction_payload, **cfg.evaluation)
    )
    return {key: float(value) for key, value in evaluator.evaluate().items()}


def compare_metrics(reported, recomputed, description):
    reported = metric_values(reported)
    require(
        reported.keys() == recomputed.keys(),
        f"{description} metric keys differ from independent evaluator",
    )
    deltas = {
        key: reported[key] - recomputed[key]
        for key in sorted(reported)
    }
    require(
        all(abs(delta) <= 1.0e-12 for delta in deltas.values()),
        f"{description} metrics differ from independent evaluator: {deltas}",
    )
    return recomputed


def expected_policy(cfg, mode_spec):
    policy = {
        "filter_invalid_proposals": bool(
            cfg.post_processing.filter_invalid_proposals
        ),
        "proposal_min_duration": float(
            cfg.post_processing.proposal_min_duration
        ),
        "round_before_cross_window_nms": bool(
            cfg.post_processing.round_before_cross_window_nms
        ),
        "round_after_cross_window_nms": bool(
            cfg.post_processing.round_after_cross_window_nms
        ),
        "segment_round_digits": int(
            cfg.post_processing.segment_round_digits
        ),
        "score_round_digits": int(cfg.post_processing.score_round_digits),
    }
    policy.update(mode_spec)
    return policy


def build_delta_report(mode_metrics, prediction_counts):
    report = {}
    for name, (lhs_name, rhs_name) in EXPECTED_COMPARISONS.items():
        lhs = mode_metrics[lhs_name]
        rhs = mode_metrics[rhs_name]
        metric_keys = sorted(set(lhs).intersection(rhs))
        report[name] = {
            "status": "comparable",
            "lhs": lhs_name,
            "rhs": rhs_name,
            "metric_delta_fraction": {
                key: lhs[key] - rhs[key] for key in metric_keys
            },
            "metric_delta_percentage_points": {
                key: 100.0 * (lhs[key] - rhs[key]) for key in metric_keys
            },
            "prediction_count_delta": (
                prediction_counts[lhs_name] - prediction_counts[rhs_name]
            ),
        }
    return report


def require_nested_close(observed, expected, description):
    if isinstance(expected, dict):
        require(isinstance(observed, dict), f"{description} is not a dictionary")
        require(
            observed.keys() == expected.keys(),
            f"{description} keys differ: "
            f"observed={sorted(observed)}, expected={sorted(expected)}",
        )
        for key in expected:
            require_nested_close(
                observed[key],
                expected[key],
                f"{description}.{key}",
            )
        return
    if isinstance(expected, float):
        require(
            math.isfinite(float(observed))
            and abs(float(observed) - expected) <= 1.0e-12,
            f"{description} differs: observed={observed}, expected={expected}",
        )
        return
    require(observed == expected, f"{description} differs")


def validate_replay(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path, "P0 run manifest")
    require(
        manifest.get("schema_version") == "phystime_p0_inference_manifest_v1",
        "P0 run manifest schema mismatch",
    )
    started_at = float(manifest.get("started_at_unix", float("nan")))
    require(math.isfinite(started_at), "P0 run start time is invalid")
    arm = manifest.get("arm")
    weights_source = manifest.get("weights_source")
    require(
        arm in {"selected_axis", "physical_metric"},
        "unsupported P0 arm",
    )
    require(weights_source in {"online", "ema"}, "unsupported P0 weight source")
    require(manifest.get("new_training") is False, "P0 replay must not train")
    require(
        int(manifest.get("evaluation_epoch", -1)) == EXPECTED_EPOCH,
        "P0 run manifest epoch mismatch",
    )

    config_path = Path(manifest["config"]).resolve()
    cfg = Config.fromfile(config_path, lazy_import=False)
    effective_cfg = Config.fromfile(config_path, lazy_import=False)
    effective_cfg.merge_from_dict(
        {
            "work_dir": str(Path(run_dir, "direct_work").resolve()),
            "solver.ema": weights_source == "ema",
            "model.backbone.custom.pretrain": str(
                Path(manifest["videomae_checkpoint"]).resolve()
            ),
        }
    )
    require(
        canonical_sha256(effective_cfg.to_dict())
        == manifest["effective_config_sha256"],
        "P0 effective config hash mismatch",
    )
    gate_path = Path(manifest["gate"]).resolve()
    gate = read_json(gate_path, "P0 gate")
    require(gate.get("gate_pass") is True, "P0 gate did not pass")
    require(
        sha256_file(gate_path) == manifest["gate_sha256"],
        "P0 gate hash differs from run manifest",
    )
    require(
        gate["runtime"]["commit"] == manifest["runtime_commit"]
        and gate["runtime"]["git_tree"] == manifest["runtime_tree"],
        "P0 gate and run manifest runtime identities differ",
    )
    environment = manifest.get("environment", {})
    gate_environment = gate["runtime"]["environment"]
    require(
        environment.get("init_mode")
        in {
            "module_cuda11.8_miniforge3_24.11",
            "fixed_conda_path_no_module_command",
        }
        and environment.get("cuda_available") is True
        and int(environment.get("cuda_device_count", 0)) >= 1
        and environment.get("torch") == gate_environment.get("torch")
        and environment.get("torch_cuda") == gate_environment.get("torch_cuda")
        and environment.get("cudnn") == gate_environment.get("cudnn")
        and environment.get("python_executable")
        == gate_environment.get("python_executable"),
        "P0 replay runtime environment differs from the gate",
    )
    source_gate = gate["source_full60"]["arms"][arm]
    require(
        sha256_file(manifest["checkpoint"])
        == manifest["checkpoint_sha256"]
        == source_gate["checkpoint_sha256"],
        "P0 checkpoint hash differs from manifest/gate",
    )
    require(
        sha256_file(manifest["videomae_checkpoint"])
        == manifest["videomae_checkpoint_sha256"]
        == gate["runtime"]["videomae_checkpoint_sha256"],
        "P0 VideoMAE checkpoint hash differs from manifest/gate",
    )
    require(
        sha256_file(manifest["source_completion"])
        == manifest["source_completion_sha256"]
        == source_gate["completion_sha256"],
        "source completion hash differs from manifest/gate",
    )
    require(
        sha256_file(manifest["source_manifest"])
        == manifest["source_manifest_sha256"]
        == source_gate["manifest_sha256"],
        "source manifest hash differs from manifest/gate",
    )

    inference_log_path = run_dir / "inference.out"
    inference_log = inference_log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    require(
        str(Path(manifest["checkpoint"]).resolve()) in inference_log,
        "inference log does not bind the source checkpoint path",
    )
    ema_log_present = "Using Model EMA..." in inference_log
    require(
        ema_log_present == (weights_source == "ema"),
        "inference log does not match the requested online/EMA source",
    )
    require(
        f"Checkpoint is epoch {EXPECTED_EPOCH}." in inference_log,
        "inference log does not bind epoch 59",
    )

    direct_dir = run_dir / "direct_work" / "gpu1_id0"
    direct_result_path = direct_dir / "result_detection.json"
    direct_metrics_path = direct_dir / "evaluation_metrics.json"
    pre_cross_path = direct_dir / "pre_cross_window_detections.json.gz"
    direct_audit_path = direct_dir / "post_processing_audit.json"
    direct_result = read_json(direct_result_path, "direct full-precision result")
    direct_metrics = read_json(
        direct_metrics_path,
        "direct full-precision metrics",
    )
    pre_cross = read_gzip_json(
        pre_cross_path,
        "full-precision pre-cross-window predictions",
    )
    direct_audit = read_json(
        direct_audit_path,
        "direct post-processing audit",
    )
    for payload, description in (
        (direct_result, "direct result"),
        (direct_metrics, "direct metrics"),
        (pre_cross, "pre-cross-window artifact"),
        (direct_audit, "direct audit"),
    ):
        require(
            int(payload.get("evaluation_epoch", -1)) == EXPECTED_EPOCH,
            f"{description} epoch mismatch",
        )
    require(
        pre_cross.get("git_commit") == manifest["runtime_commit"]
        and pre_cross.get("git_tree") == manifest["runtime_tree"],
        "pre-cross-window artifact runtime identity mismatch",
    )
    require(
        direct_audit.get("schema_version")
        == "opentad_post_processing_audit_v1",
        "direct audit schema mismatch",
    )
    require(
        Path(
            direct_audit["pre_cross_window_artifact"]["path"]
        ).resolve()
        == pre_cross_path.resolve()
        and direct_audit["pre_cross_window_artifact"]["sha256"]
        == sha256_file(pre_cross_path),
        "direct audit pre-cross-window hash mismatch",
    )
    direct_recomputed = compare_metrics(
        direct_metrics,
        recompute_metrics(cfg, direct_result),
        "direct fullprecision-filtered",
    )
    direct_aggregate = direct_audit["post_processing"]["aggregate"]
    validate_audit_count_conservation(
        direct_audit["post_processing"],
        pre_cross,
        direct_result,
        "direct fullprecision-filtered",
    )
    require(
        direct_aggregate["invalid_detections"] == 0
        and direct_aggregate["post_nms_invalid_detections"] == 0,
        "direct fullprecision-filtered path contains invalid proposals",
    )

    replay_dir = run_dir / "replay"
    replay_completion_path = replay_dir / "P0_REPLAY_COMPLETE.json"
    replay_completion = read_json(
        replay_completion_path,
        "P0 replay completion",
    )
    require(
        replay_completion.get("validation_pass") is True,
        "P0 replay self-completion did not pass",
    )
    require(replay_completion.get("arm") == arm, "P0 replay arm mismatch")
    require(
        replay_completion.get("weights_source") == weights_source,
        "P0 replay weight source mismatch",
    )
    require(
        replay_completion.get("evaluation_epoch") == EXPECTED_EPOCH,
        "P0 replay evaluation epoch mismatch",
    )
    require(
        replay_completion["runtime"]["commit"] == manifest["runtime_commit"]
        and replay_completion["runtime"]["git_tree"]
        == manifest["runtime_tree"],
        "P0 replay runtime identity mismatch",
    )
    require(
        Path(replay_completion["runtime"]["config"]).resolve() == config_path
        and replay_completion["runtime"]["config_sha256"]
        == canonical_sha256(cfg.to_dict()),
        "P0 replay runtime config binding mismatch",
    )
    require(
        replay_completion["source_full60"]["checkpoint_sha256"]
        == manifest["checkpoint_sha256"],
        "P0 replay checkpoint hash mismatch",
    )
    require(
        replay_completion["source_full60"]["commit"]
        == manifest["source_commit"]
        and replay_completion["source_full60"]["git_tree"]
        == manifest["source_tree"]
        and replay_completion["source_full60"]["completion_sha256"]
        == manifest["source_completion_sha256"]
        == source_gate["completion_sha256"]
        and replay_completion["source_full60"]["manifest_sha256"]
        == manifest["source_manifest_sha256"]
        == source_gate["manifest_sha256"],
        "P0 replay source completion/manifest binding mismatch",
    )
    for name, path in (
        ("pre_cross_window", pre_cross_path),
        ("direct_result", direct_result_path),
        ("direct_metrics", direct_metrics_path),
    ):
        require(
            replay_completion["inference_artifacts"][f"{name}_sha256"]
            == sha256_file(path),
            f"P0 replay {name} hash mismatch",
        )

    mode_metrics = {}
    mode_artifacts = {}
    prediction_counts = {}
    mode_predictions = {}
    for mode_name, mode_spec in EXPECTED_MODE_SPECS.items():
        mode_dir = replay_dir / "modes" / mode_name
        mode_report_path = mode_dir / "mode_report.json"
        prediction_path = mode_dir / "result_detection.json"
        metrics_path = mode_dir / "evaluation_metrics.json"
        audit_path = mode_dir / "post_processing_audit.json"
        report = read_json(mode_report_path, f"{mode_name} report")
        prediction = read_json(prediction_path, f"{mode_name} predictions")
        metrics = read_json(metrics_path, f"{mode_name} metrics")
        audit = read_json(audit_path, f"{mode_name} audit")
        require(
            report.get("schema_version") == "phystime_p0_replay_mode_v1"
            and report.get("mode") == mode_name
            and report.get("policy") == mode_spec
            and report.get("status") == "completed",
            f"{mode_name} report contract mismatch",
        )
        require(
            replay_completion["mode_reports"][mode_name] == report,
            f"{mode_name} completion/report payloads differ",
        )
        require(
            report.get("metrics") == metrics,
            f"{mode_name} report/metrics payloads differ",
        )
        require(
            int(prediction.get("evaluation_epoch", -1)) == EXPECTED_EPOCH,
            f"{mode_name} prediction epoch mismatch",
        )
        validate_evaluation_epoch(metrics, f"{mode_name} metrics")
        independently_recomputed = compare_metrics(
            metrics,
            recompute_metrics(cfg, prediction),
            mode_name,
        )
        count = sum(len(items) for items in prediction["results"].values())
        require(
            int(report["prediction_count"]) == count,
            f"{mode_name} prediction count mismatch",
        )
        require(
            report["prediction_sha256"] == sha256_file(prediction_path)
            and report["canonical_results_sha256"]
            == canonical_sha256(prediction["results"])
            and report["metrics_sha256"] == sha256_file(metrics_path)
            and report["audit_sha256"] == sha256_file(audit_path),
            f"{mode_name} recorded artifact hash mismatch",
        )
        require(
            audit.get("schema_version")
            == "opentad_cross_window_nms_audit_v1",
            f"{mode_name} audit schema mismatch",
        )
        require(
            audit["policy"] == expected_policy(cfg, mode_spec),
            f"{mode_name} audit policy mismatch",
        )
        require(
            report["audit_summary"]
            == {
                "policy": audit["policy"],
                "aggregate": audit["aggregate"],
            },
            f"{mode_name} report audit summary mismatch",
        )
        require(
            audit["aggregate"]["invalid_detections"] == 0
            and audit["aggregate"]["post_nms_invalid_detections"] == 0,
            f"{mode_name} contains invalid proposals",
        )
        validate_audit_count_conservation(
            audit,
            pre_cross,
            prediction,
            mode_name,
        )
        mode_metrics[mode_name] = independently_recomputed
        prediction_counts[mode_name] = count
        mode_predictions[mode_name] = prediction
        mode_artifacts[mode_name] = {
            "report": artifact(mode_report_path, started_at),
            "predictions": artifact(prediction_path, started_at),
            "metrics": artifact(metrics_path, started_at),
            "audit": artifact(audit_path, started_at),
            "prediction_count": count,
            "audit_aggregate": audit["aggregate"],
        }

    direct_prediction_sha = canonical_sha256(direct_result["results"])
    replay_prediction_sha = canonical_sha256(
        mode_predictions["fullprecision_filtered"]["results"]
    )
    require(
        direct_prediction_sha == replay_prediction_sha,
        "direct and replayed fullprecision-filtered predictions differ",
    )
    require_nested_close(
        direct_recomputed,
        mode_metrics["fullprecision_filtered"],
        "direct/replay fullprecision-filtered metrics",
    )

    source_completion = read_json(
        manifest["source_completion"],
        "source full60 completion",
    )
    if weights_source == "ema":
        source_prediction_path = Path(
            source_gate["evaluation_artifacts"]["predictions"]["path"]
        )
        source_prediction = read_json(
            source_prediction_path,
            "source full60 predictions",
        )
        validate_evaluation_epoch(
            source_prediction,
            "source full60 predictions",
        )
        require(
            sha256_file(source_prediction_path)
            == source_gate["evaluation_artifacts"]["predictions"]["sha256"],
            "source legacy prediction file differs from the P0 gate",
        )
        legacy_prediction = mode_predictions["legacy_unfiltered"]
        require(
            canonical_sha256(source_prediction["results"])
            == canonical_sha256(legacy_prediction["results"]),
            "legacy EMA replay does not reproduce source predictions",
        )
        source_metrics = {
            key: float(value)
            for key, value in source_completion["metrics"].items()
        }
        require_nested_close(
            mode_metrics["legacy_unfiltered"],
            source_metrics,
            "legacy EMA/source metrics",
        )
        source_legacy_equivalence = {
            "status": "independently_compared",
            "match": True,
            "source_prediction": artifact(source_prediction_path, 0.0),
        }
    else:
        source_legacy_equivalence = {
            "status": "not_applicable_online_weights",
            "match": True,
        }

    independent_delta_report = build_delta_report(
        mode_metrics,
        prediction_counts,
    )
    require_nested_close(
        replay_completion["delta_report"],
        independent_delta_report,
        "producer delta report",
    )

    artifacts = {
        "manifest": artifact(manifest_path, started_at),
        "inference_log": artifact(inference_log_path, started_at),
        "gate": artifact(gate_path, 0.0),
        "checkpoint": artifact(manifest["checkpoint"], 0.0),
        "source_completion": artifact(
            manifest["source_completion"],
            0.0,
        ),
        "source_manifest": artifact(manifest["source_manifest"], 0.0),
        "direct_result": artifact(direct_result_path, started_at),
        "direct_metrics": artifact(direct_metrics_path, started_at),
        "pre_cross_window": artifact(pre_cross_path, started_at),
        "direct_audit": artifact(direct_audit_path, started_at),
        "replay_completion": artifact(replay_completion_path, started_at),
    }
    return {
        "schema_version": "phystime_p0_fullprecision_completion_v2",
        "validation_pass": True,
        "completed_at_unix": time.time(),
        "run_dir": str(run_dir),
        "arm": arm,
        "weights_source": weights_source,
        "evaluation_epoch": EXPECTED_EPOCH,
        "runtime_commit": manifest["runtime_commit"],
        "runtime_tree": manifest["runtime_tree"],
        "source_commit": manifest["source_commit"],
        "source_tree": manifest["source_tree"],
        "new_training": False,
        "direct_fullprecision_filtered_metrics": direct_recomputed,
        "direct_audit_aggregate": direct_aggregate,
        "mode_metrics": mode_metrics,
        "delta_report": independent_delta_report,
        "source_legacy_ema_equivalence": source_legacy_equivalence,
        "artifacts": artifacts,
        "mode_artifacts": mode_artifacts,
    }


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main():
    args = parse_args()
    completion = validate_replay(args.run_dir)
    atomic_write_json(args.output, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
