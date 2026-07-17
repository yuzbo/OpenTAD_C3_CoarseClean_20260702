from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

from mmengine.config import Config

from opentad.evaluations import build_evaluator
from tools.bata.run_phystime_g1a_real_gate import (
    _canonical_sha256,
    build_dataset_manifest,
    validate_gate_report,
)


SCHEMA_VERSION = "phystime_g1_matched_medium_completion_v1"
MANIFEST_SCHEMA_VERSION = "phystime_g1_matched_medium_manifest_v1"
G1B_GATE_SCHEMA_VERSION = "phystime_g1b_sdpq_real_gate_v1"
REQUIRED_METRICS = (
    "average_mAP",
    "mAP@0.3",
    "mAP@0.4",
    "mAP@0.5",
    "mAP@0.6",
    "mAP@0.7",
)
VARIANT_CONFIGS = {
    "selected_axis": "phystime_g1a_selected_axis_native_j192.py",
    "physical_metric": "phystime_g1a_physical_metric_native_j192.py",
    "g1b_sdpq": "phystime_g1b_sdpq_pool_native_j192.py",
}
METRIC_REL_TOL = 1.0e-6
METRIC_ABS_TOL = 1.0e-8


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path, label):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def _artifact(path, started_at):
    path = Path(path).resolve()
    _require(path.is_file(), f"required artifact is missing: {path}")
    stat = path.stat()
    _require(stat.st_size > 0, f"artifact is empty: {path}")
    _require(
        stat.st_mtime + 1.0 >= float(started_at),
        f"artifact predates this run: {path}",
    )
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_unix": float(stat.st_mtime),
        "sha256": _sha256_file(path),
    }


def _validate_finite_tree(value, torch, path="checkpoint"):
    if torch.is_tensor(value):
        _require(
            bool(torch.isfinite(value).all().item()),
            f"checkpoint contains a non-finite tensor at {path}",
        )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite_tree(child, torch, f"{path}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite_tree(child, torch, f"{path}[{index}]")
        return
    if isinstance(value, (int, float)):
        _require(math.isfinite(float(value)), f"non-finite numeric value at {path}")


def _validate_lightweight_checkpoint(path, final_epoch):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"failed to import torch for checkpoint validation: {exc}") from exc

    try:
        checkpoint = torch.load(Path(path), map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"failed to load medium checkpoint {path}: {exc}") from exc
    _require(isinstance(checkpoint, Mapping), "medium checkpoint must be a mapping")
    _require(
        int(checkpoint.get("epoch", -1)) == int(final_epoch),
        f"checkpoint epoch must be {final_epoch}",
    )
    state_dict = checkpoint.get("state_dict")
    _require(isinstance(state_dict, Mapping) and state_dict, "checkpoint state_dict is empty")
    state_dict_ema = checkpoint.get("state_dict_ema")
    _require(
        isinstance(state_dict_ema, Mapping) and state_dict_ema,
        "checkpoint state_dict_ema is empty",
    )
    _require(
        set(state_dict_ema) == set(state_dict),
        "checkpoint EMA and online state_dict keys differ",
    )
    for forbidden in ("optimizer", "scheduler"):
        _require(
            forbidden not in checkpoint,
            f"lightweight medium checkpoint unexpectedly contains {forbidden}",
        )
    _validate_finite_tree(state_dict, torch, "checkpoint.state_dict")
    _validate_finite_tree(state_dict_ema, torch, "checkpoint.state_dict_ema")
    return {
        "epoch": int(final_epoch),
        "state_dict_entries": len(state_dict),
        "state_dict_ema_entries": len(state_dict_ema),
        "lightweight": True,
        "evaluated_weights_replayable": True,
    }


def _validate_predictions(path, final_epoch):
    payload = _read_json(path, "prediction JSON")
    _require(
        int(payload.get("evaluation_epoch", -1)) == int(final_epoch),
        f"prediction evaluation_epoch must be {final_epoch}",
    )
    results = payload.get("results")
    _require(isinstance(results, dict) and results, "prediction results must be non-empty")
    count = 0
    for video_name, detections in results.items():
        _require(isinstance(video_name, str) and video_name, "invalid prediction video id")
        _require(isinstance(detections, list), f"predictions for {video_name} must be a list")
        for detection in detections:
            _require(isinstance(detection, dict), "each detection must be an object")
            segment = detection.get("segment")
            _require(isinstance(segment, list) and len(segment) == 2, "invalid segment")
            start, end = float(segment[0]), float(segment[1])
            score = float(detection.get("score", float("nan")))
            _require(
                all(math.isfinite(value) for value in (start, end, score)),
                "prediction contains a non-finite value",
            )
            _require(end >= start, "prediction segment is reversed")
            _require(0.0 <= score <= 1.0, "prediction score lies outside [0,1]")
            _require(detection.get("label") is not None, "prediction label is missing")
            count += 1
    _require(count > 0, "medium run produced no detections")
    return count, results


def _validate_reported_metrics(path, final_epoch):
    payload = _read_json(path, "evaluation metrics")
    _require(
        int(payload.get("evaluation_epoch", -1)) == int(final_epoch),
        f"metric evaluation_epoch must be {final_epoch}",
    )
    values = {}
    for key in REQUIRED_METRICS:
        _require(key in payload, f"evaluation metrics missing {key}")
        value = float(payload[key])
        _require(math.isfinite(value), f"evaluation metric {key} is non-finite")
        _require(0.0 <= value <= 1.0, f"evaluation metric {key} lies outside [0,1]")
        values[key] = value
    return values


def _recompute_metrics(config_path, results, reported_metrics):
    cfg = Config.fromfile(str(config_path), lazy_import=False)
    evaluation_cfg = dict(cfg.evaluation)
    evaluation_cfg.pop("output_metrics_path", None)
    evaluation_cfg["prediction_filename"] = {"results": results}
    evaluator = build_evaluator(evaluation_cfg)
    recomputed_payload = evaluator.evaluate()
    _require(isinstance(recomputed_payload, Mapping), "recomputed metrics must be a mapping")
    recomputed = {}
    for key in REQUIRED_METRICS:
        _require(key in recomputed_payload, f"recomputed metrics missing {key}")
        value = float(recomputed_payload[key])
        _require(
            math.isclose(
                value,
                reported_metrics[key],
                rel_tol=METRIC_REL_TOL,
                abs_tol=METRIC_ABS_TOL,
            ),
            f"{key} mismatch: reported={reported_metrics[key]}, recomputed={value}",
        )
        recomputed[key] = value
    return recomputed


def _resolve_gate_config(runtime_root, gate_config):
    path = Path(gate_config)
    if not path.is_absolute():
        path = runtime_root / path
    return path.resolve()


def _validate_bindings(manifest):
    variant = manifest.get("variant")
    _require(variant in VARIANT_CONFIGS, f"unsupported medium variant: {variant}")
    epochs = int(manifest.get("epochs", -1))
    final_epoch = int(manifest.get("final_epoch", -1))
    _require(epochs == 20 and final_epoch == 19, "matched medium contract requires 20 epochs")
    _require(int(manifest.get("seed", -1)) == 42, "matched medium seed must be 42")
    _require(int(manifest.get("K_raw_observations", -1)) == 384, "K must be 384")
    _require(int(manifest.get("J_native_tubelet_tokens", -1)) == 192, "J must be 192")
    _require(manifest.get("feature_interpolation") is False, "interpolation must be disabled")

    runtime_root = Path(manifest.get("runtime_root", "")).resolve()
    _require(runtime_root.is_dir(), f"runtime root is missing: {runtime_root}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=runtime_root, text=True
    )
    _require(status.strip() == "", "runtime snapshot is not clean")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=runtime_root, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=runtime_root, text=True
    ).strip()
    _require(commit == manifest.get("commit"), "manifest commit differs from runtime")
    _require(tree == manifest.get("git_tree"), "manifest tree differs from runtime")

    config_path = Path(manifest.get("config", "")).resolve()
    _require(config_path.name == VARIANT_CONFIGS[variant], "variant/config mismatch")
    try:
        config_path.relative_to(runtime_root)
    except ValueError as exc:
        raise RuntimeError("config lies outside the fixed runtime snapshot") from exc
    cfg = Config.fromfile(str(config_path), lazy_import=False)
    config_sha256 = _canonical_sha256(cfg.to_dict())
    _require(config_sha256 == manifest.get("config_sha256"), "config hash mismatch")

    checkpoint_path = Path(manifest.get("pretrained_checkpoint", "")).resolve()
    _require(checkpoint_path.is_file(), "pretrained checkpoint is missing")
    _require(
        _sha256_file(checkpoint_path) == manifest.get("pretrained_checkpoint_sha256"),
        "pretrained checkpoint hash mismatch",
    )

    g1a_gate_path = Path(manifest.get("g1a_gate", "")).resolve()
    g1b_gate_path = Path(manifest.get("g1b_gate", "")).resolve()
    _require(_sha256_file(g1a_gate_path) == manifest.get("g1a_gate_sha256"), "G1a gate hash mismatch")
    _require(_sha256_file(g1b_gate_path) == manifest.get("g1b_gate_sha256"), "G1b gate hash mismatch")
    g1a_gate = _read_json(g1a_gate_path, "G1a gate")
    g1b_gate = _read_json(g1b_gate_path, "G1b gate")
    validate_gate_report(g1a_gate)
    _require(
        g1b_gate.get("schema_version") == G1B_GATE_SCHEMA_VERSION
        and g1b_gate.get("gate_pass") is True,
        "G1b gate did not pass",
    )
    for gate in (g1a_gate, g1b_gate):
        _require(gate.get("git_commit") == commit, "gate commit differs from runtime")
        _require(gate.get("git_tree") == tree, "gate tree differs from runtime")
    _require(g1b_gate.get("feature_interpolation") is False, "G1b gate used interpolation")
    _require(int(g1b_gate.get("gt_without_assigned_query", -1)) == 0, "G1b gate missed GT assignment")
    _require(
        int(g1b_gate.get("short_gt_without_assigned_query", -1)) == 0,
        "G1b gate missed short-GT assignment",
    )

    if variant in {"selected_axis", "physical_metric"}:
        _require(
            g1a_gate.get("variants", {})
            .get(variant, {})
            .get("canonical_config_sha256")
            == config_sha256,
            "G1a gate config differs from medium config",
        )
    else:
        _require(
            _resolve_gate_config(runtime_root, g1b_gate.get("config", "")) == config_path,
            "G1b gate config differs from medium config",
        )

    ground_truth = g1a_gate.get("evaluation_ground_truth_filename")
    _require(bool(ground_truth), "G1a gate lacks evaluation ground truth")
    _, dataset_manifest_sha256 = build_dataset_manifest(cfg, ground_truth)
    _require(
        dataset_manifest_sha256 == g1a_gate.get("dataset_manifest_sha256"),
        "medium dataset differs from G1a gate dataset",
    )
    _require(
        dataset_manifest_sha256 == manifest.get("dataset_manifest_sha256"),
        "medium dataset differs from manifest",
    )
    return {
        "variant": variant,
        "runtime_root": str(runtime_root),
        "config": str(config_path),
        "pretrained_checkpoint": str(checkpoint_path),
        "g1a_gate": str(g1a_gate_path),
        "g1b_gate": str(g1b_gate_path),
    }


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_medium_artifacts(run_dir, *, output=None):
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path, "medium manifest")
    _require(
        manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION,
        "medium manifest schema mismatch",
    )
    bindings = _validate_bindings(manifest)
    started_at = float(manifest.get("started_at_unix", float("nan")))
    _require(math.isfinite(started_at), "manifest start time is invalid")
    final_epoch = int(manifest["final_epoch"])

    work_dir = run_dir / "work_dir" / "gpu1_id0"
    prediction_path = work_dir / "result_detection.json"
    metrics_path = work_dir / "evaluation_metrics.json"
    checkpoint_path = work_dir / "checkpoint" / f"epoch_{final_epoch}.pth"
    artifacts = {
        "manifest": _artifact(manifest_path, started_at),
        "predictions": _artifact(prediction_path, started_at),
        "metrics": _artifact(metrics_path, started_at),
        "checkpoint": _artifact(checkpoint_path, started_at),
    }
    prediction_count, results = _validate_predictions(prediction_path, final_epoch)
    reported_metrics = _validate_reported_metrics(metrics_path, final_epoch)
    recomputed_metrics = _recompute_metrics(
        manifest["config"], results, reported_metrics
    )
    checkpoint_contract = _validate_lightweight_checkpoint(
        checkpoint_path, final_epoch
    )
    completion = {
        "schema_version": SCHEMA_VERSION,
        "validation_pass": True,
        "completed_at_unix": time.time(),
        "run_dir": str(run_dir),
        "variant": manifest["variant"],
        "epochs": int(manifest["epochs"]),
        "evaluation_epoch": final_epoch,
        "prediction_count": prediction_count,
        "metrics": recomputed_metrics,
        "checkpoint_contract": checkpoint_contract,
        "bindings": bindings,
        "artifacts": artifacts,
    }
    if output is not None:
        _atomic_write_json(output, completion)
    return completion


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate PhysTime G1 matched 20-epoch medium artifacts"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    completion = validate_medium_artifacts(args.run_dir, output=args.output)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
