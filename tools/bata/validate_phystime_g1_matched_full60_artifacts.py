from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

from mmengine.config import Config

from tools.bata.run_phystime_g1a_real_gate import (
    _canonical_sha256,
    build_dataset_manifest,
    validate_gate_report,
)
from tools.bata.validate_phystime_g1_matched_medium_artifacts import (
    _artifact,
    _read_json,
    _recompute_metrics,
    _require,
    _sha256_file,
    _validate_lightweight_checkpoint,
    _validate_predictions,
    _validate_reported_metrics,
)


SCHEMA_VERSION = "phystime_g1_matched_full60_completion_v1"
MANIFEST_SCHEMA_VERSION = "phystime_g1_matched_full60_manifest_v1"
EXPECTED_EPOCHS = 60
EXPECTED_FINAL_EPOCH = 59
EXPECTED_SEED = 42
VARIANT_CONFIGS = {
    "selected_axis": "phystime_g1a_selected_axis_native_j192.py",
    "physical_metric": "phystime_g1a_physical_metric_native_j192.py",
}


def _validate_bindings(manifest):
    variant = manifest.get("variant")
    _require(variant in VARIANT_CONFIGS, f"unsupported full60 variant: {variant}")
    _require(
        int(manifest.get("epochs", -1)) == EXPECTED_EPOCHS
        and int(manifest.get("final_epoch", -1)) == EXPECTED_FINAL_EPOCH,
        "matched full60 contract requires exactly 60 epochs",
    )
    _require(int(manifest.get("seed", -1)) == EXPECTED_SEED, "matched full60 seed must be 42")
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

    gate_path = Path(manifest.get("g1a_gate", "")).resolve()
    _require(_sha256_file(gate_path) == manifest.get("g1a_gate_sha256"), "G1a gate hash mismatch")
    gate = _read_json(gate_path, "G1a gate")
    validate_gate_report(gate)
    _require(gate.get("git_commit") == commit, "gate commit differs from runtime")
    _require(gate.get("git_tree") == tree, "gate tree differs from runtime")
    _require(
        gate.get("variants", {}).get(variant, {}).get("canonical_config_sha256")
        == config_sha256,
        "G1a gate config differs from full60 config",
    )

    ground_truth = gate.get("evaluation_ground_truth_filename")
    _require(bool(ground_truth), "G1a gate lacks evaluation ground truth")
    _, dataset_manifest_sha256 = build_dataset_manifest(cfg, ground_truth)
    _require(
        dataset_manifest_sha256 == gate.get("dataset_manifest_sha256"),
        "full60 dataset differs from G1a gate dataset",
    )
    _require(
        dataset_manifest_sha256 == manifest.get("dataset_manifest_sha256"),
        "full60 dataset differs from manifest",
    )
    run_dir = Path(manifest.get("run_dir", "")).resolve()
    expected_overrides = {
        "model.backbone.custom.pretrain": str(checkpoint_path),
        "work_dir": str(run_dir / "work_dir"),
        "scheduler.max_epoch": EXPECTED_EPOCHS,
        "workflow.end_epoch": EXPECTED_EPOCHS,
        "workflow.val_start_epoch": 40,
        "workflow.val_eval_interval": 2,
        "workflow.checkpoint_interval": EXPECTED_EPOCHS,
        "workflow.checkpoint_save_mode": "lightweight",
        "workflow.checkpoint_include_ema": True,
        "post_processing.save_dict": True,
    }
    _require(
        manifest.get("effective_overrides") == expected_overrides,
        "effective full60 overrides differ from the frozen contract",
    )
    cfg.merge_from_dict(expected_overrides)
    _require(int(cfg.scheduler.max_epoch) == EXPECTED_EPOCHS, "scheduler must end at epoch 60")
    _require(int(cfg.workflow.end_epoch) == EXPECTED_EPOCHS, "workflow must end at epoch 60")
    _require(
        _canonical_sha256(cfg.to_dict()) == manifest.get("effective_config_sha256"),
        "effective full60 config hash mismatch",
    )
    return {
        "variant": variant,
        "runtime_root": str(runtime_root),
        "config": str(config_path),
        "pretrained_checkpoint": str(checkpoint_path),
        "g1a_gate": str(gate_path),
        "effective_config_sha256": manifest["effective_config_sha256"],
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


def validate_full60_artifacts(run_dir, *, output=None):
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path, "full60 manifest")
    _require(
        manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION,
        "full60 manifest schema mismatch",
    )
    bindings = _validate_bindings(manifest)
    started_at = float(manifest.get("started_at_unix", float("nan")))
    _require(math.isfinite(started_at), "manifest start time is invalid")

    work_dir = run_dir / "work_dir" / "gpu1_id0"
    prediction_path = work_dir / "result_detection.json"
    metrics_path = work_dir / "evaluation_metrics.json"
    checkpoint_path = work_dir / "checkpoint" / f"epoch_{EXPECTED_FINAL_EPOCH}.pth"
    artifacts = {
        "manifest": _artifact(manifest_path, started_at),
        "predictions": _artifact(prediction_path, started_at),
        "metrics": _artifact(metrics_path, started_at),
        "checkpoint": _artifact(checkpoint_path, started_at),
    }
    prediction_count, results = _validate_predictions(
        prediction_path, EXPECTED_FINAL_EPOCH
    )
    reported_metrics = _validate_reported_metrics(
        metrics_path, EXPECTED_FINAL_EPOCH
    )
    recomputed_metrics = _recompute_metrics(
        manifest["config"], results, reported_metrics
    )
    checkpoint_contract = _validate_lightweight_checkpoint(
        checkpoint_path, EXPECTED_FINAL_EPOCH
    )
    completion = {
        "schema_version": SCHEMA_VERSION,
        "validation_pass": True,
        "completed_at_unix": time.time(),
        "run_dir": str(run_dir),
        "variant": manifest["variant"],
        "epochs": EXPECTED_EPOCHS,
        "evaluation_epoch": EXPECTED_FINAL_EPOCH,
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
        description="Validate PhysTime G1 matched 60-epoch full-run artifacts"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    completion = validate_full60_artifacts(args.run_dir, output=args.output)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
