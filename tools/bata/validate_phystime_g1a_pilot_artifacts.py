from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping
from numbers import Number
from pathlib import Path

from mmengine.config import Config

from opentad.evaluations import build_evaluator
from tools.bata.run_phystime_g1a_real_gate import (
    _canonical_sha256,
    build_dataset_manifest,
    validate_gate_report,
)
from tools.bata.audit_phystime_g0_native_geometry import (
    SCHEMA_VERSION as G0_SCHEMA_VERSION,
)
from tools.bata.validate_phystime_g1a_track import (
    SCHEMA_VERSION as CONTRACT_SCHEMA_VERSION,
)


SCHEMA_VERSION = "phystime_g1a_pilot_completion_v3"
MANIFEST_SCHEMA_VERSION = "phystime_g1a_pilot_manifest_v3"
REQUIRED_METRICS = ("average_mAP", "mAP@0.3", "mAP@0.4", "mAP@0.5", "mAP@0.6", "mAP@0.7")
EXPECTED_EVALUATION_EPOCH = 5
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


def _is_hex_digest(value, length):
    if not isinstance(value, str) or len(value) != int(length):
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_file_hash(path, expected, label):
    path = Path(path).resolve()
    _require(path.is_file(), f"pilot manifest {label} file is missing: {path}")
    _require(_is_hex_digest(expected, 64), f"pilot manifest {label} hash is invalid")
    actual = _sha256_file(path)
    _require(actual == expected, f"pilot manifest {label} hash mismatch")
    return path


def _validate_manifest_bindings(manifest, run_dir):
    _require(_is_hex_digest(manifest.get("commit"), 40), "pilot manifest commit is invalid")
    _require(_is_hex_digest(manifest.get("git_tree"), 40), "pilot manifest git tree is invalid")
    variant = manifest.get("variant")
    _require(variant in {"selected_axis", "physical_metric"}, "pilot manifest variant is invalid")
    for key in (
        "config_sha256",
        "checkpoint_sha256",
        "gate_sha256",
        "contract_sha256",
        "static_g0_sha256",
        "dataset_manifest_sha256",
    ):
        _require(_is_hex_digest(manifest.get(key), 64), f"pilot manifest {key} is invalid")

    runtime_root = Path(manifest.get("runtime_root", "")).resolve()
    _require(runtime_root.is_dir(), f"pilot manifest runtime root is missing: {runtime_root}")
    tree_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=runtime_root, text=True
    )
    _require(tree_status.strip() == "", "pilot runtime Git tree is not clean")
    runtime_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=runtime_root, text=True
    ).strip()
    runtime_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=runtime_root, text=True
    ).strip()
    _require(runtime_commit == manifest["commit"], "pilot manifest commit differs from runtime")
    _require(runtime_tree == manifest["git_tree"], "pilot manifest git tree differs from runtime")

    config_path = Path(manifest.get("config", "")).resolve()
    _require(config_path.is_file(), f"pilot manifest config file is missing: {config_path}")
    try:
        config_path.relative_to(runtime_root)
    except ValueError as exc:
        raise RuntimeError("pilot manifest config is outside the fixed runtime root") from exc
    cfg = Config.fromfile(config_path, lazy_import=False)
    config_sha256 = _canonical_sha256(cfg.to_dict())
    _require(
        config_sha256 == manifest["config_sha256"],
        "pilot manifest config hash mismatch",
    )
    checkpoint_path = _require_file_hash(
        manifest.get("checkpoint", ""), manifest["checkpoint_sha256"], "checkpoint"
    )
    gate_path = _require_file_hash(
        manifest.get("gate", ""), manifest["gate_sha256"], "gate"
    )
    contract_path = _require_file_hash(
        manifest.get("contract", ""), manifest["contract_sha256"], "contract"
    )
    static_g0_path = _require_file_hash(
        manifest.get("static_g0", ""), manifest["static_g0_sha256"], "static G0"
    )
    gate_payload = _read_json_object(gate_path, "real gate")
    validate_gate_report(gate_payload)
    contract_payload = _read_json_object(contract_path, "static contract")
    static_g0_payload = _read_json_object(static_g0_path, "static G0")
    _require(
        contract_payload.get("schema_version") == CONTRACT_SCHEMA_VERSION
        and contract_payload.get("contract_pass") is True,
        "pilot static contract schema/pass mismatch",
    )
    _require(
        static_g0_payload.get("schema_version") == G0_SCHEMA_VERSION
        and static_g0_payload.get("static_precheck_pass") is True
        and static_g0_payload.get("gate_pass") is False,
        "pilot static G0 schema/pass mismatch",
    )
    _require(gate_payload.get("git_commit") == manifest["commit"], "pilot manifest commit differs from gate")
    _require(gate_payload.get("git_tree") == manifest["git_tree"], "pilot manifest git tree differs from gate")
    _require(
        gate_payload.get("dataset_manifest_sha256")
        == manifest["dataset_manifest_sha256"],
        "pilot manifest dataset hash differs from gate",
    )
    _require(
        gate_payload.get("checkpoint_sha256") == manifest["checkpoint_sha256"],
        "pilot manifest checkpoint hash differs from gate",
    )
    _require(
        gate_payload.get("contract_sha256") == manifest["contract_sha256"],
        "pilot manifest contract hash differs from gate",
    )
    _require(
        gate_payload.get("static_g0_sha256") == manifest["static_g0_sha256"],
        "pilot manifest static G0 hash differs from gate",
    )
    _require(
        gate_payload.get("variants", {})
        .get(variant, {})
        .get("canonical_config_sha256")
        == manifest["config_sha256"],
        "pilot manifest variant config differs from gate",
    )
    for label, payload in (
        ("static contract", contract_payload),
        ("static G0", static_g0_payload),
    ):
        _require(
            payload.get("git_commit") == manifest["commit"],
            f"pilot {label} commit differs from runtime",
        )
        _require(
            payload.get("git_tree") == manifest["git_tree"],
            f"pilot {label} git tree differs from runtime",
        )
        _require(
            payload.get("config_sha256", {}).get(variant)
            == manifest["config_sha256"],
            f"pilot {label} variant config differs from runtime",
        )
    evaluation_ground_truth = gate_payload.get("evaluation_ground_truth_filename")
    _require(bool(evaluation_ground_truth), "pilot real gate lacks evaluation ground truth")
    _, recomputed_dataset_sha256 = build_dataset_manifest(
        cfg, evaluation_ground_truth
    )
    _require(
        recomputed_dataset_sha256 == manifest["dataset_manifest_sha256"],
        "pilot dataset manifest differs from the current formal dataset",
    )
    return {
        "variant": variant,
        "runtime_root": str(runtime_root),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "gate": str(gate_path),
        "contract": str(contract_path),
        "static_g0": str(static_g0_path),
    }


def _artifact(path, started_at):
    path = Path(path).resolve()
    _require(path.is_file(), f"required pilot artifact missing: {path}")
    stat = path.stat()
    _require(stat.st_size > 0, f"pilot artifact is empty: {path}")
    _require(stat.st_mtime + 1.0 >= float(started_at), f"pilot artifact predates this run: {path}")
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_unix": float(stat.st_mtime),
        "sha256": _sha256_file(path),
    }


def _read_json_object(path, label):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def _validate_evaluation_epoch(payload, label):
    epoch = payload.get("evaluation_epoch")
    _require(
        isinstance(epoch, int) and not isinstance(epoch, bool) and epoch == EXPECTED_EVALUATION_EPOCH,
        f"{label} evaluation_epoch must equal {EXPECTED_EVALUATION_EPOCH}, got {epoch!r}",
    )


def _validate_metrics(path):
    payload = _read_json_object(path, "evaluation metrics")
    _validate_evaluation_epoch(payload, "evaluation metrics")
    values = {}
    for key in REQUIRED_METRICS:
        _require(key in payload, f"evaluation metrics missing {key}")
        value = float(payload[key])
        _require(math.isfinite(value), f"evaluation metric {key} is non-finite")
        _require(0.0 <= value <= 1.0, f"evaluation metric {key} lies outside [0,1]")
        values[key] = value
    return values


def _validate_predictions(path):
    payload = _read_json_object(path, "prediction JSON")
    _validate_evaluation_epoch(payload, "prediction JSON")
    results = payload.get("results")
    _require(isinstance(results, dict) and results, "prediction JSON must contain a non-empty results object")
    count = 0
    for video_name, detections in results.items():
        _require(isinstance(video_name, str) and video_name, "prediction video id must be a non-empty string")
        _require(isinstance(detections, list), f"predictions for {video_name} must be a list")
        for detection in detections:
            _require(isinstance(detection, dict), f"prediction for {video_name} must be an object")
            segment = detection.get("segment")
            _require(isinstance(segment, list) and len(segment) == 2, "prediction segment must have two values")
            start, end = float(segment[0]), float(segment[1])
            score = float(detection.get("score", float("nan")))
            _require(all(math.isfinite(value) for value in (start, end, score)), "prediction values must be finite")
            _require(end >= start, "prediction segment is reversed")
            _require(0.0 <= score <= 1.0, "prediction score lies outside [0,1]")
            _require(detection.get("label") is not None, "prediction label is missing")
            count += 1
    _require(count > 0, "pilot produced no detections")
    return count, results


def _validate_finite_tree(value, torch, path="checkpoint"):
    if torch.is_tensor(value):
        try:
            finite = bool(torch.isfinite(value).all().item())
        except Exception as exc:
            raise RuntimeError(f"failed to inspect tensor finiteness at {path}: {exc}") from exc
        _require(finite, f"checkpoint contains a non-finite tensor at {path}")
        return

    if isinstance(value, Number):
        if isinstance(value, complex):
            finite = math.isfinite(float(value.real)) and math.isfinite(float(value.imag))
        else:
            finite = math.isfinite(float(value))
        _require(finite, f"checkpoint contains a non-finite numeric value at {path}")
        return

    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite_tree(key, torch, f"{path}.key")
            _validate_finite_tree(child, torch, f"{path}[{key!r}]")
        return

    if isinstance(value, (list, tuple, set)):
        for index, child in enumerate(value):
            _validate_finite_tree(child, torch, f"{path}[{index}]")


def _validate_checkpoint(path):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"failed to import torch for checkpoint validation: {exc}") from exc

    try:
        checkpoint = torch.load(Path(path), map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"failed to load pilot checkpoint {path}: {exc}") from exc
    _require(isinstance(checkpoint, Mapping), "pilot checkpoint must be a mapping")
    _require(
        checkpoint.get("epoch") == EXPECTED_EVALUATION_EPOCH,
        f"pilot checkpoint epoch must equal {EXPECTED_EVALUATION_EPOCH}, got {checkpoint.get('epoch')!r}",
    )
    state_dict = checkpoint.get("state_dict")
    _require(isinstance(state_dict, Mapping) and state_dict, "pilot checkpoint state_dict must be non-empty")
    state_dict_ema = checkpoint.get("state_dict_ema")
    _require(
        isinstance(state_dict_ema, Mapping) and state_dict_ema,
        "pilot checkpoint EMA state_dict must be non-empty",
    )
    _require(
        set(state_dict_ema) == set(state_dict),
        "pilot checkpoint EMA parameter schema differs from the model",
    )
    optimizer = checkpoint.get("optimizer")
    _require(isinstance(optimizer, Mapping), "pilot checkpoint optimizer must be a mapping")
    _require(
        isinstance(optimizer.get("state"), Mapping) and optimizer["state"],
        "pilot checkpoint optimizer.state must be non-empty",
    )
    _require(
        isinstance(optimizer.get("param_groups"), list) and optimizer["param_groups"],
        "pilot checkpoint optimizer.param_groups must be a non-empty list",
    )
    scheduler = checkpoint.get("scheduler")
    _require(
        isinstance(scheduler, Mapping) and scheduler,
        "pilot checkpoint scheduler must be non-empty",
    )
    optimizer_parameter_ids = [
        parameter_id
        for group in optimizer["param_groups"]
        for parameter_id in group.get("params", [])
    ]
    _require(
        optimizer_parameter_ids
        and len(optimizer_parameter_ids) == len(set(optimizer_parameter_ids)),
        "pilot checkpoint optimizer parameter groups are empty or duplicated",
    )
    optimizer_state = optimizer["state"]
    _require(
        set(optimizer_state) == set(optimizer_parameter_ids),
        "pilot checkpoint optimizer state does not cover every optimized parameter",
    )
    optimizer_steps = []
    for parameter_id in optimizer_parameter_ids:
        state = optimizer_state[parameter_id]
        _require(
            isinstance(state, Mapping) and "step" in state,
            "pilot checkpoint optimizer state lacks a step counter",
        )
        step = state["step"]
        optimizer_steps.append(int(step.item()) if torch.is_tensor(step) else int(step))
    _require(
        min(optimizer_steps) == max(optimizer_steps) and min(optimizer_steps) > 0,
        "pilot checkpoint optimizer steps are inconsistent",
    )
    scheduler_last_epoch = int(scheduler.get("last_epoch", -1))
    _require(
        scheduler_last_epoch == optimizer_steps[0],
        "pilot checkpoint scheduler step differs from optimizer state",
    )
    _validate_finite_tree(checkpoint, torch)
    return {
        "epoch": EXPECTED_EVALUATION_EPOCH,
        "state_dict_entries": len(state_dict),
        "state_dict_ema_entries": len(state_dict_ema),
        "optimizer_param_groups": len(optimizer["param_groups"]),
        "optimizer_parameter_count": len(optimizer_parameter_ids),
        "optimizer_step": optimizer_steps[0],
        "scheduler_last_epoch": scheduler_last_epoch,
    }


def _recompute_metrics(config_path, results, reported_metrics):
    config_path = Path(config_path).expanduser().resolve()
    _require(config_path.is_file(), f"formal pilot config missing: {config_path}")
    try:
        cfg = Config.fromfile(str(config_path), lazy_import=False)
    except Exception as exc:
        raise RuntimeError(f"failed to load formal pilot config {config_path}: {exc}") from exc
    _require("evaluation" in cfg, f"formal pilot config has no evaluation section: {config_path}")
    evaluation_cfg = dict(cfg.evaluation)
    evaluation_cfg.pop("output_metrics_path", None)
    evaluation_cfg["prediction_filename"] = {"results": results}
    try:
        evaluator = build_evaluator(evaluation_cfg)
        recomputed_payload = evaluator.evaluate()
    except Exception as exc:
        raise RuntimeError(f"failed to recompute pilot metrics from {config_path}: {exc}") from exc
    _require(isinstance(recomputed_payload, Mapping), "recomputed evaluation metrics must be a mapping")

    recomputed = {}
    for key in REQUIRED_METRICS:
        _require(key in recomputed_payload, f"recomputed evaluation metrics missing {key}")
        value = float(recomputed_payload[key])
        _require(math.isfinite(value), f"recomputed evaluation metric {key} is non-finite")
        _require(0.0 <= value <= 1.0, f"recomputed evaluation metric {key} lies outside [0,1]")
        _require(
            math.isclose(
                value,
                reported_metrics[key],
                rel_tol=METRIC_REL_TOL,
                abs_tol=METRIC_ABS_TOL,
            ),
            f"evaluation metric {key} does not match recomputation: "
            f"reported={reported_metrics[key]:.12g}, recomputed={value:.12g}",
        )
        recomputed[key] = value
    return recomputed


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_pilot_artifacts(run_dir, *, output=None):
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    _require(manifest_path.is_file(), f"pilot manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(manifest, dict), "pilot manifest must be a JSON object")
    _require(manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION, "pilot manifest schema mismatch")
    _require(int(manifest.get("pilot_epochs", -1)) == 6, "formal G1a pilot must contain exactly six epochs")
    _require(int(manifest.get("warmup_epochs", -1)) < 6, "formal G1a pilot must cross warmup")
    for key in (
        "commit",
        "git_tree",
        "runtime_root",
        "variant",
        "config",
        "config_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "gate",
        "gate_sha256",
        "contract",
        "contract_sha256",
        "static_g0",
        "static_g0_sha256",
        "dataset_manifest_sha256",
    ):
        _require(bool(manifest.get(key)), f"pilot manifest missing {key}")
    manifest_bindings = _validate_manifest_bindings(manifest, run_dir)
    started_at = float(manifest.get("started_at_unix", float("nan")))
    _require(math.isfinite(started_at), "pilot manifest has an invalid start time")

    work_dir = run_dir / "work_dir" / "gpu1_id0"
    result_path = work_dir / "result_detection.json"
    metrics_path = work_dir / "evaluation_metrics.json"
    checkpoint_path = work_dir / "checkpoint" / "epoch_5.pth"
    artifacts = {
        "manifest": _artifact(manifest_path, started_at),
        "predictions": _artifact(result_path, started_at),
        "metrics": _artifact(metrics_path, started_at),
        "checkpoint": _artifact(checkpoint_path, started_at),
    }
    prediction_count, results = _validate_predictions(result_path)
    reported_metrics = _validate_metrics(metrics_path)
    checkpoint_contract = _validate_checkpoint(checkpoint_path)
    metrics = _recompute_metrics(manifest["config"], results, reported_metrics)
    completion = {
        "schema_version": SCHEMA_VERSION,
        "validation_pass": True,
        "completed_at_unix": time.time(),
        "run_dir": str(run_dir),
        "effective_work_dir": str(work_dir),
        "prediction_count": prediction_count,
        "metrics": metrics,
        "checkpoint_contract": checkpoint_contract,
        "manifest_bindings": manifest_bindings,
        "manifest_contract": {
            key: manifest[key]
            for key in (
                "commit",
                "git_tree",
                "config_sha256",
                "dataset_manifest_sha256",
                "gate_sha256",
                "pilot_epochs",
                "warmup_epochs",
            )
        },
        "artifacts": artifacts,
    }
    if output is not None:
        _atomic_write_json(output, completion)
    return completion


def parse_args():
    parser = argparse.ArgumentParser(description="Validate formal PhysTime G1a pilot artifacts")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    completion = validate_pilot_artifacts(args.run_dir, output=args.output)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
