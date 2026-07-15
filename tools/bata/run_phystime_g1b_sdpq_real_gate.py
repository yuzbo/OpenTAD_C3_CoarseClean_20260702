from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from mmengine.config import Config
from torch.cuda.amp import GradScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.cores import build_optimizer
from opentad.datasets import build_dataset
from opentad.datasets.builder import collate
from opentad.models import build_detector


SCHEMA_VERSION = "phystime_g1b_sdpq_real_gate_v1"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import imgaug

        imgaug.seed(seed)
    except ImportError:
        pass


def _move_batch(batch, device):
    batch["inputs"] = batch["inputs"].to(device, non_blocking=False)
    batch["masks"] = batch["masks"].to(device, non_blocking=False)
    batch["gt_segments"] = [value.to(device) for value in batch["gt_segments"]]
    batch["gt_labels"] = [value.to(device) for value in batch["gt_labels"]]
    return batch


def _finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


def _gradient_stats(named_parameters, predicate):
    matched = [(name, parameter) for name, parameter in named_parameters if predicate(name, parameter)]
    finite = [
        (name, parameter)
        for name, parameter in matched
        if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
    ]
    nonzero = [
        (name, parameter)
        for name, parameter in finite
        if float(parameter.grad.detach().abs().sum().item()) > 0.0
    ]
    return {
        "parameter_count": len(matched),
        "finite_gradient_count": len(finite),
        "nonzero_gradient_count": len(nonzero),
        "all_finite": len(finite) == len(matched),
        "nonzero": bool(nonzero),
        "gradient_l1": float(sum(parameter.grad.detach().abs().sum().item() for _, parameter in nonzero)),
    }


def _optimizer_coverage(cfg, model):
    logger = logging.getLogger("phystime_g1b_sdpq_gate")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    optimizer = build_optimizer(
        copy.deepcopy(cfg.optimizer),
        SimpleNamespace(module=model),
        logger,
    )
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    required = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (not name.startswith("backbone.") or "adapter" in name.lower())
    ]
    return optimizer, {
        "optimizer_parameter_count": len(optimized),
        "required_parameter_count": len(required),
        "duplicate_parameter_count": len(optimized) - len({id(parameter) for parameter in optimized}),
        "covered": {id(parameter) for parameter in optimized} == {id(parameter) for parameter in required}
        and len(optimized) == len({id(parameter) for parameter in optimized}),
    }


def _choose_sample(dataset, requested_index):
    if requested_index >= 0:
        _require(requested_index < len(dataset), "requested sample index is outside the training dataset")
        return requested_index
    snippet_stride = int(dataset.snippet_stride)
    required_frames = 768 * snippet_stride
    for index, row in enumerate(dataset.data_list):
        info = row[1]
        if int(info.get("frame", 0)) >= required_frames:
            return index
    raise RuntimeError("no THUMOS training video can provide a full 768 logical window")


def _validate_config(cfg):
    _require(cfg.model.type == "PhysTimeTAD", "G1b SDPQ must use PhysTimeTAD")
    _require(
        cfg.model.projection.type == "PhysTimeMeasureProjection",
        "G1b SDPQ must use PhysTimeMeasureProjection",
    )
    _require(
        cfg.model.projection.get("keep_uncovered_queries", False) is True,
        "G1b SDPQ must keep uncovered physical queries",
    )
    _require(
        cfg.model.rpn_head.type == "SupportDecoupledPhysicalQueryHead",
        "G1b SDPQ must use SupportDecoupledPhysicalQueryHead",
    )
    post_types = [step["type"] for step in cfg.model.backbone.custom.post_processing_pipeline]
    _require("Interpolate" not in post_types, "G1b SDPQ forbids J192-to-K384 feature interpolation")
    _require(int(cfg.model.native_temporal_geometry.expected_raw_count) == 384, "expected K=384")
    _require(int(cfg.model.native_temporal_geometry.expected_token_count) == 192, "expected J=192")
    for split in ("train", "val", "test"):
        native = next(
            step
            for step in cfg.dataset[split].pipeline
            if step["type"] == "BuildPhysTimeNativeTubeletGeometry"
        )
        _require(native["coordinate_mode"] == "physical_time_seconds", "G1b must use physical seconds metadata")


def run_gate(config, checkpoint, device, expected_commit=None, expected_tree=None, seed=42, sample_index=-1):
    config = Path(config).resolve()
    checkpoint = Path(checkpoint).resolve()
    _require(config.is_file(), f"config not found: {config}")
    _require(checkpoint.is_file(), f"checkpoint not found: {checkpoint}")
    _require(device.type == "cuda" and torch.cuda.is_available(), "G1b SDPQ real gate requires CUDA")

    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip()
    if expected_commit is not None:
        _require(git_commit == str(expected_commit), "runtime commit differs from submitted commit")
    if expected_tree is not None:
        _require(git_tree == str(expected_tree), "runtime tree differs from submitted tree")

    cfg = Config.fromfile(str(config), lazy_import=False)
    _validate_config(cfg)
    cfg.model.backbone.custom.pretrain = str(checkpoint)
    _seed(seed)
    dataset = build_dataset(cfg.dataset.train)
    index = _choose_sample(dataset, int(sample_index))
    _seed(seed)
    sample = dataset[index]
    _require(int(sample["inputs"].shape[2]) == 384, "decoded raw observation count must be K=384")
    _require(int(sample["masks"].sum().item()) == 384, "gate sample must contain 384 valid raw observations")
    batch = _move_batch(collate([sample]), device)

    _seed(seed)
    model = build_detector(cfg.model).to(device).train()
    optimizer, optimizer_report = _optimizer_coverage(cfg, model)
    scaler = GradScaler(enabled=bool(cfg.solver.get("amp", False)), init_scale=float(cfg.solver.get("amp_init_scale", 1024.0)))
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.cuda.amp.autocast(enabled=bool(cfg.solver.get("amp", False))):
        losses = model(
            inputs=batch["inputs"],
            masks=batch["masks"],
            metas=batch["metas"],
            gt_segments=batch["gt_segments"],
            gt_labels=batch["gt_labels"],
            return_loss=True,
        )
    _require("cost" in losses and _finite_tree(losses), "G1b SDPQ produced non-finite losses")
    scaler.scale(losses["cost"]).backward()
    scaler.unscale_(optimizer)
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    _require(
        all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item()) for parameter in optimized),
        "G1b SDPQ produced non-finite gradients",
    )
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    _require(scaler.get_scale() >= scale_before, "G1b SDPQ AMP skipped optimizer step")
    torch.cuda.synchronize(device)
    train_ms = (time.perf_counter() - started) * 1000.0
    head_debug = model.rpn_head.collect_debug_state()

    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=bool(cfg.solver.get("amp", False))):
            predictions = model.forward_test(batch["inputs"], batch["masks"], batch["metas"])
    _require(_finite_tree(predictions), "G1b SDPQ produced non-finite predictions")
    proposals, scores = predictions
    native_audit = model.collect_native_temporal_geometry_audit()
    named_parameters = list(model.named_parameters())
    gradients = {
        "adapter": _gradient_stats(named_parameters, lambda name, p: name.startswith("backbone.") and "adapter" in name.lower() and p.requires_grad),
        "projection": _gradient_stats(named_parameters, lambda name, p: name.startswith("projection.") and p.requires_grad),
        "null_evidence": _gradient_stats(named_parameters, lambda name, p: name.endswith("null_evidence") and p.requires_grad),
        "classification": _gradient_stats(named_parameters, lambda name, p: name.startswith("rpn_head.cls_head.") and p.requires_grad),
        "regression": _gradient_stats(named_parameters, lambda name, p: name.startswith("rpn_head.reg_head.") and p.requires_grad),
        "endpoint": _gradient_stats(named_parameters, lambda name, p: name.startswith("rpn_head.endpoint_head.") and p.requires_grad),
    }
    assignment_rows = head_debug.get("target_assignment", [])
    gt_without = sum(int(row.get("gt_without_assigned_query", 0)) for row in assignment_rows)
    short_without = sum(int(row.get("short_gt_without_assigned_query", 0)) for row in assignment_rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate_pass": True,
        "git_commit": git_commit,
        "git_tree": git_tree,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "sample_index": int(index),
        "sample_video": str(batch["metas"][0]["video_name"]),
        "K_raw_observations": 384,
        "J_native_tubelet_tokens": int(native_audit.get("native_token_count", -1)),
        "feature_interpolation": native_audit.get("feature_interpolation"),
        "query_tensor_count": int(native_audit.get("query_tensor_count", -1)),
        "prediction_time_unit": batch["metas"][0].get("prediction_time_unit"),
        "optimizer": optimizer_report,
        "optimizer_step_count": 1,
        "losses": {key: float(value.detach().cpu().item()) for key, value in losses.items()},
        "finite_predictions": True,
        "proposal_count": int(proposals[0].shape[0]),
        "score_shape": list(scores[0].shape),
        "gradients": gradients,
        "native_geometry_audit": native_audit,
        "head_debug": head_debug,
        "gt_without_assigned_query": int(gt_without),
        "short_gt_without_assigned_query": int(short_without),
        "train_forward_backward_ms": float(train_ms),
        "peak_cuda_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0**2)),
    }
    required_nonzero_gradients = ("adapter", "projection", "classification", "regression", "endpoint")
    report["gate_pass"] = bool(
        optimizer_report["covered"]
        and report["J_native_tubelet_tokens"] == 192
        and report["feature_interpolation"] is False
        and report["query_tensor_count"] > 0
        and report["proposal_count"] > 0
        and gt_without == 0
        and short_without == 0
        and report["gradients"]["null_evidence"]["parameter_count"] > 0
        and report["gradients"]["null_evidence"]["all_finite"]
        and all(report["gradients"][key]["nonzero"] for key in required_nonzero_gradients)
        and all(report["gradients"][key]["all_finite"] for key in required_nonzero_gradients)
    )
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Run the PhysTime G1b SDPQ real THUMOS gate")
    parser.add_argument("--config", default="configs/adatad/thumos/phystime_g1b_sdpq_pool_native_j192.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--expected-tree", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-index", type=int, default=-1)
    return parser.parse_args()


def _write(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    try:
        report = run_gate(
            args.config,
            args.checkpoint,
            torch.device(args.device),
            expected_commit=args.expected_commit,
            expected_tree=args.expected_tree,
            seed=args.seed,
            sample_index=args.sample_index,
        )
    except Exception as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "gate_pass": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        raise SystemExit(1) from error
    _write(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
