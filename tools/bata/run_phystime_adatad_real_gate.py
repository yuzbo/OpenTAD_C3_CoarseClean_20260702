from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import logging
import random
import subprocess
import sys
import tempfile
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
from opentad.evaluations import build_evaluator
from opentad.models import build_detector


GATE_CONFIGS = {
    "selected_axis": ROOT / "configs/adatad/thumos/selected_axis_adatad_sparse_k384.py",
    "physical_grid": ROOT / "configs/adatad/thumos/physical_grid_adatad_sparse_k384.py",
    "phystime": ROOT / "configs/adatad/thumos/phystime_adatad_sparse_k384.py",
}
SCHEMA_VERSION = "phystime_adatad_real_gate_v2"
TARGET_LEN = 384
LOGICAL_WINDOW = 768
OPTIMIZER_STEPS = 3
_AUGMENTATION_LIBRARIES_WARMED = False


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _seed_core(seed):
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


def _seed_everything(seed):
    global _AUGMENTATION_LIBRARIES_WARMED

    _seed_core(seed)
    if not _AUGMENTATION_LIBRARIES_WARMED:
        from mmaction.datasets.transforms import ColorJitter, ImgAug

        ImgAug(transforms="default")
        ColorJitter()
        _AUGMENTATION_LIBRARIES_WARMED = True
        _seed_core(seed)
    try:
        import cv2

        cv2.setRNGSeed(seed)
    except ImportError:
        pass


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return _sha256_bytes(payload)


def _tensor_sha256(value):
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    header = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode("utf-8")
    return _sha256_bytes(header + tensor.numpy().tobytes())


def _selected_index_checksum(meta):
    values = np.asarray(meta.get("selected_raw_frame_indices", []), dtype=np.int64).reshape(-1)
    _require(values.size == TARGET_LEN, f"expected {TARGET_LEN} selected raw frames, got {values.size}")
    _require(np.all(np.diff(values) > 0), "selected raw frame indices must be strictly increasing")
    return _sha256_bytes(values.tobytes()), values


def _pipeline_types(cfg, split):
    return [str(step["type"]) for step in cfg.dataset[split].pipeline]


def _validate_raw_config(cfg, variant):
    _require(int(cfg.window_size) == TARGET_LEN, f"{variant} window_size must be {TARGET_LEN}")
    _require(int(cfg.dense_window_size) == LOGICAL_WINDOW, f"{variant} dense_window_size must be {LOGICAL_WINDOW}")
    for split in ("train", "val", "test"):
        types = _pipeline_types(cfg, split)
        _require("LoadFeats" not in types, f"{variant}/{split} must not use pre-extracted features")
        _require("mmaction.DecordDecode" in types, f"{variant}/{split} must decode raw RGB frames")
        _require(types.count("LoadFrames") == 1, f"{variant}/{split} must contain one LoadFrames transform")


def _validate_evaluators(configs):
    ground_truth_paths = {
        str(Path(cfg.evaluation.ground_truth_filename).resolve())
        for cfg in configs.values()
    }
    _require(len(ground_truth_paths) == 1, "matched configs must use one evaluator ground truth")
    ground_truth = Path(next(iter(ground_truth_paths)))
    _require(ground_truth.is_file(), f"evaluator ground truth not found: {ground_truth}")
    with tempfile.TemporaryDirectory(prefix="phystime_evaluator_gate_") as temp_dir:
        prediction = Path(temp_dir) / "empty_predictions.json"
        prediction.write_text('{"results": {}}\n', encoding="utf-8")
        for variant, cfg in configs.items():
            evaluator_cfg = dict(cfg.evaluation)
            evaluator_cfg["prediction_filename"] = str(prediction)
            evaluator = build_evaluator(evaluator_cfg)
            _require(
                evaluator.ground_truth is not None,
                f"{variant} evaluator did not load ground truth",
            )
    return str(ground_truth)


def _choose_full_length_sample(dataset, requested_index):
    if requested_index >= 0:
        _require(requested_index < len(dataset), "requested sample index is outside the training dataset")
        return requested_index
    snippet_stride = int(dataset.snippet_stride)
    required_frames = LOGICAL_WINDOW * snippet_stride
    for index, row in enumerate(dataset.data_list):
        video_info = row[1]
        if int(video_info.get("frame", 0)) >= required_frames:
            return index
    raise RuntimeError("no THUMOS training video can provide one full logical 768-position window")


def _load_real_sample(cfg, seed, requested_index):
    _seed_everything(seed)
    dataset = build_dataset(cfg.dataset.train)
    sample_index = _choose_full_length_sample(dataset, requested_index)
    _seed_everything(seed)
    sample = dataset[sample_index]
    _require("inputs" in sample and sample["inputs"].ndim == 5, "raw pipeline must return NCTHW frames")
    _require(int(sample["inputs"].shape[2]) == TARGET_LEN, "raw pipeline must decode exactly K=384 frames")
    _require(int(sample["masks"].sum().item()) == TARGET_LEN, "real gate requires 384 valid observations")
    checksum, selected = _selected_index_checksum(sample["metas"])
    return dataset, sample_index, sample, checksum, selected


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
        "gradient_l1": float(sum(parameter.grad.detach().abs().sum().item() for _, parameter in nonzero)),
        "nonzero": bool(nonzero),
        "all_finite": len(finite) == len(matched),
    }


def _optimized_parameters(optimizer):
    return [parameter for group in optimizer.param_groups for parameter in group["params"]]


def _all_finite_parameters(parameters):
    return all(bool(torch.isfinite(parameter).all().item()) for parameter in parameters)


def _optimizer_coverage(cfg, model):
    optimizer_cfg = copy.deepcopy(cfg.optimizer)
    optimizer = build_optimizer(optimizer_cfg, SimpleNamespace(module=model), logging.getLogger("phystime_gate"))
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    required = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (not name.startswith("backbone.") or "adapter" in name.lower())
    }
    duplicates = len(optimizer_ids) - len(set(optimizer_ids))
    missing = sorted(name for parameter_id, name in required.items() if parameter_id not in set(optimizer_ids))
    return optimizer, {
        "optimizer_parameter_count": len(optimizer_ids),
        "required_parameter_count": len(required),
        "duplicate_parameter_count": duplicates,
        "missing_required_parameters": missing,
        "covered": duplicates == 0 and not missing,
    }


def _run_variant(variant, cfg, sample, checkpoint, device, seed):
    cfg.model.backbone.custom.pretrain = str(checkpoint)
    _seed_everything(seed)
    model = build_detector(cfg.model).to(device).train()
    optimizer, optimizer_report = _optimizer_coverage(cfg, model)
    optimizer.zero_grad(set_to_none=True)
    batch = _move_batch(collate([sample]), device)

    backbone_lengths = []

    def capture_backbone_length(_module, _inputs, output):
        _require(torch.is_tensor(output) and output.ndim == 3, "backbone must emit [B,C,K]")
        backbone_lengths.append(int(output.shape[-1]))

    hook = model.backbone.register_forward_hook(capture_backbone_length)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    train_started = time.perf_counter()
    use_amp = bool(cfg.solver.get("amp", False))
    scaler = GradScaler(enabled=use_amp)
    optimized_parameters = _optimized_parameters(optimizer)
    step_losses = []
    all_gradients_finite = True
    finite_parameters_after_steps = True
    for step_index in range(OPTIMIZER_STEPS):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            losses = model(
                inputs=batch["inputs"],
                masks=batch["masks"],
                metas=batch["metas"],
                gt_segments=batch["gt_segments"],
                gt_labels=batch["gt_labels"],
                return_loss=True,
            )
        _require(
            "cost" in losses and torch.is_tensor(losses["cost"]),
            f"{variant} must return losses['cost']",
        )
        _require(
            _finite_tree(losses),
            f"{variant} produced non-finite training losses at optimizer step {step_index}",
        )
        scaler.scale(losses["cost"]).backward()
        scaler.unscale_(optimizer)
        step_gradients_finite = all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in optimized_parameters
        )
        all_gradients_finite = all_gradients_finite and step_gradients_finite
        _require(
            step_gradients_finite,
            f"{variant} produced non-finite gradients at optimizer step {step_index}",
        )
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in optimized_parameters if parameter.grad is not None],
            float(cfg.solver.clip_grad_norm),
            error_if_nonfinite=True,
        )
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        _require(
            scaler.get_scale() >= scale_before_step,
            f"{variant} GradScaler skipped optimizer step {step_index}",
        )
        step_parameters_finite = _all_finite_parameters(optimized_parameters)
        finite_parameters_after_steps = (
            finite_parameters_after_steps and step_parameters_finite
        )
        _require(
            step_parameters_finite,
            f"{variant} optimizer produced non-finite parameters at step {step_index}",
        )
        step_losses.append(
            {key: float(value.detach().cpu().item()) for key, value in losses.items()}
        )
    torch.cuda.synchronize(device)
    train_ms = (time.perf_counter() - train_started) * 1000.0

    named_parameters = list(model.named_parameters())
    adapter = _gradient_stats(
        named_parameters,
        lambda name, parameter: parameter.requires_grad
        and name.startswith("backbone.")
        and "adapter" in name.lower(),
    )
    detector = _gradient_stats(
        named_parameters,
        lambda name, parameter: parameter.requires_grad and not name.startswith("backbone."),
    )

    projection = classification = regression = endpoint = None
    if variant == "phystime":
        projection = _gradient_stats(
            named_parameters, lambda name, _parameter: name.startswith("projection.")
        )
        classification = _gradient_stats(
            named_parameters, lambda name, _parameter: name.startswith("rpn_head.cls_head.")
        )
        regression = _gradient_stats(
            named_parameters, lambda name, _parameter: name.startswith("rpn_head.reg_head.")
        )
        endpoint = _gradient_stats(
            named_parameters, lambda name, _parameter: name.startswith("rpn_head.endpoint_head.")
        )

    model.eval()
    torch.cuda.synchronize(device)
    infer_started = time.perf_counter()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            predictions = model.forward_test(batch["inputs"], batch["masks"], batch["metas"])
    torch.cuda.synchronize(device)
    infer_ms = (time.perf_counter() - infer_started) * 1000.0
    hook.remove()

    report = {
        "decoded_frame_count": int(batch["inputs"].shape[3]),
        "valid_observation_count": int(batch["masks"].sum().item()),
        "backbone_feature_length": int(backbone_lengths[0]),
        "inference_backbone_feature_length": int(backbone_lengths[-1]),
        "adapter_gradient_nonzero": adapter["nonzero"],
        "detector_gradient_nonzero": detector["nonzero"],
        "amp_enabled": use_amp,
        "finite_loss": _finite_tree(losses),
        "finite_predictions": _finite_tree(predictions),
        "optimizer_step_count": OPTIMIZER_STEPS,
        "all_gradients_finite": all_gradients_finite,
        "finite_parameters_after_steps": finite_parameters_after_steps,
        "optimizer_coverage": optimizer_report["covered"],
        "optimizer": optimizer_report,
        "adapter_gradient": adapter,
        "detector_gradient": detector,
        "losses": {key: float(value.detach().cpu().item()) for key, value in losses.items()},
        "optimizer_step_losses": step_losses,
        "train_forward_backward_ms": train_ms,
        "inference_ms": infer_ms,
        "peak_cuda_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0**2)),
        "config_sha256": _canonical_sha256(cfg.to_dict()),
    }
    if variant == "phystime":
        report.update(
            projection_gradient_nonzero=projection["nonzero"],
            classification_gradient_nonzero=classification["nonzero"],
            regression_gradient_nonzero=regression["nonzero"],
            endpoint_gradient_nonzero=endpoint["nonzero"],
            projection_gradient=projection,
            classification_gradient=classification,
            regression_gradient=regression,
            endpoint_gradient=endpoint,
        )

    del optimizer, model, batch
    gc.collect()
    torch.cuda.empty_cache()
    return report


def validate_gate_report(report):
    exact = {
        "schema_version": SCHEMA_VERSION,
        "gate_pass": True,
        "input_source": "raw_thumos_mp4",
        "logical_window": LOGICAL_WINDOW,
        "decoded_frame_count": TARGET_LEN,
        "backbone_feature_length": TARGET_LEN,
        "selected_index_checksum_match": True,
        "decoded_input_checksum_match": True,
        "adapter_gradient_nonzero": True,
        "projection_gradient_nonzero": True,
        "classification_gradient_nonzero": True,
        "regression_gradient_nonzero": True,
        "endpoint_gradient_nonzero": True,
        "prediction_time_unit": "seconds",
        "uses_preextracted_features": False,
        "evaluator_constructed": True,
    }
    for key, expected in exact.items():
        _require(report.get(key) == expected, f"gate report requires {key}={expected!r}")
    variants = report.get("variants", {})
    _require(set(variants) == set(GATE_CONFIGS), "gate report must contain exactly the three matched variants")
    for variant, result in variants.items():
        for key, expected in {
            "decoded_frame_count": TARGET_LEN,
            "valid_observation_count": TARGET_LEN,
            "backbone_feature_length": TARGET_LEN,
            "inference_backbone_feature_length": TARGET_LEN,
            "adapter_gradient_nonzero": True,
            "detector_gradient_nonzero": True,
            "amp_enabled": True,
            "finite_loss": True,
            "finite_predictions": True,
            "optimizer_step_count": OPTIMIZER_STEPS,
            "all_gradients_finite": True,
            "finite_parameters_after_steps": True,
            "optimizer_coverage": True,
        }.items():
            _require(result.get(key) == expected, f"{variant}.{key} must be {expected!r}")
        for gradient_key in ("adapter_gradient", "detector_gradient"):
            _require(
                result.get(gradient_key, {}).get("all_finite") is True,
                f"{variant}.{gradient_key}.all_finite must be true",
            )
    for key in (
        "projection_gradient_nonzero",
        "classification_gradient_nonzero",
        "regression_gradient_nonzero",
        "endpoint_gradient_nonzero",
    ):
        _require(variants["phystime"].get(key) is True, f"phystime.{key} must be true")
    for gradient_key in (
        "projection_gradient",
        "classification_gradient",
        "regression_gradient",
        "endpoint_gradient",
    ):
        _require(
            variants["phystime"].get(gradient_key, {}).get("all_finite") is True,
            f"phystime.{gradient_key}.all_finite must be true",
        )
    return True


def run_gate(checkpoint, device, seed=42, sample_index=-1):
    checkpoint = Path(checkpoint).resolve()
    _require(checkpoint.is_file(), f"VideoMAE-S checkpoint not found: {checkpoint}")
    _require(device.type == "cuda", "the real PhysTime-AdaTAD gate requires CUDA")
    _require(torch.cuda.is_available(), "CUDA is not available")

    configs = {name: Config.fromfile(str(path)) for name, path in GATE_CONFIGS.items()}
    for name, cfg in configs.items():
        _validate_raw_config(cfg, name)
    evaluation_ground_truth = _validate_evaluators(configs)

    samples = {}
    sample_indices = {}
    selected_checksums = {}
    input_checksums = {}
    selected_values = {}
    video_names = {}
    datasets = {}
    for name, cfg in configs.items():
        dataset, resolved_index, sample, checksum, selected = _load_real_sample(
            cfg, seed=seed, requested_index=sample_index
        )
        datasets[name] = dataset
        samples[name] = sample
        sample_indices[name] = resolved_index
        selected_checksums[name] = checksum
        selected_values[name] = selected
        input_checksums[name] = _tensor_sha256(sample["inputs"])
        video_names[name] = str(sample["metas"]["video_name"])

    _require(len(set(sample_indices.values())) == 1, "three configs resolved different training samples")
    _require(len(set(video_names.values())) == 1, "three configs decoded different videos")
    _require(len(set(selected_checksums.values())) == 1, "three configs selected different raw frame indices")
    _require(
        len(set(input_checksums.values())) == 1,
        f"three configs produced different decoded/augmented tensors: {input_checksums}",
    )
    phystime_meta = samples["phystime"]["metas"]
    _require(
        np.array_equal(
            np.asarray(phystime_meta.get("phystime_selected_raw_frame_indices", []), dtype=np.int64),
            selected_values["phystime"],
        ),
        "PhysTime geometry raw-frame audit does not match the shared selected indices",
    )

    variant_reports = {}
    for name in GATE_CONFIGS:
        print(f"[PhysTime-AdaTAD gate] running {name}", flush=True)
        _seed_everything(seed)
        variant_reports[name] = _run_variant(
            name, configs[name], samples[name], checkpoint, device, seed
        )

    phystime = variant_reports["phystime"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate_pass": True,
        "input_source": "raw_thumos_mp4",
        "logical_window": LOGICAL_WINDOW,
        "decoded_frame_count": TARGET_LEN,
        "backbone_feature_length": TARGET_LEN,
        "selected_index_checksum_match": True,
        "decoded_input_checksum_match": True,
        "selected_index_sha256": next(iter(selected_checksums.values())),
        "decoded_input_sha256": next(iter(input_checksums.values())),
        "selected_index_first": int(selected_values["phystime"][0]),
        "selected_index_last": int(selected_values["phystime"][-1]),
        "sample_index": next(iter(sample_indices.values())),
        "sample_video": next(iter(video_names.values())),
        "adapter_gradient_nonzero": all(
            result["adapter_gradient_nonzero"] for result in variant_reports.values()
        ),
        "projection_gradient_nonzero": phystime["projection_gradient_nonzero"],
        "classification_gradient_nonzero": phystime["classification_gradient_nonzero"],
        "regression_gradient_nonzero": phystime["regression_gradient_nonzero"],
        "endpoint_gradient_nonzero": phystime["endpoint_gradient_nonzero"],
        "prediction_time_unit": str(phystime_meta.get("prediction_time_unit")),
        "uses_preextracted_features": False,
        "evaluator_constructed": True,
        "evaluation_ground_truth_filename": evaluation_ground_truth,
        "seed": int(seed),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "cuda_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "variants": variant_reports,
    }
    validate_gate_report(report)
    del datasets, samples
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Run the matched raw-video K384 PhysTime-AdaTAD gate")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-index", type=int, default=-1)
    return parser.parse_args()


def _write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    try:
        report = run_gate(
            checkpoint=args.checkpoint,
            device=torch.device(args.device),
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
        _write_report(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        raise SystemExit(1) from error
    _write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
