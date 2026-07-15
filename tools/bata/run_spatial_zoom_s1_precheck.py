from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.validate_spatial_zoom_s1 import (  # noqa: E402
    CONFIG_PATHS,
    validate_config_matrix,
)
from tools.bata.spatial_zoom_s1_contract import (  # noqa: E402
    S1_PRETRAINED_CHECKPOINT_FILENAME,
    S1_PRETRAINED_CHECKPOINT_SHA256,
    canonical_sha256,
    sha256_file,
)

S1_PRECHECK_SCHEMA = "spatial_zoom_s1_precheck_v5"


def _register_opentad_runtime_modules() -> None:
    """Mirror the registry imports performed by the official train entrypoint."""

    importlib.import_module("opentad.datasets")
    importlib.import_module("opentad.models.backbones")


def build_precheck_spec(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    cfg = Config.fromfile(str(config_path))
    contract = cfg.spatial_zoom_s1_contract
    backbone = cfg.model.backbone.backbone
    resolution = int(contract.runtime_resolution)
    patch_size = int(backbone.patch_size)
    tubelet_size = int(backbone.get("tubelet_size", 2))
    num_frames = int(backbone.num_frames)
    embed_dims = int(backbone.embed_dims)
    native_grid_size = int(backbone.img_size) // patch_size
    if resolution % patch_size:
        raise ValueError("S1 resolution must be divisible by the VideoMAE patch size")
    runtime_grid_size = resolution // patch_size
    if num_frames % tubelet_size:
        raise ValueError("VideoMAE clip length must be divisible by tubelet_size")
    interpolation_expected = runtime_grid_size != native_grid_size
    return {
        "config": str(config_path),
        "resolution": resolution,
        "patch_size": patch_size,
        "tubelet_size": tubelet_size,
        "clip_frames": num_frames,
        "native_position_grid": [native_grid_size, native_grid_size],
        "runtime_grid": [runtime_grid_size, runtime_grid_size],
        "position_interpolation_expected": interpolation_expected,
        "expected_interpolation_target_calls": (
            [[runtime_grid_size, runtime_grid_size]] if interpolation_expected else []
        ),
        "clip_token_count": (num_frames // tubelet_size) * runtime_grid_size**2,
        "clip_output_shape": [
            1,
            embed_dims,
            num_frames // tubelet_size,
            runtime_grid_size,
            runtime_grid_size,
        ],
        "full_detector_feature_shape": [
            1,
            embed_dims,
            int(contract.detector_time_grid),
        ],
        "temporal_interpolation": str(contract.temporal_interpolation),
        "temporal_interpolation_input_points": int(
            contract.temporal_interpolation_input_points
        ),
        "full_window_input_shape": [
            1,
            1,
            3,
            int(contract.temporal_window),
            resolution,
            resolution,
        ],
    }


def _validate_interpolation_calls(
    spec: dict[str, Any], calls: list[list[int] | None]
) -> None:
    expected = spec["expected_interpolation_target_calls"]
    if calls != expected:
        raise AssertionError(
            "S1 positional interpolation calls do not exactly match the audited branch: "
            f"observed={calls}, expected={expected}"
        )


def _audit_loaded_pretrained_state(
    backbone_wrapper: Any,
    checkpoint_path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Prove that every non-adapter VideoMAE parameter came from the checkpoint."""

    import torch

    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != str(expected_sha256).lower():
        raise ValueError("formal S1 precheck pretrained checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(
            "formal S1 pretrained checkpoint must contain a state dictionary"
        )
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError("formal S1 pretrained checkpoint state is not a mapping")
    normalized = {
        (str(key)[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in state.items()
    }
    wrapped_model = backbone_wrapper.model
    model_state = wrapped_model.state_dict()
    core_names = sorted(
        name
        for name, _parameter in wrapped_model.named_parameters()
        if name.startswith("backbone.")
        and "adapter" not in name
        and not name.startswith("backbone.chronotransport.")
    )
    if not core_names:
        raise ValueError(
            "formal S1 precheck found no auditable VideoMAE core parameters"
        )
    missing = [name for name in core_names if name not in normalized]
    shape_mismatch = [
        name
        for name in core_names
        if name in normalized
        and tuple(normalized[name].shape) != tuple(model_state[name].shape)
    ]
    if missing or shape_mismatch:
        raise ValueError(
            "formal S1 pretrained checkpoint does not cover the complete VideoMAE core: "
            f"missing={missing[:8]}, shape_mismatch={shape_mismatch[:8]}"
        )
    value_mismatch = [
        name
        for name in core_names
        if not torch.equal(
            model_state[name].detach().cpu(), normalized[name].detach().cpu()
        )
    ]
    if value_mismatch:
        raise ValueError(
            f"formal S1 VideoMAE core tensors were not loaded: {value_mismatch[:8]}"
        )
    required_families = (
        "backbone.patch_embed.",
        "backbone.blocks.0.",
        "backbone.blocks.11.",
        "backbone.fc_norm.",
    )
    absent_families = [
        prefix
        for prefix in required_families
        if not any(name.startswith(prefix) for name in core_names)
    ]
    if absent_families:
        raise ValueError(
            f"formal S1 VideoMAE audit lacks critical parameter families: {absent_families}"
        )
    total_numel = sum(int(model_state[name].numel()) for name in core_names)
    return {
        "verified": True,
        "checkpoint_sha256": actual_sha256,
        "model_core_parameter_count": len(core_names),
        "loaded_core_parameter_count": len(core_names),
        "model_core_parameter_numel": total_numel,
        "loaded_core_parameter_numel": total_numel,
        "core_parameter_numel_coverage": 1.0,
        "core_keyset_sha256": canonical_sha256(
            [
                (name, list(model_state[name].shape), str(model_state[name].dtype))
                for name in core_names
            ]
        ),
        "missing_core_keys": [],
        "shape_mismatch_core_keys": [],
        "value_mismatch_core_keys": [],
        "required_parameter_families": list(required_families),
    }


def _validate_pretrained_load_audit(audit: Any, *, expected_sha256: str) -> None:
    if not isinstance(audit, dict) or audit.get("verified") is not True:
        raise ValueError("S1 precheck has no verified pretrained-load audit")
    if (
        audit.get("checkpoint_sha256") != expected_sha256
        or audit.get("loaded_core_parameter_count")
        != audit.get("model_core_parameter_count")
        or audit.get("loaded_core_parameter_numel")
        != audit.get("model_core_parameter_numel")
        or audit.get("core_parameter_numel_coverage") != 1.0
        or audit.get("missing_core_keys")
        or audit.get("shape_mismatch_core_keys")
        or audit.get("value_mismatch_core_keys")
    ):
        raise ValueError("S1 precheck pretrained-load audit is incomplete")
    keyset_hash = str(audit.get("core_keyset_sha256", "")).lower()
    if len(keyset_hash) != 64 or any(
        character not in "0123456789abcdef" for character in keyset_hash
    ):
        raise ValueError("S1 precheck pretrained core keyset hash is invalid")


def _memory_snapshot(torch: Any, device: Any) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"peak_allocated_mb": None, "peak_reserved_mb": None}
    return {
        "peak_allocated_mb": float(
            torch.cuda.max_memory_allocated(device) / (1024**2)
        ),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024**2)),
    }


def _run_clip(
    config_path: str | Path, *, device_text: str, amp: bool
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from mmaction.registry import MODELS

    _register_opentad_runtime_modules()

    cfg = Config.fromfile(str(config_path))
    spec = build_precheck_spec(config_path)
    backbone_cfg = copy.deepcopy(cfg.model.backbone.backbone)
    model = MODELS.build(backbone_cfg).to(device_text).eval()
    device = torch.device(device_text)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    resolution = int(spec["resolution"])
    input_tensor = torch.randn(
        1,
        3,
        int(spec["clip_frames"]),
        resolution,
        resolution,
        device=device,
    )
    interpolation_calls = []
    original_interpolate = functional.interpolate

    def observed_interpolate(value, *args, **kwargs):
        size = kwargs.get("size", args[0] if args else None)
        if value.ndim == 4 and list(value.shape[-2:]) == spec["native_position_grid"]:
            interpolation_calls.append(None if size is None else list(size))
        return original_interpolate(value, *args, **kwargs)

    started = time.perf_counter()
    functional.interpolate = observed_interpolate
    try:
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(amp and device.type == "cuda"),
        ):
            output = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        functional.interpolate = original_interpolate
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    output_shape = list(output.shape)
    if output_shape != spec["clip_output_shape"]:
        raise AssertionError(
            f"S1 clip output shape {output_shape} != expected {spec['clip_output_shape']}"
        )
    _validate_interpolation_calls(spec, interpolation_calls)
    expected_call = bool(interpolation_calls)
    result = {
        "mode": "real_videomae_clip",
        "output_shape": output_shape,
        "position_interpolation_observed": expected_call,
        "interpolation_target_calls": interpolation_calls,
        "latency_ms_diagnostic_only": elapsed_ms,
        "random_initialization": True,
        "paper_cost_claim_allowed": False,
        **_memory_snapshot(torch, device),
    }
    del output, input_tensor, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _run_full_detector(
    config_path: str | Path,
    *,
    device_text: str,
    amp: bool,
    expected_pretrained_sha256: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    _register_opentad_runtime_modules()
    from opentad.models import build_detector

    cfg = Config.fromfile(str(config_path))
    spec = build_precheck_spec(config_path)
    model_cfg = copy.deepcopy(cfg.model)
    pretrain_path = ROOT / str(model_cfg.backbone.custom.pretrain)
    if pretrain_path.name != S1_PRETRAINED_CHECKPOINT_FILENAME:
        raise ValueError(
            "formal S1 config does not use the frozen VideoMAE checkpoint filename"
        )
    if not pretrain_path.is_file():
        raise FileNotFoundError(
            f"formal S1 full precheck requires the real pretrained checkpoint: {pretrain_path}"
        )
    model = build_detector(model_cfg)
    pretrained_load_audit = _audit_loaded_pretrained_state(
        model.backbone,
        pretrain_path,
        expected_sha256=expected_pretrained_sha256,
    )
    vision_backbone = model.backbone.model.backbone
    if list(vision_backbone.grid_size) != spec["native_position_grid"]:
        raise AssertionError("S1 VideoMAE native positional grid changed")
    model = model.to(device_text).eval()
    device = torch.device(device_text)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    captured: dict[str, list[int]] = {}

    def capture_backbone(_module, _inputs, output):
        captured["backbone_output_shape"] = list(output.shape)

    def capture_projection(_module, inputs):
        captured["projection_input_shape"] = list(inputs[0].shape)

    handles = [
        model.backbone.register_forward_hook(capture_backbone),
        model.projection.register_forward_pre_hook(capture_projection),
    ]
    interpolation_calls = []
    original_interpolate = functional.interpolate

    def observed_interpolate(value, *args, **kwargs):
        size = kwargs.get("size", args[0] if args else None)
        if value.ndim == 4 and list(value.shape[-2:]) == spec["native_position_grid"]:
            interpolation_calls.append(None if size is None else list(size))
        return original_interpolate(value, *args, **kwargs)

    resolution = int(spec["resolution"])
    inputs = torch.randn(
        *spec["full_window_input_shape"], device=device, dtype=torch.float32
    )
    masks = torch.ones((1, 768), device=device, dtype=torch.bool)
    metas = [
        {
            "video_name": f"s1_dense{resolution}_precheck",
            "fps": 30.0,
            "duration": 25.6,
            "snippet_stride": 1,
            "window_start_frame": 0,
            "window_size": 768,
            "offset_frames": 0,
        }
    ]
    started = time.perf_counter()
    functional.interpolate = observed_interpolate
    try:
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(amp and device.type == "cuda"),
        ):
            predictions = model.forward_test(inputs, masks, metas, cfg.inference)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        functional.interpolate = original_interpolate
        for handle in handles:
            handle.remove()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if captured.get("projection_input_shape") != spec["full_detector_feature_shape"]:
        raise AssertionError(
            "S1 full detector did not preserve the official 768-point detector feature grid"
        )
    if not isinstance(predictions, (tuple, list)) or len(predictions) != 2:
        raise AssertionError(
            "S1 full detector forward_test returned an unexpected structure"
        )
    prediction_container_length = len(predictions)
    _validate_interpolation_calls(spec, interpolation_calls)
    expected_call = bool(interpolation_calls)
    del predictions

    model.train()
    model.zero_grad(set_to_none=True)
    gt_segments = [
        torch.tensor([[128.0, 320.0]], device=device, dtype=torch.float32)
    ]
    gt_labels = [torch.tensor([0], device=device, dtype=torch.long)]
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=bool(amp and device.type == "cuda"),
    ):
        losses = model.forward_train(
            inputs,
            masks,
            metas,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
        )
        train_cost = losses["cost"]
    if not bool(torch.isfinite(train_cost).item()):
        raise FloatingPointError("S1 full precheck produced a non-finite train cost")
    train_cost.backward()
    required_gradient_components = ("backbone", "projection", "rpn_head")
    gradient_coverage: dict[str, dict[str, Any]] = {}
    missing_gradient_parameters = []
    finite_gradient_tensors = 0
    nonzero_gradient_tensors = 0
    trainable_parameter_tensors = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_parameter_tensors += 1
        component = name.split(".", 1)[0]
        row = gradient_coverage.setdefault(
            component,
            {
                "trainable_parameter_tensors": 0,
                "gradient_tensors": 0,
                "nonzero_gradient_tensors": 0,
                "all_present_gradients_finite": True,
            },
        )
        row["trainable_parameter_tensors"] += 1
        if parameter.grad is None:
            missing_gradient_parameters.append(name)
            continue
        row["gradient_tensors"] += 1
        if not bool(torch.isfinite(parameter.grad).all().item()):
            row["all_present_gradients_finite"] = False
            raise FloatingPointError(
                f"S1 full precheck produced a non-finite gradient for {name}"
            )
        finite_gradient_tensors += 1
        if bool(torch.count_nonzero(parameter.grad).item()):
            row["nonzero_gradient_tensors"] += 1
            nonzero_gradient_tensors += 1
    if missing_gradient_parameters:
        raise RuntimeError(
            "S1 full precheck left trainable parameters outside the detector "
            f"backward graph: count={len(missing_gradient_parameters)}, "
            f"examples={missing_gradient_parameters[:8]}"
        )
    for component in required_gradient_components:
        row = gradient_coverage.get(component)
        if not row or int(row["trainable_parameter_tensors"]) <= 0:
            raise RuntimeError(
                f"S1 full precheck found no trainable {component} parameters"
            )
        if row["gradient_tensors"] != row["trainable_parameter_tensors"]:
            raise RuntimeError(
                f"S1 full precheck has incomplete {component} gradient coverage"
            )
        if int(row["nonzero_gradient_tensors"]) <= 0:
            raise RuntimeError(
                f"S1 full precheck produced no nonzero {component} gradient"
            )
    if finite_gradient_tensors != trainable_parameter_tensors:
        raise RuntimeError("S1 full precheck gradient coverage is incomplete")
    result = {
        "mode": "real_full_detector_window",
        **captured,
        "prediction_container_length": prediction_container_length,
        "position_interpolation_observed": expected_call,
        "interpolation_target_calls": interpolation_calls,
        "pretrained_checkpoint": str(pretrain_path.resolve()),
        "pretrained_checkpoint_sha256": sha256_file(pretrain_path),
        "pretrained_checkpoint_loaded": True,
        "pretrained_load_audit": pretrained_load_audit,
        "latency_ms_diagnostic_only": elapsed_ms,
        "random_initialization": False,
        "paper_cost_claim_allowed": False,
        "strict_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "strict_deterministic_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "deterministic_backward_passed": True,
        "train_cost": float(train_cost.detach().float().cpu().item()),
        "trainable_parameter_tensors": int(trainable_parameter_tensors),
        "finite_gradient_tensors": int(finite_gradient_tensors),
        "nonzero_gradient_tensors": int(nonzero_gradient_tensors),
        "gradient_coverage": gradient_coverage,
        **_memory_snapshot(torch, device),
    }
    del losses, train_cost, gt_segments, gt_labels, inputs, masks, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_precheck(
    config_paths: list[str | Path],
    *,
    mode: str,
    device: str,
    amp: bool,
    expected_pretrained_sha256: str | None = None,
) -> dict[str, Any]:
    matrix = validate_config_matrix()
    from tools.bata.spatial_zoom_s1_training import current_git_commit

    code_commit = current_git_commit()
    slurm_allocation = None
    if mode == "full":
        from tools.bata.spatial_zoom_s1_training import (
            require_clean_git_checkout,
            require_slurm_single_gpu_allocation,
        )

        expected_paths = {
            (ROOT / CONFIG_PATHS[value]).resolve() for value in (160, 224, 256)
        }
        actual_paths = {Path(path).resolve() for path in config_paths}
        if actual_paths != expected_paths or len(config_paths) != 3:
            raise ValueError(
                "formal S1 full precheck requires the complete audited 3-config matrix"
            )
        if str(device) != "cuda:0" or not amp:
            raise ValueError(
                "formal S1 full precheck requires cuda:0 inside a single-GPU Slurm allocation with AMP"
            )
        expected_pretrained_sha256 = str(expected_pretrained_sha256 or "").lower()
        if expected_pretrained_sha256 != S1_PRETRAINED_CHECKPOINT_SHA256:
            raise ValueError(
                "formal S1 full precheck expected pretrained SHA-256 must equal the frozen contract"
            )
        physical_gpu_id = require_slurm_single_gpu_allocation()
        slurm_allocation = {
            "job_id": os.environ["SLURM_JOB_ID"],
            "physical_gpu_id": physical_gpu_id,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
            "logical_device": "cuda:0",
        }
        require_clean_git_checkout(expected_commit=code_commit)
        from opentad.utils import set_seed

        set_seed(3407001, deterministic_warn_only=False)
    rows = []
    for path in config_paths:
        spec = build_precheck_spec(path)
        row: dict[str, Any] = {"spec": spec, "runtime": None}
        if mode == "clip":
            row["runtime"] = _run_clip(path, device_text=device, amp=amp)
        elif mode == "full":
            row["runtime"] = _run_full_detector(
                path,
                device_text=device,
                amp=amp,
                expected_pretrained_sha256=expected_pretrained_sha256,
            )
        elif mode != "static":
            raise ValueError(f"unknown S1 precheck mode {mode!r}")
        rows.append(row)
    report = {
        "schema_version": S1_PRECHECK_SCHEMA,
        "status": "PASS",
        "mode": mode,
        "device": device,
        "amp": bool(amp),
        "config_matrix_protocol_fingerprint": matrix["protocol_fingerprint"],
        "code_commit": code_commit,
        "expected_pretrained_checkpoint_sha256": expected_pretrained_sha256
        if mode == "full"
        else None,
        "slurm_allocation": slurm_allocation,
        "rows": rows,
        "formal_training_ready": mode == "full" and len(rows) == 3,
        "paper_result": False,
        "strict_deterministic_algorithms": bool(
            mode == "full"
            and __import__("torch").are_deterministic_algorithms_enabled()
        ),
        "strict_deterministic_warn_only": bool(
            mode == "full"
            and __import__("torch").is_deterministic_algorithms_warn_only_enabled()
        ),
    }
    report["precheck_sha256"] = canonical_sha256(report)
    return report


def validate_precheck_certificate(
    certificate: dict[str, Any], *, require_full: bool = True
) -> dict[str, Any]:
    from tools.bata.spatial_zoom_s1_training import current_git_commit

    checked = json.loads(json.dumps(dict(certificate)))
    certificate_hash = checked.pop("precheck_sha256", None)
    if not certificate_hash or canonical_sha256(checked) != certificate_hash:
        raise ValueError("S1 precheck certificate self-hash mismatch")
    checked["precheck_sha256"] = certificate_hash
    matrix = validate_config_matrix()
    expected = {
        "schema_version": S1_PRECHECK_SCHEMA,
        "status": "PASS",
        "config_matrix_protocol_fingerprint": matrix["protocol_fingerprint"],
        "code_commit": current_git_commit(),
        "paper_result": False,
    }
    for key, value in expected.items():
        if checked.get(key) != value:
            raise ValueError(f"S1 precheck certificate {key} mismatch")
    if (
        require_full
        and checked.get("expected_pretrained_checkpoint_sha256")
        != S1_PRETRAINED_CHECKPOINT_SHA256
    ):
        raise ValueError(
            "S1 precheck certificate does not use the frozen pretrained checkpoint identity"
        )
    rows = checked.get("rows")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("S1 precheck certificate requires all three resolutions")
    expected_specs = {
        resolution: build_precheck_spec(ROOT / CONFIG_PATHS[resolution])
        for resolution in (160, 224, 256)
    }
    actual_resolutions = []
    for row in rows:
        spec = row.get("spec")
        if not isinstance(spec, dict):
            raise ValueError("S1 precheck row has no shape specification")
        resolution = int(spec["resolution"])
        actual_resolutions.append(resolution)
        if spec != expected_specs.get(resolution):
            raise ValueError("S1 precheck shape specification changed")
        runtime = row.get("runtime")
        if require_full:
            if (
                not isinstance(runtime, dict)
                or runtime.get("mode") != "real_full_detector_window"
            ):
                raise ValueError(
                    "formal S1 precheck requires real full-detector evidence"
                )
            if (
                runtime.get("projection_input_shape")
                != spec["full_detector_feature_shape"]
            ):
                raise ValueError("S1 precheck detector feature grid mismatch")
            if bool(runtime.get("position_interpolation_observed")) != bool(
                spec["position_interpolation_expected"]
            ):
                raise ValueError(
                    "S1 precheck positional interpolation evidence mismatch"
                )
            if (
                runtime.get("interpolation_target_calls")
                != spec["expected_interpolation_target_calls"]
            ):
                raise ValueError(
                    "S1 precheck positional interpolation call sequence mismatch"
                )
            if runtime.get("strict_deterministic_algorithms") is not True:
                raise ValueError(
                    "formal S1 precheck did not enable strict deterministic algorithms"
                )
            if runtime.get("strict_deterministic_warn_only") is not False:
                raise ValueError(
                    "formal S1 precheck left deterministic algorithms in warn-only mode"
                )
            if runtime.get("deterministic_backward_passed") is not True:
                raise ValueError(
                    "formal S1 precheck did not pass a real detector backward"
                )
            trainable_parameter_tensors = int(
                runtime.get("trainable_parameter_tensors", 0)
            )
            if (
                trainable_parameter_tensors <= 0
                or int(runtime.get("finite_gradient_tensors", 0))
                != trainable_parameter_tensors
                or int(runtime.get("nonzero_gradient_tensors", 0)) <= 0
            ):
                raise ValueError(
                    "formal S1 precheck has incomplete detector-gradient evidence"
                )
            coverage = runtime.get("gradient_coverage")
            if not isinstance(coverage, dict):
                raise ValueError("formal S1 precheck has no gradient coverage map")
            for component in ("backbone", "projection", "rpn_head"):
                component_row = coverage.get(component)
                if (
                    not isinstance(component_row, dict)
                    or int(component_row.get("trainable_parameter_tensors", 0)) <= 0
                    or component_row.get("gradient_tensors")
                    != component_row.get("trainable_parameter_tensors")
                    or int(component_row.get("nonzero_gradient_tensors", 0)) <= 0
                    or component_row.get("all_present_gradients_finite") is not True
                ):
                    raise ValueError(
                        f"formal S1 precheck has invalid {component} gradient coverage"
                    )
            if not runtime.get("pretrained_checkpoint_loaded") or runtime.get(
                "random_initialization"
            ):
                raise ValueError("formal S1 precheck did not use pretrained VideoMAE")
            pretrain_path = Path(runtime.get("pretrained_checkpoint", ""))
            if not pretrain_path.is_file() or sha256_file(pretrain_path) != runtime.get(
                "pretrained_checkpoint_sha256"
            ):
                raise ValueError(
                    "S1 precheck pretrained checkpoint provenance mismatch"
                )
            expected_pretrained = checked.get("expected_pretrained_checkpoint_sha256")
            if runtime.get("pretrained_checkpoint_sha256") != expected_pretrained:
                raise ValueError(
                    "S1 precheck pretrained checkpoint differs from the preregistered SHA-256"
                )
            _validate_pretrained_load_audit(
                runtime.get("pretrained_load_audit"),
                expected_sha256=expected_pretrained,
            )
            if (
                runtime.get("peak_allocated_mb") is None
                or runtime.get("peak_reserved_mb") is None
            ):
                raise ValueError("formal S1 precheck has no CUDA memory evidence")
    if sorted(actual_resolutions) != [160, 224, 256]:
        raise ValueError("S1 precheck certificate resolution matrix is incomplete")
    if require_full and (
        checked.get("mode") != "full"
        or checked.get("device") != "cuda:0"
        or checked.get("amp") is not True
        or checked.get("formal_training_ready") is not True
    ):
        raise ValueError("S1 full precheck execution contract mismatch")
    if require_full:
        if checked.get("strict_deterministic_algorithms") is not True:
            raise ValueError(
                "formal S1 precheck certificate is not strict deterministic"
            )
        if checked.get("strict_deterministic_warn_only") is not False:
            raise ValueError(
                "formal S1 precheck certificate is deterministic warn-only"
            )
        allocation = checked.get("slurm_allocation")
        if (
            not isinstance(allocation, dict)
            or not allocation.get("job_id")
            or not allocation.get("physical_gpu_id")
            or "," in str(allocation.get("physical_gpu_id"))
            or not allocation.get("cuda_visible_devices")
            or "," in str(allocation.get("cuda_visible_devices"))
            or str(allocation.get("gpus_on_node")) != "1"
            or allocation.get("logical_device") != "cuda:0"
        ):
            raise ValueError("S1 full precheck Slurm allocation evidence is invalid")
    return checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run S1 spatial-shape and positional-interpolation prechecks"
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="repeat for selected configs; defaults to the complete 160/224/256 matrix",
    )
    parser.add_argument("--mode", choices=("static", "clip", "full"), default="clip")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--expected-pretrained-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config_paths = args.config or [
        str(ROOT / CONFIG_PATHS[value]) for value in (160, 224, 256)
    ]
    if args.output and args.output.exists():
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": "FileExistsError",
                    "error": "refusing to overwrite an S1 precheck certificate",
                },
                indent=2,
            )
        )
        return 1
    try:
        summary = run_precheck(
            config_paths,
            mode=args.mode,
            device=args.device,
            amp=args.amp,
            expected_pretrained_sha256=args.expected_pretrained_sha256,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
            )
        )
        return 1
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = [
    "_register_opentad_runtime_modules",
    "_validate_interpolation_calls",
    "_validate_pretrained_load_audit",
    "build_precheck_spec",
    "run_precheck",
    "validate_precheck_certificate",
]
