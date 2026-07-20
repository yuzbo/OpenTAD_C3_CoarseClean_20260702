from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.continuous_roi_s2_contract import (  # noqa: E402
    canonical_sha256,
    finalize_self_hash,
    load_protocol,
    validate_protocol,
)
from tools.bata.native_crop_s1_contract import (  # noqa: E402
    NATIVE_CROP_PRETRAINED_FILENAME,
    NATIVE_CROP_PRETRAINED_SHA256,
)
from tools.bata.run_native_crop_s1_precheck import (  # noqa: E402
    audit_loaded_pretrained_state,
)
from tools.bata.spatial_zoom_s1_contract import sha256_file  # noqa: E402
from tools.bata.validate_continuous_roi_s2_implementation import (  # noqa: E402
    CONFIGS,
    validate_implementation,
)


GATE_SCHEMA = "continuous_roi_s2_full_model_one_step_cuda_gate_v1"
AUDITED_SOURCE_PATHS = (
    "configs/_base_/models/actionformer.py",
    "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py",
    "configs/adatad/thumos/continuous_roi_s2_d160_videomae_s_768x1_adapter.py",
    "configs/adatad/thumos/continuous_roi_s2_g96_videomae_s_768x1_adapter.py",
    "configs/adatad/thumos/continuous_roi_s2_u128_videomae_s_768x1_adapter.py",
    "docs/methods/continuous_roi_s2_v2_1_contract.md",
    "docs/methods/continuous_roi_s2_v2_1_protocol.json",
    "opentad/cores/optimizer.py",
    "opentad/cores/train_engine.py",
    "opentad/datasets/transforms/native_crop.py",
    "opentad/models/backbones/continuous_roi_geometry.py",
    "opentad/models/backbones/continuous_roi_sampler.py",
    "opentad/models/backbones/continuous_roi_wrapper.py",
    "opentad/models/backbones/native_crop_wrapper.py",
    "opentad/models/backbones/vit_adapter.py",
    "opentad/models/detectors/actionformer.py",
    "scripts/run_continuous_roi_s2_cuda_gate_slurm.sh",
    "tests/test_continuous_roi_geometry_sampler.py",
    "tests/test_continuous_roi_representation.py",
    "tests/test_continuous_roi_s2_implementation_static.py",
    "tests/test_continuous_roi_s2_protocol.py",
    "tests/test_continuous_roi_s2_one_step_gate.py",
    "tools/bata/continuous_roi_s2_contract.py",
    "tools/bata/run_continuous_roi_s2_one_step_gate.py",
    "tools/bata/validate_continuous_roi_s2_implementation.py",
)


class _Logger:
    def __init__(self):
        self.messages: list[str] = []

    def info(self, message, *args) -> None:
        if args:
            message = message % args
        self.messages.append(str(message))


def _run_git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def audit_code_provenance(expected_commit: str) -> dict:
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("expected_commit must be one full Git commit")
    actual_commit = _run_git("rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            f"checkout {actual_commit} differs from expected {expected_commit}"
        )
    worktree_status = _run_git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if worktree_status:
        raise ValueError("formal Gate requires a completely clean worktree")

    source_hashes = {}
    for relative_path in AUDITED_SOURCE_PATHS:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        _run_git("ls-files", "--error-unmatch", "--", relative_path)
        working_bytes = path.read_bytes()
        committed = subprocess.run(
            ["git", "show", f"{expected_commit}:{relative_path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if working_bytes != committed:
            raise ValueError(f"audited source differs from HEAD: {relative_path}")
        source_hashes[relative_path] = hashlib.sha256(working_bytes).hexdigest()
    return {
        "expected_commit": expected_commit,
        "git_commit": actual_commit,
        "complete_worktree_clean": True,
        "audited_source_sha256": source_hashes,
    }


def _parameter_component(name: str) -> str:
    if name.startswith("backbone.model.backbone") and ".adapter." in f".{name}.":
        return "shared_adapter"
    if name.startswith("backbone.fusion."):
        return "fusion"
    if name.startswith("backbone.global_aux_head."):
        return "global_aux_head"
    if name.startswith("backbone.local_aux_head."):
        return "local_aux_head"
    if name.startswith("projection."):
        return "projection"
    if name.startswith("rpn_head."):
        return "rpn_head"
    if name.startswith("neck."):
        return "neck"
    return name.split(".", 1)[0]


def audit_optimizer_coverage(model, optimizer) -> dict:
    required = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    occurrences = Counter(
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    missing = sorted(required[identifier] for identifier in required.keys() - occurrences)
    duplicates = sorted(
        required.get(identifier, f"unexpected:{identifier}")
        for identifier, count in occurrences.items()
        if count != 1
    )
    unexpected = sorted(
        identifier for identifier in occurrences if identifier not in required
    )
    if missing or duplicates or unexpected:
        raise RuntimeError(
            "optimizer coverage is not exact: "
            f"missing={missing} duplicates={duplicates} unexpected={unexpected}"
        )
    group_sizes = [len(group["params"]) for group in optimizer.param_groups]
    if any(size <= 0 for size in group_sizes):
        raise RuntimeError("optimizer contains an empty parameter group")
    component_tensors = Counter(
        _parameter_component(name) for name in required.values()
    )
    for component in (
        "shared_adapter",
        "fusion",
        "global_aux_head",
        "local_aux_head",
        "projection",
        "rpn_head",
    ):
        if component_tensors[component] <= 0:
            raise RuntimeError(f"optimizer has no trainable {component} parameters")
    return {
        "requires_grad_parameter_tensors": len(required),
        "optimizer_parameter_tensors": sum(group_sizes),
        "optimizer_group_sizes": group_sizes,
        "trainable_tensors_by_component": dict(sorted(component_tensors.items())),
        "every_requires_grad_parameter_exactly_once": True,
        "frozen_parameters_excluded": True,
    }


def audit_frozen_shared_core(model) -> dict:
    trainable_adapters = []
    leaked_core = []
    frozen_core = []
    for name, parameter in model.backbone.model.backbone.named_parameters():
        if ".adapter." in f".{name}.":
            if not parameter.requires_grad:
                raise RuntimeError(f"official adapter was unexpectedly frozen: {name}")
            trainable_adapters.append(name)
        elif parameter.requires_grad:
            leaked_core.append(name)
        else:
            frozen_core.append(name)
    if leaked_core:
        raise RuntimeError(f"pretrained VideoMAE core is trainable: {leaked_core}")
    if not trainable_adapters:
        raise RuntimeError("no trainable official VideoMAE adapters were found")
    for forbidden in ("fc_norm.weight", "fc_norm.bias", "norm.weight", "norm.bias"):
        matches = [
            parameter.requires_grad
            for name, parameter in model.backbone.model.backbone.named_parameters()
            if name == forbidden
        ]
        if matches and any(matches):
            raise RuntimeError(f"unused/core norm parameter leaked: {forbidden}")
    return {
        "trainable_adapter_parameter_tensors": len(trainable_adapters),
        "frozen_core_parameter_tensors": len(frozen_core),
        "non_adapter_core_trainable_tensors": 0,
        "unused_fc_norm_frozen": True,
    }


def _nonzero_finite_gradient_audit(
    names: list[str],
    gradients,
    *,
    required_components: tuple[str, ...],
) -> dict:
    component_total = Counter()
    component_nonzero = Counter()
    missing = []
    nonfinite = []
    for name, gradient in zip(names, gradients):
        component = _parameter_component(name)
        component_total[component] += 1
        if gradient is None:
            missing.append(name)
            continue
        if not bool(__import__("torch").isfinite(gradient).all().item()):
            nonfinite.append(name)
        elif bool(__import__("torch").count_nonzero(gradient).item()):
            component_nonzero[component] += 1
    if nonfinite:
        raise FloatingPointError(f"non-finite gradients: {nonfinite}")
    for component in required_components:
        if component_total[component] <= 0:
            raise RuntimeError(f"gradient audit has no {component} targets")
        if component_nonzero[component] <= 0:
            raise RuntimeError(f"no nonzero gradient reached {component}")
    return {
        "target_tensors_by_component": dict(sorted(component_total.items())),
        "nonzero_gradient_tensors_by_component": dict(
            sorted(component_nonzero.items())
        ),
        "missing_target_gradients": sorted(missing),
        "all_present_gradients_finite": True,
    }


def audit_sampler_geometry_gradient(device) -> dict:
    import torch

    source_y, source_x = torch.meshgrid(
        torch.linspace(0.0, 1.0, 8, device=device),
        torch.linspace(0.0, 1.0, 8, device=device),
        indexing="ij",
    )
    frame = (source_x + 0.5 * source_y).expand(3, 8, 8)
    source = frame.reshape(1, 3, 1, 8, 8)
    boxes = torch.tensor(
        [[[0.50, 0.50, 0.50, 0.50]]],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    from opentad.models.backbones.continuous_roi_sampler import (
        sample_continuous_roi,
        sample_continuous_roi_runtime,
    )

    sampled = sample_continuous_roi(
        source,
        boxes,
        output_height=5,
        output_width=5,
        frames_per_clip=1,
    )
    objective = sampled.mean()
    analytic = torch.autograd.grad(objective, boxes)[0][0, 0, 0]
    epsilon = 1.0e-3

    def evaluate(center_x: float):
        probe = boxes.detach().clone()
        probe[0, 0, 0] = center_x
        value = sample_continuous_roi(
            source,
            probe,
            output_height=5,
            output_width=5,
            frames_per_clip=1,
        )
        return value.mean()

    numeric = (
        evaluate(0.50 + epsilon) - evaluate(0.50 - epsilon)
    ) / (2.0 * epsilon)
    if not bool(torch.isfinite(analytic).item() and torch.isfinite(numeric).item()):
        raise FloatingPointError("sampler geometry gradient is non-finite")
    if float(analytic.abs().item()) <= 1.0e-5:
        raise RuntimeError("sampler geometry gradient is zero")
    relative_error = float(
        ((analytic - numeric).abs() / numeric.abs().clamp_min(1.0e-6)).item()
    )
    if relative_error > 0.05:
        raise RuntimeError(
            f"sampler finite-difference gradient mismatch: {relative_error}"
        )
    runtime = sample_continuous_roi_runtime(
        source,
        boxes.detach(),
        output_height=5,
        output_width=5,
        frames_per_clip=1,
    )
    if not torch.equal(sampled.detach(), runtime):
        raise RuntimeError("training and runtime samplers are not bitwise identical")
    return {
        "analytic_d_center_x": float(analytic.detach().cpu().item()),
        "numeric_d_center_x": float(numeric.detach().cpu().item()),
        "relative_error": relative_error,
        "runtime_bitwise_parity": True,
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_gate(
    *,
    config_path: Path,
    checkpoint_path: Path,
    expected_commit: str,
    device_text: str,
    amp: bool,
) -> dict:
    import torch

    importlib.import_module("opentad.datasets")
    importlib.import_module("opentad.models")
    importlib.import_module("opentad.models.backbones")
    from opentad.cores import build_optimizer
    from opentad.models import build_detector

    if device_text != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("formal Continuous-RoI Gate requires Slurm logical cuda:0")
    if checkpoint_path.name != NATIVE_CROP_PRETRAINED_FILENAME:
        raise ValueError("pretrained checkpoint filename changed")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != NATIVE_CROP_PRETRAINED_SHA256:
        raise ValueError("pretrained checkpoint SHA-256 mismatch")

    provenance = audit_code_provenance(expected_commit)
    protocol_audit = validate_protocol(load_protocol())
    implementation_audit = validate_implementation()
    cfg = Config.fromfile(str(config_path))
    if config_path.resolve() != CONFIGS["U128"].resolve():
        raise ValueError("formal Gate must use the canonical U128 config")
    gate_cfg = cfg.continuous_roi_s2_gate
    if (
        gate_cfg.precheck_only is not True
        or gate_cfg.allow_detector_training is not False
        or gate_cfg.official_test_open_allowed is not False
        or cfg.dataset.test is not None
    ):
        raise ValueError("U128 config does not remain fail-closed before Gate")

    model_cfg = copy.deepcopy(cfg.model)
    configured_checkpoint = Path(str(model_cfg.backbone.custom.pretrain))
    if configured_checkpoint.name != NATIVE_CROP_PRETRAINED_FILENAME:
        raise ValueError("canonical config changed checkpoint identity")
    model_cfg.backbone.custom.pretrain = str(checkpoint_path.resolve())
    model = build_detector(model_cfg)
    pretrained_audit = audit_loaded_pretrained_state(model, checkpoint_path)
    device = torch.device(device_text)
    model = model.to(device).train()
    model.backbone.set_successful_update_index(0)
    frozen_core_audit = audit_frozen_shared_core(model)

    optimizer_cfg = copy.deepcopy(cfg.optimizer)
    optimizer = build_optimizer(
        optimizer_cfg,
        SimpleNamespace(module=model),
        _Logger(),
    )
    optimizer_audit = audit_optimizer_coverage(model, optimizer)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260720)
    global_view = torch.randint(
        0,
        256,
        (1, 1, 3, 768, 96, 96),
        dtype=torch.uint8,
        generator=generator,
    ).to(device)
    source = torch.randint(
        0,
        256,
        (1, 1, 3, 768, 180, 320),
        dtype=torch.uint8,
        generator=generator,
    ).pin_memory()
    inputs = {
        "global": global_view,
        "source": source,
        "sample_key": torch.tensor([914], dtype=torch.long),
        "window_start": torch.tensor([0], dtype=torch.long),
    }
    masks = torch.ones((1, 768), device=device, dtype=torch.bool)
    metas = [
        {
            "video_name": "continuous_roi_s2_synthetic_development_gate",
            "fps": 30.0,
            "duration": 25.6,
            "snippet_stride": 1,
            "window_start_frame": 0,
            "window_size": 768,
            "offset_frames": 0,
        }
    ]
    gt_segments = [
        torch.tensor(
            [[96.0, 288.0], [360.0, 560.0]],
            device=device,
            dtype=torch.float32,
        )
    ]
    gt_labels = [torch.tensor([0, 7], device=device, dtype=torch.long)]

    captured: dict[str, object] = {}
    branch_features = {}

    def capture_projection(_module, args):
        captured["projection_input_shape"] = list(args[0].shape)

    def capture_fusion_inputs(_module, args):
        if len(args) != 2:
            raise RuntimeError("U128 fusion did not receive two branch features")
        branch_features["global_branch_feature"] = args[0]
        branch_features["local_branch_feature"] = args[1]
        args[0].retain_grad()
        args[1].retain_grad()

    def forward_losses():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(amp),
        ):
            values = model.forward_train(
                inputs,
                masks,
                metas,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
            )
        required_loss_keys = {
            "continuous_roi_global_aux_loss",
            "continuous_roi_local_aux_loss",
            "cost",
        }
        if not required_loss_keys.issubset(values):
            raise RuntimeError(
                f"full model losses lack {sorted(required_loss_keys - set(values))}"
            )
        for name, value in values.items():
            if not torch.is_tensor(value) or value.ndim != 0:
                raise TypeError(f"loss {name} is not a scalar tensor")
            if not bool(torch.isfinite(value).all().item()):
                raise FloatingPointError(f"loss {name} is non-finite")
        return values

    projection_hook = model.projection.register_forward_pre_hook(capture_projection)
    fusion_hook = model.backbone.fusion.register_forward_pre_hook(
        capture_fusion_inputs
    )
    started = time.perf_counter()
    try:
        detector_losses = forward_losses()
        detector_terms = [
            value
            for name, value in detector_losses.items()
            if name != "cost" and not name.startswith("continuous_roi_")
        ]
        if not detector_terms:
            raise RuntimeError("official detector produced no detector loss")
        detector_cost = sum(detector_terms)
        detector_target_names = []
        detector_gradients = []
        detector_cost.backward()
        torch.cuda.synchronize(device)
        for name, parameter in model.named_parameters():
            component = _parameter_component(name)
            if component in ("shared_adapter", "fusion", "projection", "rpn_head"):
                detector_target_names.append(name)
                detector_gradients.append(parameter.grad)
        for name in ("global_branch_feature", "local_branch_feature"):
            feature = branch_features.get(name)
            if feature is None:
                raise RuntimeError(f"missing captured {name}")
            detector_target_names.append(name)
            detector_gradients.append(feature.grad)
        detector_gradient_audit = _nonzero_finite_gradient_audit(
            detector_target_names,
            detector_gradients,
            required_components=(
                "shared_adapter",
                "fusion",
                "projection",
                "rpn_head",
                "global_branch_feature",
                "local_branch_feature",
            ),
        )
        optimizer.zero_grad(set_to_none=True)
        branch_features.clear()
        model.backbone.set_successful_update_index(0)
        losses = forward_losses()
        losses["cost"].backward()
        torch.cuda.synchronize(device)
    finally:
        projection_hook.remove()
        fusion_hook.remove()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    total_names = []
    total_gradients = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            total_names.append(name)
            total_gradients.append(parameter.grad)
    total_gradient_audit = _nonzero_finite_gradient_audit(
        total_names,
        total_gradients,
        required_components=(
            "shared_adapter",
            "fusion",
            "global_aux_head",
            "local_aux_head",
            "projection",
            "rpn_head",
        ),
    )
    if total_gradient_audit["missing_target_gradients"]:
        raise RuntimeError(
            "trainable parameters are disconnected from total cost: "
            f"{total_gradient_audit['missing_target_gradients']}"
        )
    optimizer.step()
    torch.cuda.synchronize(device)

    if captured.get("projection_input_shape") != [1, 384, 768]:
        raise RuntimeError("U128 did not preserve the detector input contract")
    wrapper_audit = model.backbone.latest_continuous_roi_audit
    if (
        wrapper_audit is None
        or wrapper_audit["shared_backbone_instances"] != 1
        or wrapper_audit["videomae_evaluations"] != 2
        or wrapper_audit["contains_selector"] is not False
        or wrapper_audit["policy_head_parameters"] != 0
        or wrapper_audit["new_parameters"] != 609449
        or wrapper_audit["intermediate_shape"] != [1, 384, 384]
        or wrapper_audit["output_shape"] != [1, 384, 768]
        or wrapper_audit["families"] != ["anchor"]
    ):
        raise RuntimeError(f"U128 wrapper audit is incomplete: {wrapper_audit}")
    if any(
        token in name.lower()
        for name, _ in model.named_parameters()
        for token in ("selector", "roi_policy", "policy_head")
    ):
        raise RuntimeError("S2 model unexpectedly contains a selector/policy parameter")

    try:
        model.backbone._resolve_boxes(
            {model.backbone.boxes_key: torch.zeros((1, 48, 4))},
            sample_keys=torch.tensor([1]),
            window_starts=torch.tensor([0]),
            device=device,
            dtype=torch.float32,
        )
    except ValueError as error:
        if "training forbids externally supplied geometry" not in str(error):
            raise
    else:
        raise RuntimeError("training accepted externally supplied geometry")

    sampler_gradient_audit = audit_sampler_geometry_gradient(device)
    loss_values = {
        name: float(value.detach().float().cpu().item())
        for name, value in losses.items()
    }
    result = {
        "schema_version": GATE_SCHEMA,
        "status": "PASS",
        "expected_commit": expected_commit,
        "code_provenance": provenance,
        "protocol_audit": protocol_audit,
        "implementation_audit": implementation_audit,
        "config_path": str(config_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "pretrained_load_audit": pretrained_audit,
        "device": device_text,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_device_uuid": str(torch.cuda.get_device_properties(device).uuid),
        "amp": bool(amp),
        "diagnostic_elapsed_ms": elapsed_ms,
        "paper_latency_claim_allowed": False,
        "peak_gpu_memory_bytes_diagnostic": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "losses": loss_values,
        "projection_input_shape": captured["projection_input_shape"],
        "backbone_audit": wrapper_audit,
        "frozen_core_audit": frozen_core_audit,
        "optimizer_audit": optimizer_audit,
        "detector_only_gradient_audit": detector_gradient_audit,
        "total_gradient_audit": total_gradient_audit,
        "sampler_geometry_gradient_audit": sampler_gradient_audit,
        "optimizer_step_completed": True,
        "training_external_geometry_rejected": True,
        "official_test_annotation_records_loaded": 0,
        "official_test_video_files_opened": 0,
        "teacher_used": False,
        "oracle_used": False,
        "learned_selector_present": False,
        "paper_claim_allowed": False,
        "formal_training_authorized_by_this_gate": True,
    }
    return finalize_self_hash(result, "gate_sha256")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run the Continuous-RoI S2 v2.1 full-model one-step CUDA Gate."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS["U128"],
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_gate(
        config_path=args.config.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        expected_commit=args.expected_commit,
        device_text=args.device,
        amp=args.amp,
    )
    _atomic_write_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "status": report["status"],
                "gate_sha256": report["gate_sha256"],
                "projection_input_shape": report["projection_input_shape"],
                "optimizer_step_completed": report["optimizer_step_completed"],
                "official_test_video_files_opened": 0,
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
