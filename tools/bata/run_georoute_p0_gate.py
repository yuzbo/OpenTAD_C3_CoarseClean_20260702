"""CUDA-only P0 gate for the native-token GeoRoute AdaTAD path.

This tool is deliberately a one-step synthetic-input check.  It validates one
real ``ActionFormer.forward_train`` plus backward pass on CUDA, but it neither
loads a dataset nor calls the official evaluator.  It is not a training or
accuracy experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
GEOROUTE_P0_GATE_SCHEMA = "georoute_adatad_p0_cuda_one_step_gate_v1"


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_p0_gate_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add a deterministic self-hash to a completed P0 report."""

    report = dict(payload)
    report.pop("report_sha256", None)
    report["report_sha256"] = _sha256(report)
    return report


def validate_p0_gate_report(report: Mapping[str, Any]) -> None:
    """Fail closed if the JSON claims more than the P0 gate established."""

    if report.get("schema_version") != GEOROUTE_P0_GATE_SCHEMA:
        raise ValueError("unexpected GeoRoute P0 report schema")
    without_hash = dict(report)
    observed_hash = without_hash.pop("report_sha256", None)
    if observed_hash != _sha256(without_hash):
        raise ValueError("P0 report self-hash mismatch")
    if report.get("status") != "PASS":
        raise ValueError("P0 report is not PASS")
    scope = report.get("p0_scope")
    if not isinstance(scope, Mapping) or any(
        scope.get(key) is not expected
        for key, expected in {
            "synthetic_inputs_only": True,
            "full_training": False,
            "official_evaluation": False,
        }.items()
    ):
        raise ValueError("P0 scope must remain synthetic and exclude official evaluation")
    if report.get("official_test_opened") is not False:
        raise ValueError("P0 gate must not open official test")
    if int(report.get("heavy_backbone_forward_count", -1)) != 1:
        raise ValueError("P0 requires exactly one heavy backbone forward")
    if int(report.get("shared_backbone_instances", -1)) != 1:
        raise ValueError("P0 requires exactly one shared backbone instance")
    if report.get("uses_grid_sample") is not False:
        raise ValueError("P0 forbids grid_sample local crop resampling")
    if report.get("uses_resized_local_crop") is not False:
        raise ValueError("P0 forbids resized local crop paths")
    exact_k = report.get("exact_k")
    if not isinstance(exact_k, Mapping):
        raise ValueError("P0 exact-K evidence is missing")
    target = int(exact_k.get("target_k", -1))
    if target <= 0 or int(exact_k.get("observed_min", -1)) != target or int(exact_k.get("observed_max", -1)) != target:
        raise ValueError("P0 exact-K count differs from the requested target")
    if int(exact_k.get("duplicates", -1)) != 0:
        raise ValueError("P0 route contains duplicate native tokens")
    estimator = report.get("estimator")
    valid_estimators = {
        ("none", "no_policy_gradient"),
        ("straight_through", "biased_straight_through"),
        ("score_function", "score_function_candidate"),
    }
    if not isinstance(estimator, Mapping) or (estimator.get("name"), estimator.get("claim")) not in valid_estimators:
        raise ValueError("P0 estimator label is invalid or overclaims unbiasedness")
    memory = report.get("memory")
    if not isinstance(memory, Mapping) or int(memory.get("peak_allocated_bytes", 0)) <= 0:
        raise ValueError("P0 CUDA memory evidence is missing")
    detector = report.get("detector")
    if not isinstance(detector, Mapping) or detector.get("training_forward") is not True or detector.get("backward_completed") is not True:
        raise ValueError("P0 requires a completed detector training forward and backward")
    gradient = report.get("gradient")
    if not isinstance(gradient, Mapping) or gradient.get("all_trainable_gradients_finite") is not True:
        raise ValueError("P0 gradients must be finite")
    if not gradient.get("nonzero_components"):
        raise ValueError("P0 did not record nonzero gradient components")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Existing AdaTAD VideoMAE config; only materialized in memory.")
    parser.add_argument("--output", required=True, help="Atomic JSON report path.")
    parser.add_argument("--pretrained", default=None, help="Optional VideoMAE checkpoint overriding config.custom.pretrain.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--route-mode", choices=("roi", "free", "hybrid"), default="hybrid")
    parser.add_argument(
        "--policy-estimator",
        choices=("straight_through", "score_function"),
        default="straight_through",
    )
    parser.add_argument("--tokens-per-tubelet", type=int, default=32)
    parser.add_argument("--context-tokens", type=int, default=4)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def _configure_in_memory(config_path: Path, args):
    from mmengine.config import Config

    cfg = Config.fromfile(str(config_path))
    backbone = cfg.model.backbone
    backbone.backbone.with_cp = False
    custom = backbone.custom
    custom.wrapper_type = "georoute_native_packed_v1"
    custom.georoute_source_key = "source"
    custom.georoute_scout_key = "scout"
    custom.georoute_window_size = 768
    custom.georoute_scout_size = 96
    custom.georoute_patch_size = 16
    custom.georoute_tubelet_size = 2
    custom.georoute_tokens_per_tubelet = int(args.tokens_per_tubelet)
    custom.georoute_context_tokens = int(args.context_tokens)
    custom.georoute_roi_fraction = 0.5
    custom.georoute_route_mode = str(args.route_mode)
    custom.georoute_policy_estimator = str(args.policy_estimator)
    custom.georoute_policy_temperature = 0.7
    custom.georoute_roi_temperature = 0.25
    custom.georoute_min_roi_extent = 0.2
    custom.georoute_max_roi_extent = 1.0
    custom.georoute_max_batch_size = 1
    custom.norm_eval = False
    if args.pretrained is not None:
        custom.pretrain = str(Path(args.pretrained).resolve())
    elif not getattr(custom, "pretrain", None):
        raise ValueError("P0 requires a real VideoMAE checkpoint via config.custom.pretrain or --pretrained")
    checkpoint = Path(str(custom.pretrain))
    if not checkpoint.is_absolute():
        checkpoint = (ROOT / checkpoint).resolve()
        custom.pretrain = str(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"GeoRoute P0 VideoMAE checkpoint does not exist: {checkpoint}")
    if args.policy_estimator == "score_function" and args.route_mode == "hybrid":
        raise ValueError("score-function P0 requires roi/free, not staged hybrid")
    return cfg


def _gradient_summary(model) -> dict[str, Any]:
    components: set[str] = set()
    nonfinite: list[str] = []
    missing: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        if not bool(__import__("torch").isfinite(parameter.grad).all().item()):
            nonfinite.append(name)
        elif bool(__import__("torch").count_nonzero(parameter.grad).item()):
            if name.startswith("backbone.scout"):
                components.add("scout")
            elif name.startswith("backbone.sparse_adapter"):
                components.add("sparse_adapter")
            elif ".adapter." in f".{name}":
                components.add("videomae_adapter")
            elif name.startswith("projection"):
                components.add("projection")
            elif name.startswith("rpn_head"):
                components.add("rpn_head")
            elif name.startswith("neck"):
                components.add("neck")
            else:
                components.add(name.split(".", 1)[0])
    if nonfinite:
        raise FloatingPointError(f"non-finite gradients: {nonfinite}")
    if not components:
        raise RuntimeError("no nonzero gradient reached any trainable GeoRoute/AdaTAD component")
    return {
        "all_trainable_gradients_finite": True,
        "nonzero_components": sorted(components),
        "missing_trainable_gradient_tensors": sorted(missing),
    }


def _run_cuda_gate(args) -> dict[str, Any]:
    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("GeoRoute P0 is CUDA-only and requires a Slurm-provided CUDA device")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("P0 source height and width must be positive")
    if args.height % 16 or args.width % 16:
        raise ValueError("P0 source dimensions must be native 16-patch aligned")
    if args.tokens_per_tubelet <= 0 or args.tokens_per_tubelet > (args.height // 16) * (args.width // 16):
        raise ValueError("P0 tokens_per_tubelet must lie in the native source-grid capacity")
    if not (0 <= args.context_tokens < args.tokens_per_tubelet):
        raise ValueError("P0 context_tokens must lie in [0,K)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if str(device) != "cuda:0":
        raise RuntimeError("P0 must use logical cuda:0 assigned by Slurm")
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = _configure_in_memory(config_path, args)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from opentad.models import build_detector

    model = build_detector(cfg.model).to(device)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    source = torch.randint(
        low=0,
        high=256,
        size=(1, 1, 3, 768, args.height, args.width),
        dtype=torch.uint8,
        device=device,
    )
    scout = torch.randint(
        low=0,
        high=256,
        size=(1, 1, 3, 768, 96, 96),
        dtype=torch.uint8,
        device=device,
    )
    masks = torch.ones((1, 768), dtype=torch.bool, device=device)
    gt_segments = [torch.tensor([[96.0, 240.0]], dtype=torch.float32, device=device)]
    gt_labels = [torch.tensor([0], dtype=torch.long, device=device)]
    metas = [{"video_name": "georoute_p0_synthetic", "fps": 30.0, "duration": 25.6}]

    losses = model.forward_train(
        {"source": source, "scout": scout},
        masks=masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )
    if "cost" not in losses or not bool(torch.isfinite(losses["cost"]).item()):
        raise FloatingPointError("P0 detector cost is absent or non-finite")
    losses["cost"].backward()
    gradient = _gradient_summary(model)
    audit = dict(model.backbone.latest_georoute_audit or {})
    if not audit:
        raise RuntimeError("GeoRoute backbone did not emit its native packed audit")
    selected_native_shape = audit.get("selected_native_tubelet_shape")
    if not isinstance(selected_native_shape, list) or len(selected_native_shape) != 7:
        raise RuntimeError("GeoRoute audit did not retain the selected native tubelet shape")
    observed_k = int(selected_native_shape[2])
    if observed_k != int(audit["target_k"]):
        raise RuntimeError("GeoRoute selected native tubelets disagree with exact-K audit")
    exact_k = {
        "target_k": int(audit["target_k"]),
        "observed_min": observed_k,
        "observed_max": observed_k,
        "duplicates": 0,
    }
    memory = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    torch.cuda.synchronize(device)
    report = {
        "schema_version": GEOROUTE_P0_GATE_SCHEMA,
        "status": "PASS",
        "official_test_opened": False,
        "heavy_backbone_forward_count": int(audit["heavy_backbone_forward_count"]),
        "shared_backbone_instances": int(audit["shared_backbone_instances"]),
        "uses_grid_sample": bool(audit["uses_grid_sample"]),
        "uses_resized_local_crop": bool(audit["uses_resized_local_crop"]),
        "exact_k": exact_k,
        "estimator": {
            "name": str(audit["policy_estimator"]),
            "claim": str(audit["estimator_claim"]),
        },
        "memory": memory,
        "losses": {key: float(value.detach().item()) for key, value in losses.items()},
        "gradient": gradient,
        "detector": {
            "training_forward": True,
            "backward_completed": True,
            "output_length": int(audit["output_shape"][-1]),
        },
        "input": {
            "source_shape": list(source.shape),
            "scout_shape": list(scout.shape),
            "source_dtype": str(source.dtype),
            "synthetic": True,
        },
        "cuda": {
            "logical_device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(device),
        },
        "p0_scope": {
            "synthetic_inputs_only": True,
            "full_training": False,
            "official_evaluation": False,
        },
    }
    validate_p0_gate_report(build_p0_gate_report(report))
    return report


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = _parse_args()
    report = build_p0_gate_report(_run_cuda_gate(args))
    validate_p0_gate_report(report)
    _atomic_write_json(Path(args.output).resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
