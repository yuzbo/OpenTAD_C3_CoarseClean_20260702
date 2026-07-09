from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from mmengine.config import Config, ConfigDict

from opentad.models import build_detector


SCHEMA_VERSION = "duca_official_adatad_one_step_grad_proof_v1"
FIXED_CONFIG_DEFAULT = "configs/adatad/thumos/duca_online_official_adatad_backend_full_train.py"
MUST_CONFIG_DEFAULT = "configs/adatad/thumos/duca_must_dynamic_official_adatad_backend_full_train.py"


class _ProofTemporalMeanBackbone(nn.Module):
    """Proof-only raw-video reducer that keeps gradients flowing to selected frames."""

    freeze_backbone = True

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv1d(int(in_channels), int(out_channels), kernel_size=1, bias=False)
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(f"official DUCA proof backbone expects [B,C,T,H,W], got {tuple(inputs.shape)}")
        return self.proj(inputs.mean(dim=(3, 4)))


def _path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _set_nested(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    current: dict[str, Any] = mapping
    parts = str(dotted).split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _grad_sum(module: nn.Module | None) -> float | None:
    if module is None:
        return None
    total = 0.0
    seen = False
    for param in module.parameters():
        if not param.requires_grad:
            continue
        seen = True
        if param.grad is not None:
            total += float(param.grad.detach().abs().sum().cpu().item())
    return total if seen else None


def _fail_closed(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"fail-closed official detector backend proof: {message}")


def _contract(cfg: Config) -> Any:
    for key in ("duca_online_main_contract", "duca_must_dynamic_contract"):
        if hasattr(cfg, key):
            return getattr(cfg, key)
    return None


def _require_official_adatad_config(cfg: Config, *, route: str) -> None:
    contract = _contract(cfg)
    head_type = str(cfg.model.rpn_head.type)
    model_type = str(cfg.model.type)
    model_text = repr(cfg.model)
    _fail_closed(contract is not None, f"{route}: DUCA official contract is missing")
    _fail_closed(
        bool(getattr(contract, "official_adatad_backend", False)),
        f"{route}: config does not declare official_adatad_backend=True",
    )
    _fail_closed(model_type == "ActionFormer", f"{route}: model.type must be ActionFormer, got {model_type!r}")
    _fail_closed(head_type == "ActionFormerHead", f"{route}: rpn_head.type must be ActionFormerHead, got {head_type!r}")
    _fail_closed(
        "DucaOnlinePrecheckHead" not in model_text,
        f"{route}: DucaOnlinePrecheckHead is a precheck/fake head, not the official detector backend",
    )
    _fail_closed(
        str(getattr(contract, "detector_head_type", "")) == "ActionFormerHead",
        f"{route}: contract detector_head_type must be ActionFormerHead",
    )


def _scaled_selector_cfg(
    cfg: Config,
    *,
    route: str,
    proof_temporal_len: int,
    proof_budget: int,
    proof_budget_min: int,
    proof_budget_target: int,
    proof_budget_multiple: int,
    proof_spatial_size: int,
    proof_hidden_dim: int,
) -> dict[str, Any]:
    selector_cfg = _plain(cfg.model.frame_selector)
    selector_cfg["dense_window_size"] = int(proof_temporal_len)
    selector_cfg["selector_hidden_channels"] = int(proof_hidden_dim)
    selector_cfg["coarse_hidden_dim"] = int(proof_hidden_dim)
    selector_cfg["use_coarse_hidden_features"] = True
    selector_cfg["require_coarse_hidden_features"] = True
    selector_cfg["max_radius"] = min(int(selector_cfg.get("max_radius", 16)), max(1, int(proof_temporal_len) // 4))
    selector_cfg["profile_runtime"] = False
    selector_cfg["profile_sync_cuda"] = False
    selector_cfg["detector_gradient_mode"] = "soft_to_hard_resample"
    if selector_cfg.get("budget_mode") == "dynamic_must":
        selector_cfg["budget"] = None
        selector_cfg["budget_min"] = int(proof_budget_min)
        selector_cfg["budget_max"] = int(proof_budget)
        selector_cfg["target_budget"] = int(proof_budget_target)
        selector_cfg["budget_multiple"] = int(proof_budget_multiple)
        selector_cfg["allow_external_budget_override"] = False
    else:
        selector_cfg["budget"] = int(proof_budget)
        selector_cfg["budget_mode"] = "fixed"
    actionness_cfg = selector_cfg.get("actionness_source_cfg")
    _fail_closed(isinstance(actionness_cfg, dict), f"{route}: selector must have actionness_source_cfg")
    actionness_cfg["spatial_size"] = int(proof_spatial_size)
    actionness_cfg["tcn_hidden_dim"] = int(proof_hidden_dim)
    actionness_cfg["official_num_layers"] = int(actionness_cfg.get("official_num_layers", 1))
    actionness_cfg["frozen"] = False
    actionness_cfg["trainable"] = True
    actionness_cfg["return_hidden_features"] = True
    actionness_cfg["require_hidden_features"] = True
    actionness_cfg["mobilenet_pretrained"] = False
    actionness_cfg["require_checkpoint"] = False
    _fail_closed(actionness_cfg.get("probe_model") == "official-action-seg", f"{route}: coarse probe must be official-action-seg")
    _fail_closed(
        actionness_cfg.get("official_action_seg_backend") == "official_asformer",
        f"{route}: coarse probe must use official_asformer backend",
    )
    _fail_closed(actionness_cfg.get("tcn_variant") != "asformer_lite", f"{route}: asformer_lite is forbidden")
    for key, value in {
        "loss_weight_schedule.shape": "linear",
        "loss_weight_schedule.warmup_steps": 0,
        "loss_weight_schedule.transition_steps": 1,
        "loss_weight_schedule.detector_gradient.start": 1.0,
        "loss_weight_schedule.detector_gradient.end": 1.0,
        "loss_weight_schedule.actionness.start": 0.5,
        "loss_weight_schedule.actionness.end": 0.5,
        "loss_weight_schedule.boundary.start": 1.0,
        "loss_weight_schedule.boundary.end": 1.0,
        "loss_weight_schedule.detector_utility.start": 0.1,
        "loss_weight_schedule.detector_utility.end": 0.1,
        "loss_weight_schedule.hole.start": 0.0,
        "loss_weight_schedule.hole.end": 0.0,
    }.items():
        _set_nested(selector_cfg, key, value)
    if selector_cfg.get("budget_mode") == "dynamic_must":
        _set_nested(selector_cfg, "loss_weight_schedule.lagrangian_budget.start", 1.0)
        _set_nested(selector_cfg, "loss_weight_schedule.lagrangian_budget.end", 1.0)
        _set_nested(selector_cfg, "loss_weight_schedule.marginal_monotonic.start", 0.01)
        _set_nested(selector_cfg, "loss_weight_schedule.marginal_monotonic.end", 0.01)
    return selector_cfg


def _scaled_model_cfg(
    cfg: Config,
    *,
    selector_cfg: dict[str, Any],
    proof_budget: int,
    proof_feature_dim: int,
) -> dict[str, Any]:
    feature_dim = int(proof_feature_dim)
    levels = 3
    projection = _plain(cfg.model.projection)
    projection.update(
        {
            "in_channels": feature_dim,
            "out_channels": feature_dim,
            "arch": (1, 0, levels - 1),
            "conv_cfg": {"kernel_size": 3, "proj_pdrop": 0.0},
            "norm_cfg": {"type": "LN"},
            "attn_cfg": {"n_head": 2, "n_mha_win_size": -1},
            "path_pdrop": 0.0,
            "use_abs_pe": False,
            "max_seq_len": int(proof_budget),
        }
    )
    neck = _plain(cfg.model.neck)
    neck.update({"in_channels": feature_dim, "out_channels": feature_dim, "num_levels": levels})
    rpn_head = _plain(cfg.model.rpn_head)
    if isinstance(rpn_head.get("loss"), dict):
        rpn_head["loss"] = ConfigDict(rpn_head["loss"])
    rpn_head.update(
        {
            "type": "ActionFormerHead",
            "in_channels": feature_dim,
            "feat_channels": feature_dim,
            "num_convs": 1,
            "loss_normalizer": 1,
            "prior_generator": {
                "type": "PointGenerator",
                "strides": [1, 2, 4],
                "regression_range": [(0, 4), (4, 8), (8, 10000)],
            },
        }
    )
    return {
        "type": "ActionFormer",
        "frame_selector": selector_cfg,
        "projection": projection,
        "neck": neck,
        "rpn_head": rpn_head,
    }


def build_official_proof_model(
    *,
    config_path: str | Path,
    route: str,
    proof_temporal_len: int,
    proof_budget: int,
    proof_budget_min: int,
    proof_budget_target: int,
    proof_budget_multiple: int,
    proof_spatial_size: int,
    proof_hidden_dim: int,
    proof_feature_dim: int,
) -> tuple[nn.Module, dict[str, Any]]:
    cfg = Config.fromfile(str(_path(config_path)))
    _require_official_adatad_config(cfg, route=route)
    selector_cfg = _scaled_selector_cfg(
        cfg,
        route=route,
        proof_temporal_len=proof_temporal_len,
        proof_budget=proof_budget,
        proof_budget_min=proof_budget_min,
        proof_budget_target=proof_budget_target,
        proof_budget_multiple=proof_budget_multiple,
        proof_spatial_size=proof_spatial_size,
        proof_hidden_dim=proof_hidden_dim,
    )
    model_cfg = _scaled_model_cfg(cfg, selector_cfg=selector_cfg, proof_budget=proof_budget, proof_feature_dim=proof_feature_dim)
    model = build_detector(model_cfg)
    model.backbone = _ProofTemporalMeanBackbone(in_channels=3, out_channels=int(proof_feature_dim))
    _fail_closed(model.__class__.__name__ == "ActionFormer", f"{route}: built model is not ActionFormer")
    _fail_closed(model.rpn_head.__class__.__name__ == "ActionFormerHead", f"{route}: built rpn_head is not ActionFormerHead")
    _fail_closed(
        model.rpn_head.__class__.__name__ != "DucaOnlinePrecheckHead",
        f"{route}: DucaOnlinePrecheckHead cannot prove the official detector backend",
    )
    summary = {
        "route": route,
        "config_path": str(_path(config_path)),
        "official_config_loaded": True,
        "proof_mode": "scaled_official_config_actionformer_head_loss",
        "model_type": model.__class__.__name__,
        "rpn_head_type": model.rpn_head.__class__.__name__,
        "proof_uses_precheck_head": model.rpn_head.__class__.__name__ == "DucaOnlinePrecheckHead",
        "proof_temporal_len": int(proof_temporal_len),
        "proof_budget": int(proof_budget),
        "proof_backbone": "_ProofTemporalMeanBackbone",
        "official_detector_backend": "ActionFormerHead",
    }
    return model, summary


def assert_frame_selector_optimizer_coverage(model: nn.Module, *, lr: float, weight_decay: float) -> dict[str, Any]:
    groups = model.get_optim_groups({"lr": float(lr), "weight_decay": float(weight_decay)})
    covered = {id(param) for group in groups for param in group["params"]}
    trainable = [
        name
        for name, param in model.named_parameters()
        if name.startswith("frame_selector.") and param.requires_grad
    ]
    missing = [
        name
        for name, param in model.named_parameters()
        if name.startswith("frame_selector.") and param.requires_grad and id(param) not in covered
    ]
    if missing:
        raise RuntimeError(f"optimizer coverage failed: missing trainable frame_selector params {missing}")
    return {
        "optimizer_group_count": len(groups),
        "trainable_frame_selector_param_count": len(trainable),
        "covered_frame_selector_param_count": len(trainable) - len(missing),
        "missing_frame_selector_params": missing,
    }


def _run_route(
    *,
    config_path: str | Path,
    route: str,
    proof_temporal_len: int,
    proof_budget: int,
    proof_budget_min: int,
    proof_budget_target: int,
    proof_budget_multiple: int,
    proof_spatial_size: int,
    proof_hidden_dim: int,
    proof_feature_dim: int,
    device: str,
) -> dict[str, Any]:
    model, summary = build_official_proof_model(
        config_path=config_path,
        route=route,
        proof_temporal_len=proof_temporal_len,
        proof_budget=proof_budget,
        proof_budget_min=proof_budget_min,
        proof_budget_target=proof_budget_target,
        proof_budget_multiple=proof_budget_multiple,
        proof_spatial_size=proof_spatial_size,
        proof_hidden_dim=proof_hidden_dim,
        proof_feature_dim=proof_feature_dim,
    )
    model.to(device)
    model.train()
    optimizer_coverage = assert_frame_selector_optimizer_coverage(model, lr=1.0e-4, weight_decay=0.05)
    optimizer = torch.optim.AdamW(model.get_optim_groups({"lr": 1.0e-4, "weight_decay": 0.05}))
    torch.manual_seed(20260709)
    inputs = torch.randn(
        1,
        3,
        int(proof_temporal_len),
        int(proof_spatial_size),
        int(proof_spatial_size),
        device=device,
    )
    masks = torch.ones(1, int(proof_temporal_len), dtype=torch.bool, device=device)
    gt_segments = [torch.tensor([[1.0, float(max(3, proof_temporal_len - 2))]], dtype=torch.float32, device=device)]
    gt_labels = [torch.tensor([1], dtype=torch.long, device=device)]
    losses = model(
        inputs,
        masks,
        [{"video_name": f"{route}_official_grad_proof"}],
        gt_segments=gt_segments,
        gt_labels=gt_labels,
        return_loss=True,
    )
    for key in ("cost", "cls_loss", "reg_loss"):
        _fail_closed(key in losses, f"{route}: official ActionFormerHead loss key {key!r} is missing")
    optimizer.zero_grad(set_to_none=True)
    losses["cost"].backward()
    optimizer.step()
    after_optimizer_step_summary = model.after_optimizer_step()
    selector = model.frame_selector
    result = dict(summary)
    result.update(
        {
            "proof_passed": True,
            "budget_mode": str(selector.budget_mode),
            "loss_cost": float(losses["cost"].detach().cpu().item()),
            "loss_keys": sorted(str(key) for key in losses.keys()),
            "optimizer_step_ran": True,
            "optimizer_coverage": optimizer_coverage,
            "coarse_probe_grad_sum": _grad_sum(selector.raw_actionness_source),
            "selector_encoder_grad_sum": _grad_sum(selector.adapter.encoder),
            "selector_center_head_grad_sum": _grad_sum(selector.adapter.center_head),
            "budget_controller_grad_sum": _grad_sum(selector.adapter.budget_controller),
            "after_optimizer_step_summary": after_optimizer_step_summary,
            "loss_schedule_step_update": selector.last_forward_summary.get("loss_schedule_step_update", {}),
            "loss_schedule": selector.last_forward_summary.get("loss_weight_schedule", {}),
            "selected_count": int(selector.last_forward_summary.get(selector.metadata_keys["selected_count"], 0)),
        }
    )
    required = ["coarse_probe_grad_sum", "selector_encoder_grad_sum", "selector_center_head_grad_sum"]
    if selector.budget_mode == "dynamic_must":
        required.append("budget_controller_grad_sum")
    missing_grad = [key for key in required if result.get(key) is None or float(result[key]) <= 0.0]
    if missing_grad:
        raise RuntimeError(f"{route}: losses['cost'].backward() did not produce nonzero gradients for {missing_grad}")
    return result


def run_proof(
    *,
    fixed_config: str | Path = FIXED_CONFIG_DEFAULT,
    must_config: str | Path = MUST_CONFIG_DEFAULT,
    proof_temporal_len: int = 32,
    proof_budget: int = 16,
    proof_budget_min: int = 4,
    proof_budget_target: int = 8,
    proof_budget_multiple: int = 4,
    proof_spatial_size: int = 16,
    proof_hidden_dim: int = 16,
    proof_feature_dim: int = 16,
    device: str = "cpu",
) -> dict[str, Any]:
    fixed = _run_route(
        config_path=fixed_config,
        route="fixed384",
        proof_temporal_len=proof_temporal_len,
        proof_budget=proof_budget,
        proof_budget_min=proof_budget_min,
        proof_budget_target=proof_budget_target,
        proof_budget_multiple=proof_budget_multiple,
        proof_spatial_size=proof_spatial_size,
        proof_hidden_dim=proof_hidden_dim,
        proof_feature_dim=proof_feature_dim,
        device=device,
    )
    must = _run_route(
        config_path=must_config,
        route="duca_must",
        proof_temporal_len=proof_temporal_len,
        proof_budget=proof_budget,
        proof_budget_min=proof_budget_min,
        proof_budget_target=proof_budget_target,
        proof_budget_multiple=proof_budget_multiple,
        proof_spatial_size=proof_spatial_size,
        proof_hidden_dim=proof_hidden_dim,
        proof_feature_dim=proof_feature_dim,
        device=device,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "proof_passed": True,
        "fixed384": fixed,
        "duca_must": must,
    }


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = _path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run fail-closed DUCA official AdaTAD/ActionFormer one-step gradient proof."
    )
    parser.add_argument("--fixed-config", default=FIXED_CONFIG_DEFAULT)
    parser.add_argument("--must-config", default=MUST_CONFIG_DEFAULT)
    parser.add_argument("--output-json")
    parser.add_argument("--proof-temporal-len", type=int, default=int(os.environ.get("DUCA_OFFICIAL_PROOF_TEMPORAL_LEN", "32")))
    parser.add_argument("--proof-budget", type=int, default=int(os.environ.get("DUCA_OFFICIAL_PROOF_BUDGET", "16")))
    parser.add_argument("--proof-budget-min", type=int, default=int(os.environ.get("DUCA_OFFICIAL_PROOF_BUDGET_MIN", "4")))
    parser.add_argument("--proof-budget-target", type=int, default=int(os.environ.get("DUCA_OFFICIAL_PROOF_BUDGET_TARGET", "8")))
    parser.add_argument("--proof-budget-multiple", type=int, default=int(os.environ.get("DUCA_OFFICIAL_PROOF_BUDGET_MULTIPLE", "4")))
    parser.add_argument("--proof-spatial-size", type=int, default=int(os.environ.get("DUCA_OFFICIAL_PROOF_SPATIAL_SIZE", "16")))
    parser.add_argument("--proof-hidden-dim", type=int, default=int(os.environ.get("DUCA_OFFICIAL_PROOF_HIDDEN_DIM", "16")))
    parser.add_argument("--proof-feature-dim", type=int, default=int(os.environ.get("DUCA_OFFICIAL_PROOF_FEATURE_DIM", "16")))
    parser.add_argument("--device", default=os.environ.get("DUCA_OFFICIAL_PROOF_DEVICE", "cpu"))
    args = parser.parse_args(argv)
    try:
        summary = run_proof(
            fixed_config=args.fixed_config,
            must_config=args.must_config,
            proof_temporal_len=args.proof_temporal_len,
            proof_budget=args.proof_budget,
            proof_budget_min=args.proof_budget_min,
            proof_budget_target=args.proof_budget_target,
            proof_budget_multiple=args.proof_budget_multiple,
            proof_spatial_size=args.proof_spatial_size,
            proof_hidden_dim=args.proof_hidden_dim,
            proof_feature_dim=args.proof_feature_dim,
            device=args.device,
        )
    except Exception as exc:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "proof_passed": False,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "fail_closed": True,
            "fixed_config": str(args.fixed_config),
            "must_config": str(args.must_config),
        }
        if args.output_json:
            _write_json(args.output_json, summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 1
    if args.output_json:
        _write_json(args.output_json, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
