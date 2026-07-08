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
from mmengine.config import Config

from opentad.models import build_detector


SCHEMA_VERSION = "duca_jct_one_step_grad_proof_v1"


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


def _grad_sum(module: torch.nn.Module | None) -> float | None:
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
    selector_cfg["max_radius"] = min(int(selector_cfg.get("max_radius", 16)), max(1, int(proof_temporal_len) // 2))
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
    if isinstance(actionness_cfg, dict):
        actionness_cfg["spatial_size"] = int(proof_spatial_size)
        actionness_cfg["tcn_hidden_dim"] = int(proof_hidden_dim)
        actionness_cfg["official_num_layers"] = int(actionness_cfg.get("official_num_layers", 1))
        actionness_cfg["frozen"] = False
        actionness_cfg["trainable"] = True
        if actionness_cfg.get("tcn_variant") == "asformer_lite":
            raise ValueError(f"{route}: asformer_lite is forbidden for DUCA-JCT proof")
        if actionness_cfg.get("probe_model") != "official-action-seg":
            raise ValueError(f"{route}: proof expects official-action-seg coarse probe")
        if actionness_cfg.get("official_action_seg_backend") != "official_asformer":
            raise ValueError(f"{route}: proof expects official_asformer backend")
    for key, value in {
        "loss_weight_schedule.warmup_steps": 0,
        "loss_weight_schedule.transition_steps": 1,
        "loss_weight_schedule.shape": "linear",
        "loss_weight_schedule.detector_gradient.start": 1.0,
        "loss_weight_schedule.detector_gradient.end": 1.0,
        "loss_weight_schedule.actionness.start": 0.5,
        "loss_weight_schedule.actionness.end": 0.5,
        "loss_weight_schedule.hole.start": 0.1,
        "loss_weight_schedule.hole.end": 0.1,
    }.items():
        _set_nested(selector_cfg, key, value)
    if selector_cfg.get("budget_mode") == "dynamic_must":
        _set_nested(selector_cfg, "loss_weight_schedule.lagrangian_budget.start", 1.0)
        _set_nested(selector_cfg, "loss_weight_schedule.lagrangian_budget.end", 1.0)
        _set_nested(selector_cfg, "loss_weight_schedule.marginal_monotonic.start", 0.01)
        _set_nested(selector_cfg, "loss_weight_schedule.marginal_monotonic.end", 0.01)
    return selector_cfg


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
    device: str,
) -> dict[str, Any]:
    cfg = Config.fromfile(str(_path(config_path)))
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
    model = build_detector(
        {
            "type": "SingleStageDetector",
            "frame_selector": selector_cfg,
            "rpn_head": {
                "type": "DucaOnlinePrecheckHead",
                "in_channels": 3,
                "require_gt_in_train": True,
                "require_selected_metas": True,
                "require_original_time_positions": True,
                "require_selected_axis_remap": True,
                "metadata_keys": selector_cfg.get("metadata_keys"),
            },
        }
    ).to(device)
    model.train()
    torch.manual_seed(20260709)
    inputs = torch.randn(1, 3, int(proof_temporal_len), int(proof_spatial_size), int(proof_spatial_size), device=device)
    masks = torch.ones(1, int(proof_temporal_len), dtype=torch.bool, device=device)
    gt_segments = [torch.tensor([[1.0, float(max(2, proof_temporal_len - 1))]], dtype=torch.float32, device=device)]
    gt_labels = [torch.tensor([1], dtype=torch.long, device=device)]
    losses = model(
        inputs,
        masks,
        [{"video_name": f"{route}_one_step"}],
        gt_segments=gt_segments,
        gt_labels=gt_labels,
        return_loss=True,
    )
    cost = losses["cost"]
    cost.backward()
    optimizer = torch.optim.SGD((param for param in model.parameters() if param.requires_grad), lr=1e-4)
    optimizer.step()
    dual_update = model.after_optimizer_step()
    selector = model.frame_selector
    result = {
        "route": route,
        "config_path": str(_path(config_path)),
        "official_config_loaded": True,
        "proof_mode": "scaled_official_config_selector_one_step",
        "proof_temporal_len": int(proof_temporal_len),
        "proof_budget": int(proof_budget),
        "budget_mode": str(selector.budget_mode),
        "loss_cost": float(cost.detach().cpu().item()),
        "optimizer_step_ran": True,
        "loss_keys": sorted(str(key) for key in losses.keys()),
        "coarse_probe_grad_sum": _grad_sum(selector.raw_actionness_source),
        "selector_encoder_grad_sum": _grad_sum(selector.adapter.encoder),
        "selector_center_head_grad_sum": _grad_sum(selector.adapter.center_head),
        "budget_controller_grad_sum": _grad_sum(selector.adapter.budget_controller),
        "dynamic_budget_dual_update": dual_update,
        "loss_schedule": selector.last_forward_summary.get("loss_weight_schedule", {}),
        "selected_count": int(selector.last_forward_summary.get(selector.metadata_keys["selected_count"], 0)),
        "compute_profile": selector.last_forward_summary.get("compute_profile", {}),
    }
    required = ["coarse_probe_grad_sum", "selector_encoder_grad_sum", "selector_center_head_grad_sum"]
    if selector.budget_mode == "dynamic_must":
        required.append("budget_controller_grad_sum")
    missing = [key for key in required if result.get(key) is None or float(result[key]) <= 0.0]
    if missing:
        raise RuntimeError(f"{route}: nonzero gradient proof failed for {missing}")
    if selector.budget_mode == "dynamic_must" and not (isinstance(dual_update, dict) and dual_update.get("updated")):
        raise RuntimeError(f"{route}: dynamic budget dual update did not run")
    return result


def run_proof(
    *,
    fixed_config: str | Path,
    must_config: str | Path,
    proof_temporal_len: int = 16,
    proof_budget: int = 16,
    proof_budget_min: int = 4,
    proof_budget_target: int = 8,
    proof_budget_multiple: int = 4,
    proof_spatial_size: int = 16,
    proof_hidden_dim: int = 16,
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
    parser = argparse.ArgumentParser(description="Run one-step DUCA-JCT joint gradient proof from official configs.")
    parser.add_argument("--fixed-config", required=True)
    parser.add_argument("--must-config", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--proof-temporal-len", type=int, default=int(os.environ.get("DUCA_JCT_PROOF_TEMPORAL_LEN", "16")))
    parser.add_argument("--proof-budget", type=int, default=int(os.environ.get("DUCA_JCT_PROOF_BUDGET", "16")))
    parser.add_argument("--proof-budget-min", type=int, default=int(os.environ.get("DUCA_JCT_PROOF_BUDGET_MIN", "4")))
    parser.add_argument("--proof-budget-target", type=int, default=int(os.environ.get("DUCA_JCT_PROOF_BUDGET_TARGET", "8")))
    parser.add_argument("--proof-budget-multiple", type=int, default=int(os.environ.get("DUCA_JCT_PROOF_BUDGET_MULTIPLE", "4")))
    parser.add_argument("--proof-spatial-size", type=int, default=int(os.environ.get("DUCA_JCT_PROOF_SPATIAL_SIZE", "16")))
    parser.add_argument("--proof-hidden-dim", type=int, default=int(os.environ.get("DUCA_JCT_PROOF_HIDDEN_DIM", "16")))
    parser.add_argument("--device", default=os.environ.get("DUCA_JCT_PROOF_DEVICE", "cpu"))
    args = parser.parse_args(argv)
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
        device=args.device,
    )
    if args.output_json:
        _write_json(args.output_json, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
