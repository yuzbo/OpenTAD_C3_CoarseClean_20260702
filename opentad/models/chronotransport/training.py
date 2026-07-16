from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn

from .losses import pinball_loss, transport_consistency_loss


def _set_trainable(module: nn.Module, predicate) -> tuple[str, ...]:
    names = []
    for name, parameter in module.named_parameters():
        enabled = bool(predicate(name))
        parameter.requires_grad = enabled
        if enabled:
            names.append(name)
    return tuple(names)


def configure_stage_b(model: nn.Module) -> tuple[str, ...]:
    return _set_trainable(
        model,
        lambda name: "chronotransport.transport" in name or "chronotransport.risk_predictor" in name,
    )


def set_stage_b_module_modes(model: nn.Module) -> tuple[str, ...]:
    model.eval()
    enabled = []
    for name, module in model.named_modules():
        if (
            "chronotransport.transport" in name
            or "chronotransport.risk_predictor" in name
        ):
            module.train()
            enabled.append(name)
    if not enabled:
        raise RuntimeError("Stage B found no ChronoTransport train-mode modules")
    return tuple(enabled)


def snapshot_model_state(model: nn.Module) -> dict[str, Tensor]:
    """Capture an immutable CPU snapshot for post-step mutation auditing."""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _is_stage_b_dynamic_state(name: str) -> bool:
    return any(
        fragment in name
        for fragment in (
            "chronotransport.transport",
            "chronotransport.risk_predictor",
            # The scheduler holds a registered alias of risk_predictor.
            "chronotransport.scheduler.predictor",
        )
    )


def validate_stage_b_state_changes(
    before: Mapping[str, Tensor], model: nn.Module
) -> dict[str, object]:
    """Fail unless a Stage-B step changes only its dynamic state."""
    after = model.state_dict()
    missing = sorted(set(before) - set(after))
    unexpected = sorted(set(after) - set(before))
    if missing or unexpected:
        raise RuntimeError(
            f"model state keys changed during Stage B: missing={missing}, unexpected={unexpected}"
        )

    changed = []
    nonfinite = []
    for name, value in after.items():
        current = value.detach().cpu()
        if not torch.equal(before[name], current):
            changed.append(name)
        if (current.is_floating_point() or current.is_complex()) and not torch.isfinite(
            current
        ).all():
            nonfinite.append(name)

    dynamic_changed = sorted(name for name in changed if _is_stage_b_dynamic_state(name))
    frozen_changed = sorted(name for name in changed if not _is_stage_b_dynamic_state(name))
    if frozen_changed:
        raise RuntimeError(f"frozen model state changed during Stage B: {frozen_changed}")
    if not dynamic_changed:
        raise RuntimeError("Stage B completed without changing dynamic model state")
    if nonfinite:
        raise RuntimeError(f"non-finite model state after Stage B: {sorted(nonfinite)}")
    return {
        "status": "PASS",
        "changed_total": len(changed),
        "dynamic_changed": len(dynamic_changed),
        "dynamic_changed_examples": dynamic_changed[:8],
        "frozen_changed": [],
        "nonfinite": [],
    }


def configure_stage_c(model: nn.Module) -> tuple[str, ...]:
    """Compatibility entry point backed by the strict r2 object registry.

    Generic parameter-name substring grouping is intentionally disabled.
    """

    from .stage_c import build_stage_c_parameter_groups

    groups = build_stage_c_parameter_groups(model)
    return groups.parameter_names


def validate_split_partition(
    fit_ids: Iterable[str],
    calibration_ids: Iterable[str],
    evaluation_ids: Iterable[str],
) -> None:
    groups = [set(map(str, values)) for values in (fit_ids, calibration_ids, evaluation_ids)]
    if any(not values for values in groups):
        raise ValueError("fit, calibration, and evaluation splits must be non-empty")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("risk fit/calibration/evaluation splits must be disjoint")


@dataclass
class StageBLosses:
    total: Tensor
    task: Tensor
    transport: Tensor
    risk: Tensor


def compose_stage_b_loss(
    *,
    counterfactual_task_loss: Tensor,
    transported: Tensor,
    dense_reference: Tensor,
    predicted_quantile: Tensor,
    regret_target: Tensor,
    transport_weight: float,
    risk_weight: float,
    quantile: float,
) -> StageBLosses:
    task = counterfactual_task_loss.float().mean()
    transport = transport_consistency_loss(transported, dense_reference.detach())
    risk = pinball_loss(predicted_quantile, regret_target.detach(), quantile=quantile)
    total = task + float(transport_weight) * transport + float(risk_weight) * risk
    return StageBLosses(total=total, task=task, transport=transport, risk=risk)


def checkpoint_readiness_metadata(
    *,
    stage: str,
    calibration_ready: bool,
    measured_cost_ready: bool,
    split_hashes: Mapping[str, str],
) -> dict:
    stage = str(stage)
    if stage not in {"B", "C"}:
        raise ValueError("checkpoint stage must be B or C")
    return {
        "chronotransport_stage": stage,
        "calibration_ready": bool(calibration_ready),
        "measured_cost_ready": bool(measured_cost_ready),
        "split_hashes": dict(split_hashes),
        "deploy_claim_allowed": False,
        "metric_claim_allowed": False,
        "latency_claim_allowed": False,
        "paper_claim_allowed": False,
    }
