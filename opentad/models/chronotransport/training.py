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


def configure_stage_c(model: nn.Module) -> tuple[str, ...]:
    return _set_trainable(
        model,
        lambda name: (
            ("adapter" in name and "chronotransport" not in name)
            or "chronotransport.transport" in name
            or "chronotransport.risk_predictor" in name
        ),
    )


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
