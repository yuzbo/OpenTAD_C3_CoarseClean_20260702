from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from opentad.models.selectors.duca_utility_geometry_targets import GeometryValueTarget


@dataclass
class DucaValueLossBundle:
    geometry_value_loss: torch.Tensor
    self_ema_value_distill_loss: torch.Tensor
    gated_portal_value_loss: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


def _standardize(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    denom = valid.to(values.dtype).sum(dim=1).clamp_min(1.0)
    mean = values.masked_fill(~valid, 0.0).sum(dim=1) / denom
    centered = (values - mean[:, None]).masked_fill(~valid, 0.0)
    var = (centered * centered).sum(dim=1) / denom
    return centered / var.clamp_min(1.0e-6).sqrt()[:, None]


def geometry_value_loss(value: torch.Tensor, target: GeometryValueTarget, valid: torch.Tensor) -> torch.Tensor:
    if value.shape != target.frame_target.shape or value.shape != target.pair_weight.shape:
        raise ValueError("value and geometry target shapes must match")
    residual = F.smooth_l1_loss(value, target.frame_target, reduction="none")
    weighted = residual * target.pair_weight
    num = weighted.masked_fill(~valid, 0.0).sum()
    den = valid.to(weighted.dtype).sum().clamp_min(1.0)
    return num / den


def self_ema_value_distill_loss(value: torch.Tensor, ema_target: torch.Tensor, valid: torch.Tensor, *, lambda_scale: float = 1.0) -> torch.Tensor:
    if value.shape != ema_target.shape:
        raise ValueError("value and EMA target shapes must match")
    z_value = _standardize(value, valid)
    z_target = _standardize(ema_target, valid)
    den = valid.to(value.dtype).sum().clamp_min(1.0)
    rank_corr = (z_value * z_target).masked_fill(~valid, 0.0).sum() / den
    rank_loss = 1.0 - rank_corr
    scale_loss = F.smooth_l1_loss(z_value.masked_select(valid), z_target.masked_select(valid), reduction="mean")
    return rank_loss + float(lambda_scale) * scale_loss


def gated_portal_value_loss(value: torch.Tensor, *, enabled: bool = False) -> torch.Tensor:
    return value.sum() * 0.0


def build_value_learning_losses(
    *,
    value: torch.Tensor,
    valid: torch.Tensor,
    geometry_target: GeometryValueTarget | None,
    ema_target: torch.Tensor | None,
    geometry_weight: float,
    ema_weight: float,
    portal_enabled: bool,
) -> DucaValueLossBundle:
    zero = value.sum() * 0.0
    geo_loss = geometry_value_loss(value, geometry_target, valid) * float(geometry_weight) if geometry_target is not None and float(geometry_weight) > 0.0 else zero
    ema_loss = self_ema_value_distill_loss(value, ema_target, valid) * float(ema_weight) if ema_target is not None and float(ema_weight) > 0.0 else zero
    portal_loss = gated_portal_value_loss(value, enabled=portal_enabled)
    metadata = {
        "geometry_weight": float(geometry_weight),
        "ema_weight": float(ema_weight),
        "portal_enabled": bool(portal_enabled),
        "uses_dense_detector_teacher": False,
        "uses_self_ema_teacher": ema_target is not None,
        "uses_gt_geometry_target": geometry_target is not None,
        "uses_detector_feedback_at_inference": False,
        "uses_raw_prediction_cache": False,
    }
    return DucaValueLossBundle(
        geometry_value_loss=geo_loss,
        self_ema_value_distill_loss=ema_loss,
        gated_portal_value_loss=portal_loss,
        metadata=metadata,
    )
