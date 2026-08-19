from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass
class FoveaLossBundle:
    mask_loss: torch.Tensor
    coarse_loss: torch.Tensor
    cycle_loss: torch.Tensor
    budget_loss: torch.Tensor
    diversity_loss: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


def _segments_as_list(gt_segments):
    if torch.is_tensor(gt_segments):
        return [row for row in gt_segments]
    return list(gt_segments)


def build_gt_geometry_mask(
    *,
    gt_segments,
    valid: torch.Tensor,
    boundary_radius: int,
) -> torch.Tensor:
    if valid.ndim != 2:
        raise ValueError("valid must be [B,T]")
    batch, length = valid.shape
    device = valid.device
    mask = torch.zeros((batch, length), dtype=torch.float32, device=device)
    idx = torch.arange(length, device=device, dtype=torch.float32)
    for b, segments in enumerate(_segments_as_list(gt_segments)):
        if segments is None:
            continue
        seg = torch.as_tensor(segments, device=device, dtype=torch.float32).reshape(-1, 2)
        for start, end in seg:
            inside = (idx >= start) & (idx <= end)
            dist = torch.minimum((idx - start).abs(), (idx - end).abs())
            endpoint = inside & (dist <= float(boundary_radius))
            mask[b, inside] = torch.maximum(mask[b, inside], torch.full_like(mask[b, inside], 1.0))
            mask[b, endpoint] = torch.maximum(mask[b, endpoint], torch.full_like(mask[b, endpoint], 2.0))
    mask = mask.masked_fill(~valid.bool(), 0.0)
    return mask


def focal_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    v = valid.bool()
    prob = torch.sigmoid(logits)
    p = torch.where(target > 0.5, prob, 1.0 - prob)
    loss = -((1.0 - p) ** gamma) * torch.log(p.clamp_min(1.0e-7))
    return loss.masked_fill(~v, 0.0).sum() / v.to(loss.dtype).sum().clamp_min(1.0)


def build_fovea_losses(
    *,
    contribution: torch.Tensor,
    frame_score: torch.Tensor,
    coarse_logits: torch.Tensor,
    coarse_center: torch.Tensor,
    coarse_width: torch.Tensor,
    valid: torch.Tensor,
    gt_segments,
    boundary_radius: int,
    cycle_mask: torch.Tensor | None = None,
    budget_target: int,
    selected_count: torch.Tensor,
    weights: Mapping[str, float] | None = None,
) -> FoveaLossBundle:
    if contribution.ndim != 3 or frame_score.ndim != 2 or valid.shape != frame_score.shape:
        raise ValueError("fovea losses expect A [B,M,T], score [B,T], valid [B,T]")
    cfg = {
        "mask": 1.0,
        "coarse": 1.0,
        "cycle": 0.0,
        "budget": 0.05,
        "diversity": 0.05,
    }
    if weights:
        cfg.update({str(k): float(v) for k, v in weights.items()})
    zero = frame_score.new_zeros(())
    # Dense geometry mask: interior 1, endpoint neighborhood 2, background 0.
    gt_mask = build_gt_geometry_mask(gt_segments=gt_segments, valid=valid, boundary_radius=boundary_radius)
    agg_contribution = contribution.mean(dim=1).masked_fill(~valid.bool(), 0.0)
    mask_loss = focal_bce(agg_contribution, (gt_mask > 0.5).float(), valid) * float(cfg["mask"])
    # Coarse proposal supervision.
    action_target = (gt_mask > 0.5).float()
    cls_loss = focal_bce(coarse_logits, action_target, valid)
    center_loss = F.smooth_l1_loss(
        coarse_center.masked_select(valid),
        torch.zeros((int(valid.bool().long().sum().item()),), dtype=coarse_center.dtype, device=coarse_center.device),
        reduction="mean",
    ) if int(valid.bool().long().sum().item()) else zero
    width_loss = coarse_width.masked_select(valid).mean() if int(valid.bool().long().sum().item()) else zero
    coarse_loss = (cls_loss + 0.5 * center_loss + 0.1 * width_loss) * float(cfg["coarse"])
    # Cycle feedback is train-only and detached. No teacher enters inference.
    if cycle_mask is not None and float(cfg["cycle"]) > 0.0:
        cycle_loss = focal_bce(frame_score, cycle_mask.detach(), valid) * float(cfg["cycle"])
    else:
        cycle_loss = zero
    # Soft budget regularizer: keep expected selected count close to target.
    count = selected_count.to(frame_score.dtype).sum().clamp_min(1.0)
    budget_loss = ((count - float(budget_target)) / float(budget_target)) ** 2 * float(cfg["budget"])
    # Diversity: entropy over query-frame contribution plus query orthogonality.
    query_dist = torch.softmax(contribution.flatten(1), dim=-1)
    entropy = -(query_dist * (query_dist + 1.0e-8).log()).sum(dim=-1).mean()
    diversity_loss = entropy * float(cfg["diversity"])
    metadata = {
        "loss_names": ("mask", "coarse", "cycle", "budget", "diversity"),
        "weights": dict(cfg),
        "cycle_enabled": cycle_mask is not None and float(cfg["cycle"]) > 0.0,
        "uses_gt": True,
        "uses_teacher": False,
        "uses_raw_prediction_cache": False,
    }
    return FoveaLossBundle(
        mask_loss=mask_loss,
        coarse_loss=coarse_loss,
        cycle_loss=cycle_loss,
        budget_loss=budget_loss,
        diversity_loss=diversity_loss,
        metadata=metadata,
    )
