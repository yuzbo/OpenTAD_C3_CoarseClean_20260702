from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


def _as_bct(feature: torch.Tensor, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(feature):
        raise TypeError(f"{name} must be a tensor")
    if feature.ndim != 3:
        raise ValueError(f"{name} must be [B,C,T] or [B,T,C], got {tuple(feature.shape)}")
    if feature.shape[1] <= feature.shape[2]:
        return feature
    return feature.transpose(1, 2).contiguous()


def merge_p0_p1_feature_tokens(
    feature_levels: Sequence[torch.Tensor],
    *,
    weights: Sequence[float] = (0.5, 0.5),
) -> torch.Tensor:
    if len(feature_levels) < 2:
        raise ValueError("signed Taylor attribution requires fixed P0 and P1 feature levels")
    p0 = _as_bct(feature_levels[0], name="P0")
    p1 = _as_bct(feature_levels[1], name="P1")
    if p0.shape[0] != p1.shape[0] or p0.shape[1] != p1.shape[1]:
        raise ValueError("P0 and P1 must share batch and channel dimensions")
    p1_up = F.interpolate(p1.float(), size=int(p0.shape[-1]), mode="linear", align_corners=False)
    w0, w1 = float(weights[0]), float(weights[1])
    denom = max(abs(w0) + abs(w1), 1.0e-6)
    return (w0 * p0.float() + w1 * p1_up) / denom


def signed_feature_taylor_target(
    detector_objective: torch.Tensor,
    feature_tokens: torch.Tensor,
    *,
    retain_graph: bool = True,
) -> torch.Tensor:
    if detector_objective.ndim != 0:
        detector_objective = detector_objective.float().sum()
    tokens = _as_bct(feature_tokens, name="feature_tokens")
    grad = torch.autograd.grad(
        detector_objective.float(),
        tokens,
        retain_graph=bool(retain_graph),
        create_graph=False,
        allow_unused=False,
    )[0]
    with torch.no_grad():
        target = -(
            grad.detach().float() * tokens.detach().float()
        ).sum(dim=1)
        target = target.relu()
    return target


def signed_feature_taylor_target_from_levels(
    detector_objective: torch.Tensor,
    feature_levels: Sequence[torch.Tensor],
    *,
    retain_graph: bool = True,
    weights: Sequence[float] = (0.5, 0.5),
) -> torch.Tensor:
    tokens = merge_p0_p1_feature_tokens(feature_levels, weights=weights)
    return signed_feature_taylor_target(detector_objective, tokens, retain_graph=retain_graph)


def listwise_distribution_loss(
    scores: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if scores.shape != target.shape:
        raise ValueError("scores and target must have the same shape")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if valid_mask is None:
        valid = torch.ones_like(scores, dtype=torch.bool)
    else:
        if valid_mask.shape != scores.shape:
            raise ValueError("valid_mask must match scores")
        valid = valid_mask.to(device=scores.device, dtype=torch.bool)
    mask_value = torch.finfo(scores.dtype).min / 4.0
    score_logp = F.log_softmax(scores.masked_fill(~valid, mask_value) / float(temperature), dim=1)
    target_positive = target.detach().float().masked_fill(~valid, 0.0)
    target_dist = target_positive / target_positive.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    loss = -(target_dist * score_logp).sum(dim=1)
    active = valid.long().sum(dim=1) > 0
    if not bool(active.any().item()):
        return scores.new_zeros(())
    return loss[active].mean()


def pairwise_ranking_loss(
    scores: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    margin: float = 0.05,
) -> torch.Tensor:
    if scores.shape != target.shape:
        raise ValueError("scores and target must have the same shape")
    if valid_mask is None:
        valid = torch.ones_like(scores, dtype=torch.bool)
    else:
        if valid_mask.shape != scores.shape:
            raise ValueError("valid_mask must match scores")
        valid = valid_mask.to(device=scores.device, dtype=torch.bool)
    losses = []
    for score_row, target_row, valid_row in zip(scores, target.detach(), valid):
        idx = torch.nonzero(valid_row, as_tuple=False).flatten()
        if int(idx.numel()) < 2:
            continue
        s = score_row[idx]
        t = target_row[idx]
        order = t[:, None] - t[None, :]
        pair_mask = order > 0
        if not bool(pair_mask.any().item()):
            continue
        diff = s[:, None] - s[None, :]
        losses.append(F.relu(float(margin) - diff[pair_mask]).mean())
    if not losses:
        return scores.new_zeros(())
    return torch.stack(losses).mean()


def should_refresh_taylor_target(successful_update: int, *, period: int = 4) -> bool:
    period = int(period)
    if period <= 0:
        raise ValueError("period must be positive")
    return int(successful_update) % period == 0


def update_ema_target(
    previous: Optional[torch.Tensor],
    current: torch.Tensor,
    *,
    decay: float = 0.9,
) -> torch.Tensor:
    decay = float(decay)
    if decay < 0.0 or decay >= 1.0:
        raise ValueError("decay must lie in [0, 1)")
    if previous is None:
        return current.detach()
    if previous.shape != current.shape:
        raise ValueError("previous and current Taylor EMA targets must share shape")
    return (decay * previous.detach() + (1.0 - decay) * current.detach()).detach()


def taylor_attribution_contract() -> Dict[str, object]:
    return {
        "schema_version": "duca_signed_feature_taylor_contract_v1",
        "feature_levels": ["P0", "P1"],
        "formula": "relu(-(detached_gradient * detached_feature).sum(channel))",
        "create_graph": False,
        "retain_graph": True,
        "target_detached": True,
        "default_update_period_successful_steps": 4,
        "supervision": ["pairwise_ranking", "listwise_distribution"],
        "one_swap_split": "train_only",
    }
