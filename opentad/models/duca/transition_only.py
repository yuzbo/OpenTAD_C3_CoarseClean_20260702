from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


ASFORMER_ENCODER_HIDDEN_KIND = "official_asformer_encoder_hidden"


def balanced_binary_actionness_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    max_positive_weight: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.shape != target.shape:
        raise ValueError("actionness logits and target must have identical [B,T] shape")
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    if valid.shape != logits.shape:
        raise ValueError("valid_mask must align with actionness logits")
    target = target.to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
    positive = (target * valid.to(dtype=target.dtype)).sum()
    negative = ((1.0 - target) * valid.to(dtype=target.dtype)).sum()
    ratio = torch.where(positive > 0.0, negative / positive.clamp_min(1.0), negative.new_ones(()))
    positive_weight = ratio.clamp(1.0, float(max_positive_weight)).detach()
    per_element = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=positive_weight,
    ).masked_fill(~valid, 0.0)
    denominator = valid.to(dtype=logits.dtype).sum().clamp_min(1.0)
    return per_element.sum() / denominator, positive_weight


def _validate_temporal_inputs(
    actionness_logits: torch.Tensor,
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if actionness_logits.ndim != 2:
        raise ValueError("actionness_logits must be [B,T]")
    if hidden.ndim != 3 or hidden.shape[:2] != actionness_logits.shape:
        raise ValueError("hidden must be [B,T,D] and align with actionness_logits")
    valid = valid_mask.to(device=actionness_logits.device, dtype=torch.bool)
    if valid.shape != actionness_logits.shape:
        raise ValueError("valid_mask must align with actionness_logits [B,T]")
    if not actionness_logits.is_floating_point() or not hidden.is_floating_point():
        raise ValueError("actionness_logits and hidden must be floating-point tensors")
    return valid


def build_transition_descriptors(
    actionness_logits: torch.Tensor,
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Build relational temporal evidence without exposing absolute hidden states."""

    valid = _validate_temporal_inputs(actionness_logits, hidden, valid_mask)
    logits = actionness_logits.float()
    state = hidden.float()
    probability = torch.sigmoid(logits)
    eps = torch.finfo(probability.dtype).eps
    entropy = -(
        probability.clamp(eps, 1.0 - eps) * torch.log(probability.clamp(eps, 1.0 - eps))
        + (1.0 - probability).clamp(eps, 1.0 - eps)
        * torch.log((1.0 - probability).clamp(eps, 1.0 - eps))
    )

    delta_logits = torch.diff(logits, dim=1, prepend=logits[:, :1])
    delta_entropy = torch.diff(entropy, dim=1, prepend=entropy[:, :1])
    delta_hidden = torch.diff(state, dim=1, prepend=state[:, :1])
    cosine_change = logits.new_zeros(logits.shape)
    if logits.shape[1] > 1:
        cosine_change[:, 1:] = 1.0 - F.cosine_similarity(
            state[:, 1:],
            state[:, :-1],
            dim=-1,
            eps=1e-6,
        )

    transition_valid = valid.clone()
    transition_valid[:, 0] = False
    if valid.shape[1] > 1:
        transition_valid[:, 1:] &= valid[:, :-1]
    descriptors = torch.cat(
        (
            delta_logits[:, :, None],
            delta_logits.abs()[:, :, None],
            delta_entropy[:, :, None],
            delta_entropy.abs()[:, :, None],
            delta_hidden,
            delta_hidden.abs(),
            cosine_change[:, :, None],
        ),
        dim=-1,
    )
    return descriptors.masked_fill(~transition_valid[:, :, None], 0.0).to(dtype=hidden.dtype)


class DucaTransitionUtilityScorer(nn.Module):
    """One shared utility head for auxiliary transition and policy routes."""

    def __init__(self, hidden_dim: int, scorer_hidden_dim: int) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        scorer_hidden_dim = int(scorer_hidden_dim)
        if hidden_dim <= 0 or scorer_hidden_dim <= 0:
            raise ValueError("hidden_dim and scorer_hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.input_dim = 2 * hidden_dim + 5
        self.scorer_hidden_dim = scorer_hidden_dim
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, scorer_hidden_dim),
            nn.GELU(),
            nn.Linear(scorer_hidden_dim, 1),
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        if descriptors.ndim != 3 or int(descriptors.shape[-1]) != self.input_dim:
            raise ValueError(f"transition descriptors must be [B,T,{self.input_dim}]")
        return self.net(descriptors).squeeze(-1)


def transition_utility_paths(
    scorer: DucaTransitionUtilityScorer,
    actionness_logits: torch.Tensor,
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    compute_auxiliary: bool = True,
) -> Dict[str, torch.Tensor | str]:
    """Create equal-valued routes with intentionally different gradient ownership."""

    valid = _validate_temporal_inputs(actionness_logits, hidden, valid_mask)
    descriptors = build_transition_descriptors(actionness_logits.detach(), hidden, valid)
    policy_scores = scorer(descriptors.detach()).masked_fill(~valid, 0.0)
    auxiliary_scores = (
        scorer(descriptors).masked_fill(~valid, 0.0)
        if bool(compute_auxiliary)
        else policy_scores
    )
    if not torch.equal(auxiliary_scores.detach(), policy_scores.detach()):
        raise RuntimeError("shared transition scorer routes must be numerically identical")
    return {
        "transition_descriptors": descriptors,
        "auxiliary_scores": auxiliary_scores,
        "policy_scores": policy_scores,
        "hidden_kind": ASFORMER_ENCODER_HIDDEN_KIND,
    }


def _normalize_valid_scores(scores: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.to(device=scores.device, dtype=torch.bool)
    count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=scores.dtype)
    masked = scores.masked_fill(~valid, 0.0)
    mean = masked.sum(dim=1, keepdim=True) / count
    centered = (scores - mean).masked_fill(~valid, 0.0)
    variance = centered.square().sum(dim=1, keepdim=True) / count
    normalized = centered / variance.clamp_min(1e-6).sqrt()
    return normalized.masked_fill(~valid, 0.0)


def _uniform_reference_scores(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    valid = valid_mask.to(device=scores.device, dtype=torch.bool)
    reference = scores.new_zeros(scores.shape)
    for batch_idx in range(int(scores.shape[0])):
        valid_positions = torch.nonzero(valid[batch_idx], as_tuple=False).flatten()
        valid_count = int(valid_positions.numel())
        effective_k = min(max(int(k), 0), valid_count)
        if effective_k <= 0:
            continue
        positions = torch.arange(valid_count, device=scores.device, dtype=scores.dtype)
        targets = (
            (torch.arange(effective_k, device=scores.device, dtype=scores.dtype) + 0.5)
            * float(valid_count)
            / float(effective_k)
            - 0.5
        )
        values = -(positions[:, None] - targets[None, :]).abs().min(dim=1).values
        reference[batch_idx, valid_positions] = values
    return reference


def continuous_policy_logits(
    learned_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
    alpha: float,
) -> torch.Tensor:
    """Interpolate continuously between feasible uniform and learned utilities."""

    if learned_scores.ndim != 2:
        raise ValueError("learned_scores must be [B,T]")
    valid = valid_mask.to(device=learned_scores.device, dtype=torch.bool)
    if valid.shape != learned_scores.shape:
        raise ValueError("valid_mask must align with learned_scores")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("policy alpha must lie in [0,1]")
    learned = _normalize_valid_scores(learned_scores, valid)
    reference = _normalize_valid_scores(_uniform_reference_scores(learned_scores, valid, int(k)), valid)
    mixed = (1.0 - alpha) * reference + alpha * learned
    return mixed.masked_fill(~valid, 0.0)


def transition_distribution_loss(
    transition_scores: torch.Tensor,
    transition_target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    temperature: float = 0.7,
) -> torch.Tensor:
    if transition_scores.shape != transition_target.shape:
        raise ValueError("transition_scores and transition_target must have identical [B,T] shape")
    valid = valid_mask.to(device=transition_scores.device, dtype=torch.bool)
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("transition distribution temperature must be finite and positive")
    target = transition_target.to(device=transition_scores.device, dtype=transition_scores.dtype)
    target = target.masked_fill(~valid, 0.0)
    mass = target.sum(dim=1, keepdim=True)
    normalized = torch.where(mass > 0.0, target / mass.clamp_min(1e-8), target)
    logits = (transition_scores / temperature).masked_fill(~valid, -1.0e4)
    per_batch = -(normalized * F.log_softmax(logits, dim=1)).sum(dim=1)
    active = mass.squeeze(1) > 0.0
    if not bool(active.any().item()):
        return transition_scores.sum() * 0.0
    return per_batch[active].mean()


def local_boundary_coverage_loss(
    soft_occupancy: torch.Tensor,
    boundary_target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    radius: int,
) -> torch.Tensor:
    if soft_occupancy.shape != boundary_target.shape:
        raise ValueError("soft_occupancy and boundary_target must have identical [B,T] shape")
    valid = valid_mask.to(device=soft_occupancy.device, dtype=torch.bool)
    if valid.shape != soft_occupancy.shape:
        raise ValueError("valid_mask must align with soft_occupancy")
    radius = int(radius)
    if radius < 0:
        raise ValueError("boundary coverage radius must be non-negative")
    with torch.cuda.amp.autocast(enabled=False):
        occupancy = soft_occupancy.float().clamp(0.0, 1.0).masked_fill(~valid, 0.0)
        kernel = occupancy.new_ones((1, 1, 2 * radius + 1))
        local_mass = F.conv1d(occupancy[:, None, :], kernel, padding=radius).squeeze(1).clamp_min(1e-8)
        target = boundary_target.to(device=occupancy.device, dtype=occupancy.dtype).masked_fill(~valid, 0.0)
        target_mass = target.sum(dim=1, keepdim=True)
        normalized = torch.where(target_mass > 0.0, target / target_mass.clamp_min(1e-8), target)
        per_batch = -(normalized * torch.log(local_mass)).sum(dim=1)
        active = target_mass.squeeze(1) > 0.0
        if not bool(active.any().item()):
            return soft_occupancy.sum() * 0.0
        loss = per_batch[active].mean()
    return loss.to(dtype=soft_occupancy.dtype)


__all__ = [
    "ASFORMER_ENCODER_HIDDEN_KIND",
    "DucaTransitionUtilityScorer",
    "balanced_binary_actionness_loss",
    "build_transition_descriptors",
    "continuous_policy_logits",
    "local_boundary_coverage_loss",
    "transition_distribution_loss",
    "transition_utility_paths",
]
