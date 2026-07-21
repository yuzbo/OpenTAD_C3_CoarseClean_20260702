from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .structured_selection import exact_uniform_reference_scores, structured_local_coverage_probability


ASFORMER_ENCODER_HIDDEN_KIND = "official_asformer_encoder_hidden"


def balanced_binary_actionness_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    positive_prior: float = 0.5,
    max_positive_weight: float = 8.0,
    reduction_mode: str = "posterior_bce",
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.shape != target.shape:
        raise ValueError("actionness logits and target must have identical [B,T] shape")
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    if valid.shape != logits.shape:
        raise ValueError("valid_mask must align with actionness logits")
    work = logits.float()
    target = target.to(device=logits.device, dtype=torch.float32).clamp(0.0, 1.0)
    positive_prior = float(positive_prior)
    max_positive_weight = float(max_positive_weight)
    if not math.isfinite(positive_prior) or not 0.0 < positive_prior < 1.0:
        raise ValueError("positive_prior must be a fixed finite value in (0,1)")
    if not math.isfinite(max_positive_weight) or max_positive_weight < 1.0:
        raise ValueError("max_positive_weight must be finite and at least one")
    if reduction_mode not in {"posterior_bce", "class_balanced_mean"}:
        raise ValueError(
            "reduction_mode must be posterior_bce or class_balanced_mean"
        )
    prior_odds = (1.0 - positive_prior) / positive_prior
    positive_weight = work.new_tensor(min(max(prior_odds, 1.0), max_positive_weight))
    if reduction_mode == "class_balanced_mean":
        per_element = F.binary_cross_entropy_with_logits(
            work,
            target,
            reduction="none",
        )
        positive = valid & (target >= 0.5)
        negative = valid & ~positive
        class_terms = []
        class_weights = []
        if bool(positive.any().item()):
            class_terms.append(per_element[positive].mean())
            class_weights.append(positive_prior)
        if bool(negative.any().item()):
            class_terms.append(per_element[negative].mean())
            class_weights.append(1.0 - positive_prior)
        if not class_terms:
            return work.sum() * 0.0, positive_weight
        weight_total = sum(class_weights)
        loss = sum(
            term * (weight / weight_total)
            for term, weight in zip(class_terms, class_weights)
        )
        return loss, positive_weight
    per_element = F.binary_cross_entropy_with_logits(
        work,
        target,
        reduction="none",
        pos_weight=positive_weight,
    ).masked_fill(~valid, 0.0)
    denominator = valid.to(dtype=torch.float32).sum().clamp_min(1.0)
    return per_element.sum() / denominator, positive_weight


def calibrated_actionness_probability(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    bias: float = 0.0,
) -> torch.Tensor:
    """Map logits to probabilities with frozen train-only calibration parameters."""

    temperature = float(temperature)
    bias = float(bias)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("calibration temperature must be finite and positive")
    if not math.isfinite(bias):
        raise ValueError("calibration bias must be finite")
    return torch.sigmoid((logits.float() + bias) / temperature).to(dtype=logits.dtype)


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
    *,
    calibration_temperature: float = 1.0,
    calibration_bias: float = 0.0,
) -> torch.Tensor:
    """Build relational temporal evidence without exposing absolute hidden states."""

    valid = _validate_temporal_inputs(actionness_logits, hidden, valid_mask)
    logits = actionness_logits.float()
    state = hidden.float()
    probability = calibrated_actionness_probability(
        logits,
        temperature=calibration_temperature,
        bias=calibration_bias,
    )
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
    """Shared transition-center scorer with an optional local burst profile."""

    def __init__(
        self,
        hidden_dim: int,
        scorer_hidden_dim: int,
        *,
        zero_init_output: bool = False,
        burst_radius: int = 0,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        scorer_hidden_dim = int(scorer_hidden_dim)
        if hidden_dim <= 0 or scorer_hidden_dim <= 0:
            raise ValueError("hidden_dim and scorer_hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.input_dim = 2 * hidden_dim + 5
        self.scorer_hidden_dim = scorer_hidden_dim
        self.zero_init_output = bool(zero_init_output)
        self.burst_radius = int(burst_radius)
        if self.burst_radius < 0:
            raise ValueError("burst_radius must be non-negative")
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, scorer_hidden_dim),
            nn.GELU(),
            nn.Linear(scorer_hidden_dim, 1),
        )
        if self.zero_init_output:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        self.burst_offset_head = (
            nn.Linear(scorer_hidden_dim, 2 * self.burst_radius + 1)
            if self.burst_radius > 0
            else None
        )
        if self.burst_offset_head is not None:
            # A symmetric c-R...c+R profile is the stable Oracle-like start.
            nn.init.zeros_(self.burst_offset_head.weight)
            nn.init.zeros_(self.burst_offset_head.bias)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        if descriptors.ndim != 3 or int(descriptors.shape[-1]) != self.input_dim:
            raise ValueError(f"transition descriptors must be [B,T,{self.input_dim}]")
        features = self.net[:-1](descriptors)
        return self.net[-1](features).squeeze(-1)

    def forward_with_burst(
        self,
        descriptors: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if descriptors.ndim != 3 or int(descriptors.shape[-1]) != self.input_dim:
            raise ValueError(f"transition descriptors must be [B,T,{self.input_dim}]")
        features = self.net[:-1](descriptors)
        center_scores = self.net[-1](features).squeeze(-1)
        offset_logits = (
            None
            if self.burst_offset_head is None
            else self.burst_offset_head(features)
        )
        return center_scores, offset_logits


class DucaProtectedTransitionScorer(nn.Module):
    """Explicit selector parameter boundary for the protected-E2E route."""

    def __init__(
        self,
        hidden_dim: int = 96,
        scorer_hidden_dim: int = 64,
        *,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        scorer_hidden_dim = int(scorer_hidden_dim)
        if hidden_dim <= 0 or scorer_hidden_dim <= 0:
            raise ValueError("hidden_dim and scorer_hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.input_dim = 2 * hidden_dim + 5
        self.scorer_hidden_dim = scorer_hidden_dim
        self.zero_init_output = bool(zero_init_output)
        self.selector_adapter = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, scorer_hidden_dim),
            nn.GELU(),
        )
        self.selector_score_head = nn.Linear(scorer_hidden_dim, 1)
        if self.zero_init_output:
            nn.init.zeros_(self.selector_score_head.weight)
            nn.init.zeros_(self.selector_score_head.bias)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        if descriptors.ndim != 3 or int(descriptors.shape[-1]) != self.input_dim:
            raise ValueError(f"transition descriptors must be [B,T,{self.input_dim}]")
        return self.selector_score_head(self.selector_adapter(descriptors)).squeeze(-1)


def coverage_floor_distribution(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    floor_weight: float = 0.10,
    score_temperature: float = 0.70,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix learned temporal mass with a fixed uniform coverage floor."""

    if scores.ndim != 2 or not scores.is_floating_point():
        raise ValueError("scores must be a floating-point [B,T] tensor")
    valid = valid_mask.to(device=scores.device, dtype=torch.bool)
    if valid.shape != scores.shape:
        raise ValueError("valid_mask must align with scores")
    if bool(torch.any(valid.sum(dim=1) == 0).item()):
        raise ValueError("coverage-floor distribution requires one valid point per row")
    floor_weight = float(floor_weight)
    score_temperature = float(score_temperature)
    if not math.isfinite(floor_weight) or not 0.0 <= floor_weight < 1.0:
        raise ValueError("floor_weight must lie in [0,1)")
    if not math.isfinite(score_temperature) or score_temperature <= 0.0:
        raise ValueError("score_temperature must be finite and positive")

    work = scores.float()
    learned = F.softmax(
        (work / score_temperature).masked_fill(~valid, float("-inf")),
        dim=1,
    ).masked_fill(~valid, 0.0)
    valid_count = valid.sum(dim=1, keepdim=True).to(dtype=work.dtype)
    uniform = valid.to(dtype=work.dtype) / valid_count
    probabilities = (
        (1.0 - floor_weight) * learned + floor_weight * uniform
    ).masked_fill(~valid, 0.0)
    row_sum = probabilities.sum(dim=1)
    if not torch.allclose(
        row_sum,
        torch.ones_like(row_sum),
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise RuntimeError("coverage-floor probabilities must sum to one")
    log_probabilities = torch.where(
        valid,
        probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log(),
        probabilities.new_full((), float("-inf")),
    )
    return probabilities.to(dtype=scores.dtype), log_probabilities.to(dtype=scores.dtype)


def transition_utility_paths(
    scorer: nn.Module,
    actionness_logits: torch.Tensor,
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    compute_auxiliary: bool = True,
    policy_hidden: Optional[torch.Tensor] = None,
    policy_hidden_gradient_scale: float = 0.0,
    auxiliary_hidden_gradient_scale: float = 1.0,
    calibration_temperature: float = 1.0,
    calibration_bias: float = 0.0,
) -> Dict[str, torch.Tensor | str]:
    """Create equal-valued routes with intentionally different gradient ownership."""

    policy_hidden_gradient_scale = float(policy_hidden_gradient_scale)
    if not math.isfinite(policy_hidden_gradient_scale) or not 0.0 <= policy_hidden_gradient_scale <= 1.0:
        raise ValueError("policy_hidden_gradient_scale must lie in [0,1]")
    auxiliary_hidden_gradient_scale = float(auxiliary_hidden_gradient_scale)
    if (
        not math.isfinite(auxiliary_hidden_gradient_scale)
        or not 0.0 <= auxiliary_hidden_gradient_scale <= 1.0
    ):
        raise ValueError("auxiliary_hidden_gradient_scale must lie in [0,1]")
    valid = _validate_temporal_inputs(actionness_logits, hidden, valid_mask)
    descriptors = build_transition_descriptors(
        actionness_logits.detach(),
        hidden,
        valid,
        calibration_temperature=calibration_temperature,
        calibration_bias=calibration_bias,
    )
    policy_descriptor_source = descriptors
    if policy_hidden is not None:
        policy_descriptor_source = build_transition_descriptors(
            actionness_logits.detach(),
            policy_hidden,
            valid,
            calibration_temperature=calibration_temperature,
            calibration_bias=calibration_bias,
        )
        if not torch.allclose(
            policy_descriptor_source.detach(),
            descriptors.detach(),
            atol=1.0e-5,
            rtol=1.0e-5,
        ):
            raise RuntimeError(
                "policy-specific hidden route must be numerically identical to the shared ASFormer hidden"
            )
    elif policy_hidden_gradient_scale > 0.0:
        raise ValueError(
            "positive policy_hidden_gradient_scale requires a restricted policy_hidden route"
        )
    policy_descriptors = policy_descriptor_source.detach() + policy_hidden_gradient_scale * (
        policy_descriptor_source - policy_descriptor_source.detach()
    )
    if hasattr(scorer, "forward_with_burst"):
        policy_scores, policy_offset_logits = scorer.forward_with_burst(
            policy_descriptors
        )
    else:
        policy_scores = scorer(policy_descriptors)
        policy_offset_logits = None
    policy_scores = policy_scores.masked_fill(~valid, 0.0)
    if policy_offset_logits is not None:
        policy_offset_logits = policy_offset_logits.masked_fill(
            ~valid[:, :, None],
            0.0,
        )
    auxiliary_descriptors = descriptors.detach() + auxiliary_hidden_gradient_scale * (
        descriptors - descriptors.detach()
    )
    if bool(compute_auxiliary):
        if hasattr(scorer, "forward_with_burst"):
            auxiliary_scores, auxiliary_offset_logits = scorer.forward_with_burst(
                auxiliary_descriptors
            )
        else:
            auxiliary_scores = scorer(auxiliary_descriptors)
            auxiliary_offset_logits = None
        auxiliary_scores = auxiliary_scores.masked_fill(~valid, 0.0)
        if auxiliary_offset_logits is not None:
            auxiliary_offset_logits = auxiliary_offset_logits.masked_fill(
                ~valid[:, :, None],
                0.0,
            )
    else:
        auxiliary_scores = policy_scores
        auxiliary_offset_logits = policy_offset_logits
    if not torch.equal(auxiliary_scores.detach(), policy_scores.detach()):
        raise RuntimeError("shared transition scorer routes must be numerically identical")
    output = {
        "transition_descriptors": descriptors,
        "policy_descriptors": policy_descriptors,
        "auxiliary_descriptors": auxiliary_descriptors,
        "auxiliary_scores": auxiliary_scores,
        "policy_scores": policy_scores,
        "policy_hidden_gradient_scale": policy_hidden_gradient_scale,
        "auxiliary_hidden_gradient_scale": auxiliary_hidden_gradient_scale,
        "hidden_kind": ASFORMER_ENCODER_HIDDEN_KIND,
    }
    if policy_offset_logits is not None:
        if auxiliary_offset_logits is None or not torch.equal(
            auxiliary_offset_logits.detach(),
            policy_offset_logits.detach(),
        ):
            raise RuntimeError("shared burst-offset routes must be numerically identical")
        output.update(
            {
                "policy_offset_logits": policy_offset_logits,
                "auxiliary_offset_logits": auxiliary_offset_logits,
            }
        )
    return output


def _normalize_valid_scores(scores: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.to(device=scores.device, dtype=torch.bool)
    count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=scores.dtype)
    masked = scores.masked_fill(~valid, 0.0)
    mean = masked.sum(dim=1, keepdim=True) / count
    centered = (scores - mean).masked_fill(~valid, 0.0)
    variance = centered.square().sum(dim=1, keepdim=True) / count
    normalized = centered / variance.clamp_min(1e-6).sqrt()
    return normalized.masked_fill(~valid, 0.0)


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
    reference = _normalize_valid_scores(exact_uniform_reference_scores(learned_scores, valid, int(k)), valid)
    mixed = (1.0 - alpha) * reference + alpha * learned
    return mixed.masked_fill(~valid, 0.0)


def build_boundary_burst_utility(
    center_scores: torch.Tensor,
    offset_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
    radius: int,
    quota: float,
    boundary_budget_fraction: float = 0.25,
    context_weight: float = 0.05,
    center_temperature: float = 0.7,
    offset_temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Spread learned transition centers into saturating bilateral micro-clusters."""

    if center_scores.ndim != 2 or not center_scores.is_floating_point():
        raise ValueError("center_scores must be a floating-point [B,T] tensor")
    radius = int(radius)
    k = int(k)
    quota = float(quota)
    quota_int = int(round(quota))
    boundary_budget_fraction = float(boundary_budget_fraction)
    context_weight = float(context_weight)
    center_temperature = float(center_temperature)
    offset_temperature = float(offset_temperature)
    if radius <= 0:
        raise ValueError("boundary burst radius must be positive")
    if offset_logits.shape != (*center_scores.shape, 2 * radius + 1):
        raise ValueError("offset_logits must be [B,T,2*radius+1]")
    valid = valid_mask.to(device=center_scores.device, dtype=torch.bool)
    if valid.shape != center_scores.shape:
        raise ValueError("valid_mask must align with center_scores")
    if bool(torch.any(valid.sum(dim=1) == 0).item()):
        raise ValueError("boundary burst utility requires one valid point per row")
    if k <= 0 or quota <= 0.0:
        raise ValueError("boundary burst k and quota must be positive")
    if abs(quota - float(quota_int)) > 1.0e-6 or quota_int > 2 * radius + 1:
        raise ValueError("boundary burst quota must be an integer within its radius")
    if not 0.0 < boundary_budget_fraction <= 1.0:
        raise ValueError("boundary_budget_fraction must lie in (0,1]")
    if context_weight < 0.0 or not math.isfinite(context_weight):
        raise ValueError("context_weight must be finite and non-negative")
    if center_temperature <= 0.0 or offset_temperature <= 0.0:
        raise ValueError("boundary burst temperatures must be positive")

    work = center_scores.float()
    offsets = offset_logits.float()
    center_probabilities = F.softmax(
        (work / center_temperature).masked_fill(~valid, float("-inf")),
        dim=1,
    ).masked_fill(~valid, 0.0)

    batch, temporal_len = work.shape
    offset_valid = torch.zeros_like(offsets, dtype=torch.bool)
    for offset_idx, delta in enumerate(range(-radius, radius + 1)):
        if delta < 0:
            offset_valid[:, -delta:, offset_idx] = valid[:, : temporal_len + delta]
        elif delta > 0:
            offset_valid[:, : temporal_len - delta, offset_idx] = valid[:, delta:]
        else:
            offset_valid[:, :, offset_idx] = valid
    offset_axis = torch.arange(
        -radius,
        radius + 1,
        device=offsets.device,
        dtype=offsets.dtype,
    )
    # Break an all-zero initialization toward a centered bilateral profile.
    # Learned evidence easily dominates this deterministic tie break.
    offset_scores = offsets / offset_temperature - 1.0e-4 * offset_axis.abs()
    masked_offset_scores = offset_scores.masked_fill(~offset_valid, -1.0e4)
    offset_probabilities = F.softmax(masked_offset_scores, dim=-1).masked_fill(
        ~offset_valid,
        0.0,
    )
    effective_quota = offset_valid.sum(dim=-1).clamp(max=quota_int)
    soft_offset_inclusion = offset_probabilities * effective_quota[:, :, None].to(
        offset_probabilities.dtype
    )
    hard_offset_inclusion = torch.zeros_like(offset_probabilities)
    flat_scores = masked_offset_scores.reshape(-1, masked_offset_scores.shape[-1])
    flat_hard = hard_offset_inclusion.reshape(-1, hard_offset_inclusion.shape[-1])
    flat_quota = effective_quota.reshape(-1)
    for row_quota in range(1, quota_int + 1):
        rows = torch.nonzero(flat_quota == row_quota, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        indices = torch.topk(
            flat_scores[rows],
            k=row_quota,
            dim=-1,
            sorted=False,
        ).indices
        flat_hard[rows[:, None], indices] = 1.0
    hard_offset_inclusion = flat_hard.reshape_as(hard_offset_inclusion)
    offset_inclusion = (
        hard_offset_inclusion.detach()
        + soft_offset_inclusion
        - soft_offset_inclusion.detach()
    ).masked_fill(~offset_valid, 0.0)

    expected_centers = max(
        float(k) * boundary_budget_fraction / quota,
        1.0,
    )
    center_mass = center_probabilities * expected_centers
    burst_mass = work.new_zeros((batch, temporal_len))
    for offset_idx, delta in enumerate(range(-radius, radius + 1)):
        contribution = center_mass * offset_inclusion[:, :, offset_idx]
        if delta < 0:
            burst_mass[:, : temporal_len + delta] += contribution[:, -delta:]
        elif delta > 0:
            burst_mass[:, delta:] += contribution[:, : temporal_len - delta]
        else:
            burst_mass += contribution
    burst_mass = burst_mass.masked_fill(~valid, 0.0)
    burst_utility = (1.0 - torch.exp(-burst_mass)).masked_fill(~valid, 0.0)
    context_reference = exact_uniform_reference_scores(work, valid, k)
    policy_utility = _normalize_valid_scores(burst_utility, valid)
    if context_weight > 0.0:
        policy_utility = policy_utility + context_weight * _normalize_valid_scores(
            context_reference,
            valid,
        )
    return {
        "policy_utility": policy_utility.masked_fill(~valid, 0.0).to(center_scores.dtype),
        "burst_mass": burst_mass.to(center_scores.dtype),
        "burst_utility": burst_utility.to(center_scores.dtype),
        "center_probabilities": center_probabilities.to(center_scores.dtype),
        "offset_probabilities": offset_probabilities.to(center_scores.dtype),
        "offset_inclusion": offset_inclusion.to(center_scores.dtype),
        "effective_offset_quota": effective_quota,
        "context_reference": context_reference.to(center_scores.dtype),
    }


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
    work = transition_scores.float()
    target = transition_target.to(device=transition_scores.device, dtype=torch.float32)
    target = target.masked_fill(~valid, 0.0)
    mass = target.sum(dim=1, keepdim=True)
    normalized = torch.where(mass > 0.0, target / mass.clamp_min(1e-8), target)
    logits = (work / temperature).masked_fill(~valid, -1.0e4)
    per_batch = -(normalized * F.log_softmax(logits, dim=1)).sum(dim=1)
    active = mass.squeeze(1) > 0.0
    if not bool(active.any().item()):
        return work.sum() * 0.0
    return per_batch[active].mean()


def local_boundary_coverage_loss(
    policy_logits: torch.Tensor,
    boundary_target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    radius: int,
    k: int,
    max_unselected_hole: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Negative log exact structured probability of covering GT boundaries."""

    if policy_logits.shape != boundary_target.shape:
        raise ValueError("policy_logits and boundary_target must have identical [B,T] shape")
    valid = valid_mask.to(device=policy_logits.device, dtype=torch.bool)
    if valid.shape != policy_logits.shape:
        raise ValueError("valid_mask must align with soft_occupancy")
    radius = int(radius)
    if radius < 0:
        raise ValueError("boundary coverage radius must be non-negative")
    with torch.cuda.amp.autocast(enabled=False):
        # This train-only auxiliary subtracts long-horizon log partitions.
        # Keep it in FP64 so AMP scaling cannot destabilize their gradients.
        logits = policy_logits.double()
        target = boundary_target.to(device=logits.device, dtype=logits.dtype).masked_fill(~valid, 0.0)
        per_batch = []
        for batch_idx in range(logits.shape[0]):
            valid_positions = torch.nonzero(valid[batch_idx], as_tuple=False).flatten()
            target_row = target[batch_idx, valid_positions]
            boundary_ranks = torch.nonzero(target_row > 0.0, as_tuple=False).flatten()
            if boundary_ranks.numel() == 0:
                continue
            row_logits = logits[batch_idx, valid_positions][None, :]
            ranks = torch.arange(valid_positions.numel(), device=logits.device)
            events = (ranks[None, :] - boundary_ranks[:, None]).abs() <= radius
            probabilities = structured_local_coverage_probability(
                row_logits,
                events[None, :, :],
                k=min(int(k), int(valid_positions.numel())),
                max_unselected_hole=int(max_unselected_hole),
                temperature=float(temperature),
            )[0].clamp_min(torch.finfo(logits.dtype).eps)
            weights = target_row[boundary_ranks]
            per_batch.append(-(weights * torch.log(probabilities)).sum() / weights.sum().clamp_min(1e-8))
        if not per_batch:
            return policy_logits.sum() * 0.0
        loss = torch.stack(per_batch).mean()
    return loss


def local_boundary_mass_coverage_loss(
    soft_occupancy: torch.Tensor,
    boundary_target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    radius: int,
) -> torch.Tensor:
    """Penalize missing expected structured-selection mass near GT boundaries."""

    if soft_occupancy.shape != boundary_target.shape:
        raise ValueError("soft_occupancy and boundary_target must have identical [B,T] shape")
    valid = valid_mask.to(device=soft_occupancy.device, dtype=torch.bool)
    if valid.shape != soft_occupancy.shape:
        raise ValueError("valid_mask must align with soft_occupancy")
    radius = int(radius)
    if radius < 0:
        raise ValueError("boundary coverage radius must be non-negative")
    occupancy = soft_occupancy.float().masked_fill(~valid, 0.0)
    target = boundary_target.to(device=occupancy.device, dtype=occupancy.dtype).masked_fill(~valid, 0.0)
    rows = []
    for batch_idx in range(occupancy.shape[0]):
        boundary_positions = torch.nonzero(target[batch_idx] > 0.0, as_tuple=False).flatten()
        if boundary_positions.numel() == 0:
            continue
        positions = torch.arange(occupancy.shape[1], device=occupancy.device)
        neighborhoods = (positions[None, :] - boundary_positions[:, None]).abs() <= radius
        neighborhood_mass = (occupancy[batch_idx][None, :] * neighborhoods).sum(dim=1)
        weights = target[batch_idx, boundary_positions]
        rows.append((weights * torch.exp(-neighborhood_mass)).sum() / weights.sum().clamp_min(1e-8))
    if not rows:
        return soft_occupancy.sum() * 0.0
    return torch.stack(rows).mean()


def boundary_burst_coverage_loss(
    soft_occupancy: torch.Tensor,
    endpoint_target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    radius: int,
    quota: float,
    side_min_mass: float = 1.0,
    anchor_weight: float = 1.0,
    bilateral_weight: float = 1.0,
    quota_weight: float = 1.0,
    fairness_weight: float = 0.5,
    overfill_weight: float = 0.25,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Match structured occupancy to Oracle-like bilateral boundary bursts."""

    if soft_occupancy.shape != endpoint_target.shape:
        raise ValueError("soft_occupancy and endpoint_target must have identical [B,T] shape")
    valid = valid_mask.to(device=soft_occupancy.device, dtype=torch.bool)
    if valid.shape != soft_occupancy.shape:
        raise ValueError("valid_mask must align with soft_occupancy")
    radius = int(radius)
    quota = float(quota)
    side_min_mass = float(side_min_mass)
    weights = tuple(
        float(value)
        for value in (
            anchor_weight,
            bilateral_weight,
            quota_weight,
            fairness_weight,
            overfill_weight,
        )
    )
    if radius <= 0 or quota <= 0.0 or side_min_mass < 0.0:
        raise ValueError("boundary burst radius/quota/side mass are invalid")
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("boundary burst loss weights must be finite and non-negative")

    occupancy = soft_occupancy.float().masked_fill(~valid, 0.0).clamp(0.0, 1.0)
    target = endpoint_target.to(device=occupancy.device, dtype=torch.float32)
    target = target.masked_fill(~valid, 0.0)
    anchors = []
    bilaterals = []
    quotas = []
    overfills = []
    deficits = []
    for batch_idx in range(occupancy.shape[0]):
        endpoint_positions = torch.nonzero(
            target[batch_idx] > 0.0,
            as_tuple=False,
        ).flatten()
        for endpoint in endpoint_positions.tolist():
            lo = max(0, int(endpoint) - radius)
            hi = min(int(occupancy.shape[1]), int(endpoint) + radius + 1)
            left = occupancy[batch_idx, lo:int(endpoint)]
            right = occupancy[batch_idx, int(endpoint) + 1 : hi]
            center = occupancy[batch_idx, int(endpoint)]
            left_target = min(side_min_mass, float(left.numel()))
            right_target = min(side_min_mass, float(right.numel()))
            left_mass = left.sum()
            right_mass = right.sum()
            total_mass = center + left_mass + right_mass

            anchors.append(-torch.log(center.clamp_min(1.0e-6)))
            left_loss = F.relu(left.new_tensor(left_target) - left_mass).square()
            right_loss = F.relu(right.new_tensor(right_target) - right_mass).square()
            side_denom = max(left_target * left_target + right_target * right_target, 1.0)
            bilaterals.append((left_loss + right_loss) / side_denom)
            quotas.append(F.relu(total_mass.new_tensor(quota) - total_mass).square() / (quota * quota))
            overfills.append(F.relu(total_mass - total_mass.new_tensor(quota)).square() / (quota * quota))
            deficits.append((1.0 - total_mass / quota).clamp(0.0, 1.0))

    if not anchors:
        # Keep empty-background windows connected to the structured policy so
        # a loss-only backward remains valid under DDP/AMP.
        zero = occupancy.sum() * 0.0
        return zero, {
            "anchor": zero,
            "bilateral": zero,
            "quota": zero,
            "fairness": zero,
            "overfill": zero,
        }
    anchor_loss = torch.stack(anchors).mean()
    bilateral_loss = torch.stack(bilaterals).mean()
    quota_loss = torch.stack(quotas).mean()
    overfill_loss = torch.stack(overfills).mean()
    fairness_loss = torch.stack(deficits).amax()
    total = (
        anchor_weight * anchor_loss
        + bilateral_weight * bilateral_loss
        + quota_weight * quota_loss
        + fairness_weight * fairness_loss
        + overfill_weight * overfill_loss
    )
    return total, {
        "anchor": anchor_loss.detach(),
        "bilateral": bilateral_loss.detach(),
        "quota": quota_loss.detach(),
        "fairness": fairness_loss.detach(),
        "overfill": overfill_loss.detach(),
    }


__all__ = [
    "ASFORMER_ENCODER_HIDDEN_KIND",
    "DucaProtectedTransitionScorer",
    "DucaTransitionUtilityScorer",
    "balanced_binary_actionness_loss",
    "boundary_burst_coverage_loss",
    "build_transition_descriptors",
    "build_boundary_burst_utility",
    "calibrated_actionness_probability",
    "continuous_policy_logits",
    "coverage_floor_distribution",
    "local_boundary_coverage_loss",
    "local_boundary_mass_coverage_loss",
    "transition_distribution_loss",
    "transition_utility_paths",
]
