from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .structured_selection import (
    exact_uniform_positions,
    physical_exact_k_select,
    physical_exact_k_viterbi,
    physical_exact_uniform_gap_cap,
)


RIME_DECODER_FAMILIES = ("independent", "strict_nested", "weak_overlap")
RIME_CONTRACT = "duca_rime_physical_dynamic_k_v1"


def build_cost_matched_mixed_k_cycle(
    candidate_budgets: Sequence[int],
    schedule_counts: Sequence[int],
    *,
    candidate_costs: Sequence[float] | None = None,
    target_mean_cost: float,
    schedule_seed: int,
) -> tuple[int, ...]:
    """Build an immutable, deterministically permuted successful-update K cycle."""

    budgets = tuple(int(value) for value in candidate_budgets)
    counts = tuple(int(value) for value in schedule_counts)
    costs = (
        tuple(float(value) for value in candidate_costs)
        if candidate_costs is not None
        else tuple(float(value) for value in budgets)
    )
    if (
        len(budgets) < 2
        or tuple(sorted(set(budgets))) != budgets
        or len(counts) != len(budgets)
        or len(costs) != len(budgets)
        or any(value <= 0 for value in budgets)
        or any(value <= 0 for value in counts)
        or any(not math.isfinite(value) or value <= 0.0 for value in costs)
    ):
        raise ValueError("mixed-K budgets, costs, and positive schedule counts must align")
    cycle_len = sum(counts)
    target = float(target_mean_cost)
    realized = sum(count * cost for count, cost in zip(counts, costs)) / cycle_len
    if not math.isfinite(target) or not math.isclose(
        realized,
        target,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("mixed-K schedule mean cost does not match its frozen target")
    tokens = [
        (
            hashlib.sha256(
                f"{int(schedule_seed)}|{budget}|{occurrence}".encode("utf-8")
            ).digest(),
            budget,
        )
        for budget, count in zip(budgets, counts)
        for occurrence in range(count)
    ]
    tokens.sort(key=lambda item: item[0])
    cycle = tuple(int(budget) for _digest, budget in tokens)
    if any(cycle.count(budget) != count for budget, count in zip(budgets, counts)):
        raise RuntimeError("mixed-K cycle construction changed its registered histogram")
    return cycle


@dataclass(frozen=True)
class RimeBudgetDecision:
    candidate_budgets: torch.Tensor
    predicted_utility: torch.Tensor
    predicted_risk: torch.Tensor
    predicted_uncertainty: torch.Tensor
    risk_upper: torch.Tensor
    measured_cost: torch.Tensor
    objective: torch.Tensor
    feasible: torch.Tensor
    selected_index: torch.Tensor
    requested_k: torch.Tensor
    fallback_to_kmax: torch.Tensor
    frozen_price: torch.Tensor
    target_mean_cost: float
    policy_name: str = "rime_frozen_price_per_video"

    def validate(self, *, batch_size: int | None = None) -> "RimeBudgetDecision":
        if self.candidate_budgets.ndim != 1:
            raise ValueError("RIME candidate budgets must be one-dimensional")
        if self.candidate_budgets.numel() < 2:
            raise ValueError("RIME requires at least two candidate budgets")
        if bool(torch.any(self.candidate_budgets[1:] <= self.candidate_budgets[:-1]).item()):
            raise ValueError("RIME candidate budgets must be strictly increasing")
        batch, candidate_count = self.predicted_utility.shape
        if batch_size is not None and batch != int(batch_size):
            raise ValueError("RIME decision batch size mismatch")
        expected = (batch, candidate_count)
        for name, value in (
            ("predicted_risk", self.predicted_risk),
            ("predicted_uncertainty", self.predicted_uncertainty),
            ("risk_upper", self.risk_upper),
            ("measured_cost", self.measured_cost),
            ("objective", self.objective),
            ("feasible", self.feasible),
        ):
            if value.shape != expected:
                raise ValueError(f"RIME {name} must be [B,M]")
        if candidate_count != int(self.candidate_budgets.numel()):
            raise ValueError("RIME candidate tensor width mismatch")
        if self.selected_index.shape != (batch,) or self.requested_k.shape != (batch,):
            raise ValueError("RIME hard decision tensors must be [B]")
        if self.fallback_to_kmax.shape != (batch,):
            raise ValueError("RIME fallback flags must be [B]")
        if not bool(
            torch.isfinite(
                torch.cat(
                    (
                        self.predicted_utility,
                        self.predicted_risk,
                        self.predicted_uncertainty,
                        self.risk_upper,
                        self.measured_cost,
                        self.objective,
                    ),
                    dim=1,
                )
            ).all().item()
        ):
            raise ValueError("RIME decision contains non-finite values")
        if bool(torch.any((self.predicted_risk < 0.0) | (self.predicted_risk > 1.0)).item()):
            raise ValueError("RIME predicted risks must lie in [0,1]")
        if bool(torch.any(self.predicted_uncertainty < 0.0).item()):
            raise ValueError("RIME uncertainty must be non-negative")
        if bool(torch.any(self.measured_cost <= 0.0).item()):
            raise ValueError("RIME measured costs must be positive")
        if bool(torch.any((self.selected_index < 0) | (self.selected_index >= candidate_count)).item()):
            raise ValueError("RIME selected candidate index is out of range")
        expected_k = self.candidate_budgets.to(self.selected_index.device)[
            self.selected_index
        ]
        if not torch.equal(expected_k.to(self.requested_k.dtype), self.requested_k):
            raise ValueError("RIME requested K disagrees with the selected candidate")
        return self


@dataclass(frozen=True)
class RimeCostLedger:
    requested_k: tuple[int, ...]
    effective_k: tuple[int, ...]
    unique_k: tuple[int, ...]
    backbone_input_k: tuple[int, ...]
    padded_k: tuple[int, ...]
    risk_fallback: tuple[bool, ...]
    dynamic_compute_realized: bool
    unit: str = "heavy_rgb_frames"

    def validate(self, *, require_no_padding: bool = True) -> "RimeCostLedger":
        lengths = {
            len(self.requested_k),
            len(self.effective_k),
            len(self.unique_k),
            len(self.backbone_input_k),
            len(self.padded_k),
            len(self.risk_fallback),
        }
        if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
            raise ValueError("RIME ledger fields must have one shared nonempty batch")
        for index, values in enumerate(
            zip(
                self.requested_k,
                self.effective_k,
                self.unique_k,
                self.backbone_input_k,
                self.padded_k,
            )
        ):
            requested, effective, unique, backbone, padded = values
            if min(values) <= 0:
                raise ValueError(f"RIME ledger row {index} contains a non-positive K")
            if effective > requested or unique != effective:
                raise ValueError(
                    f"RIME ledger row {index} violates requested/effective/unique K"
                )
            if backbone < unique or padded < unique:
                raise ValueError(
                    f"RIME ledger row {index} under-reports heavy input or padding"
                )
            if require_no_padding and not (
                backbone == unique and padded == unique
            ):
                raise ValueError(
                    "RIME execution padded a sample beyond its unique heavy-frame count"
                )
        if require_no_padding and not self.dynamic_compute_realized:
            raise ValueError("RIME no-padding ledger must mark dynamic compute realized")
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "unit": self.unit,
            "requested_k": list(self.requested_k),
            "effective_k": list(self.effective_k),
            "unique_k": list(self.unique_k),
            "backbone_input_k": list(self.backbone_input_k),
            "padded_k": list(self.padded_k),
            "risk_fallback": list(self.risk_fallback),
            "dynamic_compute_realized": bool(self.dynamic_compute_realized),
        }


@dataclass(frozen=True)
class RimeDecodeOutput:
    hard_occupancy: torch.Tensor
    hard_slot_assignment: torch.Tensor
    hard_positions: torch.Tensor
    hard_slot_mask: torch.Tensor
    soft_occupancy: torch.Tensor | None
    soft_slot_assignment: torch.Tensor | None
    selection_st: torch.Tensor | None
    requested_k: torch.Tensor
    effective_k: torch.Tensor
    max_gap_seconds: torch.Tensor
    decoder_family: str
    overlap_fraction: float | None
    constant_uniform_identity: torch.Tensor
    ledger: RimeCostLedger


def _masked_summary(
    evidence: torch.Tensor,
    policy_scores: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if evidence.ndim != 3:
        raise ValueError("RIME evidence must be [B,T,D]")
    if policy_scores.shape != evidence.shape[:2] or valid_mask.shape != policy_scores.shape:
        raise ValueError("RIME evidence, policy scores and validity must align")
    valid = valid_mask.to(device=evidence.device, dtype=torch.bool)
    if bool(torch.any(valid.sum(dim=1) <= 0).item()):
        raise ValueError("RIME requires one nonempty valid prefix per sample")
    work = evidence.float()
    weights = valid.to(dtype=work.dtype)
    count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (work * weights[:, :, None]).sum(dim=1) / count
    centered = (work - mean[:, None, :]).masked_fill(~valid[:, :, None], 0.0)
    std = torch.sqrt(
        centered.square().sum(dim=1) / count
        + torch.finfo(work.dtype).eps
    )
    floor = -torch.finfo(work.dtype).max
    maximum = work.masked_fill(~valid[:, :, None], floor).max(dim=1).values

    scores = policy_scores.float().masked_fill(~valid, 0.0)
    score_mean = scores.sum(dim=1, keepdim=True) / count
    score_centered = (scores - score_mean).masked_fill(~valid, 0.0)
    score_std = torch.sqrt(
        score_centered.square().sum(dim=1, keepdim=True) / count
        + torch.finfo(work.dtype).eps
    )
    score_max = policy_scores.float().masked_fill(~valid, floor).max(
        dim=1, keepdim=True
    ).values
    probability = torch.sigmoid(policy_scores.float()).clamp(1.0e-6, 1.0 - 1.0e-6)
    entropy = (
        -probability * probability.log()
        - (1.0 - probability) * (1.0 - probability).log()
    ).masked_fill(~valid, 0.0)
    entropy_mean = entropy.sum(dim=1, keepdim=True) / count
    if policy_scores.shape[1] > 1:
        adjacent_valid = valid[:, 1:] & valid[:, :-1]
        delta = (policy_scores[:, 1:] - policy_scores[:, :-1]).abs().float()
        delta_count = adjacent_valid.sum(dim=1, keepdim=True).clamp_min(1)
        delta_mean = delta.masked_fill(~adjacent_valid, 0.0).sum(
            dim=1, keepdim=True
        ) / delta_count
    else:
        delta_mean = score_mean * 0.0
    valid_fraction = count / float(policy_scores.shape[1])
    return torch.cat(
        (
            mean,
            std,
            maximum,
            score_mean,
            score_std,
            score_max,
            entropy_mean,
            delta_mean,
            valid_fraction,
        ),
        dim=1,
    )


class RimeBudgetController(nn.Module):
    """Batch-invariant finite-K utility/risk allocation with a frozen price."""

    def __init__(
        self,
        evidence_dim: int,
        candidate_budgets: Sequence[int],
        *,
        candidate_costs: Sequence[float] | None = None,
        hidden_dim: int = 128,
        frozen_price: float = 0.0,
        target_mean_cost: float | None = None,
        risk_weight: float = 1.0,
        risk_threshold: float = 0.35,
        uncertainty_z: float = 1.645,
        use_risk: bool = True,
    ) -> None:
        super().__init__()
        budgets = tuple(int(value) for value in candidate_budgets)
        if len(budgets) < 2 or tuple(sorted(set(budgets))) != budgets:
            raise ValueError("candidate_budgets must be unique and strictly increasing")
        if budgets[0] <= 0:
            raise ValueError("candidate budgets must be positive")
        costs = (
            tuple(float(value) for value in candidate_costs)
            if candidate_costs is not None
            else tuple(float(value) for value in budgets)
        )
        if len(costs) != len(budgets) or any(
            not math.isfinite(value) or value <= 0.0 for value in costs
        ):
            raise ValueError("candidate costs must be positive finite and align with K")
        if any(right <= left for left, right in zip(costs, costs[1:])):
            raise ValueError("candidate costs must be strictly increasing")
        if not math.isfinite(float(frozen_price)) or float(frozen_price) < 0.0:
            raise ValueError("frozen_price must be finite and non-negative")
        if not 0.0 <= float(risk_threshold) <= 1.0:
            raise ValueError("risk_threshold must lie in [0,1]")
        if min(float(risk_weight), float(uncertainty_z)) < 0.0:
            raise ValueError("risk and uncertainty weights must be non-negative")
        target = (
            float(sum(costs) / len(costs))
            if target_mean_cost is None
            else float(target_mean_cost)
        )
        if not min(costs) <= target <= max(costs):
            raise ValueError("target_mean_cost must lie on the candidate cost range")

        self.evidence_dim = int(evidence_dim)
        self.hidden_dim = int(hidden_dim)
        self.risk_weight = float(risk_weight)
        self.risk_threshold = float(risk_threshold)
        self.uncertainty_z = float(uncertainty_z)
        self.use_risk = bool(use_risk)
        self.target_mean_cost = target
        summary_dim = 3 * self.evidence_dim + 6
        self.trunk = nn.Sequential(
            nn.Linear(summary_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        candidate_count = len(budgets)
        self.marginal_utility_head = nn.Linear(
            self.hidden_dim,
            candidate_count - 1,
        )
        self.risk_head = nn.Linear(self.hidden_dim, candidate_count)
        self.uncertainty_head = nn.Linear(self.hidden_dim, candidate_count)
        self.register_buffer(
            "candidate_budgets",
            torch.tensor(budgets, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "candidate_costs",
            torch.tensor(costs, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "frozen_price",
            torch.tensor(float(frozen_price), dtype=torch.float32),
            persistent=True,
        )

    def forward(
        self,
        evidence: torch.Tensor,
        policy_scores: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> RimeBudgetDecision:
        summary = _masked_summary(evidence, policy_scores, valid_mask)
        hidden = self.trunk(summary)
        increments = F.softplus(self.marginal_utility_head(hidden))
        utility = torch.cat(
            (
                increments.new_zeros((increments.shape[0], 1)),
                increments.cumsum(dim=1),
            ),
            dim=1,
        )
        risk = torch.sigmoid(self.risk_head(hidden))
        uncertainty = F.softplus(self.uncertainty_head(hidden))
        risk_upper = (risk + self.uncertainty_z * uncertainty).clamp(0.0, 1.0)
        if not self.use_risk:
            risk_term = risk_upper * 0.0
            feasible = torch.ones_like(risk_upper, dtype=torch.bool)
        else:
            risk_term = risk_upper
            feasible = risk_upper <= self.risk_threshold
        # Kmax is a deterministic safety fallback even when the learned risk
        # surrogate calls every candidate infeasible.
        feasible = feasible.clone()
        feasible[:, -1] = True
        costs = self.candidate_costs.to(device=utility.device, dtype=utility.dtype)
        measured_cost = costs[None, :].expand_as(utility)
        normalized_cost = measured_cost / measured_cost[:, -1:].clamp_min(
            torch.finfo(measured_cost.dtype).eps
        )
        objective = (
            utility
            - self.risk_weight * risk_term
            - self.frozen_price.to(device=utility.device, dtype=utility.dtype)
            * normalized_cost
        )
        objective = objective.masked_fill(~feasible, -torch.finfo(objective.dtype).max)
        selected_index = objective.argmax(dim=1)
        nonmax_feasible = feasible[:, :-1].any(dim=1)
        fallback_tensor = (
            ~nonmax_feasible
            if self.use_risk
            else torch.zeros_like(nonmax_feasible, dtype=torch.bool)
        )
        selected_index = torch.where(
            fallback_tensor,
            torch.full_like(selected_index, utility.shape[1] - 1),
            selected_index,
        )
        requested = self.candidate_budgets.to(selected_index.device)[selected_index]
        return RimeBudgetDecision(
            candidate_budgets=self.candidate_budgets,
            predicted_utility=utility,
            predicted_risk=risk,
            predicted_uncertainty=uncertainty,
            risk_upper=risk_upper,
            measured_cost=measured_cost,
            objective=objective,
            feasible=feasible,
            selected_index=selected_index,
            requested_k=requested,
            fallback_to_kmax=fallback_tensor,
            frozen_price=self.frozen_price,
            target_mean_cost=self.target_mean_cost,
        ).validate(batch_size=evidence.shape[0])


def choose_rime_budget(
    utility: torch.Tensor,
    risk_upper: torch.Tensor,
    measured_cost: torch.Tensor,
    *,
    price: float,
    risk_weight: float,
    risk_threshold: float,
    use_risk: bool = True,
) -> torch.Tensor:
    if (
        utility.ndim != 2
        or risk_upper.shape != utility.shape
        or measured_cost.shape != utility.shape
    ):
        raise ValueError("RIME calibration tensors must be aligned [N,M]")
    if not bool(
        torch.isfinite(
            torch.cat((utility, risk_upper, measured_cost), dim=1)
        ).all().item()
    ):
        raise ValueError("RIME calibration tensors must be finite")
    if float(price) < 0.0 or float(risk_weight) < 0.0:
        raise ValueError("RIME price and risk weight must be non-negative")
    feasible = (
        risk_upper <= float(risk_threshold)
        if use_risk
        else torch.ones_like(risk_upper, dtype=torch.bool)
    )
    feasible = feasible.clone()
    feasible[:, -1] = True
    normalized_cost = measured_cost / measured_cost[:, -1:].clamp_min(
        torch.finfo(measured_cost.dtype).eps
    )
    score = (
        utility
        - float(risk_weight) * (risk_upper if use_risk else risk_upper * 0.0)
        - float(price) * normalized_cost
    ).masked_fill(~feasible, -torch.finfo(utility.dtype).max)
    selected = score.argmax(dim=1)
    if use_risk:
        selected = torch.where(
            feasible[:, :-1].any(dim=1),
            selected,
            torch.full_like(selected, utility.shape[1] - 1),
        )
    return selected


def calibrate_rime_price(
    utility: torch.Tensor,
    risk_upper: torch.Tensor,
    measured_cost: torch.Tensor,
    *,
    target_mean_cost: float,
    risk_weight: float,
    risk_threshold: float,
    use_risk: bool = True,
    max_price: float = 1.0e6,
    iterations: int = 80,
) -> dict[str, float | list[int]]:
    """Freeze the smallest price whose calibration-set mean cost meets target."""

    if utility.shape[0] < 2:
        raise ValueError("RIME price calibration requires at least two videos")
    target = float(target_mean_cost)
    if not math.isfinite(target):
        raise ValueError("target_mean_cost must be finite")
    candidate_cost = measured_cost.detach().float()
    minimum = float(candidate_cost.min(dim=1).values.mean().item())
    maximum = float(candidate_cost.max(dim=1).values.mean().item())
    if not minimum <= target <= maximum:
        raise ValueError("target mean cost is outside the attainable calibration range")

    def evaluate(price: float) -> tuple[torch.Tensor, float]:
        selected = choose_rime_budget(
            utility.detach().float(),
            risk_upper.detach().float(),
            candidate_cost,
            price=price,
            risk_weight=risk_weight,
            risk_threshold=risk_threshold,
            use_risk=use_risk,
        )
        chosen = candidate_cost.gather(1, selected[:, None]).squeeze(1)
        return selected, float(chosen.mean().item())

    low = 0.0
    high = 1.0
    _, high_mean = evaluate(high)
    while high_mean > target and high < float(max_price):
        high *= 2.0
        _, high_mean = evaluate(high)
    if high_mean > target:
        raise RuntimeError("RIME price search cannot attain the target mean cost")
    for _ in range(int(iterations)):
        middle = 0.5 * (low + high)
        _, mean_cost = evaluate(middle)
        if mean_cost <= target:
            high = middle
        else:
            low = middle
    selected, realized = evaluate(high)
    return {
        "frozen_price": float(high),
        "target_mean_cost": target,
        "realized_mean_cost": realized,
        "selected_indices": [int(value) for value in selected.cpu().tolist()],
    }


def rime_budget_supervision_losses(
    decision: RimeBudgetDecision,
    utility_target: torch.Tensor,
    risk_target: torch.Tensor,
    target_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    decision.validate()
    utility = utility_target.to(
        device=decision.predicted_utility.device,
        dtype=decision.predicted_utility.dtype,
    )
    risk = risk_target.to(
        device=decision.predicted_risk.device,
        dtype=decision.predicted_risk.dtype,
    )
    if utility.shape != decision.predicted_utility.shape or risk.shape != utility.shape:
        raise ValueError("RIME utility/risk targets must align with candidate budgets")
    mask = (
        torch.ones_like(utility, dtype=torch.bool)
        if target_mask is None
        else target_mask.to(device=utility.device, dtype=torch.bool)
    )
    if mask.shape != utility.shape or not bool(mask.any().item()):
        raise ValueError("RIME target mask must select at least one [B,M] label")
    if not bool(torch.isfinite(utility[mask]).all().item()):
        raise ValueError("RIME utility targets must be finite where active")
    if bool(torch.any((risk[mask] < 0.0) | (risk[mask] > 1.0)).item()):
        raise ValueError("RIME risk targets must be binary/probabilistic in [0,1]")
    utility_loss = F.smooth_l1_loss(
        decision.predicted_utility[mask],
        utility[mask],
    )
    risk_loss = F.binary_cross_entropy(
        decision.predicted_risk[mask],
        risk[mask],
    )
    uncertainty_target = (
        decision.predicted_utility.detach() - utility
    ).abs()
    uncertainty_loss = F.smooth_l1_loss(
        decision.predicted_uncertainty[mask],
        uncertainty_target[mask],
    )
    return {
        "selector_rime_utility_loss": utility_loss,
        "selector_rime_risk_loss": risk_loss,
        "selector_rime_uncertainty_loss": uncertainty_loss,
    }


def rime_rank_alignment_loss(
    policy_scores: torch.Tensor,
    hard_utility: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if policy_scores.shape != hard_utility.shape or valid_mask.shape != policy_scores.shape:
        raise ValueError("RIME frame utility and policy score tensors must align [B,T]")
    valid = valid_mask.to(device=policy_scores.device, dtype=torch.bool)
    losses = []
    for batch_index in range(int(policy_scores.shape[0])):
        active = valid[batch_index]
        if int(active.sum().item()) < 2:
            continue
        target = hard_utility[batch_index, active].detach().float()
        if not bool(torch.isfinite(target).all().item()):
            raise ValueError("RIME hard frame utility must be finite")
        target = (target - target.mean()) / target.std(unbiased=False).clamp_min(1.0e-6)
        score = policy_scores[batch_index, active].float()
        score = (score - score.mean()) / score.std(unbiased=False).clamp_min(1.0e-6)
        losses.append(F.smooth_l1_loss(score, target))
    if not losses:
        return policy_scores.float().sum() * 0.0
    return torch.stack(losses).mean()


def _constant_valid_scores(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    tolerance: float = 1.0e-12,
) -> torch.Tensor:
    rows = []
    for batch_index in range(int(scores.shape[0])):
        active = scores[batch_index, valid_mask[batch_index].bool()].detach().double()
        rows.append(
            bool(active.numel() <= 1)
            or float((active.max() - active.min()).item()) <= float(tolerance)
        )
    return torch.tensor(rows, device=scores.device, dtype=torch.bool)


def _uniform_positions_row(valid_len: int, k: int, device: torch.device) -> torch.Tensor:
    return exact_uniform_positions(
        valid_len,
        min(int(k), int(valid_len)),
        device=device,
    )


def _hard_from_positions(
    positions: torch.Tensor,
    *,
    temporal_len: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    occupancy = torch.zeros(
        temporal_len,
        device=positions.device,
        dtype=dtype,
    )
    occupancy.scatter_(0, positions, 1.0)
    slots = torch.zeros(
        (int(positions.numel()), temporal_len),
        device=positions.device,
        dtype=dtype,
    )
    slots.scatter_(1, positions[:, None], 1.0)
    return occupancy, slots


def _required_for_weak_overlap(
    previous: torch.Tensor,
    scores: torch.Tensor,
    *,
    overlap_fraction: float,
) -> torch.Tensor:
    retain = int(math.ceil(float(overlap_fraction) * int(previous.numel())))
    retain = min(max(retain, 1), int(previous.numel()))
    ranked = sorted(
        (int(value) for value in previous.tolist()),
        key=lambda position: (-float(scores[position].detach().item()), position),
    )
    return torch.tensor(
        sorted(ranked[:retain]),
        device=previous.device,
        dtype=torch.long,
    )


def _decode_rime_row(
    scores: torch.Tensor,
    physical_seconds: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    requested_k: int,
    candidate_budgets: Sequence[int],
    decoder_family: str,
    weak_overlap_fraction: float,
    training: bool,
    force_uniform: bool,
    execution_quantum: int,
) -> dict[str, torch.Tensor | bool]:
    valid_len = int(valid_mask.sum().item())
    quantum = int(execution_quantum)
    if quantum <= 0:
        raise ValueError("RIME execution_quantum must be positive")
    available = valid_len - valid_len % quantum
    if available <= 0:
        raise ValueError(
            "RIME valid window is shorter than one heavy-backbone execution quantum"
        )
    effective_k = min(int(requested_k), available)
    effective_k -= effective_k % quantum
    if effective_k <= 0:
        raise ValueError("RIME requested K cannot realize one execution quantum")
    constant = bool(
        _constant_valid_scores(
            scores[None, :],
            valid_mask[None, :],
        )[0].item()
    )
    cap = physical_exact_uniform_gap_cap(
        physical_seconds[None, :],
        valid_mask[None, :],
        k=effective_k,
    )
    if force_uniform or (decoder_family == "independent" and constant):
        positions = _uniform_positions_row(valid_len, effective_k, scores.device)
        hard, slots = _hard_from_positions(
            positions,
            temporal_len=int(scores.numel()),
            dtype=scores.dtype,
        )
        return {
            "hard": hard,
            "hard_slots": slots,
            "positions": positions,
            "soft": hard,
            "soft_slots": slots,
            "selection_st": slots,
            "cap": cap[0],
            "constant_uniform_identity": bool(constant),
        }

    required_positions = torch.empty(
        0,
        device=scores.device,
        dtype=torch.long,
    )
    ladder = [
        min(int(value), valid_len)
        for value in candidate_budgets
        if int(value) <= int(requested_k)
    ]
    if effective_k not in ladder:
        ladder.append(effective_k)
    ladder = sorted(set(value for value in ladder if value > 0))
    if decoder_family == "independent":
        ladder = [effective_k]
    final = None
    for ladder_k in ladder:
        required_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        if required_positions.numel():
            required_mask.scatter_(0, required_positions, True)
        ladder_cap = physical_exact_uniform_gap_cap(
            physical_seconds[None, :],
            valid_mask[None, :],
            k=ladder_k,
        )
        if training and ladder_k == effective_k:
            selected = physical_exact_k_select(
                scores[None, :],
                physical_seconds[None, :],
                valid_mask[None, :],
                k=ladder_k,
                max_gap_seconds=ladder_cap,
                required_mask=required_mask[None, :],
            )
            positions = selected.hard_positions[0, : int(selected.effective_k[0].item())]
            final = {
                "hard": selected.hard_occupancy[0],
                "hard_slots": selected.hard_slot_assignment[
                    0, : positions.numel()
                ],
                "positions": positions,
                "soft": selected.soft_occupancy[0],
                "soft_slots": selected.soft_slot_assignment[
                    0, : positions.numel()
                ],
                "selection_st": selected.selection_st[0, : positions.numel()],
                "cap": selected.max_gap_seconds[0],
            }
        else:
            selected = physical_exact_k_viterbi(
                scores[None, :],
                physical_seconds[None, :],
                valid_mask[None, :],
                k=ladder_k,
                max_gap_seconds=ladder_cap,
                required_mask=required_mask[None, :],
            )
            positions = selected.hard_positions[0, : int(selected.effective_k[0].item())]
            hard = selected.hard_occupancy[0]
            hard_slots = selected.hard_slot_assignment[0, : positions.numel()]
            final = {
                "hard": hard,
                "hard_slots": hard_slots,
                "positions": positions,
                "soft": hard,
                "soft_slots": hard_slots,
                "selection_st": hard_slots,
                "cap": selected.max_gap_seconds[0],
            }
        if ladder_k == effective_k:
            break
        if decoder_family == "strict_nested":
            required_positions = positions
        elif decoder_family == "weak_overlap":
            required_positions = _required_for_weak_overlap(
                positions,
                scores,
                overlap_fraction=weak_overlap_fraction,
            )
        else:
            raise ValueError(f"unknown RIME decoder family: {decoder_family}")
    if final is None:
        raise RuntimeError("RIME decoder produced no exact-K path")
    final["constant_uniform_identity"] = False
    return final


def decode_rime_exact_k(
    policy_log_potential: torch.Tensor,
    physical_seconds: torch.Tensor,
    valid_mask: torch.Tensor,
    requested_k: torch.Tensor | Sequence[int] | int,
    *,
    candidate_budgets: Sequence[int],
    decoder_family: str = "independent",
    weak_overlap_fraction: float = 0.50,
    training: bool = False,
    force_uniform: bool = False,
    risk_fallback: torch.Tensor | Sequence[bool] | None = None,
    require_homogeneous_execution: bool = True,
    execution_quantum: int = 1,
) -> RimeDecodeOutput:
    if decoder_family not in RIME_DECODER_FAMILIES:
        raise ValueError(f"decoder_family must be one of {RIME_DECODER_FAMILIES}")
    if not 0.0 < float(weak_overlap_fraction) <= 1.0:
        raise ValueError("weak_overlap_fraction must lie in (0,1]")
    if (
        policy_log_potential.ndim != 2
        or physical_seconds.shape != policy_log_potential.shape
        or valid_mask.shape != policy_log_potential.shape
    ):
        raise ValueError("RIME decoder tensors must align [B,T]")
    valid = valid_mask.to(device=policy_log_potential.device, dtype=torch.bool)
    requested = torch.as_tensor(
        requested_k,
        device=policy_log_potential.device,
        dtype=torch.long,
    ).reshape(-1)
    if requested.numel() == 1 and policy_log_potential.shape[0] > 1:
        requested = requested.expand(policy_log_potential.shape[0])
    if requested.shape != (policy_log_potential.shape[0],):
        raise ValueError("RIME requested_k must be scalar or [B]")
    allowed = {int(value) for value in candidate_budgets}
    if any(int(value) not in allowed for value in requested.tolist()):
        raise ValueError("RIME requested K must belong to the frozen candidate set")
    quantum = int(execution_quantum)
    if quantum <= 0:
        raise ValueError("RIME execution_quantum must be positive")
    valid_counts = valid.long().sum(dim=1)
    available = valid_counts - torch.remainder(valid_counts, quantum)
    effective = torch.minimum(requested, available)
    effective = effective - torch.remainder(effective, quantum)
    if bool(torch.any(effective <= 0).item()):
        raise ValueError(
            "RIME valid window is shorter than one heavy-backbone execution "
            "quantum after requested-K clipping"
        )
    if require_homogeneous_execution and bool(
        torch.any(effective != effective[0]).item()
    ):
        raise ValueError(
            "RIME heavy execution requires a homogeneous effective-K bucket; "
            "use batch_size=1 or dispatch by K before the heavy backbone"
        )
    rows = [
        _decode_rime_row(
            policy_log_potential[index],
            physical_seconds[index],
            valid[index],
            requested_k=int(requested[index].item()),
            candidate_budgets=candidate_budgets,
            decoder_family=decoder_family,
            weak_overlap_fraction=weak_overlap_fraction,
            training=bool(training),
            force_uniform=bool(force_uniform),
            execution_quantum=quantum,
        )
        for index in range(int(policy_log_potential.shape[0]))
    ]
    output_width = int(effective.max().item())
    temporal_len = int(policy_log_potential.shape[1])
    position_rows = []
    slot_mask_rows = []
    hard_slot_rows = []
    soft_slot_rows = []
    selection_st_rows = []
    for row in rows:
        positions = row["positions"]
        row_k = int(positions.numel())
        padded_positions = torch.full(
            (output_width,),
            -1,
            device=policy_log_potential.device,
            dtype=torch.long,
        )
        padded_positions[:row_k] = positions
        position_rows.append(padded_positions)
        slot_mask = torch.zeros(
            output_width,
            device=policy_log_potential.device,
            dtype=torch.bool,
        )
        slot_mask[:row_k] = True
        slot_mask_rows.append(slot_mask)
        hard_slot_rows.append(
            F.pad(row["hard_slots"], (0, 0, 0, output_width - row_k))
        )
        soft_slot_rows.append(
            F.pad(row["soft_slots"], (0, 0, 0, output_width - row_k))
        )
        selection_st_rows.append(
            F.pad(row["selection_st"], (0, 0, 0, output_width - row_k))
        )
    if risk_fallback is None:
        fallback = torch.zeros_like(requested, dtype=torch.bool)
    else:
        fallback = torch.as_tensor(
            risk_fallback,
            device=requested.device,
            dtype=torch.bool,
        ).reshape(-1)
        if fallback.shape != requested.shape:
            raise ValueError("RIME risk_fallback must align with requested K")
    no_padding = bool(torch.all(effective == output_width).item())
    ledger = RimeCostLedger(
        requested_k=tuple(int(value) for value in requested.cpu().tolist()),
        effective_k=tuple(int(value) for value in effective.cpu().tolist()),
        unique_k=tuple(int(value) for value in effective.cpu().tolist()),
        backbone_input_k=tuple(output_width for _ in range(int(requested.numel()))),
        padded_k=tuple(output_width for _ in range(int(requested.numel()))),
        risk_fallback=tuple(bool(value) for value in fallback.cpu().tolist()),
        dynamic_compute_realized=no_padding,
    )
    ledger.validate(require_no_padding=require_homogeneous_execution)
    return RimeDecodeOutput(
        hard_occupancy=torch.stack([row["hard"] for row in rows], dim=0),
        hard_slot_assignment=torch.stack(hard_slot_rows, dim=0),
        hard_positions=torch.stack(position_rows, dim=0),
        hard_slot_mask=torch.stack(slot_mask_rows, dim=0),
        soft_occupancy=(
            torch.stack([row["soft"] for row in rows], dim=0)
            if training
            else None
        ),
        soft_slot_assignment=(
            torch.stack(soft_slot_rows, dim=0) if training else None
        ),
        selection_st=(
            torch.stack(selection_st_rows, dim=0) if training else None
        ),
        requested_k=requested,
        effective_k=effective,
        max_gap_seconds=torch.stack([row["cap"] for row in rows], dim=0),
        decoder_family=decoder_family,
        overlap_fraction=(
            float(weak_overlap_fraction)
            if decoder_family == "weak_overlap"
            else None
        ),
        constant_uniform_identity=torch.tensor(
            [bool(row["constant_uniform_identity"]) for row in rows],
            device=policy_log_potential.device,
            dtype=torch.bool,
        ),
        ledger=ledger,
    )


def decode_rime_panel(
    policy_log_potential: torch.Tensor,
    physical_seconds: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    candidate_budgets: Sequence[int],
    weak_overlap_fraction: float = 0.50,
    execution_quantum: int = 1,
) -> Mapping[str, Mapping[int, RimeDecodeOutput]]:
    panel: dict[str, dict[int, RimeDecodeOutput]] = {}
    for family in RIME_DECODER_FAMILIES:
        panel[family] = {}
        for budget in candidate_budgets:
            panel[family][int(budget)] = decode_rime_exact_k(
                policy_log_potential,
                physical_seconds,
                valid_mask,
                int(budget),
                candidate_budgets=candidate_budgets,
                decoder_family=family,
                weak_overlap_fraction=weak_overlap_fraction,
                training=False,
                require_homogeneous_execution=False,
                execution_quantum=execution_quantum,
            )
    return panel


__all__ = [
    "RIME_CONTRACT",
    "RIME_DECODER_FAMILIES",
    "RimeBudgetController",
    "RimeBudgetDecision",
    "RimeCostLedger",
    "RimeDecodeOutput",
    "calibrate_rime_price",
    "build_cost_matched_mixed_k_cycle",
    "choose_rime_budget",
    "decode_rime_exact_k",
    "decode_rime_panel",
    "rime_budget_supervision_losses",
    "rime_rank_alignment_loss",
]
