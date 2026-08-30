from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DynamicBudgetDecision:
    """Differentiable sample-wise budget decision with a hard forward value."""

    budget_hard: torch.Tensor
    budget_soft: torch.Tensor
    expected_cost: torch.Tensor
    continue_logits: torch.Tensor
    continue_soft: torch.Tensor
    continue_hard: torch.Tensor
    prefix_soft: torch.Tensor
    prefix_hard: torch.Tensor
    marginal_utility: torch.Tensor
    lambda_dual: torch.Tensor
    budget_min: int
    budget_max: int
    budget_multiple: int
    target_budget: float
    policy_name: str = "prefix_marginal_utility_stop"
    dual_target_unit: str = "detector_valid_temporal_observations"

    @property
    def hard_requested_k(self) -> torch.Tensor:
        return self.budget_hard

    @property
    def st_budget_k(self) -> torch.Tensor:
        return self.budget_soft

    @property
    def soft_expected_k(self) -> torch.Tensor:
        return self.expected_cost

    def validate(self, batch_size: Optional[int] = None) -> "DynamicBudgetDecision":
        if self.budget_hard.ndim != 1:
            raise ValueError("budget_hard must be [B]")
        if self.budget_soft.shape != self.budget_hard.shape:
            raise ValueError("budget_soft must match budget_hard")
        if batch_size is not None and self.budget_hard.numel() != int(batch_size):
            raise ValueError("budget decision batch size mismatch")
        if int(self.budget_min) <= 0:
            raise ValueError("budget_min must be positive")
        if int(self.budget_max) < int(self.budget_min):
            raise ValueError("budget_max must be >= budget_min")
        if int(self.budget_multiple) <= 0:
            raise ValueError("budget_multiple must be positive")
        if (int(self.budget_max) - int(self.budget_min)) % int(self.budget_multiple) != 0:
            raise ValueError("budget_max - budget_min must be divisible by budget_multiple")
        hard = self.budget_hard.to(dtype=torch.long)
        if torch.any(hard < int(self.budget_min)) or torch.any(hard > int(self.budget_max)):
            raise ValueError("budget_hard must lie within [budget_min, budget_max]")
        if torch.any((hard - int(self.budget_min)) % int(self.budget_multiple) != 0):
            raise ValueError("budget_hard must align to budget_multiple")
        if not torch.isfinite(self.budget_soft).all():
            raise ValueError("budget_soft must be finite")
        if self.expected_cost.shape != self.budget_hard.shape or not torch.isfinite(self.expected_cost).all():
            raise ValueError("expected_cost must be a finite [B] true soft expectation")
        if torch.any(self.budget_soft < float(self.budget_min) - 1e-4):
            raise ValueError("budget_soft below budget_min")
        if torch.any(self.budget_soft > float(self.budget_max) + 1e-4):
            raise ValueError("budget_soft above budget_max")
        if self.prefix_hard.ndim != 2:
            raise ValueError("prefix_hard must be [B,J]")
        if self.prefix_hard.shape != self.prefix_soft.shape:
            raise ValueError("prefix_hard/prefix_soft shape mismatch")
        if self.prefix_hard.shape != self.continue_hard.shape:
            raise ValueError("prefix_hard/continue_hard shape mismatch")
        if self.prefix_hard.shape[1] > 1 and torch.any(self.prefix_hard[:, 1:] > self.prefix_hard[:, :-1]):
            raise ValueError("prefix_hard must be monotonic non-increasing")
        return self


@dataclass
class VideoBudgetAllocation:
    """One video's discrete heavy-observation allocation.

    ``budget`` is the requested tier and ``actual_cost`` is the number of real
    observations available to the heavy path after short-window truncation.
    The allocator never represents missing observations with padding.
    """

    budget: torch.Tensor
    effective_budget: torch.Tensor
    actual_cost: torch.Tensor
    execution_slots: torch.Tensor
    padding_slots: torch.Tensor
    collapsed_to_baseline: torch.Tensor
    changed_mask: torch.Tensor
    predicted_total_utility: torch.Tensor
    target_actual_cost: int
    feasible: bool
    reason: str
    policy_name: str = "video_level_exact_total_marginal_reallocation"

    def validate(self, *, window_count: Optional[int] = None) -> "VideoBudgetAllocation":
        if self.budget.ndim != 1:
            raise ValueError("video budget allocation must be one-dimensional")
        if self.actual_cost.shape != self.budget.shape:
            raise ValueError("actual_cost must align with budget")
        if self.effective_budget.shape != self.budget.shape:
            raise ValueError("effective_budget must align with budget")
        if self.execution_slots.shape != self.budget.shape:
            raise ValueError("execution_slots must align with budget")
        if self.padding_slots.shape != self.budget.shape:
            raise ValueError("padding_slots must align with budget")
        if self.collapsed_to_baseline.shape != self.budget.shape:
            raise ValueError("collapsed_to_baseline must align with budget")
        if self.collapsed_to_baseline.dtype != torch.bool:
            raise ValueError("collapsed_to_baseline must be boolean")
        if self.changed_mask.shape != self.budget.shape:
            raise ValueError("changed_mask must align with budget")
        if self.changed_mask.dtype != torch.bool:
            raise ValueError("changed_mask must be boolean")
        if window_count is not None and self.budget.numel() != int(window_count):
            raise ValueError("video budget allocation window count mismatch")
        if torch.any(self.actual_cost <= 0) or torch.any(self.actual_cost > self.budget):
            raise ValueError("actual observation cost must lie in (0, requested budget]")
        if torch.any(self.execution_slots < self.actual_cost):
            raise ValueError("execution slots cannot be smaller than actual observation cost")
        if torch.any(self.execution_slots % 16 != 0):
            raise ValueError("execution slots must be packet aligned")
        if not torch.equal(self.padding_slots, self.execution_slots - self.actual_cost):
            raise ValueError("padding_slots must equal execution_slots - actual_cost")
        if torch.any(self.changed_mask != (self.effective_budget != 384)):
            raise ValueError("changed_mask must describe effective, not requested, tier changes")
        if self.predicted_total_utility.ndim != 0 or not bool(
            torch.isfinite(self.predicted_total_utility).item()
        ):
            raise ValueError("predicted_total_utility must be one finite scalar")
        if self.feasible and int(self.actual_cost.sum().item()) != int(
            self.target_actual_cost
        ):
            raise ValueError("feasible allocation must meet the exact actual-cost target")
        return self


def marginal_budget_accounting(
    valid_observations: torch.Tensor,
    requested_budget: int,
    *,
    baseline_budget: int = 384,
    packet_size: int = 16,
) -> dict[str, torch.Tensor]:
    """Canonicalize one requested tier under the frozen short-window contract."""

    valid = valid_observations.to(dtype=torch.long).reshape(-1)
    requested_budget = int(requested_budget)
    baseline_budget = int(baseline_budget)
    packet_size = int(packet_size)
    if torch.any(valid <= 0):
        raise ValueError("valid_observations must be positive")
    if requested_budget not in {256, 384, 512}:
        raise ValueError("requested marginal budget must be 256, 384, or 512")
    if baseline_budget != 384 or packet_size != 16:
        raise ValueError("the frozen marginal contract uses baseline 384 and packet size 16")

    requested = torch.full_like(valid, requested_budget)
    baseline = torch.full_like(valid, baseline_budget)
    actual = torch.minimum(valid, requested)
    baseline_actual = torch.minimum(valid, baseline)
    collapsed = (requested_budget != baseline_budget) & (actual == baseline_actual)
    effective = torch.where(collapsed, baseline, requested)
    distinct_nonbaseline = effective != baseline_budget
    packetized = ((actual + packet_size - 1) // packet_size) * packet_size
    execution = torch.where(distinct_nonbaseline, packetized, baseline)
    padding = execution - actual
    if torch.any(distinct_nonbaseline & (padding >= packet_size)):
        raise ValueError("distinct nonbaseline padding must be confined to the final packet")
    return {
        "requested_budget": requested,
        "actual_cost": actual,
        "baseline_actual_cost": baseline_actual,
        "effective_budget": effective,
        "execution_slots": execution,
        "padding_slots": padding,
        "collapsed_to_baseline": collapsed,
    }


class SignedTwoSidedMarginalUtilityHead(nn.Module):
    """Predict signed downgrade penalty and upgrade gain for one window."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("utility head dimensions must be positive")
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
        )

    def forward(self, window_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if window_features.ndim != 2 or int(window_features.shape[1]) != self.input_dim:
            raise ValueError(
                f"window_features must be [W,{self.input_dim}], got {tuple(window_features.shape)}"
            )
        values = self.net(window_features.float())
        return {
            "downgrade_penalty": values[:, 0],
            "upgrade_gain": values[:, 1],
        }


def validate_real_heavy_observation_tensor(
    observations: torch.Tensor,
    *,
    actual_observations: torch.Tensor | int,
    execution_slots: int,
    acquisition_mask: torch.Tensor,
    baseline_execution: bool = False,
) -> torch.Tensor:
    """Validate packetized execution separately from unique-observation cost."""

    execution_slots = int(execution_slots)
    if execution_slots <= 0 or execution_slots > 512 or execution_slots % 16 != 0:
        raise ValueError("execution_slots must be packet aligned and lie in (0, 512]")
    if not torch.is_tensor(observations) or observations.ndim not in {5, 6}:
        raise ValueError(
            "heavy observations must be [B,C,K,H,W] or [B,N,C,K,H,W]"
        )
    temporal_dim = 2 if observations.ndim == 5 else 3
    actual = int(observations.shape[temporal_dim])
    if actual != execution_slots:
        raise ValueError(
            f"heavy execution length {actual} does not match execution_slots {execution_slots}"
        )
    batch_size = int(observations.shape[0])
    counts = torch.as_tensor(
        actual_observations,
        device=observations.device,
        dtype=torch.long,
    ).reshape(-1)
    if counts.numel() == 1 and batch_size != 1:
        counts = counts.expand(batch_size)
    if counts.shape != (batch_size,) or torch.any(counts <= 0) or torch.any(counts > execution_slots):
        raise ValueError("actual_observations must contain one valid count per batch item")
    mask = acquisition_mask.to(device=observations.device, dtype=torch.bool)
    if mask.shape != (batch_size, execution_slots):
        raise ValueError("acquisition_mask must be [B, execution_slots]")
    expected_mask = torch.arange(execution_slots, device=observations.device)[None, :] < counts[:, None]
    if not torch.equal(mask, expected_mask):
        raise ValueError("acquisition_mask must be one active prefix followed by trailing padding")
    padding = execution_slots - counts
    if not baseline_execution and torch.any(padding >= 16):
        raise ValueError("distinct nonbaseline execution may pad only the final packet")
    mask_view = [batch_size] + [1] * (observations.ndim - 1)
    mask_view[temporal_dim] = execution_slots
    padded_values = observations.masked_select(~mask.view(mask_view).expand_as(observations))
    if padded_values.numel() and torch.any(padded_values != 0):
        raise ValueError("trailing heavy-execution padding must be exactly zero")
    return observations


def _masked_window_statistics(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or values.shape != valid.shape:
        raise ValueError("window statistic values and validity must be aligned [W,T]")
    masked = values.float().masked_fill(~valid, 0.0)
    count = valid.long().sum(dim=1).clamp_min(1).to(dtype=masked.dtype)
    mean = masked.sum(dim=1) / count
    centered = (masked - mean[:, None]).masked_fill(~valid, 0.0)
    std = torch.sqrt(centered.square().sum(dim=1) / count)
    maximum = masked.masked_fill(~valid, torch.finfo(masked.dtype).min).amax(dim=1)
    return torch.stack((mean, std, maximum), dim=1)


def build_frozen_scout_marginal_features(
    selector_outputs: Mapping[str, torch.Tensor],
    baseline_positions: torch.Tensor,
) -> torch.Tensor:
    """Pool only deploy-visible frozen-Scout evidence for the utility head.

    Ground truth, detector predictions, and counterfactual losses are deliberately
    not accepted as inputs.  The final four values describe temporal redundancy
    of the sealed K=384 observations; all other values come directly from the
    existing H65 Scout state.
    """

    valid = selector_outputs.get("valid_mask")
    hidden = selector_outputs.get("coarse_hidden_features")
    if hidden is None:
        hidden = selector_outputs.get("selection_features")
    if not torch.is_tensor(valid) or valid.ndim != 2:
        raise ValueError("frozen Scout marginal features require valid_mask [W,T]")
    valid = valid.bool()
    if not torch.is_tensor(hidden) or hidden.ndim != 3 or hidden.shape[:2] != valid.shape:
        raise ValueError("frozen Scout marginal features require hidden state [W,T,D]")
    if baseline_positions.ndim != 2 or baseline_positions.shape[0] != valid.shape[0]:
        raise ValueError("baseline_positions must be [W,K]")
    if baseline_positions.device != valid.device:
        raise ValueError("baseline positions and Scout state must share one device")
    if torch.any(valid.long().sum(dim=1) <= 0):
        raise ValueError("every marginal window must contain a valid observation")

    valid_float = valid.to(dtype=hidden.dtype)
    pooled_hidden = (hidden * valid_float[:, :, None]).sum(dim=1)
    pooled_hidden = pooled_hidden / valid_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled_hidden = pooled_hidden.float()

    scalar_names = ("p_action", "transition_score", "uncertainty")
    scalar_stats = []
    for name in scalar_names:
        values = selector_outputs.get(name)
        if not torch.is_tensor(values) or values.shape != valid.shape:
            raise ValueError(f"frozen Scout marginal features require {name} [W,T]")
        scalar_stats.append(_masked_window_statistics(values, valid))

    redundancy_rows = []
    valid_counts = valid.long().sum(dim=1)
    for row in range(int(valid.shape[0])):
        positions = baseline_positions[row]
        positions = positions[positions >= 0]
        if positions.numel() > 1 and torch.any(positions[1:] <= positions[:-1]):
            raise ValueError("baseline positions must be strictly time ordered")
        if positions.numel() and torch.any(positions >= valid_counts[row]):
            raise ValueError("baseline positions exceed the valid Scout window")
        denominator = max(1, int(valid_counts[row].item()) - 1)
        if positions.numel() > 1:
            gaps = (positions[1:] - positions[:-1]).float() / float(denominator)
            gap_mean = gaps.mean()
            gap_std = gaps.std(unbiased=False)
            gap_max = gaps.max()
        else:
            gap_mean = gap_std = gap_max = pooled_hidden.new_zeros(())
        coverage_ratio = pooled_hidden.new_tensor(
            float(positions.numel()) / float(max(1, int(valid_counts[row].item())))
        )
        redundancy_rows.append(torch.stack((gap_mean, gap_std, gap_max, coverage_ratio)))
    redundancy = torch.stack(redundancy_rows, dim=0).to(
        device=pooled_hidden.device,
        dtype=pooled_hidden.dtype,
    )
    return torch.cat((pooled_hidden, *scalar_stats, redundancy), dim=1).detach()


def allocate_video_budgets_exact(
    relative_utility: torch.Tensor,
    valid_observations: torch.Tensor,
    *,
    budget_levels: Sequence[int],
    baseline_budget: int,
    target_actual_cost: Optional[int] = None,
    max_changed_fraction: float = 0.5,
) -> VideoBudgetAllocation:
    """Maximize detached predicted utility under one exact per-video cost.

    This is a small control-plane dynamic program over the three preregistered
    budget tiers. It is intentionally independent for each video, so allocations
    cannot change with dataloader batch composition.
    """

    if relative_utility.ndim != 2:
        raise ValueError("relative_utility must be [W,J]")
    levels = tuple(int(value) for value in budget_levels)
    if not levels or tuple(sorted(set(levels))) != levels:
        raise ValueError("budget_levels must be unique and strictly increasing")
    if int(relative_utility.shape[1]) != len(levels):
        raise ValueError("relative_utility columns must match budget_levels")
    if int(baseline_budget) not in levels:
        raise ValueError("baseline_budget must be one of budget_levels")
    valid = valid_observations.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if valid.numel() != int(relative_utility.shape[0]) or torch.any(valid <= 0):
        raise ValueError("valid_observations must contain one positive count per window")
    utility = relative_utility.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(utility).all().item()):
        raise ValueError("relative_utility must be finite")
    fraction = float(max_changed_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("max_changed_fraction must lie in [0,1]")

    window_count = int(valid.numel())
    baseline_index = levels.index(int(baseline_budget))
    baseline_requested = torch.full((window_count,), int(baseline_budget), dtype=torch.long)
    baseline_actual = torch.minimum(baseline_requested, valid)
    frozen_target = int(baseline_actual.sum().item())
    target = frozen_target if target_actual_cost is None else int(target_actual_cost)
    if target != frozen_target:
        raise ValueError(
            "target_actual_cost must equal sum(min(valid_observations, baseline_budget))"
        )
    max_changed = int(window_count * fraction)

    # state[(actual_cost, changed_count)] = (utility, tuple(level_indices))
    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {
        (0, 0): (0.0, ())
    }
    for window_index in range(window_count):
        next_states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
        for (running_cost, changed_count), (running_utility, choices) in states.items():
            baseline_cost = int(baseline_actual[window_index].item())
            for level_index, level in enumerate(levels):
                actual = min(int(level), int(valid[window_index].item()))
                if level_index != baseline_index and actual == baseline_cost:
                    continue
                changed = changed_count + int(level_index != baseline_index)
                if changed > max_changed:
                    continue
                new_cost = running_cost + actual
                if new_cost > target:
                    continue
                score = running_utility + float(utility[window_index, level_index].item())
                key = (new_cost, changed)
                candidate = (score, choices + (level_index,))
                current = next_states.get(key)
                tie_rank = {baseline_index: 0, levels.index(256): 1, levels.index(512): 2}
                candidate_tie = tuple(tie_rank[index] for index in candidate[1])
                current_tie = (
                    tuple(tie_rank[index] for index in current[1])
                    if current is not None
                    else None
                )
                if current is None or score > current[0] + 1.0e-12 or (
                    abs(score - current[0]) <= 1.0e-12 and candidate_tie < current_tie
                ):
                    next_states[key] = candidate
        states = next_states
        if not states:
            break

    candidates = [
        (score, changed, choices)
        for (cost, changed), (score, choices) in states.items()
        if cost == target
    ]
    baseline_is_exact = int(baseline_actual.sum().item()) == target
    if not candidates:
        reason = "the all-K384 baseline could not be recovered at its own actual-cost target"
        baseline_accounting = marginal_budget_accounting(valid, int(baseline_budget))
        result = VideoBudgetAllocation(
            budget=baseline_requested.to(device=relative_utility.device),
            effective_budget=baseline_accounting["effective_budget"].to(device=relative_utility.device),
            actual_cost=baseline_actual.to(device=relative_utility.device),
            execution_slots=baseline_accounting["execution_slots"].to(device=relative_utility.device),
            padding_slots=baseline_accounting["padding_slots"].to(device=relative_utility.device),
            collapsed_to_baseline=baseline_accounting["collapsed_to_baseline"].to(device=relative_utility.device),
            changed_mask=torch.zeros(
                window_count, device=relative_utility.device, dtype=torch.bool
            ),
            predicted_total_utility=torch.zeros(
                (), device=relative_utility.device, dtype=relative_utility.dtype
            ),
            target_actual_cost=target,
            feasible=False,
            reason=reason,
        )
        return result.validate(window_count=window_count)

    # Prefer larger utility, then fewer changed windows, then the deterministic
    # lexicographically smaller tier sequence.
    tie_rank = {baseline_index: 0, levels.index(256): 1, levels.index(512): 2}
    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1],
            tuple(tie_rank[index] for index in item[2]),
        )
    )
    best_score, _changed, best_choices = candidates[0]
    if best_score <= 0.0 and baseline_is_exact:
        best_choices = (baseline_index,) * window_count
        best_score = 0.0
    requested = torch.tensor([levels[index] for index in best_choices], dtype=torch.long)
    accounting_rows = [
        marginal_budget_accounting(valid[index : index + 1], int(requested[index].item()))
        for index in range(window_count)
    ]
    actual = torch.cat([row["actual_cost"] for row in accounting_rows])
    effective = torch.cat([row["effective_budget"] for row in accounting_rows])
    execution = torch.cat([row["execution_slots"] for row in accounting_rows])
    padding = torch.cat([row["padding_slots"] for row in accounting_rows])
    collapsed = torch.cat([row["collapsed_to_baseline"] for row in accounting_rows])
    changed_mask = effective != int(baseline_budget)
    result = VideoBudgetAllocation(
        budget=requested.to(device=relative_utility.device),
        effective_budget=effective.to(device=relative_utility.device),
        actual_cost=actual.to(device=relative_utility.device),
        execution_slots=execution.to(device=relative_utility.device),
        padding_slots=padding.to(device=relative_utility.device),
        collapsed_to_baseline=collapsed.to(device=relative_utility.device),
        changed_mask=changed_mask.to(device=relative_utility.device),
        predicted_total_utility=torch.tensor(
            best_score,
            device=relative_utility.device,
            dtype=relative_utility.dtype,
        ),
        target_actual_cost=target,
        feasible=True,
        reason=(
            "all-K384 fallback: no positive exact nonbaseline transfer"
            if int(changed_mask.sum().item()) == 0
            else "exact actual-observation target satisfied"
        ),
    )
    return result.validate(window_count=window_count)


def allocate_equal_budget_marginal_reallocation(
    downgrade_penalty: torch.Tensor,
    upgrade_gain: torch.Tensor,
    valid_observations: torch.Tensor,
    *,
    lower_budget: int = 256,
    baseline_budget: int = 384,
    upper_budget: int = 512,
    max_changed_fraction: float = 0.5,
) -> VideoBudgetAllocation:
    """Allocate K in {lower, baseline, upper} at the baseline total cost."""

    if downgrade_penalty.ndim != 1 or upgrade_gain.shape != downgrade_penalty.shape:
        raise ValueError("downgrade_penalty and upgrade_gain must be aligned [W] tensors")
    relative = torch.stack(
        (-downgrade_penalty, torch.zeros_like(downgrade_penalty), upgrade_gain),
        dim=1,
    )
    return allocate_video_budgets_exact(
        relative,
        valid_observations,
        budget_levels=(int(lower_budget), int(baseline_budget), int(upper_budget)),
        baseline_budget=int(baseline_budget),
        target_actual_cost=None,
        max_changed_fraction=max_changed_fraction,
    )


class PrefixMarginalUtilityBudgetController(nn.Module):
    """Predict K(x) by stopping when marginal detector utility no longer pays for cost."""

    policy_name = "prefix_marginal_utility_stop"

    def __init__(
        self,
        hidden_dim: int,
        budget_min: int = 64,
        budget_max: int = 384,
        budget_multiple: int = 16,
        target_budget: Optional[float] = None,
        tau: float = 1.0,
        lambda_init: float = 1e-3,
        lambda_max: float = 10.0,
        dual_lr: float = 1e-2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.budget_min = int(budget_min)
        self.budget_max = int(budget_max)
        self.budget_multiple = int(budget_multiple)
        self.target_budget = float(self.budget_max if target_budget is None else target_budget)
        self.tau = float(tau)
        self.lambda_max = float(lambda_max)
        self.dual_lr = float(dual_lr)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.budget_min <= 0:
            raise ValueError("budget_min must be positive")
        if self.budget_max < self.budget_min:
            raise ValueError("budget_min must be <= budget_max")
        if self.budget_multiple <= 0:
            raise ValueError("budget_multiple must be positive")
        if (self.budget_max - self.budget_min) % self.budget_multiple != 0:
            raise ValueError("budget_multiple must divide budget_max - budget_min")
        if not (0.0 < self.target_budget <= float(self.budget_max)):
            raise ValueError("target_budget must lie in (0, budget_max]")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")

        self.num_extra_blocks = (self.budget_max - self.budget_min) // self.budget_multiple
        rank_count = max(1, self.num_extra_blocks)
        self.rank_embed = nn.Embedding(rank_count, self.hidden_dim)
        self.global_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.block_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.delta_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.register_buffer("lambda_dual", torch.tensor(float(lambda_init), dtype=torch.float32), persistent=True)

    def forward(
        self,
        selection_features: torch.Tensor,
        center_scores: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> DynamicBudgetDecision:
        if selection_features.ndim != 3:
            raise ValueError("selection_features must be [B,T,H]")
        if center_scores.ndim != 2:
            raise ValueError("center_scores must be [B,T]")
        if valid_mask.shape != center_scores.shape:
            raise ValueError("valid_mask must match center_scores")
        if selection_features.shape[:2] != center_scores.shape:
            raise ValueError("selection_features and center_scores must share [B,T]")
        if selection_features.shape[-1] != self.hidden_dim:
            raise ValueError(f"selection_features hidden dim must be {self.hidden_dim}")
        valid = valid_mask.bool()
        if torch.any(valid.long().sum(dim=1) <= 0):
            raise ValueError("each sample must contain at least one valid observation")

        dtype = selection_features.dtype
        device = selection_features.device
        batch_size, temporal_len, hidden_dim = selection_features.shape
        masked_scores = center_scores.masked_fill(~valid, torch.finfo(center_scores.dtype).min / 4.0)
        topk = min(max(1, self.budget_max), temporal_len)
        ranked_idx = torch.topk(masked_scores, k=topk, dim=1).indices
        ranked = torch.gather(
            selection_features,
            dim=1,
            index=ranked_idx[:, :, None].expand(-1, -1, hidden_dim),
        )
        ranked_valid = torch.gather(valid, dim=1, index=ranked_idx).to(dtype=dtype)
        ranked = ranked * ranked_valid[:, :, None]
        if topk < self.budget_max:
            pad = torch.zeros(batch_size, self.budget_max - topk, hidden_dim, device=device, dtype=dtype)
            ranked = torch.cat((ranked, pad), dim=1)
            ranked_valid = torch.cat(
                (ranked_valid, torch.zeros(batch_size, self.budget_max - topk, device=device, dtype=dtype)),
                dim=1,
            )

        valid_float = valid.to(dtype=dtype)
        global_feat = (selection_features * valid_float[:, :, None]).sum(dim=1) / valid_float.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        if self.num_extra_blocks == 0:
            empty = torch.zeros(batch_size, 0, device=device, dtype=dtype)
            budget = torch.full((batch_size,), self.budget_min, device=device, dtype=torch.long)
            decision = DynamicBudgetDecision(
                budget_hard=budget,
                budget_soft=budget.to(dtype=dtype),
                expected_cost=budget.to(dtype=dtype),
                continue_logits=empty,
                continue_soft=empty,
                continue_hard=empty,
                prefix_soft=empty,
                prefix_hard=empty,
                marginal_utility=empty,
                lambda_dual=self.lambda_dual.to(device=device, dtype=dtype),
                budget_min=self.budget_min,
                budget_max=self.budget_max,
                budget_multiple=self.budget_multiple,
                target_budget=self.target_budget,
            )
            return decision.validate(batch_size=batch_size)

        blocks = []
        for block_idx in range(self.num_extra_blocks):
            start = self.budget_min + block_idx * self.budget_multiple
            end = start + self.budget_multiple
            block = ranked[:, start:end, :]
            block_valid = ranked_valid[:, start:end]
            block_mean = block.sum(dim=1) / block_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            blocks.append(block_mean)
        block_features = torch.stack(blocks, dim=1)
        rank_ids = torch.arange(self.num_extra_blocks, device=device)
        fused = self.global_proj(global_feat)[:, None, :] + self.block_proj(block_features) + self.rank_embed(rank_ids)
        marginal = F.softplus(self.delta_head(fused).squeeze(-1))
        cost = self.lambda_dual.to(device=device, dtype=dtype).clamp(0.0, self.lambda_max)
        continue_logits = (marginal - cost) / self.tau
        continue_soft_raw = torch.sigmoid(continue_logits)
        continue_hard_raw = (continue_soft_raw >= 0.5).to(dtype=dtype)
        prefix_soft_raw = torch.cumprod(continue_soft_raw, dim=1)
        prefix_hard = torch.cumprod(continue_hard_raw, dim=1)
        prefix_st = prefix_hard + prefix_soft_raw - prefix_soft_raw.detach()
        soft_expected_k = float(self.budget_min) + float(self.budget_multiple) * prefix_soft_raw.sum(dim=1)
        budget_soft = float(self.budget_min) + float(self.budget_multiple) * prefix_st.sum(dim=1)
        budget_hard = self.budget_min + self.budget_multiple * prefix_hard.sum(dim=1).to(dtype=torch.long)
        budget_hard = budget_hard.clamp(min=self.budget_min, max=self.budget_max)
        budget_soft = budget_soft.clamp(min=float(self.budget_min), max=float(self.budget_max))
        soft_expected_k = soft_expected_k.clamp(min=float(self.budget_min), max=float(self.budget_max))
        decision = DynamicBudgetDecision(
            budget_hard=budget_hard,
            budget_soft=budget_soft,
            expected_cost=soft_expected_k,
            continue_logits=continue_logits,
            continue_soft=continue_soft_raw,
            continue_hard=continue_hard_raw,
            prefix_soft=prefix_st,
            prefix_hard=prefix_hard,
            marginal_utility=marginal,
            lambda_dual=self.lambda_dual.to(device=device, dtype=dtype),
            budget_min=self.budget_min,
            budget_max=self.budget_max,
            budget_multiple=self.budget_multiple,
            target_budget=self.target_budget,
        )
        return decision.validate(batch_size=batch_size)

    @torch.no_grad()
    def update_dual(self, observed_mean_budget: torch.Tensor | float) -> torch.Tensor:
        value = torch.as_tensor(observed_mean_budget, device=self.lambda_dual.device, dtype=self.lambda_dual.dtype)
        normalized_residual = (value - float(self.target_budget)) / max(float(self.target_budget), 1.0)
        self.lambda_dual.add_(self.dual_lr * normalized_residual)
        self.lambda_dual.clamp_(0.0, self.lambda_max)
        return self.lambda_dual
