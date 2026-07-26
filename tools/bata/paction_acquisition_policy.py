from __future__ import annotations

import bisect
import copy
import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tools.bata import paction_budget_contract


if os.name != "nt":
    from torch.nn import Module as _TorchModuleBase
else:  # Keep pure-Python imports usable on Windows machines with broken torch DLLs.
    _TorchModuleBase = object


PACTION_FEATURE_NAMES = (
    "p_action",
    "delta_p_action",
    "abs_delta_p_action",
    "entropy",
    "uncertainty",
    "time",
    "valid",
)
DEFAULT_DYNAMIC_BUDGET_BUCKETS = (128, 192, 256, 320, 384, 512, 768)
LEARNED_FIXED_STRATEGY = "learned_paction_gap_loss_value"
LEARNED_DYNAMIC_STRATEGY = "learned_paction_gap_loss_dynamic_budget"
DEFAULT_GAP_LOSS_MAX_GAP = 3
DEFAULT_BOUNDARY_LOSS_RADIUS = 1
DEFAULT_POLICY_LOSS_TERMS = {
    "value_transport": 1.0,
    "boundary_miss": 2.0,
    "large_gap": 1.5,
    "temporal_hole": 1.0,
    "budget": 1.0,
    "redundancy": 0.1,
}


@dataclass(frozen=True)
class BudgetOracleDecision:
    budget: int
    reason: str
    metrics: Mapping[str, Any]


def feature_index(name: str) -> int:
    return PACTION_FEATURE_NAMES.index(str(name))


def _clamp_probability(value: Any) -> float:
    prob = float(value)
    if math.isnan(prob):
        return 0.0
    return max(0.0, min(1.0, prob))


def _binary_entropy(probability: float) -> float:
    prob = _clamp_probability(probability)
    if prob <= 0.0 or prob >= 1.0:
        return 0.0
    return -(prob * math.log(prob) + (1.0 - prob) * math.log(1.0 - prob)) / math.log(2.0)


def _valid_mask(valid: Sequence[Any] | None, length: int) -> list[bool]:
    if valid is None:
        return [True] * int(length)
    if len(valid) != int(length):
        raise ValueError("valid mask length must match p_action length")
    return [bool(item) for item in valid]


def build_paction_feature_matrix(
    p_action: Sequence[Any],
    *,
    valid: Sequence[Any] | None = None,
) -> list[list[float]]:
    probabilities = [_clamp_probability(item) for item in p_action]
    valid_mask = _valid_mask(valid, len(probabilities))
    valid_indices = [idx for idx, is_valid in enumerate(valid_mask) if is_valid]
    valid_rank = {idx: rank for rank, idx in enumerate(valid_indices)}
    valid_denominator = max(1, len(valid_indices) - 1)

    features: list[list[float]] = []
    previous_valid_probability: float | None = None
    for idx, probability in enumerate(probabilities):
        is_valid = bool(valid_mask[idx])
        if is_valid and previous_valid_probability is not None:
            delta = round(probability - previous_valid_probability, 12)
        else:
            delta = 0.0
        if is_valid:
            previous_valid_probability = probability
        time_value = float(valid_rank[idx]) / float(valid_denominator) if is_valid and valid_indices else 0.0
        features.append(
            [
                probability if is_valid else 0.0,
                delta if is_valid else 0.0,
                abs(delta) if is_valid else 0.0,
                _binary_entropy(probability) if is_valid else 0.0,
                (1.0 - abs(2.0 * probability - 1.0)) if is_valid else 0.0,
                time_value,
                1.0 if is_valid else 0.0,
            ]
        )
    return features


def _unselected_hole_runs(selected: set[int], valid_indices: Sequence[int]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    run_start: int | None = None
    previous = -1
    for idx in valid_indices:
        idx = int(idx)
        if idx in selected:
            if run_start is not None:
                runs.append((int(run_start), int(previous), int(previous - run_start + 1)))
                run_start = None
        elif run_start is None:
            run_start = idx
        previous = idx
    if run_start is not None:
        runs.append((int(run_start), int(previous), int(previous - run_start + 1)))
    return runs


def _max_unselected_hole(selected: set[int], valid_indices: Sequence[int]) -> int:
    runs = _unselected_hole_runs(selected, valid_indices)
    return max((length for _, _, length in runs), default=0)


def _selected_valid_rank_map(valid_indices: Sequence[int]) -> dict[int, int]:
    return {int(idx): rank for rank, idx in enumerate(valid_indices)}


def _removal_hole_length(
    selected_idx: int,
    *,
    selected_ranks: Sequence[int],
    selected_rank_set: set[int],
    valid_count: int,
) -> int:
    rank = int(selected_idx)
    if rank not in selected_rank_set:
        raise ValueError(f"selected rank {rank} is not selected")
    offset = bisect.bisect_left(selected_ranks, rank)
    left_rank = int(selected_ranks[offset - 1]) if offset > 0 else None
    right_rank = int(selected_ranks[offset + 1]) if offset + 1 < len(selected_ranks) else None
    if left_rank is None and right_rank is None:
        return int(valid_count)
    if left_rank is None:
        return int(right_rank)
    if right_rank is None:
        return int(valid_count - 1 - left_rank)
    return int(right_rank - left_rank - 1)


def _choose_gap_repair_removal(
    *,
    selected: set[int],
    added: int,
    valid_rank: Mapping[int, int],
    valid_count: int,
    values: Sequence[Any],
    max_hole: int,
) -> int | None:
    selected_ranks = sorted(valid_rank[int(idx)] for idx in selected if int(idx) in valid_rank)
    selected_rank_set = set(selected_ranks)
    safe: list[tuple[float, int]] = []
    fallback: list[tuple[int, float, int]] = []
    for candidate in selected:
        candidate = int(candidate)
        if candidate == int(added) or candidate not in valid_rank:
            continue
        removal_hole = _removal_hole_length(
            valid_rank[candidate],
            selected_ranks=selected_ranks,
            selected_rank_set=selected_rank_set,
            valid_count=int(valid_count),
        )
        score = float(values[candidate])
        if removal_hole <= int(max_hole):
            safe.append((score, candidate))
        fallback.append((int(removal_hole), score, candidate))
    if safe:
        _score, remove_idx = min(safe, key=lambda item: (item[0], -item[1]))
        return int(remove_idx)
    if fallback:
        _hole, _score, remove_idx = min(fallback, key=lambda item: (item[0], item[1], -item[2]))
        return int(remove_idx)
    return None


def _score_key(values: Sequence[Any], idx: int) -> tuple[float, int]:
    return (float(values[int(idx)]), -int(idx))


def constrained_topk(
    values: Sequence[Any],
    *,
    valid: Sequence[Any] | None = None,
    budget: int,
    max_unselected_hole: int | None = None,
) -> list[int]:
    valid_mask = _valid_mask(valid, len(values))
    if int(budget) <= 0:
        return []
    valid_indices = [idx for idx, is_valid in enumerate(valid_mask) if is_valid]
    ranked = sorted(valid_indices, key=lambda idx: (float(values[idx]), -idx), reverse=True)
    selected = set(ranked[: min(int(budget), len(ranked))])
    if max_unselected_hole is None or not selected:
        return sorted(selected)
    max_hole = int(max_unselected_hole)
    if max_hole < 0:
        raise ValueError("max_unselected_hole must be non-negative")
    if len(selected) >= len(valid_indices):
        return sorted(selected)
    minimum_required = (len(valid_indices) + max_hole) // (max_hole + 1)
    if len(selected) < minimum_required:
        return sorted(selected)

    valid_rank = _selected_valid_rank_map(valid_indices)
    valid_count = len(valid_indices)
    for _ in range(len(valid_indices) + 1):
        violating = [
            run
            for run in _unselected_hole_runs(selected, valid_indices)
            if int(run[2]) > max_hole
        ]
        if not violating:
            break
        start, end, _length = max(violating, key=lambda item: (int(item[2]), -int(item[0])))
        feasible_start = max(int(start), int(end) - max_hole)
        feasible_end = min(int(end), int(start) + max_hole)
        repair_candidates = [
            idx
            for idx in valid_indices
            if feasible_start <= int(idx) <= feasible_end and int(idx) not in selected
        ]
        if not repair_candidates:
            repair_candidates = [
                idx
                for idx in valid_indices
                if int(start) <= int(idx) <= int(end) and int(idx) not in selected
            ]
        if not repair_candidates:
            break
        added = max(repair_candidates, key=lambda idx: _score_key(values, int(idx)))
        selected.add(int(added))
        if len(selected) <= int(budget):
            continue

        remove_idx = _choose_gap_repair_removal(
            selected=selected,
            added=int(added),
            valid_rank=valid_rank,
            valid_count=valid_count,
            values=values,
            max_hole=max_hole,
        )
        if remove_idx is None:
            selected.remove(int(added))
            break
        selected.remove(int(remove_idx))
    return sorted(selected)


def temporal_gap_hole_loss_from_probabilities(
    selection_probabilities: Sequence[Any],
    *,
    valid: Sequence[Any] | None = None,
    max_gap: int = DEFAULT_GAP_LOSS_MAX_GAP,
    eps: float = 1e-6,
) -> float:
    """Surrogate loss: every local window should contain learned selection mass.

    A zero-valued window of length ``max_gap + 1`` means a hard decoded policy can
    create a gap larger than ``max_gap``. The loss is intentionally a training
    pressure, not a decoder-side uniform fill or guard.
    """

    valid_mask = _valid_mask(valid, len(selection_probabilities))
    max_gap = int(max_gap)
    if max_gap <= 0:
        return 0.0
    probabilities = [
        _clamp_probability(value) if is_valid else 0.0
        for value, is_valid in zip(selection_probabilities, valid_mask)
    ]
    valid_indices = [idx for idx, is_valid in enumerate(valid_mask) if is_valid]
    if not valid_indices:
        return 0.0
    window_width = max_gap + 1
    if len(valid_indices) <= window_width:
        coverage = 1.0
        for idx in valid_indices:
            coverage *= 1.0 - probabilities[idx]
        coverage = 1.0 - coverage
        return -math.log(max(float(eps), coverage))
    losses: list[float] = []
    for offset in range(0, len(valid_indices) - window_width + 1):
        window = valid_indices[offset : offset + window_width]
        no_select_probability = 1.0
        for idx in window:
            no_select_probability *= 1.0 - probabilities[idx]
        coverage = 1.0 - no_select_probability
        losses.append(-math.log(max(float(eps), coverage)))
    return sum(losses) / float(len(losses)) if losses else 0.0


def boundary_miss_loss_from_probabilities(
    selection_probabilities: Sequence[Any],
    boundary_positions: Sequence[Any],
    *,
    valid: Sequence[Any] | None = None,
    radius: int = DEFAULT_BOUNDARY_LOSS_RADIUS,
    eps: float = 1e-6,
) -> float:
    valid_mask = _valid_mask(valid, len(selection_probabilities))
    probabilities = [
        _clamp_probability(value) if is_valid else 0.0
        for value, is_valid in zip(selection_probabilities, valid_mask)
    ]
    valid_len = len(probabilities)
    losses: list[float] = []
    for raw_boundary in boundary_positions:
        boundary = int(round(float(raw_boundary)))
        left = max(0, boundary - int(radius))
        right = min(valid_len - 1, boundary + int(radius))
        support = [idx for idx in range(left, right + 1) if valid_mask[idx]]
        if not support:
            continue
        no_select_probability = 1.0
        for idx in support:
            no_select_probability *= 1.0 - probabilities[idx]
        coverage = 1.0 - no_select_probability
        losses.append(-math.log(max(float(eps), coverage)))
    return sum(losses) / float(len(losses)) if losses else 0.0


def paction_gap_loss_training_objective(
    policy_outputs: Mapping[str, Any],
    *,
    action_target: Any | None = None,
    boundary_target: Any | None = None,
    valid: Any | None = None,
    target_budget: Any | None = None,
    gap_loss_max_gap: int = DEFAULT_GAP_LOSS_MAX_GAP,
    loss_terms: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Torch objective for the learned p_action acquisition route.

    This is the train-time counterpart of the strict ledger decoder: no uniform
    scaffold or conversion-time fill is injected. Large holes are discouraged by
    requiring every local window to carry learned selection probability mass.
    """

    import torch
    import torch.nn.functional as F

    weights = dict(DEFAULT_POLICY_LOSS_TERMS)
    if loss_terms is not None:
        weights.update({str(key): float(value) for key, value in loss_terms.items()})
    frame_logits = policy_outputs.get("frame_value")
    if frame_logits is None:
        raise ValueError("policy_outputs must contain frame_value logits")
    frame_logits = frame_logits.float()
    if valid is None:
        valid_mask = torch.ones_like(frame_logits, dtype=torch.bool)
    else:
        valid_mask = valid.to(device=frame_logits.device).bool()
        if tuple(valid_mask.shape) != tuple(frame_logits.shape):
            raise ValueError("valid mask must match frame_value shape")
    probabilities = torch.sigmoid(frame_logits).masked_fill(~valid_mask, 0.0)
    losses: dict[str, Any] = {}
    total = frame_logits.sum() * 0.0

    if action_target is not None and float(weights.get("value_transport", 0.0)) > 0.0:
        action = action_target.to(device=frame_logits.device).float()
        if tuple(action.shape) != tuple(frame_logits.shape):
            raise ValueError("action_target must match frame_value shape")
        if bool(valid_mask.any().item()):
            value_loss = F.binary_cross_entropy_with_logits(frame_logits[valid_mask], action[valid_mask])
        else:
            value_loss = frame_logits.sum() * 0.0
        losses["value_transport_loss"] = value_loss
        total = total + value_loss * float(weights["value_transport"])

    if boundary_target is not None and float(weights.get("boundary_miss", 0.0)) > 0.0:
        boundary = boundary_target.to(device=frame_logits.device).float()
        if tuple(boundary.shape) != tuple(frame_logits.shape):
            raise ValueError("boundary_target must match frame_value shape")
        radius = max(0, int(DEFAULT_BOUNDARY_LOSS_RADIUS))
        kernel = 2 * radius + 1
        local_mass = F.max_pool1d(
            probabilities.unsqueeze(1),
            kernel_size=kernel,
            stride=1,
            padding=radius,
        ).squeeze(1)
        boundary_mask = (boundary > 0.0) & valid_mask
        if bool(boundary_mask.any().item()):
            boundary_loss = -torch.log(local_mass[boundary_mask].clamp_min(1e-6)).mean()
        else:
            boundary_loss = frame_logits.sum() * 0.0
        losses["boundary_miss_loss"] = boundary_loss
        total = total + boundary_loss * float(weights["boundary_miss"])

    gap_weight = float(weights.get("large_gap", 0.0)) + float(weights.get("temporal_hole", 0.0))
    if gap_weight > 0.0:
        window_width = max(1, int(gap_loss_max_gap) + 1)
        if probabilities.shape[-1] >= window_width:
            window_mass = F.avg_pool1d(
                probabilities.unsqueeze(1),
                kernel_size=window_width,
                stride=1,
            ).squeeze(1) * float(window_width)
            window_valid_count = F.avg_pool1d(
                valid_mask.float().unsqueeze(1),
                kernel_size=window_width,
                stride=1,
            ).squeeze(1) * float(window_width)
            valid_windows = window_valid_count >= float(window_width)
            if bool(valid_windows.any().item()):
                gap_loss = F.relu(1.0 - window_mass[valid_windows]).square().mean()
            else:
                window_mass = probabilities.sum(dim=-1)
                gap_loss = F.relu(1.0 - window_mass).square().mean()
        else:
            window_mass = probabilities.sum(dim=-1)
            gap_loss = F.relu(1.0 - window_mass).square().mean()
        losses["large_gap_hole_loss"] = gap_loss
        total = total + gap_loss * gap_weight

    if target_budget is not None and float(weights.get("budget", 0.0)) > 0.0:
        target = torch.as_tensor(target_budget, dtype=probabilities.dtype, device=probabilities.device)
        if target.ndim == 0:
            target = target.expand(probabilities.shape[0])
        selected_mass = probabilities.sum(dim=-1)
        budget_loss = ((selected_mass - target.float()) / target.float().clamp_min(1.0)).square().mean()
        losses["budget_loss"] = budget_loss
        total = total + budget_loss * float(weights["budget"])

    if float(weights.get("redundancy", 0.0)) > 0.0:
        smoothness = (probabilities[:, 1:] - probabilities[:, :-1]).abs()
        smooth_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
        redundancy_loss = smoothness[smooth_valid].mean() if bool(smooth_valid.any().item()) else frame_logits.sum() * 0.0
        losses["redundancy_loss"] = redundancy_loss
        total = total + redundancy_loss * float(weights["redundancy"])

    losses["total_loss"] = total
    losses["gap_control"] = "learned_gap_hole_loss_no_uniform_fill"
    return losses


def decode_budget_from_scores(
    scores: Sequence[Any],
    budget_buckets: Sequence[Any] = DEFAULT_DYNAMIC_BUDGET_BUCKETS,
    *,
    valid_len: int,
    min_budget: int | None = None,
    max_budget: int | None = None,
) -> int:
    if not scores:
        raise ValueError("scores must not be empty")
    if len(scores) != len(budget_buckets):
        raise ValueError("scores and budget_buckets must have the same length")
    best_idx = max(range(len(scores)), key=lambda idx: (float(scores[idx]), -idx))
    budget = int(budget_buckets[best_idx])
    if min_budget is not None:
        budget = max(int(min_budget), budget)
    if max_budget is not None:
        budget = min(int(max_budget), budget)
    budget = min(max(0, budget), max(0, int(valid_len)))
    return int(budget)


def short_valid_ratio_budget(required_count: int, *, valid_len: int, dense_len: int) -> int:
    expected = paction_budget_contract.expected_selected_count(
        int(required_count),
        valid_len=int(valid_len),
        dense_len=int(dense_len),
        allow_short_valid_ratio_count=True,
    )
    return 0 if expected is None else int(expected)


def oracle_budget_from_quality_curve(
    quality_by_budget: Mapping[Any, Mapping[str, Any]],
    *,
    min_boundary_support: float,
    min_action_coverage: float,
    boundary_key: str = "boundary_support_r1",
    action_key: str = "action_positive_coverage",
) -> BudgetOracleDecision:
    if not quality_by_budget:
        raise ValueError("quality_by_budget must not be empty")
    fallback_budget = None
    fallback_metrics: Mapping[str, Any] = {}
    for raw_budget in sorted(quality_by_budget, key=lambda item: int(item)):
        budget = int(raw_budget)
        metrics = quality_by_budget[raw_budget]
        fallback_budget = budget
        fallback_metrics = metrics
        boundary = metrics.get(boundary_key)
        action = metrics.get(action_key)
        if boundary is None or action is None:
            continue
        if float(boundary) >= float(min_boundary_support) and float(action) >= float(min_action_coverage):
            return BudgetOracleDecision(budget=budget, reason="meets_quality_target", metrics=dict(metrics))
    assert fallback_budget is not None
    return BudgetOracleDecision(budget=int(fallback_budget), reason="fallback_max_budget", metrics=dict(fallback_metrics))


def add_policy_decision_to_sample_row(
    row: Mapping[str, Any],
    *,
    frame_values: Sequence[Any],
    fixed_budget: int,
    dynamic_budget_scores: Sequence[Any],
    dynamic_budget_buckets: Sequence[Any] = DEFAULT_DYNAMIC_BUDGET_BUCKETS,
    fixed_strategy_name: str = LEARNED_FIXED_STRATEGY,
    dynamic_strategy_name: str = LEARNED_DYNAMIC_STRATEGY,
    gap_loss_max_gap: int = DEFAULT_GAP_LOSS_MAX_GAP,
    max_unselected_hole: int | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(dict(row))
    valid_len = int(out.get("valid_len") or out.get("dense_len") or len(frame_values))
    dense_len = int(out.get("dense_len") or len(frame_values))
    valid = [idx < valid_len for idx in range(len(frame_values))]
    requested_fixed_budget = int(fixed_budget)
    fixed_budget = short_valid_ratio_budget(
        requested_fixed_budget,
        valid_len=valid_len,
        dense_len=dense_len,
    )
    dynamic_budget = decode_budget_from_scores(dynamic_budget_scores, dynamic_budget_buckets, valid_len=valid_len)
    strategies = dict(out.get("strategy_selected_positions") or {})
    fixed_selected = constrained_topk(
        frame_values,
        valid=valid,
        budget=int(fixed_budget),
        max_unselected_hole=max_unselected_hole,
    )
    dynamic_selected = constrained_topk(
        frame_values,
        valid=valid,
        budget=int(dynamic_budget),
        max_unselected_hole=max_unselected_hole,
    )
    strategies[str(fixed_strategy_name)] = fixed_selected
    strategies[str(dynamic_strategy_name)] = dynamic_selected
    fixed_mask = [1.0 if idx in set(fixed_selected) else 0.0 for idx in range(len(frame_values))]
    dynamic_mask = [1.0 if idx in set(dynamic_selected) else 0.0 for idx in range(len(frame_values))]
    out["strategy_selected_positions"] = strategies
    gap_control = (
        "learned_score_constrained_gap_no_uniform_fill"
        if max_unselected_hole is not None
        else "learned_gap_hole_loss_no_uniform_fill"
    )
    out["paction_policy"] = {
        "selection_signal": "p_action_gap_loss_policy_value",
        "fixed_strategy": str(fixed_strategy_name),
        "dynamic_strategy": str(dynamic_strategy_name),
        "requested_fixed_budget": int(requested_fixed_budget),
        "fixed_budget": int(fixed_budget),
        "fixed_budget_dense_len": int(dense_len),
        "fixed_budget_uses_short_valid_ratio_count": int(valid_len) < int(dense_len),
        "dynamic_budget": int(dynamic_budget),
        "dynamic_budget_buckets": [int(item) for item in dynamic_budget_buckets],
        "gap_control": gap_control,
        "gap_loss_max_gap": int(gap_loss_max_gap),
        "max_unselected_hole": None if max_unselected_hole is None else int(max_unselected_hole),
        "loss_terms": dict(DEFAULT_POLICY_LOSS_TERMS),
        "uses_uniform_scaffold": False,
        "uses_uniform_fill": False,
        "fixed_decoded_gap_hole_loss": temporal_gap_hole_loss_from_probabilities(
            fixed_mask,
            valid=valid,
            max_gap=int(gap_loss_max_gap),
        ),
        "dynamic_decoded_gap_hole_loss": temporal_gap_hole_loss_from_probabilities(
            dynamic_mask,
            valid=valid,
            max_gap=int(gap_loss_max_gap),
        ),
    }
    return out


class PActionDynamicAcquisitionPolicy(_TorchModuleBase):
    """Tiny torch policy with frame-value and dynamic-budget heads."""

    def __init__(
        self,
        *,
        input_dim: int = len(PACTION_FEATURE_NAMES),
        hidden_dim: int = 64,
        num_layers: int = 3,
        budget_buckets: Sequence[int] = DEFAULT_DYNAMIC_BUDGET_BUCKETS,
        dropout: float = 0.10,
    ) -> None:
        import torch
        import torch.nn as nn

        super().__init__()
        self.torch = torch
        self.nn = nn
        self.budget_buckets = tuple(int(item) for item in budget_buckets)
        layers: list[Any] = [
            nn.Conv1d(int(input_dim), int(hidden_dim), kernel_size=1, bias=False),
            nn.BatchNorm1d(int(hidden_dim)),
            nn.SiLU(inplace=True),
        ]
        for layer_idx in range(max(1, int(num_layers))):
            dilation = 2 ** layer_idx
            layers.extend(
                [
                    nn.Conv1d(
                        int(hidden_dim),
                        int(hidden_dim),
                        kernel_size=3,
                        padding=int(dilation),
                        dilation=int(dilation),
                        groups=1,
                        bias=False,
                    ),
                    nn.BatchNorm1d(int(hidden_dim)),
                    nn.SiLU(inplace=True),
                    nn.Dropout(float(dropout)),
                ]
            )
        self.encoder = nn.Sequential(*layers)
        self.frame_value_head = nn.Conv1d(int(hidden_dim), 1, kernel_size=1)
        self.budget_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_dim), len(self.budget_buckets)),
        )

    def forward(self, features: Any, valid: Any | None = None) -> dict[str, Any]:
        torch = self.torch
        if features.ndim != 3:
            raise ValueError(f"paction policy expects [B,T,F], got {tuple(features.shape)}")
        encoded = self.encoder(features.float().transpose(1, 2))
        frame_value = self.frame_value_head(encoded).squeeze(1)
        if valid is not None:
            valid = valid.to(device=frame_value.device).bool()
            frame_value = frame_value.masked_fill(~valid, 0.0)
            weights = valid.float()
            pooled = (encoded * weights[:, None, :]).sum(dim=-1) / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        else:
            pooled = encoded.mean(dim=-1)
        budget_logits = self.budget_head(pooled)
        return {
            "frame_value": frame_value,
            "budget_logits": budget_logits,
            "budget_buckets": torch.tensor(self.budget_buckets, dtype=torch.long, device=budget_logits.device),
        }
