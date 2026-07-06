from __future__ import annotations

import copy
import math
import os
from typing import Any, Mapping, Sequence

from tools.bata import paction_acquisition_policy as base_policy


if os.name != "nt":
    from torch.nn import Module as _TorchModuleBase
else:
    _TorchModuleBase = object


GAS_VT_FIXED_384_STRATEGY = "gas_vt_fixed_384"
GAS_VT_FIXED_768_STRATEGY = "gas_vt_fixed_768"
GAS_VT_DYNAMIC_STRATEGY = "gas_vt_dynamic"
GAS_VT_STRATEGIES = (GAS_VT_FIXED_384_STRATEGY, GAS_VT_FIXED_768_STRATEGY, GAS_VT_DYNAMIC_STRATEGY)
GAS_VT_CHECKPOINT_POLICY_SOURCE = "learned_paction_gas_vt_policy_checkpoint"
GAS_VT_BOOTSTRAP_POLICY_SOURCE = "bootstrap_gas_vt_surrogate_policy"
DEFAULT_GAS_VT_DYNAMIC_BUDGET_BUCKETS = base_policy.DEFAULT_DYNAMIC_BUDGET_BUCKETS
DEFAULT_GAS_VT_LOSS_TERMS = {
    "value_bce": 1.0,
    "boundary_coverage": 2.0,
    "boundary_bracket": 1.0,
    "action_interior_bin": 1.0,
    "cvar_max_hole": 1.5,
    "budget": 1.0,
    "paction_dependence": 0.05,
}
GAS_VT_FEATURE_NAMES = base_policy.PACTION_FEATURE_NAMES + (
    "local_density",
    "local_change",
    "distance_to_last_selection",
    "remaining_budget",
    "remaining_time",
    "budget_pressure",
    "gap_urgency",
)


def feature_index(name: str) -> int:
    return GAS_VT_FEATURE_NAMES.index(str(name))


def _valid_mask(valid: Sequence[Any] | None, length: int) -> list[bool]:
    if valid is None:
        return [True] * int(length)
    if len(valid) != int(length):
        raise ValueError("valid mask length must match p_action length")
    return [bool(item) for item in valid]


def _clamp(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(out):
        return 0.0
    return max(0.0, min(1.0, out))


def build_gap_aware_feature_matrix(
    p_action: Sequence[Any],
    *,
    valid: Sequence[Any] | None = None,
    selected_so_far: Sequence[int] | None = None,
    target_budget: int | None = None,
    local_radius: int = 2,
    max_gap: int = base_policy.DEFAULT_GAP_LOSS_MAX_GAP,
) -> list[list[float]]:
    base_features = base_policy.build_paction_feature_matrix(p_action, valid=valid)
    valid_mask = _valid_mask(valid, len(base_features))
    valid_indices = [idx for idx, is_valid in enumerate(valid_mask) if is_valid]
    valid_rank = {idx: rank for rank, idx in enumerate(valid_indices)}
    selected = sorted({int(item) for item in (selected_so_far or []) if 0 <= int(item) < len(base_features)})
    selected_count = len(selected)
    budget = max(1, int(target_budget or max(1, len(valid_indices))))
    p_values = [_clamp(item) if valid_mask[idx] else 0.0 for idx, item in enumerate(p_action)]
    out: list[list[float]] = []
    for idx, row in enumerate(base_features):
        if not valid_mask[idx]:
            out.append(list(row) + [0.0] * (len(GAS_VT_FEATURE_NAMES) - len(base_policy.PACTION_FEATURE_NAMES)))
            continue
        left = max(0, idx - int(local_radius))
        right = min(len(p_values) - 1, idx + int(local_radius))
        local = [p_values[pos] for pos in range(left, right + 1) if valid_mask[pos]]
        local_density = sum(local) / float(len(local)) if local else 0.0
        local_change = max(local) - min(local) if local else 0.0
        previous = [pos for pos in selected if pos <= idx]
        if previous:
            distance = float(idx - previous[-1])
        elif selected:
            distance = float(min(abs(idx - pos) for pos in selected))
        else:
            distance = float(valid_rank[idx] + 1)
        rank = int(valid_rank[idx])
        remaining_time_count = max(0, len(valid_indices) - rank)
        remaining_budget_count = max(0, budget - selected_count)
        remaining_budget = float(remaining_budget_count) / float(budget)
        remaining_time = float(remaining_time_count) / float(max(1, len(valid_indices)))
        budget_pressure = float(remaining_budget_count) / float(max(1, remaining_time_count))
        gap_urgency = min(1.0, distance / float(max(1, max_gap + 1)))
        out.append(
            list(row)
            + [
                float(local_density),
                float(local_change),
                float(distance),
                float(remaining_budget),
                float(remaining_time),
                float(budget_pressure),
                float(gap_urgency),
            ]
        )
    return out


def max_unselected_hole(selected: Sequence[int], *, valid_len: int) -> int:
    selected_set = {int(item) for item in selected}
    current = 0
    max_hole = 0
    for idx in range(max(0, int(valid_len))):
        if idx in selected_set:
            max_hole = max(max_hole, current)
            current = 0
        else:
            current += 1
    return max(max_hole, current)


def hard_gap_aware_topk(
    frame_values: Sequence[Any],
    *,
    valid: Sequence[Any] | None = None,
    budget: int,
    max_unselected_hole: int | None = None,
) -> list[int]:
    return base_policy.constrained_topk(
        frame_values,
        valid=valid,
        budget=int(budget),
        max_unselected_hole=max_unselected_hole,
    )


def _strategy_budget_for_name(name: str, fixed_budgets: Sequence[int]) -> int:
    budgets = [int(item) for item in fixed_budgets]
    if str(name) == GAS_VT_FIXED_384_STRATEGY:
        return budgets[0]
    if str(name) == GAS_VT_FIXED_768_STRATEGY:
        return budgets[1] if len(budgets) > 1 else budgets[0]
    raise ValueError(f"unknown GAS-VT fixed strategy: {name}")


def add_gas_vt_decision_to_sample_row(
    row: Mapping[str, Any],
    *,
    frame_values: Sequence[Any],
    fixed_budgets: Sequence[int] = (384, 768),
    dynamic_budget_scores: Sequence[Any],
    dynamic_budget_buckets: Sequence[Any] = DEFAULT_GAS_VT_DYNAMIC_BUDGET_BUCKETS,
    max_unselected_hole: int | None = None,
    source: str | None = None,
    checkpoint_path: str | None = None,
    checkpoint_sha256: str | None = None,
    source_jsonl_sha256: str | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(dict(row))
    valid_len = int(out.get("valid_len") or out.get("dense_len") or len(frame_values))
    dense_len = int(out.get("dense_len") or len(frame_values))
    valid = [idx < valid_len for idx in range(len(frame_values))]
    strategies = dict(out.get("strategy_selected_positions") or {})
    for strategy_name in (GAS_VT_FIXED_384_STRATEGY, GAS_VT_FIXED_768_STRATEGY):
        requested = _strategy_budget_for_name(strategy_name, fixed_budgets)
        budget = base_policy.short_valid_ratio_budget(requested, valid_len=valid_len, dense_len=dense_len)
        strategies[strategy_name] = hard_gap_aware_topk(
            frame_values,
            valid=valid,
            budget=budget,
            max_unselected_hole=max_unselected_hole,
        )
    dynamic_budget = base_policy.decode_budget_from_scores(
        dynamic_budget_scores,
        dynamic_budget_buckets,
        valid_len=valid_len,
    )
    strategies[GAS_VT_DYNAMIC_STRATEGY] = hard_gap_aware_topk(
        frame_values,
        valid=valid,
        budget=dynamic_budget,
        max_unselected_hole=max_unselected_hole,
    )
    out["strategy_selected_positions"] = strategies
    out["gas_vt_policy"] = {
        "policy_family": "GAS-VT",
        "selection_signal": "p_action_gap_aware_value_transport",
        "source": source,
        "decode_mode": "hard_gap_aware_topk",
        "fixed_strategies": [GAS_VT_FIXED_384_STRATEGY, GAS_VT_FIXED_768_STRATEGY],
        "dynamic_strategy": GAS_VT_DYNAMIC_STRATEGY,
        "fixed_budgets": [int(item) for item in fixed_budgets],
        "dynamic_budget": int(dynamic_budget),
        "dynamic_budget_buckets": [int(item) for item in dynamic_budget_buckets],
        "max_unselected_hole": None if max_unselected_hole is None else int(max_unselected_hole),
        "uses_uniform_fill": False,
        "uses_uniform_scaffold": False,
        "loss_terms": dict(DEFAULT_GAS_VT_LOSS_TERMS),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "policy_checkpoint_sha256": checkpoint_sha256,
        "source_jsonl_sha256": source_jsonl_sha256,
    }
    return out


def gas_vt_training_objective(
    policy_outputs: Mapping[str, Any],
    *,
    action_target: Any | None = None,
    boundary_target: Any | None = None,
    valid: Any | None = None,
    target_budget: Any | None = None,
    action_interior_bins: Any | None = None,
    loss_terms: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    weights = dict(DEFAULT_GAS_VT_LOSS_TERMS)
    if loss_terms is not None:
        weights.update({str(key): float(value) for key, value in loss_terms.items()})
    frame_logits = policy_outputs["frame_value"].float()
    if valid is None:
        valid_mask = torch.ones_like(frame_logits, dtype=torch.bool)
    else:
        valid_mask = valid.to(device=frame_logits.device).bool()
    probabilities = torch.sigmoid(frame_logits).masked_fill(~valid_mask, 0.0)
    if "st_selected_mask" in policy_outputs:
        selected_mask = policy_outputs["st_selected_mask"].to(device=frame_logits.device).float()
        mask_source = "st_selected_mask"
    elif "hard_selected_mask" in policy_outputs:
        selected_mask = policy_outputs["hard_selected_mask"].to(device=frame_logits.device).float()
        mask_source = "hard_selected_mask"
    else:
        selected_mask = probabilities
        mask_source = "sigmoid_probability_fallback"
    selected_mask = selected_mask.masked_fill(~valid_mask, 0.0)
    zero = frame_logits.sum() * 0.0
    total = zero
    losses: dict[str, Any] = {}
    if action_target is not None:
        action = action_target.to(device=frame_logits.device).float()
        losses["value_bce_loss"] = F.binary_cross_entropy_with_logits(frame_logits[valid_mask], action[valid_mask]) if bool(valid_mask.any().item()) else zero
        total = total + losses["value_bce_loss"] * weights["value_bce"]
    else:
        losses["value_bce_loss"] = zero
    if boundary_target is not None:
        boundary = boundary_target.to(device=frame_logits.device).float()
        boundary_mask = (boundary > 0.0) & valid_mask
        if bool(boundary_mask.any().item()):
            local = F.max_pool1d(selected_mask.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
            losses["boundary_coverage_loss"] = -torch.log(local[boundary_mask].clamp_min(1e-6)).mean()
            bracket_terms = []
            for batch_idx, boundary_idx in torch.nonzero(boundary_mask, as_tuple=False).tolist():
                left_start = max(0, int(boundary_idx) - 1)
                left_stop = int(boundary_idx)
                right_start = int(boundary_idx) + 1
                right_stop = min(selected_mask.shape[1], int(boundary_idx) + 2)
                if left_start >= left_stop or right_start >= right_stop:
                    continue
                left_hit = selected_mask[batch_idx, left_start:left_stop].max()
                right_hit = selected_mask[batch_idx, right_start:right_stop].max()
                bracket_terms.append((left_hit * right_hit).clamp_min(1e-6))
            if bracket_terms:
                losses["boundary_bracket_loss"] = -torch.log(torch.stack(bracket_terms)).mean()
            else:
                losses["boundary_bracket_loss"] = zero
        else:
            losses["boundary_coverage_loss"] = zero
            losses["boundary_bracket_loss"] = zero
        total = total + losses["boundary_coverage_loss"] * weights["boundary_coverage"]
        total = total + losses["boundary_bracket_loss"] * weights["boundary_bracket"]
    else:
        losses["boundary_coverage_loss"] = zero
        losses["boundary_bracket_loss"] = zero
    if action_interior_bins is not None:
        bins = action_interior_bins.to(device=frame_logits.device).float()
        if bins.ndim == 3 and bins.numel() > 0:
            mass = torch.einsum("bt,bnt->bn", selected_mask, bins)
            denom = bins.sum(dim=-1).clamp_min(1.0)
            coverage = (mass / denom).clamp(max=1.0)
            losses["action_interior_bin_loss"] = (1.0 - coverage).mean()
        else:
            losses["action_interior_bin_loss"] = zero
        total = total + losses["action_interior_bin_loss"] * weights["action_interior_bin"]
    else:
        losses["action_interior_bin_loss"] = zero
    if selected_mask.shape[-1] > 1:
        window = min(8, selected_mask.shape[-1])
        hole_mass = F.avg_pool1d(selected_mask.unsqueeze(1), kernel_size=window, stride=1).squeeze(1) * float(window)
        losses["cvar_max_hole_loss"] = torch.topk(F.relu(1.0 - hole_mass), k=max(1, hole_mass.shape[-1] // 4), dim=-1).values.mean()
    else:
        losses["cvar_max_hole_loss"] = zero
    total = total + losses["cvar_max_hole_loss"] * weights["cvar_max_hole"]
    if target_budget is not None:
        target = torch.as_tensor(target_budget, dtype=selected_mask.dtype, device=selected_mask.device)
        if target.ndim == 0:
            target = target.expand(selected_mask.shape[0])
        losses["budget_loss"] = ((selected_mask.sum(dim=-1) - target.float()) / target.float().clamp_min(1.0)).square().mean()
        total = total + losses["budget_loss"] * weights["budget"]
    else:
        losses["budget_loss"] = zero
    paction_dependence = torch.relu(0.05 - probabilities.std(dim=-1)).mean()
    losses["paction_dependence_loss"] = paction_dependence
    total = total + paction_dependence * weights["paction_dependence"]
    losses["total_loss"] = total
    losses["selection_mask_source"] = mask_source
    losses["policy_family"] = "GAS-VT"
    return losses


class GapAwareSequentialAcquisitionPolicy(_TorchModuleBase):
    def __init__(
        self,
        *,
        input_dim: int = len(GAS_VT_FEATURE_NAMES),
        hidden_dim: int = 64,
        num_layers: int = 3,
        budget_buckets: Sequence[int] = DEFAULT_GAS_VT_DYNAMIC_BUDGET_BUCKETS,
        dropout: float = 0.10,
    ) -> None:
        import torch
        import torch.nn as nn

        super().__init__()
        self.torch = torch
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
                    nn.Conv1d(int(hidden_dim), int(hidden_dim), kernel_size=3, padding=dilation, dilation=dilation, bias=False),
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

    def forward(self, features: Any, valid: Any | None = None, target_budget: Any | None = None) -> dict[str, Any]:
        torch = self.torch
        if features.ndim != 3:
            raise ValueError(f"GAS-VT policy expects [B,T,F], got {tuple(features.shape)}")
        encoded = self.encoder(features.float().transpose(1, 2))
        frame_value = self.frame_value_head(encoded).squeeze(1)
        if valid is not None:
            valid = valid.to(device=frame_value.device).bool()
            frame_value = frame_value.masked_fill(~valid, -1e4)
            weights = valid.float()
            pooled = (encoded * weights[:, None, :]).sum(dim=-1) / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        else:
            valid = torch.ones_like(frame_value, dtype=torch.bool)
            pooled = encoded.mean(dim=-1)
        budget_logits = self.budget_head(pooled)
        budget = target_budget
        if budget is None:
            budget = torch.full((features.shape[0],), min(384, features.shape[1]), dtype=torch.long, device=features.device)
        budget = torch.as_tensor(budget, dtype=torch.long, device=features.device)
        hard = torch.zeros_like(frame_value)
        for row_idx in range(frame_value.shape[0]):
            k = int(max(0, min(int(budget[row_idx].item()), int(valid[row_idx].sum().item()))))
            if k <= 0:
                continue
            topk = torch.topk(frame_value[row_idx].masked_fill(~valid[row_idx], -1e4), k=k).indices
            hard[row_idx, topk] = 1.0
        probabilities = torch.sigmoid(frame_value)
        st_mask = hard + probabilities - probabilities.detach()
        return {
            "frame_value": frame_value,
            "budget_logits": budget_logits,
            "budget_buckets": torch.tensor(self.budget_buckets, dtype=torch.long, device=features.device),
            "hard_selected_mask": hard.masked_fill(~valid, 0.0),
            "st_selected_mask": st_mask.masked_fill(~valid, 0.0),
        }
