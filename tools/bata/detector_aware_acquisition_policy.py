from __future__ import annotations

import copy
import math
import os
from typing import Any, Mapping, Sequence

from tools.bata import gas_vt_paction_policy as gas_vt
from tools.bata import paction_acquisition_policy as base_policy


if os.name != "nt":
    from torch.nn import Module as _TorchModuleBase
else:
    _TorchModuleBase = object


STAGE_LABEL = "Stage-2 detector-aware offline selector"
DETECTOR_AWARE_FIXED_384_STRATEGY = "detector_aware_fixed_384"
DETECTOR_AWARE_FIXED_768_STRATEGY = "detector_aware_fixed_768"
DETECTOR_AWARE_DYNAMIC_STRATEGY = "detector_aware_dynamic"
DETECTOR_AWARE_STRATEGIES = (
    DETECTOR_AWARE_FIXED_384_STRATEGY,
    DETECTOR_AWARE_FIXED_768_STRATEGY,
    DETECTOR_AWARE_DYNAMIC_STRATEGY,
)
DETECTOR_AWARE_CHECKPOINT_POLICY_SOURCE = "learned_detector_aware_policy_checkpoint"
DETECTOR_AWARE_BOOTSTRAP_POLICY_SOURCE = "bootstrap_detector_aware_surrogate_policy"
DEFAULT_DETECTOR_AWARE_DYNAMIC_BUDGET_BUCKETS = gas_vt.DEFAULT_GAS_VT_DYNAMIC_BUDGET_BUCKETS
DETECTOR_AWARE_FEATURE_NAMES = gas_vt.GAS_VT_FEATURE_NAMES
DEFAULT_DYNAMIC_GAIN_CALIBRATION = {
    "score_semantics": "calibrated_marginal_gain",
    "calibration_scope": "cross_video_comparable",
    "target_source": "abs_signed_detector_utility",
    "budget_decoding": "bucket_logits_are_cross_video_marginal_gain_threshold_surrogates",
}
DEFAULT_DETECTOR_AWARE_LOSS_TERMS = {
    "utility_mse": 1.0,
    "utility_bce": 0.5,
    "selected_utility": 1.0,
    "cvar_max_hole": 1.0,
    "budget": 1.0,
}


def feature_index(name: str) -> int:
    return DETECTOR_AWARE_FEATURE_NAMES.index(str(name))


def build_detector_aware_feature_matrix(
    p_action: Sequence[Any],
    *,
    valid: Sequence[Any] | None = None,
    selected_so_far: Sequence[int] | None = None,
    target_budget: int | None = None,
) -> list[list[float]]:
    return gas_vt.build_gap_aware_feature_matrix(
        p_action,
        valid=valid,
        selected_so_far=selected_so_far,
        target_budget=target_budget,
    )


def _strategy_budget_for_name(name: str, fixed_budgets: Sequence[int]) -> int:
    budgets = [int(item) for item in fixed_budgets]
    if str(name) == DETECTOR_AWARE_FIXED_384_STRATEGY:
        return budgets[0]
    if str(name) == DETECTOR_AWARE_FIXED_768_STRATEGY:
        return budgets[1] if len(budgets) > 1 else budgets[0]
    raise ValueError(f"unknown detector-aware fixed strategy: {name}")


def add_detector_aware_decision_to_sample_row(
    row: Mapping[str, Any],
    *,
    frame_values: Sequence[Any],
    fixed_budgets: Sequence[int] = (384, 768),
    dynamic_budget_scores: Sequence[Any],
    dynamic_budget_buckets: Sequence[Any] = DEFAULT_DETECTOR_AWARE_DYNAMIC_BUDGET_BUCKETS,
    dynamic_gain_calibration: Mapping[str, Any] | None = None,
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
    for strategy_name in (DETECTOR_AWARE_FIXED_384_STRATEGY, DETECTOR_AWARE_FIXED_768_STRATEGY):
        requested = _strategy_budget_for_name(strategy_name, fixed_budgets)
        budget = base_policy.short_valid_ratio_budget(requested, valid_len=valid_len, dense_len=dense_len)
        strategies[strategy_name] = gas_vt.hard_gap_aware_topk(
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
    calibration = dict(DEFAULT_DYNAMIC_GAIN_CALIBRATION)
    if dynamic_gain_calibration is not None:
        calibration.update(dict(dynamic_gain_calibration))
    strategies[DETECTOR_AWARE_DYNAMIC_STRATEGY] = gas_vt.hard_gap_aware_topk(
        frame_values,
        valid=valid,
        budget=dynamic_budget,
        max_unselected_hole=max_unselected_hole,
    )
    out["strategy_selected_positions"] = strategies
    out["detector_aware_policy"] = {
        "policy_family": "detector_aware_offline_selector",
        "stage_label": STAGE_LABEL,
        "route_label": "DIVERGENT_INNOVATION_DETECTOR_AWARE_UTILITY_DO_NOT_MERGE_WITH_C3",
        "selection_signal": "p_action_to_detector_utility",
        "teacher_target_scope": "train_only",
        "source": source,
        "decode_mode": "hard_gap_aware_topk",
        "fixed_strategies": [DETECTOR_AWARE_FIXED_384_STRATEGY, DETECTOR_AWARE_FIXED_768_STRATEGY],
        "dynamic_strategy": DETECTOR_AWARE_DYNAMIC_STRATEGY,
        "fixed_budgets": [int(item) for item in fixed_budgets],
        "dynamic_budget": int(dynamic_budget),
        "dynamic_budget_buckets": [int(item) for item in dynamic_budget_buckets],
        "dynamic_budget_calibration": calibration,
        "max_unselected_hole": None if max_unselected_hole is None else int(max_unselected_hole),
        "uses_uniform_fill": False,
        "uses_uniform_scaffold": False,
        "end_to_end": False,
        "deploy_claim_allowed": False,
        "runtime_flops_claim_allowed": False,
        "paper_claim_allowed": False,
        "map_claim_allowed": False,
        "loss_terms": dict(DEFAULT_DETECTOR_AWARE_LOSS_TERMS),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "policy_checkpoint_sha256": checkpoint_sha256,
        "source_jsonl_sha256": source_jsonl_sha256,
    }
    return out


def detector_utility_metrics(
    selected: Sequence[int],
    utility: Sequence[Any],
    *,
    valid_len: int | None = None,
) -> dict[str, float | None]:
    valid_len = len(utility) if valid_len is None else min(int(valid_len), len(utility))
    values = [max(0.0, float(item)) for item in utility[:valid_len]]
    selected_set = {int(item) for item in selected if 0 <= int(item) < valid_len}
    total = sum(values)
    selected_utility = sum(values[idx] for idx in selected_set)
    coverage = None if total <= 0 else selected_utility / total
    gains = [values[idx] for idx in selected if 0 <= int(idx) < valid_len]
    dcg = sum(float(gain) / math.log2(rank + 2.0) for rank, gain in enumerate(gains))
    ideal = sorted(values, reverse=True)[: len(selected_set)]
    idcg = sum(float(gain) / math.log2(rank + 2.0) for rank, gain in enumerate(ideal))
    ndcg = None if idcg <= 0 else dcg / idcg
    return {
        "detector_utility_coverage": coverage,
        "detector_utility_ndcg": ndcg,
        "detector_utility_selected_sum": selected_utility,
        "detector_utility_total_sum": total,
    }


def detector_aware_training_objective(
    policy_outputs: Mapping[str, Any],
    *,
    detector_utility_target: Any,
    detector_gain_target: Any | None = None,
    valid: Any | None = None,
    target_budget: Any | None = None,
    loss_terms: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    weights = dict(DEFAULT_DETECTOR_AWARE_LOSS_TERMS)
    if loss_terms is not None:
        weights.update({str(key): float(value) for key, value in loss_terms.items()})
    frame_logits = policy_outputs["frame_value"].float()
    if valid is None:
        valid_mask = torch.ones_like(frame_logits, dtype=torch.bool)
    else:
        valid_mask = valid.to(device=frame_logits.device).bool()
    signed_target = detector_utility_target.to(device=frame_logits.device).float().clamp(-1.0, 1.0)
    gain_target = (
        detector_gain_target.to(device=frame_logits.device).float().abs().clamp(0.0, 1.0)
        if detector_gain_target is not None
        else signed_target.abs()
    )
    signed_prediction = torch.tanh(frame_logits).masked_fill(~valid_mask, 0.0)
    probabilities = torch.sigmoid(frame_logits).masked_fill(~valid_mask, 0.0)
    selected_mask = policy_outputs.get("st_selected_mask", probabilities).to(device=frame_logits.device).float()
    selected_mask = selected_mask.masked_fill(~valid_mask, 0.0)
    zero = frame_logits.sum() * 0.0
    if bool(valid_mask.any().item()):
        utility_mse = F.mse_loss(signed_prediction[valid_mask], signed_target[valid_mask])
        utility_bce = F.binary_cross_entropy_with_logits(frame_logits[valid_mask], gain_target[valid_mask])
    else:
        utility_mse = zero
        utility_bce = zero
    denom = (gain_target * valid_mask.float()).sum(dim=-1).clamp_min(1e-6)
    selected_coverage = ((selected_mask * gain_target).sum(dim=-1) / denom).clamp(max=1.0)
    selected_utility_loss = (1.0 - selected_coverage).mean()
    if selected_mask.shape[-1] > 1:
        window = min(8, selected_mask.shape[-1])
        hole_mass = F.avg_pool1d(selected_mask.unsqueeze(1), kernel_size=window, stride=1).squeeze(1) * float(window)
        cvar_max_hole_loss = torch.topk(F.relu(1.0 - hole_mass), k=max(1, hole_mass.shape[-1] // 4), dim=-1).values.mean()
    else:
        cvar_max_hole_loss = zero
    if target_budget is not None:
        budget = torch.as_tensor(target_budget, dtype=selected_mask.dtype, device=selected_mask.device)
        if budget.ndim == 0:
            budget = budget.expand(selected_mask.shape[0])
        budget_loss = ((selected_mask.sum(dim=-1) - budget.float()) / budget.float().clamp_min(1.0)).square().mean()
    else:
        budget_loss = zero
    total = (
        utility_mse * weights["utility_mse"]
        + utility_bce * weights["utility_bce"]
        + selected_utility_loss * weights["selected_utility"]
        + cvar_max_hole_loss * weights["cvar_max_hole"]
        + budget_loss * weights["budget"]
    )
    return {
        "utility_mse_loss": utility_mse,
        "utility_bce_loss": utility_bce,
        "selected_utility_loss": selected_utility_loss,
        "cvar_max_hole_loss": cvar_max_hole_loss,
        "budget_loss": budget_loss,
        "total_loss": total,
        "policy_family": "detector_aware_offline_selector",
    }


class DetectorAwareSequentialAcquisitionPolicy(_TorchModuleBase):
    def __init__(
        self,
        *,
        input_dim: int = len(DETECTOR_AWARE_FEATURE_NAMES),
        hidden_dim: int = 64,
        num_layers: int = 3,
        budget_buckets: Sequence[int] = DEFAULT_DETECTOR_AWARE_DYNAMIC_BUDGET_BUCKETS,
        dropout: float = 0.10,
    ) -> None:
        import torch
        import torch.nn as nn

        super().__init__()
        self.torch = torch
        self.budget_buckets = tuple(int(item) for item in budget_buckets)
        self.encoder = gas_vt.GapAwareSequentialAcquisitionPolicy(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            budget_buckets=self.budget_buckets,
            dropout=float(dropout),
        )
        self.identity = nn.Identity()

    def forward(self, features: Any, valid: Any | None = None, target_budget: Any | None = None) -> dict[str, Any]:
        return self.encoder(features, valid=valid, target_budget=target_budget)
