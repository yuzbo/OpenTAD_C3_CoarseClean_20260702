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
    "target_source": "positive_observation_gain_from_signed_detector_utility",
    "budget_decoding": "bucket_logits_are_train_calibrated_marginal_gain_threshold_surrogates",
    "requires_fit_evidence": True,
    "calibration_fitted": False,
    "calibrated_dynamic_claim_allowed": False,
}
UNCALIBRATED_DYNAMIC_BUDGET_CALIBRATION = {
    "score_semantics": "uncalibrated_dynamic_budget_score",
    "calibration_scope": "per_run_or_bootstrap_only",
    "target_source": "unknown",
    "budget_decoding": "bucket_logits_without_train_fit_calibration",
    "requires_fit_evidence": True,
    "calibration_fitted": False,
    "calibrated_dynamic_claim_allowed": False,
}
DEFAULT_DETECTOR_AWARE_LOSS_TERMS = {
    "utility_mse": 1.0,
    "utility_bce": 0.5,
    "utility_risk_bce": 0.5,
    "selected_utility": 1.0,
    "cvar_max_hole": 1.0,
    "learned_spacing": 0.75,
    "action_local_hole": 1.0,
    "context_radius_cost": 0.05,
    "budget": 1.0,
}
DEFAULT_DETECTOR_AWARE_SCORE_DILATION_RADII = (2, 4)
DEFAULT_CONTEXT_RADIUS_MIN = 2.0
DEFAULT_CONTEXT_RADIUS_MAX = 16.0


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


def fixed_deploy_budget(requested_budget: int, *, valid_len: int) -> int:
    """Detector-aware fixed budgets fill the valid dense grid, capped by budget."""
    requested = int(requested_budget)
    valid = int(valid_len)
    if requested <= 0:
        return 0
    if valid <= 0:
        raise ValueError(f"valid_len must be positive, got {valid_len}")
    return min(requested, valid)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(out) or math.isinf(out):
        return 0.0
    return out


def _clamp_context_radius(value: Any) -> float:
    return max(DEFAULT_CONTEXT_RADIUS_MIN, min(DEFAULT_CONTEXT_RADIUS_MAX, _safe_float(value)))


def _is_score_dilation_seed(values: Sequence[float], valid_mask: Sequence[bool], center: int) -> bool:
    """Only strict local positive peaks seed context dilation."""
    if not bool(valid_mask[center]):
        return False
    center_score = float(values[center])
    if center_score <= 0.0:
        return False
    neighbours: list[float] = []
    for pos in (center - 1, center + 1):
        if 0 <= pos < len(values) and bool(valid_mask[pos]):
            neighbours.append(float(values[pos]))
    if not neighbours:
        return True
    return center_score > max(neighbours)


def score_dilated_frame_values(
    frame_values: Sequence[Any],
    *,
    valid: Sequence[Any] | None = None,
    valid_len: int | None = None,
    radii: Sequence[int] = DEFAULT_DETECTOR_AWARE_SCORE_DILATION_RADII,
    context_radii: Sequence[Any] | None = None,
) -> list[float]:
    """Spread learned detector-utility scores to local context around peaks.

    This is not a uniform scaffold or slot rule: only learned positive peaks
    seed their neighbours, and the original score still dominates when it is
    higher. Radii 2 and 4 give AdaTAD local evidence around responsibility
    points while preserving a hard <=budget selection.
    """

    values = [_safe_float(item) for item in frame_values]
    length = len(values)
    if valid is None:
        valid_mask = [idx < (length if valid_len is None else int(valid_len)) for idx in range(length)]
    else:
        if len(valid) != length:
            raise ValueError("valid mask length must match frame_values length")
        valid_mask = [bool(item) for item in valid]
    limit = length if valid_len is None else max(0, min(int(valid_len), length))
    valid_mask = [bool(flag) and idx < limit for idx, flag in enumerate(valid_mask)]
    learned_radii: list[float] | None = None
    if context_radii is not None:
        if len(context_radii) != length:
            raise ValueError("context_radii length must match frame_values length")
        learned_radii = [_clamp_context_radius(item) for item in context_radii]
        clean_radii = [max(1, int(math.ceil(max(learned_radii) if learned_radii else 0.0)))]
    else:
        clean_radii = sorted({int(radius) for radius in radii if int(radius) > 0})
    if not clean_radii:
        return values
    max_radius = max(clean_radii)
    inner_radius = min(clean_radii)
    out = list(values)
    for center, center_score in enumerate(values):
        if not _is_score_dilation_seed(values, valid_mask, center):
            continue
        center_radius = float(learned_radii[center]) if learned_radii is not None else float(max_radius)
        for distance in range(1, int(math.ceil(center_radius)) + 1):
            if learned_radii is not None:
                weight = max(0.0, 1.0 - (0.08 * float(distance) / max(1.0, center_radius)))
            elif distance <= inner_radius:
                weight = max(0.0, 1.0 - 0.02 * float(distance))
            else:
                weight = max(0.0, 0.94 - 0.04 * float(distance - inner_radius))
            candidate_score = center_score * weight - 1e-6 * float(distance)
            for pos in (center - distance, center + distance):
                if 0 <= pos < length and valid_mask[pos]:
                    out[pos] = max(float(out[pos]), float(candidate_score))
    return [float(item) for item in out]


def _has_dynamic_gain_calibration_evidence(calibration: Mapping[str, Any] | None) -> bool:
    if not isinstance(calibration, Mapping):
        return False
    if calibration.get("calibration_fitted") is not True:
        return False
    if calibration.get("fit_split") not in {"training", "train"}:
        return False
    if calibration.get("score_semantics") != "calibrated_marginal_gain":
        return False
    if calibration.get("calibration_scope") != "cross_video_comparable":
        return False
    if calibration.get("target_source") != "positive_observation_gain_from_signed_detector_utility":
        return False
    if not isinstance(calibration.get("budget_buckets"), Sequence) or isinstance(calibration.get("budget_buckets"), (str, bytes)):
        return False
    threshold = calibration.get("gain_threshold")
    return threshold is None or isinstance(threshold, (int, float))


def _dynamic_budget_calibration_metadata(calibration: Mapping[str, Any] | None) -> dict[str, Any]:
    if not _has_dynamic_gain_calibration_evidence(calibration):
        out = dict(UNCALIBRATED_DYNAMIC_BUDGET_CALIBRATION)
        if isinstance(calibration, Mapping):
            out["provided_score_semantics"] = calibration.get("score_semantics")
            out["provided_calibration_fitted"] = calibration.get("calibration_fitted")
        return out
    out = dict(DEFAULT_DYNAMIC_GAIN_CALIBRATION)
    out.update(dict(calibration or {}))
    out["calibration_fitted"] = True
    out["calibrated_dynamic_claim_allowed"] = (
        out.get("calibrated_dynamic_claim_allowed") is True
        and out.get("gain_threshold") is not None
    )
    return out


def add_detector_aware_decision_to_sample_row(
    row: Mapping[str, Any],
    *,
    frame_values: Sequence[Any],
    frame_values_by_strategy: Mapping[str, Sequence[Any]] | None = None,
    context_radii_by_strategy: Mapping[str, Sequence[Any]] | None = None,
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
    strategy_frame_values = dict(frame_values_by_strategy or {})
    strategy_context_radii = dict(context_radii_by_strategy or {})
    for strategy_name in (DETECTOR_AWARE_FIXED_384_STRATEGY, DETECTOR_AWARE_FIXED_768_STRATEGY):
        requested = _strategy_budget_for_name(strategy_name, fixed_budgets)
        budget = fixed_deploy_budget(requested, valid_len=valid_len)
        decoded_values = score_dilated_frame_values(
            strategy_frame_values.get(strategy_name, frame_values),
            valid=valid,
            valid_len=valid_len,
            context_radii=strategy_context_radii.get(strategy_name),
        )
        strategies[strategy_name] = gas_vt.hard_gap_aware_topk(
            decoded_values,
            valid=valid,
            budget=budget,
            max_unselected_hole=max_unselected_hole,
        )
    dynamic_budget = base_policy.decode_budget_from_scores(
        dynamic_budget_scores,
        dynamic_budget_buckets,
        valid_len=valid_len,
    )
    calibration = _dynamic_budget_calibration_metadata(dynamic_gain_calibration)
    dynamic_decoded_values = score_dilated_frame_values(
        strategy_frame_values.get(DETECTOR_AWARE_DYNAMIC_STRATEGY, frame_values),
        valid=valid,
        valid_len=valid_len,
        context_radii=strategy_context_radii.get(DETECTOR_AWARE_DYNAMIC_STRATEGY),
    )
    strategies[DETECTOR_AWARE_DYNAMIC_STRATEGY] = gas_vt.hard_gap_aware_topk(
        dynamic_decoded_values,
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
        "acquisition_unit": "temporal_observation_center",
        "selected_positions_coordinate_system": "local_dense_snippet_index",
        "dense_grid_unit": "snippet_center",
        "dense_grid_size": int(dense_len),
        "selected_observation_budget_unit": "temporal_observation_center",
        "claim_unit_allowed": "selected_temporal_observations_only",
        "raw_frame_claim_allowed": False,
        "teacher_target_scope": "train_only",
        "source": source,
        "decode_mode": "hard_gap_aware_topk_after_learned_score_dilation_r2_r4",
        "score_dilation_radii": [int(item) for item in DEFAULT_DETECTOR_AWARE_SCORE_DILATION_RADII],
        "score_dilation_signal": "learned_positive_detector_utility_peaks_only",
        "learned_context_radius_used": bool(context_radii_by_strategy),
        "context_radius_range": [float(DEFAULT_CONTEXT_RADIUS_MIN), float(DEFAULT_CONTEXT_RADIUS_MAX)],
        "fixed_strategies": [DETECTOR_AWARE_FIXED_384_STRATEGY, DETECTOR_AWARE_FIXED_768_STRATEGY],
        "dynamic_strategy": DETECTOR_AWARE_DYNAMIC_STRATEGY,
        "fixed_budgets": [int(item) for item in fixed_budgets],
        "fixed_budget_contract": "min_requested_budget_or_valid_len_no_short_ratio",
        "dynamic_budget": int(dynamic_budget),
        "dynamic_budget_buckets": [int(item) for item in dynamic_budget_buckets],
        "dynamic_budget_calibration": calibration,
        "max_unselected_hole": None if max_unselected_hole is None else int(max_unselected_hole),
        "uses_uniform_fill": False,
        "uses_uniform_scaffold": False,
        "budget_conditioned_frame_values": bool(frame_values_by_strategy),
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
    detector_risk_target: Any | None = None,
    action_target: Any | None = None,
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
        detector_gain_target.to(device=frame_logits.device).float().clamp(0.0, 1.0)
        if detector_gain_target is not None
        else signed_target.clamp_min(0.0)
    )
    risk_target = (
        detector_risk_target.to(device=frame_logits.device).float().clamp(0.0, 1.0)
        if detector_risk_target is not None
        else (-signed_target).clamp_min(0.0)
    )
    signed_prediction = torch.tanh(frame_logits).masked_fill(~valid_mask, 0.0)
    probabilities = torch.sigmoid(frame_logits).masked_fill(~valid_mask, 0.0)
    selected_mask = policy_outputs.get("st_selected_mask", probabilities).to(device=frame_logits.device).float()
    selected_mask = selected_mask.masked_fill(~valid_mask, 0.0)
    context_radius = policy_outputs.get("context_radius")
    if context_radius is not None:
        context_radius = context_radius.to(device=frame_logits.device).float().masked_fill(~valid_mask, 0.0)
    zero = frame_logits.sum() * 0.0
    if bool(valid_mask.any().item()):
        utility_mse = F.mse_loss(signed_prediction[valid_mask], signed_target[valid_mask])
        utility_bce = F.binary_cross_entropy_with_logits(frame_logits[valid_mask], gain_target[valid_mask])
        utility_risk_bce = F.binary_cross_entropy_with_logits(-frame_logits[valid_mask], risk_target[valid_mask])
    else:
        utility_mse = zero
        utility_bce = zero
        utility_risk_bce = zero
    denom = (gain_target * valid_mask.float()).sum(dim=-1).clamp_min(1e-6)
    selected_coverage = ((selected_mask * gain_target).sum(dim=-1) / denom).clamp(max=1.0)
    selected_utility_loss = (1.0 - selected_coverage).mean()
    if selected_mask.shape[-1] > 1:
        window = min(8, selected_mask.shape[-1])
        hole_mass = F.avg_pool1d(selected_mask.unsqueeze(1), kernel_size=window, stride=1).squeeze(1) * float(window)
        cvar_max_hole_loss = torch.topk(F.relu(1.0 - hole_mass), k=max(1, hole_mass.shape[-1] // 4), dim=-1).values.mean()
    else:
        cvar_max_hole_loss = zero
    spacing_terms = []
    action_terms = []
    action = None if action_target is None else action_target.to(device=frame_logits.device).float().clamp(0.0, 1.0)
    if context_radius is not None and selected_mask.shape[-1] > 1:
        steps = torch.arange(selected_mask.shape[-1], device=selected_mask.device, dtype=selected_mask.dtype)
        distance = (steps[None, :, None] - steps[None, None, :]).abs()
        radius = context_radius[:, :, None].clamp(DEFAULT_CONTEXT_RADIUS_MIN, DEFAULT_CONTEXT_RADIUS_MAX)
        soft_context_gate = torch.sigmoid((radius - distance) * 2.0) * valid_mask[:, None, :].float()
        soft_context_coverage = (selected_mask[:, :, None] * soft_context_gate).amax(dim=1).masked_fill(~valid_mask, 0.0)
    else:
        soft_context_coverage = selected_mask
    for radius in DEFAULT_DETECTOR_AWARE_SCORE_DILATION_RADII:
        window = min(int(2 * int(radius) + 1), int(selected_mask.shape[-1]))
        if window <= 1:
            continue
        selection_mass = F.avg_pool1d(soft_context_coverage.unsqueeze(1), kernel_size=window, stride=1).squeeze(1) * float(window)
        valid_mass = F.avg_pool1d(valid_mask.float().unsqueeze(1), kernel_size=window, stride=1).squeeze(1) * float(window)
        valid_windows = valid_mass >= float(window)
        if bool(valid_windows.any().item()):
            spacing_gap = F.relu(1.0 - selection_mass)
            spacing_terms.append(spacing_gap[valid_windows].square().mean())
            if action is not None:
                action_mass = F.avg_pool1d(action.unsqueeze(1), kernel_size=window, stride=1).squeeze(1) * float(window)
                action_windows = valid_windows & (action_mass > 0.0)
                if bool(action_windows.any().item()):
                    normalized_weight = (action_mass / float(window)).clamp(0.0, 1.0)
                    action_terms.append((spacing_gap.square() * normalized_weight)[action_windows].mean())
    learned_spacing_loss = torch.stack(spacing_terms).mean() if spacing_terms else zero
    action_local_hole_loss = torch.stack(action_terms).mean() if action_terms else zero
    if context_radius is not None:
        radius_cost = (((context_radius - DEFAULT_CONTEXT_RADIUS_MIN).clamp_min(0.0) / (DEFAULT_CONTEXT_RADIUS_MAX - DEFAULT_CONTEXT_RADIUS_MIN)).square() * selected_mask).sum(dim=-1)
        radius_cost = (radius_cost / selected_mask.sum(dim=-1).clamp_min(1.0)).mean()
    else:
        radius_cost = zero
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
        + utility_risk_bce * weights["utility_risk_bce"]
        + selected_utility_loss * weights["selected_utility"]
        + cvar_max_hole_loss * weights["cvar_max_hole"]
        + learned_spacing_loss * weights["learned_spacing"]
        + action_local_hole_loss * weights["action_local_hole"]
        + radius_cost * weights["context_radius_cost"]
        + budget_loss * weights["budget"]
    )
    return {
        "utility_mse_loss": utility_mse,
        "utility_bce_loss": utility_bce,
        "utility_risk_bce_loss": utility_risk_bce,
        "selected_utility_loss": selected_utility_loss,
        "cvar_max_hole_loss": cvar_max_hole_loss,
        "learned_spacing_loss": learned_spacing_loss,
        "action_local_hole_loss": action_local_hole_loss,
        "context_radius_cost_loss": radius_cost,
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
        radius_hidden = max(8, int(hidden_dim) // 2)
        self.context_radius_head = nn.Sequential(
            nn.Linear(int(input_dim), int(radius_hidden)),
            nn.SiLU(inplace=True),
            nn.Linear(int(radius_hidden), 1),
        )
        self.identity = nn.Identity()

    def forward(self, features: Any, valid: Any | None = None, target_budget: Any | None = None) -> dict[str, Any]:
        outputs = self.encoder(features, valid=valid, target_budget=target_budget)
        radius_logits = self.context_radius_head(features.float()).squeeze(-1)
        if valid is not None:
            valid_mask = valid.to(device=radius_logits.device).bool()
            radius_logits = radius_logits.masked_fill(~valid_mask, -1e4)
        context_radius = DEFAULT_CONTEXT_RADIUS_MIN + (
            DEFAULT_CONTEXT_RADIUS_MAX - DEFAULT_CONTEXT_RADIUS_MIN
        ) * self.torch.sigmoid(radius_logits)
        outputs["context_radius_logits"] = radius_logits
        outputs["context_radius"] = context_radius
        outputs["context_radius_range"] = self.torch.tensor(
            [DEFAULT_CONTEXT_RADIUS_MIN, DEFAULT_CONTEXT_RADIUS_MAX],
            dtype=features.dtype,
            device=features.device,
        )
        return outputs
