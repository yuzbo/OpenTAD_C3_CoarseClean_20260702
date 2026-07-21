from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any, Optional

import torch
import torch.nn as nn

from opentad.duca_loss_contract import (
    DUCA_LOSS_TO_WEIGHT_KEY,
    DUCA_LOSS_WEIGHT_DEFAULTS,
)

from ..builder import SELECTORS
from ..duca import (
    C3CoarseProbeActionnessSource,
    DucaAcquisitionAdapter,
    DucaTemporalSamplingContract,
    ZeroShotActionnessSource,
    duca_losses,
)
from ..duca.counterfactual_utility import (
    build_finite_hard_one_swap_candidates,
    build_local_cell_hard_flip_candidates,
    build_swap_incidence_matrix,
    counterfactual_pair_scores,
    counterfactual_utility_distillation_loss,
    local_cell_signed_logistic_loss,
    score_space_utility_alignment,
    signed_one_swap_proximal_loss,
)
from ..duca.acquisition import (
    _assert_no_forbidden_payload,
    _elapsed_ms,
    _sync_profile_clock,
    validate_actionness_provenance,
)
from ..duca.structured_selection import exact_uniform_positions
from ..utils.truetime_geometry import SELECTED_AXIS, TRUE_TIME_AXIS, TrueTimeMap


_DEFAULT_METADATA_KEYS = {
    "selected_positions": "duca_online_selected_positions",
    "selected_positions_unit": "duca_online_selected_positions_unit",
    "selected_mask": "duca_online_selected_mask",
    "selected_count": "duca_online_selected_count",
    "remap": "duca_online_selected_axis_remap",
    "source": "duca_online_actionness_source",
}

_EXTERNAL_ACTIONNESS_PAYLOAD_KEYS = {
    "duca_external_p_action",
    "duca_external_actionness_logits",
    "duca_external_actionness_valid",
    "duca_external_actionness_provenance",
    "duca_external_actionness_source",
    "duca_external_actionness_observation_times",
    "duca_external_actionness_jsonl",
}

_ACTIONNESS_KWARGS = {
    "feature_dim",
    "hidden_dim",
    "frozen",
    "mode",
    "p_action",
    "uncertainty",
    "video_text_model",
    "tokenizer",
    "action_prompts",
    "background_prompts",
    "temperature",
    "provenance",
    "source_name",
    "checkpoint_hash",
    "thumos_trained",
    "uses_labels",
    "uses_teacher",
    "uses_gt",
    "uses_prediction_cache",
    "calibration_split",
    "prompt_hash",
}


def _time_descriptors_btc(inputs: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(inputs):
        raise ValueError("inputs must be a tensor")
    descriptor_inputs = inputs if torch.is_floating_point(inputs) or torch.is_complex(inputs) else inputs.float()
    if inputs.ndim == 3:
        return descriptor_inputs.transpose(1, 2).contiguous()
    if inputs.ndim == 5:
        return descriptor_inputs.mean(dim=(3, 4)).transpose(1, 2).contiguous()
    if inputs.ndim == 6:
        return descriptor_inputs.mean(dim=(1, 4, 5)).transpose(1, 2).contiguous()
    raise ValueError(f"unsupported DUCA selector input shape: {tuple(inputs.shape)}")


def _gather_time(inputs: torch.Tensor, selected_positions: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
    idx = selected_positions.to(device=inputs.device, dtype=torch.long).clamp_min(0)
    if inputs.ndim == 3:
        gathered = torch.gather(inputs, dim=2, index=idx[:, None, :].expand(-1, inputs.shape[1], -1))
        return gathered * slot_mask[:, None, :].to(dtype=gathered.dtype)
    if inputs.ndim == 5:
        gathered = torch.gather(
            inputs,
            dim=2,
            index=idx[:, None, :, None, None].expand(-1, inputs.shape[1], -1, inputs.shape[3], inputs.shape[4]),
        )
        return gathered * slot_mask[:, None, :, None, None].to(dtype=gathered.dtype)
    if inputs.ndim == 6:
        gathered = torch.gather(
            inputs,
            dim=3,
            index=idx[:, None, None, :, None, None].expand(
                -1, inputs.shape[1], inputs.shape[2], -1, inputs.shape[4], inputs.shape[5]
            ),
        )
        return gathered * slot_mask[:, None, None, :, None, None].to(dtype=gathered.dtype)
    raise ValueError(f"unsupported DUCA selector input shape: {tuple(inputs.shape)}")


def _apply_slot_weights(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if inputs.ndim == 3:
        return inputs * weights[:, None, :].to(dtype=inputs.dtype)
    if inputs.ndim == 5:
        return inputs * weights[:, None, :, None, None].to(dtype=inputs.dtype)
    if inputs.ndim == 6:
        return inputs * weights[:, None, None, :, None, None].to(dtype=inputs.dtype)
    raise ValueError(f"unsupported DUCA selector input shape: {tuple(inputs.shape)}")


def _training_uniform_companion_mask(
    batch_size: int,
    *,
    fraction: float,
    device: torch.device,
) -> torch.Tensor:
    """Choose uniform-view rows while retaining one learned row per multi-row batch."""

    mask = torch.zeros(int(batch_size), device=device, dtype=torch.bool)
    if batch_size <= 1 or fraction <= 0.0:
        return mask
    uniform_count = max(1, int(round(float(batch_size) * float(fraction))))
    uniform_count = min(uniform_count, int(batch_size) - 1)
    permutation = torch.randperm(int(batch_size), device=device)
    mask[permutation[:uniform_count]] = True
    return mask


def _training_uniform_companion_bridge_scale(
    companion_mask: torch.Tensor,
    *,
    normalize_learned_gradient: bool,
) -> torch.Tensor:
    """Return per-row selector-gradient scales for a mixed learned/uniform batch."""

    if companion_mask.ndim != 1 or companion_mask.dtype != torch.bool:
        raise ValueError("uniform companion mask must be a one-dimensional bool tensor")
    learned_mask = ~companion_mask
    learned_count = int(learned_mask.long().sum().item())
    if learned_count <= 0:
        raise ValueError("uniform companion training requires at least one learned row")
    scales = learned_mask.to(dtype=torch.float32)
    if normalize_learned_gradient:
        scales = scales * (float(companion_mask.numel()) / float(learned_count))
    return scales


def _exact_uniform_companion_tensors(
    valid_mask: torch.Tensor,
    *,
    slot_count: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the canonical round(linspace) hard path and its one-hot slots."""

    valid = valid_mask.bool()
    batch, temporal_len = valid.shape
    positions = torch.full(
        (batch, int(slot_count)),
        -1,
        device=valid.device,
        dtype=torch.long,
    )
    dense_mask = torch.zeros(
        (batch, temporal_len),
        device=valid.device,
        dtype=torch.bool,
    )
    slot_assignment = torch.zeros(
        (batch, int(slot_count), temporal_len),
        device=valid.device,
        dtype=dtype,
    )
    for batch_idx in range(batch):
        valid_positions = torch.nonzero(valid[batch_idx], as_tuple=False).flatten()
        effective_k = min(int(slot_count), int(valid_positions.numel()))
        if effective_k <= 0:
            raise ValueError("uniform companion requires one valid candidate")
        anchors = exact_uniform_positions(
            int(valid_positions.numel()),
            effective_k,
            device=valid.device,
        )
        selected = valid_positions[anchors]
        positions[batch_idx, :effective_k] = selected
        dense_mask[batch_idx, selected] = True
        slots = torch.arange(effective_k, device=valid.device)
        slot_assignment[batch_idx, slots, selected] = 1.0
    return positions, dense_mask, slot_assignment


def _add_soft_context_gradient_path(
    hard_selected: torch.Tensor,
    dense_inputs: torch.Tensor,
    soft_coverage: torch.Tensor,
    slot_mask: torch.Tensor,
    bridge_weight: float = 1.0,
) -> torch.Tensor:
    bridge = float(bridge_weight)
    if bridge <= 0.0:
        return hard_selected
    context_inputs = dense_inputs if torch.is_floating_point(dense_inputs) or torch.is_complex(dense_inputs) else dense_inputs.float()
    hard_base = hard_selected if torch.is_floating_point(hard_selected) or torch.is_complex(hard_selected) else hard_selected.float()
    weights = soft_coverage.to(device=context_inputs.device, dtype=context_inputs.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
    if context_inputs.ndim == 3:
        context = torch.einsum("bct,bt->bc", context_inputs, weights)
        context = context[:, :, None].expand_as(hard_selected)
        slot = slot_mask[:, None, :]
    elif context_inputs.ndim == 5:
        context = (context_inputs * weights[:, None, :, None, None]).sum(dim=2)
        context = context[:, :, None, :, :].expand_as(hard_selected)
        slot = slot_mask[:, None, :, None, None]
    elif context_inputs.ndim == 6:
        context = (context_inputs * weights[:, None, None, :, None, None]).sum(dim=3)
        context = context[:, :, :, None, :, :].expand_as(hard_selected)
        slot = slot_mask[:, None, None, :, None, None]
    else:
        raise ValueError(f"unsupported DUCA selector input shape: {tuple(dense_inputs.shape)}")
    return hard_base + (context - context.detach()) * slot.to(dtype=context.dtype) * bridge


def _add_soft_to_hard_resample_gradient_path(
    hard_selected: torch.Tensor,
    dense_inputs: torch.Tensor,
    *,
    selected_positions: torch.Tensor,
    slot_mask: torch.Tensor,
    center_scores: torch.Tensor,
    radius: torch.Tensor,
    valid_mask: torch.Tensor,
    bridge_weight: float = 1.0,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    bridge = float(bridge_weight)
    if bridge <= 0.0:
        return hard_selected, None
    context_inputs = dense_inputs if torch.is_floating_point(dense_inputs) or torch.is_complex(dense_inputs) else dense_inputs.float()
    hard_base = hard_selected if torch.is_floating_point(hard_selected) or torch.is_complex(hard_selected) else hard_selected.float()
    batch, temporal_len = int(center_scores.shape[0]), int(center_scores.shape[1])
    if selected_positions.shape[0] != batch:
        raise ValueError("selected_positions batch must match center_scores")
    positions = torch.arange(temporal_len, device=center_scores.device, dtype=center_scores.dtype)
    selected = selected_positions.to(device=center_scores.device, dtype=torch.long).clamp_min(0)
    selected_float = selected.to(dtype=center_scores.dtype)
    slot_radius = torch.gather(radius.to(device=center_scores.device, dtype=center_scores.dtype), 1, selected)
    sigma = slot_radius.clamp_min(0.0) + 1.0
    distance = (positions[None, None, :] - selected_float[:, :, None]).abs()
    logits = center_scores[:, None, :] - distance / sigma[:, :, None].clamp_min(torch.finfo(center_scores.dtype).eps)
    valid = valid_mask.to(device=center_scores.device, dtype=torch.bool)
    logits = logits.masked_fill(~valid[:, None, :], torch.finfo(center_scores.dtype).min / 4.0)
    weights = torch.softmax(logits, dim=-1).masked_fill(~valid[:, None, :], 0.0)
    weights = weights * slot_mask.to(device=center_scores.device, dtype=weights.dtype)[:, :, None]
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
    weights = weights.to(device=context_inputs.device, dtype=context_inputs.dtype)
    if context_inputs.ndim == 3:
        soft = torch.einsum("bct,bkt->bck", context_inputs, weights)
        slot = slot_mask[:, None, :]
    elif context_inputs.ndim == 5:
        soft = torch.einsum("bcthw,bkt->bckhw", context_inputs, weights)
        slot = slot_mask[:, None, :, None, None]
    elif context_inputs.ndim == 6:
        soft = torch.einsum("bncthw,bkt->bnckhw", context_inputs, weights)
        slot = slot_mask[:, None, None, :, None, None]
    else:
        raise ValueError(f"unsupported DUCA selector input shape: {tuple(dense_inputs.shape)}")
    return hard_base + (soft - soft.detach()) * slot.to(dtype=soft.dtype) * bridge, weights


def _add_structured_zero_forward_gradient_path(
    hard_selected: torch.Tensor,
    dense_inputs: torch.Tensor,
    *,
    soft_slot_assignment: torch.Tensor,
    slot_mask: torch.Tensor,
    bridge_weight: float,
) -> torch.Tensor:
    """Legacy local surrogate; forbidden as detector-utility evidence.

    Its hard forward is exact, but its gradient need not agree with the loss
    change from an actual discrete one-swap selection. Formal transition-only
    configs must use detached hard counterfactual utility distillation instead.
    """
    if soft_slot_assignment.ndim != 3:
        raise ValueError("structured soft_slot_assignment must be [B,K,T]")
    temporal_dim = 2 if dense_inputs.ndim in {3, 5} else 3 if dense_inputs.ndim == 6 else None
    if temporal_dim is None:
        raise ValueError(f"unsupported DUCA selector input shape: {tuple(dense_inputs.shape)}")
    expected = (int(hard_selected.shape[0]), int(slot_mask.shape[1]), int(dense_inputs.shape[temporal_dim]))
    if tuple(soft_slot_assignment.shape) != expected:
        raise ValueError(
            "structured soft_slot_assignment shape must match [batch, selected slots, dense time]: "
            f"expected {expected}, got {tuple(soft_slot_assignment.shape)}"
        )
    if slot_mask.shape != soft_slot_assignment.shape[:2]:
        raise ValueError("slot_mask must match structured soft_slot_assignment [B,K]")
    if not torch.isfinite(soft_slot_assignment).all():
        raise ValueError("structured soft_slot_assignment must be finite")
    if torch.any(soft_slot_assignment < 0):
        raise ValueError("structured soft_slot_assignment must be non-negative")
    slot_mass = soft_slot_assignment.sum(dim=-1)
    active = slot_mask.to(device=slot_mass.device, dtype=torch.bool)
    if active.any() and not torch.allclose(
        slot_mass[active],
        torch.ones_like(slot_mass[active]),
        atol=1.0e-4,
        rtol=1.0e-4,
    ):
        raise ValueError("every active structured slot assignment must sum to one")
    if (~active).any() and not torch.allclose(
        slot_mass[~active],
        torch.zeros_like(slot_mass[~active]),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError("inactive structured slot assignments must have zero mass")
    bridge = float(bridge_weight)
    if bridge <= 0.0:
        return hard_selected
    context_inputs = dense_inputs if torch.is_floating_point(dense_inputs) or torch.is_complex(dense_inputs) else dense_inputs.float()
    hard_base = hard_selected if torch.is_floating_point(hard_selected) or torch.is_complex(hard_selected) else hard_selected.float()
    weights = soft_slot_assignment.to(device=context_inputs.device, dtype=context_inputs.dtype)
    if context_inputs.ndim == 3:
        soft = torch.einsum("bct,bkt->bck", context_inputs, weights)
        slot = slot_mask[:, None, :]
    elif context_inputs.ndim == 5:
        soft = torch.einsum("bcthw,bkt->bckhw", context_inputs, weights)
        slot = slot_mask[:, None, :, None, None]
    elif context_inputs.ndim == 6:
        soft = torch.einsum("bncthw,bkt->bnckhw", context_inputs, weights)
        slot = slot_mask[:, None, None, :, None, None]
    return hard_base + bridge * (soft - soft.detach()) * slot.to(dtype=soft.dtype)


def _add_protected_structured_transport_gradient_path(
    hard_selected: torch.Tensor,
    dense_inputs: torch.Tensor,
    *,
    selected_positions: torch.Tensor,
    soft_slot_assignment: torch.Tensor,
    slot_mask: torch.Tensor,
    bridge_weight: float,
    bridge_row_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Use a hard-anchored local transport derivative for detector feedback.

    The forward value is the exact hard gather. Backward uses the expected
    ordered slot position from the same exact-K/max-gap structured family and
    the local temporal derivative around each actually selected observation.
    This remains a surrogate and must pass the hard-swap alignment gate before
    any full training run is admissible.
    """

    if soft_slot_assignment.ndim != 3:
        raise ValueError("protected structured soft_slot_assignment must be [B,K,T]")
    temporal_dim = 2 if dense_inputs.ndim in {3, 5} else 3 if dense_inputs.ndim == 6 else None
    if temporal_dim is None:
        raise ValueError(f"unsupported DUCA selector input shape: {tuple(dense_inputs.shape)}")
    expected = (int(hard_selected.shape[0]), int(slot_mask.shape[1]), int(dense_inputs.shape[temporal_dim]))
    if tuple(soft_slot_assignment.shape) != expected:
        raise ValueError(
            "protected structured soft_slot_assignment shape must match [batch, selected slots, dense time]: "
            f"expected {expected}, got {tuple(soft_slot_assignment.shape)}"
        )
    if selected_positions.shape != slot_mask.shape or slot_mask.shape != soft_slot_assignment.shape[:2]:
        raise ValueError("selected_positions and slot_mask must match protected structured slots [B,K]")
    if not torch.isfinite(soft_slot_assignment).all():
        raise ValueError("protected structured soft_slot_assignment must be finite")
    if torch.any(soft_slot_assignment < 0):
        raise ValueError("protected structured soft_slot_assignment must be non-negative")
    slot_mass = soft_slot_assignment.sum(dim=-1)
    active = slot_mask.to(device=slot_mass.device, dtype=torch.bool)
    if active.any() and not torch.allclose(
        slot_mass[active],
        torch.ones_like(slot_mass[active]),
        atol=1.0e-4,
        rtol=1.0e-4,
    ):
        raise ValueError("every active protected structured slot assignment must sum to one")
    if (~active).any() and not torch.allclose(
        slot_mass[~active],
        torch.zeros_like(slot_mass[~active]),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError("inactive protected structured slot assignments must have zero mass")
    if active.any():
        active_positions = selected_positions.to(device=active.device)[active]
        if torch.any(active_positions < 0) or torch.any(active_positions >= expected[2]):
            raise ValueError("active protected structured selected_positions are out of dense-time bounds")

    bridge = float(bridge_weight)
    if bridge <= 0.0:
        return hard_selected, None
    if bridge_row_scale is None:
        row_scale = torch.ones(
            int(hard_selected.shape[0]),
            device=hard_selected.device,
            dtype=torch.float32,
        )
    else:
        if bridge_row_scale.ndim != 1 or int(bridge_row_scale.shape[0]) != int(
            hard_selected.shape[0]
        ):
            raise ValueError("protected structured bridge_row_scale must be [B]")
        if not torch.isfinite(bridge_row_scale).all() or torch.any(bridge_row_scale < 0):
            raise ValueError("protected structured bridge_row_scale must be finite and non-negative")
        row_scale = bridge_row_scale.to(device=hard_selected.device, dtype=torch.float32)
    context_inputs = (
        dense_inputs
        if torch.is_floating_point(dense_inputs) or torch.is_complex(dense_inputs)
        else dense_inputs.float()
    )
    hard_base = (
        hard_selected
        if torch.is_floating_point(hard_selected) or torch.is_complex(hard_selected)
        else hard_selected.float()
    )
    weights = soft_slot_assignment.to(device=context_inputs.device, dtype=context_inputs.dtype)
    time_axis = torch.arange(expected[2], device=weights.device, dtype=weights.dtype)
    expected_positions = torch.einsum("bkt,t->bk", weights, time_axis)

    hard_positions = selected_positions.to(device=context_inputs.device, dtype=torch.long).clamp_min(0)
    left_positions = (hard_positions - 1).clamp_min(0)
    right_positions = (hard_positions + 1).clamp_max(expected[2] - 1)
    left = _gather_time(context_inputs, left_positions, slot_mask)
    right = _gather_time(context_inputs, right_positions, slot_mask)
    denominator = (right_positions - left_positions).clamp_min(1).to(dtype=context_inputs.dtype)
    if context_inputs.ndim == 3:
        slope = (right - left) / denominator[:, None, :]
        displacement = (expected_positions - expected_positions.detach())[:, None, :]
        slot = slot_mask[:, None, :]
    elif context_inputs.ndim == 5:
        slope = (right - left) / denominator[:, None, :, None, None]
        displacement = (expected_positions - expected_positions.detach())[:, None, :, None, None]
        slot = slot_mask[:, None, :, None, None]
    else:
        slope = (right - left) / denominator[:, None, None, :, None, None]
        displacement = (expected_positions - expected_positions.detach())[:, None, None, :, None, None]
        slot = slot_mask[:, None, None, :, None, None]
    row_scale_view = row_scale.to(dtype=displacement.dtype).view(
        int(row_scale.shape[0]), *([1] * (displacement.ndim - 1))
    )
    surrogate_delta = slope.detach() * displacement * row_scale_view
    bridged = hard_base + bridge * surrogate_delta * slot.to(dtype=surrogate_delta.dtype)
    return bridged, expected_positions


@SELECTORS.register_module()
class DucaOnlineFrameSelector(nn.Module):
    """Registry-buildable full-window DUCA selector for offline TAD."""

    def __init__(
        self,
        in_channels: int,
        budget: Optional[int] = 384,
        budget_mode: str = "fixed",
        budget_min: int = 64,
        budget_max: Optional[int] = None,
        budget_multiple: int = 16,
        target_budget: Optional[float] = None,
        allow_external_budget_override: Optional[bool] = None,
        max_radius: int = 16,
        acquisition_policy: str = "legacy_center_radius",
        structured_temperature: float = 1.0,
        local_cell_force_exact_uniform: bool = False,
        inference_policy_alpha: float = 1.0,
        training_uniform_companion_fraction: float = 0.0,
        training_uniform_companion_normalize_learned_gradient: bool = False,
        dense_window_size: Optional[int] = None,
        selector_hidden_channels: int = 0,
        coarse_trunk_lr: float = 2.5e-5,
        action_head_lr: float = 5.0e-5,
        transition_scorer_lr: float = 1.0e-4,
        actionness_weight: float = 0.05,
        transition_weight: float = 1.0,
        uncertainty_weight: float = 0.25,
        utility_weight: float = 0.50,
        boundary_weight: float = 1.0,
        selector_variant: str = "direct_boundary",
        coarse_hidden_dim: Optional[int] = None,
        use_coarse_hidden_features: bool = True,
        require_coarse_hidden_features: Optional[bool] = None,
        allow_frozen_coarse_probe: bool = False,
        policy_hidden_gradient_scale: float = 0.0,
        auxiliary_hidden_gradient_scale: float = 1.0,
        max_unselected_hole: Optional[int] = None,
        hard_max_gap_repair: bool = True,
        fail_on_infeasible_max_gap: bool = True,
        max_gap_loss_max_unselected_hole: Optional[int] = None,
        max_gap_loss_min_window_mass: float = 1.0,
        soft_max_gap_loss_enabled: bool = True,
        transition_target_sigma: float = 2.0,
        transition_target_radius: int = 4,
        transition_boundary_radius: int = 4,
        transition_distribution_temperature: float = 0.7,
        actionness_source_cfg: Optional[Mapping[str, Any]] = None,
        detector_gradient_mode: str = "st_sparse_gather",
        counterfactual_utility_distillation_weight: float = 0.0,
        counterfactual_utility_temperature: float = 1.0,
        counterfactual_max_candidates: int = 4,
        counterfactual_objective: str = "global_gram_proximal",
        require_counterfactual_utility_teacher: bool = False,
        coordinate_space: str = SELECTED_AXIS,
        detector_output_coordinate_space: str = SELECTED_AXIS,
        selected_positions_unit: str = "original_time_index",
        true_time_source_axis: str = TRUE_TIME_AXIS,
        temporal_sampling_contract: Optional[Mapping[str, Any]] = None,
        loss_weights: Optional[Mapping[str, float]] = None,
        actionness_loss_mode: str = "posterior_bce",
        strict_loss_contract: bool = False,
        loss_weight_schedule: Optional[Mapping[str, Any]] = None,
        no_ledger_decision: bool = True,
        remap_gt_to_selected_axis: bool = True,
        selected_axis_remap_required: bool = True,
        forbid_ledger: bool = True,
        forbid_raw_prediction_cache: bool = True,
        forbid_external_actionness: bool = False,
        external_actionness_meta_key: Optional[str] = None,
        external_actionness_logits_meta_key: Optional[str] = None,
        external_actionness_provenance_meta_key: Optional[str] = None,
        external_actionness_source_meta_key: Optional[str] = None,
        require_external_actionness: bool = False,
        profile_runtime: bool = False,
        profile_sync_cuda: bool = True,
        retain_gradient_audit_tensors: bool = False,
        metadata_keys: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.budget_mode = str(budget_mode)
        if self.budget_mode not in {"fixed", "dynamic_must"}:
            raise ValueError("budget_mode must be fixed or dynamic_must")
        if self.budget_mode == "dynamic_must":
            if budget is not None:
                raise ValueError("dynamic_must selector must set fixed budget=None and use budget_max")
            if budget_max is None:
                raise ValueError("dynamic_must selector requires budget_max")
            if allow_external_budget_override is True:
                raise ValueError("main dynamic_must selector forbids external budget override")
            self.budget = int(budget_max)
            self.allow_external_budget_override = False
        else:
            if budget is None:
                raise ValueError("fixed selector budget cannot be None")
            self.budget = int(budget)
            self.allow_external_budget_override = (
                True if allow_external_budget_override is None else bool(allow_external_budget_override)
            )
        self.budget_min = int(budget_min if self.budget_mode == "dynamic_must" else self.budget)
        self.budget_max = int(self.budget)
        self.budget_multiple = int(budget_multiple)
        self.target_budget = float(self.budget if target_budget is None else target_budget)
        self.max_radius = int(max_radius)
        self.acquisition_policy = str(acquisition_policy)
        self.structured_temperature = float(structured_temperature)
        self.local_cell_force_exact_uniform = bool(local_cell_force_exact_uniform)
        self.inference_policy_alpha = float(inference_policy_alpha)
        if not 0.0 <= self.inference_policy_alpha <= 1.0:
            raise ValueError("inference_policy_alpha must lie in [0,1]")
        self.training_uniform_companion_fraction = float(
            training_uniform_companion_fraction
        )
        self.training_uniform_companion_normalize_learned_gradient = bool(
            training_uniform_companion_normalize_learned_gradient
        )
        if (
            not math.isfinite(self.training_uniform_companion_fraction)
            or not 0.0 <= self.training_uniform_companion_fraction < 1.0
        ):
            raise ValueError(
                "training_uniform_companion_fraction must lie in [0,1)"
            )
        self.dense_window_size = None if dense_window_size is None else int(dense_window_size)
        self.coarse_trunk_lr = float(coarse_trunk_lr)
        self.action_head_lr = float(action_head_lr)
        self.transition_scorer_lr = float(transition_scorer_lr)
        if min(self.coarse_trunk_lr, self.action_head_lr, self.transition_scorer_lr) <= 0.0:
            raise ValueError("transition-only component learning rates must be positive")
        self.actionness_weight = float(actionness_weight)
        self.transition_weight = float(transition_weight)
        self.uncertainty_weight = float(uncertainty_weight)
        self.utility_weight = float(utility_weight)
        self.boundary_weight = float(boundary_weight)
        self.selector_variant = str(selector_variant)
        if self.selector_variant not in {"direct_boundary", "transition_only"}:
            raise ValueError("selector_variant must be direct_boundary or transition_only")
        self.coarse_hidden_dim = 0 if coarse_hidden_dim in (None, 0) else int(coarse_hidden_dim)
        if self.coarse_hidden_dim < 0:
            raise ValueError("coarse_hidden_dim must be non-negative")
        self.use_coarse_hidden_features = bool(use_coarse_hidden_features)
        self.require_coarse_hidden_features = (
            None if require_coarse_hidden_features is None else bool(require_coarse_hidden_features)
        )
        self.allow_frozen_coarse_probe = bool(allow_frozen_coarse_probe)
        self.policy_hidden_gradient_scale = float(policy_hidden_gradient_scale)
        if (
            not math.isfinite(self.policy_hidden_gradient_scale)
            or not 0.0 <= self.policy_hidden_gradient_scale <= 1.0
        ):
            raise ValueError("policy_hidden_gradient_scale must lie in [0,1]")
        self.auxiliary_hidden_gradient_scale = float(auxiliary_hidden_gradient_scale)
        if (
            not math.isfinite(self.auxiliary_hidden_gradient_scale)
            or not 0.0 <= self.auxiliary_hidden_gradient_scale <= 1.0
        ):
            raise ValueError("auxiliary_hidden_gradient_scale must lie in [0,1]")
        self.max_unselected_hole = None if max_unselected_hole in (None, 0) else int(max_unselected_hole)
        if self.max_unselected_hole is not None and self.max_unselected_hole < 0:
            raise ValueError("max_unselected_hole must be non-negative")
        self.hard_max_gap_repair = bool(hard_max_gap_repair)
        self.fail_on_infeasible_max_gap = bool(fail_on_infeasible_max_gap)
        self.max_gap_loss_max_unselected_hole = (
            self.max_unselected_hole
            if max_gap_loss_max_unselected_hole in (None, 0)
            else int(max_gap_loss_max_unselected_hole)
        )
        self.max_gap_loss_min_window_mass = float(max_gap_loss_min_window_mass)
        self.soft_max_gap_loss_enabled = bool(soft_max_gap_loss_enabled)
        self.transition_target_sigma = float(transition_target_sigma)
        self.transition_target_radius = int(transition_target_radius)
        self.transition_boundary_radius = int(transition_boundary_radius)
        self.transition_distribution_temperature = float(transition_distribution_temperature)
        if self.transition_target_sigma <= 0.0:
            raise ValueError("transition_target_sigma must be positive")
        if self.transition_distribution_temperature <= 0.0:
            raise ValueError("transition_distribution_temperature must be positive")
        if self.transition_target_radius < 0 or self.transition_boundary_radius < 0:
            raise ValueError("transition target/coverage radii must be non-negative")
        self.detector_gradient_mode = str(detector_gradient_mode)
        self.counterfactual_utility_distillation_weight = float(counterfactual_utility_distillation_weight)
        self.counterfactual_utility_temperature = float(counterfactual_utility_temperature)
        self.counterfactual_max_candidates = int(counterfactual_max_candidates)
        self.counterfactual_objective = str(counterfactual_objective)
        self.require_counterfactual_utility_teacher = bool(require_counterfactual_utility_teacher)
        if self.counterfactual_utility_distillation_weight < 0.0:
            raise ValueError("counterfactual_utility_distillation_weight must be non-negative")
        if self.counterfactual_utility_temperature <= 0.0:
            raise ValueError("counterfactual_utility_temperature must be positive")
        if self.counterfactual_max_candidates <= 0:
            raise ValueError("counterfactual_max_candidates must be positive")
        if self.counterfactual_objective not in {"global_gram_proximal", "local_cell_signed_logistic"}:
            raise ValueError(
                "counterfactual_objective must be global_gram_proximal or local_cell_signed_logistic"
            )
        if self.require_counterfactual_utility_teacher and self.counterfactual_utility_distillation_weight <= 0.0:
            raise ValueError("required counterfactual utility teacher needs a positive distillation weight")
        self.selected_positions_coordinate = str(coordinate_space)
        self.detector_output_coordinate_space = str(detector_output_coordinate_space)
        self.coordinate_space = self.detector_output_coordinate_space
        self.selected_positions_unit = str(selected_positions_unit)
        self.true_time_source_axis = str(true_time_source_axis)
        self.temporal_sampling_contract = (
            None
            if temporal_sampling_contract is None
            else DucaTemporalSamplingContract.from_mapping(dict(temporal_sampling_contract))
        )
        self.loss_weights = dict(loss_weights or {})
        self.actionness_loss_mode = str(actionness_loss_mode)
        if self.actionness_loss_mode not in {"posterior_bce", "class_balanced_mean"}:
            raise ValueError(
                "actionness_loss_mode must be posterior_bce or class_balanced_mean"
            )
        self.strict_loss_contract = bool(strict_loss_contract)
        if self.strict_loss_contract and set(self.loss_weights) != set(
            DUCA_LOSS_WEIGHT_DEFAULTS
        ):
            missing = sorted(set(DUCA_LOSS_WEIGHT_DEFAULTS) - set(self.loss_weights))
            extra = sorted(set(self.loss_weights) - set(DUCA_LOSS_WEIGHT_DEFAULTS))
            raise ValueError(
                "strict DUCA selector loss contract requires every loss weight "
                f"exactly once; missing={missing}, extra={extra}"
            )
        self.loss_weight_schedule = self._normalize_loss_weight_schedule(loss_weight_schedule)
        self.register_buffer("_loss_weight_schedule_step", torch.zeros((), dtype=torch.long), persistent=True)
        self.no_ledger_decision = bool(no_ledger_decision)
        self.remap_gt_to_selected_axis = bool(remap_gt_to_selected_axis)
        self.selected_axis_remap_required = bool(selected_axis_remap_required)
        self.forbid_ledger = bool(forbid_ledger)
        self.forbid_raw_prediction_cache = bool(forbid_raw_prediction_cache)
        self.forbid_external_actionness = bool(forbid_external_actionness)
        self.external_actionness_meta_key = external_actionness_meta_key
        self.external_actionness_logits_meta_key = external_actionness_logits_meta_key
        self.external_actionness_provenance_meta_key = external_actionness_provenance_meta_key
        self.external_actionness_source_meta_key = external_actionness_source_meta_key
        self.require_external_actionness = bool(require_external_actionness)
        self.profile_runtime = bool(profile_runtime)
        self.profile_sync_cuda = bool(profile_sync_cuda)
        self.retain_gradient_audit_tensors = bool(retain_gradient_audit_tensors)
        self.metadata_keys = dict(_DEFAULT_METADATA_KEYS)
        if metadata_keys:
            self.metadata_keys.update(dict(metadata_keys))
        self.extra_config = dict(kwargs)
        self.last_forward_summary: dict[str, Any] = {}
        self.last_dual_update_summary: dict[str, Any] = {}
        self.last_loss_schedule_update_summary: dict[str, Any] = {}
        self._pending_loss_schedule_advance = False
        self._pending_dynamic_budget_dual_mean: Optional[torch.Tensor] = None
        self._pending_dynamic_budget_dual_summary: Optional[dict[str, Any]] = None
        if self.detector_gradient_mode not in {
            "none",
            "st_sparse_gather",
            "st_sparse_gather_soft_context",
            "soft_to_hard_resample",
            "structured_zero_forward",
            "protected_structured_transport",
        }:
            raise ValueError(
                "detector_gradient_mode must be none, st_sparse_gather, "
                "st_sparse_gather_soft_context, soft_to_hard_resample, structured_zero_forward, "
                "or protected_structured_transport"
            )
        if self.selected_positions_coordinate not in {"original_time", SELECTED_AXIS, TRUE_TIME_AXIS}:
            raise ValueError("coordinate_space must describe original-time selected positions or selected-axis detector output")
        if self.detector_output_coordinate_space not in {SELECTED_AXIS, TRUE_TIME_AXIS}:
            raise ValueError("detector_output_coordinate_space must be selected-axis or true-time")
        if self.selected_positions_unit != "original_time_index":
            raise ValueError("selected_positions_unit must be original_time_index")
        if self.true_time_source_axis != TRUE_TIME_AXIS:
            raise ValueError("true_time_source_axis must be true_time_dense_index")
        if self.temporal_sampling_contract is not None:
            contract = self.temporal_sampling_contract
            if contract.hard_budget != self.budget:
                raise ValueError("temporal sampling contract hard_budget must match selector budget")
            if self.dense_window_size is None or contract.dense_window_size != self.dense_window_size:
                raise ValueError("temporal sampling contract dense_window_size must match selector")
            if contract.max_unselected_hole_dense_candidates != self.max_unselected_hole:
                raise ValueError("temporal sampling contract max gap must match selector max_unselected_hole")
            if contract.detector_axis != self.detector_output_coordinate_space:
                raise ValueError("temporal sampling contract detector_axis must match selector output axis")
        if self.dense_window_size is not None and self.dense_window_size <= 0:
            raise ValueError("dense_window_size must be positive")
        if not self.no_ledger_decision:
            raise ValueError("DUCA online selector requires no_ledger_decision=True")
        if self.detector_output_coordinate_space == SELECTED_AXIS and not self.remap_gt_to_selected_axis:
            raise ValueError("selected-axis detector output requires remap_gt_to_selected_axis=True")

        if self.forbid_external_actionness and (
            self.external_actionness_meta_key
            or self.external_actionness_logits_meta_key
            or self.require_external_actionness
        ):
            raise ValueError("forbid_external_actionness conflicts with configured external actionness inputs")
        if self.selector_variant == "transition_only":
            if self.budget_mode != "fixed":
                raise ValueError("transition_only currently supports only a fixed exact budget")
            if self.acquisition_policy not in {"global_structured_topk", "local_cell_deformation"}:
                raise ValueError("transition_only requires a structured exact-budget acquisition policy")
            if self.acquisition_policy == "global_structured_topk" and self.max_unselected_hole is None:
                raise ValueError("global transition_only requires max_unselected_hole")
            if not self.use_coarse_hidden_features:
                raise ValueError("transition_only requires official ASFormer encoder hidden features")
            if not self.forbid_external_actionness:
                raise ValueError("transition_only requires an in-graph coarse probe, not external actionness")
        if self.training_uniform_companion_fraction > 0.0:
            if self.selector_variant != "transition_only":
                raise ValueError(
                    "uniform companion training is restricted to transition_only"
                )
            if self.budget_mode != "fixed":
                raise ValueError(
                    "uniform companion training requires a fixed exact budget"
                )
            if self.acquisition_policy != "global_structured_topk":
                raise ValueError(
                    "uniform companion training requires global_structured_topk"
                )
            if self.detector_output_coordinate_space != SELECTED_AXIS:
                raise ValueError(
                    "uniform companion training is defined on the selected-axis detector path"
                )
        if self.training_uniform_companion_normalize_learned_gradient:
            if self.training_uniform_companion_fraction <= 0.0:
                raise ValueError(
                    "learned-row gradient normalization requires a positive uniform companion fraction"
                )
            if self.detector_gradient_mode != "protected_structured_transport":
                raise ValueError(
                    "learned-row gradient normalization requires protected_structured_transport"
                )
        if self.counterfactual_objective == "local_cell_signed_logistic":
            if self.acquisition_policy != "local_cell_deformation":
                raise ValueError("local-cell counterfactual utility requires local_cell_deformation")
            if self.local_cell_force_exact_uniform and self.counterfactual_utility_distillation_weight > 0.0:
                raise ValueError("the exact-uniform control must not request counterfactual utility")

        actionness_source = None
        self.raw_actionness_source = None
        self.actionness_source_name = "duca_adapter_internal"
        if actionness_source_cfg:
            cfg = dict(actionness_source_cfg)
            source_type = cfg.pop("type", "ZeroShotActionnessSource")
            self.actionness_source_name = str(cfg.get("source_name") or source_type)
            if source_type == "C3CoarseProbeActionnessSource":
                if self.selector_variant == "transition_only":
                    if str(cfg.get("probe_model", "")) != "official-action-seg":
                        raise ValueError("transition_only requires probe_model='official-action-seg'")
                    if str(cfg.get("official_action_seg_backend", "")) != "official_asformer":
                        raise ValueError("transition_only requires the official ASFormer backend")
                    coarse_frozen = bool(cfg.get("frozen", False)) or cfg.get("trainable") is False
                    if coarse_frozen and not self.allow_frozen_coarse_probe:
                        raise ValueError(
                            "transition_only frozen ASFormer requires "
                            "allow_frozen_coarse_probe=True"
                        )
                    cfg.setdefault("hidden_output_kind", "official_asformer_encoder_hidden")
                    if cfg["hidden_output_kind"] != "official_asformer_encoder_hidden":
                        raise ValueError("transition_only requires official ASFormer encoder hidden output")
                inferred_hidden_dim = int(
                    cfg.get("coarse_hidden_dim")
                    or cfg.get("tcn_hidden_dim")
                    or cfg.get("hidden_dim")
                    or 0
                )
                if self.coarse_hidden_dim <= 0 and inferred_hidden_dim > 0:
                    self.coarse_hidden_dim = inferred_hidden_dim
                cfg.setdefault("return_hidden_features", bool(self.use_coarse_hidden_features))
                cfg.setdefault(
                    "require_hidden_features",
                    bool(self.use_coarse_hidden_features and self.require_coarse_hidden_features is not False),
                )
                cfg.setdefault("source_name", self.actionness_source_name)
                self.raw_actionness_source = C3CoarseProbeActionnessSource(**cfg)
                actionness_source = ZeroShotActionnessSource(
                    mode="motion",
                    source_name=f"{self.actionness_source_name}_logit_passthrough",
                    thumos_trained=False,
                    uses_labels=False,
                    uses_teacher=False,
                    uses_gt=False,
                    uses_prediction_cache=False,
                    calibration_split="none",
                    checkpoint_hash="online_c3_coarse_probe_passthrough",
                )
            elif source_type in {"DucaOnlineProbeActionnessSource", "ZeroShotMotionActionnessSource"}:
                cfg.setdefault("mode", "motion")
                cfg.setdefault("source_name", self.actionness_source_name)
                cfg.setdefault("thumos_trained", False)
                cfg.setdefault("uses_labels", False)
                cfg.setdefault("uses_teacher", False)
                cfg = {key: value for key, value in cfg.items() if key in _ACTIONNESS_KWARGS}
                actionness_source = ZeroShotActionnessSource(**cfg)
            elif source_type == "ZeroShotActionnessSource":
                cfg = {key: value for key, value in cfg.items() if key in _ACTIONNESS_KWARGS}
                actionness_source = ZeroShotActionnessSource(**cfg)
            else:
                raise ValueError(f"unsupported actionness_source_cfg type {source_type!r}")
        self.adapter = DucaAcquisitionAdapter(
            feature_dim=self.in_channels,
            budget=None if self.budget_mode == "dynamic_must" else self.budget,
            budget_mode=self.budget_mode,
            budget_min=self.budget_min,
            budget_max=self.budget_max if self.budget_mode == "dynamic_must" else None,
            budget_multiple=self.budget_multiple,
            target_budget=self.target_budget,
            allow_external_budget_override=self.allow_external_budget_override,
            max_radius=self.max_radius,
            acquisition_policy=self.acquisition_policy,
            structured_temperature=self.structured_temperature,
            local_cell_force_exact_uniform=self.local_cell_force_exact_uniform,
            hidden_dim=int(selector_hidden_channels),
            actionness_source=actionness_source,
            actionness_weight=self.actionness_weight,
            transition_weight=self.transition_weight,
            uncertainty_weight=self.uncertainty_weight,
            utility_weight=self.utility_weight,
            boundary_weight=self.boundary_weight,
            selector_variant=self.selector_variant,
            coarse_hidden_dim=self.coarse_hidden_dim if self.use_coarse_hidden_features else 0,
            require_coarse_hidden_features=bool(
                self.use_coarse_hidden_features
                and self.raw_actionness_source is not None
                and self.require_coarse_hidden_features is not False
            ),
            policy_hidden_gradient_scale=self.policy_hidden_gradient_scale,
            auxiliary_hidden_gradient_scale=self.auxiliary_hidden_gradient_scale,
            max_unselected_hole=self.max_unselected_hole,
            hard_max_gap_repair=self.hard_max_gap_repair,
            fail_on_infeasible_max_gap=self.fail_on_infeasible_max_gap,
            profile_runtime=self.profile_runtime,
            profile_sync_cuda=self.profile_sync_cuda,
        )

    def capture_amp_replay_state(self) -> dict[str, Any]:
        """Capture non-buffer forward state that must not leak across a replay."""
        pending_dual = self._pending_dynamic_budget_dual_mean
        adapter = getattr(self, "adapter", None)
        return {
            "last_forward_summary": copy.deepcopy(self.last_forward_summary),
            "last_dual_update_summary": copy.deepcopy(self.last_dual_update_summary),
            "last_loss_schedule_update_summary": copy.deepcopy(
                self.last_loss_schedule_update_summary
            ),
            "last_counterfactual_summary": copy.deepcopy(
                getattr(self, "last_counterfactual_summary", None)
            ),
            "last_selected_positions": copy.deepcopy(
                getattr(self, "_last_selected_positions", None)
            ),
            "last_detector_grid_positions": copy.deepcopy(
                getattr(self, "_last_detector_grid_positions", None)
            ),
            "pending_loss_schedule_advance": bool(
                self._pending_loss_schedule_advance
            ),
            "pending_dynamic_budget_dual_mean": (
                None if pending_dual is None else pending_dual.detach().clone()
            ),
            "pending_dynamic_budget_dual_summary": copy.deepcopy(
                self._pending_dynamic_budget_dual_summary
            ),
            "adapter_last_compute_profile": copy.deepcopy(
                getattr(adapter, "last_compute_profile", None)
            ),
        }

    def restore_amp_replay_state(self, snapshot: Mapping[str, Any]) -> None:
        self.last_forward_summary = copy.deepcopy(snapshot["last_forward_summary"])
        self.last_dual_update_summary = copy.deepcopy(
            snapshot["last_dual_update_summary"]
        )
        self.last_loss_schedule_update_summary = copy.deepcopy(
            snapshot["last_loss_schedule_update_summary"]
        )
        self._restore_optional_replay_attribute(
            "last_counterfactual_summary",
            snapshot.get("last_counterfactual_summary"),
        )
        self._restore_optional_replay_attribute(
            "_last_selected_positions",
            snapshot.get("last_selected_positions"),
        )
        self._restore_optional_replay_attribute(
            "_last_detector_grid_positions",
            snapshot.get("last_detector_grid_positions"),
        )
        self._pending_loss_schedule_advance = bool(
            snapshot["pending_loss_schedule_advance"]
        )
        pending_dual = snapshot.get("pending_dynamic_budget_dual_mean")
        self._pending_dynamic_budget_dual_mean = (
            None if pending_dual is None else pending_dual.detach().clone()
        )
        self._pending_dynamic_budget_dual_summary = copy.deepcopy(
            snapshot.get("pending_dynamic_budget_dual_summary")
        )
        adapter = getattr(self, "adapter", None)
        if adapter is not None:
            adapter.last_compute_profile = copy.deepcopy(
                snapshot.get("adapter_last_compute_profile")
            )

    def _restore_optional_replay_attribute(self, name: str, value: Any) -> None:
        if value is None:
            if hasattr(self, name):
                delattr(self, name)
            return
        setattr(self, name, copy.deepcopy(value))

    def forward_train(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas,
        gt_segments=None,
        gt_labels=None,
        teacher_utility: Optional[torch.Tensor] = None,
        budget=None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._reject_external_actionness_payload(metas)
        self._reject_train_decision_payload(metas)
        action_target = self._action_target_from_gt_segments(gt_segments, masks)
        transition_target = None
        if self.selector_variant == "transition_only":
            start_target = end_target = context_target = None
            transition_target = self._transition_target_from_gt_segments(
                gt_segments,
                masks,
                sigma=self.transition_target_sigma,
                truncate_radius=self.transition_target_radius,
            )
            boundary_target = transition_target
            boundary_utility_proxy_target = None
        else:
            endpoint_targets = self._endpoint_targets_from_gt_segments(gt_segments, masks)
            if endpoint_targets is None:
                start_target = end_target = context_target = boundary_target = None
                boundary_utility_proxy_target = None
            else:
                start_target, end_target, context_target = endpoint_targets
                boundary_target = start_target + end_target
                boundary_utility_proxy_target = 0.45 * start_target + 0.45 * end_target + 0.10 * context_target
                proxy_mass = boundary_utility_proxy_target.sum(dim=1, keepdim=True)
                boundary_utility_proxy_target = torch.where(
                    proxy_mass > 0,
                    boundary_utility_proxy_target / proxy_mass.clamp_min(torch.finfo(proxy_mass.dtype).eps),
                    boundary_utility_proxy_target,
                )
        schedule_state = self._loss_schedule_state()
        outputs = self._forward_select(inputs, masks, metas, budget=budget, schedule_state=schedule_state)
        outputs["selector_outputs"]["loss_weight_schedule"] = schedule_state
        outputs["selector_outputs"]["training_provenance"] = {
            "uses_labels": gt_segments is not None,
            "uses_gt_segments": gt_segments is not None,
            "uses_gt_labels": gt_labels is not None,
            "label_scope": "train_only",
            "uses_labels_at_inference": False,
            "target_kinds": (
                ["coarse_actionness", "transition_boundary"]
                if self.selector_variant == "transition_only"
                else [
                    "coarse_actionness",
                    "start_endpoint",
                    "end_endpoint",
                    "boundary_context",
                ]
            )
            if gt_segments is not None
            else [],
        }
        if action_target is not None:
            outputs["selector_outputs"]["action_target"] = action_target
        if boundary_target is not None:
            outputs["selector_outputs"]["boundary_target"] = boundary_target
            if self.selector_variant == "transition_only":
                outputs["selector_outputs"]["transition_target"] = transition_target
                outputs["selector_outputs"]["transition_target_kind"] = (
                    "equal_mass_fixed_sigma_truncated_start_end_gaussians"
                )
            else:
                outputs["selector_outputs"]["start_target"] = start_target
                outputs["selector_outputs"]["end_target"] = end_target
                outputs["selector_outputs"]["context_target"] = context_target
        if boundary_utility_proxy_target is not None:
            outputs["selector_outputs"]["boundary_utility_proxy_target"] = boundary_utility_proxy_target
            outputs["selector_outputs"]["boundary_utility_proxy_target_kind"] = (
                "instance_normalized_start_end_context_proxy"
            )
            outputs["selector_outputs"]["detector_utility_target"] = boundary_utility_proxy_target
            outputs["selector_outputs"]["detector_utility_target_kind"] = "deprecated_alias_to_gt_boundary_utility_proxy"
        gt_segments, gt_labels, metas = self._remap_train_targets_to_selected_axis(
            gt_segments, gt_labels, outputs["metas"]
        )
        active_loss_weights = {
            key: float(schedule_state["weights"][key])
            for key in DUCA_LOSS_WEIGHT_DEFAULTS
            if key in schedule_state["weights"]
        }
        selector_losses = duca_losses(
            outputs["selector_outputs"],
            teacher_utility=teacher_utility,
            boundary_target=boundary_target,
            start_target=start_target if self.selector_variant == "direct_boundary" else None,
            end_target=end_target if self.selector_variant == "direct_boundary" else None,
            context_target=context_target if self.selector_variant == "direct_boundary" else None,
            action_target=action_target,
            transition_target=transition_target,
            boundary_utility_proxy_target=boundary_utility_proxy_target,
            transition_boundary_radius=self.transition_boundary_radius,
            transition_distribution_temperature=self.transition_distribution_temperature,
            max_unselected_hole=(
                self.max_gap_loss_max_unselected_hole if self.soft_max_gap_loss_enabled else 0
            ),
            max_gap_loss_min_window_mass=self.max_gap_loss_min_window_mass,
            actionness_loss_mode=self.actionness_loss_mode,
            loss_weights=active_loss_weights,
            strict_loss_contract=self.strict_loss_contract,
        )
        supervision_loss_audit = self._supervision_loss_audit(
            selector_losses,
            active_loss_weights,
        )
        outputs["selector_outputs"]["supervision_loss_audit"] = (
            supervision_loss_audit
        )
        if isinstance(self.last_forward_summary, dict):
            self.last_forward_summary["supervision_loss_audit"] = (
                supervision_loss_audit
            )
        if self.require_counterfactual_utility_teacher and teacher_utility is not None:
            raise RuntimeError("integrated counterfactual teacher forbids externally supplied teacher_utility")
        if self.counterfactual_utility_distillation_weight > 0.0 and teacher_utility is not None:
            if self.counterfactual_objective == "local_cell_signed_logistic":
                raise RuntimeError("local-cell utility requires the integrated hard-flip detector teacher")
            selector_losses["counterfactual_utility_distillation_loss"] = (
                    counterfactual_utility_distillation_loss(
                        outputs["selector_outputs"]["center_scores"],
                        teacher_utility,
                        outputs["selector_outputs"]["valid_mask"],
                        temperature=self.counterfactual_utility_temperature,
                    )
                    * self.counterfactual_utility_distillation_weight
                )
            outputs["selector_outputs"]["counterfactual_teacher_kind"] = (
                    "train_only_detached_hard_one_swap_detector_loss_reduction"
                )
            outputs["selector_outputs"]["counterfactual_direct_detector_gradient"] = False
        counterfactual_request = None
        if self.counterfactual_utility_distillation_weight > 0.0 and teacher_utility is None:
            positions = outputs["selector_outputs"]["grid"].selected_positions
            if self.counterfactual_objective == "local_cell_signed_logistic":
                cell_starts = outputs["selector_outputs"].get("local_cell_starts")
                cell_ends = outputs["selector_outputs"].get("local_cell_ends")
                if cell_starts is None or cell_ends is None:
                    raise RuntimeError("local-cell counterfactual teacher requires decoded cell bounds")
                counterfactual_request = build_local_cell_hard_flip_candidates(
                    positions,
                    outputs["selector_outputs"]["center_scores"],
                    outputs["selector_outputs"]["valid_mask"],
                    cell_starts,
                    cell_ends,
                    outputs["selector_outputs"].get("detector_grid_positions"),
                    max_candidates=self.counterfactual_max_candidates,
                )
            else:
                counterfactual_request = build_finite_hard_one_swap_candidates(
                    positions,
                    outputs["selector_outputs"]["center_scores"],
                    outputs["selector_outputs"]["valid_mask"],
                    max_candidates=self.counterfactual_max_candidates,
                    max_unselected_hole=self.max_unselected_hole,
                )
        self._record_pending_loss_schedule_step()
        return {
            "inputs": outputs["inputs"],
            "masks": outputs["masks"],
            "metas": metas,
            "gt_segments": gt_segments,
            "gt_labels": gt_labels,
            "losses": selector_losses,
            "selector_outputs": outputs["selector_outputs"],
            "counterfactual_request": counterfactual_request,
        }

    @staticmethod
    def _supervision_loss_audit(selector_losses, weights):
        audit = {}
        for loss_name, weight_name in DUCA_LOSS_TO_WEIGHT_KEY.items():
            value = selector_losses.get(loss_name)
            if not torch.is_tensor(value):
                raise RuntimeError(f"DUCA loss inventory is missing {loss_name}")
            weight = float(weights.get(weight_name, 0.0))
            weighted = float(value.detach().float().cpu().item())
            audit[loss_name] = {
                "weight_name": weight_name,
                "weight": weight,
                "active": bool(weight != 0.0),
                "requires_grad": bool(value.requires_grad),
                "weighted": weighted,
                "unweighted": None if weight == 0.0 else weighted / weight,
            }
        return audit

    def counterfactual_distillation_loss(
        self,
        selector_outputs,
        candidate_positions,
        replaced_slots,
        candidate_utility,
        candidate_valid,
        *,
        baseline_detector_loss=None,
        candidate_detector_loss=None,
    ):
        valid = candidate_valid.bool()
        center_scores = selector_outputs["center_scores"]
        baseline_positions = selector_outputs["grid"].selected_positions
        pair_scores = counterfactual_pair_scores(
            center_scores,
            candidate_positions,
            replaced_slots,
            baseline_positions,
            valid,
        )
        if self.counterfactual_objective == "local_cell_signed_logistic":
            loss = local_cell_signed_logistic_loss(
                center_scores,
                candidate_positions,
                replaced_slots,
                baseline_positions,
                candidate_utility.detach(),
                valid,
                temperature=self.counterfactual_utility_temperature,
            ) * self.counterfactual_utility_distillation_weight
            distillation_loss_kind = "distinct_local_cell_weighted_signed_logistic"
            gradient_alignment = {
                "status": "not_computed_in_training_forward",
                "spearman": 0.0,
                "sign_agreement": 0.0,
            }
            alignment_available = False
        else:
            swap_incidence = build_swap_incidence_matrix(
                center_scores,
                candidate_positions,
                replaced_slots,
                baseline_positions,
                valid,
            )
            loss = signed_one_swap_proximal_loss(
                center_scores,
                swap_incidence,
                candidate_utility.detach(),
                valid,
                temperature=self.counterfactual_utility_temperature,
            ) * self.counterfactual_utility_distillation_weight
            distillation_loss_kind = "swap_gram_whitened_signed_proximal"
            try:
                gradient_alignment = score_space_utility_alignment(
                    center_scores,
                    loss,
                    swap_incidence,
                    candidate_utility.detach(),
                    valid,
                    temperature=self.counterfactual_utility_temperature,
                )
                alignment_available = True
            except ValueError:
                gradient_alignment = {"spearman": 0.0, "sign_agreement": 0.0}
                alignment_available = False
        with torch.no_grad():
            student_pair = pair_scores[valid].float()
            utility = candidate_utility.detach()[valid].float()
            if (
                student_pair.numel() >= 2
                and not bool(torch.all(student_pair == student_pair[0]))
                and not bool(torch.all(utility == utility[0]))
            ):
                student_rank = torch.argsort(torch.argsort(student_pair)).float()
                utility_rank = torch.argsort(torch.argsort(utility)).float()
                centered_d = student_rank - student_rank.mean()
                centered_u = utility_rank - utility_rank.mean()
                denom = centered_d.norm() * centered_u.norm()
                spearman = float((centered_d @ centered_u / denom.clamp_min(1e-12)).item())
            else:
                spearman = 0.0
            informative = (student_pair != 0) & (utility != 0)
            distinct_utility = bool(
                utility.numel() >= 2 and torch.unique(utility).numel() >= 2
            )
            sign = (
                float(((student_pair[informative] * utility[informative]) > 0).float().mean().item())
                if informative.any() else 0.0
            )
            safe_slots = replaced_slots.clamp_min(0)
            removed_positions = torch.gather(baseline_positions.clamp_min(0), 1, safe_slots)
            utility_consistency_max_abs_error = None
            baseline_values = []
            candidate_loss_values = []
            if baseline_detector_loss is not None and candidate_detector_loss is not None:
                baseline_tensor = baseline_detector_loss.detach().float()
                candidate_loss_tensor = candidate_detector_loss.detach().float()
                if bool(valid.any()):
                    expected_utility = baseline_tensor[:, None] - candidate_loss_tensor
                    utility_consistency_max_abs_error = float(
                        (expected_utility[valid] - candidate_utility.detach().float()[valid]).abs().max().item()
                    )
                    baseline_values = [
                        float(baseline_tensor[batch_index].item())
                        for batch_index in range(int(baseline_tensor.shape[0]))
                        for _ in range(int(valid[batch_index].sum().item()))
                    ]
                    candidate_loss_values = [
                        float(value) for value in candidate_loss_tensor[valid].cpu().tolist()
                    ]
                else:
                    utility_consistency_max_abs_error = 0.0
            self.last_counterfactual_summary = {
                "teacher_kind": (
                    "detached_distinct_local_cell_hard_flip_official_actionformer_cls_plus_reg"
                    if self.counterfactual_objective == "local_cell_signed_logistic"
                    else "detached_hard_one_swap_official_actionformer_cls_plus_reg"
                ),
                "candidate_count": int(candidate_valid.sum().item()),
                "direct_detector_gradient": False,
                "sign_agreement": sign,
                "spearman": spearman,
                "finite": bool(torch.isfinite(candidate_utility[candidate_valid]).all().item()),
                "distillation_loss_kind": distillation_loss_kind,
                "no_op_teacher_utility": 0.0,
                "no_op_student_score_delta": 0.0,
                "no_op_role": "fixed_score_delta_reference_not_competition_class",
                "candidate_utility_positive_count": int((utility > 0).sum().item()),
                "candidate_utility_negative_count": int((utility < 0).sum().item()),
                "candidate_utility_zero_count": int((utility == 0).sum().item()),
                "candidate_utility_values": [float(value) for value in utility.cpu().tolist()],
                "student_pair_score_values": [float(value) for value in student_pair.cpu().tolist()],
                "candidate_add_positions": [int(value) for value in candidate_positions[valid].cpu().tolist()],
                "candidate_remove_positions": [int(value) for value in removed_positions[valid].cpu().tolist()],
                "candidate_cell_indices": [int(value) for value in replaced_slots[valid].cpu().tolist()],
                "baseline_detector_loss_values": baseline_values,
                "candidate_detector_loss_values": candidate_loss_values,
                "utility_consistency_max_abs_error": utility_consistency_max_abs_error,
                "alignment_kind": "score_space_pair_direction_vs_signed_detector_swap_gain",
                "distillation_gradient_alignment": gradient_alignment,
                "distillation_gradient_alignment_available": bool(alignment_available and distinct_utility),
                "utility_alignment_informative": distinct_utility,
            }
        return loss

    @staticmethod
    def _normalize_loss_weight_schedule(config: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
        if config is None:
            return None
        if not isinstance(config, Mapping):
            raise ValueError("loss_weight_schedule must be a mapping")
        out = dict(config)
        schedule_type = str(out.get("type", "progressive_joint"))
        if schedule_type not in {"progressive_joint", "constant"}:
            raise ValueError("loss_weight_schedule.type must be progressive_joint or constant")
        out["type"] = schedule_type
        out["warmup_steps"] = int(out.get("warmup_steps", 0))
        out["transition_steps"] = int(out.get("transition_steps", out.get("ramp_steps", 1)))
        if out["warmup_steps"] < 0:
            raise ValueError("loss_weight_schedule.warmup_steps must be non-negative")
        if out["transition_steps"] < 0:
            raise ValueError("loss_weight_schedule.transition_steps must be non-negative")
        out["shape"] = str(out.get("shape", out.get("curve", "linear"))).lower()
        if out["shape"] not in {"linear", "cosine"}:
            raise ValueError("loss_weight_schedule.shape must be linear or cosine")
        entries: dict[str, dict[str, Any]] = {}
        reserved = {"type", "warmup_steps", "transition_steps", "ramp_steps", "shape", "curve", "enabled"}
        for key, value in out.items():
            if key in reserved:
                continue
            if isinstance(value, Mapping):
                if "start" not in value or "end" not in value:
                    raise ValueError(f"loss_weight_schedule.{key} must define start and end")
                entry_warmup = int(value.get("warmup_steps", out["warmup_steps"]))
                entry_transition = int(value.get("transition_steps", out["transition_steps"]))
                entry_shape = str(value.get("shape", out["shape"])).lower()
                if entry_warmup < 0 or entry_transition < 0:
                    raise ValueError(f"loss_weight_schedule.{key} step counts must be non-negative")
                if entry_shape not in {"linear", "cosine"}:
                    raise ValueError(f"loss_weight_schedule.{key}.shape must be linear or cosine")
                entries[str(key)] = {
                    "start": float(value["start"]),
                    "end": float(value["end"]),
                    "warmup_steps": entry_warmup,
                    "transition_steps": entry_transition,
                    "shape": entry_shape,
                }
        out["entries"] = entries
        return out

    def _loss_schedule_state(self) -> dict[str, Any]:
        step = int(self._loss_weight_schedule_step.detach().item())
        weights = {str(key): float(value) for key, value in self.loss_weights.items()}
        if self.loss_weight_schedule is None or self.loss_weight_schedule.get("type") == "constant":
            return {
                "enabled": False,
                "type": "constant",
                "step": step,
                "progress": 1.0,
                "phase": "constant_joint",
                "weights": weights,
                "detector_gradient_weight": 1.0,
            }
        progress = self._loss_schedule_progress(step)
        entry_progress = {}
        for key, entry in self.loss_weight_schedule["entries"].items():
            item_progress = self._interpolation_progress(
                step,
                warmup=int(entry["warmup_steps"]),
                transition=int(entry["transition_steps"]),
                shape=str(entry["shape"]),
            )
            weights[key] = float(entry["start"] + (entry["end"] - entry["start"]) * item_progress)
            entry_progress[key] = float(item_progress)
        if self.selector_variant == "transition_only":
            policy_alpha = float(weights.get("policy_alpha", 1.0))
            detector_gradient = float(weights.get("detector_gradient", 0.0))
            detector_end = float(
                self.loss_weight_schedule["entries"].get("detector_gradient", {}).get("end", detector_gradient)
            )
            if policy_alpha <= 0.0:
                phase = "uniform_policy_coarse_transition_learning"
            elif policy_alpha < 1.0:
                phase = "continuous_policy_homotopy"
            elif detector_gradient < detector_end:
                phase = "protected_detector_bridge_homotopy"
            else:
                phase = "joint_transition_detection"
        elif progress <= 0.0:
            phase = "coarse_actionness_warmup"
        elif progress >= 1.0:
            phase = "joint_detection_selection"
        else:
            phase = "transition_detector_utility"
        return {
            "enabled": True,
            "type": str(self.loss_weight_schedule["type"]),
            "shape": str(self.loss_weight_schedule["shape"]),
            "step": step,
            "warmup_steps": int(self.loss_weight_schedule["warmup_steps"]),
            "transition_steps": int(self.loss_weight_schedule["transition_steps"]),
            "progress": float(progress),
            "phase": phase,
            "weights": weights,
            "entry_progress": entry_progress,
            "detector_gradient_weight": float(weights.get("detector_gradient", weights.get("detector", 1.0))),
        }

    def _loss_schedule_progress(self, step: int) -> float:
        if self.loss_weight_schedule is None:
            return 1.0
        return self._interpolation_progress(
            step,
            warmup=int(self.loss_weight_schedule["warmup_steps"]),
            transition=int(self.loss_weight_schedule["transition_steps"]),
            shape=str(self.loss_weight_schedule.get("shape", "linear")),
        )

    @staticmethod
    def _interpolation_progress(step: int, *, warmup: int, transition: int, shape: str) -> float:
        if step <= warmup:
            raw = 0.0
        elif transition <= 0:
            raw = 1.0
        else:
            raw = min(1.0, max(0.0, float(step - warmup) / float(transition)))
        if shape == "cosine":
            pi = torch.acos(torch.zeros((), dtype=torch.float64)).item() * 2.0
            raw = 0.5 - 0.5 * torch.cos(torch.tensor(raw * pi, dtype=torch.float64)).item()
        return float(raw)

    def _use_stable_structured_selection(self, schedule_state: Optional[Mapping[str, Any]]) -> bool:
        if self.selector_variant == "transition_only":
            return False
        if self.acquisition_policy != "global_structured_topk" or not self.training:
            return False
        if not isinstance(schedule_state, Mapping) or not bool(schedule_state.get("enabled", False)):
            return False
        progress = float(schedule_state.get("progress", 1.0))
        if progress <= 0.0:
            return True
        if progress >= 1.0:
            return False
        step = int(schedule_state.get("step", 0))
        learned_threshold = int(round(progress * 100.0))
        return (step % 100) >= learned_threshold

    def _record_pending_loss_schedule_step(self) -> None:
        self._pending_loss_schedule_advance = bool(self.training and self.loss_weight_schedule is not None)

    def after_optimizer_step(self) -> Optional[dict[str, Any]]:
        schedule_summary = self._advance_loss_schedule_step_after_optimizer_step()
        if self.budget_mode != "dynamic_must":
            return schedule_summary
        return self._update_dynamic_budget_dual_after_optimizer_step()

    def _advance_loss_schedule_step_after_optimizer_step(self) -> dict[str, Any]:
        if not self._pending_loss_schedule_advance:
            return {"updated": False, "reason": "no_pending_loss_schedule_step"}
        before = int(self._loss_weight_schedule_step.detach().item())
        self._loss_weight_schedule_step.add_(1)
        after = int(self._loss_weight_schedule_step.detach().item())
        self._pending_loss_schedule_advance = False
        summary = {
            "updated": True,
            "source": "optimizer_step",
            "step_before": before,
            "step_after": after,
        }
        self.last_loss_schedule_update_summary = summary
        if isinstance(self.last_forward_summary, dict):
            self.last_forward_summary["loss_schedule_step_update"] = summary
        return summary

    def _update_dynamic_budget_dual_after_optimizer_step(self) -> dict[str, Any]:
        pending = self._pending_dynamic_budget_dual_mean
        if pending is None:
            return {"updated": False, "reason": "no_pending_dynamic_budget"}
        controller = getattr(self.adapter, "budget_controller", None)
        if controller is None or not hasattr(controller, "update_dual"):
            return {"updated": False, "reason": "missing_dynamic_budget_controller"}
        before = controller.lambda_dual.detach().clone()
        updated = controller.update_dual(pending)
        after = updated.detach()
        summary = dict(self._pending_dynamic_budget_dual_summary or {})
        summary.update(
            {
                "updated": True,
                "source": "dynamic_must_expected_cost",
                "lambda_before": float(before.detach().cpu().item()),
                "lambda_after": float(after.detach().cpu().item()),
            }
        )
        self.last_dual_update_summary = summary
        if isinstance(self.last_forward_summary, dict):
            self.last_forward_summary["dynamic_budget_dual_update"] = summary
        self._pending_dynamic_budget_dual_mean = None
        self._pending_dynamic_budget_dual_summary = None
        return summary

    @staticmethod
    def _action_target_from_gt_segments(gt_segments, masks: torch.Tensor) -> Optional[torch.Tensor]:
        if gt_segments is None:
            return None
        if masks.ndim != 2:
            raise ValueError("DUCA action target generation expects dense masks [B,T]")
        batch, temporal_len = int(masks.shape[0]), int(masks.shape[1])
        if len(gt_segments) != batch:
            raise ValueError("gt_segments length must match batch size for DUCA action target generation")
        device = masks.device
        dtype = torch.float32
        centers = torch.arange(temporal_len, device=device, dtype=dtype)
        target = torch.zeros(batch, temporal_len, device=device, dtype=dtype)
        valid = masks.to(device=device, dtype=torch.bool)
        for batch_idx, segments in enumerate(gt_segments):
            if segments is None:
                continue
            seg = segments if torch.is_tensor(segments) else torch.as_tensor(segments, dtype=dtype)
            seg = seg.to(device=device, dtype=dtype)
            if seg.numel() == 0:
                continue
            seg = seg.reshape(-1, 2)
            starts = torch.minimum(seg[:, 0], seg[:, 1])[:, None]
            ends = torch.maximum(seg[:, 0], seg[:, 1])[:, None]
            covered = ((centers[None, :] >= starts) & (centers[None, :] < ends)).any(dim=0)
            target[batch_idx] = covered.to(dtype=dtype)
        return target.masked_fill(~valid, 0.0)

    @staticmethod
    def _endpoint_targets_from_gt_segments(
        gt_segments,
        masks: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if gt_segments is None:
            return None
        if masks.ndim != 2:
            raise ValueError("DUCA endpoint target generation expects dense masks [B,T]")
        batch, temporal_len = int(masks.shape[0]), int(masks.shape[1])
        if len(gt_segments) != batch:
            raise ValueError("gt_segments length must match batch size for DUCA endpoint target generation")
        device = masks.device
        dtype = torch.float32
        centers = torch.arange(temporal_len, device=device, dtype=dtype)
        start_target = torch.zeros(batch, temporal_len, device=device, dtype=dtype)
        end_target = torch.zeros(batch, temporal_len, device=device, dtype=dtype)
        context_target = torch.zeros(batch, temporal_len, device=device, dtype=dtype)
        valid = masks.to(device=device, dtype=torch.bool)
        for batch_idx, segments in enumerate(gt_segments):
            if segments is None:
                continue
            seg = segments if torch.is_tensor(segments) else torch.as_tensor(segments, dtype=dtype)
            seg = seg.to(device=device, dtype=dtype)
            if seg.numel() == 0:
                continue
            seg = seg.reshape(-1, 2)
            starts = torch.minimum(seg[:, 0], seg[:, 1])
            ends = torch.maximum(seg[:, 0], seg[:, 1])
            row_valid = valid[batch_idx].to(dtype=dtype)
            for start, end in zip(starts, ends):
                duration = (end - start).clamp_min(1.0)
                sigma = (0.08 * duration).clamp(min=0.75, max=4.0)
                start_kernel = torch.exp(-0.5 * ((centers - start) / sigma).square()) * row_valid
                end_kernel = torch.exp(-0.5 * ((centers - end) / sigma).square()) * row_valid
                start_target[batch_idx] += start_kernel / start_kernel.sum().clamp_min(1.0e-6)
                end_target[batch_idx] += end_kernel / end_kernel.sum().clamp_min(1.0e-6)

                context_offset = (1.5 * sigma).clamp(min=1.0, max=6.0)
                pre_kernel = torch.exp(-0.5 * ((centers - (start - context_offset)) / sigma).square()) * row_valid
                post_kernel = torch.exp(-0.5 * ((centers - (end + context_offset)) / sigma).square()) * row_valid
                context_target[batch_idx] += 0.5 * pre_kernel / pre_kernel.sum().clamp_min(1.0e-6)
                context_target[batch_idx] += 0.5 * post_kernel / post_kernel.sum().clamp_min(1.0e-6)
        return (
            start_target.masked_fill(~valid, 0.0),
            end_target.masked_fill(~valid, 0.0),
            context_target.masked_fill(~valid, 0.0),
        )

    @staticmethod
    def _transition_target_from_gt_segments(
        gt_segments,
        masks: torch.Tensor,
        *,
        sigma: float,
        truncate_radius: int,
    ) -> Optional[torch.Tensor]:
        if gt_segments is None:
            return None
        if masks.ndim != 2:
            raise ValueError("DUCA transition target generation expects dense masks [B,T]")
        batch, temporal_len = int(masks.shape[0]), int(masks.shape[1])
        if len(gt_segments) != batch:
            raise ValueError("gt_segments length must match batch size for transition targets")
        sigma = float(sigma)
        truncate_radius = int(truncate_radius)
        if sigma <= 0.0 or truncate_radius < 0:
            raise ValueError("transition target sigma/radius are invalid")
        device = masks.device
        centers = torch.arange(temporal_len, device=device, dtype=torch.float32)
        valid = masks.to(device=device, dtype=torch.bool)
        target = torch.zeros(batch, temporal_len, device=device, dtype=torch.float32)
        for batch_idx, segments in enumerate(gt_segments):
            if segments is None:
                continue
            seg = segments if torch.is_tensor(segments) else torch.as_tensor(segments, dtype=torch.float32)
            seg = seg.to(device=device, dtype=torch.float32).reshape(-1, 2)
            if seg.numel() == 0:
                continue
            endpoints = torch.stack(
                (torch.minimum(seg[:, 0], seg[:, 1]), torch.maximum(seg[:, 0], seg[:, 1])),
                dim=1,
            ).reshape(-1)
            row_valid = valid[batch_idx].to(dtype=torch.float32)
            endpoint_mass = 1.0 / float(endpoints.numel())
            for endpoint in endpoints:
                distance = centers - endpoint
                kernel = torch.exp(-0.5 * (distance / sigma).square())
                kernel = kernel * (distance.abs() <= float(truncate_radius)).to(kernel.dtype) * row_valid
                kernel_mass = kernel.sum()
                if float(kernel_mass.detach().item()) > 0.0:
                    target[batch_idx] += endpoint_mass * kernel / kernel_mass
        return target.masked_fill(~valid, 0.0)

    @staticmethod
    def _boundary_target_from_gt_segments(
        gt_segments,
        masks: torch.Tensor,
        *,
        boundary_radius: int,
    ) -> Optional[torch.Tensor]:
        del boundary_radius
        targets = DucaOnlineFrameSelector._endpoint_targets_from_gt_segments(gt_segments, masks)
        if targets is None:
            return None
        start_target, end_target, _ = targets
        return start_target + end_target

    @staticmethod
    def _boundary_utility_proxy_target_from_gt_segments(
        gt_segments,
        masks: torch.Tensor,
        *,
        boundary_radius: int,
    ) -> Optional[torch.Tensor]:
        del boundary_radius
        targets = DucaOnlineFrameSelector._endpoint_targets_from_gt_segments(gt_segments, masks)
        if targets is None:
            return None
        start_target, end_target, context_target = targets
        target = 0.45 * start_target + 0.45 * end_target + 0.10 * context_target
        mass = target.sum(dim=1, keepdim=True)
        return torch.where(
            mass > 0,
            target / mass.clamp_min(torch.finfo(target.dtype).eps),
            target,
        )

    def forward_test(self, inputs: torch.Tensor, masks: torch.Tensor, metas=None, budget=None, **kwargs: Any) -> dict[str, Any]:
        self._reject_external_actionness_payload(metas)
        _assert_no_forbidden_payload({"metas": metas, "kwargs": kwargs})
        outputs = self._forward_select(inputs, masks, metas, budget=budget)
        return {
            "inputs": outputs["inputs"],
            "masks": outputs["masks"],
            "metas": outputs["metas"],
            "selector_outputs": outputs["selector_outputs"],
        }

    def _apply_training_uniform_companion(self, grid, scores, valid_mask):
        companion_mask = _training_uniform_companion_mask(
            int(valid_mask.shape[0]),
            fraction=self.training_uniform_companion_fraction,
            device=valid_mask.device,
        )
        scores["training_uniform_companion_mask"] = companion_mask
        scores["training_uniform_companion_fraction"] = float(
            self.training_uniform_companion_fraction
        )
        if not bool(companion_mask.any().item()):
            return companion_mask

        slot_count = int(grid.selected_positions.shape[1])
        score_dtype = scores["center_scores"].dtype
        uniform_positions, uniform_dense_mask, uniform_assignment = (
            _exact_uniform_companion_tensors(
                valid_mask,
                slot_count=slot_count,
                dtype=score_dtype,
            )
        )
        mask_bk = companion_mask[:, None]
        mask_bkt = companion_mask[:, None, None]
        blended_positions = torch.where(
            mask_bk,
            uniform_positions,
            grid.selected_positions,
        )
        blended_dense_mask = torch.where(
            mask_bk,
            uniform_dense_mask,
            grid.selected_mask.bool(),
        )

        structured_assignment = scores.get("structured_soft_slot_assignment")
        if structured_assignment is None:
            raise ValueError(
                "uniform companion requires structured exact-K slot marginals"
            )
        scores["structured_soft_slot_assignment"] = torch.where(
            mask_bkt,
            uniform_assignment,
            structured_assignment,
        )
        selected_mask_st = scores.get("selected_mask_st")
        if selected_mask_st is None:
            raise ValueError("uniform companion requires selected_mask_st")
        scores["selected_mask_st"] = torch.where(
            mask_bk,
            uniform_dense_mask.to(dtype=selected_mask_st.dtype),
            selected_mask_st,
        )
        detector_positions = scores.get("detector_grid_positions")
        if detector_positions is not None:
            scores["detector_grid_positions"] = torch.where(
                mask_bk,
                uniform_positions,
                detector_positions,
            )
        scores["selected_indices_st"] = blended_positions
        scores["training_uniform_companion_positions"] = uniform_positions

        grid.selected_positions = blended_positions
        grid.selected_mask = blended_dense_mask
        grid.detector_input_length = blended_dense_mask.long().sum(dim=1)
        grid.metadata = dict(grid.metadata)
        grid.metadata.update(
            {
                "training_uniform_companion": True,
                "training_uniform_companion_fraction": float(
                    self.training_uniform_companion_fraction
                ),
                "training_uniform_companion_count": int(
                    companion_mask.long().sum().item()
                ),
                "inference_uniform_companion": False,
            }
        )
        scores["decode_metadata"] = grid.metadata
        grid.validate()
        return companion_mask

    def _forward_select(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas,
        budget=None,
        schedule_state: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        profile_enabled = bool(self.profile_runtime)
        sync_enabled = profile_enabled and bool(self.profile_sync_cuda)
        total_start = _sync_profile_clock(inputs, enabled=sync_enabled) if profile_enabled else None
        descriptor_start = _sync_profile_clock(inputs, enabled=sync_enabled) if profile_enabled else None
        if self.selector_variant == "transition_only":
            descriptors = torch.zeros(
                int(masks.shape[0]),
                int(masks.shape[1]),
                self.in_channels,
                device=inputs.device,
                dtype=torch.float32,
            )
        else:
            descriptors = _time_descriptors_btc(inputs)
        descriptor_ms = _elapsed_ms(descriptor_start, inputs, enabled=sync_enabled)
        descriptor_profile = self._descriptor_compute_profile(inputs, descriptors)
        if self.selector_variant == "transition_only":
            descriptor_profile = {
                "operation": "omitted_raw_rgb_descriptor_for_transition_only",
                "input_shape": [int(item) for item in inputs.shape],
                "output_shape": [int(item) for item in descriptors.shape],
                "estimated_macs": 0,
                "estimated_flops": 0,
                "uses_input_values": False,
            }
        if descriptors.shape[-1] != self.in_channels:
            raise ValueError(f"DUCA selector expected {self.in_channels} channels, got {descriptors.shape[-1]}")
        masks = masks.to(device=inputs.device, dtype=torch.bool)
        external_actionness = self._external_actionness_from_metas(metas, descriptors=descriptors)
        online_actionness = None
        if external_actionness is None and self.raw_actionness_source is not None:
            online_actionness = self.raw_actionness_source(inputs, valid_mask=masks)
        profile_context = {
            "external_cached_actionness": external_actionness is not None,
            "external_actionness_source_name": (
                None if external_actionness is None else external_actionness.get("source_name")
            ),
            "descriptor_profile": descriptor_profile,
        }
        actionness_logits = None
        p_action = None
        actionness_provenance = None
        coarse_hidden_features = None
        coarse_policy_hidden_features = None
        coarse_hidden_kind = None
        if external_actionness is not None:
            actionness_logits = external_actionness.get("actionness_logits")
            p_action = external_actionness.get("p_action")
            actionness_provenance = external_actionness["provenance"]
        elif online_actionness is not None:
            actionness_logits = online_actionness.get("logits")
            if actionness_logits is None:
                actionness_logits = online_actionness.get("actionness_logits")
            p_action = online_actionness.get("p_action")
            actionness_provenance = online_actionness["provenance"]
            if self.use_coarse_hidden_features:
                coarse_hidden_features = online_actionness.get("coarse_hidden_features")
                if coarse_hidden_features is None:
                    coarse_hidden_features = online_actionness.get("hidden_features")
                coarse_policy_hidden_features = online_actionness.get(
                    "policy_hidden_features"
                )
                coarse_hidden_kind = online_actionness.get("hidden_kind")
        stable_selection = self._use_stable_structured_selection(schedule_state)
        policy_mix_alpha = self.inference_policy_alpha
        if self.training and self.selector_variant == "transition_only" and isinstance(schedule_state, Mapping):
            schedule_weights = schedule_state.get("weights", {})
            if isinstance(schedule_weights, Mapping):
                policy_mix_alpha = float(schedule_weights.get("policy_alpha", schedule_state.get("progress", 1.0)))
        grid, scores = self.adapter.acquire(
            descriptors,
            budget=budget,
            valid_mask=masks,
            actionness_logits=actionness_logits,
            p_action=p_action,
            actionness_provenance=actionness_provenance,
            coarse_hidden_features=coarse_hidden_features,
            coarse_policy_hidden_features=coarse_policy_hidden_features,
            coarse_hidden_kind=coarse_hidden_kind,
            compute_profile_context=profile_context,
            stable_selection=stable_selection,
            policy_mix_alpha=policy_mix_alpha,
        )
        uniform_companion_mask = torch.zeros(
            int(masks.shape[0]),
            device=masks.device,
            dtype=torch.bool,
        )
        if self.training and self.training_uniform_companion_fraction > 0.0:
            uniform_companion_mask = self._apply_training_uniform_companion(
                grid,
                scores,
                masks,
            )
        companion_bridge_scale = _training_uniform_companion_bridge_scale(
            uniform_companion_mask,
            normalize_learned_gradient=(
                self.training
                and self.training_uniform_companion_normalize_learned_gradient
            ),
        ).to(device=inputs.device)
        scores["training_uniform_companion_bridge_scale"] = companion_bridge_scale
        actionness_source_name = self.actionness_source_name
        if external_actionness is not None:
            scores["external_actionness_provenance"] = external_actionness["provenance"]
            scores["external_actionness_source"] = external_actionness["source_name"]
            actionness_source_name = external_actionness["source_name"]
        elif online_actionness is not None:
            scores["online_actionness_provenance"] = online_actionness["provenance"]
            scores["online_actionness_source"] = online_actionness["source_name"]
            actionness_source_name = online_actionness["source_name"]
        validate_actionness_provenance(scores.get("provenance", {}), context="DUCA selector actionness provenance")
        positions = grid.selected_positions.to(device=inputs.device)
        detector_grid_positions = scores.get("detector_grid_positions")
        if detector_grid_positions is None:
            detector_grid_positions = positions
        detector_grid_positions = detector_grid_positions.to(device=inputs.device, dtype=torch.long)
        if detector_grid_positions.shape != positions.shape:
            raise ValueError("detector_grid_positions must align with acquisition positions")
        if not torch.equal(detector_grid_positions >= 0, positions >= 0):
            raise ValueError("detector-grid and acquisition slot masks must be identical")
        temporal_contract_audit = None
        if self.temporal_sampling_contract is not None:
            temporal_contract_audit = self.temporal_sampling_contract.audit_positions(positions, masks)
            scores["temporal_sampling_contract_audit"] = temporal_contract_audit
        self._last_selected_positions = positions.detach()
        self._last_detector_grid_positions = detector_grid_positions.detach()
        slot_mask = positions >= 0
        gather_start = _sync_profile_clock(inputs, enabled=sync_enabled) if profile_enabled else None
        hard_gathered = _gather_time(inputs, positions, slot_mask)
        hard_selected = hard_gathered
        # The surrogate exists only to carry detector gradients during training.
        # Running its zero-forward raw-pixel mixture in eval changes no values and
        # only adds dense compute and memory traffic.
        detector_gradient_weight = self._detector_gradient_weight(schedule_state) if self.training else 0.0
        soft_resample_weights = None
        structured_expected_positions = None
        bridge_start = _sync_profile_clock(inputs, enabled=sync_enabled) if profile_enabled else None
        bridge_ms = None
        if self.detector_gradient_mode == "st_sparse_gather_soft_context":
            hard_selected = _add_soft_context_gradient_path(
                hard_selected,
                inputs,
                scores["soft_coverage"],
                slot_mask,
                bridge_weight=detector_gradient_weight,
            )
        elif self.detector_gradient_mode == "soft_to_hard_resample":
            hard_selected, soft_resample_weights = _add_soft_to_hard_resample_gradient_path(
                hard_selected,
                inputs,
                selected_positions=positions,
                slot_mask=slot_mask,
                center_scores=scores["center_scores"],
                radius=scores["radius"],
                valid_mask=masks,
                bridge_weight=detector_gradient_weight,
            )
        elif self.detector_gradient_mode == "structured_zero_forward":
            assignment = scores.get("structured_soft_slot_assignment")
            if assignment is None:
                raise ValueError("structured_zero_forward requires structured slot marginals")
            hard_selected = _add_structured_zero_forward_gradient_path(
                hard_selected,
                inputs,
                soft_slot_assignment=assignment,
                slot_mask=slot_mask,
                bridge_weight=detector_gradient_weight,
            )
        elif self.detector_gradient_mode == "protected_structured_transport":
            assignment = scores.get("structured_soft_slot_assignment")
            if assignment is None:
                raise ValueError("protected_structured_transport requires structured slot marginals")
            hard_selected, structured_expected_positions = _add_protected_structured_transport_gradient_path(
                hard_selected,
                inputs,
                selected_positions=positions,
                soft_slot_assignment=assignment,
                slot_mask=slot_mask,
                bridge_weight=detector_gradient_weight,
                bridge_row_scale=companion_bridge_scale,
            )
        bridge_ms = _elapsed_ms(bridge_start, inputs, enabled=sync_enabled)
        hard_slot_weights = slot_mask.to(dtype=scores["center_scores"].dtype)
        if self.detector_gradient_mode in {
            "none",
            "structured_zero_forward",
            "protected_structured_transport",
        }:
            st_weights = hard_slot_weights
        else:
            st_weights = torch.gather(scores["selected_mask_st"], 1, positions.clamp_min(0)) * slot_mask.to(
                dtype=scores["selected_mask_st"].dtype
            )
            hard_slot_weights = slot_mask.to(dtype=st_weights.dtype)
            st_weights = hard_slot_weights + float(detector_gradient_weight) * (st_weights - st_weights.detach())
        selected_inputs = _apply_slot_weights(hard_selected, st_weights)
        gather_ms = _elapsed_ms(gather_start, inputs, enabled=sync_enabled)
        total_selector_ms = _elapsed_ms(total_start, inputs, enabled=sync_enabled)
        compute_profile = dict(scores.get("compute_profile", {}))
        latency_ms = dict(compute_profile.get("latency_ms", {}))
        latency_ms.update(
            {
                "enabled": profile_enabled,
                "descriptor_ms": descriptor_ms,
                "detector_gradient_bridge_ms": bridge_ms,
                "gather_ms": gather_ms,
                "total_selector_ms": total_selector_ms,
            }
        )
        compute_profile["latency_ms"] = latency_ms
        compute_profile["descriptor"] = descriptor_profile
        if external_actionness is not None:
            actionness_profile = dict(compute_profile.get("actionness", {}))
            actionness_profile.update(
                {
                    "source_name": external_actionness["source_name"],
                    "source_kind": "external_cached_prior",
                    "cache_lookup_or_interpolation": True,
                    "online_backbone_flops_included": False,
                }
            )
            compute_profile["actionness"] = actionness_profile
            compute_profile["pre_backbone_model"] = "ExternalCachedActionness+DUCASelectorMLP"
        elif online_actionness is not None:
            compute_profile = self._replace_actionness_profile(
                compute_profile,
                dict(online_actionness.get("compute_profile", {})),
            )
            if self.selector_variant == "transition_only":
                compute_profile["pre_backbone_model"] = "OfficialASFormer+TransitionUtilityScorer"
        self._add_detector_gradient_bridge_profile(
            compute_profile,
            dense_inputs=inputs,
            batch_size=int(descriptors.shape[0]),
            slot_count=int(positions.shape[1]),
            temporal_len=int(descriptors.shape[1]),
            mode=self.detector_gradient_mode,
            bridge_weight=float(detector_gradient_weight),
            bridge_ms=bridge_ms,
        )
        scores["compute_profile"] = compute_profile
        selected_masks = slot_mask.to(device=inputs.device, dtype=torch.bool)
        scores["grid"] = grid
        scores["hard_selected_inputs"] = hard_gathered
        scores["selected_input_st_gradient_path"] = self.detector_gradient_mode
        scores["detector_gradient_weight"] = float(detector_gradient_weight)
        if soft_resample_weights is not None:
            scores["soft_resample_weights"] = soft_resample_weights
        if structured_expected_positions is not None:
            scores["structured_expected_positions"] = structured_expected_positions
        if self.retain_gradient_audit_tensors:
            self._gradient_audit_tensors = {
                "center_scores": scores["center_scores"],
                "structured_soft_slot_assignment": scores.get("structured_soft_slot_assignment"),
                "selected_positions": positions,
                "detector_gradient_weight": float(detector_gradient_weight),
            }
        scores["sparse_grid"] = grid
        selected_counts = selected_masks.long().sum(dim=1).detach().cpu().tolist()
        self._record_pending_dynamic_budget_dual(scores, grid)
        self.last_forward_summary = {
            self.metadata_keys["selected_count"]: int(selected_counts[0]) if selected_counts else 0,
            "budget": int(grid.budget),
            "requested_budget": [
                int(item) for item in grid.requested_budget.detach().cpu().reshape(-1).tolist()
            ],
            "effective_budget": [
                int(item) for item in grid.effective_budget.detach().cpu().reshape(-1).tolist()
            ],
            "dynamic_budget": bool(grid.metadata.get("budget_is_dynamic", False)),
            "budget_policy": str(grid.metadata.get("budget_policy", "fixed_budget")),
            "budget_unit": grid.budget_unit,
            "coordinate": grid.coordinate,
            self.metadata_keys["source"]: actionness_source_name,
            "compute_profile": compute_profile,
            "detector_gradient_weight": float(detector_gradient_weight),
            "uses_coarse_hidden_features": bool(scores.get("uses_coarse_hidden_features", False)),
            "max_unselected_hole": self.max_unselected_hole,
            "selection_path": scores.get("selection_path", "legacy_center_radius"),
            "selector_variant": self.selector_variant,
            "coarse_hidden_kind": scores.get("coarse_hidden_kind"),
            "policy_hidden_gradient_scale": self.policy_hidden_gradient_scale,
            "policy_mix_alpha": float(scores.get("policy_mix_alpha", policy_mix_alpha)),
            "training_uniform_companion_fraction": float(
                self.training_uniform_companion_fraction
            ),
            "training_uniform_companion_count": int(
                uniform_companion_mask.long().sum().detach().cpu().item()
            ),
            "training_uniform_companion_normalize_learned_gradient": bool(
                self.training_uniform_companion_normalize_learned_gradient
            ),
            "training_uniform_companion_bridge_scale": [
                float(value)
                for value in companion_bridge_scale.detach().cpu().tolist()
            ],
            "inference_uniform_companion": False,
        }
        if temporal_contract_audit is not None:
            self.last_forward_summary["temporal_sampling_contract"] = temporal_contract_audit
        if isinstance(schedule_state, Mapping):
            self.last_forward_summary["loss_weight_schedule"] = dict(schedule_state)
        return {
            "inputs": selected_inputs,
            "masks": selected_masks,
            "metas": self._write_metas(
                metas,
                grid,
                detector_grid_positions=detector_grid_positions,
                actionness_source_name=actionness_source_name,
                compute_profile=compute_profile,
            ),
            "selector_outputs": scores,
        }

    @staticmethod
    def _add_detector_gradient_bridge_profile(
        profile: dict[str, Any],
        *,
        dense_inputs: torch.Tensor,
        batch_size: int,
        slot_count: int,
        temporal_len: int,
        mode: str,
        bridge_weight: float,
        bridge_ms: Optional[float],
    ) -> None:
        components = dict(profile.get("components", {}))
        component_name = str(mode)
        supported_modes = {
            "soft_to_hard_resample",
            "structured_zero_forward",
            "protected_structured_transport",
        }
        enabled = component_name in supported_modes and float(bridge_weight) > 0.0
        dense_input_elements = int(dense_inputs.numel())
        if temporal_len <= 0 or dense_input_elements % int(temporal_len) != 0:
            raise ValueError("detector gradient bridge requires a valid dense temporal dimension")
        soft_selected_output_elements = (dense_input_elements // int(temporal_len)) * int(slot_count)
        if enabled and component_name == "protected_structured_transport":
            macs = int(batch_size * slot_count * temporal_len + 3 * soft_selected_output_elements)
            complexity = "O(B*K*T + numel(selected_raw_video)) hard-anchored local transport"
            accounting_scope = "expected_position_and_local_temporal_slope_lower_bound"
        else:
            macs = dense_input_elements * int(slot_count) if enabled else 0
            complexity = "O(numel(dense_raw_video)*K) soft slot resampling" if enabled else "disabled"
            accounting_scope = "dominant_einsum_lower_bound"
        softmax_flops = (
            int(batch_size * slot_count * temporal_len * 8)
            if enabled and component_name == "soft_to_hard_resample"
            else 0
        )
        flops = int(2 * macs + softmax_flops)
        context_element_size = int(dense_inputs.element_size()) if dense_inputs.is_floating_point() else 4
        component = {
            "enabled": bool(enabled),
            "mode": component_name,
            "slot_count": int(slot_count),
            "dense_temporal_len": int(temporal_len),
            "dense_input_shape": [int(value) for value in dense_inputs.shape],
            "dense_input_dtype": str(dense_inputs.dtype),
            "dense_input_elements": dense_input_elements,
            "estimated_macs": int(macs),
            "estimated_flops": int(flops),
            "dense_float_copy_bytes": (
                dense_input_elements * 4 if enabled and not dense_inputs.is_floating_point() else 0
            ),
            "soft_selected_output_elements": int(soft_selected_output_elements),
            "soft_selected_output_bytes": (
                int(soft_selected_output_elements * context_element_size) if enabled else 0
            ),
            "slot_assignment_elements": int(batch_size * slot_count * temporal_len),
            "complexity": complexity,
            "accounting_scope": accounting_scope,
            "estimated_flops_are_lower_bound": True,
            "complete_memory_accounting": False,
        }
        components[component_name] = component
        profile["components"] = components
        profile["estimated_macs"] = int(profile.get("estimated_macs", 0) or 0) + int(macs)
        profile["estimated_flops"] = int(profile.get("estimated_flops", 0) or 0) + int(flops)
        if enabled:
            profile["estimated_flops_are_lower_bound"] = True
            profile["complete_memory_accounting"] = False
        profile["detector_gradient_bridge"] = {
            "mode": component_name,
            "bridge_weight": float(bridge_weight),
            "latency_ms": bridge_ms,
            "component": component_name,
            "training_only": True,
        }

    def _record_pending_dynamic_budget_dual(self, scores: Mapping[str, Any], grid) -> None:
        self._pending_dynamic_budget_dual_mean = None
        self._pending_dynamic_budget_dual_summary = None
        if not self.training or self.budget_mode != "dynamic_must":
            return
        decision = scores.get("budget_decision")
        if decision is None:
            return
        expected = decision.expected_cost.detach().float()
        observed_mean = expected.mean()
        hard_mean = decision.budget_hard.detach().float().mean()
        selected_mean = grid.selected_count.detach().float().mean()
        target = float(decision.target_budget)
        self._pending_dynamic_budget_dual_mean = observed_mean
        self._pending_dynamic_budget_dual_summary = {
            "observed_mean_budget": float(observed_mean.detach().cpu().item()),
            "hard_mean_budget": float(hard_mean.detach().cpu().item()),
            "selected_mean_budget": float(selected_mean.detach().cpu().item()),
            "target_budget": target,
            "budget_min": int(decision.budget_min),
            "budget_max": int(decision.budget_max),
            "budget_multiple": int(decision.budget_multiple),
        }

    @staticmethod
    def _detector_gradient_weight(schedule_state: Optional[Mapping[str, Any]]) -> float:
        if not isinstance(schedule_state, Mapping):
            return 1.0
        if "detector_gradient_weight" in schedule_state:
            return float(schedule_state["detector_gradient_weight"])
        weights = schedule_state.get("weights", {})
        if isinstance(weights, Mapping):
            return float(weights.get("detector_gradient", weights.get("detector", 1.0)))
        return 1.0

    @staticmethod
    def _replace_actionness_profile(profile: Mapping[str, Any], actionness_profile: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(profile)
        components = dict(out.get("components", {}))
        old = dict(components.get("actionness", {}))
        new = dict(actionness_profile)
        components["actionness"] = new
        out["components"] = components
        out["actionness"] = new
        old_macs = int(old.get("estimated_macs", 0) or 0)
        old_flops = int(old.get("estimated_flops", 0) or 0)
        new_macs = int(new.get("estimated_macs", 0) or 0)
        new_flops = int(new.get("estimated_flops", 0) or 0)
        out["estimated_macs"] = int(out.get("estimated_macs", 0) or 0) - old_macs + new_macs
        out["estimated_flops"] = int(out.get("estimated_flops", 0) or 0) - old_flops + new_flops
        old_params = old.get("parameters", {})
        new_params = new.get("parameters", {})
        if isinstance(old_params, Mapping) and isinstance(new_params, Mapping):
            total_params = int(out.get("parameters", {}).get("total", 0) or 0)
            trainable_params = int(out.get("parameters", {}).get("trainable", 0) or 0)
            total_params = total_params - int(old_params.get("total", 0) or 0) + int(new_params.get("total", 0) or 0)
            trainable_params = trainable_params - int(old_params.get("trainable", 0) or 0) + int(
                new_params.get("trainable", 0) or 0
            )
            out["parameters"] = {"total": int(total_params), "trainable": int(trainable_params)}
        out["pre_backbone_model"] = f"{new.get('model_family', new.get('source_name', 'C3CoarseProbe'))}+DUCASelectorMLP"
        latency = dict(out.get("latency_ms", {}))
        probe_latency = new.get("latency_ms")
        if isinstance(probe_latency, Mapping):
            latency.update(probe_latency)
        out["latency_ms"] = latency
        out["flop_accounting"] = "static_estimate_for_online_coarse_probe_plus_selector_path_excludes_detector_backbone"
        return out

    @staticmethod
    def _descriptor_compute_profile(inputs: torch.Tensor, descriptors: torch.Tensor) -> dict[str, Any]:
        shape = [int(item) for item in inputs.shape]
        output_shape = [int(item) for item in descriptors.shape]
        if inputs.ndim == 3:
            return {
                "operation": "transpose_feature_sequence",
                "input_shape": shape,
                "output_shape": output_shape,
                "estimated_macs": 0,
                "estimated_flops": 0,
            }
        if inputs.ndim == 5:
            batch, channels, temporal, height, width = [int(item) for item in inputs.shape]
            reduce_count = int(height * width)
            flops = batch * channels * temporal * max(reduce_count, 1)
            return {
                "operation": "spatial_mean_to_rgb_temporal_descriptor",
                "input_shape": shape,
                "output_shape": output_shape,
                "estimated_macs": 0,
                "estimated_flops": int(flops),
                "reduction_elements_per_descriptor": reduce_count,
            }
        if inputs.ndim == 6:
            batch, clips, channels, temporal, height, width = [int(item) for item in inputs.shape]
            reduce_count = int(clips * height * width)
            flops = batch * channels * temporal * max(reduce_count, 1)
            return {
                "operation": "clip_spatial_mean_to_rgb_temporal_descriptor",
                "input_shape": shape,
                "output_shape": output_shape,
                "estimated_macs": 0,
                "estimated_flops": int(flops),
                "reduction_elements_per_descriptor": reduce_count,
            }
        return {
            "operation": "unknown_descriptor",
            "input_shape": shape,
            "output_shape": output_shape,
            "estimated_macs": 0,
            "estimated_flops": 0,
        }

    def _external_actionness_from_metas(self, metas, descriptors: torch.Tensor) -> Optional[dict[str, Any]]:
        self._reject_external_actionness_payload(metas)
        if not (
            self.external_actionness_meta_key
            or self.external_actionness_logits_meta_key
            or self.require_external_actionness
        ):
            return None
        if metas is None:
            if self.require_external_actionness:
                raise ValueError("external actionness is required but metas are missing")
            return None
        if len(metas) != descriptors.shape[0]:
            raise ValueError("external actionness metas length must match batch size")

        p_rows = []
        logit_rows = []
        provenances = []
        source_names = []
        need_p = self.external_actionness_meta_key
        need_logits = self.external_actionness_logits_meta_key
        for batch_idx, meta in enumerate(metas):
            if not isinstance(meta, Mapping):
                raise ValueError(f"metas[{batch_idx}] must be a mapping for external actionness")
            p_value = self._lookup_meta_value(meta, need_p) if need_p else None
            logits_value = self._lookup_meta_value(meta, need_logits) if need_logits else None
            if p_value is None and logits_value is None:
                if self.require_external_actionness:
                    raise ValueError(f"external actionness is required for metas[{batch_idx}]")
                return None
            if self.external_actionness_provenance_meta_key is None:
                raise ValueError("external actionness provenance meta key is required")
            provenance = self._lookup_meta_value(meta, self.external_actionness_provenance_meta_key)
            validate_actionness_provenance(provenance, context=f"external actionness provenance metas[{batch_idx}]")
            source_name = None
            if self.external_actionness_source_meta_key:
                source_name = self._lookup_meta_value(meta, self.external_actionness_source_meta_key)
            source_name = str(source_name or provenance.get("source_name") or "external_actionness")
            source_names.append(source_name)
            provenances.append(dict(provenance))
            if p_value is not None:
                p_rows.append(self._actionness_row_tensor(p_value, descriptors, batch_idx, name="external p_action"))
            if logits_value is not None:
                logit_rows.append(
                    self._actionness_row_tensor(logits_value, descriptors, batch_idx, name="external actionness logits")
                )
        if len(set(source_names)) != 1:
            raise ValueError(f"external actionness source must be identical within a batch, got {source_names}")
        first_provenance = provenances[0]
        for provenance in provenances[1:]:
            for key in ("thumos_trained", "uses_labels", "uses_teacher", "uses_gt", "uses_prediction_cache"):
                if provenance.get(key) != first_provenance.get(key):
                    raise ValueError(f"external actionness provenance field {key} differs within batch")

        output: dict[str, Any] = {
            "source_name": source_names[0],
            "provenance": dict(first_provenance),
        }
        if p_rows:
            output["p_action"] = torch.stack(p_rows, dim=0)
        if logit_rows:
            output["actionness_logits"] = torch.stack(logit_rows, dim=0)
        return output

    def _reject_external_actionness_payload(self, metas) -> None:
        if not self.forbid_external_actionness or metas is None:
            return
        hits: list[str] = []

        def walk(value: Any, path: str) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    key_str = str(key)
                    child_path = f"{path}.{key_str}"
                    if (
                        key_str in _EXTERNAL_ACTIONNESS_PAYLOAD_KEYS
                        or key_str.startswith("duca_external_actionness")
                        or key_str.startswith("external_actionness")
                    ):
                        hits.append(child_path)
                    walk(child, child_path)
            elif isinstance(value, (list, tuple)):
                for idx, child in enumerate(value):
                    walk(child, f"{path}[{idx}]")

        walk(metas, "metas")
        if hits:
            raise ValueError(f"DUCA main selector forbids external actionness payloads: {hits}")

    @staticmethod
    def _lookup_meta_value(meta: Mapping[str, Any], key: Optional[str]) -> Any:
        if not key:
            return None
        if key in meta:
            return meta[key]
        current: Any = meta
        for part in str(key).split("."):
            if not isinstance(current, Mapping) or part not in current:
                return None
            current = current[part]
        return current

    @staticmethod
    def _actionness_row_tensor(value: Any, reference: torch.Tensor, batch_idx: int, *, name: str) -> torch.Tensor:
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)
        tensor = tensor.to(device=reference.device, dtype=torch.float32)
        if tensor.ndim != 1:
            raise ValueError(f"{name} for metas[{batch_idx}] must be a 1-D sequence")
        if tensor.numel() != reference.shape[1]:
            raise ValueError(
                f"{name} for metas[{batch_idx}] length {tensor.numel()} must match dense window {reference.shape[1]}"
            )
        return tensor

    @staticmethod
    def _reject_train_decision_payload(metas) -> None:
        if metas is None:
            return
        holders = [metas] if isinstance(metas, Mapping) else metas
        for idx, meta in enumerate(holders):
            if not isinstance(meta, Mapping):
                raise ValueError(f"metas[{idx}] must be a mapping")
            if meta.get("selection_uses_gt") is True:
                raise ValueError("DUCA train selection forbids GT-driven selection")
            if meta.get("selection_uses_teacher") is True:
                raise ValueError("DUCA train selection forbids teacher-driven selection")
            _assert_no_forbidden_payload(
                {"meta": meta},
                forbidden_keys={
                    "teacher_points",
                    "dense_teacher",
                    "dense_teacher_payload",
                    "dense_teacher_points",
                    "oracle_boundary",
                    "prediction_cache",
                    "raw_prediction",
                    "raw_predictions",
                    "ledger",
                    "ledger_path",
                },
            )

    def _remap_train_targets_to_selected_axis(self, gt_segments, gt_labels, metas):
        if gt_segments is None:
            return gt_segments, gt_labels, metas
        if not self.remap_gt_to_selected_axis:
            return gt_segments, gt_labels, metas
        if len(gt_segments) != len(metas):
            raise ValueError("gt_segments length must match metas length")
        remapped_segments = []
        updated_metas = [dict(meta) for meta in metas]
        for idx, (segments, meta) in enumerate(zip(gt_segments, updated_metas)):
            if segments is None:
                remapped_segments.append(segments)
                continue
            positions = meta.get("selected_axis_to_true_time_dense_index")
            if not positions:
                raise ValueError("DUCA GT remap requires selected_axis_to_true_time_dense_index metadata")
            segments_tensor = segments if torch.is_tensor(segments) else torch.as_tensor(segments, dtype=torch.float32)
            true_map = TrueTimeMap(
                positions,
                dense_len=int(meta.get("truetime_dense_len", max(positions) + 1)),
                valid_len=int(meta.get("truetime_dense_valid_len", meta.get("truetime_dense_len", max(positions) + 1))),
            )
            remapped = true_map.remap_segments(
                segments_tensor,
                source_coordinate_space=TRUE_TIME_AXIS,
                target_coordinate_space=SELECTED_AXIS,
            ).to(device=segments_tensor.device, dtype=segments_tensor.dtype)
            remapped_segments.append(remapped)
            meta["gt_segments_original_time"] = segments_tensor.detach().cpu().tolist()
            meta["gt_segments_selected_axis"] = remapped.detach().cpu().tolist()
            meta["gt_remapped_to_selected_axis"] = True
            meta["gt_coordinate_space"] = SELECTED_AXIS
            meta["gt_original_coordinate_space"] = TRUE_TIME_AXIS
            updated_metas[idx] = meta
        return remapped_segments, gt_labels, updated_metas

    def _write_metas(
        self,
        metas,
        grid,
        *,
        detector_grid_positions: Optional[torch.Tensor] = None,
        actionness_source_name: str,
        compute_profile: Optional[Mapping[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        batch = int(grid.selected_positions.shape[0])
        if metas is None:
            out = [{} for _ in range(batch)]
        else:
            if len(metas) != batch:
                raise ValueError("metas length must match batch size")
            out = [dict(meta) for meta in metas]
        positions_cpu = grid.selected_positions.detach().cpu().long()
        if detector_grid_positions is None:
            detector_grid_positions = grid.selected_positions
        if detector_grid_positions.shape != grid.selected_positions.shape:
            raise ValueError("detector_grid_positions must align with acquisition positions")
        detector_positions_cpu = detector_grid_positions.detach().cpu().long()
        valid_lens = grid.valid_len.detach().cpu().long()
        requested_budget = grid.requested_budget.detach().cpu().long()
        effective_budget = grid.effective_budget.detach().cpu().long()
        for idx, meta in enumerate(out):
            positions = [int(item) for item in positions_cpu[idx].tolist() if int(item) >= 0]
            detector_positions = [
                int(item) for item in detector_positions_cpu[idx].tolist() if int(item) >= 0
            ]
            if len(detector_positions) != len(positions):
                raise ValueError("detector-grid and acquisition positions must have the same active K")
            dense_valid_len = int(valid_lens[idx].item())
            remap = {
                "source": SELECTED_AXIS,
                "target": TRUE_TIME_AXIS,
                "selected_to_original": {int(axis): int(pos) for axis, pos in enumerate(detector_positions)},
                "original_to_selected": {int(pos): int(axis) for axis, pos in enumerate(detector_positions)},
                "selected_axis_to_true_time_dense_index": detector_positions,
                "acquisition_positions": positions,
            }
            meta["duca_online_selected_positions"] = positions
            meta["duca_acquisition_positions"] = positions
            meta["duca_detector_grid_positions"] = detector_positions
            meta["duca_online_selected_positions_unit"] = self.selected_positions_unit
            meta["duca_online_selected_mask"] = [True] * len(positions)
            meta["duca_online_budget"] = int(grid.budget)
            meta["duca_online_requested_budget"] = int(requested_budget[idx].item())
            meta["duca_online_effective_budget"] = int(effective_budget[idx].item())
            meta["duca_online_dynamic_budget"] = bool(grid.metadata.get("budget_is_dynamic", False))
            meta["duca_online_budget_policy"] = str(grid.metadata.get("budget_policy", "fixed_budget"))
            meta["duca_online_budget_target"] = float(grid.metadata.get("budget_target", float(grid.budget)))
            meta["duca_online_budget_multiple"] = int(grid.metadata.get("budget_multiple", 1))
            meta["duca_online_selected_count"] = len(positions)
            meta["duca_online_selected_axis_remap"] = remap
            meta["duca_online_actionness_source"] = actionness_source_name
            if compute_profile is not None:
                meta["duca_online_compute_profile"] = dict(compute_profile)
            meta["duca_online_budget_unit"] = grid.budget_unit
            meta["duca_online_coordinate"] = grid.coordinate
            if self.temporal_sampling_contract is not None:
                fps = meta.get("avg_fps", meta.get("fps"))
                meta["duca_temporal_sampling_contract"] = self.temporal_sampling_contract.to_dict(
                    fps=None if fps is None else float(fps)
                )
            meta["detector_output_coordinate_space"] = self.detector_output_coordinate_space
            meta["detector_prediction_inverse_map_required"] = self.detector_output_coordinate_space == SELECTED_AXIS
            meta["selected_axis_to_true_time_dense_index"] = detector_positions
            meta["truetime_selected_positions"] = detector_positions
            meta["truetime_dense_len"] = int(grid.original_length)
            meta["truetime_dense_valid_len"] = dense_valid_len
            meta["irregular_selected_positions"] = positions
            meta["irregular_native_axis"] = True
            meta["irregular_selected_count"] = len(positions)
            meta["irregular_dense_valid_len"] = dense_valid_len
            meta["irregular_selected_valid_len"] = len(positions)
            meta[self.metadata_keys["selected_positions"]] = positions
            meta[self.metadata_keys["selected_positions_unit"]] = self.selected_positions_unit
            meta[self.metadata_keys["selected_mask"]] = [True] * len(positions)
            meta[self.metadata_keys["selected_count"]] = len(positions)
            meta[self.metadata_keys["remap"]] = remap
            meta[self.metadata_keys["source"]] = actionness_source_name
        return out
