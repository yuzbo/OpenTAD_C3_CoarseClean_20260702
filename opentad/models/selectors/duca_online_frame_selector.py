from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import torch
import torch.nn as nn

from ..builder import SELECTORS
from ..duca import C3CoarseProbeActionnessSource, DucaAcquisitionAdapter, ZeroShotActionnessSource, duca_losses
from ..duca.acquisition import (
    _assert_no_forbidden_payload,
    _elapsed_ms,
    _sync_profile_clock,
    validate_actionness_provenance,
)
from ..utils.truetime_geometry import SELECTED_AXIS, TRUE_TIME_AXIS, TrueTimeMap


_DEFAULT_METADATA_KEYS = {
    "selected_positions": "duca_online_selected_positions",
    "selected_positions_unit": "duca_online_selected_positions_unit",
    "selected_mask": "duca_online_selected_mask",
    "selected_count": "duca_online_selected_count",
    "remap": "duca_online_selected_axis_remap",
    "source": "duca_online_actionness_source",
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


@SELECTORS.register_module()
class DucaOnlineFrameSelector(nn.Module):
    """Registry-buildable online DUCA selector for OpenTAD frame_selector hooks."""

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
        dense_window_size: Optional[int] = None,
        selector_hidden_channels: int = 0,
        actionness_source_cfg: Optional[Mapping[str, Any]] = None,
        detector_gradient_mode: str = "st_sparse_gather",
        coordinate_space: str = SELECTED_AXIS,
        detector_output_coordinate_space: str = SELECTED_AXIS,
        selected_positions_unit: str = "original_time_index",
        true_time_source_axis: str = TRUE_TIME_AXIS,
        loss_weights: Optional[Mapping[str, float]] = None,
        loss_weight_schedule: Optional[Mapping[str, Any]] = None,
        no_ledger_decision: bool = True,
        remap_gt_to_selected_axis: bool = True,
        selected_axis_remap_required: bool = True,
        forbid_ledger: bool = True,
        forbid_raw_prediction_cache: bool = True,
        external_actionness_meta_key: Optional[str] = None,
        external_actionness_logits_meta_key: Optional[str] = None,
        external_actionness_provenance_meta_key: Optional[str] = None,
        external_actionness_source_meta_key: Optional[str] = None,
        require_external_actionness: bool = False,
        profile_runtime: bool = False,
        profile_sync_cuda: bool = True,
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
        self.dense_window_size = None if dense_window_size is None else int(dense_window_size)
        self.detector_gradient_mode = str(detector_gradient_mode)
        self.selected_positions_coordinate = str(coordinate_space)
        self.detector_output_coordinate_space = str(detector_output_coordinate_space)
        self.coordinate_space = self.detector_output_coordinate_space
        self.selected_positions_unit = str(selected_positions_unit)
        self.true_time_source_axis = str(true_time_source_axis)
        self.loss_weights = dict(loss_weights or {})
        self.loss_weight_schedule = self._normalize_loss_weight_schedule(loss_weight_schedule)
        self.register_buffer("_loss_weight_schedule_step", torch.zeros((), dtype=torch.long), persistent=True)
        self.no_ledger_decision = bool(no_ledger_decision)
        self.remap_gt_to_selected_axis = bool(remap_gt_to_selected_axis)
        self.selected_axis_remap_required = bool(selected_axis_remap_required)
        self.forbid_ledger = bool(forbid_ledger)
        self.forbid_raw_prediction_cache = bool(forbid_raw_prediction_cache)
        self.external_actionness_meta_key = external_actionness_meta_key
        self.external_actionness_logits_meta_key = external_actionness_logits_meta_key
        self.external_actionness_provenance_meta_key = external_actionness_provenance_meta_key
        self.external_actionness_source_meta_key = external_actionness_source_meta_key
        self.require_external_actionness = bool(require_external_actionness)
        self.profile_runtime = bool(profile_runtime)
        self.profile_sync_cuda = bool(profile_sync_cuda)
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
        if self.detector_gradient_mode not in {"st_sparse_gather", "st_sparse_gather_soft_context"}:
            raise ValueError("detector_gradient_mode must be st_sparse_gather or st_sparse_gather_soft_context")
        if self.selected_positions_coordinate not in {"original_time", SELECTED_AXIS, TRUE_TIME_AXIS}:
            raise ValueError("coordinate_space must describe original-time selected positions or selected-axis detector output")
        if self.detector_output_coordinate_space not in {SELECTED_AXIS, TRUE_TIME_AXIS}:
            raise ValueError("detector_output_coordinate_space must be selected-axis or true-time")
        if self.selected_positions_unit != "original_time_index":
            raise ValueError("selected_positions_unit must be original_time_index")
        if self.true_time_source_axis != TRUE_TIME_AXIS:
            raise ValueError("true_time_source_axis must be true_time_dense_index")
        if self.dense_window_size is not None and self.dense_window_size <= 0:
            raise ValueError("dense_window_size must be positive")
        if not self.no_ledger_decision:
            raise ValueError("DUCA online selector requires no_ledger_decision=True")
        if self.detector_output_coordinate_space == SELECTED_AXIS and not self.remap_gt_to_selected_axis:
            raise ValueError("selected-axis detector output requires remap_gt_to_selected_axis=True")

        actionness_source = None
        self.raw_actionness_source = None
        self.actionness_source_name = "duca_adapter_internal"
        if actionness_source_cfg:
            cfg = dict(actionness_source_cfg)
            source_type = cfg.pop("type", "ZeroShotActionnessSource")
            self.actionness_source_name = str(cfg.get("source_name") or source_type)
            if source_type == "C3CoarseProbeActionnessSource":
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
            hidden_dim=int(selector_hidden_channels),
            actionness_source=actionness_source,
            profile_runtime=self.profile_runtime,
            profile_sync_cuda=self.profile_sync_cuda,
        )

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
        self._reject_train_decision_payload(metas)
        action_target = self._action_target_from_gt_segments(gt_segments, masks)
        schedule_state = self._loss_schedule_state()
        outputs = self._forward_select(inputs, masks, metas, budget=budget, schedule_state=schedule_state)
        outputs["selector_outputs"]["loss_weight_schedule"] = schedule_state
        if action_target is not None:
            outputs["selector_outputs"]["action_target"] = action_target
        gt_segments, gt_labels, metas = self._remap_train_targets_to_selected_axis(
            gt_segments, gt_labels, outputs["metas"]
        )
        selector_losses = duca_losses(
            outputs["selector_outputs"],
            teacher_utility=teacher_utility,
            action_target=action_target,
            loss_weights=schedule_state["weights"],
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
        }

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
        entries: dict[str, tuple[float, float]] = {}
        reserved = {"type", "warmup_steps", "transition_steps", "ramp_steps", "shape", "curve", "enabled"}
        for key, value in out.items():
            if key in reserved:
                continue
            if isinstance(value, Mapping):
                if "start" not in value or "end" not in value:
                    raise ValueError(f"loss_weight_schedule.{key} must define start and end")
                entries[str(key)] = (float(value["start"]), float(value["end"]))
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
        for key, (start, end) in self.loss_weight_schedule["entries"].items():
            weights[key] = float(start + (end - start) * progress)
        if progress <= 0.0:
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
            "detector_gradient_weight": float(weights.get("detector_gradient", weights.get("detector", 1.0))),
        }

    def _loss_schedule_progress(self, step: int) -> float:
        if self.loss_weight_schedule is None:
            return 1.0
        warmup = int(self.loss_weight_schedule["warmup_steps"])
        transition = int(self.loss_weight_schedule["transition_steps"])
        if step <= warmup:
            raw = 0.0
        elif transition <= 0:
            raw = 1.0
        else:
            raw = min(1.0, max(0.0, float(step - warmup) / float(transition)))
        if self.loss_weight_schedule.get("shape") == "cosine":
            pi = torch.acos(torch.zeros((), dtype=torch.float64)).item() * 2.0
            raw = 0.5 - 0.5 * torch.cos(torch.tensor(raw * pi, dtype=torch.float64)).item()
        return float(raw)

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
        centers = torch.arange(temporal_len, device=device, dtype=dtype) + 0.5
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
            covered = ((centers[None, :] >= starts) & (centers[None, :] <= ends)).any(dim=0)
            target[batch_idx] = covered.to(dtype=dtype)
        return target.masked_fill(~valid, 0.0)

    def forward_test(self, inputs: torch.Tensor, masks: torch.Tensor, metas=None, budget=None, **kwargs: Any) -> dict[str, Any]:
        _assert_no_forbidden_payload({"metas": metas, "kwargs": kwargs})
        outputs = self._forward_select(inputs, masks, metas, budget=budget)
        return {
            "inputs": outputs["inputs"],
            "masks": outputs["masks"],
            "metas": outputs["metas"],
            "selector_outputs": outputs["selector_outputs"],
        }

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
        descriptors = _time_descriptors_btc(inputs)
        descriptor_ms = _elapsed_ms(descriptor_start, inputs, enabled=sync_enabled)
        descriptor_profile = self._descriptor_compute_profile(inputs, descriptors)
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
        if external_actionness is not None:
            actionness_logits = external_actionness.get("actionness_logits")
            p_action = external_actionness.get("p_action")
        elif online_actionness is not None:
            actionness_logits = online_actionness.get("logits")
            if actionness_logits is None:
                actionness_logits = online_actionness.get("actionness_logits")
        grid, scores = self.adapter.acquire(
            descriptors,
            budget=budget,
            valid_mask=masks,
            actionness_logits=actionness_logits,
            p_action=p_action,
            compute_profile_context=profile_context,
        )
        actionness_source_name = self.actionness_source_name
        if external_actionness is not None:
            scores["provenance"] = external_actionness["provenance"]
            scores["external_actionness_provenance"] = external_actionness["provenance"]
            scores["external_actionness_source"] = external_actionness["source_name"]
            actionness_source_name = external_actionness["source_name"]
        elif online_actionness is not None:
            scores["provenance"] = online_actionness["provenance"]
            scores["online_actionness_provenance"] = online_actionness["provenance"]
            scores["online_actionness_source"] = online_actionness["source_name"]
            actionness_source_name = online_actionness["source_name"]
        validate_actionness_provenance(scores.get("provenance", {}), context="DUCA selector actionness provenance")
        positions = grid.selected_positions.to(device=inputs.device)
        slot_mask = positions >= 0
        gather_start = _sync_profile_clock(inputs, enabled=sync_enabled) if profile_enabled else None
        hard_selected = _gather_time(inputs, positions, slot_mask)
        detector_gradient_weight = self._detector_gradient_weight(schedule_state)
        if self.detector_gradient_mode == "st_sparse_gather_soft_context":
            hard_selected = _add_soft_context_gradient_path(
                hard_selected,
                inputs,
                scores["soft_coverage"],
                slot_mask,
                bridge_weight=detector_gradient_weight,
            )
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
        scores["compute_profile"] = compute_profile
        selected_masks = slot_mask.to(device=inputs.device, dtype=torch.bool)
        scores["grid"] = grid
        scores["hard_selected_inputs"] = hard_selected
        scores["selected_input_st_gradient_path"] = self.detector_gradient_mode
        scores["detector_gradient_weight"] = float(detector_gradient_weight)
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
        }
        if isinstance(schedule_state, Mapping):
            self.last_forward_summary["loss_weight_schedule"] = dict(schedule_state)
        return {
            "inputs": selected_inputs,
            "masks": selected_masks,
            "metas": self._write_metas(
                metas,
                grid,
                actionness_source_name=actionness_source_name,
                compute_profile=compute_profile,
            ),
            "selector_outputs": scores,
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
        valid_lens = grid.valid_len.detach().cpu().long()
        requested_budget = grid.requested_budget.detach().cpu().long()
        effective_budget = grid.effective_budget.detach().cpu().long()
        for idx, meta in enumerate(out):
            positions = [int(item) for item in positions_cpu[idx].tolist() if int(item) >= 0]
            dense_valid_len = int(valid_lens[idx].item())
            remap = {
                "source": SELECTED_AXIS,
                "target": TRUE_TIME_AXIS,
                "selected_to_original": {int(axis): int(pos) for axis, pos in enumerate(positions)},
                "original_to_selected": {int(pos): int(axis) for axis, pos in enumerate(positions)},
                "selected_axis_to_true_time_dense_index": positions,
            }
            meta["duca_online_selected_positions"] = positions
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
            meta["detector_output_coordinate_space"] = self.detector_output_coordinate_space
            meta["detector_prediction_inverse_map_required"] = self.detector_output_coordinate_space == SELECTED_AXIS
            meta["selected_axis_to_true_time_dense_index"] = positions
            meta["truetime_selected_positions"] = positions
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
