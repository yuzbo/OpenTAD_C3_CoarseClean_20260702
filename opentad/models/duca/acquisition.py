from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from opentad.duca_loss_contract import (
    DUCA_LOSS_TO_WEIGHT_KEY,
    DUCA_LOSS_WEIGHT_DEFAULTS,
)

from .dynamic_budget import DynamicBudgetDecision, PrefixMarginalUtilityBudgetController
from .structured_selection import (
    budget_calibrated_sampling_rate,
    continuous_density_transport,
    exact_uniform_positions,
    exact_uniform_reference_scores,
    global_structured_topk,
    local_cell_deformation,
)
from .transition_only import (
    ASFORMER_ENCODER_HIDDEN_KIND,
    DucaMixtureDensityHead,
    DucaTransitionUtilityScorer,
    balanced_binary_actionness_loss,
    boundary_burst_coverage_loss,
    build_boundary_burst_utility,
    build_mandatory_bilateral_set,
    continuous_policy_logits,
    local_boundary_coverage_loss,
    local_boundary_mass_coverage_loss,
    transition_distribution_loss,
    transition_utility_paths,
)


TensorLikeBudget = Union[int, torch.Tensor]

DEFAULT_BUDGET_UNIT = "detector_consumed_temporal_observation"
DEFAULT_COORDINATE = "original_time"
DUCA_ACTIONNESS_FEATURE_NAMES = (
    "p_action",
    "uncertainty",
    "entropy",
    "delta_p_action",
    "abs_delta_p_action",
    "uncertainty_peak",
    "transition_score",
)
DUCA_ACTIONNESS_FEATURE_DIM = len(DUCA_ACTIONNESS_FEATURE_NAMES)
FORBIDDEN_DECISION_KEYS = {
    "teacher_utility",
    "teacher_points",
    "dense_teacher",
    "dense_teacher_payload",
    "dense_teacher_points",
    "gt_segments",
    "gt_labels",
    "oracle_boundary",
    "prediction_cache",
    "raw_prediction",
    "raw_predictions",
    "ledger",
    "ledger_path",
}

_PROVENANCE_REQUIRED_KEYS = (
    "thumos_trained",
    "uses_labels",
    "uses_teacher",
    "uses_gt",
    "uses_prediction_cache",
)

_PROVENANCE_ALWAYS_FALSE_KEYS = (
    "uses_teacher",
    "uses_prediction_cache",
)


def _sync_profile_clock(reference: torch.Tensor, *, enabled: bool) -> float:
    if enabled and torch.is_tensor(reference) and reference.is_cuda:
        torch.cuda.synchronize(reference.device)
    return time.perf_counter()


def _elapsed_ms(start: Optional[float], reference: torch.Tensor, *, enabled: bool) -> Optional[float]:
    if start is None:
        return None
    end = _sync_profile_clock(reference, enabled=enabled)
    return float((end - start) * 1000.0)


def _module_param_counts(module: Optional[nn.Module]) -> Dict[str, int]:
    if module is None:
        return {"total": 0, "trainable": 0}
    total = 0
    trainable = 0
    for param in module.parameters():
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
    return {"total": total, "trainable": trainable}


def _known_number(value: Optional[Union[int, float]]) -> int:
    if value is None:
        return 0
    return int(value)


def _sum_known(values: Iterable[Optional[Union[int, float]]]) -> int:
    return int(sum(_known_number(value) for value in values))


def validate_actionness_provenance(
    provenance: Mapping[str, object],
    *,
    context: str = "actionness provenance",
) -> None:
    """Fail closed unless an actionness source is explicitly deployable."""
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{context} must be a mapping")
    missing = [key for key in _PROVENANCE_REQUIRED_KEYS if key not in provenance]
    if missing:
        raise ValueError(f"{context} missing explicit fields: {', '.join(missing)}")
    unsafe = [key for key in _PROVENANCE_ALWAYS_FALSE_KEYS if provenance.get(key) is not False]
    supervised_fields = {
        "thumos_trained": ("trained_with_thumos_labels", "uses_labels_at_inference"),
        "uses_labels": ("trained_with_thumos_labels", "uses_labels_at_inference"),
        "uses_gt": ("trained_with_gt_segments", "uses_gt_at_inference"),
    }
    for key, (disclosure_key, inference_key) in supervised_fields.items():
        value = provenance.get(key)
        if value is False:
            continue
        disclosed_train_only = (
            value is True
            and provenance.get(disclosure_key) is True
            and provenance.get(inference_key) is False
            and provenance.get("training_supervision_scope") == "train_only"
        )
        if not disclosed_train_only:
            unsafe.append(key)
    if unsafe:
        raise ValueError(
            f"{context} is not deployable/no-target-label clean: "
            + ", ".join(f"{key}={provenance.get(key)!r}" for key in unsafe)
        )
    calibration_split = provenance.get("calibration_split")
    if calibration_split not in (None, "", "none", "train_only"):
        raise ValueError(
            f"{context} has target calibration split {calibration_split!r}; "
            "deployable DUCA actionness must not be calibrated on val/test labels"
        )


def _neg(dtype: torch.dtype) -> float:
    return float(torch.finfo(dtype).min / 4.0)


def _as_valid_mask(reference: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if reference.ndim != 2:
        raise ValueError(f"reference must be [B,T], got {tuple(reference.shape)}")
    if valid_mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    if valid_mask.shape != reference.shape:
        raise ValueError(f"valid_mask must match [B,T] {tuple(reference.shape)}, got {tuple(valid_mask.shape)}")
    if valid_mask.dtype != torch.bool:
        if not torch.logical_or(valid_mask == 0, valid_mask == 1).all():
            raise ValueError("valid_mask must be boolean or binary")
    valid = valid_mask.bool()
    if torch.any(valid.long().sum(dim=1) <= 0):
        raise ValueError("each sample must contain at least one valid temporal observation")
    return valid


def _actionness_transition_payload(
    p_action: torch.Tensor,
    valid: torch.Tensor,
    uncertainty_override: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    prob = p_action.float().masked_fill(~valid, 0.0)
    if uncertainty_override is None:
        uncertainty = 1.0 - torch.abs(2.0 * prob - 1.0)
    else:
        uncertainty = uncertainty_override.to(device=prob.device, dtype=prob.dtype)
        if uncertainty.shape != prob.shape:
            raise ValueError("uncertainty_override must match p_action shape")
    uncertainty = uncertainty.masked_fill(~valid, 0.0)
    entropy = _binary_entropy(prob).masked_fill(~valid, 0.0)
    delta = torch.zeros_like(prob)
    if prob.shape[1] > 1:
        delta[:, 1:] = prob[:, 1:] - prob[:, :-1]
        prev_valid = torch.zeros_like(valid)
        prev_valid[:, 1:] = valid[:, 1:] & valid[:, :-1]
        delta = delta.masked_fill(~prev_valid, 0.0)
    abs_delta = delta.abs().masked_fill(~valid, 0.0)
    if prob.shape[1] > 1:
        left = torch.zeros_like(uncertainty)
        right = torch.zeros_like(uncertainty)
        left[:, 1:] = uncertainty[:, :-1]
        right[:, :-1] = uncertainty[:, 1:]
        neighbor = torch.maximum(left, right)
        uncertainty_peak = F.relu(uncertainty - neighbor)
    else:
        uncertainty_peak = uncertainty
    uncertainty_peak = uncertainty_peak.masked_fill(~valid, 0.0)
    transition_score = (abs_delta + uncertainty_peak).masked_fill(~valid, 0.0)
    features = torch.stack(
        (
            prob,
            uncertainty,
            entropy,
            delta,
            abs_delta,
            uncertainty_peak,
            transition_score,
        ),
        dim=-1,
    )
    return {
        "p_action": prob,
        "uncertainty": uncertainty,
        "entropy": entropy,
        "delta_p_action": delta,
        "abs_delta_p_action": abs_delta,
        "uncertainty_peak": uncertainty_peak,
        "transition_score": transition_score,
        "features": features,
    }


def _budget_tensor(budget: TensorLikeBudget, batch_size: int, device: torch.device) -> torch.Tensor:
    if isinstance(budget, int):
        out = torch.full((batch_size,), int(budget), dtype=torch.long, device=device)
    elif torch.is_tensor(budget):
        if budget.ndim == 0:
            out = torch.full((batch_size,), int(budget.item()), dtype=torch.long, device=device)
        elif budget.ndim == 1 and budget.numel() == batch_size:
            out = budget.to(device=device, dtype=torch.long)
        else:
            raise ValueError(f"budget tensor must be scalar or [B], got {tuple(budget.shape)}")
    else:
        raise TypeError("budget must be int or tensor")
    if torch.any(out <= 0):
        raise ValueError("budget must be positive")
    return out


def _per_sample_value(value: Union[int, torch.Tensor], row: int) -> int:
    if isinstance(value, int):
        return int(value)
    if value.ndim == 0:
        return int(value.item())
    return int(value[row].item())


@dataclass
class SparseTemporalGrid:
    """Fail-closed detector-consumed original-time sparse temporal grid.

    `selected_positions` are the only temporal observations the detector is
    allowed to consume. Center and radius decisions can be stored in metadata,
    but they are not the budget unit.
    """

    selected_positions: torch.Tensor
    selected_mask: torch.Tensor
    original_length: int
    valid_len: Optional[torch.Tensor]
    budget: int
    requested_budget: Optional[torch.Tensor] = None
    effective_budget: Optional[torch.Tensor] = None
    budget_unit: str = DEFAULT_BUDGET_UNIT
    coordinate: str = DEFAULT_COORDINATE
    detector_consumes_selected_positions: bool = True
    detector_input_length: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dense_length(self) -> int:
        return int(self.original_length)

    @property
    def selected_count(self) -> torch.Tensor:
        return self.selected_mask.bool().long().sum(dim=1)

    def validate(self) -> "SparseTemporalGrid":
        if not torch.is_tensor(self.selected_positions) or self.selected_positions.ndim != 2:
            raise ValueError("selected_positions must be a [B,K] tensor")
        if not torch.is_tensor(self.selected_mask) or self.selected_mask.ndim != 2:
            raise ValueError("selected_mask must be a dense [B,T] tensor")
        if self.selected_positions.shape[0] != self.selected_mask.shape[0]:
            raise ValueError("selected_positions and selected_mask batch dimensions must match")
        if int(self.original_length) <= 0:
            raise ValueError("original_length must be positive")
        if self.selected_mask.shape[1] != int(self.original_length):
            raise ValueError("selected_mask temporal dimension must equal original_length")
        if self.valid_len is None:
            raise ValueError("valid_len is required for fail-closed validation")
        if self.valid_len.ndim == 0:
            valid_len = self.valid_len.expand(self.selected_positions.shape[0]).to(
                device=self.selected_positions.device, dtype=torch.long
            )
        elif self.valid_len.ndim == 1 and self.valid_len.numel() == self.selected_positions.shape[0]:
            valid_len = self.valid_len.to(device=self.selected_positions.device, dtype=torch.long)
        else:
            raise ValueError("valid_len must be scalar or [B]")
        if torch.any(valid_len <= 0) or torch.any(valid_len > int(self.original_length)):
            raise ValueError("valid_len must lie in (0, original_length]")
        if int(self.budget) <= 0:
            raise ValueError("budget must be positive")
        if self.requested_budget is None:
            raise ValueError("requested_budget is required for fail-closed validation")
        if self.effective_budget is None:
            raise ValueError("effective_budget is required for fail-closed validation")
        requested_budget = self.requested_budget.to(device=self.selected_positions.device, dtype=torch.long)
        effective_budget = self.effective_budget.to(device=self.selected_positions.device, dtype=torch.long)
        if requested_budget.ndim == 0:
            requested_budget = requested_budget.expand(self.selected_positions.shape[0])
        if effective_budget.ndim == 0:
            effective_budget = effective_budget.expand(self.selected_positions.shape[0])
        if requested_budget.shape != valid_len.shape or effective_budget.shape != valid_len.shape:
            raise ValueError("requested_budget/effective_budget must be scalar or [B]")
        if torch.any(requested_budget <= 0) or torch.any(effective_budget <= 0):
            raise ValueError("requested_budget/effective_budget must be positive")
        if torch.any(requested_budget > int(self.budget)):
            raise ValueError("requested_budget cannot exceed max budget")
        if torch.any(effective_budget > requested_budget):
            raise ValueError("effective_budget cannot exceed requested_budget")
        if torch.any(effective_budget > valid_len):
            raise ValueError("effective_budget cannot exceed valid_len")
        if self.budget_unit != DEFAULT_BUDGET_UNIT:
            raise ValueError(f"budget_unit must be {DEFAULT_BUDGET_UNIT}")
        if self.coordinate != DEFAULT_COORDINATE:
            raise ValueError(f"coordinate must be {DEFAULT_COORDINATE}")
        if not bool(self.detector_consumes_selected_positions):
            raise ValueError("detector_consumes_selected_positions must be True")

        selected_mask = self.selected_mask.bool()
        count_from_mask = selected_mask.long().sum(dim=1)
        if torch.any(count_from_mask > int(self.budget)):
            raise ValueError("selected count exceeds detector-consumed budget")
        if torch.any(count_from_mask > effective_budget):
            raise ValueError("selected count exceeds effective_budget")

        pos = self.selected_positions.to(device=selected_mask.device, dtype=torch.long)
        for batch_idx in range(pos.shape[0]):
            row_pos = pos[batch_idx]
            valid_pos = row_pos[row_pos >= 0]
            if valid_pos.numel() != int(count_from_mask[batch_idx].item()):
                raise ValueError("selected_positions count must equal selected_mask count")
            if valid_pos.numel() == 0:
                raise ValueError("each sample must select at least one observation")
            if torch.any(valid_pos >= int(self.original_length)):
                raise ValueError("selected_positions contain out-of-range original-time indices")
            if torch.any(valid_pos >= int(valid_len[batch_idx].item())):
                raise ValueError("selected_positions exceed per-sample valid_len")
            if valid_pos.numel() > 1:
                if not torch.all(valid_pos[1:] > valid_pos[:-1]):
                    raise ValueError("selected_positions must be sorted and unique")
            mask_pos = torch.nonzero(selected_mask[batch_idx], as_tuple=False).flatten()
            if not torch.equal(valid_pos.cpu(), mask_pos.cpu()):
                raise ValueError("selected_mask must exactly match selected_positions")

        if self.detector_input_length is not None:
            if self.detector_input_length.shape != count_from_mask.shape:
                raise ValueError("detector_input_length must be [B]")
            if not torch.equal(self.detector_input_length.to(count_from_mask.device, dtype=torch.long), count_from_mask):
                raise ValueError("detector_input_length must equal selected count")
        return self


class ZeroShotActionnessSource(nn.Module):
    """Deploy-visible no-THUMOS-label actionness abstraction.

    The default fallback is lightweight motion / feature-energy actionness. A
    CLIP-like video-text source can be injected through `video_text_model` and
    `tokenizer`; unit tests do not download or require those models.
    """

    def __init__(
        self,
        feature_dim: Optional[int] = None,
        hidden_dim: int = 64,
        frozen: bool = True,
        mode: str = "motion",
        p_action: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        video_text_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        action_prompts: Optional[Iterable[str]] = None,
        background_prompts: Optional[Iterable[str]] = None,
        temperature: float = 1.0,
        provenance: Optional[Mapping[str, Any]] = None,
        source_name: Optional[str] = None,
        checkpoint_hash: Optional[str] = None,
        thumos_trained: Optional[bool] = None,
        uses_labels: Optional[bool] = None,
        uses_teacher: Optional[bool] = None,
        uses_gt: Optional[bool] = None,
        uses_prediction_cache: Optional[bool] = None,
        calibration_split: Optional[str] = None,
        prompt_hash: Optional[str] = None,
    ) -> None:
        super().__init__()
        if mode not in {"motion", "feature_mlp", "manual", "video_text"}:
            raise ValueError("mode must be one of motion, feature_mlp, manual, video_text")
        self.mode = mode
        self.temperature = float(temperature)
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.video_text_model = video_text_model
        self.tokenizer = tokenizer
        self.action_prompts = tuple(
            action_prompts
            or (
                "a video clip of a person performing an action",
                "a video clip showing a human activity",
                "a video clip with a salient temporal action",
                "a video clip where something is happening",
            )
        )
        self.background_prompts = tuple(
            background_prompts
            or (
                "a video clip of background",
                "a video clip where no action is happening",
                "a video clip without a salient action",
                "a video clip of irrelevant context",
            )
        )
        default_no_thumos = mode in {"motion", "video_text"}
        explicit = dict(provenance or {})
        self._provenance_override = {
            "source_name": source_name or explicit.get("source_name") or mode,
            "checkpoint_hash": checkpoint_hash if checkpoint_hash is not None else explicit.get("checkpoint_hash"),
            "thumos_trained": (
                thumos_trained
                if thumos_trained is not None
                else explicit.get("thumos_trained", False if default_no_thumos else None)
            ),
            "uses_labels": (
                uses_labels if uses_labels is not None else explicit.get("uses_labels", False if default_no_thumos else None)
            ),
            "uses_teacher": (
                uses_teacher if uses_teacher is not None else explicit.get("uses_teacher", False if default_no_thumos else None)
            ),
            "uses_gt": uses_gt if uses_gt is not None else explicit.get("uses_gt", False if default_no_thumos else None),
            "uses_prediction_cache": (
                uses_prediction_cache
                if uses_prediction_cache is not None
                else explicit.get("uses_prediction_cache", False if default_no_thumos else None)
            ),
            "calibration_split": calibration_split if calibration_split is not None else explicit.get("calibration_split"),
            "prompt_hash": prompt_hash if prompt_hash is not None else explicit.get("prompt_hash"),
        }

        if mode == "manual":
            if p_action is None or uncertainty is None:
                raise ValueError("manual mode requires p_action and uncertainty")
            if p_action.shape != uncertainty.shape:
                raise ValueError("manual p_action and uncertainty must share shape")
            self.register_buffer("manual_p_action", p_action.float())
            self.register_buffer("manual_uncertainty", uncertainty.float())
        else:
            self.manual_p_action = None
            self.manual_uncertainty = None

        self.feature_dim = None if feature_dim is None else int(feature_dim)
        if mode == "feature_mlp":
            if self.feature_dim is None:
                raise ValueError("feature_mlp mode requires feature_dim")
            self.net = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), 1),
            )
        else:
            self.net = None

        if frozen:
            for param in self.parameters():
                param.requires_grad_(False)

    @classmethod
    def from_manual(
        cls,
        p_action: torch.Tensor,
        uncertainty: torch.Tensor,
        provenance: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "ZeroShotActionnessSource":
        return cls(mode="manual", p_action=p_action, uncertainty=uncertainty, provenance=provenance, **kwargs)

    def _provenance(self) -> Dict[str, Any]:
        return {
            "source_type": self.mode,
            "source_name": self._provenance_override["source_name"],
            "checkpoint_hash": self._provenance_override["checkpoint_hash"],
            "thumos_trained": self._provenance_override["thumos_trained"],
            "uses_labels": self._provenance_override["uses_labels"],
            "uses_teacher": self._provenance_override["uses_teacher"],
            "calibration_split": self._provenance_override["calibration_split"],
            "prompt_hash": self._provenance_override["prompt_hash"],
            "uses_gt": self._provenance_override["uses_gt"],
            "uses_prediction_cache": self._provenance_override["uses_prediction_cache"],
            "temperature": self.temperature,
            "action_prompts": list(self.action_prompts),
            "background_prompts": list(self.background_prompts),
        }

    def forward(
        self,
        features: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        p_action: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if p_action is not None:
            if p_action.ndim != 2:
                raise ValueError("p_action must be [B,T]")
            valid = _as_valid_mask(p_action, valid_mask)
            prob = p_action.float().clamp(0.0, 1.0).masked_fill(~valid, 0.0)
            logits_out = torch.logit(prob.clamp(1e-6, 1.0 - 1e-6)).masked_fill(~valid, _neg(prob.dtype))
            return self._format_output(prob, logits_out, valid)

        if self.mode == "manual":
            prob = self.manual_p_action
            valid = _as_valid_mask(prob, valid_mask)
            prob = prob.masked_fill(~valid, 0.0)
            logits_out = torch.logit(prob.clamp(1e-6, 1.0 - 1e-6)).masked_fill(~valid, _neg(prob.dtype))
            manual_uncertainty = self.manual_uncertainty.to(prob.device, prob.dtype)
            return self._format_output(prob, logits_out, valid, uncertainty=manual_uncertainty)

        if logits is not None:
            if logits.ndim == 3 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            if logits.ndim != 2:
                raise ValueError("logits must be [B,T] or [B,T,1]")
            valid = _as_valid_mask(logits, valid_mask)
            logits_out = logits.float().masked_fill(~valid, _neg(logits.float().dtype))
            prob = torch.sigmoid(logits_out / self.temperature).masked_fill(~valid, 0.0)
            return self._format_output(prob, logits_out, valid)

        if features is None:
            raise ValueError("features, logits, or p_action must be provided")
        if features.ndim != 3:
            raise ValueError(f"features must be [B,T,C], got {tuple(features.shape)}")

        ref = features[..., 0]
        valid = _as_valid_mask(ref, valid_mask)
        if self.mode == "feature_mlp":
            if features.shape[-1] != self.feature_dim:
                raise ValueError(f"expected feature_dim={self.feature_dim}, got {features.shape[-1]}")
            logits_out = self.net(features).squeeze(-1).masked_fill(~valid, _neg(features.dtype))
            prob = torch.sigmoid(logits_out / self.temperature).masked_fill(~valid, 0.0)
            return self._format_output(prob, logits_out, valid)

        if self.mode == "video_text":
            if self.video_text_model is None or self.tokenizer is None:
                raise ValueError("video_text mode requires injected video_text_model and tokenizer")
            logits_out = self._video_text_logits(features).masked_fill(~valid, _neg(features.dtype))
            prob = torch.sigmoid(logits_out / self.temperature).masked_fill(~valid, 0.0)
            return self._format_output(prob, logits_out, valid)

        # Lightweight fallback: feature energy plus temporal change, normalized
        # within each video without any dataset labels or calibration.
        energy = features.float().pow(2).mean(dim=-1).sqrt()
        delta = torch.zeros_like(energy)
        delta[:, 1:] = (features[:, 1:].float() - features[:, :-1].float()).pow(2).mean(dim=-1).sqrt()
        raw = energy + delta
        raw = raw.masked_fill(~valid, 0.0)
        denom = valid.long().sum(dim=1).clamp_min(1).to(raw.dtype)
        mean = raw.sum(dim=1, keepdim=True) / denom[:, None]
        centered = raw - mean
        scale = centered.masked_fill(~valid, 0.0).abs().sum(dim=1, keepdim=True) / denom[:, None]
        logits_out = (centered / scale.clamp_min(1e-6)).masked_fill(~valid, _neg(raw.dtype))
        prob = torch.sigmoid(logits_out).masked_fill(~valid, 0.0)
        return self._format_output(prob, logits_out, valid)

    def _video_text_logits(self, features: torch.Tensor) -> torch.Tensor:
        flat = features.reshape(features.shape[0] * features.shape[1], features.shape[2])
        video_feat = self.video_text_model.encode_video(flat)
        action_tokens = self.tokenizer(list(self.action_prompts)).to(features.device)
        background_tokens = self.tokenizer(list(self.background_prompts)).to(features.device)
        action_text = self.video_text_model.encode_text(action_tokens)
        background_text = self.video_text_model.encode_text(background_tokens)
        video_feat = F.normalize(video_feat.float(), dim=-1)
        action_text = F.normalize(action_text.float(), dim=-1)
        background_text = F.normalize(background_text.float(), dim=-1)
        sim_action = video_feat @ action_text.t()
        sim_background = video_feat @ background_text.t()
        logits = sim_action.mean(dim=-1) - sim_background.mean(dim=-1)
        return logits.reshape(features.shape[0], features.shape[1])

    def _format_output(
        self,
        p_action: torch.Tensor,
        logits: torch.Tensor,
        valid: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        transition = _actionness_transition_payload(p_action, valid, uncertainty_override=uncertainty)
        return {
            "p_action": transition["p_action"],
            "uncertainty": transition["uncertainty"],
            "entropy": transition["entropy"],
            "delta_p_action": transition["delta_p_action"],
            "abs_delta_p_action": transition["abs_delta_p_action"],
            "uncertainty_peak": transition["uncertainty_peak"],
            "transition_score": transition["transition_score"],
            "features": transition["features"],
            "logits": logits,
            "actionness_logits": logits,
            "valid_mask": valid,
            "provenance": self._provenance(),
        }


class C3CoarseProbeActionnessSource(nn.Module):
    """Task-adapted coarse frame classifier source for DUCA pre-backbone selection.

    This wraps the validated low-resolution MobileNet/TCN/ASFormer-style C3
    probes and exposes the same p_action/logit contract as DUCA's actionness
    source. It is intended to run before the detector backbone and can be
    jointly optimized with the selector and detector when configured trainable.
    """

    def __init__(
        self,
        probe_model: str = "temporal-tcn",
        spatial_size: int = 64,
        checkpoint_path: Optional[str] = None,
        require_checkpoint: bool = False,
        frozen: bool = True,
        trainable: Optional[bool] = None,
        source_name: Optional[str] = None,
        checkpoint_hash: Optional[str] = None,
        tcn_variant: str = "lite",
        tcn_hidden_dim: int = 96,
        dropout: float = 0.0,
        spatial_norm: str = "batchnorm",
        mobilenet_pretrained: bool = True,
        mobilenet_variant: str = "small",
        mobilenet_freeze_backbone: bool = True,
        mobilenet_weights_path: Optional[str] = None,
        mobilenet_preserve_pretrained_classifier: bool = False,
        slowfast_fast_pretrained: bool = True,
        slowfast_fast_weights_path: Optional[str] = None,
        train_free_evidence_mode: str = "learned_logits",
        official_action_seg_backend: str = "official_asformer",
        official_num_layers: int = 2,
        matrix_model_id: str = "timm_mobilenetv3_large_100_tsm_tcn",
        matrix_pretrained: bool = True,
        matrix_freeze_backbone: bool = True,
        train_split_supervised: bool = True,
        calibration_split: Optional[str] = "none",
        calibration_temperature: float = 1.0,
        calibration_bias: float = 0.0,
        calibration_artifact: Optional[str] = None,
        calibration_artifact_sha256: Optional[str] = None,
        thumos_trained: bool = False,
        uses_labels: bool = False,
        uses_teacher: bool = False,
        uses_gt: bool = False,
        uses_prediction_cache: bool = False,
        trained_with_thumos_labels: bool = False,
        trained_with_gt_segments: bool = False,
        training_dataset: Optional[str] = None,
        training_supervision_scope: Optional[str] = None,
        return_hidden_features: bool = True,
        require_hidden_features: bool = True,
        hidden_output_kind: str = "pre_temporal_spatial_stem_hidden",
        policy_hidden_gradient_scope: str = "none",
        temporal_probe_stride: int = 1,
        temporal_interpolation_mode: str = "hidden_linear",
        **_: Any,
    ) -> None:
        super().__init__()
        self.probe_model = str(probe_model)
        self.spatial_size = int(spatial_size)
        self.checkpoint_path = self._resolve_path(checkpoint_path)
        self.require_checkpoint = bool(require_checkpoint)
        self.frozen = bool(frozen if trainable is None else not bool(trainable))
        self.tcn_variant = str(tcn_variant)
        self.tcn_hidden_dim = int(tcn_hidden_dim)
        self.dropout = float(dropout)
        self.train_free_evidence_mode = str(train_free_evidence_mode)
        if self.train_free_evidence_mode not in {
            "learned_logits",
            "frozen_feature_change",
            "frozen_semantic_saliency",
            "frozen_transition_fusion",
        }:
            raise ValueError("unsupported train_free_evidence_mode")
        if self.train_free_evidence_mode != "learned_logits":
            if not self.frozen:
                raise ValueError("train-free evidence requires a fully frozen probe")
            if bool(train_split_supervised) or any(
                bool(value)
                for value in (
                    thumos_trained,
                    uses_labels,
                    uses_teacher,
                    uses_gt,
                    uses_prediction_cache,
                    trained_with_thumos_labels,
                    trained_with_gt_segments,
                )
            ):
                raise ValueError("train-free evidence forbids target-dataset supervision and caches")
            if str(calibration_split or "none") != "none":
                raise ValueError("train-free evidence forbids target-dataset calibration")
        self.spatial_norm = str(spatial_norm).lower()
        if self.spatial_norm not in {"batchnorm", "groupnorm"}:
            raise ValueError("spatial_norm must be batchnorm or groupnorm")
        self.source_name = str(source_name or self._default_source_name())
        self.train_split_supervised = bool(train_split_supervised)
        self.calibration_split = str(calibration_split or "none")
        self.calibration_temperature = float(calibration_temperature)
        self.calibration_bias = float(calibration_bias)
        self.calibration_artifact = self._resolve_path(calibration_artifact)
        if not math.isfinite(self.calibration_temperature) or self.calibration_temperature <= 0.0:
            raise ValueError("calibration_temperature must be finite and positive")
        if not math.isfinite(self.calibration_bias):
            raise ValueError("calibration_bias must be finite")
        if self.calibration_split == "train_only":
            if not self.calibration_artifact or not os.path.isfile(self.calibration_artifact):
                raise ValueError("train_only actionness calibration requires a real calibration_artifact")
            actual_hash = self._checkpoint_hash(self.calibration_artifact)
            if not calibration_artifact_sha256 or actual_hash != str(calibration_artifact_sha256):
                raise ValueError("calibration_artifact_sha256 is required and must match the artifact")
            with open(self.calibration_artifact, "r", encoding="utf-8") as handle:
                calibration_payload = json.load(handle)
            if calibration_payload.get("fit_split") not in {"train", "training", "train_only"}:
                raise ValueError("calibration_artifact must declare a train-only fit_split")
            artifact_temperature = float(calibration_payload.get("temperature", float("nan")))
            artifact_bias = float(calibration_payload.get("bias", float("nan")))
            if not math.isclose(artifact_temperature, self.calibration_temperature, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("calibration_temperature does not match calibration_artifact")
            if not math.isclose(artifact_bias, self.calibration_bias, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("calibration_bias does not match calibration_artifact")
        elif self.calibration_split != "none":
            raise ValueError("calibration_split must be 'none' or 'train_only'")
        self.calibration_artifact_sha256 = (
            self._checkpoint_hash(self.calibration_artifact) if self.calibration_artifact else None
        )
        self.return_hidden_features = bool(return_hidden_features)
        self.require_hidden_features = bool(require_hidden_features)
        self.temporal_probe_stride = int(temporal_probe_stride)
        self.temporal_interpolation_mode = str(temporal_interpolation_mode)
        if self.temporal_probe_stride <= 0:
            raise ValueError("temporal_probe_stride must be positive")
        if self.temporal_interpolation_mode not in {"hidden_linear", "nearest"}:
            raise ValueError(
                "temporal_interpolation_mode must be hidden_linear or nearest"
            )
        self.policy_hidden_gradient_scope = str(policy_hidden_gradient_scope)
        if self.policy_hidden_gradient_scope not in {
            "none",
            "asformer_last_encoder_layer",
            "asformer_full_encoder",
        }:
            raise ValueError(
                "policy_hidden_gradient_scope must be none, asformer_last_encoder_layer, "
                "or asformer_full_encoder"
            )
        if self.policy_hidden_gradient_scope != "none" and (
            self.probe_model != "official-action-seg"
            or str(official_action_seg_backend) != "official_asformer"
        ):
            raise ValueError(
                "restricted policy hidden is only supported by the official ASFormer probe"
            )
        if self.require_checkpoint and not self.checkpoint_path:
            raise ValueError("C3CoarseProbeActionnessSource requires checkpoint_path")
        if self.checkpoint_path and not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"C3 coarse probe checkpoint missing: {self.checkpoint_path}")
        self.checkpoint_hash = checkpoint_hash or self._checkpoint_hash(self.checkpoint_path)
        self._provenance_override = {
            "source_type": "c3_coarse_probe",
            "source_name": self.source_name,
            "probe_model": self.probe_model,
            "tcn_variant": self.tcn_variant if self.probe_model == "temporal-tcn" else None,
            "official_action_seg_backend": (
                str(official_action_seg_backend) if self.probe_model == "official-action-seg" else None
            ),
            "matrix_model_id": str(matrix_model_id) if self.probe_model == "matrix-zoo" else None,
            "spatial_size": self.spatial_size,
            "spatial_norm": self.spatial_norm,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
            "train_split_supervised": self.train_split_supervised,
            "calibration_split": self.calibration_split,
            "calibration_temperature": self.calibration_temperature,
            "calibration_bias": self.calibration_bias,
            "calibration_artifact": self.calibration_artifact,
            "calibration_artifact_sha256": self.calibration_artifact_sha256,
            "probability_semantics": (
                "train_only_temperature_bias_calibrated_posterior"
                if self.calibration_split == "train_only"
                else "uncalibrated_sigmoid_score"
            ),
            "thumos_trained": bool(thumos_trained),
            "uses_labels": bool(uses_labels),
            "uses_teacher": bool(uses_teacher),
            "uses_gt": bool(uses_gt),
            "uses_prediction_cache": bool(uses_prediction_cache),
            "trained_with_thumos_labels": bool(trained_with_thumos_labels),
            "trained_with_gt_segments": bool(trained_with_gt_segments),
            "training_dataset": training_dataset,
            "training_supervision_scope": training_supervision_scope,
            "uses_labels_at_inference": False,
            "uses_gt_at_inference": False,
            "uses_teacher_at_inference": False,
            "uses_prediction_cache_at_inference": False,
            "joint_trainable": not self.frozen,
            "checkpoint_is_initialization": bool(self.checkpoint_path),
            "returns_hidden_features": self.return_hidden_features,
            "requires_hidden_features": self.require_hidden_features,
            "hidden_output_kind": str(hidden_output_kind),
            "policy_hidden_gradient_scope": self.policy_hidden_gradient_scope,
            "temporal_probe_stride_dense_candidates": self.temporal_probe_stride,
            "temporal_probe_source_frame_interval": 4 * self.temporal_probe_stride,
            "temporal_interpolation_mode": self.temporal_interpolation_mode,
            "interpolated_hidden_is_selector_evidence": True,
            "selector_receives_anchor_mask": False,
            "selector_receives_anchor_distance": False,
            "train_free_evidence_mode": self.train_free_evidence_mode,
            "target_dataset_optimization": self.train_free_evidence_mode == "learned_logits",
            "probability_semantics": (
                "parameter_free_rank_score_not_action_posterior"
                if self.train_free_evidence_mode != "learned_logits"
                else (
                    "train_only_temperature_bias_calibrated_posterior"
                    if self.calibration_split == "train_only"
                    else "uncalibrated_sigmoid_score"
                )
            ),
        }

        probe_mod = self._probe_module()
        if self.probe_model == "mobilenetv3":
            self.probe = probe_mod.C3MobileNetV3ActionProbe(
                pretrained=bool(mobilenet_pretrained),
                variant=str(mobilenet_variant),
                freeze_backbone=bool(mobilenet_freeze_backbone),
                weights_path=mobilenet_weights_path,
                preserve_pretrained_classifier=bool(
                    mobilenet_preserve_pretrained_classifier
                    or self.train_free_evidence_mode != "learned_logits"
                ),
            )
        elif self.probe_model == "slowfast-fast":
            if self.train_free_evidence_mode not in {
                "frozen_feature_change",
                "frozen_transition_fusion",
            }:
                raise ValueError("SlowFast Fast supports frozen feature-change evidence only")
            self.probe = probe_mod.C3SlowFastFastFrozenProbe(
                pretrained=bool(slowfast_fast_pretrained),
                weights_path=slowfast_fast_weights_path,
            )
        elif self.probe_model == "temporal-tcn":
            self.probe = probe_mod.C3TemporalTCNActionProbe(
                variant=self.tcn_variant,
                spatial_size=self.spatial_size,
                hidden_dim=self.tcn_hidden_dim,
                dropout=self.dropout,
            )
        elif self.probe_model == "official-action-seg":
            self.probe = probe_mod.C3OfficialActionSegmentationProbe(
                backend=str(official_action_seg_backend),
                spatial_size=self.spatial_size,
                hidden_dim=self.tcn_hidden_dim,
                num_layers=int(official_num_layers),
                dropout=self.dropout,
                spatial_norm=self.spatial_norm,
                hidden_output_kind=str(hidden_output_kind),
                policy_hidden_gradient_scope=self.policy_hidden_gradient_scope,
            )
        elif self.probe_model == "matrix-zoo":
            self.probe = probe_mod.C3MatrixZooActionProbe(
                model_id=str(matrix_model_id),
                pretrained=bool(matrix_pretrained),
                freeze_backbone=bool(matrix_freeze_backbone),
            )
        else:
            raise ValueError(
                "probe_model must be one of mobilenetv3, slowfast-fast, temporal-tcn, official-action-seg, or matrix-zoo"
            )
        module = getattr(self.probe, "module", None)
        if isinstance(module, nn.Module):
            self.probe_module = module
        else:
            self.probe_module = None
        if self.checkpoint_path:
            probe_mod._load_probe_checkpoint(self.probe, self.checkpoint_path)
        if self.frozen:
            self.eval()
            for param in self.parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True) -> "C3CoarseProbeActionnessSource":
        if self.frozen:
            super().train(False)
            return self
        return super().train(mode)

    @staticmethod
    def _resolve_path(path: Optional[str]) -> Optional[str]:
        if path in (None, ""):
            return None
        return os.path.abspath(os.path.expandvars(os.path.expanduser(str(path))))

    @staticmethod
    def _checkpoint_hash(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _default_source_name(self) -> str:
        if self.probe_model == "temporal-tcn":
            return f"c3_{self.tcn_variant}_coarse_actionness"
        return f"c3_{self.probe_model}_coarse_actionness"

    @staticmethod
    def _probe_module():
        try:
            from tools.bata import train_lowres_action_probe as probe_mod
        except Exception as exc:  # pragma: no cover - import errors are environment dependent.
            raise RuntimeError("C3 coarse probe source requires tools.bata.train_lowres_action_probe") from exc
        return probe_mod

    def _provenance(self) -> Dict[str, Any]:
        return dict(self._provenance_override)

    def _prepare_probe_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        probe_mod = self._probe_module()
        return probe_mod.prepare_probe_inputs(inputs, probe_model=self.probe_model, spatial_size=self.spatial_size)

    @staticmethod
    def _rank_normalize(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if values.shape != valid.shape:
            raise ValueError("rank-normalized values and valid mask must align")
        rows = []
        for row, keep in zip(values.float(), valid):
            active = row[keep]
            normalized = row.new_zeros(row.shape)
            if active.numel() == 1:
                normalized[keep] = 0.5
            elif active.numel() > 1:
                unique, inverse = torch.unique(active, sorted=True, return_inverse=True)
                if unique.numel() == 1:
                    normalized[keep] = 0.0
                else:
                    normalized[keep] = inverse.to(row.dtype) / float(unique.numel() - 1)
            rows.append(normalized)
        return torch.stack(rows, dim=0)

    @classmethod
    def _parameter_free_evidence(
        cls,
        hidden: torch.Tensor,
        valid: torch.Tensor,
        class_logits: Optional[torch.Tensor],
        mode: str,
    ) -> Dict[str, torch.Tensor]:
        normalized_hidden = F.normalize(hidden.float(), dim=-1, eps=1.0e-6)
        pair_delta = 1.0 - (normalized_hidden[:, 1:] * normalized_hidden[:, :-1]).sum(dim=-1)
        left = hidden.new_zeros(hidden.shape[:2], dtype=torch.float32)
        right = hidden.new_zeros(hidden.shape[:2], dtype=torch.float32)
        left[:, 1:] = pair_delta
        right[:, :-1] = pair_delta
        feature_change = cls._rank_normalize(torch.maximum(left, right), valid)
        if class_logits is not None:
            probs = class_logits.float().softmax(dim=-1)
            entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).eps).log()).sum(dim=-1)
            normalized_entropy = entropy / math.log(float(probs.shape[-1]))
            semantic_raw = 0.5 * probs.amax(dim=-1) + 0.5 * (1.0 - normalized_entropy)
            uncertainty = normalized_entropy.masked_fill(~valid, 0.0)
        else:
            semantic_raw = torch.linalg.vector_norm(hidden.float(), dim=-1)
            semantic_rank = cls._rank_normalize(semantic_raw, valid)
            uncertainty = (1.0 - (2.0 * semantic_rank - 1.0).abs()).masked_fill(~valid, 0.0)
        semantic_saliency = cls._rank_normalize(semantic_raw, valid)
        previous = F.pad(uncertainty[:, :-1], (1, 0), value=0.0)
        following = F.pad(uncertainty[:, 1:], (0, 1), value=0.0)
        uncertainty_peak = cls._rank_normalize(
            (uncertainty - torch.maximum(previous, following)).clamp_min(0.0),
            valid,
        )
        fusion = (
            0.75 * feature_change
            + 0.20 * semantic_saliency
            + 0.05 * uncertainty_peak
        ).masked_fill(~valid, 0.0)
        evidence = {
            "frozen_feature_change": feature_change,
            "frozen_semantic_saliency": semantic_saliency,
            "frozen_transition_fusion": fusion,
        }[mode]
        return {
            "evidence": evidence,
            "feature_change": feature_change,
            "semantic_saliency": semantic_saliency,
            "uncertainty_peak": uncertainty_peak,
            "fusion": fusion,
        }

    @staticmethod
    def _sparse_probe_positions(valid: torch.Tensor, stride: int) -> torch.Tensor:
        if valid.ndim != 2 or valid.dtype != torch.bool:
            raise ValueError("sparse probe valid mask must be bool [B,T]")
        if stride <= 0:
            raise ValueError("sparse probe stride must be positive")
        temporal_len = int(valid.shape[1])
        if temporal_len <= 0 or not bool(valid.any(dim=1).all().item()):
            raise ValueError("every sparse probe row needs at least one valid candidate")
        regular = torch.arange(0, temporal_len, int(stride), device=valid.device)
        indices = torch.arange(temporal_len, device=valid.device)
        first = torch.where(valid, indices[None, :], temporal_len).amin(dim=1)
        last = torch.where(valid, indices[None, :], -1).amax(dim=1)
        return torch.unique(torch.cat((regular, first, last)), sorted=True)

    @staticmethod
    def _gather_sparse_inputs(inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        temporal_dim = 3 if inputs.ndim == 6 else 2 if inputs.ndim == 5 else None
        if temporal_dim is None:
            raise ValueError("sparse coarse probe expects raw video with five or six dimensions")
        return torch.index_select(inputs, temporal_dim, positions.to(device=inputs.device))

    @staticmethod
    def _reconstruct_sparse_sequence(
        values: torch.Tensor,
        *,
        anchor_positions: torch.Tensor,
        anchor_valid: torch.Tensor,
        dense_valid: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        if values.ndim not in {2, 3}:
            raise ValueError("sparse probe values must be [B,A] or [B,A,D]")
        if anchor_valid.shape != values.shape[:2]:
            raise ValueError("anchor_valid must match sparse values [B,A]")
        if dense_valid.ndim != 2 or dense_valid.shape[0] != values.shape[0]:
            raise ValueError("dense_valid must be [B,T]")
        if anchor_positions.ndim != 1 or anchor_positions.numel() != values.shape[1]:
            raise ValueError("anchor_positions must match sparse axis A")
        if mode not in {"hidden_linear", "nearest"}:
            raise ValueError("unknown sparse probe reconstruction mode")
        dense_len = int(dense_valid.shape[1])
        queries = torch.arange(dense_len, device=values.device, dtype=torch.long)
        rows = []
        for batch_idx in range(int(values.shape[0])):
            keep = anchor_valid[batch_idx]
            x = anchor_positions.to(device=values.device, dtype=torch.long)[keep]
            y = values[batch_idx, keep]
            if x.numel() == 0:
                raise ValueError("every sparse probe row needs one valid anchor")
            right = torch.searchsorted(x, queries).clamp(max=int(x.numel()) - 1)
            left = (right - 1).clamp(min=0)
            right_x = x[right]
            left_x = x[left]
            if mode == "nearest":
                choose_right = (queries - left_x) > (right_x - queries)
                gather = torch.where(choose_right, right, left)
                row = y[gather]
            else:
                denom = (right_x - left_x).clamp_min(1).to(dtype=values.dtype)
                weight = (queries - left_x).to(dtype=values.dtype) / denom
                if values.ndim == 3:
                    weight = weight[:, None]
                row = y[left] + weight * (y[right] - y[left])
            row_valid = dense_valid[batch_idx]
            if values.ndim == 3:
                row = row.masked_fill(~row_valid[:, None], 0.0)
            else:
                row = row.masked_fill(~row_valid, 0.0)
            rows.append(row)
        return torch.stack(rows, dim=0)

    def _estimate_probe_profile(self, inputs: torch.Tensor, logits: torch.Tensor, latency_ms: Optional[float]) -> Dict[str, Any]:
        params = _module_param_counts(self)
        batch = int(logits.shape[0])
        temporal_len = int(logits.shape[1])
        tokens = batch * temporal_len
        spatial = int(self.spatial_size)
        estimate_breakdown: Dict[str, int] = {}
        if self.probe_model == "mobilenetv3":
            per_frame_macs = int(56_500_000 * (spatial / 224.0) ** 2)
            macs = tokens * per_frame_macs
            family = "MobileNetV3-small"
        elif self.probe_model == "slowfast-fast":
            macs = int(tokens * max(params["total"], 1))
            family = "SlowFast-R50/Fast-pathway-only"
        elif self.probe_model == "temporal-tcn":
            hidden = max(96 if self.tcn_variant in {"asformer_lite", "fact_lite", "temporal_mamba_lite", "ms_tcnpp", "c2f_tcn"} else 32, self.tcn_hidden_dim)
            stem_macs = tokens * (3 * hidden * 9 * max(1, spatial // 2) * max(1, spatial // 2) // 16)
            temporal_macs = tokens * hidden * hidden * (10 if self.tcn_variant == "asformer_lite" else 6)
            if self.tcn_variant == "asformer_lite":
                temporal_macs += batch * temporal_len * temporal_len * hidden * 4
            macs = int(stem_macs + temporal_macs)
            family = f"TemporalTCN/{self.tcn_variant}"
        elif self.probe_model == "official-action-seg":
            hidden = max(16, self.tcn_hidden_dim)
            first_spatial = (spatial + 1) // 2
            second_spatial = (first_spatial + 1) // 2
            stem_conv1_macs = tokens * first_spatial * first_spatial * hidden * 3 * 3 * 3
            stem_conv2_macs = tokens * second_spatial * second_spatial * hidden * hidden * 3 * 3
            temporal_layers = max(1, int(getattr(self.probe, "num_layers", 1)))
            temporal_linear_macs = tokens * hidden * hidden * 10 * temporal_layers
            sliding_window = min(64, temporal_len)
            temporal_attention_macs = (
                batch * temporal_len * sliding_window * hidden * 4 * temporal_layers * 2
            )
            estimate_breakdown = {
                "spatial_stem_conv1_macs": int(stem_conv1_macs),
                "spatial_stem_conv2_macs": int(stem_conv2_macs),
                "official_asformer_linear_macs_approx": int(temporal_linear_macs),
                "official_asformer_sliding_attention_macs_approx": int(temporal_attention_macs),
            }
            macs = int(sum(estimate_breakdown.values()))
            family = f"OfficialActionSeg/{self._provenance_override.get('official_action_seg_backend')}"
        else:
            macs = int(tokens * max(params["total"], 1))
            family = "matrix-zoo"
        return {
            "source_name": self.source_name,
            "source_type": "c3_coarse_probe",
            "source_kind": "task_adapted_coarse_classifier",
            "probe_model": self.probe_model,
            "model_family": family,
            "official_action_seg_backend": self._provenance_override.get("official_action_seg_backend"),
            "spatial_size": spatial,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
            "train_split_supervised": self.train_split_supervised,
            "cache_lookup_or_interpolation": False,
            "online_backbone_flops_included": True,
            "estimated_macs": int(macs),
            "estimated_flops": int(2 * macs),
            "estimate_breakdown": estimate_breakdown,
            "estimate_semantics": "architecture_analytic_estimate_not_operator_profiler_measurement",
            "parameters": params,
            "latency_ms": {"coarse_probe_ms": latency_ms},
            "input_shape": [int(v) for v in inputs.shape],
            "output_shape": [int(v) for v in logits.shape],
        }

    def forward(
        self,
        inputs: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        if inputs.ndim not in {5, 6}:
            raise ValueError(f"C3 coarse probe actionness expects raw video [B,C,T,H,W] or [B,N,C,T,H,W], got {tuple(inputs.shape)}")
        if inputs.ndim == 6:
            temporal_len = int(inputs.shape[3])
            batch = int(inputs.shape[0])
        else:
            temporal_len = int(inputs.shape[2])
            batch = int(inputs.shape[0])
        if valid_mask is None:
            valid = torch.ones(batch, temporal_len, dtype=torch.bool, device=inputs.device)
        else:
            valid = valid_mask.to(device=inputs.device, dtype=torch.bool)
        if valid.shape != (batch, temporal_len):
            raise ValueError(f"valid_mask must be [B,T]={batch, temporal_len}, got {tuple(valid.shape)}")
        start = time.perf_counter()
        anchor_positions = self._sparse_probe_positions(valid, self.temporal_probe_stride)
        sparse_inputs = self._gather_sparse_inputs(inputs, anchor_positions)
        sparse_valid = torch.index_select(valid, 1, anchor_positions)
        probe_inputs = self._prepare_probe_inputs(sparse_inputs)
        if hasattr(probe_inputs, "to"):
            probe_inputs = probe_inputs.to(device=inputs.device)
        def call_probe() -> Any:
            def invoke() -> Any:
                try:
                    return self.probe(
                        probe_inputs,
                        sparse_valid,
                        return_hidden=self.return_hidden_features,
                    )
                except TypeError:
                    if self.return_hidden_features and self.require_hidden_features:
                        raise
                    return self.probe(probe_inputs, sparse_valid)

            if self.probe_model == "official-action-seg":
                # ASFormer temporal conv gradients can overflow under the outer
                # FP16 autocast and initial GradScaler scale. Keep this small
                # coarse path differentiable but numerically stable in FP32.
                with torch.autocast(device_type=inputs.device.type, enabled=False):
                    return invoke()
            return invoke()

        if self.frozen:
            with torch.no_grad():
                probe_output = call_probe()
        else:
            probe_output = call_probe()
        hidden = None
        class_logits = None
        policy_hidden = None
        hidden_kind = None
        official_source_sha256 = None
        official_source_normalized_lf_sha256 = None
        official_source_file = None
        if isinstance(probe_output, Mapping):
            logits = probe_output.get("logits")
            hidden = (
                probe_output.get("coarse_hidden_features")
                if probe_output.get("coarse_hidden_features") is not None
                else probe_output.get("hidden")
            )
            policy_hidden = probe_output.get("policy_hidden")
            if logits is None:
                raise ValueError("C3 coarse probe output mapping must contain logits")
            hidden_kind = probe_output.get("hidden_kind")
            official_source_sha256 = probe_output.get("official_source_sha256")
            official_source_normalized_lf_sha256 = probe_output.get(
                "official_source_normalized_lf_sha256"
            )
            official_source_file = probe_output.get("official_source_file")
            class_logits = probe_output.get("class_logits")
        else:
            logits = probe_output
        if hidden is None and self.return_hidden_features and self.require_hidden_features:
            raise ValueError("C3 coarse probe must return hidden features for final DUCA selector fusion")
        logits = logits.float().to(device=inputs.device)
        if logits.shape != sparse_valid.shape:
            raise ValueError("C3 sparse coarse logits must align with sparse anchors [B,A]")
        if hidden is not None:
            if hidden.ndim != 3:
                raise ValueError(f"C3 coarse hidden features must be [B,T,D], got {tuple(hidden.shape)}")
            if hidden.shape[:2] != logits.shape:
                raise ValueError("C3 coarse hidden features must align with sparse logits [B,A]")
            hidden = hidden.float().to(device=inputs.device)
        if class_logits is not None:
            if class_logits.ndim != 3 or class_logits.shape[:2] != logits.shape:
                raise ValueError("frozen class logits must align with sparse anchors [B,A,C]")
            class_logits = class_logits.float().to(device=inputs.device)
        if policy_hidden is not None:
            if hidden is None:
                raise ValueError("restricted policy hidden requires the shared coarse hidden")
            if policy_hidden.shape != hidden.shape:
                raise ValueError("restricted policy hidden must match shared coarse hidden [B,T,D]")
            policy_hidden = policy_hidden.float().to(device=inputs.device)
        sparse_logits = logits
        sparse_hidden = hidden
        sparse_policy_hidden = policy_hidden
        sparse_class_logits = class_logits
        if self.temporal_probe_stride > 1:
            logits = self._reconstruct_sparse_sequence(
                sparse_logits,
                anchor_positions=anchor_positions,
                anchor_valid=sparse_valid,
                dense_valid=valid,
                mode=self.temporal_interpolation_mode,
            )
            if sparse_hidden is not None:
                hidden = self._reconstruct_sparse_sequence(
                    sparse_hidden,
                    anchor_positions=anchor_positions,
                    anchor_valid=sparse_valid,
                    dense_valid=valid,
                    mode=self.temporal_interpolation_mode,
                )
            if sparse_policy_hidden is not None:
                policy_hidden = self._reconstruct_sparse_sequence(
                    sparse_policy_hidden,
                    anchor_positions=anchor_positions,
                    anchor_valid=sparse_valid,
                    dense_valid=valid,
                    mode=self.temporal_interpolation_mode,
                )
            if sparse_class_logits is not None:
                class_logits = self._reconstruct_sparse_sequence(
                    sparse_class_logits,
                    anchor_positions=anchor_positions,
                    anchor_valid=sparse_valid,
                    dense_valid=valid,
                    mode=self.temporal_interpolation_mode,
                )
        logits = logits.masked_fill(~valid, _neg(torch.float32))
        if hidden is not None:
            hidden = hidden.masked_fill(~valid[:, :, None], 0.0)
        if policy_hidden is not None:
            policy_hidden = policy_hidden.masked_fill(~valid[:, :, None], 0.0)
        latency_ms = float((time.perf_counter() - start) * 1000.0)
        train_free_payload = None
        if self.train_free_evidence_mode != "learned_logits":
            if hidden is None:
                raise ValueError("train-free evidence requires frozen hidden features")
            train_free_payload = self._parameter_free_evidence(
                hidden,
                valid,
                class_logits,
                self.train_free_evidence_mode,
            )
            p_action = train_free_payload["evidence"].clamp(1.0e-6, 1.0 - 1.0e-6)
            p_action = p_action.masked_fill(~valid, 0.0)
            calibrated_logits = torch.logit(p_action).masked_fill(~valid, _neg(torch.float32))
            logits = calibrated_logits
        else:
            calibrated_logits = (logits + self.calibration_bias) / self.calibration_temperature
            p_action = torch.sigmoid(calibrated_logits).masked_fill(~valid, 0.0)
        transition = _actionness_transition_payload(p_action, valid)
        profile = self._estimate_probe_profile(sparse_inputs, sparse_logits, latency_ms)
        hidden_width = 0 if hidden is None else int(hidden.shape[-1])
        policy_width = 0 if policy_hidden is None else int(policy_hidden.shape[-1])
        interpolated_values = batch * temporal_len * (1 + hidden_width + policy_width)
        interpolation_macs = 0 if self.temporal_probe_stride == 1 else 3 * interpolated_values
        profile.update(
            {
                "temporal_probe_stride_dense_candidates": self.temporal_probe_stride,
                "temporal_probe_source_frame_interval": 4 * self.temporal_probe_stride,
                "sparse_anchor_count": int(anchor_positions.numel()),
                "dense_output_length": temporal_len,
                "computed_frame_fraction": float(anchor_positions.numel()) / float(temporal_len),
                "temporal_interpolation_mode": self.temporal_interpolation_mode,
                "interpolation_estimated_macs": int(interpolation_macs),
                "estimated_macs": int(profile["estimated_macs"]) + int(interpolation_macs),
                "estimated_flops": int(profile["estimated_flops"]) + int(2 * interpolation_macs),
                "output_shape": [batch, temporal_len],
                "cache_lookup_or_interpolation": self.temporal_probe_stride > 1,
            }
        )
        output = {
            "p_action": transition["p_action"],
            "logits": logits,
            "actionness_logits": logits,
            "calibrated_actionness_logits": calibrated_logits.masked_fill(~valid, _neg(torch.float32)),
            "uncertainty": transition["uncertainty"],
            "entropy": transition["entropy"],
            "delta_p_action": transition["delta_p_action"],
            "abs_delta_p_action": transition["abs_delta_p_action"],
            "uncertainty_peak": transition["uncertainty_peak"],
            "transition_score": transition["transition_score"],
            "features": transition["features"],
            "valid_mask": valid,
            "provenance": self._provenance(),
            "compute_profile": profile,
            "source_name": self.source_name,
            "sparse_probe_anchor_positions": anchor_positions.detach().cpu().tolist(),
        }
        if train_free_payload is not None:
            output.update(
                {
                    "train_free_feature_change": train_free_payload["feature_change"],
                    "train_free_semantic_saliency": train_free_payload["semantic_saliency"],
                    "train_free_uncertainty_peak": train_free_payload["uncertainty_peak"],
                    "train_free_transition_fusion": train_free_payload["fusion"],
                    "train_free_evidence_mode": self.train_free_evidence_mode,
                }
            )
        if hidden is not None:
            output["coarse_hidden_features"] = hidden
            output["hidden_features"] = hidden
            output["hidden_kind"] = hidden_kind
            output["official_source_sha256"] = official_source_sha256
            output["official_source_normalized_lf_sha256"] = official_source_normalized_lf_sha256
            output["official_source_file"] = official_source_file
            if policy_hidden is not None:
                output["policy_hidden_features"] = policy_hidden
                output["policy_hidden_gradient_scope"] = self.policy_hidden_gradient_scope
            profile["hidden_output_shape"] = [int(v) for v in hidden.shape]
            profile["hidden_kind"] = hidden_kind
            profile["official_source_sha256"] = official_source_sha256
            profile["official_source_normalized_lf_sha256"] = official_source_normalized_lf_sha256
            output["provenance"].update(
                {
                    "hidden_kind": hidden_kind,
                    "official_source_sha256": official_source_sha256,
                    "official_source_normalized_lf_sha256": official_source_normalized_lf_sha256,
                    "official_source_file": official_source_file,
                }
            )
        return output


class DucaAcquisitionAdapter(nn.Module):
    """Online DUCA acquisition adapter with hard budgeted sparse output."""

    def __init__(
        self,
        feature_dim: Optional[int] = None,
        hidden_dim: int = 96,
        actionness_source: Optional[nn.Module] = None,
        budget: Optional[int] = 384,
        budget_mode: str = "fixed",
        budget_min: int = 64,
        budget_max: Optional[int] = None,
        budget_multiple: int = 16,
        target_budget: Optional[float] = None,
        allow_external_budget_override: Optional[bool] = None,
        budget_controller: Optional[nn.Module] = None,
        max_radius: int = 16,
        acquisition_policy: str = "legacy_center_radius",
        structured_temperature: float = 1.0,
        local_cell_force_exact_uniform: bool = False,
        density_temperature: float = 0.7,
        density_coverage_floor: float = 0.05,
        density_smoothing_kernel: int = 5,
        semantic_phase_sigma: float = 2.0,
        semantic_phase_scaffold_budget: int = 128,
        semantic_phase_onset_budget: int = 64,
        semantic_phase_offset_budget: int = 64,
        semantic_phase_core_budget: int = 128,
        sampling_rate_utility_components: str = "none",
        actionness_weight: float = 0.05,
        transition_weight: float = 1.0,
        uncertainty_weight: float = 0.25,
        utility_weight: float = 0.50,
        boundary_weight: float = 1.0,
        selector_variant: str = "direct_boundary",
        parameter_free_selector: bool = False,
        transition_objective: str = "gaussian_mass",
        boundary_burst_radius: int = 2,
        boundary_burst_quota: float = 5.0,
        boundary_burst_budget_fraction: float = 0.25,
        boundary_burst_context_weight: float = 0.05,
        boundary_burst_center_temperature: float = 0.7,
        boundary_burst_offset_temperature: float = 1.0,
        boundary_burst_require_bilateral_offsets: bool = False,
        boundary_burst_require_global_mandatory_groups: bool = False,
        coarse_hidden_dim: Optional[int] = None,
        require_coarse_hidden_features: bool = False,
        policy_hidden_gradient_scale: float = 0.0,
        auxiliary_hidden_gradient_scale: float = 1.0,
        max_unselected_hole: Optional[int] = None,
        hard_max_gap_repair: bool = True,
        fail_on_infeasible_max_gap: bool = True,
        profile_runtime: bool = False,
        profile_sync_cuda: bool = True,
    ) -> None:
        super().__init__()
        self.budget_mode = str(budget_mode)
        if self.budget_mode not in {"fixed", "dynamic_must"}:
            raise ValueError("budget_mode must be fixed or dynamic_must")
        if self.budget_mode == "dynamic_must":
            if budget_max is None:
                raise ValueError("dynamic_must requires budget_max")
            hard_cap = int(budget_max)
            self.allow_external_budget_override = (
                False if allow_external_budget_override is None else bool(allow_external_budget_override)
            )
        else:
            if budget is None:
                raise ValueError("fixed budget mode requires budget")
            hard_cap = int(budget)
            self.allow_external_budget_override = (
                True if allow_external_budget_override is None else bool(allow_external_budget_override)
            )
        self.dynamic_budget = self.budget_mode == "dynamic_must"
        self.budget = int(hard_cap)
        self.default_budget = int(hard_cap if budget is None else budget)
        self.budget_min = int(budget_min if self.dynamic_budget else self.budget)
        self.budget_max = int(self.budget)
        self.budget_multiple = int(budget_multiple)
        self.target_budget = float(self.budget if target_budget is None else target_budget)
        self.max_radius = int(max_radius)
        self.acquisition_policy = str(acquisition_policy)
        if self.acquisition_policy not in {
            "legacy_center_radius",
            "global_structured_topk",
            "local_cell_deformation",
            "continuous_density_transport",
            "continuous_mixture_density_transport",
            "budget_calibrated_sampling_rate",
            "semantic_phase_sampling",
        }:
            raise ValueError(
                "acquisition_policy must be legacy_center_radius, global_structured_topk, "
                "local_cell_deformation, continuous_density_transport, or "
                "continuous_mixture_density_transport, budget_calibrated_sampling_rate, "
                "or semantic_phase_sampling"
            )
        self.structured_temperature = float(structured_temperature)
        self.local_cell_force_exact_uniform = bool(local_cell_force_exact_uniform)
        self.density_temperature = float(density_temperature)
        self.density_coverage_floor = float(density_coverage_floor)
        self.density_smoothing_kernel = int(density_smoothing_kernel)
        self.semantic_phase_sigma = float(semantic_phase_sigma)
        self.semantic_phase_budgets = {
            "scaffold": int(semantic_phase_scaffold_budget),
            "onset": int(semantic_phase_onset_budget),
            "offset": int(semantic_phase_offset_budget),
            "core": int(semantic_phase_core_budget),
        }
        self.sampling_rate_utility_components = str(
            sampling_rate_utility_components
        ).lower()
        if self.sampling_rate_utility_components not in {"none", "cls", "reg", "both"}:
            raise ValueError(
                "sampling_rate_utility_components must be none, cls, reg, or both"
            )
        if not math.isfinite(self.structured_temperature) or self.structured_temperature <= 0.0:
            raise ValueError("structured_temperature must be finite and positive")
        if not math.isfinite(self.density_temperature) or self.density_temperature <= 0.0:
            raise ValueError("density_temperature must be finite and positive")
        if not math.isfinite(self.density_coverage_floor) or not 0.0 <= self.density_coverage_floor < 1.0:
            raise ValueError("density_coverage_floor must lie in [0,1)")
        if self.density_smoothing_kernel <= 0 or self.density_smoothing_kernel % 2 == 0:
            raise ValueError("density_smoothing_kernel must be a positive odd integer")
        if not math.isfinite(self.semantic_phase_sigma) or self.semantic_phase_sigma <= 0.0:
            raise ValueError("semantic_phase_sigma must be finite and positive")
        if any(value < 0 for value in self.semantic_phase_budgets.values()):
            raise ValueError("semantic phase budgets must be non-negative")
        if self.acquisition_policy == "semantic_phase_sampling":
            if self.dynamic_budget:
                raise ValueError("semantic_phase_sampling requires a fixed exact budget")
            if sum(self.semantic_phase_budgets.values()) != int(hard_cap):
                raise ValueError("semantic phase budgets must sum to the fixed detector budget")
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if self.dynamic_budget:
            if self.budget_min <= 0 or self.budget_min > self.budget:
                raise ValueError("budget_min must lie in (0, budget_max]")
            if self.budget_multiple <= 0:
                raise ValueError("budget_multiple must be positive")
            if (self.budget - self.budget_min) % self.budget_multiple != 0:
                raise ValueError("budget_multiple must divide budget_max - budget_min")
            if not (0.0 < self.target_budget <= float(self.budget)):
                raise ValueError("target_budget must lie in (0, budget_max]")
        if self.max_radius < 0:
            raise ValueError("max_radius must be non-negative")
        self.actionness_weight = float(actionness_weight)
        self.transition_weight = float(transition_weight)
        self.uncertainty_weight = float(uncertainty_weight)
        self.utility_weight = float(utility_weight)
        self.boundary_weight = float(boundary_weight)
        self.selector_variant = str(selector_variant)
        if self.selector_variant not in {"direct_boundary", "transition_only"}:
            raise ValueError("selector_variant must be direct_boundary or transition_only")
        self.parameter_free_selector = bool(parameter_free_selector)
        if self.parameter_free_selector:
            if self.selector_variant != "direct_boundary":
                raise ValueError("parameter-free selection uses the direct_boundary adapter path")
            if self.dynamic_budget:
                raise ValueError("parameter-free selection requires a fixed budget")
            if self.acquisition_policy != "global_structured_topk":
                raise ValueError("parameter-free selection requires global_structured_topk")
        self.transition_objective = str(transition_objective)
        if self.transition_objective not in {"gaussian_mass", "boundary_burst"}:
            raise ValueError("transition_objective must be gaussian_mass or boundary_burst")
        self.boundary_burst_radius = int(boundary_burst_radius)
        self.boundary_burst_quota = float(boundary_burst_quota)
        self.boundary_burst_budget_fraction = float(boundary_burst_budget_fraction)
        self.boundary_burst_context_weight = float(boundary_burst_context_weight)
        self.boundary_burst_center_temperature = float(boundary_burst_center_temperature)
        self.boundary_burst_offset_temperature = float(boundary_burst_offset_temperature)
        self.boundary_burst_require_bilateral_offsets = bool(
            boundary_burst_require_bilateral_offsets
        )
        self.boundary_burst_require_global_mandatory_groups = bool(
            boundary_burst_require_global_mandatory_groups
        )
        if self.transition_objective == "boundary_burst":
            if self.selector_variant != "transition_only" and not self.parameter_free_selector:
                raise ValueError("boundary_burst requires transition_only or parameter-free evidence")
            if self.boundary_burst_radius <= 0 or self.boundary_burst_quota <= 0.0:
                raise ValueError("boundary burst radius/quota must be positive")
            if not 0.0 < self.boundary_burst_budget_fraction <= 1.0:
                raise ValueError("boundary burst budget fraction must lie in (0,1]")
            if self.boundary_burst_context_weight < 0.0:
                raise ValueError("boundary burst context weight must be non-negative")
            if min(
                self.boundary_burst_center_temperature,
                self.boundary_burst_offset_temperature,
            ) <= 0.0:
                raise ValueError("boundary burst temperatures must be positive")
        self.coarse_hidden_dim = 0 if coarse_hidden_dim in (None, 0) else int(coarse_hidden_dim)
        if self.coarse_hidden_dim < 0:
            raise ValueError("coarse_hidden_dim must be non-negative")
        self.require_coarse_hidden_features = bool(require_coarse_hidden_features)
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
        if self.parameter_free_selector and self.max_unselected_hole is None:
            raise ValueError("parameter-free selection requires an explicit max_unselected_hole")
        self.hard_max_gap_repair = bool(hard_max_gap_repair)
        if self.acquisition_policy in {
            "global_structured_topk",
            "local_cell_deformation",
            "continuous_density_transport",
            "continuous_mixture_density_transport",
            "budget_calibrated_sampling_rate",
            "semantic_phase_sampling",
        } and self.hard_max_gap_repair:
            raise ValueError("structured acquisition policies encode coverage and forbid hard repair")
        self.fail_on_infeasible_max_gap = bool(fail_on_infeasible_max_gap)
        self.profile_runtime = bool(profile_runtime)
        self.profile_sync_cuda = bool(profile_sync_cuda)
        self.last_compute_profile: Dict[str, Any] = {}
        self.actionness_source = actionness_source or ZeroShotActionnessSource(feature_dim=feature_dim, mode="motion")
        self.feature_dim = None if feature_dim is None else int(feature_dim)
        self.transition_scorer = None
        self.density_mixture_head = None
        self.sampling_rate_utility_fusion = None
        if self.selector_variant == "transition_only":
            if self.dynamic_budget:
                raise ValueError("transition_only is intentionally fixed-budget until its fixed policy is validated")
            if self.acquisition_policy not in {
                "global_structured_topk",
                "local_cell_deformation",
                "continuous_density_transport",
                "continuous_mixture_density_transport",
                "budget_calibrated_sampling_rate",
                "semantic_phase_sampling",
            }:
                raise ValueError("transition_only requires a structured exact-budget acquisition policy")
            if self.acquisition_policy == "global_structured_topk" and self.max_unselected_hole is None:
                raise ValueError("global_structured_topk requires an explicit max_unselected_hole")
            if self.acquisition_policy in {
                "continuous_density_transport",
                "continuous_mixture_density_transport",
                "budget_calibrated_sampling_rate",
                "semantic_phase_sampling",
            } and self.dynamic_budget:
                raise ValueError("continuous density/rate/phase transport currently requires a fixed exact budget")
            if self.coarse_hidden_dim <= 0:
                raise ValueError("transition_only requires official ASFormer encoder hidden features")
            if int(hidden_dim) <= 0:
                raise ValueError("transition_only requires a positive scorer hidden dimension")
            self.encoder = None
            self.center_head = None
            self.radius_head = None
            self.start_head = None
            self.end_head = None
            self.context_head = None
            self.utility_head = None
            self.transition_scorer = DucaTransitionUtilityScorer(
                hidden_dim=self.coarse_hidden_dim,
                scorer_hidden_dim=int(hidden_dim),
                zero_init_output=self.acquisition_policy in {
                    "local_cell_deformation",
                    "continuous_density_transport",
                    "continuous_mixture_density_transport",
                },
                burst_radius=(
                    self.boundary_burst_radius
                    if self.transition_objective == "boundary_burst"
                    else 0
                ),
            )
            if self.acquisition_policy == "continuous_mixture_density_transport":
                self.density_mixture_head = DucaMixtureDensityHead(
                    hidden_dim=self.coarse_hidden_dim,
                    scorer_hidden_dim=int(hidden_dim),
                )
            if self.acquisition_policy in {"budget_calibrated_sampling_rate", "semantic_phase_sampling"}:
                self.sampling_rate_utility_fusion = nn.Linear(2, 1)
                nn.init.zeros_(self.sampling_rate_utility_fusion.weight)
                nn.init.zeros_(self.sampling_rate_utility_fusion.bias)
            selector_feature_dim = int(self.transition_scorer.input_dim)
        elif self.feature_dim is None or self.parameter_free_selector:
            self.encoder = None
            self.center_head = None
            self.radius_head = None
            self.start_head = None
            self.end_head = None
            self.context_head = None
            self.utility_head = None
            selector_feature_dim = DUCA_ACTIONNESS_FEATURE_DIM
        else:
            in_dim = self.feature_dim + DUCA_ACTIONNESS_FEATURE_DIM + self.coarse_hidden_dim
            if self.dynamic_budget and int(hidden_dim) <= 0:
                raise ValueError("dynamic_must requires a positive hidden_dim when feature_dim is provided")
            self.encoder = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.GELU(),
            )
            self.center_head = nn.Linear(int(hidden_dim), 1)
            self.radius_head = (
                nn.Linear(int(hidden_dim), 1) if self.acquisition_policy == "legacy_center_radius" else None
            )
            self.start_head = nn.Linear(int(hidden_dim), 1)
            self.end_head = nn.Linear(int(hidden_dim), 1)
            self.context_head = nn.Linear(int(hidden_dim), 1)
            self.utility_head = nn.Linear(int(hidden_dim), 1)
            selector_feature_dim = int(hidden_dim)
        if self.dynamic_budget:
            self.budget_controller = budget_controller or PrefixMarginalUtilityBudgetController(
                hidden_dim=selector_feature_dim,
                budget_min=self.budget_min,
                budget_max=self.budget,
                budget_multiple=self.budget_multiple,
                target_budget=self.target_budget,
            )
        else:
            self.budget_controller = budget_controller

    def _estimate_actionness_profile(
        self,
        *,
        batch_size: int,
        temporal_len: int,
        feature_dim: int,
        external_cached: bool,
        external_source_name: Optional[str],
    ) -> Dict[str, Any]:
        source_mode = str(getattr(self.actionness_source, "mode", "unknown"))
        source_name = str(external_source_name or getattr(self.actionness_source, "_provenance", lambda: {})().get("source_name", source_mode))
        tokens = int(batch_size) * int(temporal_len)
        params = _module_param_counts(self.actionness_source)
        if external_cached:
            flops = tokens * 16
            return {
                "source_name": source_name,
                "source_type": source_mode,
                "source_kind": "external_cached_prior",
                "cache_lookup_or_interpolation": True,
                "online_backbone_flops_included": False,
                "estimated_macs": 0,
                "estimated_flops": int(flops),
                "parameters": params,
            }
        if source_mode == "motion":
            motion_flops = tokens * (8 * int(feature_dim) + 40)
            return {
                "source_name": source_name,
                "source_type": source_mode,
                "source_kind": "online_zero_shot_motion",
                "cache_lookup_or_interpolation": False,
                "online_backbone_flops_included": True,
                "estimated_macs": 0,
                "estimated_flops": int(motion_flops),
                "parameters": params,
            }
        if source_mode == "feature_mlp" and getattr(self.actionness_source, "net", None) is not None:
            hidden = int(self.actionness_source.net[1].out_features)
            macs = tokens * (int(feature_dim) * hidden + hidden)
            flops = 2 * macs + tokens * (6 * int(feature_dim) + 8 * hidden + 16)
            return {
                "source_name": source_name,
                "source_type": source_mode,
                "source_kind": "online_feature_mlp",
                "cache_lookup_or_interpolation": False,
                "online_backbone_flops_included": True,
                "estimated_macs": int(macs),
                "estimated_flops": int(flops),
                "parameters": params,
            }
        if source_mode == "video_text":
            return {
                "source_name": source_name,
                "source_type": source_mode,
                "source_kind": "online_video_text_external_model",
                "cache_lookup_or_interpolation": False,
                "online_backbone_flops_included": False,
                "estimated_macs": 0,
                "estimated_flops": 0,
                "parameters": params,
                "note": "injected video-text backbone FLOPs are provider/model dependent and are not estimated here",
            }
        flops = tokens * 16
        return {
            "source_name": source_name,
            "source_type": source_mode,
            "source_kind": f"online_{source_mode}",
            "cache_lookup_or_interpolation": False,
            "online_backbone_flops_included": True,
            "estimated_macs": 0,
            "estimated_flops": int(flops),
            "parameters": params,
        }

    def _estimate_selector_profile(
        self,
        *,
        batch_size: int,
        temporal_len: int,
        feature_dim: int,
    ) -> Dict[str, Any]:
        tokens = int(batch_size) * int(temporal_len)
        params = _module_param_counts(self)
        actionness_params = _module_param_counts(self.actionness_source)
        selector_params = {
            "total": int(params["total"] - actionness_params["total"]),
            "trainable": int(params["trainable"] - actionness_params["trainable"]),
        }
        if self.encoder is None:
            if self.transition_scorer is not None:
                hidden = int(self.transition_scorer.scorer_hidden_dim)
                in_dim = int(self.transition_scorer.input_dim)
                macs = tokens * (in_dim * hidden + hidden)
                flops = 2 * macs + tokens * (6 * in_dim + 8 * hidden + 16)
                density_head = "single_transition_density"
                if self.density_mixture_head is not None:
                    mixture_macs = tokens * (
                        (int(self.coarse_hidden_dim) + 2) * hidden + hidden
                    ) + int(batch_size) * 9
                    macs += mixture_macs
                    flops += 2 * mixture_macs + tokens * (8 * hidden + 16)
                    density_head = "boundary_uncertainty_context_mixture"
                return {
                    "head": "DucaTransitionUtilityScorer",
                    "density_head": density_head,
                    "selector_variant": self.selector_variant,
                    "hidden_dim": hidden,
                    "input_dim": in_dim,
                    "coarse_hidden_dim": int(self.coarse_hidden_dim),
                    "uses_absolute_hidden_features": False,
                    "uses_raw_rgb_descriptors": False,
                    "estimated_macs": int(macs),
                    "estimated_flops": int(flops),
                    "parameters": selector_params,
                }
            flops = tokens * 24
            return {
                "head": "analytic_score_no_mlp",
                "hidden_dim": 0,
                "estimated_macs": 0,
                "estimated_flops": int(flops),
                "parameters": selector_params,
            }
        hidden = int(self.center_head.in_features)
        in_dim = int(feature_dim) + DUCA_ACTIONNESS_FEATURE_DIM + int(self.coarse_hidden_dim)
        macs = tokens * (in_dim * hidden + hidden * hidden + 6 * hidden)
        flops = 2 * macs + tokens * (6 * in_dim + 16 * hidden + 64)
        return {
            "head": "DUCASelectorMLP",
            "hidden_dim": hidden,
            "input_dim": in_dim,
            "coarse_hidden_dim": int(self.coarse_hidden_dim),
            "uses_coarse_hidden_features": bool(self.coarse_hidden_dim > 0),
            "estimated_macs": int(macs),
            "estimated_flops": int(flops),
            "parameters": selector_params,
        }

    def _estimate_budget_controller_profile(
        self,
        *,
        batch_size: int,
        temporal_len: int,
    ) -> Dict[str, Any]:
        params = _module_param_counts(self.budget_controller)
        if not self.dynamic_budget or self.budget_controller is None:
            return {
                "enabled": False,
                "policy": "fixed_budget",
                "estimated_macs": 0,
                "estimated_flops": 0,
                "parameters": params,
            }
        hidden = int(getattr(self.budget_controller, "hidden_dim", 0))
        blocks = int(getattr(self.budget_controller, "num_extra_blocks", 0))
        macs = int(batch_size) * (hidden * hidden + blocks * (2 * hidden * hidden + hidden))
        flops = 2 * macs + int(batch_size) * (int(temporal_len) * hidden + blocks * (16 * hidden + 20))
        return {
            "enabled": True,
            "policy": str(getattr(self.budget_controller, "policy_name", "prefix_marginal_utility_stop")),
            "num_extra_blocks": blocks,
            "estimated_macs": int(macs),
            "estimated_flops": int(flops),
            "parameters": params,
        }

    def compute_profile(
        self,
        dense_observations: torch.Tensor,
        *,
        external_cached_actionness: bool = False,
        external_actionness_source_name: Optional[str] = None,
        descriptor_profile: Optional[Mapping[str, Any]] = None,
        latency_ms: Optional[Mapping[str, Optional[float]]] = None,
    ) -> Dict[str, Any]:
        if dense_observations.ndim != 3:
            raise ValueError("dense_observations must be [B,T,C] for DUCA compute profiling")
        batch_size, temporal_len, feature_dim = [int(v) for v in dense_observations.shape]
        actionness = self._estimate_actionness_profile(
            batch_size=batch_size,
            temporal_len=temporal_len,
            feature_dim=feature_dim,
            external_cached=bool(external_cached_actionness),
            external_source_name=external_actionness_source_name,
        )
        selector = self._estimate_selector_profile(
            batch_size=batch_size,
            temporal_len=temporal_len,
            feature_dim=feature_dim,
        )
        budget_controller = self._estimate_budget_controller_profile(
            batch_size=batch_size,
            temporal_len=temporal_len,
        )
        descriptor = dict(descriptor_profile or {"estimated_macs": 0, "estimated_flops": 0})
        if self.acquisition_policy == "global_structured_topk":
            max_hole = temporal_len if self.max_unselected_hole is None else int(self.max_unselected_hole)
            dp_states = batch_size * temporal_len * min(self.budget, temporal_len) * (max_hole + 1)
            soft_coverage_macs = dp_states * (3 if self.training else 1)
            soft_coverage_flops = soft_coverage_macs * 8
            structured_complexity = "O(B*T*K*(G+1)) exact-K/max-gap dynamic program"
        elif self.acquisition_policy == "local_cell_deformation":
            soft_coverage_macs = batch_size * temporal_len
            soft_coverage_flops = soft_coverage_macs * 8
            structured_complexity = "O(B*T) one-categorical-choice-per-exact-uniform-cell"
        elif self.acquisition_policy in {
            "continuous_density_transport",
            "continuous_mixture_density_transport",
            "budget_calibrated_sampling_rate",
        }:
            soft_coverage_macs = batch_size * (4 * temporal_len + 3 * self.budget)
            soft_coverage_flops = soft_coverage_macs * 6
            if self.acquisition_policy == "continuous_mixture_density_transport":
                structured_complexity = "O(B*(C*T+K)) mixture inverse-CDF density transport"
            elif self.acquisition_policy == "budget_calibrated_sampling_rate":
                structured_complexity = "O(B*(T+K)) capped-rate calibration plus systematic sampling"
            else:
                structured_complexity = "O(B*(T+K)) inverse-CDF density transport"
        else:
            soft_coverage_macs = batch_size * temporal_len * temporal_len
            soft_coverage_flops = soft_coverage_macs * 8
            structured_complexity = "O(B*T^2) center-radius soft coverage"
        gather_flops = batch_size * min(self.budget, temporal_len) * feature_dim
        components = {
            "descriptor": descriptor,
            "actionness": actionness,
            "selector": selector,
            "budget_controller": budget_controller,
            "soft_coverage": {
                "estimated_macs": int(soft_coverage_macs),
                "estimated_flops": int(soft_coverage_flops),
                "complexity": structured_complexity,
            },
            "hard_decode": {
                "estimated_macs": 0,
                "estimated_flops": 0,
                "topk_elements": int(batch_size * temporal_len),
                "note": "top-k and Python interval fill are reported through latency rather than FLOP accounting",
            },
            "sparse_gather": {
                "estimated_macs": 0,
                "estimated_flops": int(gather_flops),
            },
        }
        total_macs = _sum_known(component.get("estimated_macs") for component in components.values())
        total_flops = _sum_known(component.get("estimated_flops") for component in components.values())
        param_counts = _module_param_counts(self)
        profile = {
            "pre_backbone_model": (
                "ExternalCachedActionness+DUCASelectorMLP"
                if external_cached_actionness
                else f"ZeroShotActionnessSource(mode={getattr(self.actionness_source, 'mode', 'unknown')})+DUCASelectorMLP"
            ),
            "input_shape": [batch_size, temporal_len, feature_dim],
            "budget_mode": self.budget_mode,
            "budget_max": int(self.budget),
            "estimated_macs": int(total_macs),
            "estimated_flops": int(total_flops),
            "estimated_flops_are_lower_bound": False,
            "complete_memory_accounting": False,
            "parameters": param_counts,
            "actionness": actionness,
            "components": components,
            "latency_ms": dict(latency_ms or {"enabled": False}),
            "flop_accounting": "static_estimate_for_selector_path_excludes_detector_backbone_and_external_cached_x3d_extraction",
        }
        self.last_compute_profile = profile
        return profile

    def forward_scores(
        self,
        dense_observations: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        actionness_logits: Optional[torch.Tensor] = None,
        p_action: Optional[torch.Tensor] = None,
        actionness_provenance: Optional[Mapping[str, Any]] = None,
        coarse_hidden_features: Optional[torch.Tensor] = None,
        coarse_policy_hidden_features: Optional[torch.Tensor] = None,
        coarse_hidden_kind: Optional[str] = None,
        policy_hidden_gradient_scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        if dense_observations.ndim != 3:
            raise ValueError(f"dense_observations must be [B,T,C], got {tuple(dense_observations.shape)}")
        if actionness_provenance is not None:
            validate_actionness_provenance(
                actionness_provenance,
                context="DUCA adapter actionness provenance",
            )
            if actionness_logits is not None and p_action is not None:
                temperature = actionness_provenance.get("calibration_temperature")
                bias = actionness_provenance.get("calibration_bias")
                if temperature is not None or bias is not None:
                    temperature = float(1.0 if temperature is None else temperature)
                    bias = float(0.0 if bias is None else bias)
                    if not math.isfinite(temperature) or temperature <= 0.0 or not math.isfinite(bias):
                        raise ValueError("actionness provenance contains invalid calibration parameters")
                    valid_for_check = _as_valid_mask(actionness_logits, valid_mask)
                    expected = torch.sigmoid((actionness_logits.float() + bias) / temperature)
                    supplied = p_action.to(device=expected.device, dtype=expected.dtype)
                    if supplied.shape != expected.shape or not torch.allclose(
                        supplied[valid_for_check], expected[valid_for_check], rtol=1e-5, atol=1e-6
                    ):
                        raise ValueError("p_action does not match the calibration declared by its provenance")
        source = self.actionness_source(
            dense_observations,
            logits=actionness_logits,
            valid_mask=valid_mask,
            p_action=p_action,
        )
        if actionness_provenance is not None:
            source["provenance"] = dict(actionness_provenance)
        valid = source["valid_mask"]
        transition_score = source.get("transition_score", source["uncertainty"])
        transition_score = transition_score.to(dense_observations.device, dense_observations.dtype).masked_fill(~valid, 0.0)
        actionness_aux = source["p_action"].to(dense_observations.device, dense_observations.dtype).masked_fill(~valid, 0.0)
        coarse_hidden = None
        has_coarse_hidden_features = False
        if coarse_hidden_features is not None:
            if coarse_hidden_features.ndim != 3:
                raise ValueError("coarse_hidden_features must be [B,T,D]")
            if coarse_hidden_features.shape[:2] != dense_observations.shape[:2]:
                raise ValueError("coarse_hidden_features must align with dense_observations [B,T]")
            if self.coarse_hidden_dim <= 0:
                raise ValueError("coarse_hidden_dim must be configured when coarse_hidden_features are provided")
            if int(coarse_hidden_features.shape[-1]) != int(self.coarse_hidden_dim):
                raise ValueError(
                    f"expected coarse_hidden_dim={self.coarse_hidden_dim}, got {coarse_hidden_features.shape[-1]}"
                )
            coarse_hidden = coarse_hidden_features.to(dense_observations.device, dense_observations.dtype)
            coarse_hidden = coarse_hidden.masked_fill(~valid[:, :, None], 0.0)
            has_coarse_hidden_features = True
        elif self.require_coarse_hidden_features:
            raise ValueError("DUCA final selector requires online coarse_hidden_features")
        elif self.coarse_hidden_dim > 0:
            coarse_hidden = dense_observations.new_zeros(
                dense_observations.shape[0],
                dense_observations.shape[1],
                int(self.coarse_hidden_dim),
            )
        coarse_policy_hidden = None
        if coarse_policy_hidden_features is not None:
            if coarse_hidden is None or coarse_hidden_features is None:
                raise ValueError("coarse_policy_hidden_features require coarse_hidden_features")
            if coarse_policy_hidden_features.shape != coarse_hidden_features.shape:
                raise ValueError(
                    "coarse_policy_hidden_features must match coarse_hidden_features [B,T,D]"
                )
            coarse_policy_hidden = coarse_policy_hidden_features.to(
                dense_observations.device,
                dense_observations.dtype,
            )
            coarse_policy_hidden = coarse_policy_hidden.masked_fill(
                ~valid[:, :, None],
                0.0,
            )
        transition_paths = None
        transition_center_scores = None
        burst_outputs = None
        density_mixture_outputs = None
        detector_contribution_logits = None
        active_policy_hidden_gradient_scale = self.policy_hidden_gradient_scale
        if policy_hidden_gradient_scale is not None:
            active_policy_hidden_gradient_scale = float(policy_hidden_gradient_scale)
            if not math.isfinite(active_policy_hidden_gradient_scale) or not 0.0 <= active_policy_hidden_gradient_scale <= 1.0:
                raise ValueError("policy_hidden_gradient_scale override must lie in [0,1]")
        if self.parameter_free_selector:
            transition_center_scores = actionness_aux
            if self.transition_objective == "boundary_burst":
                offset_logits = transition_center_scores.new_zeros(
                    (*transition_center_scores.shape, 2 * self.boundary_burst_radius + 1)
                )
                burst_outputs = build_boundary_burst_utility(
                    transition_center_scores,
                    offset_logits,
                    valid,
                    k=self.budget,
                    radius=self.boundary_burst_radius,
                    quota=self.boundary_burst_quota,
                    boundary_budget_fraction=self.boundary_burst_budget_fraction,
                    context_weight=self.boundary_burst_context_weight,
                    center_temperature=self.boundary_burst_center_temperature,
                    offset_temperature=self.boundary_burst_offset_temperature,
                    require_bilateral_offsets=self.boundary_burst_require_bilateral_offsets,
                )
                center_scores = burst_outputs["policy_utility"]
            else:
                center_scores = transition_center_scores
            selection_features = source["features"].float()
            radius = center_scores.new_zeros(center_scores.shape)
            start_logits = center_scores.new_zeros(center_scores.shape)
            end_logits = center_scores.new_zeros(center_scores.shape)
            context_logits = center_scores.new_zeros(center_scores.shape)
            utility_scores = center_scores
        elif self.selector_variant == "transition_only":
            if coarse_hidden_kind != ASFORMER_ENCODER_HIDDEN_KIND:
                raise ValueError(
                    "transition_only requires hidden_kind="
                    f"{ASFORMER_ENCODER_HIDDEN_KIND!r}, got {coarse_hidden_kind!r}"
                )
            if coarse_hidden is None or self.transition_scorer is None:
                raise RuntimeError("transition_only requires a shared transition scorer and coarse hidden state")
            transition_paths = transition_utility_paths(
                self.transition_scorer,
                source["logits"],
                coarse_hidden,
                valid,
                compute_auxiliary=self.training,
                policy_hidden=coarse_policy_hidden,
                policy_hidden_gradient_scale=active_policy_hidden_gradient_scale,
                auxiliary_hidden_gradient_scale=self.auxiliary_hidden_gradient_scale,
            )
            transition_center_scores = transition_paths["policy_scores"]
            if self.transition_objective == "boundary_burst":
                offset_logits = transition_paths.get("policy_offset_logits")
                if offset_logits is None:
                    raise RuntimeError("boundary_burst requires burst offset logits")
                burst_outputs = build_boundary_burst_utility(
                    transition_center_scores,
                    offset_logits,
                    valid,
                    k=self.budget,
                    radius=self.boundary_burst_radius,
                    quota=self.boundary_burst_quota,
                    boundary_budget_fraction=self.boundary_burst_budget_fraction,
                    context_weight=self.boundary_burst_context_weight,
                    center_temperature=self.boundary_burst_center_temperature,
                    offset_temperature=self.boundary_burst_offset_temperature,
                    require_bilateral_offsets=(
                        self.boundary_burst_require_bilateral_offsets
                    ),
                )
                center_scores = burst_outputs["policy_utility"]
            else:
                center_scores = transition_center_scores
            if self.acquisition_policy == "continuous_mixture_density_transport":
                if self.density_mixture_head is None:
                    raise RuntimeError("mixture density acquisition requires its density head")
                mixture_hidden_source = (
                    coarse_policy_hidden
                    if coarse_policy_hidden is not None
                    else coarse_hidden
                )
                mixture_hidden = mixture_hidden_source.detach() + active_policy_hidden_gradient_scale * (
                    mixture_hidden_source - mixture_hidden_source.detach()
                )
                density_mixture_outputs = self.density_mixture_head(
                    boundary_logits=transition_center_scores,
                    p_action=source["p_action"],
                    uncertainty=source["uncertainty"],
                    uncertainty_peak=source["uncertainty_peak"],
                    hidden=mixture_hidden,
                    valid_mask=valid,
                )
            if self.acquisition_policy in {
                "budget_calibrated_sampling_rate",
                "semantic_phase_sampling",
            }:
                if self.sampling_rate_utility_fusion is None:
                    raise RuntimeError("sampling-rate acquisition requires its utility fusion head")
                # A rate-only control must be a real ablation: do not merely
                # mask the contribution logits after constructing their head.
                # Otherwise detector loss can still update that head through
                # the zero-valued branch, which invalidates the comparison.
                if self.sampling_rate_utility_components != "none":
                    detector_contribution_logits = (
                        self.transition_scorer.detector_utility_logits(
                            transition_paths["policy_descriptors"]
                        ).masked_fill(~valid[:, :, None], 0.0)
                    )
                    utility_mask = detector_contribution_logits.new_tensor(
                        {
                            "cls": (1.0, 0.0),
                            "reg": (0.0, 1.0),
                            "both": (1.0, 1.0),
                        }[self.sampling_rate_utility_components]
                    )
                    rate_utility = self.sampling_rate_utility_fusion(
                        detector_contribution_logits * utility_mask
                    ).squeeze(-1)
                    center_scores = center_scores + rate_utility
            selection_features = transition_paths["transition_descriptors"]
            radius = center_scores.new_zeros(center_scores.shape)
            start_logits = center_scores.new_zeros(center_scores.shape)
            end_logits = center_scores.new_zeros(center_scores.shape)
            context_logits = center_scores.new_zeros(center_scores.shape)
            utility_scores = center_scores
        elif self.encoder is None:
            delta = source["delta_p_action"].to(dense_observations.device, dense_observations.dtype)
            eps = torch.finfo(delta.dtype).eps
            start_prob = delta.clamp(0.0, 1.0).clamp(eps, 1.0 - eps)
            end_prob = (-delta).clamp(0.0, 1.0).clamp(eps, 1.0 - eps)
            start_logits = torch.logit(start_prob)
            end_logits = torch.logit(end_prob)
            context_logits = torch.logit((1.0 - actionness_aux).clamp(eps, 1.0 - eps))
            boundary_prob = 0.5 * (start_prob + end_prob)
            utility_scores = transition_score + 0.5 * actionness_aux
            utility_prob = torch.sigmoid(utility_scores)
            center_scores = (
                self.transition_weight * transition_score
                + self.boundary_weight * boundary_prob
                + self.utility_weight * utility_prob
                + self.uncertainty_weight * source["uncertainty"]
                + self.actionness_weight * actionness_aux
            )
            radius = self.max_radius * torch.sigmoid(source["uncertainty"] * 4.0 - 2.0)
            selection_features = source["features"].float()
        else:
            if dense_observations.shape[-1] != self.feature_dim:
                raise ValueError(f"expected feature_dim={self.feature_dim}, got {dense_observations.shape[-1]}")
            feature_parts = [dense_observations.float(), source["features"].float()]
            if coarse_hidden is not None:
                feature_parts.append(coarse_hidden.float())
            browser_features = torch.cat(feature_parts, dim=-1)
            encoded = self.encoder(browser_features)
            selection_features = encoded
            center_base = torch.tanh(self.center_head(encoded).squeeze(-1))
            radius = (
                self.max_radius * torch.sigmoid(self.radius_head(encoded).squeeze(-1))
                if self.radius_head is not None
                else encoded.new_zeros(encoded.shape[:2])
            )
            start_logits = self.start_head(encoded).squeeze(-1)
            end_logits = self.end_head(encoded).squeeze(-1)
            context_logits = self.context_head(encoded).squeeze(-1)
            utility_scores = self.utility_head(encoded).squeeze(-1)
            boundary_prob = 0.5 * (torch.sigmoid(start_logits) + torch.sigmoid(end_logits))
            utility_prob = torch.sigmoid(utility_scores)
            center_scores = (
                center_base
                + self.transition_weight * transition_score
                + self.uncertainty_weight * source["uncertainty"]
                + self.utility_weight * utility_prob
                + self.boundary_weight * boundary_prob
                + self.actionness_weight * actionness_aux
            )
        center_scores = center_scores.masked_fill(~valid, _neg(center_scores.dtype))
        radius = radius.masked_fill(~valid, 0.0).clamp(0.0, float(self.max_radius))
        output = {
            "center_scores": center_scores,
            "scores": center_scores,
            "radius": radius,
            "start_logits": start_logits.masked_fill(~valid, 0.0),
            "end_logits": end_logits.masked_fill(~valid, 0.0),
            "context_logits": context_logits.masked_fill(~valid, 0.0),
            "boundary_logits": torch.maximum(start_logits, end_logits).masked_fill(~valid, 0.0),
            "utility_scores": utility_scores.masked_fill(~valid, 0.0),
            "p_action": source["p_action"],
            "uncertainty": source["uncertainty"],
            "entropy": source["entropy"],
            "delta_p_action": source["delta_p_action"],
            "abs_delta_p_action": source["abs_delta_p_action"],
            "uncertainty_peak": source["uncertainty_peak"],
            "transition_score": transition_score.masked_fill(~valid, 0.0),
            "actionness_logits": source["logits"] if actionness_logits is None else actionness_logits,
            "raw_actionness_logits": actionness_logits,
            "calibrated_actionness_logits": source["logits"],
            "selection_features": selection_features.masked_fill(~valid[:, :, None], 0.0),
            "coarse_hidden_features": None if coarse_hidden is None else coarse_hidden.masked_fill(~valid[:, :, None], 0.0),
            "coarse_policy_hidden_features": coarse_policy_hidden,
            "uses_coarse_hidden_features": bool(has_coarse_hidden_features),
            "coarse_hidden_kind": coarse_hidden_kind,
            "selector_variant": self.selector_variant,
            "parameter_free_selector": self.parameter_free_selector,
            "transition_objective": self.transition_objective,
            "boundary_burst_local_bilateral_utility_enabled": bool(
                self.transition_objective == "boundary_burst"
                and self.boundary_burst_require_bilateral_offsets
            ),
            "boundary_burst_global_mandatory_groups_enabled": bool(
                self.transition_objective == "boundary_burst"
                and self.boundary_burst_require_global_mandatory_groups
            ),
            "valid_mask": valid,
            "provenance": source["provenance"],
        }
        if transition_center_scores is not None:
            output["transition_center_scores"] = transition_center_scores
        if transition_paths is not None:
            output.update(
                {
                    "transition_descriptors": transition_paths["transition_descriptors"],
                    "transition_auxiliary_scores": transition_paths["auxiliary_scores"],
                    "transition_policy_scores": transition_paths["policy_scores"],
                    "transition_center_scores": transition_paths["policy_scores"],
                    "policy_hidden_gradient_scale": active_policy_hidden_gradient_scale,
                    "uses_absolute_hidden_features": bool(
                        density_mixture_outputs is not None
                    ),
                    "density_mixture_uses_absolute_hidden_features": bool(
                        density_mixture_outputs is not None
                    ),
                    "uses_raw_rgb_descriptors": False,
                    "legacy_direct_heads_enabled": False,
                }
            )
            if transition_paths.get("policy_offset_logits") is not None:
                output["burst_offset_logits"] = transition_paths[
                    "policy_offset_logits"
                ]
        if burst_outputs is not None:
            output.update(
                {
                    "boundary_burst_mass": burst_outputs["burst_mass"],
                    "boundary_burst_utility": burst_outputs["burst_utility"],
                    "boundary_burst_center_probabilities": burst_outputs["center_probabilities"],
                    "boundary_burst_offset_probabilities": burst_outputs["offset_probabilities"],
                    "boundary_burst_offset_inclusion": burst_outputs["offset_inclusion"],
                    "boundary_burst_effective_offset_quota": burst_outputs["effective_offset_quota"],
                    "boundary_burst_context_reference": burst_outputs["context_reference"],
                    "boundary_burst_bilateral_offset_feasible": burst_outputs["bilateral_offset_feasible"],
                    "boundary_burst_bilateral_offset_satisfied": burst_outputs["bilateral_offset_satisfied"],
                }
            )
        if density_mixture_outputs is not None:
            output.update(
                {
                    "density_component_logits": density_mixture_outputs[
                        "component_logits"
                    ],
                    "density_mixture_logits": density_mixture_outputs[
                        "mixture_logits"
                    ],
                    "density_mixture_gate_features": density_mixture_outputs[
                        "mixture_gate_features"
                    ],
                    "density_component_names": density_mixture_outputs[
                        "component_names"
                    ],
                }
            )
        if detector_contribution_logits is not None:
            output.update(
                {
                    "detector_contribution_logits": detector_contribution_logits,
                    "sampling_rate_utility_components": self.sampling_rate_utility_components,
                    "sampling_rate_utility_head_enabled": True,
                }
            )
        return output

    def _masked_gaussian_smooth(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        sigma = float(self.semantic_phase_sigma)
        radius = max(1, int(math.ceil(3.0 * sigma)))
        offsets = torch.arange(-radius, radius + 1, device=values.device, dtype=values.dtype)
        kernel = torch.exp(-0.5 * (offsets / sigma).pow(2))
        kernel = (kernel / kernel.sum().clamp_min(torch.finfo(values.dtype).eps)).view(1, 1, -1)
        valid = valid_mask.to(device=values.device, dtype=values.dtype)
        numer = F.conv1d(values.masked_fill(~valid_mask, 0.0)[:, None], kernel, padding=radius).squeeze(1)
        denom = F.conv1d(valid[:, None], kernel, padding=radius).squeeze(1).clamp_min(torch.finfo(values.dtype).eps)
        return (numer / denom).masked_fill(~valid_mask, 0.0)

    @staticmethod
    def _centered_derivative(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        derivative = torch.zeros_like(values)
        if values.shape[1] <= 1:
            return derivative
        valid = valid_mask.to(device=values.device, dtype=torch.bool)
        prev_values = torch.zeros_like(values)
        next_values = torch.zeros_like(values)
        prev_values[:, 1:] = values[:, :-1]
        next_values[:, :-1] = values[:, 1:]

        prev_valid = torch.zeros_like(valid)
        next_valid = torch.zeros_like(valid)
        prev_valid[:, 1:] = valid[:, :-1]
        next_valid[:, :-1] = valid[:, 1:]

        centered = 0.5 * (next_values - prev_values)
        forward = next_values - values
        backward = values - prev_values
        derivative = torch.where(
            prev_valid & next_valid,
            centered,
            torch.where(next_valid, forward, torch.where(prev_valid, backward, derivative)),
        )
        return derivative.masked_fill(~valid, 0.0)

    @staticmethod
    def _robust_q90_unit(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if int(valid.sum().item()) == 0:
            return values * 0.0
        active = values[valid]
        scale = torch.quantile(active.detach().abs().float(), 0.9).to(
            device=values.device,
            dtype=values.dtype,
        )
        scale = scale.clamp_min(torch.finfo(values.dtype).eps)
        return (values / scale).masked_fill(~valid, 0.0)

    @staticmethod
    def _ranked_pick_excluding(
        scores: torch.Tensor,
        *,
        valid_count: int,
        quota: int,
        selected: set[int],
    ) -> List[int]:
        if quota <= 0 or valid_count <= 0:
            return []
        positions = torch.arange(valid_count, device=scores.device, dtype=scores.dtype)
        tie_break = positions * scores.new_tensor(1.0e-7)
        order = torch.argsort(scores[:valid_count] - tie_break, descending=True)
        picked: List[int] = []
        for index in order.detach().cpu().tolist():
            index = int(index)
            if index in selected:
                continue
            selected.add(index)
            picked.append(index)
            if len(picked) >= quota:
                break
        return picked

    def _phase_reference_scores(self, values: torch.Tensor, valid_count: int, effective_k: int) -> torch.Tensor:
        if valid_count <= 0:
            return values * 0.0
        if effective_k <= 0:
            return values.new_zeros(valid_count)
        anchors = exact_uniform_positions(valid_count, effective_k, device=values.device).to(dtype=values.dtype)
        positions = torch.arange(valid_count, device=values.device, dtype=values.dtype)
        distance = (positions[:, None] - anchors[None, :]).abs().min(dim=1).values
        return -distance

    def _phase_scores_with_reference(
        self,
        learned: torch.Tensor,
        *,
        valid_count: int,
        effective_k: int,
        alpha: float,
    ) -> torch.Tensor:
        learned = learned[:valid_count]
        if alpha <= 0.0:
            return self._phase_reference_scores(learned, valid_count, effective_k)
        if alpha >= 1.0:
            return learned
        reference = self._phase_reference_scores(learned, valid_count, effective_k)
        return reference * float(1.0 - alpha) + learned * float(alpha)

    def _phase_soft_mass(
        self,
        component: torch.Tensor,
        *,
        valid_count: int,
        quota: int,
        alpha: float,
        effective_k: int,
    ) -> torch.Tensor:
        soft = component.new_zeros(component.shape[0])
        if quota <= 0 or valid_count <= 0:
            return soft
        logits = self._phase_scores_with_reference(
            component,
            valid_count=valid_count,
            effective_k=effective_k,
            alpha=alpha,
        )
        probs = F.softmax(logits / float(self.structured_temperature), dim=0) * float(quota)
        soft[:valid_count] = probs
        return soft

    def _phase_slot_assignment(
        self,
        aggregate_scores: torch.Tensor,
        selected_positions: torch.Tensor,
        *,
        valid_count: int,
        effective_k: int,
        temporal_len: int,
    ) -> torch.Tensor:
        slots = aggregate_scores.new_zeros((int(self.budget), temporal_len))
        if effective_k <= 0 or valid_count <= 0:
            return slots
        positions = torch.arange(valid_count, device=aggregate_scores.device, dtype=aggregate_scores.dtype)
        anchors = selected_positions[:effective_k].to(device=aggregate_scores.device, dtype=aggregate_scores.dtype)
        local_logits = aggregate_scores[:valid_count][None, :] - (
            positions[None, :] - anchors[:, None]
        ).abs() / float(self.structured_temperature)
        slots[:effective_k, :valid_count] = F.softmax(local_logits, dim=1)
        return slots

    def _decode_semantic_phase_sampling(
        self,
        actionness_logits: Optional[torch.Tensor],
        p_action: Optional[torch.Tensor],
        valid_mask: torch.Tensor,
        budgets: torch.Tensor,
        *,
        stable_selection: bool,
        policy_mix_alpha: float,
    ) -> Dict[str, Any]:
        if torch.any(budgets != int(self.budget)):
            raise ValueError("semantic_phase_sampling requires the configured fixed budget")
        if actionness_logits is None:
            if p_action is None:
                raise ValueError("semantic_phase_sampling requires ASFormer logits or p_action")
            eps = 1.0e-4
            actionness_logits = torch.logit(p_action.clamp(eps, 1.0 - eps))
        logits = actionness_logits.to(dtype=torch.float32).masked_fill(~valid_mask, 0.0)
        smooth = self._masked_gaussian_smooth(logits, valid_mask)
        derivative = self._centered_derivative(smooth, valid_mask)
        core = torch.sigmoid(smooth).masked_fill(~valid_mask, 0.0)
        onset = F.relu(derivative).masked_fill(~valid_mask, 0.0)
        offset = F.relu(-derivative).masked_fill(~valid_mask, 0.0)

        batch, temporal_len = logits.shape
        max_slots = int(self.budget)
        alpha = 0.0 if stable_selection else float(policy_mix_alpha)
        alpha = max(0.0, min(1.0, alpha))
        position_rows = []
        dense_masks = []
        selection_st_rows = []
        soft_rows = []
        slot_rows = []
        effective_rows = []
        policy_rows = []
        diagnostics = []

        for batch_idx in range(batch):
            valid_positions = torch.nonzero(valid_mask[batch_idx], as_tuple=False).flatten()
            valid_count = int(valid_positions.numel())
            expected = torch.arange(valid_count, device=valid_positions.device, dtype=valid_positions.dtype)
            if not torch.equal(valid_positions, expected):
                raise ValueError("semantic_phase_sampling requires a contiguous valid prefix")
            effective_k = min(int(budgets[batch_idx].item()), valid_count)
            row = torch.full((max_slots,), -1, dtype=torch.long, device=logits.device)
            dense_mask = torch.zeros(temporal_len, dtype=torch.bool, device=logits.device)
            if effective_k <= 0:
                position_rows.append(row)
                dense_masks.append(dense_mask)
                selection_st_rows.append(logits.new_zeros(temporal_len))
                soft_rows.append(logits.new_zeros(temporal_len))
                slot_rows.append(logits.new_zeros((max_slots, temporal_len)))
                effective_rows.append(effective_k)
                policy_rows.append(logits.new_zeros(temporal_len))
                diagnostics.append({"effective_k": 0, "backfill_count": 0})
                continue

            if alpha <= 0.0 or stable_selection:
                selected_tensor = exact_uniform_positions(
                    valid_count,
                    effective_k,
                    device=logits.device,
                )
                row[:effective_k] = selected_tensor
                dense_mask[selected_tensor] = True
                policy_row = logits.new_zeros(temporal_len)
                policy_row[:valid_count] = self._phase_reference_scores(
                    logits[batch_idx, :valid_count],
                    valid_count,
                    effective_k,
                )
                slots = logits.new_zeros((max_slots, temporal_len))
                if effective_k > 0:
                    slot_ids = torch.arange(effective_k, device=logits.device)
                    slots[slot_ids, selected_tensor] = 1.0
                soft = dense_mask.to(dtype=logits.dtype)
                gaps = selected_tensor[1:] - selected_tensor[:-1] if effective_k > 1 else selected_tensor.new_zeros(1)
                diagnostics.append(
                    {
                        "effective_k": int(effective_k),
                        "valid_count": int(valid_count),
                        "budget_scaffold": 0,
                        "budget_onset": 0,
                        "budget_offset": 0,
                        "budget_core": 0,
                        "backfill_count": 0,
                        "selected_unique_count": int(effective_k),
                        "uniform_reference": True,
                        "gap_mean": float(gaps.float().mean().detach().cpu().item()),
                        "gap_p95": float(torch.quantile(gaps.float(), 0.95).detach().cpu().item()),
                        "gap_max": int(gaps.max().detach().cpu().item()),
                    }
                )
                position_rows.append(row)
                dense_masks.append(dense_mask)
                selection_st_rows.append(soft)
                soft_rows.append(soft)
                slot_rows.append(slots)
                effective_rows.append(effective_k)
                policy_rows.append(policy_row)
                continue

            component_valid = torch.ones(valid_count, device=logits.device, dtype=torch.bool)
            core_row = self._robust_q90_unit(core[batch_idx, :valid_count], component_valid)
            onset_row = self._robust_q90_unit(onset[batch_idx, :valid_count], component_valid)
            offset_row = self._robust_q90_unit(offset[batch_idx, :valid_count], component_valid)

            quotas = {
                "scaffold": min(self.semantic_phase_budgets["scaffold"], effective_k),
                "onset": self.semantic_phase_budgets["onset"],
                "offset": self.semantic_phase_budgets["offset"],
                "core": self.semantic_phase_budgets["core"],
            }
            selected: set[int] = set()
            selected_by_group = {"scaffold": [], "onset": [], "offset": [], "core": [], "backfill": []}
            scaffold = exact_uniform_positions(
                valid_count,
                quotas["scaffold"],
                device=logits.device,
            ).detach().cpu().tolist()
            for index in scaffold:
                index = int(index)
                if index not in selected:
                    selected.add(index)
                    selected_by_group["scaffold"].append(index)
            remaining = effective_k - len(selected)
            for group_name, group_scores in (
                ("onset", onset_row),
                ("offset", offset_row),
                ("core", core_row),
            ):
                quota = min(int(quotas[group_name]), remaining)
                ranked = self._phase_scores_with_reference(
                    group_scores,
                    valid_count=valid_count,
                    effective_k=effective_k,
                    alpha=alpha,
                )
                picked = self._ranked_pick_excluding(
                    ranked,
                    valid_count=valid_count,
                    quota=quota,
                    selected=selected,
                )
                selected_by_group[group_name].extend(picked)
                remaining = effective_k - len(selected)
                if remaining <= 0:
                    break
            if remaining > 0:
                backfill_candidates = exact_uniform_positions(
                    valid_count,
                    effective_k,
                    device=logits.device,
                ).detach().cpu().tolist()
                backfill_candidates.extend(range(valid_count))
                for index in backfill_candidates:
                    index = int(index)
                    if index in selected:
                        continue
                    selected.add(index)
                    selected_by_group["backfill"].append(index)
                    if len(selected) >= effective_k:
                        break
            selected_sorted = sorted(selected)
            if len(selected_sorted) != effective_k:
                raise ValueError("semantic_phase_sampling failed to produce exact unique K")
            selected_tensor = torch.tensor(selected_sorted, device=logits.device, dtype=torch.long)
            row[:effective_k] = selected_tensor
            dense_mask[selected_tensor] = True

            aggregate = core_row + onset_row + offset_row
            policy_row = logits.new_zeros(temporal_len)
            policy_row[:valid_count] = self._phase_scores_with_reference(
                aggregate,
                valid_count=valid_count,
                effective_k=effective_k,
                alpha=alpha,
            )
            scaffold_soft = logits.new_zeros(temporal_len)
            if selected_by_group["scaffold"]:
                scaffold_idx = torch.tensor(selected_by_group["scaffold"], device=logits.device, dtype=torch.long)
                scaffold_soft[scaffold_idx] = 1.0
            soft = scaffold_soft
            soft[:valid_count] = soft[:valid_count] + self._phase_soft_mass(
                onset_row,
                valid_count=valid_count,
                quota=min(self.semantic_phase_budgets["onset"], effective_k),
                alpha=alpha,
                effective_k=effective_k,
            )
            soft[:valid_count] = soft[:valid_count] + self._phase_soft_mass(
                offset_row,
                valid_count=valid_count,
                quota=min(self.semantic_phase_budgets["offset"], effective_k),
                alpha=alpha,
                effective_k=effective_k,
            )
            soft[:valid_count] = soft[:valid_count] + self._phase_soft_mass(
                core_row,
                valid_count=valid_count,
                quota=min(self.semantic_phase_budgets["core"], effective_k),
                alpha=alpha,
                effective_k=effective_k,
            )
            soft = soft.clamp(max=1.0).masked_fill(~valid_mask[batch_idx], 0.0)
            selection_st = dense_mask.to(dtype=soft.dtype) + soft - soft.detach()
            slots = self._phase_slot_assignment(
                policy_row,
                selected_tensor,
                valid_count=valid_count,
                effective_k=effective_k,
                temporal_len=temporal_len,
            )

            gaps = selected_tensor[1:] - selected_tensor[:-1] if effective_k > 1 else selected_tensor.new_zeros(1)
            diagnostics.append(
                {
                    "effective_k": int(effective_k),
                    "valid_count": int(valid_count),
                    "budget_scaffold": int(len(selected_by_group["scaffold"])),
                    "budget_onset": int(len(selected_by_group["onset"])),
                    "budget_offset": int(len(selected_by_group["offset"])),
                    "budget_core": int(len(selected_by_group["core"])),
                    "backfill_count": int(len(selected_by_group["backfill"])),
                    "selected_by_group": {
                        name: list(indices)
                        for name, indices in selected_by_group.items()
                    },
                    "selected_unique_count": int(len(selected_sorted)),
                    "gap_mean": float(gaps.float().mean().detach().cpu().item()),
                    "gap_p95": float(torch.quantile(gaps.float(), 0.95).detach().cpu().item()),
                    "gap_max": int(gaps.max().detach().cpu().item()),
                }
            )
            position_rows.append(row)
            dense_masks.append(dense_mask)
            selection_st_rows.append(selection_st)
            soft_rows.append(soft)
            slot_rows.append(slots)
            effective_rows.append(effective_k)
            policy_rows.append(policy_row)

        effective = torch.tensor(effective_rows, dtype=torch.long, device=logits.device)
        return {
            "selected_positions": torch.stack(position_rows, dim=0),
            "selected_mask": torch.stack(dense_masks, dim=0),
            "selection_st": torch.stack(selection_st_rows, dim=0),
            "soft_coverage": torch.stack(soft_rows, dim=0).to(dtype=actionness_logits.dtype),
            "soft_slot_assignment": torch.stack(slot_rows, dim=0).to(dtype=actionness_logits.dtype),
            "effective_budget": effective,
            "detector_input_length": effective.clone(),
            "selected_centers": [[] for _ in range(batch)],
            "selected_radius": [[] for _ in range(batch)],
            "fill_strategy": ["semantic_phase_fixed_quota" for _ in range(batch)],
            "max_gap_repair": [{"enabled": False, "encoded_in_policy": False} for _ in range(batch)],
            "selection_path": (
                "semantic_phase_exact_uniform_reference"
                if bool(stable_selection or alpha <= 0.0)
                else "semantic_phase_fixed_budget_logits"
            ),
            "decode_policy_logits": torch.stack(policy_rows, dim=0).to(dtype=actionness_logits.dtype),
            "policy_mix_alpha": alpha,
            "semantic_phase_diagnostics": diagnostics,
            "semantic_phase_smoothed_logits": smooth.to(dtype=actionness_logits.dtype),
            "semantic_phase_onset_scores": onset.to(dtype=actionness_logits.dtype),
            "semantic_phase_offset_scores": offset.to(dtype=actionness_logits.dtype),
            "semantic_phase_core_scores": core.to(dtype=actionness_logits.dtype),
        }

    def _decode_global_structured(
        self,
        center_scores: torch.Tensor,
        valid_mask: torch.Tensor,
        budgets: torch.Tensor,
        *,
        stable_selection: bool,
        policy_mix_alpha: float,
        mandatory_center_scores: Optional[torch.Tensor] = None,
        mandatory_offset_inclusion: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        batch, temporal_len = center_scores.shape
        max_slots = int(self.budget)
        position_rows = []
        dense_masks = []
        selection_st_rows = []
        soft_rows = []
        slot_rows = []
        effective_rows = []
        policy_rows = []
        mandatory_rows = []
        mandatory_center_rows = []
        mandatory_group_counts = []
        for batch_idx in range(batch):
            valid_positions = torch.nonzero(valid_mask[batch_idx], as_tuple=False).flatten()
            valid_count = int(valid_positions.numel())
            expected = torch.arange(valid_count, device=valid_positions.device, dtype=valid_positions.dtype)
            if not torch.equal(valid_positions, expected):
                raise ValueError("global_structured_topk requires a contiguous valid prefix")
            effective_k = min(int(budgets[batch_idx].item()), valid_count)
            max_hole = valid_count if self.max_unselected_hole is None else int(self.max_unselected_hole)
            policy_scores = center_scores[batch_idx : batch_idx + 1, :valid_count].detach()
            if self.selector_variant == "transition_only":
                policy_scores = continuous_policy_logits(
                    policy_scores,
                    torch.ones_like(policy_scores, dtype=torch.bool),
                    k=effective_k,
                    alpha=float(policy_mix_alpha),
                )
            elif stable_selection:
                learned_scores = policy_scores
                reference_scores = exact_uniform_reference_scores(
                    policy_scores,
                    torch.ones_like(policy_scores, dtype=torch.bool),
                    effective_k,
                )
                policy_scores = reference_scores + learned_scores * 0.0
            policy_row = center_scores.new_zeros(temporal_len)
            policy_row[:valid_count] = policy_scores[0]
            policy_rows.append(policy_row)
            required_mask = torch.zeros_like(policy_scores, dtype=torch.bool)
            retained_center_mask = torch.zeros_like(policy_scores, dtype=torch.bool)
            retained_group_count = 0
            if (
                self.transition_objective == "boundary_burst"
                and self.boundary_burst_require_global_mandatory_groups
                and not stable_selection
                and mandatory_center_scores is not None
                and mandatory_offset_inclusion is not None
            ):
                max_mandatory = int(
                    math.floor(
                        effective_k
                        * self.boundary_burst_budget_fraction
                        * float(policy_mix_alpha)
                    )
                )
                mandatory = build_mandatory_bilateral_set(
                    mandatory_center_scores[
                        batch_idx : batch_idx + 1, :valid_count
                    ],
                    mandatory_offset_inclusion[
                        batch_idx : batch_idx + 1, :valid_count
                    ],
                    torch.ones_like(policy_scores, dtype=torch.bool),
                    radius=self.boundary_burst_radius,
                    quota=int(round(self.boundary_burst_quota)),
                    max_mandatory=max_mandatory,
                )
                required_mask = mandatory["mandatory_mask"]
                retained_center_mask = mandatory["retained_center_mask"]
                retained_group_count = int(
                    mandatory["retained_group_count"][0].item()
                )
            hard_structured = global_structured_topk(
                policy_scores,
                k=effective_k,
                max_unselected_hole=max_hole,
                required_mask=required_mask,
                temperature=self.structured_temperature,
                training=False,
            )
            row = torch.full((max_slots,), -1, dtype=torch.long, device=center_scores.device)
            row[:effective_k] = hard_structured.selected_positions[0]
            position_rows.append(row)
            dense_mask = torch.zeros(temporal_len, dtype=torch.bool, device=center_scores.device)
            dense_mask[:valid_count] = hard_structured.hard_occupancy[0].bool()
            dense_masks.append(dense_mask)
            mandatory_row = torch.zeros(
                temporal_len, dtype=torch.bool, device=center_scores.device
            )
            mandatory_row[:valid_count] = required_mask[0]
            mandatory_rows.append(mandatory_row)
            mandatory_center_row = torch.zeros_like(mandatory_row)
            mandatory_center_row[:valid_count] = retained_center_mask[0]
            mandatory_center_rows.append(mandatory_center_row)
            mandatory_group_counts.append(retained_group_count)
            if self.training:
                # The relaxed path must describe the same feasible family as
                # the hard path. Run it on the real valid prefix/effective K,
                # then pad inactive batch slots with exact zeros.
                if valid_count == 0:
                    soft = center_scores[batch_idx].float() * 0.0
                    slots = soft[None, :].expand(max_slots, -1) * 0.0
                    hard_dense = dense_mask.to(dtype=soft.dtype)
                    selection_st = hard_dense + soft - soft.detach()
                    selection_st_rows.append(selection_st)
                    soft_rows.append(soft)
                    slot_rows.append(slots)
                    effective_rows.append(effective_k)
                    continue
                surrogate_scores = center_scores[batch_idx : batch_idx + 1, :valid_count].float()
                surrogate_valid = torch.ones_like(surrogate_scores, dtype=torch.bool)
                if self.selector_variant == "transition_only":
                    surrogate_policy = continuous_policy_logits(
                        surrogate_scores,
                        surrogate_valid,
                        k=effective_k,
                        alpha=float(policy_mix_alpha),
                    )
                elif stable_selection:
                    reference_scores = exact_uniform_reference_scores(
                        surrogate_scores,
                        surrogate_valid,
                        effective_k,
                    )
                    surrogate_policy = reference_scores + surrogate_scores * 0.0
                else:
                    surrogate_policy = surrogate_scores
                surrogate = global_structured_topk(
                    surrogate_policy,
                    k=effective_k,
                    max_unselected_hole=max_hole,
                    required_mask=required_mask,
                    temperature=self.structured_temperature,
                    training=True,
                )
                slots = F.pad(
                    surrogate.soft_slot_assignment[0],
                    (0, temporal_len - valid_count, 0, max_slots - effective_k),
                )
                soft = F.pad(
                    surrogate.soft_occupancy[0],
                    (0, temporal_len - valid_count),
                )
                hard_dense = dense_mask.to(dtype=soft.dtype)
                selection_st = hard_dense + soft - soft.detach()
            else:
                soft = dense_mask.to(dtype=center_scores.dtype)
                selection_st = soft
                slots = center_scores.new_zeros((max_slots, temporal_len))
                if effective_k > 0:
                    slots[:effective_k].scatter_(1, row[:effective_k, None], 1.0)
            selection_st_rows.append(selection_st)
            soft_rows.append(soft)
            slot_rows.append(slots)
            effective_rows.append(effective_k)
        effective = torch.tensor(effective_rows, dtype=torch.long, device=center_scores.device)
        return {
            "selected_positions": torch.stack(position_rows, dim=0),
            "selected_mask": torch.stack(dense_masks, dim=0),
            "selection_st": torch.stack(selection_st_rows, dim=0),
            "soft_coverage": torch.stack(soft_rows, dim=0),
            "soft_slot_assignment": torch.stack(slot_rows, dim=0),
            "effective_budget": effective,
            "detector_input_length": effective.clone(),
            "selected_centers": [[] for _ in range(batch)],
            "selected_radius": [[] for _ in range(batch)],
            "fill_strategy": ["global_structured_map" for _ in range(batch)],
            "max_gap_repair": [{"enabled": False, "encoded_in_policy": True} for _ in range(batch)],
            "selection_path": (
                "parameter_free_transition_prior"
                if self.parameter_free_selector
                else
                "transition_uniform_reference"
                if self.selector_variant == "transition_only" and float(policy_mix_alpha) <= 0.0
                else "transition_learned"
                if self.selector_variant == "transition_only" and float(policy_mix_alpha) >= 1.0
                else "transition_continuous_homotopy"
                if self.selector_variant == "transition_only"
                else "stable_structured_reference"
                if stable_selection
                else "learned_global_structured"
            ),
            "decode_policy_logits": torch.stack(policy_rows, dim=0),
            "policy_mix_alpha": float(policy_mix_alpha),
            "mandatory_boundary_mask": torch.stack(mandatory_rows, dim=0),
            "mandatory_boundary_centers": torch.stack(
                mandatory_center_rows, dim=0
            ),
            "mandatory_boundary_group_count": mandatory_group_counts,
        }

    def _decode_continuous_density(
        self,
        center_scores: torch.Tensor,
        valid_mask: torch.Tensor,
        budgets: torch.Tensor,
        *,
        stable_selection: bool,
        policy_mix_alpha: float,
        component_logits: Optional[torch.Tensor] = None,
        component_mixture_logits: Optional[torch.Tensor] = None,
        component_names: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, Any]:
        if torch.any(budgets != int(self.budget)):
            raise ValueError("continuous density transport currently requires the configured fixed budget")
        alpha = 0.0 if stable_selection else float(policy_mix_alpha)
        decoded = continuous_density_transport(
            center_scores,
            valid_mask,
            k=int(self.budget),
            max_unselected_hole=self.max_unselected_hole,
            component_logits=component_logits,
            component_mixture_logits=component_mixture_logits,
            temperature=self.density_temperature,
            coverage_floor=self.density_coverage_floor,
            smoothing_kernel=self.density_smoothing_kernel,
            policy_alpha=alpha,
            training=self.training,
            force_exact_uniform=bool(stable_selection or alpha <= 0.0),
        )
        hard_cap_enabled = self.max_unselected_hole is not None
        observed_holes = decoded.observed_max_unselected_hole.detach().cpu().tolist()
        return {
            "selected_positions": decoded.selected_positions,
            "selected_mask": decoded.hard_occupancy.bool(),
            "selection_st": decoded.selection_st,
            "soft_coverage": decoded.soft_occupancy,
            "soft_slot_assignment": decoded.soft_slot_assignment,
            "effective_budget": decoded.effective_k,
            "detector_input_length": decoded.effective_k.clone(),
            "selected_centers": [[] for _ in range(int(center_scores.shape[0]))],
            "selected_radius": [[] for _ in range(int(center_scores.shape[0]))],
            "fill_strategy": [
                (
                    "inverse_cdf_density_transport_hard_max_projection"
                    if hard_cap_enabled
                    else "inverse_cdf_density_transport_unconstrained_projection"
                )
                for _ in range(int(center_scores.shape[0]))
            ],
            "max_gap_repair": [
                {
                    "enabled": hard_cap_enabled,
                    "parameter_free": True,
                    "role": (
                        "hard_max_gap_projection_ablation"
                        if hard_cap_enabled
                        else "observed_only_no_hard_max_gap"
                    ),
                    "configured_max_unselected_hole": self.max_unselected_hole,
                    "observed_max_unselected_hole": int(observed_holes[index]),
                }
                for index in range(int(center_scores.shape[0]))
            ],
            "selection_path": (
                "density_exact_uniform_reference"
                if bool(stable_selection or alpha <= 0.0)
                else (
                    "continuous_mixture_density_inverse_cdf"
                    if component_logits is not None
                    else "continuous_density_inverse_cdf"
                )
            ),
            "decode_policy_logits": center_scores * alpha,
            "policy_mix_alpha": alpha,
            "density_probabilities": decoded.density,
            "density_component_probabilities": decoded.component_densities,
            "density_mixture_weights": decoded.mixture_weights,
            "density_component_names": component_names,
            "density_cdf": decoded.cdf,
            "density_continuous_positions": decoded.continuous_positions,
            "density_projection_abs_displacement": (
                decoded.projection_abs_displacement
            ),
            "density_slot_mask": decoded.slot_mask,
            "density_observed_max_unselected_hole": decoded.observed_max_unselected_hole,
        }

    def _decode_budget_calibrated_sampling_rate(
        self,
        center_scores: torch.Tensor,
        valid_mask: torch.Tensor,
        budgets: torch.Tensor,
        *,
        stable_selection: bool,
        policy_mix_alpha: float,
    ) -> Dict[str, Any]:
        if torch.any(budgets != int(self.budget)):
            raise ValueError("budget-calibrated sampling rates require the configured fixed budget")
        alpha = 0.0 if stable_selection else float(policy_mix_alpha)
        decoded = budget_calibrated_sampling_rate(
            center_scores,
            valid_mask,
            k=int(self.budget),
            temperature=self.density_temperature,
            coverage_floor=self.density_coverage_floor,
            smoothing_kernel=self.density_smoothing_kernel,
            policy_alpha=alpha,
            training=self.training,
            force_exact_uniform=bool(stable_selection or alpha <= 0.0),
        )
        observed_holes = decoded.observed_max_unselected_hole.detach().cpu().tolist()
        return {
            "selected_positions": decoded.selected_positions,
            "selected_mask": decoded.hard_occupancy.bool(),
            "selection_st": decoded.selection_st,
            "soft_coverage": decoded.soft_occupancy,
            "soft_slot_assignment": decoded.soft_slot_assignment,
            "effective_budget": decoded.effective_k,
            "detector_input_length": decoded.effective_k.clone(),
            "selected_centers": [[] for _ in range(int(center_scores.shape[0]))],
            "selected_radius": [[] for _ in range(int(center_scores.shape[0]))],
            "fill_strategy": [
                "budget_calibrated_per_frame_rate_systematic_sampling"
                for _ in range(int(center_scores.shape[0]))
            ],
            "max_gap_repair": [
                {
                    "enabled": False,
                    "parameter_free": True,
                    "role": "observed_only_no_hard_max_gap",
                    "configured_max_unselected_hole": None,
                    "observed_max_unselected_hole": int(observed_holes[index]),
                }
                for index in range(int(center_scores.shape[0]))
            ],
            "selection_path": (
                "sampling_rate_exact_uniform_reference"
                if bool(stable_selection or alpha <= 0.0)
                else "budget_calibrated_sampling_rate_systematic"
            ),
            "decode_policy_logits": center_scores * alpha,
            "policy_mix_alpha": alpha,
            "sampling_rates": decoded.sampling_rates,
            "sampling_density": decoded.sampling_density,
            "sampling_cumulative_rates": decoded.cumulative_rates,
            "sampling_continuous_positions": decoded.continuous_positions,
            "sampling_slot_mask": decoded.slot_mask,
            "sampling_calibration_residual": decoded.calibration_residual,
            "sampling_observed_max_unselected_hole": decoded.observed_max_unselected_hole,
        }

    def _decode_local_cell(
        self,
        center_scores: torch.Tensor,
        valid_mask: torch.Tensor,
        budgets: torch.Tensor,
        *,
        stable_selection: bool,
    ) -> Dict[str, Any]:
        batch, temporal_len = center_scores.shape
        max_slots = int(self.budget)
        position_rows = []
        dense_masks = []
        selection_st_rows = []
        soft_rows = []
        slot_rows = []
        effective_rows = []
        policy_rows = []
        anchor_rows = []
        cell_start_rows = []
        cell_end_rows = []
        max_hole_rows = []
        force_uniform = bool(stable_selection or self.local_cell_force_exact_uniform)
        for batch_idx in range(batch):
            valid_positions = torch.nonzero(valid_mask[batch_idx], as_tuple=False).flatten()
            valid_count = int(valid_positions.numel())
            if valid_count <= 0:
                raise ValueError("local_cell_deformation requires at least one valid observation")
            expected = torch.arange(valid_count, device=valid_positions.device, dtype=valid_positions.dtype)
            if not torch.equal(valid_positions, expected):
                raise ValueError("local_cell_deformation requires a contiguous valid prefix")
            effective_k = min(int(budgets[batch_idx].item()), valid_count)
            policy_scores = center_scores[batch_idx : batch_idx + 1, :valid_count]
            decoded = local_cell_deformation(
                policy_scores,
                k=effective_k,
                temperature=self.structured_temperature,
                training=self.training,
                force_exact_uniform=force_uniform,
            )
            if self.max_unselected_hole is not None and decoded.max_unselected_hole > self.max_unselected_hole:
                raise ValueError(
                    "local-cell coverage is infeasible under configured max_unselected_hole: "
                    f"required={decoded.max_unselected_hole}, configured={self.max_unselected_hole}"
                )

            row = torch.full((max_slots,), -1, dtype=torch.long, device=center_scores.device)
            row[:effective_k] = decoded.selected_positions[0]
            position_rows.append(row)
            dense_mask = torch.zeros(temporal_len, dtype=torch.bool, device=center_scores.device)
            dense_mask[:valid_count] = decoded.hard_occupancy[0].bool()
            dense_masks.append(dense_mask)
            selection_st_rows.append(F.pad(decoded.selection_st[0], (0, temporal_len - valid_count)))
            soft_rows.append(F.pad(decoded.soft_occupancy[0], (0, temporal_len - valid_count)))
            slot_rows.append(
                F.pad(
                    decoded.soft_slot_assignment[0],
                    (0, temporal_len - valid_count, 0, max_slots - effective_k),
                )
            )
            policy_row = center_scores.new_zeros(temporal_len)
            policy_row[:valid_count] = policy_scores[0]
            policy_rows.append(policy_row)
            for values, rows in (
                (decoded.anchor_positions, anchor_rows),
                (decoded.cell_starts, cell_start_rows),
                (decoded.cell_ends, cell_end_rows),
            ):
                padded = torch.full((max_slots,), -1, dtype=torch.long, device=center_scores.device)
                padded[:effective_k] = values
                rows.append(padded)
            effective_rows.append(effective_k)
            max_hole_rows.append(int(decoded.max_unselected_hole))

        effective = torch.tensor(effective_rows, dtype=torch.long, device=center_scores.device)
        return {
            "selected_positions": torch.stack(position_rows, dim=0),
            "selected_mask": torch.stack(dense_masks, dim=0),
            "selection_st": torch.stack(selection_st_rows, dim=0),
            "soft_coverage": torch.stack(soft_rows, dim=0),
            "soft_slot_assignment": torch.stack(slot_rows, dim=0),
            "effective_budget": effective,
            "detector_input_length": effective.clone(),
            "selected_centers": [[] for _ in range(batch)],
            "selected_radius": [[] for _ in range(batch)],
            "fill_strategy": ["one_frame_per_exact_uniform_cell" for _ in range(batch)],
            "max_gap_repair": [
                {
                    "enabled": False,
                    "encoded_in_policy": True,
                    "theoretical_max_unselected_hole": max_hole_rows[index],
                }
                for index in range(batch)
            ],
            "selection_path": "local_cell_exact_uniform" if force_uniform else "local_cell_transition_deformation",
            "decode_policy_logits": torch.stack(policy_rows, dim=0),
            "policy_mix_alpha": 0.0 if force_uniform else 1.0,
            "local_cell_anchor_positions": torch.stack(anchor_rows, dim=0),
            "local_cell_starts": torch.stack(cell_start_rows, dim=0),
            "local_cell_ends": torch.stack(cell_end_rows, dim=0),
            "local_cell_max_unselected_hole": max_hole_rows,
        }

    def acquire(
        self,
        dense_observations: torch.Tensor,
        budget: Optional[TensorLikeBudget] = None,
        valid_mask: Optional[torch.Tensor] = None,
        actionness_logits: Optional[torch.Tensor] = None,
        p_action: Optional[torch.Tensor] = None,
        actionness_provenance: Optional[Mapping[str, Any]] = None,
        coarse_hidden_features: Optional[torch.Tensor] = None,
        coarse_policy_hidden_features: Optional[torch.Tensor] = None,
        coarse_hidden_kind: Optional[str] = None,
        compute_profile_context: Optional[Mapping[str, Any]] = None,
        stable_selection: bool = False,
        policy_mix_alpha: float = 1.0,
        policy_hidden_gradient_scale: Optional[float] = None,
    ) -> Tuple[SparseTemporalGrid, Dict[str, Any]]:
        profile_enabled = bool(self.profile_runtime)
        sync_enabled = profile_enabled and bool(self.profile_sync_cuda)
        adapter_start = _sync_profile_clock(dense_observations, enabled=sync_enabled) if profile_enabled else None
        score_start = _sync_profile_clock(dense_observations, enabled=sync_enabled) if profile_enabled else None
        scores = self.forward_scores(
            dense_observations=dense_observations,
            valid_mask=valid_mask,
            actionness_logits=actionness_logits,
            p_action=p_action,
            actionness_provenance=actionness_provenance,
            coarse_hidden_features=coarse_hidden_features,
            coarse_policy_hidden_features=coarse_policy_hidden_features,
            coarse_hidden_kind=coarse_hidden_kind,
            policy_hidden_gradient_scale=policy_hidden_gradient_scale,
        )
        score_ms = _elapsed_ms(score_start, dense_observations, enabled=sync_enabled)
        budget_decision: Optional[DynamicBudgetDecision] = None
        budget_controller_ms: Optional[float] = None
        if self.dynamic_budget:
            if budget is not None and not self.allow_external_budget_override:
                raise ValueError("external budget override is forbidden for dynamic_must DUCA acquisition")
            if self.budget_controller is None:
                raise ValueError("dynamic_must requires a budget_controller")
            budget_controller_start = (
                _sync_profile_clock(dense_observations, enabled=sync_enabled) if profile_enabled else None
            )
            budget_decision = self.budget_controller(
                scores["selection_features"],
                scores["center_scores"],
                scores["valid_mask"],
            )
            budget_controller_ms = _elapsed_ms(budget_controller_start, dense_observations, enabled=sync_enabled)
            budget_decision.validate(batch_size=dense_observations.shape[0])
            if budget is None:
                budgets = budget_decision.budget_hard.to(device=dense_observations.device, dtype=torch.long)
            else:
                budgets = _budget_tensor(budget, dense_observations.shape[0], dense_observations.device)
        else:
            if budget is None:
                budget = self.budget
            budgets = _budget_tensor(budget, dense_observations.shape[0], dense_observations.device)
        if torch.any(budgets > self.budget):
            raise ValueError(
                f"budget override exceeds hard cap: requested={budgets.detach().cpu().tolist()} "
                f"hard_cap={self.budget}"
            )
        decode_start = _sync_profile_clock(dense_observations, enabled=sync_enabled) if profile_enabled else None
        if self.acquisition_policy == "global_structured_topk":
            decoded = self._decode_global_structured(
                scores["center_scores"],
                scores["valid_mask"],
                budgets,
                stable_selection=bool(stable_selection),
                policy_mix_alpha=float(policy_mix_alpha),
                mandatory_center_scores=scores.get("transition_center_scores"),
                mandatory_offset_inclusion=scores.get(
                    "boundary_burst_offset_inclusion"
                ),
            )
        elif self.acquisition_policy in {
            "continuous_density_transport",
            "continuous_mixture_density_transport",
        }:
            decoded = self._decode_continuous_density(
                scores["center_scores"],
                scores["valid_mask"],
                budgets,
                stable_selection=bool(stable_selection),
                policy_mix_alpha=float(policy_mix_alpha),
                component_logits=scores.get("density_component_logits"),
                component_mixture_logits=scores.get("density_mixture_logits"),
                component_names=scores.get("density_component_names"),
            )
        elif self.acquisition_policy == "budget_calibrated_sampling_rate":
            decoded = self._decode_budget_calibrated_sampling_rate(
                scores["center_scores"],
                scores["valid_mask"],
                budgets,
                stable_selection=bool(stable_selection),
                policy_mix_alpha=float(policy_mix_alpha),
            )
        elif self.acquisition_policy == "semantic_phase_sampling":
            decoded = self._decode_semantic_phase_sampling(
                scores.get("actionness_logits"),
                scores.get("p_action"),
                scores["valid_mask"],
                budgets,
                stable_selection=bool(stable_selection),
                policy_mix_alpha=float(policy_mix_alpha),
            )
        elif self.acquisition_policy == "local_cell_deformation":
            decoded = self._decode_local_cell(
                scores["center_scores"],
                scores["valid_mask"],
                budgets,
                stable_selection=bool(stable_selection),
            )
        else:
            decoded = budgeted_center_radius_decode(
                center_scores=scores["center_scores"],
                radius=scores["radius"],
                budget=budgets,
                valid_mask=scores["valid_mask"],
                max_radius=self.max_radius,
                output_slots=int(self.budget),
                max_unselected_hole=self.max_unselected_hole,
                hard_max_gap_repair=self.hard_max_gap_repair,
                fail_on_infeasible_max_gap=self.fail_on_infeasible_max_gap,
            )
        decode_ms = _elapsed_ms(decode_start, dense_observations, enabled=sync_enabled)
        valid_len = scores["valid_mask"].long().sum(dim=1)
        effective_budget = decoded["effective_budget"].to(device=dense_observations.device, dtype=torch.long)
        hard_unique_k = decoded["detector_input_length"].to(device=dense_observations.device, dtype=torch.long)
        padded_detector_k = torch.full_like(hard_unique_k, int(self.budget))
        cost_ledger = {
            "unit": DEFAULT_BUDGET_UNIT,
            "hard_requested_k": [int(item) for item in budgets.detach().cpu().tolist()],
            "hard_effective_k": [int(item) for item in effective_budget.detach().cpu().tolist()],
            "hard_unique_k": [int(item) for item in hard_unique_k.detach().cpu().tolist()],
            "padded_detector_k": [int(item) for item in padded_detector_k.detach().cpu().tolist()],
            "backbone_input_k": [int(item) for item in padded_detector_k.detach().cpu().tolist()],
            "dynamic_compute_realized": False,
            "dynamic_compute_blocker": "detector_tensor_is_padded_to_budget_max" if self.dynamic_budget else None,
        }
        if budget_decision is not None:
            cost_ledger.update(
                {
                    "soft_expected_k": [
                        float(item) for item in budget_decision.soft_expected_k.detach().cpu().tolist()
                    ],
                    "st_budget_k": [float(item) for item in budget_decision.st_budget_k.detach().cpu().tolist()],
                    "dual_target_unit": str(budget_decision.dual_target_unit),
                }
            )
        grid = SparseTemporalGrid(
            selected_positions=decoded["selected_positions"],
            selected_mask=decoded["selected_mask"],
            original_length=int(dense_observations.shape[1]),
            valid_len=valid_len,
            budget=int(self.budget),
            requested_budget=budgets,
            effective_budget=effective_budget,
            detector_input_length=decoded["detector_input_length"],
            metadata={
                "selected_centers": decoded["selected_centers"],
                "selected_radius": decoded["selected_radius"],
                "fill_strategy": decoded["fill_strategy"],
                "decoder": self.acquisition_policy,
                "selection_scope": "full_window_non_streaming",
                "radius_is_metadata": self.acquisition_policy == "legacy_center_radius",
                "budget_is_dynamic": bool(self.dynamic_budget),
                "budget_policy": (
                    getattr(budget_decision, "policy_name", "fixed_budget") if budget_decision is not None else "fixed_budget"
                ),
                "budget_max": int(self.budget),
                "budget_min": int(self.budget_min),
                "budget_multiple": int(self.budget_multiple),
                "budget_target": float(self.target_budget),
                "predicted_budget": budgets.detach().cpu().tolist(),
                "detector_physical_input_length": int(self.budget),
                "max_unselected_hole": self.max_unselected_hole,
                "hard_max_gap_repair": bool(self.hard_max_gap_repair),
                "max_gap_repair": decoded.get("max_gap_repair", []),
                "mandatory_boundary_count": (
                    None
                    if decoded.get("mandatory_boundary_mask") is None
                    else decoded["mandatory_boundary_mask"].sum(dim=1).detach().cpu().tolist()
                ),
                "mandatory_boundary_group_count": decoded.get(
                    "mandatory_boundary_group_count"
                ),
                "boundary_burst_local_bilateral_utility_enabled": bool(
                    self.transition_objective == "boundary_burst"
                    and self.boundary_burst_require_bilateral_offsets
                ),
                "boundary_burst_global_mandatory_groups_enabled": bool(
                    self.transition_objective == "boundary_burst"
                    and self.boundary_burst_require_global_mandatory_groups
                ),
                "local_cell_anchor_positions": (
                    None
                    if decoded.get("local_cell_anchor_positions") is None
                    else decoded["local_cell_anchor_positions"].detach().cpu().tolist()
                ),
                "local_cell_starts": (
                    None
                    if decoded.get("local_cell_starts") is None
                    else decoded["local_cell_starts"].detach().cpu().tolist()
                ),
                "local_cell_ends": (
                    None
                    if decoded.get("local_cell_ends") is None
                    else decoded["local_cell_ends"].detach().cpu().tolist()
                ),
                "local_cell_max_unselected_hole": decoded.get("local_cell_max_unselected_hole"),
                "density_observed_max_unselected_hole": (
                    None
                    if decoded.get("density_observed_max_unselected_hole") is None
                    else decoded["density_observed_max_unselected_hole"]
                    .detach()
                    .cpu()
                    .tolist()
                ),
                "density_component_names": decoded.get("density_component_names"),
                "sampling_rate_observed_max_unselected_hole": (
                    None
                    if decoded.get("sampling_observed_max_unselected_hole") is None
                    else decoded["sampling_observed_max_unselected_hole"]
                    .detach()
                    .cpu()
                    .tolist()
                ),
                "sampling_rate_calibration_residual": (
                    None
                    if decoded.get("sampling_calibration_residual") is None
                    else decoded["sampling_calibration_residual"]
                    .detach()
                    .cpu()
                    .tolist()
                ),
                "semantic_phase_diagnostics": decoded.get(
                    "semantic_phase_diagnostics"
                ),
                "semantic_phase_budgets": (
                    dict(self.semantic_phase_budgets)
                    if self.acquisition_policy == "semantic_phase_sampling"
                    else None
                ),
                "semantic_phase_sigma": (
                    float(self.semantic_phase_sigma)
                    if self.acquisition_policy == "semantic_phase_sampling"
                    else None
                ),
                "cost_ledger": cost_ledger,
            },
        ).validate()
        if torch.any(grid.selected_count > int(self.budget)):
            raise RuntimeError("DUCA dynamic acquisition selected more observations than the hard cap")
        soft_coverage_start = _sync_profile_clock(dense_observations, enabled=sync_enabled) if profile_enabled else None
        if self.acquisition_policy in {
            "global_structured_topk",
            "local_cell_deformation",
            "continuous_density_transport",
            "continuous_mixture_density_transport",
            "budget_calibrated_sampling_rate",
            "semantic_phase_sampling",
        }:
            soft_coverage = decoded["soft_coverage"]
            selection_st = decoded["selection_st"]
        else:
            soft_coverage = soft_center_radius_coverage(
                center_scores=scores["center_scores"],
                radius=scores["radius"],
                valid_mask=scores["valid_mask"],
                budget=budgets,
                max_radius=self.max_radius,
            )
            hard_union = grid.selected_mask.to(dtype=scores["center_scores"].dtype)
            selection_st = hard_union + soft_coverage - soft_coverage.detach()
        soft_coverage_ms = _elapsed_ms(soft_coverage_start, dense_observations, enabled=sync_enabled)
        selected_indices = grid.selected_positions
        adapter_total_ms = _elapsed_ms(adapter_start, dense_observations, enabled=sync_enabled)
        profile_context = dict(compute_profile_context or {})
        latency_ms = {
            "enabled": profile_enabled,
            "adapter_total_ms": adapter_total_ms,
            "forward_scores_ms": score_ms,
            "budget_controller_ms": budget_controller_ms,
            "hard_decode_ms": decode_ms,
            "soft_coverage_ms": soft_coverage_ms,
        }
        compute_profile = self.compute_profile(
            dense_observations,
            external_cached_actionness=bool(profile_context.get("external_cached_actionness", False)),
            external_actionness_source_name=profile_context.get("external_actionness_source_name"),
            descriptor_profile=profile_context.get("descriptor_profile"),
            latency_ms=latency_ms,
        )
        scores.update(
            {
                "selected_mask_st": selection_st,
                "selected_indices_st": selected_indices,
                "soft_coverage": soft_coverage,
                "structured_soft_slot_assignment": decoded.get("soft_slot_assignment"),
                "selection_path": decoded.get("selection_path", "legacy_center_radius"),
                "decode_policy_logits": decoded.get("decode_policy_logits"),
                "policy_mix_alpha": float(decoded.get("policy_mix_alpha", policy_mix_alpha)),
                "mandatory_boundary_mask": decoded.get("mandatory_boundary_mask"),
                "mandatory_boundary_centers": decoded.get(
                    "mandatory_boundary_centers"
                ),
                "mandatory_boundary_group_count": decoded.get(
                    "mandatory_boundary_group_count"
                ),
                "local_cell_anchor_positions": decoded.get("local_cell_anchor_positions"),
                "local_cell_starts": decoded.get("local_cell_starts"),
                "local_cell_ends": decoded.get("local_cell_ends"),
                "local_cell_max_unselected_hole": decoded.get("local_cell_max_unselected_hole"),
                "density_probabilities": decoded.get("density_probabilities"),
                "density_component_probabilities": decoded.get(
                    "density_component_probabilities"
                ),
                "density_mixture_weights": decoded.get(
                    "density_mixture_weights"
                ),
                "density_component_names": decoded.get(
                    "density_component_names"
                ),
                "density_cdf": decoded.get("density_cdf"),
                "density_continuous_positions": decoded.get(
                    "density_continuous_positions"
                ),
                "density_projection_abs_displacement": decoded.get(
                    "density_projection_abs_displacement"
                ),
                "density_slot_mask": decoded.get("density_slot_mask"),
                "density_observed_max_unselected_hole": decoded.get(
                    "density_observed_max_unselected_hole"
                ),
                "sampling_rates": decoded.get("sampling_rates"),
                "sampling_density": decoded.get("sampling_density"),
                "sampling_cumulative_rates": decoded.get("sampling_cumulative_rates"),
                "sampling_continuous_positions": decoded.get("sampling_continuous_positions"),
                "sampling_slot_mask": decoded.get("sampling_slot_mask"),
                "sampling_calibration_residual": decoded.get("sampling_calibration_residual"),
                "sampling_observed_max_unselected_hole": decoded.get(
                    "sampling_observed_max_unselected_hole"
                ),
                "semantic_phase_diagnostics": decoded.get(
                    "semantic_phase_diagnostics"
                ),
                "semantic_phase_smoothed_logits": decoded.get(
                    "semantic_phase_smoothed_logits"
                ),
                "semantic_phase_onset_scores": decoded.get(
                    "semantic_phase_onset_scores"
                ),
                "semantic_phase_offset_scores": decoded.get(
                    "semantic_phase_offset_scores"
                ),
                "semantic_phase_core_scores": decoded.get(
                    "semantic_phase_core_scores"
                ),
                "detector_grid_positions": decoded.get(
                    "local_cell_anchor_positions",
                    decoded["selected_positions"],
                ),
                "decode_metadata": grid.metadata,
                "budget_decision": budget_decision,
                "dynamic_budget": bool(self.dynamic_budget),
                "budget_mode": self.budget_mode,
                "requested_budget": budgets,
                "effective_budget": effective_budget,
                "max_unselected_hole": self.max_unselected_hole,
                "structured_temperature": self.structured_temperature,
                "budget_metrics": {
                    "budget_mean": float(grid.selected_count.float().mean().detach().cpu().item()),
                    "budget_max": int(self.budget),
                    "budget_target": float(self.target_budget),
                    "budget_policy": grid.metadata["budget_policy"],
                    "max_unselected_hole": self.max_unselected_hole,
                },
                "compute_profile": compute_profile,
            }
        )
        return grid, scores

    def forward_acquire(
        self,
        dense_observations: torch.Tensor,
        budget: Optional[TensorLikeBudget] = None,
        valid_mask: Optional[torch.Tensor] = None,
        actionness_logits: Optional[torch.Tensor] = None,
        p_action: Optional[torch.Tensor] = None,
        coarse_hidden_features: Optional[torch.Tensor] = None,
        coarse_hidden_kind: Optional[str] = None,
        return_audit: bool = False,
    ) -> Dict[str, Any]:
        grid, scores = self.acquire(
            dense_observations=dense_observations,
            budget=budget,
            valid_mask=valid_mask,
            actionness_logits=actionness_logits,
            p_action=p_action,
            coarse_hidden_features=coarse_hidden_features,
            coarse_hidden_kind=coarse_hidden_kind,
        )
        gathered = gather_selected_observations(dense_observations, grid.selected_positions, grid.selected_mask)
        structured_assignment = scores.get("structured_soft_slot_assignment")
        if structured_assignment is not None:
            if dense_observations.ndim < 3:
                raise ValueError("structured detector bridge expects dense observations shaped [B,T,...]")
            expected = (
                int(dense_observations.shape[0]),
                int(grid.selected_positions.shape[1]),
                int(dense_observations.shape[1]),
            )
            if tuple(structured_assignment.shape) != expected:
                raise ValueError(
                    "structured detector bridge assignment must match [B,K,T]: "
                    f"expected {expected}, got {tuple(structured_assignment.shape)}"
                )
            flat_dense = dense_observations.reshape(dense_observations.shape[0], dense_observations.shape[1], -1)
            soft_gathered = torch.einsum(
                "bkt,btd->bkd",
                structured_assignment.to(device=flat_dense.device, dtype=flat_dense.dtype),
                flat_dense,
            ).reshape_as(gathered["observations"])
            slot_mask = grid.selected_positions >= 0
            slot = slot_mask.to(device=soft_gathered.device, dtype=soft_gathered.dtype)
            slot = slot.reshape(slot.shape + (1,) * (soft_gathered.ndim - 2))
            detector_input = gathered["observations"] + (soft_gathered - soft_gathered.detach()) * slot
            st_weights = slot_mask.to(device=dense_observations.device, dtype=dense_observations.dtype)
        else:
            st_weights = torch.gather(scores["selected_mask_st"], 1, grid.selected_positions.clamp_min(0))
            view_shape = (st_weights.shape[0], st_weights.shape[1]) + (1,) * (gathered["observations"].ndim - 2)
            detector_input = gathered["observations"] * st_weights.view(view_shape).to(gathered["observations"].dtype)
        out: Dict[str, Any] = {
            "grid": grid,
            "sparse_grid": grid,
            "detector_input": detector_input,
            "hard_detector_input": gathered["observations"],
            "detector_input_st_weights": st_weights,
            "detector_input_mask": gathered["mask"],
            "selected_positions": grid.selected_positions,
            "selected_mask": grid.selected_mask,
            "detector_input_length": grid.detector_input_length,
        }
        out.update(scores)
        if return_audit:
            out["audit"] = make_audit_record(grid, uses_teacher=False, mode="acquire")
        return out

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.forward_acquire(*args, **kwargs)


def hard_topk_st(
    scores: torch.Tensor,
    budget: Optional[TensorLikeBudget] = None,
    valid_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    k: Optional[TensorLikeBudget] = None,
    return_aux: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
    """Hard top-k forward with straight-through softmax surrogate backward."""

    if budget is None:
        if k is None:
            raise ValueError("budget or k must be provided")
        budget = k
    elif k is not None:
        raise ValueError("provide only one of budget or k")
    valid = _as_valid_mask(scores, valid_mask)
    budgets = _budget_tensor(budget, scores.shape[0], scores.device)
    valid_counts = valid.long().sum(dim=1)
    effective_budgets = torch.minimum(budgets, valid_counts)
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")

    masked_scores = scores.masked_fill(~valid, _neg(scores.dtype))
    max_budget = int(effective_budgets.max().item())
    topk_idx = torch.topk(masked_scores, k=max_budget, dim=1, largest=True, sorted=False).indices
    rank = torch.arange(max_budget, device=scores.device)[None, :]
    keep = rank < effective_budgets[:, None]
    hard = torch.zeros_like(scores)
    hard.scatter_(1, topk_idx, keep.to(dtype=scores.dtype))
    hard = hard.masked_fill(~valid, 0.0)

    soft = F.softmax(masked_scores / float(temperature), dim=1) * effective_budgets.to(scores.dtype)[:, None]
    soft = soft.masked_fill(~valid, 0.0)
    st_mask = hard + soft - soft.detach()
    sorted_idx = topk_idx.masked_fill(~keep, scores.shape[1]).sort(dim=1).values
    sorted_idx = sorted_idx.masked_fill(sorted_idx == scores.shape[1], -1)
    if return_aux:
        return st_mask, sorted_idx, {
            "hard_mask": hard,
            "soft_mask": soft,
            "budget": budgets,
            "effective_budget": effective_budgets,
            "valid_count": valid_counts,
        }
    return st_mask, sorted_idx


def soft_center_radius_coverage(
    center_scores: torch.Tensor,
    radius: torch.Tensor,
    budget: TensorLikeBudget,
    valid_mask: Optional[torch.Tensor] = None,
    max_radius: int = 16,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Differentiable coverage surrogate aligned to center-radius decoding."""

    valid = _as_valid_mask(center_scores, valid_mask)
    budgets = _budget_tensor(budget, center_scores.shape[0], center_scores.device)
    valid_counts = valid.long().sum(dim=1)
    effective_budget = torch.minimum(budgets, valid_counts).to(center_scores.dtype)
    masked_scores = center_scores.masked_fill(~valid, _neg(center_scores.dtype))
    center_mass = F.softmax(masked_scores / float(temperature), dim=1) * effective_budget[:, None]
    positions = torch.arange(center_scores.shape[1], device=center_scores.device, dtype=center_scores.dtype)
    distance = (positions[None, :, None] - positions[None, None, :]).abs()
    sigma = radius.to(center_scores.dtype).clamp(0.0, float(max_radius))[:, None, :] + 1.0
    kernel = torch.exp(-0.5 * (distance / sigma).pow(2))
    raw_coverage = torch.bmm(kernel, center_mass[:, :, None]).squeeze(-1).masked_fill(~valid, 0.0)
    scale = raw_coverage.detach().amax(dim=1, keepdim=True).clamp_min(1e-6)
    coverage = (raw_coverage / scale).masked_fill(~valid, 0.0)
    return coverage


def temporal_max_gap_hole_loss(
    selection_mass: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    max_unselected_hole: int,
    min_window_mass: float = 1.0,
) -> torch.Tensor:
    """Penalize local windows that could decode into large temporal holes."""

    if selection_mass.ndim != 2:
        raise ValueError(f"selection_mass must be [B,T], got {tuple(selection_mass.shape)}")
    max_hole = int(max_unselected_hole)
    if max_hole <= 0:
        return selection_mass.float().new_zeros(())
    valid = _as_valid_mask(selection_mass, valid_mask)
    mass = selection_mass.to(dtype=torch.float32).clamp_min(0.0).masked_fill(~valid, 0.0)
    valid_float = valid.to(dtype=mass.dtype)
    window = max_hole + 1
    valid_counts = valid_float.sum(dim=1)
    penalties: List[torch.Tensor] = []
    short = valid_counts <= float(window)
    if bool(short.any().item()):
        total_mass = mass.sum(dim=1)
        penalties.append(F.relu(float(min_window_mass) - total_mass[short]).pow(2))
    if selection_mass.shape[1] >= window:
        window_mass = F.avg_pool1d(mass[:, None, :], kernel_size=window, stride=1)[:, 0, :] * float(window)
        window_valid = F.avg_pool1d(valid_float[:, None, :], kernel_size=window, stride=1)[:, 0, :] * float(window)
        full = window_valid >= float(window)
        if bool(full.any().item()):
            penalties.append(F.relu(float(min_window_mass) - window_mass[full]).pow(2))
    if not penalties:
        return selection_mass.float().new_zeros(())
    return torch.cat([item.reshape(-1) for item in penalties], dim=0).mean()


def _unselected_hole_runs(selected: set[int], valid_positions: List[int]) -> List[Tuple[int, int, int]]:
    runs: List[Tuple[int, int, int]] = []
    start: Optional[int] = None
    previous = -1
    for pos in valid_positions:
        pos = int(pos)
        if pos in selected:
            if start is not None:
                runs.append((int(start), int(previous), int(previous - start + 1)))
                start = None
        elif start is None:
            start = pos
        previous = pos
    if start is not None:
        runs.append((int(start), int(previous), int(previous - start + 1)))
    return runs


def _max_unselected_hole(selected: set[int], valid_positions: List[int]) -> int:
    return max((length for _start, _end, length in _unselected_hole_runs(selected, valid_positions)), default=0)


def _minimum_selection_for_max_hole(valid_count: int, max_unselected_hole: int) -> int:
    if valid_count <= 0:
        return 0
    max_hole = int(max_unselected_hole)
    if max_hole < 0:
        raise ValueError("max_unselected_hole must be non-negative")
    if int(valid_count) <= max_hole:
        return 1
    numerator = max(0, int(valid_count) - max_hole)
    return max(1, int((numerator + max_hole) // (max_hole + 1)))


def _max_gap_scaffold_positions(valid_positions: List[int], max_unselected_hole: int) -> List[int]:
    """Return a minimal deterministic scaffold that satisfies the hard max-gap."""

    if not valid_positions:
        return []
    max_hole = int(max_unselected_hole)
    if max_hole < 0:
        raise ValueError("max_unselected_hole must be non-negative")
    valid = [int(pos) for pos in valid_positions]
    if len(valid) <= max_hole:
        return [valid[-1]]
    step = max_hole + 1
    ranks = list(range(max_hole, len(valid), step))
    if not ranks or ranks[-1] != len(valid) - 1:
        ranks.append(len(valid) - 1)
    scaffold = [valid[int(rank)] for rank in ranks]
    return sorted(set(scaffold))


def _repair_selected_max_unselected_hole(
    selected_positions: List[int],
    score_values: torch.Tensor,
    valid_positions: List[int],
    *,
    budget: int,
    max_unselected_hole: int,
) -> Tuple[List[int], Dict[str, Any]]:
    max_hole = int(max_unselected_hole)
    original_selected = {int(pos) for pos in selected_positions}
    if len(original_selected) != int(budget):
        raise ValueError("hard max-gap repair expects a strict-budget selection")
    minimum_required = _minimum_selection_for_max_hole(len(valid_positions), max_hole)
    metadata: Dict[str, Any] = {
        "enabled": True,
        "mode": "scaffold_first",
        "max_unselected_hole": max_hole,
        "minimum_required_budget": int(minimum_required),
        "requested_budget": int(budget),
        "feasible": int(budget) >= int(minimum_required),
        "repair_count": 0,
        "scaffold_count": 0,
        "remaining_budget_after_scaffold": max(0, int(budget)),
        "max_unselected_hole_before": int(_max_unselected_hole(original_selected, valid_positions)),
        "max_unselected_hole_after": None,
    }
    if not metadata["feasible"]:
        metadata["max_unselected_hole_after"] = metadata["max_unselected_hole_before"]
        return sorted(original_selected), metadata
    if metadata["max_unselected_hole_before"] <= max_hole:
        metadata["max_unselected_hole_after"] = metadata["max_unselected_hole_before"]
        metadata["satisfied"] = True
        return sorted(original_selected), metadata

    def score(pos: int) -> float:
        return float(score_values[int(pos)].detach().cpu().item())

    scaffold = _max_gap_scaffold_positions(valid_positions, max_hole)
    selected = {int(pos) for pos in scaffold}
    metadata["scaffold_count"] = int(len(selected))
    metadata["remaining_budget_after_scaffold"] = max(0, int(budget) - int(len(selected)))
    ranked_remaining = sorted(
        (int(pos) for pos in valid_positions if int(pos) not in selected),
        key=lambda pos: (score(pos), -int(pos)),
        reverse=True,
    )
    for pos in ranked_remaining:
        if len(selected) >= int(budget):
            break
        selected.add(int(pos))
    metadata["repair_count"] = int(len(selected.difference(original_selected)))
    metadata["max_unselected_hole_after"] = int(_max_unselected_hole(selected, valid_positions))
    metadata["satisfied"] = metadata["max_unselected_hole_after"] <= max_hole
    return sorted(selected), metadata


def budgeted_center_radius_decode(
    center_scores: Optional[torch.Tensor] = None,
    radius: Optional[torch.Tensor] = None,
    budget: Optional[TensorLikeBudget] = None,
    valid_mask: Optional[torch.Tensor] = None,
    centers: Optional[torch.Tensor] = None,
    radii: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    dense_positions: Optional[torch.Tensor] = None,
    dense_len: Optional[int] = None,
    max_radius: int = 16,
    candidate_multiplier: float = 2.0,
    output_slots: Optional[int] = None,
    max_unselected_hole: Optional[int] = None,
    hard_max_gap_repair: bool = True,
    fail_on_infeasible_max_gap: bool = True,
) -> Dict[str, Any]:
    """Decode center/radius decisions into detector-consumed positions.

    The output budget is measured only in final `selected_positions`. Center and
    radius are recorded as metadata and never counted as detector observations.
    """

    if center_scores is None:
        center_scores = scores
    if center_scores is None:
        raise ValueError("center_scores or scores must be provided")
    if center_scores.ndim != 2:
        raise ValueError("center_scores must be [B,T]")
    if budget is None:
        raise ValueError("budget must be provided")
    if radius is not None and radii is not None:
        raise ValueError("provide only one of radius or radii")
    if radius is None:
        radius = radii
    valid = _as_valid_mask(center_scores, valid_mask)
    budgets = _budget_tensor(budget, center_scores.shape[0], center_scores.device)
    valid_counts = valid.long().sum(dim=1)
    effective_budgets = torch.minimum(budgets, valid_counts)
    batch_size, temporal_len = center_scores.shape
    if dense_positions is None:
        if dense_len is None:
            dense_positions = torch.arange(temporal_len, device=center_scores.device).view(1, -1).expand(batch_size, -1)
        else:
            dense_positions = torch.linspace(0, int(dense_len) - 1, steps=temporal_len, device=center_scores.device)
            dense_positions = dense_positions.round().long().view(1, -1).expand(batch_size, -1)
    if dense_positions.shape != center_scores.shape:
        raise ValueError("dense_positions must be [B,T]")
    expected_positions = torch.arange(center_scores.shape[1], device=center_scores.device).view(1, -1).expand_as(center_scores)
    if not torch.equal(dense_positions.to(center_scores.device, dtype=torch.long), expected_positions):
        raise ValueError("dense_positions must be contiguous arange original-time indices in this decoder")
    if radius is None:
        radius = torch.zeros_like(center_scores)
    if radius.ndim == 1:
        radius = radius[:, None].expand_as(center_scores)
    if radius.shape != center_scores.shape:
        raise ValueError("radius must be [B,T] or [B]")
    radius = radius.to(center_scores.device, dtype=center_scores.dtype).clamp(0.0, float(max_radius))

    candidate_positions = dense_positions
    if centers is not None:
        if centers.shape != center_scores.shape:
            raise ValueError("centers must match center_scores when provided")
        candidate_positions = centers.to(center_scores.device).round().long()
        max_position = int(dense_positions.max().item())
        candidate_positions = candidate_positions.clamp(0, max_position)

    max_out = int(effective_budgets.max().item())
    if output_slots is not None:
        max_out = int(output_slots)
        if max_out <= 0:
            raise ValueError("output_slots must be positive when provided")
        if torch.any(effective_budgets > max_out):
            raise ValueError("output_slots cannot be smaller than any effective budget")
    rows: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    center_rows: List[torch.Tensor] = []
    radius_rows: List[torch.Tensor] = []
    fill_strategies: List[str] = []
    max_gap_repairs: List[Dict[str, Any]] = []
    masked_scores = center_scores.masked_fill(~valid, _neg(center_scores.dtype))

    for bidx in range(batch_size):
        target = int(effective_budgets[bidx].item())
        order_len = min(int(valid_counts[bidx].item()), max(target, int(target * float(candidate_multiplier))))
        order = torch.topk(masked_scores[bidx], k=order_len, largest=True, sorted=True).indices.tolist()
        selected: Dict[int, float] = {}
        selected_centers: List[int] = []
        selected_radius: List[int] = []
        valid_set = set(torch.nonzero(valid[bidx], as_tuple=False).flatten().tolist())

        for candidate_idx in order:
            if len(selected) >= target:
                break
            if candidate_idx not in valid_set:
                continue
            center_position = int(candidate_positions[bidx, candidate_idx].item())
            rad = int(torch.round(radius[bidx, candidate_idx]).item())
            left = max(0, center_position - rad)
            right = min(temporal_len - 1, center_position + rad)
            interval = [pos for pos in range(left, right + 1) if pos in valid_set and pos not in selected]
            if not interval:
                continue
            interval = sorted(
                interval,
                key=lambda pos: (
                    float(masked_scores[bidx, pos].item()),
                    -abs(pos - center_position),
                    -pos,
                ),
                reverse=True,
            )
            for pos in interval:
                if len(selected) >= target:
                    break
                selected[pos] = float(masked_scores[bidx, candidate_idx].item())
            selected_centers.append(center_position)
            selected_radius.append(rad)

        if len(selected) < target:
            fill_strategies.append("score_residual_fill")
            for pos in torch.argsort(masked_scores[bidx], descending=True).tolist():
                if len(selected) >= target:
                    break
                if pos in valid_set and pos not in selected:
                    selected[int(pos)] = float(masked_scores[bidx, pos].item())
        else:
            fill_strategies.append("center_radius_union")

        selected_positions = sorted(selected.keys())
        if len(selected_positions) != target:
            raise ValueError("decoder failed to produce a valid strict-budget selection")
        repair_meta: Dict[str, Any] = {"enabled": False}
        if hard_max_gap_repair and max_unselected_hole not in (None, 0):
            selected_positions, repair_meta = _repair_selected_max_unselected_hole(
                selected_positions,
                masked_scores[bidx],
                sorted(valid_set),
                budget=target,
                max_unselected_hole=int(max_unselected_hole),
            )
            if not repair_meta.get("feasible", True) and fail_on_infeasible_max_gap:
                raise ValueError(
                    "max_unselected_hole is infeasible for this valid length and budget: "
                    f"valid_count={len(valid_set)} budget={target} "
                    f"max_unselected_hole={int(max_unselected_hole)} "
                    f"minimum_required_budget={repair_meta.get('minimum_required_budget')}"
                )
            if repair_meta.get("feasible", True) and not repair_meta.get("satisfied", False) and fail_on_infeasible_max_gap:
                raise RuntimeError(
                    "hard max-gap repair failed to satisfy max_unselected_hole: "
                    f"{repair_meta}"
                )
            if int(repair_meta.get("repair_count", 0)) > 0:
                fill_strategies[-1] = f"{fill_strategies[-1]}+max_gap_repair"
        max_gap_repairs.append(repair_meta)
        row = torch.full((max_out,), -1, dtype=torch.long, device=center_scores.device)
        row[: len(selected_positions)] = torch.tensor(selected_positions, dtype=torch.long, device=center_scores.device)
        rows.append(row)
        dense_mask = torch.zeros((temporal_len,), dtype=torch.bool, device=center_scores.device)
        if selected_positions:
            dense_mask[torch.tensor(selected_positions, dtype=torch.long, device=center_scores.device)] = True
        masks.append(dense_mask)
        center_row = torch.full((max_out,), -1, dtype=torch.long, device=center_scores.device)
        radius_row = torch.full((max_out,), -1, dtype=torch.long, device=center_scores.device)
        usable = min(len(selected_centers), max_out)
        if usable:
            center_row[:usable] = torch.tensor(selected_centers[:usable], dtype=torch.long, device=center_scores.device)
            radius_row[:usable] = torch.tensor(selected_radius[:usable], dtype=torch.long, device=center_scores.device)
        center_rows.append(center_row)
        radius_rows.append(radius_row)

    selected_positions = torch.stack(rows, dim=0)
    selected_mask = torch.stack(masks, dim=0)
    selected_centers = torch.stack(center_rows, dim=0)
    selected_radius = torch.stack(radius_rows, dim=0)
    detector_input_length = selected_mask.long().sum(dim=1)
    return {
        "selected_positions": selected_positions,
        "positions": selected_positions,
        "selected_mask": selected_mask,
        "selected_centers": selected_centers,
        "selected_radius": selected_radius,
        "detector_input_length": detector_input_length,
        "fill_strategy": fill_strategies,
        "max_gap_repair": max_gap_repairs,
        "budget": budgets,
        "effective_budget": effective_budgets,
    }


def gather_selected_observations(
    x: torch.Tensor,
    selected_positions: torch.Tensor,
    selected_mask: Optional[torch.Tensor] = None,
    time_dim: int = 1,
    pad_value: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Gather detector input only from original-time selected positions."""

    if x.ndim < 3:
        raise ValueError("x must be at least [B,T,C] or [B,C,T]")
    if selected_positions.ndim != 2:
        raise ValueError("selected_positions must be [B,K]")
    if selected_positions.shape[0] != x.shape[0]:
        raise ValueError("selected_positions batch size must match x")
    if time_dim < 0:
        time_dim = x.ndim + time_dim
    if time_dim <= 0 or time_dim >= x.ndim:
        raise ValueError("time_dim must refer to a non-batch dimension")
    length = x.shape[time_dim]
    pos = selected_positions.to(device=x.device, dtype=torch.long)
    valid_pos = pos >= 0
    if torch.any(pos[valid_pos] >= length):
        raise ValueError("selected_positions contain out-of-range indices for x")
    if selected_mask is not None:
        if selected_mask.shape != (x.shape[0], length):
            raise ValueError("selected_mask must be dense [B,T] matching x time dimension")
        for bidx in range(x.shape[0]):
            mask_pos = torch.nonzero(selected_mask[bidx].to(device=x.device).bool(), as_tuple=False).flatten()
            row_pos = pos[bidx][pos[bidx] >= 0]
            if not torch.equal(mask_pos.cpu(), row_pos.cpu()):
                raise ValueError("selected_mask does not match selected_positions")

    moved = x.movedim(time_dim, 1)
    gather_idx = pos.clamp_min(0)
    expand_shape = (gather_idx.shape[0], gather_idx.shape[1]) + moved.shape[2:]
    gather_idx = gather_idx.view(gather_idx.shape[0], gather_idx.shape[1], *([1] * (moved.ndim - 2))).expand(expand_shape)
    gathered = moved.gather(dim=1, index=gather_idx)
    gathered = gathered.masked_fill(~valid_pos.view(valid_pos.shape[0], valid_pos.shape[1], *([1] * (gathered.ndim - 2))), pad_value)
    if time_dim != 1:
        gathered = gathered.movedim(1, time_dim)
    return {"observations": gathered, "sparse_observations": gathered, "features": gathered, "mask": valid_pos}


class DucaOnlineSparseDetectorWrapper(nn.Module):
    """Online acquisition wrapper that inserts DUCA before a sparse-capable detector.

    This is the minimal deployable contract: the adapter makes a hard online
    acquisition decision, and the wrapped detector receives only the gathered
    detector-consumed temporal observations.
    """

    def __init__(
        self,
        detector: nn.Module,
        adapter: Optional[DucaAcquisitionAdapter] = None,
        feature_dim: Optional[int] = None,
        budget: int = 384,
        max_radius: int = 16,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.adapter = adapter or DucaAcquisitionAdapter(feature_dim=feature_dim, budget=budget, max_radius=max_radius)
        self.budget = int(budget)

    def forward(
        self,
        batch: Optional[Mapping[str, Any]] = None,
        dense_observations: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        train_mode = self.training if mode is None else str(mode).lower() in {"loss", "train", "training"}
        default_budget = None if bool(getattr(self.adapter, "dynamic_budget", False)) else self.budget
        common = {
            "detector": self.detector,
            "adapter": self.adapter,
            "batch": batch,
            "dense_observations": dense_observations,
            "budget": kwargs.pop("budget", default_budget),
            "valid_mask": kwargs.pop("valid_mask", None),
            "actionness_logits": kwargs.pop("actionness_logits", None),
            "p_action": kwargs.pop("p_action", None),
        }
        if train_mode:
            return duca_forward_train(
                **common,
                teacher_utility=kwargs.pop("teacher_utility", None),
                loss_weights=kwargs.pop("loss_weights", None),
            )
        return duca_forward_test(**common)

    def forward_sparse(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.forward(*args, **kwargs)

    def forward_acquire(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.adapter.forward_acquire(*args, **kwargs)


def duca_forward_train(
    detector: Optional[nn.Module] = None,
    adapter: Optional[DucaAcquisitionAdapter] = None,
    batch: Optional[Mapping[str, Any]] = None,
    dense_observations: Optional[torch.Tensor] = None,
    budget: Optional[TensorLikeBudget] = None,
    valid_mask: Optional[torch.Tensor] = None,
    teacher_utility: Optional[torch.Tensor] = None,
    actionness_logits: Optional[torch.Tensor] = None,
    p_action: Optional[torch.Tensor] = None,
    loss_weights: Optional[Mapping[str, float]] = None,
    acquisition: Optional[DucaAcquisitionAdapter] = None,
) -> Dict[str, Any]:
    """Hard-forward train path. Teacher utility is train-loss-only."""

    adapter = adapter or acquisition
    if adapter is None:
        raise ValueError("adapter or acquisition must be provided")
    batch_dict = dict(batch or {})
    observations = dense_observations if dense_observations is not None else _get_observations(batch_dict)
    valid = valid_mask if valid_mask is not None else batch_dict.get("valid_mask")
    teacher = teacher_utility if teacher_utility is not None else batch_dict.get("teacher_utility")
    out = adapter.forward_acquire(
        dense_observations=observations,
        budget=budget,
        valid_mask=valid,
        actionness_logits=actionness_logits if actionness_logits is not None else batch_dict.get("actionness_logits"),
        p_action=p_action if p_action is not None else batch_dict.get("p_action"),
        return_audit=True,
    )
    detector_output = None
    detector_loss = None
    if detector is not None:
        detector_batch = _sanitize_detector_batch(
            batch_dict,
            forbidden_keys=FORBIDDEN_DECISION_KEYS - {"gt_segments", "gt_labels"},
        )
        detector_output = _call_detector(detector, out["detector_input"], out["grid"], detector_batch, train=True)
        detector_loss = _extract_detector_loss(detector_output, observations.device)
    losses = duca_losses(
        scores=out,
        teacher_utility=teacher,
        boundary_target=batch_dict.get("boundary_target"),
        action_target=batch_dict.get("action_target"),
        detector_loss=detector_loss,
        loss_weights=loss_weights,
    )
    out["detector_output"] = detector_output
    out["losses"] = losses
    out["audit"] = make_audit_record(
        out["grid"],
        uses_teacher=False,
        mode="train_forward",
        extra={"uses_teacher_for_decision": False, "uses_teacher_for_train_loss": teacher is not None},
    )
    return out


def duca_forward_test(
    detector: Optional[nn.Module] = None,
    adapter: Optional[DucaAcquisitionAdapter] = None,
    batch: Optional[Mapping[str, Any]] = None,
    dense_observations: Optional[torch.Tensor] = None,
    budget: Optional[TensorLikeBudget] = None,
    valid_mask: Optional[torch.Tensor] = None,
    actionness_logits: Optional[torch.Tensor] = None,
    p_action: Optional[torch.Tensor] = None,
    acquisition: Optional[DucaAcquisitionAdapter] = None,
) -> Dict[str, Any]:
    """Teacher-free hard-forward inference path."""

    batch_dict = dict(batch or {})
    _assert_no_forbidden_payload(batch_dict, FORBIDDEN_DECISION_KEYS)
    adapter = adapter or acquisition
    if adapter is None:
        raise ValueError("adapter or acquisition must be provided")
    observations = dense_observations if dense_observations is not None else _get_observations(batch_dict)
    valid = valid_mask if valid_mask is not None else batch_dict.get("valid_mask")
    out = adapter.forward_acquire(
        dense_observations=observations,
        budget=budget,
        valid_mask=valid,
        actionness_logits=actionness_logits if actionness_logits is not None else batch_dict.get("actionness_logits"),
        p_action=p_action if p_action is not None else batch_dict.get("p_action"),
        return_audit=True,
    )
    detector_output = None
    if detector is not None:
        detector_output = _call_detector(detector, out["detector_input"], out["grid"], batch_dict, train=False)
    out["detector_output"] = detector_output
    out["audit"] = make_audit_record(out["grid"], uses_teacher=False, mode="test_forward")
    return out


def _target_distribution_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != target.shape or logits.shape != valid_mask.shape:
        raise ValueError("distribution logits, target, and valid_mask must have identical [B,T] shapes")
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    work = logits.float()
    target = target.to(device=logits.device, dtype=torch.float32).clamp_min(0.0).masked_fill(~valid, 0.0)
    eps = torch.finfo(work.dtype).eps
    target_mass = target.sum(dim=1)
    active = target_mass > eps
    if not bool(active.any().item()):
        return work.new_zeros(())
    target_dist = target / target_mass.clamp_min(eps)[:, None]
    masked_logits = work.masked_fill(~valid, _neg(work.dtype))
    log_probs = F.log_softmax(masked_logits, dim=1)
    return -(target_dist * log_probs).sum(dim=1)[active].mean()


def duca_losses(
    scores: Union[Mapping[str, Any], torch.Tensor],
    selected_mask_st: Optional[torch.Tensor] = None,
    budget: Optional[TensorLikeBudget] = None,
    valid_mask: Optional[torch.Tensor] = None,
    teacher_utility: Optional[torch.Tensor] = None,
    boundary_target: Optional[torch.Tensor] = None,
    start_target: Optional[torch.Tensor] = None,
    end_target: Optional[torch.Tensor] = None,
    context_target: Optional[torch.Tensor] = None,
    action_target: Optional[torch.Tensor] = None,
    transition_target: Optional[torch.Tensor] = None,
    detector_loss: Optional[torch.Tensor] = None,
    utility_gain: Optional[torch.Tensor] = None,
    utility_risk: Optional[torch.Tensor] = None,
    detector_utility_target: Optional[torch.Tensor] = None,
    boundary_utility_proxy_target: Optional[torch.Tensor] = None,
    radius: Optional[torch.Tensor] = None,
    p_action: Optional[torch.Tensor] = None,
    uncertainty: Optional[torch.Tensor] = None,
    actionness_logits: Optional[torch.Tensor] = None,
    max_unselected_hole: Optional[int] = None,
    max_gap_loss_min_window_mass: float = 1.0,
    max_gap_loss_source: str = "soft_coverage",
    transition_boundary_radius: int = 4,
    transition_distribution_temperature: float = 0.7,
    transition_objective: str = "gaussian_mass",
    boundary_burst_quota: float = 5.0,
    boundary_burst_side_min_mass: float = 1.0,
    boundary_burst_anchor_weight: float = 1.0,
    boundary_burst_bilateral_weight: float = 1.0,
    boundary_burst_quota_weight: float = 1.0,
    boundary_burst_fairness_weight: float = 0.5,
    boundary_burst_overfill_weight: float = 0.25,
    actionness_loss_mode: str = "posterior_bce",
    loss_weights: Optional[Mapping[str, float]] = None,
    strict_loss_contract: bool = False,
) -> Dict[str, torch.Tensor]:
    """DUCA acquisition regularizers plus optional train-only utility loss."""

    budget_decision = None
    grid = None
    start_logits = None
    end_logits = None
    context_logits = None
    utility_scores = None
    transition_auxiliary_scores = None
    selector_variant = "direct_boundary"
    if isinstance(scores, Mapping):
        output = scores
        center_scores = output["center_scores"] if "center_scores" in output else output["scores"]
        selected_mask_st = selected_mask_st if selected_mask_st is not None else output["selected_mask_st"]
        grid = output.get("grid")
        budget_decision = output.get("budget_decision")
        if budget is None:
            if output.get("requested_budget") is not None:
                budget = output["requested_budget"]
            elif grid is not None:
                budget = int(grid.budget)
        valid_mask = valid_mask if valid_mask is not None else output.get("valid_mask")
        radius = radius if radius is not None else output.get("radius")
        p_action = p_action if p_action is not None else output.get("p_action")
        uncertainty = uncertainty if uncertainty is not None else output.get("uncertainty")
        actionness_logits = actionness_logits if actionness_logits is not None else output.get("actionness_logits")
        start_logits = output.get("start_logits")
        end_logits = output.get("end_logits")
        context_logits = output.get("context_logits")
        utility_scores = output.get("utility_scores")
        transition_auxiliary_scores = output.get("transition_auxiliary_scores")
        selector_variant = str(output.get("selector_variant", "direct_boundary"))
        detector_utility_target = (
            detector_utility_target
            if detector_utility_target is not None
            else output.get("detector_utility_target")
        )
        boundary_utility_proxy_target = (
            boundary_utility_proxy_target
            if boundary_utility_proxy_target is not None
            else output.get("boundary_utility_proxy_target")
        )
        if max_unselected_hole is None:
            max_unselected_hole = output.get("max_unselected_hole")
    else:
        center_scores = scores
    if selected_mask_st is None:
        raise ValueError("selected_mask_st must be provided")
    if budget is None:
        raise ValueError("budget must be provided")
    valid = _as_valid_mask(center_scores, valid_mask)
    if selected_mask_st.shape != center_scores.shape:
        raise ValueError("selected_mask_st must match scores")
    weights = dict(DUCA_LOSS_WEIGHT_DEFAULTS)
    strict_loss_contract = bool(strict_loss_contract)
    if loss_weights is not None:
        unknown = set(loss_weights) - set(DUCA_LOSS_WEIGHT_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown DUCA loss weights: {sorted(unknown)}")
        if strict_loss_contract and set(loss_weights) != set(DUCA_LOSS_WEIGHT_DEFAULTS):
            missing = sorted(set(DUCA_LOSS_WEIGHT_DEFAULTS) - set(loss_weights))
            raise ValueError(
                "strict DUCA loss contract requires every weight explicitly; "
                f"missing={missing}"
            )
        weights.update({key: float(value) for key, value in loss_weights.items()})
    elif strict_loss_contract:
        raise ValueError("strict DUCA loss contract requires an explicit loss_weights mapping")
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("DUCA loss weights must be finite and non-negative")
    if actionness_loss_mode not in {"posterior_bce", "class_balanced_mean"}:
        raise ValueError(
            "actionness_loss_mode must be posterior_bce or class_balanced_mean"
        )
    budgets = _budget_tensor(budget, center_scores.shape[0], center_scores.device).to(center_scores.dtype)
    selected = selected_mask_st.masked_fill(~valid, 0.0)
    zero = center_scores.float().new_zeros(())
    losses: Dict[str, torch.Tensor] = {
        loss_name: zero for loss_name in DUCA_LOSS_TO_WEIGHT_KEY
    }
    if detector_loss is not None and weights["detector"] != 0.0:
        losses["detector_loss"] = detector_loss * weights["detector"]
    if weights["budget"] != 0.0:
        over = F.relu(selected.sum(dim=1) - budgets)
        losses["budget_loss"] = over.pow(2).mean() * weights["budget"]
    if budget_decision is not None:
        if not isinstance(budget_decision, DynamicBudgetDecision):
            raise TypeError("budget_decision must be a DynamicBudgetDecision")
        budget_decision.validate(batch_size=center_scores.shape[0])
        if grid is not None and torch.any(grid.selected_count.to(center_scores.device) > int(budget_decision.budget_max)):
            raise RuntimeError("selected count exceeds dynamic hard budget cap")
        if weights["lagrangian_budget"] != 0.0:
            dynamic_cost = budget_decision.expected_cost.to(device=center_scores.device, dtype=center_scores.dtype)
            target = torch.as_tensor(
                float(budget_decision.target_budget),
                device=center_scores.device,
                dtype=center_scores.dtype,
            )
            lambda_dual = budget_decision.lambda_dual.to(device=center_scores.device, dtype=center_scores.dtype).detach()
            losses["lagrangian_budget_loss"] = (
                lambda_dual * (dynamic_cost.mean() - target) / target.clamp_min(1.0)
            ) * weights["lagrangian_budget"]
        if weights["marginal_monotonic"] != 0.0:
            marginal = budget_decision.marginal_utility.to(device=center_scores.device, dtype=center_scores.dtype)
            if marginal.shape[1] > 1:
                monotonic = F.relu(marginal[:, 1:] - marginal[:, :-1]).pow(2).mean()
            else:
                monotonic = zero
            losses["marginal_monotonic_loss"] = monotonic * weights["marginal_monotonic"]
        if weights["hard_budget_cap"] != 0.0:
            hard_over = F.relu(
                budget_decision.budget_hard.to(center_scores.dtype)
                - float(budget_decision.budget_max)
            )
            losses["hard_budget_cap_loss"] = (
                hard_over.pow(2).mean() * weights["hard_budget_cap"]
            )
    if teacher_utility is not None and weights["teacher"] != 0.0:
        if teacher_utility.shape != center_scores.shape:
            raise ValueError("teacher_utility must match scores [B,T]")
        utility = teacher_utility.to(center_scores.device, center_scores.dtype).masked_fill(~valid, 0.0)
        positive_utility = utility.clamp_min(0.0)
        negative_utility = (-utility).clamp_min(0.0)
        gain_loss = -((selected * positive_utility).sum(dim=1) / budgets.clamp_min(1.0)).mean()
        risk_loss = ((selected * negative_utility).sum(dim=1) / budgets.clamp_min(1.0)).mean()
        losses["teacher_utility_loss"] = (gain_loss + risk_loss) * weights["teacher"]
    elif weights["teacher"] != 0.0 and (
        utility_gain is not None or utility_risk is not None
    ):
        gain = torch.zeros_like(center_scores) if utility_gain is None else utility_gain.to(center_scores.device, center_scores.dtype)
        risk = torch.zeros_like(center_scores) if utility_risk is None else utility_risk.to(center_scores.device, center_scores.dtype)
        losses["teacher_utility_loss"] = (
            -((selected * gain.clamp_min(0.0)).sum(dim=1) / budgets.clamp_min(1.0)).mean()
            + ((selected * risk.clamp_min(0.0)).sum(dim=1) / budgets.clamp_min(1.0)).mean()
        ) * weights["teacher"]
    utility_proxy = boundary_utility_proxy_target if boundary_utility_proxy_target is not None else detector_utility_target
    if utility_proxy is not None and weights["detector_utility"] != 0.0:
        if utility_proxy.shape != center_scores.shape:
            raise ValueError("boundary_utility_proxy_target must match scores")
        utility = utility_proxy.to(center_scores.device, center_scores.dtype).clamp_min(0.0)
        utility = utility.masked_fill(~valid, 0.0)
        utility_logits = center_scores if utility_scores is None else utility_scores.to(center_scores.device, center_scores.dtype)
        losses["boundary_utility_proxy_distribution_loss"] = (
            _target_distribution_loss(utility_logits, utility, valid)
            * weights["detector_utility"]
        )
    if start_target is not None and weights["start"] != 0.0:
        if start_logits is None:
            raise ValueError("start_logits are required when start_target is provided")
        losses["start_endpoint_distribution_loss"] = (
            _target_distribution_loss(start_logits.to(center_scores.dtype), start_target, valid) * weights["start"]
        )
    if end_target is not None and weights["end"] != 0.0:
        if end_logits is None:
            raise ValueError("end_logits are required when end_target is provided")
        losses["end_endpoint_distribution_loss"] = (
            _target_distribution_loss(end_logits.to(center_scores.dtype), end_target, valid) * weights["end"]
        )
    if context_target is not None and weights["context"] != 0.0:
        if context_logits is None:
            raise ValueError("context_logits are required when context_target is provided")
        losses["boundary_context_distribution_loss"] = (
            _target_distribution_loss(context_logits.to(center_scores.dtype), context_target, valid) * weights["context"]
        )
    if transition_target is not None and weights["transition"] != 0.0:
        if transition_auxiliary_scores is None:
            raise ValueError(
                "transition_auxiliary_scores are required for active transition supervision"
            )
        losses["transition_distribution_loss"] = (
            transition_distribution_loss(
                transition_auxiliary_scores.to(center_scores.dtype),
                transition_target,
                valid,
                temperature=float(transition_distribution_temperature),
            )
            * weights["transition"]
        )
    if transition_target is not None and weights["transition_boundary"] != 0.0:
        if not isinstance(scores, Mapping):
            raise ValueError(
                "active transition boundary coverage requires structured decoder outputs"
            )
        soft_occupancy = scores.get("soft_coverage")
        if soft_occupancy is None or soft_occupancy.shape != center_scores.shape:
            raise ValueError("transition boundary coverage requires aligned structured soft_coverage")
        if str(transition_objective) == "boundary_burst":
            burst_loss, burst_components = boundary_burst_coverage_loss(
                soft_occupancy,
                transition_target,
                valid,
                radius=int(transition_boundary_radius),
                quota=float(boundary_burst_quota),
                side_min_mass=float(boundary_burst_side_min_mass),
                anchor_weight=float(boundary_burst_anchor_weight),
                bilateral_weight=float(boundary_burst_bilateral_weight),
                quota_weight=float(boundary_burst_quota_weight),
                fairness_weight=float(boundary_burst_fairness_weight),
                overfill_weight=float(boundary_burst_overfill_weight),
            )
            losses["transition_boundary_coverage_loss"] = (
                burst_loss * weights["transition_boundary"]
            )
            if isinstance(scores, dict):
                scores["boundary_burst_loss_components"] = burst_components
        elif str(transition_objective) == "gaussian_mass":
            losses["transition_boundary_coverage_loss"] = (
                local_boundary_mass_coverage_loss(
                    soft_occupancy,
                    transition_target,
                    valid,
                    radius=int(transition_boundary_radius),
                )
                * weights["transition_boundary"]
            )
        else:
            raise ValueError(
                "transition_objective must be gaussian_mass or boundary_burst"
            )
    if boundary_target is not None and weights["boundary"] != 0.0:
        if boundary_target.shape != center_scores.shape:
            raise ValueError("boundary_target must match scores")
        target = boundary_target.to(center_scores.device, torch.float32).masked_fill(~valid, 0.0)
        selected_for_boundary = selected.float()
        denom = target.sum(dim=1).clamp_min(1.0)
        uncovered = (target * (1.0 - selected_for_boundary.clamp(0.0, 1.0))).sum(dim=1) / denom
        losses["boundary_coverage_loss"] = uncovered.mean() * weights["boundary"]
    if action_target is not None:
        if action_target.shape != center_scores.shape:
            raise ValueError("action_target must match scores")
        action = action_target.to(center_scores.device, center_scores.dtype).masked_fill(~valid, 0.0)
        if actionness_logits is not None and weights["actionness"] != 0.0:
            logits = actionness_logits.to(center_scores.device, center_scores.dtype)
            if logits.shape != center_scores.shape:
                raise ValueError("actionness_logits must match scores when action_target is provided")
            logits = logits.masked_fill(~valid, 0.0)
            if selector_variant == "transition_only":
                actionness_loss, positive_weight = balanced_binary_actionness_loss(
                    logits,
                    action,
                    valid,
                    reduction_mode=actionness_loss_mode,
                )
                losses["actionness_bce_loss"] = actionness_loss * weights["actionness"]
                if isinstance(scores, dict):
                    scores["actionness_positive_weight"] = positive_weight
            else:
                bce = F.binary_cross_entropy_with_logits(
                    logits.float(), action.float(), reduction="none"
                ).masked_fill(~valid, 0.0)
                denom = valid.to(torch.float32).sum(dim=1).clamp_min(1.0)
                losses["actionness_bce_loss"] = ((bce.sum(dim=1) / denom).mean()) * weights["actionness"]
        if weights["hole"] != 0.0:
            local = F.max_pool1d(
                selected[:, None, :].clamp(0.0, 1.0),
                kernel_size=9,
                stride=1,
                padding=4,
            ).squeeze(1)
            denom = action.sum(dim=1).clamp_min(1.0)
            losses["action_local_hole_loss"] = (
                ((action * (1.0 - local)).sum(dim=1) / denom).mean()
                * weights["hole"]
            )
    if max_unselected_hole not in (None, 0) and weights["max_gap_hole"] != 0.0:
        if isinstance(scores, Mapping) and str(max_gap_loss_source) == "soft_coverage" and scores.get("soft_coverage") is not None:
            gap_mass = scores["soft_coverage"]
        else:
            gap_mass = selected
        losses["temporal_max_gap_hole_loss"] = (
            temporal_max_gap_hole_loss(
                gap_mass,
                valid,
                max_unselected_hole=int(max_unselected_hole),
                min_window_mass=float(max_gap_loss_min_window_mass),
            )
            * weights["max_gap_hole"]
        )
    else:
        losses["temporal_max_gap_hole_loss"] = zero
    if weights["redundancy"] != 0.0:
        losses["redundancy_loss"] = (
            selected[:, 1:] * selected[:, :-1]
        ).mean() * weights["redundancy"]
    else:
        losses["redundancy_loss"] = zero
    if radius is not None and weights["radius"] != 0.0:
        losses["radius_cost_loss"] = radius.to(center_scores.dtype).masked_fill(~valid, 0.0).mean() * weights["radius"]
    else:
        losses["radius_cost_loss"] = zero
    if weights["entropy"] == 0.0:
        losses["entropy_anti_collapse_loss"] = zero
        return losses
    if p_action is not None:
        prob = p_action.to(center_scores.dtype).clamp(1e-6, 1.0 - 1e-6)
        entropy = _binary_entropy(prob).masked_fill(~valid, 0.0)
    elif uncertainty is not None:
        entropy = uncertainty.to(center_scores.dtype).masked_fill(~valid, 0.0)
    else:
        entropy = _binary_entropy(torch.sigmoid(center_scores)).masked_fill(~valid, 0.0)
    losses["entropy_anti_collapse_loss"] = -entropy.mean() * weights["entropy"]
    return losses


def make_audit_record(grid: SparseTemporalGrid, uses_teacher: bool, mode: str, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    grid.validate()
    record: Dict[str, Any] = {
        "mode": str(mode),
        "budget": int(grid.budget),
        "budget_unit": grid.budget_unit,
        "coordinate": grid.coordinate,
        "detector_consumes_selected_positions": bool(grid.detector_consumes_selected_positions),
        "selected_count": [int(item) for item in grid.selected_count.detach().cpu().tolist()],
        "uses_gt": False,
        "uses_teacher": bool(uses_teacher),
        "uses_teacher_for_decision": False,
        "uses_teacher_for_train_loss": bool(uses_teacher) if str(mode).startswith("train") else False,
        "uses_teacher_at_inference": False,
        "uses_oracle": False,
        "uses_raw_prediction": False,
        "uses_prediction_cache": False,
        "uses_ledger_for_decision": False,
        "no_leak_scan_passed": True,
    }
    if extra:
        record.update(dict(extra))
    return record


def _binary_entropy(prob: torch.Tensor) -> torch.Tensor:
    prob = prob.clamp(1e-6, 1.0 - 1e-6)
    return -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())


def _get_observations(batch: Mapping[str, Any]) -> torch.Tensor:
    for key in ("observations", "dense_observations", "features", "inputs", "x"):
        value = batch.get(key)
        if torch.is_tensor(value):
            return value
    raise ValueError("batch must contain observations/dense_observations/features/inputs/x")


def _assert_no_forbidden_payload(obj: Any, forbidden_keys: Optional[set[str]] = None, path: str = "batch") -> None:
    keys = forbidden_keys or FORBIDDEN_DECISION_KEYS
    hits: List[str] = []

    def walk(value: Any, current: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_str = str(key)
                child_path = f"{current}.{key_str}"
                if key_str in keys or key_str.startswith("dense_teacher"):
                    hits.append(child_path)
                walk(child, child_path)
        elif isinstance(value, (list, tuple)):
            for idx, child in enumerate(value):
                walk(child, f"{current}[{idx}]")

    walk(obj, path)
    if hits:
        raise ValueError(f"DUCA test/inference forbids decision-time payloads: {hits}")


def _sanitize_detector_batch(obj: Any, forbidden_keys: Optional[set[str]] = None) -> Any:
    keys = forbidden_keys or FORBIDDEN_DECISION_KEYS
    if isinstance(obj, Mapping):
        out: Dict[Any, Any] = {}
        for key, value in obj.items():
            key_str = str(key)
            if key_str in keys or key_str.startswith("dense_teacher"):
                continue
            out[key] = _sanitize_detector_batch(value, keys)
        return out
    if isinstance(obj, list):
        return [_sanitize_detector_batch(value, keys) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_detector_batch(value, keys) for value in obj)
    return obj


def _extract_detector_loss(detector_output: Any, device: torch.device) -> Optional[torch.Tensor]:
    if detector_output is None:
        return None
    if torch.is_tensor(detector_output):
        return detector_output
    if isinstance(detector_output, Mapping):
        for key in ("loss", "total_loss", "cost"):
            value = detector_output.get(key)
            if torch.is_tensor(value):
                return value
        losses = detector_output.get("losses")
        if isinstance(losses, Mapping):
            tensors = [value for value in losses.values() if torch.is_tensor(value)]
            if tensors:
                total = torch.zeros((), device=device)
                for value in tensors:
                    total = total + value.to(device)
                return total
    return None


def _call_detector(
    detector: nn.Module,
    observations: torch.Tensor,
    grid: SparseTemporalGrid,
    batch: Mapping[str, Any],
    train: bool,
) -> Any:
    if hasattr(detector, "forward_sparse"):
        return detector.forward_sparse(observations, sparse_grid=grid, batch=batch, mode="loss" if train else "predict")
    try:
        return detector(observations, sparse_grid=grid, batch=batch, mode="loss" if train else "predict")
    except TypeError:
        try:
            return detector(observations, sparse_grid=grid)
        except TypeError:
            return detector(observations)
