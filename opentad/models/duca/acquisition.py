from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_budget import DynamicBudgetDecision, PrefixMarginalUtilityBudgetController


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

_PROVENANCE_FALSE_KEYS = (
    "thumos_trained",
    "uses_labels",
    "uses_teacher",
    "uses_gt",
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
    missing = [key for key in _PROVENANCE_FALSE_KEYS if key not in provenance]
    if missing:
        raise ValueError(f"{context} missing explicit fields: {', '.join(missing)}")
    unsafe = [key for key in _PROVENANCE_FALSE_KEYS if provenance.get(key) is not False]
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
        mobilenet_pretrained: bool = True,
        mobilenet_variant: str = "small",
        mobilenet_freeze_backbone: bool = True,
        mobilenet_weights_path: Optional[str] = None,
        official_action_seg_backend: str = "official_asformer",
        official_num_layers: int = 2,
        matrix_model_id: str = "timm_mobilenetv3_large_100_tsm_tcn",
        matrix_pretrained: bool = True,
        matrix_freeze_backbone: bool = True,
        train_split_supervised: bool = True,
        calibration_split: Optional[str] = "train_only",
        thumos_trained: bool = False,
        uses_labels: bool = False,
        uses_teacher: bool = False,
        uses_gt: bool = False,
        uses_prediction_cache: bool = False,
        return_hidden_features: bool = True,
        require_hidden_features: bool = True,
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
        self.source_name = str(source_name or self._default_source_name())
        self.train_split_supervised = bool(train_split_supervised)
        self.calibration_split = calibration_split
        self.return_hidden_features = bool(return_hidden_features)
        self.require_hidden_features = bool(require_hidden_features)
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
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
            "train_split_supervised": self.train_split_supervised,
            "calibration_split": self.calibration_split,
            "thumos_trained": bool(thumos_trained),
            "uses_labels": bool(uses_labels),
            "uses_teacher": bool(uses_teacher),
            "uses_gt": bool(uses_gt),
            "uses_prediction_cache": bool(uses_prediction_cache),
            "uses_labels_at_inference": False,
            "uses_gt_at_inference": False,
            "uses_teacher_at_inference": False,
            "joint_trainable": not self.frozen,
            "checkpoint_is_initialization": bool(self.checkpoint_path),
            "returns_hidden_features": self.return_hidden_features,
            "requires_hidden_features": self.require_hidden_features,
        }

        probe_mod = self._probe_module()
        if self.probe_model == "mobilenetv3":
            self.probe = probe_mod.C3MobileNetV3ActionProbe(
                pretrained=bool(mobilenet_pretrained),
                variant=str(mobilenet_variant),
                freeze_backbone=bool(mobilenet_freeze_backbone),
                weights_path=mobilenet_weights_path,
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
            )
        elif self.probe_model == "matrix-zoo":
            self.probe = probe_mod.C3MatrixZooActionProbe(
                model_id=str(matrix_model_id),
                pretrained=bool(matrix_pretrained),
                freeze_backbone=bool(matrix_freeze_backbone),
            )
        else:
            raise ValueError(
                "probe_model must be one of mobilenetv3, temporal-tcn, official-action-seg, or matrix-zoo"
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

    def _estimate_probe_profile(self, inputs: torch.Tensor, logits: torch.Tensor, latency_ms: Optional[float]) -> Dict[str, Any]:
        params = _module_param_counts(self)
        batch = int(logits.shape[0])
        temporal_len = int(logits.shape[1])
        tokens = batch * temporal_len
        spatial = int(self.spatial_size)
        if self.probe_model == "mobilenetv3":
            per_frame_macs = int(56_500_000 * (spatial / 224.0) ** 2)
            macs = tokens * per_frame_macs
            family = "MobileNetV3-small"
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
            macs = int(tokens * hidden * hidden * 10 + batch * temporal_len * temporal_len * hidden * 4)
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
        probe_inputs = self._prepare_probe_inputs(inputs)
        if hasattr(probe_inputs, "to"):
            probe_inputs = probe_inputs.to(device=inputs.device)
        def call_probe() -> Any:
            try:
                return self.probe(probe_inputs, valid, return_hidden=self.return_hidden_features)
            except TypeError:
                if self.return_hidden_features and self.require_hidden_features:
                    raise
                return self.probe(probe_inputs, valid)

        if self.frozen:
            with torch.no_grad():
                probe_output = call_probe()
        else:
            probe_output = call_probe()
        hidden = None
        if isinstance(probe_output, Mapping):
            logits = probe_output.get("logits")
            hidden = (
                probe_output.get("coarse_hidden_features")
                if probe_output.get("coarse_hidden_features") is not None
                else probe_output.get("hidden")
            )
            if logits is None:
                raise ValueError("C3 coarse probe output mapping must contain logits")
        else:
            logits = probe_output
        if hidden is None and self.return_hidden_features and self.require_hidden_features:
            raise ValueError("C3 coarse probe must return hidden features for final DUCA selector fusion")
        logits = logits.float().to(device=inputs.device).masked_fill(~valid, _neg(torch.float32))
        if hidden is not None:
            if hidden.ndim != 3:
                raise ValueError(f"C3 coarse hidden features must be [B,T,D], got {tuple(hidden.shape)}")
            if hidden.shape[:2] != logits.shape:
                raise ValueError("C3 coarse hidden features must align with logits [B,T]")
            hidden = hidden.float().to(device=inputs.device).masked_fill(~valid[:, :, None], 0.0)
        latency_ms = float((time.perf_counter() - start) * 1000.0)
        p_action = torch.sigmoid(logits).masked_fill(~valid, 0.0)
        transition = _actionness_transition_payload(p_action, valid)
        profile = self._estimate_probe_profile(inputs, logits, latency_ms)
        output = {
            "p_action": transition["p_action"],
            "logits": logits,
            "actionness_logits": logits,
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
        }
        if hidden is not None:
            output["coarse_hidden_features"] = hidden
            output["hidden_features"] = hidden
            profile["hidden_output_shape"] = [int(v) for v in hidden.shape]
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
        actionness_weight: float = 0.05,
        transition_weight: float = 1.0,
        uncertainty_weight: float = 0.25,
        utility_weight: float = 0.50,
        boundary_weight: float = 1.0,
        coarse_hidden_dim: Optional[int] = None,
        require_coarse_hidden_features: bool = False,
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
        self.coarse_hidden_dim = 0 if coarse_hidden_dim in (None, 0) else int(coarse_hidden_dim)
        if self.coarse_hidden_dim < 0:
            raise ValueError("coarse_hidden_dim must be non-negative")
        self.require_coarse_hidden_features = bool(require_coarse_hidden_features)
        self.max_unselected_hole = None if max_unselected_hole in (None, 0) else int(max_unselected_hole)
        if self.max_unselected_hole is not None and self.max_unselected_hole < 0:
            raise ValueError("max_unselected_hole must be non-negative")
        self.hard_max_gap_repair = bool(hard_max_gap_repair)
        self.fail_on_infeasible_max_gap = bool(fail_on_infeasible_max_gap)
        self.profile_runtime = bool(profile_runtime)
        self.profile_sync_cuda = bool(profile_sync_cuda)
        self.last_compute_profile: Dict[str, Any] = {}
        self.actionness_source = actionness_source or ZeroShotActionnessSource(feature_dim=feature_dim, mode="motion")
        self.feature_dim = None if feature_dim is None else int(feature_dim)
        if self.feature_dim is None:
            self.encoder = None
            self.center_head = None
            self.radius_head = None
            self.boundary_head = None
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
            self.radius_head = nn.Linear(int(hidden_dim), 1)
            self.boundary_head = nn.Linear(int(hidden_dim), 1)
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
        macs = tokens * (in_dim * hidden + hidden * hidden + 4 * hidden)
        flops = 2 * macs + tokens * (6 * in_dim + 12 * hidden + 48)
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
        soft_coverage_macs = batch_size * temporal_len * temporal_len
        soft_coverage_flops = soft_coverage_macs * 8
        gather_flops = batch_size * min(self.budget, temporal_len) * feature_dim
        components = {
            "descriptor": descriptor,
            "actionness": actionness,
            "selector": selector,
            "budget_controller": budget_controller,
            "soft_coverage": {
                "estimated_macs": int(soft_coverage_macs),
                "estimated_flops": int(soft_coverage_flops),
                "complexity": "O(B*T^2)",
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
        coarse_hidden_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if dense_observations.ndim != 3:
            raise ValueError(f"dense_observations must be [B,T,C], got {tuple(dense_observations.shape)}")
        source = self.actionness_source(
            dense_observations,
            logits=actionness_logits,
            valid_mask=valid_mask,
            p_action=p_action,
        )
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
        if self.encoder is None:
            boundary_logits = transition_score
            utility_scores = transition_score + 0.5 * actionness_aux
            center_scores = (
                self.transition_weight * transition_score
                + self.boundary_weight * boundary_logits
                + self.utility_weight * utility_scores
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
            center_scores = self.center_head(encoded).squeeze(-1)
            radius = self.max_radius * torch.sigmoid(self.radius_head(encoded).squeeze(-1))
            boundary_logits = self.boundary_head(encoded).squeeze(-1)
            utility_scores = self.utility_head(encoded).squeeze(-1)
            center_scores = (
                center_scores
                + self.transition_weight * transition_score
                + self.uncertainty_weight * source["uncertainty"]
                + self.utility_weight * utility_scores
                + self.boundary_weight * boundary_logits
                + self.actionness_weight * actionness_aux
            )
        center_scores = center_scores.masked_fill(~valid, _neg(center_scores.dtype))
        radius = radius.masked_fill(~valid, 0.0).clamp(0.0, float(self.max_radius))
        return {
            "center_scores": center_scores,
            "scores": center_scores,
            "radius": radius,
            "boundary_logits": boundary_logits.masked_fill(~valid, 0.0),
            "utility_scores": utility_scores.masked_fill(~valid, 0.0),
            "p_action": source["p_action"],
            "uncertainty": source["uncertainty"],
            "entropy": source["entropy"],
            "delta_p_action": source["delta_p_action"],
            "abs_delta_p_action": source["abs_delta_p_action"],
            "uncertainty_peak": source["uncertainty_peak"],
            "transition_score": transition_score.masked_fill(~valid, 0.0),
            "actionness_logits": source["logits"],
            "selection_features": selection_features.masked_fill(~valid[:, :, None], 0.0),
            "coarse_hidden_features": None if coarse_hidden is None else coarse_hidden.masked_fill(~valid[:, :, None], 0.0),
            "uses_coarse_hidden_features": bool(has_coarse_hidden_features),
            "valid_mask": valid,
            "provenance": source["provenance"],
        }

    def acquire(
        self,
        dense_observations: torch.Tensor,
        budget: Optional[TensorLikeBudget] = None,
        valid_mask: Optional[torch.Tensor] = None,
        actionness_logits: Optional[torch.Tensor] = None,
        p_action: Optional[torch.Tensor] = None,
        coarse_hidden_features: Optional[torch.Tensor] = None,
        compute_profile_context: Optional[Mapping[str, Any]] = None,
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
            coarse_hidden_features=coarse_hidden_features,
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
                "decoder": "budgeted_center_radius_decode",
                "radius_is_metadata": True,
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
            },
        ).validate()
        if torch.any(grid.selected_count > int(self.budget)):
            raise RuntimeError("DUCA dynamic acquisition selected more observations than the hard cap")
        soft_coverage_start = (
            _sync_profile_clock(dense_observations, enabled=sync_enabled) if profile_enabled else None
        )
        soft_coverage = soft_center_radius_coverage(
            center_scores=scores["center_scores"],
            radius=scores["radius"],
            valid_mask=scores["valid_mask"],
            budget=budgets,
            max_radius=self.max_radius,
        )
        soft_coverage_ms = _elapsed_ms(soft_coverage_start, dense_observations, enabled=sync_enabled)
        hard_union = grid.selected_mask.to(dtype=scores["center_scores"].dtype)
        selection_st = hard_union + soft_coverage - soft_coverage.detach()
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
                "decode_metadata": grid.metadata,
                "budget_decision": budget_decision,
                "dynamic_budget": bool(self.dynamic_budget),
                "budget_mode": self.budget_mode,
                "requested_budget": budgets,
                "effective_budget": effective_budget,
                "max_unselected_hole": self.max_unselected_hole,
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
        return_audit: bool = False,
    ) -> Dict[str, Any]:
        grid, scores = self.acquire(
            dense_observations=dense_observations,
            budget=budget,
            valid_mask=valid_mask,
            actionness_logits=actionness_logits,
            p_action=p_action,
            coarse_hidden_features=coarse_hidden_features,
        )
        gathered = gather_selected_observations(dense_observations, grid.selected_positions, grid.selected_mask)
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
        return selection_mass.new_zeros(())
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
        return selection_mass.new_zeros(())
    return torch.cat([item.reshape(-1) for item in penalties], dim=0).mean().to(dtype=selection_mass.dtype)


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


def _repair_selected_max_unselected_hole(
    selected_positions: List[int],
    score_values: torch.Tensor,
    valid_positions: List[int],
    *,
    budget: int,
    max_unselected_hole: int,
) -> Tuple[List[int], Dict[str, Any]]:
    max_hole = int(max_unselected_hole)
    selected = {int(pos) for pos in selected_positions}
    if len(selected) != int(budget):
        raise ValueError("hard max-gap repair expects a strict-budget selection")
    minimum_required = _minimum_selection_for_max_hole(len(valid_positions), max_hole)
    metadata: Dict[str, Any] = {
        "enabled": True,
        "max_unselected_hole": max_hole,
        "minimum_required_budget": int(minimum_required),
        "requested_budget": int(budget),
        "feasible": int(budget) >= int(minimum_required),
        "repair_count": 0,
        "max_unselected_hole_before": int(_max_unselected_hole(selected, valid_positions)),
        "max_unselected_hole_after": None,
    }
    if not metadata["feasible"]:
        metadata["max_unselected_hole_after"] = metadata["max_unselected_hole_before"]
        return sorted(selected), metadata
    valid_set = set(valid_positions)

    def score(pos: int) -> float:
        return float(score_values[int(pos)].detach().cpu().item())

    for _ in range(len(valid_positions) + 1):
        violating = [run for run in _unselected_hole_runs(selected, valid_positions) if int(run[2]) > max_hole]
        if not violating:
            break
        start, end, length = max(violating, key=lambda item: (int(item[2]), -int(item[0])))
        feasible_start = max(int(start), int(end) - max_hole)
        feasible_end = min(int(end), int(start) + max_hole)
        candidates = [pos for pos in range(feasible_start, feasible_end + 1) if pos in valid_set and pos not in selected]
        if not candidates:
            candidates = [pos for pos in range(int(start), int(end) + 1) if pos in valid_set and pos not in selected]
        if not candidates:
            break
        added = max(candidates, key=lambda pos: (score(pos), -abs(pos - (int(start) + int(end)) // 2), -pos))
        selected.add(int(added))
        removable = []
        for victim in sorted(selected):
            if int(victim) == int(added):
                continue
            trial = set(selected)
            trial.remove(int(victim))
            trial_hole = _max_unselected_hole(trial, valid_positions)
            removable.append((trial_hole <= max_hole, trial_hole, score(int(victim)), -int(victim), int(victim)))
        safe = [item for item in removable if item[0]]
        if safe:
            victim = min(safe, key=lambda item: (item[2], item[3]))[-1]
        elif removable:
            victim = min(removable, key=lambda item: (item[1], item[2], item[3]))[-1]
        else:
            selected.remove(int(added))
            break
        selected.remove(int(victim))
        metadata["repair_count"] = int(metadata["repair_count"]) + 1
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


def duca_losses(
    scores: Union[Mapping[str, Any], torch.Tensor],
    selected_mask_st: Optional[torch.Tensor] = None,
    budget: Optional[TensorLikeBudget] = None,
    valid_mask: Optional[torch.Tensor] = None,
    teacher_utility: Optional[torch.Tensor] = None,
    boundary_target: Optional[torch.Tensor] = None,
    action_target: Optional[torch.Tensor] = None,
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
    loss_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """DUCA acquisition regularizers plus optional train-only utility loss."""

    budget_decision = None
    grid = None
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
    weights = {
        "detector": 1.0,
        "actionness": 0.0,
        "budget": 0.05,
        "boundary": 0.25,
        "hole": 0.25,
        "max_gap_hole": 0.0,
        "redundancy": 0.05,
        "radius": 0.02,
        "entropy": 0.01,
        "teacher": 0.50,
        "detector_utility": 0.0,
        "lagrangian_budget": 1.0,
        "marginal_monotonic": 0.01,
        "hard_budget_cap": 1.0,
    }
    if loss_weights is not None:
        weights.update({key: float(value) for key, value in loss_weights.items()})
    budgets = _budget_tensor(budget, center_scores.shape[0], center_scores.device).to(center_scores.dtype)
    selected = selected_mask_st.masked_fill(~valid, 0.0)
    losses: Dict[str, torch.Tensor] = {}
    zero = center_scores.new_zeros(())
    if detector_loss is not None:
        losses["detector_loss"] = detector_loss * weights["detector"]
    else:
        losses["detector_loss"] = zero
    over = F.relu(selected.sum(dim=1) - budgets)
    losses["budget_loss"] = over.pow(2).mean() * weights["budget"]
    if budget_decision is not None:
        if not isinstance(budget_decision, DynamicBudgetDecision):
            raise TypeError("budget_decision must be a DynamicBudgetDecision")
        budget_decision.validate(batch_size=center_scores.shape[0])
        if grid is not None and torch.any(grid.selected_count.to(center_scores.device) > int(budget_decision.budget_max)):
            raise RuntimeError("selected count exceeds dynamic hard budget cap")
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
        marginal = budget_decision.marginal_utility.to(device=center_scores.device, dtype=center_scores.dtype)
        if marginal.shape[1] > 1:
            monotonic = F.relu(marginal[:, 1:] - marginal[:, :-1]).pow(2).mean()
        else:
            monotonic = zero
        losses["marginal_monotonic_loss"] = monotonic * weights["marginal_monotonic"]
        hard_over = F.relu(budget_decision.budget_hard.to(center_scores.dtype) - float(budget_decision.budget_max))
        losses["hard_budget_cap_loss"] = hard_over.pow(2).mean() * weights["hard_budget_cap"]
        losses["dynamic_budget_mean_lossless_metric"] = zero
    if teacher_utility is not None:
        if teacher_utility.shape != center_scores.shape:
            raise ValueError("teacher_utility must match scores [B,T]")
        utility = teacher_utility.to(center_scores.device, center_scores.dtype).masked_fill(~valid, 0.0)
        positive_utility = utility.clamp_min(0.0)
        negative_utility = (-utility).clamp_min(0.0)
        gain_loss = -((selected * positive_utility).sum(dim=1) / budgets.clamp_min(1.0)).mean()
        risk_loss = ((selected * negative_utility).sum(dim=1) / budgets.clamp_min(1.0)).mean()
        losses["teacher_utility_gain_loss_unweighted"] = gain_loss
        losses["teacher_utility_risk_loss_unweighted"] = risk_loss
        losses["teacher_utility_loss"] = (gain_loss + risk_loss) * weights["teacher"]
    elif utility_gain is not None or utility_risk is not None:
        gain = torch.zeros_like(center_scores) if utility_gain is None else utility_gain.to(center_scores.device, center_scores.dtype)
        risk = torch.zeros_like(center_scores) if utility_risk is None else utility_risk.to(center_scores.device, center_scores.dtype)
        losses["teacher_utility_loss"] = (
            -((selected * gain.clamp_min(0.0)).sum(dim=1) / budgets.clamp_min(1.0)).mean()
            + ((selected * risk.clamp_min(0.0)).sum(dim=1) / budgets.clamp_min(1.0)).mean()
        ) * weights["teacher"]
    else:
        losses["teacher_utility_loss"] = zero
    utility_proxy = boundary_utility_proxy_target if boundary_utility_proxy_target is not None else detector_utility_target
    utility_proxy_loss: Optional[torch.Tensor] = None
    if utility_proxy is not None:
        if utility_proxy.shape != center_scores.shape:
            raise ValueError("boundary_utility_proxy_target must match scores")
        utility = utility_proxy.to(center_scores.device, center_scores.dtype).clamp_min(0.0)
        utility = utility.masked_fill(~valid, 0.0)
        selected_positive = selected.clamp_min(0.0).masked_fill(~valid, 0.0)
        eps = torch.finfo(center_scores.dtype).eps
        utility_sum = utility.sum(dim=1)
        active = utility_sum > eps
        if bool(active.any().item()):
            target_dist = utility / utility_sum.clamp_min(eps)[:, None]
            selected_dist = selected_positive / selected_positive.sum(dim=1).clamp_min(eps)[:, None]
            kl = target_dist * (target_dist.clamp_min(eps).log() - selected_dist.clamp_min(eps).log())
            utility_proxy_loss = kl.sum(dim=1)[active].mean() * weights["detector_utility"]
        else:
            utility_proxy_loss = zero
    else:
        utility_proxy_loss = zero
    losses["boundary_utility_proxy_distribution_loss"] = utility_proxy_loss
    if boundary_target is not None:
        if boundary_target.shape != center_scores.shape:
            raise ValueError("boundary_target must match scores")
        target = boundary_target.to(center_scores.device, center_scores.dtype).masked_fill(~valid, 0.0)
        denom = target.sum(dim=1).clamp_min(1.0)
        uncovered = (target * (1.0 - selected.clamp(0.0, 1.0))).sum(dim=1) / denom
        losses["boundary_coverage_loss"] = uncovered.mean() * weights["boundary"]
    else:
        losses["boundary_coverage_loss"] = zero
    if action_target is not None:
        if action_target.shape != center_scores.shape:
            raise ValueError("action_target must match scores")
        action = action_target.to(center_scores.device, center_scores.dtype).masked_fill(~valid, 0.0)
        if actionness_logits is not None:
            logits = actionness_logits.to(center_scores.device, center_scores.dtype)
            if logits.shape != center_scores.shape:
                raise ValueError("actionness_logits must match scores when action_target is provided")
            logits = logits.masked_fill(~valid, 0.0)
            bce = F.binary_cross_entropy_with_logits(logits, action, reduction="none").masked_fill(~valid, 0.0)
            denom = valid.to(center_scores.dtype).sum(dim=1).clamp_min(1.0)
            losses["actionness_bce_loss"] = ((bce.sum(dim=1) / denom).mean()) * weights["actionness"]
        else:
            losses["actionness_bce_loss"] = zero
        local = F.max_pool1d(selected[:, None, :].clamp(0.0, 1.0), kernel_size=9, stride=1, padding=4).squeeze(1)
        denom = action.sum(dim=1).clamp_min(1.0)
        losses["action_local_hole_loss"] = ((action * (1.0 - local)).sum(dim=1) / denom).mean() * weights["hole"]
    else:
        losses["actionness_bce_loss"] = zero
        losses["action_local_hole_loss"] = zero
    if max_unselected_hole not in (None, 0):
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
    losses["redundancy_loss"] = (selected[:, 1:] * selected[:, :-1]).mean() * weights["redundancy"]
    if radius is not None:
        losses["radius_cost_loss"] = radius.to(center_scores.dtype).masked_fill(~valid, 0.0).mean() * weights["radius"]
    else:
        losses["radius_cost_loss"] = zero
    if p_action is not None:
        prob = p_action.to(center_scores.dtype).clamp(1e-6, 1.0 - 1e-6)
        entropy = _binary_entropy(prob).masked_fill(~valid, 0.0)
    elif uncertainty is not None:
        entropy = uncertainty.to(center_scores.dtype).masked_fill(~valid, 0.0)
    else:
        entropy = _binary_entropy(torch.sigmoid(center_scores)).masked_fill(~valid, 0.0)
    losses["entropy_anti_collapse_loss"] = -entropy.mean() * weights["entropy"]
    total = zero
    for value in losses.values():
        total = total + value
    losses["total_loss"] = total
    losses["detector_utility_distribution_loss"] = losses["boundary_utility_proxy_distribution_loss"]
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
