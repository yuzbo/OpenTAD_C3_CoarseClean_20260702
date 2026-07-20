from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .continuous_roi_geometry import (
    CONTINUOUS_ROI_GENERATOR_SCHEMA,
    anchor_knot_logits,
    common_support_clip_boxes,
    decode_continuous_roi_logits,
    interpolate_knot_logits,
)
from .continuous_roi_sampler import (
    CONTINUOUS_ROI_SAMPLER_SCHEMA,
    sample_continuous_roi,
)
from .native_crop_wrapper import (
    NativeCropBackboneWrapper,
    deterministic_linear_2x,
)


CONTINUOUS_ROI_BACKBONE_SCHEMA = "continuous_roi_common_support_u128_v2_1"


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3:
            raise ValueError("ChannelLayerNorm expects [B,C,T]")
        return self.norm(value.transpose(1, 2)).transpose(1, 2)


class ContinuousRoiFeatureFusion(nn.Module):
    """Registered global/local fusion with exactly 594,049 parameters."""

    EXPECTED_PARAMETERS = 594_049

    def __init__(self, channels: int = 384):
        super().__init__()
        if channels != 384:
            raise ValueError("Continuous-RoI v2.1 fusion is frozen to 384 channels")
        joined_channels = 3 * channels
        self.global_norm = ChannelLayerNorm(channels)
        self.local_norm = ChannelLayerNorm(channels)
        self.alpha = nn.Conv1d(joined_channels, 1, kernel_size=1)
        self.delta_in = nn.Conv1d(joined_channels, channels, kernel_size=1)
        self.delta_out = nn.Conv1d(channels, channels, kernel_size=1)
        self.output_norm = ChannelLayerNorm(channels)
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count != self.EXPECTED_PARAMETERS:
            raise RuntimeError(
                "Continuous-RoI fusion parameter contract changed: "
                f"{parameter_count} != {self.EXPECTED_PARAMETERS}"
            )

    def forward(
        self,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if global_features.shape != local_features.shape:
            raise ValueError(
                "Continuous-RoI branches must align before fusion; "
                f"global={tuple(global_features.shape)} "
                f"local={tuple(local_features.shape)}"
            )
        global_norm = self.global_norm(global_features)
        local_norm = self.local_norm(local_features)
        joined = torch.cat(
            (global_norm, local_norm, local_norm - global_norm),
            dim=1,
        )
        alpha = 0.25 + 0.50 * torch.sigmoid(self.alpha(joined))
        delta = self.delta_out(F.gelu(self.delta_in(joined)))
        fused = self.output_norm(
            (1.0 - alpha) * global_features
            + alpha * local_features
            + 0.10 * torch.tanh(delta)
        )
        return fused, alpha


def temporal_class_occupancy_targets(
    gt_segments: Sequence[torch.Tensor],
    gt_labels: Sequence[torch.Tensor],
    *,
    batch_size: int,
    num_classes: int,
    output_length: int,
    detector_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert fit-only temporal annotations to coarse class occupancy."""

    if len(gt_segments) != batch_size or len(gt_labels) != batch_size:
        raise ValueError("GT occupancy inputs do not match the feature batch")
    if min(num_classes, output_length, detector_length) <= 0:
        raise ValueError("occupancy geometry must be positive")
    centers = (
        torch.arange(output_length, device=device, dtype=dtype) + 0.5
    ) * (float(detector_length) / float(output_length))
    target = torch.zeros(
        (batch_size, num_classes, output_length),
        device=device,
        dtype=dtype,
    )
    for batch_index, (segments_raw, labels_raw) in enumerate(
        zip(gt_segments, gt_labels)
    ):
        segments = torch.as_tensor(
            segments_raw,
            device=device,
            dtype=dtype,
        ).reshape(-1, 2)
        labels = torch.as_tensor(
            labels_raw,
            device=device,
            dtype=torch.long,
        ).reshape(-1)
        if segments.shape[0] != labels.shape[0]:
            raise ValueError("GT segments and labels have different lengths")
        if not bool(torch.isfinite(segments).all().item()):
            raise ValueError("GT segments must be finite")
        if bool(((labels < 0) | (labels >= num_classes)).any().item()):
            raise ValueError("GT labels fall outside the registered class range")
        for segment, label in zip(segments, labels):
            start, end = segment.unbind()
            if end <= start:
                continue
            occupied = (centers >= start) & (centers < end)
            target[batch_index, label, occupied] = 1.0
    return target


def auxiliary_loss_weights(successful_update: int) -> tuple[float, float]:
    if successful_update < 0:
        raise ValueError("successful_update must be non-negative")
    rho = min(max((float(successful_update) - 800.0) / 1600.0, 0.0), 1.0)
    return 0.25 - 0.15 * rho, 0.50 - 0.30 * rho


class ContinuousRoiBackboneWrapper(NativeCropBackboneWrapper):
    """Selector-free common-support representation for Continuous-RoI S2."""

    AUXILIARY_PARAMETERS = 15_400
    TOTAL_NEW_PARAMETERS = 609_449

    def __init__(self, cfg):
        custom_cfg = cfg.custom
        self.source_key = str(
            getattr(custom_cfg, "continuous_roi_source_key", "source")
        )
        self.sample_key = str(
            getattr(custom_cfg, "continuous_roi_sample_key", "sample_key")
        )
        self.window_start_key = str(
            getattr(custom_cfg, "continuous_roi_window_start_key", "window_start")
        )
        self.boxes_key = str(
            getattr(custom_cfg, "continuous_roi_boxes_key", "roi_clip_boxes")
        )
        self.training_seed = int(
            getattr(custom_cfg, "continuous_roi_training_seed", 3407)
        )
        self.source_height = int(
            getattr(custom_cfg, "continuous_roi_source_height", 180)
        )
        self.source_width = int(
            getattr(custom_cfg, "continuous_roi_source_width", 320)
        )
        self.knots = int(getattr(custom_cfg, "continuous_roi_knots", 12))
        self.frames_per_clip = int(
            getattr(custom_cfg, "continuous_roi_frames_per_clip", 16)
        )
        self.local_clips_per_call = int(
            getattr(custom_cfg, "continuous_roi_local_clips_per_call", 4)
        )
        self.num_classes = int(
            getattr(custom_cfg, "continuous_roi_num_classes", 20)
        )
        self.auxiliary_detector_length = int(
            getattr(custom_cfg, "continuous_roi_detector_length", 768)
        )
        super().__init__(cfg)
        if self.knots != 12 or self.chunk_num != 48:
            raise ValueError(
                "Continuous-RoI v2.1 requires 12 knots and 48 temporal clips"
            )
        if self.frames_per_clip * self.chunk_num != self.auxiliary_detector_length:
            raise ValueError("Continuous-RoI source time geometry is inconsistent")
        if self.expected_local_size != 128 or self.expected_global_size != 96:
            raise ValueError("Continuous-RoI U128 requires global96 and local128")
        self.fusion = ContinuousRoiFeatureFusion(channels=384)
        self.global_aux_head = nn.Conv1d(384, self.num_classes, kernel_size=1)
        self.local_aux_head = nn.Conv1d(384, self.num_classes, kernel_size=1)
        auxiliary_parameters = sum(
            parameter.numel()
            for module in (self.global_aux_head, self.local_aux_head)
            for parameter in module.parameters()
        )
        if auxiliary_parameters != self.AUXILIARY_PARAMETERS:
            raise RuntimeError("Continuous-RoI auxiliary parameter contract changed")
        new_parameters = (
            sum(parameter.numel() for parameter in self.fusion.parameters())
            + auxiliary_parameters
        )
        if new_parameters != self.TOTAL_NEW_PARAMETERS:
            raise RuntimeError("Continuous-RoI total new parameter contract changed")
        self._successful_update_index = None
        self._pending_auxiliary = None
        self.latest_continuous_roi_audit = None

    def set_successful_update_index(self, index: int) -> None:
        index = int(index)
        if index < 0:
            raise ValueError("successful update index must be non-negative")
        self._successful_update_index = index

    def _validate_inputs(self, frames: Mapping) -> tuple[torch.Tensor, ...]:
        if not isinstance(frames, Mapping):
            raise TypeError("ContinuousRoiBackboneWrapper requires a mapping")
        required = {
            self.global_key,
            self.source_key,
            self.sample_key,
            self.window_start_key,
        }
        allowed = required | {self.boxes_key}
        if not required.issubset(frames) or not set(frames).issubset(allowed):
            raise ValueError(
                "Continuous-RoI inputs are fail-closed; "
                f"required={sorted(required)} optional={self.boxes_key!r} "
                f"got={sorted(frames)}"
            )
        global_view = frames[self.global_key]
        source = frames[self.source_key]
        self._validate_view(
            global_view,
            name="global",
            size=self.expected_global_size,
        )
        if not isinstance(source, torch.Tensor) or source.ndim != 6:
            raise ValueError("Continuous-RoI source must be [B,1,3,T,H,W]")
        if source.shape[1:3] != (1, 3):
            raise ValueError("Continuous-RoI source requires N=1 and RGB")
        if source.shape[-3:] != (
            self.auxiliary_detector_length,
            self.source_height,
            self.source_width,
        ):
            raise ValueError(
                "Continuous-RoI source geometry mismatch: "
                f"{tuple(source.shape[-3:])}"
            )
        if source.dtype != torch.uint8:
            raise TypeError("Continuous-RoI source must stay uint8 before sampling")
        if global_view.shape[0] != source.shape[0] or global_view.shape[-3] != source.shape[-3]:
            raise ValueError("Continuous-RoI source/global batch or time axis mismatch")
        sample_keys = torch.as_tensor(frames[self.sample_key]).reshape(-1)
        window_starts = torch.as_tensor(frames[self.window_start_key]).reshape(-1)
        if sample_keys.shape != (source.shape[0],) or window_starts.shape != (
            source.shape[0],
        ):
            raise ValueError("Continuous-RoI semantic keys must be one scalar per sample")
        return global_view, source, sample_keys, window_starts

    def _resolve_boxes(
        self,
        frames: Mapping,
        *,
        sample_keys: torch.Tensor,
        window_starts: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, tuple[str, ...], torch.Tensor]:
        batch_size = int(sample_keys.numel())
        if self.boxes_key in frames:
            if self.training:
                raise ValueError(
                    "Continuous-RoI training forbids externally supplied geometry"
                )
            boxes = torch.as_tensor(
                frames[self.boxes_key],
                device=device,
                dtype=dtype,
            )
            if boxes.shape != (batch_size, self.chunk_num, 4):
                raise ValueError(
                    "external Continuous-RoI boxes must be [B,48,4]"
                )
            decoded = boxes
            families = tuple("external_registered" for _ in range(batch_size))
            knot_logits = torch.empty(
                (batch_size, 0, 4),
                device=device,
                dtype=dtype,
            )
        elif self.training:
            if self._successful_update_index is None:
                raise RuntimeError(
                    "Continuous-RoI training requires a successful-update index hook"
                )
            decoded, families, knot_logits = common_support_clip_boxes(
                training_seed=self.training_seed,
                successful_update=self._successful_update_index,
                sample_keys=sample_keys,
                window_starts=window_starts,
                knots=self.knots,
                clips=self.chunk_num,
                dtype=dtype,
                device=device,
            )
        else:
            knot_logits = anchor_knot_logits(
                batch_size=batch_size,
                knots=self.knots,
                dtype=dtype,
                device=device,
            )
            decoded = decode_continuous_roi_logits(
                interpolate_knot_logits(knot_logits, clips=self.chunk_num)
            )
            families = tuple("anchor" for _ in range(batch_size))
        # Reuse the decoder's strict geometric validation for external boxes.
        center = decoded[..., :2]
        extent = decoded[..., 2:]
        if not bool(torch.isfinite(decoded).all().item()):
            raise ValueError("Continuous-RoI boxes must be finite")
        if bool((extent <= 0.0).any().item()):
            raise ValueError("Continuous-RoI box extents must be positive")
        if bool(
            (
                (center - 0.5 * extent < -1e-6)
                | (center + 0.5 * extent > 1.0 + 1e-6)
            ).any().item()
        ):
            raise ValueError("Continuous-RoI boxes must remain in source bounds")
        return decoded, families, knot_logits

    def forward(self, frames, masks=None):
        global_view, source, sample_keys, window_starts = self._validate_inputs(frames)
        self.set_norm_layer()
        global_features, global_audit = self._encode_view(
            global_view,
            name="global",
        )
        boxes, families, knot_logits = self._resolve_boxes(
            frames,
            sample_keys=sample_keys,
            window_starts=window_starts,
            device=global_features.device,
            dtype=global_features.dtype,
        )
        local_view = sample_continuous_roi(
            source,
            boxes,
            output_height=self.expected_local_size,
            output_width=self.expected_local_size,
            frames_per_clip=self.frames_per_clip,
            clips_per_call=self.local_clips_per_call,
        )
        local_features, local_audit = self._encode_view(
            local_view,
            name="local",
        )
        fused, alpha = self.fusion(global_features, local_features)
        if self.training:
            if self._pending_auxiliary is not None:
                raise RuntimeError(
                    "Continuous-RoI auxiliary logits were not consumed exactly once"
                )
            self._pending_auxiliary = {
                "global_logits": self.global_aux_head(global_features),
                "local_logits": self.local_aux_head(local_features),
                "successful_update": int(self._successful_update_index),
            }
        else:
            self._pending_auxiliary = None
        output = deterministic_linear_2x(fused)
        if output.shape != (
            global_view.shape[0],
            global_features.shape[1],
            self.output_length,
        ):
            raise RuntimeError(
                f"Continuous-RoI detector feature contract violated: {tuple(output.shape)}"
            )
        if masks is not None:
            if masks.shape != (output.shape[0], output.shape[-1]):
                raise ValueError("Continuous-RoI mask does not match detector time axis")
            output = output * masks.to(output.device).unsqueeze(1).detach().to(output.dtype)
        self.latest_continuous_roi_audit = {
            "schema_version": CONTINUOUS_ROI_BACKBONE_SCHEMA,
            "geometry_generator_schema": CONTINUOUS_ROI_GENERATOR_SCHEMA,
            "sampler_schema": CONTINUOUS_ROI_SAMPLER_SCHEMA,
            "shared_backbone_instances": 1,
            "videomae_evaluations": 2,
            "contains_selector": False,
            "policy_head_parameters": 0,
            "new_parameters": self.TOTAL_NEW_PARAMETERS,
            "families": list(families),
            "successful_update": self._successful_update_index if self.training else None,
            "knot_logits_shape": list(knot_logits.shape),
            "boxes_shape": list(boxes.shape),
            "box_min": float(boxes.detach().min().item()),
            "box_max": float(boxes.detach().max().item()),
            "alpha_mean": float(alpha.detach().mean().item()),
            "intermediate_shape": list(global_features.shape),
            "output_shape": list(output.shape),
            "global": global_audit,
            "local": local_audit,
            "uses_gt_for_geometry": False,
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_test_evidence": False,
        }
        return output.to(torch.float32)

    def consume_training_auxiliary_losses(
        self,
        *,
        masks: torch.Tensor,
        gt_segments: Sequence[torch.Tensor],
        gt_labels: Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not self.training or self._pending_auxiliary is None:
            raise RuntimeError(
                "Continuous-RoI auxiliary losses require one preceding training forward"
            )
        pending = self._pending_auxiliary
        self._pending_auxiliary = None
        global_logits = pending["global_logits"]
        local_logits = pending["local_logits"]
        target = temporal_class_occupancy_targets(
            gt_segments,
            gt_labels,
            batch_size=global_logits.shape[0],
            num_classes=self.num_classes,
            output_length=global_logits.shape[-1],
            detector_length=self.auxiliary_detector_length,
            device=global_logits.device,
            dtype=global_logits.dtype,
        )
        valid = F.interpolate(
            masks.to(global_logits.device).unsqueeze(1).to(global_logits.dtype),
            size=global_logits.shape[-1],
            mode="nearest",
        )
        denominator = valid.sum().clamp_min(1.0) * float(self.num_classes)
        global_raw = (
            F.binary_cross_entropy_with_logits(
                global_logits,
                target,
                reduction="none",
            )
            * valid
        ).sum() / denominator
        local_raw = (
            F.binary_cross_entropy_with_logits(
                local_logits,
                target,
                reduction="none",
            )
            * valid
        ).sum() / denominator
        global_weight, local_weight = auxiliary_loss_weights(
            pending["successful_update"]
        )
        self.latest_continuous_roi_audit.update(
            {
                "global_aux_raw": float(global_raw.detach().item()),
                "local_aux_raw": float(local_raw.detach().item()),
                "global_aux_weight": global_weight,
                "local_aux_weight": local_weight,
            }
        )
        return {
            "continuous_roi_global_aux_loss": global_weight * global_raw,
            "continuous_roi_local_aux_loss": local_weight * local_raw,
        }
