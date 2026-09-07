from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import torch


@dataclass(frozen=True)
class DucaTemporalSamplingContract:
    """Auditable coordinate and physical-gap contract for offline DUCA."""

    hard_budget: int
    dense_window_size: int
    max_unselected_hole_dense_candidates: int
    dataset_feature_stride_source_frames: int
    dataset_sample_stride: int
    requested_max_source_frame_interval: int
    detector_axis: str = "selected_axis_index"
    dense_axis_unit: str = "dense_candidate_index"
    task: str = "offline_temporal_action_detection"

    def __post_init__(self) -> None:
        integer_fields = (
            "hard_budget",
            "dense_window_size",
            "max_unselected_hole_dense_candidates",
            "dataset_feature_stride_source_frames",
            "dataset_sample_stride",
            "requested_max_source_frame_interval",
        )
        for name in integer_fields:
            value = int(getattr(self, name))
            if value <= 0 and name != "max_unselected_hole_dense_candidates":
                raise ValueError(f"{name} must be positive")
            if name == "max_unselected_hole_dense_candidates" and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.hard_budget > self.dense_window_size:
            raise ValueError("hard_budget cannot exceed dense_window_size")
        if self.task != "offline_temporal_action_detection":
            raise ValueError("DUCA temporal contract is only valid for offline temporal action detection")
        if self.dense_axis_unit != "dense_candidate_index":
            raise ValueError("dense_axis_unit must be dense_candidate_index")
        if self.detector_axis != "selected_axis_index":
            raise ValueError("protected DUCA currently requires the audited selected-axis detector adapter")
        if self.max_selected_interval_source_frames > self.requested_max_source_frame_interval:
            raise ValueError(
                "quantized source-frame interval exceeds requested physical cap: "
                f"{self.max_selected_interval_source_frames} > {self.requested_max_source_frame_interval}"
            )

    @property
    def candidate_stride_source_frames(self) -> int:
        return self.dataset_feature_stride_source_frames * self.dataset_sample_stride

    @property
    def max_selected_interval_dense_steps(self) -> int:
        return self.max_unselected_hole_dense_candidates + 1

    @property
    def max_selected_interval_source_frames(self) -> int:
        return self.max_selected_interval_dense_steps * self.candidate_stride_source_frames

    def max_selected_interval_seconds(self, fps: float) -> float:
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be finite and positive")
        return float(self.max_selected_interval_source_frames) / fps

    def to_dict(self, *, fps: Optional[float] = None) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": "duca_temporal_sampling_contract_v1",
                "candidate_stride_source_frames": self.candidate_stride_source_frames,
                "max_selected_interval_dense_steps": self.max_selected_interval_dense_steps,
                "max_selected_interval_source_frames": self.max_selected_interval_source_frames,
                "max_selected_interval_seconds_formula": (
                    "max_selected_interval_source_frames / video_fps"
                ),
                "selected_positions_runtime_alias": "original_time_index",
                "selected_positions_semantics": "dense_candidate_index",
                "gt_axis_before_remap": "dense_candidate_index",
                "gt_axis_in_detector": self.detector_axis,
                "prediction_remap_target": "dense_candidate_index",
                "inference_teacher_free": True,
                "inference_gt_free": True,
                "inference_cache_free": True,
            }
        )
        if fps is not None:
            payload["video_fps"] = float(fps)
            payload["max_selected_interval_seconds"] = self.max_selected_interval_seconds(fps)
        return payload

    def audit_positions(
        self,
        selected_positions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, Any]:
        if selected_positions.ndim != 2 or valid_mask.ndim != 2:
            raise ValueError("selected_positions and valid_mask must be rank-two")
        if selected_positions.shape[0] != valid_mask.shape[0]:
            raise ValueError("selected_positions and valid_mask batch sizes must match")
        valid = valid_mask.to(device=selected_positions.device, dtype=torch.bool)
        rows = []
        for batch_index in range(int(selected_positions.shape[0])):
            valid_len = int(valid[batch_index].sum().item())
            if valid_len <= 0 or not bool(valid[batch_index, :valid_len].all()):
                raise ValueError("valid_mask must contain a non-empty contiguous prefix")
            if bool(valid[batch_index, valid_len:].any()):
                raise ValueError("valid_mask must contain a non-empty contiguous prefix")
            expected_k = min(self.hard_budget, valid_len)
            row = selected_positions[batch_index]
            active = row[row >= 0].to(dtype=torch.long)
            if int(active.numel()) != expected_k:
                raise ValueError(
                    f"protected DUCA requires K_eff={expected_k}, got {int(active.numel())}"
                )
            if active.numel() > 1 and bool(((active[1:] - active[:-1]) <= 0).any()):
                raise ValueError("selected positions must be unique and strictly increasing")
            if bool((active < 0).any()) or bool((active >= valid_len).any()):
                raise ValueError("selected positions must lie inside the valid dense prefix")
            holes = torch.cat(
                (
                    active[:1],
                    active[1:] - active[:-1] - 1,
                    active.new_tensor([valid_len - int(active[-1].item()) - 1]),
                )
            )
            max_hole = int(holes.max().item())
            if max_hole > self.max_unselected_hole_dense_candidates:
                raise ValueError(
                    "selected positions violate the physical max-gap contract: "
                    f"observed dense hole {max_hole}, cap {self.max_unselected_hole_dense_candidates}"
                )
            rows.append(
                {
                    "valid_dense_candidates": valid_len,
                    "selected_count": expected_k,
                    "max_unselected_hole_dense_candidates": max_hole,
                    "max_selected_interval_dense_steps": max_hole + 1,
                    "max_selected_interval_source_frames": (
                        (max_hole + 1) * self.candidate_stride_source_frames
                    ),
                }
            )
        return {
            "contract": self.to_dict(),
            "rows": rows,
            "passed": True,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DucaTemporalSamplingContract":
        allowed = {
            "hard_budget",
            "dense_window_size",
            "max_unselected_hole_dense_candidates",
            "dataset_feature_stride_source_frames",
            "dataset_sample_stride",
            "requested_max_source_frame_interval",
            "detector_axis",
            "dense_axis_unit",
            "task",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown DUCA temporal contract fields: {unknown}")
        return cls(**{key: value[key] for key in value})


__all__ = ["DucaTemporalSamplingContract"]
