from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


SELECTED_AXIS = "selected_axis_index"
TRUE_TIME_AXIS = "true_time_dense_index"


def _as_float_tensor(values, *, device=None):
    if torch.is_tensor(values):
        return values.to(device=device, dtype=torch.float32) if device is not None else values.to(dtype=torch.float32)
    return torch.as_tensor(values, dtype=torch.float32, device=device)


@dataclass(frozen=True)
class TrueTimeMap:
    """Map detector selected-axis coordinates back to original dense-time coordinates."""

    selected_positions: object
    dense_len: int
    valid_len: int | None = None
    selected_axis_name: str = SELECTED_AXIS
    true_time_axis_name: str = TRUE_TIME_AXIS

    def __post_init__(self):
        positions = _as_float_tensor(self.selected_positions)
        if positions.ndim != 1:
            raise ValueError("selected_positions must be a 1D sequence")
        if int(positions.numel()) == 0:
            raise ValueError("selected_positions must be non-empty")
        dense_len = int(self.dense_len)
        valid_len = dense_len if self.valid_len is None else int(self.valid_len)
        if dense_len <= 0 or valid_len <= 0:
            raise ValueError("dense_len and valid_len must be positive")
        if valid_len > dense_len:
            raise ValueError("valid_len must not exceed dense_len")
        if not bool(torch.isfinite(positions).all().item()):
            raise ValueError("selected_positions must be finite")
        if bool((positions < 0).any().item()) or bool((positions >= valid_len).any().item()):
            raise ValueError("selected_positions must stay inside valid_len")
        if positions.numel() > 1 and bool((positions[1:] <= positions[:-1]).any().item()):
            raise ValueError("selected_positions must be strictly increasing")
        object.__setattr__(self, "selected_positions", positions)
        object.__setattr__(self, "dense_len", dense_len)
        object.__setattr__(self, "valid_len", valid_len)

    @property
    def selected_len(self) -> int:
        return int(self.selected_positions.numel())

    def selected_to_true(self, values):
        values_t = _as_float_tensor(values, device=self.selected_positions.device)
        if self.selected_len == 1:
            return torch.zeros_like(values_t) + self.selected_positions[0]

        clipped = values_t.clamp(0.0, float(self.selected_len - 1))
        left = torch.floor(clipped).long()
        right = torch.clamp(left + 1, max=self.selected_len - 1)
        frac = clipped - left.to(dtype=clipped.dtype)
        left_pos = self.selected_positions[left]
        right_pos = self.selected_positions[right]
        return left_pos + (right_pos - left_pos) * frac

    def true_to_selected(self, values):
        values_t = _as_float_tensor(values, device=self.selected_positions.device)
        if self.selected_len == 1:
            return torch.zeros_like(values_t)

        clipped = values_t.clamp(float(self.selected_positions[0].item()), float(self.selected_positions[-1].item()))
        right = torch.searchsorted(self.selected_positions, clipped, right=False)
        right = torch.clamp(right, min=1, max=self.selected_len - 1)
        left = right - 1
        left_pos = self.selected_positions[left]
        right_pos = self.selected_positions[right]
        denom = (right_pos - left_pos).clamp_min(1.0e-6)
        frac = (clipped - left_pos) / denom
        return left.to(dtype=clipped.dtype) + frac

    def remap_segments(self, segments, *, source_coordinate_space: str, target_coordinate_space: str):
        if source_coordinate_space == target_coordinate_space:
            return _as_float_tensor(segments, device=self.selected_positions.device).clone()
        if source_coordinate_space == self.selected_axis_name and target_coordinate_space == self.true_time_axis_name:
            return self.selected_to_true(segments)
        if source_coordinate_space == self.true_time_axis_name and target_coordinate_space == self.selected_axis_name:
            return self.true_to_selected(segments)
        raise ValueError(
            "unsupported segment remap: "
            f"{source_coordinate_space!r} -> {target_coordinate_space!r}; "
            f"expected {self.selected_axis_name!r} or {self.true_time_axis_name!r}"
        )


def inverse_map_prediction_segments(predictions: Mapping, time_map: TrueTimeMap, *, segment_key: str = "segments"):
    source_space = predictions.get("coordinate_space")
    if source_space != time_map.selected_axis_name:
        raise ValueError(
            "prediction inverse-map requires detector outputs in selected-axis coordinates; "
            f"got {source_space!r}"
        )
    if segment_key not in predictions:
        raise ValueError(f"predictions missing segment key {segment_key!r}")

    mapped = dict(predictions)
    mapped[segment_key] = time_map.remap_segments(
        predictions[segment_key],
        source_coordinate_space=time_map.selected_axis_name,
        target_coordinate_space=time_map.true_time_axis_name,
    )
    mapped["source_coordinate_space"] = source_space
    mapped["coordinate_space"] = time_map.true_time_axis_name
    mapped["true_time_map"] = {
        "selected_positions": [int(item) for item in time_map.selected_positions.long().tolist()],
        "dense_len": int(time_map.dense_len),
        "valid_len": int(time_map.valid_len),
        "selected_axis_name": time_map.selected_axis_name,
        "true_time_axis_name": time_map.true_time_axis_name,
    }
    return mapped
