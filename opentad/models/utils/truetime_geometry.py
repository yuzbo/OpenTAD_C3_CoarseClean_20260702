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


def _metadata_scalar(value, *, key: str):
    if torch.is_tensor(value):
        if int(value.numel()) != 1:
            raise ValueError(f"{key} must be scalar, got tensor shape {tuple(value.shape)}")
        return value.detach().cpu().reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{key} must be scalar, got sequence length {len(value)}")
        return value[0]
    return value


def _metadata_int_value(value, *, key: str) -> int:
    value = _metadata_scalar(value, key=key)
    return int(round(float(value)))


def _metadata_int(metadata: Mapping, keys: tuple[str, ...], *, default: int | None = None) -> int:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return _metadata_int_value(value, key=key)
    if default is None:
        raise ValueError(f"metadata missing required integer field; expected one of {keys!r}")
    return int(default)


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

    def _interpolation_knots(self):
        selected_axis = torch.arange(
            self.selected_len,
            device=self.selected_positions.device,
            dtype=self.selected_positions.dtype,
        )
        true_time = self.selected_positions
        if float(true_time[0].item()) > 0.0:
            selected_axis = torch.cat((selected_axis.new_tensor([-1.0]), selected_axis))
            true_time = torch.cat((true_time.new_tensor([0.0]), true_time))
        if float(true_time[-1].item()) < float(self.valid_len):
            selected_axis = torch.cat((selected_axis, selected_axis.new_tensor([float(self.selected_len)])))
            true_time = torch.cat((true_time, true_time.new_tensor([float(self.valid_len)])))
        return selected_axis, true_time

    @staticmethod
    def _piecewise_linear(values, source_knots, target_knots):
        clipped = values.clamp(float(source_knots[0].item()), float(source_knots[-1].item()))
        right = torch.searchsorted(source_knots, clipped, right=False)
        right = torch.clamp(right, min=1, max=int(source_knots.numel()) - 1)
        left = right - 1
        left_source = source_knots[left]
        right_source = source_knots[right]
        fraction = (clipped - left_source) / (right_source - left_source).clamp_min(1.0e-6)
        return target_knots[left] + (target_knots[right] - target_knots[left]) * fraction

    def selected_to_true(self, values):
        values_t = _as_float_tensor(values, device=self.selected_positions.device)
        selected_axis, true_time = self._interpolation_knots()
        return self._piecewise_linear(values_t, selected_axis, true_time)

    def true_to_selected(self, values):
        values_t = _as_float_tensor(values, device=self.selected_positions.device)
        selected_axis, true_time = self._interpolation_knots()
        return self._piecewise_linear(values_t, true_time, selected_axis)

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


def truetime_map_from_metadata(metadata: Mapping, *, require_inverse_map: bool | None = None) -> TrueTimeMap:
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    remap_required = (
        bool(metadata.get("detector_prediction_inverse_map_required", False))
        if require_inverse_map is None
        else bool(require_inverse_map)
    )
    positions = metadata.get("selected_axis_to_true_time_dense_index")
    if positions is None:
        message = "metadata missing selected_axis_to_true_time_dense_index"
        if remap_required:
            raise ValueError(f"{message} while detector_prediction_inverse_map_required is true")
        raise ValueError(message)

    positions_t = _as_float_tensor(positions)
    if positions_t.ndim != 1 or int(positions_t.numel()) == 0:
        raise ValueError("selected_axis_to_true_time_dense_index must be a non-empty 1D sequence")

    dense_valid_len = _metadata_int(
        metadata,
        ("irregular_dense_valid_len", "truetime_dense_valid_len", "valid_len"),
        default=int(positions_t.max().item()) + 1,
    )
    dense_len = _metadata_int(
        metadata,
        ("truetime_dense_len", "dense_len", "window_size"),
        default=max(dense_valid_len, int(positions_t.max().item()) + 1),
    )

    selected_count = metadata.get("irregular_selected_count")
    selected_valid_len = metadata.get("irregular_selected_valid_len")
    if selected_count is not None and _metadata_int_value(selected_count, key="irregular_selected_count") != int(positions_t.numel()):
        raise ValueError("irregular_selected_count must match selected_axis_to_true_time_dense_index length")
    if selected_valid_len is not None and _metadata_int_value(selected_valid_len, key="irregular_selected_valid_len") != int(positions_t.numel()):
        raise ValueError("irregular_selected_valid_len must match selected_axis_to_true_time_dense_index length")

    return TrueTimeMap(selected_positions=positions_t, dense_len=dense_len, valid_len=dense_valid_len)


def remap_selected_axis_segments_to_true_time(segments, metadata: Mapping):
    time_map = truetime_map_from_metadata(metadata, require_inverse_map=True)
    return time_map.remap_segments(
        segments,
        source_coordinate_space=time_map.selected_axis_name,
        target_coordinate_space=time_map.true_time_axis_name,
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
        "selected_positions": [float(item) for item in time_map.selected_positions.tolist()],
        "dense_len": int(time_map.dense_len),
        "valid_len": int(time_map.valid_len),
        "selected_axis_name": time_map.selected_axis_name,
        "true_time_axis_name": time_map.true_time_axis_name,
    }
    return mapped
