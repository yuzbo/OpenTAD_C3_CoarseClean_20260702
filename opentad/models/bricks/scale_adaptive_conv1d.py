import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import Parameter, functional as F, init

from .deform_conv1d import deform_conv1d


def _axis_to_time(positions: Tensor, axis: Tensor) -> Tensor:
    """Linear interpolation from selected-axis coordinates to physical time."""

    if positions.ndim != 2:
        raise ValueError("positions must be [B,T]")
    batch, length = positions.shape
    axis = axis.to(device=positions.device, dtype=positions.dtype).clamp(
        min=0.0, max=float(max(length - 1, 0))
    )
    left = axis.floor().to(dtype=torch.long).clamp(min=0, max=length - 1)
    right = (left + 1).clamp(max=length - 1)
    weight = axis - left.to(dtype=positions.dtype)
    row = torch.arange(batch, device=positions.device)[:, None]
    t0 = positions[row, left]
    t1 = positions[row, right]
    return t0 + weight * (t1 - t0)


def inverse_piecewise_linear_1d(positions: Tensor, target_time: Tensor, eps: float = 1.0e-6) -> Tensor:
    """Invert monotone per-sample physical times into selected-axis coordinates."""

    if positions.ndim != 2:
        raise ValueError("positions must be [B,T]")
    if target_time.ndim != 3:
        raise ValueError("target_time must be [B,K,T_out]")
    if positions.shape[0] != target_time.shape[0]:
        raise ValueError("positions and target_time batch sizes differ")
    if positions.shape[1] < 2:
        return torch.zeros_like(target_time)
    if bool(torch.any(positions[:, 1:] < positions[:, :-1]).item()):
        raise ValueError("temporal_positions must be monotonically non-decreasing")

    batch, length = positions.shape
    flat_target = target_time.reshape(batch, -1).to(dtype=positions.dtype)
    clamped = flat_target.clamp(
        min=positions[:, :1],
        max=positions[:, -1:],
    )
    right = torch.searchsorted(positions.contiguous(), clamped.contiguous(), right=True)
    right = right.clamp(min=1, max=length - 1)
    left = right - 1
    row = torch.arange(batch, device=positions.device)[:, None]
    t0 = positions[row, left]
    t1 = positions[row, right]
    denom = (t1 - t0).clamp(min=float(eps))
    frac = (clamped - t0) / denom
    axis = left.to(dtype=positions.dtype) + frac
    return axis.reshape_as(target_time)


class ContinuousTimeScaleAdaptiveConv1d(nn.Module):
    """Residual-gated continuous-time 1D convolution.

    The standard branch is always present. The CT branch averages a local
    sampling grid at delta tau=2 and a context grid at delta tau=2*2^level.
    The learnable gate is initialized to zero, so the layer starts as an exact
    Conv1d equivalent and only departs from it when eta is learned away from 0.
    """

    enable_learned_modulation = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        local_ref_delta_t: float = 2.0,
        context_ref_delta_t: float = 2.0,
        context_level_base: float = 2.0,
        max_abs_offset: Optional[float] = 64.0,
        eps: float = 1.0e-6,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise TypeError(f"unexpected CT conv kwargs: {sorted(kwargs)}")
        if groups != 1:
            raise ValueError("ContinuousTimeScaleAdaptiveConv1d currently supports groups=1")
        if int(kernel_size) <= 0:
            raise ValueError("kernel_size must be positive")
        if int(stride) <= 0 or int(dilation) <= 0:
            raise ValueError("stride and dilation must be positive")
        for name, value in (
            ("local_ref_delta_t", local_ref_delta_t),
            ("context_ref_delta_t", context_ref_delta_t),
            ("context_level_base", context_level_base),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.groups = int(groups)
        self.local_ref_delta_t = float(local_ref_delta_t)
        self.context_ref_delta_t = float(context_ref_delta_t)
        self.context_level_base = float(context_level_base)
        self.max_abs_offset = None if max_abs_offset is None else float(max_abs_offset)
        self.eps = float(eps)

        self.weight = Parameter(torch.empty(out_channels, in_channels // groups, self.kernel_size))
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.eta = Parameter(torch.zeros(()))
        self.last_diagnostics = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _offsets_for_spacing(self, temporal_positions: Tensor, out_len: int, spacing: float) -> Tensor:
        batch, length = temporal_positions.shape
        device = temporal_positions.device
        dtype = temporal_positions.dtype
        kernel_axis = torch.arange(self.kernel_size, device=device, dtype=dtype)
        center = (float(self.kernel_size) - 1.0) * 0.5
        out_axis = torch.arange(out_len, device=device, dtype=dtype)
        center_axis = out_axis * float(self.stride) - float(self.padding) + center * float(self.dilation)
        center_axis = center_axis[None, :].expand(batch, -1)
        center_time = _axis_to_time(temporal_positions, center_axis)

        rel = (kernel_axis - center) * float(self.dilation)
        target_time = center_time[:, None, :] + rel[None, :, None] * float(spacing)
        desired_axis = inverse_piecewise_linear_1d(temporal_positions, target_time, eps=self.eps)
        base_axis = (
            out_axis[None, None, :] * float(self.stride)
            - float(self.padding)
            + kernel_axis[None, :, None] * float(self.dilation)
        )
        offsets = desired_axis - base_axis.to(dtype=dtype, device=device)
        if self.max_abs_offset is not None:
            offsets = offsets.clamp(min=-self.max_abs_offset, max=self.max_abs_offset)
        return offsets

    @staticmethod
    def _positions_from_delta_t(delta_t: Tensor) -> Tensor:
        if delta_t.ndim != 2:
            raise ValueError("delta_t must be [B,T]")
        if delta_t.shape[1] == 0:
            return delta_t
        first = delta_t.new_zeros(delta_t.shape[0], 1)
        tail = torch.cumsum(delta_t[:, :-1].clamp_min(0.0), dim=1)
        return torch.cat([first, tail], dim=1)

    def _update_diagnostics(
        self,
        temporal_positions: Optional[Tensor],
        offsets_local: Optional[Tensor],
        offsets_context: Optional[Tensor],
        level_index: Optional[int],
        used_ct: bool,
    ) -> None:
        with torch.no_grad():
            diag = {
                "enabled": bool(used_ct),
                "eta": float(self.eta.detach().float().item()),
                "level_index": None if level_index is None else int(level_index),
                "local_ref_delta_t": float(self.local_ref_delta_t),
                "context_ref_delta_t": (
                    float(self.context_ref_delta_t)
                    if level_index is None
                    else float(self.context_ref_delta_t * (self.context_level_base ** int(level_index)))
                ),
            }
            if temporal_positions is not None and temporal_positions.numel() > 1:
                diag["temporal_position_min"] = float(temporal_positions.detach().amin().item())
                diag["temporal_position_max"] = float(temporal_positions.detach().amax().item())
            if offsets_local is not None and offsets_context is not None:
                merged = torch.cat(
                    [
                        offsets_local.detach().float().reshape(-1).abs(),
                        offsets_context.detach().float().reshape(-1).abs(),
                    ],
                    dim=0,
                )
                if merged.numel() > 0:
                    diag["offset_abs_mean"] = float(merged.mean().item())
                    diag["offset_abs_p95"] = float(torch.quantile(merged, 0.95).item())
                    diag["offset_abs_max"] = float(merged.max().item())
            self.last_diagnostics = diag

    def forward(
        self,
        input: Tensor,
        delta_t: Optional[Tensor] = None,
        temporal_positions: Optional[Tensor] = None,
        level_index: Optional[int] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        y_std = F.conv1d(
            input,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if temporal_positions is None and delta_t is not None:
            temporal_positions = self._positions_from_delta_t(delta_t)
        if temporal_positions is None:
            self._update_diagnostics(None, None, None, level_index, used_ct=False)
            return y_std
        if temporal_positions.ndim == 1:
            temporal_positions = temporal_positions[None, :].expand(input.shape[0], -1)
        temporal_positions = temporal_positions.to(device=input.device, dtype=input.dtype)
        if temporal_positions.shape[0] != input.shape[0] or temporal_positions.shape[1] != input.shape[-1]:
            raise ValueError(
                "temporal_positions must match Conv1d input as [B,T_in]; "
                f"got {tuple(temporal_positions.shape)} for input {tuple(input.shape)}"
            )
        if temporal_positions.shape[1] < 2:
            self._update_diagnostics(temporal_positions, None, None, level_index, used_ct=False)
            return y_std

        level = 0 if level_index is None else int(level_index)
        context_spacing = self.context_ref_delta_t * (self.context_level_base ** level)
        offsets_local = self._offsets_for_spacing(
            temporal_positions,
            out_len=y_std.shape[-1],
            spacing=self.local_ref_delta_t,
        )
        offsets_context = self._offsets_for_spacing(
            temporal_positions,
            out_len=y_std.shape[-1],
            spacing=context_spacing,
        )
        y_local = deform_conv1d(
            input,
            offsets_local,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        y_context = deform_conv1d(
            input,
            offsets_context,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        y_ct = 0.5 * (y_local + y_context)
        self._update_diagnostics(
            temporal_positions,
            offsets_local,
            offsets_context,
            level_index,
            used_ct=True,
        )
        return y_std + self.eta.to(dtype=y_std.dtype) * (y_ct - y_std)
