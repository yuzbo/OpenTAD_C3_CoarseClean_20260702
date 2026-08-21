"""Physical-time primitives for sparse VideoMAE observations.

The module keeps the released VideoMAE Conv3D projection as the source of
pretrained weights.  It only changes how the two temporal kernel slices are
combined when the selected observations are not equally spaced.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PhysicalTimeTubeletEmbedding(nn.Module):
    """Apply spatial projection per observation before physical-time mixing.

    For a nominally spaced pair this is exactly the original Conv3D operation:
    ``W0*x0 + W1*x1 + bias``.  A zero-initialized residual can subsequently
    learn how a non-nominal gap changes that pair, while a physical coordinate
    embedding exposes the actual midpoint/gap to the first transformer block.
    """

    def __init__(
        self,
        embed_dims: int,
        *,
        nominal_pair_gap: float = 2.0,
        physical_extent: float = 768.0,
        max_abs_log_gap: float = 4.0,
    ) -> None:
        super().__init__()
        if int(embed_dims) <= 0:
            raise ValueError("embed_dims must be positive")
        if float(nominal_pair_gap) <= 0.0:
            raise ValueError("nominal_pair_gap must be positive")
        if float(physical_extent) <= 1.0:
            raise ValueError("physical_extent must exceed one observation")
        self.embed_dims = int(embed_dims)
        self.nominal_pair_gap = float(nominal_pair_gap)
        self.physical_extent = float(physical_extent)
        self.max_abs_log_gap = float(max_abs_log_gap)

        self.gap_residual_scale = nn.Parameter(torch.zeros(self.embed_dims))
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(2, self.embed_dims),
            nn.GELU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        nn.init.zeros_(self.coordinate_mlp[-1].weight)
        nn.init.zeros_(self.coordinate_mlp[-1].bias)

    @staticmethod
    def _validate_projection(projection: nn.Conv3d) -> None:
        if not isinstance(projection, nn.Conv3d):
            raise TypeError("physical tubelet embedding requires the original Conv3d projection")
        if tuple(projection.kernel_size[:1]) != (2,) or tuple(projection.stride[:1]) != (2,):
            raise ValueError("physical tubelet embedding requires temporal kernel=stride=2")
        if int(projection.dilation[0]) != 1 or int(projection.padding[0]) != 0:
            raise ValueError("physical tubelet embedding requires zero temporal padding and dilation one")

    def forward(
        self,
        x: Tensor,
        source_positions: Tensor,
        valid_mask: Tensor | None,
        projection: nn.Conv3d,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        self._validate_projection(projection)
        if x.ndim != 5 or int(x.shape[2]) <= 0 or int(x.shape[2]) % 2:
            raise ValueError("physical tubelet input must be [B,C,even_T,H,W]")
        batch, _, temporal_len, _, _ = x.shape
        expected = (int(batch), int(temporal_len))
        if tuple(source_positions.shape) != expected:
            raise ValueError(
                f"source_positions must match [B,T]={expected}, got {tuple(source_positions.shape)}"
            )
        source_positions = source_positions.to(device=x.device, dtype=torch.float32)
        if valid_mask is None:
            valid_mask = torch.ones(expected, device=x.device, dtype=torch.bool)
        else:
            if tuple(valid_mask.shape) != expected:
                raise ValueError(f"valid_mask must match [B,T]={expected}")
            valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)

        left_pos = source_positions[:, 0::2]
        right_pos = source_positions[:, 1::2]
        pair_valid = valid_mask[:, 0::2] & valid_mask[:, 1::2]
        gap = right_pos - left_pos
        if bool(torch.any(pair_valid & (gap <= 0)).item()):
            raise ValueError("valid physical tubelet pairs require strictly increasing positions")
        safe_gap = torch.where(pair_valid, gap, gap.new_full(gap.shape, self.nominal_pair_gap))
        log_gap = torch.log(safe_gap / self.nominal_pair_gap).clamp(
            min=-self.max_abs_log_gap,
            max=self.max_abs_log_gap,
        )

        left = x[:, :, 0::2].transpose(1, 2).reshape(-1, int(x.shape[1]), int(x.shape[3]), int(x.shape[4]))
        right = x[:, :, 1::2].transpose(1, 2).reshape_as(left)
        spatial_stride = tuple(int(value) for value in projection.stride[1:])
        spatial_padding = tuple(int(value) for value in projection.padding[1:])
        spatial_dilation = tuple(int(value) for value in projection.dilation[1:])
        left_feat = F.conv2d(
            left,
            projection.weight[:, :, 0],
            bias=projection.bias,
            stride=spatial_stride,
            padding=spatial_padding,
            dilation=spatial_dilation,
            groups=projection.groups,
        )
        right_feat = F.conv2d(
            right,
            projection.weight[:, :, 1],
            bias=None,
            stride=spatial_stride,
            padding=spatial_padding,
            dilation=spatial_dilation,
            groups=projection.groups,
        )
        tubelets = int(temporal_len) // 2
        out_h, out_w = int(left_feat.shape[-2]), int(left_feat.shape[-1])
        left_feat = left_feat.reshape(int(batch), tubelets, self.embed_dims, out_h, out_w)
        right_feat = right_feat.reshape_as(left_feat)
        base = left_feat + right_feat

        gap_factor = torch.tanh(log_gap).view(int(batch), tubelets, 1, 1, 1)
        channel_scale = self.gap_residual_scale.view(1, 1, self.embed_dims, 1, 1)
        output = base + gap_factor * channel_scale * (right_feat - left_feat)

        midpoint = 0.5 * (left_pos + right_pos)
        midpoint_normalized = (2.0 * midpoint / (self.physical_extent - 1.0)) - 1.0
        coordinate_input = torch.stack((midpoint_normalized, log_gap), dim=-1)
        coordinate_embedding = self.coordinate_mlp(coordinate_input).view(
            int(batch), tubelets, self.embed_dims, 1, 1
        )
        output = output + coordinate_embedding
        output = output * pair_valid[:, :, None, None, None].to(dtype=output.dtype)

        tokens = output.permute(0, 1, 3, 4, 2).reshape(int(batch), -1, self.embed_dims)
        support = torch.stack((left_pos, right_pos), dim=-1)
        return tokens, midpoint, support, pair_valid


def physical_gap_scaled_depthwise_conv1d(
    x: Tensor,
    conv: nn.Conv1d,
    positions: Tensor,
    valid_mask: Tensor,
    *,
    nominal_gap: float,
) -> Tensor:
    """Run a kernel-3 depthwise convolution with physical neighbor attenuation.

    The function is exactly ``conv(x)`` when all adjacent physical gaps equal
    ``nominal_gap`` and all observations are valid.  Larger gaps attenuate only
    the cross-gap neighbor term; the center term and learned weights are kept.
    """

    if x.ndim != 3:
        raise ValueError("physical depthwise convolution expects [B,C,T]")
    if tuple(positions.shape) != (int(x.shape[0]), int(x.shape[2])):
        raise ValueError("positions must match the depthwise temporal axis")
    if tuple(valid_mask.shape) != tuple(positions.shape):
        raise ValueError("valid_mask must match positions")
    if conv.groups != int(x.shape[1]) or tuple(conv.kernel_size) != (3,):
        raise ValueError("physical adapter requires a depthwise kernel-3 Conv1d")
    if tuple(conv.dilation) != (1,) or tuple(conv.stride) != (1,) or tuple(conv.padding) != (1,):
        raise ValueError("physical adapter requires stride/dilation one and padding one")
    if float(nominal_gap) <= 0.0:
        raise ValueError("nominal_gap must be positive")

    positions = positions.to(device=x.device, dtype=torch.float32)
    valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)
    if int(x.shape[2]) > 1:
        edge_gap = positions[:, 1:] - positions[:, :-1]
        valid_edge = valid_mask[:, 1:] & valid_mask[:, :-1]
        if bool(torch.any(valid_edge & (edge_gap <= 0)).item()):
            raise ValueError("physical adapter requires increasing valid positions")
        safe_gap = torch.where(valid_edge, edge_gap, edge_gap.new_full(edge_gap.shape, float(nominal_gap)))
        edge_scale = (float(nominal_gap) / safe_gap).clamp(max=1.0)
        edge_scale = edge_scale * valid_edge.to(dtype=edge_scale.dtype)
    else:
        edge_scale = positions.new_zeros((int(x.shape[0]), 0))

    left = F.pad(x[:, :, :-1], (1, 0))
    right = F.pad(x[:, :, 1:], (0, 1))
    left_scale = F.pad(edge_scale, (1, 0)).unsqueeze(1).to(dtype=x.dtype)
    right_scale = F.pad(edge_scale, (0, 1)).unsqueeze(1).to(dtype=x.dtype)
    weight = conv.weight[:, 0]
    output = (
        left * weight[:, 0][None, :, None] * left_scale
        + x * weight[:, 1][None, :, None]
        + right * weight[:, 2][None, :, None] * right_scale
    )
    if conv.bias is not None:
        output = output + conv.bias[None, :, None]
    return output * valid_mask[:, None, :].to(dtype=output.dtype)
