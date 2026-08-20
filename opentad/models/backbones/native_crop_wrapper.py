"""Shared temporal interpolation retained by the supported GeoRoute path."""

from __future__ import annotations

import torch


def deterministic_linear_2x(value: torch.Tensor) -> torch.Tensor:
    """Match 2x linear interpolation with align_corners=False."""

    if value.ndim != 3 or value.shape[-1] < 1:
        raise ValueError("deterministic_linear_2x expects a non-empty [B,C,T] tensor")
    if value.shape[-1] == 1:
        return torch.cat((value, value), dim=-1)
    left = value[..., :-1]
    right = value[..., 1:]
    between = torch.stack(
        (
            0.75 * left + 0.25 * right,
            0.25 * left + 0.75 * right,
        ),
        dim=-1,
    ).flatten(start_dim=-2)
    return torch.cat((value[..., :1], between, value[..., -1:]), dim=-1)
