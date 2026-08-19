from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class GeometryValueTarget:
    frame_target: torch.Tensor
    pair_weight: torch.Tensor
    audit: dict[str, Any] = field(default_factory=dict)


def _segments_as_list(gt_segments):
    if torch.is_tensor(gt_segments):
        return [row for row in gt_segments]
    return list(gt_segments)


def build_geometry_value_target(
    *,
    gt_segments,
    gt_labels,
    valid_mask: torch.Tensor,
    boundary_radius: int,
    short_action_duration_sec: float,
    short_action_weight: float,
) -> GeometryValueTarget:
    if not torch.is_tensor(valid_mask) or valid_mask.ndim != 2:
        raise ValueError("valid_mask must be [B,T]")
    valid = valid_mask.bool()
    batch, length = valid.shape
    if int(boundary_radius) < 0:
        raise ValueError("boundary_radius must be non-negative")
    if float(short_action_duration_sec) <= 0.0:
        raise ValueError("short_action_duration_sec must be positive")
    if float(short_action_weight) < 0.0:
        raise ValueError("short_action_weight must be non-negative")

    device = valid.device
    raw_target = torch.zeros((batch, length), dtype=torch.float32, device=device)
    pair_weight = torch.ones((batch, length), dtype=torch.float32, device=device)
    idx = torch.arange(length, device=device, dtype=torch.float32)

    endpoint_count = 0
    interior_count = 0
    background_count = 0
    short_segment_count = 0
    action_mask = torch.zeros((batch, length), dtype=torch.bool, device=device)

    for b, segments in enumerate(_segments_as_list(gt_segments)):
        if segments is None:
            continue
        seg = torch.as_tensor(segments, device=device, dtype=torch.float32).reshape(-1, 2)
        for start, end in seg:
            duration = float((end - start).item())
            short_factor = float(max(0.0, min(1.0, 1.0 - duration / float(short_action_duration_sec))))
            if short_factor > 0.0:
                short_segment_count += 1
            inside = (idx >= start) & (idx <= end)
            dist_start = (idx - start).abs()
            dist_end = (idx - end).abs()
            endpoint = inside & (torch.minimum(dist_start, dist_end) <= float(boundary_radius))
            interior = inside & ~endpoint
            raw_target[b, endpoint] = 1.0
            raw_target[b, interior] = 0.0
            action_mask[b] = action_mask[b] | inside
            weight = 1.0 + float(short_action_weight) * short_factor
            pair_weight[b, inside] = torch.maximum(pair_weight[b, inside], torch.full_like(pair_weight[b, inside], weight))

    raw_target = raw_target.masked_fill(~action_mask & valid, -1.0)

    endpoint_count += int(((raw_target > 0.5) & valid).sum().item())
    interior_count += int(((raw_target.abs() <= 0.5) & valid).sum().item())
    background_count += int(((raw_target < -0.5) & valid).sum().item())

    denom = valid.to(torch.float32).sum(dim=1).clamp_min(1.0)
    centered = raw_target - (raw_target.masked_fill(~valid, 0.0).sum(dim=1) / denom)[:, None]
    centered = centered.masked_fill(~valid, 0.0)
    pair_weight = pair_weight.masked_fill(~valid, 0.0)

    audit = {
        "target_kind": "gt_geometry_signed_residual_v0",
        "boundary_radius": int(boundary_radius),
        "short_action_duration_sec": float(short_action_duration_sec),
        "short_action_weight": float(short_action_weight),
        "raw_endpoint_positives": endpoint_count,
        "raw_interior_zeros": interior_count,
        "raw_background_negatives": background_count,
        "short_segment_count": short_segment_count,
        "uses_gt": True,
        "inference_visible": False,
    }
    return GeometryValueTarget(frame_target=centered, pair_weight=pair_weight, audit=audit)
