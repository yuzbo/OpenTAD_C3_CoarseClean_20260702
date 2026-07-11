import math

import torch


_AUDITED_SUPPORT_PROVENANCE = {
    "original_feature_ownership_cells",
    "original_raw_dense_cells",
    "contiguous_decoded_clips",
    "synthetic_explicit_support",
}


def _check_shape(tensor, shape, name):
    if tensor.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions, got {tuple(tensor.shape)}")
    for axis, expected in enumerate(shape):
        if expected is not None and tensor.shape[axis] != expected:
            raise ValueError(f"{name} shape mismatch at axis {axis}: expected {expected}, got {tensor.shape[axis]}")


def validate_physical_observations(timestamps_sec, support_intervals_sec, valid_mask, duration_sec):
    """Validate an explicit irregular-observation geometry in canonical seconds."""
    _check_shape(timestamps_sec, (None, None), "timestamps_sec")
    batch_size, observation_count = timestamps_sec.shape
    _check_shape(support_intervals_sec, (batch_size, observation_count, 2), "support_intervals_sec")
    _check_shape(valid_mask, (batch_size, observation_count), "valid_mask")
    _check_shape(duration_sec, (batch_size,), "duration_sec")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean")
    if not torch.isfinite(duration_sec).all() or torch.any(duration_sec <= 0):
        raise ValueError("duration_sec must be finite and positive")

    for batch_idx in range(batch_size):
        mask = valid_mask[batch_idx]
        valid_count = int(mask.sum().item())
        expected_mask = torch.arange(observation_count, device=mask.device) < valid_count
        if not torch.equal(mask, expected_mask):
            raise ValueError("valid observations must form a prefix; holes belong in support geometry, not padding")
        if valid_count == 0:
            raise ValueError("each sample must contain at least one valid observation")

        timestamps = timestamps_sec[batch_idx, :valid_count]
        supports = support_intervals_sec[batch_idx, :valid_count]
        if not torch.isfinite(timestamps).all() or not torch.isfinite(supports).all():
            raise ValueError("valid timestamps and support intervals must be finite")
        if valid_count > 1 and torch.any(timestamps[1:] <= timestamps[:-1]):
            raise ValueError("valid timestamps must be strictly increasing")

        left, right = supports.unbind(dim=-1)
        if torch.any(right <= left):
            raise ValueError("every valid support interval must be non-empty")
        eps = torch.finfo(timestamps.dtype).eps * 16
        if torch.any(timestamps < left - eps) or torch.any(timestamps > right + eps):
            raise ValueError("every valid support interval must contain its timestamp")
        duration = duration_sec[batch_idx]
        if torch.any(left < -eps) or torch.any(right > duration + eps):
            raise ValueError("support intervals must lie within the video duration")
    return None


def clip_to_ownership_intervals(timestamps_sec, support_intervals_sec, valid_mask, duration_sec):
    """Remove support overlap without ever expanding a supplied support interval."""
    validate_physical_observations(timestamps_sec, support_intervals_sec, valid_mask, duration_sec)
    ownership = torch.zeros_like(support_intervals_sec)
    for batch_idx in range(timestamps_sec.shape[0]):
        valid_count = int(valid_mask[batch_idx].sum().item())
        timestamps = timestamps_sec[batch_idx, :valid_count]
        supports = support_intervals_sec[batch_idx, :valid_count]
        left = supports[:, 0].clone()
        right = supports[:, 1].clone()
        if valid_count > 1:
            midpoints = 0.5 * (timestamps[:-1] + timestamps[1:])
            right[:-1] = torch.minimum(right[:-1], midpoints)
            left[1:] = torch.maximum(left[1:], midpoints)
        if torch.any(right <= left):
            raise ValueError("ownership clipping produced an empty support; input supports are inconsistent")
        ownership[batch_idx, :valid_count, 0] = left
        ownership[batch_idx, :valid_count, 1] = right
    return ownership


def support_overlap_mass(ownership_intervals_sec, query_intervals_sec, observation_mask):
    """Return physical overlap mass with shape [B, Q, K]."""
    _check_shape(ownership_intervals_sec, (None, None, 2), "ownership_intervals_sec")
    batch_size, observation_count, _ = ownership_intervals_sec.shape
    _check_shape(query_intervals_sec, (batch_size, None, 2), "query_intervals_sec")
    _check_shape(observation_mask, (batch_size, observation_count), "observation_mask")

    left = torch.maximum(
        query_intervals_sec[:, :, None, 0],
        ownership_intervals_sec[:, None, :, 0],
    )
    right = torch.minimum(
        query_intervals_sec[:, :, None, 1],
        ownership_intervals_sec[:, None, :, 1],
    )
    mass = (right - left).clamp_min(0)
    return mass * observation_mask[:, None, :].to(dtype=mass.dtype)


def build_physical_query_pyramid(
    duration_sec,
    domain_start_sec,
    domain_end_sec,
    *,
    base_spacing_sec,
    num_levels,
):
    """Build globally aligned physical query cells independent of observation count."""
    _check_shape(duration_sec, (None,), "duration_sec")
    batch_size = duration_sec.shape[0]
    _check_shape(domain_start_sec, (batch_size,), "domain_start_sec")
    _check_shape(domain_end_sec, (batch_size,), "domain_end_sec")
    if not math.isfinite(float(base_spacing_sec)) or float(base_spacing_sec) <= 0:
        raise ValueError("base_spacing_sec must be finite and positive")
    if int(num_levels) <= 0:
        raise ValueError("num_levels must be positive")
    if not torch.isfinite(domain_start_sec).all() or not torch.isfinite(domain_end_sec).all():
        raise ValueError("query domains must be finite")
    if torch.any(domain_start_sec < 0) or torch.any(domain_end_sec > duration_sec):
        raise ValueError("query domains must lie within duration_sec")
    if torch.any(domain_end_sec <= domain_start_sec):
        raise ValueError("query domains must be non-empty")

    pyramid = []
    for level in range(int(num_levels)):
        width = float(base_spacing_sec) * (2**level)
        start_indices = torch.floor(domain_start_sec / width).to(torch.long)
        end_indices = torch.ceil(domain_end_sec / width).to(torch.long)
        counts = end_indices - start_indices
        max_count = int(counts.max().item())
        slot = torch.arange(max_count, device=duration_sec.device, dtype=torch.long)
        cell_index = start_indices[:, None] + slot[None, :]
        valid_mask = slot[None, :] < counts[:, None]

        raw_left = cell_index.to(duration_sec.dtype) * width
        raw_right = raw_left + width
        left = raw_left.clamp_min(0)
        right = torch.minimum(raw_right, duration_sec[:, None])
        valid_mask = valid_mask & (right > left)
        intervals = torch.stack((left, right), dim=-1)
        intervals = intervals * valid_mask[:, :, None].to(intervals.dtype)
        centers = 0.5 * (intervals[..., 0] + intervals[..., 1])
        widths = (intervals[..., 1] - intervals[..., 0]) * valid_mask.to(intervals.dtype)
        pyramid.append(
            {
                "level": level,
                "spacing_sec": width,
                "centers_sec": centers,
                "intervals_sec": intervals,
                "widths_sec": widths,
                "valid_mask": valid_mask,
            }
        )
    return pyramid


def _meta_tensor(meta, key, *, dtype, device):
    if key not in meta:
        raise ValueError(f"PhysTime metadata is missing required key: {key}")
    return torch.as_tensor(meta[key], dtype=dtype, device=device)


def geometry_from_metas(metas, valid_mask, *, dtype, device):
    """Batch strict PhysTime metadata without interpreting selected-token rank as time."""
    if not isinstance(metas, (list, tuple)) or len(metas) != valid_mask.shape[0]:
        raise ValueError("metas must contain one dictionary per batch sample")
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.to(dtype=torch.bool)
    valid_mask = valid_mask.to(device=device)
    batch_size, max_observations = valid_mask.shape
    timestamps = torch.zeros((batch_size, max_observations), dtype=dtype, device=device)
    supports = torch.zeros((batch_size, max_observations, 2), dtype=dtype, device=device)
    durations = torch.zeros((batch_size,), dtype=dtype, device=device)
    domain_start = torch.zeros_like(durations)
    domain_end = torch.zeros_like(durations)

    forbidden_gt_flags = (
        "remap_gt_to_selected_axis",
        "gt_remapped_to_selected_axis",
        "pc_ot_mras_prebackbone_remap_gt_to_selected_axis",
    )
    for batch_idx, meta in enumerate(metas):
        if not isinstance(meta, dict):
            raise ValueError("each PhysTime metadata entry must be a dictionary")
        provenance = meta.get("phystime_support_provenance")
        if provenance not in _AUDITED_SUPPORT_PROVENANCE:
            raise ValueError(f"unaudited PhysTime support provenance: {provenance!r}")
        if any(bool(meta.get(key, False)) for key in forbidden_gt_flags):
            raise ValueError("PhysTime forbids selected-axis GT remapping")

        count = int(valid_mask[batch_idx].sum().item())
        expected_mask = torch.arange(max_observations, device=device) < count
        if not torch.equal(valid_mask[batch_idx], expected_mask):
            raise ValueError("PhysTime padding mask must be a valid prefix")
        sample_timestamps = _meta_tensor(
            meta, "phystime_timestamps_sec", dtype=dtype, device=device
        ).reshape(-1)
        sample_supports = _meta_tensor(
            meta, "phystime_support_intervals_sec", dtype=dtype, device=device
        ).reshape(-1, 2)
        if sample_timestamps.numel() != count or sample_supports.shape[0] != count:
            raise ValueError(
                "PhysTime metadata count must equal the number of valid observation tokens"
            )
        timestamps[batch_idx, :count] = sample_timestamps
        supports[batch_idx, :count] = sample_supports
        durations[batch_idx] = _meta_tensor(
            meta, "phystime_duration_sec", dtype=dtype, device=device
        ).reshape(-1)[0]
        domain_start[batch_idx] = _meta_tensor(
            meta, "phystime_domain_start_sec", dtype=dtype, device=device
        ).reshape(-1)[0]
        domain_end[batch_idx] = _meta_tensor(
            meta, "phystime_domain_end_sec", dtype=dtype, device=device
        ).reshape(-1)[0]

    validate_physical_observations(timestamps, supports, valid_mask, durations)
    if torch.any(domain_start < 0) or torch.any(domain_end > durations) or torch.any(domain_end <= domain_start):
        raise ValueError("PhysTime query domains must be non-empty and lie inside duration")
    ownership = clip_to_ownership_intervals(timestamps, supports, valid_mask, durations)
    return {
        "timestamps_sec": timestamps,
        "support_intervals_sec": supports,
        "ownership_intervals_sec": ownership,
        "valid_mask": valid_mask,
        "duration_sec": durations,
        "domain_start_sec": domain_start,
        "domain_end_sec": domain_end,
    }
