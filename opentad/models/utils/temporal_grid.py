from collections.abc import Mapping

import torch


def pjst_pair_metadata(selected_positions, dense_valid_len, prefix_mask):
    """Build PJST-D1 global K384 pair metadata without float time arithmetic."""
    if not torch.is_tensor(selected_positions) or selected_positions.ndim != 2:
        raise ValueError("selected_positions must be [B,K]")
    if selected_positions.dtype != torch.int64:
        raise ValueError("selected_positions must be int64")
    if selected_positions.shape[1] != 384:
        raise ValueError("PJST-D1 requires K=384")
    if prefix_mask.shape != selected_positions.shape or prefix_mask.dtype != torch.bool:
        raise ValueError("prefix_mask must be bool [B,384]")
    lengths = torch.as_tensor(dense_valid_len, device=selected_positions.device, dtype=torch.int64).flatten()
    if lengths.numel() != selected_positions.shape[0]:
        raise ValueError("dense_valid_len must be [B]")
    valid_counts = prefix_mask.sum(1)
    canonical = torch.zeros_like(selected_positions)
    from opentad.models.duca.structured_selection import exact_uniform_positions
    for b in range(selected_positions.shape[0]):
        n = int(valid_counts[b].item())
        if n <= 0 or not torch.equal(prefix_mask[b], torch.arange(384, device=prefix_mask.device) < n):
            raise ValueError("prefix_mask must be a non-empty contiguous prefix")
        p = selected_positions[b, :n]
        if lengths[b] <= 0 or (p < 0).any() or (p >= lengths[b]).any() or (p[1:] <= p[:-1]).any():
            raise ValueError("selected positions must be in-range and strictly increasing")
        canonical[b, :n] = exact_uniform_positions(int(lengths[b]), n, device=selected_positions.device)
    exact_uniform = torch.stack([torch.equal(selected_positions[b, :int(valid_counts[b])], canonical[b, :int(valid_counts[b])]) for b in range(selected_positions.shape[0])]).to(selected_positions.device)
    p0 = selected_positions[:, 0::2].reshape(-1, 8)
    p1 = selected_positions[:, 1::2].reshape(-1, 8)
    u0 = canonical[:, 0::2].reshape(-1, 8)
    u1 = canonical[:, 1::2].reshape(-1, 8)
    pair_valid = (prefix_mask[:, 0::2] & prefix_mask[:, 1::2]).reshape(-1, 8)
    actual_delta = p1 - p0
    canonical_delta = u1 - u0
    if ((actual_delta[pair_valid] <= 0) | (canonical_delta[pair_valid] <= 0)).any():
        raise ValueError("valid pair deltas must be positive")
    scale = torch.ones_like(actual_delta, dtype=torch.float32)
    irregular = ~exact_uniform.repeat_interleave(24)
    valid_irregular = pair_valid & irregular[:, None]
    scale[valid_irregular] = canonical_delta[valid_irregular].float() / actual_delta[valid_irregular].float()
    return {"pair_scale": scale, "pair_valid": pair_valid, "exact_uniform_identity": exact_uniform.repeat_interleave(24), "actual_delta": actual_delta, "canonical_delta": canonical_delta}


def apply_pjst_derivative_only(x, pair_scale, pair_valid, exact_uniform_identity):
    """Apply derivative-only PJST-D1 to packed [Bclips,3,16,H,W] input."""
    if x.ndim != 5 or x.shape[2] != 16 or pair_scale.shape != (x.shape[0], 8):
        raise ValueError("expected packed clips and [Bclips,8] metadata")
    y = x.clone()
    rows = (~exact_uniform_identity)[:, None] & pair_valid
    if rows.any():
        z = x.float().reshape(x.shape[0], 3, 8, 2, *x.shape[-2:])
        m = (z[:, :, :, 0] + z[:, :, :, 1]) / 2
        v = pair_scale.float()[:, None, :, None, None] * (z[:, :, :, 1] - z[:, :, :, 0]) / 2
        out = torch.stack((m - v, m + v), dim=3)
        y[rows.any(1)] = out[rows.any(1)].reshape(-1, 3, 16, *x.shape[-2:]).to(x.dtype)
    return y


def global_rank_clip_coordinates(
    irregular_selected_positions,
    dense_valid_len,
    *,
    selected_valid_mask=None,
    k=384,
    clip_len=16,
    tubelet_size=2,
):
    """Derive global-rank canonical/actual coordinates, then pack into clips.

    Selection is intentionally not performed here: callers provide the one global
    strictly increasing rank sequence. Canonical anchors are generated once per
    sample over the full dense window.
    """
    from opentad.models.duca.structured_selection import exact_uniform_positions
    positions = irregular_selected_positions
    if not torch.is_tensor(positions) or positions.ndim != 2:
        raise ValueError("irregular_selected_positions must be [B,K]")
    if positions.shape[1] != int(k):
        raise ValueError("global selected positions must have exactly K entries")
    lengths = dense_valid_len if torch.is_tensor(dense_valid_len) else torch.as_tensor(dense_valid_len, device=positions.device)
    lengths = lengths.to(device=positions.device, dtype=torch.long).flatten()
    if lengths.numel() != positions.shape[0]:
        raise ValueError("dense_valid_len must be [B]")
    if selected_valid_mask is None:
        slot_mask = positions >= 0
    else:
        if not torch.is_tensor(selected_valid_mask) or selected_valid_mask.shape != positions.shape:
            raise ValueError("selected_valid_mask must match irregular_selected_positions [B,K]")
        slot_mask = selected_valid_mask.to(device=positions.device, dtype=torch.bool)
        if not torch.equal(slot_mask, positions >= 0):
            raise ValueError("selected_valid_mask must exactly identify non-negative selected positions")
    valid_counts = slot_mask.long().sum(dim=1)
    if (valid_counts <= 0).any():
        raise ValueError("each sample must contain at least one valid selected position")
    canonical = torch.zeros_like(positions)
    sanitized = torch.zeros_like(positions)
    for batch_idx in range(positions.shape[0]):
        count = int(valid_counts[batch_idx].item())
        if not bool(slot_mask[batch_idx, :count].all()) or bool(slot_mask[batch_idx, count:].any()):
            raise ValueError("selected_valid_mask must be a contiguous valid prefix")
        valid_positions = positions[batch_idx, :count]
        length = int(lengths[batch_idx].item())
        if length <= 0 or count > length:
            raise ValueError("valid selected count must be in [1, dense_valid_len]")
        if count > 1 and bool((valid_positions[1:] <= valid_positions[:-1]).any()):
            raise ValueError("valid selected positions must be strictly increasing")
        if bool((valid_positions < 0).any()) or bool((valid_positions >= length).any()):
            raise ValueError("valid selected positions out of dense window")
        sanitized[batch_idx, :count] = valid_positions
        canonical[batch_idx, :count] = exact_uniform_positions(length, count, device=positions.device)
    if int(k) % int(clip_len) != 0:
        raise ValueError("K must be divisible by clip_len")
    clips = int(k) // int(clip_len)
    actual = sanitized.reshape(positions.shape[0], clips, int(clip_len))
    canon = canonical.reshape(canonical.shape[0], clips, int(clip_len))
    frame_valid = slot_mask.reshape(slot_mask.shape[0], clips, int(clip_len))
    if int(tubelet_size) > 1:
        if int(clip_len) % int(tubelet_size):
            raise ValueError("clip_len must be divisible by tubelet_size")
        # Coordinates may remain long physical dense indices in metadata.  Cast
        # only the attention tubelet centers so integer/half-integer means work;
        # preserve the raw selected positions in the returned metadata.
        actual = actual.to(dtype=torch.float32).reshape(actual.shape[0], clips, -1, int(tubelet_size)).mean(-1)
        canon = canon.to(dtype=torch.float32).reshape(canon.shape[0], clips, -1, int(tubelet_size)).mean(-1)
        tubelet_valid = frame_valid.reshape(frame_valid.shape[0], clips, -1, int(tubelet_size)).all(-1)
    else:
        tubelet_valid = frame_valid
    actual = actual * tubelet_valid.to(dtype=actual.dtype)
    canon = canon * tubelet_valid.to(dtype=canon.dtype)
    return {
        "actual": actual,
        "canonical": canon,
        "tubelet_valid_mask": tubelet_valid,
        "irregular_selected_positions": positions,
        "irregular_dense_valid_len": lengths,
        "selected_valid_mask": slot_mask,
    }


def relative_physical_time_residual(actual, canonical, *, dense_valid_len=None, valid_mask=None):
    """Return shared pairwise residual, or None for exact-uniform identity path."""
    if actual.shape != canonical.shape:
        raise ValueError("actual and canonical coordinates must have identical shape")
    delta = actual.to(dtype=torch.float32) - canonical.to(dtype=torch.float32)
    if dense_valid_len is not None:
        lengths = torch.as_tensor(dense_valid_len, device=delta.device, dtype=delta.dtype).flatten()
        if lengths.numel() != delta.shape[0]:
            raise ValueError("dense_valid_len must align with residual batch")
        delta = delta / (lengths - 1.0).clamp_min(1.0)[:, None]
    mask = None
    if valid_mask is not None:
        if not torch.is_tensor(valid_mask) or valid_mask.shape != actual.shape:
            raise ValueError("valid_mask must match actual coordinates")
        mask = valid_mask.to(device=delta.device, dtype=torch.bool)
        delta = delta * mask.to(dtype=delta.dtype)
    if torch.equal(delta, torch.zeros_like(delta)):
        return None
    # caller may flatten clips; preserve batch and token axes
    pair = ((delta.unsqueeze(-1) - delta.unsqueeze(-2)) / 2.0).clamp(-1.0, 1.0)
    if mask is not None:
        pair = pair * (mask.unsqueeze(-1) & mask.unsqueeze(-2)).to(dtype=pair.dtype)
    return pair


def clip_relative_physical_time_mask(
    actual,
    canonical,
    *,
    dense_valid_len,
    tubelet_valid_mask,
    spatial_tokens,
):
    """Expand 8 tubelet coordinates to the temporal-major ViT token layout."""
    if actual.shape != canonical.shape or actual.ndim != 2:
        raise ValueError("actual and canonical must both be [Bclips, tubelets]")
    if int(spatial_tokens) <= 0:
        raise ValueError("spatial_tokens must be positive")
    pair = relative_physical_time_residual(
        actual,
        canonical,
        dense_valid_len=dense_valid_len,
        valid_mask=tubelet_valid_mask,
    )
    if pair is None:
        return None
    # PatchEmbed/VideoMAE flatten order is [tubelet, spatial patch].
    expanded = pair.repeat_interleave(int(spatial_tokens), dim=1).repeat_interleave(int(spatial_tokens), dim=2)
    return expanded.unsqueeze(1)


def single_clock_distortion_summary(actual, canonical, *, dense_valid_len, tubelet_valid_mask):
    """Return detached diagnostics without changing the SingleClock computation."""
    delta = actual.to(dtype=torch.float32) - canonical.to(dtype=torch.float32)
    lengths = torch.as_tensor(dense_valid_len, device=delta.device, dtype=delta.dtype).flatten()
    if lengths.numel() != delta.shape[0]:
        raise ValueError("dense_valid_len must align with distortion batch")
    mask = tubelet_valid_mask.to(device=delta.device, dtype=torch.bool)
    normalized = delta / (lengths - 1.0).clamp_min(1.0)[:, None]
    active = normalized[mask]
    return {
        "valid_tubelets": int(mask.long().sum().detach().cpu().item()),
        "max_abs_normalized_displacement": (
            float(active.abs().max().detach().cpu().item()) if active.numel() else 0.0
        ),
        "mean_abs_normalized_displacement": (
            float(active.abs().mean().detach().cpu().item()) if active.numel() else 0.0
        ),
    }


def _masked_mean(value, mask, dim=-1, keepdim=False, eps=1e-6):
    weight = mask.to(value.dtype)
    numer = (value * weight).sum(dim=dim, keepdim=keepdim)
    denom = weight.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return numer / denom


def _ensure_2d_tensor(tensor, name):
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D [B, T], got shape {tuple(tensor.shape)}.")


def _ensure_prefix_mask(mask, name):
    _ensure_2d_tensor(mask, name)
    mask = mask.bool()
    if mask.shape[1] == 0:
        raise ValueError(f"{name} must have a non-empty temporal dimension.")
    valid_counts = mask.sum(dim=1)
    if (valid_counts <= 0).any().item():
        raise ValueError(f"{name} must contain at least one valid token per sample.")
    for batch_idx in range(mask.shape[0]):
        valid_count = int(valid_counts[batch_idx].item())
        if not mask[batch_idx, :valid_count].all().item() or mask[batch_idx, valid_count:].any().item():
            raise ValueError(f"{name} must be a contiguous valid prefix for sample {batch_idx}.")
    return mask


def _validate_temporal_grid(grid, context="temporal_grid"):
    required = ("center", "cell_left", "cell_right", "valid_mask", "fresh_mask", "level_scale")
    for key in required:
        if key not in grid:
            raise KeyError(f"{context} is missing required key '{key}'.")

    center = grid["center"]
    valid_mask = grid["valid_mask"]
    fresh_mask = grid["fresh_mask"]
    _ensure_2d_tensor(center, f"{context}.center")
    valid_mask = _ensure_prefix_mask(valid_mask, f"{context}.valid_mask")
    _ensure_2d_tensor(fresh_mask, f"{context}.fresh_mask")
    if fresh_mask.shape != valid_mask.shape:
        raise ValueError(f"{context}.fresh_mask shape must match valid_mask shape.")

    for key in ("cell_left", "cell_right"):
        _ensure_2d_tensor(grid[key], f"{context}.{key}")
        if grid[key].shape != center.shape:
            raise ValueError(f"{context}.{key} shape must match center shape.")
    if grid["level_scale"].ndim != 1 or grid["level_scale"].shape[0] != center.shape[0]:
        raise ValueError(f"{context}.level_scale must be [B], got shape {tuple(grid['level_scale'].shape)}.")


def _check_monotonic_center(center, valid_mask, context):
    valid_counts = valid_mask.sum(dim=1)
    for batch_idx in range(center.shape[0]):
        valid_count = int(valid_counts[batch_idx].item())
        valid_center = center[batch_idx, :valid_count]
        if valid_count > 1:
            deltas = valid_center[1:] - valid_center[:-1]
            if (deltas <= 0).any().item():
                raise ValueError(f"{context}.center must be strictly increasing for sample {batch_idx}.")


def build_temporal_grid(
    center,
    valid_mask,
    fresh_mask=None,
    cell_left=None,
    cell_right=None,
    min_scale=1e-4,
    strict=True,
):
    """Build a dense-coordinate temporal grid aligned to a [B, T] prefix mask."""

    _ensure_2d_tensor(center, "center")
    valid_mask = _ensure_prefix_mask(valid_mask.bool(), "valid_mask")
    center = center.to(dtype=torch.float32)
    valid_mask = valid_mask.to(device=center.device)

    if center.shape != valid_mask.shape:
        raise ValueError(f"center shape {tuple(center.shape)} must match valid_mask shape {tuple(valid_mask.shape)}.")

    if fresh_mask is None:
        fresh_mask = valid_mask
    else:
        _ensure_2d_tensor(fresh_mask, "fresh_mask")
        fresh_mask = fresh_mask.to(device=center.device).bool()
        if fresh_mask.shape != valid_mask.shape:
            raise ValueError("fresh_mask shape must match valid_mask shape.")

    if strict:
        _check_monotonic_center(center, valid_mask, "temporal_grid")

    if cell_left is not None or cell_right is not None:
        if cell_left is None or cell_right is None:
            raise ValueError("cell_left and cell_right must be provided together.")
        left = cell_left.to(device=center.device, dtype=center.dtype)
        right = cell_right.to(device=center.device, dtype=center.dtype)
        if left.shape != center.shape or right.shape != center.shape:
            raise ValueError("cell_left and cell_right must match center shape.")
    else:
        left_rows = []
        right_rows = []
        valid_counts = valid_mask.sum(dim=1)
        for batch_idx in range(center.shape[0]):
            valid_count = int(valid_counts[batch_idx].item())
            valid_center = center[batch_idx, :valid_count]
            if valid_count == 1:
                left_valid = valid_center.new_ones((1,))
                right_valid = valid_center.new_ones((1,))
            else:
                delta = (valid_center[1:] - valid_center[:-1]).clamp_min(min_scale)
                left_valid = torch.cat((delta[:1], delta), dim=0)
                right_valid = torch.cat((delta, delta[-1:]), dim=0)
            if valid_count < center.shape[1]:
                pad_len = center.shape[1] - valid_count
                left_valid = torch.cat((left_valid, left_valid[-1:].expand(pad_len)), dim=0)
                right_valid = torch.cat((right_valid, right_valid[-1:].expand(pad_len)), dim=0)
            left_rows.append(left_valid)
            right_rows.append(right_valid)
        left = torch.stack(left_rows, dim=0)
        right = torch.stack(right_rows, dim=0)

    left = left.clamp_min(min_scale)
    right = right.clamp_min(min_scale)
    level_scale = _masked_mean(0.5 * (left + right), valid_mask, dim=1)
    grid = {
        "center": center,
        "cell_left": left,
        "cell_right": right,
        "valid_mask": valid_mask,
        "fresh_mask": fresh_mask & valid_mask,
        "level_scale": level_scale,
    }
    _validate_temporal_grid(grid)
    return grid


def normalize_temporal_grid_input(temporal_grid, mask, device=None, dtype=torch.float32, required=False, strict=True):
    _ensure_2d_tensor(mask, "mask")
    mask = _ensure_prefix_mask(mask.bool(), "mask")
    out_device = mask.device if device is None else device

    if temporal_grid is None:
        if required:
            raise ValueError("temporal_grid is required but was not provided.")
        batch, length = mask.shape
        center = torch.arange(length, device=out_device, dtype=dtype)[None].repeat(batch, 1)
        return build_temporal_grid(center, valid_mask=mask.to(out_device), strict=strict)

    if torch.is_tensor(temporal_grid):
        center = temporal_grid.to(device=out_device, dtype=dtype)
        return build_temporal_grid(center, valid_mask=mask.to(out_device), strict=strict)

    if not isinstance(temporal_grid, dict):
        raise TypeError("temporal_grid must be None, a tensor, or a dict.")
    if "center" not in temporal_grid:
        raise KeyError("temporal_grid dict must contain 'center'.")

    center = temporal_grid["center"].to(device=out_device, dtype=dtype)
    fresh_mask = temporal_grid.get("fresh_mask", mask).to(device=out_device).bool()
    cell_left = temporal_grid.get("cell_left", None)
    cell_right = temporal_grid.get("cell_right", None)
    if cell_left is not None:
        cell_left = cell_left.to(device=out_device, dtype=dtype)
    if cell_right is not None:
        cell_right = cell_right.to(device=out_device, dtype=dtype)

    grid = build_temporal_grid(
        center,
        valid_mask=mask.to(out_device),
        fresh_mask=fresh_mask,
        cell_left=cell_left,
        cell_right=cell_right,
        strict=strict,
    )
    dense_valid_len = temporal_grid.get("dense_valid_len", None)
    if dense_valid_len is not None:
        dense_valid_len = dense_valid_len.to(device=out_device, dtype=dtype)
        if dense_valid_len.ndim != 1 or dense_valid_len.shape[0] != mask.shape[0]:
            raise ValueError("temporal_grid.dense_valid_len must be [B].")
        grid["dense_valid_len"] = dense_valid_len
    return grid


def _bridge_tensor_temporal_payload(
    meta,
    *,
    mask_row,
    true_valid,
    positions_key,
    valid_len_key,
    batch_idx,
):
    if not isinstance(meta, Mapping):
        return None
    bridge = meta.get("pc_ot_mras_bridge")
    if not isinstance(bridge, Mapping):
        return None

    has_positions = "selected_dense_positions" in bridge
    has_dense_len = "dense_valid_len_tensor" in bridge
    if not has_positions and not has_dense_len:
        return None
    if not has_positions or not has_dense_len:
        raise ValueError(
            f"meta[{batch_idx}].pc_ot_mras_bridge tensor temporal metadata must contain "
            "'selected_dense_positions' and 'dense_valid_len_tensor' together."
        )

    raw_positions = bridge["selected_dense_positions"]
    raw_dense_len = bridge["dense_valid_len_tensor"]
    if not torch.is_tensor(raw_positions):
        raise ValueError(f"meta[{batch_idx}].pc_ot_mras_bridge.selected_dense_positions must be a tensor.")
    if not torch.is_tensor(raw_dense_len):
        raise ValueError(f"meta[{batch_idx}].pc_ot_mras_bridge.dense_valid_len_tensor must be a tensor.")

    positions = raw_positions.to(device=mask_row.device, dtype=torch.float32).flatten()
    dense_len_tensor = raw_dense_len.to(device=mask_row.device, dtype=torch.float32).flatten()
    if dense_len_tensor.numel() != 1:
        raise ValueError(f"meta[{batch_idx}].pc_ot_mras_bridge.dense_valid_len_tensor must be scalar.")
    if positions.numel() < true_valid:
        raise ValueError(
            f"meta[{batch_idx}].pc_ot_mras_bridge.selected_dense_positions length must cover "
            f"valid token count {true_valid}, got {positions.numel()}."
        )

    raw_selected_mask = bridge.get("selected_mask")
    if raw_selected_mask is not None:
        if not torch.is_tensor(raw_selected_mask):
            raise ValueError(f"meta[{batch_idx}].pc_ot_mras_bridge.selected_mask must be a tensor.")
        selected_mask = raw_selected_mask.to(device=mask_row.device).bool().flatten()
        if selected_mask.numel() != positions.numel():
            raise ValueError(
                f"meta[{batch_idx}].pc_ot_mras_bridge.selected_mask shape must match selected_dense_positions."
            )
        expected = torch.arange(selected_mask.numel(), device=mask_row.device) < int(true_valid)
        if not torch.equal(selected_mask, expected):
            raise ValueError(f"meta[{batch_idx}].pc_ot_mras_bridge.selected_mask must match the feature mask prefix.")
        if positions.numel() == mask_row.numel() and not torch.equal(selected_mask, mask_row.bool()):
            raise ValueError(f"meta[{batch_idx}].pc_ot_mras_bridge.selected_mask must match the feature mask.")

    valid_positions = positions[:true_valid]
    dense_len = float(dense_len_tensor[0].item())
    if dense_len <= 0:
        raise ValueError(f"meta[{batch_idx}].pc_ot_mras_bridge.dense_valid_len_tensor must be positive.")
    if true_valid > 0 and (
        valid_positions.detach().min().item() < 0 or valid_positions.detach().max().item() >= dense_len
    ):
        raise ValueError(
            f"meta[{batch_idx}].pc_ot_mras_bridge.selected_dense_positions must be in [0, {dense_len})."
        )

    dense_valid_len_key = "irregular_dense_valid_len" if "irregular_dense_valid_len" in meta else valid_len_key
    if positions_key in meta:
        legacy_positions = torch.as_tensor(meta[positions_key], device=mask_row.device, dtype=torch.float32).flatten()
        if legacy_positions.numel() != true_valid:
            raise ValueError(
                f"meta[{batch_idx}]['{positions_key}'] length must equal valid token count {true_valid}."
            )
        if not torch.allclose(legacy_positions, valid_positions.detach(), atol=1e-4, rtol=1e-4):
            raise ValueError(
                f"meta[{batch_idx}] bridge tensor temporal positions must match '{positions_key}' legacy alias."
            )
    if dense_valid_len_key in meta:
        if abs(float(meta[dense_valid_len_key]) - dense_len) > 1e-4:
            raise ValueError(
                f"meta[{batch_idx}] bridge tensor dense_valid_len must match '{dense_valid_len_key}' legacy alias."
            )
    if "irregular_dense_valid_len" in meta and valid_len_key in meta:
        if float(meta["irregular_dense_valid_len"]) != float(meta[valid_len_key]):
            raise ValueError(f"meta[{batch_idx}] irregular_dense_valid_len must match '{valid_len_key}' alias.")

    return valid_positions, dense_len_tensor[0]


def temporal_grid_from_metas(
    metas,
    mask,
    positions_key="irregular_selected_positions",
    valid_len_key="irregular_selected_valid_len",
    required=True,
    strict=True,
):
    """Build a dense-coordinate temporal grid from dataloader metadata."""

    _ensure_2d_tensor(mask, "mask")
    mask = _ensure_prefix_mask(mask.bool(), "mask")
    if metas is None:
        if required:
            raise ValueError("metas are required to build temporal_grid.")
        return normalize_temporal_grid_input(None, mask, required=False, strict=strict)
    if len(metas) != mask.shape[0]:
        raise ValueError(f"metas length {len(metas)} must match batch size {mask.shape[0]}.")

    center_rows = []
    dense_valid_len_rows = []
    valid_counts = mask.sum(dim=1)

    for batch_idx, meta in enumerate(metas):
        if not isinstance(meta, Mapping):
            raise ValueError(f"meta[{batch_idx}] must be a mapping when temporal_grid is enabled.")
        true_valid = int(valid_counts[batch_idx].item())
        if "irregular_selected_count" in meta and int(meta["irregular_selected_count"]) != true_valid:
            raise ValueError(
                f"meta[{batch_idx}] irregular_selected_count must equal valid token count {true_valid}."
            )
        tensor_payload = _bridge_tensor_temporal_payload(
            meta,
            mask_row=mask[batch_idx],
            true_valid=true_valid,
            positions_key=positions_key,
            valid_len_key=valid_len_key,
            batch_idx=batch_idx,
        )
        dense_valid_len_key = "irregular_dense_valid_len" if "irregular_dense_valid_len" in meta else valid_len_key
        if tensor_payload is not None:
            valid_positions, dense_len_tensor = tensor_payload
            dense_len = float(dense_len_tensor.item())
        elif positions_key not in meta or dense_valid_len_key not in meta:
            if required:
                raise ValueError(
                    f"meta[{batch_idx}] must contain '{positions_key}' and '{dense_valid_len_key}' "
                    "when temporal_grid is enabled."
                )
            return normalize_temporal_grid_input(None, mask, required=False, strict=strict)
        else:
            if "irregular_dense_valid_len" in meta and valid_len_key in meta:
                if float(meta["irregular_dense_valid_len"]) != float(meta[valid_len_key]):
                    raise ValueError(
                        f"meta[{batch_idx}] irregular_dense_valid_len must match '{valid_len_key}' alias."
                    )
            positions = torch.as_tensor(meta[positions_key], device=mask.device, dtype=torch.float32).flatten()
            if positions.numel() != true_valid:
                raise ValueError(
                    f"meta[{batch_idx}]['{positions_key}'] length must equal valid token count "
                    f"{true_valid}; padded tail positions are not allowed, got {positions.numel()}."
                )
            valid_positions = positions[:true_valid]
            dense_len = float(meta[dense_valid_len_key])
            dense_len_tensor = mask.new_tensor(dense_len, dtype=torch.float32)

        if true_valid > 1 and ((valid_positions[1:] - valid_positions[:-1]) <= 0).any().item():
            raise ValueError(f"meta[{batch_idx}]['{positions_key}'] must be strictly increasing.")

        if dense_len <= 0:
            raise ValueError(f"meta[{batch_idx}]['{dense_valid_len_key}'] must be positive.")
        if true_valid > 0 and (valid_positions.min().item() < 0 or valid_positions.max().item() >= dense_len):
            raise ValueError(
                f"meta[{batch_idx}]['{positions_key}'] must be in [0, {dense_len}), "
                f"got min={valid_positions.min().item()}, max={valid_positions.max().item()}."
            )

        row_center = valid_positions
        if true_valid < mask.shape[1]:
            row_center = torch.cat((row_center, valid_positions[-1:].expand(mask.shape[1] - true_valid)), dim=0)
        center_rows.append(row_center)
        dense_valid_len_rows.append(dense_len_tensor)

    center = torch.stack(center_rows, dim=0)
    dense_valid_len = torch.stack(dense_valid_len_rows, dim=0).to(device=mask.device, dtype=torch.float32)
    grid = build_temporal_grid(center, valid_mask=mask, fresh_mask=mask, strict=strict)
    grid["dense_valid_len"] = dense_valid_len
    return grid


def validate_temporal_grid_alignment(grid, mask, context="temporal_grid"):
    _ensure_2d_tensor(mask, "mask")
    _validate_temporal_grid(grid, context=context)
    if grid["center"].shape != mask.shape:
        raise ValueError(f"{context}.center shape must match mask shape.")
    if not torch.equal(grid["valid_mask"].to(mask.device), mask.bool()):
        raise ValueError(f"{context}.valid_mask must match the feature mask.")


def downsample_temporal_grid(grid, min_scale=1e-4):
    _validate_temporal_grid(grid)
    center = grid["center"].to(dtype=torch.float32)
    valid_mask = grid["valid_mask"].bool()
    fresh_mask = grid["fresh_mask"].bool()
    cell_left = grid["cell_left"].to(dtype=center.dtype).clamp_min(min_scale)
    cell_right = grid["cell_right"].to(dtype=center.dtype).clamp_min(min_scale)

    def split_even_odd(tensor, pad_value):
        even = tensor[:, 0::2]
        odd = tensor[:, 1::2]
        if odd.shape[1] < even.shape[1]:
            pad_len = even.shape[1] - odd.shape[1]
            pad_tensor = tensor.new_full((tensor.shape[0], pad_len), pad_value)
            odd = torch.cat([odd, pad_tensor], dim=1)
        return even, odd

    even_center, odd_center = split_even_odd(center, 0.0)
    even_valid, odd_valid = split_even_odd(valid_mask, False)
    even_fresh, odd_fresh = split_even_odd(fresh_mask, False)
    even_left, odd_left = split_even_odd(cell_left, 0.0)
    even_right, odd_right = split_even_odd(cell_right, 0.0)

    pair_valid = even_valid | odd_valid
    even_width = 0.5 * (even_left + even_right) * even_valid.to(center.dtype)
    odd_width = 0.5 * (odd_left + odd_right) * odd_valid.to(center.dtype)
    pair_weight = even_width + odd_width
    fallback_center = torch.where(
        even_valid,
        even_center,
        torch.where(odd_valid, odd_center, even_center.new_zeros(even_center.shape)),
    )
    merged_center = torch.where(
        pair_valid,
        (even_center * even_width + odd_center * odd_width) / pair_weight.clamp_min(min_scale),
        fallback_center,
    )

    even_start = even_center - 0.5 * even_left
    even_end = even_center + 0.5 * even_right
    odd_start = odd_center - 0.5 * odd_left
    odd_end = odd_center + 0.5 * odd_right
    inf = center.new_full(even_center.shape, float("inf"))
    neg_inf = center.new_full(even_center.shape, float("-inf"))
    merged_start = torch.minimum(
        torch.where(even_valid, even_start, inf),
        torch.where(odd_valid, odd_start, inf),
    )
    merged_end = torch.maximum(
        torch.where(even_valid, even_end, neg_inf),
        torch.where(odd_valid, odd_end, neg_inf),
    )
    merged_start = torch.where(pair_valid, merged_start, merged_center)
    merged_end = torch.where(pair_valid, merged_end, merged_center)
    merged_left = (2.0 * (merged_center - merged_start)).clamp_min(min_scale)
    merged_right = (2.0 * (merged_end - merged_center)).clamp_min(min_scale)
    merged_fresh = (even_fresh & even_valid) | (odd_fresh & odd_valid)

    downsampled = build_temporal_grid(
        merged_center,
        valid_mask=pair_valid,
        fresh_mask=merged_fresh,
        cell_left=merged_left,
        cell_right=merged_right,
        min_scale=min_scale,
        strict=True,
    )
    if "dense_valid_len" in grid:
        downsampled["dense_valid_len"] = grid["dense_valid_len"].to(
            device=downsampled["center"].device,
            dtype=downsampled["center"].dtype,
        )
    return downsampled


def _dense_valid_len_for_area_grid(grid):
    center = grid["center"]
    if "dense_valid_len" in grid:
        dense_valid_len = grid["dense_valid_len"].to(device=center.device, dtype=center.dtype)
    else:
        right = grid["cell_right"].to(device=center.device, dtype=center.dtype).clamp_min(1e-4)
        dense_valid_len = center.max(dim=1).values + 0.5 * right.max(dim=1).values
    if dense_valid_len.ndim != 1 or dense_valid_len.shape[0] != center.shape[0]:
        raise ValueError("dense_valid_len must be [B] for area time grids.")
    return dense_valid_len.clamp_min(1.0)


def build_area_time_grid(
    temporal_grid,
    observation_half_width="cell_support",
    observation_support_scale=0.25,
    min_width=1e-4,
):
    """Build observation cells and inter-observation gap cells for P2 heads.

    Observation cells are narrow deploy-visible footprints around selected dense
    positions. Gap cells are the unobserved real-time intervals between those
    footprints, including leading and trailing gaps. Detection can therefore
    score observed evidence separately from uncertain boundaries inside gaps.
    """

    _validate_temporal_grid(temporal_grid, context="area_time_grid.temporal_grid")
    use_cell_support = observation_half_width is None or str(observation_half_width) == "cell_support"
    if not use_cell_support and float(observation_half_width) <= 0:
        raise ValueError("observation_half_width must be positive or 'cell_support'.")
    if observation_support_scale <= 0:
        raise ValueError("observation_support_scale must be positive.")
    if min_width <= 0:
        raise ValueError("min_width must be positive.")

    center = temporal_grid["center"].to(dtype=torch.float32)
    valid_mask = temporal_grid["valid_mask"].to(device=center.device).bool()
    valid_mask = _ensure_prefix_mask(valid_mask, "area_time_grid.valid_mask")
    dense_valid_len = _dense_valid_len_for_area_grid(temporal_grid)
    batch, length = center.shape

    obs_start = torch.zeros_like(center)
    obs_end = torch.zeros_like(center)
    obs_half = torch.zeros_like(center)
    gap_start = center.new_zeros((batch, length + 1))
    gap_end = center.new_zeros((batch, length + 1))
    gap_valid_mask = torch.zeros((batch, length + 1), device=center.device, dtype=torch.bool)
    valid_counts = valid_mask.sum(dim=1)

    for batch_idx in range(batch):
        valid_count = int(valid_counts[batch_idx].item())
        if valid_count <= 0:
            raise ValueError("area_time_grid requires at least one valid observation.")

        positions = center[batch_idx, :valid_count]
        dense_len = dense_valid_len[batch_idx]
        if valid_count > 1 and ((positions[1:] - positions[:-1]) <= 0).any().item():
            raise ValueError(f"area_time_grid positions must be strictly increasing for sample {batch_idx}.")

        if use_cell_support:
            left_half = (
                0.5
                * temporal_grid["cell_left"][batch_idx, :valid_count].to(device=center.device, dtype=center.dtype)
                * float(observation_support_scale)
            ).clamp_min(min_width)
            right_half = (
                0.5
                * temporal_grid["cell_right"][batch_idx, :valid_count].to(device=center.device, dtype=center.dtype)
                * float(observation_support_scale)
            ).clamp_min(min_width)
        else:
            base_half = positions.new_full((valid_count,), float(observation_half_width))
            if valid_count > 1:
                prev_dist = positions[1:] - positions[:-1]
                limited_half = 0.5 * prev_dist.clamp_min(min_width)
                left_half = torch.cat(
                    (base_half[:1], torch.minimum(base_half[1:], limited_half)),
                    dim=0,
                )
                right_half = torch.cat(
                    (torch.minimum(base_half[:-1], limited_half), base_half[-1:]),
                    dim=0,
                )
            else:
                left_half = base_half
                right_half = base_half
        left_half = left_half.clamp_min(min_width)
        right_half = right_half.clamp_min(min_width)

        start = torch.maximum(positions - left_half, positions.new_zeros(positions.shape))
        end = torch.minimum(positions + right_half, dense_len.expand_as(positions))
        end = torch.maximum(end, start + min_width)
        end = torch.minimum(end, dense_len.expand_as(end))
        start = torch.minimum(start, (end - min_width).clamp_min(0.0))

        obs_start[batch_idx, :valid_count] = start
        obs_end[batch_idx, :valid_count] = end
        obs_half[batch_idx, :valid_count] = 0.5 * (end - start)
        if valid_count < length:
            obs_start[batch_idx, valid_count:] = start[-1]
            obs_end[batch_idx, valid_count:] = end[-1]
            obs_half[batch_idx, valid_count:] = obs_half[batch_idx, valid_count - 1]

        starts = []
        ends = []
        starts.append(positions.new_tensor(0.0))
        ends.append(start[0])
        for idx in range(1, valid_count):
            starts.append(end[idx - 1])
            ends.append(start[idx])
        starts.append(end[valid_count - 1])
        ends.append(dense_len)

        sample_gap_start = torch.stack(starts)
        sample_gap_end = torch.stack(ends)
        sample_gap_valid = sample_gap_end > sample_gap_start + min_width
        gap_start[batch_idx, : valid_count + 1] = sample_gap_start
        gap_end[batch_idx, : valid_count + 1] = sample_gap_end
        gap_valid_mask[batch_idx, : valid_count + 1] = sample_gap_valid
        if valid_count + 1 < length + 1:
            gap_start[batch_idx, valid_count + 1 :] = sample_gap_start[-1]
            gap_end[batch_idx, valid_count + 1 :] = sample_gap_end[-1]

    obs_width = (obs_end - obs_start).clamp_min(min_width)
    gap_width = (gap_end - gap_start).clamp_min(min_width)
    area_grid = {
        "obs_center": center,
        "obs_start": obs_start,
        "obs_end": obs_end,
        "obs_width": obs_width,
        "obs_half_width": obs_half.clamp_min(min_width),
        "obs_valid_mask": valid_mask,
        "gap_start": gap_start,
        "gap_end": gap_end,
        "gap_center": 0.5 * (gap_start + gap_end),
        "gap_width": gap_width,
        "gap_valid_mask": gap_valid_mask,
        "dense_valid_len": dense_valid_len,
    }
    validate_area_time_grid(area_grid)
    return area_grid


def validate_area_time_grid(area_grid, context="area_time_grid"):
    required = (
        "obs_center",
        "obs_start",
        "obs_end",
        "obs_width",
        "obs_valid_mask",
        "gap_start",
        "gap_end",
        "gap_center",
        "gap_width",
        "gap_valid_mask",
        "dense_valid_len",
    )
    for key in required:
        if key not in area_grid:
            raise KeyError(f"{context} is missing required key '{key}'.")
    _ensure_2d_tensor(area_grid["obs_center"], f"{context}.obs_center")
    obs_valid = _ensure_prefix_mask(area_grid["obs_valid_mask"], f"{context}.obs_valid_mask")
    for key in ("obs_start", "obs_end", "obs_width"):
        _ensure_2d_tensor(area_grid[key], f"{context}.{key}")
        if area_grid[key].shape != obs_valid.shape:
            raise ValueError(f"{context}.{key} shape must match obs_valid_mask.")
    gap_valid = area_grid["gap_valid_mask"]
    _ensure_2d_tensor(gap_valid, f"{context}.gap_valid_mask")
    gap_valid = gap_valid.bool()
    for key in ("gap_start", "gap_end", "gap_center", "gap_width"):
        _ensure_2d_tensor(area_grid[key], f"{context}.{key}")
        if area_grid[key].shape != gap_valid.shape:
            raise ValueError(f"{context}.{key} shape must match gap_valid_mask.")
    if area_grid["dense_valid_len"].ndim != 1 or area_grid["dense_valid_len"].shape[0] != obs_valid.shape[0]:
        raise ValueError(f"{context}.dense_valid_len must be [B].")
    if ((area_grid["obs_end"] - area_grid["obs_start"])[obs_valid] <= 0).any().item():
        raise ValueError(f"{context} contains non-positive observation cells.")
    if gap_valid.any().item() and ((area_grid["gap_end"] - area_grid["gap_start"])[gap_valid] <= 0).any().item():
        raise ValueError(f"{context} contains non-positive gap cells.")


def segment_area_integral(starts, ends, cell_start, cell_end, values, eps=1e-4):
    """Integrate per-cell values over real-time interval overlap.

    Args:
        starts, ends: proposal boundaries shaped [P].
        cell_start, cell_end: observation cell boundaries shaped [K].
        values: per-cell values shaped [K] or [K, C].

    Returns:
        A dict with `score` shaped [P] or [P, C] and `observed_fraction` shaped
        [P], measuring how much of each proposal is covered by observation
        cells.
    """

    if starts.ndim != 1 or ends.ndim != 1:
        raise ValueError("starts and ends must be 1D tensors.")
    if cell_start.ndim != 1 or cell_end.ndim != 1:
        raise ValueError("cell_start and cell_end must be 1D tensors.")
    if starts.shape != ends.shape:
        raise ValueError("starts and ends must have the same shape.")
    if cell_start.shape != cell_end.shape:
        raise ValueError("cell_start and cell_end must have the same shape.")
    if values.shape[0] != cell_start.shape[0]:
        raise ValueError("values first dimension must match cell count.")

    if starts.numel() == 0:
        empty_score_shape = starts.shape if values.ndim == 1 else (starts.numel(), values.shape[1])
        return {
            "score": values.new_zeros(empty_score_shape),
            "observed_fraction": starts.new_zeros(starts.shape),
        }

    starts = starts.to(device=cell_start.device, dtype=cell_start.dtype)
    ends = ends.to(device=cell_start.device, dtype=cell_start.dtype)
    values = values.to(device=cell_start.device, dtype=cell_start.dtype)
    duration = (ends - starts).clamp_min(eps)
    overlap_start = torch.maximum(cell_start[None, :], starts[:, None])
    overlap_end = torch.minimum(cell_end[None, :], ends[:, None])
    overlap = (overlap_end - overlap_start).clamp_min(0.0)
    observed_mass = overlap.sum(dim=1)
    if values.ndim == 1:
        weighted = (overlap * values[None, :]).sum(dim=1)
        score = weighted / observed_mass.clamp_min(eps)
    elif values.ndim == 2:
        weighted = overlap @ values
        score = weighted / observed_mass[:, None].clamp_min(eps)
    else:
        raise ValueError("values must be [K] or [K, C].")
    observed_fraction = (observed_mass / duration).clamp(0.0, 1.0)
    score = torch.where(
        observed_mass.reshape((-1,) + (1,) * (score.ndim - 1)) > 0,
        score,
        torch.zeros_like(score),
    )
    return {"score": score.clamp(0.0, 1.0), "observed_fraction": observed_fraction}


@torch.no_grad()
def prepare_area_targets(area_grid, gt_segments, gt_labels, num_classes, boundary_tau=1.0, duration_range=None):
    """Prepare P2 area and gap-boundary targets in dense physical time."""

    validate_area_time_grid(area_grid)
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if boundary_tau <= 0:
        raise ValueError("boundary_tau must be positive.")
    if duration_range is not None:
        if len(duration_range) != 2:
            raise ValueError("duration_range must be a pair [min_duration, max_duration].")
        min_duration = float(duration_range[0])
        max_duration = float(duration_range[1])
        if min_duration < 0 or max_duration < min_duration:
            raise ValueError("duration_range must satisfy 0 <= min_duration <= max_duration.")
    else:
        min_duration = None
        max_duration = None

    obs_start = area_grid["obs_start"]
    obs_end = area_grid["obs_end"]
    obs_width = area_grid["obs_width"].clamp_min(1e-4)
    obs_valid = area_grid["obs_valid_mask"]
    gap_start = area_grid["gap_start"]
    gap_end = area_grid["gap_end"]
    gap_center = area_grid["gap_center"]
    gap_width = area_grid["gap_width"].clamp_min(1e-4)
    gap_valid = area_grid["gap_valid_mask"]
    batch_size, obs_count = obs_start.shape
    gap_count = gap_center.shape[1]

    area = obs_start.new_zeros((batch_size, obs_count, num_classes))
    start_gap = obs_start.new_zeros((batch_size, gap_count, num_classes))
    end_gap = obs_start.new_zeros((batch_size, gap_count, num_classes))
    start_offset = obs_start.new_zeros((batch_size, gap_count, num_classes))
    end_offset = obs_start.new_zeros((batch_size, gap_count, num_classes))
    start_offset_weight = obs_start.new_zeros((batch_size, gap_count, num_classes))
    end_offset_weight = obs_start.new_zeros((batch_size, gap_count, num_classes))

    for batch_idx, (gt_segment, gt_label) in enumerate(zip(gt_segments, gt_labels)):
        if gt_segment.shape[0] == 0:
            continue
        gt_segment = gt_segment.to(device=obs_start.device, dtype=obs_start.dtype)
        gt_label = gt_label.to(device=obs_start.device).long()
        gt_start = gt_segment[:, 0]
        gt_end = gt_segment[:, 1]
        if duration_range is not None:
            gt_duration = gt_end - gt_start
            keep = (gt_duration >= min_duration) & (gt_duration <= max_duration)
            if not keep.any().item():
                continue
            gt_start = gt_start[keep]
            gt_end = gt_end[keep]
            gt_label = gt_label[keep]

        overlap_start = torch.maximum(obs_start[batch_idx, :, None], gt_start[None, :])
        overlap_end = torch.minimum(obs_end[batch_idx, :, None], gt_end[None, :])
        coverage = (overlap_end - overlap_start).clamp_min(0.0) / obs_width[batch_idx, :, None]
        coverage = coverage * obs_valid[batch_idx, :, None].to(coverage.dtype)

        gap_scale = (boundary_tau * gap_width[batch_idx, :, None]).clamp_min(1e-4)
        start_score = torch.exp(-torch.abs(gap_center[batch_idx, :, None] - gt_start[None, :]) / gap_scale)
        end_score = torch.exp(-torch.abs(gap_center[batch_idx, :, None] - gt_end[None, :]) / gap_scale)
        start_score = start_score * gap_valid[batch_idx, :, None].to(start_score.dtype)
        end_score = end_score * gap_valid[batch_idx, :, None].to(end_score.dtype)
        start_inside = (
            (gt_start[None, :] >= gap_start[batch_idx, :, None])
            & (gt_start[None, :] <= gap_end[batch_idx, :, None])
            & gap_valid[batch_idx, :, None]
        )
        end_inside = (
            (gt_end[None, :] >= gap_start[batch_idx, :, None])
            & (gt_end[None, :] <= gap_end[batch_idx, :, None])
            & gap_valid[batch_idx, :, None]
        )
        start_offset_value = (
            (gt_start[None, :] - gap_center[batch_idx, :, None]) / (0.5 * gap_width[batch_idx, :, None]).clamp_min(1e-4)
        ).clamp(-1.0, 1.0)
        end_offset_value = (
            (gt_end[None, :] - gap_center[batch_idx, :, None]) / (0.5 * gap_width[batch_idx, :, None]).clamp_min(1e-4)
        ).clamp(-1.0, 1.0)
        start_weight = start_score * start_inside.to(start_score.dtype)
        end_weight = end_score * end_inside.to(end_score.dtype)

        for gt_idx, label in enumerate(gt_label):
            cls_idx = int(label.item())
            if cls_idx < 0 or cls_idx >= num_classes:
                raise ValueError(f"gt label {cls_idx} is outside [0, {num_classes}).")
            area[batch_idx, :, cls_idx] = torch.maximum(area[batch_idx, :, cls_idx], coverage[:, gt_idx])
            start_gap[batch_idx, :, cls_idx] = torch.maximum(start_gap[batch_idx, :, cls_idx], start_score[:, gt_idx])
            end_gap[batch_idx, :, cls_idx] = torch.maximum(end_gap[batch_idx, :, cls_idx], end_score[:, gt_idx])
            better_start = start_weight[:, gt_idx] > start_offset_weight[batch_idx, :, cls_idx]
            better_end = end_weight[:, gt_idx] > end_offset_weight[batch_idx, :, cls_idx]
            start_offset[batch_idx, better_start, cls_idx] = start_offset_value[better_start, gt_idx]
            end_offset[batch_idx, better_end, cls_idx] = end_offset_value[better_end, gt_idx]
            start_offset_weight[batch_idx, :, cls_idx] = torch.maximum(
                start_offset_weight[batch_idx, :, cls_idx],
                start_weight[:, gt_idx],
            )
            end_offset_weight[batch_idx, :, cls_idx] = torch.maximum(
                end_offset_weight[batch_idx, :, cls_idx],
                end_weight[:, gt_idx],
            )

    return {
        "area": area.clamp(0.0, 1.0),
        "start_gap": start_gap.clamp(0.0, 1.0),
        "end_gap": end_gap.clamp(0.0, 1.0),
        "start_offset": start_offset.clamp(-1.0, 1.0),
        "end_offset": end_offset.clamp(-1.0, 1.0),
        "start_offset_weight": start_offset_weight.clamp(0.0, 1.0),
        "end_offset_weight": end_offset_weight.clamp(0.0, 1.0),
    }
