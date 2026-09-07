"""Geometry primitives for native pairs and disjoint observation support."""
import torch
from torch import Tensor
from .contracts import NativeLayout, VideoBatch


def detector_validity(valid_frames: Tensor, query_length: int):
    """Frame-grid queries keep frame validity; half-grid queries own native pairs.

    A half-grid cell is valid when either of its two source frames is valid.
    This reduction is never expanded back to the frame detection grid.
    """
    if query_length == valid_frames.shape[-1]:
        return valid_frames.bool()
    if 2 * query_length == valid_frames.shape[-1]:
        return valid_frames.reshape(*valid_frames.shape[:-1], query_length, 2).any(-1)
    raise ValueError("detector queries must use the original frame grid or native pair grid")


def native_layout(batch: VideoBatch, patch_size=16, parent_frames=16):
    b, _, t, h, w = batch.frames_hi.shape
    if t % parent_frames or parent_frames % 2 or h % patch_size or w % patch_size:
        raise ValueError("input must use complete native parents and spatial patches")
    if batch.pts_s.shape != (b, t) or batch.valid_frames.shape != (b, t):
        raise ValueError("frame timestamps/validity must match the dense video")
    if batch.source_intervals_s.shape != (b, t, 2):
        raise ValueError("actual per-frame display intervals are required")
    h, w, tn = h // patch_size, w // patch_size, t // 2
    device = batch.frames_hi.device
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    pair_valid = batch.valid_frames.reshape(b, tn, 2).bool()
    return NativeLayout(
        (torch.arange(tn, device=device) // (parent_frames // 2))[:, None, None].expand(tn, h, w),
        torch.arange(tn * h * w, device=device).reshape(tn, h, w),
        torch.stack((yy, xx), -1)[None].expand(tn, h, w, 2),
        batch.source_intervals_s.reshape(b, tn, 2, 2), pair_valid,
        batch.target_time_s.reshape(b, tn, 2).mean(-1),
        pair_valid.any(-1)[:, :, None, None].expand(b, tn, h, w), parent_frames // 2,
    )


def canonical_atoms(tubelets, height, width, parent_tubelets=8, temporal_atom=1, spatial_group=2, axis="ST", device=None):
    """No atom crosses a parent; row members preserve native patch indices."""
    if axis not in {"T", "ST"} or temporal_atom not in {1, 2, 4, 8}:
        raise ValueError("unsupported native axis or temporal atom")
    if tubelets % parent_tubelets or parent_tubelets % temporal_atom:
        raise ValueError("temporal atoms must partition each parent")
    if axis == "ST" and (height % spatial_group or width % spatial_group):
        raise ValueError("spatial groups must partition the native patch grid")
    ids = torch.arange(tubelets * height * width, device=device).reshape(tubelets, height, width)
    rows = []
    sy, sx = (height, width) if axis == "T" else (spatial_group, spatial_group)
    for t in range(0, tubelets, temporal_atom):
        for y in range(0, height, sy):
            for x in range(0, width, sx):
                rows.append(ids[t:t + temporal_atom, y:y + sy, x:x + sx].reshape(-1))
    return torch.stack(rows)


def support_distance(query_s: Tensor, intervals: Tensor, valid: Tensor):
    """[B,Q] to [B,E,U,2] union distance; a gap remains unobserved."""
    q = query_s[:, :, None, None]
    left, right = intervals[..., 0][:, None], intervals[..., 1][:, None]
    d = torch.maximum(left - q, q - right).clamp_min(0)
    return d.masked_fill(~valid[:, None].bool(), float("inf")).amin(-1)


def union_duration(intervals: Tensor, valid: Tensor):
    """Exact length of a union; repeated source frames do not double count."""
    starts, order = intervals[..., 0].masked_fill(~valid, float("inf")).sort(-1)
    ends = intervals[..., 1].masked_fill(~valid, -float("inf")).gather(-1, order)
    covered = ends.cummax(-1).values
    previous = torch.cat((torch.full_like(covered[..., :1], -float("inf")), covered[..., :-1]), -1)
    return (ends - torch.maximum(starts, previous)).clamp_min(0).sum(-1)


def clipped_roi(box: Tensor):
    box = box.clamp(0, 1)
    if bool((box[..., 2:] <= box[..., :2]).any()):
        raise ValueError("ROI has no source area after clipping")
    return box
