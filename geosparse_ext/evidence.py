"""Build B evidence slots from only the native tokens actually encoded."""
import torch
from .contracts import EvidenceBatch


def spatial_slots(features, selected, layout, slots=4, spatial_transform=None):
    """features [B,D,T,H,W], selection [B,T,H,W]. Empty slots stay invalid."""
    if slots not in {1, 4, 8}:
        raise ValueError("registered evidence slots are 1,4,8")
    b, d, t, h, w = features.shape
    rows, cols = {1: (1, 1), 4: (2, 2), 8: (2, 4)}[slots]
    sy = torch.arange(slots, device=features.device) // cols
    sx = torch.arange(slots, device=features.device) % cols
    y0, y1 = sy * h // rows, (sy + 1) * h // rows
    x0, x1 = sx * w // cols, (sx + 1) * w // cols
    yy, xx = torch.meshgrid(torch.arange(h, device=features.device), torch.arange(w, device=features.device), indexing="ij")
    members = ((yy[None] >= y0[:, None, None]) & (yy[None] < y1[:, None, None])
               & (xx[None] >= x0[:, None, None]) & (xx[None] < x1[:, None, None])).flatten(1).to(features.dtype)
    dtype = features.dtype
    with torch.autocast(device_type=features.device.type, enabled=False):
        mask = (selected & layout.valid).reshape(b * t, h * w).float()
        counts = mask @ members.float().T
        encoded = features.permute(0, 2, 1, 3, 4).reshape(b * t, d, h * w).float()
        values = (encoded * mask[:, None]) @ members.float().T
        features = (values / counts[:, None].clamp_min(1)).transpose(1, 2).reshape(b, t * slots, d).to(dtype)
    valid = (counts > 0).reshape(b, t * slots)
    # Remove slots invalid for the entire batch, so they do not occupy attention.
    keep = valid.any(0)
    support = layout.source_support.repeat_interleave(slots, 1)
    support_valid = layout.support_valid.repeat_interleave(slots, 1)
    support_valid = support_valid & valid[..., None]
    pair_times = (support[..., 0] * support_valid).sum(-1) / support_valid.sum(-1).clamp_min(1)
    parents = (torch.arange(t, device=features.device) // layout.parent_tubelets).repeat_interleave(slots)[None].expand(b, -1)
    boxes = torch.stack((x0 / w, y0 / h, x1 / w, y1 / h), -1).to(features.dtype).repeat(t, 1)[None].expand(b, -1, -1)
    corners = torch.stack((boxes[..., [0, 1]], boxes[..., [2, 1]],
                           boxes[..., [2, 3]], boxes[..., [0, 3]]), -2).float()
    if spatial_transform is not None:
        homogeneous = torch.cat((corners, torch.ones_like(corners[..., :1])), -1)
        original = torch.einsum("bij,bekj->beki", torch.linalg.inv(spatial_transform.float()), homogeneous)
        corners = original[..., :2] / original[..., 2:]
    boxes = torch.cat((corners.amin(-2), corners.amax(-2)), -1).clamp(0, 1).to(features.dtype)
    return EvidenceBatch(features[:, keep], support[:, keep], support_valid[:, keep], pair_times[:, keep],
                         boxes[:, keep], parents[:, keep], valid[:, keep],
                         features.new_ones((b, int(keep.sum()))), corners[:, keep])
