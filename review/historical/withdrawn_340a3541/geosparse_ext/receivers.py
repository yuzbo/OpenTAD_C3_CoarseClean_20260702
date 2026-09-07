"""B receivers: identical evidence tensors, five explicitly different geometries."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import support_distance, union_duration


def _interpolate(times, features, queries):
    # Collapse spatial slots at the same timestamp before temporal interpolation.
    times, order = times.sort(stable=True)
    features = features[order]
    unique, inverse, counts = times.unique_consecutive(return_inverse=True, return_counts=True)
    values = features.new_zeros(len(unique), features.shape[-1]).index_add(0, inverse, features)
    values = values / counts[:, None]
    if len(unique) == 1:
        return values.expand(len(queries), -1)
    right = torch.searchsorted(unique.contiguous(), queries.contiguous()).clamp(1, len(unique) - 1)
    left = right - 1
    alpha = ((queries - unique[left]) / (unique[right] - unique[left])).clamp(0, 1)
    return values[left] + alpha[:, None] * (values[right] - values[left])


class EvidenceAttention(nn.Module):
    def __init__(self, width, heads=4, support_aware=True, allow_null=True):
        super().__init__()
        self.heads, self.support_aware, self.allow_null = heads, support_aware, allow_null
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.output = nn.Linear(width, width, bias=False)
        self.geometry_bias = nn.Sequential(nn.Linear(7 if support_aware else 1, 32), nn.GELU(), nn.Linear(32, heads))
        self.null_bias = nn.Parameter(torch.zeros(heads))

    def forward(self, query, values, times, evidence, radius_s):
        b, qn, width = query.shape
        en = values.shape[1]
        if en == 0:
            return torch.zeros_like(query)
        delta = evidence.physical_time_s[:, None] - times[:, :, None]
        if self.support_aware:
            distances = support_distance(times, evidence.source_support, evidence.support_valid)
            duration = union_duration(evidence.source_support, evidence.support_valid)
            features = torch.cat((delta[..., None], duration[:, None, :, None].expand(-1, qn, -1, -1),
                                  evidence.fidelity[:, None, :, None].expand(-1, qn, -1, -1),
                                  evidence.roi_xyxy[:, None].expand(-1, qn, -1, -1)), -1)
        else:
            distances, features = delta.abs(), delta[..., None]
        allowed = evidence.valid[:, None] & (distances <= radius_s[:, None, None])
        if not self.allow_null:
            allowed = torch.where(allowed.any(-1, keepdim=True), allowed, evidence.valid[:, None].expand(-1, qn, -1))
        split = lambda x: x.reshape(b, -1, self.heads, width // self.heads).transpose(1, 2)
        q, k, v = split(self.q(query)), split(self.k(values)), split(self.v(values))
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(width // self.heads)
        logits = logits + self.geometry_bias(features.to(query.dtype)).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~allowed[:, None], -torch.inf)
        # A zero-value null token is removed only in the registered no-null arm.
        # Even no-null needs an empty-row numerical bypass when E is wholly invalid.
        empty = ~allowed.any(-1)
        null = self.null_bias[None, :, None, None].expand(b, -1, qn, 1)
        if not self.allow_null:
            null = null.masked_fill(~empty[:, None, :, None], -torch.inf)
        probabilities = torch.cat((logits, null), -1).softmax(-1)
        out = probabilities[..., :en] @ v
        out = self.output(out.transpose(1, 2).reshape(b, qn, width))
        return out * (~empty)[..., None].to(out.dtype)


class Receiver(nn.Module):
    modes = {"rank_interp", "physical_interp", "concat_scatter", "timestamp_attention", "support_attention"}

    def __init__(self, evidence_width, width=256, mode="support_attention", layers=2, heads=4, fusion="residual"):
        super().__init__()
        if mode not in self.modes or fusion not in {"residual", "no_null", "feature_l2", "coarse_overwrite"}:
            raise ValueError("unregistered receiver/fusion")
        self.mode, self.fusion = mode, fusion
        self.evidence_proj = nn.Sequential(nn.Linear(evidence_width, width), nn.LayerNorm(width))
        self.query_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        if mode.endswith("attention"):
            self.layers = nn.ModuleList([EvidenceAttention(width, heads, mode == "support_attention", fusion != "no_null") for _ in range(layers)])
        elif mode == "concat_scatter":
            self.layers = nn.ModuleList([nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, width)) for _ in range(layers)])
        else:
            self.layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(layers)])

    def forward(self, query, times, evidence, radius_s):
        if evidence.features.shape[1] == 0:
            return query
        features = evidence.features
        if self.fusion == "feature_l2":
            features = F.normalize(features, dim=-1)
        values = self.evidence_proj(features)
        distances = support_distance(times, evidence.source_support, evidence.support_valid)
        relevant = (evidence.valid[:, None] & (distances <= radius_s[:, None, None])).any(-1)
        if self.fusion == "no_null":
            relevant = evidence.valid.any(-1)[:, None].expand_as(relevant)
        x = query
        for norm, layer in zip(self.query_norms, self.layers):
            if self.mode.endswith("attention"):
                delta = layer(norm(x), values, times, evidence, radius_s)
            else:
                received = torch.zeros_like(x)
                received_mask = torch.zeros_like(relevant)
                for b in range(x.shape[0]):
                    valid = evidence.valid[b]
                    if not bool(valid.any()):
                        continue
                    ev, ts = values[b, valid], evidence.physical_time_s[b, valid]
                    if self.mode == "rank_interp":
                        ts, ix = ts.sort(stable=True)
                        unique, inv, counts = ts.unique_consecutive(return_inverse=True, return_counts=True)
                        pooled = ev.new_zeros(len(unique), ev.shape[-1]).index_add(0, inv, ev[ix]) / counts[:, None]
                        received[b] = F.interpolate(pooled.T[None], size=x.shape[1], mode="linear", align_corners=False)[0].T
                        received_mask[b] = True
                    elif self.mode == "physical_interp":
                        received[b] = _interpolate(ts, ev, times[b])
                        received_mask[b] = relevant[b]
                    else:
                        idx = (times[b, :, None] - ts[None]).abs().argmin(0)
                        counts = ev.new_zeros(x.shape[1]).index_add(0, idx, ev.new_ones(len(idx)))
                        received[b] = received[b].index_add(0, idx, ev) / counts[:, None].clamp_min(1)
                        received_mask[b] = counts > 0
                delta = layer(torch.cat((norm(x), received), -1) if self.mode == "concat_scatter" else received)
                delta = delta * received_mask[..., None].to(x.dtype)
            if self.fusion == "coarse_overwrite":
                x = torch.where(relevant[..., None], delta, x)
            else:
                x = x + delta
        return x
