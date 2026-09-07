"""Exact ordered Plackett–Luce log-probability for hard subset sampling.

The sampling order is a latent policy action. Sort by native coordinates before
video execution, but preserve this order for the score-function estimator.
These primitives do not implement a video backbone or train an experiment.
"""
from __future__ import annotations
import torch
from torch import Tensor


def ordered_log_prob(logits: Tensor, order: Tensor) -> Tensor:
    if logits.ndim != 1 or order.ndim != 1:
        raise ValueError("expected one-dimensional tensors")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite floating point")
    if order.dtype != torch.long or order.device != logits.device:
        raise ValueError("order must be int64 on logits.device")
    n, k = logits.numel(), order.numel()
    if k > n or (k and (int(order.min()) < 0 or int(order.max()) >= n)):
        raise ValueError("invalid selected index")
    if order.unique().numel() != k:
        raise ValueError("duplicate selected index")
    if not k:
        return logits.sum() * 0.0
    # Preserve float64 tests; accumulate low-precision model logits in float32.
    x = logits if logits.dtype == torch.float64 else logits.float()
    selected = x[order]
    mask = torch.ones(n, dtype=torch.bool, device=x.device)
    mask[order] = False
    remaining_logsum = torch.logsumexp(x[mask], dim=0)
    reverse_selected = torch.logcumsumexp(selected.flip(0), dim=0).flip(0)
    denom = torch.logaddexp(reverse_selected, remaining_logsum)
    return (selected - denom).sum()


def sample_ordered_topk(logits: Tensor, k: int, generator=None) -> tuple[Tensor, Tensor]:
    if logits.ndim != 1 or not 0 <= k <= logits.numel():
        raise ValueError("invalid shape or k")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite floating point")
    if k in (0, logits.numel()):
        # Full/empty execution is deterministic; avoid pointless order variance.
        order = torch.arange(k, dtype=torch.long, device=logits.device)
        return order, logits.sum()*0.0
    x = logits if logits.dtype == torch.float64 else logits.float()
    u = torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    eps = torch.finfo(x.dtype).eps
    u = u.clamp(min=eps, max=1.0-eps)
    gumbel = -torch.log(-torch.log(u))
    order = torch.topk(x+gumbel, k=k, sorted=True).indices
    return order, ordered_log_prob(logits, order)


def actor_loss(reward_cost: Tensor, baseline: Tensor, joint_log_prob: Tensor) -> Tensor:
    """Minimize: higher-than-baseline cost should become less probable."""
    return ((reward_cost - baseline).detach() * joint_log_prob).mean()
