"""Frozen, non-learning controls used by the ChronoTransport r2 gates."""

from __future__ import annotations

import hashlib
import unicodedata

import torch
from torch import Tensor

from .actions import ChronoAction


class InvalidImplementationError(RuntimeError):
    """Evidence must be discarded rather than silently repaired."""


_RANDOM_PREFIX = b"CT-P3R-3S-r2-random-v1\0"


def _recompute_count(period: int, num_chunks: int = 48) -> int:
    period = int(period)
    if period not in (2, 4, 8):
        raise ValueError("r2 control period must be one of 2, 4, or 8")
    return len(range(0, int(num_chunks), period))


def motion_topk_actions(motion: Tensor, *, period: int) -> Tensor:
    """Select the exact periodic comparator count using visible cosine change."""

    motion = torch.as_tensor(motion)
    if motion.ndim == 2:
        motion = motion.unsqueeze(-1)
    if motion.ndim != 3 or int(motion.shape[1]) != 48:
        raise ValueError("motion must have shape [B,48] or [B,48,G]")
    if not bool(torch.isfinite(motion).all().item()):
        raise InvalidImplementationError("non-finite motion requires dense invalid-implementation fallback")
    count = _recompute_count(period)
    actions = torch.full(motion.shape, int(ChronoAction.HOLD), dtype=torch.long, device=motion.device)
    actions[:, 0, :] = int(ChronoAction.RECOMPUTE)
    values = motion.detach().cpu()
    for batch in range(int(motion.shape[0])):
        for group in range(int(motion.shape[2])):
            ranked = sorted(range(1, 48), key=lambda clip: (-float(values[batch, clip, group]), clip))
            selected = ranked[: count - 1]
            actions[batch, selected, group] = int(ChronoAction.RECOMPUTE)
    return actions


def random_exact_count_actions(
    window_id: str,
    *,
    seed: int,
    num_groups: int,
    period: int,
) -> Tensor:
    """Build the frozen hash-ranked random comparator for one window."""

    if not isinstance(window_id, str):
        raise TypeError("window_id must be a string")
    window_bytes = unicodedata.normalize("NFC", window_id).encode("utf-8")
    seed_bytes = str(int(seed)).encode("ascii")
    num_groups = int(num_groups)
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    period = int(period)
    count = _recompute_count(period)
    actions = torch.full((48, num_groups), int(ChronoAction.HOLD), dtype=torch.long)
    actions[0, :] = int(ChronoAction.RECOMPUTE)
    for group in range(num_groups):
        group_bytes = str(group).encode("ascii")
        ranked: list[tuple[bytes, int]] = []
        for clip in range(1, 48):
            digest = hashlib.sha256(
                _RANDOM_PREFIX
                + window_bytes
                + b"\0"
                + seed_bytes
                + b"\0"
                + group_bytes
                + b"\0"
                + str(period).encode("ascii")
                + b"\0"
                + str(clip).encode("ascii")
            ).digest()
            ranked.append((digest, clip))
        for _, clip in sorted(ranked)[: count - 1]:
            actions[clip, group] = int(ChronoAction.RECOMPUTE)
    return actions

