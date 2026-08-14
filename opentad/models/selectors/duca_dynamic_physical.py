"""Minimal DUCA dynamic-B physical acquisition primitives.

Infrastructure-only helpers: deterministic outer-K policy, bounded monotone/local
exact-K transport, canonical uniform positions (F1), and nonce Fisher-Yates (F2).
"""
from __future__ import annotations

import hashlib
from typing import Iterable


def dynamic_outer_k(score: float, *, min_k: int, target_k: int, max_k: int) -> int:
    """Map a normalized deploy-visible score to a bounded monotone integer K."""
    if not (0 < min_k <= target_k <= max_k):
        raise ValueError("require 0 < min_k <= target_k <= max_k")
    x = max(0.0, min(1.0, float(score)))
    if x <= 0.5:
        value = min_k + (target_k - min_k) * (x / 0.5)
    else:
        value = target_k + (max_k - target_k) * ((x - 0.5) / 0.5)
    return max(min_k, min(max_k, int(value + 0.5)))


def f1_uniform_positions(valid_len: int, k: int) -> list[int]:
    """Endpoint-inclusive integer-half-up exact-uniform positions."""
    n, count = int(valid_len), int(k)
    if n <= 0 or count <= 0 or count > n:
        raise ValueError("require 0 < k <= valid_len")
    if count == 1:
        return [0]
    return [int((i * (n - 1) * 2 + (count - 1)) // (2 * (count - 1))) for i in range(count)]


def f2_nonce_shuffle(rows: Iterable[int], nonce: str) -> list[int]:
    """Canonical-row-order deterministic Fisher--Yates using nonce-derived stream."""
    out = [int(x) for x in rows]
    state = hashlib.sha256(str(nonce).encode("utf-8")).digest()
    for i in range(len(out) - 1, 0, -1):
        state = hashlib.sha256(state + i.to_bytes(8, "big")).digest()
        j = int.from_bytes(state[:8], "big") % (i + 1)
        out[i], out[j] = out[j], out[i]
    return out


def bounded_monotone_local_exact_k(
    scores: Iterable[float], k: int, *, local_radius: int, valid_mask: Iterable[bool] | None = None
) -> list[int]:
    """Select exactly K unique indices, preserving order and bounded locality."""
    values = [float(x) for x in scores]
    n = len(values)
    if not (0 < int(k) <= n) or int(local_radius) < 0:
        raise ValueError("require 0 < k <= len(scores) and local_radius >= 0")
    valid = [True] * n if valid_mask is None else [bool(x) for x in valid_mask]
    candidates = [i for i, ok in enumerate(valid) if ok]
    if len(candidates) < int(k):
        raise ValueError("valid_mask contains fewer than k positions")
    radius = int(local_radius)
    anchors = f1_uniform_positions(len(candidates), int(k))
    selected = sorted({candidates[i] for i in anchors})
    # Radius zero is a real locality constraint, not an escape hatch.  More
    # than one unique exact-K position cannot satisfy it.
    if radius == 0 and len(selected) > 1:
        raise ValueError("locality contract impossible for local_radius=0 and k>1")
    for idx in sorted(candidates, key=lambda i: (-values[i], i)):
        if len(selected) >= int(k):
            break
        if all(abs(idx - j) <= radius for j in selected):
            selected.append(idx)
            selected.sort()
    if len(selected) < int(k):
        raise ValueError("locality contract impossible for requested exact-K selection")
    return selected[: int(k)]


def attach_physical_timestamps(meta: dict, positions: Iterable[int], *, fps: float, window_start: float = 0.0) -> dict:
    """Carry physical coordinates before downstream filtering/NMS/serialization."""
    if float(fps) <= 0:
        raise ValueError("fps must be positive")
    pos = [int(x) for x in positions]
    meta = dict(meta)
    meta["duca_physical_positions"] = pos
    meta["duca_physical_timestamps"] = [float(window_start) + x / float(fps) for x in pos]
    meta["duca_timestamp_stage"] = "before_filter_topk_iou_nms_voting_serialization"
    return meta


__all__ = ["dynamic_outer_k", "f1_uniform_positions", "f2_nonce_shuffle", "bounded_monotone_local_exact_k", "attach_physical_timestamps"]
