"""Frozen, non-learning controls used by the ChronoTransport r2 gates."""

from __future__ import annotations

import hashlib
import unicodedata

import torch
from torch import Tensor

from .actions import ChronoAction
from .protocol import R2_PROTOCOL_ID, canonical_json_bytes, canonical_sha256


class InvalidImplementationError(RuntimeError):
    """Evidence must be discarded rather than silently repaired."""


_RANDOM_PREFIX = b"CT-P3R-3S-r2-random-v1\0"
R2_RANDOM_CONTROL_SEED = 3407


def _hashed_algorithm(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["sha256"] = canonical_sha256(result)
    return result


def r2_control_algorithm_identity() -> dict[str, object]:
    """Return the frozen, result-independent identity of both Gate-1 controls."""

    motion_topk = _hashed_algorithm(
        {
            "schema": "chronotransport-r2-motion-topk-algorithm-v1",
            "protocol": R2_PROTOCOL_ID,
            "periods": [2, 4, 8],
            "num_chunks": 48,
            "clip0": "RECOMPUTE",
            "fallback": "HOLD",
            "signal": "deploy_visible_cosine_change_per_window_group",
            "selection": "descending_finite_signal",
            "tie_break": "ascending_clip_index",
            "recompute_count": "exact_periodic_comparator_count_per_group",
            "nonfinite": "dense_fallback_and_INVALID_IMPLEMENTATION",
        }
    )
    random_control = _hashed_algorithm(
        {
            "schema": "chronotransport-r2-random-control-algorithm-v1",
            "protocol": R2_PROTOCOL_ID,
            "periods": [2, 4, 8],
            "num_chunks": 48,
            "clip0": "RECOMPUTE",
            "fallback": "HOLD",
            "digest": "SHA256_raw_32_bytes_ascending",
            "prefix_hex": _RANDOM_PREFIX.hex(),
            "field_encoding": "NFC_UTF8_window_id_and_no-leading-zero_decimal_ASCII_integers",
            "field_order": ["window_id", "seed", "group", "period", "clip"],
            "field_separator_hex": "00",
            "control_seed": R2_RANDOM_CONTROL_SEED,
            "tie_break": "ascending_clip_index",
            "recompute_count": "exact_periodic_comparator_count_per_group",
        }
    )
    identity: dict[str, object] = {
        "schema": "chronotransport-r2-control-algorithms-v1",
        "protocol": R2_PROTOCOL_ID,
        "motion_topk": motion_topk,
        "random": random_control,
    }
    identity["control_algorithms_sha256"] = canonical_sha256(identity)
    return identity


def validate_r2_control_algorithm_identity(identity) -> dict[str, object]:
    """Deeply validate the immutable control algorithms and all self-hashes."""

    if not isinstance(identity, dict):
        raise TypeError("control algorithm identity must be a mapping")
    for name in ("motion_topk", "random"):
        nested = identity.get(name)
        if not isinstance(nested, dict):
            raise ValueError(f"control algorithm {name} identity is missing")
        unsigned = dict(nested)
        digest = unsigned.pop("sha256", None)
        if digest != canonical_sha256(unsigned):
            raise ValueError(f"control algorithm {name} hash mismatch")
    unsigned_root = dict(identity)
    root_digest = unsigned_root.pop("control_algorithms_sha256", None)
    if root_digest != canonical_sha256(unsigned_root):
        raise ValueError("control algorithm root hash mismatch")
    expected = r2_control_algorithm_identity()
    if identity != expected or canonical_json_bytes(identity) != canonical_json_bytes(expected):
        raise ValueError("control algorithm identity differs from the frozen r2 definition")
    return dict(identity)


def _recompute_count(period: int, num_chunks: int = 48) -> int:
    if isinstance(period, bool) or not isinstance(period, int):
        raise TypeError("r2 control period must be an integer")
    if period not in (2, 4, 8):
        raise ValueError("r2 control period must be one of 2, 4, or 8")
    return len(range(0, int(num_chunks), period))


def motion_topk_actions(motion: Tensor, *, period: int) -> Tensor:
    """Select the exact periodic comparator count using visible cosine change."""

    motion = torch.as_tensor(motion)
    if motion.ndim == 2:
        motion = motion.unsqueeze(-1)
    if (
        motion.ndim != 3
        or int(motion.shape[0]) <= 0
        or int(motion.shape[1]) != 48
        or int(motion.shape[2]) != 3
    ):
        raise ValueError("motion must have shape [nonempty-B,48,3]")
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
    if not window_id or "\x00" in window_id:
        raise ValueError("window_id must be non-empty and contain no NUL")
    window_bytes = unicodedata.normalize("NFC", window_id).encode("utf-8")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed != R2_RANDOM_CONTROL_SEED
    ):
        raise ValueError(
            "seed must equal the frozen r2 random control seed 3407"
        )
    seed_bytes = str(seed).encode("ascii")
    if isinstance(num_groups, bool) or not isinstance(num_groups, int) or num_groups != 3:
        raise ValueError("num_groups must equal the frozen r2 value 3")
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
