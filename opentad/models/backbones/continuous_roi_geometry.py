from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


CONTINUOUS_ROI_GEOMETRY_SCHEMA = "bounded_area_pixel_aspect_v1"
CONTINUOUS_ROI_GENERATOR_SCHEMA = "stateless_philox_common_support_v1"

AREA_MIN = 0.18
AREA_MAX = 0.36
RATIO_MIN = 0.75
RATIO_MAX = 2.25
SOURCE_ASPECT = 16.0 / 9.0
ANCHOR_LOGITS = (0.0, 0.0, 0.3237870770938973, -1.0363260485493035)
FAMILY_CYCLE = ("anchor", "fixed_size", "variable_size")

_UINT32_MASK = (1 << 32) - 1
_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85


def _mulhilo32(lhs: int, rhs: int) -> tuple[int, int]:
    product = (int(lhs) & _UINT32_MASK) * (int(rhs) & _UINT32_MASK)
    return (product >> 32) & _UINT32_MASK, product & _UINT32_MASK


def _philox4x32_10(
    counter: Sequence[int],
    key: Sequence[int],
) -> tuple[int, int, int, int]:
    """Random123-compatible Philox4x32-10 block."""

    if len(counter) != 4 or len(key) != 2:
        raise ValueError("Philox4x32 requires a four-word counter and two-word key")
    c0, c1, c2, c3 = (int(value) & _UINT32_MASK for value in counter)
    k0, k1 = (int(value) & _UINT32_MASK for value in key)
    for round_index in range(10):
        hi0, lo0 = _mulhilo32(_PHILOX_M0, c0)
        hi1, lo1 = _mulhilo32(_PHILOX_M1, c2)
        c0, c1, c2, c3 = (
            (hi1 ^ c1 ^ k0) & _UINT32_MASK,
            lo1,
            (hi0 ^ c3 ^ k1) & _UINT32_MASK,
            lo0,
        )
        if round_index != 9:
            k0 = (k0 + _PHILOX_W0) & _UINT32_MASK
            k1 = (k1 + _PHILOX_W1) & _UINT32_MASK
    return c0, c1, c2, c3


def _increment_counter(counter: list[int]) -> None:
    for index in range(4):
        counter[index] = (counter[index] + 1) & _UINT32_MASK
        if counter[index] != 0:
            return


def stateless_philox_uniform(
    shape: Sequence[int],
    *,
    key_parts: Sequence[object],
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate a stateless Philox tensor from an explicit semantic key."""

    shape = tuple(int(value) for value in shape)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError("Philox output shape must contain only positive dimensions")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("stateless Philox output must be float32 or float64")
    digest = hashlib.sha256(
        "\x1f".join(str(value) for value in key_parts).encode("utf-8")
    ).digest()
    words = [
        int.from_bytes(digest[offset : offset + 4], "little")
        for offset in range(0, 24, 4)
    ]
    key = words[:2]
    counter = list(words[2:6])
    count = math.prod(shape)
    values = []
    while len(values) < count:
        values.extend(_philox4x32_10(counter, key))
        _increment_counter(counter)
    scale = float(1 << 32)
    tensor = torch.tensor(
        [(value + 0.5) / scale for value in values[:count]],
        dtype=torch.float64,
    ).reshape(shape)
    return tensor.to(device=device, dtype=dtype)


def anchor_knot_logits(
    *,
    batch_size: int = 1,
    knots: int = 12,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if batch_size <= 0 or knots <= 0:
        raise ValueError("batch_size and knots must be positive")
    anchor = torch.tensor(ANCHOR_LOGITS, dtype=dtype, device=device)
    return anchor.view(1, 1, 4).expand(batch_size, knots, 4).clone()


def temporal_filter_knot_logits(logits: torch.Tensor, *, passes: int = 2) -> torch.Tensor:
    """Apply the registered replicate-padded [0.25, 0.5, 0.25] filter."""

    if logits.ndim < 2 or logits.shape[-1] != 4:
        raise ValueError("knot logits must end in [K,4]")
    if logits.shape[-2] <= 0 or passes < 0:
        raise ValueError("knot count must be positive and passes non-negative")
    output = logits
    for _ in range(int(passes)):
        previous = torch.cat((output[..., :1, :], output[..., :-1, :]), dim=-2)
        following = torch.cat((output[..., 1:, :], output[..., -1:, :]), dim=-2)
        output = 0.25 * previous + 0.50 * output + 0.25 * following
    return output


def interpolate_knot_logits(logits: torch.Tensor, *, clips: int = 48) -> torch.Tensor:
    """Interpolate knot logits to clip logits while preserving both endpoints."""

    if logits.ndim != 3 or logits.shape[-1] != 4:
        raise ValueError("interpolate_knot_logits expects [B,K,4]")
    if clips <= 0:
        raise ValueError("clips must be positive")
    if logits.shape[1] == 1:
        return logits.expand(logits.shape[0], clips, 4).clone()
    return F.interpolate(
        logits.transpose(1, 2),
        size=int(clips),
        mode="linear",
        align_corners=True,
    ).transpose(1, 2)


def decode_continuous_roi_logits(
    logits: torch.Tensor,
    *,
    area_min: float = AREA_MIN,
    area_max: float = AREA_MAX,
    ratio_min: float = RATIO_MIN,
    ratio_max: float = RATIO_MAX,
    source_aspect: float = SOURCE_ASPECT,
) -> torch.Tensor:
    """Decode bounded logits into normalized source boxes `(cx, cy, w, h)`."""

    if logits.shape[-1] != 4:
        raise ValueError("continuous ROI logits must have four channels")
    if not 0.0 < area_min < area_max <= 1.0:
        raise ValueError("invalid area bounds")
    if not 0.0 < ratio_min < ratio_max:
        raise ValueError("invalid pixel-aspect bounds")
    if source_aspect <= 0.0:
        raise ValueError("source_aspect must be positive")
    sx, sy, sa, sr = logits.unbind(dim=-1)
    area = area_min + (area_max - area_min) * torch.sigmoid(sa)
    log_ratio_min = math.log(ratio_min)
    log_ratio_span = math.log(ratio_max / ratio_min)
    ratio = torch.exp(log_ratio_min + log_ratio_span * torch.sigmoid(sr))
    width = torch.sqrt(area * ratio / source_aspect)
    height = torch.sqrt(area * source_aspect / ratio)
    center_x = 0.5 * width + (1.0 - width) * torch.sigmoid(sx)
    center_y = 0.5 * height + (1.0 - height) * torch.sigmoid(sy)
    boxes = torch.stack((center_x, center_y, width, height), dim=-1)
    if not bool(torch.isfinite(boxes).all().item()):
        raise FloatingPointError("continuous ROI decoder produced non-finite boxes")
    tolerance = 1e-6
    left = center_x - 0.5 * width
    right = center_x + 0.5 * width
    top = center_y - 0.5 * height
    bottom = center_y + 0.5 * height
    if bool(
        (
            (left < -tolerance)
            | (top < -tolerance)
            | (right > 1.0 + tolerance)
            | (bottom > 1.0 + tolerance)
        ).any().item()
    ):
        raise RuntimeError("analytic continuous ROI decoder violated source bounds")
    return boxes


def family_for_sample(successful_update: int, batch_slot: int) -> str:
    if successful_update < 0 or batch_slot < 0:
        raise ValueError("successful_update and batch_slot must be non-negative")
    return FAMILY_CYCLE[(2 * int(successful_update) + int(batch_slot)) % 3]


def exogenous_common_support_logits(
    *,
    training_seed: int,
    successful_update: int,
    sample_keys: Sequence[int] | torch.Tensor,
    window_starts: Sequence[int] | torch.Tensor,
    knots: int = 12,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Generate the registered retry-stable common-support knot logits."""

    if successful_update < 0:
        raise ValueError("successful_update must be non-negative")
    sample_keys = torch.as_tensor(sample_keys, dtype=torch.int64).reshape(-1)
    window_starts = torch.as_tensor(window_starts, dtype=torch.int64).reshape(-1)
    if sample_keys.numel() == 0 or sample_keys.shape != window_starts.shape:
        raise ValueError("sample_keys and window_starts must be aligned and non-empty")
    anchor = torch.tensor(ANCHOR_LOGITS, dtype=dtype, device=device)
    rows = []
    families = []
    for batch_slot, (sample_key, window_start) in enumerate(
        zip(sample_keys.tolist(), window_starts.tolist())
    ):
        family = family_for_sample(successful_update, batch_slot)
        families.append(family)
        if family == "anchor":
            row = anchor.view(1, 4).expand(knots, 4).clone()
        else:
            uniform = stateless_philox_uniform(
                (knots, 4),
                key_parts=(
                    CONTINUOUS_ROI_GENERATOR_SCHEMA,
                    int(training_seed),
                    int(successful_update),
                    int(batch_slot),
                    int(sample_key),
                    int(window_start),
                ),
                dtype=dtype,
                device=device,
            )
            row = torch.empty_like(uniform)
            row[:, 0] = 3.0 * (2.0 * uniform[:, 0] - 1.0)
            row[:, 1] = 3.0 * (2.0 * uniform[:, 1] - 1.0)
            row[:, 2] = anchor[2] + 1.5 * (2.0 * uniform[:, 2] - 1.0)
            row[:, 3] = anchor[3] + 1.5 * (2.0 * uniform[:, 3] - 1.0)
            if family == "fixed_size":
                row[:, 2:] = anchor[2:]
        rows.append(temporal_filter_knot_logits(row, passes=2))
    return torch.stack(rows, dim=0), tuple(families)


def common_support_clip_boxes(
    *,
    training_seed: int,
    successful_update: int,
    sample_keys: Sequence[int] | torch.Tensor,
    window_starts: Sequence[int] | torch.Tensor,
    knots: int = 12,
    clips: int = 48,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, tuple[str, ...], torch.Tensor]:
    knot_logits, families = exogenous_common_support_logits(
        training_seed=training_seed,
        successful_update=successful_update,
        sample_keys=sample_keys,
        window_starts=window_starts,
        knots=knots,
        dtype=dtype,
        device=device,
    )
    clip_logits = interpolate_knot_logits(knot_logits, clips=clips)
    return decode_continuous_roi_logits(clip_logits), families, knot_logits
