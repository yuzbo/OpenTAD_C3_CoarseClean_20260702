from __future__ import annotations

import math
import struct
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from tools.bata.duca_admission_v2_1_hashing import (
    canonical_text,
    draw_mod,
    sha256_bytes,
    u32be,
    u64be,
    u8,
)
from tools.bata.duca_admission_v2_1_metrics import (
    METRIC_IDS,
    SCALE_FLOOR,
    StructuralCatastrophe,
    validate_numeric_cells,
)


METHOD_NAME = "single_step_fixed_scale_standardized_maxT"
ROLE_CONTRAST_ORDER = ("calibration", "admission_holdout")
CELL_COEFFICIENT = 1.0 / 64.0
ALPHA_TAIL = 0.05
MULTIPLIER_VALUES = (0.5, 3.0)
MULTIPLIER_PROBABILITIES = (0.8, 0.2)
MULTIPLIER_KAPPA = 1.0


def is_binary64_positive_zero(value: Any) -> bool:
    return type(value) is float and struct.pack(">d", value) == b"\x00" * 8


def positive_two_point_factor(*key_fields: bytes) -> float:
    draw = draw_mod(5, "product-multiplier-factor", *key_fields)
    factor = 3.0 if draw == 0 else 0.5
    return MULTIPLIER_KAPPA * factor


def factor_moments() -> dict[str, float]:
    values = MULTIPLIER_VALUES
    probabilities = MULTIPLIER_PROBABILITIES
    mean = math.fsum(
        value * probability for value, probability in zip(values, probabilities)
    )
    second = math.fsum(
        value * value * probability for value, probability in zip(values, probabilities)
    )
    return {"mean": mean, "variance": second - mean * mean, "second_moment": second}


def _ordered_role_cells(
    cells: Sequence[Mapping[str, Any]], role_id: str
) -> list[Mapping[str, Any]]:
    selected = [cell for cell in cells if cell.get("role_id") == role_id]
    if len(selected) != 64:
        raise StructuralCatastrophe(f"{role_id} must contain exactly 64 cells")
    rank_to_video: dict[int, str] = {}
    pairs: list[tuple[int, int]] = []
    processes: list[int] = []
    for cell in selected:
        rank = cell.get("canonical_video_rank")
        slot = cell.get("slot")
        process = cell.get("logical_process_index")
        video_id = cell.get("video_id")
        if (
            type(rank) is not int
            or type(slot) is not int
            or type(process) is not int
            or not isinstance(video_id, str)
        ):
            raise StructuralCatastrophe("statistical cell identities use invalid types")
        canonical_text(video_id, field_name="video_id")
        if not 0 <= rank < 32 or slot not in (0, 1) or not 0 <= process < 8:
            raise StructuralCatastrophe(
                "statistical cell rank/slot/process is out of range"
            )
        if cell.get("process_id") != f"{role_id}:p{process:02d}":
            raise StructuralCatastrophe("statistical cell process_id drifted")
        if rank in rank_to_video and rank_to_video[rank] != video_id:
            raise StructuralCatastrophe("one canonical rank maps to multiple video IDs")
        rank_to_video[rank] = video_id
        pairs.append((rank, slot))
        processes.append(process)
    if sorted(pairs) != [(rank, slot) for rank in range(32) for slot in (0, 1)]:
        raise StructuralCatastrophe(
            "statistical cells are not a complete 32x2 role grid"
        )
    if len(set(rank_to_video.values())) != 32:
        raise StructuralCatastrophe(
            "statistical role does not contain 32 unique videos"
        )
    if Counter(processes) != Counter({index: 8 for index in range(8)}):
        raise StructuralCatastrophe(
            "statistical role does not have process degree eight"
        )
    return sorted(
        selected, key=lambda cell: (cell["canonical_video_rank"], cell["slot"])
    )


def estimate_role_contrast(
    *,
    cells: Sequence[Mapping[str, Any]],
    scale_normalizers: Mapping[str, float],
    allow_signed_simulation_values: bool = False,
) -> dict[str, Any]:
    if allow_signed_simulation_values:
        for cell in cells:
            for metric_id in METRIC_IDS:
                value = cell.get(metric_id)
                if type(value) is not float or not math.isfinite(value):
                    raise StructuralCatastrophe(
                        f"simulation cell value {metric_id} must be finite binary64"
                    )
    else:
        validate_numeric_cells(cells)
    if set(scale_normalizers) != set(METRIC_IDS):
        raise StructuralCatastrophe(
            "scale normalizer family does not match metric registry"
        )
    for metric_id, value in scale_normalizers.items():
        if type(value) is not float or not math.isfinite(value) or value < SCALE_FLOOR:
            raise StructuralCatastrophe(f"invalid scale normalizer for {metric_id}")

    role_cells: dict[str, list[Mapping[str, Any]]] = {}
    for role_id in ROLE_CONTRAST_ORDER:
        role_cells[role_id] = _ordered_role_cells(cells, role_id)

    exact_zero: dict[str, bool] = {}
    means: dict[str, dict[str, float]] = {
        role_id: {} for role_id in ROLE_CONTRAST_ORDER
    }
    residuals: dict[str, dict[str, list[float]]] = {
        role_id: {} for role_id in ROLE_CONTRAST_ORDER
    }
    for metric_id in METRIC_IDS:
        raw = [
            cell[metric_id]
            for role_id in ROLE_CONTRAST_ORDER
            for cell in role_cells[role_id]
        ]
        exact_zero[metric_id] = all(is_binary64_positive_zero(value) for value in raw)
        for role_id in ROLE_CONTRAST_ORDER:
            values = [
                float(cell[metric_id]) / float(scale_normalizers[metric_id])
                for cell in role_cells[role_id]
            ]
            mean = math.fsum(values) * CELL_COEFFICIENT
            if not math.isfinite(mean):
                raise StructuralCatastrophe("role mean is nonfinite")
            means[role_id][metric_id] = mean
            residuals[role_id][metric_id] = [value - mean for value in values]
    delta_hat = {
        metric_id: (
            means["admission_holdout"][metric_id] - means["calibration"][metric_id]
        )
        for metric_id in METRIC_IDS
    }
    if not all(math.isfinite(value) for value in delta_hat.values()):
        raise StructuralCatastrophe("observed role contrast is nonfinite")
    return {
        "means": means,
        "residuals": residuals,
        "delta_hat": delta_hat,
        "exact_zero": exact_zero,
        "role_cells": role_cells,
    }


def _registry_hash_fields(registry_hashes: Sequence[str]) -> tuple[bytes, ...]:
    if len(registry_hashes) != 4:
        raise ValueError(
            "registry hashes must be simulation/production, role, incidence and metric"
        )
    return tuple(
        sha256_bytes(value, field_name=f"registry_hash[{index}]")
        for index, value in enumerate(registry_hashes)
    )


def multiplier_replicates(
    *,
    role_cells: Mapping[str, Sequence[Mapping[str, Any]]],
    residuals: Mapping[str, Mapping[str, Sequence[float]]],
    registry_hashes: Sequence[str],
    stream_id: int,
    replicate_start: int,
    replicate_stop: int,
) -> list[dict[str, float]]:
    if replicate_start < 0 or replicate_stop < replicate_start:
        raise ValueError("invalid replicate range")
    registry_fields = _registry_hash_fields(registry_hashes)
    stream_bytes = u32be(stream_id)
    output: list[dict[str, float]] = []
    for replicate_index in range(replicate_start, replicate_stop):
        replicate_bytes = u64be(replicate_index)
        role_vectors: dict[str, dict[str, float]] = {}
        for role_id in ROLE_CONTRAST_ORDER:
            cells = list(role_cells[role_id])
            if cells != _ordered_role_cells(cells, role_id):
                raise ValueError(
                    "multiplier cells are not in canonical rank/slot order"
                )
            role_bytes = canonical_text(role_id, field_name="role_id")
            video_weights = [
                positive_two_point_factor(
                    *registry_fields,
                    stream_bytes,
                    replicate_bytes,
                    role_bytes,
                    b"video",
                    u8(index),
                )
                for index in range(32)
            ]
            process_weights = [
                positive_two_point_factor(
                    *registry_fields,
                    stream_bytes,
                    replicate_bytes,
                    role_bytes,
                    b"process",
                    u8(index),
                )
                for index in range(8)
            ]
            role_vectors[role_id] = {}
            for metric_id in METRIC_IDS:
                metric_residuals = list(residuals[role_id][metric_id])
                if len(metric_residuals) != 64:
                    raise ValueError("residual vector must contain 64 values")
                terms = []
                for cell_index, cell in enumerate(cells):
                    video_index = int(cell["canonical_video_rank"])
                    process_index = int(cell["logical_process_index"])
                    multiplier = (
                        video_weights[video_index] * process_weights[process_index]
                        - 1.0
                    )
                    terms.append(multiplier * float(metric_residuals[cell_index]))
                value = math.fsum(terms) * CELL_COEFFICIENT
                if not math.isfinite(value):
                    raise StructuralCatastrophe("multiplier replicate is nonfinite")
                role_vectors[role_id][metric_id] = value
        output.append(
            {
                metric_id: (
                    role_vectors["admission_holdout"][metric_id]
                    - role_vectors["calibration"][metric_id]
                )
                for metric_id in METRIC_IDS
            }
        )
    return output


def type1_max_t_order_index(replicate_count: int, alpha: float) -> int:
    if type(replicate_count) is not int:
        raise TypeError("replicate_count must be an integer")
    if type(alpha) is not float:
        raise TypeError("alpha must be a binary64 float")
    if replicate_count <= 0 or not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("invalid type-1 maxT arguments")
    index = math.ceil((replicate_count + 1) * (1.0 - alpha))
    if not 1 <= index <= replicate_count:
        raise ValueError("INVALID_TYPE1_ORDER_INDEX")
    return index


def _require_finite(values: Sequence[float], *, label: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise StructuralCatastrophe(f"{label} contains a nonfinite value")


def finalize_max_t(
    *,
    delta_hat: Mapping[str, float],
    replicates: Sequence[Mapping[str, float]],
    exact_zero: Mapping[str, bool],
    alpha: float = ALPHA_TAIL,
) -> dict[str, Any]:
    if set(delta_hat) != set(METRIC_IDS) or set(exact_zero) != set(METRIC_IDS):
        raise ValueError("maxT inputs do not match the metric registry")
    if any(type(exact_zero[metric_id]) is not bool for metric_id in METRIC_IDS):
        raise ValueError("exact-zero registry values must be booleans")
    if any(
        type(delta_hat[metric_id]) is not float
        or not math.isfinite(delta_hat[metric_id])
        for metric_id in METRIC_IDS
    ):
        raise StructuralCatastrophe("observed contrasts must be finite binary64")
    for metric_id in METRIC_IDS:
        if exact_zero[metric_id] and not is_binary64_positive_zero(
            delta_hat[metric_id]
        ):
            raise StructuralCatastrophe(
                "exact-zero metric has a non-positive-zero contrast"
            )
    active = [metric_id for metric_id in METRIC_IDS if not exact_zero[metric_id]]
    lower = {metric_id: 0.0 for metric_id in METRIC_IDS}
    upper = {metric_id: 0.0 for metric_id in METRIC_IDS}
    scales = {metric_id: 0.0 for metric_id in METRIC_IDS}
    if not active:
        return {
            "method": METHOD_NAME,
            "numeric_tail_status": "PASSED_EXACT_ZERO",
            "mc_required": False,
            "active_metrics": [],
            "q_plus": 0.0,
            "q_minus": 0.0,
            "scales": scales,
            "lower": lower,
            "upper": upper,
            "t_observed": 0.0,
            "numeric_tail_passed": True,
        }
    replicate_count = len(replicates)
    if replicate_count < 2:
        raise ValueError("maxT requires at least two replicates")
    for row in replicates:
        if not isinstance(row, Mapping) or set(row) != set(METRIC_IDS):
            raise ValueError("maxT replicate row does not match the metric registry")
        if any(
            type(row[metric_id]) is not float or not math.isfinite(row[metric_id])
            for metric_id in METRIC_IDS
        ):
            raise StructuralCatastrophe(
                "maxT replicate contains non-binary64/nonfinite values"
            )
    index = type1_max_t_order_index(replicate_count, alpha)
    centered: dict[str, list[float]] = {}
    for metric_id in active:
        values = [float(row[metric_id]) for row in replicates]
        _require_finite(values, label=f"replicates:{metric_id}")
        mean = math.fsum(values) / replicate_count
        centered_values = [value - mean for value in values]
        scale = math.sqrt(
            math.fsum(value * value for value in centered_values)
            / (replicate_count - 1)
        )
        if not math.isfinite(scale) or scale <= SCALE_FLOOR:
            raise ValueError("DEGENERATE_SCALE")
        centered[metric_id] = centered_values
        scales[metric_id] = scale
    t_plus = [
        max(centered[metric_id][row] / scales[metric_id] for metric_id in active)
        for row in range(replicate_count)
    ]
    t_minus = [
        max(-centered[metric_id][row] / scales[metric_id] for metric_id in active)
        for row in range(replicate_count)
    ]
    _require_finite(t_plus, label="positive maxT")
    _require_finite(t_minus, label="negative maxT")
    q_plus = sorted(t_plus)[index - 1]
    q_minus = sorted(t_minus)[index - 1]
    for metric_id in active:
        observed = float(delta_hat[metric_id])
        if not math.isfinite(observed):
            raise StructuralCatastrophe("observed contrast is nonfinite")
        lower[metric_id] = observed - q_plus * scales[metric_id]
        upper[metric_id] = observed + q_minus * scales[metric_id]
    t_observed = max(
        float(delta_hat[metric_id]) / scales[metric_id] for metric_id in active
    )
    _require_finite(
        [q_plus, q_minus, t_observed, *lower.values(), *upper.values()],
        label="maxT result",
    )
    passed = t_observed <= q_plus
    return {
        "method": METHOD_NAME,
        "numeric_tail_status": "PASSED" if passed else "FAILED_CLOSED",
        "failure_code": None if passed else "NUMERIC_TAIL_ALARM",
        "mc_required": True,
        "active_metrics": active,
        "q_plus": q_plus,
        "q_minus": q_minus,
        "scales": scales,
        "lower": lower,
        "upper": upper,
        "t_observed": t_observed,
        "numeric_tail_passed": passed,
        "type1_order_index": index,
        "replicate_count": replicate_count,
    }
