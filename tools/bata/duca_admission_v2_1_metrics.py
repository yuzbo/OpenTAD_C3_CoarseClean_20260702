from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from tools.bata.duca_admission_v2_1_hashing import (
    PROTOCOL_ID,
    canonical_text,
    sha256_bytes,
)
from tools.bata.duca_evidence_io import with_content_sha256


METRIC_REGISTRY_SCHEMA = "duca_admission_v2_1_metric_registry_v1"
METRIC_IDS = tuple(f"M{index:02d}" for index in range(12))
METRIC_NAMES = (
    "eval.tensor_abs_max",
    "eval.tensor_abs_q999",
    "eval.tensor_rel_l2",
    "train.loss_total_abs",
    "train.loss_cls_abs",
    "train.loss_reg_abs",
    "train.grad_abs_max",
    "train.grad_rel_l2",
    "train.param_delta_abs_max",
    "decode.score_abs_max_pre_nms",
    "decode.segment_abs_max_pre_nms_true_time",
    "coordinate.roundtrip_abs_max",
)
SCALE_FLOOR = 2.0**-40


class StructuralCatastrophe(ValueError):
    failure_code = "STRUCTURAL_CATASTROPHE"


def build_metric_registry() -> dict[str, Any]:
    payload = {
        "schema": METRIC_REGISTRY_SCHEMA,
        "status": "PASSED",
        "protocol_id": PROTOCOL_ID,
        "metrics": [
            {
                "metric_id": metric_id,
                "name": name,
                "positive_direction": "worse",
                "cell_summary": "mean_of_largest_ceil_n_over_20",
            }
            for metric_id, name in zip(METRIC_IDS, METRIC_NAMES)
        ],
        "relative_l2_denominator_floor": 2.0**-24,
        "scale_floor": SCALE_FLOOR,
        "authorization_scope": "NONE",
        "phase1_v2_authorized": False,
        "holdout_open_authorized": False,
        "paper_claim_allowed": False,
        "official_final_sealed": True,
    }
    return with_content_sha256(payload)


def _finite_binary64(value: Any, *, label: str, nonnegative: bool = True) -> float:
    if type(value) is not float:
        raise StructuralCatastrophe(f"{label} must be a binary64 float")
    if not math.isfinite(value):
        raise StructuralCatastrophe(f"{label} is nonfinite")
    if nonnegative and value < 0.0:
        raise StructuralCatastrophe(f"{label} is negative")
    return value


def summarize_cell_metric(observations: Sequence[Mapping[str, Any]]) -> float:
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise StructuralCatastrophe("metric observations must be a sequence")
    if not observations:
        raise StructuralCatastrophe("metric observation set is empty")
    rows: list[tuple[float, bytes]] = []
    seen_ids: set[str] = set()
    for raw in observations:
        if not isinstance(raw, Mapping):
            raise StructuralCatastrophe("metric observation must be an object")
        observation_id = raw.get("observation_id")
        if not isinstance(observation_id, str):
            raise StructuralCatastrophe("observation_id must be a Unicode string")
        observation_bytes = canonical_text(observation_id, field_name="observation_id")
        if observation_id in seen_ids:
            raise StructuralCatastrophe("metric observation IDs must be unique")
        seen_ids.add(observation_id)
        value = _finite_binary64(
            raw.get("value"), label=f"observation {observation_id}"
        )
        rows.append((value, observation_bytes))
    rows.sort(key=lambda item: (-item[0], item[1]))
    tail_count = max(1, math.ceil(len(rows) / 20))
    return math.fsum(value for value, _identifier in rows[:tail_count]) / tail_count


def build_cell_metric_summary(
    *,
    cell_identity: Mapping[str, Any],
    observations_by_metric: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if set(observations_by_metric) != set(METRIC_IDS):
        raise StructuralCatastrophe("cell metric family does not match the registry")
    if "block_id" in cell_identity:
        raise StructuralCatastrophe("statistical cells must not contain block_id")
    output = dict(cell_identity)
    for metric_id in METRIC_IDS:
        output[metric_id] = summarize_cell_metric(observations_by_metric[metric_id])
    return output


def validate_numeric_cells(cells: Sequence[Mapping[str, Any]]) -> None:
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        raise StructuralCatastrophe("cells must be a sequence")
    cell_ids: set[str] = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise StructuralCatastrophe("cell must be an object")
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str):
            raise StructuralCatastrophe("cell IDs must be strings")
        sha256_bytes(cell_id, field_name="cell_id")
        if cell_id in cell_ids:
            raise StructuralCatastrophe("cell IDs are missing or duplicated")
        cell_ids.add(cell_id)
        if "block_id" in cell:
            raise StructuralCatastrophe("statistical cells must not contain block_id")
        for metric_id in METRIC_IDS:
            _finite_binary64(cell.get(metric_id), label=f"{cell_id}:{metric_id}")


def type1_quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("type-1 quantile requires at least one value")
    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
        raise ValueError("type-1 probability must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("type-1 quantile values must be finite")
    index = math.ceil(probability * len(ordered))
    if not 1 <= index <= len(ordered):
        raise ValueError("type-1 quantile order index is invalid")
    return ordered[index - 1]


def fit_scale_normalizers(
    scale_fit_cells: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    if len(scale_fit_cells) != 64:
        raise StructuralCatastrophe("scale_fit must contain exactly 64 cells")
    validate_numeric_cells(scale_fit_cells)
    if any(cell.get("role_id") != "scale_fit" for cell in scale_fit_cells):
        raise StructuralCatastrophe("scale normalizers may only use scale_fit cells")
    output: dict[str, float] = {}
    for metric_id in METRIC_IDS:
        # Type-1 Q_0.5 is the lower order statistic for an even sample:
        # with 64 cells this is sorted index 31, never the average of indices 31/32.
        median = type1_quantile(
            [float(cell[metric_id]) for cell in scale_fit_cells], 0.5
        )
        output[metric_id] = max(median, SCALE_FLOOR)
    return output
