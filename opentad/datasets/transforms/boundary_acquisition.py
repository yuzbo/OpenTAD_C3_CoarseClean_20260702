import json
import re
from pathlib import Path

import numpy as np


INTEGER_TEXT_RE = re.compile(r"^[+-]?(?:0|[1-9][0-9]*)$")


FORBIDDEN_VALUE_TRANSPORT_FLAGS = (
    "uses_gt",
    "uses_teacher",
    "uses_oracle",
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "uses_checkpoint",
    "prediction_uses_gt",
)


def _is_true(value):
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _strict_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if INTEGER_TEXT_RE.fullmatch(text):
            return int(text)
    raise ValueError(f"{name} must be an integer")


def _validate_optional_context_radius_contract(row, positions, line_no, valid_len):
    if "context_radius_by_position" not in row and "selected_observations" not in row:
        return
    if row.get("selected_positions_are_centers") is not True:
        raise ValueError(f"line {line_no}: context-radius ledger rows must set selected_positions_are_centers=true")
    if row.get("context_radius_unit") != "local_dense_snippet_index":
        raise ValueError(f"line {line_no}: context_radius_unit must be local_dense_snippet_index")
    radius_range = row.get("context_radius_range")
    if not isinstance(radius_range, list) or len(radius_range) != 2:
        raise ValueError(f"line {line_no}: context_radius_range must be [min, max]")
    radius_min, radius_max = float(radius_range[0]), float(radius_range[1])
    if radius_min < 0 or radius_max < radius_min:
        raise ValueError(f"line {line_no}: invalid context_radius_range")
    radii = row.get("context_radius_by_position")
    if not isinstance(radii, list) or len(radii) != len(positions):
        raise ValueError(f"line {line_no}: context_radius_by_position length must match selected_positions")
    observations = row.get("selected_observations")
    if not isinstance(observations, list) or len(observations) != len(positions):
        raise ValueError(f"line {line_no}: selected_observations length must match selected_positions")
    expanded = row.get("expanded_selected_positions")
    if not isinstance(expanded, list) or not expanded:
        raise ValueError(f"line {line_no}: expanded_selected_positions must be a non-empty list")
    expanded_positions = [_strict_int(value, f"line {line_no}: expanded_selected_positions[{idx}]") for idx, value in enumerate(expanded)]
    if expanded_positions != sorted(expanded_positions) or len(set(expanded_positions)) != len(expanded_positions):
        raise ValueError(f"line {line_no}: expanded_selected_positions must be sorted unique")
    expected_expanded = set()
    for idx, (center, raw_radius, observation) in enumerate(zip(positions, radii, observations)):
        radius = _strict_int(raw_radius, f"line {line_no}: context_radius_by_position[{idx}]")
        if radius < radius_min or radius > radius_max:
            raise ValueError(f"line {line_no}: context_radius_by_position[{idx}] outside context_radius_range")
        if not isinstance(observation, dict):
            raise ValueError(f"line {line_no}: selected_observations[{idx}] must be a JSON object")
        if _strict_int(observation.get("center"), f"line {line_no}: selected_observations[{idx}].center") != center:
            raise ValueError(f"line {line_no}: selected_observations[{idx}].center mismatch")
        if _strict_int(observation.get("radius"), f"line {line_no}: selected_observations[{idx}].radius") != radius:
            raise ValueError(f"line {line_no}: selected_observations[{idx}].radius mismatch")
        start = max(0, center - radius)
        end = center + radius if valid_len is None else min(_strict_int(valid_len, f"line {line_no}: valid_len") - 1, center + radius)
        if _strict_int(observation.get("expanded_start"), f"line {line_no}: selected_observations[{idx}].expanded_start") != start:
            raise ValueError(f"line {line_no}: selected_observations[{idx}].expanded_start mismatch")
        if _strict_int(observation.get("expanded_end"), f"line {line_no}: selected_observations[{idx}].expanded_end") != end:
            raise ValueError(f"line {line_no}: selected_observations[{idx}].expanded_end mismatch")
        expected_expanded.update(range(start, end + 1))
    if expanded_positions != sorted(expected_expanded):
        raise ValueError(f"line {line_no}: expanded_selected_positions mismatch selected_observations")
    expanded_count = row.get("expanded_selected_count")
    if expanded_count is not None and _strict_int(expanded_count, f"line {line_no}: expanded_selected_count") != len(expanded_positions):
        raise ValueError(f"line {line_no}: expanded_selected_count mismatch")


def validate_value_transport_selection_row(row, line_no=0, require_deployable=True):
    if not isinstance(row, dict):
        raise ValueError(f"line {line_no}: value-transport row must be a JSON object")
    if row.get("selected_positions_unit") != "local_dense_index":
        raise ValueError(f"line {line_no}: selected_positions_unit must be local_dense_index")

    raw_positions = row.get("selected_positions")
    if not isinstance(raw_positions, list) or not raw_positions:
        raise ValueError(f"line {line_no}: selected_positions must be a non-empty list")
    positions = []
    for idx, value in enumerate(raw_positions):
        positions.append(_strict_int(value, f"line {line_no}: selected_positions[{idx}]"))
    if positions != sorted(positions):
        raise ValueError(f"line {line_no}: selected_positions must be sorted")
    if len(set(positions)) != len(positions):
        raise ValueError(f"line {line_no}: selected_positions must be unique")
    if positions[0] < 0:
        raise ValueError(f"line {line_no}: selected_positions must be non-negative")

    selected_count = row.get("selected_count")
    if selected_count is not None and _strict_int(selected_count, f"line {line_no}: selected_count") != len(positions):
        raise ValueError(f"line {line_no}: selected_count does not match selected_positions")
    valid_len = row.get("valid_len")
    dense_len = row.get("dense_len")
    if valid_len is not None and positions[-1] >= _strict_int(valid_len, f"line {line_no}: valid_len"):
        raise ValueError(f"line {line_no}: selected_positions exceed valid_len")
    if dense_len is not None and positions[-1] >= _strict_int(dense_len, f"line {line_no}: dense_len"):
        raise ValueError(f"line {line_no}: selected_positions exceed dense_len")
    _validate_optional_context_radius_contract(row, positions, line_no, valid_len)

    for key in FORBIDDEN_VALUE_TRANSPORT_FLAGS:
        if _is_true(row.get(key, False)):
            raise ValueError(f"line {line_no}: forbidden deploy-invisible flag {key}=true")
    if require_deployable and not bool(row.get("deploy_selection_ledger", False)):
        raise ValueError(f"line {line_no}: deployable value-transport rows must set deploy_selection_ledger=true")
    if require_deployable and bool(row.get("diagnostic_only", False)):
        raise ValueError(f"line {line_no}: deployable value-transport rows must not be diagnostic_only")
    return np.asarray(positions, dtype=np.int64)


def load_value_transport_selection_ledger(path, require_deployable=True):
    if path is None:
        raise ValueError("value-transport ledger path is required")
    ledger_path = Path(path).expanduser()
    if not ledger_path.is_file():
        raise FileNotFoundError(f"value-transport ledger not found: {ledger_path}")

    rows = {}
    with ledger_path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            positions = validate_value_transport_selection_row(
                row,
                line_no=line_no,
                require_deployable=bool(require_deployable),
            )
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"line {line_no}: sample_id must be a non-empty string")
            if sample_id in rows:
                raise ValueError(f"line {line_no}: duplicate sample_id={sample_id}")
            checked = dict(row)
            checked["selected_positions"] = positions
            rows[sample_id] = checked
    if not rows:
        raise ValueError(f"value-transport ledger has no rows: {ledger_path}")
    return rows
