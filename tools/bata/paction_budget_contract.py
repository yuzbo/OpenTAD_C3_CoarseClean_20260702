from __future__ import annotations

import math


def expected_selected_count(
    required_count: int | None,
    *,
    valid_len: int,
    dense_len: int,
    allow_short_valid_ratio_count: bool,
) -> int | None:
    if required_count is None:
        return None
    required = int(required_count)
    valid = int(valid_len)
    dense = int(dense_len)
    if required <= 0:
        return 0
    if valid <= 0:
        raise ValueError(f"valid_len must be positive, got {valid_len}")
    if (not allow_short_valid_ratio_count) or dense <= 0 or valid >= dense:
        return min(required, valid)
    scaled = int(math.ceil(float(valid) * float(required) / float(dense)))
    return max(1, min(required, valid, scaled))


def required_count_matches(
    actual_count: int,
    required_count: int | None,
    *,
    valid_len: int,
    dense_len: int,
    allow_short_valid_ratio_count: bool,
) -> tuple[bool, int | None]:
    expected = expected_selected_count(
        required_count,
        valid_len=valid_len,
        dense_len=dense_len,
        allow_short_valid_ratio_count=allow_short_valid_ratio_count,
    )
    if expected is None:
        return True, None
    return int(actual_count) == int(expected), int(expected)
