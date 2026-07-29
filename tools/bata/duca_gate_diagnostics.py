from __future__ import annotations

import math
from collections import Counter
from numbers import Integral
from collections.abc import Mapping, Sequence
from typing import Any


def uniform_axis_geometry_report(
    *,
    valid_len: int,
    positions: Sequence[int],
) -> dict[str, Any]:
    """Describe whether integer exact-uniform anchors define one affine map.

    A globally affine selected-to-physical coordinate map is a necessary, but
    not sufficient, precondition for detector-loss conjugacy.  The protected
    gate records this report so a non-affine integer grid cannot be mistaken
    for a mere floating-point tolerance miss.
    """

    if isinstance(valid_len, bool) or not isinstance(valid_len, Integral):
        raise TypeError("valid_len must be an integer")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in positions):
        raise TypeError("positions must contain only integers")
    valid_len = int(valid_len)
    anchors = [int(value) for value in positions]
    if valid_len <= 0:
        raise ValueError("valid_len must be positive")
    if not anchors:
        raise ValueError("positions must contain at least one anchor")
    if any(value < 0 or value >= valid_len for value in anchors):
        raise ValueError("positions must lie inside the valid physical axis")
    if any(right <= left for left, right in zip(anchors, anchors[1:])):
        raise ValueError("positions must be strictly increasing")

    effective_k = len(anchors)
    endpoint_anchored = anchors[0] == 0 and anchors[-1] == valid_len - 1
    steps = [right - left for left, right in zip(anchors, anchors[1:])]
    step_histogram: dict[str, int] = {}
    for step in steps:
        key = str(int(step))
        step_histogram[key] = step_histogram.get(key, 0) + 1

    if effective_k == 1:
        divisibility_precondition = valid_len == 1
        globally_affine = valid_len == 1 and endpoint_anchored
        nominal_step = None
    else:
        divisibility_precondition = (valid_len - 1) % (effective_k - 1) == 0
        globally_affine = endpoint_anchored and len(step_histogram) == 1
        nominal_step = float(valid_len - 1) / float(effective_k - 1)

    return {
        "schema": "duca_uniform_axis_geometry_diagnostic_v1",
        "valid_len": valid_len,
        "effective_k": effective_k,
        "first_anchor": anchors[0],
        "last_anchor": anchors[-1],
        "endpoint_anchored": endpoint_anchored,
        "nominal_physical_step": nominal_step,
        "physical_step_histogram": step_histogram,
        "min_physical_step": min(steps) if steps else None,
        "max_physical_step": max(steps) if steps else None,
        "divisibility_precondition_satisfied": divisibility_precondition,
        "global_affine_coordinate_precondition_satisfied": globally_affine,
        "interpretation": (
            "global affine coordinate conjugacy remains only a necessary precondition"
            if globally_affine
            else "scalar detector-loss equality is not implied by this non-affine integer grid"
        ),
    }


def implemented_uniform_axis_geometry_report(
    *,
    valid_len: int,
    positions: Sequence[int],
) -> dict[str, Any]:
    """Report the complete half-open map implemented by ``TrueTimeMap``.

    The legacy anchor-only report omits the terminal selected coordinate
    ``K -> valid_len`` and, when the first selected anchor is positive, the
    leading ``-1 -> 0`` knot.  Both extensions affect the actual piecewise
    linear coordinate map and therefore must be represented in evidence.
    """

    anchor_report = uniform_axis_geometry_report(
        valid_len=valid_len,
        positions=positions,
    )
    anchors = [int(value) for value in positions]
    effective_k = len(anchors)

    selected_knots = list(range(effective_k))
    physical_knots = list(anchors)
    leading_extension = anchors[0] > 0
    if leading_extension:
        selected_knots.insert(0, -1)
        physical_knots.insert(0, 0)

    # This terminal knot is always present for valid integer anchors because
    # the last anchor is at most valid_len - 1.
    terminal_extension = anchors[-1] < int(valid_len)
    if terminal_extension:
        selected_knots.append(effective_k)
        physical_knots.append(int(valid_len))

    selected_steps = [
        right - left for left, right in zip(selected_knots, selected_knots[1:])
    ]
    physical_steps = [
        right - left for left, right in zip(physical_knots, physical_knots[1:])
    ]
    if any(step <= 0 for step in selected_steps) or any(
        step <= 0 for step in physical_steps
    ):
        raise ValueError("implemented coordinate map must be strictly increasing")

    slope_pairs = [
        (physical_step, selected_step)
        for physical_step, selected_step in zip(physical_steps, selected_steps)
    ]
    slope_histogram = Counter(
        f"{physical_step}/{selected_step}"
        for physical_step, selected_step in slope_pairs
    )
    implemented_global_affine = len(set(slope_pairs)) == 1
    identity_map = (
        selected_knots == physical_knots and implemented_global_affine
    )
    return {
        "schema": "duca_implemented_axis_geometry_v2",
        "valid_len": int(valid_len),
        "effective_k": effective_k,
        "anchor_grid": anchor_report,
        "anchor_grid_affine": bool(
            anchor_report[
                "global_affine_coordinate_precondition_satisfied"
            ]
        ),
        "implemented_selected_knots": selected_knots,
        "implemented_physical_knots": physical_knots,
        "leading_half_open_extension": {
            "applied": leading_extension,
            "selected_knot": -1 if leading_extension else None,
            "physical_knot": 0 if leading_extension else None,
        },
        "terminal_half_open_extension": {
            "applied": terminal_extension,
            "selected_knot": effective_k if terminal_extension else None,
            "physical_knot": int(valid_len) if terminal_extension else None,
        },
        "implemented_physical_step_histogram": {
            str(step): int(count)
            for step, count in sorted(Counter(physical_steps).items())
        },
        "implemented_slope_histogram": {
            key: int(value) for key, value in sorted(slope_histogram.items())
        },
        "implemented_global_affine": bool(implemented_global_affine),
        "identity_map": bool(identity_map),
        "general_loss_conjugacy_applicable": bool(identity_map),
        "interpretation": (
            "identity/no-op conjugacy fixture"
            if identity_map
            else "physical-vs-selected scalar loss equality is diagnostic only"
        ),
    }


def detector_loss_parity_report(
    physical_losses: Mapping[str, float],
    selected_axis_losses: Mapping[str, float],
    *,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Return a JSON-safe, per-key detector-loss comparison."""

    relative_tolerance = float(relative_tolerance)
    if not math.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and non-negative")
    physical_keys = set(physical_losses)
    selected_keys = set(selected_axis_losses)
    if not physical_keys or not selected_keys:
        raise ValueError("detector loss mappings must be non-empty")
    if physical_keys != selected_keys:
        raise ValueError(
            "physical and selected-axis detector-loss keys differ: "
            f"physical_only={sorted(physical_keys - selected_keys)}, "
            f"selected_only={sorted(selected_keys - physical_keys)}"
        )

    per_key: dict[str, dict[str, Any]] = {}
    for key in sorted(physical_keys):
        physical = float(physical_losses[key])
        selected = float(selected_axis_losses[key])
        if not math.isfinite(physical) or not math.isfinite(selected):
            raise ValueError(f"detector loss {key!r} is non-finite")
        absolute_error = abs(physical - selected)
        scale = max(1.0, abs(physical), abs(selected))
        threshold = relative_tolerance * scale
        per_key[key] = {
            "physical": physical,
            "selected_axis": selected,
            "absolute_error": absolute_error,
            "normalized_error": absolute_error / scale,
            "scale": scale,
            "threshold": threshold,
            "equal_within_registered_tolerance": absolute_error <= threshold,
        }

    worst_key = max(
        per_key,
        key=lambda key: per_key[key]["normalized_error"],
        default=None,
    )
    return {
        "schema": "duca_detector_loss_parity_diagnostic_v1",
        "relative_tolerance": relative_tolerance,
        "all_equal_within_registered_tolerance": all(
            row["equal_within_registered_tolerance"] for row in per_key.values()
        ),
        "worst_key": worst_key,
        "per_key": per_key,
    }
