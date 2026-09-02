"""Small numerical diagnostics shared by the DUCA correction path."""
from __future__ import annotations

import math
from typing import Any

import torch


def assert_finite_tensor(value: Any, name: str, **context: Any) -> Any:
    """Fail fast on a non-finite tensor while preserving finite-path values.

    This is intentionally a diagnostic assertion rather than a sanitiser: replacing
    NaN/Inf with zeros would hide the first invalid operation and change the model.
    """
    if torch.is_tensor(value):
        if not (value.is_floating_point() or value.is_complex()):
            return value
        finite = torch.isfinite(value.detach())
        if bool(finite.all().item()):
            return value
        total = int(finite.numel())
        finite_count = int(finite.sum().item())
        finite_ratio = float(finite_count / max(total, 1))
        finite_values = value.detach()[finite]
        if finite_values.numel():
            min_value = float(finite_values.min().item())
            max_value = float(finite_values.max().item())
        else:
            min_value = math.nan
            max_value = math.nan
        details = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": tuple(value.shape),
            "finite_ratio": finite_ratio,
            "min_finite": min_value,
            "max_finite": max_value,
        }
        details.update(context)
        raise FloatingPointError(f"non-finite tensor detected: {details}")
    if isinstance(value, float) and not math.isfinite(value):
        details = {"name": name, "value": value}
        details.update(context)
        raise FloatingPointError(f"non-finite scalar detected: {details}")
    return value
