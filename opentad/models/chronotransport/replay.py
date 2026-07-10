from __future__ import annotations

from contextlib import contextmanager
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .losses import nonnegative_detector_regret


@dataclass
class RNGSnapshot:
    python_state: object
    numpy_state: tuple
    torch_cpu_state: Tensor
    torch_cuda_state: list[Tensor] | None

    @classmethod
    def capture(cls) -> "RNGSnapshot":
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        return cls(random.getstate(), np.random.get_state(), torch.get_rng_state(), cuda_state)

    def restore(self) -> None:
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_cpu_state)
        if self.torch_cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.torch_cuda_state)


def _loss_total(losses: Mapping[str, Any]) -> Tensor:
    values = [
        value
        for key, value in losses.items()
        if "loss" in str(key).lower() and isinstance(value, Tensor)
    ]
    if not values and isinstance(losses.get("cost"), Tensor):
        values = [losses["cost"]]
    if not values:
        raise ValueError("detector forward produced no tensor loss")
    return torch.stack([value.float().mean() for value in values]).sum()


@contextmanager
def runtime_schedule(model: nn.Module, schedule: str) -> Iterator[None]:
    changed = []
    for module in model.modules():
        if module.__class__.__name__ == "ChronoTransportRuntime":
            changed.append((module, module.forced_schedule))
            module.forced_schedule = str(schedule)
    if not changed:
        raise ValueError("model has no ChronoTransportRuntime")
    try:
        yield
    finally:
        for module, previous in changed:
            module.forced_schedule = previous


@dataclass
class PairedReplayResult:
    dense_losses: Mapping[str, Any]
    counterfactual_losses: Mapping[str, Any]
    dense_total: Tensor
    counterfactual_total: Tensor
    regret: Tensor


def paired_detector_losses(
    detector: nn.Module,
    forward_kwargs: Mapping[str, Any],
    *,
    counterfactual_schedule: str,
    track_counterfactual_grad: bool = True,
) -> PairedReplayResult:
    initial = RNGSnapshot.capture()
    with runtime_schedule(detector, "dense"), torch.no_grad():
        dense_losses = detector(**dict(forward_kwargs))
        dense_total = _loss_total(dense_losses)
    initial.restore()
    counterfactual_context = nullcontext() if track_counterfactual_grad else torch.no_grad()
    with runtime_schedule(detector, counterfactual_schedule), counterfactual_context:
        counterfactual_losses = detector(**dict(forward_kwargs))
        counterfactual_total = _loss_total(counterfactual_losses)
    initial.restore()
    regret = nonnegative_detector_regret(counterfactual_total, dense_total)
    return PairedReplayResult(
        dense_losses=dense_losses,
        counterfactual_losses=counterfactual_losses,
        dense_total=dense_total.detach(),
        counterfactual_total=counterfactual_total,
        regret=regret,
    )


_COMPACT_KEYS = {
    "sample_id",
    "split",
    "schedule",
    "signals",
    "pooled_targets",
    "cost",
    "regret",
    "endpoint_regret",
    "high_iou_regret",
    "short_action_regret",
}


def validate_compact_record(record: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(record) - _COMPACT_KEYS
    if unknown:
        raise ValueError(f"counterfactual record contains forbidden keys: {sorted(unknown)}")
    required = {"sample_id", "split", "schedule", "cost", "regret"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"counterfactual record missing keys: {sorted(missing)}")
    if str(record["split"]) not in {"train", "fit", "calibration", "evaluation", "diagnostic"}:
        raise ValueError("invalid counterfactual split")
    return dict(record)


def canonical_record_line(record: Mapping[str, Any]) -> str:
    return json.dumps(validate_compact_record(record), sort_keys=True, separators=(",", ":"))


def records_sha256(records: list[Mapping[str, Any]]) -> str:
    payload = ("\n".join(canonical_record_line(item) for item in records) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
