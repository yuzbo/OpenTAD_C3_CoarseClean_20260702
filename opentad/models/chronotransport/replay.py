from __future__ import annotations

from contextlib import contextmanager
from contextlib import nullcontext
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import random
import re
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .losses import nonnegative_detector_regret
from .protocol import canonical_sha256
from .scheduler import R2_NON_DENSE_NAMES


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


@dataclass
class StatefulObjectSnapshot:
    name: str
    target: Any
    state: Any

    @classmethod
    def capture(cls, name: str, target: Any) -> "StatefulObjectSnapshot":
        if not hasattr(target, "state_dict") or not hasattr(target, "load_state_dict"):
            raise TypeError(
                f"paired replay state object {name!r} must implement state_dict/load_state_dict"
            )
        return cls(str(name), target, copy.deepcopy(target.state_dict()))

    def restore(self) -> None:
        self.target.load_state_dict(copy.deepcopy(self.state))


@dataclass
class PairedReplaySnapshot:
    rng: RNGSnapshot
    stateful: tuple[StatefulObjectSnapshot, ...]

    @classmethod
    def capture(
        cls, stateful_objects: Mapping[str, Any] | None = None
    ) -> "PairedReplaySnapshot":
        stateful_objects = {} if stateful_objects is None else dict(stateful_objects)
        snapshots = tuple(
            StatefulObjectSnapshot.capture(name, stateful_objects[name])
            for name in sorted(stateful_objects)
        )
        return cls(rng=RNGSnapshot.capture(), stateful=snapshots)

    def restore(self) -> None:
        self.rng.restore()
        for snapshot in self.stateful:
            snapshot.restore()


_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _hash_value(hasher: "hashlib._Hash", value: Any) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        hasher.update(b"tensor\0")
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(tensor.numpy().tobytes(order="C"))
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(b"ndarray\0")
        hasher.update(str(array.dtype).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(array.tobytes(order="C"))
    elif isinstance(value, Mapping):
        hasher.update(b"mapping\0")
        for key in sorted(value, key=lambda item: str(item)):
            _hash_value(hasher, str(key))
            _hash_value(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        hasher.update(b"sequence\0")
        for item in value:
            _hash_value(hasher, item)
    elif value is None or isinstance(value, (str, int, float, bool)):
        hasher.update(
            json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
    else:
        raise TypeError(f"unsupported materialized batch value: {type(value).__name__}")
    hasher.update(b"\0")


def materialized_batch_sha256(forward_kwargs: Mapping[str, Any]) -> str:
    hasher = hashlib.sha256(b"CT-P3R-3S-r2-materialized-batch-v1\0")
    _hash_value(hasher, forward_kwargs)
    return hasher.hexdigest()


def validate_candidate_order_invariance(
    canonical_regret: Mapping[str, float | Tensor],
    permuted_regret: Mapping[str, float | Tensor],
) -> None:
    """Fail closed if a paired replay result depends on candidate iteration order."""

    if not isinstance(canonical_regret, Mapping) or not isinstance(
        permuted_regret, Mapping
    ):
        raise TypeError("candidate-order regression inputs must be mappings")
    if not canonical_regret or set(canonical_regret) != set(permuted_regret):
        raise ValueError("candidate-order regression requires identical non-empty candidate sets")
    for name in sorted(canonical_regret):
        left = torch.as_tensor(canonical_regret[name]).detach().cpu()
        right = torch.as_tensor(permuted_regret[name]).detach().cpu()
        if left.numel() != 1 or right.numel() != 1:
            raise ValueError("candidate-order regret values must be scalar")
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise RuntimeError("candidate-order regression contains non-finite regret")
        if left.dtype != right.dtype or not torch.equal(left, right):
            raise RuntimeError(
                f"candidate-order-dependent regret detected for {name!r}"
            )


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
    dense_features: Tensor | None = None
    counterfactual_features: Tensor | None = None
    materialized_window_sha256: str | None = None
    counterfactual_window_sha256: str | None = None
    augmentation_sha256: str | None = None


@contextmanager
def _registered_action_context(
    detector: nn.Module,
    *,
    candidate_name: str,
    actions: Tensor,
) -> Iterator[None]:
    from .runtime import ChronoTransportRuntime

    runtimes = [module for module in detector.modules() if type(module) is ChronoTransportRuntime]
    if len(runtimes) != 1:
        raise ValueError("Gate 1 paired runner requires exactly one ChronoTransportRuntime")
    runtime = runtimes[0]
    previous = (
        runtime.forced_schedule,
        runtime.forced_action_name,
        runtime.forced_actions.detach().clone(),
    )
    runtime.set_registered_forced_actions(actions, candidate_name=candidate_name)
    try:
        yield
    finally:
        runtime.forced_schedule = previous[0]
        runtime.forced_action_name = previous[1]
        runtime.forced_actions = previous[2]


def _gate1_detector_loss(
    detector: nn.Module,
    forward_kwargs: Mapping[str, Any],
    *,
    candidate_name: str,
    actions: Tensor,
    snapshot: PairedReplaySnapshot,
    expected_batch_sha256: str,
    capture_motion: list[Tensor] | None = None,
) -> float:
    snapshot.restore()
    hook = None
    if capture_motion is not None:
        from .runtime import ChronoTransportRuntime

        runtimes = [
            module for module in detector.modules() if type(module) is ChronoTransportRuntime
        ]
        if len(runtimes) != 1:
            raise ValueError("Gate 1 motion capture requires one exact runtime")
        runtime = runtimes[0]

        def capture(module, args):
            if capture_motion:
                raise RuntimeError("Gate 1 motion capture hook ran more than once")
            x = args[0]
            batch_size = int(x.shape[0]) // int(module.chunks_per_window)
            state = x.reshape(
                batch_size,
                int(module.chunks_per_window),
                int(x.shape[1]),
                int(x.shape[2]),
            )
            signals = module._signals(state).detach()
            motion = signals[..., 3]
            if tuple(motion.shape) != (1, 48) or not torch.isfinite(motion).all():
                raise RuntimeError("Gate 1 deploy-visible motion is invalid")
            capture_motion.append(motion[0].cpu().clone())

        hook = runtime.register_forward_pre_hook(capture)
    try:
        with _registered_action_context(
            detector, candidate_name=candidate_name, actions=actions
        ), torch.no_grad():
            losses = detector(**dict(forward_kwargs))
            total = _loss_total(losses)
        if materialized_batch_sha256(forward_kwargs) != expected_batch_sha256:
            raise RuntimeError("Gate 1 paired runner mutated the materialized batch")
        value = float(total.detach().cpu())
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError("Gate 1 paired runner produced an invalid detector loss")
        return value
    finally:
        if hook is not None:
            hook.remove()
        snapshot.restore()


def run_registered_gate1_paired_replay(
    *,
    registration: Mapping[str, object],
    split: str,
    repository_root: str,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, object]:
    """Run and consume the sole repository-owned formal replay session."""

    from .registration import (
        validate_formal_gate1_context,
    )
    from .adjudication import validate_gate1_paired_replay_artifact
    from tools.bata.chronotransport_r2_gate1_replay_factory import (
        build_registered_gate1_replay_session,
    )

    registered = validate_formal_gate1_context(
        registration,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    artifact = build_registered_gate1_replay_session(registered).run_split(split)
    return validate_gate1_paired_replay_artifact(
        artifact,
        registration=registered,
        expected_split=split,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )


def _ephemeral_runtime_output(model: nn.Module) -> Tensor | None:
    outputs = [
        module.latest_output
        for module in model.modules()
        if module.__class__.__name__ == "ChronoTransportRuntime"
        and isinstance(getattr(module, "latest_output", None), Tensor)
    ]
    if not outputs:
        return None
    if len(outputs) != 1:
        raise ValueError("paired replay v1 requires exactly one ChronoTransportRuntime output")
    return outputs[0]


def paired_detector_losses(
    detector: nn.Module,
    forward_kwargs: Mapping[str, Any],
    *,
    counterfactual_schedule: str,
    track_counterfactual_grad: bool = True,
    stateful_objects: Mapping[str, Any] | None = None,
    augmentation_sha256: str | None = None,
) -> PairedReplayResult:
    if augmentation_sha256 is not None and not _LOWER_SHA256.fullmatch(
        str(augmentation_sha256)
    ):
        raise ValueError("augmentation_sha256 must be one lowercase SHA-256 digest")
    initial_hash = materialized_batch_sha256(forward_kwargs)
    initial = PairedReplaySnapshot.capture(stateful_objects)
    sdp_context = (
        nullcontext()
        if track_counterfactual_grad
        else torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        )
    )
    try:
        with sdp_context:
            with runtime_schedule(detector, "dense"), torch.no_grad():
                dense_losses = detector(**dict(forward_kwargs))
                dense_total = _loss_total(dense_losses)
                dense_features = _ephemeral_runtime_output(detector)
                if dense_features is not None:
                    dense_features = dense_features.detach()
                dense_hash = materialized_batch_sha256(forward_kwargs)
                if dense_hash != initial_hash:
                    raise RuntimeError("dense replay mutated the materialized batch")
            initial.restore()
            counterfactual_context = (
                nullcontext() if track_counterfactual_grad else torch.no_grad()
            )
            with runtime_schedule(detector, counterfactual_schedule), counterfactual_context:
                counterfactual_losses = detector(**dict(forward_kwargs))
                counterfactual_total = _loss_total(counterfactual_losses)
                counterfactual_features = _ephemeral_runtime_output(detector)
                counterfactual_hash = materialized_batch_sha256(forward_kwargs)
                if counterfactual_hash != initial_hash:
                    raise RuntimeError("counterfactual replay mutated the materialized batch")
    finally:
        initial.restore()
    regret = nonnegative_detector_regret(counterfactual_total, dense_total)
    return PairedReplayResult(
        dense_losses=dense_losses,
        counterfactual_losses=counterfactual_losses,
        dense_total=dense_total.detach(),
        counterfactual_total=counterfactual_total,
        regret=regret,
        dense_features=dense_features,
        counterfactual_features=counterfactual_features,
        materialized_window_sha256=dense_hash,
        counterfactual_window_sha256=counterfactual_hash,
        augmentation_sha256=None
        if augmentation_sha256 is None
        else str(augmentation_sha256),
    )


def run_fixed_r2_paired_detector_loss_vector(
    detector: nn.Module,
    forward_kwargs: Mapping[str, Any],
    *,
    stateful_objects: Mapping[str, Any] | None = None,
    augmentation_sha256: str | None = None,
) -> dict[str, object]:
    """Run the frozen 16 static schedules twice and prove order invariance."""

    initial_batch_sha256 = materialized_batch_sha256(forward_kwargs)

    def run(order: tuple[str, ...]) -> tuple[float, dict[str, float]]:
        dense_reference: float | None = None
        losses: dict[str, float] = {}
        for candidate_name in order:
            result = paired_detector_losses(
                detector,
                forward_kwargs,
                counterfactual_schedule=candidate_name,
                track_counterfactual_grad=False,
                stateful_objects=stateful_objects,
                augmentation_sha256=augmentation_sha256,
            )
            if result.materialized_window_sha256 != initial_batch_sha256:
                raise RuntimeError("fixed paired replay changed the materialized batch")
            dense_value = float(result.dense_total.detach().cpu())
            candidate_value = float(result.counterfactual_total.detach().cpu())
            if dense_reference is None:
                dense_reference = dense_value
            elif dense_value != dense_reference:
                raise RuntimeError("fixed paired replay dense reference changed by candidate order")
            losses[candidate_name] = candidate_value
        if dense_reference is None:
            raise RuntimeError("fixed paired replay candidate order cannot be empty")
        return dense_reference, losses

    canonical_order = tuple(R2_NON_DENSE_NAMES)
    reverse_order = tuple(reversed(canonical_order))
    dense_loss, canonical_losses = run(canonical_order)
    reverse_dense_loss, reverse_losses = run(reverse_order)
    if reverse_dense_loss != dense_loss or any(
        reverse_losses[name] != canonical_losses[name] for name in canonical_order
    ):
        raise RuntimeError("fixed paired replay candidate-order probe changed detector losses")
    candidate_loss_vector = [canonical_losses[name] for name in canonical_order]
    regret_vector = [max(value - dense_loss, 0.0) for value in candidate_loss_vector]
    return {
        "candidate_names": list(canonical_order),
        "candidate_order_sha256": canonical_sha256(canonical_order),
        "dense_detector_loss": dense_loss,
        "candidate_detector_loss": candidate_loss_vector,
        "detector_regret": regret_vector,
        "order_probe_candidate_names": list(reverse_order),
        "order_probe_candidate_detector_loss": [
            reverse_losses[name] for name in reverse_order
        ],
        "materialized_window_sha256": initial_batch_sha256,
        "augmentation_sha256": augmentation_sha256,
    }


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
