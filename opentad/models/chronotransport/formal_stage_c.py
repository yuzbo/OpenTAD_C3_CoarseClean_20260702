"""Formal paired ChronoTransport/ matched-dense Stage-C workflow state.

This module deliberately owns the long-running 4,200-success transaction.  The
single-step gradient algorithms live in :mod:`stage_c`; this layer binds those
algorithms to one materialized batch stream, paired arm invariants, projected
EMA state, checkpoint/resume, and a completion ledger.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import math
import pickle
import random
from typing import Any, Callable, Mapping, MutableSequence, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from opentad.cores.scheduler import LinearWarmupCosineAnnealingLR

from .protocol import R2_PROTOCOL_ID, canonical_sha256
from .scheduler import R2_NON_DENSE_NAMES
from .stage_c import (
    MatchedDenseParameterGroup,
    StageCInvalidImplementationError,
    StageCParameterGroups,
    StageCStateSurface,
    build_matched_dense_optimizer,
    build_matched_dense_parameter_group,
    build_stage_c_optimizer,
    build_stage_c_parameter_groups,
    hash_materialized_batch,
    run_matched_dense_amp_with_retry,
    run_stage_c_amp_with_retry,
    validate_matched_dense_optimizer,
    validate_stage_c_optimizer,
)


STAGE_C_TOTAL_SUCCESSFUL_UPDATES = 4200
STAGE_C_UPDATES_PER_EPOCH = 70
STAGE_C_EPOCHS = 60
STAGE_C_BATCH_SIZE = 2
STAGE_C_CHECKPOINT_FREQUENCY = 70
STAGE_C_EMA_DECAY = 0.999
STAGE_C_CHECKPOINT_SCHEMA = "chronotransport-r2-stage-c-paired-checkpoint-v1"
STAGE_C_TEST_CHECKPOINT_SCHEMA = (
    "chronotransport-r2-stage-c-paired-checkpoint-test-only-v1"
)
STAGE_C_COMPLETION_SCHEMA = "chronotransport-r2-stage-c-paired-completion-v1"


def _tensor_bytes(value: Tensor) -> bytes:
    if value.layout != torch.strided:
        raise ValueError("Stage-C checkpoint hashing requires strided tensors")
    return (
        value.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes(order="C")
    )


def tensor_mapping_sha256(values: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(b"chronotransport-r2-stage-c-tensor-map-v1\0")
    for name in sorted(values):
        value = values[name]
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError("Stage-C tensor maps require string keys and tensors")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(_tensor_bytes(value))
        digest.update(b"\0")
    return digest.hexdigest()


def _clone_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone(memory_format=torch.preserve_format)
    if isinstance(value, Mapping):
        return {key: _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    return copy.deepcopy(value)


def capture_stage_c_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else tuple()
        ),
    }


def restore_stage_c_rng_state(state: Mapping[str, Any]) -> None:
    if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Stage-C RNG state fields mismatch")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    cuda_states = tuple(state["torch_cuda"])
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("Stage-C CUDA RNG state does not match visible devices")
        torch.cuda.set_rng_state_all(list(cuda_states))


def _rng_sha256(state: Mapping[str, Any]) -> str:
    return hashlib.sha256(pickle.dumps(_clone_cpu(state), protocol=5)).hexdigest()


def _normalizer(model: nn.Module) -> tuple[str, Tensor]:
    matches = []
    for path, module in model.named_modules():
        value = module._buffers.get("loss_normalizer")
        if path.endswith("rpn_head") and isinstance(value, Tensor):
            matches.append((f"{path}.loss_normalizer", value))
    if len(matches) != 1 or matches[0][1].numel() != 1:
        raise StageCInvalidImplementationError(
            "paired Stage C requires one scalar rpn_head.loss_normalizer"
        )
    return matches[0]


def _state_by_names(model: nn.Module, names: Sequence[str]) -> dict[str, Tensor]:
    state = model.state_dict()
    if len(names) != len(set(names)) or any(name not in state for name in names):
        raise StageCInvalidImplementationError(
            "Stage-C canonical trainable names are missing or duplicated"
        )
    return {name: state[name].detach().clone() for name in sorted(names)}


def _load_state_by_names(
    model: nn.Module, values: Mapping[str, Tensor], expected_names: Sequence[str]
) -> None:
    if set(values) != set(expected_names):
        raise ValueError("Stage-C checkpoint trainable parameter names mismatch")
    state = model.state_dict()
    with torch.no_grad():
        for name in sorted(expected_names):
            source = values[name]
            target = state[name]
            if (
                not isinstance(source, Tensor)
                or source.dtype != target.dtype
                or tuple(source.shape) != tuple(target.shape)
                or not torch.isfinite(source).all()
            ):
                raise ValueError(f"Stage-C checkpoint tensor {name!r} is invalid")
            target.copy_(source.to(device=target.device))


class StageCProjectedEMA:
    """Repository-formula EMA over exactly the mutable Stage-C state."""

    def __init__(
        self,
        model: nn.Module,
        *,
        parameter_names: Sequence[str],
        buffer_names: Sequence[str],
        decay: float = STAGE_C_EMA_DECAY,
    ) -> None:
        if float(decay) != STAGE_C_EMA_DECAY:
            raise ValueError("formal Stage-C EMA decay must equal 0.999")
        names = tuple(sorted((*parameter_names, *buffer_names)))
        if not names or len(names) != len(set(names)):
            raise ValueError("Stage-C projected EMA names must be unique and non-empty")
        state = model.state_dict()
        if any(name not in state for name in names):
            raise ValueError("Stage-C projected EMA name is absent from model state")
        if any(state[name].dtype == torch.bool for name in names):
            raise ValueError("Stage-C projected EMA does not support boolean state")
        self.decay = float(decay)
        self.names = names
        self.shadow = {name: state[name].detach().clone() for name in names}
        self.stage_c_update_count = 0

    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        with torch.no_grad():
            for name in self.names:
                current = state[name]
                shadow = self.shadow[name]
                if current.dtype != shadow.dtype or tuple(current.shape) != tuple(shadow.shape):
                    raise StageCInvalidImplementationError(
                        "Stage-C EMA model state changed shape or dtype"
                    )
                # Match ``opentad.utils.ema.ModelEma`` exactly, including its
                # copy/cast semantics for the integer loss_normalizer buffer.
                shadow.copy_(
                    self.decay * shadow + (1.0 - self.decay) * current
                )
        self.stage_c_update_count += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "chronotransport-r2-stage-c-projected-ema-v1",
            "decay": self.decay,
            "names": self.names,
            "shadow": {name: value.detach().clone() for name, value in self.shadow.items()},
            "stage_c_update_count": self.stage_c_update_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {
            "schema",
            "decay",
            "names",
            "shadow",
            "stage_c_update_count",
        }:
            raise ValueError("Stage-C projected EMA state fields mismatch")
        if (
            state["schema"] != "chronotransport-r2-stage-c-projected-ema-v1"
            or float(state["decay"]) != self.decay
            or tuple(state["names"]) != self.names
            or set(state["shadow"]) != set(self.names)
        ):
            raise ValueError("Stage-C projected EMA identity mismatch")
        count = state["stage_c_update_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Stage-C projected EMA count is invalid")
        with torch.no_grad():
            for name in self.names:
                source = state["shadow"][name]
                target = self.shadow[name]
                if (
                    not isinstance(source, Tensor)
                    or source.dtype != target.dtype
                    or tuple(source.shape) != tuple(target.shape)
                    or not torch.isfinite(source).all()
                ):
                    raise ValueError("Stage-C projected EMA tensor is invalid")
                target.copy_(source.to(device=target.device))
        self.stage_c_update_count = count

    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        with torch.no_grad():
            for name in self.names:
                state[name].copy_(self.shadow[name].to(device=state[name].device))


def build_stage_c_lr_scheduler(
    optimizer: torch.optim.Optimizer,
) -> LinearWarmupCosineAnnealingLR:
    return LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epoch=350,
        max_epoch=7000,
        warmup_start_lr=0.0,
        eta_min=1e-8,
    )


def _new_control_objects(ema: StageCProjectedEMA, scheduler: Any) -> dict[str, Any]:
    return {
        "ema": ema,
        "scheduler": scheduler,
        "diagnostics": StageCStateSurface(),
        "profiler": StageCStateSurface(),
        "sampler": StageCStateSurface(),
        "successful_cursor": StageCStateSurface(),
        "exposure_cursor": StageCStateSurface(),
        "shadow_ledger": [],
    }


@dataclass
class PairedStageCState:
    ct_model: nn.Module
    matched_model: nn.Module
    ct_groups: StageCParameterGroups
    matched_group: MatchedDenseParameterGroup
    ct_optimizer: torch.optim.Optimizer
    matched_optimizer: torch.optim.Optimizer
    ct_scaler: Any
    matched_scaler: Any
    ct_scheduler: Any
    matched_scheduler: Any
    ct_objects: dict[str, Any]
    matched_objects: dict[str, Any]
    trace: list[dict[str, Any]] = field(default_factory=list)
    ct_retry_audit: list[dict[str, Any]] = field(default_factory=list)
    matched_retry_audit: list[dict[str, Any]] = field(default_factory=list)

    @property
    def successful_updates(self) -> int:
        return int(self.ct_objects["successful_cursor"].value)


def _assert_plain_single_process(model: nn.Module) -> None:
    if isinstance(model, nn.parallel.DistributedDataParallel):
        raise StageCInvalidImplementationError("formal Stage C forbids DDP")
    if model.__class__.__name__ in {
        "FullyShardedDataParallel",
        "DataParallel",
    } or callable(getattr(model, "no_sync", None)):
        raise StageCInvalidImplementationError(
            "formal Stage C requires a plain single-process module"
        )


def build_paired_stage_c_state(
    ct_model: nn.Module,
    matched_model: nn.Module,
    *,
    ct_scaler: Any | None = None,
    matched_scaler: Any | None = None,
) -> PairedStageCState:
    if ct_model is matched_model:
        raise ValueError("CT and matched-dense models must be distinct objects")
    _assert_plain_single_process(ct_model)
    _assert_plain_single_process(matched_model)
    ct_model.train()
    matched_model.train()
    ct_groups = build_stage_c_parameter_groups(ct_model)
    matched_group = build_matched_dense_parameter_group(matched_model)

    matched_names = tuple(sorted(matched_group.parameter_names))
    if not set(matched_names).issubset(ct_groups.parameter_names):
        raise StageCInvalidImplementationError(
            "matched common-A names are not a subset of CT A/T/R names"
        )
    ct_common = _state_by_names(ct_model, matched_names)
    matched_common = _state_by_names(matched_model, matched_names)
    if tensor_mapping_sha256(ct_common) != tensor_mapping_sha256(matched_common):
        raise StageCInvalidImplementationError(
            "CT and matched common-A initial values are not bitwise equal"
        )
    ct_normalizer_path, ct_normalizer = _normalizer(ct_model)
    matched_normalizer_path, matched_normalizer = _normalizer(matched_model)
    if ct_normalizer_path != matched_normalizer_path or not torch.equal(
        ct_normalizer.detach().cpu(), matched_normalizer.detach().cpu()
    ):
        raise StageCInvalidImplementationError(
            "CT and matched loss_normalizer initial values differ"
        )

    ct_optimizer = build_stage_c_optimizer(ct_groups)
    matched_optimizer = build_matched_dense_optimizer(matched_group)
    ct_scheduler = build_stage_c_lr_scheduler(ct_optimizer)
    matched_scheduler = build_stage_c_lr_scheduler(matched_optimizer)
    validate_stage_c_optimizer(ct_groups, ct_optimizer, lr_scheduler=ct_scheduler)
    validate_matched_dense_optimizer(
        matched_group, matched_optimizer, lr_scheduler=matched_scheduler
    )
    cuda = next(iter(ct_groups.all)).device.type == "cuda"
    if ct_scaler is None:
        ct_scaler = torch.cuda.amp.GradScaler(enabled=cuda)
    if matched_scaler is None:
        matched_scaler = torch.cuda.amp.GradScaler(enabled=cuda)
    ct_ema = StageCProjectedEMA(
        ct_model,
        parameter_names=ct_groups.parameter_names,
        buffer_names=(ct_normalizer_path,),
    )
    matched_ema = StageCProjectedEMA(
        matched_model,
        parameter_names=matched_group.parameter_names,
        buffer_names=(matched_normalizer_path,),
    )
    state = PairedStageCState(
        ct_model=ct_model,
        matched_model=matched_model,
        ct_groups=ct_groups,
        matched_group=matched_group,
        ct_optimizer=ct_optimizer,
        matched_optimizer=matched_optimizer,
        ct_scaler=ct_scaler,
        matched_scaler=matched_scaler,
        ct_scheduler=ct_scheduler,
        matched_scheduler=matched_scheduler,
        ct_objects=_new_control_objects(ct_ema, ct_scheduler),
        matched_objects=_new_control_objects(matched_ema, matched_scheduler),
    )
    validate_paired_stage_c_state(state)
    return state


def _scalar_tensor_sha256(value: Tensor) -> str:
    return tensor_mapping_sha256({"loss_normalizer": value.detach().cpu()})


def validate_paired_stage_c_state(state: PairedStageCState) -> None:
    ct_cursor = int(state.ct_objects["successful_cursor"].value)
    matched_cursor = int(state.matched_objects["successful_cursor"].value)
    if ct_cursor != matched_cursor or len(state.trace) != ct_cursor:
        raise StageCInvalidImplementationError(
            "paired Stage-C successful cursors/trace length diverged"
        )
    for objects, scheduler in (
        (state.ct_objects, state.ct_scheduler),
        (state.matched_objects, state.matched_scheduler),
    ):
        if (
            objects["scheduler"] is not scheduler
            or objects["sampler"].value != ct_cursor
            or objects["exposure_cursor"].value != 2 * ct_cursor
            or objects["ema"].stage_c_update_count != ct_cursor
            or len(objects["shadow_ledger"]) != ct_cursor
            or int(scheduler.last_epoch) != ct_cursor
            or int(scheduler._step_count) != ct_cursor + 1
        ):
            raise StageCInvalidImplementationError(
                "paired Stage-C arm state is not coherent with successful updates"
            )
    if state.ct_objects["shadow_ledger"] != state.matched_objects["shadow_ledger"]:
        raise StageCInvalidImplementationError(
            "CT and matched shadow candidate ledgers diverged"
        )
    _, ct_normalizer = _normalizer(state.ct_model)
    _, matched_normalizer = _normalizer(state.matched_model)
    if not torch.equal(ct_normalizer.detach().cpu(), matched_normalizer.detach().cpu()):
        raise StageCInvalidImplementationError(
            "CT and matched loss_normalizer traces diverged"
        )
    ct_lr = float(state.ct_optimizer.param_groups[0]["lr"])
    matched_lr = float(state.matched_optimizer.param_groups[0]["lr"])
    if ct_lr != matched_lr:
        raise StageCInvalidImplementationError("CT and matched common-A LR traces diverged")


def _batch_window_ids(batch: Mapping[str, Any]) -> tuple[str, str]:
    value = batch.get("window_id")
    if not isinstance(value, (list, tuple)) or len(value) != STAGE_C_BATCH_SIZE:
        raise StageCInvalidImplementationError(
            "Stage-C materialized batch requires two ordered window IDs"
        )
    ids = tuple(map(str, value))
    if any(not item for item in ids):
        raise StageCInvalidImplementationError("Stage-C window IDs must be non-empty")
    return ids  # type: ignore[return-value]


def _candidate_counts(trace: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in R2_NON_DENSE_NAMES}
    for row in trace:
        exposures = row.get("exposures")
        if not isinstance(exposures, list) or len(exposures) != 2:
            raise ValueError("Stage-C trace requires two exposures per success")
        for exposure in exposures:
            name = exposure.get("candidate_id") if isinstance(exposure, Mapping) else None
            if name not in counts:
                raise ValueError("Stage-C trace contains an unknown candidate")
            counts[name] += 1
    return counts


def _control_checkpoint(objects: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: _clone_cpu(objects[name].state_dict())
        for name in ("diagnostics", "profiler", "sampler", "successful_cursor", "exposure_cursor")
    } | {"shadow_ledger": copy.deepcopy(objects["shadow_ledger"])}


def _arm_checkpoint(
    *,
    model: nn.Module,
    names: Sequence[str],
    optimizer: Any,
    scaler: Any,
    scheduler: Any,
    objects: Mapping[str, Any],
) -> dict[str, Any]:
    trainable = _state_by_names(model, names)
    normalizer_path, normalizer = _normalizer(model)
    trainable_cpu = _clone_cpu(trainable)
    return {
        "parameter_names": tuple(sorted(names)),
        "trainable_state": trainable_cpu,
        "trainable_state_sha256": tensor_mapping_sha256(trainable_cpu),
        "normalizer_path": normalizer_path,
        "normalizer": normalizer.detach().cpu().clone(),
        "normalizer_sha256": _scalar_tensor_sha256(normalizer),
        "optimizer": _clone_cpu(optimizer.state_dict()),
        "scaler": _clone_cpu(scaler.state_dict()),
        "scheduler": _clone_cpu(scheduler.state_dict()),
        "ema": _clone_cpu(objects["ema"].state_dict()),
        "controls": _control_checkpoint(objects),
    }


def build_paired_stage_c_checkpoint(
    state: PairedStageCState,
    *,
    seed: int,
    fit_window_ids: Sequence[str],
    provenance: Mapping[str, Any],
    total_successful_updates: int,
    formal: bool,
) -> dict[str, Any]:
    validate_paired_stage_c_state(state)
    cursor = state.successful_updates
    if cursor > total_successful_updates:
        raise ValueError("Stage-C cursor exceeds the target")
    if formal and (
        total_successful_updates != STAGE_C_TOTAL_SUCCESSFUL_UPDATES
        or len(fit_window_ids) != 140
        or cursor % STAGE_C_UPDATES_PER_EPOCH != 0
    ):
        raise ValueError("formal Stage-C checkpoint cadence/target is invalid")
    counts = _candidate_counts(state.trace)
    checkpoint = {
        "schema": STAGE_C_CHECKPOINT_SCHEMA if formal else STAGE_C_TEST_CHECKPOINT_SCHEMA,
        "protocol": R2_PROTOCOL_ID,
        "status": "COMPLETE" if cursor == total_successful_updates else "PERIODIC",
        "seed": int(seed),
        "successful_updates": cursor,
        "total_successful_updates": int(total_successful_updates),
        "fit_window_ids": tuple(map(str, fit_window_ids)),
        "fit_window_ids_sha256": canonical_sha256(list(map(str, fit_window_ids))),
        "provenance": copy.deepcopy(dict(provenance)),
        "provenance_sha256": canonical_sha256(provenance),
        "ct": _arm_checkpoint(
            model=state.ct_model,
            names=state.ct_groups.parameter_names,
            optimizer=state.ct_optimizer,
            scaler=state.ct_scaler,
            scheduler=state.ct_scheduler,
            objects=state.ct_objects,
        ),
        "matched_dense": _arm_checkpoint(
            model=state.matched_model,
            names=state.matched_group.parameter_names,
            optimizer=state.matched_optimizer,
            scaler=state.matched_scaler,
            scheduler=state.matched_scheduler,
            objects=state.matched_objects,
        ),
        "paired_trace": copy.deepcopy(state.trace),
        "paired_trace_sha256": canonical_sha256(state.trace),
        "ct_retry_audit": copy.deepcopy(state.ct_retry_audit),
        "matched_retry_audit": copy.deepcopy(state.matched_retry_audit),
        "candidate_counts": counts,
        "rng_state": _clone_cpu(capture_stage_c_rng_state()),
    }
    validate_paired_stage_c_checkpoint(
        checkpoint,
        expected_seed=seed,
        expected_fit_window_ids=fit_window_ids,
        expected_provenance=provenance,
        formal=formal,
        require_complete=cursor == total_successful_updates,
        expected_total_successful_updates=total_successful_updates,
    )
    return checkpoint


_CHECKPOINT_KEYS = {
    "schema",
    "protocol",
    "status",
    "seed",
    "successful_updates",
    "total_successful_updates",
    "fit_window_ids",
    "fit_window_ids_sha256",
    "provenance",
    "provenance_sha256",
    "ct",
    "matched_dense",
    "paired_trace",
    "paired_trace_sha256",
    "ct_retry_audit",
    "matched_retry_audit",
    "candidate_counts",
    "rng_state",
}
_ARM_KEYS = {
    "parameter_names",
    "trainable_state",
    "trainable_state_sha256",
    "normalizer_path",
    "normalizer",
    "normalizer_sha256",
    "optimizer",
    "scaler",
    "scheduler",
    "ema",
    "controls",
}


def validate_paired_stage_c_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_fit_window_ids: Sequence[str],
    expected_provenance: Mapping[str, Any],
    formal: bool,
    require_complete: bool,
    expected_total_successful_updates: int,
) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != _CHECKPOINT_KEYS:
        raise ValueError("Stage-C paired checkpoint fields mismatch")
    expected_schema = STAGE_C_CHECKPOINT_SCHEMA if formal else STAGE_C_TEST_CHECKPOINT_SCHEMA
    if checkpoint["schema"] != expected_schema or checkpoint["protocol"] != R2_PROTOCOL_ID:
        raise ValueError("Stage-C paired checkpoint schema/protocol mismatch")
    cursor = checkpoint["successful_updates"]
    target = checkpoint["total_successful_updates"]
    if (
        isinstance(cursor, bool)
        or not isinstance(cursor, int)
        or isinstance(target, bool)
        or not isinstance(target, int)
        or not 0 <= cursor <= target
        or target != expected_total_successful_updates
        or checkpoint["seed"] != expected_seed
    ):
        raise ValueError("Stage-C paired checkpoint cursor/seed mismatch")
    if formal and (
        target != STAGE_C_TOTAL_SUCCESSFUL_UPDATES
        or cursor % STAGE_C_UPDATES_PER_EPOCH != 0
    ):
        raise ValueError("formal Stage-C checkpoint cursor is off cadence")
    status = "COMPLETE" if cursor == target else "PERIODIC"
    if checkpoint["status"] != status or (require_complete and status != "COMPLETE"):
        raise ValueError("Stage-C paired checkpoint completion status mismatch")
    fit_ids = tuple(map(str, expected_fit_window_ids))
    if (
        tuple(checkpoint["fit_window_ids"]) != fit_ids
        or checkpoint["fit_window_ids_sha256"] != canonical_sha256(list(fit_ids))
        or (formal and len(fit_ids) != 140)
    ):
        raise ValueError("Stage-C paired checkpoint fit-window identity mismatch")
    if (
        dict(checkpoint["provenance"]) != dict(expected_provenance)
        or checkpoint["provenance_sha256"] != canonical_sha256(expected_provenance)
    ):
        raise ValueError("Stage-C paired checkpoint provenance mismatch")
    trace = checkpoint["paired_trace"]
    if (
        not isinstance(trace, list)
        or len(trace) != cursor
        or checkpoint["paired_trace_sha256"] != canonical_sha256(trace)
    ):
        raise ValueError("Stage-C paired checkpoint trace length mismatch")
    if checkpoint["candidate_counts"] != _candidate_counts(trace):
        raise ValueError("Stage-C paired checkpoint candidate counts mismatch")
    if formal and status == "COMPLETE" and (
        checkpoint["candidate_counts"] != {name: 525 for name in R2_NON_DENSE_NAMES}
        or cursor != STAGE_C_TOTAL_SUCCESSFUL_UPDATES
    ):
        raise ValueError("formal Stage-C completion exposure balance mismatch")
    for index, row in enumerate(trace):
        pair_index = index % (len(fit_ids) // STAGE_C_BATCH_SIZE)
        expected_pair = list(
            fit_ids[
                STAGE_C_BATCH_SIZE * pair_index : STAGE_C_BATCH_SIZE * pair_index
                + STAGE_C_BATCH_SIZE
            ]
        )
        if (
            not isinstance(row, Mapping)
            or row.get("successful_update") != index
            or row.get("epoch") != index // STAGE_C_UPDATES_PER_EPOCH
            or row.get("batch_in_epoch") != index % STAGE_C_UPDATES_PER_EPOCH
            or row.get("window_ids") != expected_pair
            or row.get("ct_batch_hash") != row.get("matched_batch_hash")
            or row.get("ct_action_batch_sha256")
            != row.get("matched_action_batch_sha256")
            or row.get("ct_exposures") != row.get("matched_exposures")
            or row.get("exposures") != row.get("ct_exposures")
            or row.get("ct_common_a_lr") != row.get("matched_common_a_lr")
            or row.get("ct_ema_updates") != index + 1
            or row.get("matched_ema_updates") != index + 1
        ):
            raise ValueError(f"Stage-C paired trace mismatch at update {index}")
        for field in (
            "ct_batch_hash",
            "ct_action_batch_sha256",
            "ct_loss_normalizer_sha256",
            "matched_loss_normalizer_sha256",
            "pre_arm_rng_sha256",
            "ct_post_rng_sha256",
            "matched_post_rng_sha256",
        ):
            value = row.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"Stage-C paired trace field {field} is invalid at update {index}"
                )
    for arm_name in ("ct", "matched_dense"):
        arm = checkpoint[arm_name]
        if not isinstance(arm, Mapping) or set(arm) != _ARM_KEYS:
            raise ValueError(f"Stage-C {arm_name} checkpoint fields mismatch")
        names = tuple(arm["parameter_names"])
        tensors = arm["trainable_state"]
        normalizer = arm["normalizer"]
        if (
            not names
            or len(names) != len(set(names))
            or not isinstance(tensors, Mapping)
            or set(tensors) != set(names)
            or tensor_mapping_sha256(tensors) != arm["trainable_state_sha256"]
            or not isinstance(normalizer, Tensor)
            or normalizer.numel() != 1
            or not torch.isfinite(normalizer).all()
            or _scalar_tensor_sha256(normalizer) != arm["normalizer_sha256"]
        ):
            raise ValueError(f"Stage-C {arm_name} tensor state is invalid")
        controls = arm["controls"]
        if not isinstance(controls, Mapping) or set(controls) != {
            "diagnostics",
            "profiler",
            "sampler",
            "successful_cursor",
            "exposure_cursor",
            "shadow_ledger",
        }:
            raise ValueError(f"Stage-C {arm_name} control state fields mismatch")
        if (
            controls["successful_cursor"] != {"value": cursor}
            or controls["sampler"] != {"value": cursor}
            or controls["exposure_cursor"] != {"value": 2 * cursor}
            or len(controls["shadow_ledger"]) != cursor
            or arm["ema"].get("stage_c_update_count") != cursor
        ):
            raise ValueError(f"Stage-C {arm_name} control cursor mismatch")
    if checkpoint["ct"]["controls"]["shadow_ledger"] != checkpoint[
        "matched_dense"
    ]["controls"]["shadow_ledger"]:
        raise ValueError("Stage-C checkpoint shadow ledgers diverged")
    if not torch.equal(
        checkpoint["ct"]["normalizer"], checkpoint["matched_dense"]["normalizer"]
    ):
        raise ValueError("Stage-C checkpoint normalizer traces diverged")
    if not isinstance(checkpoint["ct_retry_audit"], list) or not isinstance(
        checkpoint["matched_retry_audit"], list
    ):
        raise ValueError("Stage-C retry audits must be lists")
    if set(checkpoint["rng_state"]) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Stage-C checkpoint RNG fields mismatch")
    return dict(checkpoint)


def _restore_controls(objects: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    for name in ("diagnostics", "profiler", "sampler", "successful_cursor", "exposure_cursor"):
        objects[name].load_state_dict(copy.deepcopy(state[name]))
    ledger = objects["shadow_ledger"]
    if not isinstance(ledger, MutableSequence):
        raise ValueError("Stage-C shadow ledger is not mutable")
    ledger[:] = copy.deepcopy(state["shadow_ledger"])


def _restore_arm(
    *,
    arm: Mapping[str, Any],
    model: nn.Module,
    expected_names: Sequence[str],
    optimizer: Any,
    scaler: Any,
    scheduler: Any,
    objects: Mapping[str, Any],
) -> None:
    _load_state_by_names(model, arm["trainable_state"], expected_names)
    normalizer_path, normalizer = _normalizer(model)
    if arm["normalizer_path"] != normalizer_path:
        raise ValueError("Stage-C normalizer path mismatch on resume")
    with torch.no_grad():
        normalizer.copy_(arm["normalizer"].to(device=normalizer.device))
    optimizer.load_state_dict(copy.deepcopy(arm["optimizer"]))
    scaler.load_state_dict(copy.deepcopy(arm["scaler"]))
    scheduler.load_state_dict(copy.deepcopy(arm["scheduler"]))
    objects["ema"].load_state_dict(copy.deepcopy(arm["ema"]))
    _restore_controls(objects, arm["controls"])


def load_paired_stage_c_checkpoint(
    state: PairedStageCState,
    checkpoint: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_fit_window_ids: Sequence[str],
    expected_provenance: Mapping[str, Any],
    formal: bool,
    expected_total_successful_updates: int,
) -> None:
    validated = validate_paired_stage_c_checkpoint(
        checkpoint,
        expected_seed=expected_seed,
        expected_fit_window_ids=expected_fit_window_ids,
        expected_provenance=expected_provenance,
        formal=formal,
        require_complete=False,
        expected_total_successful_updates=expected_total_successful_updates,
    )
    _restore_arm(
        arm=validated["ct"],
        model=state.ct_model,
        expected_names=state.ct_groups.parameter_names,
        optimizer=state.ct_optimizer,
        scaler=state.ct_scaler,
        scheduler=state.ct_scheduler,
        objects=state.ct_objects,
    )
    _restore_arm(
        arm=validated["matched_dense"],
        model=state.matched_model,
        expected_names=state.matched_group.parameter_names,
        optimizer=state.matched_optimizer,
        scaler=state.matched_scaler,
        scheduler=state.matched_scheduler,
        objects=state.matched_objects,
    )
    state.trace[:] = copy.deepcopy(validated["paired_trace"])
    state.ct_retry_audit[:] = copy.deepcopy(validated["ct_retry_audit"])
    state.matched_retry_audit[:] = copy.deepcopy(validated["matched_retry_audit"])
    restore_stage_c_rng_state(validated["rng_state"])
    validate_paired_stage_c_state(state)


def _trace_row(
    *,
    update: int,
    window_ids: Sequence[str],
    ct_result: Mapping[str, Any],
    matched_result: Mapping[str, Any],
    state: PairedStageCState,
    pre_arm_rng_sha256: str,
    ct_post_rng_sha256: str,
    matched_post_rng_sha256: str,
) -> dict[str, Any]:
    _, ct_normalizer = _normalizer(state.ct_model)
    _, matched_normalizer = _normalizer(state.matched_model)
    ct_lr = float(state.ct_optimizer.param_groups[0]["lr"])
    matched_lr = float(state.matched_optimizer.param_groups[0]["lr"])
    if not math.isfinite(ct_lr) or ct_lr != matched_lr:
        raise StageCInvalidImplementationError("paired common-A LR is invalid")
    return {
        "successful_update": update,
        "epoch": update // STAGE_C_UPDATES_PER_EPOCH,
        "batch_in_epoch": update % STAGE_C_UPDATES_PER_EPOCH,
        "window_ids": list(map(str, window_ids)),
        "ct_batch_hash": ct_result["batch_hash"],
        "matched_batch_hash": matched_result["batch_hash"],
        "ct_action_batch_sha256": ct_result["action_batch_sha256"],
        "matched_action_batch_sha256": matched_result["action_batch_sha256"],
        "ct_exposures": copy.deepcopy(ct_result["exposures"]),
        "matched_exposures": copy.deepcopy(matched_result["exposures"]),
        "exposures": copy.deepcopy(ct_result["exposures"]),
        "ct_attempts": int(ct_result["attempts"]),
        "matched_attempts": int(matched_result["attempts"]),
        "ct_retries": int(ct_result["retries"]),
        "matched_retries": int(matched_result["retries"]),
        "ct_common_a_lr": ct_lr,
        "matched_common_a_lr": matched_lr,
        "ct_ema_updates": int(state.ct_objects["ema"].stage_c_update_count),
        "matched_ema_updates": int(
            state.matched_objects["ema"].stage_c_update_count
        ),
        "ct_loss_normalizer_sha256": _scalar_tensor_sha256(ct_normalizer),
        "matched_loss_normalizer_sha256": _scalar_tensor_sha256(matched_normalizer),
        "ct_transport_executed": bool(
            ct_result["a3_a4_audit"]["transport_executed"]
        ),
        "ct_transport_grad_norm": float(
            ct_result["a3_a4_audit"]["transport_grad_norm"]
        ),
        "pre_arm_rng_sha256": pre_arm_rng_sha256,
        "ct_post_rng_sha256": ct_post_rng_sha256,
        "matched_post_rng_sha256": matched_post_rng_sha256,
    }


def run_paired_stage_c_training(
    state: PairedStageCState,
    *,
    materialize_batch: Callable[[int], Mapping[str, Any]],
    fit_window_ids: Sequence[str],
    seed: int,
    provenance: Mapping[str, Any],
    formal: bool,
    total_successful_updates: int = STAGE_C_TOTAL_SUCCESSFUL_UPDATES,
    checkpoint_frequency: int = STAGE_C_CHECKPOINT_FREQUENCY,
    checkpoint_sink: Callable[[int, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if seed not in (3407, 3408, 3409) or isinstance(seed, bool):
        raise ValueError("Stage-C seed must be 3407, 3408, or 3409")
    fit_ids = tuple(map(str, fit_window_ids))
    if len(fit_ids) % STAGE_C_BATCH_SIZE != 0 or len(set(fit_ids)) != len(fit_ids):
        raise ValueError("Stage-C fit window order must be unique and batchable by two")
    if formal and (
        len(fit_ids) != 140
        or total_successful_updates != STAGE_C_TOTAL_SUCCESSFUL_UPDATES
        or checkpoint_frequency != STAGE_C_CHECKPOINT_FREQUENCY
    ):
        raise ValueError("formal Stage-C population/target/checkpoint cadence is frozen")
    if (
        isinstance(total_successful_updates, bool)
        or not isinstance(total_successful_updates, int)
        or total_successful_updates <= 0
        or isinstance(checkpoint_frequency, bool)
        or not isinstance(checkpoint_frequency, int)
        or checkpoint_frequency <= 0
    ):
        raise ValueError("Stage-C target and checkpoint frequency must be positive integers")
    validate_paired_stage_c_state(state)
    if state.successful_updates > total_successful_updates:
        raise ValueError("Stage-C resume cursor exceeds target")

    while state.successful_updates < total_successful_updates:
        update = state.successful_updates
        batch = materialize_batch(update)
        if not isinstance(batch, Mapping):
            raise StageCInvalidImplementationError(
                "Stage-C materializer must return a mapping"
            )
        window_ids = _batch_window_ids(batch)
        pair_index = update % (len(fit_ids) // STAGE_C_BATCH_SIZE)
        expected_window_ids = fit_ids[
            STAGE_C_BATCH_SIZE * pair_index : STAGE_C_BATCH_SIZE * pair_index + 2
        ]
        if window_ids != expected_window_ids:
            raise StageCInvalidImplementationError(
                "Stage-C materialized batch differs from canonical fit-window order"
            )
        materialized_hash = hash_materialized_batch(batch)
        pre_arm_rng = capture_stage_c_rng_state()
        pre_arm_rng_sha256 = _rng_sha256(pre_arm_rng)

        ct_result = run_stage_c_amp_with_retry(
            materialized_batch=batch,
            model=state.ct_model,
            groups=state.ct_groups,
            optimizer=state.ct_optimizer,
            scaler=state.ct_scaler,
            lr_scheduler=state.ct_scheduler,
            seed=seed,
            rollback_objects=state.ct_objects,
            retry_audit=state.ct_retry_audit,
        )
        ct_post_rng = capture_stage_c_rng_state()
        restore_stage_c_rng_state(pre_arm_rng)
        matched_result = run_matched_dense_amp_with_retry(
            materialized_batch=batch,
            model=state.matched_model,
            group=state.matched_group,
            optimizer=state.matched_optimizer,
            scaler=state.matched_scaler,
            lr_scheduler=state.matched_scheduler,
            seed=seed,
            rollback_objects=state.matched_objects,
            retry_audit=state.matched_retry_audit,
        )
        matched_post_rng = capture_stage_c_rng_state()
        if hash_materialized_batch(batch) != materialized_hash:
            raise StageCInvalidImplementationError(
                "Stage-C paired arms mutated the materialized batch"
            )
        if (
            ct_result["batch_hash"] != matched_result["batch_hash"]
            or ct_result["batch_hash"] != materialized_hash
            or ct_result["action_batch_sha256"]
            != matched_result["action_batch_sha256"]
            or ct_result["exposures"] != matched_result["exposures"]
        ):
            raise StageCInvalidImplementationError(
                "CT and matched batch/action/exposure traces diverged"
            )
        row = _trace_row(
            update=update,
            window_ids=window_ids,
            ct_result=ct_result,
            matched_result=matched_result,
            state=state,
            pre_arm_rng_sha256=pre_arm_rng_sha256,
            ct_post_rng_sha256=_rng_sha256(ct_post_rng),
            matched_post_rng_sha256=_rng_sha256(matched_post_rng),
        )
        if (
            row["ct_loss_normalizer_sha256"]
            != row["matched_loss_normalizer_sha256"]
        ):
            raise StageCInvalidImplementationError(
                "CT and matched loss_normalizer trace hashes diverged"
            )
        state.trace.append(row)
        validate_paired_stage_c_state(state)

        if state.successful_updates % checkpoint_frequency == 0:
            checkpoint = build_paired_stage_c_checkpoint(
                state,
                seed=seed,
                fit_window_ids=fit_ids,
                provenance=provenance,
                total_successful_updates=total_successful_updates,
                formal=formal,
            )
            if checkpoint_sink is not None:
                checkpoint_sink(state.successful_updates, checkpoint)

    complete = build_paired_stage_c_checkpoint(
        state,
        seed=seed,
        fit_window_ids=fit_ids,
        provenance=provenance,
        total_successful_updates=total_successful_updates,
        formal=formal,
    )
    transport_rows = [row for row in state.trace if row["ct_transport_executed"]]
    aggregate_transport_grad = sum(
        float(row["ct_transport_grad_norm"]) for row in transport_rows
    )
    if formal and (not transport_rows or aggregate_transport_grad <= 0.0):
        raise StageCInvalidImplementationError(
            "formal Stage-C TRANSPORT gradient aggregate is not positive"
        )
    return {
        "status": "COMPLETE",
        "successful_updates": state.successful_updates,
        "window_exposures": 2 * state.successful_updates,
        "candidate_counts": dict(complete["candidate_counts"]),
        "ct_attempted_updates": len(state.ct_retry_audit),
        "matched_attempted_updates": len(state.matched_retry_audit),
        "ct_retries": len(state.ct_retry_audit) - state.successful_updates,
        "matched_retries": len(state.matched_retry_audit)
        - state.successful_updates,
        "transport_successful_batches": len(transport_rows),
        "aggregate_transport_grad_norm": aggregate_transport_grad,
        "checkpoint": complete,
    }


def build_stage_c_completion_marker(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: str,
    checkpoint_file_sha256: str,
    ledger_path: str,
    ledger_file_sha256: str,
) -> dict[str, Any]:
    if checkpoint.get("schema") != STAGE_C_CHECKPOINT_SCHEMA:
        raise ValueError("formal Stage-C completion requires a formal checkpoint")
    if checkpoint.get("status") != "COMPLETE":
        raise ValueError("formal Stage-C completion requires a complete checkpoint")
    for value, label in (
        (checkpoint_file_sha256, "checkpoint"),
        (ledger_file_sha256, "ledger"),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"Stage-C {label} file SHA-256 is invalid")
    marker = {
        "schema": STAGE_C_COMPLETION_SCHEMA,
        "protocol": R2_PROTOCOL_ID,
        "status": "SUCCESS",
        "registration_sha256": checkpoint["provenance"].get("registration_sha256"),
        "registration_commit": checkpoint["provenance"].get("registration_commit"),
        "seed": checkpoint["seed"],
        "successful_updates_per_arm": checkpoint["successful_updates"],
        "window_exposures_per_arm": 2 * checkpoint["successful_updates"],
        "candidate_counts_per_arm": dict(checkpoint["candidate_counts"]),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_file_sha256,
            "logical_provenance_sha256": checkpoint["provenance_sha256"],
        },
        "ledger": {"path": str(ledger_path), "sha256": ledger_file_sha256},
        "claim_flags": {
            "stage_c_complete": True,
            "post_stage_c_gate3_pass": False,
            "metric_adatad_thumos14_official_full_video": False,
            "latency_slurm_single_gpu_fixed_stack": False,
            "deploy": False,
            "paper": False,
        },
    }
    marker["artifact_sha256"] = canonical_sha256(marker)
    return marker


__all__ = [
    "PairedStageCState",
    "STAGE_C_CHECKPOINT_FREQUENCY",
    "STAGE_C_CHECKPOINT_SCHEMA",
    "STAGE_C_COMPLETION_SCHEMA",
    "STAGE_C_EMA_DECAY",
    "STAGE_C_TOTAL_SUCCESSFUL_UPDATES",
    "StageCProjectedEMA",
    "build_paired_stage_c_checkpoint",
    "build_paired_stage_c_state",
    "build_stage_c_completion_marker",
    "build_stage_c_lr_scheduler",
    "capture_stage_c_rng_state",
    "load_paired_stage_c_checkpoint",
    "restore_stage_c_rng_state",
    "run_paired_stage_c_training",
    "tensor_mapping_sha256",
    "validate_paired_stage_c_checkpoint",
    "validate_paired_stage_c_state",
]
