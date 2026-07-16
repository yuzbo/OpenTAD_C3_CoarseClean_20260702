"""Stage-C object ownership and loss-specific AMP update primitives."""

from __future__ import annotations

import copy
from contextlib import nullcontext
import hashlib
import math
import random
import struct
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, MutableSequence, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .actions import ChronoAction, LayerGroup
from .protocol import canonical_sha256, stage_c_batch_exposures
from .runtime import ChronoTransportRuntime
from .scheduler import R2_NON_DENSE_NAMES, ScheduleLibrary


_FORMAL_CHRONOTRANSPORT_RUNTIME = ChronoTransportRuntime
_FORMAL_CHRONOTRANSPORT_FORWARD = ChronoTransportRuntime.forward


class StageCInvalidImplementationError(RuntimeError):
    """The frozen r2 Stage-C execution contract was violated."""


class StageCStateSurface:
    """Minimal state-dict surface for formal Stage-C cursors/diagnostics."""

    def __init__(self, value: int = 0) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Stage-C state value must be a non-negative integer")
        self.value = value

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"value"}:
            raise ValueError("Stage-C state surface requires only value")
        value = state["value"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Stage-C state value must be a non-negative integer")
        self.value = value


class StageCTrackedEMA:
    """Add exact successful-update accounting to the repository EMA object."""

    def __init__(self, ema: Any) -> None:
        if not callable(getattr(ema, "update", None)):
            raise TypeError("Stage-C EMA backend requires update(model)")
        if not callable(getattr(ema, "state_dict", None)) or not callable(
            getattr(ema, "load_state_dict", None)
        ):
            raise TypeError("Stage-C EMA backend requires state_dict/load_state_dict")
        self.ema = ema
        self.stage_c_update_count = 0

    def update(self, model: nn.Module) -> None:
        self.ema.update(model)
        self.stage_c_update_count += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "ema": copy.deepcopy(self.ema.state_dict()),
            "stage_c_update_count": self.stage_c_update_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"ema", "stage_c_update_count"}:
            raise ValueError("Stage-C tracked EMA state fields mismatch")
        count = state["stage_c_update_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Stage-C EMA update count must be a non-negative integer")
        self.ema.load_state_dict(copy.deepcopy(state["ema"]))
        self.stage_c_update_count = count


@dataclass(frozen=True)
class StageCAttemptLosses:
    detector_loss: Tensor
    feature_loss: Tensor
    risk_loss: Tensor


_R2_STAGE_C_LIBRARY = ScheduleLibrary.r2(
    layer_groups=(LayerGroup(0, 4), LayerGroup(4, 8), LayerGroup(8, 12))
)
_R2_STAGE_C_ACTIONS = tuple(
    _R2_STAGE_C_LIBRARY.find(name).actions.detach().cpu().to(torch.int8).clone()
    for name in R2_NON_DENSE_NAMES
)


def _unique_parameters(modules: Iterable[nn.Module]) -> tuple[nn.Parameter, ...]:
    by_identity: dict[int, nn.Parameter] = {}
    for module in modules:
        for parameter in module.parameters():
            by_identity[id(parameter)] = parameter
    return tuple(by_identity[key] for key in sorted(by_identity))


@dataclass(frozen=True)
class StageCParameterGroups:
    adapters: tuple[nn.Parameter, ...]
    transport: tuple[nn.Parameter, ...]
    risk: tuple[nn.Parameter, ...]
    adapter_paths: tuple[str, ...]
    transport_path: str
    risk_path: str
    scheduler_predictor_paths: tuple[str, ...]
    parameter_names: tuple[str, ...]

    @property
    def all(self) -> tuple[nn.Parameter, ...]:
        return tuple(sorted((*self.adapters, *self.transport, *self.risk), key=id))


@dataclass(frozen=True)
class MatchedDenseParameterGroup:
    """The exact common-A ownership surface for the matched-dense arm."""

    adapters: tuple[nn.Parameter, ...]
    adapter_paths: tuple[str, ...]
    parameter_names: tuple[str, ...]

    @property
    def all(self) -> tuple[nn.Parameter, ...]:
        return tuple(sorted(self.adapters, key=id))


def _named_modules_with_aliases(model: nn.Module) -> list[tuple[str, nn.Module]]:
    try:
        return list(model.named_modules(remove_duplicate=False))
    except TypeError as error:  # pragma: no cover - frozen runtime requires this API
        raise StageCInvalidImplementationError(
            "Stage C requires named_modules(remove_duplicate=False) alias auditing"
        ) from error


def _named_parameters_with_aliases(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    try:
        return list(model.named_parameters(remove_duplicate=False))
    except TypeError as error:  # pragma: no cover - frozen runtime requires this API
        raise StageCInvalidImplementationError(
            "Stage C requires named_parameters(remove_duplicate=False) alias auditing"
        ) from error


def _named_buffers_with_aliases(model: nn.Module) -> list[tuple[str, Tensor]]:
    try:
        return list(model.named_buffers(remove_duplicate=False))
    except TypeError as error:  # pragma: no cover - frozen runtime requires this API
        raise StageCInvalidImplementationError(
            "Stage C requires named_buffers(remove_duplicate=False) alias auditing"
        ) from error


def build_stage_c_parameter_groups(model: nn.Module) -> StageCParameterGroups:
    """Derive the frozen A/T/R registry from the AdaTAD/CT model topology."""

    modules = _named_modules_with_aliases(model)
    module_by_path = dict(modules)
    runtimes = [
        (path, module)
        for path, module in modules
        if path.endswith("chronotransport")
        and isinstance(getattr(module, "transport", None), nn.Module)
        and isinstance(getattr(module, "risk_predictor", None), nn.Module)
        and isinstance(getattr(module, "scheduler", None), nn.Module)
    ]
    if len(runtimes) != 1:
        raise ValueError("Stage C requires exactly one canonical ChronoTransport runtime")
    runtime_path, runtime = runtimes[0]
    backbone_path = runtime_path.rsplit(".", 1)[0] if "." in runtime_path else ""
    backbone = module_by_path.get(backbone_path, model if not backbone_path else None)
    blocks = getattr(backbone, "blocks", None)
    if backbone is None or not isinstance(blocks, (nn.ModuleList, list, tuple)) or not blocks:
        raise ValueError("Stage C requires the explicit AdaTAD backbone.blocks adapter registry")

    adapter_modules: list[nn.Module] = []
    adapter_paths: list[str] = []
    for index, block in enumerate(blocks):
        enabled = bool(getattr(block, "use_adapter", False))
        adapter = getattr(block, "adapter", None)
        path = f"{backbone_path + '.' if backbone_path else ''}blocks.{index}.adapter"
        if enabled:
            if not isinstance(adapter, nn.Module):
                raise ValueError(f"enabled AdaTAD adapter block {index} has no adapter module")
            if module_by_path.get(path) is not adapter:
                raise ValueError(f"AdaTAD adapter registry path mismatch at block {index}")
            adapter_modules.append(adapter)
            adapter_paths.append(path)
        elif isinstance(adapter, nn.Module):
            raise ValueError(f"disabled adapter block {index} unexpectedly owns an adapter module")
    if not adapter_modules:
        raise ValueError("Stage C requires at least one enabled AdaTAD adapter module")

    transport_module = runtime.transport
    risk_module = runtime.risk_predictor
    transport_paths = [
        path for path, module in modules if path.endswith("chronotransport.transport") and module is transport_module
    ]
    risk_paths = [
        path for path, module in modules if path.endswith("chronotransport.risk_predictor") and module is risk_module
    ]
    scheduler_paths = [
        path for path, module in modules if path.endswith("chronotransport.scheduler.predictor")
    ]
    if len(transport_paths) != 1 or len(risk_paths) != 1:
        raise ValueError("Stage C requires unique canonical transport and risk modules")
    if not scheduler_paths or any(module_by_path[path] is not risk_module for path in scheduler_paths):
        raise ValueError("scheduler.predictor must alias the canonical risk predictor")

    adapters = _unique_parameters(adapter_modules)
    transport = _unique_parameters((transport_module,))
    risk = _unique_parameters((risk_module,))
    if not adapters or not transport or not risk:
        raise ValueError("Stage C requires non-empty A, T, and R parameter groups")
    identity_sets = [set(map(id, values)) for values in (adapters, transport, risk)]
    if identity_sets[0] & identity_sets[1] or identity_sets[0] & identity_sets[2] or identity_sets[1] & identity_sets[2]:
        raise ValueError("Stage C parameter ownership overlap")
    selected_parameters = tuple(sorted((*adapters, *transport, *risk), key=id))
    aliases: dict[int, list[str]] = {}
    for name, parameter in _named_parameters_with_aliases(model):
        aliases.setdefault(id(parameter), []).append(name)
    if any(id(parameter) not in aliases for parameter in selected_parameters):
        raise ValueError("Stage C could not resolve canonical parameter aliases")
    parameter_names = tuple(sorted(aliases[id(parameter)])[0] for parameter in selected_parameters)
    groups = StageCParameterGroups(
        adapters=adapters,
        transport=transport,
        risk=risk,
        adapter_paths=tuple(adapter_paths),
        transport_path=transport_paths[0],
        risk_path=risk_paths[0],
        scheduler_predictor_paths=tuple(sorted(scheduler_paths)),
        parameter_names=parameter_names,
    )
    selected = set(map(id, groups.all))
    model_parameters = tuple(model.parameters())
    missing = selected - set(map(id, model_parameters))
    if missing:
        raise ValueError("Stage C ownership includes parameters outside the model")
    for parameter in model_parameters:
        parameter.requires_grad = id(parameter) in selected
    requires_grad = {id(parameter) for parameter in model_parameters if parameter.requires_grad}
    if requires_grad != selected:
        raise RuntimeError("Stage C requires_grad union does not equal A/T/R ownership")
    return groups


def build_matched_dense_parameter_group(
    model: nn.Module,
) -> MatchedDenseParameterGroup:
    """Derive A from the same topology as CT, then freeze every non-A object."""

    stage_c = build_stage_c_parameter_groups(model)
    selected = set(map(id, stage_c.adapters))
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in selected
    if {id(parameter) for parameter in model.parameters() if parameter.requires_grad} != selected:
        raise StageCInvalidImplementationError(
            "matched-dense requires_grad union does not equal common A ownership"
        )
    aliases: dict[int, list[str]] = {}
    for name, parameter in _named_parameters_with_aliases(model):
        aliases.setdefault(id(parameter), []).append(name)
    return MatchedDenseParameterGroup(
        adapters=stage_c.adapters,
        adapter_paths=stage_c.adapter_paths,
        parameter_names=tuple(
            sorted(aliases[id(parameter)])[0]
            for parameter in sorted(stage_c.adapters, key=id)
        ),
    )


def _validate_stage_c_optimizer_groups(
    groups: StageCParameterGroups, optimizer: torch.optim.Optimizer
) -> int:
    expected_groups = (
        ("A", groups.adapters, 2e-4, 0.05),
        ("T", groups.transport, 1e-4, 0.0),
        ("R", groups.risk, 1e-4, 0.0),
    )
    if len(optimizer.param_groups) != 3:
        raise ValueError("Stage C optimizer requires exactly three ordered groups A/T/R")
    actual_ids: list[int] = []
    for actual, (name, parameters, lr, weight_decay) in zip(optimizer.param_groups, expected_groups):
        if actual.get("stage_c_group") != name or tuple(map(id, actual["params"])) != tuple(map(id, parameters)):
            raise ValueError("Stage C optimizer three ordered groups must exactly match A/T/R objects")
        if float(actual.get("stage_c_base_lr", float("nan"))) != lr or float(actual["weight_decay"]) != weight_decay:
            raise ValueError("Stage C optimizer group hyperparameters do not match the frozen contract")
        current_lr = float(actual["lr"])
        if not math.isfinite(current_lr) or current_lr < 0.0:
            raise ValueError("Stage C optimizer current LR must be finite and non-negative")
        actual_ids.extend(map(id, actual["params"]))
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(map(id, groups.all)):
        raise ValueError("Stage C optimizer must contain every A/T/R parameter object exactly once")
    return len(actual_ids)


def validate_stage_c_optimizer(
    groups: StageCParameterGroups,
    optimizer: torch.optim.Optimizer,
    *,
    lr_scheduler: Any,
) -> int:
    """Bind A/T/R optimizer ownership to the real frozen OpenTAD scheduler."""

    count = _validate_stage_c_optimizer_groups(groups, optimizer)
    from opentad.cores.scheduler import LinearWarmupCosineAnnealingLR

    if not isinstance(lr_scheduler, LinearWarmupCosineAnnealingLR):
        raise ValueError("Stage C requires OpenTAD LinearWarmupCosineAnnealingLR")
    if lr_scheduler.optimizer is not optimizer:
        raise ValueError("Stage C scheduler must reference the exact optimizer object")
    expected_base_lrs = [2e-4, 1e-4, 1e-4]
    if [float(value) for value in lr_scheduler.base_lrs] != expected_base_lrs:
        raise ValueError("Stage C scheduler base_lrs must equal the frozen A/T/R base LRs")
    initial_lrs = [float(group.get("initial_lr", float("nan"))) for group in optimizer.param_groups]
    if initial_lrs != expected_base_lrs:
        raise ValueError("Stage C optimizer initial_lr values must equal scheduler base_lrs")
    if int(lr_scheduler.warmup_epoch) != 350 or int(lr_scheduler.max_epoch) != 7000:
        raise ValueError("Stage C scheduler requires warmup=350 and max=7000 steps")
    if float(lr_scheduler.warmup_start_lr) != 0.0:
        raise ValueError("Stage C scheduler warmup_start_lr must equal 0")
    if float(lr_scheduler.eta_min) != 1e-8:
        raise ValueError("Stage C scheduler eta_min must equal 1e-8")
    current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    last_lrs = [float(value) for value in lr_scheduler.get_last_lr()]
    if current_lrs != last_lrs:
        raise ValueError("Stage C current LR must correspond to the bound scheduler state")
    closed_form_lrs = [float(value) for value in lr_scheduler._get_closed_form_lr()]
    if len(closed_form_lrs) != len(current_lrs) or any(
        not math.isclose(current, expected, rel_tol=1e-12, abs_tol=1e-15)
        for current, expected in zip(current_lrs, closed_form_lrs)
    ):
        raise ValueError("Stage C current LR must equal the scheduler closed-form LR")
    return count


def build_stage_c_optimizer(groups: StageCParameterGroups) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(
        [
            {"stage_c_group": "A", "stage_c_base_lr": 2e-4, "params": groups.adapters, "lr": 2e-4, "weight_decay": 0.05},
            {"stage_c_group": "T", "stage_c_base_lr": 1e-4, "params": groups.transport, "lr": 1e-4, "weight_decay": 0.0},
            {"stage_c_group": "R", "stage_c_base_lr": 1e-4, "params": groups.risk, "lr": 1e-4, "weight_decay": 0.0},
        ]
    )
    _validate_stage_c_optimizer_groups(groups, optimizer)
    return optimizer


def validate_matched_dense_optimizer(
    group: MatchedDenseParameterGroup,
    optimizer: torch.optim.Optimizer,
    *,
    lr_scheduler: Any,
) -> int:
    """Bind the matched arm to one common-A AdamW group and the same LR trace."""

    if len(optimizer.param_groups) != 1:
        raise ValueError("matched-dense optimizer requires exactly one A group")
    actual = optimizer.param_groups[0]
    if (
        actual.get("stage_c_group") != "A"
        or tuple(map(id, actual["params"])) != tuple(map(id, group.adapters))
        or float(actual.get("stage_c_base_lr", float("nan"))) != 2e-4
        or float(actual.get("weight_decay", float("nan"))) != 0.05
    ):
        raise ValueError("matched-dense optimizer does not exactly own common A")
    current_lr = float(actual["lr"])
    if not math.isfinite(current_lr) or current_lr < 0.0:
        raise ValueError("matched-dense current LR must be finite and non-negative")
    if list(map(id, actual["params"])) != list(
        dict.fromkeys(map(id, actual["params"]))
    ):
        raise ValueError("matched-dense optimizer contains duplicate A parameters")

    from opentad.cores.scheduler import LinearWarmupCosineAnnealingLR

    if not isinstance(lr_scheduler, LinearWarmupCosineAnnealingLR):
        raise ValueError("matched-dense requires OpenTAD LinearWarmupCosineAnnealingLR")
    if lr_scheduler.optimizer is not optimizer:
        raise ValueError("matched-dense scheduler must reference its exact optimizer")
    if [float(value) for value in lr_scheduler.base_lrs] != [2e-4]:
        raise ValueError("matched-dense scheduler base LR must equal common-A LR")
    if float(actual.get("initial_lr", float("nan"))) != 2e-4:
        raise ValueError("matched-dense optimizer initial_lr must equal common-A LR")
    if (
        int(lr_scheduler.warmup_epoch) != 350
        or int(lr_scheduler.max_epoch) != 7000
        or float(lr_scheduler.warmup_start_lr) != 0.0
        or float(lr_scheduler.eta_min) != 1e-8
    ):
        raise ValueError("matched-dense scheduler hyperparameters differ from Stage C")
    if [float(value) for value in lr_scheduler.get_last_lr()] != [current_lr]:
        raise ValueError("matched-dense optimizer and scheduler current LR differ")
    expected_lr = float(lr_scheduler._get_closed_form_lr()[0])
    if not math.isclose(current_lr, expected_lr, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("matched-dense current LR differs from the closed-form trace")
    return len(group.adapters)


def build_matched_dense_optimizer(
    group: MatchedDenseParameterGroup,
) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(
        [
            {
                "stage_c_group": "A",
                "stage_c_base_lr": 2e-4,
                "params": group.adapters,
                "lr": 2e-4,
                "weight_decay": 0.05,
            }
        ]
    )
    if len(optimizer.param_groups) != 1:
        raise AssertionError("matched-dense optimizer construction failed")
    return optimizer


def _gradient_sum(first: Tensor | None, second: Tensor | None, parameter: nn.Parameter) -> Tensor:
    if first is None and second is None:
        return torch.zeros_like(parameter)
    if first is None:
        return second.clone()
    if second is None:
        return first.clone()
    return (first + second).clone()


def _grad_audit(parameters: Sequence[nn.Parameter]) -> tuple[bool, bool, float]:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return True, False, 0.0
    finite = all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
    norm = _global_grad_norm_float64(parameters)
    return finite, norm > 0.0, norm


def _global_grad_norm_float64(parameters: Sequence[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        squared += float(parameter.grad.detach().double().square().sum().item())
    return math.sqrt(squared)


def _all_gradients_finite(parameters: Sequence[nn.Parameter]) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in parameters
    )


def _canonical_action_tensor(action_payload: Any) -> Tensor:
    if isinstance(action_payload, Tensor):
        actions = action_payload
    elif isinstance(action_payload, Mapping) and isinstance(action_payload.get("actions"), Tensor):
        actions = action_payload["actions"]
    else:
        raise ValueError("Stage-C action payload must provide the actual actions Tensor")
    if actions.ndim != 3 or tuple(actions.shape) != (2, 48, 3):
        raise ValueError("Stage-C actions must have canonical global-batch-two shape [2,48,3]")
    if actions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("Stage-C actions must use an integer dtype")
    values = torch.unique(actions.detach().cpu())
    if any(int(value.item()) not in (0, 1, 2) for value in values):
        raise ValueError("Stage-C actions contain an invalid ChronoAction value")
    if not bool(torch.all(actions[:, 0, :] == 0).item()):
        raise ValueError("Stage-C actions must RECOMPUTE the first chunk")
    return actions


def _transport_executed(action_payload: Any) -> bool:
    actions = _canonical_action_tensor(action_payload)
    return bool(torch.any(actions == 1).item())


def loss_specific_amp_step(
    *,
    detector_loss: Tensor,
    feature_loss: Tensor,
    risk_loss: Tensor,
    groups: StageCParameterGroups,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    scaler,
    action_payload: Any,
) -> dict[str, object]:
    """Perform the exact r2 loss-ownership update with one scaler step/update."""

    validate_stage_c_optimizer(groups, optimizer, lr_scheduler=lr_scheduler)
    transport_executed = _transport_executed(action_payload)
    optimizer.zero_grad(set_to_none=True)
    initial_scale = float(scaler.get_scale())
    if not math.isfinite(initial_scale) or initial_scale <= 0.0:
        raise StageCInvalidImplementationError("GradScaler scale must be finite and positive")
    losses_finite = all(
        loss.numel() == 1 and bool(torch.isfinite(loss.detach()).all().item())
        for loss in (detector_loss, feature_loss, risk_loss)
    )
    scale_trace: list[float] = []

    def scale_once(loss: Tensor) -> Tensor:
        before = float(scaler.get_scale())
        scale_trace.append(before)
        if before != initial_scale:
            raise StageCInvalidImplementationError(
                "GradScaler scale changed before the three fixed scale calls completed"
            )
        scaled = scaler.scale(loss)
        if float(scaler.get_scale()) != initial_scale:
            raise StageCInvalidImplementationError(
                "GradScaler scale changed within one Stage-C scale call"
            )
        return scaled

    scaled_detector = scale_once(detector_loss)
    detector_gradients = torch.autograd.grad(
        scaled_detector,
        groups.adapters + groups.transport,
        retain_graph=True,
        allow_unused=True,
    )
    scaled_feature = scale_once(0.1 * feature_loss)
    feature_gradients = torch.autograd.grad(
        scaled_feature,
        groups.transport,
        retain_graph=True,
        allow_unused=True,
    )
    scaled_risk = scale_once(0.1 * risk_loss)
    risk_gradients = torch.autograd.grad(
        scaled_risk,
        groups.risk,
        retain_graph=False,
        allow_unused=True,
    )
    if scale_trace != [initial_scale, initial_scale, initial_scale]:
        raise StageCInvalidImplementationError("Stage C must use one scale for exactly three losses")

    adapter_count = len(groups.adapters)
    for parameter, gradient in zip(groups.adapters, detector_gradients[:adapter_count]):
        parameter.grad = _gradient_sum(gradient, None, parameter)
    for index, parameter in enumerate(groups.transport):
        parameter.grad = _gradient_sum(
            detector_gradients[adapter_count + index], feature_gradients[index], parameter
        )
    for parameter, gradient in zip(groups.risk, risk_gradients):
        parameter.grad = _gradient_sum(gradient, None, parameter)

    scaler.unscale_(optimizer)
    adapter_finite, adapter_nonzero, adapter_norm = _grad_audit(groups.adapters)
    transport_finite, transport_nonzero, transport_norm = _grad_audit(groups.transport)
    risk_finite, risk_nonzero, risk_norm = _grad_audit(groups.risk)
    gradients_finite = adapter_finite and transport_finite and risk_finite
    preclip_norm = _global_grad_norm_float64(groups.all)
    postclip_norm = float("nan")
    postclip_finite = False
    if gradients_finite:
        if not losses_finite:
            optimizer.zero_grad(set_to_none=True)
            raise StageCInvalidImplementationError(
                "non-finite loss with finite gradients cannot be treated as a successful update"
            )
        if not adapter_nonzero:
            raise RuntimeError("Stage C aggregate adapter detector gradient must be nonzero")
        if not risk_nonzero:
            raise RuntimeError("Stage C aggregate risk gradient must be nonzero")
        if not math.isfinite(preclip_norm):
            raise StageCInvalidImplementationError("Stage C float64 global gradient norm is non-finite")
        clip_coefficient = min(1.0, 1.0 / (preclip_norm + 1e-6))
        for parameter in groups.all:
            if parameter.grad is not None:
                parameter.grad.mul_(clip_coefficient)
        postclip_finite = _all_gradients_finite(groups.all)
        postclip_norm = _global_grad_norm_float64(groups.all)
        if not postclip_finite or not math.isfinite(postclip_norm):
            raise StageCInvalidImplementationError("Stage C gradients became non-finite after clipping")
        if postclip_norm > 1.0 + 1e-6:
            raise StageCInvalidImplementationError("Stage C fixed global clip norm 1.0 was violated")
    scaler.step(optimizer)
    scaler.update()
    post_scale = float(scaler.get_scale())
    overflow = post_scale < initial_scale
    if (not gradients_finite or not losses_finite) and not overflow:
        raise StageCInvalidImplementationError(
            "non-finite Stage-C loss/gradients were not detected as GradScaler overflow"
        )
    return {
        "scale": initial_scale,
        "post_scale": post_scale,
        "overflow": overflow,
        "transport_executed": transport_executed,
        "losses_finite": losses_finite,
        "scale_trace": scale_trace,
        "adapter_detector_grad_finite": adapter_finite,
        "adapter_detector_grad_nonzero": adapter_nonzero,
        "adapter_grad_norm": adapter_norm,
        "transport_grad_finite": transport_finite,
        "transport_grad_nonzero": transport_nonzero,
        "transport_grad_norm": transport_norm,
        "risk_grad_finite": risk_finite,
        "risk_grad_nonzero": risk_nonzero,
        "risk_grad_norm": risk_norm,
        "preclip_global_grad_norm": preclip_norm,
        "preclip_global_grad_norm_float64": preclip_norm,
        "postclip_global_grad_norm_float64": postclip_norm,
        "postclip_gradients_finite": postclip_finite,
    }


def matched_dense_amp_step(
    *,
    detector_loss: Tensor,
    group: MatchedDenseParameterGroup,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    scaler: Any,
) -> dict[str, object]:
    """Perform the matched arm's sole common-A detector update."""

    validate_matched_dense_optimizer(group, optimizer, lr_scheduler=lr_scheduler)
    optimizer.zero_grad(set_to_none=True)
    initial_scale = float(scaler.get_scale())
    if not math.isfinite(initial_scale) or initial_scale <= 0.0:
        raise StageCInvalidImplementationError(
            "matched-dense GradScaler scale must be finite and positive"
        )
    loss_finite = (
        detector_loss.numel() == 1
        and bool(torch.isfinite(detector_loss.detach()).all().item())
    )
    scaled = scaler.scale(detector_loss)
    if float(scaler.get_scale()) != initial_scale:
        raise StageCInvalidImplementationError(
            "matched-dense GradScaler changed within its sole scale call"
        )
    gradients = torch.autograd.grad(
        scaled,
        group.adapters,
        retain_graph=False,
        allow_unused=True,
    )
    for parameter, gradient in zip(group.adapters, gradients):
        parameter.grad = _gradient_sum(gradient, None, parameter)
    scaler.unscale_(optimizer)
    finite, nonzero, preclip_norm = _grad_audit(group.adapters)
    postclip_norm = float("nan")
    postclip_finite = False
    if finite:
        if not loss_finite:
            optimizer.zero_grad(set_to_none=True)
            raise StageCInvalidImplementationError(
                "matched-dense non-finite loss with finite gradients is invalid"
            )
        if not nonzero:
            raise RuntimeError(
                "matched-dense aggregate adapter detector gradient must be nonzero"
            )
        clip_coefficient = min(1.0, 1.0 / (preclip_norm + 1e-6))
        for parameter in group.adapters:
            if parameter.grad is not None:
                parameter.grad.mul_(clip_coefficient)
        postclip_finite = _all_gradients_finite(group.adapters)
        postclip_norm = _global_grad_norm_float64(group.adapters)
        if (
            not postclip_finite
            or not math.isfinite(postclip_norm)
            or postclip_norm > 1.0 + 1e-6
        ):
            raise StageCInvalidImplementationError(
                "matched-dense fixed global clip norm 1.0 was violated"
            )
    scaler.step(optimizer)
    scaler.update()
    post_scale = float(scaler.get_scale())
    overflow = post_scale < initial_scale
    if (not finite or not loss_finite) and not overflow:
        raise StageCInvalidImplementationError(
            "matched-dense non-finite loss/gradients escaped GradScaler overflow"
        )
    return {
        "scale": initial_scale,
        "post_scale": post_scale,
        "overflow": overflow,
        "loss_finite": loss_finite,
        "adapter_detector_grad_finite": finite,
        "adapter_detector_grad_nonzero": nonzero,
        "adapter_grad_norm": preclip_norm,
        "preclip_global_grad_norm_float64": preclip_norm,
        "postclip_global_grad_norm_float64": postclip_norm,
        "postclip_gradients_finite": postclip_finite,
    }


def validate_transport_gradient_ledger(
    rows: Sequence[Mapping[str, Any]], *, expected_transport_exposures: int
) -> dict[str, float | int]:
    if (
        isinstance(expected_transport_exposures, bool)
        or not isinstance(expected_transport_exposures, int)
        or expected_transport_exposures < 0
    ):
        raise ValueError("expected_transport_exposures must be a non-negative integer")
    transport_rows = [row for row in rows if row.get("transport_executed") is True]
    if len(transport_rows) != expected_transport_exposures:
        raise ValueError("ledger does not contain the expected TRANSPORT exposures")
    for row in transport_rows:
        finite = row.get("transport_grad_finite")
        norm = row.get("transport_grad_norm")
        if finite is not True or isinstance(norm, bool) or not isinstance(norm, (int, float)):
            raise ValueError("every TRANSPORT exposure requires a finite gradient audit")
        if not math.isfinite(float(norm)) or float(norm) < 0.0:
            raise ValueError("TRANSPORT exposure gradient norms must be finite and non-negative")
    aggregate = sum(float(row["transport_grad_norm"]) for row in transport_rows)
    if expected_transport_exposures > 0 and aggregate <= 0.0:
        raise ValueError("aggregate TRANSPORT exposure gradient norm must be greater than zero")
    return {
        "transport_exposures": len(transport_rows),
        "aggregate_transport_grad_norm": aggregate,
    }


def _hash_value(hasher: Any, value: Any) -> None:
    """Feed an unambiguous, byte-exact representation into ``hasher``."""

    if value is None:
        hasher.update(b"N")
    elif isinstance(value, bool):
        hasher.update(b"B1" if value else b"B0")
    elif isinstance(value, int):
        encoded = str(value).encode("ascii")
        hasher.update(b"I" + struct.pack(">Q", len(encoded)) + encoded)
    elif isinstance(value, float):
        hasher.update(b"F" + struct.pack(">d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        hasher.update(b"S" + struct.pack(">Q", len(encoded)) + encoded)
    elif isinstance(value, bytes):
        hasher.update(b"Y" + struct.pack(">Q", len(value)) + value)
    elif isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        metadata = f"{tensor.dtype}|{tuple(tensor.shape)}".encode("ascii")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        hasher.update(b"T" + struct.pack(">Q", len(metadata)) + metadata)
        hasher.update(struct.pack(">Q", len(raw)) + raw)
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        metadata = f"{array.dtype.str}|{array.shape}".encode("ascii")
        raw = array.tobytes()
        hasher.update(b"A" + struct.pack(">Q", len(metadata)) + metadata)
        hasher.update(struct.pack(">Q", len(raw)) + raw)
    elif isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("materialized batch mapping keys must be strings")
        hasher.update(b"M" + struct.pack(">Q", len(value)))
        for key in sorted(value):
            _hash_value(hasher, key)
            _hash_value(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        hasher.update((b"L" if isinstance(value, list) else b"Q") + struct.pack(">Q", len(value)))
        for item in value:
            _hash_value(hasher, item)
    else:
        raise TypeError(f"unsupported materialized batch value: {type(value).__name__}")


def hash_materialized_batch(materialized_batch: Any) -> str:
    hasher = hashlib.sha256()
    _hash_value(hasher, materialized_batch)
    return hasher.hexdigest()


def stage_c_action_hash(
    *,
    seed: int,
    successful_update: int,
    batch_position: int,
    action_payload: Any,
) -> str:
    exposures = stage_c_batch_exposures(seed, successful_update)
    if (
        isinstance(batch_position, bool)
        or not isinstance(batch_position, int)
        or batch_position not in (0, 1)
    ):
        raise ValueError("batch_position must be 0 or 1")
    if isinstance(action_payload, Tensor) and tuple(action_payload.shape) == (48, 3):
        action = action_payload
    else:
        action = _canonical_action_tensor(action_payload)[batch_position]
    candidate_ordinal = exposures[batch_position]["candidate"]
    candidate_id = R2_NON_DENSE_NAMES[candidate_ordinal]
    hasher = hashlib.sha256()
    hasher.update(b"chronotransport-r2-stage-c-window-action-v2\x00")
    _hash_value(hasher, seed)
    _hash_value(hasher, successful_update)
    _hash_value(hasher, batch_position)
    _hash_value(hasher, exposures[batch_position]["window_exposure_ordinal"])
    _hash_value(hasher, candidate_ordinal)
    _hash_value(hasher, candidate_id)
    _hash_value(hasher, action)
    return hasher.hexdigest()


def _action_batch_sha256(exposures: Sequence[Mapping[str, Any]]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"chronotransport-r2-stage-c-action-batch-v1\x00")
    for exposure in exposures:
        _hash_value(hasher, dict(exposure))
    return hasher.hexdigest()


def _validate_stage_c_attempt_actions(
    *, seed: int, successful_update: int, action_payload: Any
) -> tuple[tuple[dict[str, Any], dict[str, Any]], str]:
    actions = _canonical_action_tensor(action_payload)
    protocol_rows = stage_c_batch_exposures(seed, successful_update)
    rows: list[dict[str, Any]] = []
    for batch_position, protocol_row in enumerate(protocol_rows):
        candidate_ordinal = protocol_row["candidate"]
        candidate_id = R2_NON_DENSE_NAMES[candidate_ordinal]
        actual = actions[batch_position].detach().cpu().to(torch.int8)
        expected = _R2_STAGE_C_ACTIONS[candidate_ordinal]
        if not torch.equal(actual, expected):
            raise StageCInvalidImplementationError(
                "actual action bytes do not match the canonical Stage-C exposure"
            )
        rows.append(
            {
                "batch_position": batch_position,
                "window_exposure_ordinal": protocol_row["window_exposure_ordinal"],
                "candidate_ordinal": candidate_ordinal,
                "candidate_id": candidate_id,
                "actual_action_sha256": stage_c_action_hash(
                    seed=seed,
                    successful_update=successful_update,
                    batch_position=batch_position,
                    action_payload=actual,
                ),
            }
        )
    frozen = (rows[0], rows[1])
    return frozen, _action_batch_sha256(frozen)


def _canonical_stage_c_runtime(
    model: nn.Module, groups: StageCParameterGroups
) -> tuple[str, nn.Module]:
    suffix = ".transport"
    if not groups.transport_path.endswith(suffix):
        raise StageCInvalidImplementationError(
            "Stage C transport registry cannot identify the canonical runtime"
        )
    runtime_path = groups.transport_path[: -len(suffix)]
    matches = [
        module
        for path, module in _named_modules_with_aliases(model)
        if path == runtime_path
    ]
    if len(matches) != 1:
        raise StageCInvalidImplementationError(
            "Stage C requires one uniquely addressed canonical ChronoTransport runtime"
        )
    runtime = matches[0]
    if (
        type(runtime) is not _FORMAL_CHRONOTRANSPORT_RUNTIME
        or type(runtime).forward is not _FORMAL_CHRONOTRANSPORT_FORWARD
        or "forward" in runtime.__dict__
    ):
        raise StageCInvalidImplementationError(
            "Stage C formal evidence requires the production ChronoTransportRuntime class/source identity"
        )
    if (
        getattr(runtime, "transport", None) is None
        or getattr(runtime, "risk_predictor", None) is None
        or not isinstance(getattr(runtime, "forced_actions", None), Tensor)
        or "forced_actions" not in runtime._buffers
        or not hasattr(runtime, "latest_schedule")
        or not hasattr(runtime, "latest_summary")
        or not hasattr(runtime, "capture_replay_signals")
    ):
        raise StageCInvalidImplementationError(
            "canonical ChronoTransport runtime lacks the formal action evidence capability"
        )
    return runtime_path, runtime


def _canonical_matched_dense_runtime(
    model: nn.Module, group: MatchedDenseParameterGroup
) -> tuple[str, ChronoTransportRuntime]:
    matches = [
        (path, module)
        for path, module in _named_modules_with_aliases(model)
        if type(module) is _FORMAL_CHRONOTRANSPORT_RUNTIME
        and type(module).forward is _FORMAL_CHRONOTRANSPORT_FORWARD
        and "forward" not in module.__dict__
    ]
    if len(matches) != 1:
        raise StageCInvalidImplementationError(
            "matched-dense requires one production ChronoTransportRuntime"
        )
    runtime_path, runtime = matches[0]
    if not runtime_path.endswith("chronotransport"):
        raise StageCInvalidImplementationError(
            "matched-dense runtime path is not canonical"
        )
    selected = set(map(id, group.adapters))
    if not selected or {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    } != selected:
        raise StageCInvalidImplementationError(
            "matched-dense trainable parameters differ from common A"
        )
    return runtime_path, runtime


def _canonical_stage_c_action_batch(
    *, seed: int, successful_update: int, device: torch.device
) -> Tensor:
    exposures = stage_c_batch_exposures(seed, successful_update)
    actions = torch.stack(
        [_R2_STAGE_C_ACTIONS[row["candidate"]] for row in exposures]
    ).to(device=device, dtype=torch.long)
    return _canonical_action_tensor(actions).detach().clone()


def _install_stage_c_action_batch(
    runtime: nn.Module, *, seed: int, successful_update: int
) -> Tensor:
    current = runtime.forced_actions
    actions = _canonical_stage_c_action_batch(
        seed=seed, successful_update=successful_update, device=current.device
    )
    runtime.forced_actions = actions
    if hasattr(runtime, "forced_schedule"):
        runtime.forced_schedule = None
    runtime.forced_action_name = (
        f"stage_c_seed_{seed}_successful_update_{successful_update}"
    )
    runtime.capture_replay_signals = True
    runtime.latest_schedule = None
    runtime.latest_summary = None
    for name in ("latest_output", "latest_signals"):
        if hasattr(runtime, name):
            setattr(runtime, name, None)
    return actions.detach().clone()


_FORMAL_RUNTIME_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "enabled",
        "forced_dense_exact_path",
        "whole_window_dense_fallback",
        "selected_schedule_names",
        "executed_schedule_name",
        "recompute_rows",
        "transport_rows",
        "hold_rows",
        "adapter_dense_forward_count",
        "runtime_fail_closed_repairs",
        "schedule_repair_count",
        "first_chunk_forced_recompute",
        "dense_output_shape_preserved",
        "adapter_path_dense",
        "adapter_writeback",
        "heavy_attention_mlp_gathered",
        "cache_reset_per_window",
        "transport_uses_latest_cache",
        "cost_is_measured",
        "registered_cost_profile_sha256",
        "registered_gate3_calibration_sha256",
        "registered_q_conf",
        "registered_budget",
        "risk_ready",
        "checkpoint_loaded",
        "require_checkpoint_for_dynamic",
        "external_dense_grid_preserved_by_post_interpolation",
        "chunks_per_window",
        "tubelets_per_chunk",
        "internal_tubelet_points",
        "spatial_tokens_per_tubelet",
        "upper_risk",
        "estimated_cost",
        "requested_estimated_cost",
        "executed_estimated_cost",
        "fail_closed",
        "action_counts",
        "requested_action_counts",
        "requested_action_sha256",
        "executed_action_sha256",
        "evidence_valid",
        "invalid_implementation_reason",
        "profile",
    }
)


def _finite_number_list(value: Any, *, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in value
        )
    )


def _validate_formal_runtime_summary(
    summary: Mapping[str, Any], *, runtime: ChronoTransportRuntime, actions: Tensor
) -> None:
    if set(summary) != _FORMAL_RUNTIME_SUMMARY_KEYS:
        raise StageCInvalidImplementationError(
            "formal runtime summary schema keys are not exact"
        )
    exact_true = (
        "enabled",
        "dense_output_shape_preserved",
        "adapter_path_dense",
        "heavy_attention_mlp_gathered",
        "cache_reset_per_window",
        "transport_uses_latest_cache",
        "external_dense_grid_preserved_by_post_interpolation",
        "evidence_valid",
        "cost_is_measured",
    )
    exact_false = (
        "forced_dense_exact_path",
        "whole_window_dense_fallback",
        "first_chunk_forced_recompute",
    )
    typed_booleans = (
        "risk_ready",
        "checkpoint_loaded",
        "require_checkpoint_for_dynamic",
    )
    if any(summary[name] is not True for name in exact_true) or any(
        summary[name] is not False for name in exact_false
    ):
        raise StageCInvalidImplementationError(
            "formal runtime summary contains unsafe boolean evidence"
        )
    if any(type(summary[name]) is not bool for name in typed_booleans):
        raise StageCInvalidImplementationError(
            "formal runtime summary boolean field types are invalid"
        )
    if (
        summary["schema_version"] != "chronotransport_runtime_v1"
        or summary["adapter_writeback"] != "all_rows"
        or summary["invalid_implementation_reason"] is not None
        or summary["executed_schedule_name"] != runtime.forced_action_name
        or summary["selected_schedule_names"]
        != [runtime.forced_action_name, runtime.forced_action_name]
        or summary["fail_closed"] != [False, False]
    ):
        raise StageCInvalidImplementationError(
            "formal runtime summary identity/fallback evidence is invalid"
        )
    profile_sha256 = summary["registered_cost_profile_sha256"]
    if (
        profile_sha256 != runtime.scheduler.registered_cost_profile_sha256
        or (
            profile_sha256 is not None
            and (
                not isinstance(profile_sha256, str)
                or len(profile_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in profile_sha256
                )
            )
        )
    ):
        raise StageCInvalidImplementationError(
            "formal runtime summary registered cost provenance is invalid"
        )
    if any(
        summary[name] is not None
        for name in (
            "registered_gate3_calibration_sha256",
            "registered_q_conf",
            "registered_budget",
        )
    ):
        raise StageCInvalidImplementationError(
            "Stage C must run before post-Stage-C Gate3 calibration is installed"
        )
    integer_fields = (
        "recompute_rows",
        "transport_rows",
        "hold_rows",
        "adapter_dense_forward_count",
        "runtime_fail_closed_repairs",
        "schedule_repair_count",
        "chunks_per_window",
        "tubelets_per_chunk",
        "internal_tubelet_points",
        "spatial_tokens_per_tubelet",
    )
    if any(
        isinstance(summary[name], bool)
        or not isinstance(summary[name], int)
        or summary[name] < 0
        for name in integer_fields
    ):
        raise StageCInvalidImplementationError(
            "formal runtime summary counter/geometry types are invalid"
        )
    if (
        summary["runtime_fail_closed_repairs"] != 0
        or summary["schedule_repair_count"] != 0
        or summary["chunks_per_window"] != 48
        or summary["tubelets_per_chunk"] <= 0
        or summary["internal_tubelet_points"]
        != 48 * summary["tubelets_per_chunk"]
        or summary["spatial_tokens_per_tubelet"] <= 0
    ):
        raise StageCInvalidImplementationError(
            "formal runtime summary repair/geometry evidence is unsafe"
        )
    if not all(
        _finite_number_list(summary[name], length=2)
        for name in (
            "upper_risk",
            "estimated_cost",
            "requested_estimated_cost",
            "executed_estimated_cost",
        )
    ):
        raise StageCInvalidImplementationError(
            "formal runtime summary risk/cost evidence is invalid"
        )
    expected_counts = {
        "recompute": int((actions == 0).sum().item()),
        "transport": int((actions == 1).sum().item()),
        "hold": int((actions == 2).sum().item()),
    }
    expected_sha256 = canonical_sha256(
        actions.detach().cpu().to(torch.long).tolist()
    )
    if (
        summary["action_counts"] != expected_counts
        or summary["requested_action_counts"] != expected_counts
        or summary["requested_action_sha256"] != expected_sha256
        or summary["executed_action_sha256"] != expected_sha256
        or not isinstance(summary["profile"], Mapping)
    ):
        raise StageCInvalidImplementationError(
            "formal runtime summary action/profile evidence is invalid"
        )


@dataclass(frozen=True)
class _TensorBoundarySnapshot:
    name: str
    reference: Tensor
    value: Tensor
    version: int
    layout: torch.layout
    dtype: torch.dtype
    device: torch.device
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    storage_cdata: int
    storage_data_ptr: int
    storage_nbytes: int
    requires_grad: bool
    is_leaf: bool
    grad_fn_type: str | None


def _snapshot_tensor_boundary(name: str, tensor: Any) -> _TensorBoundarySnapshot:
    if not isinstance(tensor, Tensor) or tensor.layout != torch.strided:
        raise StageCInvalidImplementationError(
            f"Stage C {name} boundary must publish one strided Tensor"
        )
    storage = tensor.untyped_storage()
    return _TensorBoundarySnapshot(
        name=name,
        reference=tensor,
        value=tensor.detach().clone(memory_format=torch.preserve_format),
        version=int(tensor._version),
        layout=tensor.layout,
        dtype=tensor.dtype,
        device=tensor.device,
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        storage_offset=int(tensor.storage_offset()),
        storage_cdata=int(storage._cdata),
        storage_data_ptr=int(storage.data_ptr()),
        storage_nbytes=int(storage.nbytes()),
        requires_grad=bool(tensor.requires_grad),
        is_leaf=bool(tensor.is_leaf),
        grad_fn_type=(
            None if tensor.grad_fn is None else type(tensor.grad_fn).__qualname__
        ),
    )


def _assert_tensor_boundary(
    snapshot: _TensorBoundarySnapshot, tensor: Any
) -> None:
    if tensor is not snapshot.reference:
        raise StageCInvalidImplementationError(
            f"Stage C {snapshot.name} boundary reference identity changed"
        )
    if not isinstance(tensor, Tensor) or tensor.layout != torch.strided:
        raise StageCInvalidImplementationError(
            f"Stage C {snapshot.name} boundary Tensor layout changed"
        )
    storage = tensor.untyped_storage()
    metadata = (
        tensor.layout,
        tensor.dtype,
        tensor.device,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        int(tensor.storage_offset()),
        bool(tensor.requires_grad),
        bool(tensor.is_leaf),
        None if tensor.grad_fn is None else type(tensor.grad_fn).__qualname__,
    )
    expected_metadata = (
        snapshot.layout,
        snapshot.dtype,
        snapshot.device,
        snapshot.shape,
        snapshot.stride,
        snapshot.storage_offset,
        snapshot.requires_grad,
        snapshot.is_leaf,
        snapshot.grad_fn_type,
    )
    storage_metadata = (
        int(storage._cdata),
        int(storage.data_ptr()),
        int(storage.nbytes()),
    )
    expected_storage_metadata = (
        snapshot.storage_cdata,
        snapshot.storage_data_ptr,
        snapshot.storage_nbytes,
    )
    if metadata != expected_metadata or storage_metadata != expected_storage_metadata:
        raise StageCInvalidImplementationError(
            f"Stage C {snapshot.name} boundary storage/metadata changed"
        )
    if int(tensor._version) != snapshot.version or not torch.equal(
        tensor.detach(), snapshot.value
    ):
        raise StageCInvalidImplementationError(
            f"Stage C {snapshot.name} boundary logical bytes/value changed"
        )


@dataclass(frozen=True)
class _StageCRuntimeEvidence:
    actions: Tensor
    summary: Mapping[str, Any]
    detector_output: Tensor
    detector_boundary: _TensorBoundarySnapshot
    feature_output: Tensor
    feature_boundary: _TensorBoundarySnapshot
    signals_boundary: _TensorBoundarySnapshot
    risk_output: Tensor


def _run_attempt_with_runtime_evidence(
    *,
    attempt: Callable[[], StageCAttemptLosses],
    model: nn.Module,
    runtime: nn.Module,
    expected_actions: Tensor,
    require_cuda_autocast: bool,
) -> tuple[StageCAttemptLosses, _StageCRuntimeEvidence]:
    state: dict[str, Any] = {
        "model_depth": 0,
        "model_forwards": 0,
        "runtime_forwards": 0,
        "risk_forwards": 0,
        "actions": None,
        "summary": None,
        "detector_output": None,
        "detector_boundary": None,
        "feature_output": None,
        "feature_boundary": None,
        "signals_boundary": None,
        "risk_output": None,
    }

    def model_pre_hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
        del module, inputs
        if require_cuda_autocast and not torch.is_autocast_enabled():
            raise StageCInvalidImplementationError(
                "formal CUDA Stage C requires autocast during the audited model forward"
            )
        if state["model_depth"] != 0 or state["model_forwards"] != 0:
            raise StageCInvalidImplementationError(
                "Stage C attempt must execute exactly one top-level model forward"
            )
        state["model_depth"] = 1
        state["model_forwards"] = 1

    def model_forward_hook(
        module: nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> None:
        del module, inputs
        if not isinstance(output, Tensor) or not output.requires_grad:
            raise StageCInvalidImplementationError(
                "Stage C model forward must publish one differentiable canonical detector Tensor"
            )
        state["detector_output"] = output
        state["detector_boundary"] = _snapshot_tensor_boundary(
            "detector output", output
        )
        state["model_depth"] = 0

    def runtime_pre_hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
        del inputs
        if state["model_depth"] != 1:
            raise StageCInvalidImplementationError(
                "canonical ChronoTransport runtime must execute inside the audited model forward"
            )
        if state["runtime_forwards"] != 0:
            raise StageCInvalidImplementationError(
                "Stage C attempt executed the canonical runtime more than once"
            )
        actual_forced = _canonical_action_tensor(module.forced_actions)
        if not torch.equal(
            actual_forced.detach().cpu().to(torch.long),
            expected_actions.detach().cpu().to(torch.long),
        ):
            raise StageCInvalidImplementationError(
                "canonical Stage-C action batch changed before runtime forward"
            )
        state["runtime_forwards"] = 1

    def runtime_forward_hook(
        module: nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> None:
        del inputs
        schedule = getattr(module, "latest_schedule", None)
        summary = getattr(module, "latest_summary", None)
        actions = getattr(schedule, "actions", None)
        if not isinstance(actions, Tensor) or not isinstance(summary, Mapping):
            raise StageCInvalidImplementationError(
                "canonical runtime forward did not publish executed action evidence"
            )
        state["actions"] = actions.detach().clone()
        state["summary"] = copy.deepcopy(dict(summary))
        feature_output = getattr(module, "latest_output", None)
        if not isinstance(feature_output, Tensor) or not feature_output.requires_grad:
            raise StageCInvalidImplementationError(
                "canonical runtime did not publish a differentiable feature output"
            )
        if output is not feature_output:
            raise StageCInvalidImplementationError(
                "canonical runtime return and feature output boundary identities differ"
            )
        signals = getattr(module, "latest_signals", None)
        if not isinstance(signals, Tensor):
            raise StageCInvalidImplementationError(
                "canonical runtime did not publish deploy-visible forward signals"
            )
        state["feature_output"] = feature_output
        state["feature_boundary"] = _snapshot_tensor_boundary(
            "feature output", feature_output
        )
        state["signals_boundary"] = _snapshot_tensor_boundary(
            "latest signals", signals
        )

    def risk_pre_hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
        del module
        if require_cuda_autocast and not torch.is_autocast_enabled():
            raise StageCInvalidImplementationError(
                "formal CUDA Stage C requires autocast during the audited risk forward"
            )
        if (
            state["model_depth"] != 0
            or state["runtime_forwards"] != 1
            or state["risk_forwards"] != 0
            or len(inputs) < 2
        ):
            raise StageCInvalidImplementationError(
                "Stage C risk predictor must execute exactly once after the audited runtime forward"
            )
        signals, actions = inputs[:2]
        signals_boundary = state["signals_boundary"]
        if not isinstance(signals_boundary, _TensorBoundarySnapshot):
            raise StageCInvalidImplementationError(
                "Stage C risk predictor lacks a runtime signals boundary"
            )
        _assert_tensor_boundary(signals_boundary, runtime.latest_signals)
        if signals is not signals_boundary.reference or not isinstance(actions, Tensor):
            raise StageCInvalidImplementationError(
                "Stage C risk predictor inputs are not the audited runtime signals/actions"
            )
        if not torch.equal(
            actions.detach().cpu().to(torch.long),
            state["actions"].detach().cpu().to(torch.long),
        ):
            raise StageCInvalidImplementationError(
                "Stage C risk predictor action input differs from executed actions"
            )
        state["risk_forwards"] = 1

    def risk_forward_hook(
        module: nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> None:
        del module, inputs
        if not isinstance(output, Tensor) or not output.requires_grad:
            raise StageCInvalidImplementationError(
                "Stage C risk predictor did not publish a differentiable Tensor"
            )
        signals_boundary = state["signals_boundary"]
        if not isinstance(signals_boundary, _TensorBoundarySnapshot):
            raise StageCInvalidImplementationError(
                "Stage C risk predictor lacks a runtime signals boundary"
            )
        _assert_tensor_boundary(signals_boundary, runtime.latest_signals)
        state["risk_output"] = output

    handles = (
        model.register_forward_pre_hook(model_pre_hook),
        model.register_forward_hook(model_forward_hook),
        runtime.register_forward_pre_hook(runtime_pre_hook),
        runtime.register_forward_hook(runtime_forward_hook),
        runtime.risk_predictor.register_forward_pre_hook(risk_pre_hook),
        runtime.risk_predictor.register_forward_hook(risk_forward_hook),
    )
    try:
        losses = attempt()
    finally:
        for handle in handles:
            handle.remove()
    if not isinstance(losses, StageCAttemptLosses):
        raise TypeError("attempt must return StageCAttemptLosses")
    if (
        state["model_forwards"] != 1
        or state["model_depth"] != 0
        or state["runtime_forwards"] != 1
        or state["risk_forwards"] != 1
        or not isinstance(state["actions"], Tensor)
        or not isinstance(state["summary"], Mapping)
        or not isinstance(state["detector_output"], Tensor)
        or not isinstance(state["detector_boundary"], _TensorBoundarySnapshot)
        or not isinstance(state["feature_output"], Tensor)
        or not isinstance(state["feature_boundary"], _TensorBoundarySnapshot)
        or not isinstance(state["signals_boundary"], _TensorBoundarySnapshot)
        or not isinstance(state["risk_output"], Tensor)
    ):
        raise StageCInvalidImplementationError(
            "Stage C attempt lacks one complete canonical runtime model forward"
        )
    summary = state["summary"]
    _validate_formal_runtime_summary(
        summary, runtime=runtime, actions=state["actions"]
    )
    _assert_tensor_boundary(state["signals_boundary"], runtime.latest_signals)
    _assert_tensor_boundary(
        state["detector_boundary"], state["detector_output"]
    )
    _assert_tensor_boundary(
        state["feature_boundary"], runtime.latest_output
    )
    evidence = _StageCRuntimeEvidence(
        actions=state["actions"],
        summary=copy.deepcopy(dict(summary)),
        detector_output=state["detector_output"],
        detector_boundary=state["detector_boundary"],
        feature_output=state["feature_output"],
        feature_boundary=state["feature_boundary"],
        signals_boundary=state["signals_boundary"],
        risk_output=state["risk_output"],
    )
    return losses, evidence


def _nonzero_tensor(value: Tensor) -> bool:
    return bool(torch.any(value.detach() != 0).item())


def _gradient_matches(left: Tensor | None, right: Tensor | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(left.detach(), right.detach())
    )


def _assert_loss_source_provenance(
    *,
    name: str,
    loss: Tensor,
    source: Tensor,
    parameters: Sequence[nn.Parameter],
) -> None:
    """Prove that a loss's selected gradients are exactly source-mediated.

    Connectivity alone is insufficient because ``0 * source + direct_loss``
    remains connected.  We therefore compare the loss's full selected gradient
    with a VJP through the audited forward source using the detached upstream
    gradient.  Any additional direct parameter path changes that equality.
    """

    if loss.numel() != 1 or not loss.requires_grad:
        raise StageCInvalidImplementationError(
            f"Stage C {name} loss must be one differentiable scalar"
        )
    (source_gradient,) = torch.autograd.grad(
        loss,
        (source,),
        retain_graph=True,
        allow_unused=True,
    )
    if source_gradient is None or not _nonzero_tensor(source_gradient):
        raise StageCInvalidImplementationError(
            f"Stage C {name} loss is not bound to its audited forward source"
        )
    loss_finite = bool(torch.isfinite(loss.detach()).all().item())
    if loss_finite and not bool(torch.isfinite(source_gradient.detach()).all().item()):
        raise StageCInvalidImplementationError(
            f"Stage C finite {name} loss has non-finite forward-source gradient"
        )
    # NaN arithmetic cannot support an equality proof.  A non-finite attempt is
    # allowed only to reach GradScaler's fail-closed overflow path; the eventual
    # successful finite replay is subjected to the exact VJP equality below.
    if not loss_finite:
        return
    full_gradients = torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=True,
        allow_unused=True,
    )
    mediated_gradients = torch.autograd.grad(
        source,
        tuple(parameters),
        grad_outputs=source_gradient.detach(),
        retain_graph=True,
        allow_unused=True,
    )
    if any(
        not _gradient_matches(full, mediated)
        for full, mediated in zip(full_gradients, mediated_gradients)
    ):
        raise StageCInvalidImplementationError(
            f"Stage C {name} loss has a direct/unapproved path outside its audited forward source"
        )


def _assert_attempt_loss_provenance(
    losses: StageCAttemptLosses,
    evidence: _StageCRuntimeEvidence,
    groups: StageCParameterGroups,
) -> None:
    _assert_loss_source_provenance(
        name="detector",
        loss=losses.detector_loss,
        source=evidence.detector_output,
        parameters=groups.adapters + groups.transport,
    )
    _assert_loss_source_provenance(
        name="feature",
        loss=losses.feature_loss,
        source=evidence.feature_output,
        parameters=groups.transport,
    )
    _assert_loss_source_provenance(
        name="risk",
        loss=losses.risk_loss,
        source=evidence.risk_output,
        parameters=groups.risk,
    )


@dataclass(frozen=True)
class _RNGSnapshot:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: Tensor
    torch_cuda: tuple[Tensor, ...] | None


@dataclass(frozen=True)
class _BufferSnapshot:
    reference: Tensor | None
    value: Tensor | None
    version: int | None
    layout: torch.layout | None
    dtype: torch.dtype | None
    device: torch.device | None
    shape: tuple[int, ...] | None
    stride: tuple[int, ...] | None
    storage_offset: int | None
    storage_cdata: int | None
    storage_data_ptr: int | None
    storage_nbytes: int | None
    requires_grad: bool | None
    persistent: bool


@dataclass(frozen=True)
class _ParameterSnapshot:
    name: str
    reference: nn.Parameter
    value: Tensor | None
    version: int
    dtype: torch.dtype
    device: torch.device
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    storage_data_ptr: int
    storage_nbytes: int
    tensor_nbytes: int
    requires_grad: bool
    frozen_logical_sha256: str | None


@dataclass(frozen=True)
class _ModelTopologySnapshot:
    modules: tuple[tuple[str, int, type], ...]
    parameters: tuple[tuple[str, int, type], ...]
    buffers: tuple[tuple[str, int, type], ...]
    module_alias_multiplicity: tuple[tuple[int, int], ...]
    parameter_alias_multiplicity: tuple[tuple[int, int], ...]
    buffer_alias_multiplicity: tuple[tuple[int, int], ...]
    registrations: tuple[
        tuple[
            str,
            tuple[tuple[str, int | None, type | None], ...],
            tuple[tuple[str, int | None, type | None], ...],
            tuple[tuple[str, int | None, type | None], ...],
        ],
        ...,
    ]


def _identity_multiplicity(
    rows: Sequence[tuple[str, int, type]],
) -> tuple[tuple[int, int], ...]:
    counts: dict[int, int] = {}
    for _, identity, _ in rows:
        counts[identity] = counts.get(identity, 0) + 1
    return tuple(sorted(counts.items()))


def _registration_rows(values: Mapping[str, Any]) -> tuple[tuple[str, int | None, type | None], ...]:
    return tuple(
        (name, None if value is None else id(value), None if value is None else type(value))
        for name, value in values.items()
    )


def _model_topology_state(model: nn.Module) -> _ModelTopologySnapshot:
    modules_with_aliases = _named_modules_with_aliases(model)
    module_rows = tuple((path, id(module), type(module)) for path, module in modules_with_aliases)
    parameter_rows = tuple(
        (path, id(parameter), type(parameter))
        for path, parameter in _named_parameters_with_aliases(model)
    )
    buffer_rows = tuple(
        (path, id(buffer), type(buffer))
        for path, buffer in _named_buffers_with_aliases(model)
    )
    registrations = tuple(
        (
            path,
            _registration_rows(module._modules),
            _registration_rows(module._parameters),
            _registration_rows(module._buffers),
        )
        for path, module in modules_with_aliases
    )
    return _ModelTopologySnapshot(
        modules=module_rows,
        parameters=parameter_rows,
        buffers=buffer_rows,
        module_alias_multiplicity=_identity_multiplicity(module_rows),
        parameter_alias_multiplicity=_identity_multiplicity(parameter_rows),
        buffer_alias_multiplicity=_identity_multiplicity(buffer_rows),
        registrations=registrations,
    )


def _assert_model_topology(
    expected: _ModelTopologySnapshot, model: nn.Module, *, where: str
) -> None:
    if _model_topology_state(model) != expected:
        raise StageCInvalidImplementationError(
            f"model topology/alias/parent registration order mutated {where}"
        )


def _snapshot_rng() -> _RNGSnapshot:
    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return _RNGSnapshot(
        python=copy.deepcopy(random.getstate()),
        numpy=copy.deepcopy(np.random.get_state()),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=cuda_state,
    )


def _restore_rng(snapshot: _RNGSnapshot) -> None:
    random.setstate(snapshot.python)
    np.random.set_state(snapshot.numpy)
    torch.set_rng_state(snapshot.torch_cpu)
    if snapshot.torch_cuda is not None:
        if not torch.cuda.is_available():
            raise StageCInvalidImplementationError("CUDA RNG snapshot cannot be restored")
        torch.cuda.set_rng_state_all(list(snapshot.torch_cuda))


def _module_buffer_state(model: nn.Module) -> dict[str, dict[str, _BufferSnapshot]]:
    state: dict[str, dict[str, _BufferSnapshot]] = {}
    for path, module in model.named_modules():
        buffers: dict[str, _BufferSnapshot] = {}
        for name, buffer in module._buffers.items():
            persistent = name not in module._non_persistent_buffers_set
            if buffer is None:
                buffers[name] = _BufferSnapshot(
                    reference=None,
                    value=None,
                    version=None,
                    layout=None,
                    dtype=None,
                    device=None,
                    shape=None,
                    stride=None,
                    storage_offset=None,
                    storage_cdata=None,
                    storage_data_ptr=None,
                    storage_nbytes=None,
                    requires_grad=None,
                    persistent=persistent,
                )
                continue
            if buffer.layout != torch.strided:
                raise StageCInvalidImplementationError(
                    f"Stage C cannot snapshot non-strided buffer {path or '<root>'}.{name}"
                )
            storage = buffer.untyped_storage()
            buffers[name] = _BufferSnapshot(
                reference=buffer,
                value=buffer.detach().clone(memory_format=torch.preserve_format),
                version=int(buffer._version),
                layout=buffer.layout,
                dtype=buffer.dtype,
                device=buffer.device,
                shape=tuple(buffer.shape),
                stride=tuple(buffer.stride()),
                storage_offset=int(buffer.storage_offset()),
                storage_cdata=int(storage._cdata),
                storage_data_ptr=int(storage.data_ptr()),
                storage_nbytes=int(storage.nbytes()),
                requires_grad=bool(buffer.requires_grad),
                persistent=persistent,
            )
        state[path] = buffers
    return state


def _model_parameter_state(
    model: nn.Module,
    groups: StageCParameterGroups | MatchedDenseParameterGroup,
) -> tuple[_ParameterSnapshot, ...]:
    """Clone only A/T/R bytes; cover frozen heavy Parameters by exact metadata."""

    snapshots = []
    seen: set[int] = set()
    selected = {id(parameter) for parameter in groups.all}
    for name, parameter in _named_parameters_with_aliases(model):
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.layout != torch.strided:
            raise StageCInvalidImplementationError(
                f"Stage C cannot snapshot non-strided Parameter {name}"
            )
        storage = parameter.untyped_storage()
        snapshots.append(
            _ParameterSnapshot(
                name=name,
                reference=parameter,
                value=(
                    parameter.detach().clone(memory_format=torch.preserve_format)
                    if id(parameter) in selected
                    else None
                ),
                version=int(parameter._version),
                dtype=parameter.dtype,
                device=parameter.device,
                shape=tuple(parameter.shape),
                stride=tuple(parameter.stride()),
                storage_offset=int(parameter.storage_offset()),
                storage_data_ptr=int(storage.data_ptr()),
                storage_nbytes=int(storage.nbytes()),
                tensor_nbytes=int(parameter.numel() * parameter.element_size()),
                requires_grad=bool(parameter.requires_grad),
                frozen_logical_sha256=(
                    None
                    if id(parameter) in selected
                    else _parameter_logical_sha256(parameter)
                ),
            )
        )
    if len(snapshots) != len(tuple(model.parameters())):
        raise StageCInvalidImplementationError("model Parameter identity registry is inconsistent")
    return tuple(snapshots)


def _parameter_logical_sha256(parameter: nn.Parameter) -> str:
    """Hash logical tensor bytes plus the exact strided tensor interpretation."""

    if parameter.layout != torch.strided:
        raise StageCInvalidImplementationError(
            "Stage C frozen-Parameter hashing requires a strided Tensor"
        )
    storage = parameter.untyped_storage()
    digest = hashlib.sha256()
    digest.update(b"chronotransport-stage-c-frozen-parameter-v1\0")
    metadata = (
        str(parameter.dtype),
        tuple(parameter.shape),
        tuple(parameter.stride()),
        int(parameter.storage_offset()),
        int(storage.nbytes()),
        int(parameter.numel() * parameter.element_size()),
    )
    digest.update(repr(metadata).encode("utf-8"))
    digest.update(b"\0")
    logical_bytes = (
        parameter.detach()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes(order="C")
    )
    digest.update(logical_bytes)
    return digest.hexdigest()


def stage_c_parameter_snapshot_stats(
    model: nn.Module, groups: StageCParameterGroups
) -> dict[str, int]:
    """Report the retry snapshot byte budget without cloning frozen heavy weights."""

    snapshots = _model_parameter_state(model, groups)
    return {
        "cloned_parameter_bytes": sum(
            snapshot.tensor_nbytes for snapshot in snapshots if snapshot.value is not None
        ),
        "frozen_parameter_bytes_cloned": 0,
        "frozen_parameter_bytes_covered_by_metadata": sum(
            snapshot.tensor_nbytes for snapshot in snapshots if snapshot.value is None
        ),
        "trainable_parameter_count": sum(
            snapshot.value is not None for snapshot in snapshots
        ),
        "frozen_parameter_count": sum(
            snapshot.value is None for snapshot in snapshots
        ),
    }


def _assert_parameters_unchanged(snapshots: Sequence[_ParameterSnapshot], model: nn.Module) -> None:
    current = tuple(model.parameters())
    if len(current) != len(snapshots) or {id(item) for item in current} != {
        id(snapshot.reference) for snapshot in snapshots
    }:
        raise StageCInvalidImplementationError("Parameter identity mutation during overflow attempt")
    for snapshot in snapshots:
        parameter = snapshot.reference
        storage = parameter.untyped_storage()
        if (
            int(parameter._version) != snapshot.version
            or parameter.dtype != snapshot.dtype
            or parameter.device != snapshot.device
            or tuple(parameter.shape) != snapshot.shape
            or tuple(parameter.stride()) != snapshot.stride
            or int(parameter.storage_offset()) != snapshot.storage_offset
            or int(storage.data_ptr()) != snapshot.storage_data_ptr
            or int(storage.nbytes()) != snapshot.storage_nbytes
            or bool(parameter.requires_grad) != snapshot.requires_grad
            or (
                snapshot.value is not None
                and not torch.equal(parameter.detach(), snapshot.value)
            )
            or (
                snapshot.frozen_logical_sha256 is not None
                and _parameter_logical_sha256(parameter)
                != snapshot.frozen_logical_sha256
            )
        ):
            raise StageCInvalidImplementationError(
                f"Parameter value/version/metadata/bytes mutation during overflow attempt: {snapshot.name}"
            )


def _assert_success_parameter_transition(
    snapshots: Sequence[_ParameterSnapshot],
    model: nn.Module,
    groups: StageCParameterGroups | MatchedDenseParameterGroup,
    *,
    allow_unchanged: frozenset[int] = frozenset(),
) -> None:
    """Allow optimizer changes only for A/T/R while freezing full-model topology."""

    current = tuple(model.parameters())
    if len(current) != len(snapshots) or {id(item) for item in current} != {
        id(snapshot.reference) for snapshot in snapshots
    }:
        raise StageCInvalidImplementationError(
            "success Parameter topology/identity changed outside the Stage-C optimizer"
        )
    selected = {id(parameter) for parameter in groups.all}
    for snapshot in snapshots:
        parameter = snapshot.reference
        storage = parameter.untyped_storage()
        metadata_changed = (
            parameter.dtype != snapshot.dtype
            or parameter.device != snapshot.device
            or tuple(parameter.shape) != snapshot.shape
            or tuple(parameter.stride()) != snapshot.stride
            or int(parameter.storage_offset()) != snapshot.storage_offset
            or int(storage.data_ptr()) != snapshot.storage_data_ptr
            or int(storage.nbytes()) != snapshot.storage_nbytes
            or bool(parameter.requires_grad) != snapshot.requires_grad
        )
        if metadata_changed:
            raise StageCInvalidImplementationError(
                f"success Parameter metadata/storage mutation: {snapshot.name}"
            )
        if id(parameter) not in selected:
            if (
                int(parameter._version) != snapshot.version
                or snapshot.frozen_logical_sha256 is None
                or _parameter_logical_sha256(parameter)
                != snapshot.frozen_logical_sha256
            ):
                raise StageCInvalidImplementationError(
                    f"frozen Parameter bytes changed during successful Stage-C step: {snapshot.name}"
                )
            continue
        if int(parameter._version) <= snapshot.version:
            if (
                id(parameter) in allow_unchanged
                and snapshot.value is not None
                and torch.equal(parameter.detach(), snapshot.value)
            ):
                continue
            raise StageCInvalidImplementationError(
                f"trainable A/T/R Parameter did not receive the optimizer step: {snapshot.name}"
            )
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            raise StageCInvalidImplementationError(
                f"trainable A/T/R Parameter became non-finite: {snapshot.name}"
            )


def _restore_module_buffers(
    model: nn.Module, state: Mapping[str, Mapping[str, _BufferSnapshot]]
) -> None:
    modules = dict(model.named_modules())
    if set(modules) != set(state):
        raise StageCInvalidImplementationError("model module topology changed during Stage-C attempt")
    for path, module in modules.items():
        snapshots = state[path]
        restored: dict[str, Tensor | None] = {}
        non_persistent: set[str] = set()
        for name, snapshot in snapshots.items():
            if not snapshot.persistent:
                non_persistent.add(name)
            reference = snapshot.reference
            if reference is None:
                restored[name] = None
                continue
            storage = reference.untyped_storage()
            metadata = (
                reference.layout,
                reference.dtype,
                reference.device,
                tuple(reference.shape),
                tuple(reference.stride()),
                int(reference.storage_offset()),
                int(storage._cdata),
                int(storage.data_ptr()),
                int(storage.nbytes()),
                bool(reference.requires_grad),
            )
            expected = (
                snapshot.layout,
                snapshot.dtype,
                snapshot.device,
                snapshot.shape,
                snapshot.stride,
                snapshot.storage_offset,
                snapshot.storage_cdata,
                snapshot.storage_data_ptr,
                snapshot.storage_nbytes,
                snapshot.requires_grad,
            )
            if metadata != expected:
                raise StageCInvalidImplementationError(
                    f"buffer metadata mutation cannot be restored exactly: {path or '<root>'}.{name}"
                )
            if int(reference._version) != snapshot.version:
                raise StageCInvalidImplementationError(
                    f"buffer in-place version mutation cannot be restored exactly: "
                    f"{path or '<root>'}.{name}"
                )
            if not torch.equal(reference.detach(), snapshot.value):
                raise StageCInvalidImplementationError(
                    f"buffer value mutation without a version change cannot be restored exactly: "
                    f"{path or '<root>'}.{name}"
                )
            restored[name] = reference
        module._buffers.clear()
        module._buffers.update(restored)
        module._non_persistent_buffers_set.clear()
        module._non_persistent_buffers_set.update(non_persistent)


def _buffer_state_equal(
    expected: Mapping[str, Mapping[str, _BufferSnapshot]], model: nn.Module
) -> bool:
    current = _module_buffer_state(model)
    if set(expected) != set(current):
        return False
    for path, buffers in expected.items():
        if set(buffers) != set(current[path]):
            return False
        for name, snapshot in buffers.items():
            actual = current[path][name]
            if snapshot.reference is not actual.reference or snapshot.persistent != actual.persistent:
                return False
            if snapshot.reference is None:
                continue
            if (
                snapshot.dtype != actual.dtype
                or snapshot.layout != actual.layout
                or snapshot.version != actual.version
                or snapshot.device != actual.device
                or snapshot.shape != actual.shape
                or snapshot.stride != actual.stride
                or snapshot.storage_offset != actual.storage_offset
                or snapshot.storage_cdata != actual.storage_cdata
                or snapshot.storage_data_ptr != actual.storage_data_ptr
                or snapshot.storage_nbytes != actual.storage_nbytes
                or snapshot.requires_grad != actual.requires_grad
                or not torch.equal(snapshot.value, actual.value)
            ):
                return False
    return True


def _buffer_full_path(module_path: str, name: str) -> str:
    return f"{module_path + '.' if module_path else ''}{name}"


def _transactional_restore_module_buffers(
    model: nn.Module,
    state: Mapping[str, Mapping[str, _BufferSnapshot]],
) -> None:
    """Restore logical buffer bytes while preserving registered objects/storage.

    PyTorch Tensor version counters cannot be decremented.  A3 therefore binds
    rollback to object identity, storage metadata, dtype/layout/shape and exact
    logical value, then allows the monotonically increasing internal version.
    """

    modules = dict(model.named_modules())
    if set(modules) != set(state):
        raise StageCInvalidImplementationError(
            "model module topology changed during Stage-C buffer rollback"
        )
    with torch.no_grad():
        for path, module in modules.items():
            snapshots = state[path]
            if set(module._buffers) != set(snapshots):
                raise StageCInvalidImplementationError(
                    f"buffer registration changed during Stage-C rollback: {path or '<root>'}"
                )
            for name, snapshot in snapshots.items():
                current = module._buffers[name]
                if snapshot.reference is None:
                    if current is not None:
                        raise StageCInvalidImplementationError(
                            f"None buffer changed during Stage-C rollback: {_buffer_full_path(path, name)}"
                        )
                    continue
                if current is not snapshot.reference:
                    raise StageCInvalidImplementationError(
                        f"buffer identity changed during Stage-C rollback: {_buffer_full_path(path, name)}"
                    )
                storage = current.untyped_storage()
                metadata = (
                    current.layout,
                    current.dtype,
                    current.device,
                    tuple(current.shape),
                    tuple(current.stride()),
                    int(current.storage_offset()),
                    int(storage._cdata),
                    int(storage.data_ptr()),
                    int(storage.nbytes()),
                    bool(current.requires_grad),
                    name not in module._non_persistent_buffers_set,
                )
                expected = (
                    snapshot.layout,
                    snapshot.dtype,
                    snapshot.device,
                    snapshot.shape,
                    snapshot.stride,
                    snapshot.storage_offset,
                    snapshot.storage_cdata,
                    snapshot.storage_data_ptr,
                    snapshot.storage_nbytes,
                    snapshot.requires_grad,
                    snapshot.persistent,
                )
                if metadata != expected:
                    raise StageCInvalidImplementationError(
                        f"buffer metadata changed during Stage-C rollback: {_buffer_full_path(path, name)}"
                    )
                if not torch.equal(current.detach(), snapshot.value):
                    current.copy_(snapshot.value)


def _buffer_logical_state_equal(
    expected: Mapping[str, Mapping[str, _BufferSnapshot]],
    model: nn.Module,
    *,
    allowed_value_changes: frozenset[str] = frozenset(),
) -> bool:
    current = _module_buffer_state(model)
    if set(expected) != set(current):
        return False
    for path, buffers in expected.items():
        if set(buffers) != set(current[path]):
            return False
        for name, snapshot in buffers.items():
            actual = current[path][name]
            full_path = _buffer_full_path(path, name)
            if snapshot.reference is not actual.reference or snapshot.persistent != actual.persistent:
                return False
            if snapshot.reference is None:
                continue
            if (
                snapshot.dtype != actual.dtype
                or snapshot.layout != actual.layout
                or snapshot.device != actual.device
                or snapshot.shape != actual.shape
                or snapshot.stride != actual.stride
                or snapshot.storage_offset != actual.storage_offset
                or snapshot.storage_cdata != actual.storage_cdata
                or snapshot.storage_data_ptr != actual.storage_data_ptr
                or snapshot.storage_nbytes != actual.storage_nbytes
                or snapshot.requires_grad != actual.requires_grad
            ):
                return False
            if full_path not in allowed_value_changes and not torch.equal(
                snapshot.value, actual.value
            ):
                return False
    return True


def _canonical_loss_normalizer(model: nn.Module) -> tuple[str, Tensor]:
    matches = []
    for path, module in model.named_modules():
        if path.endswith("rpn_head") and "loss_normalizer" in module._buffers:
            value = module._buffers["loss_normalizer"]
            if isinstance(value, Tensor):
                matches.append((_buffer_full_path(path, "loss_normalizer"), value))
    if len(matches) != 1 or matches[0][1].numel() != 1:
        raise StageCInvalidImplementationError(
            "formal Stage C requires exactly one scalar rpn_head.loss_normalizer buffer"
        )
    return matches[0]


_MODULE_INTERNAL_ATTRIBUTES = (frozenset(nn.Module().__dict__) - {"training"}) | {
    "_parameters",
    "_buffers",
    "_modules",
}

_REQUIRED_ROLLBACK_OBJECTS = frozenset(
    {
        "ema",
        "scheduler",
        "diagnostics",
        "profiler",
        "sampler",
        "successful_cursor",
        "exposure_cursor",
        "shadow_ledger",
    }
)


def _snapshot_python_value(
    value: Any, memo: dict[int, Any] | None = None
) -> Any:
    if memo is None:
        memo = {}
    external_reference = isinstance(value, (nn.Module, torch.optim.Optimizer)) or callable(value)
    graph_managed = (
        isinstance(value, (Tensor, np.ndarray, dict, list, tuple, set))
        or (hasattr(value, "__dict__") and not external_reference)
    )
    if graph_managed:
        if id(value) in memo:
            return (
                "graph_ref",
                {
                    "reference": value,
                    "reference_id": id(value),
                    "type": type(value),
                },
            )
        memo[id(value)] = value
    if isinstance(value, Tensor):
        if value.layout != torch.strided:
            raise StageCInvalidImplementationError(
                "Stage C cannot snapshot a non-strided Python Tensor state"
            )
        storage = value.untyped_storage()
        return (
            "tensor_ref",
            {
                "reference": value,
                "reference_id": id(value),
                "value": value.detach().clone(memory_format=torch.preserve_format),
                "version": int(value._version),
                "layout": value.layout,
                "dtype": value.dtype,
                "device": value.device,
                "shape": tuple(value.shape),
                "stride": tuple(value.stride()),
                "storage_offset": int(value.storage_offset()),
                "storage_cdata": int(storage._cdata),
                "storage_data_ptr": int(storage.data_ptr()),
                "storage_nbytes": int(storage.nbytes()),
                "requires_grad": bool(value.requires_grad),
                "is_leaf": bool(value.is_leaf),
                "grad_fn_type": None if value.grad_fn is None else type(value.grad_fn).__qualname__,
            },
        )
    if isinstance(value, np.ndarray):
        return (
            "ndarray_ref",
            {
                "reference": value,
                "reference_id": id(value),
                "dtype": value.dtype,
                "shape": value.shape,
                "value": value.copy(),
            },
        )
    if isinstance(value, dict):
        return (
            "dict_ref",
            {
                "reference": value,
                "reference_id": id(value),
                "items": [
                    (_snapshot_python_value(key, memo), _snapshot_python_value(item, memo))
                    for key, item in value.items()
                ],
            },
        )
    if isinstance(value, list):
        return (
            "list_ref",
            {
                "reference": value,
                "reference_id": id(value),
                "items": [_snapshot_python_value(item, memo) for item in value],
            },
        )
    if isinstance(value, tuple):
        return (
            "tuple_ref",
            {
                "reference": value,
                "reference_id": id(value),
                "items": [_snapshot_python_value(item, memo) for item in value],
            },
        )
    if isinstance(value, set):
        return (
            "set_ref",
            {
                "reference": value,
                "reference_id": id(value),
                "items": [_snapshot_python_value(item, memo) for item in value],
            },
        )
    if external_reference:
        return ("reference", value)
    if hasattr(value, "__dict__"):
        return (
            "object_ref",
            {
                "reference": value,
                "reference_id": id(value),
                "type": type(value),
                "state": _python_attribute_state(value, memo=memo),
            },
        )
    try:
        return ("value", copy.deepcopy(value))
    except Exception as error:  # pragma: no cover - caller-specific object
        raise StageCInvalidImplementationError(
            f"cannot snapshot Python value of type {type(value).__name__}"
        ) from error


def _restore_python_value(snapshot: Any) -> Any:
    kind, value = snapshot
    if kind == "graph_ref":
        reference = value["reference"]
        if id(reference) != value["reference_id"] or type(reference) is not value["type"]:
            raise StageCInvalidImplementationError("Python object graph identity/type changed")
        return reference
    if kind == "tensor_ref":
        reference = value["reference"]
        if id(reference) != value["reference_id"]:
            raise StageCInvalidImplementationError("Python Tensor reference identity changed")
        if reference.layout != torch.strided:
            raise StageCInvalidImplementationError(
                "Python Tensor layout changed; exact rollback is impossible"
            )
        storage = reference.untyped_storage()
        if int(reference._version) != value["version"]:
            raise StageCInvalidImplementationError(
                "in-place Python Tensor version mutation cannot be restored without breaking autograd"
            )
        metadata = (
            reference.layout,
            reference.dtype,
            reference.device,
            tuple(reference.shape),
            tuple(reference.stride()),
            int(reference.storage_offset()),
            int(storage._cdata),
            int(storage.data_ptr()),
            int(storage.nbytes()),
            bool(reference.requires_grad),
            bool(reference.is_leaf),
            None if reference.grad_fn is None else type(reference.grad_fn).__qualname__,
        )
        expected = (
            value["layout"],
            value["dtype"],
            value["device"],
            value["shape"],
            value["stride"],
            value["storage_offset"],
            value["storage_cdata"],
            value["storage_data_ptr"],
            value["storage_nbytes"],
            value["requires_grad"],
            value["is_leaf"],
            value["grad_fn_type"],
        )
        if metadata != expected:
            raise StageCInvalidImplementationError(
                "Python Tensor layout/storage/metadata/alias identity changed; "
                "exact rollback is impossible"
            )
        if not torch.equal(reference.detach(), value["value"]):
            raise StageCInvalidImplementationError(
                "Python Tensor value/autograd mutation cannot be restored exactly"
            )
        return reference
    if kind == "ndarray_ref":
        reference = value["reference"]
        if (
            id(reference) != value["reference_id"]
            or reference.dtype != value["dtype"]
            or reference.shape != value["shape"]
        ):
            raise StageCInvalidImplementationError("NumPy array identity/metadata changed")
        np.copyto(reference, value["value"])
        return reference
    if kind == "dict_ref":
        reference = value["reference"]
        if id(reference) != value["reference_id"]:
            raise StageCInvalidImplementationError("Python dict identity changed")
        reference.clear()
        for key, item in value["items"]:
            reference[_restore_python_value(key)] = _restore_python_value(item)
        return reference
    if kind == "list_ref":
        reference = value["reference"]
        if id(reference) != value["reference_id"]:
            raise StageCInvalidImplementationError("Python list identity changed")
        reference.clear()
        reference.extend(_restore_python_value(item) for item in value["items"])
        return reference
    if kind == "tuple_ref":
        reference = value["reference"]
        if id(reference) != value["reference_id"] or len(reference) != len(value["items"]):
            raise StageCInvalidImplementationError("Python tuple identity/length changed")
        for item in value["items"]:
            _restore_python_value(item)
        return reference
    if kind == "set_ref":
        reference = value["reference"]
        if id(reference) != value["reference_id"]:
            raise StageCInvalidImplementationError("Python set identity changed")
        reference.clear()
        reference.update(_restore_python_value(item) for item in value["items"])
        return reference
    if kind == "reference":
        return value
    if kind == "object_ref":
        reference = value["reference"]
        if id(reference) != value["reference_id"] or type(reference) is not value["type"]:
            raise StageCInvalidImplementationError("Python object reference identity/type changed")
        current = set(reference.__dict__)
        expected = set(value["state"])
        for name in current - expected:
            object.__delattr__(reference, name)
        for name, item in value["state"].items():
            object.__setattr__(reference, name, _restore_python_value(item))
        return reference
    if kind == "value":
        return copy.deepcopy(value)
    raise AssertionError(kind)  # pragma: no cover - internal invariant


def _python_attribute_state(
    value: Any,
    *,
    module: bool = False,
    memo: dict[int, Any] | None = None,
    ignored: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not hasattr(value, "__dict__"):
        return {}
    if memo is None:
        memo = {}
    return {
        name: _snapshot_python_value(item, memo)
        for name, item in sorted(value.__dict__.items())
        if name not in ignored
        and (not module or name not in _MODULE_INTERNAL_ATTRIBUTES)
    }


def _restore_python_attributes(value: Any, state: Mapping[str, Any], *, module: bool = False) -> None:
    current = {
        name
        for name in value.__dict__
        if not module or name not in _MODULE_INTERNAL_ATTRIBUTES
    }
    for name in current - set(state):
        delattr(value, name)
    for name, item in state.items():
        setattr(value, name, _restore_python_value(item))


def _module_python_state(
    model: nn.Module,
    *,
    ignored_by_path: Mapping[str, frozenset[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    memo: dict[int, Any] = {}
    ignored_by_path = ignored_by_path or {}
    for path, module in _named_modules_with_aliases(model):
        state[path] = _python_attribute_state(
            module,
            module=True,
            memo=memo,
            ignored=ignored_by_path.get(path, frozenset()),
        )
    return state


def _approved_success_python_attributes(
    runtime_path: str,
) -> dict[str, frozenset[str]]:
    return {
        runtime_path: frozenset(
            {
            "latest_schedule",
            "latest_summary",
            "latest_output",
            "latest_signals",
            }
        ),
        "": frozenset({"latest_chronotransport_summary"}),
    }


def _assert_success_python_state(
    expected: Mapping[str, Mapping[str, Any]],
    model: nn.Module,
    *,
    runtime_path: str,
) -> None:
    current = _module_python_state(
        model,
        ignored_by_path=_approved_success_python_attributes(runtime_path),
    )
    if not _state_equal(expected, current):
        mismatch = _first_state_mismatch(expected, current)
        raise StageCInvalidImplementationError(
            f"unapproved model Python state mutation during successful Stage-C attempt: {mismatch}"
        )


def _restore_module_python_state(model: nn.Module, state: Mapping[str, Mapping[str, Any]]) -> None:
    modules = dict(_named_modules_with_aliases(model))
    if set(modules) != set(state):
        raise StageCInvalidImplementationError("model module topology changed during Stage-C attempt")
    for path, module in modules.items():
        _restore_python_attributes(module, state[path], module=True)


def _clone_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(state))


def _snapshot_object(value: Any) -> tuple[str, Any]:
    if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
        return "stateful", {
            "state_dict": _clone_state_dict(value.state_dict()),
            "python": _python_attribute_state(value, module=isinstance(value, nn.Module)),
        }
    if isinstance(value, dict):
        return "dict", _snapshot_python_value(value)
    if isinstance(value, list):
        return "list", _snapshot_python_value(value)
    if hasattr(value, "__dict__"):
        return "object", _python_attribute_state(value)
    raise StageCInvalidImplementationError(
        f"rollback object {type(value).__name__} has no restorable state"
    )


def _restore_object(value: Any, snapshot: tuple[str, Any]) -> None:
    kind, state = snapshot
    if kind == "stateful":
        value.load_state_dict(copy.deepcopy(state["state_dict"]))
        _restore_python_attributes(value, state["python"], module=isinstance(value, nn.Module))
    elif kind == "dict":
        if _restore_python_value(state) is not value:
            raise StageCInvalidImplementationError("rollback dict identity changed")
    elif kind == "list":
        if _restore_python_value(state) is not value:
            raise StageCInvalidImplementationError("rollback list identity changed")
    elif kind == "object":
        _restore_python_attributes(value, state)
    else:  # pragma: no cover - internal invariant
        raise AssertionError(kind)


def _state_equal(first: Any, second: Any) -> bool:
    if first is second:
        return True
    if isinstance(first, Tensor) and isinstance(second, Tensor):
        return first.dtype == second.dtype and first.shape == second.shape and torch.equal(first, second)
    if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
        return first.dtype == second.dtype and first.shape == second.shape and np.array_equal(first, second)
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return set(first) == set(second) and all(_state_equal(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)) and isinstance(second, type(first)):
        return len(first) == len(second) and all(_state_equal(a, b) for a, b in zip(first, second))
    return bool(first == second)


def _first_state_mismatch(first: Any, second: Any, path: str = "<root>") -> str:
    if isinstance(first, Tensor) and isinstance(second, Tensor):
        return path
    if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
        return path
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        if set(first) != set(second):
            missing = sorted(str(key) for key in set(first) - set(second))
            extra = sorted(str(key) for key in set(second) - set(first))
            return f"{path} keys(missing={missing}, extra={extra})"
        for key in first:
            if not _state_equal(first[key], second[key]):
                return _first_state_mismatch(first[key], second[key], f"{path}.{key}")
        return path
    if isinstance(first, (list, tuple)) and isinstance(second, type(first)):
        if len(first) != len(second):
            return f"{path} length({len(first)} != {len(second)})"
        for index, (left, right) in enumerate(zip(first, second)):
            if not _state_equal(left, right):
                return _first_state_mismatch(left, right, f"{path}[{index}]")
        return path
    return path


def _rng_equal(first: _RNGSnapshot, second: _RNGSnapshot) -> bool:
    return (
        _state_equal(first.python, second.python)
        and _state_equal(first.numpy, second.numpy)
        and _state_equal(first.torch_cpu, second.torch_cpu)
        and _state_equal(first.torch_cuda, second.torch_cuda)
    )


def _counter_value(value: Any, name: str) -> int:
    current = getattr(value, "value", None)
    if isinstance(current, bool) or not isinstance(current, int) or current < 0:
        raise StageCInvalidImplementationError(f"{name}.value must be a non-negative integer")
    return current


def _canonical_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


_SUCCESS_CONTROL_OBJECTS = (
    "ema",
    "scheduler",
    "sampler",
    "successful_cursor",
    "exposure_cursor",
    "shadow_ledger",
)


def _expected_ledger_exposures(
    *, seed: int, successful_update: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_rows = stage_c_batch_exposures(seed, successful_update)
    rows = []
    for row in protocol_rows:
        batch_position = row["batch_position"]
        candidate_ordinal = row["candidate"]
        rows.append(
            {
                "batch_position": batch_position,
                "window_exposure_ordinal": row["window_exposure_ordinal"],
                "candidate_ordinal": candidate_ordinal,
                "candidate_id": R2_NON_DENSE_NAMES[candidate_ordinal],
                "actual_action_sha256": stage_c_action_hash(
                    seed=seed,
                    successful_update=successful_update,
                    batch_position=batch_position,
                    action_payload=_R2_STAGE_C_ACTIONS[candidate_ordinal],
                ),
            }
        )
    return rows[0], rows[1]


def _validate_global_success_state(
    *, objects: Mapping[str, Any], lr_scheduler: Any, seed: int
) -> int:
    """Require one coherent successful-update trace across every state surface."""

    successful = _counter_value(objects["successful_cursor"], "successful_cursor")
    ema_count = getattr(objects["ema"], "stage_c_update_count", None)
    sampler = _counter_value(objects["sampler"], "sampler")
    exposure = _counter_value(objects["exposure_cursor"], "exposure_cursor")
    ledger = objects["shadow_ledger"]
    if not isinstance(ledger, MutableSequence):
        raise StageCInvalidImplementationError("shadow_ledger must be an appendable sequence")
    coherent = (
        not isinstance(ema_count, bool)
        and isinstance(ema_count, int)
        and ema_count == successful
        and int(lr_scheduler.last_epoch) == successful
        and int(lr_scheduler._step_count) == successful + 1
        and sampler == successful
        and exposure == 2 * successful
        and len(ledger) == successful
    )
    if not coherent:
        raise StageCInvalidImplementationError(
            "global success-state coherence requires "
            "EMA == scheduler epoch == sampler == successful == ledger length, "
            "scheduler step_count == successful + 1, and exposure == 2 * successful"
        )
    for index, row in enumerate(ledger):
        expected_exposures = _expected_ledger_exposures(
            seed=seed, successful_update=index
        )
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "successful_update",
                "seed",
                "exposure_start",
                "exposure_stop",
                "batch_hash",
                "action_batch_sha256",
                "exposures",
            }
            or row.get("successful_update") != index
            or row.get("seed") != seed
            or row.get("exposure_start") != 2 * index
            or row.get("exposure_stop") != 2 * index + 2
            or not _canonical_sha256(row.get("batch_hash"))
            or not _canonical_sha256(row.get("action_batch_sha256"))
            or not isinstance(row.get("exposures"), list)
            or row["exposures"] != list(expected_exposures)
            or row["action_batch_sha256"] != _action_batch_sha256(expected_exposures)
        ):
            raise StageCInvalidImplementationError(
                f"shadow ledger continuity is invalid at successful update {index}"
            )
    return successful


def _assert_control_objects_at_common_start(
    *, objects: Mapping[str, Any], snapshots: Mapping[str, tuple[str, Any]]
) -> None:
    for name in _SUCCESS_CONTROL_OBJECTS:
        current = _snapshot_object(objects[name])
        expected = snapshots[name]
        if current[0] != expected[0] or not _state_equal(current[1], expected[1]):
            raise StageCInvalidImplementationError(
                f"{name} left the common starting state before the Stage-C primitive advanced it"
            )


def _advance_success_state(
    *,
    objects: Mapping[str, Any],
    model: nn.Module,
    lr_scheduler: Any,
    seed: int,
    batch_hash: str,
    action_batch_sha256: str,
    exposures: Sequence[Mapping[str, Any]],
) -> None:
    ema = objects["ema"]
    ema_count = getattr(ema, "stage_c_update_count", None)
    if isinstance(ema_count, bool) or not isinstance(ema_count, int) or ema_count < 0:
        raise StageCInvalidImplementationError("EMA requires a stage_c_update_count contract")
    if not callable(getattr(ema, "update", None)):
        raise StageCInvalidImplementationError("EMA requires update(model)")
    scheduler_epoch = int(lr_scheduler.last_epoch)
    scheduler_steps = int(lr_scheduler._step_count)
    sampler_value = _counter_value(objects["sampler"], "sampler")
    successful_value = _counter_value(objects["successful_cursor"], "successful_cursor")
    exposure_value = _counter_value(objects["exposure_cursor"], "exposure_cursor")
    ledger = objects["shadow_ledger"]
    if not isinstance(ledger, MutableSequence):
        raise StageCInvalidImplementationError("shadow_ledger must be an appendable sequence")
    ledger_length = len(ledger)

    ema.update(model)
    lr_scheduler.step()
    objects["sampler"].value = sampler_value + 1
    objects["successful_cursor"].value = successful_value + 1
    objects["exposure_cursor"].value = exposure_value + 2
    ledger.append(
        {
            "successful_update": successful_value,
            "seed": seed,
            "exposure_start": exposure_value,
            "exposure_stop": exposure_value + 2,
            "batch_hash": batch_hash,
            "action_batch_sha256": action_batch_sha256,
            "exposures": [dict(row) for row in exposures],
        }
    )

    if getattr(ema, "stage_c_update_count", None) != ema_count + 1:
        raise StageCInvalidImplementationError("EMA must advance exactly once per successful update")
    if int(lr_scheduler.last_epoch) != scheduler_epoch + 1 or int(lr_scheduler._step_count) != scheduler_steps + 1:
        raise StageCInvalidImplementationError("scheduler must advance exactly once per successful update")
    if _counter_value(objects["sampler"], "sampler") != sampler_value + 1:
        raise StageCInvalidImplementationError("sampler must advance exactly once per successful update")
    if _counter_value(objects["successful_cursor"], "successful_cursor") != successful_value + 1:
        raise StageCInvalidImplementationError("successful cursor must advance exactly once")
    if _counter_value(objects["exposure_cursor"], "exposure_cursor") != exposure_value + 2:
        raise StageCInvalidImplementationError("exposure cursor must advance by the batch size two")
    if len(ledger) != ledger_length + 1:
        raise StageCInvalidImplementationError("shadow ledger must append exactly one successful batch row")


_STAGE_C_BATCH_FORWARD_FIELDS = frozenset(
    {"inputs", "masks", "metas", "gt_segments", "gt_labels"}
)
_STAGE_C_BATCH_METADATA_FIELDS = frozenset(
    {
        "video_id",
        "window_id",
        "manifest_window_sha256",
        "manifest_sampled_frame_indices_sha256",
        "augmentation_sha256",
        "sample_id",
        "split",
    }
)


@dataclass(frozen=True)
class _StageCPairedForwardEvidence:
    losses: StageCAttemptLosses
    runtime: _StageCRuntimeEvidence
    dense_task_loss: Tensor
    counterfactual_task_loss: Tensor
    normalizer_before: Tensor
    normalizer_after_dense: Tensor
    normalizer_after_counterfactual: Tensor
    model_forward_count: int
    runtime_forward_count: int
    risk_forward_count: int


@dataclass(frozen=True)
class _MatchedDenseForwardEvidence:
    detector_loss: Tensor
    per_window_task_loss: Tensor
    detector_boundary: _TensorBoundarySnapshot
    normalizer_before: Tensor
    normalizer_after: Tensor
    model_forward_count: int
    runtime_forward_count: int
    risk_forward_count: int


def _stage_c_forward_kwargs(materialized_batch: Any) -> dict[str, Any]:
    if not isinstance(materialized_batch, Mapping):
        raise TypeError("formal Stage-C materialized batch must be a mapping")
    missing = sorted(_STAGE_C_BATCH_FORWARD_FIELDS - set(materialized_batch))
    unknown = sorted(
        set(materialized_batch)
        - _STAGE_C_BATCH_FORWARD_FIELDS
        - _STAGE_C_BATCH_METADATA_FIELDS
    )
    if missing or unknown:
        raise ValueError(
            f"formal Stage-C batch fields mismatch: missing={missing}, unknown={unknown}"
        )
    return {
        key: materialized_batch[key] for key in _STAGE_C_BATCH_FORWARD_FIELDS
    } | {
        "return_loss": True,
        "chronotransport_per_window_output": True,
    }


def _run_stage_c_paired_actionformer_forward(
    *,
    materialized_batch: Any,
    model: nn.Module,
    runtime: nn.Module,
    expected_actions: Tensor,
    buffer_state: Mapping[str, Mapping[str, _BufferSnapshot]],
    python_state: Mapping[str, Mapping[str, Any]],
    rng_state: _RNGSnapshot,
    topology_state: _ModelTopologySnapshot,
    require_cuda_autocast: bool,
) -> _StageCPairedForwardEvidence:
    from ..detectors.actionformer import ActionFormerPerWindowTrainOutput

    forward_kwargs = _stage_c_forward_kwargs(materialized_batch)
    batch_size = int(forward_kwargs["inputs"].shape[0])
    if batch_size != 2:
        raise StageCInvalidImplementationError(
            "formal Stage C requires exact global batch size two"
        )
    normalizer_path, normalizer = _canonical_loss_normalizer(model)
    normalizer_before = normalizer.detach().clone()
    state: dict[str, Any] = {
        "model_forwards": 0,
        "runtime_forwards": 0,
        "risk_forwards": 0,
        "risk_inputs": None,
    }

    def model_pre_hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
        del module, inputs
        index = int(state["model_forwards"])
        if index >= 2:
            raise StageCInvalidImplementationError(
                "Stage-C attempt executed more than two top-level model forwards"
            )
        expected_grad = index == 1
        if torch.is_grad_enabled() != expected_grad:
            raise StageCInvalidImplementationError(
                "Stage-C dense/CF model grad modes differ from A4"
            )
        if require_cuda_autocast and not torch.is_autocast_enabled():
            raise StageCInvalidImplementationError(
                "formal CUDA Stage C requires autocast for both model forwards"
            )
        state["model_forwards"] = index + 1

    def model_forward_hook(
        module: nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> None:
        del module, inputs
        if not isinstance(output, ActionFormerPerWindowTrainOutput):
            raise StageCInvalidImplementationError(
                "Stage-C model forward must return ActionFormer per-window evidence"
            )

    def runtime_pre_hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
        del module, inputs
        if int(state["runtime_forwards"]) >= int(state["model_forwards"]):
            raise StageCInvalidImplementationError(
                "canonical runtime escaped its audited model forward"
            )
        state["runtime_forwards"] = int(state["runtime_forwards"]) + 1

    def risk_pre_hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
        del module
        if (
            int(state["model_forwards"]) != 2
            or int(state["runtime_forwards"]) != 2
            or int(state["risk_forwards"]) != 0
            or len(inputs) < 2
        ):
            raise StageCInvalidImplementationError(
                "Stage-C risk predictor must run exactly once after the CF model"
            )
        if require_cuda_autocast and not torch.is_autocast_enabled():
            raise StageCInvalidImplementationError(
                "formal CUDA Stage C requires autocast for the risk forward"
            )
        state["risk_forwards"] = 1
        state["risk_inputs"] = inputs[:2]

    handles = (
        model.register_forward_pre_hook(model_pre_hook),
        model.register_forward_hook(model_forward_hook),
        runtime.register_forward_pre_hook(runtime_pre_hook),
        runtime.risk_predictor.register_forward_pre_hook(risk_pre_hook),
    )
    try:
        with torch.no_grad():
            runtime.forced_actions.copy_(
                torch.full_like(expected_actions, int(ChronoAction.RECOMPUTE))
            )
            runtime.forced_action_name = "stage_c_dense_reference"
            runtime.latest_schedule = None
            runtime.latest_summary = None
            runtime.latest_output = None
            runtime.latest_signals = None
            with (
                torch.cuda.amp.autocast()
                if require_cuda_autocast
                else nullcontext()
            ):
                dense_output = model(**forward_kwargs)
        if not isinstance(dense_output, ActionFormerPerWindowTrainOutput):
            raise StageCInvalidImplementationError(
                "dense reference lacks ActionFormer per-window evidence"
            )
        dense_task_loss = dense_output.per_window_task_loss.detach().clone()
        dense_features = dense_output.detector_features.detach().clone()
        normalizer_after_dense = normalizer.detach().clone()

        _assert_model_topology(topology_state, model, where="after dense reference")
        _transactional_restore_module_buffers(model, buffer_state)
        _restore_module_python_state(model, python_state)
        _restore_rng(rng_state)
        if not _buffer_logical_state_equal(buffer_state, model):
            raise StageCInvalidImplementationError(
                "dense-reference buffers were not logically restored before CF"
            )
        if not torch.equal(normalizer.detach(), normalizer_before):
            raise StageCInvalidImplementationError(
                "dense-reference loss_normalizer was not restored before CF"
            )

        with (
            torch.cuda.amp.autocast()
            if require_cuda_autocast
            else nullcontext()
        ):
            counterfactual_output = model(**forward_kwargs)
            if not isinstance(
                counterfactual_output, ActionFormerPerWindowTrainOutput
            ):
                raise StageCInvalidImplementationError(
                    "counterfactual lacks ActionFormer per-window evidence"
                )
            schedule = runtime.latest_schedule
            summary = runtime.latest_summary
            signals = runtime.latest_signals
            executed_actions = getattr(schedule, "actions", None)
            if (
                not isinstance(executed_actions, Tensor)
                or not isinstance(summary, Mapping)
                or not isinstance(signals, Tensor)
            ):
                raise StageCInvalidImplementationError(
                    "counterfactual runtime evidence is incomplete"
                )
            if not torch.equal(
                executed_actions.detach().to(torch.long),
                expected_actions.detach().to(torch.long),
            ):
                raise StageCInvalidImplementationError(
                    "counterfactual executed actions differ from frozen exposure"
                )
            risk_prediction = runtime.risk_predictor(
                signals.detach(), executed_actions.detach().unsqueeze(1)
            ).squeeze(1)
            target = (
                counterfactual_output.per_window_task_loss.detach()
                - dense_task_loss
            ).clamp_min(0.0)
            if tuple(risk_prediction.shape) != (2,) or tuple(target.shape) != (2,):
                raise StageCInvalidImplementationError(
                    "Stage-C risk prediction/target must each have shape [2]"
                )
            quantile = float(runtime.risk_predictor.quantile)
            residual = target - risk_prediction
            risk_loss = torch.maximum(
                quantile * residual, (quantile - 1.0) * residual
            ).mean()
            feature_delta = (
                counterfactual_output.detector_features.float()
                - dense_features.float()
            )
            feature_loss = feature_delta.square().mean()
            losses = StageCAttemptLosses(
                detector_loss=counterfactual_output.loss_dict["cost"],
                feature_loss=feature_loss,
                risk_loss=risk_loss,
            )
        normalizer_after_counterfactual = normalizer.detach().clone()
    finally:
        for handle in handles:
            handle.remove()

    if (
        int(state["model_forwards"]) != 2
        or int(state["runtime_forwards"]) != 2
        or int(state["risk_forwards"]) != 1
    ):
        raise StageCInvalidImplementationError(
            "A4 requires exactly two model/runtime forwards and one risk forward"
        )
    risk_inputs = state["risk_inputs"]
    if not isinstance(risk_inputs, tuple) or len(risk_inputs) < 2:
        raise StageCInvalidImplementationError(
            "risk predictor did not expose the audited signals/actions inputs"
        )
    # ``detach()`` creates a view object, so identity cannot be compared.  The
    # pre-hook instead freezes the exact call arguments and verifies their
    # values against the local CF evidence below.
    if not torch.equal(risk_inputs[0], signals.detach()):
        raise StageCInvalidImplementationError(
            "risk predictor signals differ from the CF runtime signals"
        )
    if not torch.equal(
        risk_inputs[1], executed_actions.detach().unsqueeze(1)
    ):
        raise StageCInvalidImplementationError(
            "risk predictor actions differ from the CF runtime actions"
        )
    if not torch.equal(normalizer_after_dense, normalizer_after_counterfactual):
        raise StageCInvalidImplementationError(
            "dense and CF loss_normalizer formulas diverged on the same batch"
        )
    detector_boundary = _snapshot_tensor_boundary(
        "counterfactual per-window task loss",
        counterfactual_output.per_window_task_loss,
    )
    feature_boundary = _snapshot_tensor_boundary(
        "counterfactual detector features",
        counterfactual_output.detector_features,
    )
    signals_boundary = _snapshot_tensor_boundary(
        "counterfactual runtime signals", signals
    )
    runtime_evidence = _StageCRuntimeEvidence(
        actions=executed_actions.detach().clone(),
        summary=copy.deepcopy(dict(summary)),
        detector_output=counterfactual_output.per_window_task_loss,
        detector_boundary=detector_boundary,
        feature_output=counterfactual_output.detector_features,
        feature_boundary=feature_boundary,
        signals_boundary=signals_boundary,
        risk_output=risk_prediction,
    )
    return _StageCPairedForwardEvidence(
        losses=losses,
        runtime=runtime_evidence,
        dense_task_loss=dense_task_loss,
        counterfactual_task_loss=counterfactual_output.per_window_task_loss,
        normalizer_before=normalizer_before,
        normalizer_after_dense=normalizer_after_dense,
        normalizer_after_counterfactual=normalizer_after_counterfactual,
        model_forward_count=2,
        runtime_forward_count=2,
        risk_forward_count=1,
    )


def _run_matched_dense_actionformer_forward(
    *,
    materialized_batch: Any,
    model: nn.Module,
    runtime: nn.Module,
    require_cuda_autocast: bool,
) -> _MatchedDenseForwardEvidence:
    from ..detectors.actionformer import ActionFormerPerWindowTrainOutput

    forward_kwargs = _stage_c_forward_kwargs(materialized_batch)
    if int(forward_kwargs["inputs"].shape[0]) != 2:
        raise StageCInvalidImplementationError(
            "matched-dense requires exact global batch size two"
        )
    _, normalizer = _canonical_loss_normalizer(model)
    normalizer_before = normalizer.detach().clone()
    state = {"model": 0, "runtime": 0, "risk": 0}

    def model_pre_hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
        del module, inputs
        if state["model"] != 0 or not torch.is_grad_enabled():
            raise StageCInvalidImplementationError(
                "matched-dense requires exactly one differentiable model forward"
            )
        if require_cuda_autocast and not torch.is_autocast_enabled():
            raise StageCInvalidImplementationError(
                "formal CUDA matched-dense requires autocast"
            )
        state["model"] = 1

    def model_forward_hook(
        module: nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> None:
        del module, inputs
        if not isinstance(output, ActionFormerPerWindowTrainOutput):
            raise StageCInvalidImplementationError(
                "matched-dense model must publish ActionFormer per-window evidence"
            )

    def runtime_pre_hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
        del module, inputs
        if state["model"] != 1 or state["runtime"] != 0:
            raise StageCInvalidImplementationError(
                "matched-dense canonical runtime forward count is invalid"
            )
        state["runtime"] = 1

    def risk_pre_hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
        del module, inputs
        state["risk"] += 1
        raise StageCInvalidImplementationError(
            "matched-dense must not execute the risk predictor"
        )

    handles = (
        model.register_forward_pre_hook(model_pre_hook),
        model.register_forward_hook(model_forward_hook),
        runtime.register_forward_pre_hook(runtime_pre_hook),
        runtime.risk_predictor.register_forward_pre_hook(risk_pre_hook),
    )
    try:
        with (
            torch.cuda.amp.autocast()
            if require_cuda_autocast
            else nullcontext()
        ):
            output = model(**forward_kwargs)
    finally:
        for handle in handles:
            handle.remove()
    if not isinstance(output, ActionFormerPerWindowTrainOutput):
        raise StageCInvalidImplementationError(
            "matched-dense forward lacks ActionFormer evidence"
        )
    if state != {"model": 1, "runtime": 1, "risk": 0}:
        raise StageCInvalidImplementationError(
            "matched-dense requires one model/runtime forward and zero risk forwards"
        )
    schedule = runtime.latest_schedule
    summary = runtime.latest_summary
    actions = getattr(schedule, "actions", None)
    if (
        not isinstance(actions, Tensor)
        or tuple(actions.shape) != (2, 48, 3)
        or bool(torch.any(actions != int(ChronoAction.RECOMPUTE)).item())
        or not isinstance(summary, Mapping)
        or summary.get("forced_dense_exact_path") is not True
        or summary.get("schedule_repair_count") != 0
        or summary.get("runtime_fail_closed_repairs") != 0
        or summary.get("cache_reset_per_window") is not True
        or summary.get("dense_output_shape_preserved") is not True
    ):
        raise StageCInvalidImplementationError(
            "matched-dense runtime did not execute the exact forced-dense path"
        )
    boundary = _snapshot_tensor_boundary(
        "matched-dense per-window task loss", output.per_window_task_loss
    )
    return _MatchedDenseForwardEvidence(
        detector_loss=output.loss_dict["cost"],
        per_window_task_loss=output.per_window_task_loss,
        detector_boundary=boundary,
        normalizer_before=normalizer_before,
        normalizer_after=normalizer.detach().clone(),
        model_forward_count=1,
        runtime_forward_count=1,
        risk_forward_count=0,
    )


def run_stage_c_amp_with_retry_for_test_only(
    *,
    materialized_batch: Any,
    attempt: Callable[[], StageCAttemptLosses],
    model: nn.Module,
    groups: StageCParameterGroups,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    lr_scheduler: Any,
    seed: int,
    rollback_objects: Mapping[str, Any] | None = None,
    retry_audit: MutableSequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one successful r2 update, retrying at most three AMP overflows.

    The materialized batch and all rollback state are frozen before the first
    forward.  GradScaler is intentionally excluded so its overflow backoff is
    retained.  ``retry_audit`` is likewise external and append-only.
    """

    stage_c_batch_exposures(seed, 0)
    objects = dict(rollback_objects or {})
    missing_objects = sorted(_REQUIRED_ROLLBACK_OBJECTS - set(objects))
    if missing_objects:
        raise ValueError(
            f"Stage C rollback_objects missing required state surfaces: {missing_objects}"
        )
    if objects.get("scheduler") is not lr_scheduler:
        raise ValueError("rollback_objects must contain the same scheduler object")
    validate_stage_c_optimizer(groups, optimizer, lr_scheduler=lr_scheduler)
    selected_devices = {parameter.device for parameter in groups.all}
    if len(selected_devices) != 1:
        raise StageCInvalidImplementationError(
            "formal Stage C requires all A/T/R Parameters on one device"
        )
    require_cuda_autocast = next(iter(selected_devices)).type == "cuda"
    if require_cuda_autocast and (
        type(scaler) is not torch.cuda.amp.GradScaler
        or not bool(scaler.is_enabled())
    ):
        raise StageCInvalidImplementationError(
            "formal CUDA Stage C requires the exact enabled torch.cuda.amp.GradScaler"
        )
    runtime_path, runtime = _canonical_stage_c_runtime(model, groups)
    forbidden = {id(model), id(optimizer), id(scaler)}
    if any(id(value) in forbidden for value in objects.values()):
        raise ValueError("model, optimizer, and GradScaler cannot be duplicated in rollback_objects")
    if len({id(value) for value in objects.values()}) != len(objects):
        raise ValueError("rollback_objects must not contain object aliases")
    successful_update = _validate_global_success_state(
        objects=objects, lr_scheduler=lr_scheduler, seed=seed
    )
    if successful_update >= 4200:
        raise StageCInvalidImplementationError(
            "Stage C cannot advance beyond 4200 successful updates"
        )
    expected_actions = _install_stage_c_action_batch(
        runtime, seed=seed, successful_update=successful_update
    )

    audit = retry_audit if retry_audit is not None else []
    if any(id(value) == id(audit) for value in objects.values()):
        raise ValueError("append-only retry_audit cannot be a rollback object")
    batch_hash = hash_materialized_batch(materialized_batch)
    rng_snapshot = _snapshot_rng()
    topology_state = _model_topology_state(model)
    parameter_state = _model_parameter_state(model, groups)
    model_buffer_state = _module_buffer_state(model)
    model_python_state = _module_python_state(model)
    success_python_state = _module_python_state(
        model,
        ignored_by_path=_approved_success_python_attributes(runtime_path),
    )
    optimizer_state = _clone_state_dict(optimizer.state_dict())
    object_states = {name: _snapshot_object(value) for name, value in objects.items()}

    for attempt_index in range(4):
        _assert_model_topology(
            topology_state, model, where=f"before attempt {attempt_index + 1}"
        )
        if runtime.latest_schedule is not None or runtime.latest_summary is not None:
            raise StageCInvalidImplementationError(
                f"canonical runtime {runtime_path} retained stale forward evidence"
            )
        losses, runtime_evidence = _run_attempt_with_runtime_evidence(
            attempt=attempt,
            model=model,
            runtime=runtime,
            expected_actions=expected_actions,
            require_cuda_autocast=require_cuda_autocast,
        )
        _assert_model_topology(
            topology_state, model, where=f"during attempt {attempt_index + 1} forward"
        )
        _assert_control_objects_at_common_start(
            objects=objects, snapshots=object_states
        )
        _assert_parameters_unchanged(parameter_state, model)
        executed_actions = runtime_evidence.actions
        transport_executed = _transport_executed(executed_actions)
        exposures, action_batch_sha256 = _validate_stage_c_attempt_actions(
            seed=seed,
            successful_update=successful_update,
            action_payload=executed_actions,
        )
        if hash_materialized_batch(materialized_batch) != batch_hash:
            raise StageCInvalidImplementationError("materialized batch changed during Stage-C attempt")

        _assert_attempt_loss_provenance(losses, runtime_evidence, groups)

        step_audit = loss_specific_amp_step(
            detector_loss=losses.detector_loss,
            feature_loss=losses.feature_loss,
            risk_loss=losses.risk_loss,
            groups=groups,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            action_payload=executed_actions,
        )
        _assert_model_topology(
            topology_state, model, where=f"during attempt {attempt_index + 1} optimizer step"
        )
        overflow = bool(step_audit["overflow"])
        audit.append(
            {
                "attempt": attempt_index + 1,
                "retry": attempt_index,
                "batch_hash": batch_hash,
                "seed": seed,
                "successful_update": successful_update,
                "action_batch_sha256": action_batch_sha256,
                "exposures": [dict(row) for row in exposures],
                "scale": step_audit["scale"],
                "post_scale": step_audit["post_scale"],
                "overflow": overflow,
                "transport_executed": transport_executed,
                "transport_grad_finite": step_audit["transport_grad_finite"],
                "transport_grad_norm": step_audit["transport_grad_norm"],
            }
        )
        if not overflow:
            _assert_control_objects_at_common_start(
                objects=objects, snapshots=object_states
            )
            if not _buffer_state_equal(model_buffer_state, model):
                raise StageCInvalidImplementationError(
                    "unapproved registered buffer identity/storage/metadata/bytes mutation "
                    "during successful Stage-C attempt"
                )
            _assert_success_parameter_transition(parameter_state, model, groups)
            _assert_success_python_state(
                success_python_state, model, runtime_path=runtime_path
            )
            _advance_success_state(
                objects=objects,
                model=model,
                lr_scheduler=lr_scheduler,
                seed=seed,
                batch_hash=batch_hash,
                action_batch_sha256=action_batch_sha256,
                exposures=exposures,
            )
            if not _buffer_state_equal(model_buffer_state, model):
                raise StageCInvalidImplementationError(
                    "unapproved registered buffer identity/storage/metadata/bytes mutation "
                    "during successful state advance"
                )
            _assert_success_python_state(
                success_python_state, model, runtime_path=runtime_path
            )
            _assert_tensor_boundary(
                runtime_evidence.signals_boundary, runtime.latest_signals
            )
            _assert_tensor_boundary(
                runtime_evidence.detector_boundary,
                runtime_evidence.detector_output,
            )
            _assert_tensor_boundary(
                runtime_evidence.feature_boundary, runtime.latest_output
            )
            validate_stage_c_optimizer(groups, optimizer, lr_scheduler=lr_scheduler)
            _validate_global_success_state(
                objects=objects, lr_scheduler=lr_scheduler, seed=seed
            )
            _assert_model_topology(
                topology_state, model, where="during successful state advance"
            )
            optimizer.zero_grad(set_to_none=True)
            return {
                "status": "SUCCESS",
                "batch_hash": batch_hash,
                "seed": seed,
                "successful_update": successful_update,
                "action_batch_sha256": action_batch_sha256,
                "exposures": [dict(row) for row in exposures],
                "attempts": attempt_index + 1,
                "retries": attempt_index,
                "gradient_audit": step_audit,
            }

        current_optimizer_state = optimizer.state_dict()
        _assert_model_topology(
            topology_state, model, where="during overflow handling"
        )
        _assert_parameters_unchanged(parameter_state, model)
        if not _state_equal(optimizer_state, current_optimizer_state):
            raise StageCInvalidImplementationError(
                "GradScaler overflow changed optimizer parameters or state"
            )

        optimizer.zero_grad(set_to_none=True)
        _restore_module_buffers(model, model_buffer_state)
        _restore_module_python_state(model, model_python_state)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        for name, value in objects.items():
            _restore_object(value, object_states[name])
        _restore_rng(rng_snapshot)

        _assert_model_topology(
            topology_state, model, where="after overflow rollback"
        )
        _assert_parameters_unchanged(parameter_state, model)
        if not _buffer_state_equal(model_buffer_state, model):
            raise StageCInvalidImplementationError(
                "model buffers were not restored by reference, metadata, and value after overflow"
            )
        restored_python_state = _module_python_state(model)
        if not _state_equal(model_python_state, restored_python_state):
            mismatch = _first_state_mismatch(model_python_state, restored_python_state)
            raise StageCInvalidImplementationError(
                f"model Python state was not restored bitwise after overflow: {mismatch}"
            )
        if not _state_equal(optimizer_state, optimizer.state_dict()):
            raise StageCInvalidImplementationError("optimizer state was not restored bitwise after overflow")
        for name, value in objects.items():
            restored = _snapshot_object(value)
            if restored[0] != object_states[name][0] or not _state_equal(
                object_states[name][1], restored[1]
            ):
                raise StageCInvalidImplementationError(
                    f"rollback object {name} was not restored bitwise after overflow"
                )
        if not _rng_equal(rng_snapshot, _snapshot_rng()):
            raise StageCInvalidImplementationError("RNG state was not restored bitwise after overflow")
        if attempt_index == 3:
            raise StageCInvalidImplementationError(
                "four overflow attempts exhausted the fixed initial-plus-three-retry budget"
            )

    raise AssertionError("unreachable")


def run_stage_c_amp_with_retry(
    *,
    materialized_batch: Any,
    model: nn.Module,
    groups: StageCParameterGroups,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    lr_scheduler: Any,
    seed: int,
    rollback_objects: Mapping[str, Any] | None = None,
    retry_audit: MutableSequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute one callback-free A3/A4 ActionFormer Stage-C transaction."""

    stage_c_batch_exposures(seed, 0)
    objects = dict(rollback_objects or {})
    missing_objects = sorted(_REQUIRED_ROLLBACK_OBJECTS - set(objects))
    if missing_objects:
        raise ValueError(
            f"Stage C rollback_objects missing required state surfaces: {missing_objects}"
        )
    if objects.get("scheduler") is not lr_scheduler:
        raise ValueError("rollback_objects must contain the same scheduler object")
    validate_stage_c_optimizer(groups, optimizer, lr_scheduler=lr_scheduler)
    selected_devices = {parameter.device for parameter in groups.all}
    if len(selected_devices) != 1:
        raise StageCInvalidImplementationError(
            "formal Stage C requires all A/T/R Parameters on one device"
        )
    require_cuda_autocast = next(iter(selected_devices)).type == "cuda"
    if require_cuda_autocast and (
        type(scaler) is not torch.cuda.amp.GradScaler
        or not bool(scaler.is_enabled())
    ):
        raise StageCInvalidImplementationError(
            "formal CUDA Stage C requires the exact enabled torch.cuda.amp.GradScaler"
        )
    runtime_path, runtime = _canonical_stage_c_runtime(model, groups)
    normalizer_path, normalizer = _canonical_loss_normalizer(model)
    forbidden = {id(model), id(optimizer), id(scaler)}
    if any(id(value) in forbidden for value in objects.values()):
        raise ValueError(
            "model, optimizer, and GradScaler cannot be rollback_objects"
        )
    if len({id(value) for value in objects.values()}) != len(objects):
        raise ValueError("rollback_objects must not contain object aliases")
    successful_update = _validate_global_success_state(
        objects=objects, lr_scheduler=lr_scheduler, seed=seed
    )
    if successful_update >= 4200:
        raise StageCInvalidImplementationError(
            "Stage C cannot advance beyond 4200 successful updates"
        )
    expected_actions = _install_stage_c_action_batch(
        runtime, seed=seed, successful_update=successful_update
    )

    audit = retry_audit if retry_audit is not None else []
    if any(id(value) == id(audit) for value in objects.values()):
        raise ValueError("append-only retry_audit cannot be a rollback object")
    batch_hash = hash_materialized_batch(materialized_batch)
    rng_snapshot = _snapshot_rng()
    topology_state = _model_topology_state(model)
    parameter_state = _model_parameter_state(model, groups)
    model_buffer_state = _module_buffer_state(model)
    model_python_state = _module_python_state(model)
    success_python_state = _module_python_state(
        model,
        ignored_by_path=_approved_success_python_attributes(runtime_path),
    )
    optimizer_state = _clone_state_dict(optimizer.state_dict())
    object_states = {
        name: _snapshot_object(value) for name, value in objects.items()
    }

    for attempt_index in range(4):
        _assert_model_topology(
            topology_state, model, where=f"before A4 attempt {attempt_index + 1}"
        )
        paired = _run_stage_c_paired_actionformer_forward(
            materialized_batch=materialized_batch,
            model=model,
            runtime=runtime,
            expected_actions=expected_actions,
            buffer_state=model_buffer_state,
            python_state=model_python_state,
            rng_state=rng_snapshot,
            topology_state=topology_state,
            require_cuda_autocast=require_cuda_autocast,
        )
        _assert_model_topology(
            topology_state,
            model,
            where=f"during A4 attempt {attempt_index + 1}",
        )
        _assert_control_objects_at_common_start(
            objects=objects, snapshots=object_states
        )
        _assert_parameters_unchanged(parameter_state, model)
        executed_actions = paired.runtime.actions
        transport_executed = _transport_executed(executed_actions)
        exposures, action_batch_sha256 = _validate_stage_c_attempt_actions(
            seed=seed,
            successful_update=successful_update,
            action_payload=executed_actions,
        )
        if hash_materialized_batch(materialized_batch) != batch_hash:
            raise StageCInvalidImplementationError(
                "materialized batch changed during Stage-C attempt"
            )
        _validate_formal_runtime_summary(
            paired.runtime.summary,
            runtime=runtime,
            actions=executed_actions,
        )
        _assert_attempt_loss_provenance(paired.losses, paired.runtime, groups)

        step_audit = loss_specific_amp_step(
            detector_loss=paired.losses.detector_loss,
            feature_loss=paired.losses.feature_loss,
            risk_loss=paired.losses.risk_loss,
            groups=groups,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            action_payload=executed_actions,
        )
        overflow = bool(step_audit["overflow"])
        audit_row = {
            "attempt": attempt_index + 1,
            "retry": attempt_index,
            "batch_hash": batch_hash,
            "seed": seed,
            "successful_update": successful_update,
            "action_batch_sha256": action_batch_sha256,
            "exposures": [dict(row) for row in exposures],
            "scale": step_audit["scale"],
            "post_scale": step_audit["post_scale"],
            "overflow": overflow,
            "transport_executed": transport_executed,
            "transport_grad_finite": step_audit["transport_grad_finite"],
            "transport_grad_norm": step_audit["transport_grad_norm"],
            "model_forward_count": paired.model_forward_count,
            "runtime_forward_count": paired.runtime_forward_count,
            "risk_forward_count": paired.risk_forward_count,
            "loss_normalizer_before": float(paired.normalizer_before.item()),
            "loss_normalizer_after_dense_temporary": float(
                paired.normalizer_after_dense.item()
            ),
            "loss_normalizer_after_counterfactual": float(
                paired.normalizer_after_counterfactual.item()
            ),
        }
        audit.append(audit_row)

        if not overflow:
            _assert_control_objects_at_common_start(
                objects=objects, snapshots=object_states
            )
            if not torch.equal(
                normalizer.detach(), paired.normalizer_after_counterfactual
            ):
                raise StageCInvalidImplementationError(
                    "successful Stage-C normalizer differs from the one CF update"
                )
            if not _buffer_logical_state_equal(
                model_buffer_state,
                model,
                allowed_value_changes=frozenset({normalizer_path}),
            ):
                raise StageCInvalidImplementationError(
                    "a buffer other than rpn_head.loss_normalizer changed on success"
                )
            allow_unchanged = (
                frozenset(map(id, groups.transport))
                if not transport_executed
                else frozenset()
            )
            _assert_success_parameter_transition(
                parameter_state,
                model,
                groups,
                allow_unchanged=allow_unchanged,
            )
            _assert_success_python_state(
                success_python_state, model, runtime_path=runtime_path
            )
            _advance_success_state(
                objects=objects,
                model=model,
                lr_scheduler=lr_scheduler,
                seed=seed,
                batch_hash=batch_hash,
                action_batch_sha256=action_batch_sha256,
                exposures=exposures,
            )
            if not _buffer_logical_state_equal(
                model_buffer_state,
                model,
                allowed_value_changes=frozenset({normalizer_path}),
            ):
                raise StageCInvalidImplementationError(
                    "successful state advance mutated an unapproved model buffer"
                )
            _assert_success_python_state(
                success_python_state, model, runtime_path=runtime_path
            )
            _assert_tensor_boundary(
                paired.runtime.signals_boundary, runtime.latest_signals
            )
            _assert_tensor_boundary(
                paired.runtime.detector_boundary,
                paired.runtime.detector_output,
            )
            _assert_tensor_boundary(
                paired.runtime.feature_boundary,
                paired.runtime.feature_output,
            )
            validate_stage_c_optimizer(
                groups, optimizer, lr_scheduler=lr_scheduler
            )
            _validate_global_success_state(
                objects=objects, lr_scheduler=lr_scheduler, seed=seed
            )
            optimizer.zero_grad(set_to_none=True)
            return {
                "status": "SUCCESS",
                "batch_hash": batch_hash,
                "seed": seed,
                "successful_update": successful_update,
                "action_batch_sha256": action_batch_sha256,
                "exposures": [dict(row) for row in exposures],
                "attempts": attempt_index + 1,
                "retries": attempt_index,
                "gradient_audit": step_audit,
                "a3_a4_audit": audit_row,
            }

        if not _state_equal(optimizer_state, optimizer.state_dict()):
            raise StageCInvalidImplementationError(
                "GradScaler overflow changed optimizer parameters or state"
            )
        optimizer.zero_grad(set_to_none=True)
        _transactional_restore_module_buffers(model, model_buffer_state)
        _restore_module_python_state(model, model_python_state)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        for name, value in objects.items():
            _restore_object(value, object_states[name])
        _restore_rng(rng_snapshot)

        _assert_model_topology(
            topology_state, model, where="after A3 overflow rollback"
        )
        _assert_parameters_unchanged(parameter_state, model)
        if not _buffer_logical_state_equal(model_buffer_state, model):
            raise StageCInvalidImplementationError(
                "model buffers were not logically restored after overflow"
            )
        restored_python_state = _module_python_state(model)
        if not _state_equal(model_python_state, restored_python_state):
            mismatch = _first_state_mismatch(
                model_python_state, restored_python_state
            )
            raise StageCInvalidImplementationError(
                f"model Python state was not restored after overflow: {mismatch}"
            )
        if not _state_equal(optimizer_state, optimizer.state_dict()):
            raise StageCInvalidImplementationError(
                "optimizer state was not restored after overflow"
            )
        for name, value in objects.items():
            restored = _snapshot_object(value)
            if restored[0] != object_states[name][0] or not _state_equal(
                object_states[name][1], restored[1]
            ):
                raise StageCInvalidImplementationError(
                    f"rollback object {name} was not restored after overflow"
                )
        if not _rng_equal(rng_snapshot, _snapshot_rng()):
            raise StageCInvalidImplementationError(
                "RNG state was not restored after overflow"
            )
        if attempt_index == 3:
            raise StageCInvalidImplementationError(
                "four overflow attempts exhausted the fixed retry budget"
            )

    raise AssertionError("unreachable")


def run_matched_dense_amp_with_retry(
    *,
    materialized_batch: Any,
    model: nn.Module,
    group: MatchedDenseParameterGroup,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    lr_scheduler: Any,
    seed: int,
    rollback_objects: Mapping[str, Any] | None = None,
    retry_audit: MutableSequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute one forced-dense common-A transaction with shadow exposure."""

    stage_c_batch_exposures(seed, 0)
    objects = dict(rollback_objects or {})
    missing_objects = sorted(_REQUIRED_ROLLBACK_OBJECTS - set(objects))
    if missing_objects:
        raise ValueError(
            "matched-dense rollback_objects missing required state surfaces: "
            f"{missing_objects}"
        )
    if objects.get("scheduler") is not lr_scheduler:
        raise ValueError(
            "matched-dense rollback_objects must contain the same scheduler"
        )
    validate_matched_dense_optimizer(group, optimizer, lr_scheduler=lr_scheduler)
    selected_devices = {parameter.device for parameter in group.all}
    if len(selected_devices) != 1:
        raise StageCInvalidImplementationError(
            "matched-dense requires all common-A Parameters on one device"
        )
    require_cuda_autocast = next(iter(selected_devices)).type == "cuda"
    if require_cuda_autocast and (
        type(scaler) is not torch.cuda.amp.GradScaler
        or not bool(scaler.is_enabled())
    ):
        raise StageCInvalidImplementationError(
            "formal CUDA matched-dense requires the exact enabled GradScaler"
        )
    runtime_path, runtime = _canonical_matched_dense_runtime(model, group)
    normalizer_path, normalizer = _canonical_loss_normalizer(model)
    forbidden = {id(model), id(optimizer), id(scaler)}
    if any(id(value) in forbidden for value in objects.values()):
        raise ValueError(
            "model, optimizer, and GradScaler cannot be matched rollback objects"
        )
    if len({id(value) for value in objects.values()}) != len(objects):
        raise ValueError("matched-dense rollback_objects contain aliases")
    successful_update = _validate_global_success_state(
        objects=objects, lr_scheduler=lr_scheduler, seed=seed
    )
    if successful_update >= 4200:
        raise StageCInvalidImplementationError(
            "matched-dense cannot advance beyond 4200 successful updates"
        )
    shadow_actions = _canonical_stage_c_action_batch(
        seed=seed,
        successful_update=successful_update,
        device=runtime.forced_actions.device,
    )
    runtime.forced_actions = torch.zeros_like(shadow_actions, dtype=torch.long)
    runtime.forced_schedule = None
    runtime.forced_action_name = (
        f"matched_dense_seed_{seed}_successful_update_{successful_update}"
    )
    runtime.capture_replay_signals = False
    runtime.latest_schedule = None
    runtime.latest_summary = None
    runtime.latest_output = None
    runtime.latest_signals = None

    audit = retry_audit if retry_audit is not None else []
    if any(id(value) == id(audit) for value in objects.values()):
        raise ValueError("matched retry_audit cannot be a rollback object")
    batch_hash = hash_materialized_batch(materialized_batch)
    exposures, action_batch_sha256 = _validate_stage_c_attempt_actions(
        seed=seed,
        successful_update=successful_update,
        action_payload=shadow_actions,
    )
    rng_snapshot = _snapshot_rng()
    topology_state = _model_topology_state(model)
    parameter_state = _model_parameter_state(model, group)
    model_buffer_state = _module_buffer_state(model)
    model_python_state = _module_python_state(model)
    success_python_state = _module_python_state(
        model,
        ignored_by_path=_approved_success_python_attributes(runtime_path),
    )
    optimizer_state = _clone_state_dict(optimizer.state_dict())
    object_states = {
        name: _snapshot_object(value) for name, value in objects.items()
    }

    for attempt_index in range(4):
        _assert_model_topology(
            topology_state,
            model,
            where=f"before matched attempt {attempt_index + 1}",
        )
        evidence = _run_matched_dense_actionformer_forward(
            materialized_batch=materialized_batch,
            model=model,
            runtime=runtime,
            require_cuda_autocast=require_cuda_autocast,
        )
        _assert_model_topology(
            topology_state,
            model,
            where=f"during matched attempt {attempt_index + 1}",
        )
        _assert_control_objects_at_common_start(
            objects=objects, snapshots=object_states
        )
        _assert_parameters_unchanged(parameter_state, model)
        if hash_materialized_batch(materialized_batch) != batch_hash:
            raise StageCInvalidImplementationError(
                "matched-dense materialized batch changed during an attempt"
            )
        _assert_tensor_boundary(
            evidence.detector_boundary, evidence.per_window_task_loss
        )
        _assert_loss_source_provenance(
            name="matched-dense detector",
            loss=evidence.detector_loss,
            source=evidence.per_window_task_loss,
            parameters=group.adapters,
        )
        step_audit = matched_dense_amp_step(
            detector_loss=evidence.detector_loss,
            group=group,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
        )
        overflow = bool(step_audit["overflow"])
        audit_row = {
            "attempt": attempt_index + 1,
            "retry": attempt_index,
            "batch_hash": batch_hash,
            "seed": seed,
            "successful_update": successful_update,
            "action_batch_sha256": action_batch_sha256,
            "exposures": [dict(row) for row in exposures],
            "scale": step_audit["scale"],
            "post_scale": step_audit["post_scale"],
            "overflow": overflow,
            "model_forward_count": evidence.model_forward_count,
            "runtime_forward_count": evidence.runtime_forward_count,
            "risk_forward_count": evidence.risk_forward_count,
            "loss_normalizer_before": float(evidence.normalizer_before.item()),
            "loss_normalizer_after": float(evidence.normalizer_after.item()),
        }
        audit.append(audit_row)

        if not overflow:
            _assert_control_objects_at_common_start(
                objects=objects, snapshots=object_states
            )
            if not torch.equal(normalizer.detach(), evidence.normalizer_after):
                raise StageCInvalidImplementationError(
                    "matched-dense successful normalizer differs from its forward"
                )
            if not _buffer_logical_state_equal(
                model_buffer_state,
                model,
                allowed_value_changes=frozenset({normalizer_path}),
            ):
                raise StageCInvalidImplementationError(
                    "matched-dense changed an unapproved model buffer"
                )
            _assert_success_parameter_transition(
                parameter_state, model, group
            )
            _assert_success_python_state(
                success_python_state, model, runtime_path=runtime_path
            )
            _advance_success_state(
                objects=objects,
                model=model,
                lr_scheduler=lr_scheduler,
                seed=seed,
                batch_hash=batch_hash,
                action_batch_sha256=action_batch_sha256,
                exposures=exposures,
            )
            if not _buffer_logical_state_equal(
                model_buffer_state,
                model,
                allowed_value_changes=frozenset({normalizer_path}),
            ):
                raise StageCInvalidImplementationError(
                    "matched-dense success advance changed an unapproved buffer"
                )
            _assert_success_python_state(
                success_python_state, model, runtime_path=runtime_path
            )
            _assert_tensor_boundary(
                evidence.detector_boundary, evidence.per_window_task_loss
            )
            validate_matched_dense_optimizer(
                group, optimizer, lr_scheduler=lr_scheduler
            )
            _validate_global_success_state(
                objects=objects, lr_scheduler=lr_scheduler, seed=seed
            )
            optimizer.zero_grad(set_to_none=True)
            return {
                "status": "SUCCESS",
                "batch_hash": batch_hash,
                "seed": seed,
                "successful_update": successful_update,
                "action_batch_sha256": action_batch_sha256,
                "exposures": [dict(row) for row in exposures],
                "attempts": attempt_index + 1,
                "retries": attempt_index,
                "gradient_audit": step_audit,
                "matched_audit": audit_row,
            }

        if not _state_equal(optimizer_state, optimizer.state_dict()):
            raise StageCInvalidImplementationError(
                "matched-dense overflow changed optimizer state"
            )
        optimizer.zero_grad(set_to_none=True)
        _transactional_restore_module_buffers(model, model_buffer_state)
        _restore_module_python_state(model, model_python_state)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        for name, value in objects.items():
            _restore_object(value, object_states[name])
        _restore_rng(rng_snapshot)
        _assert_model_topology(
            topology_state, model, where="after matched overflow rollback"
        )
        _assert_parameters_unchanged(parameter_state, model)
        if not _buffer_logical_state_equal(model_buffer_state, model):
            raise StageCInvalidImplementationError(
                "matched-dense buffers were not restored after overflow"
            )
        if not _state_equal(model_python_state, _module_python_state(model)):
            raise StageCInvalidImplementationError(
                "matched-dense Python state was not restored after overflow"
            )
        if not _state_equal(optimizer_state, optimizer.state_dict()):
            raise StageCInvalidImplementationError(
                "matched-dense optimizer was not restored after overflow"
            )
        for name, value in objects.items():
            restored = _snapshot_object(value)
            if restored[0] != object_states[name][0] or not _state_equal(
                object_states[name][1], restored[1]
            ):
                raise StageCInvalidImplementationError(
                    f"matched-dense rollback object {name} was not restored"
                )
        if not _rng_equal(rng_snapshot, _snapshot_rng()):
            raise StageCInvalidImplementationError(
                "matched-dense RNG was not restored after overflow"
            )
        if attempt_index == 3:
            raise StageCInvalidImplementationError(
                "matched-dense exhausted the fixed four-attempt budget"
            )

    raise AssertionError("unreachable")
