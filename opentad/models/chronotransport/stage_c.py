"""Stage-C object ownership and loss-specific AMP update primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn


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

    @property
    def all(self) -> tuple[nn.Parameter, ...]:
        return tuple(sorted((*self.adapters, *self.transport, *self.risk), key=id))


def build_stage_c_parameter_groups(
    model: nn.Module,
    *,
    adapter_modules: Sequence[nn.Module],
    transport_module: nn.Module,
    risk_module: nn.Module,
) -> StageCParameterGroups:
    adapters = _unique_parameters(adapter_modules)
    transport = _unique_parameters((transport_module,))
    risk = _unique_parameters((risk_module,))
    if not adapters or not transport or not risk:
        raise ValueError("Stage C requires non-empty A, T, and R parameter groups")
    identity_sets = [set(map(id, values)) for values in (adapters, transport, risk)]
    if identity_sets[0] & identity_sets[1] or identity_sets[0] & identity_sets[2] or identity_sets[1] & identity_sets[2]:
        raise ValueError("Stage C parameter ownership overlap")
    groups = StageCParameterGroups(adapters, transport, risk)
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
    norm = float(sum(gradient.detach().float().square().sum() for gradient in gradients).sqrt().item())
    return finite, norm > 0.0, norm


def loss_specific_amp_step(
    *,
    detector_loss: Tensor,
    feature_loss: Tensor,
    risk_loss: Tensor,
    groups: StageCParameterGroups,
    optimizer: torch.optim.Optimizer,
    scaler,
    transport_executed: bool,
    max_grad_norm: float = 1.0,
) -> dict[str, object]:
    """Perform the exact r2 loss-ownership update with one scaler step/update."""

    optimizer.zero_grad(set_to_none=True)
    initial_scale = float(scaler.get_scale())
    scaled_detector = scaler.scale(detector_loss)
    detector_gradients = torch.autograd.grad(
        scaled_detector,
        groups.adapters + groups.transport,
        retain_graph=True,
        allow_unused=True,
    )
    scaled_feature = scaler.scale(0.1 * feature_loss)
    feature_gradients = torch.autograd.grad(
        scaled_feature,
        groups.transport,
        retain_graph=True,
        allow_unused=True,
    )
    scaled_risk = scaler.scale(0.1 * risk_loss)
    risk_gradients = torch.autograd.grad(
        scaled_risk,
        groups.risk,
        retain_graph=False,
        allow_unused=True,
    )
    if float(scaler.get_scale()) != initial_scale:
        raise RuntimeError("GradScaler scale changed within one Stage-C attempt")

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
    if not (adapter_finite and transport_finite and risk_finite):
        raise FloatingPointError("Stage C produced non-finite unscaled gradients")
    if not adapter_nonzero:
        raise RuntimeError("Stage C aggregate adapter detector gradient must be nonzero")
    if not risk_nonzero:
        raise RuntimeError("Stage C aggregate risk gradient must be nonzero")
    if bool(transport_executed) and not transport_nonzero:
        raise RuntimeError("Stage C transport exposure produced zero aggregate transport gradient")
    total_norm = torch.nn.utils.clip_grad_norm_(groups.all, float(max_grad_norm))
    scaler.step(optimizer)
    scaler.update()
    return {
        "scale": initial_scale,
        "adapter_detector_grad_finite": adapter_finite,
        "adapter_detector_grad_nonzero": adapter_nonzero,
        "adapter_grad_norm": adapter_norm,
        "transport_grad_finite": transport_finite,
        "transport_grad_nonzero": transport_nonzero,
        "transport_grad_norm": transport_norm,
        "risk_grad_finite": risk_finite,
        "risk_grad_nonzero": risk_nonzero,
        "risk_grad_norm": risk_norm,
        "preclip_global_grad_norm": float(total_norm),
    }
