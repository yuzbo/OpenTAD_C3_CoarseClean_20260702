"""Offline decomposition of the frozen trainer's actual loss gradients.

The caller supplies a loaded detector and a training batch. No optimizer step,
EMA update, budget update or checkpoint selection is performed here.
"""
from contextlib import contextmanager
from itertools import combinations
import copy
import math
import random

import numpy as np
import torch

from geosparse_ext.detector import restore_buffers
from geosparse_ext.runtime import subset_batch


AUXILIARY = ("actor_loss", "critic_loss", "actionness_loss", "acquisition_loss")


@contextmanager
def _training_measurement_state(model):
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    modes = [(module, module.training) for module in model.modules()]
    parameters = [(parameter, parameter.requires_grad, parameter.grad) for parameter in model.parameters()]
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None)
    missing = object()
    objects = [(model, name) for name in ("minibatch", "latest_route_plan", "latest_acquisition", "pending_cost")]
    objects += [(model.geosparse, "execution_rng"), (model.geosparse.encoder, "trace")]
    if hasattr(model.geosparse, "regular_tia"):
        objects.append((model.geosparse.regular_tia, "temporal_size"))
    attributes = [(obj, name, getattr(obj, name, missing)) for obj, name in objects]
    try:
        # Isolate the list before forward_train appends dynamic costs to it.
        model.pending_cost = list(model.pending_cost)
        model.train()
        with torch.enable_grad():
            yield
    finally:
        restore_buffers(model, buffers)
        for obj, name, value in attributes:
            if value is missing:
                if hasattr(obj, name):
                    delattr(obj, name)
            else:
                setattr(obj, name, value)
        for module, mode in modes:
            module.training = mode
        for parameter, flag, grad in parameters:
            parameter.requires_grad_(flag)
            parameter.grad = grad
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])


def _parameter_group(name):
    if name.startswith("geosparse.scout."):
        part = name.split(".")[2]
        return "scout_shared" if part in {"spatial", "temporal"} else "scout_" + part
    if name.startswith("geosparse.encoder.source."):
        return "source_tia" if ".adapter." in name else "source_backbone"
    if name.startswith("geosparse."):
        return name.split(".")[1] if not name.startswith("geosparse.encoder.") else "coarse_projection"
    return name.split(".")[0]


def _dot(left, right):
    return sum(float((left[name].double() * right[name].double()).sum()) for name in left.keys() & right.keys())


def _norm(gradients):
    squared = _dot(gradients, gradients)
    return math.sqrt(squared) if math.isfinite(squared) else None


def _gradient_summary(gradients):
    groups = {}
    for name, value in gradients.items():
        groups.setdefault(_parameter_group(name), {})[name] = value
    return dict(norm=_norm(gradients), connected_parameters=len(gradients),
                nonfinite_elements=sum(int((~torch.isfinite(value)).sum()) for value in gradients.values()),
                groups={name: dict(norm=_norm(values), connected_parameters=len(values))
                        for name, values in groups.items()})


def measure_training_gradients(model, batch, *, microbatch_size=None, amp=False, loss_scale=1., clip_norm=1.):
    """Measure task/actor/critic/CF gradients from the same primary forwards.

    Keep epoch, minibatch and loaded scalar normalizer at their supplied values.
    microbatch_size must match the run being diagnosed: changing it changes the
    source head's local-positive normalization. loss_scale is the run's current
    GradScaler scale when reproducing AMP; all reported gradients are unscaled.
    Reentrant activation checkpoints require backward(), not autograd.grad().
    """
    if model.geosparse.config["route"] not in {"A", "B", "C"}:
        raise ValueError("gradient diagnostic currently supports A/B/C")
    size = len(batch["inputs"])
    microbatch_size = size if microbatch_size is None else microbatch_size
    if size == 0 or not isinstance(microbatch_size, int) or not 1 <= microbatch_size <= size:
        raise ValueError("microbatch_size must be between one and the effective batch size")
    if not math.isfinite(loss_scale) or loss_scale <= 0 or not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("loss_scale and clip_norm must be finite and positive")
    if amp and not next(model.parameters()).is_cuda:
        raise ValueError("production FP16 autocast requires a CUDA detector")
    named_parameters = dict(model.named_parameters())
    gradients, scalar_losses, records = {}, {}, []
    epoch, first_minibatch = model.epoch, model.minibatch
    with _training_measurement_state(model):
        for begin in range(0, size, microbatch_size):
            end = min(size, begin + microbatch_size)
            weight = (end - begin) / size
            normalizer_before = float(model.rpn_head.loss_normalizer)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                losses = model.forward_train(**subset_batch(batch, begin, end))
            if set(losses) - {"cost", "cls_loss", "reg_loss", *AUXILIARY}:
                raise ValueError("unrecognized source loss: extend the decomposition explicitly")
            # Source ActionFormer sums these terms. Avoid cost - auxiliaries:
            # a large actor scalar can destroy the small task scalar numerically.
            components = {"task": losses["cls_loss"] + losses["reg_loss"]}
            components.update({name.removesuffix("_loss"): losses[name] for name in AUXILIARY if name in losses})
            components["total"] = losses["cost"]
            if not all(bool(torch.isfinite(value)) for value in components.values()):
                raise FloatingPointError("nonfinite loss in training-gradient diagnostic")
            plan = model.latest_route_plan
            records.append(dict(begin=begin, end=end, minibatch=model.minibatch - 1,
                                sample_weight=weight, normalizer_before=normalizer_before,
                                normalizer_after=float(model.rpn_head.loss_normalizer),
                                selected_atoms=plan.selected_atoms.sum(-1).detach().cpu().tolist(),
                                selected_native=plan.selected_native.sum((-1, -2)).detach().cpu().tolist(),
                                learned_sample=plan.learned_sample.detach().cpu().tolist(),
                                acquisition=copy.deepcopy(model.latest_acquisition),
                                trace=copy.deepcopy(model.geosparse.encoder.trace)))
            # Retaining one graph fixes the sampled plan, dropout and probe target.
            # backward() also preserves the actual reentrant-checkpoint semantics.
            for index, (component, loss) in enumerate(components.items()):
                for parameter in named_parameters.values():
                    parameter.grad = None
                (loss * weight * loss_scale).backward(retain_graph=index < len(components) - 1)
                destination = gradients.setdefault(component, {})
                for name, parameter in named_parameters.items():
                    if parameter.grad is not None:
                        value = parameter.grad.detach().float().cpu() / loss_scale
                        if name in destination:
                            destination[name].add_(value)
                        else:
                            destination[name] = value.clone()
                scalar_losses[component] = scalar_losses.get(component, 0.) + weight * float(loss.detach())
            del losses, components
    summaries = {name: _gradient_summary(values) for name, values in gradients.items()}
    pairs = []
    for left, right in combinations([name for name in gradients if name != "total"], 2):
        dot = _dot(gradients[left], gradients[right])
        denominator = (summaries[left]["norm"] or 0.) * (summaries[right]["norm"] or 0.)
        cosine = dot / denominator if denominator > 0 and math.isfinite(dot) else None
        pairs.append(dict(left=left, right=right, dot=dot if math.isfinite(dot) else None, cosine=cosine))
    reconstructed = {}
    for component, values in gradients.items():
        if component == "total":
            continue
        for name, value in values.items():
            if name in reconstructed:
                reconstructed[name].add_(value.double())
            else:
                reconstructed[name] = value.double()
    total = gradients["total"]
    differences = {name: reconstructed.get(name, 0.) - total.get(name, 0.)
                   for name in reconstructed.keys() | total.keys()}
    error_norm = _norm(differences)
    total_norm = summaries["total"]["norm"]
    finite = all(row["nonfinite_elements"] == 0 for row in summaries.values())
    coefficient = min(1., clip_norm / (total_norm + 1e-6)) if total_norm is not None else None
    return dict(measurement="training_loss_gradient_components", epoch=epoch, minibatch=first_minibatch,
                effective_batch=size, microbatch_size=microbatch_size, amp=amp, loss_scale=loss_scale,
                optimizer_step_performed=False, state_restored=True,
                primary_forward_calls=len(records), acquisition_forward_calls=sum(r["acquisition"] is not None for r in records),
                gradient_status="FINITE" if finite else "NONFINITE", scalar_losses=scalar_losses,
                gradients=summaries, pairwise=pairs, microbatches=records,
                global_clip=dict(max_norm=clip_norm, preclip_norm=total_norm, coefficient=coefficient),
                reconstruction=dict(absolute_l2=error_norm,
                                    relative_l2=error_norm / max(total_norm, 1e-12)
                                    if error_norm is not None and total_norm is not None else None))
