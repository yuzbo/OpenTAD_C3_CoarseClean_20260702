"""Same-parent, equal-cost interventions on the actual GeoSparse executor.

This module measures source R0 in eval mode. It does not train a new router,
implement instance-balanced R1, or assign a policy likelihood to edited plans.
"""
from contextlib import contextmanager
from dataclasses import replace
import copy
import random

import numpy as np
import torch

from geosparse_ext.detector import restore_buffers
from geosparse_ext.sparse import heavy_macs


def parent_token_counts(plan, native_valid, route):
    """Count actually packed tokens, including C's nonzero coarse floor."""
    if route not in {"A", "B", "C"}:
        raise ValueError("paired plan counts support A/B/C only")
    valid = native_valid.reshape_as(plan.selected_native)
    if route != "C":
        return (plan.selected_native & valid).sum(-1)
    atoms = plan.atom_to_native
    if atoms.shape[1] != 4:
        raise ValueError("C requires native single-tubelet 2x2 groups")
    n = valid.shape[-1]
    parents = atoms // n
    if not torch.equal(parents, parents[:, :1].expand_as(parents)):
        raise ValueError("C atom crosses a Heavy parent")
    members = valid.flatten(1)[:, atoms].sum(-1)
    packed = torch.where(plan.selected_atoms, members, (members > 0).long())
    counts = torch.zeros(valid.shape[:2], dtype=torch.long, device=valid.device)
    return counts.scatter_add(1, parents[:, 0][None].expand_as(packed), packed)


def equal_cost_swap(plan, native_valid, route, row, donor, candidate):
    """Return an edited diagnostic plan; keep each parent's packed count."""
    if donor == candidate or not bool(plan.selected_atoms[row, donor]) or bool(plan.selected_atoms[row, candidate]):
        raise ValueError("swap requires a selected donor and an unselected candidate")
    atoms = plan.atom_to_native
    valid = native_valid.reshape_as(plan.selected_native)
    n = valid.shape[-1]
    indices = atoms[[donor, candidate]]
    if not bool((indices // n == indices[0, 0] // n).all()):
        raise ValueError("equal-cost swap must stay within one Heavy parent")
    member_valid = valid[row].flatten()[indices]
    sizes = member_valid.sum(-1)
    if int(sizes[0]) == 0 or int(sizes[0]) != int(sizes[1]):
        raise ValueError("swap must exchange nonempty equal-size valid atoms")
    if len(torch.unique(indices)) != indices.numel():
        raise ValueError("swap atoms must have disjoint native members")
    selected = plan.selected_atoms.clone()
    selected[row, donor], selected[row, candidate] = False, True
    native = plan.selected_native.clone()
    native[row].flatten()[indices[0]] = False
    native[row].flatten()[indices[1]] = member_valid[1]
    result = replace(plan, selected_atoms=selected, selected_native=native,
                     sampling_order=[torch.empty(0, dtype=torch.long, device=native.device) for _ in selected],
                     execution_order=[torch.where(mask.flatten())[0] for mask in native],
                     log_prob=torch.zeros_like(plan.log_prob), learned_sample=torch.zeros_like(plan.learned_sample))
    if not torch.equal(parent_token_counts(plan, valid, route), parent_token_counts(result, valid, route)):
        raise ValueError("edited plan changed per-parent Heavy capacity")
    return result


@contextmanager
def _measurement_state(model):
    """Restore state that eval forwards can touch, also when a pair fails."""
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    modes = [(module, module.training) for module in model.modules()]
    grad_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None)
    missing = object()
    objects = [(model.geosparse, "execution_rng"), (model.geosparse.encoder, "trace")]
    if hasattr(model.geosparse, "regular_tia"):
        objects.append((model.geosparse.regular_tia, "temporal_size"))
    attributes = [(obj, name, getattr(obj, name, missing)) for obj, name in objects]

    def reset():
        restore_buffers(model, buffers)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])

    try:
        model.eval()
        with torch.no_grad():
            yield reset
    finally:
        reset()
        for obj, name, value in attributes:
            if value is missing:
                if hasattr(obj, name):
                    delattr(obj, name)
            else:
                setattr(obj, name, value)
        for module, mode in modes:
            module.training = mode
        for parameter, flag in grad_flags:
            parameter.requires_grad_(flag)


class CounterfactualRunner:
    """Fresh R0 forwards under fixed loaded weights; caller owns run identity."""

    def __init__(self, detector):
        config = detector.geosparse.config
        if config["route"] not in {"A", "B", "C"}:
            raise ValueError("research pair requires an A/B/C detector")
        if config["selector"] == "none" and config["budget"] == 1:
            raise ValueError("source-full control overrides forced plans; use a routed configuration")
        self.detector = detector

    def evaluate_pair(self, batch, metas, gt_segments, gt_labels, plan_a, plan_b, *, require_equal_cost=True):
        model = self.detector
        if batch.frames_hi.shape[0] != 1:
            raise ValueError("R0 diagnostic records one window at a time; do not mix per-example utilities")
        records = []
        with _measurement_state(model) as reset:
            for plan in (plan_a, plan_b):
                reset()
                state, actual, _, _, _ = model.geosparse(batch, model.epoch, forced_plan=plan)
                losses = model._task_losses(state, batch.valid_frames, metas, gt_segments, gt_labels,
                                            batch.frames_hi.shape[2])
                values = {key: float(value.detach()) for key, value in losses.items()}
                if not all(np.isfinite(value) for value in values.values()):
                    raise ValueError("nonfinite source risk in paired forward")
                trace = copy.deepcopy(model.geosparse.encoder.trace)
                records.append(dict(risk=values, heavy_macs=heavy_macs(trace, model.geosparse.encoder.source.embed_dims),
                                    trace=trace, detector_mask=state.valid.detach().cpu().tolist(),
                                    selected_atoms=actual.selected_atoms.detach().cpu().tolist(),
                                    selected_native=actual.selected_native.detach().cpu().tolist()))
        left, right = records
        signature = lambda row: [(s["layer"], s["parent"], s["qkv_tokens"], s["mlp_tokens"]) for s in row["trace"]]
        if require_equal_cost and (signature(left) != signature(right) or left["heavy_macs"] != right["heavy_macs"]):
            raise ValueError("actual Heavy trace changed capacity in an equal-cost pair")
        if left["detector_mask"] != right["detector_mask"]:
            raise ValueError("paired plans changed the detection valid domain")
        return dict(risk_name="R0_source_eval", normalization="source eval positive-count normalization",
                    operation_scope="entire forced-plan encoder and downstream detector",
                    video_id=batch.video_id[0], window_id=batch.window_id[0],
                    eval_mode=True, fresh_forward=True, cache_scope="none", equal_cost=require_equal_cost,
                    left=left, right=right,
                    signed_gain={key: left["risk"][key] - right["risk"][key] for key in left["risk"]})
