"""Inference-only, rank-prefix allocation under an actual per-window Heavy cap.

The cap covers the default, layer-invariant A/B/C QKV/attention/MLP executor.
It does not cover Scout/TIA/receiver, fit a policy, or solve a utility knapsack.
"""
from dataclasses import replace
from fractions import Fraction
import math

import torch


def _cost(counts, width, layers):
    return layers * (12 * counts * width**2 + 2 * counts.square() * width).sum()


@torch.no_grad()
def allocate_heavy_prefix(plan, native_valid, route, *, width, layers, max_fraction):
    """Select the longest feasible prefix of the original signed-gain ranking.

    All legal atoms are candidates, independent of the old budget-head argmax.
    ``max_fraction`` is a cap relative to same-window eligible full Heavy MACs,
    not a token fraction. Each example must satisfy its own cap. C's coarse
    floor is mandatory; an infeasible cap is rejected. Negative scores are not
    relabelled or clipped: this ranking control fills the feasible prefix and
    makes no claim to maximize signed utility or know acquisition interactions.

    Input atoms must be the disjoint native partition of the actual executor;
    C specifically uses its native 2x2, single-tubelet groups. Edited plans have
    no policy likelihood and must not be substituted into a PG training loss.
    Width/layers must be the actual default Heavy backbone, without depth/BCR
    or per-layer rerouting. Planning runs on CPU; its latency is not hidden.
    """
    if route not in {'A', 'B', 'C'}:
        raise ValueError('Heavy cap supports A/B/C only')
    if not isinstance(width, int) or not isinstance(layers, int) or width <= 0 or layers <= 0:
        raise ValueError('positive actual source width and layer count required')
    if not math.isfinite(max_fraction) or not 0 <= max_fraction <= 1:
        raise ValueError('max_fraction must be finite and in [0,1]')
    device = plan.selected_native.device
    b, parents, n = plan.selected_native.shape
    atoms = plan.atom_to_native.detach().cpu()
    valid = native_valid.detach().cpu().reshape(b, parents, n)
    scores = plan.predicted_gain.detach().cpu()
    if valid.dtype != torch.bool or atoms.dtype != torch.long or atoms.ndim != 2:
        raise ValueError('boolean native validity and int64 atom table required')
    if scores.shape != (b, len(atoms)) or not torch.isfinite(scores).all():
        raise ValueError('one finite signed score per example/atom required')
    if atoms.numel() != parents * n or not torch.equal(atoms.flatten().sort().values, torch.arange(parents * n)):
        raise ValueError('atoms must partition the complete native grid')
    atom_parent = atoms[:, 0] // n
    if not torch.equal(atoms // n, atom_parent[:, None].expand_as(atoms)):
        raise ValueError('atom crosses a source Heavy parent')
    if route == 'C' and atoms.shape[1] != 4:
        raise ValueError('C requires original four-member native spatial groups')
    members = valid.flatten(1)[:, atoms].sum(-1)
    selected = torch.zeros((b, len(atoms)), dtype=torch.bool)
    native = torch.zeros((b, parents * n), dtype=torch.bool)
    records = []
    fraction = Fraction(str(max_fraction))
    for row in range(b):
        full = int(_cost(valid[row].sum(-1), width, layers))
        cap = full * fraction.numerator // fraction.denominator
        floor_counts = torch.zeros(parents, dtype=torch.long)
        if route == 'C':
            floor_counts.scatter_add_(0, atom_parent, (members[row] > 0).long())
        floor = int(_cost(floor_counts, width, layers))
        if floor > cap:
            raise ValueError(f'INFEASIBLE Heavy cap for row{row}: coarse floor {floor} exceeds {cap}')
        eligible = torch.where(members[row] > 0)[0]
        order = eligible[torch.argsort(scores[row, eligible], descending=True, stable=True)]
        increments = members[row, order] - (1 if route == 'C' else 0)
        parent_ids = atom_parent[order]
        marginal = torch.zeros(len(order), dtype=torch.long)
        for parent in range(parents):
            positions = torch.where(parent_ids == parent)[0]
            delta = increments[positions]
            before = floor_counts[parent] + delta.cumsum(0) - delta
            marginal[positions] = layers * (12 * delta * width**2 + 2 * width * (2 * before * delta + delta.square()))
        prefix_cost = floor + marginal.cumsum(0)
        k = int((prefix_cost <= cap).sum())
        chosen = order[:k]
        selected[row, chosen] = True
        native[row, atoms[chosen].flatten()] = True
        native[row] &= valid[row].flatten()
        counts = floor_counts.clone()
        counts.scatter_add_(0, parent_ids[:k], increments[:k])
        actual = int(_cost(counts, width, layers))
        if actual > cap or (k < len(order) and int(prefix_cost[k]) <= cap):
            raise RuntimeError('prefix cost accounting did not satisfy the cap')
        records.append(dict(row=row, cap_macs=cap, full_valid_heavy_macs=full,
                            coarse_floor_macs=floor, allocated_heavy_macs=actual,
                            actual_heavy_fraction=actual / full if full else None,
                            selected_atoms=k, eligible_atoms=len(eligible),
                            next_prefix_macs=int(prefix_cost[k]) if k < len(order) else None,
                            heavy_tokens_by_parent=counts.tolist()))
    native = native.reshape(b, parents, n).to(device)
    edited = replace(plan, selected_native=native, selected_atoms=selected.to(device),
                     sampling_order=[torch.empty(0, dtype=torch.long, device=device) for _ in range(b)],
                     execution_order=[torch.where(row.flatten())[0] for row in native],
                     log_prob=torch.zeros_like(plan.log_prob), learned_sample=torch.zeros_like(plan.learned_sample),
                     requested_budget=torch.full_like(plan.requested_budget, float(max_fraction)),
                     budget_id=torch.full_like(plan.budget_id, -1), predicted_gain=plan.predicted_gain.detach())
    return edited, dict(method='signed_gain_rank_prefix', constraint='per_window_eligible_full_heavy_mac_cap',
                        max_fraction=float(max_fraction), route=route, width=width, layers=layers,
                        planning_device='cpu', differentiable_policy=False, utility_optimality_claim=False,
                        includes_scout_tia_receiver=False, requested_budget_semantics='Heavy MAC fraction cap', rows=records)
