from __future__ import annotations

import math
import random
from typing import Any, Iterable

import torch


def enumerate_legal_local_hard_swaps(
    selected_positions: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    max_unselected_hole: int,
    max_displacement: int = 2,
    max_candidates_per_sample: int = 16,
) -> dict[str, torch.Tensor]:
    """Enumerate deterministic legal local replacements around a hard policy."""

    if selected_positions.ndim != 2 or valid_mask.ndim != 2:
        raise ValueError("selected_positions and valid_mask must be rank-two")
    if selected_positions.shape[0] != valid_mask.shape[0]:
        raise ValueError("selected_positions and valid_mask batch sizes must match")
    if max_unselected_hole < 0 or max_displacement <= 0 or max_candidates_per_sample <= 0:
        raise ValueError("hard-swap bounds must be positive, with a non-negative max hole")
    batch, slots = selected_positions.shape
    candidates = selected_positions[:, None, :].repeat(1, max_candidates_per_sample, 1)
    add_positions = selected_positions.new_full((batch, max_candidates_per_sample), -1)
    remove_positions = selected_positions.new_full((batch, max_candidates_per_sample), -1)
    candidate_valid = torch.zeros(
        batch,
        max_candidates_per_sample,
        dtype=torch.bool,
        device=selected_positions.device,
    )
    displacement = selected_positions.new_zeros((batch, max_candidates_per_sample))
    for batch_index in range(batch):
        valid = valid_mask[batch_index].bool()
        valid_len = int(valid.sum().item())
        if valid_len <= 0 or not bool(valid[:valid_len].all()) or bool(valid[valid_len:].any()):
            raise ValueError("valid_mask must be a non-empty contiguous prefix")
        base = selected_positions[batch_index]
        active = base[(base >= 0) & (base < valid_len)]
        if int(active.numel()) != min(slots, valid_len):
            raise ValueError("baseline hard selection is not exact K_eff")
        if active.numel() > 1 and bool(((active[1:] - active[:-1]) <= 0).any()):
            raise ValueError("baseline hard selection must be unique and ordered")
        occupied = {int(value) for value in active.detach().cpu().tolist()}
        proposals: list[tuple[int, int, int, torch.Tensor]] = []
        for slot_index, remove_tensor in enumerate(active):
            remove = int(remove_tensor.item())
            for distance in range(1, max_displacement + 1):
                for direction in (-1, 1):
                    add = remove + direction * distance
                    if add < 0 or add >= valid_len or add in occupied:
                        continue
                    proposal = active.clone()
                    proposal[slot_index] = add
                    proposal, _ = torch.sort(proposal)
                    sentinels = torch.cat(
                        (
                            proposal.new_tensor([-1]),
                            proposal,
                            proposal.new_tensor([valid_len]),
                        )
                    )
                    observed_hole = int((sentinels[1:] - sentinels[:-1] - 1).max().item())
                    if observed_hole > max_unselected_hole:
                        continue
                    proposals.append((distance, remove, add, proposal))
        proposals.sort(key=lambda item: (item[0], item[1], item[2]))
        if len(proposals) <= max_candidates_per_sample:
            selected_proposals = proposals
        elif max_candidates_per_sample == 1:
            selected_proposals = [proposals[len(proposals) // 2]]
        else:
            proposal_indices = [
                round(index * (len(proposals) - 1) / (max_candidates_per_sample - 1))
                for index in range(max_candidates_per_sample)
            ]
            selected_proposals = [proposals[index] for index in proposal_indices]
        for candidate_index, (distance, remove, add, proposal) in enumerate(selected_proposals):
            candidates[batch_index, candidate_index, : int(proposal.numel())] = proposal
            add_positions[batch_index, candidate_index] = add
            remove_positions[batch_index, candidate_index] = remove
            candidate_valid[batch_index, candidate_index] = True
            displacement[batch_index, candidate_index] = distance
    return {
        "candidate_selections": candidates,
        "add_positions": add_positions,
        "remove_positions": remove_positions,
        "candidate_valid": candidate_valid,
        "absolute_displacement_dense_steps": displacement,
    }


def surrogate_hard_swap_descent(
    policy_gradient: torch.Tensor,
    add_positions: torch.Tensor,
    remove_positions: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    """Predict hard-swap utility from the detector-loss descent direction."""

    if policy_gradient.ndim != 2:
        raise ValueError("policy_gradient must be [B,T]")
    if (
        add_positions.shape != remove_positions.shape
        or add_positions.shape != candidate_valid.shape
        or add_positions.shape[0] != policy_gradient.shape[0]
    ):
        raise ValueError("hard-swap tensors must align")
    valid = candidate_valid.bool()
    safe_add = add_positions.clamp_min(0)
    safe_remove = remove_positions.clamp_min(0)
    if bool((safe_add[valid] >= policy_gradient.shape[1]).any()) or bool(
        (safe_remove[valid] >= policy_gradient.shape[1]).any()
    ):
        raise ValueError("hard-swap positions are out of gradient bounds")
    add_gradient = torch.gather(policy_gradient, 1, safe_add)
    remove_gradient = torch.gather(policy_gradient, 1, safe_remove)
    return (-(add_gradient - remove_gradient)).masked_fill(~valid, 0.0)


def _average_tie_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * float(start + end - 1)
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(first: list[float], second: list[float]) -> float:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("correlation requires equally sized vectors with at least two values")
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    first_centered = [value - first_mean for value in first]
    second_centered = [value - second_mean for value in second]
    numerator = sum(left * right for left, right in zip(first_centered, second_centered))
    denominator = math.sqrt(
        sum(value * value for value in first_centered)
        * sum(value * value for value in second_centered)
    )
    if denominator <= 0.0:
        raise ValueError("correlation requires non-constant vectors")
    return numerator / denominator


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _bootstrap_spearman_interval(
    predicted: list[float],
    observed: list[float],
    *,
    batch_ids: list[int],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not (len(predicted) == len(observed) == len(batch_ids)):
        raise ValueError("cluster bootstrap vectors must have equal lengths")
    unique_batches = sorted(set(batch_ids))
    if len(unique_batches) < 2:
        raise ValueError("cluster bootstrap requires at least two distinct batches")
    indices_by_batch = {
        batch_id: [index for index, value in enumerate(batch_ids) if value == batch_id]
        for batch_id in unique_batches
    }
    rng = random.Random(int(seed))
    values = []
    for _ in range(int(samples)):
        sampled_batches = [
            unique_batches[rng.randrange(len(unique_batches))]
            for _ in unique_batches
        ]
        indices = [
            index
            for batch_id in sampled_batches
            for index in indices_by_batch[batch_id]
        ]
        predicted_sample = [predicted[index] for index in indices]
        observed_sample = [observed[index] for index in indices]
        try:
            value = _pearson(
                _average_tie_ranks(predicted_sample),
                _average_tie_ranks(observed_sample),
            )
        except ValueError:
            continue
        values.append(value)
    if not values:
        raise ValueError("bootstrap produced no informative correlation samples")
    values.sort()
    lower_index = max(0, int(math.floor(0.025 * (len(values) - 1))))
    upper_index = min(len(values) - 1, int(math.ceil(0.975 * (len(values) - 1))))
    return values[lower_index], values[upper_index]


def _bootstrap_cluster_mean_interval(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("cluster-mean bootstrap requires at least two batches")
    rng = random.Random(int(seed))
    means = []
    for _ in range(int(samples)):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lower_index = max(0, int(math.floor(0.025 * (len(means) - 1))))
    upper_index = min(len(means) - 1, int(math.ceil(0.975 * (len(means) - 1))))
    return means[lower_index], means[upper_index]


def hard_soft_alignment_report(
    predicted_utility: Iterable[float],
    hard_utility: Iterable[float],
    *,
    batch_ids: Iterable[int],
    epsilon: float = 1.0e-8,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260720,
) -> dict[str, Any]:
    predicted_all = [float(value) for value in predicted_utility]
    hard_all = [float(value) for value in hard_utility]
    batch_all = [int(value) for value in batch_ids]
    if not (len(predicted_all) == len(hard_all) == len(batch_all)):
        raise ValueError("alignment vectors must have equal lengths")
    informative = [
        index
        for index, (predicted, observed) in enumerate(zip(predicted_all, hard_all))
        if math.isfinite(predicted)
        and math.isfinite(observed)
        and abs(predicted) > float(epsilon)
        and abs(observed) > float(epsilon)
    ]
    if len(informative) < 2:
        raise ValueError("alignment requires at least two finite non-zero hard swaps")
    predicted = [predicted_all[index] for index in informative]
    observed = [hard_all[index] for index in informative]
    batches = [batch_all[index] for index in informative]
    if len(set(predicted)) < 2 or len(set(observed)) < 2:
        raise ValueError("alignment requires non-constant predicted and observed utilities")
    signs = [
        int((predicted_value > 0.0) == (observed_value > 0.0))
        for predicted_value, observed_value in zip(predicted, observed)
    ]
    successes = sum(signs)
    sign_rate = successes / len(signs)
    sign_interval = _wilson_interval(successes, len(signs))
    batch_sign_rates = []
    for batch_id in sorted(set(batches)):
        batch_signs = [
            sign for sign, value in zip(signs, batches) if value == batch_id
        ]
        batch_sign_rates.append(sum(batch_signs) / len(batch_signs))
    cluster_sign_interval = _bootstrap_cluster_mean_interval(
        batch_sign_rates,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    spearman = _pearson(_average_tie_ranks(predicted), _average_tie_ranks(observed))
    spearman_interval = _bootstrap_spearman_interval(
        predicted,
        observed,
        batch_ids=batches,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return {
        "candidate_count_total": len(predicted_all),
        "informative_nonzero_count": len(informative),
        "informative_batch_count": len(set(batches)),
        "sign_agreement": sign_rate,
        "sign_agreement_wilson95": {
            "lower": sign_interval[0],
            "upper": sign_interval[1],
            "unit": "swap_event_descriptive_only",
        },
        "sign_agreement_by_batch": batch_sign_rates,
        "sign_agreement_batch_mean": sum(batch_sign_rates) / len(batch_sign_rates),
        "sign_agreement_cluster_bootstrap95": {
            "lower": cluster_sign_interval[0],
            "upper": cluster_sign_interval[1],
            "samples_requested": int(bootstrap_samples),
            "resampling_unit": "real_batch_cluster",
        },
        "spearman": spearman,
        "spearman_bootstrap95": {
            "lower": spearman_interval[0],
            "upper": spearman_interval[1],
            "samples_requested": int(bootstrap_samples),
            "resampling_unit": "real_batch_cluster",
        },
        "predicted_utility": predicted,
        "hard_utility": observed,
        "batch_ids": batches,
    }


def preregistered_hard_soft_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Apply thresholds registered before the real-loader evidence is observed."""

    checks = {
        "at_least_24_informative_swaps": int(report["informative_nonzero_count"]) >= 24,
        "at_least_4_distinct_batches": int(report["informative_batch_count"]) >= 4,
        "sign_agreement_at_least_0_70": float(report["sign_agreement"]) >= 0.70,
        "batch_mean_sign_agreement_at_least_0_70": (
            float(report["sign_agreement_batch_mean"]) >= 0.70
        ),
        "cluster_sign_bootstrap95_lower_above_chance": (
            float(report["sign_agreement_cluster_bootstrap95"]["lower"]) > 0.50
        ),
        "spearman_at_least_0_20": float(report["spearman"]) >= 0.20,
        "spearman_bootstrap95_lower_positive": (
            float(report["spearman_bootstrap95"]["lower"]) > 0.0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failure_action": (
            "STOP_before_official60_and_do_not_claim_direct_detector_gradient"
        ),
    }


__all__ = [
    "enumerate_legal_local_hard_swaps",
    "hard_soft_alignment_report",
    "preregistered_hard_soft_gate",
    "surrogate_hard_swap_descent",
]
