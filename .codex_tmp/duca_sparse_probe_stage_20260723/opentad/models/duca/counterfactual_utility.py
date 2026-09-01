from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Dict

import torch
import torch.nn.functional as F


def build_finite_hard_one_swap_candidates(
    selected_positions: torch.Tensor,
    policy_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    max_candidates: int,
    max_unselected_hole: int,
) -> Dict[str, torch.Tensor]:
    """Build a small, deterministic set of feasible hard one-swap policies."""
    if selected_positions.ndim != 2 or policy_scores.ndim != 2 or valid_mask.shape != policy_scores.shape:
        raise ValueError("invalid counterfactual candidate shapes")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    batch, length = policy_scores.shape
    candidates = selected_positions[:, None, :].repeat(1, max_candidates, 1)
    candidate_positions = selected_positions.new_full((batch, max_candidates), -1)
    replaced_slots = selected_positions.new_full((batch, max_candidates), -1)
    candidate_valid = torch.zeros((batch, max_candidates), dtype=torch.bool, device=selected_positions.device)
    for b in range(batch):
        base = selected_positions[b]
        valid_length = int(valid_mask[b].sum().item())
        base = base[(base >= 0) & (base < valid_length)]
        base = torch.unique(base, sorted=True)
        if valid_length < selected_positions.shape[1]:
            if base.numel() != valid_length:
                raise ValueError("short-window baseline must contain every valid position exactly once")
            continue
        if base.numel() != selected_positions.shape[1]:
            raise ValueError("baseline selection must be exact-K, unique and non-negative")
        selected_set = set(int(x) for x in base.tolist())
        additions = [
            int(x) for x in torch.argsort(policy_scores[b], descending=True).tolist()
            if bool(valid_mask[b, x]) and x not in selected_set
        ]
        removals = torch.argsort(policy_scores[b, base], descending=False).tolist()
        out_index = 0
        for add in additions:
            for local_slot in removals:
                proposal = base.clone()
                proposal[int(local_slot)] = add
                proposal, _ = torch.sort(proposal)
                if torch.unique(proposal).numel() != proposal.numel():
                    continue
                if max_unselected_hole > 0:
                    sentinels = torch.cat((proposal.new_tensor([-1]), proposal, proposal.new_tensor([valid_length])))
                    if int((sentinels[1:] - sentinels[:-1] - 1).max().item()) > max_unselected_hole:
                        continue
                candidates[b, out_index] = proposal
                candidate_positions[b, out_index] = add
                replaced_slots[b, out_index] = int(local_slot)
                candidate_valid[b, out_index] = True
                out_index += 1
                break
            if out_index == max_candidates:
                break
    return {
        "candidate_selections": candidates,
        "candidate_positions": candidate_positions,
        "replaced_slots": replaced_slots,
        "candidate_valid": candidate_valid,
    }


def build_local_cell_hard_flip_candidates(
    selected_positions: torch.Tensor,
    policy_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    cell_starts: torch.Tensor,
    cell_ends: torch.Tensor,
    detector_grid_positions: torch.Tensor | None = None,
    *,
    max_candidates: int,
) -> Dict[str, torch.Tensor]:
    """Build at most one within-cell hard flip for each distinct local cell."""

    if selected_positions.ndim != 2 or policy_scores.ndim != 2 or valid_mask.shape != policy_scores.shape:
        raise ValueError("invalid local-cell counterfactual tensor shapes")
    if cell_starts.shape != selected_positions.shape or cell_ends.shape != selected_positions.shape:
        raise ValueError("local-cell bounds must align with selected positions [B,K]")
    if detector_grid_positions is None:
        detector_grid_positions = selected_positions
    if detector_grid_positions.shape != selected_positions.shape:
        raise ValueError("detector_grid_positions must align with selected positions [B,K]")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    batch = int(policy_scores.shape[0])
    candidates = selected_positions[:, None, :].repeat(1, max_candidates, 1)
    candidate_positions = selected_positions.new_full((batch, max_candidates), -1)
    replaced_slots = selected_positions.new_full((batch, max_candidates), -1)
    candidate_cells = selected_positions.new_full((batch, max_candidates), -1)
    candidate_valid = torch.zeros((batch, max_candidates), dtype=torch.bool, device=selected_positions.device)
    candidate_score_delta = policy_scores.new_zeros((batch, max_candidates))

    for batch_index in range(batch):
        valid_positions = torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten()
        expected = torch.arange(valid_positions.numel(), device=valid_positions.device, dtype=valid_positions.dtype)
        if not torch.equal(valid_positions, expected):
            raise ValueError("local-cell counterfactuals require a contiguous valid prefix")
        proposals = []
        for slot_index in range(int(selected_positions.shape[1])):
            remove = int(selected_positions[batch_index, slot_index].item())
            start = int(cell_starts[batch_index, slot_index].item())
            end = int(cell_ends[batch_index, slot_index].item())
            if remove < 0:
                if start >= 0 or end >= 0:
                    raise ValueError("inactive local-cell slots must use -1 bounds")
                continue
            if not (0 <= start <= remove < end <= int(valid_positions.numel())):
                raise ValueError("selected position must lie inside its valid local cell")
            if end - start <= 1:
                continue
            local_positions = torch.arange(start, end, device=selected_positions.device)
            alternatives = local_positions[local_positions != remove]
            alternative_scores = policy_scores[batch_index, alternatives].detach()
            best_local = int(torch.argmax(alternative_scores).item())
            add = int(alternatives[best_local].item())
            score_delta = float(
                (policy_scores[batch_index, add] - policy_scores[batch_index, remove]).detach().float().item()
            )
            proposals.append((score_delta, slot_index, add))

        # Prefer the closest hard competitors; cell index is the deterministic tie-break.
        proposals.sort(key=lambda item: (-item[0], item[1]))
        for candidate_index, (score_delta, slot_index, add) in enumerate(proposals[:max_candidates]):
            proposal = selected_positions[batch_index].clone()
            proposal[slot_index] = add
            if bool(torch.any(proposal[1:] < proposal[:-1]).item()):
                raise RuntimeError("within-cell flips must preserve temporal slot order")
            candidates[batch_index, candidate_index] = proposal
            candidate_positions[batch_index, candidate_index] = add
            replaced_slots[batch_index, candidate_index] = slot_index
            candidate_cells[batch_index, candidate_index] = slot_index
            candidate_valid[batch_index, candidate_index] = True
            candidate_score_delta[batch_index, candidate_index] = score_delta

    return {
        "candidate_selections": candidates,
        "candidate_positions": candidate_positions,
        "replaced_slots": replaced_slots,
        "candidate_cell_indices": candidate_cells,
        "candidate_valid": candidate_valid,
        "candidate_score_delta_at_request": candidate_score_delta,
        "detector_grid_positions": detector_grid_positions.detach().clone(),
        "candidate_policy": "distinct_cells_best_within_cell_hard_flip",
    }


def detached_hard_one_swap_utilities(
    selected_positions: torch.Tensor,
    candidate_positions: torch.Tensor,
    evaluate_detector_loss: Callable[[torch.Tensor], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Measure train-only detector utility using actual discrete one-swap selections.

    Utility is the reduction in detector loss relative to the unmodified hard
    selection. Detector graphs are deliberately detached: this function creates
    supervision for the selector and is not a detector-gradient estimator.
    """
    if selected_positions.ndim != 2 or candidate_positions.ndim != 2:
        raise ValueError("selected_positions and candidate_positions must be [B,K] and [B,M]")
    if selected_positions.shape[0] != candidate_positions.shape[0]:
        raise ValueError("selected and candidate positions must share the batch dimension")
    if selected_positions.numel() == 0:
        raise ValueError("selected_positions must not be empty")
    if torch.any(selected_positions < 0) or torch.any(candidate_positions < 0):
        raise ValueError("hard one-swap positions must be non-negative")

    with torch.no_grad():
        baseline = evaluate_detector_loss(selected_positions.clone()).reshape(-1)
        if baseline.numel() != selected_positions.shape[0] or not torch.isfinite(baseline).all():
            raise ValueError("evaluate_detector_loss must return one finite loss per batch item")
        utilities = baseline.new_full(candidate_positions.shape, float("-inf"))
        replaced_slots = torch.full_like(candidate_positions, -1)
        for batch_index in range(selected_positions.shape[0]):
            selected = selected_positions[batch_index]
            for candidate_index, candidate in enumerate(candidate_positions[batch_index]):
                if torch.any(selected == candidate):
                    utilities[batch_index, candidate_index] = 0.0
                    continue
                best_utility = None
                best_slot = -1
                for slot_index in range(selected.numel()):
                    swapped = selected_positions.clone()
                    swapped[batch_index, slot_index] = candidate
                    swapped[batch_index], _ = torch.sort(swapped[batch_index])
                    loss = evaluate_detector_loss(swapped).reshape(-1)
                    if loss.numel() != selected_positions.shape[0] or not torch.isfinite(loss).all():
                        raise ValueError("counterfactual detector losses must be finite and batch-aligned")
                    utility = baseline[batch_index] - loss[batch_index]
                    if best_utility is None or utility > best_utility:
                        best_utility = utility
                        best_slot = slot_index
                utilities[batch_index, candidate_index] = best_utility
                replaced_slots[batch_index, candidate_index] = best_slot
    return {
        "baseline_detector_loss": baseline.detach(),
        "candidate_utility": utilities.detach(),
        "replaced_slot": replaced_slots.detach(),
        "teacher_kind": "detached_hard_one_swap_detector_loss_reduction",
        "direct_detector_gradient": False,
    }


def counterfactual_utility_distillation_loss(
    policy_scores: torch.Tensor,
    teacher_utility: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Distill signed hard-swap utility relative to a fixed no-op baseline.

    The extra class has detector utility zero and selector score delta zero, so
    the optimum preserves whether a swap is better or worse than retaining the
    current hard selection. This categorical helper does not guarantee each
    candidate's local descent direction when swaps share selected positions;
    the integrated hard-swap path uses ``signed_one_swap_proximal_loss`` for
    that stronger score-space contract.
    """
    if policy_scores.shape != teacher_utility.shape or policy_scores.shape != valid_mask.shape:
        raise ValueError("policy_scores, teacher_utility and valid_mask must have identical [B,T] shapes")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    valid = valid_mask.bool()
    active = valid.any(dim=1)
    student = policy_scores.float()
    teacher = teacher_utility.detach().to(device=policy_scores.device, dtype=torch.float32)
    if not torch.isfinite(teacher[valid]).all():
        raise ValueError("valid counterfactual utilities must be finite")

    batch = int(student.shape[0])
    student = torch.cat((student.new_zeros((batch, 1)), student), dim=1)
    teacher = torch.cat((teacher.new_zeros((batch, 1)), teacher), dim=1)
    valid = torch.cat(
        (
            torch.ones((batch, 1), dtype=torch.bool, device=valid.device),
            valid,
        ),
        dim=1,
    )
    neg = torch.finfo(student.dtype).min
    student_logits = (student / temperature).masked_fill(~valid, neg)
    teacher_logits = (teacher / temperature).masked_fill(~valid, neg)
    target = torch.softmax(teacher_logits, dim=-1)
    per_item = -(target * torch.log_softmax(student_logits, dim=-1)).sum(dim=-1)
    active_weight = active.to(per_item.dtype)
    return (per_item * active_weight).sum() / active_weight.sum().clamp_min(1.0)


def counterfactual_pair_scores(
    policy_scores: torch.Tensor,
    candidate_positions: torch.Tensor,
    replaced_slots: torch.Tensor,
    baseline_positions: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    """Return deploy-time score(add) - score(remove) for each hard swap."""
    if policy_scores.ndim != 2 or baseline_positions.ndim != 2:
        raise ValueError("policy_scores and baseline_positions must be rank two")
    if candidate_positions.shape != replaced_slots.shape or candidate_positions.shape != candidate_valid.shape:
        raise ValueError("counterfactual candidate tensors must share [B,M] shape")
    if policy_scores.shape[0] != candidate_positions.shape[0] or baseline_positions.shape[0] != policy_scores.shape[0]:
        raise ValueError("counterfactual tensors must share the batch dimension")
    valid = candidate_valid.bool()
    if torch.any(candidate_positions[valid] < 0) or torch.any(replaced_slots[valid] < 0):
        raise ValueError("valid counterfactual add positions and remove slots must be non-negative")
    safe_add = candidate_positions.clamp(min=0)
    safe_slot = replaced_slots.clamp(min=0)
    if torch.any(safe_add[valid] >= policy_scores.shape[1]) or torch.any(safe_slot[valid] >= baseline_positions.shape[1]):
        raise ValueError("counterfactual add position or remove slot is out of range")
    remove_positions = torch.gather(baseline_positions, 1, safe_slot)
    if torch.any(remove_positions[valid] < 0):
        raise ValueError("valid counterfactual swaps must remove a selected position")
    remove_positions = remove_positions.clamp_min(0)
    add_scores = torch.gather(policy_scores, 1, safe_add)
    remove_scores = torch.gather(policy_scores, 1, remove_positions)
    return (add_scores - remove_scores).masked_fill(~valid, 0.0)


def build_swap_incidence_matrix(
    policy_scores: torch.Tensor,
    candidate_positions: torch.Tensor,
    replaced_slots: torch.Tensor,
    baseline_positions: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    """Build rows whose dot product with scores is score(add)-score(remove)."""
    if policy_scores.ndim != 2 or baseline_positions.ndim != 2:
        raise ValueError("policy_scores and baseline_positions must be rank two")
    if candidate_positions.shape != replaced_slots.shape or candidate_positions.shape != candidate_valid.shape:
        raise ValueError("counterfactual candidate tensors must share [B,M] shape")
    if policy_scores.shape[0] != candidate_positions.shape[0] or baseline_positions.shape[0] != policy_scores.shape[0]:
        raise ValueError("counterfactual tensors must share the batch dimension")
    valid = candidate_valid.bool()
    if torch.any(candidate_positions[valid] < 0) or torch.any(replaced_slots[valid] < 0):
        raise ValueError("valid counterfactual add positions and remove slots must be non-negative")
    safe_add = candidate_positions.clamp_min(0)
    safe_slot = replaced_slots.clamp_min(0)
    if torch.any(safe_add[valid] >= policy_scores.shape[1]) or torch.any(safe_slot[valid] >= baseline_positions.shape[1]):
        raise ValueError("counterfactual add position or remove slot is out of range")
    remove_positions = torch.gather(baseline_positions, 1, safe_slot)
    if torch.any(remove_positions[valid] < 0):
        raise ValueError("valid counterfactual swaps must remove a selected position")
    remove_positions = remove_positions.clamp_min(0)
    if torch.any(safe_add[valid] == remove_positions[valid]):
        raise ValueError("a counterfactual swap cannot add and remove the same position")

    incidence = policy_scores.new_zeros(
        (policy_scores.shape[0], candidate_positions.shape[1], policy_scores.shape[1]),
        dtype=torch.float32,
    )
    for batch_index in range(int(policy_scores.shape[0])):
        for candidate_index in range(int(candidate_positions.shape[1])):
            if not bool(valid[batch_index, candidate_index]):
                continue
            add = int(safe_add[batch_index, candidate_index].item())
            remove = int(remove_positions[batch_index, candidate_index].item())
            incidence[batch_index, candidate_index, add] = 1.0
            incidence[batch_index, candidate_index, remove] = -1.0
    return incidence


def _normalized_signed_swap_utility(
    teacher_utility: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    valid = valid_mask.bool()
    teacher = teacher_utility.detach().float()
    if not torch.isfinite(teacher[valid]).all():
        raise ValueError("valid counterfactual utilities must be finite")
    count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=teacher.dtype)
    mean_abs = teacher.abs().masked_fill(~valid, 0.0).sum(dim=1, keepdim=True) / count
    scale = (mean_abs * float(temperature)).clamp_min(torch.finfo(teacher.dtype).eps)
    normalized = torch.tanh(teacher / scale)
    return normalized.masked_fill(~valid, 0.0)


def signed_one_swap_proximal_loss(
    policy_scores: torch.Tensor,
    swap_incidence: torch.Tensor,
    teacher_utility: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    step_size: float = 1.0,
) -> torch.Tensor:
    """Create a score-space update whose swap direction matches signed utility.

    Candidate swaps can share a removed frame, so independently regressing their
    pair scores creates conflicting center-score gradients. Whitening by the
    small swap Gram matrix makes the first-order change of every valid pair
    score proportional to its detached, normalized detector utility.
    """
    if policy_scores.ndim != 2 or swap_incidence.ndim != 3:
        raise ValueError("policy_scores and swap_incidence must be [B,T] and [B,M,T]")
    if swap_incidence.shape[0] != policy_scores.shape[0] or swap_incidence.shape[2] != policy_scores.shape[1]:
        raise ValueError("swap incidence must share policy score batch and time dimensions")
    if teacher_utility.shape != valid_mask.shape or teacher_utility.shape != swap_incidence.shape[:2]:
        raise ValueError("teacher utility and valid mask must match swap incidence [B,M]")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")

    valid = valid_mask.bool()
    autocast_context = (
        torch.autocast(device_type=policy_scores.device.type, enabled=False)
        if policy_scores.device.type in {"cpu", "cuda"}
        else nullcontext()
    )
    with autocast_context:
        normalized_utility = _normalized_signed_swap_utility(
            teacher_utility,
            valid,
            temperature=temperature,
        )
        pair_scores = torch.einsum("bmt,bt->bm", swap_incidence.float(), policy_scores.float())
        per_sample = []
        for batch_index in range(int(policy_scores.shape[0])):
            active = valid[batch_index]
            if not bool(active.any()):
                continue
            incidence = swap_incidence[batch_index, active].float()
            utility = normalized_utility[batch_index, active]
            gram = incidence @ incidence.transpose(0, 1)
            if int(torch.linalg.matrix_rank(gram).item()) != int(gram.shape[0]):
                raise ValueError("valid swap incidence Gram matrix must be full rank")
            whitened_utility = torch.linalg.solve(gram, utility[:, None]).squeeze(1)
            current = pair_scores[batch_index, active]
            target = current.detach() + float(step_size) * whitened_utility
            per_sample.append(0.5 * (current - target).square().mean())
    if not per_sample:
        finite_scores = torch.where(
            torch.isfinite(policy_scores),
            policy_scores,
            torch.zeros_like(policy_scores),
        )
        return (finite_scores * 0.0).sum()
    return torch.stack(per_sample).mean()


def local_cell_signed_logistic_loss(
    policy_scores: torch.Tensor,
    candidate_positions: torch.Tensor,
    replaced_slots: torch.Tensor,
    baseline_positions: torch.Tensor,
    teacher_utility: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Match each independent within-cell score difference to signed utility.

    Candidate rows are required to modify distinct cells. Utility magnitude is
    used only as a bounded detached weight; its sign determines whether the
    add-minus-remove score should increase or decrease.
    """

    if teacher_utility.shape != valid_mask.shape:
        raise ValueError("teacher_utility and valid_mask must share [B,M] shape")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    valid = valid_mask.bool()
    utility = teacher_utility.detach().float()
    if not torch.isfinite(utility[valid]).all():
        raise ValueError("valid local-cell utilities must be finite")
    for batch_index in range(int(valid.shape[0])):
        active_slots = replaced_slots[batch_index, valid[batch_index]]
        if active_slots.numel() != torch.unique(active_slots).numel():
            raise ValueError("local-cell utility candidates must modify distinct cells")

    pair_scores = counterfactual_pair_scores(
        policy_scores,
        candidate_positions,
        replaced_slots,
        baseline_positions,
        valid,
    ).float()
    active_count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=utility.dtype)
    mean_abs = utility.abs().masked_fill(~valid, 0.0).sum(dim=1, keepdim=True) / active_count
    scale = mean_abs.clamp_min(torch.finfo(utility.dtype).eps)
    weight = torch.tanh(utility.abs() / scale).masked_fill(~valid, 0.0)
    label = utility.sign()
    per_candidate = weight * F.softplus(-label * pair_scores / float(temperature))
    denominator = weight.sum(dim=1)
    active = denominator > 0.0
    if not bool(active.any().item()):
        finite_scores = torch.where(torch.isfinite(policy_scores), policy_scores, torch.zeros_like(policy_scores))
        return (finite_scores * 0.0).sum()
    per_sample = per_candidate.sum(dim=1) / denominator.clamp_min(torch.finfo(weight.dtype).eps)
    return per_sample[active].mean()


def _average_tie_ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.flatten().float()
    order = torch.argsort(values)
    sorted_values = values[order]
    ranks = values.new_empty(values.shape)
    start = 0
    while start < int(values.numel()):
        end = start + 1
        while end < int(values.numel()) and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        ranks[order[start:end]] = 0.5 * float(start + end - 1)
        start = end
    return ranks


def score_space_utility_alignment(
    policy_scores: torch.Tensor,
    loss: torch.Tensor,
    swap_incidence: torch.Tensor,
    hard_swap_utility: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    temperature: float,
) -> Dict[str, object]:
    """Audit the actual pair-score change induced through shared center scores."""
    if swap_incidence.shape[:2] != hard_swap_utility.shape or hard_swap_utility.shape != valid_mask.shape:
        raise ValueError("swap incidence, utility and valid mask must align")
    gradient = torch.autograd.grad(loss, policy_scores, retain_graph=True)[0].detach().float()
    normalized_utility = _normalized_signed_swap_utility(
        hard_swap_utility,
        valid_mask,
        temperature=temperature,
    )
    valid = valid_mask.bool()
    normalized_directions = []
    normalized_targets = []
    raw_directions = []
    condition_numbers = []
    per_sample_alignment = []
    for batch_index in range(int(policy_scores.shape[0])):
        active = valid[batch_index]
        if not bool(active.any()):
            continue
        audit_autocast_context = (
            torch.autocast(device_type=policy_scores.device.type, enabled=False)
            if policy_scores.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with audit_autocast_context:
            incidence = swap_incidence[batch_index, active].float()
            direction = incidence @ (-gradient[batch_index])
            target = normalized_utility[batch_index, active].float()
            gram = incidence @ incidence.transpose(0, 1)
            condition_number = float(torch.linalg.cond(gram).item())
        condition_numbers.append(condition_number)
        raw_directions.append(direction)
        sample_nonzero = hard_swap_utility[batch_index, active].detach().float() != 0
        sample_sign = (
            float(
                (
                    direction[sample_nonzero]
                    * hard_swap_utility[batch_index, active].detach().float()[sample_nonzero]
                    > 0
                )
                .float()
                .mean()
                .item()
            )
            if bool(sample_nonzero.any())
            else 1.0
        )
        if float(target.abs().max().item()) > 0.0:
            normalized_direction = direction / direction.abs().max().clamp_min(1.0e-12)
            normalized_target = target / target.abs().max().clamp_min(1.0e-12)
            normalized_directions.append(normalized_direction)
            normalized_targets.append(normalized_target)
            direction_ranks = _average_tie_ranks(normalized_direction)
            target_ranks = _average_tie_ranks(normalized_target)
            centered_direction = direction_ranks - direction_ranks.mean()
            centered_target = target_ranks - target_ranks.mean()
            denominator = centered_direction.norm() * centered_target.norm()
            sample_spearman = (
                float(((centered_direction @ centered_target) / denominator).item())
                if float(denominator.item()) > 0.0
                else 1.0
            )
            sample_error = float((normalized_direction - normalized_target).abs().max().item())
        else:
            sample_spearman = 1.0
            sample_error = 0.0
        per_sample_alignment.append(
            {
                "batch_index": batch_index,
                "candidate_count": int(active.sum().item()),
                "spearman": sample_spearman,
                "sign_agreement": sample_sign,
                "normalized_direction_max_abs_error": sample_error,
                "swap_gram_condition_number": condition_number,
            }
        )
    if not raw_directions:
        raise ValueError("score-space alignment requires at least one valid swap")

    direction = torch.cat(raw_directions)
    utility = hard_swap_utility.detach()[valid].float()
    nonzero = utility != 0
    sign = (
        ((direction[nonzero] * utility[nonzero]) > 0).float().mean()
        if bool(nonzero.any()) else direction.new_tensor(1.0)
    )
    spearman = min(item["spearman"] for item in per_sample_alignment)
    max_error = 0.0
    if normalized_directions:
        max_error = max(
            float((observed - expected).abs().max().item())
            for observed, expected in zip(normalized_directions, normalized_targets)
        )
    return {
        "spearman": float(spearman),
        "sign_agreement": float(sign.item()),
        "normalized_direction_max_abs_error": max_error,
        "score_space_direction_values": [float(value) for value in direction.cpu().tolist()],
        "normalized_utility_values": [
            float(value) for value in normalized_utility[valid].detach().cpu().tolist()
        ],
        "swap_gram_condition_numbers": condition_numbers,
        "per_sample_alignment": per_sample_alignment,
    }


def gradient_utility_alignment(
    policy_logits: torch.Tensor,
    loss: torch.Tensor,
    hard_swap_utility: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    """Gate whether a surrogate descent direction agrees with hard-swap utility."""
    gradient = torch.autograd.grad(loss, policy_logits, retain_graph=True)[0]
    while gradient.ndim > hard_swap_utility.ndim and gradient.shape[1] == 1:
        gradient = gradient.squeeze(1)
    if gradient.shape != hard_swap_utility.shape or gradient.shape != valid_mask.shape:
        raise ValueError("gradient, hard-swap utility and valid mask must align")
    descent = -gradient.detach()[valid_mask.bool()].float()
    utility = hard_swap_utility.detach()[valid_mask.bool()].float()
    if descent.numel() < 2 or torch.all(descent == descent[0]) or torch.all(utility == utility[0]):
        raise ValueError("alignment requires at least two non-constant valid candidates")
    descent_rank = torch.argsort(torch.argsort(descent)).float()
    utility_rank = torch.argsort(torch.argsort(utility)).float()
    spearman = F.cosine_similarity(
        descent_rank - descent_rank.mean(), utility_rank - utility_rank.mean(), dim=0
    )
    nonzero = utility != 0
    sign = ((descent[nonzero] * utility[nonzero]) > 0).float().mean() if nonzero.any() else descent.new_tensor(1.0)
    return {"spearman": float(spearman.item()), "sign_agreement": float(sign.item())}
