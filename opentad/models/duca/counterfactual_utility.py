from __future__ import annotations

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
        base = base[base >= 0]
        if base.numel() != selected_positions.shape[1] or torch.unique(base).numel() != base.numel():
            raise ValueError("baseline selection must be exact-K, unique and non-negative")
        valid_length = int(valid_mask[b].sum().item())
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
        if out_index == 0:
            raise RuntimeError("no feasible hard one-swap counterfactual candidate")
    return {
        "candidate_selections": candidates,
        "candidate_positions": candidate_positions,
        "replaced_slots": replaced_slots,
        "candidate_valid": candidate_valid,
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
    """Distill detached hard-swap utilities into deploy-time policy scores."""
    if policy_scores.shape != teacher_utility.shape or policy_scores.shape != valid_mask.shape:
        raise ValueError("policy_scores, teacher_utility and valid_mask must have identical [B,T] shapes")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    valid = valid_mask.bool()
    if not valid.any(dim=1).all():
        raise ValueError("every sample must contain at least one valid counterfactual candidate")
    student = policy_scores.float()
    teacher = teacher_utility.detach().to(device=policy_scores.device, dtype=torch.float32)
    if not torch.isfinite(teacher[valid]).all():
        raise ValueError("valid counterfactual utilities must be finite")
    neg = torch.finfo(student.dtype).min
    student_logits = (student / temperature).masked_fill(~valid, neg)
    teacher_logits = (teacher / temperature).masked_fill(~valid, neg)
    target = torch.softmax(teacher_logits, dim=-1)
    return -(target * torch.log_softmax(student_logits, dim=-1)).sum(dim=-1).mean()


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
    safe_add = candidate_positions.clamp(min=0)
    safe_slot = replaced_slots.clamp(min=0)
    if torch.any(safe_add[valid] >= policy_scores.shape[1]) or torch.any(safe_slot[valid] >= baseline_positions.shape[1]):
        raise ValueError("counterfactual add position or remove slot is out of range")
    remove_positions = torch.gather(baseline_positions, 1, safe_slot)
    add_scores = torch.gather(policy_scores, 1, safe_add)
    remove_scores = torch.gather(policy_scores, 1, remove_positions)
    return (add_scores - remove_scores).masked_fill(~valid, 0.0)


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
