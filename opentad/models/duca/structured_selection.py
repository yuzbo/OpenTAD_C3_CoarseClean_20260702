from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class StructuredSelectionOutput:
    hard_occupancy: torch.Tensor
    soft_occupancy: torch.Tensor
    soft_slot_assignment: torch.Tensor
    selection_st: torch.Tensor
    selected_positions: torch.Tensor
    log_partition: torch.Tensor
    k: int
    max_unselected_hole: int
    temperature: float
    selection_scope: str = "full_window_non_streaming"


def exact_uniform_positions(temporal_len: int, k: int, *, device=None) -> torch.Tensor:
    """Return rounded-endpoint anchors with explicit round-half-to-even semantics."""

    temporal_len = int(temporal_len)
    k = int(k)
    if temporal_len < 0 or k < 0 or k > temporal_len:
        raise ValueError("exact-uniform requires 0 <= k <= temporal_len")
    if k == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    if k == 1:
        return torch.zeros((1,), device=device, dtype=torch.long)
    denominator = k - 1
    anchors = []
    for index in range(k):
        numerator = index * (temporal_len - 1)
        quotient, remainder = divmod(numerator, denominator)
        if 2 * remainder > denominator or (2 * remainder == denominator and quotient % 2 == 1):
            quotient += 1
        anchors.append(quotient)
    return torch.tensor(anchors, device=device, dtype=torch.long)


def exact_uniform_reference_scores(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Score valid positions by distance to the canonical exact-uniform anchors."""

    if not torch.is_tensor(scores) or scores.ndim != 2 or not scores.is_floating_point():
        raise ValueError("scores must be a floating-point [B,T] tensor")
    valid = valid_mask.to(device=scores.device, dtype=torch.bool)
    if valid.shape != scores.shape:
        raise ValueError("valid_mask must align with scores")
    reference = scores.new_zeros(scores.shape)
    for batch_idx in range(int(scores.shape[0])):
        valid_positions = torch.nonzero(valid[batch_idx], as_tuple=False).flatten()
        effective_k = min(max(int(k), 0), int(valid_positions.numel()))
        if effective_k == 0:
            continue
        anchors = exact_uniform_positions(
            int(valid_positions.numel()),
            effective_k,
            device=scores.device,
        )
        ranks = torch.arange(valid_positions.numel(), device=scores.device)
        distance = (ranks[:, None] - anchors[None, :]).abs().min(dim=1).values
        reference[batch_idx, valid_positions] = -distance.to(dtype=scores.dtype)
    return reference


def _validate_contract(logits: torch.Tensor, k: int, max_hole: int, temperature: float) -> tuple[int, int, float]:
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError("policy logits must be a [B,T] tensor")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all().item()):
        raise ValueError("policy logits must be finite floating-point values")
    k = int(k)
    max_hole = int(max_hole)
    temperature = float(temperature)
    if k < 0 or k > int(logits.shape[1]):
        raise ValueError("k must lie in [0,T]")
    if max_hole < 0:
        raise ValueError("max_unselected_hole must be non-negative")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    temporal_len = int(logits.shape[1])
    if temporal_len - k > (k + 1) * max_hole:
        raise ValueError(
            "infeasible exact-K/max-unselected-hole contract: "
            f"T={temporal_len}, K={k}, G={max_hole}"
        )
    return k, max_hole, temperature


def _hard_viterbi(logits: torch.Tensor, *, k: int, max_hole: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch, temporal_len = logits.shape
    work = logits.float()
    neg_inf = torch.tensor(float("-inf"), device=work.device, dtype=work.dtype)
    dp = torch.full((batch, k + 1, max_hole + 1), neg_inf, device=work.device, dtype=work.dtype)
    dp[:, 0, 0] = 0.0
    select_prev_gaps: list[torch.Tensor] = []

    for time_idx in range(temporal_len):
        if k > 0:
            best_select, best_gap = dp[:, :k, :].max(dim=2)
            selected_scores = best_select + work[:, time_idx, None]
            select_gap_zero = torch.cat(
                (torch.full((batch, 1), neg_inf, device=work.device, dtype=work.dtype), selected_scores),
                dim=1,
            )
            back_gap = torch.cat(
                (torch.zeros((batch, 1), device=work.device, dtype=torch.long), best_gap),
                dim=1,
            )
        else:
            select_gap_zero = torch.full((batch, 1), neg_inf, device=work.device, dtype=work.dtype)
            back_gap = torch.zeros((batch, 1), device=work.device, dtype=torch.long)
        skipped = dp[:, :, :max_hole] if max_hole > 0 else dp[:, :, :0]
        dp = torch.cat((select_gap_zero[:, :, None], skipped), dim=2)
        select_prev_gaps.append(back_gap)

    terminal_scores, terminal_gap = dp[:, k, :].max(dim=1)
    if not bool(torch.isfinite(terminal_scores).all().item()):
        raise RuntimeError("structured Viterbi failed to reach an exact-budget terminal state")

    hard = torch.zeros((batch, temporal_len), device=logits.device, dtype=logits.dtype)
    for batch_idx in range(batch):
        count = int(k)
        gap = int(terminal_gap[batch_idx].item())
        for time_idx in range(temporal_len - 1, -1, -1):
            if gap == 0:
                hard[batch_idx, time_idx] = 1.0
                gap = int(select_prev_gaps[time_idx][batch_idx, count].item())
                count -= 1
            else:
                gap -= 1
        if count != 0:
            raise RuntimeError("structured Viterbi backtracking did not recover exactly K selections")
    positions = torch.arange(temporal_len, device=logits.device, dtype=torch.long)[None, :]
    positions = positions.expand(batch, -1)[hard.bool()].reshape(batch, k)
    return hard, positions


def _soft_forward_backward(
    logits: torch.Tensor,
    *,
    k: int,
    max_hole: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, temporal_len = logits.shape
    work = logits.float() / float(temperature)
    work = work - work.detach().amax(dim=1, keepdim=True)
    work = work.clamp(min=-80.0, max=0.0)
    # A finite sentinel avoids undefined gradients from logaddexp(-inf, -inf)
    # while remaining far outside any attainable path score.
    neg_inf = torch.tensor(-1.0e9, device=work.device, dtype=work.dtype)
    alpha = torch.full((batch, k + 1, max_hole + 1), neg_inf, device=work.device, dtype=work.dtype)
    alpha[:, 0, 0] = 0.0
    alphas = [alpha]
    for time_idx in range(temporal_len):
        if k > 0:
            select_score = torch.logsumexp(alpha[:, :k, :], dim=2) + work[:, time_idx, None]
            select_zero = torch.cat(
                (torch.full((batch, 1), neg_inf, device=work.device, dtype=work.dtype), select_score),
                dim=1,
            )
        else:
            select_zero = torch.full((batch, 1), neg_inf, device=work.device, dtype=work.dtype)
        skipped = alpha[:, :, :max_hole] if max_hole > 0 else alpha[:, :, :0]
        alpha = torch.cat((select_zero[:, :, None], skipped), dim=2)
        alphas.append(alpha)

    beta = torch.full_like(alpha, neg_inf)
    beta[:, k, :] = 0.0
    betas: list[torch.Tensor] = [beta]
    for time_idx in range(temporal_len - 1, -1, -1):
        if k > 0:
            select_by_count = torch.cat(
                (
                    beta[:, 1:, 0] + work[:, time_idx, None],
                    torch.full((batch, 1), neg_inf, device=work.device, dtype=work.dtype),
                ),
                dim=1,
            )
        else:
            select_by_count = torch.full((batch, 1), neg_inf, device=work.device, dtype=work.dtype)
        select_candidate = select_by_count[:, :, None].expand(-1, -1, max_hole + 1)
        if max_hole > 0:
            skip_candidate = torch.cat(
                (
                    beta[:, :, 1:],
                    torch.full((batch, k + 1, 1), neg_inf, device=work.device, dtype=work.dtype),
                ),
                dim=2,
            )
        else:
            skip_candidate = torch.full_like(beta, neg_inf)
        beta = torch.logaddexp(select_candidate, skip_candidate)
        betas.append(beta)
    betas.reverse()

    log_partition = betas[0][:, 0, 0]
    if not bool(torch.isfinite(log_partition).all().item()):
        raise RuntimeError("structured soft DP failed to reach an exact-budget terminal state")
    if k == 0:
        slots = work.new_zeros((batch, 0, temporal_len))
    else:
        slot_steps = []
        for time_idx in range(temporal_len):
            transition_log_prob = (
                alphas[time_idx][:, :k, :]
                + work[:, time_idx, None, None]
                + betas[time_idx + 1][:, 1:, 0, None]
                - log_partition[:, None, None]
            )
            slot_steps.append(torch.exp(torch.logsumexp(transition_log_prob, dim=2)))
        slots = torch.stack(slot_steps, dim=2)
        slot_mass = slots.sum(dim=2, keepdim=True).clamp_min(torch.finfo(slots.dtype).tiny)
        slots = slots / slot_mass
    occupancy = slots.sum(dim=1)
    return occupancy.to(dtype=logits.dtype), slots.to(dtype=logits.dtype), log_partition


def _structured_log_partition(
    logits: torch.Tensor,
    *,
    k: int,
    max_hole: int,
    temperature: float,
    selection_allowed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return log Z for exact-K/max-hole paths, optionally forbidding selections."""

    batch, temporal_len = logits.shape
    work = logits.float() / float(temperature)
    if selection_allowed is None:
        allowed = torch.ones_like(logits, dtype=torch.bool)
    else:
        allowed = selection_allowed.to(device=logits.device, dtype=torch.bool)
        if allowed.shape != logits.shape:
            raise ValueError("selection_allowed must align with logits")
    neg_inf = work.new_tensor(-1.0e9)

    def safe_logsumexp(values: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.logsumexp(values, dim=dim)

    alpha = work.new_full((batch, k + 1, max_hole + 1), neg_inf.item())
    alpha[:, 0, 0] = 0.0
    for time_idx in range(temporal_len):
        if k > 0:
            selected = safe_logsumexp(alpha[:, :k, :], dim=2) + work[:, time_idx, None]
            selected = selected.masked_fill(~allowed[:, time_idx, None], neg_inf.item())
            select_zero = torch.cat((work.new_full((batch, 1), neg_inf.item()), selected), dim=1)
        else:
            select_zero = work.new_full((batch, 1), neg_inf.item())
        skipped = alpha[:, :, :max_hole] if max_hole > 0 else alpha[:, :, :0]
        alpha = torch.cat((select_zero[:, :, None], skipped), dim=2)
    return safe_logsumexp(alpha[:, k, :], dim=1)


def structured_local_coverage_probability(
    policy_logits: torch.Tensor,
    event_mask: torch.Tensor,
    *,
    k: int,
    max_unselected_hole: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Exact probability that a structured path selects inside each event mask.

    ``event_mask`` is ``[B,N,T]``. The result is ``[B,N]`` and is computed as
    ``1 - Z(no selection in event) / Z`` under the same exact-K/max-hole path
    distribution used by :func:`global_structured_topk`.
    """

    k, max_hole, temperature = _validate_contract(
        policy_logits, k, max_unselected_hole, temperature
    )
    events = event_mask.to(device=policy_logits.device, dtype=torch.bool)
    if events.ndim != 3 or events.shape[0] != policy_logits.shape[0] or events.shape[2] != policy_logits.shape[1]:
        raise ValueError("event_mask must be [B,N,T] and align with policy_logits")
    log_z = _structured_log_partition(
        policy_logits, k=k, max_hole=max_hole, temperature=temperature
    )
    probabilities = []
    for event_idx in range(events.shape[1]):
        log_z_miss = _structured_log_partition(
            policy_logits,
            k=k,
            max_hole=max_hole,
            temperature=temperature,
            selection_allowed=~events[:, event_idx],
        )
        log_miss = (log_z_miss - log_z).clamp(max=0.0)
        probability = -torch.expm1(log_miss)
        impossible_miss = log_z_miss <= -5.0e8
        probabilities.append(torch.where(impossible_miss, torch.ones_like(probability), probability))
    if not probabilities:
        return policy_logits.new_zeros((policy_logits.shape[0], 0))
    return torch.stack(probabilities, dim=1).to(dtype=policy_logits.dtype).clamp(0.0, 1.0)


def global_structured_topk(
    policy_logits: torch.Tensor,
    *,
    k: int,
    max_unselected_hole: int,
    temperature: float = 1.0,
    training: bool = False,
) -> StructuredSelectionOutput:
    """Full-window exact-budget selection with a shared hard/soft structured policy."""

    k, max_hole, temperature = _validate_contract(
        policy_logits,
        k,
        max_unselected_hole,
        temperature,
    )
    hard, positions = _hard_viterbi(policy_logits.detach(), k=k, max_hole=max_hole)
    if training:
        soft, slots, log_partition = _soft_forward_backward(
            policy_logits,
            k=k,
            max_hole=max_hole,
            temperature=temperature,
        )
        selection_st = hard + (soft - soft.detach())
    else:
        soft = hard
        slots = policy_logits.new_zeros((policy_logits.shape[0], k, policy_logits.shape[1]))
        if k > 0:
            slots.scatter_(2, positions[:, :, None], 1.0)
        selection_st = hard
        log_partition = policy_logits.new_full((policy_logits.shape[0],), float("nan"))
    return StructuredSelectionOutput(
        hard_occupancy=hard,
        soft_occupancy=soft,
        soft_slot_assignment=slots,
        selection_st=selection_st,
        selected_positions=positions,
        log_partition=log_partition,
        k=k,
        max_unselected_hole=max_hole,
        temperature=temperature,
    )


__all__ = [
    "StructuredSelectionOutput",
    "exact_uniform_positions",
    "exact_uniform_reference_scores",
    "global_structured_topk",
    "structured_local_coverage_probability",
]
