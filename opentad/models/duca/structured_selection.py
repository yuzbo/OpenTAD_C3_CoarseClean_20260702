from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


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


@dataclass(frozen=True)
class PhysicalExactKSelectionOutput:
    hard_occupancy: torch.Tensor
    hard_slot_assignment: torch.Tensor
    hard_positions: torch.Tensor
    hard_slot_mask: torch.Tensor
    soft_occupancy: torch.Tensor
    soft_slot_assignment: torch.Tensor
    selection_st: torch.Tensor
    log_partition: torch.Tensor
    edge_count: torch.Tensor
    effective_k: torch.Tensor
    max_gap_seconds: torch.Tensor
    temperature: float
    selection_scope: str = "full_window_offline_physical_exact_k"


@dataclass(frozen=True)
class PhysicalExactKGraph:
    predecessor_index: torch.Tensor
    predecessor_valid: torch.Tensor
    successor_index: torch.Tensor
    successor_valid: torch.Tensor
    source_valid: torch.Tensor
    sink_valid: torch.Tensor
    max_gap_seconds: torch.Tensor
    edge_count: int


@dataclass(frozen=True)
class PhysicalExactKHardOutput:
    hard_occupancy: torch.Tensor
    hard_slot_assignment: torch.Tensor
    hard_positions: torch.Tensor
    hard_slot_mask: torch.Tensor
    edge_count: torch.Tensor
    effective_k: torch.Tensor
    max_gap_seconds: torch.Tensor


@dataclass(frozen=True)
class PhysicalExactKSoftOutput:
    soft_occupancy: torch.Tensor
    soft_slot_assignment: torch.Tensor
    log_partition: torch.Tensor
    edge_count: torch.Tensor
    effective_k: torch.Tensor
    max_gap_seconds: torch.Tensor
    temperature: float


@dataclass(frozen=True)
class LocalCellSelectionOutput:
    hard_occupancy: torch.Tensor
    soft_occupancy: torch.Tensor
    soft_slot_assignment: torch.Tensor
    selection_st: torch.Tensor
    selected_positions: torch.Tensor
    log_partition: torch.Tensor
    anchor_positions: torch.Tensor
    cell_starts: torch.Tensor
    cell_ends: torch.Tensor
    k: int
    max_unselected_hole: int
    temperature: float
    force_exact_uniform: bool
    selection_scope: str = "full_window_non_streaming_local_cells"


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


def _validate_physical_axis_row(
    physical_seconds: torch.Tensor,
    valid_mask: torch.Tensor,
) -> int:
    if physical_seconds.ndim != 1 or valid_mask.ndim != 1:
        raise ValueError("physical time and validity rows must be one-dimensional")
    if physical_seconds.shape != valid_mask.shape:
        raise ValueError("physical time and validity rows must align")
    valid = valid_mask.to(dtype=torch.bool)
    valid_len = int(valid.sum().item())
    expected = torch.arange(
        valid_len,
        device=valid.device,
        dtype=torch.long,
    )
    observed = torch.nonzero(valid, as_tuple=False).flatten()
    if not torch.equal(observed, expected):
        raise ValueError("physical exact-K selection requires a contiguous valid prefix")
    if valid_len == 0:
        return 0
    active = physical_seconds[:valid_len]
    if not active.is_floating_point() or not bool(torch.isfinite(active).all().item()):
        raise ValueError("valid physical seconds must be finite floating-point values")
    if valid_len > 1 and not bool(torch.all(active[1:] > active[:-1]).item()):
        raise ValueError("valid physical seconds must be strictly increasing")
    return valid_len


def physical_exact_uniform_gap_cap(
    physical_seconds: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
) -> torch.Tensor:
    """Return each sample's exact-uniform-reference maximum interval in seconds."""

    if physical_seconds.ndim != 2 or not physical_seconds.is_floating_point():
        raise ValueError("physical_seconds must be a floating-point [B,T] tensor")
    valid = valid_mask.to(device=physical_seconds.device, dtype=torch.bool)
    if valid.shape != physical_seconds.shape:
        raise ValueError("valid_mask must align with physical_seconds")
    k = int(k)
    if k < 1:
        raise ValueError("k must be positive")
    caps = []
    for batch_idx in range(int(physical_seconds.shape[0])):
        valid_len = _validate_physical_axis_row(
            physical_seconds[batch_idx],
            valid[batch_idx],
        )
        effective_k = min(k, valid_len)
        if valid_len == 0 or effective_k == 0:
            caps.append(physical_seconds.new_zeros(()))
            continue
        row = physical_seconds[batch_idx, :valid_len]
        anchors = exact_uniform_positions(
            valid_len,
            effective_k,
            device=physical_seconds.device,
        )
        selected = row[anchors]
        intervals = [selected[0] - row[0], row[-1] - selected[-1]]
        if effective_k > 1:
            intervals.append(selected[1:] - selected[:-1])
        caps.append(torch.cat([item.reshape(-1) for item in intervals]).max())
    return torch.stack(caps)


def _build_physical_exact_k_graph(
    physical_seconds: torch.Tensor,
    max_gap_seconds: torch.Tensor,
) -> PhysicalExactKGraph:
    """Build the one graph object consumed by both Viterbi and Gibbs paths."""

    temporal_len = int(physical_seconds.numel())
    if temporal_len == 0:
        raise ValueError("physical exact-K graph requires at least one valid candidate")
    cap = max_gap_seconds.to(
        device=physical_seconds.device,
        dtype=physical_seconds.dtype,
    )
    if cap.ndim != 0 or not bool(torch.isfinite(cap).item()) or float(cap.item()) < 0.0:
        raise ValueError("max_gap_seconds must be a finite non-negative scalar")
    tolerance = max(1.0e-9, 8.0 * torch.finfo(physical_seconds.dtype).eps)
    lower = physical_seconds - cap - tolerance
    left = torch.searchsorted(physical_seconds, lower, right=False)
    positions = torch.arange(
        temporal_len,
        device=physical_seconds.device,
        dtype=torch.long,
    )
    predecessor_span = positions - left
    max_span = int(predecessor_span.max().item())
    if max_span == 0:
        predecessor_index = positions.new_zeros((temporal_len, 0))
        predecessor_valid = torch.zeros(
            (temporal_len, 0),
            device=physical_seconds.device,
            dtype=torch.bool,
        )
    else:
        columns = torch.arange(
            max_span,
            device=physical_seconds.device,
            dtype=torch.long,
        )
        predecessor_index = left[:, None] + columns[None, :]
        predecessor_valid = columns[None, :] < predecessor_span[:, None]
        predecessor_index = predecessor_index.clamp(max=temporal_len - 1)

    upper = physical_seconds + cap + tolerance
    right = torch.searchsorted(physical_seconds, upper, right=True)
    successor_span = right - positions - 1
    max_successors = int(successor_span.max().item())
    if max_successors == 0:
        successor_index = positions.new_zeros((temporal_len, 0))
        successor_valid = torch.zeros(
            (temporal_len, 0),
            device=physical_seconds.device,
            dtype=torch.bool,
        )
    else:
        columns = torch.arange(
            max_successors,
            device=physical_seconds.device,
            dtype=torch.long,
        )
        successor_index = positions[:, None] + 1 + columns[None, :]
        successor_valid = columns[None, :] < successor_span[:, None]
        successor_index = successor_index.clamp(max=temporal_len - 1)

    source_valid = physical_seconds - physical_seconds[0] <= cap + tolerance
    sink_valid = physical_seconds[-1] - physical_seconds <= cap + tolerance
    predecessor_edges = int(predecessor_valid.sum().item())
    successor_edges = int(successor_valid.sum().item())
    if predecessor_edges != successor_edges:
        raise RuntimeError("physical exact-K predecessor/successor edge tables disagree")
    return PhysicalExactKGraph(
        predecessor_index=predecessor_index,
        predecessor_valid=predecessor_valid,
        successor_index=successor_index,
        successor_valid=successor_valid,
        source_valid=source_valid,
        sink_valid=sink_valid,
        max_gap_seconds=cap,
        edge_count=(
            predecessor_edges
            + int(source_valid.sum().item())
            + int(sink_valid.sum().item())
        ),
    )


def _safe_masked_logsumexp(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    valid = valid.to(device=values.device, dtype=torch.bool)
    if valid.shape != values.shape:
        valid = valid.expand_as(values)
    reachable = valid & torch.isfinite(values)
    has_reachable = reachable.any(dim=dim)
    finite_floor = values.new_full((), -1.0e30)
    reduced = torch.logsumexp(
        torch.where(reachable, values, finite_floor),
        dim=dim,
    )
    return torch.where(
        has_reachable,
        reduced,
        values.new_full(reduced.shape, float("-inf")),
    )


def _physical_row_viterbi(
    node_log_probs: torch.Tensor,
    *,
    k: int,
    graph: PhysicalExactKGraph,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    temporal_len = int(node_log_probs.numel())
    if k == 0:
        return (
            node_log_probs.new_zeros((temporal_len,)),
            torch.empty((0,), device=node_log_probs.device, dtype=torch.long),
            node_log_probs.new_zeros((0, temporal_len)),
        )
    if graph.source_valid.numel() != temporal_len:
        raise ValueError("physical exact-K graph length does not match node scores")
    scores = node_log_probs.float()
    neg_inf = scores.new_full((), float("-inf"))
    dp = scores.new_full((k, temporal_len), float("-inf"))
    back = torch.full(
        (k, temporal_len),
        -1,
        device=scores.device,
        dtype=torch.long,
    )
    ranks = torch.full(
        (k, temporal_len),
        -1,
        device=scores.device,
        dtype=torch.long,
    )
    dp[0] = torch.where(graph.source_valid, scores, neg_inf)
    initial_positions = torch.nonzero(graph.source_valid, as_tuple=False).flatten()
    ranks[0, initial_positions] = torch.arange(
        initial_positions.numel(),
        device=scores.device,
        dtype=torch.long,
    )
    for slot_idx in range(1, k):
        if graph.predecessor_index.shape[1] == 0:
            continue
        candidates = dp[slot_idx - 1][graph.predecessor_index]
        predecessor_ranks = ranks[slot_idx - 1][graph.predecessor_index]
        candidate_valid = (
            graph.predecessor_valid
            & torch.isfinite(candidates)
            & (predecessor_ranks >= 0)
        )
        candidates = candidates.masked_fill(~candidate_valid, float("-inf"))
        best_value = candidates.max(dim=1).values
        best_score_mask = candidate_valid & (candidates == best_value[:, None])
        rank_sentinel = torch.iinfo(torch.long).max
        tied_ranks = predecessor_ranks.masked_fill(~best_score_mask, rank_sentinel)
        best_rank, best_offset = tied_ranks.min(dim=1)
        reachable = torch.isfinite(best_value) & (best_rank != rank_sentinel)
        best_predecessor = graph.predecessor_index.gather(
            1,
            best_offset[:, None],
        ).squeeze(1)
        dp[slot_idx] = torch.where(
            reachable,
            scores + best_value,
            neg_inf,
        )
        back[slot_idx] = torch.where(
            reachable,
            best_predecessor,
            back[slot_idx],
        )
        reachable_positions = torch.nonzero(reachable, as_tuple=False).flatten()
        if reachable_positions.numel():
            parent_rank = ranks[slot_idx - 1, best_predecessor[reachable_positions]]
            lex_keys = parent_rank * (temporal_len + 1) + reachable_positions
            order = torch.argsort(lex_keys, stable=True)
            ranks[slot_idx, reachable_positions[order]] = torch.arange(
                reachable_positions.numel(),
                device=scores.device,
                dtype=torch.long,
            )
    terminal_valid = graph.sink_valid & torch.isfinite(dp[k - 1]) & (ranks[k - 1] >= 0)
    terminal_value = dp[k - 1].masked_fill(~terminal_valid, float("-inf")).max()
    if not bool(torch.isfinite(terminal_value).item()):
        raise RuntimeError(
            "physical exact-K Viterbi could not reach a legal source-to-sink path"
        )
    tied_terminals = terminal_valid & (dp[k - 1] == terminal_value)
    terminal_ranks = ranks[k - 1].masked_fill(
        ~tied_terminals,
        torch.iinfo(torch.long).max,
    )
    terminal_index = terminal_ranks.argmin()
    positions = torch.empty(
        (k,),
        device=node_log_probs.device,
        dtype=torch.long,
    )
    positions[-1] = terminal_index
    for slot_idx in range(k - 1, 0, -1):
        predecessor = back[slot_idx, positions[slot_idx]]
        if int(predecessor.item()) < 0:
            raise RuntimeError("physical exact-K Viterbi backtracking failed")
        positions[slot_idx - 1] = predecessor
    if k > 1 and not bool(torch.all(positions[1:] > positions[:-1]).item()):
        raise RuntimeError("physical exact-K Viterbi produced unordered positions")
    hard = node_log_probs.new_zeros((temporal_len,))
    hard.scatter_(0, positions, 1.0)
    hard_slots = node_log_probs.new_zeros((k, temporal_len))
    hard_slots.scatter_(1, positions[:, None], 1.0)
    return hard, positions, hard_slots


def _physical_row_forward_backward(
    node_log_probs: torch.Tensor,
    *,
    k: int,
    graph: PhysicalExactKGraph,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    temporal_len = int(node_log_probs.numel())
    if k == 0:
        return (
            node_log_probs.new_zeros((temporal_len,)),
            node_log_probs.new_zeros((0, temporal_len)),
            node_log_probs.float().new_zeros(()),
        )
    if graph.source_valid.numel() != temporal_len:
        raise ValueError("physical exact-K graph length does not match node scores")
    scores = node_log_probs.float() / float(temperature)
    alpha_rows = []
    alpha = torch.where(
        graph.source_valid,
        scores,
        scores.new_full(scores.shape, float("-inf")),
    )
    alpha_rows.append(alpha)
    for _slot_idx in range(1, k):
        if graph.predecessor_index.shape[1] == 0:
            alpha = scores.new_full(scores.shape, float("-inf"))
        else:
            candidates = alpha[graph.predecessor_index]
            predecessor_mass = _safe_masked_logsumexp(
                candidates,
                graph.predecessor_valid,
                dim=1,
            )
            alpha = scores + predecessor_mass
        alpha_rows.append(alpha)
    alpha_table = torch.stack(alpha_rows, dim=0)
    log_partition = _safe_masked_logsumexp(
        alpha_table[-1],
        graph.sink_valid,
        dim=0,
    )
    if not bool(torch.isfinite(log_partition).item()):
        raise RuntimeError(
            "physical exact-K forward-backward could not reach a legal source-to-sink path"
        )

    beta_rows = [scores.new_empty((temporal_len,)) for _ in range(k)]
    beta = torch.where(
        graph.sink_valid,
        scores.new_zeros(scores.shape),
        scores.new_full(scores.shape, float("-inf")),
    )
    beta_rows[k - 1] = beta
    for slot_idx in range(k - 2, -1, -1):
        if graph.successor_index.shape[1] == 0:
            beta = scores.new_full(scores.shape, float("-inf"))
        else:
            candidates = (
                scores[graph.successor_index]
                + beta[graph.successor_index]
            )
            beta = _safe_masked_logsumexp(
                candidates,
                graph.successor_valid,
                dim=1,
            )
        beta_rows[slot_idx] = beta
    beta_table = torch.stack(beta_rows, dim=0)
    log_marginal = alpha_table + beta_table - log_partition
    finite = torch.isfinite(log_marginal)
    slots = torch.where(
        finite,
        torch.exp(log_marginal),
        log_marginal.new_zeros(()),
    )
    row_mass = slots.sum(dim=1)
    if not torch.allclose(
        row_mass,
        torch.ones_like(row_mass),
        atol=2.0e-4,
        rtol=2.0e-4,
    ):
        raise RuntimeError("physical exact-K slot marginals do not sum to one")
    occupancy = slots.sum(dim=0)
    if bool(torch.any(occupancy > 1.0 + 5.0e-4).item()):
        raise RuntimeError("physical exact-K slot marginals exceed unit column occupancy")
    expectations = (
        slots
        * torch.arange(
            temporal_len,
            device=slots.device,
            dtype=slots.dtype,
        )[None, :]
    ).sum(dim=1)
    if k > 1 and not bool(torch.all(expectations[1:] > expectations[:-1]).item()):
        raise RuntimeError("physical exact-K slot expectations are not strictly ordered")
    return occupancy, slots, log_partition


def _prepare_physical_exact_k_batch(
    node_log_probs: torch.Tensor,
    physical_seconds: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
    max_gap_seconds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    if (
        node_log_probs.ndim != 2
        or not node_log_probs.is_floating_point()
        or physical_seconds.shape != node_log_probs.shape
    ):
        raise ValueError(
            "node_log_probs and physical_seconds must be aligned floating-point [B,T] tensors"
        )
    valid = valid_mask.to(device=node_log_probs.device, dtype=torch.bool)
    physical_seconds = physical_seconds.to(
        device=node_log_probs.device,
        dtype=torch.float64,
    )
    if valid.shape != node_log_probs.shape:
        raise ValueError("valid_mask must align with node_log_probs")
    k = int(k)
    if k < 1:
        raise ValueError("k must be positive")
    for batch_idx in range(int(node_log_probs.shape[0])):
        valid_len = _validate_physical_axis_row(
            physical_seconds[batch_idx],
            valid[batch_idx],
        )
        if valid_len == 0:
            raise ValueError("physical exact-K selection requires at least one valid candidate")
        if not bool(
            torch.isfinite(node_log_probs[batch_idx, :valid_len]).all().item()
        ):
            raise ValueError("valid node_log_probs must be finite")
    if max_gap_seconds is None:
        caps = physical_exact_uniform_gap_cap(
            physical_seconds,
            valid,
            k=k,
        )
    else:
        caps = torch.as_tensor(
            max_gap_seconds,
            device=node_log_probs.device,
            dtype=torch.float64,
        ).reshape(-1)
        if caps.numel() == 1 and node_log_probs.shape[0] > 1:
            caps = caps.expand(node_log_probs.shape[0])
        if caps.shape != (node_log_probs.shape[0],):
            raise ValueError("max_gap_seconds must be scalar or [B]")
        if not bool(torch.isfinite(caps).all().item()) or bool(torch.any(caps < 0).item()):
            raise ValueError("max_gap_seconds must contain finite non-negative values")
    return physical_seconds, valid, k, caps


def _pad_physical_hard_row(
    hard_valid: torch.Tensor,
    active_positions: torch.Tensor,
    active_slots: torch.Tensor,
    *,
    temporal_len: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    effective_k = int(active_positions.numel())
    valid_len = int(hard_valid.numel())
    hard = F.pad(hard_valid, (0, temporal_len - valid_len))
    hard_slots = F.pad(
        active_slots,
        (0, temporal_len - valid_len, 0, k - effective_k),
    )
    positions = torch.full(
        (k,),
        -1,
        device=active_positions.device,
        dtype=torch.long,
    )
    positions[:effective_k] = active_positions
    slot_mask = torch.zeros(
        (k,),
        device=active_positions.device,
        dtype=torch.bool,
    )
    slot_mask[:effective_k] = True
    return hard, hard_slots, positions, slot_mask


def _pad_physical_soft_row(
    soft_valid: torch.Tensor,
    active_slots: torch.Tensor,
    *,
    temporal_len: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    effective_k, valid_len = active_slots.shape
    return (
        F.pad(soft_valid, (0, temporal_len - valid_len)),
        F.pad(
            active_slots,
            (0, temporal_len - valid_len, 0, k - effective_k),
        ),
    )


def physical_exact_k_select(
    node_log_probs: torch.Tensor,
    physical_seconds: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
    max_gap_seconds: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> PhysicalExactKSelectionOutput:
    """Return hard Viterbi and soft Gibbs assignments from the same graph."""

    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    physical_seconds, valid, k, caps = _prepare_physical_exact_k_batch(
        node_log_probs,
        physical_seconds,
        valid_mask,
        k=k,
        max_gap_seconds=max_gap_seconds,
    )

    hard_rows = []
    hard_slot_rows = []
    position_rows = []
    slot_mask_rows = []
    soft_rows = []
    slot_rows = []
    partition_rows = []
    edge_counts = []
    effective_rows = []
    temporal_len = int(node_log_probs.shape[1])
    for batch_idx in range(int(node_log_probs.shape[0])):
        valid_len = int(valid[batch_idx].sum().item())
        effective_k = min(k, valid_len)
        effective_rows.append(effective_k)
        row_scores = node_log_probs[batch_idx, :valid_len]
        row_seconds = physical_seconds[batch_idx, :valid_len]
        graph = _build_physical_exact_k_graph(row_seconds, caps[batch_idx])
        hard_valid, active_positions, active_hard_slots = _physical_row_viterbi(
            row_scores.detach(),
            k=effective_k,
            graph=graph,
        )
        soft_valid, active_soft_slots, partition = _physical_row_forward_backward(
            row_scores,
            k=effective_k,
            graph=graph,
            temperature=temperature,
        )
        hard, hard_slots, positions, slot_mask = _pad_physical_hard_row(
            hard_valid,
            active_positions,
            active_hard_slots,
            temporal_len=temporal_len,
            k=k,
        )
        soft, slots = _pad_physical_soft_row(
            soft_valid,
            active_soft_slots,
            temporal_len=temporal_len,
            k=k,
        )
        hard_rows.append(hard)
        hard_slot_rows.append(hard_slots)
        position_rows.append(positions)
        slot_mask_rows.append(slot_mask)
        soft_rows.append(soft)
        slot_rows.append(slots)
        partition_rows.append(partition)
        edge_counts.append(graph.edge_count)

    hard_occupancy = torch.stack(hard_rows, dim=0)
    hard_slot_assignment = torch.stack(hard_slot_rows, dim=0)
    soft_occupancy = torch.stack(soft_rows, dim=0)
    soft_slot_assignment = torch.stack(slot_rows, dim=0)
    selection_st = (
        hard_slot_assignment
        + (soft_slot_assignment - soft_slot_assignment.detach())
    )
    return PhysicalExactKSelectionOutput(
        hard_occupancy=hard_occupancy,
        hard_slot_assignment=hard_slot_assignment,
        hard_positions=torch.stack(position_rows, dim=0),
        hard_slot_mask=torch.stack(slot_mask_rows, dim=0),
        soft_occupancy=soft_occupancy,
        soft_slot_assignment=soft_slot_assignment,
        selection_st=selection_st,
        log_partition=torch.stack(partition_rows, dim=0),
        edge_count=torch.tensor(
            edge_counts,
            device=node_log_probs.device,
            dtype=torch.long,
        ),
        effective_k=torch.tensor(
            effective_rows,
            device=node_log_probs.device,
            dtype=torch.long,
        ),
        max_gap_seconds=caps.to(
            device=node_log_probs.device,
            dtype=node_log_probs.dtype,
        ),
        temperature=temperature,
    )


def physical_exact_k_viterbi(
    node_log_probs: torch.Tensor,
    physical_seconds: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
    max_gap_seconds: torch.Tensor | None = None,
) -> PhysicalExactKHardOutput:
    physical_seconds, valid, k, caps = _prepare_physical_exact_k_batch(
        node_log_probs,
        physical_seconds,
        valid_mask,
        k=k,
        max_gap_seconds=max_gap_seconds,
    )
    temporal_len = int(node_log_probs.shape[1])
    hard_rows = []
    hard_slot_rows = []
    position_rows = []
    slot_mask_rows = []
    edge_counts = []
    effective_rows = []
    for batch_idx in range(int(node_log_probs.shape[0])):
        valid_len = int(valid[batch_idx].sum().item())
        effective_k = min(k, valid_len)
        graph = _build_physical_exact_k_graph(
            physical_seconds[batch_idx, :valid_len],
            caps[batch_idx],
        )
        hard_valid, active_positions, active_slots = _physical_row_viterbi(
            node_log_probs[batch_idx, :valid_len].detach(),
            k=effective_k,
            graph=graph,
        )
        hard, hard_slots, positions, slot_mask = _pad_physical_hard_row(
            hard_valid,
            active_positions,
            active_slots,
            temporal_len=temporal_len,
            k=k,
        )
        hard_rows.append(hard)
        hard_slot_rows.append(hard_slots)
        position_rows.append(positions)
        slot_mask_rows.append(slot_mask)
        edge_counts.append(graph.edge_count)
        effective_rows.append(effective_k)
    return PhysicalExactKHardOutput(
        hard_occupancy=torch.stack(hard_rows, dim=0),
        hard_slot_assignment=torch.stack(hard_slot_rows, dim=0),
        hard_positions=torch.stack(position_rows, dim=0),
        hard_slot_mask=torch.stack(slot_mask_rows, dim=0),
        edge_count=torch.tensor(edge_counts, device=node_log_probs.device, dtype=torch.long),
        effective_k=torch.tensor(effective_rows, device=node_log_probs.device, dtype=torch.long),
        max_gap_seconds=caps.to(device=node_log_probs.device, dtype=node_log_probs.dtype),
    )


def physical_exact_k_forward_backward(
    node_log_probs: torch.Tensor,
    physical_seconds: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
    max_gap_seconds: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> PhysicalExactKSoftOutput:
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    physical_seconds, valid, k, caps = _prepare_physical_exact_k_batch(
        node_log_probs,
        physical_seconds,
        valid_mask,
        k=k,
        max_gap_seconds=max_gap_seconds,
    )
    temporal_len = int(node_log_probs.shape[1])
    soft_rows = []
    slot_rows = []
    partition_rows = []
    edge_counts = []
    effective_rows = []
    for batch_idx in range(int(node_log_probs.shape[0])):
        valid_len = int(valid[batch_idx].sum().item())
        effective_k = min(k, valid_len)
        graph = _build_physical_exact_k_graph(
            physical_seconds[batch_idx, :valid_len],
            caps[batch_idx],
        )
        soft_valid, active_slots, partition = _physical_row_forward_backward(
            node_log_probs[batch_idx, :valid_len],
            k=effective_k,
            graph=graph,
            temperature=temperature,
        )
        soft, slots = _pad_physical_soft_row(
            soft_valid,
            active_slots,
            temporal_len=temporal_len,
            k=k,
        )
        soft_rows.append(soft)
        slot_rows.append(slots)
        partition_rows.append(partition)
        edge_counts.append(graph.edge_count)
        effective_rows.append(effective_k)
    return PhysicalExactKSoftOutput(
        soft_occupancy=torch.stack(soft_rows, dim=0),
        soft_slot_assignment=torch.stack(slot_rows, dim=0),
        log_partition=torch.stack(partition_rows, dim=0),
        edge_count=torch.tensor(edge_counts, device=node_log_probs.device, dtype=torch.long),
        effective_k=torch.tensor(effective_rows, device=node_log_probs.device, dtype=torch.long),
        max_gap_seconds=caps.to(device=node_log_probs.device, dtype=node_log_probs.dtype),
        temperature=temperature,
    )


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


def physical_exact_k_homotopy_log_potential(
    learned_log_potential: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
    alpha: float,
) -> torch.Tensor:
    """Mix exact-uniform and learned node potentials before exact-K decoding."""

    if (
        not torch.is_tensor(learned_log_potential)
        or learned_log_potential.ndim != 2
        or not learned_log_potential.is_floating_point()
    ):
        raise ValueError(
            "learned_log_potential must be a floating-point [B,T] tensor"
        )
    valid = valid_mask.to(device=learned_log_potential.device, dtype=torch.bool)
    if valid.shape != learned_log_potential.shape:
        raise ValueError("valid_mask must align with learned_log_potential")
    if bool(torch.any(valid.sum(dim=1) == 0).item()):
        raise ValueError("physical exact-K homotopy requires one valid point per row")
    if not bool(torch.isfinite(learned_log_potential[valid]).all().item()):
        raise ValueError("valid learned log-potentials must be finite")
    k = int(k)
    if k < 1 or k > int(learned_log_potential.shape[1]):
        raise ValueError("k must lie in [1,T]")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("homotopy alpha must lie in [0,1]")

    reference = exact_uniform_reference_scores(
        learned_log_potential,
        valid,
        k,
    )
    learned_valid = learned_log_potential.masked_fill(~valid, 0.0)
    mixed = (1.0 - alpha) * reference + alpha * learned_valid
    return mixed.masked_fill(~valid, float("-inf"))


def exact_uniform_cell_bounds(
    temporal_len: int,
    k: int,
    *,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Partition a window into nearest-anchor cells with deterministic midpoint ties.

    Every exact-uniform anchor owns one non-empty contiguous cell. A temporal
    midpoint equidistant from adjacent anchors is assigned to the earlier cell.
    """

    temporal_len = int(temporal_len)
    k = int(k)
    anchors = exact_uniform_positions(temporal_len, k, device=device)
    if k == 0:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return anchors, empty, empty
    starts = torch.zeros((k,), device=device, dtype=torch.long)
    ends = torch.full((k,), temporal_len, device=device, dtype=torch.long)
    if k > 1:
        starts[1:] = torch.div(anchors[:-1] + anchors[1:], 2, rounding_mode="floor") + 1
        ends[:-1] = starts[1:]
    if torch.any(starts >= ends):
        raise RuntimeError("exact-uniform cells must be non-empty")
    if torch.any(anchors < starts) or torch.any(anchors >= ends):
        raise RuntimeError("each exact-uniform anchor must lie inside its cell")
    return anchors, starts, ends


def _max_unselected_hole(positions: torch.Tensor, temporal_len: int) -> int:
    if positions.numel() == 0:
        return int(temporal_len)
    sentinels = torch.cat(
        (
            positions.new_tensor([-1]),
            positions,
            positions.new_tensor([int(temporal_len)]),
        )
    )
    return int((sentinels[1:] - sentinels[:-1] - 1).max().item())


def _local_cell_max_hole(starts: torch.Tensor, ends: torch.Tensor, temporal_len: int) -> int:
    if starts.numel() == 0:
        return int(temporal_len)
    candidates = [int(ends[0].item()) - 1, int(temporal_len) - int(starts[-1].item()) - 1]
    for cell_index in range(int(starts.numel()) - 1):
        candidates.append(int(ends[cell_index + 1].item()) - int(starts[cell_index].item()) - 2)
    return max(candidates)


def local_cell_deformation(
    policy_logits: torch.Tensor,
    *,
    k: int,
    temperature: float = 1.0,
    training: bool = False,
    force_exact_uniform: bool = False,
) -> LocalCellSelectionOutput:
    """Select exactly one frame per exact-uniform cell.

    The hard and relaxed paths share the same product-of-categorical policy.
    Hard ties explicitly choose the cell's exact-uniform anchor, making a
    zero-initialized policy identical to exact uniform sampling.
    """

    if not torch.is_tensor(policy_logits) or policy_logits.ndim != 2:
        raise ValueError("policy_logits must be a [B,T] tensor")
    if not policy_logits.is_floating_point() or not bool(torch.isfinite(policy_logits).all().item()):
        raise ValueError("policy_logits must contain finite floating-point values")
    temporal_len = int(policy_logits.shape[1])
    k = int(k)
    temperature = float(temperature)
    if k < 0 or k > temporal_len:
        raise ValueError("k must lie in [0,T]")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    anchors, starts, ends = exact_uniform_cell_bounds(temporal_len, k, device=policy_logits.device)
    batch = int(policy_logits.shape[0])
    hard = policy_logits.new_zeros((batch, temporal_len))
    slots = policy_logits.new_zeros((batch, k, temporal_len))
    selected = torch.empty((batch, k), device=policy_logits.device, dtype=torch.long)
    log_partition = policy_logits.new_zeros((batch,), dtype=torch.float32)

    for cell_index in range(k):
        start = int(starts[cell_index].item())
        end = int(ends[cell_index].item())
        anchor = int(anchors[cell_index].item())
        cell_scores = policy_logits[:, start:end]
        if force_exact_uniform:
            hard_position = torch.full((batch,), anchor, device=policy_logits.device, dtype=torch.long)
        else:
            detached = cell_scores.detach()
            max_value = detached.max(dim=1).values
            tied = detached == max_value[:, None]
            absolute_positions = torch.arange(start, end, device=policy_logits.device)
            anchor_distance = (absolute_positions - anchor).abs()[None, :].expand(batch, -1)
            invalid_distance = torch.full_like(anchor_distance, temporal_len + 1)
            closest_tied = torch.where(tied, anchor_distance, invalid_distance).argmin(dim=1)
            hard_position = closest_tied + start
        selected[:, cell_index] = hard_position
        hard.scatter_(1, hard_position[:, None], 1.0)

        if training and not force_exact_uniform:
            cell_logits = cell_scores.float() / temperature
            cell_weights = torch.softmax(cell_logits, dim=1).to(dtype=policy_logits.dtype)
            slots[:, cell_index, start:end] = cell_weights
            log_partition = log_partition + torch.logsumexp(cell_logits, dim=1)
        else:
            slots[:, cell_index].scatter_(1, hard_position[:, None], 1.0)

    soft = slots.sum(dim=1)
    selection_st = hard + soft - soft.detach() if training and not force_exact_uniform else hard
    theoretical_max_hole = _local_cell_max_hole(starts, ends, temporal_len)
    if k > 0:
        observed = max(_max_unselected_hole(selected[row], temporal_len) for row in range(batch))
        if observed > theoretical_max_hole:
            raise RuntimeError("local-cell selection violated its maximum-hole contract")
    return LocalCellSelectionOutput(
        hard_occupancy=hard,
        soft_occupancy=soft,
        soft_slot_assignment=slots,
        selection_st=selection_st,
        selected_positions=selected,
        log_partition=log_partition,
        anchor_positions=anchors,
        cell_starts=starts,
        cell_ends=ends,
        k=k,
        max_unselected_hole=theoretical_max_hole,
        temperature=temperature,
        force_exact_uniform=bool(force_exact_uniform),
    )


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
        reachable = values > -5.0e8
        has_reachable = reachable.any(dim=dim, keepdim=True)
        masked_values = torch.where(reachable, values, values.new_full((), -1.0e30))
        reduced = torch.logsumexp(masked_values, dim=dim, keepdim=True)
        return torch.where(has_reachable, reduced, neg_inf).squeeze(dim)

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
        impossible_miss = log_z_miss <= -5.0e8
        safe_log_z_miss = torch.where(impossible_miss, log_z.detach(), log_z_miss)
        log_miss = (safe_log_z_miss - log_z).clamp(max=0.0)
        probability = -torch.expm1(log_miss)
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
    "LocalCellSelectionOutput",
    "PhysicalExactKGraph",
    "PhysicalExactKHardOutput",
    "PhysicalExactKSelectionOutput",
    "PhysicalExactKSoftOutput",
    "StructuredSelectionOutput",
    "exact_uniform_cell_bounds",
    "exact_uniform_positions",
    "exact_uniform_reference_scores",
    "global_structured_topk",
    "local_cell_deformation",
    "physical_exact_k_forward_backward",
    "physical_exact_k_homotopy_log_potential",
    "physical_exact_k_select",
    "physical_exact_k_viterbi",
    "physical_exact_uniform_gap_cap",
    "structured_local_coverage_probability",
]
