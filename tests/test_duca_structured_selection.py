from __future__ import annotations

from itertools import combinations
import os

import pytest

if os.name == "nt":
    pytest.skip("local Windows torch/c10.dll import is unstable; Linux remote runs this suite", allow_module_level=True)

import torch

from opentad.models.duca.structured_selection import (
    exact_uniform_positions,
    exact_uniform_reference_scores,
    global_structured_topk,
    physical_exact_k_select,
    physical_exact_k_viterbi,
    physical_exact_uniform_gap_cap,
    structured_local_coverage_probability,
)


def _max_hole(indices: tuple[int, ...], temporal_len: int) -> int:
    if not indices:
        return temporal_len
    return max(
        indices[0],
        *(right - left - 1 for left, right in zip(indices, indices[1:])),
        temporal_len - indices[-1] - 1,
    )


@pytest.mark.parametrize(("temporal_len", "budget"), [(768, 384), (17, 8), (8, 4), (1, 1)])
def test_exact_uniform_helper_is_the_rounded_endpoint_contract(temporal_len: int, budget: int) -> None:
    positions = exact_uniform_positions(temporal_len, budget)
    expected = torch.linspace(0, temporal_len - 1, steps=budget).round().long()

    assert torch.equal(positions.cpu(), expected)
    assert positions.unique().numel() == budget


def test_exact_uniform_reference_uses_valid_rank_then_maps_to_physical_positions() -> None:
    scores = torch.zeros(1, 8)
    valid = torch.tensor([[False, True, True, False, True, True, True, False]])
    reference = exact_uniform_reference_scores(scores, valid, k=3)

    assert torch.equal(reference[0, torch.tensor([1, 4, 6])], torch.zeros(3))
    assert torch.equal(reference[0, torch.tensor([2, 5])], -torch.ones(2))
    assert torch.equal(reference.masked_select(~valid), torch.zeros(3))


@pytest.mark.parametrize("temporal_len", range(1, 33))
def test_exact_uniform_routes_share_one_rounding_contract(temporal_len: int) -> None:
    scores = torch.zeros(1, temporal_len)
    valid = torch.ones_like(scores, dtype=torch.bool)
    for budget in range(1, temporal_len + 1):
        positions = exact_uniform_positions(temporal_len, budget)
        reference = exact_uniform_reference_scores(scores, valid, budget)
        assert torch.equal(torch.nonzero(reference[0] == 0, as_tuple=False).flatten(), positions)
        assert positions.unique().numel() == budget


def test_physical_gap_cap_is_not_narrowed_to_amp_score_precision() -> None:
    seconds = torch.tensor(
        [[0.0, 0.11, 0.31, 0.52, 0.73, 0.94]],
        dtype=torch.float64,
    )
    valid = torch.ones((1, 6), dtype=torch.bool)
    scores = torch.tensor(
        [[0.2, -0.4, 0.7, 0.1, -0.2, 0.6]],
        dtype=torch.float16,
    )
    cap = physical_exact_uniform_gap_cap(seconds, valid, k=3)

    hard = physical_exact_k_viterbi(
        scores,
        seconds,
        valid,
        k=3,
        max_gap_seconds=cap,
    )
    joint = physical_exact_k_select(
        scores,
        seconds,
        valid,
        k=3,
        max_gap_seconds=cap,
    )

    assert hard.max_gap_seconds.dtype == torch.float64
    assert joint.max_gap_seconds.dtype == torch.float64
    assert torch.equal(hard.max_gap_seconds, cap)
    assert torch.equal(joint.max_gap_seconds, cap)


def test_physical_exact_k_slot_marginals_match_small_bruteforce_distribution() -> None:
    scores = torch.tensor([[0.2, -0.4, 1.1, 0.7, -0.3, 0.5]])
    seconds = torch.arange(6, dtype=torch.float64)[None, :]
    valid = torch.ones((1, 6), dtype=torch.bool)
    k = 3

    output = physical_exact_k_select(
        scores,
        seconds,
        valid,
        k=k,
        max_gap_seconds=torch.tensor([10.0], dtype=torch.float64),
    )
    paths = tuple(combinations(range(6), k))
    weights = torch.stack([torch.exp(scores[0, list(path)].sum()) for path in paths])
    expected = torch.zeros((k, 6), dtype=weights.dtype)
    for slot_index in range(k):
        for position in range(6):
            mask = torch.tensor(
                [path[slot_index] == position for path in paths],
                dtype=torch.bool,
            )
            expected[slot_index, position] = weights[mask].sum() / weights.sum()

    assert torch.allclose(output.soft_slot_assignment[0], expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        output.soft_slot_assignment.sum(dim=2),
        torch.ones((1, k)),
        atol=1e-6,
        rtol=1e-6,
    )


def test_physical_exact_k_long_high_dynamic_range_has_finite_exact_marginals() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(7582)
    temporal_len, k = 768, 384
    scores = (
        torch.randn((1, temporal_len), generator=generator, device=device) * 16.0
    ).requires_grad_(True)
    seconds = (
        torch.arange(temporal_len, device=device, dtype=torch.float64) * 4.0 / 30.0
    )[None, :]
    valid = torch.ones((1, temporal_len), device=device, dtype=torch.bool)

    output = physical_exact_k_select(scores, seconds, valid, k=k)

    active_slots = output.soft_slot_assignment[0, :k, :temporal_len]
    assert torch.isfinite(active_slots).all()
    assert torch.allclose(
        active_slots.sum(dim=1),
        torch.ones((k,), device=device),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.all(active_slots.sum(dim=0) <= 1.0 + 5.0e-4)
    positions = output.hard_positions[0, :k]
    assert positions.unique().numel() == k
    assert torch.all(positions[1:] > positions[:-1])

    time_cost = torch.arange(temporal_len, device=device, dtype=scores.dtype)
    (output.selection_st[0] * time_cost[None, :]).sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum().item() > 0.0


def test_structured_local_coverage_matches_bruteforce_path_distribution() -> None:
    logits = torch.tensor([[0.2, -0.4, 1.1, 0.7, -0.3]], dtype=torch.float64, requires_grad=True)
    k, max_hole, temperature = 2, 2, 0.8
    events = torch.tensor([[[False, True, True, False, False], [False, False, False, False, True]]])
    actual = structured_local_coverage_probability(
        logits,
        events,
        k=k,
        max_unselected_hole=max_hole,
        temperature=temperature,
    )
    feasible = [
        choice
        for choice in combinations(range(logits.shape[1]), k)
        if _max_hole(choice, logits.shape[1]) <= max_hole
    ]
    weights = torch.stack([torch.exp(logits[0, list(choice)].sum() / temperature) for choice in feasible])
    expected = []
    for event in events[0]:
        covered = torch.tensor(
            [any(bool(event[idx]) for idx in choice) for choice in feasible],
            dtype=weights.dtype,
        )
        expected.append((weights * covered).sum() / weights.sum())
    assert torch.allclose(actual[0], torch.stack(expected), atol=1e-6, rtol=1e-6)
    actual.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_structured_local_coverage_respects_exact_k_dependence() -> None:
    logits = torch.zeros(1, 3)
    whole_window = torch.ones(1, 1, 3, dtype=torch.bool)
    probability = structured_local_coverage_probability(
        logits, whole_window, k=1, max_unselected_hole=2
    )
    assert probability.item() == pytest.approx(1.0)


def test_structured_local_coverage_impossible_miss_has_finite_zero_gradient() -> None:
    logits = torch.randn(2, 32, dtype=torch.float32, requires_grad=True)
    unavoidable_event = torch.ones(2, 1, 32, dtype=torch.bool)

    probability = structured_local_coverage_probability(
        logits,
        unavoidable_event,
        k=16,
        max_unselected_hole=15,
        temperature=0.7,
    )

    assert torch.equal(probability, torch.ones_like(probability))
    probability.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_structured_local_coverage_exhausts_small_state_spaces() -> None:
    for temporal_len in range(1, 7):
        logits = torch.linspace(-0.5, 0.7, temporal_len, dtype=torch.float64)[None, :]
        for k in range(temporal_len + 1):
            for max_hole in range(temporal_len + 1):
                feasible = [
                    choice
                    for choice in combinations(range(temporal_len), k)
                    if _max_hole(choice, temporal_len) <= max_hole
                ]
                if not feasible:
                    continue
                events = torch.eye(temporal_len, dtype=torch.bool)[None, :, :]
                actual = structured_local_coverage_probability(
                    logits,
                    events,
                    k=k,
                    max_unselected_hole=max_hole,
                    temperature=0.9,
                )[0]
                weights = torch.stack(
                    [torch.exp(logits[0, list(choice)].sum() / 0.9) for choice in feasible]
                )
                expected = torch.stack(
                    [
                        (weights * torch.tensor([idx in choice for choice in feasible])).sum() / weights.sum()
                        for idx in range(temporal_len)
                    ]
                ).to(dtype=logits.dtype)
                assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_structured_hard_map_matches_bruteforce_optimum() -> None:
    logits = torch.tensor([[0.2, 1.4, -0.7, 0.8, 1.1, -0.2, 0.5]])
    k = 3
    max_hole = 2
    feasible = [
        choice
        for choice in combinations(range(logits.shape[1]), k)
        if _max_hole(choice, logits.shape[1]) <= max_hole
    ]
    expected = max(feasible, key=lambda choice: sum(float(logits[0, idx]) for idx in choice))

    out = global_structured_topk(logits, k=k, max_unselected_hole=max_hole, training=False)

    assert tuple(out.selected_positions[0].tolist()) == expected
    assert out.hard_occupancy.sum().item() == k
    assert _max_hole(expected, logits.shape[1]) <= max_hole


def test_structured_soft_slots_obey_exact_budget_and_st_forward_identity() -> None:
    logits = torch.randn(2, 9, requires_grad=True)
    out = global_structured_topk(
        logits,
        k=4,
        max_unselected_hole=2,
        temperature=0.7,
        training=True,
    )

    assert torch.equal(out.selection_st.detach(), out.hard_occupancy)
    assert torch.allclose(out.soft_slot_assignment.sum(dim=2), torch.ones(2, 4), atol=1e-5)
    assert torch.allclose(out.soft_occupancy.sum(dim=1), torch.full((2,), 4.0), atol=1e-5)
    assert torch.allclose(out.soft_occupancy, out.soft_slot_assignment.sum(dim=1), atol=1e-6)

    time_cost = torch.arange(9, dtype=logits.dtype)[None, :]
    (out.selection_st * time_cost).sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0


def test_structured_selection_fails_closed_when_budget_gap_contract_is_infeasible() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        global_structured_topk(
            torch.randn(1, 8),
            k=1,
            max_unselected_hole=2,
            training=False,
        )


def test_structured_soft_budget_remains_exact_on_longer_sequences() -> None:
    logits = torch.randn(1, 128, requires_grad=True)

    out = global_structured_topk(
        logits,
        k=64,
        max_unselected_hole=15,
        temperature=0.7,
        training=True,
    )

    assert torch.allclose(out.soft_slot_assignment.sum(dim=2), torch.ones(1, 64), atol=1e-6)
    assert torch.allclose(out.soft_occupancy.sum(dim=1), torch.tensor([64.0]), atol=1e-5)


def test_structured_selection_is_global_not_prefix_invariant() -> None:
    prefix = torch.tensor([[2.0, 1.0, 0.5, 0.2]])
    first = torch.cat((prefix, torch.full((1, 4), -5.0)), dim=1)
    second = torch.cat((prefix, torch.full((1, 4), 5.0)), dim=1)

    first_out = global_structured_topk(first, k=4, max_unselected_hole=4, training=False)
    second_out = global_structured_topk(second, k=4, max_unselected_hole=4, training=False)

    assert not torch.equal(first_out.hard_occupancy[:, :4], second_out.hard_occupancy[:, :4])
    assert first_out.selection_scope == "full_window_non_streaming"
    assert second_out.selection_scope == "full_window_non_streaming"
