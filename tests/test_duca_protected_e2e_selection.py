from __future__ import annotations

import itertools
import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "opentad"
    / "models"
    / "duca"
    / "structured_selection.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_duca_structured_selection_under_test",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_SELECTION = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SELECTION
_SPEC.loader.exec_module(_SELECTION)

physical_exact_k_forward_backward = _SELECTION.physical_exact_k_forward_backward
physical_exact_k_select = _SELECTION.physical_exact_k_select
physical_exact_k_viterbi = _SELECTION.physical_exact_k_viterbi
physical_exact_uniform_gap_cap = _SELECTION.physical_exact_uniform_gap_cap


def _legal_paths(seconds, k, cap):
    tolerance = 1.0e-9
    paths = []
    for path in itertools.combinations(range(len(seconds)), k):
        intervals = [seconds[path[0]] - seconds[0]]
        intervals.extend(
            seconds[right] - seconds[left]
            for left, right in zip(path, path[1:])
        )
        intervals.append(seconds[-1] - seconds[path[-1]])
        if max(intervals) <= cap + tolerance:
            paths.append(path)
    return paths


def _exhaustive_distribution(scores, seconds, k, cap, temperature):
    paths = _legal_paths(seconds, k, cap)
    assert paths
    objectives = torch.tensor(
        [sum(float(scores[index]) for index in path) / temperature for path in paths],
        dtype=torch.float64,
    )
    probabilities = torch.softmax(objectives, dim=0)
    slots = torch.zeros((k, len(seconds)), dtype=torch.float64)
    for probability, path in zip(probabilities, paths):
        for slot, position in enumerate(path):
            slots[slot, position] += probability
    best_objective = max(
        sum(float(scores[index]) for index in path)
        for path in paths
    )
    best_path = min(
        path
        for path in paths
        if math.isclose(
            sum(float(scores[index]) for index in path),
            best_objective,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )
    return paths, slots, torch.logsumexp(objectives, dim=0), best_path


def test_physical_exact_k_matches_exhaustive_irregular_graph():
    scores = torch.tensor(
        [[-0.4, 0.7, -0.1, 1.2, 0.3, -0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    seconds = torch.tensor(
        [[0.0, 0.3, 0.9, 1.6, 2.0, 2.8]],
        dtype=torch.float64,
    )
    valid = torch.ones_like(scores, dtype=torch.bool)
    cap = torch.tensor([1.3], dtype=torch.float64)
    temperature = 0.8

    paths, expected_slots, expected_log_z, expected_hard = _exhaustive_distribution(
        scores.detach()[0],
        seconds[0],
        3,
        float(cap.item()),
        temperature,
    )
    output = physical_exact_k_select(
        scores,
        seconds,
        valid,
        k=3,
        max_gap_seconds=cap,
        temperature=temperature,
    )

    assert tuple(output.hard_positions[0].tolist()) == expected_hard
    assert torch.equal(output.selection_st.detach(), output.hard_slot_assignment)
    assert torch.allclose(
        output.soft_slot_assignment[0].double(),
        expected_slots,
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    assert torch.allclose(
        output.log_partition[0].double(),
        expected_log_z,
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    assert torch.allclose(
        output.soft_slot_assignment.sum(dim=-1),
        torch.ones((1, 3), dtype=output.soft_slot_assignment.dtype),
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    assert bool((output.soft_slot_assignment.sum(dim=1) <= 1.0 + 1.0e-5).all())
    assert output.edge_count.item() == (
        sum(
            seconds[0, right] - seconds[0, left] <= cap[0] + 1.0e-9
            for right in range(seconds.shape[1])
            for left in range(right)
        )
        + sum(seconds[0] - seconds[0, 0] <= cap[0] + 1.0e-9)
        + sum(seconds[0, -1] - seconds[0] <= cap[0] + 1.0e-9)
    )
    supported = expected_slots > 0
    assert supported.sum().item() <= len(paths) * 3

    weighted = (
        output.soft_slot_assignment
        * torch.arange(1, 7, dtype=output.soft_slot_assignment.dtype)[None, None]
    ).sum()
    weighted.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum() > 0


def test_physical_exact_k_lexicographic_tie_break():
    scores = torch.zeros((1, 5), dtype=torch.float32)
    seconds = torch.arange(5, dtype=torch.float64)[None]
    valid = torch.ones((1, 5), dtype=torch.bool)
    hard = physical_exact_k_viterbi(
        scores,
        seconds,
        valid,
        k=3,
        max_gap_seconds=torch.tensor([2.0], dtype=torch.float64),
    )
    assert hard.hard_positions.tolist() == [[0, 1, 2]]


def test_physical_exact_k_short_rows_select_every_valid_candidate_once():
    scores = torch.tensor(
        [
            [0.2, -0.4, 0.7, 99.0, 99.0, 99.0],
            [0.2, 0.1, 0.0, -0.1, -0.2, -0.3],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    seconds = torch.tensor(
        [
            [0.0, 0.4, 0.9, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.7, 1.1, 1.8, 2.0],
        ],
        dtype=torch.float64,
    )
    valid = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, True, True, True],
        ]
    )
    output = physical_exact_k_select(scores, seconds, valid, k=5)

    assert output.effective_k.tolist() == [3, 5]
    assert output.hard_positions[0].tolist() == [0, 1, 2, -1, -1]
    assert output.hard_slot_mask[0].tolist() == [True, True, True, False, False]
    assert torch.equal(output.hard_occupancy[0], valid[0].to(dtype=scores.dtype))
    assert output.hard_slot_assignment[0, 3:].count_nonzero().item() == 0
    assert output.soft_slot_assignment[0, 3:].count_nonzero().item() == 0
    assert output.soft_slot_assignment[0, :, 3:].count_nonzero().item() == 0
    assert torch.isfinite(output.log_partition).all()

    output.soft_occupancy.sum().backward()
    assert scores.grad is not None
    assert scores.grad[0, 3:].count_nonzero().item() == 0
    assert torch.isfinite(scores.grad).all()


def test_physical_exact_k_separate_hard_and_soft_paths_share_contract():
    scores = torch.tensor([[0.1, -0.5, 0.6, 0.3]], dtype=torch.float32)
    seconds = torch.tensor([[0.0, 0.4, 1.0, 1.4]], dtype=torch.float64)
    valid = torch.ones((1, 4), dtype=torch.bool)
    cap = torch.tensor([0.8], dtype=torch.float64)

    combined = physical_exact_k_select(
        scores,
        seconds,
        valid,
        k=2,
        max_gap_seconds=cap,
    )
    hard = physical_exact_k_viterbi(
        scores,
        seconds,
        valid,
        k=2,
        max_gap_seconds=cap,
    )
    soft = physical_exact_k_forward_backward(
        scores,
        seconds,
        valid,
        k=2,
        max_gap_seconds=cap,
    )

    assert torch.equal(combined.hard_positions, hard.hard_positions)
    assert torch.equal(combined.hard_slot_assignment, hard.hard_slot_assignment)
    assert torch.allclose(combined.soft_slot_assignment, soft.soft_slot_assignment)
    assert torch.equal(combined.edge_count, hard.edge_count)
    assert torch.equal(combined.edge_count, soft.edge_count)


def test_uniform_reference_cap_is_translation_invariant_and_feasible():
    seconds = torch.tensor(
        [
            [0.0, 0.2, 0.7, 1.5, 1.7, 2.6],
            [9.0, 9.2, 9.7, 10.5, 10.7, 11.6],
        ],
        dtype=torch.float64,
    )
    valid = torch.ones((2, 6), dtype=torch.bool)
    caps = physical_exact_uniform_gap_cap(seconds, valid, k=3)
    assert torch.allclose(caps[0], caps[1], atol=1.0e-12, rtol=0.0)

    hard = physical_exact_k_viterbi(
        torch.zeros((2, 6)),
        seconds,
        valid,
        k=3,
    )
    assert torch.equal(hard.hard_positions[0], hard.hard_positions[1])
    assert torch.equal(hard.effective_k, torch.tensor([3, 3]))


@pytest.mark.parametrize(
    "seconds, valid",
    [
        ([[0.0, 0.5, 0.4]], [[True, True, True]]),
        ([[0.0, float("nan"), 1.0]], [[True, True, True]]),
        ([[0.0, 0.5, 1.0]], [[True, False, True]]),
    ],
)
def test_physical_exact_k_rejects_invalid_axes(seconds, valid):
    with pytest.raises(ValueError):
        physical_exact_k_select(
            torch.zeros((1, 3)),
            torch.tensor(seconds, dtype=torch.float64),
            torch.tensor(valid, dtype=torch.bool),
            k=2,
        )


def test_physical_exact_k_rejects_empty_valid_prefix():
    with pytest.raises(ValueError, match="at least one valid candidate"):
        physical_exact_k_select(
            torch.zeros((1, 3)),
            torch.zeros((1, 3), dtype=torch.float64),
            torch.zeros((1, 3), dtype=torch.bool),
            k=2,
        )
