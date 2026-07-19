from __future__ import annotations

from itertools import combinations
import random

import pytest

from tools.bata.duca_allocation_families import (
    PhysicalAxis,
    physical_gap_report,
    resolve_physical_cap,
)
from tools.bata.duca_exact_physical_solver import (
    GroundTruthObjectiveSpec,
    quantize_scores,
    select_family_d,
    solve_additive_exact_k_physical,
    solve_additive_unrestricted,
    solve_ground_truth_lexicographic,
)


def _axis(length: int) -> PhysicalAxis:
    return PhysicalAxis.from_source_frames(
        [3 * index for index in range(length)],
        decoder_fps=30.0,
        annotation_fps=30.0,
    )


def _brute_force(axis, quantized, budget, cap):
    feasible = []
    for positions in combinations(range(axis.valid_len), budget):
        report = physical_gap_report(axis, positions)
        if report.source_frame_max_interval <= cap.max_source_frame_interval + 1.0e-9:
            feasible.append((sum(quantized[index] for index in positions), positions))
    return min(feasible, key=lambda item: (-item[0], item[1]))


def test_additive_physical_dp_matches_exhaustive_small_instances() -> None:
    generator = random.Random(20260720)
    for length in range(4, 10):
        for budget in range(2, length):
            axis = _axis(length)
            cap = resolve_physical_cap(axis, requested_budget=budget)
            scores = [generator.uniform(-2.0, 2.0) for _ in range(length)]
            solved = solve_additive_exact_k_physical(
                axis,
                scores,
                requested_budget=budget,
                cap=cap,
                quantization_scale=1000,
            )
            expected_objective, expected_positions = _brute_force(
                axis,
                solved.quantized_scores.values,
                budget,
                cap,
            )
            assert solved.quantized_objective == expected_objective
            assert solved.positions == expected_positions


def test_additive_physical_dp_uses_lexicographic_tie_break() -> None:
    axis = _axis(9)
    cap = resolve_physical_cap(axis, requested_budget=4)
    solved = solve_additive_exact_k_physical(
        axis,
        [0.0] * axis.valid_len,
        requested_budget=4,
        cap=cap,
    )
    _, expected = _brute_force(
        axis,
        solved.quantized_scores.values,
        4,
        cap,
    )
    assert solved.positions == expected


def test_decimal_quantization_has_explicit_half_even_semantics() -> None:
    quantized = quantize_scores(
        [0.0000005, 0.0000015, -0.0000005, -0.0000015],
        scale=1_000_000,
    )
    assert quantized.values == (0, 2, 0, -2)


def test_family_d_is_exact_deployable_and_cap_compliant() -> None:
    axis = _axis(18)
    cap = resolve_physical_cap(axis, requested_budget=7)
    family, solved = select_family_d(
        axis,
        [float((index * 7) % 11) for index in range(axis.valid_len)],
        requested_budget=7,
        cap=cap,
    )
    assert family.positions == solved.positions
    assert family.family == "D_global_exact_k_physical_gap"
    assert family.exact
    assert family.deployable
    assert not family.privileged
    assert family.physical_cap_compliant


def test_unrestricted_additive_solver_is_exact_quantized_topk() -> None:
    axis = _axis(7)
    solved = solve_additive_unrestricted(
        axis,
        [2.0, 5.0, 5.0, -1.0, 3.0, 5.0, 0.0],
        requested_budget=3,
    )
    assert solved.positions == (1, 2, 5)


def test_gt_milp_returns_privileged_exact_lexicographic_solution() -> None:
    pytest.importorskip("scipy.optimize")
    axis = PhysicalAxis.from_source_frames(
        range(8),
        decoder_fps=2.0,
        annotation_fps=2.0,
    )
    cap = resolve_physical_cap(axis, requested_budget=4)
    result = solve_ground_truth_lexicographic(
        axis,
        [(1.0, 2.0), (5.0, 6.0)],
        requested_budget=4,
        cap=cap,
        objective_spec=GroundTruthObjectiveSpec(lex_block_size=8),
        compute_upper_envelopes=True,
    )
    assert result.positions == (1, 2, 5, 6)
    assert result.objective_vector["both_endpoints_r0"] == 2
    assert result.objective_vector["distinct_endpoint_hits_r0"] == 4
    assert result.objective_vector["total_endpoint_distance_q"] == 0
    assert result.solver_status == "OPTIMAL"
    assert result.exact
    assert result.privileged
    assert not result.deployable


def test_gt_milp_matches_exhaustive_lexicographic_objective() -> None:
    pytest.importorskip("scipy.optimize")
    spec = GroundTruthObjectiveSpec(
        boundary_radii=(0, 1, 2),
        short_action_max_length=3.0,
        distance_scale=10,
        lex_block_size=8,
    )
    fixtures = (
        (7, 3, ((1.0, 2.0), (4.0, 5.0))),
        (8, 4, ((0.5, 2.5), (5.0, 7.0))),
        (9, 4, ((1.0, 5.0), (6.0, 7.0))),
    )
    for length, budget, segments in fixtures:
        axis = PhysicalAxis.from_source_frames(
            range(length),
            decoder_fps=2.0,
            annotation_fps=2.0,
        )
        cap = resolve_physical_cap(axis, requested_budget=budget)
        feasible = []
        for positions in combinations(range(length), budget):
            report = physical_gap_report(axis, positions)
            if report.source_frame_max_interval <= cap.max_source_frame_interval + 1.0e-9:
                feasible.append(
                    (_gt_exhaustive_key(positions, segments, spec, length), positions)
                )
        expected = min(feasible)[1]
        solved = solve_ground_truth_lexicographic(
            axis,
            segments,
            requested_budget=budget,
            cap=cap,
            objective_spec=spec,
        )
        assert solved.positions == expected


def _gt_exhaustive_key(positions, segments, spec, valid_len):
    selected = set(positions)
    endpoints = tuple(value for segment in segments for value in segment)
    key = []
    for radius in spec.boundary_radii:
        both = sum(
            any(abs(position - start) <= radius for position in selected)
            and any(abs(position - end) <= radius for position in selected)
            for start, end in segments
        )
        key.append(-both)
    for radius in spec.boundary_radii:
        hits = sum(
            any(abs(position - endpoint) <= radius for position in selected)
            for endpoint in endpoints
        )
        key.append(-hits)
    distance = sum(
        min(
            int(round(abs(position - endpoint) * spec.distance_scale))
            for position in selected
        )
        for endpoint in endpoints
    )
    key.append(distance)
    short_support = sum(
        end - start <= spec.short_action_max_length
        and any(start <= position <= end for position in selected)
        for start, end in segments
    )
    key.append(-short_support)
    background = sum(
        not any(start <= position <= end for start, end in segments)
        for position in selected
    )
    key.append(background)
    uniform = set(
        tuple(
            round(index * (valid_len - 1) / (len(positions) - 1))
            for index in range(len(positions))
        )
    )
    key.append(-len(selected & uniform))
    key.append(tuple(positions))
    return tuple(key)
