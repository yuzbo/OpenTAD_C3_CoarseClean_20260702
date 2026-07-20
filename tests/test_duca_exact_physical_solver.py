from __future__ import annotations

from itertools import combinations
import random

import pytest

from tools.bata.duca_allocation_families import (
    AllocationContractError,
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


def test_gt_milp_canonicalizes_tiny_integer_residuals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    scipy_optimize = pytest.importorskip("scipy.optimize")
    original_milp = scipy_optimize.milp
    axis = PhysicalAxis.from_source_frames(
        range(31),
        decoder_fps=2.0,
        annotation_fps=2.0,
    )
    cap = resolve_physical_cap(axis, requested_budget=15)
    spec = GroundTruthObjectiveSpec(lex_block_size=30)
    expected = solve_ground_truth_lexicographic(
        axis,
        [(3.0, 9.0), (18.0, 25.0)],
        requested_budget=15,
        cap=cap,
        objective_spec=spec,
        compute_upper_envelopes=True,
    )

    def perturbed_milp(*args, **kwargs):
        result = original_milp(*args, **kwargs)
        if result.x is not None:
            result.x = result.x.copy()
            integer_indices = np.flatnonzero(np.asarray(kwargs["integrality"]) != 0)
            for index in integer_indices:
                result.x[index] += 1.0e-12
        return result

    monkeypatch.setattr(scipy_optimize, "milp", perturbed_milp)
    solved = solve_ground_truth_lexicographic(
        axis,
        [(3.0, 9.0), (18.0, 25.0)],
        requested_budget=15,
        cap=cap,
        objective_spec=spec,
        compute_upper_envelopes=True,
    )
    assert solved.positions == expected.positions
    assert solved.objective_vector == expected.objective_vector
    assert solved.metric_upper_envelopes == expected.metric_upper_envelopes
    assert solved.mip_gap == 0.0


def test_gt_milp_rejects_material_integer_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    scipy_optimize = pytest.importorskip("scipy.optimize")
    original_milp = scipy_optimize.milp

    def perturbed_milp(*args, **kwargs):
        result = original_milp(*args, **kwargs)
        if result.x is not None:
            result.x = result.x.copy()
            index = int(np.flatnonzero(np.asarray(kwargs["integrality"]) != 0)[0])
            direction = 1.0 if result.x[index] < 0.5 else -1.0
            result.x[index] += direction * 1.0e-3
        return result

    monkeypatch.setattr(scipy_optimize, "milp", perturbed_milp)
    axis = _axis(8)
    cap = resolve_physical_cap(axis, requested_budget=4)
    with pytest.raises(AllocationContractError, match="violates integrality"):
        solve_ground_truth_lexicographic(
            axis,
            [(1.0, 2.0), (5.0, 6.0)],
            requested_budget=4,
            cap=cap,
            objective_spec=GroundTruthObjectiveSpec(lex_block_size=8),
        )


@pytest.mark.parametrize("invalid_gap", [True, 1.0e-6])
def test_gt_milp_requires_numeric_exact_zero_gap(
    monkeypatch: pytest.MonkeyPatch,
    invalid_gap,
) -> None:
    scipy_optimize = pytest.importorskip("scipy.optimize")
    original_milp = scipy_optimize.milp

    def invalid_gap_milp(*args, **kwargs):
        result = original_milp(*args, **kwargs)
        result.mip_gap = invalid_gap
        return result

    monkeypatch.setattr(scipy_optimize, "milp", invalid_gap_milp)
    axis = _axis(8)
    cap = resolve_physical_cap(axis, requested_budget=4)
    expected_message = "numeric mip_gap" if invalid_gap is True else "zero MIP gap"
    with pytest.raises(AllocationContractError, match=expected_message):
        solve_ground_truth_lexicographic(
            axis,
            [(1.0, 2.0), (5.0, 6.0)],
            requested_budget=4,
            cap=cap,
            objective_spec=GroundTruthObjectiveSpec(lex_block_size=8),
        )


def test_gt_milp_rejects_terminal_positions_not_certified_by_solver_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scipy_optimize = pytest.importorskip("scipy.optimize")
    original_milp = scipy_optimize.milp
    call_count = 0

    def swapped_terminal_milp(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = original_milp(*args, **kwargs)
        if call_count == 7 and result.x is not None:
            result.x = result.x.copy()
            assert result.x[1] > 0.5
            assert result.x[0] < 0.5
            result.x[1] = 0.0
            result.x[0] = 1.0
        return result

    monkeypatch.setattr(scipy_optimize, "milp", swapped_terminal_milp)
    axis = PhysicalAxis.from_source_frames(
        range(8),
        decoder_fps=2.0,
        annotation_fps=2.0,
    )
    cap = resolve_physical_cap(axis, requested_budget=4)
    with pytest.raises(
        AllocationContractError,
        match="contradicts solver fun",
    ):
        solve_ground_truth_lexicographic(
            axis,
            [(1.0, 2.0), (5.0, 6.0)],
            requested_budget=4,
            cap=cap,
            objective_spec=GroundTruthObjectiveSpec(
                boundary_radii=(0,),
                lex_block_size=8,
            ),
        )


def test_gt_milp_rejects_uncertified_upper_envelope_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scipy_optimize = pytest.importorskip("scipy.optimize")
    original_milp = scipy_optimize.milp
    call_count = 0

    def swapped_upper_envelope_milp(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = original_milp(*args, **kwargs)
        if call_count == 1 and result.x is not None:
            result.x = result.x.copy()
            assert result.x[1] > 0.5
            assert result.x[0] < 0.5
            result.x[1] = 0.0
            result.x[0] = 1.0
        return result

    monkeypatch.setattr(scipy_optimize, "milp", swapped_upper_envelope_milp)
    axis = PhysicalAxis.from_source_frames(
        range(8),
        decoder_fps=2.0,
        annotation_fps=2.0,
    )
    cap = resolve_physical_cap(axis, requested_budget=4)
    with pytest.raises(
        AllocationContractError,
        match="both_endpoints_r0 is inconsistent with selected positions",
    ):
        solve_ground_truth_lexicographic(
            axis,
            [(1.0, 2.0), (5.0, 6.0)],
            requested_budget=4,
            cap=cap,
            objective_spec=GroundTruthObjectiveSpec(
                boundary_radii=(0,),
                lex_block_size=8,
            ),
            compute_upper_envelopes=True,
        )


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
