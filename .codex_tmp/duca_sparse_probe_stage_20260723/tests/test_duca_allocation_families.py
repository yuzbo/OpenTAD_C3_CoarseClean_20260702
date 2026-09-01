from __future__ import annotations

import pytest

from tools.bata.duca_allocation_families import (
    AllocationContractError,
    PhysicalAxis,
    exact_uniform_positions,
    minimum_physical_scaffold,
    physical_gap_report,
    resolve_physical_cap,
    select_family_a,
    select_family_b,
    select_family_c,
    uniform_cell_bounds,
)


def _axis(length: int, stride: int = 4) -> PhysicalAxis:
    return PhysicalAxis.from_source_frames(
        [stride * index for index in range(length)],
        decoder_fps=25.0,
        annotation_fps=25.0,
    )


def test_exact_uniform_matches_registered_768_to_384_contract() -> None:
    positions = exact_uniform_positions(768, 384)
    assert len(positions) == 384
    assert positions[0] == 0
    assert positions[-1] == 767
    gaps = [right - left for left, right in zip(positions, positions[1:])]
    assert gaps.count(2) == 382
    assert gaps.count(3) == 1
    assert physical_gap_report(_axis(768), positions).dense_max_unselected_hole == 2


def test_uniform_cells_are_contiguous_nonempty_and_own_their_anchors() -> None:
    anchors, starts, ends = uniform_cell_bounds(31, 12)
    assert starts[0] == 0
    assert ends[-1] == 31
    assert list(ends[:-1]) == list(starts[1:])
    assert all(start <= anchor < end for anchor, start, end in zip(anchors, starts, ends))


def test_explicit_frame_cap_reproduces_registered_scaffold_fixture() -> None:
    axis = _axis(768)
    cap = resolve_physical_cap(
        axis,
        requested_budget=384,
        policy="explicit_frames",
        value=15,
    )
    arbitrary = minimum_physical_scaffold(axis, cap)
    uniform_subset = minimum_physical_scaffold(
        axis,
        cap,
        candidate_positions=exact_uniform_positions(768, 384),
    )
    assert len(arbitrary) == 255
    assert len(uniform_subset) == 382
    assert physical_gap_report(axis, arbitrary).source_frame_max_interval == 12


def test_uniform_reference_cap_includes_exact_uniform_by_construction() -> None:
    axis = _axis(23, stride=5)
    cap = resolve_physical_cap(axis, requested_budget=9)
    family = select_family_a(axis, requested_budget=9, cap=cap)
    assert family.positions == exact_uniform_positions(23, 9)
    assert family.physical_cap_compliant
    assert family.deployable
    assert not family.privileged


def test_family_b_reports_cap_violation_instead_of_hiding_current_limitation() -> None:
    axis = _axis(16, stride=1)
    cap = resolve_physical_cap(axis, requested_budget=4)
    anchors, starts, ends = uniform_cell_bounds(16, 4)
    scores = [0.0] * 16
    choices = [ends[0] - 1, starts[1], ends[2] - 1, starts[3]]
    for rank, position in enumerate(choices):
        scores[position] = 10.0 + rank
    family = select_family_b(axis, scores, requested_budget=4, cap=cap)
    assert len(family.positions) == 4
    assert all(
        start <= position < end
        for position, start, end in zip(family.positions, starts, ends)
    )
    assert family.gap_report.source_frame_max_interval >= cap.max_source_frame_interval
    assert family.positions != anchors


def test_family_c_keeps_minimum_uniform_scaffold_and_fills_exact_budget() -> None:
    axis = _axis(40, stride=4)
    cap = resolve_physical_cap(
        axis,
        requested_budget=20,
        policy="explicit_frames",
        value=15,
    )
    scores = [float(index) for index in range(axis.valid_len)]
    family = select_family_c(axis, scores, requested_budget=20, cap=cap)
    expected_scaffold = minimum_physical_scaffold(
        axis,
        cap,
        candidate_positions=exact_uniform_positions(40, 20),
    )
    assert family.scaffold_positions == expected_scaffold
    assert len(family.positions) == 20
    assert set(family.scaffold_positions).isdisjoint(family.residual_positions)
    assert family.physical_cap_compliant


@pytest.mark.parametrize(
    "frames",
    [
        [],
        [0, 0, 4],
        [0, 8, 4],
        [0, float("nan"), 8],
    ],
)
def test_physical_axis_rejects_ambiguous_or_nonmonotonic_coordinates(frames) -> None:
    with pytest.raises(AllocationContractError):
        PhysicalAxis.from_source_frames(
            frames,
            decoder_fps=25.0,
            annotation_fps=25.0,
        )
