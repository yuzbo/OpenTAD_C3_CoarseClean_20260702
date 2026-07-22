from __future__ import annotations

import pytest

from tools.bata.duca_allocation_families import PhysicalAxis, resolve_physical_cap
from tools.bata.duca_exact_physical_solver import solve_boundary_burst_oracle


@pytest.mark.parametrize("radius,quota", [(2, 3), (4, 5)])
def test_boundary_burst_oracle_proves_per_sample_contract(radius: int, quota: int) -> None:
    axis = PhysicalAxis.from_source_frames(
        [4 * index for index in range(48)],
        decoder_fps=30.0,
        annotation_fps=30.0,
    )
    cap = resolve_physical_cap(
        axis,
        requested_budget=24,
        policy="explicit_frames",
        value=12,
    )
    result = solve_boundary_burst_oracle(
        axis,
        [[14.2, 33.4]],
        [[True, True]],
        requested_budget=24,
        cap=cap,
        radius=radius,
        quota=quota,
        max_unselected_hole=2,
    )

    assert len(result.positions) == 24
    assert result.positions[0] == 0
    assert result.positions[-1] == 47
    assert result.max_unselected_hole <= 2
    assert result.residual_fill_count > 0
    assert result.background_selected_count > 0
    assert all(row["quota_pass"] for row in result.endpoint_contracts)
    assert all(row["bilateral_pass"] for row in result.endpoint_contracts)
    assert all(row["selected_in_radius"] >= quota for row in result.endpoint_contracts)


def test_boundary_burst_oracle_does_not_supervise_crop_cut_endpoint() -> None:
    axis = PhysicalAxis.from_source_frames(
        [4 * index for index in range(32)],
        decoder_fps=30.0,
        annotation_fps=30.0,
    )
    cap = resolve_physical_cap(
        axis,
        requested_budget=16,
        policy="explicit_frames",
        value=12,
    )
    result = solve_boundary_burst_oracle(
        axis,
        [[0.0, 20.0]],
        [[False, True]],
        requested_budget=16,
        cap=cap,
        radius=2,
        quota=3,
        max_unselected_hole=2,
    )

    assert result.invalid_endpoint_count == 1
    assert len(result.endpoint_contracts) == 1
    assert result.endpoint_contracts[0]["endpoint"] == "end"


def test_boundary_burst_oracle_jointly_allocates_overlapping_endpoint_quotas() -> None:
    axis = PhysicalAxis.from_source_frames(
        list(range(7)),
        decoder_fps=30.0,
        annotation_fps=30.0,
    )
    cap = resolve_physical_cap(
        axis,
        requested_budget=5,
        policy="explicit_frames",
        value=3,
    )
    result = solve_boundary_burst_oracle(
        axis,
        [[2.0, 4.0]],
        [[True, True]],
        requested_budget=5,
        cap=cap,
        radius=4,
        quota=5,
        max_unselected_hole=2,
    )

    assert len(result.positions) == 5
    assert result.positions[0] == 0
    assert result.positions[-1] == 6
    assert {2, 3}.issubset(result.positions)
    assert result.max_unselected_hole <= 2
    assert all(row["selected_in_radius"] >= 5 for row in result.endpoint_contracts)
    assert all(row["bilateral_pass"] for row in result.endpoint_contracts)
    assert result.mip_gap == 0.0
    assert result.solver_identity == "duca_r0_exact_quota_physical_milp_v1"


def test_unrestricted_boundary_burst_oracle_removes_coverage_scaffold() -> None:
    axis = PhysicalAxis.from_source_frames(
        [4 * index for index in range(24)],
        decoder_fps=30.0,
        annotation_fps=30.0,
    )
    cap = resolve_physical_cap(
        axis,
        requested_budget=6,
        policy="explicit_frames",
        value=92,
    )
    result = solve_boundary_burst_oracle(
        axis,
        [[10.0, 11.0]],
        [[True, False]],
        requested_budget=6,
        cap=cap,
        radius=4,
        quota=5,
        max_unselected_hole=23,
        enforce_global_coverage=False,
    )

    assert len(result.positions) == 6
    assert 0 not in result.required_positions
    assert 23 not in result.required_positions
    assert result.positions[-1] < 23
    assert result.max_unselected_hole > 2
    assert result.endpoint_contracts[0]["quota_pass"] is True
