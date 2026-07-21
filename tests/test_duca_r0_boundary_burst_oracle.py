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
