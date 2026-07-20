from __future__ import annotations

import pytest
import torch

from opentad.models.duca.temporal_contract import DucaTemporalSamplingContract


def _contract() -> DucaTemporalSamplingContract:
    return DucaTemporalSamplingContract(
        hard_budget=4,
        dense_window_size=8,
        max_unselected_hole_dense_candidates=2,
        dataset_feature_stride_source_frames=4,
        dataset_sample_stride=1,
        requested_max_source_frame_interval=15,
    )


def test_temporal_contract_separates_dense_frame_and_second_units() -> None:
    contract = _contract()
    payload = contract.to_dict(fps=30.0)

    assert contract.candidate_stride_source_frames == 4
    assert contract.max_selected_interval_dense_steps == 3
    assert contract.max_selected_interval_source_frames == 12
    assert payload["max_selected_interval_seconds"] == pytest.approx(0.4)
    assert payload["selected_positions_semantics"] == "dense_candidate_index"
    assert payload["task"] == "offline_temporal_action_detection"


def test_temporal_contract_audits_exact_k_order_and_gap() -> None:
    audit = _contract().audit_positions(
        torch.tensor([[0, 2, 5, 7]]),
        torch.ones(1, 8, dtype=torch.bool),
    )

    assert audit["passed"] is True
    assert audit["rows"][0]["selected_count"] == 4
    assert audit["rows"][0]["max_unselected_hole_dense_candidates"] == 2
    assert audit["rows"][0]["max_selected_interval_source_frames"] == 12


def test_temporal_contract_rejects_mislabeled_g15_as_fifteen_source_frames() -> None:
    with pytest.raises(ValueError, match="quantized source-frame interval exceeds"):
        DucaTemporalSamplingContract(
            hard_budget=384,
            dense_window_size=768,
            max_unselected_hole_dense_candidates=15,
            dataset_feature_stride_source_frames=4,
            dataset_sample_stride=1,
            requested_max_source_frame_interval=15,
        )


@pytest.mark.parametrize(
    "positions,match",
    [
        (torch.tensor([[0, 2, 2, 7]]), "unique and strictly increasing"),
        (torch.tensor([[0, 1, 2, 7]]), "physical max-gap contract"),
        (torch.tensor([[0, 2, 5, -1]]), "requires K_eff"),
    ],
)
def test_temporal_contract_fails_closed_on_illegal_hard_selection(
    positions: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _contract().audit_positions(positions, torch.ones(1, 8, dtype=torch.bool))
