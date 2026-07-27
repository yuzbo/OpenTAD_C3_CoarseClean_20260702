from __future__ import annotations

import pytest
import torch

from opentad.models.duca.rime import (
    RimeBudgetController,
    RimeCostLedger,
    calibrate_rime_price,
    decode_rime_exact_k,
)
from opentad.models.duca.structured_selection import (
    physical_exact_k_forward_backward,
    physical_exact_k_select,
    physical_exact_k_viterbi,
)


def _physical_inputs(temporal_len: int = 9):
    scores = torch.linspace(-1.0, 1.0, temporal_len)[None, :]
    seconds = torch.arange(temporal_len, dtype=torch.float64)[None, :]
    valid = torch.ones((1, temporal_len), dtype=torch.bool)
    return scores, seconds, valid


def _active_positions(output) -> set[int]:
    return set(
        int(value)
        for value in output.hard_positions[0, output.hard_slot_mask[0]].tolist()
    )


def test_required_physical_positions_are_hard_and_soft_mandatory():
    scores, seconds, valid = _physical_inputs(6)
    scores = scores.clone()
    scores[0, 3] = -100.0
    required = torch.zeros_like(valid)
    required[0, 3] = True

    hard = physical_exact_k_viterbi(
        scores,
        seconds,
        valid,
        k=3,
        required_mask=required,
    )
    soft = physical_exact_k_forward_backward(
        scores,
        seconds,
        valid,
        k=3,
        required_mask=required,
    )
    joint = physical_exact_k_select(
        scores.requires_grad_(),
        seconds,
        valid,
        k=3,
        required_mask=required,
    )

    assert 3 in hard.hard_positions[0, hard.hard_slot_mask[0]].tolist()
    assert soft.soft_occupancy[0, 3].item() == pytest.approx(1.0, abs=2.0e-4)
    assert joint.hard_occupancy[0, 3].item() == 1.0
    joint.soft_occupancy.sum().backward()
    assert scores.grad is not None


def test_independent_constant_evidence_is_canonical_exact_uniform():
    _, seconds, valid = _physical_inputs(9)
    output = decode_rime_exact_k(
        torch.zeros((1, 9)),
        seconds,
        valid,
        4,
        candidate_budgets=(2, 4),
        decoder_family="independent",
    )

    assert output.hard_positions.tolist() == [[0, 3, 5, 8]]
    assert output.constant_uniform_identity.tolist() == [True]
    assert output.ledger.requested_k == (4,)
    assert output.ledger.unique_k == (4,)
    assert output.ledger.backbone_input_k == (4,)
    assert output.ledger.padded_k == (4,)
    assert output.ledger.dynamic_compute_realized is True


def test_nested_and_weak_overlap_decoder_families_obey_their_contracts():
    scores, seconds, valid = _physical_inputs(9)
    scores = torch.tensor([[0.0, 3.0, 2.0, 9.0, 8.0, 1.0, 7.0, 6.0, 0.0]])
    nested_small = decode_rime_exact_k(
        scores,
        seconds,
        valid,
        2,
        candidate_budgets=(2, 4),
        decoder_family="strict_nested",
    )
    nested_large = decode_rime_exact_k(
        scores,
        seconds,
        valid,
        4,
        candidate_budgets=(2, 4),
        decoder_family="strict_nested",
    )
    weak_small = decode_rime_exact_k(
        scores,
        seconds,
        valid,
        2,
        candidate_budgets=(2, 4),
        decoder_family="weak_overlap",
        weak_overlap_fraction=0.5,
    )
    weak_large = decode_rime_exact_k(
        scores,
        seconds,
        valid,
        4,
        candidate_budgets=(2, 4),
        decoder_family="weak_overlap",
        weak_overlap_fraction=0.5,
    )

    assert _active_positions(nested_small) <= _active_positions(nested_large)
    assert len(_active_positions(weak_small) & _active_positions(weak_large)) >= 1


def test_controller_is_batch_invariant_and_no_risk_has_vector_fallback():
    torch.manual_seed(7)
    controller = RimeBudgetController(
        evidence_dim=3,
        candidate_budgets=(2, 4, 6),
        candidate_costs=(2.0, 4.0, 6.0),
        use_risk=False,
        frozen_price=0.2,
    ).eval()
    evidence = torch.randn(2, 7, 3)
    scores = torch.randn(2, 7)
    valid = torch.tensor(
        [[True] * 7, [True, True, True, True, False, False, False]]
    )

    together = controller(evidence, scores, valid)
    separate = [
        controller(evidence[index : index + 1], scores[index : index + 1], valid[index : index + 1])
        for index in range(2)
    ]

    assert together.fallback_to_kmax.shape == (2,)
    assert not together.fallback_to_kmax.any()
    for index, decision in enumerate(separate):
        assert torch.allclose(
            together.predicted_utility[index],
            decision.predicted_utility[0],
            atol=1.0e-6,
        )
        assert together.requested_k[index].item() == decision.requested_k[0].item()


def test_price_calibration_meets_attainable_mean_cost_without_test_labels():
    utility = torch.tensor(
        [
            [0.0, 1.0, 3.0],
            [0.0, 0.8, 1.1],
            [0.0, 0.3, 0.4],
            [0.0, 2.0, 2.1],
        ]
    )
    risk = torch.zeros_like(utility)
    cost = torch.tensor([[1.0, 2.0, 3.0]]).expand_as(utility)
    result = calibrate_rime_price(
        utility,
        risk,
        cost,
        target_mean_cost=2.0,
        risk_weight=0.0,
        risk_threshold=1.0,
        use_risk=False,
    )

    assert result["frozen_price"] >= 0.0
    assert result["realized_mean_cost"] <= 2.0
    assert len(result["selected_indices"]) == 4


def test_cost_ledger_fails_closed_on_pad_to_kmax():
    ledger = RimeCostLedger(
        requested_k=(2,),
        effective_k=(2,),
        unique_k=(2,),
        backbone_input_k=(4,),
        padded_k=(4,),
        risk_fallback=(False,),
        dynamic_compute_realized=False,
    )

    with pytest.raises(ValueError, match="padded"):
        ledger.validate(require_no_padding=True)


def test_mixed_k_batch_must_be_bucketed_before_heavy_execution():
    scores = torch.zeros((2, 6))
    seconds = torch.arange(6, dtype=torch.float64)[None, :].expand(2, -1)
    valid = torch.ones((2, 6), dtype=torch.bool)

    with pytest.raises(ValueError, match="homogeneous effective-K bucket"):
        decode_rime_exact_k(
            scores,
            seconds,
            valid,
            torch.tensor([2, 4]),
            candidate_budgets=(2, 4),
        )


def test_tail_window_quantizes_effective_k_without_padding_or_duplicates():
    temporal_len = 205
    scores = torch.zeros((1, temporal_len))
    seconds = torch.arange(temporal_len, dtype=torch.float64)[None, :]
    valid = torch.ones((1, temporal_len), dtype=torch.bool)

    output = decode_rime_exact_k(
        scores,
        seconds,
        valid,
        256,
        candidate_budgets=(192, 256),
        force_uniform=True,
        execution_quantum=16,
    )

    assert output.requested_k.tolist() == [256]
    assert output.effective_k.tolist() == [192]
    assert output.hard_positions.shape == (1, 192)
    assert len(set(output.hard_positions[0].tolist())) == 192
    assert output.ledger.requested_k == (256,)
    assert output.ledger.effective_k == (192,)
    assert output.ledger.backbone_input_k == (192,)
    assert output.ledger.padded_k == (192,)
