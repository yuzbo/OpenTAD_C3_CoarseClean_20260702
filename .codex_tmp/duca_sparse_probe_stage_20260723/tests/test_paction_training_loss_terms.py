from __future__ import annotations

import os

import pytest

from tools.bata import train_paction_acquisition_policy as train_policy
from tools.bata import paction_acquisition_policy as policy


def test_paction_training_parser_exposes_model_loss_weights() -> None:
    parser = train_policy.build_arg_parser()
    args = parser.parse_args(
        [
            "--train-jsonl",
            "train.jsonl",
            "--out-dir",
            "out",
            "--value-transport-loss-weight",
            "0.5",
            "--boundary-miss-loss-weight",
            "4.0",
            "--large-gap-loss-weight",
            "2.5",
            "--temporal-hole-loss-weight",
            "0.75",
            "--budget-loss-weight",
            "0.25",
            "--redundancy-loss-weight",
            "0.0",
        ]
    )

    assert train_policy._parse_loss_terms(args) == {
        "value_transport": 0.5,
        "boundary_miss": 4.0,
        "large_gap": 2.5,
        "temporal_hole": 0.75,
        "budget": 0.25,
        "redundancy": 0.0,
    }


def test_paction_training_parser_accepts_explicit_loss_term_overrides() -> None:
    parser = train_policy.build_arg_parser()
    args = parser.parse_args(
        [
            "--train-jsonl",
            "train.jsonl",
            "--out-dir",
            "out",
            "--boundary-miss-loss-weight",
            "2.0",
            "--loss-term",
            "boundary_miss=5.0",
            "--loss-term",
            "large_gap=3.0",
        ]
    )

    assert train_policy._parse_loss_terms(args)["boundary_miss"] == 5.0
    assert train_policy._parse_loss_terms(args)["large_gap"] == 3.0


def test_paction_training_parser_rejects_unknown_loss_term() -> None:
    parser = train_policy.build_arg_parser()
    args = parser.parse_args(
        [
            "--train-jsonl",
            "train.jsonl",
            "--out-dir",
            "out",
            "--loss-term",
            "not_a_loss=1.0",
        ]
    )

    with pytest.raises(ValueError, match="unknown p_action loss term"):
        train_policy._parse_loss_terms(args)


def test_paction_loss_term_weights_change_training_objective() -> None:
    if os.name == "nt":
        pytest.skip("local Windows torch DLL loading is not reliable; run this objective test on the remote Linux training env")
    torch = pytest.importorskip("torch")
    outputs = {"frame_value": torch.zeros((1, 6), dtype=torch.float32)}
    action = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    boundary = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    valid = torch.ones((1, 6), dtype=torch.bool)
    low = policy.paction_gap_loss_training_objective(
        outputs,
        action_target=action,
        boundary_target=boundary,
        valid=valid,
        target_budget=torch.tensor([2.0]),
        gap_loss_max_gap=2,
        loss_terms={
            "value_transport": 0.0,
            "boundary_miss": 0.1,
            "large_gap": 0.1,
            "temporal_hole": 0.0,
            "budget": 0.0,
            "redundancy": 0.0,
        },
    )
    high = policy.paction_gap_loss_training_objective(
        outputs,
        action_target=action,
        boundary_target=boundary,
        valid=valid,
        target_budget=torch.tensor([2.0]),
        gap_loss_max_gap=2,
        loss_terms={
            "value_transport": 0.0,
            "boundary_miss": 4.0,
            "large_gap": 3.0,
            "temporal_hole": 2.0,
            "budget": 0.0,
            "redundancy": 0.0,
        },
    )

    assert float(high["total_loss"].item()) > float(low["total_loss"].item())
    assert "boundary_miss_loss" in high
    assert "large_gap_hole_loss" in high
