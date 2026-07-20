from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
import torch.nn as nn
import pytest

from opentad.models.duca.acquisition import DucaAcquisitionAdapter
from opentad.models.selectors.duca_online_frame_selector import (
    _add_protected_structured_transport_gradient_path,
    _add_structured_zero_forward_gradient_path,
)


ROOT = Path(__file__).resolve().parents[1]


def _grad_sum(module: nn.Module) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum().item())
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )


def test_structured_bridge_is_exact_hard_forward_but_only_has_local_surrogate_gradient() -> None:
    logits = torch.tensor([[[2.0, 0.0, -1.0]]], requires_grad=True)
    assignment = torch.softmax(logits, dim=-1)
    dense = torch.tensor([[[0.0, 1.0, 4.0]]])
    hard = dense[:, :, :1]

    bridged = _add_structured_zero_forward_gradient_path(
        hard,
        dense,
        soft_slot_assignment=assignment,
        slot_mask=torch.ones(1, 1, dtype=torch.bool),
        bridge_weight=1.0,
    )

    assert torch.equal(bridged.detach(), hard)
    (-bridged.mean()).backward()
    assert logits.grad is not None
    assert logits.grad[0, 0, 2] < 0
    assert logits.grad[0, 0, 0] > 0
    # This linear toy assertion is not evidence of alignment with discrete
    # ActionFormer cls+reg one-swap utility. The nonlinear alignment gate lives
    # in test_duca_counterfactual_utility.py and rejects this bridge.


@pytest.mark.parametrize(
    "assignment,match",
    [
        (torch.tensor([[[0.5, 0.5]]]), "shape must match"),
        (torch.tensor([[[0.5, float('nan'), 0.5]]]), "must be finite"),
        (torch.tensor([[[0.8, -0.1, 0.3]]]), "must be non-negative"),
        (torch.tensor([[[0.2, 0.2, 0.2]]]), "must sum to one"),
    ],
)
def test_structured_bridge_rejects_invalid_slot_marginals(assignment: torch.Tensor, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _add_structured_zero_forward_gradient_path(
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 3),
            soft_slot_assignment=assignment,
            slot_mask=torch.ones(1, 1, dtype=torch.bool),
            bridge_weight=1.0,
        )


def test_protected_transport_is_exact_hard_forward_and_has_policy_gradient() -> None:
    logits = torch.tensor(
        [[[2.0, 1.0, -1.0, -2.0, -3.0], [-3.0, -2.0, -1.0, 1.0, 2.0]]],
        requires_grad=True,
    )
    assignment = torch.softmax(logits, dim=-1)
    dense = torch.tensor(
        [[[0.0, 1.0, 4.0, 2.0, 5.0]]],
        requires_grad=True,
    )
    positions = torch.tensor([[1, 3]])
    slot_mask = torch.ones(1, 2, dtype=torch.bool)
    hard = torch.tensor([[[1.0, 2.0]]], requires_grad=True)

    bridged, expected_positions = _add_protected_structured_transport_gradient_path(
        hard,
        dense,
        selected_positions=positions,
        soft_slot_assignment=assignment,
        slot_mask=slot_mask,
        bridge_weight=0.25,
    )

    assert torch.equal(bridged.detach(), hard)
    assert expected_positions is not None and expected_positions.shape == positions.shape
    bridged.sum().backward()
    assert torch.equal(hard.grad, torch.ones_like(hard))
    assert dense.grad is None or torch.equal(dense.grad, torch.zeros_like(dense))
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0


def test_protected_transport_zero_weight_is_identity_without_surrogate() -> None:
    hard = torch.tensor([[[1.0, 2.0]]])
    out, expected_positions = _add_protected_structured_transport_gradient_path(
        hard,
        torch.tensor([[[0.0, 1.0, 4.0, 2.0, 5.0]]]),
        selected_positions=torch.tensor([[1, 3]]),
        soft_slot_assignment=torch.tensor(
            [[[0.5, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5, 0.5]]]
        ),
        slot_mask=torch.ones(1, 2, dtype=torch.bool),
        bridge_weight=0.0,
    )

    assert out is hard
    assert expected_positions is None


def test_acquisition_detector_only_loss_moves_structured_policy_and_preserves_hard_input() -> None:
    torch.manual_seed(13)
    adapter = DucaAcquisitionAdapter(
        feature_dim=1,
        hidden_dim=8,
        budget=2,
        max_radius=0,
        acquisition_policy="global_structured_topk",
        structured_temperature=0.7,
        max_unselected_hole=2,
        hard_max_gap_repair=False,
    )
    dense = torch.tensor([[[0.0], [0.5], [2.0], [0.2], [3.0]]])
    out = adapter.forward_acquire(dense, valid_mask=torch.ones(1, 5, dtype=torch.bool))

    assert torch.equal(out["detector_input"].detach(), out["hard_detector_input"].detach())
    detector = nn.Linear(1, 1, bias=False)
    detector.weight.data.fill_(1.0)
    detector_loss = -detector(out["detector_input"]).mean()
    detector_loss.backward()

    assert _grad_sum(adapter.center_head) > 0.0
    assert _grad_sum(adapter.encoder) > 0.0


def test_optimizer_covers_every_trainable_bridge_policy_parameter() -> None:
    adapter = DucaAcquisitionAdapter(
        feature_dim=2,
        hidden_dim=8,
        budget=2,
        max_radius=0,
        acquisition_policy="global_structured_topk",
        structured_temperature=0.7,
        max_unselected_hole=2,
        hard_max_gap_repair=False,
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1.0e-3)
    covered = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    missing = [name for name, parameter in adapter.named_parameters() if parameter.requires_grad and id(parameter) not in covered]

    assert missing == []


def test_official_actionformer_detector_losses_alone_reach_probe_and_selector() -> None:
    if os.name == "nt":
        pytest.skip("local Windows torch/c10.dll is unstable; Linux runs the official detector proof")
    script = ROOT / "tools" / "bata" / "run_duca_official_adatad_one_step_grad_proof.py"
    spec = importlib.util.spec_from_file_location("duca_detector_only_proof", script)
    assert spec is not None and spec.loader is not None
    proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proof)
    model, summary = proof.build_official_proof_model(
        config_path=ROOT / "configs" / "adatad" / "thumos" / "duca_online_official_adatad_backend_full_train.py",
        route="fixed384",
        proof_temporal_len=24,
        proof_budget=16,
        proof_budget_min=4,
        proof_budget_target=8,
        proof_budget_multiple=4,
        proof_spatial_size=16,
        proof_hidden_dim=16,
        proof_feature_dim=16,
    )
    assert summary["rpn_head_type"] == "ActionFormerHead"
    selector = model.frame_selector
    schedule = selector.loss_weight_schedule
    if schedule is not None and schedule.get("type") != "constant":
        selector._loss_weight_schedule_step.fill_(
            int(schedule["warmup_steps"]) + max(1, int(schedule["transition_steps"]))
        )
    inputs = torch.randn(1, 3, 24, 16, 16)
    losses = model(
        inputs,
        torch.ones(1, 24, dtype=torch.bool),
        [{"video_name": "detector_only_bridge_proof"}],
        gt_segments=[torch.tensor([[2.0, 21.0]])],
        gt_labels=[torch.tensor([1], dtype=torch.long)],
        return_loss=True,
    )
    model.zero_grad(set_to_none=True)
    detector_loss = losses["cls_loss"] + losses["reg_loss"]
    detector_loss.backward()

    assert _grad_sum(selector.raw_actionness_source) > 0.0
    assert _grad_sum(selector.adapter.encoder) > 0.0
    assert _grad_sum(selector.adapter.center_head) > 0.0
