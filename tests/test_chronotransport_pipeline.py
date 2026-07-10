from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from opentad.models.chronotransport.cost_lookup import CostLookupKey, ScheduleCostLookup
from opentad.models.chronotransport.replay import (
    paired_detector_losses,
    records_sha256,
    validate_compact_record,
)
from opentad.models.chronotransport.training import (
    compose_stage_b_loss,
    configure_stage_b,
    configure_stage_c,
    validate_split_partition,
)


class ChronoTransportRuntime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forced_schedule = None
        self.transport = nn.Linear(2, 2)
        self.risk_predictor = nn.Linear(2, 1)


class FakeDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.chronotransport = ChronoTransportRuntime()
        self.adapter = nn.Linear(2, 2)
        self.heavy = nn.Linear(2, 2)

    def forward(self, inputs, **kwargs):
        del kwargs
        value = self.heavy(inputs)
        if self.chronotransport.forced_schedule != "dense":
            value = value + 0.25
        return {"loss_cls": value.square().mean()}


def test_paired_replay_restores_rng_and_produces_one_sided_regret() -> None:
    model = FakeDetector()
    torch.manual_seed(9)
    before = torch.get_rng_state().clone()
    result = paired_detector_losses(
        model,
        {"inputs": torch.ones(2, 2)},
        counterfactual_schedule="periodic2_transport",
    )
    assert torch.equal(before, torch.get_rng_state())
    assert result.regret.item() >= 0
    assert model.chronotransport.forced_schedule is None


def test_compact_ledger_rejects_forbidden_payload_and_hashes_deterministically() -> None:
    valid = {"sample_id": "a", "split": "train", "schedule": "dense", "cost": {}, "regret": 0.0}
    assert records_sha256([valid]) == records_sha256([dict(valid)])
    with pytest.raises(ValueError, match="forbidden"):
        validate_compact_record({**valid, "predictions": [1, 2, 3]})


def test_stage_b_and_c_trainable_parameter_contracts() -> None:
    model = FakeDetector()
    stage_b = configure_stage_b(model)
    assert stage_b and all("chronotransport" in name for name in stage_b)
    stage_c = configure_stage_c(model)
    assert any(name.startswith("adapter") for name in stage_c)
    assert all("heavy" not in name for name in stage_c)


def test_stage_b_loss_detaches_dense_reference() -> None:
    transported = torch.randn(2, 3, requires_grad=True)
    reference = torch.randn(2, 3, requires_grad=True)
    predicted = torch.randn(2, requires_grad=True)
    target = torch.ones(2, requires_grad=True)
    losses = compose_stage_b_loss(
        counterfactual_task_loss=torch.tensor(1.0, requires_grad=True),
        transported=transported,
        dense_reference=reference,
        predicted_quantile=predicted,
        regret_target=target,
        transport_weight=0.5,
        risk_weight=0.25,
        quantile=0.9,
    )
    losses.total.backward()
    assert transported.grad is not None
    assert predicted.grad is not None
    assert reference.grad is None
    assert target.grad is None


def test_split_partition_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        validate_split_partition(["a"], ["a"], ["b"])


def test_schedule_cost_lookup_is_shape_specific(tmp_path: Path) -> None:
    key = CostLookupKey("A100", "amp", 1, "periodic2_transport", (24, 24, 24))
    payload = ScheduleCostLookup.payload([(key, 3.0, 4.0)])
    path = tmp_path / "cost.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    lookup = ScheduleCostLookup.from_json(path)
    assert lookup.get(key, "p50") == 3.0
    other = CostLookupKey("A100", "amp", 2, "periodic2_transport", (24, 24, 24))
    with pytest.raises(KeyError):
        lookup.get(other)
