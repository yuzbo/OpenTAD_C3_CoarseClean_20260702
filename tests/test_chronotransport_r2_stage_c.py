import copy

import pytest
import torch
from torch import nn

from opentad.models.chronotransport.stage_c import (
    build_stage_c_parameter_groups,
    loss_specific_amp_step,
)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Linear(2, 2, bias=False)
        self.transport = nn.Linear(2, 2, bias=False)
        self.risk = nn.Linear(2, 1, bias=False)
        self.heavy = nn.Linear(2, 2, bias=False)


class _FakeScaler:
    def __init__(self, scale=8.0):
        self._scale = float(scale)
        self.scale_calls = []
        self.unscale_calls = 0
        self.step_calls = 0
        self.update_calls = 0

    def get_scale(self):
        return self._scale

    def scale(self, loss):
        self.scale_calls.append(self._scale)
        return loss * self._scale

    def unscale_(self, optimizer):
        self.unscale_calls += 1
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self._scale)

    def step(self, optimizer):
        self.step_calls += 1
        return optimizer.step()

    def update(self):
        self.update_calls += 1


def test_stage_c_ownership_uses_object_identity_and_freezes_everything_else():
    model = _Model()
    groups = build_stage_c_parameter_groups(
        model,
        adapter_modules=[model.adapter],
        transport_module=model.transport,
        risk_module=model.risk,
    )
    assert len(groups.adapters) == 1
    assert len(groups.transport) == 1
    assert len(groups.risk) == 1
    identities = [id(parameter) for parameter in groups.all]
    assert identities == sorted(identities)
    assert len(identities) == len(set(identities))
    assert all(parameter.requires_grad for parameter in groups.all)
    assert not any(parameter.requires_grad for parameter in model.heavy.parameters())


def test_stage_c_rejects_overlapping_parameter_ownership():
    model = _Model()
    with pytest.raises(ValueError, match="overlap"):
        build_stage_c_parameter_groups(
            model,
            adapter_modules=[model.adapter],
            transport_module=model.adapter,
            risk_module=model.risk,
        )


def test_loss_specific_amp_gradients_do_not_leak_feature_loss_into_adapter():
    model = _Model()
    groups = build_stage_c_parameter_groups(
        model,
        adapter_modules=[model.adapter],
        transport_module=model.transport,
        risk_module=model.risk,
    )
    optimizer = torch.optim.SGD(
        [
            {"params": groups.adapters},
            {"params": groups.transport},
            {"params": groups.risk},
        ],
        lr=0.01,
    )
    scaler = _FakeScaler()
    a = groups.adapters[0]
    t = groups.transport[0]
    r = groups.risk[0]
    before = [parameter.detach().clone() for parameter in groups.all]
    detector_loss = (a.sum() + t.sum()) ** 2
    feature_loss = (3.0 * a.sum() + 5.0 * t.sum()) ** 2
    risk_loss = r.sum() ** 2
    audit = loss_specific_amp_step(
        detector_loss=detector_loss,
        feature_loss=feature_loss,
        risk_loss=risk_loss,
        groups=groups,
        optimizer=optimizer,
        scaler=scaler,
        transport_executed=True,
        max_grad_norm=1000.0,
    )
    assert scaler.scale_calls == [8.0, 8.0, 8.0]
    assert scaler.unscale_calls == scaler.step_calls == scaler.update_calls == 1
    assert audit["adapter_detector_grad_nonzero"] is True
    assert audit["risk_grad_nonzero"] is True
    assert audit["transport_grad_finite"] is True
    assert all(not torch.equal(old, new) for old, new in zip(before, groups.all))


def test_transport_gradient_may_be_unused_only_without_transport_cells():
    model = _Model()
    groups = build_stage_c_parameter_groups(
        model,
        adapter_modules=[model.adapter],
        transport_module=model.transport,
        risk_module=model.risk,
    )
    optimizer = torch.optim.SGD(groups.all, lr=0.01)
    scaler = _FakeScaler()
    a, r = groups.adapters[0], groups.risk[0]
    audit = loss_specific_amp_step(
        detector_loss=a.sum() ** 2,
        feature_loss=(a.sum() * 0.0),
        risk_loss=r.sum() ** 2,
        groups=groups,
        optimizer=optimizer,
        scaler=scaler,
        transport_executed=False,
    )
    assert audit["transport_grad_nonzero"] is False
