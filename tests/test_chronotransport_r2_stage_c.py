import copy
from contextlib import nullcontext
import random
import warnings

import numpy as np
import pytest
import torch
from mmengine.config import ConfigDict
from torch import nn

from opentad.models.backbones.vit_adapter import VisionTransformerAdapter
from opentad.cores.scheduler import LinearWarmupCosineAnnealingLR
from opentad.utils.ema import ModelEma
from opentad.models import build_detector
from opentad.models.chronotransport.training import configure_stage_c
from opentad.models.chronotransport.runtime import ChronoTransportRuntime
import opentad.models.chronotransport.stage_c as stage_c_module
from opentad.models.chronotransport.protocol import stage_c_batch_exposures
from opentad.models.chronotransport.formal_stage_c import (
    build_paired_stage_c_state,
    load_paired_stage_c_checkpoint,
    run_paired_stage_c_training,
    validate_paired_stage_c_checkpoint,
)
from opentad.models.chronotransport.stage_c import (
    StageCAttemptLosses,
    StageCInvalidImplementationError,
    StageCStateSurface,
    StageCTrackedEMA,
    build_stage_c_optimizer,
    build_stage_c_parameter_groups,
    build_matched_dense_optimizer,
    build_matched_dense_parameter_group,
    hash_materialized_batch,
    loss_specific_amp_step,
    run_stage_c_amp_with_retry,
    run_matched_dense_amp_with_retry,
    run_stage_c_amp_with_retry_for_test_only,
    stage_c_action_hash,
    validate_transport_gradient_ledger,
    validate_stage_c_optimizer,
)


_TEST_MEASURED_COST = {
    "recompute": (4.0, 4.0, 4.0),
    "transport": (0.6, 0.6, 0.6),
    "hold": (0.01, 0.01, 0.01),
    "scheduler_overhead": 0.0,
}


class _AdapterBlock(nn.Module):
    def __init__(self, enabled=True):
        super().__init__()
        self.use_adapter = enabled
        if enabled:
            self.adapter = nn.Linear(2, 2, bias=False)


class _Scheduler(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor


class _ScheduleEvidence:
    def __init__(self, actions):
        self.actions = actions


class _Runtime(nn.Module):
    def __init__(self):
        super().__init__()
        self.transport = nn.Linear(2, 2, bias=False)
        self.risk_predictor = nn.Linear(2, 1, bias=False)
        self.scheduler = _Scheduler(self.risk_predictor)
        self.register_buffer(
            "forced_actions", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.forced_schedule = None
        self.forced_action_name = "unset"
        self.latest_schedule = None
        self.latest_summary = None
        self.force_dense_execution = False

    def forward(self):
        actions = self.forced_actions.detach().clone()
        if self.force_dense_execution:
            actions = torch.zeros_like(actions)
        self.latest_schedule = _ScheduleEvidence(actions)
        self.latest_summary = {
            "forced_dense_exact_path": bool(self.force_dense_execution),
            "whole_window_dense_fallback": False,
            "schedule_repair_count": 0,
            "runtime_fail_closed_repairs": 0,
            "evidence_valid": not self.force_dense_execution,
        }
        return actions


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_AdapterBlock(True), _AdapterBlock(False)])
        self.chronotransport = _Runtime()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _Backbone()
        self.heavy = nn.Linear(2, 2, bias=False)

    @property
    def adapter(self):
        return self.backbone.blocks[0].adapter

    @property
    def transport(self):
        return self.backbone.chronotransport.transport

    @property
    def risk(self):
        return self.backbone.chronotransport.risk_predictor

    def forward(self):
        return self.backbone.chronotransport()


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


class _OverflowFakeScaler(_FakeScaler):
    def __init__(self, overflow_attempts):
        super().__init__()
        self._overflow_attempts = set(overflow_attempts)
        self._attempt = 0
        self._overflow = False

    def unscale_(self, optimizer):
        super().unscale_(optimizer)
        self._overflow = self._attempt in self._overflow_attempts

    def step(self, optimizer):
        self.step_calls += 1
        if not self._overflow:
            return optimizer.step()
        return None

    def update(self):
        self.update_calls += 1
        if self._overflow:
            self._scale *= 0.5
        self._attempt += 1


class _GradientAwareFakeScaler(_FakeScaler):
    def __init__(self):
        super().__init__()
        self._overflow = False

    def unscale_(self, optimizer):
        super().unscale_(optimizer)
        self._overflow = any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for group in optimizer.param_groups
            for parameter in group["params"]
        )

    def step(self, optimizer):
        self.step_calls += 1
        if not self._overflow:
            return optimizer.step()
        return None

    def update(self):
        self.update_calls += 1
        if self._overflow:
            self._scale *= 0.5


class _BadOverflowScaler(_OverflowFakeScaler):
    def step(self, optimizer):
        self.step_calls += 1
        return optimizer.step()


class _Stateful:
    def __init__(self):
        self.value = 0
        self.python_counter = 0
        self.latest_output = None

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


class _EMAState(_Stateful):
    def __init__(self, mode="once"):
        super().__init__()
        self.stage_c_update_count = 0
        self.mode = mode

    def update(self, model):
        del model
        if self.mode == "noop":
            return
        self.stage_c_update_count += 2 if self.mode == "twice" else 1
        self.value += 2 if self.mode == "twice" else 1


class _RetryModel(_Model):
    def __init__(self):
        super().__init__()
        self.register_buffer("forward_counter", torch.zeros((), dtype=torch.int64))
        self.register_buffer(
            "forced_actions",
            torch.zeros(2, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "graph_buffer",
            torch.ones((2, 2), dtype=torch.float32),
            persistent=False,
        )
        self.python_forward_counter = 0
        self.latest_actions = []


class _Depth12RetryModel(_RetryModel):
    def __init__(self):
        super().__init__()
        self.backbone.blocks = nn.ModuleList(
            [_AdapterBlock(index in (3, 5, 7, 11)) for index in range(12)]
        )


class _FormalToyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)

    def forward(self, value, h, w):
        del h, w
        return value + self.proj(value)


class _FormalToyBlock(nn.Module):
    def __init__(self, use_adapter):
        super().__init__()
        self.norm1 = nn.Identity()
        self.attn = nn.Linear(2, 2, bias=False)
        self.norm2 = nn.Identity()
        self.mlp = nn.Linear(2, 2, bias=False)
        self.drop_path = nn.Identity()
        self.use_adapter = bool(use_adapter)
        self.with_cp = False
        if self.use_adapter:
            self.adapter = _FormalToyAdapter()

    def forward(self, value, h, w):
        value = value + self.drop_path(self.attn(self.norm1(value)))
        value = value + self.drop_path(self.mlp(self.norm2(value)))
        if self.use_adapter:
            value = self.adapter(value, h, w)
        return value


class _FormalToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_FormalToyBlock(index in (3, 5, 7, 11)) for index in range(12)]
        )
        self.chronotransport = ChronoTransportRuntime(
            embed_dims=2,
            depth=12,
            chunks_per_window=48,
            layer_groups=((0, 4), (4, 8), (8, 12)),
            signal_dims=6,
            risk_hidden_dims=4,
            transport_bottleneck_dims=2,
            hard_cache_validity_age=47,
            transport_age_embedding_cap=8,
            cache_detach=True,
            profile_sync_cuda=False,
            measured_cost=_TEST_MEASURED_COST,
        )
        self.chronotransport.capture_replay_signals = True
        self.projection = nn.Linear(2, 2, bias=False)
        self.head = nn.Linear(2, 2, bias=False)
        self.register_buffer(
            "toy_input",
            torch.linspace(-0.5, 0.5, 96 * 2).reshape(96, 1, 2),
            persistent=False,
        )

    def forward(self):
        features = self.chronotransport(self.toy_input, self.blocks, 1, 1)
        return self.head(self.projection(features))


class _FormalRetryModel(_FormalToyModel):
    def __init__(self):
        super().__init__()
        self.heavy = nn.Linear(2, 2, bias=False)
        self.register_buffer("forward_counter", torch.zeros((), dtype=torch.int64))
        self.register_buffer(
            "forced_actions", torch.zeros(2, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "graph_buffer",
            torch.ones((2, 2), dtype=torch.float32),
            persistent=False,
        )
        self.python_forward_counter = 0
        self.latest_actions = []

    @property
    def backbone(self):
        return self


def test_stage_c_ownership_uses_object_identity_and_freezes_everything_else():
    model = _Model()
    groups = build_stage_c_parameter_groups(model)
    assert len(groups.adapters) == 1
    assert len(groups.transport) == 1
    assert len(groups.risk) == 1
    identities = [id(parameter) for parameter in groups.all]
    assert identities == sorted(identities)
    assert len(identities) == len(set(identities))
    assert all(parameter.requires_grad for parameter in groups.all)
    assert not any(parameter.requires_grad for parameter in model.heavy.parameters())


def test_model_derived_adapter_registry_rejects_disabled_block_adapter_and_freezes_name_traps():
    model = _Model()
    model.fake_adapter = nn.Linear(2, 2, bias=False)
    names = configure_stage_c(model)
    assert names
    assert not model.fake_adapter.weight.requires_grad
    assert all("fake_adapter" not in name for name in names)

    model.backbone.blocks[1].adapter = nn.Linear(2, 2, bias=False)
    with pytest.raises(ValueError, match="disabled adapter block"):
        build_stage_c_parameter_groups(model)


def test_stage_c_rejects_caller_supplied_ownership_and_bad_model_aliases():
    model = _Model()
    with pytest.raises(TypeError, match="transport_module"):
        build_stage_c_parameter_groups(model, transport_module=model.adapter)

    model.backbone.chronotransport.scheduler.predictor = nn.Linear(2, 1, bias=False)
    with pytest.raises(ValueError, match="must alias"):
        build_stage_c_parameter_groups(model)


def test_stage_c_optimizer_contains_every_owned_object_exactly_once():
    model = _Model()
    groups = build_stage_c_parameter_groups(model)
    valid = build_stage_c_optimizer(groups)
    assert [group["stage_c_group"] for group in valid.param_groups] == ["A", "T", "R"]
    assert [group["lr"] for group in valid.param_groups] == [2e-4, 1e-4, 1e-4]
    assert [group["stage_c_base_lr"] for group in valid.param_groups] == [2e-4, 1e-4, 1e-4]
    assert [group["weight_decay"] for group in valid.param_groups] == [0.05, 0.0, 0.0]
    valid_scheduler = _build_lr_scheduler(valid)
    assert validate_stage_c_optimizer(groups, valid, lr_scheduler=valid_scheduler) == len(groups.all)

    missing = torch.optim.AdamW(groups.adapters + groups.transport, lr=2e-4, weight_decay=0.05)
    with pytest.raises(ValueError, match="three ordered groups"):
        validate_stage_c_optimizer(groups, missing, lr_scheduler=_build_lr_scheduler(missing))

    # PyTorch rejects duplicates across groups, but permits them inside one group.
    with pytest.warns(UserWarning, match="duplicate parameters"):
        duplicate = torch.optim.AdamW(
            [groups.adapters[0], groups.adapters[0], *groups.transport, *groups.risk],
            lr=2e-4,
        )
    with pytest.raises(ValueError, match="three ordered groups"):
        validate_stage_c_optimizer(groups, duplicate, lr_scheduler=_build_lr_scheduler(duplicate))

    for group in valid.param_groups:
        group["lr"] = 0.0
    assert validate_stage_c_optimizer(groups, valid, lr_scheduler=valid_scheduler) == len(groups.all)
    valid.param_groups[1]["lr"] = float("nan")
    with pytest.raises(ValueError, match="current LR"):
        validate_stage_c_optimizer(groups, valid, lr_scheduler=valid_scheduler)
    valid.param_groups[1]["lr"] = 1e-4
    valid.param_groups[1]["stage_c_base_lr"] = 2e-4
    with pytest.raises(ValueError, match="hyperparameters"):
        validate_stage_c_optimizer(groups, valid, lr_scheduler=valid_scheduler)


def _build_lr_scheduler(optimizer, *, at_base=False):
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epoch=350,
        max_epoch=7000,
    )
    if at_base:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scheduler.step(350)
    return scheduler


def test_stage_c_binds_real_opentad_scheduler_and_rejects_forged_base_marker():
    model = _Model()
    groups = build_stage_c_parameter_groups(model)
    optimizer = build_stage_c_optimizer(groups)
    scheduler = _build_lr_scheduler(optimizer)
    assert [group["lr"] for group in optimizer.param_groups] == [0.0, 0.0, 0.0]
    assert validate_stage_c_optimizer(groups, optimizer, lr_scheduler=scheduler) == len(groups.all)

    forged_optimizer = build_stage_c_optimizer(groups)
    for group in forged_optimizer.param_groups:
        group["initial_lr"] = group["stage_c_base_lr"] * 2.0
    forged_scheduler = _build_lr_scheduler(forged_optimizer)
    with pytest.raises(ValueError, match="base_lrs|initial_lr"):
        validate_stage_c_optimizer(groups, forged_optimizer, lr_scheduler=forged_scheduler)

    scheduler.warmup_start_lr = 0.5
    with pytest.raises(ValueError, match="warmup_start_lr"):
        validate_stage_c_optimizer(groups, optimizer, lr_scheduler=scheduler)
    scheduler.warmup_start_lr = 0.0
    scheduler.eta_min = 0.5
    with pytest.raises(ValueError, match="eta_min"):
        validate_stage_c_optimizer(groups, optimizer, lr_scheduler=scheduler)
    scheduler.eta_min = 1e-8
    for group in optimizer.param_groups:
        group["lr"] = 0.123
    scheduler._last_lr = [0.123, 0.123, 0.123]
    with pytest.raises(ValueError, match="closed-form"):
        validate_stage_c_optimizer(groups, optimizer, lr_scheduler=scheduler)


def test_parameter_snapshot_clones_bytes_only_for_trainable_a_t_r():
    model = _RetryModel()
    groups = build_stage_c_parameter_groups(model)
    stats = stage_c_module.stage_c_parameter_snapshot_stats(model, groups)
    expected = sum(parameter.numel() * parameter.element_size() for parameter in groups.all)
    heavy = sum(parameter.numel() * parameter.element_size() for parameter in model.heavy.parameters())
    assert stats["cloned_parameter_bytes"] == expected
    assert stats["frozen_parameter_bytes_cloned"] == 0
    assert stats["frozen_parameter_bytes_covered_by_metadata"] >= heavy


def test_loss_specific_amp_gradients_do_not_leak_feature_loss_into_adapter():
    model = _Model()
    groups = build_stage_c_parameter_groups(model)
    optimizer = build_stage_c_optimizer(groups)
    lr_scheduler = _build_lr_scheduler(optimizer, at_base=True)
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
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        action_payload=_action_payload(groups),
    )
    assert scaler.scale_calls == [8.0, 8.0, 8.0]
    assert scaler.unscale_calls == scaler.step_calls == scaler.update_calls == 1
    assert audit["adapter_detector_grad_nonzero"] is True
    assert audit["risk_grad_nonzero"] is True
    assert audit["transport_grad_finite"] is True
    assert all(not torch.equal(old, new) for old, new in zip(before, groups.all))


def test_transport_gradient_may_be_unused_only_without_transport_cells():
    model = _Model()
    groups = build_stage_c_parameter_groups(model)
    optimizer = build_stage_c_optimizer(groups)
    lr_scheduler = _build_lr_scheduler(optimizer, at_base=True)
    scaler = _FakeScaler()
    a, r = groups.adapters[0], groups.risk[0]
    audit = loss_specific_amp_step(
        detector_loss=a.sum() ** 2,
        feature_loss=(a.sum() * 0.0),
        risk_loss=r.sum() ** 2,
        groups=groups,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        action_payload=_dense_action_payload(groups),
    )
    assert audit["transport_grad_nonzero"] is False


@pytest.mark.parametrize("missing", ["adapter", "risk"])
def test_expected_unused_rejects_missing_required_aggregate_gradient(missing):
    model = _Model()
    groups = build_stage_c_parameter_groups(model)
    optimizer = build_stage_c_optimizer(groups)
    lr_scheduler = _build_lr_scheduler(optimizer, at_base=True)
    a, t, r = groups.adapters[0], groups.transport[0], groups.risk[0]
    detector_loss = (a.sum() + t.sum()) ** 2
    feature_loss = t.sum() ** 2
    risk_loss = r.sum() ** 2
    expected = missing
    if missing == "adapter":
        detector_loss = (a.sum() * 0.0 + t.sum()) ** 2
    elif missing == "risk":
        risk_loss = r.sum() * 0.0
    with pytest.raises(RuntimeError, match=expected):
        loss_specific_amp_step(
            detector_loss=detector_loss,
            feature_loss=feature_loss,
            risk_loss=risk_loss,
            groups=groups,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=_FakeScaler(),
            action_payload=_action_payload(groups),
        )


def test_transport_gradient_is_finite_per_exposure_but_nonzero_only_in_aggregate_ledger():
    model = _Model()
    groups = build_stage_c_parameter_groups(model)
    optimizer = build_stage_c_optimizer(groups)
    lr_scheduler = _build_lr_scheduler(optimizer, at_base=True)
    a, r = groups.adapters[0], groups.risk[0]
    audit = loss_specific_amp_step(
        detector_loss=a.sum() ** 2,
        feature_loss=a.sum() * 0.0,
        risk_loss=r.sum() ** 2,
        groups=groups,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=_FakeScaler(),
        action_payload=_action_payload(groups),
    )
    assert audit["transport_grad_finite"] is True
    assert audit["transport_grad_norm"] == 0.0
    zero_row = {
        "transport_executed": True,
        "transport_grad_finite": True,
        "transport_grad_norm": 0.0,
    }
    with pytest.raises(ValueError, match="aggregate"):
        validate_transport_gradient_ledger([zero_row], expected_transport_exposures=1)
    assert validate_transport_gradient_ledger(
        [zero_row, {**zero_row, "transport_grad_norm": 0.25}],
        expected_transport_exposures=2,
    )["aggregate_transport_grad_norm"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="expected TRANSPORT exposures"):
        validate_transport_gradient_ledger([], expected_transport_exposures=1)


def test_loss_finiteness_fixed_float64_clip_and_three_scale_trace():
    model = _Model()
    groups = build_stage_c_parameter_groups(model)
    optimizer = build_stage_c_optimizer(groups)
    lr_scheduler = _build_lr_scheduler(optimizer, at_base=True)
    scaler = _FakeScaler()
    a, t, r = groups.adapters[0], groups.transport[0], groups.risk[0]
    audit = loss_specific_amp_step(
        detector_loss=1e6 * (a.sum() + t.sum()) ** 2,
        feature_loss=1e6 * t.sum() ** 2,
        risk_loss=1e6 * r.sum() ** 2,
        groups=groups,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        action_payload=_action_payload(groups),
    )
    assert audit["losses_finite"] is True
    assert audit["scale_trace"] == [8.0, 8.0, 8.0]
    assert audit["preclip_global_grad_norm_float64"] > 1.0
    assert audit["postclip_global_grad_norm_float64"] <= 1.0 + 1e-6
    assert audit["postclip_gradients_finite"] is True

    optimizer = build_stage_c_optimizer(groups)
    lr_scheduler = _build_lr_scheduler(optimizer, at_base=True)
    before = [parameter.detach().clone() for parameter in groups.all]
    with pytest.raises(StageCInvalidImplementationError, match="non-finite loss"):
        loss_specific_amp_step(
            detector_loss=(a.sum() + t.sum()) ** 2 + torch.tensor(float("inf")),
            feature_loss=t.sum() ** 2,
            risk_loss=r.sum() ** 2,
            groups=groups,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=_FakeScaler(),
            action_payload=_action_payload(groups),
        )
    assert all(torch.equal(old, parameter) for old, parameter in zip(before, groups.all))


def test_materialized_batch_hash_binds_augmentation_tensor_bytes():
    first = {"frames": torch.tensor([[1.0, 2.0]]), "augmentation": {"flip": False}}
    second = {"frames": torch.tensor([[1.0, 3.0]]), "augmentation": {"flip": False}}
    assert hash_materialized_batch(first) == hash_materialized_batch(copy.deepcopy(first))
    assert hash_materialized_batch(first) != hash_materialized_batch(second)


def _retry_fixture(overflow_attempts):
    model = _FormalRetryModel().train()
    groups = build_stage_c_parameter_groups(model)
    optimizer = build_stage_c_optimizer(groups)
    scheduler = _build_lr_scheduler(optimizer)
    scaler = _OverflowFakeScaler(overflow_attempts)
    objects = {
        name: _Stateful()
        for name in (
            "diagnostics",
            "profiler",
            "sampler",
            "successful_cursor",
            "exposure_cursor",
            "shadow_ledger",
        )
    }
    objects["ema"] = _EMAState()
    objects["scheduler"] = scheduler
    objects["shadow_ledger"] = []
    return model, groups, optimizer, scaler, objects


def _duck_retry_fixture(overflow_attempts):
    model = _RetryModel()
    groups = build_stage_c_parameter_groups(model)
    optimizer = build_stage_c_optimizer(groups)
    scheduler = _build_lr_scheduler(optimizer)
    scaler = _OverflowFakeScaler(overflow_attempts)
    objects = {
        name: _Stateful()
        for name in (
            "diagnostics",
            "profiler",
            "sampler",
            "successful_cursor",
            "exposure_cursor",
            "shadow_ledger",
        )
    }
    objects["ema"] = _EMAState()
    objects["scheduler"] = scheduler
    objects["shadow_ledger"] = []
    return model, groups, optimizer, scaler, objects


def _action_payload(groups, *, seed=3407, successful_update=0):
    device = groups.adapters[0].device
    exposures = stage_c_batch_exposures(seed, successful_update)
    return torch.stack(
        [
            stage_c_module._R2_STAGE_C_ACTIONS[row["candidate"]]
            for row in exposures
        ]
    ).to(device=device)


def _dense_action_payload(groups):
    return torch.zeros((2, 48, 3), dtype=torch.int8, device=groups.adapters[0].device)


def _losses_only(groups):
    a, t, r = groups.adapters[0], groups.transport[0], groups.risk[0]
    return StageCAttemptLosses(
        detector_loss=(a.sum() + t.sum()) ** 2,
        feature_loss=t.sum() ** 2,
        risk_loss=r.sum() ** 2,
    )


def _direct_losses_only(groups):
    a, t, r = groups.adapters[0], groups.transport[0], groups.risk[0]
    return StageCAttemptLosses(
        detector_loss=a.sum() + t.sum(),
        feature_loss=t.sum(),
        risk_loss=r.sum(),
    )


def _finite_attempt(model, groups):
    return _formal_toy_bound_attempt(model, groups)


def _formal_toy_fixture(overflow_attempts):
    model = _FormalToyModel().train()
    groups = build_stage_c_parameter_groups(model)
    optimizer = build_stage_c_optimizer(groups)
    scheduler = _build_lr_scheduler(optimizer)
    scaler = _OverflowFakeScaler(overflow_attempts)
    objects = {
        name: _Stateful()
        for name in (
            "diagnostics",
            "profiler",
            "sampler",
            "successful_cursor",
            "exposure_cursor",
            "shadow_ledger",
        )
    }
    objects["ema"] = _EMAState()
    objects["scheduler"] = scheduler
    objects["shadow_ledger"] = []
    return model, groups, optimizer, scaler, objects


def _formal_bound_losses_after_forward(
    model, detector_output, *, detector_multiplier=1.0
):
    runtime = model.chronotransport
    feature_output = runtime.latest_output
    risk_output = runtime.risk_predictor(
        runtime.latest_signals, runtime.latest_schedule.actions
    )
    return StageCAttemptLosses(
        detector_loss=(
            detector_output.float().square().mean() * detector_multiplier
        ),
        feature_loss=feature_output.float().square().mean(),
        risk_loss=risk_output.float().square().mean(),
    )


def _formal_toy_bound_attempt(model, groups):
    del groups
    return _formal_bound_losses_after_forward(model, model())


def _formal_vit_stage_c_fixture(device):
    model = VisionTransformerAdapter(
        img_size=16,
        patch_size=16,
        in_channels=3,
        embed_dims=8,
        depth=12,
        num_heads=2,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        num_frames=16,
        tubelet_size=2,
        use_mean_pooling=True,
        return_feat_map=True,
        with_cp=True,
        adapter_mlp_ratio=0.5,
        total_frames=768,
        adapter_index=[3, 5, 7, 11],
        chronotransport={
            "enabled": True,
            "layer_groups": [(0, 4), (4, 8), (8, 12)],
            "signal_dims": 6,
            "risk_hidden_dims": 8,
            "transport_bottleneck_dims": 4,
            "risk_quantile": 0.9,
            "risk_epsilon": 1.0,
            "hard_cache_validity_age": 47,
            "transport_age_embedding_cap": 8,
            "forced_schedule": None,
            "cache_detach": True,
            "profile_sync_cuda": False,
            "measured_cost": _TEST_MEASURED_COST,
            "allow_unmeasured_cost_for_debug": False,
            "risk_ready": False,
            "require_checkpoint_for_dynamic": False,
            "allow_legacy_checkpoint": False,
        },
    ).to(device).train()
    groups = build_stage_c_parameter_groups(model)
    actions = _action_payload(groups)
    model.chronotransport.forced_actions = actions.to(device=device, dtype=torch.long)
    model.chronotransport.forced_action_name = "stage_c_protocol_batch"
    optimizer = build_stage_c_optimizer(groups)
    scheduler = _build_lr_scheduler(optimizer)
    ema = StageCTrackedEMA(ModelEma(model, decay=0.999, device=device))
    objects = {
        "ema": ema,
        "scheduler": scheduler,
        "diagnostics": StageCStateSurface(),
        "profiler": StageCStateSurface(),
        "sampler": StageCStateSurface(),
        "successful_cursor": StageCStateSurface(),
        "exposure_cursor": StageCStateSurface(),
        "shadow_ledger": [],
    }
    frames = torch.randn(96, 3, 16, 16, 16, device=device)
    return model, groups, optimizer, objects, frames, actions


def _formal_amp_context(device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()


def _run_formal_vit_stage_c_smoke(device, scaler):
    model, groups, optimizer, objects, frames, actions = _formal_vit_stage_c_fixture(device)
    dense_actions = torch.zeros_like(actions, dtype=torch.long, device=device)
    model.chronotransport.forced_actions = dense_actions
    with _formal_amp_context(device):
        dense_output = model(frames)
    assert model.latest_chronotransport_summary["forced_dense_exact_path"] is True
    assert torch.isfinite(dense_output).all()
    del dense_output

    model.chronotransport.forced_actions = actions.to(device=device, dtype=torch.long)
    calls = []

    def attempt():
        with _formal_amp_context(device):
            output = model(frames)
            multiplier = float("inf") if not calls else 1.0
            calls.append(multiplier)
            detector_loss = output.square().mean() * multiplier
            feature_loss = model.chronotransport.latest_output.square().mean()
            risk = model.chronotransport.risk_predictor(
                model.chronotransport.latest_signals,
                model.chronotransport.latest_schedule.actions,
            )
            risk_loss = risk.square().mean()
        return StageCAttemptLosses(
            detector_loss=detector_loss,
            feature_loss=feature_loss,
            risk_loss=risk_loss,
        )

    result = run_stage_c_amp_with_retry_for_test_only(
        materialized_batch={"frames_sha": "formal-videomae-stage-c-smoke"},
        attempt=attempt,
        model=model,
        groups=groups,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=objects["scheduler"],
        rollback_objects=objects,
        seed=3407,
    )
    assert result["attempts"] == 2
    assert model.latest_chronotransport_summary["transport_rows"] > 0
    assert objects["ema"].stage_c_update_count == 1
    assert objects["scheduler"].last_epoch == 1
    assert objects["successful_cursor"].value == 1
    assert (
        model.chronotransport.schedule_library
        is model.chronotransport.scheduler.schedule_library
    )
    return result


def test_retry_requires_all_state_surfaces_and_success_advancers():
    model, groups, optimizer, scaler, objects = _retry_fixture(set())
    incomplete = dict(objects)
    del incomplete["profiler"]
    with pytest.raises(ValueError, match="profiler"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": torch.ones(2, 2)},
            attempt=lambda: _finite_attempt(model, groups),
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=incomplete,
            seed=3407,
        )



def test_amp_overflow_restores_full_attempt_state_and_replays_same_rng_batch_action():
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    model, groups, optimizer, scaler, objects = _retry_fixture({0})
    materialized = {"frames": torch.arange(8).reshape(2, 4), "augmentation": {"crop": [1, 3]}}
    initial_model = copy.deepcopy(model.state_dict())
    forced_actions_reference = model.forced_actions
    rng_samples = []
    forced_action_starts = []
    retained_buffer_gradients = []
    attempt_batches = []
    attempt_actions = []
    audit = []
    retained_buffer_graph = (groups.adapters[0] * model.graph_buffer).sum()

    def attempt():
        detector_output = model()
        attempt_batches.append(hash_materialized_batch(materialized))
        action = _action_payload(groups)
        attempt_actions.append(
            tuple(
                stage_c_action_hash(
                    seed=3407,
                    successful_update=0,
                    batch_position=position,
                    action_payload=action,
                )
                for position in range(2)
            )
        )
        rng_samples.append((random.random(), float(np.random.rand()), float(torch.rand(()))))
        forced_action_starts.append(model.forced_actions.detach().clone())
        retained_buffer_gradients.append(
            torch.autograd.grad(
                retained_buffer_graph,
                groups.adapters[0],
                retain_graph=True,
            )[0].detach().clone()
        )
        assert model.heavy.training is True
        if len(attempt_batches) == 1:
            model.heavy.train(False)
            model.python_forward_counter += 1
            model.latest_actions.append(action.detach().clone())
        objects["diagnostics"].value += 1
        objects["diagnostics"].python_counter += 1
        objects["diagnostics"].latest_output = groups.adapters[0].sum() * 2.0
        objects["profiler"].value += 1
        return _formal_bound_losses_after_forward(model, detector_output)

    result = run_stage_c_amp_with_retry_for_test_only(
        materialized_batch=materialized,
        attempt=attempt,
        model=model,
        groups=groups,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=objects["scheduler"],
        rollback_objects=objects,
        seed=3407,
        retry_audit=audit,
    )

    assert result["status"] == "SUCCESS"
    assert result["attempts"] == 2
    assert result["retries"] == 1
    assert attempt_batches == [result["batch_hash"], result["batch_hash"]]
    assert attempt_actions == [attempt_actions[0], attempt_actions[0]]
    assert rng_samples[0] == rng_samples[1]
    assert [values.tolist() for values in forced_action_starts] == [[0, 0], [0, 0]]
    assert model.forced_actions is forced_actions_reference
    assert forced_actions_reference.tolist() == [0, 0]
    assert model.forced_actions.tolist() == [0, 0]
    assert len(retained_buffer_gradients) == 2
    assert torch.equal(retained_buffer_gradients[0], retained_buffer_gradients[1])
    assert scaler.step_calls == scaler.update_calls == 2
    assert scaler._scale == 4.0
    assert model.python_forward_counter == 0
    assert model.latest_actions == []
    assert model.heavy.training is True
    assert objects["diagnostics"].value == objects["profiler"].value == 1
    assert objects["diagnostics"].python_counter == 1
    assert objects["ema"].stage_c_update_count == 1
    assert objects["scheduler"].last_epoch == 1
    assert objects["sampler"].value == objects["successful_cursor"].value == 1
    assert objects["exposure_cursor"].value == 2
    assert len(objects["shadow_ledger"]) == 1
    assert [row["overflow"] for row in audit] == [True, False]
    assert all(row["batch_hash"] == result["batch_hash"] for row in audit)
    assert all(torch.equal(initial_model[name], value) for name, value in model.state_dict().items())
    assert optimizer.state
    assert all(int(state["step"].item()) == 1 for state in optimizer.state.values())


def test_two_consecutive_batches_restore_state_dict_plus_python_and_nonleaf_latest_output():
    model, groups, optimizer, scaler, objects = _retry_fixture(set())
    run_stage_c_amp_with_retry_for_test_only(
        materialized_batch={"frames": torch.ones(2, 2), "batch": 0},
        attempt=lambda: _finite_attempt(model, groups),
        model=model,
        groups=groups,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=objects["scheduler"],
        rollback_objects=objects,
        seed=3407,
    )
    objects["diagnostics"].latest_output = groups.adapters[0].sum() * 3.0
    objects["diagnostics"].python_counter = 9
    initial_latest = objects["diagnostics"].latest_output.detach().clone()
    retry_scaler = _OverflowFakeScaler({0})
    starts = []
    old_graph_gradients = []

    def second_attempt():
        starts.append(
            (
                objects["diagnostics"].python_counter,
                objects["diagnostics"].latest_output,
            )
        )
        old_graph_gradients.append(
            torch.autograd.grad(
                objects["diagnostics"].latest_output,
                groups.adapters[0],
                retain_graph=True,
            )[0].detach().clone()
        )
        objects["diagnostics"].python_counter += 1
        objects["diagnostics"].latest_output = groups.adapters[0].sum() * 4.0
        return _finite_attempt(model, groups)

    result = run_stage_c_amp_with_retry_for_test_only(
        materialized_batch={"frames": torch.ones(2, 2), "batch": 1},
        attempt=second_attempt,
        model=model,
        groups=groups,
        optimizer=optimizer,
        scaler=retry_scaler,
        lr_scheduler=objects["scheduler"],
        rollback_objects=objects,
        seed=3407,
    )
    assert result["attempts"] == 2
    assert [value for value, _ in starts] == [9, 9]
    assert all(latest is starts[0][1] for _, latest in starts)
    assert starts[0][1] is not objects["diagnostics"].latest_output
    assert all(latest.requires_grad and not latest.is_leaf and latest.grad_fn is not None for _, latest in starts)
    assert all(torch.equal(latest.detach(), initial_latest) for _, latest in starts)
    assert len(old_graph_gradients) == 2
    assert torch.equal(old_graph_gradients[0], old_graph_gradients[1])
    assert objects["diagnostics"].python_counter == 10
    assert objects["ema"].stage_c_update_count == 2
    assert objects["scheduler"].last_epoch == 2
    assert objects["successful_cursor"].value == 2
    assert objects["exposure_cursor"].value == 4
    assert len(objects["shadow_ledger"]) == 2


def test_inplace_python_tensor_version_mutation_is_invalid_instead_of_fake_restore():
    model, groups, optimizer, scaler, objects = _retry_fixture({0})
    objects["diagnostics"].latest_output = groups.adapters[0].sum() * 3.0
    original = objects["diagnostics"].latest_output
    original_version = original._version

    def attempt():
        objects["diagnostics"].latest_output.add_(1.0)
        return _finite_attempt(model, groups)

    with pytest.raises(StageCInvalidImplementationError, match="in-place Python Tensor"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": torch.ones(2, 2)},
            attempt=attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )
    assert objects["diagnostics"].latest_output is original
    assert original._version > original_version


def test_inplace_buffer_version_mutation_with_retained_graph_is_invalid():
    model, groups, optimizer, scaler, objects = _retry_fixture({0})
    retained = (groups.adapters[0] * model.graph_buffer).sum()

    def attempt():
        torch.autograd.grad(retained, groups.adapters[0], retain_graph=True)
        model.graph_buffer.add_(1.0)
        return _finite_attempt(model, groups)

    with pytest.raises(StageCInvalidImplementationError, match="buffer.*version"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": torch.ones(2, 2)},
            attempt=attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_overflow_rejects_any_parameter_value_or_version_mutation_including_frozen_params():
    model, groups, optimizer, scaler, objects = _retry_fixture({0})

    def attempt():
        with torch.no_grad():
            model.heavy.weight.add_(1.0)
        return _finite_attempt(model, groups)

    with pytest.raises(StageCInvalidImplementationError, match="Parameter.*mutation"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": torch.ones(2, 2)},
            attempt=attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_transport_execution_is_derived_from_actual_action_not_caller_boolean():
    model = _Model()
    groups = build_stage_c_parameter_groups(model)
    a, t, r = groups.adapters[0], groups.transport[0], groups.risk[0]
    with pytest.raises(TypeError, match="transport_executed"):
        StageCAttemptLosses(
            detector_loss=(a.sum() + t.sum()) ** 2,
            feature_loss=t.sum() ** 2,
            risk_loss=r.sum() ** 2,
            transport_executed=False,
        )


@pytest.mark.parametrize("mode", ["noop", "twice"])
def test_success_transition_rejects_noop_or_multi_update_ema(mode):
    model, groups, optimizer, scaler, objects = _retry_fixture(set())
    objects["ema"] = _EMAState(mode=mode)
    with pytest.raises(StageCInvalidImplementationError, match="EMA.*exactly once"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": torch.ones(2, 2)},
            attempt=lambda: _finite_attempt(model, groups),
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_success_transition_rejects_wrong_scheduler_object_binding():
    model, groups, optimizer, scaler, objects = _retry_fixture(set())
    other_optimizer = build_stage_c_optimizer(groups)
    other_scheduler = _build_lr_scheduler(other_optimizer)
    with pytest.raises(ValueError, match="same scheduler object"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": torch.ones(2, 2)},
            attempt=lambda: _finite_attempt(model, groups),
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=other_scheduler,
            rollback_objects=objects,
            seed=3407,
        )


def test_global_success_state_coherence_and_existing_ledger_are_fail_closed():
    model, groups, optimizer, scaler, objects = _retry_fixture(set())
    objects["successful_cursor"].value = 1
    with pytest.raises(StageCInvalidImplementationError, match="global success-state coherence"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": torch.ones(2, 2)},
            attempt=lambda: _finite_attempt(model, groups),
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )

    model, groups, optimizer, scaler, objects = _retry_fixture(set())
    run_stage_c_amp_with_retry_for_test_only(
        materialized_batch={"frames": torch.ones(2, 2), "batch": 0},
        attempt=lambda: _finite_attempt(model, groups),
        model=model,
        groups=groups,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=objects["scheduler"],
        rollback_objects=objects,
        seed=3407,
    )
    objects["shadow_ledger"][0]["successful_update"] = 9
    with pytest.raises(StageCInvalidImplementationError, match="shadow ledger continuity"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": torch.ones(2, 2)},
            attempt=lambda: _finite_attempt(model, groups),
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_nonfinite_gradient_overflow_still_steps_updates_backoff_and_retries():
    model, groups, optimizer, _, objects = _retry_fixture(set())
    scaler = _GradientAwareFakeScaler()
    calls = []
    audit = []

    def attempt():
        detector_output = model()
        multiplier = float("inf") if not calls else 1.0
        calls.append(multiplier)
        return _formal_bound_losses_after_forward(
            model, detector_output, detector_multiplier=multiplier
        )

    result = run_stage_c_amp_with_retry_for_test_only(
        materialized_batch={"frames": torch.ones(2, 2)},
        attempt=attempt,
        model=model,
        groups=groups,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=objects["scheduler"],
        rollback_objects=objects,
        seed=3407,
        retry_audit=audit,
    )
    assert result["attempts"] == 2
    assert scaler.step_calls == scaler.update_calls == 2
    assert scaler.get_scale() == 4.0
    assert [row["overflow"] for row in audit] == [True, False]


def test_retry_fails_closed_if_action_changes_or_overflow_updates_optimizer():
    model, groups, optimizer, scaler, objects = _retry_fixture({0})
    attempt_count = 0

    def changed_action_attempt():
        nonlocal attempt_count
        if attempt_count:
            runtime_actions = model.backbone.chronotransport.forced_actions
            runtime_actions[1, 1, 0] = (
                int(runtime_actions[1, 1, 0].item()) + 1
            ) % 3
        attempt_count += 1
        return _finite_attempt(model, groups)

    with pytest.raises(StageCInvalidImplementationError, match="canonical Stage-C action batch"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": torch.ones(2, 2)},
            attempt=changed_action_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )

    model, groups, optimizer, _, objects = _retry_fixture(set())
    bad_scaler = _BadOverflowScaler({0})

    def bad_step_attempt():
        return _finite_attempt(model, groups)

    with pytest.raises(StageCInvalidImplementationError, match="Parameter.*mutation|changed optimizer"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": torch.ones(2, 2)},
            attempt=bad_step_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=bad_scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_fourth_overflow_is_invalid_and_preserves_only_scaler_and_append_only_audit():
    random.seed(21)
    np.random.seed(22)
    torch.manual_seed(23)
    model, groups, optimizer, scaler, objects = _retry_fixture({0, 1, 2, 3})
    initial_model = copy.deepcopy(model.state_dict())
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    audit = []

    def attempt():
        detector_output = model()
        model.python_forward_counter += 1
        model.latest_actions.append(_action_payload(groups))
        objects["diagnostics"].value += 1
        return _formal_bound_losses_after_forward(model, detector_output)

    with pytest.raises(StageCInvalidImplementationError, match="four overflow attempts"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": torch.ones(2, 2), "augmentation": "fixed"},
            attempt=attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
            retry_audit=audit,
        )

    assert len(audit) == 4
    assert all(row["overflow"] for row in audit)
    assert scaler.step_calls == scaler.update_calls == 4
    assert scaler._scale == 0.5
    assert all(torch.equal(initial_model[name], value) for name, value in model.state_dict().items())
    assert optimizer.state_dict() == initial_optimizer
    assert objects["ema"].stage_c_update_count == 0
    assert objects["scheduler"].last_epoch == 0
    assert objects["sampler"].value == objects["successful_cursor"].value == 0
    assert objects["exposure_cursor"].value == 0
    assert objects["shadow_ledger"] == []


def test_matched_arms_retry_independently_while_preserving_successful_batch_order():
    materialized = {"frames": torch.ones(2, 2), "augmentation": {"seed": 3407}}
    results = {}
    audits = {}
    for arm, overflows in (("ct", {0}), ("dense", set())):
        model, groups, optimizer, scaler, objects = _retry_fixture(overflows)
        audits[arm] = []

        def attempt(model=model, groups=groups):
            return _formal_toy_bound_attempt(model, groups)

        results[arm] = run_stage_c_amp_with_retry_for_test_only(
            materialized_batch=materialized,
            attempt=attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
            retry_audit=audits[arm],
        )

    assert results["ct"]["batch_hash"] == results["dense"]["batch_hash"]
    assert results["ct"]["attempts"] == 2
    assert results["dense"]["attempts"] == 1
    assert len(audits["ct"]) == 2
    assert len(audits["dense"]) == 1


def test_success_rejects_caller_pre_advanced_global_state():
    model, groups, optimizer, scaler, objects = _retry_fixture(set())

    def attempt():
        objects["ema"].stage_c_update_count = 1
        objects["ema"].value = 1
        objects["scheduler"].step()
        objects["sampler"].value = 1
        objects["successful_cursor"].value = 1
        objects["exposure_cursor"].value = 2
        objects["shadow_ledger"].append(
            {
                "successful_update": 0,
                "exposure_start": 0,
                "exposure_stop": 2,
                "batch_hash": "0" * 64,
                "action_hash": "1" * 64,
                "candidate_ordinal": 7,
                "candidate_id": "candidate-07",
            }
        )
        return _finite_attempt(model, groups)

    with pytest.raises(StageCInvalidImplementationError, match="common starting state|premature"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": torch.ones(2, 2)},
            attempt=attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_success_rejects_frozen_parameter_mutation_after_optimizer_step():
    model, groups, optimizer, _, objects = _retry_fixture(set())

    class _MutatingSuccessScaler(_FakeScaler):
        def step(self, bound_optimizer):
            result = super().step(bound_optimizer)
            with torch.no_grad():
                model.heavy.weight.add_(1.0)
            return result

    with pytest.raises(StageCInvalidImplementationError, match="frozen Parameter|success.*Parameter"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": torch.ones(2, 2)},
            attempt=lambda: _finite_attempt(model, groups),
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=_MutatingSuccessScaler(),
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_batch_two_actions_cannot_self_authenticate_wrong_protocol_exposures():
    model, groups, optimizer, scaler, objects = _retry_fixture(set())
    wrong_actions = _action_payload(groups)
    wrong_actions[1, 1, 0] = (int(wrong_actions[1, 1, 0].item()) + 1) % 3
    with pytest.raises(TypeError, match="action_payload"):
        StageCAttemptLosses(
            detector_loss=groups.adapters[0].sum() ** 2,
            feature_loss=groups.transport[0].sum() ** 2,
            risk_loss=groups.risk[0].sum() ** 2,
            action_payload=wrong_actions,
        )

    with pytest.raises(TypeError, match="expected_action_hash"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": torch.ones(2, 2)},
            attempt=lambda: _finite_attempt(model, groups),
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
            expected_action_hash="0" * 64,
        )


@pytest.mark.parametrize("seed", [3407, 3408, 3409])
def test_batch_two_protocol_exposures_and_full_ledger_are_canonical(seed):
    model, groups, optimizer, scaler, objects = _retry_fixture(set())
    result = run_stage_c_amp_with_retry_for_test_only(
        materialized_batch={"frames": torch.ones(2, 2), "seed": seed},
        attempt=lambda: _finite_attempt(model, groups),
        model=model,
        groups=groups,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=objects["scheduler"],
        rollback_objects=objects,
        seed=seed,
    )
    expected = stage_c_batch_exposures(seed, 0)
    assert [row["candidate_ordinal"] for row in result["exposures"]] == [
        row["candidate"] for row in expected
    ]
    assert [row["window_exposure_ordinal"] for row in result["exposures"]] == [0, 1]
    assert all(len(row["actual_action_sha256"]) == 64 for row in result["exposures"])
    assert objects["shadow_ledger"] == [
        {
            "successful_update": 0,
            "seed": seed,
            "exposure_start": 0,
            "exposure_stop": 2,
            "batch_hash": result["batch_hash"],
            "action_batch_sha256": result["action_batch_sha256"],
            "exposures": result["exposures"],
        }
    ]


def test_real_vit_runtime_formal_wrappers_cpu_smoke_covers_dense_and_transport():
    result = _run_formal_vit_stage_c_smoke(
        torch.device("cpu"), _OverflowFakeScaler({0})
    )
    assert result["status"] == "SUCCESS"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a protected CUDA allocation")
def test_real_cuda_gradscaler_overflow_retry_smoke():
    scaler = torch.cuda.amp.GradScaler(init_scale=8.0, growth_interval=1000)
    result = _run_formal_vit_stage_c_smoke(torch.device("cuda"), scaler)
    assert result["attempts"] == 2
    assert scaler.get_scale() == 4.0


def test_stage_c_rejects_attempt_without_canonical_runtime_forward_evidence():
    model, groups, optimizer, scaler, objects = _retry_fixture(set())
    with pytest.raises(StageCInvalidImplementationError, match="runtime|forward|executed"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": torch.ones(2, 2)},
            attempt=lambda: _losses_only(groups),
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_stage_c_rejects_dense_forward_hidden_by_forged_canonical_action_payload():
    model, groups, optimizer, scaler, objects = _retry_fixture(set())
    runtime = model.backbone.chronotransport

    def forged_attempt():
        runtime.force_dense_execution = True
        model()
        runtime.latest_schedule = _ScheduleEvidence(_action_payload(groups))
        runtime.latest_summary = {
            "forced_dense_exact_path": False,
            "schedule_repair_count": 0,
            "runtime_fail_closed_repairs": 0,
            "evidence_valid": True,
        }
        return _losses_only(groups)

    with pytest.raises(StageCInvalidImplementationError, match="actual action|runtime|executed"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": torch.ones(2, 2)},
            attempt=forged_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


@pytest.mark.parametrize("overflow_attempts", [set(), {0}])
def test_stage_c_rejects_depth12_block_order_swap_on_success_and_overflow(
    overflow_attempts,
):
    model = _FormalRetryModel().train()
    groups = build_stage_c_parameter_groups(model)
    optimizer = build_stage_c_optimizer(groups)
    scheduler = _build_lr_scheduler(optimizer)

    class _BlockSwapScaler(_OverflowFakeScaler):
        def step(self, bound_optimizer):
            result = super().step(bound_optimizer)
            blocks = model.backbone.blocks
            blocks[0], blocks[1] = blocks[1], blocks[0]
            return result

    scaler = _BlockSwapScaler(overflow_attempts)
    objects = {
        name: _Stateful()
        for name in (
            "diagnostics",
            "profiler",
            "sampler",
            "successful_cursor",
            "exposure_cursor",
            "shadow_ledger",
        )
    }
    objects["ema"] = _EMAState()
    objects["scheduler"] = scheduler
    objects["shadow_ledger"] = []

    with pytest.raises(StageCInvalidImplementationError, match="topology|module|alias|order"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": torch.ones(2, 2)},
            attempt=lambda: _finite_attempt(model, groups),
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=scheduler,
            rollback_objects=objects,
            seed=3407,
        )


def test_overflow_rollback_preserves_shared_mutable_python_attribute_alias():
    model, groups, optimizer, scaler, objects = _retry_fixture({0})
    shared = {"events": []}
    model.shared_state_a = shared
    model.shared_state_b = shared
    calls = []

    def attempt():
        if not calls:
            model.shared_state_a = {"events": ["bad-a"]}
            model.shared_state_b = {"events": ["bad-b"]}
        calls.append(1)
        return _finite_attempt(model, groups)

    result = run_stage_c_amp_with_retry_for_test_only(
        materialized_batch={"frames": torch.ones(2, 2)},
        attempt=attempt,
        model=model,
        groups=groups,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=objects["scheduler"],
        rollback_objects=objects,
        seed=3407,
    )
    assert result["attempts"] == 2
    assert model.shared_state_a is model.shared_state_b
    assert model.shared_state_a == {"events": []}


def test_formal_toy_rejects_dummy_audited_forward_with_disconnected_direct_losses():
    model, groups, optimizer, scaler, objects = _formal_toy_fixture(set())

    def dummy_attempt():
        model()
        return _direct_losses_only(groups)

    with pytest.raises(StageCInvalidImplementationError, match="loss|forward|provenance|bound"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "formal-toy-dummy-forward"},
            attempt=dummy_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_real_vit_rejects_dummy_audited_forward_with_disconnected_direct_losses():
    device = torch.device("cpu")
    model, groups, optimizer, objects, frames, _ = _formal_vit_stage_c_fixture(device)

    def dummy_attempt():
        model(frames)
        return _direct_losses_only(groups)

    with pytest.raises(StageCInvalidImplementationError, match="loss|forward|provenance|bound"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames_sha": "real-vit-dummy-forward"},
            attempt=dummy_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=_OverflowFakeScaler(set()),
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


@pytest.mark.parametrize("overflow_attempts", [set(), {0}])
@pytest.mark.parametrize(
    "frozen_parameter_path",
    ["blocks.0.attn.weight", "projection.weight", "head.weight"],
)
def test_stage_c_rejects_frozen_parameter_data_byte_mutation_on_success_and_overflow(
    overflow_attempts, frozen_parameter_path
):
    model, groups, optimizer, scaler, objects = _formal_toy_fixture(overflow_attempts)
    frozen = dict(model.named_parameters())[frozen_parameter_path]
    version = frozen._version

    def mutated_attempt():
        frozen.data.reshape(-1)[0].add_(0.125)
        assert frozen._version == version
        return _formal_toy_bound_attempt(model, groups)

    with pytest.raises(StageCInvalidImplementationError, match="frozen Parameter|bytes|hash"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "frozen-data-mutation"},
            attempt=mutated_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


@pytest.mark.parametrize(
    "mode,field,value",
    [
        ("remove", "whole_window_dense_fallback", None),
        ("remove", "forced_dense_exact_path", None),
        ("remove", "evidence_valid", None),
        ("replace", "whole_window_dense_fallback", 0),
        ("replace", "forced_dense_exact_path", "false"),
        ("replace", "evidence_valid", 1),
        ("replace", "cost_is_measured", False),
    ],
)
def test_stage_c_runtime_summary_schema_is_exact_and_fail_closed(mode, field, value):
    model, groups, optimizer, scaler, objects = _formal_toy_fixture(set())
    runtime = model.chronotransport

    def corrupt_summary(module, inputs, output):
        del inputs, output
        if mode == "remove":
            module.latest_summary.pop(field)
        else:
            module.latest_summary[field] = value

    handle = runtime.register_forward_hook(corrupt_summary)
    try:
        with pytest.raises(StageCInvalidImplementationError, match="summary|evidence|field|schema"):
            run_stage_c_amp_with_retry_for_test_only(
                materialized_batch={"frames": f"summary-{mode}-{field}"},
                attempt=lambda: _formal_toy_bound_attempt(model, groups),
                model=model,
                groups=groups,
                optimizer=optimizer,
                scaler=scaler,
                lr_scheduler=objects["scheduler"],
                rollback_objects=objects,
                seed=3407,
            )
    finally:
        handle.remove()


def test_stage_c_rejects_duck_typed_runtime_as_formal_execution_evidence():
    model, groups, optimizer, scaler, objects = _duck_retry_fixture(set())
    with pytest.raises(StageCInvalidImplementationError, match="production|runtime|source|class"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "duck-runtime"},
            attempt=lambda: _finite_attempt(model, groups),
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_stage_c_success_rejects_unapproved_model_python_state_mutation():
    model, groups, optimizer, scaler, objects = _formal_toy_fixture(set())

    def mutated_attempt():
        losses = _formal_toy_bound_attempt(model, groups)
        model.unapproved_success_state = {"mutated": True}
        return losses

    with pytest.raises(StageCInvalidImplementationError, match="Python state|success"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "python-success-mutation"},
            attempt=mutated_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_stage_c_rejects_latest_signals_reference_replacement_before_risk_forward():
    model, groups, optimizer, scaler, objects = _formal_toy_fixture(set())
    runtime = model.chronotransport

    def replaced_attempt():
        detector_output = model()
        original_signals = runtime.latest_signals
        runtime.latest_signals = original_signals.clone()
        assert runtime.latest_signals is not original_signals
        assert torch.equal(runtime.latest_signals, original_signals)
        return _formal_bound_losses_after_forward(model, detector_output)

    with pytest.raises(StageCInvalidImplementationError, match="signals|boundary|identity"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "latest-signals-reference-replacement"},
            attempt=replaced_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_stage_c_rejects_latest_signals_data_mutation_before_risk_forward():
    model, groups, optimizer, scaler, objects = _formal_toy_fixture(set())
    runtime = model.chronotransport

    def mutated_attempt():
        detector_output = model()
        before = runtime.latest_signals.detach().clone()
        version = runtime.latest_signals._version
        runtime.latest_signals.data.add_(7.0)
        assert runtime.latest_signals._version == version
        assert not torch.equal(runtime.latest_signals, before)
        return _formal_bound_losses_after_forward(model, detector_output)

    with pytest.raises(StageCInvalidImplementationError, match="signals|boundary|bytes|value"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "latest-signals-data-mutation"},
            attempt=mutated_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_stage_c_success_rejects_registered_buffer_data_mutation():
    model, groups, optimizer, scaler, objects = _formal_toy_fixture(set())

    def mutated_attempt():
        detector_output = model()
        losses = _formal_bound_losses_after_forward(model, detector_output)
        before = model.toy_input.detach().clone()
        version = model.toy_input._version
        model.toy_input.data.add_(7.0)
        assert model.toy_input._version == version
        assert not torch.equal(model.toy_input, before)
        return losses

    with pytest.raises(StageCInvalidImplementationError, match="buffer|bytes|value"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "registered-buffer-data-mutation"},
            attempt=mutated_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_stage_c_rejects_detector_output_data_mutation_after_forward():
    model, groups, optimizer, scaler, objects = _formal_toy_fixture(set())

    def mutated_attempt():
        detector_output = model()
        before = detector_output.detach().clone()
        version = detector_output._version
        detector_output.data.add_(7.0)
        assert detector_output._version == version
        assert not torch.equal(detector_output, before)
        return _formal_bound_losses_after_forward(model, detector_output)

    with pytest.raises(StageCInvalidImplementationError, match="detector|boundary|bytes|value"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "detector-output-data-mutation"},
            attempt=mutated_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_stage_c_rejects_feature_output_data_mutation_after_forward():
    model, groups, optimizer, scaler, objects = _formal_toy_fixture(set())
    runtime = model.chronotransport

    def mutated_attempt():
        detector_output = model()
        before = runtime.latest_output.detach().clone()
        version = runtime.latest_output._version
        runtime.latest_output.data.add_(7.0)
        assert runtime.latest_output._version == version
        assert not torch.equal(runtime.latest_output, before)
        return _formal_bound_losses_after_forward(model, detector_output)

    with pytest.raises(StageCInvalidImplementationError, match="feature|boundary|bytes|value"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "feature-output-data-mutation"},
            attempt=mutated_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_stage_c_rejects_feature_output_reference_replacement_after_forward():
    model, groups, optimizer, scaler, objects = _formal_toy_fixture(set())
    runtime = model.chronotransport

    def replaced_attempt():
        detector_output = model()
        original_output = runtime.latest_output
        runtime.latest_output = original_output.clone()
        assert runtime.latest_output is not original_output
        assert torch.equal(runtime.latest_output, original_output)
        return _formal_bound_losses_after_forward(model, detector_output)

    with pytest.raises(StageCInvalidImplementationError, match="feature|boundary|identity"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "feature-output-reference-replacement"},
            attempt=replaced_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_stage_c_rejects_latest_signals_equal_value_storage_replacement():
    model, groups, optimizer, scaler, objects = _formal_toy_fixture(set())
    runtime = model.chronotransport

    def replaced_attempt():
        detector_output = model()
        signals = runtime.latest_signals
        storage_cdata = int(signals.untyped_storage()._cdata)
        signals.data = signals.data.clone()
        assert int(signals.untyped_storage()._cdata) != storage_cdata
        return _formal_bound_losses_after_forward(model, detector_output)

    with pytest.raises(StageCInvalidImplementationError, match="signals|boundary|storage"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "latest-signals-storage-replacement"},
            attempt=replaced_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_stage_c_rejects_detector_output_equal_value_storage_replacement():
    model, groups, optimizer, scaler, objects = _formal_toy_fixture(set())

    def replaced_attempt():
        detector_output = model()
        storage_cdata = int(detector_output.untyped_storage()._cdata)
        detector_output.data = detector_output.data.clone()
        assert int(detector_output.untyped_storage()._cdata) != storage_cdata
        return _formal_bound_losses_after_forward(model, detector_output)

    with pytest.raises(StageCInvalidImplementationError, match="detector|boundary|storage"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "detector-output-storage-replacement"},
            attempt=replaced_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_stage_c_rejects_feature_output_equal_value_storage_replacement():
    model, groups, optimizer, scaler, objects = _formal_toy_fixture(set())
    runtime = model.chronotransport

    def replaced_attempt():
        detector_output = model()
        feature_output = runtime.latest_output
        storage_cdata = int(feature_output.untyped_storage()._cdata)
        feature_output.data = feature_output.data.clone()
        assert int(feature_output.untyped_storage()._cdata) != storage_cdata
        return _formal_bound_losses_after_forward(model, detector_output)

    with pytest.raises(StageCInvalidImplementationError, match="feature|boundary|storage"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "feature-output-storage-replacement"},
            attempt=replaced_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_stage_c_success_rejects_registered_buffer_equal_value_storage_replacement():
    model, groups, optimizer, scaler, objects = _formal_toy_fixture(set())

    def replaced_attempt():
        detector_output = model()
        losses = _formal_bound_losses_after_forward(model, detector_output)
        storage_cdata = int(model.toy_input.untyped_storage()._cdata)
        model.toy_input.data = model.toy_input.data.clone()
        assert int(model.toy_input.untyped_storage()._cdata) != storage_cdata
        return losses

    with pytest.raises(StageCInvalidImplementationError, match="buffer|storage"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "registered-buffer-storage-replacement"},
            attempt=replaced_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_stage_c_success_rejects_buffer_mutation_from_success_state_advancer():
    model, groups, optimizer, scaler, objects = _formal_toy_fixture(set())

    class _BufferMutatingEMA(_EMAState):
        def update(self, updated_model):
            super().update(updated_model)
            updated_model.toy_input.data.add_(7.0)

    objects["ema"] = _BufferMutatingEMA()
    with pytest.raises(StageCInvalidImplementationError, match="buffer|bytes|value"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "success-advancer-buffer-mutation"},
            attempt=lambda: _formal_toy_bound_attempt(model, groups),
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_stage_c_success_rejects_training_mode_mutation_from_success_state_advancer():
    model, groups, optimizer, scaler, objects = _formal_toy_fixture(set())

    class _TrainingMutatingEMA(_EMAState):
        def update(self, updated_model):
            super().update(updated_model)
            updated_model.eval()

    objects["ema"] = _TrainingMutatingEMA()
    with pytest.raises(StageCInvalidImplementationError, match="Python state|training|success"):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "success-advancer-training-mutation"},
            attempt=lambda: _formal_toy_bound_attempt(model, groups),
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_stage_c_success_rejects_ordinary_python_tensor_equal_value_storage_rebind():
    model, groups, optimizer, scaler, objects = _formal_toy_fixture(set())
    model.python_tensor_state = torch.arange(6.0)
    model.python_tensor_view = model.python_tensor_state[1:5]
    assert "python_tensor_state" not in model._buffers
    assert "python_tensor_view" not in model._buffers
    original_value = model.python_tensor_state.detach().clone()
    original_version = model.python_tensor_state._version
    original_storage = int(model.python_tensor_state.untyped_storage()._cdata)
    assert int(model.python_tensor_view.untyped_storage()._cdata) == original_storage

    def rebound_attempt():
        losses = _formal_toy_bound_attempt(model, groups)
        model.python_tensor_state.data = model.python_tensor_state.data.clone()
        assert model.python_tensor_state._version == original_version
        assert torch.equal(model.python_tensor_state, original_value)
        assert int(model.python_tensor_state.untyped_storage()._cdata) != original_storage
        assert int(model.python_tensor_view.untyped_storage()._cdata) == original_storage
        return losses

    with pytest.raises(
        StageCInvalidImplementationError,
        match="Python state|storage|alias|success",
    ):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "python-tensor-success-storage-rebind"},
            attempt=rebound_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )


def test_stage_c_overflow_rejects_ordinary_python_tensor_storage_rebind_instead_of_fake_restore():
    model, groups, optimizer, scaler, objects = _formal_toy_fixture({0})
    model.python_tensor_state = torch.arange(6.0)
    model.python_tensor_view = model.python_tensor_state[1:5]
    assert "python_tensor_state" not in model._buffers
    assert "python_tensor_view" not in model._buffers
    original_value = model.python_tensor_state.detach().clone()
    original_version = model.python_tensor_state._version
    original_storage = int(model.python_tensor_state.untyped_storage()._cdata)
    assert int(model.python_tensor_view.untyped_storage()._cdata) == original_storage
    calls = []

    def rebound_attempt():
        losses = _formal_toy_bound_attempt(model, groups)
        if not calls:
            model.python_tensor_state.data = model.python_tensor_state.data.clone()
            assert model.python_tensor_state._version == original_version
            assert torch.equal(model.python_tensor_state, original_value)
            assert int(model.python_tensor_state.untyped_storage()._cdata) != original_storage
            assert int(model.python_tensor_view.untyped_storage()._cdata) == original_storage
        calls.append(1)
        return losses

    with pytest.raises(
        StageCInvalidImplementationError,
        match="Python Tensor|storage|alias|rollback",
    ):
        run_stage_c_amp_with_retry_for_test_only(
            materialized_batch={"frames": "python-tensor-overflow-storage-rebind"},
            attempt=rebound_attempt,
            model=model,
            groups=groups,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=objects["scheduler"],
            rollback_objects=objects,
            seed=3407,
        )
    assert len(calls) == 1


class _FormalActionFormerChronoBackbone(nn.Module):
    """Small real-runtime backbone for callback-free A3/A4 integration tests."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _FormalToyBlock(index in (3, 5, 7, 11))
                for index in range(12)
            ]
        )
        self.chronotransport = ChronoTransportRuntime(
            embed_dims=2,
            depth=12,
            chunks_per_window=48,
            layer_groups=((0, 4), (4, 8), (8, 12)),
            signal_dims=6,
            risk_hidden_dims=4,
            transport_bottleneck_dims=2,
            risk_quantile=0.9,
            hard_cache_validity_age=47,
            transport_age_embedding_cap=8,
            cache_detach=True,
            profile_sync_cuda=False,
            measured_cost=_TEST_MEASURED_COST,
            risk_ready=False,
            require_checkpoint_for_dynamic=False,
        )

    def forward(self, inputs):
        if tuple(inputs.shape) != (2, 2, 48):
            raise ValueError("formal tiny backbone requires [2,2,48]")
        flattened = inputs.transpose(1, 2).reshape(96, 1, 2)
        output = self.chronotransport(flattened, self.blocks, 1, 1)
        return output.reshape(2, 48, 2).transpose(1, 2).contiguous()


def _formal_actionformer_stage_c_fixture(overflow_attempts):
    model = build_detector(
        dict(
            type="ActionFormer",
            projection=dict(
                type="Conv1DTransformerProj",
                in_channels=2,
                out_channels=2,
                arch=(1, 0, 1),
                conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
                norm_cfg=dict(type="LN"),
                attn_cfg=dict(n_head=1, n_mha_win_size=1),
                path_pdrop=0.0,
                use_abs_pe=False,
                max_seq_len=48,
            ),
            rpn_head=dict(
                type="ActionFormerHead",
                num_classes=2,
                in_channels=2,
                feat_channels=2,
                num_convs=1,
                cls_prior_prob=0.01,
                prior_generator=dict(
                    type="PointGenerator",
                    strides=[1, 2],
                    regression_range=[(0, 8), (8, 10000)],
                ),
                loss_normalizer=100,
                loss_normalizer_momentum=0.9,
                center_sample="radius",
                center_sample_radius=1.5,
                label_smoothing=0.0,
                loss=ConfigDict(
                    cls_loss=dict(type="FocalLoss"),
                    reg_loss=dict(type="DIOULoss"),
                ),
            ),
        )
    )
    model.backbone = _FormalActionFormerChronoBackbone()
    model.train()
    groups = build_stage_c_parameter_groups(model)
    optimizer = build_stage_c_optimizer(groups)
    scheduler = _build_lr_scheduler(optimizer)
    objects = {
        "ema": _EMAState(),
        "scheduler": scheduler,
        "diagnostics": StageCStateSurface(),
        "profiler": StageCStateSurface(),
        "sampler": StageCStateSurface(),
        "successful_cursor": StageCStateSurface(),
        "exposure_cursor": StageCStateSurface(),
        "shadow_ledger": [],
    }
    materialized_batch = {
        "inputs": torch.linspace(-1.0, 1.0, 2 * 2 * 48).reshape(2, 2, 48),
        "masks": torch.ones(2, 48, dtype=torch.bool),
        "metas": [
            {"window_id": "stage-c-000"},
            {"window_id": "stage-c-001"},
        ],
        "gt_segments": [
            torch.tensor([[2.0, 11.0], [19.0, 27.0]]),
            torch.tensor([[8.0, 16.0], [31.0, 45.0]]),
        ],
        "gt_labels": [torch.tensor([0, 1]), torch.tensor([1, 0])],
        "window_id": ["stage-c-000", "stage-c-001"],
        "augmentation_sha256": "0" * 64,
        "split": "train",
    }
    return (
        model,
        groups,
        optimizer,
        _OverflowFakeScaler(overflow_attempts),
        objects,
        materialized_batch,
    )


def _formal_actionformer_matched_fixture(overflow_attempts):
    model, _groups, _optimizer, _scaler, _objects, batch = (
        _formal_actionformer_stage_c_fixture(set())
    )
    group = build_matched_dense_parameter_group(model)
    optimizer = build_matched_dense_optimizer(group)
    scheduler = _build_lr_scheduler(optimizer)
    objects = {
        "ema": _EMAState(),
        "scheduler": scheduler,
        "diagnostics": StageCStateSurface(),
        "profiler": StageCStateSurface(),
        "sampler": StageCStateSurface(),
        "successful_cursor": StageCStateSurface(),
        "exposure_cursor": StageCStateSurface(),
        "shadow_ledger": [],
    }
    return (
        model,
        group,
        optimizer,
        _OverflowFakeScaler(overflow_attempts),
        objects,
        batch,
    )


@pytest.mark.parametrize(
    ("overflow_attempts", "expected_attempts"),
    [(set(), 1), ({0}, 2)],
)
def test_callback_free_stage_c_executes_exact_paired_actionformer_transaction(
    overflow_attempts, expected_attempts
):
    torch.manual_seed(3407)
    model, groups, optimizer, scaler, objects, batch = (
        _formal_actionformer_stage_c_fixture(overflow_attempts)
    )
    normalizer = model.rpn_head.loss_normalizer
    normalizer_identity = id(normalizer)
    normalizer_storage = int(normalizer.untyped_storage()._cdata)
    normalizer_before = normalizer.detach().clone()
    retry_audit = []

    result = run_stage_c_amp_with_retry(
        materialized_batch=batch,
        model=model,
        groups=groups,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=objects["scheduler"],
        rollback_objects=objects,
        retry_audit=retry_audit,
        seed=3407,
    )

    assert result["status"] == "SUCCESS"
    assert result["attempts"] == expected_attempts
    assert result["retries"] == expected_attempts - 1
    assert len(retry_audit) == expected_attempts
    assert [row["overflow"] for row in retry_audit] == (
        [False] if expected_attempts == 1 else [True, False]
    )
    assert all(row["model_forward_count"] == 2 for row in retry_audit)
    assert all(row["runtime_forward_count"] == 2 for row in retry_audit)
    assert all(row["risk_forward_count"] == 1 for row in retry_audit)
    assert all(
        row["loss_normalizer_after_dense_temporary"]
        == row["loss_normalizer_after_counterfactual"]
        for row in retry_audit
    )
    assert id(model.rpn_head.loss_normalizer) == normalizer_identity
    assert (
        int(model.rpn_head.loss_normalizer.untyped_storage()._cdata)
        == normalizer_storage
    )
    assert not torch.equal(model.rpn_head.loss_normalizer, normalizer_before)
    assert objects["ema"].stage_c_update_count == 1
    assert objects["scheduler"].last_epoch == 1
    assert objects["successful_cursor"].value == 1
    assert objects["exposure_cursor"].value == 2
    assert len(objects["shadow_ledger"]) == 1


@pytest.mark.parametrize(
    ("overflow_attempts", "expected_attempts"),
    [(set(), 1), ({0}, 2)],
)
def test_matched_dense_executes_one_dense_forward_and_one_common_a_update(
    overflow_attempts, expected_attempts
):
    torch.manual_seed(3407)
    model, group, optimizer, scaler, objects, batch = (
        _formal_actionformer_matched_fixture(overflow_attempts)
    )
    audit = []
    result = run_matched_dense_amp_with_retry(
        materialized_batch=batch,
        model=model,
        group=group,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=objects["scheduler"],
        rollback_objects=objects,
        retry_audit=audit,
        seed=3407,
    )
    assert result["status"] == "SUCCESS"
    assert result["attempts"] == expected_attempts
    assert len(audit) == expected_attempts
    assert [row["overflow"] for row in audit] == (
        [False] if expected_attempts == 1 else [True, False]
    )
    assert all(row["model_forward_count"] == 1 for row in audit)
    assert all(row["runtime_forward_count"] == 1 for row in audit)
    assert all(row["risk_forward_count"] == 0 for row in audit)
    assert objects["successful_cursor"].value == 1
    assert objects["exposure_cursor"].value == 2
    assert objects["ema"].stage_c_update_count == 1
    assert objects["scheduler"].last_epoch == 1
    assert len(objects["shadow_ledger"]) == 1


def test_ct_and_matched_dense_share_batch_exposure_lr_and_normalizer_trace():
    torch.manual_seed(3407)
    ct_model, ct_groups, ct_optimizer, ct_scaler, ct_objects, ct_batch = (
        _formal_actionformer_stage_c_fixture(set())
    )
    torch.manual_seed(3407)
    dense_model, dense_group, dense_optimizer, dense_scaler, dense_objects, dense_batch = (
        _formal_actionformer_matched_fixture(set())
    )
    assert hash_materialized_batch(ct_batch) == hash_materialized_batch(dense_batch)
    torch.testing.assert_close(
        ct_model.rpn_head.loss_normalizer,
        dense_model.rpn_head.loss_normalizer,
        rtol=0.0,
        atol=0.0,
    )
    ct = run_stage_c_amp_with_retry(
        materialized_batch=ct_batch,
        model=ct_model,
        groups=ct_groups,
        optimizer=ct_optimizer,
        scaler=ct_scaler,
        lr_scheduler=ct_objects["scheduler"],
        rollback_objects=ct_objects,
        seed=3407,
    )
    dense = run_matched_dense_amp_with_retry(
        materialized_batch=dense_batch,
        model=dense_model,
        group=dense_group,
        optimizer=dense_optimizer,
        scaler=dense_scaler,
        lr_scheduler=dense_objects["scheduler"],
        rollback_objects=dense_objects,
        seed=3407,
    )
    assert ct["batch_hash"] == dense["batch_hash"]
    assert ct["action_batch_sha256"] == dense["action_batch_sha256"]
    assert ct["exposures"] == dense["exposures"]
    assert ct_objects["shadow_ledger"] == dense_objects["shadow_ledger"]
    assert ct_objects["scheduler"].get_last_lr()[0] == pytest.approx(
        dense_objects["scheduler"].get_last_lr()[0]
    )
    torch.testing.assert_close(
        ct_model.rpn_head.loss_normalizer,
        dense_model.rpn_head.loss_normalizer,
        rtol=0.0,
        atol=0.0,
    )


def test_paired_stage_c_workflow_checkpoints_and_resumes_real_actionformer():
    torch.manual_seed(4407)
    ct_model, _groups, _optimizer, _scaler, _objects, batch = (
        _formal_actionformer_stage_c_fixture(set())
    )
    torch.manual_seed(4407)
    matched_model, _group, _optimizer, _scaler, _objects, matched_batch = (
        _formal_actionformer_matched_fixture(set())
    )
    assert hash_materialized_batch(batch) == hash_materialized_batch(matched_batch)
    state = build_paired_stage_c_state(ct_model, matched_model)
    checkpoints = []
    provenance = {"registration_sha256": "a" * 64, "registration_commit": "b" * 40}

    result = run_paired_stage_c_training(
        state,
        materialize_batch=lambda update: batch,
        fit_window_ids=("stage-c-000", "stage-c-001"),
        seed=3407,
        provenance=provenance,
        formal=False,
        total_successful_updates=1,
        checkpoint_frequency=1,
        checkpoint_sink=lambda cursor, checkpoint: checkpoints.append(
            (cursor, copy.deepcopy(checkpoint))
        ),
    )
    checkpoint = result["checkpoint"]
    assert result["successful_updates"] == 1
    assert result["window_exposures"] == 2
    assert sum(result["candidate_counts"].values()) == 2
    assert checkpoints[0][0] == 1
    validate_paired_stage_c_checkpoint(
        checkpoint,
        expected_seed=3407,
        expected_fit_window_ids=("stage-c-000", "stage-c-001"),
        expected_provenance=provenance,
        formal=False,
        require_complete=True,
        expected_total_successful_updates=1,
    )
    assert checkpoint["ct"]["controls"]["shadow_ledger"] == checkpoint[
        "matched_dense"
    ]["controls"]["shadow_ledger"]
    assert torch.equal(
        checkpoint["ct"]["normalizer"], checkpoint["matched_dense"]["normalizer"]
    )

    torch.manual_seed(4407)
    resumed_ct, _groups, _optimizer, _scaler, _objects, _batch = (
        _formal_actionformer_stage_c_fixture(set())
    )
    torch.manual_seed(4407)
    resumed_matched, _group, _optimizer, _scaler, _objects, _batch = (
        _formal_actionformer_matched_fixture(set())
    )
    resumed = build_paired_stage_c_state(resumed_ct, resumed_matched)
    load_paired_stage_c_checkpoint(
        resumed,
        checkpoint,
        expected_seed=3407,
        expected_fit_window_ids=("stage-c-000", "stage-c-001"),
        expected_provenance=provenance,
        formal=False,
        expected_total_successful_updates=1,
    )
    assert resumed.successful_updates == 1
    assert resumed.trace == state.trace

    tampered = copy.deepcopy(checkpoint)
    tampered["matched_dense"]["controls"]["shadow_ledger"][0]["batch_hash"] = (
        "f" * 64
    )
    with pytest.raises(ValueError, match="shadow ledgers"):
        validate_paired_stage_c_checkpoint(
            tampered,
            expected_seed=3407,
            expected_fit_window_ids=("stage-c-000", "stage-c-001"),
            expected_provenance=provenance,
            formal=False,
            require_complete=True,
            expected_total_successful_updates=1,
        )
