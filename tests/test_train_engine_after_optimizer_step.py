from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("local Windows torch/c10.dll import is unstable; Linux remote runs this suite", allow_module_level=True)

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - local environment guard.
    pytest.skip(f"torch is unavailable in this environment: {exc}", allow_module_level=True)

from opentad.cores import train_engine


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args) -> None:
        if args:
            message = message % args
        self.messages.append(str(message))


class _InnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.after_step_calls = 0

    def forward(self, x: torch.Tensor, return_loss: bool = True):
        return {"cost": (self.weight * x).sum()}

    def after_optimizer_step(self):
        self.after_step_calls += 1
        return {"updated": True}


class _DDPStyleWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = _InnerModel()

    def forward(self, **kwargs):
        return self.module(**kwargs)


class _Selector:
    def __init__(self) -> None:
        self.last_forward_summary = {
            "budget": 384,
            "dynamic_budget": True,
            "budget_policy": "prefix_marginal_utility_stop",
            "requested_budget": [256, 320],
            "effective_budget": [256, 320],
            "loss_weight_schedule": {
                "step": 12,
                "phase": "transition_detector_utility",
                "progress": 0.25,
                "detector_gradient_weight": 0.5,
                "weights": {
                    "actionness": 0.75,
                    "hole": 0.0125,
                    "lagrangian_budget": 0.25,
                },
            },
        }


class _FakeScaler:
    def __init__(self, *, skip_step: bool) -> None:
        self.skip_step = bool(skip_step)
        self._scale = 128.0

    def get_scale(self) -> float:
        return float(self._scale)

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer) -> None:
        if not self.skip_step:
            optimizer.step()

    def update(self) -> None:
        if self.skip_step:
            self._scale *= 0.5


class _Scheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def get_last_lr(self) -> list[float]:
        return [0.1]

    def step(self) -> None:
        self.step_calls += 1


def _run_one_train_iter(monkeypatch: pytest.MonkeyPatch, *, skip_step: bool) -> tuple[_DDPStyleWrapper, _Scheduler]:
    model = _DDPStyleWrapper()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _Scheduler()
    loader = [{"x": torch.tensor(2.0)}]
    monkeypatch.setattr(train_engine, "reduce_loss", lambda losses: losses)

    train_engine.train_one_epoch(
        loader,
        model,
        optimizer,
        scheduler,
        curr_epoch=0,
        logger=_Logger(),
        scaler=_FakeScaler(skip_step=skip_step),
        max_train_iters=1,
        logging_interval=999,
    )
    return model, scheduler


def test_after_optimizer_step_hook_does_not_run_when_amp_scaler_skips_step(monkeypatch: pytest.MonkeyPatch) -> None:
    model, scheduler = _run_one_train_iter(monkeypatch, skip_step=True)

    assert model.module.after_step_calls == 0
    assert model.module.weight.detach().item() == pytest.approx(1.0)
    assert scheduler.step_calls == 0


def test_after_optimizer_step_hook_runs_after_real_amp_optimizer_step(monkeypatch: pytest.MonkeyPatch) -> None:
    model, scheduler = _run_one_train_iter(monkeypatch, skip_step=False)

    assert model.module.after_step_calls == 1
    assert model.module.weight.detach().item() < 1.0
    assert scheduler.step_calls == 1


def test_frame_selector_diagnostics_format_schedule_and_budget_state() -> None:
    model = _DDPStyleWrapper()
    model.module.frame_selector = _Selector()

    message = train_engine._format_frame_selector_diagnostics(model)

    assert "duca_schedule_step=12" in message
    assert "duca_phase=transition_detector_utility" in message
    assert "duca_schedule_progress=0.2500" in message
    assert "duca_detector_grad_w=0.5000" in message
    assert "duca_actionness_w=0.7500" in message
    assert "duca_lagrangian_budget_w=0.2500" in message
    assert "duca_budget=384" in message
    assert "duca_dynamic_budget=True" in message
    assert "duca_requested_budget_mean=288.00" in message


def test_gradient_failure_summary_names_non_finite_optimizer_parameter() -> None:
    model = _DDPStyleWrapper()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.module.weight.grad = torch.tensor(float("nan"))

    summary = train_engine._gradient_failure_summary(model, optimizer)

    assert "module.weight" in summary
    assert "nan=1" in summary
