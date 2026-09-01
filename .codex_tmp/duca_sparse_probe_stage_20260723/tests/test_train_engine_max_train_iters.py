import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ENGINE_PATH = ROOT / "opentad" / "cores" / "train_engine.py"


class _Autocast:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _NoGrad(_Autocast):
    pass


class _AverageMeter:
    def __init__(self):
        self.values = []
        self.avg = 0.0

    def update(self, value):
        self.values.append(float(value))
        self.avg = sum(self.values) / len(self.values)


class _Loss:
    def __init__(self, value, owner):
        self.value = float(value)
        self.owner = owner
        self.data = self

    def backward(self):
        self.owner.backward_calls += 1

    def clone(self):
        return self

    def div_(self, value):
        self.value /= float(value)
        return self

    def item(self):
        return self.value


class _Buffer:
    def __init__(self, value=0):
        self.value = int(value)

    def detach(self):
        return self

    def clone(self):
        return _Buffer(self.value)

    def copy_(self, other):
        self.value = int(other.value)
        return self


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        if args:
            message = message % args
        self.messages.append(str(message))


class _ToyLoader:
    def __init__(self, length):
        self.length = int(length)

    def __len__(self):
        return self.length

    def __iter__(self):
        for idx in range(self.length):
            yield {"x": idx + 1}


class _ToyModel:
    def __init__(self, *, with_after_step_hook=False, mutate_replay_state=False):
        self.module = types.SimpleNamespace()
        if with_after_step_hook:
            self.module.after_optimizer_step = self.after_optimizer_step
        self.train_calls = 0
        self.forward_calls = 0
        self.backward_calls = 0
        self.after_optimizer_step_calls = 0
        self.after_optimizer_seen_steps = []
        self.optimizer = None
        self.mutate_replay_state = bool(mutate_replay_state)
        self.loss_normalizer = _Buffer(0)
        self.custom_forward_state = 0

    def train(self):
        self.train_calls += 1

    def __call__(self, x, return_loss=False):
        assert return_loss is True
        self.forward_calls += 1
        if self.mutate_replay_state:
            self.loss_normalizer.value += 1
            self.custom_forward_state += 1
        cost = _Loss(x, self)
        return {"cost": cost, "aux_loss": _Loss(x * 0.5, self)}

    def after_optimizer_step(self):
        self.after_optimizer_step_calls += 1
        if self.optimizer is not None:
            self.after_optimizer_seen_steps.append(self.optimizer.steps)

    def named_buffers(self):
        return [("loss_normalizer", self.loss_normalizer)]

    def named_modules(self):
        return [("", self)]

    def capture_amp_replay_state(self):
        return {"custom_forward_state": self.custom_forward_state}

    def restore_amp_replay_state(self, snapshot):
        self.custom_forward_state = int(snapshot["custom_forward_state"])


class _ToyOptimizer:
    def __init__(self):
        self.zero_grad_calls = 0
        self.steps = 0

    def zero_grad(self):
        self.zero_grad_calls += 1

    def step(self):
        self.steps += 1


class _ToyScheduler:
    def __init__(self):
        self.steps = 0

    def get_last_lr(self):
        return [1.0e-3]

    def step(self):
        self.steps += 1


class _ToyScaler:
    def __init__(self, skipped_attempts=1):
        self.remaining_skips = int(skipped_attempts)
        self.current_scale = 65536.0
        self.last_step_skipped = False

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return optimizer

    def get_scale(self):
        return self.current_scale

    def step(self, optimizer):
        self.last_step_skipped = self.remaining_skips > 0
        if self.last_step_skipped:
            self.remaining_skips -= 1
        else:
            optimizer.step()

    def update(self):
        if self.last_step_skipped:
            self.current_scale /= 2.0


def _load_train_engine_with_fake_runtime(monkeypatch):
    fake_torch = types.SimpleNamespace(
        float16="float16",
        get_rng_state=lambda: "cpu-rng",
        set_rng_state=lambda _state: None,
        no_grad=_NoGrad,
        isfinite=lambda _value: types.SimpleNamespace(
            all=lambda: types.SimpleNamespace(item=lambda: True)
        ),
        cuda=types.SimpleNamespace(
            amp=types.SimpleNamespace(autocast=_Autocast),
            max_memory_allocated=lambda: 0,
            get_rng_state_all=lambda: ["cuda-rng"],
            set_rng_state_all=lambda _states: None,
        ),
        nn=types.SimpleNamespace(
            utils=types.SimpleNamespace(clip_grad_norm_=lambda *args, **kwargs: None),
        ),
    )
    fake_tqdm = types.SimpleNamespace(tqdm=lambda iterable, disable=False: iterable)
    fake_misc = types.SimpleNamespace(
        AverageMeter=_AverageMeter,
        reduce_loss=lambda losses: losses,
    )
    fake_utils = types.ModuleType("opentad.utils")
    fake_utils.misc = fake_misc
    fake_opentad = types.ModuleType("opentad")
    fake_opentad.utils = fake_utils

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm)
    monkeypatch.setitem(sys.modules, "opentad", fake_opentad)
    monkeypatch.setitem(sys.modules, "opentad.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "opentad.utils.misc", fake_misc)

    module_name = "train_engine_fake_runtime_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, TRAIN_ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_one_epoch_stops_after_max_train_iters(monkeypatch):
    train_engine = _load_train_engine_with_fake_runtime(monkeypatch)
    model = _ToyModel()
    optimizer = _ToyOptimizer()
    scheduler = _ToyScheduler()
    logger = _Logger()

    train_engine.train_one_epoch(
        _ToyLoader(length=5),
        model,
        optimizer,
        scheduler,
        curr_epoch=0,
        logger=logger,
        logging_interval=1,
        max_train_iters=2,
    )

    assert model.train_calls == 1
    assert model.forward_calls == 2
    assert model.backward_calls == 2
    assert optimizer.zero_grad_calls == 2
    assert optimizer.steps == 2
    assert scheduler.steps == 2
    assert any("max_train_iters=2 reached" in message for message in logger.messages)


def test_train_one_epoch_calls_after_optimizer_step_hook_after_each_optimizer_step(monkeypatch):
    train_engine = _load_train_engine_with_fake_runtime(monkeypatch)
    model = _ToyModel(with_after_step_hook=True)
    optimizer = _ToyOptimizer()
    model.optimizer = optimizer
    scheduler = _ToyScheduler()

    train_engine.train_one_epoch(
        _ToyLoader(length=3),
        model,
        optimizer,
        scheduler,
        curr_epoch=0,
        logger=_Logger(),
        logging_interval=1,
    )

    assert optimizer.steps == 3
    assert model.after_optimizer_step_calls == 3
    assert model.after_optimizer_seen_steps == [1, 2, 3]


def test_train_one_epoch_calls_nested_frame_selector_hook_once(monkeypatch):
    train_engine = _load_train_engine_with_fake_runtime(monkeypatch)
    model = _ToyModel()
    selector = types.SimpleNamespace(calls=0)

    def after_optimizer_step():
        selector.calls += 1
        return {"updated": True}

    selector.after_optimizer_step = after_optimizer_step
    model.module.frame_selector = selector
    audit = {}

    train_engine.train_one_epoch(
        _ToyLoader(length=2),
        model,
        _ToyOptimizer(),
        _ToyScheduler(),
        curr_epoch=0,
        logger=_Logger(),
        logging_interval=1,
        update_audit=audit,
    )

    assert selector.calls == 2
    assert audit["duca_schedule_updates"] == 2


def test_train_one_epoch_replays_amp_skip_without_advancing_state(monkeypatch):
    train_engine = _load_train_engine_with_fake_runtime(monkeypatch)
    model = _ToyModel(with_after_step_hook=True, mutate_replay_state=True)
    optimizer = _ToyOptimizer()
    model.optimizer = optimizer
    scheduler = _ToyScheduler()
    audit = {}

    train_engine.train_one_epoch(
        _ToyLoader(length=2),
        model,
        optimizer,
        scheduler,
        curr_epoch=0,
        logger=_Logger(),
        logging_interval=1,
        scaler=_ToyScaler(skipped_attempts=1),
        max_amp_retries_per_batch=4,
        fail_on_amp_replay_exhaustion=True,
        require_finite_loss=True,
        update_audit=audit,
    )

    assert model.forward_calls == 3
    assert optimizer.steps == 2
    assert scheduler.steps == 2
    assert model.after_optimizer_step_calls == 2
    assert model.loss_normalizer.value == 2
    assert model.custom_forward_state == 2
    assert audit == {
        "attempted_batches": 2,
        "optimizer_attempts": 3,
        "successful_optimizer_updates": 2,
        "amp_skipped_attempts": 1,
        "replayed_batches": 1,
        "replay_exhaustions": 0,
        "scheduler_updates": 2,
        "ema_updates": 0,
        "duca_schedule_updates": 0,
        "forced_amp_overflow_attempts": 0,
        "max_amp_retries_observed": 1,
    }


def test_train_one_epoch_fails_closed_when_amp_replay_is_exhausted(monkeypatch):
    train_engine = _load_train_engine_with_fake_runtime(monkeypatch)
    model = _ToyModel(mutate_replay_state=True)
    optimizer = _ToyOptimizer()
    scheduler = _ToyScheduler()
    audit = {}

    with pytest.raises(FloatingPointError, match="could not produce"):
        train_engine.train_one_epoch(
            _ToyLoader(length=1),
            model,
            optimizer,
            scheduler,
            curr_epoch=0,
            logger=_Logger(),
            scaler=_ToyScaler(skipped_attempts=3),
            max_amp_retries_per_batch=1,
            fail_on_amp_replay_exhaustion=True,
            update_audit=audit,
        )

    assert optimizer.steps == 0
    assert scheduler.steps == 0
    assert model.loss_normalizer.value == 0
    assert model.custom_forward_state == 0
    assert audit["replay_exhaustions"] == 1
    assert audit["successful_optimizer_updates"] == 0


def test_train_one_epoch_rejects_amp_replay_without_scaler(monkeypatch):
    train_engine = _load_train_engine_with_fake_runtime(monkeypatch)
    with pytest.raises(ValueError, match="requires a GradScaler"):
        train_engine.train_one_epoch(
            _ToyLoader(length=1),
            _ToyModel(),
            _ToyOptimizer(),
            _ToyScheduler(),
            curr_epoch=0,
            logger=_Logger(),
            max_amp_retries_per_batch=1,
        )


@pytest.mark.parametrize("max_train_iters", [0, -1])
def test_train_one_epoch_rejects_non_positive_max_train_iters(monkeypatch, max_train_iters):
    train_engine = _load_train_engine_with_fake_runtime(monkeypatch)
    model = _ToyModel()
    optimizer = _ToyOptimizer()
    scheduler = _ToyScheduler()

    with pytest.raises(ValueError, match="max_train_iters must be positive"):
        train_engine.train_one_epoch(
            _ToyLoader(length=5),
            model,
            optimizer,
            scheduler,
            curr_epoch=0,
            logger=_Logger(),
            max_train_iters=max_train_iters,
        )

    assert model.train_calls == 0
    assert model.forward_calls == 0
    assert scheduler.steps == 0
