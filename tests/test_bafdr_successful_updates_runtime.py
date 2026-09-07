import ast
import logging
import os
import random
from pathlib import Path

import numpy as np
import pytest

if os.name == "nt":
    pytest.skip("Project Windows Torch DLL is unavailable; run on N16R4", allow_module_level=True)

try:
    import torch
except (ImportError, OSError) as exc:
    pytest.skip(str(exc), allow_module_level=True)


ROOT = Path(__file__).resolve().parents[1]


def load_epoch():
    # Execute the production function without importing unrelated model registries.
    source = ROOT / "tools/bata/bafdr_k16_fullmatrix_train.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_epoch"]
    engine = ast.parse((ROOT / "opentad/cores/train_engine.py").read_text(encoding="utf-8"))
    nodes += [node for node in engine.body if isinstance(node, ast.FunctionDef)
              and node.name in ("_capture_model_buffers", "_restore_model_buffers")]
    namespace = dict(torch=torch, np=np, random=random, EXPECTED_UPDATES_PER_EPOCH=100,
                     autocast=torch.cuda.amp.autocast, Mapping=dict,
                     prepare_forward_kwargs=lambda data, device: data)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"), namespace)
    return namespace


class Toy(torch.nn.Module):
    def __init__(self, device, bad_loss=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25, device=device))
        self.register_buffer("loss_normalizer", torch.zeros((), device=device))
        self.draws = []
        self.bad_loss = bad_loss

    def forward(self, **kwargs):
        self.loss_normalizer.add_(1)
        draw = torch.rand((), device=self.weight.device)
        self.draws.append(float(draw))
        value = self.weight.square() + draw * self.weight
        return {"cost": value * (float("nan") if self.bad_loss else 1)}


class SkipScaler:
    def __init__(self, skips):
        self.skips = skips
        self.scale_value = 8.0
        self.skipped = False

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def get_scale(self):
        return self.scale_value

    def step(self, optimizer):
        self.skipped = self.skips > 0
        if self.skipped:
            self.skips -= 1
        else:
            optimizer.step()

    def update(self):
        if self.skipped:
            self.scale_value /= 2


class Ema:
    def __init__(self):
        self.calls = 0

    def update(self, model):
        self.calls += 1


def setup(device="cpu", bad_loss=False):
    model = Toy(device, bad_loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
    data = dict(inputs={}, masks=None, metas=None, gt_segments=[], gt_labels=[])
    return model, optimizer, scheduler, [data, data]


def test_replay_restores_student_teacher_rng_and_counts():
    ns = load_epoch()
    model, optimizer, scheduler, loader = setup()
    teacher = Toy("cpu")
    def kd(**kwargs):
        with torch.no_grad():
            teacher()
        return {"kd": model.weight.square() * 0.2}
    ns["compute_bafdr_distillation_losses"] = kd
    audit, ema = {}, Ema()
    updates = ns["train_epoch"](model, loader, optimizer, scheduler, SkipScaler(1), ema,
        torch.device("cpu"), 0, logging.getLogger(), teacher=teacher, use_amp=False,
        update_audit=audit, updates_per_epoch=2)
    assert updates == ema.calls == scheduler.last_epoch == 2
    assert model.loss_normalizer.item() == teacher.loss_normalizer.item() == 2
    assert model.draws[0] == model.draws[1]
    assert teacher.draws[0] == teacher.draws[1]
    assert audit["amp_skipped_attempts"] == audit["replay_attempts"] == 1
    assert audit["optimizer_attempts"] == 3
    assert int(optimizer.state[model.weight]["step"]) == 2


@pytest.mark.parametrize("bad_loss", [False, True])
def test_failure_never_advances_optimizer_scheduler_or_ema(bad_loss):
    ns = load_epoch()
    model, optimizer, scheduler, loader = setup(bad_loss=bad_loss)
    audit, ema = {}, Ema()
    with pytest.raises(FloatingPointError, match="non-finite loss" if bad_loss else "AMP replay exhausted"):
        ns["train_epoch"](model, loader, optimizer, scheduler, SkipScaler(20), ema,
            torch.device("cpu"), 0, logging.getLogger(), use_amp=False,
            update_audit=audit, updates_per_epoch=2, max_amp_retries_per_batch=2)
    assert not optimizer.state
    assert scheduler.last_epoch == ema.calls == audit["successful_optimizer_updates"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA admission required")
def test_real_cuda_gradscaler_overflow_replays_same_batch():
    ns = load_epoch()
    model, optimizer, scheduler, loader = setup("cuda")
    seen = []
    def inject_once(grad):
        seen.append(True)
        return grad * float("inf") if len(seen) == 1 else grad
    handle = model.weight.register_hook(inject_once)
    audit, ema = {}, Ema()
    scaler = torch.cuda.amp.GradScaler(init_scale=8)
    try:
        ns["train_epoch"](model, loader, optimizer, scheduler, scaler, ema,
            torch.device("cuda"), 0, logging.getLogger(), use_amp=True,
            update_audit=audit, updates_per_epoch=2)
    finally:
        handle.remove()
    assert int(optimizer.state[model.weight]["step"]) == 2
    assert scheduler.last_epoch == ema.calls == model.loss_normalizer.item() == 2
    assert model.draws[0] == model.draws[1]
    assert audit["amp_skipped_attempts"] == audit["replay_attempts"] == 1
