#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from opentad.models.chronotransport.training import (
    checkpoint_readiness_metadata,
    configure_stage_b,
    set_stage_b_module_modes,
    snapshot_model_state,
    validate_stage_b_state_changes,
)


def _factory(spec: str):
    module_name, function_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)


def _strip_ddp_prefix(state: Mapping[str, object]) -> dict[str, object]:
    return {
        str(key)[7:] if str(key).startswith("module.") else str(key): value
        for key, value in state.items()
    }


def _trainable_ema(model, *, decay: float, state=None) -> dict[str, torch.Tensor]:
    decay = float(decay)
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must lie in [0, 1)")
    source = _strip_ddp_prefix(state or {})
    return {
        name: (
            source[name].detach().to(device=parameter.device, dtype=parameter.dtype).clone()
            if name in source
            else parameter.detach().clone()
        )
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


@torch.no_grad()
def _update_ema(model, shadow: dict[str, torch.Tensor], decay: float) -> None:
    parameters = dict(model.named_parameters())
    for name, value in shadow.items():
        value.mul_(float(decay)).add_(parameters[name].detach(), alpha=1.0 - float(decay))


def _checkpoint_state(model, ema: Mapping[str, torch.Tensor]) -> tuple[dict, dict]:
    raw = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    ema_state = {name: value.clone() for name, value in raw.items()}
    for name, value in ema.items():
        ema_state[name] = value.detach().cpu().clone()
        alias = name.replace(
            "chronotransport.risk_predictor",
            "chronotransport.scheduler.predictor",
        )
        if alias in ema_state:
            ema_state[alias] = value.detach().cpu().clone()
    return (
        {f"module.{name}": value for name, value in raw.items()},
        {f"module.{name}": value for name, value in ema_state.items()},
    )


def _write_checkpoint(
    *,
    model,
    optimizer,
    ema,
    output: Path,
    completed: int,
    metadata: Mapping[str, object],
) -> None:
    state_dict, state_dict_ema = _checkpoint_state(model, ema)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(completed),
            "state_dict": state_dict,
            "state_dict_ema": state_dict_ema,
            "optimizer": optimizer.state_dict(),
            "meta": dict(metadata),
        },
        output,
    )


def _loss_payload(loss_step, model, batch, step: int) -> tuple[torch.Tensor, dict]:
    parameters = inspect.signature(loss_step).parameters
    value = loss_step(model, batch, step) if len(parameters) >= 3 else loss_step(model, batch)
    if isinstance(value, torch.Tensor):
        return value, {}
    if not isinstance(value, Mapping) or not isinstance(value.get("loss"), torch.Tensor):
        raise TypeError("Stage-B loss_step must return a Tensor or mapping with Tensor 'loss'")
    loss = value["loss"]
    metrics = {}
    for key, item in value.items():
        if key == "loss":
            continue
        if isinstance(item, torch.Tensor) and item.numel() == 1:
            metrics[str(key)] = float(item.detach().cpu())
        elif isinstance(item, (str, int, float, bool)):
            metrics[str(key)] = item
    return loss, metrics


def run_training(
    *,
    model,
    batch_source: Iterable | Callable[[int], Iterable],
    loss_step,
    optimizer_factory,
    output: Path,
    steps: int,
    resume: Path | None = None,
    metrics_path: Path | None = None,
    checkpoint_interval: int = 0,
    ema_decay: float = 0.999,
    split_hashes: Mapping[str, str] | None = None,
    seed: int = 3407,
    max_grad_norm: float | None = None,
) -> dict:
    steps = int(steps)
    if steps <= 0:
        raise ValueError("Stage-B steps must be positive")
    trainable = configure_stage_b(model)
    if not trainable:
        raise RuntimeError("Stage B found no trainable ChronoTransport parameters")
    optimizer = optimizer_factory(parameter for parameter in model.parameters() if parameter.requires_grad)
    start_step = 0
    epoch_index = 0
    batch_in_epoch = 0
    resume_checkpoint = None
    if resume is not None:
        resume_checkpoint = torch.load(Path(resume), map_location="cpu")
        model.load_state_dict(
            _strip_ddp_prefix(resume_checkpoint["state_dict"]), strict=True
        )
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        resume_meta = dict(resume_checkpoint.get("meta", {}))
        start_step = int(resume_meta.get("steps", resume_checkpoint.get("epoch", 0)))
        epoch_index = int(resume_meta.get("epoch_index", 0))
        batch_in_epoch = int(resume_meta.get("batch_in_epoch", 0))
    if start_step >= steps:
        raise ValueError("resume checkpoint has already reached requested Stage-B steps")
    set_stage_b_module_modes(model)
    ema = _trainable_ema(
        model,
        decay=ema_decay,
        state=None if resume_checkpoint is None else resume_checkpoint.get("state_dict_ema"),
    )
    state_before = snapshot_model_state(model)
    completed = start_step
    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if resume is None:
            metrics_path.write_text("", encoding="utf-8")

    while completed < steps:
        batches = batch_source(epoch_index) if callable(batch_source) else batch_source
        iterator = iter(batches)
        skipped = 0
        while skipped < batch_in_epoch:
            try:
                next(iterator)
            except StopIteration:
                epoch_index += 1
                batch_in_epoch = 0
                break
            skipped += 1
        if skipped < batch_in_epoch:
            continue
        consumed = False
        for batch in iterator:
            consumed = True
            step = completed + 1
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _loss_payload(loss_step, model, batch, step)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite Stage-B loss")
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(max_grad_norm),
                )
            optimizer.step()
            _update_ema(model, ema, ema_decay)
            completed = step
            batch_in_epoch += 1
            row = {
                "step": completed,
                "epoch_index": epoch_index,
                "batch_in_epoch": batch_in_epoch,
                "seed": int(seed),
                "loss": float(loss.detach().cpu()),
                **metrics,
            }
            if metrics_path is not None:
                with metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            if checkpoint_interval > 0 and completed % int(checkpoint_interval) == 0:
                periodic = output.with_name(f"{output.stem}.step{completed}{output.suffix}")
                _write_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    ema=ema,
                    output=periodic,
                    completed=completed,
                    metadata={
                        **checkpoint_readiness_metadata(
                            stage="B",
                            calibration_ready=False,
                            measured_cost_ready=False,
                            split_hashes=split_hashes or {},
                        ),
                        "steps": completed,
                        "epoch_index": epoch_index,
                        "batch_in_epoch": batch_in_epoch,
                        "seed": int(seed),
                        "ema_decay": float(ema_decay),
                        "ema_semantics": "trainable_parameter_ema",
                    },
                )
            if completed >= steps:
                break
        if completed >= steps:
            break
        if not consumed and not callable(batch_source):
            raise RuntimeError("Stage B exhausted a non-restartable batch source")
        epoch_index += 1
        batch_in_epoch = 0

    state_audit = validate_stage_b_state_changes(state_before, model)
    metadata = checkpoint_readiness_metadata(
        stage="B",
        calibration_ready=False,
        measured_cost_ready=False,
        split_hashes=split_hashes or {},
    )
    metadata.update(
        steps=completed,
        start_step=start_step,
        epoch_index=epoch_index,
        batch_in_epoch=batch_in_epoch,
        seed=int(seed),
        trainable_parameters=list(trainable),
        ema_decay=float(ema_decay),
        ema_semantics="trainable_parameter_ema",
        stage_b_state_audit=state_audit,
        resumed_from=None if resume is None else str(resume),
    )
    _write_checkpoint(
        model=model,
        optimizer=optimizer,
        ema=ema,
        output=Path(output),
        completed=completed,
        metadata=metadata,
    )
    return {"steps": completed, "trainable": len(trainable), "checkpoint": str(output)}


def run(factory_spec: str, output: Path, steps: int) -> dict:
    model, batches, loss_step, optimizer_factory = _factory(factory_spec)()
    return run_training(
        model=model,
        batch_source=batches,
        loss_step=loss_step,
        optimizer_factory=optimizer_factory,
        output=output,
        steps=steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(args.factory, args.output, args.steps), indent=2))


if __name__ == "__main__":
    main()
