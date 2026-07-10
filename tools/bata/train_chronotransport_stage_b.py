#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
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


def run(factory_spec: str, output: Path, steps: int) -> dict:
    if int(steps) <= 0:
        raise ValueError("Stage-B steps must be positive")
    model, batches, loss_step, optimizer_factory = _factory(factory_spec)()
    trainable = configure_stage_b(model)
    if not trainable:
        raise RuntimeError("Stage B found no trainable ChronoTransport parameters")
    optimizer = optimizer_factory(parameter for parameter in model.parameters() if parameter.requires_grad)
    set_stage_b_module_modes(model)
    state_before = snapshot_model_state(model)
    completed = 0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_step(model, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite Stage-B loss")
        loss.backward()
        optimizer.step()
        completed += 1
        if completed >= steps:
            break
    if completed != int(steps):
        raise RuntimeError(f"Stage B exhausted data after {completed} of {steps} requested steps")
    state_audit = validate_stage_b_state_changes(state_before, model)
    metadata = checkpoint_readiness_metadata(
        stage="B",
        calibration_ready=False,
        measured_cost_ready=False,
        split_hashes={},
    )
    metadata.update(
        steps=completed,
        trainable_parameters=list(trainable),
        ema_semantics="export_alias_without_moving_average",
        stage_b_state_audit=state_audit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {f"module.{key}": value for key, value in model.state_dict().items()}
    torch.save(
        {
            "epoch": completed,
            "state_dict": state_dict,
            "state_dict_ema": state_dict,
            "optimizer": optimizer.state_dict(),
            "meta": metadata,
        },
        output,
    )
    return {"steps": completed, "trainable": len(trainable), "checkpoint": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(args.factory, args.output, args.steps), indent=2))


if __name__ == "__main__":
    main()
