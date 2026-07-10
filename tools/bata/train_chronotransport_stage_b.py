#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch

from opentad.models.chronotransport.training import (
    checkpoint_readiness_metadata,
    configure_stage_b,
)


def _factory(spec: str):
    module_name, function_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)


def run(factory_spec: str, output: Path, steps: int) -> dict:
    model, batches, loss_step, optimizer_factory = _factory(factory_spec)()
    trainable = configure_stage_b(model)
    if not trainable:
        raise RuntimeError("Stage B found no trainable ChronoTransport parameters")
    optimizer = optimizer_factory(parameter for parameter in model.parameters() if parameter.requires_grad)
    model.train()
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
    metadata = checkpoint_readiness_metadata(
        stage="B",
        calibration_ready=False,
        measured_cost_ready=False,
        split_hashes={},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": metadata}, output)
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
