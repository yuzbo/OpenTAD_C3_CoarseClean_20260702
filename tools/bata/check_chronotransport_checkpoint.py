#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from mmengine.config import Config

import opentad.datasets  # noqa: F401 - registers configured data transforms
from opentad.models import build_detector


def _strip_ddp_prefix(state: Mapping[str, object]) -> dict[str, object]:
    keys = tuple(str(key) for key in state)
    prefixed = tuple(key.startswith("module.") for key in keys)
    if any(prefixed) and not all(prefixed):
        raise ValueError("checkpoint state dict mixes DDP-prefixed and unprefixed keys")
    if keys and all(prefixed):
        return {str(key)[7:]: value for key, value in state.items()}
    return {str(key): value for key, value in state.items()}


def classify_incompatible_keys(
    *,
    missing: Iterable[str],
    unexpected: Iterable[str],
    allow_chronotransport_missing: bool,
) -> dict[str, object]:
    missing_keys = sorted(str(key) for key in missing)
    unexpected_keys = sorted(str(key) for key in unexpected)
    allowed_missing = []
    if allow_chronotransport_missing:
        allowed_missing = [
            key for key in missing_keys if "chronotransport" in key.split(".")
        ]
    forbidden_missing = [key for key in missing_keys if key not in allowed_missing]
    return {
        "status": "PASS" if not forbidden_missing and not unexpected_keys else "FAIL",
        "allowed_chronotransport_missing": len(allowed_missing),
        "forbidden_missing": forbidden_missing,
        "unexpected": unexpected_keys,
    }


def select_checkpoint_state(
    checkpoint: Mapping[str, object], *, use_ema: bool
) -> tuple[Mapping[str, object], str]:
    state_key = "state_dict_ema" if use_ema else "state_dict"
    if state_key not in checkpoint:
        raise KeyError(
            f"config solver.ema={use_ema} requires checkpoint key {state_key!r}"
        )
    state = checkpoint[state_key]
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint {state_key!r} must be a mapping")
    return state, state_key


def check(config_path: str, checkpoint_path: str) -> dict[str, object]:
    cfg = Config.fromfile(config_path)
    cfg.model.backbone.custom.pretrain = None
    model = build_detector(cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    use_ema = bool(getattr(cfg.solver, "ema", False))
    state, state_key = select_checkpoint_state(checkpoint, use_ema=use_ema)
    state = _strip_ddp_prefix(dict(state))
    incompatible = model.load_state_dict(state, strict=False)
    runtime_cfg = cfg.model.backbone.backbone.chronotransport
    allow_legacy_checkpoint = bool(runtime_cfg.allow_legacy_checkpoint)
    result = classify_incompatible_keys(
        missing=incompatible.missing_keys,
        unexpected=incompatible.unexpected_keys,
        allow_chronotransport_missing=allow_legacy_checkpoint,
    )
    result.update(
        checkpoint=checkpoint_path,
        epoch=checkpoint.get("epoch"),
        state_key=state_key,
        allow_legacy_checkpoint=allow_legacy_checkpoint,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    result = check(args.config, args.checkpoint)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
