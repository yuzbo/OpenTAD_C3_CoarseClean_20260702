from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


RESETTABLE_SELECTOR_STATE_KEYS = frozenset({"_loss_weight_schedule_step"})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def selector_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {
        (str(key)[len("module.") :] if str(key).startswith("module.") else str(key)): value
        for key, value in state_dict.items()
    }
    selected = {
        key[len("frame_selector.") :]: value
        for key, value in normalized.items()
        if key.startswith("frame_selector.")
    }
    if not selected:
        raise ValueError("frontend checkpoint has no frame_selector.* state")
    return selected


def _as_mapping(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    raise TypeError("selector_initialization must be a mapping")


def initialize_frame_selector_from_checkpoint(
    model,
    config: Mapping[str, Any] | Any,
    *,
    logger=None,
) -> dict[str, Any] | None:
    cfg = _as_mapping(config)
    if not cfg or not bool(cfg.get("enabled", True)):
        return None
    allowed = {
        "enabled",
        "checkpoint_path",
        "checkpoint_sha256",
        "state_key",
        "expected_checkpoint_epoch",
        "reset_state_keys",
    }
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise ValueError(f"unknown selector_initialization keys: {unknown}")

    checkpoint_path = Path(
        os.path.expandvars(os.path.expanduser(str(cfg.get("checkpoint_path", ""))))
    ).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"selector initialization checkpoint is missing: {checkpoint_path}"
        )
    expected_sha256 = str(cfg.get("checkpoint_sha256", "")).lower()
    if len(expected_sha256) != 64:
        raise ValueError("selector_initialization.checkpoint_sha256 is required")
    observed_sha256 = sha256_file(checkpoint_path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError("selector initialization checkpoint SHA256 mismatch")

    state_key = str(cfg.get("state_key", "state_dict_ema"))
    if state_key not in {"state_dict", "state_dict_ema"}:
        raise ValueError("selector initialization state_key must be state_dict or state_dict_ema")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("selector initialization checkpoint must be a mapping")
    state = checkpoint.get(state_key)
    if not isinstance(state, Mapping):
        raise ValueError(f"selector initialization checkpoint lacks {state_key}")

    expected_epoch = cfg.get("expected_checkpoint_epoch")
    observed_epoch = checkpoint.get("epoch")
    if expected_epoch is not None and int(observed_epoch) != int(expected_epoch):
        raise RuntimeError(
            "selector initialization checkpoint epoch mismatch: "
            f"expected {int(expected_epoch)}, got {observed_epoch!r}"
        )

    selector = getattr(model, "frame_selector", None)
    if selector is None:
        raise ValueError("selector initialization requires model.frame_selector")
    source_state = selector_state_dict(state)
    target_state = selector.state_dict()
    reset_keys = tuple(
        str(key)
        for key in cfg.get(
            "reset_state_keys", sorted(RESETTABLE_SELECTOR_STATE_KEYS)
        )
    )
    unsupported_reset = sorted(set(reset_keys) - RESETTABLE_SELECTOR_STATE_KEYS)
    if unsupported_reset:
        raise ValueError(
            f"selector initialization cannot reset unapproved state: {unsupported_reset}"
        )
    for key in reset_keys:
        if key not in target_state or key not in source_state:
            raise RuntimeError(f"selector initialization reset state is missing: {key}")
        source_state[key] = target_state[key].detach().clone()

    missing = sorted(set(target_state) - set(source_state))
    unexpected = sorted(set(source_state) - set(target_state))
    if missing or unexpected:
        raise RuntimeError(
            "selector initialization state mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    incompatible = selector.load_state_dict(source_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "selector initialization strict load failed: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    receipt = {
        "schema": "duca_frontend_initialization_v1",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": observed_sha256,
        "checkpoint_epoch": None if observed_epoch is None else int(observed_epoch),
        "checkpoint_state_key": state_key,
        "loaded_selector_state_count": len(source_state),
        "reset_state_keys": list(reset_keys),
        "detector_state_loaded": False,
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    if logger is not None:
        logger.info(
            "Initialized frame_selector from %s (%s, epoch=%s, reset=%s)",
            checkpoint_path,
            state_key,
            observed_epoch,
            list(reset_keys),
        )
    return receipt


__all__ = [
    "RESETTABLE_SELECTOR_STATE_KEYS",
    "canonical_sha256",
    "initialize_frame_selector_from_checkpoint",
    "selector_state_dict",
    "sha256_file",
]
