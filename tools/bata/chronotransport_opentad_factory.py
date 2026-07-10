#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from mmengine.config import Config

from opentad.datasets import build_dataloader, build_dataset
from opentad.models import build_detector
from opentad.utils import set_seed
from tools.bata.check_chronotransport_checkpoint import (
    _strip_ddp_prefix,
    select_checkpoint_state,
)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def prepare_replay_batch(
    source: Mapping[str, object], *, batch_index: int, split: str
) -> dict[str, object]:
    split = str(split)
    if split not in {"train", "diagnostic"}:
        raise ValueError("paired replay split must be train or diagnostic")
    batch = dict(source)
    metas = batch.get("metas")
    if not isinstance(metas, (list, tuple)) or len(metas) != 1:
        raise ValueError("paired replay v1 requires batch_size=1 with one metadata record")
    meta = metas[0]
    if not isinstance(meta, Mapping) or not str(meta.get("video_name", "")).strip():
        raise ValueError("paired replay metadata requires video_name")
    video_name = str(meta["video_name"])
    batch["sample_id"] = f"{video_name}:{int(batch_index):06d}"
    batch["split"] = split
    batch["return_loss"] = True
    return batch


def move_batch_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_batch_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_batch_to_device(item, device) for item in value)
    return value


def _prepared_batches(
    loader, *, split: str, device: torch.device
) -> Iterator[dict[str, object]]:
    for batch_index, batch in enumerate(loader):
        moved = move_batch_to_device(batch, device)
        yield prepare_replay_batch(moved, batch_index=batch_index, split=split)


def paired_replay_factory():
    config_path = os.environ.get(
        "CHRONOTRANSPORT_CONFIG",
        "configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_a.py",
    )
    checkpoint_path = _required_env("CHRONOTRANSPORT_CHECKPOINT")
    split = os.environ.get("CHRONOTRANSPORT_REPLAY_SPLIT", "diagnostic").strip()
    seed = int(os.environ.get("CHRONOTRANSPORT_SEED", "42"))
    if not torch.cuda.is_available():
        raise RuntimeError("paired replay requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("paired replay expects exactly one visible CUDA device")

    set_seed(seed)
    torch.cuda.set_device(0)
    cfg = Config.fromfile(config_path)
    dataset = build_dataset(cfg.dataset.train)
    loader = build_dataloader(
        dataset,
        batch_size=1,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    detector = build_detector(cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    use_ema = bool(getattr(cfg.solver, "ema", False))
    state, _ = select_checkpoint_state(checkpoint, use_ema=use_ema)
    detector.load_state_dict(_strip_ddp_prefix(state), strict=True)
    device = torch.device("cuda:0")
    detector = detector.to(device)
    detector.train()
    return detector, _prepared_batches(loader, split=split, device=device)
