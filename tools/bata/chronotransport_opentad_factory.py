#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from mmengine.config import Config

from opentad.datasets import build_dataloader, build_dataset
from opentad.models import build_detector
from opentad.models.chronotransport.replay import paired_detector_losses
from opentad.models.chronotransport.training import compose_stage_b_loss
from opentad.models.chronotransport.formal_stage_b import (
    select_schedule_for_step,
    validate_split_manifest,
)
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
    if split not in {"train", "fit", "calibration", "evaluation", "diagnostic"}:
        raise ValueError(
            "paired replay split must be train or diagnostic or formal fit/calibration/evaluation"
        )
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


def dataset_video_ids(dataset) -> list[str]:
    data_list = getattr(dataset, "data_list", None)
    if not isinstance(data_list, list) or not data_list:
        raise ValueError("OpenTAD Stage-B dataset requires a non-empty data_list")
    return sorted({str(row[0]) for row in data_list})


def filter_dataset_by_video_ids(dataset, video_ids) -> None:
    requested = set(map(str, video_ids))
    available = set(dataset_video_ids(dataset))
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"formal split ids are not present in dataset: {missing}")
    dataset.data_list = [row for row in dataset.data_list if str(row[0]) in requested]
    if not dataset.data_list:
        raise ValueError("formal Stage-B dataset filter produced no samples")


def make_replay_batch_source(
    *,
    config_path: str,
    video_ids,
    split: str,
    device: torch.device,
    seed: int,
):
    cfg = Config.fromfile(config_path)
    dataset = build_dataset(cfg.dataset.train)
    filter_dataset_by_video_ids(dataset, video_ids)

    def source(epoch: int = 0):
        set_seed(int(seed) + int(epoch))
        loader = build_dataloader(
            dataset,
            batch_size=1,
            rank=0,
            world_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        return _prepared_batches(loader, split=split, device=device)

    return source


def _chronotransport_runtime(detector):
    runtimes = [
        module
        for module in detector.modules()
        if module.__class__.__name__ == "ChronoTransportRuntime"
    ]
    if len(runtimes) != 1:
        raise RuntimeError("OpenTAD factory requires exactly one ChronoTransportRuntime")
    return runtimes[0]


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
    manifest_path = os.environ.get("CHRONOTRANSPORT_SPLIT_MANIFEST", "").strip()
    if manifest_path:
        manifest = validate_split_manifest(
            json.loads(Path(manifest_path).read_text(encoding="utf-8")),
            expected_video_ids=dataset_video_ids(dataset),
        )
        active_split = os.environ.get("CHRONOTRANSPORT_ACTIVE_SPLIT", split).strip()
        if active_split not in manifest["splits"]:
            raise ValueError(f"unknown formal Stage-B split: {active_split}")
        filter_dataset_by_video_ids(dataset, manifest["splits"][active_split])

    detector = build_detector(cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    use_ema = bool(getattr(cfg.solver, "ema", False))
    state, _ = select_checkpoint_state(checkpoint, use_ema=use_ema)
    detector.load_state_dict(_strip_ddp_prefix(state), strict=True)
    device = torch.device("cuda:0")
    detector = detector.to(device)
    detector.train()
    _chronotransport_runtime(detector).capture_replay_signals = True
    source = make_replay_batch_source(
        config_path=config_path,
        video_ids=dataset_video_ids(dataset),
        split=split,
        device=device,
        seed=seed,
    )
    restartable = os.environ.get("CHRONOTRANSPORT_RESTARTABLE_BATCHES", "0") == "1"
    return detector, source if restartable else source(0)


def stage_b_factory():
    split = os.environ.get("CHRONOTRANSPORT_REPLAY_SPLIT", "train").strip()
    if split not in {"train", "fit"}:
        raise ValueError("Stage B factory requires train or formal fit split")
    detector, batches = paired_replay_factory()
    schedule_text = os.environ.get("CHRONOTRANSPORT_STAGE_B_SCHEDULES", "").strip()
    if schedule_text:
        schedules = tuple(value.strip() for value in schedule_text.split(",") if value.strip())
    else:
        schedules = (
            os.environ.get(
                "CHRONOTRANSPORT_STAGE_B_SCHEDULE", "periodic2_transport"
            ).strip(),
        )
    if not schedules or len(schedules) != len(set(schedules)):
        raise ValueError("Stage B schedules must be non-empty and unique")
    transport_weight = float(os.environ.get("CHRONOTRANSPORT_TRANSPORT_WEIGHT", "0.1"))
    risk_weight = float(os.environ.get("CHRONOTRANSPORT_RISK_WEIGHT", "0.1"))
    learning_rate = float(os.environ.get("CHRONOTRANSPORT_STAGE_B_LR", "0.0001"))

    def loss_step(model, batch, step=1):
        schedule = select_schedule_for_step(step, schedules)
        forward_batch = dict(batch)
        forward_batch.pop("sample_id", None)
        forward_batch.pop("split", None)
        result = paired_detector_losses(
            model,
            forward_batch,
            counterfactual_schedule=schedule,
            track_counterfactual_grad=True,
        )
        if result.dense_features is None or result.counterfactual_features is None:
            raise RuntimeError("Stage B requires ephemeral dense/counterfactual features")
        runtime = _chronotransport_runtime(model)
        signals = runtime.latest_signals
        executed = runtime.latest_schedule
        if signals is None or executed is None:
            raise RuntimeError("Stage B requires deploy-visible signals and executed schedule")
        predicted = runtime.risk_predictor(
            signals,
            executed.actions.unsqueeze(1),
        ).squeeze(1)
        target = result.regret.detach().reshape(1).expand_as(predicted)
        losses = compose_stage_b_loss(
            counterfactual_task_loss=result.counterfactual_total,
            transported=result.counterfactual_features,
            dense_reference=result.dense_features,
            predicted_quantile=predicted,
            regret_target=target,
            transport_weight=transport_weight,
            risk_weight=risk_weight,
            quantile=float(runtime.risk_predictor.quantile),
        )
        return {
            "loss": losses.total,
            "task": losses.task,
            "transport": losses.transport,
            "risk": losses.risk,
            "regret": result.regret,
            "dense_loss": result.dense_total,
            "counterfactual_loss": result.counterfactual_total,
            "schedule": schedule,
        }

    def optimizer_factory(parameters):
        return torch.optim.AdamW(list(parameters), lr=learning_rate, weight_decay=0.0)

    return detector, batches, loss_step, optimizer_factory
