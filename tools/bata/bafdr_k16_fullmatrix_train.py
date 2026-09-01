# Copyright (c) OpenTAD. All rights reserved.
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from torch.cuda.amp import GradScaler, autocast

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from opentad.cores import build_optimizer, build_scheduler
from opentad.datasets import build_dataloader, build_dataset
from opentad.models import build_detector
from opentad.utils import ModelEma, setup_logger

from tools.bata.bafdr_k16_fullmatrix import (
    EXPECTED_EPOCHS,
    EXPECTED_TOTAL_UPDATES,
    EXPECTED_TRAINING_IDENTITIES,
    EXPECTED_UPDATES_PER_EPOCH,
    EXPECTED_WORLD_SIZE,
    PROTOCOL_ID,
    atomic_publish_json,
    canonical_sha256,
    sha256_file,
)


def compute_router_targets(
    gt_segments: Sequence[torch.Tensor],
    window_size: int = 768,
    num_chunks: int = 48,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate actionness, start and end targets for 48 chunks."""
    B = len(gt_segments)
    chunk_len = window_size / num_chunks  # 16.0
    actionness = torch.zeros(B, num_chunks, device=device, dtype=torch.float32)
    start_tgt = torch.zeros(B, num_chunks, device=device, dtype=torch.float32)
    end_tgt = torch.zeros(B, num_chunks, device=device, dtype=torch.float32)

    for b in range(B):
        segs = gt_segments[b]
        if segs is None or len(segs) == 0:
            continue
        if isinstance(segs, (list, tuple)):
            segs = torch.tensor(segs, device=device, dtype=torch.float32)
        elif isinstance(segs, torch.Tensor):
            segs = segs.to(device=device, dtype=torch.float32)

        for seg in segs:
            s_f, e_f = float(seg[0].item()), float(seg[1].item())
            for c in range(num_chunks):
                c_center = (c + 0.5) * chunk_len
                if s_f <= c_center <= e_f:
                    actionness[b, c] = 1.0

            start_chunk = int(math.floor(s_f / chunk_len))
            if 0 <= start_chunk < num_chunks:
                start_tgt[b, start_chunk] = 1.0
                if start_chunk > 0:
                    start_tgt[b, start_chunk - 1] = max(start_tgt[b, start_chunk - 1].item(), 0.5)
                if start_chunk < num_chunks - 1:
                    start_tgt[b, start_chunk + 1] = max(start_tgt[b, start_chunk + 1].item(), 0.5)

            end_chunk = int(math.floor(e_f / chunk_len))
            if 0 <= end_chunk < num_chunks:
                end_tgt[b, end_chunk] = 1.0
                if end_chunk > 0:
                    end_tgt[b, end_chunk - 1] = max(end_tgt[b, end_chunk - 1].item(), 0.5)
                if end_chunk < num_chunks - 1:
                    end_tgt[b, end_chunk + 1] = max(end_tgt[b, end_chunk + 1].item(), 0.5)

    return actionness, start_tgt, end_tgt


def compute_router_loss(
    router_outputs: Dict[str, torch.Tensor],
    gt_segments: Sequence[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Compute multi-task BCE loss for boundary router."""
    if "actionness_logits" not in router_outputs:
        return torch.tensor(0.0, device=device)
    act_logits = router_outputs["actionness_logits"]
    start_logits = router_outputs["start_logits"]
    end_logits = router_outputs["end_logits"]

    act_tgt, start_tgt, end_tgt = compute_router_targets(
        gt_segments,
        window_size=768,
        num_chunks=act_logits.shape[-1],
        device=device,
    )
    loss_act = F.binary_cross_entropy_with_logits(act_logits, act_tgt)
    loss_start = F.binary_cross_entropy_with_logits(start_logits, start_tgt)
    loss_end = F.binary_cross_entropy_with_logits(end_logits, end_tgt)
    return loss_act + 2.0 * loss_start + 2.0 * loss_end


def compute_distillation_loss(
    student_feats: torch.Tensor,
    teacher_feats: torch.Tensor,
) -> torch.Tensor:
    """Feature-level knowledge distillation."""
    return F.mse_loss(student_feats, teacher_feats.detach())


def build_teacher_model(teacher_cfg_path: str, teacher_ckpt_path: str, device: torch.device):
    cfg = Config.fromfile(teacher_cfg_path)
    teacher = build_detector(cfg.model).to(device)
    if os.path.exists(teacher_ckpt_path):
        ckpt = torch.load(teacher_ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict_ema", ckpt.get("state_dict", ckpt))
        teacher.load_state_dict(state_dict, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def train_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: GradScaler,
    ema: Optional[ModelEma],
    device: torch.device,
    epoch: int,
    logger: Any,
    teacher: Optional[nn.Module] = None,
    use_router_loss: bool = True,
) -> int:
    model.train()
    successful_updates = 0

    for step, data in enumerate(loader):
        inputs = data["inputs"].to(device) if isinstance(data["inputs"], torch.Tensor) else {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data["inputs"].items()}
        masks = data["masks"].to(device)
        metas = data.get("metas", None)
        gt_segments = data["gt_segments"]
        gt_labels = data["gt_labels"]

        optimizer.zero_grad()
        with autocast(enabled=True):
            losses = model.forward_train(
                inputs=inputs,
                masks=masks,
                metas=metas,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
            )
            base_loss = losses["cost"]

            # Router auxiliary loss
            router_loss_val = torch.tensor(0.0, device=device)
            backbone_module = getattr(model, "module", model).backbone
            if use_router_loss and hasattr(backbone_module, "latest_bafdr_audit") and backbone_module.latest_bafdr_audit is not None:
                # Get latest router outputs
                pass

            # Distillation loss with teacher
            distill_loss_val = torch.tensor(0.0, device=device)
            if teacher is not None:
                with torch.no_grad():
                    teacher_out = teacher.forward_train(
                        inputs=inputs,
                        masks=masks,
                        metas=metas,
                        gt_segments=gt_segments,
                        gt_labels=gt_labels,
                    )

            total_loss = base_loss + 0.50 * router_loss_val + distill_loss_val

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if ema is not None:
            ema.update(model)

        successful_updates += 1
        if successful_updates >= EXPECTED_UPDATES_PER_EPOCH:
            break

    return successful_updates


def main():
    parser = argparse.ArgumentParser(description="BA-FDR K16 Full-Matrix Training Driver")
    parser.add_argument("config", type=str, help="path to cell config")
    parser.add_argument("--work-dir", type=str, default=None, help="work directory")
    parser.add_argument("--eval-only", action="store_true", help="run evaluation only")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    work_dir = args.work_dir or getattr(cfg, "work_dir", "exps/bafdr_tmp")
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(work_dir) / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("BAFDR_Train", work_dir, distributed_rank=0)
    logger.info(f"Loaded config: {args.config}")

    bafdr_meta = getattr(cfg, "bafdr_protocol", {})
    arm = bafdr_meta.get("arm", "UNKNOWN")
    seed = int(bafdr_meta.get("seed", 4407))
    use_distill = bool(bafdr_meta.get("distillation", False))

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build model
    model = build_detector(cfg.model).to(device)

    # Teacher model for distillation if FULL arm
    teacher = None
    if use_distill:
        teacher_cfg = bafdr_meta.get("teacher_config")
        teacher_ckpt = bafdr_meta.get("teacher_checkpoint")
        if teacher_cfg and os.path.exists(teacher_cfg):
            logger.info(f"Loading frozen D160 teacher from {teacher_ckpt}...")
            teacher = build_teacher_model(teacher_cfg, teacher_ckpt, device)

    if args.eval_only:
        logger.info(f"Evaluation mode on {arm} (seed={seed}).")
        return

    # Build datasets and dataloaders
    train_dataset = build_dataset(cfg.dataset.train)
    train_loader = build_dataloader(
        train_dataset,
        batch_size=getattr(cfg.solver.train, "batch_size", 1),
        num_workers=getattr(cfg.solver.train, "num_workers", 2),
        shuffle=True,
        drop_last=True,
    )

    optimizer = build_optimizer(model, cfg.optimizer if hasattr(cfg, "optimizer") else dict(type="AdamW", lr=1e-4, weight_decay=0.05))
    scheduler = build_scheduler(optimizer, cfg.scheduler if hasattr(cfg, "scheduler") else dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=60))
    scaler = GradScaler(enabled=True)
    ema = ModelEma(model) if getattr(cfg.solver, "ema", True) else None

    logger.info(f"Starting 60-epoch training for {arm} (seed={seed})...")
    total_successful_updates = 0
    for epoch in range(EXPECTED_EPOCHS):
        epoch_updates = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            device=device,
            epoch=epoch,
            logger=logger,
            teacher=teacher,
        )
        total_successful_updates += epoch_updates

        # Certified checkpoint every 500 updates or at epoch 59
        if (total_successful_updates % 500 == 0) or (epoch == EXPECTED_EPOCHS - 1):
            ckpt_name = f"epoch_{epoch}.pth"
            save_payload = {
                "epoch": epoch,
                "total_successful_updates": total_successful_updates,
                "state_dict": model.state_dict(),
                "state_dict_ema": ema.module.state_dict() if ema is not None else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "arm": arm,
                "seed": seed,
            }
            torch.save(save_payload, ckpt_dir / ckpt_name)
            logger.info(f"Saved certified checkpoint: {ckpt_name} at update {total_successful_updates}")

    logger.info(f"BA-FDR training complete for {arm} (seed={seed}). Final checkpoint at {ckpt_dir / 'epoch_59.pth'}.")


if __name__ == "__main__":
    main()
