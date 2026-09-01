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
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import load_checkpoint

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

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
    """Generate actionness, start and end targets for 48 chunks.
    Args:
        gt_segments: list of [N, 2] tensors with start and end times in frames [0, window_size]
    Returns:
        actionness_targets: [B, 48]
        start_targets: [B, 48]
        end_targets: [B, 48]
    """
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
            # Actionness: chunk centers
            for c in range(num_chunks):
                c_center = (c + 0.5) * chunk_len
                if s_f <= c_center <= e_f:
                    actionness[b, c] = 1.0

            # Start target: exact chunk + adjacent 0.5
            start_chunk = int(math.floor(s_f / chunk_len))
            if 0 <= start_chunk < num_chunks:
                start_tgt[b, start_chunk] = 1.0
                if start_chunk > 0:
                    start_tgt[b, start_chunk - 1] = max(start_tgt[b, start_chunk - 1].item(), 0.5)
                if start_chunk < num_chunks - 1:
                    start_tgt[b, start_chunk + 1] = max(start_tgt[b, start_chunk + 1].item(), 0.5)

            # End target: exact chunk + adjacent 0.5
            end_chunk = int(math.floor(e_f / chunk_len))
            if 0 <= end_chunk < num_chunks:
                end_tgt[b, end_chunk] = 1.0
                if end_chunk > 0:
                    end_tgt[b, end_chunk - 1] = max(end_tgt[b, end_chunk - 1].item(), 0.5)
                if end_chunk < num_chunks - 1:
                    end_tgt[b, end_chunk + 1] = max(end_tgt[b, end_chunk + 1].item(), 0.5)

    return actionness, start_tgt, end_tgt


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


def main():
    parser = argparse.ArgumentParser(description="BA-FDR K16 Full-Matrix Training Driver")
    parser.add_argument("config", type=str, help="path to cell config")
    parser.add_argument("--work-dir", type=str, default=None, help="work directory")
    parser.add_argument("--resume", action="store_true", help="resume training from recovery")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    work_dir = args.work_dir or getattr(cfg, "work_dir", "exps/bafdr_tmp")
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("BAFDR_Train", work_dir, distributed_rank=0)
    logger.info(f"Loaded config: {args.config}")

    bafdr_meta = getattr(cfg, "bafdr_protocol", {})
    arm = bafdr_meta.get("arm", "UNKNOWN")
    seed = int(bafdr_meta.get("seed", 4407))
    use_distill = bool(bafdr_meta.get("distillation", False))

    # Reproducibility seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build model
    model = build_detector(cfg.model).to(device)
    model.train()

    # Teacher model for distillation if FULL arm
    teacher = None
    if use_distill:
        teacher_cfg = bafdr_meta.get("teacher_config")
        teacher_ckpt = bafdr_meta.get("teacher_checkpoint")
        if teacher_cfg and os.path.exists(teacher_cfg):
            logger.info(f"Loading frozen D160 teacher from {teacher_ckpt}...")
            teacher = build_teacher_model(teacher_cfg, teacher_ckpt, device)

    logger.info(f"Model initialized for arm {arm} (seed={seed}, distill={use_distill}).")
    logger.info("BA-FDR training engine ready.")


if __name__ == "__main__":
    main()
