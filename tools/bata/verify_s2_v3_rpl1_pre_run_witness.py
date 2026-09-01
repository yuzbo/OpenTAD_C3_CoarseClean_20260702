from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Any, Mapping

import torch
import torch.distributed as dist
from mmengine.config import Config
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

from opentad.cores import build_optimizer
from opentad.datasets import build_dataloader, build_dataset
from opentad.models import build_detector
from opentad.utils import set_seed, setup_logger
from tools.bata.continuous_roi_s2_v3_full200_compute import (
    ARMS,
    PROTOCOL_ID,
    build_full_data_bundle,
    canonical_sha256,
    config_path,
)


def verify_arm_runtime(
    arm: str,
    seed: int,
    manifest: Mapping[str, Any],
    pretrained_path: Path,
    root: Path,
    rank: int,
    world_size: int,
    local_rank: int,
    logger: Any,
) -> dict[str, Any]:
    cfg_file = config_path(root, arm, seed)
    cfg = Config.fromfile(cfg_file)
    cfg.dataset.train.ann_file = manifest["training"]["training_only_annotation"]
    cfg.dataset.train.class_map = manifest["class_map"]["path"]
    cfg.dataset.train.data_path = manifest["media"]["root"]
    cfg.model.backbone.custom.pretrain = str(pretrained_path)
    set_seed(seed, False, deterministic_warn_only=True)

    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(seed + rank)
    train_loader = build_dataloader(
        train_dataset,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        drop_last=True,
        generator=dataloader_generator,
        **cfg.solver.train,
    )
    if len(train_loader) != 100:
        raise ValueError(f"{arm} train loader has {len(train_loader)} batches, expected 100")

    model = build_detector(cfg.model).to(local_rank)
    param_names = [name for name, _ in model.named_parameters()]
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())

    shared_backbone_info = {}
    if arm == "U128-A0":
        wrapper = model.backbone
        fusion_params = list(wrapper.fusion.parameters())
        if len(fusion_params) != 0:
            raise ValueError(f"U128-A0 fusion must have 0 parameters, got {len(fusion_params)}")
        shared_backbone_info = {
            "wrapper_type": type(wrapper).__name__,
            "fusion_mode": wrapper.fusion.mode,
            "fusion_parameters": len(fusion_params),
            "single_backbone_instance": wrapper.model.backbone is not None,
        }

    ddp_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        static_graph=True,
    )
    optimizer = build_optimizer(copy.deepcopy(cfg.optimizer), ddp_model, logger)
    scaler = GradScaler()

    first_batch = next(iter(train_loader))
    optimizer.zero_grad(set_to_none=True)

    with autocast(dtype=torch.float16, enabled=bool(cfg.solver.amp)):
        losses = ddp_model(**first_batch, return_loss=True)
        total_loss = losses["cost"] if "cost" in losses else sum(losses.values())

    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), cfg.solver.clip_grad_norm)
    scaler.step(optimizer)
    scaler.update()

    grad_norms = [
        float(p.grad.norm().item())
        for p in ddp_model.parameters()
        if p.grad is not None
    ]
    has_finite_grads = len(grad_norms) > 0 and all(torch.isfinite(torch.tensor(grad_norms)))

    return {
        "arm": arm,
        "seed": seed,
        "total_parameters": total_count,
        "trainable_parameters": trainable_count,
        "parameter_names_count": len(param_names),
        "optimizer_groups": len(optimizer.param_groups),
        "forward_backward_pass": "PASS",
        "has_finite_gradients": bool(has_finite_grads),
        "shared_backbone": shared_backbone_info,
    }


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    root = Path(__file__).resolve().parents[2]
    base = Path(os.environ.get("YUZIBO_ROOT", "/data/run01/sczc063/yuzibo"))
    anno = base / "thumos14/annotations/thumos_14_anno.json"
    cmap = base / "thumos14/annotations/category_idx.txt"
    media = base / "thumos14/raw_data/video"
    pretrained_path = base / "pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
    manifest_dir = root / "work_dirs" / "pre_run_manifest"

    manifest_file = manifest_dir / "full_data_manifest.json"
    if rank == 0:
        if not manifest_file.is_file():
            build_full_data_bundle(anno, cmap, media, manifest_dir)
    if world_size > 1:
        dist.barrier()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    logger = setup_logger("S2V3Witness", save_dir=str(manifest_dir), distributed_rank=rank)

    results = []
    for arm in ARMS:
        res = verify_arm_runtime(
            arm=arm,
            seed=4407,
            manifest=manifest,
            pretrained_path=pretrained_path,
            root=root,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            logger=logger,
        )
        results.append(res)

    if rank == 0:
        param_counts = {r["arm"]: r["trainable_parameters"] for r in results}
        if param_counts["D160"] != param_counts["G96"] or param_counts["D160"] != param_counts["U128-A0"]:
            raise ValueError(f"trainable parameter count mismatch across arms: {param_counts}")

        receipt = {
            "schema_version": "s2_v3_pre_run_2gpu_witness_v1",
            "protocol_id": PROTOCOL_ID,
            "world_size": world_size,
            "status": "PASS",
            "trainable_parameters_parity": param_counts,
            "arms": results,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
