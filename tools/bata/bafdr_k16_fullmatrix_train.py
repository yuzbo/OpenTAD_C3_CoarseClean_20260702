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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from opentad.cores import build_optimizer, build_scheduler, eval_one_epoch
from opentad.datasets import build_dataloader, build_dataset
from opentad.models import build_detector
from opentad.utils import ModelEma, setup_logger

PROTOCOL_ID = "ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001"
EXPECTED_EPOCHS = 60
EXPECTED_UPDATES_PER_EPOCH = 100
EXPECTED_TOTAL_UPDATES = 6000
EXPECTED_WORLD_SIZE = 2
EXPECTED_TRAIN_SAMPLES = 200
EXPECTED_EVAL_WINDOWS = 792
EXPECTED_GLOBAL_BATCH_SIZE = 2
EXPECTED_LOCAL_BATCH_SIZE = 1
WINDOW_SIZE = 768


def atomic_publish_json(target_path: Path, payload: Mapping[str, Any]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, target_path)


def atomic_save_checkpoint(target_path: Path, payload: Mapping[str, Any]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temp_path)
    os.replace(temp_path, target_path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(repo_root: Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def is_rank0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    current = model
    while hasattr(current, "module"):
        current = current.module
    return current


def _capture_model_buffers(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
        if buffer is not None
    }


def _restore_model_buffers(model: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
    current = {name: buffer for name, buffer in model.named_buffers() if buffer is not None}
    if set(current) != set(snapshot):
        raise RuntimeError("BA-FDR model buffer registry changed during an AMP retry")
    for name, saved in snapshot.items():
        current[name].copy_(saved)


def strip_module_prefix(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    if not state_dict:
        return dict(state_dict)
    if all(str(key).startswith("module.") for key in state_dict):
        return {str(key)[7:]: value for key, value in state_dict.items()}
    return dict(state_dict)


def add_module_prefix(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    if not state_dict:
        return dict(state_dict)
    if any(str(key).startswith("module.") for key in state_dict):
        return dict(state_dict)
    return {f"module.{key}": value for key, value in state_dict.items()}


def load_state_dict_strict(module: nn.Module, state_dict: Mapping[str, Any], *, label: str) -> None:
    candidates = [dict(state_dict), strip_module_prefix(state_dict), add_module_prefix(state_dict)]
    errors = []
    for candidate in candidates:
        try:
            module.load_state_dict(candidate, strict=True)
            return
        except RuntimeError as exc:
            errors.append(str(exc))
    raise RuntimeError(f"Strict checkpoint load failed for {label}:\n" + "\n---\n".join(errors))


def resolve_path(path_value: str | Path, *, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def resolve_existing_path(path_value: str | Path, *, repo_root: Path, label: str) -> Path:
    path = resolve_path(path_value, repo_root=repo_root)
    if not path.exists():
        raise FileNotFoundError(f"Required {label} not found: {path}")
    return path


def resolve_teacher_checkpoint(path_value: str | Path, *, repo_root: Path, work_dir: Path, seed: int) -> Path:
    configured = resolve_path(path_value, repo_root=repo_root)
    if configured.exists():
        return configured
    sibling = work_dir.parent / f"bafdr_k16_d160_seed{seed}" / "checkpoint" / "epoch_59.pth"
    if sibling.exists():
        return sibling
    raise FileNotFoundError(
        "Required D160 teacher checkpoint not found. Tried configured path "
        f"{configured} and matrix sibling {sibling}."
    )


def init_distributed(expected_world_size: int, allow_single_process: bool) -> tuple[torch.device, int, int, int]:
    if not allow_single_process and int(expected_world_size) != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            "BA-FDR formal execution fixes world_size=2; override it only for "
            "an explicitly allowed local precheck/smoke run"
        )
    env_has_rank = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not env_has_rank:
        if not allow_single_process:
            raise RuntimeError(
                "BA-FDR formal driver must be launched with torchrun/srun world_size=2. "
                "Use --allow-single-process only for local smoke/precheck runs."
            )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    if world_size != expected_world_size and not allow_single_process:
        raise RuntimeError(f"BA-FDR protocol requires world_size={expected_world_size}; got {world_size}")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = torch.device(f"cuda:{local_rank}")
    elif allow_single_process:
        backend = "gloo"
        device = torch.device("cpu")
    else:
        raise RuntimeError("BA-FDR formal distributed training requires CUDA devices")

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return device, rank, world_size, local_rank


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def move_tensor_tree(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_tensor_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensor_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensor_tree(item, device) for item in value)
    return value


def move_inputs_to_device(inputs: Any, device: torch.device) -> Any:
    if torch.is_tensor(inputs):
        return inputs.to(device, non_blocking=True)
    if isinstance(inputs, Mapping):
        moved = {}
        for key, value in inputs.items():
            if key == "source" and torch.is_tensor(value):
                moved[key] = value
            elif torch.is_tensor(value):
                moved[key] = value.to(device, non_blocking=True)
            else:
                moved[key] = move_tensor_tree(value, device)
        return moved
    return inputs


def prepare_forward_kwargs(data: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    kwargs = dict(data)
    if "inputs" in kwargs:
        kwargs["inputs"] = move_inputs_to_device(kwargs["inputs"], device)
    if "masks" in kwargs:
        kwargs["masks"] = move_tensor_tree(kwargs["masks"], device)
    if "gt_segments" in kwargs:
        kwargs["gt_segments"] = move_tensor_tree(kwargs["gt_segments"], device)
    if "gt_labels" in kwargs:
        kwargs["gt_labels"] = move_tensor_tree(kwargs["gt_labels"], device)
    return kwargs


class DeviceBatchAdapter(nn.Module):
    def __init__(self, module: nn.Module, device: torch.device):
        super().__init__()
        self.module = module
        self.device = device

    def forward(self, **kwargs):
        return self.module(**prepare_forward_kwargs(kwargs, self.device))


def compute_router_targets(
    gt_segments: Sequence[torch.Tensor],
    window_size: int = WINDOW_SIZE,
    num_chunks: int = 48,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = len(gt_segments)
    chunk_len = window_size / num_chunks
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
                    start_tgt[b, start_chunk - 1] = torch.maximum(
                        start_tgt[b, start_chunk - 1], start_tgt.new_tensor(0.5)
                    )
                if start_chunk < num_chunks - 1:
                    start_tgt[b, start_chunk + 1] = torch.maximum(
                        start_tgt[b, start_chunk + 1], start_tgt.new_tensor(0.5)
                    )

            end_chunk = int(math.floor(e_f / chunk_len))
            if 0 <= end_chunk < num_chunks:
                end_tgt[b, end_chunk] = 1.0
                if end_chunk > 0:
                    end_tgt[b, end_chunk - 1] = torch.maximum(
                        end_tgt[b, end_chunk - 1], end_tgt.new_tensor(0.5)
                    )
                if end_chunk < num_chunks - 1:
                    end_tgt[b, end_chunk + 1] = torch.maximum(
                        end_tgt[b, end_chunk + 1], end_tgt.new_tensor(0.5)
                    )

    return actionness, start_tgt, end_tgt


def compute_router_loss(
    router_outputs: Dict[str, torch.Tensor],
    gt_segments: Sequence[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    if "actionness_logits" not in router_outputs:
        return torch.tensor(0.0, device=device)
    act_logits = router_outputs["actionness_logits"]
    start_logits = router_outputs["start_logits"]
    end_logits = router_outputs["end_logits"]

    act_tgt, start_tgt, end_tgt = compute_router_targets(
        gt_segments,
        window_size=WINDOW_SIZE,
        num_chunks=act_logits.shape[-1],
        device=device,
    )
    loss_act = F.binary_cross_entropy_with_logits(act_logits, act_tgt)
    loss_start = F.binary_cross_entropy_with_logits(start_logits, start_tgt)
    loss_end = F.binary_cross_entropy_with_logits(end_logits, end_tgt)
    return loss_act + 2.0 * loss_start + 2.0 * loss_end


def build_teacher_model(teacher_cfg_path: Path, teacher_ckpt_path: Path, device: torch.device) -> nn.Module:
    cfg = Config.fromfile(str(teacher_cfg_path))
    teacher = build_detector(cfg.model).to(device)
    ckpt = torch.load(teacher_ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict_ema", ckpt.get("state_dict", ckpt))
    load_state_dict_strict(teacher, state_dict, label=f"D160 teacher {teacher_ckpt_path}")
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def make_teacher_d160_inputs(inputs: Mapping[str, Any], device: torch.device, output_size: int = 160) -> torch.Tensor:
    source = inputs.get("source")
    if not torch.is_tensor(source):
        raise ValueError("BA-FDR FULL distillation requires inputs['source'] for D160 teacher replay")
    if source.ndim != 6:
        raise ValueError(f"BA-FDR source must be [B, N, 3, T, H, W], got {tuple(source.shape)}")
    if source.dtype != torch.uint8:
        raise TypeError(f"BA-FDR source must stay uint8 before teacher replay, got {source.dtype}")

    B, N, C, T, H, W = source.shape
    if C != 3:
        raise ValueError(f"D160 teacher replay requires RGB source, got channels={C}")

    flat = source.permute(0, 1, 3, 2, 4, 5).reshape(B * N * T, C, H, W).to(device=device, dtype=torch.float32)
    scale = min(float(output_size) / float(H), float(output_size) / float(W))
    resized_h = max(int(round(H * scale)), 1)
    resized_w = max(int(round(W * scale)), 1)
    resized = F.interpolate(flat, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    canvas = resized.new_zeros((flat.shape[0], C, output_size, output_size))
    top = (output_size - resized_h) // 2
    left = (output_size - resized_w) // 2
    canvas[:, :, top : top + resized_h, left : left + resized_w] = resized
    teacher = canvas.clamp_(0, 255).round_().to(dtype=torch.uint8)
    return teacher.reshape(B, N, T, C, output_size, output_size).permute(0, 1, 3, 2, 4, 5).contiguous()


def boundary_short_weights(gt_segments: Sequence[torch.Tensor], length: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32)
    weights = torch.ones(len(gt_segments), length, device=device, dtype=torch.float32)
    sigma = max(float(length) / 96.0, 1.0)
    scale = float(length) / float(WINDOW_SIZE)
    for batch_idx, segs in enumerate(gt_segments):
        if segs is None or len(segs) == 0:
            continue
        segs = segs.to(device=device, dtype=torch.float32) if torch.is_tensor(segs) else torch.tensor(segs, device=device)
        for seg in segs:
            for edge in (float(seg[0].item()), float(seg[1].item())):
                center = edge * scale
                weights[batch_idx] = torch.maximum(
                    weights[batch_idx],
                    1.0 + 2.0 * torch.exp(-0.5 * ((positions - center) / sigma) ** 2),
                )
    return weights


def tensor_from_backbone_output(output: Any) -> torch.Tensor:
    if isinstance(output, Mapping):
        for key in ("fused_features", "feats", "global_features"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
    if torch.is_tensor(output):
        return output
    raise RuntimeError("Cannot extract tensor feature from backbone output for BA-FDR KD")


def feature_distill_loss(student: torch.Tensor, teacher: torch.Tensor, gt_segments, device: torch.device) -> torch.Tensor:
    if student.ndim != 3 or teacher.ndim != 3:
        raise ValueError("Feature KD expects [B, C, T] tensors")
    if student.shape[1] != teacher.shape[1]:
        raise ValueError(f"Feature KD channel mismatch: student={student.shape[1]}, teacher={teacher.shape[1]}")
    if student.shape[-1] != teacher.shape[-1]:
        teacher = F.interpolate(teacher, size=student.shape[-1], mode="linear", align_corners=False)
    weights = boundary_short_weights(gt_segments, student.shape[-1], device).to(student.dtype)
    loss = F.smooth_l1_loss(student, teacher.detach(), reduction="none").mean(dim=1)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def projection_l01_loss(student_output: Any, teacher_output: Any, gt_segments, device: torch.device) -> torch.Tensor:
    if not isinstance(student_output, (list, tuple)) or not isinstance(teacher_output, (list, tuple)):
        return torch.tensor(0.0, device=device)
    losses = []
    for student_level, teacher_level in zip(student_output[:2], teacher_output[:2]):
        if torch.is_tensor(student_level) and torch.is_tensor(teacher_level):
            losses.append(feature_distill_loss(student_level, teacher_level, gt_segments, device))
    if not losses:
        return torch.tensor(0.0, device=device)
    return sum(losses) / float(len(losses))


def head_kd_tensors(kd_outputs: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cls = torch.cat([item.permute(0, 2, 1) for item in kd_outputs["cls_pred"]], dim=1)
    reg = torch.cat([item.permute(0, 2, 1) for item in kd_outputs["reg_pred"]], dim=1)
    mask = torch.cat([item.bool() for item in kd_outputs["mask_list"]], dim=1)
    return cls, reg, mask


def align_temporal_tensor(source: torch.Tensor, target_length: int, mode: str = "linear") -> torch.Tensor:
    if source.shape[1] == target_length:
        return source
    if mode == "nearest":
        return (
            F.interpolate(source.transpose(1, 2).float(), size=target_length, mode="nearest")
            .transpose(1, 2)
            .bool()
        )
    return F.interpolate(source.transpose(1, 2), size=target_length, mode="linear", align_corners=False).transpose(1, 2)


def head_distill_losses(student_head: Mapping[str, Any], teacher_head: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    student_cls, student_reg, student_mask = head_kd_tensors(student_head)
    teacher_cls, teacher_reg, teacher_mask = head_kd_tensors(teacher_head)
    if teacher_cls.shape[1] != student_cls.shape[1]:
        teacher_cls = align_temporal_tensor(teacher_cls, student_cls.shape[1])
        teacher_reg = align_temporal_tensor(teacher_reg, student_reg.shape[1])
        teacher_mask = align_temporal_tensor(teacher_mask.unsqueeze(-1), student_mask.shape[1], mode="nearest").squeeze(-1)
    if student_cls.shape[-1] != teacher_cls.shape[-1] or student_reg.shape[-1] != teacher_reg.shape[-1]:
        raise ValueError("Head KD class/regression channel mismatch")

    valid = (student_mask & teacher_mask).unsqueeze(-1)
    teacher_prob = torch.sigmoid(teacher_cls.detach())
    cls_loss = F.binary_cross_entropy_with_logits(student_cls, teacher_prob, reduction="none")
    cls_loss = (cls_loss * valid).sum() / valid.sum().clamp_min(1).to(cls_loss.dtype)

    teacher_conf = teacher_prob.amax(dim=-1)
    reg_valid = student_mask & teacher_mask & (teacher_conf >= 0.30)
    reg_loss = F.smooth_l1_loss(student_reg, teacher_reg.detach(), reduction="none").mean(dim=-1)
    reg_loss = (reg_loss * reg_valid).sum() / reg_valid.sum().clamp_min(1).to(reg_loss.dtype)
    return cls_loss, reg_loss


def compute_bafdr_distillation_losses(
    student_model: nn.Module,
    teacher_model: nn.Module,
    *,
    inputs: Mapping[str, Any],
    masks: torch.Tensor,
    metas: Any,
    gt_segments: Sequence[torch.Tensor],
    gt_labels: Sequence[torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    teacher_inputs = make_teacher_d160_inputs(inputs, device)
    with torch.no_grad():
        teacher_model(
            inputs=teacher_inputs,
            masks=masks,
            metas=metas,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            return_loss=True,
        )

    student = unwrap_model(student_model)
    teacher = unwrap_model(teacher_model)
    student_backbone = tensor_from_backbone_output(getattr(student, "_latest_backbone_output", None))
    teacher_backbone = tensor_from_backbone_output(getattr(teacher, "_latest_backbone_output", None))
    backbone_loss = feature_distill_loss(student_backbone, teacher_backbone, gt_segments, device)
    proj_loss = projection_l01_loss(
        getattr(student, "_latest_projection_output", None),
        getattr(teacher, "_latest_projection_output", None),
        gt_segments,
        device,
    )
    if torch.is_tensor(proj_loss) and float(proj_loss.detach().item()) != 0.0:
        feature_loss = 0.5 * (backbone_loss + proj_loss)
    else:
        feature_loss = backbone_loss

    student_head = getattr(student.rpn_head, "latest_kd_outputs", None)
    teacher_head = getattr(teacher.rpn_head, "latest_kd_outputs", None)
    if student_head is None or teacher_head is None:
        raise RuntimeError("BA-FDR KD requires rpn_head.latest_kd_outputs from both student and teacher")
    cls_kd, reg_kd = head_distill_losses(student_head, teacher_head)
    return {
        "feature_distill_loss": 0.20 * feature_loss,
        "cls_kd_loss": 0.20 * cls_kd,
        "reg_kd_loss": 0.10 * reg_kd,
    }


def validate_dataset_lengths(train_dataset: Any, val_dataset: Any, logger: Any) -> None:
    train_len = len(train_dataset)
    val_len = len(val_dataset)
    if train_len != EXPECTED_TRAIN_SAMPLES:
        raise RuntimeError(f"BA-FDR protocol requires {EXPECTED_TRAIN_SAMPLES} training samples; got {train_len}")
    if val_len != EXPECTED_EVAL_WINDOWS:
        raise RuntimeError(f"BA-FDR protocol requires {EXPECTED_EVAL_WINDOWS} eval windows; got {val_len}")
    logger.info(f"Protocol population check passed: train={train_len}, eval_windows={val_len}")


def validate_loader_batch_contract(split: str, batch_size: int, world_size: int) -> None:
    if batch_size != EXPECTED_GLOBAL_BATCH_SIZE:
        raise RuntimeError(
            f"BA-FDR protocol requires solver.{split}.batch_size={EXPECTED_GLOBAL_BATCH_SIZE} "
            f"(job-global); got {batch_size}"
        )
    if batch_size % world_size != 0:
        raise RuntimeError(
            f"solver.{split}.batch_size={batch_size} must be divisible by world_size={world_size}"
        )
    local_batch_size = batch_size // world_size
    if world_size == EXPECTED_WORLD_SIZE and local_batch_size != EXPECTED_LOCAL_BATCH_SIZE:
        raise RuntimeError(
            f"BA-FDR formal world_size={EXPECTED_WORLD_SIZE} requires local batch "
            f"{EXPECTED_LOCAL_BATCH_SIZE}; got {local_batch_size}"
        )


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
    use_amp: bool = True,
    max_amp_retries_per_batch: int = 0,
    fail_on_skipped_update: bool = True,
    schedule_and_ema_on_success_only: bool = True,
    fail_on_nonfinite_loss: bool = True,
) -> int:
    model.train()
    successful_updates = 0

    max_amp_retries_per_batch = int(max_amp_retries_per_batch)
    if max_amp_retries_per_batch < 0:
        raise ValueError("max_amp_retries_per_batch must be non-negative")
    if max_amp_retries_per_batch > 0 and not use_amp:
        raise ValueError("AMP retries require use_amp=True")

    for _, data in enumerate(loader):
        kwargs = prepare_forward_kwargs(data, device)
        retry_count = 0
        cpu_rng_state = torch.get_rng_state() if max_amp_retries_per_batch > 0 else None
        cuda_rng_states = (
            torch.cuda.get_rng_state_all()
            if max_amp_retries_per_batch > 0 and torch.cuda.is_available()
            else None
        )
        buffer_state = (
            _capture_model_buffers(model) if max_amp_retries_per_batch > 0 else None
        )

        while True:
            if retry_count > 0:
                # Replaying the same sampled batch keeps a retry from changing the
                # stochastic route or dropout mask.
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
                if buffer_state is not None:
                    _restore_model_buffers(model, buffer_state)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                losses = model(
                    inputs=kwargs["inputs"],
                    masks=kwargs["masks"],
                    metas=kwargs.get("metas", None),
                    gt_segments=kwargs["gt_segments"],
                    gt_labels=kwargs["gt_labels"],
                    return_loss=True,
                )
                base_loss = losses["cost"]
                if teacher is not None:
                    if not isinstance(kwargs["inputs"], Mapping):
                        raise TypeError("BA-FDR FULL distillation requires mapping inputs")
                    kd_losses = compute_bafdr_distillation_losses(
                        student_model=model,
                        teacher_model=teacher,
                        inputs=kwargs["inputs"],
                        masks=kwargs["masks"],
                        metas=kwargs.get("metas", None),
                        gt_segments=kwargs["gt_segments"],
                        gt_labels=kwargs["gt_labels"],
                        device=device,
                    )
                    distill_loss = sum(kd_losses.values())
                    losses.update(kd_losses)
                    losses["distill_loss"] = distill_loss
                    losses["cost"] = base_loss + distill_loss

            total_loss = losses["cost"]
            if fail_on_nonfinite_loss and not bool(torch.isfinite(total_loss.detach()).all()):
                raise FloatingPointError(
                    f"BA-FDR produced a non-finite loss at epoch={epoch}, retry={retry_count}"
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norm_tensor = (
                grad_norm
                if torch.is_tensor(grad_norm)
                else torch.as_tensor(grad_norm, device=total_loss.device)
            )
            if (
                fail_on_nonfinite_loss
                and not use_amp
                and not bool(torch.isfinite(grad_norm_tensor).all())
            ):
                raise FloatingPointError(
                    f"BA-FDR produced non-finite gradients at epoch={epoch}, retry={retry_count}"
                )

            scale_before = float(scaler.get_scale()) if use_amp else None
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale()) if use_amp else None
            # GradScaler lowers its scale when it skipped the optimizer step. A
            # successful update must be the only event that advances the schedule.
            update_succeeded = (not use_amp) or scale_after >= scale_before
            if update_succeeded:
                break

            retry_count += 1
            if retry_count > max_amp_retries_per_batch:
                if fail_on_skipped_update:
                    raise FloatingPointError(
                        "BA-FDR AMP could not produce a successful optimizer update "
                        f"after {max_amp_retries_per_batch} retries"
                    )
                break
            logger.info(
                "[Train]: AMP skipped batch; retry %d/%d with scale %.1f",
                retry_count,
                max_amp_retries_per_batch,
                scale_after,
            )

        successful_updates += int(update_succeeded)
        if update_succeeded or not schedule_and_ema_on_success_only:
            scheduler.step()
            if ema is not None:
                ema.update(model)
        if successful_updates >= EXPECTED_UPDATES_PER_EPOCH:
            break

    if successful_updates != EXPECTED_UPDATES_PER_EPOCH:
        raise RuntimeError(
            f"Epoch {epoch} produced {successful_updates} updates; "
            f"expected {EXPECTED_UPDATES_PER_EPOCH}"
        )
    return successful_updates


def build_eval_loader(cfg: Config, rank: int, world_size: int, logger: Any):
    val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=logger))
    val_batch_size = int(getattr(cfg.solver.val, "batch_size", 2))
    validate_loader_batch_contract("val", val_batch_size, world_size)
    val_num_workers = int(getattr(cfg.solver.val, "num_workers", 2))
    val_loader = build_dataloader(
        val_dataset,
        batch_size=val_batch_size,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        drop_last=False,
        num_workers=val_num_workers,
    )
    return val_dataset, val_loader


def run_eval(
    *,
    model: nn.Module,
    ema: Optional[ModelEma],
    val_loader: Any,
    cfg: Config,
    logger: Any,
    rank: int,
    world_size: int,
    device: torch.device,
    use_amp: bool,
    not_eval: bool,
) -> Optional[Dict[str, Any]]:
    eval_model = DeviceBatchAdapter(model, device)
    restore_state = None
    if ema is not None:
        restore_state = copy.deepcopy(model.state_dict())
        load_state_dict_strict(model, ema.module.state_dict(), label="evaluation EMA")
    try:
        return eval_one_epoch(
            test_loader=val_loader,
            model=eval_model,
            cfg=cfg,
            logger=logger,
            rank=rank,
            model_ema=None,
            use_amp=use_amp,
            world_size=world_size,
            not_eval=not_eval,
            epoch=EXPECTED_EPOCHS - 1,
        )
    finally:
        if restore_state is not None:
            load_state_dict_strict(model, restore_state, label="restore non-EMA model")


def save_training_receipt(
    *,
    work_dir: Path,
    cfg_path: Path,
    arm: str,
    seed: int,
    total_updates: int,
    checkpoint_path: Path,
    eval_results: Optional[Mapping[str, Any]],
    rank: int,
    world_size: int,
    train_samples: Optional[int] = None,
    eval_windows: Optional[int] = None,
    receipt_name: str = "eval_receipt.json",
    phase: str = "evaluation",
    metric_opened: bool = True,
) -> None:
    if not is_rank0():
        return
    receipt = {
        "schema_version": "ZOOMTOKEN-BA-FDR-K16-RECEIPT-v002",
        "protocol_id": PROTOCOL_ID,
        "arm": arm,
        "seed": seed,
        "rank_seed": seed + rank,
        "rank": rank,
        "world_size": world_size,
        "phase": phase,
        "metric_opened": metric_opened,
        "expected_epochs": EXPECTED_EPOCHS,
        "expected_total_updates": EXPECTED_TOTAL_UPDATES,
        "total_successful_updates": total_updates,
        "train_samples": None if train_samples is None else int(train_samples),
        "eval_windows": None if eval_windows is None else int(eval_windows),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.exists() else None,
        "raw_prediction_dir": str(work_dir / "outputs"),
        "config": str(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "commit_sha": git_head(root_dir),
        "eval_results": dict(eval_results or {}),
        "timestamp": time.time(),
    }
    atomic_publish_json(work_dir / receipt_name, receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description="BA-FDR K16 Full-Matrix Training Driver")
    parser.add_argument("config", type=str, help="path to cell config")
    parser.add_argument("--work-dir", type=str, default=None, help="work directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint path for evaluation")
    parser.add_argument("--eval-only", action="store_true", help="run evaluation only")
    parser.add_argument("--prediction-only", action="store_true", help="save raw predictions without opening metrics")
    parser.add_argument("--open-metrics", action="store_true", help="load sealed raw predictions and run the evaluator")
    parser.add_argument("--precheck-only", action="store_true", help="build and validate objects without training")
    parser.add_argument("--allow-single-process", action="store_true", help="allow local single-process smoke/precheck execution")
    parser.add_argument("--expected-world-size", type=int, default=EXPECTED_WORLD_SIZE, help="formal DDP world size")
    args = parser.parse_args()

    cfg_path = resolve_existing_path(args.config, repo_root=root_dir, label="config")
    cfg = Config.fromfile(str(cfg_path))
    work_dir = Path(args.work_dir or getattr(cfg, "work_dir", "exps/bafdr_tmp"))
    if not work_dir.is_absolute():
        work_dir = root_dir / work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg.work_dir = str(work_dir)
    ckpt_dir = work_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device, rank, world_size, local_rank = init_distributed(args.expected_world_size, args.allow_single_process)
    logger = setup_logger("BAFDR_Train", str(work_dir), distributed_rank=rank)
    logger.info(f"Loaded config: {cfg_path}")
    logger.info(f"Distributed context: rank={rank}, world_size={world_size}, local_rank={local_rank}, device={device}")

    bafdr_meta = getattr(cfg, "bafdr_protocol", {})
    if bafdr_meta.get("protocol") != PROTOCOL_ID:
        raise ValueError(f"BA-FDR protocol mismatch: {bafdr_meta.get('protocol')} != {PROTOCOL_ID}")
    arm = bafdr_meta.get("arm", "UNKNOWN")
    seed = int(bafdr_meta.get("seed", 4407))
    use_distill = bool(bafdr_meta.get("distillation", False))
    set_random_seed(seed + rank)

    model = build_detector(cfg.model).to(device)

    checkpoint_to_load = args.checkpoint
    if args.eval_only and checkpoint_to_load is None:
        checkpoint_to_load = str(ckpt_dir / f"epoch_{EXPECTED_EPOCHS - 1}.pth")
    loaded_ckpt = None
    loaded_ckpt_path = None
    if checkpoint_to_load is not None:
        ckpt_path = resolve_existing_path(checkpoint_to_load, repo_root=root_dir, label="checkpoint")
        loaded_ckpt_path = ckpt_path
        loaded_ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = loaded_ckpt.get("state_dict_ema", loaded_ckpt.get("state_dict", loaded_ckpt))
        load_state_dict_strict(model, state_dict, label=f"model checkpoint {ckpt_path}")
        logger.info(f"Loaded checkpoint: {ckpt_path}")

    if world_size > 1:
        ddp_static_graph = bool(getattr(cfg.solver, "static_graph", False))
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
            static_graph=ddp_static_graph,
        )

    use_amp = bool(getattr(cfg.solver, "amp", True)) and device.type == "cuda"
    ema = ModelEma(model) if bool(getattr(cfg.solver, "ema", True)) else None
    if loaded_ckpt is not None and ema is not None and loaded_ckpt.get("state_dict_ema") is not None:
        load_state_dict_strict(ema.module, loaded_ckpt["state_dict_ema"], label="EMA checkpoint")

    teacher = None
    if use_distill and not args.eval_only:
        teacher_cfg = bafdr_meta.get("teacher_config")
        teacher_ckpt = bafdr_meta.get("teacher_checkpoint")
        if not teacher_cfg or not teacher_ckpt:
            raise ValueError("BAFDR-K16-FULL requires teacher_config and teacher_checkpoint")
        teacher_cfg_path = resolve_existing_path(teacher_cfg, repo_root=root_dir, label="teacher config")
        teacher_ckpt_path = resolve_teacher_checkpoint(teacher_ckpt, repo_root=root_dir, work_dir=work_dir, seed=seed)
        logger.info(f"Loading frozen D160 teacher: cfg={teacher_cfg_path}, ckpt={teacher_ckpt_path}")
        teacher = build_teacher_model(teacher_cfg_path, teacher_ckpt_path, device)

    val_dataset, val_loader = build_eval_loader(cfg, rank, world_size, logger)

    if args.eval_only:
        if args.prediction_only and args.open_metrics:
            raise ValueError("--prediction-only and --open-metrics are mutually exclusive")
        if args.open_metrics:
            cfg.inference.load_from_raw_predictions = True
            cfg.inference.save_raw_prediction = False
        prediction_only = bool(args.prediction_only) and not args.open_metrics
        eval_results = run_eval(
            model=model,
            ema=ema,
            val_loader=val_loader,
            cfg=cfg,
            logger=logger,
            rank=rank,
            world_size=world_size,
            device=device,
            use_amp=use_amp,
            not_eval=prediction_only,
        )
        save_training_receipt(
            work_dir=work_dir,
            cfg_path=cfg_path,
            arm=arm,
            seed=seed,
            total_updates=int(loaded_ckpt.get("total_successful_updates", 0)) if isinstance(loaded_ckpt, Mapping) else 0,
            checkpoint_path=loaded_ckpt_path,
            eval_results=eval_results,
            rank=rank,
            world_size=world_size,
            eval_windows=len(val_dataset),
            receipt_name="prediction_receipt.json" if prediction_only else "eval_receipt.json",
            phase="prediction_seal" if prediction_only else "metric_opening",
            metric_opened=not prediction_only,
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return

    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    validate_dataset_lengths(train_dataset, val_dataset, logger)
    train_batch_size = int(getattr(cfg.solver.train, "batch_size", 2))
    validate_loader_batch_contract("train", train_batch_size, world_size)
    train_num_workers = int(getattr(cfg.solver.train, "num_workers", 2))
    train_loader = build_dataloader(
        train_dataset,
        batch_size=train_batch_size,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        drop_last=True,
        num_workers=train_num_workers,
    )
    if len(train_loader) < EXPECTED_UPDATES_PER_EPOCH:
        raise RuntimeError(
            f"Train loader has {len(train_loader)} local batches; expected at least {EXPECTED_UPDATES_PER_EPOCH}"
        )

    optimizer = build_optimizer(copy.deepcopy(cfg.optimizer), model, logger)
    scheduler, max_epoch = build_scheduler(copy.deepcopy(cfg.scheduler), optimizer, len(train_loader))
    if int(max_epoch) != EXPECTED_EPOCHS:
        raise RuntimeError(f"BA-FDR protocol requires {EXPECTED_EPOCHS} epochs; scheduler produced {max_epoch}")
    scaler = GradScaler(enabled=use_amp)

    if args.precheck_only:
        if is_rank0():
            atomic_publish_json(
                work_dir / "precheck_receipt.json",
                {
                    "schema_version": "ZOOMTOKEN-BA-FDR-K16-PRECHECK-v001",
                    "protocol_id": PROTOCOL_ID,
                    "arm": arm,
                    "seed": seed,
                    "rank_seed": seed + rank,
                    "world_size": world_size,
                    "train_len": len(train_dataset),
                    "eval_windows": len(val_dataset),
                    "teacher_loaded": teacher is not None,
                    "status": "PASS",
                    "timestamp": time.time(),
                },
            )
        return

    logger.info(f"Starting BA-FDR formal training for {arm} seed={seed}")
    total_successful_updates = 0
    workflow = getattr(cfg, "workflow", {})
    max_amp_retries_per_batch = int(getattr(workflow, "max_amp_retries_per_batch", 0))
    fail_on_skipped_update = bool(getattr(workflow, "fail_on_skipped_update", True))
    schedule_and_ema_on_success_only = bool(
        getattr(workflow, "schedule_and_ema_on_success_only", True)
    )
    fail_on_nonfinite_loss = bool(getattr(workflow, "fail_on_nonfinite_loss", True))
    for epoch in range(EXPECTED_EPOCHS):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
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
            use_amp=use_amp,
            max_amp_retries_per_batch=max_amp_retries_per_batch,
            fail_on_skipped_update=fail_on_skipped_update,
            schedule_and_ema_on_success_only=schedule_and_ema_on_success_only,
            fail_on_nonfinite_loss=fail_on_nonfinite_loss,
        )
        total_successful_updates += epoch_updates

        if is_rank0() and ((total_successful_updates % 500 == 0) or (epoch == EXPECTED_EPOCHS - 1)):
            ckpt_name = f"epoch_{epoch}.pth"
            checkpoint_path = ckpt_dir / ckpt_name
            atomic_save_checkpoint(
                checkpoint_path,
                {
                    "schema_version": "ZOOMTOKEN-BA-FDR-K16-CHECKPOINT-v001",
                    "protocol_id": PROTOCOL_ID,
                    "epoch": epoch,
                    "total_successful_updates": total_successful_updates,
                    "state_dict": model.state_dict(),
                    "state_dict_ema": ema.module.state_dict() if ema is not None else None,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "arm": arm,
                    "seed": seed,
                    "world_size": world_size,
                    "config_sha256": sha256_file(cfg_path),
                    "commit_sha": git_head(root_dir),
                },
            )
            logger.info(f"Saved checkpoint {checkpoint_path} at update {total_successful_updates}")

    if total_successful_updates != EXPECTED_TOTAL_UPDATES:
        raise RuntimeError(
            f"BA-FDR protocol requires {EXPECTED_TOTAL_UPDATES} updates; got {total_successful_updates}"
        )
    final_checkpoint = ckpt_dir / f"epoch_{EXPECTED_EPOCHS - 1}.pth"
    save_training_receipt(
        work_dir=work_dir,
        cfg_path=cfg_path,
        arm=arm,
        seed=seed,
        total_updates=total_successful_updates,
        checkpoint_path=final_checkpoint,
        eval_results=None,
        rank=rank,
        world_size=world_size,
        train_samples=len(train_dataset),
        eval_windows=len(val_dataset),
        receipt_name="train_receipt.json",
        phase="training",
        metric_opened=False,
    )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
