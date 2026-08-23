"""Fail-closed, read-only admission checks for H65 First-Mixing SingleClock."""

import argparse
import copy
import hashlib
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from mmengine.config import Config

import opentad.datasets  # noqa: F401 - registers OpenTAD data transforms.
from opentad.cores.optimizer import build_optimizer, prepare_optimizer_parameter_freezing
from opentad.models.builder import build_detector
from opentad.models.detectors.actionformer import ActionFormer
from opentad.models.duca.structured_selection import exact_uniform_positions


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    "STAGE1": ROOT / "configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py",
    "STAGE2_OFF": ROOT / "configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py",
    "STAGE2_ON": ROOT / "configs/adatad/thumos/duca_h65_first_singleclock_cycle4.py",
}


class _ModuleWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


def _get(obj, key, default=None):
    return obj.get(key, default) if hasattr(obj, "get") else getattr(obj, key, default)


def global_rank_clip_slices(batch, clips=24, clip_len=16, tubelet=2):
    positions = exact_uniform_positions(clips * clip_len * tubelet, clips * clip_len)
    return positions.repeat(batch, 1).reshape(batch, clips, -1).reshape(batch * clips, -1)


def _single_clock_gradient_probe(model, optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    theta = model.backbone.model.backbone.blocks[0].relative_physical_time_scale
    theta.data.zero_()

    generator = torch.Generator(device=device)
    generator.manual_seed(3407)
    frames = torch.randn(
        (1, 1, 3, 384, 16, 16),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    masks = torch.ones((1, 384), device=device, dtype=torch.bool)
    uniform = exact_uniform_positions(768, 384, device=device)

    def metadata(positions):
        return [
            {
                "irregular_selected_positions": positions.detach().cpu().tolist(),
                "irregular_selected_valid_len": 384,
                "irregular_dense_valid_len": 768,
            }
        ]

    optimizer.zero_grad(set_to_none=True)
    uniform_output = model._forward_backbone_with_single_clock(
        frames,
        masks,
        metadata(uniform),
    )
    uniform_output.float().square().mean().backward()
    uniform_grad = theta.grad
    if uniform_grad is not None and float(uniform_grad.detach().abs().cpu().item()) != 0.0:
        raise SystemExit("exact-uniform identity path must not create a SingleClock gradient")

    irregular = uniform.clone()
    irregular[1] = 1
    if not bool((irregular[1:] > irregular[:-1]).all()):
        raise SystemExit("deterministic irregular gradient probe must remain strictly increasing")
    optimizer.zero_grad(set_to_none=True)
    irregular_output = model._forward_backbone_with_single_clock(
        frames,
        masks,
        metadata(irregular),
    )
    irregular_output.float().square().mean().backward()
    if theta.grad is None or not bool(torch.isfinite(theta.grad).all()):
        raise SystemExit("irregular SingleClock theta gradient must be finite")
    gradient = float(theta.grad.detach().cpu().item())
    if gradient == 0.0:
        raise SystemExit("irregular SingleClock theta gradient must be non-zero")
    before = float(theta.detach().cpu().item())
    optimizer.step()
    after = float(theta.detach().cpu().item())
    if not torch.isfinite(theta).all() or after == before:
        raise SystemExit("irregular SingleClock optimizer step must update theta")

    model.single_clock_gate_zero = True
    with torch.no_grad():
        gate_zero_output = model._forward_backbone_with_single_clock(
            frames,
            masks,
            metadata(irregular),
        )
    model.single_clock_gate_zero = False
    theta.data.zero_()
    with torch.no_grad():
        theta_zero_output = model._forward_backbone_with_single_clock(
            frames,
            masks,
            metadata(irregular),
        )
    torch.testing.assert_close(gate_zero_output, theta_zero_output, rtol=0.0, atol=0.0)
    print(
        "SINGLE_CLOCK_GRADIENT_PROBE "
        + json.dumps(
            {
                "device": str(device),
                "uniform_residual_identity": True,
                "irregular_theta_gradient": gradient,
                "theta_before": before,
                "theta_after": after,
                "gate_zero_same_kernel_identity": True,
                "selected_rgb_unchanged": True,
            },
            sort_keys=True,
        )
    )


def _validate_resolved_single_clock(cfg):
    build_cfg = copy.deepcopy(cfg.model)
    build_cfg.backbone.custom.pretrain = None
    model = build_detector(build_cfg)
    if not isinstance(model, ActionFormer):
        raise SystemExit("resolved detector must be ActionFormer")
    if model.single_clock_admission is not True or model.single_clock_gate_zero is not False:
        raise SystemExit("resolved ActionFormer SingleClock admission/gate contract failed")
    vit = model.backbone.model.backbone
    scales = [block.relative_physical_time_scale for block in vit.blocks]
    if scales[0] is None or any(scale is not None for scale in scales[1:]):
        raise SystemExit("only VideoMAE block0 may own the SingleClock scalar")
    if scales[0].dtype != torch.float32 or float(scales[0].detach().cpu().item()) != 0.0:
        raise SystemExit("SingleClock scalar must be FP32 and initialized to zero")

    logger = logging.getLogger("duca_single_clock_validator")
    optimizer_cfg = copy.deepcopy(cfg.optimizer)
    prepare_optimizer_parameter_freezing(copy.deepcopy(optimizer_cfg), model, logger)
    wrapper = _ModuleWrapper(model)
    optimizer = build_optimizer(copy.deepcopy(optimizer_cfg), wrapper, logger)
    theta = scales[0]
    matching_groups = [
        group for group in optimizer.param_groups if any(param is theta for param in group["params"])
    ]
    if len(matching_groups) != 1:
        raise SystemExit("SingleClock scalar must occur in exactly one optimizer group")
    group = matching_groups[0]
    if float(group["lr"]) != 2e-4 or float(group["weight_decay"]) != 0.0:
        raise SystemExit("SingleClock scalar optimizer contract must be lr=2e-4, weight_decay=0")
    _single_clock_gradient_probe(model, optimizer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=CONFIGS, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--category", type=Path, required=True)
    parser.add_argument("--pretrain", type=Path, required=True)
    parser.add_argument("--stage1", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--chunk-dim", type=int, default=0)
    args = parser.parse_args()

    for path in (args.annotation, args.category, args.pretrain):
        if not path.is_file():
            raise SystemExit(f"canonical resource unreadable: {path}")
    entries = list(args.video_root.iterdir())
    if len(entries) != 411 or any(not item.is_symlink() or not item.exists() for item in entries):
        raise SystemExit("canonical video root must contain 411 valid symlinks")

    if args.target != "STAGE1":
        if not args.stage1 or not args.sha256 or args.epoch != 29 or len(args.sha256) != 64:
            raise SystemExit("Stage2 requires epoch29 Stage1 checkpoint and SHA")
        actual = hashlib.sha256(args.stage1.read_bytes()).hexdigest()
        if actual != args.sha256.lower():
            raise SystemExit("Stage1 checkpoint sha256 mismatch")
        checkpoint = torch.load(args.stage1, map_location="cpu", weights_only=False)
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("epoch") != 29
            or not isinstance(checkpoint.get("state_dict_ema"), dict)
        ):
            raise SystemExit("Stage1 checkpoint must contain epoch=29 state_dict_ema")
        os.environ.update(
            DUCA_STAGE1_CHECKPOINT=str(args.stage1.resolve()),
            DUCA_STAGE1_CHECKPOINT_SHA256=actual,
            DUCA_STAGE1_CHECKPOINT_EPOCH="29",
        )

    cfg = Config.fromfile(str(CONFIGS[args.target]))
    if args.target != "STAGE1" and (
        _get(cfg, "seed"),
        _get(cfg, "total_epochs"),
        _get(cfg, "max_updates"),
    ) != (3407, 60, 6000):
        raise SystemExit("training contract failed")
    if args.target == "STAGE2_ON":
        _validate_resolved_single_clock(cfg)
    if args.chunk_dim != 0 or global_rank_clip_slices(2).shape != (48, 16):
        raise SystemExit("chunk/global helper contract failed")
    print(
        f"PASS H65 First-Mixing SingleClock {args.target}: "
        "resources=411 valid_symlinks chunk_dim=0"
    )


if __name__ == "__main__":
    main()
