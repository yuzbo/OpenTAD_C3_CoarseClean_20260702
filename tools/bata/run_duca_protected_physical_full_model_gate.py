from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from mmengine.config import Config
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Subset

from opentad.cores import build_scheduler
from opentad.cores.optimizer import (
    assert_optimizer_exact_coverage,
    build_optimizer,
    prepare_optimizer_parameter_freezing,
)
from opentad.datasets import build_dataloader, build_dataset
from opentad.models import build_detector
from opentad.models.duca.structured_selection import exact_uniform_positions
from opentad.utils import ModelEma


SCHEMA = "duca_protected_physical_full_model_gate_v1"
CONTRACT = "duca_protected_e2e_physical_v1"
SUPPORTED_ARMS = {
    "protected_e2e",
    "protected_e2e_bridge025",
    "protected_e2e_uni_companion",
    "protected_e2e_rho001",
}


class ProtectedPhysicalGateFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtectedPhysicalGateFailure(
            f"protected physical full-model gate failed: {message}"
        )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def _bind_runtime(expected_commit: str) -> dict[str, Any]:
    expected = str(expected_commit).lower()
    _require(
        re.fullmatch(r"[0-9a-f]{40}", expected) is not None,
        "--expected-commit must be an exact commit",
    )
    _require(_git_output("rev-parse", "HEAD") == expected, "commit drift")
    _require(
        not _git_output("status", "--porcelain", "--untracked-files=normal"),
        "clean tree required",
    )
    _require(os.environ.get("SLURM_JOB_ID") is not None, "Slurm is required")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(torch.cuda.device_count() == 1, "one logical Slurm GPU is required")
    _require(int(os.environ.get("WORLD_SIZE", "0")) == 1, "WORLD_SIZE must be one")
    _require(int(os.environ.get("LOCAL_RANK", "-1")) == 0, "LOCAL_RANK must be zero")
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return {
        "git_commit": expected,
        "git_tree": _git_output("rev-parse", "HEAD^{tree}"),
        "slurm_job_id": str(os.environ["SLURM_JOB_ID"]),
        "world_size": int(dist.get_world_size()),
        "logical_device": "cuda:0",
        "device_name": torch.cuda.get_device_name(0),
    }


def _load_protocol_manifest(
    path: str,
    expected_sha256: str,
    *,
    expected_commit: str,
) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), "P0 protocol manifest is missing")
    _require(
        re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256)) is not None,
        "P0 protocol manifest SHA256 is invalid",
    )
    _require(
        _sha256(resolved) == str(expected_sha256),
        "P0 protocol manifest SHA256 drift",
    )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(
        isinstance(payload, Mapping)
        and payload.get("schema") == "duca_protected_physical_protocol_manifest_v1"
        and payload.get("ok") is True,
        "P0 protocol manifest did not pass",
    )
    _require(payload.get("git_commit") == expected_commit, "P0 commit drift")
    return dict(payload), resolved


def _validate_config(cfg: Config) -> str:
    arm = str(cfg.model.frame_selector.arm)
    _require(arm in SUPPORTED_ARMS, f"unsupported gate arm {arm!r}")
    _require(cfg.model.type == "ActionFormer", "detector must be ActionFormer")
    _require(
        cfg.model.rpn_head.type == "ActionFormerHead", "head must be ActionFormerHead"
    )
    _require(
        cfg.model.frame_selector.type == "DucaProtectedE2EFrameSelector",
        "selector type drift",
    )
    _require(int(cfg.model.frame_selector.budget) == 384, "budget must be 384")
    _require(
        int(cfg.model.frame_selector.dense_window_size) == 768,
        "dense window must be 768",
    )
    _require(
        cfg.model.rpn_head.physical_grid_actionformer.contract == CONTRACT,
        "physical head contract drift",
    )
    _require(
        bool(cfg.model.rpn_head.physical_grid_actionformer.enabled)
        and bool(cfg.model.rpn_head.physical_grid_actionformer.required)
        and bool(cfg.model.rpn_head.physical_grid_actionformer.strict),
        "physical head must be enabled/required/strict",
    )
    _require(
        cfg.duca_protected_physical_contract.backbone_tail_padding
        == "replicate_last_selected",
        "backbone tail-padding contract drift",
    )
    _require(int(cfg.workflow.end_epoch) == 60, "official protocol must be 60 epochs")
    _require(
        int(cfg.workflow.val_eval_interval) < 0, "training must seal test evaluation"
    )
    _require(
        bool(cfg.workflow.seal_eval_dataloaders_during_training),
        "training must not construct validation/test loaders",
    )
    _require(cfg.solver.static_graph is False, "static_graph must be false")
    _require(
        cfg.solver.find_unused_parameters is True,
        "find_unused_parameters must be true",
    )
    return arm


def _cuda_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        "inputs": batch["inputs"].to("cuda:0", non_blocking=True),
        "masks": batch["masks"].to("cuda:0", dtype=torch.bool, non_blocking=True),
        "metas": [dict(meta) for meta in batch["metas"]],
        "gt_segments": [
            value.to("cuda:0", non_blocking=True) for value in batch["gt_segments"]
        ],
        "gt_labels": [
            value.to("cuda:0", non_blocking=True) for value in batch["gt_labels"]
        ],
    }
    if "gt_boundary_validity" in batch:
        output["gt_boundary_validity"] = [
            value.to("cuda:0", dtype=torch.bool, non_blocking=True)
            for value in batch["gt_boundary_validity"]
        ]
    return output


def _concat_gate_batches(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "inputs": torch.cat([first["inputs"], second["inputs"]], dim=0),
        "masks": torch.cat([first["masks"], second["masks"]], dim=0),
        "metas": [*first["metas"], *second["metas"]],
        "gt_segments": [*first["gt_segments"], *second["gt_segments"]],
        "gt_labels": [*first["gt_labels"], *second["gt_labels"]],
        "gt_boundary_validity": [
            *first["gt_boundary_validity"],
            *second["gt_boundary_validity"],
        ],
    }


def _real_gate_batches(
    cfg: Config,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    dataset = build_dataset(
        copy.deepcopy(cfg.dataset.train),
        default_args={"logger": None},
    )
    loader_cfg = copy.deepcopy(cfg.solver.train)
    loader_cfg["batch_size"] = 1
    snippet_stride = int(dataset.snippet_stride)
    indexed_lengths = []
    for index, item in enumerate(dataset.data_list):
        video_info = item[1]
        candidate_len = int(
            math.ceil(float(video_info["frame"]) / float(snippet_stride))
        )
        indexed_lengths.append((index, candidate_len))
    descending = sorted(indexed_lengths, key=lambda row: (-row[1], row[0]))
    ascending = sorted(indexed_lengths, key=lambda row: (row[1], row[0]))
    full_indices = [index for index, length in descending if length >= 768][:24]
    padded_indices = [index for index, length in descending if 384 < length < 768][:24]
    short_padded_indices = [index for index, length in ascending if 0 < length < 384][
        :48
    ]
    candidate_indices = []
    for index in full_indices + padded_indices + short_padded_indices:
        if index not in candidate_indices:
            candidate_indices.append(index)
    _require(full_indices, "no likely full training video was found")
    _require(padded_indices, "no likely padded training video was found")
    _require(
        short_padded_indices,
        "no likely short padded training video was found",
    )
    loader = build_dataloader(
        Subset(dataset, candidate_indices),
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **loader_cfg,
    )
    batches: dict[str, dict[str, Any]] = {}
    evidence: dict[str, Any] = {
        "dataset_length": int(len(dataset)),
        "candidate_indices_scanned": list(candidate_indices),
    }
    for batch_index, raw_batch in enumerate(loader):
        batch = _cuda_batch(raw_batch)
        valid_len = int(batch["masks"][0].sum().item())
        if valid_len == 768:
            key = "full"
        elif valid_len > 384:
            key = "padded"
        else:
            key = "short_padded"
        if key in batches:
            continue
        meta = batch["metas"][0]
        _require("frame_inds" in meta, "train loader omitted frame_inds")
        _require("avg_fps" in meta, "train loader omitted avg_fps")
        _require(
            "gt_boundary_validity" in batch,
            "train loader omitted true-boundary validity",
        )
        batches[key] = batch
        evidence[key] = {
            "dataset_index": int(candidate_indices[batch_index]),
            "batch_index": int(batch_index),
            "valid_len": valid_len,
            "video_name": str(meta.get("video_name", "")),
        }
        if set(batches) == {"full", "padded", "short_padded"}:
            return batches, evidence
    raise ProtectedPhysicalGateFailure(
        "real THUMOS gate requires full, padded, and short-padded training windows"
    )


def _capture_mutable_state(model) -> dict[str, Any]:
    return {
        "buffers": {
            name: value.detach().clone() for name, value in model.named_buffers()
        },
        "module_training": {
            name: bool(module.training) for name, module in model.named_modules()
        },
        "custom_replay": {
            name: module.capture_amp_replay_state()
            for name, module in model.named_modules()
            if callable(getattr(module, "capture_amp_replay_state", None))
            and callable(getattr(module, "restore_amp_replay_state", None))
        },
        "cpu_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }


def _restore_mutable_state(model, state: Mapping[str, Any]) -> None:
    modules = dict(model.named_modules())
    _require(
        set(state["module_training"]) == set(modules),
        "model module set drifted",
    )
    for name, training in state["module_training"].items():
        modules[name].training = bool(training)
    current = dict(model.named_buffers())
    _require(set(state["buffers"]).issubset(current), "model buffer set drifted")
    with torch.no_grad():
        for name, value in state["buffers"].items():
            current[name].copy_(value)
    for name, snapshot in state["custom_replay"].items():
        _require(name in modules, f"custom replay module {name!r} disappeared")
        modules[name].restore_amp_replay_state(snapshot)
    torch.random.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])


def _is_action_head(name: str) -> bool:
    marker = ".official_temporal."
    if marker not in name:
        return False
    tail = name.split(marker, 1)[1]
    parts = tail.split(".")
    return tail.startswith("encoder.conv_out.") or (
        len(parts) >= 4 and parts[0] == "decoders" and parts[2] == "conv_out"
    )


def _parameter_group(model, name: str) -> str:
    selector = model.frame_selector
    probe_prefix = "frame_selector.raw_actionness_source.probe_module."
    scorer_prefix = "frame_selector.transition_scorer."
    layers = (
        selector.raw_actionness_source.probe_module.official_temporal.encoder.layers
    )
    last_prefix = probe_prefix + f"official_temporal.encoder.layers.{len(layers) - 1}."
    if name.startswith(scorer_prefix):
        return "selector"
    if _is_action_head(name):
        return "action_head"
    if name.startswith(last_prefix):
        return "asformer_last_encoder_layer"
    if name.startswith(probe_prefix):
        return "asformer_earlier_or_spatial"
    if name.startswith("frame_selector."):
        return "unexpected_selector"
    if name.startswith("backbone."):
        return "videomae_adapter"
    if name.startswith("projection."):
        return "projection"
    if name.startswith("neck."):
        return "neck"
    if name.startswith("rpn_head."):
        return "actionformer_head"
    return "unexpected_detector"


def _optimizer_group_ids(optimizer) -> dict[int, int]:
    mapping = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter in group["params"]:
            parameter_id = id(parameter)
            _require(
                parameter_id not in mapping,
                "a parameter appears in multiple optimizer groups",
            )
            mapping[parameter_id] = int(group_index)
    return mapping


def _gradient_report(model, optimizer_ids: Mapping[int, int]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    per_parameter = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        _require(id(parameter) in optimizer_ids, f"optimizer omitted {name}")
        group = _parameter_group(model, name)
        grad_l1 = 0.0
        grad_l2 = 0.0
        grad_max = 0.0
        finite = True
        grad_none = parameter.grad is None
        if parameter.grad is not None:
            finite = bool(torch.isfinite(parameter.grad).all().item())
            grad = parameter.grad.detach().float()
            grad_l1 = float(grad.abs().sum().item())
            grad_l2 = float(torch.linalg.vector_norm(grad).item())
            grad_max = float(grad.abs().max().item())
        _require(finite, f"non-finite gradient in {name}")
        per_parameter[name] = {
            "group": group,
            "parameter_count": int(parameter.numel()),
            "requires_grad": bool(parameter.requires_grad),
            "grad_none": grad_none,
            "grad_l1": grad_l1,
            "grad_l2": grad_l2,
            "grad_max": grad_max,
            "finite": finite,
            "optimizer_group_id": int(optimizer_ids[id(parameter)]),
        }
        summary = groups.setdefault(
            group,
            {
                "parameter_tensor_count": 0,
                "parameter_count": 0,
                "requires_grad_count": 0,
                "grad_none_count": 0,
                "active_parameter_count": 0,
                "grad_l1": 0.0,
                "grad_l2_squared": 0.0,
                "grad_max": 0.0,
                "finite": True,
                "optimizer_group_ids": set(),
            },
        )
        summary["parameter_tensor_count"] += 1
        summary["parameter_count"] += int(parameter.numel())
        summary["requires_grad_count"] += int(parameter.numel())
        summary["grad_none_count"] += int(grad_none)
        summary["active_parameter_count"] += int(grad_l1 > 0.0)
        summary["grad_l1"] += grad_l1
        summary["grad_l2_squared"] += grad_l2 * grad_l2
        summary["grad_max"] = max(float(summary["grad_max"]), grad_max)
        summary["finite"] = bool(summary["finite"] and finite)
        summary["optimizer_group_ids"].add(int(optimizer_ids[id(parameter)]))
    for summary in groups.values():
        summary["grad_l2"] = float(summary.pop("grad_l2_squared") ** 0.5)
        summary["optimizer_group_ids"] = sorted(summary["optimizer_group_ids"])
    _require(
        groups.get("unexpected_selector", {}).get("parameter_tensor_count", 0) == 0,
        "unexpected trainable selector parameter",
    )
    _require(
        groups.get("unexpected_detector", {}).get("parameter_tensor_count", 0) == 0,
        "unexpected trainable detector parameter",
    )
    return {"groups": groups, "per_parameter": per_parameter}


def _group_mass(report: Mapping[str, Any], group: str) -> float:
    return float(report["groups"].get(group, {}).get("grad_l1", 0.0))


def _scaled_backward(
    objective: torch.Tensor,
    *,
    model,
    optimizer,
    optimizer_ids,
) -> dict[str, Any]:
    _require(
        objective.ndim == 0 and bool(torch.isfinite(objective.detach()).item()),
        "objective is not a finite scalar",
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    scaler.scale(objective).backward()
    scaler.unscale_(optimizer)
    report = _gradient_report(model, optimizer_ids)
    report["objective"] = float(objective.detach().float().item())
    report["grad_scaler_scale"] = float(scaler.get_scale())
    optimizer.zero_grad(set_to_none=True)
    return report


def _hard_gather(inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    temporal_dim = 3 if inputs.ndim == 6 else 2
    active = positions >= 0
    effective_k = active.sum(dim=1)
    _require(
        bool(torch.all(effective_k > 0).item()),
        "hard gather requires one active slot per sample",
    )
    prefix = (
        torch.arange(active.shape[1], device=active.device)[None] < effective_k[:, None]
    )
    _require(
        torch.equal(active, prefix),
        "hard gather requires a contiguous active slot prefix",
    )
    last_active = positions.gather(1, (effective_k - 1)[:, None])
    safe = torch.where(active, positions, last_active.expand_as(positions))
    view = [safe.shape[0]] + [1] * (inputs.ndim - 1)
    view[temporal_dim] = safe.shape[1]
    expand = list(inputs.shape)
    expand[temporal_dim] = safe.shape[1]
    return torch.gather(
        inputs,
        temporal_dim,
        safe.view(view).expand(expand),
    )


def _perturb_unselected(
    inputs: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    temporal_dim = 3 if inputs.ndim == 6 else 2
    selected = torch.zeros(
        positions.shape[0],
        inputs.shape[temporal_dim],
        device=inputs.device,
        dtype=torch.bool,
    )
    selected.scatter_(1, positions.clamp_min(0), positions >= 0)
    view = [selected.shape[0]] + [1] * (inputs.ndim - 1)
    view[temporal_dim] = selected.shape[1]
    unselected = (~selected).view(view)
    perturbation = torch.randn_like(inputs) * 3.0 + 1.0
    return torch.where(unselected, inputs + perturbation, inputs)


def _hard_detector_loss(
    model,
    *,
    selected_inputs: torch.Tensor,
    selected_masks: torch.Tensor,
    metas,
    batch: Mapping[str, Any],
    mutable_state: Mapping[str, Any],
) -> float:
    _restore_mutable_state(model, mutable_state)
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
        cache_enabled=False,
    ):
        losses = model.forward_train(
            selected_inputs,
            selected_masks,
            copy.deepcopy(metas),
            batch["gt_segments"],
            batch["gt_labels"],
            _duca_skip_frame_selector=True,
        )
        objective = model._duca_detector_objective(losses)
    _require(bool(torch.isfinite(objective).item()), "hard replay loss is non-finite")
    return float(objective.detach().float().item())


def _exact_uniform_position_tensor(
    masks: torch.Tensor,
    *,
    budget: int = 384,
) -> torch.Tensor:
    positions = torch.full(
        (int(masks.shape[0]), int(budget)),
        -1,
        device=masks.device,
        dtype=torch.long,
    )
    for batch_index, row in enumerate(masks):
        valid_len = int(row.sum().item())
        effective_k = min(int(budget), valid_len)
        positions[batch_index, :effective_k] = exact_uniform_positions(
            valid_len,
            effective_k,
            device=masks.device,
        )
    return positions


def _remap_gt_to_selected_axis(
    gt_segments,
    positions: torch.Tensor,
    masks: torch.Tensor,
) -> list[torch.Tensor]:
    output = []
    for batch_index, segments in enumerate(gt_segments):
        valid_len = int(masks[batch_index].sum().item())
        active = positions[batch_index]
        active = active[active >= 0].detach().float()
        xp = torch.cat(
            (
                active,
                active.new_tensor([float(valid_len)]),
            )
        )
        fp = torch.arange(
            active.numel() + 1,
            device=active.device,
            dtype=torch.float32,
        )
        row = segments.detach().float().reshape(-1, 2)
        flat = row.reshape(-1).clamp(min=0.0, max=float(valid_len))
        right = torch.searchsorted(xp, flat, right=True).clamp(
            min=1,
            max=xp.numel() - 1,
        )
        left = right - 1
        denominator = (xp[right] - xp[left]).clamp(min=1.0e-6)
        mapped = fp[left] + ((flat - xp[left]) / denominator) * (fp[right] - fp[left])
        output.append(mapped.reshape_as(row))
    return output


def _set_physical_head_enabled(head, enabled: bool) -> dict[str, Any]:
    state = {
        "physical_grid_enabled": bool(head.physical_grid_enabled),
        "physical_grid_required": bool(head.physical_grid_required),
        "physical_grid_strict": bool(head.physical_grid_strict),
        "protected_physical_grid": bool(head.protected_physical_grid),
    }
    head.physical_grid_enabled = bool(enabled)
    head.physical_grid_required = bool(enabled)
    head.physical_grid_strict = bool(enabled)
    head.protected_physical_grid = bool(enabled)
    return state


def _restore_physical_head_state(head, state: Mapping[str, Any]) -> None:
    for key, value in state.items():
        setattr(head, key, value)


def _detector_loss_report(
    model,
    *,
    selected_inputs: torch.Tensor,
    selected_masks: torch.Tensor,
    metas,
    gt_segments,
    gt_labels,
    mutable_state: Mapping[str, Any],
) -> dict[str, float]:
    _restore_mutable_state(model, mutable_state)
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
        cache_enabled=False,
    ):
        losses = model.forward_train(
            selected_inputs,
            selected_masks,
            copy.deepcopy(metas),
            gt_segments,
            gt_labels,
            _duca_skip_frame_selector=True,
        )
        objective = model._duca_detector_objective(losses)
    report = {
        key: float(value.detach().float().item())
        for key, value in losses.items()
        if key.endswith("loss") and ("cls" in key or "reg" in key)
    }
    report["objective"] = float(objective.detach().float().item())
    _require(
        all(math.isfinite(value) for value in report.values()),
        "uniform parity produced non-finite detector losses",
    )
    return report


def _target_assignment_parity(
    head,
    *,
    base_points,
    base_masks,
    physical_metas,
    dense_masks: torch.Tensor,
    positions: torch.Tensor,
    physical_segments,
    legacy_segments,
    gt_labels,
) -> dict[str, Any]:
    physical_points, physical_masks = head._build_physical_points_and_masks(
        [point.clone() for point in base_points],
        [mask.clone() for mask in base_masks],
        metas=copy.deepcopy(physical_metas),
        train_mode=True,
    )
    legacy_points = [point.clone() for point in base_points]
    legacy_masks = [mask.clone() for mask in base_masks]
    physical_cls, physical_reg = head.prepare_targets(
        physical_points,
        physical_segments,
        gt_labels,
    )
    legacy_cls, legacy_reg = head.prepare_targets(
        legacy_points,
        legacy_segments,
        gt_labels,
    )
    physical_cls = torch.stack(physical_cls)
    legacy_cls = torch.stack(legacy_cls)
    physical_valid = torch.cat(physical_masks, dim=1).bool()
    legacy_valid = torch.cat(legacy_masks, dim=1).bool()
    physical_pos = (physical_cls.sum(dim=-1) > 0) & physical_valid
    legacy_pos = (legacy_cls.sum(dim=-1) > 0) & legacy_valid
    cls_error = float((physical_cls - legacy_cls).abs().max().detach().float().item())
    _require(
        torch.equal(physical_valid, legacy_valid),
        "exact-uniform physical and selected-axis valid masks disagree",
    )
    _require(
        cls_error <= 1.0e-6,
        "exact-uniform physical and selected-axis classification targets disagree",
    )
    _require(
        torch.equal(physical_pos, legacy_pos),
        "exact-uniform physical and selected-axis positive masks disagree",
    )
    positive_count = int(physical_pos.sum().item())
    _require(
        positive_count > 0,
        "exact-uniform target parity batch has no positive detector points",
    )

    split_sizes = [int(point.shape[-2]) for point in physical_points]
    physical_reg_levels = (
        torch.stack(physical_reg)
        .permute(0, 2, 1)
        .split(
            split_sizes,
            dim=-1,
        )
    )
    legacy_reg_levels = (
        torch.stack(legacy_reg)
        .permute(0, 2, 1)
        .split(
            split_sizes,
            dim=-1,
        )
    )
    physical_target_segments = head.get_refined_proposals(
        physical_points,
        physical_reg_levels,
    )
    legacy_target_segments = head.get_refined_proposals(
        legacy_points,
        legacy_reg_levels,
    )
    mapped_legacy_rows = []
    for batch_index, legacy_row in enumerate(legacy_target_segments):
        valid_len = int(dense_masks[batch_index].sum().item())
        active = positions[batch_index]
        active = active[active >= 0].to(dtype=legacy_row.dtype)
        mapped_legacy_rows.append(
            head._selected_axis_to_physical_axis(
                legacy_row,
                active,
                valid_len,
            )
        )
    mapped_legacy_targets = torch.stack(mapped_legacy_rows)
    regression_error = float(
        (physical_target_segments[physical_pos] - mapped_legacy_targets[legacy_pos])
        .abs()
        .max()
        .detach()
        .float()
        .item()
    )
    _require(
        regression_error <= 1.0e-4,
        "exact-uniform physical and selected-axis regression targets disagree",
    )
    return {
        "classification_targets_equal": True,
        "classification_target_max_abs_error": cls_error,
        "positive_masks_equal": True,
        "positive_count": positive_count,
        "physical_regression_targets_equal": True,
        "physical_regression_target_max_abs_error": regression_error,
    }


def _uniform_physical_legacy_parity(
    model,
    *,
    batch: Mapping[str, Any],
    mutable_state: Mapping[str, Any],
) -> dict[str, Any]:
    positions = _exact_uniform_position_tensor(batch["masks"])
    materialized = model.frame_selector.materialize_hard_positions(
        batch["inputs"],
        batch["masks"],
        batch["metas"],
        positions,
    )
    legacy_segments = _remap_gt_to_selected_axis(
        batch["gt_segments"],
        positions,
        batch["masks"],
    )
    physical_losses = _detector_loss_report(
        model,
        selected_inputs=materialized["inputs"],
        selected_masks=materialized["masks"],
        metas=materialized["metas"],
        gt_segments=batch["gt_segments"],
        gt_labels=batch["gt_labels"],
        mutable_state=mutable_state,
    )
    physical_debug = model.rpn_head.collect_debug_state()
    head_state = _set_physical_head_enabled(model.rpn_head, False)
    try:
        legacy_losses = _detector_loss_report(
            model,
            selected_inputs=materialized["inputs"],
            selected_masks=materialized["masks"],
            metas=batch["metas"],
            gt_segments=legacy_segments,
            gt_labels=batch["gt_labels"],
            mutable_state=mutable_state,
        )
    finally:
        _restore_physical_head_state(model.rpn_head, head_state)
    loss_errors = {
        key: abs(physical_losses[key] - legacy_losses[key]) for key in physical_losses
    }
    _require(
        all(
            error
            <= 1.0e-4
            * max(
                1.0,
                abs(physical_losses[key]),
                abs(legacy_losses[key]),
            )
            for key, error in loss_errors.items()
        ),
        "exact-uniform physical and selected-axis detector losses disagree",
    )

    _restore_mutable_state(model, mutable_state)
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
        cache_enabled=False,
    ):
        features = model.backbone(materialized["inputs"])
        features, feature_masks = model.pad_data(
            features,
            materialized["masks"],
        )
        features, feature_masks = model.projection(features, feature_masks)
        physical_metas = copy.deepcopy(materialized["metas"])
        if model.with_neck:
            features, feature_masks, physical_metas = model._call_neck_forward(
                features,
                feature_masks,
                metas=physical_metas,
            )
        base_points = model.rpn_head.prior_generator(features)
        target_assignment = _target_assignment_parity(
            model.rpn_head,
            base_points=base_points,
            base_masks=feature_masks,
            physical_metas=physical_metas,
            dense_masks=batch["masks"],
            positions=positions,
            physical_segments=batch["gt_segments"],
            legacy_segments=legacy_segments,
            gt_labels=batch["gt_labels"],
        )
        physical_proposals, physical_scores = model.rpn_head.forward_test(
            features,
            feature_masks,
            metas=physical_metas,
        )
        head_state = _set_physical_head_enabled(model.rpn_head, False)
        try:
            legacy_proposals, legacy_scores = model.rpn_head.forward_test(
                features,
                feature_masks,
                metas=copy.deepcopy(batch["metas"]),
            )
        finally:
            _restore_physical_head_state(model.rpn_head, head_state)

    proposal_errors = []
    score_errors = []
    for batch_index, (physical, legacy, physical_score, legacy_score) in enumerate(
        zip(
            physical_proposals,
            legacy_proposals,
            physical_scores,
            legacy_scores,
        )
    ):
        valid_len = int(batch["masks"][batch_index].sum().item())
        active = positions[batch_index]
        active = active[active >= 0].to(dtype=legacy.dtype)
        mapped = model.rpn_head._selected_axis_to_physical_axis(
            legacy,
            active,
            valid_len,
        )
        _require(
            mapped.shape == physical.shape,
            "uniform decode proposal shape mismatch",
        )
        _require(
            legacy_score.shape == physical_score.shape,
            "uniform decode score shape mismatch",
        )
        proposal_errors.append(
            float((mapped - physical).abs().max().float().item())
            if mapped.numel()
            else 0.0
        )
        score_errors.append(
            float((legacy_score - physical_score).abs().max().float().item())
            if legacy_score.numel()
            else 0.0
        )
    _require(
        max(proposal_errors, default=0.0) <= 1.0e-4,
        "exact-uniform physical and remapped selected-axis proposals disagree",
    )
    _require(
        max(score_errors, default=0.0) <= 1.0e-6,
        "exact-uniform physical and selected-axis scores disagree",
    )
    _restore_mutable_state(model, mutable_state)
    return {
        "positions": [int(value) for value in positions[0].detach().cpu().tolist()],
        "physical_losses": physical_losses,
        "legacy_selected_axis_losses": legacy_losses,
        "loss_abs_errors": loss_errors,
        "proposal_max_abs_error": max(proposal_errors, default=0.0),
        "score_max_abs_error": max(score_errors, default=0.0),
        "physical_head_debug": physical_debug,
        "target_assignment": target_assignment,
        "target_assignment_parity": True,
        "decode_parity": True,
        "target_and_decode_parity": True,
    }


def _padded_window_audit(
    model,
    ddp,
    *,
    batch: Mapping[str, Any],
    mutable_state: Mapping[str, Any],
) -> dict[str, Any]:
    _restore_mutable_state(model, mutable_state)
    captured: list[torch.Tensor] = []

    def capture_backbone_input(_module, args):
        captured.append(args[0].detach().clone())

    hook = model.backbone.register_forward_pre_hook(capture_backbone_input)
    try:
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
            cache_enabled=False,
        ):
            losses = ddp(
                batch["inputs"],
                batch["masks"],
                batch["metas"],
                gt_segments=batch["gt_segments"],
                gt_labels=batch["gt_labels"],
                gt_boundary_validity=batch["gt_boundary_validity"],
                return_loss=True,
            )
            objective = losses["cost"]
        _require(
            bool(torch.isfinite(objective).item()),
            "padded real-window objective is non-finite",
        )
        positions = model.frame_selector._last_selected_positions
        _require(positions is not None, "padded window omitted hard positions")
        valid_len = int(batch["masks"][0].sum().item())
        _require(
            valid_len < 384,
            "padded tail audit requires a real window shorter than K",
        )
        expected_k = min(384, valid_len)
        _require(
            int((positions[0] >= 0).sum().item()) == expected_k,
            "padded window violates K_eff=min(K,valid_len)",
        )
        expected_hard = _hard_gather(batch["inputs"], positions)
        _require(
            len(captured) == 1 and torch.equal(captured[0], expected_hard),
            "padded window backbone input is not exact hard gather",
        )
        temporal_dim = 3 if expected_hard.ndim == 6 else 2
        last_active = expected_hard.select(temporal_dim, expected_k - 1)
        inactive_tail = expected_hard.narrow(
            temporal_dim,
            expected_k,
            int(expected_hard.shape[temporal_dim]) - expected_k,
        )
        tail_view = list(last_active.shape)
        tail_view.insert(temporal_dim, 1)
        tail_expand = list(expected_hard.shape)
        tail_expand[temporal_dim] = int(inactive_tail.shape[temporal_dim])
        _require(
            torch.equal(
                inactive_tail,
                last_active.reshape(tail_view).expand(tail_expand),
            ),
            "padded backbone tail does not replicate the last selected frame",
        )
        debug = model.rpn_head.collect_debug_state()
        _require(
            int(debug.get("physical_grid_actionformer_selected_count", -1))
            == expected_k,
            "padded window physical head selected count mismatch",
        )
        return {
            "valid_len": valid_len,
            "effective_k": expected_k,
            "objective": float(objective.detach().float().item()),
            "hard_forward_equal": True,
            "tail_padding_mode": "replicate_last_selected",
            "tail_padding_reference_equal": True,
            "physical_head_debug": debug,
        }
    finally:
        hook.remove()
        _restore_mutable_state(model, mutable_state)


def _real_optimizer_step_audit(
    model,
    ddp,
    optimizer,
    *,
    cfg: Config,
    protocol: Mapping[str, Any],
    batches: Mapping[str, Mapping[str, Any]],
    reports: Mapping[str, Any],
    mutable_state: Mapping[str, Any],
) -> dict[str, Any]:
    loader_length = int(protocol["train_loader_contract"]["loader_length"])
    scheduler, _ = build_scheduler(
        copy.deepcopy(cfg.scheduler),
        optimizer,
        loader_length,
    )
    model_ema = ModelEma(ddp)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    best_by_group = {}
    for name, row in reports["total"]["per_parameter"].items():
        if float(row["grad_l1"]) <= 1.0e-12:
            continue
        current = best_by_group.get(row["group"])
        if current is None or float(row["grad_max"]) > float(current[1]["grad_max"]):
            best_by_group[row["group"]] = (name, row)
    named_parameters = dict(model.named_parameters())
    active_parameters = {
        name: named_parameters[name].detach().cpu().clone()
        for name, _row in best_by_group.values()
    }
    _require(active_parameters, "no active parameters for optimizer-step audit")
    ema_root = getattr(model_ema.module, "module", model_ema.module)
    ema_before = {
        name: dict(ema_root.named_parameters())[name].detach().cpu().clone()
        for name in active_parameters
    }
    initial_scheduler_epoch = int(scheduler.last_epoch)
    initial_scaler_scale = float(scaler.get_scale())
    _restore_mutable_state(model, mutable_state)
    successful_updates = 0
    attempts = 0
    objective_values = []
    scaler_history = []
    successful_batch_updates = []
    for batch_name in ("full", "padded", "short_padded"):
        _require(
            batch_name in batches,
            f"optimizer-step audit omitted {batch_name} batch",
        )
        batch = batches[batch_name]
        batch_succeeded = False
        batch_attempts = 0
        while not batch_succeeded and batch_attempts < 5:
            attempts += 1
            batch_attempts += 1
            attempt_state = _capture_mutable_state(model)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
                cache_enabled=False,
            ):
                losses = ddp(
                    batch["inputs"],
                    batch["masks"],
                    batch["metas"],
                    gt_segments=batch["gt_segments"],
                    gt_labels=batch["gt_labels"],
                    gt_boundary_validity=batch["gt_boundary_validity"],
                    return_loss=True,
                )
                objective = losses["cost"]
            _require(
                objective.ndim == 0 and bool(torch.isfinite(objective.detach()).item()),
                f"{batch_name} optimizer-step objective is non-finite",
            )
            scale_before = float(scaler.get_scale())
            scaler.scale(objective).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg.solver.clip_grad_norm),
            )
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            batch_succeeded = scale_after >= scale_before
            scaler_history.append(
                {
                    "attempt": attempts,
                    "batch": batch_name,
                    "before": scale_before,
                    "after": scale_after,
                    "optimizer_step_ran": batch_succeeded,
                }
            )
            if not batch_succeeded:
                _restore_mutable_state(model, attempt_state)
                continue
            schedule_summary = model.after_optimizer_step()
            _require(
                isinstance(schedule_summary, Mapping)
                and schedule_summary.get("updated") is False,
                "protected selector advanced a hidden schedule",
            )
            scheduler.step()
            model_ema.update(ddp)
            successful_updates += 1
            successful_batch_updates.append(batch_name)
            objective_values.append(
                {
                    "batch": batch_name,
                    "value": float(objective.detach().float().item()),
                }
            )
        _require(
            batch_succeeded,
            f"real AMP gate did not update on {batch_name} batch",
        )
    _require(
        successful_updates == 3,
        "real AMP gate did not complete three optimizer updates",
    )
    parameter_changes = {
        name: float(
            (dict(model.named_parameters())[name].detach().cpu() - before)
            .abs()
            .max()
            .item()
        )
        for name, before in active_parameters.items()
    }
    ema_changes = {
        name: float(
            (dict(ema_root.named_parameters())[name].detach().cpu() - before)
            .abs()
            .max()
            .item()
        )
        for name, before in ema_before.items()
    }
    _require(
        max(parameter_changes.values(), default=0.0) > 0.0,
        "three real optimizer steps changed no audited parameter",
    )
    _require(
        max(ema_changes.values(), default=0.0) > 0.0,
        "three real optimizer steps changed no audited EMA parameter",
    )
    _require(bool(optimizer.state), "real optimizer has no state after updates")
    _require(
        int(scheduler.last_epoch) == initial_scheduler_epoch + 3,
        "scheduler did not advance exactly once per successful update",
    )
    return {
        "attempts": attempts,
        "successful_optimizer_updates": successful_updates,
        "successful_batch_updates": successful_batch_updates,
        "full_batch_update": "full" in successful_batch_updates,
        "padded_batch_update": "padded" in successful_batch_updates,
        "short_padded_batch_update": "short_padded" in successful_batch_updates,
        "objective_values": objective_values,
        "initial_scheduler_last_epoch": initial_scheduler_epoch,
        "final_scheduler_last_epoch": int(scheduler.last_epoch),
        "initial_grad_scaler_scale": initial_scaler_scale,
        "final_grad_scaler_scale": float(scaler.get_scale()),
        "scaler_history": scaler_history,
        "optimizer_state_parameter_count": int(len(optimizer.state)),
        "audited_parameter_max_abs_changes": parameter_changes,
        "audited_ema_parameter_max_abs_changes": ema_changes,
        "selector_schedule_step": 0,
        "scheduler_and_ema_updated": True,
    }


def _assert_gradient_ownership(arm: str, reports: Mapping[str, Any]) -> None:
    action = reports["action"]
    transition = reports["transition"]
    detector = reports["detector"]
    total = reports["total"]

    detector_groups = (
        "videomae_adapter",
        "projection",
        "neck",
        "actionformer_head",
    )
    _require(
        all(_group_mass(action, group) <= 1.0e-12 for group in detector_groups),
        "action loss reached detector",
    )
    _require(
        _group_mass(action, "selector") <= 1.0e-12,
        "action loss reached selector",
    )
    _require(
        _group_mass(action, "action_head") > 1.0e-12,
        "action loss missed action head",
    )
    _require(
        _group_mass(action, "asformer_last_encoder_layer") > 1.0e-12,
        "action loss missed ASFormer last layer",
    )
    _require(
        _group_mass(action, "asformer_earlier_or_spatial") > 1.0e-12,
        "action loss missed ASFormer trunk/stem",
    )

    _require(
        all(_group_mass(transition, group) <= 1.0e-12 for group in detector_groups),
        "transition loss reached detector",
    )
    _require(
        _group_mass(transition, "action_head") <= 1.0e-12,
        "transition loss reached action head",
    )
    _require(
        _group_mass(transition, "selector") > 1.0e-12,
        "transition loss missed selector",
    )
    _require(
        _group_mass(transition, "asformer_last_encoder_layer") > 1.0e-12,
        "transition loss missed ASFormer last layer",
    )
    _require(
        _group_mass(transition, "asformer_earlier_or_spatial") > 1.0e-12,
        "transition loss missed ASFormer trunk/stem",
    )

    for group in detector_groups:
        _require(
            _group_mass(detector, group) > 1.0e-12,
            f"detector loss missed {group}",
        )
    _require(
        _group_mass(detector, "selector") > 1.0e-12,
        "detector loss missed selector",
    )
    _require(
        _group_mass(detector, "action_head") <= 1.0e-12,
        "detector loss reached action head",
    )
    if arm != "protected_e2e_rho001":
        _require(
            _group_mass(detector, "asformer_last_encoder_layer") <= 1.0e-12
            and _group_mass(detector, "asformer_earlier_or_spatial") <= 1.0e-12,
            "main detector loss leaked into coarse ASFormer",
        )
    else:
        _require(
            _group_mass(detector, "asformer_last_encoder_layer") > 1.0e-12,
            "rho detector loss missed the last ASFormer layer",
        )
        _require(
            _group_mass(detector, "asformer_earlier_or_spatial") <= 1.0e-12,
            "rho detector loss leaked before the last ASFormer layer",
        )

    for group in (
        *detector_groups,
        "selector",
        "action_head",
        "asformer_last_encoder_layer",
        "asformer_earlier_or_spatial",
    ):
        _require(
            _group_mass(total, group) > 1.0e-12,
            f"total loss missed {group}",
        )


def run_gate(
    *,
    config_path: str,
    expected_commit: str,
    protocol_manifest: str,
    protocol_manifest_sha256: str,
    adatad_pretrain: str,
    adatad_pretrain_sha256: str,
    output_json: str,
) -> dict[str, Any]:
    output = Path(output_json).expanduser().resolve()
    _require(not output.exists(), "refusing to overwrite gate evidence")
    try:
        output.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ProtectedPhysicalGateFailure(
            "gate evidence must be outside the Git worktree"
        )

    runtime = _bind_runtime(expected_commit)
    protocol, protocol_path = _load_protocol_manifest(
        protocol_manifest,
        protocol_manifest_sha256,
        expected_commit=expected_commit,
    )
    _require(
        protocol.get("git_tree") == runtime["git_tree"],
        "P0 Git tree differs from the gate tree",
    )
    cfg_path = (ROOT / config_path).resolve()
    cfg = Config.fromfile(str(cfg_path))
    arm = _validate_config(cfg)
    protocol_arm = protocol.get("configs", {}).get("arms", {}).get(arm)
    _require(
        isinstance(protocol_arm, Mapping),
        f"P0 manifest has no config evidence for {arm}",
    )
    _require(
        protocol_arm.get("source_sha256") == _sha256(cfg_path),
        f"{arm} config source differs from P0",
    )
    pretrain = Path(adatad_pretrain).expanduser().resolve()
    _require(pretrain.is_file(), "VideoMAE-S pretrain is missing")
    actual_pretrain_sha = _sha256(pretrain)
    _require(
        actual_pretrain_sha == str(adatad_pretrain_sha256).lower(),
        "VideoMAE-S pretrain SHA256 mismatch",
    )
    _require(
        protocol.get("videomae_pretrain", {}).get("sha256") == actual_pretrain_sha,
        "VideoMAE-S pretrain differs from P0",
    )

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.backbone.custom.pretrain = str(pretrain)
    model = build_detector(model_cfg)
    _require(model.__class__.__name__ == "ActionFormer", "wrong detector class")
    _require(
        model.rpn_head.__class__.__name__ == "ActionFormerHead",
        "wrong detector head class",
    )
    _require(model.frame_selector.arm == arm, "built selector arm drift")
    logger = logging.getLogger("duca-protected-physical-gate")
    prepare_optimizer_parameter_freezing(
        copy.deepcopy(cfg.optimizer),
        model,
        logger,
    )
    model.to("cuda:0")
    optimizer = build_optimizer(
        copy.deepcopy(cfg.optimizer),
        SimpleNamespace(module=model),
        logger,
    )
    assert_optimizer_exact_coverage(model, optimizer)
    optimizer_ids = _optimizer_group_ids(optimizer)
    _require(
        len(optimizer_ids)
        == sum(1 for parameter in model.parameters() if parameter.requires_grad),
        "optimizer coverage count mismatch",
    )
    model.train()
    ddp = DistributedDataParallel(
        model,
        device_ids=[0],
        output_device=0,
        find_unused_parameters=True,
        static_graph=False,
    )
    gate_batches, loader_evidence = _real_gate_batches(cfg)
    full_batch = gate_batches["full"]
    partial_padded_batch = gate_batches["padded"]
    short_padded_batch = gate_batches["short_padded"]
    batch = (
        _concat_gate_batches(full_batch, partial_padded_batch)
        if arm == "protected_e2e_uni_companion"
        else full_batch
    )
    loader_evidence["gradient_batch_size"] = int(batch["inputs"].shape[0])
    mutable_state = _capture_mutable_state(model)
    reports: dict[str, Any] = {}

    for route in ("action", "transition"):
        _restore_mutable_state(model, mutable_state)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
            cache_enabled=False,
        ):
            outputs = model.frame_selector.forward_train(
                inputs=batch["inputs"],
                masks=batch["masks"],
                metas=batch["metas"],
                gt_segments=batch["gt_segments"],
                gt_labels=batch["gt_labels"],
                gt_boundary_validity=batch["gt_boundary_validity"],
            )
            losses = outputs["losses"]
            if route == "action":
                objective = losses["selector_action_loss"]
            else:
                objective = (
                    losses["selector_transition_loss"]
                    + losses["selector_transition_boundary_loss"]
                )
        reports[route] = _scaled_backward(
            objective,
            model=model,
            optimizer=optimizer,
            optimizer_ids=optimizer_ids,
        )

    captured_backbone_inputs: list[torch.Tensor] = []

    def capture_backbone_input(_module, args):
        captured_backbone_inputs.append(args[0].detach().clone())

    hook = model.backbone.register_forward_pre_hook(capture_backbone_input)
    try:
        for route in ("detector", "total"):
            _restore_mutable_state(model, mutable_state)
            optimizer.zero_grad(set_to_none=True)
            captured_backbone_inputs.clear()
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
                cache_enabled=False,
            ):
                losses = ddp(
                    batch["inputs"],
                    batch["masks"],
                    batch["metas"],
                    gt_segments=batch["gt_segments"],
                    gt_labels=batch["gt_labels"],
                    gt_boundary_validity=batch["gt_boundary_validity"],
                    return_loss=True,
                )
                objective = (
                    model._duca_detector_objective(losses)
                    if route == "detector"
                    else losses["cost"]
                )
            _require(
                len(captured_backbone_inputs) == 1,
                f"{route} did not call the real backbone exactly once",
            )
            positions = model.frame_selector._last_selected_positions
            _require(positions is not None, "selector did not expose hard positions")
            expected_hard = _hard_gather(batch["inputs"], positions)
            _require(
                torch.equal(captured_backbone_inputs[0], expected_hard),
                f"{route} backbone input is not exact hard gather",
            )
            reports[route] = _scaled_backward(
                objective,
                model=model,
                optimizer=optimizer,
                optimizer_ids=optimizer_ids,
            )
    finally:
        hook.remove()

    _assert_gradient_ownership(arm, reports)
    companion_summary = copy.deepcopy(model.frame_selector.last_forward_summary)
    if arm == "protected_e2e_uni_companion":
        _require(
            companion_summary.get("uniform_companion_count") == 1,
            "Uni companion gate did not route exactly one row through uniform",
        )
        _require(
            companion_summary.get("learned_detector_count") == 1,
            "Uni companion gate did not retain exactly one learned row",
        )
    positions = model.frame_selector._last_selected_positions
    physical_metas = model.frame_selector._last_physical_metas
    _require(positions is not None, "missing hard positions after total route")
    _require(physical_metas is not None, "missing physical metadata after total route")
    selected_masks = positions >= 0
    hard_original = _hard_gather(batch["inputs"], positions)
    perturbed_dense = _perturb_unselected(batch["inputs"], positions)
    hard_perturbed = _hard_gather(perturbed_dense, positions)
    _require(
        torch.equal(hard_original, hard_perturbed),
        "unselected-frame perturbation changed fixed hard gather",
    )
    hard_loss_original = _hard_detector_loss(
        model,
        selected_inputs=hard_original,
        selected_masks=selected_masks,
        metas=physical_metas,
        batch=batch,
        mutable_state=mutable_state,
    )
    hard_loss_perturbed = _hard_detector_loss(
        model,
        selected_inputs=hard_perturbed,
        selected_masks=selected_masks,
        metas=physical_metas,
        batch=batch,
        mutable_state=mutable_state,
    )
    _require(
        abs(hard_loss_original - hard_loss_perturbed) <= 1.0e-6,
        "fixed-selection detector output changed after unselected perturbation",
    )
    head_debug = model.rpn_head.collect_debug_state()
    _require(
        head_debug.get("physical_grid_actionformer_enabled") is True,
        "physical head path did not execute",
    )
    _require(
        int(head_debug.get("physical_grid_actionformer_selected_count", -1)) == 384,
        "physical head did not consume exact K=384",
    )
    full_uniform_parity = _uniform_physical_legacy_parity(
        model,
        batch=batch,
        mutable_state=mutable_state,
    )
    padded_uniform_parity = _uniform_physical_legacy_parity(
        model,
        batch=short_padded_batch,
        mutable_state=mutable_state,
    )
    uniform_parity = {
        "full_window": full_uniform_parity,
        "short_padded_window": padded_uniform_parity,
        "target_assignment_parity": True,
        "decode_parity": True,
        "target_and_decode_parity": True,
    }
    padded_audit = _padded_window_audit(
        model,
        ddp,
        batch=short_padded_batch,
        mutable_state=mutable_state,
    )
    optimizer_step_audit = _real_optimizer_step_audit(
        model,
        ddp,
        optimizer,
        cfg=cfg,
        protocol=protocol,
        batches={
            "full": full_batch,
            "padded": partial_padded_batch,
            "short_padded": short_padded_batch,
        },
        reports=reports,
        mutable_state=mutable_state,
    )
    payload = {
        "schema": SCHEMA,
        "ok": True,
        "status": "p1_p2_full_model_gate_passed",
        "runtime": runtime,
        "protocol_manifest": {
            "path": str(protocol_path),
            "sha256": str(protocol_manifest_sha256),
            "content_sha256": protocol.get("manifest_content_sha256"),
        },
        "config": {
            "path": str(cfg_path),
            "sha256": _sha256(cfg_path),
            "arm": arm,
            "contract": CONTRACT,
        },
        "adatad_pretrain": {
            "path": str(pretrain),
            "sha256": actual_pretrain_sha,
        },
        "loader_evidence": loader_evidence,
        "training_companion_audit": {
            "training_only": arm == "protected_e2e_uni_companion",
            "detector_forward_count": 1,
            "uniform_companion_count": int(
                companion_summary.get("uniform_companion_count", 0)
            ),
            "learned_detector_count": int(
                companion_summary.get("learned_detector_count", 0)
            ),
            "detector_bridge_gradient_scale": float(
                companion_summary.get("detector_bridge_gradient_scale", 0.0)
            ),
        },
        "hard_forward_equals_real_backbone_input": True,
        "unselected_perturbation_audit": {
            "hard_gather_equal": True,
            "base_detector_loss": hard_loss_original,
            "perturbed_detector_loss": hard_loss_perturbed,
            "max_abs_error": abs(hard_loss_original - hard_loss_perturbed),
        },
        "physical_head_debug": head_debug,
        "exact_uniform_physical_legacy_parity": uniform_parity,
        "padded_real_window_audit": padded_audit,
        "real_optimizer_step_audit": optimizer_step_audit,
        "optimizer_exact_coverage": True,
        "gradient_ownership": reports,
        "numeric_contract": {
            "amp": True,
            "ddp": True,
            "find_unused_parameters": True,
            "static_graph": False,
            "real_optimizer_step": True,
            "scheduler_step": True,
            "ema_update": True,
            "full_partial_and_short_padded_real_windows": True,
            "short_padded_backward_update": True,
            "backbone_tail_padding_mode": "replicate_last_selected",
        },
        "paper_claim_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--protocol-manifest-sha256", required=True)
    parser.add_argument("--adatad-pretrain", required=True)
    parser.add_argument("--adatad-pretrain-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    try:
        run_gate(
            config_path=args.config,
            expected_commit=args.expected_commit,
            protocol_manifest=args.protocol_manifest,
            protocol_manifest_sha256=args.protocol_manifest_sha256,
            adatad_pretrain=args.adatad_pretrain,
            adatad_pretrain_sha256=args.adatad_pretrain_sha256,
            output_json=args.output_json,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "ok": False,
                    "status": "p1_p2_full_model_gate_failed",
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
