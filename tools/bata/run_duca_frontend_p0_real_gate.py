from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
from mmengine.config import Config
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as comm_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Subset

from opentad.cores import build_scheduler, train_one_epoch
from opentad.cores.optimizer import (
    assert_optimizer_exact_coverage,
    build_optimizer,
    prepare_optimizer_parameter_freezing,
)
from opentad.datasets import build_dataloader, build_dataset
from opentad.duca_loss_contract import DUCA_LOSS_TO_WEIGHT_KEY
from opentad.models import build_detector
from opentad.utils import ModelEma
from tools.bata.validate_duca_frontend_p0_contract import validate_config


SCHEMA = "duca_frontend_p0_real_cuda_gate_v1"
DEFAULT_CONFIG = "configs/adatad/thumos/duca_frontend_pretrain_a1_t010_b16.py"
DEFAULT_VARIANT_CONFIGS = (
    "configs/adatad/thumos/duca_frontend_pretrain_a1_t005_b8.py",
    "configs/adatad/thumos/duca_frontend_pretrain_a1_t010_b16.py",
    "configs/adatad/thumos/duca_frontend_pretrain_a1_t020_b32.py",
)
ACTIVE_LOSS_NAMES = {
    "actionness_bce_loss",
    "transition_distribution_loss",
    "transition_boundary_coverage_loss",
}
AUDITED_PATHS = (
    "configs/adatad/thumos/duca_frontend_pretrain_fixed384_base.py",
    "opentad/duca_loss_contract.py",
    "opentad/cores/optimizer.py",
    "opentad/cores/train_engine.py",
    "opentad/models/detectors/actionformer.py",
    "opentad/models/duca/acquisition.py",
    "opentad/models/duca/transition_only.py",
    "opentad/models/selectors/duca_online_frame_selector.py",
    "tools/bata/run_duca_frontend_p0_real_gate.py",
    "tools/bata/train_lowres_action_probe.py",
    "tools/bata/validate_duca_frontend_p0_contract.py",
)


class P0RealGateFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P0RealGateFailure(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _clean_commit(expected_commit: str) -> dict[str, Any]:
    observed = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=normal")
    _require(observed == expected_commit, "gate checkout does not match expected commit")
    _require(not status, "real P0 gate requires a clean exact-commit checkout")
    return {
        "git_commit": observed,
        "git_tree": _git("rev-parse", "HEAD^{tree}"),
        "git_tree_clean": True,
    }


def _outside_worktree(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise P0RealGateFailure("gate evidence must be written outside the source worktree")
    _require(not resolved.exists(), f"refusing to overwrite gate evidence: {resolved}")
    return resolved


def _audited_hashes() -> dict[str, str]:
    missing = [path for path in AUDITED_PATHS if not (ROOT / path).is_file()]
    _require(not missing, f"audited implementation files are missing: {missing}")
    return {path: _sha256(ROOT / path) for path in AUDITED_PATHS}


def _slurm_cuda_binding() -> dict[str, Any]:
    job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    _require(job_id.isdigit(), "real P0 gate requires a Slurm allocation")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(torch.cuda.device_count() == 1, "exactly one Slurm-visible GPU is required")
    _require(int(os.environ.get("WORLD_SIZE", "-1")) == 1, "gate requires world size one")
    _require(int(os.environ.get("LOCAL_RANK", "-1")) == 0, "gate requires local rank zero")
    return {
        "slurm_job_id": job_id,
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
        "logical_device": "cuda:0",
        "logical_cuda_device_count": 1,
        "physical_gpu_index_assumed": False,
    }


def _logger() -> logging.Logger:
    logger = logging.getLogger("duca-frontend-p0-real-gate")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _cuda_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    converted = dict(batch)
    converted["inputs"] = batch["inputs"].to(device, non_blocking=True)
    converted["masks"] = batch["masks"].to(device, non_blocking=True).bool()
    for key in ("gt_segments", "gt_labels"):
        converted[key] = [value.to(device, non_blocking=True) for value in batch[key]]
    return converted


def _full_window_indices(dataset, *, dense_window_size: int, batch_size: int) -> list[int]:
    _require(hasattr(dataset, "data_list"), "real train dataset lacks annotation index")
    snippet_stride = int(getattr(dataset, "snippet_stride", 0))
    _require(snippet_stride > 0, "real train dataset has invalid snippet stride")
    indices = []
    for index, row in enumerate(dataset.data_list):
        frame_count = int(row[1].get("frame", 0))
        if math.ceil(frame_count / snippet_stride) >= dense_window_size:
            indices.append(index)
        if len(indices) == batch_size:
            break
    _require(len(indices) == batch_size, "could not form one real full-window P0 batch")
    return indices


def _real_batch(dataset, cfg: Config, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    batch_size = int(cfg.solver.train.batch_size)
    indices = _full_window_indices(
        dataset,
        dense_window_size=int(cfg.dense_window_size),
        batch_size=batch_size,
    )
    subset = Subset(dataset, indices)
    loader_cfg = copy.deepcopy(cfg.solver.train)
    loader = build_dataloader(
        subset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=True,
        **loader_cfg,
    )
    batch = next(iter(loader))
    converted = _cuda_batch(batch, device)
    masks = converted["masks"]
    _require(bool(masks.all().item()), "proof batch must contain only full T=768 windows")
    video_names = [str(meta.get("video_name")) for meta in converted["metas"]]
    return converted, {
        "indices": indices,
        "video_names": video_names,
        "valid_lengths": [int(value) for value in masks.long().sum(dim=1).cpu().tolist()],
        "synthetic_inputs_used": False,
        "real_dataset_loader_executed": True,
    }


def _parameter_group(name: str) -> str:
    normalized = name.removeprefix("module.")
    if normalized.startswith("frame_selector.raw_actionness_source.probe_module."):
        return "coarse_probe"
    if normalized.startswith("frame_selector.adapter.transition_scorer."):
        return "transition_scorer"
    if normalized.startswith("frame_selector."):
        return "unexpected_selector"
    return "detector"


def _coarse_subgroup(name: str) -> str:
    normalized = name.removeprefix("module.")
    if ".spatial_stem." in normalized:
        return "spatial_stem"
    if ".official_temporal." in normalized and ".conv_out." in normalized:
        return "action_head"
    if ".official_temporal." in normalized:
        return "temporal_trunk"
    return "coarse_other"


def _gradient_evidence(model: torch.nn.Module) -> dict[str, Any]:
    sums: dict[str, float] = {}
    names: dict[str, list[str]] = {}
    coarse_sums: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().abs().sum().item())
        _require(math.isfinite(value), f"non-finite gradient in {name}")
        if value <= 0.0:
            continue
        group = _parameter_group(name)
        sums[group] = sums.get(group, 0.0) + value
        names.setdefault(group, []).append(name)
        if group == "coarse_probe":
            subgroup = _coarse_subgroup(name)
            coarse_sums[subgroup] = coarse_sums.get(subgroup, 0.0) + value
    return {
        "absolute_gradient_sum": sums,
        "nonzero_parameter_count": {key: len(value) for key, value in names.items()},
        "coarse_subgroup_absolute_gradient_sum": coarse_sums,
        "representative_parameter": {
            key: value[0] for key, value in names.items() if value
        },
    }


def _detector_execution_hooks(model: torch.nn.Module):
    calls: dict[str, int] = {}
    handles = []
    for name in ("backbone", "projection", "neck", "rpn_head"):
        module = getattr(model, name, None)
        if module is None:
            continue
        calls[name] = 0

        def record(_module, _inputs, _output, *, key=name):
            calls[key] += 1

        handles.append(module.register_forward_hook(record))
    return calls, handles


def _remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()


def _frozen_detector_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("frame_selector."):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _validate_loss_inventory(losses: Mapping[str, torch.Tensor], selector) -> dict[str, Any]:
    expected = set(DUCA_LOSS_TO_WEIGHT_KEY) | {"cost"}
    _require(set(losses) == expected, "real forward loss inventory drifted")
    audit = selector.last_forward_summary.get("supervision_loss_audit")
    _require(isinstance(audit, Mapping), "real forward lacks supervision loss audit")
    inactive = sorted(set(DUCA_LOSS_TO_WEIGHT_KEY) - ACTIVE_LOSS_NAMES)
    for name in inactive:
        value = losses[name]
        _require(float(value.detach().float().cpu().item()) == 0.0, f"inactive loss {name} is nonzero")
        _require(not value.requires_grad, f"inactive loss {name} still owns a graph")
        _require(audit[name].get("active") is False, f"inactive audit drifted for {name}")
    active_sum = sum(losses[name] for name in sorted(ACTIVE_LOSS_NAMES))
    _require(
        bool(torch.allclose(losses["cost"], active_sum, rtol=0.0, atol=1.0e-6)),
        "P0 cost is not exactly the three declared objectives",
    )
    return {
        "complete_loss_count": len(DUCA_LOSS_TO_WEIGHT_KEY),
        "active_loss_names": sorted(ACTIVE_LOSS_NAMES),
        "inactive_loss_names": inactive,
        "inactive_losses_graph_free_zero": True,
        "cost_equals_declared_active_sum": True,
        "supervision_loss_audit": dict(audit),
    }


def _forward_gradient_audit(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    selected_losses: Sequence[str],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    losses = model(**batch, return_loss=True)
    selected = sum(losses[name] for name in selected_losses)
    _require(bool(torch.isfinite(selected.detach()).all().item()), "manual gradient loss is non-finite")
    selected.backward()
    return _gradient_evidence(model), dict(losses)


def _normalization_metamorphic_audit(model, batch: Mapping[str, Any]) -> dict[str, Any]:
    source = model.frame_selector.raw_actionness_source
    group_norm_count = sum(isinstance(module, torch.nn.GroupNorm) for module in source.modules())
    batch_norm_count = sum(
        isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
        for module in source.modules()
    )
    _require(group_norm_count > 0, "coarse probe contains no GroupNorm")
    _require(batch_norm_count == 0, "coarse probe still contains BatchNorm")
    was_training = source.training
    source.eval()
    with torch.no_grad():
        single = source(batch["inputs"][:1], valid_mask=batch["masks"][:1])
        paired = source(batch["inputs"], valid_mask=batch["masks"])
    if was_training:
        source.train()
    logit_error = float((single["logits"][0] - paired["logits"][0]).abs().max().item())
    hidden_error = float(
        (single["hidden_features"][0] - paired["hidden_features"][0]).abs().max().item()
    )
    _require(logit_error <= 1.0e-5, "coarse logits depend on another batch member")
    _require(hidden_error <= 1.0e-5, "coarse hidden state depends on another batch member")
    return {
        "group_norm_module_count": group_norm_count,
        "batch_norm_module_count": batch_norm_count,
        "sample_alone_vs_paired_logit_max_abs_error": logit_error,
        "sample_alone_vs_paired_hidden_max_abs_error": hidden_error,
        "batch_content_invariant": True,
    }


def _optimizer_partition(model, optimizer) -> dict[str, Any]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    seen_categories = set()
    for index, group in enumerate(optimizer.param_groups):
        group_names = [names[id(parameter)] for parameter in group["params"]]
        categories = sorted({_parameter_group(name) for name in group_names})
        _require(categories and set(categories) <= {"coarse_probe", "transition_scorer"}, "optimizer contains a detector or unexpected selector parameter")
        _require(len(categories) == 1, "an optimizer group mixes coarse and transition parameters")
        seen_categories.update(categories)
        groups.append(
            {
                "index": index,
                "category": categories[0],
                "parameter_count": len(group_names),
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
            }
        )
    _require(seen_categories == {"coarse_probe", "transition_scorer"}, "optimizer lacks a P0 branch")
    return {
        "optimizer_type": optimizer.__class__.__name__,
        "single_optimizer_with_disjoint_parameter_groups": True,
        "global_gradient_clipping_enabled": False,
        "categories": sorted(seen_categories),
        "groups": groups,
    }


def _representative_parameter(model, name: str) -> torch.nn.Parameter:
    mapping = dict(model.named_parameters())
    _require(name in mapping, f"missing representative parameter {name}")
    return mapping[name]


def _optimizer_step(optimizer, parameter: torch.nn.Parameter) -> int:
    state = optimizer.state.get(parameter)
    _require(isinstance(state, Mapping) and "step" in state, "optimizer state was not created")
    value = state["step"]
    return int(value.detach().cpu().item()) if torch.is_tensor(value) else int(value)


class _OneBatchLoader:
    def __init__(self, batch: Mapping[str, Any]) -> None:
        self.batch = dict(batch)

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        yield self.batch


def _position_scheduler(scheduler, step: int) -> list[float]:
    scheduler.last_epoch = int(step)
    lrs = [float(value) for value in scheduler._get_closed_form_lr()]
    _require(any(value > 0.0 for value in lrs), "proof scheduler position has zero LR")
    for group, lr in zip(scheduler.optimizer.param_groups, lrs):
        group["lr"] = lr
    scheduler._last_lr = list(lrs)
    scheduler._step_count = max(int(getattr(scheduler, "_step_count", 0)), 1)
    return lrs


def run_gate(
    *,
    config_path: str | Path,
    variant_config_paths: Sequence[str | Path],
    expected_commit: str,
    checkpoint_path: str | Path,
    official_repos_root: str | Path,
    split_manifest: str | Path,
    expected_split_sha256: str,
) -> dict[str, Any]:
    _require(len(expected_commit) == 40, "expected commit must be exact")
    initial_git = _clean_commit(expected_commit)
    audited_hashes = _audited_hashes()
    cuda_binding = _slurm_cuda_binding()
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    _require(dist.get_world_size() == 1 and dist.get_rank() == 0, "unexpected process group")
    device = torch.device("cuda:0")
    logger = _logger()
    seed = 20260721
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(0)

    split_path = Path(split_manifest).expanduser().resolve()
    _require(split_path.is_file(), "frontend split manifest is missing")
    _require(_sha256(split_path) == expected_split_sha256, "frontend split manifest hash drift")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    _require(split.get("test_subset_consumed") is False, "P0 split consumed the test subset")

    config_contracts = {}
    for raw in variant_config_paths:
        path = Path(raw).expanduser().resolve()
        contract = validate_config(path)
        _require(contract["git_commit"] == expected_commit, "static P0 contract commit drift")
        config_contracts[str(path)] = contract
    config_file = Path(config_path).expanduser().resolve()
    _require(str(config_file) in config_contracts, "executed config is outside the audited P0 grid")
    cfg = Config.fromfile(str(config_file))

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    official_source = Path(official_repos_root).expanduser().resolve() / "ASFormer" / "model.py"
    train_block = Path(os.environ.get("DUCA_FRONTEND_TRAIN_BLOCK_LIST", "")).expanduser().resolve()
    _require(checkpoint.is_file(), "AdaTAD VideoMAE checkpoint is missing")
    _require(official_source.is_file(), "official ASFormer source is missing")
    _require(train_block.is_file(), "frontend train block list is missing")
    cfg.model.backbone.custom.pretrain = str(checkpoint)

    train_dataset = build_dataset(copy.deepcopy(cfg.dataset.train), default_args={"logger": logger})
    _require(train_dataset.__class__.__name__ == "ThumosPaddingDataset", "P0 did not build ThumosPaddingDataset")
    full_loader = build_dataloader(
        train_dataset,
        rank=0,
        world_size=1,
        shuffle=True,
        drop_last=True,
        **copy.deepcopy(cfg.solver.train),
    )
    batch, batch_evidence = _real_batch(train_dataset, cfg, device)

    model = build_detector(copy.deepcopy(cfg.model)).to(device)
    _require(model.__class__.__name__ == "ActionFormer", "P0 model is not ActionFormer")
    _require(model.rpn_head.__class__.__name__ == "ActionFormerHead", "P0 detector head is not ActionFormerHead")
    _require(model.selector_train_only and model.selector_train_only_skip_detector, "P0 detector skip is disabled")
    selector = model.frame_selector
    _require(selector.__class__.__name__ == "DucaOnlineFrameSelector", "P0 selector type drifted")
    _require(selector.raw_actionness_source.__class__.__name__ == "C3CoarseProbeActionnessSource", "P0 coarse source drifted")
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    categories = {_parameter_group(name) for name in trainable}
    _require(categories == {"coarse_probe", "transition_scorer"}, "P0 trainable parameter surface drifted")
    _require(all(not parameter.requires_grad for name, parameter in model.named_parameters() if not name.startswith("frame_selector.")), "detector parameters remain trainable in P0")
    prepare_optimizer_parameter_freezing(copy.deepcopy(cfg.optimizer), model, logger)

    action_target = selector._action_target_from_gt_segments(batch["gt_segments"], batch["masks"])
    valid = batch["masks"].bool()
    positive_count = int((action_target.bool() & valid).sum().item())
    negative_count = int(((~action_target.bool()) & valid).sum().item())
    _require(positive_count > 0 and negative_count > 0, "real P0 batch lacks both action classes")

    model.train()
    selector._loss_weight_schedule_step.fill_(100)
    selector._pending_loss_schedule_advance = False
    normalization = _normalization_metamorphic_audit(model, batch)
    calls, handles = _detector_execution_hooks(model)
    try:
        action_gradients, action_losses = _forward_gradient_audit(
            model,
            batch,
            selected_losses=("actionness_bce_loss",),
        )
        loss_inventory = _validate_loss_inventory(action_losses, selector)
        action_sums = action_gradients["absolute_gradient_sum"]
        action_coarse = action_gradients["coarse_subgroup_absolute_gradient_sum"]
        _require(action_sums.get("coarse_probe", 0.0) > 0.0, "actionness did not train coarse probe")
        _require(action_sums.get("transition_scorer", 0.0) == 0.0, "actionness leaked into transition scorer")
        for subgroup in ("spatial_stem", "temporal_trunk", "action_head"):
            _require(action_coarse.get(subgroup, 0.0) > 0.0, f"actionness lacks {subgroup} gradients")

        transition_gradients, transition_losses = _forward_gradient_audit(
            model,
            batch,
            selected_losses=(
                "transition_distribution_loss",
                "transition_boundary_coverage_loss",
            ),
        )
        _validate_loss_inventory(transition_losses, selector)
        transition_sums = transition_gradients["absolute_gradient_sum"]
        _require(transition_sums.get("transition_scorer", 0.0) > 0.0, "transition objectives did not train scorer")
        _require(transition_sums.get("coarse_probe", 0.0) == 0.0, "transition objectives rewrote coarse probe")
        _require(transition_sums.get("detector", 0.0) == 0.0, "transition objectives reached detector")
    finally:
        _remove_hooks(handles)
    _require(all(value == 0 for value in calls.values()), "manual P0 audit executed the detector")

    coarse_name = action_gradients["representative_parameter"]["coarse_probe"]
    scorer_name = transition_gradients["representative_parameter"]["transition_scorer"]
    frozen_state_before = _frozen_detector_state_sha256(model)
    model.zero_grad(set_to_none=True)
    selector._pending_loss_schedule_advance = False

    ddp_model = DistributedDataParallel(
        model,
        device_ids=[0],
        output_device=0,
        find_unused_parameters=bool(cfg.solver.find_unused_parameters),
        static_graph=bool(cfg.solver.static_graph),
    )
    if bool(cfg.solver.get("fp16_compress", False)):
        ddp_model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    optimizer = build_optimizer(copy.deepcopy(cfg.optimizer), ddp_model, logger)
    assert_optimizer_exact_coverage(model, optimizer)
    optimizer_partition = _optimizer_partition(model, optimizer)
    scheduler, scheduler_max_epoch = build_scheduler(
        copy.deepcopy(cfg.scheduler), optimizer, len(full_loader)
    )
    _require(int(scheduler_max_epoch) == int(cfg.workflow.end_epoch), "P0 scheduler epoch contract drifted")
    scheduler_seed_step = max(1, len(full_loader))
    initial_lrs = _position_scheduler(scheduler, scheduler_seed_step)
    selector._loss_weight_schedule_step.fill_(100)
    selector._pending_loss_schedule_advance = False
    model_ema = ModelEma(ddp_model)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    coarse_parameter = _representative_parameter(model, coarse_name)
    scorer_parameter = _representative_parameter(model, scorer_name)
    coarse_before = coarse_parameter.detach().clone()
    scorer_before = scorer_parameter.detach().clone()
    ema_root = getattr(model_ema.module, "module", model_ema.module)
    ema_named = dict(ema_root.named_parameters())
    ema_coarse_before = ema_named[coarse_name].detach().clone()
    ema_scorer_before = ema_named[scorer_name].detach().clone()
    schedule_before = int(selector._loss_weight_schedule_step.item())
    scheduler_before = int(scheduler.last_epoch)
    update_audit: dict[str, Any] = {}
    calls, handles = _detector_execution_hooks(model)
    try:
        probe = train_one_epoch(
            _OneBatchLoader(batch),
            ddp_model,
            optimizer,
            scheduler,
            0,
            logger,
            model_ema=model_ema,
            clip_grad_l2norm=float(cfg.solver.clip_grad_norm),
            logging_interval=1,
            scaler=scaler,
            max_train_iters=1,
            collect_training_probe=True,
            max_amp_retries_per_batch=int(cfg.workflow.max_amp_retries_per_batch),
            fail_on_amp_replay_exhaustion=True,
            require_finite_loss=True,
            update_audit=update_audit,
        )
    finally:
        _remove_hooks(handles)
    _require(all(value == 0 for value in calls.values()), "production P0 update executed the detector")
    _require(int(update_audit.get("successful_optimizer_updates", 0)) == 1, "P0 gate lacks one successful optimizer update")
    _require(int(update_audit.get("scheduler_updates", 0)) == 1, "P0 scheduler did not update once")
    _require(int(update_audit.get("ema_updates", 0)) == 1, "P0 EMA did not update once")
    _require(int(update_audit.get("duca_schedule_updates", 0)) == 1, "P0 loss schedule did not update once")
    _require(isinstance(probe, Mapping), "training engine returned no P0 probe")
    _require(int(probe.get("successful_optimizer_steps", 0)) == 1, "training probe lacks a successful step")
    coverage = probe.get("parameter_group_coverage", {})
    _require(int(coverage.get("coarse_probe", {}).get("gradient_seen", 0)) > 0, "production update lacks coarse gradients")
    _require(int(coverage.get("selector", {}).get("gradient_seen", 0)) > 0, "production update lacks scorer gradients")
    for forbidden in ("backbone", "projection", "neck", "detector_head"):
        _require(int(coverage.get(forbidden, {}).get("trainable", 0)) == 0, f"production optimizer contains {forbidden}")

    coarse_delta = float((coarse_parameter.detach() - coarse_before).abs().max().item())
    scorer_delta = float((scorer_parameter.detach() - scorer_before).abs().max().item())
    ema_coarse_delta = float((ema_named[coarse_name].detach() - ema_coarse_before).abs().max().item())
    ema_scorer_delta = float((ema_named[scorer_name].detach() - ema_scorer_before).abs().max().item())
    _require(coarse_delta > 0.0 and scorer_delta > 0.0, "production update did not change both P0 branches")
    _require(ema_coarse_delta > 0.0 and ema_scorer_delta > 0.0, "EMA did not update both P0 branches")
    _require(_optimizer_step(optimizer, coarse_parameter) == 1, "coarse Adam state did not advance once")
    _require(_optimizer_step(optimizer, scorer_parameter) == 1, "scorer Adam state did not advance once")
    _require(int(selector._loss_weight_schedule_step.item()) == schedule_before + 1, "P0 schedule step drifted")
    _require(int(scheduler.last_epoch) == scheduler_before + 1, "LR scheduler step drifted")
    frozen_state_after = _frozen_detector_state_sha256(model)
    _require(frozen_state_after == frozen_state_before, "frozen AdaTAD state changed during P0")

    selected_positions = selector._last_selected_positions
    _require(selected_positions.shape[1] == int(cfg.window_size), "P0 selector output width drifted")
    _require(bool((selected_positions >= 0).all().item()), "full-window P0 batch did not emit exact K")
    max_holes = []
    for row in selected_positions.detach().cpu().tolist():
        holes = [row[0]] + [right - left - 1 for left, right in zip(row[:-1], row[1:])]
        holes.append(int(cfg.dense_window_size) - row[-1] - 1)
        max_holes.append(max(holes))
    _require(max(max_holes) <= int(selector.max_unselected_hole), "P0 hard selection violated max hole")

    final_git = _clean_commit(expected_commit)
    _require(final_git["git_tree"] == initial_git["git_tree"], "source tree changed during gate")
    _require(_audited_hashes() == audited_hashes, "audited implementation changed during gate")
    return {
        "schema": SCHEMA,
        "ok": True,
        "fail_closed": True,
        "task": "offline_temporal_action_detection",
        "git_binding": initial_git,
        "final_git_binding": final_git,
        "audited_file_sha256": audited_hashes,
        "slurm_cuda_binding": cuda_binding,
        "config_path": str(config_file),
        "config_sha256": _sha256(config_file),
        "variant_config_contracts": config_contracts,
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": expected_split_sha256,
        "train_block_list_path": str(train_block),
        "train_block_list_sha256": _sha256(train_block),
        "assets": {
            "videomae_checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
            "official_asformer_source": {"path": str(official_source), "sha256": _sha256(official_source)},
        },
        "dataset": {
            "type": train_dataset.__class__.__name__,
            "size": len(train_dataset),
            "train_batches_per_epoch": len(full_loader),
            **batch_evidence,
            "action_positive_count": positive_count,
            "action_negative_count": negative_count,
        },
        "model": {
            "type": model.__class__.__name__,
            "detector_head_type": model.rpn_head.__class__.__name__,
            "selector_type": selector.__class__.__name__,
            "coarse_source_type": selector.raw_actionness_source.__class__.__name__,
            "selector_train_only": True,
            "detector_skipped": True,
            "trainable_parameter_count": len(trainable),
            "trainable_categories": sorted(categories),
        },
        "normalization_metamorphic_audit": normalization,
        "loss_inventory": loss_inventory,
        "gradient_ownership": {
            "actionness_only": action_gradients,
            "transition_and_boundary_only": transition_gradients,
            "actionness_updates_coarse_only": True,
            "transition_updates_scorer_only": True,
            "detector_gradient_present": False,
        },
        "optimizer": optimizer_partition,
        "production_amp_update": {
            "amp_enabled": True,
            "grad_scaler_enabled": bool(scaler.is_enabled()),
            "initial_nonzero_lrs": initial_lrs,
            "scheduler_seed_step": scheduler_seed_step,
            "successful_optimizer_updates": 1,
            "coarse_parameter": coarse_name,
            "coarse_parameter_max_abs_change": coarse_delta,
            "scorer_parameter": scorer_name,
            "scorer_parameter_max_abs_change": scorer_delta,
            "ema_coarse_parameter_max_abs_change": ema_coarse_delta,
            "ema_scorer_parameter_max_abs_change": ema_scorer_delta,
            "update_audit": update_audit,
            "training_probe": dict(probe),
        },
        "hard_selection": {
            "requested_k": int(cfg.window_size),
            "effective_k": [int((row >= 0).sum().item()) for row in selected_positions],
            "max_unselected_hole": max_holes,
            "configured_max_unselected_hole": int(selector.max_unselected_hole),
        },
        "frozen_detector_state": {
            "sha256_before": frozen_state_before,
            "sha256_after": frozen_state_after,
            "byte_invariant": True,
            "forward_call_count": calls,
        },
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "claims": {
            "real_thumos_batch": True,
            "official_asformer_coarse_probe": True,
            "complete_adatad_object_built": True,
            "adatad_detector_executed_during_p0": False,
            "strict_loss_inventory": True,
            "gradient_ownership_verified": True,
            "optimizer_exact_coverage": True,
            "one_production_amp_scheduler_ema_update": True,
            "metric_claim_allowed": False,
            "paper_ready": False,
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the exact real-data DUCA frontend P0 CUDA gate.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--variant-config", action="append", default=[])
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--official-repos-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    output = _outside_worktree(args.output_json)
    variant_configs = args.variant_config or list(DEFAULT_VARIANT_CONFIGS)
    try:
        payload = run_gate(
            config_path=args.config,
            variant_config_paths=variant_configs,
            expected_commit=args.expected_commit,
            checkpoint_path=args.checkpoint,
            official_repos_root=args.official_repos_root,
            split_manifest=args.split_manifest,
            expected_split_sha256=args.expected_split_sha256,
        )
        exit_code = 0
    except Exception as exc:
        payload = {
            "schema": SCHEMA,
            "ok": False,
            "fail_closed": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    _write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
