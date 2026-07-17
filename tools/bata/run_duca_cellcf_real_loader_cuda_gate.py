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
import tempfile
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping


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

from opentad.cores import (
    build_optimizer,
    build_scheduler,
    prepare_optimizer_parameter_freezing,
    train_one_epoch,
)
from opentad.datasets import build_dataloader, build_dataset
from opentad.models import build_detector
from opentad.models.duca.structured_selection import exact_uniform_cell_bounds
from opentad.models.utils.truetime_geometry import SELECTED_AXIS, TRUE_TIME_AXIS, TrueTimeMap
from opentad.utils import ModelEma
from tools.bata.duca_cellcf_protocol import protocol_from_workflow


SCHEMA = "duca_cellcf_real_loader_cuda_gate_v1"
SYNTHETIC_GATE_SCHEMA = "duca_cellcf_synthetic_gate_v1"
CONFIG_DEFAULT = (
    "configs/adatad/thumos/"
    "duca_cellcf_fixed384_official_adatad_backend_full_train.py"
)
ASFORMER_BINDING_CONFIG = (
    "configs/adatad/thumos/"
    "duca_transition_only_fixed384_official_adatad_backend_full_train.py"
)
GATE_SEED = 20260716
EXPECTED_OFFICIAL_ASFORMER_NORMALIZED_LF_SHA256 = (
    "e075ee4825a201cfe324d5fbfb1332c0800f532e85b9d3809f6ca5180381c600"
)
EXPECTED_VIDEOMAE_CONFIG_NAME = (
    "vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
)
AUDITED_PATHS = (
    "opentad/cores/optimizer.py",
    "opentad/cores/scheduler.py",
    "opentad/cores/train_engine.py",
    "opentad/datasets/builder.py",
    "opentad/datasets/thumos.py",
    "opentad/datasets/transforms/end_to_end.py",
    "opentad/models/backbones/backbone_wrapper.py",
    "opentad/models/detectors/actionformer.py",
    "opentad/models/duca/acquisition.py",
    "opentad/models/duca/counterfactual_utility.py",
    "opentad/models/duca/structured_selection.py",
    "opentad/models/duca/transition_only.py",
    "opentad/models/selectors/duca_online_frame_selector.py",
    "opentad/utils/training_guard.py",
    "opentad/models/utils/truetime_geometry.py",
    "opentad/utils/ema.py",
    "configs/_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py",
    "configs/_base_/models/actionformer.py",
    "configs/adatad/thumos/duca_cellcf_fixed384_official_adatad_backend_full_train.py",
    "configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py",
    "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py",
    "tools/bata/run_duca_cellcf_synthetic_gate.py",
    "tools/bata/run_duca_cellcf_real_loader_cuda_gate.py",
    "tools/bata/validate_duca_cellcf_real_loader_gate.py",
    "tools/bata/duca_cellcf_protocol.py",
    "tools/bata/duca_cellcf_training.py",
    "tools/bata/finalize_duca_cellcf_run.py",
    "tools/bata/validate_duca_cellcf_ddp_pilot.py",
    "tools/bata/validate_duca_cellcf_fixed384.py",
    "tools/bata/validate_duca_cellcf_suite.py",
    "tools/train.py",
    "tools/test.py",
    "tests/test_duca_cellcf_real_loader_gate_contract.py",
)


class GateFailure(RuntimeError):
    def __init__(self, message: str, *, evidence: Mapping[str, Any] | None = None) -> None:
        super().__init__(f"fail-closed CellCF real-loader CUDA gate: {message}")
        self.evidence = dict(evidence or {})


def _require(
    condition: bool,
    message: str,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> None:
    if not condition:
        raise GateFailure(message, evidence=evidence)


def _path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_lf_sha256(path: str | Path) -> str:
    source = _path(path).read_bytes()
    return hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _position_sha256(values: list[int]) -> str:
    return _json_sha256([int(value) for value in values])


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def _bind_clean_commit(expected_commit: str) -> dict[str, Any]:
    expected = str(expected_commit).strip().lower()
    _require(
        re.fullmatch(r"[0-9a-f]{40}", expected) is not None,
        "--expected-commit must be an exact 40-character commit id",
    )
    head = _git_output("rev-parse", "--verify", "HEAD")
    tree = _git_output("rev-parse", "HEAD^{tree}")
    branch = _git_output("rev-parse", "--abbrev-ref", "HEAD")
    status = _git_output("status", "--porcelain", "--untracked-files=normal")
    _require(
        head == expected,
        "checked-out commit does not match --expected-commit",
        evidence={"expected_commit": expected, "observed_commit": head},
    )
    _require(
        not status,
        "the gate requires a clean exact-commit checkout",
        evidence={"git_commit": head, "git_status_porcelain": status.splitlines()},
    )
    return {
        "git_commit": head,
        "git_tree": tree,
        "git_branch": branch,
        "git_tree_clean": True,
        "expected_commit_exact_match": True,
    }


def _bind_synthetic_gate(path: str | Path, *, git_commit: str) -> dict[str, Any]:
    gate_path = _path(path)
    _require(gate_path.is_file(), f"synthetic gate JSON is missing: {gate_path}")
    try:
        payload = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateFailure(f"synthetic gate JSON is unreadable: {exc}") from exc
    _require(isinstance(payload, Mapping), "synthetic gate JSON must contain an object")
    _require(payload.get("schema") == SYNTHETIC_GATE_SCHEMA, "synthetic gate schema mismatch")
    _require(payload.get("ok") is True, "synthetic gate did not pass")
    _require(payload.get("git_tree_clean") is True, "synthetic gate was not produced from a clean tree")
    _require(
        payload.get("git_commit") == git_commit,
        "synthetic gate commit does not match this real-loader gate commit",
        evidence={
            "synthetic_git_commit": payload.get("git_commit"),
            "real_loader_git_commit": git_commit,
        },
    )
    _require(
        payload.get("real_dataset_loader_executed") is False,
        "bound synthetic gate must honestly declare that it did not run the real loader",
    )
    training_profile = payload.get("training_profile")
    _require(
        training_profile in {"exposure132", "official60"},
        "synthetic gate has no valid CellCF training profile",
    )
    audited = payload.get("audited_file_sha256")
    _require(isinstance(audited, Mapping), "synthetic gate lacks audited_file_sha256")
    synthetic_script = "tools/bata/run_duca_cellcf_synthetic_gate.py"
    _require(
        audited.get(synthetic_script) == _sha256(ROOT / synthetic_script),
        "synthetic gate does not bind the current synthetic gate implementation",
    )
    protocol_module = "tools/bata/duca_cellcf_protocol.py"
    _require(
        audited.get(protocol_module) == _sha256(ROOT / protocol_module),
        "synthetic gate does not bind the current training-profile contract",
    )
    return {
        "path": str(gate_path),
        "sha256": _sha256(gate_path),
        "schema": SYNTHETIC_GATE_SCHEMA,
        "ok": True,
        "git_commit": git_commit,
        "git_tree_clean": True,
        "training_profile": training_profile,
        "input_provenance": payload.get("input_provenance"),
        "claims": dict(payload.get("claims", {})),
    }


def _require_output_outside_worktree(path: str | Path) -> Path:
    output = _path(path)
    _require(
        not output.exists(),
        "refusing to overwrite real-loader gate evidence",
    )
    try:
        output.relative_to(ROOT)
    except ValueError:
        return output
    raise GateFailure(
        "--output-json must be outside the Git worktree so evidence creation cannot dirty the bound tree"
    )


def _bind_slurm_logical_cuda(device: str) -> dict[str, Any]:
    _require(str(device) == "cuda:0", "the gate is fixed to Slurm logical cuda:0")
    job_id = os.environ.get("SLURM_JOB_ID")
    _require(bool(job_id), "SLURM_JOB_ID is required; do not run this CUDA gate on a login node")
    _require(
        bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
        "Slurm must provide CUDA_VISIBLE_DEVICES",
    )
    for name in ("SLURM_LOCALID", "LOCAL_RANK", "RANK"):
        value = os.environ.get(name)
        if value is not None:
            _require(int(value) == 0, f"{name} must be logical rank zero for this one-GPU gate")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _require(world_size == 1, "the real-loader gate is a one-process Slurm CUDA proof")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(
        torch.cuda.device_count() == 1,
        "the Slurm allocation must expose exactly one logical CUDA device",
        evidence={"logical_cuda_device_count": torch.cuda.device_count()},
    )
    torch.cuda.set_device(0)
    _require(torch.cuda.current_device() == 0, "failed to select logical cuda:0")
    return {
        "slurm_job_id": str(job_id),
        "slurm_local_id": int(os.environ.get("SLURM_LOCALID", "0")),
        "world_size": 1,
        "logical_device": "cuda:0",
        "logical_cuda_device_count": 1,
        "cuda_visible_devices_supplied_by_slurm": True,
        "physical_gpu_index_assumed": False,
        "device_name": torch.cuda.get_device_name(0),
        "torch_cuda_version": torch.version.cuda,
    }


@contextmanager
def _single_process_nccl_group():
    owned = False
    rendezvous_path: Path | None = None
    if dist.is_initialized():
        _require(dist.get_backend() == "nccl", "existing process group must use NCCL")
        _require(dist.get_rank() == 0 and dist.get_world_size() == 1, "existing process group must be rank 0/1")
    else:
        handle, name = tempfile.mkstemp(prefix="duca-cellcf-real-loader-nccl-")
        os.close(handle)
        rendezvous_path = Path(name)
        rendezvous_path.unlink()
        dist.init_process_group(
            backend="nccl",
            init_method=rendezvous_path.as_uri(),
            rank=0,
            world_size=1,
            timeout=timedelta(minutes=5),
        )
        owned = True
    try:
        yield {
            "backend": str(dist.get_backend()),
            "rank": int(dist.get_rank()),
            "world_size": int(dist.get_world_size()),
            "initialized_by_gate": owned,
        }
    finally:
        if owned and dist.is_initialized():
            dist.destroy_process_group()
        if rendezvous_path is not None:
            rendezvous_path.unlink(missing_ok=True)


def _config_path(value: str | Path) -> Path:
    candidate = Path(value)
    return _path(ROOT / candidate) if not candidate.is_absolute() else _path(candidate)


def _validate_main_config(cfg: Config, config_path: Path) -> dict[str, Any]:
    expected_path = _path(ROOT / CONFIG_DEFAULT)
    _require(config_path == expected_path, f"the gate is fixed to the CellCF main config: {expected_path}")
    contract = cfg.get("duca_transition_only_contract")
    _require(contract is not None, "CellCF contract is missing")
    required_contract = {
        "route": "DUCA_LOCAL_CELL_COUNTERFACTUAL_FIXED384",
        "task": "offline_temporal_action_detection",
        "online_tad": False,
        "streaming": False,
        "full_window_selector": True,
        "official_adatad_backend": True,
        "detector_head_type": "ActionFormerHead",
        "detector_head_changed": False,
        "detector_loss_changed": False,
        "detector_nms_changed": False,
        "acquisition_policy": "local_cell_deformation",
        "hard_soft_feasible_family": "one_frame_per_exact_uniform_voronoi_cell",
        "exact_budget": 384,
        "dense_window_size": 768,
        "counterfactual_teacher_producer_integrated": True,
        "counterfactual_teacher_detector_objective": "official_actionformer_cls_plus_reg",
        "teacher_free_eval": True,
    }
    drift = {
        key: {"expected": expected, "observed": contract.get(key)}
        for key, expected in required_contract.items()
        if contract.get(key) != expected
    }
    _require(not drift, "CellCF main contract drifted", evidence={"contract_drift": drift})

    selector = cfg.model.frame_selector
    source = selector.actionness_source_cfg
    training_protocol = protocol_from_workflow(cfg.workflow)
    _require(cfg.model.type == "ActionFormer", "model.type must remain ActionFormer")
    _require(cfg.model.rpn_head.type == "ActionFormerHead", "detector head must remain ActionFormerHead")
    official_base_path = _path(ROOT / "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py")
    official_base = Config.fromfile(str(official_base_path))
    _require(
        _plain(cfg.model.rpn_head) == _plain(official_base.model.rpn_head),
        "resolved ActionFormerHead config drifted from the official AdaTAD base",
    )
    _require(
        _plain(cfg.post_processing.nms) == _plain(official_base.post_processing.nms),
        "resolved ActionFormer NMS config drifted from the official AdaTAD base",
    )
    _require(selector.acquisition_policy == "local_cell_deformation", "selector must use local-cell deformation")
    _require(int(selector.budget) == 384, "CellCF budget must remain K=384")
    _require(int(selector.dense_window_size) == 768, "CellCF dense window must remain T=768")
    _require(selector.budget_mode == "fixed", "CellCF must use a fixed budget")
    _require(selector.detector_gradient_mode == "none", "CellCF must not enable a direct detector gradient bridge")
    _require(
        selector.counterfactual_objective == "local_cell_signed_logistic",
        "CellCF must use the local signed counterfactual objective",
    )
    _require(selector.require_counterfactual_utility_teacher is True, "integrated counterfactual teacher is required")
    _require(float(selector.counterfactual_utility_distillation_weight) > 0.0, "counterfactual weight must be positive")
    _require(source.probe_model == "official-action-seg", "coarse source must use official action segmentation")
    _require(source.official_action_seg_backend == "official_asformer", "coarse source must use official ASFormer")
    _require(source.hidden_output_kind == "official_asformer_encoder_hidden", "official ASFormer hidden output is required")
    _require(source.trainable is True and source.frozen is False, "official ASFormer source must be trainable")

    _require(int(cfg.window_size) == 384 and int(cfg.dense_window_size) == 768, "resolved K/T contract drifted")
    _require(int(cfg.model.backbone.backbone.total_frames) == 384, "VideoMAE must receive K=384")
    _require(cfg.model.backbone.backbone.with_cp is False, "formal dynamic-DDP path requires with_cp=False")
    _require(int(cfg.model.projection.max_seq_len) == 384, "projection grid must remain K=384")
    _require(
        int(cfg.workflow.end_epoch) == training_protocol.end_epoch,
        "training epoch profile drifted",
    )
    _require(
        int(cfg.scheduler.max_epoch) == training_protocol.end_epoch,
        "scheduler epoch profile drifted",
    )
    _require(int(cfg.workflow.checkpoint_interval) == 5, "checkpoint cadence must remain every 5 epochs")
    _require(int(cfg.workflow.expected_train_batches_per_epoch) == 100, "formal schedule expects 100 batches per epoch")
    _require(
        int(cfg.workflow.expected_successful_optimizer_updates)
        == training_protocol.expected_successful_optimizer_updates,
        "formal successful-update profile drifted",
    )
    _require(
        int(cfg.workflow.primary_checkpoint_epoch)
        == training_protocol.terminal_epoch,
        "primary checkpoint must remain the terminal epoch",
    )
    _require(cfg.workflow.primary_checkpoint_state_key == "state_dict_ema", "primary checkpoint must use EMA state")
    _require(cfg.solver.amp is True and cfg.solver.ema is True, "official path requires AMP and EMA")
    _require(cfg.solver.static_graph is False, "real mixed windows require dynamic DDP")
    _require(cfg.solver.find_unused_parameters is True, "dynamic DDP must discover unused parameters")
    _require(cfg.solver.fp16_compress is True, "official DDP FP16 compression must remain enabled")
    _require(int(cfg.solver.train.batch_size) == 2, "official CellCF train batch size must remain two")
    _require(int(cfg.solver.train.num_workers) == 2, "gate must use the configured train worker count")
    _require(cfg.inference.load_from_raw_predictions is False, "raw prediction loading is forbidden")
    _require(cfg.inference.save_raw_prediction is False, "raw prediction saving is forbidden")
    _require(contract.paper_claim_allowed is False, "an untested CellCF config cannot grant a paper claim")
    _require(contract.metric_claim_allowed is False, "an untested CellCF config cannot grant a metric claim")

    pipeline = list(cfg.dataset.train.pipeline)
    load_frames = [item for item in pipeline if item.get("type") == "LoadFrames"]
    _require(len(load_frames) == 1, "train pipeline must contain exactly one LoadFrames transform")
    load_frames_cfg = load_frames[0]
    _require(load_frames_cfg.method == "random_trunc", "real gate requires the dense random_trunc loader")
    _require(int(load_frames_cfg.trunc_len) == 768, "real train loader must decode T=768")
    forbidden_methods = {
        "exact_uniform_fixed_subsample",
        "random_fixed_subsample",
        "bata_value_transport_ledger_subsample",
    }
    _require(
        not any(item.get("method") in forbidden_methods for item in pipeline),
        "real-loader gate forbids preselected or ledger-backed train inputs",
    )
    configured_pretrain = Path(str(cfg.model.backbone.custom.pretrain)).name
    _require(
        configured_pretrain == EXPECTED_VIDEOMAE_CONFIG_NAME,
        "official VideoMAE pretrain filename drifted in the base config",
    )
    binding_config_path = _path(ROOT / ASFORMER_BINDING_CONFIG)
    _require(binding_config_path.is_file(), f"ASFormer binding config is missing: {binding_config_path}")
    binding_cfg = Config.fromfile(str(binding_config_path))
    expected_asformer_hash = str(
        binding_cfg.duca_transition_only_contract.official_asformer_source_normalized_lf_sha256
    )
    _require(
        expected_asformer_hash == EXPECTED_OFFICIAL_ASFORMER_NORMALIZED_LF_SHA256,
        "audited parent config ASFormer hash drifted from the gate constant",
    )
    return {
        "path": str(config_path),
        "sha256": _sha256(config_path),
        "resolved_sha256": _json_sha256(cfg.to_dict()),
        "route": str(contract.route),
        "task": str(contract.task),
        "offline_tad": True,
        "online_tad": False,
        "dense_window_size": 768,
        "fixed_budget": 384,
        "training_profile": training_protocol.name,
        "training_protocol_purpose": training_protocol.purpose,
        "end_epoch": training_protocol.end_epoch,
        "checkpoint_interval": 5,
        "expected_train_batches_per_epoch": 100,
        "expected_successful_updates": (
            training_protocol.expected_successful_optimizer_updates
        ),
        "expected_asformer_normalized_lf_sha256": expected_asformer_hash,
        "asformer_hash_binding_config": str(binding_config_path),
        "asformer_hash_binding_config_sha256": _sha256(binding_config_path),
        "configured_videomae_pretrain_name": configured_pretrain,
        "official_base_path": str(official_base_path),
        "official_base_sha256": _sha256(official_base_path),
        "official_actionformer_head_config_exact_match": True,
        "official_actionformer_nms_config_exact_match": True,
        "primary_checkpoint_epoch": training_protocol.terminal_epoch,
        "primary_checkpoint_state_key": training_protocol.terminal_state_key,
    }


def _bind_external_assets(
    cfg: Config,
    *,
    videomae_checkpoint: str | Path,
    expected_videomae_sha256: str,
    official_repos_root: str | Path,
) -> dict[str, Any]:
    checkpoint = _path(videomae_checkpoint)
    expected_checkpoint_hash = str(expected_videomae_sha256).strip().lower()
    _require(checkpoint.is_file(), f"official VideoMAE checkpoint is missing: {checkpoint}")
    _require(checkpoint.stat().st_size > 1024 * 1024, "VideoMAE checkpoint is implausibly small")
    _require(
        re.fullmatch(r"[0-9a-f]{64}", expected_checkpoint_hash) is not None,
        "--expected-videomae-sha256 must be a 64-character SHA-256",
    )
    observed_checkpoint_hash = _sha256(checkpoint)
    _require(
        observed_checkpoint_hash == expected_checkpoint_hash,
        "VideoMAE checkpoint SHA-256 mismatch",
        evidence={
            "expected_videomae_sha256": expected_checkpoint_hash,
            "observed_videomae_sha256": observed_checkpoint_hash,
        },
    )

    repos_root = _path(official_repos_root)
    source = repos_root / "ASFormer" / "model.py"
    _require(repos_root.is_dir(), f"official action-segmentation repository root is missing: {repos_root}")
    _require(source.is_file(), f"official ASFormer source is missing: {source}")
    source_raw_hash = _sha256(source)
    source_normalized_hash = _normalized_lf_sha256(source)
    expected_source_hash = EXPECTED_OFFICIAL_ASFORMER_NORMALIZED_LF_SHA256
    _require(
        source_normalized_hash == expected_source_hash,
        "official ASFormer normalized source hash does not match the config contract",
        evidence={
            "expected_normalized_lf_sha256": expected_source_hash,
            "observed_normalized_lf_sha256": source_normalized_hash,
        },
    )
    os.environ["C3_OFFICIAL_ACTION_SEG_REPOS"] = str(repos_root)
    cfg.model.backbone.custom.pretrain = str(checkpoint)
    return {
        "videomae_checkpoint": {
            "path": str(checkpoint),
            "sha256": observed_checkpoint_hash,
            "expected_sha256_exact_match": True,
            "size_bytes": int(checkpoint.stat().st_size),
            "loaded_through_backbone_custom_pretrain": True,
        },
        "official_asformer_source": {
            "repository_root": str(repos_root),
            "path": str(source),
            "sha256": source_raw_hash,
            "normalized_lf_sha256": source_normalized_hash,
            "config_hash_exact_match": True,
        },
    }


def _dataset_bindings(cfg: Config) -> dict[str, Any]:
    ann_file = _path(cfg.dataset.train.ann_file)
    class_map = _path(cfg.dataset.train.class_map)
    data_path = _path(cfg.dataset.train.data_path)
    _require(ann_file.is_file(), f"THUMOS train annotation is missing: {ann_file}")
    _require(class_map.is_file(), f"THUMOS class map is missing: {class_map}")
    _require(data_path.is_dir(), f"THUMOS train video directory is missing: {data_path}")
    return {
        "annotation_path": str(ann_file),
        "annotation_sha256": _sha256(ann_file),
        "class_map_path": str(class_map),
        "class_map_sha256": _sha256(class_map),
        "data_path": str(data_path),
        "subset_name": str(cfg.dataset.train.subset_name),
        "dataset_type": str(cfg.dataset.train.type),
    }


def _plan_validity_coverage(dataset, *, dense_window_size: int, batch_size: int) -> dict[str, dict[str, Any]]:
    _require(hasattr(dataset, "data_list"), "THUMOS train dataset does not expose its annotation index")
    _require(int(batch_size) == 2, "CellCF validity planner is fixed to the configured batch size two")
    snippet_stride = int(getattr(dataset, "snippet_stride", 0))
    _require(snippet_stride > 0, "THUMOS train dataset has an invalid snippet stride")
    full_indices: list[int] = []
    short_indices: list[int] = []
    expected_valid_lengths: dict[int, int] = {}
    for index, row in enumerate(dataset.data_list):
        _require(len(row) >= 2 and isinstance(row[1], Mapping), "malformed THUMOS annotation index row")
        frame_count = int(row[1].get("frame", 0))
        _require(frame_count > 0, "THUMOS annotation index contains a non-positive frame count")
        valid_length = int(math.ceil(frame_count / snippet_stride))
        expected_valid_lengths[index] = min(valid_length, int(dense_window_size))
        if valid_length >= int(dense_window_size):
            full_indices.append(index)
        else:
            short_indices.append(index)

    def entry(indices: list[int] | None, reason: str) -> dict[str, Any]:
        return {
            "obtainable": indices is not None,
            "executed": False,
            "indices": [] if indices is None else [int(index) for index in indices],
            "annotation_expected_valid_lengths": (
                [] if indices is None else [expected_valid_lengths[index] for index in indices]
            ),
            "reason": reason,
        }

    plans = {
        "full": entry(
            full_indices[:batch_size] if len(full_indices) >= batch_size else None,
            "two annotation-index videos reach T=768"
            if len(full_indices) >= batch_size
            else "fewer than two annotation-index videos reach T=768",
        ),
        "mixed": entry(
            [full_indices[0], short_indices[0]] if full_indices and short_indices else None,
            "one full and one short annotation-index video exist"
            if full_indices and short_indices
            else "the annotation index does not contain both a full and a short video",
        ),
        "all_short": entry(
            short_indices[:batch_size] if len(short_indices) >= batch_size else None,
            "two annotation-index videos are shorter than T=768"
            if len(short_indices) >= batch_size
            else "fewer than two annotation-index videos are shorter than T=768",
        ),
    }
    plans["annotation_inventory"] = {
        "full_video_count": len(full_indices),
        "short_video_count": len(short_indices),
        "dataset_size": len(dataset.data_list),
        "snippet_stride": snippet_stride,
    }
    return plans


def _classify_validity_pattern(masks: torch.Tensor, dense_window_size: int) -> tuple[str, list[int]]:
    _require(torch.is_tensor(masks) and masks.ndim == 2, "real loader masks must be [B,T]")
    _require(int(masks.shape[1]) == int(dense_window_size), "real loader temporal mask length drifted")
    valid = masks.bool()
    valid_lengths = valid.long().sum(dim=1)
    expected_prefix = torch.arange(valid.shape[1], device=valid.device)[None, :] < valid_lengths[:, None]
    _require(torch.equal(valid, expected_prefix), "real loader masks must be contiguous valid prefixes")
    lengths = [int(value) for value in valid_lengths.detach().cpu().tolist()]
    _require(all(0 < value <= int(dense_window_size) for value in lengths), "invalid real-loader validity length")
    full = [value == int(dense_window_size) for value in lengths]
    if all(full):
        return "full", lengths
    if not any(full):
        return "all_short", lengths
    return "mixed", lengths


def _video_path(dataset, index: int) -> Path:
    video_name = str(dataset.data_list[int(index)][0])
    return _path(Path(dataset.data_path) / f"{video_name}.mp4")


def _load_planned_batch(
    dataset,
    cfg: Config,
    *,
    indices: list[int],
    expected_pattern: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(len(indices) == int(cfg.solver.train.batch_size), "planned subset must fill one configured batch")
    video_paths = [_video_path(dataset, index) for index in indices]
    missing = [str(path) for path in video_paths if not path.is_file()]
    _require(not missing, "planned real THUMOS videos are missing", evidence={"missing_video_paths": missing})
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    subset = Subset(dataset, indices)
    loader = build_dataloader(
        subset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=True,
        **copy.deepcopy(cfg.solver.train),
    )
    _require(len(loader) == 1, "planned real-loader subset must produce exactly one batch")
    batch = next(iter(loader))
    observed_pattern, valid_lengths = _classify_validity_pattern(batch["masks"], int(cfg.dense_window_size))
    _require(
        observed_pattern == expected_pattern,
        "annotation-planned validity pattern did not match decoded real masks",
        evidence={
            "expected_pattern": expected_pattern,
            "observed_pattern": observed_pattern,
            "valid_lengths": valid_lengths,
            "indices": indices,
        },
    )
    inputs = batch.get("inputs")
    _require(torch.is_tensor(inputs) and inputs.ndim in {5, 6}, "decoded real inputs have an invalid shape")
    temporal_dim = 3 if inputs.ndim == 6 else 2
    _require(int(inputs.shape[temporal_dim]) == 768, "decoded real batch is not dense T=768")
    metas = batch.get("metas")
    _require(isinstance(metas, list) and len(metas) == len(indices), "real batch metas are malformed")
    video_names = [str(meta.get("video_name", "")) for meta in metas]
    _require(all(video_names), "real batch metadata lacks video names")
    return batch, {
        "loader_class": loader.__class__.__name__,
        "dataset_class": dataset.__class__.__name__,
        "subset_wrapper_class": subset.__class__.__name__,
        "uses_opentad_build_dataloader": True,
        "shuffle": False,
        "drop_last": True,
        "batch_size": int(cfg.solver.train.batch_size),
        "num_workers": int(cfg.solver.train.num_workers),
        "indices": [int(index) for index in indices],
        "video_names": video_names,
        "video_paths": [str(path) for path in video_paths],
        "video_size_bytes": [int(path.stat().st_size) for path in video_paths],
        "input_shape": [int(value) for value in inputs.shape],
        "input_dtype": str(inputs.dtype),
        "valid_lengths": valid_lengths,
        "observed_pattern": observed_pattern,
    }


def _cuda_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    required = {"inputs", "masks", "metas", "gt_segments", "gt_labels"}
    _require(required.issubset(batch), "real train batch is missing required fields")
    return {
        "inputs": batch["inputs"].to(device=device, non_blocking=True),
        "masks": batch["masks"].to(device=device, dtype=torch.bool, non_blocking=True),
        "metas": [dict(meta) for meta in batch["metas"]],
        "gt_segments": [value.to(device=device, non_blocking=True) for value in batch["gt_segments"]],
        "gt_labels": [value.to(device=device, non_blocking=True) for value in batch["gt_labels"]],
    }


def _audit_selector_real_batch(
    selector,
    batch: Mapping[str, Any],
    *,
    dense_window_size: int,
    budget: int,
    expected_asformer_source: Path,
    expected_asformer_normalized_hash: str,
    require_observed_deformation: bool,
) -> dict[str, Any]:
    selector.eval()
    with torch.no_grad():
        outputs = selector.forward_train(**batch)
    scores = outputs["selector_outputs"]
    provenance = scores.get("online_actionness_provenance")
    _require(isinstance(provenance, Mapping), "real selector forward lacks online ASFormer provenance")
    observed_source = _path(provenance.get("official_source_file", ""))
    _require(observed_source == expected_asformer_source, "selector did not execute the bound official ASFormer source")
    _require(
        provenance.get("official_source_normalized_lf_sha256") == expected_asformer_normalized_hash,
        "runtime ASFormer source hash does not match the bound source",
    )
    _require(
        provenance.get("hidden_kind") == "official_asformer_encoder_hidden",
        "selector did not consume official ASFormer encoder hidden features",
    )

    grid = scores.get("grid")
    detector_grid = scores.get("detector_grid_positions")
    _require(grid is not None and torch.is_tensor(detector_grid), "selector lacks local-cell grid evidence")
    samples = []
    total_deformations = 0
    total_gt = 0
    for index, meta in enumerate(outputs["metas"]):
        valid_len = int(batch["masks"][index].long().sum().item())
        effective_k = min(int(budget), valid_len)
        acquisition = grid.selected_positions[index, :effective_k].detach().cpu().long()
        anchors, starts, ends = exact_uniform_cell_bounds(valid_len, effective_k)
        detector_positions = detector_grid[index, :effective_k].detach().cpu().long()
        _require(torch.equal(detector_positions, anchors), "detector grid is not the fixed exact-uniform anchor grid")
        _require(torch.all(acquisition >= starts).item(), "acquisition fell below its local cell")
        _require(torch.all(acquisition < ends).item(), "acquisition exceeded its local cell")
        _require(int(outputs["masks"][index].long().sum().item()) == effective_k, "selected mask violates effective K")

        acquisition_values = [int(value) for value in acquisition.tolist()]
        detector_values = [int(value) for value in detector_positions.tolist()]
        _require(meta.get("duca_acquisition_positions") == acquisition_values, "acquisition metadata drifted")
        _require(meta.get("duca_detector_grid_positions") == detector_values, "detector-grid metadata drifted")
        _require(
            meta.get("selected_axis_to_true_time_dense_index") == detector_values,
            "detector true-time mapping must use fixed anchors",
        )
        remap_meta = meta.get("duca_online_selected_axis_remap")
        _require(isinstance(remap_meta, Mapping), "selected-axis remap metadata is missing")
        _require(remap_meta.get("acquisition_positions") == acquisition_values, "remap lost acquisition positions")
        _require(
            remap_meta.get("selected_axis_to_true_time_dense_index") == detector_values,
            "remap does not bind detector anchors",
        )

        original_gt = batch["gt_segments"][index]
        remapped_gt = outputs["gt_segments"][index]
        time_map = TrueTimeMap(detector_values, dense_len=dense_window_size, valid_len=valid_len)
        expected_gt = time_map.remap_segments(
            original_gt,
            source_coordinate_space=TRUE_TIME_AXIS,
            target_coordinate_space=SELECTED_AXIS,
        ).to(dtype=remapped_gt.dtype)
        _require(
            torch.allclose(remapped_gt.detach().cpu(), expected_gt.detach().cpu(), atol=1.0e-5, rtol=0.0),
            "real GT remap does not follow the fixed detector anchor grid",
        )
        _require(meta.get("gt_remapped_to_selected_axis") is True, "real GT remap was not recorded")
        _require(meta.get("gt_segments_original_time") == original_gt.detach().cpu().tolist(), "original GT audit drifted")
        _require(
            torch.allclose(
                torch.as_tensor(meta.get("gt_segments_selected_axis")),
                remapped_gt.detach().cpu(),
                atol=1.0e-5,
                rtol=0.0,
            ),
            "selected-axis GT metadata drifted",
        )

        deformation_count = int((acquisition != detector_positions).sum().item())
        total_deformations += deformation_count
        total_gt += int(original_gt.shape[0])
        samples.append(
            {
                "video_name": str(meta.get("video_name")),
                "valid_len": valid_len,
                "effective_k": effective_k,
                "gt_segment_count": int(original_gt.shape[0]),
                "gt_remap_verified_against_detector_grid": True,
                "acquisition_and_detector_fields_separate": True,
                "acquisition_position_sha256": _position_sha256(acquisition_values),
                "detector_grid_position_sha256": _position_sha256(detector_values),
                "acquisition_vs_detector_difference_count": deformation_count,
                "detector_grid_matches_exact_uniform": True,
                "acquisition_stays_inside_local_cells": True,
            }
        )
    _require(total_gt > 0, "real batch contains no GT segments to remap")
    if require_observed_deformation:
        _require(
            total_deformations > 0,
            "a trainable full-window CellCF update did not produce any acquisition/anchor deformation",
        )
    request = outputs.get("counterfactual_request")
    candidate_count = 0
    if isinstance(request, Mapping) and torch.is_tensor(request.get("candidate_valid")):
        candidate_count = int(request["candidate_valid"].sum().item())
    return {
        "executed": True,
        "real_gt_remap_verified": True,
        "actual_acquisition_positions_recorded_separately": True,
        "fixed_detector_grid_anchors_verified": True,
        "observed_acquisition_anchor_deformation": total_deformations > 0,
        "acquisition_anchor_difference_count": total_deformations,
        "selector_only_counterfactual_candidate_count": candidate_count,
        "official_asformer_runtime_source_verified": True,
        "official_asformer_runtime_normalized_lf_sha256": provenance.get(
            "official_source_normalized_lf_sha256"
        ),
        "samples": samples,
    }


def _build_real_model(cfg: Config, logger: logging.Logger):
    model = build_detector(copy.deepcopy(cfg.model))
    _require(model.__class__.__name__ == "ActionFormer", "built detector is not ActionFormer")
    _require(model.rpn_head.__class__.__name__ == "ActionFormerHead", "built detector head is not ActionFormerHead")
    _require(model.backbone.__class__.__name__ == "BackboneWrapper", "gate did not build the real backbone wrapper")
    recognizer = getattr(model.backbone, "model", None)
    _require(recognizer is not None and recognizer.__class__.__name__ == "Recognizer3D", "real mmaction Recognizer3D is missing")
    videomae = getattr(recognizer, "backbone", None)
    _require(videomae is not None and videomae.__class__.__name__ == "VisionTransformerAdapter", "real VideoMAE adapter is missing")
    prepare_optimizer_parameter_freezing(copy.deepcopy(cfg.optimizer), model, logger)
    return model


def _transition_output_weight(model) -> tuple[str, torch.nn.Parameter]:
    matches = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.endswith("frame_selector.adapter.transition_scorer.net.3.weight")
    ]
    _require(len(matches) == 1, "could not bind the CellCF transition scorer output weight")
    return matches[0]


def _position_scheduler_at_successful_step(scheduler, *, step: int) -> list[float]:
    _require(step > 0, "scheduler proof position must be positive")
    scheduler.last_epoch = int(step)
    lrs = [float(value) for value in scheduler._get_closed_form_lr()]
    _require(len(lrs) == len(scheduler.optimizer.param_groups), "scheduler LR group count drifted")
    _require(any(value > 0.0 for value in lrs), "scheduler proof position must have a non-zero LR")
    for group, lr in zip(scheduler.optimizer.param_groups, lrs):
        group["lr"] = lr
    scheduler._last_lr = list(lrs)
    scheduler._step_count = max(int(getattr(scheduler, "_step_count", 0)), 1)
    return lrs


class _OneCudaBatchLoader:
    def __init__(self, batch: Mapping[str, Any]) -> None:
        self.batch = dict(batch)

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        yield self.batch


def _optimizer_state_step(optimizer, parameter: torch.nn.Parameter) -> int:
    state = optimizer.state.get(parameter)
    _require(isinstance(state, Mapping) and "step" in state, "optimizer did not create state for the proof parameter")
    value = state["step"]
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    return int(value)


def _verify_counterfactual_summary(selector) -> dict[str, Any]:
    summary = getattr(selector, "last_counterfactual_summary", None)
    _require(isinstance(summary, Mapping), "integrated local counterfactual teacher produced no summary")
    _require(
        summary.get("teacher_kind")
        == "detached_distinct_local_cell_hard_flip_official_actionformer_cls_plus_reg",
        "counterfactual teacher is not the local official ActionFormer cls+reg teacher",
    )
    _require(
        summary.get("distillation_loss_kind") == "distinct_local_cell_weighted_signed_logistic",
        "counterfactual learner is not the local signed logistic objective",
    )
    candidate_count = int(summary.get("candidate_count", 0))
    utilities = [float(value) for value in summary.get("candidate_utility_values", [])]
    _require(candidate_count > 0 and len(utilities) == candidate_count, "local counterfactual teacher produced no valid hard flips")
    _require(all(math.isfinite(value) for value in utilities), "counterfactual utilities are non-finite")
    _require(any(value != 0.0 for value in utilities), "counterfactual utilities are all zero")
    _require(summary.get("finite") is True, "counterfactual teacher did not declare finite utility")
    _require(summary.get("direct_detector_gradient") is False, "counterfactual teacher leaked a direct detector gradient")
    consistency = float(summary.get("utility_consistency_max_abs_error", float("inf")))
    _require(
        math.isfinite(consistency) and consistency <= 1.0e-6,
        "counterfactual utility is not baseline detector loss minus candidate loss",
    )
    _require(float(summary.get("no_op_teacher_utility", float("nan"))) == 0.0, "no-op teacher utility drifted")
    _require(float(summary.get("no_op_student_score_delta", float("nan"))) == 0.0, "no-op score delta drifted")
    return {
        "teacher_kind": str(summary["teacher_kind"]),
        "detector_objective": "official_actionformer_cls_plus_reg",
        "distillation_loss_kind": str(summary["distillation_loss_kind"]),
        "candidate_count": candidate_count,
        "candidate_utility_positive_count": int(summary.get("candidate_utility_positive_count", 0)),
        "candidate_utility_negative_count": int(summary.get("candidate_utility_negative_count", 0)),
        "candidate_utility_zero_count": int(summary.get("candidate_utility_zero_count", 0)),
        "candidate_utility_min": min(utilities),
        "candidate_utility_max": max(utilities),
        "utilities_finite": True,
        "at_least_one_nonzero_utility": True,
        "utility_consistency_max_abs_error": consistency,
        "direct_detector_gradient": False,
        "local_signed_counterfactual_teacher_verified": True,
    }


def _verify_training_update(
    *,
    cfg: Config,
    model: DistributedDataParallel,
    optimizer,
    scheduler,
    model_ema: ModelEma,
    scaler: torch.cuda.amp.GradScaler,
    batch: Mapping[str, Any],
    logger: logging.Logger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = model.module
    selector = root.frame_selector
    schedule_seed_step = 1
    initial_lrs = _position_scheduler_at_successful_step(scheduler, step=schedule_seed_step)
    selector._loss_weight_schedule_step.fill_(schedule_seed_step)
    selector._pending_loss_schedule_advance = False

    parameter_name, parameter = _transition_output_weight(root)
    ema_root = getattr(model_ema.module, "module", model_ema.module)
    ema_parameter = dict(ema_root.named_parameters())[parameter_name]
    parameter_before = parameter.detach().clone()
    ema_before = ema_parameter.detach().clone()
    schedule_before = int(selector._loss_weight_schedule_step.item())
    scheduler_before = int(scheduler.last_epoch)
    scale_before = float(scaler.get_scale())
    update_audit: dict[str, Any] = {}
    probe = train_one_epoch(
        _OneCudaBatchLoader(batch),
        model,
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
        force_amp_overflow_attempts=1,
        update_audit=update_audit,
    )
    expected_audit = {
        "attempted_batches": 1,
        "optimizer_attempts": 2,
        "successful_optimizer_updates": 1,
        "amp_skipped_attempts": 1,
        "replayed_batches": 1,
        "replay_exhaustions": 0,
        "scheduler_updates": 1,
        "ema_updates": 1,
        "duca_schedule_updates": 1,
        "forced_amp_overflow_attempts": 1,
        "max_amp_retries_observed": 1,
        "replay_state_restorations": 1,
    }
    drift = {
        key: {"expected": expected, "observed": update_audit.get(key)}
        for key, expected in expected_audit.items()
        if int(update_audit.get(key, -1)) != expected
    }
    _require(not drift, "AMP replay/update audit drifted", evidence={"update_audit_drift": drift})
    _require(isinstance(probe, Mapping), "training engine did not return its update probe")
    _require(int(probe.get("attempted_steps", 0)) == 2, "training probe did not see replayed backward attempts")
    _require(int(probe.get("skipped_optimizer_steps", 0)) == 1, "training probe did not see forced overflow")
    _require(int(probe.get("successful_optimizer_steps", 0)) == 1, "training probe did not see one success")
    _require(int(probe.get("finite_loss_steps", 0)) == 2, "both replayed forwards must have finite losses")
    _require(int(probe.get("finite_gradient_steps", 0)) == 1, "only the successful replay may have finite gradients")
    for group in ("backbone", "coarse_probe", "selector", "projection", "neck", "detector_head"):
        counts = probe.get("parameter_group_coverage", {}).get(group, {})
        _require(int(counts.get("gradient_seen", 0)) > 0, f"successful update lacks {group} gradients")

    history = list(update_audit.get("amp_scale_history", []))
    _require(len(history) == 2, "AMP scale history must contain the overflow and replay")
    _require(history[0].get("optimizer_step_ran") is False, "forced overflow attempt was not skipped")
    _require(history[1].get("optimizer_step_ran") is True, "replayed attempt did not update")
    _require(float(history[0]["after"]) < float(history[0]["before"]), "forced overflow did not reduce GradScaler scale")
    _require(float(history[1]["after"]) >= float(history[1]["before"]), "successful replay was misclassified")

    schedule_after = int(selector._loss_weight_schedule_step.item())
    scheduler_after = int(scheduler.last_epoch)
    _require(schedule_after == schedule_before + 1, "DUCA schedule did not advance exactly once")
    _require(scheduler_after == scheduler_before + 1, "LR scheduler did not advance exactly once")
    parameter_after = parameter.detach().clone()
    ema_after = ema_parameter.detach().clone()
    parameter_delta = float((parameter_after - parameter_before).abs().max().item())
    ema_delta = float((ema_after - ema_before).abs().max().item())
    gradient_sum = float(parameter.grad.detach().abs().sum().item()) if parameter.grad is not None else 0.0
    _require(parameter_delta > 0.0, "successful optimizer step did not change the CellCF scorer")
    _require(gradient_sum > 0.0 and math.isfinite(gradient_sum), "CellCF scorer gradient is missing or non-finite")
    _require(ema_delta > 0.0, "EMA update did not change the CellCF scorer shadow")
    expected_ema = model_ema.decay * ema_before + (1.0 - model_ema.decay) * parameter_after
    ema_error = float((ema_after - expected_ema).abs().max().item())
    _require(ema_error <= 1.0e-7, "EMA shadow does not match the configured update formula")
    optimizer_step = _optimizer_state_step(optimizer, parameter)
    _require(optimizer_step == 1, "proof parameter optimizer state did not advance exactly once")

    counterfactual = _verify_counterfactual_summary(selector)
    update = {
        "amp_enabled": True,
        "grad_scaler_enabled": bool(scaler.is_enabled()),
        "forced_overflow_attempts": 1,
        "overflow_attempt_skipped": True,
        "same_batch_replayed": update_audit.get("attempt_batch_indices") == [0, 0],
        "replay_state_restored": int(update_audit["replay_state_restorations"]) == 1,
        "successful_optimizer_updates": 1,
        "ema_updates": 1,
        "scheduler_updates": 1,
        "duca_schedule_updates": 1,
        "schedule_seed_step": schedule_seed_step,
        "duca_schedule_step_before": schedule_before,
        "duca_schedule_step_after": schedule_after,
        "lr_scheduler_step_before": scheduler_before,
        "lr_scheduler_step_after": scheduler_after,
        "initial_nonzero_scheduler_lrs": initial_lrs,
        "grad_scale_before_gate": scale_before,
        "grad_scale_history": history,
        "grad_scale_after_gate": float(scaler.get_scale()),
        "proof_parameter": parameter_name,
        "proof_parameter_gradient_abs_sum": gradient_sum,
        "proof_parameter_max_abs_change": parameter_delta,
        "proof_parameter_optimizer_state_step": optimizer_step,
        "ema_parameter_max_abs_change": ema_delta,
        "ema_formula_max_abs_error": ema_error,
        "optimizer_update_verified": True,
        "ema_update_verified": True,
        "schedule_update_verified": True,
        "update_audit": update_audit,
        "training_probe": dict(probe),
    }
    return update, counterfactual


def _audited_hashes() -> dict[str, str]:
    missing = [path for path in AUDITED_PATHS if not (ROOT / path).is_file()]
    _require(not missing, "audited implementation files are missing", evidence={"missing_audited_paths": missing})
    return {path: _sha256(ROOT / path) for path in AUDITED_PATHS}


def _verify_final_clean_binding(
    initial_git: Mapping[str, Any],
    initial_hashes: Mapping[str, str],
) -> dict[str, Any]:
    final_head = _git_output("rev-parse", "--verify", "HEAD")
    final_tree = _git_output("rev-parse", "HEAD^{tree}")
    final_status = _git_output("status", "--porcelain", "--untracked-files=normal")
    _require(
        final_head == initial_git.get("git_commit") and final_tree == initial_git.get("git_tree"),
        "Git commit or tree changed while the CUDA gate was running",
        evidence={
            "initial_commit": initial_git.get("git_commit"),
            "final_commit": final_head,
            "initial_tree": initial_git.get("git_tree"),
            "final_tree": final_tree,
        },
    )
    _require(
        not final_status,
        "the worktree became dirty while the CUDA gate was running",
        evidence={"final_git_status_porcelain": final_status.splitlines()},
    )
    final_hashes = _audited_hashes()
    changed = {
        path: {"initial": initial_hashes.get(path), "final": final_hashes.get(path)}
        for path in sorted(set(initial_hashes) | set(final_hashes))
        if initial_hashes.get(path) != final_hashes.get(path)
    }
    _require(
        not changed,
        "audited files changed while the CUDA gate was running",
        evidence={"changed_audited_files": changed},
    )
    return {
        "git_commit_unchanged": True,
        "git_tree_unchanged": True,
        "git_tree_clean_after_gate": True,
        "audited_hashes_unchanged": True,
    }


def _logger() -> logging.Logger:
    logger = logging.getLogger("duca-cellcf-real-loader-cuda-gate")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def run_gate(
    *,
    expected_commit: str,
    synthetic_gate_json: str | Path,
    videomae_checkpoint: str | Path,
    expected_videomae_sha256: str,
    official_repos_root: str | Path,
    config_path: str | Path = CONFIG_DEFAULT,
    device: str = "cuda:0",
) -> dict[str, Any]:
    git = _bind_clean_commit(expected_commit)
    synthetic = _bind_synthetic_gate(synthetic_gate_json, git_commit=git["git_commit"])
    cuda_binding = _bind_slurm_logical_cuda(device)
    config_file = _config_path(config_path)
    _require(config_file.is_file(), f"CellCF main config is missing: {config_file}")
    cfg = Config.fromfile(str(config_file))
    config = _validate_main_config(cfg, config_file)
    _require(
        synthetic["training_profile"] == config["training_profile"],
        "synthetic and real-loader gates use different training profiles",
    )
    assets = _bind_external_assets(
        cfg,
        videomae_checkpoint=videomae_checkpoint,
        expected_videomae_sha256=expected_videomae_sha256,
        official_repos_root=official_repos_root,
    )
    dataset_binding = _dataset_bindings(cfg)
    audited_hashes = _audited_hashes()

    random.seed(GATE_SEED)
    np.random.seed(GATE_SEED)
    torch.manual_seed(GATE_SEED)
    torch.cuda.manual_seed_all(GATE_SEED)
    torch.cuda.reset_peak_memory_stats(0)
    logger = _logger()
    torch_device = torch.device("cuda:0")

    with _single_process_nccl_group() as process_group:
        train_dataset = build_dataset(copy.deepcopy(cfg.dataset.train), default_args={"logger": logger})
        _require(train_dataset.__class__.__name__ == "ThumosPaddingDataset", "real train dataset is not ThumosPaddingDataset")
        full_train_loader = build_dataloader(
            train_dataset,
            rank=0,
            world_size=1,
            shuffle=True,
            drop_last=True,
            **copy.deepcopy(cfg.solver.train),
        )
        _require(
            len(full_train_loader) == int(cfg.workflow.expected_train_batches_per_epoch),
            "real train loader length does not match the frozen schedule contract",
            evidence={
                "observed_train_batches": len(full_train_loader),
                "expected_train_batches": int(cfg.workflow.expected_train_batches_per_epoch),
            },
        )
        coverage = _plan_validity_coverage(
            train_dataset,
            dense_window_size=int(cfg.dense_window_size),
            batch_size=int(cfg.solver.train.batch_size),
        )
        update_pattern = "mixed" if coverage["mixed"]["obtainable"] else "full"
        _require(
            coverage[update_pattern]["obtainable"],
            "no real full-window sample is available to prove a nontrivial local counterfactual teacher",
            evidence={"coverage_plan": coverage},
        )

        root_model = _build_real_model(cfg, logger).to(torch_device)
        ddp_model = DistributedDataParallel(
            root_model,
            device_ids=[0],
            output_device=0,
            find_unused_parameters=bool(cfg.solver.find_unused_parameters),
            static_graph=bool(cfg.solver.static_graph),
        )
        ddp_model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
        root_model.eval()
        update_batch: dict[str, Any] | None = None
        preupdate_selector_audits: dict[str, Any] = {}
        expected_source = _path(assets["official_asformer_source"]["path"])
        expected_source_hash = str(assets["official_asformer_source"]["normalized_lf_sha256"])
        for offset, pattern in enumerate(("full", "mixed", "all_short")):
            plan = coverage[pattern]
            if not plan["obtainable"]:
                plan["coverage_status"] = "not_obtainable_from_real_train_annotation_index"
                continue
            cpu_batch, loader_evidence = _load_planned_batch(
                train_dataset,
                cfg,
                indices=list(plan["indices"]),
                expected_pattern=pattern,
                seed=GATE_SEED + offset,
            )
            cuda_batch = _cuda_batch(cpu_batch, torch_device)
            selector_audit = _audit_selector_real_batch(
                root_model.frame_selector,
                cuda_batch,
                dense_window_size=int(cfg.dense_window_size),
                budget=int(cfg.window_size),
                expected_asformer_source=expected_source,
                expected_asformer_normalized_hash=expected_source_hash,
                require_observed_deformation=False,
            )
            plan.update(loader_evidence)
            plan["executed"] = True
            plan["coverage_status"] = "executed_real_loader_selector_forward"
            plan["selector_audit"] = selector_audit
            preupdate_selector_audits[pattern] = selector_audit
            if pattern == update_pattern:
                update_batch = cuda_batch
            else:
                del cuda_batch
                torch.cuda.empty_cache()
        _require(
            all((not value["obtainable"]) or value["executed"] for key, value in coverage.items() if key != "annotation_inventory"),
            "an obtainable validity pattern was not exercised",
        )
        _require(update_batch is not None, "real update batch was not retained")

        model_ema = ModelEma(ddp_model)
        optimizer = build_optimizer(copy.deepcopy(cfg.optimizer), ddp_model, logger)
        scheduler, scheduler_max_epoch = build_scheduler(
            copy.deepcopy(cfg.scheduler),
            optimizer,
            len(full_train_loader),
        )
        _require(
            int(scheduler_max_epoch) == int(cfg.workflow.end_epoch),
            "built scheduler max epoch drifted",
        )
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        update, counterfactual = _verify_training_update(
            cfg=cfg,
            model=ddp_model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_ema=model_ema,
            scaler=scaler,
            batch=update_batch,
            logger=logger,
        )
        ddp_model.eval()
        postupdate_geometry = _audit_selector_real_batch(
            root_model.frame_selector,
            update_batch,
            dense_window_size=int(cfg.dense_window_size),
            budget=int(cfg.window_size),
            expected_asformer_source=expected_source,
            expected_asformer_normalized_hash=expected_source_hash,
            require_observed_deformation=True,
        )
        update_video_names = [str(meta.get("video_name")) for meta in update_batch["metas"]]
        model_binding = {
            "model_type": root_model.__class__.__name__,
            "backbone_wrapper_type": root_model.backbone.__class__.__name__,
            "mmaction_recognizer_type": root_model.backbone.model.__class__.__name__,
            "videomae_backbone_type": root_model.backbone.model.backbone.__class__.__name__,
            "detector_head_type": root_model.rpn_head.__class__.__name__,
            "selector_type": root_model.frame_selector.__class__.__name__,
            "acquisition_policy": str(root_model.frame_selector.acquisition_policy),
            "counterfactual_objective": str(root_model.frame_selector.counterfactual_objective),
            "ddp_type": ddp_model.__class__.__name__,
            "ddp_static_graph": bool(cfg.solver.static_graph),
            "ddp_find_unused_parameters": bool(cfg.solver.find_unused_parameters),
            "ddp_fp16_compress_hook_registered": True,
            "official_videomae_checkpoint_loaded_during_model_build": True,
        }
        del update_batch, model_ema, optimizer, scheduler, scaler, ddp_model, root_model
        torch.cuda.empty_cache()

    obtainable_patterns = [
        key for key in ("full", "mixed", "all_short") if coverage[key]["obtainable"]
    ]
    executed_patterns = [
        key for key in ("full", "mixed", "all_short") if coverage[key]["executed"]
    ]
    final_clean_binding = _verify_final_clean_binding(git, audited_hashes)
    result = {
        "schema": SCHEMA,
        "ok": True,
        "fail_closed": True,
        "git_commit": git["git_commit"],
        "git_tree_clean": True,
        "git_binding": git,
        "final_clean_binding": final_clean_binding,
        "synthetic_gate_path": synthetic["path"],
        "synthetic_gate_sha256": synthetic["sha256"],
        "synthetic_gate_schema": synthetic["schema"],
        "synthetic_gate_binding": synthetic,
        "audited_file_sha256": audited_hashes,
        "config_path": config["path"],
        "config_sha256": config["sha256"],
        "config_contract": config,
        "device": "cuda:0",
        "slurm_cuda_binding": cuda_binding,
        "process_group": process_group,
        "assets": assets,
        "input_provenance": "real_thumos14_train_video_decode_from_cellcf_main_config",
        "real_dataset_loader_executed": True,
        "synthetic_inputs_used": False,
        "dataset": {
            **dataset_binding,
            "dataset_size": int(coverage["annotation_inventory"]["dataset_size"]),
            "train_batches_per_epoch": int(config["expected_train_batches_per_epoch"]),
            "loader_builder": "opentad.datasets.build_dataloader",
            "train_pipeline_from_main_config": True,
        },
        "validity_window_coverage": {
            "obtainable_patterns": obtainable_patterns,
            "executed_patterns": executed_patterns,
            "all_obtainable_patterns_executed": set(obtainable_patterns) == set(executed_patterns),
            "full": coverage["full"],
            "mixed": coverage["mixed"],
            "all_short": coverage["all_short"],
            "annotation_inventory": coverage["annotation_inventory"],
        },
        "model": model_binding,
        "update_batch_pattern": update_pattern,
        "update_batch_video_names": update_video_names,
        "preupdate_selector_audits": preupdate_selector_audits,
        "postupdate_real_gt_and_coordinate_audit": postupdate_geometry,
        "local_signed_counterfactual_teacher": counterfactual,
        "amp_replay_and_successful_update": update,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "claims": {
            "offline_tad": True,
            "online_tad": False,
            "fixed_k384": True,
            "official_adatad_actionformer_semantics_preserved": True,
            "training_profile": str(config["training_profile"]),
            "training_profile_preserved": True,
            "checkpoint_every_5_preserved": True,
            "real_loader_cuda_gate_passed": True,
            "all_obtainable_validity_patterns_exercised": set(obtainable_patterns) == set(executed_patterns),
            "real_gt_remap_verified": True,
            "actual_acquisition_separate_from_fixed_detector_grid": True,
            "local_signed_counterfactual_teacher_verified": True,
            "forced_amp_overflow_replay_verified": True,
            "one_successful_optimizer_ema_schedule_update_verified": True,
            "metric_claim_allowed": False,
            "paper_ready": False,
        },
    }
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_real_loader_gate_artifact(
    path: str | Path,
    *,
    expected_commit: str,
    expected_sha256: str | None = None,
    require_clean: bool = False,
) -> dict[str, Any]:
    """Revalidate a gate artifact and every file-backed claim it carries."""
    artifact_path = _path(path)
    _require(artifact_path.is_file(), f"real-loader gate artifact is missing: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "real-loader gate artifact must be a JSON object")
    artifact_sha256 = _sha256(artifact_path)
    if expected_sha256 is not None:
        _require(artifact_sha256 == str(expected_sha256), "real-loader gate artifact SHA-256 mismatch")
    _require(payload.get("schema") == SCHEMA, "real-loader gate artifact schema mismatch")
    _require(payload.get("ok") is True and payload.get("fail_closed") is True, "real-loader gate did not pass fail closed")
    _require(payload.get("git_commit") == expected_commit, "real-loader gate artifact is stale")
    _require(payload.get("git_tree_clean") is True, "real-loader gate was not produced from a clean tree")
    if require_clean:
        _require(_git_output("rev-parse", "HEAD") == expected_commit, "current checkout differs from the gate commit")
        _require(not _git_output("status", "--porcelain", "--untracked-files=normal"), "current checkout is dirty")

    final_binding = payload.get("final_clean_binding")
    _require(isinstance(final_binding, Mapping), "real-loader gate lacks its final clean binding")
    for key in (
        "git_commit_unchanged",
        "git_tree_unchanged",
        "git_tree_clean_after_gate",
        "audited_hashes_unchanged",
    ):
        _require(final_binding.get(key) is True, f"real-loader gate final binding failed: {key}")

    audited = payload.get("audited_file_sha256")
    _require(isinstance(audited, Mapping), "real-loader gate lacks audited file hashes")
    _require(set(audited) == set(AUDITED_PATHS), "real-loader gate audited surface drifted")
    for relative_path in AUDITED_PATHS:
        _require(audited.get(relative_path) == _sha256(ROOT / relative_path), f"audited file hash drifted: {relative_path}")

    synthetic_path = _path(payload.get("synthetic_gate_path", ""))
    _require(synthetic_path.is_file(), "bound synthetic gate is missing")
    _require(payload.get("synthetic_gate_sha256") == _sha256(synthetic_path), "bound synthetic gate hash drifted")
    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
    _require(synthetic.get("schema") == SYNTHETIC_GATE_SCHEMA and synthetic.get("ok") is True, "bound synthetic gate is invalid")
    _require(synthetic.get("git_commit") == expected_commit, "bound synthetic gate is stale")
    _require(synthetic.get("real_dataset_loader_executed") is False, "synthetic gate provenance is dishonest")

    config_path = _path(payload.get("config_path", ""))
    _require(config_path == _path(ROOT / CONFIG_DEFAULT), "real-loader gate used another config")
    _require(payload.get("config_sha256") == _sha256(config_path), "real-loader gate config hash drifted")
    assets = payload.get("assets")
    _require(isinstance(assets, Mapping), "real-loader gate lacks external asset bindings")
    for key in ("videomae_checkpoint", "official_asformer_source"):
        asset = assets.get(key)
        _require(isinstance(asset, Mapping), f"real-loader gate lacks {key} binding")
        asset_path = _path(asset.get("path", ""))
        _require(asset_path.is_file(), f"bound {key} is missing")
        _require(asset.get("sha256") == _sha256(asset_path), f"bound {key} hash drifted")

    slurm = payload.get("slurm_cuda_binding")
    _require(isinstance(slurm, Mapping), "real-loader gate lacks Slurm CUDA binding")
    _require(str(slurm.get("slurm_job_id", "")).isdigit(), "real-loader gate lacks a numeric Slurm job id")
    _require(slurm.get("logical_device") == "cuda:0", "real-loader gate did not use logical cuda:0")
    _require(slurm.get("logical_cuda_device_count") == 1, "real-loader gate did not use one logical GPU")
    _require(slurm.get("physical_gpu_index_assumed") is False, "real-loader gate assumed a physical GPU index")

    _require(payload.get("real_dataset_loader_executed") is True, "real THUMOS loader was not executed")
    _require(payload.get("synthetic_inputs_used") is False, "real-loader gate used synthetic inputs")
    dataset = payload.get("dataset")
    _require(isinstance(dataset, Mapping), "real-loader gate lacks dataset evidence")
    for path_key, hash_key in (("annotation_path", "annotation_sha256"), ("class_map_path", "class_map_sha256")):
        bound_path = _path(dataset.get(path_key, ""))
        _require(bound_path.is_file(), f"gate dataset artifact is missing: {path_key}")
        _require(dataset.get(hash_key) == _sha256(bound_path), f"gate dataset hash drifted: {hash_key}")
    _require(dataset.get("loader_builder") == "opentad.datasets.build_dataloader", "gate used another data loader")
    _require(dataset.get("train_pipeline_from_main_config") is True, "gate data pipeline drifted")

    coverage = payload.get("validity_window_coverage")
    _require(isinstance(coverage, Mapping), "real-loader gate lacks validity coverage")
    _require(coverage.get("all_obtainable_patterns_executed") is True, "gate skipped an obtainable validity pattern")
    _require(set(coverage.get("obtainable_patterns", ())) == set(coverage.get("executed_patterns", ())), "validity coverage evidence is inconsistent")

    teacher = payload.get("local_signed_counterfactual_teacher")
    _require(isinstance(teacher, Mapping), "real-loader gate lacks local counterfactual evidence")
    _require(teacher.get("local_signed_counterfactual_teacher_verified") is True, "local counterfactual teacher was not verified")
    _require(teacher.get("direct_detector_gradient") is False, "gate unexpectedly used direct detector gradients")
    _require(teacher.get("candidate_count", 0) > 0 and teacher.get("at_least_one_nonzero_utility") is True, "gate utility evidence is uninformative")

    update = payload.get("amp_replay_and_successful_update")
    _require(isinstance(update, Mapping), "real-loader gate lacks update evidence")
    required_update = {
        "forced_overflow_attempts": 1,
        "overflow_attempt_skipped": True,
        "same_batch_replayed": True,
        "replay_state_restored": True,
        "successful_optimizer_updates": 1,
        "ema_updates": 1,
        "scheduler_updates": 1,
        "duca_schedule_updates": 1,
        "optimizer_update_verified": True,
        "ema_update_verified": True,
        "schedule_update_verified": True,
    }
    for key, expected in required_update.items():
        _require(update.get(key) == expected, f"real-loader update proof failed: {key}")

    claims = payload.get("claims")
    _require(isinstance(claims, Mapping), "real-loader gate lacks scoped claims")
    for key in (
        "offline_tad",
        "fixed_k384",
        "official_adatad_actionformer_semantics_preserved",
        "real_loader_cuda_gate_passed",
        "real_gt_remap_verified",
        "actual_acquisition_separate_from_fixed_detector_grid",
        "local_signed_counterfactual_teacher_verified",
        "forced_amp_overflow_replay_verified",
        "one_successful_optimizer_ema_schedule_update_verified",
    ):
        _require(claims.get(key) is True, f"real-loader gate claim failed: {key}")
    _require(claims.get("online_tad") is False, "real-loader gate mislabeled the task as online")
    _require(claims.get("metric_claim_allowed") is False and claims.get("paper_ready") is False, "gate overclaimed evidence")
    return {
        "path": str(artifact_path),
        "sha256": artifact_sha256,
        "git_commit": expected_commit,
        "slurm_job_id": str(slurm["slurm_job_id"]),
        "synthetic_gate_sha256": str(payload["synthetic_gate_sha256"]),
        "dataset_annotation_sha256": str(dataset["annotation_sha256"]),
        "dataset_class_map_sha256": str(dataset["class_map_sha256"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed CellCF real-THUMOS-loader CUDA gate on Slurm logical cuda:0."
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--synthetic-gate-json", required=True)
    parser.add_argument("--videomae-checkpoint", required=True)
    parser.add_argument("--expected-videomae-sha256", required=True)
    parser.add_argument("--official-repos-root", required=True)
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    output_path: Path | None = None
    try:
        if args.output_json:
            output_path = _require_output_outside_worktree(args.output_json)
        summary = run_gate(
            expected_commit=args.expected_commit,
            synthetic_gate_json=args.synthetic_gate_json,
            videomae_checkpoint=args.videomae_checkpoint,
            expected_videomae_sha256=args.expected_videomae_sha256,
            official_repos_root=args.official_repos_root,
            config_path=args.config,
            device=args.device,
        )
    except Exception as exc:
        summary = {
            "schema": SCHEMA,
            "ok": False,
            "fail_closed": True,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "claims": {
                "real_loader_cuda_gate_passed": False,
                "metric_claim_allowed": False,
                "paper_ready": False,
            },
        }
        evidence = getattr(exc, "evidence", None)
        if evidence:
            summary["failure_evidence"] = dict(evidence)
        code = 1
    else:
        code = 0
    if output_path is not None:
        _write_json(output_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
