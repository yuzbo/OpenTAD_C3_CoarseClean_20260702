from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from mmengine.config import Config

from opentad.datasets import build_dataset
from opentad.datasets.builder import collate
from opentad.models import build_detector
from tools.bata import duca_paper_training
from tools.bata.validate_duca_paper_code_gate import validate_code_gate_artifact


SCHEMA = "duca_paper_real_short_window_heavy_backbone_gate_v1"
CONFIG_DEFAULT = (
    "configs/adatad/thumos/"
    "duca_paper_uniform_mixed_train_k384_eval_full200.py"
)
REQUESTED_BUDGETS = (192, 256, 384, 512)
EXECUTION_QUANTUM = 16
GATE_SEED = 20260803
AUDITED_PATHS = (
    "configs/adatad/thumos/duca_paper_full200_base.py",
    "configs/adatad/thumos/duca_paper_rime_selected_axis_base.py",
    "configs/adatad/thumos/duca_paper_uniform_mixed_train_k384_eval_full200.py",
    "opentad/datasets/builder.py",
    "opentad/datasets/duca_stateless.py",
    "opentad/datasets/thumos.py",
    "opentad/datasets/transforms/end_to_end.py",
    "opentad/models/backbones/backbone_wrapper.py",
    "opentad/models/detectors/actionformer.py",
    "opentad/models/detectors/single_stage.py",
    "opentad/models/duca/rime.py",
    "opentad/models/duca/structured_selection.py",
    "opentad/models/selectors/duca_protected_e2e_frame_selector.py",
    "opentad/models/selectors/duca_rime_frame_selector.py",
    "tools/bata/duca_paper_training.py",
    "tools/bata/run_duca_paper_short_window_gate.py",
    "tools/bata/validate_duca_paper_code_gate.py",
    "tools/bata/validate_duca_paper_short_window_gate.py",
)


class GateFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(f"fail-closed DUCA paper short-window gate: {message}")


def _path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _git(*args: str) -> str:
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
        "an exact 40-character commit is required",
    )
    head = _git("rev-parse", "--verify", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=normal")
    _require(head == expected, "checked-out commit drift")
    _require(not status, "gate requires a clean checkout")
    return {
        "git_commit": head,
        "git_tree": _git("rev-parse", "HEAD^{tree}"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_tree_clean": True,
    }


def _bind_slurm_cuda() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _require(job_id.isdigit(), "gate must run inside a numeric Slurm job")
    _require(bool(visible), "Slurm did not provide CUDA_VISIBLE_DEVICES")
    _require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "gate is one process")
    for name in ("RANK", "LOCAL_RANK", "SLURM_LOCALID"):
        if name in os.environ:
            _require(int(os.environ[name]) == 0, f"{name} must be zero")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(torch.cuda.device_count() == 1, "exactly one logical GPU is required")
    torch.cuda.set_device(0)
    return {
        "slurm_job_id": job_id,
        "logical_device": "cuda:0",
        "logical_cuda_device_count": 1,
        "cuda_visible_devices_supplied_by_slurm": True,
        "physical_gpu_index_assumed": False,
        "device_name": torch.cuda.get_device_name(0),
        "torch_cuda_version": torch.version.cuda,
    }


def _require_hashed_file(
    path: str | Path,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    source = _path(path)
    expected = str(expected_sha256).strip().lower()
    _require(source.is_file(), f"{label} is missing: {source}")
    _require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, f"invalid {label} hash")
    observed = _sha256(source)
    _require(observed == expected, f"{label} SHA-256 drift")
    return {
        "path": str(source),
        "sha256": observed,
        "size_bytes": int(source.stat().st_size),
    }


def _audited_hashes() -> dict[str, str]:
    missing = [name for name in AUDITED_PATHS if not (ROOT / name).is_file()]
    _require(not missing, f"audited files are missing: {missing}")
    return {name: _sha256(ROOT / name) for name in AUDITED_PATHS}


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    required = {"inputs", "masks", "metas", "gt_segments", "gt_labels"}
    _require(required.issubset(batch), "real Stage-A batch is incomplete")
    output = {
        "inputs": batch["inputs"].to(device=device, non_blocking=True),
        "masks": batch["masks"].to(
            device=device,
            dtype=torch.bool,
            non_blocking=True,
        ),
        "metas": [dict(meta) for meta in batch["metas"]],
        "gt_segments": [value.to(device=device) for value in batch["gt_segments"]],
        "gt_labels": [value.to(device=device) for value in batch["gt_labels"]],
    }
    if "gt_boundary_validity" in batch:
        output["gt_boundary_validity"] = [
            value.to(device=device) for value in batch["gt_boundary_validity"]
        ]
    return output


def _expected_effective(requested: int, valid_length: int) -> int:
    _require(valid_length >= EXECUTION_QUANTUM, "subquantum full200 sample is unsupported")
    return min(
        int(requested),
        (int(valid_length) // EXECUTION_QUANTUM) * EXECUTION_QUANTUM,
    )


def _inventory(dataset, train_data_path: Path) -> tuple[dict[str, Any], int]:
    _require(len(dataset) == 200, "Stage-A real training set is not full200")
    snippet_stride = int(getattr(dataset, "snippet_stride", 0))
    _require(snippet_stride > 0, "dataset snippet stride is invalid")
    short = []
    subquantum = []
    for index, row in enumerate(dataset.data_list):
        _require(
            len(row) >= 2 and isinstance(row[1], Mapping),
            "malformed full200 annotation index",
        )
        frame_count = int(row[1].get("frame", 0))
        _require(frame_count > 0, "full200 annotation has a non-positive frame count")
        valid_length = min(768, int(math.ceil(frame_count / snippet_stride)))
        video_name = str(row[0])
        video_path = train_data_path / f"{video_name}.mp4"
        record = {
            "index": int(index),
            "video_name": video_name,
            "annotation_valid_length": valid_length,
            "video_path": str(video_path),
            "video_exists": video_path.is_file(),
        }
        if valid_length < EXECUTION_QUANTUM:
            subquantum.append(record)
        if (valid_length // EXECUTION_QUANTUM) * EXECUTION_QUANTUM < 512:
            short.append(record)
    _require(not subquantum, "full200 contains a subquantum natural sample")
    available = [record for record in short if record["video_exists"]]
    _require(available, "no real natural short-window video is available")
    chosen = min(
        available,
        key=lambda record: (
            abs(int(record["annotation_valid_length"]) - 231),
            str(record["video_name"]),
        ),
    )
    return {
        "dataset_class": dataset.__class__.__name__,
        "dataset_size": len(dataset),
        "snippet_stride": snippet_stride,
        "natural_short_count": len(short),
        "natural_short_with_video_count": len(available),
        "subquantum_count": len(subquantum),
        "selected_sample": chosen,
    }, int(chosen["index"])


def _run_one_request(
    *,
    model,
    dataset,
    sample_index: int,
    requested_k: int,
    device: torch.device,
) -> dict[str, Any]:
    selector = model.frame_selector
    cycle = [int(value) for value in selector.mixed_k_schedule.detach().cpu().tolist()]
    cycle_index = cycle.index(int(requested_k))
    epoch = (cycle_index - int(sample_index)) % len(cycle)
    dataset.set_epoch(epoch)
    cpu_batch = collate([dataset[int(sample_index)]])
    inputs = cpu_batch.get("inputs")
    masks = cpu_batch.get("masks")
    _require(torch.is_tensor(inputs) and inputs.ndim == 6, "real RGB input is not [B,N,C,T,H,W]")
    _require(torch.is_tensor(masks) and tuple(masks.shape) == (1, 768), "real mask is not [1,768]")
    valid_length = int(masks.bool().sum().item())
    prefix = torch.arange(768)[None, :] < valid_length
    _require(torch.equal(masks.bool(), prefix), "natural-window mask is not a contiguous valid prefix")
    expected_effective = _expected_effective(requested_k, valid_length)
    batch = _move_batch(cpu_batch, device)
    _require(
        int(batch["metas"][0].get("duca_stateless_epoch", -1)) == epoch
        and int(batch["metas"][0].get("duca_stateless_sample_index", -1))
        == sample_index,
        "stateless schedule identity drift",
    )

    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
    ):
        selected = selector.forward_train(**batch)
        requested_observed = int(
            selected["selector_outputs"]["requested_k"][0].item()
        )
        positions = selector._last_selected_positions[0]
        active_positions = [int(value) for value in positions.detach().cpu().tolist()]
        features = model._forward_backbone_with_temporal_mask(
            selected["inputs"],
            selected["masks"],
        )

    contract = dict(model.backbone.last_forward_input_contract or {})
    physical_meta = dict(selector._last_physical_metas[0])
    effective = int(physical_meta.get("duca_effective_k", -1))
    unique = len(set(active_positions))
    backbone = int(contract.get("wrapper_temporal_k", -1))
    selected_temporal = int(selected["inputs"].shape[3])
    _require(requested_observed == requested_k, "mixed schedule requested another K")
    _require(effective == expected_effective, "requested-to-effective mapping drift")
    _require(
        unique == backbone == selected_temporal == expected_effective,
        "selector/gather/heavy-backbone K equality failed",
    )
    _require(
        active_positions == sorted(set(active_positions)),
        "selected positions are not ordered and unique",
    )
    _require(
        all(0 <= value < valid_length for value in active_positions),
        "selector emitted an invalid position",
    )
    _require(bool(selected["masks"].all().item()), "selected K bucket contains padding")
    _require(
        contract.get("measurement_source")
        == "actual_backbone_wrapper_and_videomae_input_tensors"
        and int(contract.get("inner_reconstructed_k", -1)) == expected_effective
        and contract.get("padding_or_repetition_observed") is False,
        "actual heavy-backbone boundary evidence drift",
    )
    _require(
        torch.is_tensor(features)
        and features.ndim == 3
        and int(features.shape[-1]) == expected_effective,
        "heavy backbone did not return the selected temporal axis",
    )
    return {
        "requested_k": requested_observed,
        "effective_k": effective,
        "unique_k": unique,
        "backbone_input_k": backbone,
        "dense_valid_length": valid_length,
        "execution_quantum": EXECUTION_QUANTUM,
        "stateless_epoch": epoch,
        "stateless_sample_index": int(sample_index),
        "mixed_k_cycle_index": cycle_index,
        "selected_dense_indices": active_positions,
        "selected_dense_indices_sha256": _canonical_sha256(active_positions),
        "selector_output_shape": [int(value) for value in selected["inputs"].shape],
        "heavy_feature_shape": [int(value) for value in features.shape],
        "backbone_input_contract": contract,
        "no_padding": True,
        "no_repetition": True,
        "no_invalid_index": True,
        "heavy_backbone_forward_completed": True,
    }


def run_gate(
    *,
    expected_commit: str,
    code_gate_receipt_path: str | Path,
    code_gate_receipt_sha256: str,
    pretrain_path: str | Path,
    pretrain_sha256: str,
    annotation_path: str | Path,
    annotation_sha256: str,
    class_map_path: str | Path,
    class_map_sha256: str,
    train_data_path: str | Path,
    config_path: str | Path = CONFIG_DEFAULT,
) -> dict[str, Any]:
    git = _bind_clean_commit(expected_commit)
    slurm = _bind_slurm_cuda()
    code_gate = validate_code_gate_artifact(
        code_gate_receipt_path,
        expected_commit=expected_commit,
        expected_sha256=code_gate_receipt_sha256,
    )
    config_file = _path(ROOT / config_path if not Path(config_path).is_absolute() else config_path)
    _require(config_file == _path(ROOT / CONFIG_DEFAULT), "gate is fixed to the Stage-A mixed-K config")
    _require(config_file.is_file(), "Stage-A mixed-K config is missing")
    cfg = Config.fromfile(str(config_file))
    static_contract = duca_paper_training.validate_static_config(cfg)
    _require(
        static_contract.get("variant") == "uniform_mixed_train_k384_eval",
        "config is not the Stage-A mixed-K arm",
    )
    pretrain = _require_hashed_file(pretrain_path, pretrain_sha256, "VideoMAE initialization")
    annotation = _require_hashed_file(annotation_path, annotation_sha256, "THUMOS14 annotation")
    class_map = _require_hashed_file(class_map_path, class_map_sha256, "THUMOS14 class map")
    train_data = _path(train_data_path)
    _require(train_data.is_dir(), "THUMOS14 training video directory is missing")
    cfg.dataset.train.ann_file = annotation["path"]
    cfg.dataset.train.class_map = class_map["path"]
    cfg.dataset.train.data_path = str(train_data)
    cfg.model.backbone.custom.pretrain = pretrain["path"]

    random.seed(GATE_SEED)
    np.random.seed(GATE_SEED)
    torch.manual_seed(GATE_SEED)
    torch.cuda.manual_seed_all(GATE_SEED)
    device = torch.device("cuda:0")
    dataset = build_dataset(copy.deepcopy(cfg.dataset.train))
    _require(
        dataset.__class__.__name__ == "DucaStatelessThumosPaddingDataset",
        "gate did not build the real Stage-A stateless dataset",
    )
    inventory, sample_index = _inventory(dataset, train_data)
    model = build_detector(copy.deepcopy(cfg.model)).to(device)
    _require(model.__class__.__name__ == "ActionFormer", "gate did not build ActionFormer")
    _require(model.frame_selector.__class__.__name__ == "DucaRimeFrameSelector", "gate did not build RIME")
    _require(model.frame_selector.rime_arm == "uniform_mixed_k", "selector is not in mixed training mode")
    _require(model.backbone.__class__.__name__ == "BackboneWrapper", "real backbone wrapper is missing")
    _require(model.backbone.model.__class__.__name__ == "Recognizer3D", "real mmaction recognizer is missing")
    _require(
        model.backbone.model.backbone.__class__.__name__ == "VisionTransformerAdapter",
        "real VideoMAE adapter is missing",
    )
    model.train()
    executions = [
        _run_one_request(
            model=model,
            dataset=dataset,
            sample_index=sample_index,
            requested_k=requested,
            device=device,
        )
        for requested in REQUESTED_BUDGETS
    ]
    del model
    torch.cuda.empty_cache()

    final_head = _git("rev-parse", "--verify", "HEAD")
    final_tree = _git("rev-parse", "HEAD^{tree}")
    final_status = _git("status", "--porcelain", "--untracked-files=normal")
    _require(final_head == git["git_commit"] and final_tree == git["git_tree"], "Git identity changed during gate")
    _require(not final_status, "worktree became dirty during gate")
    payload = {
        "schema_version": SCHEMA,
        "status": "passed",
        "fail_closed": True,
        "git_commit": git["git_commit"],
        "git_binding": git,
        "final_clean_binding": {
            "git_commit_unchanged": True,
            "git_tree_unchanged": True,
            "git_tree_clean_after_gate": True,
        },
        "slurm_cuda_binding": slurm,
        "prerequisite_clean_linux_code_gate": code_gate,
        "config": {
            "path": str(config_file),
            "sha256": _sha256(config_file),
            "resolved_before_asset_binding_sha256": duca_paper_training.canonical_sha256(
                Config.fromfile(str(config_file)).to_dict()
            ),
            "formal_protocol": duca_paper_training.FORMAL_PROTOCOL,
            "arm": static_contract["variant"],
        },
        "assets": {
            "pretrain": pretrain,
            "annotation": annotation,
            "class_map": class_map,
            "train_data_path": str(train_data),
        },
        "audited_file_sha256": _audited_hashes(),
        "dataset": inventory,
        "input_provenance": "real_thumos14_full200_train_video_decode",
        "synthetic_inputs_used": False,
        "validation_or_test_data_used": False,
        "mixed_training_mode": True,
        "requested_budget_order": list(REQUESTED_BUDGETS),
        "executions": executions,
        "selector_to_unique_gather_to_heavy_backbone_completed": True,
        "paper_metric_claim_allowed": False,
        "paper_method_performance_evidence": False,
        "claim_scope": "engineering_short_window_execution_only",
        "stage_a_rerun_required": True,
        "stage_b_enabled": False,
        "official_final_consumed": False,
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    return payload


def _write_new(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = _path(path)
    try:
        target.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise GateFailure("gate receipt must stay outside the Git worktree")
    _require(not target.exists(), "refusing to overwrite short-window gate evidence")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--code-gate-receipt", required=True)
    parser.add_argument("--code-gate-receipt-sha256", required=True)
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--pretrain-sha256", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--annotation-sha256", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--class-map-sha256", required=True)
    parser.add_argument("--train-data-path", required=True)
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    result = run_gate(
        expected_commit=args.expected_commit,
        code_gate_receipt_path=args.code_gate_receipt,
        code_gate_receipt_sha256=args.code_gate_receipt_sha256,
        pretrain_path=args.pretrain,
        pretrain_sha256=args.pretrain_sha256,
        annotation_path=args.annotation,
        annotation_sha256=args.annotation_sha256,
        class_map_path=args.class_map,
        class_map_sha256=args.class_map_sha256,
        train_data_path=args.train_data_path,
        config_path=args.config,
    )
    target = _write_new(args.output_json, result)
    print(f"ENGINEERING_STATUS short-window gate receipt: {target}")
    print(f"ENGINEERING_STATUS receipt SHA-256: {_sha256(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
