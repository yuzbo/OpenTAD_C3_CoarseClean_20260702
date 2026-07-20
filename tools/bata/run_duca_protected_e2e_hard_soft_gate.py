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
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from mmengine.config import Config
from torch.utils.data import Subset

from opentad.datasets import build_dataloader, build_dataset
from opentad.models import build_detector
from opentad.models.duca.hard_soft_alignment import (
    enumerate_legal_local_hard_swaps,
    hard_soft_alignment_report,
    preregistered_hard_soft_gate,
    surrogate_hard_swap_descent,
)
from tools.bata.validate_duca_protected_e2e_official60 import validate_config


SCHEMA = "duca_protected_e2e_hard_soft_alignment_gate_v1"
DEFAULT_CONFIG = (
    "configs/adatad/thumos/duca_protected_e2e_fixed384_official60.py"
)


class GateFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(f"fail-closed DUCA protected-E2E hard-soft gate: {message}")


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
        "--expected-commit must be an exact 40-character commit",
    )
    head = _git_output("rev-parse", "--verify", "HEAD")
    status = _git_output("status", "--porcelain", "--untracked-files=normal")
    _require(head == expected, f"expected commit {expected}, observed {head}")
    _require(not status, "gate requires a clean exact-commit checkout")
    return {
        "git_commit": head,
        "git_tree": _git_output("rev-parse", "HEAD^{tree}"),
        "git_branch": _git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "git_tree_clean": True,
    }


def _bind_slurm_cuda() -> dict[str, Any]:
    _require(bool(os.environ.get("SLURM_JOB_ID")), "SLURM_JOB_ID is required")
    _require(bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "Slurm must provide CUDA_VISIBLE_DEVICES")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(torch.cuda.device_count() == 1, "gate requires exactly one Slurm logical GPU")
    torch.cuda.set_device(0)
    return {
        "slurm_job_id": str(os.environ["SLURM_JOB_ID"]),
        "logical_device": "cuda:0",
        "physical_gpu_index_assumed": False,
        "device_name": torch.cuda.get_device_name(0),
        "torch_cuda_version": torch.version.cuda,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_trained_checkpoint(
    model,
    *,
    checkpoint_path: str,
    expected_sha256: str,
    source_commit: str,
    evidence_path: str,
    evidence_sha256: str,
) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser().resolve()
    _require(path.is_file(), f"trained checkpoint is missing: {path}")
    actual_sha256 = _sha256(path)
    _require(
        re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256).lower()) is not None,
        "--checkpoint-sha256 must be an exact SHA256",
    )
    _require(actual_sha256 == str(expected_sha256).lower(), "trained checkpoint SHA256 mismatch")
    _require(
        re.fullmatch(r"[0-9a-f]{40}", str(source_commit).lower()) is not None,
        "--checkpoint-source-commit must be an exact commit",
    )
    evidence_file = Path(evidence_path).expanduser().resolve()
    _require(evidence_file.is_file(), "trained checkpoint evidence is missing")
    _require(
        re.fullmatch(r"[0-9a-f]{64}", str(evidence_sha256).lower()) is not None,
        "--checkpoint-evidence-sha256 must be an exact SHA256",
    )
    _require(
        _sha256(evidence_file) == str(evidence_sha256).lower(),
        "trained checkpoint evidence SHA256 mismatch",
    )
    evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
    _require(
        evidence.get("schema") == "duca_cellcf_post_run_evidence_v1"
        and evidence.get("ok") is True,
        "trained checkpoint evidence schema/status is invalid",
    )
    _require(
        evidence.get("task") == "offline_temporal_action_detection",
        "trained checkpoint evidence is not offline TAD",
    )
    _require(
        evidence.get("variant") == "transition_beta0",
        "P3 audit checkpoint must be the trained transition_beta0 control",
    )
    _require(
        evidence.get("git_commit") == str(source_commit).lower(),
        "checkpoint evidence source commit mismatch",
    )
    _require(
        Path(str(evidence.get("checkpoint_path"))).resolve() == path
        and evidence.get("checkpoint_sha256") == actual_sha256,
        "checkpoint evidence path/hash mismatch",
    )
    _require(
        evidence.get("checkpoint_state_key") == "state_dict_ema"
        and int(evidence.get("checkpoint_epoch", -1)) == 131
        and int(evidence.get("successful_optimizer_updates", -1)) == 13200,
        "checkpoint evidence terminal training protocol mismatch",
    )
    _require(
        evidence.get("non_finite_collapse") is False,
        "checkpoint evidence reports non-finite collapse",
    )
    manifest_path = Path(str(evidence.get("run_manifest_path"))).resolve()
    _require(
        manifest_path.is_file()
        and _sha256(manifest_path) == evidence.get("run_manifest_sha256"),
        "checkpoint run manifest binding is invalid",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("git_commit") == str(source_commit).lower()
        and manifest.get("variant") == "transition_beta0"
        and manifest.get("training_profile") == "exposure132"
        and int(manifest.get("expected_successful_optimizer_updates", -1)) == 13200,
        "checkpoint run manifest protocol mismatch",
    )
    payload = torch.load(path, map_location="cpu")
    _require(isinstance(payload, Mapping), "trained checkpoint must be a mapping")
    _require("state_dict_ema" in payload, "trained checkpoint must contain terminal state_dict_ema")
    _require(
        int(payload.get("epoch", -1)) == 131
        and int(payload.get("scheduler", {}).get("last_epoch", -1)) == 13200,
        "checkpoint payload epoch/update mismatch",
    )
    state_dict = payload["state_dict_ema"]
    _require(isinstance(state_dict, Mapping), "state_dict_ema must be a mapping")
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    _require(not missing, f"trained checkpoint is missing model keys: {missing}")
    _require(not unexpected, f"trained checkpoint has unexpected model keys: {unexpected}")
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "source_commit": str(source_commit).lower(),
        "state_key": "state_dict_ema",
        "evidence_path": str(evidence_file),
        "evidence_sha256": str(evidence_sha256).lower(),
        "source_variant": "transition_beta0",
        "source_training_profile": "exposure132",
        "source_successful_optimizer_updates": 13200,
        "source_terminal_average_map": float(evidence["metrics"]["average_mAP"]),
        "strict_parameter_and_buffer_match": True,
    }


def _cuda_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    required = {"inputs", "masks", "metas", "gt_segments", "gt_labels"}
    _require(required.issubset(batch), "real THUMOS batch is missing required fields")
    return {
        "inputs": batch["inputs"].to(device="cuda:0", non_blocking=True),
        "masks": batch["masks"].to(device="cuda:0", dtype=torch.bool, non_blocking=True),
        "metas": [dict(meta) for meta in batch["metas"]],
        "gt_segments": [value.to(device="cuda:0", non_blocking=True) for value in batch["gt_segments"]],
        "gt_labels": [value.to(device="cuda:0", non_blocking=True) for value in batch["gt_labels"]],
    }


def _snapshot_detector_runtime(model) -> dict[str, Any]:
    return {
        "buffers": {
            name: value.detach().clone()
            for name, value in model.named_buffers()
            if not name.startswith("frame_selector.")
        },
        "cpu_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }


def _restore_detector_runtime(model, state: Mapping[str, Any]) -> None:
    buffers = dict(model.named_buffers())
    for name, value in state["buffers"].items():
        _require(name in buffers, f"detector buffer disappeared during replay: {name}")
        buffers[name].copy_(value)
    torch.random.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])


def _candidate_detector_loss(
    model,
    selector,
    batch: Mapping[str, Any],
    positions: torch.Tensor,
) -> float:
    selected_inputs = model._duca_gather_raw(batch["inputs"], positions)
    selected_masks = positions >= 0
    active = positions[0, selected_masks[0]].detach().cpu().tolist()
    meta = dict(batch["metas"][0])
    meta.update(
        {
            "selected_axis_to_true_time_dense_index": [int(value) for value in active],
            "truetime_dense_len": int(batch["masks"].shape[-1]),
            "truetime_dense_valid_len": int(batch["masks"][0].sum().item()),
            "detector_output_coordinate_space": "selected_axis_index",
        }
    )
    remapped_segments, remapped_labels, remapped_metas = selector._remap_train_targets_to_selected_axis(
        batch["gt_segments"],
        batch["gt_labels"],
        [meta],
    )
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
        cache_enabled=False,
    ):
        losses = model.forward_train(
            selected_inputs,
            selected_masks,
            remapped_metas,
            remapped_segments,
            remapped_labels,
            _duca_skip_frame_selector=True,
            _duca_counterfactual_eval=True,
        )
        objective = model._duca_detector_objective(losses)
    _require(torch.isfinite(objective).item(), "candidate detector loss is non-finite")
    return float(objective.detach().cpu().item())


def _action_length_group(segments: torch.Tensor, valid_len: int) -> str:
    if segments.numel() == 0:
        return "background_window"
    duration = float((segments[:, 1] - segments[:, 0]).clamp_min(0).max().item())
    ratio = duration / max(1.0, float(valid_len))
    if ratio < 0.25:
        return "short_action"
    if ratio < 0.50:
        return "medium_action"
    return "long_action"


def _nearest_boundary_distance(
    add_position: int,
    remove_position: int,
    segments: torch.Tensor,
) -> float:
    if segments.numel() == 0:
        return float("inf")
    boundaries = torch.cat((segments[:, 0], segments[:, 1])).float()
    points = boundaries.new_tensor([float(add_position), float(remove_position)])
    return float((points[:, None] - boundaries[None, :]).abs().min().item())


def _group_descriptives(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    output = {}
    for name, group in sorted(groups.items()):
        predicted = [float(row["surrogate_predicted_utility"]) for row in group]
        observed = [float(row["hard_detector_utility"]) for row in group]
        informative = [
            (left, right)
            for left, right in zip(predicted, observed)
            if math.isfinite(left)
            and math.isfinite(right)
            and abs(left) > 1.0e-8
            and abs(right) > 1.0e-8
        ]
        output[name] = {
            "count": len(group),
            "informative_count": len(informative),
            "sign_agreement": (
                None
                if not informative
                else sum((left > 0.0) == (right > 0.0) for left, right in informative)
                / len(informative)
            ),
            "mean_predicted_utility": sum(predicted) / len(predicted),
            "mean_hard_utility": sum(observed) / len(observed),
        }
    return output


def run_gate(
    *,
    config_path: str,
    expected_commit: str,
    output_json: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    checkpoint_source_commit: str,
    checkpoint_evidence: str,
    checkpoint_evidence_sha256: str,
    real_batches: int,
    candidates_per_batch: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    output = Path(output_json).expanduser().resolve()
    _require(not output.exists(), "refusing to overwrite hard-soft gate evidence")
    try:
        output.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise GateFailure("output evidence must be outside the Git worktree")
    source = _bind_clean_commit(expected_commit)
    slurm = _bind_slurm_cuda()
    config_summary = validate_config(ROOT / config_path)
    cfg = Config.fromfile(str(ROOT / config_path))
    _require(int(real_batches) >= 4, "pre-registered gate requires at least four real batches")
    _require(int(candidates_per_batch) >= 6, "pre-registered gate requires at least six candidates per batch")

    random.seed(20260720)
    np.random.seed(20260720)
    torch.manual_seed(20260720)
    torch.cuda.manual_seed_all(20260720)
    dataset = build_dataset(copy.deepcopy(cfg.dataset.train), default_args={"logger": None})
    loader_cfg = copy.deepcopy(cfg.solver.train)
    loader_cfg["batch_size"] = 1
    loader = build_dataloader(
        Subset(dataset, list(range(min(len(dataset), max(32, real_batches * 4))))),
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **loader_cfg,
    )

    model = build_detector(copy.deepcopy(cfg.model)).to("cuda:0")
    checkpoint = _load_trained_checkpoint(
        model,
        checkpoint_path=checkpoint_path,
        expected_sha256=checkpoint_sha256,
        source_commit=checkpoint_source_commit,
        evidence_path=checkpoint_evidence,
        evidence_sha256=checkpoint_evidence_sha256,
    )
    model.train()
    selector = model.frame_selector
    selector.retain_gradient_audit_tensors = True
    schedule = selector.loss_weight_schedule
    bridge_ready_step = int(schedule["detector_gradient"]["warmup_steps"]) + int(
        schedule["detector_gradient"]["transition_steps"]
    )
    selector._loss_weight_schedule_step.fill_(bridge_ready_step)
    _require(
        selector._loss_schedule_state()["detector_gradient_weight"] == 0.25,
        "protected bridge did not reach its preregistered final weight",
    )
    frozen_normalizer = model.rpn_head.loss_normalizer.detach().clone()
    model.rpn_head.duca_set_frozen_loss_normalizer(frozen_normalizer)

    predicted_values: list[float] = []
    hard_values: list[float] = []
    batch_ids: list[int] = []
    rows: list[dict[str, Any]] = []
    accepted_batches = 0
    try:
        for raw_batch in loader:
            if accepted_batches >= int(real_batches):
                break
            batch = _cuda_batch(raw_batch)
            valid_len = int(batch["masks"][0].sum().item())
            if valid_len != int(cfg.dense_window_size):
                continue
            detector_entry_state: dict[str, Any] = {}

            def capture_detector_entry(_module, _args):
                if detector_entry_state:
                    raise GateFailure("detector backbone was entered more than once")
                detector_entry_state.update(_snapshot_detector_runtime(model))

            entry_hook = model.backbone.register_forward_pre_hook(capture_detector_entry)
            try:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=True,
                    cache_enabled=False,
                ):
                    losses = model(
                        batch["inputs"],
                        batch["masks"],
                        batch["metas"],
                        gt_segments=batch["gt_segments"],
                        gt_labels=batch["gt_labels"],
                        return_loss=True,
                    )
            finally:
                entry_hook.remove()
            _require(detector_entry_state, "failed to capture detector-entry replay state")
            detector_loss = model._duca_detector_objective(losses)
            _require(torch.isfinite(detector_loss).item(), "baseline detector loss is non-finite")
            post_baseline_state = _snapshot_detector_runtime(model)
            audit_tensors = getattr(selector, "_gradient_audit_tensors", None)
            _require(isinstance(audit_tensors, Mapping), "selector did not retain P3 audit tensors")
            center_scores = audit_tensors["center_scores"]
            gradient = torch.autograd.grad(detector_loss, center_scores, retain_graph=False)[0].detach()
            _require(torch.isfinite(gradient).all().item(), "AMP policy gradient is non-finite")
            positions = audit_tensors["selected_positions"].detach()
            swaps = enumerate_legal_local_hard_swaps(
                positions,
                batch["masks"],
                max_unselected_hole=int(selector.max_unselected_hole),
                max_displacement=2,
                max_candidates_per_sample=int(candidates_per_batch),
            )
            valid_candidates = swaps["candidate_valid"][0]
            if int(valid_candidates.sum().item()) < int(candidates_per_batch):
                continue
            predicted = surrogate_hard_swap_descent(
                gradient,
                swaps["add_positions"],
                swaps["remove_positions"],
                swaps["candidate_valid"],
            )
            baseline = float(detector_loss.detach().cpu().item())
            video_name = str(batch["metas"][0].get("video_name", f"batch_{accepted_batches}"))
            action_group = _action_length_group(batch["gt_segments"][0], valid_len)
            for candidate_index in torch.nonzero(valid_candidates, as_tuple=False).flatten().tolist():
                candidate_positions = swaps["candidate_selections"][:, candidate_index]
                _restore_detector_runtime(model, detector_entry_state)
                try:
                    candidate_loss = _candidate_detector_loss(
                        model,
                        selector,
                        batch,
                        candidate_positions,
                    )
                finally:
                    _restore_detector_runtime(model, post_baseline_state)
                hard_utility = baseline - candidate_loss
                predicted_utility = float(predicted[0, candidate_index].detach().cpu().item())
                add = int(swaps["add_positions"][0, candidate_index].item())
                remove = int(swaps["remove_positions"][0, candidate_index].item())
                predicted_values.append(predicted_utility)
                hard_values.append(hard_utility)
                batch_ids.append(accepted_batches)
                rows.append(
                    {
                        "batch_id": accepted_batches,
                        "video_name": video_name,
                        "window_type": "full",
                        "action_length_group": action_group,
                        "add_dense_candidate": add,
                        "remove_dense_candidate": remove,
                        "absolute_displacement_dense_steps": int(
                            swaps["absolute_displacement_dense_steps"][0, candidate_index].item()
                        ),
                        "nearest_gt_boundary_distance_dense_candidates": _nearest_boundary_distance(
                            add,
                            remove,
                            batch["gt_segments"][0],
                        ),
                        "surrogate_predicted_utility": predicted_utility,
                        "hard_detector_utility": hard_utility,
                        "baseline_detector_loss": baseline,
                        "candidate_detector_loss": candidate_loss,
                    }
                )
                boundary_distance = rows[-1][
                    "nearest_gt_boundary_distance_dense_candidates"
                ]
                if boundary_distance <= 4.0:
                    rows[-1]["boundary_distance_group"] = "near_0_4"
                elif boundary_distance <= 16.0:
                    rows[-1]["boundary_distance_group"] = "middle_5_16"
                else:
                    rows[-1]["boundary_distance_group"] = "far_over_16"
            accepted_batches += 1
    finally:
        model.rpn_head.duca_set_frozen_loss_normalizer(None)

    _require(accepted_batches >= int(real_batches), "not enough full real THUMOS windows were audited")
    report = hard_soft_alignment_report(
        predicted_values,
        hard_values,
        batch_ids=batch_ids,
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_seed=20260720,
    )
    decision = preregistered_hard_soft_gate(report)
    payload = {
        "schema": SCHEMA,
        "ok": bool(decision["passed"]),
        "status": (
            "p3_hard_soft_alignment_passed"
            if decision["passed"]
            else "p3_hard_soft_alignment_failed_stop_before_official60"
        ),
        "source": source,
        "slurm": slurm,
        "config_contract": config_summary,
        "config_sha256": _sha256((ROOT / config_path).resolve()),
        "trained_checkpoint": checkpoint,
        "real_dataset_loader_executed": True,
        "numeric_contract": {
            "autocast_enabled": True,
            "autocast_device_type": "cuda",
            "autocast_dtype": "float16",
            "grad_scaler_direction_effect": "positive_scalar_only_not_required_for_sign",
            "baseline_and_all_hard_candidates_share_autocast": True,
        },
        "real_batches_requested": int(real_batches),
        "real_batches_audited": accepted_batches,
        "candidates_per_batch": int(candidates_per_batch),
        "alignment_report": report,
        "alignment_descriptives": {
            "by_action_length": _group_descriptives(rows, "action_length_group"),
            "by_boundary_distance": _group_descriptives(
                rows,
                "boundary_distance_group",
            ),
        },
        "preregistered_gate": decision,
        "hard_swap_rows": rows,
        "claims": {
            "direct_detector_gradient_supported": bool(decision["passed"]),
            "official60_submission_allowed": bool(decision["passed"]),
            "paper_ready": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    output.write_text(text, encoding="utf-8")
    print(text)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-source-commit", required=True)
    parser.add_argument("--checkpoint-evidence", required=True)
    parser.add_argument("--checkpoint-evidence-sha256", required=True)
    parser.add_argument("--real-batches", type=int, default=4)
    parser.add_argument("--candidates-per-batch", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    try:
        payload = run_gate(
            config_path=args.config,
            expected_commit=args.expected_commit,
            output_json=args.output_json,
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            checkpoint_source_commit=args.checkpoint_source_commit,
            checkpoint_evidence=args.checkpoint_evidence,
            checkpoint_evidence_sha256=args.checkpoint_evidence_sha256,
            real_batches=args.real_batches,
            candidates_per_batch=args.candidates_per_batch,
            bootstrap_samples=args.bootstrap_samples,
        )
    except Exception as exc:
        payload = {
            "schema": SCHEMA,
            "ok": False,
            "status": "p3_gate_execution_failed",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
