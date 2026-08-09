from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from mmengine.config import Config
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel

from opentad.cores import (
    build_optimizer,
    build_scheduler,
    prepare_optimizer_parameter_freezing,
)
from opentad.cores.train_engine import (
    _call_after_optimizer_step,
    _capture_custom_replay_state,
    _capture_model_buffers,
    _capture_rng_state,
    _restore_custom_replay_state,
    _restore_model_buffers,
    _restore_rng_state,
)
from opentad.datasets import build_dataloader, build_dataset
from opentad.models import build_detector
from opentad.models.duca import structured_selection as structured
from opentad.utils import set_seed
from tools.bata import duca_paper_training
from tools.bata.run_duca_paper_legacy_numeric_regression import (
    _legacy_raw_slot_mass_drift,
)
from tools.bata.validate_duca_paper_code_gate import validate_code_gate_artifact
from tools.bata.validate_duca_paper_short_window_gate import validate_gate_artifact


SCHEMA = "duca_paper_physical_exactk_numeric_gate_v2"
FAILURE_SCHEMA = "duca_paper_physical_exactk_numeric_gate_failure_v1"
CONFIG_DEFAULT = "configs/adatad/thumos/duca_paper_duca_fixed_k384_full200.py"
GATE_SEED = 5801
MAX_ATTEMPTED_UPDATES = 100
EXPECTED_WORLD_SIZE = 2
EXPECTED_GLOBAL_BATCH_SIZE = 2
EXPECTED_T = 768
EXPECTED_K = 384
PROCESS_GROUP_TIMEOUT_SECONDS = 600
PROCESS_WATCHDOG_TIMEOUT_SECONDS = 14_400
THRESHOLDS = {
    "fp32_fp64_slot_atol": 5.0e-5,
    "fp32_fp64_slot_rtol": 5.0e-5,
    "slot_row_mass_max_abs": 8.0e-6,
    "dual_logz_max_abs": 5.0e-5,
    "edge_flow_linf_max_abs": 2.0e-5,
    "edge_flow_per_slot_l1_max_abs": 1.0e-4,
    "fp32_fp64_gradient_atol": 2.0e-5,
    "fp32_fp64_gradient_rtol": 2.0e-3,
    "column_occupancy_max": 1.0 + 5.0e-4,
    "ordered_slot_expectation_min_gap_gt": 0.0,
    "additive_shift": 37.0,
}


class GateFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(f"fail-closed DUCA paper numeric gate: {message}")


def _distributed_require(
    condition: bool,
    message: str,
    *,
    device: torch.device,
) -> None:
    failed = torch.tensor(
        0 if condition else 1,
        device=device,
        dtype=torch.int64,
    )
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if int(failed.item()) != 0:
        raise GateFailure(f"fail-closed DUCA paper numeric gate: {message}")


def _path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return duca_paper_training.canonical_sha256(value)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
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


def _require_hashed_file(
    path: str | Path,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    source = _path(path)
    expected = str(expected_sha256).strip().lower()
    _require(source.is_file(), f"{label} is missing: {source}")
    _require(
        re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
        f"invalid {label} SHA-256",
    )
    observed = _sha256(source)
    _require(observed == expected, f"{label} SHA-256 drift")
    return {"path": str(source), "sha256": observed}


def _graph_to_cpu(graph: structured.PhysicalExactKGraph) -> dict[str, Any]:
    return {
        name: getattr(graph, name).detach().cpu().clone()
        for name in (
            "predecessor_index",
            "predecessor_valid",
            "successor_index",
            "successor_valid",
            "source_valid",
            "sink_valid",
            "max_gap_seconds",
        )
    } | {"edge_count": int(graph.edge_count)}


def _graph_to_device(payload: Mapping[str, Any], device: torch.device):
    return structured.PhysicalExactKGraph(
        predecessor_index=payload["predecessor_index"].to(device=device),
        predecessor_valid=payload["predecessor_valid"].to(device=device),
        successor_index=payload["successor_index"].to(device=device),
        successor_valid=payload["successor_valid"].to(device=device),
        source_valid=payload["source_valid"].to(device=device),
        sink_valid=payload["sink_valid"].to(device=device),
        max_gap_seconds=payload["max_gap_seconds"].to(device=device),
        edge_count=int(payload["edge_count"]),
    )


def _normalized_tables(
    node_log_probs: torch.Tensor,
    *,
    k: int,
    graph: structured.PhysicalExactKGraph,
    temperature: float,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    scores = node_log_probs.to(dtype=dtype) / float(temperature)
    gauge = scores.detach().mean()
    scores = scores - gauge
    alpha_rows = []
    alpha_raw = torch.where(
        graph.source_valid,
        scores,
        scores.new_full(scores.shape, float("-inf")),
    )
    alpha, alpha_scale = structured._normalize_log_message(
        alpha_raw, failure_context="numeric gate forward source"
    )
    alpha_rows.append(alpha)
    for slot in range(1, int(k)):
        candidates = alpha[graph.predecessor_index]
        mass = structured._safe_masked_logsumexp(
            candidates, graph.predecessor_valid, dim=1
        )
        alpha, local = structured._normalize_log_message(
            scores + mass, failure_context=f"numeric gate forward slot {slot}"
        )
        alpha_scale = alpha_scale + local
        alpha_rows.append(alpha)
    alpha_table = torch.stack(alpha_rows, dim=0)
    forward_centered = alpha_scale + structured._safe_masked_logsumexp(
        alpha_table[-1], graph.sink_valid, dim=0
    )

    beta_rows = [scores.new_empty(scores.shape) for _ in range(int(k))]
    beta_raw = torch.where(
        graph.sink_valid,
        scores.new_zeros(scores.shape),
        scores.new_full(scores.shape, float("-inf")),
    )
    beta, beta_scale = structured._normalize_log_message(
        beta_raw, failure_context="numeric gate backward sink"
    )
    beta_rows[-1] = beta
    for slot in range(int(k) - 2, -1, -1):
        candidates = scores[graph.successor_index] + beta[graph.successor_index]
        beta, local = structured._normalize_log_message(
            structured._safe_masked_logsumexp(
                candidates, graph.successor_valid, dim=1
            ),
            failure_context=f"numeric gate backward slot {slot}",
        )
        beta_scale = beta_scale + local
        beta_rows[slot] = beta
    beta_table = torch.stack(beta_rows, dim=0)
    backward_centered = beta_scale + structured._safe_masked_logsumexp(
        scores + beta_table[0], graph.source_valid, dim=0
    )
    joint = alpha_table + beta_table
    finite = torch.isfinite(joint)
    log_mass = structured._safe_masked_logsumexp(joint, finite, dim=1)
    slots = torch.where(
        finite,
        torch.exp(joint - log_mass[:, None]),
        joint.new_zeros(()),
    )
    slots = slots / slots.sum(dim=1, keepdim=True)
    return {
        "scores": scores,
        "alpha": alpha_table,
        "beta": beta_table,
        "slots": slots,
        "forward_logz": forward_centered + float(k) * gauge,
        "backward_logz": backward_centered + float(k) * gauge,
    }


def _edge_flow_residuals(
    tables: Mapping[str, torch.Tensor],
    graph: structured.PhysicalExactKGraph,
) -> tuple[float, float]:
    scores = tables["scores"]
    alpha = tables["alpha"]
    beta = tables["beta"]
    slots = tables["slots"]
    predecessor_index = graph.predecessor_index
    predecessor_valid = graph.predecessor_valid
    linf = 0.0
    per_slot_l1 = 0.0
    for slot in range(int(slots.shape[0]) - 1):
        edge_log = (
            alpha[slot][predecessor_index]
            + scores[:, None]
            + beta[slot + 1][:, None]
        )
        edge_log = edge_log.masked_fill(~predecessor_valid, float("-inf"))
        normalizer = torch.logsumexp(edge_log.flatten(), dim=0)
        edge_prob = torch.where(
            predecessor_valid,
            torch.exp(edge_log - normalizer),
            edge_log.new_zeros(()),
        )
        incoming = edge_prob.sum(dim=1)
        outgoing = torch.zeros_like(incoming)
        outgoing.scatter_add_(
            0,
            predecessor_index.flatten(),
            edge_prob.flatten(),
        )
        incoming_delta = (incoming - slots[slot + 1]).abs()
        outgoing_delta = (outgoing - slots[slot]).abs()
        linf = max(
            linf,
            float(incoming_delta.max().item()),
            float(outgoing_delta.max().item()),
        )
        per_slot_l1 = max(
            per_slot_l1,
            float(incoming_delta.sum().item()),
            float(outgoing_delta.sum().item()),
        )
    return linf, per_slot_l1


def _relative_max(delta: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = reference.abs().clamp_min(1.0e-12)
    return float((delta.abs() / denominator).max().item())


def _run_oracle_diagnostics(
    capture: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    node = capture["node_log_probs"].to(device=device, dtype=torch.float32)
    graph = _graph_to_device(capture["graph"], device)
    k = int(capture["k"])
    temperature = float(capture["temperature"])
    x32 = node.detach().clone().requires_grad_(True)
    occupancy32, slots32, logz32 = structured._physical_row_forward_backward(
        x32, k=k, graph=graph, temperature=temperature
    )
    x64 = node.detach().double().requires_grad_(True)
    occupancy64, slots64, logz64 = structured._physical_row_forward_backward(
        x64, k=k, graph=graph, temperature=temperature
    )
    weights32 = torch.linspace(
        -1.0, 1.0, int(slots32.numel()), device=device, dtype=torch.float32
    ).reshape_as(slots32)
    objective32 = (slots32 * weights32).sum() + 0.01 * logz32
    grad32 = torch.autograd.grad(objective32, x32)[0]
    weights64 = weights32.double()
    objective64 = (slots64 * weights64).sum() + 0.01 * logz64
    grad64 = torch.autograd.grad(objective64, x64)[0]
    additive_shift = float(THRESHOLDS["additive_shift"])
    shifted32 = (node.detach() + additive_shift).requires_grad_(True)
    occupancy_shift32, slots_shift32, logz_shift32 = (
        structured._physical_row_forward_backward(
            shifted32,
            k=k,
            graph=graph,
            temperature=temperature,
        )
    )
    objective_shift32 = (slots_shift32 * weights32).sum() + 0.01 * logz_shift32
    grad_shift32 = torch.autograd.grad(objective_shift32, shifted32)[0]
    shifted64 = (node.detach().double() + additive_shift).requires_grad_(True)
    _, slots_shift64, logz_shift64 = structured._physical_row_forward_backward(
        shifted64,
        k=k,
        graph=graph,
        temperature=temperature,
    )
    objective_shift64 = (slots_shift64 * weights64).sum() + 0.01 * logz_shift64
    grad_shift64 = torch.autograd.grad(objective_shift64, shifted64)[0]
    slot_delta = slots32.double() - slots64
    grad_delta = grad32.double() - grad64
    tables32 = _normalized_tables(
        node, k=k, graph=graph, temperature=temperature, dtype=torch.float32
    )
    edge_linf, edge_l1 = _edge_flow_residuals(tables32, graph)
    hard32 = structured._physical_row_viterbi(
        node.detach(), k=k, graph=graph
    )[1]
    hard64 = structured._physical_row_viterbi(
        node.detach().double(), k=k, graph=graph
    )[1]
    hard_shift32 = structured._physical_row_viterbi(
        node.detach() + additive_shift, k=k, graph=graph
    )[1]
    row_mass_error = float(
        (slots32.sum(dim=1) - 1.0).abs().max().item()
    )
    column_occupancy_max = float(occupancy32.max().item())
    slot_expectations = (
        slots32
        * torch.arange(
            int(slots32.shape[1]),
            device=device,
            dtype=slots32.dtype,
        )[None, :]
    ).sum(dim=1)
    ordered_min_gap = float(
        (slot_expectations[1:] - slot_expectations[:-1]).min().item()
    )
    partition_residual = float(
        (tables32["forward_logz"] - tables32["backward_logz"]).abs().item()
    )
    expected_logz_shift = float(k) * additive_shift / temperature
    fp64_logz_shift_residual = abs(
        float((logz_shift64 - logz64).item()) - expected_logz_shift
    )
    shift_slots_allclose = bool(
        torch.allclose(
            slots_shift32,
            slots32,
            atol=THRESHOLDS["fp32_fp64_slot_atol"],
            rtol=THRESHOLDS["fp32_fp64_slot_rtol"],
        )
    )
    shift_gradients_allclose = bool(
        torch.allclose(
            grad_shift32,
            grad32,
            atol=THRESHOLDS["fp32_fp64_gradient_atol"],
            rtol=THRESHOLDS["fp32_fp64_gradient_rtol"],
        )
        and torch.allclose(
            grad_shift64,
            grad64,
            atol=THRESHOLDS["fp32_fp64_gradient_atol"],
            rtol=THRESHOLDS["fp32_fp64_gradient_rtol"],
        )
    )
    result = {
        "fp32_fp64_slot_max_abs": float(slot_delta.abs().max().item()),
        "fp32_fp64_slot_max_relative": _relative_max(slot_delta, slots64),
        "slot_row_mass_max_abs": row_mass_error,
        "dual_logz_max_abs": partition_residual,
        "edge_flow_linf_max_abs": edge_linf,
        "edge_flow_per_slot_l1_max_abs": edge_l1,
        "all_gradients_finite": bool(
            torch.isfinite(grad32).all().item()
            and torch.isfinite(grad64).all().item()
        ),
        "fp32_fp64_gradient_max_abs": float(grad_delta.abs().max().item()),
        "fp32_fp64_gradient_max_relative": _relative_max(grad_delta, grad64),
        "fp32_fp64_slots_allclose": bool(
            torch.allclose(
                slots32.double(),
                slots64,
                atol=THRESHOLDS["fp32_fp64_slot_atol"],
                rtol=THRESHOLDS["fp32_fp64_slot_rtol"],
            )
        ),
        "fp32_fp64_gradients_allclose": bool(
            torch.allclose(
                grad32.double(),
                grad64,
                atol=THRESHOLDS["fp32_fp64_gradient_atol"],
                rtol=THRESHOLDS["fp32_fp64_gradient_rtol"],
            )
        ),
        "hard_path_exact_identical": bool(torch.equal(hard32, hard64)),
        "column_occupancy_max": column_occupancy_max,
        "column_occupancy_within_bound": bool(
            column_occupancy_max <= THRESHOLDS["column_occupancy_max"]
        ),
        "ordered_slot_expectation_min_gap": ordered_min_gap,
        "ordered_slot_expectations_strict": bool(
            ordered_min_gap > THRESHOLDS["ordered_slot_expectation_min_gap_gt"]
        ),
        "additive_shift_slots_allclose": shift_slots_allclose,
        "additive_shift_gradients_allclose": shift_gradients_allclose,
        "additive_shift_hard_path_exact_identical": bool(
            torch.equal(hard32, hard_shift32)
        ),
        "additive_shift_fp64_logz_residual": fp64_logz_shift_residual,
        "hard_path_sha256": _canonical_sha256(
            [int(value) for value in hard32.detach().cpu().tolist()]
        ),
        "occupancy_sum": float(occupancy32.sum().item()),
        "effective_k": k,
    }
    result["thresholds"] = dict(THRESHOLDS)
    result["passed"] = bool(
        result["fp32_fp64_slots_allclose"]
        and row_mass_error <= THRESHOLDS["slot_row_mass_max_abs"]
        and partition_residual <= THRESHOLDS["dual_logz_max_abs"]
        and edge_linf <= THRESHOLDS["edge_flow_linf_max_abs"]
        and edge_l1 <= THRESHOLDS["edge_flow_per_slot_l1_max_abs"]
        and result["all_gradients_finite"]
        and result["fp32_fp64_gradients_allclose"]
        and result["hard_path_exact_identical"]
        and abs(result["occupancy_sum"] - float(k)) <= 2.0e-4 * float(k)
        and result["column_occupancy_within_bound"]
        and result["ordered_slot_expectations_strict"]
        and result["additive_shift_slots_allclose"]
        and result["additive_shift_gradients_allclose"]
        and result["additive_shift_hard_path_exact_identical"]
        and fp64_logz_shift_residual <= THRESHOLDS["dual_logz_max_abs"]
    )
    return result


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device=device, non_blocking=True)
        elif isinstance(value, list) and value and all(torch.is_tensor(item) for item in value):
            output[key] = [item.to(device=device, non_blocking=True) for item in value]
        elif key == "metas":
            output[key] = [dict(item) for item in value]
        else:
            output[key] = value
    return output


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise GateFailure(f"refusing to overwrite numeric-gate artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _failure_class(error: BaseException) -> str:
    message = str(error)
    if "NO_TARGET_CAPTURE" in message:
        return "NO_TARGET_CAPTURE"
    if "AMP replay could not produce" in message:
        return "AMP_REPLAY_EXHAUSTED"
    if "optimizer-step outcome diverged" in message:
        return "DDP_STEP_OUTCOME_DIVERGENCE"
    if (
        "thresholds failed" in message
        or "structural oracle" in message
        or "normalized solver" in message
    ):
        return "CURRENT_STRUCTURAL_ORACLE_FAILED"
    if "watchdog" in message.lower() or "collective" in message.lower():
        return "WATCHDOG_OR_COLLECTIVE_FAILURE"
    return "INTERNAL_ERROR"


def _write_failure_receipt_best_effort(
    args: argparse.Namespace,
    error: BaseException,
) -> None:
    try:
        output_root = _path(args.output_root)
        try:
            output_root.relative_to(ROOT)
        except ValueError:
            pass
        else:
            return
        output_root.mkdir(parents=True, exist_ok=True)
        rank = int(os.environ.get("RANK", "0"))
        try:
            observed_commit = _git("rev-parse", "HEAD")
        except Exception:
            observed_commit = None
        payload = {
            "schema_version": FAILURE_SCHEMA,
            "status": "failed",
            "fail_closed": True,
            "failure_class": _failure_class(error),
            "failure_type": type(error).__name__,
            "failure_message": str(error)[:2000],
            "expected_commit": str(args.expected_commit),
            "observed_commit": observed_commit,
            "slurm_job_id": str(os.environ.get("SLURM_JOB_ID", "")),
            "rank": rank,
            "validation_or_test_data_used": False,
            "checkpoint_created": False,
            "prediction_generated": False,
            "loss_values_recorded": False,
            "metric_accessed": False,
            "paper_metric_claim_allowed": False,
            "paper_method_performance_evidence": False,
            "stage_a_release_prerequisite_satisfied": False,
            "stage_b_enabled": False,
            "official_final_consumed": False,
            "claim_scope": "engineering_numeric_gate_failure_only",
        }
        payload["content_sha256"] = _canonical_sha256(payload)
        rank_path = output_root / f"failure.rank{rank:04d}.receipt.json"
        if not rank_path.exists():
            _write_new_json(rank_path, payload)
        if rank == 0:
            terminal = output_root / "numeric_gate.failure.receipt.json"
            if not terminal.exists():
                _write_new_json(terminal, payload)
    except Exception:
        # Failure evidence is best-effort here; the outer Slurm launcher owns
        # timeout/rank-death consolidation and must still terminate non-zero.
        return


def run_gate(args: argparse.Namespace) -> dict[str, Any] | None:
    git = _bind_clean_commit(args.expected_commit)
    job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    _require(job_id.isdigit(), "gate must run inside a numeric Slurm job")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(
        int(os.environ.get("WORLD_SIZE", "0")) == EXPECTED_WORLD_SIZE,
        "numeric gate requires torchrun world size two",
    )
    _require(
        int(os.environ.get("DUCA_PAPER_NUMERIC_GATE_WALL_TIMEOUT_SECONDS", "-1"))
        == PROCESS_WATCHDOG_TIMEOUT_SECONDS,
        "numeric gate requires the frozen outer process watchdog",
    )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=PROCESS_GROUP_TIMEOUT_SECONDS),
    )
    output_root = _path(args.output_root)
    try:
        output_root.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise GateFailure("numeric-gate output must stay outside the Git worktree")
    if rank == 0:
        _require(not output_root.exists(), "numeric gate requires a fresh output root")
        output_root.mkdir(parents=True)
    dist.barrier()

    code_gate = validate_code_gate_artifact(
        args.code_gate_receipt,
        expected_commit=args.expected_commit,
        expected_sha256=args.code_gate_receipt_sha256,
    )
    short_gate = validate_gate_artifact(
        args.short_window_gate,
        expected_commit=args.expected_commit,
        expected_sha256=args.short_window_gate_sha256,
    )
    pretrain = _require_hashed_file(
        args.pretrain, args.pretrain_sha256, "VideoMAE initialization"
    )
    annotation = _require_hashed_file(
        args.annotation, args.annotation_sha256, "THUMOS14 annotation"
    )
    class_map = _require_hashed_file(
        args.class_map, args.class_map_sha256, "THUMOS14 class map"
    )
    train_data = _path(args.train_data_path)
    _require(train_data.is_dir(), "THUMOS14 training video directory is missing")
    config_file = _path(
        ROOT / args.config if not Path(args.config).is_absolute() else args.config
    )
    _require(
        config_file == _path(ROOT / CONFIG_DEFAULT),
        "numeric gate is fixed to the formal learned-DUCA config",
    )
    cfg = Config.fromfile(str(config_file))
    static = duca_paper_training.validate_static_config(cfg)
    _require(static["variant"] == "duca_fixed_k384", "numeric gate config arm drift")
    cfg.dataset.train.ann_file = annotation["path"]
    cfg.dataset.train.class_map = class_map["path"]
    cfg.dataset.train.data_path = str(train_data)
    cfg.model.backbone.custom.pretrain = pretrain["path"]
    set_seed(GATE_SEED, disable_deterministic=False)
    logger = logging.getLogger(f"duca-paper-numeric-gate-rank{rank}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    dataset = build_dataset(copy.deepcopy(cfg.dataset.train), default_args=dict(logger=logger))
    loader = build_dataloader(
        dataset,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        drop_last=True,
        **cfg.solver.train,
    )
    _require(len(loader) == MAX_ATTEMPTED_UPDATES, "full200 loader is not 100 updates")
    duca_paper_training.derive_train_loader_contract(
        cfg=cfg,
        train_dataset=dataset,
        train_loader=loader,
        world_size=world_size,
    )
    dataset.set_epoch(0)
    loader.sampler.set_epoch(0)

    model = build_detector(copy.deepcopy(cfg.model))
    prepare_optimizer_parameter_freezing(cfg.optimizer, model, logger)
    model = model.to(device)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        static_graph=False,
    )
    optimizer = build_optimizer(copy.deepcopy(cfg.optimizer), model, logger)
    scheduler, _ = build_scheduler(copy.deepcopy(cfg.scheduler), optimizer, len(loader))
    scaler = GradScaler()
    max_amp_retries = int(cfg.workflow.max_amp_retries_per_batch)
    _require(max_amp_retries == 8, "formal AMP replay limit drift")

    original_solver = structured._physical_row_forward_backward
    observer_state: dict[str, Any] = {"candidate": None}

    def observed_solver(node_log_probs, *, k, graph, temperature):
        if (
            observer_state["candidate"] is None
            and int(node_log_probs.numel()) == EXPECTED_T
            and int(k) == EXPECTED_K
        ):
            observer_state["candidate"] = {
                "node_log_probs": node_log_probs.detach().float().cpu().clone(),
                "k": int(k),
                "temperature": float(temperature),
                "graph": _graph_to_cpu(graph),
                "input_dtype": str(node_log_probs.dtype),
                "cuda_autocast_enabled": bool(torch.is_autocast_enabled()),
            }
        return original_solver(
            node_log_probs, k=k, graph=graph, temperature=temperature
        )

    structured._physical_row_forward_backward = observed_solver
    attempted_updates = 0
    successful_updates = 0
    optimizer_attempts = 0
    amp_skipped_attempts = 0
    replay_state_restorations = 0
    max_amp_retries_observed = 0
    target_capture_updates = 0
    local_target_capture = None
    local_legacy = None
    capture_owner_rank = world_size
    capture_successful_update_index = None
    all_training_gradients_finite = True
    trainable_gradient_tensor_count = 0
    try:
        model.train()
        for update_index, cpu_batch in enumerate(loader):
            if update_index >= MAX_ATTEMPTED_UPDATES:
                break
            attempted_updates += 1
            batch = _move_batch(cpu_batch, device)
            retry_count = 0
            rng_state = _capture_rng_state()
            model_buffer_state = _capture_model_buffers(model)
            custom_replay_state = _capture_custom_replay_state(model)
            while True:
                if retry_count > 0:
                    _restore_rng_state(rng_state)
                observer_state["candidate"] = None
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                    losses = model(**batch, return_loss=True)
                _distributed_require(
                    isinstance(losses, Mapping)
                    and "cost" in losses
                    and bool(torch.isfinite(losses["cost"]).all().item()),
                    "production call returned a non-finite training objective",
                    device=device,
                )
                scaler.scale(losses["cost"]).backward()
                scaler.unscale_(optimizer)
                gradient_tensors = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                trainable_gradient_tensor_count = max(
                    trainable_gradient_tensor_count, len(gradient_tensors)
                )
                finite = bool(gradient_tensors) and all(
                    bool(torch.isfinite(gradient).all().item())
                    for gradient in gradient_tensors
                )
                clip = float(cfg.solver.clip_grad_norm)
                if clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                optimizer_attempts += 1
                local_step_ran = float(scaler.get_scale()) >= scale_before
                step_min = torch.tensor(
                    int(local_step_ran), device=device, dtype=torch.int64
                )
                step_max = step_min.clone()
                dist.all_reduce(step_min, op=dist.ReduceOp.MIN)
                dist.all_reduce(step_max, op=dist.ReduceOp.MAX)
                _require(
                    int(step_min.item()) == int(step_max.item()),
                    "AMP optimizer-step outcome diverged across DDP ranks",
                )
                optimizer_step_ran = int(step_min.item()) == 1
                if optimizer_step_ran:
                    _distributed_require(
                        finite,
                        "successful AMP update retained non-finite gradients",
                        device=device,
                    )
                    all_training_gradients_finite = (
                        all_training_gradients_finite and finite
                    )
                    break
                amp_skipped_attempts += 1
                _restore_model_buffers(model, model_buffer_state)
                _restore_custom_replay_state(model, custom_replay_state)
                replay_state_restorations += 1
                retry_count += 1
                max_amp_retries_observed = max(
                    max_amp_retries_observed, retry_count
                )
                _require(
                    retry_count <= max_amp_retries,
                    "formal AMP replay could not produce a successful optimizer "
                    f"update after {max_amp_retries} retries",
                )
            _call_after_optimizer_step(model)
            scheduler.step()
            successful_updates += 1
            capture = observer_state.get("candidate")
            if capture is not None:
                target_capture_updates += 1
            flag = torch.tensor(
                int(capture is not None), device=device, dtype=torch.int64
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if int(flag.item()) == 1:
                owner = torch.tensor(
                    rank if capture is not None else world_size,
                    device=device,
                    dtype=torch.int64,
                )
                dist.all_reduce(owner, op=dist.ReduceOp.MIN)
                capture_owner_rank = int(owner.item())
                update_min = torch.tensor(
                    successful_updates, device=device, dtype=torch.int64
                )
                update_max = update_min.clone()
                dist.all_reduce(update_min, op=dist.ReduceOp.MIN)
                dist.all_reduce(update_max, op=dist.ReduceOp.MAX)
                _require(
                    int(update_min.item()) == int(update_max.item()),
                    "first eligible successful-update index diverged across DDP ranks",
                )
                capture_successful_update_index = int(update_min.item()) - 1
                if rank == capture_owner_rank:
                    _require(capture is not None, "capture owner lacks its tensor")
                    local_target_capture = capture
                    graph = _graph_to_device(capture["graph"], device)
                    local_legacy = _legacy_raw_slot_mass_drift(
                        capture["node_log_probs"].to(device=device),
                        k=int(capture["k"]),
                        graph=graph,
                        temperature=float(capture["temperature"]),
                    )
                break
    finally:
        structured._physical_row_forward_backward = original_solver

    owner_rank = int(capture_owner_rank)
    _require(
        owner_rank < world_size,
        "NO_TARGET_CAPTURE: no eligible successful-update T768/K384 solver call "
        "was captured within 100 successful updates",
    )
    _require(
        capture_successful_update_index is not None,
        "NO_TARGET_CAPTURE: capture index is missing",
    )
    _distributed_require(
        1 <= attempted_updates <= MAX_ATTEMPTED_UPDATES
        and successful_updates == attempted_updates,
        "bounded update execution drift",
        device=device,
    )

    del model, optimizer, scheduler, loader, dataset, losses, batch, cpu_batch
    gc.collect()
    torch.cuda.empty_cache()
    diagnostic = None
    tensor_artifact = None
    tensor_artifact_sha = None
    if rank == owner_rank:
        _require(local_target_capture is not None, "capture owner lacks its captured tensor")
        diagnostic = _run_oracle_diagnostics(local_target_capture, device=device)
    diagnostic_pass = torch.tensor(
        int(rank != owner_rank or bool(diagnostic["passed"])),
        device=device,
        dtype=torch.int64,
    )
    dist.broadcast(diagnostic_pass, src=owner_rank)
    _require(
        int(diagnostic_pass.item()) == 1,
        "frozen FP32/FP64/flow/hard thresholds failed",
    )
    if rank == owner_rank:
        tensor_artifact = output_root / f"captured_solver_input.rank{rank:04d}.pt"
        torch.save(
            {
                "schema_version": "duca_paper_numeric_gate_tensor_v1",
                "node_log_probs": local_target_capture["node_log_probs"],
                "k": int(local_target_capture["k"]),
                "temperature": float(local_target_capture["temperature"]),
                "graph": local_target_capture["graph"],
                "contains_gt": False,
                "contains_predictions": False,
                "contains_metrics": False,
                "contains_loss_values": False,
            },
            tensor_artifact,
        )
        tensor_artifact_sha = _sha256(tensor_artifact)

    rank_payload = {
        "rank": rank,
        "device_name": torch.cuda.get_device_name(device),
        "owner_rank": owner_rank,
        "attempted_updates": attempted_updates,
        "successful_updates": successful_updates,
        "optimizer_attempts": optimizer_attempts,
        "amp_skipped_attempts": amp_skipped_attempts,
        "replay_state_restorations": replay_state_restorations,
        "max_amp_retries_observed": max_amp_retries_observed,
        "target_capture_updates": target_capture_updates,
        "capture_successful_update_index": capture_successful_update_index,
        "all_training_gradients_finite": all_training_gradients_finite,
        "trainable_gradient_tensor_count": trainable_gradient_tensor_count,
        "legacy_diagnostic": (
            {
                "role": "diagnostic_only",
                "admission_effect": "none",
                **local_legacy,
            }
            if rank == owner_rank
            else None
        ),
        "capture": (
            {
                "t": EXPECTED_T,
                "k": EXPECTED_K,
                "valid_len": EXPECTED_T,
                "score_dtype": local_target_capture["input_dtype"],
                "cuda_autocast_enabled": local_target_capture[
                    "cuda_autocast_enabled"
                ],
                "successful_update_index": capture_successful_update_index,
                "selection_policy": (
                    "first_eligible_successful_update_min_rank_first_solver_call"
                ),
            }
            if rank == owner_rank
            else None
        ),
        "diagnostic": diagnostic,
        "tensor_artifact_path": str(tensor_artifact) if tensor_artifact else None,
        "tensor_artifact_sha256": tensor_artifact_sha,
    }
    rank_path = output_root / f"rank{rank:04d}.summary.json"
    _write_new_json(rank_path, rank_payload)
    dist.barrier()

    result = None
    if rank == 0:
        rank_summaries = [
            json.loads((output_root / f"rank{index:04d}.summary.json").read_text(encoding="utf-8"))
            for index in range(world_size)
        ]
        owner_summary = rank_summaries[owner_rank]
        _require(owner_summary["diagnostic"]["passed"], "owner diagnostic is not passed")
        payload = {
            "schema_version": SCHEMA,
            "status": "passed",
            "fail_closed": True,
            "git_commit": git["git_commit"],
            "git_binding": git,
            "slurm_job_id": job_id,
            "slurm_cuda_binding": {
                "world_size": world_size,
                "global_batch_size": EXPECTED_GLOBAL_BATCH_SIZE,
                "logical_devices_per_node": world_size,
                "cuda_visible_devices_supplied_by_slurm": bool(
                    os.environ.get("CUDA_VISIBLE_DEVICES", "")
                ),
                "physical_gpu_index_assumed": False,
                "process_group_timeout_seconds": PROCESS_GROUP_TIMEOUT_SECONDS,
                "outer_process_watchdog_seconds": PROCESS_WATCHDOG_TIMEOUT_SECONDS,
                "elastic_worker_supervision": True,
                "nccl_async_error_handling": bool(
                    os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING", "")
                    or os.environ.get("NCCL_ASYNC_ERROR_HANDLING", "")
                ),
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "device_names_by_rank": [
                    str(row["device_name"]) for row in rank_summaries
                ],
            },
            "prerequisite_clean_linux_code_gate": code_gate,
            "prerequisite_real_short_window_gate": short_gate,
            "config": {
                "path": str(config_file),
                "sha256": _sha256(config_file),
                "resolved_before_runtime_binding_sha256": _canonical_sha256(
                    Config.fromfile(str(config_file)).to_dict()
                ),
                "runtime_resolved_sha256": _canonical_sha256(cfg.to_dict()),
                "arm": static["variant"],
            },
            "assets": {
                "pretrain": pretrain,
                "annotation": annotation,
                "class_map": class_map,
                "train_data_path": str(train_data),
            },
            "bounded_execution": {
                "seed": GATE_SEED,
                "world_size": world_size,
                "global_batch_size": EXPECTED_GLOBAL_BATCH_SIZE,
                "maximum_attempted_updates": MAX_ATTEMPTED_UPDATES,
                "attempted_updates_until_capture": max(
                    int(row["attempted_updates"]) for row in rank_summaries
                ),
                "successful_updates_until_capture": max(
                    int(row["successful_updates"]) for row in rank_summaries
                ),
                "maximum_amp_retries_per_batch": max_amp_retries,
                "optimizer_attempts_by_rank": [
                    int(row["optimizer_attempts"]) for row in rank_summaries
                ],
                "amp_skipped_attempts_by_rank": [
                    int(row["amp_skipped_attempts"]) for row in rank_summaries
                ],
                "max_amp_retries_observed_by_rank": [
                    int(row["max_amp_retries_observed"])
                    for row in rank_summaries
                ],
                "replay_state_restorations_by_rank": [
                    int(row["replay_state_restorations"])
                    for row in rank_summaries
                ],
                "target_capture_updates_by_rank": [
                    int(row["target_capture_updates"]) for row in rank_summaries
                ],
                "capture_owner_rank": owner_rank,
                "first_eligible_successful_update_index": (
                    capture_successful_update_index
                ),
                "first_eligible_successful_capture": True,
                "actual_full_model_forward_backward_optimizer_step": True,
                "all_training_gradients_finite": all(
                    bool(row["all_training_gradients_finite"])
                    for row in rank_summaries
                ),
            },
            "capture": owner_summary["capture"],
            "legacy_production_diagnostic": owner_summary["legacy_diagnostic"],
            "capture_state": (
                "TARGET_CAPTURED_LEGACY_PRESENT"
                if owner_summary["legacy_diagnostic"]["old_guard_triggered"]
                else "TARGET_CAPTURED_LEGACY_ABSENT"
            ),
            "candidate_fp32_oracle_diagnostic": owner_summary["diagnostic"],
            "tensor_artifact_path": owner_summary["tensor_artifact_path"],
            "tensor_artifact_sha256": owner_summary["tensor_artifact_sha256"],
            "rank_summary_sha256": {
                str(index): _sha256(output_root / f"rank{index:04d}.summary.json")
                for index in range(world_size)
            },
            "validation_or_test_data_used": False,
            "checkpoint_created": False,
            "prediction_generated": False,
            "loss_values_recorded": False,
            "metric_accessed": False,
            "paper_metric_claim_allowed": False,
            "paper_method_performance_evidence": False,
            "claim_scope": "engineering_learned_exactk_numeric_stability_only",
            "stage_a_release_prerequisite_satisfied": True,
            "stage_b_enabled": False,
            "official_final_consumed": False,
        }
        payload["content_sha256"] = _canonical_sha256(payload)
        receipt = output_root / "numeric_gate.receipt.json"
        _write_new_json(receipt, payload)
        result = {
            "path": str(receipt),
            "sha256": _sha256(receipt),
            "payload": payload,
        }
    dist.destroy_process_group()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--code-gate-receipt", required=True)
    parser.add_argument("--code-gate-receipt-sha256", required=True)
    parser.add_argument("--short-window-gate", required=True)
    parser.add_argument("--short-window-gate-sha256", required=True)
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--pretrain-sha256", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--annotation-sha256", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--class-map-sha256", required=True)
    parser.add_argument("--train-data-path", required=True)
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_gate(args)
    except BaseException as error:
        _write_failure_receipt_best_effort(args, error)
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        raise
    if result is not None:
        print(json.dumps({"path": result["path"], "sha256": result["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
