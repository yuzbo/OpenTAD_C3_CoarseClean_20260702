from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from mmengine.config import Config
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Subset

from opentad.cores import build_scheduler, train_one_epoch
from opentad.cores.optimizer import (
    assert_optimizer_exact_coverage,
    build_optimizer,
    prepare_optimizer_parameter_freezing,
)
from opentad.datasets import build_dataloader, build_dataset
from opentad.models import build_detector
from opentad.models.duca.structured_selection import exact_uniform_positions
from opentad.models.selectors.duca_online_frame_selector import _gather_time
from opentad.utils import ModelEma, set_seed
from tools.bata.validate_duca_protected_e2e_official60 import validate_config
from tools.bata.duca_frontend_initialization import (
    initialize_frame_selector_from_checkpoint,
)


SCHEMA = "duca_protected_e2e_exact_full_model_gradient_gate_v1"
FORMAL_SEED = 3407
DEFAULT_CONFIG = (
    "configs/adatad/thumos/duca_protected_e2e_fixed384_official60.py"
)


class ExactGateFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExactGateFailure(f"exact full-model protected-E2E gate failed: {message}")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_lf_sha256(path: str | Path) -> str:
    source_bytes = Path(path).read_bytes()
    return hashlib.sha256(source_bytes.replace(b"\r\n", b"\n")).hexdigest()


def _official_asformer_binding(cfg: Config, selector) -> dict[str, str]:
    contract = cfg.get("duca_transition_only_contract", {})
    expected = str(
        contract.get("official_asformer_source_normalized_lf_sha256", "")
    ).lower()
    _require(
        len(expected) == 64 and all(char in "0123456789abcdef" for char in expected),
        "config lacks an official ASFormer normalized-LF SHA256 declaration",
    )
    source = selector.raw_actionness_source
    probe = getattr(source, "probe", None)
    source_metadata = getattr(probe, "official_source", None)
    if not isinstance(source_metadata, Mapping):
        # Older focused fixtures attach provenance to the registered parameter
        # container. Production C3 probes keep it on the probe wrapper.
        source_metadata = getattr(
            getattr(source, "probe_module", None), "official_source", None
        )
    _require(
        isinstance(source_metadata, Mapping),
        "official ASFormer probe did not expose source provenance",
    )
    source_path = Path(str(source_metadata.get("source_file", ""))).expanduser().resolve()
    _require(source_path.is_file(), "official ASFormer source is missing")
    observed = _normalized_lf_sha256(source_path)
    _require(
        observed == expected,
        "official ASFormer normalized-LF SHA256 differs from the config declaration",
    )
    _require(
        str(source_metadata.get("source_normalized_lf_sha256", "")).lower() == observed,
        "official ASFormer runtime provenance differs from the reopened source",
    )
    return {
        "path": str(source_path),
        "sha256": _sha256(source_path),
        "normalized_lf_sha256": observed,
        "config_declared_normalized_lf_sha256": expected,
    }


def _boundary_validity_evidence(
    boundary_validity: Any,
    gt_segments: list[torch.Tensor],
) -> dict[str, int]:
    _require(
        isinstance(boundary_validity, (list, tuple)),
        "real loader did not provide gt_boundary_validity as a per-video sequence",
    )
    _require(
        len(boundary_validity) == len(gt_segments),
        "gt_boundary_validity batch size differs from gt_segments",
    )
    endpoint_count = 0
    valid_endpoint_count = 0
    for index, (value, segments) in enumerate(zip(boundary_validity, gt_segments)):
        validity = torch.as_tensor(value)
        _require(
            validity.dtype == torch.bool and tuple(validity.shape) == tuple(segments.shape),
            f"gt_boundary_validity row {index} is not a boolean [N,2] pipeline artifact",
        )
        endpoint_count += int(validity.numel())
        valid_endpoint_count += int(validity.sum().item())
    _require(endpoint_count > 0, "real full-model batch has no boundary-validity endpoints")
    return {
        "batch_size": len(boundary_validity),
        "endpoint_count": endpoint_count,
        "valid_endpoint_count": valid_endpoint_count,
    }


def _record_boundary_validity_consumption(selector, boundary_validity, evidence):
    original_forward_train = selector.forward_train
    audit = {"selector_forward_train_calls": 0, "training_forward_train_calls": 0}

    def recording_forward_train(*args, **kwargs):
        observed = kwargs.get("gt_boundary_validity")
        _require(
            isinstance(observed, (list, tuple))
            and len(observed) == len(boundary_validity)
            and all(
                torch.equal(
                    torch.as_tensor(left).detach().cpu(),
                    torch.as_tensor(right).detach().cpu(),
                )
                for left, right in zip(observed, boundary_validity)
            ),
            "selector did not consume the exact gt_boundary_validity values emitted by the real loader",
        )
        _require(
            _boundary_validity_evidence(observed, kwargs["gt_segments"]) == evidence,
            "selector consumed altered gt_boundary_validity rather than the real loader artifact",
        )
        audit["selector_forward_train_calls"] += 1
        return original_forward_train(*args, **kwargs)

    selector.forward_train = recording_forward_train
    return original_forward_train, audit


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def _bind_exact_runtime(expected_commit: str) -> dict[str, Any]:
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
    _require(os.environ.get("SLURM_JOB_ID") is not None, "Slurm allocation is required")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(torch.cuda.device_count() == 1, "exactly one Slurm logical GPU is required")
    _require(int(os.environ.get("WORLD_SIZE", "0")) == 1, "torchrun WORLD_SIZE must be one")
    _require(int(os.environ.get("LOCAL_RANK", "-1")) == 0, "LOCAL_RANK must be zero")
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return {
        "git_commit": expected,
        "git_tree": _git_output("rev-parse", "HEAD^{tree}"),
        "slurm_job_id": str(os.environ["SLURM_JOB_ID"]),
        "world_size": dist.get_world_size(),
        "rank": dist.get_rank(),
        "logical_device": "cuda:0",
        "device_name": torch.cuda.get_device_name(0),
    }


def _cuda_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    _require("gt_boundary_validity" in batch, "real loader omitted gt_boundary_validity")
    return {
        "inputs": batch["inputs"].to("cuda:0", non_blocking=True),
        "masks": batch["masks"].to("cuda:0", dtype=torch.bool, non_blocking=True),
        "metas": [dict(meta) for meta in batch["metas"]],
        "gt_segments": [
            value.to("cuda:0", non_blocking=True) for value in batch["gt_segments"]
        ],
        "gt_labels": [
            value.to("cuda:0", non_blocking=True) for value in batch["gt_labels"]
        ],
        # Keep the unmodified transform output so crop endpoint validity reaches training.
        "gt_boundary_validity": batch["gt_boundary_validity"],
    }


class _OneCudaBatchLoader:
    def __init__(self, batch: Mapping[str, Any]) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        yield self.batch


def _real_full_batch(cfg: Config) -> dict[str, Any]:
    dataset = build_dataset(copy.deepcopy(cfg.dataset.train), default_args={"logger": None})
    loader_cfg = copy.deepcopy(cfg.solver.train)
    loader_cfg["batch_size"] = 2
    loader = build_dataloader(
        Subset(dataset, list(range(min(len(dataset), 64)))),
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **loader_cfg,
    )
    for raw_batch in loader:
        batch = _cuda_batch(raw_batch)
        if (
            int(batch["masks"].shape[0]) == 2
            and bool(
                torch.all(
                    batch["masks"].sum(dim=1) == int(cfg.dense_window_size)
                ).item()
            )
        ):
            return batch
    raise ExactGateFailure("no two-row full T=768 real THUMOS training batch was found")


def _is_action_head(name: str) -> bool:
    marker = ".official_temporal."
    if marker not in name:
        return False
    tail = name.split(marker, 1)[1]
    parts = tail.split(".")
    return tail.startswith("encoder.conv_out.") or (
        len(parts) >= 4 and parts[0] == "decoders" and parts[2] == "conv_out"
    )


def _grad_sum(model, predicate: Callable[[str], bool]) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if not predicate(name) or parameter.grad is None:
            continue
        _require(torch.isfinite(parameter.grad).all().item(), f"non-finite gradient: {name}")
        total += float(parameter.grad.detach().abs().sum().item())
    return total


def _gradient_partition(model) -> dict[str, float]:
    selector = model.frame_selector
    layers = selector.raw_actionness_source.probe_module.official_temporal.encoder.layers
    last_prefix = (
        "frame_selector.raw_actionness_source.probe_module."
        f"official_temporal.encoder.layers.{len(layers) - 1}."
    )
    probe_prefix = "frame_selector.raw_actionness_source.probe_module."
    return {
        "detector": _grad_sum(model, lambda name: not name.startswith("frame_selector.")),
        "selector_scorer": _grad_sum(
            model,
            lambda name: name.startswith("frame_selector.adapter.transition_scorer."),
        ),
        "density_mixture_head": _grad_sum(
            model,
            lambda name: name.startswith(
                "frame_selector.adapter.density_mixture_head."
            ),
        ),
        "asformer_last_encoder_layer": _grad_sum(
            model,
            lambda name: name.startswith(last_prefix),
        ),
        "asformer_earlier_or_spatial": _grad_sum(
            model,
            lambda name: name.startswith(probe_prefix)
            and not name.startswith(last_prefix)
            and not _is_action_head(name),
        ),
        "action_head": _grad_sum(model, _is_action_head),
    }


def _finite_backward(
    objective: torch.Tensor,
    *,
    model,
    optimizer,
) -> dict[str, float]:
    _require(objective.ndim == 0 and torch.isfinite(objective).item(), "objective is non-finite")
    objective.backward()
    partition = _gradient_partition(model)
    optimizer.zero_grad(set_to_none=True)
    return partition


def _transition_output_weight(model) -> tuple[str, torch.nn.Parameter]:
    scorer = model.frame_selector.adapter.transition_scorer
    parameter = scorer.net[-1].weight
    for name, candidate in model.named_parameters():
        if candidate is parameter:
            return name, parameter
    raise ExactGateFailure("transition scorer output weight is absent from the model")


def _detector_output_weight(model) -> tuple[str, torch.nn.Parameter]:
    candidates = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("rpn_head.")
        and parameter.requires_grad
        and parameter.ndim >= 2
    ]
    if not candidates:
        raise ExactGateFailure("ActionFormerHead exposes no trainable output weight")
    return candidates[-1]


def _optimizer_state_step(optimizer, parameter: torch.nn.Parameter) -> int:
    step = optimizer.state.get(parameter, {}).get("step", 0)
    if torch.is_tensor(step):
        step = step.detach().cpu().item()
    return int(step)


def _position_scheduler_at_successful_step(scheduler, *, step: int) -> list[float]:
    _require(step > 0, "scheduler proof position must be positive")
    scheduler.last_epoch = int(step)
    learning_rates = [float(value) for value in scheduler._get_closed_form_lr()]
    _require(
        len(learning_rates) == len(scheduler.optimizer.param_groups),
        "scheduler LR group count drifted",
    )
    _require(
        any(value > 0.0 for value in learning_rates),
        "scheduler proof position must expose a non-zero LR",
    )
    for group, learning_rate in zip(scheduler.optimizer.param_groups, learning_rates):
        group["lr"] = learning_rate
    scheduler._last_lr = list(learning_rates)
    scheduler._step_count = max(int(getattr(scheduler, "_step_count", 0)), 1)
    return learning_rates


def run_gate(
    *,
    config_path: str,
    expected_commit: str,
    adatad_pretrain: str,
    adatad_pretrain_sha256: str,
    output_json: str,
) -> dict[str, Any]:
    output = Path(output_json).expanduser().resolve()
    _require(not output.exists(), "refusing to overwrite exact gate evidence")
    try:
        output.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ExactGateFailure("gate evidence must be outside the Git worktree")

    runtime = _bind_exact_runtime(expected_commit)
    set_seed(FORMAL_SEED, disable_deterministic=False)
    runtime["formal_seed"] = FORMAL_SEED
    contract = validate_config(ROOT / config_path)
    cfg = Config.fromfile(str(ROOT / config_path))
    pretrain = Path(adatad_pretrain).expanduser().resolve()
    _require(pretrain.is_file(), "AdaTAD VideoMAE pretrain is missing")
    _require(
        _sha256(pretrain) == str(adatad_pretrain_sha256).lower(),
        "AdaTAD pretrain SHA256 mismatch",
    )
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.backbone.custom.pretrain = str(pretrain)
    model = build_detector(model_cfg)
    selector_initialization = initialize_frame_selector_from_checkpoint(
        model,
        cfg.workflow.get("selector_initialization", None),
    )
    _require(model.__class__.__name__ == "ActionFormer", "detector is not ActionFormer")
    _require(
        model.rpn_head.__class__.__name__ == "ActionFormerHead",
        "detector head is not ActionFormerHead",
    )
    _require(
        model.backbone.__class__.__name__ != "_ProofTemporalMeanBackbone",
        "proof backbone substitution is forbidden",
    )
    logger = logging.getLogger("duca-protected-exact-gate")
    prepare_optimizer_parameter_freezing(copy.deepcopy(cfg.optimizer), model, logger)
    model.to("cuda:0")
    optimizer = build_optimizer(
        copy.deepcopy(cfg.optimizer),
        SimpleNamespace(module=model),
        logger,
    )
    assert_optimizer_exact_coverage(model, optimizer)
    model.train()
    selector = model.frame_selector
    official_asformer_binding = _official_asformer_binding(cfg, selector)
    route = str(cfg.duca_transition_only_contract.route)
    exact_uniform_route = route == "DUCA_EXACT_UNIFORM_FIXED384_OFFICIAL60"
    two_stage_route = str(cfg.duca_transition_only_contract.get("stage", "")) == (
        "uniform_detector_cowarmup_then_joint_detection"
    )
    batch = _real_full_batch(cfg)
    boundary_validity_evidence = _boundary_validity_evidence(
        batch["gt_boundary_validity"], batch["gt_segments"]
    )
    original_selector_forward_train, boundary_validity_audit = (
        _record_boundary_validity_consumption(
            selector,
            batch["gt_boundary_validity"],
            boundary_validity_evidence,
        )
    )
    uniform_warmup_audit = None
    if two_stage_route:
        warmup_state = selector._loss_schedule_state()
        warmup_weights = warmup_state.get("weights", {})
        _require(
            all(
                float(warmup_weights.get(name, float("nan"))) == 0.0
                for name in (
                    "actionness",
                    "transition",
                    "transition_boundary",
                )
            ),
            "two-stage uniform warmup has nonzero frontend loss weights",
        )
        _require(
            float(warmup_weights.get("policy_alpha", float("nan"))) == 0.0
            and float(warmup_state.get("detector_gradient_weight", float("nan")))
            == 0.0,
            "two-stage uniform warmup activated policy or detector bridge",
        )
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
            cache_enabled=False,
        ):
            warmup_outputs = selector.forward_train(
                inputs=batch["inputs"],
                masks=batch["masks"],
                metas=batch["metas"],
                gt_segments=batch["gt_segments"],
                gt_labels=batch["gt_labels"],
                gt_boundary_validity=batch["gt_boundary_validity"],
            )
        expected_uniform = exact_uniform_positions(
            int(cfg.dense_window_size),
            int(cfg.model.frame_selector.budget),
            device=selector._last_selected_positions.device,
        )
        _require(
            torch.equal(
                selector._last_selected_positions,
                expected_uniform[None, :].expand_as(
                    selector._last_selected_positions
                ),
            ),
            "two-stage warmup did not feed canonical exact-uniform positions",
        )
        warmup_frontend_cost = sum(
            float(warmup_outputs["losses"][name].detach().float().item())
            for name in (
                "actionness_bce_loss",
                "transition_distribution_loss",
                "transition_boundary_coverage_loss",
            )
        )
        _require(
            warmup_frontend_cost == 0.0,
            "two-stage detector warmup retained a frontend optimization cost",
        )
        selector._pending_loss_schedule_advance = False
        uniform_warmup_audit = {
            "successful_update_range": [0, 999],
            "selected_positions": "canonical_round_linspace_k384",
            "frontend_loss_weights_exactly_zero": True,
            "detector_gradient_bridge_weight": 0.0,
            "frontend_gradient_clipping_coupling": False,
        }
    bridge_ready = int(
        selector.loss_weight_schedule["detector_gradient"]["warmup_steps"]
    ) + int(selector.loss_weight_schedule["detector_gradient"]["transition_steps"])
    selector._loss_weight_schedule_step.fill_(bridge_ready)
    expected_bridge_weight = float(
        selector.loss_weight_schedule["detector_gradient"]["end"]
    )
    _require(
        float(selector._loss_schedule_state()["detector_gradient_weight"])
        == expected_bridge_weight,
        "detector bridge did not reach the route endpoint",
    )
    def selector_route(loss_names: tuple[str, ...]) -> dict[str, float]:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
            cache_enabled=False,
        ):
            outputs = selector.forward_train(
                inputs=batch["inputs"],
                masks=batch["masks"],
                metas=batch["metas"],
                gt_segments=batch["gt_segments"],
                gt_labels=batch["gt_labels"],
                gt_boundary_validity=batch["gt_boundary_validity"],
            )
            route_losses = outputs["losses"]
            _require(
                set(loss_names).issubset(route_losses),
                f"selector route losses are missing: {sorted(set(loss_names) - set(route_losses))}",
            )
            objective = sum(route_losses[name] for name in loss_names)
        return _finite_backward(objective, model=model, optimizer=optimizer)

    coarse_frozen = bool(selector.raw_actionness_source.frozen)
    frontend_objectives_disabled = exact_uniform_route and two_stage_route
    if coarse_frozen or frontend_objectives_disabled:
        optimizer.zero_grad(set_to_none=True)
        action_gradients = _gradient_partition(model)
    else:
        action_gradients = selector_route(("actionness_bce_loss",))
    if frontend_objectives_disabled:
        optimizer.zero_grad(set_to_none=True)
        transition_gradients = _gradient_partition(model)
    else:
        transition_gradients = selector_route(
            ("transition_distribution_loss", "transition_boundary_coverage_loss")
        )

    captured_inputs: list[torch.Tensor] = []

    def capture_backbone_input(_module, args):
        captured_inputs.append(args[0].detach().clone())

    hook = model.backbone.register_forward_pre_hook(capture_backbone_input)
    ddp = DistributedDataParallel(
        model,
        device_ids=[0],
        output_device=0,
        find_unused_parameters=True,
        static_graph=False,
    )
    optimizer.zero_grad(set_to_none=True)
    try:
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
            detector_objective = model._duca_detector_objective(losses)
        detector_gradients = _finite_backward(
            detector_objective,
            model=model,
            optimizer=optimizer,
        )
    finally:
        hook.remove()
    _require(len(captured_inputs) == 1, "real detector backbone was not called exactly once")
    positions = selector._last_selected_positions
    slot_mask = positions >= 0
    expected_hard = _gather_time(batch["inputs"], positions, slot_mask)
    _require(
        torch.equal(captured_inputs[0], expected_hard),
        "real AdaTAD backbone input differs from exact hard gather",
    )
    temporal_sampling_contract = getattr(
        selector, "temporal_sampling_contract", None
    )
    if temporal_sampling_contract is None:
        observed_holes = []
        for row in positions:
            valid_positions = row[row >= 0]
            _require(
                int(valid_positions.numel()) == int(selector.budget),
                "density transport did not emit exact K",
            )
            _require(
                bool(torch.all(valid_positions[1:] > valid_positions[:-1]).item()),
                "density transport positions are not strictly increasing",
            )
            sentinels = torch.cat(
                (
                    valid_positions.new_tensor([-1]),
                    valid_positions,
                    valid_positions.new_tensor([int(cfg.dense_window_size)]),
                )
            )
            observed_holes.append(
                int((sentinels[1:] - sentinels[:-1] - 1).max().item())
            )
        if selector.max_unselected_hole is not None:
            _require(
                max(observed_holes) <= int(selector.max_unselected_hole),
                "density hard-max projection violated its configured cap",
            )
        temporal_audit = {
            "contract_kind": "continuous_density_exact_k",
            "hard_budget": int(selector.budget),
            "configured_max_unselected_hole": selector.max_unselected_hole,
            "observed_max_unselected_hole": observed_holes,
            "strictly_increasing_unique": True,
        }
    else:
        temporal_audit = temporal_sampling_contract.audit_positions(
            positions,
            batch["masks"],
        )
    if exact_uniform_route:
        expected_uniform = exact_uniform_positions(
            int(cfg.dense_window_size),
            int(cfg.model.frame_selector.budget),
            device=positions.device,
        )
        _require(
            torch.equal(
                positions,
                expected_uniform[None, :].expand_as(positions),
            ),
            "exact-uniform control did not emit canonical endpoint positions",
        )
    companion_fraction = float(
        getattr(selector, "training_uniform_companion_fraction", 0.0)
    )
    companion_count = int(
        model.frame_selector.last_forward_summary.get(
            "training_uniform_companion_count",
            0,
        )
    )
    if companion_fraction > 0.0:
        _require(
            companion_count == 1,
            "two-row Uni companion gate must route one uniform and one learned row",
        )
        normalize_learned_gradient = bool(
            getattr(
                selector,
                "training_uniform_companion_normalize_learned_gradient",
                False,
            )
        )
        companion_bridge_scale = model.frame_selector.last_forward_summary.get(
            "training_uniform_companion_bridge_scale",
            [],
        )
        if normalize_learned_gradient:
            _require(
                sorted(float(value) for value in companion_bridge_scale)
                == [0.0, 2.0],
                "two-row companion did not preserve matched learned-row gradient exposure",
            )
    else:
        _require(
            companion_count == 0,
            "non-companion gate unexpectedly routed a uniform training row",
        )
        normalize_learned_gradient = False
        companion_bridge_scale = model.frame_selector.last_forward_summary.get(
            "training_uniform_companion_bridge_scale",
            [],
        )

    rho_arm = route == "DUCA_PROTECTED_E2E_RHO_FIXED384_OFFICIAL60"
    _require(detector_gradients["detector"] > 0.0, "detector loss missed detector parameters")
    _require(detector_gradients["action_head"] == 0.0, "detector loss leaked into action head")
    detector_feedback_enabled = expected_bridge_weight > 0.0
    if exact_uniform_route or not detector_feedback_enabled:
        _require(
            detector_gradients["selector_scorer"] == 0.0
            and detector_gradients["asformer_last_encoder_layer"] == 0.0
            and detector_gradients["asformer_earlier_or_spatial"] == 0.0,
            "exact-uniform detector loss leaked into the selector or coarse probe",
        )
    elif rho_arm:
        _require(
            detector_gradients["selector_scorer"] > 0.0,
            "rho detector loss missed selector scorer",
        )
        _require(
            detector_gradients["asformer_last_encoder_layer"] > 0.0,
            "rho detector loss missed the last ASFormer layer",
        )
        _require(
            detector_gradients["asformer_earlier_or_spatial"] == 0.0,
            "rho detector loss leaked before the last ASFormer layer",
        )
    else:
        _require(
            detector_gradients["selector_scorer"] > 0.0,
            "protected detector loss missed selector scorer",
        )
        _require(
            detector_gradients["asformer_last_encoder_layer"] == 0.0
            and detector_gradients["asformer_earlier_or_spatial"] == 0.0,
            "protected detector loss leaked into ASFormer",
        )
    mixture_density = (
        selector.acquisition_policy == "continuous_mixture_density_transport"
    )
    if mixture_density and detector_feedback_enabled:
        _require(
            detector_gradients["density_mixture_head"] > 0.0,
            "detector loss missed the boundary-uncertainty-context density head",
        )
    else:
        _require(
            detector_gradients["density_mixture_head"] == 0.0,
            "a non-mixture route unexpectedly updated the density mixture head",
        )
    _require(
        action_gradients["detector"] == 0.0
        and action_gradients["selector_scorer"] == 0.0,
        "action BCE leaked outside the coarse probe",
    )
    if coarse_frozen or frontend_objectives_disabled:
        _require(
            action_gradients["action_head"] == 0.0
            and action_gradients["asformer_last_encoder_layer"] == 0.0
            and action_gradients["asformer_earlier_or_spatial"] == 0.0,
            "frozen coarse probe unexpectedly received action gradients",
        )
    else:
        _require(
            action_gradients["action_head"] > 0.0
            and action_gradients["asformer_last_encoder_layer"] > 0.0
            and action_gradients["asformer_earlier_or_spatial"] > 0.0,
            "action BCE did not train the complete coarse probe",
        )
    _require(
        transition_gradients["detector"] == 0.0
        and transition_gradients["action_head"] == 0.0,
        "transition supervision leaked into detector or action head",
    )
    if frontend_objectives_disabled:
        _require(
            transition_gradients["selector_scorer"] == 0.0,
            "disabled exact-uniform frontend objective reached the selector scorer",
        )
    else:
        _require(
            transition_gradients["selector_scorer"] > 0.0,
            "transition supervision missed the selector scorer",
        )
    if coarse_frozen or frontend_objectives_disabled:
        _require(
            transition_gradients["asformer_last_encoder_layer"] == 0.0
            and transition_gradients["asformer_earlier_or_spatial"] == 0.0,
            "transition supervision leaked into the frozen coarse probe",
        )
    else:
        _require(
            transition_gradients["asformer_last_encoder_layer"] > 0.0
            and transition_gradients["asformer_earlier_or_spatial"] > 0.0,
            "transition supervision missed the trainable coarse probe",
        )

    # Reuse the production train engine for one real AMP update. The forced
    # first overflow proves that the same batch is replayed and that no
    # optimizer, selector schedule, LR scheduler, or EMA state advances until
    # a finite update actually succeeds.
    selector._pending_loss_schedule_advance = False
    scheduler, _ = build_scheduler(
        copy.deepcopy(cfg.scheduler),
        optimizer,
        int(cfg.workflow.expected_train_batches_per_epoch),
    )
    initial_scheduler_lrs = _position_scheduler_at_successful_step(
        scheduler,
        step=1,
    )
    model_ema = ModelEma(ddp)
    ema_root = getattr(model_ema.module, "module", model_ema.module)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    parameter_name, parameter = (
        _detector_output_weight(model)
        if exact_uniform_route
        else _transition_output_weight(model)
    )
    ema_parameter = dict(ema_root.named_parameters())[parameter_name]
    parameter_before = parameter.detach().clone()
    ema_before = ema_parameter.detach().clone()
    selector_step_before = int(selector._loss_weight_schedule_step.item())
    scheduler_epoch_before = int(scheduler.last_epoch)
    scale_before = float(scaler.get_scale())
    update_audit: dict[str, Any] = {}
    selector_calls_before_training = int(
        boundary_validity_audit["selector_forward_train_calls"]
    )
    training_probe = train_one_epoch(
        _OneCudaBatchLoader(batch),
        ddp,
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
    boundary_validity_audit["training_forward_train_calls"] = int(
        boundary_validity_audit["selector_forward_train_calls"]
    ) - selector_calls_before_training
    _require(
        boundary_validity_audit["training_forward_train_calls"] > 0,
        "production train_one_epoch did not consume gt_boundary_validity",
    )
    selector.forward_train = original_selector_forward_train
    _require(isinstance(training_probe, Mapping), "training engine returned no update probe")
    skipped_attempts = int(update_audit.get("amp_skipped_attempts", -1))
    optimizer_attempts = int(update_audit.get("optimizer_attempts", -1))
    _require(
        int(update_audit.get("attempted_batches", -1)) == 1
        and int(update_audit.get("successful_optimizer_updates", -1)) == 1
        and int(update_audit.get("replay_exhaustions", -1)) == 0
        and int(update_audit.get("scheduler_updates", -1)) == 1
        and int(update_audit.get("ema_updates", -1)) == 1
        and int(update_audit.get("duca_schedule_updates", -1)) == 1
        and int(update_audit.get("forced_amp_overflow_attempts", -1)) == 1,
        "production AMP replay/update counters drifted",
    )
    _require(
        skipped_attempts >= 1
        and optimizer_attempts == skipped_attempts + 1
        and int(update_audit.get("replayed_batches", -1)) == 1
        and int(update_audit.get("replay_state_restorations", -1)) == skipped_attempts,
        "AMP overflow was not replayed on the same batch",
    )
    _require(
        int(training_probe.get("attempted_steps", -1)) == optimizer_attempts
        and int(training_probe.get("skipped_optimizer_steps", -1)) == skipped_attempts
        and int(training_probe.get("successful_optimizer_steps", -1)) == 1
        and int(training_probe.get("finite_loss_steps", -1)) == optimizer_attempts
        and int(training_probe.get("finite_gradient_steps", -1)) == 1,
        "training probe disagrees with the AMP replay audit",
    )
    _require(
        int(selector._loss_weight_schedule_step.item()) == selector_step_before + 1
        and int(scheduler.last_epoch) == scheduler_epoch_before + 1,
        "successful update did not advance selector and LR schedules exactly once",
    )
    parameter_after = parameter.detach().clone()
    ema_after = ema_parameter.detach().clone()
    parameter_delta = float((parameter_after - parameter_before).abs().max().item())
    ema_delta = float((ema_after - ema_before).abs().max().item())
    gradient_sum = (
        float(parameter.grad.detach().abs().sum().item())
        if parameter.grad is not None
        else 0.0
    )
    _require(parameter_delta > 0.0, "real optimizer step did not change the selector scorer")
    _require(
        math.isfinite(gradient_sum) and gradient_sum > 0.0,
        "successful replay left no finite selector scorer gradient",
    )
    _require(ema_delta > 0.0, "EMA did not update the selector scorer")
    expected_ema = model_ema.decay * ema_before + (1.0 - model_ema.decay) * parameter_after
    ema_error = float((ema_after - expected_ema).abs().max().item())
    _require(ema_error <= 1.0e-7, "EMA shadow differs from its configured update")
    _require(bool(optimizer.state), "optimizer state is empty after the real update")
    _require(
        _optimizer_state_step(optimizer, parameter) == 1,
        "selector scorer optimizer state did not advance exactly once",
    )
    ema_selector = getattr(ema_root, "frame_selector", None)
    _require(
        ema_selector is not None
        and int(ema_selector._loss_weight_schedule_step.item())
        == selector_step_before + 1,
        "EMA selector schedule buffer differs from the trained model",
    )
    optimizer_step_audit = {
        "successful_optimizer_updates": 1,
        "forced_amp_overflow_attempts": 1,
        "amp_skipped_attempts": skipped_attempts,
        "optimizer_attempts": optimizer_attempts,
        "active_parameter": parameter_name,
        "active_parameter_gradient_abs_sum": gradient_sum,
        "active_parameter_max_abs_change": parameter_delta,
        "active_parameter_optimizer_state_step": _optimizer_state_step(
            optimizer,
            parameter,
        ),
        "ema_parameter_max_abs_change": ema_delta,
        "ema_formula_max_abs_error": ema_error,
        "scaler_before": scale_before,
        "scaler_after": float(scaler.get_scale()),
        "selector_step_before": selector_step_before,
        "selector_step_after": int(selector._loss_weight_schedule_step.item()),
        "scheduler_epoch_before": scheduler_epoch_before,
        "scheduler_epoch_after": int(scheduler.last_epoch),
        "initial_nonzero_scheduler_lrs": initial_scheduler_lrs,
        "ema_updated": True,
        "update_audit": update_audit,
        "training_probe": dict(training_probe),
    }
    payload = {
        "schema": SCHEMA,
        "ok": True,
        "status": "p1_p2_exact_full_model_amp_ddp_gate_passed",
        "runtime": runtime,
        "config_contract": contract,
        "config_sha256": _sha256(ROOT / config_path),
        "adatad_pretrain": {
            "path": str(pretrain),
            "sha256": str(adatad_pretrain_sha256).lower(),
        },
        "official_asformer_source": official_asformer_binding,
        "real_thumos_loader_executed": True,
        "gt_boundary_validity": {
            **boundary_validity_evidence,
            **boundary_validity_audit,
            "preserved_from_real_loader": True,
            "consumed_by_production_training": True,
        },
        "real_model": {
            "detector": model.__class__.__name__,
            "backbone": model.backbone.__class__.__name__,
            "head": model.rpn_head.__class__.__name__,
            "proof_backbone_substitution": False,
        },
        "numeric_contract": {
            "amp": True,
            "autocast_dtype": "float16",
            "grad_scaler": True,
            "ddp": True,
            "find_unused_parameters": True,
            "static_graph": False,
        },
        "hard_forward_equals_real_backbone_input": True,
        "temporal_sampling_audit": temporal_audit,
        "training_uniform_companion": {
            "fraction": companion_fraction,
            "count": companion_count,
            "batch_size": int(batch["inputs"].shape[0]),
            "normalize_learned_gradient": normalize_learned_gradient,
            "bridge_scale": companion_bridge_scale,
            "inference_extra_cost": False,
        },
        "uniform_detector_warmup": uniform_warmup_audit,
        "gradient_ownership": {
            "detector_only": detector_gradients,
            "action_bce_only": action_gradients,
            "transition_auxiliary_only": transition_gradients,
        },
        "selector_initialization": selector_initialization,
        "coarse_probe_frozen": coarse_frozen,
        "optimizer_exact_coverage": True,
        "real_optimizer_step_audit": optimizer_step_audit,
        "paper_claim_allowed": False,
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
    parser.add_argument("--adatad-pretrain", required=True)
    parser.add_argument("--adatad-pretrain-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    try:
        run_gate(
            config_path=args.config,
            expected_commit=args.expected_commit,
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
                    "status": "p1_p2_exact_full_model_gate_failed",
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
