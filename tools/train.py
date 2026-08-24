import os
import random
import re
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import json
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import train_one_epoch, val_one_epoch, eval_one_epoch, build_optimizer, build_scheduler
from opentad.utils import (
    set_seed,
    update_workdir,
    create_folder,
    save_config,
    setup_logger,
    ModelEma,
    save_checkpoint,
    save_best_checkpoint,
)


ZOOMTOKEN_RECOVERY_SCHEMA = "zoomtoken_same_cell_recovery_v001"
ZOOMTOKEN_RECOVERY_ARMS = {"DN", "G", "R1"}
ZOOMTOKEN_RECOVERY_ARMS.update(
    {
        "R2",
        "R2-SHUF48",
        "Q48-GLOBAL",
        "R3",
        "R3-AREA-SHIFT",
        "R4",
        "R4-SHUF15",
        "Q64-GLOBAL",
        "R1-DROP32",
        "R1-MOD32-KV",
        "R1-RC32-KV",
        "R1-DSR6-KV",
        "R1-APM32-CTX64",
        "R1-CUR32-CTX64",
        "AMOD50",
    }
)
ZOOMTOKEN_UPDATE_INDEX_ARMS = ZOOMTOKEN_RECOVERY_ARMS - {"DN"}
ZOOMTOKEN_R3_ARMS = {"R3", "R3-AREA-SHIFT"}
ZOOMTOKEN_CANONICAL_SOURCE_ARMS = ZOOMTOKEN_RECOVERY_ARMS - {"DN", "G"}
ZOOMTOKEN_TEMPORAL_PREFLIGHT_ARMS = {
    "R1-APM32-CTX64": "apm32_ctx64",
    "R1-CUR32-CTX64": "cur32_ctx64",
}


class _SingleBatchLoader:
    """Expose exactly one production-loader batch without changing its sampler."""

    def __init__(self, loader):
        self.loader = loader
        self.sampler = loader.sampler

    def __len__(self):
        return 1

    def __iter__(self):
        for batch in self.loader:
            yield batch
            return
        raise RuntimeError("ZoomToken mechanical preflight received an empty train loader")


def _assert_nested_state_equal(actual, expected, path="state"):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(
            actual.detach().cpu(), expected.detach().cpu()
        ):
            raise RuntimeError(f"ZoomToken recovery failed to restore {path}")
        return
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or not np.array_equal(actual, expected):
            raise RuntimeError(f"ZoomToken recovery failed to restore {path}")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RuntimeError(f"ZoomToken recovery key set changed at {path}")
        for key in expected:
            _assert_nested_state_equal(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise RuntimeError(f"ZoomToken recovery sequence changed at {path}")
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            _assert_nested_state_equal(
                actual_value,
                expected_value,
                f"{path}[{index}]",
            )
        return
    if actual != expected:
        raise RuntimeError(f"ZoomToken recovery value changed at {path}")


def _zoomtoken_temporal_preflight_summary(model, recovery_contract):
    expected_mode = ZOOMTOKEN_TEMPORAL_PREFLIGHT_ARMS.get(
        recovery_contract["arm_surface"]
    )
    if expected_mode is None:
        raise ValueError("mechanical temporal preflight accepts only APM32/CUR32")
    candidate_model = getattr(model, "module", model)
    route_backbone = getattr(candidate_model, "backbone", None)
    vit_backbone = getattr(getattr(route_backbone, "model", None), "backbone", None)
    summary = getattr(vit_backbone, "latest_native_packed_summary", None)
    if not isinstance(summary, dict):
        raise RuntimeError("temporal preflight produced no native-ragged execution ledger")
    expected = {
        "refresh_execution_mode": expected_mode,
        "heavy_backbone_forward_count": 1,
        "padded_heavy_tokens_per_window": 0,
    }
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            raise RuntimeError(
                f"temporal preflight ledger mismatch for {key!r}: "
                f"expected {expected_value!r}, got {summary.get(key)!r}"
            )
    requested = int(summary.get("requested_physical_tokens_per_window", -1))
    unique = int(summary.get("unique_physical_tokens_per_window", -1))
    executed = int(summary.get("executed_patch_tokens_per_window", -1))
    if requested <= 0 or (requested, unique, executed) != (
        requested,
        requested,
        requested,
    ):
        raise RuntimeError("temporal preflight lost exact K64 physical support")
    refresh_by_batch = summary.get("refresh_query_tokens_per_window_by_batch")
    if (
        not isinstance(refresh_by_batch, list)
        or len(refresh_by_batch) != int(summary.get("batch_size", -1))
        or any(
            not isinstance(value, int)
            or value < requested // 2
            or value > requested
            for value in refresh_by_batch
        )
    ):
        raise RuntimeError("temporal preflight refresh ledger is outside K32/K64")
    alignment = summary.get("temporal_alignment")
    if not isinstance(alignment, dict):
        raise RuntimeError("temporal preflight produced no alignment ledger")
    alignment_expected = {
        "carrier_mode": expected_mode,
        "memory_lifetime_tubelets": 1,
        "clip_reset_tubelets": 8,
        "similarity_threshold": 0.8,
        "search_radius": 2,
        "new_trainable_parameters": 0,
        "previous_memory_detached": True,
        "current_position_restored": True,
        "future_tubelet_access": False,
    }
    for key, expected_value in alignment_expected.items():
        if alignment.get(key) != expected_value:
            raise RuntimeError(f"temporal alignment ledger mismatch for {key!r}")
    total_tubelets = int(alignment.get("total_tubelets", -1))
    refreshed = int(alignment.get("refreshed_tokens", -1))
    retained = int(alignment.get("retained_tokens", -1))
    fallback = int(alignment.get("fallback_tubelets", -1))
    normal = int(alignment.get("normal_tubelets", -1))
    if total_tubelets <= 0 or fallback + normal != total_tubelets:
        raise RuntimeError("temporal alignment fallback ledger does not reconcile")
    if refreshed + retained != total_tubelets * 64:
        raise RuntimeError("temporal refresh/retain ledger does not reconcile")
    return {
        "mode": expected_mode,
        "requested_tokens_per_window": requested,
        "refresh_tokens_per_window_by_batch": list(refresh_by_batch),
        "fallback_tubelets": fallback,
        "normal_tubelets": normal,
    }


def _assert_no_temporal_memory_in_checkpoint(checkpoint):
    forbidden = (
        "apm_memory",
        "previous_tubelet_memory",
        "temporal_memory_cache",
    )
    for state_name in ("state_dict", "state_dict_ema"):
        state = checkpoint.get(state_name)
        if not isinstance(state, dict):
            raise RuntimeError(f"temporal preflight checkpoint lacks {state_name}")
        bad = [key for key in state if any(fragment in key for fragment in forbidden)]
        if bad:
            raise RuntimeError(
                f"temporal preflight serialized live memory in {state_name}: {bad[:3]}"
            )
    training_state = checkpoint.get("training_state")
    if not isinstance(training_state, dict):
        raise RuntimeError("temporal preflight checkpoint lacks training_state")
    bad = [key for key in training_state if any(fragment in key for fragment in forbidden)]
    if bad:
        raise RuntimeError(f"temporal preflight serialized live memory metadata: {bad}")


def _load_zoomtoken_checkpoint_state(
    checkpoint,
    model,
    model_ema,
    optimizer,
    scheduler,
    scaler,
    args,
    cfg,
    train_loader,
    recovery_contract,
):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if model_ema is not None:
        if "state_dict_ema" not in checkpoint:
            raise ValueError("resume checkpoint lacks the required EMA state")
        model_ema.module.load_state_dict(checkpoint["state_dict_ema"])
    next_successful_update_index = None
    if recovery_contract is not None:
        if scaler is not None:
            if "scaler" not in checkpoint:
                raise ValueError("ZoomToken recovery checkpoint lacks scaler state")
            scaler.load_state_dict(checkpoint["scaler"])
        next_successful_update_index = _restore_zoomtoken_training_state(
            checkpoint,
            args,
            cfg,
            train_loader,
            recovery_contract,
            model=model,
        )
    return next_successful_update_index


def _zoomtoken_recovery_contract(cfg):
    recovery = cfg.get("zoomtoken_recovery", None)
    if recovery is None:
        return None

    p1_config = cfg.get("zoomtoken_p1_config", None)
    if p1_config is None:
        raise ValueError("ZoomToken recovery requires zoomtoken_p1_config")
    arm_surface = p1_config.get("arm_surface", None)
    if arm_surface not in ZOOMTOKEN_RECOVERY_ARMS:
        raise ValueError(
            "ZoomToken recovery is restricted to the frozen route surfaces"
        )

    contract = dict(recovery)
    expected = {
        "schema_version": ZOOMTOKEN_RECOVERY_SCHEMA,
        "enabled": True,
        "interval_epochs": 5,
        "keep_latest": 3,
        "save_final": True,
        "full_state": True,
        "same_cell_only": True,
        "unsealed_only": True,
        "seal_marker": ".zoomtoken_cell_sealed",
    }
    for key, expected_value in expected.items():
        if contract.get(key) != expected_value:
            raise ValueError(
                f"invalid ZoomToken recovery contract field {key!r}: "
                f"expected {expected_value!r}, got {contract.get(key)!r}"
            )
    if cfg.workflow.get("checkpoint_interval", None) != 5:
        raise ValueError("ZoomToken recovery must run every five epochs")
    if cfg.workflow.get("checkpoint_policy", None) != "recovery_latest3_plus_final":
        raise ValueError("ZoomToken checkpoint policy is not recovery/latest3/final")
    successful_update_workflow = {
        "require_successful_update_hook": True,
        "schedule_and_ema_on_success_only": True,
        "max_amp_retries_per_batch": 8,
        "fail_on_skipped_update": True,
    }
    for key, expected_value in successful_update_workflow.items():
        if cfg.workflow.get(key, None) != expected_value:
            raise ValueError(
                f"invalid ZoomToken successful-update workflow field {key!r}: "
                f"expected {expected_value!r}, got {cfg.workflow.get(key, None)!r}"
            )
    if not cfg.solver.get("amp", False) or not cfg.solver.get("ema", False):
        raise ValueError("ZoomToken full-state recovery requires the frozen AMP/EMA recipe")
    source_commit = p1_config.get("source_commit", None)
    if arm_surface in ZOOMTOKEN_CANONICAL_SOURCE_ARMS and (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        raise ValueError("ZoomToken route recovery requires a canonical source commit")
    work_dir_parts = os.path.normpath(cfg.work_dir).replace("\\", "/").split("/")
    if p1_config.get("runner_binding_required", False) and any(
        part.endswith("_unbound") for part in work_dir_parts
    ):
        raise ValueError(
            "ZoomToken work_dir must be explicitly bound by --work-dir "
            "or the stage runner"
        )
    contract["arm_surface"] = arm_surface
    contract["seed"] = int(p1_config["seed"])
    contract["source_commit"] = source_commit
    contract["max_amp_retries_per_batch"] = successful_update_workflow[
        "max_amp_retries_per_batch"
    ]
    return contract


def _validate_zoomtoken_resume_path(resume_path, cfg, recovery_contract):
    resume_path = os.path.realpath(os.path.expanduser(resume_path))
    checkpoint_dir = os.path.realpath(os.path.join(cfg.work_dir, "checkpoint"))
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"ZoomToken recovery checkpoint not found: {resume_path}")
    if os.path.normcase(os.path.dirname(resume_path)) != os.path.normcase(
        checkpoint_dir
    ):
        raise ValueError("ZoomToken resume checkpoint must belong to the same cell")
    if re.fullmatch(r"recovery_epoch_\d+\.pth", os.path.basename(resume_path)) is None:
        raise ValueError("ZoomToken resume requires a recovery_epoch_<N>.pth artifact")
    seal_path = os.path.join(cfg.work_dir, recovery_contract["seal_marker"])
    if os.path.exists(seal_path):
        raise RuntimeError("sealed ZoomToken cells are not resumable")
    return resume_path


def _capture_zoomtoken_training_state(
    args,
    cfg,
    train_loader,
    epoch,
    recovery_contract,
    next_successful_update_index,
    model=None,
):
    if recovery_contract["arm_surface"] in ZOOMTOKEN_UPDATE_INDEX_ARMS:
        if (
            not isinstance(next_successful_update_index, int)
            or next_successful_update_index < 0
        ):
            raise ValueError(
                "ZoomToken route recovery requires a non-negative update index"
            )
    elif next_successful_update_index is not None:
        raise ValueError("ZoomToken DN recovery must not carry an update index")
    local_rng_state = {
        "rank": args.rank,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": bytes(torch.get_rng_state().cpu().tolist()),
        "torch_cuda_rng_state": bytes(
            torch.cuda.get_rng_state(args.local_rank).cpu().tolist()
        ),
    }
    rng_state_by_rank = [None for _ in range(args.world_size)]
    dist.all_gather_object(rng_state_by_rank, local_rng_state)
    for rank, rank_state in enumerate(rng_state_by_rank):
        if rank_state is None or rank_state.get("rank") != rank:
            raise RuntimeError("failed to collect complete per-rank recovery RNG state")

    training_state = {
        "schema_version": ZOOMTOKEN_RECOVERY_SCHEMA,
        "arm_surface": recovery_contract["arm_surface"],
        "seed": args.seed,
        "source_commit": recovery_contract.get("source_commit", None),
        "config_path": os.path.realpath(args.config),
        "work_dir": os.path.realpath(cfg.work_dir),
        "world_size": args.world_size,
        "completed_epoch": epoch,
        "next_epoch": epoch + 1,
        "sampler_epoch": int(getattr(train_loader.sampler, "epoch", epoch)),
        "batches_per_epoch": len(train_loader),
        "completed_batches": (epoch + 1) * len(train_loader),
        "next_successful_update_index": next_successful_update_index,
        "rng_state_by_rank": rng_state_by_rank,
    }
    if recovery_contract["arm_surface"] in ZOOMTOKEN_R3_ARMS:
        candidate_model = getattr(model, "module", model)
        backbone = getattr(candidate_model, "backbone", None)
        exporter = getattr(backbone, "export_r3_recovery_state", None)
        if not callable(exporter):
            raise ValueError("R3 recovery requires its backbone dual-state exporter")
        r3_dual_state = exporter()
        if not isinstance(r3_dual_state, dict):
            raise ValueError("R3 recovery exporter returned no dual state")
        if (
            r3_dual_state.get("last_completed_update")
            != next_successful_update_index - 1
            or r3_dual_state.get("last_completed_epoch") != epoch
        ):
            raise ValueError("R3 dual state disagrees with epoch/update identity")
        training_state["r3_dual_state"] = r3_dual_state
    return training_state


def _restore_zoomtoken_training_state(
    checkpoint, args, cfg, train_loader, recovery_contract, model=None
):
    if checkpoint.get("checkpoint_role") != "recovery":
        raise ValueError("ZoomToken resume accepts only a recovery checkpoint")
    training_state = checkpoint.get("training_state", None)
    if not isinstance(training_state, dict):
        raise ValueError("ZoomToken recovery checkpoint lacks full training_state")

    expected = {
        "schema_version": ZOOMTOKEN_RECOVERY_SCHEMA,
        "arm_surface": recovery_contract["arm_surface"],
        "seed": args.seed,
        "source_commit": recovery_contract.get("source_commit", None),
        "config_path": os.path.realpath(args.config),
        "work_dir": os.path.realpath(cfg.work_dir),
        "world_size": args.world_size,
        "completed_epoch": checkpoint["epoch"],
        "next_epoch": checkpoint["epoch"] + 1,
        "batches_per_epoch": len(train_loader),
        "completed_batches": (checkpoint["epoch"] + 1) * len(train_loader),
    }
    for key, expected_value in expected.items():
        if training_state.get(key) != expected_value:
            raise ValueError(
                f"ZoomToken recovery state mismatch for {key!r}: "
                f"expected {expected_value!r}, got {training_state.get(key)!r}"
            )
    if training_state.get("sampler_epoch") != checkpoint["epoch"]:
        raise ValueError("ZoomToken recovery sampler epoch is inconsistent")
    next_successful_update_index = training_state.get(
        "next_successful_update_index", None
    )
    if recovery_contract["arm_surface"] in ZOOMTOKEN_UPDATE_INDEX_ARMS:
        if (
            not isinstance(next_successful_update_index, int)
            or next_successful_update_index < 0
        ):
            raise ValueError("ZoomToken route recovery lacks a valid update index")
    elif next_successful_update_index is not None:
        raise ValueError("ZoomToken DN recovery must not carry an update index")

    rng_state_by_rank = training_state.get("rng_state_by_rank", None)
    if not isinstance(rng_state_by_rank, list) or len(rng_state_by_rank) != args.world_size:
        raise ValueError("ZoomToken recovery lacks complete per-rank RNG state")
    local_rng_state = rng_state_by_rank[args.rank]
    if not isinstance(local_rng_state, dict) or local_rng_state.get("rank") != args.rank:
        raise ValueError("ZoomToken recovery RNG state is bound to another rank")
    for key in (
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
    ):
        if key not in local_rng_state:
            raise ValueError(f"ZoomToken recovery RNG state lacks {key}")

    train_loader.sampler.set_epoch(training_state["next_epoch"])
    random.setstate(local_rng_state["python_rng_state"])
    np.random.set_state(local_rng_state["numpy_rng_state"])
    torch.set_rng_state(
        torch.tensor(
            list(local_rng_state["torch_cpu_rng_state"]),
            dtype=torch.uint8,
            device="cpu",
        )
    )
    torch.cuda.set_rng_state(
        torch.tensor(
            list(local_rng_state["torch_cuda_rng_state"]),
            dtype=torch.uint8,
            device="cpu",
        ),
        device=args.local_rank,
    )
    if recovery_contract["arm_surface"] in ZOOMTOKEN_R3_ARMS:
        r3_dual_state = training_state.get("r3_dual_state", None)
        if not isinstance(r3_dual_state, dict):
            raise ValueError("R3 recovery lacks its dual state")
        if (
            r3_dual_state.get("last_completed_update")
            != next_successful_update_index - 1
            or r3_dual_state.get("last_completed_epoch") != checkpoint["epoch"]
        ):
            raise ValueError("R3 recovery dual identity is inconsistent")
        candidate_model = getattr(model, "module", model)
        backbone = getattr(candidate_model, "backbone", None)
        restorer = getattr(backbone, "restore_r3_recovery_state", None)
        if not callable(restorer):
            raise ValueError("R3 recovery requires its backbone dual-state restorer")
        restorer(r3_dual_state)
    elif "r3_dual_state" in training_state:
        raise ValueError("non-R3 recovery cannot carry R3 dual state")
    return next_successful_update_index


def _save_zoomtoken_checkpoint(
    model,
    model_ema,
    optimizer,
    scheduler,
    scaler,
    epoch,
    args,
    cfg,
    train_loader,
    recovery_contract,
    next_successful_update_index,
    is_final,
):
    training_state = _capture_zoomtoken_training_state(
        args,
        cfg,
        train_loader,
        epoch,
        recovery_contract,
        next_successful_update_index,
        model=model,
    )
    if args.rank == 0:
        checkpoint_role = "final" if is_final else "recovery"
        recovery_keep_latest = None if is_final else recovery_contract["keep_latest"]
        save_checkpoint(
            model,
            model_ema,
            optimizer,
            scheduler,
            epoch,
            work_dir=cfg.work_dir,
            scaler=scaler,
            training_state=training_state,
            checkpoint_role=checkpoint_role,
            recovery_keep_latest=recovery_keep_latest,
        )
    dist.barrier()


def _draw_zoomtoken_rng_probe(args):
    return {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch_cpu": torch.rand(4),
        "torch_cuda": torch.rand(4, device=f"cuda:{args.local_rank}").cpu(),
    }


def _run_zoomtoken_temporal_preflight(
    args,
    cfg,
    train_loader,
    model,
    model_ema,
    optimizer,
    scheduler,
    scaler,
    recovery_contract,
    logger,
    next_successful_update_index,
):
    if recovery_contract is None or recovery_contract["arm_surface"] not in (
        ZOOMTOKEN_TEMPORAL_PREFLIGHT_ARMS
    ):
        raise ValueError("temporal mechanical preflight requires APM32/CUR32 recovery")
    if args.resume is not None:
        raise ValueError("temporal mechanical preflight always starts from the frozen pretrain")
    if model_ema is None or scaler is None:
        raise ValueError("temporal mechanical preflight requires the frozen EMA/AMP recipe")

    single_batch_loader = _SingleBatchLoader(train_loader)
    single_batch_loader.sampler.set_epoch(0)
    finite_loss_observations = []

    def _observe_finite_loss(_module, _inputs, output):
        if not isinstance(output, dict) or "cost" not in output:
            raise RuntimeError("temporal preflight detector returned no training cost")
        cost = output["cost"]
        if not isinstance(cost, torch.Tensor) or not bool(torch.isfinite(cost).all().item()):
            raise RuntimeError("temporal preflight detector loss is non-finite")
        finite_loss_observations.append(True)

    hook = model.module.register_forward_hook(_observe_finite_loss)
    try:
        next_successful_update_index = train_one_epoch(
            single_batch_loader,
            model,
            optimizer,
            scheduler,
            0,
            logger,
            model_ema=model_ema,
            clip_grad_l2norm=cfg.solver.clip_grad_norm,
            logging_interval=1,
            scaler=scaler,
            successful_update_index=next_successful_update_index,
            max_amp_retries_per_batch=recovery_contract[
                "max_amp_retries_per_batch"
            ],
        )
    finally:
        hook.remove()
    if not finite_loss_observations:
        raise RuntimeError("temporal preflight executed no detector forward")
    execution_receipt = _zoomtoken_temporal_preflight_summary(
        model,
        recovery_contract,
    )

    _save_zoomtoken_checkpoint(
        model,
        model_ema,
        optimizer,
        scheduler,
        scaler,
        0,
        args,
        cfg,
        single_batch_loader,
        recovery_contract,
        next_successful_update_index,
        is_final=False,
    )
    checkpoint_path = os.path.join(
        cfg.work_dir,
        "checkpoint",
        "recovery_epoch_0.pth",
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=f"cuda:{args.local_rank}",
    )
    required_fields = {
        "state_dict",
        "state_dict_ema",
        "optimizer",
        "scheduler",
        "scaler",
        "training_state",
    }
    if not required_fields.issubset(checkpoint):
        missing = sorted(required_fields - set(checkpoint))
        raise RuntimeError(f"temporal preflight recovery lacks full state: {missing}")
    _assert_no_temporal_memory_in_checkpoint(checkpoint)

    with torch.no_grad():
        next(model.parameters()).add_(1.0)
        next(model_ema.module.parameters()).add_(1.0)
    optimizer.param_groups[0]["lr"] = -1.0
    if hasattr(scheduler, "last_epoch"):
        scheduler.last_epoch += 7
    scaler_state = scaler.state_dict()
    if "scale" in scaler_state:
        scaler_state["scale"] = float(scaler_state["scale"]) * 2.0
        scaler.load_state_dict(scaler_state)

    restored_update_index = _load_zoomtoken_checkpoint_state(
        checkpoint,
        model,
        model_ema,
        optimizer,
        scheduler,
        scaler,
        args,
        cfg,
        single_batch_loader,
        recovery_contract,
    )
    if restored_update_index != next_successful_update_index:
        raise RuntimeError("temporal preflight restored the wrong update index")
    _assert_nested_state_equal(
        model.state_dict(),
        checkpoint["state_dict"],
        "model",
    )
    _assert_nested_state_equal(
        model_ema.module.state_dict(),
        checkpoint["state_dict_ema"],
        "ema",
    )
    _assert_nested_state_equal(
        optimizer.state_dict(),
        checkpoint["optimizer"],
        "optimizer",
    )
    _assert_nested_state_equal(
        scheduler.state_dict(),
        checkpoint["scheduler"],
        "scheduler",
    )
    _assert_nested_state_equal(
        scaler.state_dict(),
        checkpoint["scaler"],
        "scaler",
    )

    first_rng_probe = _draw_zoomtoken_rng_probe(args)
    second_update_index = _restore_zoomtoken_training_state(
        checkpoint,
        args,
        cfg,
        single_batch_loader,
        recovery_contract,
        model=model,
    )
    second_rng_probe = _draw_zoomtoken_rng_probe(args)
    if second_update_index != restored_update_index:
        raise RuntimeError("temporal preflight recovery update identity is unstable")
    _assert_nested_state_equal(second_rng_probe, first_rng_probe, "rng_continuation")

    receipt = {
        "schema_version": "zoomtoken_apm32_ctx64_mechanical_preflight_v001",
        "status": "PASS_MECHANICAL_ONLY",
        "arm_surface": recovery_contract["arm_surface"],
        "source_commit": recovery_contract["source_commit"],
        "config_path": os.path.realpath(args.config),
        "work_dir": os.path.realpath(cfg.work_dir),
        "seed": args.seed,
        "world_size": args.world_size,
        "single_batch_forward_backward": True,
        "finite_loss": True,
        "metric_evaluation_performed": False,
        "execution_ledger": execution_receipt,
        "recovery_checkpoint": os.path.realpath(checkpoint_path),
        "full_state_restored": True,
        "rng_continuation_restored": True,
        "temporal_memory_serialized": False,
        "accuracy_claim_allowed": False,
        "efficiency_claim_allowed": False,
    }
    if args.rank == 0:
        control_dir = os.path.join(cfg.work_dir, "control")
        os.makedirs(control_dir, exist_ok=True)
        receipt_path = os.path.join(control_dir, "temporal_preflight.json")
        temporary_path = receipt_path + ".tmp"
        with open(temporary_path, "x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, receipt_path)
        logger.info("Temporal mechanical preflight PASS: %s", receipt_path)
    dist.barrier()
    return receipt


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="explicit bounded work directory for a training cell",
    )
    parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument(
        "--zoomtoken-temporal-preflight-only",
        action="store_true",
        help="run one result-blind APM/CUR train batch plus full-state recovery fixture",
    )
    parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
    parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # set random seed, create work_dir, and save config
    set_seed(args.seed, args.disable_deterministic)
    cfg = update_workdir(cfg, args.id, args.world_size)
    recovery_contract = _zoomtoken_recovery_contract(cfg)
    if recovery_contract is not None and args.seed != recovery_contract["seed"]:
        raise ValueError(
            f"ZoomToken {recovery_contract['arm_surface']} requires seed "
            f"{recovery_contract['seed']}, got {args.seed}"
        )
    if args.zoomtoken_temporal_preflight_only:
        if recovery_contract is None or recovery_contract["arm_surface"] not in (
            ZOOMTOKEN_TEMPORAL_PREFLIGHT_ARMS
        ):
            raise ValueError(
                "temporal mechanical preflight accepts only frozen APM32/CUR32"
            )
        if args.resume is not None:
            raise ValueError("temporal mechanical preflight forbids resume input")
    next_successful_update_index = (
        0
        if recovery_contract is not None
        and recovery_contract["arm_surface"] in ZOOMTOKEN_UPDATE_INDEX_ARMS
        else None
    )
    if args.resume is not None and recovery_contract is not None:
        args.resume = _validate_zoomtoken_resume_path(
            args.resume, cfg, recovery_contract
        )
    if args.rank == 0:
        create_folder(cfg.work_dir)
        save_config(args.config, cfg.work_dir)

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    train_loader = build_dataloader(
        train_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=True,
        drop_last=True,
        **cfg.solver.train,
    )

    if args.zoomtoken_temporal_preflight_only:
        val_loader = None
        test_loader = None
    else:
        val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=logger))
        val_loader = build_dataloader(
            val_dataset,
            rank=args.rank,
            world_size=args.world_size,
            shuffle=False,
            drop_last=False,
            **cfg.solver.val,
        )

        test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
        test_loader = build_dataloader(
            test_dataset,
            rank=args.rank,
            world_size=args.world_size,
            shuffle=False,
            drop_last=False,
            **cfg.solver.test,
        )

    # build model
    model = build_detector(cfg.model)

    # DDP
    use_static_graph = getattr(cfg.solver, "static_graph", False)
    model = model.to(args.local_rank)
    model = DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False if use_static_graph else True,
        static_graph=use_static_graph,  # default is False, should be true when use activation checkpointing in E2E
    )
    logger.info(f"Using DDP with total {args.world_size} GPUS...")

    # FP16 compression
    use_fp16_compress = getattr(cfg.solver, "fp16_compress", False)
    if use_fp16_compress:
        logger.info("Using FP16 compression ...")
        model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    # Model EMA
    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        logger.info("Using Model EMA...")
        model_ema = ModelEma(model)
    else:
        model_ema = None

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")
        scaler = GradScaler()
    else:
        scaler = None

    # build optimizer and scheduler
    optimizer = build_optimizer(cfg.optimizer, model, logger)
    scheduler, max_epoch = build_scheduler(cfg.scheduler, optimizer, len(train_loader))

    # override the max_epoch
    max_epoch = cfg.workflow.get("end_epoch", max_epoch)

    # resume: reset epoch, load checkpoint / best rmse
    if args.resume is not None:
        logger.info("Resume training from: {}".format(args.resume))
        device = f"cuda:{args.local_rank}"
        checkpoint = torch.load(args.resume, map_location=device)
        resume_epoch = checkpoint["epoch"]
        logger.info("Resume epoch is {}".format(resume_epoch))
        restored_update_index = _load_zoomtoken_checkpoint_state(
            checkpoint,
            model,
            model_ema,
            optimizer,
            scheduler,
            scaler,
            args,
            cfg,
            train_loader,
            recovery_contract,
        )
        if recovery_contract is not None:
            next_successful_update_index = restored_update_index

        del checkpoint  # save memory if the model is very large such as ViT-g
        torch.cuda.empty_cache()
    else:
        resume_epoch = -1

    if args.zoomtoken_temporal_preflight_only:
        _run_zoomtoken_temporal_preflight(
            args,
            cfg,
            train_loader,
            model,
            model_ema,
            optimizer,
            scheduler,
            scaler,
            recovery_contract,
            logger,
            next_successful_update_index,
        )
        logger.info("Temporal mechanical preflight over; no validation/evaluation run.\n")
        return

    # train the detector
    logger.info("Training Starts...\n")
    val_loss_best = 1e6
    val_start_epoch = cfg.workflow.get("val_start_epoch", 0)
    for epoch in range(resume_epoch + 1, max_epoch):
        train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        next_successful_update_index = train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            logger,
            model_ema=model_ema,
            clip_grad_l2norm=cfg.solver.clip_grad_norm,
            logging_interval=cfg.workflow.logging_interval,
            scaler=scaler,
            successful_update_index=next_successful_update_index,
            max_amp_retries_per_batch=(
                recovery_contract["max_amp_retries_per_batch"]
                if recovery_contract is not None
                else None
            ),
        )

        # save checkpoint
        if recovery_contract is not None:
            is_final = epoch == max_epoch - 1
            is_recovery_epoch = (
                (epoch + 1) % recovery_contract["interval_epochs"] == 0
            )
            if is_final or is_recovery_epoch:
                _save_zoomtoken_checkpoint(
                    model,
                    model_ema,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    args,
                    cfg,
                    train_loader,
                    recovery_contract,
                    next_successful_update_index,
                    is_final=is_final,
                )
        elif (epoch == max_epoch - 1) or (
            (epoch + 1) % cfg.workflow.checkpoint_interval == 0
        ):
            if args.rank == 0:
                save_checkpoint(
                    model,
                    model_ema,
                    optimizer,
                    scheduler,
                    epoch,
                    work_dir=cfg.work_dir,
                )

        # val for one epoch
        if epoch >= val_start_epoch:
            if (cfg.workflow.val_loss_interval > 0) and ((epoch + 1) % cfg.workflow.val_loss_interval == 0):
                val_loss = val_one_epoch(
                    val_loader,
                    model,
                    logger,
                    args.rank,
                    epoch,
                    model_ema=model_ema,
                    use_amp=use_amp,
                )

                # save the best checkpoint
                if val_loss < val_loss_best:
                    logger.info(f"New best epoch {epoch}")
                    val_loss_best = val_loss
                    if args.rank == 0:
                        save_best_checkpoint(model, model_ema, epoch, work_dir=cfg.work_dir)

        # eval for one epoch
        if epoch >= val_start_epoch:
            if (cfg.workflow.val_eval_interval > 0) and ((epoch + 1) % cfg.workflow.val_eval_interval == 0):
                eval_one_epoch(
                    test_loader,
                    model,
                    cfg,
                    logger,
                    args.rank,
                    model_ema=model_ema,
                    use_amp=use_amp,
                    world_size=args.world_size,
                    not_eval=args.not_eval,
                )
    logger.info("Training Over...\n")


if __name__ == "__main__":
    main()
