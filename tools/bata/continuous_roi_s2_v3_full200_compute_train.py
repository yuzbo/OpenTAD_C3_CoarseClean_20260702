from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np

from tools.bata.continuous_roi_s2_v3_full200_compute import (
    EXPECTED_EPOCHS,
    EXPECTED_TOTAL_UPDATES,
    EXPECTED_TRAINING_IDENTITIES,
    EXPECTED_UPDATES_PER_EPOCH,
    EXPECTED_WORLD_SIZE,
    atomic_publish_json,
    canonical_sha256,
    require_clean_commit,
    sha256_file,
)
from tools.bata.zoomtoken_full200_matrix_spec import (
    binding_from_config,
    get_matrix_spec,
    validate_matrix_cell,
)


MATRIX_SPEC = get_matrix_spec()
PROTOCOL_ID = MATRIX_SPEC.protocol_id


CERTIFIED_RECOVERY_UPDATES = tuple(range(0, EXPECTED_TOTAL_UPDATES + 1, 500))
RECOVERY_SCHEMA = "s2_v3_full200_epoch_boundary_recovery_v1"
REQUIRED_IDENTITY_HASHES = {
    "code_sha256",
    "protocol_sha256",
    "config_sha256",
    "annotation_sha256",
    "class_map_sha256",
    "media_manifest_sha256",
    "pretrained_sha256",
}
LOCAL_RNG_STATE_KEYS = {
    "rank",
    "python_rng_state",
    "numpy_rng_state",
    "torch_cpu_rng_state",
    "torch_cuda_rng_state",
    "dataloader_generator_state",
    "augmentation_rng_streams",
}
REQUIRED_PAYLOAD_KEYS = {
    "schema_version",
    "protocol_id",
    "model_state_dict",
    "ema_state_dict",
    "ema_update_counter",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "grad_scaler_state_dict",
    "next_epoch_index",
    "accepted_successful_updates",
    "successful_updates_per_completed_epoch",
    "next_epoch_identity_order",
    "distributed_sampler_state",
    "rank_rng_states",
    "world_size",
    "local_batch_size",
    "global_batch_size",
    "identity_hashes",
    "completed_sample_order_trace_sha256",
    "discarded_preemption_steps",
    "zero_pending_gradient",
}


def capture_rng_state(
    *,
    rank: int,
    dataloader_generator: Any,
    augmentation_rng_streams: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    return {
        "rank": int(rank),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state()
        if torch.cuda.is_available()
        else None,
        "dataloader_generator_state": dataloader_generator.get_state(),
        "augmentation_rng_streams": dict(augmentation_rng_streams),
    }


def restore_rng_state(
    payload: Mapping[str, Any],
    *,
    rank: int,
    dataloader_generator: Any,
    augmentation_rng_stream_restorers: Mapping[str, Any],
) -> None:
    import torch

    states = payload["rank_rng_states"]
    if not isinstance(states, list) or not 0 <= int(rank) < len(states):
        raise ValueError("recovery payload has no RNG state for this rank")
    local = states[int(rank)]
    if int(local["rank"]) != int(rank):
        raise ValueError("rank RNG state order changed")
    random.setstate(local["python_rng_state"])
    np.random.set_state(local["numpy_rng_state"])
    torch.set_rng_state(local["torch_cpu_rng_state"])
    cuda_state = local["torch_cuda_rng_state"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG state cannot be restored without CUDA")
        torch.cuda.set_rng_state(cuda_state)
    dataloader_generator.set_state(local["dataloader_generator_state"])
    saved_streams = local["augmentation_rng_streams"]
    if set(saved_streams) != set(augmentation_rng_stream_restorers):
        raise ValueError("augmentation/view RNG stream identity changed")
    for name, restore in augmentation_rng_stream_restorers.items():
        restore(saved_streams[name])


def build_recovery_payload(
    *,
    model: Any,
    ema_model: Any,
    ema_update_counter: int,
    optimizer: Any,
    scheduler: Any,
    grad_scaler: Any,
    next_epoch_index: int,
    accepted_successful_updates: int,
    successful_updates_per_completed_epoch: Sequence[int],
    next_epoch_identity_order: Sequence[str],
    distributed_sampler_state: Mapping[str, Any],
    rank_rng_states: Sequence[Mapping[str, Any]],
    world_size: int,
    local_batch_size: int,
    global_batch_size: int,
    identity_hashes: Mapping[str, str],
    completed_sample_order_trace_sha256: str,
    discarded_preemption_steps: int,
    zero_pending_gradient: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RECOVERY_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "ema_update_counter": int(ema_update_counter),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "grad_scaler_state_dict": grad_scaler.state_dict(),
        "next_epoch_index": int(next_epoch_index),
        "accepted_successful_updates": int(accepted_successful_updates),
        "successful_updates_per_completed_epoch": list(
            map(int, successful_updates_per_completed_epoch)
        ),
        "next_epoch_identity_order": list(map(str, next_epoch_identity_order)),
        "distributed_sampler_state": dict(distributed_sampler_state),
        "world_size": int(world_size),
        "local_batch_size": int(local_batch_size),
        "global_batch_size": int(global_batch_size),
        "identity_hashes": dict(identity_hashes),
        "completed_sample_order_trace_sha256": str(
            completed_sample_order_trace_sha256
        ),
        "discarded_preemption_steps": int(discarded_preemption_steps),
        "zero_pending_gradient": bool(zero_pending_gradient),
        "rank_rng_states": [dict(row) for row in rank_rng_states],
    }
    validate_recovery_payload(payload)
    return payload


def validate_recovery_payload(
    payload: Mapping[str, Any],
    *,
    expected_identity_hashes: Mapping[str, str] | None = None,
) -> None:
    missing = sorted(REQUIRED_PAYLOAD_KEYS - set(payload))
    extra = sorted(set(payload) - REQUIRED_PAYLOAD_KEYS)
    if missing or extra:
        raise ValueError(f"recovery payload fields changed: missing={missing} extra={extra}")
    if payload["schema_version"] != RECOVERY_SCHEMA or payload["protocol_id"] != PROTOCOL_ID:
        raise ValueError("recovery schema or protocol changed")
    updates = int(payload["accepted_successful_updates"])
    if updates not in CERTIFIED_RECOVERY_UPDATES:
        raise ValueError("checkpoint is not at a certified 500-update boundary")
    next_epoch = int(payload["next_epoch_index"])
    if next_epoch != updates // EXPECTED_UPDATES_PER_EPOCH or not 0 <= next_epoch <= EXPECTED_EPOCHS:
        raise ValueError("checkpoint epoch/update identity is inconsistent")
    per_epoch = list(map(int, payload["successful_updates_per_completed_epoch"]))
    if per_epoch != [EXPECTED_UPDATES_PER_EPOCH] * next_epoch:
        raise ValueError("completed epoch successful-update receipt is incomplete")
    identity_order = list(map(str, payload["next_epoch_identity_order"]))
    if len(identity_order) != EXPECTED_TRAINING_IDENTITIES or len(set(identity_order)) != EXPECTED_TRAINING_IDENTITIES:
        raise ValueError("next epoch identity order is not the complete 200-video population")
    sampler = payload["distributed_sampler_state"]
    if (
        not isinstance(sampler, Mapping)
        or int(sampler.get("epoch", -1)) != next_epoch
        or int(sampler.get("cursor", -1)) != 0
        or int(sampler.get("world_size", -1)) != EXPECTED_WORLD_SIZE
        or len(sampler.get("rank_shards", [])) != EXPECTED_WORLD_SIZE
    ):
        raise ValueError("distributed sampler state is not an epoch-boundary 2-rank state")
    if (
        int(payload["world_size"]) != EXPECTED_WORLD_SIZE
        or int(payload["local_batch_size"]) != 1
        or int(payload["global_batch_size"]) != 2
    ):
        raise ValueError("recovery world-size or batch identity changed")
    rank_rng_states = payload["rank_rng_states"]
    if (
        not isinstance(rank_rng_states, list)
        or len(rank_rng_states) != EXPECTED_WORLD_SIZE
        or [int(row.get("rank", -1)) for row in rank_rng_states]
        != list(range(EXPECTED_WORLD_SIZE))
        or any(set(row) != LOCAL_RNG_STATE_KEYS for row in rank_rng_states)
    ):
        raise ValueError("recovery does not contain one complete RNG state per rank")
    if not bool(payload["zero_pending_gradient"]):
        raise ValueError("checkpoint has a pending gradient")
    if int(payload["ema_update_counter"]) != updates:
        raise ValueError("EMA update counter differs from successful updates")
    hashes = payload["identity_hashes"]
    if not isinstance(hashes, Mapping) or set(hashes) != REQUIRED_IDENTITY_HASHES:
        raise ValueError("recovery identity hash surface is incomplete")
    if any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
        raise ValueError("recovery identity hashes must be SHA256 hex strings")
    if expected_identity_hashes is not None and dict(hashes) != dict(expected_identity_hashes):
        raise ValueError("recovery identity hashes differ from formal execution")
    trace = str(payload["completed_sample_order_trace_sha256"])
    if len(trace) != 64:
        raise ValueError("sample-order trace digest is not SHA256")


def validate_epoch_sampler_state(
    state: Mapping[str, Any], *, expected_identities: Sequence[str]
) -> tuple[str, ...]:
    identities = tuple(map(str, expected_identities))
    shards = state.get("rank_shards")
    if (
        int(state.get("world_size", -1)) != EXPECTED_WORLD_SIZE
        or int(state.get("cursor", -1)) != 0
        or not isinstance(shards, list)
        or len(shards) != EXPECTED_WORLD_SIZE
        or any(len(shard) != EXPECTED_UPDATES_PER_EPOCH for shard in shards)
    ):
        raise ValueError("epoch sampler is not a complete 2-rank/100-step state")
    flattened_indices = [
        int(shards[rank][slot])
        for slot in range(EXPECTED_UPDATES_PER_EPOCH)
        for rank in range(EXPECTED_WORLD_SIZE)
    ]
    if sorted(flattened_indices) != list(range(EXPECTED_TRAINING_IDENTITIES)):
        raise ValueError("epoch sampler does not consume every training identity exactly once")
    order = tuple(identities[index] for index in flattened_indices)
    if len(identities) != EXPECTED_TRAINING_IDENTITIES or len(set(order)) != len(order):
        raise ValueError("epoch sampler identity population is incomplete or duplicated")
    return order


def build_epoch_sampler_state(dataset: Any, *, epoch: int) -> dict[str, Any]:
    import torch

    shards = []
    for rank in range(EXPECTED_WORLD_SIZE):
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=EXPECTED_WORLD_SIZE,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        sampler.set_epoch(int(epoch))
        shards.append(list(map(int, sampler)))
    return {
        "epoch": int(epoch),
        "cursor": 0,
        "world_size": EXPECTED_WORLD_SIZE,
        "rank_shards": shards,
    }


def validate_full_data_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    checked = dict(payload)
    digest = checked.pop("manifest_sha256", None)
    if not isinstance(digest, str) or canonical_sha256(checked) != digest:
        raise ValueError("full-data manifest self-hash mismatch")
    training = payload.get("training", {})
    evaluation = payload.get("evaluation", {})
    identities = list(map(str, training.get("identity_order", ())))
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or int(training.get("identity_count", -1)) != EXPECTED_TRAINING_IDENTITIES
        or len(identities) != EXPECTED_TRAINING_IDENTITIES
        or len(set(identities)) != EXPECTED_TRAINING_IDENTITIES
        or int(evaluation.get("video_count", -1)) != 211
        or int(evaluation.get("ordered_window_count", -1)) != 792
    ):
        raise ValueError("full-data manifest population differs from the frozen protocol")
    annotation = Path(training["training_only_annotation"])
    if not annotation.is_file() or sha256_file(annotation) != training[
        "training_only_annotation_sha256"
    ]:
        raise ValueError("training-only annotation is missing or changed")
    return payload


def save_atomic_recovery_checkpoint(
    path: str | Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    import torch

    validate_recovery_payload(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "schema_version": "s2_v3_recovery_checkpoint_receipt_v1",
        "protocol_id": PROTOCOL_ID,
        "checkpoint_path": target.resolve().as_posix(),
        "checkpoint_sha256": sha256_file(target),
        "accepted_successful_updates": int(payload["accepted_successful_updates"]),
        "next_epoch_index": int(payload["next_epoch_index"]),
        "payload_fields_sha256": hashlib.sha256(
            "\n".join(sorted(payload)).encode("utf-8")
        ).hexdigest(),
        "atomic_complete": True,
    }
    atomic_publish_json(target.with_suffix(target.suffix + ".receipt.json"), receipt)
    return receipt


def load_recovery_checkpoint(
    path: str | Path,
    *,
    expected_identity_hashes: Mapping[str, str],
) -> dict[str, Any]:
    import torch

    path = Path(path).resolve()
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    if not receipt_path.is_file():
        raise ValueError("recovery checkpoint has no atomic-complete receipt")
    receipt = __import__("json").loads(receipt_path.read_text(encoding="utf-8"))
    if not receipt.get("atomic_complete") or receipt.get("checkpoint_sha256") != sha256_file(path):
        raise ValueError("recovery checkpoint receipt or hash is invalid")
    payload = torch.load(path, map_location="cpu")
    validate_recovery_payload(payload, expected_identity_hashes=expected_identity_hashes)
    return payload


def _atomic_torch_publish(path: str | Path, payload: Mapping[str, Any]) -> None:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _load_identity_hashes(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(payload) != REQUIRED_IDENTITY_HASHES or any(
        not isinstance(value, str) or len(value) != 64 for value in payload.values()
    ):
        raise ValueError("formal identity hash file is incomplete")
    return {str(key): str(value) for key, value in payload.items()}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recovery-safe full-200 task-local 2-GPU training driver"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--matrix-kind",
        choices=("s2_v3", "d2s", "patad"),
        default=MATRIX_SPEC.key,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--identity-hashes", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.matrix_kind != MATRIX_SPEC.key:
        raise ValueError(
            "--matrix-kind must match ZOOMTOKEN_MATRIX_KIND before module import"
        )
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("formal training requires a Slurm allocation")
    import torch
    import torch.distributed as dist
    from mmengine.config import Config
    from torch.cuda.amp import GradScaler
    from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
    from torch.nn.parallel import DistributedDataParallel

    from opentad.cores import build_optimizer, build_scheduler, train_one_epoch
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from opentad.utils import ModelEma, create_folder, save_config, set_seed, setup_logger

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    if world_size != EXPECTED_WORLD_SIZE or torch.cuda.device_count() < EXPECTED_WORLD_SIZE:
        raise RuntimeError("formal training requires exactly the frozen 2-GPU world")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    try:
        cfg = Config.fromfile(args.config)
        binding = binding_from_config(cfg, MATRIX_SPEC)
        arm = str(binding.arm)
        if arm not in MATRIX_SPEC.arms:
            raise ValueError("config arm is outside the selected full-200 matrix")
        validate_matrix_cell(
            args.config, arm=arm, seed=args.seed, spec=MATRIX_SPEC
        )
        require_clean_commit(args.expected_commit, Path(__file__).resolve().parents[2])
        manifest = validate_full_data_manifest(args.manifest)
        identity_hashes = _load_identity_hashes(args.identity_hashes)
        if args.seed != int(binding.seed):
            raise ValueError("CLI seed differs from the frozen config")
        cfg.dataset.train.ann_file = manifest["training"]["training_only_annotation"]
        cfg.dataset.train.class_map = manifest["class_map"]["path"]
        cfg.dataset.train.data_path = manifest["media"]["root"]
        cfg.dataset.val.ann_file = manifest["evaluation"]["heldout_inference_annotation"]
        cfg.dataset.val.class_map = manifest["class_map"]["path"]
        cfg.dataset.val.data_path = manifest["media"]["root"]
        cfg.work_dir = str(args.work_dir.resolve())
        set_seed(args.seed, False, deterministic_warn_only=True)
        if args.resume is None and args.work_dir.exists():
            raise FileExistsError("fresh formal cell requires a new work directory")
        if rank == 0:
            create_folder(cfg.work_dir)
            save_config(str(args.config), cfg.work_dir)
        dist.barrier()
        logger = setup_logger(
            f"{MATRIX_SPEC.key}Full200Train",
            save_dir=cfg.work_dir,
            distributed_rank=rank,
        )

        train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
        dataset_identities = tuple(str(row[0]) for row in train_dataset.data_list)
        expected_identities = tuple(map(str, manifest["training"]["identity_order"]))
        if (
            len(dataset_identities) != EXPECTED_TRAINING_IDENTITIES
            or len(set(dataset_identities)) != EXPECTED_TRAINING_IDENTITIES
            or set(dataset_identities) != set(expected_identities)
        ):
            raise ValueError("runtime training loader is not the complete 200-video population")
        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(args.seed + rank)
        train_loader = build_dataloader(
            train_dataset,
            rank=rank,
            world_size=world_size,
            shuffle=True,
            drop_last=True,
            generator=dataloader_generator,
            **cfg.solver.train,
        )
        if len(train_loader) != EXPECTED_UPDATES_PER_EPOCH:
            raise RuntimeError("formal cell does not have exactly 100 rank-local batches")
        model = build_detector(cfg.model).to(local_rank)
        raw_model = model
        trainable_params = [p for p in raw_model.parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for p in trainable_params)
        backbone_module = getattr(raw_model, "backbone", None)
        projection_module = getattr(raw_model, "projection", None)
        
        runtime_identity_receipt = {
            "model_class": raw_model.__class__.__name__,
            "total_trainable_parameters": total_trainable,
            "backbone_class": backbone_module.__class__.__name__ if backbone_module else None,
            "projection_class": projection_module.__class__.__name__ if projection_module else None,
        }
        
        if hasattr(backbone_module, "fusion"):
            fusion_params = sum(p.numel() for p in backbone_module.fusion.parameters())
            runtime_identity_receipt["fusion_parameters"] = fusion_params
            runtime_identity_receipt["fusion_mode"] = getattr(backbone_module, "fusion_mode", None)
            if fusion_params != 0:
                raise RuntimeError(f"Audit failure: fusion has {fusion_params} parameters, expected 0")
            if getattr(backbone_module, "fusion_mode", None) != "fixed_mean":
                raise RuntimeError(f"Audit failure: fusion_mode is {backbone_module.fusion_mode}, expected fixed_mean")

        if rank == 0:
            logger.info(f"[RUNTIME_IDENTITY_AUDIT] {runtime_identity_receipt}")

        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            static_graph=True,
        )
        if not bool(cfg.solver.fp16_compress):
            raise ValueError("formal cell changed the frozen FP16 communication policy")
        model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
        if not bool(cfg.solver.ema) or not bool(cfg.solver.amp):
            raise ValueError("formal cell requires EMA and AMP")
        model_ema = ModelEma(model)
        scaler = GradScaler()
        optimizer = build_optimizer(copy.deepcopy(cfg.optimizer), model, logger)
        scheduler, scheduler_epochs = build_scheduler(
            copy.deepcopy(cfg.scheduler), optimizer, len(train_loader)
        )
        if int(cfg.workflow.end_epoch) != EXPECTED_EPOCHS or int(scheduler_epochs) < EXPECTED_EPOCHS:
            raise ValueError("formal scheduler does not cover all 60 epochs")

        successful_updates = 0
        start_epoch = 0
        ema_update_counter = 0
        per_epoch_updates: list[int] = []
        completed_orders: list[dict[str, Any]] = []
        update_audit = {
            "optimizer_attempts": 0,
            "amp_skipped_attempts": 0,
            "max_amp_retries_observed": 0,
            "consumed_batches": 0,
            "replay_attempts": 0,
            "scheduler_advances": 0,
            "ema_updates": 0,
        }
        if args.resume is not None:
            recovery = load_recovery_checkpoint(
                args.resume, expected_identity_hashes=identity_hashes
            )
            model.load_state_dict(recovery["model_state_dict"])
            model_ema.module.load_state_dict(recovery["ema_state_dict"])
            optimizer.load_state_dict(recovery["optimizer_state_dict"])
            scheduler.load_state_dict(recovery["scheduler_state_dict"])
            scaler.load_state_dict(recovery["grad_scaler_state_dict"])
            successful_updates = int(recovery["accepted_successful_updates"])
            start_epoch = int(recovery["next_epoch_index"])
            ema_update_counter = int(recovery["ema_update_counter"])
            per_epoch_updates = list(
                map(int, recovery["successful_updates_per_completed_epoch"])
            )
            for epoch in range(start_epoch):
                state = build_epoch_sampler_state(train_dataset, epoch=epoch)
                order = validate_epoch_sampler_state(
                    state, expected_identities=dataset_identities
                )
                completed_orders.append({"epoch": epoch, "identity_order": list(order)})
            if canonical_sha256(completed_orders) != recovery[
                "completed_sample_order_trace_sha256"
            ]:
                raise ValueError("reconstructed completed sample order differs from recovery")
            restore_rng_state(
                recovery,
                rank=rank,
                dataloader_generator=dataloader_generator,
                augmentation_rng_stream_restorers={},
            )
            torch.cuda.empty_cache()

        def publish_recovery(next_epoch: int) -> None:
            optimizer.zero_grad(set_to_none=True)
            next_sampler = build_epoch_sampler_state(train_dataset, epoch=next_epoch)
            next_order = validate_epoch_sampler_state(
                next_sampler, expected_identities=dataset_identities
            )
            local_rng = capture_rng_state(
                rank=rank,
                dataloader_generator=dataloader_generator,
                augmentation_rng_streams={},
            )
            rank_rng_states: list[Any] = [None] * world_size
            dist.all_gather_object(rank_rng_states, local_rng)
            if rank == 0:
                payload = build_recovery_payload(
                    model=model,
                    ema_model=model_ema.module,
                    ema_update_counter=ema_update_counter,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    grad_scaler=scaler,
                    next_epoch_index=next_epoch,
                    accepted_successful_updates=successful_updates,
                    successful_updates_per_completed_epoch=per_epoch_updates,
                    next_epoch_identity_order=next_order,
                    distributed_sampler_state=next_sampler,
                    rank_rng_states=rank_rng_states,
                    world_size=world_size,
                    local_batch_size=1,
                    global_batch_size=2,
                    identity_hashes=identity_hashes,
                    completed_sample_order_trace_sha256=canonical_sha256(
                        completed_orders
                    ),
                    discarded_preemption_steps=0,
                    zero_pending_gradient=all(
                        parameter.grad is None for parameter in model.parameters()
                    ),
                )
                save_atomic_recovery_checkpoint(
                    args.recovery_dir / f"update_{successful_updates:04d}.pth",
                    payload,
                )
            dist.barrier()

        if args.resume is None:
            publish_recovery(0)
        for epoch in range(start_epoch, EXPECTED_EPOCHS):
            sampler_state = build_epoch_sampler_state(train_dataset, epoch=epoch)
            order = validate_epoch_sampler_state(
                sampler_state, expected_identities=dataset_identities
            )
            train_loader.sampler.set_epoch(epoch)
            if list(map(int, train_loader.sampler)) != sampler_state["rank_shards"][rank]:
                raise RuntimeError("runtime sampler differs from the frozen epoch order")
            before = successful_updates
            successful_updates += train_one_epoch(
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
                max_train_iters=EXPECTED_UPDATES_PER_EPOCH,
                fail_on_skipped_update=True,
                max_amp_retries_per_batch=int(cfg.workflow.max_amp_retries_per_batch),
                update_audit=update_audit,
                successful_update_start=before,
                require_successful_update_hook=False,
                schedule_and_ema_on_success_only=True,
                capture_amp_rng_state=True,
                fail_on_nonfinite_loss=True,
            )
            delta = successful_updates - before
            if delta != EXPECTED_UPDATES_PER_EPOCH:
                raise RuntimeError(f"epoch {epoch} completed {delta} successful updates")
            ema_update_counter += delta
            per_epoch_updates.append(delta)
            completed_orders.append({"epoch": epoch, "identity_order": list(order)})
            if successful_updates % 500 == 0:
                publish_recovery(epoch + 1)

            # Metric-bearing validation is deliberately deferred until all nine
            # label-free prediction bundles have been sealed.

        if successful_updates != EXPECTED_TOTAL_UPDATES or ema_update_counter != EXPECTED_TOTAL_UPDATES:
            raise RuntimeError("formal training did not complete exactly 6000 successful updates")
        if rank == 0:
            final_path = args.work_dir / "checkpoint" / "epoch_59.pth"
            _atomic_torch_publish(
                final_path,
                {
                    "epoch": 59,
                    "successful_updates": successful_updates,
                    "state_dict": model.state_dict(),
                    "state_dict_ema": model_ema.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "grad_scaler": scaler.state_dict(),
                    "identity_hashes": identity_hashes,
                    "sample_order_trace_sha256": canonical_sha256(completed_orders),
                },
            )
            receipt: dict[str, Any] = {
                "schema_version": "s2_v3_full200_training_terminal_v1",
                "protocol_id": PROTOCOL_ID,
                "arm": arm,
                "seed": args.seed,
                "epochs": EXPECTED_EPOCHS,
                "successful_updates": successful_updates,
                "training_identity_count": EXPECTED_TRAINING_IDENTITIES,
                "world_size": world_size,
                "local_batch_size": 1,
                "global_batch_size": 2,
                "checkpoint_path": final_path.resolve().as_posix(),
                "checkpoint_sha256": sha256_file(final_path),
                "checkpoint_state": "epoch_59_state_dict_ema_update_6000",
                "sample_order_trace_sha256": canonical_sha256(completed_orders),
                "runtime_identity": runtime_identity_receipt,
                "update_audit": update_audit,
                "complete": True,
            }
            receipt["receipt_sha256"] = canonical_sha256(receipt)
            atomic_publish_json(args.work_dir / "training_terminal_receipt.json", receipt)
        dist.barrier()
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


__all__ = [
    "CERTIFIED_RECOVERY_UPDATES",
    "RECOVERY_SCHEMA",
    "REQUIRED_PAYLOAD_KEYS",
    "build_epoch_sampler_state",
    "build_recovery_payload",
    "capture_rng_state",
    "validate_epoch_sampler_state",
    "validate_full_data_manifest",
    "load_recovery_checkpoint",
    "restore_rng_state",
    "save_atomic_recovery_checkpoint",
    "validate_recovery_payload",
]


if __name__ == "__main__":
    raise SystemExit(main())
