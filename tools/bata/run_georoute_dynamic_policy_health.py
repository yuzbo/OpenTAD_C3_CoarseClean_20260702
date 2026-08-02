#!/usr/bin/env python3
"""Run the 64-update real-fit-data dynamic SCNR policy-health gate."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_dynamic_policy_health import (  # noqa: E402
    DYNAMIC_POLICY_HEALTH_PASS,
    HEALTH_SEED,
    INITIAL_LOSS_SCALE,
    MAX_AMP_RETRIES_PER_BATCH,
    TARGET_SUCCESSFUL_UPDATES,
    DynamicPolicyHealthObserver,
    audit_no_performance_artifacts,
    bind_dynamic_policy_health_config,
    publish_dynamic_policy_health_report,
    require_clean_git_checkout,
    require_slurm_single_gpu,
    validate_dynamic_policy_health_config,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--seed", type=int, default=HEALTH_SEED)
    return parser.parse_args()


def _runtime_rank_contract() -> tuple[int, int, int]:
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            "policy-health runner must be launched by torchrun"
        ) from error
    if (local_rank, rank, world_size) != (0, 0, 1):
        raise RuntimeError("policy-health runner is frozen to one local rank on cuda:0")
    return local_rank, rank, world_size


def _execute(args: argparse.Namespace) -> dict:
    import torch
    import torch.distributed as dist
    from torch.cuda.amp import GradScaler
    from torch.nn.parallel import DistributedDataParallel

    from opentad.cores import build_optimizer, build_scheduler, train_one_epoch
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from opentad.utils import ModelEma, setup_logger, set_seed

    expected_commit = str(args.expected_commit).lower()
    source_receipt = require_clean_git_checkout(
        expected_commit=expected_commit,
        root=ROOT,
    )
    slurm_receipt = require_slurm_single_gpu()
    local_rank, rank, world_size = _runtime_rank_contract()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("policy-health requires exactly one visible CUDA device")

    run_root = args.run_root.resolve()
    if run_root.exists():
        raise FileExistsError(
            "policy-health run root already exists; resume is forbidden"
        )
    cfg = bind_dynamic_policy_health_config(
        source_config_path=args.source_config,
        work_dir=run_root,
        manifest_path=args.manifest,
        development_annotation_path=args.development_annotation,
        class_map_path=args.class_map,
        development_video_root=args.development_video_root,
        pretrained_checkpoint_path=args.pretrained,
        runtime_commit=expected_commit,
        seed=args.seed,
    )
    binding = validate_dynamic_policy_health_config(cfg, seed=args.seed)
    run_root.mkdir(parents=True, exist_ok=False)
    control_root = run_root / "control"
    control_root.mkdir(parents=True, exist_ok=False)
    bound_config_path = control_root / "bound_config.py"
    cfg.dump(str(bound_config_path))

    logger = setup_logger(
        "DynamicPolicyHealth", save_dir=run_root, distributed_rank=rank
    )
    observer = DynamicPolicyHealthObserver()
    update_audit = {
        "optimizer_attempts": 0,
        "amp_skipped_attempts": 0,
        "max_amp_retries_observed": 0,
        "consumed_batches": 0,
        "replay_attempts": 0,
        "scheduler_advances": 0,
        "ema_updates": 0,
    }
    successful_updates = 0
    execution_error: BaseException | None = None
    execution_traceback: str | None = None
    model = None
    model_ema = None
    dist_initialized = False
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        dist_initialized = True
        torch.cuda.set_device(local_rank)
        set_seed(args.seed, False, deterministic_warn_only=True)
        if (
            not torch.are_deterministic_algorithms_enabled()
            or not torch.is_deterministic_algorithms_warn_only_enabled()
        ):
            raise RuntimeError("policy-health deterministic warn-only contract failed")

        train_dataset = build_dataset(
            cfg.dataset.train,
            default_args=dict(logger=logger),
        )
        runtime_ids = {str(row[0]) for row in train_dataset.data_list}
        if runtime_ids != set(binding["training_video_ids"]):
            raise RuntimeError("policy-health train dataset differs from frozen Fit")
        train_loader = build_dataloader(
            train_dataset,
            rank=rank,
            world_size=world_size,
            shuffle=True,
            drop_last=True,
            **cfg.solver.train,
        )
        if len(train_loader) < TARGET_SUCCESSFUL_UPDATES:
            raise RuntimeError("policy-health Fit loader has fewer than 64 batches")

        model = build_detector(cfg.model).to(local_rank)
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            static_graph=False,
        )
        model_ema = ModelEma(model)
        optimizer = build_optimizer(copy.deepcopy(cfg.optimizer), model, logger)
        scheduler, _ = build_scheduler(
            copy.deepcopy(cfg.scheduler), optimizer, len(train_loader)
        )
        scaler = GradScaler(init_scale=INITIAL_LOSS_SCALE)
        train_loader.sampler.set_epoch(0)
        torch.cuda.reset_peak_memory_stats(local_rank)
        successful_updates = train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            0,
            logger,
            model_ema=model_ema,
            clip_grad_l2norm=float(cfg.solver.clip_grad_norm),
            logging_interval=int(cfg.workflow.logging_interval),
            scaler=scaler,
            max_train_iters=TARGET_SUCCESSFUL_UPDATES,
            fail_on_skipped_update=True,
            max_amp_retries_per_batch=MAX_AMP_RETRIES_PER_BATCH,
            update_audit=update_audit,
            successful_update_start=0,
            require_successful_update_hook=True,
            schedule_and_ema_on_success_only=True,
            capture_amp_rng_state=True,
            fail_on_nonfinite_loss=True,
            amp_diagnostic_observer=observer,
        )
        torch.cuda.synchronize(local_rank)
    except BaseException as error:
        execution_error = error
        execution_traceback = traceback.format_exc()

    try:
        try:
            peak_cuda_allocated_bytes = int(torch.cuda.max_memory_allocated(local_rank))
        except BaseException:
            peak_cuda_allocated_bytes = 0
        artifact_audit = audit_no_performance_artifacts(run_root)
        report = observer.build_report(
            binding=binding,
            source=source_receipt,
            slurm=slurm_receipt,
            update_audit=update_audit,
            successful_updates=successful_updates,
            artifact_audit=artifact_audit,
            peak_cuda_allocated_bytes=peak_cuda_allocated_bytes,
            execution_error=execution_error,
            execution_traceback=execution_traceback,
        )
        publish_dynamic_policy_health_report(binding["report_path"], report)
        summary_path = control_root / "execution_summary.json"
        summary = {
            "status": report["status"],
            "report_path": binding["report_path"],
            "report_sha256": report["report_sha256"],
            "summary": report["summary"],
            "bound_config_path": str(bound_config_path.resolve()),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    finally:
        if dist_initialized:
            dist.destroy_process_group()
    if execution_error is not None:
        raise RuntimeError(
            "dynamic policy-health execution failed; sealed report was published"
        ) from execution_error
    return report


def main() -> int:
    report = _execute(_parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_sha256": report["report_sha256"],
                "summary": report["summary"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == DYNAMIC_POLICY_HEALTH_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
