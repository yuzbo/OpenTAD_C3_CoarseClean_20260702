import copy
import hashlib
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import (
    train_one_epoch,
    val_one_epoch,
    eval_one_epoch,
    build_optimizer,
    build_scheduler,
)
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
from opentad.utils.training_guard import (
    assert_detector_training_allowed,
    assert_safe_entrypoint_args_for_gated_config,
    assert_safe_cfg_options_for_gated_config,
)
from opentad.utils.train_schedule import should_eval_epoch
from tools.bata.duca_runtime_contract import resolve_effective_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument(
        "--resume", type=str, default=None, help="resume from a checkpoint"
    )
    parser.add_argument(
        "--not_eval", action="store_true", help="whether not to eval, only do inference"
    )
    parser.add_argument(
        "--disable_deterministic",
        action="store_true",
        help="disable deterministic for faster speed",
    )
    parser.add_argument(
        "--cfg-options", nargs="+", action=DictAction, help="override settings"
    )
    args = parser.parse_args()
    return args


DUCA_UNIFIED_CHECKPOINT_SIDECAR_SCHEMA = "duca_unified_checkpoint_metadata_v1"


def _resolve_effective_seed(cfg, args):
    return resolve_effective_seed(cfg, args.seed)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _write_json(path, payload):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _build_duca_unified_checkpoint_metadata(
    cfg,
    args,
    epoch,
    successful_updates,
    train_batches_per_epoch,
    update_audit,
):
    config_path = os.path.abspath(args.config)
    metadata = {
        "schema": DUCA_UNIFIED_CHECKPOINT_SIDECAR_SCHEMA,
        "code_commit": _git_head_commit(),
        "config_path": config_path,
        "config_sha256": _sha256_file(config_path),
        "matrix_id": cfg.get("matrix_id", None),
        "matrix_index": cfg.get("matrix_index", None),
        "phase": cfg.get("matrix_phase", cfg.get("phase", None)),
        "task_id": cfg.get("task_id", None),
        "arm_id": cfg.get("arm_id", None),
        "seed": int(args.seed),
        "epoch": int(epoch),
        "terminal_epoch_zero_based": int(cfg.get("terminal_epoch_zero_based", epoch)),
        "successful_updates": int(successful_updates),
        "max_successful_updates": int(cfg.get("max_successful_updates", 0)),
        "train_batches_per_epoch": int(train_batches_per_epoch),
        "terminal_state_key": cfg.get("terminal_state_key", None),
        "amp_skipped_attempts": int(update_audit.get("amp_skipped_attempts", 0)),
        "optimizer_attempts": int(update_audit.get("optimizer_attempts", 0)),
        "max_amp_retries_observed": int(
            update_audit.get("max_amp_retries_observed", 0)
        ),
        "artifact_contract": cfg.get("duca_unified_contract", {}),
    }
    return metadata


def _is_duca_unified_cfg(cfg):
    return "duca_unified_contract" in cfg or str(cfg.get("matrix_id", "")).startswith(
        "DUCA_UNIFIED_FULLMATRIX"
    )


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    assert_safe_cfg_options_for_gated_config(
        cfg, args.cfg_options, entrypoint="tools/train.py"
    )
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    args.seed = _resolve_effective_seed(cfg, args)
    s1_binding = None
    s1_checkpoint_sidecar_schema = None
    if "spatial_zoom_s1_contract" in cfg:
        from tools.bata.spatial_zoom_s1_training import (
            S1_CHECKPOINT_SIDECAR_SCHEMA,
            build_s1_checkpoint_metadata,
            require_clean_git_checkout,
            require_slurm_single_gpu_allocation,
            validate_bound_s1_training_config,
        )

        s1_checkpoint_sidecar_schema = S1_CHECKPOINT_SIDECAR_SCHEMA
        require_slurm_single_gpu_allocation()
        s1_binding = validate_bound_s1_training_config(cfg, seed=args.seed)
        if not s1_binding["formal_precheck_verified"]:
            raise RuntimeError(
                "formal S1 training requires a bound full precheck certificate"
            )
        require_clean_git_checkout(expected_commit=s1_binding["code_commit"])
        if args.cfg_options is not None:
            raise ValueError("formal S1 training forbids --cfg-options")
        if args.resume is not None:
            raise ValueError(
                "formal S1 training forbids resume; start a fresh bound run"
            )
        if args.disable_deterministic or args.not_eval:
            raise ValueError(
                "formal S1 training requires deterministic execution with gate evaluation"
            )
    assert_safe_entrypoint_args_for_gated_config(cfg, args, entrypoint="tools/train.py")
    assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    if s1_binding is not None and args.world_size != 1:
        raise RuntimeError("formal S1 training is frozen to one Slurm GPU process")
    print(
        f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})"
    )
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # set random seed, create work_dir, and save config
    set_seed(args.seed, args.disable_deterministic)
    if s1_binding is None:
        cfg = update_workdir(cfg, args.id, args.world_size)
    elif args.id != 0:
        raise ValueError("formal S1 work_dir is manifest-bound; --id must remain zero")
    elif os.path.exists(cfg.work_dir):
        raise FileExistsError("formal S1 training requires a fresh bound work_dir")
    if args.rank == 0:
        create_folder(cfg.work_dir)
        save_config(args.config, cfg.work_dir)

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(
        f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}"
    )
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
    if s1_binding is not None:
        runtime_ids = {
            "train": {str(row[0]) for row in train_dataset.data_list},
            "val": {str(row[0]) for row in val_dataset.data_list},
            "test": {str(row[0]) for row in test_dataset.data_list},
        }
        if runtime_ids["train"] != set(s1_binding["fit_video_ids"]):
            raise ValueError(
                "formal S1 training dataset does not match the frozen fit split"
            )
        if runtime_ids["val"] != set(s1_binding["gate_video_ids"]) or runtime_ids[
            "test"
        ] != set(s1_binding["gate_video_ids"]):
            raise ValueError(
                "formal S1 gate loaders do not match the frozen gate split"
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
    optimizer = build_optimizer(copy.deepcopy(cfg.optimizer), model, logger)
    scheduler, max_epoch = build_scheduler(
        copy.deepcopy(cfg.scheduler), optimizer, len(train_loader)
    )

    # override the max_epoch
    max_epoch = cfg.workflow.get("end_epoch", max_epoch)

    # resume: reset epoch, optimizer, scheduler, and EMA
    if args.resume != None:
        logger.info("Resume training from: {}".format(args.resume))
        device = f"cuda:{args.local_rank}"
        checkpoint = torch.load(args.resume, map_location=device)
        resume_epoch = checkpoint["epoch"]
        logger.info("Resume epoch is {}".format(resume_epoch))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if model_ema != None:
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])

        del checkpoint  #  save memory if the model is very large such as ViT-g
        torch.cuda.empty_cache()
    else:
        resume_epoch = -1

    # train the detector
    logger.info("Training Starts...\n")
    val_loss_best = 1e6
    val_start_epoch = cfg.workflow.get("val_start_epoch", 0)
    disable_checkpoint = cfg.workflow.get("disable_checkpoint", False)
    successful_updates = 0
    max_successful_updates = cfg.get("max_successful_updates", None)
    if max_successful_updates is not None:
        max_successful_updates = int(max_successful_updates)
        if max_successful_updates <= 0:
            raise ValueError("max_successful_updates must be positive when configured")
    formal_update_target = max_successful_updates is not None
    update_audit = {
        "optimizer_attempts": 0,
        "amp_skipped_attempts": 0,
        "max_amp_retries_observed": 0,
    }
    configured_amp_retry_limit = int(cfg.workflow.get("max_amp_retries_per_batch", 8))
    amp_retry_limit = configured_amp_retry_limit if (
        use_amp and (s1_binding is not None or formal_update_target)
    ) else 0
    for epoch in range(resume_epoch + 1, max_epoch):
        train_loader.sampler.set_epoch(epoch)

        # train for one epoch
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
            max_train_iters=cfg.workflow.get("max_train_iters", None),
            fail_on_skipped_update=s1_binding is not None or formal_update_target,
            max_amp_retries_per_batch=amp_retry_limit,
            max_successful_updates=max_successful_updates,
            successful_updates_start=successful_updates,
            update_audit=(
                update_audit if (s1_binding is not None or formal_update_target) else None
            ),
        )

        # save checkpoint
        if not disable_checkpoint and (
            (epoch == max_epoch - 1)
            or ((epoch + 1) % cfg.workflow.checkpoint_interval == 0)
        ):
            if args.rank == 0:
                checkpoint_metadata = None
                checkpoint_sidecar_schema = s1_checkpoint_sidecar_schema
                if s1_binding is not None:
                    checkpoint_metadata = build_s1_checkpoint_metadata(
                        cfg,
                        seed=args.seed,
                        epoch=epoch,
                        successful_updates=successful_updates,
                        train_batches_per_epoch=len(train_loader),
                        amp_skipped_attempts=update_audit["amp_skipped_attempts"],
                        max_amp_retries_per_batch=amp_retry_limit,
                        max_amp_retries_observed=update_audit[
                            "max_amp_retries_observed"
                        ],
                    )
                elif _is_duca_unified_cfg(cfg):
                    if (
                        epoch == int(cfg.get("terminal_epoch_zero_based", epoch))
                        and max_successful_updates is not None
                        and successful_updates != max_successful_updates
                    ):
                        raise RuntimeError(
                            "DUCA Unified terminal checkpoint requires exactly "
                            f"{max_successful_updates} successful updates; got "
                            f"{successful_updates}"
                        )
                    checkpoint_metadata = _build_duca_unified_checkpoint_metadata(
                        cfg,
                        args,
                        epoch=epoch,
                        successful_updates=successful_updates,
                        train_batches_per_epoch=len(train_loader),
                        update_audit=update_audit,
                    )
                    checkpoint_sidecar_schema = DUCA_UNIFIED_CHECKPOINT_SIDECAR_SCHEMA
                    _write_json(
                        os.path.join(cfg.work_dir, "duca_unified_runtime_identity.json"),
                        checkpoint_metadata,
                    )
                save_checkpoint(
                    model,
                    model_ema,
                    optimizer,
                    scheduler,
                    epoch,
                    work_dir=cfg.work_dir,
                    experiment_metadata=checkpoint_metadata,
                    experiment_sidecar_schema=checkpoint_sidecar_schema,
                )

        # val for one epoch
        if epoch >= val_start_epoch:
            if (cfg.workflow.val_loss_interval > 0) and (
                (epoch + 1) % cfg.workflow.val_loss_interval == 0
            ):
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
                    if not disable_checkpoint and args.rank == 0:
                        save_best_checkpoint(
                            model, model_ema, epoch, work_dir=cfg.work_dir
                        )

        # eval for one epoch
        if epoch >= val_start_epoch:
            if should_eval_epoch(epoch, cfg.workflow):
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
                    epoch=epoch,
                )
        if max_successful_updates is not None and successful_updates >= max_successful_updates:
            logger.info(
                "[Train]: stopping after %d successful optimizer updates.",
                successful_updates,
            )
            break
    if max_successful_updates is not None and successful_updates != max_successful_updates:
        raise RuntimeError(
            "training ended before reaching the configured successful update target: "
            f"{successful_updates}/{max_successful_updates}"
        )
    logger.info("Training Over...\n")


if __name__ == "__main__":
    main()
