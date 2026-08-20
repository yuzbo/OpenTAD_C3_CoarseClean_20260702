import os
import random
import re
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
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
ZOOMTOKEN_RECOVERY_ARMS = {"DN", "G"}


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
            "ZoomToken recovery is restricted to the frozen DN/G surfaces"
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
        raise ValueError("ZoomToken DN/G recovery must run every five epochs")
    if cfg.workflow.get("checkpoint_policy", None) != "recovery_latest3_plus_final":
        raise ValueError("ZoomToken DN/G checkpoint policy is not recovery/latest3/final")
    if not cfg.solver.get("amp", False) or not cfg.solver.get("ema", False):
        raise ValueError("ZoomToken full-state recovery requires the frozen AMP/EMA recipe")
    work_dir_parts = os.path.normpath(cfg.work_dir).replace("\\", "/").split("/")
    if p1_config.get("runner_binding_required", False) and any(
        part.endswith("_unbound") for part in work_dir_parts
    ):
        raise ValueError(
            "ZoomToken DN/G work_dir must be explicitly bound by --work-dir "
            "or the stage runner"
        )
    contract["arm_surface"] = arm_surface
    contract["seed"] = int(p1_config["seed"])
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
):
    if recovery_contract["arm_surface"] == "G":
        if (
            not isinstance(next_successful_update_index, int)
            or next_successful_update_index < 0
        ):
            raise ValueError("ZoomToken G recovery requires a non-negative update index")
    elif next_successful_update_index is not None:
        raise ValueError("ZoomToken DN recovery must not carry a GeoRoute update index")
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

    return {
        "schema_version": ZOOMTOKEN_RECOVERY_SCHEMA,
        "arm_surface": recovery_contract["arm_surface"],
        "seed": args.seed,
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


def _restore_zoomtoken_training_state(
    checkpoint, args, cfg, train_loader, recovery_contract
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
    if recovery_contract["arm_surface"] == "G":
        if (
            not isinstance(next_successful_update_index, int)
            or next_successful_update_index < 0
        ):
            raise ValueError("ZoomToken G recovery lacks a valid update index")
    elif next_successful_update_index is not None:
        raise ValueError("ZoomToken DN recovery must not carry a GeoRoute update index")

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
    next_successful_update_index = (
        0
        if recovery_contract is not None
        and recovery_contract["arm_surface"] == "G"
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
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if model_ema is not None:
            if "state_dict_ema" not in checkpoint:
                raise ValueError("resume checkpoint lacks the required EMA state")
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])
        if recovery_contract is not None:
            if scaler is not None:
                if "scaler" not in checkpoint:
                    raise ValueError("ZoomToken recovery checkpoint lacks scaler state")
                scaler.load_state_dict(checkpoint["scaler"])
            next_successful_update_index = _restore_zoomtoken_training_state(
                checkpoint, args, cfg, train_loader, recovery_contract
            )

        del checkpoint  # save memory if the model is very large such as ViT-g
        torch.cuda.empty_cache()
    else:
        resume_epoch = -1

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
