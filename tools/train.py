import json
import hashlib
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
    prepare_optimizer_parameter_freezing,
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


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_training_probe_bindings(cfg, args):
    probe_json = cfg.workflow.get("training_probe_json", None)
    if not probe_json:
        return None
    require_context = bool(cfg.workflow.get("require_training_probe_context", False))
    context_path = os.environ.get("DUCA_TRAINING_PROBE_CONTEXT_JSON", "")
    if require_context and not context_path:
        raise RuntimeError("formal training probe requires DUCA_TRAINING_PROBE_CONTEXT_JSON")
    context = None
    context_sha256 = None
    if context_path:
        context_path = os.path.abspath(os.path.expanduser(context_path))
        if not os.path.isfile(context_path):
            raise RuntimeError(f"training probe context is missing: {context_path}")
        with open(context_path, "r", encoding="utf-8") as handle:
            context = json.load(handle)
        context_sha256 = _sha256(context_path)

    config_path = os.path.abspath(os.path.expanduser(args.config))
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True, encoding="utf-8"
    ).strip()
    bindings = {
        "git_commit": git_commit,
        "seed": int(args.seed),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_config_path": config_path,
        "source_config_sha256": _sha256(config_path),
        "resolved_config_sha256": _canonical_sha256(cfg.to_dict()),
        "training_probe_json": os.path.abspath(os.path.expanduser(str(probe_json))),
        "context_json": context_path or None,
        "context_json_sha256": context_sha256,
        "context": context,
    }
    if require_context:
        expected = {
            "git_commit": bindings["git_commit"],
            "seed": bindings["seed"],
            "slurm_job_id": bindings["slurm_job_id"],
            "source_config_sha256": bindings["source_config_sha256"],
            "training_probe_json": bindings["training_probe_json"],
        }
        for key, value in expected.items():
            if context.get(key) != value:
                raise RuntimeError(
                    f"training probe context {key} mismatch: expected {value!r}, got {context.get(key)!r}"
                )
    return bindings


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
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
    assert_safe_cfg_options_for_gated_config(cfg, args.cfg_options, entrypoint="tools/train.py")
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    training_probe_bindings = _build_training_probe_bindings(cfg, args)
    assert_safe_entrypoint_args_for_gated_config(cfg, args, entrypoint="tools/train.py")
    assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

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

    # Optimizer exclusions change requires_grad and must be applied before DDP
    # captures its parameter set.
    prepare_optimizer_parameter_freezing(cfg.optimizer, model, logger)

    # DDP
    use_static_graph = getattr(cfg.solver, "static_graph", False)
    find_unused_parameters = getattr(cfg.solver, "find_unused_parameters", not use_static_graph)
    model = model.to(args.local_rank)
    model = DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=find_unused_parameters,
        static_graph=use_static_graph,
    )
    logger.info(
        "Using DDP with total %d GPUS (static_graph=%s, find_unused_parameters=%s)...",
        args.world_size,
        use_static_graph,
        find_unused_parameters,
    )

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
    for epoch in range(resume_epoch + 1, max_epoch):
        train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        training_probe_json = cfg.workflow.get("training_probe_json", None)
        training_probe = train_one_epoch(
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
            collect_training_probe=bool(training_probe_json),
        )
        if training_probe_json and args.rank == 0:
            training_probe.update(
                {
                    "epoch": int(epoch),
                    "config": str(args.config),
                    "world_size": int(args.world_size),
                    "static_graph": bool(use_static_graph),
                    "find_unused_parameters": bool(find_unused_parameters),
                    "bindings": training_probe_bindings,
                }
            )
            probe_path = os.path.abspath(os.path.expanduser(str(training_probe_json)))
            os.makedirs(os.path.dirname(probe_path), exist_ok=True)
            temporary_path = probe_path + ".tmp"
            try:
                with open(temporary_path, "w", encoding="utf-8") as handle:
                    json.dump(training_probe, handle, indent=2, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, probe_path)
            finally:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)

        # save checkpoint
        if not disable_checkpoint and (
            (epoch == max_epoch - 1) or ((epoch + 1) % cfg.workflow.checkpoint_interval == 0)
        ):
            if args.rank == 0:
                save_checkpoint(model, model_ema, optimizer, scheduler, epoch, work_dir=cfg.work_dir)

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
                    if not disable_checkpoint and args.rank == 0:
                        save_best_checkpoint(model, model_ema, epoch, work_dir=cfg.work_dir)

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
                )
    logger.info("Training Over...\n")


if __name__ == "__main__":
    main()
