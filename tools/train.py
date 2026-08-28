import copy
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
from tools.bata import (
    duca_cellcf_training,
    duca_p0_training,
    duca_protected_physical_training,
    duca_selected_axis_training,
)
from tools.bata.duca_frontend_initialization import (
    initialize_model_from_checkpoint,
    initialize_frame_selector_from_checkpoint,
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_intermediate_evaluation(work_dir, epoch, evaluation, *, select_best=False):
    """Seal one official validation result without changing optimization."""

    if evaluation is None:
        return
    folder = os.path.join(work_dir, "intermediate_validation")
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, f"epoch_{epoch + 1:03d}_ema.json")
    payload = {
        "epoch": int(epoch + 1),
        "state_key": "state_dict_ema",
        "checkpoint_path": os.path.abspath(
            os.path.join(work_dir, "checkpoint", f"epoch_{epoch}.pth")
        ),
        "metrics": evaluation.get("metrics"),
        "result_path": evaluation.get("result_path"),
        "result_count": int(evaluation.get("result_count", 0)),
        "video_count": int(evaluation.get("video_count", 0)),
        "evaluator": evaluation.get("evaluator"),
    }
    temporary = target + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)

    if not select_best:
        return

    metrics = evaluation.get("metrics") or {}
    average_map = metrics.get("average_mAP")
    if average_map is None:
        return
    best_path = os.path.join(folder, "best_validation_ema.json")
    previous = None
    if os.path.isfile(best_path):
        with open(best_path, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
    if previous is not None and float(previous.get("average_mAP", float("-inf"))) >= float(average_map):
        return
    best_payload = {
        "selection_metric": "average_mAP",
        "average_mAP": float(average_map),
        "epoch": int(epoch + 1),
        "state_key": "state_dict_ema",
        "checkpoint_path": payload["checkpoint_path"],
        "evaluation_path": os.path.abspath(target),
    }
    best_temporary = best_path + ".tmp"
    try:
        with open(best_temporary, "w", encoding="utf-8") as handle:
            json.dump(best_payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(best_temporary, best_path)
    finally:
        if os.path.exists(best_temporary):
            os.remove(best_temporary)


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


def _select_duca_training(formal_protocol):
    if formal_protocol == "duca_cellcf_v1":
        return duca_cellcf_training
    if formal_protocol == duca_protected_physical_training.FORMAL_PROTOCOL:
        return duca_protected_physical_training
    if duca_selected_axis_training.is_formal_protocol(formal_protocol):
        return duca_selected_axis_training
    return duca_p0_training


def _dispatch_duca_runtime_bindings(
    duca_training,
    runtime_binding_kwargs,
    *,
    selector_initialization=None,
    formal_protocol=None,
    r5_cell=None,
):
    kwargs = dict(runtime_binding_kwargs)
    if duca_training is duca_selected_axis_training:
        kwargs["selector_initialization"] = selector_initialization
        kwargs["formal_protocol"] = formal_protocol
        kwargs["r5_cell"] = r5_cell
    return duca_training.build_runtime_bindings(**kwargs)


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
    formal_protocol = str(cfg.workflow.get("formal_protocol", ""))
    duca_training = _select_duca_training(formal_protocol)
    source_config_sha256 = _sha256(args.config)
    source_resolved_config_sha256 = _canonical_sha256(cfg.to_dict())
    duca_formal_contract = duca_training.formal_training_contract(cfg)
    if duca_training is duca_cellcf_training:
        duca_cellcf_training.assert_safe_cfg_options(
            cfg, args.cfg_options, entrypoint="tools/train.py"
        )
    elif duca_training is duca_protected_physical_training:
        duca_protected_physical_training.assert_safe_cfg_options(
            args.cfg_options,
            entrypoint="tools/train.py",
        )
    elif duca_training is duca_selected_axis_training:
        duca_selected_axis_training.assert_safe_cfg_options(
            args.cfg_options,
            entrypoint="tools/train.py",
        )
    assert_safe_cfg_options_for_gated_config(cfg, args.cfg_options, entrypoint="tools/train.py")
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if duca_training is duca_cellcf_training:
        duca_formal_contract = duca_cellcf_training.formal_training_contract(cfg)
    duca_git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True, encoding="utf-8"
    ).strip()
    if duca_formal_contract is not None:
        expected_commit = os.environ.get("DUCA_EXPECTED_COMMIT")
        if expected_commit != duca_git_commit:
            raise RuntimeError("formal DUCA checkout differs from DUCA_EXPECTED_COMMIT")
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=path,
            text=True,
            encoding="utf-8",
        ).strip()
        if status:
            raise RuntimeError("formal DUCA training requires a clean exact-commit checkout")
        if args.disable_deterministic or args.not_eval:
            raise ValueError("formal DUCA training requires deterministic execution")
    training_probe_bindings = _build_training_probe_bindings(cfg, args)
    assert_safe_entrypoint_args_for_gated_config(cfg, args, entrypoint="tools/train.py")
    assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    if duca_formal_contract is not None and args.world_size != 1:
        raise RuntimeError("formal DUCA P0 is frozen to one Slurm GPU process")
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # set random seed, create work_dir, and save config
    set_seed(args.seed, args.disable_deterministic)
    cfg = update_workdir(cfg, args.id, args.world_size)
    duca_runtime_bindings = None
    if duca_formal_contract is not None:
        variant = (
            os.environ.get("DUCA_PROTECTED_VARIANT", "")
            if duca_training is duca_protected_physical_training
            else (
                os.environ.get("DUCA_SELECTED_OPT_VARIANT", "")
                if duca_training is duca_selected_axis_training
                else os.environ.get("DUCA_P0_VARIANT", "")
            )
        )
        runtime_binding_kwargs = dict(
            git_commit=duca_git_commit,
            variant=variant,
            seed=args.seed,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            source_config_path=args.config,
            source_config_sha256=source_config_sha256,
            resolved_config_sha256=source_resolved_config_sha256,
            runtime_config_sha256=_canonical_sha256(cfg.to_dict()),
            evaluation_annotation_path=cfg.evaluation.ground_truth_filename,
            evaluation_class_map_path=cfg.dataset.test.class_map,
            evaluation_config=cfg.evaluation,
        )
        if duca_training in (
            duca_cellcf_training,
            duca_protected_physical_training,
            duca_selected_axis_training,
        ):
            runtime_binding_kwargs["runtime_pretrain_path"] = cfg.model.backbone.custom.pretrain
        duca_runtime_bindings = _dispatch_duca_runtime_bindings(
            duca_training,
            runtime_binding_kwargs,
            selector_initialization=cfg.workflow.get(
                "selector_initialization", None
            ),
            formal_protocol=formal_protocol,
            r5_cell=cfg.get("r5_cell", None),
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

    seal_eval_loaders = bool(
        cfg.workflow.get("seal_eval_dataloaders_during_training", False)
    )
    if seal_eval_loaders:
        if cfg.dataset.get("val", None) is not None:
            raise RuntimeError(
                "sealed formal training requires dataset.val=None"
            )
        if int(cfg.workflow.get("val_loss_interval", -1)) > 0 or int(
            cfg.workflow.get("val_eval_interval", -1)
        ) > 0:
            raise RuntimeError(
                "sealed formal training forbids validation/test intervals"
            )
        val_loader = None
        test_loader = None
        logger.info(
            "Validation and test dataloaders are sealed during formal training."
        )
    else:
        val_dataset = build_dataset(
            cfg.dataset.val, default_args=dict(logger=logger)
        )
        val_loader = build_dataloader(
            val_dataset,
            rank=args.rank,
            world_size=args.world_size,
            shuffle=False,
            drop_last=False,
            **cfg.solver.val,
        )

        test_dataset = build_dataset(
            cfg.dataset.test, default_args=dict(logger=logger)
        )
        test_loader = build_dataloader(
            test_dataset,
            rank=args.rank,
            world_size=args.world_size,
            shuffle=False,
            drop_last=False,
            **cfg.solver.test,
        )
    if duca_formal_contract is not None:
        bind_loader = getattr(
            duca_training, "bind_train_loader_contract", None
        )
        if callable(bind_loader):
            duca_formal_contract, train_loader_binding = bind_loader(
                duca_formal_contract,
                cfg=cfg,
                train_dataset=train_dataset,
                train_loader=train_loader,
                world_size=args.world_size,
            )
            duca_runtime_bindings["train_loader_contract_sha256"] = (
                train_loader_binding["contract_sha256"]
            )
        expected_batches = int(
            duca_formal_contract["expected_train_batches_per_epoch"]
        )
        if len(train_loader) != expected_batches:
            raise RuntimeError(
                f"formal DUCA loader has {len(train_loader)} batches, "
                f"expected {expected_batches}"
            )

    # build model
    model = build_detector(cfg.model)
    model_initialization_cfg = cfg.workflow.get("model_initialization", None)
    selector_initialization_cfg = cfg.workflow.get("selector_initialization", None)
    if model_initialization_cfg and selector_initialization_cfg:
        raise ValueError(
            "model_initialization and selector_initialization are mutually exclusive"
        )
    model_initialization_receipt = initialize_model_from_checkpoint(
        model,
        model_initialization_cfg,
        logger=logger,
    )
    selector_initialization_receipt = initialize_frame_selector_from_checkpoint(
        model,
        selector_initialization_cfg,
        logger=logger,
    )
    if model_initialization_receipt is not None and duca_runtime_bindings is not None:
        duca_runtime_bindings["model_initialization_receipt"] = dict(
            model_initialization_receipt
        )
    if selector_initialization_receipt is not None and duca_runtime_bindings is not None:
        duca_runtime_bindings["selector_initialization_receipt"] = dict(
            selector_initialization_receipt
        )

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
    if duca_formal_contract is not None and (not use_amp or model_ema is None):
        raise RuntimeError("formal DUCA requires both AMP and model EMA")

    # build optimizer and scheduler
    optimizer = build_optimizer(copy.deepcopy(cfg.optimizer), model, logger)
    scheduler, max_epoch = build_scheduler(
        copy.deepcopy(cfg.scheduler), optimizer, len(train_loader)
    )

    # override the max_epoch
    max_epoch = cfg.workflow.get("end_epoch", max_epoch)

    max_amp_retries_per_batch = int(
        cfg.workflow.get("max_amp_retries_per_batch", 0)
    )
    max_nonfinite_loss_retries = int(
        cfg.workflow.get("max_nonfinite_loss_retries", 0)
    )
    fail_on_amp_replay_exhaustion = bool(
        cfg.workflow.get("fail_on_amp_replay_exhaustion", False)
    )
    require_finite_train_loss = bool(
        cfg.workflow.get("require_finite_train_loss", False)
    )
    collect_update_audit = bool(
        duca_formal_contract is not None
        or max_amp_retries_per_batch > 0
        or max_nonfinite_loss_retries > 0
        or cfg.workflow.get("training_probe_json", None)
    )
    update_audit = duca_training.new_update_audit() if collect_update_audit else None
    preserve_resume_state = bool(cfg.workflow.get("preserve_resume_state", False))
    epoch_records = []

    # resume: reset epoch, optimizer, scheduler, EMA, scaler, and formal audit
    if args.resume != None:
        logger.info("Resume training from: {}".format(args.resume))
        # Keep serialized RNG tensors on CPU. Model and optimizer state loading
        # moves tensors to their parameter devices, while torch.set_rng_state
        # requires a CPU ByteTensor.
        checkpoint = torch.load(args.resume, map_location="cpu")
        resume_epoch = checkpoint["epoch"]
        logger.info("Resume epoch is {}".format(resume_epoch))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if model_ema != None:
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])
        if scaler is not None and "grad_scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["grad_scaler"])
        elif duca_formal_contract is not None or preserve_resume_state:
            raise RuntimeError("formal DUCA resume checkpoint lacks GradScaler state")
        if duca_formal_contract is not None:
            update_audit, epoch_records = duca_training.restore_training_state(
                checkpoint,
                contract=duca_formal_contract,
                bindings=duca_runtime_bindings,
            )
            duca_training.validate_checkpoint_successful_optimizer_updates(
                checkpoint, update_audit
            )
            duca_training.validate_update_state(
                contract=duca_formal_contract,
                epoch=resume_epoch,
                train_batches_per_epoch=len(train_loader),
                update_audit=update_audit,
                scheduler_last_epoch=scheduler.last_epoch,
                selector_step=duca_training.selector_schedule_step(model),
                uses_ema=model_ema is not None,
            )
            rng_state = checkpoint.get("rng_state")
            if not isinstance(rng_state, dict):
                raise RuntimeError("formal DUCA resume checkpoint lacks global RNG state")
            duca_training.restore_global_rng_state(rng_state)
        elif preserve_resume_state:
            successful_updates = checkpoint.get("successful_optimizer_updates")
            if (
                update_audit is None
                or not isinstance(successful_updates, int)
                or isinstance(successful_updates, bool)
                or successful_updates < 0
            ):
                raise RuntimeError("resume checkpoint lacks a valid optimizer update count")
            update_audit["successful_optimizer_updates"] = successful_updates
            rng_state = checkpoint.get("rng_state")
            if not isinstance(rng_state, dict):
                raise RuntimeError("resume checkpoint lacks global RNG state")
            duca_training.restore_global_rng_state(rng_state)

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
        dataset_set_epoch = getattr(train_loader.dataset, "set_epoch", None)
        if bool(cfg.workflow.get("derive_train_loader_contract", False)):
            if not callable(dataset_set_epoch):
                raise RuntimeError(
                    "formal stateless training dataset has no set_epoch"
                )
            dataset_set_epoch(epoch)
        train_loader.sampler.set_epoch(epoch)
        audit_before_epoch = (
            dict(update_audit) if update_audit is not None else None
        )

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
            max_amp_retries_per_batch=max_amp_retries_per_batch,
            max_nonfinite_loss_retries=max_nonfinite_loss_retries,
            fail_on_amp_replay_exhaustion=fail_on_amp_replay_exhaustion,
            require_finite_loss=require_finite_train_loss,
            force_amp_overflow_attempts=int(
                cfg.workflow.get("force_amp_overflow_attempts", 0)
            ),
            update_audit=update_audit,
            update_audit_json=cfg.workflow.get("training_update_audit_json", None),
        )
        training_audit = None
        if duca_formal_contract is not None:
            duca_training.validate_update_state(
                contract=duca_formal_contract,
                epoch=epoch,
                train_batches_per_epoch=len(train_loader),
                update_audit=update_audit,
                scheduler_last_epoch=scheduler.last_epoch,
                selector_step=duca_training.selector_schedule_step(model),
                uses_ema=model_ema is not None,
            )
            delta = {
                key: int(update_audit[key]) - int(audit_before_epoch[key])
                for key in update_audit
            }
            expected_batches = int(
                duca_formal_contract["expected_train_batches_per_epoch"]
            )
            if delta["attempted_batches"] != expected_batches:
                raise RuntimeError(
                    "formal DUCA epoch did not consume the derived batch count"
                )
            if delta["successful_optimizer_updates"] != expected_batches:
                raise RuntimeError(
                    "formal DUCA epoch did not execute the derived update count"
                )
            epoch_records.append(
                {
                    "epoch": int(epoch),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "counter_delta": delta,
                    "scheduler_last_epoch": int(scheduler.last_epoch),
                    "selector_schedule_step": int(duca_training.selector_schedule_step(model)),
                    "grad_scaler_scale": (
                        None if scaler is None else float(scaler.get_scale())
                    ),
                }
            )
            training_audit = duca_training.build_training_audit(
                contract=duca_formal_contract,
                bindings=duca_runtime_bindings,
                epoch=epoch,
                train_batches_per_epoch=len(train_loader),
                update_audit=update_audit,
                epoch_records=epoch_records,
                scheduler_last_epoch=scheduler.last_epoch,
                selector_step=duca_training.selector_schedule_step(model),
                scaler_scale=None if scaler is None else scaler.get_scale(),
                uses_ema=model_ema is not None,
                complete=epoch == max_epoch - 1,
            )
            if args.rank == 0:
                duca_training.atomic_write_json(
                    os.path.join(
                        cfg.work_dir,
                        getattr(
                            duca_training,
                            "DUCA_TRAINING_AUDIT_FILENAME",
                            "duca_p0_training_audit.json",
                        ),
                    ),
                    training_audit,
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
                    "update_audit": None
                    if update_audit is None
                    else dict(update_audit),
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
                checkpoint_metadata = (
                    None
                    if training_audit is None
                    else duca_training.build_checkpoint_metadata(training_audit)
                )
                save_checkpoint(
                    model,
                    model_ema,
                    optimizer,
                    scheduler,
                    epoch,
                    work_dir=cfg.work_dir,
                    scaler=scaler,
                    rng_state=(
                        None
                        if duca_formal_contract is None and not preserve_resume_state
                        else duca_training.capture_global_rng_state()
                    ),
                    successful_optimizer_updates=(
                        None
                        if update_audit is None
                        else update_audit["successful_optimizer_updates"]
                    ),
                    experiment_metadata=checkpoint_metadata,
                    experiment_sidecar_schema=(
                        None
                        if checkpoint_metadata is None
                        else duca_training.DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA
                    ),
                )

        # val for one epoch
        if epoch >= val_start_epoch:
            if (cfg.workflow.val_loss_interval > 0) and ((epoch + 1) % cfg.workflow.val_loss_interval == 0):
                val_loss = val_one_epoch(
                    # val_loader is non-None whenever this branch is enabled.
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
                    if (
                        duca_formal_contract is None
                        and not disable_checkpoint
                        and args.rank == 0
                    ):
                        save_best_checkpoint(model, model_ema, epoch, work_dir=cfg.work_dir)

        # eval for one epoch
        if epoch >= val_start_epoch:
            if should_eval_epoch(epoch, cfg.workflow):
                evaluation = eval_one_epoch(
                    # test_loader is non-None whenever this branch is enabled.
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
                if args.rank == 0:
                    _write_intermediate_evaluation(
                        cfg.work_dir,
                        epoch,
                        evaluation,
                        select_best=bool(
                            cfg.workflow.get(
                                "intermediate_validation_selects_checkpoint", False
                            )
                        ),
                    )
    logger.info("Training Over...\n")


if __name__ == "__main__":
    main()
