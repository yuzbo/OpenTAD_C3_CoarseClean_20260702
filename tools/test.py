import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.models.utils.pc_ot_mras_raw_prediction_guard import (
    assert_no_raw_prediction_shortcut_for_pc_ot_mras,
)
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import eval_one_epoch
from opentad.utils import update_workdir, set_seed, create_folder, setup_logger
from opentad.utils.training_guard import (
    assert_detector_training_allowed,
    assert_safe_cfg_options_for_gated_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument(
        "--checkpoint", type=str, default="none", help="the checkpoint path"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument(
        "--not_eval",
        action="store_true",
        help="whether to not to eval, only do inference",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="optional positive inference-batch limit for smoke tests",
    )
    parser.add_argument(
        "--cfg-options", nargs="+", action=DictAction, help="override settings"
    )
    parser.add_argument(
        "--s1-test-open-certificate",
        type=str,
        default=None,
        help="required sealed-test certificate for manifest-bound S1 configs",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    assert_safe_cfg_options_for_gated_config(
        cfg, args.cfg_options, entrypoint="tools/test.py"
    )
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    s1_binding = None
    s1_bound_cfg = None
    if "spatial_zoom_s1_contract" in cfg:
        if args.checkpoint == "none" or not args.s1_test_open_certificate:
            raise ValueError(
                "S1 test requires an explicit selected checkpoint and test-open certificate"
            )
        if (
            args.cfg_options is not None
            or args.id != 0
            or args.not_eval
            or args.max_batches is not None
        ):
            raise ValueError(
                "formal S1 test forbids cfg overrides, alternate ids, partial inference, and no-eval mode"
            )
        from tools.bata.spatial_zoom_s1_test_open import validate_test_open_certificate
        from tools.bata.spatial_zoom_s1_contract import canonical_sha256
        from tools.bata.spatial_zoom_s1_evidence import S1_TEST_OPEN_MARKER_SCHEMA
        from tools.bata.spatial_zoom_s1_training import (
            require_clean_git_checkout,
            require_slurm_single_gpu_allocation,
            validate_bound_s1_training_config,
            validate_s1_checkpoint_sidecar,
        )

        require_slurm_single_gpu_allocation()
        s1_bound_cfg = Config.fromfile(args.config)
        if args.cfg_options is not None:
            s1_bound_cfg.merge_from_dict(args.cfg_options)
        s1_binding = validate_bound_s1_training_config(s1_bound_cfg, seed=args.seed)
        if not s1_binding["formal_precheck_verified"]:
            raise RuntimeError(
                "formal S1 test requires the bound full precheck certificate"
            )
        require_clean_git_checkout(expected_commit=s1_binding["code_commit"])
        certificate_path = os.path.abspath(args.s1_test_open_certificate)
        with open(certificate_path, "r", encoding="utf-8") as handle:
            certificate = json.load(handle)
        certificate = validate_test_open_certificate(
            certificate,
            cfg=s1_bound_cfg,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
        )
        sidecar = validate_s1_checkpoint_sidecar(args.checkpoint)
        cfg.dataset.test.subset_name = "validation"
        cfg.dataset.test.block_list = None
        cfg.evaluation.subset = "validation"
        cfg.post_processing.save_dict = False
        cfg.spatial_zoom_s1_test_binding = dict(
            bound_config_path=os.path.abspath(args.config),
            certificate_path=certificate_path,
            checkpoint_path=os.path.abspath(args.checkpoint),
            checkpoint_epoch=int(sidecar["experiment_metadata"]["epoch"]),
            state_key="state_dict_ema"
            if bool(cfg.solver.get("ema", False))
            else "state_dict",
            seed=int(args.seed),
        )
    assert_detector_training_allowed(cfg, entrypoint="tools/test.py")
    assert_no_raw_prediction_shortcut_for_pc_ot_mras(cfg)

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    if s1_binding is not None and args.world_size != 1:
        raise RuntimeError("formal S1 test is frozen to one Slurm GPU process")
    print(
        f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})"
    )
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # set random seed, create work_dir
    set_seed(args.seed)
    cfg = update_workdir(cfg, args.id, torch.cuda.device_count())
    if s1_binding is not None:
        marker_path = os.path.join(cfg.work_dir, "test_open_started.json")
        if os.path.exists(marker_path):
            raise FileExistsError(
                "sealed S1 test was already opened; refusing a second open"
            )
    if args.rank == 0:
        create_folder(cfg.work_dir)
        if s1_binding is not None:
            marker = {
                "schema_version": S1_TEST_OPEN_MARKER_SCHEMA,
                "resolution": int(s1_binding["resolution"]),
                "seed": int(args.seed),
                "bound_config_sha256": canonical_sha256(s1_bound_cfg.to_dict()),
                "code_commit": s1_binding["code_commit"],
                "experiment_namespace": s1_binding["experiment_namespace"],
                "canonical_experiment_root": s1_binding["canonical_experiment_root"],
                "checkpoint_sha256": sidecar["checkpoint_sha256"],
                "test_open_certificate_sha256": certificate["certificate_sha256"],
            }
            marker["marker_sha256"] = canonical_sha256(marker)
            with open(marker_path, "x", encoding="utf-8") as handle:
                json.dump(marker, handle, indent=2, sort_keys=True)
                handle.write("\n")
            cfg.spatial_zoom_s1_test_binding.open_marker_path = os.path.abspath(
                marker_path
            )

    # setup logger
    logger = setup_logger("Test", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(
        f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}"
    )
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    if s1_binding is not None:
        with open(s1_binding["manifest_path"], "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        runtime_test_ids = {str(row[0]) for row in test_dataset.data_list}
        if runtime_test_ids != set(manifest["splits"]["test"]):
            raise ValueError(
                "formal S1 test dataset does not match the sealed test split"
            )
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
    model = model.to(args.local_rank)
    model = DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank
    )
    logger.info(f"Using DDP with total {args.world_size} GPUS...")

    if (
        cfg.inference.load_from_raw_predictions
    ):  # if load with saved predictions, no need to load checkpoint
        logger.info(f"Loading from raw predictions: {cfg.inference.fuse_list}")
    else:  # load checkpoint: args -> config -> best
        if args.checkpoint != "none":
            checkpoint_path = args.checkpoint
        elif "test_epoch" in cfg.inference.keys():
            checkpoint_path = os.path.join(
                cfg.work_dir, f"checkpoint/epoch_{cfg.inference.test_epoch}.pth"
            )
        else:
            checkpoint_path = os.path.join(cfg.work_dir, "checkpoint/best.pth")
        logger.info("Loading checkpoint from: {}".format(checkpoint_path))
        device = f"cuda:{args.rank % torch.cuda.device_count()}"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info("Checkpoint is epoch {}.".format(checkpoint["epoch"]))
        if s1_binding is not None:
            if checkpoint.get("experiment_metadata") != sidecar["experiment_metadata"]:
                raise ValueError(
                    "S1 checkpoint payload metadata does not match its sidecar"
                )
            if int(checkpoint["epoch"]) != int(
                cfg.spatial_zoom_s1_test_binding.checkpoint_epoch
            ):
                raise ValueError("S1 test checkpoint epoch mismatch")

        # Model EMA
        use_ema = getattr(cfg.solver, "ema", False)
        if use_ema:
            model.load_state_dict(checkpoint["state_dict_ema"])
            logger.info("Using Model EMA...")
        else:
            model.load_state_dict(checkpoint["state_dict"])

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")

    # test the detector
    logger.info("Testing Starts...\n")
    eval_one_epoch(
        test_loader,
        model,
        cfg,
        logger,
        args.rank,
        model_ema=None,  # since we have loaded the ema model above
        use_amp=use_amp,
        world_size=args.world_size,
        not_eval=args.not_eval,
        max_batches=args.max_batches,
        epoch=None if s1_binding is None else int(checkpoint["epoch"]),
    )
    logger.info("Testing Over...\n")


if __name__ == "__main__":
    main()
