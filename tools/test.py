import os
import sys
from pathlib import Path

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
    parser.add_argument(
        "--s1-profile-recovery-certificate",
        type=str,
        default=None,
        help=(
            "audited profile-runtime certificate required when formal S1 test "
            "runs from an infrastructure-only recovery commit"
        ),
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
    georoute_official_development_binding = None
    georoute_dynamic_floor_m2_binding = None
    georoute_dynamic_floor_m2_checkpoint_sidecar = None
    strict_dynamic_floor_m2_determinism = False
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
        from tools.bata.spatial_zoom_s1_contract import (
            atomic_publish_json,
            canonical_sha256,
            sha256_file,
        )
        from tools.bata.spatial_zoom_s1_evidence import S1_TEST_OPEN_MARKER_SCHEMA
        from tools.bata.spatial_zoom_s1_profile_recovery import (
            S1_STEP_SCOPED_TEST_RUNTIME_MODE,
            load_profile_recovery_certificate,
        )
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
        recovery = None
        recovery_path = None
        if args.s1_profile_recovery_certificate:
            recovery_path = os.path.abspath(args.s1_profile_recovery_certificate)
            recovery = load_profile_recovery_certificate(
                recovery_path,
                binding=s1_binding,
                verify_checkout=True,
            )
            if (
                recovery.get("formal_test_runtime_mode")
                != S1_STEP_SCOPED_TEST_RUNTIME_MODE
            ):
                raise RuntimeError(
                    "formal S1 recovery test requires the step-scoped runtime contract"
                )
        else:
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
        if recovery is not None:
            cfg.spatial_zoom_s1_test_binding.update(
                formal_test_runtime_mode=S1_STEP_SCOPED_TEST_RUNTIME_MODE,
                training_code_commit=s1_binding["code_commit"],
                test_runtime_code_commit=recovery["profile_code_commit"],
                profile_recovery_certificate_path=recovery_path,
                profile_recovery_certificate_file_sha256=(sha256_file(recovery_path)),
                profile_recovery_certificate_sha256=(recovery["certificate_sha256"]),
                profile_recovery_campaign_id=recovery["campaign_id"],
            )
    elif args.s1_profile_recovery_certificate:
        raise ValueError(
            "--s1-profile-recovery-certificate is valid only for formal S1 configs"
        )
    if "georoute_official_development_binding" in cfg:
        if s1_binding is not None:
            raise RuntimeError(
                "formal GeoRoute development test cannot share an S1 binding"
            )
        if (
            args.checkpoint == "none"
            or args.cfg_options is not None
            or args.id != 0
            or args.not_eval
            or args.max_batches is not None
        ):
            raise ValueError(
                "formal GeoRoute development evaluation requires an explicit "
                "checkpoint and forbids overrides, alternate ids, partial "
                "inference, and no-eval mode"
            )
        from tools.bata.georoute_official_comparable_contract import (
            require_clean_formal_checkout,
            require_formal_world2_slurm,
            validate_formal_checkpoint_sidecar,
            validate_formal_development_config,
        )

        require_formal_world2_slurm()
        georoute_official_development_binding = validate_formal_development_config(
            cfg, seed=args.seed
        )
        require_clean_formal_checkout(
            expected_commit=georoute_official_development_binding["runtime_commit"],
            root=Path(path).resolve(),
        )
        georoute_official_checkpoint_sidecar = validate_formal_checkpoint_sidecar(
            args.checkpoint,
            binding=georoute_official_development_binding,
        )
    if "georoute_dynamic_floor_m2_binding" in cfg:
        if s1_binding is not None or georoute_official_development_binding is not None:
            raise RuntimeError(
                "dynamic floor M2 evaluation cannot share another formal binding"
            )
        from tools.bata.georoute_dynamic_floor_m2_contract import (
            DYNAMIC_FLOOR_M2_RESIDUAL_CENTERING_PROBE_SCHEMA,
            DYNAMIC_FLOOR_M2_ROLE_STRICT_TRIPLET_SCHEMA,
            require_clean_dynamic_floor_m2_checkout,
            require_dynamic_floor_m2_world1_slurm,
            resolve_dynamic_floor_m2_accuracy_execution_commit,
            validate_dynamic_floor_m2_checkpoint_sidecar,
            validate_dynamic_floor_m2_config,
        )

        phase_m_binding = cfg.get("georoute_phase_m_binding", {})
        residual_centering_probe = bool(
            isinstance(phase_m_binding, dict)
            and phase_m_binding.get("schema_version")
            == DYNAMIC_FLOOR_M2_RESIDUAL_CENTERING_PROBE_SCHEMA
        )
        if (
            args.checkpoint == "none"
            or args.cfg_options is not None
            or args.id != 0
            or bool(args.not_eval) is not residual_centering_probe
            or args.max_batches is not None
        ):
            raise ValueError(
                "dynamic floor M2 evaluation requires its final checkpoint and "
                "forbids overrides, alternate ids, and partial inference; only the "
                "bound residual-centering mechanism probe requires no-eval"
            )

        require_dynamic_floor_m2_world1_slurm()
        arm = str(cfg.georoute_dynamic_floor_m2_binding.arm)
        georoute_dynamic_floor_m2_binding = validate_dynamic_floor_m2_config(
            cfg, arm=arm, phase="accuracy"
        )
        dynamic_floor_m2_execution_commit = (
            resolve_dynamic_floor_m2_accuracy_execution_commit(
                cfg,
                binding=georoute_dynamic_floor_m2_binding,
            )
        )
        strict_dynamic_floor_m2_determinism = bool(
            isinstance(phase_m_binding, dict)
            and phase_m_binding.get("schema_version")
            in {
                DYNAMIC_FLOOR_M2_ROLE_STRICT_TRIPLET_SCHEMA,
                DYNAMIC_FLOOR_M2_RESIDUAL_CENTERING_PROBE_SCHEMA,
            }
        )
        require_clean_dynamic_floor_m2_checkout(
            expected_commit=dynamic_floor_m2_execution_commit,
            root=Path(path).resolve(),
        )
        georoute_dynamic_floor_m2_checkpoint_sidecar = (
            validate_dynamic_floor_m2_checkpoint_sidecar(
                args.checkpoint,
                binding=georoute_dynamic_floor_m2_binding,
            )
        )
    assert_detector_training_allowed(cfg, entrypoint="tools/test.py")
    assert_no_raw_prediction_shortcut_for_pc_ot_mras(cfg)

    # DDP init
    if strict_dynamic_floor_m2_determinism:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    if s1_binding is not None and args.world_size != 1:
        raise RuntimeError("formal S1 test is frozen to one Slurm GPU process")
    if georoute_official_development_binding is not None and args.world_size != int(
        georoute_official_development_binding["world_size"]
    ):
        raise RuntimeError("formal GeoRoute development evaluation world size changed")
    if georoute_dynamic_floor_m2_binding is not None and args.world_size != 1:
        raise RuntimeError("dynamic floor M2 evaluation world size changed")
    print(
        f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})"
    )
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # set random seed, create work_dir
    set_seed(
        args.seed,
        # The pinned AdaTAD recipe uses the repository's legacy
        # deterministic-warn-only semantics. GeoRoute development keeps that
        # transition behavior except for the explicitly diagnostic strict-math
        # triplet; the separately sealed S1 protocol is also strict.
        deterministic_warn_only=(
            s1_binding is None and not strict_dynamic_floor_m2_determinism
        ),
    )
    if strict_dynamic_floor_m2_determinism:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print(
            "Dynamic floor M2 strict determinism diagnostic: "
            "math SDPA only, TF32 disabled"
        )
    if georoute_dynamic_floor_m2_binding is None:
        cfg = update_workdir(cfg, args.id, torch.cuda.device_count())
    elif os.path.exists(cfg.work_dir):
        raise FileExistsError(
            "dynamic floor M2 accuracy replay requires a fresh bound work_dir"
        )
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
            if recovery is not None:
                marker.update(
                    formal_test_runtime_mode=S1_STEP_SCOPED_TEST_RUNTIME_MODE,
                    test_runtime_code_commit=recovery["profile_code_commit"],
                    profile_recovery_certificate_path=recovery_path,
                    profile_recovery_certificate_file_sha256=(
                        sha256_file(recovery_path)
                    ),
                    profile_recovery_certificate_sha256=(
                        recovery["certificate_sha256"]
                    ),
                    profile_recovery_campaign_id=recovery["campaign_id"],
                )
            marker["marker_sha256"] = canonical_sha256(marker)
            atomic_publish_json(marker_path, marker)
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
    if georoute_official_development_binding is not None:
        runtime_test_ids = {str(row[0]) for row in test_dataset.data_list}
        if runtime_test_ids != set(
            georoute_official_development_binding["evaluation_video_ids"]
        ):
            raise ValueError(
                "formal GeoRoute evaluation left the frozen development Gate"
            )
    if georoute_dynamic_floor_m2_binding is not None:
        runtime_test_ids = {str(row[0]) for row in test_dataset.data_list}
        if runtime_test_ids != set(
            georoute_dynamic_floor_m2_binding["evaluation_video_ids"]
        ):
            raise ValueError("dynamic floor M2 evaluation left development Gate")
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
        if georoute_official_development_binding is not None:
            if (
                checkpoint.get("experiment_metadata")
                != georoute_official_checkpoint_sidecar["experiment_metadata"]
            ):
                raise ValueError(
                    "formal GeoRoute checkpoint payload metadata does not "
                    "match its sidecar"
                )
            if int(checkpoint.get("epoch", -1)) != int(
                georoute_official_checkpoint_sidecar["experiment_metadata"]["epoch"]
            ):
                raise ValueError(
                    "formal GeoRoute checkpoint epoch differs from its sidecar"
                )
        if georoute_dynamic_floor_m2_binding is not None:
            dynamic_metadata = georoute_dynamic_floor_m2_checkpoint_sidecar[
                "experiment_metadata"
            ]
            if checkpoint.get("experiment_metadata") != dynamic_metadata:
                raise ValueError(
                    "dynamic floor M2 checkpoint payload metadata differs from sidecar"
                )
            if int(checkpoint.get("epoch", -1)) != int(dynamic_metadata["epoch"]):
                raise ValueError("dynamic floor M2 checkpoint epoch mismatch")

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
