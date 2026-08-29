import os
import json
import hashlib
import subprocess
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.models.utils.pc_ot_mras_raw_prediction_guard import assert_no_raw_prediction_shortcut_for_pc_ot_mras
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import eval_one_epoch
from opentad.utils import update_workdir, set_seed, create_folder, setup_logger
from opentad.utils.training_guard import (
    assert_detector_training_allowed,
    assert_safe_cfg_options_for_gated_config,
)
from tools.bata.duca_p0_training import atomic_write_json, sha256_file
from tools.bata import duca_cellcf_training
from tools.bata import duca_protected_physical_training
from tools.bata import duca_selected_axis_training
from tools.bata.duca_p0_evaluation import (
    canonical_jsonable,
    evaluation_config_sha256,
    normalize_evaluation_config,
    official_evaluator_identity,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--checkpoint", type=str, default="none", help="the checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--not_eval", action="store_true", help="whether to not to eval, only do inference")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--expected-checkpoint-epoch", type=int, default=None)
    parser.add_argument(
        "--checkpoint-state-key",
        choices=("auto", "state_dict", "state_dict_ema"),
        default="auto",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    formal_protocol = str(cfg.workflow.get("formal_protocol", ""))
    cellcf_formal = formal_protocol == "duca_cellcf_v1"
    protected_physical_formal = (
        formal_protocol == "duca_protected_physical_v1"
    )
    selected_axis_formal = duca_selected_axis_training.is_formal_protocol(
        formal_protocol
    )
    r5_formal = formal_protocol == duca_selected_axis_training.R5_FORMAL_PROTOCOL
    source_resolved_config_sha256 = _canonical_sha256(cfg.to_dict())
    if cellcf_formal:
        duca_cellcf_training.assert_safe_cfg_options(
            cfg, args.cfg_options, entrypoint="tools/test.py"
        )
    elif protected_physical_formal:
        duca_protected_physical_training.assert_safe_cfg_options(
            args.cfg_options,
            entrypoint="tools/test.py",
        )
    elif selected_axis_formal:
        duca_selected_axis_training.assert_safe_cfg_options(
            args.cfg_options,
            entrypoint="tools/test.py",
        )
    assert_safe_cfg_options_for_gated_config(cfg, args.cfg_options, entrypoint="tools/test.py")
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    assert_detector_training_allowed(cfg, entrypoint="tools/test.py")
    assert_no_raw_prediction_shortcut_for_pc_ot_mras(cfg)

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    if cellcf_formal or protected_physical_formal or selected_axis_formal:
        expected_commit = os.environ.get("DUCA_EXPECTED_COMMIT")
        observed_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, encoding="utf-8"
        ).strip()
        if expected_commit != observed_commit:
            raise RuntimeError("formal CellCF evaluation checkout differs from DUCA_EXPECTED_COMMIT")
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=path,
            text=True,
            encoding="utf-8",
        ).strip()
        if status:
            raise RuntimeError(
                "formal evaluation requires a clean exact-commit checkout"
            )
        if args.world_size != 1 or args.not_eval:
            raise RuntimeError(
                "formal evaluation requires one process and official mAP evaluation"
            )
    if protected_physical_formal:
        if source_resolved_config_sha256 != os.environ.get(
            "DUCA_RESOLVED_CONFIG_SHA256"
        ):
            raise RuntimeError(
                "formal protected evaluation source config differs from P0"
            )
        if (
            args.seed != 3407
            or args.expected_checkpoint_epoch != 59
            or args.checkpoint_state_key != "state_dict_ema"
            or not args.metrics_json
        ):
            raise RuntimeError(
                "formal protected evaluation must use seed 3407, "
                "terminal epoch-59 EMA, and structured metrics"
            )
    if selected_axis_formal:
        expected_seed = int(cfg.r5_cell.seed) if r5_formal else 3407
        if (
            args.seed != expected_seed
            or args.expected_checkpoint_epoch != 59
            or args.checkpoint_state_key != "state_dict_ema"
            or not args.metrics_json
        ):
            raise RuntimeError(
                f"formal selected-axis evaluation must use seed {expected_seed}, "
                "terminal epoch-59 EMA, and structured metrics"
            )
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # set random seed, create work_dir
    set_seed(args.seed)
    cfg = update_workdir(cfg, args.id, torch.cuda.device_count())
    runtime_config_sha256 = _canonical_sha256(cfg.to_dict())
    if cellcf_formal:
        expected_runtime = os.environ.get("DUCA_CELLCF_EVAL_RUNTIME_CONFIG_SHA256")
        if expected_runtime != runtime_config_sha256:
            raise RuntimeError("formal CellCF evaluation effective config differs from the frozen launch")
    if args.rank == 0:
        create_folder(cfg.work_dir)

    # setup logger
    logger = setup_logger("Test", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
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
    model = model.to(args.local_rank)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    logger.info(f"Using DDP with total {args.world_size} GPUS...")

    checkpoint_path = None
    checkpoint_epoch = None
    checkpoint_state_key = None
    selected_axis_terminal_identity = None
    if cfg.inference.load_from_raw_predictions:  # if load with saved predictions, no need to load checkpoint
        logger.info(f"Loading from raw predictions: {cfg.inference.fuse_list}")
    else:  # load checkpoint: args -> config -> best
        if args.checkpoint != "none":
            checkpoint_path = args.checkpoint
        elif "test_epoch" in cfg.inference.keys():
            checkpoint_path = os.path.join(cfg.work_dir, f"checkpoint/epoch_{cfg.inference.test_epoch}.pth")
        else:
            checkpoint_path = os.path.join(cfg.work_dir, "checkpoint/best.pth")
        logger.info("Loading checkpoint from: {}".format(checkpoint_path))
        device = f"cuda:{args.rank % torch.cuda.device_count()}"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        checkpoint_epoch = int(checkpoint["epoch"])
        logger.info("Checkpoint is epoch {}.".format(checkpoint_epoch))
        if (
            args.expected_checkpoint_epoch is not None
            and checkpoint_epoch != args.expected_checkpoint_epoch
        ):
            raise RuntimeError(
                f"checkpoint epoch {checkpoint_epoch} differs from expected "
                f"{args.expected_checkpoint_epoch}"
            )

        # Model EMA
        use_ema = getattr(cfg.solver, "ema", False)
        checkpoint_state_key = args.checkpoint_state_key
        if checkpoint_state_key == "auto":
            checkpoint_state_key = "state_dict_ema" if use_ema else "state_dict"
        if checkpoint_state_key not in checkpoint:
            raise RuntimeError(
                f"checkpoint does not contain requested state {checkpoint_state_key}"
            )
        if selected_axis_formal:
            selected_axis_terminal_identity = (
                duca_selected_axis_training.validate_terminal_checkpoint_binding(
                    checkpoint_path=checkpoint_path,
                    checkpoint=checkpoint,
                    git_commit=os.environ["DUCA_EXPECTED_COMMIT"],
                    variant=os.environ.get("DUCA_SELECTED_OPT_VARIANT", ""),
                    seed=args.seed,
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                    source_config_path=args.config,
                    source_config_sha256=sha256_file(args.config),
                    resolved_config_sha256=source_resolved_config_sha256,
                    checkpoint_epoch=checkpoint_epoch,
                    checkpoint_state_key=checkpoint_state_key,
                    evaluation_annotation_path=cfg.evaluation.ground_truth_filename,
                    evaluation_class_map_path=cfg.dataset.test.class_map,
                    evaluation_config=cfg.evaluation,
                    runtime_pretrain_path=cfg.model.backbone.custom.pretrain,
                    frozen_pretrain_path=os.environ.get(
                        "DUCA_ADATAD_PRETRAIN_PATH", ""
                    ),
                    frozen_pretrain_sha256=os.environ.get(
                        "DUCA_ADATAD_PRETRAIN_SHA256", ""
                    ),
                    selector_initialization=cfg.workflow.get(
                        "selector_initialization", None
                    ),
                    formal_protocol=formal_protocol,
                    r5_cell=cfg.get("r5_cell", None),
                )
            )
        model.load_state_dict(checkpoint[checkpoint_state_key])
        if checkpoint_state_key == "state_dict_ema":
            logger.info("Using Model EMA...")

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")

    # test the detector
    logger.info("Testing Starts...\n")
    evaluation_summary = eval_one_epoch(
        test_loader,
        model,
        cfg,
        logger,
        args.rank,
        model_ema=None,  # since we have loaded the ema model above
        use_amp=use_amp,
        world_size=args.world_size,
        not_eval=args.not_eval,
    )
    if args.rank == 0 and args.metrics_json:
        if checkpoint_path is None or checkpoint_state_key is None:
            raise RuntimeError("structured metric evidence requires a checkpoint")
        if not isinstance(evaluation_summary, dict):
            raise RuntimeError("evaluation did not return a structured summary")
        result_path = evaluation_summary.get("result_path")
        if not result_path or not os.path.isfile(result_path):
            raise RuntimeError("structured metric evidence requires saved predictions")
        formal_protocol = str(cfg.workflow.get("formal_protocol", ""))
        r0_selected_axis_replay = (
            formal_protocol == "duca_r0_selected_axis_holdout_replay_v1"
        )
        expected_evaluation_subset = (
            "training" if r0_selected_axis_replay else "validation"
        )
        evaluation_config = normalize_evaluation_config(
            cfg.evaluation,
            expected_subset=expected_evaluation_subset,
        )
        evaluator_identity = official_evaluator_identity()
        if evaluation_summary.get("evaluator") != evaluator_identity:
            raise RuntimeError("runtime evaluator differs from the frozen OpenTAD mAP evaluator")
        if formal_protocol == "duca_cellcf_v1":
            evaluation_schema = "duca_cellcf_terminal_evaluation_v1"
        elif formal_protocol == "duca_protected_physical_v1":
            evaluation_schema = (
                "duca_protected_physical_terminal_evaluation_v1"
            )
        elif r5_formal:
            evaluation_schema = "duca_r5_terminal_evaluation_v1"
        elif formal_protocol == duca_selected_axis_training.FORMAL_PROTOCOL:
            evaluation_schema = "duca_selected_axis_terminal_evaluation_v1"
        elif r0_selected_axis_replay:
            evaluation_schema = "duca_r0_selected_axis_evaluation_v1"
        else:
            evaluation_schema = "duca_p0_terminal_evaluation_v3"
        payload = {
            "schema_version": evaluation_schema,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=path, text=True
            ).strip(),
            "task": "offline_temporal_action_detection",
            "config_path": os.path.abspath(args.config),
            "config_sha256": sha256_file(args.config),
            "resolved_config_sha256": source_resolved_config_sha256,
            "runtime_config_sha256": runtime_config_sha256,
            "checkpoint_path": os.path.abspath(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_state_key": checkpoint_state_key,
            "prediction_path": os.path.abspath(result_path),
            "prediction_sha256": sha256_file(result_path),
            "metrics": _jsonable(evaluation_summary.get("metrics")),
            "result_count": int(evaluation_summary.get("result_count", 0)),
            "video_count": int(evaluation_summary.get("video_count", 0)),
            "evaluator": evaluator_identity,
            "evaluation_config": evaluation_config,
            "evaluation_config_sha256": evaluation_config_sha256(
                evaluation_config,
                expected_subset=expected_evaluation_subset,
            ),
            "evaluation_annotation_path": os.path.abspath(
                os.path.expanduser(str(cfg.evaluation.ground_truth_filename))
            ),
            "evaluation_annotation_sha256": sha256_file(
                cfg.evaluation.ground_truth_filename
            ),
            "evaluation_class_map_path": os.path.abspath(
                os.path.expanduser(str(cfg.dataset.test.class_map))
            ),
            "evaluation_class_map_sha256": sha256_file(
                cfg.dataset.test.class_map
            ),
        }
        evidence_model = model.module if hasattr(model, "module") else model
        if hasattr(evidence_model, "native_tubelet_compute_evidence"):
            compute_evidence = evidence_model.native_tubelet_compute_evidence()
            if compute_evidence is not None:
                payload["native_tubelet_compute"] = _jsonable(compute_evidence)
        if selected_axis_formal:
            payload.update(
                {
                    "seed": int(args.seed),
                    "variant": os.environ.get("DUCA_SELECTED_OPT_VARIANT"),
                    "training_identity": selected_axis_terminal_identity,
                }
            )
            if r5_formal:
                payload["r5_cell"] = _jsonable(cfg.r5_cell)
        if r0_selected_axis_replay:
            allocation_artifact = os.path.abspath(
                os.path.expanduser(
                    os.environ.get("DUCA_ALLOCATION_ARTIFACT_PATH", "")
                )
            )
            allocation_sha256 = os.environ.get(
                "DUCA_ALLOCATION_ARTIFACT_SHA256", ""
            )
            family_key = os.environ.get("DUCA_ALLOCATION_FAMILY_KEY", "")
            blocked_path = evaluation_config.get("blocked_videos")
            if (
                not family_key
                or not os.path.isfile(allocation_artifact)
                or sha256_file(allocation_artifact) != allocation_sha256
                or not blocked_path
                or not os.path.isfile(blocked_path)
            ):
                raise RuntimeError("R0 selected-axis replay identity is incomplete")
            payload.update(
                {
                    "seed": int(args.seed),
                    "family": family_key,
                    "allocation_artifact_path": allocation_artifact,
                    "allocation_artifact_sha256": allocation_sha256,
                    "evaluation_blocked_videos_path": os.path.abspath(blocked_path),
                    "evaluation_blocked_videos_sha256": sha256_file(blocked_path),
                    "source_subset": "training_internal_holdout",
                    "test_subset_consumed": False,
                    "runtime_gt_input_to_selector": False,
                }
            )
        payload["evaluation_sha256"] = _canonical_sha256(payload)
        atomic_write_json(args.metrics_json, payload)
    logger.info("Testing Over...\n")


def _jsonable(value):
    return canonical_jsonable(value)


def _canonical_sha256(value):
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    main()
