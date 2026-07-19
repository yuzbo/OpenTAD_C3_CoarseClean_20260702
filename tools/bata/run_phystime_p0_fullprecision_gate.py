#!/usr/bin/env python3
"""Fail-closed gate for the PhysTime P0 frozen full-precision replay."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from mmengine.config import Config

SOURCE_COMMIT = "0dc5851a8feb12b97d16bdb5ea8fc60e9273d132"
SOURCE_TREE = "bddc9b9386604d00d213275a47ce7997b35d3f4c"
EXPECTED_EPOCH = 59
ARM_CONFIGS = {
    "selected_axis": (
        "configs/adatad/thumos/"
        "phystime_g1a_selected_axis_native_j192_p0_replay.py"
    ),
    "physical_metric": (
        "configs/adatad/thumos/"
        "phystime_g1a_physical_metric_native_j192_p0_replay.py"
    ),
}
EXPECTED_COORDINATE_MODES = {
    "selected_axis": "uniform_rank_seconds",
    "physical_metric": "physical_time_seconds",
}
EXPECTED_HEAD_TIME_CONTRACT = {
    "enabled": True,
    "required": True,
    "strict": True,
    "positions_key": "phystime_g1a_axis_positions_sec",
    "selected_count_keys": ["phystime_native_valid_count"],
    "axis_start_key": "phystime_g1a_axis_start_sec",
    "axis_end_key": "phystime_g1a_axis_end_sec",
}
P0_POST_PROCESSING_KEYS = {
    "filter_invalid_proposals",
    "proposal_min_duration",
    "round_before_cross_window_nms",
    "round_after_cross_window_nms",
    "segment_round_digits",
    "score_round_digits",
    "save_pre_cross_window_detections",
    "save_post_processing_audit",
    "pre_cross_window_detections_path",
    "post_processing_audit_path",
    "save_dict",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate source checkpoints and runtime contracts for P0 replay."
    )
    parser.add_argument("--selected-source-dir", required=True)
    parser.add_argument("--physical-source-dir", required=True)
    parser.add_argument("--selected-checkpoint", required=True)
    parser.add_argument("--physical-checkpoint", required=True)
    parser.add_argument("--videomae-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-runtime-commit", required=True)
    parser.add_argument("--expected-runtime-tree", required=True)
    parser.add_argument("--focused-tests-log", required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: value.item(),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _directory_inventory(path):
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError(f"dataset directory not found: {path}")
    files = []
    leaves = []
    total_bytes = 0
    items = sorted(
        (value for value in path.rglob("*") if value.is_file()),
        key=lambda value: value.as_posix(),
    )
    for item in items:
        record = {
            "relative_path": item.relative_to(path).as_posix(),
            "size_bytes": int(item.stat().st_size),
            "sha256": sha256_file(item),
        }
        files.append(record)
        leaves.append(
            hashlib.sha256(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).digest()
        )
        total_bytes += record["size_bytes"]
    if not files:
        raise ValueError(f"dataset directory contains no files: {path}")
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])
        leaves = [
            hashlib.sha256(leaves[index] + leaves[index + 1]).digest()
            for index in range(0, len(leaves), 2)
        ]
    return {
        "path": str(path),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
        "inventory_sha256": leaves[0].hex(),
        "hash_scope": "full_file_content_sha256_merkle_v1",
    }


def build_dataset_manifest(cfg, evaluation_ground_truth):
    class_map_path = Path(cfg.dataset.train.class_map).resolve()
    manifest = {
        "annotation": str(Path(evaluation_ground_truth).resolve()),
        "annotation_sha256": sha256_file(evaluation_ground_truth),
        "class_map": str(class_map_path),
        "class_map_sha256": sha256_file(class_map_path),
        "train_videos": _directory_inventory(cfg.dataset.train.data_path),
        "test_videos": _directory_inventory(cfg.dataset.test.data_path),
    }
    return manifest, canonical_sha256(manifest)


def read_json(path, description):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{description} is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def current_git_identity():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()
    return commit, tree


def inference_semantic_payload(cfg):
    post_processing = dict(cfg.post_processing)
    for key in P0_POST_PROCESSING_KEYS:
        post_processing.pop(key, None)
    return {
        "model": cfg.model.to_dict(),
        "dataset_test": cfg.dataset.test.to_dict(),
        "evaluation": cfg.evaluation.to_dict(),
        "inference": cfg.inference.to_dict(),
        "solver_test": cfg.solver.test.to_dict(),
        "post_processing_shared": post_processing,
    }


def coordinate_modes_from_config(cfg):
    pipeline_modes = {}
    for split in ("train", "val", "test"):
        modes = [
            step["coordinate_mode"]
            for step in cfg.dataset[split].pipeline
            if step["type"] == "BuildPhysTimeNativeTubeletGeometry"
        ]
        if len(modes) != 1:
            raise ValueError(
                f"{split} pipeline must contain exactly one native geometry builder"
            )
        pipeline_modes[split] = modes[0]
    head = cfg.model.rpn_head.physical_grid_actionformer
    head_contract = {
        "enabled": bool(head.enabled),
        "required": bool(head.required),
        "strict": bool(head.strict),
        "positions_key": str(head.positions_key),
        "selected_count_keys": list(head.selected_count_keys),
        "axis_start_key": str(head.axis_start_key),
        "axis_end_key": str(head.axis_end_key),
    }
    return {
        "pipelines": pipeline_modes,
        "head_time_contract": head_contract,
    }


def load_bound_source_config(source_config_path, manifest):
    source_config_path = Path(source_config_path).resolve()
    source_cfg = Config.fromfile(source_config_path, lazy_import=False)
    canonical_hash = canonical_sha256(source_cfg.to_dict())
    if canonical_hash != manifest.get("config_sha256"):
        raise ValueError(
            "source config canonical hash differs from the full60 manifest"
        )
    return source_cfg, {
        "path": str(source_config_path),
        "canonical_sha256": canonical_hash,
        "file_sha256": sha256_file(source_config_path),
    }


def validate_runtime_config(arm, config_path):
    cfg = Config.fromfile(config_path, lazy_import=False)
    post_cfg = cfg.post_processing
    expected = {
        "filter_invalid_proposals": True,
        "proposal_min_duration": 1.0e-6,
        "round_before_cross_window_nms": False,
        "round_after_cross_window_nms": False,
        "segment_round_digits": 2,
        "score_round_digits": 4,
        "save_pre_cross_window_detections": True,
        "save_post_processing_audit": True,
        "save_dict": True,
    }
    observed = {key: post_cfg.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"{arm} P0 post-processing contract mismatch: "
            f"expected={expected}, observed={observed}"
        )
    if int(cfg.raw_observation_count) != 384:
        raise ValueError(f"{arm} P0 config does not use K=384")
    native_geometry = cfg.model.native_temporal_geometry
    if int(native_geometry.expected_token_count) != 192:
        raise ValueError(f"{arm} P0 config does not use J=192")
    post_types = [
        step["type"] for step in cfg.model.backbone.custom.post_processing_pipeline
    ]
    if "Interpolate" in post_types:
        raise ValueError(f"{arm} P0 config unexpectedly enables interpolation")
    expected_coordinate_mode = EXPECTED_COORDINATE_MODES[arm]
    coordinate_modes = coordinate_modes_from_config(cfg)
    observed_coordinate_modes = set(coordinate_modes["pipelines"].values())
    if observed_coordinate_modes != {expected_coordinate_mode}:
        raise ValueError(
            f"{arm} coordinate-mode contract mismatch: "
            f"expected={expected_coordinate_mode}, observed={coordinate_modes}"
        )
    if coordinate_modes["head_time_contract"] != EXPECTED_HEAD_TIME_CONTRACT:
        raise ValueError(
            f"{arm} detector time-position contract mismatch: "
            f"expected={EXPECTED_HEAD_TIME_CONTRACT}, "
            f"observed={coordinate_modes['head_time_contract']}"
        )
    semantic_payload = inference_semantic_payload(cfg)
    return {
        "path": str(Path(config_path).resolve()),
        "canonical_sha256": canonical_sha256(cfg.to_dict()),
        "inference_semantic_sha256": canonical_sha256(semantic_payload),
        "K_raw_observations": 384,
        "J_native_tokens": 192,
        "feature_interpolation": False,
        "coordinate_modes": coordinate_modes,
        "evaluation_contract": {
            "ground_truth_filename": str(
                Path(cfg.evaluation.ground_truth_filename).resolve()
            ),
            "subset": str(cfg.evaluation.subset),
            "tiou_thresholds": [
                float(value) for value in cfg.evaluation.tiou_thresholds
            ],
            "blocked_videos": (
                None
                if cfg.evaluation.get("blocked_videos") is None
                else str(
                    Path(cfg.evaluation.blocked_videos).resolve()
                )
            ),
        },
        "post_processing_contract": expected,
    }


def validate_source_arm(
    arm,
    source_dir,
    checkpoint_path,
    *,
    videomae_checkpoint_sha256,
    runtime_inference_semantic_sha256,
):
    source_dir = Path(source_dir).resolve()
    completion_path = source_dir / "FULL_COMPLETE.json"
    manifest_path = source_dir / "run_manifest.json"
    completion = read_json(completion_path, f"{arm} full60 completion")
    manifest = read_json(manifest_path, f"{arm} full60 manifest")
    if completion.get("validation_pass") is not True:
        raise ValueError(f"{arm} source completion did not pass")
    if completion.get("variant") != arm or manifest.get("variant") != arm:
        raise ValueError(f"{arm} source variant mismatch")
    if int(completion.get("evaluation_epoch", -1)) != EXPECTED_EPOCH:
        raise ValueError(f"{arm} source completion is not epoch {EXPECTED_EPOCH}")
    if manifest.get("commit") != SOURCE_COMMIT:
        raise ValueError(f"{arm} source commit mismatch")
    if manifest.get("git_tree") != SOURCE_TREE:
        raise ValueError(f"{arm} source tree mismatch")
    if (
        manifest.get("pretrained_checkpoint_sha256")
        != videomae_checkpoint_sha256
    ):
        raise ValueError(f"{arm} VideoMAE checkpoint hash mismatch")
    source_config_path = Path(manifest["config"]).resolve()
    source_cfg, source_config_binding = load_bound_source_config(
        source_config_path,
        manifest,
    )
    source_coordinate_modes = coordinate_modes_from_config(source_cfg)
    expected_coordinate_mode = EXPECTED_COORDINATE_MODES[arm]
    if set(source_coordinate_modes["pipelines"].values()) != {
        expected_coordinate_mode
    }:
        raise ValueError(f"{arm} source coordinate-mode contract mismatch")
    if (
        source_coordinate_modes["head_time_contract"]
        != EXPECTED_HEAD_TIME_CONTRACT
    ):
        raise ValueError(f"{arm} source detector time-position contract mismatch")
    source_inference_semantic_sha256 = canonical_sha256(
        inference_semantic_payload(source_cfg)
    )
    if source_inference_semantic_sha256 != runtime_inference_semantic_sha256:
        raise ValueError(
            f"{arm} runtime/source inference semantics differ after removing "
            "the pre-registered P0 artifact-policy fields"
        )

    checkpoint_path = Path(checkpoint_path).resolve()
    recorded_checkpoint = Path(
        completion["artifacts"]["checkpoint"]["path"]
    ).resolve()
    if checkpoint_path != recorded_checkpoint:
        raise ValueError(f"{arm} checkpoint path differs from FULL_COMPLETE")
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != completion["artifacts"]["checkpoint"]["sha256"]:
        raise ValueError(f"{arm} checkpoint hash differs from FULL_COMPLETE")
    bound_evaluation_artifacts = {}
    for artifact_name in ("predictions", "metrics"):
        recorded = completion["artifacts"][artifact_name]
        artifact_path = Path(recorded["path"]).resolve()
        artifact_sha = sha256_file(artifact_path)
        if artifact_sha != recorded["sha256"]:
            raise ValueError(
                f"{arm} source {artifact_name} hash differs from FULL_COMPLETE"
            )
        bound_evaluation_artifacts[artifact_name] = {
            "path": str(artifact_path),
            "sha256": artifact_sha,
        }

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if int(checkpoint.get("epoch", -1)) != EXPECTED_EPOCH:
        raise ValueError(f"{arm} checkpoint epoch mismatch")
    for key in ("state_dict", "state_dict_ema"):
        state = checkpoint.get(key)
        if not isinstance(state, dict) or not state:
            raise ValueError(f"{arm} checkpoint is missing non-empty {key}")
    forbidden = {"optimizer", "scheduler", "optimizer_state_dict"}
    forbidden_present = sorted(forbidden.intersection(checkpoint))
    if forbidden_present:
        raise ValueError(
            f"{arm} checkpoint contains forbidden training state: "
            f"{forbidden_present}"
        )
    metrics = {
        key: float(value)
        for key, value in completion.get("metrics", {}).items()
    }
    if not metrics or not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"{arm} source metrics are missing or non-finite")
    return {
        "source_dir": str(source_dir),
        "completion": str(completion_path),
        "completion_sha256": sha256_file(completion_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_config": str(source_config_path),
        "source_config_canonical_sha256": source_config_binding[
            "canonical_sha256"
        ],
        "source_config_file_sha256": source_config_binding["file_sha256"],
        "source_inference_semantic_sha256": (
            source_inference_semantic_sha256
        ),
        "source_coordinate_modes": source_coordinate_modes,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": EXPECTED_EPOCH,
        "checkpoint_keys": sorted(checkpoint),
        "online_parameter_tensors": len(checkpoint["state_dict"]),
        "ema_parameter_tensors": len(checkpoint["state_dict_ema"]),
        "metrics": metrics,
        "evaluation_artifacts": bound_evaluation_artifacts,
        "dataset_manifest_sha256": manifest.get("dataset_manifest_sha256"),
        "videomae_checkpoint_sha256": videomae_checkpoint_sha256,
    }


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main():
    import torch

    args = parse_args()
    runtime_commit, runtime_tree = current_git_identity()
    if runtime_commit != args.expected_runtime_commit:
        raise SystemExit("P0 gate runtime commit mismatch")
    if runtime_tree != args.expected_runtime_tree:
        raise SystemExit("P0 gate runtime tree mismatch")
    if subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True,
    ).strip():
        raise SystemExit("P0 gate runtime snapshot is dirty")
    if not torch.cuda.is_available():
        raise SystemExit("P0 gate does not have an available Slurm CUDA device")
    environment_init_mode = os.environ.get("PHYSTIME_ENV_INIT_MODE")
    if environment_init_mode not in {
        "module_cuda11.8_miniforge3_24.11",
        "fixed_conda_path_no_module_command",
    }:
        raise SystemExit("P0 gate environment initialization mode is unbound")

    focused_tests_log = Path(args.focused_tests_log).resolve()
    if not focused_tests_log.is_file():
        raise SystemExit("focused test log is missing")
    focused_tests_text = focused_tests_log.read_text(
        encoding="utf-8",
        errors="replace",
    )
    if "passed" not in focused_tests_text:
        raise SystemExit("focused test log does not contain a passing pytest summary")

    runtime_root = Path.cwd().resolve()
    config_reports = {
        arm: validate_runtime_config(arm, runtime_root / relative_path)
        for arm, relative_path in ARM_CONFIGS.items()
    }
    runtime_cfgs = {
        arm: Config.fromfile(runtime_root / relative_path, lazy_import=False)
        for arm, relative_path in ARM_CONFIGS.items()
    }
    dataset_bindings = {
        arm: {
            "annotation": str(
                Path(cfg.evaluation.ground_truth_filename).resolve()
            ),
            "class_map": str(Path(cfg.dataset.train.class_map).resolve()),
            "train_videos": str(Path(cfg.dataset.train.data_path).resolve()),
            "test_videos": str(Path(cfg.dataset.test.data_path).resolve()),
        }
        for arm, cfg in runtime_cfgs.items()
    }
    if dataset_bindings["selected_axis"] != dataset_bindings["physical_metric"]:
        raise SystemExit("selected-axis and physical-metric dataset paths differ")
    dataset_manifest, dataset_manifest_sha256 = build_dataset_manifest(
        runtime_cfgs["selected_axis"],
        runtime_cfgs["selected_axis"].evaluation.ground_truth_filename,
    )
    for arm in ARM_CONFIGS:
        config_reports[arm]["dataset_bindings"] = dataset_bindings[arm]
        config_reports[arm][
            "dataset_manifest_sha256"
        ] = dataset_manifest_sha256
    videomae_checkpoint = Path(args.videomae_checkpoint).resolve()
    if not videomae_checkpoint.is_file():
        raise SystemExit("VideoMAE checkpoint is missing")
    videomae_checkpoint_sha256 = sha256_file(videomae_checkpoint)
    source_reports = {
        "selected_axis": validate_source_arm(
            "selected_axis",
            args.selected_source_dir,
            args.selected_checkpoint,
            videomae_checkpoint_sha256=videomae_checkpoint_sha256,
            runtime_inference_semantic_sha256=config_reports[
                "selected_axis"
            ]["inference_semantic_sha256"],
        ),
        "physical_metric": validate_source_arm(
            "physical_metric",
            args.physical_source_dir,
            args.physical_checkpoint,
            videomae_checkpoint_sha256=videomae_checkpoint_sha256,
            runtime_inference_semantic_sha256=config_reports[
                "physical_metric"
            ]["inference_semantic_sha256"],
        ),
    }
    for arm in ARM_CONFIGS:
        if (
            dataset_manifest_sha256
            != source_reports[arm]["dataset_manifest_sha256"]
        ):
            raise SystemExit(
                f"{arm} P0 dataset manifest differs from the source full60 run"
            )
    report = {
        "schema_version": "phystime_p0_fullprecision_gate_v1",
        "gate_pass": True,
        "completed_at_unix": time.time(),
        "runtime": {
            "root": str(runtime_root),
            "commit": runtime_commit,
            "git_tree": runtime_tree,
            "clean": True,
            "videomae_checkpoint": str(videomae_checkpoint),
            "videomae_checkpoint_sha256": videomae_checkpoint_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "dataset_manifest": dataset_manifest,
            "environment": {
                "python": sys.version,
                "python_executable": sys.executable,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES"
                ),
                "loaded_modules": os.environ.get("LOADEDMODULES", ""),
                "init_mode": os.environ.get("PHYSTIME_ENV_INIT_MODE"),
            },
        },
        "source_full60": {
            "commit": SOURCE_COMMIT,
            "git_tree": SOURCE_TREE,
            "evaluation_epoch": EXPECTED_EPOCH,
            "arms": source_reports,
        },
        "runtime_configs": config_reports,
        "focused_tests": {
            "path": str(focused_tests_log),
            "sha256": sha256_file(focused_tests_log),
            "passed": True,
        },
        "experiment_contract": {
            "new_training": False,
            "frozen_checkpoint_replay": True,
            "arms": ["selected_axis", "physical_metric"],
            "weights_sources": ["online", "ema"],
            "replay_modes": list(
                (
                    "legacy_unfiltered",
                    "legacy_filtered",
                    "fullprecision_unfiltered",
                    "fullprecision_filtered",
                )
            ),
        },
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
