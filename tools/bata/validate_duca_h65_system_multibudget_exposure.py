"""Read-only PRE_RUN admission for the matched H65 exposure experiment."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.prepare_duca_h65_system_multibudget_exposure import (
    HELD_OUT_ID_SHA256,
    TOTAL_UPDATES,
    TRAINING_ID_SHA256,
    id_manifest_sha256,
    sha256_file,
)


CONFIG_ROOT = ROOT / "configs" / "adatad" / "thumos"
CONTROL_CONFIG = CONFIG_ROOT / "duca_h65_system_multibudget_exposure_control.py"
CANDIDATE_CONFIG = CONFIG_ROOT / "duca_h65_system_multibudget_exposure_candidate.py"
STAGE1_CONFIG = CONFIG_ROOT / "duca_sampling_rate_curriculum_stage1_uniform384.py"
SEEDS = (3407, 3408, 3409)


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    return value


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise SystemExit("Stage-1 checkpoint must contain one mapping")
    return value


def _validate_resources(
    *,
    video_root: Path,
    annotation: Path,
    category: Path,
    pretrain: Path,
    pretrain_sha256: str,
) -> None:
    for path in (annotation, category, pretrain):
        if not path.is_file():
            raise SystemExit(f"canonical resource unreadable: {path}")
    entries = list(video_root.iterdir())
    if len(entries) != 411 or any(not entry.is_symlink() or not entry.exists() for entry in entries):
        raise SystemExit("canonical video root must contain 411 valid symlinks")
    if not pretrain.is_absolute():
        raise SystemExit("VideoMAE pretrain binding must be an absolute path")
    if sha256_file(pretrain) != pretrain_sha256.lower():
        raise SystemExit("VideoMAE pretrain SHA256 mismatch")


def _validate_checkpoint(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or not path.is_absolute():
        raise SystemExit("Stage-1 checkpoint must be a readable absolute path")
    if sha256_file(path) != expected_sha256.lower():
        raise SystemExit("Stage-1 checkpoint SHA256 mismatch")
    checkpoint = _load_checkpoint(path)
    if int(checkpoint.get("epoch", -1)) != 29:
        raise SystemExit("Stage-1 checkpoint must be terminal epoch 29")
    if not isinstance(checkpoint.get("state_dict_ema"), dict):
        raise SystemExit("Stage-1 checkpoint lacks state_dict_ema")
    copied_config = path.parent.parent / STAGE1_CONFIG.name
    if not copied_config.is_file() or sha256_file(copied_config) != sha256_file(STAGE1_CONFIG):
        raise SystemExit("Stage-1 checkpoint provenance config mismatch")


def _validate_calibration(calibration_dir: Path) -> dict[str, Any]:
    report_path = calibration_dir / "pre_run_calibration.json"
    manifest_path = calibration_dir / "held_out_fixed_mixed_budget_manifest.json"
    ids_path = calibration_dir / "held_out_video_ids.txt"
    bootstrap_path = calibration_dir / "paired_bootstrap_indices.npy"
    inference_annotation_path = calibration_dir / "held_out_inference_annotation.json"
    for path in (
        report_path,
        manifest_path,
        ids_path,
        bootstrap_path,
        inference_annotation_path,
    ):
        if not path.is_file():
            raise SystemExit(f"PRE_RUN artifact missing: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("training_video_count") != 200
        or report.get("held_out_video_count") != 211
        or report.get("training_id_sha256") != TRAINING_ID_SHA256
        or report.get("held_out_id_sha256") != HELD_OUT_ID_SHA256
    ):
        raise SystemExit("PRE_RUN report does not bind the admitted 200/211 identities")
    if report.get("held_out_action_labels_or_segments_used_before_prediction_sealing") is not False:
        raise SystemExit("PRE_RUN calibration must not use held-out action semantics")
    probabilities = {int(key): float(value) for key, value in report["probabilities"].items()}
    means = {int(key): float(value) for key, value in report["actual_observation_means"].items()}
    if not means[256] < means[384] < means[512]:
        raise SystemExit("actual observation means are not strictly monotone")
    expected_p256 = 0.5 * (means[512] - means[384]) / (means[512] - means[256])
    expected = {256: expected_p256, 384: 0.5, 512: 0.5 - expected_p256}
    if any(abs(probabilities[key] - expected[key]) > 1.0e-12 for key in expected):
        raise SystemExit("PRE_RUN probabilities do not follow the frozen actual-cost formula")
    counts = {int(key): int(value) for key, value in report["update_counts"].items()}
    if sum(counts.values()) != TOTAL_UPDATES or counts[384] != TOTAL_UPDATES // 2:
        raise SystemExit("PRE_RUN update multiset is not the frozen 6000-update course")
    for seed in SEEDS:
        cost = report["seed_training_cost_replay"].get(str(seed))
        if not isinstance(cost, dict) or abs(float(cost["relative_delta"])) > 0.005:
            raise SystemExit(f"seed {seed} does not satisfy the matched training-cost bound")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_report = report["held_out_manifest"]
    if sha256_file(manifest_path) != manifest_report["sha256"]:
        raise SystemExit("held-out mixed manifest SHA256 mismatch")
    if len(manifest) != int(manifest_report["unique_manifest_key_count"]):
        raise SystemExit("held-out mixed manifest key count mismatch")
    if int(manifest_report["executed_window_count"]) < len(manifest):
        raise SystemExit("held-out executed-window count cannot be smaller than unique keys")
    if int(manifest_report["mixed_actual_observations"]) > int(
        manifest_report["control_actual_observations"]
    ):
        raise SystemExit("fixed held-out mixed manifest exceeds the K384 observation cost")
    ids = [line for line in ids_path.read_text(encoding="utf-8").splitlines() if line]
    if len(ids) != 211 or id_manifest_sha256(ids) != HELD_OUT_ID_SHA256:
        raise SystemExit("held-out ID artifact differs from the admitted 211-video manifest")
    inference_annotation = json.loads(
        inference_annotation_path.read_text(encoding="utf-8")
    )
    inference_database = inference_annotation.get("database")
    inference_report = report.get("held_out_inference_annotation", {})
    if (
        not isinstance(inference_database, dict)
        or set(inference_database) != set(ids)
        or any(
            set(video_info) != {"subset", "frame", "duration"}
            or video_info.get("subset") != "validation"
            for video_info in inference_database.values()
        )
        or inference_report.get("contains_action_labels_or_segments") is not False
        or int(inference_report.get("video_count", -1)) != 211
        or sha256_file(inference_annotation_path) != inference_report.get("sha256")
    ):
        raise SystemExit(
            "held-out inference annotation must contain only the admitted video geometry"
        )
    bootstrap = np.load(bootstrap_path, mmap_mode="r", allow_pickle=False)
    if bootstrap.shape != (10_000, 211) or bootstrap.dtype != np.uint16:
        raise SystemExit("paired bootstrap indices must be uint16[10000,211]")
    if int(bootstrap.min()) < 0 or int(bootstrap.max()) >= 211:
        raise SystemExit("paired bootstrap indices leave the 211-video population")
    if sha256_file(bootstrap_path) != report["paired_bootstrap_indices"]["sha256"]:
        raise SystemExit("paired bootstrap index SHA256 mismatch")
    return report


def _load_matched_configs(
    *,
    seed: int,
    stage1: Path,
    stage1_sha256: str,
    pretrain: Path,
    probabilities: dict[int, float],
) -> tuple[Config, Config]:
    os.environ.update(
        DUCA_STAGE1_CHECKPOINT=str(stage1),
        DUCA_STAGE1_CHECKPOINT_SHA256=stage1_sha256.lower(),
        DUCA_STAGE1_CHECKPOINT_EPOCH="29",
        DUCA_VIDEOMAE_PRETRAIN=str(pretrain),
        DUCA_EXPERIMENT_SEED=str(seed),
        DUCA_MB_P256=f"{probabilities[256]:.17g}",
        DUCA_MB_P512=f"{probabilities[512]:.17g}",
    )
    os.environ.pop("DUCA_MB_EVAL_MANIFEST", None)
    os.environ.pop("DUCA_MB_EVAL_BUDGET", None)
    return Config.fromfile(str(CONTROL_CONFIG)), Config.fromfile(str(CANDIDATE_CONFIG))


def _validate_matched_configs(
    *,
    stage1: Path,
    stage1_sha256: str,
    pretrain: Path,
    probabilities: dict[int, float],
) -> None:
    for seed in SEEDS:
        control, candidate = _load_matched_configs(
            seed=seed,
            stage1=stage1,
            stage1_sha256=stage1_sha256,
            pretrain=pretrain,
            probabilities=probabilities,
        )
        if control.seed != seed or candidate.seed != seed:
            raise SystemExit(f"seed {seed} is not bound identically across arms")
        workflow = control.workflow
        expected_workflow = {
            "end_epoch": 60,
            "expected_train_batches_per_epoch": 100,
            "expected_successful_optimizer_updates": 6000,
            "primary_checkpoint_epoch": 59,
            "primary_checkpoint_state_key": "state_dict_ema",
            "checkpoint_interval": 5,
        }
        for key, expected in expected_workflow.items():
            if workflow.get(key) != expected:
                raise SystemExit(f"control workflow.{key} differs from {expected!r}")
        if workflow.formal_successful_update_contract:
            raise SystemExit(
                "H65 multi-budget training must not route through the legacy P0 binder"
            )
        if (
            not workflow.seal_eval_dataloaders_during_training
            or control.dataset.val is not None
            or workflow.val_eval_interval > 0
            or workflow.val_loss_interval > 0
            or workflow.intermediate_validation_selects_checkpoint
        ):
            raise SystemExit("formal training must seal held-out loaders and terminal checkpoint selection")
        init = workflow.model_initialization
        if (
            Path(init.checkpoint_path) != stage1
            or init.checkpoint_sha256.lower() != stage1_sha256.lower()
            or init.expected_checkpoint_epoch != 29
            or init.state_key != "state_dict_ema"
            or init.reset_state_keys != ["frame_selector._loss_weight_schedule_step"]
        ):
            raise SystemExit("Stage-2 initialization is not bound to the frozen Stage-1 EMA")
        if Path(control.model.backbone.custom.pretrain) != pretrain:
            raise SystemExit("formal config does not bind the canonical absolute VideoMAE pretrain")
        if candidate.workflow != {
            **control.workflow,
            "training_profile": "duca_h65_system_multibudget_exposure_candidate",
        }:
            raise SystemExit("matched workflows differ outside their arm name")
        for key in ("optimizer", "scheduler", "solver", "dataset", "evaluation", "post_processing"):
            if _plain(control[key]) != _plain(candidate[key]):
                raise SystemExit(f"matched arms differ outside exposure at {key}")
        control_model = copy.deepcopy(_plain(control.model))
        candidate_model = copy.deepcopy(_plain(candidate.model))
        exposure = candidate_model["frame_selector"].pop("multi_budget_exposure", None)
        if exposure is None or control_model != candidate_model:
            raise SystemExit("matched models differ outside multi_budget_exposure")
        observed_probabilities = {
            int(key): float(value) for key, value in exposure["probabilities"].items()
        }
        if any(
            abs(observed_probabilities[key] - probabilities[key]) > 1.0e-12
            for key in probabilities
        ):
            raise SystemExit("candidate config differs from PRE_RUN-calibrated probabilities")
        if exposure["evaluation_manifest"] is not None or exposure["evaluation_budget"] != 384:
            raise SystemExit("formal training must not bind the held-out mixed manifest")


def _validate_smoke_artifacts(probe_path: Path, checkpoint_path: Path) -> None:
    if not probe_path.is_file() or not checkpoint_path.is_file():
        raise SystemExit("four-update CUDA smoke artifacts are incomplete")
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if (
        probe.get("schema_version") != "duca_training_probe_v1"
        or int(probe.get("attempted_steps", -1)) != 4
        or int(probe.get("successful_optimizer_steps", -1)) != 4
        or int(probe.get("skipped_optimizer_steps", -1)) != 0
        or int(probe.get("finite_loss_steps", -1)) != 4
        or int(probe.get("finite_gradient_steps", -1)) != 4
    ):
        raise SystemExit("CUDA smoke did not complete four finite successful updates")
    steps = probe.get("selector_steps")
    if not isinstance(steps, list) or len(steps) != 4:
        raise SystemExit("CUDA smoke lacks four selector execution records")
    expected_budgets = (384, 256, 384, 512)
    observed_variable_execution = False
    for step, expected_budget in zip(steps, expected_budgets):
        exposure = step.get("h65_multi_budget_exposure")
        execution = step.get("backbone_execution_profile")
        if not isinstance(exposure, dict) or not isinstance(execution, dict):
            raise SystemExit("CUDA smoke lacks selector/backbone exposure accounting")
        requested = [int(value) for value in exposure.get("requested_budget", [])]
        if not requested or set(requested) != {expected_budget}:
            raise SystemExit("CUDA smoke does not follow the frozen successful-update budget course")
        for key in ("requested_budget", "effective_budget", "actual_observations", "execution_slots"):
            if [int(value) for value in exposure.get(key, [])] != [
                int(value) for value in execution.get(key, [])
            ]:
                raise SystemExit(f"selector and VideoMAE execution disagree at {key}")
        if sum(int(value) for value in execution["actual_observations"]) != int(
            execution["total_actual_observations"]
        ):
            raise SystemExit("actual observation counter does not equal the VideoMAE batch inputs")
        used_legacy = bool(execution.get("used_legacy_k384_path"))
        if expected_budget == 384 and not used_legacy:
            raise SystemExit("forced K384 smoke did not use the exact legacy path")
        if expected_budget != 384 and not used_legacy:
            observed_variable_execution = True
        if used_legacy and (
            set(int(value) for value in execution["effective_budget"]) != {384}
            or set(int(value) for value in execution["execution_slots"]) != {384}
        ):
            raise SystemExit("legacy execution was used outside a true K384 collapse")
    if not observed_variable_execution:
        raise SystemExit("CUDA smoke did not execute any real variable-length VideoMAE bucket")
    groups = probe.get("parameter_group_coverage", {})
    for group in ("backbone", "coarse_probe", "selector", "projection", "detector_head"):
        values = groups.get(group)
        if not isinstance(values, dict) or int(values.get("gradient_seen", 0)) <= 0:
            raise SystemExit(f"CUDA smoke observed no authorized gradient in {group}")
    checkpoint = _load_checkpoint(checkpoint_path)
    if int(checkpoint.get("epoch", -1)) != 0:
        raise SystemExit("CUDA smoke checkpoint must be epoch 0")
    if int(checkpoint.get("successful_optimizer_updates", -1)) != 4:
        raise SystemExit("CUDA smoke checkpoint must record four successful updates")
    for key in ("state_dict", "state_dict_ema", "optimizer", "scheduler", "grad_scaler"):
        if key not in checkpoint:
            raise SystemExit(f"CUDA smoke checkpoint lacks {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--category", type=Path, required=True)
    parser.add_argument("--pretrain", type=Path, required=True)
    parser.add_argument("--pretrain-sha256", required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage1-sha256", required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--smoke-probe", type=Path)
    parser.add_argument("--smoke-checkpoint", type=Path)
    args = parser.parse_args()

    paths = {
        name: value.expanduser().resolve()
        for name, value in {
            "video_root": args.video_root,
            "annotation": args.annotation,
            "category": args.category,
            "pretrain": args.pretrain,
            "stage1": args.stage1,
            "calibration_dir": args.calibration_dir,
        }.items()
    }
    _validate_resources(
        video_root=paths["video_root"],
        annotation=paths["annotation"],
        category=paths["category"],
        pretrain=paths["pretrain"],
        pretrain_sha256=args.pretrain_sha256,
    )
    _validate_checkpoint(paths["stage1"], args.stage1_sha256)
    report = _validate_calibration(paths["calibration_dir"])
    if sha256_file(paths["annotation"]) != report.get("annotation_sha256"):
        raise SystemExit("canonical annotation differs from PRE_RUN calibration")
    probabilities = {int(key): float(value) for key, value in report["probabilities"].items()}
    _validate_matched_configs(
        stage1=paths["stage1"],
        stage1_sha256=args.stage1_sha256,
        pretrain=paths["pretrain"],
        probabilities=probabilities,
    )
    if (args.smoke_probe is None) != (args.smoke_checkpoint is None):
        raise SystemExit("--smoke-probe and --smoke-checkpoint must be supplied together")
    if args.smoke_probe is not None:
        _validate_smoke_artifacts(
            args.smoke_probe.expanduser().resolve(),
            args.smoke_checkpoint.expanduser().resolve(),
        )
    print("PASS H65 system multi-budget exposure: 200/211 identities, artifacts, and matched configs")


if __name__ == "__main__":
    main()
