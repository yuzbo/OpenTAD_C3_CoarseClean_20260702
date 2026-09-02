from __future__ import annotations

import argparse
import csv
import json
import pprint
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "docs" / "experiments" / "duca_unified_matrix_manifest.yaml"
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos" / "duca_unified_fullmatrix"
SCRIPT_DIR = ROOT / "scripts" / "duca_unified_fullmatrix"


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be a mapping: {path}")
    return manifest


def _schedule_steps(manifest: dict[str, Any], schedule_name: str) -> tuple[int, int]:
    stages = manifest["schedules"][schedule_name]
    warmup = None
    transition = None
    for stage in stages:
        if stage["name"] == "uniform_warmup":
            warmup = int(stage["end_update"])
        elif stage["name"] == "transition":
            transition = int(stage["end_update"]) - int(stage["start_update"])
    if warmup is None or transition is None:
        raise ValueError(f"schedule {schedule_name} must define uniform_warmup and transition")
    return warmup, transition


def _acquisition_policy(arm: dict[str, Any]) -> str:
    allocation = str(arm["allocation"])
    if allocation == "exact_uniform":
        return "exact_uniform"
    if allocation in {"legacy_dual_phase", "h65_original_retention_transition"}:
        return "legacy_dual_phase"
    if allocation == "robust_phase":
        return "robust_phase"
    raise ValueError(f"unsupported allocation: {allocation}")


def _phase_quota_mode(arm: dict[str, Any]) -> str:
    quota = str(arm.get("quota", "adaptive"))
    if quota in {"fixed", "legacy", "h65_original"}:
        return "fixed"
    if quota == "adaptive":
        return "adaptive"
    if quota == "none":
        return "adaptive"
    raise ValueError(f"unsupported quota: {quota}")


def _actionness_source(arm: dict[str, Any]) -> dict[str, Any]:
    prior = str(arm["prior"])
    if prior == "semantic":
        return {
            "type": "C3CoarseProbeActionnessSource",
            "probe_model": "official-action-seg",
            "official_action_seg_backend": "official_asformer",
            "official_num_layers": 2,
            "spatial_size": 64,
            "tcn_hidden_dim": 96,
            "dropout": 0.0,
            "frozen": False,
            "trainable": True,
            "require_checkpoint": False,
            "train_split_supervised": True,
            "calibration_split": "train_only",
            "thumos_trained": False,
            "uses_labels": False,
            "uses_teacher": False,
            "uses_gt": False,
            "uses_prediction_cache": False,
            "source_name": "duca_semantic_asformer_pre_sigmoid_online",
        }
    if prior in {"motion", "none"}:
        return {
            "type": "ZeroShotMotionActionnessSource",
            "mode": "motion",
            "source_name": f"duca_{prior}_motion_actionness",
            "thumos_trained": False,
            "uses_labels": False,
            "uses_teacher": False,
            "uses_gt": False,
            "uses_prediction_cache": False,
            "calibration_split": "none",
            "checkpoint_hash": "none",
        }
    raise ValueError(f"unsupported prior: {prior}")


def _selector_config(manifest: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    selected = int(manifest["common_contract"]["selected_frames"])
    dense = int(manifest["common_contract"]["input_frames"])
    warmup_steps, transition_steps = _schedule_steps(manifest, str(arm["schedule"]))
    semantic = str(arm["prior"]) == "semantic"
    return {
        "type": "DucaOnlineFrameSelector",
        "in_channels": 3,
        "budget": selected,
        "budget_mode": "fixed",
        "budget_min": selected,
        "budget_max": selected,
        "budget_multiple": 16,
        "target_budget": float(selected),
        "max_radius": 16,
        "dense_window_size": dense,
        "selector_hidden_channels": 96,
        "actionness_source_cfg": _actionness_source(arm),
        "acquisition_policy": _acquisition_policy(arm),
        "phase_quota_mode": _phase_quota_mode(arm),
        "phase_sigmas": tuple(float(value) for value in manifest["phase_field_contract"]["gaussian_scales"]),
        "phase_aggregate": "median",
        "phase_use_curvature": bool(arm["curvature"]),
        "phase_temporal_nms_radius": 1,
        "phase_curvature_weight": 0.05,
        "legacy_scaffold_budget": int(manifest["phase_field_contract"]["fixed_quota"]["scaffold"]),
        "legacy_burst_budget": selected - int(manifest["phase_field_contract"]["fixed_quota"]["scaffold"]),
        "legacy_burst_radius": 2,
        "coordinate_space": "original_time",
        "detector_output_coordinate_space": "selected_axis",
        "selected_positions_unit": "original_time_index",
        "true_time_source_axis": "true_time_dense_index",
        "detector_gradient_mode": "st_sparse_gather",
        "loss_weights": {
            "actionness": 0.05,
            "boundary": 0.25,
            "hole": 0.25,
            "redundancy": 0.05,
            "coverage": 0.05,
            "entropy": 0.01,
            "teacher": 0.0,
            "detector_utility": 0.25 if str(arm["attribution"]) == "signed_feature_taylor" else 0.0,
        },
        "loss_weight_schedule": {
            "type": "progressive_joint",
            "warmup_steps": warmup_steps,
            "transition_steps": transition_steps,
            "shape": "cosine",
            "detector_utility": {
                "start": 0.0,
                "end": 0.25 if str(arm["attribution"]) == "signed_feature_taylor" else 0.0,
            },
            "actionness": {"start": 0.10 if semantic else 0.0, "end": 0.05 if semantic else 0.0},
        },
        "no_ledger_decision": True,
        "remap_gt_to_selected_axis": True,
        "selected_axis_remap_required": True,
        "forbid_ledger": True,
        "forbid_raw_prediction_cache": True,
        "forbid_external_actionness": True,
        "profile_runtime": True,
        "profile_sync_cuda": False,
        "use_coarse_hidden_features": semantic,
        "require_coarse_hidden_features": semantic,
        "coarse_hidden_dim": 96 if semantic else None,
        "coarse_hidden_proj_dim": 32 if semantic else 0,
        "coarse_hidden_dropout": 0.0,
    }


def _model_config(manifest: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    selected = int(manifest["common_contract"]["selected_frames"])
    physical_time = bool(arm["physical_time"])
    mod_enabled = bool(arm["mod"])
    backbone_cfg: dict[str, Any] = {
        "backbone": {
            "total_frames": 16,
            "ct_tubelet": physical_time,
            "amod_config": {
                "enabled": mod_enabled,
                "capacity": float(manifest["mod_contract"]["final_capacity"]),
                "capacity_schedule_successful_steps": manifest["mod_contract"]["capacity_schedule_successful_steps"],
                "amod_layers": [1, 3, 5, 7, 9, 11],
                "boundary_prior_scale": 0.25,
                "route_temperature": 1.0,
                "dense_companion_period_successful_steps": int(
                    manifest["mod_contract"]["dense_companion_period_successful_steps"]
                ),
            },
        },
        "custom": {
            "strict_temporal_padding_mask": True,
            "pre_processing_pipeline": [
                {
                    "type": "Rearrange",
                    "keys": ["frames"],
                    "ops": "b n c (t1 t) h w -> (b t1) n c t h w",
                    "t1": selected // 16,
                }
            ],
            "post_processing_pipeline": [
                {"type": "Reduce", "keys": ["feats"], "ops": "b n c t h w -> b c t", "reduction": "mean"},
                {"type": "Rearrange", "keys": ["feats"], "ops": "(b t1) c t -> b c (t1 t)", "t1": selected // 16},
                {"type": "Interpolate", "keys": ["feats"], "size": selected},
            ],
            "norm_eval": False,
            "freeze_backbone": False,
        },
    }
    head_cfg: dict[str, Any] = {
        "in_channels": 512,
        "feat_channels": 512,
        "physical_grid_actionformer": {
            "enabled": physical_time,
            "required": physical_time,
            "strict": True,
            "positions_key": "irregular_selected_positions",
            "selected_count_keys": ("selected_valid_len", "irregular_selected_count"),
            "dense_valid_len_key": "irregular_selected_valid_len",
        },
    }
    if physical_time:
        head_cfg["conv_cfg"] = {"type": "ContinuousTimeScaleAdaptiveConv1d", "ref_delta_t": 1.0}
    return {
        "type": "ActionFormer",
        "frame_selector": _selector_config(manifest, arm),
        "backbone": backbone_cfg,
        "projection": {
            "in_channels": 384,
            "out_channels": 512,
            "max_seq_len": selected,
            "attn_cfg": {"n_mha_win_size": -1},
        },
        "rpn_head": head_cfg,
        "native_temporal_geometry": {
            "enabled": True,
            "tubelet_size": 2,
            "expected_raw_count": selected,
            "expected_token_count": selected,
            "expected_transformer_depth": 12,
            "expected_adapter_indices": list(range(12)),
            "expected_adapter_kernel_size": 3,
            "expected_adapter_dilation": 1,
        },
    }


def _config_text(
    manifest: dict[str, Any],
    arm: dict[str, Any],
    *,
    phase: str,
    seed: int,
    matrix_index: int,
    work_dir: str,
) -> str:
    selected = int(manifest["common_contract"]["selected_frames"])
    dense = int(manifest["common_contract"]["input_frames"])
    model = _model_config(manifest, arm)
    config = {
        "_base_": ["../e2e_thumos_videomae_s_768x1_160_adapter.py"],
        "matrix_id": manifest["matrix_id"],
        "matrix_index": matrix_index,
        "matrix_phase": phase,
        "arm_id": arm["id"],
        "seed": int(seed),
        "window_size": dense,
        "dense_window_size": dense,
        "selected_budget": selected,
        "scaffold_budget": int(manifest["phase_field_contract"]["fixed_quota"]["scaffold"]),
        "burst_budget": selected - int(manifest["phase_field_contract"]["fixed_quota"]["scaffold"]),
        "chunk_num": selected // 16,
        "max_successful_updates": int(manifest["common_contract"]["training"]["successful_optimizer_updates"]),
        "terminal_epoch_zero_based": int(manifest["common_contract"]["training"]["terminal_epoch_zero_based"]),
        "terminal_state_key": manifest["common_contract"]["training"]["terminal_state_key"],
        "duca_unified_arm": dict(arm),
        "duca_unified_contract": {
            "exact_k": True,
            "selected_frames": selected,
            "input_frames": dense,
            "successful_optimizer_updates": int(
                manifest["common_contract"]["training"]["successful_optimizer_updates"]
            ),
            "validation_best_checkpoint_forbidden": True,
            "map_to_original_time_before_nms": True,
        },
        "duca_taylor_attribution": {
            "enabled": str(arm["attribution"]) == "signed_feature_taylor",
            "feature_levels": list(manifest["attribution_contract"]["feature_levels"]),
            "formula": manifest["attribution_contract"]["formula"],
            "update_period_successful_steps": int(
                manifest["attribution_contract"]["update_period_successful_steps"]
            ),
            "target_ema": bool(manifest["attribution_contract"]["target_ema"]),
            "create_graph": False,
            "retain_graph": True,
        },
        "model": model,
        "solver": {
            "train": {"batch_size": 2, "num_workers": 2},
            "val": {"batch_size": 2, "num_workers": 2},
            "test": {"batch_size": 2, "num_workers": 2},
            "clip_grad_norm": 1,
            "amp": True,
            "fp16_compress": True,
            "static_graph": False,
            "ema": True,
        },
        "scheduler": {"type": "LinearWarmupCosineAnnealingLR", "warmup_epoch": 5, "max_epoch": 60},
        "workflow": {
            "logging_interval": 50,
            "checkpoint_interval": 5,
            "val_loss_interval": -1,
            "val_eval_interval": 0,
            "val_start_epoch": 60,
            "end_epoch": 60,
            "max_train_iters": 100,
        },
        "inference": {"load_from_raw_predictions": False, "save_raw_prediction": True},
        "post_processing": {
            "nms": {
                "use_soft_nms": True,
                "sigma": 0.7,
                "max_seg_num": 2000,
                "multiclass": True,
                "voting_thresh": 0.7,
            },
            "save_dict": True,
        },
        "work_dir": work_dir,
    }
    lines = [
        "# Auto-generated by tools/bata/generate_duca_unified_fullmatrix.py.",
        "# Source: docs/experiments/duca_unified_matrix_manifest.yaml",
    ]
    for key, value in config.items():
        lines.append(f"{key} = {pprint.pformat(value, width=110, sort_dicts=False)}")
    return "\n\n".join(lines) + "\n"


def _matrix_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    dev_seed = int(manifest["common_contract"]["seeds"]["development"])
    confirmation_seeds = [int(seed) for seed in manifest["common_contract"]["seeds"]["confirmation"]]
    confirmation_arms = set(str(arm_id) for arm_id in manifest["confirmation_arm_ids"])
    rows: list[dict[str, Any]] = []

    def add_row(phase: str, arm: dict[str, Any], seed: int) -> None:
        idx = len(rows)
        slug = f"{phase}_{str(arm['id']).lower()}_seed{seed}"
        config_rel = f"configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_{slug}.py"
        work_dir = f"exps/thumos/adatad/duca_unified_fullmatrix/{manifest['matrix_id']}/{slug}"
        rows.append(
            {
                "index": idx,
                "task_id": slug,
                "phase": phase,
                "arm_id": str(arm["id"]),
                "seed": seed,
                "panel": str(arm["panel"]),
                "prior": str(arm["prior"]),
                "allocation": str(arm["allocation"]),
                "quota": str(arm["quota"]),
                "curvature": bool(arm["curvature"]),
                "physical_time": bool(arm["physical_time"]),
                "attribution": str(arm["attribution"]),
                "mod": bool(arm["mod"]),
                "schedule": str(arm["schedule"]),
                "role": str(arm.get("role", "")),
                "config_path": config_rel,
                "work_dir": work_dir,
                "primary_candidate": str(arm["id"]) == "A11",
                "confirmation": phase == "confirmation",
            }
        )

    for arm in manifest["arms"]:
        add_row("development", arm, dev_seed)
    for seed in confirmation_seeds:
        for arm in manifest["arms"]:
            if str(arm["id"]) in confirmation_arms:
                add_row("confirmation", arm, seed)
    return rows


def _write_matrix_files(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    with (SCRIPT_DIR / "matrix.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": "duca_unified_fullmatrix_tasks_v1",
        "matrix_id": manifest["matrix_id"],
        "base_revision": manifest["integration_base"]["revision"],
        "rows": rows,
    }
    with (SCRIPT_DIR / "matrix.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def _write_configs(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    arm_by_id = {str(arm["id"]): arm for arm in manifest["arms"]}
    for row in rows:
        arm = arm_by_id[row["arm_id"]]
        text = _config_text(
            manifest,
            arm,
            phase=str(row["phase"]),
            seed=int(row["seed"]),
            matrix_index=int(row["index"]),
            work_dir=str(row["work_dir"]),
        )
        (ROOT / row["config_path"]).write_text(text, encoding="utf-8")


def _write_freeze_doc(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    doc = ROOT / "docs" / "experiments" / "DUCA_UNIFIED_FULLMATRIX_FREEZE.md"
    development = [row for row in rows if row["phase"] == "development"]
    confirmation = [row for row in rows if row["phase"] == "confirmation"]
    lines = [
        "# DUCA Unified Full-Matrix Freeze",
        "",
        f"- Matrix ID: `{manifest['matrix_id']}`",
        f"- Integration base: `{manifest['integration_base']['revision']}`",
        f"- Branch: `{manifest['deployment']['branch_name']}`",
        f"- Development tasks: `{len(development)}`",
        f"- Confirmation tasks: `{len(confirmation)}`",
        f"- Total train/eval tasks: `{len(rows)}`",
        f"- Bootstrap shards: `{manifest['common_contract']['statistics']['bootstrap_shards']}`",
        f"- Cost arms: `{', '.join(manifest['cost_benchmark']['arm_ids'])}`",
        "",
        "The manifest copied into this repository is the source of truth. Historical references in that manifest are descriptive anchors only; matched conclusions must come from this 41-task matrix.",
        "",
        "Primary contrast: `A11 - A10` on official THUMOS14 average mAP under a strict 6000-successful-update budget.",
    ]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_slurm_scripts(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    train_count = len(rows)
    bootstrap_shards = int(manifest["common_contract"]["statistics"]["bootstrap_shards"])
    cost_count = len(manifest["cost_benchmark"]["arm_ids"])
    matrix_id = manifest["matrix_id"]

    (SCRIPT_DIR / "preflight.sbatch").write_text(
        f"""#!/bin/bash
#SBATCH -J duca_preflight
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH -o /data/run01/sczc063/yuzibo/slurm_logs/%x_%j.out
#SBATCH -e /data/run01/sczc063/yuzibo/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${{PROJECT_DIR:?PROJECT_DIR is required}}"
RUN_ROOT="${{RUN_ROOT:-/data/run01/sczc063/yuzibo/runs/{matrix_id}}}"
mkdir -p "${{RUN_ROOT}}" /data/run01/sczc063/yuzibo/slurm_logs
cd "${{PROJECT_DIR}}"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

python -m py_compile tools/train.py tools/test.py tools/bata/generate_duca_unified_fullmatrix.py tools/bata/aggregate_duca_unified_fullmatrix.py tools/bata/bootstrap_duca_unified_fullmatrix.py
python tools/bata/generate_duca_unified_fullmatrix.py --check
python -m pytest tests/test_duca_unified_phase.py tests/test_duca_unified_physical_time.py tests/test_duca_unified_attribution.py tests/test_duca_unified_mod.py tests/test_duca_unified_curriculum.py -q
""",
        encoding="utf-8",
    )

    (SCRIPT_DIR / "train_eval_array.sbatch").write_text(
        f"""#!/bin/bash
#SBATCH -J duca_train_eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-{train_count - 1}%{manifest['deployment']['default_max_concurrent_gpu_tasks']}
#SBATCH -t 7-00:00:00
#SBATCH -o /data/run01/sczc063/yuzibo/slurm_logs/%x_%A_%a.out
#SBATCH -e /data/run01/sczc063/yuzibo/slurm_logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${{PROJECT_DIR:?PROJECT_DIR is required}}"
RUN_ROOT="${{RUN_ROOT:-/data/run01/sczc063/yuzibo/runs/{matrix_id}}}"
MATRIX_JSON="${{PROJECT_DIR}}/scripts/duca_unified_fullmatrix/matrix.json"
mkdir -p "${{RUN_ROOT}}/train_eval" /data/run01/sczc063/yuzibo/slurm_logs
cd "${{PROJECT_DIR}}"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

read -r CONFIG SEED TASK_ID WORK_DIR <<EOF_ROW
$(python - "$MATRIX_JSON" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
row = payload["rows"][int(sys.argv[2])]
print(row["config_path"], row["seed"], row["task_id"], row["work_dir"])
PY
)
EOF_ROW

echo "DUCA train/eval task=${{TASK_ID}} seed=${{SEED}} config=${{CONFIG}}"
torchrun --standalone --nnodes=1 --nproc_per_node=1 tools/train.py "${{CONFIG}}" --seed "${{SEED}}" --id 0
python tools/test.py "${{CONFIG}}" --seed "${{SEED}}" --id 0 --checkpoint "${{WORK_DIR}}/gpu1_id0/checkpoint/epoch_59.pth" --cfg-options inference.load_from_raw_predictions=False inference.save_raw_prediction=True > "${{RUN_ROOT}}/train_eval/${{TASK_ID}}.test.log" 2>&1
""",
        encoding="utf-8",
    )

    (SCRIPT_DIR / "cost_array.sbatch").write_text(
        f"""#!/bin/bash
#SBATCH -J duca_cost
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-{cost_count - 1}
#SBATCH -t 04:00:00
#SBATCH -o /data/run01/sczc063/yuzibo/slurm_logs/%x_%A_%a.out
#SBATCH -e /data/run01/sczc063/yuzibo/slurm_logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${{PROJECT_DIR:?PROJECT_DIR is required}}"
RUN_ROOT="${{RUN_ROOT:-/data/run01/sczc063/yuzibo/runs/{matrix_id}}}"
MATRIX_JSON="${{PROJECT_DIR}}/scripts/duca_unified_fullmatrix/matrix.json"
mkdir -p "${{RUN_ROOT}}/cost" /data/run01/sczc063/yuzibo/slurm_logs
cd "${{PROJECT_DIR}}"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

python tools/bata/aggregate_duca_unified_fullmatrix.py --matrix "$MATRIX_JSON" --run-root "$RUN_ROOT" --cost-index "$SLURM_ARRAY_TASK_ID" --output "${{RUN_ROOT}}/cost/cost_${{SLURM_ARRAY_TASK_ID}}.json"
""",
        encoding="utf-8",
    )

    (SCRIPT_DIR / "bootstrap_array.sbatch").write_text(
        f"""#!/bin/bash
#SBATCH -J duca_bootstrap
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH --array=0-{bootstrap_shards - 1}
#SBATCH -t 02:00:00
#SBATCH -o /data/run01/sczc063/yuzibo/slurm_logs/%x_%A_%a.out
#SBATCH -e /data/run01/sczc063/yuzibo/slurm_logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${{PROJECT_DIR:?PROJECT_DIR is required}}"
RUN_ROOT="${{RUN_ROOT:-/data/run01/sczc063/yuzibo/runs/{matrix_id}}}"
mkdir -p "${{RUN_ROOT}}/bootstrap" /data/run01/sczc063/yuzibo/slurm_logs
cd "${{PROJECT_DIR}}"

module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

python tools/bata/bootstrap_duca_unified_fullmatrix.py --matrix scripts/duca_unified_fullmatrix/matrix.json --run-root "$RUN_ROOT" --shard "$SLURM_ARRAY_TASK_ID" --num-shards {bootstrap_shards} --draws {manifest['common_contract']['statistics']['video_cluster_bootstrap_draws']} --output "${{RUN_ROOT}}/bootstrap/bootstrap_${{SLURM_ARRAY_TASK_ID}}.json"
""",
        encoding="utf-8",
    )

    (SCRIPT_DIR / "finalize.sbatch").write_text(
        f"""#!/bin/bash
#SBATCH -J duca_finalize
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH -t 02:00:00
#SBATCH -o /data/run01/sczc063/yuzibo/slurm_logs/%x_%j.out
#SBATCH -e /data/run01/sczc063/yuzibo/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${{PROJECT_DIR:?PROJECT_DIR is required}}"
RUN_ROOT="${{RUN_ROOT:-/data/run01/sczc063/yuzibo/runs/{matrix_id}}}"
cd "${{PROJECT_DIR}}"

module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

python tools/bata/aggregate_duca_unified_fullmatrix.py --matrix scripts/duca_unified_fullmatrix/matrix.json --run-root "$RUN_ROOT" --bootstrap-dir "${{RUN_ROOT}}/bootstrap" --output "${{RUN_ROOT}}/duca_unified_fullmatrix_summary.json"
""",
        encoding="utf-8",
    )

    (SCRIPT_DIR / "audit_afterany.sbatch").write_text(
        f"""#!/bin/bash
#SBATCH -J duca_audit
#SBATCH -p cpu
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -o /data/run01/sczc063/yuzibo/slurm_logs/%x_%j.out
#SBATCH -e /data/run01/sczc063/yuzibo/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${{PROJECT_DIR:?PROJECT_DIR is required}}"
RUN_ROOT="${{RUN_ROOT:-/data/run01/sczc063/yuzibo/runs/{matrix_id}}}"
cd "${{PROJECT_DIR}}"

module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

python tools/bata/aggregate_duca_unified_fullmatrix.py --matrix scripts/duca_unified_fullmatrix/matrix.json --run-root "$RUN_ROOT" --audit-only --output "${{RUN_ROOT}}/duca_unified_fullmatrix_audit.json"
""",
        encoding="utf-8",
    )

    (SCRIPT_DIR / "submit_all.sh").write_text(
        f"""#!/bin/bash
set -euo pipefail

ORIGINAL_ARGV=("$0" "$@")

usage() {{
  cat <<'EOF_USAGE'
Usage: bash scripts/duca_unified_fullmatrix/submit_all.sh \\
  --repo-root <remote clean checkout> \\
  --revision <final commit sha> \\
  --run-root <run output root> \\
  --base <remote base path> \\
  --account <slurm account> \\
  --partition <gpu partition> \\
  --qos <slurm qos> \\
  --max-concurrent <train array concurrency>
EOF_USAGE
}}

BASE="${{BASE:-/data/run01/sczc063/yuzibo}}"
PROJECT_DIR="${{PROJECT_DIR:-$(pwd)}}"
REVISION="${{REVISION:-}}"
RUN_ROOT="${{RUN_ROOT:-}}"
ACCOUNT="${{ACCOUNT:-sczc063}}"
PARTITION="${{PARTITION:-gpu}}"
QOS="${{QOS:-normal}}"
MAX_CONCURRENT="${{MAX_CONCURRENT:-{manifest['deployment']['default_max_concurrent_gpu_tasks']}}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      PROJECT_DIR="$2"
      shift 2
      ;;
    --revision)
      REVISION="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --base)
      BASE="$2"
      shift 2
      ;;
    --account)
      ACCOUNT="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    --max-concurrent)
      MAX_CONCURRENT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"
if [[ -z "$REVISION" ]]; then
  REVISION="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
fi
if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="${{BASE}}/experiments/duca_unified_fullmatrix_${{REVISION:0:12}}_$(date +%Y%m%d_%H%M%S)"
fi
if [[ ! "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-concurrent must be a positive integer: $MAX_CONCURRENT" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT" "${{BASE}}/slurm_logs"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available; run this script on the N16R4 Slurm login node." >&2
  exit 2
fi

cd "$PROJECT_DIR"
ACTUAL_REVISION="$(git rev-parse HEAD)"
if [[ "$ACTUAL_REVISION" != "$REVISION" ]]; then
  echo "checkout revision mismatch: expected $REVISION, got $ACTUAL_REVISION" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked working tree is not clean at $PROJECT_DIR" >&2
  git status --short --untracked-files=no >&2
  exit 2
fi

export PROJECT_DIR RUN_ROOT

SLURM_SHARED_ARGS=()
if [[ -n "$ACCOUNT" ]]; then
  SLURM_SHARED_ARGS+=("--account=$ACCOUNT")
fi
if [[ -n "$QOS" ]]; then
  SLURM_SHARED_ARGS+=("--qos=$QOS")
fi
SLURM_GPU_ARGS=("${{SLURM_SHARED_ARGS[@]}}")
if [[ -n "$PARTITION" ]]; then
  SLURM_GPU_ARGS+=("--partition=$PARTITION")
fi

preflight=$(sbatch --parsable "${{SLURM_GPU_ARGS[@]}}" scripts/duca_unified_fullmatrix/preflight.sbatch)
train=$(sbatch --parsable "${{SLURM_GPU_ARGS[@]}}" --dependency=afterok:$preflight --array=0-{train_count - 1}%$MAX_CONCURRENT scripts/duca_unified_fullmatrix/train_eval_array.sbatch)
cost=$(sbatch --parsable "${{SLURM_GPU_ARGS[@]}}" --dependency=afterok:$train scripts/duca_unified_fullmatrix/cost_array.sbatch)
boot=$(sbatch --parsable "${{SLURM_SHARED_ARGS[@]}}" --dependency=afterok:$train scripts/duca_unified_fullmatrix/bootstrap_array.sbatch)
finalize=$(sbatch --parsable "${{SLURM_SHARED_ARGS[@]}}" --dependency=afterok:$train:$cost:$boot scripts/duca_unified_fullmatrix/finalize.sbatch)
audit=$(sbatch --parsable "${{SLURM_SHARED_ARGS[@]}}" --dependency=afterany:$train:$cost:$boot:$finalize scripts/duca_unified_fullmatrix/audit_afterany.sbatch)

printf -v SUBMISSION_ARGV '%q ' "${{ORIGINAL_ARGV[@]}}"
SUBMISSION_ARGV="${{SUBMISSION_ARGV%% }}"

python - "$RUN_ROOT" "$PROJECT_DIR" "$REVISION" "$PROJECT_DIR" "$SUBMISSION_ARGV" "$preflight" "$train" "$cost" "$boot" "$finalize" "$audit" <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess
import sys

run_root = pathlib.Path(sys.argv[1])
project_dir = pathlib.Path(sys.argv[2])
revision = sys.argv[3]
remote_repo = sys.argv[4]
submission_argv = sys.argv[5]
job_ids = sys.argv[6:12]

def atomic_write(path: pathlib.Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)

def atomic_copy(src: pathlib.Path, dst: pathlib.Path) -> None:
    atomic_write(dst, src.read_bytes())

run_root.mkdir(parents=True, exist_ok=True)
atomic_copy(project_dir / "scripts/duca_unified_fullmatrix/matrix.tsv", run_root / "matrix.tsv")
atomic_copy(project_dir / "docs/experiments/DUCA_UNIFIED_FULLMATRIX_FREEZE.md", run_root / "scientific_freeze.md")

tracked = subprocess.check_output(
    ["git", "ls-files"],
    cwd=project_dir,
    text=True,
    encoding="utf-8",
).splitlines()
hash_lines = []
for rel_path in sorted(tracked):
    src = project_dir / rel_path
    if src.is_file():
        hash_lines.append(f"{{hashlib.sha256(src.read_bytes()).hexdigest()}}  {{rel_path}}")
atomic_write(run_root / "source_manifest_sha256.txt", ("\\n".join(hash_lines) + "\\n").encode("utf-8"))

payload = {{
    "schema_version": "duca_unified_slurm_submission_v1",
    "matrix_id": "{matrix_id}",
    "preflight_job_id": job_ids[0],
    "train_eval_array_job_id": job_ids[1],
    "cost_array_job_id": job_ids[2],
    "bootstrap_array_job_id": job_ids[3],
    "finalizer_job_id": job_ids[4],
    "audit_afterany_job_id": job_ids[5],
    "final_commit": revision,
    "remote_repo": remote_repo,
    "run_root": str(run_root),
    "submission_argv": submission_argv,
}}
data = (json.dumps(payload, indent=2, sort_keys=True) + "\\n").encode("utf-8")
atomic_write(run_root / "submission_manifest.json", data)
atomic_write(run_root / "slurm_submission_manifest.json", data)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
""",
        encoding="utf-8",
    )


def generate(manifest_path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    manifest = _load_manifest(manifest_path)
    rows = _matrix_rows(manifest)
    expected = int(manifest["deployment"]["train_eval_array_tasks"])
    if len(rows) != expected:
        raise ValueError(f"matrix produced {len(rows)} rows, manifest expects {expected}")
    _write_configs(manifest, rows)
    _write_matrix_files(manifest, rows)
    _write_freeze_doc(manifest, rows)
    _write_slurm_scripts(manifest, rows)
    return rows


def _check(manifest_path: Path = MANIFEST_PATH) -> None:
    manifest = _load_manifest(manifest_path)
    rows = _matrix_rows(manifest)
    matrix_path = SCRIPT_DIR / "matrix.json"
    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)
    with matrix_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("matrix_id") != manifest["matrix_id"]:
        raise AssertionError("matrix_id mismatch")
    if payload.get("base_revision") != manifest["integration_base"]["revision"]:
        raise AssertionError("base revision mismatch")
    if payload.get("rows") != rows:
        raise AssertionError("matrix rows do not match manifest-derived rows")
    missing = [row["config_path"] for row in rows if not (ROOT / row["config_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"missing generated configs: {missing[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DUCA unified full-matrix configs and Slurm DAG scripts")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--check", action="store_true", help="verify generated files match manifest-derived matrix")
    args = parser.parse_args()
    if args.check:
        _check(args.manifest)
        print("DUCA unified full-matrix generated files are in sync.")
        return
    rows = generate(args.manifest)
    print(f"Generated {len(rows)} DUCA unified full-matrix train/eval tasks.")


if __name__ == "__main__":
    main()
