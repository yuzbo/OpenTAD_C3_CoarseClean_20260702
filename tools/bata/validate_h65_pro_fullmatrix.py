from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DOC_MATRIX = ROOT / "docs" / "experiments" / "h65_pro_fullmatrix_20260902" / "03_EXPERIMENT_MATRIX.csv"
EXPECTED_F = {
    "F01": ("OFF", "OFF", "OFF", "OFF", "ON"),
    "F02": ("OFF", "OFF", "OFF", "ON", "OFF"),
    "F03": ("OFF", "OFF", "ON", "OFF", "OFF"),
    "F04": ("OFF", "OFF", "ON", "ON", "ON"),
    "F05": ("OFF", "ON", "OFF", "OFF", "OFF"),
    "F06": ("OFF", "ON", "OFF", "ON", "ON"),
    "F07": ("OFF", "ON", "ON", "OFF", "ON"),
    "F08": ("OFF", "ON", "ON", "ON", "OFF"),
    "F09": ("ON", "OFF", "OFF", "OFF", "OFF"),
    "F10": ("ON", "OFF", "OFF", "ON", "ON"),
    "F11": ("ON", "OFF", "ON", "OFF", "ON"),
    "F12": ("ON", "OFF", "ON", "ON", "OFF"),
    "F13": ("ON", "ON", "OFF", "OFF", "ON"),
    "F14": ("ON", "ON", "OFF", "ON", "OFF"),
    "F15": ("ON", "ON", "ON", "OFF", "OFF"),
    "F16": ("ON", "ON", "ON", "ON", "ON"),
}
EXPECTED_CANONICAL = {
    "C0": ("OFF", "OFF", "OFF", "OFF", "ON", {5417, 9173}),
    "C1": ("ON", "OFF", "OFF", "OFF", "ON", {3407, 5417, 9173}),
    "C2": ("ON", "ON", "OFF", "OFF", "ON", {5417, 9173}),
    "C3": ("ON", "ON", "ON", "ON", "ON", {5417, 9173}),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_physical_time_optimizer(cfg: Config, experiment_id: str) -> None:
    backbone_cfg = cfg.model.backbone.backbone
    if not bool(backbone_cfg.get("relative_physical_time_residual", False)):
        return
    custom_groups = {
        str(group["name"]): group for group in cfg.optimizer.backbone.custom
    }
    scale_group = custom_groups.get("relative_physical_time_scale")
    require(scale_group is not None, f"{experiment_id}: physical-time scale missing from optimizer")
    require(float(scale_group["lr"]) > 0.0, f"{experiment_id}: physical-time scale lr must be positive")
    require(
        float(scale_group["weight_decay"]) == 0.0,
        f"{experiment_id}: physical-time residual scalar must not use weight decay",
    )


def validate_config(row: dict[str, str]) -> None:
    cfg_path = ROOT / row["config"]
    require(cfg_path.is_file(), f"missing config: {row['config']}")
    cfg = Config.fromfile(str(cfg_path))
    require(int(cfg.get("total_epochs", cfg.workflow.end_epoch)) == 60, f"{row['experiment_id']}: epochs drift")
    require(int(cfg.get("max_updates", 6000)) == 6000, f"{row['experiment_id']}: update budget drift")
    require(int(cfg.workflow.end_epoch) == 60, f"{row['experiment_id']}: workflow end_epoch drift")
    require(int(cfg.workflow.get("primary_checkpoint_epoch", 59)) == 59, f"{row['experiment_id']}: primary epoch drift")
    require(str(cfg.workflow.get("primary_checkpoint_state_key", "state_dict_ema")) == "state_dict_ema", f"{row['experiment_id']}: primary state drift")
    if row["experiment_id"] == "REF-D768":
        require(cfg.model.get("frame_selector", None) is None, "REF-D768 must not use acquisition")
        require(int(cfg.model.projection.max_seq_len) == 768, "REF-D768 must keep 768 detector frames")
        require(str(cfg.workflow.get("formal_protocol", "")) == "h65_pro_dense_reference_official60_v1", "REF-D768: formal protocol drift")
        require(str(cfg.workflow.get("training_profile", "")) == "official60", "REF-D768: training profile drift")
        require(bool(cfg.workflow.get("formal_successful_update_contract", False)), "REF-D768: successful-update contract missing")
        require(int(cfg.workflow.expected_train_batches_per_epoch) == 100, "REF-D768: train batches drift")
        require(int(cfg.workflow.expected_successful_optimizer_updates) == 6000, "REF-D768: update count drift")
        require(int(cfg.workflow.max_amp_retries_per_batch) == 8, "REF-D768: AMP replay budget drift")
        require(bool(cfg.workflow.fail_on_amp_replay_exhaustion), "REF-D768: AMP replay must fail closed")
        require(bool(cfg.workflow.require_finite_train_loss), "REF-D768: finite-loss requirement missing")
        require(bool(cfg.workflow.get("selector_schedule_required", True)) is False, "REF-D768: selector schedule must be disabled")
        return
    validate_physical_time_optimizer(cfg, row["experiment_id"])
    if row["experiment_id"] == "REF-U384":
        selector = cfg.model.frame_selector
        require(int(selector.budget) == 384, "REF-U384: K must be 384")
        require(int(selector.dense_window_size) == 768, "REF-U384: dense T must be 768")
        require(str(selector.acquisition_policy) == "budget_calibrated_sampling_rate", "REF-U384: policy drift")
        require(float(selector.loss_weight_schedule.policy_alpha.start) == 0.0, "REF-U384: policy alpha must stay uniform")
        require(float(selector.loss_weight_schedule.policy_alpha.end) == 0.0, "REF-U384: policy alpha must stay uniform")
        return
    if row["experiment_id"] == "REF-MNV3FC384":
        selector = cfg.model.frame_selector
        require(int(selector.budget) == 384, "REF-MNV3FC384: K must be 384")
        require(int(selector.dense_window_size) == 768, "REF-MNV3FC384: dense T must be 768")
        require(str(selector.acquisition_policy) == "global_structured_topk", "REF-MNV3FC384: policy drift")
        require(bool(selector.parameter_free_selector), "REF-MNV3FC384: selector must stay train-free")
        require(str(selector.actionness_source_cfg.train_free_evidence_mode) == "frozen_feature_change", "REF-MNV3FC384: MobileNetV3 feature-change source drift")
        return
    selector = cfg.model.frame_selector
    require(int(selector.budget) == 384, f"{row['experiment_id']}: K must be 384")
    require(int(selector.dense_window_size) == 768, f"{row['experiment_id']}: dense T must be 768")
    require(str(cfg.workflow.get("formal_protocol", "")) == "duca_selected_axis_optimization_v1", f"{row['experiment_id']}: selected-axis formal protocol missing")
    if row["phase"] == "ON":
        require(str(selector.acquisition_policy) == "semantic_phase_sampling", f"{row['experiment_id']}: phase policy off")
        require(int(selector.semantic_phase_scaffold_budget) == 128, f"{row['experiment_id']}: scaffold budget drift")
        require(int(selector.semantic_phase_onset_budget) == 64, f"{row['experiment_id']}: onset budget drift")
        require(int(selector.semantic_phase_offset_budget) == 64, f"{row['experiment_id']}: offset budget drift")
        require(int(selector.semantic_phase_core_budget) == 128, f"{row['experiment_id']}: core budget drift")
    if row["ct"] == "ON":
        require(cfg.model.rpn_head.conv_cfg.type == "ContinuousTimeScaleAdaptiveConv1d", f"{row['experiment_id']}: CT conv missing")
    else:
        require(cfg.model.rpn_head.get("conv_cfg", None) is None, f"{row['experiment_id']}: CT conv unexpectedly enabled")
    amod = cfg.model.backbone.backbone.get("amod_config", {})
    require(bool(amod.get("enabled", False)) is (row["mod"] == "ON"), f"{row['experiment_id']}: MoD flag drift")
    if row["mod"] == "ON":
        require(list(amod.amod_layers) == [1, 3, 5, 7, 9, 11], f"{row['experiment_id']}: MoD layers drift")
    expected_mode = "signed_removal_utility" if row["taylor"] == "ON" else "abs_grad_times_input"
    require(str(selector.detector_contribution_mode) == expected_mode, f"{row['experiment_id']}: Taylor mode drift")
    schedule = selector.loss_weight_schedule
    if row["curriculum"] == "ON":
        require(schedule.shape == "cosine", f"{row['experiment_id']}: curriculum shape drift")
        require(int(schedule.policy_alpha.warmup_steps) == 1500, f"{row['experiment_id']}: curriculum warmup drift")
        require(int(schedule.policy_alpha.transition_steps) == 2000, f"{row['experiment_id']}: curriculum transition drift")
    else:
        require(schedule.shape == "linear", f"{row['experiment_id']}: linear schedule drift")
        require(int(schedule.policy_alpha.warmup_steps) == 0, f"{row['experiment_id']}: linear warmup drift")
        require(int(schedule.policy_alpha.transition_steps) == 3000, f"{row['experiment_id']}: linear transition drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DOC_MATRIX)
    args = parser.parse_args()
    rows = load_rows(args.matrix)
    require(len(rows) == 28, f"expected 28 matrix rows, got {len(rows)}")
    ids = [row["experiment_id"] for row in rows]
    require(len(ids) == len(set(ids)), "experiment_id values must be unique")
    identities = {(row["experiment_id"], row["seed"], row["config"]) for row in rows}
    require(len(identities) == 28, "training identities must be unique")
    by_id = {row["experiment_id"]: row for row in rows}
    for run, expected in EXPECTED_F.items():
        row = by_id.get(run)
        require(row is not None, f"missing {run}")
        require(
            (row["phase"], row["ct"], row["mod"], row["taylor"], row["curriculum"]) == expected,
            f"{run}: factor tuple drift",
        )
        require(int(row["seed"]) == 3407, f"{run}: DOE seed drift")
    for base, expected in EXPECTED_CANONICAL.items():
        phase, ct, mod, taylor, curriculum, seeds = expected
        observed = {
            int(row["seed"])
            for row in rows
            if row["category"] == "canonical" and row["experiment_id"].startswith(base + "-")
        }
        require(observed == seeds, f"{base}: canonical seed set drift")
        for seed in seeds:
            row = by_id[f"{base}-S{seed}"]
            require(
                (row["phase"], row["ct"], row["mod"], row["taylor"], row["curriculum"])
                == (phase, ct, mod, taylor, curriculum),
                f"{row['experiment_id']}: canonical factors drift",
            )
    for row in rows:
        validate_config(row)
    print("PASS H65-Pro fullmatrix: 28 unique jobs, configs, factors, and strict60 identities")


if __name__ == "__main__":
    main()
