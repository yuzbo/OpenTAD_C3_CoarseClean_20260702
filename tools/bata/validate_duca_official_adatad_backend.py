from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_DEFAULT = "configs/adatad/thumos/duca_online_official_adatad_backend_full_train.py"
OFFICIAL_BASE_CONFIG = "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _as_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _as_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(item) for item in value]
    return value


def _load(path: str | Path) -> Config:
    return Config.fromfile(str(path))


def _config_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").lower()


def _loss_schedule_summary(selector: Any, contract: Any) -> dict[str, Any]:
    _require(contract.loss_schedule_policy == "progressive_joint", "main config must declare progressive_joint loss schedule")
    _require(
        getattr(contract, "loss_schedule_step_update", "") == "optimizer_step",
        "loss schedule must advance after optimizer.step, not on raw forward count",
    )
    schedule = selector.loss_weight_schedule
    _require(schedule.type == "progressive_joint", "selector must use progressive_joint loss_weight_schedule")
    _require(str(schedule.get("shape", "linear")) in {"linear", "cosine"}, "loss schedule shape must be linear or cosine")
    _require(int(schedule.warmup_steps) >= 0, "loss schedule warmup_steps must be non-negative")
    _require(int(schedule.transition_steps) > 0, "loss schedule transition_steps must be positive")
    _require(
        int(getattr(contract, "loss_schedule_total_steps", 0)) > 0,
        "loss schedule must declare a positive total training-step horizon",
    )
    _require(
        int(getattr(contract, "loss_schedule_steps_per_epoch", 0)) > 0,
        "loss schedule must declare positive expected steps per epoch",
    )
    _require(
        0.0 <= float(getattr(contract, "loss_schedule_warmup_fraction", -1.0)) < 1.0,
        "loss schedule warmup must be declared as a fraction of total training",
    )
    _require(
        0.0 < float(getattr(contract, "loss_schedule_transition_fraction", 0.0)) <= 1.0,
        "loss schedule transition must be declared as a fraction of total training",
    )
    _require(float(schedule.actionness.start) > float(schedule.actionness.end), "actionness loss must decay after warmup")
    _require(
        getattr(contract, "detector_loss_always_trains_backend", False) is True,
        "detector loss must train the backend from the start",
    )
    _require(
        getattr(contract, "detector_gradient_bridge_enabled_after_schedule_transition", False) is True,
        "detector-to-selector gradient bridge must be explicitly scheduled",
    )
    _require(
        float(schedule.detector_gradient.start) == 0.0,
        "detector-to-selector gradient bridge must be disabled at schedule start",
    )
    _require(
        float(schedule.detector_gradient.end) == 1.0,
        "detector-to-selector gradient bridge must be fully enabled after schedule transition",
    )
    _require(float(schedule.detector_utility.start) == 0.0, "detector utility loss must be disabled at schedule start")
    _require(float(schedule.detector_utility.end) > 0.0, "detector utility loss must be enabled after transition")
    _require(float(schedule.boundary.start) > float(schedule.actionness.start), "boundary supervision must dominate actionness at schedule start")
    _require(float(schedule.boundary.end) >= float(schedule.boundary.start), "boundary supervision must stay active after transition")
    _require(float(schedule.actionness.end) < float(schedule.boundary.end), "actionness must remain auxiliary to boundary supervision")
    _require(float(schedule.hole.start) == 0.0, "action-coverage hole loss must be disabled at schedule start")
    _require(float(schedule.hole.end) == 0.0, "main DUCA selector must not optimize action-coverage hole loss")
    return {
        "loss_schedule_policy": str(schedule.type),
        "loss_schedule_step_update": str(contract.loss_schedule_step_update),
        "loss_schedule_shape": str(schedule.get("shape", "linear")),
        "loss_schedule_total_steps": int(contract.loss_schedule_total_steps),
        "loss_schedule_steps_per_epoch": int(contract.loss_schedule_steps_per_epoch),
        "loss_schedule_warmup_fraction": float(contract.loss_schedule_warmup_fraction),
        "loss_schedule_transition_fraction": float(contract.loss_schedule_transition_fraction),
        "loss_schedule_warmup_steps": int(schedule.warmup_steps),
        "loss_schedule_transition_steps": int(schedule.transition_steps),
        "loss_schedule_actionness_start": float(schedule.actionness.start),
        "loss_schedule_actionness_end": float(schedule.actionness.end),
        "loss_schedule_boundary_start": float(schedule.boundary.start),
        "loss_schedule_boundary_end": float(schedule.boundary.end),
        "loss_schedule_detector_loss_always_on": True,
        "loss_schedule_detector_gradient_bridge_scheduled": True,
        "loss_schedule_detector_gradient_start": float(schedule.detector_gradient.start),
        "loss_schedule_detector_gradient_end": float(schedule.detector_gradient.end),
        "loss_schedule_detector_utility_start": float(schedule.detector_utility.start),
        "loss_schedule_detector_utility_end": float(schedule.detector_utility.end),
        "loss_schedule_hole_start": float(schedule.hole.start),
        "loss_schedule_hole_end": float(schedule.hole.end),
    }


def validate_config(
    config_path: str = CONFIG_DEFAULT,
    *,
    max_budget: int = 384,
    require_online_c3_actionness: bool = True,
) -> dict[str, Any]:
    config_path = str(config_path)
    max_budget = int(max_budget)
    cfg = _load(config_path)
    official = _load(str(ROOT / OFFICIAL_BASE_CONFIG))
    contract = cfg.duca_online_main_contract
    selector = cfg.model.frame_selector
    head = cfg.model.rpn_head
    model_text = repr(cfg.model).lower()
    full_text = _config_text(config_path)

    _require(contract.official_adatad_backend is True, "contract must declare official_adatad_backend=True")
    _require(contract.main_method_candidate is True, "contract must declare main_method_candidate=True")
    _require(contract.diagnostic_only is False, "main official backend config must not be diagnostic_only")
    _require(contract.no_ledger_decision is True, "main config must be online/no-ledger")
    _require(contract.pre_backbone_plugin is True, "DUCA must be declared as pre-backbone plugin")
    _require(
        getattr(contract, "coarse_actionness_dominates_initial_training", True) is False,
        "main config must not declare coarse actionness as the dominant selector objective",
    )
    _require(
        getattr(contract, "state_transition_boundary_dominates_selection", False) is True,
        "main config must declare state-transition/boundary-first selection",
    )
    _require(contract.changes_detector_head is False, "main config must not change detector head")
    _require(contract.changes_loss_assignment is False, "main config must not change detector assignment/loss")
    _require(contract.physical_grid_actionformer_required is False, "main config must not require physical-grid head")
    _require(cfg.model.type == "ActionFormer", "official AdaTAD backend must keep ActionFormer detector")
    _require(selector.type == "DucaOnlineFrameSelector", "main config must use DucaOnlineFrameSelector")
    _require(selector.budget_mode == "fixed", "fixed official config must explicitly set budget_mode=fixed")
    _require(max_budget > 0, "max_budget must be positive")
    _require(int(selector.budget) <= max_budget, f"selector budget must be <={max_budget}")
    _require(int(selector.dense_window_size) == 768, "candidate dense window must stay 768")
    _require(selector.coordinate_space == "original_time", "selected positions must be original-time")
    _require(selector.detector_output_coordinate_space == "selected_axis_index", "detector outputs must be selected-axis before wrapper remap")
    _require(selector.remap_gt_to_selected_axis is True, "official head path must remap GT to selected-axis")
    _require(selector.no_ledger_decision is True, "selector must be online/no-ledger")
    _require(selector.forbid_ledger is True, "selector must forbid ledger payload")
    _require(head.type == "ActionFormerHead", "official backend must use ActionFormerHead")
    _require(_as_plain(head) == _as_plain(official.model.rpn_head), "ActionFormerHead config must match official base")
    _require("physical_grid_actionformer" not in head, "main config must not enable physical-grid ActionFormer")
    _require("bata_value_transport" not in full_text, "main config must not use value-transport ledger sampling")
    _require("ledger_path" not in model_text, "model must not include ledger_path")
    _require("load_from_raw_predictions': true" not in repr(cfg.inference).lower(), "raw prediction loading must be disabled")
    _require("save_raw_prediction': true" not in repr(cfg.inference).lower(), "raw prediction saving must be disabled")
    _require(int(cfg.window_size) == int(selector.budget), "detector-consumed DUCA sequence length must match selector budget")
    _require(int(cfg.dense_window_size) == 768, "candidate dense observation length must be 768")
    _require(int(cfg.window_size) % 16 == 0, "detector-consumed DUCA sequence length must be divisible by 16")
    _require(int(cfg.chunk_num) == int(cfg.window_size) // 16, "VideoMAE chunk count must match selected budget/16")
    _require(
        int(cfg.model.backbone.backbone.total_frames) == int(cfg.window_size),
        "VideoMAE backend must consume selected budget frames",
    )
    _require(int(cfg.model.projection.max_seq_len) == int(cfg.window_size), "projection max_seq_len must match selected budget")
    _require(cfg.dataset.train.pipeline[2].method == "random_trunc", "train LoadFrames must stay online random_trunc, not ledger")
    _require(cfg.dataset.val.pipeline[2].method == "sliding_window", "val LoadFrames must stay online sliding_window, not ledger")
    _require(cfg.dataset.test.pipeline[2].method == "sliding_window", "test LoadFrames must stay online sliding_window, not ledger")

    source_cfg = selector.actionness_source_cfg
    if require_online_c3_actionness:
        _require(
            "ducaexternalactionnessfromjsonl" not in full_text,
            "main config must not load external X3D/JSONL actionness",
        )
        _require("duca_x3d" not in full_text, "main config must not reference DUCA_X3D train-free baseline inputs")
        _require(
            selector.get("external_actionness_meta_key", None) in (None, ""),
            "main DUCA-JCT selector must not read external p_action metas",
        )
        _require(
            selector.get("external_actionness_logits_meta_key", None) in (None, ""),
            "main DUCA-JCT selector must not read external actionness logits metas",
        )
        _require(
            selector.get("external_actionness_provenance_meta_key", None) in (None, ""),
            "main DUCA-JCT selector must not read external actionness provenance metas",
        )
        _require(
            selector.get("external_actionness_source_meta_key", None) in (None, ""),
            "main DUCA-JCT selector must not read external actionness source metas",
        )
        _require(
            selector.get("require_external_actionness", False) is False,
            "main DUCA-JCT selector must not require external actionness",
        )
        _require(
            contract.actionness_source == "online_trainable_c3_coarse_probe",
            "main source must be online C3 coarse probe",
        )
        _require(contract.acquisition_policy == "duca_center_radius_st_acquisition", "unexpected acquisition policy")
        _require(contract.budget_policy == "fixed_budget", "fixed config must declare fixed_budget special case")
        _require(source_cfg.type == "C3CoarseProbeActionnessSource", "main source must be online C3 coarse probe module")
        _require(source_cfg.probe_model == "official-action-seg", "main method must use official action-seg probe")
        _require(source_cfg.get("official_action_seg_backend") == "official_asformer", "ASFormer must use official code")
        _require(
            source_cfg.get("tcn_variant", None) in (None, "", "official_asformer"),
            "official-action-seg main path must not carry a private/lightweight TCN variant",
        )
        _require(source_cfg.get("tcn_variant", None) != "asformer_lite", "main method forbids asformer_lite")
        _require(source_cfg.get("trainable") is True, "coarse probe must be trainable for the main end-to-end config")
        _require(source_cfg.get("frozen") is False, "coarse probe must not be frozen for the main end-to-end config")
        _require(float(selector.loss_weights.get("actionness", 0.0)) > 0.0, "main config must keep actionness BCE as auxiliary calibration")
        _require(
            float(selector.loss_weights.get("boundary", 0.0)) > float(selector.loss_weights.get("actionness", 0.0)),
            "main DUCA selector must make state-transition/boundary supervision dominate actionness",
        )
        _require(
            float(selector.loss_weights.get("hole", 0.0)) == 0.0,
            "main DUCA selector must disable action-coverage hole loss",
        )
        _require(contract.coarse_probe_joint_trainable is True, "contract must declare joint-trainable coarse probe")
        _require(contract.runtime_profile_available is True, "contract must expose compute/latency profiling")
        for key in ("uses_labels", "uses_teacher", "uses_gt", "uses_prediction_cache"):
            _require(source_cfg.get(key) is False, f"actionness source must set {key}=False")
    schedule_summary = _loss_schedule_summary(selector, contract)

    summary = {
        "ok": True,
        "config_path": config_path,
        "official_base_config": OFFICIAL_BASE_CONFIG,
        "official_adatad_backend": True,
        "model_type": str(cfg.model.type),
        "selector_type": str(selector.type),
        "rpn_head_type": str(head.type),
        "rpn_head_matches_official_base": True,
        "physical_grid_actionformer_enabled": False,
        "uses_ledger_for_decision": False,
        "uses_external_actionness": False,
        "requires_external_actionness": False,
        "train_free_x3d_is_main_method": False,
        "budget_lte_384": int(selector.budget) <= 384,
        "strict_budget_lte_384": int(selector.budget) <= 384,
        "budget_lte_max": int(selector.budget) <= max_budget,
        "validator_max_budget": max_budget,
        "budget": int(selector.budget),
        "dense_window_size": int(selector.dense_window_size),
        "detector_consumed_length": int(cfg.window_size),
        "changes_detector_head": False,
        "changes_loss_assignment": False,
        "pre_backbone_plugin": True,
        "selected_positions_unit": str(selector.selected_positions_unit),
        "detector_output_coordinate_space": str(selector.detector_output_coordinate_space),
    }
    summary.update(schedule_summary)
    if require_online_c3_actionness:
        summary.update(
            {
                "actionness_source": str(contract.actionness_source),
                "coarse_probe_model": str(source_cfg.probe_model),
                "coarse_probe_tcn_variant": str(source_cfg.get("tcn_variant", "")),
                "coarse_probe_official_backend": str(source_cfg.get("official_action_seg_backend", "")),
                "coarse_probe_joint_trainable": bool(contract.coarse_probe_joint_trainable),
                "runtime_profile_available": bool(contract.runtime_profile_available),
                "acquisition_policy": str(contract.acquisition_policy),
                "budget_policy": str(contract.budget_policy),
            }
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--max-budget", type=int, default=int(os.environ.get("DUCA_VALIDATOR_MAX_BUDGET", "384")))
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        summary = validate_config(args.config, max_budget=int(args.max_budget))
    except Exception as exc:
        summary = {
            "ok": False,
            "config_path": str(args.config),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return 1
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
