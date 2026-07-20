from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "duca_protected_e2e_fixed384_official60.py"
)
OFFICIAL_CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "e2e_thumos_videomae_s_768x1_160_adapter.py"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def validate_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg = Config.fromfile(str(config_path))
    official = Config.fromfile(str(OFFICIAL_CONFIG))
    contract = cfg.duca_transition_only_contract
    selector = cfg.model.frame_selector
    temporal = selector.temporal_sampling_contract
    schedule = selector.loss_weight_schedule

    _require(contract.task == "offline_temporal_action_detection", "task must be offline TAD")
    _require(contract.online_tad is False and contract.streaming is False, "online/streaming claims are forbidden")
    _require(cfg.model.type == "ActionFormer", "detector must remain ActionFormer")
    _require(cfg.model.rpn_head.type == "ActionFormerHead", "head must remain ActionFormerHead")
    _require(_plain(cfg.model.rpn_head) == _plain(official.model.rpn_head), "official ActionFormerHead config changed")
    _require(int(cfg.model.backbone.backbone.total_frames) == 384, "AdaTAD backbone must consume exactly 384 hard frames")
    _require(int(cfg.model.projection.max_seq_len) == 384, "ActionFormer projection must consume 384 slots")
    _require(cfg.model.backbone.backbone.with_cp is False, "audited dynamic DDP path requires with_cp=False")
    _require(cfg.solver.static_graph is False, "dynamic selector graph cannot use static_graph")
    _require(cfg.solver.find_unused_parameters is True, "dynamic selector graph requires unused-parameter discovery")

    _require(selector.selector_variant == "transition_only", "selector must be transition-only")
    _require(selector.budget_mode == "fixed" and int(selector.budget) == 384, "budget must be exact fixed K=384")
    _require(int(selector.dense_window_size) == 768, "dense window must be T=768")
    _require(selector.acquisition_policy == "global_structured_topk", "policy must use global structured exact-K DP")
    _require(int(selector.max_unselected_hole) == 2, "dense max-hole must be G=2")
    _require(selector.hard_max_gap_repair is False, "post-hoc max-gap repair is forbidden")
    _require(selector.soft_max_gap_loss_enabled is False, "hard feasible family must own the gap contract")
    _require(selector.forbid_external_actionness is True, "external actionness is forbidden")
    _require(selector.forbid_raw_prediction_cache is True, "prediction cache is forbidden")
    _require(selector.counterfactual_utility_distillation_weight == 0.0, "detached counterfactual teacher is not part of this matrix")
    _require(selector.require_counterfactual_utility_teacher is False, "counterfactual teacher must be disabled")
    _require(selector.remap_gt_to_selected_axis is True, "selected-axis detector adapter must be explicit")
    _require(selector.detector_output_coordinate_space == "selected_axis_index", "detector axis must be disclosed")

    _require(int(temporal.hard_budget) == 384, "temporal contract budget mismatch")
    _require(int(temporal.dense_window_size) == 768, "temporal contract dense length mismatch")
    _require(int(temporal.max_unselected_hole_dense_candidates) == 2, "temporal contract max-hole mismatch")
    _require(int(temporal.dataset_feature_stride_source_frames) == 4, "THUMOS feature stride must be four frames")
    _require(int(temporal.dataset_sample_stride) == 1, "THUMOS sample stride must be one")
    _require(int(temporal.requested_max_source_frame_interval) == 15, "requested source-frame cap must be fifteen")
    _require(int(contract.max_selected_interval_source_frames) == 12, "quantized physical interval must be twelve frames")

    _require(int(cfg.workflow.end_epoch) == 60, "official protocol must run 60 epochs")
    _require(int(cfg.scheduler.max_epoch) == 60, "scheduler must end at epoch 60")
    _require(int(cfg.workflow.expected_train_batches_per_epoch) == 100, "steps per epoch must be 100")
    _require(int(cfg.workflow.expected_successful_optimizer_updates) == 6000, "successful updates must be 6000")
    _require(int(cfg.workflow.checkpoint_interval) == 5, "checkpoint interval must remain five")
    _require(int(cfg.workflow.primary_checkpoint_epoch) == 59, "primary checkpoint must be terminal epoch 59")
    _require(cfg.workflow.primary_checkpoint_state_key == "state_dict_ema", "primary state must be terminal EMA")
    _require(int(cfg.workflow.val_eval_interval) < 0, "intermediate test-set evaluation is forbidden")
    _require(int(cfg.workflow.val_start_epoch) > 60, "intermediate validation cannot select a checkpoint")
    _require(cfg.dataset.train.subset_name == "training", "official OpenTAD THUMOS train subset changed")
    _require(cfg.dataset.test.subset_name == "validation", "official OpenTAD THUMOS terminal evaluation subset changed")

    route = str(contract.route)
    homotopy_routes = {
        "DUCA_PROTECTED_E2E_FIXED384_OFFICIAL60",
        "DUCA_PROTECTED_E2E_HOMOTOPY025_FIXED384_OFFICIAL60",
        "DUCA_PROTECTED_E2E_HOMOTOPY_UNI_COMPANION025_FIXED384_OFFICIAL60",
    }
    if route in homotopy_routes:
        _require(selector.detector_gradient_mode == "protected_structured_transport", "protected bridge mode is missing")
        _require(float(selector.policy_hidden_gradient_scale) == 0.0, "main protected arm must detach ASFormer hidden")
        _require(
            selector.actionness_source_cfg.policy_hidden_gradient_scope == "none",
            "main protected arm must not replay an ASFormer policy layer",
        )
        _require(float(schedule.detector_gradient.start) == 0.0, "bridge must start disabled")
        _require(float(schedule.detector_gradient.end) == 0.25, "bridge endpoint must be preregistered at 0.25")
        _require(int(schedule.detector_gradient.warmup_steps) == 2100, "bridge warmup must end after policy homotopy")
        _require(int(schedule.detector_gradient.transition_steps) == 1500, "bridge ramp must be 1500 updates")
        companion_fraction = float(
            selector.get("training_uniform_companion_fraction", 0.0)
        )
        if route == "DUCA_PROTECTED_E2E_HOMOTOPY_UNI_COMPANION025_FIXED384_OFFICIAL60":
            _require(
                companion_fraction == 0.50,
                "Uni companion arm must replace exactly half of each multi-row training batch",
            )
            _require(
                contract.detector_gradient_updates
                == "transition_scorer_only_on_learned_rows",
                "Uni companion gradient ownership declaration changed",
            )
            _require(
                contract.inference_uses_learned_policy_only is True
                and contract.inference_extra_companion_cost is False,
                "Uni companion must remain training-only",
            )
        else:
            _require(
                companion_fraction == 0.0,
                "non-companion homotopy arm unexpectedly enables uniform views",
            )
    elif route == "DUCA_PROTECTED_E2E_DIRECT025_FIXED384_OFFICIAL60":
        _require(selector.detector_gradient_mode == "protected_structured_transport", "direct arm requires protected bridge")
        _require(float(selector.policy_hidden_gradient_scale) == 0.0, "direct arm must detach ASFormer hidden")
        _require(
            float(schedule.policy_alpha.start) == 1.0
            and float(schedule.policy_alpha.end) == 1.0,
            "direct arm must use the learned policy from step zero",
        )
        _require(
            int(schedule.policy_alpha.warmup_steps) == 0
            and int(schedule.policy_alpha.transition_steps) == 0,
            "direct arm cannot hide a policy warmup",
        )
        _require(
            float(schedule.detector_gradient.start) == 0.25
            and float(schedule.detector_gradient.end) == 0.25,
            "direct arm bridge must remain fixed at 0.25",
        )
        _require(
            int(schedule.detector_gradient.warmup_steps) == 0
            and int(schedule.detector_gradient.transition_steps) == 0,
            "direct arm cannot hide a detector-gradient warmup",
        )
        _require(
            float(selector.get("training_uniform_companion_fraction", 0.0)) == 0.0,
            "direct arm unexpectedly enables uniform companion rows",
        )
    elif route == "DUCA_TRANSITION_NO_BRIDGE_FIXED384_OFFICIAL60":
        _require(selector.detector_gradient_mode == "none", "no-bridge arm must disable bridge code")
        _require(float(schedule.detector_gradient.end) == 0.0, "no-bridge arm must have zero bridge weight")
        _require(float(selector.policy_hidden_gradient_scale) == 0.0, "no-bridge arm must protect ASFormer")
    elif route == "DUCA_EXACT_UNIFORM_FIXED384_OFFICIAL60":
        _require(selector.detector_gradient_mode == "none", "uniform control must disable detector bridge")
        _require(float(selector.inference_policy_alpha) == 0.0, "uniform control must remain exact uniform")
        _require(float(schedule.policy_alpha.start) == 0.0 and float(schedule.policy_alpha.end) == 0.0, "uniform policy schedule changed")
        _require(float(selector.policy_hidden_gradient_scale) == 0.0, "uniform control cannot expose ASFormer")
    elif route == "DUCA_PROTECTED_E2E_RHO_FIXED384_OFFICIAL60":
        _require(selector.detector_gradient_mode == "protected_structured_transport", "rho arm must use the protected bridge")
        _require(0.0 < float(selector.policy_hidden_gradient_scale) <= 0.05, "rho must be fixed and at most 0.05")
        _require(
            selector.actionness_source_cfg.policy_hidden_gradient_scope
            == "asformer_last_encoder_layer",
            "rho arm must expose only the last official ASFormer encoder layer",
        )
        _require(
            contract.detector_gradient_updates
            == "transition_scorer_and_official_asformer_last_encoder_layer_only",
            "rho gradient ownership declaration changed",
        )
        _require(
            contract.asformer_trunk_detector_gradient is True
            and contract.earlier_asformer_detector_gradient is False
            and contract.action_head_detector_gradient is False,
            "rho ASFormer gradient ownership metadata is internally inconsistent",
        )
    else:
        raise AssertionError(f"unregistered official-60 route {route!r}")

    _require(contract.paper_claim_allowed is False, "unrun configuration cannot make paper claims")
    _require(contract.metric_claim_allowed is False, "unrun configuration cannot make metric claims")
    return {
        "ok": True,
        "status": "tested_p0_config_contract",
        "config": str(Path(config_path)),
        "route": route,
        "task": str(contract.task),
        "budget": int(selector.budget),
        "dense_window_size": int(selector.dense_window_size),
        "max_unselected_hole_dense_candidates": int(selector.max_unselected_hole),
        "max_selected_interval_source_frames": int(contract.max_selected_interval_source_frames),
        "epochs": int(cfg.workflow.end_epoch),
        "successful_optimizer_updates": int(cfg.workflow.expected_successful_optimizer_updates),
        "primary_checkpoint_epoch": int(cfg.workflow.primary_checkpoint_epoch),
        "primary_checkpoint_state_key": str(cfg.workflow.primary_checkpoint_state_key),
        "detector_gradient_mode": str(selector.detector_gradient_mode),
        "policy_hidden_gradient_scale": float(selector.policy_hidden_gradient_scale),
        "training_uniform_companion_fraction": float(
            selector.get("training_uniform_companion_fraction", 0.0)
        ),
        "paper_claim_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        payload = validate_config(args.config)
    except Exception as exc:
        payload = {
            "ok": False,
            "config": str(args.config),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
        exit_code = 1
    else:
        exit_code = 0
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
