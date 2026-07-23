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


def validate_config(config_path: str | Path) -> dict[str, Any]:
    cfg = Config.fromfile(str(config_path))
    official = Config.fromfile(str(OFFICIAL_CONFIG))
    contract = cfg.duca_sampling_rate_contract
    selector = cfg.model.frame_selector
    schedule = selector.loss_weight_schedule
    contribution_components = str(contract.get("contribution_components", "none"))
    adapts_last_asformer_layer = bool(contract.get("asformer_last_layer_adapted", False))
    adapts_full_asformer_encoder = bool(contract.get("asformer_full_encoder_adapted", False))
    _require(
        not (adapts_last_asformer_layer and adapts_full_asformer_encoder),
        "an arm cannot adapt both the last layer and full ASFormer encoder separately",
    )

    _require(contract.task == "offline_temporal_action_detection", "task must be offline TAD")
    _require(cfg.model.type == "ActionFormer", "detector must remain ActionFormer")
    _require(cfg.model.rpn_head.type == "ActionFormerHead", "head must remain ActionFormerHead")
    _require(_plain(cfg.model.rpn_head) == _plain(official.model.rpn_head), "official ActionFormerHead config changed")
    _require(int(selector.dense_window_size) == 768, "selector input must remain T=768")
    _require(selector.budget_mode == "fixed" and int(selector.budget) == 384, "budget must be exact K=384")
    _require(int(cfg.model.backbone.backbone.total_frames) == 384, "VideoMAE must consume K=384")
    _require(int(cfg.model.projection.max_seq_len) == 384, "projection must consume K=384")
    _require(
        selector.acquisition_policy == "budget_calibrated_sampling_rate",
        "config is not the budget-calibrated sampling-rate policy",
    )
    _require(
        contract.acquisition == "bounded_per_frame_retention_rate_plus_deterministic_cumulative_sampling",
        "sampling-rate acquisition declaration changed",
    )
    _require(
        contract.hard_forward == "exact_k_strictly_increasing_original_time_observations",
        "hard forward contract changed",
    )
    _require(
        contract.backward == "hard_anchored_local_cumulative_rate_slope",
        "backward contract changed",
    )
    _require(selector.detector_gradient_mode == "density_transport_st", "hard-anchored detector bridge is missing")
    _require(selector.hard_max_gap_repair is False, "sampling-rate arm must not apply post-hoc max-gap repair")
    _require(selector.max_unselected_hole is None, "base sampling-rate arm must not use a hard max-gap")
    _require(selector.soft_max_gap_loss_enabled is False, "base sampling-rate arm must not use a soft max-gap")
    _require(float(selector.loss_weights.max_gap_hole) == 0.0, "base sampling-rate arm unexpectedly uses gap loss")
    _require(selector.forbid_external_actionness is True, "external actionness is forbidden")
    _require(selector.forbid_raw_prediction_cache is True, "prediction cache is forbidden")
    _require(selector.counterfactual_utility_distillation_weight == 0.0, "counterfactual teacher is not part of this matrix")
    _require(selector.require_counterfactual_utility_teacher is False, "counterfactual teacher must be disabled")
    _require(float(selector.density_coverage_floor) > 0.0, "rate policy requires a nonzero coverage floor")
    _require(float(schedule.policy_alpha.start) == 0.0 and float(schedule.policy_alpha.end) == 1.0, "policy must be uniform-to-rate curriculum")
    _require(float(schedule.detector_gradient.start) == 0.0 and float(schedule.detector_gradient.end) == 0.25, "detector bridge endpoints changed")
    _require(float(schedule.detector_contribution.start) == 0.0 and float(schedule.detector_contribution.end) == 1.0, "contribution curriculum endpoints changed")

    _require(str(selector.sampling_rate_utility_components) == contribution_components, "rate utility inputs disagree with contract")
    _require(str(selector.detector_contribution_components) == contribution_components, "contribution prediction targets disagree with contract")
    if contribution_components == "none":
        _require(float(selector.detector_contribution_distillation_weight) == 0.0, "rate-only control must disable contribution distillation")
    else:
        _require(float(selector.detector_contribution_distillation_weight) > 0.0, "contribution arm requires positive distillation weight")
        _require(float(selector.training_uniform_companion_fraction) > 0.0, "contribution teacher requires uniform companion rows")

    scope = str(selector.actionness_source_cfg.policy_hidden_gradient_scope)
    if adapts_last_asformer_layer:
        _require(scope == "asformer_last_encoder_layer", "adapted arm must expose only the last ASFormer layer")
        _require(float(schedule.asformer_adapt.start) == 0.0 and float(schedule.asformer_adapt.end) == 1.0, "adapted arm must use a gated ASFormer schedule")
    elif adapts_full_asformer_encoder:
        _require(scope == "asformer_full_encoder", "full adaptation arm must expose the full ASFormer encoder")
        _require(float(schedule.asformer_adapt.start) == 0.0 and float(schedule.asformer_adapt.end) == 1.0, "full adaptation arm must use a gated ASFormer schedule")
    else:
        _require(scope == "none", "protected arm must stop detector gradients before ASFormer")
        _require(float(schedule.asformer_adapt.end) == 0.0, "protected arm cannot adapt ASFormer from detector utility")

    _require(int(cfg.workflow.end_epoch) == 60, "official protocol must run 60 epochs")
    _require(int(cfg.workflow.expected_successful_optimizer_updates) == 6000, "official protocol must run 6000 updates")
    _require(int(cfg.workflow.primary_checkpoint_epoch) == 59, "primary checkpoint must be epoch 59")
    _require(cfg.workflow.primary_checkpoint_state_key == "state_dict_ema", "primary state must be terminal EMA")
    _require(int(cfg.workflow.val_eval_interval) < 0, "intermediate validation is forbidden")
    _require(cfg.dataset.train.subset_name == "training", "training split changed")
    _require(cfg.dataset.test.subset_name == "validation", "validation split changed")
    _require(contract.inference_teacher_free is True, "inference must not consume contribution teacher targets")
    _require(contract.paper_claim_allowed is False, "unrun config cannot make paper claims")

    return {
        "ok": True,
        "status": "sampling_rate_official60_config_validated",
        "config": str(Path(config_path).resolve()),
        "route": str(contract.route),
        "policy": str(selector.acquisition_policy),
        "budget": int(selector.budget),
        "contribution_components": contribution_components,
        "asformer_last_layer_adapted": adapts_last_asformer_layer,
        "asformer_full_encoder_adapted": adapts_full_asformer_encoder,
        "official_validation_comparable": True,
        "paper_claim_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        payload = validate_config(args.config)
    except Exception as exc:
        payload = {"ok": False, "config": str(args.config), "error_type": exc.__class__.__name__, "error": str(exc)}
        code = 1
    else:
        code = 0
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
