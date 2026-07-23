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
    contract = cfg.duca_density_contract
    selector = cfg.model.frame_selector
    schedule = selector.loss_weight_schedule
    policy = str(selector.acquisition_policy)

    _require(contract.task == "offline_temporal_action_detection", "task must be offline TAD")
    _require(cfg.model.type == "ActionFormer", "detector must remain ActionFormer")
    _require(cfg.model.rpn_head.type == "ActionFormerHead", "head must remain ActionFormerHead")
    _require(
        _plain(cfg.model.rpn_head) == _plain(official.model.rpn_head),
        "official ActionFormerHead config changed",
    )
    _require(int(selector.dense_window_size) == 768, "selector input must remain T=768")
    _require(selector.budget_mode == "fixed" and int(selector.budget) == 384, "budget must be exact K=384")
    _require(int(cfg.model.backbone.backbone.total_frames) == 384, "VideoMAE must consume K=384")
    _require(int(cfg.model.projection.max_seq_len) == 384, "projection must consume K=384")
    _require(
        policy
        in {
            "continuous_density_transport",
            "continuous_mixture_density_transport",
        },
        "config is not a registered continuous density policy",
    )
    _require(selector.transition_objective == "gaussian_mass", "fixed burst decoding must not confound density transport")
    _require(selector.detector_gradient_mode == "density_transport_st", "real detector gradient bridge is missing")
    _require(
        contract.backward
        == "hard_anchored_local_temporal_slope_from_inverse_cdf",
        "density detector gradient must be anchored at actual hard observations",
    )
    _require(float(selector.policy_hidden_gradient_scale) == 0.0, "protected matrix must stop detector gradients before ASFormer")
    _require(
        selector.actionness_source_cfg.policy_hidden_gradient_scope == "none",
        "protected matrix cannot replay an ASFormer policy layer",
    )
    _require(selector.hard_max_gap_repair is False, "post-hoc repair is forbidden")
    _require(
        selector.get("temporal_sampling_contract", None) is None,
        "density matrix must not inherit a hard temporal contract",
    )
    _require(selector.forbid_external_actionness is True, "external actionness is forbidden")
    _require(selector.forbid_raw_prediction_cache is True, "prediction cache is forbidden")
    _require(selector.counterfactual_utility_distillation_weight == 0.0, "counterfactual teacher is not part of this matrix")
    _require(selector.require_counterfactual_utility_teacher is False, "counterfactual teacher must be disabled")
    _require(float(schedule.policy_alpha.start) == 0.0 and float(schedule.policy_alpha.end) == 1.0, "density homotopy must be uniform-to-learned")
    _require(float(schedule.detector_gradient.start) == 0.0 and float(schedule.detector_gradient.end) == 0.25, "detector-gradient endpoints changed")

    hard_enabled = selector.max_unselected_hole is not None
    soft_enabled = bool(selector.soft_max_gap_loss_enabled)
    _require(not (hard_enabled and soft_enabled), "hard and soft max-gap arms must remain separate")
    if bool(contract.hard_max_gap_enabled):
        _require(hard_enabled and int(selector.max_unselected_hole) == 14, "hard-max arm must use G=14")
        _require(not soft_enabled and float(selector.loss_weights.max_gap_hole) == 0.0, "hard-max arm cannot add soft gap loss")
    elif bool(contract.soft_max_gap_enabled):
        _require(not hard_enabled, "soft-max arm cannot enforce a hard cap")
        _require(soft_enabled and int(selector.max_gap_loss_max_unselected_hole) == 14, "soft-max target must be G=14")
        _require(float(selector.loss_weights.max_gap_hole) > 0.0, "soft-max arm requires a positive gap loss")
    else:
        _require(not hard_enabled and not soft_enabled, "no-max arm unexpectedly has a max-gap constraint")
        _require(float(selector.loss_weights.max_gap_hole) == 0.0, "no-max arm unexpectedly has gap loss")

    mixture = policy == "continuous_mixture_density_transport"
    _require(
        (str(contract.density_model) == "boundary_uncertainty_context_mixture")
        is mixture,
        "density-model declaration disagrees with acquisition policy",
    )
    if mixture:
        _require(
            contract.mixture_gate_inputs
            == "normalized_component_entropy_peak_center_spread",
            "mixture gate must be invariant to component-logit offsets",
        )
        _require(
            contract.uses_absolute_asformer_hidden_for_context is True,
            "mixture context must disclose absolute ASFormer hidden use",
        )
    _require(int(cfg.workflow.end_epoch) == 60, "official protocol must run 60 epochs")
    _require(int(cfg.workflow.expected_successful_optimizer_updates) == 6000, "official protocol must run 6000 updates")
    _require(int(cfg.workflow.primary_checkpoint_epoch) == 59, "primary checkpoint must be epoch 59")
    _require(cfg.workflow.primary_checkpoint_state_key == "state_dict_ema", "primary state must be terminal EMA")
    _require(int(cfg.workflow.val_eval_interval) < 0, "intermediate validation is forbidden")
    _require(cfg.dataset.train.subset_name == "training", "training split changed")
    _require(cfg.dataset.test.subset_name == "validation", "validation split changed")
    _require(contract.paper_claim_allowed is False, "unrun config cannot make paper claims")

    return {
        "ok": True,
        "status": "density_transport_official60_config_validated",
        "config": str(Path(config_path).resolve()),
        "route": str(contract.route),
        "policy": policy,
        "density_model": str(contract.density_model),
        "budget": int(selector.budget),
        "hard_max_unselected_hole": (
            None
            if selector.max_unselected_hole is None
            else int(selector.max_unselected_hole)
        ),
        "soft_max_gap_enabled": soft_enabled,
        "soft_max_unselected_hole_target": (
            int(selector.max_gap_loss_max_unselected_hole) if soft_enabled else None
        ),
        "detector_gradient_mode": str(selector.detector_gradient_mode),
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
        payload = {
            "ok": False,
            "config": str(args.config),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
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
