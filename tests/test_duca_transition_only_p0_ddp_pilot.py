from __future__ import annotations

import pytest
from mmengine.config import Config

from tools.bata.validate_duca_transition_only_p0_ddp_pilot import (
    EXPECTED_STEPS,
    PILOT_CONFIGS,
    VARIANT_ORDER,
    validate_probe,
)


def _probe(variant: str) -> dict:
    budget_vectors = [
        [384, 384],
        [353, 384],
        [300, 312],
        [384, 384],
        [350, 384],
        [286, 310],
        [384, 384],
        [360, 384],
        [301, 333],
        [384, 384],
    ]
    steps = []
    for index, effective_budget in enumerate(budget_vectors):
        step = {
            "effective_budget": effective_budget,
            "requested_budget": [384, 384],
            "detector_gradient_weight": 0.0 if index < 3 else 0.5,
            "policy_mix_alpha": 0.0 if index < 3 else 0.75,
        }
        if variant == "transition_counterfactual":
            step["counterfactual"] = {
                "candidate_count": 0 if index == 0 else 4,
                "finite": True,
                "teacher_kind": "detached_hard_one_swap_official_actionformer_cls_plus_reg",
            }
        steps.append(step)
    return {
        "schema_version": "duca_training_probe_v1",
        "attempted_steps": EXPECTED_STEPS,
        "successful_optimizer_steps": EXPECTED_STEPS,
        "skipped_optimizer_steps": 0,
        "finite_loss_steps": EXPECTED_STEPS,
        "finite_gradient_steps": EXPECTED_STEPS,
        "static_graph": False,
        "find_unused_parameters": True,
        "world_size": 1,
        "parameter_group_coverage": {
            group: {"trainable": 2, "gradient_seen": 1}
            for group in ("backbone", "coarse_probe", "selector", "detector_head")
        },
        "gradient_never_seen": [],
        "selector_steps": steps,
        "max_cuda_memory_mb": 8471.0,
    }


def test_pilot_configs_preserve_dynamic_ddp_and_disable_only_pilot_side_effects() -> None:
    for variant, path in PILOT_CONFIGS.items():
        cfg = Config.fromfile(path)
        assert cfg.model.backbone.backbone.with_cp is False, variant
        assert cfg.solver.static_graph is False, variant
        assert cfg.solver.find_unused_parameters is True, variant
        assert cfg.workflow.max_train_iters == EXPECTED_STEPS, variant
        assert cfg.workflow.disable_checkpoint is True, variant
        assert cfg.workflow.end_epoch == 1, variant
        assert cfg.workflow.val_start_epoch > cfg.workflow.end_epoch, variant
        assert cfg.workflow.checkpoint_interval == 5, variant


def test_pilot_validator_accepts_required_batch_and_schedule_coverage() -> None:
    for variant in VARIANT_ORDER:
        summary = validate_probe(_probe(variant), variant)
        assert summary["successful_optimizer_steps"] == EXPECTED_STEPS
        assert all(summary["budget_coverage"].values())


def test_pilot_validator_rejects_static_graph_and_missing_parameter_path() -> None:
    static = _probe("direct")
    static["static_graph"] = True
    with pytest.raises(AssertionError, match="static_graph"):
        validate_probe(static, "direct")

    disconnected = _probe("transition_beta0")
    disconnected["parameter_group_coverage"]["coarse_probe"]["gradient_seen"] = 0
    with pytest.raises(AssertionError, match="coarse_probe never received gradient"):
        validate_probe(disconnected, "transition_beta0")


def test_pilot_validator_requires_both_counterfactual_candidate_paths() -> None:
    missing_zero = _probe("transition_counterfactual")
    for step in missing_zero["selector_steps"]:
        step["counterfactual"]["candidate_count"] = 4
    with pytest.raises(AssertionError, match="zero-candidate"):
        validate_probe(missing_zero, "transition_counterfactual")
