from __future__ import annotations

import importlib.util
from pathlib import Path

from mmengine.config import Config

from tools.bata.duca_p0_training import formal_training_contract
from tools.bata import duca_selected_axis_training
from tools.bata.validate_duca_sampling_rate_official60 import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "adatad" / "thumos"


def _should_eval_epoch(epoch: int, workflow) -> bool:
    """Load the schedule module without importing the Torch-dependent utils package."""

    spec = importlib.util.spec_from_file_location(
        "duca_train_schedule_test",
        ROOT / "opentad" / "utils" / "train_schedule.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return bool(module.should_eval_epoch(epoch, workflow))
VARIANTS = {
    "rate_exact_uniform": "duca_sampling_rate_exact_uniform_fixed384_official60.py",
    "rate_only": "duca_sampling_rate_fixed384_official60.py",
    "rate_cls": "duca_sampling_rate_cls_fixed384_official60.py",
    "rate_reg": "duca_sampling_rate_reg_fixed384_official60.py",
    "rate_both": "duca_sampling_rate_both_fixed384_official60.py",
    "rate_both_asformer_adapt": "duca_sampling_rate_both_asformer_adapt_fixed384_official60.py",
    "rate_both_asformer_full_adapt": "duca_sampling_rate_both_asformer_full_adapt_fixed384_official60.py",
}
RUNNER_VARIANTS = {
    "sampling_rate_exact_uniform": "duca_sampling_rate_exact_uniform_fixed384_official60.py",
    "sampling_rate_only": "duca_sampling_rate_fixed384_official60.py",
    "sampling_rate_cls": "duca_sampling_rate_cls_fixed384_official60.py",
    "sampling_rate_reg": "duca_sampling_rate_reg_fixed384_official60.py",
    "sampling_rate_both": "duca_sampling_rate_both_fixed384_official60.py",
    "sampling_rate_both_asformer_last": "duca_sampling_rate_both_asformer_adapt_fixed384_official60.py",
    "sampling_rate_both_asformer_full": "duca_sampling_rate_both_asformer_full_adapt_fixed384_official60.py",
}


def test_sampling_rate_official60_variants_are_matched_and_validated(monkeypatch) -> None:
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    payloads = {key: validate_config(CONFIG_ROOT / value) for key, value in VARIANTS.items()}
    assert all(payload["ok"] for payload in payloads.values())
    assert payloads["rate_exact_uniform"]["force_exact_uniform_control"] is True
    assert payloads["rate_only"]["contribution_components"] == "none"
    assert payloads["rate_cls"]["contribution_components"] == "cls"
    assert payloads["rate_reg"]["contribution_components"] == "reg"
    assert payloads["rate_both"]["contribution_components"] == "both"
    assert payloads["rate_both_asformer_adapt"]["asformer_last_layer_adapted"] is True
    assert payloads["rate_both_asformer_full_adapt"]["asformer_full_encoder_adapted"] is True


def test_only_adaptation_variants_open_declared_asformer_policy_gradient(monkeypatch) -> None:
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    variants = {key: Config.fromfile(str(CONFIG_ROOT / value)) for key, value in VARIANTS.items()}
    for key, cfg in variants.items():
        selector = cfg.model.frame_selector
        assert selector.acquisition_policy == "budget_calibrated_sampling_rate"
        assert selector.max_unselected_hole is None
        assert selector.hard_max_gap_repair is False
        assert selector.soft_max_gap_loss_enabled is False
        assert int(cfg.workflow.val_eval_interval) == 5
        assert int(cfg.workflow.val_eval_interval_anchor_epoch) == 5
        assert int(cfg.workflow.val_start_epoch) == 4
        assert cfg.workflow.intermediate_validation_role == (
            "full_curve_and_best_validation_checkpoint"
        )
        assert cfg.workflow.intermediate_validation_selects_checkpoint is True
        assert (
            cfg.workflow.formal_protocol
            == duca_selected_axis_training.FORMAL_PROTOCOL
        )
        expected_scope = {
            "rate_both_asformer_adapt": "asformer_last_encoder_layer",
            "rate_both_asformer_full_adapt": "asformer_full_encoder",
        }.get(key, "asformer_full_encoder")
        assert selector.actionness_source_cfg.policy_hidden_gradient_scope == expected_scope
        assert float(selector.loss_weight_schedule.asformer_adapt.end) == (
            1.0 if key in {"rate_both_asformer_adapt", "rate_both_asformer_full_adapt"} else 0.0
        )


def test_sampling_rate_companion_uses_the_rate_transport_bridge() -> None:
    cfg = Config.fromfile(
        str(CONFIG_ROOT / "duca_sampling_rate_fixed384_official60.py")
    )
    selector = cfg.model.frame_selector

    assert selector.acquisition_policy == "budget_calibrated_sampling_rate"
    assert selector.training_uniform_companion_fraction > 0.0
    assert selector.training_uniform_companion_normalize_learned_gradient
    assert selector.detector_gradient_mode == "density_transport_st"


def test_rate_only_control_disables_the_contribution_head_path() -> None:
    source = (ROOT / "opentad" / "models" / "duca" / "acquisition.py").read_text(
        encoding="utf-8"
    )
    expected = (
        'if self.sampling_rate_utility_components != "none":\n'
        "                    detector_contribution_logits = ("
    )
    assert expected in source


def test_sampling_rate_keeps_a_nonzero_transition_output_path_for_coarse_supervision(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    cfg = Config.fromfile(
        str(CONFIG_ROOT / "duca_sampling_rate_fixed384_official60.py")
    )
    contract = formal_training_contract(cfg)

    assert contract is not None
    assert contract["intermediate_validation"] is True
    assert contract["intermediate_validation_interval"] == 5
    assert [
        epoch + 1
        for epoch in range(int(cfg.workflow.end_epoch))
        if _should_eval_epoch(epoch, cfg.workflow)
    ] == list(range(5, 61, 5))

    source = (ROOT / "opentad" / "models" / "duca" / "acquisition.py").read_text(
        encoding="utf-8"
    )
    scorer_block = source[source.index("self.transition_scorer = DucaTransitionUtilityScorer("):source.index("if self.acquisition_policy == \"continuous_mixture_density_transport\"")]
    assert '"budget_calibrated_sampling_rate"' not in scorer_block

    selected_contract = duca_selected_axis_training.formal_training_contract(cfg)
    assert selected_contract is not None
    assert selected_contract["formal_protocol"] == duca_selected_axis_training.FORMAL_PROTOCOL


def test_sampling_rate_variants_reuse_the_existing_selected_axis_runtime_contract() -> None:
    for variant, config_name in RUNNER_VARIANTS.items():
        assert duca_selected_axis_training.VARIANT_CONFIGS[variant] == config_name


def test_existing_independent_runner_exposes_sampling_rate_matrix_without_a_new_launcher() -> None:
    runner = (ROOT / "scripts" / "run_duca_independent_official60_gpu1.sh").read_text(
        encoding="utf-8"
    )
    expected = {
        "sampling_rate_exact_uniform": "duca_sampling_rate_exact_uniform_fixed384_official60.py",
        "sampling_rate_only": "duca_sampling_rate_fixed384_official60.py",
        "sampling_rate_cls": "duca_sampling_rate_cls_fixed384_official60.py",
        "sampling_rate_reg": "duca_sampling_rate_reg_fixed384_official60.py",
        "sampling_rate_both": "duca_sampling_rate_both_fixed384_official60.py",
        "sampling_rate_both_asformer_last": "duca_sampling_rate_both_asformer_adapt_fixed384_official60.py",
        "sampling_rate_both_asformer_full": "duca_sampling_rate_both_asformer_full_adapt_fixed384_official60.py",
    }
    for variant, config in expected.items():
        assert f"{variant})" in runner
        assert config in runner
    assert "validate_duca_sampling_rate_official60.py" in runner


def test_rate_curriculum_keeps_uniform_warmup_and_tad_led_joint_phase(monkeypatch) -> None:
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    stage1 = Config.fromfile(
        str(CONFIG_ROOT / "duca_sampling_rate_curriculum_stage1_uniform384.py")
    )
    selector1 = stage1.model.frame_selector
    assert stage1.duca_sampling_rate_contract.stage == "uniform_k384_full_model_coarse_convergence"
    assert int(stage1.workflow.end_epoch) == 30
    assert int(stage1.workflow.val_eval_interval) == 5
    assert stage1.workflow.formal_protocol == ""
    assert selector1.acquisition_policy == "budget_calibrated_sampling_rate"
    assert selector1.training_uniform_companion_fraction == 0.0
    assert selector1.training_uniform_companion_normalize_learned_gradient is False
    assert float(selector1.loss_weight_schedule.policy_alpha.start) == 0.0
    assert float(selector1.loss_weight_schedule.policy_alpha.end) == 0.0
    assert float(selector1.loss_weight_schedule.detector_gradient.end) == 0.0
    assert float(selector1.loss_weight_schedule.actionness.end) == 1.0
    assert float(selector1.loss_weight_schedule.transition.end) == 0.50
    assert float(selector1.loss_weight_schedule.transition_boundary.end) == 2.0

    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT", "stage1.pth")
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT_EPOCH", "29")
    stage2 = Config.fromfile(
        str(CONFIG_ROOT / "duca_sampling_rate_curriculum_stage2_joint384.py")
    )
    selector2 = stage2.model.frame_selector
    schedule = selector2.loss_weight_schedule
    assert stage2.duca_sampling_rate_contract.stage == (
        "low_lr_joint_rate_adaptation_then_tad_led_joint_training"
    )
    assert stage2.workflow.formal_protocol == ""
    assert stage2.workflow.formal_successful_update_contract is False
    assert stage2.workflow.model_initialization.state_key == "state_dict_ema"
    assert stage2.workflow.model_initialization.reset_state_keys == [
        "frame_selector._loss_weight_schedule_step"
    ]
    runner = (ROOT / "scripts" / "run_duca_sampling_rate_curriculum_gpu1.sh").read_text(
        encoding="utf-8"
    )
    assert "DUCA_STAGE1_REUSE_CHECKPOINT" in runner
    assert "reused stage1 checkpoint hash mismatch" in runner
    assert selector2.coarse_trunk_lr < selector1.coarse_trunk_lr
    assert selector2.action_head_lr < selector1.action_head_lr
    assert float(schedule.policy_alpha.start) == 0.0
    assert float(schedule.policy_alpha.end) == 1.0
    assert float(schedule.detector_gradient.start) == 0.0
    assert float(schedule.detector_gradient.end) == 0.25
    assert float(schedule.actionness.end) == 0.25
    assert float(schedule.transition.end) == 0.10
    assert float(schedule.transition_boundary.end) == 0.25
