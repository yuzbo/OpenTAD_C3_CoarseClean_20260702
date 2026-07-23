from __future__ import annotations

from pathlib import Path

from mmengine.config import Config

from tools.bata.validate_duca_sampling_rate_official60 import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "adatad" / "thumos"
VARIANTS = {
    "rate_exact_uniform": "duca_sampling_rate_exact_uniform_fixed384_official60.py",
    "rate_only": "duca_sampling_rate_fixed384_official60.py",
    "rate_cls": "duca_sampling_rate_cls_fixed384_official60.py",
    "rate_reg": "duca_sampling_rate_reg_fixed384_official60.py",
    "rate_both": "duca_sampling_rate_both_fixed384_official60.py",
    "rate_both_asformer_adapt": "duca_sampling_rate_both_asformer_adapt_fixed384_official60.py",
    "rate_both_asformer_full_adapt": "duca_sampling_rate_both_asformer_full_adapt_fixed384_official60.py",
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
