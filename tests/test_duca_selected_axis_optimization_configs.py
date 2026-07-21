from __future__ import annotations

from pathlib import Path

from mmengine.config import Config

from tools.bata.validate_duca_protected_e2e_official60 import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"
FULL_MODEL_GATE = ROOT / "tools" / "bata" / "run_duca_protected_e2e_exact_full_model_gate.py"
GATE_LAUNCHER = ROOT / "scripts" / "run_duca_selected_axis_optimization_gate_gpu1.sh"
VARIANTS = {
    "direct": "duca_protected_e2e_direct025_fixed384_official60.py",
    "homotopy": "duca_protected_e2e_homotopy025_fixed384_official60.py",
    "companion": (
        "duca_protected_e2e_homotopy_uni_companion025_fixed384_official60.py"
    ),
}


def test_selected_axis_optimization_configs_pass_the_official60_contract() -> None:
    payloads = {
        name: validate_config(CONFIG_DIR / filename)
        for name, filename in VARIANTS.items()
    }

    assert all(payload["ok"] for payload in payloads.values())
    assert payloads["direct"]["training_uniform_companion_fraction"] == 0.0
    assert payloads["homotopy"]["training_uniform_companion_fraction"] == 0.0
    assert payloads["companion"]["training_uniform_companion_fraction"] == 0.5


def test_selected_axis_variants_share_one_official_head_and_differ_only_in_training_policy() -> None:
    configs = {
        name: Config.fromfile(str(CONFIG_DIR / filename))
        for name, filename in VARIANTS.items()
    }

    first_head = configs["direct"].model.rpn_head.to_dict()
    for cfg in configs.values():
        selector = cfg.model.frame_selector
        assert cfg.model.rpn_head.to_dict() == first_head
        assert "physical_grid_actionformer" not in cfg.model.rpn_head
        assert selector.detector_output_coordinate_space == "selected_axis_index"
        assert selector.remap_gt_to_selected_axis is True
        assert selector.detector_gradient_mode == "protected_structured_transport"
        assert int(selector.budget) == 384
        assert int(selector.dense_window_size) == 768

    direct = configs["direct"].model.frame_selector.loss_weight_schedule
    homotopy = configs["homotopy"].model.frame_selector.loss_weight_schedule
    companion = configs["companion"].model.frame_selector
    assert float(direct.policy_alpha.start) == float(direct.policy_alpha.end) == 1.0
    assert float(direct.detector_gradient.start) == 0.0
    assert float(direct.detector_gradient.end) == 0.25
    assert int(direct.detector_gradient.warmup_steps) == 2100
    assert int(direct.detector_gradient.transition_steps) == 1500
    assert float(homotopy.policy_alpha.start) == 0.0
    assert float(homotopy.policy_alpha.end) == 1.0
    assert float(companion.training_uniform_companion_fraction) == 0.5


def test_selected_axis_full_model_gate_reuses_the_production_amp_replay_path() -> None:
    source = FULL_MODEL_GATE.read_text(encoding="utf-8")
    launcher = GATE_LAUNCHER.read_text(encoding="utf-8")

    assert "train_one_epoch(" in source
    assert "force_amp_overflow_attempts=1" in source
    assert "max_amp_retries_per_batch=int(cfg.workflow.max_amp_retries_per_batch)" in source
    assert "fail_on_amp_replay_exhaustion=True" in source
    assert "ModelEma(ddp)" in source
    assert "scheduler_updates" in source
    assert "duca_schedule_updates" in source
    assert "FORMAL_SEED = 3407" in source
    assert "exact_uniform_positions(" in source
    full_model_block = launcher.split("full_model_configs=(", 1)[1].split(")", 1)[0]
    assert "duca_exact_uniform_fixed384_official60.py" in full_model_block
    assert '"four_matched_variants_full_model_gate_passed"' in launcher
