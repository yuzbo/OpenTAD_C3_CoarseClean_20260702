from __future__ import annotations

from pathlib import Path

from mmengine.config import Config

from tools.bata.validate_duca_protected_e2e_official60 import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"
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
    assert float(direct.detector_gradient.start) == 0.25
    assert float(homotopy.policy_alpha.start) == 0.0
    assert float(homotopy.policy_alpha.end) == 1.0
    assert float(companion.training_uniform_companion_fraction) == 0.5
