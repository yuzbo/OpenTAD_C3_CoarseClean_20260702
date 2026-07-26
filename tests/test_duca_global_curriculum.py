from __future__ import annotations

from pathlib import Path

from mmengine.config import Config

from tools.bata.validate_duca_protected_e2e_official60 import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"
VARIANTS = {
    "uniform": "duca_two_stage_exact_uniform_fixed384_official60.py",
    "g0": "duca_global_curriculum_g0_no_feedback_fixed384_official60.py",
    "g1": "duca_global_curriculum_g1_protected_fixed384_official60.py",
    "g2": "duca_global_curriculum_g2_uni_companion_fixed384_official60.py",
}


def _load(monkeypatch):
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT", "frontend.pth")
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_EPOCH", "19")
    return {
        name: Config.fromfile(str(CONFIG_DIR / filename))
        for name, filename in VARIANTS.items()
    }


def test_global_curriculum_reuses_one_global_model_and_official_detector(monkeypatch) -> None:
    configs = _load(monkeypatch)
    learned = [configs[name] for name in ("g0", "g1", "g2")]
    reference_head = configs["uniform"].model.rpn_head.to_dict()
    for cfg in learned:
        selector = cfg.model.frame_selector
        assert selector.acquisition_policy == "global_structured_topk"
        assert selector.acquisition_policy != "local_cell_deformation"
        assert int(selector.budget) == 384
        assert int(selector.dense_window_size) == 768
        assert int(selector.max_unselected_hole) == 2
        assert selector.detector_output_coordinate_space == "selected_axis_index"
        assert selector.actionness_source_cfg.frozen is True
        assert selector.actionness_source_cfg.trainable is False
        assert cfg.model.rpn_head.to_dict() == reference_head
        assert cfg.workflow.expected_successful_optimizer_updates == 6000
        assert cfg.workflow.primary_checkpoint_epoch == 59


def test_global_curriculum_configs_pass_the_registered_official60_contract(monkeypatch) -> None:
    _load(monkeypatch)
    for filename in VARIANTS.values():
        payload = validate_config(CONFIG_DIR / filename)
        assert payload["ok"] is True


def test_global_curriculum_arms_change_only_feedback_and_companion(monkeypatch) -> None:
    configs = _load(monkeypatch)
    g0 = configs["g0"].model.frame_selector
    g1 = configs["g1"].model.frame_selector
    g2 = configs["g2"].model.frame_selector

    assert g0.detector_gradient_mode == "none"
    assert float(g0.loss_weight_schedule.detector_gradient.end) == 0.0
    assert g1.detector_gradient_mode == "protected_structured_transport"
    assert float(g1.loss_weight_schedule.detector_gradient.end) == 0.25
    assert float(g1.training_uniform_companion_fraction) == 0.0
    assert g1.training_uniform_companion_normalize_learned_gradient is False
    assert g2.detector_gradient_mode == "protected_structured_transport"
    assert float(g2.loss_weight_schedule.detector_gradient.end) == 0.25
    assert float(g2.training_uniform_companion_fraction) == 0.5
    assert g2.training_uniform_companion_normalize_learned_gradient is True

    for key in ("dataset", "optimizer", "scheduler", "workflow"):
        assert configs["g0"][key].to_dict() == configs["g1"][key].to_dict()
        assert configs["g1"][key].to_dict() == configs["g2"][key].to_dict()
