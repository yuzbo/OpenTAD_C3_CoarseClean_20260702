from __future__ import annotations

from mmengine.config import Config
from tools.bata.validate_duca_transition_only_p0_variant import validate_variant


ROOT = "configs/adatad/thumos/"
PATHS = {
    "uniform": ROOT + "duca_exact_uniform_fixed384_official_adatad_backend_full_train.py",
    "direct": ROOT + "duca_direct_boundary_fixed384_13200_official_adatad_backend_full_train.py",
    "transition_beta0": ROOT + "duca_transition_only_fixed384_no_detector_bridge_official_adatad_backend_full_train.py",
    "transition_beta025": ROOT + "duca_transition_only_fixed384_official_adatad_backend_full_train.py",
}


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def test_p0_matrix_is_matched_on_detector_data_geometry_and_training_horizon() -> None:
    configs = {name: Config.fromfile(path) for name, path in PATHS.items()}
    reference = configs["transition_beta025"]

    for name, cfg in configs.items():
        assert cfg.model.type == "ActionFormer", name
        assert cfg.model.rpn_head.type == "ActionFormerHead", name
        assert _plain(cfg.model.rpn_head) == _plain(reference.model.rpn_head), name
        assert _plain(cfg.dataset) == _plain(reference.dataset), name
        assert _plain(cfg.optimizer) == _plain(reference.optimizer), name
        assert int(cfg.window_size) == 384, name
        assert int(cfg.dense_window_size) == 768, name
        assert int(cfg.model.backbone.backbone.total_frames) == 384, name
        assert int(cfg.model.projection.max_seq_len) == 384, name
        assert int(cfg.workflow.end_epoch) == 132, name
        assert int(cfg.scheduler.max_epoch) == 132, name


def test_p0_matrix_changes_only_the_intended_selector_mechanism() -> None:
    uniform = Config.fromfile(PATHS["uniform"])
    direct = Config.fromfile(PATHS["direct"])
    beta0 = Config.fromfile(PATHS["transition_beta0"])
    beta025 = Config.fromfile(PATHS["transition_beta025"])

    assert uniform.model.frame_selector.selector_variant == "transition_only"
    assert uniform.model.frame_selector.inference_policy_alpha == 0.0
    assert uniform.model.frame_selector.loss_weight_schedule.policy_alpha.end == 0.0
    assert uniform.model.frame_selector.loss_weight_schedule.detector_gradient.end == 0.0

    assert direct.model.frame_selector.get("selector_variant", "direct_boundary") == "direct_boundary"
    assert direct.duca_loss_schedule_total_steps == 13200
    assert direct.duca_schedule_steps_per_epoch == 100

    assert beta0.model.frame_selector.selector_variant == "transition_only"
    assert beta0.model.frame_selector.loss_weight_schedule.policy_alpha.end == 1.0
    assert beta0.model.frame_selector.loss_weight_schedule.detector_gradient.end == 0.0

    assert beta025.model.frame_selector.selector_variant == "transition_only"
    assert beta025.model.frame_selector.loss_weight_schedule.policy_alpha.end == 1.0
    assert beta025.model.frame_selector.loss_weight_schedule.detector_gradient.end == 0.25
    assert beta025.model.frame_selector.soft_max_gap_loss_enabled is False


def test_p0_variant_validator_accepts_every_declared_variant() -> None:
    for variant, path in PATHS.items():
        summary = validate_variant(variant, path)
        assert summary["ok"] is True
        assert summary["variant"] == variant
        assert summary["budget"] == 384
        assert summary["paper_claim_allowed"] is False
