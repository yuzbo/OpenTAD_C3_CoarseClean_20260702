from __future__ import annotations

from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata.validate_duca_protected_e2e_official60 import validate_config
from tools.bata.validate_duca_frontend_p0_contract import (
    validate_config as validate_p0_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "adatad" / "thumos"


@pytest.fixture(autouse=True)
def _config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT", "frontend.pth")
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_EPOCH", "19")
    monkeypatch.setenv("DUCA_FRONTEND_TRAIN_BLOCK_LIST", "train.txt")


def test_boundary_burst_p0_candidates_change_only_the_preregistered_geometry() -> None:
    r2 = Config.fromfile(
        str(CONFIG_ROOT / "duca_boundary_burst_frontend_pretrain_fixed384.py")
    )
    r4 = Config.fromfile(
        str(CONFIG_ROOT / "duca_boundary_burst_r4q5_frontend_pretrain_fixed384.py")
    )

    assert r2.model.frame_selector.transition_objective == "boundary_burst"
    assert r4.model.frame_selector.transition_objective == "boundary_burst"
    assert (r2.model.frame_selector.transition_boundary_radius, r2.model.frame_selector.boundary_burst_quota) == (2, 3.0)
    assert (r4.model.frame_selector.transition_boundary_radius, r4.model.frame_selector.boundary_burst_quota) == (4, 5.0)
    assert r2.model.frame_selector.boundary_burst_require_bilateral_offsets is True
    assert r4.model.frame_selector.boundary_burst_require_bilateral_offsets is True
    assert r2.duca_transition_only_contract.hard_global_burst_support == (
        "mandatory_group_constrained_exact_k_max_hole"
    )
    assert r4.duca_transition_only_contract.hard_global_burst_support == (
        "mandatory_group_constrained_exact_k_max_hole"
    )
    assert r2.duca_transition_only_contract.transition_distribution_updates == (
        "asformer_last_encoder_layer_only"
    )
    assert r4.duca_transition_only_contract.transition_distribution_updates == (
        "asformer_last_encoder_layer_only"
    )
    assert r2.model.frame_selector.auxiliary_hidden_gradient_scale == pytest.approx(0.25)
    assert r4.model.frame_selector.auxiliary_hidden_gradient_scale == pytest.approx(0.25)
    assert r2.model.frame_selector.policy_hidden_gradient_scale == pytest.approx(0.05)
    assert r4.model.frame_selector.policy_hidden_gradient_scale == pytest.approx(0.05)
    assert (
        r2.model.frame_selector.actionness_source_cfg.policy_hidden_gradient_scope
        == "asformer_last_encoder_layer"
    )
    for key in ("dataset", "optimizer", "scheduler", "solver", "workflow"):
        assert r2[key].to_dict() == r4[key].to_dict()


def test_r2q3_hard_and_hidden_adaptation_form_a_matched_two_by_two_ablation() -> None:
    names = {
        (False, False): "duca_boundary_burst_soft_detached_frontend_pretrain_fixed384.py",
        (True, False): "duca_boundary_burst_hard_detached_frontend_pretrain_fixed384.py",
        (False, True): "duca_boundary_burst_soft_adapted_frontend_pretrain_fixed384.py",
        (True, True): "duca_boundary_burst_frontend_pretrain_fixed384.py",
    }
    configs = {
        key: Config.fromfile(str(CONFIG_ROOT / name)) for key, name in names.items()
    }
    reference = configs[(True, True)]
    for (hard, adapted), cfg in configs.items():
        assert validate_p0_config(CONFIG_ROOT / names[(hard, adapted)])["ok"] is True
        selector = cfg.model.frame_selector
        assert selector.boundary_burst_require_bilateral_offsets is hard
        assert (float(selector.auxiliary_hidden_gradient_scale) > 0.0) is adapted
        assert (float(selector.policy_hidden_gradient_scale) > 0.0) is adapted
        assert (
            selector.actionness_source_cfg.policy_hidden_gradient_scope
            == ("asformer_last_encoder_layer" if adapted else "none")
        )
        assert bool(
            cfg.duca_transition_only_contract.transition_supervision_updates_coarse_representation
        ) is adapted
        assert (
            cfg.duca_transition_only_contract.hard_global_burst_support
            == (
                "mandatory_group_constrained_exact_k_max_hole"
                if hard
                else "none"
            )
        )
        for key in ("dataset", "optimizer", "scheduler", "solver", "workflow"):
            assert cfg[key].to_dict() == reference[key].to_dict()

    soft_g0 = Config.fromfile(
        str(
            CONFIG_ROOT
            / "duca_boundary_burst_soft_g0_no_feedback_fixed384_official60.py"
        )
    )
    hard_g0 = Config.fromfile(
        str(CONFIG_ROOT / "duca_boundary_burst_g0_no_feedback_fixed384_official60.py")
    )
    assert soft_g0.model.frame_selector.boundary_burst_require_bilateral_offsets is False
    assert hard_g0.model.frame_selector.boundary_burst_require_bilateral_offsets is True
    assert soft_g0.model.rpn_head.to_dict() == hard_g0.model.rpn_head.to_dict()
    for key in ("dataset", "optimizer", "scheduler", "solver", "workflow"):
        assert soft_g0[key].to_dict() == hard_g0[key].to_dict()


def test_boundary_burst_official60_arms_keep_the_same_detector_and_protocol() -> None:
    names = (
        "duca_global_curriculum_g0_no_feedback_fixed384_official60.py",
        "duca_boundary_burst_g0_no_feedback_fixed384_official60.py",
        "duca_boundary_burst_r4q5_g0_no_feedback_fixed384_official60.py",
    )
    configs = [Config.fromfile(str(CONFIG_ROOT / name)) for name in names]
    detector_head = configs[0].model.rpn_head.to_dict()
    for cfg in configs:
        selector = cfg.model.frame_selector
        assert selector.acquisition_policy == "global_structured_topk"
        assert int(selector.budget) == 384
        assert int(selector.max_unselected_hole) == 2
        assert selector.detector_gradient_mode == "none"
        if selector.get("transition_objective") == "boundary_burst":
            assert selector.boundary_burst_require_bilateral_offsets is True
        assert cfg.model.rpn_head.to_dict() == detector_head
        assert int(cfg.workflow.expected_successful_optimizer_updates) == 6000
        assert int(cfg.workflow.primary_checkpoint_epoch) == 59
    for key in ("dataset", "optimizer", "scheduler", "workflow"):
        assert configs[0][key].to_dict() == configs[1][key].to_dict()
        assert configs[1][key].to_dict() == configs[2][key].to_dict()


@pytest.mark.parametrize(
    "name",
    (
        "duca_two_stage_exact_uniform_fixed384_official60.py",
        "duca_global_curriculum_g0_no_feedback_fixed384_official60.py",
        "duca_boundary_burst_g0_no_feedback_fixed384_official60.py",
        "duca_boundary_burst_soft_g0_no_feedback_fixed384_official60.py",
        "duca_boundary_burst_r4q5_g0_no_feedback_fixed384_official60.py",
        "duca_boundary_burst_g1_protected_fixed384_official60.py",
        "duca_boundary_burst_g2_uni_companion_fixed384_official60.py",
        "duca_boundary_burst_r4q5_g1_protected_fixed384_official60.py",
        "duca_boundary_burst_r4q5_g2_uni_companion_fixed384_official60.py",
    ),
)
def test_boundary_burst_final_matrix_passes_official60_contract(name: str) -> None:
    assert validate_config(CONFIG_ROOT / name)["ok"] is True


def test_boundary_burst_feedback_is_disclosed_as_calibrated_surrogate() -> None:
    cfg = Config.fromfile(
        str(CONFIG_ROOT / "duca_boundary_burst_g1_protected_fixed384_official60.py")
    )
    contract = cfg.duca_transition_only_contract
    assert contract.detector_gradient_is_direct is False
    assert contract.detector_gradient_estimator == (
        "hard_forward_temporal_slope_surrogate_calibrated_by_signed_hard_swap"
    )


def test_boundary_burst_train_pipeline_filters_crop_truncated_endpoints() -> None:
    cfg = Config.fromfile(
        str(CONFIG_ROOT / "duca_boundary_burst_frontend_pretrain_fixed384.py")
    )
    load_frames = next(item for item in cfg.dataset.train.pipeline if item.type == "LoadFrames")
    collect = next(item for item in cfg.dataset.train.pipeline if item.type == "Collect")

    assert load_frames.emit_boundary_validity is True
    assert "gt_boundary_validity" in collect["keys"]


def test_r0_replay_evaluates_the_training_holdout_annotation_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DUCA_FRONTEND_HOLDOUT_BLOCK_LIST", "holdout.txt")
    monkeypatch.setenv("DUCA_R0_EVAL_BLOCKED_VIDEOS", "blocked.json")
    monkeypatch.setenv("DUCA_ALLOCATION_ARTIFACT_PATH", "families.jsonl")
    monkeypatch.setenv("DUCA_ALLOCATION_ARTIFACT_SHA256", "b" * 64)
    cfg = Config.fromfile(
        str(CONFIG_ROOT / "duca_boundary_burst_r0_selected_axis_replay.py")
    )

    assert cfg.dataset.test.test_mode is True
    assert cfg.dataset.test.subset_name == "training"
    assert cfg.evaluation.subset == "training"
    assert cfg.evaluation.blocked_videos == "blocked.json"
    assert cfg.workflow.formal_protocol == "duca_r0_selected_axis_holdout_replay_v1"


def test_r0_launcher_uses_allocated_cpus_for_bootstrap_only() -> None:
    launcher = (
        ROOT / "scripts" / "run_duca_boundary_burst_r0_holdout_map_gpu1.sh"
    ).read_text(encoding="utf-8")

    assert (
        '--bootstrap-workers "${DUCA_R0_BOOTSTRAP_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"'
        in launcher
    )
    assert "--bootstrap-samples 1000" in launcher
    assert "--bootstrap-seed 3407" in launcher
    assert "--bootstrap-confidence 0.95" in launcher
