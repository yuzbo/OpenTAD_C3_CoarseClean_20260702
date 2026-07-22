from __future__ import annotations

from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata.validate_duca_protected_e2e_official60 import validate_config


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
    for key in ("dataset", "optimizer", "scheduler", "solver", "workflow"):
        assert r2[key].to_dict() == r4[key].to_dict()


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
