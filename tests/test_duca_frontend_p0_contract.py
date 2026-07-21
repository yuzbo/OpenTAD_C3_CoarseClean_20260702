from __future__ import annotations

import os
from pathlib import Path

import pytest
from mmengine.config import Config

try:
    from tools.bata.validate_duca_frontend_p0_contract import validate_config
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"DUCA contract dependencies are unavailable: {exc}", allow_module_level=True)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "adatad" / "thumos"

LOCAL_RESIDUAL_CONFIGS = {
    "u": "duca_local_residual_u_exact_uniform_fixed384_official60.py",
    "d": "duca_local_residual_d_pure_delta_fixed384_official60.py",
    "r0": "duca_local_residual_r0_no_feedback_fixed384_official60.py",
    "r1": "duca_local_residual_r1_feedback_fixed384_official60.py",
    "r1_uc": "duca_local_residual_r1_uniform_companion_fixed384_official60.py",
}


@pytest.mark.parametrize(
    "name",
    (
        "duca_frontend_pretrain_a1_t005_b8.py",
        "duca_frontend_pretrain_a1_t010_b16.py",
        "duca_frontend_pretrain_a1_t020_b32.py",
    ),
)
def test_frontend_variants_satisfy_strict_p0_contract(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DUCA_FRONTEND_TRAIN_BLOCK_LIST", "train_block.txt")

    payload = validate_config(CONFIG_ROOT / name)

    assert payload["ok"] is True
    assert payload["detector_executed"] is False
    assert payload["active_losses"] == [
        "actionness",
        "transition",
        "transition_boundary",
    ]
    assert all(
        value == 0.0
        for key, value in payload["loss_weights"].items()
        if key not in payload["active_losses"]
    )
    assert payload["spatial_norm"] == "groupnorm"
    assert payload["auxiliary_hidden_gradient_scale"] == 0.0
    assert payload["optimizer"]["global_gradient_clipping_enabled"] is False


def test_serial_curriculum_runs_one_real_gate_before_frontend_training() -> None:
    serial_launcher = (ROOT / "scripts" / "run_duca_two_stage_curriculum_serial_gpu1.sh").read_text(
        encoding="utf-8"
    )
    assert "run_duca_frontend_p0_real_gate.py" in serial_launcher
    assert "--standalone" in serial_launcher
    assert '"${DUCA_FRONTEND_ONLY:-0}" == "1"' in serial_launcher


def test_serial_curriculum_reopens_split_paths_from_manifest() -> None:
    serial_launcher = (ROOT / "scripts" / "run_duca_two_stage_curriculum_serial_gpu1.sh").read_text(
        encoding="utf-8"
    )
    assert '["train_block_list"]' in serial_launcher
    assert '["holdout_block_list"]' in serial_launcher
    assert (
        'export DUCA_FRONTEND_TRAIN_BLOCK_LIST="${RUN_ROOT}/frontend_split/'
        not in serial_launcher
    )
    assert (
        'export DUCA_FRONTEND_HOLDOUT_BLOCK_LIST="${RUN_ROOT}/frontend_split/'
        not in serial_launcher
    )
    assert '[[ -f "${DUCA_FRONTEND_TRAIN_BLOCK_LIST}" ]]' in serial_launcher
    assert '[[ -f "${DUCA_FRONTEND_HOLDOUT_BLOCK_LIST}" ]]' in serial_launcher


def test_real_gate_classifies_the_executed_spatial_stem_parameter_path() -> None:
    gate_source = (ROOT / "tools" / "bata" / "run_duca_frontend_p0_real_gate.py").read_text(
        encoding="utf-8"
    )
    assert 'if ".spatial_stem." in normalized:' in gate_source
    assert 'if ".spatial_encoder." in normalized:' not in gate_source


def test_local_residual_official60_configs_have_strict_local_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT", "frontend.pth")
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_EPOCH", "19")

    configs = {
        key: Config.fromfile(str(CONFIG_ROOT / filename))
        for key, filename in LOCAL_RESIDUAL_CONFIGS.items()
    }
    expected_modes = {
        "u": "none",
        "d": "none",
        "r0": "none",
        "r1": "local_cell_straight_through",
        "r1_uc": "local_cell_straight_through",
    }
    for key, cfg in configs.items():
        selector = cfg.model.frame_selector
        contract = cfg.duca_transition_only_contract
        assert contract.task == "offline_temporal_action_detection"
        assert contract.online_tad is False
        assert selector.acquisition_policy == "local_cell_deformation"
        assert selector.local_cell_base_policy == "abs_delta_actionness"
        assert selector.local_cell_detector_grid_mode == "selected"
        assert selector.actionness_source_cfg.frozen is True
        assert selector.actionness_source_cfg.trainable is False
        assert selector.policy_hidden_gradient_scale == 0.0
        assert selector.auxiliary_hidden_gradient_scale == 0.0
        assert selector.max_unselected_hole is None
        assert selector.temporal_sampling_contract is None
        assert selector.hard_max_gap_repair is False
        assert selector.soft_max_gap_loss_enabled is False
        assert selector.detector_gradient_mode == expected_modes[key]
        assert set(selector.loss_weights) == {
            "detector",
            "actionness",
            "budget",
            "boundary",
            "hole",
            "max_gap_hole",
            "redundancy",
            "radius",
            "entropy",
            "teacher",
            "detector_utility",
            "start",
            "end",
            "context",
            "lagrangian_budget",
            "marginal_monotonic",
            "hard_budget_cap",
            "transition",
            "transition_boundary",
        }
        assert selector.loss_weights.detector == 1.0
        assert selector.loss_weights.actionness == 0.0
        assert selector.loss_weight_schedule.actionness.end == 0.0

    assert configs["u"].model.frame_selector.local_cell_force_exact_uniform is True
    assert configs["u"].model.frame_selector.inference_policy_alpha == 0.0
    assert configs["d"].model.frame_selector.local_cell_residual_scale == 0.0
    assert configs["r0"].model.frame_selector.local_cell_residual_scale == 0.25
    assert configs["r1"].model.frame_selector.loss_weight_schedule.detector_gradient.end == 0.25
    assert configs["r1_uc"].model.frame_selector.training_uniform_companion_fraction == 0.5
