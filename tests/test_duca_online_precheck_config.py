import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
ADATAD_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "duca_online_adatad_precheck.py"
ZEROSHOT_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "duca_online_zeroshot_actionness_precheck.py"
ACTIONFORMER_NO_PHYSICAL_CONFIG = (
    ROOT / "configs" / "adatad" / "thumos" / "duca_online_actionformer_no_physical_grid_precheck.py"
)
ACTIONFORMER_PHYSICAL_CONFIG = (
    ROOT / "configs" / "adatad" / "thumos" / "duca_online_actionformer_physical_grid_precheck.py"
)
OFFICIAL_BACKEND_CONFIG = (
    ROOT / "configs" / "adatad" / "thumos" / "duca_online_official_adatad_backend_full_train.py"
)
DUCA_MUST_DYNAMIC_CONFIG = (
    ROOT / "configs" / "adatad" / "thumos" / "duca_must_dynamic_official_adatad_backend_full_train.py"
)
X3D_OFFICIAL_BACKEND_CONFIG = (
    ROOT / "configs" / "adatad" / "thumos" / "duca_online_x3d_official_adatad_backend_full_train.py"
)
X3D_MUST_DYNAMIC_CONFIG = (
    ROOT / "configs" / "adatad" / "thumos" / "duca_must_dynamic_x3d_official_adatad_backend_full_train.py"
)
VALIDATOR = ROOT / "tools" / "bata" / "validate_duca_online_adatad_precheck.py"
OFFICIAL_BACKEND_VALIDATOR = ROOT / "tools" / "bata" / "validate_duca_official_adatad_backend.py"
DUCA_MUST_DYNAMIC_VALIDATOR = ROOT / "tools" / "bata" / "validate_duca_must_dynamic_official_adatad_backend.py"
X3D_OFFICIAL_BACKEND_VALIDATOR = ROOT / "tools" / "bata" / "validate_duca_x3d_official_adatad_backend.py"
X3D_MUST_DYNAMIC_VALIDATOR = ROOT / "tools" / "bata" / "validate_duca_must_dynamic_x3d_official_adatad_backend.py"
ADATAD_LAUNCHER = ROOT / "scripts" / "run_duca_online_adatad_precheck_gpu1.sh"
ZEROSHOT_LAUNCHER = ROOT / "scripts" / "run_duca_online_zeroshot_actionness_precheck_gpu1.sh"
OFFICIAL_BACKEND_LAUNCHER = ROOT / "scripts" / "run_duca_online_official_adatad_backend_gpu1.sh"
DUCA_MUST_DYNAMIC_LAUNCHER = ROOT / "scripts" / "run_duca_must_dynamic_official_adatad_backend_gpu1.sh"
X3D_OFFICIAL_BACKEND_LAUNCHER = ROOT / "scripts" / "run_duca_x3d_official_adatad_backend_gpu1.sh"
X3D_MUST_DYNAMIC_LAUNCHER = ROOT / "scripts" / "run_duca_must_dynamic_x3d_official_adatad_backend_gpu1.sh"
OFFICIAL_BACKEND_BUDGET_CURVE_LAUNCHER = (
    ROOT / "scripts" / "run_duca_online_official_adatad_budget_curve_gpu1.sh"
)
ACTIONFORMER_NO_PHYSICAL_LAUNCHER = (
    ROOT / "scripts" / "run_duca_online_actionformer_no_physical_grid_precheck_gpu1.sh"
)
ACTIONFORMER_PHYSICAL_LAUNCHER = (
    ROOT / "scripts" / "run_duca_online_actionformer_physical_grid_diagnostic_precheck_gpu1.sh"
)


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_duca_online_adatad_precheck_test", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(path):
    return Config.fromfile(str(path))


def _assert_common_config_contract(cfg):
    assert cfg.model.type in {"SingleStageDetector", "ActionFormer"}
    assert cfg.model.frame_selector.type == "DucaOnlineFrameSelector"
    assert cfg.model.rpn_head.type in {"DucaOnlinePrecheckHead", "ActionFormerHead"}
    assert int(cfg.model.frame_selector.budget) <= 384
    assert int(cfg.duca_online_precheck_contract.budget_max) <= 384
    assert cfg.duca_online_precheck_contract.no_ledger_decision is True
    assert cfg.duca_online_precheck_contract.coordinate_space == "original_time"
    assert cfg.duca_online_precheck_contract.teacher_free_eval is True
    if cfg.duca_online_precheck_contract.get("physical_grid_actionformer_required", False):
        assert cfg.duca_online_precheck_contract.selected_axis_remap_required is False
    else:
        assert cfg.duca_online_precheck_contract.selected_axis_remap_required is True
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    selector_text = repr(cfg.model.frame_selector).lower()
    assert "ledger_path" not in selector_text
    assert "value_transport" not in selector_text


def test_duca_online_adatad_precheck_config_is_registry_buildable_contract():
    cfg = _cfg(ADATAD_CONFIG)
    _assert_common_config_contract(cfg)
    assert cfg.duca_online_precheck_contract.actionness_source == "online_probe"
    assert cfg.model.frame_selector.actionness_source_cfg.type == "DucaOnlineProbeActionnessSource"
    assert cfg.model.rpn_head.teacher_cfg.train_only is True
    assert cfg.model.rpn_head.teacher_cfg.enabled_for_inference is False


def test_duca_online_zeroshot_precheck_config_uses_zero_shot_actionness_source():
    cfg = _cfg(ZEROSHOT_CONFIG)
    _assert_common_config_contract(cfg)
    assert cfg.duca_online_precheck_contract.actionness_source == "zero_shot_motion"
    assert cfg.model.frame_selector.actionness_source_cfg.type == "ZeroShotMotionActionnessSource"
    assert cfg.model.frame_selector.actionness_source_cfg.no_train_gt is True
    assert cfg.model.frame_selector.actionness_source_cfg.no_teacher is True


def test_duca_online_actionformer_no_physical_grid_is_main_plugin_path():
    cfg = _cfg(ACTIONFORMER_NO_PHYSICAL_CONFIG)
    _assert_common_config_contract(cfg)
    assert cfg.model.type == "ActionFormer"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert "physical_grid_actionformer" not in cfg.model.rpn_head
    assert cfg.model.frame_selector.remap_gt_to_selected_axis is True
    assert cfg.model.frame_selector.detector_output_coordinate_space == "selected_axis_index"
    assert cfg.duca_online_precheck_contract.main_method_candidate is True
    assert cfg.duca_online_precheck_contract.diagnostic_only is False


def test_duca_online_actionformer_physical_grid_is_diagnostic_only():
    cfg = _cfg(ACTIONFORMER_PHYSICAL_CONFIG)
    _assert_common_config_contract(cfg)
    assert cfg.model.type == "ActionFormer"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.rpn_head.physical_grid_actionformer.enabled is True
    assert cfg.model.rpn_head.physical_grid_actionformer.required is True
    assert cfg.model.frame_selector.remap_gt_to_selected_axis is False
    assert cfg.model.frame_selector.detector_output_coordinate_space == "true_time_dense_index"
    assert cfg.duca_online_precheck_contract.diagnostic_only is True


def test_duca_online_official_backend_main_config_preserves_adatad_head_contract():
    cfg = _cfg(OFFICIAL_BACKEND_CONFIG)
    official = _cfg(ROOT / "configs" / "adatad" / "thumos" / "e2e_thumos_videomae_s_768x1_160_adapter.py")

    assert cfg.duca_online_main_contract.main_method_candidate is True
    assert cfg.duca_online_main_contract.official_adatad_backend is True
    assert cfg.duca_online_main_contract.changes_detector_head is False
    assert cfg.duca_online_main_contract.changes_loss_assignment is False
    assert cfg.duca_online_main_contract.no_ledger_decision is True
    assert cfg.duca_online_main_contract.physical_grid_actionformer_required is False
    assert cfg.model.type == "ActionFormer"
    assert cfg.model.frame_selector.type == "DucaOnlineFrameSelector"
    assert cfg.model.frame_selector.budget == 384
    assert cfg.model.frame_selector.dense_window_size == 768
    assert cfg.model.frame_selector.coordinate_space == "original_time"
    assert cfg.model.frame_selector.detector_output_coordinate_space == "selected_axis_index"
    assert cfg.model.frame_selector.remap_gt_to_selected_axis is True
    assert cfg.duca_online_main_contract.actionness_source == "online_trainable_c3_coarse_probe"
    assert cfg.duca_online_main_contract.coarse_probe_joint_trainable is True
    assert cfg.model.frame_selector.actionness_source_cfg.type == "C3CoarseProbeActionnessSource"
    assert cfg.duca_online_main_contract.acquisition_policy == "duca_center_radius_st_acquisition"
    assert cfg.duca_online_main_contract.budget_policy == "fixed_budget"
    assert cfg.model.frame_selector.budget_mode == "fixed"
    assert cfg.model.frame_selector.get("external_actionness_meta_key", None) is None
    assert cfg.model.frame_selector.get("external_actionness_logits_meta_key", None) is None
    assert cfg.model.frame_selector.get("require_external_actionness", False) is False
    assert cfg.model.frame_selector.forbid_external_actionness is True
    assert cfg.duca_online_main_contract.forbid_external_actionness is True
    assert cfg.model.frame_selector.actionness_source_cfg.probe_model == "official-action-seg"
    assert cfg.model.frame_selector.actionness_source_cfg.official_action_seg_backend == "official_asformer"
    assert cfg.model.frame_selector.actionness_source_cfg.tcn_variant == "official_asformer"
    assert cfg.model.frame_selector.actionness_source_cfg.get("tcn_variant", None) != "asformer_lite"
    assert cfg.model.frame_selector.actionness_source_cfg.trainable is True
    assert cfg.model.frame_selector.actionness_source_cfg.frozen is False
    assert cfg.model.frame_selector.loss_weights.actionness > 0
    assert cfg.model.frame_selector.loss_weights.detector_utility > 0
    assert cfg.model.frame_selector.loss_weights.hole > 0
    assert cfg.duca_online_main_contract.loss_schedule_policy == "progressive_joint"
    assert cfg.duca_online_main_contract.loss_schedule_step_update == "optimizer_step"
    assert cfg.duca_online_main_contract.loss_schedule_total_steps == (
        cfg.workflow.end_epoch * cfg.duca_schedule_steps_per_epoch
    )
    assert cfg.duca_online_main_contract.loss_schedule_warmup_fraction == 0.08
    assert cfg.duca_online_main_contract.loss_schedule_transition_fraction == 0.67
    assert cfg.model.frame_selector.loss_weight_schedule.type == "progressive_joint"
    assert cfg.model.frame_selector.loss_weight_schedule.warmup_steps == round(
        cfg.duca_online_main_contract.loss_schedule_total_steps
        * cfg.duca_online_main_contract.loss_schedule_warmup_fraction
    )
    assert cfg.model.frame_selector.loss_weight_schedule.transition_steps == round(
        cfg.duca_online_main_contract.loss_schedule_total_steps
        * cfg.duca_online_main_contract.loss_schedule_transition_fraction
    )
    assert cfg.model.frame_selector.loss_weight_schedule.actionness.start > cfg.model.frame_selector.loss_weight_schedule.actionness.end
    assert cfg.duca_online_main_contract.detector_loss_always_trains_backend is True
    assert cfg.model.frame_selector.loss_weight_schedule.detector_gradient.start == 0.0
    assert cfg.model.frame_selector.loss_weight_schedule.detector_gradient.end == 1.0
    assert cfg.model.frame_selector.loss_weight_schedule.detector_utility.start == 0.0
    assert cfg.model.frame_selector.loss_weight_schedule.detector_utility.end > 0.0
    assert cfg.model.frame_selector.loss_weight_schedule.hole.start == 0.0
    assert cfg.model.frame_selector.loss_weight_schedule.hole.end > 0.0
    assert cfg.model.rpn_head == official.model.rpn_head
    assert "physical_grid_actionformer" not in cfg.model.rpn_head
    assert "bata_value_transport" not in repr(cfg).lower()
    assert "ducaexternalactionnessfromjsonl" not in repr(cfg).lower()
    assert "duca_x3d" not in OFFICIAL_BACKEND_CONFIG.read_text(encoding="utf-8").lower()
    assert "ledger_path" not in repr(cfg.model).lower()
    assert cfg.model.backbone.backbone.total_frames == 384
    assert cfg.model.projection.max_seq_len == 384
    assert cfg.dataset.train.pipeline[2].method == "random_trunc"
    assert cfg.dataset.val.pipeline[2].method == "sliding_window"
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False


def test_duca_online_official_backend_validator_and_launcher_are_fail_closed():
    assert OFFICIAL_BACKEND_VALIDATOR.exists()
    output = subprocess.check_output(
        [
            sys.executable,
            str(OFFICIAL_BACKEND_VALIDATOR),
            "--config",
            str(OFFICIAL_BACKEND_CONFIG),
        ],
        cwd=str(ROOT),
        text=True,
    )
    summary = json.loads(output)
    assert summary["ok"] is True
    assert summary["official_adatad_backend"] is True
    assert summary["rpn_head_matches_official_base"] is True
    assert summary["physical_grid_actionformer_enabled"] is False
    assert summary["uses_ledger_for_decision"] is False
    assert summary["uses_external_actionness"] is False
    assert summary["requires_external_actionness"] is False
    assert summary["train_free_x3d_is_main_method"] is False
    assert summary["budget_lte_384"] is True
    assert summary["loss_schedule_policy"] == "progressive_joint"
    assert summary["loss_schedule_step_update"] == "optimizer_step"
    assert summary["loss_schedule_total_steps"] > 0
    assert summary["loss_schedule_warmup_fraction"] == 0.08
    assert summary["loss_schedule_transition_fraction"] == 0.67
    assert summary["loss_schedule_detector_loss_always_on"] is True
    assert summary["loss_schedule_detector_gradient_bridge_scheduled"] is True
    assert summary["loss_schedule_detector_gradient_start"] == 0.0
    assert summary["loss_schedule_detector_gradient_end"] == 1.0
    assert summary["loss_schedule_detector_utility_start"] == 0.0
    assert summary["loss_schedule_detector_utility_end"] > 0.0
    assert summary["loss_schedule_actionness_start"] > summary["loss_schedule_actionness_end"]

    text = OFFICIAL_BACKEND_LAUNCHER.read_text(encoding="utf-8")
    assert "duca_online_official_adatad_backend_full_train.py" in text
    assert "validate_duca_official_adatad_backend.py" in text
    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert "FULLTRAIN_CANDIDATE=1" in text
    assert 'if [[ -n "${SLURM_STEP_GPUS:-}${SLURM_JOB_GPUS:-}" ]]' in text
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"' in text
    assert 'SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-none}' in text
    assert "physical GPU1 outside Slurm remapping" in text
    assert "ADATAD_PRETRAIN_PATH" in text
    assert "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" in text
    assert "required file missing: ${ADATAD_PRETRAIN_PATH}" in text
    assert "tools/train.py" in text


def test_duca_online_official_backend_config_supports_env_budget_curve(monkeypatch):
    monkeypatch.setenv("DUCA_ONLINE_BUDGET", "256")

    cfg = _cfg(OFFICIAL_BACKEND_CONFIG)

    assert cfg.window_size == 256
    assert cfg.chunk_num == 16
    assert cfg.duca_online_main_contract.budget_max == 256
    assert cfg.model.frame_selector.budget == 256
    assert cfg.model.backbone.backbone.total_frames == 256
    assert cfg.model.projection.max_seq_len == 256


def test_duca_official_backend_loss_schedule_is_fraction_derived(monkeypatch):
    monkeypatch.setenv("DUCA_LOSS_SCHEDULE_TOTAL_STEPS", "1000")
    monkeypatch.setenv("DUCA_LOSS_SCHEDULE_WARMUP_FRACTION", "0.1")
    monkeypatch.setenv("DUCA_LOSS_SCHEDULE_TRANSITION_FRACTION", "0.4")

    fixed = _cfg(OFFICIAL_BACKEND_CONFIG)
    must = _cfg(DUCA_MUST_DYNAMIC_CONFIG)

    for cfg, contract_name in (
        (fixed, "duca_online_main_contract"),
        (must, "duca_must_dynamic_contract"),
    ):
        contract = getattr(cfg, contract_name)
        assert contract.loss_schedule_total_steps == 1000
        assert contract.loss_schedule_warmup_fraction == 0.1
        assert contract.loss_schedule_transition_fraction == 0.4
        assert cfg.model.frame_selector.loss_weight_schedule.warmup_steps == 100
        assert cfg.model.frame_selector.loss_weight_schedule.transition_steps == 400


def test_duca_online_official_backend_validator_allows_budget_curve_mode(monkeypatch):
    env = os.environ.copy()
    env["DUCA_ONLINE_BUDGET"] = "512"
    output = subprocess.check_output(
        [
            sys.executable,
            str(OFFICIAL_BACKEND_VALIDATOR),
            "--config",
            str(OFFICIAL_BACKEND_CONFIG),
            "--max-budget",
            "768",
        ],
        cwd=str(ROOT),
        env=env,
        text=True,
    )
    summary = json.loads(output)

    assert summary["ok"] is True
    assert summary["budget"] == 512
    assert summary["detector_consumed_length"] == 512
    assert summary["budget_lte_max"] is True
    assert summary["strict_budget_lte_384"] is False


def test_duca_online_official_backend_budget_curve_launcher_is_continuous_and_precheck_first():
    text = OFFICIAL_BACKEND_BUDGET_CURVE_LAUNCHER.read_text(encoding="utf-8")

    assert 'DUCA_ONLINE_BUDGET_START="${DUCA_ONLINE_BUDGET_START:-128}"' in text
    assert 'DUCA_ONLINE_BUDGET_END="${DUCA_ONLINE_BUDGET_END:-768}"' in text
    assert 'DUCA_ONLINE_BUDGET_STEP="${DUCA_ONLINE_BUDGET_STEP:-32}"' in text
    assert 'DUCA_ONLINE_BUDGET="${budget}"' in text
    assert 'DUCA_VALIDATOR_MAX_BUDGET="${DUCA_VALIDATOR_MAX_BUDGET:-${DUCA_ONLINE_BUDGET_END}}"' in text
    assert "run_duca_online_official_adatad_backend_gpu1.sh" in text
    assert 'PRECHECK_ONLY="${PRECHECK_ONLY}"' in text
    assert 'FULLTRAIN_CANDIDATE="${FULLTRAIN_CANDIDATE}"' in text


def test_duca_must_dynamic_main_config_declares_model_internal_budget_policy():
    cfg = _cfg(DUCA_MUST_DYNAMIC_CONFIG)
    official = _cfg(ROOT / "configs" / "adatad" / "thumos" / "e2e_thumos_videomae_s_768x1_160_adapter.py")
    text = DUCA_MUST_DYNAMIC_CONFIG.read_text(encoding="utf-8").lower()

    assert "duca_online_budget" not in text
    assert cfg.duca_must_dynamic_contract.main_method_candidate is True
    assert cfg.duca_must_dynamic_contract.dynamic_budget is True
    assert cfg.duca_must_dynamic_contract.acquisition_policy == "duca_center_radius_st_acquisition"
    assert cfg.duca_must_dynamic_contract.budget_policy == "prefix_marginal_utility_stop"
    assert cfg.duca_must_dynamic_contract.dynamic_budget_dual_update_after_optimizer_step is True
    assert cfg.duca_must_dynamic_contract.dynamic_budget_dual_update_source == "dynamic_must_expected_cost"
    assert cfg.duca_must_dynamic_contract.actionness_source == "online_trainable_c3_coarse_probe"
    assert cfg.duca_must_dynamic_contract.coarse_probe_joint_trainable is True
    assert cfg.duca_must_dynamic_contract.budget_max == 384
    assert cfg.duca_must_dynamic_contract.budget_target <= cfg.duca_must_dynamic_contract.budget_max
    assert cfg.duca_must_dynamic_contract.external_budget_override_allowed is False
    assert cfg.duca_must_dynamic_contract.runtime_flops_claim_allowed is False
    assert cfg.duca_must_dynamic_contract.actual_variable_length_detector is False
    assert cfg.model.type == "ActionFormer"
    assert cfg.model.frame_selector.type == "DucaOnlineFrameSelector"
    assert cfg.model.frame_selector.budget is None
    assert cfg.model.frame_selector.budget_mode == "dynamic_must"
    assert cfg.model.frame_selector.budget_max == 384
    assert cfg.model.frame_selector.budget_multiple == 16
    assert cfg.model.frame_selector.target_budget == 256
    assert cfg.model.frame_selector.allow_external_budget_override is False
    assert cfg.model.frame_selector.get("external_actionness_meta_key", None) is None
    assert cfg.model.frame_selector.get("external_actionness_logits_meta_key", None) is None
    assert cfg.model.frame_selector.get("require_external_actionness", False) is False
    assert cfg.model.frame_selector.forbid_external_actionness is True
    assert cfg.duca_must_dynamic_contract.forbid_external_actionness is True
    assert cfg.model.frame_selector.actionness_source_cfg.type == "C3CoarseProbeActionnessSource"
    assert cfg.model.frame_selector.actionness_source_cfg.probe_model == "official-action-seg"
    assert cfg.model.frame_selector.actionness_source_cfg.official_action_seg_backend == "official_asformer"
    assert cfg.model.frame_selector.actionness_source_cfg.tcn_variant == "official_asformer"
    assert cfg.model.frame_selector.actionness_source_cfg.get("tcn_variant", None) != "asformer_lite"
    assert cfg.model.frame_selector.actionness_source_cfg.trainable is True
    assert cfg.model.frame_selector.actionness_source_cfg.frozen is False
    assert cfg.model.frame_selector.loss_weights.actionness > 0
    assert cfg.model.frame_selector.loss_weights.detector_utility > 0
    assert cfg.model.frame_selector.loss_weights.hole > 0
    assert cfg.duca_must_dynamic_contract.loss_schedule_policy == "progressive_joint"
    assert cfg.duca_must_dynamic_contract.loss_schedule_step_update == "optimizer_step"
    assert cfg.duca_must_dynamic_contract.loss_schedule_total_steps == (
        cfg.workflow.end_epoch * cfg.duca_schedule_steps_per_epoch
    )
    assert cfg.duca_must_dynamic_contract.loss_schedule_warmup_fraction == 0.08
    assert cfg.duca_must_dynamic_contract.loss_schedule_transition_fraction == 0.67
    assert cfg.model.frame_selector.loss_weight_schedule.type == "progressive_joint"
    assert cfg.model.frame_selector.loss_weight_schedule.warmup_steps == round(
        cfg.duca_must_dynamic_contract.loss_schedule_total_steps
        * cfg.duca_must_dynamic_contract.loss_schedule_warmup_fraction
    )
    assert cfg.model.frame_selector.loss_weight_schedule.transition_steps == round(
        cfg.duca_must_dynamic_contract.loss_schedule_total_steps
        * cfg.duca_must_dynamic_contract.loss_schedule_transition_fraction
    )
    assert cfg.model.frame_selector.loss_weight_schedule.actionness.start > cfg.model.frame_selector.loss_weight_schedule.actionness.end
    assert cfg.duca_must_dynamic_contract.detector_loss_always_trains_backend is True
    assert cfg.model.frame_selector.loss_weight_schedule.detector_gradient.start == 0.0
    assert cfg.model.frame_selector.loss_weight_schedule.detector_gradient.end == 1.0
    assert cfg.model.frame_selector.loss_weight_schedule.detector_utility.start == 0.0
    assert cfg.model.frame_selector.loss_weight_schedule.detector_utility.end > 0.0
    assert cfg.model.frame_selector.loss_weight_schedule.hole.start == 0.0
    assert cfg.model.frame_selector.loss_weight_schedule.hole.end > 0.0
    assert cfg.model.frame_selector.loss_weight_schedule.lagrangian_budget.start == 0.0
    assert cfg.model.frame_selector.loss_weight_schedule.lagrangian_budget.end > 0.0
    assert cfg.model.rpn_head == official.model.rpn_head
    assert "ducaexternalactionnessfromjsonl" not in repr(cfg).lower()
    assert "duca_x3d" not in text
    assert cfg.window_size == 384
    assert cfg.model.backbone.backbone.total_frames == 384
    assert cfg.model.projection.max_seq_len == 384


def test_duca_must_dynamic_validator_is_fail_closed_and_separate_from_forced_budget_curve():
    assert DUCA_MUST_DYNAMIC_VALIDATOR.exists()
    output = subprocess.check_output(
        [
            sys.executable,
            str(DUCA_MUST_DYNAMIC_VALIDATOR),
            "--config",
            str(DUCA_MUST_DYNAMIC_CONFIG),
        ],
        cwd=str(ROOT),
        text=True,
    )
    summary = json.loads(output)

    assert summary["ok"] is True
    assert summary["dynamic_budget"] is True
    assert summary["budget_policy"] == "prefix_marginal_utility_stop"
    assert summary["loss_schedule_step_update"] == "optimizer_step"
    assert summary["dynamic_budget_dual_update_after_optimizer_step"] is True
    assert summary["dynamic_budget_dual_update_source"] == "dynamic_must_expected_cost"
    assert summary["budget_max_lte_384"] is True
    assert summary["external_budget_override_allowed"] is False
    assert summary["uses_external_actionness"] is False
    assert summary["requires_external_actionness"] is False
    assert summary["train_free_x3d_is_main_method"] is False
    assert summary["uses_env_budget_override"] is False
    assert summary["runtime_flops_claim_allowed"] is False
    assert summary["forced_budget_curve"] is False
    assert summary["loss_schedule_policy"] == "progressive_joint"
    assert summary["loss_schedule_total_steps"] > 0
    assert summary["loss_schedule_warmup_fraction"] == 0.08
    assert summary["loss_schedule_transition_fraction"] == 0.67
    assert summary["loss_schedule_detector_loss_always_on"] is True
    assert summary["loss_schedule_detector_gradient_bridge_scheduled"] is True
    assert summary["loss_schedule_detector_gradient_start"] == 0.0
    assert summary["loss_schedule_detector_gradient_end"] == 1.0
    assert summary["loss_schedule_detector_utility_start"] == 0.0
    assert summary["loss_schedule_detector_utility_end"] > 0.0
    assert summary["loss_schedule_actionness_start"] > summary["loss_schedule_actionness_end"]
    assert summary["loss_schedule_lagrangian_budget_start"] == 0.0
    assert summary["loss_schedule_lagrangian_budget_end"] > 0.0


def test_duca_must_dynamic_launcher_is_precheck_first_and_not_forced_budget_curve():
    assert DUCA_MUST_DYNAMIC_LAUNCHER.exists()
    text = DUCA_MUST_DYNAMIC_LAUNCHER.read_text(encoding="utf-8")

    assert "duca_must_dynamic_official_adatad_backend_full_train.py" in text
    assert "validate_duca_must_dynamic_official_adatad_backend.py" in text
    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert "FULLTRAIN_CANDIDATE" in text
    assert 'if [[ -n "${SLURM_STEP_GPUS:-}${SLURM_JOB_GPUS:-}" ]]' in text
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"' in text
    assert 'SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-none}' in text
    assert "physical GPU1 outside Slurm remapping" in text
    assert "ADATAD_PRETRAIN_PATH" in text
    assert "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" in text
    assert "required file missing: ${ADATAD_PRETRAIN_PATH}" in text
    assert "formal full train must run inside a Slurm allocation/step" in text
    assert "tools/train.py" in text
    assert "DUCA_ONLINE_BUDGET" not in text


def test_duca_online_validator_json_summary_fields(monkeypatch):
    monkeypatch.setenv("DUCA_ONLINE_PRECHECK_RUNTIME", "0")
    validator = _load_validator()

    summary = validator.validate_config(str(ADATAD_CONFIG), run_runtime=False)

    assert summary["config_import"] is True
    assert summary["build_detector"] == "skipped"
    assert summary["no_ledger"] is True
    assert summary["teacher_only_train_loss"] is True
    assert summary["no_teacher_in_inference"] is True
    assert summary["gt_reaches_detector_train"] == "runtime_required"
    assert summary["budget_lte_384"] is True
    assert summary["selected_positions_original_time"] == "runtime_required"
    assert summary["masks_selected_count"] == "runtime_required"
    assert summary["remap_metadata_present"] == "runtime_required"
    assert summary["raw_prediction_cache_forbidden"] is True

    output = subprocess.check_output(
        [
            sys.executable,
            str(VALIDATOR),
            "--config",
            str(ADATAD_CONFIG),
            "--no-runtime",
        ],
        cwd=str(ROOT),
        text=True,
    )
    cli_summary = json.loads(output)
    assert cli_summary["config_path"].endswith("duca_online_adatad_precheck.py")
    assert cli_summary["build_detector"] == "skipped"

    for config in (ACTIONFORMER_NO_PHYSICAL_CONFIG, ACTIONFORMER_PHYSICAL_CONFIG):
        summary = validator.validate_config(str(config), run_runtime=False)
        assert summary["config_import"] is True
        assert summary["rpn_head_type"] == "ActionFormerHead"
        assert summary["budget_lte_384"] is True
        assert summary["raw_prediction_cache_forbidden"] is True


def test_duca_online_launchers_are_precheck_first_gpu1_guarded_and_fail_closed():
    for launcher in (ADATAD_LAUNCHER, ZEROSHOT_LAUNCHER):
        text = launcher.read_text(encoding="utf-8")
        assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
        assert "FULLTRAIN_CANDIDATE" in text
        assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"' in text
        assert 'if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]' in text
        assert 'SLURM_JOB_GPUS:-' in text
        assert 'SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-none}' in text
        assert "validate_duca_online_adatad_precheck.py" in text
        assert "tests/test_duca_online_precheck_config.py" in text
        assert "formal full train must run inside a Slurm allocation/step" in text
        assert "tools/train.py" in text

    launcher_text = ACTIONFORMER_NO_PHYSICAL_LAUNCHER.read_text(encoding="utf-8")
    assert "duca_online_actionformer_no_physical_grid_precheck.py" in launcher_text
    assert "run_duca_online_adatad_precheck_gpu1.sh" in launcher_text
    launcher_text = ACTIONFORMER_PHYSICAL_LAUNCHER.read_text(encoding="utf-8")
    assert "duca_online_actionformer_physical_grid_precheck.py" in launcher_text
    assert "run_duca_online_adatad_precheck_gpu1.sh" in launcher_text


def _assert_x3d_external_pipeline(cfg, *, dynamic: bool) -> None:
    selector = cfg.model.frame_selector
    assert selector.external_actionness_meta_key == "duca_external_p_action"
    assert selector.external_actionness_logits_meta_key == "duca_external_actionness_logits"
    assert selector.external_actionness_provenance_meta_key == "duca_external_actionness_provenance"
    assert selector.external_actionness_source_meta_key == "duca_external_actionness_source"
    assert selector.require_external_actionness is True
    assert selector.actionness_source_cfg.uses_gt is False
    assert selector.actionness_source_cfg.uses_labels is False
    assert selector.actionness_source_cfg.uses_teacher is False

    for split in ("train", "val", "test"):
        pipeline = cfg.dataset[split].pipeline
        assert pipeline[3].type == "DucaExternalActionnessFromJsonl"
        assert pipeline[3].actionness_jsonl == cfg.duca_x3d_actionness_jsonl
        collect = pipeline[-1]
        assert "duca_external_p_action" in collect.meta_keys
        assert "duca_external_actionness_logits" in collect.meta_keys
        assert "duca_external_actionness_provenance" in collect.meta_keys
        assert "duca_external_actionness_source" in collect.meta_keys
    if dynamic:
        assert cfg.duca_must_dynamic_contract.external_actionness_source == "train_free_x3d_jsonl"
        assert cfg.duca_must_dynamic_contract.requires_external_actionness is True
    else:
        assert cfg.duca_online_main_contract.external_actionness_source == "train_free_x3d_jsonl"
        assert cfg.duca_online_main_contract.requires_external_actionness is True


def test_duca_x3d_official_backend_config_wires_jsonl_actionness_to_selector():
    cfg = _cfg(X3D_OFFICIAL_BACKEND_CONFIG)

    assert cfg.model.type == "ActionFormer"
    assert cfg.model.frame_selector.type == "DucaOnlineFrameSelector"
    assert cfg.model.frame_selector.budget == 384
    assert cfg.duca_online_main_contract.x3d_downstream_detector_full_train is True
    assert cfg.duca_online_main_contract.uses_offline_deploy_selection_ledger is False
    _assert_x3d_external_pipeline(cfg, dynamic=False)


def test_duca_must_dynamic_x3d_backend_config_wires_jsonl_actionness_to_selector():
    cfg = _cfg(X3D_MUST_DYNAMIC_CONFIG)

    assert cfg.model.type == "ActionFormer"
    assert cfg.model.frame_selector.type == "DucaOnlineFrameSelector"
    assert cfg.model.frame_selector.budget is None
    assert cfg.model.frame_selector.budget_mode == "dynamic_must"
    assert cfg.duca_must_dynamic_contract.x3d_downstream_detector_full_train is True
    assert cfg.duca_must_dynamic_contract.uses_offline_deploy_selection_ledger is False
    _assert_x3d_external_pipeline(cfg, dynamic=True)


def test_duca_x3d_backend_validators_require_external_actionness():
    for validator, config in (
        (X3D_OFFICIAL_BACKEND_VALIDATOR, X3D_OFFICIAL_BACKEND_CONFIG),
        (X3D_MUST_DYNAMIC_VALIDATOR, X3D_MUST_DYNAMIC_CONFIG),
    ):
        assert validator.exists()
        output = subprocess.check_output(
            [
                sys.executable,
                str(validator),
                "--config",
                str(config),
            ],
            cwd=str(ROOT),
            text=True,
        )
        summary = json.loads(output)
        assert summary["ok"] is True
        assert summary["external_actionness_source"] == "train_free_x3d_jsonl"
        assert summary["requires_external_actionness"] is True


def test_duca_x3d_backend_launchers_require_jsonl_and_delegate_to_official_scripts():
    expectations = (
        (
            X3D_OFFICIAL_BACKEND_LAUNCHER,
            "duca_online_x3d_official_adatad_backend_full_train.py",
            "validate_duca_x3d_official_adatad_backend.py",
            "run_duca_online_official_adatad_backend_gpu1.sh",
        ),
        (
            X3D_MUST_DYNAMIC_LAUNCHER,
            "duca_must_dynamic_x3d_official_adatad_backend_full_train.py",
            "validate_duca_must_dynamic_x3d_official_adatad_backend.py",
            "run_duca_must_dynamic_official_adatad_backend_gpu1.sh",
        ),
    )
    for launcher, config_name, validator_name, delegate_name in expectations:
        assert launcher.exists()
        text = launcher.read_text(encoding="utf-8")
        assert config_name in text
        assert validator_name in text
        assert delegate_name in text
        assert "DUCA_X3D_ACTIONNESS_JSONL" in text
        assert "DUCA_X3D_REQUIRE_JSONL_EXISTS=1" in text
