from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from mmengine.config import Config
from tools.bata import validate_duca_transition_only_p0_variant as p0_validator


validate_variant = p0_validator.validate_variant


ROOT = "configs/adatad/thumos/"
PATHS = {
    "uniform": ROOT + "duca_exact_uniform_fixed384_official_adatad_backend_full_train.py",
    "direct": ROOT + "duca_direct_boundary_fixed384_13200_official_adatad_backend_full_train.py",
    "transition_beta0": ROOT + "duca_transition_only_fixed384_no_detector_bridge_official_adatad_backend_full_train.py",
    "transition_counterfactual": ROOT + "duca_transition_only_fixed384_official_adatad_backend_full_train.py",
}
LAUNCHER = Path("scripts/run_duca_transition_only_p0_variant_gpu1.sh")
PREPARER = Path("scripts/prepare_duca_transition_only_p0_suite.sh")
CANONICAL_ENV = Path("scripts/duca_transition_only_p0_canonical_env.sh")
BASE_CONFIG = Path("configs/adatad/thumos/duca_online_official_adatad_backend_full_train.py")


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def test_p0_matrix_is_matched_on_detector_data_geometry_and_training_horizon() -> None:
    configs = {name: Config.fromfile(path) for name, path in PATHS.items()}
    reference = configs["transition_counterfactual"]

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
        assert cfg.model.frame_selector.actionness_source_cfg.calibration_split == "none", name
        assert cfg.model.frame_selector.actionness_source_cfg.get("calibration_artifact") in (None, ""), name


def test_p0_matrix_changes_only_the_intended_selector_mechanism() -> None:
    uniform = Config.fromfile(PATHS["uniform"])
    direct = Config.fromfile(PATHS["direct"])
    beta0 = Config.fromfile(PATHS["transition_beta0"])
    beta025 = Config.fromfile(PATHS["transition_counterfactual"])

    assert uniform.model.frame_selector.selector_variant == "transition_only"
    assert uniform.model.frame_selector.inference_policy_alpha == 0.0
    assert uniform.model.frame_selector.loss_weight_schedule.policy_alpha.end == 0.0
    assert uniform.model.frame_selector.loss_weight_schedule.detector_gradient.end == 0.0
    assert uniform.model.frame_selector.counterfactual_utility_distillation_weight == 0.0
    assert uniform.model.frame_selector.require_counterfactual_utility_teacher is False

    assert direct.model.frame_selector.selector_variant == "direct_boundary"
    assert direct.duca_loss_schedule_total_steps == 13200
    assert direct.duca_schedule_steps_per_epoch == 100

    assert beta0.model.frame_selector.selector_variant == "transition_only"
    assert beta0.model.frame_selector.loss_weight_schedule.policy_alpha.end == 1.0
    assert beta0.model.frame_selector.loss_weight_schedule.detector_gradient.end == 0.0
    assert beta0.model.frame_selector.counterfactual_utility_distillation_weight == 0.0
    assert beta0.model.frame_selector.require_counterfactual_utility_teacher is False

    assert beta025.model.frame_selector.selector_variant == "transition_only"
    assert beta025.model.frame_selector.loss_weight_schedule.policy_alpha.end == 1.0
    assert beta025.model.frame_selector.loss_weight_schedule.detector_gradient.end == 0.0
    assert beta025.model.frame_selector.counterfactual_utility_distillation_weight > 0.0
    assert beta025.model.frame_selector.require_counterfactual_utility_teacher is True
    assert beta025.model.frame_selector.soft_max_gap_loss_enabled is False


def test_p0_variant_validator_accepts_every_declared_variant() -> None:
    for variant, path in PATHS.items():
        summary = validate_variant(variant, path)
        assert summary["ok"] is True
        assert summary["variant"] == variant
        assert summary["budget"] == 384
        assert summary["paper_claim_allowed"] is False


def test_direct_control_rejects_invalid_environment_before_silent_override(monkeypatch) -> None:
    monkeypatch.setenv("DUCA_SELECTOR_ACTIONNESS_WEIGHT", "9.0")
    monkeypatch.setenv("DUCA_STRUCTURED_TEMPERATURE", "9.0")
    monkeypatch.setenv("DUCA_LOSS_BOUNDARY_END", "9.0")
    with pytest.raises(ValueError, match="small auxiliary term"):
        Config.fromfile(PATHS["direct"])


def test_p0_core_gate_requires_ok_and_matching_commit(tmp_path: Path) -> None:
    commit = "a" * 40
    config_sha = "b" * 64
    source_sha = "c" * 64
    checkpoint_sha = "d" * 64
    gate_json = tmp_path / "core_gate.json"
    gate_json.write_text(
        json.dumps(
            {
                "ok": True,
                "formal_proof_ok": True,
                "git_commit": commit,
                "reference_config_sha256": config_sha,
                "official_asformer_source_sha256": source_sha,
                "checkpoint_sha256": checkpoint_sha,
                "dense_window_size": 768,
                "selected_count": 384,
                "model_type": "ActionFormer",
                "detector_head_type": "ActionFormerHead",
                "uniform_reference_definition": "round_linspace_endpoints",
                "uniform_reference_exact": True,
                "uniform_reference_max_rank_error": 0,
            }
        ),
        encoding="utf-8",
    )

    summary = p0_validator.validate_core_gate(
        gate_json,
        expected_commit=commit,
        expected_config_sha256=config_sha,
        expected_source_sha256=source_sha,
        expected_checkpoint_sha256=checkpoint_sha,
    )

    assert summary == {
        "ok": True,
        "git_commit": commit,
        "path": str(gate_json),
        "uniform_reference_exact": True,
    }


def test_p0_core_gate_rejects_failed_or_stale_evidence(tmp_path: Path) -> None:
    commit = "a" * 40
    gate_json = tmp_path / "core_gate.json"
    gate_json.write_text(json.dumps({"ok": False, "git_commit": commit}), encoding="utf-8")
    with pytest.raises(AssertionError, match="ok=true"):
        p0_validator.validate_core_gate(
            gate_json,
            expected_commit=commit,
            expected_config_sha256="b" * 64,
            expected_source_sha256="c" * 64,
            expected_checkpoint_sha256="d" * 64,
        )

    gate_json.write_text(json.dumps({"ok": True, "git_commit": "b" * 40}), encoding="utf-8")
    with pytest.raises(AssertionError, match="git_commit"):
        p0_validator.validate_core_gate(
            gate_json,
            expected_commit=commit,
            expected_config_sha256="b" * 64,
            expected_source_sha256="c" * 64,
            expected_checkpoint_sha256="d" * 64,
        )


def test_p0_core_gate_rejects_unverified_uniform_reference(tmp_path: Path) -> None:
    commit = "a" * 40
    gate_json = tmp_path / "core_gate.json"
    gate_json.write_text(
        json.dumps(
            {
                "ok": True,
                "formal_proof_ok": True,
                "git_commit": commit,
                "reference_config_sha256": "b" * 64,
                "official_asformer_source_sha256": "c" * 64,
                "checkpoint_sha256": "d" * 64,
                "dense_window_size": 768,
                "selected_count": 384,
                "model_type": "ActionFormer",
                "detector_head_type": "ActionFormerHead",
                "uniform_reference_definition": "midpoint_distance",
                "uniform_reference_exact": False,
                "uniform_reference_max_rank_error": 180,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="uniform reference"):
        p0_validator.validate_core_gate(
            gate_json,
            expected_commit=commit,
            expected_config_sha256="b" * 64,
            expected_source_sha256="c" * 64,
            expected_checkpoint_sha256="d" * 64,
        )

def test_p0_launcher_uses_commit_bound_core_gate_and_auditable_manifest() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "DUCA_CORE_GATE_PASSED" not in text
    assert 'DUCA_CORE_GATE_JSON="${DUCA_CORE_GATE_JSON:-}"' in text
    assert 'DUCA_EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-${CURRENT_HEAD}}"' in text
    assert '--core-gate-json "${DUCA_CORE_GATE_JSON}"' in text
    assert '--expected-commit "${DUCA_EXPECTED_COMMIT}"' in text
    assert '--expected-config-sha256 "${REFERENCE_CONFIG_SHA256}"' in text
    assert '--expected-source-sha256 "${SOURCE_SHA256}"' in text
    assert '--expected-checkpoint-sha256 "${CHECKPOINT_SHA256}"' in text
    assert 'RUNTIME_SUITE_MANIFEST="${RUN_DIR}/runtime_suite_manifest.json"' in text
    assert "runtime {label} hash drift" in text
    assert "P0 full train must run inside Slurm" in text
    assert "git status --porcelain --untracked-files=normal" in text
    assert text.index("git status --porcelain --untracked-files=normal") < text.index('cat > "${RUN_DIR}/manifest.json"')
    for field in ("config_sha256", "source_sha256", "checkpoint_sha256"):
        assert f'"{field}"' in text


def test_p0_preparer_and_runtime_share_canonical_config_environment() -> None:
    prepare = PREPARER.read_text(encoding="utf-8")
    runtime = LAUNCHER.read_text(encoding="utf-8")
    helper = CANONICAL_ENV.read_text(encoding="utf-8")
    source_line = 'source "${REPO_ROOT}/scripts/duca_transition_only_p0_canonical_env.sh"'
    assert source_line in prepare
    assert source_line in runtime
    assert 'export DUCA_PROFILE_RUNTIME=0' in helper
    assert 'export DUCA_PROFILE_SYNC_CUDA=1' in helper
    assert 'export DUCA_ONLINE_BUDGET=384' in helper
    assert 'export DUCA_ONLINE_DENSE_WINDOW_SIZE=768' in helper
    assert 'export DUCA_OFFICIAL_ADATAD_END_EPOCH=132' in helper
    assert 'export DUCA_LOSS_SCHEDULE_TOTAL_STEPS=13200' in helper
    assert 'DUCA_PROFILE_RUNTIME="${DUCA_PROFILE_RUNTIME:-0}"' not in runtime

    key_block = helper.split("DUCA_P0_CANONICAL_ENV_KEYS=(", 1)[1].split(")", 1)[0]
    canonical_keys = set(re.findall(r"^\s+([A-Z][A-Z0-9_]*)\s*$", key_block, re.MULTILINE))
    config_text = BASE_CONFIG.read_text(encoding="utf-8")
    config_env_keys = set(
        re.findall(
            r'(?:_env_(?:int|float|bool)|os\.environ\.get)\(\s*"([A-Z][A-Z0-9_]*)"',
            config_text,
        )
    )
    assert config_env_keys <= canonical_keys
    assert {"C3_OFFICIAL_ACTION_SEG_REPOS", "ADATAD_PRETRAIN_PATH", "PYTHON"} <= canonical_keys

    for binding in (
        "DUCA_CANONICAL_ENV_FILE",
        "DUCA_CANONICAL_ENV_SHA256",
        "RUNTIME_CANONICAL_ENV_SHA256",
        "PREPARED_CANONICAL_ENV_SHA256",
    ):
        assert binding in runtime or binding in prepare
    assert 'cmp -s "${DUCA_CANONICAL_ENV_FILE}" "${RUNTIME_CANONICAL_ENV}"' in runtime
