from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "bata" / "run_duca_official_adatad_one_step_grad_proof.py"
FIXED_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "duca_online_official_adatad_backend_full_train.py"
MUST_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "duca_must_dynamic_official_adatad_backend_full_train.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("duca_official_grad_proof", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_grad_proof_script_is_fail_closed_against_precheck_head() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "DucaOnlinePrecheckHead" in text
    assert "ActionFormerHead" in text
    assert "official detector backend" in text
    assert "fail-closed" in text


def test_official_optimizer_groups_cover_every_trainable_frame_selector_parameter() -> None:
    if os.name == "nt":
        pytest.skip("local Windows torch/c10.dll import is unstable; Linux remote runs this coverage check")

    proof = _load_script_module()
    model, summary = proof.build_official_proof_model(
        config_path=FIXED_CONFIG,
        route="fixed384",
        proof_temporal_len=32,
        proof_budget=16,
        proof_budget_min=4,
        proof_budget_target=8,
        proof_budget_multiple=4,
        proof_spatial_size=16,
        proof_hidden_dim=16,
        proof_feature_dim=16,
    )

    coverage = proof.assert_frame_selector_optimizer_coverage(model, lr=1.0e-4, weight_decay=0.05)

    assert summary["rpn_head_type"] == "ActionFormerHead"
    assert coverage["missing_frame_selector_params"] == []
    assert coverage["covered_frame_selector_param_count"] == coverage["trainable_frame_selector_param_count"]


def test_official_adatad_one_step_cost_backward_reaches_probe_selector_and_budget_controller() -> None:
    if os.name == "nt":
        pytest.skip("local Windows torch/c10.dll import is unstable; Linux remote runs this proof")

    proof = _load_script_module()
    payload = proof.run_proof(
        fixed_config=FIXED_CONFIG,
        must_config=MUST_CONFIG,
        proof_temporal_len=32,
        proof_budget=16,
        proof_budget_min=4,
        proof_budget_target=8,
        proof_budget_multiple=4,
        proof_spatial_size=16,
        proof_hidden_dim=16,
        proof_feature_dim=16,
        device="cpu",
    )

    assert payload["schema_version"] == "duca_official_adatad_one_step_grad_proof_v1"
    assert payload["proof_passed"] is True
    for route in ("fixed384", "duca_must"):
        result = payload[route]
        assert result["model_type"] == "ActionFormer"
        assert result["rpn_head_type"] == "ActionFormerHead"
        assert result["proof_uses_precheck_head"] is False
        assert "cls_loss" in result["loss_keys"]
        assert "reg_loss" in result["loss_keys"]
        assert result["coarse_probe_grad_sum"] > 0.0
        assert result["selector_encoder_grad_sum"] > 0.0
        assert result["selector_center_head_grad_sum"] > 0.0
        assert result["proof_schedule_phase"] == "joint_detection_selection"
        assert result["optimizer_coverage"]["missing_frame_selector_params"] == []
    assert payload["duca_must"]["budget_controller_grad_sum"] > 0.0
