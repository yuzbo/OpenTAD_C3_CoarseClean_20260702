from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "bata" / "run_duca_jct_one_step_grad_proof.py"
FIXED_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "duca_online_official_adatad_backend_full_train.py"
MUST_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "duca_must_dynamic_official_adatad_backend_full_train.py"
SUITE = ROOT / "scripts" / "submit_duca_jct_experiment_suite.sh"


def test_duca_jct_one_step_grad_proof_writes_joint_training_artifact(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("local Windows torch/c10.dll is unstable; Linux remote focused job runs this proof")
    output = tmp_path / "grad_proof.json"
    env = os.environ.copy()
    env.update(
        {
            "DUCA_ONLINE_DENSE_WINDOW_SIZE": "16",
            "DUCA_ONLINE_BUDGET": "16",
            "DUCA_MUST_DENSE_WINDOW_SIZE": "16",
            "DUCA_MUST_BUDGET_MAX": "16",
            "DUCA_MUST_BUDGET_MIN": "4",
            "DUCA_MUST_BUDGET_TARGET": "8",
            "DUCA_MUST_BUDGET_MULTIPLE": "4",
            "DUCA_COARSE_SPATIAL_SIZE": "16",
            "DUCA_COARSE_HIDDEN_DIM": "16",
            "DUCA_LOSS_SCHEDULE_WARMUP_STEPS": "0",
            "DUCA_LOSS_SCHEDULE_TRANSITION_STEPS": "1",
        }
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixed-config",
            str(FIXED_CONFIG),
            "--must-config",
            str(MUST_CONFIG),
            "--output-json",
            str(output),
        ],
        cwd=str(ROOT),
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "duca_jct_one_step_grad_proof_v1"
    assert payload["proof_passed"] is True
    assert payload["fixed384"]["official_config_loaded"] is True
    assert payload["fixed384"]["optimizer_step_ran"] is True
    assert payload["fixed384"]["coarse_probe_grad_sum"] > 0.0
    assert payload["fixed384"]["selector_encoder_grad_sum"] > 0.0
    assert payload["fixed384"]["budget_controller_grad_sum"] is None
    assert payload["fixed384"]["loss_schedule_step_update"]["updated"] is True
    assert payload["fixed384"]["loss_schedule_step_update"]["source"] == "optimizer_step"
    assert payload["fixed384"]["dynamic_budget_dual_update"] is None
    assert payload["duca_must"]["official_config_loaded"] is True
    assert payload["duca_must"]["optimizer_step_ran"] is True
    assert payload["duca_must"]["coarse_probe_grad_sum"] > 0.0
    assert payload["duca_must"]["selector_encoder_grad_sum"] > 0.0
    assert payload["duca_must"]["budget_controller_grad_sum"] > 0.0
    assert payload["duca_must"]["loss_schedule_step_update"]["updated"] is True
    assert payload["duca_must"]["loss_schedule_step_update"]["source"] == "optimizer_step"
    assert payload["duca_must"]["dynamic_budget_dual_update"]["updated"] is True
    assert payload["duca_must"]["loss_schedule"]["type"] == "progressive_joint"


def test_duca_jct_suite_runs_one_step_grad_proof_before_full_runs() -> None:
    text = SUITE.read_text(encoding="utf-8")

    assert "run_duca_jct_one_step_grad_proof.py" in text
    assert "duca_jct_one_step_grad_proof.json" in text
    assert "tests/test_duca_jct_one_step_grad_proof.py" in text
