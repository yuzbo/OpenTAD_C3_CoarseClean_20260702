from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "bata" / "validate_duca_stage23_precheck.py"
STAGE3_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_truetime_joint_selector_adatad_precheck.py"
STAGE3_EXEC_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_truetime_joint_selector_adatad_precheck_exec.py"
STAGE3_RUNNER = ROOT / "scripts" / "run_duca_stage3_truetime_precheck_gpu1.sh"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_duca_stage23_precheck_test", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_duca_stage23_validator_exists_and_accepts_real_stage3_precheck_proof(tmp_path: Path) -> None:
    proof = tmp_path / "stage3_precheck_proof.json"
    summary = tmp_path / "stage3_precheck.summary.json"
    proof.write_text(
        json.dumps(
            {
                "route_variant": "DIVERGENT_INNOVATION_TRUETIME_JOINT_SELECTOR_DO_NOT_MERGE_WITH_C3",
                "stage": "stage3_true_time_e2e_adatad_selector_precheck",
                "geometry_roundtrip_passed": True,
                "prediction_inverse_map_passed": True,
                "selected_input_st_gradient_passed": True,
                "selected_input_selector_grad_norm": 0.17,
                "detector_loss_selector_grad_passed": True,
                "detector_loss_selector_grad_norm": 0.23,
                "selector_grad_norm": 0.23,
                "selector_grad_nonzero": True,
                "real_detector_proof_source": "opentad_actionformer_forward_train_cost_backward",
                "real_detector_loss_selector_grad_passed": True,
                "real_detector_loss_selector_grad_norm": 0.23,
                "real_detector_loss_keys": ["cls_loss", "reg_loss"],
                "actionformer_proof_source": "opentad_actionformer_forward_train_cost_backward",
                "actionformer_detector_loss_selector_grad_passed": True,
                "actionformer_detector_loss_selector_grad_norm": 0.23,
                "actionformer_loss_keys": ["cls_loss", "reg_loss"],
                "actionformer_selected_axis_smoke": False,
                "actionformer_physical_grid_precheck": True,
                "sparse_distill_adapter_ready": True,
                "sparse_distill_claim_allowed": False,
                "sparse_distill_map_claim_allowed": False,
                "sparse_distill_proof_source": "fail_closed_sparse_detector_distillation_adapter",
            }
        ),
        encoding="utf-8",
    )

    validator = _load_validator()
    payload = validator.main(
        [
            "--stage",
            "stage3",
            "--summary-json",
            str(summary),
            "--stage3-config",
            str(STAGE3_CONFIG),
            "--stage3-exec-config",
            str(STAGE3_EXEC_CONFIG),
            "--require-stage3-grad-proof",
            "--stage3-grad-proof-json",
            str(proof),
        ]
    )

    assert payload == 0
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    stage3 = summary_payload["stage3"]
    assert stage3["stage3_config_sha256"] == _sha256_file(STAGE3_CONFIG)
    assert stage3["stage3_exec_config_sha256"] == _sha256_file(STAGE3_EXEC_CONFIG)
    assert stage3["proof"]["proof_json_sha256"] == _sha256_file(proof)
    assert "matching config/proof hashes" in stage3["full_run_gate"]


def test_duca_stage3_runner_full_run_does_not_delegate_to_smoke_launcher() -> None:
    text = STAGE3_RUNNER.read_text(encoding="utf-8")

    assert "run_c3_truetime_joint_selector_adatad_gpu1.sh" not in text
    assert "run_truetime_joint_selector_smoke.py" not in text
    assert "run_truetime_joint_selector_precheck.py" in text
    assert "tools/train.py" in text
    assert "ALLOW_TRUETIME_JOINT_SELECTOR_FULLTRAIN" in text
    assert "proof_json_sha256" in text
    assert "stale precheck summary" in text
    assert "bound config/proof hashes" in text


def test_duca_stage3_precheck_config_keeps_default_precheck_contract_when_full_run_env(monkeypatch) -> None:
    monkeypatch.setenv("PRECHECK_ONLY", "0")
    cfg = Config.fromfile(str(STAGE3_CONFIG))

    assert cfg.truetime_joint_selector_gate.precheck_only_default is True
    assert cfg.truetime_joint_selector_gate.allow_long_training is True
    assert cfg.truetime_joint_selector_gate.smoke_only is False
