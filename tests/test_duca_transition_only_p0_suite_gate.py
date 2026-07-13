from __future__ import annotations

from pathlib import Path
import json

import pytest
from mmengine.config import Config

from tools.bata import validate_duca_transition_only_p0_suite as suite


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_suite_gate_from_concurrently_changing_variant_contract(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(suite, "validate_variant", lambda name, path: {"ok": True, "variant": name, "config_path": path})


def _gate(tmp_path: Path) -> Path:
    gate = tmp_path / "formal_gate.json"
    gate.write_text(json.dumps({"ok": True, "formal_proof_ok": True,
        "git_commit": suite._git(ROOT, "rev-parse", "HEAD"),
        "uniform_reference_definition": "round_linspace_endpoints",
        "uniform_reference_exact": True}), encoding="utf-8")
    return gate


def test_suite_gate_emits_four_arm_matched_manifest(tmp_path: Path) -> None:
    payload = suite.validate_suite(repo_root=ROOT, seed=17, core_gate_json=_gate(tmp_path))
    assert payload["ok"] is True
    assert payload["status"] == "deployable_not_submitted"
    assert payload["submission_performed"] is False
    assert payload["seed"] == 17
    assert [item["name"] for item in payload["variants"]] == list(suite.VARIANT_ORDER)
    assert len(payload["shared_protocol_sha256"]) == 64
    protocol = payload["shared_protocol"]
    assert protocol["budget"] == 384
    assert protocol["dense_window_size"] == 768
    assert protocol["expected_optimizer_steps"] == 13200
    assert protocol["detector_type"] == "ActionFormer"
    assert protocol["detector_head"]["type"] == "ActionFormerHead"
    assert protocol["solver"]["ema"] is True
    assert protocol["evaluation"]["type"] == "mAP"
    assert payload["post_run_contract"]["successful_optimizer_updates"] == 13200


def test_suite_gate_rejects_any_shared_protocol_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original = suite._shared_protocol

    def drift(cfg: Config):
        payload = original(cfg)
        if float(cfg.model.frame_selector.loss_weight_schedule.detector_gradient.end) == 0.0:
            payload["budget"] = 383
        return payload

    monkeypatch.setattr(suite, "_shared_protocol", drift)
    with pytest.raises(AssertionError, match="shared protocol differs"):
        suite.validate_suite(repo_root=ROOT, seed=0, core_gate_json=_gate(tmp_path))


def test_suite_gate_rejects_wrong_commit() -> None:
    with pytest.raises(AssertionError, match="expected commit"):
        suite.validate_suite(repo_root=ROOT, expected_commit="0" * 40)


def test_suite_gate_requires_formal_core_gate() -> None:
    with pytest.raises(AssertionError, match="formal core gate is required"):
        suite.validate_suite(repo_root=ROOT)


def test_suite_gate_binds_formal_gate_to_current_commit(tmp_path: Path) -> None:
    commit = suite._git(ROOT, "rev-parse", "HEAD")
    gate = tmp_path / "formal_gate.json"
    gate.write_text(
        json.dumps(
            {
                "ok": True,
                "formal_proof_ok": True,
                "git_commit": commit,
                "uniform_reference_definition": "round_linspace_endpoints",
                "uniform_reference_exact": True,
            }
        ),
        encoding="utf-8",
    )
    payload = suite.validate_suite(repo_root=ROOT, core_gate_json=gate)
    assert payload["formal_core_gate"]["git_commit"] == commit
    assert payload["formal_core_gate"]["sha256"] == suite._sha256(gate)

    stale = json.loads(gate.read_text(encoding="utf-8"))
    stale["git_commit"] = "0" * 40
    gate.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(AssertionError, match="commit is stale"):
        suite.validate_suite(repo_root=ROOT, core_gate_json=gate)


def test_prepare_script_is_generation_only_and_fail_closed() -> None:
    text = (ROOT / "scripts/prepare_duca_transition_only_p0_suite.sh").read_text(encoding="utf-8")
    assert "sbatch " not in text
    assert "PREPARED_NOT_SUBMITTED" in text
    assert "--require-clean" in text
    assert "DUCA_CORE_GATE_JSON must name an existing formal gate" in text
    assert "DUCA_EXPECTED_COMMIT" in text
    assert "-m tools.bata.validate_duca_transition_only_p0_suite" in text
    assert "printf 'variant\\tseed" in text
    assert 'bash -n "${job_file}"' in text
    for variant in suite.VARIANT_ORDER:
        assert variant in text


def test_post_run_contract_checks_updates_lr_ema_and_evaluator(tmp_path: Path) -> None:
    protocol_hash = "a" * 64
    payload = {"ok": True, "variant": "uniform", **suite._post_run_contract(protocol_hash)}
    evidence = tmp_path / "post_run.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    result = suite.validate_post_run_evidence(evidence, variant="uniform", protocol_sha256=protocol_hash)
    assert result["validated"] is True
    payload["successful_optimizer_updates"] -= 1
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="successful_optimizer_updates"):
        suite.validate_post_run_evidence(evidence, variant="uniform", protocol_sha256=protocol_hash)
