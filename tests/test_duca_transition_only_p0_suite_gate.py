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
        "uniform_reference_exact": True,
        "optimizer_step_ran": True,
        "optimizer_parameter_change_verified": True,
        "optimizer_changed_parameter_groups": ["detector_head"],
        "optimizer_parameter_max_abs_change": {"detector_head": 1e-5},
        "optimizer_step_loss": 1.0,
        "optimizer_step_loss_finite": True,
        "optimizer_step_gradients_finite": True,
        "loss_normalizer_contract": {
            "state_kind": "ActionFormerHead.loss_normalizer_ema_buffer",
            "finite": True, "positive": True,
            "updated_by_training_forward": True,
            "unchanged_by_optimizer_step": True,
            "before_forward": 100.0, "after_forward": 91.0,
            "after_optimizer_step": 91.0,
        }}), encoding="utf-8")
    return gate


def test_suite_gate_emits_four_arm_matched_manifest(tmp_path: Path) -> None:
    payload = suite.validate_suite(repo_root=ROOT, seed=17, core_gate_json=_gate(tmp_path))
    assert payload["ok"] is True
    assert payload["status"] == "core_gate_only_not_deployable"
    assert payload["formal_ddp_pilot"] is None
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
    assert protocol["backbone"]["backbone"]["with_cp"] is False
    assert protocol["solver"]["static_graph"] is False
    assert protocol["solver"]["find_unused_parameters"] is True
    assert protocol["evaluation"]["type"] == "mAP"
    assert payload["post_run_contract"]["uniform"]["successful_optimizer_updates"] == 13200
    for variant in payload["variants"]:
        assert len(variant["resolved_config_sha256"]) == 64
        assert len(variant["variant_contract_sha256"]) == 64
        assert variant["variant_contract"]["selector_variant"] in {"direct_boundary", "transition_only"}


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


def test_suite_gate_requires_ddp_pilot_when_formal_deployment_is_requested(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="requires a DDP pilot"):
        suite.validate_suite(
            repo_root=ROOT,
            core_gate_json=_gate(tmp_path),
            require_ddp_pilot=True,
        )


def test_suite_gate_rejects_handwritten_aggregate_without_raw_pilot_evidence(tmp_path: Path) -> None:
    gate = _gate(tmp_path)
    core_only = suite.validate_suite(repo_root=ROOT, core_gate_json=gate)
    pilot = tmp_path / "ddp_pilot.json"
    pilot.write_text(
        json.dumps(
            {
                "schema_version": "duca_p0_ddp_pilot_suite_v1",
                "ok": True,
                "git_commit": core_only["git_commit"],
                "shared_protocol_sha256": core_only["shared_protocol_sha256"],
                "core_gate_json_sha256": suite._sha256(gate),
                "variants": [
                    {"variant": name, "successful_optimizer_steps": 10}
                    for name in suite.VARIANT_ORDER
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="run manifest"):
        suite.validate_suite(
            repo_root=ROOT,
            core_gate_json=gate,
            ddp_pilot_json=pilot,
            require_ddp_pilot=True,
        )


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
                "optimizer_step_ran": True,
                "optimizer_parameter_change_verified": True,
                "optimizer_changed_parameter_groups": ["detector_head"],
                "optimizer_parameter_max_abs_change": {"detector_head": 1e-5},
                "optimizer_step_loss": 1.0,
                "optimizer_step_loss_finite": True,
                "optimizer_step_gradients_finite": True,
                "loss_normalizer_contract": {
                    "state_kind": "ActionFormerHead.loss_normalizer_ema_buffer",
                    "finite": True, "positive": True,
                    "updated_by_training_forward": True,
                    "unchanged_by_optimizer_step": True,
                    "before_forward": 100.0, "after_forward": 91.0,
                    "after_optimizer_step": 91.0,
                },
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
    assert "DUCA_DDP_PILOT_JSON must name an existing four-arm pilot" in text
    assert "--require-ddp-pilot" in text
    assert "DUCA_EXPECTED_COMMIT" in text
    assert "-m tools.bata.validate_duca_transition_only_p0_suite" in text
    assert "printf 'variant\\tseed" in text
    assert 'bash -n "${job_file}"' in text
    assert "export CUDA_VISIBLE_DEVICES=1" not in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "DUCA_RESOLVED_CONFIG_SHA256" in text
    assert "DUCA_VARIANT_CONTRACT_SHA256" in text
    assert "DUCA_SHARED_PROTOCOL_SHA256" in text
    for variant in suite.VARIANT_ORDER:
        assert variant in text


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("optimizer_step_ran", False, "did not run optimizer.step"),
        ("optimizer_parameter_change_verified", False, "trainable parameter change"),
        ("optimizer_changed_parameter_groups", [], "changed no parameter group"),
        ("optimizer_step_loss_finite", False, "loss is non-finite"),
        ("optimizer_step_gradients_finite", False, "gradients are non-finite"),
    ],
)
def test_suite_gate_rejects_incomplete_optimizer_step_contract(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    gate = _gate(tmp_path)
    payload = json.loads(gate.read_text(encoding="utf-8"))
    payload[field] = value
    gate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match=message):
        suite.validate_suite(repo_root=ROOT, core_gate_json=gate)


def test_gpu1_launcher_is_strict() -> None:
    text = (ROOT / "scripts/run_duca_transition_only_p0_variant_gpu1.sh").read_text(encoding="utf-8")
    assert '[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]' in text
    assert "torch.cuda.device_count()" in text
    assert '[[ "${VISIBLE_GPU_COUNT}" == "1" ]]' in text
    assert '== "0"' not in text


def test_post_run_contract_checks_updates_lr_ema_and_evaluator(tmp_path: Path) -> None:
    protocol_hash = "a" * 64
    bindings = {
        "git_commit": "b" * 40,
        "seed": 0,
        "config_sha256": "c" * 64,
        "resolved_config_sha256": "d" * 64,
        "variant_contract_sha256": "e" * 64,
        "core_gate_sha256": "f" * 64,
    }
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(json.dumps({
        "variant": "uniform", "git_commit": bindings["git_commit"], "seed": 0,
        "config_sha256": bindings["config_sha256"],
        "resolved_config_sha256": bindings["resolved_config_sha256"],
        "variant_contract_sha256": bindings["variant_contract_sha256"],
        "core_gate_json_sha256": bindings["core_gate_sha256"],
        "shared_protocol_sha256": protocol_hash,
    }), encoding="utf-8")
    payload = {
        "ok": True, "variant": "uniform", **suite._post_run_contract(protocol_hash, bindings),
        "run_manifest_path": str(run_manifest), "run_manifest_sha256": suite._sha256(run_manifest),
    }
    evidence = tmp_path / "post_run.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    result = suite.validate_post_run_evidence(
        evidence, variant="uniform", protocol_sha256=protocol_hash, bindings=bindings
    )
    assert result["validated"] is True
    payload["successful_optimizer_updates"] -= 1
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="successful_optimizer_updates"):
        suite.validate_post_run_evidence(
            evidence, variant="uniform", protocol_sha256=protocol_hash, bindings=bindings
        )


@pytest.mark.parametrize(
    "field",
    ["git_commit", "config_sha256", "resolved_config_sha256", "variant_contract_sha256", "core_gate_sha256"],
)
def test_post_run_contract_rejects_stale_provenance(tmp_path: Path, field: str) -> None:
    protocol_hash = "a" * 64
    bindings = {
        "git_commit": "b" * 40,
        "seed": 0,
        "config_sha256": "c" * 64,
        "resolved_config_sha256": "d" * 64,
        "variant_contract_sha256": "e" * 64,
        "core_gate_sha256": "f" * 64,
    }
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(json.dumps({
        "variant": "uniform", "git_commit": bindings["git_commit"], "seed": 0,
        "config_sha256": bindings["config_sha256"],
        "resolved_config_sha256": bindings["resolved_config_sha256"],
        "variant_contract_sha256": bindings["variant_contract_sha256"],
        "core_gate_json_sha256": bindings["core_gate_sha256"],
        "shared_protocol_sha256": protocol_hash,
    }), encoding="utf-8")
    payload = {
        "ok": True, "variant": "uniform", **suite._post_run_contract(protocol_hash, bindings),
        "run_manifest_path": str(run_manifest), "run_manifest_sha256": suite._sha256(run_manifest),
    }
    payload[field] = "0" * len(str(payload[field]))
    evidence = tmp_path / "post_run.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match=field):
        suite.validate_post_run_evidence(
            evidence, variant="uniform", protocol_sha256=protocol_hash, bindings=bindings
        )
