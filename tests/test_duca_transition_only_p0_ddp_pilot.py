from __future__ import annotations

import json
from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata import validate_duca_transition_only_p0_ddp_pilot as pilot
from tools.bata.validate_duca_transition_only_p0_ddp_pilot import (
    EXPECTED_STEPS,
    PILOT_CONFIGS,
    VARIANT_ORDER,
    validate_probe,
)


def _probe(variant: str) -> dict:
    budget_vectors = [
        [384, 384],
        [353, 384],
        [300, 312],
        [384, 384],
        [350, 384],
        [286, 310],
        [384, 384],
        [360, 384],
        [301, 333],
        [384, 384],
    ]
    steps = []
    for index, effective_budget in enumerate(budget_vectors):
        step = {
            "effective_budget": effective_budget,
            "requested_budget": [384, 384],
            "detector_gradient_weight": 0.0 if index < 3 else 0.5,
            "policy_mix_alpha": 0.0 if index < 3 else 0.75,
            "optimizer_step_ran": True,
        }
        if variant == "transition_counterfactual":
            step["counterfactual"] = {
                "candidate_count": 0 if index == 0 else 4,
                "finite": True,
                "teacher_kind": "detached_hard_one_swap_official_actionformer_cls_plus_reg",
            }
        steps.append(step)
    failed_step = dict(steps[0])
    failed_step["optimizer_step_ran"] = False
    if "counterfactual" in failed_step:
        failed_step["counterfactual"] = dict(failed_step["counterfactual"])
    steps.insert(0, failed_step)
    return {
        "schema_version": "duca_training_probe_v1",
        "attempted_steps": EXPECTED_STEPS + 1,
        "successful_optimizer_steps": EXPECTED_STEPS,
        "skipped_optimizer_steps": 1,
        "finite_loss_steps": EXPECTED_STEPS + 1,
        "finite_gradient_steps": EXPECTED_STEPS,
        "static_graph": False,
        "find_unused_parameters": True,
        "world_size": 1,
        "parameter_group_coverage": {
            group: {"trainable": 2, "gradient_seen": 2}
            for group in ("backbone", "coarse_probe", "selector", "projection", "detector_head")
        },
        "gradient_never_seen": [],
        "gradient_seen": ["module.backbone.adapter.weight"],
        "selector_steps": steps,
        "update_audit": {
            "attempted_batches": EXPECTED_STEPS,
            "optimizer_attempts": EXPECTED_STEPS + 1,
            "successful_optimizer_updates": EXPECTED_STEPS,
            "amp_skipped_attempts": 1,
            "replayed_batches": 1,
            "replay_exhaustions": 0,
            "scheduler_updates": EXPECTED_STEPS,
            "ema_updates": EXPECTED_STEPS,
            "duca_schedule_updates": EXPECTED_STEPS,
            "forced_amp_overflow_attempts": 1,
            "max_amp_retries_observed": 1,
        },
        "max_cuda_memory_mb": 8471.0,
    }


def test_pilot_configs_preserve_dynamic_ddp_and_disable_only_pilot_side_effects() -> None:
    for variant, path in PILOT_CONFIGS.items():
        cfg = Config.fromfile(path)
        assert cfg.model.backbone.backbone.with_cp is False, variant
        assert cfg.solver.static_graph is False, variant
        assert cfg.solver.find_unused_parameters is True, variant
        assert cfg.workflow.max_train_iters == EXPECTED_STEPS, variant
        assert cfg.workflow.disable_checkpoint is True, variant
        assert cfg.workflow.require_training_probe_context is True, variant
        assert cfg.workflow.end_epoch == 1, variant
        assert cfg.workflow.val_start_epoch > cfg.workflow.end_epoch, variant
        assert cfg.workflow.checkpoint_interval == 5, variant


def test_pilot_validator_accepts_required_batch_and_schedule_coverage() -> None:
    for variant in VARIANT_ORDER:
        summary = validate_probe(_probe(variant), variant)
        assert summary["successful_optimizer_steps"] == EXPECTED_STEPS
        assert all(summary["budget_coverage"].values())


def test_pilot_validator_rejects_static_graph_and_missing_parameter_path() -> None:
    static = _probe("direct")
    static["static_graph"] = True
    with pytest.raises(AssertionError, match="static_graph"):
        validate_probe(static, "direct")

    disconnected = _probe("transition_beta0")
    disconnected["parameter_group_coverage"]["coarse_probe"]["gradient_seen"] = 0
    with pytest.raises(AssertionError, match="coarse_probe has trainable parameters never receiving gradient"):
        validate_probe(disconnected, "transition_beta0")


def test_pilot_validator_requires_both_counterfactual_candidate_paths() -> None:
    missing_zero = _probe("transition_counterfactual")
    for step in missing_zero["selector_steps"]:
        step["counterfactual"]["candidate_count"] = 4
    with pytest.raises(AssertionError, match="zero-candidate"):
        validate_probe(missing_zero, "transition_counterfactual")


def test_formal_pilot_artifact_reopens_and_revalidates_raw_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    probes = run_root / "probes"
    probes.mkdir(parents=True)
    checkpoint = run_root / "checkpoint.pth"
    source = run_root / "model.py"
    canonical_env = run_root / "canonical_env.tsv"
    checkpoint.write_bytes(b"checkpoint")
    source.write_text("official", encoding="utf-8")
    canonical_env.write_text("KEY=VALUE\n", encoding="utf-8")
    root = Path.cwd().resolve()
    reference_config = (
        root
        / "configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py"
    ).resolve()
    manifest_path = run_root / "manifest.json"
    manifest = {
        "schema_version": "duca_p0_ddp_pilot_run_v1",
        "git_commit": "a" * 40,
        "core_gate_json_sha256": "b" * 64,
        "shared_protocol_sha256": "c" * 64,
        "pilot_nonce": "123-a-nonce",
        "seed": 0,
        "slurm_job_id": "123",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": pilot._sha256(checkpoint),
        "official_asformer_source": str(source.resolve()),
        "official_asformer_source_sha256": pilot._sha256(source),
        "canonical_env_path": str(canonical_env.resolve()),
        "canonical_env_sha256": pilot._sha256(canonical_env),
        "reference_config_path": str(reference_config),
        "reference_config_sha256": pilot._sha256(reference_config),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = pilot._sha256(manifest_path)
    monkeypatch.setattr(pilot, "_resolved_pilot_config_sha256", lambda context: "d" * 64)

    variants = []
    for variant in VARIANT_ORDER:
        config_path = (root / PILOT_CONFIGS[variant]).resolve()
        probe_path = (probes / f"{variant}.training_probe.json").resolve()
        context_path = (probes / f"{variant}.context.json").resolve()
        context = {
            "schema_version": "duca_p0_ddp_pilot_context_v1",
            "git_commit": manifest["git_commit"],
            "variant": variant,
            "seed": 0,
            "slurm_job_id": "123",
            "pilot_nonce": manifest["pilot_nonce"],
            "source_config_path": str(config_path),
            "source_config_sha256": pilot._sha256(config_path),
            "training_probe_json": str(probe_path),
            "context_json": str(context_path),
            "work_dir": str((run_root / "work_dirs" / variant).resolve()),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "core_gate_json_sha256": manifest["core_gate_json_sha256"],
            "shared_protocol_sha256": manifest["shared_protocol_sha256"],
            "run_manifest_path": str(manifest_path.resolve()),
            "run_manifest_sha256": manifest_sha,
        }
        context_path.write_text(json.dumps(context), encoding="utf-8")
        probe_payload = _probe(variant)
        probe_payload["bindings"] = {
            "git_commit": manifest["git_commit"],
            "seed": 0,
            "slurm_job_id": "123",
            "source_config_path": str(config_path),
            "source_config_sha256": context["source_config_sha256"],
            "resolved_config_sha256": "d" * 64,
            "training_probe_json": str(probe_path),
            "context_json": str(context_path),
            "context_json_sha256": pilot._sha256(context_path),
            "context": context,
        }
        probe_path.write_text(json.dumps(probe_payload), encoding="utf-8")
        summary = validate_probe(probe_payload, variant)
        variants.append(
            {
                **summary,
                "pilot_config": PILOT_CONFIGS[variant],
                "pilot_config_sha256": pilot._sha256(config_path),
                "probe_json": str(probe_path),
                "probe_json_sha256": pilot._sha256(probe_path),
                "context_json": str(context_path),
                "context_json_sha256": pilot._sha256(context_path),
                "validated_probe_summary_sha256": pilot._canonical_sha256(summary),
            }
        )
    artifact_path = run_root / "ddp_pilot_suite.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": "duca_p0_ddp_pilot_suite_v1",
                "ok": True,
                "git_commit": manifest["git_commit"],
                "seed": 0,
                "slurm_job_id": "123",
                "pilot_nonce": manifest["pilot_nonce"],
                "shared_protocol_sha256": manifest["shared_protocol_sha256"],
                "core_gate_json_sha256": manifest["core_gate_json_sha256"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "official_asformer_source_sha256": manifest["official_asformer_source_sha256"],
                "reference_config_sha256": manifest["reference_config_sha256"],
                "run_manifest_path": str(manifest_path.resolve()),
                "run_manifest_sha256": manifest_sha,
                "variants": variants,
            }
        ),
        encoding="utf-8",
    )
    verified = pilot.validate_pilot_artifact(
        artifact_path,
        repo_root=root,
        expected_commit=manifest["git_commit"],
        expected_protocol_sha256=manifest["shared_protocol_sha256"],
        expected_core_gate_sha256=manifest["core_gate_json_sha256"],
        expected_checkpoint_sha256=manifest["checkpoint_sha256"],
        expected_official_asformer_source_sha256=manifest["official_asformer_source_sha256"],
        expected_reference_config_sha256=manifest["reference_config_sha256"],
    )
    assert verified["seed"] == 0
    assert [item["variant"] for item in verified["variants"]] == list(VARIANT_ORDER)

    stale_probe = Path(variants[0]["probe_json"])
    stale_payload = json.loads(stale_probe.read_text(encoding="utf-8"))
    stale_payload["successful_optimizer_steps"] = 9
    stale_probe.write_text(json.dumps(stale_payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="probe hash mismatch"):
        pilot.validate_pilot_artifact(
            artifact_path,
            repo_root=root,
            expected_commit=manifest["git_commit"],
            expected_protocol_sha256=manifest["shared_protocol_sha256"],
            expected_core_gate_sha256=manifest["core_gate_json_sha256"],
            expected_checkpoint_sha256=manifest["checkpoint_sha256"],
            expected_official_asformer_source_sha256=manifest["official_asformer_source_sha256"],
            expected_reference_config_sha256=manifest["reference_config_sha256"],
        )
