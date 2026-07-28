from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mmengine.config import Config
import pytest

from tools.bata import duca_rime_training
from tools.bata.duca_p0_evaluation import evaluation_config_sha256


COMMIT = "a" * 40


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _json(path: Path, payload) -> Path:
    return _write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rime_runtime_binding_is_phase2_split_exposure_and_hash_bound(tmp_path, monkeypatch):
    monkeypatch.delenv("DUCA_RIME_REPLAY_JSONL", raising=False)
    monkeypatch.delenv("DUCA_RIME_REPLAY_SHA256", raising=False)
    protocol = _json(
        tmp_path / "protocol.json",
        {
            "schema_version": "duca_rime_budget_protocol_v1",
            "fit_split": "train_only",
            "uses_validation_or_test_labels": False,
            "target_mean_cost": 384.0,
        },
    )
    phase2 = _json(
        tmp_path / "phase2.json",
        {
            "schema_version": "duca_rime_stage_receipt_v1",
            "phase": "phase2",
            "gate_pass": True,
            "phase3_training_authorized": True,
            "official_final_subset_consumed": False,
            "git_commit": COMMIT,
            "split_assignment_sha256": "b" * 64,
            "formal_budget_protocols": [
                {
                    "target_mean_cost": 384.0,
                    "path": str(protocol.resolve()),
                    "sha256": _sha(protocol),
                },
                {
                    "target_mean_cost": 192.0,
                    "path": str(protocol.resolve()),
                    "sha256": _sha(protocol),
                },
            ],
        },
    )
    exposure = _json(
        tmp_path / "exposure.json",
        {
            "schema_version": "duca_rime_phase3_training_exposure_v1",
            "successful_detector_updates": 6000,
            "split_assignment_sha256": "b" * 64,
            "official_final_subset_consumed": False,
        },
    )
    pretrain = _write(tmp_path / "pretrain.pth", "pretrain")
    annotation = _write(tmp_path / "annotation.json", "{}")
    class_map = _write(tmp_path / "classes.txt", "action\n")
    targets = _write(tmp_path / "targets.jsonl", "{}\n")
    config = _write(tmp_path / "config.py", "# config\n")
    evaluation = {
        "type": "mAP",
        "ground_truth_filename": str(annotation),
        "subset": "training",
        "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        "top_k": None,
        "blocked_videos": None,
        "thread": 16,
    }
    values = {
        "DUCA_RIME_PHASE2_RECEIPT": str(phase2),
        "DUCA_RIME_PHASE2_RECEIPT_SHA256": _sha(phase2),
        "DUCA_RIME_TRAINING_EXPOSURE_JSON": str(exposure),
        "DUCA_RIME_TRAINING_EXPOSURE_SHA256": _sha(exposure),
        "DUCA_RIME_PRETRAIN_SHA256": _sha(pretrain),
        "DUCA_RIME_EVALUATION_ANNOTATION_SHA256": _sha(annotation),
        "DUCA_RIME_EVALUATION_CLASS_MAP_SHA256": _sha(class_map),
        "DUCA_RIME_TARGETS_JSONL": str(targets),
        "DUCA_RIME_TARGETS_SHA256": _sha(targets),
        "DUCA_RIME_BUDGET_PROTOCOL_JSON": str(protocol),
        "DUCA_RIME_BUDGET_PROTOCOL_SHA256": _sha(protocol),
        "DUCA_RESOLVED_CONFIG_SHA256": "c" * 64,
        "DUCA_RIME_EVALUATION_CONFIG_SHA256": evaluation_config_sha256(
            evaluation,
            expected_subset="training",
        ),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    bindings = duca_rime_training.build_runtime_bindings(
        git_commit=COMMIT,
        variant="RIME-full",
        seed=3407,
        slurm_job_id="123",
        source_config_path=config,
        source_config_sha256=_sha(config),
        resolved_config_sha256="c" * 64,
        runtime_config_sha256="d" * 64,
        evaluation_annotation_path=annotation,
        evaluation_class_map_path=class_map,
        evaluation_config=evaluation,
        runtime_pretrain_path=pretrain,
    )
    assert bindings["phase2_receipt_sha256"] == _sha(phase2)
    assert bindings["training_exposure_sha256"] == _sha(exposure)
    assert bindings["initialization_sha256"] == _sha(pretrain)
    assert bindings["targets_sha256"] == _sha(targets)
    assert bindings["budget_protocol_sha256"] == _sha(protocol)
    assert bindings["official_final_subset_consumed"] is False


def test_train_and_test_entrypoints_route_rime_formal_protocol():
    root = Path(__file__).resolve().parents[1]
    train = (root / "tools" / "train.py").read_text(encoding="utf-8")
    test = (root / "tools" / "test.py").read_text(encoding="utf-8")
    assert "duca_rime_training.is_formal_protocol(formal_protocol)" in train
    assert "duca_rime_training.is_formal_protocol(formal_protocol)" in test
    assert "validate_terminal_checkpoint_binding" in test
    assert "duca_rime_terminal_evaluation_v1" in test


def test_checkpoint_compatibility_modes_are_explicit_and_fail_closed():
    strict = duca_rime_training.validate_phase2_baseline_checkpoint_compatibility(
        missing_keys=[],
        unexpected_keys=[],
        mode=duca_rime_training.STRICT_EXACT_CHECKPOINT_COMPATIBILITY_MODE,
    )
    assert strict == {
        "mode": "strict_exact_v1",
        "missing_keys": [],
        "ignored_unexpected_keys": [],
    }
    with pytest.raises(RuntimeError, match="zero missing"):
        duca_rime_training.validate_phase2_baseline_checkpoint_compatibility(
            missing_keys=["module.projection.weight"],
            unexpected_keys=[],
            mode=duca_rime_training.STRICT_EXACT_CHECKPOINT_COMPATIBILITY_MODE,
        )
    with pytest.raises(RuntimeError, match="zero missing"):
        duca_rime_training.validate_phase2_baseline_checkpoint_compatibility(
            missing_keys=[],
            unexpected_keys=["module.legacy.weight"],
            mode=duca_rime_training.STRICT_EXACT_CHECKPOINT_COMPATIBILITY_MODE,
        )
    with pytest.raises(RuntimeError, match="unregistered"):
        duca_rime_training.validate_phase2_baseline_checkpoint_compatibility(
            missing_keys=[],
            unexpected_keys=[],
            mode="silent_partial_load",
        )


def test_phase1_dense_config_uses_strict_exact_checkpoint_contract(
    tmp_path,
    monkeypatch,
):
    root = Path(__file__).resolve().parents[1]
    block = _write(tmp_path / "block.txt", "blocked_video\n")
    monkeypatch.setenv("DUCA_RIME_PHASE1_EVAL_BLOCK_LIST", str(block))
    monkeypatch.setenv("DUCA_RIME_PHASE1_DENSE_VARIANT", "released_dense")
    cfg = Config.fromfile(
        str(
            root
            / "configs/adatad/thumos/duca_rime_dense_phase1_control.py"
        )
    )
    assert cfg.workflow.formal_protocol == "duca_rime_phase1_dense_control_v1"
    assert cfg.duca_rime_baseline_contract.phase == 1
    assert cfg.duca_rime_baseline_contract.variant == "released_dense"
    assert (
        cfg.duca_rime_baseline_contract.checkpoint_compatibility_mode
        == duca_rime_training.STRICT_EXACT_CHECKPOINT_COMPATIBILITY_MODE
    )
    assert cfg.evaluation.subset == "training"
    assert cfg.evaluation.blocked_videos == str(block)


def test_phase1_uniform_config_uses_registered_historical_checkpoint_contract(
    tmp_path,
    monkeypatch,
):
    root = Path(__file__).resolve().parents[1]
    block = _write(tmp_path / "block.txt", "blocked_video\n")
    monkeypatch.setenv("DUCA_RIME_PHASE2_EVAL_BLOCK_LIST", str(block))
    monkeypatch.setenv("DUCA_RIME_FIXED_BUDGET", "192")
    cfg = Config.fromfile(
        str(
            root
            / "configs/adatad/thumos/duca_rime_uniform_phase1_control.py"
        )
    )
    assert cfg.workflow.formal_protocol == "duca_protected_physical_v1"
    assert cfg.duca_rime_baseline_contract.phase == 1
    assert cfg.duca_rime_baseline_contract.variant == "uniform_k192"
    assert cfg.duca_rime_baseline_contract.position_policy == "exact_uniform"
    assert cfg.duca_rime_baseline_contract.target_mean_cost == 192.0
    assert (
        cfg.duca_rime_baseline_contract.checkpoint_compatibility_mode
        == duca_rime_training.PHASE2_BASELINE_CHECKPOINT_COMPATIBILITY_MODE
    )
    assert cfg.evaluation.blocked_videos == str(block)


def test_rime_total60_configs_activate_dedicated_6000_update_contract(
    tmp_path,
    monkeypatch,
):
    root = Path(__file__).resolve().parents[1]
    block = _write(tmp_path / "block.txt", "blocked_video\n")
    targets = _write(tmp_path / "targets.jsonl", "{}\n")
    protocol = _write(tmp_path / "protocol.json", "{}")
    monkeypatch.setenv("DUCA_RIME_TRAIN_BLOCK_LIST", str(block))
    monkeypatch.setenv("DUCA_RIME_DEVELOPMENT_BLOCK_LIST", str(block))
    monkeypatch.setenv("DUCA_RIME_TARGETS_JSONL", str(targets))
    monkeypatch.setenv("DUCA_RIME_TARGETS_SHA256", _sha(targets))
    monkeypatch.setenv("DUCA_RIME_BUDGET_PROTOCOL_JSON", str(protocol))
    monkeypatch.setenv("DUCA_RIME_BUDGET_PROTOCOL_SHA256", _sha(protocol))
    for relative in (
        "configs/adatad/thumos/duca_rime_uniform_mixed_k_total60.py",
        "configs/adatad/thumos/duca_rime_uniform_fixed384_total60.py",
        "configs/adatad/thumos/duca_rime_full_total60.py",
        "configs/adatad/thumos/duca_rime_full_tridet_total60.py",
    ):
        cfg = Config.fromfile(str(root / relative))
        contract = duca_rime_training.formal_training_contract(cfg)
        assert contract["expected_successful_optimizer_updates"] == 6000
        assert contract["rime_arm"] == cfg.duca_rime_variant.arm


def test_phase2_mixed_k_runtime_is_phase1_bound_and_target_free(
    tmp_path,
    monkeypatch,
):
    for key in (
        "DUCA_RIME_PHASE2_RECEIPT",
        "DUCA_RIME_PHASE2_RECEIPT_SHA256",
        "DUCA_RIME_TARGETS_JSONL",
        "DUCA_RIME_TARGETS_SHA256",
        "DUCA_RIME_BUDGET_PROTOCOL_JSON",
        "DUCA_RIME_BUDGET_PROTOCOL_SHA256",
        "DUCA_RIME_REPLAY_JSONL",
        "DUCA_RIME_REPLAY_SHA256",
        "DUCA_RIME_PHASE4_AUTHORIZATION",
        "DUCA_RIME_PHASE4_AUTHORIZATION_SHA256",
    ):
        monkeypatch.delenv(key, raising=False)
    phase1 = _json(
        tmp_path / "phase1.json",
        {
            "schema_version": "duca_rime_stage_receipt_v1",
            "phase": "phase1",
            "gate_pass": True,
            "git_commit": COMMIT,
            "split_assignment_sha256": "b" * 64,
            "official_final_subset_consumed": False,
        },
    )
    exposure = _json(
        tmp_path / "phase2_exposure.json",
        {
            "schema_version": "duca_rime_phase2_mixed_k_training_exposure_v1",
            "git_commit": COMMIT,
            "seed": 3407,
            "detector_backend": "ActionFormer",
            "target_mean_cost": 384.0,
            "successful_detector_updates": 6000,
            "split_assignment_sha256": "b" * 64,
            "official_final_subset_consumed": False,
        },
    )
    pretrain = _write(tmp_path / "pretrain.pth", "pretrain")
    annotation = _write(tmp_path / "annotation.json", "{}")
    class_map = _write(tmp_path / "classes.txt", "action\n")
    config = _write(tmp_path / "config.py", "# config\n")
    evaluation = {
        "type": "mAP",
        "ground_truth_filename": str(annotation),
        "subset": "training",
        "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        "top_k": None,
        "blocked_videos": None,
        "thread": 16,
    }
    values = {
        "DUCA_RIME_PHASE1_RECEIPT": str(phase1),
        "DUCA_RIME_PHASE1_RECEIPT_SHA256": _sha(phase1),
        "DUCA_RIME_TRAINING_EXPOSURE_JSON": str(exposure),
        "DUCA_RIME_TRAINING_EXPOSURE_SHA256": _sha(exposure),
        "DUCA_RIME_PRETRAIN_SHA256": _sha(pretrain),
        "DUCA_RIME_EVALUATION_ANNOTATION_SHA256": _sha(annotation),
        "DUCA_RIME_EVALUATION_CLASS_MAP_SHA256": _sha(class_map),
        "DUCA_RESOLVED_CONFIG_SHA256": "c" * 64,
        "DUCA_RIME_EVALUATION_CONFIG_SHA256": evaluation_config_sha256(
            evaluation,
            expected_subset="training",
        ),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    bindings = duca_rime_training.build_runtime_bindings(
        git_commit=COMMIT,
        variant="U-mixed-K",
        seed=3407,
        slurm_job_id="123",
        source_config_path=config,
        source_config_sha256=_sha(config),
        resolved_config_sha256="c" * 64,
        runtime_config_sha256="d" * 64,
        evaluation_annotation_path=annotation,
        evaluation_class_map_path=class_map,
        evaluation_config=evaluation,
        runtime_pretrain_path=pretrain,
    )
    assert bindings["research_phase"] == 2
    assert bindings["phase1_receipt_sha256"] == _sha(phase1)
    assert bindings["phase2_receipt_path"] is None
    assert bindings["targets_path"] is None
    assert bindings["budget_protocol_path"] is None
    assert bindings["formal_budget_panel"] == 384.0


def test_mixed_k_evaluation_contract_reports_the_executed_budget(
    tmp_path,
    monkeypatch,
):
    root = Path(__file__).resolve().parents[1]
    block = _write(tmp_path / "block.txt", "blocked_video\n")
    monkeypatch.setenv("DUCA_RIME_TRAIN_BLOCK_LIST", str(block))
    monkeypatch.setenv("DUCA_RIME_DEVELOPMENT_BLOCK_LIST", str(block))
    monkeypatch.setenv("DUCA_RIME_EVAL_FIXED_BUDGET", "192")
    cfg = Config.fromfile(
        str(
            root
            / "configs/adatad/thumos/duca_rime_uniform_mixed_k_total60.py"
        )
    )
    assert cfg.duca_rime_variant.training_target_mean_cost == 384.0
    assert cfg.duca_rime_variant.training_schedule_seed == 3407
    assert (
        cfg.duca_rime_variant.training_schedule_source
        == "stateless_epoch_plus_sample_index"
    )
    assert cfg.duca_rime_variant.exact_per_video_histogram is True
    assert cfg.duca_rime_contract.training_target_mean_cost == 384.0
    assert cfg.duca_rime_contract.target_mean_cost == 192.0


def test_phase4_authorization_covers_only_registered_seed_backend_and_budget(
    tmp_path,
    monkeypatch,
):
    authorization = _json(
        tmp_path / "phase4.json",
        {
            "schema_version": "duca_rime_stage_receipt_v1",
            "phase": "phase4_authorization",
            "status": "authorized",
            "gate_pass": True,
            "git_commit": COMMIT,
            "formal_seeds": [5801, 8123, 12011],
            "required_detectors": ["ActionFormer", "TriDet"],
            "required_budget_panels": [384, 192],
            "official_final_subset_consumed": False,
        },
    )
    monkeypatch.setenv("DUCA_RIME_PHASE4_AUTHORIZATION", str(authorization))
    monkeypatch.setenv("DUCA_RIME_PHASE4_AUTHORIZATION_SHA256", _sha(authorization))
    monkeypatch.setenv("DUCA_RIME_TARGET_MEAN_COST", "192")

    stage = duca_rime_training._training_stage_authorization(
        git_commit=COMMIT,
        variant="RIME-full-TriDet",
        seed=5801,
    )
    assert stage["research_phase"] == 4
    assert stage["detector_backend"] == "TriDet"
    assert stage["formal_budget_panel"] == 192.0

    with pytest.raises(ValueError, match="does not cover"):
        duca_rime_training._training_stage_authorization(
            git_commit=COMMIT,
            variant="RIME-full",
            seed=3407,
        )


def test_tridet_training_is_blocked_without_phase4_authorization(monkeypatch):
    monkeypatch.delenv("DUCA_RIME_PHASE4_AUTHORIZATION", raising=False)
    monkeypatch.delenv("DUCA_RIME_PHASE4_AUTHORIZATION_SHA256", raising=False)
    with pytest.raises(ValueError, match="reserved for authorized Phase-4"):
        duca_rime_training._training_stage_authorization(
            git_commit=COMMIT,
            variant="RIME-full-TriDet",
            seed=3407,
        )


def test_phase4_terminal_receipt_is_bound_to_checkpoint_and_authorization(
    tmp_path,
    monkeypatch,
):
    seed = 5801
    protocol_192 = _json(
        tmp_path / "protocol_192.json",
        {
            "schema_version": "duca_rime_budget_protocol_v1",
            "fit_split": "train_only",
            "uses_validation_or_test_labels": False,
            "target_mean_cost": 192.0,
        },
    )
    protocol_384 = _json(
        tmp_path / "protocol_384.json",
        {
            "schema_version": "duca_rime_budget_protocol_v1",
            "fit_split": "train_only",
            "uses_validation_or_test_labels": False,
            "target_mean_cost": 384.0,
        },
    )
    formal_protocols = [
        {
            "target_mean_cost": 192.0,
            "path": str(protocol_192.resolve()),
            "sha256": _sha(protocol_192),
        },
        {
            "target_mean_cost": 384.0,
            "path": str(protocol_384.resolve()),
            "sha256": _sha(protocol_384),
        },
    ]
    phase2 = _json(
        tmp_path / "phase2.json",
        {
            "schema_version": "duca_rime_stage_receipt_v1",
            "phase": "phase2",
            "gate_pass": True,
            "phase3_training_authorized": True,
            "official_final_subset_consumed": False,
            "git_commit": COMMIT,
            "split_assignment_sha256": "b" * 64,
            "formal_budget_protocols": formal_protocols,
        },
    )
    authorization = _json(
        tmp_path / "phase4.json",
        {
            "schema_version": "duca_rime_stage_receipt_v1",
            "phase": "phase4_authorization",
            "status": "authorized",
            "gate_pass": True,
            "git_commit": COMMIT,
            "formal_seeds": [seed, 8123, 12011],
            "required_detectors": ["ActionFormer", "TriDet"],
            "required_budget_panels": [384, 192],
            "official_final_subset_consumed": False,
            "phase2_receipt": {
                "path": str(phase2.resolve()),
                "sha256": _sha(phase2),
            },
            "formal_budget_protocols": formal_protocols,
        },
    )
    checkpoint_path = _write(tmp_path / "epoch_59.pth", "checkpoint-bytes")
    audit = {
        "status": "complete",
        "git_commit": COMMIT,
        "variant": "RIME-full-TriDet",
        "seed": seed,
        "expected_successful_optimizer_updates": 6000,
        "update_audit": {"successful_optimizer_updates": 6000},
        "split_assignment_sha256": "b" * 64,
        "training_exposure_sha256": "c" * 64,
        "initialization_sha256": "d" * 64,
        "phase2_receipt_path": str(phase2.resolve()),
        "phase2_receipt_sha256": _sha(phase2),
        "budget_protocol_path": str(protocol_192.resolve()),
        "budget_protocol_sha256": _sha(protocol_192),
        "research_phase": 4,
        "phase4_authorization_path": str(authorization.resolve()),
        "phase4_authorization_sha256": _sha(authorization),
        "formal_budget_panel": 192.0,
        "detector_backend": "TriDet",
    }
    audit["audit_sha256"] = duca_rime_training.duca_p0_training.canonical_sha256(
        audit
    )
    metadata = {
        "schema_version": duca_rime_training.DUCA_P0_CHECKPOINT_METADATA_SCHEMA,
        "training_audit": audit,
    }
    metadata["metadata_sha256"] = (
        duca_rime_training.duca_p0_training.canonical_sha256(metadata)
    )
    checkpoint = {
        "experiment_metadata": metadata,
        "duca_rime_compaction": {
            "schema_version": "duca_rime_compact_checkpoint_receipt_v1",
            "git_commit": COMMIT,
            "variant": "RIME-full-TriDet",
            "seed": seed,
            "evaluation_equivalent": True,
            "optimizer_state_retained": False,
            "training_resume_supported": False,
        },
    }
    compaction_receipt = _json(
        tmp_path / "terminal_ema.pth.receipt.json",
        {
            "schema_version": "duca_rime_compact_checkpoint_receipt_v1",
            "status": "passed",
            "git_commit": COMMIT,
            "variant": "RIME-full-TriDet",
            "seed": seed,
            "compact_checkpoint_path": str(checkpoint_path.resolve()),
            "compact_checkpoint_sha256": _sha(checkpoint_path),
            "evaluation_equivalent": True,
            "training_resume_supported": False,
        },
    )
    receipt_payload = {
        "schema_version": "duca_rime_phase4_training_receipt_v1",
        "status": "passed",
        "arm": "RIME-full-TriDet",
        "seed": seed,
        "git_commit": COMMIT,
        "successful_detector_updates": 6000,
        "formal_update_audit_passed": True,
        "uses_official_final": False,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha(checkpoint_path),
        "checkpoint_epoch": 59,
        "checkpoint_state_key": "state_dict_ema",
        "checkpoint_compaction_receipt_path": str(
            compaction_receipt.resolve()
        ),
        "checkpoint_compaction_receipt_sha256": _sha(compaction_receipt),
        "research_phase": 4,
        "phase4_authorization_path": str(authorization.resolve()),
        "phase4_authorization_sha256": _sha(authorization),
        "target_mean_cost": 192.0,
        "detector_backend": "TriDet",
    }
    receipt = _json(tmp_path / "receipt.json", receipt_payload)
    monkeypatch.setenv("DUCA_RIME_TRAINING_RECEIPT", str(receipt))
    monkeypatch.setenv("DUCA_RIME_TRAINING_RECEIPT_SHA256", _sha(receipt))
    identity = duca_rime_training.validate_terminal_checkpoint_binding(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        git_commit=COMMIT,
        evaluation_arm="RIME-full-TriDet",
        seed=seed,
    )
    assert identity["research_phase"] == 4
    assert identity["phase4_authorization_sha256"] == _sha(authorization)
    assert identity["phase2_receipt_sha256"] == _sha(phase2)
    assert identity["budget_protocol_sha256"] == _sha(protocol_192)

    forged = dict(receipt_payload)
    forged["target_mean_cost"] = 384.0
    forged_receipt = _json(tmp_path / "forged.json", forged)
    monkeypatch.setenv("DUCA_RIME_TRAINING_RECEIPT", str(forged_receipt))
    monkeypatch.setenv("DUCA_RIME_TRAINING_RECEIPT_SHA256", _sha(forged_receipt))
    with pytest.raises(ValueError, match="are not bound"):
        duca_rime_training.validate_terminal_checkpoint_binding(
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            git_commit=COMMIT,
            evaluation_arm="RIME-full-TriDet",
            seed=seed,
        )

    forged_audit = dict(audit)
    forged_audit["budget_protocol_sha256"] = "f" * 64
    forged_audit.pop("audit_sha256")
    forged_audit["audit_sha256"] = (
        duca_rime_training.duca_p0_training.canonical_sha256(forged_audit)
    )
    forged_metadata = {
        "schema_version": duca_rime_training.DUCA_P0_CHECKPOINT_METADATA_SCHEMA,
        "training_audit": forged_audit,
    }
    forged_metadata["metadata_sha256"] = (
        duca_rime_training.duca_p0_training.canonical_sha256(forged_metadata)
    )
    forged_checkpoint = dict(checkpoint)
    forged_checkpoint["experiment_metadata"] = forged_metadata
    monkeypatch.setenv("DUCA_RIME_TRAINING_RECEIPT", str(receipt))
    monkeypatch.setenv("DUCA_RIME_TRAINING_RECEIPT_SHA256", _sha(receipt))
    with pytest.raises(ValueError, match="are not bound"):
        duca_rime_training.validate_terminal_checkpoint_binding(
            checkpoint_path=checkpoint_path,
            checkpoint=forged_checkpoint,
            git_commit=COMMIT,
            evaluation_arm="RIME-full-TriDet",
            seed=seed,
        )
