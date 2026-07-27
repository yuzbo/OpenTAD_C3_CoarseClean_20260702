from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mmengine.config import Config

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
    protocol = _write(tmp_path / "protocol.json", "{}")
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
        "configs/adatad/thumos/duca_rime_uniform_fixed384_total60.py",
        "configs/adatad/thumos/duca_rime_full_total60.py",
        "configs/adatad/thumos/duca_rime_full_tridet_total60.py",
    ):
        cfg = Config.fromfile(str(root / relative))
        contract = duca_rime_training.formal_training_contract(cfg)
        assert contract["expected_successful_optimizer_updates"] == 6000
        assert contract["rime_arm"] == cfg.duca_rime_variant.arm
