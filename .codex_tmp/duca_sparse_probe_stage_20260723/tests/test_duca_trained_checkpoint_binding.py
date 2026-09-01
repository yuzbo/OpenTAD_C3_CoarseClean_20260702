from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.bata.duca_cellcf_training import canonical_sha256
from tools.bata.duca_trained_checkpoint_binding import (
    build_trained_checkpoint_binding,
    load_trained_checkpoint_binding,
    write_trained_checkpoint_binding,
)


COMMIT = "a" * 40


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    config = tmp_path / "dense.py"
    checkpoint = tmp_path / "epoch_59.pth"
    training = tmp_path / "training.json"
    evaluation = tmp_path / "evaluation.json"
    config.write_text("model = {}\n", encoding="utf-8")
    checkpoint.write_bytes(b"dense-trained-ema")
    training.write_text('{"ok":true}\n', encoding="utf-8")
    evaluation.write_text('{"ok":true}\n', encoding="utf-8")
    payload = {
        "schema": "opentad_trained_checkpoint_binding_v1",
        "ok": True,
        "task": "offline_temporal_action_detection",
        "role": "dense_adatad_baseline",
        "git_commit": COMMIT,
        "config_path": str(config.resolve()),
        "config_sha256": _sha(config),
        "resolved_config_sha256": "b" * 64,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "checkpoint_epoch": 59,
        "checkpoint_state_key": "state_dict_ema",
        "training_evidence_path": str(training.resolve()),
        "training_evidence_sha256": _sha(training),
        "evaluation_evidence_path": str(evaluation.resolve()),
        "evaluation_evidence_sha256": _sha(evaluation),
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return binding, config, checkpoint


def test_trained_checkpoint_binding_reopens_all_required_evidence(
    tmp_path: Path,
) -> None:
    binding, config, checkpoint = _fixture(tmp_path)

    result = load_trained_checkpoint_binding(
        binding,
        _sha(binding),
        expected_role="dense_adatad_baseline",
        expected_commit=COMMIT,
        expected_config_path=config,
        expected_config_sha256=_sha(config),
        expected_resolved_config_sha256="b" * 64,
        expected_checkpoint_path=checkpoint,
    )

    assert result["checkpoint_epoch"] == 59
    assert result["checkpoint_state_key"] == "state_dict_ema"
    assert result["role"] == "dense_adatad_baseline"


def test_trained_checkpoint_binding_rejects_replaced_checkpoint(
    tmp_path: Path,
) -> None:
    binding, config, checkpoint = _fixture(tmp_path)
    checkpoint.write_bytes(b"replaced")

    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        load_trained_checkpoint_binding(
            binding,
            _sha(binding),
            expected_role="dense_adatad_baseline",
            expected_commit=COMMIT,
            expected_config_path=config,
            expected_config_sha256=_sha(config),
            expected_resolved_config_sha256="b" * 64,
            expected_checkpoint_path=checkpoint,
        )


def test_trained_checkpoint_binding_builder_seals_dense_epoch59_ema(
    tmp_path: Path,
) -> None:
    _, config, checkpoint = _fixture(tmp_path)
    training = tmp_path / "training.json"
    evaluation = tmp_path / "evaluation.json"
    payload = build_trained_checkpoint_binding(
        role="dense_adatad_baseline",
        git_commit=COMMIT,
        config_path=config,
        resolved_config_sha256="b" * 64,
        checkpoint_path=checkpoint,
        checkpoint_epoch=59,
        checkpoint_state_key="state_dict_ema",
        training_evidence_path=training,
        evaluation_evidence_path=evaluation,
    )
    output = write_trained_checkpoint_binding(tmp_path / "built.json", payload)

    loaded = load_trained_checkpoint_binding(
        output,
        _sha(output),
        expected_role="dense_adatad_baseline",
        expected_commit=COMMIT,
        expected_config_path=config,
        expected_config_sha256=_sha(config),
        expected_resolved_config_sha256="b" * 64,
        expected_checkpoint_path=checkpoint,
    )
    assert loaded["checkpoint_epoch"] == 59
