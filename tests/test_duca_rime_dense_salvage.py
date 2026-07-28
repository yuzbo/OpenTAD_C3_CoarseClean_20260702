from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.bata.create_duca_rime_dense_salvage_manifest import create_manifest
from tools.bata.salvage_duca_rime_dense_checkpoint import (
    MANIFEST_SCHEMA,
    RAW_CHECKPOINT_KEYS,
    _validate_raw_checkpoint,
    load_salvage_manifest,
)


RECOVERY_COMMIT = "1" * 40
SOURCE_COMMIT = "2" * 40


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    failed_root = tmp_path / "failed"
    source_dir = (
        failed_root
        / "dense_actionformer"
        / "train"
        / "gpu1_id0"
        / "checkpoint"
    )
    source_dir.mkdir(parents=True)
    source = source_dir / "epoch_59.pth"
    source.write_bytes(b"immutable-raw-checkpoint")
    recovery_parent = tmp_path / "recovery"
    recovery_parent.mkdir()
    output_root = recovery_parent / "actionformer"
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "frozen",
        "recovery_git_commit": RECOVERY_COMMIT,
        "uses_official_final": False,
        "failed_transaction": {
            "root": str(failed_root),
            "git_commit": SOURCE_COMMIT,
            "terminal_state": "failed_closed",
        },
        "sources": {
            "ActionFormer": {
                "backend": "ActionFormer",
                "source_job_id": "1198115",
                "original_job_state": "FAILED",
                "source_checkpoint_path": str(source),
                "source_checkpoint_size": source.stat().st_size,
                "source_checkpoint_sha256": _sha(source),
                "checkpoint_epoch": 59,
                "checkpoint_state_key": "state_dict_ema",
                "seed": 3407,
                "variant": "dense_actionformer",
                "expected_state_dict_keys": 499,
                "embedded_training_provenance": False,
                "external_provenance_basis": [
                    "immutable_submission_manifest",
                    "source_training_log",
                ],
                "output_root": str(output_root),
            }
        },
    }
    manifest_path = tmp_path / "salvage_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, output_root, failed_root


def test_salvage_manifest_binds_failed_source_and_fresh_external_output(tmp_path):
    manifest_path, output_root, _failed_root = _fixture(tmp_path)
    manifest, source = load_salvage_manifest(
        manifest_path,
        expected_sha256=_sha(manifest_path),
        backend="ActionFormer",
        expected_recovery_commit=RECOVERY_COMMIT,
        output_root=output_root,
    )
    assert manifest["failed_transaction"]["terminal_state"] == "failed_closed"
    assert source["source_job_id"] == "1198115"
    assert source["source_checkpoint_sha256"] == _sha(
        Path(source["source_checkpoint_path"])
    )
    assert not output_root.exists()


def test_salvage_manifest_rejects_output_inside_failed_root(tmp_path):
    manifest_path, _output_root, failed_root = _fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsafe = failed_root / "recovered"
    payload["sources"]["ActionFormer"]["output_root"] = str(unsafe)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must not modify"):
        load_salvage_manifest(
            manifest_path,
            expected_sha256=_sha(manifest_path),
            backend="ActionFormer",
            expected_recovery_commit=RECOVERY_COMMIT,
            output_root=unsafe,
        )


def test_raw_checkpoint_audit_requires_exact_epoch59_ema_schema():
    state = {"weight": object(), "bias": object()}
    checkpoint = {key: {} for key in RAW_CHECKPOINT_KEYS}
    checkpoint["epoch"] = 59
    checkpoint["state_dict"] = dict(state)
    checkpoint["state_dict_ema"] = dict(state)
    audit = _validate_raw_checkpoint(
        checkpoint,
        expected_state_dict_keys=2,
    )
    assert audit["state_dict_key_count"] == 2
    checkpoint["unexpected"] = {}
    with pytest.raises(ValueError, match="top-level schema"):
        _validate_raw_checkpoint(
            checkpoint,
            expected_state_dict_keys=2,
        )


def test_manifest_generator_allows_fresh_future_transaction_roots(tmp_path):
    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    submission = failed_root / "submission_manifest.json"
    submission.write_text('{"status":"failed"}\n', encoding="utf-8")
    af_dir = failed_root / "dense_actionformer" / "train" / "gpu1_id0" / "checkpoint"
    td_dir = failed_root / "dense_tridet" / "train" / "gpu1_id0" / "checkpoint"
    af_dir.mkdir(parents=True)
    td_dir.mkdir(parents=True)
    af = af_dir / "epoch_59.pth"
    td = td_dir / "epoch_59.pth"
    af.write_bytes(b"af-raw")
    td.write_bytes(b"td-raw")
    future_root = tmp_path / "not-created-yet"
    manifest_path = tmp_path / "salvage_manifest.json"
    result = create_manifest(
        output=manifest_path,
        recovery_git_commit=RECOVERY_COMMIT,
        failed_root=failed_root,
        failed_git_commit=SOURCE_COMMIT,
        failed_submission_manifest=submission,
        failed_submission_manifest_sha256=_sha(submission),
        actionformer_checkpoint=str(af),
        actionformer_checkpoint_sha256=_sha(af),
        actionformer_checkpoint_size=af.stat().st_size,
        actionformer_source_job_id="1198115",
        actionformer_output_root=str(future_root / "dense_actionformer"),
        tridet_checkpoint=str(td),
        tridet_checkpoint_sha256=_sha(td),
        tridet_checkpoint_size=td.stat().st_size,
        tridet_source_job_id="1198116",
        tridet_output_root=str(future_root / "dense_tridet"),
    )
    assert result["sha256"] == _sha(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["failed_transaction"]["terminal_state"] == "failed_closed"
    assert not future_root.exists()


def test_manifest_generator_rejects_output_inside_failed_transaction(tmp_path):
    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    submission = failed_root / "submission_manifest.json"
    submission.write_text("{}\n", encoding="utf-8")
    af_dir = failed_root / "dense_actionformer" / "train" / "gpu1_id0" / "checkpoint"
    td_dir = failed_root / "dense_tridet" / "train" / "gpu1_id0" / "checkpoint"
    af_dir.mkdir(parents=True)
    td_dir.mkdir(parents=True)
    af = af_dir / "epoch_59.pth"
    td = td_dir / "epoch_59.pth"
    af.write_bytes(b"af-raw")
    td.write_bytes(b"td-raw")
    with pytest.raises(ValueError, match="outside the failed transaction"):
        create_manifest(
            output=tmp_path / "salvage_manifest.json",
            recovery_git_commit=RECOVERY_COMMIT,
            failed_root=failed_root,
            failed_git_commit=SOURCE_COMMIT,
            failed_submission_manifest=submission,
            failed_submission_manifest_sha256=_sha(submission),
            actionformer_checkpoint=str(af),
            actionformer_checkpoint_sha256=_sha(af),
            actionformer_checkpoint_size=af.stat().st_size,
            actionformer_source_job_id="1198115",
            actionformer_output_root=str(failed_root / "recovered_actionformer"),
            tridet_checkpoint=str(td),
            tridet_checkpoint_sha256=_sha(td),
            tridet_checkpoint_size=td.stat().st_size,
            tridet_source_job_id="1198116",
            tridet_output_root=str(tmp_path / "recovered_tridet"),
        )
