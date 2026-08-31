from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "validate_continuous_roi_s2_v2_2_exact_byte_recovery.py"
SPEC = importlib.util.spec_from_file_location("continuous_roi_v22_recovery", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _synthetic_manifest(root: Path) -> dict:
    return {
        "scan_sources": [
            {
                "id": "synthetic",
                "kind": "payload_tree",
                "root": str(root),
                "max_depth": 4,
                "candidate_basenames": ["epoch_59.pth", "epoch_59.pth.metadata.json"],
            }
        ]
    }


def test_frozen_manifest_is_self_hashed_and_matches_protocol() -> None:
    manifest, protocol = MODULE.load_and_validate_contract(
        MODULE.DEFAULT_MANIFEST, MODULE.DEFAULT_PROTOCOL
    )
    expected = MODULE.expected_artifacts_from_protocol(protocol)
    assert len(expected) == 18
    assert len({item["expected_sha256"] for item in expected}) == 18
    assert manifest["builder_plan"]["candidate_content_read_before_freeze"] is False
    assert manifest["builder_plan"]["scan_ordinal"] == "1/1"


def test_manifest_rejects_expected_hash_drift(tmp_path: Path) -> None:
    manifest = MODULE.load_json(MODULE.DEFAULT_MANIFEST)
    manifest["expected_artifacts"][0]["expected_sha256"] = "0" * 64
    manifest["manifest_sha256"] = MODULE._self_hash(manifest, "manifest_sha256")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="18-artifact query table"):
        MODULE.load_and_validate_contract(path, MODULE.DEFAULT_PROTOCOL)


def test_bounded_scan_hashes_only_frozen_candidate_basenames(tmp_path: Path) -> None:
    checkpoint = b"checkpoint-bytes"
    sidecar = b"sidecar-bytes"
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "epoch_59.pth").write_bytes(checkpoint)
    (tmp_path / "nested" / "epoch_59.pth.metadata.json").write_bytes(sidecar)
    (tmp_path / "nested" / "training_completion.json").write_bytes(b"not-a-candidate")
    expected = [
        {
            "artifact_id": "A-checkpoint",
            "filename": "epoch_59.pth",
            "expected_sha256": _sha(checkpoint),
        },
        {
            "artifact_id": "A-sidecar",
            "filename": "epoch_59.pth.metadata.json",
            "expected_sha256": _sha(sidecar),
        },
    ]
    candidates, matches = MODULE.scan_payload_sources(
        _synthetic_manifest(tmp_path), expected
    )
    assert len(candidates) == 2
    assert len(matches["A-checkpoint"]) == 1
    assert len(matches["A-sidecar"]) == 1


def test_bounded_scan_does_not_follow_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "outside"
    target.mkdir()
    (target / "epoch_59.pth").write_bytes(b"bytes")
    root = tmp_path / "root"
    root.mkdir()
    try:
        (root / "linked").symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    expected = [
        {
            "artifact_id": "A-checkpoint",
            "filename": "epoch_59.pth",
            "expected_sha256": _sha(b"bytes"),
        }
    ]
    candidates, matches = MODULE.scan_payload_sources(
        _synthetic_manifest(root), expected
    )
    assert candidates == []
    assert matches["A-checkpoint"] == []


def test_quarantine_is_all_or_none_and_hash_exact(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "epoch_59.pth"
    sidecar = source / "epoch_59.pth.metadata.json"
    checkpoint.write_bytes(b"checkpoint")
    sidecar.write_bytes(b"sidecar")
    expected = [
        {
            "artifact_id": "A-checkpoint",
            "original_relative_path": "a/checkpoint/epoch_59.pth",
            "expected_sha256": _sha(b"checkpoint"),
        },
        {
            "artifact_id": "A-sidecar",
            "original_relative_path": "a/checkpoint/epoch_59.pth.metadata.json",
            "expected_sha256": _sha(b"sidecar"),
        },
    ]
    matches = {
        "A-checkpoint": [{"source_path": str(checkpoint)}],
        "A-sidecar": [{"source_path": str(sidecar)}],
    }
    quarantine = tmp_path / "quarantine"
    report = MODULE.materialize_quarantine(quarantine, expected, matches)
    assert report["artifact_count"] == 2
    assert all(entry["parity"] for entry in report["entries"])
    assert (quarantine / "a" / "checkpoint" / "epoch_59.pth").read_bytes() == b"checkpoint"


def test_partial_matches_never_create_quarantine(tmp_path: Path) -> None:
    source = tmp_path / "epoch_59.pth"
    source.write_bytes(b"checkpoint")
    expected = [
        {
            "artifact_id": "A-checkpoint",
            "original_relative_path": "a/checkpoint/epoch_59.pth",
            "expected_sha256": _sha(b"checkpoint"),
        },
        {
            "artifact_id": "A-sidecar",
            "original_relative_path": "a/checkpoint/epoch_59.pth.metadata.json",
            "expected_sha256": _sha(b"sidecar"),
        },
    ]
    with pytest.raises(ValueError, match="all-or-none quarantine is missing A-sidecar"):
        MODULE.materialize_quarantine(
            tmp_path / "quarantine",
            expected,
            {"A-checkpoint": [{"source_path": str(source)}], "A-sidecar": []},
        )
    assert not (tmp_path / "quarantine").exists()


def test_publish_once_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    MODULE._publish_once(path, {"value": 1})
    with pytest.raises(FileExistsError):
        MODULE._publish_once(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}


def test_result_root_is_not_published_partially(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_publish = MODULE._publish_once
    calls = 0

    def fail_second_publish(path: Path, payload: dict) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic terminal write failure")
        real_publish(path, payload)

    monkeypatch.setattr(MODULE, "_publish_once", fail_second_publish)
    result_root = tmp_path / "formal"
    with pytest.raises(OSError, match="synthetic terminal write failure"):
        MODULE._publish_result_root(result_root, {"inventory": 1}, {"receipt": 1})
    assert not result_root.exists()
    assert not list(tmp_path.glob(".formal.*.tmp"))


def test_formal_exception_publishes_stop_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {"manifest_sha256": "a" * 64}
    monkeypatch.setattr(
        MODULE,
        "load_and_validate_contract",
        lambda manifest_path, protocol_path: (manifest, {}),
    )
    monkeypatch.setattr(
        MODULE,
        "precheck_sources",
        lambda observed_manifest: {"precheck_ready": True},
    )

    def fail_formal(observed_manifest: dict, protocol: dict):
        raise RuntimeError("synthetic scan failure")

    monkeypatch.setattr(MODULE, "run_formal", fail_formal)
    result_root = tmp_path / "formal"
    assert MODULE.main(["--result-root", str(result_root)]) == 2
    receipt = json.loads(
        (result_root / "terminal_receipt.json").read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (result_root / "recovery_inventory.json").read_text(encoding="utf-8")
    )
    assert receipt["terminal_classification"] == MODULE.STOP
    assert receipt["formal_action_incomplete"] is True
    assert inventory["formal_action_incomplete"] is True
    assert receipt["blockers"] == [
        "FORMAL_ACTION_INCOMPLETE::RuntimeError::synthetic scan failure"
    ]


def test_catalog_text_can_never_become_candidate_payload(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{\"sha\": \"abc\"}", encoding="utf-8")
    manifest = {
        "scan_sources": [
            {
                "id": "catalog",
                "kind": "provenance_catalog",
                "root": str(tmp_path),
                "exact_files": [
                    {"relative_path": catalog.name, "sha256": MODULE.sha256_file(catalog)}
                ],
                "candidate_payload_allowed": False,
            }
        ]
    }
    candidates, matches = MODULE.scan_payload_sources(
        manifest,
        [{"artifact_id": "A", "filename": "catalog.json", "expected_sha256": MODULE.sha256_file(catalog)}],
    )
    assert candidates == []
    assert matches == {"A": []}
