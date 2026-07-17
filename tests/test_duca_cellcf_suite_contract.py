import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata.validate_duca_cellcf_fixed384 import VARIANTS
import tools.bata.validate_duca_cellcf_suite as suite_module
from tools.bata.validate_duca_cellcf_real_loader_gate import (
    GateArtifactFailure,
    validate_real_loader_gate_artifact,
)
from tools.bata.validate_duca_cellcf_suite import (
    VARIANT_ORDER,
    _canonical_sha256,
    _shared_protocol,
    validate_suite,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifacts(tmp_path: Path, monkeypatch):
    annotation = tmp_path / "thumos_14_anno.json"
    class_map = tmp_path / "category_idx.txt"
    annotation.write_text("{}\n", encoding="utf-8")
    class_map.write_text("1 Action\n", encoding="utf-8")
    monkeypatch.setenv("THUMOS14_ANNOTATION_PATH", str(annotation))
    monkeypatch.setenv("THUMOS14_CLASS_MAP", str(class_map))
    monkeypatch.setenv("THUMOS14_TRAIN_DATA_PATH", str(tmp_path / "train"))
    monkeypatch.setenv("THUMOS14_TEST_DATA_PATH", str(tmp_path / "test"))
    monkeypatch.delenv("C3_OFFICIAL_ACTION_SEG_REPOS", raising=False)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    training_profile = os.environ.get("DUCA_CELLCF_TRAINING_PROFILE", "exposure132")
    cfg = Config.fromfile(str(ROOT / VARIANTS["uniform"]))
    protocol_sha = _canonical_sha256(_shared_protocol(cfg))
    order_sha = _canonical_sha256(list(VARIANT_ORDER))
    gate = tmp_path / "gate.json"
    _write(
        gate,
        {
            "schema": "duca_cellcf_real_loader_cuda_gate_v1",
            "ok": True,
            "git_commit": commit,
            "synthetic_gate_sha256": "a" * 64,
            "config_contract": {"training_profile": training_profile},
            "evaluation_annotation_sha256": _sha(annotation),
            "evaluation_class_map_sha256": _sha(class_map),
            "dataset": {
                "annotation_sha256": _sha(annotation),
                "class_map_sha256": _sha(class_map),
            },
        },
    )
    pilot = tmp_path / "pilot.json"
    _write(
        pilot,
        {
            "schema": "duca_cellcf_ddp_pilot_suite_v1",
            "ok": True,
            "git_commit": commit,
            "real_loader_gate_sha256": _sha(gate),
            "training_profile": training_profile,
            "variant_order": list(VARIANT_ORDER),
            "shared_protocol_sha256": protocol_sha,
            "ordered_exposure_sha256": order_sha,
        },
    )

    def validate_gate(path, *, expected_commit, **kwargs):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("git_commit") != expected_commit:
            raise ValueError("CellCF real-loader CUDA gate is stale")
        return {
            "sha256": _sha(Path(path)),
            "synthetic_gate_sha256": payload["synthetic_gate_sha256"],
            "training_profile": payload["config_contract"]["training_profile"],
        }

    def validate_pilot(path, *, expected_commit, expected_real_loader_gate_sha256, **kwargs):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("git_commit") != expected_commit:
            raise ValueError("CellCF DDP pilot is stale")
        if payload.get("real_loader_gate_sha256") != expected_real_loader_gate_sha256:
            raise ValueError("CellCF DDP pilot gate binding is stale")
        return {"sha256": _sha(Path(path))}

    monkeypatch.setattr(suite_module, "validate_real_loader_gate_artifact", validate_gate)
    monkeypatch.setattr(suite_module, "validate_pilot_artifact", validate_pilot)
    return commit, gate, pilot


def test_cellcf_suite_binds_exact_three_arm_protocol(tmp_path, monkeypatch) -> None:
    commit, gate, pilot = _artifacts(tmp_path, monkeypatch)

    payload = validate_suite(
        repo_root=ROOT,
        seed=0,
        expected_commit=commit,
        require_clean=False,
        gate_json=gate,
        pilot_json=pilot,
    )

    assert payload["ok"] is True
    assert payload["task"] == "offline_temporal_action_detection"
    assert payload["variant_order"] == list(VARIANT_ORDER)
    assert [item["name"] for item in payload["variants"]] == list(VARIANT_ORDER)
    assert payload["variants"][0]["variant_contract"]["force_exact_uniform"] is True
    assert payload["variants"][1]["variant_contract"]["counterfactual_weight"] == 0.0
    assert payload["variants"][2]["variant_contract"]["counterfactual_weight"] > 0.0


def test_cellcf_suite_resolves_native_official60_protocol(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    monkeypatch.setenv("DUCA_OFFICIAL_ADATAD_END_EPOCH", "60")
    monkeypatch.setenv("DUCA_LOSS_SCHEDULE_STEPS_PER_EPOCH", "100")
    monkeypatch.setenv("DUCA_LOSS_SCHEDULE_TOTAL_STEPS", "6000")
    commit, gate, pilot = _artifacts(tmp_path, monkeypatch)

    payload = validate_suite(
        repo_root=ROOT,
        seed=0,
        expected_commit=commit,
        require_clean=False,
        gate_json=gate,
        pilot_json=pilot,
    )

    assert payload["training_profile"] == "official60"
    assert payload["training_protocol"]["end_epoch"] == 60
    assert payload["training_protocol"]["terminal_epoch"] == 59
    assert payload["training_protocol"][
        "expected_successful_optimizer_updates"
    ] == 6000
    assert all(
        item["validation"]["training_profile"] == "official60"
        for item in payload["variants"]
    )


def test_cellcf_suite_rejects_stale_gate_commit(tmp_path, monkeypatch) -> None:
    commit, gate, pilot = _artifacts(tmp_path, monkeypatch)
    payload = json.loads(gate.read_text(encoding="utf-8"))
    payload["git_commit"] = "0" * 40
    _write(gate, payload)

    with pytest.raises(ValueError, match="stale"):
        validate_suite(
            repo_root=ROOT,
            seed=0,
            expected_commit=commit,
            require_clean=False,
            gate_json=gate,
            pilot_json=pilot,
        )


def test_minimal_handwritten_gate_is_not_accepted(tmp_path) -> None:
    gate = tmp_path / "minimal_gate.json"
    _write(
        gate,
        {
            "schema": "duca_cellcf_real_loader_cuda_gate_v1",
            "ok": True,
            "fail_closed": True,
            "git_commit": "0" * 40,
        },
    )
    with pytest.raises(GateArtifactFailure):
        validate_real_loader_gate_artifact(
            gate,
            expected_commit="0" * 40,
            expected_sha256=_sha(gate),
        )
