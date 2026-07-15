from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mmengine.config import Config

from tools.bata.spatial_zoom_s1_contract import canonical_sha256, sha256_file
from tools.bata.spatial_zoom_s1_training import (
    S1_CHECKPOINT_METADATA_SCHEMA,
    checkpoint_sidecar_path,
    validate_bound_s1_training_config,
    validate_s1_checkpoint_sidecar,
)
from tools.bata.spatial_zoom_s1_test_open import validate_test_open_certificate

S1_GATE_EVIDENCE_SCHEMA = "spatial_zoom_s1_gate_evidence_v3"
S1_TEST_EVIDENCE_SCHEMA = "spatial_zoom_s1_test_evidence_v4"
S1_TEST_OPEN_MARKER_SCHEMA = "spatial_zoom_s1_test_open_marker_v2"


def validate_s1_checkpoint_metadata_for_binding(
    metadata: Mapping[str, Any], *, binding: Mapping[str, Any], epoch: int, cfg: Config
) -> None:
    expected = {
        "schema_version": S1_CHECKPOINT_METADATA_SCHEMA,
        "resolution": int(binding["resolution"]),
        "seed": int(binding["seed"]),
        "epoch": int(epoch),
        "bound_config_sha256": canonical_sha256(cfg.to_dict()),
        "source_config_sha256": binding["source_config_sha256"],
        "code_commit": binding["code_commit"],
        "formal_precheck_verified": bool(binding["formal_precheck_verified"]),
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "manifest_sha256": binding["manifest_sha256"],
        "annotation_sha256": binding["annotation_sha256"],
        "fit_split_hash": binding["fit_split_hash"],
        "gate_split_hash": binding["gate_split_hash"],
        "official_test_opened": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"S1 checkpoint metadata {key} does not match binding")
    expected_updates = (int(epoch) + 1) * int(metadata["train_batches_per_epoch"])
    if int(metadata["successful_updates"]) != expected_updates:
        raise ValueError("S1 checkpoint does not satisfy equal successful updates")


def _validate_test_open_marker(
    marker_path: Path,
    *,
    cfg: Config,
    binding: Mapping[str, Any],
    checkpoint_sha256: str,
    certificate_sha256: str,
) -> dict[str, Any]:
    expected_path = (
        Path(binding["work_dir"]) / "gpu1_id0" / "test_open_started.json"
    ).resolve()
    if marker_path.resolve() != expected_path or not marker_path.is_file():
        raise ValueError("S1 test-open marker path is not canonical")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker_hash = marker.pop("marker_sha256", None)
    if not marker_hash or canonical_sha256(marker) != marker_hash:
        raise ValueError("S1 test-open marker self-hash mismatch")
    marker["marker_sha256"] = marker_hash
    expected = {
        "schema_version": S1_TEST_OPEN_MARKER_SCHEMA,
        "resolution": int(binding["resolution"]),
        "seed": int(binding["seed"]),
        "bound_config_sha256": canonical_sha256(cfg.to_dict()),
        "code_commit": binding["code_commit"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "checkpoint_sha256": checkpoint_sha256,
        "test_open_certificate_sha256": certificate_sha256,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(f"S1 test-open marker {key} mismatch")
    return marker


def write_s1_gate_evidence(
    *,
    result_dict: Mapping[str, Any],
    evaluated_video_ids: list[str] | tuple[str, ...],
    cfg: Config,
    epoch: int,
) -> Path:
    binding = validate_bound_s1_training_config(
        cfg, seed=int(cfg.spatial_zoom_s1_runtime_binding.seed)
    )
    epoch = int(epoch)
    if epoch not in set(map(int, binding["eligible_checkpoint_epochs"])):
        raise ValueError("S1 gate evidence epoch is outside the frozen eligible set")
    unexpected = sorted(set(map(str, result_dict)) - set(binding["gate_video_ids"]))
    if unexpected:
        raise ValueError(f"S1 gate evidence contains non-gate videos: {unexpected}")
    evaluated = sorted(set(map(str, evaluated_video_ids)))
    if evaluated != sorted(map(str, binding["gate_video_ids"])):
        raise ValueError("S1 gate dataloader did not cover the exact frozen gate split")
    checkpoint_path = Path(cfg.work_dir) / "checkpoint" / f"epoch_{epoch}.pth"
    sidecar = validate_s1_checkpoint_sidecar(checkpoint_path)
    metadata = sidecar["experiment_metadata"]
    validate_s1_checkpoint_metadata_for_binding(
        metadata, binding=binding, epoch=epoch, cfg=cfg
    )
    output_dir = Path(cfg.work_dir) / "gate_evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"epoch_{epoch}.result_detection.json"
    evidence_path = output_dir / f"epoch_{epoch}.evidence.json"
    if prediction_path.exists() or evidence_path.exists():
        raise FileExistsError("refusing to overwrite frozen S1 gate evidence")
    prediction_path.write_text(
        json.dumps({"results": dict(result_dict)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state_key = "state_dict_ema" if bool(cfg.solver.get("ema", False)) else "state_dict"
    evidence: dict[str, Any] = {
        "schema_version": S1_GATE_EVIDENCE_SCHEMA,
        "resolution": int(binding["resolution"]),
        "seed": int(binding["seed"]),
        "epoch": epoch,
        "state_key": state_key,
        "bound_config_path": str(Path(cfg.filename).resolve()),
        "bound_config_sha256": canonical_sha256(cfg.to_dict()),
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_path": binding["manifest_path"],
        "manifest_sha256": binding["manifest_sha256"],
        "gate_split_hash": binding["gate_split_hash"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_sidecar_path": str(
            checkpoint_sidecar_path(checkpoint_path).resolve()
        ),
        "checkpoint_sidecar_sha256": sha256_file(
            checkpoint_sidecar_path(checkpoint_path)
        ),
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "evaluated_video_count": len(evaluated),
        "evaluated_video_ids_sha256": canonical_sha256(evaluated),
        "successful_updates": int(metadata["successful_updates"]),
        "official_test_read": False,
        "paper_claim_allowed": False,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence_path


def validate_s1_gate_evidence(
    evidence: Mapping[str, Any],
    *,
    cfg: Config,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    checked = json.loads(json.dumps(dict(evidence)))
    expected_hash = checked.pop("evidence_sha256", None)
    if not expected_hash or canonical_sha256(checked) != expected_hash:
        raise ValueError("S1 gate evidence hash mismatch")
    checked["evidence_sha256"] = expected_hash
    if checked.get("schema_version") != S1_GATE_EVIDENCE_SCHEMA:
        raise ValueError("unsupported S1 gate evidence schema")
    expected = {
        "resolution": int(binding["resolution"]),
        "seed": int(binding["seed"]),
        "bound_config_path": str(Path(cfg.filename).resolve()),
        "bound_config_sha256": canonical_sha256(cfg.to_dict()),
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_path": binding["manifest_path"],
        "manifest_sha256": binding["manifest_sha256"],
        "gate_split_hash": binding["gate_split_hash"],
        "state_key": "state_dict_ema"
        if bool(cfg.solver.get("ema", False))
        else "state_dict",
        "official_test_read": False,
        "evaluated_video_count": len(binding["gate_video_ids"]),
        "evaluated_video_ids_sha256": canonical_sha256(
            sorted(map(str, binding["gate_video_ids"]))
        ),
    }
    for key, value in expected.items():
        if checked.get(key) != value:
            raise ValueError(f"S1 gate evidence {key} does not match binding")
    epoch = int(checked["epoch"])
    if epoch not in set(map(int, binding["eligible_checkpoint_epochs"])):
        raise ValueError("S1 gate evidence epoch is not eligible")
    checkpoint_path = Path(checked["checkpoint_path"])
    prediction_path = Path(checked["prediction_path"])
    sidecar_path = Path(checked["checkpoint_sidecar_path"])
    if sidecar_path.resolve() != checkpoint_sidecar_path(checkpoint_path).resolve():
        raise ValueError("S1 gate evidence checkpoint-sidecar path mismatch")
    for path, hash_key in (
        (checkpoint_path, "checkpoint_sha256"),
        (prediction_path, "prediction_sha256"),
        (sidecar_path, "checkpoint_sidecar_sha256"),
    ):
        if not path.is_file() or sha256_file(path) != checked.get(hash_key):
            raise ValueError(f"S1 gate evidence artifact mismatch: {path}")
    sidecar = validate_s1_checkpoint_sidecar(checkpoint_path)
    metadata = sidecar["experiment_metadata"]
    validate_s1_checkpoint_metadata_for_binding(
        metadata, binding=binding, epoch=epoch, cfg=cfg
    )
    if int(checked["successful_updates"]) != int(metadata["successful_updates"]):
        raise ValueError("S1 gate evidence successful-update mismatch")
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("S1 gate prediction is not result_detection JSON")
    unexpected = sorted(set(map(str, results)) - set(binding["gate_video_ids"]))
    if unexpected:
        raise ValueError(f"S1 gate prediction contains non-gate videos: {unexpected}")
    return checked


def write_s1_test_evidence(
    *,
    result_dict: Mapping[str, Any],
    evaluated_video_ids: list[str] | tuple[str, ...],
    cfg: Config,
    epoch: int,
) -> Path:
    test_binding = cfg.get("spatial_zoom_s1_test_binding")
    if not isinstance(test_binding, Mapping):
        raise ValueError("S1 test evidence requires a validated test-open binding")
    bound_cfg = Config.fromfile(str(test_binding["bound_config_path"]))
    binding = validate_bound_s1_training_config(
        bound_cfg, seed=int(test_binding["seed"])
    )
    certificate_path = Path(test_binding["certificate_path"])
    checkpoint_path = Path(test_binding["checkpoint_path"])
    certificate = validate_test_open_certificate(
        json.loads(certificate_path.read_text(encoding="utf-8")),
        cfg=bound_cfg,
        seed=int(binding["seed"]),
        checkpoint_path=checkpoint_path,
    )
    if int(epoch) != int(test_binding["checkpoint_epoch"]):
        raise ValueError("S1 test epoch does not match the frozen checkpoint")
    sidecar = validate_s1_checkpoint_sidecar(checkpoint_path)
    marker_path = Path(test_binding["open_marker_path"])
    marker = _validate_test_open_marker(
        marker_path,
        cfg=bound_cfg,
        binding=binding,
        checkpoint_sha256=sidecar["checkpoint_sha256"],
        certificate_sha256=certificate["certificate_sha256"],
    )
    manifest = json.loads(Path(binding["manifest_path"]).read_text(encoding="utf-8"))
    unexpected = sorted(set(map(str, result_dict)) - set(manifest["splits"]["test"]))
    if unexpected:
        raise ValueError(
            f"S1 test evidence contains videos outside sealed test: {unexpected}"
        )
    evaluated = sorted(set(map(str, evaluated_video_ids)))
    if evaluated != sorted(map(str, manifest["splits"]["test"])):
        raise ValueError("S1 test dataloader did not cover the exact sealed test split")
    output_dir = Path(cfg.work_dir) / "test_evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "result_detection.json"
    evidence_path = output_dir / "test.evidence.json"
    if prediction_path.exists() or evidence_path.exists():
        raise FileExistsError("refusing to overwrite sealed S1 test evidence")
    prediction_path.write_text(
        json.dumps({"results": dict(result_dict)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = sidecar["experiment_metadata"]
    validate_s1_checkpoint_metadata_for_binding(
        metadata, binding=binding, epoch=int(epoch), cfg=bound_cfg
    )
    evidence: dict[str, Any] = {
        "schema_version": S1_TEST_EVIDENCE_SCHEMA,
        "resolution": int(binding["resolution"]),
        "seed": int(binding["seed"]),
        "checkpoint_epoch": int(epoch),
        "state_key": test_binding["state_key"],
        "bound_config_path": str(Path(test_binding["bound_config_path"]).resolve()),
        "bound_config_sha256": canonical_sha256(bound_cfg.to_dict()),
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_path": binding["manifest_path"],
        "manifest_sha256": binding["manifest_sha256"],
        "test_split_hash": manifest["split_hashes"]["test"],
        "test_open_certificate_path": str(certificate_path.resolve()),
        "test_open_certificate_file_sha256": sha256_file(certificate_path),
        "test_open_certificate_sha256": certificate["certificate_sha256"],
        "test_open_marker_path": str(marker_path.resolve()),
        "test_open_marker_file_sha256": sha256_file(marker_path),
        "test_open_marker_sha256": marker["marker_sha256"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_sidecar_path": str(
            checkpoint_sidecar_path(checkpoint_path).resolve()
        ),
        "checkpoint_sidecar_sha256": sha256_file(
            checkpoint_sidecar_path(checkpoint_path)
        ),
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "evaluated_video_count": len(evaluated),
        "evaluated_video_ids_sha256": canonical_sha256(evaluated),
        "official_test_read": True,
        "single_test_open": True,
        "paper_claim_allowed": False,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence_path


def validate_s1_test_evidence(
    evidence: Mapping[str, Any],
    *,
    cfg: Config,
    seed: int,
) -> dict[str, Any]:
    binding = validate_bound_s1_training_config(cfg, seed=int(seed))
    checked = json.loads(json.dumps(dict(evidence)))
    expected_hash = checked.pop("evidence_sha256", None)
    if not expected_hash or canonical_sha256(checked) != expected_hash:
        raise ValueError("S1 test evidence hash mismatch")
    checked["evidence_sha256"] = expected_hash
    if checked.get("schema_version") != S1_TEST_EVIDENCE_SCHEMA:
        raise ValueError("unsupported S1 test evidence schema")
    expected = {
        "resolution": int(binding["resolution"]),
        "seed": int(seed),
        "bound_config_path": str(Path(cfg.filename).resolve()),
        "bound_config_sha256": canonical_sha256(cfg.to_dict()),
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_path": binding["manifest_path"],
        "manifest_sha256": binding["manifest_sha256"],
        "state_key": "state_dict_ema"
        if bool(cfg.solver.get("ema", False))
        else "state_dict",
        "official_test_read": True,
        "single_test_open": True,
    }
    for key, value in expected.items():
        if checked.get(key) != value:
            raise ValueError(f"S1 test evidence {key} mismatch")
    certificate_path = Path(checked["test_open_certificate_path"])
    checkpoint_path = Path(checked["checkpoint_path"])
    prediction_path = Path(checked["prediction_path"])
    sidecar_path = Path(checked["checkpoint_sidecar_path"])
    marker_path = Path(checked["test_open_marker_path"])
    if sidecar_path.resolve() != checkpoint_sidecar_path(checkpoint_path).resolve():
        raise ValueError("S1 test evidence checkpoint-sidecar path mismatch")
    for path, key in (
        (certificate_path, "test_open_certificate_file_sha256"),
        (checkpoint_path, "checkpoint_sha256"),
        (prediction_path, "prediction_sha256"),
        (sidecar_path, "checkpoint_sidecar_sha256"),
        (marker_path, "test_open_marker_file_sha256"),
    ):
        if not path.is_file() or sha256_file(path) != checked[key]:
            raise ValueError(f"S1 test evidence artifact mismatch: {path}")
    certificate = validate_test_open_certificate(
        json.loads(certificate_path.read_text(encoding="utf-8")),
        cfg=cfg,
        seed=int(seed),
        checkpoint_path=checkpoint_path,
    )
    if certificate["certificate_sha256"] != checked["test_open_certificate_sha256"]:
        raise ValueError("S1 test-open certificate identity mismatch")
    sidecar = validate_s1_checkpoint_sidecar(checkpoint_path)
    marker = _validate_test_open_marker(
        marker_path,
        cfg=cfg,
        binding=binding,
        checkpoint_sha256=sidecar["checkpoint_sha256"],
        certificate_sha256=certificate["certificate_sha256"],
    )
    if marker["marker_sha256"] != checked.get("test_open_marker_sha256"):
        raise ValueError("S1 test-open marker identity mismatch")
    validate_s1_checkpoint_metadata_for_binding(
        sidecar["experiment_metadata"],
        binding=binding,
        epoch=int(checked["checkpoint_epoch"]),
        cfg=cfg,
    )
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("S1 test prediction is not result_detection JSON")
    manifest = json.loads(Path(binding["manifest_path"]).read_text(encoding="utf-8"))
    expected_test_ids = sorted(map(str, manifest["splits"]["test"]))
    if checked.get("test_split_hash") != manifest["split_hashes"]["test"]:
        raise ValueError("S1 test evidence split hash mismatch")
    if int(checked.get("evaluated_video_count", -1)) != len(expected_test_ids):
        raise ValueError("S1 test evidence evaluated-video count mismatch")
    if checked.get("evaluated_video_ids_sha256") != canonical_sha256(expected_test_ids):
        raise ValueError("S1 test evidence evaluated-video identity mismatch")
    unexpected = sorted(set(map(str, results)) - set(manifest["splits"]["test"]))
    if unexpected:
        raise ValueError(f"S1 test prediction contains unexpected videos: {unexpected}")
    return checked


__all__ = [
    "S1_GATE_EVIDENCE_SCHEMA",
    "S1_TEST_EVIDENCE_SCHEMA",
    "validate_s1_checkpoint_metadata_for_binding",
    "validate_s1_gate_evidence",
    "validate_s1_test_evidence",
    "write_s1_gate_evidence",
    "write_s1_test_evidence",
]
