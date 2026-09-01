from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from tools.bata.duca_cellcf_training import canonical_sha256, sha256_file


SCHEMA = "opentad_trained_checkpoint_binding_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_sha256(value: Any, label: str) -> str:
    normalized = str(value or "")
    _require(re.fullmatch(r"[0-9a-f]{64}", normalized) is not None, f"{label} is not a SHA256")
    return normalized


def _bound_file(
    payload: Mapping[str, Any], path_key: str, hash_key: str, label: str
) -> tuple[Path, str]:
    path = Path(str(payload.get(path_key, ""))).expanduser().resolve()
    _require(path.is_file(), f"{label} is missing: {path}")
    observed = sha256_file(path)
    _require(observed == _require_sha256(payload.get(hash_key), f"{label} hash"), f"{label} hash mismatch")
    return path, observed


def load_trained_checkpoint_binding(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_role: str,
    expected_commit: str,
    expected_config_path: str | Path,
    expected_config_sha256: str,
    expected_resolved_config_sha256: str,
    expected_checkpoint_path: str | Path,
) -> dict[str, Any]:
    evidence_path = Path(path).expanduser().resolve()
    _require(evidence_path.is_file(), f"checkpoint binding is missing: {evidence_path}")
    evidence_sha = sha256_file(evidence_path)
    _require(
        evidence_sha == _require_sha256(expected_sha256, "checkpoint binding SHA256"),
        "checkpoint binding SHA256 mismatch",
    )
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "checkpoint binding must be a JSON object")
    _require(payload.get("schema") == SCHEMA and payload.get("ok") is True, "checkpoint binding schema/status mismatch")
    artifact_sha = payload.get("artifact_sha256")
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    _require(artifact_sha == canonical_sha256(unsigned), "checkpoint binding canonical hash mismatch")
    _require(payload.get("task") == "offline_temporal_action_detection", "checkpoint binding task mismatch")
    _require(payload.get("role") == expected_role, "checkpoint binding role mismatch")
    _require(payload.get("git_commit") == expected_commit, "checkpoint binding commit mismatch")
    _require(re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None, "expected commit is invalid")

    config_path, config_sha = _bound_file(
        payload, "config_path", "config_sha256", "bound config"
    )
    _require(config_path == Path(expected_config_path).expanduser().resolve(), "checkpoint binding config path mismatch")
    _require(config_sha == expected_config_sha256, "checkpoint binding config hash mismatch")
    _require(
        payload.get("resolved_config_sha256") == expected_resolved_config_sha256,
        "checkpoint binding resolved config mismatch",
    )
    checkpoint_path, checkpoint_sha = _bound_file(
        payload, "checkpoint_path", "checkpoint_sha256", "bound checkpoint"
    )
    _require(
        checkpoint_path == Path(expected_checkpoint_path).expanduser().resolve(),
        "checkpoint binding checkpoint path mismatch",
    )
    _require(payload.get("checkpoint_state_key") == "state_dict_ema", "checkpoint binding must use EMA")
    epoch = payload.get("checkpoint_epoch")
    _require(isinstance(epoch, int) and not isinstance(epoch, bool) and epoch >= 0, "checkpoint epoch is invalid")
    training_path, training_sha = _bound_file(
        payload,
        "training_evidence_path",
        "training_evidence_sha256",
        "training evidence",
    )
    evaluation_path, evaluation_sha = _bound_file(
        payload,
        "evaluation_evidence_path",
        "evaluation_evidence_sha256",
        "evaluation evidence",
    )
    return {
        "schema": SCHEMA,
        "path": str(evidence_path),
        "sha256": evidence_sha,
        "role": expected_role,
        "git_commit": expected_commit,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "resolved_config_sha256": expected_resolved_config_sha256,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": epoch,
        "checkpoint_state_key": "state_dict_ema",
        "training_evidence_path": str(training_path),
        "training_evidence_sha256": training_sha,
        "evaluation_evidence_path": str(evaluation_path),
        "evaluation_evidence_sha256": evaluation_sha,
    }


def build_trained_checkpoint_binding(
    *,
    role: str,
    git_commit: str,
    config_path: str | Path,
    resolved_config_sha256: str,
    checkpoint_path: str | Path,
    checkpoint_epoch: int,
    checkpoint_state_key: str,
    training_evidence_path: str | Path,
    evaluation_evidence_path: str | Path,
) -> dict[str, Any]:
    _require(re.fullmatch(r"[0-9a-f]{40}", git_commit) is not None, "git commit is invalid")
    _require(role == "dense_adatad_baseline", "unsupported checkpoint binding role")
    _require_sha256(resolved_config_sha256, "resolved config hash")
    _require(checkpoint_epoch >= 0, "checkpoint epoch is invalid")
    _require(checkpoint_state_key == "state_dict_ema", "checkpoint binding must use EMA")
    files = {
        "config": Path(config_path).expanduser().resolve(),
        "checkpoint": Path(checkpoint_path).expanduser().resolve(),
        "training_evidence": Path(training_evidence_path).expanduser().resolve(),
        "evaluation_evidence": Path(evaluation_evidence_path).expanduser().resolve(),
    }
    for label, path in files.items():
        _require(path.is_file(), f"{label} is missing: {path}")
    payload = {
        "schema": SCHEMA,
        "ok": True,
        "task": "offline_temporal_action_detection",
        "role": role,
        "git_commit": git_commit,
        "config_path": str(files["config"]),
        "config_sha256": sha256_file(files["config"]),
        "resolved_config_sha256": resolved_config_sha256,
        "checkpoint_path": str(files["checkpoint"]),
        "checkpoint_sha256": sha256_file(files["checkpoint"]),
        "checkpoint_epoch": int(checkpoint_epoch),
        "checkpoint_state_key": checkpoint_state_key,
        "training_evidence_path": str(files["training_evidence"]),
        "training_evidence_sha256": sha256_file(files["training_evidence"]),
        "evaluation_evidence_path": str(files["evaluation_evidence"]),
        "evaluation_evidence_sha256": sha256_file(files["evaluation_evidence"]),
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    return payload


def write_trained_checkpoint_binding(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    _require(not output.exists(), f"refusing to overwrite checkpoint binding: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = [
    "SCHEMA",
    "build_trained_checkpoint_binding",
    "load_trained_checkpoint_binding",
    "write_trained_checkpoint_binding",
]
