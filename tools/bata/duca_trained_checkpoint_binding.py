from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

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


__all__ = ["SCHEMA", "load_trained_checkpoint_binding"]
