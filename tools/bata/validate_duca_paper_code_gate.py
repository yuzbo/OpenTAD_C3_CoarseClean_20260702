from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA = "duca_paper_clean_linux_code_gate_v2"


class CodeGateArtifactFailure(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CodeGateArtifactFailure(message)


def _path(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("ascii")
    ).hexdigest()


def validate_code_gate_artifact(
    path: str | Path,
    *,
    expected_commit: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    artifact = _path(path)
    _require(artifact.is_file(), "clean Linux code-gate receipt is missing")
    artifact_sha = _sha256(artifact)
    if expected_sha256 is not None:
        _require(
            artifact_sha == str(expected_sha256),
            "clean Linux code-gate receipt SHA-256 mismatch",
        )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "clean Linux code-gate receipt is not an object")
    unsigned = dict(payload)
    content_sha = unsigned.pop("content_sha256", None)
    _require(content_sha == _canonical_sha256(unsigned), "clean Linux code-gate self-hash drift")
    _require(payload.get("schema_version") == SCHEMA, "clean Linux code-gate schema drift")
    _require(payload.get("status") == "passed", "clean Linux code gate did not pass")
    _require(payload.get("git_commit") == expected_commit, "clean Linux code gate is stale")
    _require(
        re.fullmatch(r"[0-9a-f]{40}", str(expected_commit)) is not None,
        "an exact clean Linux code-gate commit is required",
    )
    _require(str(payload.get("slurm_job_id", "")).isdigit(), "code gate lacks a Slurm job id")
    log_path = _path(payload.get("pytest_log_path", ""))
    _require(log_path.is_file(), "code-gate pytest log is missing")
    _require(payload.get("pytest_log_sha256") == _sha256(log_path), "code-gate pytest log drift")
    _require(int(payload.get("official_train_video_count", -1)) == 200, "code gate is not full200")
    _require(int(payload.get("official_evaluation_video_count", -1)) == 211, "code gate is not exact211")
    _require(int(payload.get("stage_a_logical_cell_count", -1)) == 12, "code gate matrix size drift")
    _require(payload.get("short_window_gate_pending") is True, "code gate overclaims short-window completion")
    _require(payload.get("stage_a_manifest_created") is False, "code gate created a premature manifest")
    _require(payload.get("stage_a_released") is False, "code gate prematurely released Stage A")
    _require(payload.get("stage_b_enabled") is False, "code gate opened Stage B")
    _require(payload.get("paper_metric_claim_allowed") is False, "code gate overclaims metric evidence")
    _require(re.fullmatch(r"[0-9a-f]{64}", str(content_sha)) is not None, "invalid code-gate content hash")
    return {
        "path": str(artifact),
        "sha256": artifact_sha,
        "git_commit": expected_commit,
        "slurm_job_id": str(payload["slurm_job_id"]),
        "status": "passed",
        "claim_scope": "engineering_clean_linux_pytorch_code_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    validate_code_gate_artifact(
        args.receipt,
        expected_commit=args.expected_commit,
        expected_sha256=args.expected_sha256,
    )
    print("ENGINEERING_STATUS DUCA paper clean Linux code-gate receipt validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
