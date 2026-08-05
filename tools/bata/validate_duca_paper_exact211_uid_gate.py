from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "duca_paper_exact211_physical_uid_gate_v1"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_exact211_uid_gate_artifact(
    path: str | Path,
    *,
    expected_commit: str,
    expected_sha256: str,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    expected_commit = str(expected_commit).strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("exact-211 UID gate requires an exact commit")
    if not source.is_file():
        raise FileNotFoundError(f"exact-211 UID gate is missing: {source}")
    observed_sha = _sha256(source)
    expected_sha256 = str(expected_sha256).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("exact-211 UID gate requires an exact SHA-256")
    if observed_sha != expected_sha256:
        raise RuntimeError("exact-211 UID gate SHA-256 drift")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("exact-211 UID gate receipt is not a mapping")
    unsigned = dict(payload)
    content_sha = unsigned.pop("content_sha256", None)
    enumeration = payload.get("enumeration", {})
    prerequisite = payload.get("prerequisite_numeric_gate", {})
    numeric_path = Path(str(prerequisite.get("path", ""))).expanduser().resolve()
    regression = enumeration.get("historical_regression_key", {})
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "passed"
        or payload.get("fail_closed") is not True
        or payload.get("git_commit") != expected_commit
        or not str(payload.get("slurm_job_id", "")).isdigit()
        or content_sha != _canonical_sha256(unsigned)
        or prerequisite.get("status") != "passed"
        or prerequisite.get("git_commit") != expected_commit
        or not numeric_path.is_file()
        or numeric_path.parent.parent != source.parent
        or _sha256(numeric_path) != prerequisite.get("sha256")
        or int(enumeration.get("video_count", -1)) != 211
        or int(enumeration.get("window_count", -1)) <= 211
        or int(enumeration.get("physical_window_uid_count", -1))
        != int(enumeration.get("window_count", -2))
        or int(enumeration.get("duplicate_video_start_count", -1)) != 0
        or int(enumeration.get("duplicate_physical_uid_count", -1)) != 0
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(enumeration.get("ordered_physical_windows_sha256", ""))
        )
        or regression
        != {
            "video_id": "video_test_0001431",
            "window_start_frame": 7680,
            "count": 1,
        }
        or payload.get("metadata_only") is not True
        or payload.get("video_decode_executed") is not False
        or payload.get("model_or_backbone_executed") is not False
        or payload.get("prediction_generated") is not False
        or payload.get("metric_accessed") is not False
        or payload.get("paper_metric_claim_allowed") is not False
        or payload.get("paper_method_performance_evidence") is not False
        or payload.get("stage_a_release_prerequisite_satisfied") is not True
        or payload.get("stage_b_enabled") is not False
        or payload.get("official_final_consumed") is not False
    ):
        raise RuntimeError("exact-211 physical UID gate contract drift")
    return {
        "schema_version": SCHEMA,
        "status": "passed",
        "git_commit": expected_commit,
        "path": str(source),
        "sha256": observed_sha,
        "slurm_job_id": str(payload["slurm_job_id"]),
        "video_count": 211,
        "window_count": int(enumeration["window_count"]),
        "claim_scope": "engineering_exact211_physical_identity_only",
        "performance_evidence": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            validate_exact211_uid_gate_artifact(
                args.receipt,
                expected_commit=args.expected_commit,
                expected_sha256=args.expected_sha256,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
