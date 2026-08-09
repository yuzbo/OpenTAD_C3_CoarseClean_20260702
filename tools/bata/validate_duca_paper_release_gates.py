from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.bata.validate_duca_paper_code_gate import validate_code_gate_artifact
from tools.bata.validate_duca_paper_exact211_uid_gate import (
    validate_exact211_uid_gate_artifact,
)
from tools.bata.validate_duca_paper_numeric_gate import (
    validate_numeric_gate_artifact,
)
from tools.bata.validate_duca_paper_short_window_gate import validate_gate_artifact


SCHEMA = "duca_paper_stage_a_release_gates_v2"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_release_gates_artifact(
    path: str | Path,
    *,
    expected_commit: str,
    expected_sha256: str,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    expected_commit = str(expected_commit).strip().lower()
    expected_sha256 = str(expected_sha256).strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("release gates require an exact commit")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("release gates require an exact SHA-256")
    if not source.is_file():
        raise FileNotFoundError(f"release gates receipt is missing: {source}")
    observed_sha = _sha256(source)
    if observed_sha != expected_sha256:
        raise RuntimeError("release gates SHA-256 drift")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("release gates receipt is not a mapping")
    unsigned = dict(payload)
    content_sha = unsigned.pop("content_sha256", None)
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "passed"
        or payload.get("fail_closed") is not True
        or payload.get("git_commit") != expected_commit
        or not str(payload.get("slurm_job_id", "")).isdigit()
        or content_sha != _canonical_sha256(unsigned)
        or payload.get("paper_metric_claim_allowed") is not False
        or payload.get("paper_method_performance_evidence") is not False
        or payload.get("stage_a_release_prerequisites_satisfied") is not True
        or payload.get("stage_b_enabled") is not False
        or payload.get("official_final_consumed") is not False
    ):
        raise RuntimeError("release gates aggregate contract drift")
    code = validate_code_gate_artifact(
        payload.get("code_gate_path", ""),
        expected_commit=expected_commit,
        expected_sha256=payload.get("code_gate_sha256", ""),
    )
    short = validate_gate_artifact(
        payload.get("short_window_gate_path", ""),
        expected_commit=expected_commit,
        expected_sha256=payload.get("short_window_gate_sha256", ""),
    )
    numeric = validate_numeric_gate_artifact(
        payload.get("numeric_gate_path", ""),
        expected_commit=expected_commit,
        expected_sha256=payload.get("numeric_gate_sha256", ""),
    )
    exact211 = validate_exact211_uid_gate_artifact(
        payload.get("exact211_uid_gate_path", ""),
        expected_commit=expected_commit,
        expected_sha256=payload.get("exact211_uid_gate_sha256", ""),
    )
    if not all(
        gate.get("status") == "passed" for gate in (code, short, numeric, exact211)
    ):
        raise RuntimeError("release gates child validation did not pass")
    return {
        "schema_version": SCHEMA,
        "status": "passed",
        "git_commit": expected_commit,
        "path": str(source),
        "sha256": observed_sha,
        "claim_scope": "engineering_stage_a_release_prerequisites_only",
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
            validate_release_gates_artifact(
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
