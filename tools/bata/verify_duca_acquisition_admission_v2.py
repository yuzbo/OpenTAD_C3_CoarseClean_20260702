from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from tools.bata.duca_acquisition_gate_schema import (
    verify_duca_acquisition_admission_artifacts,
)


def _expected_artifacts(values: Sequence[str]) -> dict[str, str]:
    output = {}
    for value in values:
        label, separator, path = str(value).partition("=")
        if not separator or not label.strip() or not path.strip():
            raise ValueError(
                "expected artifacts must use LABEL=/absolute/or/resolvable/path"
            )
        if label in output:
            raise ValueError(f"duplicate expected artifact label: {label}")
        output[label] = path
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-hash and verify an immutable runtime-produced DUCA "
            "acquisition-admission-v2 receipt."
        )
    )
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-branch")
    parser.add_argument("--expected-remote")
    parser.add_argument("--expected-artifact", action="append", default=[])
    args = parser.parse_args(argv)

    receipt = Path(args.receipt).expanduser().resolve()
    if not receipt.is_file():
        raise ValueError(f"admission receipt is missing: {receipt}")
    actual_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    if actual_sha != args.expected_sha256:
        raise ValueError("admission receipt raw SHA-256 drift")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    verified = verify_duca_acquisition_admission_artifacts(
        payload,
        expected_commit=args.expected_commit,
        repo_root=args.repo_root,
        expected_branch=args.expected_branch,
        expected_remote=args.expected_remote,
        expected_artifact_paths=_expected_artifacts(args.expected_artifact),
    )
    print(
        json.dumps(
            {
                "schema": "duca_acquisition_admission_verification_v2",
                "status": "passed",
                "git_commit": verified["identity"]["git_commit"],
                "git_tree": verified["identity"]["git_tree"],
                "branch": verified["identity"]["branch"],
                "runtime_slurm_job_id": verified["runtime"]["slurm_job_id"],
                "receipt_sha256": actual_sha,
                "uses_official_final": False,
                "phase4_submission_enabled": False,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
