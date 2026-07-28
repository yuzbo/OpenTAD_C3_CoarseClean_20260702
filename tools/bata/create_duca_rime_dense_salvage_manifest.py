from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from tools.bata.salvage_duca_rime_dense_checkpoint import (
    MANIFEST_SCHEMA,
    sha256_file,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _commit(value: str, label: str) -> str:
    _require(
        re.fullmatch(r"[0-9a-f]{40}", value) is not None,
        f"{label} must be an exact Git commit",
    )
    return value


def _sha(value: str, label: str) -> str:
    _require(
        re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{label} must be an exact SHA-256",
    )
    return value


def _source_row(
    *,
    backend: str,
    failed_root: Path,
    checkpoint_path: str,
    checkpoint_sha256: str,
    checkpoint_size: int,
    source_job_id: str,
    expected_state_dict_keys: int,
    output_root: str,
) -> dict[str, Any]:
    source = Path(checkpoint_path).expanduser().resolve()
    _require(source.is_file(), f"{backend} raw checkpoint is missing: {source}")
    _require(not source.is_symlink(), f"{backend} raw checkpoint must not be a symlink")
    _require(source.name == "epoch_59.pth", f"{backend} source must be epoch_59.pth")
    try:
        source.relative_to(failed_root)
    except ValueError as exc:
        raise ValueError(f"{backend} source is outside the failed root") from exc
    _require(
        source.stat().st_size == int(checkpoint_size),
        f"{backend} raw checkpoint size mismatch",
    )
    expected_sha = _sha(checkpoint_sha256, f"{backend} checkpoint hash")
    _require(
        sha256_file(source) == expected_sha,
        f"{backend} raw checkpoint SHA-256 mismatch",
    )
    _require(
        re.fullmatch(r"[1-9][0-9]*", str(source_job_id)) is not None,
        f"{backend} source job ID is invalid",
    )
    output = Path(output_root).expanduser().resolve()
    _require(not output.exists(), f"{backend} salvage output already exists")
    _require(
        output != failed_root and failed_root not in output.parents,
        f"{backend} salvage output must be outside the failed transaction root",
    )
    _require(
        int(expected_state_dict_keys) > 0,
        f"{backend} expected state-dict key count is invalid",
    )
    return {
        "backend": backend,
        "source_job_id": str(source_job_id),
        "original_job_state": "FAILED",
        "source_checkpoint_path": str(source),
        "source_checkpoint_size": int(checkpoint_size),
        "source_checkpoint_sha256": expected_sha,
        "checkpoint_epoch": 59,
        "checkpoint_state_key": "state_dict_ema",
        "seed": 3407,
        "variant": f"dense_{backend.lower()}",
        "expected_state_dict_keys": int(expected_state_dict_keys),
        "embedded_training_provenance": False,
        "external_provenance_basis": [
            "failed_transaction_submission_manifest",
            "source_slurm_terminal_record",
            "source_training_log",
        ],
        "output_root": str(output),
    }


def create_manifest(
    *,
    output: str | Path,
    recovery_git_commit: str,
    failed_root: str | Path,
    failed_git_commit: str,
    failed_submission_manifest: str | Path,
    failed_submission_manifest_sha256: str,
    actionformer_checkpoint: str,
    actionformer_checkpoint_sha256: str,
    actionformer_checkpoint_size: int,
    actionformer_source_job_id: str,
    actionformer_output_root: str,
    tridet_checkpoint: str,
    tridet_checkpoint_sha256: str,
    tridet_checkpoint_size: int,
    tridet_source_job_id: str,
    tridet_output_root: str,
) -> dict[str, Any]:
    target = Path(output).expanduser().resolve()
    _require(not target.exists(), "salvage manifest output must be fresh")
    _require(target.parent.is_dir(), "salvage manifest parent must already exist")
    failed = Path(failed_root).expanduser().resolve()
    _require(failed.is_dir(), "failed transaction root is missing")
    submission = Path(failed_submission_manifest).expanduser().resolve()
    _require(submission.is_file(), "failed submission manifest is missing")
    submission_sha = _sha(
        failed_submission_manifest_sha256,
        "failed submission manifest hash",
    )
    _require(
        sha256_file(submission) == submission_sha,
        "failed submission manifest SHA-256 mismatch",
    )
    payload = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "frozen",
        "recovery_git_commit": _commit(
            recovery_git_commit,
            "recovery Git commit",
        ),
        "uses_official_final": False,
        "failed_transaction": {
            "root": str(failed),
            "git_commit": _commit(
                failed_git_commit,
                "failed transaction Git commit",
            ),
            "terminal_state": "failed_closed",
            "submission_manifest_path": str(submission),
            "submission_manifest_sha256": submission_sha,
        },
        "sources": {
            "ActionFormer": _source_row(
                backend="ActionFormer",
                failed_root=failed,
                checkpoint_path=actionformer_checkpoint,
                checkpoint_sha256=actionformer_checkpoint_sha256,
                checkpoint_size=actionformer_checkpoint_size,
                source_job_id=actionformer_source_job_id,
                expected_state_dict_keys=499,
                output_root=actionformer_output_root,
            ),
            "TriDet": _source_row(
                backend="TriDet",
                failed_root=failed,
                checkpoint_path=tridet_checkpoint,
                checkpoint_sha256=tridet_checkpoint_sha256,
                checkpoint_size=tridet_checkpoint_size,
                source_job_id=tridet_source_job_id,
                expected_state_dict_keys=462,
                output_root=tridet_output_root,
            ),
        },
    }
    actionformer_output = Path(
        payload["sources"]["ActionFormer"]["output_root"]
    ).resolve()
    tridet_output = Path(payload["sources"]["TriDet"]["output_root"]).resolve()
    _require(
        actionformer_output != tridet_output,
        "ActionFormer and TriDet salvage outputs must be distinct",
    )
    temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "schema_version": MANIFEST_SCHEMA,
        "status": "frozen",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create the immutable two-backend DUCA-RIME dense salvage manifest."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--recovery-git-commit", required=True)
    parser.add_argument("--failed-root", required=True)
    parser.add_argument("--failed-git-commit", required=True)
    parser.add_argument("--failed-submission-manifest", required=True)
    parser.add_argument("--failed-submission-manifest-sha256", required=True)
    parser.add_argument("--actionformer-checkpoint", required=True)
    parser.add_argument("--actionformer-checkpoint-sha256", required=True)
    parser.add_argument("--actionformer-checkpoint-size", required=True, type=int)
    parser.add_argument("--actionformer-source-job-id", required=True)
    parser.add_argument("--actionformer-output-root", required=True)
    parser.add_argument("--tridet-checkpoint", required=True)
    parser.add_argument("--tridet-checkpoint-sha256", required=True)
    parser.add_argument("--tridet-checkpoint-size", required=True, type=int)
    parser.add_argument("--tridet-source-job-id", required=True)
    parser.add_argument("--tridet-output-root", required=True)
    args = parser.parse_args(argv)
    result = create_manifest(
        output=args.output,
        recovery_git_commit=args.recovery_git_commit,
        failed_root=args.failed_root,
        failed_git_commit=args.failed_git_commit,
        failed_submission_manifest=args.failed_submission_manifest,
        failed_submission_manifest_sha256=args.failed_submission_manifest_sha256,
        actionformer_checkpoint=args.actionformer_checkpoint,
        actionformer_checkpoint_sha256=args.actionformer_checkpoint_sha256,
        actionformer_checkpoint_size=args.actionformer_checkpoint_size,
        actionformer_source_job_id=args.actionformer_source_job_id,
        actionformer_output_root=args.actionformer_output_root,
        tridet_checkpoint=args.tridet_checkpoint,
        tridet_checkpoint_sha256=args.tridet_checkpoint_sha256,
        tridet_checkpoint_size=args.tridet_checkpoint_size,
        tridet_source_job_id=args.tridet_source_job_id,
        tridet_output_root=args.tridet_output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
