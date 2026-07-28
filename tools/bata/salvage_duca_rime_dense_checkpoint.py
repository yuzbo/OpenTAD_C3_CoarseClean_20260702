from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA = "duca_rime_dense_salvage_manifest_v1"
RECEIPT_SCHEMA = "duca_rime_dense_salvage_receipt_v1"
COMPACT_SCHEMA = "duca_rime_dense_salvaged_checkpoint_v1"
BACKENDS = {"ActionFormer", "TriDet"}
RAW_CHECKPOINT_KEYS = {
    "epoch",
    "state_dict",
    "optimizer",
    "scheduler",
    "state_dict_ema",
    "grad_scaler",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_sha256(value: Any, label: str) -> str:
    normalized = str(value or "")
    _require(
        re.fullmatch(r"[0-9a-f]{64}", normalized) is not None,
        f"{label} is not an exact SHA-256",
    )
    return normalized


def _require_commit(value: Any, label: str) -> str:
    normalized = str(value or "")
    _require(
        re.fullmatch(r"[0-9a-f]{40}", normalized) is not None,
        f"{label} is not an exact Git commit",
    )
    return normalized


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def load_salvage_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    backend: str,
    expected_recovery_commit: str,
    output_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    _require(manifest_path.is_file(), f"salvage manifest is missing: {manifest_path}")
    _require(not manifest_path.is_symlink(), "salvage manifest must not be a symlink")
    observed_sha = sha256_file(manifest_path)
    _require(
        observed_sha == _require_sha256(expected_sha256, "salvage manifest hash"),
        "salvage manifest SHA-256 mismatch",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "salvage manifest must be a JSON object")
    _require(
        payload.get("schema_version") == MANIFEST_SCHEMA,
        "salvage manifest schema mismatch",
    )
    _require(
        payload.get("status") == "frozen",
        "salvage manifest must be frozen before execution",
    )
    _require(
        payload.get("uses_official_final") is False,
        "salvage manifest must keep official-final data sealed",
    )
    recovery_commit = _require_commit(
        expected_recovery_commit,
        "expected recovery commit",
    )
    _require(
        payload.get("recovery_git_commit") == recovery_commit,
        "salvage manifest recovery commit mismatch",
    )
    failed = payload.get("failed_transaction")
    _require(
        isinstance(failed, Mapping),
        "salvage manifest lacks failed transaction identity",
    )
    _require_commit(failed.get("git_commit"), "failed transaction commit")
    _require(
        failed.get("terminal_state") == "failed_closed",
        "source transaction must remain failed closed",
    )
    failed_root = Path(str(failed.get("root", ""))).expanduser().resolve()
    _require(failed_root.is_dir(), f"failed transaction root is missing: {failed_root}")

    backend_name = str(backend)
    _require(backend_name in BACKENDS, f"unsupported dense backend: {backend_name}")
    sources = payload.get("sources")
    _require(isinstance(sources, Mapping), "salvage manifest lacks source rows")
    source = sources.get(backend_name)
    _require(
        isinstance(source, Mapping),
        f"salvage manifest lacks {backend_name} source",
    )
    source = dict(source)
    _require(
        source.get("backend") == backend_name,
        "salvage source backend mismatch",
    )
    _require(
        source.get("original_job_state") == "FAILED",
        "salvage source job must remain recorded as FAILED",
    )
    _require(
        re.fullmatch(r"[1-9][0-9]*", str(source.get("source_job_id", "")))
        is not None,
        "salvage source job ID is invalid",
    )
    _require(
        int(source.get("checkpoint_epoch", -1)) == 59,
        "salvage source must be epoch 59",
    )
    _require(
        source.get("checkpoint_state_key") == "state_dict_ema",
        "salvage source must use EMA",
    )
    _require(
        int(source.get("seed", -1)) >= 0,
        "salvage source seed is invalid",
    )
    _require(
        isinstance(source.get("variant"), str) and bool(source["variant"]),
        "salvage source variant is missing",
    )
    _require(
        int(source.get("expected_state_dict_keys", -1)) > 0,
        "salvage source state-dict key count is missing",
    )
    _require(
        source.get("embedded_training_provenance") is False,
        "raw checkpoint must not be relabeled as containing provenance",
    )
    provenance_basis = source.get("external_provenance_basis")
    _require(
        isinstance(provenance_basis, list)
        and bool(provenance_basis)
        and all(isinstance(value, str) and value for value in provenance_basis),
        "salvage source external provenance basis is invalid",
    )
    source_path = Path(
        str(source.get("source_checkpoint_path", ""))
    ).expanduser().resolve()
    _require(
        _is_within(source_path, failed_root),
        "source checkpoint is outside the failed immutable root",
    )
    _require(
        source_path.is_file()
        and not source_path.is_symlink()
        and source_path.name == "epoch_59.pth",
        "source checkpoint path is unsafe or missing",
    )
    _require(
        source_path.stat().st_size == int(source.get("source_checkpoint_size", -1)),
        "source checkpoint size mismatch",
    )
    source_sha = sha256_file(source_path)
    _require(
        source_sha
        == _require_sha256(
            source.get("source_checkpoint_sha256"),
            "source checkpoint hash",
        ),
        "source checkpoint SHA-256 mismatch",
    )

    requested_output = Path(output_root).expanduser().resolve()
    registered_output = Path(
        str(source.get("output_root", ""))
    ).expanduser().resolve()
    _require(
        requested_output == registered_output,
        "salvage output root differs from the frozen manifest",
    )
    _require(not requested_output.exists(), "salvage output root must be fresh")
    _require(
        not _is_within(requested_output, failed_root),
        "salvage output must not modify the failed transaction root",
    )
    _require(
        requested_output.parent.is_dir(),
        "salvage output parent must already exist",
    )

    source.update(
        {
            "source_checkpoint_path": str(source_path),
            "source_checkpoint_sha256": source_sha,
            "output_root": str(requested_output),
        }
    )
    payload["_manifest_path"] = str(manifest_path)
    payload["_manifest_sha256"] = observed_sha
    payload["_failed_root"] = str(failed_root)
    return payload, source


def _validate_raw_checkpoint(
    checkpoint: Any,
    *,
    expected_state_dict_keys: int,
) -> dict[str, Any]:
    _require(isinstance(checkpoint, Mapping), "raw checkpoint is not a mapping")
    _require(
        set(checkpoint) == RAW_CHECKPOINT_KEYS,
        "raw checkpoint top-level schema mismatch",
    )
    _require(int(checkpoint.get("epoch", -1)) == 59, "raw checkpoint epoch mismatch")
    state_dict = checkpoint.get("state_dict")
    state_dict_ema = checkpoint.get("state_dict_ema")
    _require(
        isinstance(state_dict, Mapping) and isinstance(state_dict_ema, Mapping),
        "raw checkpoint lacks model/EMA state mappings",
    )
    _require(
        len(state_dict) == expected_state_dict_keys
        and len(state_dict_ema) == expected_state_dict_keys,
        "raw checkpoint state-dict key count mismatch",
    )
    _require(
        set(state_dict) == set(state_dict_ema),
        "raw checkpoint model and EMA key sets differ",
    )
    _require(
        "experiment_metadata" not in checkpoint,
        "salvage path is only for checkpoints without embedded training metadata",
    )
    return {
        "top_level_keys": sorted(checkpoint),
        "state_dict_key_count": len(state_dict),
        "state_dict_ema_key_count": len(state_dict_ema),
        "state_key_set_sha256": hashlib.sha256(
            json.dumps(sorted(state_dict), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def salvage_checkpoint(
    *,
    manifest_path: str | Path,
    manifest_sha256: str,
    backend: str,
    expected_recovery_commit: str,
    output_root: str | Path,
    precheck_only: bool = False,
) -> dict[str, Any]:
    manifest, source = load_salvage_manifest(
        manifest_path,
        expected_sha256=manifest_sha256,
        backend=backend,
        expected_recovery_commit=expected_recovery_commit,
        output_root=output_root,
    )
    import torch

    source_path = Path(source["source_checkpoint_path"])
    checkpoint = torch.load(source_path, map_location="cpu")
    raw_audit = _validate_raw_checkpoint(
        checkpoint,
        expected_state_dict_keys=int(source["expected_state_dict_keys"]),
    )
    source_sha_before = sha256_file(source_path)
    common = {
        "schema_version": RECEIPT_SCHEMA,
        "backend": str(backend),
        "recovery_git_commit": str(expected_recovery_commit),
        "source_training_git_commit": str(
            manifest["failed_transaction"]["git_commit"]
        ),
        "source_transaction_root": str(manifest["_failed_root"]),
        "source_transaction_terminal_state": "failed_closed",
        "source_job_id": str(source["source_job_id"]),
        "source_job_state": "FAILED",
        "source_checkpoint_path": str(source_path),
        "source_checkpoint_size": int(source_path.stat().st_size),
        "source_checkpoint_sha256": source_sha_before,
        "source_checkpoint_epoch": 59,
        "source_checkpoint_state_key": "state_dict_ema",
        "source_seed_external_manifest": int(source["seed"]),
        "source_variant_external_manifest": str(source["variant"]),
        "embedded_training_provenance": False,
        "external_provenance_used": True,
        "external_provenance_basis": list(source["external_provenance_basis"]),
        "manifest_path": str(manifest["_manifest_path"]),
        "manifest_sha256": str(manifest["_manifest_sha256"]),
        "raw_checkpoint_audit": raw_audit,
        "source_root_mutated": False,
        "original_job_reclassified_as_success": False,
        "uses_official_final": False,
        "energy_evidence_available": False,
        "claim_scope": "engineering_dense_reference_recovery_not_method_evidence",
    }
    if precheck_only:
        return {
            **common,
            "status": "precheck_passed",
            "output_written": False,
        }

    target_root = Path(source["output_root"])
    target_root.mkdir()
    checkpoint_dir = target_root / "checkpoint"
    checkpoint_dir.mkdir()
    output_checkpoint = checkpoint_dir / "terminal_ema.pth"
    temporary_checkpoint = checkpoint_dir / (
        f".terminal_ema.pth.partial.{os.getpid()}"
    )
    try:
        compact = {
            "epoch": 59,
            "state_dict_ema": checkpoint["state_dict_ema"],
            "duca_rime_salvage": {
                "schema_version": COMPACT_SCHEMA,
                "backend": str(backend),
                "recovery_git_commit": str(expected_recovery_commit),
                "source_training_git_commit": str(
                    manifest["failed_transaction"]["git_commit"]
                ),
                "source_job_id": str(source["source_job_id"]),
                "source_checkpoint_path": str(source_path),
                "source_checkpoint_sha256": source_sha_before,
                "manifest_sha256": str(manifest["_manifest_sha256"]),
                "embedded_training_provenance": False,
                "external_provenance_used": True,
                "evaluation_equivalent_to_source_state_dict_ema": True,
                "training_resume_supported": False,
            },
        }
        torch.save(compact, temporary_checkpoint)
        verified = torch.load(temporary_checkpoint, map_location="cpu")
        _require(
            isinstance(verified, Mapping)
            and set(verified) == {"epoch", "state_dict_ema", "duca_rime_salvage"}
            and int(verified.get("epoch", -1)) == 59
            and isinstance(verified.get("state_dict_ema"), Mapping)
            and set(verified["state_dict_ema"]) == set(checkpoint["state_dict_ema"]),
            "salvaged compact checkpoint verification failed",
        )
        compact_sha = sha256_file(temporary_checkpoint)
        os.replace(temporary_checkpoint, output_checkpoint)
        source_sha_after = sha256_file(source_path)
        _require(
            source_sha_after == source_sha_before,
            "source checkpoint changed during salvage",
        )
        receipt = {
            **common,
            "status": "passed",
            "output_written": True,
            "compact_checkpoint_path": str(output_checkpoint),
            "compact_checkpoint_sha256": compact_sha,
            "compact_checkpoint_schema": COMPACT_SCHEMA,
            "compact_checkpoint_epoch": 59,
            "compact_checkpoint_state_key": "state_dict_ema",
            "optimizer_state_retained": False,
            "training_resume_supported": False,
            "evaluation_equivalent_to_source_state_dict_ema": True,
            "source_checkpoint_sha256_after": source_sha_after,
        }
        receipt_path = target_root / "salvage_receipt.json"
        _atomic_write_json(receipt_path, receipt)
        sidecar_path = output_checkpoint.with_suffix(
            output_checkpoint.suffix + ".receipt.json"
        )
        _atomic_write_json(sidecar_path, receipt)
        return {
            "status": "passed",
            "backend": str(backend),
            "checkpoint_path": str(output_checkpoint),
            "checkpoint_sha256": compact_sha,
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": sha256_file(sidecar_path),
            "source_checkpoint_unchanged": True,
        }
    finally:
        temporary_checkpoint.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recover a failed DUCA-RIME dense epoch-59 EMA into a new "
            "hash-bound transaction without modifying the source root."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    parser.add_argument("--expected-recovery-commit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--precheck-only", action="store_true")
    args = parser.parse_args(argv)
    result = salvage_checkpoint(
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        backend=args.backend,
        expected_recovery_commit=args.expected_recovery_commit,
        output_root=args.output_root,
        precheck_only=args.precheck_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
