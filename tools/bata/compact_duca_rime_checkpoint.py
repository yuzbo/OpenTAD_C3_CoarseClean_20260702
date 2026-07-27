from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata import duca_p0_training


SCHEMA = "duca_rime_compact_checkpoint_receipt_v1"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_metadata(metadata: Any, *, expected_commit: str) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise RuntimeError("RIME terminal checkpoint lacks experiment metadata")
    unsigned_metadata = dict(metadata)
    metadata_hash = unsigned_metadata.pop("metadata_sha256", None)
    if (
        metadata.get("schema_version")
        != duca_p0_training.DUCA_P0_CHECKPOINT_METADATA_SCHEMA
        or metadata_hash != duca_p0_training.canonical_sha256(unsigned_metadata)
    ):
        raise RuntimeError("RIME terminal checkpoint metadata hash drift")
    audit = metadata.get("training_audit")
    if not isinstance(audit, Mapping):
        raise RuntimeError("RIME terminal checkpoint lacks its training audit")
    unsigned_audit = dict(audit)
    audit_hash = unsigned_audit.pop("audit_sha256", None)
    if (
        audit_hash != duca_p0_training.canonical_sha256(unsigned_audit)
        or audit.get("status") != "complete"
        or audit.get("git_commit") != expected_commit
        or int(audit.get("last_completed_epoch", -1)) != 59
        or int(audit.get("expected_successful_optimizer_updates", -1)) != 6000
        or int(
            audit.get("update_audit", {}).get(
                "successful_optimizer_updates",
                -1,
            )
        )
        != 6000
    ):
        raise RuntimeError("RIME terminal checkpoint training audit is invalid")
    return audit


def compact_checkpoint(
    *,
    source: str | Path,
    output: str | Path,
    expected_commit: str,
    remove_source: bool = False,
) -> dict[str, Any]:
    import torch

    if re.fullmatch(r"[0-9a-f]{40}", str(expected_commit)) is None:
        raise ValueError("RIME checkpoint compaction requires an exact Git commit")
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if (
        not source_path.is_file()
        or source_path.is_symlink()
        or source_path.name != "epoch_59.pth"
        or output_path.parent != source_path.parent
        or output_path.name != "terminal_ema.pth"
        or output_path.exists()
    ):
        raise RuntimeError("RIME checkpoint compaction paths are unsafe or not fresh")
    checkpoint = torch.load(source_path, map_location="cpu")
    if (
        not isinstance(checkpoint, Mapping)
        or int(checkpoint.get("epoch", -1)) != 59
        or "state_dict_ema" not in checkpoint
    ):
        raise RuntimeError("RIME source is not the terminal epoch-59 EMA checkpoint")
    audit = _validate_metadata(
        checkpoint.get("experiment_metadata"),
        expected_commit=expected_commit,
    )
    source_sha = _sha256_file(source_path)
    compact = {
        "epoch": 59,
        "state_dict_ema": checkpoint["state_dict_ema"],
        "experiment_metadata": dict(checkpoint["experiment_metadata"]),
        "duca_rime_compaction": {
            "schema_version": SCHEMA,
            "source_checkpoint_path": str(source_path),
            "source_checkpoint_sha256": source_sha,
            "source_state_key": "state_dict_ema",
            "optimizer_state_retained": False,
            "training_resume_supported": False,
            "evaluation_equivalent": True,
            "git_commit": str(expected_commit),
            "variant": str(audit["variant"]),
            "seed": int(audit["seed"]),
        },
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    sidecar = output_path.with_suffix(output_path.suffix + ".receipt.json")
    if temporary.exists() or sidecar.exists():
        raise FileExistsError("RIME compact checkpoint temporary/receipt path exists")
    try:
        torch.save(compact, temporary)
        verification = torch.load(temporary, map_location="cpu")
        _validate_metadata(
            verification.get("experiment_metadata"),
            expected_commit=expected_commit,
        )
        if (
            int(verification.get("epoch", -1)) != 59
            or "state_dict_ema" not in verification
            or set(verification) != {
                "epoch",
                "state_dict_ema",
                "experiment_metadata",
                "duca_rime_compaction",
            }
        ):
            raise RuntimeError("RIME compact checkpoint verification failed")
        compact_sha = _sha256_file(temporary)
        os.replace(temporary, output_path)
        receipt = {
            "schema_version": SCHEMA,
            "status": "passed",
            "git_commit": str(expected_commit),
            "variant": str(audit["variant"]),
            "seed": int(audit["seed"]),
            "source_checkpoint_path": str(source_path),
            "source_checkpoint_sha256": source_sha,
            "compact_checkpoint_path": str(output_path),
            "compact_checkpoint_sha256": compact_sha,
            "checkpoint_epoch": 59,
            "checkpoint_state_key": "state_dict_ema",
            "successful_detector_updates": 6000,
            "optimizer_state_retained": False,
            "training_resume_supported": False,
            "evaluation_equivalent": True,
        }
        with sidecar.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if remove_source:
            source_path.unlink()
        return {
            "path": str(output_path),
            "sha256": compact_sha,
            "receipt_path": str(sidecar),
            "receipt_sha256": _sha256_file(sidecar),
            "source_removed": bool(remove_source),
        }
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compact a completed DUCA-RIME checkpoint to terminal EMA evidence."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--remove-source", action="store_true")
    args = parser.parse_args(argv)
    result = compact_checkpoint(
        source=args.source,
        output=args.output,
        expected_commit=args.expected_commit,
        remove_source=args.remove_source,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
