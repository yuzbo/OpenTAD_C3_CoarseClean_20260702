"""Promote a completed BAFDR screen submission to a formal gate receipt.

The submission launcher deliberately writes ``SUBMITTED``.  This command is
run after Slurm finishes and only writes ``PASS`` when every requested arm has
the expected terminal training receipt on the target checkout.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


ARM_SLUGS = {
    "G96": "g96",
    "U16-UNIFORM-A0": "u16_uniform_a0",
    "BAFDR-K16-LATE": "late",
    "BAFDR-K16-NOKD": "nokd",
    "BAFDR-K16-FULL": "full",
}


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def finalize_screen_gate(
    receipt_path: Path,
    *,
    work_dir_root: Path,
    expected_commit: str,
    seed: int,
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "ZOOMTOKEN-BAFDR-SCREEN-RECEIPT-v001":
        raise ValueError("unexpected BAFDR screen receipt schema")
    if receipt.get("commit_sha") != expected_commit:
        raise ValueError("screen receipt commit does not match expected commit")
    if int(receipt.get("seed", -1)) != int(seed):
        raise ValueError("screen receipt seed does not match expected seed")
    arms = list(receipt.get("arms", ()))
    if not arms or any(arm not in ARM_SLUGS for arm in arms):
        raise ValueError("screen receipt contains an unsupported or empty arm list")

    checked: list[dict[str, Any]] = []
    for arm in arms:
        train_receipt = work_dir_root / f"bafdr_k16_{ARM_SLUGS[arm]}_seed{seed}" / "train_receipt.json"
        if not train_receipt.is_file():
            raise FileNotFoundError(f"missing terminal training receipt for {arm}: {train_receipt}")
        data = json.loads(train_receipt.read_text(encoding="utf-8"))
        if data.get("protocol_id") != "ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001":
            raise ValueError(f"{arm} has the wrong protocol id")
        if data.get("phase") != "training" or data.get("metric_opened") is not False:
            raise ValueError(f"{arm} is not a terminal training receipt")
        if int(data.get("seed", -1)) != int(seed) or data.get("commit_sha") != expected_commit:
            raise ValueError(f"{arm} training receipt identity mismatch")
        if int(data.get("total_successful_updates", -1)) != 6000:
            raise ValueError(f"{arm} has an incomplete update count")
        if arm == "BAFDR-K16-FULL":
            expected_teacher = receipt.get("teacher")
            teacher_identity = data.get("teacher_identity")
            if not isinstance(expected_teacher, Mapping) or not isinstance(teacher_identity, Mapping):
                raise ValueError("BAFDR-K16-FULL is missing its bound teacher identity")
            field_map = {
                "config_sha256": "teacher_config_sha256",
                "checkpoint_sha256": "teacher_checkpoint_sha256",
                "commit": "teacher_commit",
            }
            for field, identity_field in field_map.items():
                expected_value = expected_teacher.get(field)
                observed_value = teacher_identity.get(identity_field)
                if not expected_value or observed_value != expected_value:
                    raise ValueError(f"BAFDR-K16-FULL teacher {field} does not match the screen receipt")
            if teacher_identity.get("teacher_model") != "D160":
                raise ValueError("BAFDR-K16-FULL teacher identity is not D160")
        checkpoint = Path(str(data.get("checkpoint", "")))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{arm} terminal checkpoint is missing: {checkpoint}")
        checked.append({"arm": arm, "train_receipt": str(train_receipt), "checkpoint": str(checkpoint)})

    promoted = dict(receipt)
    promoted["status"] = "PASS"
    promoted["gate"] = "all_requested_screen_arms_terminal_training"
    promoted["validated_arms"] = checked
    _atomic_write(receipt_path, promoted)
    return promoted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--work-dir-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--seed", type=int, default=4407)
    args = parser.parse_args()
    expected = str(args.expected_commit).strip().lower()
    if len(expected) != 40 or any(ch not in "0123456789abcdef" for ch in expected):
        raise SystemExit("--expected-commit must be a full 40-character SHA")
    result = finalize_screen_gate(
        args.receipt.resolve(),
        work_dir_root=args.work_dir_root.resolve(),
        expected_commit=expected,
        seed=args.seed,
    )
    print(json.dumps({"status": result["status"], "receipt": str(args.receipt.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
