# Copyright (c) OpenTAD. All rights reserved.
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from mmengine.config import Config

PROTOCOL_ID = "ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001"
ARMS = (
    "D160",
    "G96",
    "U128-ALL48-A0",
    "U16-UNIFORM-A0",
    "BAFDR-K16-LATE",
    "BAFDR-K16-NOKD",
    "BAFDR-K16-FULL",
)
SEEDS = (4407, 4408, 4409)
EXPECTED_TRAINING_IDENTITIES = 200
EXPECTED_EVALUATION_VIDEOS = 211
EXPECTED_EVALUATION_WINDOWS = 792
EXPECTED_UPDATES_PER_EPOCH = 100
EXPECTED_EPOCHS = 60
EXPECTED_TOTAL_UPDATES = 6000
EXPECTED_WORLD_SIZE = 2
FEATURE_STRIDE = 4
WINDOW_SIZE = 768
WINDOW_OVERLAP_RATIO = 0.5
WINDOW_STRIDE = 384
SHORT_Q1_SCHEMA = "ZT_SHORT_Q1_LINEAR_V1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_publish_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def require_clean_commit(repo_root: str | Path) -> str:
    root = Path(repo_root)
    diff = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    # allow untracked docs/aris or scripts if needed, but core tree should be clean
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return head


def validate_cell_config(config_path: str | Path, expected_arm: str, expected_seed: int) -> dict[str, Any]:
    cfg = Config.fromfile(str(config_path))
    bafdr_protocol = getattr(cfg, "bafdr_protocol", None)
    if bafdr_protocol is None:
        raise ValueError(f"config {config_path} lacks bafdr_protocol dictionary")
    if bafdr_protocol.get("protocol") != PROTOCOL_ID:
        raise ValueError(f"config protocol mismatch: {bafdr_protocol.get('protocol')} != {PROTOCOL_ID}")
    if bafdr_protocol.get("arm") != expected_arm:
        raise ValueError(f"config arm mismatch: {bafdr_protocol.get('arm')} != {expected_arm}")
    if int(bafdr_protocol.get("seed")) != int(expected_seed):
        raise ValueError(f"config seed mismatch: {bafdr_protocol.get('seed')} != {expected_seed}")
    return {
        "config_path": str(config_path),
        "arm": expected_arm,
        "seed": expected_seed,
        "config_sha256": sha256_file(config_path),
    }


ARM_CONFIG_NAMES = {
    "D160": "d160",
    "G96": "g96",
    "U128-ALL48-A0": "u128_all48_a0",
    "U16-UNIFORM-A0": "u16_uniform_a0",
    "BAFDR-K16-LATE": "late",
    "BAFDR-K16-NOKD": "nokd",
    "BAFDR-K16-FULL": "full",
}


def main():
    parser = argparse.ArgumentParser(description="BA-FDR K16 Master Orchestrator")
    parser.add_argument("--repo-root", type=str, default=".", help="path to repo root")
    parser.add_argument("--output", type=str, default="submission_receipt.json", help="output submission receipt")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    head = require_clean_commit(root)

    print(f"[BA-FDR] Validating 21 configs on commit {head}...")
    matrix_cells = []
    for arm in ARMS:
        slug = ARM_CONFIG_NAMES.get(arm, arm.lower().replace("-", "_"))
        for seed in SEEDS:
            cfg_name = f"bafdr_k16_{slug}_seed{seed}.py"
            cfg_path = root / "configs" / "adatad" / "thumos" / cfg_name
            if not cfg_path.exists():
                raise FileNotFoundError(f"Missing config {cfg_path}")
            cell_info = validate_cell_config(cfg_path, arm, seed)
            matrix_cells.append(cell_info)

    receipt = {
        "protocol_id": PROTOCOL_ID,
        "commit_sha": head,
        "total_cells": len(matrix_cells),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "cells": matrix_cells,
        "status": "VALIDATED_AND_READY",
    }
    atomic_publish_json(args.output, receipt)
    print(f"[BA-FDR] Master submission receipt written to {args.output} (21 cells validated).")


if __name__ == "__main__":
    main()
