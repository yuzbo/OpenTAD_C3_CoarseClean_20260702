from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from tools.bata.select_duca_frontend_checkpoint import (
    MANIFEST_SCHEMA,
    select_checkpoint,
    sha256_file,
)


EXPECTED_VARIANTS = {
    "a1_t005_b8": {"actionness": 1.0, "transition": 0.05, "transition_boundary": 8.0},
    "a1_t010_b16": {"actionness": 1.0, "transition": 0.10, "transition_boundary": 16.0},
    "a1_t020_b32": {"actionness": 1.0, "transition": 0.20, "transition_boundary": 32.0},
}


def aggregate(
    *,
    expected_commit: str,
    split_manifest_path: str | Path,
    split_manifest_sha256: str,
    receipt_paths: Sequence[str | Path],
    candidate_manifest_path: str | Path,
    decision_path: str | Path,
) -> dict:
    if len(expected_commit) != 40:
        raise ValueError("expected commit must be exact")
    split_path = Path(split_manifest_path).expanduser().resolve()
    if not split_path.is_file() or sha256_file(split_path) != split_manifest_sha256:
        raise RuntimeError("frontend split manifest drift")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("test_subset_consumed") is not False:
        raise RuntimeError("frontend split consumed the test subset")

    receipts = []
    contract_evidence = []
    for raw_path in receipt_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != "duca_frontend_variant_completion_v1"
            or payload.get("ok") is not True
            or payload.get("git_commit") != expected_commit
            or payload.get("split_manifest_sha256") != split_manifest_sha256
            or payload.get("test_subset_consumed") is not False
        ):
            raise RuntimeError(f"invalid frontend completion receipt: {path}")
        contract_path = Path(str(payload.get("p0_contract_path", ""))).resolve()
        if (
            not contract_path.is_file()
            or sha256_file(contract_path) != payload.get("p0_contract_sha256")
        ):
            raise RuntimeError(f"frontend P0 contract evidence drift: {path}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if (
            contract.get("schema_version") != "duca_frontend_p0_contract_v1"
            or contract.get("ok") is not True
            or contract.get("git_commit") != expected_commit
            or contract.get("detector_executed") is not False
            or contract.get("test_subset_consumed") is not False
        ):
            raise RuntimeError(f"invalid frontend P0 contract evidence: {contract_path}")
        contract_evidence.append(
            {
                "variant": payload.get("variant"),
                "path": str(contract_path),
                "sha256": payload["p0_contract_sha256"],
            }
        )
        receipts.append(payload)
    if {item["variant"] for item in receipts} != set(EXPECTED_VARIANTS):
        raise RuntimeError("frontend completion receipts do not cover the frozen weight grid")

    candidates = []
    for receipt in receipts:
        expected_weights = EXPECTED_VARIANTS[receipt["variant"]]
        if receipt.get("loss_weights") != expected_weights:
            raise RuntimeError("frontend loss-weight receipt drift")
        rows = receipt.get("candidates", [])
        if [int(row["epoch_one_based"]) for row in rows] != [5, 10, 15, 20]:
            raise RuntimeError("frontend checkpoint cadence drift")
        candidates.extend(rows)
    if len(candidates) != 12:
        raise RuntimeError("frontend grid must contain exactly twelve candidates")

    manifest_path = Path(candidate_manifest_path).expanduser().resolve()
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "git_commit": expected_commit,
        "source_subset": "training",
        "test_subset_consumed": False,
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": split_manifest_sha256,
        "weight_grid": EXPECTED_VARIANTS,
        "checkpoint_epochs_one_based": [5, 10, 15, 20],
        "p0_contract_evidence": contract_evidence,
        "candidates": candidates,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return select_checkpoint(manifest_path, decision_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--receipt", action="append", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--decision-json", required=True)
    args = parser.parse_args(argv)
    result = aggregate(
        expected_commit=args.expected_commit,
        split_manifest_path=args.split_manifest,
        split_manifest_sha256=args.split_manifest_sha256,
        receipt_paths=args.receipt,
        candidate_manifest_path=args.candidate_manifest,
        decision_path=args.decision_json,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
