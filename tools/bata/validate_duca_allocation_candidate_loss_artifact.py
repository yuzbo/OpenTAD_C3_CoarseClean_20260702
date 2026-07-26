from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

from tools.bata.evaluate_duca_allocation_candidates import (
    SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    _summarize,
    load_ceiling_records,
)
from tools.bata.export_duca_allocation_ceiling_inputs import (
    canonical_sha256,
    sha256,
    write_json_exclusive,
)
from tools.bata.validate_duca_allocation_ceiling_artifact import (
    _assert_numeric_tree_close,
)


_ROW_KEYS = {
    "schema_version",
    "sample_id",
    "video_id",
    "family_key",
    "selected_positions",
    "selected_count",
    "dense_valid_len",
    "privileged",
    "deployable",
    "losses",
    "source",
    "contract",
    "record_sha256",
}
_LOSS_KEYS = {
    "cls_loss",
    "reg_loss",
    "detector_loss",
    "physical_grid_debug",
}
_CONTRACT = {
    "model_training": False,
    "checkpoint_mutation": False,
    "dense_axis_gt": True,
    "selected_axis_gt_remap": False,
    "physical_grid_actionformer": True,
    "detector_loss_is_empirical_not_combinatorial_oracle": True,
}


def validate_candidate_artifact(
    *,
    ceiling_jsonl: str | Path,
    candidate_jsonl: str | Path,
    summary_json: str | Path,
) -> dict[str, Any]:
    ceiling_path = Path(ceiling_jsonl).resolve()
    candidate_path = Path(candidate_jsonl).resolve()
    summary_path = Path(summary_json).resolve()
    ceiling = load_ceiling_records(ceiling_path)
    summary = _load_mapping(summary_path)
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError("candidate summary schema mismatch")
    if Path(str(summary.get("output_jsonl"))).resolve() != candidate_path:
        raise ValueError("candidate summary output path mismatch")
    if summary.get("output_jsonl_sha256") != sha256(candidate_path):
        raise ValueError("candidate summary output SHA-256 mismatch")
    source = summary.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("candidate summary source is required")
    if source.get("ceiling_jsonl_sha256") != sha256(ceiling_path):
        raise ValueError("candidate summary is not bound to the exact ceiling artifact")
    if source.get("git_clean") is not True:
        raise ValueError("candidate evaluation source Git tree was not clean")
    summary_contract = summary.get("contract")
    if not isinstance(summary_contract, Mapping):
        raise ValueError("candidate summary contract is required")
    required_summary_contract = {
        "model_training": False,
        "dense_axis_gt": True,
        "selected_axis_gt_remap": False,
        "physical_grid_actionformer": True,
        "mAP_evaluated": False,
        "paper_claim_allowed": False,
    }
    if set(summary_contract) != set(required_summary_contract):
        raise ValueError("strict candidate summary contract fields mismatch")
    for key, value in required_summary_contract.items():
        if summary_contract.get(key) is not value:
            raise ValueError(f"candidate summary contract mismatch: {key}")

    requested_family_keys = tuple(str(value) for value in summary.get("requested_family_keys", []))
    if not requested_family_keys or len(set(requested_family_keys)) != len(requested_family_keys):
        raise ValueError("candidate summary family keys are empty or duplicated")
    rows = _read_rows(candidate_path)
    seen_pairs: set[tuple[str, str]] = set()
    samples_by_family: dict[str, set[str]] = {
        key: set() for key in requested_family_keys
    }
    for row in rows:
        _validate_row(
            row,
            ceiling=ceiling,
            requested_family_keys=requested_family_keys,
            summary_source=source,
        )
        pair = (str(row["sample_id"]), str(row["family_key"]))
        if pair in seen_pairs:
            raise ValueError(f"duplicate candidate sample/family pair: {pair}")
        seen_pairs.add(pair)
        samples_by_family[pair[1]].add(pair[0])
    reference_samples = samples_by_family[requested_family_keys[0]]
    if not reference_samples:
        raise ValueError("candidate artifact evaluated no samples")
    for family_key, sample_ids in samples_by_family.items():
        if sample_ids != reference_samples:
            raise ValueError(f"candidate sample set mismatch for {family_key}")
    if int(summary.get("sample_count", -1)) != len(reference_samples):
        raise ValueError("candidate summary sample count mismatch")
    if int(summary.get("row_count", -1)) != len(rows):
        raise ValueError("candidate summary row count mismatch")

    expected_summary = _summarize(
        rows,
        output_path=candidate_path,
        source=source,
        requested_family_keys=requested_family_keys,
    )
    _assert_numeric_tree_close(expected_summary, summary, context="candidate_summary")
    return {
        "schema_version": "duca_allocation_candidate_loss_validation_v1",
        "validation_passed": True,
        "sample_count": len(reference_samples),
        "row_count": len(rows),
        "family_keys": list(requested_family_keys),
        "ceiling_jsonl_sha256": sha256(ceiling_path),
        "candidate_jsonl_sha256": sha256(candidate_path),
        "summary_json_sha256": sha256(summary_path),
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: candidate row must be an object")
            rows.append(row)
    if not rows:
        raise ValueError("candidate artifact has no rows")
    return rows


def _validate_row(
    row: Mapping[str, Any],
    *,
    ceiling: Mapping[str, Mapping[str, Any]],
    requested_family_keys: Sequence[str],
    summary_source: Mapping[str, Any],
) -> None:
    unknown = set(row) - _ROW_KEYS
    missing = _ROW_KEYS - set(row)
    if unknown or missing:
        raise ValueError(f"strict candidate fields mismatch: unknown={unknown}, missing={missing}")
    if row.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("candidate row schema mismatch")
    unhashed = dict(row)
    recorded_hash = unhashed.pop("record_sha256", None)
    if not isinstance(recorded_hash, str) or canonical_sha256(unhashed) != recorded_hash:
        raise ValueError("candidate row SHA-256 mismatch")
    if row.get("source") != summary_source:
        raise ValueError("candidate row source differs from summary source")
    sample_id = str(row.get("sample_id"))
    family_key = str(row.get("family_key"))
    if sample_id not in ceiling:
        raise ValueError(f"candidate sample is absent from ceiling: {sample_id}")
    if family_key not in requested_family_keys:
        raise ValueError(f"candidate family was not requested: {family_key}")
    ceiling_row = ceiling[sample_id]
    families = {
        str(family["family_key"]): family
        for family in ceiling_row.get("families", [])
    }
    if family_key not in families:
        raise ValueError(f"candidate family is absent from ceiling: {family_key}")
    family = families[family_key]
    positions = tuple(int(value) for value in row.get("selected_positions", []))
    if positions != tuple(int(value) for value in family.get("positions", [])):
        raise ValueError("candidate selected positions differ from the ceiling artifact")
    if int(row.get("selected_count", -1)) != len(positions):
        raise ValueError("candidate selected count mismatch")
    if int(row.get("dense_valid_len", -1)) != int(ceiling_row.get("valid_len", -2)):
        raise ValueError("candidate dense valid length mismatch")
    if row.get("video_id") != ceiling_row.get("video_id"):
        raise ValueError("candidate video identity mismatch")
    if row.get("privileged") is not bool(family.get("privileged")):
        raise ValueError("candidate privileged flag mismatch")
    if row.get("deployable") is not bool(family.get("deployable")):
        raise ValueError("candidate deployable flag mismatch")
    contract = row.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("candidate row contract is required")
    if set(contract) != set(_CONTRACT) | {"gt_used_for_selection"}:
        raise ValueError("strict candidate row contract fields mismatch")
    for key, value in _CONTRACT.items():
        if contract.get(key) is not value:
            raise ValueError(f"candidate row contract mismatch: {key}")
    if contract.get("gt_used_for_selection") is not bool(family.get("privileged")):
        raise ValueError("candidate GT-selection role mismatch")
    losses = row.get("losses")
    if not isinstance(losses, Mapping) or set(losses) != _LOSS_KEYS:
        raise ValueError("strict candidate loss fields mismatch")
    cls_loss = float(losses["cls_loss"])
    reg_loss = float(losses["reg_loss"])
    detector_loss = float(losses["detector_loss"])
    if any(not math.isfinite(value) for value in (cls_loss, reg_loss, detector_loss)):
        raise ValueError("candidate losses must be finite")
    if not math.isclose(detector_loss, cls_loss + reg_loss, rel_tol=1.0e-7, abs_tol=1.0e-7):
        raise ValueError("candidate detector loss does not equal cls plus reg")
    debug = losses.get("physical_grid_debug")
    if not isinstance(debug, Mapping) or debug.get("physical_grid_actionformer_enabled") is not True:
        raise ValueError("candidate row did not execute physical-grid ActionFormer")


def _load_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected an object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate frozen DUCA allocation candidate-loss evidence fail closed."
    )
    parser.add_argument("--ceiling-jsonl", required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--validation-json")
    args = parser.parse_args(argv)
    result = validate_candidate_artifact(
        ceiling_jsonl=args.ceiling_jsonl,
        candidate_jsonl=args.candidate_jsonl,
        summary_json=args.summary_json,
    )
    if args.validation_json:
        write_json_exclusive(args.validation_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
