from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


SCHEMA = "duca_rime_inference_ledger_v1"
SUMMARY_SCHEMA = "duca_rime_inference_ledger_summary_v1"


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def finalize_ledger(
    *,
    shards: Sequence[str | Path],
    output_jsonl: str | Path,
    expected_arm: str,
    expected_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    if not shards:
        raise ValueError("at least one RIME inference-ledger shard is required")
    source_artifacts = []
    rows = {}
    for shard in shards:
        path = Path(shard).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        source_artifacts.append({"path": str(path), "sha256": _sha256_file(path)})
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            prefix = f"{path}:{line_number}"
            provenance = row.get("provenance")
            if (
                row.get("schema_version") != SCHEMA
                or row.get("arm") != str(expected_arm)
                or not isinstance(provenance, Mapping)
                or any(
                    bool(provenance.get(key, False))
                    for key in (
                        "uses_gt",
                        "uses_teacher",
                        "uses_prediction_cache",
                        "uses_test_batch_composition",
                        "raw_predictions_stored",
                    )
                )
            ):
                raise ValueError(f"{prefix}: contaminated or mismatched inference ledger")
            protocol_hash = row.get("budget_protocol_sha256")
            if expected_protocol_sha256 is not None and protocol_hash != str(
                expected_protocol_sha256
            ):
                raise ValueError(f"{prefix}: frozen budget protocol drift")
            requested = int(row["requested_k"])
            effective = int(row["effective_k"])
            unique = int(row["unique_k"])
            backbone = int(row["backbone_input_k"])
            padded = int(row["padded_k"])
            positions = [int(value) for value in row["selected_dense_indices"]]
            if (
                requested < effective
                or not effective == unique == backbone == padded > 0
                or positions != sorted(set(positions))
                or len(positions) != effective
            ):
                raise ValueError(f"{prefix}: exact-K/no-padding cost ledger violation")
            video = str(row.get("video_id") or "")
            start = int(row.get("window_start_frame", -1))
            if not video or start < 0:
                raise ValueError(f"{prefix}: invalid window identity")
            key = (video, start)
            if key in rows:
                raise ValueError(f"{prefix}: duplicate inference window {key}")
            rows[key] = row
    if not rows:
        raise ValueError("RIME inference ledger contains no records")
    target = Path(output_jsonl).expanduser().resolve()
    text = "".join(
        json.dumps(rows[key], sort_keys=True, separators=(",", ":")) + "\n"
        for key in sorted(rows)
    )
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite a different RIME ledger: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    requested_values = [int(row["requested_k"]) for row in rows.values()]
    effective_values = [int(row["effective_k"]) for row in rows.values()]
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "sealed",
        "arm": str(expected_arm),
        "path": str(target),
        "sha256": _sha256_file(target),
        "record_count": len(rows),
        "video_count": len({key[0] for key in rows}),
        "requested_mean_k": mean(requested_values),
        "effective_mean_k": mean(effective_values),
        "requested_k_histogram": {
            str(value): requested_values.count(value)
            for value in sorted(set(requested_values))
        },
        "no_padding_ledger": True,
        "source_shards": source_artifacts,
        "official_final_labels_used_for_decision": False,
        "claim_scope": "inference_allocation_and_cost_ledger_only",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seal RIME inference-ledger shards")
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--expected-arm", required=True)
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--summary-json")
    args = parser.parse_args(argv)
    result = finalize_ledger(
        shards=args.shard,
        output_jsonl=args.output_jsonl,
        expected_arm=args.expected_arm,
        expected_protocol_sha256=args.expected_protocol_sha256,
    )
    if args.summary_json:
        path = Path(args.summary_json).expanduser().resolve()
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different summary: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
