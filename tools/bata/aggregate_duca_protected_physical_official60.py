from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from tools.bata.duca_protected_physical_training import (
    VARIANTS,
    canonical_sha256,
    sha256_file,
)
from tools.bata.finalize_duca_protected_physical_run import EVIDENCE_SCHEMA


SUITE_SCHEMA = "duca_protected_physical_official60_suite_v1"
METRIC_KEYS = (
    "average_mAP",
    "mAP@0.3",
    "mAP@0.4",
    "mAP@0.5",
    "mAP@0.6",
    "mAP@0.7",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return resolved, payload


def _finite_metrics(value: Any, variant: str) -> dict[str, float]:
    _require(isinstance(value, Mapping), f"{variant} metrics are missing")
    metrics = {}
    for key in METRIC_KEYS:
        item = value.get(key)
        _require(
            isinstance(item, (int, float)) and math.isfinite(float(item)),
            f"{variant} metric {key} is invalid",
        )
        metrics[key] = float(item)
    return metrics


def aggregate_official60(
    *,
    expected_commit: str,
    protocol_manifest_sha256: str,
    authorization_sha256: str,
    evidence_paths: list[str | Path],
) -> dict[str, Any]:
    _require(
        len(evidence_paths) == len(VARIANTS),
        "official-60 suite requires exactly four evidence files",
    )
    by_variant: dict[str, dict[str, Any]] = {}
    expected_updates = None
    checkpoint_paths: set[str] = set()
    prediction_paths: set[str] = set()
    for index, evidence_path in enumerate(evidence_paths):
        path, evidence = _load_json(
            evidence_path,
            f"official-60 evidence {index}",
        )
        _require(
            evidence.get("schema") == EVIDENCE_SCHEMA
            and evidence.get("ok") is True,
            f"official-60 evidence {path} did not pass",
        )
        _require(
            evidence.get("task") == "offline_temporal_action_detection",
            f"official-60 evidence {path} is not offline TAD",
        )
        variant = str(evidence.get("variant", ""))
        _require(variant in VARIANTS, f"unknown official-60 variant: {variant}")
        _require(variant not in by_variant, f"duplicate variant: {variant}")
        _require(
            evidence.get("git_commit") == expected_commit,
            f"{variant} commit drift",
        )
        _require(int(evidence.get("seed", -1)) == 3407, f"{variant} seed drift")
        _require(
            evidence.get("protocol_manifest_sha256")
            == protocol_manifest_sha256,
            f"{variant} P0 binding drift",
        )
        _require(
            evidence.get("authorization_sha256") == authorization_sha256,
            f"{variant} authorization drift",
        )
        _require(
            int(evidence.get("checkpoint_epoch", -1)) == 59
            and evidence.get("checkpoint_state_key") == "state_dict_ema",
            f"{variant} did not use terminal epoch-59 EMA",
        )
        _require(
            evidence.get("metric_recomputation", {}).get("performed") is True,
            f"{variant} official mAP was not recomputed",
        )
        _require(
            evidence.get("non_finite_collapse") is False,
            f"{variant} suffered non-finite collapse",
        )
        updates = int(evidence.get("successful_optimizer_updates", -1))
        _require(updates > 0, f"{variant} update count is invalid")
        if expected_updates is None:
            expected_updates = updates
        _require(
            updates == expected_updates,
            f"{variant} optimizer exposure is unmatched",
        )
        checkpoint_path = str(evidence.get("checkpoint_path", ""))
        prediction_path = str(evidence.get("prediction_path", ""))
        _require(
            checkpoint_path and checkpoint_path not in checkpoint_paths,
            f"{variant} checkpoint path is reused",
        )
        _require(
            prediction_path and prediction_path not in prediction_paths,
            f"{variant} prediction path is reused",
        )
        checkpoint_paths.add(checkpoint_path)
        prediction_paths.add(prediction_path)
        by_variant[variant] = {
            "evidence_path": str(path),
            "evidence_sha256": sha256_file(path),
            "artifact_chain_sha256": evidence.get("artifact_chain_sha256"),
            "successful_optimizer_updates": updates,
            "checkpoint_sha256": evidence.get("checkpoint_sha256"),
            "prediction_sha256": evidence.get("prediction_sha256"),
            "metrics": _finite_metrics(evidence.get("metrics"), variant),
        }
    _require(
        set(by_variant) == set(VARIANTS),
        "official-60 suite is missing a preregistered arm",
    )

    uniform = by_variant["exact_uniform"]["metrics"]["average_mAP"]
    transition = by_variant["transition_no_bridge"]["metrics"]["average_mAP"]
    protected = by_variant["protected_e2e"]["metrics"]["average_mAP"]
    rho = by_variant["protected_e2e_rho001"]["metrics"]["average_mAP"]
    comparisons = {
        "transition_minus_uniform": transition - uniform,
        "protected_minus_transition": protected - transition,
        "protected_minus_uniform": protected - uniform,
        "rho001_minus_protected": rho - protected,
        "rho001_minus_uniform": rho - uniform,
    }
    learned = {
        variant: by_variant[variant]["metrics"]["average_mAP"]
        for variant in VARIANTS
        if variant != "exact_uniform"
    }
    best_variant = max(learned, key=learned.get)
    best_map = learned[best_variant]
    payload = {
        "schema": SUITE_SCHEMA,
        "ok": True,
        "paper_claim_allowed": False,
        "task": "offline_temporal_action_detection",
        "git_commit": expected_commit,
        "seed": 3407,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "authorization_sha256": authorization_sha256,
        "terminal_checkpoint": "epoch_59.pth/state_dict_ema",
        "successful_optimizer_updates_per_arm": expected_updates,
        "arms": by_variant,
        "comparisons": comparisons,
        "decision": {
            "best_learned_variant": best_variant,
            "best_learned_average_mAP": best_map,
            "strictly_above_65": best_map > 65.0,
            "strictly_above_matched_uniform": best_map > uniform,
            "protected_bridge_improves_transition": protected > transition,
            "rho001_improves_protected": rho > protected,
            "single_seed_only": True,
        },
        "limitations": [
            "single_seed_only",
            "paper_claim_requires_replication_and_cost_evidence",
        ],
    }
    payload["suite_sha256"] = canonical_sha256(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--protocol-manifest-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--evidence", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_json).expanduser().resolve()
    if output.exists():
        print(
            json.dumps(
                {
                    "schema": SUITE_SCHEMA,
                    "ok": False,
                    "error_type": "FileExistsError",
                    "error": "refusing to overwrite official-60 suite evidence",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    try:
        payload = aggregate_official60(
            expected_commit=args.expected_commit,
            protocol_manifest_sha256=args.protocol_manifest_sha256,
            authorization_sha256=args.authorization_sha256,
            evidence_paths=args.evidence,
        )
        code = 0
    except Exception as exc:
        payload = {
            "schema": SUITE_SCHEMA,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        code = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
