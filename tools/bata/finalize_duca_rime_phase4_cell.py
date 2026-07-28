from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from tools.bata.duca_rime_stage_contract import PHASE4_RESULT_SCHEMA


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must contain one JSON object: {resolved}")
    return resolved, payload


def _verify_content(payload: Mapping[str, Any], label: str) -> None:
    unsigned = dict(payload)
    embedded = unsigned.pop("content_sha256", None)
    if embedded != _canonical_sha256(unsigned):
        raise ValueError(f"{label} content hash is invalid")


def _metrics(
    path: str | Path,
    *,
    expected_variant: str,
    expected_backend: str,
    expected_budget: float,
    expected_seed: int,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    resolved, payload = _load(path)
    _verify_content(payload, "localization metrics")
    terminal_path, terminal = _load(payload["terminal_evaluation_path"])
    if (
        payload.get("schema_version") != "duca_rime_localization_metrics_v1"
        or int(payload.get("phase", -1)) != 4
        or payload.get("variant") != expected_variant
        or payload.get("detector_backend") != expected_backend
        or float(payload.get("target_mean_cost", math.nan)) != float(expected_budget)
        or int(payload.get("seed", -1)) != int(expected_seed)
        or payload.get("split_role") != "official_final_evaluation"
        or payload.get("uses_official_final") is not True
        or payload.get("official_final_used_for_training_or_selection") is not False
        or payload.get("padded_to_kmax") is not False
        or terminal.get("variant") != expected_variant
        or terminal.get("padded_to_kmax") is not False
        or _sha256_file(terminal_path) != payload["terminal_evaluation_sha256"]
    ):
        raise ValueError(f"invalid Phase-4 arm evidence: {resolved}")
    return resolved, payload, terminal


def _training_identity(
    terminal: Mapping[str, Any],
    *,
    expected_evaluation_arm: str,
    expected_source_arm: str,
    expected_backend: str,
    expected_budget: float,
    expected_seed: int,
    authorization_sha256: str,
) -> dict[str, Any]:
    identity = terminal.get("training_identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("evaluation_arm") != expected_evaluation_arm
        or identity.get("source_arm") != expected_source_arm
        or int(identity.get("research_phase", -1)) != 4
        or identity.get("detector_backend") != expected_backend
        or float(identity.get("target_mean_cost", math.nan)) != float(expected_budget)
        or identity.get("phase4_authorization_sha256") != authorization_sha256
        or int(identity.get("successful_detector_updates", -1)) != 6000
        or identity.get("official_final_subset_consumed_during_training") is not False
    ):
        raise ValueError("Phase-4 terminal training identity does not match its cell")
    receipt_path = Path(str(identity["training_receipt_path"])).resolve()
    if (
        not receipt_path.is_file()
        or _sha256_file(receipt_path) != identity["training_receipt_sha256"]
    ):
        raise ValueError("Phase-4 training receipt artifact drifted")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != "duca_rime_phase4_training_receipt_v1"
        or receipt.get("status") != "passed"
        or receipt.get("arm") != expected_source_arm
        or int(receipt.get("seed", -1)) != int(expected_seed)
        or receipt.get("detector_backend") != expected_backend
        or float(receipt.get("target_mean_cost", math.nan)) != float(expected_budget)
        or receipt.get("phase4_authorization_sha256") != authorization_sha256
        or receipt.get("uses_official_final") is not False
    ):
        raise ValueError("Phase-4 training receipt does not match its cell")
    return dict(identity)


def _macro(metrics: Mapping[str, Any], name: str) -> float:
    values = metrics.get("video_metrics", {}).get(name)
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"Phase-4 metric {name} is missing")
    output = [float(value) for value in values.values()]
    if not all(math.isfinite(value) for value in output):
        raise ValueError(f"Phase-4 metric {name} is nonfinite")
    return mean(output)


def finalize_cell(
    *,
    authorization_receipt: str | Path,
    rime_metrics: str | Path,
    fixed_metrics: str | Path,
    same_k_metrics: str | Path,
    comparisons: str | Path,
    cost_evidence: str | Path,
    rime_ledger_summary: str | Path,
    output: str | Path,
    detector_backend: str,
    target_mean_cost: float,
    seed: int,
) -> dict[str, Any]:
    authorization_path, authorization = _load(authorization_receipt)
    authorization_sha = _sha256_file(authorization_path)
    target = float(target_mean_cost)
    if (
        authorization.get("schema_version") != "duca_rime_stage_receipt_v1"
        or authorization.get("phase") != "phase4_authorization"
        or authorization.get("status") != "authorized"
        or authorization.get("gate_pass") is not True
        or authorization.get("official_final_subset_consumed") is not False
        or int(seed) not in {int(value) for value in authorization["formal_seeds"]}
        or detector_backend not in authorization["required_detectors"]
        or target not in {
            float(value) for value in authorization["required_budget_panels"]
        }
    ):
        raise ValueError("Phase-4 authorization does not cover this result cell")
    suffix = "-TriDet" if detector_backend == "TriDet" else ""
    rime_variant = f"RIME-full{suffix}"
    fixed_variant = f"U-fixed{suffix}"
    same_variant = f"U-same-K{suffix}"
    rime_path, rime, rime_terminal = _metrics(
        rime_metrics,
        expected_variant=rime_variant,
        expected_backend=detector_backend,
        expected_budget=target,
        expected_seed=seed,
    )
    fixed_path, fixed, fixed_terminal = _metrics(
        fixed_metrics,
        expected_variant=fixed_variant,
        expected_backend=detector_backend,
        expected_budget=target,
        expected_seed=seed,
    )
    same_path, same, same_terminal = _metrics(
        same_k_metrics,
        expected_variant=same_variant,
        expected_backend=detector_backend,
        expected_budget=target,
        expected_seed=seed,
    )
    common_keys = (
        "git_commit",
        "split_assignment_sha256",
        "evaluation_video_ids",
        "annotation_sha256",
        "duration_thresholds_seconds",
    )
    for key in common_keys:
        if rime.get(key) != fixed.get(key) or rime.get(key) != same.get(key):
            raise ValueError(f"Phase-4 arm evidence differs on {key}")
    if rime["git_commit"] != authorization["git_commit"]:
        raise ValueError("Phase-4 result commit differs from its authorization")
    rime_identity = _training_identity(
        rime_terminal,
        expected_evaluation_arm=rime_variant,
        expected_source_arm=rime_variant,
        expected_backend=detector_backend,
        expected_budget=target,
        expected_seed=seed,
        authorization_sha256=authorization_sha,
    )
    fixed_identity = _training_identity(
        fixed_terminal,
        expected_evaluation_arm=fixed_variant,
        expected_source_arm=fixed_variant,
        expected_backend=detector_backend,
        expected_budget=target,
        expected_seed=seed,
        authorization_sha256=authorization_sha,
    )
    same_identity = _training_identity(
        same_terminal,
        expected_evaluation_arm=same_variant,
        expected_source_arm=rime_variant,
        expected_backend=detector_backend,
        expected_budget=target,
        expected_seed=seed,
        authorization_sha256=authorization_sha,
    )
    if (
        same_identity["training_receipt_sha256"]
        != rime_identity["training_receipt_sha256"]
        or same_identity["training_exposure_sha256"]
        != rime_identity["training_exposure_sha256"]
        or same_identity["initialization_sha256"]
        != rime_identity["initialization_sha256"]
    ):
        raise ValueError("Phase-4 U-same-K is not bound to its RIME-full source")

    comparisons_path, comparison_payload = _load(comparisons)
    _verify_content(comparison_payload, "Phase-4 comparisons")
    if (
        comparison_payload.get("schema_version")
        != "duca_rime_phase4_comparisons_v1"
        or comparison_payload.get("git_commit") != authorization["git_commit"]
        or comparison_payload.get("detector_backend") != detector_backend
        or float(comparison_payload.get("target_mean_cost", math.nan)) != target
        or int(comparison_payload.get("seed", -1)) != int(seed)
        or comparison_payload.get("evaluation_video_ids")
        != rime.get("evaluation_video_ids")
        or comparison_payload.get("official_final_used_for_training_or_selection")
        is not False
    ):
        raise ValueError("Phase-4 comparisons differ from their result cell")

    cost_path, cost = _load(cost_evidence)
    _verify_content(cost, "Phase-4 cost evidence")
    if (
        cost.get("schema_version") != "duca_rime_paired_full_stack_cost_v2"
        or int(cost.get("research_phase", -1)) != 4
        or cost.get("arm") != rime_variant
        or int(cost.get("seed", -1)) != int(seed)
        or cost.get("detector_backend") != detector_backend
        or float(cost.get("target_mean_cost", math.nan)) != target
        or cost.get("real_full_stack_measurement") is not True
        or cost.get("matched_realized_cost") is not True
        or cost.get("target_budget_respected") is not True
        or cost.get("matched_control_arm")
        != ("U-same-K-TriDet" if detector_backend == "TriDet" else "U-same-K")
        or abs(
            float(cost.get("candidate_effective_mean_k", math.inf))
            - float(cost.get("matched_control_effective_mean_k", -math.inf))
        )
        > float(cost.get("matched_k_tolerance", -math.inf))
        or float(cost.get("candidate_effective_mean_k", math.inf))
        > target + float(cost.get("matched_k_tolerance", -math.inf))
        or cost.get("includes_probe_decoder_solver") is not True
        or float(cost.get("energy_joules_per_video", 0.0)) <= 0.0
    ):
        raise ValueError("Phase-4 paired cost evidence differs from its result cell")

    ledger_path, ledger = _load(rime_ledger_summary)
    ledger_data_path = Path(str(ledger.get("path", ""))).resolve()
    if (
        ledger.get("schema_version") != "duca_rime_inference_ledger_summary_v1"
        or ledger.get("status") != "sealed"
        or ledger.get("arm") != "rime_full"
        or ledger.get("no_padding_ledger") is not True
        or ledger.get("all_observed_gaps_within_cap") is not True
        or ledger.get("official_final_labels_used_for_decision") is not False
        or not ledger_data_path.is_file()
        or _sha256_file(ledger_data_path) != ledger.get("sha256")
    ):
        raise ValueError("Phase-4 RIME inference ledger is invalid")
    ledger_rows = [
        json.loads(line)
        for line in ledger_data_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    expected_allocation_mode = (
        "fixed_floor_budget_position_only"
        if target == 192.0
        else "frozen_price_dynamic_budget"
    )
    requested_histogram = {
        str(key): int(value)
        for key, value in ledger.get("requested_k_histogram", {}).items()
    }
    if (
        not ledger_rows
        or any(
            row.get("allocation_mode") != expected_allocation_mode
            for row in ledger_rows
        )
        or float(ledger.get("requested_mean_k", math.inf)) > target + 1.0
        or (
            target == 192.0
            and requested_histogram != {"192": len(ledger_rows)}
        )
        or (
            target == 384.0
            and len(requested_histogram) < 2
        )
    ):
        raise ValueError(
            "Phase-4 RIME ledger violates its dynamic/floor budget-panel semantics"
        )

    official = rime_terminal.get("metrics")
    if not isinstance(official, Mapping):
        raise ValueError("Phase-4 terminal evaluation lacks official metrics")
    result_metrics = {
        "avg_map": float(official["average_mAP"]),
        "map_0.6": float(official["mAP@0.6"]),
        "map_0.7": float(official["mAP@0.7"]),
        "short_map": _macro(rime, "short_map"),
        "medium_map": _macro(rime, "medium_map"),
        "long_map": _macro(rime, "long_map"),
        "boundary_error": _macro(rime, "boundary_error"),
        "pair_support": _macro(rime, "pair_support"),
        "max_gap_seconds": float(ledger["max_observed_gap_seconds"]),
    }
    payload = {
        "schema_version": PHASE4_RESULT_SCHEMA,
        "git_commit": authorization["git_commit"],
        "detector_backend": detector_backend,
        "target_mean_cost": target,
        "seed": int(seed),
        "method_frozen_before_final_evaluation": True,
        "development_seed_excluded": int(seed) not in {3407},
        "uses_official_final": True,
        "official_final_used_for_training_or_selection": False,
        "rime_successful_detector_updates": 6000,
        "fixed_successful_detector_updates": 6000,
        "same_k_successful_detector_updates": 0,
        "same_k_source_training_arm": "RIME-full",
        "padded_to_kmax": False,
        "budget_panel_semantics": (
            "exact_k192_learned_position_stress_panel"
            if target == 192.0
            else "content_conditioned_dynamic_budget_panel"
        ),
        "dynamic_budget_claim_allowed": target == 384.0,
        "evaluation_video_ids": list(rime["evaluation_video_ids"]),
        "metrics": result_metrics,
        "metric_semantics": {
            "avg_map_map_0.6_map_0.7": "official_full_set_evaluator",
            "short_medium_long_map": "predeclared_video_macro_duration_strata",
            "boundary_error_pair_support": "predeclared_video_macro_auxiliary",
            "max_gap_seconds": "maximum_observed_physical_interval_from_inference_ledger",
        },
        "comparisons": comparison_payload["comparisons"],
        "cost": dict(cost),
        "k_distribution": requested_histogram,
        "artifacts": {
            "authorization": {
                "path": str(authorization_path),
                "sha256": authorization_sha,
            },
            "rime_metrics": {"path": str(rime_path), "sha256": _sha256_file(rime_path)},
            "fixed_metrics": {"path": str(fixed_path), "sha256": _sha256_file(fixed_path)},
            "same_k_metrics": {"path": str(same_path), "sha256": _sha256_file(same_path)},
            "comparisons": {
                "path": str(comparisons_path),
                "sha256": _sha256_file(comparisons_path),
            },
            "cost": {"path": str(cost_path), "sha256": _sha256_file(cost_path)},
            "rime_ledger_summary": {
                "path": str(ledger_path),
                "sha256": _sha256_file(ledger_path),
            },
        },
        "training_identity_sha256": {
            "rime": _canonical_sha256(rime_identity),
            "fixed": _canonical_sha256(fixed_identity),
            "same_k_source": _canonical_sha256(same_identity),
        },
    }
    target_path = Path(output).expanduser().resolve()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target_path.exists() and target_path.read_text(encoding="utf-8") != text:
        raise FileExistsError(
            f"refusing to overwrite different Phase-4 cell result: {target_path}"
        )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(text, encoding="utf-8")
    return {"path": str(target_path), "sha256": _sha256_file(target_path), "payload": payload}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finalize one DUCA-RIME Phase-4 cell.")
    parser.add_argument("--authorization-receipt", required=True)
    parser.add_argument("--rime-metrics", required=True)
    parser.add_argument("--fixed-metrics", required=True)
    parser.add_argument("--same-k-metrics", required=True)
    parser.add_argument("--comparisons", required=True)
    parser.add_argument("--cost-evidence", required=True)
    parser.add_argument("--rime-ledger-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--detector-backend", choices=("ActionFormer", "TriDet"), required=True)
    parser.add_argument("--target-mean-cost", type=float, choices=(384.0, 192.0), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    result = finalize_cell(
        authorization_receipt=args.authorization_receipt,
        rime_metrics=args.rime_metrics,
        fixed_metrics=args.fixed_metrics,
        same_k_metrics=args.same_k_metrics,
        comparisons=args.comparisons,
        cost_evidence=args.cost_evidence,
        rime_ledger_summary=args.rime_ledger_summary,
        output=args.output,
        detector_backend=args.detector_backend,
        target_mean_cost=args.target_mean_cost,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
