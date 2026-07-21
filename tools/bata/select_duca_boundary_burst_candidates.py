from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.create_duca_frontend_split import validate_split_manifest
from tools.bata.select_duca_frontend_checkpoint import sha256_file


SCHEMA = "duca_boundary_burst_frontend_decision_v1"
VARIANT_SPECS = {
    "gaussian_matched": None,
    "burst_r2q3": "r2q3",
    "burst_r4q5": "r4q5",
}


def _verified_file(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    resolved = Path(path).expanduser().resolve()
    if (
        len(str(expected_sha256)) != 64
        or not resolved.is_file()
        or sha256_file(resolved) != str(expected_sha256)
    ):
        raise RuntimeError(f"{label} path/hash drift: {resolved}")
    return resolved


def validate_r0_runtime_bindings(
    *,
    split_manifest: str | Path,
    split_manifest_sha256: str,
    annotation_path: str | Path,
    annotation_sha256: str,
    train_block_list: str | Path,
    train_block_list_sha256: str,
    holdout_block_list: str | Path,
    holdout_block_list_sha256: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    pretrain_path: str | Path,
    pretrain_sha256: str,
) -> dict[str, Any]:
    """Reopen every submit-time R0 binding before any model work."""

    binding = validate_split_manifest(
        split_manifest,
        expected_manifest_sha256=split_manifest_sha256,
        annotation_path=annotation_path,
        train_block_list=train_block_list,
        holdout_block_list=holdout_block_list,
    )
    expected_reference_hashes = {
        "annotation_sha256": annotation_sha256,
        "train_block_list_sha256": train_block_list_sha256,
        "holdout_block_list_sha256": holdout_block_list_sha256,
    }
    for field, expected in expected_reference_hashes.items():
        if len(str(expected)) != 64 or binding.get(field) != str(expected):
            raise RuntimeError(f"submit-time split binding drift: {field}")

    checkpoint = _verified_file(
        checkpoint_path, checkpoint_sha256, label="R0 checkpoint"
    )
    pretrain = _verified_file(
        pretrain_path, pretrain_sha256, label="AdaTAD pretrain"
    )
    return {
        "ok": True,
        "schema": "duca_r0_runtime_bindings_v1",
        "split": binding,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "pretrain_path": str(pretrain),
        "pretrain_sha256": pretrain_sha256,
    }


def _average_map_from_payload(payload: Mapping[str, Any], *, label: str) -> float:
    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, Mapping):
        raise RuntimeError(f"{label} metrics payload is not a mapping")
    value = metrics.get("average_mAP")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} average_mAP is missing or non-numeric")
    return _finite(value, f"{label}.average_mAP")


def validate_r0_headroom_summary(
    *,
    summary_path: str | Path,
    summary_sha256: str,
    expected_commit: str,
) -> dict[str, Any]:
    """Derive the P0 headroom gate from sealed metric files, not summary copies."""

    source = _verified_file(summary_path, summary_sha256, label="R0 summary")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "duca_r0_selected_axis_boundary_burst_map_v2":
        raise RuntimeError("R0 summary schema mismatch")
    if payload.get("ok") is not True or payload.get("git_commit") != expected_commit:
        raise RuntimeError("R0 summary did not complete on the exact commit")
    if payload.get("source_subset") != "training_internal_holdout":
        raise RuntimeError("R0 did not use the sealed training holdout")
    if payload.get("test_subset_consumed") is not False:
        raise RuntimeError("R0 consumed the test subset")

    rows = {row.get("family"): row for row in payload.get("rows", [])}
    required = {
        "A_exact_uniform",
        "R2Q3_privileged_boundary_burst",
        "R4Q5_privileged_boundary_burst",
    }
    if set(rows) != required:
        raise RuntimeError("R0 family set mismatch")

    values: dict[str, float] = {}
    for family, row in rows.items():
        metrics_path = _verified_file(
            row.get("metrics_path", ""),
            str(row.get("metrics_sha256", "")),
            label=f"R0 metrics for {family}",
        )
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        value = _average_map_from_payload(metrics_payload, label=f"R0 {family}")
        copied_metrics = row.get("metrics")
        canonical_metrics = metrics_payload.get("metrics", metrics_payload)
        if copied_metrics != canonical_metrics:
            raise RuntimeError(f"R0 copied metrics mismatch for {family}")
        copied_average_map = row.get("average_mAP")
        if (
            isinstance(copied_average_map, bool)
            or not isinstance(copied_average_map, (int, float))
            or float(copied_average_map) != value
        ):
            raise RuntimeError(f"R0 copied average_mAP mismatch for {family}")
        values[str(family)] = value

    uniform = values["A_exact_uniform"]
    headroom_by_family = {
        family: values[family] - uniform
        for family in (
            "R2Q3_privileged_boundary_burst",
            "R4Q5_privileged_boundary_burst",
        )
    }
    for family, headroom in headroom_by_family.items():
        copied = rows[family].get("headroom_vs_uniform_average_mAP")
        if (
            isinstance(copied, bool)
            or not isinstance(copied, (int, float))
            or float(copied) != headroom
        ):
            raise RuntimeError(f"R0 copied headroom mismatch for {family}")
    best_headroom = max(headroom_by_family.values())
    required_headroom = _finite(
        payload.get("required_headroom_average_mAP", float("nan")),
        "required_headroom_average_mAP",
    )
    if required_headroom < 0.20:
        raise RuntimeError("R0 required headroom contract is missing or too weak")
    if not best_headroom > required_headroom:
        raise RuntimeError(
            "R0 constrained burst Oracle headroom does not clear the frozen "
            f"threshold: headroom={best_headroom}, required>{required_headroom}"
        )
    return {
        "schema": "duca_r0_headroom_gate_v1",
        "ok": True,
        "git_commit": expected_commit,
        "r0_summary_path": str(source),
        "r0_summary_sha256": summary_sha256,
        "average_mAP": values,
        "best_privileged_minus_uniform_average_mAP": best_headroom,
        "required_strict_headroom_average_mAP": required_headroom,
        "test_subset_consumed": False,
        "paper_claim_allowed": False,
    }


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _mean(summary: Mapping[str, Any], *keys: str) -> float:
    value: Any = summary
    for key in keys:
        value = value[key]
    if isinstance(value, Mapping) and "mean" in value:
        value = value["mean"]
    return _finite(value, ".".join(keys))


def _effective_budget_contract_verified(
    summary: Mapping[str, Any],
    *,
    requested_budget: int = 384,
    requested_max_unselected_hole: int = 2,
) -> bool:
    """Require analyzer-derived per-sample K/G evidence, not aggregate means."""

    protocol = summary.get("protocol", {})
    evidence = protocol.get("sampling_contract_evidence", {}) if isinstance(protocol, Mapping) else {}
    return bool(
        isinstance(protocol, Mapping)
        and isinstance(evidence, Mapping)
        and protocol.get("budget_matched") is True
        and protocol.get("valid_length_matched") is True
        and protocol.get("max_hole_matched") is True
        and int(evidence.get("sample_count", 0)) == int(summary.get("sample_count", -1))
        and int(evidence.get("sample_count", 0)) > 0
        and int(evidence.get("budget_violation_count", -1)) == 0
        and int(evidence.get("max_hole_violation_count", -1)) == 0
        and int(evidence.get("requested_budget_min", -1)) == int(requested_budget)
        and int(evidence.get("requested_budget_max", -1)) == int(requested_budget)
        and int(evidence.get("requested_max_unselected_hole_min", -1))
        == int(requested_max_unselected_hole)
        and int(evidence.get("requested_max_unselected_hole_max", -1))
        == int(requested_max_unselected_hole)
        and int(evidence.get("effective_budget_min", -1)) > 0
        and int(evidence.get("effective_budget_max", -1)) <= int(requested_budget)
        and int(evidence.get("selected_count_min", -2))
        == int(evidence.get("effective_budget_min", -1))
        and int(evidence.get("selected_count_max", -2))
        == int(evidence.get("effective_budget_max", -1))
        and int(evidence.get("observed_max_unselected_hole_max", requested_max_unselected_hole + 1))
        <= int(requested_max_unselected_hole)
    )


def _read_candidate(candidate: Mapping[str, Any], variant: str) -> dict[str, Any]:
    checkpoint = Path(candidate["checkpoint_path"]).expanduser().resolve()
    summary_path = Path(candidate["summary_path"]).expanduser().resolve()
    records_path = Path(candidate["records_path"]).expanduser().resolve()
    for path, digest, label in (
        (checkpoint, candidate["checkpoint_sha256"], "checkpoint"),
        (summary_path, candidate["summary_sha256"], "summary"),
        (records_path, candidate["records_sha256"], "records"),
    ):
        if not path.is_file() or sha256_file(path) != str(digest):
            raise RuntimeError(f"{variant} candidate {label} drift: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "duca_selection_quality_summary_v2":
        raise RuntimeError("unexpected selection-quality summary schema")

    learned = summary["selection"]["learned"]
    uniform = summary["selection"]["uniform"]
    metrics = {
        "coarse_auroc": _mean(summary, "coarse", "pooled", "auroc"),
        "coarse_auprc_lift": _mean(
            summary, "coarse", "pooled", "auprc_lift"
        ),
        "policy_transition_auroc_r0": _mean(
            summary, "transition", "r0", "policy", "auroc"
        ),
        "pure_delta_transition_auroc_r0": _mean(
            summary, "transition", "r0", "pure_abs_delta_p_action", "auroc"
        ),
        "boundary_recall_r0_gain": _mean(
            learned, "boundary_recall", "r0"
        )
        - _mean(uniform, "boundary_recall", "r0"),
        "uniform_minus_learned_endpoint_distance": _mean(
            uniform, "mean_endpoint_distance"
        )
        - _mean(learned, "mean_endpoint_distance"),
        "learned_max_unselected_hole": _mean(
            learned, "max_unselected_hole"
        ),
        "learned_selected_count": _mean(learned, "selected_count"),
    }
    burst_key = VARIANT_SPECS[variant]
    if burst_key is not None:
        learned_burst = learned["boundary_burst"][burst_key]
        uniform_burst = uniform["boundary_burst"][burst_key]
        for field in (
            "endpoint_quota_recall",
            "endpoint_bilateral_recall",
            "both_endpoints_quota_recall",
        ):
            metrics[f"{field}_gain"] = _mean(learned_burst, field) - _mean(
                uniform_burst, field
            )
    gates = {
        "coarse_auroc_at_least_0_55": metrics["coarse_auroc"] >= 0.55,
        "coarse_auprc_above_prevalence": metrics["coarse_auprc_lift"] > 1.0,
        "transition_scorer_not_worse_than_pure_delta_r0": (
            metrics["policy_transition_auroc_r0"]
            >= metrics["pure_delta_transition_auroc_r0"]
        ),
        "endpoint_centering_not_worse_than_uniform": (
            metrics["uniform_minus_learned_endpoint_distance"] >= 0.0
        ),
        "exact_effective_budget_per_sample": _effective_budget_contract_verified(
            summary,
        ),
    }
    if burst_key is not None:
        gates.update(
            {
                "burst_endpoint_quota_gain_positive": metrics[
                    "endpoint_quota_recall_gain"
                ]
                > 0.0,
                "burst_bilateral_gain_positive": metrics[
                    "endpoint_bilateral_recall_gain"
                ]
                > 0.0,
                "burst_both_endpoints_quota_gain_positive": metrics[
                    "both_endpoints_quota_recall_gain"
                ]
                > 0.0,
            }
        )
    epoch = int(candidate["epoch_one_based"])
    if epoch not in {5, 10, 15, 20}:
        raise RuntimeError("candidate epoch is outside the frozen P0 cadence")
    return {
        "variant": variant,
        "epoch_one_based": epoch,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "metrics": metrics,
        "gates": gates,
        "all_sanity_gates_pass": all(gates.values()),
    }


def _ranking_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (int(candidate["epoch_one_based"]),)


def select_variants(
    *,
    expected_commit: str,
    split_manifest: str | Path,
    split_manifest_sha256: str,
    receipt_paths: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    if len(expected_commit) != 40:
        raise ValueError("expected commit must be exact")
    split_binding = validate_split_manifest(
        split_manifest,
        expected_manifest_sha256=split_manifest_sha256,
    )
    split_path = Path(split_binding["manifest_path"])

    candidates: dict[str, list[dict[str, Any]]] = {
        key: [] for key in VARIANT_SPECS
    }
    for raw in receipt_paths:
        path = Path(raw).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        variant = str(payload.get("variant"))
        if (
            variant not in VARIANT_SPECS
            or payload.get("schema") != "duca_frontend_variant_completion_v1"
            or payload.get("ok") is not True
            or payload.get("git_commit") != expected_commit
            or payload.get("split_manifest_sha256") != split_manifest_sha256
            or payload.get("test_subset_consumed") is not False
        ):
            raise RuntimeError(f"invalid mechanism completion receipt: {path}")
        rows = payload.get("candidates", [])
        if [int(row["epoch_one_based"]) for row in rows] != [5, 10, 15, 20]:
            raise RuntimeError(f"{variant} does not cover the frozen checkpoint cadence")
        candidates[variant].extend(_read_candidate(row, variant) for row in rows)

    winners: dict[str, dict[str, Any]] = {}
    for variant, rows in candidates.items():
        if len(rows) != 4:
            raise RuntimeError(f"{variant} requires four P0 checkpoints")
        eligible = [row for row in rows if row["all_sanity_gates_pass"]]
        if eligible:
            winners[variant] = sorted(eligible, key=_ranking_key)[0]
    ok = set(winners) == set(VARIANT_SPECS)
    payload = {
        "schema": SCHEMA,
        "ok": ok,
        "status": (
            "GO_TO_MATCHED_FOUR_ARM_OFFICIAL60"
            if ok
            else "HOLD_P0_SANITY_GATE_FAILED"
        ),
        "git_commit": expected_commit,
        "decision_metric_scope": "training_subset_internal_holdout_only",
        "test_subset_consumed": False,
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": split_manifest_sha256,
        "split_binding": split_binding,
        "selection_rule": (
            "earliest passing checkpoint at epoch 5/10/15/20; no holdout metric "
            "optimization; final method ranking is reserved for matched terminal-EMA TAD mAP"
        ),
        "winners": winners,
        "candidates": candidates,
        "paper_metric_claim_allowed": False,
    }
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--receipt", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    payload = select_variants(
        expected_commit=args.expected_commit,
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_manifest_sha256,
        receipt_paths=args.receipt,
        output_path=args.output_json,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
