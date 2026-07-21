from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.select_duca_frontend_checkpoint import sha256_file


SCHEMA = "duca_boundary_burst_frontend_decision_v1"
VARIANT_SPECS = {
    "gaussian_matched": None,
    "burst_r2q3": "r2q3",
    "burst_r4q5": "r4q5",
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
    mean_selected_count: float,
    *,
    requested_budget: int = 384,
) -> bool:
    """Accept exact per-sample min(K, valid_len), including short tail windows."""

    protocol = summary.get("protocol", {})
    return bool(
        isinstance(protocol, Mapping)
        and protocol.get("budget_matched") is True
        and protocol.get("valid_length_matched") is True
        and 0.0 < float(mean_selected_count) <= float(requested_budget)
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
        "exact_effective_budget_per_sample": _effective_budget_contract_verified(
            summary,
            metrics["learned_selected_count"],
        ),
        "max_unselected_hole_at_most_2": metrics[
            "learned_max_unselected_hole"
        ]
        <= 2.0,
    }
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
    metrics = candidate["metrics"]
    if candidate["variant"].startswith("burst_"):
        mechanism = (
            -metrics["both_endpoints_quota_recall_gain"],
            -metrics["endpoint_quota_recall_gain"],
            -metrics["endpoint_bilateral_recall_gain"],
        )
    else:
        mechanism = ()
    return (
        *mechanism,
        -metrics["boundary_recall_r0_gain"],
        -metrics["uniform_minus_learned_endpoint_distance"],
        -metrics["policy_transition_auroc_r0"],
        candidate["epoch_one_based"],
    )


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
    split_path = Path(split_manifest).expanduser().resolve()
    if not split_path.is_file() or sha256_file(split_path) != split_manifest_sha256:
        raise RuntimeError("frontend split manifest drift")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("test_subset_consumed") is not False:
        raise RuntimeError("P0 split consumed the test subset")

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
        "selection_rule": (
            "select one stable checkpoint per mechanism; final method ranking is reserved "
            "for matched terminal-EMA TAD mAP"
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
