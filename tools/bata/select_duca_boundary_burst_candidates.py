from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.analyze_duca_selection_quality import (
    SUMMARY_SCHEMA_VERSION,
    analyze_jsonl,
)
from tools.bata.create_duca_frontend_split import validate_split_manifest
from tools.bata.finalize_duca_r0_boundary_burst import revalidate_r0_summary
from tools.bata.select_duca_frontend_checkpoint import sha256_file


SCHEMA = "duca_boundary_burst_frontend_decision_v1"
P0_REAL_GATE_SCHEMA = "duca_frontend_p0_real_cuda_gate_v1"
P0_ANALYZER_BOOTSTRAP_SAMPLES = 2000
P0_ANALYZER_RANDOM_SEED = 3407
P0_ANALYZER_REPRESENTATIVE_PER_STRATUM = 2
VARIANT_SPECS = {
    "gaussian_matched": None,
    "burst_r2q3": "r2q3",
    "burst_r4q5": "r4q5",
}


def _first_mismatch(
    expected: Any,
    observed: Any,
    *,
    path: str = "summary",
) -> str | None:
    """Return the first recursive difference so evidence drift is actionable."""

    if type(expected) is not type(observed):
        return f"{path} (type {type(expected).__name__} != {type(observed).__name__})"
    if isinstance(expected, Mapping):
        if set(expected) != set(observed):
            return f"{path} (keys differ)"
        for key in sorted(expected, key=str):
            mismatch = _first_mismatch(
                expected[key], observed[key], path=f"{path}.{key}"
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return f"{path} (length {len(expected)} != {len(observed)})"
        for index, (left, right) in enumerate(zip(expected, observed)):
            mismatch = _first_mismatch(left, right, path=f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    return None if expected == observed else path


def _recompute_selection_summary(
    *,
    records_path: Path,
    records_sha256: str,
    summary_path: Path,
    summary_sha256: str,
) -> dict[str, Any]:
    """Accept only a summary regenerated from the sealed production records."""

    if not records_path.read_text(encoding="utf-8").strip():
        raise RuntimeError("candidate records JSONL is empty")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise RuntimeError("selection-quality summary is not a mapping")
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise RuntimeError("unexpected selection-quality summary schema")
    try:
        summary_records_path = Path(str(summary["records_jsonl"])).expanduser().resolve()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("selection-quality summary lacks records identity") from exc
    if summary_records_path != records_path:
        raise RuntimeError("selection-quality summary records identity drift")

    # This is deliberately the same production analyzer and fixed P0 invocation.
    with tempfile.TemporaryDirectory(prefix="duca-p0-summary-reanalysis-") as output_dir:
        recomputed = analyze_jsonl(
            records_jsonl=records_path,
            output_dir=output_dir,
            bootstrap_samples=P0_ANALYZER_BOOTSTRAP_SAMPLES,
            random_seed=P0_ANALYZER_RANDOM_SEED,
            representative_per_stratum=P0_ANALYZER_REPRESENTATIVE_PER_STRATUM,
        )
    if sha256_file(records_path) != str(records_sha256):
        raise RuntimeError("candidate records drifted during production reanalysis")
    if sha256_file(summary_path) != str(summary_sha256):
        raise RuntimeError("candidate summary drifted during production reanalysis")
    mismatch = _first_mismatch(summary, recomputed)
    if mismatch is not None:
        raise RuntimeError(
            f"selection-quality summary disagrees with production reanalysis at {mismatch}"
        )
    return dict(summary)


def validate_p0_real_gate(
    *,
    gate_path: str | Path,
    gate_sha256: str,
    expected_commit: str,
) -> dict[str, Any]:
    """Reopen the real P0 CUDA gate before it can authorize any consumer."""

    path = _verified_file(gate_path, gate_sha256, label="P0 real gate")
    payload = json.loads(path.read_text(encoding="utf-8"))
    git_binding = payload.get("git_binding") if isinstance(payload, Mapping) else None
    final_git_binding = (
        payload.get("final_git_binding") if isinstance(payload, Mapping) else None
    )
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != P0_REAL_GATE_SCHEMA
        or payload.get("ok") is not True
        or payload.get("fail_closed") is not True
        or not isinstance(git_binding, Mapping)
        or not isinstance(final_git_binding, Mapping)
        or git_binding.get("git_commit") != expected_commit
        or final_git_binding.get("git_commit") != expected_commit
    ):
        raise RuntimeError("P0 real gate contract drift")
    return {
        "path": str(path),
        "sha256": str(gate_sha256),
        "schema": P0_REAL_GATE_SCHEMA,
        "git_commit": expected_commit,
        "ok": True,
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
    """Reopen and recompute the complete R0 evidence chain before P0."""

    return revalidate_r0_summary(
        summary_path=summary_path,
        summary_file_sha256=summary_sha256,
        expected_commit=expected_commit,
    )


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
    summary = _recompute_selection_summary(
        records_path=records_path,
        records_sha256=str(candidate["records_sha256"]),
        summary_path=summary_path,
        summary_sha256=str(candidate["summary_sha256"]),
    )

    learned = summary["selection"]["learned"]
    uniform = summary["selection"]["uniform"]
    simple_delta = summary["selection"].get("pure_delta_same_feasible_dp")
    if not isinstance(simple_delta, Mapping):
        raise RuntimeError(
            "selection summary lacks the same-exact-K/max-hole simple-delta DP"
        )
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
        "learned_boundary_recall_r0": _mean(
            learned, "boundary_recall", "r0"
        ),
        "simple_delta_boundary_recall_r0": _mean(
            simple_delta, "boundary_recall", "r0"
        ),
        "uniform_minus_learned_endpoint_distance": _mean(
            uniform, "mean_endpoint_distance"
        )
        - _mean(learned, "mean_endpoint_distance"),
        "simple_delta_minus_learned_endpoint_distance": _mean(
            simple_delta, "mean_endpoint_distance"
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
        simple_delta_burst = simple_delta["boundary_burst"][burst_key]
        for field in (
            "endpoint_quota_recall",
            "endpoint_bilateral_recall",
            "both_endpoints_quota_recall",
        ):
            metrics[f"{field}_gain"] = _mean(learned_burst, field) - _mean(
                uniform_burst, field
            )
            metrics[f"{field}_gain_vs_simple_delta"] = _mean(
                learned_burst, field
            ) - _mean(simple_delta_burst, field)
    simple_delta_pareto_gains = [
        metrics["learned_boundary_recall_r0"]
        - metrics["simple_delta_boundary_recall_r0"],
        metrics["simple_delta_minus_learned_endpoint_distance"],
    ]
    if burst_key is not None:
        simple_delta_pareto_gains.extend(
            metrics[f"{field}_gain_vs_simple_delta"]
            for field in (
                "endpoint_quota_recall",
                "endpoint_bilateral_recall",
                "both_endpoints_quota_recall",
            )
        )
    simple_delta_stop_rule_pass = all(
        value >= 0.0 for value in simple_delta_pareto_gains
    ) and any(value > 0.0 for value in simple_delta_pareto_gains)
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
        "learned_selector_strictly_pareto_beats_same_feasible_simple_delta": (
            simple_delta_stop_rule_pass
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
    p0_real_gate_path: str | Path | None = None,
    p0_real_gate_sha256: str | None = None,
) -> dict[str, Any]:
    if len(expected_commit) != 40:
        raise ValueError("expected commit must be exact")
    split_binding = validate_split_manifest(
        split_manifest,
        expected_manifest_sha256=split_manifest_sha256,
    )
    split_path = Path(split_binding["manifest_path"])
    if p0_real_gate_path is None or p0_real_gate_sha256 is None:
        raise RuntimeError("P0 real gate binding is required")
    p0_real_gate = validate_p0_real_gate(
        gate_path=p0_real_gate_path,
        gate_sha256=p0_real_gate_sha256,
        expected_commit=expected_commit,
    )

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
        "p0_real_gate": p0_real_gate,
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
    parser.add_argument("--p0-real-gate", required=True)
    parser.add_argument("--p0-real-gate-sha256", required=True)
    parser.add_argument("--receipt", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    payload = select_variants(
        expected_commit=args.expected_commit,
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_manifest_sha256,
        p0_real_gate_path=args.p0_real_gate,
        p0_real_gate_sha256=args.p0_real_gate_sha256,
        receipt_paths=args.receipt,
        output_path=args.output_json,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
