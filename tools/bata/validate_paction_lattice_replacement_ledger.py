from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata import apply_paction_acquisition_policy as apply_policy
from tools.bata import paction_lattice_replacement_policy as lattice
from tools.bata import validate_paction_learned_policy_ledger as base_validator


SUMMARY_SCHEMA_VERSION = "c3_paction_lattice_replacement_ledger_validation_v1"
READY = "C3_PACTION_LATTICE_REPLACEMENT_LEDGER_VALIDATION_PASS"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL has no rows: {path}")
    return rows


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assert_lattice_metadata(sample_rows: Sequence[Mapping[str, Any]], *, strategy: str) -> dict[str, Any]:
    selected_counts: list[int] = []
    replaced_counts: list[int] = []
    protected_counts: list[int] = []
    for row_index, row in enumerate(sample_rows, start=1):
        policy = row.get("paction_policy")
        if not isinstance(policy, Mapping):
            raise ValueError(f"sample row {row_index}: paction_policy metadata is required")
        if policy.get("source") != apply_policy.CHECKPOINT_POLICY_SOURCE:
            raise ValueError(f"sample row {row_index}: paction_policy.source must be {apply_policy.CHECKPOINT_POLICY_SOURCE}")
        if policy.get("selection_decoder") != "score_only_lattice_replacement_v1":
            raise ValueError(f"sample row {row_index}: selection_decoder must be score_only_lattice_replacement_v1")
        if policy.get("score_only") is not True:
            raise ValueError(f"sample row {row_index}: score_only must be true")
        for key in (
            "uses_manual_boundary_slots",
            "uses_manual_transition_slots",
            "uses_manual_uncertainty_slots",
            "uses_manual_context_slots",
            "uses_uniform_scaffold",
            "uses_uniform_fill",
        ):
            if policy.get(key) is not False:
                raise ValueError(f"sample row {row_index}: {key} must be false")
        diagnostics_by_strategy = policy.get("lattice_replacement_diagnostics_by_strategy")
        if not isinstance(diagnostics_by_strategy, Mapping) or strategy not in diagnostics_by_strategy:
            raise ValueError(f"sample row {row_index}: missing lattice diagnostics for strategy {strategy}")
        diagnostics = diagnostics_by_strategy[strategy]
        if not isinstance(diagnostics, Mapping):
            raise ValueError(f"sample row {row_index}: lattice diagnostics for {strategy} must be a JSON object")
        for forbidden in ("gt", "teacher", "oracle", "boundary", "transition", "uncertainty", "context", "role"):
            if any(forbidden in str(key).lower() for key in diagnostics):
                raise ValueError(f"sample row {row_index}: lattice diagnostics contain forbidden key fragment {forbidden}")
        selected_counts.append(int(diagnostics.get("selected_count", 0)))
        replaced_counts.append(int(diagnostics.get("replaced_uniform_count", 0)))
        protected_counts.append(int(diagnostics.get("protected_uniform_count", 0)))
    return {
        "row_count": len(sample_rows),
        "min_selected_count": min(selected_counts) if selected_counts else None,
        "max_selected_count": max(selected_counts) if selected_counts else None,
        "mean_replaced_uniform_count": None if not replaced_counts else sum(replaced_counts) / float(len(replaced_counts)),
        "min_protected_uniform_count": min(protected_counts) if protected_counts else None,
        "max_protected_uniform_count": max(protected_counts) if protected_counts else None,
    }


def validate_lattice_ledger(
    *,
    sample_jsonl: str | Path,
    metric_sample_jsonl: str | Path,
    ledger_jsonl: str | Path,
    strategy: str,
    expected_target_len: int = lattice.DEFAULT_BUDGET,
    require_selected_count: int | None = lattice.DEFAULT_BUDGET,
    allow_short_valid_ratio_count: bool = True,
    require_deployable: bool = True,
    require_checkpoint_path: str | Path | None = None,
    require_checkpoint_sha256: str | None = None,
    max_max_gap: int | None = None,
    max_p95_gap: float | None = None,
    max_unselected_hole: int | None = None,
    max_p95_unselected_hole: float | None = None,
    max_uniform_similarity: float | None = None,
    summary_json: str | Path | None = None,
) -> dict[str, Any]:
    base_summary = base_validator.validate_ledger(
        sample_jsonl=sample_jsonl,
        metric_sample_jsonl=metric_sample_jsonl,
        ledger_jsonl=ledger_jsonl,
        strategy=strategy,
        expected_target_len=int(expected_target_len),
        require_selected_count=require_selected_count,
        allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
        require_nonconstant_selected_count=False,
        require_deployable=bool(require_deployable),
        max_max_gap=max_max_gap,
        max_p95_gap=max_p95_gap,
        max_unselected_hole=max_unselected_hole,
        max_p95_unselected_hole=max_p95_unselected_hole,
        max_uniform_similarity=max_uniform_similarity,
        boundary_radii=[1, 2, 4, 8],
        require_policy_source=apply_policy.CHECKPOINT_POLICY_SOURCE,
        require_checkpoint_path=require_checkpoint_path,
        require_checkpoint_sha256=require_checkpoint_sha256,
        require_paction_provenance=bool(require_deployable),
    )
    lattice_summary = _assert_lattice_metadata(_read_jsonl(sample_jsonl), strategy=strategy)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "strategy": str(strategy),
        "sample_jsonl": str(sample_jsonl),
        "metric_sample_jsonl": str(metric_sample_jsonl),
        "ledger_jsonl": str(ledger_jsonl),
        "base_validation": base_summary,
        "lattice_metadata": lattice_summary,
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate score-only PAction lattice replacement ledgers.")
    parser.add_argument("--sample-jsonl", required=True)
    parser.add_argument("--metric-sample-jsonl", required=True)
    parser.add_argument("--ledger-jsonl", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--expected-target-len", type=int, default=lattice.DEFAULT_BUDGET)
    parser.add_argument("--require-selected-count", type=int, default=lattice.DEFAULT_BUDGET)
    parser.add_argument("--no-allow-short-valid-ratio-count", action="store_true")
    parser.add_argument("--no-require-deployable", action="store_true")
    parser.add_argument("--require-checkpoint-path")
    parser.add_argument("--require-checkpoint-sha256")
    parser.add_argument("--max-max-gap", type=int)
    parser.add_argument("--max-p95-gap", type=float)
    parser.add_argument("--max-unselected-hole", type=int)
    parser.add_argument("--max-p95-unselected-hole", type=float)
    parser.add_argument("--max-uniform-similarity", type=float)
    parser.add_argument("--summary-json")
    args = parser.parse_args(argv)

    summary = validate_lattice_ledger(
        sample_jsonl=args.sample_jsonl,
        metric_sample_jsonl=args.metric_sample_jsonl,
        ledger_jsonl=args.ledger_jsonl,
        strategy=str(args.strategy),
        expected_target_len=int(args.expected_target_len),
        require_selected_count=args.require_selected_count,
        allow_short_valid_ratio_count=not bool(args.no_allow_short_valid_ratio_count),
        require_deployable=not bool(args.no_require_deployable),
        require_checkpoint_path=args.require_checkpoint_path,
        require_checkpoint_sha256=args.require_checkpoint_sha256,
        max_max_gap=args.max_max_gap,
        max_p95_gap=args.max_p95_gap,
        max_unselected_hole=args.max_unselected_hole,
        max_p95_unselected_hole=args.max_p95_unselected_hole,
        max_uniform_similarity=args.max_uniform_similarity,
        summary_json=args.summary_json,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
