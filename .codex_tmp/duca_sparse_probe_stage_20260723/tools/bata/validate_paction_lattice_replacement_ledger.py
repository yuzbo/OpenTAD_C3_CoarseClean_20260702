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
        if policy.get("selection_decoder") not in {
            "score_only_lattice_replacement_v1",
            "score_only_lattice_replacement_with_adaptive_radius_v1",
        }:
            raise ValueError(f"sample row {row_index}: unsupported selection_decoder={policy.get('selection_decoder')!r}")
        if policy.get("score_only") is not True:
            raise ValueError(f"sample row {row_index}: score_only must be true")
        if policy.get("diagnostic_only") is not True:
            raise ValueError(f"sample row {row_index}: diagnostic_only must be true")
        if policy.get("paper_main_claim_allowed") is not False:
            raise ValueError(f"sample row {row_index}: paper_main_claim_allowed must be false")
        if policy.get("uses_uniform_scaffold") is not True:
            raise ValueError(f"sample row {row_index}: uses_uniform_scaffold must be true")
        if policy.get("scaffold_type") != "uniform_lattice_local_replacement":
            raise ValueError(f"sample row {row_index}: scaffold_type must be uniform_lattice_local_replacement")
        for key in (
            "uses_manual_boundary_slots",
            "uses_manual_transition_slots",
            "uses_manual_uncertainty_slots",
            "uses_manual_context_slots",
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
        if lattice.is_adaptive_radius_strategy(str(strategy)):
            radii_by_strategy = policy.get("context_radius_by_strategy")
            if not isinstance(radii_by_strategy, Mapping) or str(strategy) not in radii_by_strategy:
                raise ValueError(f"sample row {row_index}: missing context_radius_by_strategy for {strategy}")
            if policy.get("context_radius_unit") != "local_dense_snippet_index":
                raise ValueError(f"sample row {row_index}: context_radius_unit must be local_dense_snippet_index")
            radius_range = policy.get("context_radius_range")
            if radius_range != [0.0, 16.0]:
                raise ValueError(f"sample row {row_index}: context_radius_range must be [0.0, 16.0]")
            radius_diagnostics = policy.get("lattice_radius_diagnostics_by_strategy")
            if not isinstance(radius_diagnostics, Mapping) or str(strategy) not in radius_diagnostics:
                raise ValueError(f"sample row {row_index}: missing lattice radius diagnostics for {strategy}")
            expanded_by_strategy = policy.get("budgeted_expanded_positions_by_strategy")
            if not isinstance(expanded_by_strategy, Mapping) or str(strategy) not in expanded_by_strategy:
                raise ValueError(f"sample row {row_index}: missing budgeted expanded positions for {strategy}")
            expanded_diagnostics = policy.get("budgeted_expanded_diagnostics_by_strategy")
            if not isinstance(expanded_diagnostics, Mapping) or str(strategy) not in expanded_diagnostics:
                raise ValueError(f"sample row {row_index}: missing budgeted expanded diagnostics for {strategy}")
    return {
        "row_count": len(sample_rows),
        "min_selected_count": min(selected_counts) if selected_counts else None,
        "max_selected_count": max(selected_counts) if selected_counts else None,
        "mean_replaced_uniform_count": None if not replaced_counts else sum(replaced_counts) / float(len(replaced_counts)),
        "min_protected_uniform_count": min(protected_counts) if protected_counts else None,
        "max_protected_uniform_count": max(protected_counts) if protected_counts else None,
    }


def _assert_radius_expanded_budget(
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    ledger_jsonl: str | Path,
    require_selected_count: int | None,
    allow_short_valid_ratio_count: bool,
) -> dict[str, Any]:
    expanded_counts: list[int] = []
    center_counts: list[int] = []
    for line_no, row in enumerate(ledger_rows, start=1):
        selected = base_validator._positions(row.get("selected_positions"), name=f"{ledger_jsonl}:{line_no}: selected_positions")
        expanded = base_validator._positions(
            row.get("expanded_selected_positions"),
            name=f"{ledger_jsonl}:{line_no}: expanded_selected_positions",
        )
        valid_len = int(row.get("valid_len"))
        dense_len = int(row.get("dense_len") or valid_len)
        if row.get("selected_positions_are_centers") is not True:
            raise ValueError(f"{ledger_jsonl}:{line_no}: radius ledger selected_positions must be centers")
        if int(row.get("selected_count")) != len(selected):
            raise ValueError(f"{ledger_jsonl}:{line_no}: selected_count mismatch")
        if int(row.get("expanded_selected_count")) != len(expanded):
            raise ValueError(f"{ledger_jsonl}:{line_no}: expanded_selected_count mismatch")
        if any(item >= valid_len for item in expanded):
            raise ValueError(f"{ledger_jsonl}:{line_no}: expanded position outside valid_len")
        target_len = int(row.get("target_len"))
        if len(expanded) > target_len:
            raise ValueError(f"{ledger_jsonl}:{line_no}: expanded_selected_count exceeds target_len")
        expected_count = base_validator.paction_budget_contract.expected_selected_count(
            require_selected_count,
            valid_len=valid_len,
            dense_len=dense_len,
            allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
        )
        if expected_count is not None and len(expanded) != int(expected_count):
            raise ValueError(f"{ledger_jsonl}:{line_no}: expanded_selected_count must be {expected_count}")
        diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), Mapping) else {}
        if diagnostics.get("budgeted_expanded_selection") is not True:
            raise ValueError(f"{ledger_jsonl}:{line_no}: budgeted_expanded_selection must be true")
        if int(diagnostics.get("center_count", -1)) != len(selected):
            raise ValueError(f"{ledger_jsonl}:{line_no}: center_count mismatch")
        if int(diagnostics.get("budgeted_expanded_count", -1)) != len(expanded):
            raise ValueError(f"{ledger_jsonl}:{line_no}: budgeted_expanded_count mismatch")
        expanded_counts.append(len(expanded))
        center_counts.append(len(selected))
    return {
        "min_center_count": min(center_counts) if center_counts else None,
        "max_center_count": max(center_counts) if center_counts else None,
        "min_expanded_selected_count": min(expanded_counts) if expanded_counts else None,
        "max_expanded_selected_count": max(expanded_counts) if expanded_counts else None,
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
    is_radius = lattice.is_adaptive_radius_strategy(str(strategy))
    base_summary = base_validator.validate_ledger(
        sample_jsonl=sample_jsonl,
        metric_sample_jsonl=metric_sample_jsonl,
        ledger_jsonl=ledger_jsonl,
        strategy=strategy,
        expected_target_len=int(expected_target_len),
        require_selected_count=None if is_radius else require_selected_count,
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
        allow_policy_uniform_scaffold=True,
    )
    lattice_summary = _assert_lattice_metadata(_read_jsonl(sample_jsonl), strategy=strategy)
    radius_budget_summary = None
    if is_radius:
        radius_budget_summary = _assert_radius_expanded_budget(
            _read_jsonl(ledger_jsonl),
            ledger_jsonl=ledger_jsonl,
            require_selected_count=require_selected_count,
            allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
        )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "strategy": str(strategy),
        "sample_jsonl": str(sample_jsonl),
        "metric_sample_jsonl": str(metric_sample_jsonl),
        "ledger_jsonl": str(ledger_jsonl),
        "base_validation": base_summary,
        "lattice_metadata": lattice_summary,
        "radius_budget": radius_budget_summary,
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
    parser.add_argument("--allow-short-valid-ratio-count", action="store_true")
    parser.add_argument("--no-allow-short-valid-ratio-count", action="store_true")
    parser.add_argument("--require-deployable", action="store_true")
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
