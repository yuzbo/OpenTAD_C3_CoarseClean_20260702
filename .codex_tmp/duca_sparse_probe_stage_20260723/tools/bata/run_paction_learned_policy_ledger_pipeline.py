from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from tools.bata import apply_paction_acquisition_policy as apply_policy
from tools.bata import convert_lowres_probe_samples_to_value_transport_ledger as convert_ledger
from tools.bata import paction_acquisition_policy as policy
from tools.bata import paction_source_samples
from tools.bata import validate_paction_learned_policy_ledger as validate_ledger


SUMMARY_SCHEMA_VERSION = "c3_paction_learned_policy_ledger_pipeline_v1"
READY = "C3_PACTION_LEARNED_POLICY_LEDGER_PIPELINE_READY"


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _convert_and_validate(
    *,
    sample_jsonl: Path,
    metric_sample_jsonl: Path,
    out_dir: Path,
    name: str,
    strategy: str,
    target_len: int,
    require_selected_count: int | None,
    require_nonconstant_selected_count: bool,
    deploy_selection_ledger: bool,
    allow_short_valid_ratio_count: bool,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    min_boundary_support: float | None,
    min_action_coverage: float | None,
    max_max_gap: int | None,
    max_p95_gap: float | None,
    max_unselected_hole: int | None,
    max_p95_unselected_hole: float | None,
    max_uniform_similarity: float | None,
) -> dict[str, Any]:
    ledger_jsonl = out_dir / f"value_transport_ledger_{name}.jsonl"
    ledger_summary_json = out_dir / f"value_transport_ledger_{name}.summary.json"
    validation_summary_json = out_dir / f"value_transport_ledger_{name}.validation.json"
    ledger_summary = convert_ledger.run_conversion(
        sample_jsonl,
        ledger_jsonl,
        strategy=strategy,
        target_len=int(target_len),
        summary_json=ledger_summary_json,
        require_selected_count=require_selected_count,
        allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
        deploy_selection_ledger=bool(deploy_selection_ledger),
        deduplicate_sample_id=True,
        route_variant=f"c3_paction_learned_policy_{name}",
    )
    validation_summary = validate_ledger.validate_ledger(
        sample_jsonl=sample_jsonl,
        metric_sample_jsonl=metric_sample_jsonl,
        ledger_jsonl=ledger_jsonl,
        strategy=strategy,
        expected_target_len=int(target_len),
        require_selected_count=require_selected_count,
        allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
        require_nonconstant_selected_count=bool(require_nonconstant_selected_count),
        require_deployable=bool(deploy_selection_ledger),
        min_boundary_support=min_boundary_support,
        min_action_coverage=min_action_coverage,
        max_max_gap=max_max_gap,
        max_p95_gap=max_p95_gap,
        max_unselected_hole=max_unselected_hole,
        max_p95_unselected_hole=max_p95_unselected_hole,
        max_uniform_similarity=max_uniform_similarity,
        boundary_radii=[1, 2, 4, 8],
        require_policy_source=apply_policy.CHECKPOINT_POLICY_SOURCE,
        require_checkpoint_path=checkpoint_path,
        require_checkpoint_sha256=checkpoint_sha256,
        require_paction_provenance=bool(deploy_selection_ledger),
        summary_json=validation_summary_json,
    )
    return {
        "sample_jsonl": str(sample_jsonl),
        "metric_sample_jsonl": str(metric_sample_jsonl),
        "ledger_jsonl": str(ledger_jsonl),
        "ledger_summary_json": str(ledger_summary_json),
        "validation_summary_json": str(validation_summary_json),
        "ledger_summary": ledger_summary,
        "validation_summary": validation_summary,
    }


def run_pipeline(
    *,
    input_jsonl: str | Path,
    checkpoint_path: str | Path,
    out_dir: str | Path,
    fixed_budgets: Sequence[int] = (384, 768),
    dynamic_target_len: int = 768,
    dynamic_budget_buckets: Sequence[int] = policy.DEFAULT_DYNAMIC_BUDGET_BUCKETS,
    device: str = "cuda",
    deploy_selection_ledger: bool = True,
    allow_short_valid_ratio_count: bool = True,
    min_boundary_support: float | None = None,
    min_action_coverage: float | None = None,
    max_max_gap: int | None = None,
    max_p95_gap: float | None = None,
    max_unselected_hole: int | None = None,
    max_p95_unselected_hole: float | None = None,
    max_uniform_similarity: float | None = None,
    require_dynamic_nonconstant_count: bool = False,
    summary_json: str | Path | None = None,
) -> dict[str, Any]:
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)
    canonical_input_jsonl = out_path / "source.canonical_unique.jsonl"
    source_canonicalization = paction_source_samples.canonicalize_unique_sample_jsonl(
        input_jsonl,
        canonical_input_jsonl,
        report_json=out_path / "source.canonical_unique.report.json",
        split="",
    )
    selection_input_jsonl: Path | None = None
    selection_source_report: dict[str, Any] | None = None
    if deploy_selection_ledger:
        selection_input_jsonl = out_path / "source.selection_deploy.jsonl"
        selection_source_report = paction_source_samples.write_deploy_selection_source_jsonl(
            canonical_input_jsonl,
            selection_input_jsonl,
            report_json=out_path / "source.selection_deploy.report.json",
            split="",
        )
    input_sample_path = selection_input_jsonl if selection_input_jsonl is not None else canonical_input_jsonl
    metric_sample_path = canonical_input_jsonl
    checkpoint_sha256 = apply_policy._sha256_file(checkpoint_path)
    ledgers: dict[str, Any] = {}
    for fixed_budget in [int(item) for item in fixed_budgets]:
        name = f"learned_fixed_{fixed_budget}"
        sample_jsonl = out_path / f"samples.{name}.jsonl"
        apply_policy.run_policy_application(
            input_sample_path,
            sample_jsonl,
            summary_json=out_path / f"samples.{name}.summary.json",
            fixed_budget=int(fixed_budget),
            dynamic_budget_buckets=[int(item) for item in dynamic_budget_buckets],
            checkpoint_path=checkpoint_path,
            device=device,
            strip_deploy_invisible_payload=True,
            strict_deploy_source=bool(deploy_selection_ledger),
            max_unselected_hole=max_unselected_hole,
        )
        ledgers[name] = _convert_and_validate(
            sample_jsonl=sample_jsonl,
            metric_sample_jsonl=metric_sample_path,
            out_dir=out_path,
            name=name,
            strategy=policy.LEARNED_FIXED_STRATEGY,
            target_len=int(fixed_budget),
            require_selected_count=int(fixed_budget),
            require_nonconstant_selected_count=False,
            deploy_selection_ledger=bool(deploy_selection_ledger),
            allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            min_boundary_support=min_boundary_support,
            min_action_coverage=min_action_coverage,
            max_max_gap=max_max_gap,
            max_p95_gap=max_p95_gap,
            max_unselected_hole=max_unselected_hole,
            max_p95_unselected_hole=max_p95_unselected_hole,
            max_uniform_similarity=max_uniform_similarity,
        )
    dynamic_sample_jsonl = out_path / "samples.learned_dynamic.jsonl"
    apply_policy.run_policy_application(
        input_sample_path,
        dynamic_sample_jsonl,
        summary_json=out_path / "samples.learned_dynamic.summary.json",
        fixed_budget=int(dynamic_target_len),
        dynamic_budget_buckets=[int(item) for item in dynamic_budget_buckets],
        checkpoint_path=checkpoint_path,
        device=device,
        strip_deploy_invisible_payload=True,
        strict_deploy_source=bool(deploy_selection_ledger),
        max_unselected_hole=max_unselected_hole,
    )
    ledgers["learned_dynamic"] = _convert_and_validate(
        sample_jsonl=dynamic_sample_jsonl,
        metric_sample_jsonl=metric_sample_path,
        out_dir=out_path,
        name="learned_dynamic",
        strategy=policy.LEARNED_DYNAMIC_STRATEGY,
        target_len=int(dynamic_target_len),
        require_selected_count=None,
        require_nonconstant_selected_count=bool(require_dynamic_nonconstant_count),
        deploy_selection_ledger=bool(deploy_selection_ledger),
        allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        min_boundary_support=min_boundary_support,
        min_action_coverage=min_action_coverage,
        max_max_gap=max_max_gap,
        max_p95_gap=max_p95_gap,
        max_unselected_hole=max_unselected_hole,
        max_p95_unselected_hole=max_p95_unselected_hole,
        max_uniform_similarity=max_uniform_similarity,
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "input_jsonl": str(input_jsonl),
        "canonical_input_jsonl": str(canonical_input_jsonl),
        "source_canonicalization": source_canonicalization,
        "selection_sample_jsonl": None if selection_input_jsonl is None else str(selection_input_jsonl),
        "selection_source_report": selection_source_report,
        "metric_sample_jsonl": str(metric_sample_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "out_dir": str(out_path),
        "fixed_budgets": [int(item) for item in fixed_budgets],
        "dynamic_target_len": int(dynamic_target_len),
        "dynamic_budget_buckets": [int(item) for item in dynamic_budget_buckets],
        "deploy_selection_ledger": bool(deploy_selection_ledger),
        "gap_control": (
            "learned_score_constrained_gap_no_uniform_fill"
            if max_unselected_hole is not None
            else "learned_gap_hole_loss_no_uniform_fill"
        ),
        "uses_uniform_scaffold": False,
        "uses_uniform_fill": False,
        "strip_deploy_invisible_payload": True,
        "required_policy_source": apply_policy.CHECKPOINT_POLICY_SOURCE,
        "require_dynamic_nonconstant_count": bool(require_dynamic_nonconstant_count),
        "quality_gate": {
            "min_boundary_support": min_boundary_support,
            "min_action_coverage": min_action_coverage,
            "max_max_gap": max_max_gap,
            "max_p95_gap": max_p95_gap,
            "max_unselected_hole": max_unselected_hole,
            "max_p95_unselected_hole": max_p95_unselected_hole,
            "max_uniform_similarity": max_uniform_similarity,
            "mode": "hard_fail_when_thresholds_are_set",
        },
        "ledgers": ledgers,
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate fixed and dynamic strict ledgers from a learned p_action policy checkpoint.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--fixed-budgets", type=int, nargs="+", default=[384, 768])
    parser.add_argument("--dynamic-target-len", type=int, default=768)
    parser.add_argument("--dynamic-budget-buckets", type=int, nargs="+", default=list(policy.DEFAULT_DYNAMIC_BUDGET_BUCKETS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnostic-ledger", action="store_true")
    parser.add_argument("--disable-short-valid-ratio-count", action="store_true")
    parser.add_argument("--min-boundary-support", type=float)
    parser.add_argument("--min-action-coverage", type=float)
    parser.add_argument("--max-max-gap", type=int)
    parser.add_argument("--max-p95-gap", type=float)
    parser.add_argument("--max-unselected-hole", type=int)
    parser.add_argument("--max-p95-unselected-hole", type=float)
    parser.add_argument("--max-uniform-similarity", type=float)
    parser.add_argument("--require-dynamic-nonconstant-count", action="store_true")
    args = parser.parse_args(argv)
    summary = run_pipeline(
        input_jsonl=args.input_jsonl,
        checkpoint_path=args.checkpoint_path,
        out_dir=args.out_dir,
        fixed_budgets=[int(item) for item in args.fixed_budgets],
        dynamic_target_len=int(args.dynamic_target_len),
        dynamic_budget_buckets=[int(item) for item in args.dynamic_budget_buckets],
        device=str(args.device),
        deploy_selection_ledger=not bool(args.diagnostic_ledger),
        allow_short_valid_ratio_count=not bool(args.disable_short_valid_ratio_count),
        min_boundary_support=args.min_boundary_support,
        min_action_coverage=args.min_action_coverage,
        max_max_gap=args.max_max_gap,
        max_p95_gap=args.max_p95_gap,
        max_unselected_hole=args.max_unselected_hole,
        max_p95_unselected_hole=args.max_p95_unselected_hole,
        max_uniform_similarity=args.max_uniform_similarity,
        require_dynamic_nonconstant_count=bool(args.require_dynamic_nonconstant_count),
        summary_json=args.summary_json,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
