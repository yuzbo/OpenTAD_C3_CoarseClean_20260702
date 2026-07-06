from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from tools.bata import apply_gap_aware_acquisition_policy as apply_policy
from tools.bata import convert_lowres_probe_samples_to_value_transport_ledger as convert_ledger
from tools.bata import gas_vt_paction_policy as gas_vt
from tools.bata import paction_source_samples
from tools.bata import validate_paction_learned_policy_ledger as validate_ledger


SUMMARY_SCHEMA_VERSION = "c3_gas_vt_ledger_pipeline_v1"
READY = "C3_GAS_VT_LEDGER_PIPELINE_READY"


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
    deploy_selection_ledger: bool,
    allow_short_valid_ratio_count: bool,
    require_nonconstant_selected_count: bool,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    max_hole_top10_csv: Path,
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
        route_variant=f"c3_gas_vt_{name}",
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
        boundary_radii=[1, 2, 4, 8],
        require_policy_source=gas_vt.GAS_VT_CHECKPOINT_POLICY_SOURCE,
        require_checkpoint_path=checkpoint_path,
        require_checkpoint_sha256=checkpoint_sha256,
        require_paction_provenance=bool(deploy_selection_ledger),
        summary_json=validation_summary_json,
        max_hole_top10_csv=max_hole_top10_csv,
        max_unselected_hole=max_unselected_hole,
        max_p95_unselected_hole=max_p95_unselected_hole,
        max_uniform_similarity=max_uniform_similarity,
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
    dynamic_budget_buckets: Sequence[int] = gas_vt.DEFAULT_GAS_VT_DYNAMIC_BUDGET_BUCKETS,
    device: str = "cuda",
    deploy_selection_ledger: bool = True,
    allow_short_valid_ratio_count: bool = True,
    max_unselected_hole: int | None = None,
    max_p95_unselected_hole: float | None = None,
    max_uniform_similarity: float | None = None,
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
    applied_sample_jsonl = out_path / "samples.gas_vt_all.jsonl"
    apply_policy.run_policy_application(
        input_sample_path,
        applied_sample_jsonl,
        summary_json=out_path / "samples.gas_vt_all.summary.json",
        fixed_budgets=[int(item) for item in fixed_budgets],
        dynamic_budget_buckets=[int(item) for item in dynamic_budget_buckets],
        checkpoint_path=checkpoint_path,
        device=device,
        strip_deploy_invisible_payload=True,
        strict_deploy_source=bool(deploy_selection_ledger),
        max_unselected_hole=max_unselected_hole,
        source_jsonl_for_hash=input_sample_path,
    )
    fixed_budget_list = [int(item) for item in fixed_budgets]
    variant_specs = {
        "gas_vt_fixed_384": dict(
            strategy=gas_vt.GAS_VT_FIXED_384_STRATEGY,
            target_len=fixed_budget_list[0],
            require_selected_count=fixed_budget_list[0],
            require_nonconstant_selected_count=False,
        ),
        "gas_vt_fixed_768": dict(
            strategy=gas_vt.GAS_VT_FIXED_768_STRATEGY,
            target_len=fixed_budget_list[1] if len(fixed_budget_list) > 1 else fixed_budget_list[0],
            require_selected_count=fixed_budget_list[1] if len(fixed_budget_list) > 1 else fixed_budget_list[0],
            require_nonconstant_selected_count=False,
        ),
        "gas_vt_dynamic": dict(
            strategy=gas_vt.GAS_VT_DYNAMIC_STRATEGY,
            target_len=int(dynamic_target_len),
            require_selected_count=None,
            require_nonconstant_selected_count=True,
        ),
    }
    ledgers: dict[str, Any] = {}
    for name, spec in variant_specs.items():
        ledgers[name] = _convert_and_validate(
            sample_jsonl=applied_sample_jsonl,
            metric_sample_jsonl=metric_sample_path,
            out_dir=out_path,
            name=name,
            strategy=spec["strategy"],
            target_len=spec["target_len"],
            require_selected_count=spec["require_selected_count"],
            deploy_selection_ledger=bool(deploy_selection_ledger),
            allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
            require_nonconstant_selected_count=bool(spec["require_nonconstant_selected_count"]),
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            max_hole_top10_csv=out_path / f"value_transport_ledger_{name}.max_holes.csv",
            max_unselected_hole=max_unselected_hole,
            max_p95_unselected_hole=max_p95_unselected_hole,
            max_uniform_similarity=max_uniform_similarity,
        )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "policy_family": "GAS-VT",
        "input_jsonl": str(input_jsonl),
        "canonical_input_jsonl": str(canonical_input_jsonl),
        "source_canonicalization": source_canonicalization,
        "selection_sample_jsonl": None if selection_input_jsonl is None else str(selection_input_jsonl),
        "selection_source_report": selection_source_report,
        "metric_sample_jsonl": str(metric_sample_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "out_dir": str(out_path),
        "fixed_budgets": fixed_budget_list,
        "dynamic_target_len": int(dynamic_target_len),
        "dynamic_budget_buckets": [int(item) for item in dynamic_budget_buckets],
        "deploy_selection_ledger": bool(deploy_selection_ledger),
        "max_unselected_hole": max_unselected_hole,
        "max_p95_unselected_hole": max_p95_unselected_hole,
        "max_uniform_similarity": max_uniform_similarity,
        "required_policy_source": gas_vt.GAS_VT_CHECKPOINT_POLICY_SOURCE,
        "uses_uniform_scaffold": False,
        "uses_uniform_fill": False,
        "ledgers": ledgers,
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate strict GAS-VT value-transport ledgers.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--fixed-budgets", type=int, nargs="+", default=[384, 768])
    parser.add_argument("--dynamic-target-len", type=int, default=768)
    parser.add_argument("--dynamic-budget-buckets", type=int, nargs="+", default=list(gas_vt.DEFAULT_GAS_VT_DYNAMIC_BUDGET_BUCKETS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnostic-ledger", action="store_true")
    parser.add_argument("--disable-short-valid-ratio-count", action="store_true")
    parser.add_argument("--max-unselected-hole", type=int)
    parser.add_argument("--max-p95-unselected-hole", type=float)
    parser.add_argument("--max-uniform-similarity", type=float)
    args = parser.parse_args(argv)
    summary = run_pipeline(
        input_jsonl=args.input_jsonl,
        checkpoint_path=args.checkpoint_path,
        out_dir=args.out_dir,
        fixed_budgets=args.fixed_budgets,
        dynamic_target_len=args.dynamic_target_len,
        dynamic_budget_buckets=args.dynamic_budget_buckets,
        device=args.device,
        deploy_selection_ledger=not bool(args.diagnostic_ledger),
        allow_short_valid_ratio_count=not bool(args.disable_short_valid_ratio_count),
        max_unselected_hole=args.max_unselected_hole,
        max_p95_unselected_hole=args.max_p95_unselected_hole,
        max_uniform_similarity=args.max_uniform_similarity,
        summary_json=args.summary_json,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
