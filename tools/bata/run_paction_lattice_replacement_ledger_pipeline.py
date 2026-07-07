from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata import apply_paction_acquisition_policy as apply_policy
from tools.bata import apply_paction_lattice_replacement_policy as apply_lattice
from tools.bata import convert_lowres_probe_samples_to_value_transport_ledger as convert_ledger
from tools.bata import paction_lattice_replacement_policy as lattice
from tools.bata import paction_source_samples
from tools.bata import validate_paction_lattice_replacement_ledger as validate_lattice


SUMMARY_SCHEMA_VERSION = "c3_paction_lattice_replacement_ledger_pipeline_v1"
READY = "C3_PACTION_LATTICE_REPLACEMENT_LEDGER_PIPELINE_READY"
SCAFFOLD_TYPE = "uniform_lattice_local_replacement"


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    return rows


def _numeric_summary(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    numeric = [float(item) for item in values]
    return {
        "min": min(numeric),
        "max": max(numeric),
        "mean": sum(numeric) / float(len(numeric)),
    }


def _lattice_scaffold_summary_by_variant(
    sample_jsonl: str | Path,
    variants: Sequence[str],
) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(sample_jsonl)
    out: dict[str, dict[str, Any]] = {}
    metric_keys = (
        "protected_uniform_count",
        "replaceable_uniform_count",
        "inserted_candidate_count",
        "replaced_uniform_count",
        "selected_count",
        "base_uniform_jaccard",
        "phase_shift_uniform_similarity",
        "paction_topk_overlap",
    )
    for variant in variants:
        values_by_key: dict[str, list[float]] = {key: [] for key in metric_keys}
        for row_index, row in enumerate(rows, start=1):
            policy = row.get("paction_policy")
            if not isinstance(policy, Mapping):
                raise ValueError(f"{sample_jsonl}: row {row_index}: paction_policy metadata is required")
            diagnostics_by_strategy = policy.get("lattice_replacement_diagnostics_by_strategy")
            if not isinstance(diagnostics_by_strategy, Mapping) or str(variant) not in diagnostics_by_strategy:
                raise ValueError(f"{sample_jsonl}: row {row_index}: missing lattice diagnostics for {variant}")
            diagnostics = diagnostics_by_strategy[str(variant)]
            if not isinstance(diagnostics, Mapping):
                raise ValueError(f"{sample_jsonl}: row {row_index}: lattice diagnostics for {variant} must be an object")
            for key in metric_keys:
                if key in diagnostics and diagnostics[key] is not None:
                    values_by_key[key].append(float(diagnostics[key]))
        out[str(variant)] = {
            "uses_uniform_scaffold": True,
            "scaffold_type": SCAFFOLD_TYPE,
            "uses_uniform_fill": False,
            **{key: _numeric_summary(values) for key, values in values_by_key.items()},
            "uniform_jaccard_before_after": _numeric_summary(values_by_key["base_uniform_jaccard"]),
        }
    return out


def _formal_provenance_mode(*, deploy_selection_ledger: bool, allow_inferred: bool) -> str:
    if not deploy_selection_ledger:
        return "diagnostic_ledger"
    if allow_inferred:
        return "inferred_paction_positive_provenance_explicit_opt_in"
    return "formal_explicit_paction_positive_provenance"


def run_pipeline(
    *,
    input_jsonl: str | Path,
    checkpoint_path: str | Path,
    out_dir: str | Path,
    variants: Sequence[str] = apply_lattice.DEFAULT_VARIANTS,
    fixed_budget: int = lattice.DEFAULT_BUDGET,
    device: str = "cuda",
    deploy_selection_ledger: bool = False,
    allow_inferred_paction_positive_provenance: bool = False,
    allow_short_valid_ratio_count: bool = True,
    local_radius: int = 2,
    distance_penalty: float = 0.0,
    geometry_distortion_penalty: float = 0.0,
    max_gap_growth: int | None = None,
    max_max_gap: int | None = None,
    max_p95_gap: float | None = None,
    max_unselected_hole: int | None = None,
    max_p95_unselected_hole: float | None = None,
    max_uniform_similarity: float | None = None,
    summary_json: str | Path | None = None,
) -> dict[str, Any]:
    if not variants:
        raise ValueError("at least one lattice replacement variant is required")
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
            allow_inferred_paction_positive_provenance=bool(allow_inferred_paction_positive_provenance),
        )
    input_sample_path = selection_input_jsonl if selection_input_jsonl is not None else canonical_input_jsonl
    metric_sample_path = canonical_input_jsonl
    checkpoint_sha256 = apply_policy._sha256_file(checkpoint_path)

    sample_jsonl = out_path / "samples.paction_lattice_replacement.jsonl"
    apply_summary_json = out_path / "samples.paction_lattice_replacement.summary.json"
    apply_summary = apply_lattice.run_lattice_replacement_application(
        input_sample_path,
        sample_jsonl,
        checkpoint_path=checkpoint_path,
        summary_json=apply_summary_json,
        variants=[str(item) for item in variants],
        fixed_budget=int(fixed_budget),
        device=device,
        local_radius=int(local_radius),
        distance_penalty=float(distance_penalty),
        geometry_distortion_penalty=float(geometry_distortion_penalty),
        max_gap_growth=max_gap_growth,
        allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
        strip_deploy_invisible_payload=bool(deploy_selection_ledger),
        strict_deploy_source=bool(deploy_selection_ledger),
    )
    lattice_scaffold_summary = _lattice_scaffold_summary_by_variant(
        sample_jsonl,
        [str(item) for item in variants],
    )
    apply_summary_for_pipeline = dict(apply_summary)
    apply_summary_for_pipeline.update(
        {
            "diagnostic_only": True,
            "paper_main_claim_allowed": False,
            "diagnostic_scope": "paction_lattice_replacement_policy_diagnostic_not_main_method",
            "uses_uniform_scaffold": True,
            "scaffold_type": SCAFFOLD_TYPE,
            "uses_uniform_fill": False,
            "lattice_scaffold_summary_by_variant": lattice_scaffold_summary,
        }
    )
    _write_json(apply_summary_json, apply_summary_for_pipeline)

    ledgers: dict[str, Any] = {}
    for variant in variants:
        name = str(variant)
        ledger_jsonl = out_path / f"value_transport_ledger_{name}.jsonl"
        ledger_summary_json = out_path / f"value_transport_ledger_{name}.summary.json"
        validation_summary_json = out_path / f"value_transport_ledger_{name}.validation.json"
        ledger_summary = convert_ledger.run_conversion(
            sample_jsonl,
            ledger_jsonl,
            strategy=name,
            target_len=int(fixed_budget),
            summary_json=ledger_summary_json,
            require_selected_count=int(fixed_budget),
            allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
            deploy_selection_ledger=bool(deploy_selection_ledger),
            deduplicate_sample_id=True,
            route_variant=f"c3_paction_lattice_replacement_{name}",
        )
        ledger_rows = _read_jsonl(ledger_jsonl)
        for row_index, row in enumerate(ledger_rows, start=1):
            row["uses_uniform_scaffold"] = True
            row["scaffold_type"] = SCAFFOLD_TYPE
            row["paper_main_claim_allowed"] = False
            row["diagnostic_scope"] = "paction_lattice_replacement_policy_diagnostic_not_main_method"
            if not deploy_selection_ledger:
                row["diagnostic_only"] = True
                row["deploy_selection_ledger"] = False
            if row.get("diagnostic_only") is not True:
                row["diagnostic_boundary"] = "diagnostic_experiment_ledger_not_paper_main_claim"
            ledger_rows[row_index - 1] = row
        convert_ledger._write_jsonl(ledger_jsonl, ledger_rows)
        ledger_summary = dict(ledger_summary)
        ledger_summary.update(
            {
                "uses_uniform_scaffold": True,
                "scaffold_type": SCAFFOLD_TYPE,
                "diagnostic_only": True,
                "paper_main_claim_allowed": False,
                "diagnostic_scope": "paction_lattice_replacement_policy_diagnostic_not_main_method",
            }
        )
        convert_ledger.write_json(ledger_summary_json, ledger_summary)
        validation_summary = validate_lattice.validate_lattice_ledger(
            sample_jsonl=sample_jsonl,
            metric_sample_jsonl=metric_sample_path,
            ledger_jsonl=ledger_jsonl,
            strategy=name,
            expected_target_len=int(fixed_budget),
            require_selected_count=int(fixed_budget),
            allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
            require_deployable=bool(deploy_selection_ledger),
            require_checkpoint_path=checkpoint_path,
            require_checkpoint_sha256=checkpoint_sha256,
            max_max_gap=max_max_gap,
            max_p95_gap=max_p95_gap,
            max_unselected_hole=max_unselected_hole,
            max_p95_unselected_hole=max_p95_unselected_hole,
            max_uniform_similarity=max_uniform_similarity,
            summary_json=validation_summary_json,
        )
        ledgers[name] = {
            "ledger_jsonl": str(ledger_jsonl),
            "ledger_summary_json": str(ledger_summary_json),
            "validation_summary_json": str(validation_summary_json),
            "ledger_summary": ledger_summary,
            "validation_summary": validation_summary,
        }

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "input_jsonl": str(input_jsonl),
        "canonical_input_jsonl": str(canonical_input_jsonl),
        "source_canonicalization": source_canonicalization,
        "selection_sample_jsonl": None if selection_input_jsonl is None else str(selection_input_jsonl),
        "selection_source_report": selection_source_report,
        "metric_sample_jsonl": str(metric_sample_path),
        "sample_jsonl": str(sample_jsonl),
        "apply_summary_json": str(apply_summary_json),
        "apply_summary": apply_summary_for_pipeline,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "out_dir": str(out_path),
        "variants": [str(item) for item in variants],
        "fixed_budget": int(fixed_budget),
        "deploy_selection_ledger": bool(deploy_selection_ledger),
        "diagnostic_only": True,
        "paper_main_claim_allowed": False,
        "diagnostic_scope": "paction_lattice_replacement_policy_diagnostic_not_main_method",
        "allow_inferred_paction_positive_provenance": bool(allow_inferred_paction_positive_provenance),
        "formal_provenance_mode": _formal_provenance_mode(
            deploy_selection_ledger=bool(deploy_selection_ledger),
            allow_inferred=bool(allow_inferred_paction_positive_provenance),
        ),
        "requires_explicit_paction_positive_provenance": bool(
            deploy_selection_ledger and not allow_inferred_paction_positive_provenance
        ),
        "inferred_paction_positive_provenance_scope": (
            "diagnostic_or_migration_explicit_opt_in"
            if allow_inferred_paction_positive_provenance
            else "disabled_for_formal_deploy"
        ),
        "selection_decoder": "score_only_lattice_replacement_v1",
        "score_only": True,
        "uses_manual_slots": False,
        "uses_uniform_scaffold": True,
        "scaffold_type": SCAFFOLD_TYPE,
        "uses_uniform_fill": False,
        "lattice_scaffold_summary_by_variant": lattice_scaffold_summary,
        "required_policy_source": apply_policy.CHECKPOINT_POLICY_SOURCE,
        "ledgers": ledgers,
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate score-only PAction lattice replacement ledgers.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--variants", nargs="+", default=list(apply_lattice.DEFAULT_VARIANTS))
    parser.add_argument("--fixed-budget", type=int, default=lattice.DEFAULT_BUDGET)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnostic-ledger", action="store_true", default=True)
    parser.add_argument(
        "--deploy-selection-ledger",
        action="store_false",
        dest="diagnostic_ledger",
        help="Compatibility escape hatch for legacy deployability checks; lattice remains diagnostic-only metadata.",
    )
    parser.add_argument("--allow-inferred-paction-positive-provenance", action="store_true")
    parser.add_argument(
        "--no-allow-inferred-paction-positive-provenance",
        action="store_false",
        dest="allow_inferred_paction_positive_provenance",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-allow-short-valid-ratio-count", action="store_true")
    parser.add_argument("--local-radius", type=int, default=2)
    parser.add_argument("--distance-penalty", type=float, default=0.0)
    parser.add_argument("--geometry-distortion-penalty", type=float, default=0.0)
    parser.add_argument("--max-gap-growth", type=int)
    parser.add_argument("--max-max-gap", type=int)
    parser.add_argument("--max-p95-gap", type=float)
    parser.add_argument("--max-unselected-hole", type=int)
    parser.add_argument("--max-p95-unselected-hole", type=float)
    parser.add_argument("--max-uniform-similarity", type=float)
    args = parser.parse_args(argv)

    summary = run_pipeline(
        input_jsonl=args.input_jsonl,
        checkpoint_path=args.checkpoint_path,
        out_dir=args.out_dir,
        variants=[str(item) for item in args.variants],
        fixed_budget=int(args.fixed_budget),
        device=str(args.device),
        deploy_selection_ledger=not bool(args.diagnostic_ledger),
        allow_inferred_paction_positive_provenance=bool(args.allow_inferred_paction_positive_provenance),
        allow_short_valid_ratio_count=not bool(args.no_allow_short_valid_ratio_count),
        local_radius=int(args.local_radius),
        distance_penalty=float(args.distance_penalty),
        geometry_distortion_penalty=float(args.geometry_distortion_penalty),
        max_gap_growth=args.max_gap_growth,
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
