from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata import apply_detector_aware_acquisition_policy as apply_policy
from tools.bata import convert_detector_aware_samples_to_value_transport_ledger as convert_ledger
from tools.bata import detector_aware_acquisition_policy as detector_policy
from tools.bata import detector_deploy_leakage
from tools.bata import paction_source_samples
from tools.bata import validate_detector_aware_policy_ledger as validate_ledger


SUMMARY_SCHEMA_VERSION = "c3_detector_aware_ledger_pipeline_v1"
READY = "C3_DETECTOR_AWARE_LEDGER_PIPELINE_READY"
DETECTOR_DEPLOY_STRIP_KEYS = (
    "teacher_utility",
    "teacher_utility_provenance",
    "frame_utility",
    "teacher_dense_points",
    "dense_teacher_points",
)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _detector_canonical_signature(row: Mapping[str, Any]) -> str:
    text = json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _detector_canonicalize_unique_sample_jsonl(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    report_json: str | Path | None = None,
    split: str = "",
) -> dict[str, Any]:
    rows = paction_source_samples._read_jsonl(input_jsonl)
    seen: dict[str, tuple[str, int]] = {}
    out_rows: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for line_no, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{input_jsonl}:{line_no}: sample row is missing sample_id")
        signature = _detector_canonical_signature(row)
        if sample_id in seen:
            old_signature, old_line = seen[sample_id]
            item = {
                "sample_id": sample_id,
                "first_line": int(old_line),
                "duplicate_line": int(line_no),
                "identical": bool(signature == old_signature),
            }
            duplicates.append(item)
            if signature == old_signature:
                continue
            raise ValueError(
                f"{input_jsonl}:{line_no}: conflicting duplicate sample_id={sample_id}; "
                "detector-aware canonicalization includes teacher utility/provenance"
            )
        seen[sample_id] = (signature, line_no)
        out_rows.append(dict(row))
    _write_jsonl(output_jsonl, out_rows)
    report = {
        "schema_version": "c3_detector_aware_source_sample_canonicalization_v1",
        "split": str(split),
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "signature_scope": "full_json_row_including_teacher_utility_when_present",
        "input_rows": len(rows),
        "output_rows": len(out_rows),
        "unique_sample_ids": len(seen),
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
    }
    if report_json is not None:
        _write_json(report_json, report)
    return report


def _detector_deploy_source_jsonl(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    report_json: str | Path | None = None,
    split: str = "",
    allow_inferred_paction_positive_provenance: bool = False,
) -> dict[str, Any]:
    rows = paction_source_samples._read_jsonl(input_jsonl)
    out_rows: list[dict[str, Any]] = []
    stripped_key_counts: dict[str, int] = {}
    inferred_count = 0
    for line_no, row in enumerate(rows, start=1):
        source_name = f"{input_jsonl}:{line_no}"
        try:
            provenance = paction_source_samples.paction_positive_provenance_from_row(
                row,
                source_name=source_name,
                strict=True,
            )
        except ValueError as exc:
            if "p_action positive provenance is required" not in str(exc) or not bool(allow_inferred_paction_positive_provenance):
                raise
            inference_row = dict(row)
            if isinstance(row.get("teacher_utility"), Mapping) and isinstance(row.get("teacher_utility_provenance"), Mapping):
                inference_row["uses_teacher"] = False
                inference_row["training_only"] = False
            provenance = paction_source_samples.infer_paction_positive_provenance_from_row(
                inference_row,
                source_name=source_name,
            )
            inferred_count += 1
        for key in paction_source_samples.SELECTION_SOURCE_FORBIDDEN_TRUE_FLAGS:
            if key in paction_source_samples.SELECTION_SOURCE_STRIPPABLE_DIAGNOSTIC_TRUE_FLAGS:
                continue
            if (
                key in {"uses_teacher", "training_only"}
                and bool(allow_inferred_paction_positive_provenance)
                and isinstance(row.get("teacher_utility"), Mapping)
                and isinstance(row.get("teacher_utility_provenance"), Mapping)
            ):
                continue
            if paction_source_samples._is_true(row.get(key, False)):
                raise ValueError(f"{source_name}: forbidden strict deploy p_action source flag {key}=true")
        stripped = dict(row)
        if (
            bool(allow_inferred_paction_positive_provenance)
            and isinstance(row.get("teacher_utility"), Mapping)
            and isinstance(row.get("teacher_utility_provenance"), Mapping)
        ):
            stripped["uses_teacher"] = False
            stripped["training_only"] = False
        for key in tuple(paction_source_samples.SELECTION_SOURCE_STRIP_KEYS) + DETECTOR_DEPLOY_STRIP_KEYS:
            if key in stripped:
                stripped_key_counts[key] = stripped_key_counts.get(key, 0) + 1
                stripped.pop(key, None)
        stripped = detector_deploy_leakage.strip_detector_deploy_forbidden_payloads(stripped)
        stripped["paction_positive_provenance"] = provenance
        stripped["deploy_selection_source_stripped"] = True
        stripped["detector_teacher_payload_stripped"] = True
        detector_deploy_leakage.reject_detector_deploy_forbidden_payloads(
            stripped,
            source_name=f"{source_name}:detector_selection_deploy_source",
        )
        paction_source_samples.reject_strict_deploy_source_row(
            stripped,
            source_name=f"{source_name}:detector_selection_deploy_source",
            reject_payload=True,
        )
        for key in DETECTOR_DEPLOY_STRIP_KEYS:
            if key in stripped:
                raise ValueError(f"{source_name}: detector deploy source still contains {key}")
        out_rows.append(stripped)
    _write_jsonl(output_jsonl, out_rows)
    report = {
        "schema_version": "c3_detector_aware_deploy_selection_source_v1",
        "split": str(split),
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "input_rows": len(rows),
        "output_rows": len(out_rows),
        "stripped_key_counts": stripped_key_counts,
        "requires_paction_positive_provenance": True,
        "allow_inferred_paction_positive_provenance": bool(allow_inferred_paction_positive_provenance),
        "inferred_paction_positive_provenance_count": int(inferred_count),
        "teacher_payload_visible_to_deploy": False,
    }
    if report_json is not None:
        _write_json(report_json, report)
    return report


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
        route_variant=f"c3_detector_aware_{name}",
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
        require_policy_source=detector_policy.DETECTOR_AWARE_CHECKPOINT_POLICY_SOURCE,
        require_checkpoint_path=checkpoint_path,
        require_checkpoint_sha256=checkpoint_sha256,
        require_paction_provenance=bool(deploy_selection_ledger),
        max_unselected_hole=max_unselected_hole,
        max_p95_unselected_hole=max_p95_unselected_hole,
        max_uniform_similarity=max_uniform_similarity,
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


def _effective_decoder_max_unselected_hole(
    max_unselected_hole: int | None,
    max_p95_unselected_hole: float | None,
) -> int | None:
    candidates: list[int] = []
    if max_unselected_hole is not None:
        candidates.append(int(max_unselected_hole))
    if max_p95_unselected_hole is not None:
        candidates.append(int(float(max_p95_unselected_hole) // 1))
    if not candidates:
        return None
    effective = min(candidates)
    if effective < 0:
        raise ValueError("hole thresholds must be non-negative")
    return int(effective)


def run_pipeline(
    *,
    input_jsonl: str | Path,
    checkpoint_path: str | Path,
    out_dir: str | Path,
    fixed_budgets: Sequence[int] = (384, 768),
    dynamic_target_len: int = 768,
    dynamic_budget_buckets: Sequence[int] = detector_policy.DEFAULT_DETECTOR_AWARE_DYNAMIC_BUDGET_BUCKETS,
    device: str = "cuda",
    deploy_selection_ledger: bool = True,
    allow_short_valid_ratio_count: bool = False,
    max_unselected_hole: int | None = None,
    max_p95_unselected_hole: float | None = None,
    max_uniform_similarity: float | None = None,
    allow_tiny_dynamic_diagnostic: bool = False,
    allow_inferred_paction_positive_provenance: bool = False,
    require_point_responsibility_utility: bool = False,
    summary_json: str | Path | None = None,
) -> dict[str, Any]:
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)
    canonical_input_jsonl = out_path / "source.canonical_unique.jsonl"
    source_canonicalization = _detector_canonicalize_unique_sample_jsonl(
        input_jsonl,
        canonical_input_jsonl,
        report_json=out_path / "source.canonical_unique.report.json",
        split="",
    )
    if allow_tiny_dynamic_diagnostic:
        rows = paction_source_samples._read_jsonl(canonical_input_jsonl)
        max_len = max(int(row.get("valid_len") or row.get("dense_len") or 0) for row in rows)
        if len(rows) > 2 or max_len > 16:
            raise ValueError("allow_tiny_dynamic_diagnostic is only allowed for <=2 rows with valid_len<=16")
    selection_input_jsonl: Path | None = None
    selection_source_report: dict[str, Any] | None = None
    if deploy_selection_ledger:
        selection_input_jsonl = out_path / "source.selection_deploy.detector_clean.jsonl"
        selection_source_report = _detector_deploy_source_jsonl(
            canonical_input_jsonl,
            selection_input_jsonl,
            report_json=out_path / "source.selection_deploy.detector_clean.report.json",
            split="",
            allow_inferred_paction_positive_provenance=bool(allow_inferred_paction_positive_provenance),
        )
    input_sample_path = selection_input_jsonl if selection_input_jsonl is not None else canonical_input_jsonl
    metric_sample_path = canonical_input_jsonl
    checkpoint_sha256 = apply_policy._sha256_file(checkpoint_path)
    applied_sample_jsonl = out_path / "samples.detector_aware_all.jsonl"
    decoder_max_unselected_hole = _effective_decoder_max_unselected_hole(
        max_unselected_hole,
        max_p95_unselected_hole,
    )
    apply_policy.run_policy_application(
        input_sample_path,
        applied_sample_jsonl,
        summary_json=out_path / "samples.detector_aware_all.summary.json",
        fixed_budgets=[int(item) for item in fixed_budgets],
        dynamic_budget_buckets=[int(item) for item in dynamic_budget_buckets],
        checkpoint_path=checkpoint_path,
        device=device,
        strip_deploy_invisible_payload=True,
        strict_deploy_source=bool(deploy_selection_ledger),
        max_unselected_hole=decoder_max_unselected_hole,
        source_jsonl_for_hash=input_sample_path,
        require_point_responsibility_utility=bool(require_point_responsibility_utility),
    )
    fixed_budget_list = [int(item) for item in fixed_budgets]
    variant_specs = {
        "detector_aware_fixed_384": dict(
            strategy=detector_policy.DETECTOR_AWARE_FIXED_384_STRATEGY,
            target_len=fixed_budget_list[0],
            require_selected_count=fixed_budget_list[0],
            require_nonconstant_selected_count=False,
        ),
        "detector_aware_fixed_768": dict(
            strategy=detector_policy.DETECTOR_AWARE_FIXED_768_STRATEGY,
            target_len=fixed_budget_list[1] if len(fixed_budget_list) > 1 else fixed_budget_list[0],
            require_selected_count=fixed_budget_list[1] if len(fixed_budget_list) > 1 else fixed_budget_list[0],
            require_nonconstant_selected_count=False,
        ),
        "detector_aware_dynamic": dict(
            strategy=detector_policy.DETECTOR_AWARE_DYNAMIC_STRATEGY,
            target_len=int(dynamic_target_len),
            require_selected_count=None,
            require_nonconstant_selected_count=not bool(allow_tiny_dynamic_diagnostic),
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
            max_unselected_hole=max_unselected_hole,
            max_p95_unselected_hole=max_p95_unselected_hole,
            max_uniform_similarity=max_uniform_similarity,
        )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "stage_label": detector_policy.STAGE_LABEL,
        "route_label": "DIVERGENT_INNOVATION_DETECTOR_AWARE_UTILITY_DO_NOT_MERGE_WITH_C3",
        "question": "Can dense AdaTAD teacher utility train an acquisition policy better than p_action-only?",
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
        "dynamic_gain_calibration": dict(detector_policy.DEFAULT_DYNAMIC_GAIN_CALIBRATION),
        "decoder_effective_max_unselected_hole": decoder_max_unselected_hole,
        "validator_max_unselected_hole": None if max_unselected_hole is None else int(max_unselected_hole),
        "validator_max_p95_unselected_hole": None if max_p95_unselected_hole is None else float(max_p95_unselected_hole),
        "deploy_selection_ledger": bool(deploy_selection_ledger),
        "allow_inferred_paction_positive_provenance": bool(allow_inferred_paction_positive_provenance),
        "require_point_responsibility_utility": bool(require_point_responsibility_utility),
        "required_policy_source": detector_policy.DETECTOR_AWARE_CHECKPOINT_POLICY_SOURCE,
        "baseline_comparison": {
            "matched_budget_baselines": ["p_action_only", "GAS-VT"],
            "required_variants": ["fixed_384", "fixed_768", "dynamic"],
            "decision_metrics": [
                "detector_utility_coverage",
                "detector_utility_ndcg",
                "boundary_bracket_support",
                "action_interior_bin_coverage",
                "max_unselected_hole",
                "p95_unselected_hole",
                "mean_uniform_similarity",
                "AdaTAD_mAP_after_full_train",
            ],
        },
        "full_detector_map_required_for_claim": True,
        "adatad_map": None,
        "map_claim_allowed": False,
        "end_to_end": False,
        "uses_uniform_scaffold": False,
        "uses_uniform_fill": False,
        "dynamic_budget_diagnostic_allow_constant_tiny": bool(allow_tiny_dynamic_diagnostic),
        "ledgers": ledgers,
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate strict Stage-2 detector-aware fixed/dynamic value-transport ledgers.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--fixed-budgets", type=int, nargs="+", default=[384, 768])
    parser.add_argument("--dynamic-target-len", type=int, default=768)
    parser.add_argument("--dynamic-budget-buckets", type=int, nargs="+", default=list(detector_policy.DEFAULT_DETECTOR_AWARE_DYNAMIC_BUDGET_BUCKETS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnostic-ledger", action="store_true")
    parser.add_argument("--allow-short-valid-ratio-count", action="store_true")
    parser.add_argument("--disable-short-valid-ratio-count", action="store_true")
    parser.add_argument("--max-unselected-hole", type=int)
    parser.add_argument("--max-p95-unselected-hole", type=float)
    parser.add_argument("--max-uniform-similarity", type=float)
    parser.add_argument("--allow-tiny-dynamic-diagnostic", action="store_true")
    parser.add_argument("--allow-inferred-paction-positive-provenance", action="store_true")
    parser.add_argument("--require-point-responsibility-utility", action="store_true")
    args = parser.parse_args(argv)
    allow_short_valid_ratio_count = bool(args.allow_short_valid_ratio_count)
    if bool(args.disable_short_valid_ratio_count):
        allow_short_valid_ratio_count = False
    summary = run_pipeline(
        input_jsonl=args.input_jsonl,
        checkpoint_path=args.checkpoint_path,
        out_dir=args.out_dir,
        fixed_budgets=args.fixed_budgets,
        dynamic_target_len=args.dynamic_target_len,
        dynamic_budget_buckets=args.dynamic_budget_buckets,
        device=args.device,
        deploy_selection_ledger=not bool(args.diagnostic_ledger),
        allow_short_valid_ratio_count=allow_short_valid_ratio_count,
        max_unselected_hole=args.max_unselected_hole,
        max_p95_unselected_hole=args.max_p95_unselected_hole,
        max_uniform_similarity=args.max_uniform_similarity,
        allow_tiny_dynamic_diagnostic=bool(args.allow_tiny_dynamic_diagnostic),
        allow_inferred_paction_positive_provenance=bool(args.allow_inferred_paction_positive_provenance),
        require_point_responsibility_utility=bool(args.require_point_responsibility_utility),
        summary_json=args.summary_json,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
