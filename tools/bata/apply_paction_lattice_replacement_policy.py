from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata import apply_paction_acquisition_policy as apply_policy
from tools.bata import paction_acquisition_policy as paction_policy
from tools.bata import paction_lattice_replacement_policy as lattice
from tools.bata import paction_source_samples


SUMMARY_SCHEMA_VERSION = "c3_paction_lattice_replacement_application_v1"
READY = "C3_PACTION_LATTICE_REPLACEMENT_APPLICATION_READY"
DEFAULT_VARIANTS = (
    lattice.MOVE50_STRATEGY,
    lattice.MOVE75_STRATEGY,
    lattice.NO_PROTECT_STRATEGY,
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: sample row must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"input JSONL has no rows: {path}")
    return rows


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _strip_deploy_invisible_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    stripped = copy.deepcopy(dict(row))
    for key in apply_policy.DEPLOY_INVISIBLE_PAYLOAD_KEYS:
        stripped.pop(key, None)
    stripped["deploy_invisible_payload_stripped"] = True
    return stripped


def _frame_value_summary(frame_values: Sequence[float]) -> dict[str, float | None]:
    if not frame_values:
        return {"min": None, "max": None, "mean": None}
    values = [float(item) for item in frame_values]
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / float(len(values)),
    }


def run_lattice_replacement_application(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    checkpoint_path: str | Path,
    summary_json: str | Path | None = None,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    fixed_budget: int = lattice.DEFAULT_BUDGET,
    device: str = "cuda",
    local_radius: int = 2,
    distance_penalty: float = 0.0,
    geometry_distortion_penalty: float = 0.0,
    max_gap_growth: int | None = None,
    strip_deploy_invisible_payload: bool = False,
    strict_deploy_source: bool = False,
) -> dict[str, Any]:
    if not variants:
        raise ValueError("at least one lattice replacement variant is required")

    rows = _read_jsonl(input_jsonl)
    checkpoint_sha256 = apply_policy._sha256_file(checkpoint_path)
    checkpoint_model, checkpoint_payload = apply_policy.load_policy_checkpoint(checkpoint_path, device=device)
    dynamic_budget_buckets = checkpoint_payload.get(
        "dynamic_budget_buckets",
        paction_policy.DEFAULT_DYNAMIC_BUDGET_BUCKETS,
    )

    enriched_rows: list[dict[str, Any]] = []
    replacement_counts_by_variant: dict[str, list[int]] = {str(name): [] for name in variants}
    selected_counts_by_variant: dict[str, list[int]] = {str(name): [] for name in variants}

    for line_no, row in enumerate(rows, start=1):
        if strict_deploy_source:
            p_action_provenance = paction_source_samples.reject_strict_deploy_source_row(
                row,
                source_name=f"{input_jsonl}:{line_no}",
                reject_payload=True,
            )
        else:
            apply_policy._reject_forbidden_source_flags(row, line_no=line_no)
            p_action_provenance = apply_policy._paction_positive_provenance(row)

        p_action = apply_policy._extract_paction(row, line_no=line_no)
        valid_len = int(row.get("valid_len") or row.get("dense_len") or len(p_action))
        valid = [idx < valid_len for idx in range(len(p_action))]
        frame_values, budget_scores = apply_policy.checkpoint_policy_scores(
            checkpoint_model,
            p_action,
            valid=valid,
            device=device,
        )

        enriched = copy.deepcopy(dict(row))
        strategies = dict(enriched.get("strategy_selected_positions") or {})
        diagnostics_by_variant: dict[str, Any] = {}
        for variant in variants:
            result = lattice.decode_paction_lattice_replacement(
                frame_values=frame_values,
                valid=valid,
                variant=str(variant),
                budget=int(fixed_budget),
                local_radius=int(local_radius),
                distance_penalty=float(distance_penalty),
                geometry_distortion_penalty=float(geometry_distortion_penalty),
                max_gap_growth=max_gap_growth,
            )
            strategies[str(variant)] = result.selected_positions
            diagnostics_by_variant[str(variant)] = result.diagnostics
            replacement_counts_by_variant[str(variant)].append(int(result.diagnostics["replaced_uniform_count"]))
            selected_counts_by_variant[str(variant)].append(int(result.diagnostics["selected_count"]))

        enriched["strategy_selected_positions"] = strategies
        enriched["paction_policy"] = {
            "source": apply_policy.CHECKPOINT_POLICY_SOURCE,
            "policy_family": "paction_score_lattice_replacement",
            "selection_signal": "p_action_gap_loss_policy_frame_value",
            "selection_decoder": "score_only_lattice_replacement_v1",
            "score_source": apply_policy.CHECKPOINT_POLICY_SOURCE,
            "score_only": True,
            "diagnostic_only": True,
            "paper_main_claim_allowed": False,
            "diagnostic_scope": "paction_lattice_replacement_policy_diagnostic_not_main_method",
            "uses_manual_boundary_slots": False,
            "uses_manual_transition_slots": False,
            "uses_manual_uncertainty_slots": False,
            "uses_manual_context_slots": False,
            "uses_uniform_scaffold": True,
            "scaffold_type": "uniform_lattice_local_replacement",
            "uses_uniform_fill": False,
            "geometry_constraint": "local_lattice_replacement",
            "geometry_lattice_budget": int(fixed_budget),
            "local_radius": int(local_radius),
            "distance_penalty": float(distance_penalty),
            "geometry_distortion_penalty": float(geometry_distortion_penalty),
            "max_gap_growth": None if max_gap_growth is None else int(max_gap_growth),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": str(checkpoint_sha256),
            "p_action_provenance": p_action_provenance,
            "dynamic_budget_buckets": [int(item) for item in dynamic_budget_buckets],
            "budget_logits": [float(item) for item in budget_scores],
            "frame_value_summary": _frame_value_summary([float(item) for item in frame_values]),
            "lattice_replacement_diagnostics_by_strategy": diagnostics_by_variant,
        }
        if strip_deploy_invisible_payload:
            enriched = _strip_deploy_invisible_payload(enriched)
        enriched_rows.append(enriched)

    _write_jsonl(output_jsonl, enriched_rows)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "row_count": len(enriched_rows),
        "variants": [str(item) for item in variants],
        "fixed_budget": int(fixed_budget),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "source": apply_policy.CHECKPOINT_POLICY_SOURCE,
        "selection_decoder": "score_only_lattice_replacement_v1",
        "score_only": True,
        "diagnostic_only": True,
        "paper_main_claim_allowed": False,
        "diagnostic_scope": "paction_lattice_replacement_policy_diagnostic_not_main_method",
        "uses_manual_slots": False,
        "uses_uniform_scaffold": True,
        "scaffold_type": "uniform_lattice_local_replacement",
        "uses_uniform_fill": False,
        "strip_deploy_invisible_payload": bool(strip_deploy_invisible_payload),
        "strict_deploy_source": bool(strict_deploy_source),
        "local_radius": int(local_radius),
        "distance_penalty": float(distance_penalty),
        "geometry_distortion_penalty": float(geometry_distortion_penalty),
        "max_gap_growth": None if max_gap_growth is None else int(max_gap_growth),
        "replacement_count_summary_by_variant": {
            key: {
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": None if not values else sum(values) / float(len(values)),
            }
            for key, values in replacement_counts_by_variant.items()
        },
        "selected_count_summary_by_variant": {
            key: {
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": None if not values else sum(values) / float(len(values)),
            }
            for key, values in selected_counts_by_variant.items()
        },
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply score-only PAction lattice replacement to C3 sample rows.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--fixed-budget", type=int, default=lattice.DEFAULT_BUDGET)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-radius", type=int, default=2)
    parser.add_argument("--distance-penalty", type=float, default=0.0)
    parser.add_argument("--geometry-distortion-penalty", type=float, default=0.0)
    parser.add_argument("--max-gap-growth", type=int)
    parser.add_argument("--strip-deploy-invisible-payload", action="store_true")
    parser.add_argument("--strict-deploy-source", action="store_true")
    args = parser.parse_args(argv)

    summary = run_lattice_replacement_application(
        args.input_jsonl,
        args.output_jsonl,
        checkpoint_path=args.checkpoint_path,
        summary_json=args.summary_json,
        variants=[str(item) for item in args.variants],
        fixed_budget=int(args.fixed_budget),
        device=str(args.device),
        local_radius=int(args.local_radius),
        distance_penalty=float(args.distance_penalty),
        geometry_distortion_penalty=float(args.geometry_distortion_penalty),
        max_gap_growth=args.max_gap_growth,
        strip_deploy_invisible_payload=bool(args.strip_deploy_invisible_payload),
        strict_deploy_source=bool(args.strict_deploy_source),
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
