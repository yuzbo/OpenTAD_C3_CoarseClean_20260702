#!/usr/bin/env python3
"""Render auditable GeoRoute paper tables from validated result records.

The renderer intentionally writes only to a caller-supplied output directory.
It never pools away raw seed rows and it does not decide whether a result is a
paper claim. The paired/statistical decision remains in the separately frozen
analysis protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from georoute_result_schema import (
    GeoRouteResultSchemaError,
    canonical_json_sha256,
    load_records,
    validate_records,
)


MATCHED_CONTROLS = (
    "fixed_lattice",
    "fixed_lattice_geometry",
    "random",
    "free_token_select",
    "roi_only",
    "roi_residual",
)


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) <= 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, variance**0.5


def _format_mean_std(values: Sequence[float], digits: int = 2) -> str:
    mean, std = _mean_std(values)
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def _latex_escape(value: Any) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _raw_row(record: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = record["metrics"]
    cost = record["cost"]
    return {
        "stage": record["stage"],
        "split_role": record["split_role"],
        "dataset": record["dataset"],
        "detector": record["detector"],
        "variant": record["variant"],
        "tokens_per_tubelet": record["budget"]["tokens_per_tubelet"],
        "seed": record["seed"],
        "average_map": float(metrics["average_map"]),
        "map_03": float(metrics["map_by_tiou"]["0.3"]),
        "map_04": float(metrics["map_by_tiou"]["0.4"]),
        "map_05": float(metrics["map_by_tiou"]["0.5"]),
        "map_06": float(metrics["map_by_tiou"]["0.6"]),
        "map_07": float(metrics["map_by_tiou"]["0.7"]),
        "p50_ms": float(cost["end_to_end_p50_ms"]),
        "p95_ms": float(cost["end_to_end_p95_ms"]),
        "peak_memory_mb": float(cost["peak_memory_mb"]),
        "gross_gpu_energy_j": float(cost["gross_gpu_energy_j"]),
        "runtime_commit": record["evidence"]["runtime_commit"],
        "run_receipt_sha256": record["evidence"]["run_receipt_sha256"],
    }


def raw_seed_rows(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = [_raw_row(record) for record in records]
    return sorted(
        rows,
        key=lambda row: (
            row["stage"],
            row["split_role"],
            row["dataset"],
            row["detector"],
            row["tokens_per_tubelet"],
            row["variant"],
            row["seed"],
        ),
    )


def summary_rows(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_seed_rows(records):
        key = (
            row["stage"],
            row["split_role"],
            row["dataset"],
            row["detector"],
            row["variant"],
            row["tokens_per_tubelet"],
        )
        grouped[key].append(row)
    output: List[Dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        stage, split_role, dataset, detector, variant, budget = key
        output.append(
            {
                "stage": stage,
                "split_role": split_role,
                "dataset": dataset,
                "detector": detector,
                "variant": variant,
                "tokens_per_tubelet": budget,
                "n": len(rows),
                "average_map": _format_mean_std([row["average_map"] for row in rows]),
                "map_05": _format_mean_std([row["map_05"] for row in rows]),
                "map_06": _format_mean_std([row["map_06"] for row in rows]),
                "map_07": _format_mean_std([row["map_07"] for row in rows]),
                "p50_ms": _format_mean_std([row["p50_ms"] for row in rows]),
                "p95_ms": _format_mean_std([row["p95_ms"] for row in rows]),
                "peak_memory_mb": _format_mean_std(
                    [row["peak_memory_mb"] for row in rows], digits=1
                ),
                "gross_gpu_energy_j": _format_mean_std(
                    [row["gross_gpu_energy_j"] for row in rows]
                ),
            }
        )
    return output


def _write_raw_csv(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    fieldnames = [
        "stage",
        "split_role",
        "dataset",
        "detector",
        "variant",
        "tokens_per_tubelet",
        "seed",
        "average_map",
        "map_03",
        "map_04",
        "map_05",
        "map_06",
        "map_07",
        "p50_ms",
        "p95_ms",
        "peak_memory_mb",
        "gross_gpu_energy_j",
        "runtime_commit",
        "run_receipt_sha256",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_markdown(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    headers = (
        "Variant",
        "K",
        "n",
        "Avg-mAP",
        "mAP@0.5",
        "mAP@0.6",
        "mAP@0.7",
        "p50 ms",
        "p95 ms",
        "Peak MB",
        "Energy J",
    )
    lines = [
        "# GeoRoute result summary",
        "",
        "All values are arithmetic mean +/- sample standard deviation. Raw seed rows are in "
        "`georoute_raw_seed_table.csv`; this table is descriptive and is not a significance test.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = (
            row["variant"],
            row["tokens_per_tubelet"],
            row["n"],
            row["average_map"],
            row["map_05"],
            row["map_06"],
            row["map_07"],
            row["p50_ms"],
            row["p95_ms"],
            row["peak_memory_mb"],
            row["gross_gpu_energy_j"],
        )
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_latex(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    lines = [
        "% Generated from validated GeoRoute records; do not edit values manually.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Development or sealed-test evidence under the recorded fixed token budget. "
        r"Each entry is mean $\pm$ sample standard deviation across raw seed rows.}",
        r"\label{tab:georoute-summary}",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & $K$ & $n$ & Avg-mAP & mAP@0.5 & mAP@0.6 & mAP@0.7 & p50 (ms) \\",
        r"\midrule",
    ]
    for row in rows:
        cells = (
            _latex_escape(row["variant"]),
            row["tokens_per_tubelet"],
            row["n"],
            row["average_map"].replace("+/-", r"$\pm$"),
            row["map_05"].replace("+/-", r"$\pm$"),
            row["map_06"].replace("+/-", r"$\pm$"),
            row["map_07"].replace("+/-", r"$\pm$"),
            row["p50_ms"].replace("+/-", r"$\pm$"),
        )
        lines.append(" & ".join(str(cell) for cell in cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def _write_claim_audit(
    records: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
    output: Path,
) -> None:
    groups: Dict[Tuple[Any, ...], set[str]] = defaultdict(set)
    for record in records:
        groups[
            (
                record["stage"],
                record["split_role"],
                record["dataset"],
                record["detector"],
                record["budget"]["tokens_per_tubelet"],
            )
        ].add(record["variant"])
    lines = [
        "# GeoRoute evidence inventory",
        "",
        "This is a coverage audit, not an automated accept/reject decision and not a paper claim.",
        f"- Validated record hash: `{canonical_json_sha256(records)}`",
        f"- Bound analysis hash: `{analysis.get('analysis_sha256', 'missing')}`",
        "",
        "## Matched-control coverage",
    ]
    for key, variants in sorted(groups.items()):
        missing = [control for control in MATCHED_CONTROLS if control not in variants]
        label = " / ".join(str(value) for value in key)
        if missing:
            lines.append(f"- `{label}`: incomplete; missing {', '.join(missing)}.")
        else:
            lines.append(f"- `{label}`: all primary matched controls present.")
    lines.extend(
        [
            "",
            "## Claim guard",
            "",
            "- Do not infer an mAP theorem from these tables.",
            "- Do not infer causal ROI benefit without the matched free-token and "
            "fixed-lattice-plus-geometry controls.",
            "- Do not call a cost result end-to-end unless every charged scope flag "
            "is present in each raw record.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_tables(
    records: Iterable[Mapping[str, Any]], analysis: Mapping[str, Any], output_dir: Path
) -> List[Path]:
    validated = validate_records(records, development_only=bool(analysis.get("development_only")))
    if analysis.get("input_records_sha256") != canonical_json_sha256(validated):
        raise GeoRouteResultSchemaError(
            "analysis JSON does not bind the supplied validated result records; "
            "rerun analyze_georoute_results.py"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = raw_seed_rows(validated)
    summaries = summary_rows(validated)
    raw_path = output_dir / "georoute_raw_seed_table.csv"
    markdown_path = output_dir / "georoute_summary_table.md"
    latex_path = output_dir / "georoute_summary_table.tex"
    audit_path = output_dir / "georoute_evidence_inventory.md"
    _write_raw_csv(rows, raw_path)
    _write_summary_markdown(summaries, markdown_path)
    _write_summary_latex(summaries, latex_path)
    _write_claim_audit(validated, analysis, audit_path)
    return [raw_path, markdown_path, latex_path, audit_path]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        records = load_records(args.records)
        analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
        outputs = render_tables(records, analysis, args.output_dir)
    except (GeoRouteResultSchemaError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"written": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
