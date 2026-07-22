from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from tools.bata.aggregate_duca_r5_paper_matrix import _canonical_sha256
from tools.bata.plot_duca_r5_performance_cost import IOU_KEYS, _validate_aggregate


SCHEMA = "duca_official_budget_curve_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"aggregate is not an object: {resolved}")
    return payload, resolved


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty budget curve")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_budget_curve(
    *,
    aggregate_jsons: Sequence[str | Path],
    expected_budgets: Sequence[int] = (384, 320, 256, 192, 128),
) -> dict[str, Any]:
    expected = tuple(int(value) for value in expected_budgets)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected budgets must be non-empty and unique")
    sources: list[dict[str, Any]] = []
    rows_by_key: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for aggregate_json in aggregate_jsons:
        payload, path = _load(aggregate_json)
        rows = _validate_aggregate(payload)
        source_budgets = sorted({int(row["K"]) for row in rows}, reverse=True)
        sources.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "git_commit": str(payload.get("git_commit", "")),
                "budgets": source_budgets,
            }
        )
        for row in rows:
            key = (
                str(row["backend"]),
                str(row["arm"]),
                int(row["K"]),
                int(row["seed"]),
            )
            if key in rows_by_key:
                raise RuntimeError(f"duplicate official budget-curve cell: {key}")
            rows_by_key[key] = dict(row)
    observed_budgets = {key[2] for key in rows_by_key}
    if observed_budgets != set(expected):
        raise RuntimeError(
            f"budget curve mismatch: expected={sorted(expected)}, observed={sorted(observed_budgets)}"
        )
    backends = {key[0] for key in rows_by_key}
    arms = {key[1] for key in rows_by_key}
    seeds = {key[3] for key in rows_by_key}
    expected_keys = {
        (backend, arm, budget, seed)
        for backend in backends
        for arm in arms
        for budget in expected
        for seed in seeds
    }
    if set(rows_by_key) != expected_keys:
        missing = sorted(expected_keys - set(rows_by_key))
        raise RuntimeError(f"budget curve is not a complete Cartesian matrix; missing={missing[:5]}")

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for key, row in rows_by_key.items():
        grouped[key[:3]].append(row)
    summary_rows: list[dict[str, Any]] = []
    metric_names = ("Avg-mAP", *IOU_KEYS)
    for (backend, arm, budget), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], -item[0][2])
    ):
        seeds_in_group = sorted(int(row["seed"]) for row in group)
        record: dict[str, Any] = {
            "backend": backend,
            "arm": arm,
            "K": budget,
            "frame_fraction": budget / 768.0,
            "seed_count": len(group),
            "seeds": ",".join(str(seed) for seed in seeds_in_group),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in group]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"non-finite {metric} in {(backend, arm, budget)}")
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(record)
    raw_rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    payload = {
        "schema": SCHEMA,
        "ok": True,
        "task": "offline_temporal_action_detection",
        "protocol": "full_official_validation_terminal_epoch59_ema",
        "expected_budgets": list(expected),
        "sources": sources,
        "raw_rows": raw_rows,
        "summary_rows": summary_rows,
        "paper_claim_allowed": False,
        "status": "official_budget_curve_evidence_complete_pending_claim_adjudication",
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def _plot(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"unavailable: {exc}"
    markers = {"uniform": "o", "learned": "s"}
    colors = {"actionformer": "#1769aa", "temporalmaxer": "#b54832"}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    for backend in sorted({str(row["backend"]) for row in rows}):
        for arm in sorted({str(row["arm"]) for row in rows}):
            group = sorted(
                (row for row in rows if row["backend"] == backend and row["arm"] == arm),
                key=lambda row: int(row["K"]),
            )
            label = f"{backend}-{arm}"
            style = "-" if arm == "learned" else "--"
            for axis, metric in zip(axes, ("Avg-mAP", "mAP@0.7")):
                axis.errorbar(
                    [int(row["K"]) for row in group],
                    [float(row[f"{metric}_mean"]) for row in group],
                    yerr=[float(row[f"{metric}_std"]) for row in group],
                    color=colors[backend],
                    marker=markers[arm],
                    linestyle=style,
                    capsize=3,
                    label=label,
                )
                axis.set_xlabel("selected frames K")
                axis.set_ylabel(metric)
                axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"duca_official_budget_curve.{suffix}", dpi=300)
    plt.close(fig)
    return "generated"


def export_budget_curve(
    *,
    aggregate_jsons: Sequence[str | Path],
    output_dir: str | Path,
    expected_budgets: Sequence[int] = (384, 320, 256, 192, 128),
) -> dict[str, Any]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    payload = build_budget_curve(
        aggregate_jsons=aggregate_jsons,
        expected_budgets=expected_budgets,
    )
    _write_csv(out / "duca_official_budget_curve_raw.csv", payload["raw_rows"])
    _write_csv(out / "duca_official_budget_curve_summary.csv", payload["summary_rows"])
    payload["plot_status"] = _plot(payload["summary_rows"], out)
    payload.pop("result_sha256", None)
    payload["result_sha256"] = _canonical_sha256(payload)
    (out / "duca_official_budget_curve.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sealed R5 aggregates into an official K-budget curve")
    parser.add_argument("--aggregate-json", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-budgets", nargs="+", type=int, default=[384, 320, 256, 192, 128])
    args = parser.parse_args()
    print(
        json.dumps(
            export_budget_curve(
                aggregate_jsons=args.aggregate_json,
                output_dir=args.output_dir,
                expected_budgets=args.expected_budgets,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
