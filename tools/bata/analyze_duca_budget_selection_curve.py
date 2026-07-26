from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.analyze_duca_selection_quality import analyze_jsonl
from tools.bata.export_duca_selection_quality import export_records


SCHEMA = "duca_budget_selection_curve_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(summary: Mapping[str, Any], *keys: str) -> float | None:
    value: Any = summary
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    if isinstance(value, Mapping):
        value = value.get("mean")
    return None if value is None else float(value)


def _load_cells(
    matrix_summaries: Sequence[str | Path],
    *,
    expected_budgets: Sequence[int],
    backend: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: dict[int, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for item in matrix_summaries:
        path = Path(item).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "duca_r5_paper_matrix_v1":
            raise RuntimeError(f"not an R5 matrix summary: {path}")
        sha_file = Path(str(payload.get("matrix_summary_sha256_file", ""))).resolve()
        if not sha_file.is_file() or sha_file.read_text(encoding="utf-8").strip() != _sha256(path):
            raise RuntimeError(f"matrix summary hash drift: {path}")
        sources.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "git_commit": str(payload.get("git_commit", "")),
                "budgets": list(payload.get("budgets", [])),
            }
        )
        root = Path(str(payload.get("output_dir", path.parent))).resolve()
        for cell in payload.get("cells", []):
            if (
                cell.get("backend") != backend
                or cell.get("arm") != "learned"
                or int(cell.get("seed", -1)) != int(seed)
            ):
                continue
            budget = int(cell["budget"])
            if budget in selected:
                raise RuntimeError(f"duplicate selection-quality cell for K={budget}")
            selected[budget] = {
                **dict(cell),
                "checkpoint": str(
                    root
                    / "runs"
                    / str(cell["id"])
                    / "gpu1_id0/checkpoint/epoch_59.pth"
                ),
            }
    expected = {int(value) for value in expected_budgets}
    if set(selected) != expected:
        raise RuntimeError(
            f"selection curve budget mismatch: expected={sorted(expected)}, observed={sorted(selected)}"
        )
    return [selected[budget] for budget in sorted(selected, reverse=True)], sources


def _metric_row(*, budget: int, cell_id: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    learned = summary["selection"]["learned"]
    pooled = learned["pooled"]
    burst = learned["boundary_burst"]["r2q3"]
    return {
        "cell_id": cell_id,
        "K": budget,
        "frame_fraction": budget / 768.0,
        "coarse_auroc": _mean(summary, "coarse", "pooled", "auroc"),
        "coarse_auprc": _mean(summary, "coarse", "pooled", "auprc"),
        "transition_r0_policy_auprc": _mean(summary, "transition", "r0", "policy", "auprc"),
        "selected_count": _mean(learned, "selected_count"),
        "mean_endpoint_distance": _mean(learned, "mean_endpoint_distance"),
        "max_unselected_hole": _mean(learned, "max_unselected_hole"),
        "action_enrichment": _mean(learned, "action_enrichment"),
        "boundary_recall_r0": _mean(pooled, "boundary_recall", "r0"),
        "boundary_recall_r2": _mean(pooled, "boundary_recall", "r2"),
        "boundary_recall_r4": _mean(pooled, "boundary_recall", "r4"),
        "r2q3_endpoint_quota_recall": _mean(burst, "endpoint_quota_recall"),
        "r2q3_endpoint_bilateral_recall": _mean(burst, "endpoint_bilateral_recall"),
        "r2q3_both_endpoints_quota_recall": _mean(burst, "both_endpoints_quota_recall"),
        "paired_gain_boundary_recall_r0": _mean(
            summary, "comparison", "paired_learned_minus_uniform_boundary_recall_r0"
        ),
        "paired_uniform_minus_learned_endpoint_distance": _mean(
            summary, "comparison", "paired_uniform_minus_learned_endpoint_distance"
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: Sequence[Mapping[str, Any]], output: Path) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"unavailable: {exc}"
    ordered = sorted(rows, key=lambda row: int(row["K"]))
    x = [int(row["K"]) for row in ordered]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    for radius, color in ((0, "#b54832"), (2, "#1769aa"), (4, "#3b7d3a")):
        axes[0].plot(x, [row[f"boundary_recall_r{radius}"] for row in ordered], marker="o", color=color, label=f"r={radius}")
    axes[0].set_ylabel("boundary recall")
    axes[0].legend(frameon=False)
    axes[1].plot(x, [row["r2q3_endpoint_quota_recall"] for row in ordered], marker="s", label="quota")
    axes[1].plot(x, [row["r2q3_endpoint_bilateral_recall"] for row in ordered], marker="^", label="bilateral")
    axes[1].set_ylabel("boundary-burst coverage")
    axes[1].legend(frameon=False)
    axes[2].plot(x, [row["mean_endpoint_distance"] for row in ordered], marker="o", label="endpoint distance")
    axes[2].plot(x, [row["max_unselected_hole"] for row in ordered], marker="s", label="max hole")
    axes[2].set_ylabel("dense-grid frames")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.set_xlabel("selected frames K")
        axis.grid(alpha=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"duca_budget_selection_curve.{suffix}", dpi=300)
    plt.close(fig)
    return "generated"


def analyze_budget_selection_curve(
    *,
    matrix_summaries: Sequence[str | Path],
    output_dir: str | Path,
    expected_budgets: Sequence[int] = (384, 320, 256, 192, 128),
    backend: str = "actionformer",
    seed: int = 3407,
    device: str = "cuda:0",
    bootstrap_samples: int = 1000,
    num_workers: int = 4,
) -> dict[str, Any]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cells, sources = _load_cells(
        matrix_summaries,
        expected_budgets=expected_budgets,
        backend=backend,
        seed=seed,
    )
    rows: list[dict[str, Any]] = []
    for cell in cells:
        budget = int(cell["budget"])
        checkpoint = Path(str(cell["checkpoint"])).resolve()
        config = Path(str(cell["config"])).resolve()
        if not checkpoint.is_file() or not config.is_file():
            raise RuntimeError(f"terminal selection evidence is missing for {cell['id']}")
        cell_out = out / f"k{budget}"
        records = cell_out / "selection_quality_records.jsonl"
        export_records(
            config=config,
            checkpoint=checkpoint,
            output_jsonl=records,
            summary_json=cell_out / "selection_quality_export.json",
            split="val",
            device=device,
            use_ema="true",
            use_amp=True,
            num_workers=num_workers,
            seed=seed,
        )
        summary = analyze_jsonl(
            records_jsonl=records,
            output_dir=cell_out,
            bootstrap_samples=bootstrap_samples,
            random_seed=seed + budget,
        )
        rows.append(_metric_row(budget=budget, cell_id=str(cell["id"]), summary=summary))
    rows.sort(key=lambda row: int(row["K"]), reverse=True)
    _write_csv(out / "duca_budget_selection_curve.csv", rows)
    payload = {
        "schema": SCHEMA,
        "ok": True,
        "task": "offline_tad_selection_diagnostic",
        "backend": backend,
        "seed": seed,
        "expected_budgets": [int(value) for value in expected_budgets],
        "sources": sources,
        "rows": rows,
        "protocol": {
            "terminal_checkpoint": "epoch_59_state_dict_ema",
            "split": "full_validation",
            "gt_role": "evaluation_only_not_selector_input",
            "paired_uniform_reference": "round_linspace_endpoints",
        },
        "plot_status": _plot(rows, out),
        "paper_claim_allowed": False,
    }
    (out / "duca_budget_selection_curve.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DUCA selection distributions over a K-budget curve")
    parser.add_argument("--matrix-summary", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-budgets", nargs="+", type=int, default=[384, 320, 256, 192, 128])
    parser.add_argument("--backend", default="actionformer")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze_budget_selection_curve(
                matrix_summaries=args.matrix_summary,
                output_dir=args.output_dir,
                expected_budgets=args.expected_budgets,
                backend=args.backend,
                seed=args.seed,
                device=args.device,
                bootstrap_samples=args.bootstrap_samples,
                num_workers=args.num_workers,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
