from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean(summary: dict[str, Any], *keys: str) -> float | None:
    value: Any = summary
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, dict):
        value = value.get("mean")
    return None if value is None else float(value)


def _arm_record(name: str, completion_path: Path) -> dict[str, Any]:
    completion = _load(completion_path)
    if not completion.get("ok"):
        raise ValueError(f"{name}: completion is not successful")
    metrics = completion.get("metrics")
    if not isinstance(metrics, dict) or "average_mAP" not in metrics:
        raise ValueError(f"{name}: official mAP metrics are missing")
    summary_path = completion.get("selection_quality_summary")
    records_path = completion.get("selection_quality_records")
    if not summary_path or not records_path:
        raise ValueError(f"{name}: trained selector distribution evidence is missing")
    summary = _load(summary_path)
    learned = summary["selection"]["learned"]
    return {
        "name": name,
        "completion": str(completion_path.resolve()),
        "git_commit": completion.get("git_commit"),
        "average_mAP": 100.0 * float(metrics["average_mAP"]),
        "mAP@0.3": 100.0 * float(metrics["mAP@0.3"]),
        "mAP@0.4": 100.0 * float(metrics["mAP@0.4"]),
        "mAP@0.5": 100.0 * float(metrics["mAP@0.5"]),
        "mAP@0.6": 100.0 * float(metrics["mAP@0.6"]),
        "mAP@0.7": 100.0 * float(metrics["mAP@0.7"]),
        "mean_endpoint_distance": float(learned["mean_endpoint_distance"]["mean"]),
        "max_unselected_hole": float(learned["max_unselected_hole"]["mean"]),
        "boundary_recall_r1": float(learned["boundary_recall"]["r1"]["mean"]),
        "boundary_recall_r2": float(learned["boundary_recall"]["r2"]["mean"]),
        "r2q3_cluster_size": float(
            learned["boundary_burst"]["r2q3"]["mean_endpoint_selected_count"]["mean"]
        ),
        "r4q5_cluster_size": float(
            learned["boundary_burst"]["r4q5"]["mean_endpoint_selected_count"]["mean"]
        ),
        "records": str(Path(records_path).resolve()),
        "analyzed_records": str(
            (Path(summary_path).parent / "selection_quality_analyzed.jsonl").resolve()
        ),
    }


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "name",
        "average_mAP",
        "mAP@0.3",
        "mAP@0.4",
        "mAP@0.5",
        "mAP@0.6",
        "mAP@0.7",
        "mean_endpoint_distance",
        "max_unselected_hole",
        "boundary_recall_r1",
        "boundary_recall_r2",
        "r2q3_cluster_size",
        "r4q5_cluster_size",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def _plots(rows: list[dict[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 8, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42}
    )
    names = [row["name"] for row in rows]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"][: len(rows)]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.7), constrained_layout=True)
    axes[0, 0].bar(names, [row["average_mAP"] for row in rows], color=colors)
    axes[0, 0].set(ylabel="Official Avg-mAP (%)", title="Detection performance")
    axes[0, 1].bar(names, [row["mAP@0.7"] for row in rows], color=colors)
    axes[0, 1].set(ylabel="mAP@0.7 (%)", title="Boundary-sensitive performance")
    axes[1, 0].bar(names, [row["boundary_recall_r1"] for row in rows], color=colors)
    axes[1, 0].set(ylabel="Endpoint recall (r=1)", ylim=(0, 1), title="Boundary coverage")
    axes[1, 1].bar(names, [row["max_unselected_hole"] for row in rows], color=colors)
    axes[1, 1].set(ylabel="Mean maximum hole", title="Context coverage")
    for ax in axes.flat:
        ax.tick_params(axis="x", labelrotation=18)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"density_transport_performance_distribution.{suffix}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.2), constrained_layout=True)
    for row, color in zip(rows, colors):
        ax.scatter(
            row["max_unselected_hole"],
            row["average_mAP"],
            s=70,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=row["name"],
        )
    ax.set(xlabel="Mean maximum unselected hole", ylabel="Official Avg-mAP (%)")
    ax.legend(frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"density_transport_map_vs_hole.{suffix}", dpi=300)
    plt.close(fig)


def aggregate(arms: list[tuple[str, Path]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_arm_record(name, path) for name, path in arms]
    commits = {row["git_commit"] for row in rows}
    if len(commits) != 1:
        raise ValueError(f"density arms do not share one exact commit: {sorted(commits)}")
    _write_table(output_dir / "density_transport_results.tsv", rows)
    _plots(rows, output_dir)
    result = {
        "schema": "duca_density_transport_official60_comparison_v1",
        "ok": True,
        "task": "offline_temporal_action_detection",
        "git_commit": rows[0]["git_commit"],
        "official_validation_comparable": True,
        "rows": rows,
    }
    (output_dir / "density_transport_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        action="append",
        nargs=2,
        metavar=("NAME", "COMPLETION_JSON"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(
        [(name, Path(path)) for name, path in args.arm], args.output_dir.resolve()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
