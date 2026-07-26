"""Render fixed validation-window sampling evidence across checkpoints.

The input JSONL is produced by ``export_duca_selection_quality``.  It contains
only teacher-free selector inference; GT segments are used below solely as a
post-hoc overlay for interpreting boundary neighborhoods.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tools.bata.plot_selector_geometry import require_matplotlib, save_figure


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _epoch(record: dict[str, Any]) -> int:
    source = record.get("source", {})
    if not isinstance(source, dict):
        return -1
    epoch = source.get("checkpoint_epoch")
    return -1 if epoch is None else int(epoch) + 1


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def _boundaries(record: dict[str, Any]) -> list[float]:
    out: list[float] = []
    for segment, validity in zip(
        record.get("gt_segments", []), record.get("gt_boundary_validity", [])
    ):
        if not isinstance(segment, Sequence) or len(segment) != 2:
            continue
        flags = validity if isinstance(validity, Sequence) and len(validity) == 2 else (True, True)
        if bool(flags[0]):
            out.append(float(segment[0]))
        if bool(flags[1]):
            out.append(float(segment[1]))
    return out


def _draw_gt(ax: Any, record: dict[str, Any]) -> None:
    for segment, validity in zip(
        record.get("gt_segments", []), record.get("gt_boundary_validity", [])
    ):
        if not isinstance(segment, Sequence) or len(segment) != 2:
            continue
        start, end = float(segment[0]), float(segment[1])
        ax.axvspan(start, end, color="#d8eadc", alpha=0.20, linewidth=0)
        flags = validity if isinstance(validity, Sequence) and len(validity) == 2 else (True, True)
        if bool(flags[0]):
            ax.axvline(start, color="#bd2f2f", linestyle="--", linewidth=0.8)
        if bool(flags[1]):
            ax.axvline(end, color="#bd2f2f", linestyle="--", linewidth=0.8)


def _rate(record: dict[str, Any]) -> list[float] | None:
    values = record.get("sampling_rates")
    if values is None:
        values = record.get("sampling_density", record.get("density_probabilities"))
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None
    return [float(value) for value in values]


def _distances(positions: Iterable[int], boundaries: Sequence[float]) -> list[float]:
    if not boundaries:
        return []
    return [min(abs(float(position) - boundary) for boundary in boundaries) for position in positions]


def _plot_one_sample(rows: Sequence[dict[str, Any]], output_prefix: Path) -> list[Path]:
    plt = require_matplotlib()
    first = rows[0]
    valid_len = int(first["valid_len"])
    if any(int(row["valid_len"]) != valid_len for row in rows):
        raise ValueError("fixed validation samples must retain the same valid length")
    x = list(range(valid_len))
    colors = plt.cm.viridis([index / max(len(rows) - 1, 1) for index in range(len(rows))])
    figure, axes = plt.subplots(4, 1, figsize=(12.0, 9.2), sharex=False)
    figure.suptitle(
        "Teacher-free fixed validation window across checkpoints | "
        f"{first['sample_id']} | GT overlay is not selector input",
        fontsize=10,
    )
    evidence_ax, rate_ax, selected_ax, boundary_ax = axes
    _draw_gt(evidence_ax, first)
    _draw_gt(rate_ax, first)
    for color, row in zip(colors, rows):
        label = f"epoch {_epoch(row)}"
        transition = row.get("transition_policy_scores")
        if isinstance(transition, Sequence):
            evidence_ax.plot(x, transition, color=color, linewidth=1.0, label=label)
        rate = _rate(row)
        if rate is not None:
            rate_ax.plot(x, rate, color=color, linewidth=1.1, label=label)
        selected = [int(item) for item in row.get("selected_positions", [])]
        selected_ax.scatter(
            selected,
            [_epoch(row)] * len(selected),
            color=color,
            marker="|",
            s=58,
            linewidths=1.0,
            label=label,
        )
        distances = _distances(selected, _boundaries(row))
        if distances:
            boundary_ax.hist(
                distances,
                bins=[0, 1, 2, 4, 8, 16, 32, 64, max(65, valid_len)],
                density=True,
                histtype="step",
                linewidth=1.2,
                color=color,
                label=label,
            )
    evidence_ax.set_ylabel("transition score")
    rate_ax.set_ylabel("sampling rate")
    selected_ax.set_ylabel("checkpoint epoch")
    selected_ax.set_ylim(min(_epoch(row) for row in rows) - 1, max(_epoch(row) for row in rows) + 1)
    boundary_ax.set_ylabel("selected density")
    boundary_ax.set_xlabel("distance to nearest GT boundary")
    evidence_ax.set_xlabel("dense temporal index")
    rate_ax.set_xlabel("dense temporal index")
    selected_ax.set_xlabel("dense temporal index")
    for axis in (evidence_ax, rate_ax, selected_ax):
        axis.set_xlim(0, max(valid_len - 1, 1))
        axis.grid(True, axis="x", alpha=0.18)
        axis.legend(loc="upper right", ncol=min(4, len(rows)), fontsize=7)
    boundary_ax.grid(True, axis="x", alpha=0.18)
    boundary_ax.legend(loc="upper right", ncol=min(4, len(rows)), fontsize=7)
    figure.tight_layout()
    paths: list[Path] = []
    stem = f"{output_prefix.name}_{_safe_stem(str(first['sample_id']))}"
    for suffix in ("png", "pdf"):
        path = output_prefix.parent / f"{stem}.{suffix}"
        save_figure(figure, path)
        paths.append(path)
    plt.close(figure)
    return paths


def plot_inference_selection(
    *,
    records_jsonl: Sequence[str | Path],
    output_prefix: str | Path,
    sample_id: str | None = None,
    all_fixed_samples: bool = False,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw_path in records_jsonl:
        for record in _read_jsonl(Path(raw_path).expanduser().resolve()):
            grouped.setdefault(str(record["sample_id"]), []).append(record)
    if not grouped:
        raise ValueError("no inference selection records were found")
    if sample_id is not None and all_fixed_samples:
        raise ValueError("sample_id and all_fixed_samples are mutually exclusive")
    selected_ids = (
        sorted(grouped)
        if all_fixed_samples
        else [sample_id if sample_id is not None else sorted(grouped)[0]]
    )
    if any(item not in grouped for item in selected_ids):
        raise ValueError("requested sample_id is absent from the supplied records")
    prefix = Path(output_prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    epochs: dict[str, list[int]] = {}
    for chosen in selected_ids:
        rows = sorted(grouped[chosen], key=_epoch)
        outputs.extend(_plot_one_sample(rows, prefix))
        epochs[chosen] = [_epoch(row) for row in rows]
    summary = {
        "schema_version": "duca_inference_selection_plot_summary_v1",
        "sample_ids": selected_ids,
        "epochs": epochs,
        "outputs": [str(path) for path in outputs],
        "selector_only_inference": True,
        "gt_overlay_not_selector_input": True,
    }
    (prefix.parent / f"{prefix.name}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot fixed validation-window DUCA sampling evidence across checkpoints."
    )
    parser.add_argument("--records-jsonl", required=True, nargs="+")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--all-fixed-samples", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    summary = plot_inference_selection(
        records_jsonl=args.records_jsonl,
        output_prefix=args.output_prefix,
        sample_id=args.sample_id,
        all_fixed_samples=bool(args.all_fixed_samples),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
