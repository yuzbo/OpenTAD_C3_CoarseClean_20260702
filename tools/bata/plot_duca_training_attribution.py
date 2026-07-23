"""Plot fixed-batch DUCA training-attribution JSONL records across checkpoints."""

from __future__ import annotations

import argparse
import json
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
    value = source.get("checkpoint_epoch") if isinstance(source, dict) else None
    return -1 if value is None else int(value)


def _normalize(values: Sequence[float] | None) -> list[float] | None:
    if values is None:
        return None
    total = float(sum(max(0.0, float(value)) for value in values))
    if total <= 1.0e-12:
        return [0.0 for _ in values]
    return [max(0.0, float(value)) / total for value in values]


def _channel(values: Sequence[Sequence[float]] | None, index: int) -> list[float] | None:
    if values is None:
        return None
    if any(len(row) <= index for row in values):
        return None
    return [float(row[index]) for row in values]


def _draw_gt_boundaries(ax: Any, record: dict[str, Any]) -> None:
    for segment, validity in zip(
        record.get("gt_segments", []), record.get("gt_boundary_validity", [])
    ):
        if len(segment) != 2:
            continue
        start, end = float(segment[0]), float(segment[1])
        ax.axvspan(start, end, color="#d8eadc", alpha=0.22, linewidth=0)
        if len(validity) == 2 and bool(validity[0]):
            ax.axvline(start, color="#bd2f2f", linestyle="--", linewidth=0.9)
        if len(validity) == 2 and bool(validity[1]):
            ax.axvline(end, color="#bd2f2f", linestyle="--", linewidth=0.9)


def _selected_ticks(ax: Any, positions: Iterable[int], level: float = 0.0) -> None:
    ticks = list(int(position) for position in positions)
    if ticks:
        ax.vlines(ticks, level - 0.03, level + 0.03, color="#202020", linewidth=0.7)


def _plot_epoch(record: dict[str, Any], output_prefix: Path) -> list[Path]:
    plt = require_matplotlib()
    valid_len = int(record["valid_len"])
    x = list(range(valid_len))
    epoch = _epoch(record)
    sample_id = str(record["sample_id"])
    figure, axes = plt.subplots(4, 1, figsize=(12.0, 8.8), sharex=True)
    figure.suptitle(
        f"Training-only attribution | {sample_id} | epoch {epoch} | GT is overlay only, not inference input",
        fontsize=10,
    )

    score_ax, rate_ax, utility_ax, grad_ax = axes
    score_ax.plot(x, record.get("p_action", []), label="p_action", linewidth=1.2)
    score_ax.plot(x, record.get("abs_delta_p_action", []), label="|delta p|", linewidth=1.1)
    score_ax.plot(x, record.get("transition_policy_scores", []), label="transition score", linewidth=1.1)
    _draw_gt_boundaries(score_ax, record)
    score_ax.set_ylabel("coarse evidence")
    score_ax.legend(loc="upper right", ncol=3, fontsize=7)

    rate_ax.plot(x, record.get("sampling_rates", []), label="sampling rate", linewidth=1.3)
    density = record.get("sampling_density")
    if density is not None:
        rate_ax.plot(x, density, label="sampling density", linewidth=1.0, linestyle=":")
    _draw_gt_boundaries(rate_ax, record)
    _selected_ticks(rate_ax, record.get("selected_positions", []), level=0.0)
    rate_ax.set_ylabel("rate / density")
    rate_ax.legend(loc="upper right", fontsize=7)

    cls_input = _normalize(
        record.get("detector_cls_input_x_gradient_dense_interpolated")
    )
    reg_input = _normalize(
        record.get("detector_reg_input_x_gradient_dense_interpolated")
    )
    cls_pred = _channel(record.get("detector_contribution_prediction_distribution"), 0)
    reg_pred = _channel(record.get("detector_contribution_prediction_distribution"), 1)
    for values, label, style in (
        (cls_input, "cls input x grad (interpolated)", "-"),
        (reg_input, "reg input x grad (interpolated)", "-"),
        (cls_pred, "predicted cls contribution", "--"),
        (reg_pred, "predicted reg contribution", "--"),
    ):
        if values is not None:
            utility_ax.plot(x, values, label=label, linewidth=1.0, linestyle=style)
    _draw_gt_boundaries(utility_ax, record)
    utility_ax.set_ylabel("normalized contribution")
    utility_ax.legend(loc="upper right", ncol=2, fontsize=7)

    gradient = record.get("sampling_rate_logit_gradient_abs")
    if gradient is not None:
        grad_ax.plot(x, gradient, label="|d detector loss / d rate logit|", linewidth=1.2)
    logits = record.get("sampling_rate_logits")
    if logits is not None:
        grad_ax.plot(x, logits, label="sampling-rate logit", linewidth=0.9, linestyle=":")
    _draw_gt_boundaries(grad_ax, record)
    _selected_ticks(grad_ax, record.get("selected_positions", []), level=0.0)
    grad_ax.set_ylabel("selector gradient")
    grad_ax.set_xlabel("dense temporal index")
    grad_ax.legend(loc="upper right", fontsize=7)
    for axis in axes:
        axis.set_xlim(0, max(valid_len - 1, 1))
        axis.grid(True, axis="x", alpha=0.18)
    figure.tight_layout()
    stem = f"{output_prefix.name}_epoch{epoch:03d}"
    paths = []
    for suffix in ("png", "pdf"):
        path = output_prefix.parent / f"{stem}.{suffix}"
        save_figure(figure, path)
        paths.append(path)
    plt.close(figure)
    return paths


def _plot_epoch_overlay(rows: Sequence[dict[str, Any]], output_prefix: Path) -> list[Path]:
    """Overlay the same fixed training window across checkpoints.

    The individual epoch pages answer "what did the model do then?".  This
    compact four-lane view answers the more useful training question: whether
    the same evidence, sampling-rate mass, detector contribution and selector
    gradient actually move together over time.  Curves are never averaged
    across videos.
    """

    plt = require_matplotlib()
    first = rows[0]
    valid_len = int(first["valid_len"])
    sample_id = str(first["sample_id"])
    if any(int(row["valid_len"]) != valid_len for row in rows):
        raise ValueError("all overlaid records must use the same fixed window length")
    x = list(range(valid_len))
    figure, axes = plt.subplots(4, 1, figsize=(12.0, 9.0), sharex=True)
    figure.suptitle(
        "Fixed training window across checkpoints | "
        f"{sample_id} | GT is overlay only, not inference input",
        fontsize=10,
    )
    colors = plt.cm.viridis(
        [index / max(len(rows) - 1, 1) for index in range(len(rows))]
    )
    score_ax, rate_ax, utility_ax, grad_ax = axes
    _draw_gt_boundaries(score_ax, first)
    _draw_gt_boundaries(rate_ax, first)
    _draw_gt_boundaries(utility_ax, first)
    _draw_gt_boundaries(grad_ax, first)
    for color, row in zip(colors, rows):
        epoch = _epoch(row)
        label = f"epoch {epoch}"
        score_ax.plot(x, row.get("transition_policy_scores", []), color=color, linewidth=1.05, label=label)
        rate_ax.plot(x, row.get("sampling_rates", []), color=color, linewidth=1.05, label=label)
        rate_ax.scatter(
            row.get("selected_positions", []),
            [0.0] * len(row.get("selected_positions", [])),
            color=color,
            marker="|",
            s=42,
            linewidths=1.0,
        )
        utility = _normalize(row.get("detector_cls_input_x_gradient_dense_interpolated"))
        if utility is not None:
            utility_ax.plot(x, utility, color=color, linewidth=1.05, label=label)
        gradient = row.get("sampling_rate_logit_gradient_abs")
        if gradient is not None:
            grad_ax.plot(x, gradient, color=color, linewidth=1.05, label=label)
    score_ax.set_ylabel("transition score")
    rate_ax.set_ylabel("sampling rate")
    utility_ax.set_ylabel("cls contribution\n(interpolated)")
    grad_ax.set_ylabel("|dL/d rate logit|")
    grad_ax.set_xlabel("dense temporal index")
    for axis in axes:
        axis.set_xlim(0, max(valid_len - 1, 1))
        axis.grid(True, axis="x", alpha=0.18)
        axis.legend(loc="upper right", ncol=min(4, len(rows)), fontsize=7)
    figure.tight_layout()
    paths: list[Path] = []
    stem = f"{output_prefix.name}_overlay"
    for suffix in ("png", "pdf"):
        path = output_prefix.parent / f"{stem}.{suffix}"
        save_figure(figure, path)
        paths.append(path)
    plt.close(figure)
    return paths


def plot_training_attribution(
    *,
    records_jsonl: Sequence[str | Path],
    output_prefix: str | Path,
    sample_id: str | None = None,
) -> dict[str, Any]:
    """Render one page per epoch for one fixed training video window."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw_path in records_jsonl:
        path = Path(raw_path).expanduser().resolve()
        for record in _read_jsonl(path):
            grouped.setdefault(str(record["sample_id"]), []).append(record)
    if not grouped:
        raise ValueError("no attribution records were found")
    chosen = sample_id or sorted(grouped)[0]
    if chosen not in grouped:
        raise ValueError(f"sample_id {chosen!r} is absent from the supplied records")
    rows = sorted(grouped[chosen], key=_epoch)
    output = Path(output_prefix).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for row in rows:
        paths.extend(_plot_epoch(row, output))
    paths.extend(_plot_epoch_overlay(rows, output))
    summary = {
        "schema_version": "duca_training_attribution_plot_summary_v1",
        "sample_id": chosen,
        "epochs": [_epoch(row) for row in rows],
        "outputs": [str(path) for path in paths],
        "overlay_outputs": [
            str(path) for path in paths if path.stem.endswith("_overlay")
        ],
        "gt_overlay_not_inference_input": True,
    }
    summary_path = output.parent / f"{output.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a fixed-video DUCA training-attribution timeline across epochs."
    )
    parser.add_argument("--records-jsonl", required=True, nargs="+", help="one exported JSONL per checkpoint")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--sample-id", help="defaults to the first shared sample")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    summary = plot_training_attribution(
        records_jsonl=args.records_jsonl,
        output_prefix=args.output_prefix,
        sample_id=args.sample_id,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
