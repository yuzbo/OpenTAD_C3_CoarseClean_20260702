from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = Path("analysis_outputs") / "remote_move25_move50_loss_curves_20260709"

TRAIN_RE = re.compile(
    r"\[Train\]: \[(?P<epoch>\d+)\]\[(?P<iter>\d+)/(?P<total>\d+)\]\s+"
    r"Loss=(?P<loss>[0-9.]+)\s+cls_loss=(?P<cls>[0-9.]+)\s+reg_loss=(?P<reg>[0-9.]+)"
)
AVG_MAP_RE = re.compile(r"Average-mAP:\s+(?P<value>[0-9.]+)")
TIOU_RE = re.compile(r"mAP at tIoU (?P<thr>[0-9.]+) is (?P<value>[0-9.]+)%")
LOSS_LIKE_RE = re.compile(r"(?i)(val|test).*loss|loss.*(val|test)|\[Val\]|\[Test\]")
ERROR_RE = re.compile(
    r"(?i)(traceback|runtimeerror|childfailederror|pytorchstreamwriter|unexpected pos|"
    r"file write failed|srun: error|out of memory|\bnan\b|\binf\b)"
)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _parse_train_log(path: Path, *, run_label: str, source_note: str) -> dict[str, Any]:
    epoch_last: dict[int, dict[str, Any]] = {}
    eval_rows: list[dict[str, Any]] = []
    current_eval: dict[str, Any] | None = None
    last_train_epoch: int | None = None
    loss_like_lines: list[str] = []
    error_lines: list[str] = []

    lines = _read_lines(path)
    for line in lines:
        train_match = TRAIN_RE.search(line)
        if train_match:
            epoch = int(train_match.group("epoch"))
            row = {
                "run": run_label,
                "epoch": epoch,
                "iter": int(train_match.group("iter")),
                "total_iter": int(train_match.group("total")),
                "train_loss": float(train_match.group("loss")),
                "train_cls_loss": float(train_match.group("cls")),
                "train_reg_loss": float(train_match.group("reg")),
                "source_note": source_note,
            }
            epoch_last[epoch] = row
            last_train_epoch = epoch
            continue

        avg_match = AVG_MAP_RE.search(line)
        if avg_match:
            current_eval = {
                "run": run_label,
                "epoch": last_train_epoch,
                "average_map": float(avg_match.group("value")),
                "source_note": source_note,
            }
            eval_rows.append(current_eval)
            continue

        tiou_match = TIOU_RE.search(line)
        if tiou_match and current_eval is not None:
            key = "map_tiou_" + tiou_match.group("thr").replace(".", "_")
            current_eval[key] = float(tiou_match.group("value"))
            continue

        if LOSS_LIKE_RE.search(line) and "val_loss_interval" not in line:
            loss_like_lines.append(line.strip())
        if ERROR_RE.search(line):
            error_lines.append(line.strip())

    train_rows = [epoch_last[key] for key in sorted(epoch_last)]
    return {
        "path": str(path),
        "run": run_label,
        "source_note": source_note,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "loss_like_lines": loss_like_lines[:50],
        "error_lines": error_lines[-50:],
        "line_count": len(lines),
    }


def _read_error_lines(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    return [line.strip() for line in _read_lines(path) if ERROR_RE.search(line)][-80:]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _summarize_run(train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "train_epoch_count": len(train_rows),
        "first_epoch": train_rows[0]["epoch"] if train_rows else None,
        "last_epoch": train_rows[-1]["epoch"] if train_rows else None,
        "last_train_loss": train_rows[-1]["train_loss"] if train_rows else None,
        "last_train_cls_loss": train_rows[-1]["train_cls_loss"] if train_rows else None,
        "last_train_reg_loss": train_rows[-1]["train_reg_loss"] if train_rows else None,
        "eval_count": len(eval_rows),
    }
    if eval_rows:
        best_avg = max(eval_rows, key=lambda row: float(row.get("average_map", float("-inf"))))
        last_eval = eval_rows[-1]
        summary.update(
            {
                "best_average_map": best_avg.get("average_map"),
                "best_average_map_epoch": best_avg.get("epoch"),
                "last_average_map": last_eval.get("average_map"),
                "last_eval_epoch": last_eval.get("epoch"),
            }
        )
        for key in ("map_tiou_0_50", "map_tiou_0_60", "map_tiou_0_70"):
            rows_with_key = [row for row in eval_rows if key in row]
            if rows_with_key:
                best = max(rows_with_key, key=lambda row: float(row[key]))
                summary[f"best_{key}"] = best[key]
                summary[f"best_{key}_epoch"] = best.get("epoch")
                summary[f"last_{key}"] = last_eval.get(key)
    return summary


def _write_summary_csv(path: Path, parsed: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in parsed:
        summary = _summarize_run(item["train_rows"], item["eval_rows"])
        rows.append({"run": item["run"], "source_note": item["source_note"], **summary})
    _write_csv(path, rows)
    return rows


def _plot_train_loss(train_rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    _set_style()
    runs = list(dict.fromkeys(str(row["run"]) for row in train_rows))
    colors = {"move25": "#2563eb", "move50": "#f97316"}

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax_loss, ax_parts = axes
    for run in runs:
        rows = [row for row in train_rows if str(row["run"]) == run]
        epochs = [int(row["epoch"]) for row in rows]
        color = colors.get(run, "#475569")
        ax_loss.plot(
            epochs,
            [float(row["train_loss"]) for row in rows],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3.0,
            label=run,
        )
        ax_parts.plot(
            epochs,
            [float(row["train_cls_loss"]) for row in rows],
            color=color,
            linewidth=1.8,
            linestyle="-",
            label=f"{run} 分类",
        )
        ax_parts.plot(
            epochs,
            [float(row["train_reg_loss"]) for row in rows],
            color=color,
            linewidth=1.8,
            linestyle="--",
            label=f"{run} 回归",
        )

    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("训练总损失")
    ax_loss.grid(axis="both", alpha=0.18)
    ax_loss.legend(frameon=False)

    ax_parts.set_xlabel("epoch")
    ax_parts.set_ylabel("损失分量")
    ax_parts.grid(axis="both", alpha=0.18)
    ax_parts.legend(frameon=False, ncol=1)

    fig.text(0.01, 0.985, "远端检测器训练 loss-vs-epoch", fontsize=15, weight="bold", va="top")
    fig.text(
        0.01,
        0.935,
        "每个 epoch 取 train.out 中该 epoch 最后一条迭代日志；move50 来自 failure_evidence tree，训练到 epoch 57 后保存 checkpoint 失败。",
        fontsize=10.5,
        color="#475569",
        va="top",
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.14, wspace=0.25)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_eval_map(eval_rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    _set_style()
    runs = list(dict.fromkeys(str(row["run"]) for row in eval_rows))
    colors = {"move25": "#2563eb", "move50": "#f97316"}
    metrics = [
        ("average_map", "平均 mAP"),
        ("map_tiou_0_50", "mAP@0.5"),
        ("map_tiou_0_70", "mAP@0.7"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=False)
    for ax, (key, label) in zip(axes, metrics):
        for run in runs:
            rows = [row for row in eval_rows if str(row["run"]) == run and key in row and row.get("epoch") is not None]
            if not rows:
                continue
            ax.plot(
                [int(row["epoch"]) for row in rows],
                [float(row[key]) for row in rows],
                color=colors.get(run, "#475569"),
                linewidth=2.0,
                marker="o",
                markersize=4,
                label=run,
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel(label + " (%)")
        ax.grid(axis="both", alpha=0.18)
        ax.legend(frameon=False)

    fig.text(0.01, 0.985, "远端验证性能曲线", fontsize=15, weight="bold", va="top")
    fig.text(
        0.01,
        0.93,
        "日志没有记录 test/val loss；OpenTAD 当前配置 val_loss_interval=-1，因此这里展示验证 mAP 曲线。",
        fontsize=10.5,
        color="#475569",
        va="top",
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.14, wspace=0.28)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def build_curves(
    *,
    output_dir: Path,
    move25_log: Path,
    move50_log: Path,
    move50_driver_log: Path | None,
    move50_precheck_log: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed = [
        _parse_train_log(move25_log, run_label="move25", source_note="budgeted-radius move25 full detector training"),
        _parse_train_log(
            move50_log,
            run_label="move50",
            source_note="failure_evidence tree move50 detector training; checkpoint save failed after epoch 57",
        ),
    ]
    train_rows = [row for item in parsed for row in item["train_rows"]]
    eval_rows = [row for item in parsed for row in item["eval_rows"]]

    train_csv = output_dir / "remote_move25_move50_train_loss_by_epoch.csv"
    eval_csv = output_dir / "remote_move25_move50_eval_map_by_epoch.csv"
    summary_csv = output_dir / "remote_move25_move50_summary.csv"
    _write_csv(train_csv, train_rows)
    _write_csv(eval_csv, eval_rows)
    summary_rows = _write_summary_csv(summary_csv, parsed)

    train_fig = output_dir / "remote_move25_move50_train_loss_vs_epoch.png"
    _plot_train_loss(train_rows, train_fig)

    eval_fig = output_dir / "remote_move25_move50_eval_map_vs_epoch.png"
    _plot_eval_map(eval_rows, eval_fig)

    move50_precheck_note = None
    if move50_precheck_log is not None and move50_precheck_log.exists():
        text = move50_precheck_log.read_text(encoding="utf-8", errors="replace")
        move50_precheck_note = {
            "path": str(move50_precheck_log),
            "contains_precheck_only_complete": "PRECHECK_ONLY all variants complete" in text,
            "contains_full_train_log": "[Train]:" in text and "Loss=" in text,
        }

    manifest = {
        "schema_version": "remote_move25_move50_loss_curves_v2",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evidence_boundary": {
            "move25": "complete detector train.out parsed through epoch 59",
            "move50": "near-complete detector train.out parsed through epoch 57; run failed while saving checkpoint, so no epoch 59/60 final checkpoint evidence",
            "move50_latest_budgeted_radius": "latest budgeted-radius move50 directory only contains precheck/ledger validation; no train.out was found there",
            "test_loss": "not available in inspected train.out files because val_loss_interval=-1; validation mAP evaluation is plotted separately",
        },
        "parsed_logs": [
            {
                "run": item["run"],
                "path": item["path"],
                "line_count": item["line_count"],
                "train_epoch_count": len(item["train_rows"]),
                "first_epoch": item["train_rows"][0]["epoch"] if item["train_rows"] else None,
                "last_epoch": item["train_rows"][-1]["epoch"] if item["train_rows"] else None,
                "eval_count": len(item["eval_rows"]),
                "loss_like_lines": item["loss_like_lines"],
                "error_lines": item["error_lines"],
            }
            for item in parsed
        ],
        "summary": summary_rows,
        "move50_driver_error_lines": _read_error_lines(move50_driver_log),
        "move50_latest_precheck": move50_precheck_note,
        "outputs": {
            "summary_csv": str(summary_csv),
            "train_loss_csv": str(train_csv),
            "eval_map_csv": str(eval_csv),
            "train_loss_png": str(train_fig),
            "train_loss_pdf": str(train_fig.with_suffix(".pdf")),
            "eval_map_png": str(eval_fig),
            "eval_map_pdf": str(eval_fig.with_suffix(".pdf")),
        },
    }
    manifest_path = output_dir / "remote_move25_move50_loss_curves_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse remote move25/move50 detector logs and draw loss-vs-epoch curves.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--move25-log", default=str(DEFAULT_OUTPUT_DIR / "raw_logs" / "move25_budgeted_radius_train.out"))
    parser.add_argument("--move50-log", default=str(DEFAULT_OUTPUT_DIR / "raw_logs" / "move50_e5d6b78_near_complete_train.out"))
    parser.add_argument("--move50-driver-log", default=str(DEFAULT_OUTPUT_DIR / "raw_logs" / "move50_e5d6b78_driver.log"))
    parser.add_argument("--move50-precheck-log", default=str(DEFAULT_OUTPUT_DIR / "raw_logs" / "move50_budgeted_radius_precheck.log"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    manifest = build_curves(
        output_dir=Path(args.output_dir),
        move25_log=Path(args.move25_log),
        move50_log=Path(args.move50_log),
        move50_driver_log=Path(args.move50_driver_log) if args.move50_driver_log else None,
        move50_precheck_log=Path(args.move50_precheck_log) if args.move50_precheck_log else None,
    )
    print(json.dumps(manifest["outputs"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
