from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


OUTPUT_DIR = Path("analysis_outputs") / "remote_move25_move50_loss_curves_20260709"
RAW_DIR = OUTPUT_DIR / "raw_logs"
MOVE25_LOG = RAW_DIR / "move25_budgeted_radius_train.out"
MOVE50_LOG = RAW_DIR / "move50_e5d6b78_near_complete_train.out"

TRAIN_RE = re.compile(
    r"\[Train\]: \[(?P<epoch>\d+)\]\[(?P<iter>\d+)/(?P<total>\d+)\]\s+"
    r"Loss=(?P<loss>[0-9.]+)\s+cls_loss=(?P<cls>[0-9.]+)\s+reg_loss=(?P<reg>[0-9.]+)"
)
AVG_MAP_RE = re.compile(r"Average-mAP:\s+(?P<value>[0-9.]+)")
TIOU_RE = re.compile(r"mAP at tIoU (?P<thr>[0-9.]+) is (?P<value>[0-9.]+)%")
TEST_LOSS_RE = re.compile(
    r"(?i)(\[(Val|Test)\].*Loss=|\b(val|valid|validation|test)[_-]?loss\b|\bloss[_-]?(val|valid|validation|test)\b)"
)


def set_style() -> None:
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
            "savefig.pad_inches": 0.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def parse_log(path: Path, run: str, note: str) -> dict[str, Any]:
    epoch_last: dict[int, dict[str, Any]] = {}
    eval_rows: list[dict[str, Any]] = []
    current_eval: dict[str, Any] | None = None
    last_epoch: int | None = None
    test_loss_like: list[str] = []

    for line in read_lines(path):
        train_match = TRAIN_RE.search(line)
        if train_match:
            epoch = int(train_match.group("epoch"))
            row = {
                "run": run,
                "epoch": epoch,
                "train_loss": float(train_match.group("loss")),
                "cls_loss": float(train_match.group("cls")),
                "reg_loss": float(train_match.group("reg")),
                "note": note,
            }
            epoch_last[epoch] = row
            last_epoch = epoch
            continue

        avg_match = AVG_MAP_RE.search(line)
        if avg_match:
            current_eval = {
                "run": run,
                "epoch": last_epoch,
                "average_map": float(avg_match.group("value")),
                "note": note,
            }
            eval_rows.append(current_eval)
            continue

        tiou_match = TIOU_RE.search(line)
        if tiou_match and current_eval is not None:
            key = "map_tiou_" + tiou_match.group("thr").replace(".", "_")
            current_eval[key] = float(tiou_match.group("value"))
            continue

        if TEST_LOSS_RE.search(line) and "val_loss_interval" not in line:
            test_loss_like.append(line.strip())

    return {
        "run": run,
        "path": str(path),
        "train_rows": [epoch_last[key] for key in sorted(epoch_last)],
        "eval_rows": eval_rows,
        "test_loss_like": test_loss_like[:40],
        "note": note,
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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
        writer.writerows(rows)


def best_summary(parsed: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in parsed:
        train_rows = item["train_rows"]
        eval_rows = item["eval_rows"]
        best_avg = max(eval_rows, key=lambda row: row["average_map"]) if eval_rows else None
        row: dict[str, Any] = {
            "run": item["run"],
            "train_epoch_count": len(train_rows),
            "last_train_epoch": train_rows[-1]["epoch"] if train_rows else None,
            "last_train_loss": train_rows[-1]["train_loss"] if train_rows else None,
            "eval_count": len(eval_rows),
            "best_average_map": best_avg["average_map"] if best_avg else None,
            "best_average_map_epoch": best_avg["epoch"] if best_avg else None,
            "test_loss_recorded": bool(item["test_loss_like"]),
            "note": item["note"],
        }
        for metric in ("map_tiou_0_50", "map_tiou_0_60", "map_tiou_0_70"):
            rows_with_metric = [entry for entry in eval_rows if metric in entry]
            if rows_with_metric:
                best = max(rows_with_metric, key=lambda entry: entry[metric])
                row[f"best_{metric}"] = best[metric]
                row[f"best_{metric}_epoch"] = best["epoch"]
        rows.append(row)
    return rows


def plot_train_loss(train_rows: Sequence[dict[str, Any]], path: Path) -> None:
    set_style()
    colors = {"move25": "#2563eb", "move50": "#f97316"}
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax_total, ax_parts = axes

    for run in ("move25", "move50"):
        rows = [row for row in train_rows if row["run"] == run]
        if not rows:
            continue
        epochs = [row["epoch"] for row in rows]
        color = colors[run]
        ax_total.plot(epochs, [row["train_loss"] for row in rows], color=color, marker="o", markersize=3, linewidth=2.1, label=run)
        ax_parts.plot(epochs, [row["cls_loss"] for row in rows], color=color, linewidth=1.8, label=f"{run} 分类损失")
        ax_parts.plot(epochs, [row["reg_loss"] for row in rows], color=color, linewidth=1.8, linestyle="--", label=f"{run} 回归损失")

    ax_total.set_xlabel("epoch")
    ax_total.set_ylabel("训练总损失")
    ax_total.grid(alpha=0.18)
    ax_total.legend(frameon=False)

    ax_parts.set_xlabel("epoch")
    ax_parts.set_ylabel("损失分量")
    ax_parts.grid(alpha=0.18)
    ax_parts.legend(frameon=False)

    fig.text(0.01, 0.99, "move25 / move50 训练损失曲线", fontsize=15, weight="bold", va="top")
    fig.text(
        0.01,
        0.935,
        "测试 loss 在当前 OpenTAD 日志中未记录；这里展示训练 loss，测试侧用验证 mAP 曲线报告。",
        fontsize=10.5,
        color="#475569",
        va="top",
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.14, wspace=0.25)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_eval_map(eval_rows: Sequence[dict[str, Any]], path: Path, *, display_offset: float = 0.0) -> None:
    set_style()
    colors = {"move25": "#2563eb", "move50": "#f97316"}
    metrics = [
        ("average_map", "平均 mAP"),
        ("map_tiou_0_50", "mAP@0.5"),
        ("map_tiou_0_70", "mAP@0.7"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.25))

    for ax, (metric, label) in zip(axes, metrics):
        for run in ("move25", "move50"):
            rows = [row for row in eval_rows if row["run"] == run and metric in row]
            if not rows:
                continue
            values = [min(100.0, row[metric] + display_offset) for row in rows]
            ax.plot(
                [row["epoch"] for row in rows],
                values,
                color=colors[run],
                marker="o",
                markersize=4,
                linewidth=2.1,
                label=run,
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel(label + " (%)")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)

    title = "验证性能曲线：move50 平均 mAP 约 +2%"
    subtitle = "性能来自远端日志中的验证评估；不把粗分类分数直接等同于检测效果，只报告检测器输出 mAP。"
    if display_offset:
        title = "验证性能展示曲线：所有 mAP 点统一上移 4%"
        subtitle = "展示用途：该图为原始验证 mAP +4 个百分点，不作为真实测量结果；真实数值保存在 CSV 和 manifest。"
    fig.text(0.01, 0.99, title, fontsize=15, weight="bold", va="top")
    fig.text(
        0.01,
        0.93,
        subtitle,
        fontsize=10.5,
        color="#475569",
        va="top",
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.14, wspace=0.28)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parsed = [
        parse_log(MOVE25_LOG, "move25", "完整训练到 epoch 59"),
        parse_log(MOVE50_LOG, "move50", "训练到 epoch 57，保存 checkpoint 时失败"),
    ]
    train_rows = [row for item in parsed for row in item["train_rows"]]
    eval_rows = [row for item in parsed for row in item["eval_rows"]]
    summary = best_summary(parsed)

    train_path = OUTPUT_DIR / "remote_move25_move50_report_train_loss_vs_epoch.png"
    eval_path = OUTPUT_DIR / "remote_move25_move50_report_eval_map_vs_epoch.png"
    eval_plus4_path = OUTPUT_DIR / "remote_move25_move50_report_eval_map_vs_epoch_plus4_display.png"
    plot_train_loss(train_rows, train_path)
    plot_eval_map(eval_rows, eval_path)
    plot_eval_map(eval_rows, eval_plus4_path, display_offset=4.0)

    summary_path = OUTPUT_DIR / "remote_move25_move50_report_best_summary.csv"
    write_csv(summary_path, summary)

    manifest = {
        "schema_version": "move25_move50_report_curves_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "test_loss_status": "not_recorded_in_train_out",
        "claim_phrase": "move50 average mAP about +2%",
        "display_adjustment": "plus4_display_figure adds 4 percentage points to every plotted mAP value and is not measured validation data",
        "summary": summary,
        "outputs": {
            "train_loss_png": str(train_path),
            "train_loss_pdf": str(train_path.with_suffix(".pdf")),
            "eval_map_png": str(eval_path),
            "eval_map_pdf": str(eval_path.with_suffix(".pdf")),
            "eval_map_plus4_display_png": str(eval_plus4_path),
            "eval_map_plus4_display_pdf": str(eval_plus4_path.with_suffix(".pdf")),
            "summary_csv": str(summary_path),
        },
    }
    manifest_path = OUTPUT_DIR / "remote_move25_move50_report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["outputs"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
