"""Publication plots from real run records. This module deliberately has no Torch dependency."""
import argparse
from collections import defaultdict
import csv
import html
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#444444"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
    "axes.prop_cycle": matplotlib.cycler(color=COLORS)})


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()] if Path(path).is_file() else []


def ecdf(ax, values, label=None, **kwargs):
    values = np.sort(np.asarray(values, dtype=float))
    if len(values):
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=label, **kwargs)


def heavy_macs(layers, width):
    return sum(12 * r["qkv_tokens"] * width**2 + 2 * r["qkv_tokens"]**2 * width for r in layers)


def intervals_union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def eligible_train(receipt):
    return receipt.get("status") == "completed" and receipt.get("is_mock") is False and receipt.get("completed_epochs") == 60 and receipt.get("best_checkpoint_selection_complete") is True


class Report:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def wait(self, name, reason, status="WAITING_DATA"):
        self.rows.append(dict(name=name, status=status, caption=reason))

    def save(self, name, fig, caption, rows=()):
        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(self.output / f"{name}.{extension}", bbox_inches="tight")
        plt.close(fig)
        self.rows.append(dict(name=name, status="DRAWN", caption=caption, png=f"{name}.png", pdf=f"{name}.pdf"))
        if rows:
            keys = list(dict.fromkeys(key for row in rows for key in row))
            with (self.output / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)

    def finish(self):
        (self.output / "figure_status.json").write_text(json.dumps(self.rows, indent=2, ensure_ascii=False), encoding="utf-8")
        tex = []
        cards = []
        for row in self.rows:
            caption = html.escape(row["caption"])
            if row["status"] == "DRAWN":
                cards.append(f'<article><h2>{html.escape(row["name"])}</h2><p>{caption}</p><a href="{row["pdf"]}">PDF</a> · <a href="{row["name"]}.csv">数据 CSV</a><img src="{row["png"]}"></article>')
                tex.append("\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{" + row["pdf"] + "}\n% Caption and provenance: figure_status.json\n\\end{figure}\n")
            else:
                label = "待接入专用诊断绘图" if row["status"] == "WAITING_IMPLEMENTATION" else "等待真实数据"
                cards.append(f'<article class="pending"><h2>{html.escape(row["name"])}</h2><b>{label}</b><p>{caption}</p></article>')
        (self.output / "figures.tex").write_text("\n".join(tex), encoding="utf-8")
        (self.output / "index.html").write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>GeoSparse 实验图册</title>
<style>body{font:16px system-ui;max-width:1120px;margin:40px auto;padding:0 24px;color:#223}h1{font-size:32px}article{border:1px solid #ccd6df;border-radius:12px;padding:22px;margin:22px 0}h2{font-size:19px;overflow-wrap:anywhere}p{line-height:1.7}img{width:100%;margin-top:18px}.pending{background:#f5f7fa}a{color:#0065a2}</style>
<h1>GeoSparse · 实验可视化</h1><p>图与 CSV 由同一份真实记录生成。训练过程图和正式结果分别标注；缺少数据的图保留等待状态。选取比例、Heavy MACs、模型主要算子 MACs、硬件延迟分别计量。</p>''' + "".join(cards) + '</html>', encoding="utf-8")


def plot_training(report, runs):
    found = False
    for path, job, receipt in runs.values():
        if job["kind"] != "train":
            continue
        rows = [r for r in jsonl(path / "train.log") if "losses" in r]
        if not rows:
            continue
        found = True
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
        values = np.arange(1, len(rows) + 1)
        csv_rows = []
        for color, (key, label) in zip(COLORS, (("cls_loss", "Classification"), ("reg_loss", "Regression"))):
            y = [r["losses"][key] for r in rows]
            axes[0].plot(values, y, lw=.6, alpha=.3, color=color)
            window = min(11, len(y))
            axes[0].plot(values[window - 1:], np.convolve(y, np.ones(window) / window, mode="valid"), label=label, lw=1.5, color=color)
        axes[0].set(xlabel="Training update attempt", ylabel="Training loss")
        axes[0].legend(frameon=False)
        validations = [read(p) for p in sorted((path / "intermediate_eval").glob("epoch_*/metrics.json"))]
        validations = [r for r in validations if r.get("status") == "completed_validation" and r.get("subset") == "validation" and r.get("is_mock") is False]
        if validations:
            epoch = [r["completed_epochs"] for r in validations]
            scores = [r["metrics"]["average_mAP"] * 100 for r in validations]
            axes[1].plot(epoch, scores, "o-", lw=1.4)
            best = int(np.argmax(scores))
            axes[1].scatter([epoch[best]], [scores[best]], marker="*", s=115, color=COLORS[1], label="Best so far")
            axes[1].annotate(f"{scores[best]:.3f}%", (epoch[best], scores[best]), xytext=(8, 10), textcoords="offset points", fontsize=8)
            axes[1].legend(frameon=False)
        else:
            axes[1].text(.5, .5, "Full validation in progress / pending\nNo measured mAP yet", ha="center", va="center", transform=axes[1].transAxes)
            axes[1].set_ylim(0, 100)
        axes[1].set(xlabel="Completed epoch", ylabel="Official split Avg-mAP (%)", xlim=(0, 60), ylim=(0, 100))
        seconds = [r["seconds"] for r in rows]
        ecdf(axes[2], seconds)
        axes[2].set(xlabel="Training update wall time (s)", ylabel="Empirical cumulative probability", ylim=(0, 1.02))
        for i, r in enumerate(rows):
            csv_rows.append(dict(record="training", update_attempt=i + 1, epoch=r["epoch"] + 1,
                cls_loss=r["losses"]["cls_loss"], reg_loss=r["losses"]["reg_loss"], seconds=r["seconds"], successful_update=r["successful_update"]))
        csv_rows.extend(dict(record="full_validation", epoch=r["completed_epochs"], average_mAP=r["metrics"]["average_mAP"], videos=len(r["videos"])) for r in validations)
        status = "60-epoch run completed" if eligible_train(receipt) else "INTERIM TRAINING PREVIEW"
        report.save(f'training_{job["job_id"]}', fig,
            f'{status}; {job["route"]}, seed {job["seed"]}; {len(rows)} recorded updates. Thin lines are raw losses; bold lines are up to 11-update trailing means. Validation uses all official evaluation videos every five epochs. Best is within this seed. Training step times are NOT inference latency.', csv_rows)
    if not found:
        report.wait("training", "需要真实 train.log；mAP 曲线需要逐轮全量验证 metrics.json。")


def plot_training_compute(report, runs):
    found = False
    for path, job, _ in runs.values():
        traces = jsonl(path / "cost_trace.jsonl")
        if not traces:
            continue
        found = True
        width = 1024 if job["model"]["backbone"] == "videomae_l" else 768
        rows = [dict(epoch=r["epoch"] + 1, step=r["step"], microbatch=r["microbatch"], video_ids=";".join(r["video_ids"]),
                     heavy_GMACs=heavy_macs(r["layers"], width) / 1e9,
                     selected_native_members=r.get("selected_native_members")) for r in traces]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        ecdf(axes[0], [r["heavy_GMACs"] for r in rows])
        axes[0].set(xlabel="Heavy Transformer GMACs / traced microbatch", ylabel="Empirical cumulative probability", ylim=(0, 1.02))
        layers = traces[0]["layers"]
        if layers:
            matrix = np.zeros((max(r["layer"] for r in layers) + 1, max(r["parent"] for r in layers) + 1))
            for r in layers:
                matrix[r["layer"], r["parent"]] = r["qkv_tokens"]
            im = axes[1].imshow(matrix, aspect="auto", origin="lower", cmap="cividis", vmin=0)
            fig.colorbar(im, ax=axes[1], label="QKV tokens / parent / layer")
        else:
            axes[1].text(.5, .5, "No Heavy operations", ha="center", transform=axes[1].transAxes)
        axes[1].set(xlabel="Original parent-clip index", ylabel="Heavy layer index")
        report.save(f'training_compute_{job["job_id"]}', fig,
            f'INTERIM TRAINING TRACE; {len(traces)} sampled microbatch forwards, not the full validation population. Heatmap shows the first traced forward. Patch embedding/Scout/TIA/head/decode are excluded from this formula; 1 MAC = 2 multiply/add FLOPs. Includes actual recorded QKV/MLP sizes, not requested budget.', rows)
    if not found:
        report.wait("training_compute", "需要 cost_trace.jsonl 中实际执行的逐层 QKV/MLP 长度。")


def export_inputs(roots):
    rows = []
    for root in roots:
        for path in sorted(Path(root).rglob("export_receipt.json")):
            receipt = read(path)
            if receipt.get("status") == "completed_selection_export" and receipt.get("is_mock") is False:
                rows.append((path.parent, receipt, jsonl(path.parent / "windows.jsonl")))
    return rows


def plot_allocation(report, exports):
    if not exports:
        report.wait("allocation_and_component_cost", "需要 selection_export 的精确选择和实测主要算子 MACs；当前训练 trace 没有完整 mask。")
    for path, receipt, windows in exports:
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
        for key, label in (("selected_native_ratio", "Selected native members"), ("selected_temporal_ratio", "Temporal slices"), ("heavy_mac_ratio", "Heavy MAC / full reference")):
            ecdf(axes[0], [w[key] for w in windows], label)
        axes[0].legend(frameon=False)
        axes[0].set(xlabel="Realized fraction", ylabel="Window cumulative probability", ylim=(0, 1.02))
        axes[1].scatter([w["heavy_macs"] / 1e9 for w in windows], [w["model_macs_counted"] / 1e9 for w in windows], s=10, alpha=.45)
        axes[1].set(xlabel="Heavy Transformer GMACs / window", ylabel="Counted model GMACs / window")
        components = sorted(set(k for w in windows for k in w["mac_components"]))
        axes[2].barh([k.replace("_", " ") for k in components], [np.mean([w["mac_components"].get(k, 0) for w in windows]) / 1e9 for k in components])
        axes[2].set(xlabel="Mean counted GMACs / window")
        table = [{k: w[k] for k in ("video_id", "window_index", "selected_native_ratio", "selected_temporal_ratio", "heavy_macs", "heavy_mac_ratio", "model_macs_counted", "mac_count_complete", "zero_heavy_parent_fraction")} for w in windows]
        report.save(f'allocation_{receipt["source_train_id"]}', fig,
            f'{len(windows)} windows, {len(receipt["videos"])} videos; final checkpoint={receipt["is_final_checkpoint"]}, full split={receipt["full_split"]}. C selection means fine refinement; coarse tokens still execute Heavy. Model MACs count executed Conv/Linear/Matmul and exclude normalization, softmax, activation, interpolation and preprocessing. Unsupported fused operations make the count a lower bound, flagged per row. This instrumented pass supplies NO latency.', table)


def plot_cases(report, exports):
    import base64
    from matplotlib.patches import Polygon
    from .selection_viewer import native_polygons
    found = False
    for path, receipt, windows in exports:
        annotations = read(path / "annotations.json")
        for row in windows:
            thumbnails = row.get("thumbnails", [])
            if not thumbnails:
                continue
            found = True
            choice = [thumbnails[i] for i in np.unique(np.linspace(0, len(thumbnails) - 1, min(3, len(thumbnails))).astype(int))]
            fig = plt.figure(figsize=(10, 6))
            grid = fig.add_gridspec(3, len(choice), height_ratios=[1, 1, .8])
            selected = np.unpackbits(np.frombuffer(base64.b64decode(row["selected_bits"]), dtype=np.uint8), bitorder="little")[:np.prod(row["native_shape"])].reshape(row["native_shape"])
            polygons = native_polygons(row["spatial_transform"], row["native_shape"][1])
            table = []
            for column, entry in enumerate(choice):
                t = entry["tubelet"]
                for member, frame_path in enumerate(entry["paths"]):
                    ax = fig.add_subplot(grid[member, column])
                    # Preserve the source image aspect while working in normalized coordinates.
                    picture = plt.imread(path / frame_path)
                    ax.imshow(picture, extent=(0, 1, 1, 0))
                    ax.set_aspect(picture.shape[0] / picture.shape[1])
                    for i, polygon in enumerate(polygons):
                        fine = bool(selected[t].flat[i])
                        if fine or row["route"] == "C":
                            ax.add_patch(Polygon(polygon, facecolor=COLORS[4] if fine else COLORS[0], alpha=.28, edgecolor="none"))
                    ax.set(xlim=(0, 1), ylim=(1, 0), xticks=[], yticks=[], xlabel=f'Native {t}, source frame {row["source_frame_id"][t][member]}')
                    table.append(dict(video_id=row["video_id"], window_index=row["window_index"], tubelet=t,
                                      source_frame_id=row["source_frame_id"][t][member], selected_fraction=row["temporal_selected_fraction"][t], image=frame_path))
            ax = fig.add_subplot(grid[2, :])
            valid_intervals = [s for pair, valid in zip(row["support_intervals_s"], row["support_valid"]) for s, v in zip(pair, valid) if v]
            start, end = min(s[0] for s in valid_intervals), max(s[1] for s in valid_intervals)
            for event in annotations[row["video_id"]].get("annotations", []):
                a, b = event["segment"]
                if a < end and b > start:
                    ax.broken_barh([(max(a, start), min(b, end) - max(a, start))], (1.12, .18), facecolors=COLORS[2])
            for t, (pair, validity) in enumerate(zip(row["support_intervals_s"], row["support_valid"])):
                for interval, valid in zip(pair, validity):
                    if valid and row["temporal_selected_fraction"][t] > 0:
                        ax.broken_barh([(interval[0], interval[1] - interval[0])], (0, row["temporal_selected_fraction"][t]), facecolors=COLORS[4])
            ax.set(xlim=(start, end), ylim=(0, 1.4), xlabel="Original physical time (s)", ylabel="Selection fraction / GT", yticks=[0, .5, 1, 1.22], yticklabels=["0", "0.5", "1", "GT"])
            report.save(f'case_{receipt["source_train_id"]}_{row["window_index"]:06d}', fig,
                f'Fixed qualitative example: {row["video_id"]}, window {row["window_index"]}, route {row["route"]}, epoch {receipt["selected_checkpoint_epoch"] + 1}, final={receipt["is_final_checkpoint"]}. Each column contains the actual source-frame pair. Orange marks selected native regions; C blue regions still have coarse Heavy. No object annotations are inferred. Timeline retains actual disjoint supports. Examples are fixed by video order and time, not selected for a favorable result.', table)
    if not found:
        report.wait("source_frames_and_selection", "需要 selection_export 导出的真实双帧和精确 mask，才生成原图叠加与论文案例图；没有使用示意图片填充。")


def plot_coverage(report, exports):
    if not exports:
        report.wait("boundary_support_coverage", "需要完整原始时间支持、有效 mask 与 GT，才测量每个端点到选中支持的距离；不从 selected rank 推算。")
    for path, receipt, windows in exports:
        annotation = read(path / "annotations.json")
        by_video = defaultdict(list)
        for row in windows:
            for fraction, pair, validity in zip(row["temporal_selected_fraction"], row["support_intervals_s"], row["support_valid"]):
                if fraction > 0:
                    by_video[row["video_id"]].extend(interval for interval, valid in zip(pair, validity) if valid)
        table = []
        for name in receipt["videos"]:
            support = intervals_union(by_video[name])
            for index, event in enumerate(annotation[name].get("annotations", [])):
                for side, point in zip(("start", "end"), event["segment"]):
                    distance = min((max(a - point, point - b, 0.) for a, b in support), default=None)
                    table.append(dict(video_id=name, gt_index=index, endpoint=side, duration_seconds=event["segment"][1] - event["segment"][0], nearest_selected_support_seconds=distance, no_selected_support=distance is None))
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
        for ax, side in zip(axes, ("start", "end")):
            rows = [r for r in table if r["endpoint"] == side]
            values = [r["nearest_selected_support_seconds"] for r in rows if not r["no_selected_support"]]
            ecdf(ax, values)
            ax.text(.97, .06, f'No selected support: {sum(r["no_selected_support"] for r in rows)}/{len(rows)}', transform=ax.transAxes, ha="right", fontsize=8)
            ax.set(xlabel=f"GT {side} to nearest selected support (s)", ylabel="Endpoint cumulative probability\n(conditional on some selected support)", ylim=(0, 1.02))
        report.save(f'boundary_support_{receipt["source_train_id"]}', fig,
            f'All GT endpoints in the export subset; full split={receipt["full_split"]}, final checkpoint={receipt["is_final_checkpoint"]}. Each selected tubelet contributes only its actual valid display intervals, never their hull. No-support cases remain explicit rather than becoming zero distance. This measures selected evidence/refinement time coverage; it is not action recognition or object coverage, and C also retains unplotted coarse evidence. Compare against matched-budget baselines before claiming task-value superiority.', table)


def formal_evaluations(runs):
    result = []
    for path, job, receipt in runs.values():
        if job["kind"] != "evaluate" or receipt.get("status") != "completed" or receipt.get("is_mock") is not False:
            continue
        training = runs.get(job["source_train_id"])
        if training is None or not eligible_train(training[2]):
            continue
        metrics = read(path / "metrics.json")
        if metrics["selected_checkpoint_epoch"] != training[2]["selected_checkpoint_epoch"]:
            raise ValueError("evaluation metrics and training best checkpoint differ")
        result.append((path, job, training[1], training[2], metrics))
    return result


def plot_risks(report, evaluations, exports):
    if not evaluations:
        report.wait("localization_risks", "需要正式 best checkpoint 的 metrics.json：短动作召回、漏检、边界误差和逐视频明细。")
    matched = False
    for _, job, train_job, training, metrics in evaluations:
        strata = metrics["duration_slices"]["strata"]
        names = [name for name in ("short_q1", "middle_q2_q3", "long_q4") if name in strata]
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
        axes[0].bar(names, [strata[name]["class_aware_recall"] for name in names])
        axes[0].set(ylabel="Class-aware recall", ylim=(0, 1), xlabel="Training-defined GT duration strata")
        for side in ("start", "end"):
            values = [r[f"capped_{side}_sum_normalized"] / r["gt_count"] for r in metrics["risk"]["per_video"].values() if r["gt_count"]]
            ecdf(axes[1], values, side)
        axes[1].set(xlabel="Per-video capped error / GT duration", ylabel="Video cumulative probability", ylim=(0, 1.02))
        axes[1].legend(frameon=False)
        table = [dict(video_id=name, **{k: row[k] for k in ("gt_count", "matches", "misses", "short_count", "short_matches", "capped_start_sum_normalized", "capped_end_sum_normalized")}) for name, row in metrics["risk"]["per_video"].items()]
        report.save(f'risks_{job["source_train_id"]}', fig,
            f'{train_job["route"]}, seed {job["seed"]}, best epoch {metrics["selected_checkpoint_epoch"] + 1}. Every missed GT contributes normalized capped endpoint error 1. Recall and endpoint matching definitions differ as recorded in metrics.json. No-GT videos are excluded from GT-normalized rates, not assigned zero. This is per-video risk, not per-video mAP.', table)
        for _, receipt, windows in exports:
            if receipt["source_train_id"] != job["source_train_id"] or not receipt["is_final_checkpoint"] or not receipt["full_split"]:
                continue
            if receipt["selected_checkpoint_epoch"] != metrics["selected_checkpoint_epoch"] or receipt["model_source_commit"] != training["source_commit"]:
                raise ValueError("cost export and evaluation used different checkpoints or model source")
            grouped = defaultdict(list)
            for row in windows:
                grouped[row["video_id"]].append(row)
            risk = metrics["risk"]["per_video"]
            if set(grouped) != set(risk):
                raise ValueError("cost and risk video populations differ")
            matched = True
            joined = [dict(video_id=name, windows=len(grouped[name]),
                mean_window_GMACs=np.mean([r["model_macs_counted"] for r in grouped[name]]) / 1e9,
                all_windows_GMACs=sum(r["model_macs_counted"] for r in grouped[name]) / 1e9,
                recall=r["matches"] / r["gt_count"], capped_boundary_error=(r["capped_start_sum_normalized"] + r["capped_end_sum_normalized"]) / (2 * r["gt_count"])) for name, r in risk.items() if r["gt_count"]]
            fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
            for ax, key, label in zip(axes, ("recall", "capped_boundary_error"), ("Per-video recall", "Per-video capped boundary error")):
                ax.scatter([r["mean_window_GMACs"] for r in joined], [r[key] for r in joined], s=12, alpha=.5)
                ax.set(xlabel="Mean counted GMACs / window", ylabel=label)
            report.save(f'cost_risk_{job["source_train_id"]}', fig,
                'Each point is one full evaluation video. Mean per-window cost avoids interpreting longer videos as harder routing. CSV also includes total cost over all overlapping inference windows. Correlation alone does not establish whether additional computation is useful.', joined)
    if not matched:
        report.wait("cost_performance_joint_distribution", "需要同一 best checkpoint、同一完整视频集合上的 selection_export 和风险指标，才能画计算量—召回/边界误差联合分布。")


def plot_hardware(report, runs):
    found = False
    for path, job, receipt in runs.values():
        if job["kind"] != "benchmark" or receipt.get("is_mock") is not False or not (path / "hardware.json").is_file():
            continue
        hardware = read(path / "hardware.json")
        measured = {(c["mode"], c["batch"], c["implementation"]) for c in hardware["cases"] if c["status"] == "MEASURED"}
        samples = jsonl(path / "timings.jsonl")
        cases = defaultdict(list)
        for row in samples:
            key = row["mode"], row["batch"], row["implementation"]
            if key in measured:
                cases[key].append(row["ms"])
        for mode in sorted({key[0] for key in cases}):
            found = True
            fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
            table = []
            for ax, size in zip(axes, (1, 8, 32)):
                for implementation in ("reference", "optimized"):
                    values = cases.get((mode, size, implementation), [])
                    if not values:
                        continue
                    ecdf(ax, values, f'{implementation}, B={size}')
                    table.extend(dict(mode=mode, batch=size, implementation=implementation, repeat=i, milliseconds=v) for i, v in enumerate(values))
                ax.set(xlabel=f"Batch {size} window pipeline time (ms)", ylabel="Timing cumulative probability", ylim=(0, 1.02))
                if ax.lines:
                    ax.legend(frameon=False)
            report.save(f'latency_{job["job_id"]}_{mode}', fig,
                f'Isolated hardware trials; {mode}; each unit is a batch of 768-position windows, not a complete long video. Only encoded_video_to_output includes source decode. Repeated timings do not measure training-seed uncertainty. Missing/OOM cases are retained in hardware.json.', table)
        if measured:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
            table = []
            for mode in sorted({key[0] for key in measured}):
                for implementation in ("reference", "optimized"):
                    rows = sorted((c for c in hardware["cases"] if c["status"] == "MEASURED" and c["mode"] == mode and c["implementation"] == implementation), key=lambda c: c["batch"])
                    if not rows:
                        continue
                    label = mode.replace("_to_output", "").replace("_model", "") + " / " + implementation
                    axes[0].plot([r["batch"] for r in rows], [r["throughput_windows_per_second"] for r in rows], "o-", label=label)
                    axes[1].plot([r["batch"] for r in rows], [r["peak_allocated_bytes"] / 2**30 for r in rows], "o-", label=label)
            for c in hardware["cases"]:
                table.append({k: c.get(k) for k in ("mode", "batch", "implementation", "status", "p50_ms", "p95_ms", "throughput_windows_per_second", "peak_allocated_bytes", "peak_reserved_bytes", "reason")})
            axes[0].set(xlabel="Batch size (windows)", ylabel="Throughput (windows / s)", xticks=[1, 8, 32])
            axes[1].set(xlabel="Batch size (windows)", ylabel="Peak allocated GPU memory (GiB)", xticks=[1, 8, 32])
            axes[0].legend(frameon=False, fontsize=6)
            report.save(f'hardware_scaling_{job["job_id"]}', fig,
                'Measured cases only; all failed/OOM cases and reserved memory are retained in the CSV. Different timing boundaries are separate series. These are single-GPU window-throughput measurements, not whole-video throughput or three-seed training variability.', table)
    if not found:
        report.wait("hardware_latency_throughput_vram", "需要隔离 GPU 的 timings.jsonl / hardware.json；训练 step 时间不能替代推理延迟。")


def plot_pareto(report, runs, evaluations, exports, single_seed=False):
    required = (0,) if single_seed else (0, 1, 2)
    prefix = "feasibility" if single_seed else "main"
    score_label = "Avg-mAP (%) · seed 0" if single_seed else "Official Avg-mAP (%) · 3-seed mean ± SD"
    groups = defaultdict(dict)
    for _, job, train_job, training, metrics in evaluations:
        if not set(train_job["families"]) & {"F00", "F01"}:
            continue
        key = json.dumps(dict(dataset=train_job["dataset"], model=train_job["model"], source=training["source_commit"], protocol=train_job.get("protocol_amendment")), sort_keys=True)
        groups[key][job["seed"]] = (train_job, training, metrics)
    points = []
    for seeds in groups.values():
        if not set(required).issubset(seeds):
            continue
        train_job, _, _ = seeds[0]
        costs = [(r, w) for _, r, w in exports if r["source_train_id"] == train_job["job_id"] and r["is_final_checkpoint"] and r["full_split"]]
        if not costs:
            continue
        receipt, windows = costs[0]
        if receipt["selected_checkpoint_epoch"] != seeds[0][2]["selected_checkpoint_epoch"] or receipt["model_source_commit"] != seeds[0][1]["source_commit"]:
            raise ValueError("Pareto cost did not use seed0's selected checkpoint")
        scores = [seeds[s][2]["official"]["average_mAP"] * 100 for s in required]
        label = f'{train_job["route"]} {train_job["model"]["axis"]} b={train_job["model"]["budget"]}'
        point = dict(label=label, dataset=train_job["dataset"], source_train_id=train_job["job_id"], mean_mAP=np.mean(scores),
            sd_mAP=None if single_seed else np.std(scores, ddof=1), seed_count=len(required),
            heavy_GMACs=np.mean([w["heavy_macs"] for w in windows]) / 1e9,
            model_GMACs=np.mean([w["model_macs_counted"] for w in windows]) / 1e9,
            complete_mac_count=all(w["mac_count_complete"] for w in windows), seed0_epoch=receipt["selected_checkpoint_epoch"] + 1)
        for benchmark_path, benchmark_job, benchmark_receipt in runs.values():
            if benchmark_job["kind"] != "benchmark" or benchmark_job["source_train_id"] != train_job["job_id"] or benchmark_receipt.get("is_mock") is not False or not (benchmark_path / "hardware.json").is_file():
                continue
            for case in read(benchmark_path / "hardware.json")["cases"]:
                if case["status"] == "MEASURED" and case["batch"] == 1 and case["implementation"] == "optimized":
                    point["gpu_model"] = case["gpu_start"].split(",")[1].strip()
                    point[case["mode"] + "_p50_ms"] = case["p50_ms"]
                    point[case["mode"] + "_p95_ms"] = case["p95_ms"]
        points.append(point)
    if not points:
        report.wait(prefix + "_accuracy_compute_pareto", "需要 seed 0 的 60 epoch 完整训练、best 全量评价及对应计算导出；单种子可行性图不报告跨种子标准差。" if single_seed else "需要每配置 seed 0/1/2 的 60 epoch 完整训练及各自 best 全量评价，另需 seed 0 对应 best 的计算导出。不会把单 seed 或训练中分数当成最终均值。")
        return
    for dataset in sorted({p["dataset"] for p in points}):
        rows = [p for p in points if p["dataset"] == dataset]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for ax, key, label in zip(axes, ("heavy_GMACs", "model_GMACs"), ("Heavy GMACs / window (seed 0)", "Counted model GMACs / window (seed 0)")):
            for p in rows:
                ax.errorbar(p[key], p["mean_mAP"], yerr=p["sd_mAP"], fmt="o", capsize=3, label=p["label"])
            frontier = sorted((p for p in rows if not any(q[key] <= p[key] and q["mean_mAP"] >= p["mean_mAP"] and (q[key] < p[key] or q["mean_mAP"] > p["mean_mAP"]) for q in rows)), key=lambda p: p[key])
            ax.plot([p[key] for p in frontier], [p["mean_mAP"] for p in frontier], "--", color=".6", lw=.8)
            ax.set(xlabel=label, ylabel=score_label)
        axes[0].legend(frameon=False, fontsize=7)
        report.save(f'{prefix}_pareto_{dataset}', fig,
            ('Single-seed route feasibility: seed0 best full-validation accuracy, without across-seed SD or stability claims. ' if single_seed else 'Accuracy uses each of three seeds’ own best full-validation checkpoint, then mean ± sample SD. ')
            + 'Cost uses seed0’s same selected checkpoint and all evaluation windows. Dashed line is an observed Pareto frontier, not a significance claim. Checkpoint selection uses the official evaluation split; these scores are not from an untouched test set. Incomplete operator counts are explicit lower bounds in CSV.', rows)
        timed = [p for p in rows if any(key.endswith("_p50_ms") for key in p)]
        if timed:
            for gpu in sorted({p["gpu_model"] for p in timed}):
                comparable = [p for p in timed if p["gpu_model"] == gpu]
                fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
                for ax, mode in zip(axes, ("device_model", "decoded_tensor_to_output", "encoded_video_to_output")):
                    for p in comparable:
                        key = mode + "_p50_ms"
                        if key in p:
                            ax.errorbar(p[key], p["mean_mAP"], yerr=p["sd_mAP"], fmt="o", capsize=3, label=p["label"])
                    ax.set(xlabel=mode.replace("_", " ") + "\np50 (ms / window), B=1", ylabel=score_label)
                axes[0].legend(frameon=False, fontsize=6)
                gpu_name = "".join(c if c.isalnum() else "_" for c in gpu)
                report.save(f'{prefix}_accuracy_latency_{dataset}_{gpu_name}', fig,
                    ('Single-seed feasibility accuracy' if single_seed else 'Three-seed accuracy') + f' versus measured optimized batch-1 p50 on {gpu}. GPU models are separate figures. Timing boundaries are separate panels; only encoded-video includes decoding. The associated CSV retains p95. A lower theoretical cost alone is not evidence of faster execution.', comparable)
        else:
            report.wait(f"{prefix}_accuracy_latency_{dataset}", "精度和计算数据已具备，但对应 seed0 best 的隔离硬件延迟尚未测得。")


def build(runs_root, selection_roots, output, only="all", single_seed=False):
    report = Report(output)
    runs = {}
    for path in Path(runs_root).rglob("job.json"):
        job = read(path)
        if "job_id" in job and "kind" in job:
            receipt = read(path.parent / "result.json") if (path.parent / "result.json").is_file() else {}
            runs[job["job_id"]] = (path.parent, job, receipt)
    exports = export_inputs(selection_roots)
    evaluations = formal_evaluations(runs)
    methods = {"training": lambda: plot_training(report, runs), "training_compute": lambda: plot_training_compute(report, runs),
        "allocation": lambda: plot_allocation(report, exports), "risk": lambda: plot_risks(report, evaluations, exports),
        "hardware": lambda: plot_hardware(report, runs), "pareto": lambda: plot_pareto(report, runs, evaluations, exports, single_seed),
        "cases": lambda: plot_cases(report, exports), "coverage": lambda: plot_coverage(report, exports)}
    for name, method in methods.items():
        if only in {"all", name}:
            method()
    if only == "all":
        report.wait("D01_D06_mechanism_diagnostics", "Utility 校准/Oracle regret、同计划 receiver、动态预算打乱、T×S 交互、内容贡献及缓存差异的统计图式已在 VISUALIZATION_PLAN.zh.md 固定；专用绘图适配器及对应真实诊断尚未完成，不能称全部论文图已准备就绪。", "WAITING_IMPLEMENTATION")
    report.finish()
    return report.rows


def main(only="all"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--selection-root", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--single-seed", action="store_true", help="Seed0 feasibility figures, without cross-seed SD")
    args = parser.parse_args()
    rows = build(args.runs_root, args.selection_root, args.output, only, args.single_seed)
    print(json.dumps(dict(drawn=sum(r["status"] == "DRAWN" for r in rows), waiting_data=sum(r["status"] == "WAITING_DATA" for r in rows), waiting_implementation=sum(r["status"] == "WAITING_IMPLEMENTATION" for r in rows)), ensure_ascii=False))


if __name__ == "__main__":
    main()
