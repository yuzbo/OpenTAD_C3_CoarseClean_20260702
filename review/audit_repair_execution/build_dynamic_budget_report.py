"""Build a measured-budget report from real frozen-EMA Scout replays."""
import csv
import html
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "dynamic_budget_analysis_20260908"
SOURCE = Path("E:/DeskTop/TAD/GeoSparse_Audit_Fixes_20260908/geosparse_ext")
COLORS = {"A": "#2166ac", "B": "#c87815", "C": "#19856b"}


def number(value, scale=1, digits=2):
    return "NA" if value is None else f"{value * scale:.{digits}f}"


def main():
    data = json.loads((HERE / "dynamic_budget_replay_summary.json").read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    rows = []
    complete = True
    reference_windows = None
    for cluster in ("N16R4", "A100"):
        host = data[cluster]
        complete &= bool(host["receipt"])
        cases = {(c["job_id"], c["completed_epochs"]): c for c in host["header"]["cases"]}
        for record in host["rows"]:
            case = cases[record["job_id"], record["completed_epochs"]]
            row = {k: v for k, v in record.items() if k not in {"raw_rows", "validation_receipt"}}
            row.update(cluster=cluster, route=case["config"]["route"], mode=case["config"]["budget_mode"],
                       source_commit=host["header"]["source_commit"], checkpoint_path=case["checkpoint_path"],
                       config=case["config"], provenance=case["provenance"])
            row["label"] = f"{row['route']} {'dynamic' if row['mode']=='dynamic' else 'fixed .5'} / E{row['completed_epochs']}"
            row["mAP_percent"] = (row["metrics"]["average_mAP"] * 100) if row["metrics"] else None
            raw = record.get("raw_rows")
            if host["receipt"] and raw is not None:
                if len(raw) != 792 or [r["window_index"] for r in raw] != list(range(792)):
                    raise ValueError("incomplete window export")
                if {r["video_id"] for r in raw} != set(host["header"]["videos"]):
                    raise ValueError("video coverage differs")
                cohort = [(r["window_index"], r["video_id"], r["window_id"], r["valid_frames"]) for r in raw]
                if reference_windows is None:
                    reference_windows = cohort
                elif cohort != reference_windows:
                    raise ValueError("paired methods do not share the same windows and valid input lengths")
                filename = f"{row['job_id']}.epoch_{row['completed_epochs']:03d}.windows.jsonl"
                (OUT / filename).write_text("".join(json.dumps(r) + "\n" for r in raw), encoding="utf-8")
                row["local_windows"] = filename
            if host["receipt"]:
                (OUT / f"receipt.{cluster}.json").write_text(json.dumps(host["receipt"], indent=2), encoding="utf-8")
            rows.append(row)
    rows.sort(key=lambda r: (r["route"], r["completed_epochs"], r["mode"] != "dynamic"))
    status = "completed_budget_analysis" if complete else "partial_budget_analysis"
    summary = dict(status=status, observed_at_utc=data["observed_at_utc"],
                   scope="Actual EMA Scout choices on official test pixels; analytical Heavy MACs; accuracy from existing full EMA validation.",
                   rows=rows, training_changed=False, heavy_forward_executed=False, latency_measured=False)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["label", "route", "mode", "completed_epochs", "windows", "videos", "mAP_percent",
              "full_budget_fraction", "heavy_valid_weighted", "heavy_padded_mean", "requested_budget_mean",
              "probability_expected_budget_mean", "job_id", "checkpoint_sha256", "validation_receipt_path"]
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.05, 1]})
    labels = [r["label"] for r in rows]
    bottoms = np.zeros(len(rows))
    for q, color in zip([0, .25, .5, .75, 1], ["#e0e4eb", "#8ebbd8", "#3488ba", "#e7ac4d", "#bb473f"]):
        shares = np.array([sum(n for value, n in r["budget_hist"].items() if float(value) == q) / r["windows"] for r in rows])
        left.barh(labels, shares, left=bottoms, color=color, label=f"q={q:g}")
        for i, share in enumerate(shares):
            if share > .12:
                left.text(bottoms[i]+share/2, i, f"{100*share:.0f}%", ha="center", va="center", color="white" if q>=.5 else "#25364b")
        bottoms += shares
    left.invert_yaxis()
    left.set(xlim=(0, 1), xlabel="Fraction of observed validation windows", title="Actual executed budget choices")
    left.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.13), frameon=False)
    right.axvline(1, color="#777777", ls="--", lw=1, label="All eligible native tokens")
    for row in rows:
        if row["mAP_percent"] is None:
            continue
        x, y = row["heavy_valid_weighted"], row["mAP_percent"]
        right.scatter(x, y, s=78, marker="o" if row["mode"]=="dynamic" else "s", color=COLORS[row["route"]])
        offset = (-8, 8) if x>.9 else (7, 7)
        if row["route"]=="A" and row["mode"]=="dynamic" and row["completed_epochs"]==30:
            offset=(-8,-14)
        if row["route"]=="C" and row["mode"]=="fixed":
            offset=(7,-14)
        right.annotate(row["label"], (x,y), xytext=offset, textcoords="offset points", fontsize=8,
                       ha="right" if x>.9 else "left")
    right.set(xlim=(0,1.12), ylim=(0,80), xlabel="Heavy MAC / full eligible-token Heavy MAC", ylabel="Average mAP (%)",
              title="Accuracy versus actual-plan Heavy cost")
    right.grid(alpha=.2)
    fig.suptitle("Frozen EMA budget audit" + (" — PARTIAL coverage" if not complete else " — 211 videos / 792 windows per case"), fontsize=14)
    fig.text(.5,.015,"Analytical QKV / attention / MLP only; excludes Scout, TIA, receiver, decode and latency. Epochs are shown explicitly.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.94))
    for extension in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"budget_and_accuracy.{extension}", dpi=180)
    plt.close(fig)

    stamp = datetime.fromisoformat(data["observed_at_utc"]).astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    table_headers = ["配置 / 完成轮数", "覆盖窗口", "mAP %", "实际预算分布", "全量档占比", "有效全量 Heavy 比例", "含 padding 稠密参考比例"]
    table_rows = []
    for row in rows:
        hist = "；".join(f"q={float(q):g}: {n}" for q,n in sorted(row["budget_hist"].items(), key=lambda item:float(item[0])))
        table_rows.append([row["label"],f"{row['windows']}/792",number(row["mAP_percent"]),hist,
                           number(row["full_budget_fraction"],100)+"%",number(row["heavy_valid_weighted"],100)+"%",
                           number(row["heavy_padded_mean"],100)+"%"])
    md_table = "|"+"|".join(table_headers)+"|\n|"+"|".join(["---"]*len(table_headers))+"|\n"+"\n".join("|"+"|".join(r)+"|" for r in table_rows)
    control_headers = ["动态检查点", "训练成本EMA（含padding参考）", "惩罚系数λ", "验证softmax期望q（非MAC）", "验证实际平均q"]
    control_rows = [[r["label"],number(r["controller"]["cost_ema"],100)+"%",number(r["controller"]["dual"],digits=5),
                     number(r["probability_expected_budget_mean"],100)+"%",number(r["requested_budget_mean"],100)+"%"]
                    for r in rows if r["mode"]=="dynamic"]
    control_md = "|"+"|".join(control_headers)+"|\n|"+"|".join(["---"]*len(control_headers))+"|\n"+"\n".join("|"+"|".join(r)+"|" for r in control_rows)
    explanations = [
        "测量边界：读取冻结的 b70ae056 EMA 检查点，用官方验证像素、原 Scout、原 make_plan 复放选档和选择；没有重训，没有重跑 Heavy 或重算 mAP。mAP 来自同一训练 ID、完成轮数、EMA 和数据划分的完整验证收据。",
        "有效全量 Heavy 比例 = 全部窗口实际计划的 Heavy MAC 总和 / 同一批有效 native token 全量执行的 Heavy MAC 总和。含 padding 参考仍支付48个800-token parent；两者的差额属于无效 padding，不是内容自适应的选择收益。",
        "Heavy 计数采用冻结执行器的 QKV、attention 和 MLP 规则：12层 × Σparent(12 k d² + 2 k² d)，d=768。A/B 的 k 是被选中的有效 native token 数；C 的 k=有效 native数/4+3×fine成员数/4。该数不是完整模型 FLOPs，也不证明真实加速。",
        "当前损失确实控制成本：R=L_TAD+λ(c−0.5)，actor使用停止梯度的优势乘采样log probability；critic拟合R。成功更新后，cost EMA以0.1更新，λ←clip(λ+0.001(costEMA−0.5),0,10)。这是训练随机策略的软平均约束。",
        "验证直接argmax预算logits，并未施加实际成本约束。最大档概率最高，不要求其概率接近1；即使训练中不同预算混合，贪心验证也可能所有窗口选满。不能把mean softmax预算、抽样训练预算和验证实际成本视为同一指标。",
        "现有训练cost_trace只抽记每轮首批和发生acquisition probe的批次。全选时无可追加候选，部分批次不会进入这类trace；因此它不是无偏的全训练预算统计。本次没有用抽样训练trace代替验证分布。",
        "判读：预算=1且有效Heavy比例=1，表示所有有效位置都执行Heavy；该点可保留其真实精度，但不能用于50%成本或内容自适应收益主张。固定q=.5也不等于Heavy MAC=.5，尤其C的q是mixed-token比例。A第40轮与第30轮分开列出，方法增量优先看同轮且同成本比较。",
        "修复方向：将预算可行性落实到训练与推理使用的实际分配器，并明确是每视频平均成本还是逐窗口上限；允许难窗口取1、其他窗口少算，但整体必须满足既定成本。不能只在测试上调大λ、删除最大档或用全量精度冒充动态优势。",
        "最快的定位实验是复用现有动态EMA，在另一个有明确身份的诊断运行中固定q=.5，重做全211视频评价，分离权重训练收益与验证追加计算收益。它尚未执行，不计作新的训练或正式修订模型结果。旧训练和检查点保持原实现身份。",
    ]
    heading = "全量核查完成" if complete else "核查进行中：以下成本分布是部分窗口，不能冒充全量统计"
    source_links = [("训练成本目标", SOURCE/"routing.py",233),("验证选档",SOURCE/"routing.py",185),
                    ("自适应惩罚系数",SOURCE/"detector.py",137),("Heavy成本分母",SOURCE/"model.py",103)]
    text = "# 动态预算是否逼近全量计算\n\n"+heading+"\n\n观察时间："+stamp+" 北京时间。\n\n"+md_table+"\n\n"+control_md+"\n\n"
    text += "\n\n".join(explanations)+"\n\n"
    text += "源码依据："+"；".join(f"[{name}]({path.as_posix()}:{line})" for name,path,line in source_links)+"。\n\n"
    text += "原始窗口数据与检查点身份见同目录 summary.json、各 *.windows.jsonl 与 receipt.*.json；NA表示尚无配对完整指标。\n"
    (OUT / "REPORT.zh.md").write_text(text, encoding="utf-8")
    table = "<table><thead><tr>"+"".join("<th>"+html.escape(t)+"</th>" for t in table_headers)+"</tr></thead><tbody>"
    table += "".join("<tr>"+"".join("<td>"+html.escape(v)+"</td>" for v in r)+"</tr>" for r in table_rows)+"</tbody></table>"
    control_table = "<table><tr>"+"".join("<th>"+html.escape(v)+"</th>" for v in control_headers)+"</tr>"
    control_table += "".join("<tr>"+"".join("<td>"+html.escape(v)+"</td>" for v in r)+"</tr>" for r in control_rows)+"</table>"
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>GeoSparse 动态预算核查</title><style>body{font:15px/1.7 system-ui;margin:32px auto;max-width:1400px;padding:0 24px;color:#172b42;background:#f4f7fb}section{padding:24px;background:white;border-radius:10px;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #dbe3ec}img{max-width:100%}.notice{border-left:5px solid #b94740}a{color:#246ab0}code{font-size:12px}</style>'
    page += f"<h1>动态预算是否逼近全量计算</h1><p>{stamp} 北京时间 · 冻结实现 b70ae056 · seed0</p><section class='notice'><b>{heading}</b><p>配置中的50%是训练目标；实际验证是否少算，以下列计划和成本为准。</p></section>"
    page += "<section>"+table+"</section><section><img src='budget_and_accuracy.svg' alt='预算选档分布与准确率—Heavy成本图'></section>"
    page += "<section><b>软约束、概率分布和实际执行是不同量</b>"+control_table+"<p>训练成本EMA是检查点中的训练监测状态；验证两列来自EMA Scout的真实输出。二者数据、随机策略与权重状态不同，不能直接替代验证成本。</p></section>"
    page += "<section>"+"".join("<p>"+html.escape(p)+"</p>" for p in explanations)+"</section>"
    page += '<section><a href="REPORT.zh.md">完整说明</a> · <a href="summary.csv">汇总 CSV</a> · <a href="summary.json">检查点身份与数据</a> · <a href="budget_and_accuracy.pdf">PDF 图</a></section></html>'
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps(dict(status=status,path=str(OUT),cases=len(rows),windows=[r["windows"] for r in rows]),ensure_ascii=False))


if __name__ == "__main__":
    main()
