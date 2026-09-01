---
type: experiment
node_id: exp:phystime-g1-matched-medium20
title: "PhysTime G1 native-J192 matched three-arm 20-epoch comparison"
idea: idea:phystime-tad-2
status: completed
verdict: physical_metric_supported_sdpq_not_supported
confidence: single_seed_matched_medium
metrics: "Raw values are recorded only in docs/evaluation/results.md: physical-metric 44.88 Avg-mAP, selected-axis 30.42, G1b SDPQ 30.88."
provenance: "/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1_matched_5e8a821_medium20_20260717_132000_0800"
added: 2026-07-17T13:30:00+08:00
---

# PhysTime G1 Matched Three-Arm Medium Comparison

## Question

在完全相同的 commit、数据、K384/J192、无 GT sampler、seed=42、20 epochs、优化器、scheduler 和评价协议下，G1b SDPQ 是否优于官方 ActionFormer selected-axis 与 physical-metric controls？

## Fixed Arms

1. `selected_axis`：官方 ActionFormer，uniform-rank-derived seconds；
2. `physical_metric`：官方 ActionFormer，真实 physical seconds；
3. `g1b_sdpq`：support-decoupled physical query sparse head。

三臂都不做 feature interpolation，使用同一 VideoMAE checkpoint。最终轻量 checkpoint 保存 online 与 EMA 权重，排除 optimizer/scheduler，并要求预测、IoU-wise mAP 与 checkpoint 独立验证。

## Deployment

- Code commit: `5e8a8219c27785c15d720c5ed3c6b37298a2a866`
- Git tree: `7dfdf3d1c1e1c681a5df23f5916e2aa53de221ea`
- Clean snapshot: `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1_matched_5e8a821_20260717`
- Shared gate: job `1168484`, `COMPLETED 0:0`
- Selected-axis: job `1168485`, `COMPLETED 0:0`
- Physical-metric: job `1168486`, `COMPLETED 0:0`
- G1b SDPQ: job `1168487`, `COMPLETED 0:0`

Shared gate 已确认 exact commit/tree、K384、J192、无 feature interpolation、所有 GT 与短 GT 均有 assignment。三条训练均由 `afterok:1168484` 释放并完成 20 epochs；三个 `MEDIUM_COMPLETE.json`、独立 evaluator 和 online/EMA checkpoint validator 均通过。

## Result And Attribution

原始数值只登记在 `docs/evaluation/results.md`。决定性差值如下：

- physical-metric 相对 selected-axis：Avg-mAP `+14.46`，mAP@0.7 `+9.82`；
- G1b SDPQ 相对 selected-axis：Avg-mAP `+0.46`，mAP@0.7 `+2.42`；
- G1b SDPQ 相对 physical-metric：Avg-mAP `-14.00`。

因此，本实验强支持“在真实物理时间度量中进行 ActionFormer assignment/回归”这一机制，但不支持“当前 SDPQ 结构优于 matched ActionFormer physical-metric control”。G1b 在高 IoU 上有小幅信号，同时损失低 IoU 覆盖，仍是诊断候选而非主方法。

## Evidence Boundary

三臂均为 `runnable`、gate-passed 且完成 matched medium；physical-metric 达到 `matched-medium-supported`。由于仍只有 THUMOS 单数据集、单 seed、20 epochs，且缺完整成本与误差分解，任何方法都不是 `paper_ready`。60-epoch full train 不自动启动；下一阶段应优先复现 physical-metric survivor，而不是直接放大 G1b。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
