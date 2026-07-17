---
type: experiment
node_id: exp:phystime-g1-matched-medium20
title: "PhysTime G1 native-J192 matched three-arm 20-epoch comparison"
idea: idea:phystime-tad-2
status: experiment_running
verdict: pending_matched_medium_results
confidence: pending
metrics: "NA until all three MEDIUM_COMPLETE.json artifacts pass; raw values belong in docs/evaluation/results.md."
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
- Selected-axis: job `1168485`
- Physical-metric: job `1168486`
- G1b SDPQ: job `1168487`

Shared gate 已确认 exact commit/tree、K384、J192、无 feature interpolation、所有 GT 与短 GT 均有 assignment。三条训练均由 `afterok:1168484` 释放并进入 epoch 0。

## Evidence Boundary

当前状态仅为 `experiment_running`。在三条 `MEDIUM_COMPLETE.json` 均通过、指标由独立 evaluator 重算、online/EMA checkpoint 均有限且可重放之前，Avg-mAP 为 NA，不能判断方法优越性，不能启动 60-epoch full train。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
