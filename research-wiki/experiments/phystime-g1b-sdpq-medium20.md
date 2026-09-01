---
type: experiment
node_id: exp:phystime-g1b-sdpq-medium20
title: "PhysTime G1b SDPQ 20-epoch medium run"
idea: idea:phystime-tad-2
status: empirically_supported
verdict: medium_run_trainability_supported_but_superiority_unresolved
confidence: medium
metrics: "Authoritative raw values are recorded only in docs/evaluation/results.md."
provenance: "/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1b_sdpq_4a57577_gtboundaryfix_medium20_20260716_190900_0800"
added: 2026-07-17T13:10:00+08:00
---

# PhysTime G1b SDPQ 20-Epoch Medium Run

## Question

在 K384 原始帧、J192 原生 tubelet token、物理 query 与稀疏 support 解耦的条件下，G1b SDPQ 能否稳定训练 20 轮，并持续改善检测性能？

## Contract

- 离线 THUMOS14 raw-video TAD，不是在线任务；
- 无学习、无 GT 的固定不规则采样；
- K384 raw observations、J192 native tokens，不做 feature interpolation；
- 物理秒坐标用于 GT、assignment、回归、decode、NMS 与评价；
- GT/window end-exclusive 越界采用可审计 clamp/filter，不允许随机 DataLoader 崩溃；
- real gate、训练 commit/tree/config/checkpoint 与独立 evaluator 逐级绑定；
- 20 个 epoch，最终轻量 checkpoint 和结构化完成 artifact 必须同时通过验证。

## Evidence

Gate 与 medium run 均已完成，训练过程稳定，最终预测与指标可由独立 evaluator 重算。原始 mAP 只见 `docs/evaluation/results.md`。

该证据只支持两点：G1b 已从“能运行”提升为“20 轮训练有效且持续学习”；GT/window 与 final-only checkpoint 修复形成了稳定闭环。

残余 artifact 缺口：该 run 的轻量 checkpoint 只保留 online `state_dict`，而验证使用 EMA 权重，因此精确 evaluated-weight replay 不可用。新的三臂 suite 必须同时保存 online/EMA state dict，并继续排除 optimizer/scheduler。

## Limit

现有 G1a selected-axis 与 physical-metric 结果来自更早 commit 和六轮训练，不能与本实验直接比较。因此，G1b 是否优于 matched controls 仍是未知，当前不能称 `paper_ready`，也不能据此启动论文正式 full train。

## Next Gate

在同一新 commit 下运行 selected-axis、physical-metric 与 G1b SDPQ 三臂 20-epoch 对照，统一数据、K/J、采样、seed、优化器、scheduler、评价器和 checkpoint 合同。只有该比较给出稳定且有意义的优势，才解锁 60-epoch full train。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
