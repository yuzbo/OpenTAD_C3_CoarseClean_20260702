---
type: idea
node_id: idea:chronotransport
title: "ChronoTransport 动态特征刷新"
stage: paused_after_failed_p3
outcome: negative_gate
tags: ["feature-refresh", "transport", "parallel-route"]
added: 2026-07-11
---

# ChronoTransport 动态特征刷新

## One-line thesis

保持外部 detector 网格，仅在 VideoMAE time×layer 上选择 RECOMPUTE/TRANSPORT/HOLD，减少 heavy subpath 重算。

## 为什么提出

避免 pre-backbone 删除帧引起 selected-axis 几何和 full decode 争议。

## 已有证据

Stage-A、paired replay 和正式 Stage-B fit/calibration/evaluation 已落地。`92029ea` 的预注册 P3 science gate 为 FAIL：risk-regret 排序为负，cell-risk/window-target 尺度错配，feature transport 改善不稳定；Stage C/P5 未解锁。

## 当前选择或否定理由

暂停，不作为当前主线。它证明 conditional-compute 工程闭环可运行，但没有证明 risk-certified transport 的科学有效性，也不能混用 DUCA 结果改写 claim。

## 风险与失败模式

transport 可能不优于 HOLD；cache 状态与校准；真实 kernel cost。

## 下一次允许采取的动作

只有重新定义风险聚合尺度、重新预注册并通过 P3，才允许启动 Stage C、多 seed 与 P5。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
