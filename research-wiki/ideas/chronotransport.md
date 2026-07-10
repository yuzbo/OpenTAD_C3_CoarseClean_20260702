---
type: idea
node_id: idea:chronotransport
title: "ChronoTransport 动态特征刷新"
stage: parallel_active
outcome: pending
tags: ["feature-refresh", "transport", "parallel-route"]
added: 2026-07-11
---

# ChronoTransport 动态特征刷新

## One-line thesis

保持外部 detector 网格，仅在 VideoMAE time×layer 上选择 RECOMPUTE/TRANSPORT/HOLD，减少 heavy subpath 重算。

## 为什么提出

避免 pre-backbone 删除帧引起 selected-axis 几何和 full decode 争议。

## 已有证据

Stage-A/B、paired replay、risk/regret gates 和部署代码已落地，正式 paper gate 尚未闭环。

## 当前选择或否定理由

作为与 C3/DUCA 并行路线，不能混用 DUCA 结果或改写其 claim。

## 风险与失败模式

transport 可能不优于 HOLD；cache 状态与校准；真实 kernel cost。

## 下一次允许采取的动作

先过 transport-vs-HOLD、risk-regret correlation、held-out calibration，再启动多 seed。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
