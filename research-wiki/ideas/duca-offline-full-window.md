---
type: idea
node_id: idea:duca-offline-full-window
title: "DUCA 离线全窗口 pre-backbone 插件"
stage: active_candidate
outcome: pending
tags: ["duca", "offline-full-window", "pre-backbone"]
added: 2026-07-11
---

# DUCA 离线全窗口 pre-backbone 插件

## One-line thesis

同一 forward 内由可训练 coarse probe 和结构化 selector 生成 hard sparse observations，送入 official-derived TAD detector，并接受 detector feedback。

## 为什么提出

将粗分类、间接选择和检测放入统一训练图，同时保持推理无 ledger、无 teacher、无 cache。

## 已有证据

核心 selector、official backend config、joint loss、optimizer coverage、max-gap、监控和 full-stack profiler 已实现；70aa fixed-384 正在正式训练。

## 当前选择或否定理由

冻结为待裁决完整 baseline，不再堆模块；通过成本、surrogate、geometry 和 matched baseline 后才可成为主方法。

## 风险与失败模式

full-window probe 成本、selected-axis geometry、ST mismatch、effective K 波动、只在 AdaTAD 上验证。

## 下一次允许采取的动作

完成 G1/G2/G3/G4/G6/G8 决定性门槛。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
