---
type: idea
node_id: idea:duca-must
title: "DUCA-MUST dynamic budget"
stage: paused
outcome: negative
tags: ["dynamic-budget", "must", "primal-dual"]
added: 2026-07-11
---

# DUCA-MUST dynamic budget

## One-line thesis

学习每个窗口的预算 K，并用 primal-dual/成本约束形成连续 budget-quality curve。

## 为什么提出

最终目标是任务感知动态采集，而不是永远固定 384。

## 已有证据

MUST controller、target curve 和多个历史 full run 已部署；早期出现 64/384 跳变与低 mAP。

## 当前选择或否定理由

暂不主打 dynamic。固定策略站稳且真实 K/成本闭环后再恢复。

## 风险与失败模式

expected K 不等于执行 K；clamp/binary jump；budget loss 与 detector utility 冲突。

## 下一次允许采取的动作

若恢复，先在 384/320/256 target 稳定，使用 same-mean fixed baseline 和真实 latency dual。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
