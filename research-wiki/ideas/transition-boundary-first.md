---
type: idea
node_id: idea:transition-boundary-first
title: "Transition / boundary / utility-first selector"
stage: active_component
outcome: pending
tags: ["transition", "boundary", "selector"]
added: 2026-07-11
---

# Transition / boundary / utility-first selector

## One-line thesis

selector 显式读取粗 probe 隐藏特征、delta/abs-delta、不确定性，并让状态转换、边界和 utility 优先于 actionness。

## 为什么提出

历史结果说明 actionness coverage 容易选动作内部；边界覆盖与 high-tIoU 更相关。

## 已有证据

544eca6 后评分与监督已改为 transition/boundary-first，actionness 权重降为辅助。

## 当前选择或否定理由

这是当前设计初心的强约束，不能再退回 actionness-top-k。

## 风险与失败模式

GT boundary proxy 可能过拟合标注；transition peak 受粗 probe 平滑和相机运动影响。

## 下一次允许采取的动作

做 actionness weight、hidden feature、boundary proxy、detector-gradient 消融与边界距离分析。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
