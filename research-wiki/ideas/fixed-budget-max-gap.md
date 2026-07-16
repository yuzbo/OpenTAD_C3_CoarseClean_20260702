---
type: idea
node_id: idea:fixed-budget-max-gap
title: "Fixed budget anchor 与 max-gap 约束"
stage: active_anchor
outcome: mixed
tags: ["fixed-budget", "max-gap", "structured-policy"]
added: 2026-07-11
---

# Fixed budget anchor 与 max-gap 约束

## One-line thesis

fixed K 是归因与公平比较锚点；structured policy 应在可行路径内同时满足 exact budget 和最大未选间隔。

## 为什么提出

防止 top-k 过度聚集并保证 detector 最低时间覆盖。

## 已有证据

fixed 384/256/128、soft max-gap loss、scaffold/repair 与相关测试已实现过；7e3 修复 hard max-gap。

## 当前选择或否定理由

fixed-384 先作为主锚点；max-gap 应尽量在 policy 内编码并披露，不用后处理修复掩盖学习。

## 风险与失败模式

requested K 与 effective/unique K 不一致；低预算下 scaffold 占用过多 slot。

## 下一次允许采取的动作

核验每样本实际 backbone K，做 no-gap/soft/hard/soft+hard 消融。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
