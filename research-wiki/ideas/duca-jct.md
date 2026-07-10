---
type: idea
node_id: idea:duca-jct
title: "DUCA-JCT collaborative joint training"
stage: archived
outcome: negative
thesis: "在单次训练作业内协调 coarse actionness、selector、detector gradient 与预算约束。"
risks: "progressive schedule 可隐藏目标冲突；nonzero gradient 不等于正确 hard policy；旧 runs 受实现缺陷污染。"
based_on: []
target_gaps: []
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# DUCA-JCT

## 训练设想

detector 从 step 0 训练；actionness 早期较强；detector-to-selector bridge、gap 与 budget loss 按 optimizer step 渐进打开。目标是避免三个独立 checkpoint。

## 教训

- schedule 必须按真实 optimizer step，不按 forward 次数。
- detector loss 与 selector auxiliary loss 必须单点聚合。
- 早期 stable input 不能使 DDP 后期出现新 unused parameters。
- soft branch 必须是 hard policy 的 relaxation。

## 当前裁决

单作业协同训练思想可复用，但该路线效果低于分离训练且因 loss duplication、proxy domination 与 hard/soft mismatch 被否定为主线。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
