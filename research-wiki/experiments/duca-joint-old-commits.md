---
type: experiment
node_id: exp:duca-joint-old-commits
title: "DUCA fixed/MUST/JCT old-commit runs"
idea: idea:duca-jct
status: superseded
verdict: no
confidence: high
metrics: "Old-commit trends only; excluded from final paper evidence."
provenance: "docs/methods/2026-07-09-duca-jct-progressive-deployment.md"
added: 2026-07-11T00:00:00+08:00
---

# DUCA Old Joint Runs

## Scope

包括 7e3a508 及更早 fixed384/fixed256/fixed128、MUST384/320/256、JCT、budget curve 等运行。

## Why invalid for final claims

- commit 间 loss aggregation、boundary proxy、max-gap、hard/soft bridge、optimizer、uint8 与 DDP 行为不同；
- 部分 job 因 checkpoint、permission、dependency 或旧 suite 逻辑失败；
- 动态 route 未实现真实 variable detector compute。

## Verdict

只用于解释低性能、训练不稳定和实现缺陷。不得与 PhysTime 或修复后的 DUCA 结果混表。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
