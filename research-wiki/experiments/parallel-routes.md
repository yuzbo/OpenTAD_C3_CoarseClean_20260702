---
type: experiment
node_id: exp:parallel-routes
title: "ChronoTransport / PhysTime parallel route gates"
idea: idea:chronotransport
verdict: pending
confidence: medium
commit: "78d4c00 and PhysTime branches"
jobs: "1157170 and dependent PhysTime jobs; ChronoTransport gates"
updated: 2026-07-11
---

# ChronoTransport / PhysTime parallel route gates

## Raw metrics / observations

代码、precheck、数据恢复和部分 paired replay 已部署；正式 paper-level 多 seed 与 matched result 尚未完成。

## Interpretation

它们是对 DUCA frame dropping 成本/几何风险的独立替代假设，不能与 DUCA 证据合并。

## Limitations

仍需真实成本、官方 detector parity、risk/regret 或 physical-time 收益门槛。

## Provenance

docs/methods/2026-07-10-chronotransport-implementation-plan.md; PhysTime branches and Slurm queue

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
