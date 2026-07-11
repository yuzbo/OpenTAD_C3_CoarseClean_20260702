---
type: experiment
node_id: exp:parallel-routes
title: "ChronoTransport / PhysTime parallel route gates"
idea: idea:chronotransport
verdict: mixed
confidence: medium
commit: "92029ea and PhysTime branches"
jobs: "1157170 and dependent PhysTime jobs; ChronoTransport gates"
updated: 2026-07-11
---

# ChronoTransport / PhysTime parallel route gates

## Raw metrics / observations

ChronoTransport 已完成正式单种子 Stage-B，但 P3 science gate 失败；PhysTime 仍处于独立设计/实现与 gate 推进状态。两条路线不能合并成一个 pending 成功信号。

## Interpretation

它们是对 DUCA frame dropping 成本/几何风险的独立替代假设，不能与 DUCA 证据合并。

## Limitations

ChronoTransport 需先修复风险尺度并重新通过 P3；PhysTime 仍需真实数据、官方 detector parity 与 matched physical-time 收益门槛。

## Provenance

docs/methods/2026-07-10-chronotransport-implementation-plan.md; PhysTime branches and Slurm queue

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
