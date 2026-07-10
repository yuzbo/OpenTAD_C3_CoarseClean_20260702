---
type: experiment
node_id: exp:duca-a5e-cost
title: "a5e1774 DUCA full-stack cost audit"
idea: idea:duca-offline-full-window
verdict: partial
confidence: high
commit: "a5e1774b9941312569ca645341da1abad339db61"
jobs: "1156079"
updated: 2026-07-11
---

# a5e1774 DUCA full-stack cost audit

## Raw metrics / observations

本地契约/成本测试 40 passed；远端完整 DUCA 141 passed；成本 precheck 21 passed；GPU random-init smoke 完成。

## Interpretation

证明成本统计链路与 contract 可运行，并修复 raw-pixel structured bridge 漏记、GPU UUID 和功耗积分问题。

## Limitations

random-init、2 sample、dirty scratch 的 latency/energy 数字不能进入论文；尚无 trained checkpoint 正式矩阵。

## Provenance

docs/methods/2026-07-10-70aa069-researchclaw-duca-divergent-audit-absorption.md; Job 1156079

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
