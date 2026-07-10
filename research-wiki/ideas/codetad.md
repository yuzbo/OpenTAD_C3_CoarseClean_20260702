---
type: idea
node_id: idea:codetad
title: "CoDeTAD GOP-dependent partial decode"
stage: archived
outcome: pending
thesis: "按 GOP reference dependency 和定位 distortion 决定 partial decode 模式。"
risks: "codec API/hardware 依赖；逻辑帧请求不等于真实 decode saving；跨 codec 泛化困难。"
based_on: []
target_gaps: ["gap:G7"]
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# CoDeTAD

## 与 CoDeR-TAL 的关系

CoDeTAD 来自 ResearchClaw 24-idea 集，CoDeR-TAL 来自后续 23-idea 集；两者都以 codec-native localization rate-distortion 为核心，但命名、细节与评审轮次不同，Wiki 分开保留。

## 当前裁决

未实现。只有 profiling 证明 decode/GOP dependency 是主要成本后才允许恢复。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
