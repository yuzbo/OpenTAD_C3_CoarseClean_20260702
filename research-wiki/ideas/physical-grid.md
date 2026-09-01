---
type: idea
node_id: idea:physical-grid
title: "Physical-grid / continuous-time ActionFormer"
stage: deferred
outcome: unknown
tags: ["geometry", "physical-time", "actionformer"]
added: 2026-07-11
---

# Physical-grid / continuous-time ActionFormer

## One-line thesis

让 detector point、卷积或 regression range 感知 selected positions 的真实时间间隔。

## 为什么提出

post-hoc inverse map 无法恢复 selected-axis 内部卷积与 assignment 的物理时间语义。

## 已有证据

存在设计、precheck 和可选源码扩展；当前 DUCA 主配置未启用。

## 当前选择或否定理由

明确风险但暂缓实现，避免在主结果未站稳前重写 detector。

## 风险与失败模式

会削弱即插即用主张，需要所有 baseline 使用相同 head 改动；实现复杂。

## 下一次允许采取的动作

先做 same-selected-frames 对照，证明几何问题实质影响后再决定。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
