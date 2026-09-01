---
type: experiment
node_id: exp:stage1-paction-gasvt
title: "Stage1 PAction / GAS-VT strict-ledger experiments"
idea: idea:paction-strict-ledger
verdict: partial
confidence: medium
commit: "53124a2/c69c1a0-era"
jobs: "multiple historical runs"
updated: 2026-07-11
---

# Stage1 PAction / GAS-VT strict-ledger experiments

## Raw metrics / observations

历史日志显示固定 384 的粗信号路线能较快获得可用 mAP；PAction learned 曾优于 GAS-VT。由于提交、数据、预训练与评估节奏未完全匹配，不在此固化未经审计的最终数字。

## Interpretation

支持“低成本 p_action 信号有用”和“复杂 GAS-VT 未必优于简单策略”，不支持 GAS-VT 为主方法。

## Limitations

缺 matched exact-uniform/random/dense；GAS-VT train/apply mismatch 与 hard repair；旧结果 provenance 不统一。

## Provenance

docs/methods/gas-vt-stage01-53124a2-review-absorption.md; 2026-07-07-c69c1a0-paction-gasvt-hold-review-absorption.md

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
