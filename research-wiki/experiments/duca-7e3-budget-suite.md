---
type: experiment
node_id: exp:duca-7e3-budget-suite
title: "7e3a508 fixed-K / MUST budget suite"
idea: idea:duca-offline-full-window
verdict: no
confidence: high
commit: "7e3a5081f58958fc924accf43088b24e2bf3093a"
jobs: "1152688-1152693"
updated: 2026-07-11
---

# 7e3a508 fixed-K / MUST budget suite

## Raw metrics / observations

fixed384、fixed256、fixed128、MUST384、MUST320 完成；MUST256 为 NODE_FAIL。历史训练性能与预算行为不足以作为最终论文证据。

## Interpretation

暴露 max-gap、预算 controller、旧 loss 聚合与训练/推理契约问题，推动 1684f6b/c26b349/70aa069 重构。

## Limitations

旧提交；不能证明最新 DUCA；dynamic K 曾在 64/384 间跳。

## Provenance

Slurm sacct; docs/methods/2026-07-10-88e50b1-duca-final-method-audit-review-absorption.md

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
