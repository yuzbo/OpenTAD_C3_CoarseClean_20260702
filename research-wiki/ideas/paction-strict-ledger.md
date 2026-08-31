---
type: idea
node_id: idea:paction-strict-ledger
title: "PAction learned strict ledger"
stage: archived_baseline
outcome: mixed
tags: ["ledger", "paction", "fixed-budget"]
added: 2026-07-11
---

# PAction learned strict ledger

## One-line thesis

从 p_action 派生并学习固定预算 frame score，经过严格 no-leak ledger 接入 AdaTAD。

## 为什么提出

提供比 raw top-k 更强的固定预算安全锚点，并验证 sparse loader、预算与 provenance 链路。

## 已有证据

工程链路和 validator 已落地；历史观察中 PAction learned 比复杂 GAS-VT 更强。

## 当前选择或否定理由

保留为 Stage1 baseline 和粗信号有效性证据，不作为最终论文方法。

## 风险与失败模式

多阶段、detector-unaware、hard gap decoder；旧结果缺少同提交 matched baseline。

## 下一次允许采取的动作

仅在统一 commit/config 下作为 PAction baseline 重跑，不再扩展其独立训练路线。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
