---
type: paper
node_id: paper:raposo2024-mixture-of-depths
title: "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"
authors: ["David Raposo", "Sam Ritter", "Blake Richards", "Timothy Lillicrap", "Peter Conway Humphreys", "Adam Santoro"]
year: 2024
venue: "arXiv"
external_ids: {arxiv: "2404.02258", doi: null, s2: null}
tags: ["conditional-compute", "token-routing", "depth"]
added: 2026-07-11
---

# Mixture-of-Depths

## One-line thesis

每层以固定 top-k 容量动态选择参与 attention/MLP 的 token，使总计算可预测而 token
身份随上下文变化。

## Problem / Gap

Transformer 对所有 token 和层均匀分配计算。

## Method

Per-block router、top-k token selection 与 residual bypass。

## Key Results

在语言模型中以更少 FLOPs 匹配基线，并报告实际 step 加速。

## Assumptions

固定每层容量；原论文对象是语言模型。

## Limitations / Failure Modes

不维护时序 transport 状态，也不针对 TAD 边界或 dense physical-time lattice。

## Reusable Ingredients

Token×layer 条件计算和固定容量路由。

## Open Questions

固定容量能否保护 TAD 短动作与边界？

## Claims

无本项目 proof-checker claim。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。

## Relevance to This Project

是 ChronoTransport 的强方法论近邻，否定“首次 time×depth 动态计算”的表述。
