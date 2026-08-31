---
type: paper
node_id: paper:wang2026-conformal-thinking
title: "Conformal Thinking: Risk Control for Reasoning on a Compute Budget"
authors: ["Xi Wang", "Anushri Suresh", "Alvin Zhang", "Rishi More", "William Jurayj", "Benjamin Van Durme", "Mehrdad Farajtabar", "Daniel Khashabi", "Eric Nalisnick"]
year: 2026
venue: "ICML 2026"
external_ids: {arxiv: "2602.03814", doi: null, s2: null}
tags: ["risk-control", "adaptive-compute", "conformal", "early-exit"]
added: 2026-07-11
---

# Conformal Thinking

## One-line thesis

把自适应推理预算转化为分布无关风险控制，在给定风险容忍度下选择高效停止规则。

## Problem / Gap

Adaptive compute 将预算选择转化为不可解释阈值选择，并不自动提供风险保证。

## Method

Validation-set risk control、上下阈值与 efficiency loss。

## Key Results

在 reasoning tasks 上报告风险控制下的计算节省。

## Assumptions

校准与测试满足所需交换性/分布条件。

## Limitations / Failure Modes

对象是 LLM reasoning；distribution shift 会破坏 transfer。

## Reusable Ingredients

Risk-budget formulation 与 held-out threshold selection。

## Open Questions

如何为结构化 TAD loss 构造有效、非过度保守的风险分数？

## Claims

无本项目 proof-checker claim。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。

## Relevance to This Project

否定“首次用校准风险分配计算”的宽泛表述；项目 delta 必须是 TAD 结构化风险。
