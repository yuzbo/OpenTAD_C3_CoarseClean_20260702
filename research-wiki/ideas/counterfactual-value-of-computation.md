---
type: idea
node_id: idea:counterfactual-value-of-computation
title: "Counterfactual Value-of-Computation for TAD"
stage: proposed
outcome: unknown
tags: ["counterfactual", "compute-value", "causal-attribution", "offline-tad"]
added: 2026-07-11
---

# Counterfactual Value-of-Computation for TAD

## One-line thesis

直接学习某个 `time × layer` 计算干预对最终结构化定位损失的边际价值及交互，而不是
预测动作概率、特征变化或 schedule 总风险。

## 为什么提出

ChronoTransport 的窗口风险聚合无法回答“哪一次计算真正有用”。Value-of-Compute
把决策变量改为可审计的边际干预效应，可形成预算下的组合优化。

## 当前选择或否定理由

独立查新暂评 `7/10`。理论解释力较强，但 counterfactual 数量和交互估计成本很高，
暂列第二候选。

## 风险与失败模式

单元价值可能高度非加性；离线 teacher 干预成本大；学习到的价值可能只在固定 detector
和 checkpoint 上成立。

## 下一次允许采取的动作

在一个冻结 checkpoint 上采样 one-cell、one-interval 与 pair interventions，先检验
可加性、排序稳定性和跨视频泛化，再决定是否构建策略。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
