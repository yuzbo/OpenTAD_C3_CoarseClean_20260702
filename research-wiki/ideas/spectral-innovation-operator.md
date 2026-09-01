---
type: idea
node_id: idea:spectral-innovation-operator
title: "Spectral Innovation Operator for Dense TAD"
stage: high_risk_proposed
outcome: unknown
tags: ["spectral", "innovation", "boundary", "dense-lattice"]
added: 2026-07-11
---

# Spectral Innovation Operator for Dense TAD

## One-line thesis

把时间特征分解为低频语义场与稀疏高频边界 innovation，仅对自适应谱系数运行重型
算子，再重建完整物理时间网格。

## 为什么提出

它改变表示而不只是改变执行 schedule：长期动作内部主要由低频场描述，动作开始、结束
和短动作由高频 innovation 描述，因此不依赖 `HOLD/TRANSPORT/RECOMPUTE` 拼装。

## 当前选择或否定理由

独立查新暂评 `8/10`，创新潜力最高，但理论、kernel 和训练风险也最高；不作为第一实现。

## 风险与失败模式

边界并不必然是稳定的频谱高频；谱变换与稀疏 kernel 的真实成本可能超过收益；类别语义
与时间频率可能不可分离。

## 下一次允许采取的动作

先对 dense VideoMAE 特征做频带能量—boundary regret 分析；若高频能量不能稳定预测
边界误差，立即否定该路线。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
