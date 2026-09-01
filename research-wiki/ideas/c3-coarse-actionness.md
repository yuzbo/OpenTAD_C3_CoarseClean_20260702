---
type: idea
node_id: idea:c3-coarse-actionness
title: "C3 低成本粗分类动作性"
stage: active_baseline
outcome: mixed
tags: ["coarse-probe", "actionness", "c3"]
added: 2026-07-11
---

# C3 低成本粗分类动作性

## One-line thesis

低成本粗分类器不直接完成 TAD，而是提供动作/背景概率、状态变化和隐藏特征，供边界敏感 selector 间接决策。

## 为什么提出

小模型在低分辨率或稀疏输入上仍能区分动作与背景；这比直接要求它精确定位边界更符合能力边界。

## 已有证据

已形成 MobileNet/official-ASFormer 等 probe、p_action 导出与在线 C3 source；早期 PAction 结果表明粗信号有用。

## 当前选择或否定理由

保留为主方法输入侧模块，但 actionness 只能作为二分类校准和辅助分数，不能再次主导最终选帧。

## 风险与失败模式

动作内部置信度高、边界处概率平滑；粗分类误差与 selector 偏移可能被混淆。

## 下一次允许采取的动作

分别报告 actionness AUC/F1、transition peak-to-boundary distance、隐藏特征消融和 selector 条件性能。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
