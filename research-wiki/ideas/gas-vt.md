---
type: idea
node_id: idea:gas-vt
title: "GAS-VT value-transport acquisition"
stage: archived_diagnostic
outcome: negative
tags: ["gas-vt", "ledger", "value-transport"]
added: 2026-07-11
---

# GAS-VT value-transport acquisition

## One-line thesis

学习动作、边界、覆盖、预算和 gap 的 frame value，再生成严格 sparse ledger。

## 为什么提出

试图比 p_action top-k 更系统地描述被选帧对后续时间位置的价值传输。

## 已有证据

Stage0/1 ledger 与 AdaTAD 接口已实现；早期 mAP 上升但随后 plateau。

## 当前选择或否定理由

不能作为主方法。降级为 engineered constrained baseline/negative control。

## 风险与失败模式

训练/应用特征不一致；deploy 不是忠实 sequential value transport；多损失与 hard repair 可能趋向 uniform。

## 下一次允许采取的动作

不再继续修补成主方法；只保留 pure top-k、gap repair 与 PAction 的机制对照。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
