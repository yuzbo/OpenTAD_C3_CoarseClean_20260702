---
type: idea
node_id: idea:strict-rectangle-roi-routing
title: "Strict rectangular ROI routing for ZoomToken"
stage: designed
status: user_approved_spec_pending_implementation
tags: ["offline-tad", "pre-backbone", "roi", "token-selection", "dynamic-compute"]
added: 2026-08-22
updated: 2026-08-22
---

# ZoomToken 严格矩形 ROI 路由

当前 `70dcbe10…` 主干前 C 使用连续 ROI 分数排序后 Top-64，执行支持不保证形成完整矩形。
用户确认下一实现周期应把矩形定义为硬支持：patch 中心在矩形内则属于 ROI，框外不得通过
ROI 分支进入 VideoMAE 重主干。

已设计四条机制臂：固定 8×8 严格矩形 R1；8×8 矩形内选 48 的 R2；连续宽高、执行全部
矩形成员且 `K_t` 自然变化的 R3；6×8 矩形核心加 16 个框外关键 free token、总数 64 的
R4。R4 的框外名额独立且有上限，不能退回旧 ROI modifier 的全局混排。四臂均在重主干前
gather 原生 tubelet，复用同一 true-ragged VideoMAE、sparse adapter、ActionFormer 与官方训练
配方。

设计规格：`docs/superpowers/specs/2026-08-22-zoomtoken-strict-rectangle-roi-routing-design.md`。
当前证据等级仅为 `designed`；尚未实现、测试、训练或获得性能/成本证据。
