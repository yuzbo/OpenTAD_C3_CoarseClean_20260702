---
type: idea
node_id: idea:phystime
title: "PhysTime 时间戳感知 TAD"
stage: parallel_active
outcome: pending
tags: ["physical-time", "parallel-route", "tad"]
added: 2026-07-11
---

# PhysTime 时间戳感知 TAD

## One-line thesis

显式保留/使用物理时间戳或不规则采样几何，避免 selected-axis 等间隔假设。

## 为什么提出

直接回应 DUCA selected-axis geometry 风险，并探索不删失时间语义的 detector。

## 已有证据

PhysTime-TAD/PhysTime-AdaTAD 设计、代码与远端数据/门控任务正在推进。

## 当前选择或否定理由

独立并行路线，不在 DUCA fixed-384 结束前偷换主方法。

## 风险与失败模式

可能变成 detector 重写而非插件；数据与官方 AdaTAD 对齐复杂。

## 下一次允许采取的动作

完成真实数据 gate、官方 AdaTAD parity 与 matched baseline。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
