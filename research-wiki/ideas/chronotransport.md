---
type: idea
node_id: idea:chronotransport
title: "ChronoTransport 动态特征刷新"
stage: spec_approved
outcome: bounded_appeal_pending
tags: ["feature-refresh", "transport", "parallel-route"]
added: 2026-07-11
---

# ChronoTransport 动态特征刷新

## One-line thesis

保持外部 detector 网格，仅在 VideoMAE time×layer 上选择 RECOMPUTE/TRANSPORT/HOLD，减少 heavy subpath 重算。

## 为什么提出

避免 pre-backbone 删除帧引起 selected-axis 几何和 full decode 争议。

## 已有证据

Stage-A、paired replay 和正式 Stage-B fit/calibration/evaluation 已落地。`92029ea` 的预注册 P3 science gate 为 FAIL：risk-regret 排序为负，cell-risk/window-target 尺度错配，feature transport 改善不稳定；Stage C/P5 未解锁。

## 当前选择或否定理由

历史 P3 保持负结论，Stage C/P5 仍未解锁。用户批准一次有界上诉，但 GitHub-visible
Pro 复核裁决 `REVISE_SPEC_BEFORE_PLAN`：必须先完成 r2 规格、精确 SHA 与 spec-only
复核，不能把书面修订误写成已经实现或实验支持。

## 风险与失败模式

transport 可能不优于 HOLD；cache 状态与校准；真实 kernel cost。

## 下一次允许采取的动作

修订规格最终冻结为 commit `e4422f5`，exact-byte SHA-256 为
`87FA305CCAFC3A29176C3971F593489F86EDD23A4C02C1BFBDAE4144FCF34CF8`；空白上下文
spec-only reviewer 已返回 `APPROVE_SPEC_FOR_PLAN`。当前只解锁 implementation plan；实现
完成后仍必须先通过 pre-Gate1 registration 与 Gate 1，才可按 stop chain 继续 Gate 2--4。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
