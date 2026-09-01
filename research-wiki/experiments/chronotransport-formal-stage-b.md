---
type: experiment
node_id: exp:chronotransport-formal-stage-b
title: "ChronoTransport formal Stage-B gate"
idea: idea:chronotransport
verdict: negative
confidence: high
commit: "92029ea"
jobs: "formal seed-3407 fit/calibration/evaluation artifacts recorded in method log"
updated: 2026-07-11
---

# ChronoTransport Formal Stage-B Gate

## Raw metrics / observations

正式 fit/calibration/evaluation 闭环、EMA、恢复、split 隔离和统计 gate 可运行。P3 总 gate 为 FAIL：risk-regret 排序为负；cell-risk 求和与窗口 regret target 尺度严重错配；transport 相对 HOLD 的局部 detector-regret 改善不能抵消 feature-level 改善不稳定。

实验数字仅保留在 `docs/methods/2026-07-10-chronotransport-implementation-plan.md` 及对应正式 artifact，本页不重复建立第二数字来源。

## Interpretation

该结果支持工程可行性，但否定当前 risk-certified transport 规格直接进入 Stage C/P5 或论文主方法的资格。

## Limitations

单种子负 gate 不能证明所有 feature-transport 思路无效；它只否定当前风险聚合、目标尺度和训练规格。

## Superseding protocol review

2026-07-11 Pro review 不改变本实验的 negative verdict，也没有提供新实验数据。它裁决
`b74101d` 不能原样成为重跑协议：Gate 1 只能证明 oracle headroom；Gate 3 必须改用
window candidate-vector ranking、unique-window cluster bootstrap 和单边 coverage 下界；
full-stack p50/p95 必须来自完整 total samples。

在新的 `CT-P3R-3S-r1` 规格获得独立 SHA 并经用户复核前，本实验仍锁住任何新 profiler、
Gate 1、Stage B seed 和 Stage C/P5。reviewer 的 standalone generic primitives
`10 passed` 不是本仓库集成测试，也不是新的 experiment node。

后续本地静态源码审计进一步确认，当前 formal runner 仍只有 6-schedule 轮转、candidate-row
校准/相关性和 per-seed manifest，尚未实现 r1 的共享 split、16-candidate exposure、
simultaneous window calibration 或三 seed 统计。因此现有 runner 不能被改名后直接复跑；
这不改变 `92029ea` 的 negative verdict，也不构成新实验结果。

## Provenance

Commit `92029ea`; `docs/methods/2026-07-10-chronotransport-implementation-plan.md`。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
