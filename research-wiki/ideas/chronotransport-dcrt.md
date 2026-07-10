---
type: idea
node_id: idea:chronotransport-dcrt
title: "ChronoTransport / DCRT"
stage: paused_after_failed_p3
outcome: negative_gate
thesis: "在 dense physical-time lattice 上，对 time x layer 单元选择 recompute、transport 或 reuse，并以定位风险约束节省。"
risks: "接近 MoD、DFF、AdaFuse 和 feature cache；层级动作僵硬；系统实现与多任务验证过大。"
based_on: []
target_gaps: ["gap:G7"]
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# ChronoTransport / DCRT

## Pro 推荐理由

它把决策单位从 frame 改为 time x layer x update operation，保持 dense 时间网格，以 counterfactual localization regret 而非 actionness proxy 控制风险，并能攻击重复 backbone 计算这一更一般问题。

## 用户否决理由

- 直觉上是更复杂的 MoD/feature reuse；
- “以 16 帧、帧、token 还是 layer block 为单位”会变成僵硬超参数；
- 需要 cache、transport、risk calibration、真实系统 profile 和跨任务验证；
- 三个月内很难同时证明算法新颖性和系统收益；
- 不符合当前希望做一个完整新 TAD 检测方法的偏好。

## 可复用资产

No-Free-Frames 全栈计费、counterfactual risk 与 periodic refresh baseline 仍应保留为未来工作。

## 实现事实

该路线并非只有 idea：本地分支 `codex/c3-coarse-clean-20260702` 已实现到正式 Stage-B fit/calibration/evaluation，最新方法 commit 为 `92029ea`。P3 science gate 因 risk 尺度/排序失败和 feature transport 优势不稳定而失败；对应 origin 落后 15 commits，P5 未解锁。完整代码、训练和证据边界见 `routes/chronotransport-complete-record.md`。

## 恢复条件

只有 profiling 证明重复 feature recompute 是主瓶颈，且简单 periodic refresh 明显弱于 risk-certified policy，才可新开独立项目。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
