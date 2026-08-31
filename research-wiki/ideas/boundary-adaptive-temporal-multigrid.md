---
type: idea
node_id: idea:boundary-adaptive-temporal-multigrid
title: "Boundary-Adaptive Temporal Multigrid"
stage: proposed_primary_candidate
outcome: unknown
tags: ["multigrid", "boundary", "offline-tad", "adaptive-refinement"]
added: 2026-07-11
---

# Boundary-Adaptive Temporal Multigrid

## One-line thesis

先在完整窗口的粗时间网格上计算，再依据 detector residual 与边界不确定性递归细化
局部时间区间，最后通过可学习 prolongation 重建完整 detector lattice。

## 为什么提出

它利用离线全窗口条件，直接分配“时间分辨率层级”，而不是顺序维护 cache 或选择
动作帧；同时避免 ChronoTransport 的链式 transport 漂移和固定 48×3 调度。

## 当前选择或否定理由

独立查新暂评 `7.5/10`，是创新性、可实现性和现有 AdaTAD 兼容性的最佳平衡。
目前只是候选，不得声称优于 ChronoTransport。

## 风险与失败模式

粗网格可能在 refinement 前已经丢失短动作；prolongation 可能只做平滑插值；递归细化
若不能产生真实 packed execution，就不会带来延迟收益。

## 下一次允许采取的动作

先做 frozen-feature oracle：比较 uniform 384、单层自适应细化和两层 multigrid，在相同
真实计算预算下检查 mAP@0.7、短动作与边界距离。未胜过 uniform 即停止。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
