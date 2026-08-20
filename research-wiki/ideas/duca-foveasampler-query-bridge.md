---
type: idea
node_id: idea:duca-foveasampler-query-bridge
title: "DUCA-FoveaSampler / Query-Bridge"
stage: implemented_component
outcome: pending
tags: ["ducca", "query-bridge", "foveated-sampler", "pre-backbone-selector"]
added: 2026-08-19
---

# DUCA-FoveaSampler / Query-Bridge

## One-line thesis

用 4 个 class-agnostic query token 与低成本 FoveaScout 建桥：query 对时序记忆的贡献
热图 A 监督帧重要性，内部记忆 Q1 只留在 selector，手工三分支
（saliency + boundary_edge + uncertainty_context）产生帧分数；FoveatedSampler 做
global top-k + boundary neighborhood quota + greedy MMR 的 exact-K 选择。

## 为什么提出

用户批准 FoveaSampler/Query-Bridge 契约，要求保留手工三分支并作为完整实现部署。
它把 Uni-AdaFocus 式的低分辨率全局观察、辅助任务监督和 hard-sampling straight-through
落成 DUCA pre-backbone selector，同时避免 Q1 进入 heavy detector。

## 已有证据

- 代码：`0975aac3`..`4ae50671`，模块位于 `opentad/models/selectors/fovea_*`、`query_bridge.py` 与 `opentad/models/losses/fovea_losses.py`。
- 测试：远端 CPU `tests/test_fovea_query_bridge.py` 11/11 通过。
- 完整模型在远端 CPU 可构建；GPU one-step gate `1244839` pending。
- 5/7 arm 单 seed full-train 完成；尚无 matched 基线与剩余两臂，不能称 empirically_supported。

## 当前选择或否定理由

手工三分支按用户要求保留；Q1 只作为 selector-internal query context；cycle feedback
detached 且仅训练期可用；inference 无 GT/teacher/cycle/cache。

## 风险与失败模式

- hard Gumbel-TopK surrogate 可能与 detector utility 方向不一致；
- boundary neighborhood quota 可能退化为 repair 而非学习；
- cycle 需要后置 head 二次 forward，训练成本上升；
- 尚无 matched uniform/random/dense 同提交基线。

## 下一次允许采取的动作

先跑 GPU one-step gate，再提交 `baseline_fused / query_only / query_gt_mask / query_cycle /
query_fovea / query_fovea_dpp / full` 单 seed 开发矩阵；有结果后再扩展 seed 和消融。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
