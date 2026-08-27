---
type: idea
node_id: idea:duca-fsu
title: "DUCA-FSU：可行硬交换效用蒸馏"
stage: discussed
outcome: pending
tags: ["duca", "counterfactual-utility", "fixed-budget", "offline-tad"]
added: 2026-07-13
---

# DUCA-FSU：可行硬交换效用蒸馏

## One-line thesis

在 exact-K/max-gap 可行集合中，用训练期真实 hard one-swap 的 detector-loss gain
监督 transition-derived utility difference，替代未经验证的 raw-pixel soft bridge。

## 与当前 DUCA 的区别

- 保留二分类 action-state probe、transition descriptor、单 utility scorer 和 hard DP。
- 删除 `structured_zero_forward`、soft RGB mixture 和 beta schedule。
- detector counterfactual 只生成 stop-gradient target；detector 主 loss 仍只训练 detector。
- 推理只执行 low-res probe、hard DP、gather/reconstruction 和 detector。

## 当前证据

无实验证据。旧 beta=0.25 比 beta=0 低 0.79 best Avg-mAP，只说明 current bridge 未获支持；
它不支持 FSU，也不构成对所有 detector-aware utility learning 的反证。

## 必须先通过的机制门

1. common exact-uniform route 与 corrected CUDA gate。
2. coarse action-state AUROC/AUPRC/ECE 与 transition peak quality。
3. 固定 detector 下 feasible one-swap gain 的可重复性。
4. utility difference 与 detector gain 的 window-cluster correlation/sign audit。
5. same-selected-frames geometry 与 full-stack cost。

## 风险与停止条件

- detector gain 可能高度非加性，`u_t-u_s` 无法表达 frame interactions。
- one-swap target 可能受 detector stochastic state、normalizer、RNG 和数值噪声污染。
- physical-time reconstruction 可能平滑边界或增加额外成本。
- 同 K 下 DUCA 必然比 no-probe uniform 多 probe/select 成本；必须靠更高精度或更小 K
  的 cost-matched Pareto 才能成立。

## 状态纪律

本节点仅为 external review 推荐后登记的 `discussed` 路线，不代表项目已经选择、实现、
测试或认可它为最终模型。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
