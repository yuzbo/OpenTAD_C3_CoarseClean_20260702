---
type: idea
node_id: idea:duca-rime
title: "DUCA-RIME: risk-calibrated marginal evidence allocation"
stage: discussed
outcome: pending
tags: ["duca", "offline-tad", "dynamic-budget", "hard-utility", "paired-boundary", "physical-time"]
added: 2026-07-27
---

# DUCA-RIME

## One-line thesis

在离线 TAD 中，用重型 backbone 前的低成本证据预测各真实帧预算下经 hard
counterfactual 验证的分类、回归和 paired-boundary 风险，以训练侧冻结的 per-video
dual policy 选择 `K`，再由有界物理时间 exact-K 解码器选择真实 RGB 帧。

## Why this idea exists

当前 K=192 诊断显示动作富集和局部边界邻近改善，却同时损伤宽双端支持并增大最大
空洞。固定 actionness/transition 分数没有学习“新增一组物理观测是否共同支撑完整动作
区间”。另一方面，动态 K、scorer、ILP、inverse-CDF 和 cheap-to-heavy 视频计算均有
强先验工作，不能单独成为创新。

候选科学中心因此是：

```text
hard budget-conditional TAD value
+ paired endpoint / high-IoU risk
+ bounded physical-frame acquisition
+ batch-invariant realized-cost allocation
```

## Invariant model core

1. 任务始终是 offline TAD。
2. low-cost evidence 必须在 heavy backbone 之前可见。
3. 对每个候选 K，输出有序、唯一、strict exact-K 物理帧。
4. constant evidence/density 必须严格退化为 canonical uniform。
5. raw proposals 从 selector 坐标逆映射到物理时间后，才调用不变的官方 NMS。
6. 推理不使用 GT、teacher、raw-prediction cache 或 test-batch global solver。
7. 外层决策为有限 K 集合上的 `utility - risk - frozen_price * measured_cost`。
8. 真实 requested/effective/unique/backbone K 和完整成本必须一致登记。

## Unresolved decoder family

本轮审阅内部同时提出 independent-per-K 与 strict nested，不能同时冻结。

- `independent`：共享 cheap evidence 和 bounded density，但每个 K 独立 exact-K
  解码；预测的是 budget-policy value。
- `strict nested`：`S(K1) subset S(K2)`，预测 group-add marginal utility。
- `weak overlap`：只要求相邻预算保持预注册比例的交集。

当前推荐的是 **regret-gated decision protocol**：先在 train-only Oracle 上比较三者，
再冻结唯一 decoder。若 strict nested 的视频级 regret 不可忽略，就不得为了“增量”
叙事牺牲定位；正式方法回到 independent 或一次性 weak-overlap 备选，并准确改名效用
语义。

## Hard utility and risk

候选监督不是 raw detector gradient magnitude，而是 train-only frozen-detector 的：

- cls/reg hard group gain；
- proposal/high-IoU gain；
- start miss、end miss 和 pair miss；
- 2/4/8/16 帧、1%/5%/10%、连续片段和 paired-endpoint perturbations；
- matched random、score shuffle 和 score reverse null controls。

`G_rank` 失败时删除 utility-head claim；pair-risk 不能优于 actionness/transition 时
删除 paired-boundary contribution；`G_direct` 是后续可选增强，不是默认主模型。

## Novelty boundary

不允许声称：

- 首次动态视频预算；
- 首次 pre-backbone cheap-to-heavy allocation；
- 首次多预算质量曲线、scorer、ILP、inverse-CDF 或 nested prefix；
- 首次风险校准或高效 TAD。

可争取但尚未验证的组合命题是：面向离线区间检测的 hard budget-conditional utility
与 paired endpoints 风险，被用于真实物理帧 exact-K 采集和 batch-invariant 平均成本
分配，并在不改 detector 内部结构的条件下保护 high-IoU。

## Decisive gates

1. clean dense/native uniform/wrapper parity；
2. raw-proposal physical-time-before-NMS；
3. dynamic Oracle headroom；
4. nested/weak/independent Oracle regret；
5. video-cluster `G_rank` 与 pair-risk；
6. one development seed under identical mixed-K exposure and 6,000 updates；
7. three fresh seeds, second detector and full-stack cost only after pass。

所有具体 `pp`、gap-recovery、risk 和 cost thresholds 在 clean baseline 方差与视频级
功效分析后、正式结果前冻结。

## Kill rule

- fixed-K inner policy 不能超过 clean uniform：停止当前 DUCA 采集路线；
- dynamic Oracle 不超过 best fixed at matched realized cost：删除 dynamic-K 主线；
- hard utility 不可预测：删除 utility scorer，不用 soft gradient 替代；
- pair-risk 不修复宽双端/高 IoU：删除 localization-preserving claim；
- 真实执行 pad 到 Kmax 或完整成本无净省：删除 efficiency claim；
- 只有 detector 内 true-time 改造有效：改称 time-aware integration；
- 第二 detector 方向反转：收缩为 AdaTAD-specific 或停止通用插件 claim。

## Status

`discussed / mathematical_contract_not_frozen / implementation_not_started /
experiment_not_started / paper_claim_not_allowed`。

完整审计：
`docs/methods/2026-07-27-duca-dynamic-k-adaptok-research-takeover-absorption.md`。

## Dual-response adjudication

两份后续接管回复已经逐行吸收并字节一致归档。它们在科学中心上高度一致：
dynamic K 必须正面裁决，hard detector utility、paired/high-IoU risk、物理时间
exact-K 和训练侧 frozen dual 构成候选组合；AdapTok 式多预算 scorer/ILP 只能作
direct-transfer baseline。

它们不是一致的实现合同：回复 A 使用 DUCA-METER/METER-TAD，回复 B 使用 MERTAD；
二者对 single-vs-paired heavy forward、K grid、loss、数值门槛、风险硬/软约束和
fallback 的定义不同。二者都在没有 nested-regret 证据时偏向 strict nesting。

项目裁决保持：

1. 内部候选名继续使用 `DUCA-RIME`，投稿名不冻结；
2. dynamic K 是 required decisive candidate，不是已验证主创新；
3. strict nested、independent 和一个 weak-overlap family 先做 train-only Oracle
   regret，再冻结唯一 decoder；
4. 确定性几何约束可硬执行，learned risk 在独立校准前只能是 empirical surrogate；
5. total-60 训练合同、K set 和所有数值门槛在 clean variance/power 后另行冻结；
6. 在 clean parity、`q -> t -> NMS`、O1--O4 通过前，不实现或训练完整 RIME。

完整比较：
`docs/methods/2026-07-27-duca-dynamic-k-adaptok-dual-response-comparison.md`。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
