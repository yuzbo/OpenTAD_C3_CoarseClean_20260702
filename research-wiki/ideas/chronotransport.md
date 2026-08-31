---
type: idea
node_id: idea:chronotransport
title: "ChronoTransport 动态特征刷新"
stage: second_independent_review_requires_spec_revision
outcome: negative_gate
review_status: revise_spec_before_code
next_protocol: CT-P3R-3S-r1
spec_commit: "02199f8"
spec_sha256: "871420261BD1C19CC515218A6016A91ED7D553B73740AB41C2E02AA7F96609F9"
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

暂停，不作为当前主线。它证明 conditional-compute 工程闭环可运行，但没有证明 risk-certified transport 的科学有效性，也不能混用 DUCA 结果改写 claim。

2026-07-11 查新进一步将其文献新颖性暂评为 `4.5/10`。MoD、Eventful
Transformers、ResidualViT、Progressive Block Drop、Adaptive Temporal Refinement、
SCOPE 与 conformal compute control 已覆盖大部分基础部件。可守住的最小 delta 仅是：
离线 TAD 的 dense physical-time lattice、time×depth 联合调度、相对 dense 的单侧
结构化 localization regret，以及实测全栈成本约束的组合。

## Result-to-Claim 裁决

- H1（输入相关 time×depth 非均匀重算价值）：`no/unsupported`，尚无 equal-cost
  oracle gap；不是已反证，但目前没有正证据。
- H2（TRANSPORT 稳定优于 matched HOLD）：`partial`，只有 P2 detector-regret CI
  为正，feature CI 跨 0。
- H3（deploy-visible risk 可预测并校准 regret）：`no`，当前实现被正式否定。
- H4（保护高 tIoU/短动作的真实 full-stack 加速）：`no/unverified`。

因此当前 seed-3407 checkpoint 与 144-cell 非负求和 risk 规格已经死亡；更宽泛的
ChronoTransport 假设族尚未彻底死亡，但只有一次 medium-low 概率的上诉资格。

## 风险与失败模式

transport 可能不优于 HOLD；144-cell risk sum 与窗口 target 尺度错配；SCOPE 已有
`cache/predict/recompute` 三模近邻；真实 kernel cost 尚未证明。

## 下一次允许采取的动作

只允许一次有界修复：重新定义窗口级风险聚合尺度、稳定 feature transport 并预注册
重跑 P3。若 Spearman 仍低于 `0.2`、feature improvement CI 再次跨 0，或 full-stack
p50 saving 低于 `15%`，立即降级为 baseline，不启动 Stage C。

唯一允许的上诉协议为 `CT-P3R-3S`，固定 seeds `3407/3408/3409`。冻结 backbone、
cache/action 语义和 transport 架构，仅把 cell-sum risk 改为一个 window-level quantile
head；依次通过 oracle feasibility、matched TRANSPORT-vs-HOLD、selection-aware risk、
真实 scheduler/P5 四道门。任一道失败即永久冻结，不允许再换 head/loss/权重重试。

## Pro review 与 r1 状态

2026-07-11 Pro review 裁决为 `REVISE_SPEC_BEFORE_CODE`。`b74101d` 保留为历史设计，
但不能原样执行；它的双边 coverage hard gate、candidate-row pooled Spearman、
Gate-1 oracle→input-dependence 归因、evaluation-best comparator 隔离和 full-stack
percentile 定义存在可在实验前修正的统计/定义问题。

下一合法协议名为 `CT-P3R-3S-r1`。r1 必须保持 head、seeds、candidate library、140 次
successful updates、quantile、epsilon 和 Gate 1/2/4 数值门槛，只修正统计单位、claim
边界和成本/曝光定义。r1 新 spec SHA 经用户复核前，禁止写争议代码、运行 profiler/Gate 1、
训练新 seed 或解锁 Stage C。

该 review 没有看到本地 ChronoTransport 源码，因此 optimizer LR、packed route、
loss normalizer、Stage C runner 等代码问题仅是待本地核验风险，不是已确认 bug。

## r1 本地源码复核

对 commit `375094d` 的静态审计确认：cell-sum risk、row-level calibration、pooled-row
Spearman、per-seed split、残缺 candidate library、缺 full-stack total sampler、缺完整
provenance/Stage C/P5 都是真实缺口。另一方面，Stage B 使用独立非零 LR AdamW，head 在
paired replay 中为 eval，RNG 会恢复，CT config 也已关闭 frame selector 与 packed route；
因此 Pro 对 Stage-B 静默冻结、loss-normalizer 顺序污染和共享路由同时激活的推测不成立。

本地还发现 reviewer 未见的 adapter 语义缺陷：当前 dense TIA 虽被计算，却只写回
RECOMPUTE rows，HOLD/TRANSPORT rows 实际绕过 TIA。r1 必须把动作限定为 heavy
attention/MLP 子路径，并让恢复后的完整 tensor 对所有 rows 经过原 AdaTAD adapter。
详细证据见 `sources/2026-07-11-chronotransport-r1-local-source-audit.md`。

## 独立 agent 复核

空白上下文 secondary reviewer 独立给出相同 verdict：`LOCAL_CORRECTED_R1`，并确认
pre-adapter heavy cache + all-row TIA 是正确边界。它新增发现 `max_cache_age=8` 与
`hold_only/transport_only` 的 47-clip 连续复用冲突，以及 runtime repair 后 requested
cost 不能代表 executed cost。r1 必须拆分 hard cache validity 与 transport embedding cap，
并登记 requested/executed schedule/cost。完整记录见
`sources/2026-07-11-chronotransport-r1-independent-agent-review.md`。

## r1 书面规格

用户已批准独立复核后的设计。正式书面规格已在 commit `02199f8` 单文件提交，文件
SHA-256 为 `871420261BD1C19CC515218A6016A91ED7D553B73740AB41C2E02AA7F96609F9`。
当前状态是 `written_spec_pending_user_review`；在用户对该文件做最终书面批准前，仍不得
调用 implementation planning、修改模型代码或运行 Gate 1。

2026-07-12 新的空白上下文 reviewer 对 commit `02199f8` 给出
`REVISE_SPEC_BEFORE_PLAN`。它确认现有 cache/adapter/age/cost 修复，但发现 simple
offset 的 mod-4 exposure confounding、video/window unit mismatch、Stage-C owned-gradient
与 AMP retry 未闭合、Gate-1 shuffle tautology和 immutable identity 缺口。当前状态改为
`second_independent_review_requires_spec_revision`，仍禁止 writing-plans。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
