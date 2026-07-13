---
type: idea
node_id: idea:sm-ptaf
title: "Support-Measure Physical-Time ActionFormer (SM-PTAF)"
stage: designed
outcome: pending
thesis: "以原生 tubelet 多原子支持为 feature provenance，用测度守恒且不插值的 set-to-physical-query lift 建立候选数与容量匹配的物理时间 TAD detector。"
risks: "可能被批评为 mTAN-style regridding + ActionFormer；TIA 仍按 selected rank 混合；收益可能仅来自候选数与容量恢复。"
based_on: ["idea:phystime-tad-2", "paper:zhang2022_actionformer", "paper:liu2024_adatad", "paper:shukla2021_mtan", "paper:kim2024_te_tad"]
target_gaps: ["gap:G1", "gap:G2", "gap:G3", "gap:G4", "gap:G5"]
added: 2026-07-13T00:00:00+08:00
updated: 2026-07-13T00:00:00+08:00
---

# SM-PTAF

## 最小不可分核心

1. 删除主路线的 `192 -> 384` feature interpolation。
2. 将每个原生 VideoMAE tubelet token 绑定到不跨 sparse gap 合并的 raw support atoms。
3. 用 support-overlap mass 构造显式保底路径，再叠加有界 content/relative correction。
4. 在 absolute-video seconds 上构造候选、assignment、regression、decode、NMS 与 evaluation。
5. 与 ActionFormer control 对齐容量、跨 query context、候选拓扑和 target assignment。

## 公平性 control

SM-PTAF 不能直接和旧 PhysTime 1.0 比较后声称有效。必须先建立：

- capacity-matched selected-coordinate ActionFormer；
- capacity-matched physical-coordinate ActionFormer；
- physical-coordinate ActionFormer + support-measure lift。

前两者隔离坐标贡献，后两者隔离 detector operator 贡献。

## 当前状态

`designed`。外部 Pro 回复给出了公式、伪代码和 patch map，但仓库尚未实现这些新类，也没有新 gate、训练或 mAP。PhysTime 1.0 继续冻结为负基线。

## 必须先关闭的风险

- native token 与 support atom 的 provenance 及 TIA rank mixing；
- K 只控制 candidate cardinality，不定义物理坐标；
- zero-coverage query 不得伪装成已观测 feature；
- dropout 不得破坏 mass base path；
- tied-shortest multi-label assignment 必须和 ActionFormer 同构；
- 参数匹配不能靠 dummy parameters 或无意义容量填充；
- 提升不能由候选数恢复、容量增加或 endpoint 额外监督解释。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
