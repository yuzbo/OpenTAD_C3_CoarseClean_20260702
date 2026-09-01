---
doc_id: CURRENT_RESEARCH_STATE
version: v014
status: prepared
date: 2026-08-23
supersedes: CURRENT_RESEARCH_STATE-v013
stage: G4_SCIENTIFIC_REVISE_REQUESTED
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
github_revision: 04c35a3b76897e6c1569eeede41ed3aecaf7f854
---

# DUCA 当前研究状态：H65-60 与 TrueTime-Aware Bridge

## 当前问题

DUCA 研究低成本语义侦察器如何在真实减少 VideoMAE 重型计算的同时保护时序动作检测（Temporal Action Detection, TAD）的高 IoU 定位。当前讨论只回答一个问题：在保持 H65 的语义间接非均匀逐帧选择、固定 K=384、训练课程、检测器、损失、NMS 和官方评估器不变时，应当怎样把真实物理时间引入重型表示和检测器前重建，使增益能够被因果归因。

## 冻结代码身份

- GitHub：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- H65-60 分支：`codex/duca-h65-60-curriculum-20260823`
- 当前 clean HEAD：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`
- H65-60 Stage-2 冻结源码：`87ff0883651a631d48468ab4f9d6392f587c15e4`
- TrueTime 严格配对冻结源码：`11126684af779aa2916a68ecf617c4f14c805478`，分支 `codex/duca-truetime-curriculum-v3-20260822`
- 历史 H65 代码证据锚：`42dba3f90b37243e7965d18b6707e88e81bf7109`；该锚包含后续诊断提交，历史 65.385724 的训练身份必须通过 Wiki/收据解释，不能把该 SHA 单独称为 65.385724 的训练 checkpoint。

## 已有证据

1. 历史 H65 的 65.385724 Avg-mAP 是非均匀 K384 逐帧输入，不是 uniform。它使用 30 epoch uniform 预热加 60 epoch learned Stage-2，并包含 ASFormer 语义预测、sampling-rate/density transport、贡献蒸馏和适配等多因素，因此是正信号而非公平单变量结果。
2. 严格同提交 RankPack/TrueTime 配对使用相同 RGB 帧集合、K384、seed 3407、60 epoch、6000 次成功更新和官方 evaluator：RankPack 61.5722，TrueTime 62.1930，Avg 增量 +0.6208；mAP@0.6 增量 +1.6885，@0.7 增量 +0.7915。单 seed，只能作为部分机制证据。
3. 连续 cliplet FZ/JT 两臂完整 60 epoch 得到 49.89/47.24，且执行账本证明真实只向 patch embedding 输入 K384。该路线作为当前主候选已经停止；不得用 Query/cycle 或 dynamic-M 挽救。
4. UVT 的 off/geo/geo+EMA 为 57.35/55.93/55.92；Fovea 五臂为 42.94/45.26/49.16/54.67/43.77。它们同时改变选择、预算、辅助损失和协同训练，不能把下降单独归因于时间表示。
5. 当前 H65 Stage-1：30 epoch 终点 59.4231；压缩 20 epoch 终点 49.5389。两条 Stage-2 正在 N16R4 运行：Job 1251622（总 60 轮压缩候选）与 Job 1251782（30+60 原始日程复现）。截至 2026-08-23 22:38 +08:00，两者均为 RUNNING，尚无 learned H65 终态 mAP。

## 证据与反主张边界

- 不宣称 H65-60 已保住历史 65.385724。
- 不宣称 TrueTime 已稳定有效或能解释 UVT/Fovea 全部下降。
- 不重复 dense、exact-uniform、random 或既有 fixed-K 基线训练。
- 不复活连续 cliplet 主线。
- fixed K=384 是当前表示归因门；dynamic K 仍是论文核心候选，但只能在表示门通过后进入。
- 不在官方 validation/test 使用 GT、teacher、cache 或 EMA 决定采样位置/预算。

## 当前唯一科学动作

请求 fresh exact-Project Pro 冻结一条 H65-60-compatible TrueTime 路线。候选必须从下列因果分解出发，而不是扩大 selector：

- `Clock OFF / Bridge OFF`：正在运行的 H65-60 selected-rank 身份，作为只读基线；
- `Clock ON / Bridge OFF`：只在 VideoMAE 第一次重型时间混合前使用真实物理相对时间；
- `Clock OFF / Bridge ON`：VideoMAE 保持 selected-rank，但在 detector 前使用有 gap/support 约束的确定性物理时间重建；
- `Clock ON / Bridge ON`：只有前两项显示互补机制后才准入。

Pro 必须决定是否接受该分解、冻结最小实现与 stop rule，或提出一个更简洁且仍保持 H65 输入/训练合同的单一路线。现有 H65 Stage-2 继续运行；任何新正式训练等待其终态、Pro 决策、独立 Critic 和 Evaluator PRE_RUN。

