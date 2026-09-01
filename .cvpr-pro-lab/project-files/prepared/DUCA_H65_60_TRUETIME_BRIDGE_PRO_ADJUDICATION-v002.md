---
doc_id: DUCA_H65_60_TRUETIME_BRIDGE_PRO_ADJUDICATION
version: v002
status: prepared
date: 2026-08-23
supersedes: DUCA_H65_60_TRUETIME_BRIDGE_PRO_ADJUDICATION-v001
stage: G4_SCIENTIFIC_REVISE_REQUESTED
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
github_revision: 04c35a3b76897e6c1569eeede41ed3aecaf7f854
---

# DUCA H65-60 TrueTime-Aware Bridge：最终设计裁决请求

## Pro 身份与任务

请作为本模型、实验计划和论文的 Scientific First-Author Agent 与 Primary Research Owner。你必须作出一个唯一的 `CONTINUE`、`REVISE`、`PIVOT` 或 `STOP` 决定，不把路线选择交回人类或 Codex。优先选择最短、最可证伪、最有出版价值的真实实验路径。Builder 只实现，Critic 独立攻击，Evaluator 只做 PRE_RUN 与证据评估。

这是 exact DUCA Project 中的新会话。不得依赖旧聊天。请从 `CURRENT_RESEARCH_STATE-v015.md`、本文件以及下列 GitHub revision 重建事实：

- H65-60/current branch：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`
- H65-60 Stage-2 source：`87ff0883651a631d48468ab4f9d6392f587c15e4`
- TrueTime paired source：`11126684af779aa2916a68ecf617c4f14c805478`
- 历史 H65 evidence anchor：`42dba3f90b37243e7965d18b6707e88e81bf7109`

重点路径：

- `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py`
- `opentad/models/selectors/truetime_joint_selector.py`
- `opentad/models/backbones/backbone_wrapper.py`
- `opentad/models/backbones/physical_time.py`
- `opentad/models/backbones/vit_adapter.py`
- `opentad/models/utils/truetime_geometry.py`
- `opentad/models/detectors/single_stage.py`
- `configs/adatad/thumos/duca_truetime_indirect_curriculum_k384_base.py`
- `configs/adatad/thumos/duca_truetime_k384_curriculum.py`
- `scripts/run_duca_truetime_curriculum_official60_gpu1.sh`

## 不可漂移的科学合同

1. H65 的 learned task 是低成本 scout 的动作性/边界语义预测；确定性规则间接产生非均匀逐帧位置。主方法不是小模型直接预测 frame index。
2. 第一归因阶段固定 K=384；dynamic outer-K 是后续论文核心，但不得与 TrueTime 表示门同时引入。
3. 保持同一 selected RGB、seed、训练更新数、VideoMAE-S、Adapter、ActionFormer、loss、NMS、官方 THUMOS14 split/evaluator 与 final/final-EMA 规则。
4. 不重新训练 dense、uniform、random；这些已有只读证据。
5. 不使用连续 cliplet 作为当前主路线：FZ/JT 完整负结果已经触发 stop。
6. 真正减少重型计算；不能用 padding 或 metadata 复制伪造 executed K。
7. validation/test acquisition 不使用 GT、teacher、cache、EMA 或 test-derived threshold。

## 已知机制问题

历史 H65 把有序但物理间隔不均的 K384 帧按 selected rank 切成 16-frame VideoMAE 输入。selected-axis proposal 在 NMS 前恢复物理时间，但这种后置 remap 不能修复 VideoMAE 内部已经发生的时间解释。TrueTime 严格配对只获得 +0.6208 Avg、+1.6885 mAP@0.6，说明物理时间可能有价值，但尚不足以解释全部性能差距。

## 请严厉裁决的三个方向

### A. First-Mixing SingleClock

在第一次 VideoMAE 重型时序混合前，把全局 K384 selected positions 映射为真实物理时间；仅向首个 temporal attention/tubelet mixing 注入零初始化、受限幅度的相对时间偏置。后续 detector 路径保持当前 H65 物理 pre-NMS remap。优点是可修复 representation corruption；风险是预训练 VideoMAE 分布偏移和零初始化参数未必能在 60 epoch 内学到。

### B. Deterministic Support-Aware Bridge

VideoMAE 内仍保持 selected-rank；在重型特征输出与 ActionFormer 输入之间，根据真实 timestamp/tubelet support 做确定性局部重建。只允许相邻支持内插值；超过冻结 max-gap 的区域输出 missing-evidence mask/confidence，而不是跨大空洞合成特征。优点是可解释、低容量；风险是无法修复 backbone 内部时间误读，并会改变 detector 的支持密度和平滑。

### C. Staged 2×2 Clock/Bridge Factorization（Coordinator 推荐）

先把 Clock 和 Bridge 作为两个独立二元因素：

1. OFF/OFF：正在运行的 H65-60 结果，只读引用；
2. ON/OFF：Clock 单变量；
3. OFF/ON：Bridge 单变量；
4. ON/ON：只有 2、3 至少一项通过且二者机制互补时才运行。

该设计不重复 selector/baseline，但仍可能需要两次完整训练。请判断它是否是最小充分因果矩阵；若不是，请删减到一条最有信息量的路线。

## 必须回答的问题

1. 你选择 A、B、C，还是一个严格更好的 H65-compatible TrueTime 方案？为什么它在 TAD 上具有审稿人会感到意外的机制预测，而不只是工程修补？
2. TrueTime 应进入哪个精确位置：Conv3D tubelet embedding 前、首层 temporal attention、所有 temporal attention，还是只进入 detector bridge？请给出张量定义、单位、padding/short-video 处理和 identity-at-init 条件。
3. Bridge 的输入/输出、物理 support、max-gap/missing-evidence 语义应如何定义？如果 max-gap 需要调参，怎样只在 FIT/CAL 内冻结并避免 validation 泄漏？
4. 如何确保同一 RGB frame set、同一 K、同一成功 optimizer update、同一 full-stack compute 口径？新增算子成本如何报告？
5. 现有 RankPack/TrueTime +0.62 应解释为表示瓶颈的部分证据、随机波动，还是实现不足？哪一项无重训练分析最先改变你的判断？
6. 是否必须先对既有 prediction 执行 10,000 次逐视频 paired bootstrap？冻结 CI、统计量和停止规则。
7. 冻结最小 Builder patch：允许修改的文件/符号、禁止修改的变量、必要的 shape/identity/gradient/physical-coordinate tests。
8. 冻结独立 Critic 与 Evaluator PRE_RUN 要验证的关键事实。
9. 冻结完整 THUMOS14 N16R4 实验：seed、60 epoch/6000 updates、checkpoint 每 5 epoch、final/final-EMA、同提交 baseline 引用、官方 evaluator、成本上限、成功/失败门。
10. 若 Clock/Bridge 失败，下一假设应是语义 acquisition/训练成熟度，而不是继续堆叠 Query、Wasserstein、learned cross-attention 或 dynamic K。请明确 stop/pivot 条件。

## 需要返回的终稿

- 唯一科学决定；
- 一句话论文问题、机制与反主张；
- 冻结张量/时间坐标/Bridge 合同；
- 最小实现与审查链；
- 不重复既有对照的实验矩阵；
- 最便宜 falsifier 与正式实验门；
- 客观 stop/revise/continue 阈值；
- `next_owner / next_action / dependency / expected_return_at`。

当前两个 H65 Stage-2 作业仍在运行，不能把其终态 mAP写成已知。你的决定可以冻结实现设计，但任何新正式训练必须等待这些作业终态、独立 Critic PASS 和 Evaluator PRE_RUN_READY。

