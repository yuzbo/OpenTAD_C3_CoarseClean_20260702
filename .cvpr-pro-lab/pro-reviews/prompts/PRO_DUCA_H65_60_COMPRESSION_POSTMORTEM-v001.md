# DUCA H65 90→60 训练压缩终态科学复盘（fresh Project Pro turn）

## 路由与角色

- expected ChatGPT Project: `DUCA-RIME: Dynamic-Budget Task-Aware Temporal Acquisition for Offline TAD`
- exact Project ID: `g-p-6a796fef9a00819194024cf1de3bd697`
- nonce: `DUCA-H65-60-COMPRESSION-POSTMORTEM-v001-20260824`
- GitHub repository: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- frozen schedule-attribution revision: `ae3642a138c5b2e1ac2daad75a6d43d17cdb6c2f`
- commit: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/ae3642a138c5b2e1ac2daad75a6d43d17cdb6c2f
- branch: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-h65-60-lr-schedules-20260824

请作为本项目的 Scientific First-Author Agent 与 Primary Research Owner。你不是泛泛的建议者：请基于下面的终态证据，对“为什么 H65 从 90 轮压到 60 轮下降，以及还应不应该继续调整”作出唯一科学裁决。Codex 只负责有界实现、审查和证据反馈；路线、实验含义和停止条件由你决定。人类保留法律作者与最终投稿责任。

这是 DUCA Project 中的一次全新会话。不得依赖旧聊天；请从固定 Git revision、以下终态事实与代码路径重建判断。Project Sources 可能落后，本 prompt 的终态数值与 GitHub revision 是本轮权威增量。

## 已冻结的模型与对比边界

H65 是语义间接非均匀逐帧采样：低成本模型学习动作性与边界/转移证据，确定性地产生逐帧采样价值；本归因阶段固定 `K=384`。选中的真实 RGB 帧进入 VideoMAE-S/Adapter/ActionFormer，检测 loss、NMS、THUMOS14 split 与官方 evaluator 保持一致。动态 K、Query-Bridge、UVT、Fovea、SingleClock/TrueTime Bridge 都不是本轮处理变量，不得用这些新机制挽救训练日程子问题。

历史 H65 30+60 与压缩方案在 selector 机制、输入方式、重型骨干、检测器、loss 终值、seed=3407、211-video THUMOS14 validation 和 evaluator 上相同。变化是课程与优化暴露：

- 历史：Stage-1 exact-uniform 全模型训练 30 epochs / 3000 successful updates；Stage-2 joint training 60 epochs / 6000 updates。
- 20+40 压缩：Stage-1 20 epochs；Stage-2 40 epochs / 4000 updates，transition 20 + joint tail 20。
- 后续 30+30 归因：恢复成熟 Stage-1 30 epochs；Stage-2 固定 30 epochs / 3000 updates，只比较两个 LR schedule package。
- Stage-2 从 Stage-1 terminal EMA whole-model handoff；optimizer/scheduler/AMP/Stage-2 EMA 重建。scheduler、EMA、semantic/policy clock 仅随成功 optimizer update 推进。
- 参数组基础 LR 不变：detector `1e-4`，adapter `2e-4`，coarse trunk `1e-5`，action head `2e-5`，transition scorer `5e-5`；各组共享相对倍率。
- 30+30 中 semantic/policy transition 共 2000 updates；detector-feedback warmup 1000 + transition 1000；只剩约 1000-update full-joint tail。

## 终态真实结果

证据等级均为 `FULL_TRAINING / OFFICIAL_VALIDATION / SINGLE_SEED / TERMINAL_EMA`，不可当作多 seed 或论文结论。

| 方案 | Slurm | Stage-1+Stage-2 | Avg-mAP | mAP@0.7 | 末端趋势 |
|---|---:|---:|---:|---:|---|
| historical H65 reproduction | 1251782 | 30+60 | 65.1257 | 43.3137 | terminal epoch-59 EMA |
| compressed curriculum | 1251622 | 20+40 | 62.4648 | 39.9434 | terminal epoch-39 EMA |
| AM-RPCH25 | 1252979 | 30+30 | 63.22 | 41.25 | epoch24 63.35 → epoch29 63.22 |
| LongCosine-H6000 analogue | 1252980 | 30+30 | 63.56 | 41.01 | epoch24 63.58 → epoch29 63.56 |

20+40 相对 30+60：Avg `-2.6609 pp`，@0.7 `-3.3703 pp`。

30+30 LR 方案：

1. AM-RPCH25：500-update warmup，1000 plateau，1000 decay，500 hold，terminal factor 0.25；累计相对 LR factor exposure `1999.625`。
2. LongCosine-H6000 analogue：使用历史 6000-update cosine horizon 的前 3000 updates，terminal factor `0.571157`；累计 exposure 约 `2366.228`，比 A 高约 18.33%。

两者都没有达到预注册恢复邻域（Avg ≥64.6257 且 @0.7 ≥42.8137），都触发 clear failure（Avg <64.1257 或 @0.7 <42.3137），且 epoch29 Avg 不高于 epoch24。它们说明：恢复 Stage-1 成熟度、保留基础 LR、减慢 decay/保留非零尾 LR，仍不足以恢复 30+60。

## 可核验代码路径

- `configs/adatad/thumos/duca_h65_60_stage2_am_rpch25.py`
- `configs/adatad/thumos/duca_h65_60_stage2_longcosine_h6000.py`
- `scripts/run_duca_h65_60_lr_schedule_n16r4.sbatch`
- `tests/test_duca_h65_60_lr_schedules.py`
- `opentad/cores/scheduler.py`
- `tools/train.py`
- `opentad/cores/train_engine.py`

终态本地证据：

- `.cvpr-pro-lab/receipts/H65_CURRICULUM_TERMINAL_COMPARISON-v001.md`
- `.cvpr-pro-lab/receipts/H65_LR_SCHEDULE_TERMINAL_RESULTS-v001.md`

## 要你裁决的问题

请给出唯一 `CONTINUE / REVISE / PIVOT / STOP`，并明确对象是“60-epoch H65 无损压缩子问题”，不是否定 H65 语义间接选帧本身。

### 1. 因果诊断

按证据强弱解释 2–3 pp 下降。必须区分且不得虚构数值归因：

- 缺失的 3000 个 Stage-2 minibatch/AdamW/weight-decay 更新；
- semantic/policy、detector-feedback 和 full-joint exposure 被压缩；
- LR 曲线/累计更新剂量；
- 新 Stage-2 EMA 在 3000 updates 下的滞后；
- Stage-1 handoff 成熟度（20+40 与 30+30 的区别）；
- 高 IoU 更明显下降是否提示 boundary-support/feedback-clock 失败。

明确回答：现有结果是否已经足以拒绝“只改 LR decay 就能让 30+30 无损等价 30+60”。

### 2. 后续调整：只能选一个有界方向

如果 `STOP`：冻结 30+60 为性能 recipe，说明为什么不应继续参数组 LR sweep，以及下一项与论文主线相关、但不重复本 schedule 子问题的动作。

如果 `CONTINUE/REVISE`：只能提出一个最小、能区分假设的后续实验；不得给网格搜索。必须冻结：

- 是否仍受 60 epochs / 6000 total successful updates 约束；
- 精确 Stage-1/Stage-2 epochs 与 successful-update counts；
- semantic/policy/feedback/full-joint clocks；
- optimizer/EMA 是否继承或重建；
- 每参数组 LR 公式与 terminal factor；
- 唯一 primary checkpoint（terminal EMA）与不得中间选优；
- seed、官方 evaluator、5-epoch recovery checkpoint；
- 预注册 success/failure/stop threshold。

解释这个实验如何区分“joint exposure 不足”与“LR/EMA 路径不合适”。如果无法在同一 60-epoch 预算内区分或恢复，请直接 STOP，不要用增加 epoch 冒充压缩成功。

### 3. 只读诊断优先级

给出在不重训的前提下，最值得从现有 30+60、20+40、30+30 日志/checkpoint 提取的最多五项诊断。每项必须写：观察量、支持/反驳哪个原因、什么结果会改变下一步。优先考虑：对齐 update=2000/2500/3000 的历史轨迹、online-vs-EMA gap、未加权 boundary loss、参数组梯度/位移、selector 熵/最大空洞/边界覆盖。

### 4. 返回可执行合同

给出：

- active claim / anti-claim / cheapest falsifier；
- Builder（若需要）的最小配置改动或明确 idle；
- Independent Critic 要核验的身份与因果隔离事实；
- Evaluator 的 PRE_RUN/结果准入门；
- `next_owner / next_action / dependency / expected_return_at`。

不要：重复 dense/uniform/random 对照；引入新的模型模块；把 65.696 当 H65 matched 结果；把单 seed 当统计结论；把日程优化写成论文创新；根据中间 checkpoint 选优；推断尚未测量的 FLOPs/效率或显著性。
