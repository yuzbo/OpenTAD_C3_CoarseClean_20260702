# DUCA 全模型矩阵、实现正确性与同预算增益路线：最终对抗性科学审查

**Expected exact Project:** `g-p-6a796fef9a00819194024cf1de3bd697`（DUCA）  
**Turn nonce:** `DUCA-ALL-MATRIX-SAME-BUDGET-v001-20260824`  
**Primary repository:** https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702  
**Primary frozen remote revision:** `b2ccfccab5b4912b59954afcc9b0364955327f7c`  
**Primary branch:** `codex/duca-h65-firstmix-singleclock-20260824`

## 0. 你的职责与输出约束

请作为本项目的 **Scientific First-Author Agent and Primary Research Owner**，同时以最严厉的 CVPR 审稿人和具有科研审美的时序动作检测研究者视角，对当前全部 DUCA 矩阵作一次独立、实现感知、结果感知的裁决。Codex 仅作为受限的实现与证据反馈系统：Builder 实现，Critic 攻击，Evaluator 测量，Coordinator 只路由材料。科学路线选择由你完成。

这是 exact DUCA Project 中的一次全新会话，不依赖旧聊天。以下内联材料和 GitHub 固定提交是本轮唯一上下文。不要把选择退回给用户或 Codex；请返回且只返回一个明确的 `CONTINUE / REVISE / PIVOT / STOP`（或无歧义等价词），并冻结一个最短、最可证伪、同训练资源下有机会超过当前最强可比模型的下一步。

你必须回答：

1. 各实验最初要回答的问题是否设计合理；
2. 根据可访问的固定代码提交，核心实现是否忠实、是否存在会改变结论的错误；
3. 现有结果能支持什么，不能支持什么，哪些结论被混杂因素误导；
4. “相同训练资源或轮次”应如何操作化；
5. 在不重复无效实验的前提下，下一项唯一实验和最小实现是什么；
6. 给出 claim、anti-claim、falsifier、停止规则，以及 Builder→Critic→Evaluator 的有界交接。

不得仅给泛化建议、再开一轮理论讨论或要求人类选择。不得把未终态 Job 推断成结果。

## 1. 共同科学问题

DUCA 研究的是离线时序动作检测中的任务感知稀疏视频计算：低成本 scout 预测动作性和边界重要性，确定性地形成非均匀逐帧采样价值；高成本 VideoMAE/AdaTAD 仅处理被选高分辨率帧，并在 NMS 前恢复真实物理时间。长期论文目标要求 dynamic outer-K；固定 K 只能作为机制归因、基线和 fallback。

当前最强、最接近可比的 DUCA 合同是 H65：固定 `K=384`、语义间接非均匀逐帧选择、真实 RGB hard gather、selected-rank VideoMAE 输入、检测前 q→physical time 回映。历史高分来自复合训练协议，不允许把它简化为“某个单模块带来 65%”。

## 2. 代码与实验身份总表

### 2.1 官方 dense 参照（共享，不重复训练）

- Official repository: https://github.com/sming256/OpenTAD
- Official AdaTAD release commit: https://github.com/sming256/OpenTAD/commit/01c58b9f2370e914150cf94d392208a4e211c053
- 共享 evaluation Job `1245842`，THUMOS14 official dense、seed 42、60 epochs。
- Avg-mAP `68.73%`；公开 anchor `69.03%`。
- 它是背景上限，不是下面任意本地方法的 same-commit causal baseline。

### 2.2 历史 H65 与当前 matched H65

- Historical branch: `codex/duca-density-transport-20260723`
- Historical commit: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/42dba3f90b37243e7965d18b6707e88e81bf7109
- Historical Job `1193610`，30-epoch exact-uniform Stage-1 + 60-epoch Stage-2，seed 3407，terminal EMA。
- Historical Avg-mAP `65.385724%`; @0.3–0.7 = `80.1932/75.6625/68.6072/58.5818/43.8840`。
- 当前 matched OFF branch: `codex/duca-h65-60-curriculum-20260823`
- Commit: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/04c35a3b76897e6c1569eeede41ed3aecaf7f854
- Job `1251782`，30+60、seed 3407、terminal epoch-59 EMA。
- Avg-mAP `65.1257%`; @0.3–0.7 = `80.2808/75.7109/68.5475/57.7757/43.3137`。

H65 的复合合同包括：Stage-1 exact-uniform detector warmup；Stage-2 systematic exact-K 非均匀选择；训练期 50% uniform companion；contribution distillation；ASFormer action/transition adaptation；selected-axis detector；NMS 前物理时间回映。该结果不能归因于任何一个组件。

关键代码请直接审查：

- `opentad/models/selectors/duca_online_frame_selector.py`
- `opentad/models/backbones/backbone_wrapper.py`
- `opentad/models/detectors/two_stage.py`
- `opentad/models/detectors/actionformer.py`
- H65 configs/launchers under `configs/adatad/thumos/` and `scripts/`

### 2.3 60-epoch 压缩及 LR 归因（该子问题已 STOP）

- 20+40 commit: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/87ff0883651a631d48468ab4f9d6392f587c15e4
- Job `1251622`; Avg-mAP `62.4648%`; @0.3–0.7 = `78.0914/73.4479/65.0772/55.7639/39.9434`。
- 相对 matched 30+60 OFF：Avg `−2.6609 pp`，@0.7 `−3.3703 pp`。
- LR attribution branch: `codex/duca-h65-60-lr-schedules-20260824`
- Commit: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/ae3642a138c5b2e1ac2daad75a6d43d17cdb6c2f
- AM-RPCH25 Job `1252979`: Avg/@0.7 = `63.22/41.25`。
- LongCosine-H6000 Job `1252980`: Avg/@0.7 = `63.56/41.01`。
- 两者均复用成熟 30-epoch Stage-1，只给 Stage-2 30 epochs / 3000 successful updates；较慢 decay 和更大 LR 面积未恢复 30+60 高 IoU。

已接受结论：`STOP_60_EPOCH_COMPRESSION`。冻结 30+60 terminal-EMA 作为当前 H65 性能配方；不得继续 LR/warmup/cosine/hold/terminal-factor/stage-ratio sweep。H65 语义间接非均匀选择本身没有被该负结果否定。

### 2.4 RankPack 与 TrueTime

- Branch: `codex/duca-truetime-curriculum-v3-20260822`
- Commit: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/11126684af779aa2916a68ecf617c4f14c805478
- 同 selected RGB、同 K=384、同 seed 3407、60 epochs/6000 updates、terminal EMA。
- RankPack Job `1248822`: Avg `61.5722%`; @0.3–0.7 = `78.6567/73.8490/65.3328/52.9221/37.1003`。
- TrueTime Job `1248823`: Avg `62.1930%`; @0.3–0.7 = `78.7428/74.2565/65.4630/54.6107/37.8918`。
- TrueTime − RankPack: Avg `+0.6208 pp`, @0.6 `+1.6885 pp`, @0.7 `+0.7915 pp`。
- 两个 Slurm 顶层状态因 post-run config-hash seal mismatch 为 `FAILED 1:0`，但训练、prediction、official metric 已终态且独立数值复算一致；因此是部分机制证据，不是 paper-ready。

相关路径：

- `opentad/models/backbones/physical_time.py`
- `opentad/models/utils/truetime_geometry.py`
- `opentad/models/detectors/single_stage.py`

### 2.5 DUCA-UVT

- Branch: `codex/duca-uvt-utility-value-20260819`
- Training commit: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/df544c78ce515d925dc7019f106fce09a53c09f8
- Job `1244840`，三臂、seed 3407、60 epochs。
- `off=57.35%`, `geo=55.93%`, `geo_ema=55.92%` Avg-mAP。
- `geo`/`geo_ema` 同时改变 selection score（`fused+0.10·V`）与 dynamic-K evidence（`mean sigmoid(V)`），因此负结果不能单独归因于 V-score 或 V-budget。
- boundary-foveated decoder 与 portal feedback 均关闭；单 seed；无同提交 dense/uniform/random matched causal baseline。

相关路径：

- `configs/adatad/thumos/duca_uvt_thumos.py`
- UVT selector/value-head implementation under `opentad/models/`

### 2.6 FoveaSampler / Query-Bridge

- Branch: `codex/duca-fovea-query-bridge-20260819`
- Training commit: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/4ae5067100c4490c7110c00a1ad406230ba603cd
- Job `1244851`，seed 3407、60 epochs。
- Avg-mAP: `baseline_fused=42.94`, `query_only=45.26`, `query_gt_mask=49.16`, `query_cycle=54.67`, `query_fovea=43.77`。
- `full` 与 `query_fovea_dpp` 未运行。
- 训练用 Gumbel-TopK straight-through；推理用 top-k + boundary quota + greedy MMR，存在 train/inference policy mismatch。
- `query_only` 实质为 Query 直接贡献帧选择；GT-mask/cycle/Query/动态预算为复合变化；全部臂共享低性能动态预算背景，不能据此否定 Query representation 或协同学习思想本身。

相关路径：

- `configs/adatad/thumos/duca_fovea_qb_thumos.py`
- `opentad/models/selectors/fovea_query_bridge_selector.py`
- `tests/test_fovea_query_bridge.py`

### 2.7 连续 cliplet 路线

- Local branch: `codex/duca-physical-cliplet-contig-s0-v001`
- Local training revision: `8a6e7ea23b5389fbbd071820f43eb194b39ae5cd`
- 该提交来自独立本地工作树，**没有可验证的远端 GitHub commit URL**，所以你不能声称已逐行审查该代码；只能审查给出的合同和终态结果。
- Jobs `1246284/1246285`，seed 3203700、60 epochs、24 个连续 16-frame cliplets，实际 heavy input 约 K=384。
- FZ/JT Avg-mAP `49.89/47.24`; FZ @0.3–0.7 `65.04/59.98/52.51/42.22/29.68`; JT `63.51/57.89/49.69/38.57/26.52`。
- 它证明该连续片段合同下定位严重退化，不证明 scout、物理时间或稀疏计算总体失败。

### 2.8 当前 First-Mixing SingleClock Unit-1

- Branch: `codex/duca-h65-firstmix-singleclock-20260824`
- Training commit: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/08a817e91867839abf3a81e24f8469512b26a6ea
- Remote evaluation commit: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b2ccfccab5b4912b59954afcc9b0364955327f7c
- 本地 clean Gate-v2 HEAD `61065a50dc044f54876de5abfcad8c25559fede4` 比远端多两个 gate/statistics commits；它不改变训练模型，但不能假装 Pro 已通过 GitHub 读取。
- Training Job `1252482` 已完成 60 epochs / 6000 successful updates。
- Unified evaluation Job `1253090` 仍为 `RUNNING`，没有合法 Gate-v2 终态。
- 当前已有四个 official validation family：
  - final ON Avg `64.0447%`; @0.3–0.7 `79.1480/74.5697/67.4004/56.9816/42.1239`
  - final gate-zero Avg `64.0404%`; `79.1526/74.5534/67.4066/56.9679/42.1217`
  - EMA ON Avg `64.4661%`; `79.6738/74.8415/67.3016/57.3720/43.1417`
  - EMA gate-zero Avg `64.4606%`; `79.6761/74.8206/67.2872/57.3753/43.1436`
- 这四项只是 same-checkpoint dependence/coadaptation diagnostics。当前 ON−gate-zero 近零，不能替代 H65 OFF 主对照，也不能发布 PASS/KILL。
- H65 OFF final/EMA 尚无本运行根的合法 terminal artifact；10k paired bootstrap/strata/finalizer 也未终态。
- 更关键的是，当前缺失：
  1. H65 replay 五边界 identity（indices、gathered RGB、VideoMAE input、raw selected-q proposal/scores/labels、canonical evaluator JSON）；
  2. nominal-uniform first-mixing 与 final backbone bit identity；
  3. machine-readable independent implementation-review receipt。
- 因此当前状态必须是 `EVIDENCE_ADMISSION_BLOCKED`，不是 SingleClock scientific KILL，也不是 efficacy PASS。

关键实现路径：

- `opentad/models/utils/temporal_grid.py`
- `opentad/models/backbones/vit_adapter.py`
- `opentad/models/detectors/actionformer.py`
- `tools/bata/finalize_duca_h65_singleclock_terminal.py`
- `tools/bata/analyze_duca_h65_singleclock_strata.py`
- `tests/test_duca_h65_singleclock_*`

实现合同：全局 canonical K384 clock 先在视频/窗口上生成，再按 rank 切成 24×16；actual−canonical relative residual 在 exact canonical-uniform 时严格为零；零初始化标量只在 block 0 第一次时序混合进入 attention bias；selected positions 必须严格递增。旧 clip-local `exact_uniform_positions(768,16)` 实现已关闭，不得混入当前判断。

## 3. 已冻结的 Unit-1 科学门

主估计量只允许：terminal EMA `SingleClock ON − H65 OFF/replay EMA` 的三个 point deltas：Avg-mAP、mAP@0.6、mAP@0.7。三项均要求 inclusive `>= −0.20 pp`。bootstrap CI 仅报告，不进入 PASS/KILL。

same-checkpoint gate-zero 只诊断模型是否依赖 clock 路径及 coadaptation；final 只诊断 EMA sensitivity；cost 仅报告；它们都不进入 Unit-1 hard decision。

hard identities：

1. H65 replay 五边界身份失败 → `evidence invalid`，无科学 decision；
2. ON/gate-zero same-checkpoint input identity 失败 → `evidence invalid`；
3. nominal-uniform first-mixing/backbone bit identity 失败 → `KILL_SINGLECLOCK_REPRESENTATION`。

若预先封存材料足够，high-gap-CV 与 high-boundary-density 两个 training-cutpoint strata 的 duration-normalized unmatched-penalized boundary error 都必须不恶化；材料不足则明确 diagnostic unavailable，不得补跑未注册推理或伪造 PASS。

Unit-1 无论 PASS/KILL 都不是 paper claim；PASS 只允许进入 Unit-2 semantic Query residual，仍不授权 dynamic-K。

## 4. 你必须攻击的核心问题

### A. 实验设计合理性

1. 这些矩阵分别回答了什么单一问题？哪些矩阵从一开始就是复合诊断而非可识别因果实验？
2. H65 30+60 的高分是否主要反映训练暴露量、优化课程、selected-axis compatibility，还是语义间接采样？现有证据最多能将哪些解释排序，不能虚构概率。
3. RankPack/TrueTime 的正增量是否足以支持“物理时间在高 IoU 有价值”，还是仍被 representation/geometry/receipt 层面限制？
4. UVT 与 Query-Bridge 的大幅下降，最可能来自动态预算、伪连续 VideoMAE clip、训练/推理选择策略不一致、直接 Query 选择、后端 geometry、优化冲突中的哪些组合？哪些判断是代码事实，哪些只能由新实验区分？
5. SingleClock 是否是当前最小、最干净地吸收 TrueTime 和 Query 思想的路线，还是其 near-zero ON−gate-zero 已经构成足够的 mechanism warning？

### B. 实现正确性

请直接阅读所有可访问的固定 GitHub commits，至少核查：

- hard selected RGB 是否真实减少 VideoMAE 重计算，而非 padding/metadata 假动态；
- 非连续 selected frames 如何被 reshape 为 16-frame VideoMAE clips，是否产生 pseudo-contiguous temporal semantics；
- q→physical 映射是否在 threshold/top-k/NMS 前后处于正确位置；
- H65 的 selected-rank、physical-time 与 SingleClock first-mixing residual 是否自洽；
- UVT 的 V-score/V-budget 是否纠缠；
- Fovea train-time Gumbel 与 inference MMR/quota 是否同一策略族；
- dynamic-K 的 executed K 是否来自实际 heavy workload；
- fixed-K/actionness-only/actionness+boundary/direct-query 六臂是否真正隔离；
- terminal EMA、successful updates、checkpoint 和 evaluator 是否绑定。

对 inaccessible/local-only commit 必须明确说“无法逐行验证”，不能用文字合同冒充代码核验。

### C. 结果与结论误区

必须区分：

- same-commit matched causal pair；
- cross-commit descriptive comparison；
- single-seed development evidence；
- post-run seal failure但 metric 可复算；
- Job running/partial；
- evidence-invalid；
- static tests / PRE_RUN / cost plumbing；
- official dense background anchor。

不要把不同训练 epochs、successful updates、LR trajectory、parameter exposure、EMA rule、seed、geometry、evaluator、NMS、数据 split 或 selected-axis/physical-axis 的数值放在同一因果 delta 列。

## 5. “同训练资源或轮次”的冻结候选定义

请裁决并给出唯一可执行定义。至少考虑：

- exact code/config/evaluator identity；
- identical data split, seed and loader exposure；
- identical number of successful optimizer updates，而不是只看 epoch；
- identical LR/scheduler trajectory and EMA update count；
- identical Stage-1/Stage-2 boundary and trainable parameter groups；
- identical terminal checkpoint selection rule；
- identical detector/loss/NMS/time mapping；
- 同样的训练 GPU-hours 是否必须进入“同资源”，以及 inference full-stack cost 如何单列。

已知约束：如果“同资源”严格限制 total 60 epochs/6000 updates，当前最好已测值只有 `63.56%`，没有支持超过 H65 65.39 的配方；已接受的 Pro 决策已停止继续 schedule sweep。若采用 30+60 H65 作为方法基座，则新增机制必须共享其最大更新预算，不得额外延长训练或用中间 validation 选 checkpoint。

## 6. 请冻结唯一下一路线

请在以下状态上做唯一决定，不要同时批准多条主路线：

1. `RETAIN/CONTINUE First-Mixing SingleClock gate`：只消费/补齐合法已有身份与终态，不重训；若 Gate-v2 PASS，再设计 Unit-2 semantic Query residual；dynamic-K 最后；
2. `REVISE`：保留 H65 30+60 与 SingleClock 表示思想，但命名一个更小、claim-preserving、同预算实现/测试；
3. `PIVOT`：若你认为 SingleClock 从机制上已不值得继续，命名一个严格 H65-compatible、不会回到 cliplet、直接选帧、动态预算混杂或 LR sweep 的单一路线；
4. `STOP`：若所有新增结构在当前证据下都不值得继续，明确应发表/记录的负结论和最小剩余实验。

你需要回答：

- 哪个观察会让 TAD 审稿人意外？
- 什么机制解释它？
- 相比 H65/fixed-K baseline，它产生什么新预测？
- 哪个最便宜的真实实验能证伪？

严禁建议重复 official dense、uniform fixed-K、random 或已完成的 H65 schedule/LR sweep。严禁恢复连续 cliplet。严禁在 representation gate 未闭合前引入 dynamic-K。不得用 local CPU/synthetic/static test 作为效能证据。

## 7. 输出必须包含

1. 唯一科学裁决；
2. 对每个矩阵的设计判定、实现判定、结果判定；
3. 最可信的性能下降原因排序，明确哪些为事实、哪些为待证假设；
4. “同资源”唯一操作定义；
5. 唯一下一模型/实验的精确张量与时间轴合同；
6. allowed files/symbols、forbidden changes、最小 Builder patch；
7. Critic 必查项；
8. Evaluator PRE_RUN 与最便宜 falsifier；
9. 统计量、point/CI 角色、阈值和停止规则；
10. 结果到 claim 的边界；
11. `next_owner / next_action / dependency / expected_return_at / single_recovery`。

如果当前 SingleClock 身份缺失使科学裁决不可合法形成，请不要泛化讨论；请明确是 `evidence invalid`，并只列出能从既有 artifact 恢复的最小身份动作。不得授权重训、Query 或 dynamic-K 来绕过身份失败。

