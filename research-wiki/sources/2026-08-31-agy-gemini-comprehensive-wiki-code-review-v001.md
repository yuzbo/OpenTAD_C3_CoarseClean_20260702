---
type: external_model_review
date: 2026-08-31
model: Gemini 3.7 Flash
effort: high
scope: complete DUCA Wiki history and cited implementation snapshots
---

# AGY Gemini 全历史与代码独立咨询原文

本文件保存 Gemini 通过 `agy` 命令行以 `gemini-3.7-flash-high`、`effort=high`
完成的完整只读咨询原文。它是提交 Pro 科学裁决前的外部技术意见，不自动成为
项目事实、路线裁决或实验授权。Pro 阅读时必须同时参考当前研究 Wiki、精确 GitHub
提交及事实校正材料；其中把未隔离机制写成“已证实根因”的句子只能作为待检验假说。

I am inspecting the specified exact commits and files in read-only mode to complete the comprehensive DUCA analysis.
# DUCA 全历史、代码与论文路径独立技术咨询报告

**角色声明**：本报告为提交 Pro 科学裁决前的独立只读技术咨询报告。工作区保持完全只读，未修改任何文件、未执行训练或远端作业、未访问 held-out 动作标签/未封存指标、未调用任何浏览器。全部分析基于本地 worktree 代码核验、Wiki 历史记录与随附证据链。

---

## 1. `VERIFIED_FACTS_AND_CORRECTIONS`

### 1.1 主要历史路线与基准事实总表

| 路线 / 模型分支 | 提交 / 作业标识 | 评估预算与协议 | 最终性能指标 (Avg-mAP / @0.7) | 证据等级 | 状态与终止 / 暂缓原因 | 事实核验与冲突纠正 |
|---|---|---|---|---|---|---|
| **Shared Dense AdaTAD** | `01c58b9f` (seed 42, 60ep) | 768 heavy observations, 60 epochs | Avg `68.73%` (公开锚点 `69.03%`) | 参考上界 (外部基线) | `RETAIN_BASELINE` (只读参考) | 仅代表 100% 重型计算上界；缺同硬件/同环境完整端到端时延、显存与能耗对照。 |
| **H65 Clean Base (Fixed K384)** | `04c35a3b` / Job `1191957` | 384/768 heavy observations | Avg `65.1257%` / @0.7 `43.3137%` (30+60 terminal: `65.3857%` / `43.8840%`) | 可靠稀疏基线 (单种子) | `RETAIN_BASELINE` (当前主基座) | **重要事实纠正**：H65 采用 30 轮 Stage-1 + 60 轮 Stage-2 = 总计 90 轮 (9,000 updates)，相对 60 轮 uniform (`64.49/42.45`) 的 +0.90 pp 增益受训练预算不匹配混杂，不能公平归因于 selector。H65 不是全局 Top-K，而是预算校准采样 (BCSR)。 |
| **Old E2E Fixed K384** | `70aa069` / Job `1154971` | 384 heavy obs, joint training | Avg `58.39%` / @0.7 `34.53%` | 混杂历史诊断 | `STOP` (已废弃) | **重要事实纠正**：旧端到端失败主因是 5,940 vs 13,080 loader 更新数严重不匹配、随机粗分支 stem 与软代理偏移，不能作为联合训练一般不可行的科学证据。 |
| **CellCF** | `1642f26` / Jobs `1167479/480/485` | 384 obs, 132ep 诊断 / 60ep 匹配 | Uniform `63.8594%`, Trans-β0 `64.2755%`, CellCF `64.0610%` | 正式科学负结果 | `STOP` (禁止重跑) | CellCF 低于 transition-beta0 控制 (-0.2145 pp)；one-per-cell 空间无法跨区域转移预算；detached utility 并非真正检测梯度。 |
| **Protected-E2E & Homotopy** | `b3222af`, `cb89586` / Jobs `1177779-1177782` | 384 obs, 60ep matched | Uniform `64.4580%`, Direct `63.7102%`, Homotopy `63.0601%`, Companion `63.6931%` | 正式科学负结果 | `STOP` (禁止重跑) | 相同物理 DAG 下所有学习臂均显著低于 exact uniform 控制。 |
| **Boundary-Burst R0 Oracle** | `e49ef69` / Job `1179517` | 124 windows training holdout | Uniform `93.5871%`, R2Q3 `94.1905%`, R4Q5 `93.9992%`, Unrestricted `93.9701%` | 机制诊断门负结果 | `STOP` (禁止重跑) | 内部 privileged oracle 在成对 bootstrap 下未过预注册门；否定该投影可行域假设。 |
| **Fast-only SlowFast + R2Q3** | `4c5604b4` / Job `1180653` | 384 obs, 60ep | Avg `63.5297%` / @0.7 `42.0937%` (vs Uniform `64.49/42.45`) | 明确负结果 (特定先验) | `STOP` (禁止重跑) | 冻结 Fast pathway 动作先验导致 Avg -0.9603 pp 下降。 |
| **Training Compression** | `6f2ed48`, `6c56e11` / AM-RPCH25, LongCosine | 384 obs, 20+40 / 30+30 | Avg `62.46%` ~ `63.56%` / @0.7 `39.94%` ~ `41.25%` | 确定性负结果 | `STOP` (禁止重跑) | 简单缩短预热或仅调整学习率尾部均无法恢复 H65 性能。 |
| **PhysTime v1 & Full60** | `0dc5851` / Jobs `1170946/1170947` | J192 native tubelet, 60ep | Selected-axis `41.28%`, Physical-metric `57.57%` (+16.29 pp) | 局部架构支持 (单种子) | `RETAIN_DIAGNOSTIC` | v1 PhysTime 为负 (`57.21%` vs `63.61%`)；G1 full60 证明物理时间在特定低性能结构中有效，但不能直接外推为 H65/Dense 上界。 |
| **TrueTime / RankPack** | `11126684` | 384 obs, 30+60 matched | RankPack `61.5722%` / `37.1003%`, TrueTime `62.1930%` / `37.8918%` (+0.6208 / +0.7915) | 部分机制支持 (单种子) | `RETAIN_DIAGNOSTIC` | 相同 Scout、K、课程下有正向点估计，但单种子且配对区间未闭合。 |
| **PJST-D1** | `c73e8418`, `7bd120f0` | 211 videos, H65 frozen selections | OFF `65.063%` / `43.646%`, ON `64.591%` / `43.769%` (-0.472 / +0.123) | 未闭环证据 (评测故障) | `HOLD` (等待科学裁决) | **重要事实纠正**：211 视频预测生成完整，但 finalizer 因预测路径错误使 10,000 次 bootstrap 退出 (0 次重采样)。当前只有点估计、无 CI，**不得表述为“显著负向”**。 |
| **Native Tubelet Coreset** | `b3339112` / Jobs `1260184/1260185` | 192 tubelets (384 frames), 60ep | Uniform `64.13%` / `42.45%`, Coreset `62.81%` / `40.56%` (-1.32 / -1.89) | 诊断性负向点估计 | `STOP` (细粒度 coreset) | 检验的是任务状态 coreset 选择策略而非配对连续性；因 `save_dict=False` 导致预测未保存，缺配对 CI 与实测成本。 |
| **Sparse Probe Hidden-Linear** | `dd3c97cf`, `cee4ccd` / Gate `1180556`, Suite `1180557`, Job `1180696` | Stride d=1/2/3/4, 20ep P0 + 60ep | Gate MACs 下降；Suite 因 BatchNorm buffer 报错；Recovery 因 `VARIANT_CONFIGS` 映射中断 | 工程中断 (0 mAP) | `ARCHIVE` (无 TAD mAP) | 证明了计算量下降与数值有限性，但未获得任何正式 60 轮 TAD mAP，属于实现中断，非科学负结果。 |
| **Coverage-v1** | `04814312` / Job `1261679` | K384 facility location coverage | Set delta `48.05%` (<80%), Anchor coverage `+3.32%` (<10%) | 预运行机制门失败 | `STOP` (禁止降门重跑) | 未通过无标签预运行机制门，在 smoke 前终止，无 mAP 产物。 |
| **Marginal-v1 / Cap-Release / 96-State** | `be5bb803`, `f67d96fd`, `d2fad7c0`, `46812fac` | 40-video holdout, 124 windows, 47,110 obs | Capped `+0.726/+0.729 pp`, Released `+0.427/+0.450 pp`, 96-state `+0.554/+0.933 pp` | 开发集 Oracle 证伪 | `STOP` (冻结动作空间) | 96/96 状态均未达到 `+0.8/+1.0` 联合门；证明加性窗口 utility 与整视频 Soft-NMS 不一致。 |
| **Whole-Video 704-State Transfer** | `33e4ed13` / Job `1262190` | 40-video holdout, 704 legal pairs, ≤47,110 obs | 最佳联合: `+0.1474/+0.4898 pp` (通过数 0/704) | 开发集 Oracle 证伪 | `STOP` (冻结动作空间) | 彻底否定在冻结 K384 检测器上进行跨视频预算转移的可行性。 |
| **Multi-Budget Detector Adaptation** | `DUCA-FULL-DATA-COMPARABLE-PROTOCOL-v001-20260831` | Full 200 train, 6,000 updates, K256/384/512 | 尚未执行 (设计冻结) | 待执行科学假说 | `CONTINUE_CANDIDATE` (唯一候选) | 检验训练期多预算暴露能否解决跨预算表示不匹配；前置阻塞为 211/212 数据核验。 |

---

## 2. `CODE_MECHANISM_AUDIT`

对指定工作区与提交的精确代码实现进行机制忠实度核验：

### 2.1 H65 干净基座与课程训练 (`04c35a3b`)
- **Scout 与状态证据**：[`opentad/models/duca/acquisition.py#L48-L57`](file:///C:/Users/skywalker/.codex/worktrees/duca-full-data-identity-audit-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/duca/acquisition.py#L48-L57) 定义了 7 维 deploy-visible 状态特征 (`p_action`, `uncertainty`, `entropy`, `delta_p_action`, `abs_delta_p_action`, `uncertainty_peak`, `transition_score`)。
- **预算校准采样 (BCSR)**：[`opentad/models/duca/acquisition.py#L824`](file:///C:/Users/skywalker/.codex/worktrees/duca-full-data-identity-audit-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/duca/acquisition.py#L824) 采用 `budget_calibrated_sampling_rate`，通过优先级调制在保证全局均匀覆盖下限的同时向高变化区域倾斜，输出由 [`SparseTemporalGrid`](file:///C:/Users/skywalker/.codex/worktrees/duca-full-data-identity-audit-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/duca/acquisition.py#L268-L380) 严格验证单调递增性与预算上限。
- **Stage-1 预热**：[`configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py#L8-L26`](file:///C:/Users/skywalker/.codex/worktrees/duca-full-data-identity-audit-v1-20260831/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py#L8-L26) 显式锁定 `policy_alpha=0.0`，前 30 轮仅在 exact-uniform K384 上训练检测器并优化 coarse actionness 与 transition 监督。
- **Stage-2 联合适应**：[`configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py#L58-L106`](file:///C:/Users/skywalker/.codex/worktrees/duca-full-data-identity-audit-v1-20260831/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py#L58-L106) 从 Stage-1 `epoch_29.pth` 的 `state_dict_ema` 初始化，重置优化器/EMA/时钟，在前 3,000 次更新平滑释放 `policy_alpha` (0→1.0) 与检测器梯度 (0→0.25)。

### 2.2 TrueTime 物理时间残差 (`11126684`)
- **残差构建**：[`opentad/models/duca/true_time_residual.py#L10-L35`](file:///E:/DeskTop/TAD/OpenTAD_DUCA_TrueTimeCurriculumV2_20260822/opentad/models/duca/true_time_residual.py#L10-L35) 实现 `TrueTimeFeatureResidual`，末层线性权重零初始化 (`nn.init.zeros_(self.projector[-1].weight)`)，保证初始状态严格为恒等映射。
- **时序几何描述子**：[`true_time_residual.py#L37-L60`](file:///E:/DeskTop/TAD/OpenTAD_DUCA_TrueTimeCurriculumV2_20260822/opentad/models/duca/true_time_residual.py#L37-L60) 的 `_row_descriptors` 构造 4 维物理特征：归一化位置 `pos/denom`、左间隔 `left/denom`、右间隔 `right/denom` 及局部不对称度 `asymmetry=(right-left)/(right+left)`。
- **特征注入**：[`true_time_residual.py#L98-L119`](file:///E:/DeskTop/TAD/OpenTAD_DUCA_TrueTimeCurriculumV2_20260822/opentad/models/duca/true_time_residual.py#L98-L119) 将投影残差加到 VideoMAE 提取的 tokens 上，不改变下游 ActionFormer 检测头结构。

### 2.3 动态预算与整视频一致性 Falsifier (`33e4ed13`)
- **短窗口预算折叠**：[`opentad/models/duca/dynamic_budget.py#L147-L187`](file:///C:/Users/skywalker/.codex/worktrees/duca-whole-video-consistent-budget-falsifier-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/duca/dynamic_budget.py#L147-L187) 的 `marginal_budget_accounting` 正确实现 `actual = min(valid_obs, requested_budget)`，将短窗口无效请求折叠回 baseline，并强制 16 帧 packet 对齐。
- **真实变长执行与验证**：[`dynamic_budget.py#L218-L266`](file:///C:/Users/skywalker/.codex/worktrees/duca-whole-video-consistent-budget-falsifier-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/duca/dynamic_budget.py#L218-L266) 的 `validate_real_heavy_observation_tensor` 严格检查变长输入，禁止非 baseline 的额外 padding。
- **整视频转移与生成顺序保全**：[`tools/bata/run_duca_whole_video_consistent_budget_falsifier.py#L129-L138`](file:///C:/Users/skywalker/.codex/worktrees/duca-whole-video-consistent-budget-falsifier-v1-20260831/OpenTAD_C3_CoarseClean_20260702/tools/bata/run_duca_whole_video_consistent_budget_falsifier.py#L129-L138) 显式修复了父提交 `c27d77` 重新排序 sample ID 破坏 Soft-NMS 结果的缺陷，保持了原始生成顺序与锚点复现误差 `0.0 pp`。

### 2.4 原生 Tubelet 与稀疏插值
- **Native Tubelet (`b3339112`)**：768 帧按 2 帧一组组成 384 个 tubelet，选 192 个打包成 24 个 16-frame VideoMAE clip。机制缺陷：跨 tubelet 打包将时间不相邻的 tubelet 强行拼在同一 clip 内，破坏了 3D 卷积的连续局部感受野。
- **Sparse Probe Interpolation (`dd3c97cf`)**：空间 stem + ASFormer 在 d=1/2/3/4 stride 下运行，将时序隐藏特征线性插值回 768 点。机制健全，但工程上因非参数 BatchNorm buffer 与 variant 映射在训练前 fail-closed，未进入检测器训练。

---

## 3. `ROUTE_DISPOSITION`

| 路线名称 | 科学/工程处置 | 处置理由 | 明确禁止的重复操作 |
|---|---|---|---|
| **CellCF / Local-Cell** | `STOP` | 结构上无法跨区域转移预算；detached 效用蒸馏低于 transition-beta0。 | 禁止任何形式的 per-cell 局部微调或重调损失权重。 |
| **Protected E2E / Homotopy** | `STOP` | 同物理 DAG 下连续同伦与 direct bridge 的 60 轮结果均低于 uniform。 | 禁止重跑 soft-to-hard 或 surrogate 连续反传。 |
| **Fast-only / 外部视频先验** | `STOP` | 预训练 Fast pathway 在 THUMOS 表现明确负向 (Avg -0.96 pp)。 | 禁止在 pre-backbone 引入未经适配的密集视频骨干。 |
| **Coverage-v1 (设施选址)** | `STOP` | 机制门未通过 (集合变化 48%<80%，覆盖增益 3.3%<10%)。 | 禁止降低阈值、调参数 M/σ/K 或引入 gap repair 抢救。 |
| **Marginal-v1 / 96-State / 704-State** | `STOP` | 冻结 K384 检测器上的窗口/联合/整视频转移 0/704 通过，已穷尽动作空间。 | 禁止在冻结检测器上继续搜索启发式转移规则或重构 oracle。 |
| **Native Tubelet Coreset** | `STOP` | 任务状态驱动 coreset 相比同 tubelet uniform 降低 -1.32 pp Avg / -1.89 pp @0.7。 | 禁止调整打分权重、端点规则或最大空洞重跑 coreset。 |
| **H65 Clean Base (Fixed K384)** | `RETAIN_BASELINE` | 当前最稳定可靠的稀疏 baseline (Avg 65.13%)。 | 禁止将其 30+60 (90 轮) 结果直接作为 60 轮公平课程增益写入论文。 |
| **Shared Dense AdaTAD** | `RETAIN_BASELINE` | 官方 768-observation 性能上界 (Avg 68.73%)。 | 禁止在未统一硬件/端到端评测标准前直接声称算力减半。 |
| **PhysTime / TrueTime 物理几何** | `RETAIN_DIAGNOSTIC` | TrueTime 有 +0.62 pp 点估计增益；G1 full60 证明物理时间在非等间隔下的必要性。 | 禁止从头重跑 selected-axis 对比；禁止在未闭合单变量前引入复杂 RoPE。 |
| **Sparse Probe / 重构核消融** | `RETAIN_DIAGNOSTIC` | 稀疏探针与隐藏特征线性重建在机制上可运行，缺乏严格的 TAD 核对比。 | 禁止重复搭建旧 bridge；若继续，只做冻结输入下的 kernel 对照。 |
| **Multi-Budget Detector Adaptation** | `CONTINUE_CANDIDATE` | **当前唯一未被已有证据否定且具有最高信息增益的单变量候选**。 | 第一轮禁止引入预算嵌入、蒸馏、动态控制器或跨数据集。 |

---

## 4. `ROOT_CAUSE_ANALYSIS`

根据严格的实证与代码审查，对历史失败与机制现象进行归因分类：

### 4.1 已证实根因 (Proven Root Causes)
1. **旧 E2E 性能暴跌 (`58.39%`)**：已证实由 5,940 步欠训练 (vs 13,080 步)、随机初始化粗 stem 与软代理梯度漂移共同引起，非联合优化理论不可行。
2. **CellCF 无法突破 Uniform**：已证实由其严格的 `one-per-uniform-cell` 结构约束引起，预算被锁死在各局部 cell 内，无法实现跨时序区域的全局预算转移。
3. **冻结检测器三档预算转移失败 (0/704)**：已证实由**跨预算检测器表示不匹配 (Cross-Budget Representation Mismatch)** 引起——下游检测头仅在 K384 下训练，面对 K256/K512 时的特征幅度与点分配分布发生偏移，导致局部加性收益无法转化为整视频 Soft-NMS 后的 AP 增益。
4. **PJST-D1 配对统计缺失**：已证实由 finalizer 脚本中预测文件路径配置错误导致 10,000 次 bootstrap 零抽样退出，属纯证据生成工程故障。
5. **稀疏粗扫中断**：已证实由非参数 BatchNorm buffer 检查与 `VARIANT_CONFIGS` 映射缺失导致运行时阻断，属配置与环境工程故障。

### 4.2 最强但未证实假说 (Strongest Unverified Hypotheses)
1. **多预算检测器适应假设**：检测器在训练期间以受控概率同时暴露于嵌套的 K256/K384/K512 输入，可消除跨预算特征分布偏移，在不改变定位精度的前提下为动态预算建立真实正向 headroom。
2. **Tubelet 局部运动失真假说**：非连续帧被强制打包成 VideoMAE tubelet 时，时间跨度跳跃扭曲了 3D 局部卷积的时空运动建模。
3. **稀疏重建边界平滑假说**：一维线性插值在较大采样空洞处抹平了高频边界突变，导致高 tIoU 定位衰减。

### 4.3 协议混杂与伪结论 (Protocol Confounding & Disproved Claims)
1. **H65 课程增益混杂**：历史 65.3857% 消耗了 30+60=90 轮总训练，相比 60 轮基线存在算力不匹配混杂，不能单独作为课程设计的论文因果证据。
2. **Native Tubelet 误读**：Native tubelet uniform (64.13%) 与 coreset (62.81%) 均基于原生 tubelet，二者差距仅否定了当前 coreset 打分规则，未检验“连续配对 vs 离散配对”的表示差异。

---

## 5. `GEMINI_RECOMMENDED_SINGLE_ROUTE`

### 推荐唯一主线：**基于 H65 嵌套位置的多预算检测器适应 (Multi-Budget Detector Adaptation)**

#### 推荐理由与科学增益
1. **直接击中核心科学瓶颈**：整视频 704 状态枚举已证实“冻结检测器无法承受预算变动”。在设计任何复杂的动态控制器之前，必须首先验证“检测器是否具备多预算适应能力”。如果检测器本身在等成本多预算混合下无法建立对固定控制的兼容性增益，任何外层控制器都将失去物理基础。
2. **严格单变量隔离**：完全冻结 H65 Scout、优先级排序、嵌套 K256/K384/K512 位置构造、VideoMAE 接口、ActionFormer 检测头、损失函数、物理时间映射与 Soft-NMS。唯一干预变量为 **Stage-2 训练期间的预算暴露分布**。
3. **继承经核验的可靠代码资产**：以 H65 clean revision `04c35a3b` 为模型基座，直接复用 `33e4ed13` 中已严格验证的真实变长 VideoMAE 执行、packet 对齐、actual-observation 计数与原始 proposal 顺序保持。

---

## 6. `MINIMAL_DECISIVE_EXPERIMENT`

在严格满足 Pro 完整数据协议的前提下，执行以下决定性闭环实验：

### 6.1 前置事实任务 (不可跳过)
- **THUMOS14 完整数据身份审计**：在 `04c35a3b` 基础上建立只读审计工具，核验 `thumos_14_anno.json` 中 `training` subset 的 200 个视频 ID、物理 MP4 存在性及 loader 输出一致性；核验 `validation` 的 211 视频与历史 ActionFormer 212 差异来源。在 Pro 获得通过并签发数据准入前，禁止启动任何模型训练。

### 6.2 匹配训练协议
- **共同起点**：两臂均加载 H65 Stage-1 `epoch_29.pth` 的 `state_dict_ema` (SHA-256: `bcbc877c...`)。
- **训练预算与时钟**：两臂在完整 200 个 training 视频上严格执行 **6,000 次成功 optimizer updates** (前 500 次 warmup，3,000 次前后半程分界)。
- **两臂定义**：
  1. **固定控制臂 (Fixed Control)**：每个训练窗口请求 K384 (实际 observation 为 `min(valid_len, 384)`)。
  2. **多预算适应臂 (Multi-Budget Adaptation)**：每个训练窗口从嵌套 priority 序列中请求 K256、K384 或 K512。概率由完整训练集上的 actual-observation 成本严格校准：
     $$p_{384} = 0.50, \quad p_{256} = 0.5 \times \frac{\mu_{512} - \mu_{384}}{\mu_{512} - \mu_{256}}, \quad p_{512} = 0.5 \times \frac{\mu_{384} - \mu_{256}}{\mu_{512} - \mu_{256}}$$
     确保候选臂的期望训练 observation 成本与全 K384 严格一致 (偏差 $\le 0.5\%$)。
- **相同配置**：AdamW、主 LR 1e-4、weight decay 0.05、seed 3407、AMP 开启、EMA 开启、batch size 每卡 2。

### 6.3 留出评估与统计判据
- **密封预测**：update 6,000 的 terminal `state_dict_ema` 在完整 `validation` 集合上生成 K256、K384、K512 及预注册的无标签 **fixed mixed-budget manifest** 预测并密封。
- **K384 安全门 (同时满足)**：
  - $\Delta \text{Avg-mAP} \ge -0.2\text{ pp}$
  - $\Delta \text{mAP@0.7} \ge -0.2\text{ pp}$
- **跨预算适应继续门 (在相同 fixed mixed manifest 下同时满足)**：
  - $\Delta \text{Avg-mAP} \ge +0.8\text{ pp}$
  - $\Delta \text{mAP@0.7} \ge +1.0\text{ pp}$
  - Mixed actual observation 成本 $\le$ 全 K384 成本
  - 10,000 次整视频配对 bootstrap 的 95% 置信区间下界 $> 0$。

---

## 7. `PUBLICATION_PATH_AND_STOP_RULES`

### 7.1 论文核心陈述与边界
- **一句话论文问题**：在保持离线 TAD 高精度定位的前提下，能否利用低成本动作状态证据构造非均匀观测，并通过多预算检测器适应在降低端到端重型计算时保护高 tIoU 边界？
- **核心机制**：低成本粗动作状态感知 + 预算校准系统采样 (BCSR) + 多预算检测器适应。
- **明确非主张**：不声称发明了通用端到端强化学习控制器；不声称无需针对性适应即可直接动态切帧；不声称在无实测时延下直接实现 50% 总体加速。

### 7.2 投稿证据基线与消融矩阵
1. **主表基线**：Dense AdaTAD (768 obs), Uniform K384, Random K384, Actionness-TopK K384, H65 Clean K384, DUCA Multi-Budget (K256/K384/K512/Mixed)。
2. **核心消融**：(a) 训练预算分布消融 (Fixed vs Multi-Budget)；(b) 采样机制消融 (Uniform vs BCSR vs Top-K)；(c) 物理时间残差消融 (TrueTime ON vs OFF)。
3. **泛化解锁条件**：
   - 只有在 THUMOS14 上**同时通过 K384 安全门与 Mixed 继续门且 CI 下界大于零**后，才允许向第二检测器 (如 TriDet) 或第二数据集 (如 ActivityNet-1.3 / Epic-Kitchens) 扩展。
   - 若第一轮未通过，禁止跨数据集调参。

### 7.3 绝对里程碑与停止规则 (基准日：2026-08-31)
1. **T+1 日 (2026-09-01)**：完成只读 211/212 数据身份核验与 CPU Evaluator 确认。若出现数据丢失或不可解释的排除，输出 `BLOCK` 并暂停。
2. **T+3 日 (2026-09-03)**：完成最小 Multi-Budget Builder 实现与 Evaluator 无标签 PRE_RUN。
3. **T+6 日 (2026-09-06)**：完成固定控制臂与多预算适应臂 6,000 次成功更新训练，密封预测与成本。
4. **T+7 日 (2026-09-07)**：一次性开放 held-out 评测并计算 10,000 次配对 bootstrap。
   - **成功 (GO)**：通过全部安全门与继续门，且区间大于零 $\rightarrow$ 锁定单种子机制证据，启动 3 种子评估与第二检测器验证。
   - **点估计过但区间含零**：判定为证据未决，当前路线终止，不得在同数据上反复搜索。
   - **失败 (STOP)**：任一门不达标 $\rightarrow$ **立即触发项目级 STOP**，关闭多预算适应路线，将该负结果作为深度归因分析写入技术报告，停止盲目扩展。

---

## 8. `QUESTIONS_PRO_MUST_DECIDE`

请 Pro 在接收本报告后对以下核心科学与流程问题作出最终裁决：

1. **科学优先级裁决**：是否批准将“多预算检测器适应 (Multi-Budget Adaptation)”作为数据准入后的**唯一第一候选实验**，而将 Tubelet 配对连续性、重构核消融及 TrueTime 扩展严格置于其后？
2. **基线归因口径裁决**：对于 H65 历史 30+60 (90 轮) 与正式 60 轮比较基准的差距，是否正式冻结以“同一 6,000 updates 的固定 K384 训练”作为论文的主对比控制，以消除训练更新数不匹配的混杂？
3. **论文核心主张定位**：论文的主力定位是主打“**等成本下的高 tIoU 定位提升**”还是“**等精度下的端到端计算削减**”？
4. **数据准入核验确认**：确认在 `audit_duca_thumos14_split_identity.py` 完成并通过独立 Critic/Evaluator 审查前，保持模型实现与 GPU 训练的完全冻结。

GEMINI_DUCA_ADVISORY_READY
