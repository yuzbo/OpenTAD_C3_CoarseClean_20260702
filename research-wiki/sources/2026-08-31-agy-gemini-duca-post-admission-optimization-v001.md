# DUCA 数据准入后模型优化独立技术咨询报告

**咨询模型身份**：Gemini 3.7 Flash (High)  
**咨询角色定位**：独立资深研究员 / 外部技术顾问（只读、证据驱动咨询）。本报告为提交 Pro 科学裁决前的独立技术意见；Pro 保持项目唯一科学决策权，本报告中的分析与建议不构成任何代码修改、远端作业或实验执行的直接授权。

---

## A. 实际检查的文件与提交 (Files and commits actually inspected)

### 1. 权威公开 GitHub 身份与精确提交
* **公共仓库**：[`yuzbo/OpenTAD_C3_CoarseClean_20260702`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702)
* **当前 Wiki / 证据分支与精确提交**：[`codex/duca-wiki-complete-sync-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-wiki-complete-sync-20260831) @ Commit [`a6c246a66ac9a81e94e2b592da15b79192a74150`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/a6c246a66ac9a81e94e2b592da15b79192a74150)
* **已准入的唯一 H65 模型与训练基座**：Commit [`04c35a3b76897e6c1569eeede41ed3aecaf7f854`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/04c35a3b76897e6c1569eeede41ed3aecaf7f854)
* **全量数据身份审计提交**：Commit [`fdd2bcdddf3f23f3546244adf90c4427ed022837`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/fdd2bcdddf3f23f3546244adf90c4427ed022837)
* **整视频冻结检测器证伪工具提交**：Commit [`33e4ed137c33eef07f0452b44506a6993bdf7535`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535)
* **当前 Builder 分支（基于 `04c35a3...`，尚未实现）**：[`feature/duca-h65-system-multibudget-exposure-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-h65-system-multibudget-exposure-v1-20260831)

### 2. 对应本地 Worktree 路径与代码/文档核验
* [`C:\Users\skywalker\.codex\worktrees\duca-wiki-complete-sync-20260831\OpenTAD_C3_CoarseClean_20260702`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702)
* [`C:\Users\skywalker\.codex\worktrees\duca-h65-system-multibudget-exposure-v1-20260831\OpenTAD_C3_CoarseClean_20260702`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702)
* [`C:\Users\skywalker\.codex\worktrees\duca-whole-video-consistent-budget-falsifier-v1-20260831\OpenTAD_C3_CoarseClean_20260702`](file:///C:/Users/skywalker/.codex/worktrees/duca-whole-video-consistent-budget-falsifier-v1-20260831/OpenTAD_C3_CoarseClean_20260702)
* [`C:\Users\skywalker\.codex\worktrees\duca-full-data-identity-audit-v1-20260831\OpenTAD_C3_CoarseClean_20260702`](file:///C:/Users/skywalker/.codex/worktrees/duca-full-data-identity-audit-v1-20260831/OpenTAD_C3_CoarseClean_20260702)

### 3. 逐项完整阅读与交叉比对的核心文献及代码清单
* **研究问询包**：[`research-wiki/query_pack.md`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702/research-wiki/query_pack.md)
* **防重复记忆总账**：[`research-wiki/anti_repetition.md`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702/research-wiki/anti_repetition.md)
* **决策演化历史**：[`research-wiki/decision_history.md`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702/research-wiki/decision_history.md)
* **论文进展缩略报告**：[`PAPER_PROGRESS.md`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702/PAPER_PROGRESS.md)
* **全量数据身份审计报告**：[`research-wiki/experiments/duca-full-data-identity-audit-v1.md`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702/research-wiki/experiments/duca-full-data-identity-audit-v1.md)
* **Pro 最新数据准入与任务裁决**：[`research-wiki/sources/2026-08-31-pro-duca-full-data-identity-admission-v001.md`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702/research-wiki/sources/2026-08-31-pro-duca-full-data-identity-admission-v001.md)
* **Pro 路线综合集成裁决**：[`research-wiki/sources/2026-08-31-pro-duca-comprehensive-route-integration-v001.md`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702/research-wiki/sources/2026-08-31-pro-duca-comprehensive-route-integration-v001.md)
* **前序 Gemini 全历史审查报告**：[`research-wiki/sources/2026-08-31-agy-gemini-comprehensive-wiki-code-review-v001.md`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702/research-wiki/sources/2026-08-31-agy-gemini-comprehensive-wiki-code-review-v001.md)
* **H65 课程与模型实现核心源码**：
  - [`configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py)
  - [`configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py)
  - [`configs/adatad/thumos/duca_sampling_rate_both_asformer_full_adapt_fixed384_official60.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_sampling_rate_both_asformer_full_adapt_fixed384_official60.py)
  - [`opentad/models/duca/acquisition.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/duca/acquisition.py)
  - [`opentad/models/duca/dynamic_budget.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-whole-video-consistent-budget-falsifier-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/duca/dynamic_budget.py)
  - [`opentad/models/selectors/duca_online_frame_selector.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/selectors/duca_online_frame_selector.py)
  - [`opentad/models/detectors/actionformer.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/detectors/actionformer.py)
  - [`tools/bata/audit_duca_thumos14_split_identity.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-full-data-identity-audit-v1-20260831/OpenTAD_C3_CoarseClean_20260702/tools/bata/audit_duca_thumos14_split_identity.py)

---

## B. 经核验的当前模型结构 (Verified current model structure)

### 1. H65 数据流与模块重构
经对基座代码 [`04c35a3...`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/) 的逐行核验，已准入的 H65 端到端数据流如下：
1. **输入与时间窗**：视频输入解封装为时序候选网格（THUMOS14 单窗口名义长度 $T=768$ 帧，有效帧长 $V \le 768$）。
2. **轻量粗分类探测 (Scout Probe)**：
   - 采用轻量空间 stem + 官方 ASFormer 时序编码器，对下采样低分辨率帧提取 7 维 deploy-visible 状态特征：`p_action`, `uncertainty`, `entropy`, `delta_p_action`, `abs_delta_p_action`, `uncertainty_peak`, `transition_score` ([`acquisition.py#L48-L57`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/duca/acquisition.py#L48-L57))。
   - 训练期间接受二元动作性（BCE）、状态转移分布以及边界支撑区间的显式监督。
3. **预算校准采样 (Budget-Calibrated Sampling Rate, BCSR)**：
   - 结合动作性与转移变化证据计算时序采样优先级，在保证全局均匀覆盖下限的同时向高概率动作起止转换区域倾斜。
   - 在固定 $K=384$ 下，确定性输出严格单调递增且无重复的离散观测位置子集 $S_{384} \subset \{0, \dots, V-1\}$，实际观测数 $K_{\text{actual}} = \min(V, 384)$，由 [`SparseTemporalGrid`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/duca/acquisition.py#L268-L380) 进行严格结构断言。
4. **重型骨干执行 (VideoMAE-S Backbone)**：
   - 通过 [`gather_selected_observations`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/duca/acquisition.py#L3618-L3659) 提取 $K_{\text{actual}}$ 帧原始高分辨率 RGB，打包为 16 帧 tubelets/clips 输入冻结的 VideoMAE-S 提取时空特征 token。
5. **检测头与后处理 (ActionFormer TAD Head)**：
   - 提取的 token 投影并插值到统一的检测器网格（$L=384$），通过多尺度 Transformer FPN 进行类别与时序边界距离回归。
   - 训练期通过 straight-through 软代理向 selector 反传受控检测梯度与贡献蒸馏损失。
   - 推理期在 NMS 前将预测 proposal 边界通过物理坐标逆映射回原始视频秒域，经过官方 Soft-NMS 产出检测结果。

### 2. “系统级多预算暴露适应”变更与固定边界对照表

| 维度 | 基准状态 (Control: H65 K384 固定暴露) | 实验状态 (Candidate: H65 多预算暴露适应) | 源码核验状态与依据 |
|---|---|---|---|
| **训练期预算暴露** | 全部 6,000 updates 固定请求 $K=384$ | 每个逻辑更新以校准概率采样 $K \in \{256, 384, 512\}$ | **唯一变量**：Stage-2 训练暴露分布 |
| **位置集合关系** | 单一 $S_{384}$ 集合 | 严格嵌套：$S_{256} \subset S_{384} \subset S_{512}$ | 保持 H65 优先级排序与时间单调 |
| **Stage-1 预热起点** | Epoch 29 `state_dict_ema` (SHA-256: `bcbc877c...`) | 同一 Epoch 29 `state_dict_ema` (完全相同) | **严格固定** |
| **Stage-2 优化时钟** | 6,000 次成功 updates (500 步预热，3,000 步分界) | 6,000 次成功 updates (完全相同) | **严格固定** |
| **可训练参数掩码** | Coarse trunk, action head, transition scorer, adapter, ActionFormer | 完全相同参数集合与各组学习率 | **严格固定**（系统级，非 detector-only） |
| **下游检测头与损失** | 官方 ActionFormer Head, IoU Loss, Cls Loss | 完全不变 | **严格固定** |
| **NMS 与官方评估器** | 官方 Soft-NMS, 211-video OpenTAD `mAP.py` | 完全不变 | **严格固定** |
| **重型执行开销** | 平均真实观测数 $\mu_{384}$ | 期望真实观测数严格匹配 $\mu_{384} \pm 0.1\%$ | **严格固定**（通过 PRE_RUN 概率校准） |
| **是否引入新模块** | 无 | **禁止**引入控制器/Gumbel/Mamba/DFT/蒸馏/嵌入 | **严格禁止多变量污染** |

---

## C. 冻结实验设计的严格审计 (Audit of the frozen experiment design)

### 1. 单变量跨预算适应隔离性核验
Control 与 Candidate 构造实现了严格的因果隔离：两臂共享相同的 Stage-1 检查点起点、完全相同的优化器与超参数配置、相同的数据顺序与随机数种子序列，且在训练期的实际 VideoMAE 重型观测期望消耗完全对齐。因此，性能差异能够干净地归因于“**多预算暴露是否改善了系统跨时序尺度的特征鲁棒性与校准稳定性**”。

### 2. 潜在混杂因素与隐式变量审计

```
[潜在混杂风险排查]
├── 1. 批次异质性 (Ragged Batches)       ──> 必须执行 Budget-Homogeneous 逻辑更新 (单 Update 单 K)
├── 2. 变长填充 (Padding Leaks)          ──> 必须执行 16-frame packet 对齐与零填充断言
├── 3. 短窗口折叠 (Short-Window)         ──> 必须按实际 valid_len 计算真实成本，识别折叠基线
├── 4. 时序坐标漂移 (Temporal Remap)     ──> 必须统一通过 interpolate_acquisition_time_to_detector_grid 映射
├── 5. 梯度归属污染 (Gradient Leak)      ──> 必须维持 Stage-2 授权参数组，禁止越界反传
├── 6. 优化与 EMA 时钟漂移 (Clock Drift) ──> AMP 重放不推进步数，严格锁定 terminal update-6000 EMA
└── 7. 数据流随机性偏移 (RNG Desync)     ──> 预算选择使用独立 PRNG，禁止打乱数据增强随机数流
```

* **逻辑批次同质性 (Budget-Homogeneous Batching)**：若在同一 mini-batch 内混用不同 $K$，会导致动态 padding、BatchNorm/LayerNorm 统计偏差以及多 GPU 同步障碍。**审计要求**：每个逻辑更新步必须仅使用单一 $K$ 档（Budget-Homogeneous update）。
* **变长执行与真实成本统计**：$K_{256}$ 与 $K_{512}$ 在 VideoMAE 执行时，严禁 pad 到 512 之后再送入骨干。必须使用已验证的 16 帧 packet 截断，且尾部 padding 必须严格少于 16 帧且数值为 0。
* **短窗口预算折叠 (Short-Window Collapse)**：对于 $V \le 256$ 的短视频窗口，请求 $K_{256}, K_{384}, K_{512}$ 实际均只能获取 $V$ 帧，发生向基线的物理折叠。**审计要求**：预算概率计算必须基于全量 200 视频测得的实际观测均值 $\mu_{256}, \mu_{384}, \mu_{512}$，而非字面常数 256/384/512。
* **物理时序逆映射一致性**：不同 $K$ 选出的帧在送入 ActionFormer 前，必须通过统一的 `interpolate_acquisition_time_to_detector_grid` 映射至 384 长度的检测器网格，并在 proposal 输出时准确反演回真实秒数。
* **可训练掩码与时钟对齐**：必须锁定 Stage-2 的 5 个参数组（Coarse trunk 1e-5, Action head 2e-5, Transition scorer 5e-5, Adapter 2e-4, Base 1e-4），任何 AMP 重放均需恢复原始 batch 且不推进 EMA / 优化器步数。

---

## D. 面向当前 Builder 的最小实现建议 (Minimal Builder implementation recommendations)

### 1. 允许修改的最小代码表面
为保证仓库纯净与可追溯性，Builder 在 [`feature/duca-h65-system-multibudget-exposure-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-h65-system-multibudget-exposure-v1-20260831) 分支上仅允许触及以下文件：

```text
opentad/models/duca/dynamic_budget.py                  # 变长 packet 验证与实际成本统计
opentad/models/selectors/duca_online_frame_selector.py # 嵌套多预算前向接口 (forward_marginal_prefixes)
opentad/models/detectors/actionformer.py               # 变长 VideoMAE 执行与检测网格映射对齐
configs/adatad/thumos/duca_h65_stage2_control_k384_official60.py       # Control 训练配置
configs/adatad/thumos/duca_h65_stage2_candidate_multibudget_official60.py # Candidate 训练配置
tests/test_c3_h65_multibudget_exposure_invariants.py   # 聚焦不变性与单步门禁测试
scripts/run_h65_multibudget_stage2_suite.sh            # 极简 Slurm 启动脚本
```

### 2. 必须在代码中严格固化的六大不变量 (Invariants)
1. **严格集合嵌套 (Strict Nestedness)**：
   $$\forall i \in [1, B], \quad S_{256}^{(i)} \subset S_{384}^{(i)} \subset S_{512}^{(i)}$$
   且各集合内部必须按原始时间顺序严格递增排列。
2. **K384 恒等复现性 (K384 Parity)**：
   在 Candidate 模型上强制指定 $K=384$ 时，前向输出特征与损失必须与基座 H65 在数值容差（$\text{rtol} \le 10^{-5}, \text{atol} \le 10^{-6}$）内严格一致。
3. **真实变长执行 (No Artificial Padding)**：
   输入 VideoMAE 的 tensor 长度必须恰好为 $\lceil K_{\text{actual}} / 16 \rceil \times 16$，且非基线执行的 padding 槽位必须严格全零。
4. **经验成本单调性 (Cost Monotonicity)**：
   在 200 个训练视频上实测必须满足 $\mu_{256} < \mu_{384} < \mu_{512}$。
5. **精确成本校准概率 (Exact Expected Cost Match)**：
   $$p_{384} = 0.5, \quad p_{256} = 0.5 \times \frac{\mu_{512} - \mu_{384}}{\mu_{512} - \mu_{256}}, \quad p_{512} = 0.5 \times \frac{\mu_{384} - \mu_{256}}{\mu_{512} - \mu_{256}}$$
   要求 $p_{256}, p_{512} \in (0, 0.5)$ 且期望训练观测成本与 Control 偏差 $\le 0.1\%$。
6. **无 held-out 标签/预测缓存泄漏**：
   严禁在训练或 PRE_RUN 阶段访问 211 验证集的动作标签或边界，严禁使用 `.npy`/`.pkl` 预测缓存绕过在线前向。

### 3. 运行前必须阻断的科学失效情形 (Pre-Run Failures)
* 若 $\mu_{256} \ge \mu_{384}$ 或 $\mu_{384} \ge \mu_{512}$（短窗口极端塌缩），PRE_RUN 立即阻断并退出；
* 若计算出的 $p_{256}$ 或 $p_{512} \notin (0, 0.5)$，立即阻断；
* 若单步梯度检查发现未授权参数（如 VideoMAE backbone）产生非零梯度，立即阻断；
* 若 200 视频训练 loader 重放数量不等于 200，立即阻断。

---

## E. 成功与失败机制分级分析 (Ranked success/failure mechanisms)

```
                       [科学现象与机制分类体系]
                                   │
      ┌────────────────────────────┼───────────────────────────┐
      ▼                            ▼                           ▼
【已证实事实与边界】          【最强待检验假说】          【推测性替代解释】
1. 冻结检测器转移 0/704 失败   1. 跨预算表示/校准不匹配    1. 纯时序 Drop 规则化效应
2. Dense 上界 68.73           2. 系统级多模块协同适应     2. 粗细粒度边界平滑效应
3. H65 稀疏基准 65.13
4. 数据身份 200/211 闭环
```

### 1. 已证实事实与边界 (Established Evidence)
* **冻结检测器三档动作空间穷尽证伪**：在冻结的 K384 H65 检测器上，整视频 704 个合法预算转移状态在开发集 privileged oracle 评测中无一达到 $+0.8/+1.0\text{ pp}$ 门槛（最佳联合仅 $+0.15/+0.49\text{ pp}$）。这证实了**冻结 K384 检测器无法直接承受动态预算切换**。
* **性能与成本锚点清晰**：Dense AdaTAD 提供 768 观测下的 68.73 Avg-mAP 上界；H65 提供 384 观测下的 65.13 Avg-mAP / 43.31 mAP@0.7 基线；数据身份 200 训练 / 211 留出已严格核验并准入。
* **多条历史单点路线已终结**：Coverage-v1（设施选址）、CellCF（局部网格）、Protected-E2E 同伦、Fast-only SlowFast 及原生 tubelet coreset 均已被证明无效，严禁重复。

### 2. 最强待检验假说 (Strongest Untested Hypotheses)
* **跨预算表示与校准不匹配假说 (Cross-Budget Representation & Calibration Mismatch)**：下游 ActionFormer 及时序适配器仅在固定 $K=384$ 密度下优化，导致其对 $K=256$（稀疏丢失局部上下文）和 $K=512$（密集特征幅值与点分布偏移）产生系统性误校准。在 Stage-2 训练期引入多预算暴露，可使检测器学会尺度不变的时序表征，为后续动态计算打开物理裕量。
* **系统级协同适应假说 (System-Level Co-adaptation)**：Scout 粗分类特征与下游检测反馈在多预算暴露下共同演进，相比单一预算能更好地识别哪些时序位置在不同总预算下均具备高辨识度。

### 3. 推测性替代解释 (Speculative Alternatives)
* **时序数据增强假说 (Temporal Augmentation / Dropout Effect)**：多预算暴露带来的性能改善可能并非来自“跨预算兼容性”，而是随机预算抽样等价于对时序输入施加了随机长度扰动（类似 DropPath/Temporal Dropout），带来通用的正则化增益。
* **边界平滑与局部响应假说**：$K=512$ 的引入可能仅改善了长动作内部响应，而对高 tIoU 关键边界的定位精度无实质帮助。

---

## F. PASS 与 FAIL 分支后的模型演进路径 (Model-improvement path after PASS and after FAIL)

```
                            [一次性打开 Held-Out 评测]
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
            【全门通过 PASS】                           【任一门失败 FAIL】
  (K384 ≥ -0.2 pp, Mixed ≥ +0.8/+1.0 pp,        (未能达到门槛或区间包含 0)
         95% CI 下界 > 0)                                     │
                    │                                         ▼
                    ▼                          ┌─────────────────────────────┐
  ┌───────────────────────────────────┐        │ 1. 正式停止 H65 多预算适应路线│
  │ 1. 锁定跨预算表示兼容性机制成立     │        │ 2. 封存负结果与全链路归因   │
  │ 2. 最小下一步：设计部署可见的      │        │ 3. 科学开放方向：转向纯空间  │
  │    Scout 语义驱动动态控制器        │        │    Dense-Time Zoom 路线     │
  │ 3. 必需消融：对比同分布内容无关    │        └─────────────────────────────┘
  │    Fixed Mixed 控制 (排除伪动态)  │
  └───────────────────────────────────┘
```

### 1. 判定分支：实验 PASS
* **科学结论**：证实“多预算训练暴露能够建立检测系统的跨预算表示兼容性”，否定“稀疏 TAD 必须受限于单一固定预算”的悲观假设。
* **最小且唯一合理的下一步模型改进**：
  在保持冻结的 Stage-2 多预算检测器与嵌套选帧器完全不变的前提下，引入基于低成本 Scout 特征（如全视频动作复杂度、边界密度分布）的**部署可见动态预算控制器 (Deploy-Visible Scout-based Budget Controller)**。
* **必需的控制变量消融 (Mandatory Ablations)**：
  1. *语义控制器 vs 内容无关同分布 Fixed Mixed Manifest*：必须证明基于视频内容的动态预算分配显著优于随机分配同等 $K$ 数量分布的基准。
  2. *Scout 特征敏感度消融*：对比 7 维状态特征 vs 纯 actionness vs 纯时序方差对预算决策的贡献。
  3. *模块适应度剥离*：评估若仅适应 ActionFormer 检测头而冻结 Scout 时，多预算增益的保留比例。

### 2. 判定分支：实验 FAIL
* **科学结论**：证实即使在训练期引入多预算暴露，当前 H65 的 BCSR 嵌套选帧与 ActionFormer 结构也无法跨越固有的时序采样非连续性瓶颈，彻底否定基于 H65 嵌套选帧的时序动态预算路线。
* **处置与停止规则**：立即触发项目级 `STOP`，不再进行任何超参数微调、损失权重搜索或更换随机种子抢救，将该彻底负结果连同 704 状态证伪作为技术报告的严密归因章节进行封存。
* **科学上依然开放的备选方向**：
  转向保留全部密集时间维度的**空间动态裁剪 (Dense-Time Spatial Zoom / DART-Zoom)** 路线，彻底规避时序帧删除带来的时序几何畸变与边界信息丢失。

---

## G. 论文发表路径与主张边界 (Publication path and claim boundaries)

### 1. 论文核心科学主张 (Central Claim)
> **“提出一种基于低成本动作状态转移感知的非均匀帧采集与多预算适应 TAD 系统（DUCA）。在保持离线高精度时序动作检测定位的前提下，DUCA 利用二元状态变化证据构造嵌套稀疏观测，并通过训练期多预算暴露消除跨尺度表示偏移，在将重型视频骨干观测输入减半的同时，有效保护了高 tIoU 边界定位性能。”**

### 2. 必需的全量数据证据与不确定性分析
* **全量数据集评测**：完整 200 训练视频训练，完整 211 留出验证集评测（无 held-out 泄漏，单一终态一次性打开）。
* **三随机种子重复**：`3407, 3408, 3409` 全盲执行，汇报点估计、三种子均值与种子间标准差 $\sigma_{\text{seed}}$。
* **统计显著性验证**：基于 10,000 次整视频配对 Bootstrap，给出 Avg-mAP 与 mAP@0.7 增益的 95% 置信区间，要求区间下界严格大于 0。

### 3. 完整端到端开销口径与时延界限 (Cost Boundaries)

```
[完整系统推理延迟分解 (Wall-Clock Latency Profile)]
┌─────────────────────────┬───────────────────────────────┬─────────────────────────┐
│ 1. 密集解码与前处理     │ 2. Scout 粗分类与选帧         │ 3. 重型 VideoMAE 骨干   │ 4. ActionFormer 与 NMS  │
│ (Video Decode / Resize) │ (ASFormer Stem + BCSR Select) │ (Variable-Length Tube)  │ (FPN / Cls / Reg / NMS) │
└─────────────────────────┴───────────────────────────────┴─────────────────────────┘
◄─────────────────────────────────── 完整端到端延迟 (End-to-End Time) ──────────────────────────────────►
```

* **区分重型观测与真实时延**：严格区分“重型骨干观测帧数削减 50%（384 vs 768）”与“端到端墙上时钟加速”。论文必须明确披露前置视频解码、Scout 粗探针推理、帧提取与后处理 NMS 的分项耗时。
* **同硬件基准测量**：在相同单卡环境、相同 batch 设定下，测量并汇报端到端 P50/P95 延迟、吞吐量 (FPS) 以及 GPU 峰值显存占用。

### 4. 严禁超范围宣称的非主张 (Non-Claims)
* 严禁宣称“DUCA 已超越密集官方 AdaTAD”（Dense 68.73 仍为上界）；
* 严禁宣称“观测减半直接等价于端到端加速 50%”；
* 严禁宣称“该方法已在未经测试的其他检测器（如 TriDet）或数据集（如 ActivityNet）上实现通用”。

---

## H. 阻断性冲突或缺失证据 (Blocking conflicts or missing evidence)

经全面比对最新 Pro 准入裁决、历史 Wiki 与当前代码，发现并闭环以下关键事项：

1. **种子执行顺序的历史冲突已由 Pro 权威裁决闭环**：
   - *历史分歧*：前序综合裁决曾考虑“先跑 3407，达标后再跑 3408/3409”的条件执行；
   - *最新裁决*：Pro 在 [`2026-08-31-pro-duca-full-data-identity-admission-v001.md`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702/research-wiki/sources/2026-08-31-pro-duca-full-data-identity-admission-v001.md) 中作出最终权威判定：**推翻条件执行，冻结为 `3407 → 3408 → 3409` 全三种子盲执行，全部训练与预测封存后，一次性打开 211 held-out 评测**。
   - *独立顾问意见*：完全支持 Pro 的最新盲执行判定。条件执行会导致后续实验依赖初步 held-out 观察，破坏留出集的绝对隔离性。
2. **211 与 212 数据身份差异已被唯一解释**：
   - 审计确认 OpenTAD 官方 211 验证集排除了标注错误的 `video_test_0000270`，与 ActionFormer 字面 212 的差异完全透明可闭环，不存在数据层面的阻断。
3. **当前无未解决的代码或科学冲突**：
   - 基座 `04c35a3...` 处于干净状态，可支持 Builder 开展最小化多预算实现。

---

## I. 向 Pro 的简明独立建议 (Concise recommendation to Pro)

1. **维持唯一任务授权**：
   保持对 **“H65 系统级多预算暴露适应（全量 200/211、三种子、一次性 held-out 裁决）”** 的唯一任务锁定，坚决不向当前阶段引入任何动态控制器、预算嵌入、蒸馏损失或架构重构。
2. **督导 Builder 严格落实 Budget-Homogeneous 逻辑批次与 Packet 对齐**：
   确保 Builder 在代码实现中杜绝不同 $K$ 的混合 batch 填充，并严格执行 16 帧 packet 截断与尾部全零断言。
3. **严格执行单次终态开箱与硬停止门槛**：
   在 6 个训练单元全部完成并封存预测哈希前，严禁任何形式的提前验证或指标刺探；若三 seed 联合 bootstrap 95% CI 下界含零或未达 $+0.8/+1.0\text{ pp}$ 门槛，果断终止该路线，保持科研严谨性。

---

GEMINI_DUCA_POST_ADMISSION_OPTIMIZATION_READY
