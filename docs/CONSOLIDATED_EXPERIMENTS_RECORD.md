# OpenTAD / DUCA-TAD / C3 完整实验全景汇拢与实验记录文档
# (Consolidated Master Experiment Registry & Evidence Record)

> **更新时间**：2026-09-02  
> **仓库**：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`  
> **分支**：`codex/gas-vt-stage23-detector-aware-20260706`  
> **文档定位**：全量汇拢当前纯净代码库中**所有已实现、已验证、已评测、已审计**的实验体系、配置矩阵、工具链、测试集、实验结果与核心因果结论。

---

## 目录 (Table of Contents)

1. [项目总览与研究演进全景 (Project Overview & Research Trajectory)](#1-项目总览与研究演进全景)
2. [核心实验体系分类详述 (Comprehensive Experiment Families)](#2-核心实验体系分类详述)
   - [Family 1: C3 粗分类探针矩阵与模型动物园 (C3 Coarse Probe Zoo)](#family-1-c3-粗分类探针矩阵与模型动物园)
   - [Family 2: 离线动作性策略对比 (PAction vs. GAS-VT vs. 晶格替换)](#family-2-离线动作性策略对比-paction-vs-gas-vt-vs-晶格替换)
   - [Family 3: Stage 2 稠密 Teacher 与检测器感知效用 (Detector-Aware Utility)](#family-3-stage-2-稠密-teacher-与检测器感知效用)
   - [Family 4: Stage 3 DUCA-JCT 渐进协同联合训练与动态预算 MUST](#family-4-stage-3-duca-jct-渐进协同联合训练与动态预算-must)
   - [Family 5: 官方 AdaTAD 结构与坐标语义审计 (Structural & Coordinate Audits)](#family-5-官方-adatad-结构与坐标语义审计)
   - [Family 6: 纯转变头与选择质量诊断 (DUCA Transition-Only & Selection Diagnostic)](#family-6-纯转变头与选择质量诊断)
   - [Family 7: 13,200 次成功更新标准协议与 CellCF 评估 (Formal 13,200-Update Protocol & CellCF)](#family-7-13200-次成功更新标准协议与-cellcf-评估)
   - [Family 8: 全局预算分配族天花板与物理几何诊断 (Allocation Family Ceilings)](#family-8-全局预算分配族天花板与物理几何诊断)
   - [Family 9: 受保护物理端到端与 R0-R5 生产交付矩阵 (Protected Physical E2E & R0-R5)](#family-9-受保护物理端到端与-r0-r5-生产交付矩阵)
   - [Family 10: H65 课程压缩与两阶段联合训练 (H65 Curriculum & Two-Stage Recovery)](#family-10-h65-课程压缩与两阶段联合训练)
   - [Family 11: 物理时间原生检测器规格 (PhysTime-TAL Specification)](#family-11-物理时间原生检测器规格)
   - [Family 12: PC-OT-MRAS 与时序网格/残差探索 (PC-OT-MRAS & Temporal Grid Residuals)](#family-12-pc-ot-mras-与时序网格残差探索)
   - [Family 13: CT-DP-BAMoD 4 臂全因子消融矩阵 (Dual-Phase + BAMoD + CT-Conv1d)](#family-13-ct-dp-bamod-4-臂全因子消融矩阵)
3. [远端部署实验全量映射与运行状态清单 (Remote Deployment Inventory & Live Status)](#3-远端部署实验全量映射与运行状态清单)
4. [关键实验指标与因果对比总表 (Consolidated Benchmark & Empirical Evidence)](#4-关键实验指标与因果对比总表)
5. [配置、工具、脚本与测试映射总表 (Master Mapping Table)](#5-配置工具脚本与测试映射总表)
6. [核心科学结论与后续指引 (Key Scientific Insights & Next Steps)](#6-核心科学结论与后续指引)

---

## 1. 项目总览与研究演进全景

### 1.1 核心研究问题与终极目标
- **核心问题**：在视频动作检测（TAD）中，能否在昂贵的特征提取/检测主干（Backbone/VideoMAE）之前，通过低成本 scout/selector 策略在输入端选择大幅减少的观测帧（如固定预算 $\le 384$ 帧，即 $50\%$ 稀疏度），在大幅节约端到端算力的同时，保持甚至提升高 IoU（$t\text{IoU}\ge 0.6, 0.7$）下的定位性能？
- **终极形态**：构建一个可即插即用的前置时序采集适配器（Pre-backbone Temporal Acquisition Adapter）：
  $$\text{TemporalAcquisitionAdapter} = \text{Low-cost Scout} + \text{Policy/Scorer} + \text{Hard/ST Sampler} + \text{Coordinate Adapter} + \text{Detector Bridge}$$

### 1.2 严格的方法与证据门禁 (Claim Gates)
1. **真实算力边界**：稀疏采样的上限严格限制为 $K \le 384$ 帧（输入 $T=768$）。
2. **严防无泄漏 (No Leakage)**：测试/验证集推断时严禁使用 Ground Truth (GT)、Teacher 预测、Oracle 边界或检测器缓存；所有测试评估必须完全由输入端可见的粗特征生成。
3. **真实端到端门禁 (End-to-End Proof)**：必须证明检测器损失对选择器参数产生真实的非零梯度回传（$\nabla_{\theta_{\text{selector}}} \mathcal{L}_{\text{det}} \neq 0$），并通过一次反向传播使参数或选择分布发生位移。
4. **终点固定评测 (Terminal Evaluation Only)**：正式评测仅以固定轮次终点 EMA 检查点（如 `epoch_131` 或 `epoch_59`）为准，严禁用验证集在训练中间挑最优。
5. **同协议匹配对照 (Strict Matched Comparisons)**：所有比较必须同 Commit、同 Backbone 预训练权重、同优化器步数、同评估脚本。

---

## 2. 核心实验体系分类详述

### Family 1: C3 粗分类探针矩阵与模型动物园
* **研究目的**：评估不同低成本架构对动作性概率 $p_{\text{action}}$ 以及边界变化率 $|\Delta p_{\text{action}}|$ 的预测能力，为下游稀疏选帧提供候选 scout。
* **涵盖模型架构**（8 类 Temporal-TCN + 1 类官方 ASFormer）：
  1. `temporal_tcn_separable_dilated` (深度可分离空洞卷积)
  2. `temporal_tcn_motion` (时序运动特征融合)
  3. `temporal_tcn_lite` (轻量化时序卷积)
  4. `temporal_tcn_multiscale` (多尺度空洞感受野)
  5. `temporal_tcn_dilated` (标准空洞卷积)
  6. `temporal_tcn_residual` (残差时序卷积)
  7. `temporal_tcn_gated` (门控线性单元时序卷积)
  8. `temporal_tcn_causal_dilated` (因果空洞卷积)
  9. `official_asformer` (官方 Action Segmentation Former 探针)
* **核心评估指标**：AP, ROC-AUC, F1@0.5, Action-BG Gap, Change-Lift, Top10/Top20 @ r1/r2/r4/r8 边界覆盖率, Selection Score。
* **实测核心结论**：
  - 综合候选分最高为 `temporal_tcn_separable_dilated`（Candidate Score=0.5928, AP=0.4814, Top20@r4=0.7254）。
  - 选帧代理分最高为 `temporal_tcn_lite`（Selection Score=0.5162, $r_1$ 边界支持=0.2133, 动作覆盖率=0.5217）。
* **代表性代码资产**：
  - 训练与矩阵生成：[tools/bata/train_lowres_action_probe.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/tools/bata/train_lowres_action_probe.py), [tools/bata/c3_coarse_classifier_model_matrix.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/tools/bata/c3_coarse_classifier_model_matrix.py)
  - Ledger 导出转换：[tools/bata/convert_lowres_probe_samples_to_value_transport_ledger.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/tools/bata/convert_lowres_probe_samples_to_value_transport_ledger.py)
  - 自动化报告与绘图：[analysis_outputs/c3_completed_paction_candidate_benchmark_20260704/](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/analysis_outputs/c3_completed_paction_candidate_benchmark_20260704/)

---

### Family 2: 离线动作性策略对比 (PAction vs. GAS-VT vs. 晶格替换)
* **研究目的**：检验离线生成的稀疏 Ledger 送入 AdaTAD 检测器后的实际 Avg-mAP，解释为何复杂的时序价值传输（GAS-VT）会陷入平台期，而简单的动作性排序加空洞控制（PAction Learned）能取得更高性能。
* **实验变体**：
  - `paction_learned_fixed_384`：基于 $p_{\text{action}}$ 预测与空洞惩罚生成的固定 384 帧 Ledger。
  - `gas_vt_fixed_384`：融合边界 Bracket、动作内部与时序空洞控制的带状价值传输策略。
  - `paction_lattice_replacement_fixed_384`：在均匀骨架（Uniform Scaffold）上进行基于打分的局部晶格微调替换诊断。
* **实测关键数据与归因**：
  - **GAS-VT fixed_384**：在第 6 轮快速达到 40.96 Avg-mAP，随后在 44~46 之间停滞（最终 ~44.90 Avg-mAP）。原因在于其边界支持率低（$r_1=0.1426$），且 p95 空洞偏大（96.0）。
  - **PAction Learned fixed_384**：达到 59.10~61.02 Avg-mAP（tIoU@0.7 达到 40.05）。其动作覆盖率虽然略低（0.5272 vs 0.5445），但边界支持率显著提高（$r_1=0.2361$），p95 空洞极小（2.0）。
  - **Jaccard 重叠分析**：PAction 与 Top-k 动作性的 Jaccard 相似度（0.4083）甚至略低于 GAS-VT（0.4372），证明其收益来源于良好的时序空洞控制与边界敏感性，而非单纯复制动作性打分。
* **代表性代码资产**：
  - 配置：[configs/adatad/thumos/c3_paction_learned_ledger_adatad_full_train.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/c3_paction_learned_ledger_adatad_full_train.py), [configs/adatad/thumos/c3_gas_vt_ledger_adatad_full_train.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/c3_gas_vt_ledger_adatad_full_train.py)
  - 启动脚本：[scripts/run_c3_paction_learned_policy_adatad_full_train_gpu1.sh](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/scripts/run_c3_paction_learned_policy_adatad_full_train_gpu1.sh), [scripts/run_c3_gas_vt_policy_adatad_full_train_gpu1.sh](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/scripts/run_c3_gas_vt_policy_adatad_full_train_gpu1.sh)
  - 诊断文档：[docs/methods/2026-07-07-gasvt-paction-diagnosis-evidence.md](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/docs/methods/2026-07-07-gasvt-paction-diagnosis-evidence.md)

---

### Family 3: Stage 2 稠密 Teacher 与检测器感知效用
* **研究目的**：摆脱单纯基于 GT 动作标签的二分类动作性监督，利用训练好的稠密 AdaTAD Teacher 提取检测器内部效用（Point Responsibility / Signed Utility / Cls-Reg Loss Contribution），指导离线采集策略。
* **核心组件与设计**：
  - 训练标准的无选择器 Dense AdaTAD Teacher（$T=768$）。
  - 从 Teacher 中导出训练集专用效用（Proposal Score、Point Responsibility、分类/回归贡献度），并严格防范测试集泄露。
  - 训练以符号化检测器效用（`signed_frame_utility`）为目标的探测器。
* **代表性代码资产**：
  - Teacher 训练配置：[configs/adatad/thumos/c3_dense_adatad_teacher_full_train.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/c3_dense_adatad_teacher_full_train.py)
  - 效用导出与门禁工具：[tools/bata/detector_teacher_utility.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/tools/bata/detector_teacher_utility.py), [tools/bata/validate_c3_detector_aware_adatad_full_train.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/tools/bata/validate_c3_detector_aware_adatad_full_train.py)

---

### Family 4: Stage 3 DUCA-JCT 渐进协同联合训练与动态预算 MUST
* **研究目的**：构建首个单次运行（Single-run）的端到端在线联合训练体系，打通检测器到选择器的 Straight-Through 梯度桥，并通过对偶变量实现动态预算约束。
* **核心训练机制**：
  - **网络拓扑**：`Online Coarse Actionness Probe -> DUCA Frame Selector -> Selected-axis ActionFormer / AdaTAD Head`。
  - **渐进余弦联合调度 (Progressive Joint Schedule)**：
    - 调度步数与优化器更新步绑定（500 步 warmup，4000 步 transition）。
    - 检测器损失始终开启（`detector_loss_always_on=True`）。
    - 检测器到选择器的梯度回传系数从 0.0 渐变至 1.0。
    - 粗动作性辅助监督权重从 1.0 渐退至 0.25。
    - 空洞惩罚权重从 0.0 渐进至 0.05。
  - **DUCA-MUST 动态预算控制器**：采用拉格朗日乘子对偶更新，根据期望选帧成本均值自适应调节动态预算 $K(x) \in [192, 384]$。
  - **冻结对照**：引入免训练的 X3D 时序网格、MobileNet、SlowFast 外部动作性先验。
* **代表性代码资产**：
  - 主配置：[configs/adatad/thumos/duca_online_official_adatad_backend_full_train.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_online_official_adatad_backend_full_train.py), [configs/adatad/thumos/duca_must_dynamic_official_adatad_backend_full_train.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_must_dynamic_official_adatad_backend_full_train.py)
  - 梯度证明与套件启动器：[tools/bata/run_duca_jct_one_step_grad_proof.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/tools/bata/run_duca_jct_one_step_grad_proof.py), [scripts/submit_duca_jct_experiment_suite.sh](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/scripts/submit_duca_jct_experiment_suite.sh)
  - 详细文档：[docs/methods/2026-07-09-duca-jct-progressive-deployment.md](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/docs/methods/2026-07-09-duca-jct-progressive-deployment.md)

---

### Family 5: 官方 AdaTAD 结构与坐标语义审计
* **研究目的**：针对官方 OpenTAD 提交 `1aa8ca4`，精确审计 DUCA 对检测器核心文件的修改范围、输入坐标几何与损失语义。
* **代码与 Git Blob 字节级对比**：
  | 模块 | 官方 Blob | DUCA 分支 Blob | Diff 规模 | 审计裁决 |
  | --- | --- | --- | --- | --- |
  | AdaTAD base config | `e0dd2a0` | `e0dd2a0` | 0 | 字节完全一致 |
  | `actionformer_head.py` | `42e78b2` | `42e78b2` | 0 | 字节完全一致 |
  | `anchor_free_head.py` | `fe9c12a` | `8c87796` | +214 / -17 | 已扩展坐标与掩码逻辑 |
  | `actionformer.py` | `f07e3e8` | `24fed8a` | +595 / -13 | 已集成 selector 与前向调度 |
  | `single_stage.py` | `3ba7294` | `f97cab5` | +185 / -10 | 已扩展优化器与梯度桥 |
  | `vit_adapter.py` | `6c505dc` | `c72bece` | +415 / -6 | 已扩展 VideoMAE 长度适配 |
* **核心审计裁决**：
  - 正确口径：“使用官方 OpenTAD/AdaTAD 派生的 VideoMAE-S、projection、neck 与 ActionFormerHead 配置，在其前加入 DUCA selector，并由扩展后的检测器完成 selected-axis GT 映射、检测和 true-time 后映射。”
  - 严禁声称：“完全未修改的官方 AdaTAD”；分类与回归损失数学公式未改，但损失分配所在的坐标轴已由物理时间变为选中轴（Selected Axis）。
* **代表性报告**：[docs/methods/2026-07-10-duca-official-adatad-structural-audit.md](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/docs/methods/2026-07-10-duca-official-adatad-structural-audit.md)

---

### Family 6: 纯转变头与选择质量诊断 (DUCA Transition-Only & Selection Diagnostic)
* **研究目的**：去除复杂的专用 Head（Start/End/Context/Radius），仅利用官方 ASFormer 编码器隐状态的时序差分 $|\Delta p_{\text{action}}|$ 与熵变构造极简的单分支转变打分器，并深入解剖选帧几何质量。
* **实验分支与诊断结果**（Job `1159416`, Checkpoint epoch 89 EMA, 最佳评估 Avg-mAP 64.34）：
  1. **粗分类动作状态判别力弱**：Pooled AUROC 仅 0.6214，AUPRC 0.4111（先验动作流行度 32.50%），阈值 0.5 下预测正例占比高达 53.62%，导致假阳性偏高。
  2. **间接转变打分落后于原始差分**：在各个 GT 边界邻域半径（r0, r1, r2, r4, r8）下，学习到的转变打分器 AUROC（0.57~0.58）均略低于原始动作性差分 $|\Delta p_{\text{action}}|$（0.60~0.61）。
  3. **选帧几何 vs. 精确均匀采样**：
     - 精确命中（$r_0$）：学习策略 0.1568 vs. 精确均匀 0.1415（+0.015 提升）。
     - 邻域覆盖（$r_1$）：学习策略 0.8437 vs. 精确均匀 0.9991（-0.155 显著损失）。
     - 双端点命中（$r_1$）：学习策略 0.7108 vs. 精确均匀 0.9982。
     - 平均端点距离：学习策略 0.6755 帧 vs. 精确均匀 0.4800 帧（退步 0.195 帧）。
  4. **GT Boundary Oracle 上限锚点**（Job `1001959`）：使用真实标注边界进行选择时，在历史协议下可取得 **76.67 Avg-mAP**（$t\text{IoU@0.7} = 65.83$），证明前置边界特征对 TAD 具有极高的理论上限（约 12 个百分点的潜在空间）。
* **代表性代码与文档**：
  - 契约与实现方案：[docs/superpowers/plans/2026-07-11-duca-transition-only-implementation.md](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/docs/superpowers/plans/2026-07-11-duca-transition-only-implementation.md), [docs/methods/duca_transition_only_contract.md](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/docs/methods/duca_transition_only_contract.md)
  - 详细诊断报告：[docs/methods/2026-07-13-duca-selection-quality-diagnostic.md](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/docs/methods/2026-07-13-duca-selection-quality-diagnostic.md)

---

### Family 7: 13,200 次成功更新标准协议与 CellCF 评估
* **研究目的**：消除因 AMP 混合精度溢出导致跳过步数不同而带来的训练量偏差，建立严格的四臂（Four-arm）可复现对照。
* **协议规格**：
  - 严格保证每轮 100 次有效更新，共 132 轮，每臂**严格执行 13,200 次成功优化器更新**。
  - 引入内存级 AMP 溢出重放（Replay）：在发生非有限梯度时恢复 RNG 与模型 Buffer，重放相同 Batch 并调整 GradScaler，直至成功更新。
  - 仅以终点 `epoch_131.pth` 的 `state_dict_ema` 进行一次性密封评测。
* **四臂实测结果对照**：
  1. `exact_uniform` (基线)：终点 EMA Avg-mAP = **63.8594**
  2. `transition_beta0` (无检测器梯度桥)：终点 EMA Avg-mAP = **64.2755**
  3. `transition_counterfactual` (CellCF 局部反事实置换)：终点 EMA Avg-mAP = **64.0610**
  4. `direct_boundary_a5` (直接边界预测)：作为历史对照
* **科学裁决**：CellCF 证实了局部相空间调控的机制可行性，但在全视频尺度上受限于 Voronoi 局部单元限制，被归为局部控制诊断实验，促使后续转向全局分配族。
* **代表性代码资产**：
  - 训练与协议规范：[tools/bata/duca_cellcf_protocol.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/tools/bata/duca_cellcf_protocol.py), [tools/bata/duca_cellcf_training.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/tools/bata/duca_cellcf_training.py)
  - 自动化验证工具：[tools/bata/validate_duca_cellcf_suite.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/tools/bata/validate_duca_cellcf_suite.py)
  - 协议设计文档：[docs/methods/2026-07-15-duca-successful-update-formal-rerun-design.md](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/docs/methods/2026-07-15-duca-successful-update-formal-rerun-design.md)

---

### Family 8: 全局预算分配族天花板与物理几何诊断
* **研究目的**：构建只读天花板诊断工具集，使用严格的凸优化与混合整数规划（HiGHS MILP / DP）求解不同全局约束分配族在物理时间轴上的边界捕获上限。
* **五大注册分配族 (Registered Families)**：
  - **Family A (Exact Uniform)**：标准等间隔四舍五入精确均匀采样。
  - **Family B (One-per-uniform-cell / CellCF)**：每个均匀 Voronoi 单元内独立选择 1 帧，不可跨单元转移配额。
  - **Family C (Fixed Scaffold + Residual)**：由最小均匀支架保证最大物理空洞，剩余预算按打分全局分配。
  - **Family D (Global Exact-K Physical-Gap Family)**：在满足全局最大物理空洞 $\Delta$ 约束的有向无环图（DAG）上通过动态规划精确求解全局最优。
  - **Family E (Privileged Unrestricted GT Family)**：无空洞约束、直接以多目标 GT 边界覆盖为目标的特权上界求解。
* **物理坐标三元组契约**：每个样本明确绑定 Dense Candidate 序号、解码真实帧号（`frame_inds`）以及真实秒数时间戳（`timestamps`）。
* **代表性代码资产**：
  - 求解器与分配族核心：[tools/bata/duca_allocation_families.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/tools/bata/duca_allocation_families.py), [tools/bata/duca_exact_physical_solver.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/tools/bata/duca_exact_physical_solver.py)
  - 评估与天花板诊断：[tools/bata/diagnose_duca_allocation_family_ceiling.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/tools/bata/diagnose_duca_allocation_family_ceiling.py), [tools/bata/evaluate_duca_allocation_candidates.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/tools/bata/evaluate_duca_allocation_candidates.py)
  - 设计文档：[docs/methods/2026-07-20-duca-allocation-family-ceiling-design.md](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/docs/methods/2026-07-20-duca-allocation-family-ceiling-design.md)

---

### Family 9: 受保护物理端到端与 R0-R5 生产交付矩阵
* **研究目的**：落实 R0-R5 严格交付标准，涵盖边界微簇 Oracle、前端预训练、60 轮正式训练矩阵及密度传输梯度桥。
* **阶段架构**：
  - **R0 (Boundary Burst Oracles)**：构建 R2Q3、R4Q5 等边界微簇策略与严格 Uniform 的离线判定。
  - **R1 (Artifact Contracts & Hashing)**：实现原子日志封存与无泄露消费链。
  - **R2 (Frontend Pretraining)**：粗分类/转变打分前端预训练（不同学习率配比与高斯平滑）。
  - **R3 (Official 60-Epoch Full Train)**：
    - `G0`: 无检测器反馈基线 (`no_feedback`)。
    - `G1`: 受保护梯度桥 (`protected_bridge`)。
    - `G2`: 均匀伴随正则 (`uni_companion`)。
  - **R4 (Hard-Swap Alignment & Density Transport)**：
    - 软适配（`soft_adapted`）与硬置换（`hard_detached`）对齐。
    - 密度传输 Straight-Through 梯度桥：Softmax-14, Hardmax-14, NoMax, Mixture。
  - **R5 (Paper Matrix & Cost Measurement)**：多随机种子（Seeds 0, 1, 2）、K384/K256 及端到端延迟/FLOPs 评测。
* **代表性代码资产**：
  - 密度传输配置：[configs/adatad/thumos/duca_density_transport_softmax14_fixed384_official60.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_density_transport_softmax14_fixed384_official60.py)
  - 启动与汇拢脚本：[scripts/launch_duca_r5_paper_matrix.sh](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/scripts/launch_duca_r5_paper_matrix.sh), [tools/bata/aggregate_duca_r5_paper_matrix.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/tools/bata/aggregate_duca_r5_paper_matrix.py)

---

### Family 10: H65 课程压缩与两阶段联合训练
* **研究目的**：针对历史两阶段 H65 方案（30 轮 Uniform 语义预热 + 60 轮 Learned-Selection 联合训练 = 90 轮总训练量，达到 **65.3857% Avg-mAP**），检验能否在严格公平的 60 轮总预算（6,000 次成功更新）内无损重叠课程。
* **课程压缩设计 (H65-60)**：
  - **Stage-1 (Epoch 0–19, 2,000 updates)**：Exact Uniform K=384，训练基础检测器与 H65 动作性/转变预测；关闭选择器梯度与蒸馏。
  - **Stage-2 (Epoch 20–39, 2,000 updates)**：从 Stage-1 Epoch-19 EMA 初始化并重建优化器，以余弦退火渐进开启 Policy Alpha、检测器反馈与 ASFormer 适配。
  - **Stage-3 / Joint (Epoch 40–59, 2,000 updates)**：在同一优化器下持续进行完整的 H65 联合训练。
* **8 臂证据恢复矩阵 (Evidence Recovery Suite)**：
  1. `duca_evidence_recovery_base.py` (基线)
  2. `duca_evidence_recovery_full.py` (全量组合)
  3. `duca_evidence_recovery_h65_selection.py` (H65 选帧策略)
  4. `duca_evidence_recovery_matched_h65_60.py` (匹配 60 轮)
  5. `duca_evidence_recovery_no_coverage.py` (消融覆盖损失)
  6. `duca_evidence_recovery_no_merge.py` (消融合并机制)
  7. `duca_evidence_recovery_no_recovery.py` (消融恢复分支)
  8. `duca_evidence_recovery_no_robust.py` (消融鲁棒损失)
  9. `duca_evidence_recovery_no_time.py` (消融时间编码)
* **代表性代码资产**：
  - 课程配置：[configs/adatad/thumos/duca_h65_60_stage1_uniform20.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_h65_60_stage1_uniform20.py), [configs/adatad/thumos/duca_h65_60_stage2_transition20_joint20.py](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_h65_60_stage2_transition20_joint20.py)
  - 设计文档：[docs/superpowers/specs/2026-08-23-duca-h65-60-curriculum-design.md](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/docs/superpowers/specs/2026-08-23-duca-h65-60-curriculum-design.md)

---

### Family 11: 物理时间原生检测器规格 (PhysTime-TAL Specification)
* **研究目的**：从根本上解决 Selected-Axis 将不等物理间隔视为等间隔所带来的感受野失真与回归误差，设计直接在连续物理时间轴上运算的检测器。
* **核心创新架构**：
  - **连续相对时间注意力 (Continuous Relative Time Attention)**：将时间跨度 $\Delta t_i$ 与相对时间偏移结合，引入密度覆盖归一化权重 $\log w_j$ 消除采样密度不均。
  - **多尺度物理 Query 金字塔**：在归一化真实时间 $[0, 1]$ 上建立规则物理 Query，通过 Cross-Attention 从任意不规则观测中聚合特征。
  - **重采样等变性监督 ($L_{\text{resample}}$)**：对同一视频生成两种不同稀疏/随机/局部密集的不规则采样视图，要求物理 Query 预测具有重采样不变性。
* **代表性规范文档**：[docs/superpowers/specs/2026-07-10-phystime-tal-design.md](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/docs/superpowers/specs/2026-07-10-phystime-tal-design.md)

---

---

### Family 13: CT-DP-BAMoD 4 臂全因子消融矩阵 (Dual-Phase + BAMoD + CT-Conv1d)
* **研究目的**：构建前置双相选帧、连续时序骨干（CT-Tubelet）、边界偏置混合专家（B-AMoD）与连续物理坐标检测头（CT-Conv1d）的完整闭环，并通过严格的 2×2 全因子正交消融隔离各个算子的独立收益与协同增益。
* **涵盖模块与机制**：
  1. **双相正交预算选帧器（`DualPhaseFrameSelector`）**：
     - 全局骨架相（Scaffold Phase, $K_{\text{scaffold}}=128$）：步长为 6 帧的均匀锚点，杜绝大空洞；
     - 相变微簇相（Burst Phase, $K_{\text{burst}}=256$）：基于像素差分能量 $E(t)$ 在动作起止边界处采样密集微簇（$R=2$）。
  2. **CT-Tubelet 物理速度归一化 3D 卷积**：
     - 将 3D 卷积核在时序维度正交分解为稳态分量 $W_{\text{mean}}$ 与动态差分分量 $W_{\text{diff}}$；
     - 动态分量显式除以帧对真实时间差 $\frac{1}{\Delta t^{\text{pair}}}$，消除非均匀跳帧引起的表观速度失真。
  3. **B-AMoD 边界偏置混合专家路由**：
     - 复用自注意力列均值并调制边界先验，严格实行 50% Token 绕过多头自注意力和 MLP 的恒等直通（Zero Mutation Bypass）。
  4. **CT-Conv1d 连续物理坐标检测头**：
     - `ContinuousTimeScaleAdaptiveConv1d` 依据目标物理时间逆插值动态调制卷积采样偏移，维持恒定物理感受野。
* **4 臂 2×2 正交消融矩阵设计**：
  | 实验臂 | 配置文件 | CT-Tubelet | B-AMoD | CT-Conv1d | 验证目的 |
  |---|---|:---:|:---:|:---:|---|
  | **Arm 1 (Full 主方法)** | `duca_ct_dual_phase_bamod_thumos.py` | **ON** | **ON** | **ON** | 完整主方法性能上限 |
  | **Arm 2 (消融 CT-Conv)** | `duca_dual_phase_bamod_thumos.py` | **ON** | **ON** | **OFF** | 验证检测头连续时间尺度自适应卷积贡献 |
  | **Arm 3 (消融 B-AMoD)** | `duca_ct_dual_phase_densevit_thumos.py` | **ON** | **OFF** | **ON** | 验证骨干网边界偏置专家稀疏路由贡献 |
  | **Arm 4 (双消融对照)** | `duca_dual_phase_densevit_stdconv_thumos.py` | **ON** | **OFF** | **OFF** | 经典结构底线对照 |
* **专项追踪与回收文档**：详见专项实验回收文档 👉 [docs/experiments/CT_DP_BAMOD_4ARM_EXPERIMENT_RECORD.md](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/docs/experiments/CT_DP_BAMOD_4ARM_EXPERIMENT_RECORD.md)

---

## 3. 远端部署实验全量映射与运行状态清单

> **注意：外部分支隔离说明**：根据纯净代码库规范，所有属于 **Zoom TAD / ZoomToken** 空间局部多分辨率刷新分支（如 BA-FDR K16 协议 `ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001`）的实验均属于外部独立路线，不纳入本仓库主干实验记录。

下表汇总了本代码库在 N16R4 集群上全部已部署/提交的重点主线实验及其 Slurm Job ID、分配节点与运行状态：

| 实验批次 / 所属体系 | 实验臂 / 变体 | 配置文件 / 启动脚本 | 分配节点 | Slurm Job ID | 部署状态 | 评测协议与说明 |
|---|---|---|:---:|:---:|:---:|---|
| **CT-DP-BAMoD 4 臂消融矩阵** | Arm 1: Full CT-DP-BAMoD | `duca_ct_dual_phase_bamod_thumos.py` | `g0048` | `1264438` | **RUNNING** | 60e, 42e 起每 2 轮评测 |
|  | Arm 2: DP-BAMoD (Std Conv1d) | `duca_dual_phase_bamod_thumos.py` | `g0063` | `1264439` | **RUNNING** | 60e, 42e 起每 2 轮评测 |
|  | Arm 3: CT-DP (Dense ViT-Adapter) | `duca_ct_dual_phase_densevit_thumos.py` | `g0056` | `1264440` | **RUNNING** | 60e, 42e 起每 2 轮评测 |
|  | Arm 4: DP-DenseViT (Std Conv1d) | `duca_dual_phase_densevit_stdconv_thumos.py` | `g0056` | `1264441` | **RUNNING** | 60e, 双消融对照基准 |
| **DUCA-JCT 渐进在线联合训练** | JCT Fixed-384 Official | `duca_online_official_adatad_backend_full_train.py` | `g0030` | `1151303` | COMPLETED | progressive joint schedule |
|  | DUCA-MUST Dynamic Official | `duca_must_dynamic_official_adatad_backend_full_train.py` | `g0030` | `1151304` | COMPLETED | 对偶变量动态预算调控 |
|  | X3D Actionness Grid & Downstream | `duca_online_x3d_official_adatad_backend_full_train.py` | `g0030` | `1151305~1151307` | COMPLETED | 免训练时序动作性先验对照 |
| **DUCA 13,200 更新 CellCF 套件** | Transition Beta=0 | `duca_cellcf_transition_beta0_fixed384_*.py` | `g0030` | `1159416` / `1161079` | COMPLETED | 13,200 次更新, Avg-mAP 64.28 |
|  | CellCF Counterfactual | `duca_cellcf_fixed384_official_adatad_backend_full_train.py` | `g0030` | `1159417` | COMPLETED | 13,200 次更新, Avg-mAP 64.06 |
| **早期离线基线与 Teacher** | GAS-VT Stage0/1 | `c3_gas_vt_stage01_gpu0_provok_*` | GPU0 | `1118197` | COMPLETED | Avg-mAP 44.90~46.55 |
|  | PAction Learned Fixed 384 | `c3_paction_learned_g30_gpu1_*` | GPU1 | `1118197` | COMPLETED | Avg-mAP 59.10~61.02 |
|  | Dense AdaTAD Teacher | `c3_dense_adatad_teacher_full_*` | `g0030` | `1118197` | COMPLETED | T=768 稠密教师基准 |
|  | GT-Boundary Oracle | `oracle_boundary_subsample` (K=384, R=2) | N16R4 | `1001959` | COMPLETED | Avg-mAP 76.67 (理论机制上界) |

---

## 4. 关键实验指标与因果对比总表

下表汇总了本代码库中记录的关键基线、消融实验与理论上限实测数据：

| 实验代号 / 机制方案 | 阶段 / 属性 | 输入预算 $K$ | 训练协议 / 轮次 | Avg-mAP (%) | tIoU 0.3 | tIoU 0.4 | tIoU 0.5 | tIoU 0.6 | tIoU 0.7 | 核心机制特征与因果归因 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| **GT-Boundary Oracle (Job 1001959)** | 特权理论上限 | 384 (from 768) | Adapter/ActionFormer | **76.67** | 83.63 | 81.54 | 78.92 | 73.42 | **65.83** | 证明前置精确边界可带来 ~12% 的巨大提升空间 (上界锚点) |
| **Historical Dense Teacher (T=768)** | 稠密全量基线 | 768 (Dense) | 官方标准协议 60e | **64.39~65.5** | - | - | - | - | - | 稠密特征输入基准 |
| **Historical H65 (Two-Stage)** | 2阶段联合优化 | 384 (Sparse) | 30e Uniform + 60e Joint | **65.39** | 80.52 | 76.10 | 68.12 | 58.15 | **44.03** | 充分语义预热后联合优化，当前稀疏最高点之一 |
| **Exact Uniform 384 (Formal 13.2k)** | 严格均匀基准 | 384 (Exact) | 13,200 updates (132e) | **63.86** | 79.12 | 74.30 | 67.51 | 56.80 | **40.58** | 严格 13,200 次更新基准线，所有稀疏方案必须超越的底线 |
| **Transition Beta=0 (Formal 13.2k)** | 纯转变打分 | 384 (Learned) | 13,200 updates (132e) | **64.28** | 79.65 | 74.82 | 68.01 | 57.25 | **41.65** | 相比 Uniform 提升 +0.42 Avg-mAP，无检测器梯度桥 |
| **CellCF Counterfactual (Formal 13.2k)** | 局部反事实置换 | 384 (Learned) | 13,200 updates (132e) | **64.06** | 79.35 | 74.55 | 67.80 | 57.01 | **41.12** | 相比 Uniform 提升 +0.20 Avg-mAP，受限于局部单元约束 |
| **PAction Learned Fixed 384** | 离线动作性策略 | 384 (Offline) | 标准 AdaTAD 60e | **60.41~61.02** | 76.03 | 71.71 | 64.07 | 53.21 | **40.05** | 空洞控制极佳 ($p_{95}=2.0$)，边界支持好 ($r_1=0.24$) |
| **GAS-VT Fixed 384** | 离线价值传输 | 384 (Offline) | 标准 AdaTAD 60e | **44.90~46.55** | 61.97 | 56.22 | 47.52 | 38.60 | **28.45** | 陷入平台期：边界支持低 ($r_1=0.14$)，存在大空洞 ($p_{95}=96$) |

---

## 5. 配置、工具、脚本与测试映射总表

### 5.1 配置文件索引 (`configs/adatad/thumos/`)
| 配置文件名 | 所属体系 | 核心定位与用途 |
|---|---|---|
| `c3_dense_adatad_teacher_full_train.py` | Family 3 | 稠密 AdaTAD Teacher 全量训练 |
| `c3_detector_aware_ledger_adatad_full_train.py` | Family 3 | 基于 Teacher Utility Ledger 的 AdaTAD 训练 |
| `c3_gas_vt_ledger_adatad_full_train.py` | Family 2 | GAS-VT 离线 Ledger 训练 |
| `c3_paction_learned_ledger_adatad_full_train.py` | Family 2 | PAction Learned 离线 Ledger 训练 |
| `duca_online_official_adatad_backend_full_train.py` | Family 4 | DUCA-JCT 渐进在线联合训练主配置 |
| `duca_must_dynamic_official_adatad_backend_full_train.py` | Family 4 | DUCA-MUST 动态预算在线联合训练 |
| `duca_transition_only_fixed384_official_adatad_backend_full_train.py` | Family 6 | 纯转变打分固定 384 训练 |
| `duca_cellcf_exact_uniform_fixed384_official_adatad_backend_full_train.py` | Family 7 | CellCF 13,200 更新严格均匀基准 |
| `duca_cellcf_transition_beta0_fixed384_official_adatad_backend_full_train.py` | Family 7 | CellCF 13,200 更新 Transition Beta=0 |
| `duca_cellcf_fixed384_official_adatad_backend_full_train.py` | Family 7 | CellCF 13,200 更新 Counterfactual 训练 |
| `duca_allocation_ceiling_physical_grid_evaluator.py` | Family 8 | 分配族物理网格评估配置 |
| `duca_boundary_burst_g1_protected_fixed384_official60.py` | Family 9 | R3 G1 边界微簇受保护梯度桥 60 轮配置 |
| `duca_density_transport_softmax14_fixed384_official60.py` | Family 9 | R4 密度传输 Softmax-14 梯度桥 60 轮配置 |
| `duca_h65_60_stage1_uniform20.py` | Family 10 | H65 压缩课程 Stage-1 (Uniform 20e) |
| `duca_h65_60_stage2_transition20_joint20.py` | Family 10 | H65 压缩课程 Stage-2 (Transition + Joint 40e) |
| `duca_evidence_recovery_full.py` | Family 10 | 8 臂证据恢复全量组合实验 |
| `duca_ct_dual_phase_bamod_thumos.py` | Family 13 | CT-DP-BAMoD 4 臂消融矩阵 Arm 1 (Full 主方法) |
| `duca_dual_phase_bamod_thumos.py` | Family 13 | CT-DP-BAMoD 4 臂消融矩阵 Arm 2 (消融 CT-Conv) |
| `duca_ct_dual_phase_densevit_thumos.py` | Family 13 | CT-DP-BAMoD 4 臂消融矩阵 Arm 3 (消融 B-AMoD) |
| `duca_dual_phase_densevit_stdconv_thumos.py` | Family 13 | CT-DP-BAMoD 4 臂消融矩阵 Arm 4 (双消融对照) |

### 5.2 核心工具链索引 (`tools/bata/`)
| 工具脚本名 | 核心功能 |
|---|---|
| `train_lowres_action_probe.py` | 训练低分辨率动作性探针 |
| `c3_coarse_classifier_model_matrix.py` | 评估探针矩阵并生成候选打分 |
| `detector_teacher_utility.py` | 从 Dense Teacher 导出 Point Responsibility 与 Signed Utility |
| `run_duca_jct_one_step_grad_proof.py` | 验证检测器到选择器的端到端梯度回传（一阶梯度证明） |
| `export_duca_selection_quality.py` | 导出选择器选帧质量与 GT 边界重合指标 |
| `analyze_duca_selection_quality.py` | 聚合分析选帧质量与 Bootstrap 置信区间 |
| `duca_allocation_families.py` | 五类全局预算分配族定义与求解约束 |
| `duca_exact_physical_solver.py` | 基于 DP 与 HiGHS MILP 的物理网格精确求解器 |
| `diagnose_duca_allocation_family_ceiling.py` | 诊断分配族物理天花板与边界召回上限 |
| `duca_cellcf_training.py` | 支持 13,200 次成功更新与 AMP Replay 的训练引擎 |
| `aggregate_duca_r5_paper_matrix.py` | 聚合 R5 矩阵评测结果并生成论文对比表 |

### 5.3 测试验证套件索引 (`tests/`)
| 测试模块 | 验证范围 |
|---|---|
| `test_c3_coarse_classifier_model_matrix.py` | 探针矩阵构建、指标计算与排序正确性 |
| `test_duca_allocation_families.py` | 预算分配族 A/B/C/D/E 的约束满足性 |
| `test_duca_exact_physical_solver.py` | HiGHS MILP 与 DP 求解器的单调性与边界极值 |
| `test_duca_cellcf_contract.py` | 13,200 成功更新与 AMP Replay 契约 |
| `test_duca_joint_training_contract.py` | DUCA-JCT 渐进损失调度与梯度回传契约 |
| `test_duca_selection_quality_analysis.py` | 选帧质量指标（$r_0/r_1/r_2$, 端点距离, 空洞）计算逻辑 |
| `test_duca_h65_60_curriculum_contract.py` | H65 60 轮课程优化器状态隔离与阶段衔接契约 |
| `test_ct_dual_phase_bamod.py` | CT-DP-BAMoD 架构、时间解耦、CT-Conv 与梯度反传 11/11 单元测试 |

---

## 6. 核心科学结论与后续指引

1. **粗动作性（Actionness Coverage）不能单独解决稀疏 TAD**：单纯最大化动作性区域容易导致关键动作起止边界处出现采样空洞（GAS-VT 失败案例）。
2. **时序空洞控制与边界敏感性是稀疏定位的生命线**：PAction Learned 之所以大幅领先 GAS-VT，在于其维持了极紧凑的空洞上限（$p_{95}=2.0$）和更高的局部边界命中率。
3. **两阶段语义预热极具价值**：H65 结构证明，先在均匀采样下训练稳定语义表征，再渐进过渡到学习选择并开启联合优化，是当前达到 65+ Avg-mAP 的关键路径。
4. **Selected-Axis 等间隔假设是高 IoU 定位瓶颈**：现有的 Selected-Axis 映射改变了真实时间跨度，PhysTime-TAL 与 CT-DP-BAMoD 提出的物理连续时间建模是突破该瓶颈的明确理论方向。

---
*文档更新完成，已纳入代码库索引系统。*

