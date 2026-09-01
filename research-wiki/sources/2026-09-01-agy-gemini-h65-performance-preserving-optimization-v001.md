---
title: AGY Gemini 对 H65 性能保持与后续优化的只读咨询
date: 2026-09-01
source_type: external_model_advisory
model: gemini-3.7-flash-high
effort: high
authority: advisory_only
terminal_marker: GEMINI_DUCA_H65_PRESERVATION_ADVISORY_READY
---

# AGY Gemini：H65 性能保持与后续优化咨询

本文件保存 AGY CLI 使用 `gemini-3.7-flash-high`、`effort=high` 产生的完整原文。该回复只提供外部咨询，不改变 Pro 的科学所有权，不授权代码修改、实验部署、held-out 读取或路线切换。项目对回复的事实与因果审计见
`research-wiki/sources/2026-09-01-agy-gemini-h65-preservation-review-audit-v001.md`。

## 完整原文

### 咨询模型身份与顾问声明

* **咨询模型身份**：Gemini 3.7 Flash (High)
* **顾问角色与权限声明**：本报告由独立资深研究员（外部技术顾问）在只读、基于代码与实证的原则下生成。**Pro 仍是项目唯一的科学决策人与路线负责人**。本报告属于外部同行咨询建议，**不构成**对任何代码库的写入授权、不授权打开/探测处于封存状态的留出评测结果、不授权提交新的正式实验作业，亦不代替 Pro 决定路线选择。
* **三层证据隔离声明**：本报告严格区分：
  1. **已证实事实 (Verified Facts)**：代码字面实现、机器审计输出、具有完整数据/种子/评测封存的数值记录；
  2. **机理假说 (Mechanism Hypotheses)**：对现象与因果关系的理论推演与候选解释；
  3. **顾问建议 (Recommendations)**：面向 Pro 的单变量、可证伪实验设计与方法学意见。

---

### A. 实际检查的文件与精确提交 (Files and exact commits inspected)

#### 1. 权威公开 GitHub 身份与提交
* **公共仓库**：[`yuzbo/OpenTAD_C3_CoarseClean_20260702`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702)
* **当前 Wiki / 证据分支与精确提交**：[`codex/duca-wiki-complete-sync-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-wiki-complete-sync-20260831) @ Commit [`aece35372ac8d4a37ceff4ec7f88a1aff0896fb6`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/aece35372ac8d4a37ceff4ec7f88a1aff0896fb6)
* **已准入的干净 H65 科学基座**：Commit [`04c35a3b76897e6c1569eeede41ed3aecaf7f854`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/04c35a3b76897e6c1569eeede41ed3aecaf7f854)
* **当前多预算实现分支与权威恢复提交**：[`feature/duca-h65-system-multibudget-exposure-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-h65-system-multibudget-exposure-v1-20260831) @ Commit [`2b3b3243066a89e5a4be5acdb178c318fbeceac0`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/2b3b3243066a89e5a4be5acdb178c318fbeceac0)
* **中间实现与修复提交**：
  - Initial Builder Commit: [`0d67d49c2fc4a5f50aa784f7809c0dd936492109`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/0d67d49c2fc4a5f50aa784f7809c0dd936492109)
  - True-Time Metadata Fix Commit: [`409f370a7ed14e7077bc87138196ab6abe459f99`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/409f370a7ed14e7077bc87138196ab6abe459f99)
* **数据身份审计权威提交**：Commit [`fdd2bcdddf3f23f3546244adf90c4427ed022837`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/fdd2bcdddf3f23f3546244adf90c4427ed022837)
* **整视频冻结检测器证伪工具提交**：Commit [`33e4ed137c33eef07f0452b44506a6993bdf7535`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535)

#### 2. 本地可读工作区 (Local Worktrees)
* Complete Wiki: `C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702`
* Current Implementation: `C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702`
* Historical Worktrees Checked: `duca-whole-video-consistent-budget-falsifier-v1-20260831`, `duca-full-data-identity-audit-v1-20260831`, `duca-coverage-only-v1-20260829`, `duca-marginal-budget-v1-20260830`, `duca-native-tubelet-coreset-20260828`, `duca-semantic-budget-matched-20260828`.

#### 3. 逐字与逐行检查的核心文档与源码
* **文献与决策总账**：
  - [`PAPER_PROGRESS.md`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702/PAPER_PROGRESS.md)
  - [`research-wiki/query_pack.md`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702/research-wiki/query_pack.md)
  - [`research-wiki/anti_repetition.md`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702/research-wiki/anti_repetition.md)
  - [`research-wiki/decision_history.md`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702/research-wiki/decision_history.md)
  - [`research-wiki/experiments/duca-multi-budget-detector-adaptation.md`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702/research-wiki/experiments/duca-multi-budget-detector-adaptation.md)
  - [`research-wiki/sources/2026-08-31-pro-duca-full-data-identity-admission-v001.md`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702/research-wiki/sources/2026-08-31-pro-duca-full-data-identity-admission-v001.md)
  - [`research-wiki/sources/2026-08-31-duca-full-data-comparable-protocol-v001.md`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702/research-wiki/sources/2026-08-31-duca-full-data-comparable-protocol-v001.md)
  - [`research-wiki/sources/2026-08-31-agy-gemini-duca-post-admission-optimization-v001.md`](file:///C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702/research-wiki/sources/2026-08-31-agy-gemini-duca-post-admission-optimization-v001.md)
* **核心源码与配置**：
  - [`configs/adatad/thumos/duca_h65_system_multibudget_exposure_control.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_h65_system_multibudget_exposure_control.py)
  - [`configs/adatad/thumos/duca_h65_system_multibudget_exposure_candidate.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_h65_system_multibudget_exposure_candidate.py)
  - [`configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py)
  - [`configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py)
  - [`opentad/models/duca/acquisition.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/duca/acquisition.py)
  - [`opentad/models/duca/dynamic_budget.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/duca/dynamic_budget.py)
  - [`opentad/models/selectors/duca_online_frame_selector.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/selectors/duca_online_frame_selector.py)
  - [`opentad/models/detectors/actionformer.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/detectors/actionformer.py)
  - [`tools/bata/validate_duca_h65_system_multibudget_exposure.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/tools/bata/validate_duca_h65_system_multibudget_exposure.py)
  - [`tools/bata/prepare_duca_h65_system_multibudget_exposure.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/tools/bata/prepare_duca_h65_system_multibudget_exposure.py)
  - [`tools/bata/seal_duca_h65_system_multibudget_prediction.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/tools/bata/seal_duca_h65_system_multibudget_prediction.py)
  - [`tools/bata/evaluate_duca_h65_system_multibudget_exposure.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/tools/bata/evaluate_duca_h65_system_multibudget_exposure.py)
  - [`tests/test_duca_h65_system_multibudget_exposure.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/tests/test_duca_h65_system_multibudget_exposure.py)

* **无法直接检查的项目 (Uninspectable items)**：
  - 本地环境中的外部物理路径 `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702` 受沙箱权限保护无法直接作为 Git 根访问，但其在 `C:/Users/skywalker/.codex/worktrees/` 下的分支完整代码均已直接检视。
  - 远端运行中及封存的 held-out 预测文件与 mAP 结果，严格遵守科学协议保持**物理未生成/未打开**状态，顾问未行任何探测。

---

### B. 干净 H65 能力栈拆解与证据级别 (Clean H65 capability stack and evidence level)

经对基座代码 [`04c35a3b...`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/) 的重构分析，干净 H65（Avg-mAP 65.13%，mAP@0.7 43.31%）的性能由以下六个核心能力组件共同支撑：

```
                              [H65 完整能力栈]
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ 1. 轻量粗分类与边界感知探针 (ASFormer Scout: p_action + transition_score) │ [代码确证 + 实证支撑]
 ├──────────────────────────────────────────────────────────────────────────┤
 │ 2. 预算校准系统采样 (BCSR: Coverage Floor 0.05 + 严格单调累积确定性选点)  │ [代码确证 + 实证支撑]
 ├──────────────────────────────────────────────────────────────────────────┤
 │ 3. 两阶段课程学习 (Stage-1 30 epochs 均匀预热 + Stage-2 60 epochs 渐进同伦)│ [代码确证 + 实证支撑]
 ├──────────────────────────────────────────────────────────────────────────┤
 │ 4. 直通式受控检测器反馈 (Straight-Through Bridge + 余弦退火权重 0.25)    │ [代码确证 + 实证支撑]
 ├──────────────────────────────────────────────────────────────────────────┤
 │ 5. 统一物理时间重映射 (384-Point ActionFormer Detection Grid Interpolation)│ [代码确证 + 实证支撑]
 ├──────────────────────────────────────────────────────────────────────────┤
 │ 6. 充足的优化时钟预算 (30+60 周期, 累计 ~9,000 成功更新步)                │ [关键混杂因子]
 └──────────────────────────────────────────────────────────────────────────┘
```

#### 组件详析与证据分级

1. **轻量粗分类与状态转移感知 (ASFormer Scout Probe)**
   - *代码实现*：基于 64x64 空间 stem + ASFormer 时序编码器，提取 7 维 deploy-visible 状态特征（`p_action`, `uncertainty`, `entropy`, `delta_p_action`, `abs_delta_p_action`, `uncertainty_peak`, `transition_score`）。训练期受 BCE 动作性、转移分布（Gaussian Mass）与真实边界显式监督。
   - *分级*：**代码确证 (Code-confirmed) + 实证支撑 (Evidence-supported)**。单纯的动作性高分容易在动作内部扎堆，而 $\Delta p_{\text{action}}$ 与 transition_score 能够精确捕捉动作起止边界，是高 tIoU（mAP@0.7）的关键。

2. **预算校准系统采样 (Budget-Calibrated Systematic Sampling, BCSR)**
   - *代码实现*：在 [`opentad/models/duca/acquisition.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/duca/acquisition.py) 中，由 priority 计算采样密度，设定 `density_coverage_floor = 0.05` 与平滑核（kernel=5），通过确定性累积采样输出严格递增的离散点集 $S_{384}$。
   - *分级*：**代码确证 + 实证支撑**。覆盖下限确保全视频没有观测盲区（避免漏掉短动作），而转移偏置将采样预算聚集在边界区域。它显著优于单纯的贪心 Top-K（Top-K 会产生局部极端聚集和大空洞）。

3. **两阶段课程学习与软平滑同伦 (Curriculum & Progressive Homotopy)**
   - *代码实现*：Stage-1 均匀 30 轮稳定 Scout；Stage-2 60 轮（6,000 updates）中，前 3,000 步通过余弦日程将 `policy_alpha` 从 0 升至 1.0、检测器梯度从 0 升至 0.25，后 3,000 步由 TAD 主导，但始终保留非零的语义损失权重（动作性 0.25、转移 0.10、边界 0.25）。
   - *分级*：**代码确证 + 实证支撑**。避免了训练初期未收敛的检测器梯度摧毁 Scout 的判别能力，实现了多模块平滑协同。

4. **受控的直通式检测器反馈 (Protected ST Detector Feedback)**
   - *代码实现*：通过软分配代理向 selector 传递检测梯度，配合分层学习率（Coarse trunk 1e-5, Action head 2e-5, Transition scorer 5e-5, Adapter 2e-4, ActionFormer 1e-4）与梯度裁剪。
   - *分级*：**代码确证 + 实证支撑**。参数组学习率跨越两个数量级，防止下游检测头的强梯度冲垮上游轻量特征。

5. **时序几何与物理时间重映射 (Physical-Time Grid Remapping)**
   - *代码实现*：非均匀选取的 384 帧经过 16 帧 packet 形式送入 VideoMAE-S 提取特征后，通过 `interpolate_acquisition_time_to_detector_grid` 线性插值映射到固定的 384 点检测器网格，推理时将 proposal 边界逆映射回秒域。
   - *分级*：**代码确证 + 实证支撑**。使下游标准的 ActionFormer FPN 和 Head 能在规整网格上运行，同时保留帧间真实的物理间隔信息。

6. **优化时钟预算混杂 (Optimization Budget as a Critical Confound)**
   - *分析*：H65 的 65.13 是在 30 轮 Stage-1 + 60 轮 Stage-2（累计 ~9,000 次成功更新）下获得的。多组 60 轮压缩实验（20+40, 30+30）均产生 1.5 ~ 2.66 pp 的显著下降。
   - *分级*：**实证确证的关键混杂因子**。多模块联合适应（5 个参数组协同）对优化步数极为敏感。任何试图在缩减更新步数下验证新机制的做法，都会因优化不充分而与机制失效产生混淆。

---

### C. 历史路线为何屡次破坏 H65 性能 (Why previous routes lost performance)

回顾 DUCA 演进历史，多条路线由于未能保护上述核心能力栈中的某一个关键环节，导致性能严重衰退：

```
[历史探索失败根因剖析]
├── 1. 训练压缩路线 (20+40, 30+30)           ──> [优化时钟不足] 5 个异质参数组未能完成同伦收敛 (降 1.5-2.7 pp)
├── 2. 连续片段路线 (16-frame chunking)      ──> [时序几何崩塌] 牺牲全视频跨度，产生严重时序盲区 (降 >15 pp)
├── 3. 无保护端到端桥接 (UVT, Fovea-Bridge)   ──> [梯度扰动 Scout] 缺乏课程缓冲，无约束检测梯度破坏粗分类特征 (降 8-10 pp)
├── 4. 局部网格约束 (CellCF)                 ──> [动作空间禁锢] 限制在 uniform cell 内微调，无法实现跨区域预算重分配
├── 5. 任务状态 Native-Tubelet Coreset       ──> [破坏均匀覆盖] 贪心点集破坏了时序规整性与端点支撑 (降 1.32 pp)
├── 6. Coverage-v1 (Facility Location)       ──> [偏离检测边界] 优化语义代表性而非边界梯度，错失高 tIoU 关键帧
└── 7. 冻结检测器三档转移 (Marginal-v1 704态)  ──> [跨预算特征偏移] 冻结检测器对 K256/K512 尺度失准 (0/704 通过)
```

1. **训练压缩路线 (20+40, 30+30 AM-RPCH25 / LongCosine-H6000)**
   - *现象*：20+40 降至 62.46，30+30 降至 63.22 ~ 63.56。
   - *归因*：**优化步数不足引发的收敛不完全**。Stage-2 包含 5 组学习率跨度达 20 倍的参数。前 3,000 步的同伦过渡期被压缩后，模型在尚未稳定选帧策略的情况下就过早进入 TAD 阶段，产生优化欠拟合。

2. **连续片段路线 (Continuous 16-frame clips, FZ / JT)**
   - *现象*：FZ 为 49.89，JT 为 47.24（暴跌 > 15 pp）。
   - *归因*：**时序全局感受野的毁灭性破坏**。强制连续抽取 16 帧片段，导致可覆盖的时间窗口极为狭窄，视频大部分区域完全丢失，ActionFormer 无法构建长程时序上下文，高 tIoU 定位彻底失效。

3. **无保护直接端到端桥接 (UVT, Fovea/Query-Bridge)**
   - *现象*：Avg-mAP 跌落至 54.67 ~ 57.35。
   - *归因*：**梯度穿透导致 Scout 语义漂移**。缺少 Stage-1 预热与 Stage-2 严格受控的同伦 schedule，检测头的反向梯度直接剧烈冲刷底层特征，使 Scout 丧失了准确预测 $\Delta p_{\text{action}}$ 的能力，陷入恶性循环。

4. **局部网格约束分配 (CellCF)**
   - *现象*：CellCF 64.06 vs transition 64.28。
   - *归因*：**动作空间表达能力不足**。CellCF 强制每个均匀锚点只能在自身 cell 内移动 $\pm 1$ 个点，无法将预算跨区间转移到长动作或高密度动作区域，退化为局部抖动。

5. **原生 Tubelet 时序 Coreset (Native-Tubelet Coreset)**
   - *现象*：Coreset 62.81 vs matched uniform 64.13（相对下降 1.32 pp，mAP@0.7 下降 1.89 pp）。
   - *归因*：**离散贪心选择破坏了时序平滑与端点覆盖**。Coreset 算法在 192 个 tubelet 网格上贪心选取，破坏了 VideoMAE 所依赖的时序连续性先验，且在动作边界端点处缺乏均匀支撑。

6. **设施选址覆盖 (DUCA-Coverage-v1)**
   - *现象*：在真实训练数据无标签重放门（PRE_RUN）中直接触发停止。
   - *归因*：**优化目标与 TAD 任务错位**。Facility location 追求特征空间的几何代表性（覆盖中心），而非边界变化率，导致有限的预算被浪费在平稳的背景或动作内部，未能给边界定位提供密集的时序证据。

7. **冻结检测器三档跨预算转移 (Marginal-v1 / 96-state / 704-state)**
   - *现象*：704 个全视频候选无一达到 $+0.8/+1.0$ 联合门。
   - *归因*：**冻结 ActionFormer 的跨预算表征与尺度误校准**。检测头在单一 $K=384$ 密度下训练，当接收 $K=256$（稀疏、上下文缺失）或 $K=512$（密集、特征幅值与感受野偏移）时，其内部回归分支产生系统性预测偏差。

---

### D. 当前实现 `04c35a3b...2b3b3243` 的能力保全性审计 (Preservation audit)

对当前权威实现提交 [`2b3b3243066a89e5a4be5acdb178c318fbeceac0`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/2b3b3243066a89e5a4be5acdb178c318fbeceac0) 进行代码级保全性核验：

```
[当前多预算暴露实现保全性审计矩阵]
┌───────────────────────────────────┬──────────────┬────────────────────────────────────────────────────────┐
│ 审计维度                           │ 保全状态     │ 代码证据与实现机制                                     │
├───────────────────────────────────┼──────────────┼────────────────────────────────────────────────────────┤
│ 1. 初始 K384 前向等价性 (Parity)   │ 完全保全     │ 当 requested_k=384 时直通原 H65 路径，数值误差 < 1e-6  │
│ 2. 6,000 步多预算更新抗遗忘能力   │ 结构性受控   │ 50% 步数固定走 K384，保留原参数组 LR 与 EMA 轨迹       │
│ 3. 真实重型算力匹配 (Observation)  │ 精确匹配     │ 经全量 200 视频短窗折叠校准，期望观测数严格对齐 K384    │
│ 4. 参数漂移与未授权梯度隔离       │ 严格隔离     │ VideoMAE 冻结，仅更新 5 个授权参数组，无跨界梯度反传   │
│ 5. 批次同质性 (Batch Homogeneity) │ 完全消除异质 │ 单一 Update 内部所有样本采用同一 K 档，杜绝动态 Padding│
│ 6. 独立随机数流与时钟隔离         │ 严格确定性   │ 预算序列采用代数双射 PRNG，不消费数据增强 RNG 流        │
│ 7. 短窗口真时间折叠元数据修复     │ 缺陷已修复   │ 409f370a 修复了 inactive padding 写入映射的阻断 Bug    │
│ 8. 优化器、调度器与 EMA 时钟      │ 严格单时钟   │ 仅在 successful optimizer update 时推进，AMP 失败不步进│
└───────────────────────────────────┴──────────────┴────────────────────────────────────────────────────────┘
```

#### 关键技术审计点深度分析

1. **初始 K384 前向路径恒等性 (Initial K384 Forward Parity)**
   - *源码证据*：在 [`duca_online_frame_selector.py#L1423-L1426`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/opentad/models/selectors/duca_online_frame_selector.py#L1423-L1426) 中，当 `effective_budget == 384` 时，代码直接将 `detector_grid` 与 `detector_mask` 赋予 `baseline_detector_grid` 与 `baseline_detector_mask`；[`test_duca_h65_system_multibudget_exposure.py`](file:///C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702/tests/test_duca_h65_system_multibudget_exposure.py) 包含单步数值等价测试。这确保了在初始化时刻，候选模型对 K384 的行为与基座 H65 百分之百一致。

2. **6,000 次多预算更新后的能力保全 (Preservation after Multi-Budget Updates)**
   - *分析*：Candidate 模型中，$K=384$ 的更新占比被固定为严格的 50%（3,000 次），其余 50% 均匀分配给 $K=256$（1,454 次）与 $K=512$（1,546 次）。
   - *保全机理*：由于 $S_{256} \subset S_{384} \subset S_{512}$ 呈严格嵌套关系，无论哪一档预算，所选帧的时序优先级顺序完全由同一组 Scout 分数决定。这避免了不同预算训练在参数空间产生相互冲突的梯度方向。
   - *潜在风险*：在非 K384 的 3,000 次更新中，ActionFormer 接收到的特征是插值后的 384 点表示，可能对原生的 1:1 观测表示产生轻微的参数漂移（Parameter Drift）。

3. **真实重型算力匹配与短窗口折叠 (True Heavy-Compute Matching & Short-Window Collapse)**
   - *源码证据*：代码未采用名义上的 $256/384/512$ 常数计算概率，而是在全量 200 视频上实测短窗口折叠后的真实平均观测数 $\mu_{256}=255.45, \mu_{384}=379.80, \mu_{512}=496.72$，并精确解算概率：
     $$p_{256} \approx 0.24235, \quad p_{384} = 0.50000, \quad p_{512} \approx 0.25765$$
     期望实际观测数与 Control 组严格一致（偏差 $< 0.01\%$），消除了“Candidate 靠更多帧取胜”的算力混杂。

4. **批次同质性与变长 VideoMAE 执行**
   - *源码证据*：每个 optimizer update 步仅处理单一 $K$ 档（Budget-Homogeneous Batching），VideoMAE 输入张量被精准截断至 $\lceil K_{\text{actual}} / 16 \rceil \times 16$ 帧，杜绝了同一 Batch 内长短不一导致的 Padding 浪费或 LayerNorm 统计污染。

5. **时钟、RNG 与元数据一致性**
   - *源码证据*：预算采样使用基于更新步数的代数双射 `(step * 2593 + seed) % 6000`，完全不消耗 PyTorch 数据增强的 PRNG 流，保证 Control 与 Candidate 见到的样本增强、时序翻转等数据序列完全一致。

---

### E. 正在运行的实验能证明什么、不能证明什么 (What the running experiment can and cannot prove)

当前授权的三种子盲执行实验（Control-K384 vs Candidate-Multibudget）具有严格的认识论边界：

```
                              [当前实验证明力边界]
  ┌────────────────────────────────────────┬────────────────────────────────────────┐
  │              能够证明 (CAN PROVE)       │           不能证明 (CANNOT PROVE)      │
  ├────────────────────────────────────────┼────────────────────────────────────────┤
  │ 1. Stage-2 多预算暴露能否在不降低 K384 │ 1. 不能证明“动态预算控制器 (Controller)│
  │    性能的前提下，赋予系统跨尺度鲁棒性  │    有效”（当前未引入内容感知控制器）   │
  │ 2. 多预算训练能否在固定混合清单 (Mixed)│ 2. 不能证明“DUCA 超越 Dense AdaTAD”    │
  │    上显著超越 K384 单一暴露基线        │    （Dense 68.73 仍为上界参考）        │
  │ 3. 统计不确定性与三 seed 泛化稳定性    │ 3. 不能证明在其他骨干/数据集上的通用性 │
  └────────────────────────────────────────┴────────────────────────────────────────┘
```

#### 1. 能够确立的科学事实
* 若 **PASS**：证实“**在训练期暴露于嵌套的多尺度时序观测，能够消除 downstream ActionFormer 的跨预算表征不匹配，且不破坏 K384 核心能力**”。这为后续设计部署可见的动态预算分配策略打开了必要的物理表示裕量。
* 若 **FAIL**：证实“**单纯在训练期混合预算暴露，无法克服离散选帧带来的时序非连续性障碍**”，从而彻底关闭基于 H65 嵌套时序采样的多预算适应路线。

#### 2. 绝对不能超范围宣称的事项
* **非动态控制器证明**：本实验使用的是预先登记、与视频内容无关的 `fixed mixed-budget manifest`，不包含任何基于视频复杂度或动作密度的自适应控制器，不能宣称实现了“自适应动态计算”。
* **非端到端加速证明**：虽然骨干观测输入在期望上与 K384 匹配，但尚未进行全链路 Wall-clock 测量，不能直接宣称端到端延迟加速比。

#### 3. 关于 K384 安全门（$-0.2\text{ pp}$）的科学充分性评估
* *评估结论*：**在统计学上是充分且必要的**。$-0.2\text{ pp}$（即 0.002 的绝对 mAP 损失）处于 THUMOS14 在 3 随机种子平均下的测量噪声容限之内（通常 seed 间标准差在 $\pm 0.15 \sim \pm 0.30\text{ pp}$）。
* *补充实证要求*：为在不重新打开留出集调参的前提下提供更严谨的证据，一次性打开评测时，除汇报 Avg-mAP 与 mAP@0.7 外，必须一并输出：
  1. **高 tIoU 定位起终点绝对误差分布 (Boundary Localization MAE)**；
  2. **短动作（$< 1.5\text{s}$）与长动作（$> 5.0\text{s}$）的分层 Recall**；
  3. **NMS 前后的假阳性率 (False Positive Rate at High Scores)**。

---

### F. 排序的“保全优先”单变量优化图谱 (Ranked one-variable preservation-first optimization map)

若要在后续研究中进一步提升 DUCA 而绝不摧毁 H65 核心基准，必须遵循“**严格单变量、强基准对齐、带安全锚点**”的原则。以下是顾问为 Pro 整理的候选机制排序图谱：

```
                     [保全优先 (Preservation-First) 候选机制排期]
                                          │
       ┌──────────────────────────────────┼──────────────────────────────────┐
       ▼                                  ▼                                  ▼
【机制 1 (首选/最高信息增益)】     【机制 2 (高稳定性/低风险)】       【机制 3 (轻量校准/无参数漂移)】
部署可见内容感知控制器             H65 教师特征锚定正则化             预算特定轻量归一化/时钟适配
(Deploy-Visible Controller)        (Teacher Feature Distillation)     (Budget-Specific Adaptation)
```

---

#### 机制 1（首选）：基于 Scout 状态复杂度的部署可见动态控制器 (Deploy-Visible Scout-Complexity Budget Controller)

* **保护的 H65 能力**：保护已训练完成的 H65 Stage-2 检测器与 BCSR 嵌套选帧器完全不变。
* **改变的单一变量**：将测试期预算从“静态固定 K / 随机 Mixed”改为“**基于 Scout 已输出的 7 维特征（全视频动作密度、转移峰值数量、不确定性熵）确定性映射为 $K \in \{256, 384, 512\}$**”。
* **可证伪预测**：在全量 211 验证集上，内容感知分配的 Avg-mAP 与 mAP@0.7 显著高于同等各预算频次分布的内容无关（Content-Agnostic）Random Mixed Manifest（$\Delta \ge +0.5\text{ pp}$），且总观测成本不高。
* **严格匹配对照**：同一模型在相同预算分布下的 Permuted/Random Mixed Manifest。
* **最小决定性实验**：在冻结的 Stage-2 多预算检查点上，利用 200 训练集的无标签 Scout 特征拟合阈值，对 211 留出集执行一次性无标签推理评测。
* **主要混杂与停止规则**：
  - *混杂*：控制器分配的预算恰好偏向长视频而带来的算力偏移；
  - *停止规则*：若内容感知分配未能击败同分布随机分配，立即停止该控制器。

---

#### 机制 2（备选）：H65 教师特征空间几何锚定 (H65 Teacher Latent Anchor Regularization)

* **保护的 H65 能力**：保护 ActionFormer 在 K384 上的特征表征流形不发生漂移，杜绝灾难性遗忘。
* **改变的单一变量**：在 Stage-2 多预算训练中，当采样到 $K=256$ 或 $K=512$ 时，增加一项与冻结的干净 H65 Stage-1/Stage-2 教师特征在 384 规整网格上的 **Smooth-L1 潜空间一致性损失**：
  $$\mathcal{L}_{\text{anchor}} = \frac{1}{384}\sum_{t=1}^{384} \| F_{\text{candidate}}^{(K)}(t) - \text{stop\_grad}(F_{\text{H65}}^{(384)}(t)) \|_{\text{smooth\_L1}}$$
* **可证伪预测**：Candidate 在 K384 上的性能下降幅度从 $\pm 0.2\text{ pp}$ 收窄至 $\pm 0.05\text{ pp}$，同时在 Mixed 上的增益保持在 $+0.8\text{ pp}$ 以上。
* **严格匹配对照**：不加 $\mathcal{L}_{\text{anchor}}$ 的当前 Candidate 多预算训练。
* **最小决定性实验**：200 视频、单 seed (3407) 6,000 updates，验证 K384 保持率与 Mixed 表现。
* **主要混杂与停止规则**：
  - *混杂*：锚定权重过大导致模型退化为对 H65 的死记硬背，丧失对 $K=512$ 额外信息的利用能力；
  - *停止规则*：若 $\mathcal{L}_{\text{anchor}}$ 使 Mixed 收益低于 $+0.3\text{ pp}$，立即停止。

---

#### 机制 3（备选）：预算特定轻量归一化与缩放层 (Budget-Conditioned Lightweight Scale & Shift)

* **保护的 H65 能力**：保持 ActionFormer 主干 Transformer 块全部参数共享，不改变时序感受野。
* **改变的单一变量**：仅在 ActionFormer 的 FPN 输入前，为 256、384、512 三档预算分别引入一组可学习的 1D 缩放与平移参数 $(\gamma_K, \beta_K) \in \mathbb{R}^C$（参数量 $< 0.1\%$）。
* **可证伪预测**：消除不同预算由于插值密度差异引起的特征方差偏移，显著改善高 tIoU 下的回归定位稳定性。
* **严格匹配对照**：完全无预算条件参数的当前 Candidate。
* **最小决定性实验**：200 视频训练，观察 3 档预算下各自的损失收敛曲线与 211 留出集表现。
* **主要混杂与停止规则**：
  - *混杂*：引入过多条件参数导致特定预算过拟合；
  - *停止规则*：若 K384 性能下降 $> 0.2\text{ pp}$，立即终止。

---

### G. 当前实验终态条件下的唯一科学问题推荐 (Outcome-conditioned next scientific questions)

为协助 Pro 在当前盲执行训练终态产生后迅速作出高信息增益的科学裁决，针对四种可能的终态结果，顾问分别给出**唯一推荐的下一步科学问题**：

```
                                [终态结果与科学裁决映射]
                                           │
         ┌──────────────────┬──────────────┴───────────────┬──────────────────┐
         ▼                  ▼                              ▼                  ▼
    【结果 1】          【结果 2】                     【结果 3】          【结果 4】
   K384 不安全       K384 安全但无 Mixed 增益       有点增益但区间跨零     全部门通过 PASS
         │                  │                              │                  │
         ▼                  ▼                              ▼                  ▼
  [问题 1: 表征遗忘]  [问题 2: 时序非连续性上限]     [问题 3: 种子间方差]   [问题 4: 内容感知控制器]
```

1. **若终态为：K384 安全门失败（$\Delta\text{Avg-mAP} < -0.2\text{ pp}$ 或 $\Delta\text{mAP@0.7} < -0.2\text{ pp}$）**
   - *唯一推荐科学问题*：
     > **“K384 的性能衰退是由多预算梯度对 Scout 粗探针的干扰引起，还是由 ActionFormer 检测头对插值网格的表征遗忘引起？”**
     （通过单步特征探测，隔离 Scout 分辨率漂移与 Head 参数漂移，决定是否引入 Teacher Anchor）。

2. **若终态为：K384 安全门通过，但 Mixed 增益未达标（$\Delta\text{Avg-mAP} < +0.8\text{ pp}$ 或 $\Delta\text{mAP@0.7} < +1.0\text{ pp}$）**
   - *唯一推荐科学问题*：
     > **“在 H65 嵌套时序采样框架下，阻碍跨预算性能收益的根本瓶颈是 $K=256$ 丢失了关键局部上下文，还是 $K=512$ 的线性插值无法提供额外的有效时序分辨率？”**
     （分别检验 K256 单档与 K512 单档的独立效能，确认是否达到时序离散采样的物理上限）。

3. **若终态为：点估计达到增益门，但 10,000 次整视频配对 Bootstrap 95% 置信区间下界包含零（下界 $\le 0$）**
   - *唯一推荐科学问题*：
     > **“增益统计显著性不足是由少数长视频的极大方差主导，还是由于种子间优化轨迹发散引起的样本间不稳定性？”**
     （执行逐视频方差分解与长度分层分析，判断是特定样本类型的长尾噪声，还是系统性机制不显著）。

4. **若终态为：全部门禁完全通过（PASS：K384 安全、Mixed 达标、95% CI 下界 $> 0$、算力匹配）**
   - *唯一推荐科学问题*：
     > **“在冻结当前已获得跨预算兼容性的 Stage-2 模型的前提下，基于部署可见 Scout 特征的内容感知预算分配器，能否在相同平均观测成本下显著超越当前内容无关的 Fixed Mixed 基线？”**
     （直接进入论文核心主张的最后一步：证明语义自适应动态计算的优越性）。

---

### H. 极简论文级实验阶梯 (Minimal publication-grade experiment ladder)

为确保论文成果坚不可摧，同时避免算力浪费，构建以下绝不牺牲干净 H65 锚点的四步阶梯：

```
                    [极简论文级实验推进阶梯]
 ┌─────────────────────────────────────────────────────────────┐
 │ 阶梯 4：外部泛化验证 (ActivityNet-1.3 / FineAction)         │
 │   - 仅在主方法完全锁定后执行，验证机制通用性                │
 ├─────────────────────────────────────────────────────────────┤
 │ 阶梯 3：端到端真实算力与时延标定 (Wall-Clock Latency Profile)│
 │   - 同硬件、同精度，全链路耗时分解 (Decode/Scout/MAE/Head)  │
 ├─────────────────────────────────────────────────────────────┤
 │ 阶梯 2：内容感知自适应控制器 vs 随机分配消融               │
 │   - 冻结 Stage-2 模型，验证语义动态分配相对于 Fixed Mixed 增益 │
 ├─────────────────────────────────────────────────────────────┤
 │ 阶梯 1：全量多预算暴露跨尺度兼容性验证 (当前进行中)         │
 │   - 200 训练 / 211 留出, 3 种子盲执行, 10,000 次 Bootstrap  │
 └─────────────────────────────────────────────────────────────┘
```

* **阶梯 1（基石）：全量数据、三种子、配对显著性闭环（当前正在运行）**
  - 数据：完整 200 `training` / 完整 211 `validation`；
  - 协议：6,000 successful updates，同 Stage-1 起点，Terminal update-6000 EMA；
  - 统计：10,000 次整视频配对 Bootstrap，汇报点估计、均值、标准差及 95% CI。
* **阶梯 2（核心创新）：内容感知自适应控制器消融**
  - 保持底层模型与选帧器冻结，仅对比：`固定 K384` vs `Fixed Mixed` vs `Scout 内容感知动态分配`；
  - 核心证明点：动态计算的增益来自“对视频内容的语义理解”，而非单纯的“算力混合扰动”。
* **阶梯 3（效率落地）：同硬件真实全链路时延剖析**
  - 测量指标：真实进入 VideoMAE 的 Tubelet 计数、GPU 峰值显存、端到端 P50/P95 墙上延迟（含视频解封装、Scout 粗探针、主干特征提取、ActionFormer 头与 Soft-NMS）；
  - 核心证明点：观测帧数削减能够真实转化为系统级推理延迟降低。
* **阶梯 4（扩展泛化）：跨数据集与跨骨干基线（可选扩展）**
  - 在 THUMOS14 证据链完全闭环后，原样迁移至 ActivityNet-1.3，验证方法在稀疏长动作数据集上的鲁棒性。

---

### I. 向 Pro 的简明独立咨询建议 (Concise advisory recommendation to Pro)

1. **必须绝对保持不变的基石 (Must Remain Immutable)**：
   - **H65 Stage-1 检查点与 BCSR 选帧机制不可动摇**：禁止引入 Gumbel-Softmax、Mamba、Block Drop、频域模块或重新设计选帧打分器；
   - **数据身份与留出集隔离边界不可动摇**：严格锁定 200 训练 / 211 留出，在三种子全部完成封存前，严禁任何提前探测；
   - **6,000 成功更新步数与优化配置不可动摇**：禁止通过压缩轮数或调整峰值学习率进行所谓“快速验证”。

2. **当前依然缺失的关键证据 (Missing Evidence to be Produced)**：
   - 当前已完成 seed 3407 训练终态，但 seed 3408 与 3409 仍在依赖执行中，**目前不存在任何有效的 mAP、置信区间或成本结论**；
   - 尚缺少真实硬件环境下的全链路端到端时延（Wall-clock Latency）分项实测数据。

3. **终态产生后最高信息增益的下一步行动 (Single Next Action with Highest Information Gain)**：
   - 在六个训练单元全部成功结束、九份预测视图完成 SHA-256 封存后，**一次性打开 211 held-out 评测**，计算官方 mAP 与 10,000 次整视频配对 Bootstrap。
   - 若通过，**立即将下一步锁定为“设计基于 Scout 7 维特征的极简内容感知控制器”**，直接向论文最终主张迈进；若未通过，则依据上述 F/G 节分类图谱，果断归因并封存负结果，不进行无休止的微调抢救。

---

GEMINI_DUCA_H65_PRESERVATION_ADVISORY_READY


