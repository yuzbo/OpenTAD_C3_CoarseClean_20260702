# ARIS CPR Plan — DUCA dynamic-budget route material (C/P/R)

- 日期：2026-08-14（截止 2026-08-14T16:15:00+08:00）
- 阶段：C/P/R 材料准备（transport-only correction，非科学重试）
- 执行身份：唯一替换 ARIS DeepSeek V4 Pro Executor / First Author
- 结论类型：仅本地科学规划与记忆文档；无实现、无 PRE_RUN、无 pilot、无训练、无性能/成本/claim 结论

---

## 0. 环境与证据边界事实（transport correction，必须记录）

这是一个 **transport-only correction**。前一个进程 `j-c1wd5v` 在路由决策前被终止，因为其 prompt 错误地把 SparseHead 基础提交 `a6bdc084` 当成 DUCA 锚点。

已核实并固化的精确事实：

- **cwd**：`E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702`。
- **git common-dir**：`.git`。
- **HEAD**：`a6bdc084cc145c80b6b2c68d0a38f0deea3e8518`（提交信息 `docs: freeze sparsehead evidence-first diagnostics design`，作者 Codex，2026-07-28）→ **这是 SparseHead 提交，不是 DUCA revision**。
- **branch**：`codex/duca-total60-plugin-cvpr-20260727`。
- **工作树**：脏，`git status --porcelain` 计数约 275 条。**这是 DUCA 证据边界**。

结论：**绝不能称本工作树为"clean frozen DUCA revision"**。DUCA 证据 = 命名的 working-tree 表面 + `research-wiki/`。a6bdc084、SparseHead Route-T 及任何 SparseHead 故事/revision/结果 **均不得作为 DUCA 证据**。`.codex_tmp` 材料仅在当前源码实际 import 时才视为当前生产代码，否则为历史/草案，缺失 provenance 时如实报告。

---

## 1. 工作树表面分类（parent-fidelity / 本地 MATR 检查）

以下为逐文件诚实分类，来自 `git ls-files` / `git status --porcelain` 与源码读取。

### 1.1 DUCA working-tree 表面（untracked，即 DUCA 证据边界）

| 文件 | 关键符号 | 分类 |
|---|---|---|
| `opentad/models/duca/dynamic_budget.py` | `PrefixMarginalUtilityBudgetController`（policy=`prefix_marginal_utility_stop`），`DynamicBudgetDecision` | **真实已有生产 dynamic-budget 实现**（dynamic_must / 边际效用-vs-价格前缀停止 + Lagrangian 对偶 `lambda_dual`） |
| `opentad/models/duca/acquisition.py` | `DucaAcquisitionAdapter`、`budgeted_center_radius_decode`、`soft_center_radius_coverage`、`hard_topk_st`、`duca_forward_train/test`、`duca_losses` | 历史 online/selected-axis DUCA 采集适配器（消费 `PrefixMarginalUtilityBudgetController`） |
| `opentad/models/duca/density_decode.py` | `decode_duca_density_positions_v001`、`canonical_uniform_positions`、`project_duca_density_positions`、`DUCAProjectionError` | **frozen fixed-K 有界密度分位解码器**（`DUCA_FIXEDK_BOUNDED_DENSITY_QUANTILE_ACQUISITION-v002`；纯 Python 精确整数投影） |
| `opentad/models/duca/__init__.py` | 再导出上述全部 | 生产导出面 |
| `opentad/models/selectors/duca_online_frame_selector.py` | `DucaOnlineFrameSelector`（历史名 Online） | 历史 online/selected-axis 帧选择器 |
| `opentad/models/selectors/__init__.py` | 修改 | DUCA 导出接线 |

### 1.2 SparseHead 基础提交表面（tracked、继承、**非 DUCA**）

| 文件 | 关键符号 | 分类 |
|---|---|---|
| `opentad/models/selectors/pc_ot_mras_dynamic_budget_controller.py` | `PCOTMRASDynamicBudgetController`（阈值 bucketize 的 transport-signal → 变预算） | PC-OT-MRAS 家族，SparseHead 基础，非 DUCA |
| `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py` / `pc_ot_mras_reader.py` / `lowcost_acquisition_browser.py` | PC-OT-MRAS 前选择器/reader/browser | 同上，继承，非 DUCA |
| `configs/adatad/thumos/pc_ot_mras_*.py`（7 个） | physical-grid ActionFormer + PC-OT-MRAS | 同上，继承，非 DUCA |

**provenance 备注（诚实报告）**：`density_decode.py` 为 untracked working-tree 文件，docstring 自称"authoritative production implementation"，但当前无 deploy-visible reader 消费它（仅 `__init__.py` 再导出），且 `PAPER_PROGRESS.md` 将其记录为 `PROTOTYPE_ONLY`（无 production reader/evaluator/Slurm argv）。因此：该文件是 **frozen fixed-K 解码器的源码真身**，但**只作 baseline/attribution/fallback 与内层复用原语**，不得称"已接入生产的完整实现"。

### 1.3 仅讨论、无实现（不得误报为实现/证据）

- `DUCA-RIME`（`idea:duca-rime`）：`discussed / mathematical_contract_not_frozen / implementation_not_started`。
- outer-budget-inner-transport（`exp:duca-dynamic-k-rime-oracle`）：`discussed_proposal / no_training_authorized / no_result`。
- 工作树中未发现 `outer_budget` / RIME / budget-policy 的生产实现；`budget_policy` 只是历史 selector/adapter 的元数据字符串。

**MATR 结论**：两条极端都不得误报——(i) `PrefixMarginalUtilityBudgetController` + center-radius 是**真实已有生产 dynamic-budget 工作**（证据类 `paused/negative`）；(ii) RIME / outer-budget-inner-transport 是**纯讨论/规划**，无实现、无证据。

---

## 2. 科学边界（必须保留为负证据）

- sealed U/O/R execution-surface 路线 **终止**：旧包 `657678946` 因 `--cfg-options` 可在校验前关闭 reachability gate 的 `required`（第二等价 launch-bypass 缺陷）而终止；一次修正环关闭，不得第三次修正。
- 替代包 `DUCA_UOR_SEALED_EXECUTION_SURFACE-v001`（`6515ebf5`）经独立 Critic 终态 `SEALED_REPLACEMENT_BLOCKED`：其 runner/evaluator 只校验 4 个字段 + 只读态，未校验闭合语义（`NOT_EXECUTED`/`PRE_RUN_BLOCKED`/`INACCESSIBLE_FORBIDDEN`/`EMBARGOED_NOT_COMPUTED`/`decision`/`result_state`/`claim_allowed`/manifest-bound artifact/no-access attestations/CAL arm/evaluator identity），矛盾或缺失 receipt 仍可推进 phase 或产出 final receipt。
- 这整条 U/O/R 表面**只作负证据**，绝不作为 efficacy 证据；不提议第三次修正。
- **Fixed K = baseline / attribution / fallback only，绝不作最终论文路线**。
- 动态 K 是 `required_candidate`（用户优先级），但非已验证、非单独可声称的新颖性。

---

## 3. C 阶段：三条非等价路线

三条路线在机制上非等价：A 复用已有前缀停止控制器；B 复用 frozen 解码器作内层、新建外层 hard-utility/risk 预算策略；C 把已有对偶与 frozen 解码器以"价格校准密度"方式统一，而非前缀停止或离散外层 scorer。

### Route A — 原始动态选帧插件恢复（recovery of existing prefix-marginal-utility controller）

- **问题**：仓库已有可微分的 `PrefixMarginalUtilityBudgetController`（边际效用 vs 冻结价格的前缀停止 + Lagrangian 对偶 `lambda_dual`）与 `budgeted_center_radius_decode`，但整体 `duca-must` 是 `paused/negative`，且依赖 selected-rank 轴与 greedy center-radius。
- **机制**：把"边际 detector utility 不再支付成本即停止"作为逐视频 K 的判据；中心分数经 center-radius 解码得到观测。
- **最强官方基线**：OpenTAD/AdaTAD released dense + clean native uniform K=384/K=192。
- **反直觉预测**：存在视频内预算异质性，使前缀停止点能按内容自适应 K 并保护 high-tIoU。
- **科学趣味性**：边际效用-vs-价格停止是可解释的预算机制，双变量自适应是干净的机制故事。
- **公平基线与隔离消融**：matched uniform fixed-K；消融 λ=0（K→Kmax）与 λ→∞（K→Kmin）；消融 center_scores 用 actionness/transition/motion 替换。
- **最便宜真实视频证伪器**：frozen-detector CAL 上 prefix-marginal-utility vs matched uniform at matched realized cost；high-tIoU 无正 headroom 即 kill。
- **完整训练计划**：6000 次成功 detector update，single-branch，mixed-K exposure。
- **成本**：dense coarse probe + realized heavy frames + center-radius decode。
- **新颖性/先验无效化**：MGSampler（累积运动采样）、AdaFrame（自适应选帧+utility）、AdapTok（多预算 scorer/ILP）、AdaFocusV3/SMART（联合空间时间预算）已覆盖"学预算/按内容选帧"；前缀停止本身是 near-neighbor。**新颖性不可成立为单独命题**。
- **判决：REJECTED as final paper route**。理由：该机制已被 `paused/negative`；selected-rank 时间扭曲与 greedy center-radius 是已诊断失败模式；前缀停止把"哪些帧"（中心分数）与"多少帧"（停止点）耦合在同一 `center_scores` 上，无法隔离"定位质量"与"预算分配"两个因果。**但其 `lambda_dual`（Lagrangian 对偶）是可复用的原语**，被 Route C 继承。

### Route B — 层级 Dynamic DUCA（outer per-video K + inner physical exact-K transport）= RIME 风格

- **问题**：在没有预算异质性证据、decoder-family regret 证据、hard-utility 可预测性证据和 paired-risk 证据之前，任何动态 K 都是叙事。需要在 heavy backbone 前，以 train-only hard counterfactual 估计不同真实帧预算对 cls/reg/high-IoU 与 paired endpoints 的价值，用训练侧冻结的 per-video dual policy 选 K，再由有界 physical exact-K 解码器选真实 RGB 帧，并在 pre-NMS 前逆映射到物理时间。
- **机制**：外层 = 有限 K 集合上的 `hard utility − paired/high-IoU risk − λ · measured_cost`；内层 = 复用 frozen `density_decode.decode_duca_density_positions_v001`（或等价 exact-K 物理帧解码）作为 per-K 内层；坐标 = 已有 `q -> t -> official NMS`。
- **最强官方基线**：OpenTAD/AdaTAD released dense + clean native uniform K=384/K=192。
- **反直觉预测**：(1) 视频间存在预算异质性，使 per-video dynamic Oracle 在 matched realized mean cost 下超过 best fixed K；(2) strict nested physical ladder 的 regret 在视频级可忽略（或不可忽略，从而否决 nested）；(3) paired-boundary risk 独立保护 mAP@0.7/短动作/双端覆盖，超出 actionness/transition 简单信号。
- **科学趣味性**：把"学预算"升级为"hard budget-conditional utility + paired-boundary risk + physical exact-K + batch-invariant realized cost"的组合命题，且外部决策与 detector 内部结构解耦。
- **公平基线与隔离消融**：U-fixed（uniform K=384）、U-same-K（逐视频 K 序列但位置 uniform）、F-bound（fixed-K + admitted utility/risk）、D-shuffle（K 直方图随机 shuffle）、D-no-risk（去 pair-risk）、AdapTok-TAD（direct-transfer baseline）。
- **最便宜真实视频证伪器**：O1 dynamic Oracle headroom（frozen-detector, matched realized cost, per-video vs best fixed）；O1 无 headroom 即 kill 动态主线。
- **完整训练计划**：Phase 0 split/power 冻结 → Phase 1 clean parity → Phase 2 O1/O2/O3/O4 → Phase 3 一个 development seed（6000 updates, mixed-K exposure, development seed 排除）→ Phase 4 三种子/第二 detector/成本。
- **成本**：hard-label/refresh GPU-hours 单列；dense probe + realized frames + 外层 scorer。
- **新颖性/先验无效化**：动态预算/scorer/ILP/inverse-CDF/nested prefix/cheap-to-heavy 均有先验（AdapTok、AdaFrame、MGSampler、AdaFocusV3、SMART、TAPS、Progressive Block Drop）。**可争但未验证的组合** = hard budget-conditional utility + paired endpoints risk + physical exact-K + batch-invariant 平均成本 + 不改 detector 内部结构的高-IoU 保护。
- **判决：RECOMMENDED default，但仅为条件默认**（见 §3 推荐）。

### Route C — realized-cost Lagrangian density acquisition（复用两个真实 DUCA 产物的更强可交付替代）

- **问题**：仓库有两个真实生产产物彼此脱节——`PrefixMarginalUtilityBudgetController.lambda_dual`（batch-invariant 成本对偶）与 frozen `density_decode`（连续密度 → exact-K 物理帧）——且各自带已知失败模式（控制器：selected-rank + greedy center-radius；解码器：fixed-K 无预算分配）。fixed-K learned density 从未稳定超过 matched uniform at high-tIoU；dynamic_must 已 paused。
- **机制**：统一二者。内层 = frozen `density_decode`（连续 per-time 密度 → exact-K 物理帧，保留 physical-time + pre-NMS 逆映射）。外层 = 把"离散 K 前缀停止"替换为**单一冻结全局 Lagrangian 价格 λ 作用于连续密度质量**：位置在累积密度质量跨越 λ 驱动的阈值处被 realized，产生内容自适应 per-video K；全局 dual 更新（复用 `update_dual` 的机制）保证 realized mean cost 与 fixed-K baseline 严格匹配（batch-invariant）。边际 detector utility（train-only hard counterfactual，**非 raw loss magnitude**）供密度梯度；λ 更新是纯成本控制器，与效用解耦。
- **最强官方基线**：OpenTAD/AdaTAD released dense + clean native uniform + learned fixed-K。
- **反直觉预测**：(1) 价格校准密度能在更低 realized mean cost 下恢复 high-IoU；(2) 价格机制使 realized cost 严格 batch-invariant（离散外层 scorer 会漂移）；(3) 密度形状与预算价格可被独立消融，从而把"定位质量"与"预算分配"两个因果分离。
- **科学趣味性**：比前缀停止更干净的可解释机制——"哪些帧"由密度形状决定、"多少帧"由价格决定；是比"再学一个外层 K 分类器"更轻、更可交付的路线。
- **公平基线与隔离消融**：fixed-K uniform / learned fixed-K / K-shuffle；消融 λ 冻结（固定价格）vs 自适应对偶；消融密度源（actionness/transition/motion）。
- **最便宜真实视频证伪器**：frozen-detector CAL 上 price-calibrated density vs exact-uniform at matched realized cost；high-tIoU 无正 headroom 即 kill。
- **完整训练计划**：6000 updates；密度梯度来自 train-only hard counterfactual；λ 由 batch-invariant 成本控制器更新。
- **成本**：dense probe + realized frames + hard-label generation。
- **新颖性/先验无效化**：inverse-CDF/price/DP 单独均非新；可争组合 = price-calibrated physical density + batch-invariant realized cost + hard-utility 密度梯度。**新颖性成立与否取决于 O1 是否证明价格机制（而非外层 scorer）能保持 high-IoU**。
- **判决**：**strongest deliverable alternative / fallback**（复用两个真实生产产物，移除两个已诊断失败模式）。若 Route B 的外层 scorer 过重、或 O2 证明 nested regret 过大、或 O1 证明预算异质性不足以支撑离散外层策略，C 是更轻、更近的可交付退路。

---

## 4. C 阶段推荐（条件推荐，B 仅为默认）

**唯一条件推荐：Route B（层级 Dynamic DUCA / RIME 风格）作为默认论文路线**，但**仅为默认**，且附带如下硬条件：

1. 负证据保留：sealed U/O/R execution-surface 终态失败仅作负证据，不进入 B 的任何 efficacy 论证；Fixed K 仅作 baseline/attribution/fallback。
2. 必须先通过 train-only Oracle 门 O1（动态 headroom）、O2（decoder-family regret，含 strict-nested 是否可行）、O3（`G_rank`，hard utility 可预测）、O4（pair-risk 优于简单信号），再冻结唯一 decoder family 与 K 集合。
3. 所有 `pp`/gap-recovery/risk/cost 阈值在 clean baseline 方差与视频级功效分析后、正式结果前一次性冻结。
4. 必须经过一次**全新、无实现上下文的独立 Pro 攻击**，通过后才进入 Builder→Critic→Evaluator PRE_RUN。
5. 任一 kill gate 触发即收缩：O1 无 headroom → 杀动态主线（回退 fixed-K 归因或 Route C）；O2 nested regret 过大 → 删除 strict nested；O3 `G_rank` 失败 → 删除 utility head；O4 pair-risk 不优于简单信号 → 删除 pair-risk 贡献。

**Rejected 及理由**：

- **Route A 作最终路线**：`paused/negative`；selected-rank 时间扭曲 + greedy center-radius 是已诊断失败模式；前缀停止无法隔离"定位质量"与"预算分配"。仅其 `lambda_dual` 被 Route C 继承为原语。
- **strict nested 无证据预冻结**：两份 takeover 回复在不一致合同上偏向 nesting；在没有 O2 视频级 regret 证据前冻结 nested 会牺牲定位换取叙事。
- **fixed-K 作最终路线**：明确排除（baseline/attribution/fallback only）。
- **sealed U/O/R 表面的第三次修正**：明确排除（负证据，不得修正、复审、交 Evaluator）。

---

## 5. P 阶段：Route B 的未来最小 Builder 包（仅描述，零代码改动）

### 5.1 精确拟改的 patch/config 表面（no code changes now）

1. `opentad/models/duca/`（扩展，非替换）：
   - 新增外层 per-video K dual-policy 模块，复用 `PrefixMarginalUtilityBudgetController.lambda_dual` 的对偶结构，泛化到有限离散 K 集合上的 `utility − risk − λ·cost`；不得改 `density_decode.py` 的 frozen 数学。
   - 内层复用 `decode_duca_density_positions_v001` 作为 per-K exact-K 解码器（逐候选 K 调用）。
   - 新增 train-only hard-utility / paired-boundary / high-IoU risk scorer（当前为 discussed-only，Builder 首次实现）。
2. `opentad/models/selectors/`：新增 deploy-visible `density_logits_valid` reader（v002 命名的 `duca_density_logits[b,t]` reader，当前未实现）。
3. `configs/adatad/thumos/`：新增动态 K arm 配置 + 六臂矩阵配置（见 §5.4）。
4. `tools/bata/`：Oracle/gate 工具。
5. 全新 sealed runner/launcher（吸取终态失败教训的**新设计**，非旧 sealed 表面的第三次修正）：零参数字面 launcher + 闭合 manifest + 无条件 firewall 先于任何 data/checkpoint/output import；校验完整闭合 receipt 语义（`NOT_EXECUTED`/`PRE_RUN_BLOCKED`/`INACCESSIBLE_FORBIDDEN`/`EMBARGOED_NOT_COMPUTED`/`decision`/`result_state`/`claim_allowed`/manifest-bound artifact/no-access attestations/evaluator identity）。

### 5.2 THUMOS14 split（train-population-only）

- 仅官方 training population 内做 video-disjoint FIT / CAL / HOLD 三层，禁止同一视频跨 split；official validation/test 在防火墙后，任何阶段不得装载或参与路由裁决。
- split manifest 在构建前冻结并哈希。

### 5.3 seed policy

- 第一枚种子 = development screening，排除于最终均值/方差/CI。
- 结构冻结后使用未参与开发的预登记种子；无训练侧留出集合时统一 terminal EMA（第 6000 次成功 update）。

### 5.4 首个未来真实视频矩阵（必须包含）

dense、uniform fixed-K、learned fixed-K、dynamic-K、K-shuffle、no-risk、official evaluator、full-stack cost。

### 5.5 official evaluator firewall

- 仅 terminal EMA 单次官方评估；official test 不得选择 epoch/阈值/K/seed；不允许 validation/test GT 进入测试时选择。

### 5.6 N16R4 Slurm 资源命令形式（仅形式，不 launch）

```
sbatch --gpus=1 --cpus-per-task=6 --time=06:00:00 --job-name=<arm> <sbatch_script>
```
sbatch 脚本内：`source /etc/profile` **先于** `set -u` 与所有 `module load`；`module load cuda/11.8 miniforge3/24.11`；`source "$BASE/conda_envs/opentad/bin/activate"`。单卡进程内用 `cuda:0`，不固定物理 GPU、不覆盖 `CUDA_VISIBLE_DEVICES`。

### 5.7 full-stack cost accounting

decode / preprocess / H2D / coarse probe / outer scorer / inner decoder / heavy backbone / head / NMS + hard-label/refresh GPU-hours 单列。

### 5.8 stop rule（kill gate 链）

O1 无 headroom → 杀动态主线；O2 nested regret 过大 → 删 nested；O3 `G_rank` 失败 → 删 utility；O4 pair-risk 不优于简单信号 → 删 pair-risk；development seed 不超过 best fixed at matched cost 或 high-IoU/short-action 退化 → 不补多种子/第二 detector；实际执行 pad 到 Kmax 或完整成本无净省 → 删 efficiency claim。

### 5.9 pilot-then-formal schedule

pilot（真实视频，小规模）→ gate（sealed 表面 + 数据 + 官方基线 + 资源）→ formal（6000 updates）。CPU/synthetic 永不作为 efficacy 证据。

---

## 6. R 阶段：未来 Builder → Critic → Evaluator PRE_RUN 交接与门

仅当 **fresh exact-Project Pro 接受** + **数据门** + **官方基线门** + **资源门** 全部通过后，才进入真实 N16R4 pilot 的角色链。本轮不 dispatch。

- **Builder**：最小包 authoring（零执行）；产出 changed-file/commit identity、静态依赖/pre-import traces、argv/manifest 矩阵、synthetic forbidden-root tests、字面未来 argv/phase order、旧包未触碰 attestation。
- **Critic**：独立、只读、一次终态审查，只返回 `STATIC_PASS` 或 `BLOCKED`；攻击 argument/env 注入、manifest 突变、official-validation 替换、phase-order/receipt 绕过、generic entrypoint reachability、arm 不对称、evaluator/坐标漂移、闭合 receipt 语义。BLOCKED 即 STOP，无 Builder 回应。
- **Evaluator**：一次结构/无数据 intake，只返回 `STRUCTURAL_PASS / PRE_RUN_NOT_READY` 或 `BLOCKED`；两结局均不授权数据/PRE_RUN/GPU/Slurm/training/metrics/claim。
- **门**：数据门（train-population manifest 完整 + 官方 validation/test 不可达）、官方基线门（released dense + clean native uniform 可复现）、资源门（N16R4 Slurm 形式 + 成本口径冻结）、sealed-surface 门（闭合 receipt 语义）。

---

## 7. 待持久化清单

本计划 + 决策日志 + Sources-to-Pro 请求 + `research-wiki/log.md` + `research-wiki/decision_history.md` + `PAPER_PROGRESS.md` + docs/aris 终态 receipt。全部仅科学规划/记忆，不含 browser/lease/permission/queue 噪音，不含实现/性能/成本/claim 结论。
