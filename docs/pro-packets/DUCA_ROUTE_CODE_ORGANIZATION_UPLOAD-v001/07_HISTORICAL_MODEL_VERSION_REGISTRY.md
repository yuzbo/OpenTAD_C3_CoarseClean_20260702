---
type: implementation_registry
status: active
updated: 2026-07-23
scope: local OpenTAD worktrees and DUCA model lineages
---

# DUCA 模型版本注册表

## 2026-07-23 dd3c97c 稀疏粗扫描插值候选

- Branch: `codex/duca-sparse-probe-interpolation-20260723`.
- Exact commit: `dd3c97cf5ee628c2b0b6f26ce976618e36b7cd45`.
- GitHub: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/dd3c97cf5ee628c2b0b6f26ce976618e36b7cd45`.
- Parent model identity: 继承 `4f81299/9f97f2c` 的 R2Q3、K384/G2、VideoMAE 与
  official-derived AdaTAD/ActionFormer；只新增 d=1/2/3/4 稀疏 probe 计算与 temporal-hidden
  原坐标线性重建，不新增 selector、decoder 或 anchor 旁路特征。
- Static/real evidence: 新 focused `4 passed`；Gate Job `1180556 COMPLETED/0:0`，四档
  spatial/temporal gradient 非零、数值有限、估算 MACs 单调下降。
- Formal experiment: Suite Job `1180557 RUNNING`，root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_sparse_probe_dd3c97c_20260723_011329`；
  四个一 GPU step 已同时进入 P0，之后各自执行 full-model gate 与 official-60。
- Evidence status: `experiment_running`; no terminal official mAP, no paper-ready sparse-probe claim.

## 2026-07-23 9f97f2c 当前 R0-R5 执行身份

- Branch: `codex/duca-boundary-burst-20260722`.
- Exact commit: `9f97f2c7f081b10fbf1f63d0602a621c6b43a780`.
- GitHub: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/9f97f2c7f081b10fbf1f63d0602a621c6b43a780`.
- Model identity: 与 `a00498e` 相同的 boundary-burst selected-axis 模型；只修复共享前置与
  Slurm step 单 GPU 并行，未修改 selector、decoder、loss、VideoMAE 或 detector。
- Formal Jobs: `1180490--1180496`; root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343`.
- Diagnostic child: `4f81299` 仅新增 MS-TCN2/ASFormer/FACT/Video-Mamba-ASFormer P0
  启动入口，Jobs `1180502--1180505`，不改变 DUCA 或提供 TAD mAP。
- Open model risks: selected-rank time distortion、mandatory union 无接纳前 completeability、
  local-RGB-slope detector surrogate、五预算成本 parser 不完整。
- Candidate successor: TTDI 只处于 `designed_pending_terminal_map`；应先单独验证 zero-init
  feature timestamp residual，再决定是否进入 physical-coordinate head。
- Evidence status: `experiment_running`; no terminal official mAP or paper-ready claim.

## 2026-07-22 e49 R0-R5 正式部署版本

- Branch: `codex/duca-boundary-burst-20260722`.
- Exact commit: `e49ef69605e1f98a7217957483f93a8a64bfc348`，已推送且 canonical tree clean。
- GitHub: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/e49ef69605e1f98a7217957483f93a8a64bfc348`.
- Model identity: 离线 TAD fixed-budget pre-backbone acquisition plugin；复用 V8
  coarse/official-ASFormer、transition scorer、global exact-K/max-hole DP、真实 hard RGB
  gather、official-derived AdaTAD/ActionFormer，并增加 Oracle-calibrated bilateral burst、
  protected detector feedback、真实 legal hard-swap alignment 与 TemporalMaxer 第二后端。
- Static evidence: N16R4 clean snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_e49ef69_20260722`；R0-R5
  focused `192 passed`，强制 C3 `23 passed`，compile/bash/HEAD/clean 全过。
- Independent audit: MAX `019f88bf-272f-7373-b702-5b66b142cbdc` 返回
  `GO_TO_SLURM`；不存在影响模型、梯度或实验有效性的开放 blocker。
- Formal deployment root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_20260722_155037`.
- R0--R4 jobs: `1179795 -> 1179796 -> 1179797 -> {1179798,1179799} -> 1179825 -> 1179826`.
- R5 jobs: real TemporalMaxer gate `1179827`；24 个终端训练/评估单元由四个等价
  GPU 批次 `1179861--1179864` 全部执行；9 个成本单元与最终聚合由 `1179865` 执行。
- Site-only scheduling note: N16R4 `AssocMaxSubmitJobLimit` 使逐单元 Job 不可行；六个
  未运行重复 Job `1179828--1179833` 被取消。批次只串行调用原始、哈希绑定的 sbatch，
  不改变配置、种子、预算、后端、checkpoint、训练或评价。
- Evidence status: `experiment_running`; no terminal e49 mAP/cost yet. Historical d9 R0
  internal-holdout raw mAP is diagnostic only and cannot be used as paper result.

## 2026-07-22 exact d9 unified audit candidate

- Branch: `codex/duca-boundary-burst-20260722`.
- Exact commit: `d9fb398578716d278e818745677a92976bcedf2c` (pushed, canonical tree clean).
- Scope: no deployable model, K/G, loss, selector, decoder or official detector
  change. It unifies strict MILP tie resolution with selected-family
  propagation, official-ASFormer source rehash and full-model artifact-content
  reopening.
- Linux evidence: focused DUCA `88 passed`, mandatory C3 `23 passed`,
  pycompile and all production shell syntax checks.
- Real deterministic replay: 124 windows, U/R2Q3/R4Q5/unrestricted, two
  complete solves, byte-identical JSONL SHA-256
  `b49e03c2f4222512cf7752bd3c89bad714868ae69e7c9d05980f9e9f47edd6d7`.
- Audit status: fresh independent MAX
  `019f87bb-d767-7713-825e-92b893e49a98` granted R0-only GO and separately
  held P0/full-model/official-60.
- Experiment status: corrected R0 is authorized but not yet submitted;
  P0/full-model/official-60 remain blocked; no new headroom or terminal mAP.
- Deferred official-only blockers: submit only U+selected G0 instead of four
  sentinel GPU jobs, and bind aggregate terminal pretrain path/hash back to
  the sealed P0/full-model gate identity.
- Active evidence: corrected R0-only Job `1179517`, run root
  `duca_boundary_d9fb398_r0_formal_20260722_112357`; it is running on `g0048`.
  No training job is attached.

## 2026-07-22 boundary-burst exact candidate status

- Deterministic R0 successor integrated locally as
  `c418a951a9b9b7f7f19df785ead8642a4205c804`, sourced from independent worker
  commit `e267e1f9562c91fc0ad9a60382eb829d82d41acd`.
- It preserves the first two MILP optima and adds a strict block-wise
  lexicographic pin; tied and repeated solves now have a unique result.
- This local commit is not yet the next audited deployment candidate because
  the disjoint family/source/artifact propagation patch is still in progress.

- Branch: `codex/duca-boundary-burst-20260722`
- Exact commit: `22555a4e830ce24f9bb516897b1bb7f44b70c188`
- Model family: existing V8 coarse/ASFormer scorer plus Oracle-calibrated
  bilateral boundary burst and the existing global exact-K/max-hole DP; this
  is not a new selector or decoder family.
- Exact correction: the privileged R0 Oracle now solves endpoint centers,
  within-radius quota, bilateral support and exact-K/physical-G jointly. The
  old fixed nearest-Q construction is sealed as false-infeasibility evidence;
  deployable model geometry and K/G are unchanged.
- Static evidence: remote clean snapshot passes solver/Oracle `22`, P0
  reanalysis `9`, runtime/gate/aggregate `54`, required C3 `23`, and
  pycompile/bash/HEAD/clean. The exact old failure sample now returns all four
  R0 families with `ok=true`.
- Audit status: the first fresh MAX
  `019f8743-aed1-7a80-a7d6-552b08491019` was shut down after returning no
  verdict and grants no permission. Replacement no-context MAX
  `019f875f-1668-7e51-bf97-1f565b25e106` returned HOLD for R0-only, P0,
  full-model gate and official-60. Exact commit `22555a4` must not be
  submitted.
- R0 blocker: constraint feasibility and zero-gap optimality were verified,
  but equal MILP optima are not yet resolved by a strict unique lexicographic
  rule. Worker `019f877c-4595-7431-96a0-edff1f7b8251` is allowed to modify only
  the exact solver and its tests to close this reproducibility contract.
- Execution-policy status: bounded worker
  `019f8766-c4db-7e30-8fc8-265d85d83b07` is implementing only the already
  frozen rule that R0's selected projected family is the sole mandatory
  learned P0/gate/R3 family, plus the reviewer's P0 source-hash and full-model
  gate artifact-consumption contracts. No model surface is in scope.
- Experiment status: old R0 Job `1179392` is failed before mAP and invalid as
  headroom evidence; corrected R0 is not yet submitted. P0/official-60 remain
  blocked; no V9, terminal mAP, greater-than-65 or paper-ready claim exists.

## 最终核对结论

我们已经有“固定预算、全局选点、最大间隔受约束、允许跨区域转移预算”的模型实现，
不需要重新设计一套 selector 或 decoder。

最直接的现有实现是 `global_structured_topk`：它在整个时间轴上联合求解 exact-K
和 `max_unselected_hole`，不是每个均匀 cell 固定选一帧。该实现至少从历史
`696f77d` 路线已经存在，并在 selected-axis 正式训练提交 `cb89586` 中作为主策略使用。
它允许边界附近出现相邻选帧，同时要求其他区域不能产生超过上限的空洞，因此具备
跨区域预算重分配能力。

当前真正缺少的不是“又一个模型”，而是一个精确提交同时闭合以下合同：

1. 复用现有全局 `global_structured_topk`，不使用 one-frame-per-cell；
2. 合入已实现的 P0 粗分类训练修复，但不合入 local-cell decoder；
3. 合入两阶段课程训练，并隔离优化器、裁剪、调度器和 EMA 状态；
4. 只让受保护的检测梯度更新 selector scorer，不污染二分类 action head；
5. 使用官方 AdaTAD/ActionFormer selected-axis 路径完成 matched terminal mAP；
6. 通过成本、hard-swap 对齐和无泄漏门禁。

因此后续任务是“现有组件的受控合并与验证”，不是重新发明同一套结构。

## 当前磁盘快照

- `E:\DeskTop\TAD` 下共有 41 个顶层 `OpenTAD*` Git 目录。
- 主仓库登记了 20 个 worktree；另有一个嵌套的 protected-physical 构建树。
- 顶层目录中有 10 个包含 `opentad/models/duca/`，但多数是同一继承链的快照，
  不能按目录数理解成 10 个独立模型。
- 当前协调仓库位于 Spatial-Zoom 分支且已有大量用户/研究记忆改动；它只用于记录，
  不是 DUCA 模型落地面。
- SparseHead、Spatial-Zoom、ChronoTransport 及其审计树保持只读，不能为 DUCA 改动。

## 2026-07-21 全局课程主线冻结

新的隔离实现树固定为
`E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702\.codex_tmp\OpenTAD_DUCA_GlobalCurriculum_20260721`，
分支为 `codex/duca-global-curriculum-20260721`，基线提交固定为 `9442b94`。
该提交已经包含 selected-axis 全局策略、两阶段课程与 P0 合同修复，但位于
`56c2683` local-cell 派生之前，因此不会把 one-frame-per-cell 带回主线。

本轮只允许形成以下 matched 版本，不再新增 selector、decoder 或 detector wrapper：

| 实验臂 | 复用模型 | 唯一变量 | 要回答的问题 |
| --- | --- | --- | --- |
| U | exact-uniform K=384 + 官方 AdaTAD | 不学习选帧 | 同协议基线是否回到可信水平 |
| G0 | P0 初始化 + `global_structured_topk` | 检测梯度关闭 | 粗分类/状态转变监督本身是否有效 |
| G1 | G0 + protected structured transport | 检测梯度只更新 transition scorer | 下游 TAD 梯度是否提高最终 mAP |
| G2 | G1 + 训练期 exact-uniform companion | 对 learned-row 检测梯度曝光做归一化 | 稳定 detector 的同时是否保留学习采样能力 |

G0/G1/G2 必须共享同一个 P0 checkpoint、同一个全局 exact-K/max-gap 可行集、
同一个官方 `ActionFormerHead` 和同一个 60-epoch detector 协议。G2 的 companion
只在训练时存在，推理仍只运行 learned policy；它不能把每格一帧写入 learned
policy 的可行集。现有 companion 未对 learned-row 梯度曝光下降做归一化，这是
本轮唯一允许修改的核心训练合同之一，而不是新选帧模型。

### 精确实现与部署状态

- 唯一主线分支：`codex/duca-global-curriculum-20260721`
- 精确提交：`63e25eb17e523d369f73434ed4d9b6446608861a`
- GitHub：
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-global-curriculum-20260721`
- 本地隔离树：
  `.codex_tmp/OpenTAD_DUCA_GlobalCurriculum_20260721`
- 远端干净快照：
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_global_63e25eb_20260721`
- 远端回归证据：同一 V8 模型线曾通过完整 DUCA focused regression
  `158 passed, 3 skipped`；完整入口修订通过 `18` 个受影响测试；当前门禁分类修订
  通过 `15 passed`、`py_compile`、`bash -n`、精确 HEAD 与 clean-tree 检查。
- 唯一活动正式串行作业：`1178989`，run root：
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_global_63e25eb_serial_20260721_2120`。

提交链 `4c777a6 -> e0397ec -> 2c403a8 -> 6b6363e -> 9138156 -> 63e25eb`
没有新增 selector/decoder。
它只在既有 selected-axis 全局模型上：
修正 G2 learned-row 梯度曝光归一化，增加 U/G0/G1/G2 的匹配配置和证据门禁，
并把旧 scratch/frozen 四臂替换为能够直接回答当前科学问题的四臂。正式 CUDA
全模型门禁与 terminal mAP 尚未产生，因此当前状态只能是 `experiment_running`，
不能声称超过 65 或成为论文最终方法。

## 核心源码继承关系

对 7 个关键 DUCA 树的核心源码做 SHA-256 核对后，得到以下事实：

| 核心文件 | 代码关系 | 结论 |
| --- | --- | --- |
| `dynamic_budget.py` | 7 棵关键树完全同哈希 `5E35DC8000D8` | dynamic budget 没有随近期路线发生新的模型创新，继续冻结 |
| `counterfactual_utility.py` | CellCF 之后各树完全同哈希 `8546C6261C23` | 不应重复实现 counterfactual utility |
| `acquisition.py` | Transition/Allocation 同版；Protected/Selected-axis/Two-stage 同版；LocalResidual 单独派生 | 主要差别是策略配置和后续 local-cell 扩展，不是重新建立 detector wrapper |
| `structured_selection.py` | Selected-axis/Two-stage 同版；LocalResidual 只在其上增加局部归一化；嵌套 physical 树另有物理 DAG 版 | 全局 exact-K 和 hard/soft DP 已经存在 |
| `transition_only.py` | Selected-axis/Two-stage 同版；LocalResidual 只增加 P0/loss 修复 | 粗分类与 transition scorer 不应再次从零编写 |

## 已实现模型族

| 版本 | 精确来源 | 已有能力 | 实验证据 | 当前裁决 |
| --- | --- | --- | --- | --- |
| V0 早期 DUCA/GAS-VT 插件 | `OpenTAD_GASVT_Worktree_20260706`, HEAD `696f77d`；关键历史提交 `7e3a508/70aa069` | coarse source、全局结构化选帧、max-gap、AdaTAD 接入、早期 direct-gradient 骨架 | 有历史训练和大量 contract 测试，但任务表述、几何和协议多次被修订 | 历史参考；不得恢复为主线 |
| V1 transition-only 全局选择 | immutable model `1642f26`，evidence tree `4ce69c8` | 二分类 actionness、状态转变描述、全局 structured exact-K、selected-axis AdaTAD | terminal EMA：uniform `63.8594`，transition-beta0 `64.2755`；单 seed | 有弱正信号，但未超过约 65 的历史强 uniform；诊断基座 |
| V2 CellCF | immutable model `1642f26` | 每个 exact-uniform cell 内选择一帧，detached counterfactual utility | terminal EMA `64.0610`，低于 transition-beta0 `0.2145` | 主方法身份终止；只保留 local-cell 消融 |
| V3 Allocation-Ceiling | `OpenTAD_DUCA_AllocationCeiling_20260720`, `db11aee` | 全局 exact-K/物理最大间隔、跨区 quota 的 privileged/deployable 诊断 | privileged 边界距离有 headroom，但 deploy score 比 uniform 差；冻结 detector loss 也更差；无正式 mAP | `tested` 的必要条件诊断，不是训练模型 |
| V4 Protected physical E2E | nested tree `codex/duca-physical-protected-e2e-20260720`, `ee05f61` | 物理 exact-K hard/soft DAG、protected gradient、P0-P3 工具 | focused `84` tests；正式链未形成 terminal mAP；后续 physical/selected-axis parity 门禁暴露约 24.1% 目标差异 | 组件可复用，物理 detector 表示路线 HOLD |
| V5 selected-axis 全局优化 | source commit `cb89586`，当前诊断树 `OpenTAD_DUCA_UniCompanion_20260721` | `global_structured_topk`、G=2、K=384、direct/homotopy/uni-companion、受保护 detector gradient、官方 ActionFormerHead | terminal EMA：uniform `64.4580`，direct `63.7102`，homotopy `63.0601`，companion `63.6931` | 全部 learned 旧训练臂均低于 uniform；全局主干仍复用，但禁止重跑旧 direct/homotopy/companion 合同 |
| V6 两阶段课程训练 | `OpenTAD_DUCA_TwoStage_20260721`, `6f2ed48` | P0 frontend 预训练；official-60 前 1000 updates 用 exact-uniform 预热 AdaTAD；随后全局策略联合训练 | Job `1178591` 为诊断；审计发现隐藏非零 loss、coarse 梯度归属和优化器状态隔离问题；无有效 terminal mAP | 训练日程可复用，旧合同不能直接作为正式实验 |
| V7 local-residual/local-cell | `OpenTAD_DUCA_LocalResidual_20260721`, `6c56e11` | 修复 P0 loss、padding、GroupNorm、optimizer coverage；U/D/R0/R1 局部 cell 模型 | P0 Job `1178863`；没有 official-60 mAP | P0 修复可复用；local-cell 策略仅为诊断，禁止升级为最终方法 |
| V8 全局课程主线 | `OpenTAD_DUCA_GlobalCurriculum_20260721`, `63e25eb` | 复用 V5 的 `global_structured_topk`，合入 V7 的 P0 修复、V6 的课程结构与 scorer-only protected gradient；U/G0/G1/G2 是同一模型的四个配置臂 | 模型线完整回归 `158 passed, 3 skipped`；EMA 门禁修订的受影响合同 `21 passed`；Job `1178989` 已提交，尚无 terminal mAP | 当前唯一主线，状态 `experiment_running`；在 matched 结果产生前禁止另建 selector 家族 |

历史 uniform `64.352/65.696` 只作为协议不完全匹配的背景锚点，不能填入当前 matched
主表，也不能把“超过 65”写成已经实现的结果。

### V5 部分终局证据更新

Job `1178642` 已完成 `cb89586` 的全部四臂终局评估。Avg-mAP 为：
exact-uniform `64.4579977`，direct-0.25 `63.7101546`，homotopy-0.25
`63.0600746`，homotopy+uniform-companion `63.6930734`。全部 learned 臂
均低于 uniform。这否定了旧 direct/homotopy/companion 训练合同，不是否定
全局可行集，也不能替代 V8 对修复 P0、冻结 coarse 与 scorer-only
protected gradient 的独立检验。V5 已形成终局负证据。

## 为什么出现“反复回跳”

selected-axis 全局模型本来已经允许聚集和跨区换帧，但早期学习出现了两个问题：

- coarse actionness 较弱，epoch-4 诊断 AUROC 约 `0.463`；
- scorer 产生了很强但位置不准的聚集：相邻选择比例从 uniform 的约 `4.4%`
  上升到约 `35%`，同时 radius-1 边界召回下降约 9--10 个点。

随后路线为了防止塌缩，把可行集收缩成 one-frame-per-cell。这样更容易稳定，也更容易
保持覆盖，但它改变了科学问题：从“学习把预算从背景移到边界”退化成“在均匀采样附近
挑左帧或右帧”。因此 local-cell 是优化诊断，不是原始 DUCA 思想的实现。

120-record 的 local/global GT oracle 在当前边界代理指标上接近，并不能推翻上述裁决：
K=384 时 uniform 的 radius-1 recall 已达 `0.9998`，该指标已经饱和；它既不衡量跨区
预算自由度，也不等于最终 TAD mAP。

## 唯一允许的复用地图

下一版若继续实现，只允许从下列现成组件组合，不得新增平行实现：

| 需求 | 唯一首选来源 | 明确排除 |
| --- | --- | --- |
| 全局 exact-K/max-gap decoder | `cb89586` 的 `global_structured_topk` 与同图 hard/soft DP | `local_cell_deformation`、post-hoc hard repair、新 decoder |
| 粗分类与 transition 输入 | `C3CoarseProbeActionnessSource`、官方 ASFormer hidden/delta descriptors | X3D/SlowFast 主线、actionness top-k |
| P0 数值与梯度合同 | `5d17dcb/6c56e11` 的 balanced BCE、graph-free zero loss、detached transition、GroupNorm、optimizer coverage | local-cell feasible set |
| 两阶段日程 | `6f2ed48` 的 frontend P0 + exact-uniform detector warmup + joint stage | 旧隐藏 loss 默认值、共享优化器/裁剪/EMA 污染 |
| detector 接入 | `cb89586` 的 selected-axis GT remap、official AdaTAD backbone、原始 `ActionFormerHead`、预测 inverse-map | 当前失败的 physical/selected-axis parity 假设 |
| detector 到 selector 梯度 | `cb89586/ee05f61` 的 protected structured transport，默认只更新 scorer | detector 梯度进入 action head；未经 P3 的 surrogate |
| 成本与证据工具 | `4ce69c8` 及其修复后的 profiler/evidence contracts | 重新写启动器或只报 FLOPs |

## 强制防重复合同

1. 新改动前必须在本文件登记“现有实现为什么不能直接复用”；没有具体缺口就不得新建类。
2. 禁止再新建 actionness source、exact-K solver、max-gap repair、detector wrapper、
   ActionFormerHead 或成本 profiler 的平行副本。
3. local-cell 结果无论好坏都只能进入 ablation/diagnostic，不自动解锁论文主实验。
4. 下一主线必须从现有 selected-axis 全局模型派生，并只合并 P0/课程训练/梯度保护缺口。
5. 所有状态必须按 `designed -> implemented -> tested -> experiment_running ->
   empirically_supported -> paper_ready` 推进；代码存在不等于性能成立。

## 2026-07-21 19:42 V8 证据合同修订

- 在 19:42 这个历史检查点，活动精确提交更新为
  `e0397ec0bcb917593664ce36efd8105e31d0a302`，分支仍为
  `codex/duca-global-curriculum-20260721`。这是 V8 同一模型的 P0 学习率与
  检查点选择合同修订，不是 V9，也没有新增 selector/decoder。
- 三个 P0 配置只改变 coarse trunk/action head/transition scorer 学习率；共享
  同一 coarse probe、官方 ASFormer temporal、transition scorer、固定三项监督、
  `global_structured_topk` 和后续 U/G0/G1/G2。
- 远端精确快照
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_global_e0397ec_20260721_v2`
  通过 `158 passed, 3 skipped`。旧 Job `1178911` 运行 9 分钟后取消并冻结为
  无效协议诊断；Job `1178927` 是该检查点唯一活动主线，run root 为
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_global_e0397ec_serial_20260721_1939`。
- 当前状态仍为 `experiment_running`。测试和部署不能代替 terminal-EMA mAP，
  “超过 65”仍未证明。

### 19:51 启动器证据修订

`e0397ec` 的模型与 P0 合同保持不变，但从 Job `1178863` 发现质量工具以文件
路径执行时不能解析顶层 `tools` 包。最终活动提交因此更新为
`2c403a853d55057ae772e1b8dcc0c4ebb8cbc0f5`，仅把两处调用改成
`python -m tools.bata...` 并增加回归测试。该提交仍属于 V8，不是新模型版本。
Job `1178927` 零运行时取消；该检查点的唯一活动 Job 为 `1178933`。

### 20:20 完整串行入口修订

进一步逐行审计发现 `2c403a8` 只修复了质量导出/分析工具，两个聚合器仍以
`python tools/bata/...py` 启动，而它们都导入 `tools.bata...`；本地实际执行均复现
`ModuleNotFoundError: No module named 'tools'`。提交 `6b6363e` 只把这两个调用改为
`python -m tools.bata...` 并加入回归测试，仍是同一个 V8 模型。远端精确快照通过
`18 passed` 与完整入口检查。旧 Job `1178933` 在临时审计文件尚位于旧快照时被
调度，因 clean-tree 门禁在 3 秒内失败；它没有模型更新或结果。Job `1178947`
是该检查点的唯一活动主线，后续已被页首当前指针取代。

## 2026-07-21 19:15 最终收口复核

- 再次核对 41 个顶层 OpenTAD Git 目录、20 个登记 worktree 和 2 个嵌套 DUCA
  克隆，共 43 个相关树；没有发现另一套比本表 V8 更新、同时满足全局 exact-K、
  max-gap、跨区域预算转移、P0 修复和受保护检测梯度的实现。
- 该时刻唯一允许继续修改和取主实验结果的实现是干净提交 `4c777a6`；此状态已被
  上方 19:42、19:51 与 20:20 三次同模型合同修订取代。当前精确提交为
  `9138156`，唯一活动作业为 `1178975`，因此模型状态仍为
  `experiment_running`，不能因代码收口而声称性能成立。
- 此后需求若能映射到本表“唯一允许的复用地图”，必须在原文件和原类上修复；只有
  先在本表写出一个可验证、现有实现确实不具备的合同缺口，才允许新建类或实验树。
- `local_cell_deformation`、CellCF 和 local-residual 继续保留用于历史复现和消融，
  但不会再被当作全局 DUCA 主线的候选，也不能因为训练更稳定而替代 V8。

## 2026-07-21 20:00 主线配置冻结证书

这四个实验臂是同一个 V8 模型的配置，不是四套模型：

| 臂 | 配置入口 | 固定策略 | 唯一差异 |
| --- | --- | --- | --- |
| U | `duca_two_stage_exact_uniform_fixed384_official60.py` | exact-uniform K=384 | 不学习选帧，作为匹配控制 |
| G0 | `duca_global_curriculum_g0_no_feedback_fixed384_official60.py` | `global_structured_topk`, K=384, max-hole=2 | 检测梯度关闭 |
| G1 | `duca_global_curriculum_g1_protected_fixed384_official60.py` | 与 G0 完全相同 | 检测梯度只更新 transition scorer |
| G2 | `duca_global_curriculum_g2_uni_companion_fixed384_official60.py` | 与 G0 完全相同 | G1 加训练期 uniform companion 与 learned-row 梯度曝光归一化 |

`tests/test_duca_global_curriculum.py` 已强制 G0/G1/G2 拒绝
`local_cell_deformation`，并核对 K=384、768 点输入、max-hole=2、冻结 P0 coarse
probe、官方同一 `ActionFormerHead`、6000 次 optimizer update 与 epoch-59 主
checkpoint。共享源码中保留 local-cell 函数仅用于历史复现，不能据此把活动模型
解释为 local-cell。任何后续优化只能在上述原类、原 decoder 和原四臂上修改已登记
的训练合同；不得创建同义 selector、decoder、模型族或 worktree。

### 核心文件哈希复核

对 TransitionOnly、AllocationCeiling、ProtectedE2E、UniCompanion、TwoStage、
LocalResidual 和当前 GlobalCurriculum 七棵关键树重新计算 SHA-256：

- `dynamic_budget.py` 七棵树仍完全同哈希 `5E35DC8000D8`，没有“新动态预算模型”；
- 当前全局主线的 `structured_selection.py` 为 `7571639B0230`，与 selected-axis/
  two-stage 全局 DP 同源；
- 当前 `transition_only.py` 为 `86B7D550C8EA`，继承的是 P0 数值与梯度修复；
- 当前 `acquisition.py` / `duca_online_frame_selector.py` 分别为
  `8DE699FD764A` / `A9DB8CA612E3`。源码同时保留历史策略分支，但活动配置与测试
  只允许 `global_structured_topk`。

因此近期 tree 的大部分差异是训练合同、门禁和配置继承，不是多个独立 selector。
哈希不同也不能单独作为“新模型”证据；模型身份必须由活动配置、可行集、梯度归属
和官方检测路径共同判定。

### 现有独立 ASFormer 初始化资产

远端已有同一 64 像素、hidden=96、两层官方 ASFormer 二分类 checkpoint
`probe_reader.pth`，SHA-256 为
`34e4d510441dc711bfc12599ae772f05c372a89d8988529abfbe6b3405f3bbba`；其内部
验证 `AP=0.434866`、`AUROC=0.631541`。严格兼容性探针显示：BatchNorm 版本可
完整加载，当前 GroupNorm P0 会因六个 BN running-stat/batch-counter 键而严格失败。
因此它只能作为 P0 全部候选不收敛后的有界 warm-start 研究资产；禁止静默
`strict=False`、禁止恢复 BatchNorm 的批组成依赖，也禁止把它包装成新 coarse 模型。

## 2026-07-21 20:42 用户指令下的防重写审计锁

用户明确质疑路线反复退回 local-cell 后，再次扫描磁盘：仍为 41 个顶层
`OpenTAD*` Git tree 加两个嵌套 DUCA 构建 clone，共 43 个相关 OpenTAD tree。
其中 12 个当前含 `opentad/models/duca/`，包括脏的协调仓库和两个嵌套 clone；
目录数量不能解释为模型数量。

对八条关键 DUCA 继承线的六个核心文件重新计算哈希，结果再次证明它们主要是组件
继承，而非独立模型：

- 八棵树共享 `dynamic_budget.py@5E35DC8000D8` 与
  `counterfactual_utility.py@8546C6261C23`；
- V8 复用 selected-axis/two-stage 的
  `structured_selection.py@7571639B0230`，其中已经实现全局 exact-K、max-gap
  与跨区域预算转移；
- V8 复用 local-residual 继承线修好的
  `transition_only.py@86B7D550C8EA`，但活动配置不启用其 local-cell 可行集；
- V8 的 `acquisition.py@8DE699FD764A` 与 selector
  `A9DB8CA612E3` 合并的是既有 P0、课程训练和受保护梯度合同。哈希不同不代表
  重新发明了一套 selector。

“反复回跳”的根因现正式归类为研究控制错误：早期全局策略出现错位聚集后，路线以
one-frame-per-cell 收缩可行集换取稳定性，却改变了原始科学问题。以后弱结果只能在
保留 `global_structured_topk` 的前提下诊断 coarse 证据、scorer 优化、梯度归属和
terminal mAP，不得再通过 local-cell 改写可行集。

本次审计同时发现 Job `1178947` 只暴露了门禁分类器错误：门禁把官方 ASFormer
注意力层内部的 `conv_out` 误判为二分类 action head；真实 ActionFormer optimizer
分组本来正确。精确提交 `91381568637f6358bdec67e3d8400d70869f1dd6` 统一了门禁
拓扑判定并增加回归测试，没有改变模型结构或实验臂。当前唯一实验为 Job `1178975`，
远端干净快照为
`/data/run01/sczc063/yuzibo/projects/opentad_duca_global_9138156_20260721`。

本注册表现为强制查重入口。未来只有先指出一个 V0-V8 全部缺失、可测试的具体合同，
才允许新建类、decoder、配置族或 tree。

## 2026-07-21 21:20 全树最终复核与 V8 证据门禁修订

再次实际枚举得到 `43` 个相关 Git tree：`41` 个顶层 OpenTAD 目录和 `2` 个
`.codex_tmp` 隔离 clone；其中恰有 `12` 个包含 `opentad/models/duca/`。核心文件
SHA-256 前 12 位如下。该表用于识别继承关系，不能把一个目录或一个不同哈希自动
解释成新模型。

| Tree | acquisition | structured | transition | frame selector | global/local 配置数 |
| --- | --- | --- | --- | --- | ---: |
| 协调仓库 | `5BF37726A359` | `-` | `-` | `E643CB24C365` | `0/0` |
| AllocationCeiling | `CC5D9A3E6C10` | `616DD72807F4` | `3686C099C3DF` | `A71F431FD69E` | `4/2` |
| LocalResidual | `1E5AA269EC11` | `7E84443F86C4` | `86B7D550C8EA` | `38354500C82D` | `5/4` |
| ProtectedE2E | `AA908ED2409A` | `616DD72807F4` | `300C519ECC4E` | `978C9B7A8E1F` | `5/2` |
| TransitionOnly/CellCF | `CC5D9A3E6C10` | `616DD72807F4` | `3686C099C3DF` | `A71F431FD69E` | `4/2` |
| TwoStage | `AA908ED2409A` | `7571639B0230` | `77A5D39C845E` | `02B767B11D5D` | `5/2` |
| UniCompanion | `AA908ED2409A` | `7571639B0230` | `77A5D39C845E` | `B06639A0AB2B` | `5/2` |
| GASVT CostAudit | `2E9B1F33499D` | `3113996482F9` | `-` | `F52967C5AA0A` | `2/0` |
| GASVT/PhysTime v0 | `2E9B1F33499D` | `3113996482F9` | `-` | `665E349B590A` | `2/0` |
| PhysTime DeployFix | `2E9B1F33499D` | `3113996482F9` | `-` | `665E349B590A` | `2/0` |
| GlobalCurriculum V8 | `8DE699FD764A` | `7571639B0230` | `86B7D550C8EA` | `A9DB8CA612E3` | `8/2` |
| ProtectedE2E physical clone | `AA908ED2409A` | `5FA090F9526D` | `77A5D39C845E` | `978C9B7A8E1F` | `5/2` |

Job `1178975` 随后真实执行了一次 P0 更新：actionness 与 transition/boundary
损失有限，coarse probe 和 transition scorer 均有非零梯度并更新，detector 路径为
`skipped`。失败只发生在 EMA 门禁：旧检查只观察每组第一个代表参数，`0.001 ×`
单步变化在该 FP32 参数上舍入为零。提交
`63e25eb17e523d369f73434ed4d9b6446608861a` 改为检查整个参数组，同时保留代表
参数变化作诊断；不改变模型、损失、优化器、selector、decoder 或 U/G0/G1/G2。
远端一次性副本的受影响合同为 `21 passed`，`py_compile` 与 `git diff --check`
通过。新精确快照和 Job 分别为
`opentad_duca_global_63e25eb_20260721` / `1178989`。这仍是 V8，不登记 V9。

## 2026-07-21 边界中心与左右选帧数量实现审计

### 总结

已登记的 V0--V8、PAction/GAS-VT、move25/move50、learned-context-radius 与
Oracle 代码中，存在多个可复用零件，但**没有一个版本同时实现**以下完整合同：

1. 从 deploy-visible 粗分类状态变化中预测每个动作起止边界的 transition center；
2. 对每个 center 显式决定左侧与右侧应分配的多帧数量或等价的有界密度轮廓；
3. 让边界微簇达到预注册配额后奖励饱和，并对相邻/重叠端点去重；
4. 将剩余 exact-K 预算跨区域分配，同时保持最大空洞和全局上下文；
5. 以当前 V8 的 official AdaTAD selected-axis 路径完成训练与推理同构的联合训练。

因此这是一个已经明确、但尚未实现的合同缺口，不是发现了另一套可直接恢复的旧模型。

| 历史实现 | 已经具备 | 明确缺失 | 裁决 |
| --- | --- | --- | --- |
| GT-boundary Oracle | 对每个 GT 起点/终点选择中心及 `±2`，再均匀补预算 | 依赖 train/val/test GT；不是可学习 deploy policy | 只用于揭示 Oracle 式边界微簇目标 |
| `legacy_center_radius` / MUST | `center_head`、单个对称 `radius_head`、`budgeted_center_radius_decode` | center 不绑定独立端点；没有左右独立数量；预算截断可造成单侧选择 | 历史诊断组件，不是当前主线 |
| detector-aware learned context radius | 局部峰值、可学习半径、对称 score dilation | 最终仍做全局 top-k；无每端点左右配额；依赖 Stage2/ledger teacher 语义 | 可复用 radius/peak 代码，不恢复旧多阶段路线 |
| GAS-VT `boundary_bracket_loss` | 对 GT 边界检查左、右各至少命中一次 | 二值命中而非多帧数量；无 center object；严格 ledger 路线 | 只复用 bracket 语义与诊断 |
| move25/move50 lattice + adaptive radius | 均匀骨架、局部替换、以 center 为中心的对称候选区间 | center 可能偏离边界；没有左右数量监督；膨胀和 repair 可掩盖坏分数 | 几何诊断/工程对照 |
| CellCF/local-cell | 每个均匀 cell 内稳定选择 | cell center 不是动作边界，且禁止跨 cell 聚集预算 | 与目标不一致，仅保留消融 |
| V8 `global_structured_topk` | 全局 exact-K、G=2、跨区预算转移、允许相邻聚集、official AdaTAD | 没有显式 center/左右数量；active Gaussian-mass loss 不区分中心、双侧和配额饱和 | 唯一允许承载后续修复的主干 |
| bracket/endpoint 分析器 | 左右命中、端点距离、半径召回等统计 | 只评估，不参与 deploy forward 或 active loss | 直接复用为门禁指标 |

ActionFormer/TriDet 中的 left/right regression 或 `center_sample_radius` 属于 TAD
检测头的 proposal/assigner 语义，不是 pre-backbone selector 的左右选帧数量，禁止混淆。

### 唯一后续设计合同

后续不得新建同义 selector。只有在冻结 Job `1178989` 形成终局证据并按停止条件允许
修订后，才可在 V8 现有 transition scorer、global structured DP 和 official AdaTAD
路径上加入一个单变量的 **oracle-calibrated boundary-burst allocation** 合同：

- center 来自粗分类 hidden、`p_action`、`delta_p_action` 和 uncertainty 的间接状态转变证据，
  不是推理期 GT 或另建 direct start/end detector head；
- 每个 center 输出或隐式决定 `q_left/q_center/q_right`，并以 train-only GT 端点监督
  聚集中心、左右支撑和有界局部密度；具体参数化必须先做 headroom 与可辨识性审计；
- 每个端点的收益在目标配额后饱和，近邻端点共享位置时执行确定性的去重/归属；
- 剩余预算由现有全局 exact-K/max-hole DP 分配，不能回退 one-frame-per-cell；
- `radius=0` unique event 只能作为 center anchor，不能单独构成最终目标；
- detector feedback 仍只能通过通过真实 hard-swap alignment 的 protected bridge 更新
  scorer/cluster-allocation 参数，不能污染 coarse action head/trunk。

当前状态：版本审计为 `verified_static_across_registered_versions`；上述合同为
`designed_not_implemented`。它不改变 V8 身份、不创建 V9，也不修改正在运行的 Job
`1178989`。

## 2026-07-22 最终产物与唯一改进路线冻结

最终论文交付被固定为一个 **offline-TAD pre-backbone acquisition plugin**：低成本
coarse/ASFormer 分支在完整窗口上产生动作状态证据，现有 transition scorer 的有界
扩展产生可部署 transition center 与左右边界微簇，现有 global structured DP 完成
exact-K 与剩余全局预算分配，hard selected observations 再进入 official-derived
AdaTAD/ActionFormer。它不是 Online TAD，不是新 detector，也不是三套独立部署模型。

唯一允许的结构增量是 V8 scorer 内的 bounded burst-profile 语义；不得新建同义
selector、exact-K decoder、local-cell feasible set 或 ledger 路线。profile 必须表达
`q_left/q_center/q_right`、配额饱和、公平端点分配和重叠去重。具体半径、配额及
max-hole 必须由 train-split Oracle reachability 冻结，不得根据 test mAP 选择。

完整证据顺序固定为：Oracle/KG 可达性 -> 数学与梯度合同 -> P0 单变量 objective
对照 -> matched U/G0 terminal-EMA -> real hard-swap alignment -> G1/G2 -> 三种子、
预算曲线、第二 detector 与完整成本。任何代理选帧指标只能作为门禁；论文终局由
official TAD mAP 与完整实测成本共同裁决。

当前仍不登记 V9。状态为 `canonical_successor_designed_not_implemented`；只有现有
V8 终局和上述前置门禁允许后，实际代码提交才获得新的实现身份。

## 2026-07-22 EU-CRR 版本身份裁决

外部 Pro 提议的 `Exact-Uniform Coarse Residual Reuse` 尚未实现、测试或部署，不登记
V9。它只是在 exact-uniform K384 下，把 selected frozen coarse hidden 通过零初始化
residual gate 加到 post-VideoMAE/pre-projection feature 的条件性诊断。

该诊断不修改 positions，也不实现 transition center、左右微簇、配额饱和或端点公平，
故不能替代上述 canonical successor。状态固定为
`discussed_conditional_diagnostic_not_authorized`。若未来 U1 相对 U0 通过，必须另行裁决
最终产物是否从 strict pre-backbone acquisition 扩展为 acquisition-and-fusion adapter；
若失败，只注销该 fusion 假设，不注销 G23。

## 2026-07-22 canonical final model contract

唯一权威合同已收束到 `research-wiki/duca_final_model_contract.md`。最终交付是
fixed-budget offline-TAD pre-backbone acquisition plugin：cheap coarse action state ->
indirect transition center -> Oracle-calibrated bilateral boundary burst -> existing global
exact-K/max-hole DP -> hard RGB -> official-derived AdaTAD/ActionFormer。

该合同明确 coarse action head 只接受 binary action supervision；train-only endpoint
监督只更新 scorer/burst；official stage 冻结 coarse，只有通过 real legal hard-swap
alignment 的 detector surrogate 才可更新 scorer/burst。主方法 K384，首个效率扩展
K256，dynamic budget 与 EU-CRR 不进入主线。

这仍是 `canonical_successor_designed_not_implemented`，不是 V9。新版本只在现有 V8
scorer/DP 上闭合 G23、通过 R0--R3 并获得一个 exact implementation commit 后登记。
## 2026-07-22 论文闭环位置

- 当前执行位置：V8/P0 诊断封存，exact commit `63e25eb`，Job `1178989`。
- 第一 P0 候选完成但机制门槛失败：coarse action evidence 改善，旧 transition scorer
  仍弱于 simple delta，边界分配不胜均匀采样；第二候选仍在运行。
- 下一可登记版本不是另一个 selector 名称。只有 R0 证明 projected Oracle 在冻结的
  K/G/radius/quota 下有 detector-mAP headroom，且 R1 通过后，才允许把现有 V8 scorer/DP
  的 boundary-burst 有界扩展登记为 successor implemented。
- 论文实验顺序与主表定义以 `research-wiki/duca_final_model_contract.md` 第 6.1--6.3 节为准。
- 不再需要方向级 Pro 讨论；只保留 R0、R1、R4 三个有边界的审查点。

## 2026-07-22 boundary-burst candidate identity

- `fdf25f5d08bc0bf9b550e059228ce1d6ac587499` 是当前 canonical successor
  **implementation candidate**，不是经验证的 V9。它只扩展 V8 scorer 的 burst
  profile、训练目标、质量统计与实验 DAG；沿用原 global structured decoder、
  selected-axis wrapper 和 official-derived AdaTAD/ActionFormer。
- `4a07a2a` 是首个实现提交；`fdf25f5` 修复 P0 门禁把短视频/尾窗的
  `min(K, valid_len)` 错判为“平均 selected_count 必须等于 384”的问题，模型数学不变。
- 候选明确实现 exact endpoint anchor、双侧支持、R2/Q3 与 R4/Q5 配额、overlap-aware
  saturation、worst-endpoint fairness 和 exact-uniform residual context；推理无 GT。
- 候选现处 `hold_fix_required_after_independent_max`。独立 MAX 已确认 global DP、offset
  gradient、selected-axis mapping、no-leak 与 official detector 无阻断，同时要求修复 P0/R0/
  artifact 九项合同。只有修复后新精确提交通过二次独立审计、远端 Linux/CUDA 门禁及
  R0--R3 结果闭环，才允许登记新的经验版本身份。
- 任何后续修复必须在同一分支就地完成；不得另建同义 selector、decoder、tree，
  不得修改 SparseHead、Spatial-Zoom、ChronoTransport 或回退 local-cell。

## 2026-07-22 boundary-burst exact repair candidate

- identity: `DUCA-boundary-burst-exact-candidate`
- branch/commit: `codex/duca-boundary-burst-20260722@899630a5ef4927e78ef4ca6b8cc51fdf754056da`
- parent route: V8 `global_structured_topk` and official selected-axis AdaTAD/ActionFormer
- delta: center-conditioned quota-limited offset support; R2Q3/R4Q5 bilateral losses and
  diagnostics; validity-aware endpoints; per-sample K/G evidence; R0/P0/gate/four-arm DAG
- unchanged: global exact-K/max-hole decoder, hard original-time RGB gather, official detector,
  fixed K384 primary contract, coarse-gradient ownership, G1/G2 alignment prohibition
- evidence: Linux `136 passed, 3 skipped`; required C3 `23 passed`; compile/shell/clean-tree pass
- status: `implemented_exact_candidate_linux_tested_under_independent_max_audit`
- not yet: CUDA gate, R0 headroom, P0 winner, terminal mAP, >65, V9, empirical or paper support

## 2026-07-22 second MAX audit HOLD

- `899630a` 的第二轮独立 MAX 结论为 `HOLD_FIX_REQUIRED`。这不否定 boundary-burst
  模型方向，但否定其当前提交直接部署正式 DAG 的资格。
- 未通过项限定为 runtime binder/gate config mapping、pooled crop endpoint validity、
  split/checkpoint/manifest/dependency hash 与指标源文件 provenance；模型主体、全局 DP、
  quota support、selected-axis/no-leak/official detector 未发现新的结构阻断。
- 当前身份降为 `implemented_candidate_hold_fix_required`。只有新 exact commit 通过
  Linux、全新独立 MAX 与 real CUDA gate 后，才进入 R0；R0/P0/R3 未出结果前不登记 V9。

## 2026-07-22 aa3352e exact re-audit candidate

- branch/commit: `codex/duca-boundary-burst-20260722@aa3352ecf803c81d007a62ed5398667d9551684b`
- delta from `899630a`: production four-arm runtime binding/gate mapping；pooled crop-valid
  metrics；R0/P0/aggregate source-of-truth metrics 与 split/checkpoint/pretrain/upstream seals。
- unchanged: boundary-burst 模型数学、R2Q3/R4Q5 quota、global exact-K/G DP、official
  selected-axis detector、K384 主合同与 G1/G2 禁止条件。
- evidence: remote DUCA `139 passed, 3 skipped`；required C3 `23 passed`；compile/bash/
  exact HEAD/clean tree pass。
- status: `implemented_exact_candidate_under_independent_reaudit`；无 CUDA/R0/P0/mAP/V9。

## 2026-07-22 86f7663 terminal-audit contract candidate

- branch/commit: `codex/duca-boundary-burst-20260722@86f7663a94d628eace316d17e31db7043f731f75`
- predecessor evidence repairs: `f629ad7` froze AdaTAD pretrain identity and locked G1/G2;
  `7b9ad0b` sealed terminal checkpoint/evaluation/prediction/aggregate identity.
- final delta: selected-axis production `build_training_audit()` now persists the same
  `formal_protocol` and `official60` profile required by terminal validation; success fixtures
  use the real builder; aggregate independently checks those fields.
- unchanged: coarse/transition/burst model, R2Q3/R4Q5 objective, global exact-K/max-hole DP,
  K384/G2, selected-axis mapping, official-derived AdaTAD/ActionFormer and R0--R5 plan.
- evidence: local affected DUCA `60 passed, 1 skipped`, C3/update evidence `29 passed`;
  clean Linux snapshot DUCA `64 passed`, C3/update evidence `29 passed`, compile/bash/HEAD/clean.
- status: `implemented_exact_candidate_under_final_independent_reaudit`.
- not yet: real CUDA gate, R0 headroom, P0 winner, terminal mAP, >65, V9, empirical support or
  paper readiness.

## 2026-07-22 86f7663 independent-review verdict

- reviewer: no-context MAX `019f86a6-fe1b-7921-b576-8d9cd3d4c8ac`
- verdict: `HOLD_FIX_REQUIRED`; status downgraded from final re-audit candidate to
  `implemented_candidate_hold_fix_required`.
- model identity remains unchanged and not rejected: offline pre-backbone acquisition plugin,
  binary coarse action evidence, transition-centered bilateral burst, global exact-K/max-hole DP,
  selected-axis official-derived AdaTAD.
- exact blockers: complete/recomputed/bootstrap-bound R0 Oracle evidence; same-feasible-space
  simple-delta baseline and stop rule; crop-valid diagnostics, atomic submission journal and a
  no-mock official-evaluator integration test.
- no new version number is registered. A successor commit remains the same boundary-burst
  candidate until R0--R3 empirical gates pass; there is no V9 and no paper result.

## 2026-07-22 06:54 bounded R0 evidence successor

- exact candidate: `codex/duca-boundary-burst-20260722@4ec3e078a3aad834ffe504d74d414bf7e2b6fad3`;
- model identity is unchanged: the existing binary coarse/official-ASFormer evidence,
  transition-centered bilateral burst scorer, global exact-K/max-hole DP, selected-axis mapping
  and official-derived AdaTAD remain the only implementation;
- evidence-only additions: unrestricted exact-K Oracle, projected-family official mAP bootstrap,
  complete consumer recomputation, same-feasible simple-delta stop rule, crop-valid diagnostics,
  no-mock evaluator integration and atomic Slurm submission journal;
- status: `implemented_local_tested_pending_linux_and_independent_max`;
- no V9 is registered and no CUDA/R0/P0/official-60 or terminal mAP exists.
- clean Linux evidence: affected DUCA `109 passed`, mandatory C3/ASFormer `23 passed`,
  pycompile/bash/exact-HEAD/clean-tree passed; status advances only to
  `linux_tested_pending_independent_max`.
- 07:15 launch-policy audit: R0 emits a unique projected-family decision, but the current P0/gate/
  official-60 chain still makes all Gaussian/R2Q3/R4Q5 diagnostics mandatory. This is a pending
  evidence-routing blocker, not a new model version and not a reason to create another selector.
  The canonical R3 anchor remains matched U versus the R0-selected G0 first.
- `4ec3e07` was independently held because its R0 launcher reversed the consumer-oriented block
  list. Exact successor `f90595d8620e42e8e3d74722f2ab48126c6b65f2` fixes only this production
  evidence path and adds semantic split coverage; model identity is unchanged. Remote evidence is
  `168 passed, 2 skipped` plus mandatory C3 `23 passed` and no-submit precheck. Status remains
  `linux_tested_pending_independent_max`; this does not register V9.
- Second MAX stage verdict for `f90595d`: R0 authorized; P0 and official-60 held for evidence-chain
  closure, not for a model failure. R0-only Job `1179392` is running from the exact commit; no
  downstream jobs were submitted. Version status is `r0_experiment_running_p0_hold_fix_required`,
  still not V9 or empirically supported.

## 2026-07-22 R0-R5 production implementation identity

- Production model/evidence commit: `e49ef69605e1f98a7217957483f93a8a64bfc348`.
- Status: `implemented_linux_tested_max_reviewed_full_r0_r5_slurm_deployed`.
- Implemented evidence surface: R0 privileged reachability, P0 coarse/transition/burst training,
  real ActionFormer full-model gate, matched U/G0, legal hard-swap signed utility R4, real
  TemporalMaxer second backend, 24-cell three-seed/two-budget matrix, and nine-profile full cost.
- Empirical status remains `experiment_running`: no P0 winner, terminal learned mAP, cross-backend
  result, cost result, >65 result or paper-ready claim exists yet.
- `9ed10139317c4196072d471ced883eb1dfc31703` is a statistics-execution successor only. It preserves
  the exact model, R0 families, RNG sample sequence, evaluator, confidence rule, R1--R5 configs and
  output schema while parallelizing bootstrap evaluation. It is not V9 and must not be described as
  a selector/training/detector revision.

## 2026-07-22 independent official-mAP execution commit

- Exact commit: `2bc6ca6fcf34f3e980437b5b830cabeef0de63c0`.
- Model identity is unchanged from `e49ef696`: no selector, decoder,
  ActionFormer, SparseHead, Spatial-Zoom or ChronoTransport modification.
- Delta: executable metric audit plus four self-contained official-60 jobs with
  no inter-job R0/bootstrap dependencies.
- Evidence: local/remote focused `49 passed`, Python compile, remote Bash syntax,
  no-submit manifest, and independent MAX `GO` after correcting R5 from
  pre-certified to runtime-unverified.
- Remote clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_2bc6ca6_v2_20260722`.
- Formal root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_independent_2bc6ca6_formal_20260722_1755`.
- Jobs: U `1180075`, Gaussian G0 `1180076`, R2Q3 G0 `1180077`, R4Q5 G0
  `1180078`; all were RUNNING with `Dependency=(null)`.
- Status: `official_comparable_four_arm_experiment_running`; no terminal mAP,
  >65 result, empirical support, V9 or paper-ready claim yet.
- GitHub push was blocked by destination-privacy safety review. The remote compute
  snapshot imported a verified Git bundle and preserves the same exact SHA.

## 2026-07-22 production-gate compatibility successors

- `1d5350ea054877101f133bb4cf54a90f7beef560` fixes only production ASFormer
  provenance lookup (`probe` first, legacy fixture fallback).
- `320a8a8513995964241a338dde90472559a7c4e1` changes the boundary-validity
  consumption audit from Python object identity to exact tensor-value equality.
- `8d85929ea04dc40f1eb0c3cc806061ce3b071d3f` makes that value check device
  agnostic by comparing detached CPU values. Local and remote focused evidence
  is `28 passed`; bundle SHA-256 is
  `5bf3f201fc708566e308cb9d12c4158ab3bb316cfac3091505fee464178fc833`.
- These are audit-gate compatibility fixes only. Model identity remains
  `e49ef696`; no selector, decoder, detector, loss, training schedule or metric
  changed, so no new DUCA model version is registered.
- Current official-comparable queue: U/Gaussian/R2Q3/R4Q5 Jobs
  `1180111/1180112/1180113/1180114`, all `Dependency=(null)`. Status remains
  `official_comparable_four_arm_experiment_running`; no terminal mAP or >65 evidence exists.

## 2026-07-22 cd68d89 final R0-R5 candidate

- Exact identity: `codex/duca-boundary-burst-20260722@cd68d89dcc0854baa3c0107607086e801509b552`.
- Model delta from the reviewed candidate: hard/global mandatory burst is separated from soft/local bilateral utility; R2Q3 routing is preregistered instead of selected by contaminated R0; cost pairing is ActionFormer-only; a PyTorch 2.0 CUDA row-reset incompatibility is repaired with equivalent `index_fill_` semantics.
- Deployment: Jobs `1180336--1180340` are five independent R0--R5 bundles; all five were concurrently `RUNNING` at `2026-07-22 21:04 +08:00`. `1180341` is the R5-only aggregate and remains dependency-pending. See `experiments/duca-r0-r5-cd68d89-parallel.md`.
- Status: `implemented_and_slurm_deployed_experiment_running`. This is still a candidate, not V9, not empirically supported, and not paper ready until terminal official-validation EMA mAP and paired costs exist.
- Superseded `2645e68` Jobs `1180326--1180331` contain no mAP and are runtime-failure evidence only.

## 2026-07-22 official AdaTAD parity audit

- Audited DUCA identity: `codex/duca-boundary-burst-20260722@a00498e15d69294f78d0abeadfb47bc456db0b0e`.
- Official reference: `sming256/OpenTAD@1aa8ca4ac5e846b1e8ff69298dd6607121a01589`.
- Byte-identical official components: the AdaTAD base config, `ActionFormerHead`,
  `ActionFormerProj`, focal/IoU losses, NMS implementation/config and loading transform.
- The full detector is deliberately not source-identical. DUCA extends the
  `ActionFormer`/`SingleStageDetector` wrappers, selects 384 observations from a
  dense 768-frame window, remaps GT to selected-axis indices, and inverse-maps
  predictions to original time before the unchanged NMS implementation.
- Active detector protocol also differs from the official dense config:
  `768 -> 384` VideoMAE/projection length, 48 -> 24 chunks, `with_cp=False`,
  `static_graph=False`, `find_unused_parameters=True`, and a matched 60-epoch
  scheduler horizon. These are disclosed plugin/training adaptations, not a new
  ActionFormer head.
- Paper wording is therefore fixed to **official-derived AdaTAD/ActionFormer
  backend with unchanged head/loss/NMS**, never **the complete official AdaTAD
  implementation is unchanged**. The irregular selected axis remains a material
  method risk because VideoMAE receives packed observations without explicit
  true-time interval encoding.
- Verification: current boundary-burst config/full-model focused tests `24 passed`.
  Existing gates establish contract consistency, not numerical parity with the
  official dense-768 forward graph.

## 2026-07-27 current-course parity correction and mainline recovery

- The 2026-07-22 parity entry applies to audited identity `a00498e`; it must not
  be projected unchanged onto the later K=384 course identity
  `42dba3f90b37243e7965d18b6707e88e81bf7109`.
- At `42dba3f9`, nominal head configuration, projection, cls/reg objectives and
  NMS remain official-derived, but both active `ActionFormer` and
  `AnchorFreeHead` source files are extended and are not byte-identical to
  upstream. Optional physical-grid branches may be disabled in the selected-axis
  course, but this does not establish execution equivalence.
- Consequently, K=384 exact-uniform `64.49%` remains a DUCA-wrapper control,
  not a clean native official half-rate baseline. The clean released-weight
  dense, native K=384 uniform and native K=192 uniform identities are not yet
  registered because their experiments have not been implemented and closed.
- The historical `duca_final_model_contract.md` is no longer canonical for the
  paper mainline. The active design contract is
  `duca_prebackbone_plugin_and_baseline_recovery_contract.md`; its bounded
  detector-agnostic transport model is `designed`, not implemented, tested or
  empirically supported.
