# ARIS Decision Log — DUCA dynamic-budget CPR (2026-08-14)

- 日期：2026-08-14
- 会话：ARIS DeepSeek V4 Pro Executor（唯一替换；transport-only correction）
- 决策类：仅本地科学规划；无路由执行、无实现、无 PRE_RUN、无 pilot、无训练、无性能/成本/claim

---

## 1. 环境/边界决策（已固化）

- cwd = `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702`；git common-dir = `.git`。
- HEAD = `a6bdc084cc145c80b6b2c68d0a38f0deea3e8518`（`docs: freeze sparsehead evidence-first diagnostics design`）→ **SparseHead 提交，非 DUCA revision**。
- branch = `codex/duca-total60-plugin-cvpr-20260727`；工作树脏（约 275 条 porcelain 条目）。
- **裁决**：本工作树不得称 "clean frozen DUCA revision"。DUCA 证据边界 = 命名的 untracked working-tree 表面 + `research-wiki/`。a6bdc084 / SparseHead Route-T / 任何 SparseHead 结果不得作为 DUCA 证据。

## 2. 代码表面裁决（parent-fidelity）

- `opentad/models/duca/dynamic_budget.py` → `PrefixMarginalUtilityBudgetController` = **真实已有生产 dynamic-budget 实现**（dynamic_must / 边际效用-vs-价格前缀停止 + Lagrangian 对偶 `lambda_dual`），证据类 `paused/negative`。
- `opentad/models/duca/acquisition.py` → `DucaAcquisitionAdapter` + `budgeted_center_radius_decode` = 历史 online/selected-axis DUCA 适配器（消费上述控制器）。
- `opentad/models/duca/density_decode.py` → frozen fixed-K 有界密度分位解码器（v002）；untracked、无生产 reader 消费、`PAPER_PROGRESS` 记为 `PROTOTYPE_ONLY`。**只作 baseline/attribution/fallback 与内层复用原语**。
- `opentad/models/selectors/pc_ot_mras_*` + 7 个 `configs/adatad/thumos/pc_ot_mras_*.py` = SparseHead 基础（tracked），**非 DUCA**。
- `DUCA-RIME` / outer-budget-inner-transport = **仅讨论/规划，无实现、无证据**。

## 3. 科学边界裁决

- sealed U/O/R execution-surface 路线终态失败（`--cfg-options` launch-bypass → 旧包；sealed replacement 未闭合 receipt 语义 → Critic `SEALED_REPLACEMENT_BLOCKED`）。**仅作负证据，绝不作 efficacy 证据，不提议第三次修正**。
- Fixed K = baseline/attribution/fallback only。
- 动态 K = `required_candidate`（用户优先级），非已验证、非单独可声称新颖性。

## 4. C 阶段路由裁决

- **Route A（原始动态选帧插件恢复 / prefix-marginal-utility controller）**：REJECTED as final route（`paused/negative`；selected-rank + greedy center-radius 已诊断失败；前缀停止无法隔离"定位质量"与"预算分配"）。其 `lambda_dual` 被 Route C 继承为原语。
- **Route B（层级 Dynamic DUCA / RIME：outer per-video K + inner physical exact-K transport + hard utility + paired-boundary/high-IoU risk）**：**RECOMMENDED default，仅条件默认**。
- **Route C（realized-cost Lagrangian density acquisition：复用 `lambda_dual` + frozen `density_decode`，价格校准密度）**：strongest deliverable alternative / fallback。

## 5. 条件推荐理由与保留

- B 是默认，因为动态 K 是 required decisive candidate（用户优先级）。
- B 仅为默认：必须先过 O1（动态 headroom）/O2（decoder regret，含 nested 可行）/O3（`G_rank`）/O4（pair-risk 优于简单信号），阈值在 clean 方差 + 视频级功效后冻结，并经过一次全新无上下文独立 Pro 攻击。
- C 作为更轻、更近的可交付退路，复用两个真实生产产物、移除两个已诊断失败模式；若 B 外层 scorer 过重或 O1/O2 表明离散外层策略无据，转 C。
- A 仅作机制原语提供者（`lambda_dual`），不作最终路线。

## 6. 不确定性（如实记录）

- 无任何预算异质性、decoder regret、hard-utility 可预测性、pair-risk 的证据（均 `discussed/no_result`）。
- `density_decode.py` 的 provenance 存在 docstring 自述 "production" vs `PAPER_PROGRESS` 记 `PROTOTYPE_ONLY` 的张力，已如实标注。
- 所有 C/P/R 内容为 planning；无实现、无 PRE_RUN、无运行、无性能/成本/claim。
