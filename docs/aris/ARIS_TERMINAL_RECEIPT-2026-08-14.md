# ARIS Terminal Receipt — DUCA dynamic-budget CPR (2026-08-14)

- 日期：2026-08-14T15:45:00+08:00（截止 2026-08-14T16:15:00+08:00）
- 类型：终态 receipt（仅科学规划/记忆；无实现、无 PRE_RUN、无 pilot、无训练、无性能/成本/claim）

## 必填字段

- **ARIS session**：`j-ble6ep`（`aris_cpr` job；state.json 记录 `aris_cpr.job_id=j-ble6ep`，`aris_continuation.job_id=j-vckhty` 为前一轮）。
- **CPR stage**：`C/P/R_MATERIAL_PREPARATION`（transport-only correction，非科学重试）。
- **exact candidate routes**：
  - A = 原始动态选帧插件恢复（复用已有 `PrefixMarginalUtilityBudgetController` 前缀停止 + `budgeted_center_radius_decode`）。
  - B = 层级 Dynamic DUCA（outer per-video/window K + inner physical exact-K transport（复用 frozen `density_decode`）+ train-only hard detector utility + paired-boundary/high-IoU risk）。
  - C = realized-cost Lagrangian density acquisition（复用 `lambda_dual` + frozen `density_decode`，价格校准密度）。
- **recommendation / caveats**：**B 为条件默认**（动态 K 是 required decisive candidate），但仅为默认：须先过 O1（动态 headroom）/O2（decoder regret，含 strict nested 可行）/O3（`G_rank`）/O4（pair-risk 优于简单信号），阈值在 clean 方差 + 视频级功效后一次性冻结，并经过一次全新无上下文独立 Pro 攻击；任一 kill gate 触发即收缩（回退 fixed-K 归因或 Route C）。sealed U/O/R 终态失败仅作负证据。
- **rejected routes**：A 作最终路线（`paused/negative`；selected-rank + greedy center-radius 已诊断失败；`lambda_dual` 留作 Route C 原语）；strict nested 无证据预冻结；fixed-K 作最终路线（baseline/attribution/fallback only）；sealed U/O/R 表面第三次修正（负证据，不得修正/复审/交 Evaluator）。
- **evidence classes**：`BLOCKED_PRE_RESULT`；`ARIS_CPR_MATERIAL_PREPARATION_NO_EXECUTION`。旧包 `657678946` / sealed replacement `6515ebf5` = `STATIC_READ_ONLY_TERMINAL_REVIEW`（负证据）。`PrefixMarginalUtilityBudgetController` = 生产 dynamic-budget 实现（`paused/negative`）；`density_decode.py` = `PROTOTYPE_ONLY`（frozen fixed-K 解码器源码真身，只作 baseline/fallback 与内层复用原语）；RIME/outer-budget-inner-transport = 讨论/规划，无实现。
- **explicit dirty-base / non-DUCA-SHA boundary**：cwd=`E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702`；git common-dir=`.git`；HEAD=`a6bdc084cc145c80b6b2c68d0a38f0deea3e8518`（提交 `docs: freeze sparsehead evidence-first diagnostics design`，Codex，2026-07-28）= **SparseHead 提交，非 DUCA revision**；branch=`codex/duca-total60-plugin-cvpr-20260727`；工作树脏约 275 条。本工作树绝不可称 clean frozen DUCA revision。DUCA 证据 = 命名 untracked working-tree 表面 + `research-wiki/`。a6bdc084 / SparseHead Route-T / 任何 SparseHead 结果不得作为 DUCA 证据。`pc_ot_mras_*` selectors/configs 为 SparseHead 基础（tracked，非 DUCA）。
- **package maturity**：`designed / discussion-level planning only`（无 production 改动、无 commit、无 tests、无 runner、无 launcher）。
- **prepared Source need**：`docs/aris/DUCA_ARIS_SOURCES_TO_PRO_REQUEST-2026-08-14.md`（≤2 份本地文档 + 1 个冻结问题：授权有界动态预算 Builder→Critic→Evaluator PRE_RUN 或 STOP/REVISE），待中央 browser lease 后发送 fresh exact-Project Pro。
- **next_owner**：fresh exact-Project Pro（后续为 Builder → 独立 Critic → Evaluator，仅在其接受后）。
- **next_action**：本地终态收尾（本 receipt + 交付清单核对）；后续由 Pro 裁决 `AUTHORIZE_PRE_RUN` / `STOP` / `REVISE`。本轮不 dispatch 角色、不调用 browser/Pro、不运行。
- **dependency**：fresh exact-Project Pro 接受 + 数据门 + 官方基线门 + 资源门（全部通过后才进入真实 N16R4 pilot）。
- **deadline**：2026-08-14T16:15:00+08:00。
- **single_recovery**：一次有界 liveness/durable-output 检查；不启动重复 ARIS 进程。

## 交付清单

- `docs/aris/ARIS_CPR_PLAN-2026-08-14.md`（C/P/R 主计划，已更新）
- `docs/aris/ARIS_DECISION_LOG-2026-08-14.md`（决策日志，已更新）
- `docs/aris/DUCA_ARIS_SOURCES_TO_PRO_REQUEST-2026-08-14.md`（Sources-to-Pro，新建）
- `research-wiki/log.md`（append，已更新）
- `research-wiki/decision_history.md`（§29，已更新）
- `PAPER_PROGRESS.md`（header + §9，已更新）

无实现、无 PRE_RUN、无 pilot、无训练、无性能/成本/claim。
