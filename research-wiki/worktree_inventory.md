# 跨 Worktree 与分支库存

审计时间：2026-07-11。此表用于防止把“当前 checkout 没有”误判成“项目没有实现”。提交状态以审计时本地 Git 对象为准。

| Worktree / branch | 审计 HEAD | 路线与证据边界 |
| --- | --- | --- |
| `OpenTAD_C3_CoarseClean_20260702` / `codex/c3-coarse-clean-20260702` | `92029ea` | ChronoTransport formal Stage-B；P3 负 gate；本地比 origin `3554b6f` 超前 15 commits |
| `OpenTAD_DetectorAwareSelector_Worktree_20260706` / `codex/detector-aware-selector-20260706` | `c799c48` | detector-aware ActionFormer teacher utility 早期分支 |
| `OpenTAD_DUCA_SignedUtility_Worktree_20260706` / `codex/duca-signed-utility-20260706` | `edaf589` | signed detector utility 证据修复分支 |
| `OpenTAD_DUCA_Stage234_Worktree_20260706` / `codex/duca-stage234-owner-20260706` | `b15c278` | DUCA Stage2-4 evidence gates |
| `OpenTAD_DUCA_Stage23Owner_Worktree_20260706` / `codex/duca-stage23-owner-20260706` | `679f194` | Stage3 precheck contracts |
| `OpenTAD_DUCA_Stage23Runners_Worktree_20260706` / `codex/duca-stage23-runners-20260706` | `3ce6bae` | Stage2/3 runners 与 09b0d31 review absorption |
| `OpenTAD_DUCA_Stage3E2E_Worktree_20260706` / `codex/duca-stage3-e2e-20260706` | `36c92d4` | TrueTime Stage3 E2E precheck |
| `OpenTAD_GASVT_CostAudit_20260710` / `codex/gas-vt-stage23-detector-aware-20260706` | `a5e1774` | DUCA full-stack profiler、source parity、ResearchClaw 24 ideas；已推 origin |
| `OpenTAD_GASVT_Worktree_20260706` / `codex/phystime-tad-2` | `696f77d` | PhysTime-TAD 2.0 首次实验 track；后续修复在同祖先的 deploy-fix/PhysTime-AdaTAD 分支 |
| `OpenTAD_PhysTime_DeployFix_20260710` / `codex/phystime-adatad-1` | `2b7f83f`（本轮编辑前） | 当前 Wiki 与 PhysTime-AdaTAD 1.0 规格/计划主线 |
| `OpenTAD_TrueTimeJointSelector_Worktree_20260706` / `codex/truetime-joint-selector-20260706` | `05baa48` | ActionFormer selector gradient proof 早期分支 |

## 其他关键 refs

- `codex/gas-vt-mainline-20260706` / `53124a2`：GAS-VT deployment gate 历史锚点。
- `codex/phystime-deploy-fix-20260710` / `1893004`：PhysTime feature-track 数据恢复与取消前的修复锚点。
- `codex/phystime-adatad-1` 继承 `1893004`，并追加 `9266ebc` raw-video 规格、`517785d` 实施计划和 Wiki。

## 解释规则

1. worktree HEAD 只证明代码对象存在，不证明其方法有效。
2. local-ahead 分支必须记录未推送风险；不能把本地 smoke 写成远端可复现。
3. 相互分叉的 DUCA worktree 是历史责任拆分，不应被拼成一个“同时存在的最终模型”。
4. PhysTime 当前实现边界以 `routes/phystime-complete-record.md` 为准；ChronoTransport 负结果以 `92029ea` 为准。
5. 每次全量 Wiki 审计都应刷新本表，而不是静态相信本次快照。
