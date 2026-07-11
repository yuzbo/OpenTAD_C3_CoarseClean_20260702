# Research Wiki Lint Report

生成时间：2026-07-11

## Summary

- Entity pages: 36
- Idea pages: 20
- Experiment pages: 10
- Paper pages: 6
- Claim entities: 0（按协议刻意为空）
- Gaps: 10
- Graph edges: 55
- Orphan entities: 0
- Invalid edge references: 0
- Invalid edge types: 0
- Duplicate edges: 0
- Broken local Markdown links: 0
- `query_pack.md`: 3660 characters，低于 8000 上限
- `git diff --check`: passed

## Completeness Audit

- DUCA、ChronoTransport、PhysTime 均有独立 `routes/*-complete-record.md`。
- `discussion_coverage.md` 按主题映射到路线章节和权威来源。
- `worktree_inventory.md` 覆盖审计时 11 个本地 worktree，并区分 local-only、origin 与 ancestry。
- 主任务用户侧原文及相关代理近期记录已迁入 `sources/`，并固定 SHA256。
- ChronoTransport 状态已从过时的“仅到 `78d4c00`、科学 gate 未做”纠正为：`92029ea` formal P3 已执行且 gate 为负，Stage C/P5 未解锁。
- DUCA `a5e1774` full-stack/source-parity 分支与 ResearchClaw 第二轮 24 ideas 已纳入。

## Intentional Warnings

1. `exp:phystime-adatad-k384` 是 `experiment_running`，仍无 mAP，不能产生 supports/invalidates claim edge。
2. 当前没有 claim nodes；必须在正式 result-to-claim/proof audit 后创建。
3. 14 份外部附件通过路径与 SHA256 索引；尚未全部复制进仓库，清理附件目录前必须迁移受控归档。
4. 原始讨论导出只能证明讨论历史，不能代替 commit、测试或实验 artifact。
5. PhysTime feature-token track 已取消；代码仅作算子/单测资产，不是 active paper experiment。
6. ChronoTransport 本地分支比 origin 超前 15 commits，远端复现风险仍未消除。

## Next Lint Trigger

以下任一事件发生后重新运行：worktree/branch 新增或删除、raw-video implementation 合入、real gate 提交/完成、K384 三头任一 full run 完成、方向或论文主张变化。
