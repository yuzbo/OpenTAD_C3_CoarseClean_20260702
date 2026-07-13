# Research Wiki Lint Report

生成时间：2026-07-13

## Summary

- Entity pages: 38
- Idea pages: 21
- Experiment pages: 11
- Paper pages: 6
- Claim entities: 0（按协议刻意为空）
- Gaps: 10
- Graph edges: 66
- Orphan entities: 0
- Invalid edge references: 0
- Invalid edge types: 0（含项目已有 `diagnosed_by` 扩展）
- Duplicate nodes: 0
- Duplicate edges: 0
- Broken local Markdown links outside immutable `sources/` archives: 0
- `query_pack.md`: 4430 characters，低于 8000 上限
- `git diff --check`: passed
- 2026-07-13 Pro raw archive: 1539 lines，SHA256 与附件一致

## Completeness Audit

- DUCA、ChronoTransport、PhysTime 均有独立 `routes/*-complete-record.md`。
- PhysTime 路线已经补入 1.0 full-run 负结果、性能诊断、Pro `HOLD AND REBUILD` 和 SM-PTAF designed candidate。
- `discussion_coverage.md` 将 native provenance、capacity-matched controls 与 SM-PTAF 映射到路线章节和逐字原文。
- `source_map.md` 登记附件 `4efbcdfc-7b11-46fd-a863-da1d992a110f`、固定 SHA256、GitHub branch 与 commit anchors。
- `AGENTS.md`、`RTK.md`、`current_direction.md`、`decision_register.md` 与 `query_pack.md` 已统一为 P0 rebuild 状态。
- K、native tubelet token J 与 candidate query Q 已明确分离；K 只允许决定 matched candidate cardinality，不能定义物理坐标。
- SM-PTAF 只标记为 `designed`，没有虚构 experiment node、commit、job、测试或 mAP。

## Intentional Warnings

1. `exp:phystime-adatad-k384` 已完成但 verdict 为当前实现负结果，不能产生正向 paper claim。
2. 当前没有 claim nodes；必须在修复后的 matched full runs、统计审计和 result-to-claim 之后创建。
3. PhysTime 1.0 的结果不能外推为 physical-time TAD 无效；它首先暴露了 feature provenance、容量、候选与 assignment 混杂。
4. Pro 回复中的核心 PyTorch 代码只是设计草图，`decode_seconds` 的 window offset、all-masked softmax、TIA rank mixing 与功能容量配平仍需本地 gate。
5. `research-wiki/sources/` 是不可变历史导出，含若干指向旧 worktree/生成物的绝对路径；这些路径只用于历史证据，不计为当前 Wiki 本地链接缺陷。
6. feature-token track、DUCA、X3D/SlowFast 与 ChronoTransport 均未恢复为当前主线。

## Next Lint Trigger

以下任一事件发生后重新运行：SM-PTAF/control 实现落库、native provenance gate 完成、新 experiment node 创建、远端 pilot 提交或结束、方向/论文主张变化、worktree/branch 库存变化。
