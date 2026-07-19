# Research Wiki Lint Report

生成时间：2026-07-19

## Summary

- Entity pages: 43
- Idea pages: 21
- Experiment pages: 15
- Paper pages: 7
- Claim entities: 1
- Gaps: 10
- Graph edges: 71
- Orphan entities: 0
- Invalid edge references: 0
- Invalid edge types: 0（含项目已有 `diagnosed_by` 扩展）
- Duplicate nodes: 0
- Duplicate edges: 0
- Broken local Markdown links outside immutable `sources/` archives: 0
- `query_pack.md`: 6027 characters，低于 8000 上限
- `git diff --check`: passed
- 2026-07-13 Pro raw archive: 1539 lines，SHA256 与附件一致
- 2026-07-19 Full60/Q-lift Pro raw archive: 1298 lines，SHA256
  `BBD48B6BCE5E4AC612A395561D2EABCBB1F6DB5880B329EF21CAC6808CFBD5E0`
  与附件一致

## Completeness Audit

- DUCA、ChronoTransport、PhysTime 均有独立 `routes/*-complete-record.md`。
- PhysTime 路线已经补入 1.0 full-run 负结果、性能诊断、Pro `HOLD AND REBUILD` 和 SM-PTAF designed candidate。
- `discussion_coverage.md` 将 native provenance、capacity-matched controls 与 SM-PTAF 映射到路线章节和逐字原文。
- `source_map.md` 登记附件 `4efbcdfc-7b11-46fd-a863-da1d992a110f`、固定 SHA256、GitHub branch 与 commit anchors。
- `AGENTS.md`、`RTK.md`、`current_direction.md`、`decision_register.md` 与 `query_pack.md` 已统一为 P0 rebuild 状态。
- K、native tubelet token J 与 candidate query Q 已明确分离；K 只允许决定 matched candidate cardinality，不能定义物理坐标。
- SM-PTAF 只标记为 `designed`，没有虚构 experiment node、commit、job、测试或 mAP。
- Full60/Q-lift 审查已进入 source registry、方向、query pack、禁区、idea、
  full60 experiment、完整路线和 append-only log。
- `57.57%` 保持 `full60-single-seed-supported`；新 support-preserving
  physical query lift 仍为 `designed`，没有虚构实现、gate 或结果。

## Intentional Warnings

1. `exp:phystime-adatad-k384` 已完成但 verdict 为当前实现负结果，不能产生正向 paper claim。
2. 当前 claim node 数为 1；任何新增论文主张仍必须经过 proof/result
   审计，不得由本轮外部建议直接创建。
3. PhysTime 1.0 的结果不能外推为 physical-time TAD 无效；它首先暴露了 feature provenance、容量、候选与 assignment 混杂。
4. Pro 回复中的核心 PyTorch 代码只是设计草图，`decode_seconds` 的 window offset、all-masked softmax、TIA rank mixing 与功能容量配平仍需本地 gate。
5. `research-wiki/sources/` 是不可变历史导出，含若干指向旧 worktree/生成物的绝对路径；这些路径只用于历史证据，不计为当前 Wiki 本地链接缺陷。
6. feature-token track、DUCA、X3D/SlowFast 与 ChronoTransport 均未恢复为当前主线。
7. 外部建议中的固定性能/成本阈值、非单调 timestamp shuffle、
   cross-attention “唯一方案”和 ActivityNet-v1.3 指定未被当作已证合同。

## Next Lint Trigger

以下任一事件发生后重新运行：SM-PTAF/control 实现落库、native provenance gate 完成、新 experiment node 创建、远端 pilot 提交或结束、方向/论文主张变化、worktree/branch 库存变化。
