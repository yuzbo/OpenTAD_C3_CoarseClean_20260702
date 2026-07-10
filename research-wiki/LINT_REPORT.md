# Research Wiki Lint Report

生成时间：2026-07-11

## Summary

- Entity pages: 31
- Idea pages: 17
- Experiment pages: 8
- Paper pages: 6
- Claim entities: 0（按协议刻意为空）
- Gaps: 10
- Graph edges: 48
- Orphan entities: 0
- Invalid edge references: 0
- Duplicate edges: 0
- Broken local Markdown links: 0
- `query_pack.md`: 3348 characters，低于 8000 上限
- `git diff --check`: passed

## Intentional Warnings

1. `exp:phystime-adatad-k384` 仍是 planned，没有结果，不能产生 supports/invalidates claim edge。
2. 当前没有 claim nodes；必须在正式 result-to-claim/proof audit 后创建。
3. 14 份原始附件通过路径与 SHA256 索引，没有复制进仓库；外部附件清理前应迁移受控归档。
4. PhysTime feature-token track 已取消，但代码保留用于单元测试，不应被 lint 当作 active paper experiment。
5. README 下方仍包含 C3 历史说明；顶部已明确以 research-wiki 为当前方向来源。

## Next Lint Trigger

以下任一事件发生后重新运行：raw-video implementation 合入、real gate 提交/完成、K384 三头任一 full run 完成、方向或论文主张变化。
