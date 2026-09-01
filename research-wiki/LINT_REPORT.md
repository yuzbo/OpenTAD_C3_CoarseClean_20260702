---
type: lint_report
updated: 2026-07-13
status: pass_with_source_warnings
---

# Research Wiki Lint Report

## Summary

- Curated Markdown files: structurally valid.
- Papers: 9.
- Ideas: 22.
- Experiments: 12.
- Claims: 11.
- Graph edges: 99 valid JSONL rows.
- Missing graph node references: 0.
- Orphan paper/idea/experiment/claim nodes: 0.
- Broken curated relative links: 0.
- Duplicate node IDs: 0.
- `query_pack.md`: 7662 characters, below the 8000-character limit.

## Source-record warnings

原始 thread 导出中保留了历史回复内的绝对文件链接。49 个链接在当前 checkout 下
不可解析，主要指向旧 worktree、未纳入仓库的 `analysis_outputs/` 或已迁移文件。
这些链接属于不可改写的来源记录，不计为 curated wiki 链接错误。需要使用时应先在
`source_registry.md` 登记新的可审计位置，而不是静默修改历史原文。

新归档的 CT-P3R-3S Pro raw review 含五个 `sandbox:/mnt/data` artifact 链接；附件只包含
review 文本，没有对应文件。吸收页已将其标记为 absent，不计为 curated broken links，
也不能传播其 standalone `10 passed` 为本仓库测试证据。

## Remaining content gaps

- 实现代理和论文代理目前归档最近 30 轮；完整 task 仍由 Codex thread ID 保留。
- 旧实验的未经匹配 mAP 数字没有强行固化，避免把历史观察升级为论文证据。
- DUCA `70aa069` 已完成，但 matched baseline、full-stack cost、effective-K audit、
  finite-difference 和 geometry audit 仍未完成。
- DUCA global Top-K/G15/soft-RGB bridge 路线已进入 REDESIGN；DUCA-CellCF 仅为
  discussed bounded appeal。固定-grid时间错位、cell utility非加性、same-commit
  fixed-384 pilot 和 trained-checkpoint full-stack cost 均未闭环。
- ChronoTransport formal P3 已失败；`b74101d` 经 Pro 复核不可原样执行。唯一下一动作是
  先形成、复核并冻结 `CT-P3R-3S-r1`；r1 新 SHA 前不得写争议代码或运行任何新 gate。
- 新增三个前沿候选均仅为 proposed，尚无 oracle falsification 或实现证据。
- Dense-Time Spatial Zoom 仅在 gate level 为 `designed`；当前只授权 S1 基础设施，
  S2 与 DART-Zoom 均锁定。dense-resolution headroom、teacher-reference ROI sufficiency
  与 strict total cost 仍无实验结果。
- ChronoTransport 完整 claim verdict=`no`；r1 是唯一允许的上诉协议。Gate 1 只裁决
  frozen-library oracle headroom；input dependence 归因移至 Gate 3 的 window-vector
  ranking 与实际选择。
- ChronoTransport formal P3 已记录为 negative gate；Stage C/P5 不再列作自动待运行任务，除非先有 superseding 设计与重新预注册。
