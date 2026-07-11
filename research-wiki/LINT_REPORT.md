---
type: lint_report
updated: 2026-07-11
status: pass_with_source_warnings
---

# Research Wiki Lint Report

## Summary

- Curated Markdown files: structurally valid.
- Ideas: 16.
- Experiments: 8.
- Claims: 10.
- Graph edges: 49 valid JSONL rows.
- Missing graph node references: 0.
- Orphan idea/experiment/claim nodes: 0.
- Broken curated relative links: 0.
- `query_pack.md`: 2919 characters, below the 8000-character limit.

## Source-record warnings

原始 thread 导出中保留了历史回复内的绝对文件链接。49 个链接在当前 checkout 下
不可解析，主要指向旧 worktree、未纳入仓库的 `analysis_outputs/` 或已迁移文件。
这些链接属于不可改写的来源记录，不计为 curated wiki 链接错误。需要使用时应先在
`source_registry.md` 登记新的可审计位置，而不是静默修改历史原文。

## Remaining content gaps

- 实现代理和论文代理目前归档最近 30 轮；完整 task 仍由 Codex thread ID 保留。
- 旧实验的未经匹配 mAP 数字没有强行固化，避免把历史观察升级为论文证据。
- `70aa069` fixed-384 仍在运行，完成后必须更新 experiment 与 claim 节点。
- formal trained-checkpoint cost matrix、matched baseline、finite-difference 和 geometry
  audit 仍未完成。
- ChronoTransport formal P3 已记录为 negative gate；Stage C/P5 不再列作自动待运行任务，除非先有 superseding 设计与重新预注册。
