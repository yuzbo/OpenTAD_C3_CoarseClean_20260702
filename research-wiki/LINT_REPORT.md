---
type: lint_report
updated: 2026-07-17
status: pass_with_source_warnings
---

# Research Wiki Lint Report

## Summary

- Curated Markdown files: structurally valid.
- Ideas: 17.
- Experiments: 9.
- Claims: 11.
- Graph edges: 51 valid JSONL rows.
- Missing graph node references: 0.
- Orphan idea/experiment/claim nodes: 0.
- Broken curated relative links: 0.
- `query_pack.md`: 7512 characters, below the 8000-character limit.
- The installed ARIS helper exposes no `lint` command. A read-only structural
  validator parsed all 51 JSONL edges, checked every edge endpoint against the
  current node set, and enforced the query-pack budget; all checks passed.

## Source-record warnings

原始 thread 导出中保留了历史回复内的绝对文件链接。49 个链接在当前 checkout 下
不可解析，主要指向旧 worktree、未纳入仓库的 `analysis_outputs/` 或已迁移文件。
这些链接属于不可改写的来源记录，不计为 curated wiki 链接错误。需要使用时应先在
`source_registry.md` 登记新的可审计位置，而不是静默修改历史原文。

## Remaining content gaps

- 实现代理和论文代理目前归档最近 30 轮；完整 task 仍由 Codex thread ID 保留。
- 旧实验的未经匹配 mAP 数字没有强行固化，避免把历史观察升级为论文证据。
- Spatial Zoom S1 passed its prior no-open Gate, but serial matrix Job
  `1168823` failed closed in the first cell on the frozen cadence audit. The
  v4 trace-publication repair passed exact remote tests, but replacement Gate
  `1170341` exposed a physical-slot/cgroup-selector bug before profiling. Its
  local repair is `tested_local`, and independent review returned `DEPLOY`
  with no P0/P1; no valid replacement Gate or matrix exists.
- The final paired S1 result, GO/KILL, post-result Pro review, and any S2
  authorization do not exist.
- ChronoTransport formal P3 已记录为 negative gate；Stage C/P5 不再列作自动待运行任务，除非先有 superseding 设计与重新预注册。
