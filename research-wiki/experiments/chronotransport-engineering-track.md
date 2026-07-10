---
type: experiment
node_id: exp:chronotransport-engineering-track
title: "ChronoTransport Stage-A replay and Stage-B engineering track"
idea: idea:chronotransport-dcrt
status: completed_negative_gate
verdict: negative
confidence: high
metrics: "Engineering gates pass; formal seed-3407 P3 science gate fails; P5 absent."
provenance: "local branch codex/c3-coarse-clean-20260702 at 92029ea"
added: 2026-07-11T00:00:00+08:00
---

# ChronoTransport Engineering Track

## Proven

Stage-A real model smoke、paired replay determinism、dense near-zero regret anchor、Stage-B one-step trainability、frozen-state audit，以及正式 fit/calibration/evaluation 流程已在该本地分支记录。

## Formal P3 verdict

正式单种子 P3 gate 为 FAIL：risk-regret 排序相关性为负，cell-risk 求和与窗口 regret target 尺度错配，feature transport 改善不稳定。按预注册规则未启动 Stage C。数值只保留在 `92029ea` 的方法/结果记录中。

## Not proven

未证明 risk calibration 有效、full-stack latency 达标、高-IoU/短动作通过 kill gate，亦未完成 Stage-C 三种子。局部 detector-regret 改善不能覆盖 P3 总 gate 失败。

## Reproducibility warning

相关 15 commits 尚未推送到对应 origin branch。当前 PhysTime branch 不包含 ChronoTransport source。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
