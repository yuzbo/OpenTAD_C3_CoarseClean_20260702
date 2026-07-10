---
type: idea
node_id: idea:coder-tal
title: "CoDeR-TAL"
stage: archived
outcome: pending
thesis: "以 temporal localization distortion 驱动 codec decode mode 与 rate-cost 分配。"
risks: "partial decode API、codec/hardware 依赖、系统面过大、跨 codec 泛化困难。"
based_on: []
target_gaps: ["gap:G7"]
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# CoDeR-TAL

## 核心

不是请求少量逻辑帧，而是按 GOP 在 full RGB、I-frame+MV/residual、partial reconstruction、low-resolution decode 之间做 rate-distortion 决策；distortion 包含 endpoint error、phase transition 和 short-action tail。

## 当前裁决

高风险备线。当前没有证据表明 decode/codec dependency 是主瓶颈，也没有资源实现跨 H.264/H.265 的真实 partial decoder，因此不进入主线。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
