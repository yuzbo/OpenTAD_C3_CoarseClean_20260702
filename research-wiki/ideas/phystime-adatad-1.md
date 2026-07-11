---
type: idea
node_id: idea:phystime-adatad-1
title: "PhysTime-AdaTAD 1.0 raw-video head isolation"
stage: active
outcome: pending
thesis: "在相同 K384 不规则 raw-frame observations 和 official AdaTAD backbone 下，仅比较 selected-axis、physical-grid 与 PhysTime head。"
risks: "raw integration 已测试但真实 CUDA gate 尚未通过；physical-grid baseline 可能已经足够；full-run 成本与高-IoU收益未知。"
based_on: ["paper:zhang2022_actionformer", "paper:liu2024_adatad", "paper:kim2024_te_tad"]
target_gaps: ["gap:G1", "gap:G4", "gap:G5", "gap:G7", "gap:G9"]
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T15:21:00+08:00
---

# PhysTime-AdaTAD 1.0

## Research question

相同 raw observations 与相同 official VideoMAE-S adapter 下，显式 physical-time detection 是否优于 ordinary selected-rank sequence？

## Primary contract

- logical T=768，deterministic GT-independent random fixed subsample K=384；
- decode/backbone 只消费 K；
- 三头 selected-index checksum 完全一致；
- selected baseline 才 remap GT；physical-grid 与 PhysTime 均保留 original timeline；
- PhysTime 的 GT、proposal、NMS 和 evaluation 用 absolute seconds；
- primary comparison 无 learned selector、actionness、teacher、ledger、dynamic K、paired consistency。

## 当前实现状态

代码提交 `549bb81` 已完成 raw transform、三配置、validator、合成 one-step 梯度证明、真实 gate 工具和 gate-dependent launchers；远端 focused suite `45 passed`。这只把状态提升到 `tested`：真实 THUMOS CUDA gate、正式训练和 mAP 均仍 pending，idea outcome 继续保持 pending。

## 成功/降级规则

- 胜两种 sparse baseline，且高-IoU/边界指标改善：支持完整方法候选。
- 只胜 selected-axis：仅支持 original-time geometry。
- 不胜 physical-grid：不声称 measure head 必要。
- 不具备 dense reference 的合理 accuracy-cost trade-off：不能作论文主方法。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
