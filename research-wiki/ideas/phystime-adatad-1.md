---
type: idea
node_id: idea:phystime-adatad-1
title: "PhysTime-AdaTAD 1.0 raw-video head isolation"
stage: tested
outcome: negative_current_implementation
thesis: "在相同 K384 不规则 raw-frame observations 和 official AdaTAD backbone 下，仅比较 selected-axis、physical-grid 与 PhysTime head。"
risks: "当前三头同时改变容量、时序上下文和坐标表示，不能隔离 physical-time 的科学价值；单数据集单种子证据不足。"
based_on: ["paper:zhang2022_actionformer", "paper:liu2024_adatad", "paper:kim2024_te_tad"]
target_gaps: ["gap:G1", "gap:G4", "gap:G5", "gap:G7", "gap:G9"]
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-12T14:20:00+08:00
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

最终提交 `3ac93a1` 已完成真实 gate、两 epoch 稳定性 gate 和三头 full run。只读最佳 checkpoint 复算与官方结果一致。当前 PhysTime 1.0 没有胜过两个 sparse controls，结果为负。完整性能诊断表明它同时使用更小的检测栈、缺少同等跨时间上下文、被 raw absolute seconds 主导，并在候选密度、短动作监督和粗层 attention 上存在缺陷；因此不能把该负结果外推为 physical-time detection 本身无效。

## 2026-07-12 diagnosis decision

- 保留研究问题，冻结 PhysTime 1.0 实现作为负基线。
- 下一版必须先做 capacity/context/candidate-matched control，只改变物理时间表示。
- 在因果对照通过前不启动第二数据集、三种子或论文主表扩展。
- 数字与 artifact 路径只见 `docs/evaluation/results.md`；解释见 `docs/evaluation/phystime-performance-drop-diagnosis.md`。

## 成功/降级规则

- 胜两种 sparse baseline，且高-IoU/边界指标改善：支持完整方法候选。
- 只胜 selected-axis：仅支持 original-time geometry。
- 不胜 physical-grid：不声称 measure head 必要。
- 不具备 dense reference 的合理 accuracy-cost trade-off：不能作论文主方法。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
