---
type: idea
node_id: idea:strict-rectangle-roi-routing
title: "Strict rectangular ROI routing for ZoomToken"
evidence_scope: "seed-42 accuracy; real efficiency unresolved"
open_question: "Does K64 strict support reduce matched end-to-end cost?"
tags: ["offline-tad", "pre-backbone", "roi", "token-selection", "dynamic-compute"]
added: 2026-08-22
updated: 2026-08-28
---

# ZoomToken 严格矩形 ROI 路由

主干前 C 使用连续 ROI 分数排序后的 Top-64，执行支持不保证形成完整矩形。本研究问的是：在
相同 K64 预算下，连续无孔洞的原生空间支持是否比不规则 Top-64 更能保护动作边界和高 tIoU
定位，并且是否能把 36% 原生 token 减少转化为真实端到端成本下降。

已设计并实现四条机制臂：固定 8×8 严格矩形 R1；8×8 矩形内按基础效用选 48 的 R2；
连续宽高、执行全部矩形成员且 `K_t` 自然变化的 R3；7×7 无孔洞矩形核心加 15 个框外
关键 free token、总数 64 的 R4。R4 的框外名额独立且有上限，不能退回旧 ROI modifier
的全局混排；先前讨论中的 6×8+16 方案已被 7×7+15 的单一固定实现取代。四臂均在重主干前
gather 原生 tubelet，复用同一 true-ragged VideoMAE、sparse adapter、ActionFormer 与官方训练
配方。

设计规格：`docs/superpowers/specs/2026-08-22-zoomtoken-strict-rectangle-roi-routing-design.md`。
R1 的固定实现为 revision `9e25c6d38de8c993948025629181470b858682b4`；seed 42 的 60-epoch
THUMOS14 job `1249099` 已完成。epoch-59 EMA 的
Avg-mAP/mAP@0.6/mAP@0.7 为 `69.07/61.14/46.57`，相对 C 为
`+0.85/+0.13/+1.22` 个百分点，三项预注册准确率条件全部通过。这支持“完整矩形支持可改善
当前 K64 主干前选择的高 tIoU 定位”这一 seed-42 准确率判断；尚无配对端到端成本与多 seed
结果，因此不能升级为论文级效率结论。

R2/R3/R4 及其区分性对照固定在 revision
`b1d9fa7b10209b23c4405b4be3965ee66f3c05f5`；八个 seed-42、60-epoch jobs
`1249125–1249132` 均已完成。final-EMA Avg-mAP/mAP@0.6/mAP@0.7 为：R2
`66.56/59.06/45.17`，R2-SHUF48 `66.17/58.53/44.47`，Q48-GLOBAL
`65.78/58.62/44.74`，R3 `67.88/60.32/46.41`，R3-AREA-SHIFT
`67.50/60.26/45.09`，R4 `68.02/60.32/46.26`，R4-SHUF15
`67.19/60.17/46.20`，Q64-GLOBAL `67.84/60.66/45.39`。

这些结果支持两个有限解释：R2 在这个种子上优于框内乱序和全局 Top-48，对框内内容排序与矩形
eligibility 提供单种子证据；R3 相对时间错位对照在 mAP@0.7 高 `1.32` 点，支持时间对齐主要
保护高 tIoU。R4 相对 R4-SHUF15 的 mAP@0.7 只高 `0.06` 点，没有达到预先规定的 `0.30`
门槛，因此不能声称框外 learned ranking 有效，也不为该机制增加成本或多种子实验。

BPNS-R1 的最强替代解释仍是：36% 原生 token 减少可能被其他全栈开销抵消。成本回放 job
`1257281` 因 K100 原始精度值与舍入值绑定错误，在运行 R1 前停止；它既不是效率证据，也不是
模型负结果。最小数值绑定修正已在 clean revision
`e9323448f6cd78b99bb3de53fd9ffb55f3676d65` 完成并通过 focused tests、独立 Critic 与结果盲
PRE_RUN。唯一替代 job `1258299` 正在执行不变的同硬件 K100/R1 配对回放；完整终态前不解释
任何 live/partial 数值，也不产生效率、短动作或边界结论。
