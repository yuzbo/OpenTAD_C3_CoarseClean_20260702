---
type: idea
node_id: idea:phystime
title: "PhysTime 时间戳感知 TAD"
stage: empirically_supported
outcome: mixed_physical_metric_supported_sdpq_not_superior
tags: ["physical-time", "parallel-route", "tad"]
added: 2026-07-11
---

# PhysTime 时间戳感知 TAD

## One-line thesis

显式保留/使用物理时间戳或不规则采样几何，避免 selected-axis 等间隔假设。

## 为什么提出

直接回应 DUCA selected-axis geometry 风险，并探索不删失时间语义的 detector。

## 已有证据

raw-video K=384 三头 matched 实验已完成。selected-axis、physical-grid、
PhysTime 最佳 Avg-mAP 分别为 63.61、59.14、57.21；PhysTime 仅相对
physical-grid 在 mAP@0.7 上高 2.62 点，仍比 selected-axis 低 6.91 点。

## 当前选择或否定理由

当前 PhysTime-AdaTAD v1 的整体优越性主张被否定，不能升级为论文主方法或解锁
Phase 2。保留的只是尚未归因、尚未多种子确认的高 tIoU 局部信号。

## 风险与失败模式

实际结果验证了该风险：额外物理时间结构与 endpoint 目标降低总体 mAP，并未击败
selected-axis。单种子结果也不能把 mAP@0.7 的局部收益归因于时间戳本身。

性能缺口应分两层解释。selected-axis 相对项目内 dense AdaTAD 锚点 68.29 低
4.68 点，主要对应 K=384/768 的观测删减、边界信息损失和固定不规则采样；PhysTime
又比同采样 selected-axis 低 6.40 点，说明主要新增问题位于物理时间表示、head/GT
mapping 和额外 endpoint 优化，而不能全部归罪于 sampler。selected-axis 虽在几何上
压缩真实间隔，却天然符合 ActionFormer 的规则序列与卷积金字塔先验；PhysTime 只在
head 端恢复物理时间，并未让 backbone/neck 同时变成连续时间模型。

该实验也不是容量和目标完全同构的组件消融。PhysTime 使用不同 head、投影和额外
endpoint loss，因此更低训练 loss 不代表更优检测；目前也缺少对 dense 基线的正式
端到端 latency/energy Pareto，不能用较低显存替代效率证据。

## 下一次允许采取的动作

不得直接扩展模型或调权重。若继续，只允许同一 head/训练设置下分别关闭时间戳、
support geometry 与 endpoint objective 的 matched factorial ablation，并对决定性配置
运行多种子；否则冻结为负结果。

## 2026-07-23 T1/T2 分层裁决

当前负结果不能证明“时间信息无用”。随机 K=384 首先丢失了部分边界内容，而 PhysTime v1 又
同时改变 head、标签分配、回归几何和 endpoint loss；坐标编码既不能恢复未观测内容，也无法
单独归因。selected-axis 反而与 ActionFormer 的规则卷积金字塔先验匹配，PhysTime 只在末端
恢复物理时间，前面的 VideoMAE、projection 和 neck 仍把非均匀选中帧当成等间隔序列。

下一次只允许先做低风险 T1：保持 selected-axis head、GT assignment、decode 和相同硬选帧完全
不变，把归一化原始位置、左右间隔、间隔不对称性通过末层零初始化的小投影残差加入 selected
token。初始化第 0 步必须与基线数值等价。三臂为 baseline、T1、打乱/常量时间码负对照；主要
观察 terminal official Avg-mAP 与 mAP@0.6/0.7。T1 只修复 detector 看不到间隔的问题，不改变
选帧内容；若 T1 无益，禁止直接升级 T2。只有 T1 显示稳定高 tIoU 收益，才允许在同一 head 内
逐项测试 physical prior/assignment/decode。

## 2026-07-28 SparseHead 路线合并后的裁决

PhysTime v1 的负结论保持不变，但它不能覆盖后续 native-J192 证据。matched
20-epoch 的 selected-axis/physical-metric/SDPQ Avg-mAP 为
`30.42/44.88/30.88`；full60 的 selected-axis/physical-metric EMA 经
full-precision NMS 重放为 `41.283021/57.608685`。因此 physical-metric 是
当前经验 survivor；SDPQ 只证明可训练，尚未证明优于 matched physical-metric。

当前仓库是唯一可写 SparseHead/PhysTime 路线。旧 irregular bridge 与 rank-assignment
只作诊断，SDPQ 是唯一允许继续裁决的结构候选。下一步不是继续扩大 query 或调 loss，
而是先在当前精确提交复现 real gate 与 physical-metric control；SDPQ 必须在 matched
medium、短动作/high-tIoU、support observability 和真实成本上过门后才允许 full60。
完整合并记录见 `experiments/sparsehead-route-consolidation-20260728.md`。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
