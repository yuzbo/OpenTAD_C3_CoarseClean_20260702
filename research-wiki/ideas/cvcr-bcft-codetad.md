---
type: idea
node_id: idea:cvcr-bcft-codetad
title: "CVCR / BCFT / CoDeTAD 替代路线"
stage: designed
outcome: unknown
tags: ["counterfactual", "routing", "alternative"]
added: 2026-07-11
---

# CVCR / BCFT / CoDeTAD 替代路线

## One-line thesis

不再预测“哪一帧像动作”，而是预测“在这个时间×层单元上执行一次真实计算，能以多大代价降低下游 detector regret”，并据此路由重计算、复用或跳过。

## 为什么提出

若 pre-backbone frame dropping 被成本或几何否定，需要更接近真实计算价值的替代问题定义。该路线同时回应 DUCA 的三个薄弱点：selected-axis 几何失真、ST 梯度与 hard 决策效用不等价、probe/selector 成本可能吞掉节省。

## 已有证据

ResearchClaw 发散审查提出；2026-07-11 近邻文献复核表明，自适应选帧、按帧重要性重加权、逐层动态剪枝和 TAD block dropping 均已有直接先例，因此创新点不能停留在“学习重要性并减少计算”。尚无完整实现与匹配结果。

## 收敛后的候选结构

1. **CVCR：反事实计算价值路由。** 在训练集上对少量候选时间×层单元执行 one-swap、drop/recompute 或 paired replay，估计其对 detector loss、proposal recall 和 boundary error 的边际影响；学习 cost-normalized regret predictor。推理只运行该轻量预测器，不访问 GT、teacher 或离线 ledger。
2. **BCFT：稠密低成本底座 + 稀疏重计算残差。** 保留覆盖全时间轴的廉价表示，只在高预计 regret 的位置调用昂贵 backbone block；未重算位置使用底座特征或可审计的 HOLD。它比直接删除帧更容易保护边界和物理时间几何。
3. **CoDeTAD：时间×层联合预算。** 预算作用于真实 block/token/tubelet 计算单元，而非请求 K；优化目标是 detector risk + measured latency/energy。只有 packed execution 或实际 block skip 产生真实收益时才成立。

## 与现有路线的关系

- DUCA 是输入级 acquisition baseline，可用于证明“删帧是否值得”。
- ChronoTransport 的失败说明 risk 聚合尺度和 transport 机制尚不可靠；CVCR 可以继承 paired replay 基础设施，但必须改用与执行单元同尺度的 regret target，不能直接复活旧 P3。
- PhysTime 解决不规则采样几何，科学新颖性更高，但容易演化成 detector 重写；不与 CVCR 同时作为主贡献。

## 当前选择或否定理由

作为 DUCA 失败后的首选 pivot 候选，不视为已证明比 DUCA 更优。优先级为：CVCR+BCFT > 独立 CoDeTAD 大改 > PhysTime detector 重写。

## 风险与失败模式

teacher/counterfactual 训练成本、regret target 噪声、packed kernel 无实测加速、策略退化为固定 block drop、跨 detector 泛化，以及与动态剪枝/模型压缩工作的 novelty collision。

## 下一次允许采取的动作

先写最小预注册 gate，不立即展开完整实现：同一执行单元上的预测风险必须与 held-out hard counterfactual regret 正相关，并优于 actionness、transition、attention 和随机评分；真实端到端延时必须下降。只有 gate 通过才进入 full train。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
