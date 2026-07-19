---
type: idea
node_id: idea:phystime-tad-2
title: "PhysTime-TAD 2.0"
stage: active
outcome: mixed
thesis: "把 TAD 定义为物理时间支持测度上的积分算子，使检测在观测网格细化和缺失下比 selected-rank operator 更一致。"
risks: "可能被 timestamp/interpolation/mTAN-like baseline 匹配；feature track 不等于 raw-video；新颖性必须窄而可证。"
based_on: ["paper:shukla2021_mtan", "paper:kim2024_te_tad", "paper:zeng2024_temporal_robustness", "paper:sun2026_liquidtad"]
target_gaps: ["gap:G1", "gap:G2", "gap:G3", "gap:G5"]
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-20T00:00:00+08:00
---

# PhysTime-TAD 2.0

## 单一核心机制

对 observation ownership interval `I_i` 与 physical query cell `R_q`，使用重叠长度 `m_qi = |R_q ∩ I_i|` 作为指数外的 evidence mass；每层直接从原始不规则观测投影，query grid 由 seconds spacing 构造，检测头在秒坐标 assignment、decode、NMS。

## 关键性质

- constant-kernel support split additivity；
- padding invariance；
- gap 不扩张；
- 研究坐标与 K/rank 无关；matched comparison 中 query cardinality 可由 K 决定以对齐候选容量，但 center/width/stride 仍全部是秒；
- uncovered query 输出有限零；
- endpoint intensity 按 cell width 积分为事件概率。

## 已实现范围

物理几何、measure projection、PhysTimeHead、detector、feature transforms 与 focused tests 已存在。该代码证明软件合同，不证明论文效果。

## 长期角色

这是最终独立 detector 的方法本体。`PhysTime-AdaTAD 1.0` 是它进入 raw-video official AdaTAD 的第一阶段验证。

## 2026-07-13 Pro 审查吸收

PhysTime 1.0 已冻结为负基线。下一候选被具体化为 `idea:sm-ptaf`：删除 `192 -> 384` feature interpolation，以 native tubelet multi-atom support 建立 provenance，用显式 mass residual 与有界 correction 投影到 candidate-matched physical query pyramid，并复用 ActionFormer 等级上下文与 assignment。

该候选当前只有 `designed` 状态。必须先通过 capacity-matched coordinate-only control；否则无法区分 physical coordinate、support operator、候选恢复和容量增加的贡献。

## 2026-07-17 Matched-Medium 裁决

同 commit、同 K384/J192、同 seed 和同 20-epoch schedule 的三臂实验已经完成。physical-metric ActionFormer 相对 selected-axis 获得 `+14.46` Avg-mAP，说明真实物理时间度量在当前离线稀疏 TAD 设置中具有明确价值。G1b SDPQ 相对 selected-axis 仅 `+0.46` Avg-mAP，并显著落后 physical-metric；当前 support-decoupled operator 不构成已证实优势。

因此 idea outcome 为 `mixed`：物理时间检测假设获得 matched-medium 支持，完整 support-measure/SDPQ 结构尚未获得支持。下一步以 physical-metric 为 survivor 做复现和机制拆分，不把 G1b 直接升级为主方法。

## 2026-07-19 Full60 / Q-Lift 审查吸收

单种子 60-epoch 结果进一步把 physical-metric 相对 selected-axis 的优势
固定为 `+16.29` Avg-mAP，但它仍低于不可直接横比的旧随机和 dense 锚点。
代码核验确认：真实物理时间从检测头 point construction 开始覆盖
assignment、回归和 decode，VideoMAE/TIA/projection 表示仍不见 timestamp。

下一 designed candidate 是 support-preserving physical query lift：
K384 RGB 只产生 J192 native supports，另建 Q192/Q384 deterministic
detection query，由受 mask 的 support-to-query bridge 读取 support，再复用
ActionFormer branch/head。query 不属于 observation，也不填补 dense RGB
evidence。

该候选必须在同一新 commit 下重跑
`Q192/Q384 × uniform-rank-seconds/physical-seconds` 四臂。cross-attention
只是优先实现，不是已证明的唯一结构；固定性能/成本阈值、timestamp
shuffle 和第二数据集选择仍待审计。当前 stage 保持 `active`，operator
状态保持 `designed`，不得写成 implemented/tested/paper-ready。

## 2026-07-20 STOP-Q-LIFT 裁决

第二轮代码严审否定了“physical Q192 有效，所以剩余缺口一定来自 Q 不足”
这一未经证明的跳跃。当前保留 physical-metric Q192，暂停 Q384、
cross-attention 和其他训练型 Q-lift。

下一机制链固定为：

1. full-precision NMS 冻结重放；
2. 冻结 checkpoint 的 decode cross-replay；
3. Q192 的 UU/UP/PU/PP 轴因子化；
4. 无训练 subcell Q-density replay；
5. 只有 oracle/pre-NMS 高 IoU coverage 明确改善，才恢复 Q-lift 讨论。

本地核验还发现原 Pro 报告将 decode 与 assignment 主效应公式标签互换。
当前 strict inside-GT 使用 decode center，因此四臂只分解两个实现开关，
不构成完全纯净的抽象因果分解。idea outcome 继续为 `mixed`，不创建新
paper claim。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
