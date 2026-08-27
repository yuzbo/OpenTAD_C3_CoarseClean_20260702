---
type: research_contract
title: "DUCA 端点覆盖、物理连续 cliplet 与动态预算冻结合同"
status: superseded
canonical: false
superseded_by: DUCA_PHYSICAL_CLIPLET_DYNAMIC_BUDGET_FINAL_CONTRACT-2026-08-21.md
updated: 2026-08-21
scope: "DUCA 新 clean implementation cycle；不追认历史 65.xx、UVT 或 Fovea/Query-Bridge 的性能"
---

# DUCA 端点覆盖、物理连续 cliplet 与动态预算冻结合同

> 本文件保存第一版合同，仅用于追溯。最终 Pro 裁决已删除 SQD、修正动态预算的
> 定点整数定义，并把 GAPPACK 改为依赖真实 VideoMAE 模块结构的失败关闭控制。
> 唯一规范文件为
> `DUCA_PHYSICAL_CLIPLET_DYNAMIC_BUDGET_FINAL_CONTRACT-2026-08-21.md`。

## 0. 裁决、证据状态与取代范围

本合同冻结新一轮 DUCA 的**机制、归因顺序和实验比较**。它不意味着代码已经实现、
PRE_RUN 已通过、训练已经提交，或任一性能/成本结论成立。当前状态严格为
`designed_not_implemented_not_tested`。

它取代旧的固定 K、selected-rank 输入作为新主线的解释；
`duca_final_model_contract.md`、历史 `65.385724%` 课程、UVT 和 Fovea/Query-Bridge 仍是
代码与诊断记忆，不是本合同的正向证据。历史 65.3857 使用非均匀原始帧的 selected-rank
拼接输入，且消耗 30+60 epoch；它说明“非均匀输入值得解释”，不证明本合同或 60-epoch
公平比较有效。共享官方 AdaTAD baseline 仍由 ZoomToken 维护；DUCA 只读引用其 receipt，
绝不重复训练或用历史 65/66.xx 替代它。

## 1. 一句话问题与可证伪主张

在不改变官方 TAD 检测器、损失、NMS 或评估器的前提下，能否让低成本 scout 预测
动作性和起止边界的重要性，并以确定性端点覆盖规则选择少量**物理连续**视频片段、
按视频或窗口调节其数量，从而真实减少 VideoMAE 的重计算，同时不损害高 IoU 定位？

主张的反面同样明确：如果语义证据不能在相同总重型帧数、相同 detector 训练合同下优于
固定预算或其收益不能超过预算打乱控制，则不能把动态预算或 Query-Bridge 写成有效机制。

## 2. 冻结的模型因果链

```text
低分辨率 dense scout
  -> 0/1 动作性、起点、终点和不确定性预测
  -> （可选）Query-Bridge / cycle 只改善上述语义表征
  -> 确定性端点覆盖与上下文规则
  -> 每视频/窗口片段数 M（dynamic outer budget）
  -> M 个按原始时间连续的 16-frame cliplet
  -> 只对这些 cliplet 运行 VideoMAE
  -> 带物理时间坐标的稠密特征重建
  -> 不变的 AdaTAD / ActionFormer 检测、物理时间 proposal、官方 NMS 与 evaluator
```

1. scout 的可学习输出仅为逐时刻的 action/background、start、end 及不确定性预测；
   action/start/end 均由训练侧标注构造监督。它不是直接帧索引策略。
2. acquisition 必须是这些部署可见预测的确定性函数。Query token、cycle 或知识蒸馏不得
   直接产生帧 index、K/M、proposal、NMS score 或 test-time teacher/cache 信号。
3. endpoint score 只能解释为“端点覆盖证据”，不是已校准的事件概率。起点/终点重叠、
   多动作窗口和短动作必须由训练侧校准集评估，不能以独立性或概率乘积的叙述掩盖错误。
4. primary selector 从输出 M 个连续片段得到 `K=16*M` 个不重复的源帧。所有选择结果均记录
   原始 frame index、timestamp、cliplet id 和 cliplet 内 offset；物理时间在任何阈值、top-k、
   IoU 和 NMS 前已经可用。不得以选中序号伪装均匀物理时间。
5. `executed_k` 必须从真正送入 VideoMAE 的帧/cliplet工作量产生，禁止复制 requested K、
   padding 到 dense 768 或仅在 metadata 中声明动态预算。

## 3. 连续 cliplet 是主路径；GAPPACK 是解释性强控制

### 3.1 CONTIG（主路径）

每个 cliplet 都是原视频时间轴上连续的 16 帧。它保持 VideoMAE 预训练所期待的局部运动和
速度语义；非连续 cliplet 之间可存在物理空隙，但每个送入重型 backbone 的 16 帧内部不得
跳帧。重建模块可以根据 timestamps 把特征放回 768 点物理轴，但不能把 detector 训练的
时间几何改成 selected rank。

### 3.2 GAPPACK（历史输入方式的归因控制）

GAPPACK 复用 CONTIG 已确定的**相同原始 RGB 帧集合、相同 K、相同排序、相同 detector
状态和相同物理 timestamp 输出**，仅把这些帧按 selected rank 再组成 16-frame 输入块。它
复现历史 65.3857 的关键输入风险：VideoMAE 在一个输入块内看到不等间隔的真实时刻。

因此，`CONTIG vs GAPPACK` 的任何差异只能被解释为“对 VideoMAE 的时间呈现方式不同”，
不能归因给选择质量、预算、数据、损失、head、NMS 或坐标映射。GAPPACK 不是主方法，
也不能单独成为效率或论文增益主张。

## 4. 四臂语义门（固定 M=24，K=384）

四臂共享完全相同的 scout 输入、端点覆盖/连续 cliplet acquisition、detector、loss、NMS、
official evaluator、训练 update、seed、物理时间映射和 `M=24`。它们只改变 Query-Bridge
对 scout 语义的帮助方式：

| 臂 | 名称 | 允许的额外信息或损失 | 禁止事项 | 回答的问题 |
|---|---|---|---|---|
| S0 | semantic base | 0/1 action/start/end 与不确定性监督 | 无 query、cycle、蒸馏 | 纯语义 scout 是否足够形成可用间接采样？ |
| SQ | query bridge | class-agnostic query 与 scout 的交叉注意力 | query 直出 index/K/proposal | query 是否改善端点/动作语义？ |
| SQC | query + cycle | SQ 加训练期 detached post-heavy cycle 一致性 | cycle 参与推理或用 GT/teacher 选帧 | 前后协同是否改善 scout 语义而非绕过 selector？ |
| SQD | query + cycle + distillation | SQC 加训练期 detached 语义蒸馏 | 用 detector 输出直接决定 index/K；test teacher/cache | 蒸馏在相同机制下是否有独立价值？ |

所有 cycle/蒸馏 target 必须 detach、只来自训练样本并在 inference 完全删除。每个臂都需分别
报告 action/start/end 的训练侧校准和端点覆盖诊断；仅有 loss 降低、attention 图或非零梯度
不构成语义门通过。

## 5. 冻结的预算与选择合同

- 固定预算参照为 `M=24, K=384`。
- dynamic outer budget 必须使用一个有限离散的 M 集合，并包含固定参照 `M=24`。该集合、
  上下界和覆盖阈值只允许从训练侧 FIT/CAL 的可达性与成本证据中确定，并在 P3 的 PRE_RUN
  前写入有效 config 后永久封存；在它们封存前 P3 不得启动。外部审查提出的
  `{16,20,24,28,32}` 只是待检验候选，不因本合同而成为已批准常数。每个 video/window 的 M
  必须是 scout 语义证据的确定性函数，tie-break、时间坐标单位和边界处理随 config 一并冻结。
- 所有动态臂报告 requested M、实际 executed M、每个 M 的频率、视频长度/动作复杂度分层、
  与 actionness/boundary 证据的关系，以及完整成本；预期 K 不是执行证据。
- dynamic arm 必有两个强控制：
  1. **K-shuffle**：保持同一 M 直方图和长度分层，但随机置换 M 与视频的对应关系；
  2. **actionness-only dynamic**：关闭 boundary evidence，其余预算规则和真实工作量不变。
- direct index policy 仅在后续独立 ablation 中出现；不得替代 S0--SQD 或 dynamic 主臂。
- 已在版本控制记录中多次完成的 dense、uniform、random 训练不因本合同重跑；官方 dense
  由共享 receipt 只读绑定。新实验的强归因来自 S0--SQD、CONTIG/GAPPACK、fixed/dynamic、
  K-shuffle 与 actionness-only，而不是重复历史对照。

## 6. 分阶段实验与停止规则

### P0：身份、语义和执行门（无性能主张）

在 FIT/CAL 数据上绑定 canonical THUMOS14、官方 config/evaluator、VideoMAE-S 预训练、
effective config、scout 标签生成、cliplet timestamps、detector 不变量和完整 checkpoint
恢复。检查真正的 ragged/bucketed VideoMAE 输入以及 `executed_k`；不得使用本机 CPU 或
synthetic 结果宣称模型效能。P0 不读取 official evaluation GT，不打开 held-out，也不能用
中间 mAP 选 checkpoint。

P0 失败（例如语义 target 不可构造、物理连续 cliplet 没有进入重型执行、timestamp 在 NMS 前
丢失、训练/推理图不一致、恢复不完整）时，停止在实现层修复；不得以 padding、metadata
填充或训练更久绕过。

### P1：固定预算语义门

以 full official THUMOS14 训练/评估合同运行 S0、SQ、SQC、SQD 的固定 `M=24` 比较。
只有预先固定的一个语义 winner 可进入下一阶段；没有稳定优于 S0 的结果时，Query/cycle/
distillation 不进入动态预算主臂。该阶段的目的不是重新证明 dense/uniform/random，而是隔离
语义协同训练的价值。

### P2：同帧集合的时间呈现归因

用 P1 winner 或 S0（若 P1 无 winner）的选择结果，比较 CONTIG 与 GAPPACK。两臂必须校验
原始 frame-id multiset 和 K 严格相同。若 GAPPACK 显著损害高 IoU 或边界定位，则主路径只保留
CONTIG；若不损害，也只能说明该历史混杂未主导，不能把 GAPPACK 升格成默认主张。

### P3：dynamic outer-K 的唯一效应

在 P1/P2 通过后，比较：固定 M=24、语义 dynamic M、K-shuffle、actionness-only dynamic。
所有臂以 matched realized mean K 和完整链路成本报告；detector、loss、NMS、数据、split、
updates 和 seeds 不变。只有 dynamic M 相对 fixed M 与 K-shuffle 同时显示可重复、可解释的
优势，且高 IoU 未受不可接受损伤时，才保留“语义驱动动态预算”主张。

任何阶段的负结果只拒绝该阶段的具体机制，不把“DUCA 的间接语义采样”整体判死。不得在得到
负结果后静默改变 M 范围、loss、时间呈现、split、evaluator 或 checkpoint 规则；新机制须另立
合同。

## 7. 公平性、数据隔离、成本与 checkpoint

1. FIT、CAL、HOLD 的视频清单必须互不重叠且在运行前绑定；official evaluator 的目标集合
   不得用于阈值、M、loss、模型或 checkpoint 选择。THUMOS canonical 411、annotation 和
   category map 使用项目资源地图已验证的入口。
2. baseline、S0--SQD、CONTIG/GAPPACK 与所有 dynamic controls 使用同一 detector、loss、
   optimizer update、LR、seed、增强、NMS、evaluator、final/final-EMA model-selection rule。
   若 shared official AdaTAD receipt 尚未返回，official dense 数字保持空白。
3. 每 5 epoch（若官方 recipe 更频繁则沿用官方间隔）保存可恢复 `.pth`；保留 latest-3、
   预定义 milestone、final 和 final-EMA。恢复必须还原 model、optimizer、scheduler、AMP
   scaler、epoch/update、Python/NumPy/Torch/CUDA RNG 及 DataLoader 所需状态。恢复点仅用于
   故障恢复和诊断，禁止事后选择最优 checkpoint。
4. 成本至少分解低分辨率 scout、decode/materialization、CPU transform、H2D、VideoMAE、
   feature reconstruction、detector、NMS 与总 wall-clock；只减少 backbone 输入不等于端到端
   成本下降。

## 8. 明确不在本合同内

- 不把 density decoder、prefix-budget、旧 local transport、UVT V(t) 或 Fovea 的直接
  Gumbel-TopK 作为主机制；它们可留作历史诊断，不得混入新主臂。
- 不改变 ActionFormer/AdaTAD detector head、assignment、detector loss、NMS 或 official
  evaluator 来补偿采样损失。
- 不使用 GT、teacher、cache、raw prediction、cycle 或 EMA target 参与推理选择。
- 不以 local CPU、static test、subset 或 160/40 开发结果作为 mAP 或成本结论。

## 9. 进入实现的最小交付与下一棒

新 clean Builder candidate 必须先提交一份实现计划，明确：真实 cliplet materialization 的
调用点、ragged/bucketed heavy path、physical-coordinate tensor 从采样到 pre-NMS 的链路、
S0--SQD loss/gradient 所有权、fixed/dynamic/K-shuffle 配置、FIT/CAL/HOLD manifests、
checkpoint/resume 和 launcher。随后冻结 snapshot 交由独立 Critic 和 DSH 审查；只有两者均
确认实现忠于本合同，Evaluator 才能执行 PRE_RUN。PRE_RUN 通过后，按 P1→P2→P3 顺序启动
真实 N16R4 full runs；没有单独的人类许可轮。

当前下一棒是 **Builder：在干净 DUCA base 上给出上述最小实现计划，尚不提交训练**。
