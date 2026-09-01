---
updated: 2026-07-10
status: active
scope: 吸收 ResearchClaw 原则下对 DUCA 70aa069 的代码、成本、几何和研究路线审查
out-of-scope: 把外部评审直接当作实验事实；宣称 CVCR/BCFT/CoDeTAD 已实现或已优于 DUCA
---

# DUCA ResearchClaw Divergent Audit Absorption

原始回复完整保存在：

`docs/methods/reviews/2026-07-10-70aa069-researchclaw-duca-divergent-audit-raw.txt`

原始附件 SHA256：

`E4344DAED297F02E23CE355A4B0BBA1845F2C05393820CE04843374AAB6A59AC`

审查对象：`70aa069b895322c2307ffbb13dfdef9fac0d1305`。本记录形成时本地分支还包含仅用于设计文档的 `8abc521`，模型代码与 `70aa069` 一致。

外部 Skill 固定版本：`MikaelCool/My-Own-PhD-Students@e146de166d95475a41d9216bbd605cec8fcb330a`。

## 吸收总裁决

**不完全同意“DUCA 主线应立即停止、CVCR 必须取代 DUCA”的最终裁决。**

更严格且符合证据的结论是：

> `70aa069` 只能冻结为待裁决的完整 DUCA baseline。它尚不具备主论文方法的证据闭环；必须先完成成本、离散 utility 对齐和几何三类决定性审计。审计失败时降级或停止，审计通过时才允许继续作为主方法候选。

这不是继续盲目堆模块。当前冻结新增 selector head、loss 和动态预算，优先验证核心假设。

## 完全同意并已由代码核验的部分

1. **任务语义是离线全窗口、forward 内即时生成的 pre-backbone 选择。** 它不是 causal、streaming 或 Online TAD。后续论文和正式合同应使用 `offline_full_window`、`runtime_generated`、`cache_free`、`jointly_trained`。
2. **真实成本必须覆盖完整链路。** decode/preprocess/H2D/probe/selector/backbone/head/postprocess、p50/p95、峰值显存和能耗必须统一测量；FLOPs 不能替代实测 latency。
3. **旧内部 profiler 明确漏记正式 bridge。** `structured_zero_forward` 在 raw video tensor 上执行 `[B,K,T]` soft slot assignment，但 `_add_detector_gradient_bridge_profile()` 只把 `soft_to_hard_resample` 记为 enabled，并错误使用 descriptor 的 `feature_dim=3`。因此旧静态 FLOPs 账本不能作为论文证据。
4. **现有 nonzero-gradient 测试只证明图连通。** `out["inputs"].square().mean().backward()` 不能证明真实 ActionFormer detector loss 给出了与 hard frame swap 一致的有效梯度。
5. **当前 utility 是 GT boundary-utility proxy。** start/end/context 和实例归一化边界目标不能被称为真实 detector utility、counterfactual gain 或 marginal value of compute。
6. **selected-axis 存在真实几何风险。** post-hoc `TrueTimeMap` 能恢复输出坐标，却不能自动恢复卷积、FPN stride、感受野和 regression range 对不等物理间隔的语义。
7. **强 baseline 和泛化证据缺失。** exact-uniform、periodic、stratified random、简单 residual swap，以及第二 detector/第二数据集都是主张插件价值前必须补齐的证据。
8. **MUST 暂不能作为主贡献。** expected budget 不等于实际执行 token 数，更不等于 kernel latency；动态预算必须按真实 K 和实测成本报告。

## 部分同意、但需实验裁决的部分

1. **full-window probe 的 compute circularity 是风险，不是逻辑上的致命错误。** 本项目原始目标是减少昂贵 VideoMAE/AdaTAD 后段计算，并未承诺减少 bitstream decode。若低分辨率 probe 与 selector 的总成本远小于被节省的 heavy-backbone 成本，方案仍可能有系统价值；只能由完整成本 trace 判断。
2. **selected-axis 风险不能直接等同于方法必然失效。** 标准 ActionFormer 把不等间隔样本当等间隔处理确实有模型错配，但其实际影响大小仍需 same-selected-frames 对照证明。用户已要求暂缓 physical-grid 实现，因此当前只保留为明确的未解决风险和后续裁决实验。
3. **多头多 loss 的可归因性批评成立，但“最多只能保留一个 value head”的配方不是定理。** 应通过最小化消融选择必要模块，而不是仅按形式数量删模块。
4. **联合训练可能产生目标冲突，但不应因此回退为三套独立模型。** 正确做法是报告 probe-only、proxy-only、detector-only、joint 的梯度范数/余弦和对照结果，证明联合训练是否真正有增益。

## 不同意或证据不足的部分

1. **不同意在三个决定性实验之前宣布“DUCA 主线停止”。** 评审自己也把这些实验定义为决定性实验；在结果产生前直接停止，是结论先于证据。
2. **不同意把 CVCR-TAD 视为已证明更优的主路线。** CVCR 同样面临 counterfactual teacher 成本、time-layer routing 的 novelty collision、packed kernel 实现、跨 head 泛化和真实速度收益等未验证问题。它是值得验证的新假设，不是当前事实。
3. **不同意把 full decode 等同于没有部署价值。** 对 GPU-heavy end-to-end TAD，decode 与低分辨率 probe 可能不是主成本；必须看同硬件实测占比。
4. **不接受未经独立 citation/novelty audit 的“CVPR 新颖性”结论。** 原报告明确承认没有完整加载固定 Skill 仓库，也未完成整仓逐字审计；其文献碰撞矩阵只能作为待核验线索。

## 当前执行约束

### 立即执行

1. 完成统一 full-stack cost profiler，覆盖完整数据通路和所有非重叠模型阶段，并对成本守恒 fail closed。
2. 用相同 commit、硬件、输入、batch 和 warm-up 比较 dense-768、exact-uniform-384、periodic-384、DUCA-384。
3. 修正或废弃旧内部 `structured_zero_forward` 静态 FLOPs 记账，训练期 raw-pixel bridge 必须单独报告。
4. 实现 one-swap finite-difference audit，比较 ST gradient、GT boundary proxy、actionness、transition、feature drift 和 random。
5. 保留现有 selector geometry 分析，报告边界覆盖、端点距离、max gap、聚集偏移和动作时长分层结果。

### 当前不执行

1. 暂不实现 physical-grid/continuous-time head；仅保留设计与待裁决实验合同。
2. 不继续扩展 X3D 密集 prior，不把 SlowFast/X3D 作为主方法。
3. 不新增 selector head、辅助 loss 或 dynamic MUST 主张。
4. 不在正式成本结果和 one-swap 对齐结果前切换到 CVCR/BCFT/CoDeTAD 全量实现。

## 决策门槛

DUCA 至少需要同时满足：

- 相对 exact-uniform 有明确且可重复的 mAP 或高 tIoU 增益；
- full-stack p50/p95 latency 和 energy 有实际改善；
- probe+selector 不吞掉大部分 heavy-backbone 节省；
- surrogate 与 hard swap utility 达到正相关并显著优于 random；
- 高 tIoU、短动作和边界误差不出现系统性退化；
- 结果能迁移到至少第二 detector，或论文诚实收缩为 AdaTAD-specific 方法。

任一核心门槛失败时，不再通过继续调 loss 权重挽救主张，而是按失败类型选择：降级为 baseline、移除 ST bridge、改为 simpler residual selection，或转向 CVCR/BCFT 等新路线。

## 与当前成本代码的关系

本轮新增的 full-stack profiler 正面处理了评审的成本审计要求：采用离线全窗口协议，实测 input pipeline、H2D、selector、probe、heavy backbone、projection、neck、head、postprocess、显存和 energy，并对阶段嵌套与总成本守恒 fail closed。

它不修复 selector 本身的 surrogate、proxy 或 selected-axis 几何问题，也不能用 random-init smoke 充当论文结果。正式结论仍需训练 checkpoint 下的同硬件矩阵。
