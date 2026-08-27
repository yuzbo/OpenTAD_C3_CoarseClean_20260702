# DUCA 纯前置选帧插件、官方基线与总 60 轮训练恢复合同

## 状态与作用

- 记录日期：`2026-07-27`
- 状态：`design_user_approved_initial / scientific_review_major_revision /
  baseline_audit_required / final_model_not_implemented`
- 任务：始终是离线时序动作检测。
- 本文件恢复并统一 2026-07-06 至 2026-07-26 多轮 Pro 原始审阅中的方法初心。
- 本文件不删除旧路线。它明确取代 `2026-07-27 16:00 +08:00` 将“物理时间检测头”
  设为下一主模型的临时判断。真实时间检测接口只保留为诊断或增强版，不再定义论文主方法。

## 已完整复核的本地原始讨论

复核对象是原始全文而非后来的短摘要，主要包括：

1. `docs/methods/reviews/2026-07-06-duca-5f87baf-pro-review-raw.txt`
2. `docs/methods/reviews/2026-07-07-duca-tad-cvpr-pro-paper-review-raw.txt`
3. 2026-07-08 的五份 online/plugin、真实 detector、免训练 X3D 与最终方法原始审阅
4. 2026-07-09 的 transition-first、HOLD 与 THUMOS 任务有效性三份原始审阅
5. `docs/methods/reviews/2026-07-11-a5e1774-duca-transition-only-pro-review-raw.txt`
6. 2026-07-13 的 FSU、CellCF 与 dense-time spatial-zoom 原始审阅
7. 2026-07-15 的 exact-commit 与 S1 原始审阅
8. 2026-07-16 的物理时间、Round-1 代码数学与 Round-2 方法论文裁决
9. 2026-07-19 至 2026-07-20 的 CellCF/CARA、protected-E2E、physical-allocation
   与 pre-backbone 论文准备度原始审阅
10. 2026-07-21 的 V8 与两份 two-stage curriculum 原始审计
11. 2026-07-22 的 Uni-AdaFocus/EU-CRR 与 R0-R5 原始审阅
12. `docs/methods/reviews/2026-07-23-a00498e-duca-selected-axis-ttdi-pro-review-raw.txt`
13. `docs/methods/reviews/2026-07-26-duca-multiround-joint-review-raw.txt`
14. `docs/methods/reviews/2026-07-27-duca-total60-prebackbone-pro-review-raw.txt`

这些意见并不完全一致。早期意见反复冻结“前置插件、不改检测头”；后期部分意见为修复
不规则时间轴而建议物理时间检测头。后者是一个候选补救，不是从一开始就达成的主方法共识。

## 恢复后的论文方法身份

DUCA 主方法应当是一个位于重型主干网络之前的智能选帧插件：

1. 输入低成本、部署时可见的时序证据。
2. 在长度为 `T` 的候选时间网格上输出有序、唯一、严格 `K` 个位置。
3. 重型主干网络只处理这 `K` 个位置。
4. 主实验不修改后续检测器的主干、投影、检测头、标签分配、分类/回归损失或 NMS。
5. 允许插件在训练期接收检测任务监督；推理期不得读取真值、教师、预测缓存或决策台账。
6. 允许在检测器外侧使用同一套通用单调坐标映射，把重参数化时间上的预测映回原视频时间。
   这必须是检测器无关的输入/输出适配，不能成为 AdaTAD 专用检测头。

因此，向投影层、检测头、标签分配或回归分支注入真实时间坐标，会把工作变成“稀疏检测器
重设计”。它可以作为增强版或诊断，但不能支撑“纯可插拔前置选帧插件”的主张。

## 原始“总计 60 轮多课程”合同

历史上确实讨论并实现过一种**检测器总预算只有 60 轮**的多阶段课程，而不是当前的
“30 轮完整检测器训练 + 60 轮完整联合训练”。

历史意图为：

- 可选的前端预训练只训练低成本粗模型和 selector，检测器跳过；它不是 detector update。
- 正式检测器训练始终只有 6,000 次成功更新，约 60 轮。
- 更新 `0--999`：严格均匀 `K=384`，selector 不改变检测器输入。
- 更新 `1000--2499`：平滑释放学习式采样率或有界残差。
- 更新 `2500--3999`：逐步加入贡献监督与经过验证的检测反馈。
- 更新 `4000--5999`：固定最终机制并稳定收敛。

具体权重和边界曾因隐藏损失、优化器隔离、粗模型语义污染和代理梯度方向未验证而被判
`HOLD`，不能原样复用；但“所有比较臂只使用同一 6,000 次 detector update”的科学原则
没有失效。

当前 `K=384` 课程的 Stage-1 确实训练了完整检测器 30 轮，Stage-2 又训练完整检测器
60 轮，所以：

- 终点 `65.385724%` 是 90 轮 over-budget 候选，不是公平的 official-60 主结果。
- Stage-2 epoch 50 的 `65.650497%` 是 80 轮 best-observed 诊断，也不是 60 轮结果。
- 允许选择中间最优 checkpoint，但所有比较臂必须使用相同最大训练预算、相同验证频率和
  相同选择规则。不能用 80/90 轮候选对比 60 轮均匀基线。
- 必须在一个总 60 轮预算内比较“从零联合训练”与“均匀预热后释放 selector”，才能回答
  Stage-1 是否必要。

## 官方 AdaTAD 基线复核

官方来源为 `sming256/OpenTAD@1aa8ca4ac5e846b1e8ff69298dd6607121a01589`。
官方 `configs/adatad/README.md` 对 THUMOS14 的 VideoMAE-S、768 点、160 分辨率
AdaTAD 报告：

```text
mAP@0.3/0.4/0.5/0.6/0.7 =
83.90 / 79.01 / 72.38 / 61.57 / 48.27
Average-mAP = 69.03
```

因此项目历史 `68.29%` 只能称为本地 dense 锚点，不能替代官方 69.03%。目前还没有证据
证明该差值来自随机波动、checkpoint 选择、权重、学习率日程或代码改动中的哪一项。

当前 `64.49%` 的 exact-uniform K=384 也不是“完全原始 AdaTAD 只做 1/2 均匀下采样”：

- 它使用 DUCA 扩展后的 `ActionFormer/SingleStageDetector` wrapper。
- 它从 768 点选择 384 点，并在 selected-axis 上重映射真值，再把预测逆映射回原时间。
- 模型长度从 768 改为 384，chunk 数从 48 改为 24。
- `with_cp/static_graph/find_unused_parameters` 和学习率日程与官方配置不同。
- 名义上的 `ActionFormerHead` 配置、投影结构、分类/回归目标与 NMS 参数仍从官方
  配置派生。
- 但 exact commit `42dba3f90b37243e7965d18b6707e88e81bf7109` 中的
  `ActionFormer` 与 `AnchorFreeHead` 源文件均已扩展，并不与上游逐字节相同。
  即使当前 selected-axis 课程关闭了部分 physical-grid 分支，也尚无等价性证据
  可以排除这些扩展对性能的影响。

这解释了为什么 `64.49%` 只能作为当前 DUCA wrapper 的 matched uniform control，不能
证明官方原生 1/2 均匀采样只能达到约 64%。

下一步必须分开建立三条基线：

1. **官方 dense**：先直接评估官方发布权重，确认 69.03% 的评估口径；再做干净复现。
2. **官方原生 1/2 均匀采样**：从干净官方提交出发，只把原始候选网格的采样步长改为 2，
   令 detector 输入为规则 384 点并覆盖与 dense 768 相同的真实时间范围；不加载 DUCA
   selector，不做 irregular selected-axis 重映射，不改 detector class/head/loss/NMS。
3. **DUCA-wrapper exact-uniform**：保留当前 64.49%，但明确标为坐标包装对照。

官方原生 25% 均匀采样也应以相同方式建立，作为当前 K=192 课程的必要对照。

## 当前模型性能受限的主要原因

按科学优先级排序：

1. **基线尚未对齐。** 69.03%、本地 68.29%、原生 stride-2 与 DUCA-wrapper 64.49%
   被混在一起，导致增益起点不可信。
2. **训练预算混杂。** 当前正结果来自 80/90 轮，而基线是 60 轮。
3. **时间重参数化失控。** 非均匀位置被后续检测器当成规则 rank-time；当前密度没有一个
   明确的局部伸缩或累计扭曲上界。
4. **贡献监督过重且不稳。** K=192 的分类/回归贡献项长期约为 5--6，远大于检测
   cls/reg 项；原始 `|x * grad|` 幅值可能更像梯度尺度而非稳定的帧排序价值。
5. **检测梯度只是连续代理。** 它证明连通，不等于能预测真实离散换帧后的检测收益。
6. **粗模型和完整检测器共同适配过强。** full ASFormer adaptation 与 detector
   co-adaptation 会削弱跨检测器可插拔性，也让收益难以归因于 selector。
7. **机制过多。** 采样率、贡献蒸馏、检测梯度、粗模型适配和额外训练轮数同时变化，
   当前 `+0.896pp` 不能归因到一个清晰的新理论机制。

## 主模型的理论恢复方向

主方向不再是修改检测头，而是**检测器兼容的有界单调时间重参数化**：

```text
低成本状态变化证据
  -> 正的时间密度 rho(t)
  -> 累积分布 F(t)
  -> K 个固定分位点 F^{-1}((j+0.5)/K)
  -> 有序唯一整数位置
  -> 原样的 TAD detector
```

需要同时满足：

- `rho(t)` 为常数时严格退化为均匀采样，均匀是模型内部的 identity path。
- 采样数严格等于 K，位置单调、有序、唯一。
- 对累计质量偏移或相邻位置间隔施加可解释上界，限制 rank-time 与物理时间的局部伸缩。
- 训练目标优先学习**排序与质量搬运**，不直接拟合未经校准的贡献绝对幅值。
- 检测梯度仅在真实 hard-swap 排序一致性通过后启用；否则只保留 detached rank teacher。
- 粗模型尽量冻结或只允许极小的末层适配，使插件可以迁移到不同检测器。

论文理论问题由此变成：

> 在固定预算下，如何学习一个包含均匀映射、具有有界时间扭曲的任务感知单调传输，
> 使任意后续 TAD 检测器在不改结构的情况下保留高重叠阈值定位能力？

需要进一步给出一个可证明命题：当单调映射的最大累计偏移和局部伸缩受限时，规则
rank-time 检测器看到的时间度量误差具有上界。该命题目前是 `designed`，不是已证明结论。

## 最小模型优先实验

在任何大规模种子或工程完善前，只做能决定模型的实验：

1. 恢复官方 dense 69.03 与原生 50%/25% 均匀基线。
2. 在严格总 60 轮、同一选择规则下比较：
   - 原生均匀采样；
   - 有界密度传输，从零训练；
   - 前 10 轮均匀预热、后 50 轮释放传输；
   - 上一臂加入归一化的分类/回归贡献排序蒸馏；
   - 只有 hard-swap 对齐通过时，再加入检测梯度。
3. 第一粒种子若不能显著超过干净均匀基线，不先做多种子、第二检测器或广泛工程矩阵，
   而是回到时间扭曲、teacher fidelity 与密度塌缩分析。
4. 主模型单粒种子出现明确提升后，再做两次独立运行和第二检测器，形成 CVPR 所需的
   稳定性与可插拔证据。

如果官方原生 50% 均匀基线确实已达约 65%，则当前 65.3857% 的 over-budget 课程几乎
不能证明 selector 有效。论文目标应是至少在同一 60 轮预算下显著超过干净 50% 均匀基线，
并尽量接近官方 dense 69.03%。用一半帧达到 69--70% 是强目标和潜在 headline，但在干净
基线未恢复前不能把它当成已接近的结果。

## 免训练、即插即用方向

必须区分两个层级：

1. **目标域免训练前端**：selector 不在 THUMOS 上训练，但后续 detector 仍在 THUMOS 上训练。
   当前实验属于这一层。
2. **真正即插即用**：selector 和后续已训练 detector 都不再针对目标数据重训，只在输入前
   插入选择器并直接评估。当前没有完成这一证据。

已有终端结果均为第一层，且没有超过同协议均匀采样：

- SlowFast Fast-only：`63.5297%`，比 `64.49%` 低 `0.9603pp`。
- MobileNet 特征变化：`63.27%`。
- MobileNet 语义：`62.78%`。
- 固定融合：`64.33%`，约低 `0.16pp`。
- actual-time residual T1：`64.0200%`。

所以免训练方向已经“实现并得到负结果”，不是尚未做；但真正 frozen-detector
plug-and-play 仍未验证。下一版不再用成本很高的 SlowFast 作为主方案，而应复用同一有界
密度解码器，仅使用便宜的帧差、边缘/压缩域变化、低分辨率冻结图像特征或固定状态变化融合。
它是主方法的 training-free mode，不与 task-adapted 主结果混为一谈。

## 当前进度的最终判定

| 项目 | 当前状态 |
|---|---|
| K=384 当前 30+60 课程 | 已完成，65.385724%，但为 90 轮 over-budget 候选 |
| K=384 公平总 60 轮最终结果 | 未获得 |
| 官方 dense 69.03 复现 | 未闭环 |
| 官方原生 50% 均匀基线 | 未闭环 |
| 当前 DUCA-wrapper 均匀基线 | 已完成，64.49% |
| K=192 当前 30+60 课程 | 已完成，57.967272%，但为 90 轮 over-budget 终端诊断 |
| 官方原生 25% 均匀基线 | 未开始/未闭环 |
| 纯前置有界密度主模型 | 仅设计，尚未实现 |
| 目标域免训练前端 | 已测试，当前为负结果 |
| 真正 frozen-detector 即插即用 | 未验证 |
| CVPR 最终模型 | 尚未完整实现，也没有公平主结果 |

## 冻结结论

1. 当前 30+60 双阶段不是最初讨论的总 60 轮多课程结构。
2. 69.03% 是官方 dense 基准；68.29% 是本地历史锚点。
3. 64.49% 是 DUCA-wrapper matched uniform，不是干净官方原生 1/2 均匀采样。
4. 主论文继续定义为纯 pre-backbone 智能选帧插件；检测头物理时间改造降级为诊断/增强版。
5. 最终理论创新聚焦于“均匀可退化、exact-K、有界时间扭曲的任务感知单调密度传输”。
6. 免训练前端已有负结果；真正冻结检测器的即插即用路线仍是开放问题。
7. 在官方基线、总 60 轮和主模型机制闭合前，禁止声称 DUCA 最终版本已经完成。

## 2026-07-27 总计 60 轮 Pro 大修裁决

原始审阅和项目逐项吸收记录位于：

- `docs/methods/reviews/2026-07-27-duca-total60-prebackbone-pro-review-raw.txt`
- `docs/methods/2026-07-27-duca-total60-prebackbone-pro-review-absorption.md`

项目裁决为 `major_revision_accepted_with_corrections`，不是逐字全盘接受。

完全吸收：

1. inverse-CDF、累计质量和学习选帧都不是单独的新颖性；
2. 必须形成唯一的 `e -> p -> F -> y -> S` exact-K 数学合同；
3. clean native uniform 与 wrapper parity 是任何 selector claim 的前提；
4. 非均匀选帧使用对称 warped-time I/O，raw proposals 必须先逆映射到物理时间再做
   参数不变的官方 NMS；
5. `G_rank` 与 `G_direct` 分开，分别裁决贡献教师和直接检测梯度；
6. 第一枚 development seed 排除出最终统计，正式结论使用未参与开发的预登记种子；
7. 69 至 70 保留为强目标，不是信息上界或允许不公平训练的硬门。

修正后吸收：

1. 审阅提出的 `1/(2T)..2/T` 密度界、`4/K` 累计偏移、DP 间隔/anchor 常数是首个
   候选，不直接冻结；
2. DP 是实现候选，不是论文贡献；禁止的是解码后未登记 repair，而不是数学合同内部
   的确定性冲突处理；
3. RDD 是优先候选，不是未经验证的唯一监督；`G_rank` 失败杀死当前连续梯度教师，
   不逻辑性否定所有可能的硬 counterfactual utility 学习；
4. terminal EMA 是没有独立训练侧留出集合时的主规则；若存在严格无泄漏留出集合，
   所有臂可以使用相同中间 checkpoint 选择规则；
5. reviewer 的 `+1.0pp`、40% dense-gap、第二检测器 `+0.5pp` 和成本阈值均为
   `designed_reviewer_proposal`，待 clean baseline 和功效模拟后、正式结果前冻结；
6. PR #3 的 aggregate diff 必须清理，但不进入模型科学关键路径；
7. frozen-detector train-free 模式在共享解码器与坐标合同通过后可并行建立最小基线，
   不与 task-adapted 主结果混称，也不无限延期。

审阅成稿后的事实更新：

- K=192 已得到 terminal epoch-59 EMA official Avg-mAP `57.967272%`，不再是中间值
  证据；但仍缺 clean K=192 uniform，且使用 90 轮完整模型训练。
- 其后续质量诊断显示局部 action/boundary enrichment 增强，同时 R4Q5 宽双侧边界支持
  和最大洞恶化。这支持“当前 selector 更偏局部动作富集、未保护宽边界对”的机制假设，
  但不能替代配对均匀终点和注册消融。
- 公平 total-60 bounded model 仍为 `implementation_not_started`。

## 2026-07-27 用户批准后的执行规范

用户已经批准官方基线恢复、纯前置有界密度传输、总计 60 轮五臂比较、
时间扭曲与贡献监督消融、离散换帧准入门、多种子/第二检测器和真正
frozen-detector 即插即用验证。

“一进一出单帧交换”只作为连续代理的最小局部有限差分，不能独自证明
完整策略有效。正式准入同时要求：

1. 单帧交换；
2. 约 1%、5%、10% 的分散多帧交换；
3. 相同比例的连续片段交换；
4. 沿连续代理方向做 0.25、0.5、1.0 强度的全局密度步进并重新硬解码。

只有局部和多帧层都与真实检测损失收益正对齐，直接检测梯度才允许进入
最终 60 轮模型。完整规范为
`docs/superpowers/specs/2026-07-27-duca-total60-prebackbone-plugin-cvpr-design.md`，
对应实验节点为
`research-wiki/experiments/duca-total60-prebackbone-plugin-cvpr.md`。

## 2026-07-27 dynamic-K / AdapTok 接管回复后的合同边界

新增原始审阅与项目裁决：

- `docs/methods/reviews/2026-07-27-duca-dynamic-k-adaptok-research-takeover-raw.txt`
- `docs/methods/2026-07-27-duca-dynamic-k-adaptok-research-takeover-absorption.md`
- `research-wiki/ideas/duca-rime.md`
- `research-wiki/experiments/duca-dynamic-k-rime-oracle.md`

用户要求动态 K 作为候选论文中心，项目已正式记录；这会重新打开“最终 paper main
structure”，但不推翻本合同的测量、纯插件、total-60、hard-utility 和物理时间前提。
固定 K 有界 exact-K 仍是任何动态模型必须先通过的内层因果锚点，也仍是 dynamic
Oracle 和论文 fallback。

接管回复不是冻结规范。其短版要求各 K 独立解码，扩展版却要求 strict nested ladder。
当前唯一批准的设计动作是先在 train-only hard evidence 上比较 independent、strict
nested 与一次 weak-overlap 备选，并在正式训练前冻结一个 decoder family。不得依据
official test 结果切换。

因此当前状态更新为：

```text
pure_prebackbone_identity = retained
fixed_k_inner_contract = retained_as_gate_and_baseline
dynamic_k_paper_role = reopened_required_candidate
rime_name_and_nestedness = discussed_not_frozen
implementation = not_started
long_training = not_authorized
```

clean dense/native uniform/wrapper parity、raw `q -> t -> official NMS`、dynamic
Oracle、decoder-family regret、`G_rank` 和 pair-risk 先于任何完整 dynamic-K 60 轮
训练。

## 2026-08-11 P0 density semantics clarification

Fresh Project Pro decision `PRO_P0_ROUTE_ADJUDICATION-v002` refines the
pre-backbone plugin contract without changing detector structure, detector
losses, NMS, evaluator, data split, or budget. The route is now explicitly
`DUCA_FIXEDK_BOUNDED_DENSITY_QUANTILE_ACQUISITION-v002`: a dedicated per-time
`duca_density_logits[b,t]` reader consumes dense `browser_memory` on an identity
physical candidate grid (`selection_unit=1`), then
`decode_duca_density_positions_v001` decodes fixed requested K=384 by positive
trapezoidal mass, endpoint-inclusive inverse-CDF quantiles, and a deterministic
constrained integer projection. Exact constant logits are a bit-identical call
to the canonical integer-half-up uniform generator; legacy allocation,
actionness, rank/top-k, quota and transport signals are not valid density
aliases.

The correction keeps the strict-plugin boundary: selected-q proposals are mapped
exactly once to physical-dense time before filtering, top-k, IoU and unchanged
official NMS. The route remains `designed / BLOCKED_PRE_RESULT`: no patch, test,
data access, compute, metric, or performance claim is authorized by this
clarification.

## 2026-08-12 P0 nonconstant projection policy

Fresh Project Pro decision `PRO_P0_PROJECTION_POLICY-v001` keeps `DUCA_FIXEDK_BOUNDED_DENSITY_QUANTILE_ACQUISITION-v002` and freezes only its previously unspecified nonconstant hard decoder. After exact binary64-to-fixed point (`Q=2^20`) target conversion, it selects the unique feasible integer sequence under the exact lexicographic order `(E2, E_inf, E1, U1, position vector)`. The feasible set keeps endpoints, stride 1–4 and uniform displacement at most 16. The conforming decoder is an exact DP/shortest-path solver with typed fail-closed behavior; no clipping, deduplication, heuristic, tolerance, legacy selector or uniform fallback is permitted for nonconstant inputs.

This is only a method-definition decision. P0 requires cross-implementation identity for matching serialized `(T,K,u,a)` projector inputs and remains `BLOCKED_PRE_RESULT` until an independent reference, exhaustive witness, fixed fixtures, exact certificates and Critic independence audit are later authorized and returned. No execution, result, cost, performance claim or paper claim is admitted by this policy.
