# `SCIENTIFIC_DECISION: CONTINUE`

**Nonce:** `DUCA-ALL-MATRIX-SAME-BUDGET-v001-20260824`

这里的 `CONTINUE` 只有一个含义：

> **继续并仅继续 First-Mixing SingleClock Unit-1 的 Gate-v2 既有证据闭环；不授权重训，不授权 Query，不授权 Bridge，不授权 dynamic-K，也不授权任何新的性能实验。**

当前 SingleClock 的**效能裁决尚未合法形成**。正式状态必须写为：

```text
EVIDENCE_ADMISSION_BLOCKED
SCIENTIFIC_EFFICACY_DECISION = NOT_FORMED
```

这不是 `KILL_SINGLECLOCK`，也不是 `PASS_SINGLECLOCK`。原因不是指标尚未“足够好”，而是当前远端可审查代码、运行根和最终裁决器仍不能组成冻结 Gate-v2 所要求的合法证据。

最关键的事实是：

1. 远端 `b2ccfcc…` 主要补强了 selected positions 的范围与严格递增审计，并不等于本地未推送的 Gate-v2 两个提交。
2. 远端终结器仍把 `ON−gate-zero`、`+0.50 pp`、CI、cost 和旧 strata 作为硬门；当前冻结科学门却要求 **terminal EMA `ON−H65 OFF/replay` 的 Avg、@0.6、@0.7 三个 point delta 均 `>=−0.20 pp`，CI 只报告**。两者不是同一个科学问题。
3. 远端身份文件只覆盖 selected indices、selected RGB 和 selected mask；没有完成所需的 VideoMAE input、raw selected-q proposals/scores/labels、canonical evaluator JSON 五边界身份。
4. Unified evaluation Job `1253090` 在本轮材料中仍未形成合法 Gate-v2 终态；不得把已出现的四组指标提升为正式 Unit-1 决定。

---

# 一、总裁决：论文问题、机制与反主张

## 1. 论文问题

> 在 H65 完全相同的 K=384 非均匀 RGB 帧、30+60 训练暴露、VideoMAE-S、AdaTAD/ActionFormer、损失、NMS 和物理时间回映合同下，向第一次 Transformer 自注意力注入最小的真实时间残差，能否至少不损害 H65 的高 IoU 定位？

这里必须把名称修正为：

> **First-Transformer-Mixing SingleClock**

而不是泛称“在第一次重型时间混合前”。当前代码先执行 Conv3D tubelet patch embedding，再在 Transformer block 0 的 attention 中加入残差。因此，它不能修复 tubelet embedding 已经把两个 selected-rank 相邻帧当作局部连续输入进行卷积混合的问题。VideoMAE 的 patch embedding 明确使用时间核和步长均为 2 的 Conv3D；SingleClock 只在第 0 个 Transformer block 启用。

## 2. 当前机制主张

唯一可检验主张是：

> **对 H65 已选中的同一组 K384 RGB 帧，使用全局物理位置相对 canonical-uniform 的残差，在 block-0 attention 中恢复部分真实时间几何，可以在不增加训练轮次的条件下保持 H65 性能，尤其不显著损害 mAP@0.6/@0.7。**

## 3. 反主张

即使 Unit-1 通过，也不得声称：

* SingleClock 提升了 H65；
* SingleClock 已解释 H65、UVT 或 Fovea 的全部性能差距；
* selected-rank pseudo-contiguity 已被彻底修复；
* 真实时间已进入 Conv3D tubelet mixing；
* Query residual 有效；
* dynamic-K 有效；
* DUCA 已超过 dense AdaTAD；
* 当前结果已是论文证据；
* 一个 seed 的非劣结果就是可发表贡献。

---

# 二、全模型矩阵裁决

| 矩阵                                                 | 原始问题与设计判定                                                                                             | 实现判定                                                                                                                                                                                                                     | 结果能支持什么 / 不能支持什么                                                                                                                                                                 |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Official dense AdaTAD**                          | 合理地提供 T768 全重计算背景上限；从一开始就不是本地 DUCA 的同提交因果基线。                                                          | 官方提交确实是 AdaTAD release；公开 VideoMAE-S THUMOS14 anchor 为 69.03。                                                                                                                                                            | 共享 Job 1245842 的 68.73 只能作为背景质量/成本锚。seed 42、官方仓库、dense T768 与 H65 seed 3407、稀疏 K384 不允许形成“DUCA 低多少”的因果 delta。                                                                    |
| **Historical H65 42dba3f / matched H65 OFF 04c35** | 合理的问题是“复合 H65 配方能否重现”，而不是“某个采样模块是否带来 65%”。                                                            | H65 是真实 hard selected RGB、固定 K384、50% uniform companion、contribution distillation、ASFormer action/transition adaptation、selected-axis detector 和 pre-NMS 物理回映的复合协议。其终态 checkpoint 明确使用 epoch-59 EMA，不允许中间 validation 选优。 | 65.1257 相对历史 65.385724 仅低 0.2600 pp，支持“复合配方在单 seed 上大体可复现”。但 @0.6/@0.7 分别低 0.8061/0.5703 pp，不能声称完全复现，更不能把分数归因于语义选择、蒸馏、ASFormer 或课程中的任何单项。                                        |
| **20+40 压缩 H65**                                   | 合理地回答“总训练暴露从 30+60 压到 20+40 是否可行”。它不是同训练资源比较，而是主动减少资源的压缩实验。                                           | 代码冻结了 Stage-1 20 epoch、Stage-2 40 epoch 和相应 successful-update 合同。                                                                                                                                                        | 62.4648、相对 matched H65 −2.6609 pp，且 @0.7 −3.3703 pp，明确否定这一压缩合同。它不否定 H65 的语义间接选择。                                                                                                 |
| **AM-RPCH25 / LongCosine-H6000**                   | 合理地回答“较慢 decay、更大 LR 面积能否在仅 3000 个 Stage-2 updates 下恢复性能”。不能回答“训练长度是否重要”，因为长度仍然不足。                    | LR attribution commit 对 optimizer/scheduler 和分组 LR 面积进行了显式控制与记录。                                                                                                                                                         | 63.22/63.56 均未恢复 H65，且 @0.7 仍低约 2.06/2.30 pp。结论是 **schedule 不是主要缺口，Stage-2 成熟度/暴露量更可信**。继续 LR sweep 没有科学价值。                                                                      |
| **RankPack / TrueTime**                            | 这是当前最强的单机制设计：相同 selected RGB、相同 K、seed、更新数，只改变真实时间的表示/几何处理。                                           | TrueTime 路径确实显式构造 physical-time 表示和严格 q→physical 映射；回映工具要求单调位置并在物理坐标中解释 proposal。                                                                                                                                        | `+0.6208 Avg / +1.6885 @0.6 / +0.7915 @0.7` 是“物理时间可能特别影响高 IoU”的部分机制证据。单 seed、运行根 seal mismatch 使其不能成为 paper-ready 结果，也不能证明所有时间表示都有效。                                             |
| **DUCA-UVT**                                       | 从一开始就是复合诊断，不是可识别的 V-score 或 V-budget 实验。                                                              | 同一 value 输出既以 `frame_score + α·V` 修改位置排序，又通过聚合后的 sigmoid evidence 决定动态 K；配置同时启用 variable-K、几何/value 辅助项。                                                                                                                 | 57.35→55.93/55.92 只否定这套 **V-score + V-budget + 辅助损失 + 动态执行** 的整体实现。不能单独判定 value representation、位置价值或预算价值无效。                                                                      |
| **FoveaSampler / Query-Bridge**                    | 名义上是 ablation chain，但不是干净单变量矩阵：全部臂共享动态预算和低性能背景，部分臂同时改变 supervision、selection source 与 sampler policy。 | `query_only` 直接用 Query contribution 覆盖 frame score；训练为随机 Gumbel-TopK ST，推理改成 boundary-protected greedy MMR；所有臂都保留动态预算与预算/多样性损失。                                                                                          | `query_cycle=54.67` 高于 42.94 背景只能说明复合 cycle 训练改变了这一弱系统。不能据此否定 Query representation、Query-scout 协同学习或 detached cycle。`query_fovea=43.77` 主要否定当前 policy family，而不是“foveation”一般思想。 |
| **连续 24×16 cliplet FZ/JT**                         | 问题合理：保留局部连续视频语义能否优于任意逐帧 rank packing。                                                                 | revision `8a6e7e…` 无远端可验证 URL，本轮不能声称逐行核验，只能接受给定合同与终态结果。                                                                                                                                                                  | 49.89/47.24 是足够强的负结果，终止该连续 cliplet 主路线。它不否定 scout、真实物理时间或真实稀疏重计算。                                                                                                                |
| **First-Mixing SingleClock**                       | 作为 H65-compatible 表示门是当前最小、最干净路线；前提是主比较严格为 EMA ON−H65 OFF/replay，而不是 ON−gate-zero。                    | global canonical K384 clock、24×16 packing、8 tubelet centers、block-0-only 零初始化标量和 pre-NMS 物理回映均已实现；但只修复第一个 Transformer attention，不修复 Conv3D tubelet mixing。远端身份与终结器仍不满足 Gate-v2。                                          | ON−gate-zero 近零是明确的 mechanism warning，但只是 same-checkpoint 诊断。当前不得发 PASS/KILL；必须先完成 H65 replay 五边界身份和合法 terminal artifact。                                                        |

---

# 三、实现正确性总审计

## 1. 是否真实减少 VideoMAE 重计算

### H65 / SingleClock

**通过代码层检查。**

H65 selector 先形成 hard selected K384 RGB，再由 backbone 接收这些帧。SingleClock 只附加位置元数据，并没有恢复 T768 的重型 backbone 输入。Backbone 看到的是 24 个 16-frame clips，而不是 dense 768 帧。

因此：

```text
executed_heavy_frames = 24 × 16 = 384
```

不是 padding/metadata 伪动态。

### UVT / Fovea

代码路径会按 resolved K 构造 hard gathered selected inputs，并允许 variable total frames；从实现意图看是真实 variable-K 重计算。Fovea 的 hard gather 数值上与纯索引 gather 一致，ST 部分只用于反向。

但二者都缺少论文级逐窗口：

```text
requested_k
executed_unique_k
backbone_input_frames
padded_slots
heavy_backbone_latency
```

统一账本，所以只能判为“代码支持真实重计算”，不能判为“full-stack saving 已证明”。

## 2. 非连续帧是否产生 pseudo-contiguous semantics

**是，且这是代码事实。**

H65 的全局严格递增 K384 positions 被直接 reshape 为：

```text
[B, 24, 16]
```

每两个 selected-rank 相邻帧形成一个 tubelet center；VideoMAE Conv3D patch embedding 把每个 16-frame rank-packed clip 当作 8 个本地 tubelets。即使物理位置间隔很大，tubelet embedding 仍先进行局部卷积混合。

因此：

* selected-rank packing 是 H65 与 UVT/Fovea 的真实表示风险；
* pre-NMS q→physical 回映只能修复最终 proposal 几何；
* 它不能逆转 backbone 已发生的时间解释；
* SingleClock 只补偿第一个 self-attention，不能修复 tubelet embedding。

## 3. q→physical 映射位置

在当前远端 `SingleStageDetector` 中，selected-q proposal 会先回映到物理 dense 坐标，再执行 threshold、top-k 和 NMS。TwoStage 路径也在 NMS 前调用同类 remap。

这一项代码语义是正确的。

但 Gate-v2 仍必须证明：

```text
raw selected-q proposal
→ exactly-once q-to-physical mapping
→ unchanged filtering/top-k/NMS
→ canonical evaluator JSON
```

不能只证明“函数存在”。

## 4. 现有六类方法是否隔离

现有结果**没有**形成同提交、同训练暴露的：

```text
fixed-K
actionness-only
actionness+boundary
direct-query
dynamic-K
K-shuffle
```

六臂因果矩阵。

它们分散在不同 commit、训练课程、selector、loss、geometry 和 budget controller 中。因而任何跨矩阵排序都只能是 descriptive comparison，不能进入同一 causal-delta 列。

---

# 四、最可信的性能下降原因排序

以下排序不附虚构概率。

| 排名    | 原因                                                              | 证据级别          | 裁决                                                                                 |
| ----- | --------------------------------------------------------------- | ------------- | ---------------------------------------------------------------------------------- |
| **1** | **Stage-2 训练成熟度、参数暴露和课程长度不足**                                   | 强事实证据         | 20+40 损失 2.66 pp；成熟 Stage-1 加两个 3000-update schedule 仍未恢复。LR 形状不是主要原因。             |
| **2** | **selected-rank pseudo-contiguous 时间语义**                        | 代码事实 + 部分机制证据 | 非均匀帧被作为局部连续 tubelet 输入；TrueTime 在同帧下高 IoU 有正增量。尚不能量化它占总缺口的比例。                      |
| **3** | **dynamic-K 背景本身不成熟，并与 selection score 混杂**                     | 代码与设计事实       | UVT 的 value 同时控制 score 与 K；Fovea 全臂共享 dynamic budget。                              |
| **4** | **Fovea 训练/推理策略不一致**                                            | 代码事实          | 训练 Gumbel-TopK ST，推理 boundary quota + greedy MMR；不是同一离散 policy。                    |
| **5** | **Direct Query / boundary quota 导致覆盖集中或选择分布漂移**                 | 机制事实，量级待证     | `query_only` 直接让 Query 决定 frame score；foveated inference 加 quota/MMR。不能从现有结果分离其贡献。 |
| **6** | **多目标优化冲突：selector、value、cycle、budget、diversity、detector 同时竞争** | 高优先级假设        | 现有矩阵未提供 loss-gradient 或单变量证据，不能提升为事实。                                              |
| **7** | **单 seed 随机波动**                                                 | 未量化不确定性       | 可能解释部分 0.x pp 差异，但不太可能单独解释 cliplet 的十余 pp 下降或压缩的 2.66 pp。                          |

对 H65 65% 的解释排序因此是：

```text
训练暴露/课程与 selected-axis 适配
    > 复合语义 selector + distillation + ASFormer 的联合效果
    > 任何尚未隔离的单个模块
```

不能写成：

```text
“语义非均匀采样单独带来 65.39”
```

---

# 五、“相同训练资源或轮次”的唯一操作定义

今后 DUCA 中“same training resource”统一定义为下面的不可拆分身份元组：

```text
TRAIN_RESOURCE_IDENTITY = {
  base_code_and_allowlisted_diff,
  data_split_and_manifest,
  seed_and_all_rng_partitions,
  ordered_video/window exposure,
  augmentation exposure,
  Stage-1 checkpoint hash,
  Stage-1/Stage-2 boundary,
  successful optimizer update sequence,
  optimizer state initialization,
  per-parameter-group LR value at every successful update,
  scheduler state and last_epoch,
  EMA update count and decay rule,
  base trainable parameter groups,
  terminal checkpoint rule,
  evaluator/NMS/time-mapping identity
}
```

具体冻结如下。

## 1. H65 基座

```text
Stage-1: 同一个 30-epoch terminal-EMA checkpoint hash
Stage-2: 60 epochs
successful Stage-2 optimizer updates: exactly 6000
primary checkpoint: epoch-59 state_dict_ema
seed: 3407
best-validation selection: forbidden
```

## 2. 新机制允许的唯一差异

只有预注册机制本身可不同。

对 SingleClock：

```text
allowed parameter delta:
  one scalar relative_physical_time_scale

all pre-existing parameter groups:
  identical names
  identical initialization
  identical LR sequence
  identical weight decay
  identical update count
```

新标量的参数量、LR 和零 weight decay必须单列。不能因为机制增加了一个参数就延长训练、额外 warmup、额外 Stage 或重新选择 checkpoint。

## 3. GPU-hours 是否属于“同资源”

**训练 GPU-hours不作为准确率因果身份的替代物。**

原因是相同的 6000 次更新在不同 kernel 或硬件负载下可能产生不同 wall time。真正决定训练暴露的是 successful updates、样本/窗口暴露、LR trajectory、EMA 次数和参数组。

但必须另行报告：

```text
training GPU-hours
peak GPU memory
total examples/windows
successful updates per GPU-hour
resume/retry overhead
```

它们进入系统资源报告，不得用于替代训练身份。

## 4. inference cost

推理成本单列为：

```text
decode
low-cost scout preparation
scout
selection/transport
heavy VideoMAE
detector
q→physical mapping
NMS/evaluator serialization
```

同 K 归因比较和同 full-stack cost 系统比较不得混为一谈。

## 5. 对 60-epoch 严格资源上限的结论

若“同资源”被定义成**总计只能 60 epochs / 6000 total updates**，当前最好已测值是 63.56，没有证据支持其超过 H65 65.1257/65.3857。

因此冻结：

```text
STOP_60_EPOCH_COMPRESSION
NO_MORE_LR_SCHEDULE_SWEEP
```

新机制若以 H65 为基座，必须共享 30+60 最大预算，不得再增加训练轮次。

---

# 六、SingleClock 精确张量与时间轴合同

对每个样本 (b)：

## 1. 物理帧位置

```text
p_b ∈ Z^(384)
0 ≤ p_b,0 < ... < p_b,383 < T_b
```

canonical exact-uniform：

[
u_{b,j}=
\left\lfloor
\frac{2j(T_b-1)+383}{2\cdot383}
\right\rfloor .
]

该 endpoint-inclusive integer-half-up 生成器必须是唯一实现。

## 2. RGB 输入

```text
selected RGB:
X_sel ∈ R^[B,24,3,16,H,W]

executed heavy frames per sample:
24 × 16 = 384
```

不得对 T768 做 dense VideoMAE 后再 mask。

## 3. Tubelet 物理中心

每个 clip (c\in[0,23])、tubelet (r\in[0,7])：

[
a_{b,c,r}
=========

\frac{
p_{b,16c+2r}+p_{b,16c+2r+1}
}{2},
]

[
v_{b,c,r}
=========

\frac{
u_{b,16c+2r}+u_{b,16c+2r+1}
}{2}.
]

代码确实先在全局 K384 上生成 canonical positions，再 reshape 成 24×16，并对每两个位置取平均，而不是在每个 clip 内重新生成 16 点 uniform。

## 4. 物理时间残差

[
d_{b,c,r}=\frac{a_{b,c,r}-v_{b,c,r}}{\max(T_b-1,1)}.
]

[
R_{b,c,r,s}
===========

\operatorname{clip}
\left(
\frac{d_{b,c,r}-d_{b,c,s}}{2},
-1,1
\right).
]

若每个 tubelet 有 (S) 个 spatial patches，则 (R) 被扩展为：

```text
[B×24, 1, 8S, 8S]
```

并进入 block-0 scaled-dot-product attention。

## 5. 可学习尺度

[
A=\tanh(\theta),R,\qquad \theta_0=0.
]

`gate-zero` 强制 (A=0)。只有 block 0 拥有该标量，后续 Transformer blocks 不得接收 physical-time bias。

## 6. identity-at-init / canonical-uniform identity

当 (p=u) 时：

```text
actual tubelet centers == canonical tubelet centers
R == 0 exactly
SingleClock input bias == None or exact zero
final backbone output == H65 OFF bit-identical
```

这不是近似容差门，而是 bit identity。

## 7. Proposal 坐标

```text
raw detector coordinate: selected_q
mapping: selected_q → physical_dense
mapping count: exactly once
mapping location: before filtering, top-k, IoU, NMS, voting, serialization
```

---

# 七、当前数值的合法解释

## 1. ON−gate-zero

按本轮给出的 official validation family：

```text
EMA Avg:   +0.0055 pp
EMA @0.6: −0.0033 pp
EMA @0.7: −0.0019 pp

Final Avg: +0.0043 pp
```

这是强烈的 **mechanism warning**：

* 当前 checkpoint 在推理时几乎不依赖该 bias；
* 可能是 (\theta) 很小；
* 也可能是训练期效应被其他参数吸收；
* 也可能是该测试对路径不敏感。

但冻结合同明确规定 gate-zero 只作诊断，因此它不能单独 KILL。

## 2. 当前不可准入的影子比较

把现有 EMA ON 与 matched H65 OFF 数值仅作算术影子比较：

```text
Avg:  64.4661 − 65.1257 = −0.6596 pp
@0.6: 57.3720 − 57.7757 = −0.4037 pp
@0.7: 43.1417 − 43.3137 = −0.1720 pp
```

若这些数值在合法 H65 replay、五边界 identity、同 evaluator JSON 下完全保持不变，那么：

```text
Avg  FAIL
@0.6 FAIL
@0.7 PASS
```

即三项中两项低于 `−0.20 pp`。

但现在它仍然是：

```text
CROSS-RUN-ROOT DESCRIPTIVE WARNING
NOT A UNIT-1 DECISION
```

不能为了“结果看起来会失败”而跳过身份门。否则整个项目以后任何正结果也无法获得可信的准入标准。

---

# 八、唯一下一动作：Gate-v2 artifact-only recovery

## 唯一工作包

```text
DUCA_H65_SINGLECLOCK_GATEV2_ARTIFACT_RECOVERY-v001
```

它不是新训练实验，也不是新 inference sweep。它只消费：

* Job `1252482` 的已有 terminal checkpoint；
* Job `1251782` 的 H65 OFF terminal checkpoint；
* 已提交 Job `1253090` 合法完成后产生的已有 artifacts；
* 已存在的 predictions、metrics、identity、config、checkpoint 和 evaluator files。

不得新建另一个模型运行来补采缺失证据。

## 必须恢复的五边界 H65 replay identity

对同一 ordered sample/window：

1. `selected integer indices`
2. `gathered selected RGB`
3. `exact VideoMAE input after preprocessing/packing`
4. `raw selected-q proposals + scores + labels`
5. `physical pre-NMS input and canonical evaluator JSON`

每一边界至少记录：

```text
sample_id
video_id
window_start
tensor shape
dtype
ordered bytes SHA-256
coordinate_space
checkpoint hash
config hash
```

任何一边界不存在或无法从已有 artifact 确定性恢复：

```text
EVIDENCE_INVALID
```

不得用重新训练、Query、Bridge 或 dynamic-K 绕过。

---

# 九、Builder 最小 patch

## Allowed files

仅允许修改或新增：

```text
tools/bata/finalize_duca_h65_singleclock_terminal.py
tools/bata/analyze_duca_h65_singleclock_strata.py
tools/bata/recover_duca_h65_singleclock_gate_v2.py   # 可新增
tools/bata/validate_duca_h65_singleclock_gate_v2.py  # 可新增
tests/test_duca_h65_singleclock_*
```

## Read-only implementation truth

```text
opentad/models/utils/temporal_grid.py
opentad/models/backbones/backbone_wrapper.py
opentad/models/backbones/vit_adapter.py
opentad/models/detectors/actionformer.py
opentad/models/detectors/single_stage.py
opentad/models/detectors/two_stage.py
configs/adatad/thumos/duca_h65_first_singleclock_cycle4.py
configs/adatad/thumos/duca_h65_first_singleclock_cycle4_gate_zero.py
configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py
```

## Required minimal changes

1. 删除远端终结器中把 `ON−gate-zero` 当主效应的逻辑。
2. 主估计量改为：

   ```text
   EMA ON − EMA H65 OFF/replay
   ```
3. 只对三项 point delta 应用 `>=−0.20 pp`。
4. bootstrap CI 只写入报告，不参与 PASS/KILL。
5. gate-zero、final、cost 只标记 diagnostic。
6. 旧 `short_action / low-high distortion ON−gatezero Avg-mAP` analyzer 不得冒充当前 frozen boundary-error strata。远端 analyzer 当前计算的是 ON−gate-zero 的短动作和 distortion strata，而不是 Gate-v2 所需的两类 duration-normalized unmatched-penalized boundary error。
7. 增加五边界 replay identity。
8. 输出 machine-readable implementation-review receipt。
9. 明确本地 `61065a…` 与远端 `b2cc…` 的 diff；在没有完整 diff 前不得声称 Gate-v2 已经被 Pro 审查。

## Forbidden changes

```text
frame selector
selected indices
selected RGB
K=384
VideoMAE architecture
tubelet size
adapter
ActionFormer
losses
optimizer
scheduler
LR
EMA
seed
data order
augmentation
NMS
evaluator
q→physical mapping
checkpoint contents
Query
Bridge
dynamic-K
continuous cliplet
any new training
any unregistered inference
```

---

# 十、Critic 必查项

Critic 只返回：

```text
SINGLECLOCK_GATEV2_ARTIFACT_PASS
```

或：

```text
SINGLECLOCK_GATEV2_ARTIFACT_BLOCKED
```

必须检查：

1. **Gate semantics**

   * 主比较确为 EMA ON−EMA H65 OFF/replay；
   * 三项阈值均为 inclusive `>=−0.20 pp`；
   * CI、gate-zero、final、cost 没有偷偷进入硬门。

2. **Five-boundary replay**

   * 五个边界全部存在；
   * sample/window ordering 完全一致；
   * hash 比较不是集合比较或排序后比较。

3. **Input identity**

   * ON 与 gate-zero 同 checkpoint；
   * indices、RGB、VideoMAE input 完全相同；
   * 只允许 gate flag 不同。

4. **Uniform identity**

   * canonical uniform positions 唯一；
   * residual exact zero；
   * backbone final output bit-identical；
   * 失败必须输出 `KILL_SINGLECLOCK_REPRESENTATION`，不能降级成 warning。

5. **Coordinate state**

   * raw proposals 的初始坐标是 selected-q；
   * q→physical 恰好一次；
   * NMS 接收 physical segments；
   * 不得先 selected-axis NMS 再回映。

6. **Checkpoint**

   * epoch 59；
   * state key `state_dict_ema`；
   * 6000 successful updates；
   * 相同 Stage-1 checkpoint hash；
   * 无 best-validation 选择。

7. **No hidden work**

   * 没有新训练；
   * 没有新增模型推理；
   * 没有用本地未审代码生成缺失 boundary；
   * 没有为修复身份而改变 prediction。

---

# 十一、Evaluator 准入、统计量和最便宜 falsifier

## 1. Metric embargo

在以下三项全部通过前，Evaluator 不得发布 Unit-1 decision：

```text
H65 replay five-boundary identity = PASS
ON/gate-zero input identity       = PASS
uniform first-mixing/backbone identity = PASS
```

## 2. 主统计量

仅使用 terminal EMA：

[
\Delta_{\text{Avg}}
===================

## \text{Avg-mAP}_{ON}

\text{Avg-mAP}_{H65\ OFF/replay},
]

[
\Delta_{0.6}
============

## \text{mAP@0.6}_{ON}

\text{mAP@0.6}_{H65\ OFF/replay},
]

[
\Delta_{0.7}
============

## \text{mAP@0.7}_{ON}

\text{mAP@0.7}_{H65\ OFF/replay}.
]

单位统一为 percentage points。

## 3. PASS

```text
Delta_Avg >= −0.20 pp
AND Delta_0.6 >= −0.20 pp
AND Delta_0.7 >= −0.20 pp
```

输出：

```text
PASS_UNIT1_SINGLECLOCK_NONINFERIORITY
```

该 PASS 只授权一次新的科学决策来考虑 Unit-2；它本身不授权 Query 代码或训练。

## 4. KILL

任一合法主 delta：

```text
< −0.20 pp
```

输出：

```text
KILL_SINGLECLOCK_REPRESENTATION
```

这只否定：

> block-0 first-transformer-attention relative physical-time residual 在当前 H65 30+60 配方下的表示路线。

它不否定：

* TrueTime 一般价值；
* tubelet-aware 时间建模；
* H65 语义采样；
* dynamic outer-K。

## 5. Evidence invalid

以下任何一项失败：

```text
H65 replay five-boundary identity
ON/gate-zero input identity
checkpoint/evaluator binding
```

输出：

```text
EVIDENCE_INVALID
NO_SCIENTIFIC_DECISION
```

uniform first-mixing/backbone bit identity 失败则是结构性 falsifier，应直接：

```text
KILL_SINGLECLOCK_REPRESENTATION
```

因为这说明所谓“零残差/关闭路径”仍改变 backbone。

## 6. Bootstrap

```text
paired video-cluster bootstrap
resamples = 10000
```

报告 point、95% interval，但 CI 不进入 Unit-1 PASS/KILL。

## 7. Strata

只有既有预封存材料足够时才计算：

```text
high-gap-CV
high-boundary-density
```

两者均采用：

```text
duration-normalized
unmatched-penalized
boundary error
```

要求：

```text
error_ON − error_OFF <= 0
```

若材料不够：

```text
diagnostic_unavailable
```

不得补跑未注册 inference，也不得拿远端现有 short-action/distortion mAP analyzer 代替。

## 8. 最便宜 falsifier

最便宜且最先执行的 falsifier 不需要 GPU：

> 对 canonical exact-uniform 输入，比较 H65 OFF 与 SingleClock 的最终 backbone tensor bytes。

若非 bit-identical，直接终止 SingleClock。无需读取 mAP，也无需继续 bootstrap。

---

# 十二、结果到 claim 的边界

## 若 Unit-1 PASS

唯一允许的结论：

> 在 seed 3407、固定 K384、共享 H65 30+60 优化预算和 terminal EMA 下，block-0 first-transformer physical-time residual 相对 H65 OFF 在 Avg-mAP、mAP@0.6、mAP@0.7 三项上均未超过预注册的 0.20 pp 退化界限。

不得写：

> SingleClock improves TAD。

若 ON−gate-zero 仍近零，还必须补充：

> 同 checkpoint 推理依赖尚未建立；可能是训练期 coadaptation 或路径未被有效使用。

## 若 Unit-1 KILL

允许的负结论：

> 在 H65 的 rank-packed 16-frame VideoMAE clip 合同下，仅在第一个 Transformer attention 中加入真实时间残差不足以保持性能。

不得写：

> 真实物理时间无用。

原因是 Conv3D tubelet embedding 仍先发生，SingleClock 没有修复最早的局部 rank mixing。

## 对后续路线的约束

Unit-1 无论 PASS 或 KILL：

```text
dynamic-K remains unauthorized
continuous cliplet remains stopped
LR/schedule sweep remains stopped
```

若 KILL，下一假设回到：

```text
semantic acquisition quality
Stage-2 training maturity
selected-axis/backbone compatibility
```

不得用 Query、Wasserstein、learned cross-attention、Bridge 或 dynamic-K 立即“抢救”该失败。

---

# 十三、最终交接合同

```yaml
next_owner: Builder

next_action: >
  从 Job 1253090、1252482、1251782 已有且合法终态 artifact 中，
  完成 Gate-v2 五边界 H65 replay identity、ON/gate-zero input identity、
  canonical-uniform backbone bit identity，以及正确的
  EMA ON − H65 OFF/replay 三点终结器；不得重训或新增推理。

dependency:
  - Job 1253090 必须达到真实 terminal state
  - Job 1252482 terminal epoch-59 checkpoint/state_dict_ema
  - Job 1251782 H65 OFF terminal epoch-59 checkpoint/state_dict_ema
  - Stage-1 checkpoint hash
  - six-family metric/prediction/config/evaluator artifacts
  - existing selected-input identity artifacts
  - reviewable diff for local Gate-v2 commits relative to b2ccfcc

expected_return_at: >
  DUCA_H65_SINGLECLOCK_GATEV2_TERMINAL_ADJUDICATION-v001，
  在同一次返回中包含 Builder artifact、独立 Critic closure 和
  Evaluator machine-readable final receipt；不是未来日期承诺。

single_recovery: >
  只允许一次 artifact-only 恢复与终结。
  若任一五边界身份没有被既有作业记录、不能从既有文件确定性恢复，
  则返回 EVIDENCE_INVALID，并对该运行身份停止 SingleClock，
  不重训、不补跑、不引入 Query/Bridge/dynamic-K。
```

**最终路线一句话：**

> 保留 H65 30+60 作为唯一同预算性能基座；只完成 SingleClock Gate-v2 的既有证据准入。当前已有数值对 SingleClock 构成明显负向预警，但在五边界身份与正确终结器闭合前，科学上既不能 KILL，也不能 PASS。
