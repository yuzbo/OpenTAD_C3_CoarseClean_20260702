# DUCA 稀疏 Token 物理时间表示终态裁决

**SCIENTIFIC_DECISION: `REVISE`**

**Exact Project:** `g-p-6a796fef9a00819194024cf1de3bd697`（DUCA）
**Nonce:** `DUCA-H65-SPARSE-PHYSICAL-TOKEN-TRANSFER-v001-20260825`
**Git revision:** `b2ccfccab5b4912b59954afcc9b0364955327f7c`
**Branch:** `codex/duca-h65-firstmix-singleclock-20260824`

该提交存在，提交主题为验证 SingleClock 的物理位置身份；指定分支确实指向该提交。

---

## 一、最终裁决摘要

我不继续当前 **PatchEmbed 后 block-0 SingleClock**，也不进入 dense physical query bridge，更不引入 ODE、额外 Query、dynamic-K 或第二套采样器。

唯一冻结机制是：

> **`DUCA_PHYSICAL_JACOBIAN_SUPPORT_TUBELET-v001`，简称 PJST：物理雅可比—支撑感知 Tubelet 化。**

它是候选 **B：Spatial-First, Time-Aware Tubelet Aggregation** 的严格、参数为零、预训练可精确退化版本：

1. 不再把两张 selected-rank 相邻帧直接解释为固定时间间隔的二帧 tubelet。
2. 将原 VideoMAE 二抽头 Conv3D 权重精确分解成：

   * 一个**外观零阶矩核**；
   * 一个**时间变化一阶矩核**。
3. 用原始物理时间间隔对变化分量做雅可比归一化，用每帧的物理支撑区间修正外观聚合。
4. 在规范 uniform 输入上直接旁路到原 `PatchEmbed`，要求逐字节相同。
5. 不增加可学习时间参数，因此不存在再次把时间门缩到约 `−0.0018` 而“自动忽略”的路径。
6. 只增加逐元素均值、差分和尺度运算；原 Conv、Transformer token 数、384 张重型 RGB 输入和 detector 全部不变。

**唯一真实 falsifier** 是一条新的、完整的 30+60 H65-compatible PJST ON 训练，与已经存在的匹配 H65 OFF 终态作只读配对比较；不得再跑一个表示矩阵。

---

# 二、严厉审稿人攻击

## 2.1 最早、不可逆的表示错误确实在 Conv3D PatchEmbed

代码核验支持内联材料的核心诊断：

* wrapper 已经能获得全局 selected positions，并构造 actual/canonical tubelet 坐标再传入 backbone；
* `global_rank_clip_coordinates` 将 `[B,384]` 按 rank 重排为 `24×16`，再把每两个 rank 相邻位置平均成 tubelet 中心；物理相对时间 residual 则由 actual 与 canonical 坐标计算；
* VideoMAE 的最早重型时间交互仍是 temporal kernel/stride 均为 2 的 Conv3D PatchEmbed；
* 当前物理时间 bias 是在 `self.patch_embed(x)` 已执行后才构造并进入 block 0，其尺度是一个零初始化标量。

因此，当前顺序确实是：

```text
不规则物理帧
  -> 被当成等间隔二帧进行 Conv3D 压缩
  -> 已经丢失二帧分离表征
  -> block 0 才获知“它们其实相隔很远”
```

后续 attention bias 最多改变已经形成的 tubelet 之间如何交互，不能恢复 tubelet 内被压成一个向量的两个观测，也不能重新解释 Conv3D 已提取的伪局部运动。

这是当前最早、最明确且不可逆的表示歧义。

## 2.2 但它尚未被证明是 H65 性能缺口的主因

审稿人不能从结构缺陷直接跳到“修复后一定提升”。至少还有四种更强的竞争解释：

* ASFormer scout 的语义成熟度或课程训练本身可能才是主瓶颈；
* 非均匀采样产生的 gap 与动作性内生相关，任何显式 gap 输入都可能成为 selector 强度捷径；
* rank-based temporal position 在 PatchEmbed 之后仍然存在，即使修复二帧局部混合，长程 attention 仍可能错误理解距离；
* 单 seed 的优化方差和终态身份问题可能大于时间表示效应。

所以，PJST 是**可证伪的最早层修复**，不是已经成立的因果解释。

## 2.3 SingleClock 的负向预警不能写成科学 KILL

当前标量约 `−0.0018` 的最合理解释不是“真实时间无价值”，而是：

1. 信号作用在不可逆混合之后；
2. residual 还被 `T−1` 归一化；
3. 只进入 block 0；
4. 零初始化标量允许优化器选择最容易的路径——保持预训练网络不变并将新信号压回零；
5. PatchEmbed 本身没有因为真实间隔而改变。

因此 `−0.6596 pp` 的描述性差异是对**这个具体后置、可关闭参数化机制**的负向预警，而不是对所有物理时间表示的否定。

## 2.4 物理坐标解码必须由调用轨迹而不是配置名称证明

revision 中的 anchor-free head 具有 physical-grid point/proposal 路径，可以在 head 内直接产生物理坐标。

但同一 revision 的通用 `SingleStageDetector.post_processing` 路径仍包含先置信度过滤/top-k、后 selected-to-physical remap 的可达实现表面。

这不等于当前 H65 正在走错误路径，但意味着正式身份门不能只检查配置字段。必须捕获实际调用轨迹并证明：

```text
raw proposal
 -> exactly-once physical decode
 -> filtering
 -> top-k
 -> IoU
 -> NMS
```

任何走入通用晚 remap 的样本都使实验失效，而不是降级为“近似等价”。

## 2.5 最大的论文风险不是实现，而是新颖性

“保留 token 原始坐标”“给稀疏 token 加时间编码”“根据 gap 调整卷积”都不够新。Temporal Difference Network 已显式使用时间差分，TAdaConv 已经按时间上下文校准卷积核；TE-TAD 已在 TAD 中采用实际时间线坐标表达。([Open Access CVF][1])

可辩护的新颖性必须是以下组合，而不是其中任何一项：

> **语义间接物理选帧 + 在 VideoMAE 首次重型压缩中使用物理雅可比与支撑区间 + 对 uniform 预训练严格恒等 + 不向网络暴露 gap-only 捷径 + unchanged detector + pre-NMS 物理解码 + 高 IoU TAD falsifier。**

---

# 三、跨领域核验与真实可迁移性

## 3.1 视觉 token pruning、merging 与 latent tokenization

| 方法族                     | 实际保留什么                                     | 对 DUCA 可迁移部分                      | 不能迁移的部分                                                            |
| ----------------------- | ------------------------------------------ | --------------------------------- | ------------------------------------------------------------------ |
| DynamicViT              | 删除低重要性 token，保留幸存 token 内容                 | 删除后不应把幸存 token 的物理位置重新压成连续 rank   | 它在已有 token 上 pruning，不能恢复 PatchEmbed 已发生的二帧混合。([arXiv][2])         |
| EViT                    | 保留 attentive tokens，并把不重要 token 融成摘要 token | 可借鉴“删除与合并是不同操作”                   | 融合 token 没有天然唯一坐标；跨动作边界融合对 TAD 危险。([arXiv][3])                     |
| A-ViT                   | token 自适应停止/丢弃                             | 幸存 token 的身份不应因压缩而改变              | 仍是层间 halting，不解决输入 tubelet 的物理时间。([arXiv][4])                      |
| TokenLearner            | 从稠密特征产生少量自适应 latent summaries              | 可借鉴内容压缩与计算预算分离                    | latent token 不是原始坐标 token，也没有天然支撑区间。([arXiv][5])                   |
| ToMe                    | 按相似度对 token 加权合并，可跟踪 token size/source     | `source` 和 `size` 说明合并后应保留来源集合/质量 | 默认 size/source 不是连续物理支撑区间，官方实现还会重排合并结果；不能直接用于高 IoU 边界。([arXiv][6]) |
| 视频 STTS                 | 在空间、时间维度用 scorer 做 top-k token selection   | 原始时间坐标应跟随被选 token                 | 目标主要是识别效率，未解决不规则帧先被固定 tubelet 混合的问题。([arXiv][7])                   |
| Run-Length Tokenization | 将重复时间 patch 合为一个可变长度 token，并编码长度           | 它直接证明“压缩 token 需要显式表示其时间支撑长度”     | 它压缩近重复 patch，不处理 H65 语义选帧、非均匀空洞和 TAD 边界。([arXiv][8])               |

**结论：**视觉稀疏化真正可迁移的不是某个 scorer，而是三条约束：

1. surviving token 不得把原坐标改成压缩后的行号；
2. merged token 必须携带来源/支撑；
3. 跨边界合并必须受到比识别任务更严格的限制。

它们都不能代替 PJST 对首次 Conv3D 混合的修复。

## 3.2 LLM 长上下文、KV eviction 与位置保持

StreamingLLM 保留 attention sink 和近期 KV；H2O 保留 recent 与 heavy-hitter token。它们的核心是**删掉内容后维持剩余 attention 状态的语义一致性**，不是恢复被合并观测。([arXiv][9])

RoPE 通过位置相关旋转编码绝对位置并在注意力内形成相对位置依赖；Position Interpolation 则有意把长位置索引缩回预训练窗口。由此可知，“继续使用原始坐标”与“把位置重新压缩进熟悉范围”是两种不同假设，不能混写。([arXiv][10])

EQUIP 尤其相关：它处理 token eviction 后 KV 张量空洞、RoPE 重旋转与 re-indexing，并通过保持注意力运算的等变性减少位置重计算。这支持“压缩后必须显式维护坐标变换合同”，但它仍是注意力/KV 系统优化，不处理视觉 Conv3D 的首次内容混合。([ACL Anthology][11])

因此，LLM 领域只能迁移以下原则：

> **删除 token 可以改变存储行号，但不能在没有明确变换的情况下改变其语义坐标。**

不能迁移的是：

* 将视频物理时间直接换成 LLM token index；
* 把 RoPE 重旋转当成 tubelet 内容修复；
* 认为 T5/ALiBi/relative bias 能恢复已被 Conv3D 聚合的帧。

## 3.3 不规则采样和连续时间

Time2Vec 将标量时间映射为可学习周期/线性表示；Neural CDE 和 Latent ODE 能处理不规则观测及任意时间间隔，ContiFormer进一步把连续动态与 attention 结合。([arXiv][12])

这些工作证明“不规则时间应作为一等变量”，但不是当前 H65 的正确最小解：

* ODE/CDE 会引入连续轨迹假设、数值求解器和大量新的模型自由度；
* 它们没有必要解决只有两个时间抽头的局部 Conv3D 误解释；
* 它们会破坏 VideoMAE 预训练兼容性和本轮单变量归因。

所以本轮明确不采用 ODE、CDE 或连续时间 Transformer。

## 3.4 稀疏视觉坐标交互

Deformable DETR 通过 reference point 周围少量采样位置进行注意力，说明 sparse support 应带显式坐标；TIM 以时间区间作为 query；TE-TAD 使用实际时间线坐标。这些思想适用于未来 sparse-support/dense-query 或 detector 重构。([arXiv][13])

但本轮 D 路线会：

* 改变 detector 的输入支持；
* 引入 dense query 或新的 cross-attention；
* 与既有低性能 Query-Bridge 历史难以隔离；
* 无法修复 sparse support 自身已经错误形成的问题。

因此不选 D。

---

# 四、唯一冻结机制：PJST

## 4.1 机制直觉

原 VideoMAE temporal kernel 2 可以视为两个空间卷积抽头：

[
W^- ,; W^+\in\mathbb{R}^{C\times 3\times 16\times16}.
]

原 tubelet 为：

[
z_{\text{rank}}
= W^- * x_i + W^+ * x_{i+1}+b.
]

它隐含假设 (x_i,x_{i+1}) 的时间距离始终等于预训练的规范间隔。

PJST 将其精确分解为：

* 外观零阶矩核 (W_A=W^-+W^+)；
* 时间变化一阶矩核 (W_V=W^+-W^-)。

然后用真实时间重新定义零阶和一阶输入。

## 4.2 时间与支撑坐标

对每个样本：

* selected frame positions：

[
p\in\mathbb Z^{K}, \qquad K=384,
]

严格递增。

* 物理帧中心：

[
\tau_i=p_i+0.5
]

，位于 end-exclusive 物理边界域 ([0,T_b])。

* Voronoi 型支撑边界：

[
e_0=0,\qquad
e_i=\frac{\tau_{i-1}+\tau_i}{2};(1\le i<K),\qquad
e_K=T_b .
]

* 单帧支撑：

[
I_i=[e_i,e_{i+1}),\qquad s_i=e_{i+1}-e_i.
]

对 canonical uniform positions (u_i) 同样得到
(\bar\tau_i,\bar e_i,\bar s_i)。

支撑伸缩比：

[
q_i=\frac{s_i}{\bar s_i}.
]

对 tubelet pair (i=2j,;i+1)：

[
\delta_j=\tau_{i+1}-\tau_i,\qquad
\bar\delta_j=\bar\tau_{i+1}-\bar\tau_i .
]

所有时间比值均无单位，因此 CFR 下使用 frame units 与使用 seconds 等价。若输入为变帧率视频，则必须提供严格递增的解码 PTS；没有 PTS receipt 时不得用 frame rank 冒充物理时间。

## 4.3 支撑感知零阶矩与物理一阶矩

定义：

[
m_j=
\frac{q_i x_i+q_{i+1}x_{i+1}}
{q_i+q_{i+1}},
]

[
v_j=
\frac{\bar\delta_j}{2\delta_j}
(x_{i+1}-x_i).
]

最终 tubelet：

[
\boxed{
z_j
===

(W^-+W^+)*m_j
+
(W^+-W^-)*v_j
+
b
}
]

解释：

* (m_j) 是相对于 canonical support 的外观零阶矩；
* (v_j) 是把真实时间导数投影回 VideoMAE 预训练局部时间尺度的一阶矩；
* 大 gap 不再被解释为“巨大的一步局部运动”，其变化分量按 (1/\delta_j) 衰减；
* 更密集的真实观测对应更高时间变化率；
* 若 (x_i=x_{i+1})，无论 gap/support 如何变化，输出都与几何无关，因此不存在纯 gap→actionness 的直接捷径。

这不是将 gap embedding 加到 token 上，而是重新定义 Conv3D 原本隐含的物理微分尺度。

## 4.4 预训练权重与计算

生产实现不得新建第二套卷积权重。将输入拼成：

[
[m_j,;v_j]\in\mathbb R^{6\times H\times W},
]

将原权重视图拼成：

[
[W^-+W^+,;W^+-W^+?]
]

这里必须注意正确项为：

[
[W^-+W^+,;W^+-W^-].
]

随后执行一次 6-channel Conv2D。其乘加数与原始“两帧 × 三通道”的 Conv3D 相同，只多出逐像素均值、差分和标量乘法。

**不得把上式中的第二个权重误写为 (W^+-W^+)。任何实现或文档出现该退化项都必须 fail closed。**

新参数数量：**0**。
新 optimizer group：**0**。
新学习率、weight decay、正则：**均无**。
原 PatchEmbed 冻结状态：**保持不变**。

## 4.5 canonical-uniform identity

当且仅当：

[
p_i=u_i\quad\forall i
]

并且 valid mask、dense length、timestamp source 均与 canonical receipt 相同，必须在任何支撑浮点计算之前直接旁路：

```python
return original_patch_embed(x)
```

要求：

* forward 输出逐字节相同；
* token layout 逐字节相同；
* input gradient 在确定性测试环境中逐字节相同；
* 参数对象、参数 hash 和 optimizer membership 相同。

不得用 `allclose`、epsilon、近似 constant、舍入或 tolerance 触发 identity。

近似 uniform 仍进入 PJST 公式。

对非 uniform 输入，不要求与原 PatchEmbed 接近；它本来就是科学干预。测试只验证公式、坐标与数值有限性，不允许用模糊“接近 baseline”作为正确性标准。

## 4.6 精确张量合同

输入：

```text
rgb:
    [B, 24, 3, 16, H, W]
或等价展平形式 [B*24, 3, 16, H, W]

selected_positions:
    [B, 384] int64，严格递增

canonical_positions:
    [B, 384] int64

frame_timestamps:
    [B, 384] float64 metadata
    正式 CFR 路径可由 selected_positions + 0.5 构造

frame_supports:
    [B, 384, 2] float64 metadata

valid_frame_mask:
    [B, 384] bool，必须是 prefix-contiguous

dense_physical_end:
    [B]，与 timestamp 相同单位
```

输出：

```text
tubelet_tokens:
    [B, 24, 8, N_spatial, C]
    后续 temporal-major flatten 顺序不变

tubelet_centers:
    [B, 24, 8]
    继续使用当前 (p_2j + p_2j+1) / 2 语义

tubelet_supports:
    [B, 24, 8, 2]
    J_j = [e_2j, e_2j+2)

tubelet_valid_mask:
    [B, 24, 8]
```

`tubelet_supports` 在本轮只用于 PJST 计算、mask 和审计，不作为 ActionFormer 新输入；否则会把单一表示实验变成 detector 修改。

---

# 五、TAD 专用边界合同

## 长 gap

长 gap 的一阶差分按 (\bar\delta/\delta) 衰减，不允许把相隔很远的两帧制造成强“局部运动”。

同时不得宣称 gap 内没有动作；PJST 只是避免错误的局部解释，不能重建未观测证据。

## 短动作与相邻动作

PJST 的机制预测是：

> 如果 H65 语义 scout 确实在边界附近提高采样密度，那么短动作/相邻动作边界处 (\delta) 应较小，关键变化分量会得到保留；背景大空洞的伪变化被抑制。

若高 IoU 或短动作性能下降，则说明该预测不成立，本机制必须终止，不能再通过 learned gate、Query 或额外损失挽救。

## Padding 与尾窗

* 两帧均 valid：正常计算；
* 两帧均 invalid：输出零 token，mask 为 false；
* 一帧 valid、一帧 invalid：**fail closed**；
* 不得把 padding timestamp 纳入 (\delta) 或 support；
* 不得复制最后一张真实帧来伪造 valid tubelet；
* formal run 必须证明每个实际进入 backbone 的 valid RGB 恰好属于 K=384 账本。

## 变帧率

* 有可信 PTS：全部公式在 seconds 中执行；
* 无 PTS：本轮论文主张明确限制于已验证 CFR 输入；
* 不得将 selected rank 当作 VFR 的物理时间。

## 相同外观、不同动作实例

若两个动作实例外观相同且未采到中间过渡，PJST 无法凭时间坐标凭空区分它们。这是方法限制，不得用 support metadata 伪称“恢复了完整事件”。

## 物理解码

proposal 必须以坐标状态显式标记：

```text
selected_q -> physical_dense
```

且只允许转移一次。进入 filtering、top-k、IoU、NMS、voting、serialization 前，状态必须已经是 `physical_dense`。未知状态或重复转换立即失败。

---

# 六、为什么 PJST 优于其他三个候选

### 候选 A：一般的 Physical-Support-Conditioned Conv3D

不选择一般 A，因为 learned gap MLP、kernel gate 或额外 embedding：

* 会增加可学习自由度；
* 可能直接从 gap 推断 selector 动作性；
* 容易再次把新分支缩到零；
* 难以证明预训练身份。

PJST 保留 A 的“首次混合必须看到物理时间”，但将其约束成参数为零的零阶/一阶物理分解。

### 候选 B：Spatial-First, Time-Aware Aggregation

**选择 B 的 PJST 特化。**

它无需为每帧建立两套完整 spatial embedding，也不增加卷积 MAC：

* 原 Conv3D 被代数重写为一次 6-channel spatial convolution；
* 原权重全部精确复用；
* canonical uniform 可直接调用旧路径；
* 干预只发生在首次时间压缩。

### 候选 C：Original-Coordinate Sparse Tokens

若保留现有 temporal kernel 2，C 仍然作用太晚，等价于更复杂的 SingleClock。

若 C 为避免该问题而拆掉 Conv3D temporal mixing，它实际上已经转化为 B。因此没有理由另开 C。

### 候选 D：Sparse Support → Dense Physical Query Bridge

D 会改变 detector 支撑、增加 query interaction，并重开已有混杂 Query-Bridge 路线。它不是当前最小表示归因实验。

---

# 七、最小 Builder patch

## 允许修改

### 1. `opentad/models/utils/temporal_grid.py`

新增纯函数：

```python
global_rank_frame_supports(...)
physical_jacobian_tubelet_metadata(...)
```

职责：

* 验证严格递增 timestamp；
* 构造 actual/canonical support；
* 返回 `q`、`delta`、`canonical_delta`；
* 不改变现有 proposal decode。

### 2. `opentad/models/backbones/backbone_wrapper.py`

仅扩充进入 backbone/PatchEmbed 的 metadata：

```text
actual timestamps
canonical timestamps
actual/canonical supports
valid mask
```

不得改变 selected RGB、clip 排列或 selector。

### 3. `opentad/models/backbones/vit_adapter.py`

在当前 `self.patch_embed(x)` 位置加入：

```python
if pjst_enabled:
    x = self.patch_embed.forward_physical_jacobian(...)
else:
    x = self.patch_embed(x)[0]
```

新路径必须复用原 temporal slices，不注册新参数。

### 4. 单一新配置

```text
H65 selector: unchanged
K: 384
SingleClock: OFF
PJST: ON
Stage-1/Stage-2: unchanged
```

不得出现 PJST+SingleClock 联合配置。

## 禁止修改

* `pc_ot_mras_prebackbone_frame_selector.py`
* ASFormer scout 或 sampling-rate transport
* selected positions
* `anchor_free_head.py` 的训练语义
* detector assignment/head/loss
* filtering/top-k/NMS/evaluator
* dynamic-K
* Query-Bridge、UVT、value、cycle、distillation
* seed、split、augmentation、checkpoint rule

## 必要测试

1. **Shape/layout**

   * PJST 输出 shape、temporal-major 顺序与原 PatchEmbed 完全相同。

2. **Canonical byte identity**

   * exact uniform ON 与 OFF forward bytes 相同；
   * input gradient bytes 相同；
   * 参数/optimizer identity 相同。

3. **代数正确性**

   * PJST 输出与显式
     ((W^-+W^+)*m+(W^+-W^-)*v+b)
     参考实现一致；
   * 专门防止误实现为 (W^+-W^+=0)。

4. **Constant-content geometry invariance**

   * 令 (x_i=x_{i+1})，改变 gap/support 后输出必须相同。

5. **物理尺度**

   * 相同视觉差分下，将实际 gap 加倍，一阶矩幅度应精确减半。

6. **Support partition**

   * support 正、无重叠、无空洞；
   * 总覆盖等于有效物理区间；
   * canonical 时所有 (q_i=1)。

7. **Padding**

   * half-valid pair 必须报错；
   * invalid pair 不得产生有效 token。

8. **执行成本**

   * decoded/selected/executed RGB 均为 384；
   * Conv/Transformer token 数不变；
   * 无 dense heavy path；
   * 无 Kmax padding 冒充实际 K。

9. **坐标状态**

   * 捕获 NMS 实际输入；
   * 证明 exactly-once physical decode 发生在 filtering/top-k/IoU/NMS 前。

10. **梯度与参数**

    * detector/adapter 梯度有限；
    * 新可学习参数数目严格为零；
    * selector 参数和梯度路径与 OFF 完全一致。

---

# 八、已有 +0.6208 的正确解释

RankPack/TrueTime 的 `+0.6208 Avg`、`+1.6885 @0.6`、`+0.7915 @0.7` 应分类为：

> **单 seed、同协议、方向与高 IoU 一致的部分机制支持。**

它既不能直接称为随机波动，也不能称为稳定效果；在逐视频不确定性未知前，两种判断都过强。它也不是 PJST 的直接证据，因为它没有单独干预首次 Conv3D 的零阶/一阶物理尺度。

## 第一项无重训练分析

在任何 PJST GPU 训练前，对现有 RankPack/TrueTime 完整 prediction JSON 做一次：

* 10,000 次 paired whole-video cluster bootstrap；
* 每次对视频有放回抽样；
* 两臂共享相同视频 multiset；
* 每个 resample 重新运行 pooled official evaluator；
* 严禁先算 per-video AP 再平均；
* 同时得到 Avg、@0.6、@0.7 的差值分布。

bootstrap seed：

```text
SHA256(
  "DUCA_PJST_BOOTSTRAP_V1\n"
  + "DUCA-H65-SPARSE-PHYSICAL-TOKEN-TRANSFER-v001-20260825"
)[0:8]
```

按 unsigned big-endian 转整数，RNG 使用 `PCG64`。

95% percentile bounds 使用排序后的第 500 与第 9500 个值，不插值。

这一分析只改变对历史 `+0.6208` 的置信程度，不替代 PJST 训练，也不产生新的论文 claim。

---

# 九、唯一真实 falsifier

## 9.1 实验单元

只新增一条训练：

```text
PJST ON
SingleClock OFF
Bridge OFF
H65 selector unchanged
K = 384
seed = 3407
```

只读对照：

```text
matched H65 OFF terminal-EMA
Avg-mAP = 65.1257
mAP@0.7 = 43.3137
```

不得重跑 OFF、dense、uniform、random 或其他 fixed-K 单元。

## 9.2 身份冻结

* 同一个 Stage-1 30 epoch checkpoint；
* Stage-2 60 epoch；
* 恰好 6000 successful optimizer updates；
* checkpoint 每 5 epoch；
* 主 checkpoint 为 epoch-59 terminal EMA；
* 不允许 best validation；
* 同一 VideoMAE-S、Adapter、ActionFormer；
* 同一损失、NMS、官方 evaluator；
* 同一 THUMOS14 split；
* 同一 seed 3407；
* 同一 requested/executed K384；
* 同一 selected-position 生成路径。

因为本轮要求同一 selected RGB，必须额外记录并比较：

```text
video/window identity
selected_positions
valid mask
executed RGB count
selector/scout parameter hash
```

如果既有 OFF 运行没有完整逐窗口 selection ledger，则在得到指标前必须证明 selector 与 detector 表示分支严格 detached，且由相同代码、checkpoint、seed 可确定性重放出相同位置；否则“同一 RGB”因果主张不准入。

## 9.3 统计量

对 PJST 与 OFF 的完整预测执行相同 10,000 次 paired whole-video bootstrap：

[
\Delta_{\text{Avg}},
\quad
\Delta_{0.6},
\quad
\Delta_{0.7}.
]

这仍是单训练 seed 的视频总体不确定性，不是训练 seed 不确定性。

## 9.4 唯一通过门

只有以下条件全部满足，才保留该表示机制：

[
\Delta_{\text{Avg}}\ge +0.50\text{ pp},
]

[
LCB_{95}(\Delta_{\text{Avg}})>0,
]

[
\Delta_{0.6}\ge0,\qquad
\Delta_{0.7}\ge0,
]

[
LCB_{95}(\Delta_{0.7})>-0.20\text{ pp}.
]

并且：

* 所有身份、selection、pre-NMS、K 与 evaluator 门通过；
* VideoMAE Conv/Transformer MAC 不增加；
* PJST 新增逐元素运算单独报告；
* 全栈 p50 latency 不超过 OFF 的 `1.02×`；
* peak GPU memory 不超过 OFF 的 `1.02×`；
* short-action stratum 不下降超过 `0.50 pp`。

任何一项不满足，即终止 PJST 表示路线。不得调 gap clip、加入 learned scale、启用 SingleClock、改 support kernel 或再跑一个变体。

下一科学解释转向：

> H65 的语义 acquisition 质量或 30+60 训练成熟度，而不是继续堆叠时间表示模块。

---

# 十、结果到 claim 的严格边界

## 若通过，可以说

> 在相同 H65 语义选帧、相同 selected RGB、相同 K384、相同训练与 detector 合同下，将 pretrained VideoMAE 的首次二帧 tubelet 分解为支撑感知外观矩和物理时间归一化变化矩，在单 seed terminal-EMA 配对实验中改善了高 IoU TAD 表现。

即便通过，仍必须写：

* 单 seed mechanism evidence；
* fixed-K representation attribution；
* 尚不是 dynamic-K 结果；
* 尚不是多 seed paper-ready superiority。

## 不可以说

* 首次提出原始 position ID；
* 首次提出不规则时间编码；
* 首次提出 adaptive temporal convolution；
* 一般 token pruning/merging 的新方法；
* 所有 physical-time 表示都有效；
* dynamic-K 已成立；
* SOTA；
* VFR 泛化；
* 已恢复未采样视频内容；
* SingleClock 被科学否定；
* PJST 已证明高 IoU 因果机制，除非相应 CI 也通过。

---

# 十一、prior-art 与 novelty invalidators

下列任一情况会直接削弱或消灭新颖性：

1. 已有工作已经在 pretrained video tubelet 的**第一次二帧混合**中，用真实采样间隔将核分解成零阶/一阶矩，并对 uniform 输入保持严格恒等。
2. 已有 TAD 工作同时具备：

   * pre-heavy-backbone 非均匀物理帧 acquisition；
   * token support interval；
   * support/time-aware tubeletization；
   * unchanged detector；
   * pre-NMS physical decode；
   * 高 IoU matched-K 证据。
3. 实现最终退化成 Time2Vec、relative bias、RoPE 或 ALiBi 加在 PatchEmbed 后。
4. 实现只是 TAdaConv 式 kernel calibration 或 TDN 式差分，但没有 H65 物理 acquisition、canonical identity 和 TAD 边界合同。TAdaConv 和 TDN 已使“时间自适应卷积/显式差分”本身不具备新颖性。([arXiv][14])
5. support 只被记录用于可视化，没有进入首次聚合。
6. 任何性能变化来自 selected RGB、selector 梯度、head、NMS 或训练暴露变化，而不是 PJST。
7. RLT、ToMe 或同类方法已经覆盖所写 claim；它们使“合并 token 同时记录长度/source”不能成为核心贡献。([arXiv][8])
8. TE-TAD 已覆盖“在 TAD 中使用实际时间线坐标”的宽泛 claim，因此论文不能把物理坐标本身写成首创。([Open Access CVF][15])

最安全的论文贡献表述是：

> **一种为语义非均匀物理帧 acquisition 设计的、对 pretrained uniform VideoMAE 严格恒等的 physical-Jacobian tubelet representation，并在 unchanged high-IoU TAD 合同下进行因果验证。**

---

# 十二、执行交接合同

```text
next_owner:
    Builder

next_action:
    在 b2ccfccab5b4912b59954afcc9b0364955327f7c 上实现
    DUCA_PHYSICAL_JACOBIAN_SUPPORT_TUBELET-v001；
    只提交最小 patch、resolved config diff、十项 focused tests、
    参数/计算/坐标调用轨迹 receipt，不启动训练。

dependency:
    1. exact commit 与 exact branch identity；
    2. matched H65 OFF terminal-EMA checkpoint/prediction/evaluator receipt；
    3. Stage-1 30-epoch 起点；
    4. selected-position identity 或可确定性重放证明；
    5. 现有 RankPack/TrueTime prediction JSON 的 10,000 次 bootstrap；
    6. independent Critic 对公式、canonical identity、gap shortcut、
       pre-NMS mapping 和 no-new-parameter 的静态 PASS；
    7. Evaluator 对单一 60-epoch run 的 PRE_RUN_READY。

expected_return_at:
    Builder 完整 diff + Critic 静态闭合 + Evaluator PRE_RUN_READY 后，
    且在任何 N16R4 PJST 训练提交之前，返回一次 fresh exact-Project
    科学执行门；不得以中间 patch 或单元测试请求提前放行。

single_recovery:
    只允许一次、仅针对指标解封前发现的确定性实现/身份错误，
    在完全相同 PJST 公式和实验身份下修复并重新封存。
    不允许依据任何 mAP、loss curve 或中间 checkpoint 调参。
    若正式科学门未通过，不实施第二个物理时间变体。
```

**终态结论：**当前 SingleClock 路线需要修订，但物理时间问题不应停止。最小、最有信息量且最难被网络忽略的下一步，是参数为零、同计算阶、uniform 严格恒等的 **PJST 首次 tubelet 物理雅可比修复**；用一条完整 H65-compatible 训练决定该表示假设的生死。

[1]: https://openaccess.thecvf.com/content/CVPR2021/html/Wang_TDN_Temporal_Difference_Networks_for_Efficient_Action_Recognition_CVPR_2021_paper.html "https://openaccess.thecvf.com/content/CVPR2021/html/Wang_TDN_Temporal_Difference_Networks_for_Efficient_Action_Recognition_CVPR_2021_paper.html"
[2]: https://arxiv.org/abs/2106.02034 "https://arxiv.org/abs/2106.02034"
[3]: https://arxiv.org/abs/2202.07800 "https://arxiv.org/abs/2202.07800"
[4]: https://arxiv.org/abs/2112.07658 "https://arxiv.org/abs/2112.07658"
[5]: https://arxiv.org/abs/2106.11297 "https://arxiv.org/abs/2106.11297"
[6]: https://arxiv.org/abs/2210.09461 "https://arxiv.org/abs/2210.09461"
[7]: https://arxiv.org/abs/2111.11591 "https://arxiv.org/abs/2111.11591"
[8]: https://arxiv.org/abs/2411.05222 "https://arxiv.org/abs/2411.05222"
[9]: https://arxiv.org/abs/2309.17453 "https://arxiv.org/abs/2309.17453"
[10]: https://arxiv.org/abs/2104.09864 "https://arxiv.org/abs/2104.09864"
[11]: https://aclanthology.org/2026.acl-long.1210/ "https://aclanthology.org/2026.acl-long.1210/"
[12]: https://arxiv.org/abs/1907.05321 "https://arxiv.org/abs/1907.05321"
[13]: https://arxiv.org/abs/2010.04159 "https://arxiv.org/abs/2010.04159"
[14]: https://arxiv.org/abs/2110.06178 "https://arxiv.org/abs/2110.06178"
[15]: https://openaccess.thecvf.com/content/CVPR2024/html/Kim_TE-TAD_Towards_Full_End-to-End_Temporal_Action_Detection_via_Time-Aligned_Coordinate_CVPR_2024_paper.html "https://openaccess.thecvf.com/content/CVPR2024/html/Kim_TE-TAD_Towards_Full_End-to-End_Temporal_Action_Detection_via_Time-Aligned_Coordinate_CVPR_2024_paper.html"
