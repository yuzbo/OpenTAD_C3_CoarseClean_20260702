# PIVOT

> **下一代 DUCA 的一句话核心机制：不再预测“哪些帧最重要”，而是由低成本 Scout 预测每个时间窗口再增加一组重型视觉观察能够带来的 TAD 边际收益，并在每个视频的严格总计算预算下，把 VideoMAE 计算从低收益窗口转移到高收益窗口。**

本轮停止把“固定 `K=384` 内怎样做更好覆盖”作为论文主问题。主路线转向：

> **基于边际计算价值的输入自适应重型计算分配**
> *Marginal-Value-Based Adaptive Heavy Compute Allocation*

冻结模型名：**DUCA-Marginal-v1**。

冻结的唯一下一任务不是完整 60-epoch 新模型训练，而是先完成一个**冻结 H65 检测器的端到端反事实边际预算实验**。它将在真实 THUMOS14 上直接回答：

> 在相同视频总观察预算下，把部分窗口从 `K=384` 降为 `K=256`、把省下的计算转给另一些窗口使其升为 `K=512`，能否提高官方 TAD mAP？

这是当前信息增益最高、代码代价最低、最能改变论文故事的实验。

---

# 1. 证据核对与路线更正

## 1.1 H65 不是“全时间轴纯 Top-K actionness”

这一点必须正式更正。

H65 的真实采集路径包含预算校准的帧级保留率和确定性 systematic sampling：每个候选位置拥有至多一个单位容量，保留率总和被校准到指定预算，再由累计保留率生成严格递增的原始时间索引。它不是简单地在全时间轴取动作性最高的 384 帧。代码中的 `SamplingRateSelectionOutput` 和 `budget_calibrated_sampling_rate` 已经明确表达了这一点。

因此，旧因果链：

> 纯 Top-K 聚集 → 时间空洞 → coverage 修复

没有成立的代码前提。

H65 的 `65.13%` 与 dense AdaTAD `68.73%` 之间约 3.6 个百分点的差距仍然真实，但现在至少存在六种未分离解释：

1. 每个窗口都固定使用相同预算；
2. Scout 语义证据质量不足；
3. H65 排序与真正的下游检测效用不一致；
4. 未选信息被完全丢弃；
5. 非均匀 packed observation 的训练适配不足；
6. 固定预算本身没有利用视频和窗口难度异质性。

共享 dense AdaTAD 的 `68.73%` 只能只读引用，不应重训；H65 的 `65.13% / 43.31%` 是当前干净匹配锚点。 

## 1.2 Coverage-v1 失败的是具体干预，不是 coverage 家族

实际 Coverage 分支由 H65 clean revision `04c35a3b76897e6c1569eeede41ed3aecaf7f854` 派生。首个机制提交 `4e809370...` 引入的是：

* 固定 `K=384`；
* 保留 H65 priority path；
* 96 个时间锚点；
* 只基于时间距离的 facility-location coverage；
* 不使用边界梯度；
* 不使用 Scout feature similarity；
* 不使用上下文合并；
* 不使用物理时间编码；
* 不使用动态预算。

因此，`PRE_RUN 1261679` 的真实结论只有：

> **在 H65 候选与固定语义优先级下，当前 96-anchor temporal facility-location allocation 没有形成预期的覆盖干预。**

平均 anchor coverage 仅改善约 `3.32%`，而 maximum temporal hole P95 从 `2` 恶化到 `8`，意味着它甚至没有实现自己预注册的中间机制目标。不得绕过该门启动 matched 60-epoch 训练。

它不否定：

* feature-space diversity；
* semantic redundancy；
* downstream task utility；
* dynamic budget；
* information-preserving compression；
* 其他形式的 coverage。

## 1.3 PJST-D1 不再是当前关键路径

PJST-D1 OFF/ON 的精确点值为：

* OFF：Avg-mAP `65.063283%`
* ON：Avg-mAP `64.590802%`
* 差值：`-0.472481 pp`
* mAP@0.7 差值：`+0.122739 pp`

两臂的 211 个可评估视频和逐项结果均得到精确重现，但预注册的 10,000 次整视频配对自助法因输出路径错误而得到 `0/10000` 次抽样。因此：

* 可以说该具体 PJST-D1 修补没有平均正向点估计；
* 不能说物理时间表示总体显著无效；
* 不能把孤立的 `+0.123 pp` mAP@0.7 当作收益。

**本轮不恢复 PJST bootstrap。** 原因不是它不重要，而是无论配对区间最终是“不确定”还是“显著小幅负向”，都不会改变当前主决策：该具体表示修补不值得继续作为下一主路线。

## 1.4 旧负结果的正确边界

* H65 训练压缩与第二阶段学习率修补没有恢复 30+60 性能，只是否定“靠日程压缩即可无损恢复”。
* 连续片段 FZ/JT 的 `49.89 / 47.24` 是明确负结果，因此新路线的“16-frame packet”必须表示**16 个额外的非连续 H65-ranked observations**，绝不能重新变成 16 个连续原始帧。
* UVT、Fovea、Query-Bridge 同时改变了选择、预算证据或训练信息流，只能判定其首版系统失败，不能据此删除每个 primitive。
* 库存提交 `5136011...` 只能用于找现有组件，不能作为正式实验身份；各历史实验必须回到自己的 clean revision。

---

# 2. 2025–2026 前沿文献核验与更正

## 2.1 Adaptive frame acquisition：支持“输入异质性”，但尚未直接证明 TAD 需要 dynamic K

| 工作                                    | 核验后的真实贡献                                                                                                                 | 对 DUCA 的价值                                 | 不可直接迁移之处                                                    |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ | ----------------------------------------------------------- |
| **Flexible Frame Selector，CVPR 2025** | 可学习 frame-selection policy 与 flexible selection operation，在 VideoQA 中减少下游处理帧数并保持或提高准确率。([CVF Open Access][1])            | 支持“不同输入可以输出不同数量的观察”                        | 其目标由文本问题定义，不是无 query 的 TAD 边界定位效用                           |
| **CSES，2026 年 8 月 arXiv**             | 同时自适应决定需要评分的帧数和最终选帧数；用 relevance profile prominence、active acquisition 与 coverage saturation 停止；选择目标为单调次模覆盖。([arXiv][2]) | 支持自适应停止、预算饱和与粗到细观察                         | 是 query-conditioned LVLM 预印本；Coverage-v1 只复制了一个被大幅简化的时间覆盖外壳 |
| **AKS，CVPR 2025**                     | 在**固定 visual-token 数量**下联合优化 prompt relevance 与视频 coverage。([CVF Open Access][3])                                        | 证明 relevance 与 coverage 可共同优于 uniform      | 不提供 dynamic total budget 证据                                 |
| **Q-Frame，2025**                      | training-free，使用文本—图像匹配和 Gumbel-Max 做 query-aware frame selection，并联合多分辨率适配。([arXiv][4])                                 | 支持 frame count 与 spatial resolution 联合分配   | 依赖文本 query，且不是 TAD 检测损失                                     |
| **MEC，2026 年 8 月 arXiv**              | 构造“一条排序服务任意预算”的嵌套 frame ranking；小前缀强调证据，后续前缀补充时间上下文与多样性。([arXiv][5])                                                     | 非常适合作为 DUCA 的 nested acquisition primitive | 其 evidence-to-context ranking 面向长视频问答，不能直接当作 TAD utility    |

**裁决：**这些工作没有证明“固定 K 在 TAD 中已经错误”，但共同表明固定 K 不应继续被视为默认终局。当前最值得验证的是：TAD 不同窗口是否具有可预测的 heavy-compute demand。

## 2.2 Marginal utility：这是下一代 DUCA 应吸收的核心思想

**MMG-Vid** 不再只问“哪个 token 最重要”，而是先按视频相似性形成 segment，再动态分配 segment token budget，使每次增加 token 的边际增益最大；token 层还联合 inter-frame uniqueness 与 intra-frame diversity。([AAAI Publications][6])

**AdapTok，CVPR 2026** 更直接地训练一个 scorer，预测一个视频块在不同 token 数量下的重建质量，并在推理时用整数规划分配全局 token budget。([CVF Open Access][7])

**ZOO-Prune，CVPR 2026** 的重要启发是：输出敏感度比注意力、静态重要性或纯 diversity 更接近真正的下游影响。([CVF Open Access][8])

**COBS** 将 block selection 归约为估计每个 block 的 attention mass，并指出真正困难的是用紧凑摘要预测 block 对最终输出的贡献。**KVpop** 则直接用 future-attention target 监督保留/删除决策，而不是依赖当前时刻的代理分数。([arXiv][9])

这四类工作共同指向一个比“动作性排序”更强的目标：

> **预测再投入一个计算 packet 对下游任务的条件边际收益。**

对于 TAD，最合理的边际量不是：

* 时间距离；
* 单帧动作性；
* feature norm；
* 单独的 diversity；

而应优先是：

> **在当前已观察集合与当前检测器状态下，增加一组重型观察能够减少多少分类与边界回归损失。**

## 2.3 Diversity 与 redundancy 值得作为输入证据，不应成为唯一目标

* **DivPrune，CVPR 2025** 将 token pruning 表述为 Max-Min Diversity Problem，目标是让保留 token 更能代表原集合。([CVF Open Access][10])
* **ForestPrune，CVPR Findings 2026** 在已编码 token 上构建时空 forest，并依据 tree/node 角色做高比例剪枝。([CVF Open Access][11])
* **MMG-Vid** 联合 segment-level marginal gain、inter-frame uniqueness 与 intra-frame diversity。([AAAI Publications][6])

这些工作说明 feature novelty 和 redundancy 是有价值的 Scout 输入，但纯 diversity 可能保留与 TAD 无关的背景变化。**DUCA 的最终判据必须仍然是 downstream TAD utility。**

## 2.4 Information preservation：值得保留为第二主问题，不进入第一实验

* **FrameFusion 是 ICCV 2025，不是 CVPR 2026。**它在同一 LVLM Transformer 表示空间内，对相邻帧对应视觉 token 先做 similarity-based merging，再在深层做 importance pruning。([CVF Open Access][12])
* **ShaRP** 的核心问题是浅层剪枝时的位置编码偏置与信息交互不足，机制为 segment-aware causal masking、position debiasing 与 token deduplication；“register token”不是其摘要中的核心发现。([arXiv][13])
* **Hi-Lo Prune** 在删除 token 前先让保留 token 吸收候选 token 的信息。([CVF Open Access][14])

由此得到的迁移边界是：

> 同一 VideoMAE feature space 中的 merge/summary 是合理候选；Scout hidden 直接平均后线性投影到重型 VideoMAE feature space 不是同一机制。

## 2.5 Physical time：科学问题仍成立，但当前优先级下降

Qwen2.5-VL 通过 dynamic FPS training 和 absolute time encoding，使多模态 RoPE 的时间维度索引与真实时间流速对齐。([Qwen][15])

VideoRoPE 则采用 3D RoPE、低频时间维度分配、对角布局与可调 temporal spacing。([arXiv][16])

但它们都是经过相应体系训练的 VideoLLM 位置编码，不能推出：

> 在 AdaTAD attention score 上增加一个简单 `MLP(|Δt|)` 就拥有同样保证。

结合 PJST-D1 的负向平均点估计，本轮只保留现有 physical timestamp 透传和 proposal inverse mapping，不引入新的时间编码。

## 2.6 Hardware-aligned sparsity：迁移 packet 化原则，而不是迁移 CUDA kernel

Native Sparse Attention 结合 coarse compression、fine selection 与 local precision，并从一开始以硬件可实现的层次稀疏为设计目标。([ACL Anthology][17])

DUCA 应吸收的是：

* 固定大小计算 packet；
* coarse summary；
* selective heavy compute；
* 局部最低保护；
* 真实执行而非 padding 后的名义 sparsity；
* 根据预测收益决定是否继续。

不应尝试在当前阶段移植 NSA/COBS 的 LLM attention kernel。

---

# 3. 新的论文问题

## 3.1 正式问题定义

> **在离线时序动作检测中，不同视频与时间窗口需要的视觉计算量是否显著不同？能否由低成本语义 Scout 预测“再增加一组重型视觉观察所带来的检测边际收益”，并在严格的视频级计算预算下动态分配 VideoMAE 观察数，从而优于每个窗口固定 `K=384` 的稀疏检测？**

## 3.2 审稿人应当意外的地方

不是“又一个更聪明的 frame importance scorer”，而是：

> **即使保持整段视频的重型观察总数不变，仅将计算从已经饱和的窗口转移到边界复杂、动作密集或 Scout 不确定的窗口，也可能提高官方 high-tIoU 定位。**

这使 DUCA 从“选帧启发式”转变为一个更一般的论文问题：

> **Task-conditioned allocation of heavy visual computation for temporal detection.**

## 3.3 数学形式

对于视频 \(v\) 的窗口 \(w=1,\dots,W_v\)，预算选择为：

$$
K_w \in \{256,384,512\}.
$$

定义两种真实边际量：

$$
\Delta^-_w =
\mathcal L_w(256)-\mathcal L_w(384),
$$

表示把该窗口从 384 降至 256 的损失代价；

$$
\Delta^+_w =
\mathcal L_w(384)-\mathcal L_w(512),
$$

表示把该窗口从 384 升至 512 的收益。

从全 `K=384` 开始，将窗口 \(e\) 降为 256，同时将窗口 \(h\) 升为 512。该计算转移的净收益为：

$$
G(h,e)=\Delta^+_h-\Delta^-_e.
$$

只在预测 \(G(h,e)>0\) 时执行转移，并保持：

$$
\sum_w K_w = 384W_v.
$$

因此总选中 observation 数严格不变。第一版不需要连续预算，也不需要强化学习。

---

# 4. 五个真正不同的候选原型

以下数值都是**预注册预测，不是已有证据**。

## 候选 1：DUCA-Marginal-v1——边际计算价值重分配

**核心假设：**H65 的主要未利用机会不是窗口内部的时间覆盖，而是窗口之间的计算需求异质性。

**跨领域 primitive：**MMG-Vid 的 marginal gain、AdapTok 的 budget-conditioned quality prediction、MEC 的 nested ranking、COBS/KVpop 的 future utility supervision。

**迁移方式：**

* Scout 预测 `256→384` 的避免损失与 `384→512` 的额外收益；
* 在视频内做严格总预算转移；
* 每个 K 仍使用 H65 语义 priority；
* K256 是 K384 的子集，K512 是其超集。

**不能照搬：**

* MMG-Vid 的 token importance 不是 TAD loss；
* AdapTok 的 reconstruction quality 不是 proposal localization；
* KVpop 的 future attention 不是 detector utility。

**可复用代码：**

* `PrefixMarginalUtilityBudgetController`
* `counterfactual_utility.py`
* H65 budget-calibrated sampling
* raw-frame gather
* selected timestamp mapping
* VideoMAE、AdaTAD、ActionFormerHead、官方 evaluator。

H65 clean revision 已经包含 marginal-utility budget controller 接口和相关 acquisition 导入，因此无需另建动态预算框架。

**预期：**

* Avg-mAP：相对 H65 `+0.6～+1.5 pp`
* mAP@0.7：`+0.8～+2.0 pp`
* short-action：`+1～+3 pp`
* 同平均 K384：理论观察数相同
* 平均 K320 operating point：真实 heavy latency 下降约 `10%～18%`

**最强反解释：**冻结 K384 训练出的检测器不能适应 K256/K512，观察到的差异主要是输入长度分布偏移，而不是窗口难度。

**最便宜 falsification：**冻结 H65 checkpoint，计算真实三预算反事实损失和官方 mAP，不重训 VideoMAE。

**失败只否定：**当前 frozen-H65、当前 nested ranking 和当前 Scout 对边际收益的可预测性；不自动否定经过多预算训练的 dynamic compute。

---

## 候选 2：DUCA-Relevance-Novelty——相关性与特征新颖性联合的任意预算排序

**假设：**H65 的问题是优先级只表达语义显著性，没有表达“这帧相对已选集合还提供多少新信息”。

**primitive：**CSES、DivPrune、MMG-Vid、MEC、ZOO-Prune。

**最小机制：**

$$
g(i\mid S)=
\lambda_r R_i+
\lambda_n\min_{j\in S}d(f_i,f_j)+
\lambda_u U_i.
$$

其中 \(R_i\) 是动作相关性，\(d\) 是 Scout feature novelty，\(U_i\) 是不确定度。

**与 Coverage-v1 的关键区别：**

* Coverage-v1 只有时间距离；
* 该方法使用 feature novelty；
* 生成一条 nested ranking，而不是重新求解互不兼容的独立集合。

**预期：**

* Avg-mAP：`+0.4～+1.1 pp`
* mAP@0.7：`+0.5～+1.4 pp`
* short-action：不确定
* actual compute：K 固定时基本不变。

**最强反解释：**视觉 novelty 更多对应镜头切换和背景变化，而不是动作边界。

**便宜 falsification：**在冻结 H65 上检查 feature novelty 是否与真实 hard-swap detector utility 正相关。

**失败不否定：**dynamic budget 或 information preservation。

---

## 候选 3：DUCA-Compressed-Context——稀疏原始观察加同空间残差信息

**假设：**主要瓶颈不是选错帧，而是未选帧的信息被完全删除。

**primitive：**FrameFusion、Hi-Lo Prune、local-global context aggregation。

**最小机制：**

* 重型 VideoMAE 的首个共享 feature space 形成局部 summary；
* 被删除 token 只向相邻保留 token 传递残差信息；
* 深层只处理稀疏 token。

**不能使用：**Scout hidden → 线性投影 → VideoMAE hidden 的直接跨空间 pooling。

**预期：**

* Avg-mAP：`+0.8～+2.0 pp`
* mAP@0.7：`+1.0～+2.0 pp`
* short-action：较可能受益
* actual compute：低于 dense，但高于纯 pre-backbone K384。

**反解释：**浅层 dense 编码已经消耗掉大部分成本，论文的“原始帧采集节省”不再成立。

**便宜 falsification：**只在第一层 tubelet representation 上做无学习的局部聚合，冻结其余网络，检查是否恢复 high-tIoU。

**失败不否定：**跨窗口动态预算。

---

## 候选 4：DUCA-Continuous-Time——非均匀观察的真实时间重型表示

**假设：**packed irregular observations 被当作等间隔序列，是 H65 的主要表示失配。

**primitive：**Qwen2.5-VL absolute-time alignment、VideoRoPE adjustable temporal spacing。

**最小机制：**

* 不使用简单 attention MLP bias；
* 在首个时序位置机制中直接编码归一化真实 timestamp；
* 初始化时严格退化为原模型。

**预期：**

* Avg-mAP：`+0.2～+1.0 pp`
* mAP@0.7：`+0.5～+1.5 pp`
* actual compute：近似不变。

**反解释：**AdaTAD 已通过 proposal inverse mapping 消化了主要几何问题，重型 feature encoder 对真实间隔并不敏感。

**便宜 falsification：**零初始化 timestamp residual，与完全相同 H65 selected frames 配对。

**失败不否定：**更完整 RoPE 或多预算分配，但结合 PJST-D1 后会显著降低该路线优先级。

---

## 候选 5：DUCA-Two-Level——原始观察动态预算加 backbone 内 token compression

**假设：**帧级和 token 级冗余同时存在，只做一层 sparsity 不能达到最佳性能—成本前沿。

**primitive：**NSA 层次稀疏、FrameFusion、ForestPrune、MeToM。

**预期：**

* Avg-mAP：目标为不低于 DUCA-Marginal
* mAP@0.7：取决于局部保护
* actual compute：可能下降 `25%～40%`

**反解释：**两个稀疏层相互放大信息损失，无法归因，也可能没有真实 kernel speedup。

**便宜 falsification：**只有在 DUCA-Marginal 已经证明 dynamic acquisition 后，固定 acquisition 并加入单一 backbone compression。

**失败不否定：**帧级 dynamic allocation。

---

# 5. 候选科学比较与唯一选择

| 方向                          | 是否直接继承新证据 | 论文问题强度 | 第一结果速度 | 因果可解释性 | 工程风险 | 裁决               |
| --------------------------- | --------: | -----: | -----: | -----: | ---: | ---------------- |
| Marginal dynamic allocation |    **最高** | **最高** | **最快** |  **高** |    中 | **唯一主路线**        |
| Relevance–novelty fixed-K   |         中 |      中 |      快 |      高 |    低 | 备选诊断             |
| Sparse + compressed context |         高 |      高 |      中 |      中 |    高 | 第二科学问题           |
| Continuous-time encoder     |         中 |      中 |      慢 |      高 |   中高 | 暂缓               |
| Two-level sparsity          |         中 |      高 |     最慢 |      低 |   最高 | future extension |

选择 DUCA-Marginal-v1 的决定性理由是：

1. Coverage-v1 没有产生 allocation intervention，因此继续优化 coverage objective 的预期信息增益低。
2. H65 已有 systematic sampling，窗口内部并非完全缺少覆盖保护。
3. 现有代码已经包含 marginal-utility dynamic-budget controller 与 counterfactual utility 组件，不需要重新搭框架。
4. dynamic budget 是 DUCA 长期问题，却尚无一次可信的性能—成本联合实验。
5. 冻结 H65 即可快速做真实反事实实验，不需要先完成最终系统。

---

# 6. 最终系统蓝图

```text
完整离线视频
    ↓
低成本语义 Scout
    ├─ 动作性、变化、不确定度
    ├─ 窗口级 pooled hidden
    └─ 候选观察的嵌套优先序列
    ↓
边际计算价值预测
    ├─ 预测 K256→K384 的避免损失
    └─ 预测 K384→K512 的额外收益
    ↓
视频级严格预算分配
    ├─ 从低收益窗口回收 128 observations
    ├─ 向高收益窗口增加 128 observations
    └─ 保证视频总预算不超过预注册上限
    ↓
按 K 分组的真实 packed raw-frame execution
    ↓
VideoMAE heavy backbone
    ↓
保留原始 timestamp / selected-axis inverse mapping
    ↓
原 AdaTAD / ActionFormer detector
    ↓
原 NMS 与官方 THUMOS14 evaluator
```

## 核心贡献

1. **TAD 边际计算价值预测**
2. **严格视频级预算约束下的跨窗口计算转移**
3. **一条 nested acquisition sequence 支持多个预算**

## 必需支持机制

* H65 frozen Scout 与 priority path
* 非连续观察的原始 timestamp 透传
* K 分组执行，禁止 padding 到 512 后声称节省
* 短窗口实际 valid length 计费
* 现有官方 detector、loss、NMS、evaluator。

## Future extension，不进入首轮

* feature-space novelty 排序；
* 同空间 compressed context；
* timestamp-aware heavy encoding；
* VideoMAE 内部 token compression；
* 多检测器迁移。

---

# 7. 唯一下一任务单

## 任务名称

**冻结 H65 的反事实边际预算重分配实验**

## 科学决策目标

判断：

> 当前 H65 检测器中是否已经存在可利用、且可由 Scout 预测的跨窗口计算需求异质性。

## 明确不做

* 不启动 Coverage-v1 60-epoch 训练；
* 不重训 dense AdaTAD；
* 不重训 H65 Scout；
* 不修改 VideoMAE；
* 不修改 ActionFormerHead；
* 不修改 detector loss；
* 不引入 context merging；
* 不引入新时间编码；
* 不完成 PJST bootstrap；
* 不做超参数 grid search。

## 实验臂

### A. Fixed-H65-384

* 每个窗口严格 `K=384`
* 必须逐预测复现现有 H65 checkpoint
* 是本实验的唯一主控制。

### B. Oracle-Reallocate-384，仅 train-side holdout

使用 train-side GT 产生真实三预算反事实效用，在视频总预算 `384×窗口数` 下选择最优预算。只用于判断 headroom，绝不在 test 上运行。

### C. Learned-Reallocate-384

* 每窗口 `K∈{256,384,512}`
* Scout utility head 预测升/降预算的损失变化
* 视频总 observation 数严格等于 Fixed-H65-384
* 不使用 test GT。

### D. Fixed-320

* 每个窗口固定 `K=320`
* 同一 nested ranking
* 用于平均 K320 的公平控制。

### E. Learned-Allocate-320

* 每窗口 `K∈{256,384}`
* 只给预测收益最高的一半窗口增加 128 observations
* 视频平均预算严格为 320。

---

# 8. 权威代码主线与允许代码面

## 8.1 基座

* **Clean base revision：**
  `04c35a3b76897e6c1569eeede41ed3aecaf7f854`
* **新分支：**
  `feature/duca-marginal-budget-v1-20260830`

该提交是 H65 30+60 clean reference。Coverage 分支也是从它派生，但新路线不得以 Coverage head `048143...` 为基座，以免把已失败 selector 和后续 workflow 变化带入候选。Coverage 当前 head 只适合作为历史干预证据。

## 8.2 必须复用

| 文件 / Symbol                                                                    | 用途                                    |
| ------------------------------------------------------------------------------ | ------------------------------------- |
| `opentad/models/duca/dynamic_budget.py::PrefixMarginalUtilityBudgetController` | 边际预算控制器骨架                             |
| `opentad/models/duca/counterfactual_utility.py`                                | detached detector-utility 监督与现有训练隔离逻辑 |
| `opentad/models/duca/structured_selection.py::budget_calibrated_sampling_rate` | H65 原始 systematic allocation          |
| `opentad/models/duca/acquisition.py` 中 H65 priority path                       | frozen Scout evidence                 |
| 现有 raw-frame gather / selected timestamp mapping                               | 重型输入与物理坐标                             |
| 现有 AdaTAD / VideoMAE / ActionFormerHead                                        | 完全复用                                  |
| `tools/test.py` 官方 evaluator                                                   | 最终 mAP                                |
| 现有 paired-video bootstrap 工具                                                   | 10,000 次配对区间                          |

## 8.3 允许修改

1. `opentad/models/duca/dynamic_budget.py`

   * 增加**有符号**的两侧 marginal head：

     * downgrade penalty；
     * upgrade gain。
   * 增加视频级 exact-total-budget allocator。
   * 不再把所有 marginal utility 强制为正数。

2. `opentad/models/duca/counterfactual_utility.py`

   * 新增三预算 prefix utility：

     * `L256-L384`
     * `L384-L512`
   * detector graph 必须 detach。

3. `opentad/models/duca/acquisition.py`

   * 增加 nested-prefix 输出；
   * `prefix(384)` 必须与现有 H65 K384 selected indices 完全一致；
   * `prefix(256) ⊂ prefix(384) ⊂ prefix(512)`。

4. 当前 H65 selector/wrapper 的原有文件

   * 只增加按预算 K 调用 nested prefix 和按 K 分组执行；
   * 不复制 detector wrapper。

5. 新增一个配置：

   * `configs/adatad/thumos/duca_marginal_frozen_h65_probe.py`

6. 新增一个 focused test：

   * `tests/test_duca_marginal_budget.py`

7. 一个最小实验入口：

   * 优先扩展现有 DUCA evaluation/profiling 脚本；
   * 只有现有入口无法表达三预算反事实时，才允许增加一个薄脚本。

## 8.4 禁止修改

* `opentad/models/backbones/videomae.py`
* ActionFormer/AdaTAD head
* classification/regression loss
* proposal decoding 数学
* NMS
* THUMOS14 split
* annotation
* class map
* official evaluator
* H65 Scout 权重
* H65 detector checkpoint
* selected physical timestamp inverse mapping
* dense baseline 结果
* Coverage selector
* PJST-D1。

---

# 9. Builder → independent Critic → Evaluator

项目角色应保持论文优先：Builder 只实现冻结机制；Critic 只阻塞会改变机制、数据、梯度、坐标、公平性或真实执行的问题；Evaluator 独立运行正式数据和官方指标。 

## 9.1 Builder

必须交付：

1. nested selection 的集合包含关系；
2. K384 与原 H65 indices bit-exact；
3. utility label 只来自 train-side detector loss；
4. test forward 不接受 GT、teacher utility 或 prediction cache；
5. 视频级 observation 总数守恒；
6. 不同 K 真正形成不同 shape 的 heavy tensor；
7. K256/K384/K512 分组执行；
8. timestamps 与 selected indices 一一对应；
9. 短窗口未使用预算可重新分配，不按 padding 计费；
10. utility optimizer 只更新新 head。

### 必须通过的 focused tests

* `prefix256 ⊂ prefix384 ⊂ prefix512`
* `prefix384 == frozen_H65_selection`
* `sum(K_dynamic) == 384 × num_windows`
* 固定 K384 输出与旧 H65 prediction 一致
* utility target `.requires_grad == False`
* test allocation 在删除 GT 后不变
* VideoMAE 实际输入长度等于预算
* proposal inverse mapping 在三种 K 下单调且落在原视频时间范围内
* K 分组执行与逐样本执行数值一致
* 任何 padding 到 512 后统一运行的实现必须失败。

## 9.2 Independent Critic

只检查以下 blocker：

1. K384 parity 是否被破坏；
2. utility label 是否含 test GT 或 oracle；
3. nested prefix 是否真正复用 H65 priority；
4. controller 是否偷偷改变 Scout；
5. detector loss 是否被修改；
6. K512 是否引入位置编码或投影 shape 错误；
7. dynamic policy 是否依赖 batch composition；
8. 总预算是否按 actual valid observations 计算；
9. actual CUDA execution 是否真的按 K 缩短；
10. official evaluator 与 NMS 是否原样使用。

通过即停止，不做代码风格或额外文档循环。

## 9.3 Evaluator

顺序固定：

1. 复现 Fixed-H65-384；
2. 在 train-side 生成三预算反事实；
3. 训练 utility head；
4. 运行 40-video train-side holdout 机制门；
5. 只有机制门通过，才运行官方 test；
6. 对 Fixed-384 与 Learned-Reallocate-384 做 10,000 次整视频 paired bootstrap；
7. 运行真实 CUDA profiling；
8. 返回原始 mAP、预算分布、utility correlation、成本和结果分类。

确定性 launcher、环境、路径或 Slurm shell 故障只做最小修复，不形成新模型 revision 或科学结论。

---

# 10. 精确训练、评估与成本合同

```yaml
identity:
  base_revision: "04c35a3b76897e6c1569eeede41ed3aecaf7f854"
  branch: "feature/duca-marginal-budget-v1-20260830"
  h65_checkpoint:
    source: "existing sealed H65 30+60 experiment record"
    state_key: "state_dict_ema"
    epoch: 59
    substitution_allowed: false

dataset:
  name: "THUMOS14"
  detector_train_split: "val"
  official_eval_split: "test"
  controller_fit_videos: 160
  controller_holdout_videos: 40
  split_rule: "video-level deterministic shuffle"
  split_seed: 3407
  test_labels_visible_to_allocator: false

model:
  scout:
    source: "existing H65 scout"
    frozen: true
  detector:
    source: "existing H65 terminal EMA"
    frozen: true
  acquisition:
    type: "nested H65 priority sequence"
    budgets: [256, 384, 512]
    observation_packet: 16
    contiguous_raw_clip: false
    reproduce_h65_at_k384: true
  utility_head:
    inputs:
      - pooled_scout_hidden
      - actionness_statistics
      - transition_statistics
      - scout_uncertainty
      - candidate_redundancy_statistics
    outputs:
      - downgrade_penalty_384_to_256
      - upgrade_gain_384_to_512
    target:
      - "loss_k256 - loss_k384"
      - "loss_k384 - loss_k512"
    detector_gradient: false

utility_training:
  trainable_modules: ["utility_head"]
  seed: 3407
  epochs: 20
  checkpoint_rule: "terminal_epoch_20"
  optimizer: "AdamW"
  learning_rate: 0.001
  weight_decay: 0.0001
  batch_size: 256
  large_grid_search: false

allocation:
  primary_policy:
    name: "video_level_equal_budget_reallocation"
    baseline_k: 384
    lower_k: 256
    upper_k: 512
    total_budget: "384 * actual_window_count"
    transfer_only_when_predicted_net_gain_positive: true
    max_transferred_window_fraction: 0.50
  efficiency_policy:
    name: "video_level_mean_k320"
    budgets: [256, 384]
    mean_k: 320

evaluation:
  metrics:
    - "mAP@0.3"
    - "mAP@0.4"
    - "mAP@0.5"
    - "mAP@0.6"
    - "mAP@0.7"
    - "Avg-mAP"
  diagnostics:
    - "short-action mAP, duration <= 3 seconds"
    - "boundary error"
    - "budget distribution"
    - "Spearman correlation with true marginal utility"
    - "sign accuracy"
    - "fraction of oracle gain recovered"
  bootstrap:
    unit: "whole video"
    draws: 10000
    comparison: "Learned-Reallocate-384 minus Fixed-H65-384"

profiling:
  timer: "torch.cuda.Event"
  include:
    - scout
    - utility_head
    - allocator
    - gather
    - VideoMAE
    - detector_head
    - postprocessing
  warmup_iterations: 50
  report:
    - per_video_p50_ms
    - per_video_p95_ms
    - heavy_backbone_ms
    - peak_gpu_memory_mb
    - actual_selected_observations
  padding_to_max_budget_allowed: false

output_root:
  "/data/run01/sczc063/yuzibo/duca_marginal_v1_04c35a3_20260830"
```

精确 H65 checkpoint 路径与 SHA 必须从现有 H65 30+60 原始实验记录读取；当前提供的材料没有给出该终态 checkpoint 的完整路径，因此这里不得臆造。缺少路径时由 Codex 在现有实验记录中解析，而不是创建新 checkpoint。

---

# 11. 运行前机制门

正式 test 评估前必须同时满足：

## 11.1 实现门

* K384 selected indices 与 H65 完全相同；
* K384 predictions 在同一 checkpoint 下逐项一致；
* K256/K512 不产生 shape、位置编码或解码异常；
* detector、Scout 全部冻结；
* 无 test GT；
* heavy path 无统一 padding。

## 11.2 Headroom 门

在 40-video train-side holdout 上，oracle equal-budget reallocation 相对 fixed K384：

* `ΔAvg-mAP ≥ +0.8 pp`
* `ΔmAP@0.7 ≥ +1.0 pp`

若 oracle 没有这个 headroom，不启动 test 学习策略评估，更不启动完整训练。

## 11.3 Predictability 门

* 两个边际目标的 Spearman 相关系数均 `≥0.25`；
* 边际收益正负号准确率 `≥60%`；
* learned policy 恢复 oracle Avg-mAP 增益的 `≥40%`；
* Learned-Reallocate-384 中至少 `10%` 窗口为 K256、至少 `10%` 为 K512；
* 视频总预算误差为 0。

这三类门分别回答：

* 代码是否有效；
* dynamic allocation 是否有真实 headroom；
* Scout 是否能预测该 headroom。

不得把它们混写。

---

# 12. 结果裁决表

| 类别                                       | 条件                                                                                       | 科学解释                                             | 下一动作                                            |
| ---------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------ | ----------------------------------------------- |
| **Implementation invalidity**            | K384 不复现；K512 shape 错；有 GT 泄漏；统一 padding；物理映射错误                                          | 没有产生科学证据                                         | 最小修复后重跑同一任务                                     |
| **Intervention failed**                  | 预算几乎全为 K384；总预算不守恒；utility 输出常数                                                          | 没有真正测试 dynamic allocation                        | 修复 allocator/head；不得解释 mAP                      |
| **No oracle headroom**                   | holdout oracle `ΔAvg < +0.3` 且 `ΔmAP@0.7 < +0.5`                                         | 当前冻结 H65 + nested ranking 中，跨窗口重分配缺少可利用空间        | 不训练 dynamic head；转向 information preservation    |
| **Oracle positive, predictor failed**    | oracle 过门，但相关性 `<0.25` 或 learned policy 不恢复增益                                            | dynamic budget 有价值，但当前 Scout/utility target 不足   | 下一问题是更好的 downstream utility supervision         |
| **Strong support**                       | test `ΔAvg ≥ +0.8 pp`、`ΔmAP@0.7 ≥ +1.0 pp`；Avg-mAP paired 95% CI 下界 `>0`；实际同预算延迟不增加超过 5% | Scout 可预测并利用跨窗口计算异质性                             | 锁定 DUCA-Marginal，进入多预算 task-adapted 60-epoch 训练 |
| **Partial support**                      | test `ΔAvg +0.2～+0.79 pp`，或 high-tIoU 明显正向但 Avg 未过门；oracle/predictor 门均通过                | 机制有信号，但 frozen K384 detector 限制了收益               | 允许一次完整多预算训练                                     |
| **Null**                                 | test `ΔAvg` 在 `[-0.2,+0.2]`，paired interval 跨 0，且无 subgroup 稳定收益                         | frozen-detector learned allocation 未产生效能收益       | 依据 oracle 结果决定是结束 budget 还是改 supervision        |
| **Scientific negative**                  | 机制门强通过，但 test `ΔAvg ≤ -0.8 pp` 且 paired CI 上界 `<0`                                       | 当前 Scout-predictable marginal allocation 跨视频泛化失败 | 停止该 utility route，转向 compressed context         |
| **Performance positive / cost negative** | mAP 过门，但 actual latency 增加 `>5%`                                                         | 算法信息成立，硬件实现或 packet 粒度不成立                        | 不声称效率；下一步只做硬件可实现分组                              |
| **Efficiency support at mean K320**      | Avg-mAP 不低于 H65 K384 超过 `0.5 pp`，且端到端 latency 降低 `≥10%`                                  | 初步形成性能—成本联合证据                                    | 纳入完整训练的第二 operating point                       |

由于第一任务冻结了 H65 detector，这里的 Strong Support 仍属于**强正向诊断**，不是最终论文主结果。最终 paper-level support 仍需完整任务适配训练、多种子和正式成本曲线。

---

# 13. 失败后的下一科学问题

失败后的问题不是统一的，必须按失败位置分流。

## 13.1 Oracle 没有 headroom

下一问题变为：

> **H65 与 dense 的差距是否主要来自稀疏观察的信息不可恢复损失，而不是预算配置错误？**

此时主候选转向 **Sparse Acquisition + Same-Space Compressed Context**，不再搜索 dynamic-K 阈值。

## 13.2 Oracle 有 headroom，但 Scout 无法预测

下一问题变为：

> **什么低成本信号能够预测 detector marginal utility？**

只允许比较：

* Scout uncertainty；
* feature novelty；
* boundary complexity；
* train-only detached detector utility。

不得恢复直接索引监督或 test teacher。

## 13.3 Learned reallocation 提高 mAP，但没有实际速度优势

下一问题变为：

> **如何把连续窗口级预算决策变成 GPU 可实现的少量离散 batch shape？**

此时科学路线不变，只把预算限制为少量 packet bucket，并基于 measured latency 而不是帧数优化。

## 13.4 Equal-budget 成功，但 mean-K320 失败

可以形成的论文主张是：

> 动态计算重分配提高固定总成本下的检测效用。

不能声称：

> DUCA 已经降低总成本而无损。

下一步应先 task-adapted 多预算训练，而不是加入更多 selector 模块。

---

# 14. 北京时间绝对截止时间

当前北京时间基准为 **2026 年 8 月 30 日 14:07**。

| 节点                             | 北京时间截止               | 交付                                                       |
| ------------------------------ | -------------------- | -------------------------------------------------------- |
| 科学与代码面冻结                       | **2026-08-30 16:00** | 本裁决写入研究状态；停止 Coverage 正式训练                               |
| Builder 最小实现与 focused tests    | **2026-08-30 23:00** | exact H65 parity、nested budgets、utility labels、allocator |
| Independent Critic             | **2026-08-31 00:30** | 对固定 commit 给出 PASS 或真实 blocker                           |
| 三预算反事实生成与 controller holdout 门 | **2026-08-31 06:00** | oracle headroom、相关性、预算分布                                 |
| 通过门后的官方 test 与 CUDA profiling  | **2026-08-31 12:00** | Fixed-384、Dynamic-384、Fixed-320、Dynamic-320              |
| paired bootstrap 与终态证据包        | **2026-08-31 13:00** | 10,000 次整视频区间、原始结果与成本                                    |
| Pro 结果复盘                       | **2026-08-31 14:00** | 决定是否进入完整多预算 60-epoch 训练                                  |

若 Slurm 队列或既有 checkpoint 路径形成客观阻塞，Codex 必须在对应截止点返回：

* exact commit；
* 已完成的部分；
* 唯一阻塞事实；
* 已提交 Job ID 或缺失资源；

不得用文档、审计或新路线讨论代替实验。

---

# 15. 当前可写入论文与不可声称的边界

## 当前可以写

1. H65 不是纯 Top-K，而是预算校准的非均匀 systematic sampling。
2. H65 K384 相对共享 dense AdaTAD 仍有约 3.6 pp Avg-mAP 差距。
3. Coverage-v1 的具体 96-anchor temporal facility-location 没有产生预注册的 coverage intervention。
4. PJST-D1 的特定物理时间修补没有平均正向点估计。
5. 连续片段采样在现有 DUCA/AdaTAD 体系中是明显负结果。
6. 2025–2026 前沿方法越来越从静态重要性转向输入自适应预算、边际增益、任意预算嵌套排序与信息保留，但这些跨领域结果尚未直接证明 TAD 中的效果。

## 当前不可写

* coverage 方法总体无效；
* fixed K 已被科学证伪；
* dynamic budget 已经有效；
* H65 差距由窗口预算不均造成；
* physical-time modeling 总体无效；
* DUCA 已优于 dense；
* DUCA 已获得性能—成本联合优势；
* `K=384` 是最终论文预算；
* 当前 utility controller 已经得到正确训练；
* token-merging 论文可以直接证明 Scout-to-VideoMAE feature fusion 有效。

---

# 最终冻结工作单

**唯一负责人顺序：Builder → independent Critic → Evaluator。**

**唯一代码基座：**`04c35a3b76897e6c1569eeede41ed3aecaf7f854`。

**唯一新分支：**`feature/duca-marginal-budget-v1-20260830`。

**唯一当前实验：**冻结 H65 detector，在 THUMOS14 上估计 `K256/K384/K512` 的真实反事实检测边际价值，并在严格视频总预算下比较 Fixed-H65-384 与 Learned-Reallocate-384。

**正式停止：**Coverage-v1 matched 60-epoch 训练、PJST-D1 路线恢复、dense baseline 重训、任何 merging/time-bias/token-pruning 拼接。

本轮的科学裁决不是“dynamic budget 可能值得试”，而是：

> **DUCA 现在应当直接检验：低成本 Scout 能否预测每一单位新增 VideoMAE 计算的 TAD 边际价值，并据此跨窗口重新分配真实重型计算。**

[1]: https://openaccess.thecvf.com/content/CVPR2025/html/Buch_Flexible_Frame_Selection_for_Efficient_Video_Reasoning_CVPR_2025_paper.html "https://openaccess.thecvf.com/content/CVPR2025/html/Buch_Flexible_Frame_Selection_for_Efficient_Video_Reasoning_CVPR_2025_paper.html"
[2]: https://arxiv.org/abs/2608.00714 "https://arxiv.org/abs/2608.00714"
[3]: https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Adaptive_Keyframe_Sampling_for_Long_Video_Understanding_CVPR_2025_paper.html "https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Adaptive_Keyframe_Sampling_for_Long_Video_Understanding_CVPR_2025_paper.html"
[4]: https://arxiv.org/abs/2506.22139 "https://arxiv.org/abs/2506.22139"
[5]: https://arxiv.org/abs/2608.05707 "https://arxiv.org/abs/2608.05707"
[6]: https://ojs.aaai.org/index.php/AAAI/article/view/39605 "https://ojs.aaai.org/index.php/AAAI/article/view/39605"
[7]: https://openaccess.thecvf.com/content/CVPR2026/html/Li_AdapTok_Learning_Adaptive_and_Temporally_Causal_Video_Tokenization_in_a_CVPR_2026_paper.html "https://openaccess.thecvf.com/content/CVPR2026/html/Li_AdapTok_Learning_Adaptive_and_Temporally_Causal_Video_Tokenization_in_a_CVPR_2026_paper.html"
[8]: https://openaccess.thecvf.com/content/CVPR2026/html/Kim_ZOO-Prune_Training-Free_Token_Pruning_via_Zeroth-Order_Gradient_Estimation_in_Vision-Language_CVPR_2026_paper.html "https://openaccess.thecvf.com/content/CVPR2026/html/Kim_ZOO-Prune_Training-Free_Token_Pruning_via_Zeroth-Order_Gradient_Estimation_in_Vision-Language_CVPR_2026_paper.html"
[9]: https://arxiv.org/abs/2607.09052 "https://arxiv.org/abs/2607.09052"
[10]: https://openaccess.thecvf.com/content/CVPR2025/html/Alvar_DivPrune_Diversity-based_Visual_Token_Pruning_for_Large_Multimodal_Models_CVPR_2025_paper.html "https://openaccess.thecvf.com/content/CVPR2025/html/Alvar_DivPrune_Diversity-based_Visual_Token_Pruning_for_Large_Multimodal_Models_CVPR_2025_paper.html"
[11]: https://openaccess.thecvf.com/content/CVPR2026F/html/Ju_ForestPrune_High-ratio_Visual_Token_Compression_for_Video_Multimodal_Large_Language_CVPRF_2026_paper.html "https://openaccess.thecvf.com/content/CVPR2026F/html/Ju_ForestPrune_High-ratio_Visual_Token_Compression_for_Video_Multimodal_Large_Language_CVPRF_2026_paper.html"
[12]: https://openaccess.thecvf.com/content/ICCV2025/html/Fu_FrameFusion_Combining_Similarity_and_Importance_for_Video_Token_Reduction_on_ICCV_2025_paper.html "https://openaccess.thecvf.com/content/ICCV2025/html/Fu_FrameFusion_Combining_Similarity_and_Importance_for_Video_Token_Reduction_on_ICCV_2025_paper.html"
[13]: https://arxiv.org/abs/2512.05385 "https://arxiv.org/abs/2512.05385"
[14]: https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Hi-Lo_Prune_Look_at_What_Youll_Lose_before_Pruning_with_CVPR_2026_paper.html "https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Hi-Lo_Prune_Look_at_What_Youll_Lose_before_Pruning_with_CVPR_2026_paper.html"
[15]: https://qwenlm.github.io/blog/qwen2.5-vl/ "https://qwenlm.github.io/blog/qwen2.5-vl/"
[16]: https://arxiv.org/abs/2502.05173 "https://arxiv.org/abs/2502.05173"
[17]: https://aclanthology.org/2025.acl-long.1126/ "https://aclanthology.org/2025.acl-long.1126/"
