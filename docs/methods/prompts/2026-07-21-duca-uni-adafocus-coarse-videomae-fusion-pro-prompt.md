# DUCA × Uni-AdaFocus 特征融合与训练路线 Pro 独立裁决 Prompt

你现在不是一般项目顾问，而是同时承担以下角色：

- CCF-A/CVPR 级视频理解与时序动作检测方法审稿人；
- 熟悉 VideoMAE、AdaTAD、ActionFormer、ASFormer、动态计算和稀疏采样的研究员；
- 能逐行审查 PyTorch 计算图、梯度归属、时序坐标和真实计算成本的高级工程师；
- 对“端到端”“即插即用”“任务感知”“高效计算”等表述保持敌意审查的独立裁判。

你的任务不是维护现有 DUCA 路线，也不是替我们证明某个预设方案。你必须先阅读
Uni-AdaFocus 原文、官方代码和 DUCA 精确提交，然后独立判断：当前问题是否应通过
coarse/VideoMAE 特征融合解决；如果应该，怎样融合和训练；如果不应该，应采用什么
更合理的替代路线。

## 0. 强制独立性与禁止事项

1. 不得把本 Prompt 中的项目描述当作已经证实的结论。所有描述都只是待核验入口。
2. 不得把项目既有讨论中出现过的任何融合、对齐、蒸馏或时序表示结构当作正确
   答案；必须从原文、代码和任务约束重新推导候选。
3. 不得预设 coarse feature 必须融合，也不得预设“不融合才保持插件性”。
4. 不得预设当前 selected-axis、max-gap=2、K=384、ASFormer、VideoMAE 或 AdaTAD
   必须保留；但若建议替换，必须证明不是通过更高总成本或更弱协议逃避问题。
5. 不得把 Uni-AdaFocus 的视频分类 mAP 当作 TAD mAP，不得把其识别实验直接外推为
   TAD 有效性。
6. 不得宣称下游检测梯度穿过离散硬索引，除非你从代码中展示真实可微路径。
7. 不得因为类名和配置中残留 `online` 就把任务称为 Online TAD。本项目是离线 TAD。
8. 不得只给概念图、泛泛建议或“可以尝试”。每个建议都必须落到张量、坐标、损失、
   梯度、成本、代码位置、失败条件和可证伪实验。
9. 不要迎合项目作者。若最终应停止特征融合甚至停止 DUCA 路线，请明确给出 KILL。

输出中的每个关键结论必须标注为以下之一：

- `[PAPER_FACT]`：Uni-AdaFocus 原文直接支持；
- `[UPSTREAM_CODE_FACT]`：Uni-AdaFocus 官方代码直接支持；
- `[DUCA_CODE_FACT]`：DUCA 精确提交直接支持；
- `[EXPERIMENT_FACT]`：已有原始实验结果直接支持；
- `[INFERENCE]`：从事实推导但尚未实验验证；
- `[PROPOSAL]`：你新提出的候选；
- `[UNPROVEN]`：当前无法成立的主张。

## 1. 强制阅读对象

### 1.1 当前 DUCA 精确代码

- 仓库：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- 分支：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-global-curriculum-20260721
- 精确提交：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/63e25eb17e523d369f73434ed4d9b6446608861a

必须逐行检查至少以下文件及其调用链，不得只读 README 或配置摘要：

```text
opentad/models/selectors/duca_online_frame_selector.py
opentad/models/duca/acquisition.py
opentad/models/duca/transition_only.py
opentad/models/duca/structured_selection.py
opentad/models/detectors/actionformer.py
opentad/models/backbones/vit_adapter.py
opentad/models/dense_heads/anchor_free_head.py
configs/adatad/thumos/duca_two_stage_pretrained_frozen_fixed384_official60.py
configs/adatad/thumos/duca_global_curriculum_g0_no_feedback_fixed384_official60.py
configs/adatad/thumos/duca_global_curriculum_g1_protected_fixed384_official60.py
configs/adatad/thumos/duca_global_curriculum_g2_uni_companion_fixed384_official60.py
tests/test_duca_global_curriculum.py
```

沿调用链确认：低分辨率 RGB 如何进入 coarse probe；官方 ASFormer hidden、actionness
logits、transition descriptors 和 selection scores 如何生成；硬索引如何选择；原始 RGB
如何 gather；VideoMAE 如何形成特征；AdaTAD/ActionFormer projection、neck、head 最终
消费什么；GT 和预测到底在哪个时间坐标中。

### 1.2 Uni-AdaFocus 原文和官方实现

- 原始论文页面： https://arxiv.org/abs/2412.11228
- 原始 PDF： https://arxiv.org/pdf/2412.11228
- 官方仓库： https://github.com/LeapLabTHU/Uni-AdaFocus
- 本次要求固定核验的官方提交：
  https://github.com/LeapLabTHU/Uni-AdaFocus/tree/8846488310fdd4a18412608006030643e794c36e

必须阅读论文完整方法、训练目标、消融和限制，并逐行核验官方实现中至少以下对象：

```text
policy_sample_indices
MCSampleFeature
UniAdaFocus.forward
PoolingClassifier
TemporalPolicy
SpatialPolicy
UniAdaFocus.get_optim_policies
main.py 中完整 loss 聚合与 optimizer 更新
```

不要依赖任何二手总结。请确认 global/local 特征具体在哪里、以何种运算融合；各辅助
损失分别更新哪些模块；hard temporal/spatial decisions 是否 detach；训练和推理是否
同构；所谓统一动态计算具体统一了什么。

### 1.3 可见性门禁

回答开头必须给出 `VISIBILITY_CERTIFICATE`，列出：

- 两个仓库和精确提交是否真实可读；
- 论文 PDF 是否真实读取；
- 实际读取的文件清单；
- 无法读取的对象和由此受限的结论。

若 DUCA 精确提交、Uni-AdaFocus 原文或官方核心代码任一不可见，必须返回
`VISIBILITY_BLOCKED`，列出缺失对象；禁止凭常识补写逐行审查结果。

## 2. 当前项目事实入口：必须重新核验，不得直接接受

以下只用于帮助定位，最终必须由代码或原始结果独立确认：

- 任务是 THUMOS14 离线 TAD，目标是降低完整端到端计算，同时保护高 tIoU 定位。
- 当前候选通常从 T=768 个 dense candidates 中固定选择 K=384，使用全局 exact-K
  结构化策略与 hard max-hole=2，并允许跨区域预算转移。
- coarse 路径使用低分辨率 RGB、轻量 spatial stem 和官方 ASFormer，主要接受动作二
  分类、状态转变与边界相关监督；hidden dimension 通常为 96。
- 当前实现可能只让 coarse hidden 参与 selector scoring，随后 gather 选中原始 RGB，
  再由 VideoMAE 和官方 AdaTAD/ActionFormer 路径检测；是否存在 detector feature
  fusion 必须由你逐行判定。
- 当前 V8 的 P0 尝试让 coarse/action/scorer 使用分组学习率；official-60 阶段可能
  冻结 coarse，并把受保护检测反馈限制到 transition scorer。请核验实际参数组、
  detach、loss 聚合和 optimizer coverage，而不是信任配置文字。
- 当前代码中的 hard index、soft assignment、raw-pixel bridge 和 selected-axis 时间
  重映射可能分别承担不同 forward/backward 语义。请画出真实计算图。

已有匹配但已封存的 V5 终点结果如下，只能当负证据背景：

| V5 arm | Avg-mAP | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-uniform | 64.4580 | 79.7557 | 75.5604 | 67.5863 | 56.7664 | 42.6212 |
| direct-0.25 | 63.7102 | 79.4111 | 74.5551 | 66.6419 | 55.4667 | 42.4759 |
| homotopy-0.25 | 63.0601 | 78.3237 | 73.6741 | 66.0751 | 55.0551 | 42.1723 |
| homotopy + uniform companion | 63.6931 | 79.3542 | 73.9892 | 66.7425 | 56.2650 | 42.1145 |

当前 V8 Job `1178989` 仍是 `experiment_running`，其 real-CUDA P0 gate 已通过，但
尚无最终 P0 winner 和 U/G0/G1/G2 terminal mAP。不得把门禁通过写成方法有效。
历史约 65 的 uniform 结果协议不完全匹配，也不得替换上述 matched control。

## 3. 第一部分：独立重建两套方法

### 3.1 重建 Uni-AdaFocus

用张量级数据流说明：

1. glance frames 如何取样；global CNN 看到了什么；
2. temporal/spatial policies 的输入输出和训练目标；
3. focus frames/patches 如何进入 local CNN；
4. global/local feature 的真实融合位置、映射、池化与分类方式；
5. final/global/local/temporal/spatial/KL/norm 等损失各自作用对象；
6. hard decision 的梯度边界；
7. component-specific learning rates 的真实设置和作用；
8. 哪些设计只适用于视频分类，哪些可能迁移到离线 TAD。

必须给出文件路径、函数名和行号。区分论文叙述与官方代码实际行为。

### 3.2 重建当前 DUCA

画出训练和推理两张完整计算图，至少标明：

- RGB、coarse hidden、actionness、transition evidence、hard positions；
- VideoMAE 输入和输出；
- AdaTAD 各真实模块；
- selected rank、dense candidate index、source-frame time 和预测坐标；
- 每条 detach/straight-through/soft surrogate；
- 每个 loss 到每组参数的真实梯度路径。

给出 `loss × parameter-group` 梯度所有权矩阵。任何配置声称与真实 autograd 不一致处
必须列为高优先级缺陷。

## 4. 第二部分：先判断“是否应该融合”，不得直接进入方案设计

请先独立回答：

1. coarse 二分类/转变特征与 VideoMAE/TAD 特征的监督目标是否互补、冲突或信息重复？
2. 当前性能低于均匀采样，更可能来自 coarse evidence、selection geometry、
   VideoMAE irregular-time semantics、detector optimization、信息丢弃，还是其他根因？
3. 复用 coarse feature 是否真的可能提高 TAD mAP，还是会让廉价但低质量的表征污染
   VideoMAE？给出可证伪预测。
4. 若融合发生在 VideoMAE 后，该方法还能否诚实称为 pre-backbone plugin？合理的
   方法边界和论文命名是什么？
5. 若坚持严格 pre-backbone-only，是否存在不依赖 post-backbone feature fusion 的
   更优路线？
6. Uni-AdaFocus 的 global-feature reuse 为什么在分类中有效；这种机制迁移到高 tIoU
   TAD 时有哪些根本差异？

在这一部分结束前禁止推荐具体融合模块。

## 5. 第三部分：无预设地发散生成候选

在完成前述诊断后，独立提出至少四个**机制实质不同**的候选家族。候选可以包括或
不包括 feature fusion，但不得只是更换 MLP/attention 名称或调 loss 权重。

每个候选必须完整回答：

- scientific hypothesis；
- coarse 与 VideoMAE 各自承担什么不可替代的信息；
- 融合或协作发生在 raw input、backbone token、temporal feature、proposal、logit，
  还是完全不融合；
- 不同时间长度和物理坐标如何对齐；
- 张量形状、核心公式或伪代码；
- 初始化时能否严格退化为 matched uniform/official AdaTAD baseline；
- 各损失更新哪些参数，哪些必须 stop-gradient/freeze；
- hard selection 如何获得任务反馈，是否真实对应部署决策；
- 训练与推理是否同构；
- probe、fusion、heavy backbone、head、decode、H2D 在内的完整成本；
- 对短动作、动作边界、背景运动、长空洞和多实例的预期行为；
- 最大失败模式、审稿攻击点和一项能快速杀死该候选的实验。

不要因为候选新颖就保留它。完成后使用统一评分表淘汰至少一半候选，维度至少包括：

- 与原始间接边界定位目标的一致性；
- terminal TAD mAP 潜力，尤其高 tIoU；
- 严格总成本下降的可能性；
- 训练稳定性与梯度可解释性；
- detector/backbone 可迁移性；
- 相对现有工作的创新性；
- 实现复杂度和最小可证伪性。

## 6. 第四部分：监督冲突与训练设计

对保留下来的候选，必须解决“粗分类监督与 TAD 监督方向不一致”这一核心问题，而
不是简单写成损失加权和。至少讨论并裁决：

1. 是否共享 encoder，若共享到哪一层；
2. actionness、transition、boundary、feature alignment、TAD cls/reg、policy utility
   各损失的参数归属；
3. coarse branch 是冻结、低学习率、adapter-only、部分解冻还是全量协同；
4. 是否需要 gradient surgery、stop-gradient、teacher-student、EMA 或交替优化；
5. 是否需要先独立训练 coarse，再 uniform detector warmup，再联合训练；
6. hard selector 的任务反馈应采用何种可审计机制；
7. 如何避免 TAD loss 破坏 actionness 校准，或 coarse loss 压缩 VideoMAE 语义；
8. 如何检测 representation collapse、policy collapse 和 shortcut learning；
9. 所有权重、阶段长度、冻结点和解冻条件如何由数据/门禁确定，而不是拍脑袋。

请给出最终候选的：

- 总损失公式；
- 分阶段 loss schedule；
- optimizer parameter groups 与 learning-rate ratios；
- 每阶段 `loss × parameter-group` 梯度矩阵；
- 必须记录的梯度范数、余弦冲突、actionness 校准、选帧质量和 mAP 指标；
- fail-closed 训练门禁。

## 7. 第五部分：逐行代码审查与核心实现

在独立确定首选路线后，给出针对精确 DUCA 提交的修改清单：

1. 必须修改的现有文件与行号；
2. 应复用的已有类/函数；
3. 确实缺失时才允许新增的最小模块；
4. 不应触碰的 official AdaTAD/ActionFormer 代码；
5. forward_train、forward_test、optimizer grouping、EMA、AMP、DDP 所需改动；
6. 核心 PyTorch 实现或足够具体的伪代码；
7. shape、mask、padding、physical-time、short-window 和 exact-baseline tests；
8. 能证明梯度归属的 one-step full-model tests；
9. 能证明推理无 GT/teacher/cache 泄漏的 tests；
10. full-stack FLOPs、latency、memory 与 energy 统计入口。

不要重写已经存在的 selector、structured DP、official detector wrapper 或历史路线。
若现有组件足够，必须原位复用；若建议新建模块，先逐项证明当前 V0--V8 没有等价实现。

## 8. 第六部分：最小而决定性的实验闭环

设计一个尽可能小、但能裁决首选路线的 matched 实验矩阵。要求：

- 同 commit、数据、seed、训练更新数、checkpoint 规则和 evaluator；
- 首先隔离“融合本身是否有用”，再隔离“学习选帧是否有用”；
- terminal EMA mAP 是主裁决，selector proxy 只作机制诊断；
- 报告 mAP@0.3/0.4/0.5/0.6/0.7、短动作、边界距离、max hole、选择聚集、
  actionness AUROC/AUPRC/ECE 和完整端到端成本；
- 不允许用额外训练轮次、额外预训练、不同 detector 或隐藏 dense computation 换取优势；
- 给出明确 GO/HOLD/KILL 阈值；
- 如果首选路线失败，只有一个有界 fallback，不继续无限扩展想法。

同时判断哪些后续实验是论文主表必需，哪些只能放 appendix，哪些现在不应运行。

## 9. 强制输出格式

请严格按以下顺序输出：

1. `VISIBILITY_CERTIFICATE`
2. `EXECUTIVE_VERDICT`：GO / HOLD / KILL，先给结论
3. `UNI_ADAFOCUS_RECONSTRUCTION`
4. `DUCA_RECONSTRUCTION`
5. `CODE_AND_GRADIENT_DEFECTS`：按 P0/P1/P2/P3 严重度排序，带行号
6. `SHOULD_FEATURES_BE_FUSED`：先裁决是否融合及原因
7. `DIVERGENT_CANDIDATES`：至少四个实质不同候选
8. `CANDIDATE_ELIMINATION_TABLE`
9. `RECOMMENDED_FINAL_ARCHITECTURE`
10. `FINAL_TRAINING_AND_GRADIENT_CONTRACT`
11. `CORE_IMPLEMENTATION`
12. `MINIMAL_DECISIVE_EXPERIMENT_MATRIX`
13. `COST_AND_PLUGIN_CLAIM_AUDIT`
14. `PAPER_CLAIM_BOUNDARY`
15. `NEXT_BOUNDED_ACTION`：只允许一个首选动作和一个失败 fallback

结论必须区分“代码存在”“静态测试通过”“真实模型梯度通过”“实验运行中”
“terminal mAP 支持”“论文可发表”。禁止跨级。

最后请用一句话回答：

> 在严格计算账本和高 tIoU TAD 目标下，当前 DUCA 最合理的下一步方法边界是什么？
> 为什么？
