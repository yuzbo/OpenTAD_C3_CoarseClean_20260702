# DUCA H65 / TrueTime / UVT / Query-Bridge 对抗性科学裁决请求

## 你的身份与任务

请同时以两种身份工作：

1. 一位极其严格、会拒绝混杂归因和协议不公平的 CVPR 审稿人；
2. 一位有科研审美、愿意给出可执行唯一主路线的时序动作检测（Temporal Action Detection, TAD）第一作者。

不要迎合提问者，也不要把路线选择交回人类。请直接核验下面的固定 GitHub 代码与真实结果，给出唯一的 `CONTINUE / REVISE / PIVOT / STOP` 裁决。若建议继续，必须冻结一个最小、可证伪、非重复的实现—实验方案，目标是解释并超过历史 `65.385724%`，而不是再堆叠模块。

## 仓库与固定代码身份

主仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>

### H65：历史最高的间接非均匀逐帧采样

- 固定提交：[`42dba3f90b37243e7965d18b6707e88e81bf7109`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/42dba3f90b37243e7965d18b6707e88e81bf7109)
- Stage-1 exact-uniform 配置：[`duca_sampling_rate_curriculum_stage1_uniform384.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/42dba3f90b37243e7965d18b6707e88e81bf7109/configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py)
- Stage-2 joint 配置：[`duca_sampling_rate_curriculum_stage2_joint384.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/42dba3f90b37243e7965d18b6707e88e81bf7109/configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py)
- 间接采样器：[`duca_online_frame_selector.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/42dba3f90b37243e7965d18b6707e88e81bf7109/opentad/models/selectors/duca_online_frame_selector.py)
- 检测器接线：[`two_stage.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/42dba3f90b37243e7965d18b6707e88e81bf7109/opentad/models/detectors/two_stage.py)
- 启动器：[`run_duca_sampling_rate_curriculum_stage2_recovery_gpu1.sh`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/42dba3f90b37243e7965d18b6707e88e81bf7109/scripts/run_duca_sampling_rate_curriculum_stage2_recovery_gpu1.sh)

该版本不是普通均匀选帧。它先训练 30 epoch 的 exact-uniform K=384 完整检测器，再从 Stage-1 EMA 初始化进行 60 epoch joint 训练。ASFormer 粗模型预测动作性/转移线索，结合分类与回归贡献蒸馏形成 rate utility，再经累计密度/systematic sampling 产生按原始时间排序的非均匀 K=384 帧；训练还包含 50% uniform companion、`density_transport_st` 和 ASFormer 适配。选中帧按 selected rank 打包成 24 个 16-frame VideoMAE clip，backbone 内部并不显式看到原始时间间隔；物理时间在后端 proposal/NMS 前回映。

真实结果：Stage-2 Job `1191957` 完成 5,000 updates；其作业状态失败仅来自训练后收据保存冲突。只读补评 Job `1193610` 使用同一 epoch-59 EMA checkpoint，在官方 THUMOS14 evaluator 上得到：

| Avg-mAP | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 |
|---:|---:|---:|---:|---:|---:|
| **65.385724** | 80.193191 | 75.662461 | 68.607247 | 58.581766 | 43.883956 |

相对历史 exact-uniform `64.49%`，为 `+0.896 pp`，@0.6/@0.7 为 `+1.312/+1.434 pp`。但它不是公平的 official-60 单变量对照：总训练为 30+60 epoch，并同时改变初始化、采样、蒸馏、uniform companion、梯度路径和 ASFormer 适配。因此不能直接声称“非均匀采样贡献 +0.896”。

### RankPack K384 / TrueTime K384：同快照时间语义配对

- 固定提交：[`11126684af779aa2916a68ecf617c4f14c805478`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/11126684af779aa2916a68ecf617c4f14c805478)
- 配对实现提交：[`ca83be0f1c3691fd6b1a042c1d95c0a4b241977b`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/ca83be0f1c3691fd6b1a042c1d95c0a4b241977b)
- 共同基配置：[`duca_truetime_indirect_curriculum_k384_base.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/11126684af779aa2916a68ecf617c4f14c805478/configs/adatad/thumos/duca_truetime_indirect_curriculum_k384_base.py)
- RankPack：[`duca_rankpack_k384_curriculum.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/11126684af779aa2916a68ecf617c4f14c805478/configs/adatad/thumos/duca_rankpack_k384_curriculum.py)
- TrueTime：[`duca_truetime_k384_curriculum.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/11126684af779aa2916a68ecf617c4f14c805478/configs/adatad/thumos/duca_truetime_k384_curriculum.py)
- 物理时间 patch/tubelet：[`physical_time.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/11126684af779aa2916a68ecf617c4f14c805478/opentad/models/backbones/physical_time.py)
- Backbone 接线：[`backbone_wrapper.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/11126684af779aa2916a68ecf617c4f14c805478/opentad/models/backbones/backbone_wrapper.py)
- Adapter：[`vit_adapter.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/11126684af779aa2916a68ecf617c4f14c805478/opentad/models/backbones/vit_adapter.py)
- 物理时间测试：[`test_duca_truetime_physical_backbone.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/11126684af779aa2916a68ecf617c4f14c805478/tests/test_duca_truetime_physical_backbone.py)

两臂使用同一逐帧语义 scout、同一选中 frame set、K=384、ActionFormer detector/loss/NMS/evaluator、seed=3407 和 20/20/20 curriculum。RankPack 将不等间隔帧按 rank 当作规则 16-frame clips；TrueTime 在第一次 VideoMAE 时间混合前引入原始 physical positions、gap-conditioned tubelet 与 gap-aware adapter，再把 detector proposal 映射回物理时间。需要特别检查：TrueTime 是否仍叠加标准 selected-rank `pos_embed`，从而形成冲突的双重时间坐标。

| Arm | Job | Avg-mAP | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RankPack K384 | 1248822 | 61.5722 | 78.6567 | 73.8490 | 65.3328 | 52.9221 | 37.1003 |
| TrueTime K384 | 1248823 | **62.1930** | 78.7428 | 74.2565 | 65.4630 | **54.6107** | **37.8918** |

TrueTime 单 seed 比 RankPack 高 `+0.6208 pp`，@0.6 高 `+1.6885 pp`，只支持“物理时间机制可能改善高 IoU”的 partial evidence。二者仍比 H65 低 3.19/3.81 pp，但这不是单变量差异：当前路线换了 sampler、训练日程、head、梯度所有权和 representation contract。

### DUCA-UVT：Utility Value / Query Value 诊断

- 训练提交：[`df544c78ce515d925dc7019f106fce09a53c09f8`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/df544c78ce515d925dc7019f106fce09a53c09f8)
- 完整结果分支头：[`59f27d59c322a0e85932eb56448aedc3fb454950`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/59f27d59c322a0e85932eb56448aedc3fb454950)
- 设计：[`2026-08-19-duca-uvt-value-portal-design.md`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/59f27d59c322a0e85932eb56448aedc3fb454950/docs/superpowers/specs/2026-08-19-duca-uvt-value-portal-design.md)
- 配置：[`duca_uvt_value_portal_n16r4.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/df544c78ce515d925dc7019f106fce09a53c09f8/configs/adatad/thumos/duca_uvt_value_portal_n16r4.py)
- Value losses：[`duca_value_learning_losses.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/df544c78ce515d925dc7019f106fce09a53c09f8/opentad/models/losses/duca_value_learning_losses.py)
- Value head：[`duca_value_head_group.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/df544c78ce515d925dc7019f106fce09a53c09f8/opentad/models/selectors/duca_value_head_group.py)
- EMA：[`duca_value_ema.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/df544c78ce515d925dc7019f106fce09a53c09f8/opentad/models/selectors/duca_value_ema.py)
- 结果页：[`duca-uvt-value-portal.md`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/59f27d59c322a0e85932eb56448aedc3fb454950/research-wiki/experiments/duca-uvt-value-portal.md)

Job `1244840`，seed=3407，三臂均 60 epoch + epoch-59 test：`off=57.35`、`geo=55.93`、`geo_ema=55.92` Avg-mAP。`geo` 同时把 `V(t)` 加入 selection score，并把 `mean sigmoid(V)` 用作 dynamic-K evidence；因此是 bundle 负结果，不能归因于 value score、budget evidence 或 EMA 中的任何单项。Boundary-foveated decoder 与 portal feedback 在此种子关闭。

### FoveaSampler / Query-Bridge

- 训练提交：[`4ae5067100c4490c7110c00a1ad406230ba603cd`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/4ae5067100c4490c7110c00a1ad406230ba603cd)
- 完整结果分支头：[`46c714249ff444fcc6428dbe95c52aefe55c488f`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46c714249ff444fcc6428dbe95c52aefe55c488f)
- 批准设计：[`2026-08-19-duca-foveasampler-query-bridge-design.md`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46c714249ff444fcc6428dbe95c52aefe55c488f/docs/superpowers/specs/2026-08-19-duca-foveasampler-query-bridge-design.md)
- Selector：[`fovea_query_bridge_selector.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4ae5067100c4490c7110c00a1ad406230ba603cd/opentad/models/selectors/fovea_query_bridge_selector.py)
- Query bridge：[`query_bridge.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4ae5067100c4490c7110c00a1ad406230ba603cd/opentad/models/selectors/query_bridge.py)
- Scout：[`fovea_scout.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4ae5067100c4490c7110c00a1ad406230ba603cd/opentad/models/selectors/fovea_scout.py)
- Foveated sampler：[`fovea_sampler.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4ae5067100c4490c7110c00a1ad406230ba603cd/opentad/models/selectors/fovea_sampler.py)
- Loss：[`fovea_losses.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4ae5067100c4490c7110c00a1ad406230ba603cd/opentad/models/losses/fovea_losses.py)
- 配置：[`duca_fovea_qb_thumos.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4ae5067100c4490c7110c00a1ad406230ba603cd/configs/adatad/thumos/duca_fovea_qb_thumos.py)
- 结果页：[`duca-fovea-qb-development.md`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46c714249ff444fcc6428dbe95c52aefe55c488f/research-wiki/experiments/duca-fovea-qb-development.md)

Job `1244851`，seed=3407，60 epoch：`baseline_fused=42.94`、`query_only=45.26`、`query_gt_mask=49.16`、`query_cycle=54.67`、`query_fovea=43.77` Avg-mAP。`query_cycle` 是 bundle 内最强信号，但各臂同时改变 mask/cycle/quota/MMR/dynamic budget，尚不能归因。Query 与 cycle 的“前后协同学习/知识传递”可能仍有价值，但当前证据不支持让 Query 直接产生 indices、K 或 proposal。

### 连续 cliplet 负证据

固定 M=24 个互不重叠连续 16-frame clip、K=384 的 frozen-scout / joint 两臂已真实训练并终评：`49.89 / 47.24` Avg-mAP。执行账本确认 heavy VideoMAE 实际只处理 K=384，不是 padding 假稀疏。它排除了“仅靠连续 cliplet 即可修复时间语义”的路线，但不能否定逐帧间接采样、TrueTime 或 Query semantic residual。

## 已知协议边界与禁止重复

1. 官方 dense、exact-uniform、random、actionness-only fixed-K、actionness+boundary fixed-K 已在历史矩阵多次运行。不要建议无条件重复这些完整训练；应优先引用既有冻结回执。若你认为某个同快照控制对因果识别不可替代，必须说明为什么既有结果不能复用，并给出比完整重训更便宜的反事实或权重复用方案。
2. `65.385724` 是真实结果，但 H65 总训练成本和多项机制混杂，不能冒充 official-60 单变量提升。
3. `64.352` 是较接近原生检测器的协议未完全匹配锚点；`65.696` 是 grid-aware physical geometry 的非严格可比探索。两者都不能被重命名为论文正证据。
4. UVT、Fovea、H65、TrueTime 来自不同提交和不同训练合同，不能直接按 mAP 排名后归因。
5. 最终路线必须以 dynamic outer-K 为核心；fixed K 仅能用于因果归因、基线与安全回退。
6. 任意非连续帧必须保留原始 timestamp/physical coordinate，并在 threshold、top-k、proposal decode 与 NMS 前保持一致；禁止把稀疏 selected rank 冒充均匀物理时间。
7. 不得声称已有论文级效率结果；必须报告 VideoMAE 实际输入量、端到端吞吐/显存/计算成本以及准确率—成本关系。

## 需要你攻击并直接裁决的问题

### A. 解释历史 65.385724

请按可验证源码逐项判断，H65 相对 exact-uniform 的信号最可能来自哪些因素：

- 30 epoch uniform Stage-1 + EMA warm start；
- 额外 60 epoch joint optimization；
- ASFormer action/transition hidden adaptation；
- classification/regression contribution distillation；
- density/rate systematic sampling；
- 50% uniform companion；
- `density_transport_st` 的梯度路径；
- selected-rank VideoMAE 的预训练先验兼容性；
- proposal physical-time remap。

不要给泛泛排序。请指出 file/symbol 证据、可能的因果链、最便宜可证伪观察，以及哪些机制可以共同保留、哪些必须隔离。

### B. 解释当前 3–4 pp 下降

请核验 RankPack/TrueTime 是否真的只差首次时间混合，检查下列风险：

- 当前 scout/semantic utility 是否已偏离 H65 的 ASFormer + contribution distillation；
- 20/20/20 official-60 curriculum 是否破坏了 H65 的 warm-start 优势；
- 当前 physical ActionFormer head/assignment 是否同时改变 detector geometry；
- TrueTime 是否同时叠加 physical positions 与标准 selected-rank `pos_embed`；
- gap-conditioned tubelet、mask、padding 或 coordinate MLP 是否改变 VideoMAE 预训练分布；
- 当前梯度所有权是否让 detector 过早扰动 selector；
- 训练后 archive/hash 错误是否只影响证据封存而不影响指标。

请给出“首要原因—次要原因—已被证伪原因”的明确判断，并区分代码事实与科学推断。

### C. 如何吸收 H65，而不退回旧协议

请在以下候选中冻结唯一主路线，或提出一个严格更好、仍满足 dynamic outer-K 的单一替代：

1. **H65 compatibility contract**：先完整保留 H65 的 semantic utility、density sampler、warm start、uniform companion、selected-rank detector/head，只将 heavy backbone 的时间解释从 RankPack 改为 TrueTime；
2. **公平 official-60 压缩**：用 detector-free scout pretraining、teacher/EMA 或 curriculum 压缩替代额外 30 epoch full-detector 成本，同时保持 H65 语义目标；
3. **Query semantic residual**：保留 Query-Bridge/cycle 的协同训练价值，但 Query 只修正 action/boundary semantic representation 或作为 detached teacher，不直接决定 indices、K、proposal 或 detector utility；
4. **Dynamic outer-K 最后接入**：先在 K384 归因门恢复 H65 水平，再让训练人口 FIT/CAL 上校准的 semantic evidence 决定 per-video/window K；固定 K 只保留为控制。

请明确：如何保留 VideoMAE 预训练时间先验，同时又让不等间隔逐帧输入具有真实物理时间；是否应该只在 attention bias/relative-time 中加入 gap，而去掉冲突的 rank absolute position；是否应暂时保留 H65 selected-axis detector，仅在 pre-NMS 做 q→t 物理回映。

### D. 最小且不重复的实验

请输出一个**最多 3 个新增完整训练单元**的决策矩阵。不要重跑已有 official dense/uniform/random。矩阵必须：

- 使用完整 THUMOS14 official split/evaluator，N16R4，明确 seed；
- 复用历史 frozen receipts 作为背景，而新增单元必须在同一 frozen commit 中形成可归因比较；
- 首个单元应能区分“H65 semantic/training contract 丢失”与“TrueTime representation 有害”；
- 第二个单元应检验 Query semantic residual 是否提供 H65 之外的新信息；
- 只有前两项通过时，第三个单元才允许加入 dynamic outer-K；
- 每 5 epoch 保存完整可恢复 checkpoint，保留 latest-3、milestone、final、final-EMA；final/final-EMA 规则预注册，禁止按中间验证挑最好；
- 报告 Avg-mAP、各 tIoU、尤其 @0.6/@0.7、短动作/边界风险、selected-frame 覆盖、K 分布、VideoMAE 实际帧/clip 数、端到端吞吐/显存/训练成本；
- 给出明确 kill rule、fallback 和 result-to-claim 边界。

### E. 论文有趣性

最终方法必须回答：

1. 哪个观察会让 TAD 审稿人意外？
2. 什么领域机制解释它？
3. 它产生了什么 uniform/fixed-K baseline 不会产生的新预测？
4. 哪个真实视频实验最便宜地证伪该预测？
5. 哪些 prior-art/novelty 事实会立即杀死论文主张？

如果你认为“物理时间修复 + semantic sampler”仍只是工程组合，请明确指出缺少的科学命题，并给出一个更优雅但不扩张为新模块堆叠的替代。

## 强制输出格式

请用中文给出一份可直接交给 Builder、Critic、Evaluator 的终稿，包含：

1. **唯一裁决**：`CONTINUE / REVISE / PIVOT / STOP`；
2. **代码核验边界**：哪些链接实际看过，哪些结论只是材料陈述；
3. **H65 因果解释**：最可能因素及 falsifier；
4. **当前下降解释**：首要/次要/已证伪原因；
5. **唯一冻结路线**：问题、机制、创新点、意外预测；
6. **最小改动面**：精确文件、symbol、数据流、梯度流和时间坐标合同；
7. **最多 3 个新增训练单元**：config、seed、split、evaluator、checkpoint、cost、stop；
8. **禁止复用或重复的历史实验**；
9. **claim / anti-claim / falsifier / novelty invalidator**；
10. **下一步立即动作**：若只能做一件事，具体做什么。

不要提出第二轮理论讨论，不要要求人类从多个方案中选择，不要用设计或静态测试冒充实验结果。
