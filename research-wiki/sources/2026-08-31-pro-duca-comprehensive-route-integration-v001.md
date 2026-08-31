# DUCA 全历史、代码与论文路线综合裁决

## 1. `SESSION_ASSERTION`

**Nonce：`DUCA-COMPREHENSIVE-ROUTE-INTEGRATION-v001-20260831`**

* ChatGPT Project ID：`g-p-6a91061f789881918ccd8357ca3d6c92`
* GitHub repository：`yuzbo/OpenTAD_C3_CoarseClean_20260702`
* 唯一 H65 模型与训练基座：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`
* whole-video 诊断功能提交：`33e4ed137c33eef07f0452b44506a6993bdf7535`
* 代码库存提交 `5136011ed57df8a639427a633a488a592ba95924` 仅用于发现重叠实现，不是实验或模型身份。
* 科学决定由本报告独立作出；Builder、Critic 与 Evaluator 只执行冻结后的单一任务，这与项目角色规则及论文优先原则一致。
* 数据、基线、公平更新数、无 held-out 泄漏、真实重型计算和预先冻结模型选择规则按项目规则处理。
* 本轮完整吸收了旧路线问询、材料包说明、当前研究状态、原始 PJST 评测回执和历史版本注册表。
  我完整读取了本轮上传的九个材料文件，并逐段核对了用户正文中给出的证据索引、完整数据协议、论文进展、决策历史、历史模型合同和实验页。

### 精确提交检查范围

已在相应精确提交上检查主要生产符号、配置或最终诊断入口：

`04c35a3...`、`33e4ed...`、`1642f265...`、`b3222af...`、`dd3c97cf...`、`ed0d4900...`、`11126684...`、`b3339112...`、`d127c2b2...`、`04814312...`、`f67d96fd...`，并额外检查了 PJST-D1 的 `c73e8418...` / `7bd120f0...` 和 SingleClock 的 `b2ccfcca...`。

以下历史实现只完成了提交身份或证据记录核验，未完成全部生产符号逐行展开，因此对其代码级机制细节明确标为 `NOT_INSPECTED`：旧 E2E `70aa069...` 的完整生产 forward、UVT `df544c78...` 的全部核心模型符号、Fovea/Query-Bridge `4ae50671...` 的全部核心模型符号，以及部分 boundary-burst 历史中间提交。对这些路线的处置只依赖已有正式结果和明确协议边界，不把摘要冒充逐行源码审查。

这一限制不影响唯一主路线判断，但意味着本报告不对这些 `NOT_INSPECTED` 分支作比现有原始记录更细的组件因果归因。

---

## 2. `SCIENTIFIC_DECISION`

**唯一裁决：`REVISE`**

保留的唯一科学方向是：

> **在现有 H65 嵌套 K256/K384/K512 选点构造下，通过训练期的多预算暴露建立预算兼容的稀疏 TAD 系统；只有该机制成立，才进一步研究由低成本语义证据驱动的真实预算分配。**

这不是对 Gemini 建议的机械接受，而是基于以下独立判断：

1. 冻结 K384 检测器上的 Marginal-v1、96-state 和 704-state 结果已经充分关闭旧预算转移动作空间，但没有检验经过多预算训练的系统。
2. 多预算训练是当前动态计算论文路线的必要前提：如果相同位置构造下的检测系统不能兼容 K256/K384/K512，任何语义 controller 都没有可靠的执行基础。
3. TrueTime、tubelet pairing、重构核和等更新课程仍有未闭环问题，但它们分别只回答表示或优化的局部问题，不能直接恢复已经失败的动态预算主张。
4. 第一轮只改变训练预算暴露，具有比新增时间编码、重构核或 selector 更清晰的 falsifier。
5. 现有 H65 Stage-2 不只训练检测头，还继续适配 coarse trunk、action head、transition scorer、ASFormer policy path 和 detector-feedback path。因此，第一轮的准确名称必须是 **“H65 系统的多预算暴露适应”**，而不是严格的“纯检测器适应”。`detector-only` 因果表述被撤销。

当前只有一个可执行任务：**完成 THUMOS14 全量 train/held-out 数据身份核验并解决 211/212 差异。** 数据准入前，不建立模型分支、不加载 checkpoint、不提交 GPU、不训练，也不生成新 held-out 预测。

---

## 3. `EVIDENCE_INTEGRITY_AUDIT`

### 3.1 Shared Dense AdaTAD

可信事实是 Avg-mAP `68.73`、768 个重型输入 observation。它是共享 dense 参考上界，不重复训练。

不可声称：

* 当前没有与 H65 完全统一的 mAP@0.7 原始表；
* 没有同硬件完整 latency、throughput、memory、energy 配对；
* 因而不能从 `384/768` 直接推出“端到端计算降低 50%”。

允许表述是：

> H65 将重型视觉路径的输入 observation 数从 768 降到 384。

### 3.2 H65 fixed K384

H65 是当前最可靠稀疏基线，登记结果约为 Avg-mAP `65.1257`、mAP@0.7 `43.3137`；另一封存终点为 `65.385724/43.883956`。H65 不是普通全局 Top-K，而是由动作状态和变化证据调制、含覆盖下限的预算校准系统采样。配置链同时启用 transition supervision、贡献蒸馏、uniform companion、detector feedback 和 ASFormer adaptation，因此是复合配方。

Stage-1 为 30 轮、约 3,000 updates；Stage-2 为新 optimizer/scheduler/AMP/EMA 时钟下的 6,000 updates。历史完整训练因此是 9,000 updates，而不是与 60 轮 exact-uniform 相同的 6,000 updates。

因此：

* H65 相对 60 轮 uniform 的约 `+0.90` Avg-mAP 是重要正向迹象；
* 但不能把差值完整归因于 selector、课程或贡献蒸馏；
* 下一主比较必须重新训练同 6,000-update 的 K384 控制，不能用历史 65.13 直接充当因果控制。

### 3.3 旧 fixed-K384 end-to-end

正式点估计 `58.39/34.53` 仅证明旧实现失败。已确定的混杂是 loader/update exposure 不匹配。随机 coarse stem 和 surrogate-gradient 漂移是合理解释，但没有被独立隔离。

因此：

* 不允许写成“联合训练一般不可行”；
* 不原样恢复旧实现；
* 保留为训练设计失败的历史诊断。

完整生产符号：`NOT_INSPECTED`。

### 3.4 Transition-only、CellCF 与 utility distillation

CellCF 的正式结果为 uniform `63.8594`、transition-beta0 `64.2755`、CellCF `64.0610`。

精确代码显示：

* `local_cell_deformation` 把时间轴划分为 exact-uniform cells，每格恰选一个位置；
* 预算无法跨 cell 转移；
* counterfactual utility 来自冻结或 detached detector 的 hard alternatives，并通过 signed logistic 学习，不是检测损失对 hard index 的直接梯度。

结论：

* CellCF 未超过 transition 控制；
* 该 one-per-cell 机制已被正式否定；
* 不得通过调 loss weight、扩大 cell 或增加轮次恢复。

### 3.5 Protected-E2E、direct、homotopy、uniform companion

匹配结果为：

* exact uniform `64.4580`
* direct `63.7102`
* homotopy `63.0601`
* companion `63.6931`

精确实现的 hard forward 仍是按实际选中位置 gather RGB；soft `[B,K,T]` assignment 只通过 zero-forward residual 向 selector 提供近似梯度。它是受保护 surrogate，不是 LongTensor hard index 的真实导数。配置固定 K384、global structured selection、约 `0.25` detector-gradient 权重，并可让最后的 ASFormer 时序层接受受限梯度。

已有正式结果足以停止该 exact family：

* 不重跑 direct bridge；
* 不重跑 score homotopy；
* 不重跑相同 uniform companion；
* 不把“梯度非零”解释成 hard utility 对齐。

### 3.6 Boundary-burst 与 R0

R0 使用训练内部 privileged holdout，回答可行空间是否有足够 headroom，不是 official held-out 模型性能。

虽然点值中 R2Q3、R4Q5 和 unrestricted oracle 高于 uniform，但 paired gate 未通过。该结果关闭的是已冻结的 projected boundary-burst family，不是否定所有边界优先采样。

禁止：

* 降低门槛；
* 改 radius/quota 后重跑同一 R0；
* 把内部 90% 以上的 mAP 与 official validation 数值比较；
* 把 oracle 当作部署方法。

部分历史中间提交生产符号：`NOT_INSPECTED`。

### 3.7 Fast-only SlowFast prior

`63.5297/42.0937` 低于 matched uniform `64.49/42.45`。这是否定冻结 Fast-pathway + 当前 R2Q3 的结果，不是否定所有低成本视频模型。

处置为该 exact prior `STOP`。

### 3.8 训练课程、压缩与 K192

20+40、AM-RPCH25 和 LongCosine-H6000 都没有恢复 H65。它们证明简单压缩 warmup 或只改第二阶段学习率尾部不足以恢复性能，不证明所有 60 轮轨迹在理论上无效。

K192 的配置继承 K384 的 30+60 课程，只把 budget 改为 192、VideoMAE chunks 改为 12，并保留同一 Stage-2 feedback/contribution/ASFormer 适配结构。它同样是 90 轮，且缺 matched native-K192 uniform。

因此：

* 已测试的压缩族 `STOP`；
* K192 结果只作低预算诊断；
* 不以更多 scheduler 搜索恢复。

### 3.9 PhysTime、RankPack、TrueTime

证据是混合的：

* PhysTime v1 为负；
* physical-metric full60 在特定低性能架构中产生约 `+16.29` Avg-mAP；
* RankPack/TrueTime matched 为 `61.5722` 对 `62.1930`，TrueTime 约 `+0.6208` Avg、`+0.7915` mAP@0.7；
* 但都是单种子或非 H65 主架构证据。

一项重要代码更正是：

> `11126684...` 的 RankPack/TrueTime 正式配置通过 `physical_time=True`、`PhysicalTimeTubeletEmbedding` 和物理 gap adapter 改变首次二帧时序混合；它们没有实例化随附的 `TrueTimeFeatureResidual`。

因此，不能把 `TrueTimeFeatureResidual` 写成这组 `+0.6208` 结果的生效机制。

物理时间仍是有效诊断问题，但不应抢在多预算兼容性之前。

### 3.10 `TrueTimeFeatureResidual`

该类本身具备：

* 末层零初始化，初始 forward 为严格 identity；
* 归一化绝对位置、左右 gap 和局部不对称度四维描述；
* padding residual 清零。

但存在四个需要明确的实现边界：

1. `_descriptors` 用 `active = masks.sum()` 后写入 `rows[:active]`，隐含有效 token 必须是连续前缀。若 mask 有内部空洞，描述子会与 token 错位。
2. `constant` 不是无时间信息控制：它保留了每个视频自身描述子的均值，只删除 token 间变化。
3. `reversed` 只翻转描述子行序，不重新计算“反向物理时间轴”的 gap，因而不是严格物理反演控制。
4. `left[0]=pos[0]+1` 和 `right[-1]=valid_len-pos[-1]` 包含端点外侧支撑，不能直接解释为普通相邻帧间隔。

在未来复用前必须增加 active-prefix assertion 和清晰的控制语义。该类当前归为未调用历史表面，而不是当前可靠方法核心。

### 3.11 SingleClock 与 PJST-D1

SingleClock 已有生产入口，能够从 `ActionFormer` 向 VideoMAE 传递原始位置、有效 mask 和 gate-zero 状态，并使用零初始化有界 gate。它的实施周期因 canonical clock 和 checkpoint 身份问题关闭，没有正式效能结论。

PJST-D1 则在首次二帧混合中保留普通 pair mean，只按真实时间间隔缩放 derivative term；exact canonical uniform 输入旁路原 PatchEmbed，且没有新增学习参数。

PJST-D1 的可信证据为：

* OFF/ON 均覆盖完全相同的 211 个视频；
* 每臂 422,000 predictions；
* 所有点值精确复现；
* ON−OFF Avg-mAP `−0.472481` pp；
* mAP@0.7 `+0.122739` pp；
* finalizer 因平铺路径与 `gpu1_id0` 实际路径不一致而在抽样前退出；
* `0/10,000` bootstrap，没有 CI。

裁决：

* 不宣布显著负向；
* 不把 mAP@0.7 的小正差称为收益；
* **当前不补 PJST bootstrap**。原因是该区间不会改变唯一主路线，且旧 211-video 身份尚未完成数据准入；
* 保留 sealed predictions 作为以后论文补充材料的候选，不重训、不改模型。

### 3.12 Native tubelet 与 continuous cliplet

`b3339112...` 把 768 帧组成 384 个连续二帧 native tubelet，再选 192 个。候选 score 为 actionness、boundary 和 novelty 的固定组合，并强制端点和最大空洞；均匀臂和 coreset 臂都使用同一种 native tubelet。

因此 `64.13/42.45` 对 `62.81/40.56` 只否定当前 task-state coreset 选择规则，不检验连续 pair 相对非连续 pair 的因果效应。

FZ_CONTIG/JT_CONTIG 的 `49.89/47.24` 同时改变 acquisition 和 packing，也不能作为 pairing-only 实验。

结论：

* task-state tubelet coreset `STOP`；
* CONTIG bundle `STOP`；
* pairing continuity 保持未回答，但不进入当前主线。

### 3.13 Sparse hidden-linear bridge

实现已支持 nearest 与 hidden-linear，对稀疏 anchor 上的 logits、encoder hidden 和 policy hidden 按原时间位置恢复至 768 点；selector 不接收 anchor mask 或距离旁路。

CUDA gate 证明了形状、有限值、梯度和 MACs 趋势。正式 TAD 在首个 optimizer update 前因 BatchNorm buffer 和 runtime variant mapping 失败，没有 TAD mAP。

因此：

* 不是科学负结果；
* 不重建同一 bridge；
* 归档，只有当未来选择 kernel-only 研究时才复用。

### 3.14 Dynamic native-tubelet window budget

`d127c2...` 的生产 forward 会按 16/20/24 clips 分组，真实截取 256/320/384 帧后分别调用重型 backbone，只在 backbone 后恢复共同 detector grid；它不是 padding 伪动态。

但它没有正式训练、mAP、CI 或实测成本，且预算集合不同于当前 K256/K384/K512 适应问题。保留为真实变长执行参考，不作为当前模型身份。

### 3.15 Coverage-v1

`TemporalCoverageSelector` 是固定预算 facility-location 贪心，使用 H65 priority 和固定时间 anchors，不通过梯度训练。真实 200-sample 无标签 replay 得到：

* 集合变化 `48.05% < 80%`
* anchor coverage 增益 `3.32% < 10%`
* max-gap P95 从 2 恶化至 8

这是有效的训练前机制门失败。

该 exact selector `STOP`，禁止降门或 gap repair。

### 3.16 Marginal-v1、cap-release、96-state

`forward_marginal_prefixes` 先封存 K384 H65 selection，再构造嵌套反事实。K256 是 K384 的子集；K512 在 K384 外补入未选高优先级位置，最后保持严格时间顺序。

Marginal producer 对 K256/K512 构造真实 packetized tensor，而不是只改 metadata。

但是 capped、released 和 96-state 均未达到 `+0.8/+1.0` 联合门。这关闭加性窗口 utility 和其局部联合修复，不证明跨预算训练无效。

### 3.17 Whole-video 704-state transfer

`33e4ed...` 的末次改动恢复 sealed producer 的原 proposal 顺序，避免 sample-ID 重新排序改变 Soft-NMS tie handling。它没有改变模型、三档预测、成本或门槛。

runner 的正确边界是：

* 候选集在读取标签和指标前生成；
* 只使用旧 40-video training-side holdout；
* 704 个合法状态；
* 没有模型 forward、训练、gradient 或 bootstrap；
* 结果 0/704 只否定冻结 K384 检测器、当前 nested tiers 和当前 transfer space。

runner 内的 `WHOLE_VIDEO_NO_PASS_PROJECT_LEVEL_STOP` 状态名比科学证据更宽。正确停止范围必须按上述条件限定。

### 3.18 Multi-budget adaptation

当前没有 Builder commit、PRE_RUN、训练或模型结果。它是尚未执行的可证伪假说，不能被写成已解决问题。

---

## 4. `LINE_BY_LINE_IMPLEMENTATION_REVIEW`

### 4.1 H65 `acquisition.py`

**`opentad/models/duca/acquisition.py:48-57`**

七维 deploy-visible 状态为：

`p_action`、`uncertainty`、`entropy`、`delta_p_action`、`abs_delta_p_action`、`uncertainty_peak`、`transition_score`。

这些是当前应保留的低成本语义证据，不引入 validation/test teacher。

**`:267-380` — `SparseTemporalGrid.validate`**

可靠性质：

* 原时间坐标；
* 每样本 `valid_len`；
* requested/effective budget；
* 严格递增且无重复的 physical positions；
* selected mask 与 positions 必须完全一致；
* detector input length 与 selected count 一致。

这是下一候选应继续复用的核心不变量。

**`:2628-2690` — `_decode_budget_calibrated_sampling_rate`**

关键限制：

```text
if torch.any(budgets != int(self.budget)):
    raise ValueError(...)
```

因此现有 H65 BCSR decoder 是配置固定预算的，不能通过简单传入 K256/K512 直接变成多预算训练。下一实现必须使用已经存在的 nested-prefix 构造，而不是修改 BCSR 数学、重新运行每档“预算原生 selector”或在 decoder 内偷偷改变可行集。

**`:2911-2924` — 当前 cost ledger**

当前通用 dynamic path 把：

* `padded_detector_k`
* `backbone_input_k`

都设置为配置最大 budget，并明确记录：

```text
dynamic_compute_realized = False
dynamic_compute_blocker = detector_tensor_is_padded_to_budget_max
```

该路径是成本伪动态，禁止用于下一正式候选。

**`:3618-3659` — `gather_selected_observations`**

该函数按严格递增的 selected positions gather，并验证 dense mask 一致。可以复用，但它本身不保证输出 tensor 没有被 pad 到 max K；真实变长必须由下游 producer/forward 另行保证。

### 4.2 Stage-1 config

**`duca_sampling_rate_curriculum_stage1_uniform384.py:3-26`**

文档与代码一致：

* exact-uniform K384；
* full-model warmup；
* detector 学习相同的 K384 observations；
* learned policy 与 detector-to-selector feedback 关闭。

**`:28-62`**

* `inference_policy_alpha=0`
* contribution distillation `0`
* detector-gradient schedule `0`
* ASFormer adaptation schedule `0`
* actionness/transition/boundary supervision仍开启。

**`:71-90`**

* 每 5 轮验证只作 learning curve；
* 不选择 checkpoint；
* 主 checkpoint 为 epoch 29 `state_dict_ema`。

### 4.3 Stage-2 config

**`duca_sampling_rate_curriculum_stage2_joint384.py:13-29`**

* Stage-1 checkpoint path/SHA/epoch 必需；
* seed 3407；
* 6,000 updates。

**`:31-46`**

配置声明为完整 joint rate adaptation，不是 detector-only。

**`:48-110`**

* coarse trunk、action head、transition scorer 均有非零 LR；
* actionness、transition、transition-boundary supervision 保留；
* policy alpha 0→1；
* detector gradient 0→0.25；
* detector contribution 0→1；
* ASFormer adaptation 0→1。

因此下一实验的估计量是：

> 在相同 Stage-1 起点、相同可训练参数和相同优化下，仅改变训练预算暴露分布，对整个 H65 Stage-2 系统产生的总效应。

不得称为“只有 detector 参数改变”。

**`:113-139`**

* 中间 validation 不选模；
* 从 Stage-1 EMA 初始化；
* reset selector schedule step；
* terminal EMA 是正确结果模型。

### 4.4 `TrueTimeFeatureResidual`

**`:27-33`**：末层零初始化，identity 成立。

**`:44-60`**：严格递增位置、四维描述子；端点 gap 语义需要在论文中明确定义。

**`:74-95`**：隐含 active-prefix mask，不允许在未来无 assertion 地复用。

**`:91-94`**：`reversed` 和 `constant` 不是严格的物理时间反演与无时间控制。

**`:98-118`**：输出为 feature residual；未改变检测头，但当前 matched TrueTime 结果没有调用此类。

### 4.5 `dynamic_budget.py`

**`:147-187` — `marginal_budget_accounting`**

正确实现：

* `actual=min(valid, requested)`
* 无真实差异的短窗口折叠回 baseline；
* nonbaseline 执行按 16 帧 packet 对齐；
* padding 限制在最后一个 packet。

**`:218-265` — `validate_real_heavy_observation_tensor`**

正确检查：

* 真实 tensor 长度等于 execution slots；
* active mask 必须是前缀；
* nonbaseline padding 少于一个 packet；
* padding 值严格为零。

这是下一模型可以移植的核心。

**`:538-707` — `PrefixMarginalUtilityBudgetController`**

这是旧 dynamic controller：

* 通过 top-ranked blocks 和 soft/hard prefix 决定预算；
* 带 Lagrange dual；
* 不是当前多预算训练所需组件；
* 与 `acquisition.py:2914-2923` 组合时仍为 max-padded 伪动态。

禁止移植到第一轮。

### 4.6 Whole-video runner

**`:98-159`**

正确保留 producer 原顺序；K256/K384/K512 sample sets 必须一致。

**`:243-290`**

候选只基于视频 ID 和 actual cost，在标签打开前完整生成，科学顺序正确。

**`:398-444`**

仍硬绑定旧 160/40 split，因此只能是历史 training-side diagnostic，不能进入新正式协议。

**`:521-578`**

PRE_RUN 在 candidate manifest 后才加载标签并复现 anchors，适合历史 falsifier，但不适合当前无标签数据准入任务。

**`:628-767`**

只复用 sealed predictions 调 evaluator，没有训练或 forward。0/704 是动作空间 falsifier，不是新模型性能。

### 4.7 CellCF

* `local_cell_deformation`：one-per-cell，结构性限制；
* `counterfactual_utility.py`：detached detector utility；
* 配置明示 local-cell K384。

文档描述基本忠实，但任何“detector gradient 直接训练 CellCF”表述均错误。

### 4.8 Protected-E2E

* hard forward 与 soft backward 位于同一 structured selection family；
* detector input 的实际值保持 hard gather；
* soft assignment 只产生零 forward 的 backward residual。

代码忠实于“protected surrogate”，但不能证明 surrogate 与真实 hard swap 的效用方向一致。旧结果已经停止该 exact surrogate。

### 4.9 Native tubelet

* `tubelet_coreset.py` 正确形成连续二帧 native units；
* candidate 和 uniform 共享 pairing；
* candidate 唯一改变 selection score；
* 当前结果不回答 pairing continuity。

### 4.10 Dynamic native tubelet

`ActionFormer._forward_heavy_backbone` 按不同 clip count 分组并在 backbone 前真实 slice，属于可安全参考的 variable-compute 实现，但其 16/20/24 预算语义不应直接混入当前 K256/K384/K512 实验。

### 4.11 PJST-D1

* `TemporalGridPatchEmbedWrapper` 将 physical pair metadata 送入首次混合；
* derivative-only 模式保留 pair mean，只缩放差分项；
* canonical uniform 旁路原 embedding；
* terminal evaluator 的路径失败发生在预测完成后，不影响点值。

### 4.12 `NOT_INSPECTED` 生产符号

* UVT `df544c78...`
* Fovea/Query-Bridge `4ae50671...`
* 旧 E2E `70aa069...`
* 若干 boundary-burst 中间提交

这些路线不能在本报告中获得比原始结果更细的代码因果归因。

---

## 5. `ROOT_CAUSE_SYNTHESIS`

### 5.1 已证实

1. **H65 历史主结果存在训练预算混杂。**
   30+60 相对 60 轮 uniform 不能作纯 selector 或 curriculum 因果归因。

2. **旧 E2E 存在严重 update exposure 不匹配。**
   它使 `58.39/34.53` 不能回答联合训练一般可行性。

3. **CellCF 可行集限制预算跨区域移动。**
   这与结果低于 transition 控制一致。

4. **Protected/direct/homotopy exact family 未超过 uniform。**

5. **Coverage-v1 没有产生足够大的预注册选点干预。**

6. **当前 task-state native tubelet coreset 低于 matched uniform。**

7. **加性 Marginal-v1、cap-release、96-state 和 704-state transfer 均没有达到预注册联合 headroom。**

8. **PJST-D1 缺 CI 的直接原因是 finalizer prediction path 错误。**

9. **Sparse hidden-linear 没有 TAD mAP 的原因是工程和 runtime mapping 失败，而不是性能失败。**

10. **旧通用 dynamic acquisition path 会把 backbone input pad 到最大 budget，不能证明真实动态计算。**

### 5.2 最强但未证实的假说

1. **跨预算兼容性假说。**
   只在 K384 上训练的 H65 系统没有学会对 K256/K512 的重型表示分布保持稳定；多预算暴露可能恢复兼容性。

2. **局部 pairing continuity 假说。**
   非连续物理帧在 VideoMAE 二帧 tubelet 内被当作邻近运动可能造成表示偏移；现有实验没有严格隔离。

3. **重构核假说。**
   hidden-linear 可能平滑边界，但 nearest/linear/Gaussian 尚无正式 matched TAD 比较。

4. **物理时间假说。**
   时间度量在部分架构中很重要，但在 H65-compatible 表示上的稳定收益未成立。

### 5.3 纯工程失败

* launcher 环境；
* pretrained path；
* block-list 格式；
* prediction path；
* proposal 额外排序；
* save_dict 配置；
* BatchNorm buffer validator；
* runtime variant mapping；
* canonical clock 构造。

这些不作为模型负结果。

### 5.4 当前未知

* 最终 admitted held-out 是 211 还是 212；
* 历史 ActionFormer 212 的 literal source；
* 多预算暴露是否改善 fixed mixed workload；
* 效应主要发生在 detector/adapter 还是 Scout/selector；
* 三种子稳定性；
* 完整端到端 latency、throughput、memory 和 energy；
* 语义 controller 是否优于同预算 content-independent manifest。

---

## 6. `ROUTE_DISPOSITION`

| 路线                                            | 处置                   | 冻结理由                              |
| --------------------------------------------- | -------------------- | --------------------------------- |
| Shared Dense AdaTAD                           | `RETAIN_BASELINE`    | 768-observation 共享上界；不重复训练        |
| Exact-uniform K384                            | `RETAIN_BASELINE`    | 必需的 matched 稀疏控制                  |
| Random / actionness-only / simple-transition  | `RETAIN_BASELINE`    | 只读归因基线；不机械重跑                      |
| H65 fixed K384                                | `RETAIN_BASELINE`    | 当前最可靠稀疏系统；历史 90 轮结果只作参考           |
| Old E2E 58.39                                 | `RETAIN_DIAGNOSTIC`  | update exposure 混杂；禁止原样恢复         |
| Transition-beta0                              | `RETAIN_DIAGNOSTIC`  | CellCF 的较强简单控制                    |
| CellCF / local-cell                           | `STOP`               | one-per-cell 与正式负结果               |
| Protected-E2E / direct / homotopy / companion | `STOP`               | 全部低于 matched uniform              |
| Boundary-burst projected R0                   | `STOP`               | 训练内部 oracle 未过门                   |
| Fast-only SlowFast prior                      | `STOP`               | exact prior 负向                    |
| 20+40 与 30+30 scheduler compression           | `STOP`               | 已测试压缩族未恢复                         |
| K192 30+60                                    | `ARCHIVE`            | 超预算且缺 matched native control      |
| PhysTime / physical-metric                    | `RETAIN_DIAGNOSTIC`  | 有局部强效应但非 H65 主结果                  |
| RankPack / TrueTime                           | `RETAIN_DIAGNOSTIC`  | 小幅单种子正向证据                         |
| `TrueTimeFeatureResidual`                     | `ARCHIVE`            | 当前正式结果未调用，且控制语义不闭合                |
| SingleClock                                   | `RETAIN_DIAGNOSTIC`  | 生产入口存在，无有效效能结论                    |
| PJST-D1                                       | `RETAIN_DIAGNOSTIC`  | 点估计无平均收益，CI 缺失；当前不补               |
| Native tubelet coreset                        | `STOP`               | 当前 selection package 明确低于 uniform |
| CONTIG bundle                                 | `STOP`               | 完整训练大幅下降且多变量                      |
| Pairing-only continuity                       | `ARCHIVE`            | 未回答，但优先级低于预算兼容                    |
| Sparse hidden-linear                          | `ARCHIVE`            | 工程未闭环；不重复搭建                       |
| Dynamic native-tubelet 16/20/24               | `ARCHIVE`            | 真实变长参考，无效能证据                      |
| Coverage-v1                                   | `STOP`               | 预注册干预门失败                          |
| Marginal / cap-release / 96-state             | `STOP`               | 当前加性效用空间关闭                        |
| Whole-video 704-state transfer                | `STOP`               | 冻结 K384 detector 的 transfer 空间关闭  |
| UVT                                           | `ARCHIVE`            | 多变量、跨版本、性能低；核心符号 `NOT_INSPECTED`  |
| Fovea/Query-Bridge                            | `ARCHIVE`            | 多变量、跨版本、性能低；核心符号 `NOT_INSPECTED`  |
| H65 多预算暴露适应                                   | `CONTINUE_CANDIDATE` | 唯一高信息增益且未执行的必要机制                  |
| Learned semantic controller                   | `ARCHIVE`            | 只有适应机制通过后才可另行解锁                   |

明确禁止重复：CellCF、direct/homotopy、Fast-only、Coverage-v1、Marginal/cap-release/96-state、704-state、native coreset score 调参、CONTIG bundle、旧 60 轮 scheduler 搜索和旧 E2E 原样重跑。

---

## 7. `UNIQUE_RESEARCH_ROUTE`

### 7.1 一句话论文问题

> **任务感知的稀疏时序动作检测系统能否通过等重型输入成本的多预算训练暴露，获得跨真实 observation budget 的兼容性，并由此支持语义驱动的可变计算而不牺牲高 tIoU 边界定位？**

### 7.2 最终论文机制

按证据依赖顺序分两阶段，但属于同一研究路线：

1. **预算兼容训练。**
   用同一 H65 priority sequence 构造嵌套 K256/K384/K512；固定控制只见 K384，候选在等 actual-observation 训练成本下见三档预算。

2. **语义预算分配。**
   只有第一阶段证明系统真正兼容多预算后，才允许低成本 Scout 证据决定窗口或视频预算。controller 必须与 content-independent、同预算 manifest 直接比较。

第一阶段不含 controller；第二阶段不允许修改第一阶段已经冻结的 detector、selector、位置构造、损失和时间映射。

### 7.3 最多两条可证伪主张

**主张 1：预算兼容性**

在相同完整训练集、Stage-1 起点、更新数、优化和 fixed mixed workload 下，多预算暴露比 K384-only 暴露提高 Avg-mAP 与 mAP@0.7，同时不实质损害 K384 锚点。

**主张 2：语义分配价值**

在预算兼容系统上，预注册的 Scout-based budget assignment 相对相同 K multiset 的 content-independent assignment，能够以不更高的实际重型成本改善或保持高 tIoU TAD。

### 7.4 明确非主张

* 不声称第一阶段是纯 detector-only 因果效应；
* 不声称 fixed mixed manifest 是 learned controller；
* 不声称 observation 减半等于端到端计算减半；
* 不声称 DUCA 已优于 dense AdaTAD；
* 不声称当前方法跨检测器或跨数据集；
* 不声称物理时间、tubelet continuity 或 Gaussian reconstruction 已被解决。

### 7.5 最强竞争解释

1. 多预算训练只是一种 input-length regularization，mixed workload 改善不代表语义 controller 可用。
2. 改善可能主要来自 Scout/selector 的共同适应，而非下游 detector。
3. K256/K512 在共同 384-point detector grid 上的插值本身决定结果。
4. 改善可能只出现在低 tIoU，未保护边界。

第一轮只 falsify “预算暴露是否建立兼容性”。成功后才用 content-independent 同预算控制和最小冻结-selector attribution 区分上述解释。

### 7.6 为什么优先于其他未闭环问题

* 它直接位于 0/704 后最强未检验解释上；
* 它是任何动态预算 controller 的必要前提；
* 它可通过两条完整训练臂得到决定性答案；
* pairing、kernel 和 TrueTime 即使成功，也不自动产生动态预算 headroom；
* 当前不需要发明新 selector 或新时间模块。

### 7.7 成功与失败门

**第一阶段成功：**

* K384：`ΔAvg-mAP >= −0.2 pp`
* K384：`ΔmAP@0.7 >= −0.2 pp`
* same fixed mixed manifest：`ΔAvg-mAP >= +0.8 pp`
* same fixed mixed manifest：`ΔmAP@0.7 >= +1.0 pp`
* mixed actual observations 不高于全 K384
* 两个 mixed 指标的 10,000 次 paired whole-video bootstrap 95% CI 下界均大于 0
* 无协议偏差

成功只解锁第二阶段 controller 科学裁决，不自动授权实现。

**第一阶段失败：**

任一实际效应、安全或成本门失败，即停止当前 H65+nested-three-tier 适应路线；不换 seed 抢救、不改概率、不降门、不加 embedding、蒸馏、TrueTime 或新 selector。

这不逻辑性否定所有可能的 DUCA 研究，但关闭当前资源和代码边界内最可信的 variable-budget 论文路径。

---

## 8. `CODE_ORGANIZATION_DECISION`

### 8.1 唯一 clean mainline

当前数据身份任务：

```text
base:   04c35a3b76897e6c1569eeede41ed3aecaf7f854
branch: feature/duca-full-data-identity-audit-v1-20260831
```

数据 `PASS` 并经本报告规定的 Pro 准入后，模型分支应重新从 `04c35a3...` 建立，而不是从 identity-audit commit、`33e4ed...`、库存 commit 或脏项目根继承。

建议模型分支：

```text
feature/duca-budget-compatible-training-v1-20260904
```

### 8.2 必须保留和复用

从 H65 基座保留：

* H65 Scout 与七维 deploy-visible 状态；
* BCSR K384 priority identity；
* Stage-1 EMA 初始化；
* Stage-2 loss schedules 和 trainable-set identity；
* selected-position original-time metadata；
* pre-NMS physical inverse mapping；
* official-derived AdaTAD/ActionFormer head、loss、Soft-NMS、evaluator。

从 Marginal/whole-video 历史实现仅移植最小符号：

* `NestedH65PrefixSelection`
* `nested_h65_budget_prefixes`
* `DucaOnlineFrameSelector.forward_marginal_prefixes`
* `interpolate_acquisition_time_to_detector_grid`
* `marginal_budget_accounting`
* `validate_real_heavy_observation_tensor`
* K384 parity 检查
* sealed producer 原 proposal 顺序保持
* actual-observation accounting

### 8.3 不得整分支 cherry-pick

不得整体合并 `33e4ed...`，因为它还携带：

* 160/40 split；
* labeled holdout oracle；
* Marginal/cap-release/96/704 stop logic；
* 已关闭 controller action space；
* 过宽的 project-level stop 状态名。

只移植经逐符号审查的执行原语。

### 8.4 第一轮允许修改的生产表面

数据准入后的 Builder 只允许最小修改：

```text
opentad/models/duca/dynamic_budget.py
opentad/models/selectors/duca_online_frame_selector.py
opentad/models/detectors/actionformer.py
configs/adatad/thumos/<fixed-control-config>.py
configs/adatad/thumos/<multi-budget-config>.py
tests/<focused multi-budget tests>
scripts/<one minimal PRE_RUN/train launcher family>
```

只有在 `acquisition.py` 必须暴露已存在 nested-prefix API 时，才允许极小改动；不得改变 BCSR 的数学或新增 decoder。

### 8.5 必须移出活动候选语义

以下代码可以留在 Git 历史，但不得由活动配置实例化：

* `PrefixMarginalUtilityBudgetController`
* `local_cell_deformation`
* protected homotopy bridge
* boundary-burst family
* `TemporalCoverageSelector`
* native tubelet coreset scoring
* `TrueTimeFeatureResidual`
* SingleClock/PJST
* sparse reconstruction modes
* UVT/Fovea paths
* utility predictor 和 whole-video oracle

不做大型物理删除或重构；通过配置、测试和 import whitelist 保证它们不进入候选。

### 8.6 关键实现修订

多预算臂采用 **budget-homogeneous logical batch**：

* 每个 logical optimizer update 只运行一个 K 档；
* batch 内样本使用相同 K；
* K 序列由预先冻结的 budget multiset 和独立哈希流确定；
* 避免同一 batch 内 ragged grouping、不同 BatchNorm 行为或候选专属 optimizer step；
* 如果 K512 需要 microbatch 1 + accumulation 2，固定控制也必须使用同一规则。

---

## 9. `CURRENT_TASK_ORDER`

### 9.1 当前唯一任务

**无标签、只读完成 THUMOS14 完整 train/held-out 身份核验，并给出 211/212 的 source-backed 解释。**

当前不得执行任何模型任务。

### 9.2 Builder

基于 `04c35a3...`，只允许新增或修改：

```text
tools/bata/audit_duca_thumos14_split_identity.py
tests/test_audit_duca_thumos14_split_identity.py
```

允许读取：

* config 继承链；
* subset 字面值；
* video IDs；
* media filenames；
* loader 输出 IDs；
* exclusion 源码；
  -文件存在性和基本解码；
* evaluator 输入 ID 语义；
* 历史 211 ID manifest；
* ActionFormer 212 的原始 source/manifest。

禁止读取：

* held-out 动作类别；
* held-out temporal segments；
* proposals；
* predictions 内容；
* checkpoint；
* mAP；
* per-video utility。

Builder 必须输出：

* Stage-1/Stage-2 精确 config 路径和继承链；
* annotation / loader / physical 三套 training IDs；
* annotation / loader / physical / evaluator held-out IDs；
* 历史 211 IDs；
* ActionFormer 212 source IDs 或 `SOURCE_NOT_FOUND`；
* 所有 counts、set differences、train-held-out intersection；
* 所有 exclusion 的 source line；
* annotation、class map、evaluator SHA；
  -完整排序 manifests；
* `split_identity_report.json`。

### 9.3 独立 Critic

只审查：

* 是否以 `04c35a3...` 为 parent；
* 是否只访问身份层字段；
* 是否遍历 held-out annotation 内容；
* config 继承是否完整解析；
* loader 静默过滤是否捕获；
* 差集是否来自 literal IDs；
* 缺文件或解码失败是否被错误当成排除；
* 输出是否确定、排序、无交集；
* Builder 是否擅自选择 211 或 212。

输出只有 `PASS` 或有界 blocker。

### 9.4 独立 CPU Evaluator

Critic `PASS` 后，在 N16R4 CPU 上运行一次：

* 不申请 GPU；
* 不加载 checkpoint；
* 不生成预测；
* 不计算 mAP；
* 返回 exact commit、clean tree、命令、输出根、literal manifests、counts、set differences、decode failures、exclusion source、report SHA 和 `PASS/BLOCK`。

### 9.5 数据身份判定

* annotation、loader、physical 和 evaluator 都是同一 211，并能解释 ActionFormer 212 来源：`PASS 211`。
* annotation 为 212，但 loader 少一个：先修复缺失，不能静默排除。
* annotation 为 212，且有官方精确 exclusion：可按 source-backed 规则 `PASS 211`。
* 来源找不到、集合不一致或差集无法解释：`BLOCK`。

无论 `PASS` 或 `BLOCK`，都先返回 Pro；不自动进入模型实现。

### 9.6 数据准入后的唯一模型任务

完成一条 fixed K384 控制和一条 multi-budget exposure 候选：

* 共同 Stage-1 epoch-29 EMA；
* 共同 6,000 successful updates；
* 共同 trainable set、optimizer、LR、EMA、augmentation、loss normalization、seed 和 checkpoint；
* 唯一差异为每次 update 使用的 K 档；
* 真实 variable-length heavy execution；
* K256/K384/K512 nested identity；
* actual cost matched；
* terminal update-6000 EMA。

---

## 10. `FULL_EXPERIMENT_AND_STATISTICS_PLAN`

### 10.1 数据

* 完整 `T_full`：annotation 中全部 200 个 `training` IDs；
* 完整 `E_full`：身份审计准入后的全部 held-out IDs；
* 不使用旧 40-video holdout 作正式设计或门禁；
* held-out 只在所有模型、预测、manifest、成本规则和门槛密封后开放。

### 10.2 共同起点

```text
epoch:     29
state key: state_dict_ema
SHA-256:   bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3
```

加载后两臂都重新初始化：

* optimizer
* scheduler
* AMP scaler
* EMA accumulator
* successful-update counter
* sampler
* augmentation RNG
* dropout/drop-path RNG
* Stage-2 schedule step

### 10.3 两条主臂

**Fixed control**

每个 logical update 请求 K384。

**Multi-budget exposure**

每个 logical update请求 K256、K384 或 K512，使用同一 H65 priority sequence 的 nested positions。

冻结：

```text
p384 = 0.50
p256 + p512 = 0.50
```

在完整 6,000-update occurrence plan 上计算 `μ256/μ384/μ512` 后：

```text
p256 = 0.5 × (μ512 − μ384) / (μ512 − μ256)
p512 = 0.5 × (μ384 − μ256) / (μ512 − μ256)
```

我接受该设计，原因是它保留 50% 的锚点暴露，并在无标签条件下把候选的期望 actual observations 匹配 K384。它不是历史事实，而是本报告冻结的实验选择。

阻断条件：

* `μ256 >= μ384` 或 `μ384 >= μ512`
* 概率不在 `[0,1]`
* 总 actual-observation 偏差超过 `0.5%`
* budget schedule 读取任何标签或指标
* 训练后修改概率

### 10.4 优化合同

* 6,000 successful optimizer updates；
* 500-update warmup；
* Stage-2 schedule 前后半程边界 3,000；
* AdamW；
* 主 LR `1e-4`；
* weight decay `0.05`；
* adapter `2e-4`；
* coarse trunk `1e-5`；
* action head `2e-5`；
* transition scorer `5e-5`；
* global norm clip `1`；
* AMP 和 EMA 开启；
* seed `3407`；
* terminal update-6000 `state_dict_ema` 唯一选模。

每 500 updates 保存恢复点，强制保存 3,000 和 6,000，至少保留最近三个有效点。中间 held-out mAP、best checkpoint 和 early stopping 全部禁止。

### 10.5 无标签 PRE_RUN

必须证明：

* 两臂初始化逐 tensor 相同；
* optimizer parameter groups 和 trainable names 相同；
* K384 candidate path 与 control bitwise parity；
* K256/K384/K512 selected positions 严格 nested；
* 每档 selected positions 严格递增；
* packetized tensor 长度正确；
* 只在最后 packet 零 pad；
* backbone 前不是 max-K padding；
* actual observation 单调；
* detector-grid reconstruction 输出长度相同；
* loss normalization 不随 K 改变；
* AMP、EMA、gradient accumulation 正常；
* proposal producer 原顺序保持；
* 完整 budget occurrence schedule 已冻结。

这些只证明实现正确，不是性能证据。

### 10.6 密封预测

两个 terminal EMA 均生成：

* K256
* K384
* K512
* 同一个无标签 fixed mixed-budget manifest

所有固定预算预测先独立密封。Mixed 输出只能从已密封预测按 manifest 选择；不允许重新 forward、重新排序 proposal 或读取标签。

每份保存：

* checkpoint/config SHA
* prediction SHA
* video IDs
* sample/window IDs
* producer order
* pre/post-NMS proposal counts
* actual observations
* packetized heavy input
* wall-clock 和硬件身份

### 10.7 决定性指标

主判据：

* mAP@0.3、0.4、0.5、0.6、0.7
* Avg-mAP
* mixed actual observations
* K384 safety
* 10,000 次 paired whole-video bootstrap

解释性指标：

* proposal Recall@100 / Recall@200
* recall@tIoU 0.5 / 0.7
* start/end absolute physical-time error
* duration-normalized boundary error
* 短、中、长动作 Avg-mAP 与 mAP@0.7
* pre/post-NMS proposal count
* NMS 前后 false positives
* score calibration
* 每档特征幅度和 detector-logit 分布

成本指标：

* actual observations
* VideoMAE actual input
* backbone FLOPs/MACs
* full-pipeline latency p50/p95
* throughput
* peak memory
* energy

短/中/长阈值只用 training GT duration 三分位数冻结。

### 10.8 不确定性

**单种子机制门**

seed 3407 先回答是否值得执行完整 replication。该结果不能支持训练稳定性。

**10,000 次 paired whole-video bootstrap**

* 以视频为 cluster；
* 每次有放回采样视频；
* 同一 replicate 对全部模型和预算使用同一视频样本；
* 从 sealed predictions 重算官方 mAP；
* 95% percentile interval 使用 2.5%/97.5% quantiles；
* 它只表示 held-out 视频采样不确定性，不表示 seed 不确定性。

**三种子正式主结果**

仅在 seed 3407 全部通过后，按完全冻结的方法训练 seeds `3408`、`3409`。不得根据 seed 3407 修改任何方法、概率、阈值、训练长度或 manifest。

正式主表报告：

* 三种子均值和标准差；
* 每种子 paired video interval；
* seed-level delta 的均值；
* 不把 video bootstrap 冒充训练随机性区间。

### 10.9 必要 baseline

不建立大矩阵。

只读或重放现有：

* Shared Dense AdaTAD
* historical H65
* historical exact-uniform
* random
* actionness-only / simple-transition

新训练只有：

* matched fixed K384 control
* matched multi-budget exposure candidate

若历史 baseline 的 `E_full` manifest 与新 admitted held-out 不同，只作上下文，不计算直接点差。除两条主臂外，不因身份差异重复训练整套历史 baseline。

### 10.10 第二阶段解锁

只有第一阶段通过全部门，才允许新的 Pro 裁决冻结 semantic controller。第二阶段最小比较应是：

* 同一个 budget-compatible checkpoint；
* 同一个 K multiset；
* content-independent assignment；
* Scout-semantic assignment；
* 相同实际成本。

第一阶段失败时，controller 永久不启动。

---

## 11. `PUBLICATION_PLAN`

### 11.1 工作论文题目

**Budget-Compatible Task-Aware Temporal Acquisition for Offline Action Detection**

中文工作名：

**面向离线时序动作检测的预算兼容任务感知时序采集**

### 11.2 论文主线

审稿人应感到意外的点不是“多尺度训练通常有帮助”，而是：

> 一个只在单一稀疏预算下训练、看似能产生合理语义选点的 TAD 系统，可能完全不支持跨预算计算分配；通过等成本的预算兼容训练，才可能把低成本语义证据转化为真实可变重型计算，并保护高 tIoU 边界。

### 11.3 必需表与图

**表 1：数据、模型与成本身份**

* full train / held-out IDs
* heavy observations
* actual packetized input
* full-pipeline cost
* checkpoint/evaluator identity

**表 2：主性能—成本结果**

* Dense
* exact-uniform
* H65 reference
* matched fixed control
* budget-compatible model
* semantic controller final model

**表 3：跨预算兼容性**

* 两个模型 × K256/K384/K512/mixed
* Avg-mAP、mAP@0.7、boundary error、actual cost

**表 4：必要机制归因**

仅在主效应成功后：

* fixed vs multi-budget exposure
* content-independent vs semantic budget assignment
* 最多一个 frozen-selector attribution

**图 1：性能—actual-observation 曲线**

**图 2：高 tIoU 和动作长度分层**

**图 3：边界误差与预算档位**

**图 4：典型视频中 Scout evidence、K 分配和 proposals**

### 11.4 创新风险

最大风险是多预算训练本身不够新。论文必须依靠以下组合形成贡献：

* pre-backbone task-aware acquisition；
* nested actual observations；
* budget-compatible TAD training；
* semantic variable-budget allocation；
* high-tIoU boundary evidence；
* measured real variable compute。

如果第一阶段通过而 controller 阶段失败，仅有预算兼容训练通常不足以形成完整 DUCA 方法论文；该结果更适合作为机制研究或负结果报告。

### 11.5 局限性

即使成功也必须承认：

* 首个系统基于单一 H65 selector family；
* 首个数据集为 THUMOS14；
* initial mechanism gate 只有一个 seed；
* fixed mixed manifest 不是 controller；
* BCSR 和 nested positions 未证明最优；
* observation count 与端到端成本不同；
* effect locus 可能跨 detector、adapter、Scout 和 selector；
* 物理时间、pairing continuity 与 reconstruction kernel 未被同时解决。

### 11.6 可投稿判定

只有同时满足以下条件，才标为“可投稿候选”：

1. 数据身份完整准入；
2. 第一阶段三种子预算兼容效应成立；
3. K384 anchor 安全；
4. controller 相对 content-independent 同预算控制成立；
5. actual heavy compute 与 full-pipeline cost 均被测量；
6. 至少一个第二 detector 或第二数据集复现方向；
7. 所有 predictions、per-video identity、原始指标和成本密封；
8. 没有 held-out 反馈修改方法。

任一核心机制失败，按对应停止边界删掉主张；不通过堆叠 TrueTime、蒸馏、Mamba、Block Drop 或 TensorRT 抢救。

---

## 12. `ABSOLUTE_MILESTONES`

以下为科学时间盒，不是允许后台自动执行的授权。

### 2026-09-02 18:00 America/Chicago

对应 2026-09-03 07:00 北京时间

完成数据身份 Builder、focused tests 和 literal manifests。

未完成处置：返回具体 blocker，不开始模型代码。

### 2026-09-04 18:00 America/Chicago

对应 2026-09-05 07:00 北京时间

完成独立 Critic 和 N16R4 CPU Evaluator。

输出只能是 `PASS` 或 `BLOCK`。

### 2026-09-05

Pro 根据 literal evidence 签发数据准入或维持阻断。科学路线不由 Codex重新选择。

### 2026-09-10

若数据 `PASS`，完成预算兼容训练的最小 Builder commit、独立 Critic 和无标签 PRE_RUN。

若 K384 parity、真实变长或成本匹配失败：最小修复；不启动训练。

### 2026-09-18

完成 seed 3407 的两条 6,000-update 训练臂和全部 sealed predictions。

训练失败若来自路径、调度或保存：最小确定性恢复。
训练公平性或机制隔离失败：协议无效，重新返回 Pro。

### 2026-09-20

一次性完成 seed 3407 正式评测和 10,000 次 paired whole-video bootstrap。

* 全门通过：冻结方法，启动 seeds 3408/3409 replication；
* 点值通过但 CI 含零：当前路线终结且不前进；
* 任一安全/效应/成本门失败：停止当前三档适应路线；
* 协议失败：证据无效，不作模型负结论。

### 2026-10-02

若 seed 3407 通过，完成三种子 replication 和同硬件成本测量。

### 2026-10-05

Pro 决定是否解锁唯一 semantic controller 实验。不得自动启动。

### 2026-10-20

若解锁，完成 controller 与 content-independent 同预算控制的三种子主比较。

### 2026-11-06

若 controller 主效应成立，完成预注册的第二 detector 或第二数据集验证，以及完整成本表。

### 2026-11-15

最终判定：

* **可投稿候选**：主效应、controller、成本和外部复现均成立；
* **停止正向 DUCA 方法论文**：任一核心阶段失败且没有新的独立训练侧机制证据；
* 失败结果进入完整技术报告，不通过新模块堆叠延长路线。

---

## 13. `NEXT_RETURN_CONTRACT`

Codex 下一次只返回数据身份准入证据，不返回模型方案变化。

必须包含：

### 代码身份

* branch
* exact commit
* parent `04c35a3...`
* clean-tree
* 修改文件
* focused tests
* Critic 完整结论

### 配置身份

* Stage-1 config 精确路径
* Stage-2 config 精确路径
* 完整 inheritance chain
* resolved train subset
* resolved evaluation subset
* dataset loader 文件与 source lines
* evaluator 文件、source lines 和 SHA

### 完整训练集

* 200 个 literal IDs
* annotation / loader / physical 三集合
* 全部 set differences
* decode failures
* exclusions，期望为空
* train manifest SHA

### 完整 held-out

* annotation held-out IDs
* loader IDs
* physical IDs
* evaluator IDs
* historical 211 IDs
* ActionFormer 212 source IDs，或 `SOURCE_NOT_FOUND`
* 全部 set differences
* 211/212 的 source-backed 解释
* 任何官方 exclusion 的完整 ID、规则与 source line
* held-out manifest SHA

### 隔离与完整性

* train-held-out intersection
* annotation/class map/evaluator SHA
* no-label-access test
* Evaluator 命令和输出根
* 唯一结论 `PASS` 或 `BLOCK`

下一次返回中明确禁止：

* 多预算模型代码；
* Stage-1 checkpoint 加载；
* GPU、PRE_RUN 或训练；
* held-out temporal annotation；
* 新预测；
* mAP；
* controller；
* TrueTime、蒸馏、新 selector 或重构核。

本报告已经独立完成路线选择。Codex 不得再在多预算训练、TrueTime、tubelet pairing、kernel comparison 或课程之间代选；当前只解决数据事实。
