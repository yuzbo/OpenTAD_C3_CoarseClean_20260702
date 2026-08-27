# DUCA a00498e 精确提交逐行审查与下一版模型优化 Pro Prompt

你现在不是一般项目顾问，而是同时承担以下角色：

- CCF-A/CVPR 级离线时序动作检测审稿人；
- TAD、结构化离散采样、可微选择与多任务优化研究者；
- 熟悉 PyTorch、VideoMAE、AdaTAD、ActionFormer、TemporalMaxer 和 ASFormer 的模型审计者；
- 一个会主动证伪项目假设、但不会为了显得严厉而忽略代码事实的独立首席研究员。

你的任务不是复述项目说明，也不是提出一批泛泛的新想法。请直接读取固定 GitHub 提交，沿真实调用链逐行审查当前 DUCA 模型，判断它是否忠实实现了研究初心、为什么学习选帧可能仍落后于均匀采样，并给出下一版可以直接落地的模型结构、监督目标和训练方案。

## 0. 不可违反的审查规则

1. 本任务是 **离线 TAD**，不是 Online TAD、流式 TAD 或因果在线检测。全文不得把它称为 Online TAD。
2. 最终产物应是 heavy video backbone 之前的任务感知时序采样插件，而不是一个全新 detector。
3. 必须先通过 GitHub 连接器读取精确提交。不得只看最后一个 diff，也不得仅根据本 Prompt 的摘要推断代码。
4. 每项结论必须标记为 `[CODE_FACT]`、`[EXPERIMENT_FACT]`、`[INFERENCE]`、`[PROPOSAL]` 或 `[UNKNOWN]`。
5. 所有 Critical/High 问题必须给出精确文件、行号、调用路径、触发条件、对 mAP/高 tIoU/成本的影响、最小修复及验证方法。
6. 不能把测试通过、一步梯度门禁、R0 内部回放、bootstrap、选帧质量代理或成本统计冒充最终 mAP。
7. 不要把精力浪费在复杂启动器、日志框架、哈希包装或工程洁癖上。工程问题只在会导致模型错误、协议错误、不可复现或无效结果时升级。第一优先级永远是模型机理、训练正确性和 official mAP。
8. 不得默认现行路线正确。也不得未经代码证据就建议推倒重来。先判断最小修正是否足够，再讨论高风险结构重构。
9. 不得提出“多调几个权重”“换更大 backbone”“增加更多阶段”这类无可证伪内容。每个建议必须说明数学定义、梯度归属、推理代价、失败模式和最小实验。
10. 请一次性完成审查，不要先向用户索取已经能从 GitHub 读到的信息。

## 1. 固定审查对象

- 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-boundary-burst-20260722>
- 精确提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/a00498e15d69294f78d0abeadfb47bc456db0b0e>
- 完整 SHA：`a00498e15d69294f78d0abeadfb47bc456db0b0e`
- 官方 OpenTAD 对照仓库：<https://github.com/sming256/OpenTAD>
- 代码声明的官方基准提交：`1aa8ca4ac5e846b1e8ff69298dd6607121a01589`

本地已核对 `HEAD == origin/codex/duca-boundary-burst-20260722 == a00498e...`，GitHub push 返回 `Everything up-to-date`。你仍须自己读取 GitHub 对象并输出：

```text
VISIBILITY_CERTIFICATE
repository=
branch=
exact_commit=
files_actually_read=
unreadable_dependencies=
```

若精确提交或关键文件不可见，请将相应结论标为 `UNKNOWN`，不要伪造逐行审查。

## 2. 研究初心与最终目标

我们要解决的问题很朴素：离线 TAD 为了获得精确边界，通常让昂贵 VideoMAE 对长视频的密集时序帧做计算，其中存在大量冗余。我们希望先用低成本模型观察完整低分辨率时序，决定哪些原始时间位置值得送入 heavy backbone，在降低 heavy-backbone 处理帧数和真实总成本的同时，尽量保护甚至提高高 tIoU mAP。

完整初心应当是：

```text
低成本完整时序 RGB
  -> 动作/背景二分类粗证据与 ASFormer 时序隐特征
  -> p_action、delta、熵变化、hidden change 等状态转变证据
  -> 间接语义边界定位与边界微簇预算分配
  -> 全局 exact-K / max-hole 结构化硬选择
  -> 按 original-time gather 的真实 RGB
  -> 原始 VideoMAE + official-derived AdaTAD/ActionFormerHead 或 TemporalMaxer
  -> 完整 THUMOS validation 上的 official mAP
```

关键设计约束：

- 粗分类器的主任务是动作/背景二分类，不要求轻量模型直接做精确 TAD 边界回归。
- selector 应利用状态转变和不确定性间接找到语义边界，而不是做 actionness top-k。
- 目标分布不是 one-frame-per-cell，也不是接近均匀的小扰动。应允许预算跨区域转移，并在动作起点和终点两侧形成类似 Oracle 的多帧局部微簇，同时保留必要的全局上下文。
- 边界聚集必须兼顾：中心准确、边界前后双侧观测、每个边界的有限配额、多个边界之间的公平分配、重叠边界去重，以及剩余预算的全局覆盖。
- 固定预算是主线；dynamic MUST/X3D/SlowFast/local-cell/actionness top-k 均不是当前主方法。
- 推理时不得使用 GT、teacher、detector replay、ledger、预测缓存或预先导出的 actionness JSONL。
- 当前实现先密集解码和传输低分辨率输入，再减少 heavy VideoMAE 的帧数。因此最多先主张 `post-decode heavy-backbone processed-frame reduction`，不能在没有证据时声称 K-only 视频解码或完整 I/O 降本。

请判断当前代码是否真正满足这些原则，尤其检查：代码命名为 `DucaOnlineFrameSelector` 是否只是历史名称，运行语义是否确实是离线全窗口选择。

## 3. 当前实现摘要，仅用于导航，必须由代码复核

以下内容是待核验线索，不是允许你直接采用的结论：

1. 现行 selector 读取 in-graph official ASFormer 的 actionness 与 encoder hidden；排名证据声明包括 `delta actionness logit`、`abs delta`、熵变化、hidden delta、hidden cosine change，而不是绝对 hidden 或 RGB 均值。
2. R2Q3 表示半径 2、目标配额 3 的双侧边界微簇；R4Q5 是更宽微簇诊断。配置开启 `boundary_burst_require_bilateral_offsets=True` 和 `boundary_burst_require_global_mandatory_groups=True`。
3. 硬解码器声明共享 exact-K/max-hole 动态规划，并可消费 mandatory boundary groups。请验证它到底硬保证了什么，哪些仍只是 soft loss；不要相信配置注释。
4. P0 是 20 epoch、train-only 的前端训练：binary actionness、transition distribution、boundary-burst coverage；heavy detector 被跳过。
5. G0 不接 detector feedback；G1 使用 `protected_structured_transport`，声明 detector 梯度只更新 transition scorer；G2 在同一 detector forward 中混入 50% exact-uniform companion，只对 learned rows 更新 selector。
6. official-60 主训练先从 uniform 向 learned policy 做连续课程，再延迟引入 detector surrogate；terminal epoch-59 EMA 是唯一正式 checkpoint。
7. detector 接收 selected-axis 特征，GT 被映射到 selected axis，预测再映回 true/original time。请重点检查非均匀采样后，VideoMAE、投影层与检测头是否错误地把相邻 selected rank 当成等物理时间。
8. R5 当前预算轴为 `K={384,320,256,192,128}`，对应 dense-candidate max-hole `G={2,2,3,4,6}`；后端为 ActionFormer 与 TemporalMaxer，策略为 exact-uniform 与 learned R2Q3，种子为 `3407/5801/8123`，共 60 个 official mAP cells。
9. 当前代码必须区分三个梯度域：actionness loss 到 coarse stem/ASFormer/action head；transition/burst loss 到指定 ASFormer hidden 层与 scorer；detector feedback 只到 scorer/burst，不能无意污染粗二分类语义。

请输出一张由代码重建的真实计算图，列出每个 tensor 的 shape、坐标单位、是否 detached、是否只在训练存在、由哪些 loss 更新。

## 4. 必读代码表面

必须至少逐行阅读下列文件，并继续追踪它们实际 import/callee。不要把仓库中的历史 DUCA、MUST、X3D、local-cell、SparseHead、Spatial-Zoom 或 ChronoTransport 混入现行模型。

### 4.1 模型与选择核心

- `opentad/models/selectors/duca_online_frame_selector.py`
- `opentad/models/duca/acquisition.py`
- `opentad/models/duca/structured_selection.py`
- `opentad/models/duca/transition_only.py`
- `opentad/models/duca/hard_soft_alignment.py`
- `tools/bata/train_lowres_action_probe.py` 中 official ASFormer loader/wrapper/hidden 路径

### 4.2 detector 与时间轴

- `opentad/models/detectors/actionformer.py`
- `opentad/models/dense_heads/actionformer_head.py`
- `opentad/models/projections/actionformer_proj.py`
- `opentad/models/detectors/temporalmaxer.py`
- `opentad/models/dense_heads/temporalmaxer_head.py`
- `opentad/models/projections/temporalmaxer_proj.py`
- `tools/bata/duca_selected_axis_training.py`
- `tools/bata/run_duca_temporalmaxer_one_step.py`

### 4.3 当前配置与训练课程

- `configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py`
- `configs/adatad/thumos/duca_protected_e2e_fixed384_official60.py`
- `configs/adatad/thumos/duca_frontend_pretrain_fixed384_base.py`
- `configs/adatad/thumos/duca_boundary_burst_frontend_pretrain_fixed384.py`
- `configs/adatad/thumos/duca_boundary_burst_g0_no_feedback_fixed384_official60.py`
- `configs/adatad/thumos/duca_boundary_burst_g1_protected_fixed384_official60.py`
- `configs/adatad/thumos/duca_boundary_burst_g2_uni_companion_fixed384_official60.py`
- 对应 R4Q5 配置

必须解析配置继承后的实际生效值。合同字符串、注释、基础配置默认值和最终 merged config 若冲突，以真实 merged runtime config 为准并报告冲突。

### 4.4 R0-R5、预算曲线与成本

- `tools/bata/build_duca_r0_boundary_burst_oracles.py`
- `tools/bata/finalize_duca_r0_boundary_burst.py`
- `tools/bata/duca_p0_training.py`
- `tools/bata/select_duca_boundary_burst_candidates.py`
- `tools/bata/duca_boundary_burst_hard_swap_alignment.py`
- `tools/bata/duca_r5_paper_matrix.py`
- `tools/bata/aggregate_duca_r5_paper_matrix.py`
- `tools/bata/aggregate_duca_budget_curve.py`
- `tools/bata/export_duca_selection_quality.py`
- `tools/bata/analyze_duca_selection_quality.py`
- `tools/bata/analyze_duca_budget_selection_curve.py`
- `tools/bata/profile_duca_full_stack_cost.py`
- `tools/bata/plot_duca_r5_performance_cost.py`

### 4.5 focused tests

- `tests/test_duca_boundary_burst_selection.py`
- `tests/test_duca_boundary_burst_configs.py`
- `tests/test_duca_boundary_burst_full_model_gate.py`
- `tests/test_duca_boundary_burst_hard_swap_alignment.py`
- `tests/test_duca_detector_gradient_bridge.py`
- `tests/test_duca_structured_selection.py`
- `tests/test_duca_official_asformer_hidden.py`
- `tests/test_duca_r5_paper_matrix.py`
- `tests/test_duca_budget_curve.py`
- `tests/test_duca_selection_quality_analysis.py`
- `tests/test_duca_full_stack_cost.py`

测试只能证明其断言覆盖的合同。请同时指出缺失的模型级行为测试。

## 5. 当前实验事实与证据边界

### 5.1 可比较或可参考的历史证据

- 历史 V5 同协议终点：exact-uniform `64.4580`，direct `63.7102`，homotopy `63.0601`，companion `63.6931`。旧 learned sampling 没有超过 uniform。
- CellCF 终点：uniform `63.8594`，transition-beta0 `64.2755`，CellCF `64.0610`。CellCF 未超过 transition-beta0。
- 历史 dense AdaTAD 约 `68.29`；历史 uniform `64.352/65.696` 只能作为协议未匹配背景，不能直接并入当前 matched table。

### 5.2 不可进入论文主表的 R0 诊断

- R0 40-video 内部回放曾得到 U `93.587070`、R2Q3 `94.190497`、R4Q5 `93.999241`、unrestricted Oracle `93.970057`。
- detector checkpoint 训练时见过这批 training videos，因此这些 93--94 只是 detector-seen training-internal mechanism diagnostic；即使调用 OpenTAD evaluator，也不能与完整 validation 上的 64--65 比较，更不能作为论文 absolute mAP 或 Oracle 上界。
- 自定义 paired bootstrap 只能做内部不确定性诊断，不能替代官方完整 validation 的 point mAP 和多种子 mean/std。

### 5.3 当前正在运行的唯一正式证据链

旧 K384/K256 R0-R5 bundles，模型提交 `cd68d89dcc0854baa3c0107607086e801509b552`：

- `1180336`：R0/R1；RUNNING。
- `1180337`：R2/R3 core；RUNNING。
- `1180338`：R2/R3 adapted；RUNNING。
- `1180339`：R4；RUNNING。
- `1180340`：R5 K384/K256；RUNNING。
- `1180341`：旧 R5 aggregate；等待 `1180340`。

当前增量提交 `a00498e...` 只补 K320/K192/K128 和动态预算合同，不重跑旧 K384/K256：

- `1180356`：36 个完整 TAD cells；已被 Slurm 接收，当前 PENDING(`AssocGrpGRES`)。
- `1180357`：增量 aggregate，`afterok:1180356`。
- `1180358`：合并五点 official mAP 与选帧分布，依赖旧、新两个 aggregate。

截至本 Prompt 生成时，没有新的 terminal official mAP、三种子均值或完整成本结果。当前状态只能是 `experiment_running`，不能称 empirically supported 或 paper ready。不要建议取消或重复提交这些等价实验；代码审查发现真正影响模型/协议的 Critical 错误时，明确说明哪些结果会失效。

## 6. 必须逐行回答的核心问题

### A. 当前模型到底是什么

1. 从 raw clip 到 coarse probe、transition scorer、structured decoder、RGB gather、VideoMAE、projection、TAD head 和 true-time remap，重建真实前向图与 tensor shapes。
2. 判断它是否真是 pre-backbone 插件，还是在实现中已经与 AdaTAD 结构深度耦合、难以迁移。
3. 判断 coarse probe 是“二分类辅助的间接边界证据”，还是实际上被 boundary loss/detector surrogate 拉成了一个隐式边界预测器，从而违背初心。
4. selector 是否真正看到 official ASFormer hidden、动作概率、状态差分和不确定性；是否存在名字上启用但 merged config/forward 中未消费的输入。
5. 训练和推理是否同构；validation/test 是否完全 GT/teacher/cache/ledger-free。

### B. 边界微簇是否真的像 Oracle，而不是近似均匀

1. R2Q3/R4Q5 的 center、left/right offsets、quota、fairness、overfill、dedup、context fill 在代码中分别如何计算。
2. `boundary_burst_require_bilateral_offsets` 是 soft utility、hard mandatory set 还是两者；若边界一侧不可行如何处理。
3. `boundary_burst_require_global_mandatory_groups` 是否真的进入 DP 可行域，是否可能因预算冲突静默删除弱边界。
4. exact-K 与 max-hole 约束对可聚集性的理论上限是什么。对 T=768，分别分析 K384/G2、K320/G2、K256/G3、K192/G4、K128/G6 能否形成 3--5 帧微簇，还是会被覆盖骨架压成近均匀。
5. 当前 max-hole 是以 dense candidate、source frame 还是 seconds 计量；配置、decoder、日志和论文口径是否一致。
6. 给出至少两个反例序列，证明当前 hard decoder 是否会漏边界、单侧聚集、重复消耗配额或挤掉全局上下文。

### C. selected-axis 是否破坏真实时间语义

1. 非均匀 selected frames 被 pack 后，VideoMAE 是否默认等时间间隔；其 temporal positional encoding、tubelet、adapter 与 TAD head 是否因此产生系统性时间扭曲。
2. GT 映射到 selected axis、head 预测、后处理回 original time 的插值是否单调、可逆并与训练 loss 一致。
3. 高 tIoU 性能不佳是否可能主要来自 selected-rank 表示，而不是选帧中心不准。
4. 是否需要 true-time positional encoding、时间间隔通道、piecewise-linear coordinate adapter 或其他最小修改。不要预设答案；比较实现复杂度、mAP 潜力和可迁移性。

### D. detector 是否真实且保持官方语义

1. ActionFormer/AdaTAD 路径哪些文件与官方一致，哪些是 wrapper/extension；配置相同不等于源码相同。
2. detector head、loss、NMS 是否未改；backbone 输入帧数、插值、projection max sequence length 是否改变了官方语义。
3. TemporalMaxer 是否是真实完整第二后端，包括真实训练、terminal checkpoint、raw prediction 和 OpenTAD evaluator，而不是 one-step 或名称替换。
4. 当前插件接口若迁移到第二 detector，是否只改 adapter/config，还是需要重写 selector 内部逻辑。

### E. 梯度路由是否真正实现“下游 mAP 影响选帧”

1. 对每项 loss 画出参数级梯度归属：spatial stem、ASFormer encoder/decoder、action head、transition scorer、burst offset/center、VideoMAE、projection、TAD head。
2. 验证 G1 `protected_structured_transport` 的 hard forward 与 surrogate backward；它是否真的与离散换帧方向一致，还是只提供有偏 soft resampling 梯度。
3. 检查 `center_scores.detach()`、soft slot assignment、ST gather 和 detector input 中每处 detach；确认 detector loss 最终是否对 transition scorer 产生稳定非零梯度。
4. 检查 detector gradient 是否可能污染 actionness 二分类语义或 ASFormer shared hidden；当前 `policy_hidden_gradient_scale`、`auxiliary_hidden_gradient_scale` 和 last-layer-only 限制是否自洽。
5. G2 uniform companion 是否公平：同一次 forward、相同 detector batch statistics、learned-row gradient normalization、训练成本、推理只走 learned policy。
6. hard-swap alignment 是否使用真实 selected RGB 和真实 detector loss，是否只是 detached surrogate 相关性；它能证明什么，不能证明什么。
7. 检查 optimizer coverage、AMP、DDP unused parameters、EMA、scheduler、resume 和 successful-update 计数是否会令某个模块实际上不更新。

### F. 监督与训练课程是否合理

1. P0 中 actionness=1、transition=0.10、boundary-burst=2.0（或 merged 后真实值）是否量纲匹配；请报告 raw loss、加权 loss 和各参数组梯度范数，而不是只看配置权重。
2. P0 是否应该让 transition/burst supervision 更新 ASFormer 最后一层，还是冻结 coarse semantics；比较完全冻结、last-layer partial adaptation、独立 transition adapter、梯度投影/PCGrad 等方案。
3. official-60 的 uniform warmup、policy ramp、detector-feedback delay 是否有理论和数据依据。是否存在 detector 在 uniform 轴上学稳后，难以适应 learned nonuniform 轴的问题。
4. 两阶段训练是否优于从头联合：第一阶段 coarse/transition/burst，第二阶段 detector uniform warmup，第三阶段有限联合；或者能否压缩为更优雅的两个阶段。
5. detector loss 不是直接可解释的“帧价值”。请提出能真正服务最终 mAP、又不需要对离散每帧做昂贵全反事实的训练信号，并说明偏差。
6. 对固定预算曲线，是否应共享一个 K-conditioned policy，还是每个 K 独立训练；当前代码实际做法是否会导致预算间不可比较。

## 7. 为什么此前 learned sampling 没超过 uniform

不要用“任务太难”或“需要调参”作答。请从代码与现有结果出发，对下列候选原因排序，并为每项给出一个最快可证伪实验：

- coarse actionness 本身分得不好，transition center 有系统偏移；
- coarse 分类很好，但 scorer 没把状态变化转成正确边界微簇；
- max-hole 可行域过紧，learned policy 实际接近 uniform；
- soft burst loss 与 hard DP/must-set 的实际选择不对齐；
- detector surrogate 梯度方向与真实 hard swap/mAP 不一致；
- detector feedback 污染了粗二分类特征；
- selected-axis/VideoMAE 时间语义扭曲抵消了更好的边界覆盖；
- P0 与 official-60 的课程切换造成 catastrophic forgetting 或 detector adaptation lag；
- boundary recall/center error 代理与最终高 tIoU mAP 弱相关；
- 相同 K 下，边界聚集损失了动作内部和背景上下文；
- 数据增强、random truncation 和 boundary validity 产生监督错位；
- 训练更新数、学习率、EMA 或 optimizer group 造成部分模块欠训练。

最后必须给出一个“最可能根因树”：观察到什么现象时走哪一分支，避免同时改五个变量后无法归因。

## 8. 下一版模型方案：必须给出可实现裁决

请提出 3--5 个彼此不同的候选，但最后只能推荐一个主方案。至少覆盖以下层级：

1. **最小修正版**：保留现有 ASFormer + transition scorer + exact-K/max-hole decoder，只修真正导致性能损失的结构或梯度错误。
2. **监督/优化修正版**：不换推理结构，只改变 loss、梯度归属或课程训练。
3. **时间语义修正版**：若 selected-axis 是主要瓶颈，给出最小 true-time-aware adapter。
4. **更大胆但有界的结构方案**：仅当现有证据表明必要时，重新设计 burst/context allocation 或 detector-aware utility；不得靠大模型堆算力。

每个候选必须给出：

- 数学目标与变量；
- 具体输入、模块和输出；
- 如何形成 Oracle 式边界双侧微簇与全局上下文；
- hard exact-K/max-hole 如何满足；
- 每项 loss 的公式与梯度到哪些参数；
- 训练/推理是否同构；
- 相对当前代码新增 FLOPs、延迟、显存和训练成本；
- 预期改善的是 coarse quality、selection quality、high-tIoU mAP 还是成本；
- 最可能失败方式；
- 需要修改的具体文件、类和函数；
- 关键 PyTorch 代码或接近可直接落地的伪代码。

最终推荐方案必须回答：

- 是否继续使用 official ASFormer；
- actionness head 在后期冻结、部分适配还是持续联合训练；
- selector 是否看绝对 hidden，还是只看状态变化；
- detector feedback 使用什么可微信号；
- 是否需要 true-time adapter；
- R2Q3 是否保留；
- 五个 K 是否共享权重；
- 如何避免再次退化为均匀采样或 actionness top-k。

## 9. 给出精确训练方案

请输出一个可以直接转成 config 的训练表，按 optimizer successful updates 而不是模糊“前期/后期”描述。至少包含：

- 阶段名称与持续更新数；
- coarse stem、ASFormer、action head、transition scorer、burst head、VideoMAE、TAD head 的冻结状态；
- 各参数组学习率和 weight decay；
- actionness、transition distribution、boundary burst、context、detector、surrogate/utility 等 loss 的起止权重和 ramp；
- exact-uniform、learned policy 和 uniform companion 的使用比例；
- hard forward / soft backward 的温度与退火；
- EMA、gradient clipping、AMP 和 DDP 注意事项；
- 何时只看质量代理，何时必须看完整 official mAP；
- 明确的停止/回滚条件。

训练设计必须尽量优雅。若三阶段确有必要，说明为什么两阶段无法解决；否则优先给两阶段可发表方案。

## 10. 最小而完整的实验闭环

当前 60-cell 五点预算曲线已经排队，不得为了新想法盲目重跑。请给出结果出来后的有界决策树：

1. 先分析 matched U vs learned 的 terminal epoch-59 EMA，报告 tIoU 0.3--0.7、三种子 mean/std、K 曲线和两个 backend。
2. 同时看 coarse actionness、transition-center error、boundary recall、burst bilateral coverage、quota utilization、max-hole、selected-position entropy 和 context coverage，定位是 evidence 失败还是 allocation 失败。
3. 只允许选择一个最可能影响 mAP 的结构修正，先做 K384/ActionFormer/seed0 的 U + 当前 learned + 修正版三臂。
4. 修正版只有在 official mAP 超过 matched uniform 且高 tIoU 不退化后，才扩展三种子、K256/128 和 TemporalMaxer。
5. 论文主表必须包含 dense、uniform、learned、预算、后端、三种子和完整成本；R0/bootstrap/one-step/gate 只进机制或附录。
6. 给出能够证伪论文核心主张的 KILL 条件。例如多个 K、两个 backend 都不能超过 matched uniform，或前端成本吞没 heavy-backbone 节省。

请区分：

- 立即必须修复的 P0 代码/模型错误；
- 当前结果出来后才决定的 P1 模型修正；
- 主结果通过后才需要的 P2 论文消融；
- 不应继续投入的历史路线。

## 11. 成本与论文声明审查

1. 完整成本必须包括 dense low-resolution decode/resize/H2D、coarse spatial stem、ASFormer、transition scorer、DP solver、selected RGB materialization、VideoMAE、projection、TAD head、NMS、峰值显存、吞吐与训练开销。
2. 检查 dense/candidate pairing 是否同硬件、同会话、同软件、同输入、同 backend 和同 profiler 身份；TemporalMaxer 没有同后端 dense receipt 时不得给 paired cost saving。
3. 区分 heavy-backbone frame reduction、端到端延迟下降、FLOPs 下降、显存下降和视频 I/O 下降，禁止互相替代。
4. 判断当前 idea 最能防守的 novelty 是什么：间接状态转变采样、Oracle-calibrated boundary burst、受保护 detector feedback、结构化预算约束，还是它们的组合。
5. 列出审稿人最可能攻击的五个问题，以及必须用哪张表/图/消融回答。
6. 明确回答：若最终只与 uniform 持平但成本更高，路线是否应 KILL；若只在 K128/192 优于 uniform，论文主张应如何收缩；若 K384 超过 65 但仍低于 dense 68.29，是否仍有发表价值。

## 12. 强制输出格式

严格按以下顺序输出，使用中文：

1. `VISIBILITY_CERTIFICATE`
2. `EXECUTIVE_VERDICT`
   - 代码正确性：GO / HOLD_FIX_REQUIRED / KILL
   - 模型机理：GO / HOLD_FIX_REQUIRED / KILL
   - 当前实验有效性：GO / HOLD_FIX_REQUIRED / KILL
   - 论文可发表性：GO / HOLD_FIX_REQUIRED / KILL
3. `ACTUAL_MODEL_GRAPH`
   - 前向数据流、shape、坐标、hard 决策、训练专用信号
4. `LOSS_AND_GRADIENT_OWNERSHIP`
   - loss x parameter-group 矩阵
5. `LINE_BY_LINE_FINDINGS`
   - Critical/High/Medium/Low，精确文件与行号
6. `ORIGINAL_INTENT_COMPLIANCE`
   - 哪些完全符合初心，哪些偏离，哪些只是命名误导
7. `WHY_LEARNED_LAGS_UNIFORM`
   - 根因排序、证据、最快证伪实验和决策树
8. `MODEL_OPTIONS`
   - 3--5 个方案的数学、代码、成本、风险比较
9. `FINAL_RECOMMENDED_MODEL`
   - 唯一推荐的完整结构和为什么
10. `EXACT_TRAINING_RECIPE`
    - 可直接转 config 的阶段/更新数/权重/冻结表
11. `PATCH_PLAN_AND_CORE_CODE`
    - 文件、函数、伪代码/核心 PyTorch 实现、测试
12. `BOUNDED_EXPERIMENT_PLAN`
    - 当前 60-cell 结果后的唯一有界路线
13. `PUBLISHABILITY_AND_CLAIMS`
    - 能写、不能写、KILL 条件、最防守 novelty
14. `TOP_10_ACTIONS`
    - 按性能信息增益排序，不得由工程琐事占据前列

最终不要给模糊折中结论。请明确回答三个问题：

1. 当前 `a00498e` 是否已经正确实现一个可训练、可迁移、真正 detector-aware 的 pre-backbone TAD 智能采样插件？
2. 现有结构最可能卡在 coarse evidence、边界分配、梯度估计、时间轴表示还是训练课程中的哪一项？
3. 下一版只允许做一次主要结构修改时，究竟应改什么，为什么它比其余方案更可能让 matched official mAP 超过 uniform，并向 dense 性能靠近？
