# PhysTime 完整讨论、方法演化、实现与实验档案

更新时间：2026-07-13

## 0. 结论先行

PhysTime 有三个必须分开的层次：

1. **PhysTime-TAL 1.0**：第一版连续/物理时间设计，已被严厉评审否定为论文最终规格。
2. **PhysTime-TAD 2.0**：support-integrated measure detector，核心 feature-geometry 代码已实现并通过 focused gates。
3. **PhysTime-AdaTAD 1.0**：raw-video official AdaTAD 三头 full run 已完成，当前实现为负结果并冻结；后验诊断发现 feature provenance、容量、候选与 assignment 混杂。
4. **SM-PTAF**：2026-07-13 Pro 审查提出的 designed rebuild candidate，尚未实现、部署或产生 mAP。

任何未来报告必须先声明自己属于哪一层，禁止把 2.0 feature 代码写成 1.0 raw-video 主实验已经完成。

## 1. 方向来源

DUCA 暴露了一个比 selector 更基础的问题：selected positions 即使能 post-hoc 映回 original time，ActionFormer 内部仍把不规则观测当作等间隔 rank sequence。

被破坏的并不只是输出坐标，而是：

- Conv1d/attention 邻接关系；
- receptive field 的物理跨度；
- FPN stride 和 regression range；
- point assignment 与 center sampling；
- 不同采样密度区域的 token-count bias；
- sparse gap 的可观测性。

因此问题被重写为：

> 一个 TAD detector 如何直接接受任意不规则观测及其真实时间戳，在物理时间轴上分类和定位，而不是依赖 selected-rank inverse remap？

用户明确选择这一方向，因为它是独立的新 TAD 检测方法，不是 DUCA 的补丁或插件。

## 2. 文献碰撞与新颖性边界

### mTAN

已使用 continuous-time embedding 和 attention，将可变数量不规则观测投影为固定表示。PhysTime 不能只做 timestamp embedding + attention。

### TE-TAD

已使用 actual timeline coordinate，并随视频长度调整 query 数。PhysTime 不能只声称“实际时间坐标”。

### Temporal Robustness Benchmark / FrameDrop / TRC

已系统研究 TAD temporal corruption，发现主要退化来自 localization，并提供缺帧 augmentation 与 consistency。PhysTime 不能只靠 random drop + consistency 作为贡献。

### LiquidTAD

已把 continuous/liquid dynamics prior 引入高效 TAD，并报告 sampling variation robustness。PhysTime 不能宽泛声称首个 continuous-time TAD。

### ActionFormer / AdaTAD

它们是 matched detector/backbone 基线，但默认规则时间序列或固定窗口。PhysTime 必须证明不仅是 head 名称变化。

因此最终可防守的新颖性只能是：

> explicit observation-support measure、non-expanding gaps、physical query cells 和 seconds-native TAD head 形成一个可验证的 operator/geometry 闭环。

## 3. PhysTime-TAL 1.0

### 3.1 初始设计

输入：

```text
features [B,K,C]
timestamps [B,K]
support_widths [B,K]
valid_mask [B,K]
duration [B]
```

模块：continuous timestamp embedding、relative-time attention、physical query pyramid、start/end hazard、两种不规则 view 的 resampling consistency。

### 3.2 训练设想

同一 dense video 在线生成两个独立 irregular views，使用：

```text
L = L_TAD + lambda_eq * L_resample + lambda_h * L_endpoint
```

并禁止 actionness、budget、gap、radius、selector utility 等旧 DUCA loss。

### 3.3 被 HOLD 的八类问题

1. **等变性表述过强**：缺失区域无法要求两个 view 输出完全一致。
2. **normalized time 不足**：`[0,1]` 不能替代绝对秒或原帧尺度。
3. **support width 错误**：邻接 timestamp/Voronoi 容易跨真实 gap 填充质量。
4. **固定 M 问题**：固定 query 数不保证跨时长视频具有相同物理 receptive field。
5. **hazard 名不副实**：普通 endpoint logits/BCE 不自动构成按时间积分的 intensity/hazard。
6. **feature drop 证据不足**：删除预提取 token 不等于 raw-video 缺帧/FPS 变化。
7. **consistency 可观测性缺失**：只能在共同可观测 physical region 比较。
8. **dense degeneration 未定义**：等间隔输入与标准 detector 的关系缺少严格退化合同。

结论：1.0 不直接实现为 final paper method。

## 4. PhysTime-TAD 2.0

### 4.1 输入合同

```text
features              [B,C,K]
timestamps_sec        [B,K]
support_intervals_sec [B,K,2]
valid_mask            [B,K]
duration_sec          [B]
domain_start/end_sec  [B]
```

要求 timestamp 严格递增、support 有限非空并包含 timestamp、padding support 为零。GT 与预测均为 absolute video seconds。

### 4.2 Support ownership

原始 support 可以重叠，但只能用相邻 timestamp midpoint clip 去重：

- 可以缩小 overlap；
- 不得扩展 support；
- dropped observation 留下真实 gap；
- 不能把 rank-adjacent sparse frames 拼成连续 support。

### 4.3 Support-integrated measure attention

对于 query cell `R_q` 和 observation ownership `I_i`：

```text
m_qi = length(R_q intersect I_i)
w_qi = m_qi * exp(content_logit + relative_time_logit)
y_q  = sum_i w_qi V(x_i) / (sum_i w_qi + eps)
```

mass 在指数外，保证 constant-kernel 下拆分同一 support 不改变总贡献。token count 不应给局部密集采样额外权重。

### 4.4 Physical query pyramid

原始 2.0 规格令每层 cell width 为 `base_spacing_sec * 2^l`，对齐全局 0 秒原点，并裁剪到当前 domain；query count 由物理时长/spacing 决定，与 K 无关，每层直接从原始 observations 投影。2026-07-13 的公平性裁决只对 matched comparison 修正 candidate cardinality：允许 K 决定候选数，但坐标、宽度和回归 stride 仍由秒定义，禁止 selected-rank stride-2 几何。

### 4.5 PhysTime head

每个有效 physical query 预测：

- class logits；
- 以 query cell width 为单位的左右非负距离；
- start/end event intensity。

endpoint 概率：

```text
lambda = softplus(logit)
p_event = 1 - exp(-lambda * cell_width_sec)
```

assignment、regression、decode、NMS、evaluation 均使用 seconds。

### 4.6 Primary losses

2.0 支持 classification、DIOU regression、integrated endpoint loss，以及可选 common-coverage consistency。Consistency 只在两个 view 共同有效 physical cells 上比较 pre-NMS distributions/segments。

## 5. 已实现代码

当前 branch 中真实存在：

### Geometry

`opentad/models/utils/phystime_geometry.py`

- `validate_physical_observations`
- `clip_to_ownership_intervals`
- `support_overlap_mass`
- `build_physical_query_pyramid`
- `geometry_from_metas`

### Projection

`opentad/models/projections/phystime_projection.py`

- `PhysicalQueryEmbedding`
- `SupportIntegratedMeasureAttention`
- `PhysTimeMeasureProjection`

### Head

`opentad/models/dense_heads/phystime_head.py`

- physical points/ranges；
- integrated event probability；
- classification/regression/endpoint towers；
- seconds-native decode。

### Detector

`opentad/models/detectors/phystime_tad.py`

- strict metadata/no-leak validation；
- observations extraction；
- single/paired view training；
- common coverage consistency；
- seconds inference；
- optimizer groups。

### Feature transforms

`opentad/datasets/transforms/phystime.py`

- irregular feature sampling；
- feature support geometry；
- paired views；
- selected-axis/timestamp baselines。

### Gates/tests

- geometry、measure invariance、head gradients、detector registry/optimizer；
- GT seconds conversion、no-GT sampling、gap provenance；
- selected-axis/timestamp baselines；
- CUDA/synthetic precheck；
- feature-track config/deployment contracts。

## 6. Feature-token 实验轨道

### 原计划

使用 ActionFormer I3D two-stream stride-4 features，比较：

- support-integrated PhysTime K384；
- point-Gaussian K384；
- no-consistency K384；
- selected-axis K384；
- timestamp selected-axis K384；
- PhysTime K192/K768。

训练前设置 data job、real feature one-batch gate 和七个依赖 job。

### 远端作业

- data：`1156248`
- real gate：`1156249`
- pilots：`1156250` 至 `1156256`

### 为什么取消

用户明确要求直接验证 AdaTAD raw-video 端到端路径上的稀疏头有效性和真实计算节省。预提取 I3D：

- 不验证 raw decode/backbone savings；
- 可能把 feature extractor 成本排除；
- 不能证明 official AdaTAD adapters 受 TAD loss 联合训练；
- 只回答 feature geometry，不回答最终研究目标。

因此所有 feature jobs 取消，不得作为 paper evidence。代码只保留算子与 diagnostics。

## 7. PhysTime-AdaTAD 1.0 当前方案

本节保留 2026-07-11 的预注册方案。实际实现、full-run 结果和后验审查由第 9、10、16、17 节覆盖；不能再用本节的未来时态判断当前状态。

### 7.1 唯一研究问题

> 在完全相同的不规则 raw-frame observations 和相同 official AdaTAD/VideoMAE-S backbone 下，显式 physical-time detection 是否优于 ordinary selected-rank sequence？

### 7.2 为什么先用相同无学习采样

用户明确要求采用相同、无学习、无 GT 的不规则采样，只比较检测头。这样可以隔离：

- sampling policy 收益；
- actionness/selector supervision；
- dynamic budget；
- teacher/ledger；
- head/geometry 本身。

这避免重复 DUCA 的归因混乱。

### 7.3 三个 matched systems

1. **Selected-axis AdaTAD**
   - same K frames；
   - original ActionFormerHead；
   - GT 映射到 0..K-1；
   - 代表忽略真实间隔的 baseline。

2. **Physical-grid ActionFormer AdaTAD**
   - same K frames；
   - 复用已有 physical-grid assignment；
   - GT 保留 original timeline；
   - 最强低改动 baseline。

3. **PhysTime-AdaTAD**
   - same K frames；
   - official VideoMAE-S adapters；
   - `[B,384,K]` backbone feature，其中 384 是 channel、K 是 temporal observations；
   - measure projection + PhysTimeHead；
   - seconds-native predictions。

Dense 768 AdaTAD 仅作 accuracy/compute reference。

### 7.4 Sampling contract

- logical window 最多 768 positions；
- primary K=384；
- deterministic `random_fixed_subsample`；
- sampling 不读 GT/actionness/prediction/learned score；
- train crop 可以沿用 AdaTAD annotation-aware crop，但 crop 内 sparse selection 必须 GT-independent；
- 三头逐 sample selected-index checksum 相同；
- padding 只重复最后 valid frame，valid mask 保持 prefix。

### 7.5 Raw physical metadata

需要新 transform 产生：

- timestamps_sec = original frame index / FPS；
- original dense sampling cell support，宽度 `snippet_stride/fps`；
- duration/domain；
- original raw frame indices；
- `gt_time_unit=seconds`；
- `prediction_time_unit=seconds`；
- `remap_gt_to_selected_axis=False`；
- no-GT sampling audit flags。

support 不得扩大成 Voronoi cell。GT 只从 crop/window axis 转换一次为 video-absolute seconds，double conversion 必须 fail closed。

### 7.6 Official backbone contract

- `mmaction.Recognizer3D`；
- `VisionTransformerAdapter` + VideoMAE-S initialization；
- 16-frame chunks；
- K 必须可被 16 整除；
- K384 对应 24 chunks；
- trunk LR=0；
- temporal adapters、projection/head trainable；
- no interpolate back to 768 in PhysTime branch；
- detector loss 必须到达 adapters、projection、classification、regression、endpoint。

### 7.7 Primary supervision

只使用各 head 原生 detection supervision。PhysTime 使用 cls/reg/endpoint，primary comparison 暂不开 paired consistency，因为它会增加监督量，破坏 head isolation 公平性。

## 8. 为什么可以映射回帧号，但不能映射回 selected rank

秒和原视频帧号是同一物理轴的两种单位：

```text
frame_index = round(time_sec * fps)
time_sec = frame_index / fps
```

因此预测可导出原视频 frame ID，用于可视化、帧级评测或部署接口。

不能映射到“第几个被选中的帧”，因为 selected rank 只表示列表顺序：rank 10->11 可能跨 1 帧，也可能跨 15 帧。把 GT/预测压到 rank 会重新丢失 PhysTime 要保护的物理距离。

## 9. PhysTime-AdaTAD 当前实现状态

### 已完成

- 设计规格：commit `9266ebc`；
- implementation plan：commit `517785d`；
- raw geometry、K384 三配置、same-index validator、真实 AMP/evaluator gate 与两 epoch stability gate；
- 最终稳定正式实现：commit `3ac93a1`；
- selected-axis、physical-grid 和 PhysTime 三头 full run；
- 最佳 checkpoint 只读重放与正式结果一致；
- 性能下降诊断与实验完整性审计：commit `d900c7c`。

### 尚未完成

- native tubelet feature 与 raw support atoms 的 provenance gate；
- capacity/context/candidate/assignment-matched coordinate-only controls；
- `SM-PTAF` 仓库实现与 focused tests；
- 修复后的单因素 pilot、三 seeds、第二数据集和 sampling-family matrix；
- raw decode 到 NMS 的完整成本账本与可访问 artifact bundle。

当前必须说：PhysTime-AdaTAD 1.0 已实现并完成实验，但结果为负且比较存在结构混杂；SM-PTAF 只有 `designed` 状态。

## 10. 实验阶段

### Phase 0

1.0 的 unit/registry/CUDA synthetic/real raw-video gate 已完成。下一版 Phase 0 是 native feature provenance、candidate parity、assignment parity、translation 与 optimizer contracts。

### Phase 1

首版三头 matched K384 comparison 已完成并冻结为负基线。当前 Phase 1.5 先做 capacity-matched selected-coordinate 与 physical-coordinate control，再决定是否加入 support-measure lift。

### Phase 2

仅在 Phase 1.5 机制合同与单 seed survivor 通过后解锁：

- K192/384/768；
- random、uniform、bursty、contiguous-gap；
- multi-seed；
- mTAN-like、timestamp、interpolation、FrameDrop/TRC、TE-TAD、LiquidTAD；
- second dataset、cross-FPS。

## 11. 成功、降级和停止条件

- 胜 selected-axis + physical-grid，且高-IoU/边界改善：完整 PhysTime candidate。
- 只胜 selected-axis：只支持 original-time geometry；不证明 measure head 必要。
- 不胜 physical-grid：完整 head 主张失败。
- timestamp/interpolation/mTAN-like 在 CI 内持平：停止强调复杂 operator。
- K384 明显落后 dense 且无强 Pareto：不能作为 paper main。
- mAP@0.7、短动作、held-out gaps 无收益：鲁棒定位主张失败。
- 第二数据集无法复现：收缩为 dataset-specific observation-geometry study。

## 12. 与 DUCA 的关系

PhysTime 吸收：

- no-leak/provenance；
- coordinate round trip；
- same-selected-frames geometry comparison；
- official backend/optimizer audit；
- high-IoU 与 full-cost 证据纪律。

PhysTime 丢弃：

- actionness/probe；
- learned selector；
- teacher utility；
- budget controller；
- max-gap/radius；
- ST bridge；
- ledger。

## 13. 与 ChronoTransport 的关系

PhysTime 解决 detector 时间几何；ChronoTransport 解决 heavy feature recompute。二者理论可组合，但当前不组合，以免把 geometry gain 与 conditional compute 混合，并避免方法范围爆炸。

## 14. 禁止重复

- 不得把 continuous-time 作为宽泛 novelty；
- 不得把 feature-token gate 写成 raw-video evidence；
- 不得让 sampler 读 GT；
- 不得把 support 扩过 gap；
- 不得先引入 learned selector；
- 不得 primary comparison 加 consistency；
- 不得只和 selected-axis 比而跳过 physical-grid；
- 不得只报 Avg-mAP；
- 不得省略 full-stack cost；
- 不得把 PhysTime 1.0 的负结果外推为 physical-time TAD 无效；
- 不得把 SM-PTAF 公式或代码草图称为已实现；
- 不得继续用 `192 -> 384` feature interpolation 冒充 raw support provenance。

## 15. 当前唯一下一步

执行 `HOLD AND REBUILD` 的 P0 顺序：

1. native tubelet multi-atom feature-support provenance；
2. capacity/context/candidate/assignment-matched coordinate-only control；
3. raw absolute center off、candidate parity、tied multi-label assignment；
4. mass residual 与 bounded/off content correction；
5. survivor 通过后再实现完整 SM-PTAF pilot。

在这些 gate 前不启动第二数据集、多 seed 或昂贵 full matrix。

## 16. 2026-07-12 性能下降诊断

正式结果排除了训练崩溃、重复秒转换、缺失测试窗口、错误 evaluator 和 checkpoint 读取错误。诊断确认：PhysTime 1.0 的 detector capacity/context 显著弱于 ActionFormer controls，absolute seconds 主导 query，粗层 attention 有效聚合坍缩，候选与短动作监督不足，assignment 也不同构。正确高-IoU 匹配后的边界质量并非全面失败，主要问题更接近覆盖与排序。

因此 1.0 是高价值负基线，但不能裁决 physical-time-native detector 假设。

## 17. 2026-07-13 Pro 审查与 SM-PTAF

完整原文：`docs/methods/reviews/2026-07-13-phystime-performance-drop-pro-audit-response-raw.md`。

吸收记录：`docs/methods/2026-07-13-phystime-performance-drop-pro-audit-absorption.md`。

Pro 给出 `HOLD AND REBUILD`，并把最严重的新问题定位为：原生 192 tubelet feature 被插值到 384 后与 384 raw-frame supports 一一绑定，没有建立 feature provenance。推荐候选 `SM-PTAF` 使用 native tubelet multi-atom supports、measure-preserving mass residual、bounded correction、ActionFormer 等级 physical query encoder、候选/容量/assignment parity。

本 Wiki 的吸收边界是：接受问题裁决与因果顺序；把 SM-PTAF 登记为 `designed`；不把外部 PyTorch 片段当作已实现；先完成 coordinate-only control，再判断该候选是否值得 full train。

## 18. 2026-07-19 Full60 / Q-Lift Pro 审查

原文：
`docs/methods/reviews/2026-07-19-phystime-full60-q-lift-pro-review-raw.md`。

吸收：
`docs/methods/2026-07-19-phystime-full60-q-lift-pro-review-absorption.md`。

审查确认 `0dc5851` 的 `57.57%` 是可信单种子 full60 证据，同时把其
作用域收窄为检测头后段的物理时间度量；VideoMAE/TIA/projection 尚不见
timestamp。当前 matched 两臂公平，旧随机 `63.61%` 和 dense `68.29%`
因 interpolation、Q、坐标或观测成本变化只能作历史锚点。

下一候选登记为 support-preserving physical query lift：保持 K384/J192
support，独立构造 Q192/Q384 query，做
`Q × coordinate` 四臂 20-epoch 因子实验。先修全精度跨窗口 NMS，并补
proposal recall、assignment、observability、合法 timestamp counterfactual
和成本诊断。cross-attention 是优先候选而非已证明唯一结构；外部固定阈值
和 ActivityNet 选择不直接写入研究合同。状态仍为 `designed`，新 full
train 未解锁。
