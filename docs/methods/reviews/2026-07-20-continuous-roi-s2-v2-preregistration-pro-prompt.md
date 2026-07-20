# Continuous-RoI S2 v2 Pro 严审提示词

你是计算机视觉、视频理解、时序动作检测、动态视觉计算和严谨实验设计方向的最高标准审稿人兼首席研究工程师。本轮不是让你继续发散新想法，也不是让你直接实现代码，而是要求你对一个已经明确研究目标、但尚未冻结协议的空间裁剪路线进行逐行代码审计，并输出唯一一份可以直接交给工程团队编码、验证和排队的预注册协议。

你的唯一输出标题必须是：

```text
# Continuous-RoI S2 Crop-Sufficiency Preregistration v2
```

不得输出想法清单、宽泛建议、多个候选方案、论文宣传文案或逐步探索计划。你必须替我们完成必要的技术取舍，给出一份无 `TBD`、无“可以考虑”、无悬空超参数的冻结协议。

## 0. 冻结审计对象与可见性门

仓库：

```text
https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
```

分支：

```text
codex/spatial-zoom-s1-audit-fix-20260715
```

本轮不可变审计提交：

```text
6118cd50a3601d044dab690427ad9c756ce7d827
```

提交树：

```text
https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/6118cd50a3601d044dab690427ad9c756ce7d827
```

你必须先执行可见性门：

1. 打开上述不可变提交树，并报告实际看到的 commit SHA。
2. 若 GitHub tree 页面不可读，不得立刻声称仓库不存在；必须继续尝试：

```text
https://raw.githubusercontent.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/6118cd50a3601d044dab690427ad9c756ce7d827/<FILE_PATH>
```

3. 对每个实际读取的文件列出文件路径和有效行号范围。不得凭本文摘要伪装“逐行审计”。
4. 只有 tree 与 raw 两种方式都失败时才允许输出 `BLOCKED`，并必须写明连接器或 HTTP 层面的真实失败信息以及缺失的最小文件集合。
5. 若可见性通过，后文所有代码判断必须给出真实 `file:line`；无法定位的判断必须标记为“协议设计要求”，不得伪造现有实现。

## 1. 强制阅读范围

至少逐行读取并审计：

```text
AGENTS.md
RTK.md
research-wiki/query_pack.md
research-wiki/anti_repetition.md
research-wiki/ideas/spatial-zoom-offline-tad.md
research-wiki/experiments/spatial-zoom-s1-infrastructure.md
research-wiki/experiments/native-crop-s2-crop-sufficiency.md
research-wiki/experiments/native-crop-paper-experiment-roadmap.md
docs/methods/native_crop_s1_vertical_slice_contract.md
docs/methods/reviews/2026-07-20-native-crop-s2-crop-sufficiency-preregistration-pro-raw.txt
docs/methods/reviews/2026-07-20-native-crop-s2-crop-sufficiency-preregistration-pro-absorption.md
opentad/datasets/transforms/native_crop.py
opentad/models/backbones/native_crop_wrapper.py
configs/adatad/thumos/native_crop_s1_center_videomae_s_768x1_adapter.py
tools/bata/native_crop_s1_contract.py
tools/bata/run_native_crop_s1_precheck.py
tools/bata/native_crop_s1_geometry_census.py
scripts/run_native_crop_s1_gate_slurm.sh
tests/test_native_crop_s1_vertical_slice.py
opentad/models/detectors/single_stage.py
opentad/models/detectors/actionformer.py
opentad/models/backbones/vit_adapter.py
opentad/models/projections/actionformer_proj.py
opentad/models/dense_heads/actionformer_head.py
opentad/cores/train_engine.py
opentad/cores/optimizer.py
opentad/evaluations/mAP.py
opentad/models/utils/post_processing/nms/nms.py
```

还必须沿 config 和 registry 实际追踪数据集、增强、VideoMAE checkpoint 加载、adapter、head、optimizer、evaluator 与 NMS 的真实调用链；上表不是允许你停止追踪的封闭清单。

同时阅读以下官方一手来源，不得以二手博客代替：

```text
Uni-AdaFocus paper: https://arxiv.org/abs/2412.11228
Uni-AdaFocus official repo: https://github.com/LeapLabTHU/Uni-AdaFocus
AdaFocusV2 paper: https://arxiv.org/abs/2112.14238
```

明确说明我们借鉴了哪些已验证机制，又有哪些部分不能直接搬到离线 TAD。

## 2. 不可误解的研究目标

这是离线 Temporal Action Detection，不是 Online TAD。完整时间轴保留为 768 点，不做时序选帧，不做 DUCA、dynamic temporal budget、X3D/SlowFast 先验，也不在本轮实现 learned ROI policy。

最终空间方法不是：

- 整幅图像固定降到 160、224 或 256；
- 固定大小的中心裁剪；
- 从 21 个固定 `128x128` 候选框中离散选一个；
- 固定源裁剪大小，只回归中心点；
- 以低分辨率整图实验替代空间裁剪实验。

最终研究对象是 Uni-AdaFocus 风格的连续可变形源坐标 ROI：

```text
b_t = (cx_t, cy_t, w_t, h_t),  b_t in [0,1]^4
```

中心、宽、高、面积、尺度和长宽比都可以随视频及时间组变化。允许把可变源 ROI 重采样到固定大小的局部张量，以便批处理和提供可预测的重计算成本；但必须明确：

```text
固定局部输出张量 != 固定源 ROI != 保持原生像素密度
```

S1 已通过的 CUDA gate 只证明固定中心 source crop、共享 VideoMAE、AdaTAD detector contract、梯度和无泄漏基础设施可以运行。它没有证明 crop sufficiency、连续 ROI、精度、成本、learned policy 或论文主张。

旧 S2 v1 的 21 个固定 `128x128` 候选仅允许作为 `D0 fixed-library diagnostic`，不得再作为决定 Continuous-RoI 路线生死的主协议。

## 3. 必须先整理并解决的历史问题

你的协议必须逐项给出可执行解法，而不是重复问题：

1. **研究对象错位**：过去把“空间裁剪”混成“整图降分辨率”或固定窗口，未验证连续可变框初心。
2. **固定候选歧义**：21 框只能证明有限库条件下的表现，既不是连续 ROI，也不是全局 oracle。
3. **无空间标注**：THUMOS14 只有时间段标签，没有目标框；任何“空间 oracle”“GT crop”都必须诚实定义，不能凭空制造真值。
4. **参考搜索不可证上界**：有限 restart、Sobol 点、离散候选或梯度搜索都只是算法条件下的参考，不是连续域或 global-mAP upper bound。
5. **参考失败的逻辑错误**：启发式参考失败可能是搜索覆盖不足或模型不可达，不能直接判定连续空间裁剪路线失败。
6. **训练与推理异构**：训练时 `grid_sample` 的连续重采样与推理时 source-coordinate decode/crop/resize 可能在坐标、padding、rounding、`align_corners`、边界和插值上不一致。
7. **几何退化**：直接学习宽高会出现 `w,h -> 0`、极端长宽比、越界、边界夹紧、背景裁剪和过度放大。
8. **时间抖动**：逐帧回归 ROI 会产生高频抖动、额外成本和 tubelet 不一致；固定全视频一框又可能抹掉动作移动。
9. **上下文丢失**：局部分支可能丢失场景和类别上下文；全局分支过强时又可能让局部分支及 ROI policy 被训练忽略。
10. **监督不足与早期不稳定**：仅靠最终 detector loss 可能无法学出有意义 ROI；但分阶段独立训练又偏离单模型协同学习目标。
11. **伪监督风险**：动作时段 GT 只能提供时间监督，不能被包装成空间框；teacher、test GT、oracle cache 和事后选择不得进入部署推理。
12. **成本语义错误**：如果局部输出张量尺寸固定，改变源 ROI 面积不会自动改变局部 backbone FLOPs。节省来自局部重计算张量/重分支设计，而不是“框更小”本身。
13. **双分支成本遗漏**：共享一套参数不等于只做一次 forward；global scout、box head、crop、local backbone、fusion、detector、NMS 都要计费。
14. **detector 被暗改**：必须保证后端仍是可核验的 AdaTAD/VideoMAE/ActionFormerHead 路径，明确所有必要修改，不得用“AdaTAD-like”替代官方结构。
15. **统计单位混用**：检测差异应按视频聚类并跨 seed；延时和能耗应按同机同卡 ABBA block/window 配对，不能塞进同一个 bootstrap。
16. **测试泄漏**：official test 继续封存；development gate 的 raw inference 必须先在无 GT 状态下封存，再由独立 privileged join 读取 GT。
17. **过度主张**：S2 只能验证表示充分性、连续几何 headroom 和部署成本可行性，不能提前宣称 learned Zoom 成功或论文主方法成立。
18. **基础设施反复失败**：现有路线曾遇到 selector 误拒绝 finite zero-length proposals、磁盘耗尽、功耗 sidecar 采样间隔、环境 NumPy/schema/provenance 问题；v2 必须复用已验证 fail-closed、哈希、receipt、幂等和 immutable namespace 机制，但不得让基础设施复杂度掩盖核心科学问题。
19. **Pro 可见性误判**：不得因为一次 GitHub 连接器失败，在未尝试 raw URL 前假装完成或直接给出无代码依据的协议。
20. **实验结论混淆**：deterministic geometry coverage、model-conditioned reachability、representation sufficiency、adaptive headroom 和 deployable cost viability 必须是可分别失败的 estimand。

## 4. 你必须冻结的连续几何与计算契约

你必须选择并写死一个精确的可微参数化。可以审查并采用或替换下式，但最终只能保留一个版本：

```text
w = w_min + (w_max - w_min) * sigmoid(s_w)
h = h_min + (h_max - h_min) * sigmoid(s_h)
cx = w/2 + (1-w) * sigmoid(s_x)
cy = h/2 + (1-h) * sigmoid(s_y)
```

协议必须给出并论证：

- `w_min/h_min/w_max/h_max`、面积范围和长宽比范围的唯一数值；
- 这些约束如何对应 THUMOS14 输入几何、VideoMAE patch/tubelet 和固定局部输出张量；
- 是否使用单框、每 temporal group 一框还是 spline/knots 轨迹；必须给出唯一 temporal granularity、插值方式和边界条件；
- box trajectory 的一阶/二阶平滑、面积/长宽比、coverage/context、anti-collapse 正则的精确定义、权重和 schedule；
- 如何避免框在动作时段内只追逐判别性小区域而损害边界定位；
- 如何在保留 768 点时间轴时对所有帧/片段应用空间 tube；
- 训练 `grid_sample` 与部署 source-coordinate crop 的坐标约定、像素中心、插值、padding、rounding 和允许误差；
- 固定局部输出张量的唯一尺寸，以及它为何是成本控制变量而非“固定源窗口”；
- global context、local ROI、fusion、projection 和 detector 的精确 tensor shape，特别是最终 detector contract `[B,384,768]`；
- 一套共享 VideoMAE 参数实例是否用于 global/local 两次 forward；若不是，必须说明参数与成本变化；
- detector loss 如何经 differentiable crop 和 ROI head 回传，哪些路径允许或禁止 `detach/stop-grad`；
- 推理时不允许 GT、teacher、reference search、oracle identity、离线 per-video optimization cache 或 test-derived threshold。

## 5. 必须冻结的一体化训练协议

我们要的是一个最终可联合训练的模型，而不是三个独立训练模型。你必须给出一份单一训练程序，明确：

- ROI policy 的输入：低成本 global features 的哪些层、时间分辨率和通道；
- global/local backbone、ROI head、fusion、projection、detector head 的参数共享与 optimizer groups；
- detector loss、global auxiliary loss、local auxiliary loss、box regularizers 和 temporal regularizers 的完整公式；
- 每个 loss 的唯一权重、启用 epoch、warm-up、annealing 和停止条件；
- 是否采用 AdaFocusV2 所述辅助 global/local supervision、随机 crop input diversity 和 differentiable interpolation；每项必须说明迁移到 TAD 的具体形式；
- 如何防止早期随机 ROI 破坏 detector、global branch 垄断预测、local branch 零梯度、box head 只学中心先验；
- 若采用 warm start 或 homotopy，必须仍属于一次联合训练协议，并给出精确阶段边界、冻结/解冻表和 checkpoint 选择规则，不得退化成三个独立训练模型；
- full-model one-step gradient gate：必须断言 ROI head、global/local backbone、fusion、projection 和 detector 的预期参数均有有限非零梯度，并审计所有 `requires_grad` 参数进入 optimizer；
- AMP、NaN retry、determinism、successful-update parity 和 fail-closed 规则；
- 训练只允许 fit split；gate 仅用于预注册 checkpoint selection，不反向传播。

## 6. S2 的连续参考必须被诚实定义

S2 不是直接训练最终 deployable policy，而是先回答：

1. 可变源 ROI 作为表示是否足够？
2. 连续改变中心、宽高、尺度和长宽比是否比固定框带来可测 headroom？
3. 在包含必要全局上下文和局部重计算后，成本是否仍有现实可行性？

你必须设计一个唯一、可运行、GT 使用边界清楚的 `continuous reference`。它可以基于“geometry-conditioned random-crop training + 预注册多起点连续框搜索”，也可以替换为你认为更严谨的单一方法，但必须同时满足：

- 明确哪些步骤只读 fit GT，哪些步骤可以在 development gate 使用 GT；
- gate raw model outputs、boxes、seeds、search trajectories 和 hashes 必须先无 GT 封存；任何 GT-visible ranking/join 在独立 privileged 阶段完成；
- 若连续优化必须直接读取 gate GT，就必须把它定义为 privileged diagnostic，并设计独立的 no-GT matched evaluator，绝不能把它冒充可部署推理；
- 固定随机种子、初始化分布、restart 数量、迭代数、optimizer、步长、约束投影、终止条件和搜索预算；
- 提供搜索覆盖/收敛诊断，区分搜索失败与表示失败；
- 禁止把 finite-search 最优称为 oracle、全局上界或可部署 selector；
- 参考搜索成本单独披露，绝不计入 deployable method 的成本优势；
- 即使 reference 失败，也只能进入“连续参考不足/不可裁决”状态，不能直接永久 KILL Continuous-RoI。

不要给两个备选 reference。你必须选定一个，并提供核心伪代码和所需文件。

## 7. 最小但充分的匹配实验矩阵

你必须冻结一个不过度膨胀但能完成归因的矩阵，至少覆盖：

```text
D160: same-runtime dense full-frame comparator
G: global-only branch
C: fixed center source crop
R: matched random fixed crop
D0: old 21-box fixed-library diagnostic
LC: learned center with fixed width/height
CR: continuous center/width/height reference
```

如你认为某一项统计上完全冗余，可以删除，但必须给出基于 estimand 的严格理由；不得增加无穷消融清单。冻结：

- THUMOS14 fit/gate 的样本身份与 `fit160/gate40` 规则；
- 三个固定 seeds；
- 相同数据、增强、checkpoint、成功更新数、batch exposure、optimizer、训练时长和 checkpoint 选择；
- 同一 AdaTAD/VideoMAE/ActionFormerHead detector contract；
- official test 在 S2 决策前继续 sealed；
- 每个 cell 的允许差异表和配置哈希；
- 任何 train-only geometry augmentation 的 matched 规则。

旧 21-box D0 只能回答 fixed-library 诊断问题，不能替代 CR，也不能独立触发路线 KILL。

## 8. 指标、统计与判决状态

必须分别定义以下指标和最小效应：

- Avg-mAP 与 `mAP@0.3/0.4/0.5/0.6/0.7`；
- high-tIoU、short-action Q1、boundary start/end error；
- source ROI 面积、长宽比、中心移动、速度/加速度、越界率、clamp 率、退化率和 tube jitter；
- global/local feature utilization 与 ROI-head gradient/connectivity；
- reference search coverage、restart disagreement 和 convergence；
- latency、peak memory、gross GPU energy、FLOPs/MACs 与参数量。

统计设计必须：

- 检测指标采用 paired video-cluster bootstrap，并正确纳入 seed 层级；
- 成本采用同节点、同物理 GPU、warm-serial、随机化或冻结 ABBA block/window 配对；
- detection family 与 cost family 分别做 simultaneous inference，不得混用采样单位；
- 最终联合判决使用预注册 intersection-union 逻辑；
- 在读取正式结果前运行 result-blind power/Monte-Carlo feasibility audit，冻结 effect margin、bootstrap 次数、max-T family 和不确定性报告；
- 明确 official evaluator parity、fixed class support 和 missing-prediction 行为；
- 禁止用中间结果移动门槛或选择 seed。

最终状态必须至少能区分：

```text
SUFFICIENT_AND_CONTINUOUS_HEADROOM
SUFFICIENT_FIXED_ONLY
SUFFICIENT_BUT_COST_NOT_VIABLE
CONTINUOUS_REFERENCE_INSUFFICIENT
INCONCLUSIVE_GEOMETRY_OR_SUPPORT
NO_DECISION_INVALID_EVIDENCE
```

为每个状态给出唯一布尔判定式及允许的下一步。只有第一种状态可以授权 S3 learned continuous ROI policy；第二种只说明固定 crop 表示足够，不能宣称自适应必要；第三种不能宣称高效；第四和第五不能永久 KILL 路线；第六必须修复证据后重做。

## 9. 成本口径必须完整且不自欺

冻结 inference cost ledger，至少包含：

```text
decode
CPU preprocessing
host-to-device transfer
global scout/global branch
ROI policy head
box decoding and temporal interpolation
differentiable/runtime crop and resize
local backbone
global-local fusion
projection/neck
ActionFormerHead
post-processing/NMS
```

还必须：

- 分开报告训练成本、S2 privileged reference search 成本和潜在 deployable inference 成本；
- 若 S2 尚无 deployable selector，只能报告 representation-path headroom，并为未来 selector 预留明确 latency/energy budget；
- 明确共享参数仍有两次 forward；
- 明确固定局部输出张量意味着 local backbone FLOPs 不随 source box 面积变化；
- 只在相同节点、同一物理 GPU、相同软件环境、warm-serial 下比较；
- 给出 per-window p50/p95、peak memory、gross GPU energy、sample count/max sampling gap 和 profiler self-hash；
- 禁止把 per-window warm profile 外推成 whole-video、cold-start 或系统级节能结论。

## 10. 输出必须直接支持编码和 Slurm 排队

你的唯一文档必须按以下顺序组织：

1. `Visibility and Read Certificate`
2. `Protocol Identity and Immutable Hashes`
3. `Research Thesis, Estimands, and Forbidden Claims`
4. `Data Splits, Permissions, and Sealed-Test Rules`
5. `Continuous ROI Geometry Contract`
6. `Temporal Tube and Interpolation Contract`
7. `Differentiable Training Crop vs Runtime Crop Parity`
8. `Model Architecture and Tensor Contracts`
9. `Joint Training Schedule and Exact Losses`
10. `Continuous Reference and Privileged-Join Protocol`
11. `Matched Baselines and Exact Matrix`
12. `Metrics and Diagnostics`
13. `Power Audit and Statistical Decision Rules`
14. `Full-Stack Cost Ledger`
15. `Outcome State Machine and Authorization Rules`
16. `Code Changes by Exact File`
17. `Core Implementation Pseudocode`
18. `Focused Unit/Integration/CUDA Gates`
19. `Slurm DAG, Idempotency, Receipts, and Failure Policy`
20. `Implementation-Ready Checklist`
21. `Final Verdict`

对代码部分必须给出：

- 需要新增、修改和明确禁止修改的精确文件清单；
- 连续 box decoder、temporal interpolation、differentiable crop、runtime crop parity、loss aggregation、optimizer coverage、no-leak sanitizer、reference search、sealed join、cost profiler 和 analyzer 的核心 Python/PyTorch 伪代码；
- 每个测试的名称、输入、断言和预期失败方式；
- `PRECHECK -> CPU focused tests -> real CUDA one-step gate -> three-seed fit/gate matrix -> sealed analysis` 的唯一 Slurm DAG；
- 每个 job 的依赖、资源、幂等 receipt、immutable namespace 和 fail-closed 规则；
- 哪些已有 S1 代码可以复用，哪些必须重构，哪些绝不能当作 Continuous-RoI 已实现的证据。

禁止直接打开 official test，禁止现在部署 learned policy，禁止把 21 框协议重新包装为 continuous，禁止用简化 detector 代替真实 AdaTAD 后端，禁止引入与本问题无关的新路线。

## 11. 最终裁决格式

你必须先给出一个三选一裁决：

```text
V2_READY
V2_HOLD_FOR_PROTOCOL_FIX
V2_BLOCKED_BY_EVIDENCE
```

若不是 `V2_READY`，只允许列出真正阻止编码的 P0/P1，并直接在同一回复中修正协议，使最终正文仍达到可编码状态；不要把取舍退回给用户。

最后一行必须且只能是：

```text
V2_READY_FOR_IMPLEMENTATION
```

或：

```text
V2_BLOCKED_BY_<一个且仅一个真实阻塞原因>
```
