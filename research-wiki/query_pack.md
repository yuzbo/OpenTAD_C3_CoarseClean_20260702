---
type: query_pack
updated: 2026-07-23
max_chars: 8000
---

# Research Query Pack

## GeoRoute activation update (2026-07-23)

- P1 bootstrap `1181007` reconstructed the sealed P0 parent but hit
  `AssocMaxSubmitJobLimit` after only dense leaf `1181008` was admitted. That
  partial namespace is an immutable diagnostic, not P1 evidence.
- The admitted dense leaf reached epoch 0 and exposed an untested real-input
  boundary branch. Its first flatten-to-NCHW repair still called CUDA
  `replicate` padding on uint8 decoded frames; native-padding gate `1181047`
  failed in 21 seconds because that CUDA kernel is unavailable. Its immutable
  replacement uses byte-preserving last-row/last-column concatenation, with
  bottom-only and bottom-plus-right exact-value regressions plus a fresh
  180x320 CUDA mechanical gate required before any fresh P1 DAG.
- P2/P3 remain result-gated; no metric, cost, official test, A-MoD result, or
  paper claim exists.
- Gate `1181047` is failed immutable diagnostic evidence, not a P0 revision
  or a model result. The three dead dependent jobs `1180494`--`1180496` were
  cancelled to release submit quota. No second partial P1 root, P1 Job, P2
  Job, or P3 Job has been created.

## Continuous-RoI S2 当前状态（2026-07-21）

- 正式训练提交 `9a61da27` 的 Gate `1177662` 与 exact-nine
  `1177668-1177676` 已完成；每格为 60 epochs、4,800 successful updates、final-EMA-only，
  且 official test 未打开。completion receipt 是 `PASS_TRAINING_ONLY`，不等于 crop
  sufficiency、mAP、成本或 learned ROI 结果。
- reference 协议仍 `HOLD`：fixed/variable 没有共享物理中心轨迹，且 Sobol 生成器、候选 ID
  权限、annotation-free raw、privileged join/tie/statistics 尚未共同冻结。只允许 result-blind
  v2.2 corrigendum；official test 和 S3 均继续封存。

## Geometry-Residual-Depth Routing / GeoRoute（2026-07-23）

- 状态为 `tested_p0_mechanical_only`：已实现单重主干原生 tubelet、exact-K
  ROI/free/hybrid、几何对照和结果盲 DAG；U128 与当前路线无关。
- 有效 P0 模型提交为
  [`4a9358d`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/4a9358d1fba4bde9aa7693a94f7e4dfc95d31ecc)：
  `1180906` dense、`1180907` ROI score-function、`1180927` hybrid 均
  `COMPLETED 0:0`。dense 使用匹配 autograd 的数值参考；hybrid 满足 exact-K=32、
  一次重主干和几何/residual scout 检测梯度；三者只证明 CUDA 接口与梯度机械性质。
- P0-only finalizer 修复提交
  [`c2a3c69`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/c2a3c69d9006b32b1ca18f4ce66222b59550c45f)
  的 Job `1180966` 已封存 suite `PASS_MECHANICAL_ONLY`
  (`a6f8ea…ce0b1`)；旧 `1180963` 因 Python 模块入口 fail-closed，保留诊断。
- P1 bootstrap 已实现、尚待提交：它从已封存 P0 suite 重新核验后只提交七个
  matched P1 cells，并冻结 P2/P3 的 result-gated successor policy；P2/P3 不会在
  P1/P2 的授权 receipt 前消耗 GPU。P1 仍必须同预算比较 fixed/random/free/ROI/hybrid
  与 geometry 对照；free 在高 IoU/总成本获胜即杀 ROI 主张。仍无 mAP、全成本、official-test、
  A-MoD 或论文主张。

## Native-Crop S2 协议裁决（2026-07-20）

- 用户已明确最终目标是 Uni-AdaFocus 式连续 deformable ROI：策略回归
  `(cx,cy,w,h)`，中心、宽、高、尺度和纵横比均不固定；固定 local tensor shape
  仅是批处理规格，不等于固定 source window。
- 21-candidate fixed-128 v1 已降级为 D0 离散诊断；Pro v2 虽自评 `V2_READY`，项目
  裁决仍是 `ACCEPT_WITH_MAJOR_REVISION`，因为它混入 S3 policy、比较权限不匹配且
  confidence convergence 不能证明空间覆盖。
- 接受其 source-coordinate 连续框、宽高防塌缩、temporal tube、可微/运行时 sampler
  parity、共享 VideoMAE 单实例、真实 AdaTAD-derived detection path、detector 梯度、
  no-GT raw 后 privileged join、D0 仅诊断和 full-stack cost ledger。
- 项目自写 v2.1 corrigendum 已冻结并通过静态验证：协议 SHA-256 为
  `ef806b7c...b3af`，八类合同与 128 种状态组合全部通过，审计 SHA-256 为
  `5af59b75...f9d`。它采用 selector-free common-support `U128`，成对且同权限的
  fixed/variable reference，并拆开 S2 表示充分性与 S3 learned policy。
- 完整结论见
  `docs/methods/reviews/2026-07-20-continuous-roi-s2-v2-preregistration-pro-absorption.md`。

## Native-Crop 论文实验路线

- 当前处于 `continuous S2-P v2.1 implementation`，不是主实验训练阶段。S1 Gate 只证明
  crop 数据流、模型合同、梯度和 no-leak 可执行；S2 将证明 crop sufficiency、
  adaptive headroom 与 cost viability，仍不是 deployable final method。
- 只有 `SUFFICIENT_AND_POLICY_HEADROOM` 才解锁 S3 learned crop policy；固定 crop
  已足够时转为简单 fixed-global/local 裁决，成本不成立时停止效率主张。
- 论文主证据来自 S3/S4：runtime learned crop selector、THUMOS14 official test
  三种子 accuracy-cost 主表与 Pareto 曲线。paper-ready 还要求 TriDet 第二检测头、
  ActivityNet-1.3（完整性失败则预注册 FineAction）第二数据集，以及最小机制消融。
- 完整依赖图、主表和 stop rules 见
  `research-wiki/experiments/native-crop-paper-experiment-roadmap.md`。

## Spatial Zoom 路线纠偏（2026-07-19）

- 真实研究目标是保留完整时间轴，并在源视频坐标中选择保持原生局部像素密度的空间
  crop/ROI tube；不是把整幅画面统一缩小到 160/224/256。
- 现有 `Dense-160/224/256 x 3 seeds` 只验证整图分辨率敏感性。它不含 ROI、scout、
  native crop、局部高分辨率分支或全局/局部融合，不能作为空间裁剪有效性的证据。
- 旧 S1 从必经 falsification gate 降级为 `R0 dense-resize headroom control`。保留训练、
  sealed-test、profiling 与 provenance 基础设施，但暂停继续部署其 recovery Gate/matrix；
  不得要求 R0 完成后才允许验证 native crop。
- 定向 Pro 审查已经完成。当前唯一方法任务是 geometry census 和无训练 Native-Crop
  垂直切片；它必须从源帧坐标裁剪、保持局部原始像素密度与 768 点时间轴。垂直切片和
  后续 crop sufficiency 通过前不得实现 learned crop policy。
- 当前没有任何 S1 crop 结果、GO/KILL 或可发表的 Spatial Zoom 主张。

## Native-Crop S1 历史审查

- S1 仅证明 source-native crop 数据/模型/梯度/no-leak 垂直切片可运行；它不是精度、
  成本、ROI 或论文证据。固定候选库也不能裁决连续路线。完整记录见
  `docs/methods/reviews/2026-07-20-native-crop-s1-pro-review-{raw.txt,absorption.md}`。

## R0 Dense-Resize 历史状态（冻结）

- R0 只包含全图 `160/224/256 x 3 seeds`，不含 scout、ROI、native crop 或 fusion。
  Gate means 为 `64.220/64.228/64.252`，近乎平坦；它们不是 official-test 结果。
- 唯一 official-test cell 是 dense256/seed3408：Avg-mAP `67.09`，
  mAP@0.3-0.7 `82.14/77.76/70.36/59.53/45.67`。单 cell 不能选择 resolution。
- 历史矩阵没有 exact-nine completion receipt；基础设施失败、修复、Job/hash 与 profile
  细节保留在 `experiments/spatial-zoom-s1-infrastructure.md`，禁止 resume 或拼接证据。
- R0 已冻结为历史控制，不再是 Native-Crop 的必要门槛，也不能支持 crop GO/KILL。

## 项目方向

任务是离线 TAD 的高效时序计算，不是流式/因果 Online TAD。最终目标是任务感知
动态计算分配：在昂贵 backbone 前或内部去除时序冗余，同时保护 mAP@0.6/0.7，
并以 decode、预处理、H2D、probe、selector、backbone、head、后处理、显存和能耗
构成的真实总成本证明收益。

## 当前方法候选

DUCA 当前形态：全窗口低成本 trainable C3/official-ASFormer coarse probe 产生
`p_action` 与隐藏特征；selector 读取隐藏特征、`delta_p_action`、绝对变化、不确定性
和学习特征，以 transition/boundary/utility-first 评分；fixed-K structured policy 在
预算内产生 original-time selected positions；官方 OpenTAD/AdaTAD-derived detector
在 selected axis 上运行；训练期通过 structured zero-forward bridge 接受 detector
梯度。actionness 只是二分类校准和小权重辅助，不负责最终覆盖决策。

准确协议是 `offline_full_window + runtime_generated + cache_free + jointly_trained`。
类名中的 Online 是历史命名，不能用于声称 Online TAD。

## 当前裁决

- `70aa069`：冻结为待裁决的完整 DUCA baseline，不能直接作为论文最终方法。
- `a5e1774`：最新审计提交，加入 full-stack cost profiler 与 AdaTAD 诚实契约；没有
  对应 full-train，只有成本 smoke。
- 正式 fixed-384 Job `1154971` 使用 `70aa069`，不是 `a5e1774`。
- dynamic MUST 暂不作为主贡献；X3D/SlowFast 只作为 frozen-prior diagnostic。
- ChronoTransport `92029ea` formal P3 science gate 已失败，Stage C/P5 未解锁；该路线暂停，不是比 DUCA 更成熟的替代方案。
- 暂不实现 physical-grid；selected-axis 几何风险保持公开，等待决定性对照。
- 不再增加 selector head/loss，先完成强基线、成本和 hard/soft 对齐。

## 已吸收的关键经验

1. C3 粗动作性有价值，但 actionness top-k 容易选动作内部，边界覆盖不足。
2. PAction learned 曾优于复杂 GAS-VT，说明复杂约束可能过拟合覆盖并趋向均匀。
3. GAS-VT 存在 train/apply 特征不一致、非真实 sequential value transport 和硬 gap
   repair 掩盖学习的问题，因此降级为 Stage1/工程 baseline。
4. move25/move50 显示选择会聚集，但聚集中心可偏离 GT 边界；粗分类误差与 selector
   目标不匹配必须分别诊断，不能继续靠膨胀补救。
5. detector teacher/GT 只可用于 train supervision，val/test 必须递归拒绝泄漏字段。
6. smoke、gradient nonzero、wrapper precheck 只证明接口，不证明 full detector utility。
7. requested K、effective K、unique K、padded detector K 和实际 backbone 输入必须分别
   记录；fixed-384 日志出现 effective budget 低于 384，尚待解释。
8. GT boundary target 只能称 `boundary_utility_proxy`，不能称 true detector utility。
9. 官方 base/head 配置一致不代表官方源码完全一致；DUCA 改变输入长度、GT 坐标和
   post-hoc true-time remap，必须表述为 official-derived detector components。

## 不得重走

- 不再把三阶段独立训练包装成论文最终联合模型。
- 不再用 RGB 均值或密集 X3D 代替主方法的低成本可学习 coarse probe。
- 不再让 actionness coverage 主导 selector；边界/转换优先。
- 不再用硬膨胀、后处理 repair 或 uniform scaffold 隐藏 selector 学习失败。
- 不再用旧 commit mAP 证明最新实现。
- 不再把 PENDING/smoke/precheck 写成正式实验。
- 不再只报 FLOPs；random-init 成本 smoke 不能成为论文数据。
- 不再在 matched uniform/random/dense 基线缺失时扩新模块。

## 决定性门槛

1. 同 commit exact-uniform、periodic、random、dense 与 DUCA fixed-K 对照。
2. one-swap finite-difference：ST gradient 与 hard replacement utility 正相关并优于
   actionness、transition 和 random。
3. same-selected-frames 几何对照，判断 selected-axis 风险是否实质伤害 high tIoU。
4. trained checkpoint 下的 full-stack p50/p95 latency、memory、energy；probe+selector
   不能吞掉 heavy-backbone 节省。
5. mAP@0.6/0.7、短动作、边界距离、max gap、聚集偏移不退化。
6. 至少第二 detector，或诚实收缩为 AdaTAD-specific。

任何核心门槛失败，不再继续调 loss 权重：按原因降级 DUCA、移除 ST bridge、改用
simpler residual selection，或转向 ChronoTransport/PhysTime/CVCR 等新假设。
