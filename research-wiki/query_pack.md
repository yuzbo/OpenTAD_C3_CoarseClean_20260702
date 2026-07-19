---
type: query_pack
updated: 2026-07-20
max_chars: 8000
---

# Research Query Pack

## Native-Crop S1 audit delta (2026-07-20)

- The development-only vertical slice now binds a required full Git commit,
  completely clean worktree, and byte-equal tracked `HEAD` blobs before any
  model execution.
- Its geometry evidence is not trusted by self-hash alone: the gate re-probes
  all 200 fit/gate files and checks root containment, path, size, dimensions,
  rotation, frame count, and frame rate.
- Remote focused tests pass `17/17`; the same independent max reviewer returned
  `DEPLOY` with `P0/P1/P2/P3 = 0` after three passes.
- This authorizes only clean-snapshot CUDA precheck deployment. Crop mAP,
  sufficiency, full-stack cost, GO/KILL, learned ROI, and paper claims remain
  absent.

## Native-Crop S1 实现状态（2026-07-20）

- 已实现 development-only、no-training 的真实纵向切片：
  `decoded uint8 source -> global96 letterbox + source-coordinate center128
  crop -> one shared VideoMAE-S -> fixed 384-point fusion ->
  [B,384,768] AdaTAD-derived detector contract`。
- 200 个 development 视频的可用解码源均为 `320x180`；96/112/128 crop
  均无需 padding。该事实只证明当前数据副本上的几何可行性，不证明语义充分性，
  也不能表述为原始摄像机采集分辨率。
- 真实数据人口已经按冻结 manifest 闭合为 fit 160 / gate 40。旧 `0.25`
  sliding overlap 会漏掉末尾仅有 0.7 秒动作的
  `video_validation_0000054`；新 Native-Crop 配置单独使用 `0.5`
  overlap，R0 不改动。
- focused Linux 测试和真实 decode 已通过；正式 commit-bound CUDA
  full-model gate、预训练权重逐张量核验、完整 detector-loss backward
  仍待闭合。当前不存在 crop mAP、成本结果、GO/KILL 或论文主张。

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

## Native-Crop Pro 审查吸收（2026-07-20）

- Pro 审查原始裁决为 `PROCEED_NATIVE_CROP_S1`。本项目只把它解释为授权一个
  development-only、no-training、no-official-test 的 source-native crop 垂直切片；
  不是 crop 有效性、路线 GO、learned policy 或论文贡献的证据。
- 接受的主判断：当前仓库没有 crop；旧 R0 可执行合约与新路线 split-brain；crop 必须
  在任何全图 resize 前发生；保留 768 点时间轴和 `[B,384,768]` detector 合同；先验证
  sufficiency，再训练 learned policy；必须报告 full-stack cost。
- 不直接接受：八候选库只能给出 library-conditional upper bound，失败不能杀死整个
  continuous-crop 路线；最终 masked pooling 不能撤销 ViT 内 padding-token 污染；
  `96/128/48 knots` 与数值 GO/KILL 门槛都需 geometry/statistics audit 后再冻结。
- 唯一下一步：先做 fit/gate source-geometry census，再实现无 teacher 的
  `global96 + center/local128 + shared VideoMAE-S + 384 fusion + [B,384,768]`
  垂直切片和 source-pixel/no-resize/backward/parity/no-leak/cost-schema tests。
- 原文与完整吸收记录见
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
