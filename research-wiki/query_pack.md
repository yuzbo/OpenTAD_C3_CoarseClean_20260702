---
type: query_pack
updated: 2026-07-13
max_chars: 8000
---

# DUCA Query Pack

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
- ChronoTransport `92029ea` formal P3 science gate 保持失败，Stage C/P5 未解锁。一次
  `CT-P3R-3S` bounded appeal 已冻结并通过 spec-only review：commit `e4422f5`、SHA-256
  `87FA305CCAFC3A29176C3971F593489F86EDD23A4C02C1BFBDAE4144FCF34CF8`，状态仅为
  `spec_approved`。
- r2 当前只有局部实现和项目报告的 110-test subset；固定快照 `4b07020` 的外部 Pro 源码
  审计维持 `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`：七项既有 blocker 全部成立，并新增
  config overlay 错层与 Gate-3 `30×16` conformal 展平两个 P0。registration 为 `NOT_READY`；
  禁止 I/R、formal profile、Gate 1、新 Stage-B seeds、Stage C/Gate 4；没有 r2 实验事实。
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
