---
type: query_pack
updated: 2026-07-17
max_chars: 8000
---

# Research Query Pack

## Spatial Zoom S1 最新门禁状态（2026-07-17）

- 当前唯一执行线仍是离线 TAD 的空间分辨率 falsification gate；不含 DUCA、时序选帧、
  ROI、scout、crop policy 或 fusion。
- 旧矩阵依次暴露了 dense-path `fc_norm` 断图、共享存储耗尽和错误拒绝官方 finite
  zero-length proposal 三类基础设施问题；都已保留为 fail-closed 记忆，禁止 resume、
  test 或性能引用。修复后 only gate-eligible checkpoint、96 GiB 启动门槛、官方 evaluator
  parity 和完整梯度合同均已测试。
- Replacement commit `18139b9` passed CUDA gate Job `1166358`; all fresh 3x3
  Jobs `1166361-1166369` completed `0:0` with valid ten-candidate gate-only
  selections. Gate Avg-mAP by resolution/seeds is
  `160: 64.739/64.842/63.078`, `224: 65.695/63.205/63.783`, and
  `256: 65.185/63.316/64.256`. These are selection scores, not test results.
  One sealed-test certificate was issued (internal SHA `8627866a...`). The first
  frozen cell dense256/seed3408 has raw official-test Avg-mAP `67.09` and
  mAP@0.3-0.7 `82.14/77.76/70.36/59.53/45.67`; this single cell cannot select a
  resolution. Post-processing failures then exposed Slurm physical/local GPU
  identity mismatch and one official duplicate loader exposure (792 exposures,
  791 physical windows). Audited recovery preserves the existing test evidence,
  ordinal exposure IDs, exact historical code/config/checkpoint provenance and
  immutable failed campaigns. No-open Gate `1167512` passed those contracts.
  Recovery `1167516` failed closed on sparse shell-pipe power samples; matched
  test-blind diagnostics showed native UUID-bound NVML met the unchanged
  20/100 ms cadence contract. The next Gate passed, but matrix `1167538`
  failed its first cell when an in-process sampler stalled for 2413.519 ms
  under detector memory pressure. Neither failure published a formal profile,
  descriptor, later cell, or new mAP evidence.
  Static audit identified a protocol mismatch: the formal sampler was a Python
  `threading.Thread` inside the memory-heavy detector/profile process. The
  branch implements an out-of-process UUID-bound native-NVML sidecar, dedicated
  4+1 CPU partition, immutable attempts and recursive v3 recovery. Prior HOLD
  audits fixed sidecar lifecycle, pre-lock, cgroup/job/step/CUDA identity and
  matrix-binding gaps. `5bfdc36` passed `104` remote tests. Gate `1168608`
  passed 792/791 exposures (max gap `63.098` ms).
  Matrix `1168823` failed closed in its first cell: 112107 samples, max gap
  `146.048` ms, three gaps above 100 ms, and no descriptor/later cell. Local
  v4 removes sampling-loop trace I/O, binds exact trace lifecycle and all parent
  evidence, and passed `102 passed, 5 skipped`. Three-pass independent review
  ended `DEPLOY` with no P0/P1. Commit `bc9350e` then passed `107` remote tests,
  but v4 Gate `1170341` failed before profiling because physical
  `SLURM_STEP_GPUS=1` was incorrectly used against cgroup-renumbered
  `nvidia-smi` index 0; resource-only `1170342` proved the mapping. Campaign
  `6021eaba...` is immutable failed infrastructure. A local fix preserves the
  physical identity but queries the sole visible selector and cross-checks its
  UUID against `cuda:0`; independent review returned `DEPLOY` with no P0/P1.
  It remains `tested_local`: no valid Gate, matrix, Pro or GO/KILL exists.

## 当前唯一活动任务：Spatial Zoom

当前执行线是离线 TAD 的空间去冗余，不是 DUCA，也不是时序选帧。时间轴、768 点
detector grid、VideoMAE-S、AdaTAD adapter、ActionFormer projection/head 和评估协议
保持不变。先用 `Dense-160/224/256 x seeds 3407/3408/3409` 回答一个前置问题：提高
空间分辨率是否能稳定改善 mAP@0.6/0.7、短动作与边界定位，并且代价是否可被后续稀疏
ROI 计算回收。

S1 只是 falsification gate，不包含 scout、ROI、crop policy、teacher、fusion 或新 detector。
只有 S1 在严格确定性、matched protocol、paired Bayesian video-cluster bootstrap 和真实
full-stack cost 下给出 GO，才允许进入 S2 oracle ROI/crop sufficiency。S2 再通过后才设计
learned low-resolution scout、连续 ROI tube、局部高分辨率重计算和全局/局部融合。

旧 `35204f5` 3x3 任务因 CUDA 线性上采样 backward 的 nondeterminism warning 而失效并
取消；其 checkpoint 仅作诊断，不能恢复或进入正式表。当前替换实现采用 exact-2x
deterministic interpolation、strict deterministic formal entrypoints、真实 full-model
backward precheck、无 class-support rejection 的 Bayesian cluster bootstrap 和原子证据提交。
尚无有效 S1 最终 mAP、成本表、GO/KILL 或 Zoom 方法结果。

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
