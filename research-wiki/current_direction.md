# 当前唯一方向与最终目标

2026-07-13 部署门槛更新：G1a 当前仅为 `tested`。独立 Max 审查第二轮的 4 个 P1 已修复，相关远端回归 `240 passed`；第三轮未达到零 P0/P1 前不得部署 real gate，gate 未通过前不得启动 matched pilot。

更新时间：2026-07-13

## 1. 最终研究目标

最终论文不再以“学习一个更复杂的选帧器”为中心，也不再以 PAction、GAS-VT、DUCA 或动态 K 的 mAP 调参为最终贡献。

最终目标是一个**独立的离线时序动作检测器**：输入任意数量、任意间隔、可能有连续空洞的原始视频观测及其真实时间戳和可审计支持区间，模型直接在物理时间轴上完成动作分类、起点定位和终点定位，并在不规则采样、缺帧和不同帧率下保持高 tIoU 定位稳定性。

该目标对应 `idea:phystime-tad-2`，而不是 selector plugin。selector 可以成为未来输入来源之一，但不属于检测器定义。

## 2. 当前唯一执行主线

`idea:phystime-adatad-1` 已完成稳定 full run 并冻结为负基线。当前执行阶段是它之后的 **HOLD AND REBUILD**：先用最小而严格的 capacity-matched control 回答一个问题：

> 在完全相同的不规则原始帧观测、官方 AdaTAD/VideoMAE-S backbone、检测容量、跨 query 上下文、候选拓扑、assignment 和训练更新下，只改变 selected-coordinate 与 physical-coordinate，结果是否仍有稳定差异？

### Phase 0：PhysTime 1.0 真实门控（已完成）

- 从 THUMOS14 原始 RGB 视频出发。
- 训练时先沿用标准 AdaTAD 的 GT-aware `random_trunc` 接受逻辑 768 位置窗口；随后仅在该已接受窗口内部，用相同、确定性、无学习、无 GT 的 `random_fixed_subsample` 选择 K=384。验证/测试的滑窗和窗内子采样均不使用 GT。
- `DecordDecode` 只解码这 384 个位置，未选帧不进入 VideoMAE。
- 三个头必须获得逐样本完全相同的 selected-index checksum。
- 真实样本完成 CUDA decode、forward、`losses["cost"].backward()` 和 inference。
- 记录 adapter、projection、classification、regression、endpoint 梯度及真实显存/延时。

### Phase 1：K=384 首版三头比较（已完成，负结果）

1. **Selected-axis AdaTAD**：原始 ActionFormerHead，把不规则观测压成 0..K-1；仅作为错误几何基线。
2. **Physical-grid ActionFormer AdaTAD**：复用已有 original-position assignment；是最强低改动基线。
3. **PhysTime-AdaTAD**：官方 VideoMAE-S adapter + `PhysTimeMeasureProjection` + `PhysTimeHead`，GT、query、回归、NMS 和输出都在秒坐标。
4. **Dense AdaTAD 768**：只作精度与计算参考，不属于三头公平比较。

三个稀疏系统共享数据、采样索引、backbone、预训练、空间增强、训练周期、优化器、seed、NMS 和评测。完成后的审计发现，PhysTime 还同时改变了 temporal projection、跨 query 上下文和可训练容量，因此 Phase 1 不能作为纯坐标表示隔离。

### Phase 1.5：P0 重建（当前唯一执行阶段）

1. 冻结三个 `3ac93a1` 正式配置、checkpoint 与结果，不在旧 PhysTime 1.0 上继续调参。
2. 删除主路线中原生 tubelet feature 被插值后再绑定 raw-frame support 的语义捷径，建立 native tubelet multi-atom provenance gate。
3. 将 `K=384` raw observations、`J=192` native tubelet tokens、基础候选网格 `Q0` 与多尺度总候选 `QΣ` 分开登记；不得把 `J192 -> Q0=384` lift 混入所谓 coordinate-only gate。
4. 先做 `Q0=J=192`、官方六层金字塔总候选 `QΣ=378` 的 matched temporal-metric control；selected/physical 两侧使用相同容量、上下文、候选、assignment 和更新。若需要恢复 `Q0=384`，必须先给两侧加入完全相同的中性 query lift，并单独审计对应的 `QΣ=756`。
5. 只在 matched temporal-metric control 通过后，引入有显式 mass base path、bounded correction 和 physical query encoder 的 `idea:sm-ptaf`。
6. `SM-PTAF` 当前状态仅为 `designed`；外部回复中的公式与代码片段不是实现证据，也不能把 tubelet 的 multi-atom anchor 直接表述为可加 feature measure。

当前 G1a native-J192 matched control 已达到 `tested`：远端新旧回归 `142 passed`，并修复了物理中心污染候选 mask、test evaluator 数据集错配、弱数据指纹、不可重算 artifact 与 VideoMAE 尾部 padding 泄漏。正式 dataset 消费 411 个 THUMOS14 视频；test 根目录另有 2 个未引用视频，已作为 inventory 显式登记。gate `1161304` 因旧审计范围失败，`1161353` 因标量 state byte-view 失败；`1161378` 在 selected-axis 首个真实样本因旧 gate 逐样本强制回归参数梯度非零而 fail-closed，三轮依赖 pilot 均未启动。该现象只与 ReLU dead zone 一致，旧 artifact 不足以证明根因。v3 gate 已改用正式 batch=2 DataLoader、warmup scheduler、EMA 和生产更新顺序，并记录 pre-ReLU/assignment/梯度/LR/optimizer state、trainable-only hash 与参数 delta；首轮独立审查的 4 个 P1/3 个 P2 已修复，正在复审。复审和新 clean gate 前不得部署 pilot。原始 AdaTAD interpolation 不被永久禁止；它只能在 G1b 作为双臂共享、单独归因的 query-grid lift，不能重新解释为 K 个观测。

### Phase 2：结果门控后扩展

在 Phase 1.5 的机制 gate 与单 seed survivor 出现前，以下扩展继续锁定：

- K=192/384/768；
- uniform、random、bursty、contiguous-gap；
- 多 seed；
- 第二数据集；
- 与 timestamp embedding、linear interpolation、mTAN-like projection、FrameDrop/TRC、TE-TAD、LiquidTAD 的匹配比较。

## 3. 坐标原则

- 网络内部的规范坐标是**绝对视频秒数**。
- GT 不映射到 selected-rank 轴。
- sparse gap 保持为真实缺失质量，不能用 Voronoi 或邻点支持区间偷偷填满。
- 输出可以为展示或帧级导出使用 `round(time_sec * fps)` 无损映射回原视频帧号。
- 禁止映射回“第几个被选中的帧”；那会把不规则时间再次压扁。

## 4. 当前实现事实

已实现并验证到 full-run：

- `PhysTime-TAD 2.0` 的物理时间几何、support-integrated measure projection、PhysTimeHead、registered detector、feature-token transforms 和 focused gates。
- feature-token 路线的软件契约可作为算子测试资产。
- PhysTime-AdaTAD 1.0 的 raw-frame 秒几何、三份 K384 配置、same-index/同增强 validator、真实 AMP gate、两 epoch stability gate 和三头 full run 均已完成。
- 最终稳定实现为 `3ac93a1`；最佳 checkpoint 的只读重放逐项复现正式结果。
- PhysTime 1.0 未胜 selected-axis 或 physical-grid，也未达到 dense anchor；当前实现结论为负。
- 性能诊断已经排除训练崩溃、evaluator、重复坐标换算和缺失 test windows，并确认容量/上下文混杂、absolute-second query 主导、粗层 attention 坍缩、候选密度与短动作监督不足。
- 2026-07-13 Pro 审查给出 `HOLD AND REBUILD`，进一步确认 native tubelet feature-support provenance、候选/assignment 同构和 query-mask 语义是 P0；推荐 `SM-PTAF` 作为 designed candidate，但尚无实现或实验。
- 同日独立核验认同停止 1.0 和 P0 重建，但不接受“SM-PTAF 已是唯一最终模型”：tubelet 内两帧已被非线性融合，multi-atom 只能先作为 set-valued anchor；J192 到 Q384 也是必须单独归因的新算子。
- G1a 已实现 K/J/Q 分离、canonical FPS/窗口秒域、逐层严格 padding isolation、结构性 lineage、原生 J192 official ActionFormer 路径、生产 engine 三步真实 gate、全量 timebase 审计和可重算 6 epoch artifact 合同；远端回归 `142 passed`。v3 gate 正在第二轮独立复审，尚无通过的正式 gate 或 mAP。

尚未形成的论文证据：

- capacity/context/candidate-matched 的 physical-time 因果对照；
- 修复后的一因素消融与多 seed；
- Phase 2 robustness/multi-seed/cross-dataset。

因此当前状态必须写成：**PhysTime-AdaTAD 1.0 已完成稳定 full run，但当前实现为负结果且比较存在 feature provenance、架构/容量、候选与 assignment 混杂；冻结为负基线。下一版先完成 coordinate-only control 和 native provenance gate，SM-PTAF 仍为 designed candidate，不得直接扩展论文主表。**

## 5. 明确非目标

- 不是在线/streaming/causal TAD。
- 不学习 selector，不引入 actionness、teacher、ledger 或 dynamic budget。
- 不把 I3D feature archive 当作 raw-video 端到端证据。
- 不把“continuous time”本身当作新颖性；核心是显式支持区间上的 measure operator 与物理时间检测闭环。
- 不在 primary comparison 中加入 paired-view consistency，避免监督量不公平。
- 不在 K=384 主比较完成前扩展新 idea。

## 6. 主张门槛

- 若只胜 selected-axis 而不胜 physical-grid baseline：只能说明 original-time geometry 有价值，不能证明完整 PhysTime head 必要。
- 若 K=384 明显落后 dense reference 且没有更强 accuracy-cost Pareto：不能作为论文主方法。
- 若 timestamp embedding、linear interpolation 或 mTAN-like baseline 在置信区间内持平：停止强调完整 PhysTime operator。
- 必须优先看 mAP@0.6、mAP@0.7、boundary error、短动作和最差采样模式，不只看 Avg-mAP。
