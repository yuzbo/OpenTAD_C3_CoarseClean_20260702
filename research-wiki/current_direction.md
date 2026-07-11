# 当前唯一方向与最终目标

更新时间：2026-07-11

## 1. 最终研究目标

最终论文不再以“学习一个更复杂的选帧器”为中心，也不再以 PAction、GAS-VT、DUCA 或动态 K 的 mAP 调参为最终贡献。

最终目标是一个**独立的离线时序动作检测器**：输入任意数量、任意间隔、可能有连续空洞的原始视频观测及其真实时间戳和可审计支持区间，模型直接在物理时间轴上完成动作分类、起点定位和终点定位，并在不规则采样、缺帧和不同帧率下保持高 tIoU 定位稳定性。

该目标对应 `idea:phystime-tad-2`，而不是 selector plugin。selector 可以成为未来输入来源之一，但不属于检测器定义。

## 2. 当前唯一执行主线

当前先实现 `idea:phystime-adatad-1`，用最小而严格的实验回答一个问题：

> 在完全相同的不规则原始帧观测和完全相同的官方 AdaTAD/VideoMAE-S backbone 下，显式物理时间检测是否优于把不规则观测当成连续 selected-rank 序列？

### Phase 0：真实门控

- 从 THUMOS14 原始 RGB 视频出发。
- 在逻辑 768 位置窗口中，用相同、确定性、无学习、无 GT 的策略选择 K=384。
- `DecordDecode` 只解码这 384 个位置，未选帧不进入 VideoMAE。
- 三个头必须获得逐样本完全相同的 selected-index checksum。
- 真实样本完成 CUDA decode、forward、`losses["cost"].backward()` 和 inference。
- 记录 adapter、projection、classification、regression、endpoint 梯度及真实显存/延时。

### Phase 1：K=384 头部隔离比较

1. **Selected-axis AdaTAD**：原始 ActionFormerHead，把不规则观测压成 0..K-1；仅作为错误几何基线。
2. **Physical-grid ActionFormer AdaTAD**：复用已有 original-position assignment；是最强低改动基线。
3. **PhysTime-AdaTAD**：官方 VideoMAE-S adapter + `PhysTimeMeasureProjection` + `PhysTimeHead`，GT、query、回归、NMS 和输出都在秒坐标。
4. **Dense AdaTAD 768**：只作精度与计算参考，不属于三头公平比较。

三个稀疏系统只允许改变时间几何与检测头；必须共享数据、采样索引、backbone、预训练、空间增强、训练周期、优化器、seed、NMS 和评测。

### Phase 2：结果门控后扩展

只有 Phase 1 稳定且 PhysTime 至少有竞争力时，才运行：

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

已实现并验证到代码层：

- `PhysTime-TAD 2.0` 的物理时间几何、support-integrated measure projection、PhysTimeHead、registered detector、feature-token transforms 和 focused gates。
- feature-token 路线的软件契约可作为算子测试资产。
- PhysTime-AdaTAD 1.0 的完整设计规格和逐任务实现计划已经冻结。
- PhysTime-AdaTAD 1.0 的 raw-frame 秒几何、三份 matched K384 配置、same-index validator、one-step 梯度证明、真实 gate 工具和 gate-dependent 三头正式启动器已在 `549bb81` 实现；远端 focused suite 为 `45 passed`。

尚未形成真实实验或论文证据：

- real THUMOS raw-video CUDA one-step gate 尚未运行；
- 三头正式训练尚未由 gate 释放；
- 任何 PhysTime-AdaTAD mAP 结果；
- Phase 2 robustness/multi-seed/cross-dataset。

因此当前状态必须写成：**PhysTime-AdaTAD 1.0 raw-video matched pipeline 已实现并通过 focused tests；真实 CUDA gate 与 full-run 仍 pending。**

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
