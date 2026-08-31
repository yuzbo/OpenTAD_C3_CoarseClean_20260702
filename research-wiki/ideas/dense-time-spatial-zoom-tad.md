---
type: idea
node_id: idea:dense-time-spatial-zoom-tad
title: "Dense-Time Spatial Zoom：保留全时间轴的空间分辨率分配"
stage: designed
outcome: s1_code_allowed_s2_and_full_model_locked
tags: ["offline-tad", "spatial-zoom", "roi", "conditional-compute", "boundary"]
added: 2026-07-13
---

# Dense-Time Spatial Zoom for Offline TAD

## Thesis

不再删除时间点，而是在完整离线时间网格上始终保留低分辨率全帧上下文，只把昂贵的
高分辨率 backbone 计算分配给少量、时间连续的 ROI tube。核心假设是：TAD 的高 tIoU
定位需要密集时间证据，但判别所需的高空间分辨率可能只集中在少量区域。

## Why it may fit better than frame selection

- 保留全部 768 个时间位置，天然避免 boundary radius-1 coverage、max-gap、selected-axis
  与 physical-time remap 等当前 DUCA 难题。
- 当前 AdaTAD raw-video 配置实际输入为 160x160；VideoMAE patch size 为 16。若局部分支
  原生处理 112x112 或 96x96 crop，单 tubelet 空间 token 可从 100 降至 49 或 36。
- 低分辨率全帧分支保留场景与多人上下文，局部分支补充细粒度动作证据，时间 detector
  仍可使用规则、稠密的物理时间轴。

## What can be borrowed

- Uni-AdaFocus 的 `global glance -> policy -> local heavy crop`、全局特征复用、时间平滑
  crop path 和 global/local/fused 辅助监督。
- AdaSpot 的低分辨率全局分支、训练免参的 task-aware saliency、时空平滑 ROI，以及避免
  弱定位监督下直接学习裁剪器不稳定的经验。
- 本项目可复用 official AdaTAD/ActionFormer detector 与现有 full-stack cost ledger；
  不应重写 detector。

## What must not be copied unchanged

- Uni-AdaFocus 的视频分类目标、动态丢帧和 sample-wise early exit 不适用于本路线的
  dense-time TAD 主张。
- 不能把每个 crop 再 resize 到与 full frame 相同的 heavy-backbone 输入尺寸后声称省算；
  这只是在放大局部细节，通常不会减少 heavy FLOPs。
- 不能假设单 ROI 足以覆盖 THUMOS 的多人、全局动作、相机运动或并发事件。
- 不能仅靠 detector loss 通过 `grid_sample` 学 crop center；弱空间监督下可能产生高方差、
  背景捷径和 ROI 抖动。
- 直接把 Uni-AdaFocus/AdaSpot 接到 AdaTAD 只够做 baseline，不构成新的论文主贡献。

## Three candidate levels

1. **Stable saliency zoom baseline**：低分辨率 feature saliency 决定 stop-grad ROI，时间
   平滑；global/local/fused 三路 TAD loss。稳定但创新性低。
2. **Boundary-risk-conditioned zoom**：ROI 中心先由稳定 saliency 给出，局部输入尺度或
   ROI 数量由低分辨率时间状态变化/边界风险分配；不丢帧。更贴合 TAD。
3. **Counterfactual ZoomTAD**：训练期 dense high-resolution EMA detector 评估离散 ROI
   tube/scale alternatives 的真实 detector regret，再把 preference 蒸馏给部署期 policy。
   对齐最强，但训练成本和多 ROI 非加性交互风险最高。

## Required falsification gates

1. **Spatial headroom**：同一 detector 下 dense 224/256 是否显著优于当前 dense 160，
   尤其是 mAP@0.7 与短动作；无增益则立即停止。
2. **ROI sufficiency**：oracle/teacher ROI 在等 heavy-token 预算下是否优于 fixed-center、
   motion/person 和低分辨率基线，并接近 dense high-resolution。
3. **Context and concurrency**：一 ROI、两 ROI 与低分辨率全局上下文的匹配消融。
4. **Temporal stability**：每帧独立 ROI 与 tubelet-level、平滑 ROI tube 的抖动和性能比较。
5. **Real total cost**：decode、high-resolution frame retention、resize、H2D、scout、crop、
   heavy backbone、fusion/head、p50/p95、显存与能耗全部计入。

## Kill rule

若 dense high-resolution 对当前 160 输入没有可靠高 tIoU headroom，或 oracle ROI 在匹配
成本下不能接近 dense high-resolution，则不实现 learnable crop policy，也不启动 full train。

## Status

`designed` 指路线仍只获准执行 S1/S2 gate protocol。S1 基础设施已实现并本地
`tested`，但正式 CUDA、训练、test 与 cost 尚未运行。S2 在 S1 GO 前锁定，DART-Zoom
完整模型在 S2 GO 与新书面规格复核前锁定。路线本身不是
`experiment_running`、`empirically_supported` 或 `paper_ready`。

## 2026-07-13 Pro review absorption

接受 reviewer 的 `HOLD`、S1/S2-first、AdaSpot novelty collision、规则时间网格和完整成本
裁决；不接受其具体 scout/J/K/尺寸/loss/teacher cadence/latency 阈值已经成为最终规格。
所谓 S2 label-free oracle 必须重命名为 privileged teacher-reference oracle，并只在冻结
gate split 上用于 headroom；official test 在路线冻结前保持封存。完整记录见
`docs/methods/2026-07-13-dense-time-spatial-zoom-pro-review-absorption.md`。

## 2026-07-13 S1 infrastructure implementation

S1 infrastructure is implemented and locally tested: matched 160/224/256
configs, resolved-config diff validation, frozen manifest/seed schema,
exact pretrained-load and positional-interpolation prechecks,
trained-checkpoint official-path full-stack cost profiling, immutable run
descriptors, and paired video-bootstrap AP analysis. Checkpoint selection
requires frozen-gate raw predictions and a hashed proof; statistical pooling
resamples training seeds and paired video clusters. Required focused
regression reports `46 passed`; S1 tests report `26 passed`.
Static geometry passed, but local real-clip execution was blocked by the known
Windows `c10.dll` initialization failure. The CUDA full-window gate and all S1
training runs remain pending.

The independent `gpt-5.6-sol`/`max` audit ended at
`PASS_BEFORE_REMOTE_TRAINING` after study-level test locking, deterministic
report reconstruction, checkpoint-writer identity, and frozen profile-order
remediation. This is not an empirical S1 result.

The idea stage remains `designed`, not empirically supported. The infrastructure
node is `exp:spatial-zoom-s1-infrastructure` at `tested`. S2, ROI code, and
DART-Zoom remain locked until an actual S1 GO result exists.

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
