---
type: experiment
node_id: exp:phystime-adatad-k384
title: "PhysTime-AdaTAD matched raw-video K384 head comparison"
idea: idea:phystime-adatad-1
status: experiment_running
verdict: pending
confidence: low
metrics: "No result yet."
provenance: "docs/superpowers/specs/2026-07-11-phystime-adatad-1-design.md"
added: 2026-07-11T00:00:00+08:00
---

# PhysTime-AdaTAD K384

## Primary experiment

同一 THUMOS raw-video sample、同一 deterministic GT-independent K384 indices、同一 official VideoMAE-S adapter：

1. selected-axis ActionFormerHead；
2. physical-grid ActionFormer；
3. PhysTime measure projection + PhysTimeHead。

## Gates before queue

- raw geometry/GT seconds tests；
- all-config parity + same-index checksum；
- adapter/projection/head optimizer coverage；
- one real THUMOS CUDA decode-forward-backward-inference；
- no feature archive/env；
- gate-dependent exactly three formal jobs。

## Software gate status

- implementation commit：`d31e99c` 后续 AMP 修复待提交；
- raw geometry、matched configs、validator、one-step gradient、gate contract 与 deployment contract：最新远端 focused suite `49 passed`；
- matched validator：`contract_pass=true`；
- real THUMOS CUDA gate：`1158636` passed，但为 FP32；最新 gate 已改为 AMP，待重跑；
- formal jobs：`1158637` running diagnostic；`1158638` infrastructure failed；`1158639` 暴露 AMP BCE 实现错误并已修复；
- mAP：NA。

首次部署 gate `1158528` 在进入 Python 前因计算节点非登录 shell 缺少 `module` 命令失败，三个依赖训练均未启动并取消。该事件分类为 infrastructure failure；launcher 已改为可选 module 初始化并通过回归测试，仍需在新 commit 重跑真实 gate。

第二次 gate `1158546` 已通过 matched config validator，但 submission 强制写入物理 GPU1，覆盖了 Slurm 分配的设备 mask，因而在模型构建前得到 `CUDA is not available`；三个依赖训练仍未启动并取消。launcher 已改为 Slurm 内尊重调度器 mask、仅非 Slurm 调试限制物理 GPU1，并通过专项测试。

第三次 gate `1158556` 已通过 CUDA、真实 THUMOS decode 和 same raw-frame checksum，但增强后输入 checksum 不同。根因是 `mmaction.ImgAug` 使用独立 imgaug RNG，原 `_seed_everything` 未重置它；模型仍未构建，依赖训练未启动并取消。gate 现统一重置 Python、NumPy、Torch、imgaug 与 OpenCV RNG，并新增确定性回归测试。

第四次 gate `1158576` 进一步证明分叉只发生在同一进程首个配置的 ColorJitter：ImgAug 首次构造会额外消耗 NumPy 状态。gate 现先预热增强库再重新 seed。真实逐 transform 诊断 `1158614` 已确认三头在 decode、crop、ImgAug、ColorJitter 和最终 FormatShape 后的像素 hash 全部一致；这只关闭数据确定性缺口，仍不是 detector gate 或效果证据。

第五次 gate `1158636` 已完成三头真实 raw-video decode、forward、backward 与 inference，梯度和 optimizer coverage 全部通过；但它使用 FP32，随后 formal PhysTime `1158639` 在 epoch 0 暴露 endpoint probability BCE 不兼容 AMP。现已保持积分事件概率语义，用 event-logit `BCEWithLogits` 等价改写，并把 gate 升级为 AMP。physical-grid `1158638` 是 torchrun rendezvous 基础设施失败；selected-axis `1158637` 仍仅作旧 commit 诊断。

## Current verdict

尚无 mAP 结果，不能支持任何效果 claim。状态提升为 `experiment_running` 只表示真实 gate 曾通过且 formal 作业实际启动；最新 AMP 修复 commit 的三头 matched full run 仍须重新提交，完成后由 result-to-claim 更新。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
