---
type: experiment
node_id: exp:phystime-adatad-k384
title: "PhysTime-AdaTAD matched raw-video K384 head comparison"
idea: idea:phystime-adatad-1
status: tested
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

- implementation commit：`549bb81`；
- raw geometry、matched configs、validator、one-step gradient、gate contract 与 deployment contract：远端 `45 passed`；
- matched validator：`contract_pass=true`；
- real THUMOS CUDA gate：pending；
- formal jobs：首次依赖作业未启动并取消，新提交 pending；
- mAP：NA。

首次部署 gate `1158528` 在进入 Python 前因计算节点非登录 shell 缺少 `module` 命令失败，三个依赖训练均未启动并取消。该事件分类为 infrastructure failure；launcher 已改为可选 module 初始化并通过回归测试，仍需在新 commit 重跑真实 gate。

第二次 gate `1158546` 已通过 matched config validator，但 submission 强制写入物理 GPU1，覆盖了 Slurm 分配的设备 mask，因而在模型构建前得到 `CUDA is not available`；三个依赖训练仍未启动并取消。launcher 已改为 Slurm 内尊重调度器 mask、仅非 Slurm 调试限制物理 GPU1，并通过专项测试。

第三次 gate `1158556` 已通过 CUDA、真实 THUMOS decode 和 same raw-frame checksum，但增强后输入 checksum 不同。根因是 `mmaction.ImgAug` 使用独立 imgaug RNG，原 `_seed_everything` 未重置它；模型仍未构建，依赖训练未启动并取消。gate 现统一重置 Python、NumPy、Torch、imgaug 与 OpenCV RNG，并新增确定性回归测试。

## Current verdict

尚无真实实验结果，不能支持任何效果 claim。`tested` 仅表示软件合同通过 focused tests；真实 gate 通过后才能进入 `experiment_running`，三头完成后必须由 result-to-claim 更新。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
