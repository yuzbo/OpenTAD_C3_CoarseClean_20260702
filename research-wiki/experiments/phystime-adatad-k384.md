---
type: experiment
node_id: exp:phystime-adatad-k384
title: "PhysTime-AdaTAD matched raw-video K384 head comparison"
idea: idea:phystime-adatad-1
status: empirically_supported
verdict: negative_for_phystime_adatad_1
confidence: high
metrics: "Final 3ac93a1 matched run completed; PhysTime 1.0 underperforms both sparse controls. Raw numbers live only in docs/evaluation/results.md."
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

- implementation/experiment commit：`0bbf0e9`；
- raw geometry、matched configs、validator、one-step gradient、gate contract、masked-attention 数值回归与 deployment contract：最新远端 focused suite `68 passed`；
- matched validator：`contract_pass=true`；
- real THUMOS CUDA AMP gate：`1158718` passed；
- formal jobs：`1158719/1158720/1158721` 均以 exit code 1 终止；
- mAP：NA。

首次部署 gate `1158528` 在进入 Python 前因计算节点非登录 shell 缺少 `module` 命令失败，三个依赖训练均未启动并取消。该事件分类为 infrastructure failure；launcher 已改为可选 module 初始化并通过回归测试，仍需在新 commit 重跑真实 gate。

第二次 gate `1158546` 已通过 matched config validator，但 submission 强制写入物理 GPU1，覆盖了 Slurm 分配的设备 mask，因而在模型构建前得到 `CUDA is not available`；三个依赖训练仍未启动并取消。launcher 已改为 Slurm 内尊重调度器 mask、仅非 Slurm 调试限制物理 GPU1，并通过专项测试。

第三次 gate `1158556` 已通过 CUDA、真实 THUMOS decode 和 same raw-frame checksum，但增强后输入 checksum 不同。根因是 `mmaction.ImgAug` 使用独立 imgaug RNG，原 `_seed_everything` 未重置它；模型仍未构建，依赖训练未启动并取消。gate 现统一重置 Python、NumPy、Torch、imgaug 与 OpenCV RNG，并新增确定性回归测试。

第四次 gate `1158576` 进一步证明分叉只发生在同一进程首个配置的 ColorJitter：ImgAug 首次构造会额外消耗 NumPy 状态。gate 现先预热增强库再重新 seed。真实逐 transform 诊断 `1158614` 已确认三头在 decode、crop、ImgAug、ColorJitter 和最终 FormatShape 后的像素 hash 全部一致；这只关闭数据确定性缺口，仍不是 detector gate 或效果证据。

第五次 gate `1158636` 已完成三头真实 raw-video decode、forward、backward 与 inference，梯度和 optimizer coverage 全部通过；但它使用 FP32，随后 formal PhysTime `1158639` 在 epoch 0 暴露 endpoint probability BCE 不兼容 AMP。现已保持积分事件概率语义，用 event-logit `BCEWithLogits` 等价改写，并把 gate 升级为 AMP。physical-grid `1158638` 是 torchrun rendezvous 基础设施失败；selected-axis `1158637` 仍仅作旧 commit 诊断。

AMP gate `1158668` 在 `bd27544` 完成并通过，但它只覆盖单个样本。formal PhysTime `1158671` 在 epoch 0 第 50 步出现分类、回归和端点 loss 全 NaN；根因是 support-measure attention 只用 covered logits 求 row max，却对全部 logits 求指数，未覆盖极大 logit 在 AMP 下产生 `inf * 0`。`0bbf0e9` 先将未覆盖 logits 置为 `-inf` 再指数，并新增极值回归测试。

修复后 AMP gate `1158718` 已通过，same raw-frame/input、optimizer coverage、adapter 与各 detector branch 梯度合同继续满足。formal jobs `1158719/1158720/1158721` 已在同 commit 完成 epoch 0 并进入 epoch 1；全部已记录 leaf loss 有限，当前没有 mAP。

后续终态推翻了上述“已越过已知 NaN 点”的局部判断。selected-axis 与 physical-grid 均训练到 epoch 41 且 loss 有限，但首次正式验证因 `evaluation.ground_truth_filename` 使用不存在的相对路径而失败。PhysTime 从 epoch 1 step 99 起分类、回归、端点和总 loss 持续全 NaN，仍继续写出 checkpoint，随后同样在 epoch 41 验证路径处退出。三项作业均无有效 mAP，所有 checkpoint 只能作故障诊断。

## Current verdict

本次 matched run 无效，不能支持任何效果 claim。重新运行前必须同时关闭两个独立缺口：把 evaluator annotation 解析绑定到已验证的真实绝对路径，并定位 PhysTime 在累计优化后产生非有限参数/激活/梯度的首个 step。新的 gate 还必须覆盖多 optimizer step 与真实 evaluator 构建，单样本 one-step 不再足够。

## 2026-07-12 final repair and redeployment

- `52b5756` fixed the runtime evaluator path and moved physical geometry, measure attention, and seconds-coordinate DIoU to FP32. Its real three-step AMP/evaluator gate `1159481` passed, but the two-epoch stability gate `1159482` correctly failed closed before formal training.
- Diagnostic commit `d91c7a9`, job `1159489`, localized the first failure to epoch 0 iter 47 on `video_validation_0000948` and `video_validation_0000987`: the forward losses were finite; only 11 elements of `rpn_head.cls_head.weight` gradient were Inf, with no NaN. This was default AMP loss-scale overflow, not a new physical-time forward NaN.
- Final commit `3ac93a1` uses AMP initial scale 1024, disables useless single-GPU FP16 DDP compression, skips clipping on recoverable scaled-Inf gradients, and fails on NaN, parameter pollution, more than 4 consecutive skips, or more than 8 skips per epoch.
- Remote regression suite: `102 passed`. Final real gate `1159491` passed three AMP optimizer steps, evaluator construction, optimizer coverage, same-frame/input checks, and finite gradients/parameters.
- Final stability gate `1159492` completed two full epochs with no AMP skips: epoch 0 end loss 1.5824; epoch 1 end loss 1.1674. `STABILITY_GATE_COMPLETE` exists.
- Formal matched jobs `1159493/1159494/1159495` completed from run root `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_adatad_3ac93a1_k384_final_20260712_023243_+0800`. Best-checkpoint replay jobs `1159819/1159820/1159821` reproduced the official metrics exactly.
- The current implementation is a negative result: PhysTime 1.0 loses to both sparse controls and the dense anchor. It is not paper-ready.
- Performance-drop diagnostics rule out evaluator and training-collapse explanations. They identify architecture/capacity confounding, absolute-time query dominance, coarse attention collapse, candidate-density mismatch, short-action supervision thinning, and target-assignment mismatch.
- Because the comparison changes more than coordinate representation, the general physical-time hypothesis remains unresolved. Raw numbers and artifact paths live only in `docs/evaluation/results.md`; the causal interpretation is in `docs/evaluation/phystime-performance-drop-diagnosis.md`.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
