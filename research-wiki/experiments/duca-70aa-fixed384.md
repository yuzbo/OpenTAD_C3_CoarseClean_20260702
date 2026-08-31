---
type: experiment
node_id: exp:duca-70aa-fixed384
title: "DUCA 70aa069 fixed-384 official-derived AdaTAD full train"
idea: idea:duca-offline-full-window
verdict: completed_unmatched
confidence: high
commit: "70aa069b895322c2307ffbb13dfdef9fac0d1305"
jobs: "1154971"
updated: 2026-07-11
---

# DUCA 70aa069 fixed-384 official-derived AdaTAD full train

## Raw metrics / observations

Job `1154971` 于 2026-07-11 05:03 +08:00 正常完成 60 epoch，Slurm 状态 `COMPLETED`、退出码 `0:0`，总耗时 13:07:24。12 次验证的 Avg-mAP 为 3.84、17.03、24.05、34.02、44.94、49.12、52.06、53.84、54.66、56.32、56.89、58.39，最终 epoch 59 后评估也是最佳结果。

最终 Avg-mAP 为 **58.39**；mAP@0.3/0.4/0.5/0.6/0.7 为 **76.26 / 71.06 / 61.20 / 48.90 / 34.53**。日志未出现 Traceback、OOM、non-finite loss 或训练崩溃，峰值训练日志显存约 4275 MB。

预算日志共记录 120 个 batch-level summary，其中 35 个 `duca_effective_budget_mean < 384`；记录均值为 360.55，最低 214，低于 384 的记录均值为 303.59。requested budget 始终为 384，因此当前“fixed-384”更准确地表示上限/目标预算，而非所有样本严格执行 384。

## Interpretation

曲线持续上升且最终轮最佳，说明修复后的联合 coarse-selector-detector 路径可以稳定完成训练；相对旧提交 `7e3a508` fixed-384 的 56.26 提高 2.13 点，但提交与实现不同，只能作开发趋势，不能作方法增益。

同提交 exact-uniform/random/dense/actionness-top-k 尚未完成，因此 58.39 只能证明“系统可训练并达到中等可用精度”，不能证明 learned selection 优于简单采样，也不能证明真实总成本下降。

## Limitations

不是最新 `a5e1774`，因而没有 trained-checkpoint full-stack 成本表。有效 K 低于 384 的现象需区分短有效视频/valid mask 与 selector contract 错误，并要求所有 matched baseline 按相同 effective-K 分布比较。actionness BCE 从约 0.187 降至 0.034，detector 的 cls/reg 也明显下降；但 endpoint/context 等 selector 边界分布损失长期约为 3.19/0.639，没有显示清晰下降，因此 mAP 增益可能主要来自 detector/coarse probe 学习，不能据此断言 selector 变好。日志中 `detector_loss=0` 但 `cls_loss/reg_loss` 非零、`duca_hole_w=0` 且 max-gap hole loss 为零，也需在把 detector-gradient 与 soft-gap 写成论文贡献前核验其日志/聚合语义和实际梯度作用。

## Comparative root-cause audit

当前 `58.39` 不能直接与历史“分离训练 63 / uniform 65”归因比较。DUCA 使用 `ThumosPaddingDataset`，每个 epoch 约 99 次更新，60 epoch 共 5940 次；可审计的 PAction/lattice 分离训练使用 `ThumosSlidingDataset`，每个 epoch 218 次更新，共 13080 次。DUCA 只有其 45.4% 的 optimizer exposure，且学习率仍按 epoch 衰减。分离 lattice 的可审计最佳值为 **63.18**（78.00/73.28/66.52/55.89/42.19）；PAction 可审计最佳值为 **61.02**，最终值为 59.10。当前未找到与 70aa 同协议且成功完成的 exact-uniform 65 日志；已定位的 formal uniform-384 运行在 epoch 0 因短样本只产生 260 个位置而失败，因此 65 暂只能视为历史、未匹配锚点。

还原 loss 权重后，selector 学习信号更令人担忧：最终 actionness raw BCE 约为 `0.034/0.05=0.68`，接近随机二分类的 `log(2)=0.693`；start/end raw distribution loss 约为 `3.19/0.5=6.38`，context 与 boundary-utility proxy 也约为 `0.639/0.1=6.39`，接近在约 590 个有效时间点上均匀预测的 `log(T)`。因此日志中的加权 loss 下降不能证明 coarse probe 或 boundary selector 已学会有效峰值，当前最强直接解释是 selector 仍接近 chance-level。

代码机制还存在三个待证伪风险。第一，`official-action-seg` 路径是随机两层 Conv2d spatial stem 加精简 official ASFormer temporal core，`checkpoint_path` 为空，并非此前验证过的 MobileNet/ASFormer coarse checkpoint。第二，hard Viterbi 选择由 detached logits 决定，detector backward 走 dense soft-slot zero-forward bridge；梯度连通不等于与 hard one-swap utility 对齐。第三，official-derived ActionFormer 在 irregular selected-rank axis 上仍按等间隔卷积、FPN stride、regression range 和 center sampling 工作；GT endpoint remap 不能恢复内部 physical-time geometry，uniform 的规则网格天然更符合这些假设。

决定性诊断顺序：先做同 commit、同 loader、同 optimizer-step、同 effective-K 的 exact-uniform control；再冻结 epoch-59 DUCA positions，从头训练 detector，区分“位置质量/几何错误”和“联合训练非平稳/ST 错梯度”；随后报告 coarse AUROC/AUPRC、边界 peak recall/distance、raw unweighted losses，并做 one-swap finite-difference 与 ST gradient rank-correlation。任何一步未完成前，不通过继续调 loss 权重解释差距。

## Provenance

/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_70aa069_final_20260710_1544

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
