---
type: anti_repetition
updated: 2026-07-16
---

# 禁止重走清单

## Spatial Zoom 当前边界

1. 不得把 S1 称为 Zoom/crop 模型。S1 只有 matched dense spatial-resolution matrix。
2. 不得在 S1 GO 前实现 learned ROI policy，或用 oracle ROI 结果倒推修改 S1 gate。
3. 不得把 DUCA、时序选帧、dynamic budget、max-gap 或 X3D/SlowFast prior 混入当前任务。
4. 不得恢复 `35204f5` 的 warning-bearing partial checkpoints 作为正式结果；替换矩阵必须从
   新 exact commit、新 precheck 和新 canonical experiment namespace 全量重跑。
5. 不得把 precheck、pilot、checkpoint 数量或中间 epoch 当作 S1 性能结果。
6. S1 的正式统计不得拒绝缺少稀有类的 bootstrap replicate；使用正权重 paired Bayesian
   video-cluster bootstrap，并保持 baseline/candidate 同 replicate 配对。
7. 成本只允许表述为同节点同 GPU 的 warm serial per-window latency 与 gross GPU energy，
   不得冒充 cold-start、whole-video p95、incremental energy 或完整系统能耗。
8. VideoMAE `return_feat_map=True` 会绕过分类出口 `fc_norm`；formal gradient gate 只能
   精确允许 `backbone.model.backbone.fc_norm.{weight,bias}` 两个参数无梯度。不得用前缀、
   正则或宽泛白名单掩盖新的断图。
9. S1 只持久化预注册的 gate-eligible checkpoints；不得保存不会参与选择的 pre-gate
   周期权重耗尽共享存储。任何存储故障后的矩阵不得 resume，必须新 commit、门禁和 namespace。

## 任务与叙事

1. **不要再称 Online TAD。** 当前方法观察完整离线窗口；`online` 仅表示 forward
   内生成且不查 ledger/cache。
2. **不要把 THUMOS14 解释成 key-event timestamp spotting。** 它监督动作区间；项目
   应表述为边界敏感的稀疏 interval detection。
3. **不要把插件泛化当作已证明。** 当前只有 AdaTAD-derived 主路径，第二 detector
   仍缺正式结果。

## 模型

4. 不再回到“粗分类器独立训练 → selector 独立训练 → detector 独立训练”作为最终
   方法；它只能是归因 baseline。
5. 不允许 `asformer_lite` 冒充官方 ASFormer。
6. actionness 必须由二分类 GT 校准，但 selector 必须以 transition/boundary/utility
   为首要目标；不能再次退化为 actionness top-k。
7. 不允许用硬膨胀、uniform scaffold、max-gap repair 把坏分数修成看似合理的网格而
   不披露 repair 数量和影响。
8. `detector_utility_target` 若来自 GT 边界，只能叫 boundary-utility proxy。
9. 不得声称“完全未修改官方 AdaTAD”；源码 wrapper、selected-axis 和 GT remap 已变。

## 训练与梯度

10. nonzero grad 只证明连通，不证明梯度方向等价于 hard frame utility。
11. loss schedule 必须按 optimizer step 推进，不能按 raw forward 次数。
12. detector backend loss 与 selector gradient bridge 必须分开：关闭 bridge 不得关闭
    detector 学习。
13. dynamic budget 不得只优化 expected K；必须记录真实执行 K 与实测成本。

## 实验

14. 不再重复排同一 X3D dense export/grid；它计算过慢且可能吞掉节省。
15. 不再用旧 commit、失败 suite、重复 job 或缺失 checkpoint 的运行填论文表格。
16. 不再把 smoke、precheck、toy wrapper、geometry-only 指标称为主实验。
17. 不再跳过 exact-uniform/random/dense 等同提交基线后继续扩新方法。
18. 不再只看 Avg-mAP；必须看 mAP@0.6/0.7、短动作和边界误差。
19. 不再只报模型 FLOPs；必须报告完整数据和系统通路的 p50/p95、显存、energy。
20. 不允许 validation/test GT、teacher、oracle、raw prediction cache 或外部隐式 JSONL
    参与主方法选择。

## 决策纪律

21. 讨论提出的 CVCR/BCFT/CoDeTAD/physical-grid/CFPA 不等于已经实现或更优。
22. 决定性实验未完成前，不宣布 DUCA 成功；同样也不宣布其必然失败。
23. 每次部署前必须记录 commit、配置、checkpoint、数据、Job ID 和 run root。
24. 新结果必须先更新 experiment/claim 节点，再改论文叙事。
