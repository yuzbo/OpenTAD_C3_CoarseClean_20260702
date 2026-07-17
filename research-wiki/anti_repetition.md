---
type: anti_repetition
updated: 2026-07-17
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
10. S1 selector must follow the official evaluator's prediction-domain policy:
    finite zero-length proposals remain zero-IoU false positives. Do not reject
    or delete them, because either action diverges from or inflates official AP.
    The in-training evaluator log is not a gate score when its GT population is
    broader than the frozen gate prediction population.
11. Post-processing repair code must not reconstruct a historical bound config
    against its own current `ROOT` or commit. It must derive the original clean
    repository from the recorded audited config path, verify its exact Git HEAD
    and config matrix, and validate the original precheck there. Never copy or
    rewrite bound configs merely to make a repair snapshot accept them.
12. A clean repair clone does not own the training snapshot's ignored `data/`
    mount. Repair entrypoints that instantiate the official dataset must run
    with the historical clean training snapshot as the working directory while
    importing the certificate-bound repair code explicitly. Do not add hidden
    symlinks or Git excludes to make relative dataset paths appear available.
13. Do not certify a long formal power profile from a short synthetic cadence
    test while the sampler remains a Python thread inside the detector process.
    Job `1167538` proved that native NVML can still suffer a `2413.519` ms
    observed gap under the full memory-heavy inference/NMS path despite passing
    a ten-second Gate. Keep the 20 ms target and 100 ms limit unchanged; require
    an independently scheduled UUID-bound sampler process, preserve the raw
    failure trace, and pass a representative long-duration no-open stress Gate
    before any replacement matrix.
14. A locally passing sidecar implementation is not a passed Gate. Do not
    submit a replacement matrix until a clean remote snapshot completes the
    full 792-exposure dense256/seed3408 path with the frozen 20/100 ms cadence,
    4+1 CPU isolation, UUID parity, unchanged test-evidence hash, and no formal
    profile publication. Submit exactly one serial matrix only after that Gate.
15. Do not require a separately scheduled Gate and matrix to receive the same
    physical GPU UUID. The Gate must bind its own actual UUID; the matrix must
    match the Gate's stable hardware/software class, bind its own actual UUID,
    and keep all nine cells in one allocation on one physical GPU.
16. Do not pair a sidecar report with an independently selected trace. Every
    consumer must use the shared attempt validator and recompute trace hash and
    cadence. Partial salvage may publish a missing hash-matching counterpart
    but must never overwrite an existing report or trace.
17. A matrix namespace is single-use. The persistent atomic matrix lock and
    start/completion receipts are evidence, not temporary scheduling files.
    Never remove a failed lock to resume or duplicate the same campaign.
18. Do not lower the formal 90,000 MiB memory floor to fit N16R4's 55 GB
    one-GPU outer-job default, and do not override `CUDA_VISIBLE_DEVICES`.
    Reserve the site's two-GPU outer resource only when required to obtain
    sufficient memory, then run the entire Gate or frozen matrix in one exact
    Slurm step with one GPU, five CPUs, and 96,000 MiB. Record the step-scoped
    GPU and finite cgroup limit. The idle outer GPU is scheduling overhead,
    not model compute or measured cost, and must be disclosed.

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
