---
type: anti_repetition
updated: 2026-07-13
---

# 禁止重走清单

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

## ChronoTransport bounded appeal

25. 不得执行 `02199f8`；它已被 GitHub-visible Pro 复核裁决为
    `REVISE_SPEC_BEFORE_PLAN`。
26. 不得把 200 个 video IDs 直接当 200 个 windows；Gate 1--3 必须使用一视频一
    label-free hash-frozen window，Gate 4 才使用 official full-video/sliding-window
    population。
27. 不得使用 `(update+offset)%16` 与 offsets 0/4/8；必须使用冻结的 `+5*b`
    block rotation 和完整 exposure-matrix hash。
28. 不得保留 oracle-minimum 对 shuffled assignment 的 hard Gate；它近乎由定义保证。
29. Stage-C loss ownership 必须由 object-identity sets 与独立 `autograd.grad` 实现，
    不能用 total-loss backward、name substring 或 detach adapter 输出来冒充。
30. AMP retry 必须恢复除 GradScaler backoff 外的全部 forward-mutated state；匹配的是
    successful batch/LR/EMA exposure，不是两个 arms 的 overflow vector。
31. Gate 3 的 selected coverage 不是 selected-conditional conformal guarantee；不得把
    frozen-window guarantee 转移到 Gate 4 official population。
32. formal profile/replay 前必须完成 immutable registration commit；任何 repair、fallback、
    identity mismatch 或 retry violation 都是 `INVALID_IMPLEMENTATION`，不是 science FAIL。
33. 即使 Gate 1--4 全 PASS，`deploy=false`、`paper=false` 仍保持冻结。
34. r2 resolved config 必须只在 inner
    `model.backbone.backbone.chronotransport` 生效；wrapper-level overlay 或 inner legacy age
    残留都不得生成 profile/Gate artifact。
35. Gate-3 simultaneous conformal 不得展平 `window×candidate`；必须先对每个窗口的 16 个
    candidates 取 residual 最大值，再对 30 个 window maxima 取 rank 28。
36. legacy six-schedule/old-split runner、pooled Spearman、row bootstrap 和 GT-aware
    `random_trunc` 不得从 r2 formal launcher 可达。
37. 不得把 `sandbox:/mnt/data/...` 的外部 patch proposal 当作本地可用、已应用或已测试；
    其明确状态是 unavailable + `NOT_EXECUTED_BY_REVIEWER`。
38. profiler 不得用 `count=0,p50=0,p95=0` 占位通过完整性检查；formal cost 必须来自直接
    测量的 invocation-level `total_ms`，且 exact requested cost 与 executed diagnostic cost
    分开记录。
39. 项目报告的 110 tests 只证明已覆盖的 primitive/subset；不得据此声称 formal manifest、
    Stage B/C、Gate 3/4、transactional retry、registration 或 science 已闭环。
40. 第二次独立审查返回 `APPROVE_IMPLEMENTATION_FOR_REGISTRATION` 前，不得创建 I/R 或打开
    formal profile/replay/evaluation 数据。
