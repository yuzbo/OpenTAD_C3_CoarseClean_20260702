---
id: exp:duca-five-point-budget-curve
type: experiment
status: experiment_running
updated: 2026-07-22
---

# DUCA 五点预算曲线正式实验

## 要回答的问题

在完全相同的离线 TAD 模型、训练轮次、数据划分和三种子协议下，随着重型 VideoMAE 实际处理帧数从 `384` 降到 `320/256/192/128`：

1. ActionFormer 与 TemporalMaxer 的完整官方验证集 mAP 如何变化；
2. learned R2Q3 boundary-burst 相对 exact-uniform 是否在低预算下更有优势；
3. 边界召回、端点距离、双侧微簇覆盖、动作富集和最大空洞如何变化；
4. 帧数、端到端延迟、显存与性能之间是否形成可写入论文的 Pareto 曲线。

## 预算合同

| K | 重型主干帧比例 | dense-grid max-hole G | 最大源帧间隔上界 `(G+1)*4` |
|---:|---:|---:|---:|
| 384 | 50.0% | 2 | 12 |
| 320 | 41.7% | 2 | 12 |
| 256 | 33.3% | 3 | 16 |
| 192 | 25.0% | 4 | 20 |
| 128 | 16.7% | 6 | 28 |

`K=192/128` 必须放宽 max-hole，否则会把低预算学习策略强行退化为近似均匀采样，甚至导致可行域过窄。该放宽是预算-覆盖权衡，必须与 mAP 一起公开，不能只报告节省帧数。

## 代码与门禁

- Branch: `codex/duca-boundary-burst-20260722`
- Exact commit: `a00498e15d69294f78d0abeadfb47bc456db0b0e`
- GitHub: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/a00498e15d69294f78d0abeadfb47bc456db0b0e`
- Clean snapshot: `/data/run01/sczc063/yuzibo/projects/opentad_duca_budget_a00498e_20260722`
- Remote gate: budget/R5/C3 focused `71 passed`; Python compile、bash syntax、clean detached HEAD 均通过。
- `f4b2568` 首次 Linux 门禁暴露旧运行时仍硬编码 24-cell/两预算；`a00498e` 已把预算、种子、矩阵单元数和 max-hole 改为由封存矩阵轴动态校验。该失败无 optimizer update、无 mAP，不是方法负证据。

## 已部署正式实验

旧两档矩阵继续使用 `cd68d89`，不重训：

- Root: `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_cd68d89_parallel_20260722_205506`
- R5 Job `1180340`; aggregate `1180341`。
- 24 cells = 2 backends x 2 policies x K384/K256 x 3 seeds。

新三档增量矩阵：

- Root: `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_budget_a00498e_extension_20260722_2145`
- Job `1180356`: 36 complete TAD cells = 2 backends x 2 policies x K320/K192/K128 x 3 seeds；当前 `PENDING (AssocGrpGRES)`，已合法提交，不是失败。
- Job `1180357`: `afterok:1180356`，聚合新三档 terminal epoch-59 EMA 官方结果。
- Job `1180358`: `afterok:1180341:1180357`，合并五档官方 mAP，并在 ActionFormer learned seed-3407 terminal checkpoint 上导出逐预算选帧分布。
- Jobs ledger: `${root}/jobs.incremental.tsv`
- Ledger SHA-256: `d1c32352940ff2c47926a9c95e6e571924ff0623069604d185b7c8e7e52f7bf0`

## 产物与口径

- 新矩阵聚合：`${root}/r5/final_results.json`
- 五点官方 mAP：`${root}/budget_curve_evidence/map/duca_official_budget_curve.json`
- mAP CSV/图：同目录的 raw/summary CSV 与 PNG/PDF。
- 选帧分布：`${root}/budget_curve_evidence/selection/duca_budget_selection_curve.json`
- 每档 K 还保留完整 validation JSONL、样本图、边界召回、端点距离、R2Q3 配额/双侧覆盖和 max-hole 统计。
- 只有完整 THUMOS validation、OpenTAD evaluator、tIoU 0.3--0.7、terminal epoch-59 EMA 的三种子 mAP 可进入论文主表。
- 选帧质量、成本、R0 和 bootstrap 都是机理/成本证据，不能替代最终 mAP。

## 当前状态

状态为 `experiment_running`。截至 2026-07-22 21:45，旧五个 model bundles 仍运行且错误扫描为空；新三档训练已获得有效 Job ID，因组 GPU 资源上限排队。尚无任何五点 terminal mAP，不得声称低预算性能已得到支持。

2026-07-22 22:11 更新：`1180356` 仍为合法的 `PENDING (AssocGrpGRES)`，Slurm
当前给出的估计开始时间为 `2026-07-23 21:55:29 +08:00`，该估计可随集群资源变化；
`1180357/1180358` 继续按依赖等待。旧 K384/K256 的五个模型 bundle 仍并行运行，
不得为抢占队列重复提交新三档矩阵。当前仍无 terminal official mAP。

2026-07-22 23:15 更新：`1180356` 仍为合法的 `PENDING (AssocGrpGRES)`，Slurm
估计开始时间更新为 `2026-07-23 22:47:00 +08:00`；`1180357/1180358` 继续依赖等待。
旧 K384/K256 中可见的 exact-uniform official-60 与 R4Q5-G0 P0 已到 epoch 17，
但尚无 terminal epoch-59 EMA mAP。不得用 R0 的 93--94 training-internal 诊断值填充预算曲线。
