---
type: wiki_index
updated: 2026-08-09
project: C3-DUCA efficient temporal acquisition for TAD
---

# C3 / DUCA Research Wiki

## Spatial Zoom / Native-Crop 当前节点

- [Spatial Zoom / Native-Crop offline TAD](ideas/spatial-zoom-offline-tad.md)
- [Spatial Zoom R0 infrastructure](experiments/spatial-zoom-s1-infrastructure.md)
- [Native-Crop S1 development vertical slice](experiments/native-crop-s1-vertical-slice.md)
- [Continuous-RoI S2 crop sufficiency](experiments/native-crop-s2-crop-sufficiency.md)
- [Native-Crop paper experiment roadmap](experiments/native-crop-paper-experiment-roadmap.md)
- [GeoRoute-AdaTAD native routing](experiments/georoute-adatad.md)
- [SCNR-TAD native-cell ROI floor sensitivity](experiments/scnr-geometry-floor-sensitivity-v1.md)
- [SCNR-TAD dynamic exact-budget ragged executor](experiments/scnr-dynamic-stage1-executor-v1.md)
- [SCNR-TAD dynamic role-calibration diagnostic](experiments/scnr-dynamic-role-calibration-diagnostic-v1.md)
- [SCNR-TAD residual-window centering probe](experiments/scnr-residual-window-centering-probe-v1.md)
- [SCNR-TAD residual-centering matched training](experiments/scnr-residual-centering-matched-training-v1.md)
- [SCNR-TAD residual-centering paired full-stack cost](experiments/scnr-residual-centering-paired-cost-v1.md)
- [SCNR-TAD M2 terminal experiment audit](experiments/scnr-dynamic-floor-m2-experiment-audit.md)

这是本项目研究记忆的单一入口。它区分讨论、代码、测试、实验和论文证据，
用于阻止路线遗忘、旧错误复发和旧提交结果冒充最新实现。

## 每次开工必读

1. [query_pack.md](query_pack.md)：8k 字符以内的当前状态压缩包。
2. [anti_repetition.md](anti_repetition.md)：不得重走的路线和常见误报。
3. [decision_history.md](decision_history.md)：为什么走到当前方向。
4. [gap_map.md](gap_map.md)：尚未关闭的科学与工程缺口。
5. [source_registry.md](source_registry.md)：所有判断的原始来源。
6. [LINT_REPORT.md](LINT_REPORT.md)：结构完整性与未覆盖来源。

## 当前最终目标

研究目标不是继续优化某个 ledger heuristic，也不是 Online TAD。目标是：

> 面向离线 TAD，在昂贵 backbone 前或内部进行任务感知的时序去冗余；以低成本
> 粗粒度动作/状态表征提供候选证据，以状态转换、动作边界和下游检测效用决定
> 计算分配，在真实总成本下降的同时保护高 tIoU 定位性能。

当前 DUCA 是这一目标的主要可执行候选之一，但尚未取得主论文方法资格。
`70aa069` 被冻结为待裁决完整 baseline；`a5e1774` 增加成本与官方后端审计。

## 节点索引

### Ideas

- [C3 粗分类动作性](ideas/c3-coarse-actionness.md)
- [PAction 严格 ledger](ideas/paction-strict-ledger.md)
- [GAS-VT value transport](ideas/gas-vt.md)
- [lattice / move25 / move50](ideas/lattice-boundary-replacement.md)
- [detector-aware teacher utility](ideas/detector-aware-teacher.md)
- [TrueTime 与 detector-gradient joint training](ideas/truetime-joint.md)
- [DUCA 离线全窗口插件](ideas/duca-offline-full-window.md)
- [transition/boundary-first selector](ideas/transition-boundary-first.md)
- [fixed budget 与 max-gap](ideas/fixed-budget-max-gap.md)
- [DUCA-MUST dynamic budget](ideas/duca-must.md)
- [X3D/SlowFast frozen prior](ideas/trainfree-video-prior.md)
- [physical-grid ActionFormer](ideas/physical-grid.md)
- [CFPA causal streaming policy](ideas/cfpa-streaming.md)
- [CVCR / BCFT / CoDeTAD](ideas/cvcr-bcft-codetad.md)
- [ChronoTransport](ideas/chronotransport.md)
- [PhysTime](ideas/phystime.md)
- [Geometry-Residual-Depth Routing for offline TAD](ideas/geo-route-adatad.md)
- [Structured Complementary Native Routing](ideas/structured-complementary-native-routing.md)

### Reviewed Literature

- [FlashVID: training-free VLLM token merging](papers/fan2026-flashvid.md)
- [Uni-AdaFocus: spatial-temporal dynamic computation](papers/wang2024_uniadafocus_spatialtemporal_dynamic.md)

### Experiments

- [Stage1 PAction/GAS-VT](experiments/stage1-paction-gasvt.md)
- [move25/move50 geometry](experiments/lattice-move-diagnostics.md)
- [7e3a508 budget suite](experiments/duca-7e3-budget-suite.md)
- [X3D/SlowFast diagnostic](experiments/trainfree-video-prior.md)
- [70aa069 fixed-384](experiments/duca-70aa-fixed384.md)
- [a5e1774 full-stack cost](experiments/duca-a5e-cost.md)
- [parallel routes](experiments/parallel-routes.md)
- [ChronoTransport formal Stage-B 负 gate](experiments/chronotransport-formal-stage-b.md)
- [Continuous-RoI S2 crop sufficiency](experiments/native-crop-s2-crop-sufficiency.md)
- [Native-Crop paper experiment roadmap](experiments/native-crop-paper-experiment-roadmap.md)
- [SCNR-TAD native-cell ROI floor sensitivity](experiments/scnr-geometry-floor-sensitivity-v1.md)
- [SCNR-TAD dynamic exact-budget ragged executor](experiments/scnr-dynamic-stage1-executor-v1.md)
- [SCNR-TAD residual-window centering probe](experiments/scnr-residual-window-centering-probe-v1.md)
- [SCNR-TAD residual-centering matched training](experiments/scnr-residual-centering-matched-training-v1.md)

### Claims

- [claims/project-claims.md](claims/project-claims.md)

## 状态词典

- `discussed`：只在讨论中提出。
- `designed`：存在明确设计或 prompt。
- `implemented`：代码存在。
- `tested`：focused tests 或 smoke 通过。
- `experiment_running`：正式数据实验正在运行。
- `empirically_supported`：有匹配、可审计实验支持。
- `paper_ready`：主张、基线、成本、泛化和统计均闭环。

任何代理不得跨级推断。例如 `tested` 不等于 `empirically_supported`。
