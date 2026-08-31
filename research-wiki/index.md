---
type: wiki_index
updated: 2026-08-29
project: C3-DUCA efficient temporal acquisition for TAD
---

# C3 / DUCA Research Wiki

这是本项目研究记忆的单一入口。它分别记录科学讨论、方法设计、代码实现、局部测试、正式实验和论文证据，避免路线遗忘、旧错误复发或旧提交结果被误写成当前实现的证据。

## 每次开工必读

1. [query_pack.md](query_pack.md)：8k 字符以内的当前状态压缩包。
2. [anti_repetition.md](anti_repetition.md)：不得重走的路线和常见误报。
3. [decision_history.md](decision_history.md)：为什么走到当前方向。
4. [gap_map.md](gap_map.md)：尚未关闭的科学与工程缺口。
5. [source_registry.md](source_registry.md)：所有判断的原始来源。
6. [LINT_REPORT.md](LINT_REPORT.md)：结构完整性与未覆盖来源。
7. [duca_model_version_registry.md](duca_model_version_registry.md)：所有 DUCA
   树、模型版本、复用来源和唯一主线，新增模型前必须先查。
8. [duca_final_model_contract.md](duca_final_model_contract.md)：历史 R0--R5 合同与负证据；
   不再是当前主线权威。
9. [DUCA 物理连续片段与动态预算最终合同（2026-08-21）](DUCA_PHYSICAL_CLIPLET_DYNAMIC_BUDGET_FINAL_CONTRACT-2026-08-21.md)：
   已完成但未获得支持的连续片段采样路线。正式单种子实验显示定位性能明显下降，因此不再是当前主线；合同保留用于解释该负结果及其控制条件。
   [第一版冻结合同](DUCA_PHYSICAL_CLIPLET_DYNAMIC_BUDGET_FROZEN_CONTRACT-2026-08-21.md)
   仅保留作取代记录。
10. [DUCA 全量记忆与资源审计（2026-08-17）](DUCA_WIKI_MEMORY_AUDIT-2026-08-17.md)：
   间接 actionness/boundary 路线、历史证据边界、共享 official baseline 与资源状态。
11. [DUCA 远端数据资源地图（2026-08-17）](REMOTE_DATA_RESOURCE_MAP-2026-08-17.md)：
   N16R4 登录、THUMOS14 canonical binding、共享 AdaTAD baseline 边界、checkpoint 与角色 PRE_RUN 边界。
12. [DUCA Query-Bridge Pro 审查吸收与本地核验（2026-08-20）](DUCA_PRO_REVIEW_ABSORPTION-2026-08-20.md)：
    已核验诊断、未采纳的数值提案与新的固定 K 优先证伪顺序。
13. [DUCA 第二份外部审查的对照、核验与吸收（2026-08-20）](DUCA_SECOND_REVIEW_COMPARISON_AND_ABSORPTION-2026-08-20.md)：
    两份审查的一致内核、实质分歧、代码核验与未冻结设计选择。
14. [DUCA 对抗性 RiskClip 审查：记录与核验（2026-08-21）](DUCA_ADVERSARIAL_RISKCLIP_REVIEW_VERIFICATION-2026-08-21.md)：
    历史 65 代码核验、GitHub 可见性边界、条件接受的语义动态预算和未采纳的具体参数。
15. [DUCA BSC-DK 独立审查：对照与核验（2026-08-21）](DUCA_BSC_DK_REVIEW_COMPARISON_AND_VERIFICATION-2026-08-21.md)：
    两份外部审查的一致内核、预算与训练合同冲突、以及未冻结的关键决定。
16. [DUCA SCOPE-DK 第三份外部审查：对照与核验（2026-08-21）](DUCA_SCOPE_DK_REVIEW_COMPARISON_AND_VERIFICATION-2026-08-21.md)：
    三份审查的共同内核、SCOPE 的新运行时合同建议、与 RiskClip/BSC-DK 的实质冲突及采纳边界。
17. [DUCA IPEC-K 第四份外部审查：对照与核验（2026-08-21）](DUCA_IPEC_K_FOURTH_REVIEW_COMPARISON_AND_VERIFICATION-2026-08-21.md)：
    四份审查的共同内核、IPEC 的端点覆盖建议、与前三份的合同冲突和未闭合的归因控制。
18. [DUCA PJST 首次重型时间混合审查吸收（2026-08-25）](../docs/methods/2026-08-25-b2ccfcca-duca-pjst-pro-review-absorption.md)：
   用户提供终稿的完整证据边界、代码核验、统计修正、因果口径与 derivative-only 最小优化建议；
   [逐字原文归档](../docs/methods/reviews/2026-08-25-b2ccfcca-duca-pjst-pro-response-user-supplied-raw.md)。
19. [SparseHead 唯一路线合并](experiments/sparsehead-route-consolidation-20260728.md)：
   当前唯一可写稀疏头代码面、旧仓封存边界与可继承证据。
20. [DUCA GitHub 深度审查入口（2026-08-31）](GITHUB_REVIEW_INDEX-2026-08-31.md)：
   完整 Wiki、Gemini 全量预审、关键公开代码版本与逐版本审查边界。

## 当前最终目标

研究目标不是继续优化某个 ledger heuristic，也不是 Online TAD。目标是：

> 面向离线 TAD，在昂贵 backbone 前或内部进行任务感知的时序去冗余；以低成本
> 粗粒度动作/状态表征提供候选证据，以状态转换、动作边界和下游检测效用决定
> 计算分配，在真实总成本下降的同时保护高 tIoU 定位性能。

DUCA 的长期论文问题是：低成本模型先预测逐时刻动作性与边界重要性，确定性规则据此选择非均匀帧，并进一步为不同视频或窗口分配动态预算。固定 K 只用于机制归因、公平对照和回退，不是最终论文主张。

当前最可靠的历史实现是 H65 间接非均匀逐帧选择。其 30+60 训练日程的单种子平均检测精度约为 `65.13`；20+40 压缩日程和两条学习率调整均未恢复该性能，因此压缩训练研究已停止，但 H65 选择机制并未被这些结果否定。当前受控子问题固定 `K=384`，只比较相同 H65 选择结果在重型视频编码器中的物理时间表示。PJST-D1 的 ON 相对 OFF 平均检测精度点估计为 `-0.472` 个百分点；由于预登记的整视频配对自助法在抽样前因路径错误退出，目前没有置信区间，也不能把该点估计写成总体负效应结论。

[DUCA-RIME](ideas/duca-rime.md) 与 [dynamic-K/RIME Oracle](experiments/duca-dynamic-k-rime-oracle.md) 仍是尚未实现和训练的动态预算候选，不得写成已有论文证据。旧 Gaussian-mass、MUST 和物理连续片段路线只保留为历史或负结果。ChronoTransport、Spatial-Zoom 与 SparseHead 是独立路线，不得相互改写结果。

## 节点索引

### Ideas

- [C3 粗分类动作性](ideas/c3-coarse-actionness.md)
- [PAction 严格 ledger](ideas/paction-strict-ledger.md)
- [GAS-VT value transport](ideas/gas-vt.md)
- [lattice / move25 / move50](ideas/lattice-boundary-replacement.md)
- [detector-aware teacher utility](ideas/detector-aware-teacher.md)
- [TrueTime 与 detector-gradient joint training](ideas/truetime-joint.md)
- [DUCA 离线全窗口插件](ideas/duca-offline-full-window.md)
- [DUCA-FSU 可行硬交换效用蒸馏](ideas/duca-fsu.md)
- [DUCA-CellCF coverage-preserving local deformation](ideas/duca-cellcf.md)
- [DUCA-CARA coverage-anchored residual allocation](ideas/duca-cara.md)
- [DUCA Protected-E2E physical-DAG selector](ideas/duca-protected-e2e.md)
- [transition/boundary-first selector](ideas/transition-boundary-first.md)
- [fixed budget 与 max-gap](ideas/fixed-budget-max-gap.md)
- [DUCA-MUST dynamic budget](ideas/duca-must.md)
- [X3D/SlowFast frozen prior](ideas/trainfree-video-prior.md)
- [physical-grid ActionFormer](ideas/physical-grid.md)
- [CFPA causal streaming policy](ideas/cfpa-streaming.md)
- [CVCR / BCFT / CoDeTAD](ideas/cvcr-bcft-codetad.md)
- [ChronoTransport](ideas/chronotransport.md)
- [PhysTime](ideas/phystime.md)
- [Boundary-Adaptive Temporal Multigrid](ideas/boundary-adaptive-temporal-multigrid.md)
- [Counterfactual Value-of-Computation](ideas/counterfactual-value-of-computation.md)
- [Spectral Innovation Operator](ideas/spectral-innovation-operator.md)
- [Dense-Time Spatial Zoom for TAD](ideas/dense-time-spatial-zoom-tad.md)
- [DUCA-RIME risk-calibrated marginal evidence allocation](ideas/duca-rime.md)
- [DUCA 向时序动作分割迁移的可行性与最小研究合同](ideas/duca-tas-migration.md)
- [DUCA 全量记忆与资源审计（2026-08-17）](DUCA_WIKI_MEMORY_AUDIT-2026-08-17.md)

### Experiments

- [Stage1 PAction/GAS-VT](experiments/stage1-paction-gasvt.md)
- [move25/move50 geometry](experiments/lattice-move-diagnostics.md)
- [7e3a508 budget suite](experiments/duca-7e3-budget-suite.md)
- [X3D/SlowFast diagnostic](experiments/trainfree-video-prior.md)
- [70aa069 fixed-384](experiments/duca-70aa-fixed384.md)
- [a5e1774 full-stack cost](experiments/duca-a5e-cost.md)
- [parallel routes](experiments/parallel-routes.md)
- [PhysTime-AdaTAD K=384 matched three-head run](experiments/phystime-adatad-k384-matched.md)
- [ChronoTransport formal Stage-B gate](experiments/chronotransport-formal-stage-b.md)
- [DUCA legacy epoch-89 selection quality](experiments/duca-selection-quality-epoch89.md)
- [DUCA Oracle gap/reachability audit](experiments/duca-oracle-gap-reachability-audit.md)
- [DUCA allocation-family feasible-set ceiling](experiments/duca-allocation-feasible-set-ceiling.md)
- [DUCA Protected-E2E gate diagnostics](experiments/duca-protected-e2e-gates.md)
- [DUCA selected-axis optimization official-60](experiments/duca-selected-axis-optimization-official60.md)
- [DUCA two-stage curriculum diagnostic](experiments/duca-two-stage-curriculum-official60.md)
- [DUCA global-curriculum matched official-60](experiments/duca-global-curriculum-official60.md)
- [DUCA Oracle-calibrated boundary-burst plan](experiments/duca-oracle-calibrated-boundary-burst.md)
- [DUCA EU-CRR conditional fusion diagnostic](experiments/duca-eucrr-fusion-diagnostic.md)
- [DUCA R0--R5 9f97f2c shared-bootstrap execution](experiments/duca-r0-r5-9f97f2c-shared-bootstrap.md)
- [DUCA sparse-probe interpolation and coarse-backend ablation](experiments/duca-sparse-probe-and-coarse-backend-ablation.md)
- [2026-07-23 DUCA nightly implementation and deployment ledger](experiments/duca-20260723-nightly-implementation-and-deployment.md)
- [Spatial Zoom S1 infrastructure verification](experiments/spatial-zoom-s1-infrastructure.md)
- [DUCA dynamic-K / RIME Oracle and causal gates](experiments/duca-dynamic-k-rime-oracle.md)
- [SparseHead 唯一路线合并与旧仓封存](experiments/sparsehead-route-consolidation-20260728.md)
- [PhysTime G1 native-J192 matched 20-epoch](experiments/phystime-g1-matched-medium20.md)
- [SDPQ 20-epoch trainability record](experiments/phystime-g1b-sdpq-medium20.md)
- [PhysTime G1 native-J192 matched full60](experiments/phystime-g1-matched-full60.md)
- [PhysTime P0 full-precision NMS replay](experiments/phystime-p0-fullprecision-nms-replay.md)
- [PhysTime frozen decode-cross replay](experiments/phystime-frozen-decode-cross-replay.md)
- [DUCA 窗口级动态原生 tubelet 预算](experiments/duca-dynamic-native-tubelet-window-budget.md)
- [DUCA Coverage-v1 固定预算时间覆盖归因](experiments/duca-coverage-v1.md)
- [DUCA-Marginal-v1 窗口级加性边际效用终态负结果](experiments/duca-marginal-v1.md)
- [DUCA 整视频一致预算的跨视频转移 oracle](experiments/duca-whole-video-consistent-budget-v1.md)

### Papers

- [Mixture-of-Depths](papers/raposo2024-mixture-of-depths.md)
- [Eventful Transformers](papers/dutson2023-eventful-transformers.md)
- [Progressive Block Drop for TAD](papers/chen2025-progressive-block-drop.md)
- [ResidualViT](papers/soldan2025-residualvit.md)
- [Adaptive Temporal Refinement](papers/shihab2025-adaptive-temporal-refinement.md)
- [SCOPE](papers/cui2026-scope.md)
- [Conformal Thinking](papers/wang2026-conformal-thinking.md)
- [Uni-AdaFocus](papers/wang2024-uni-adafocus.md)
- [AdaSpot](papers/xarles2026-adaspot.md)
- [ChronoTransport formal Stage-B 负 gate](experiments/chronotransport-formal-stage-b.md)

### Claims

- [claims/project-claims.md](claims/project-claims.md)

## 研究记录字段

- `discussed`：只在科学讨论中提出，尚无冻结设计。
- `designed`：已有明确设计，但不表示代码存在。
- `implemented`：代码存在，但不表示实现正确或方法有效。
- `tested`：关键局部测试或短程运行通过，但不表示获得性能证据。
- `experiment_running`：正式数据实验正在运行，尚无终态结论。
- `empirically_supported`：匹配且可审计的实验为预先定义的主张提供支持。
- `paper_ready`：论文主张、基线、成本、泛化和统计证据均已闭环。

这些英文词只用于 Wiki 元数据和检索，不是科研正文中的私人状态语言。正文应直接说明已经完成了什么、证据支持什么以及仍缺什么；不得跨级推断，例如局部测试通过不等于方法获得实验证据。
