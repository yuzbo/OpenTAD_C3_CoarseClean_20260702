# 实验计划：ODF-CR 内部决定性 2×2

**问题**：在不削弱官方 ActionFormer proposal floor 的前提下，条件 residual
是否有独立价值，以及固定 K384 residual 支持会损失多少？

**方法主张**：官方三层 dense head 作为不可削弱的 proposal floor，额外
residual 只负责可选择的增量计算。

**日期**：2026-07-31

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: 旧 G1 的主要损失来自一层 scaffold | 决定是否应保留官方质量 floor | 新 holdout、三新种子下 `d1_off - d3_off` 稳定为负，且 `d3_off` 与官方 dense 精确等价 | B0, B1 |
| C2: residual 在官方 floor 上有正价值且可被条件执行 | 决定是否存在可继续的稀疏计算路线 | `d3_all-d3_off` 通过 utility gate，随后冻结 K384 replay 通过 support gate | B2, B3 |
| Anti-claim: 结果只是 selected-loss、旧 holdout 或随机种子 | 排除旧 G1 混杂 | full-grid supervision、新 holdout-v2、三新种子、同 seed 配对和独立 evaluator | B0-B4 |

## Paper Storyline

- Main paper must prove: 当前阶段不产生论文主结果。
- Appendix can support: 内部方法淘汰逻辑、结构等价门和负结果诊断。
- Experiments intentionally cut: 官方 test、预算 sweep、selector 学习、阈值/NMS
  调优和五种子成本扩展。

## Experiment Blocks

### B0: 数据与实现门

- Claim tested: 新结果不复用旧 holdout，`d3_off` 是官方 dense 恒等实现。
- Dataset / split: builder 强制读取旧 manifest，仅从旧 train-160 中选新
  holdout-40；training-v2 是其余 160，且新旧 holdout 严格不相交。
- Compared systems: official dense, `d3_off` 初始化和 CUDA forward。
- Metrics: state/tensor/output exact equality。
- Success criterion: 全部 exact；任何差异阻断训练。
- Priority: MUST-RUN。

### B1: Scaffold depth

- Claim tested: 一层 floor 是否解释旧 G1 的主要下降。
- Compared systems: `d1_off` vs `d3_off`。
- Metrics: Avg、0.3--0.7、class/duration/boundary/recall。
- Setup: 三新种子，同 manifest/schedule/EMA/evaluator。
- Success criterion: 无正向门槛；精确报告配对差异。
- Failure interpretation: 若几乎无差，则旧归因需被推翻。
- Priority: MUST-RUN。

### B2: Residual utility

- Claim tested: all-query residual 是否在不同 floor 深度上产生独立价值。
- Compared systems: `d1_all-d1_off`、`d3_all-d3_off` 和 interaction。
- Success criterion: depth-3 Avg `>=+0.25pp`、至少 2/3 seeds positive、
  @0.6/@0.7 均 `>=0`。
- Delta contract: 每个 seed 先做配对相减，再报告三种子的算术均值与样本
  标准差；所有 delta 均为 percentage points，阈值相等算通过。
- Failure interpretation: residual 无足够价值，路线终止。
- Priority: MUST-RUN。

### B3: Frozen K384 support

- Claim tested: 保持训练轨迹和 checkpoint 不变时，K384 是否保留 residual
  价值。
- Compared systems: `d3_all` all-query vs K384 counterfactual replay。
- Selector contract: `stratified_uniform`、K=384、hash seed
  `2026073100`、每视频恰为 `min(384, valid)`，allocation IDs/hashes
  receipt-bound 且跨 arms/seeds 使用同一确定性 tie-break。
- Success criterion: Avg `>=-0.5pp`、@0.6/@0.7 `>=-1pp`。
- Failure interpretation: residual 可用但固定 K384 支持不足；禁止调 K。
- Priority: MUST-RUN only after B2 artifacts exist。

### B4: Failure and cost diagnostics

- Claim tested: 性能差异是否来自 recall、边界、类别、时长或 score compression。
- Metrics: pre/post-NMS recall、boundary、class、duration、gradient、
  residual/floor norms、完整 feature-to-detection latency。
- Claim boundary: diagnostic-only，不支持效率或官方结果。
- Priority: MUST-RUN diagnostics; cost is secondary。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | static/focused tests + holdout-v2 | CPU/Linux | exact manifest and schemas | low | split leakage |
| M1 | real-CUDA exact identity | official dense vs `d3_off` | exact tensors/outputs | <1 GPU-hour | module/RNG drift |
| M2 | four-arm factorial | 3 seeds × 4 serial arms | complete receipts | about 2× prior G1 GPU cost | 48h allocation may be short |
| M3 | aggregate and K384 replay | frozen checkpoints | utility then support gates | low | counterfactual schema drift |
| M4 | analysis/audit | no training | claim boundary complete | low | overclaiming internal results |

## Compute and Data Budget

- Total training arms: 12.
- GPU layout: Slurm array of three one-GPU tasks; four serial arms per task.
- Expected GPU time: approximately twice the prior two-arm-per-seed
  dense+DCSR three-seed G1.
- Data preparation: one immutable holdout-v2 manifest; no test subset.
- Biggest bottleneck: four serial 35-epoch arms within allocation wall time.

## Risks and Mitigations

- Adaptive reuse of old holdout: freeze a disjoint holdout-v2.
- Official floor not exact: real-CUDA G0 blocks training.
- Depth changes parameters: use residual-off and residual-on within each depth.
- Support changes optimization: K384 is frozen replay, not training.
- Internal result presented as official: every receipt forbids paper rows.
- Negative gate mistaken for job failure: valid negative aggregate exits zero.
- 三个 seeds 是同一 holdout 上的训练重复，不是三个独立 validation split；
  不允许据此宣称跨划分或总体泛化。

## First three actions

1. Implement the isolated `official_dense_floor_factorial` mode and exact G0.
2. Freeze/validate holdout-v2 and four static configs.
3. Run Linux focused tests and real-CUDA G0 before submitting factorial training.
