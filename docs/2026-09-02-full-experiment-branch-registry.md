---
updated: 2026-09-02
status: active
scope: 记录 2026-09-02 已讨论实验路线的本地工作树、GitHub 分支、远端地址、提交 SHA、矩阵规模和当前部署状态。
out-of-scope: 不记录新的 mAP、成本、bootstrap 区间或任何尚未由终态 EMA 评测回执支持的实验结论。
---

# 2026-09-02 实验分支与地址总登记

## 1. 总体口径

仓库：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`

本登记只回答三件事：

1. 每条实验路线对应哪个本地 worktree、GitHub branch 和当前远端 head。
2. 哪些提交是实现提交，哪些是后续文档/提交器绑定提交。
3. 当前远端实验 checkout 是否已经等于目标 SHA，以及现有运行能否作为最新 formal 结果。

`git ls-remote --heads origin <branch>` 已于 2026-09-02 核验，下表列出的 GitHub 分支均存在于 origin。

## 2. GitHub 分支与地址总表

| 实验路线 | 本地 worktree | GitHub branch | origin head / 当前远端分支头 | 关键实现提交 | GitHub branch URL | GitHub commit URL |
|---|---|---|---|---|---|---|
| C3 / correction-round coordination anchor | `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702` | `codex/duca-evidence-recovery-fullmatrix-20260901` | `b0ee867db1d1e4ae51ed728500b982c74e9386dd` | `b0ee867db1d1e4ae51ed728500b982c74e9386dd` | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-evidence-recovery-fullmatrix-20260901 | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b0ee867db1d1e4ae51ed728500b982c74e9386dd |
| DUCA Evidence Recovery 数值/重放更正 | `E:/DeskTop/TAD/duca_evidence_numerical_correction_20260902` | `codex/duca-evidence-recovery-numerical-correction-20260902` | `4b6df22aa8ebdaaa38fcc3a874c4229526da94cb` | `4b6df22aa8ebdaaa38fcc3a874c4229526da94cb` | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-evidence-recovery-numerical-correction-20260902 | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/4b6df22aa8ebdaaa38fcc3a874c4229526da94cb |
| DUCA CT-DP-BAMoD 几何/机制更正 | `E:/DeskTop/TAD/duca_ctdp_revised_20260902` | `codex/duca-ctdp-geometry-mechanism-correction-20260902` | `b568ca843978c8bf88ef0fc53dbf57550515e520` | `b568ca843978c8bf88ef0fc53dbf57550515e520` | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-ctdp-geometry-mechanism-correction-20260902 | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b568ca843978c8bf88ef0fc53dbf57550515e520 |
| ZoomToken BAFDR 梯度/Teacher 更正 | `E:/DeskTop/TAD/zoomtoken_bafdr_correction_20260902` | `codex/zoomtoken-bafdr-gradient-correction-20260902` | `429467797f445d77c1c49a041e3bed5a4efda962` | `429467797f445d77c1c49a041e3bed5a4efda962` | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-bafdr-gradient-correction-20260902 | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/429467797f445d77c1c49a041e3bed5a4efda962 |
| ZoomToken ET-TRC 预训练覆盖/启动更正 | `E:/DeskTop/TAD/zoomtoken_ettrc_correction_20260902` | `codex/zoomtoken-et-trc-correction-20260902` | `b3be6482eed41c433b6cb4c1a72df24d07f0cceb` | `b3be6482eed41c433b6cb4c1a72df24d07f0cceb` | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-et-trc-correction-20260902 | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b3be6482eed41c433b6cb4c1a72df24d07f0cceb |
| H65-Pro strict60 full matrix | `E:/DeskTop/TAD/OpenTAD_H65Pro_FullMatrix_20260902` | `codex/h65-pro-fullmatrix-strict60-20260902` | `e0a6471e38164a1ddcd91833117e07e300b00064` | `bd8623754a4375c39eb5c941893c606cffbcd6de` | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/h65-pro-fullmatrix-strict60-20260902 | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/bd8623754a4375c39eb5c941893c606cffbcd6de |
| DUCA Unified full matrix | `E:/DeskTop/TAD/OpenTAD_DUCA_UnifiedFullMatrix_20260902` | `codex/duca-unified-fullmatrix-20260902` | `d9f31661f6cd9973e8f45e91fcc6ba91e7faf40b` | `9aea2523d9f546ace39f1f8140ecec7a38697013`; deployment-entrypoint follow-up `90b81a6b09388b11e67c9d62a070a9285b4cc3b7` | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-unified-fullmatrix-20260902 | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/d9f31661f6cd9973e8f45e91fcc6ba91e7faf40b |

H65-Pro 和 DUCA Unified 的 branch head 包含后续文档或 review bundle 提交；形式化部署时必须在 run registry 中同时写清 `branch head` 与被声明为 `implementation commit` 的 SHA。

## 3. 实验矩阵与当前状态

### 3.1 C3 / current anchor

- 作用：保留 C3 粗分类、offline value-transport ledger、p_action learned ledger、ASFormer delta ledger 与 correction-round 汇总记录。
- 当前 branch：`codex/duca-evidence-recovery-fullmatrix-20260901`
- 当前 origin head：`b0ee867db1d1e4ae51ed728500b982c74e9386dd`
- 本地状态：该 worktree 有既有 tracked 改动：`README.md`、`docs/methods/2026-07-06-current-experiment-map-and-gpt-review-prompt.md`；另有既有未跟踪文档 `docs/CONSOLIDATED_EXPERIMENTS_RECORD.md`。本登记没有修改这些文件。
- 代表配置：
  - `configs/adatad/thumos/c3_uniform_sparse_384_ledger_adatad_full_train.py`
  - `configs/adatad/thumos/c3_paction_learned_ledger_adatad_full_train.py`
  - `configs/adatad/thumos/c3_official_asformer_delta_ledger_original_adatad_full_train.py`

### 3.2 DUCA Evidence Recovery

- Branch：`codex/duca-evidence-recovery-numerical-correction-20260902`
- GitHub：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-evidence-recovery-numerical-correction-20260902
- Commit：`4b6df22aa8ebdaaa38fcc3a874c4229526da94cb`
- 目标：修复/恢复 DUCA Evidence Recovery 的非有限值与重放路径，正式执行 seed `8261` 的 8-arm 矩阵。
- 矩阵：`C0, F, A1, A2, A3, A4, A5, A6`
- 关键配置入口：
  - `configs/adatad/thumos/duca_evidence_recovery_base.py`
  - `configs/adatad/thumos/duca_evidence_recovery_full.py`
  - `configs/adatad/thumos/duca_evidence_recovery_h65_selection.py`
- 远端状态：用户记录的远端 checkout 仍在 `ff186be`；最新 `4b6df22a` 需要重新同步并重跑 CUDA gate 与 8-arm seed `8261`。
- 结果口径：旧远端 `31 passed, 1 skipped` 是有效历史门禁信息，但不是 `4b6df22a` 上的最终 formal 实验完成证明。

### 3.3 DUCA CT-DP-BAMoD

- Branch：`codex/duca-ctdp-geometry-mechanism-correction-20260902`
- GitHub：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-ctdp-geometry-mechanism-correction-20260902
- Commit：`b568ca843978c8bf88ef0fc53dbf57550515e520`
- 目标：修正 G0/G1 selected-axis geometry，使 CT-DP geometry/mechanism 对照可归因。
- 核心矩阵：`G0, G1, G2, G3`
- 依赖矩阵：`M10, M01, M11` 在 G0-G3 门禁后再执行。
- 关键配置入口：
  - `configs/adatad/thumos/duca_ctdp_geometry_g0.py`
  - `configs/adatad/thumos/duca_ctdp_geometry_g1.py`
  - `configs/adatad/thumos/duca_ctdp_geometry_g2.py`
  - `configs/adatad/thumos/duca_ctdp_geometry_g3.py`
- 远端状态：用户记录的远端 checkout 仍在 `679b7121`；G0/G1 运行中、G2/G3 等待依赖，但这些 job 提交早于最新 geometry 修正，不能替代 `b568ca84` formal 结果。

### 3.4 ZoomToken BAFDR

- Branch：`codex/zoomtoken-bafdr-gradient-correction-20260902`
- GitHub：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-bafdr-gradient-correction-20260902
- Commit：`429467797f445d77c1c49a041e3bed5a4efda962`
- 目标：修复 BAFDR K16 梯度路径和 FULL teacher fail-closed 合同。
- 第一阶段 screen：`G96, U16-UNIFORM-A0, BAFDR-K16-LATE, BAFDR-K16-NOKD, BAFDR-K16-FULL`
- 条件开放：5-arm screen 通过后，才开放 21-cell 矩阵。
- 关键配置入口：
  - `configs/adatad/thumos/bafdr_k16_g96_seed4407.py`
  - `configs/adatad/thumos/bafdr_k16_u16_uniform_a0_seed4407.py`
  - `configs/adatad/thumos/bafdr_k16_late_seed4407.py`
  - `configs/adatad/thumos/bafdr_k16_nokd_seed4407.py`
  - `configs/adatad/thumos/bafdr_k16_full_seed4407.py`
- 远端状态：用户记录的远端 checkout 仍在 `41d5f252`；U16、LATE、NOKD 运行中，FULL 因缺少终态 Teacher checkpoint 未提交。旧 SHA 运行不能作为 `42946779` formal 结果。

### 3.5 ZoomToken ET-TRC

- Branch：`codex/zoomtoken-et-trc-correction-20260902`
- GitHub：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-et-trc-correction-20260902
- Commit：`b3be6482eed41c433b6cb4c1a72df24d07f0cceb`
- 目标：修复 ET-TRC 预训练覆盖、checkout identity、parity 测试与 ON/OFF 配对启动合同。
- 矩阵：`ET-OFF, ET-ON`
- 关键配置入口：
  - `configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_seed4407.py`
  - `configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_off_seed4407.py`
- 远端状态：用户记录的远端 checkout 仍在 `827c2526`；OFF 运行中，ON 排队。最新预训练覆盖与 parity 测试尚未在远端重跑。

### 3.6 H65-Pro strict60 full matrix

- Branch：`codex/h65-pro-fullmatrix-strict60-20260902`
- GitHub：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/h65-pro-fullmatrix-strict60-20260902
- Branch head：`e0a6471e38164a1ddcd91833117e07e300b00064`
- 实现提交：`bd8623754a4375c39eb5c941893c606cffbcd6de`
- 目标：从 verified H65 selected-axis official60 stack 出发，执行 strict60 full matrix。
- 矩阵规模：28 个 train/eval pair。
- 参考臂：`REF-D768, REF-U384, REF-MOTION384`
- 因子臂：`F01-F16`
- canonical 臂：`C0, C1, C2, C3`，多 seed。
- 关键目录：
  - `configs/adatad/thumos/h65_pro/`
  - `docs/experiments/h65_pro_fullmatrix_20260902/03_EXPERIMENT_MATRIX.csv`
  - `tools/bata/generate_h65_pro_fullmatrix.py`
  - `tools/bata/validate_h65_pro_fullmatrix.py`
  - `tools/experiments/submit_h65_pro_fullmatrix.sh`
- 已知验证：本地和远端 focused checks 已通过；远端 clean worktree 已建立在 `/data/run01/sczc063/yuzibo/OpenTAD_H65Pro_FullMatrix_20260902_bd862375`。
- 部署状态：`PRECHECK_ONLY=1` 通过；formal Slurm 部署仍受 scheduler submit limit 阻断。无 final-commit registry 和无正式 mAP。

### 3.7 DUCA Unified full matrix

- Branch：`codex/duca-unified-fullmatrix-20260902`
- GitHub：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-unified-fullmatrix-20260902
- Branch head：`d9f31661f6cd9973e8f45e91fcc6ba91e7faf40b`
- 初始实现提交：`9aea2523d9f546ace39f1f8140ecec7a38697013`
- 后续提交器/执行入口对齐提交：`90b81a6b09388b11e67c9d62a070a9285b4cc3b7`
- 目标：把 DUCA phase fields、robust phase selector、Taylor attribution、physical-time CT 约束、A-MoD 和 successful-update curriculum 放入一个不可再动态扩展的 41-task matrix。
- Matrix ID：`DUCA-UNIFIED-FULLMATRIX-v001-20260902`
- 矩阵规模：17 development + 24 confirmation = 41 train/eval tasks。
- 主比较：`A11 - A10`
- 成本 arms：`U0, A11, C11, E01, F11`
- confirmation arms：`U0, H0, A10, A11, C11, D1, E01, F11`
- confirmation seeds：`4407, 5407, 6407`
- 关键目录：
  - `configs/adatad/thumos/duca_unified_fullmatrix/`
  - `scripts/duca_unified_fullmatrix/matrix.tsv`
  - `scripts/duca_unified_fullmatrix/submit_all.sh`
  - `docs/experiments/DUCA_UNIFIED_FULLMATRIX_LOCAL_IMPLEMENTATION_REPORT.json`
- 部署状态：本地 implementation/report 存在；branch 已推到 origin；没有真实 Slurm job IDs，不能声称已部署或已有 mAP。

## 4. 重合关系记录

跨路线共享的是实验外壳，不是同一实验：

- 共享数据和主干：THUMOS14、VideoMAE-S、AdaTAD/ActionFormer。
- 共享常见预算：`T=768`、`K=384`、60 epoch / 6000 successful updates。
- 共享评估口径：终态 EMA checkpoint，禁止把 PENDING、测试通过、checkpoint 存在或旧提交运行误写为正式结果。

不能互相替代的关键差异：

- C3 是 offline ledger 输入采样路线，主要固定 detector。
- DUCA Evidence 是 H65 ledger 驱动的 evidence recovery，不是 Unified online selector。
- DUCA CT-DP 是 DualPhase/CT/B-AMoD geometry 消融，不等于 Unified C11。
- ZoomToken BAFDR 是 spatial ROI / K16 / teacher-distillation 路线。
- ZoomToken ET-TRC 是 dense adapter Taylor ON/OFF 配对。
- H65-Pro 是 H65 selected-axis strict60 factorial/canonical 矩阵。
- DUCA Unified 是当前最完整的 DUCA 机制矩阵，主比较是 `A11 - A10`。

## 5. 现有实验推进顺序

1. 先同步四条正在跑或排队的 correction branches 到各自最新 SHA：`4b6df22a`、`b568ca84`、`42946779`、`b3be6482`。
2. 先重跑门禁，再决定是否保留旧提交 job 作为诊断；旧提交 job 默认不进入 formal 结果表。
3. BAFDR 必须先补齐 D160 epoch-59 `state_dict_ema` Teacher，再允许 FULL。
4. CT-DP 必须先用 `b568ca84` 重跑 G0/G1，再让 G2/G3 依赖继续。
5. H65-Pro 保持为 28-task strict60 因子归因矩阵；formal 部署需等待 scheduler submit limit 解除或降低提交并发策略。
6. DUCA Unified 作为主线 DUCA 机制矩阵；部署时使用 branch head 或明确冻结的 final SHA，并在 `submission_manifest.json` 写入精确 revision。

## 6. 最小审计清单

每个 formal 结果进入论文或最终表格前，必须同时具备：

- GitHub branch URL。
- GitHub commit URL。
- 远端 checkout path。
- `git rev-parse HEAD` 等于登记 SHA。
- 远端 tracked working tree clean。
- 对应 gates/focused tests 通过。
- Slurm job ID 与 dependency 记录。
- 终态 EMA checkpoint。
- 独立 `tools/test.py` 评测日志。
- 无 validation/test GT、teacher、oracle 或 raw-prediction shortcut 泄漏。
