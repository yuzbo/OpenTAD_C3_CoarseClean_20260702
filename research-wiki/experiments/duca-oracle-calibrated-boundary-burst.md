---
id: exp:duca-oracle-calibrated-boundary-burst
type: experiment
status: experiment_running_r0_r5_dag
updated: 2026-07-22
canonical_contract: research-wiki/duca_final_model_contract.md
---

# DUCA Oracle-calibrated boundary-burst evidence plan

## Exact e49 full R0-R5 deployment (2026-07-22 16:05 +08:00)

- Exact commit: `e49ef69605e1f98a7217957483f93a8a64bfc348`; branch
  `codex/duca-boundary-burst-20260722`; GitHub commit URL:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/e49ef69605e1f98a7217957483f93a8a64bfc348`.
- Clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_e49ef69_20260722`.
  Exact verification is R0-R5 `192 passed`, mandatory C3 `23 passed`, plus compile,
  shell syntax, exact HEAD and clean tree.
- Predeployment independent MAX `019f88bf-272f-7373-b702-5b66b142cbdc` returned
  `GO_TO_SLURM` after checking model mechanism, gradient ownership, train/test identity,
  K384/K256 freedom, terminal evidence and dense cost binding.
- R0--R3 root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_20260722_155037_r0_r3`.
  Jobs are R0 `1179795`, P0 `1179796`, full gate `1179797`, matched U `1179798`,
  selected G0 `1179799`, aggregate `1179825`.
- R4 root ends in `_r4`; legal hard-swap/alignment Job is `1179826`, dependent on
  R3 aggregate.
- R5 root ends in `_r5`; real TemporalMaxer one-step Job is `1179827`. The frozen matrix
  contains 24 terminal cells: two real backends x U/learned x K384/K256 x three seeds,
  plus 8 matched frontend/full-stack costs and one true dense-768 historical checkpoint cost.
- N16R4 rejected the seventh independent R5 cell with `AssocMaxSubmitJobLimit`. To preserve
  every experiment while respecting the site limit, four GPU bundles `1179861--1179864`
  each execute six unchanged original cell sbatches; Job `1179865` executes all nine unchanged
  cost sbatches and the unchanged final aggregate. The six canceled jobs `1179828--1179833`
  never ran and are audit-only duplicates.
- Receipt hashes: R0--R3 jobs `a5b8abab...c4e7f0`, sealed journal
  `e2f7e9d8...dfdf82`, R5 jobs `954ec7db...1421dc`, R5 site bundle
  `e6ca0acf...5656d8`, deployment receipt `ed217ee2...aa1bd`.
- At 16:05, `1179795` is RUNNING on `g0013`; every downstream job is accepted and pending
  only on its declared dependency. Error scan is clean. No terminal e49 mAP or cost exists yet,
  so C3/C4/C7 and paper readiness remain unproven.
- At 16:11, R0 completed all 124 real windows for all four families and atomically promoted
  `holdout_families.jsonl`; SHA-256 is
  `f2cbcd274383d1f2e59df7a3c33c59575f488f4e866d9e96f2b08894e6eafa4d`.
  It then entered frozen official-AdaTAD replay beginning with exact-uniform. This closes the
  old false-infeasibility recurrence check but is not yet mAP headroom evidence.
- At 16:13, exact-uniform replay completed with raw Avg-mAP `93.5871`; mAP at
  tIoU `0.3/0.4/0.5/0.6/0.7` is `98.05/97.45/95.44/91.85/85.15`. This reproduces
  the d9 exact-uniform point estimate and proves baseline replay identity, but headroom remains
  undecided until R2Q3/R4Q5/unrestricted and paired bootstrap finish.
- At 16:17, projected R2Q3 completed with raw Avg-mAP `94.1905`, a `+0.6034 pp`
  point-estimate gain over exact-uniform. Its tIoU `0.3/0.4/0.5/0.6/0.7` mAP is
  `98.15/97.52/96.63/93.04/85.61`; gains concentrate at `0.5/0.6` (`+1.19/+1.19 pp`)
  and remain positive at `0.7` (`+0.46 pp`). This supports the boundary-burst reachability
  mechanism as a raw diagnostic, but family selection still requires the sealed paired bootstrap.
- At 16:23, projected R4Q5 completed with exact raw Avg-mAP `93.999241`, a
  `+0.412170 pp` gain over exact-uniform but `-0.191256 pp` below R2Q3. Its exact
  tIoU `0.3/0.4/0.5/0.6/0.7` mAP is
  `98.0039/97.3369/95.8351/92.8220/85.9983`. The larger burst improves high-IoU
  0.7 slightly more than R2Q3, but its average gain is smaller. Unrestricted replay and
  paired bootstrap remain pending, so no family has been sealed and P0 has not started.
- At 16:30, unrestricted GT Oracle completed with exact raw Avg-mAP `93.970057`;
  tIoU `0.3/0.4/0.5/0.6/0.7` is
  `98.1881/97.3696/96.3783/93.0427/84.8715`. It gains `+0.382987 pp` over
  exact-uniform but trails R2Q3 by `-0.220440 pp`, confirming that removing the projected
  coverage constraint does not monotonically improve detector utility. All four point estimates
  now exactly reproduce d9. Job `1179795` remains RUNNING in the preregistered 1000-sample
  paired-video bootstrap; `r0_bootstrap.json`, `r0_summary.json` and the P0 decision are pending.

## Corrected R0 execution (2026-07-22 11:24 +08:00)

- Job: `1179517` (`burst_r0_d9fb398`), running on `g0048` at 11:26 +08:00.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_d9fb398_r0_formal_20260722_112357`.
- Submission manifest SHA-256: `66bb5f5d3558073031d4f3739141ed646fe4fbe51f2c9dd62088c56191f74e15`.
- R0 sbatch SHA-256: `4a6ace6c4c955381c4cfefa1f565beedfa02c0256901bc09760709721a3fab8c`.
- Atomic one-row journal SHA-256: `25d90d861fd9fa324f4ed293a8f0c5baabc5a0418e08f295871985b183107f4a`.
- Split manifest SHA-256: `88309edf555a1d8f0629b74d6aedff0cdb1ef2fc10dbd869eb6a95885b47708f`.
- The journal contains only R0 with dependency `none`. P0, gate, official
  arms and aggregate were not submitted.
- Export started normally (`batch 0`, one sample); stderr and error scan were
  empty at the first live check.
- At 11:31 +08:00 all `124/124` input windows were exported and the job moved
  into four-family Oracle construction. Export SHA-256 is
  `5e65ac3df5cf36270243ecb4f95ced5ad23ab6be924dda5fc129079828823749`;
  final detector mAP and bootstrap artifacts do not exist yet.

## Independent staged verdict (2026-07-22 11:24 +08:00)

- MAX `019f87bb-d767-7713-825e-92b893e49a98` grants R0-only GO on exact d9.
- P0 is HOLD pending positive corrected R0 headroom and a sealed selected
  family. Full-model is HOLD pending a legal P0 artifact. Official-60 is HOLD
  on those upstream gates plus two bounded scheduling/provenance repairs.
- The authorized run is corrected R0 only. It must not submit P0, gate,
  official arms or aggregate dependencies.

## Exact d9 status (2026-07-22 11:06 +08:00)

- Current immutable candidate is
  `d9fb398578716d278e818745677a92976bcedf2c`; clean remote snapshot is
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_d9fb398_20260722`.
- Remote static evidence is focused DUCA `88 passed`, required C3 `23 passed`,
  pycompile and shell syntax pass.
- Two full replays of the real 124-window R0 input produced byte-identical
  family JSONL with SHA-256
  `b49e03c2f4222512cf7752bd3c89bad714868ae69e7c9d05980f9e9f47edd6d7`.
  This closes the reproducibility blocker without changing R0 geometry.
- Fresh independent MAX `019f87bb-d767-7713-825e-92b893e49a98` is auditing
  R0, P0, full-model and official stages separately. No stage is authorized
  merely by the replay.
- No corrected R0 job is submitted. The next allowed action is R0-only after
  an explicit R0 GO; P0 and all training remain blocked until R0 headroom and
  their separate contracts pass.

## Corrected R0 status (2026-07-22 08:40 +08:00)

- Immutable R0 Job `1179392` (`f90595d`) failed after successful 124-window
  export and before detector evaluation. It produced no headroom mAP.
- Exact failure: `video_validation_0000206|0`, valid length 768, K=384, G=2,
  33 segments and 65 valid endpoints. R2Q3 passed; old R4Q5 fixed-nearest-Q
  required positions could not coexist with the physical path cap.
- Independent mathematical audit proved this was a false infeasibility caused
  by implementation over-constraint. The intended joint center/quota/
  bilateral/K/G problem has an exact zero-gap witness; DP, crop validity,
  K_eff and physical-axis conversion were not the cause.
- Corrected exact commit:
  `22555a4e830ce24f9bb516897b1bb7f44b70c188`. Remote snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_22555a4_20260722`.
  Verification: `22 + 9 + 54 + 23` tests, pycompile/bash/HEAD/clean, and an
  exact real-sample replay with all four R0 families `ok=true`.
- P0 evidence-chain fixes are implemented in the same commit but not
  authorized for execution. Corrected R0 awaits independent MAX
  `019f8743-aed1-7a80-a7d6-552b08491019`; no P0/full-model/official-60 job is
  queued and no boundary-burst mAP exists.

The current question remains R0 only: does a correctly implemented projected
Oracle feasible family have paired-bootstrap detector-mAP headroom over the
same frozen detector and exact-uniform input? If not, kill the feasible-family
claim before frontend training. If yes, freeze the weakest passing projected
family and continue to R1/P0 without adding a new selector.

## Exact successor status (2026-07-22 04:45 +08:00)

The two aa3352e audit blockers were minimally repaired in exact commit
`f629ad79461941f405bc2028f087034abd17a840`. Submit-frozen AdaTAD pretrain
path/SHA is now consumed by P0, each frontend arm and gate, and G1/G2 are
rejected by production runtime binding before real legal hard-swap alignment.
Remote exact-snapshot evidence is affected DUCA `63 passed`, required C3
`23 passed`, plus pycompile/bash/HEAD/clean. Independent no-context MAX agent
`019f866a-6879-75a0-99f4-3c9524ebd076` is reviewing this commit. This is not
deployment GO; CUDA/R0/P0/official-60 remain unsubmitted.

## Current exact status (2026-07-22 04:26 +08:00)

Exact candidate `aa3352ecf803c81d007a62ed5398667d9551684b` received
`HOLD_FIX_REQUIRED` from an independent no-context MAX audit. Model geometry
was not rejected. Deployment remains blocked by two evidence contracts:

1. freeze the AdaTAD pretrain path/SHA at submit time and verify it in every
   P0/frontend/gate consumer before work begins;
2. make production runtime binding fail closed for G1/G2 until real legal
   hard-swap alignment passes.

No CUDA gate, R0, P0 or official-60 boundary-burst job has been submitted.
Old V8 Job `1178989` is terminal diagnostic evidence only.

## Question

在严格总预算和真实总成本约束下，能否用 deploy-visible 的粗动作状态变化证据，学习
Oracle 式“端点居中、左右聚集、达到配额后停止奖励”的全局选帧策略，并在官方 TAD
mAP 上稳定超过同提交 exact-uniform？

## Final model under test

本实验不创建新 detector。它只允许在 V8 的 coarse/official-ASFormer、transition
scorer、global exact-K/max-hole DP、selected-axis AdaTAD adapter 和原始
ActionFormerHead 上加入一个 bounded burst-profile 语义。训练期 GT 只构造
center/bilateral/quota targets；推理期只使用 coarse state evidence。

完整模块、两阶段课程、梯度所有权和成本定义以
`research-wiki/duca_final_model_contract.md` 为唯一合同；本页只维护实验门禁和结果。

## Primary claims

1. **Allocation claim:** corrected boundary-burst G0 在 matched fixed K 下的
   terminal-EMA mAP 高于 same-commit exact-uniform U。
2. **Efficiency claim:** 包含 dense cheap probe、selector、H2D、VideoMAE、head 和
   NMS 的完整推理成本低于 dense route，同时保护高 tIoU 与短动作。

Supporting claim only after a separate gate: protected detector feedback G1
improves G0 without corrupting coarse action semantics.

## Execution gates

### R0: no-training reachability

- Compare exact uniform, unrestricted GT Oracle, and GT Oracle projected into
  candidate K/G families.
- Measure official detector mAP where a compatible frozen checkpoint exists,
  plus center distance, bilateral counts, burst-size distribution, missed
  endpoints, overlap rate and max hole.
- Freeze radius/quota/max-hole from training data only. No learned run starts
  if the projected Oracle has no positive detector-mAP headroom over uniform.

### R1: implementation contract

- Brute-force exact structured probabilities/count events on small T/K/G.
- Verify active event gradients are nonzero when headroom exists and zero only
  for mathematically forced events.
- Verify exact K, chronological uniqueness, overlap deduplication, original
  time, no leak, hard/soft family equality and optimizer ownership.
- Reject post-hoc repair or one-frame-per-cell fallbacks.

### R2: frontend P0

Use one sealed train-only split and the same coarse/scorer architecture. Compare
only: old Gaussian-mass V8 objective, simple `abs(delta p_action)` diagnostic,
and corrected anchor+bilateral+quota burst objective. Select the earliest
checkpoint passing coarse calibration, scorer-versus-delta, endpoint centering,
bilateral support, quota/fairness and hard K/G gates. No validation/test mAP is
used for checkpoint choice.

### R3: matched main anchor

Run U and corrected G0 first under the same immutable commit, seed, data,
pretrain, successful-update count and terminal EMA. If G0 does not beat U,
stop the learned allocation claim for that K/G family and do not deploy G1/G2.

### R4: detector feedback

Run real legal hard-swap finite-difference alignment on the actual detector.
Only a passing sign/rank agreement artifact licenses G1. G2 then adds the
training-only exact-uniform companion as one stability ablation; it remains
absent at inference.

### R5: paper closure

- Three seeds for U and the best learned arm.
- Fixed-budget curve, initially K=384/256 and lower K only if stable.
- Dense, exact-uniform, random, actionness top-k, `abs(delta p_action)`, old
  Gaussian V8 and corrected DUCA as auditable baselines.
- AdaTAD primary backend plus one second official TAD backend.
- IoU-wise mAP, short/medium/long action strata, endpoint/burst diagnostics and
  sample visualizations.
- Full-stack p50/p95 latency, throughput, FLOPs/MACs, peak memory, energy and
  preprocessing/H2D cost.
- Separate training/reproduction dense-materialization cost from deployment
  low-resolution-proxy plus selected-high-resolution-materialization cost.

## Seed-0 GO-to-replication criterion

The THUMOS/AdaTAD seed-0 result unlocks replication only when the best learned terminal-EMA arm has
Avg-mAP `>=65.00`, improves over matched U by `>=0.20`, keeps mAP@0.6 and
mAP@0.7 within `0.20` of U, and has lower measured end-to-end inference cost
than the dense route. Final paper GO additionally requires the preregistered
three-seed mean to remain positive and the deployment-mode total-cost claim to
survive. Only then is the second detector paper evidence rather than exploration.

## Stop conditions

- Projected Oracle has no mAP headroom: revise K/G feasible family, not losses.
- Corrected P0 remains worse than simple delta: stop scorer training and audit
  coarse evidence/descriptor identifiability.
- Corrected G0 terminal EMA does not beat U: kill learned allocation at that
  K/G rather than add feedback or longer training.
- Hard-swap alignment fails: retain G0 only and remove detector-aligned claim.
- Full-stack cost is not lower than dense: remove the efficiency claim.

## Evidence status

`hold_fix_required_after_independent_max`. V8 Job `1178989` 已负向封存；新候选已通过
Linux focused tests，但第二轮独立 MAX 仍发现正式 runtime/evidence blocker。真实 CUDA
gate 与正式 DAG 尚未提交，所以仍无 R0 headroom、boundary-burst mAP 或 paper claim。

## 2026-07-22 02:00 implementation candidate

- 唯一隔离分支：`codex/duca-boundary-burst-20260722`；当前精确候选提交
  `fdf25f5d08bc0bf9b550e059228ce1d6ac587499`，基于 V8 `63e25eb`，已推送 GitHub。
  首个实现提交为 `4a07a2a`；`fdf25f5` 仅修复短窗有效预算的 P0 判定合同。
- 没有新建 selector/decoder/worktree family。现有 transition scorer 增加零初始化
  左右 offset profile；现有 global exact-K/max-hole DP 仍是唯一 hard/soft decoder。
- P0 冻结为三臂：matched Gaussian、R2/Q3、R4/Q5；每臂仅在 train-only 80/20
  split 的 holdout 上按 5/10/15/20 epoch 评估粗分类与边界微簇质量。最终机制排名只由
  same-commit terminal-EMA mAP 决定。
- R3 冻结为四臂：exact-uniform U、Gaussian G0、R2/Q3 G0、R4/Q5 G0；均为 K384、
  G2、official-60/6000 successful updates、seed3407、同一 AdaTAD/ActionFormer backend。
  G1/G2 未实现、未部署，必须等待 G0 与 hard-swap alignment。
- R0 现在可在同一冻结 `transition_beta0 epoch_131` detector 上回放 exact-uniform、
  capped GT Oracle 和 unrestricted GT Oracle 的 selected-axis mAP。它完整包含背景窗口，
  评估真值严格限定到内部留出视频，运行时 selector 不读取 GT。
- R0 绝对训练留出 mAP 不能写入论文主表：冻结 detector 的训练数据包含这些视频。
  其用途仅是比较同一 detector/同一视频下 Oracle 相对 U 的可行域增益；绝对泛化性能
  仍由 R3 THUMOS terminal test mAP 裁决。
- 本地证据：边界微簇/config/质量分析通过，新增短窗有效预算测试 `3 passed`，三套 P0 静态合同通过，脚本
  `bash -n` 与 `py_compile` 通过。Windows Torch 测试因本机 `c10.dll` 问题转到远端。
- 全新远端 Linux 快照 focused 为 `84 passed, 2 skipped`，必要 C3/ASFormer 回归为
  `23 passed`，提交 DAG `PRECHECK_ONLY=1` 通过。独立 MAX agent
  `019f85d3-cb6e-7aa0-bb33-ed2271cccc56` 的实际审核任务 ID 为
  `019f85d3-38b6-7a90-8d20-1d7c8b88fe8e`，裁决 `HOLD_FIX_REQUIRED`；正式 DAG 保持阻断。
- HOLD 修复拆为三组：逐样本 K/G、端点坐标和 earliest-pass；裁剪 endpoint validity、
  双侧配额 R0 Oracle 与 positive-headroom 决策；R0->P0 依赖、split hashes 和 U/learned
  制品语义。主线程另将 Q3/Q5 从仅 loss 参数改为预测中心的 quota-limited offset support。

## 2026-07-22 03:27 exact repair candidate

- 同一分支已就地收敛到精确提交
  `899630a5ef4927e78ef4ca6b8cc51fdf754056da`，GitHub 与本地隔离树一致且干净。
  `920d06e` 关闭 R0->P0->gate->四臂依赖、split 路径/哈希消费和 U 臂空制品语义；
  `03aa4ce` 加入 crop endpoint validity 与 exact-uniform/R2Q3/R4Q5 R0 可行族；
  `1ef0449` 完成逐样本 K/G、训练同构端点坐标、earliest-pass P0 机制门禁，以及预测
  center 的 forward-exact quota-limited offset support。后续两个提交只修正 straight-through
  浮点测试的逐位相等错误，模型数学不变。
- 最终干净 Linux 快照为
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_899630a_20260722`。
  受影响 DUCA/official-backend 回归为 `136 passed, 3 skipped`；仓库强制 C3/ASFormer
  回归为 `23 passed`；`py_compile`、两个启动脚本 `bash -n`、精确 HEAD 与 clean-tree
  检查全部通过。
- 新独立 MAX 审计任务 `019f8614-53e8-79e2-8daa-d52f7be04623` 正逐项复核上一轮九个
  blocker。真实 CUDA gate 与正式 DAG 在其给出 `GO_TO_REAL_CUDA_GATE` 前保持阻断。
- 旧 V8 诊断 Job `1178989` 已从 running 变为 `FAILED/2:0`，总运行 `05:32:02`。
  它不是 boundary-burst 主实验；失败阶段和最后一个有效 P0 制品仍需从日志封存，禁止
  把该状态误写成 terminal mAP 或重新运行旧 scorer 路线。
- 当前状态是 `implemented_exact_candidate_linux_tested_under_independent_max_audit`；仍无
  R0 headroom、P0 winner、official-60 terminal mAP、超过 65 或 V9/paper-ready 结论。

## 2026-07-22 04:00 second independent audit verdict

- 独立 MAX `019f8614-53e8-79e2-8daa-d52f7be04623` 对 `899630a` 给出
  `HOLD_FIX_REQUIRED`，不是方向 KILL，也不是模型数值失败。
- 已确认通过：核心 center-conditioned burst、R2Q3/R4Q5 forward quota、全局 exact-K/G
  DP、selected-axis 映射、no-leak、official-derived detector 主体和大多数上一轮 blocker。
- 正式阻断：boundary 四臂尚未注册进 selected-axis runtime binder，gate schema/config
  映射不一致；pooled quality export 丢失 crop endpoint validity；R0/P0/aggregate 仍需从
  已哈希源文件重建指标并固定 split/checkpoint/manifest/dependency hashes。
- pooled validity 已在同一 tree 完成最小修复，focused `20 passed`。其余 runtime 与
  provenance 修复正在原分支就地完成；不得另建 selector/decoder/tree。
- 下一许可顺序保持：修复新 exact commit -> Linux tests -> 全新独立 MAX GO -> real CUDA
  gate -> R0。任何一步未过都不得提交 official-60。

## 2026-07-22 04:05 exact blocker-repair candidate

- 上一轮三个剩余 blocker 已在同一分支收敛为精确提交
  `aa3352ecf803c81d007a62ed5398667d9551684b` 并推送 GitHub。
- 四臂 production runtime binder 已注册 U/Gaussian/R2Q3/R4Q5；boundary gate schema、
  `formal_training_unlocked` 与 config-stem full-model artifacts 已形成真实
  `build_runtime_bindings` 回归。
- pooled metrics 现消费 crop endpoint validity；R0/P0/aggregate 从已哈希 metrics/
  evaluation 源文件重建数值，并封存 split、checkpoint、pretrain、decision、gate、
  completion 依赖。
- 干净远端快照
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_aa3352e_20260722`
  通过 DUCA `139 passed, 3 skipped`、C3 `23 passed`、compile/bash/HEAD/clean。
- 全新独立 MAX `019f8647-ad93-70f3-a763-218f7552ac95` 正对精确提交做无上下文只读
  复审。CUDA gate 和正式 DAG 保持阻断；当前仍无 R0/P0/mAP 结果。

## 2026-07-22 05:45 terminal evidence producer repair

- `7b9ad0be8c7ae0bdc057dca7b491e8e0e5319fc5` sealed the terminal checkpoint,
  evaluation, prediction and aggregate chain, but inspection of a real historical selected-axis
  sidecar found a production/validator schema mismatch: the audit producer omitted
  `formal_protocol` and `training_profile` while terminal validation required both.
- Exact commit `86f7663a94d628eace316d17e31db7043f731f75` minimally fixes that mismatch.
  Tests now construct terminal audit evidence through the real production builder, and aggregate
  independently checks the selected-axis protocol/profile.
- Exact clean remote snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_86f7663_20260722`
  passed DUCA `64 passed`, C3/update evidence `29 passed`, pycompile, bash syntax, exact HEAD and
  clean-tree checks.
- Fresh no-context MAX audit `019f86a6-fe1b-7921-b576-8d9cd3d4c8ac` is running. No CUDA gate,
  R0, P0, official-60 or mAP job has been submitted. Status remains
  `implemented_exact_candidate_under_final_independent_reaudit`.

## 2026-07-22 06:21 exact-commit review HOLD

- MAX audit `019f86a6-fe1b-7921-b576-8d9cd3d4c8ac` completed with
  `HOLD_FIX_REQUIRED`; `86f7663` is no longer deployment-eligible as-is.
- Accepted: boundary-burst model identity, R2Q3/R4Q5 support, global exact-K/max-hole DP,
  selected-axis/no-leak contracts and official-derived AdaTAD path.
- Blocking R0 evidence work: add unrestricted Oracle and per-video official-evaluator bootstrap;
  independently reopen/recompute every evaluator/subset/prediction/checkpoint/config/annotation/
  class-map hash; freeze the unique weakest feasible geometry from a preregistered CI rule.
- Blocking R2 comparison work: run simple `abs(delta p_action)` through the same global DP and
  enforce the declared simple-delta stop rule in candidate eligibility.
- P2 repairs: crop-valid burst allocation metrics, crash-safe per-job submission journal and one
  no-mock producer-to-official-evaluator integration test.
- No CUDA, R0, P0 or official-60 job has been submitted. Current status is
  `implemented_candidate_hold_fix_required`; there is still no boundary-burst mAP evidence.

## 2026-07-22 06:54 R0 contract repair implementation

- Exact commit: `4ec3e078a3aad834ffe504d74d414bf7e2b6fad3`.
- R0 family order is frozen as U, projected R2Q3, projected R4Q5, unrestricted exact-K GT
  boundary-burst Oracle. The unrestricted arm removes the coverage scaffold; only projected
  arms are eligible for deployment geometry selection.
- The decision rule is paired video-cluster bootstrap with 1000 official-evaluator reruns,
  seed 3407, confidence 0.95 and CI lower bound `>0.002`; select the first feasible projected
  family in R2Q3→R4Q5 order.
- P0 revalidates all upstream files and reruns both official mAP and bootstrap. Candidate
  selection also requires a strict Pareto improvement over `pure_delta_same_feasible_dp`.
- Local protocol evidence: `22 passed`; journal evidence: `8 passed`; compile/bash/diff clean.
- Status is `implemented_local_tested_pending_linux_and_independent_max`. No experiment is
  queued and no R0 headroom, P0 winner, terminal mAP or paper claim exists.
- Clean Linux snapshot evidence is `109 passed` affected DUCA and `23 passed` mandatory C3,
  with compile/bash/HEAD/clean checks. The next action is the frozen exact-commit MAX audit;
  CUDA/R0 remains blocked pending its explicit GO.
- `PRECHECK_ONLY=1` passed without submitting Slurm jobs at
  `duca_boundary_4ec3e07_precheck_20260722_0705`. Submission-manifest SHA-256 is
  `b068843e86ce25822ddc78cc2931b04e1323e314d4b60437f223bb8f7a78b4b3`; split-manifest SHA-256 is
  `dc4ca5b4c3fbfa4749ab378108dd5ebac7107eba060a44f381219e6322b0493c`. The manifest binds the
  exact commit, transition-beta0 epoch-131 checkpoint, AdaTAD pretrain, split files and
  R0→P0→gate→four arms→aggregate dependency order.

## 2026-07-22 07:15 launch-policy consistency risk

- R0 correctly freezes exactly one `selected_weakest_projected_family` from R2Q3→R4Q5 by the
  preregistered paired-bootstrap rule.
- The current P0 selector nevertheless requires Gaussian, R2Q3 and R4Q5 all to produce passing
  winners, and the current downstream gate/DAG requires all four U/Gaussian/R2Q3/R4Q5 arms.
- This is inconsistent with R3 in the canonical paper plan, whose decisive first result is matched
  U versus the single R0-selected corrected G0. A failed diagnostic family must not veto that pair.
- Status: `audit_pending`; independent MAX `019f86e9-8aa0-75e1-8373-686265ac8b61` is still running.
  No production change or Slurm submission is allowed until its verdict is absorbed.

## 2026-07-22 07:30 split-block audit and exact repair

- Independent MAX `019f86e9-8aa0-75e1-8373-686265ac8b61` returned `HOLD_FIX_REQUIRED` for
  `4ec3e07`. It found no method-level P0 blocker; the only P1 blocker was the R0 launcher using
  `train_block_list` as holdout-evaluator blocked IDs.
- The split contract is consumer-oriented: `train_block_list == holdout_videos` and
  `holdout_block_list == train_videos`. Therefore R0 must materialize evaluator blocked IDs from
  `holdout_block_list`; otherwise all holdout GT/predictions are removed and the finalizer rejects.
- Exact repair `f90595d8620e42e8e3d74722f2ab48126c6b65f2` changes that one source, flips the false-positive
  launcher assertion and adds a real-split semantic test for blocked and target identities.
- Remote clean snapshot `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_f90595d_20260722`
  passed affected DUCA `168 passed, 2 skipped`, mandatory C3/ASFormer `23 passed`, compile/bash/
  HEAD/clean checks and `PRECHECK_ONLY=1` without `sbatch`. New manifest SHA-256 is
  `14f345dc53b246b036ba1c80c993454c0d83a1173aae8481b54ac0f8647c8a2c`.
- New no-context MAX `019f8701-edaa-7e83-a572-49024b524098` is reviewing the exact successor.
  Status remains `linux_tested_pending_independent_max`; no R0 headroom or mAP exists.

## 2026-07-22 07:54 R0-only formal deployment

- The new MAX returned `HOLD_FIX_REQUIRED` but explicitly authorized R0 and rejected downstream
  CUDA/P0/official-60 unlock. No P0 method error, leak or R0 feasible-space mismatch was found.
- R0-only Job `1179392` was atomically reserved and recorded without submitting downstream jobs.
  Exact run root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_f90595d_r0_formal_20260722_0753`.
- Submission manifest SHA-256: `d22bd987f2a706d3c0e695f8bc02e58b35afb21f006dde3b148bee2f6b29dd7b`;
  R0 sbatch SHA-256: `837d4612332442b9bc05bf65868da6f2d5726bc93ce174a49229b7be2652ec0f`;
  R0 journal SHA-256: `346961608b47b3ef0daa47a60fc89f8825f84fdba2e2f6a19973c589d1dbc6f5`.
- At 07:54 +08:00 the job is `RUNNING` on `g0006`, with empty stderr. No headroom or mAP exists yet.
- P0 and official-60 remain `hold_fix_required`; their evidence-contract fixes run in parallel and
  cannot alter this immutable R0 commit or output.

## 2026-07-22 09:12 corrected R0 review restart

- Beijing 09:00 was reached without corrected R0 detector mAP. The missed
  deadline is recorded rather than backfilled with static tests or the failed
  Job `1179392`.
- Exact candidate remains
  `22555a4e830ce24f9bb516897b1bb7f44b70c188`; no model, K/G, objective or
  detector change has been made since its remote replay.
- Reviewer `019f8743-aed1-7a80-a7d6-552b08491019` was shut down after failing
  to return a verdict. Replacement independent MAX
  `019f875f-1668-7e51-bf97-1f565b25e106` must separately authorize R0-only,
  P0, full-model gate and official-60.
- Rule-300 propagation is being implemented in parallel by bounded worker
  `019f8766-c4db-7e30-8fc8-265d85d83b07`: only the R0-selected projected
  family may be mandatory downstream. This is execution-policy work and
  cannot change or substitute the R0 mAP decision.
- Current experiment status remains `r0_exact_quota_reaudit_pending`; no
  corrected R0 job, P0 job or official-60 job is queued.

## 2026-07-22 09:37 independent MAX verdict on `22555a4`

- Reviewer `019f875f-1668-7e51-bf97-1f565b25e106` returned HOLD independently
  for R0-only, P0, full-model gate and official-60.
- The exact-quota feasible set was not rejected: exact-K, endpoint centers,
  joint radius quota, bilateral support, projected physical path/max-gap,
  unrestricted semantics, integrality and zero MIP gap were verified.
- R0-only is held solely because the final objective is not a strict unique
  lexicographic encoding for every equal optimum. This is a reproducibility
  blocker, not evidence against G2, R2Q3/R4Q5 or boundary clustering.
- The only allowed R0 change is a final per-position lexicographic pin that
  preserves the optimum uniform overlap and position-sum values. New tests
  must cover tied optima, repeated solves and the old real failure replay.
- P0 remains held for R0 headroom, selected-family propagation and training
  consumer revalidation of the official-ASFormer source hash. Full-model gate
  remains held for independent arm artifact/content consumption. Official-60
  remains held and may eventually run only matched U plus R0-selected G0.

## 2026-07-22 10:08 deterministic R0 solver fix

- Independent worker commit `e267e1f9562c91fc0ad9a60382eb829d82d41acd`
  was cherry-picked to canonical HEAD
  `c418a951a9b9b7f7f19df785ead8642a4205c804`.
- The solver now pins optimum uniform overlap and position sum before a
  block-wise lexicographic pin. This closes the exact MAX reproducibility
  blocker without changing the feasible set or method.
- Solver/GT focused tests report `24 passed`; mandatory C3 reports `23 passed`;
  the canonical tied/repeat test reports `7 passed`; pycompile passes.
- No R0 job is authorized yet. The old 124-window production replay, remote
  Linux regression and a new independent MAX must consume the final combined
  exact commit before R0 submission.

## 2026-07-22 11:46 exact-d9 R0/P0 execution state

- Corrected R0 Job `1179517` is running on `g0048`; family construction is
  complete and frozen official-AdaTAD replay has reached `R2Q3`. There is no
  stderr/error signature and no sealed mAP yet.
- P0 Job `1179533` is queued with exact dependency `afterok:1179517`. It was
  appended through the production atomic journal; jobs SHA-256 is
  `cbd7f59a94eb472daf5d94df5728822d59eed6890e0e9140d97774952148a750`.
- This is queue-overlap, not a gate bypass: P0 revalidates the sealed R0
  summary and cannot run when the R0 finalizer returns KILL/nonzero.
- Gate and official-60 remain unsubmitted. Current status is
  `experiment_running_r0_with_fail_closed_p0_dependency`.

## 2026-07-22 12:13 R0 raw frozen-detector mAP available

- Exact commit: `d9fb398578716d278e818745677a92976bcedf2c`; Job `1179517`; run root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_d9fb398_r0_formal_20260722_112357`.
- On the sealed 40-video training-internal holdout, frozen official AdaTAD produced Avg-mAP
  `93.587070` for exact-uniform, `94.190497` for projected R2Q3, `93.999241` for projected R4Q5,
  and `93.970057` for unrestricted GT Oracle. R2Q3 has the largest raw headroom over uniform:
  `+0.603427` percentage points; its mAP@0.6 gain is `+1.192729` points.
- These are privileged internal-holdout reachability numbers, not THUMOS test-set paper results and
  not evidence that a learned selector reaches the Oracle.
- The preregistered 1000-resample paired video-cluster bootstrap is still running; therefore
  `r0_bootstrap.json`, `r0_summary.json`, the CI-based GO/KILL decision and the unique downstream
  family remain pending. P0 Job `1179533` remains `afterok:1179517` and cannot consume unsealed data.

## 2026-07-22 16:55 full R0--R5 DAG and bounded bootstrap acceleration

- Exact production commit `e49ef69605e1f98a7217957483f93a8a64bfc348` is pushed and deployed from
  clean snapshot `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_e49ef69_20260722`.
- Formal roots share prefix
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_20260722_155037`.
  Accepted jobs are R0 `1179795`, P0 `1179796`, full-model gate `1179797`, U/G0
  `1179798/1179799`, R3 aggregate `1179825`, R4 `1179826`, TemporalMaxer gate `1179827`, R5
  bundles `1179861--1179864`, and R5 cost/final aggregate `1179865`.
- This covers all 24 terminal cells: ActionFormer/TemporalMaxer x uniform/learned x K384/K256 x
  seeds 3407/5801/8123, plus eight matched plugin full-stack profiles and one dense-768 profile.
  Site-limit duplicate submissions `1179828--1179833` never ran and were canceled; no cell was removed.
- Job `1179795` is currently running without Traceback/OOM/non-finite/FAIL. The four raw frozen-detector
  point estimates exactly match the prior d9 replay; the serial 1000 x 4 official-evaluator bootstrap
  remains the only active R0 stage and all downstream jobs remain correctly pending on `afterok`.
- Statistics-only successor `9ed10139317c4196072d471ced883eb1dfc31703` parallelizes the identical sealed
  bootstrap sample sequence. Clean remote evidence is R0 `35 passed`, mandatory C3 `23 passed`,
  compile/bash/HEAD/clean. Real-data benchmark Job `1179956` is running from root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9ed1013_bootstrap_benchmark_20260722_165328`.
  No formal job has been canceled or duplicated while this bounded benchmark is unresolved.

## 2026-07-22 R0 absolute-mAP invalidation and evaluator reclassification

- The 93--94 values are not comparable with historical exact-uniform 64--65. R0 evaluates only 40
  videos from the THUMOS `training` subset, whereas the paper-like baseline evaluates the full
  `validation` subset. The exact-uniform policy itself scores `93.587070` under R0, so the large
  scale jump exists before any learned or Oracle allocation is applied.
- More importantly, R0 reuses the CellCF `transition_beta0` terminal detector checkpoint trained
  with the inherited full `training` subset and no R0 holdout block list. The frontend split was
  created later for family selection. “Frozen during replay” therefore does not establish detector
  independence from the 40 videos; absolute R0 mAP is detector-seen/training-internal evidence.
- The point estimate calls OpenTAD's official mAP implementation and tIoU 0.3--0.7, but the data
  split and the 1000-resample paired video bootstrap are custom. This is fair only for a paired
  within-R0 comparison among U/R2Q3/R4Q5/unrestricted under the same checkpoint. It is not fair
  against official-validation baselines, prior paper results, or other detectors.
- Verdict: R0 is reclassified as `diagnostic_protocol_contaminated_for_absolute_map`. Its relative
  `+0.603427` pp R2Q3 observation may motivate a clean follow-up, but cannot unlock a paper claim by
  itself. Main evidence must come from standard full validation/test mAP on matched terminal
  checkpoints. A clean Oracle/headroom rerun must train the detector after fixing the split and
  exclude the same 40 videos, or be explicitly reported as privileged test diagnostic without
  hyperparameter selection.
- The serial bootstrap was an over-engineered internal confidence gate. Official paper evaluation
  does not require 4000 evaluator invocations. Multi-seed standard evaluation is the primary
  uncertainty evidence; any paired bootstrap is optional secondary analysis and must not block the
  model-training critical path.

## 2026-07-22 official metric audit and independent replacement

The formal comparison is now exactly four fixed-K384/G2 arms at commit
`2bc6ca6fcf34f3e980437b5b830cabeef0de63c0`:

| Job | Arm | P0 | Formal metric | Dependency |
|---:|---|---|---|---|
| 1180075 | exact-uniform | none | full validation, epoch-59 EMA | none |
| 1180076 | Gaussian G0 | fixed epoch 19, train only | full validation, epoch-59 EMA | none |
| 1180077 | boundary-burst R2Q3 G0 | fixed epoch 19, train only | full validation, epoch-59 EMA | none |
| 1180078 | boundary-burst R4Q5 G0 | fixed epoch 19, train only | full validation, epoch-59 EMA | none |

All jobs were simultaneously RUNNING at 17:48 +08:00 and Slurm reported
`Dependency=(null)`. Three learned arms passed the frontend contract with
`detector_executed=false`, `detector_trained=false` and
`test_subset_consumed=false`; the uniform arm passed the official-60 contract.
No terminal mAP exists yet. R4 and R5 cannot delay this four-arm answer; R5 is a
later runtime-validated generalization matrix.

## 2026-07-22 gate failures and exact replacement queue

The original table above is superseded for execution, not for model identity.
Three uniform jobs failed before official-60 training while progressively
exposing production-only gate defects: `1180075` (ASFormer provenance wrapper),
`1180097` (container identity), and `1180106` (CPU/CUDA equality). They contain
no paper mAP and are not method failures. Learned jobs attached to those exact
bad gates were canceled to avoid wasting GPU time.

The exact replacement at `8d85929ea04dc40f1eb0c3cc806061ce3b071d3f` is:

| Job | Arm | Formal metric | Slurm dependency | Initial state |
|---:|---|---|---|---|
| 1180111 | exact-uniform | full validation, epoch-59 EMA | null | PENDING (Priority) |
| 1180112 | Gaussian G0 | full validation, epoch-59 EMA | null | PENDING (Priority) |
| 1180113 | boundary-burst R2Q3 G0 | full validation, epoch-59 EMA | null | PENDING (Priority) |
| 1180114 | boundary-burst R4Q5 G0 | full validation, epoch-59 EMA | null | PENDING (Priority) |

The generated audit binds full THUMOS validation, OpenTAD `mAP`, tIoU
0.3--0.7 and `epoch_59.pth/state_dict_ema` for every arm. Queue priority may
delay allocation, but no arm waits for another experiment or for R0/bootstrap.

All four jobs subsequently entered `RUNNING` on allocated GPUs. Uniform Job
`1180111` passed the real AMP/DDP/full-model gate with status
`p1_p2_exact_full_model_amp_ddp_gate_passed` and started the official-60
training process on the full protocol. The three learned arms independently
entered finite P0 training with fixed K384; the current error scan contains no
Traceback, OOM, non-finite value, ValueError or FAIL. This establishes runtime
execution only, not terminal mAP.

## 2026-07-22 Pro audit verification and experiment impact

The e49 exact-model Pro review is archived and independently checked against the
current 8d execution snapshot. H1--H5 are confirmed code/evidence gaps: soft rather
than hard bilateral burst geometry, pre-selector dense low-resolution decode/H2D,
no independent raw-prediction reevaluation in R5, unpaired cost ratios and an
insufficiently pinned dense-baseline identity. Relevant remote tests pass
`96 passed, 1 warning`; these findings are absent contracts rather than failing
existing tests.

This does not invalidate the currently running official-validation four-arm
comparison. It changes interpretation and the next decision:

1. `1180111--1180114` remain the shortest valid answer to learned selection versus
   exact-uniform at K384; no job waits on R0 or another arm.
2. Current cost claims are limited to post-decode heavy-backbone frame reduction.
3. H3/H4/H5 are evidence-only repairs before a future paper aggregate and do not
   require model retraining.
4. H1 is not automatically converted into a hard mandatory decoder. First collect
   center error, bilateral/quota satisfaction and soft-to-hard mismatch from the
   running learned arms.
5. Full R5 expansion remains blocked on a credible terminal official-mAP signal.

Status remains `experiment_running`; the audit supplies no terminal mAP and does
not establish efficacy or paper readiness.

## 2026-07-22 final parallel R0--R5 implementation and CUDA correction

- Final exact candidate: `cd68d89dcc0854baa3c0107607086e801509b552`, clean remote snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_cd68d89_20260722`.
- R0/R1, R2/R3 core, R2/R3 adapted, R4 R2Q3 and R5 are implemented as five independent Slurm
  bundles. R5 contains 24 official-mAP cells (two backends, two arms, K384/K256, three seeds),
  four same-backend ActionFormer paired cost profiles, and a minimal performance/cost plotting CLI.
- The first `2645e68` dispatch was a runtime diagnostic only. Its soft-detached P0 arm reached the
  real loader/model/DDP/CUDA path but failed before an optimizer update because PyTorch 2.0 CUDA
  could not apply scalar advanced assignment to a two-dimensional row slice in
  `build_boundary_burst_utility` (`5920 vs 1184`). It produced no mAP and is not method evidence.
- The row reset is now expressed with dimension-explicit `index_fill_`; old Jobs `1180326--1180331`
  were canceled. Replacement Jobs are `1180336` R0/R1, `1180337` R2/R3 core, `1180338` R2/R3
  adapted, `1180339` R4, `1180340` R5, and `1180341` aggregate (`afterok:1180340`). Formal root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_cd68d89_parallel_20260722_205506`.
- At the latest check all replacements are Slurm-accepted and `PENDING(Priority)`. Implementation
  and deployment are complete; empirical efficacy, cost savings and paper readiness remain unproven
  until terminal official-validation EMA mAP and paired runtime artifacts exist.
