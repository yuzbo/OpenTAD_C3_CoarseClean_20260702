---

## 2026-07-28 SparseHead consolidation guardrails

- `OpenTAD_C3_CoarseClean_20260702` 是唯一可写 SparseHead/PhysTime 研究面；
  `OpenTAD_SparseHeadClean_20260702` 只读封存。不要在旧仓继续 repair、训练或写 claim。
- 两仓没有共同祖先。禁止 whole-tree merge/cherry-pick，也禁止把旧仓 16 个 repair
  configs 与 6 个 remote launchers 原样复活。
- irregular bridge/balanced repair 只允许 assignment 诊断；不得删除
  `diagnostic_only`、`primary_result_allowed=False`、`metric_claim_allowed=False`
  或 Slurm fail-closed。
- native-J192 physical-metric 是当前 matched survivor；SDPQ 是唯一结构候选，但
  20-epoch `30.88` 远低于 physical-metric `44.88`。不得用“唯一代码路线”偷换成
  “SDPQ 已被实验证明”。
- full60 `41.28 -> 57.61` 只证明 physical-metric 相对 selected-axis 的单种子优势，
  不含 SDPQ，不是多种子/跨数据集/成本闭环，也不是 `paper_ready`。
- 不要重启 observation-coupled G1a rank-assignment 或把 v1 PhysTime 调参复活；
  它们分别有短动作 assignment 失败和多变量负结果。
- decode-cross 的 source-dtype evidence chain 已有 Linux CPU `59 passed`，但没有
  四条件 CUDA gate/job/result；不得声称 cross-decoder confound 已关闭。
- 不得删除 physical config 对 `phystime_g1a_axis_positions_sec`、native count 和秒域
  起止的显式消费，也不得悄悄退回 `selected_dense_indices` 后仍声称输出在 seconds
  轴。该缺口已被真实远端测试捕获。
- 排序/top-k 使用的 `cls_scores` 禁止在存档前从 FP16 扩展到 FP32；captured native
  proposals 只能作审计参照，不能覆盖重建 proposal。
- 不要复活 owner manifest、`jobs.tsv`、scheduler snapshot 或自动提交重试框架；
  当前 suite 的职责只是验证显式 evidence artifact。
- `_05/_06` 远端目录是实现失败诊断，不是实验结果；只有 `_07` 达到 CPU focused
  全绿，且它仍不能替代 real CUDA gate 与 exact-commit 绑定。SDPQ 未先超过
  physical-metric matched medium gate，不得启动 full60。
- Job `1201048` 已 `FAILED 1:0`，失败签名
  `actionformer_native_temporal_geometry_constructor_contract_v1`；v4 根和日志不得
  删除或覆盖。工程故障不设固定修复次数上限；若签名重复，必须重新深入定位并产生
  有证据的新修复，禁止原样重提相同作业。
- v5 Approach A 恢复部署是 Job `1201317`，运行根后缀
  `phystime_decode_cross_approach_a_20260728_v5`，精确 runtime commit/tree 为
  `0338f4777bd02fb327573ef716f54fec76d4af0e` /
  `cb98c64c17d2983c22181d4908c4f31024a82a2f`。它已经失败，旧根与日志不得覆盖。
- 不得通过把 G1a config 改成 `PhysTimeTAD` 修复该签名：full60/P0 checkpoint
  对应 native-J192 `ActionFormer + Conv1DTransformerProj + FPNIdentity +
  ActionFormerHead`。`PhysTimeTAD` 的 constructor/projection/head/checkpoint API
  不兼容；正确恢复是历史 ActionFormer native geometry 对齐合同。
- v1--v3 都是零作业部署诊断，分别是 PowerShell stderr 截断、缺少 repo-root
  `PYTHONPATH`、以及误用通用 raw 路径；不得将它们当作运行或模型负结果。
- P0 冻结 THUMOS 视频目录是 `/data/run01/sczc063/yuzibo/thumos14/train` 与
  `/data/run01/sczc063/yuzibo/thumos14/test`，不是通用 `/raw/...` 目录。未来
  preflight 不得为了“通过”而改用未经 P0 manifest 绑定的数据。
- Job `1201317` 在同一个 allocation 中串行运行 gate → 四 replay → suite，
  dependency 为空是设计结果，不是漏依赖；脚本 fail closed，gate 未全通过时任何
  replay/suite 都不得开始。
- Job `1201317` 的四条件 real-CUDA gate 已通过：`gate_pass=true`、
  `all_native_direct_exact_equivalence=true` 且四项 raw tensor immutable。不要
  重跑或另提交 gate；但在四个 formal completion 与 suite verdict 齐全前，也不得
  声称 decoder confound 已关闭。
- 非模型原因的路径、环境、导入、启动器、资源、manifest、收据或实现正确性故障，
  必须保全旧根、加复现测试、重新 preflight 并使用全新 clean commit/run root 持续
  修复，直到获得完整最终性能；“无次数上限”不等于允许无根因的无限重复提交。
- 合法模型负结果不得伪装为工程故障。必须先报告全部原始数值，并相对 matched
  controls/full60/P0 分解 tIoU、类别、时长、边界、recall、校准、NMS、assignment、
  support、native geometry、decoder regret、online/EMA 与成本；至少形成两种竞争
  解释、反证、可证伪预测和最小决定性实验，再讨论路线继续、修改或否定。不得为了
  转正而静默改阈值、超参数、数据、checkpoint 或 evaluator。

### 2026-07-29 v5 artifact failure -> v6 recovery

- Job `1201317` 的 selected-online direct inference 虽显示 Avg-mAP `41.26`
  及 mAP@0.3--0.7 `64.50/56.39/42.66/27.82/14.90`，但 producer 没有生成
  `pre_cross_window_detections.json.gz`，随后 fail closed。失败签名固定为
  `direct_postprocessing_artifact_producer_contract_missing_v1`；其余三 replay
  和 suite 未开始。不得把它解释成模型负结果。
- v5 的 partial direct 指标只能标记为 `diagnostic_only`、
  `primary_result_allowed=False`、`metric_claim_allowed=False`；不得复制进正式
  结果表、四条件比较、abstract 或 claim。
- 唯一活动部署是 Job `1201469`，run root 后缀
  `phystime_decode_cross_approach_a_20260729_v6`，clean runtime commit/tree
  `ac326ffdc97652433b55ccc596e734b112f51806` /
  `0c58027756997995bda0de6fdd8ec0deb49966d3`，preflight SHA-256
  `97fe5af28b2647396c052c9bdf956997d98e264af74432b57e0fc983b071fb91`。
  它 pending/running 时禁止重复提交。
- v6 只恢复已声明的 direct-artifact producer 合同并新增回归；Linux focused
  suite `75 passed`。不得趁修复静默改变模型、配置、epoch-59 checkpoint、
  seed42、数据、评价器、阈值或证据边界。
- v6 在四条件 gate、四个 completion、suite 与终态指标齐全前只能记
  `experiment_running`。如果同一签名重现，必须重新深入分析并产生新的证据化
  修复，禁止原样提交 v7。
- Job `1201469` 的 v6 四条件 real-CUDA gate 已通过：
  `gate_pass=true`、`all_native_direct_exact_equivalence=true`、四项
  `raw_tensors_immutable=true`。不要重复运行或另交 gate。
- v6 `selected_online` direct inference 已完成并实际生成 pre-cross/audit/
  metrics/result 四类合同产物，证明
  `direct_postprocessing_artifact_producer_contract_missing_v1` 的修复越过原失败点。
  但 direct 指标仍不等于双轴 replay completion；当前 replay 正常运行不构成失败，
  仍必须等四份 completion 与 suite。

### 2026-07-29 v6 validator failure -> v8 recovery

- Job `1201469` 已 `FAILED 1:0`。selected-online replay producer 的双轴指标
  均已生成，但 validator completion assembly 因未绑定 `numeric_precision`
  失败；签名固定为 `decode_cross_validator_numeric_precision_scope_v1`。不得把
  validator 故障或单条件 `+8.89694669029758 pp` 伪装成四条件正式结论。
- v6 producer completion、uniform/native 与 physical-time 数值只能保留为
  `diagnostic_only`、`primary_result_allowed=False`、
  `metric_claim_allowed=False`；另三条件和 suite 没有运行。
- v7 的修复 commit/tree 为 `1631d0b60f6552a6f5eb0378d74e766850f34ffd` /
  `f485c8708e22bbbf9a73063d5293a20bc4aa658f`。其部署元数据把 focused-log
  SHA-256 末位 `6` 截断，门禁在 `sbatch --test-only` 前拒绝；签名
  `deployment_expected_sha256_truncation_v1`，零 Slurm 作业。不得覆盖或把 v7
  记作正式实验。
- 唯一正式后继是 v8 Job `1201495`，`1201494` 仅为 test-only。v8 run root
  后缀 `phystime_decode_cross_approach_a_20260729_v8`，branch
  `codex/sparsehead-evidence-recovery-20260729-v8`，同一 exact commit/tree，
  `76 passed` 且 preflight SHA-256
  `e9f36c221156e5411dad5e3bfe43508b4aa59310539fdbe24da985fc99a27d53`。
  Job pending/running 时禁止重复提交。
- 该修复不授权改变模型、配置、epoch-59 checkpoints、seed42、数据 manifest、
  evaluator、门槛或证据边界。只有四份 formal completion、explicit suite 与
  terminal metrics 齐全后才能判断模型结果；当前状态仅
  `experiment_running`。
- v8 四条件 real-CUDA gate 已通过，artifact SHA-256
  `5e323e5ccdedd7dd39d70148aed7108beca94bb5952125a124ad20accfd634f6`；
  gate 不需要另交作业重跑。当前 selected-online direct inference 正常运行，
  排队/运行不是失败，也不得重复提交。
- v8 selected-online direct 已完成并产生绑定 exact commit/tree 的 211-video
  pre-cross/audit/metrics/result；当前 physical-time replay 正常生成中。direct
  Avg-mAP `0.4125660433077075` 仍不等于 formal completion，禁止据此停止作业、
  重提作业或提前写路线 claim。

## 2026-07-22 corrected R0 execution identity

323. The only active corrected boundary-burst experiment is Job `1179517`,
     exact commit d9, run root ending `_20260722_112357`. Never merge it with
     failed old Job `1179392` or the deterministic replay directory.
324. Its journal intentionally contains only the authorized R0 role. No
     absence of P0/gate/official rows may be treated as an incomplete main DAG;
     those stages remain explicitly unauthorized.

## 2026-07-22 d9 staged MAX verdict

320. Independent MAX grants R0-only GO on exact d9. This permission cannot be
     widened to P0, a real-model gate or official-60 before positive corrected
     R0 headroom and a sealed selected family exist.
321. Do not repair future official scheduling/provenance findings by changing
     R0 model geometry or delaying the authorized R0. The findings are: four
     sentinel GPU jobs can block aggregate, and terminal pretrain identity is
     not rebound to the sealed P0/full-gate path.
322. Future main official submission must schedule only U plus selected G0.
     Unselected-family diagnostics must be separate and nonblocking; an
     in-job `exit 0` after GPU allocation is not equivalent.

## 2026-07-22 exact d9 guardrails

316. The current candidate is exactly
     `d9fb398578716d278e818745677a92976bcedf2c`, not `22555a4`, `c418a95` or
     `ae0bc73` in isolation. Never submit or compare a mixed snapshot.
317. Real 124-window repeated replay is deterministic: both family JSONL files
     have SHA-256 `b49e03c...edd6d7`. Different summary hashes are expected
     because summaries contain different output paths; do not reopen the
     solver-tie issue from summary-byte inequality.
318. Static tests and deterministic replay are not R0 headroom evidence. A
     fresh independent MAX must explicitly authorize R0-only before Slurm;
     P0, full-model and official-60 remain separately blocked.
319. R0's sealed weakest passing projected family is the only mandatory
     learned family downstream. Gaussian and the unselected projected family
     are nonblocking diagnostics and must not recreate a four-arm default.

## 2026-07-22 `22555a4` MAX guardrails

313. Independent MAX `019f875f-1668-7e51-bf97-1f565b25e106` placed exact
     commit `22555a4e830ce24f9bb516897b1bb7f44b70c188` on HOLD for every deployment
     stage. It verified the intended exact-quota feasible-space mathematics;
     the only R0 blocker is strict deterministic tie resolution. Do not change
     K/G, quota, bilateral semantics, objective priority or the detector to
     answer this HOLD.
314. The exact solver repair must freeze the optimum uniform-overlap value and
     optimum position-sum value before a final per-position lexicographic pin.
     A weighted approximation without proof, solver seed, tolerance relaxation
     or repeated-run coincidence is not a deterministic contract.
315. P0/full-model/official-60 remain blocked independently of the R0 solver:
     selected R0 family propagation, official-ASFormer training-consumer hash
     revalidation and arm gate artifact/content reopening must close first.
     Diagnostic Gaussian or unselected burst failure cannot veto U/selected-G0,
     and no four-arm default long run is allowed.
type: anti_repetition
updated: 2026-07-21
---

## 2026-07-21 selected-axis official-60 runtime guardrails

159. In the `cb89586` official-60 logs, `duca_schedule_progress=1.0000` is
     the legacy outer loss-schedule progress and is **not** the hard-policy
     mixture coefficient. Read policy alpha from the frozen
     `policy_alpha` entry: homotopy stays at zero for 300 successful updates
     and then follows the 1800-step cosine transition. At step 350 the
     expected alpha is only about `0.0019`.
160. A phase transition in the log proves schedule execution, not that hard
     selected positions have already changed. Hard-policy overlap with exact
     uniform must be measured from a hash-bound checkpoint/selector export.
161. Jobs `1177734-1177737` are immutable pre-model protocol-routing
     failures. The admissible running replacement is commit `cb89586`, gate
     `1177776`, and Jobs `1177779-1177782`; never merge their status or results.
162. Early finite loss, exact update counters and a passing CUDA gate do not
     establish `Avg-mAP > 65`. The primary comparison remains terminal
     epoch-59 EMA under the matched four-arm protocol.

## 2026-07-21 Protected-E2E deployment guardrails

154. The SSH account label `BSCC-N16R4` is not the Slurm cluster name. Use
     `--clusters=n16r4`; N16R4 one-node GPU jobs require `--gpus=1`, not
     `--gres=gpu:1`.
155. A rejected `sbatch` call is not a submitted experiment. Require a
     positive Job ID, committed `jobs.tsv`, and a matching live/accounting
     record before changing status to `experiment_running`.
156. `AssocMaxSubmitJobLimit` is an external queue-capacity block, not a model
     or numerical failure. Never cancel unrelated ChronoTransport, screening,
     or user jobs to make room without an explicit route decision.
157. Under a one-slot quota, the serial gate launcher may execute main, rho,
     P3 short/medium/long, and completion in one fail-closed GPU allocation.
     This changes scheduling only; it must reuse the same wrappers, P0 hash,
     component order, artifacts, and authorization checks.
158. The abandoned six-job roots created before `jobs.tsv` are deployment
     audit history only. They contain no admissible CUDA or P3 evidence.

## 2026-07-20 pre-backbone 论文准备度 guardrails

146. Do not infer correct backbone time semantics from a correct physical
     detector head. Nonuniform frames are packed by selected rank into
     VideoMAE clips; P1 must test chunk/feature/mask parity and timestamp
     spacing before long training.
147. Do not call fixed `K=384` dynamic-budget computation. It is
     content-adaptive placement under a fixed heavy-backbone budget.
148. Do not call the method low-cost from backbone frame count alone. Full
     cost includes dense decode, preprocessing, H2D, the 768-frame coarse
     probe, selector, backbone, head, NMS, memory, energy and training
     overhead.
149. Same hard/soft DAG and exact-hard forward do not establish a valid
     straight-through gradient. P3 hard-swap alignment remains mandatory.
150. Physical probability-floor smoothing is not a hard coverage guarantee.
     Attribute hard coverage only to the exact-K physical feasible graph and
     its audited max-gap constraint.
151. Do not claim the whole coarse module is binary-supervised. Only the
     action head has that contract while the shared ASFormer trunk also
     receives transition/boundary auxiliary gradients.
152. Do not claim first frame selection for action detection, first
     task-aware video selection, or first physical-time TAD. The defensible
     novelty, if results pass, is the bounded combination of offline
     fixed-budget physical exact-K acquisition, same-graph hard/soft learning,
     protected gradient ownership and full-stack TAD evaluation.
153. Do not patch the current zero-shot/teacher/HardTopK paper into a claimed
     Protected-E2E result before the evidence closes. Its method story and
     empty experiment tables require a full route-aligned rewrite.

## 2026-07-20 Protected-E2E freeze guardrails

137. Do not call the protected-E2E route implemented, tested, running or
     supported before P0-P3 pass on one exact commit.
138. Do not infer detector-to-selector gradient from a total-loss backward.
     Backward detector, action-BCE and transition losses separately and record
     parameter-group gradient ownership.
139. The main protected arm must keep detector gradients out of both the action
     head and shared ASFormer trunk. A small coarse-trunk gradient scale is a
     separate fourth arm, not the default method.
140. Do not revive `structured_zero_forward` or any old bridge merely by
     changing a config weight. A new bridge must pass real hard-swap
     finite-difference alignment through the official detector.
141. Do not start official-60 until exact-K, uniqueness, physical-gap,
     hard-forward equality, optimizer coverage, official-backend identity,
     no-leak, AMP/DDP and hard-soft alignment gates all pass.
142. Before the four fixed-K official-60 arms finish, do not submit dynamic
     budget, X3D, SlowFast, MobileNet, other K values, detectors, datasets,
     seeds or replacement model ideas.
143. Final mAP is the route decision metric. Detector loss, actionness quality,
     boundary coverage and gradient alignment are mechanisms and gates, not
     substitutes for mAP.
144. Keep the design distinction explicit: the coarse action head is trained
     only by binary actionness, but the shared ASFormer feature trunk currently
     also receives transition/boundary auxiliary gradients. Do not claim the
     entire coarse module is binary-supervised only; if that stricter contract
     is required, it must be a separately frozen and tested routing change.
145. Do not call the new `DucaProtectedE2EFrameSelector` integrated or P1/P2
     passed from local compile or authored tests. Its remote focused test,
     official full-model gate, native-head contract and optimizer/RNG audits
     are still required.

## 2026-07-19 CellCF KILL and CARA redesign guardrails

117. Never call one-frame-per-exact-uniform-cell CellCF boundary-adaptive
     allocation. It cannot release a background cell's quota or assign a
     second observation to a high-transition cell.
118. Do not interpret current CellCF's actual gathered frame as the detector's
     physical timestamp. Commit `1642f26` deliberately uses uniform anchor
     `detector_grid_positions` and selected-axis GT remapping.
119. Detached hard counterfactual utility is detector-derived supervision, not
     direct detector-loss backpropagation. Keep C4 wording and diagrams exact.
120. Do not call existing finite-candidate or heuristic GT diagnostics an
     exact feasible-family oracle. Exact geometric optimization and solver
     status are required; detector mAP is only a secondary frozen-checkpoint
     diagnostic.
121. `DUCA-CARA`, `G=3`, the `192+192` split, all review-supplied weights,
     cadence, thresholds and schedules are proposals. None is implemented,
     tested, empirically supported or paper-ready.
122. A max-gap proof on compact valid-order indices is not a physical-time
     guarantee when valid positions are non-contiguous. Freeze the coordinate
     unit and recompute the actual physical maximum hole.
123. Do not launch another long DUCA train before the allocation-family
     ceiling and physical-coordinate gates show that the new feasible set can
     increase boundary density and release background quota.
124. Kill only the current CellCF adaptive-allocation claim. The broader
     coarse-state-change indirect-selection program remains a redesign
     question until the bounded new-family gate is resolved. Dynamic MUST
     stays frozen.
125. The next CARA family must contain exact uniform as an explicit feasible
     member while allowing cross-region residual quota transfer. Otherwise a
     learned result cannot be attributed cleanly against the uniform anchor.
126. Do not tune dense-index `G` and then describe it as an original-frame
     interval. Freeze and record the physical coordinate conversion before
     the ceiling audit.
127. Start new bounded work from `4ce69c8`; it descends from immutable model
     commit `1642f26`. Do not mutate or relabel the completed CellCF arms.
128. Do not implement fixed `192 scaffold + 192 residual` after declaring a
     maximum interval of 15 original decoded frames. On the formal stride-4
     grid that contract has effective cap 12; a fixed scaffold needs at least
     255 arbitrary positions or 382 positions if it must be an exact-uniform
     subset.
129. The earlier phrase "10 or 15 frames" does not itself freeze a coordinate
     unit. Record whether it means dense candidate index, decoded source-frame
     index or seconds before solving or naming a max-gap result.
130. Always report dense unselected hole, original-frame interval and seconds
     interval separately. Never convert `max_unselected_hole=G` into a source-
     frame claim without the actual per-sample decoder coordinate map.
131. The 192-anchor dense-hole-3 parity-switch construction uses zero-based
     anchor ordinals `{1,3,...,191,192,194,...,382}`. Do not publish or test it
     with an unspecified indexing convention.
132. Treat global exact-K/physical-gap family D as a ceiling family, not an
     implemented final selector. Privileged GT headroom must be followed by a
     separate deploy-visible recoverability and full-stack cost test.
133. The 2026-07-20 Pro response's DP, CP-SAT and test snippets are proposals.
     They are not integrated or tested; the shown canonical GT oracle also
     omits declared distance, short-action and background objectives.
134. If floating selection scores are integer-quantized, exact DP language is
     exact only for the frozen quantized vector. Serialize the scale and input
     hash and do not imply exact preservation of arbitrary float ordering.
135. Current `val` and `test` both consume the THUMOS validation subset.
     Different overlap ratios do not create an independent held-out split.
     Freeze family/cap/objectives on training-side data and use one terminal
     validation protocol without post-hoc tuning.
136. Do not freeze paired-bootstrap, simultaneous-CI, MDE or seed-count gates
     until pairing, multiplicity correction and the meaning of seed variance
     are explicit. Seed-0 cannot estimate `sigma_seed`.

## 2026-07-18 CellCF cost-recovery guardrails

112. Jobs `1167485/1167486` are no longer running: the original cost job is
     `FAILED/1:0` and the original completion job is cancelled. Never rewrite
     them, replace their IDs in the original ledger, or call them successful.
113. Preserve `cost_recovery_5ab3042_v1` and
     `cost_recovery_67a8a0a_v1`. The first exposed N16R4's requirement that
     every one-node job request a GPU; the second exposed delayed `sacct`
     `SubmitLine` visibility. Their cancelled Jobs `1170338/1170354` are
     deployment diagnostics, not cost evidence.
114. A held-job accounting retry may repeat the same strict validation only.
     It must not relax exact `--hold` SubmitLine, current `JobHeldUser`,
     dependency, scheduler-owned script hash, commit or token checks.
115. Recovery Jobs `1170366/1170367` are now failed/cancelled diagnostics.
     The profiler and strict summary schema disagree on seven
     `*_cpu_enqueue_ms` fields. Do not resubmit before fixing and testing that
     producer-consumer schema contract on the exact evidence commit.
116. CellCF-versus-bare-uniform frontend timing is an overhead/lower-bound
     comparison. It is not evidence of savings over a dense-768 full stack.

## 2026-07-17 CellCF terminal-result guardrails

109. Do not say CellCF beats transition-beta0: its raw terminal Avg-mAP is
     `64.0610`, below transition-beta0 `64.2755` by `0.2145` points.
110. Do not turn transition-beta0's one-seed `+0.4161` point raw gain over
     exact-uniform into a robust claim before the six-job DAG, external seal
     and bounded repeat decision are complete.
111. Do not call the original six-job suite complete: Job `1167485` failed and
     Job `1167486` was cancelled. Completion now requires the separate
     hash-bound recovery and external seal, never a reconstruction.

## 2026-07-17 CellCF profile and evidence guardrails

104. Never claim that the `2a0f848` evidence-tooling commit was used to train
     the immutable `1642f26` formal suite. It may only reopen and analyze
     hash-bound artifacts after they exist.
105. Never infer a paper cost claim from reconstructed summaries. Preserve and
     hash raw per-sample full-stack timings and raw `sacct` output, then replay
     every parsed field before accepting the summary.
106. Never mix `exposure132` and `official60` receipts, job names, manifests,
     terminal checkpoints or expected successful updates. The training profile
     must be explicit at config, prepared-suite, submitter and artifact levels.
107. A submit precheck must reopen both profiles without calling `sbatch`.
     Passing that precheck is implementation evidence, not authorization to
     launch official-60 training before the 132-epoch GO/KILL result.
108. On n16r4, set `PYTHONNOUSERSITE=1` for exact validation. The user site
     currently contains NumPy 2.x while the formal environment contains
     NumPy 1.23-compatible OpenCV; an import failure from that collision is an
     environment failure, not a model regression.

## 2026-07-16 CellCF evidence-DAG guardrails

96. Do not reuse old commit `475634e` gate/pilot Jobs `1167145/1167146` for
    replacement commit `3a0f5ae`; the full exact-commit gate chain must rerun.
97. Do not call the three-arm aggregate a completed suite. Its only admissible
    status is `runs_complete_cost_pending`; final `complete` requires valid,
    reproducible trained-checkpoint cost evidence.
98. Do not reuse a static Slurm receipt without reopening job name, comment
    token, cluster and state through `squeue`/`sacct`. Only the specific normal
    `Invalid job id specified` live-queue transition may enter accounting;
    permission, connection and other query errors remain fail-closed.
99. Do not claim dense full-stack savings from CellCF-versus-bare-uniform cost.
    That pair measures frontend overhead/lower-bound distance only; a matched
    dense full-stack profile is still required for a savings claim.
100. Do not reuse failed real-gate Jobs `1167220/1167221` as CUDA evidence.
     They are immutable scheduler/environment diagnostics; the valid
     replacement gate is Job `1167222` with artifact SHA-256 `b128f587...c4334`.
101. Any Slurm gate script must have its interpreter on byte zero and must
     source `scripts/duca_cellcf_canonical_env.sh` with the frozen project
     `BASE`; a visible GPU alone does not prove the dataset/config binding.
102. Do not submit the formal three-arm suite merely because Job `1167222`
     passed. The exact-commit forced-overflow DDP pilot and bounded cost-path
     smoke must first close without fatal anomalies.
103. On n16r4 every generated DAG job, including aggregate and completion,
     must request one generic Slurm GPU. Never hard-code a physical GPU or
     overwrite Slurm's `CUDA_VISIBLE_DEVICES`; consume logical `cuda:0` only.
104. Keep the submitted dependency canonical as `afterok:a:b:c`. A live
     `squeue --json` dependency may be the strict repeated rendering
     `afterok:a(unfulfilled),afterok:b(unfulfilled)`. Accept only positive,
     unique IDs and exact `unfulfilled` annotations; reject other dependency
     types, `fulfilled`, duplicates and trailing text.
105. A live remaining dependency subset is not proof that removed predecessors
     succeeded. Require unique same-cluster `sacct` rows with `COMPLETED/0:0`;
     if the target started, also prove predecessor End <= target Start.
106. Jobs `1167469-1167471` and `1167475-1167478` are permanently invalid
     deployment diagnostics. They had zero useful runtime and must never enter
     an mAP, cost or paper table.
107. Do not independently revalidate a CellCF pilot before sourcing
     `scripts/duca_cellcf_canonical_env.sh`; config hashes include the canonical
     dataset/checkpoint environment. A mismatch without that environment is a
     verifier invocation error, not model evidence.
108. Evidence-handoff fixes invalidate previous exact-commit authorization.
     The current formal run is authorized only by commit `1642f26`, gate Job
     `1167479` and pilot Job `1167480`.

## 2026-07-16 DUCA Round-2 redesign guardrails

84. Do not substitute TAPOS for TAPS. In this project TAPS means the ACCV 2024
    temporal-attention pruning/scaling method; `arXiv:2005.10229` is a separate
    Temporal Action Parsing paper.
85. CellCF now has a local implementation and contract tests, but do not call
    it CUDA-tested, experiment-running, empirically supported, final or
    paper-ready before the exact-commit real-loader gate, forced-overflow
    three-arm pilot and matched terminal-EMA runs pass.
86. Do not reuse C4's direct-detector-gradient wording for detached local-flip
    policy utility. A new or revised claim requires formal claim audit.
87. Do not present the proposed loss weights, 20%/10% schedule, teacher cadence,
    EMA decay or numeric GO/KILL thresholds as uniquely correct or evidence-
    derived.
88. A preregistered seed-0 failure may stop the current Local-cell configuration
    and block unbounded extensions; it is not a scientific proof that every
    possible DUCA hypothesis is false.
89. Do not claim global adaptive budget allocation from one-per-uniform-cell
    deformation. The admissible claim is local semantic residual learning over
    exact uniform.
90. Do not permanently remove direct-boundary attribution if the final paper
    claims indirect state-transition selection is superior. It may be deferred
    until the minimal C3/utility screen passes.
91. Do not allow formal CellCF `--cfg-options` to mutate model, loss,
    optimizer, scheduler, dataset semantics, evaluation or postprocessing.
    Only audited runtime paths and exact evaluation booleans are allowed, and
    train/eval effective config hashes must be bound through finalization.
92. Do not trust a gate or pilot merely because its JSON says `ok=true`.
    Rehash the audited source/assets and revalidate the pilot manifest,
    contexts and raw training probes before suite preparation and training.
93. Do not submit CellCF formal runs from mutable external sbatch files unless
    the suite binds seed, commit, manifest, canonical environment and every job
    file by hash, and receipts preserve jobid/cluster/dependency identity.
94. Do not mark a CellCF suite complete from three post-run summary JSON files.
    Reopen and rehash the full terminal evidence chain, and require the
    checkpoint-bound frontend cost artifact in the fail-closed completion DAG.
95. Any code change that repairs CellCF evidence handoff invalidates earlier
    exact-commit gate/pilot authorization. Keep Jobs `1167145/1167146` as
    engineering diagnostics and rerun the whole DAG on the replacement commit.

## 2026-07-16 `7525efb` Round-1 audit guardrails

79. Do not expand `GO_TO_REAL_GATE` into permission for a pilot, full train,
    deployment, empirical support or paper claims.
80. Do not call `L_baseline-L_swap` pure frame-content utility. It is a
    selection-policy utility that includes selected-axis geometry, GT remap
    and renewed point assignment.
81. Do not accept all-zero or single-value utility as successful alignment.
    Require informative/nonzero and distinct-value counts before rank/sign
    gates are meaningful.
82. Do not use the synthetic formal gate as real-loader or DDP evidence. The
    current artifact explicitly records `real_dataset_loader_executed=False`.
83. Do not infer actual optimizer-step swap improvement from the score-space
    proximal identity alone. Shared parameters, other losses, AdamW state,
    momentum and weight decay remain outside that local proof.

## 2026-07-15 signed counterfactual utility guardrails

74. Do not revive candidate-only softmax ranking as signed detector utility.
    It cannot distinguish an all-harmful candidate set from an all-beneficial
    set relative to retaining the baseline selection.
75. Do not treat a no-op categorical class as proof of every candidate's local
    gradient direction. Shared removed positions couple pair scores; the
    current formal candidate uses the swap-incidence Gram system in commit
    `7525efb`.
76. Do not call `7525efb` deployable or empirically supported from focused
    tests. It still needs a clean exact-commit CUDA gate, real THUMOS loader
    gate, forced-overflow/mixed-batch pilot, and matched terminal-EMA mAP.
77. Never let the formal gate report a clean commit while executing dirty
    files. It must reject nonempty `git status --porcelain` and persist hashes
    for the audited implementation surface.
78. Do not call the current counterfactual route direct detector-gradient
    learning. The detector evaluates discrete hard swaps under `no_grad`; its
    detached loss reduction supervises a selector surrogate.

## 2026-07-15 DUCA successful-update guardrails

70. Formal DUCA P0 requires exactly 13,200 successful optimizer, LR, EMA, and
    selector-schedule updates. AMP-skipped attempts must replay the same
    materialized batch with RNG and all forward-mutated state restored.
71. Never accept a terminal metric JSON directly. Recompute the frozen
    OpenTAD mAP from the hash-bound prediction under the hash-bound detector
    config, evaluation config, annotation, class map, evaluator source,
    epoch-131 checkpoint, and `state_dict_ema`.
72. Never blindly rerun a partial Slurm submission. Require intent/receipt,
    full `jobid;cluster`, a submission lock, and pre-submit cluster pinning.
73. Commit `a6903ae` is only `implemented + tested` until its own CUDA gate and
    forced-overflow four-arm pilot pass. Older gates/pilots cannot authorize it.

## 2026-07-15 Spatial Zoom task-boundary guardrails

- Current-turn override: the active request is PhysTime SDPQ sparse downstream
  head refactoring. Do not continue it as Spatial Zoom, DUCA, ChronoTransport,
  or selector-policy work.
- Do not call commit `372fcbf` paper-ready. It is implemented, cleanly
  focused-tested, and real gate Job `1165340` passed; pilot Job `1165341` is
  queued. Full training is explicitly held until the pilot provides evidence.
- Do not revive G1a physical-anchor ActionFormer tuning after SDPQ. The point
  of SDPQ is to decouple complete physical query anchors from sparse
  observation support and use signed center/width regression.

- Do not reinterpret the active Spatial Zoom task as DUCA repair, temporal
  frame selection, online TAD, PhysTime, or ChronoTransport work.
- Shared Git history does not make worktrees one task. Make all Spatial Zoom
  edits only in its dedicated branch/worktree and do not transplant DUCA code
  unless a later, explicit design decision requires a generic utility.
- Do not call S1 a zoom/crop model. S1 only asks whether spatial-resolution
  headroom exists under a matched official AdaTAD protocol.
- Do not implement the learned crop policy before S1 and oracle-ROI S2 pass
  their preregistered GO gates. Conversely, do not present S1 infrastructure as
  completion of the user's requested spatial zoom route.
- Do not resume the cancelled warning-bearing S1 checkpoints as formal
  evidence. Repair and freeze determinism/inference first, pass an exact-commit
  CUDA gate, and rerun the matrix from scratch.
- Do not cancel, relaunch, or repurpose jobs from another route merely because
  a combined external audit discussed them together.

## 2026-07-15 S1 / DUCA STOP_AND_FIX guardrails

- Do not report DUCA Jobs `1164700-1164703` as formal matched evidence. Their
  successful-update schedules already lag by 3-4 updates around epochs 24-25,
  and the fixed epoch loop cannot recover the declared 13,200 updates.
- Do not use epoch count, batch-loop count, absence of NaN, or terminal mAP as
  a substitute for exact successful optimizer-update evidence.
- Do not repair DUCA by copying a new AMP loop when the repository already has
  S1 state-exact same-batch replay infrastructure. Reuse or factor it, and
  prove RNG/mutable-buffer restoration, scaler backoff, one-time schedule/EMA
  advancement, and replay-exhaustion failure.
- Do not call the running S1 3x3 matrix strict deterministic evidence. Every
  cell emits nondeterministic CUDA linear-upsampling warnings.
- Do not set `warn_only=False` before replacing or proving a deterministic
  equivalent for the offending interpolation operator; doing so only converts
  the warning into an immediate crash.
- Do not open the S1 sealed test or issue an S1 GO/KILL decision from the
  current warning-bearing artifacts.
- Do not adopt Bayesian cluster bootstrap merely because it avoids empty-class
  resamples. First freeze the inferential target and pass synthetic coverage,
  parity, and rare-class sensitivity checks.
- Do not treat terminal EMA, a specific counterfactual LCB threshold, or any
  patch in the Pro response as the unique correct solution or as implemented
  evidence. Freeze and test the selected protocol before rerunning.

## 2026-07-15 `043be401` Pro-audit guardrails

- Do not call the current counterfactual arm direct detector-gradient learning
  or signed detector utility. It is detached relative hard one-swap ranking;
  candidate-only softmax does not preserve all-harmful versus all-beneficial
  semantics without a baseline anchor.
- Do not let intermediate THUMOS test mAP select a checkpoint. For the current
  seed-0 screening, predeclare final one-based epoch 132 `state_dict_ema` as the
  only primary result; intermediate evaluations are diagnostic only.
- Do not state that this commit first evaluates at epoch 47. The mixed
  zero/one-based guards first trigger evaluation after one-based epoch 52.
- Do not describe the whole low-cost model as binary-supervision-only. The
  actionness head is binary, while shared hidden/scorer routes receive
  train-only endpoint-derived transition and coverage supervision.
- Do not call pilot `1164319` a multi-rank DDP proof. It is a one-rank
  DDP-wrapper pilot, which is sufficient for the current one-GPU jobs only.
- Do not treat the matched exact-uniform arm as the bare deployment baseline:
  it still runs and trains the coarse/transition stack. Use it for C3
  attribution, and report a separate no-probe uniform cost baseline.
- Do not implement a physical-time head merely because selected-axis geometry
  is a confound. Existing PhysTime diagnostics are negative; first require a
  fixed-selection geometry attribution result.
- Do not treat a no-op categorical softmax as the only signed-utility design.
  Compare it conceptually with baseline-vs-swap logistic or calibrated utility
  regression, then permit at most one bounded follow-up after current C4.

## 2026-07-15 dynamic-DDP deployment guardrails

- Do not restore `static_graph=True` for DUCA merely because official dense
  AdaTAD uses it. DUCA has batch-dependent selector and counterfactual
  parameter use; real diagnostics failed after several batches.
- Do not combine reentrant activation checkpointing with DUCA's required
  dynamic unused-parameter discovery on the remote Torch 2.0.1 stack. The
  admissible shared four-arm protocol is `with_cp=False`,
  `static_graph=False`, `find_unused_parameters=True`.
- Do not treat a one-step CUDA gate as deployment evidence. The exact commit
  must also pass all four ten-step pilots, including full/mixed/all-short
  effective-K batches, schedule transitions, finite optimizer updates, and
  expected gradient-group coverage.
- Do not call a suite deployable when `formal_ddp_pilot` is absent. Pilot JSON
  must bind the same commit, core-gate SHA, shared-protocol SHA, and ordered
  four variants.
- Do not alter the audited five-epoch checkpoint interval as a workaround for
  storage. Check capacity before submission and preserve the declared training
  protocol.

## 2026-07-15 PhysTime / sparse detector-head guardrails

- Do not continue the current task as a DUCA selector/gate task. The active
  continuation is sparse adaptation of the downstream TAD detection head.
- Do not full-train PhysTime G1b SDPQ commit `372fcbf` as-is. The 2026-07-16
  external verdict is `REVISE-BEFORE-FULL-TRAIN`: anchor representability is
  fixed, but support-query decoupling is incomplete and pilot support is
  engineering-only.
- Do not equate domain-valid physical queries with evidence-valid training
  positives. Uncovered queries with shared/null evidence must not silently
  receive positive assignment without explicit diagnostics and policy.
- Do not claim support intervals are exact local final-feature supports when
  the implementation uses patch-input envelopes after backbone/adapter temporal
  mixing. Envelope gap filling and support provenance must be exposed.
- Do not tune NMS, endpoint weight, or full-train length before decomposing
  high-IoU failure into assignment, proposal localization, score-quality
  ranking, top-k filtering, and Soft-NMS loss.
- Do not monitor or relaunch G1a jobs `1162048-1162050`; they completed.
- Do not call G1a physical-metric effective or paper-ready. Six-epoch pilot
  physical-metric improved Avg-mAP only from 10.26 to 10.56 and did not improve
  mAP@0.6 or mAP@0.7.
- Do not expand seeds, budgets, or new sparse samplers before explaining the
  head-side gap: physical-grid/physical-metric assignment, point generation,
  regression ranges, decode/NMS, candidate count, and loss normalization must
  be audited against selected-axis.
- Do not long-train the current physical-metric G1a head before changing or
  justifying assignment/range geometry. Commit `f2725f5` diagnostics show fewer
  physical-time positives and a much higher `<1s` no-eligible fraction than
  uniform-rank seconds on both validation and train-sampled windows.
- Do not deploy the simple `physical_time_rank_assignment` variant as a paper
  experiment. Its 2026-07-15 diagnostic was worse than physical-time seconds:
  validation positives fell to 7745 and GT no-eligible rose to 18.96%, with
  `<1s` no-eligible 65.16%; train-s5 positives fell to 14201 and no-eligible
  rose to 25.13%, with `<1s` no-eligible 72.02%. Assignment-axis swapping alone
  cannot fix missing/shifted physical anchor centers.
- Do not try to rescue the current G1a physical-anchor ActionFormer by tuning
  radius, regression range, schedule, or rank assignment. Pro audit on
  `b7a37f584ba7477159dd90ba08c14728c65fb19e` KILLs the
  observation-timestamp-coupled physical-anchor route as a method candidate.
  The nonnegative left/right regression head cannot represent a GT segment
  when no selected observation-derived center lies inside it; removing the
  physical-inside guard is invalid because it creates impossible negative
  distance targets.
- Do not call the next PhysTime route "physical-time-native" merely because
  the head uses seconds. A valid replacement must decouple complete physical
  query anchors from sparse observation support, use a regression
  parameterization that can represent off-anchor GT, and pass parity,
  support-observability, assignment, one-step, micro-overfit, and matched pilot
  gates before any empirical claim.
- Do not mix the negative full K384 PhysTime results with the G1a pilot as if
  they prove the same claim. The full K384 track rejected PhysTime superiority;
  G1a only proves a runnable native-J192 pilot with a weak low-IoU signal.

## 2026-07-15 P0 environment and evidence guardrails

- Never reuse gate `1163439`: it passed for `51330c8`, not current `cff479e`.
- Never prepare and submit a P0 suite from separately inherited `DUCA_*` or
  THUMOS environments. Both stages must source
  `scripts/duca_transition_only_p0_canonical_env.sh` and match the prepared
  canonical TSV byte-for-byte and by SHA256 before loading a config.
- A passing static/focused test or one-step formal gate is not evidence that
  the batch-varying DDP graph survives multiple real optimizer iterations.
  Run only the counterfactual arm as a bounded startup pilot first.
- Do not submit all four arms until the exact commit has independent no-P0/P1
  review, a fresh formal CUDA gate, and a stable multi-iteration pilot.
- Do not call the current route empirically supported: all completed P0 mAP
  runs predate the corrected protocol, and current C3/C4 status is unproven.

## 2026-07-13 S1 implementation guardrails

- Unit tests or static shapes are not an S1 GO result.
- Clip mode is not a full-window AdaTAD gate; formal readiness requires
  `run_spatial_zoom_s1_precheck.py --mode full` in a normal one-GPU Slurm allocation.
- Never profile random initialization or substitute FLOPs for trained-checkpoint
  decode-to-output p50/p95, allocated/reserved memory, and energy.
- Never bootstrap epoch-level mAP logs. Resample paired video clusters and
  recompute full class AP from raw predictions.
- Do not open sealed test to tune resolutions, checkpoints, manifests, bins, or
  thresholds.
- A formal S1 config is unusable unless it binds the full-CUDA precheck,
  manifest, clean Git commit, fixed seed, and fresh canonical workdir.
- Do not accept a partial eligible-epoch set or a score copied into selection
  JSON. Recompute every candidate from immutable in-process gate evidence.
- Do not issue a test-open certificate before all 3x3 selections validate. Do
  not change test `id`, profile prefix, warmup, power interval, or rerun after
  the exclusive test-open marker exists.
- Do not define short-action AP by predicted duration. Freeze duration bins
  from fit GT, filter GT only, and retain every prediction as a possible FP.
- Do not report component sums as end-to-end latency. Include an independent
  decode-to-result-accumulation wall timer and reuse the official DDP gather,
  cross-window NMS, and output reconstruction path. Persist the raw power trace
  and fail on excessive actual sampling gaps.
- Do not infer pretrained loading from file existence or a non-strict loader
  message. Preregister the checkpoint SHA and prove every non-adapter VideoMAE
  core key, shape, and loaded tensor value.
- Do not accept “the expected interpolation target appeared.” The complete
  scoped call sequence must be exactly 160=`[[10,10]]`, 224=`[]`, and
  256=`[[16,16]]`.
- Do not let cost decide whether spatial headroom exists. Apply all accuracy
  gates independently, then use measured cost only to freeze one passing
  resolution.
- Do not pool profiles across seeds unless hardware, software, sample manifest,
  protocol, precheck, and the single test-open certificate are globally equal.
- A manifest self-hash is not proof of the frozen split. Rebuild the complete
  manifest deterministically from the hash-bound annotation and require exact
  object equality.
- The pretrained checkpoint identity is a repository constant, not a runtime
  declaration. Bind the same filename/SHA through precheck, training,
  selection, test-open, profiling, descriptor, and analysis.
- Acquire one atomic, resolution/seed-specific profile-start marker before any
  measured dataset/model work. A crash must block that cell's rerun without
  blocking the other eight cells.
- Hardware equality means node, GPU UUID/PCI/driver/state, CPU, and memory;
  software equality includes the CUDA/PyTorch/OpenMMLab/decode stack. A shared
  GPU model name is insufficient.
- The complete 3x3 checkpoint matrix may open test only when all selections
  share one precheck file hash, precheck internal hash, and pretrained SHA.
- The once-only test marker belongs to the preregistered S1 study root, never
  to a commit/precheck-dependent experiment namespace. Reformatting or rerunning
  an equivalent precheck must not create another test-open path.
- The production checkpoint writer and validator must consume the same shared
  sidecar schema constant; helper-built sidecars do not prove the train entry.
- Execute the frozen profile order and matching hardware/software fingerprint
  preflight before test. The full profiler must independently repeat the check.
- A self-hashed GO/KILL JSON is insufficient. Rebuild it deterministically from
  all nine file-hashed descriptors and compare the complete report.
- Do not hand-enter a checkpoint epoch into a test descriptor. Recompute gate
  high-tIoU metrics from every eligible raw prediction, freeze the hashed
  selection proof, and cross-check the selected checkpoint epoch in profiling.
- Infrastructure at `tested` leaves the idea at `designed`; S2 and ROI/policy
  code remain locked until measured S1 GO.

## 2026-07-13 spatial-zoom guardrails

- 不得把讨论空间 zoom 写成已经放弃 DUCA、已选择新主线或已实现模型。
- gate-level `designed` 不等于 DART-Zoom 已设计冻结；当前只授权 S1 实验基础设施。
- S1 未 GO 不得实现/部署 S2；S2 未 GO 不得实现 scout/policy/Viterbi/EMA regret/fusion。
- reviewer 给出的 80px scout、J=16、K<=2、96/112、loss weights、teacher cadence、
  seeds 和 15% latency threshold 都是 proposals，不是实验事实或批准常数。
- 不得把 frozen dense-teacher reference oracle 称为 label-free deployable selector；它是
  privileged headroom diagnostic，路线决策期间不得反复读取 official test。
- 不得再称 `codex/chronotransport-pro-review` 公开不可见；当前 `git ls-remote` 已确认
  branch 指向 `1f5f7254a390f183121e6c4b7cebcebd2f2954d1`。
- 不得把 Uni-AdaFocus 原样迁移到 AdaTAD 称为新方法；AdaSpot 已覆盖更接近时间定位的
  low-resolution global + high-resolution ROI 范式。
- 不得把 crop resize 回 full-frame heavy 输入尺寸后声称节省 backbone FLOPs；必须按
  实际输入 token 数与全流程延时核算。
- 不得在 dense high-resolution 相对当前 160 输入没有高 tIoU headroom 前训练 policy。
- 不得把单 ROI、每帧独立 crop 或 detector-loss 直通 `grid_sample` 当作天然合理；必须
  检查多人/并发上下文、ROI 抖动和弱空间监督稳定性。
- 不得只报 backbone FLOPs。高分辨率 decode/retention、resize、H2D、scout、crop、
  fusion/head、p50/p95、显存和能耗必须进入 strict total-cost ledger。
- training-free saliency zoom 只能是稳定 baseline；论文主张必须有 TAD-specific 的
  boundary-risk spatial allocation、ROI-tube utility 或等价的新机制。
- 不得因 fused 主路径保持 768 点就声称所有 detector 训练语义未变；global/local auxiliary
  head reuse、loss normalizer、optimizer exposure 和 wrapper 必须分别审计。

## 2026-07-13 selection-quality guardrails

- Do not call the legacy joint coarse classifier strong: pooled AUROC/AUPRC is
  0.621/0.411 and ECE is 0.171 on the audited validation exposure.
- Do not claim learned selection beats uniform from its +1.53-point exact-r0
  recall alone. It loses 15.55 points at radius 1, increases endpoint distance,
  and is worse in 308/487 windows.
- Do not claim the learned transition head improves indirect localization: the
  audited `abs_delta + uncertainty_peak` compound proxy has higher AP and AUROC
  at every boundary radius. Pure `abs(delta p_action)` was not separately
  measured, so do not call this a pure-delta comparison.
- Do not interpret max-gap validity as selection quality. The learned decoder
  caps observed holes at 15 but overlaps utility top-k by 99.80% and still has
  much weaker local coverage than exact uniform.
- Do not transfer this diagnosis to corrected commit `0ea4e15`. The analyzed
  epoch-89 checkpoint came from the invalidated `8bfc0e5` homotopy and beta=0
  has no detector-gradient bridge.

## 2026-07-13 CellCF review guardrails

- Do not continue `global_structured_topk + G=15 + structured_zero_forward` as
  the default final DUCA merely by tuning weights; the route is under REDESIGN.
- Do not call DUCA-CellCF implemented or selected as final. It is a bounded
  `discussed/design proposed` appeal whose review snippets have not run locally.
- Do not assume assigning an off-anchor acquired frame to a fixed uniform
  detector anchor is harmless; it requires a same-selected-frames geometry test.
- Do not interpret per-cell counterfactual loss as global detector utility
  without hard-alternative rank/sign and interaction audits.
- Do not claim CellCF approaches the full GT Oracle. One-per-cell intentionally
  forbids the Oracle's cross-cell concentration and can only learn local
  residual deformation around uniform.
- Do not run CellCF full training before repo-wide uniform/target/coverage/
  diagnostic fixes and the coarse, one-swap, coverage, and geometry gates.
- If the same-commit fixed-384 pilot does not beat exact uniform, stop DUCA as
  a main method; do not unlock MUST, X3D/SlowFast, more heads, or more losses.

## 2026-07-12 exact-uniform audit guardrails

- Do not report Job `1159414` or its best Avg-mAP `55.67` as exact-uniform. Commit
  `8bfc0e5` produced all-zero alpha=0 reference logits at T=768/K=384 and a
  degenerate Viterbi tie-break path.
- Do not treat Jobs `1159416/1159417` as evidence for the intended continuous
  uniform-to-learned homotopy; their alpha=0 endpoint was defective.
- A uniform control is admissible only when the final decoded positions equal
  `round(linspace(0,T-1,K))` point by point, not merely when alpha=0 or K is
  exact.
- Do not copy historical 64.352/65.696 uniform results into the current matched
  table. They establish a real near-65 anchor but use different detector or
  geometry protocols.
- Do not reuse formal gate `1159395` after uniform-reference commit `0ea4e15`.
  A new hash-bound gate must record `uniform_reference_exact=true` and zero
  rank-aligned position error before replacement training is submitted.

## 2026-07-12 PhysTime guardrails

- Do not claim PhysTime-AdaTAD v1 improves irregular-sampling TAD: its best
  Avg-mAP was 57.21 versus 63.61 selected-axis and 59.14 physical-grid.
- Do not turn the isolated +2.62 mAP@0.7 versus physical-grid into a general
  boundary claim; PhysTime still trails selected-axis by 6.91 at tIoU 0.7.
- Do not launch Phase 2 or tune losses/heads. Reopening requires a matched
  timestamp/support/endpoint factorial ablation and multiple seeds.

## 2026-07-11 transition-only guardrails

- Do not call transition-only gate `1159350` a paper result: it proves code,
  gradients, and contracts only; it contains no full-train mAP.
- Do not reopen MUST/MobileNet/new-head work before the four-way fixed-384 P0
  matrix and one-swap/cost/geometry gates are resolved.
- Do not use `fc98eca` or scaled proof `1159350` as formal full-model evidence.
  The minimum admissible implementation commit is `8e38cca`; P0 additionally
  requires the hash-bound real 768-to-384 CUDA gate to pass.
- Do not describe task-adapted DUCA as THUMOS-free/train-free. It uses THUMOS14
  action and GT-segment transition supervision during training, while inference
  remains label/GT/teacher/cache free.
- Do not freeze optimizer-excluded parameters after DDP registration, and do
  not call a P0 optimizer protocol matched when direct/transition effective
  component learning rates differ.
- Do not reuse gate `1159385` or failed P0 jobs `1159387-1159390` after the AMP
  coverage fix. The current admissible pair is commit `8bfc0e5` plus gate
  `1159395`; current P0 jobs are `1159414-1159417`.
- Do not turn the completed seed-0 jobs `1159414-1159417` into a matched result.
  They are numerically healthy protocol-invalidated diagnostics; the strict
  result-to-claim verdict for both C3 and C4 is `no`, not a refutation.

## 2026-07-15 S1 Slurm and retry guardrails

- Formal S1 jobs must request one generic Slurm GPU and consume the assigned
  logical `cuda:0`; do not restore a physical `CUDA_VISIBLE_DEVICES=1` rule.
- Concurrent one-node jobs must not derive a static rendezvous port from a
  Slurm job ID. Use an atomic kernel-assigned port and retain a unique
  rendezvous ID.
- An AMP retry is not state-exact if it restores RNG alone. Restore every
  forward-mutated model buffer while retaining GradScaler backoff, and count
  schedules, EMA, and exposure only after a successful optimizer update.
- Do not pass the manifest-bound S1 config directly to runtime components that
  call `pop`, rescale scheduler fields, or inject inference/post-processing
  fields. Those components receive deep copies; the original config remains
  the checkpoint and gate-evidence identity.
- Do not reuse commit `9298c0e` or Jobs `1164261/1164267-1164274`; the pilot
  deterministically failed its first checkpoint and the remaining jobs were
  cancelled. The current admissible deployment starts at `35204f5` with
  precheck `1164289`.
- Full precheck PASS and a live training process are not S1 empirical support.
  Wait for complete 3x3 mAP, high-tIoU, short-action, and full-stack cost
  evidence before a GO/KILL decision.

# 禁止重走清单

## 2026-07-13 corrected P0 guardrails

- Do not reuse gates `1159395`, `1161464`, `1161466`, `1161467`, `1161468`,
  or `1161471`; they predate final fixes or failed real CUDA backward.
- The only admissible current implementation is commit `40eb86ee69e19b3105f9ddd6a977fb7693f724ad`
  plus formal gate `1161499`.
- Boundary training uses a structured-DP soft expected-neighborhood-mass
  surrogate, not an exact coverage probability.
- Jobs `1161505-1161508` are invalidated by a batch-varying DDP static graph;
  never report them as matched evidence. Commit `40eb86e` remains only `tested`
  until exact-commit independent audit and a multi-iteration counterfactual DDP
  pilot pass. Any later full-train jobs stay `experiment_running` until raw mAP, stability,
  geometry, and hash-bound post-run evidence are complete.
- Never reuse Jobs `1161482-1161485`, gate `1161489`, Job `1161492`, or Job
  `1161494` as results; they are invalid deployment diagnostics.
- Never mix invalid Job `1159414` or unmatched historical 64.352/65.696
  anchors into the corrected matched table.

## 2026-07-13 DUCA-FSU review guardrails

- 不得把 external reviewer 推荐的 DUCA-FSU 称为已选择、已实现或最终模型；当前状态仅为
  `discussed/design proposed`。
- 不得把 hard one-swap detector-loss target 称为 detector gradient 直接反传。它是
  counterfactual utility distillation，会实质改变 C4 的训练机制和论文口径。
- 不得只修 transition-only uniform helper；legacy/direct stable/control route 必须共用
  唯一 rounded-endpoint helper，并逐点审计 decoded positions。
- 不得假设 physical-time RGB reconstruction 必然解决 selected-axis geometry。PhysTime v1
  已有负结果，新 reconstruction 必须做 same-selected-frames 对照。
- 不得把模型图中的 pre-backbone 当作 pre-decode/pre-resize/pre-H2D；当前 dense input
  pipeline 的 I/O 与预处理必须计入 full-stack cost。
- 不得把 reviewer 给出的根因概率、AUROC/Spearman 阈值和 latency 门槛写成已验证事实；
  它们只能作为预注册的 proposed stopping rules。
- 不得直接粘贴 reviewer 的 uniform/swap/reconstruction/hard-exposure 代码进入主路径；
  必须先做本地数值、状态、梯度和真实 detector 回归测试。

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

## ChronoTransport 与新颖性

25. 不得声称首次提出 `time × layer` 动态计算；MoD、TAPS、Adaptive Temporal
    Refinement 等已有明确近邻。
26. 不得把 `RECOMPUTE/TRANSPORT/HOLD` 三动作本身作为核心创新；SCOPE 已有
    `cache/predict/recompute` 与稳定控制，Eventful/ResidualViT 已覆盖时序复用。
27. 不得把 held-out/conformal 风险控制本身作为首创；创新只能收缩到 TAD 的结构化
    localization regret、dense physical-time lattice 与真实成本联合合同。
28. 不得因 coverage=`1.0` 宣称校准成功；seed 3407 的 risk gross overestimate 导致
    Spearman=`-0.1914`，属于失败证据。
29. ChronoTransport 只允许一次有界 P3 修复；失败后不得继续调权重、直接启动 Stage C
    或用工程测试数量掩盖科学 gate。
30. 新方向不得只是重新组合 cache、MoD、motion、Mamba 或 conformal；必须改变表示、
    决策变量或可检验的科学问题。
31. 不得用相同 epoch 数声称训练量一致；不同 dataset/loader 必须先对齐 optimizer
    step、学习率进度和样本窗口暴露量。
32. 不得把乘过 schedule 权重的 selector loss 下降解释为模块学会；必须还原 raw loss，
    并与 `log(2)`、`log(T)` 等 chance baseline 及直接 coarse/boundary 指标比较。
33. 不得把随机 custom spatial stem + official temporal core 简称为已验证或官方完整
    ASFormer coarse classifier；checkpoint、预训练数据和实际模块必须逐项披露。
34. 不得在 one-swap finite-difference 对齐前把 zero-forward/ST 的 nonzero gradient
    写成 detector utility supervision；它当前只证明 backward connectivity。
35. 不得因 MobileNetV3/X3D/SlowFast 被称为 lightweight 就默认 probe 成本可忽略；
    dense full-window pre-backbone 执行必须进入 trained-checkpoint full-stack Pareto 表。
35. 不得在全部候选记录上做 conformal coverage 后声称部署 scheduler 安全；必须用同一
    选择规则做 selection-aware calibration/evaluation，或给出 simultaneous guarantee。
36. ChronoTransport 只允许 `CT-P3R-3S` 一次上诉；任一 oracle/mechanism/risk/runtime
    gate 失败后，不得再更换 head、loss、权重、candidate library 或 seed 延长路线。
37. 不得在 standalone/joint 使用同一 ASFormer class 但训练协议未匹配时，把性能差距
    归因于缺少 MobileNet；先对齐 steps、loss、gradient routing 和 checkpoint selection。
38. 不得给 coarse ASFormer 直接挂 GT start/end predictor 后仍称“粗分类间接边界定位”；
    GT boundary 只能监督 transition-derived selector，不能成为推理输入或粗模型主任务。
39. 不得把 pre-ASFormer stem feature 命名为 ASFormer temporal hidden；必须记录 hidden kind
    并用原 forward logits-equivalence test 证明 wrapper 未改变官方输出。
40. 不得再把 current constrained DP 误判成 hard/soft 不同可行族；它已通过 exact-K/
    max-gap 同构审计，尚未通过的是 detector-gradient 对 hard one-swap utility 的方向性。
41. 不得继续使用 `step % 100` 在两套 hard policy 间 duty-cycle 跳变并称连续 curriculum；
    若实施联合修复，应在同一 structured score 上做单调连续 homotopy。
42. 不得在 transition-only fixed-384 的 matched gates 前启动 MobileNet、MUST、更多 heads
    或新 loss；这些改变会掩盖原始状态变化假设是否成立。
43. 审稿人给出的 AUROC/Spearman/latency 数值门槛属于 proposed specification，不是已经
    经实验验证的自然常数；不得直接写成论文事实。
44. 不得执行 `b74101d`。r1 规格已写入 `02199f8`，但用户最终书面复核前仍不得进入
    implementation plan、争议代码、profiler/Gate 1、新 seed 或 Stage C。
45. 不得把 Gate 1 的候选集合 oracle minimum 直接写成 input dependence；它只证明
    frozen-library oracle headroom，输入依赖必须由 Gate 3 的 deploy-visible ranking/
    selection 证明。
46. 不得再用 candidate-row pooled Spearman 或 row bootstrap；主统计单位是
    seed/window 的完整 candidate vector，bootstrap 外层 cluster 是 unique window。
47. 不得把 coverage>0.95 作为 hard FAIL。r1 只保留下界≥0.85；overcoverage 必须标记，
    并由 pinball、sharpness、non-dense rate 与 unique-window support 共同解释。
48. 不得把 evaluation-best static 反馈给部署或后续拟合；它只允许 diagnostic。可部署
    comparator 必须在 fit/calibration 侧冻结。
49. 不得相加各阶段 p50/p95 充当 full-stack percentile；必须记录每次完整 forward 的
    total sample，并从 total distribution 计算。
50. 不得把 140 loop iterations 写成 140 training steps；formal artifact 必须证明
    successful optimizer updates、AMP skips、per-schedule exposure 与 LR/EMA update。
51. 不得让 dense safety fallback 掩盖 budget failure；dense 超过 B* 必须记录
    safety_override_budget_violation，并进入真实 cost/selection 统计。
52. 不得把 Pro 文本中的 sandbox patch、`10 passed` 或 upstream 风险当成本地集成事实；
    文件未随附件提供，且 reviewer 未看到本地 ChronoTransport 源码。
53. 不得再声称当前 Stage B 的 transport/risk 被 base optimizer `lr=0` 冻结；它使用独立
    AdamW。真正的冻结风险在尚未实现的通用 Stage C optimizer，必须用逐参数审计证明。
54. 不得把 paired replay 的 `loss_normalizer` 风险写成当前已确认污染；head 已处于 eval，
    RNG 也会恢复。必须用候选顺序 permutation test 证明合同，而不是凭推测修补状态。
55. 不得把“dense adapter 被计算”写成“adapter 语义保持不变”；当前只写回 RECOMPUTE
    rows。r1 必须明确 heavy-subpath HOLD/TRANSPORT 与全 rows dense TIA 的边界。
56. 不得把当前 6-schedule formal runner、per-seed manifest 或 stage-internal profiler
    冒充 r1 的 16-candidate、共享 split 和 full-stack total-sample protocol。
57. 不得在 `max_cache_age=8` 下把 `hold_only/transport_only` 当作原样可执行的 48-clip
    候选；必须拆分 hard validity 与 embedding cap，或从冻结库删除并重命名 repaired controls。
58. runtime repair/NaN fallback/dense fallback 后不得继续沿用 requested schedule cost；
    requested 与 executed action/hash/cost 必须分别登记。
59. 当前 Stage B 没有 AMP/GradScaler；不得把 AMP skips 写成现存 bug 或 artifact 事实。
60. 140/16 的 round-robin 不是天然 balanced；必须冻结跨 seed start offsets 并登记
    candidate×video exposure，避免固定 video order 与 candidate 绑定。
61. 不得再用简单 `(update+offset)%16` 与 offsets `0/4/8`；它把 candidate/video mod-4
    永久绑定。必须采用经断言的 block-rotated assignment 与 matrix hash。
62. 不得把 200 video IDs 直接当 200 windows；Gate 1–3 必须冻结 one-window-per-video
    manifest，Gate 4 明确属于不同 full-video/sliding-window population。
63. Stage-C loss ownership 不得只写文字目标；必须用 object-identity sets 与独立
    `autograd.grad` 明确 detector/feature/risk 对 adapter/transport/risk 的梯度边界。
64. AMP retry 不得只恢复 RNG；必须恢复所有 forward-mutated buffers/state（保留 scaler
    backoff），尤其 head `loss_normalizer`，并以 successful exposure 而非 skip vector 匹配。
65. 不得保留 oracle-minimum 对 shuffled oracle assignment 的硬门；该统计近乎定义成立。
    Gate 1 用 evaluation-only strongest oracle comparator，Gate 3 才证明 deploy-visible dependence。
66. checkpoint/data/config/annotation/window identities 必须在任何 profile/replay 前冻结，
    不得把关键选择推迟到看过结果后的 run manifest。
67. 不得在完整 `T/max_slots` 上构造 structured soft surrogate 后再掩掉短样本后缀并归一化；
    hard 与 soft 必须逐样本共用 `valid_count/effective_k/max-hole` 可行域，batch 对齐只能零填充。
68. counterfactual teacher 若在同一 CUDA autocast 区域中先于主 detector 前向执行，必须关闭
    autocast weight cache，避免 no-grad 的 detached cast 被主路径复用而切断 adapter 梯度。
69. 任何 selector/bridge 代码 SHA 变化都会使旧 formal gate 与 pilot 失效；四臂真实 10-step
    pilot 未全部通过时，不得提交 132-epoch matched suite。

## 2026-07-16 CellCF submission transaction guardrails

70. Do not rely on `set -e` inside nested Bash command substitutions. Every
    prepared-binding read, `sbatch`, binding normalization, receipt write and
    live Slurm validation must have an explicit checked branch.
71. A formal receipt requires an exact `jobid;cluster` response. A bare job ID,
    empty response, malformed response, wrong cluster or null parsed ID must
    leave only an intent and stop for manual reconciliation.
72. Jobs `1167234-1167236` and the null aggregate/cost/completion receipts under
    the `3a0f5ae_formal_seed0` root are permanently invalid. No downstream DAG
    job existed; never quote them as a submitted or partially completed suite.
73. Any submission/evidence-handoff code SHA change invalidates old gate/pilot
    authorization. Re-run the exact Linux, synthetic, real-loader CUDA and DDP
    chain before another 132-epoch submission.
74. Do not assume `sacct Comment` is populated for pending allocations on
    `n16r4`. Live jobs must bind the exact `squeue Comment`; an empty accounting
    comment may fall back only to one exact `--comment` in `sacct SubmitLine`.
    Missing, duplicate, malformed or conflicting comment evidence must remain
    fail-closed. Jobs `1167359/1167360` are cancelled zero-runtime diagnostics,
    not a formal suite.

## 2026-07-17 CellCF training-budget guardrails

75. Do not present `epoch_131.pth` as an innocuous default. It means 132 epochs
    and 13,200 successful updates, about 2.2 times the repository AdaTAD
    60-epoch training length.
76. The 132-epoch suite is a sufficient-exposure diagnostic chosen to remove
    the old 5,940-vs-13,080 update confound. It does not by itself establish an
    efficient final training recipe.
77. Do not call the existing epoch-59 checkpoint an official 60-epoch result.
    It belongs to a scheduler with a 132-epoch horizon and is only a
    convergence-trajectory point. Epoch 59/89/131 must not be used for
    post-hoc checkpoint selection.
78. If CellCF passes the 132-epoch gate, the paper claim still requires a
    same-commit official 60-epoch matched contract for exact-uniform,
    transition-beta0 and CellCF, with equal successful updates and explicit
    training GPU-hours.
79. Do not claim end-to-end efficiency from inference FLOPs alone. Report
    training GPU-hours, peak memory, counterfactual-training overhead,
    full-stack inference p50/p95 and the training/inference break-even point.
    A gain that appears only after 132 epochs does not support efficient
    training.

## 2026-07-17 CellCF post-run evidence guardrails

80. Never deploy post-run evidence commit `787569e`. Its Linux test showed
    that final hash/inode/timestamp/directory-stat equality can miss a
    transient replace-and-restore operation.
81. Formal candidate or external sealing must use exact evidence commit
    `9e96967a158534b014aacde57c1b78bd1591e71a` or a later independently gated
    successor, with the Linux mutation monitor enabled before every evidence
    read/hash.
82. A local Windows pass is insufficient. The exact commit must pass both the
    same-process and independent-process transient mutation tests with pytest
    temporary files on the target `/data/run01` mount before any post-run DAG
    submission.

## 2026-07-18 CellCF cost-schema guardrails

83. Do not add a wildcard allowance for arbitrary `*_cpu_enqueue_ms` fields.
    The contract admits exactly the seven registered nested-stage diagnostics;
    unknown names, negative values and non-finite values must fail closed.
84. CPU-enqueue diagnostics are raw-only. They must not enter canonical stage
    p50/p95, `end_to_end_serial_ms`, selector-policy latency or any paper cost
    sum unless a new schema and matched protocol are explicitly designed.
85. Job `1170932` is not a passed gate. Its real profiler artifacts validate,
    but the immutable Slurm job failed in a malformed temporary verifier.
    Job `1170940` is the exact `COMPLETED/0:0` two-sample schema gate.
86. A two-sample schema gate proves producer/consumer compatibility only. It
    is not a trained-checkpoint cost estimate, does not support C7, and cannot
    replace the preregistered repeated 500-sample CellCF/bare-uniform pair.
87. Preserve `duca-full-stack-cost-v1` exact reconstruction for historical
    reports. Raw-only diagnostics do not justify a schema-version bump or a
    new derived claim field.

## 2026-07-20 Allocation-Ceiling guardrails

88. The canonical THUMOS alignment axis for this diagnostic is decoded frame
    index. Do not reintroduce a hard full-video decoder-FPS versus
    annotation-FPS drift threshold when decoded and annotated frame counts
    agree; retain that clock difference as a reported diagnostic.
89. Failed gate Job `1174706` and cancelled zero-runtime descendants
    `1174707-1174710` are immutable deployment diagnostics, not ceiling,
    detector-loss or cost evidence.
90. Any change to exporter schema, physical-axis validation, solver,
    candidate evaluator or submission transaction invalidates older formal
    gates and requires a new exact commit, clean snapshot, focused tests and
    precheck.
91. Allocation-Ceiling is a bounded necessary-condition experiment. Do not
    call it a trained selector, DUCA-CARA implementation, final method or
    paper-ready evidence.
92. Do not consume validation/test or submit replay mAP from this training-side
    DAG. Validation remains behind a single-use human GO and a sealed v2
    authorization manifest.
93. Do not infer headroom from gate success. Only the hash-bound completion
    artifact may support measured family headroom, recoverability, frozen
    detector-loss and solver-cost statements.
94. Formal Slurm DAGs must continue to submit all jobs held, validate exact
    scheduler state before and after release, and roll back every submitted
    job if any transaction check fails.
95. Job `1174713` is not negative allocation evidence. It failed on
    `lex_block_0210_0240` because float-valued binary residuals were multiplied
    by large lexicographic weights. Jobs `1174714/1174715` were cancelled at
    zero runtime and must not be reused.
96. Do not repair exact integer MILP objectives by merely widening a raw-float
    tolerance. Canonicalize integer variables, require numeric zero MIP gap,
    bind `result.fun` and `mip_dual_bound` to the same unique integer, derive
    objective values from selected positions, and replay every pinned
    objective on the terminal solution.
97. Solver commit changes invalidate the old gate even when export data is
    unchanged. Commit `8ebdd2a` requires a new clean Linux test, exact gate and
    complete five-job DAG; no artifact from the `1d51379` chain may serve as
    its completion evidence.
98. Targeted replay Jobs `1175380` and `1175392` are immutable submission
    diagnostics: the first never entered Python because of an incompatible
    temporary shebang, and the second failed at zero runtime after command
    tokenization. Only Job `1175393` used the intact Bash payload and may serve
    as the failed-sample repair preflight.
99. Gate `1175395` proves the repaired exact solver and surrounding contracts
    execute on one real sample. It does not prove allocation headroom,
    deploy-score recoverability, detector-loss improvement, selector
    learnability or paper readiness; those require the hash-bound completion
    artifact from Jobs `1175396-1175399`.
100. The completed `8ebdd2a` suite is a strong negative proxy/loss diagnostic,
     not a final route KILL. Deploy-score geometry is worse than uniform and
     all nonuniform candidates worsen frozen detector loss, but final TAD
     judgment requires decoded/NMS mAP.
101. Do not interpret privileged boundary-distance improvement or frozen
     detector loss alone as detector utility. Before selector training, run
     one single-use hash-bound official mAP replay comparing exact uniform and
     deploy-visible transition selection under the same checkpoint and
     post-processing. Privileged GT selection must not enter that deployable
     comparison.
102. Exact solver latency (`347.835 ms` median, `363.880 ms` p95) is a
     training-side decoder diagnostic, not a deployable selector cost and not
     a full-stack saving.

## 2026-07-20 Protected-E2E Pro adjudication guardrails

103. Do not run official-60 from `b3222af` or its uncommitted successor. Its
     local structured transport, candidate-hole constraint and selected-axis
     target contract do not implement the approved physical-DAG route.
104. A real nonzero detector-to-selector gradient proves connectivity only.
     It does not prove that the surrogate follows legal hard-swap utility or
     improves terminal mAP.
105. Hard Viterbi and soft assignment must use one identical physical exact-K
     DAG, including source, internal and sink edges. Local per-slot softmax,
     local temporal slope and post-hoc repair are forbidden in protected
     configs.
106. Selected-axis GT remap is forbidden for global nonuniform allocation.
     Physical target, regression decode, proposal, NMS and evaluator axes must
     be checked end to end.
107. The approved four-arm route uses no counterfactual teacher, utility
     distillation, soft max-gap legality, learnable coverage floor, policy
     homotopy or detector-gradient ramp.
108. The main detector gradient must stop at the selector adapter/head.
     Detector loss must not update the action head or ASFormer in the main arm;
     the only rho arm opens the last temporal block with fixed rho 0.01.
109. Do not treat Job `1176948` as P3 evidence. P1/P2 exact gates passed, but
     P3 stopped on an invalid manifest-field requirement before any alignment
     statistic was computed.
110. Do not accept `5940` successful updates merely because a review wrote
     `99 steps/epoch`. P0 must derive loader length from the exact dataset,
     sampler and drop-last manifest. Use 5940 only if that exact runtime
     contract proves 99; otherwise preserve the measured count.
111. The new P3 population and thresholds are preregistered: 48 stratified
     train windows, 576 deterministic legal physical swaps, at least 512
     effective swaps and video-cluster bootstrap. The old 4x8 gate cannot be
     relabelled or expanded after observing results.
112. Physical-grid is necessary but not presumed sufficient. Prior PhysTime
     diagnostics exposed short-action positive-support degradation, so
     target/decode roundtrip, support, parity and micro-overfit gates are
     mandatory before long training.
113. Do not use findings from `OpenTAD_DUCA_ProtectedE2E_20260720` to judge the
     isolated final tree without first checking the exact path. A 2026-07-20
     read-only audit accidentally inspected that stale selected-axis tree and
     reported the new physical P3 as missing. Such path-mismatched audit text
     is not evidence.
114. `37 passed` is only a remote focused implementation gate. It does not
     authorize official-60, prove hard-swap alignment or support an mAP claim.
     Require a clean exact commit plus hash-bound P0, both P1/P2 CUDA gates,
     all three P3 shards, aggregate PASS and authorization first.
115. The performance objective is strictly greater than the matched
     approximately 65 Avg-mAP baseline. Never rewrite that objective as a
     guaranteed result, and never substitute detector training loss or P3
     sign agreement for terminal official mAP.
116. `50 passed` closes focused contracts and the evidence plumbing only. It
     does not prove that rank-packed nonuniform frames preserve VideoMAE
     tubelet temporal semantics, that P3 hard swaps align with detector
     utility, or that terminal mAP exceeds 65. Only clean exact-commit CUDA
     gates followed by the four sealed terminal-EMA arms can answer those
     questions.
117. Do not treat the 2026-07-21 `55 passed` focused suite as an experiment
     submission. It validates contracts only. A valid next state requires a
     clean exact commit, P0 hash freeze, main/rho real CUDA gates (including
     full+padded windows, real optimizer/scheduler/EMA updates and uniform
     physical/legacy parity), three original-boundary P3 shards, aggregate
     PASS and authorization. Until then the remote formal queue is empty and
     no mAP claim exists.
118. Never use a random-truncation window edge as a true action boundary.
     Binary actionness may use the visible clipped action span, but transition
     supervision and P3 boundary-distance evidence must mask clipped
     endpoints and refer to original uncropped THUMOS annotations in seconds.

## 2026-07-20 Worktree isolation guardrails

- `WT-1`: Do not infer the active implementation from an `OpenTAD_*` directory
  name. Check `research-wiki/worktree_inventory.md`, exact HEAD and dirty
  state first.
- `WT-2`: The dirty primary tree is owned by Spatial-Zoom coordination. Do not
  land DUCA model edits there during development.
- `WT-3`: Do not modify SparseHead, Spatial-Zoom, ChronoTransport, GAS-VT or
  historical selected-axis worktrees while implementing Protected-E2E.
- `WT-4`: Reuse `DUCA_AllocationCeiling` only as the audited hard physical-DAG
  semantic reference; do not make model code import from `tools/bata`.
- `WT-5`: Reuse PhysTime/TrueTime assets only after line-by-line contract
  comparison. Their physical head, short-action support and selected-axis
  assumptions are not automatically approved.
- `WT-6`: The isolated clone
  `.codex_tmp/OpenTAD_DUCA_ProtectedE2E_Final_20260720` is the only DUCA
  construction tree. A dirty draft there is not an exact commit or gate
  result.
## 2026-07-21 Uni-Companion guardrails

119. Do not claim that Uni-AdaFocus sends the heavy local-network loss through
     its hard temporal indices. Its official hard temporal sampling path is
     detached; DUCA must retain its own same-DAG hard-forward/soft-backward
     detector bridge.
120. The exact-uniform companion is training-only input diversity. It must not
     appear at validation/test, must not add a second detector forward, and
     must not replace the learned selector at inference.
121. Gradient scale `0.25` is a preregistered ablation, not evidence of better
     mAP. Only matched terminal official mAP can choose among bridge `1.0`,
     bridge `0.25`, and bridge `0.25` plus companion.
122. Do not reuse or revive Jobs `1177681`, `1177687` or
     `1177690-1177692`. They are bound to superseded commits and were
     cancelled before runtime. The current exact gate is `1177696` at commit
     `4d84acd`.
123. Jobs `1177697-1177699` are three learned arms waiting on gate
     `afterok:1177696`. Their existence proves deployment only, not a passed
     gate, a started optimizer, a terminal checkpoint, or improved mAP.
124. Do not call the current queue a complete matched four-arm result.
     Exact-uniform is frozen and tested but not queued because of
     `AssocMaxSubmitJobLimit`; watcher PID `808310` may submit it only after
     gate `1177696` completes and exact authorization exists.
125. The implemented Uni companion is one detector forward with one exact-
     uniform row and one learned row at batch size 2. Do not relabel it as a
     paired two-forward companion, detector-loss distillation, or an inference
     ensemble.
126. Do not assume a fresh learned protected selector starts from exact
     uniform. At commit `4d84acd`, `DucaProtectedTransitionScorer` uses its
     default random output-head initialization, and tied global physical
     exact-K Viterbi paths use a lexicographic tie-break. Preserve the sealed
     run, log first-batch geometry, and test any uniform-initialized successor
     under a new exact commit/P0 rather than silently changing the running
     protocol.
127. Do not approximate the `T=768,K=384` physical cap as 2 frames. The
     canonical exact-uniform ranks contain one 3-frame interval, so the
     frozen cap is 3. Under this cap, zero-score lexicographic Viterbi is
     strongly nonuniform (`0.5` set overlap and about `96.9974` rank-MAE).
     Any initialization claim must be checked through the actual
     coverage-floor plus physical-DAG decode, not inferred from score equality.
128. Never freeze or submit a protected-selector gate from a synthetic
     floating-input test alone. The real THUMOS training loader emits uint8
     RGB. The hard gather may retain raw values, but every differentiable soft
     resampling/straight-through branch must promote to floating point before
     arithmetic. Jobs `1177687/1177690-1177692` are zero-runtime diagnostics
     of this missed contract, not experiments.
129. An annotation-only `data/thumos-14` symlink is insufficient for the
     real-loader CUDA gate. Every clean snapshot must bind both
     `data/thumos-14/annotations` and `data/thumos-14/raw_data`, verify a real
     video exists, and remain git-clean.
130. The eight-window initial-policy audit proves nonuniform initialization
     but does not prove worse boundary geometry: learned mean boundary distance
     was `0.528646` versus uniform `0.555556`. Do not add policy homotopy or
     uniform initialization to the sealed `4d84acd` run without a new explicit
     design decision and separately frozen successor.
131. `protected_e2e_rho001` is not an unrestricted end-to-end coarse-probe
     update. Detector gradients may enter only the final official ASFormer
     encoder layer at scale `0.01`; the action head and all earlier temporal
     layers remain protected. Watcher PID `883230` is deployment intent, not
     a submitted Job or empirical result.
132. At batch size two, the current Uni companion sends detector-to-selector
     gradient through only one learned row. Its local coefficient is `0.25`,
     so aggregate exposure is approximately half the plain bridge-0.25 arm
     under comparable row statistics. Do not attribute a future mAP delta
     solely to companion input diversity without a separately frozen
     gradient-normalized control.
133. Watcher PID `933605` is not a submitted `transition_no_bridge` Job. It may
     submit only after the P0-bound rho arm completes, and only after reopening
     the exact P0 config hash and original-four-arm authorization. A positive
     Job ID and accounting record are still required before calling that arm
     queued or running.
134. Jobs `1177696-1177699` are zero-runtime deployment failures, not model
     experiments. Gate `1177696` failed with exit `127` because a non-login
     Slurm shell lacked the `module` function; all dependents were cancelled.
135. Every generated N16R4 sbatch must initialize `/etc/profile` before any
     `module load`, and focused tests must inspect the generated contract.
     Passing `bash -n` alone does not prove the environment-modules contract.
136. The new uniform-to-learned homotopy means only that the hard detector
     input is exact uniform at alpha zero. Do not claim its Gibbs/ST soft
     distribution is identical to a degenerate uniform path. It requires a
     new commit, P0, gates and same-commit terminal mAP comparison.
137. Job `1177713` at `be18ba5` is a failed real-gate diagnostic, not an
     admissible gate. It exposed float64 max-gap cap narrowing under AMP.
     The replacement chain is commit `bc503fc`, P0 file SHA-256
     `7b5820fe...96483`, and gate Job `1177714`. Jobs `1177696-1177699`
     remain immutable zero-runtime launcher failures.
138. `152 passed`, P0 freeze and an independent code-audit PASS authorize only
     the real CUDA/P3 gate. They do not prove optimizer stability, authorize
     official-60 training, report mAP or support the greater-than-65 claim.
139. Homotopy terminal evidence is admissible only if its full-model gate
     records one forced AMP overflow with no optimizer/model/EMA/scheduler/
     selector-schedule advance and the authorizer verifies that record.
140. Physical feasibility caps are control-plane float64 values. Never cast
     them to RGB (`uint8`) or AMP policy-score (`float16`) dtype before
     metadata validation, evidence writing or hard/soft route comparison.
141. Gate `1177714` at `bc503fc` is an immutable failed diagnostic, not an
     admissible gate. The physical-cap fix passed, but the gate-only
     unselected-frame perturbation assumed floating inputs and called
     `randn_like` on real-loader `uint8` RGB. A real-loader validation helper
     must preserve selected uint8 frames exactly and use a dtype-valid,
     deterministic perturbation for unselected frames.
142. Current replacement evidence is commit `b987c8c`, tree `d33d9194`, P0
     SHA-256 `a246dc8c...f9b99`, and gate Job `1177715`. This authorizes only
     gate execution. Do not say the four official-60 arms are submitted until
     `authorization.json` is present, hash-verified and `ok=true`.
143. Gate `1177715` failed the P1 representation contract. The physical and
     selected-axis objectives differed by 24.10% on exact-uniform real data;
     cls/reg differed by 25.35%/22.41%. This is not AMP jitter or a tolerance
     issue. Do not relax the `1e-4` threshold, relabel the failure as a gate
     bug, or submit official-60 arms from `b987c8c`.
144. Endpoint-inclusive exact uniform at `T=768,K=384` has 382 gaps of 2 and
     one gap of 3. A local physical center/stride rewrite and a selected-axis
     GT time warp are therefore not mathematically identical. The next route
     must explicitly choose one geometry or revise the representation; it
     cannot claim native physical semantics and exact selected-axis parity at
     the same time.
145. The selected-axis successor is an explicit representation decision, not
     a claim that physical and selected-axis losses are equivalent. Never
     re-enable `physical_grid_actionformer` in this matrix.
146. The Uni companion is one mixed training batch, not two detector forwards,
     distillation, an ensemble, or an inference branch. Its detector-to-selector
     gradient is intentionally absent on uniform rows.
147. `implemented_local_static` is not `tested`, `experiment_running`, or an
     mAP result. Record positive gate and Slurm Job IDs before promoting state.
148. Gate Job `1177721` is not a failed training experiment. It stopped before
     any optimizer update because its ownership-only check used a fresh
     GradScaler without the formal training engine's replay semantics.
149. Do not claim the global structured DP is numerically broken from
     `1177721`. Read-only Job `1177724` found finite gradients for each
     transition loss and their sum at scales 1 and 65536, exact soft mass 384,
     and no non-finite element on the same real T=768/K=384 data contract.
150. The direct learned-policy arm must share the homotopy arms' detector-
     gradient warmup/ramp. Otherwise a future mAP difference confounds policy
     initialization with detector-gradient exposure.
151. Every selected-axis official-60 arm, including exact-uniform, must pass a
     real full-model gate at seed 3407. The exact-uniform gate must compare
     actual selected positions with canonical endpoint-uniform positions; a
     route name or alpha value is insufficient.
152. An AMP gate must exercise the production replay path. A forced overflow
     must not advance optimizer/model/selector schedule/LR scheduler/EMA, and
     the successful replay must advance each exactly once. Never solve this by
     ignoring non-finite gradients, clipping before diagnosis, or lowering a
     fail-closed threshold without evidence.
153. Job `1177732` is a replacement full-model gate for exact commit
     `c2de186`; it is not one of the four official-60 training arms. Until the
     hash-bound gate suite exists and the four positive training Job IDs are
     recorded, do not say the experiment matrix has been submitted.
154. Gate `1177732` failed because its parameter-change proof was executed at
     the formal warmup learning rate of zero. Do not relabel this as a model
     numerical failure or alter the formal training warmup to satisfy a gate.
     The audit must move to the first nonzero successful step instead.
155. Exact replacement evidence is commit `1af6ff8`, clean 35-test snapshot
     and CUDA gate Job `1177733`. Passing static contracts inside that job is
     not final authorization; no official-60 arm is submitted until the
     hash-bound four-arm gate suite exists and reports `ok=true`.
156. Gate `1177733` is the admissible four-arm authorization: it completed
     successfully and sealed suite SHA-256 `38d5e185...c09`. Formal Jobs are
     `1177734` exact-uniform, `1177735` direct-0.25, `1177736`
     homotopy-0.25 and `1177737` homotopy+uniform-companion. Do not replace
     these with old gate/training Jobs or call them successful before terminal
     epoch-59 EMA evaluation exists.
157. All four current arms are selected-axis K=384 official-60 experiments.
     They test optimization and gradient routing, not physical-grid semantics,
     dynamic budget, another detector, or an online-TAD claim. Greater than 65
     is the GO criterion, not a guaranteed outcome.
158. Jobs `1177734-1177737` are immutable training-entry deployment failures,
     not model experiments: all stopped before model construction because the
     60-epoch configs were routed through the legacy epoch-131 checkpoint
     criterion. Never report them as running, completed, or mAP evidence.
159. The replacement route is exact commit `cb89586` with explicit formal
     protocol `duca_selected_axis_optimization_v1`; clean Linux evidence is 38
     focused plus 23 required regression tests, and replacement gate Job is
     `1177776`. Do not reuse the `1af6ff8` gate suite for this new commit.
160. Gate `1177776` completed successfully and sealed suite SHA-256
     `76628abd...0a27`; the admissible replacement formal Jobs are `1177779`
     exact-uniform, `1177780` direct-0.25, `1177781` homotopy-0.25 and
     `1177782` homotopy+uniform-companion. They have entered epoch 0 but have
     no checkpoint or mAP yet.
161. An isolated AMP skip followed by bounded replay is expected audit
     behavior, not a collapse. Escalate only on non-finite loss, replay
     exhaustion, repeated clustered skips, Traceback/OOM, or missing successful
     update accounting at the epoch boundary.
162. Epoch-4 selector diagnostics are mechanism evidence, not model selection
     or mAP. The current learned policies improve exact boundary hits while
     losing roughly 9--10 points of radius-1 boundary recall versus uniform;
     never report the r0 gain alone as better boundary coverage.
163. Before successful step 2100, `duca_detector_grad_w=0`. Detector cls/reg
     training may be active, but no detector gradient is yet allowed to update
     the selector. Early checkpoints cannot support or refute the protected
     end-to-end bridge claim.
164. The raw `abs(delta p_action)` transition AUROC exceeding the learned
     policy AUROC at epoch 4 is a bounded optimization clue, not authorization
     to replace the current formal arms mid-run. Await terminal matched mAP;
     any residual-delta successor requires a new exact commit and protocol.
165. `continuous_policy_homotopy` means continuous score/soft-occupancy
     interpolation. The hard Viterbi path is piecewise constant and may swap
     multiple frames at a score crossing. Never claim continuous hard-frame
     motion without an explicit alpha-sweep trajectory audit.
166. The current Uni companion is an AdaFocusV2-inspired batchwise uniform-
     policy companion: one row is learned and one row is uniform in a single
     detector batch. It is not the official same-video policy-view plus second
     random-view forward and must not be described as an exact reproduction.
167. With T=768/K=384/G=2, radius-2/radius-4 boundary recall is close to a
     feasibility consequence. Use r0/r1, endpoint distance, gap histogram,
     adjacent-selection rate and short-window freedom as discriminative
     selector metrics.
168. Do not say the current selector lacks clustering freedom. Excluding
     all-frame short windows, epoch-4 learned arms replace about 185--187 of
     384 uniform positions and raise adjacent-selection rate from 4.4% to
     roughly 35%. The observed failure is misplaced/aggressive clustering.
169. The old claim that epoch-4 alpha about 0.03 already retained only 51.7%
     of uniform positions is invalid. `1178004` exported in eval mode, so its
     learned arms used inference endpoint alpha=1. Exact trajectory replay
     keeps all audited paths uniform through alpha 0.1 and first changes them
     around alpha 0.3--0.4. Continue to call the hard path piecewise constant,
     but do not infer a small-alpha jump from endpoint-only export records or
     call swaps accumulated over a coarse alpha interval instantaneous.
170. The epoch-4 GT-informed heuristic is not a feasible oracle: although its
     r0 recall is 0.2472, mean max hole is 11.36 versus the frozen G=2. Never
     use it to claim achievable performance or allocation headroom under the
     current selector contract.
171. Do not blame epoch-4 coarse AUROC 0.4634 on GT/window coordinates. All 64
     records and 46,527 labels were independently reconstructed from original
     THUMOS metadata and matched exactly; +/-1 candidate shifts alter AUROC by
     less than 0.002. It is genuine early under-optimization on a limited
     40-video diagnostic subset.
172. At the 2026-07-21 10:23 checkpoint, the four formal arms were only around
     1400--1500 successful updates and `duca_detector_grad_w=0`. Do not call
     their current losses or epoch-4 selector geometry an end-to-end detector-
     feedback result. The first saved post-bridge diagnostic checkpoint is
     epoch index 24, after update 2100 and during the bridge ramp.
173. Job `1178384` is a read-only normalized raw-delta residual sweep over
     epoch-4 endpoint scores. GT is evaluation-only. Do not select a production
     gamma from this validation subset, call it a trained model, report its
     boundary metric as mAP, or deploy a successor before terminal matched mAP.
174. Do not call P0 detector pretraining. P0 freezes and skips the complete
     VideoMAE/AdaTAD path; it is frontend-only. Exact-uniform AdaTAD warmup is
     the first 1000 successful updates of official-60 and adds no detector
     epochs beyond the fixed 6000-update budget.
175. Do not run uniform AdaTAD and frontend checkpoint selection through one
     aggregate optimizer merely because their forward paths are decoupled.
     Shared global clipping, AMP and scheduler state can still couple their
     optimization. A concurrent variant requires separately audited optimizers
     and is not the current frozen route.
176. Historical separate official-ASFormer quality is motivation only. Its
     checkpoint selection consumed THUMOS test evidence and is not an
     admissible P0 result. The new P0 must select only on the sealed training
     holdout and pass the preregistered mechanism gates.
177. Job `1178480` is an immutable pre-training deployment failure. It made no
     optimizer update and has no model evidence. Replacement exact commit is
     `6f2ed48`; only Job `1178487` may advance the two-stage experiment state.
178. A P0 mechanism-gate failure is a scientific HOLD, not permission to pick
     the least-bad checkpoint or inspect test mAP. The serial DAG must stop
     before official-60 when no candidate passes all frozen gates.
179. Jobs `1177779-1177782` are no longer running. All four failed on the same
     `Disk quota exceeded` infrastructure condition around epoch 26--27 and
     have no terminal EMA or mAP. Never report them as negative method results
     or silently resume them under a different commit/protocol.
180. Job `1178487` was still listed RUNNING when `/data` reached 100%, but its
     log stopped advancing near P0 epoch 1. Do not call it healthy, converged,
     failed, or scientifically informative until storage headroom and the
     exact process/artifact state are audited.
181. Accounting later sealed Job `1178487` as `FAILED/1:0` with no checkpoint.
     Do not resume or describe it as P0 evidence; a replacement requires a new
     exact-commit receipt after storage preflight.
182. The selected-axis Jobs `1177779-1177782` have one recoverable state each:
     exact `epoch_24.pth` with model/EMA, optimizer, scheduler, scaler and RNG.
     Earlier records were intentionally deleted. Never look for epoch 4/9/14/19
     or treat epoch 24 as terminal mAP evidence.
183. Remote checkpoint retention is now one CRC-validated numeric record per
     independent directory. Do not restore periodic historical checkpoints,
     delete pretrained/data assets as "checkpoints", or bypass the sealed
     manifest `a06d3062...04b60a` when auditing storage.
184. Resume gate `1178581` is valid, but Jobs `1178582-1178585`,
     `1178614-1178617` and `1178633-1178636` are immutable launcher failures
     before checkpoint restoration, caused by missing commit, `BASE`, and gate
     environment exports. The only current continuation is v4 root
     `duca_selected_axis_cb89586_resume_e24_v4_20260721_135701`, two-GPU/two-
     wave Job `1178642`. Its runtime-binding preflight is hash-bound by
     `dae27758...09d1`. Do not restart from epoch zero, change commit/config/
     seed, or describe epoch 24 as a selected checkpoint.
185. Two-stage Job `1178591` is a from-scratch P0 restart because failed Job
     `1178487` wrote no checkpoint. The rejected parallel DAG hit only
     `AssocMaxSubmitJobLimit` and was transactionally rolled back; never count
     it as another model run or infer a scientific failure from that rejection.
186. Exact-commit audit invalidates the claim that P0 optimizes only the three
     configured objectives. `duca_losses()` hidden nonzero defaults enter the
     returned selector loss set and therefore `cost`. Formal routes require a
     complete explicit allowlist; `_delete_=True` is not sufficient.
187. Current transition-distribution supervision reaches ASFormer hidden
     features. Do not describe P0 as an action-only coarse model observed by a
     separate selector until hidden features are detached and per-loss gradient
     ownership is verified.
188. Zero scheduled frontend weights are not a freeze contract while the
     parameters remain in shared AdamW, global clipping and whole-model EMA.
     Require zero optimizer steps/state plus byte-invariant parameters and
     buffers through warmup.
189. The protected RGB-slope bridge is connectivity evidence only. It does not
     include hard-frame substitution, selected-axis GT remapping or detector
     assignment changes. Keep it diagnostic unless legal hard-swap alignment
     is measured; do not call either bridge or detached swap loss equivalent to
     terminal mAP.
190. Exact-uniform/pure-delta anchored bounded residual selection is the leading
     repaired proposal, not a unique or final method. Local radius, residual
     bound, hard-swap cadence and numerical GO/KILL thresholds require frozen
     reachability, variance, cost and matched-mAP evidence.
191. Job `1178591` is now a protocol-invalidated diagnostic. Its logs may be
     analyzed, but its current P0 gate must not automatically unlock a
     paper-grade official-60 matrix. Job `1178642` is a separate immutable
     legacy continuation and was not the exact commit audited here.
192. The two exact-commit audits agree on the HOLD diagnosis and bounded
     pure-delta residual family, but not on detector feedback. Detached
     hard-swap utility distillation and local-cell soft-RGB backward are
     distinct candidates; never describe them as one implemented method.
193. Do not freeze one-per-cell geometry, `{-1,0,+1}` radius or residual scale
     from prose alone. First compare exact uniform, pure delta, a local
     GT-boundary oracle and the global exact-K/G2 oracle on the sealed training
     holdout to establish reachable headroom.
194. The current negative entropy term and `positive_prior=0.5` BCE are not a
     protected coarse-action objective. Formal P0 requires explicit loss keys,
     separate positive/negative BCE means and no selector loss path into the
     coarse trunk.
195. Padded frames currently pass through train-mode BatchNorm before masking.
     A repaired coarse stem must use valid-only execution or padding-invariant
     normalization and pass a padding metamorphic test before P0 evidence is
     admissible.
196. In repaired official training, the coarse model remains frozen/eval.
     Detector influence is allowed only on the residual scorer after a real
     legal hard-swap alignment gate; gradient connectivity alone is not a
     method result.
197. Do not launch a full explanatory global-policy matrix before the bounded
     model passes matched `U/D/R0/R1` terminal-mAP and cost tests. Historical
     global arms remain diagnostics until the repaired candidate has evidence.
198. A one-per-uniform-cell family does not automatically satisfy the exact
     uniform gap cap. Local delta and local GT oracles must solve the cell and
     cap constraints jointly; independent per-cell argmax is inadmissible.
199. Do not use 20-bit lexicographic MILP blocks for the local reachability
     audit. HiGHS returned variables -1 and 2 on a real holdout row despite an
     OPTIMAL status. Keep strict bounds validation and the tested 8-bit blocks;
     never round an out-of-bounds solution into an oracle.
200. The epoch-19 checkpoint from Job `1178591` is diagnostic input only. The
     job completed its first P0 candidate but failed in the later exporter, and
     its training contract had already been invalidated by hidden losses and
     gradient leakage. It cannot initialize a paper-grade official run.
201. P0 is not repaired by setting three visible weights alone. Formal P0 must
     expose all 19 loss weights, construct no graph for zero-weight losses,
     detach transition supervision from coarse hidden features, use padding-
     invariant spatial normalization and disable global cross-branch clipping.
202. The 120-record training-holdout oracle audit shows that one-per-uniform-
     cell selection matches the global oracle on all reported boundary recall
     and both-endpoint coverage metrics. Do not blame local-cell reachability
     for the present learned-selector deficit; the active bottlenecks are coarse
     evidence and score learning.
203. At `K=384`, exact uniform already has radius-one boundary recall `0.9998`.
     Do not use r1 coverage as the primary P0 winner criterion or as a proxy for
     detector mAP. Retain exact-distance/r0 diagnostics, but terminal matched
     mAP remains decisive.
204. The final MILP position tie-break is not a scientific objective. When all
     semantic objectives are exactly pinned, disabling this unstable tie-break
     is admissible. Rounding out-of-range variables or replacing the
     non-additive both-endpoint objective with additive DP is not.
205. Do not turn experiment launchers into a second method. For this P0 repair,
     use one real-data CUDA one-step gate followed by three sequential frontend
     candidates. Do not automatically launch unrepaired official-60 arms.
206. Job `1178774` is a gate-classifier failure, not a model-gradient failure.
     The executed spatial parameters are named `probe_module.spatial_stem.*`;
     the old gate searched for `spatial_encoder`. Never cite this job as
     evidence that actionness supervision cannot train the spatial frontend.
207. Do not call one-frame-per-local-cell DUCA a final smart sampling method.
     It is exact-uniform local perturbation: it cannot release a background
     cell's frame budget or place multiple observations near a true boundary.
     Use it only as a conservative diagnostic/ablation unless a later design
     adds cross-region quota transfer under a hard coverage/max-gap contract.
208. Do not submit paper-grade official-60 main experiments from the local-cell
     P0 merely because its gate or holdout loss passes. The user's current
     route裁决 is that local-cell breaks the original indirect boundary-
     allocation principle; terminal mAP from such runs, if collected, is
     diagnostic rather than final-method evidence.
209. Do not design or implement another global exact-K/max-gap decoder. The
     existing `global_structured_topk` already performs full-window joint
     allocation and permits cross-region quota transfer; the formal selected-
     axis source is commit `cb89586`.
210. Do not respond to weak or misplaced global-policy clustering by shrinking
     the final feasible set to one-frame-per-cell. That changes the research
     hypothesis. Repair coarse evidence, scorer optimization and training
     isolation while retaining the global feasible family.
211. The next implementation must reuse the selected-axis global selector and
     merge only named contracts from existing commits: P0 repairs from
     `5d17dcb/6c56e11`, curriculum structure from `6f2ed48`, and protected
     gradient components from `cb89586/ee05f61`. Never copy an entire tree or
     create parallel actionness, decoder, detector-wrapper or profiler code.
212. Before creating any DUCA model class or config family, consult
     `duca_model_version_registry.md` and record the exact missing contract.
     A new name, worktree or launcher is not evidence that a new model is
     required.
213. Until the matched U/G0/G1/G2 verdict is available, the only admissible
     DUCA mainline is exact commit `63e25eb17e523d369f73434ed4d9b6446608861a`
     on `codex/duca-global-curriculum-20260721`. Do not fork another selector
     family to respond to a pending result.
214. Jobs `1178642` and `1178863` are predecessor/diagnostic runs. The former
     lacks the repaired P0/global-curriculum contract; the latter uses the
     rejected local-cell feasible set. Neither may silently replace active Job
     `1178989` in the paper main table. Jobs
     `1178911/1178927/1178933/1178947/1178975` are cancelled or failed history.
215. U/G0/G1/G2 are configurations of one shared model, not four new models:
     U is exact-uniform; G0 disables detector feedback; G1 enables protected
     feedback to the transition scorer only; G2 adds a training-only uniform
     companion with learned-row gradient-exposure normalization. Do not copy
     their files into separately named model classes.
216. A queued job, passed focused tests, gradient connectivity or improved
     boundary proxy is not a greater-than-65 result. Promotion requires the
     matched terminal-EMA official-60 mAP table from the exact frozen commit.
217. Do not repeat the `cb89586` direct-from-scratch detector-gradient arm as
     a proposed fix. Its terminal EMA is `63.7102`, below the same-commit
     exact-uniform `64.4580` by `0.7478`. The unresolved question is the
     repaired P0-frozen, scorer-only protected gradient in current V8, not
     whether the old direct route should be rerun under a new name.
218. Historical local-cell functions may remain in shared source files for
     reproducibility, but their presence does not make them part of the active
     model. Active-route identity is determined by the frozen config and gate:
     U/G0/G1/G2 must use `global_structured_topk` and tests must reject
     `local_cell_deformation`.
219. The 2026-07-21 inventory of 43 relevant OpenTAD Git trees is closed in
     `duca_model_version_registry.md`. If a requested capability already maps
     to an entry in its reuse table, modify that exact implementation; do not
     create a synonymous class, worktree, config family or launcher. A new
     implementation is admissible only after the registry names a testable
     contract that every recorded version demonstrably lacks.
220. Do not use radius-one boundary recall as a P0 hard gate or primary ranking
     key at K=384. The exact-uniform reference is already approximately 0.9998,
     so r1 is diagnostic only. P0 selection must prioritize exact/r0 boundary
     and short-action endpoint evidence, then endpoint distance, transition
     discrimination and coarse action metrics.
221. Do not repeat a P0 grid that changes only transition/boundary loss scales
     while the corresponding objectives update disjoint parameter groups. The
     next bounded diagnostic must hold those losses fixed and test the observed
     learning-speed mismatch between coarse evidence and the transition scorer.
222. Uni-AdaFocus does not send heavy-branch gradients through its hard
     temporal indices; `weights_T.detach()` is explicit in the official code.
     Reuse its component LR grouping and auxiliary task-supervision principle,
     not a fictional hard-index gradient claim or an automatic MobileNet
     requirement.
223. Commit `e0397ec` is a P0 evidence-contract revision of the existing V8
     global model, not a V9 selector. Its three P0 variants differ only in
     coarse-trunk/action-head/transition-scorer learning rates while sharing
     the exact same losses, architecture and downstream U/G0/G1/G2 matrix.
     Never duplicate them as model classes or claim three methods.
224. Job `1178911` is immutable cancelled history: it ran `00:09:01` under the
     rejected radius-one/old-LR P0 protocol. The only active replacement is
     the later exact-commit job recorded below; neither job may be relabelled
     or merged with the other's evidence.
225. Job `1178863` completed its first local-cell P0 training candidate and
     failed only when `python tools/bata/export_duca_selection_quality.py`
     could not resolve the top-level `tools` package. This is a launcher
     import-path failure, not a model-numerics verdict. Any active P0 launcher
     must invoke both quality utilities via `python -m tools.bata...`.
226. Job `1178927` was cancelled with zero runtime after rule 225 exposed the
     inherited launcher defect. Job `1178933` then failed its clean-tree guard
     after three seconds and has no model evidence. Job `1178947` subsequently
     failed only the over-broad optimizer-group evidence classifier before any
     update. Job `1178975` later executed one valid P0 update but failed an EMA
     representative-parameter audit. The sole active global-curriculum job is
     `1178989` at exact commit `63e25eb`; do not count any predecessor as a P0
     candidate or paper run.
227. The active route is identified by frozen configuration, not by every
     branch retained inside shared source. G0/G1/G2 must resolve to
     `global_structured_topk`, K=384 and max-hole=2; `local_cell_deformation`
     may remain for reproduction only. Before writing any DUCA class or
     decoder, a focused test must first demonstrate a missing contract in the
     existing V8 implementation and the registry must name that gap.
228. Every repository tool that imports `tools.bata` must be launched with
     `python -m tools.bata...`; checking only export/analyze is insufficient.
     Exact commit `6b6363e` closes the two aggregator instances and tests all
     four module-entry calls. Do not reintroduce file-path execution.
229. Never upload compatibility probes, temporary Python, bundles or audit
     helpers inside an exact experiment snapshot. Job `1178933` proves the
     clean-tree guard will correctly reject that contamination. Put temporary
     assets outside the snapshot, remove them after use, and reverify HEAD plus
     empty porcelain before submission.
230. Existing standalone official-ASFormer checkpoint SHA-256
     `34e4d510...5f3bbba` is BatchNorm-compatible but not strict-load compatible
     with the active GroupNorm P0 because of BN running-stat keys. Never use
     silent `strict=False` or restore batch-composition-sensitive BatchNorm.
     A warm-start conversion is admissible only as a preregistered fallback
     after all current P0 candidates fail and must explicitly account for every
     retained and discarded state key.
231. Job `1178947` is a P0 evidence-classifier failure, not a model or optimizer
     result. Official ASFormer attention blocks contain internal `conv_out`
     projections; only top-level `encoder.conv_out.*` and
     `decoders.<stage>.conv_out.*` are binary action heads. Never classify a
     parameter by the substring `conv_out` alone or cite this job as evidence
     against component learning rates.
232. The only admissible DUCA mainline after the gate correction is commit
     `63e25eb17e523d369f73434ed4d9b6446608861a`, Job `1178989`. It is V8 with
     a corrected gate, not V9. Do not create another selector, decoder, model
     class or worktree while its P0/U/G0/G1/G2 verdict is pending.
233. A broad `test_duca*.py` invocation must run with the exact repository as
     its working directory. Path-based failures from `/data/home/sczc063` and
     failures in intentionally retained historical-route tests are not a
     regression in V8. Report the exact selected test set and working directory
     instead of converting an invalid aggregate invocation into model evidence.
234. Never prove EMA branch updates from one arbitrarily selected representative
     parameter. At decay 0.999, one valid FP32 optimizer step can be rounded to
     zero on that tensor even when other parameters in the same branch change.
     Audit every trainable parameter in the existing group, retain representative
     deltas only as diagnostics, and never turn this evidence-gate failure into a
     new model, optimizer, selector or training route.
235. Do not rerun or rename the completed `cb89586` V5 training arms. Their
     terminal Avg-mAP values are uniform 64.4580, direct-0.25 63.7102,
     homotopy-0.25 63.0601 and homotopy plus uniform companion 63.6931. The
     companion partially repairs homotopy but none beats uniform. The only
     unresolved bounded question is current V8's repaired P0, frozen coarse
     branch and scorer-only protected feedback.
236. Uni-AdaFocus's inverse-CDF temporal sampler guarantees exactly K ordered,
     unique indices over learned probability mass; it does not guarantee a
     temporal max-gap, boundary recall or uniform physical-time coverage. Its
     robustness also comes from a uniformly observed cheap global stream and
     reuse of global features in the final classifier. Never cite the sampler
     itself as proof that DUCA can remove its TAD-specific coverage contract.
237. Do not directly add or concatenate raw VideoMAE features and coarse
     classification hidden states, and do not let TAD loss freely rewrite the
     coarse action head/trunk. Any post-V8 fusion must use an explicit
     timestamp-aware adapter, a zero-initialized baseline-preserving residual,
     and audited loss-to-parameter ownership. Context fusion and physical-grid
     gap filling are separate hypotheses and require separate matched arms.
238. Never use radius-one exact endpoint-event coverage as an active scorer
     loss under `max_unselected_hole=2` without first proving nontrivial
     headroom. Every internal three-position event is then covered by every
     feasible path, so its probability is one and its gradient is zero. Use a
     nontrivial event such as rounded radius zero or change the K/G contract in
     a separately preregistered experiment.
239. Never let a unit/plumbing gate write `formal_training_unlocked=True` for
     G1/G2 unless the current detector bridge has produced the preregistered
     real legal hard-swap alignment artifact. Gradient existence and
     hard-forward equality do not establish utility-sign alignment.
240. Never describe protected structured transport as direct differentiation
     through hard indices. It is a hard-forward, surrogate-backward local
     transport; config metadata, paper text and evidence tables must use that
     exact boundary.
241. A P0 candidate `completion.json` with `ok=true` certifies finite execution,
     protocol binding and checkpoint production only. It is not a mechanism
     pass or a winner. For `lr_control_c25_a50_s100`, epoch-20 coarse macro
     AUROC improved to `0.624512`, but the policy transition AUROC remained
     below simple `abs(delta p_action)` (`0.521321` versus `0.553237`),
     radius-one endpoint recall fell below uniform (`0.883237` versus
     `0.999775`) and endpoint distance was worse (`0.538120` versus
     `0.477457`). Never rerun or promote this profile solely from its successful
     completion receipt; wait for the preregistered quality selector and all
     remaining candidates.
242. Never turn “avoid duplicate broad-band mass” into “do not cluster around
     boundaries.” The historical GT Oracle deliberately selects each endpoint
     center plus/minus radius two before uniform filling. The desired behavior
     is a centered, bilateral and quota-limited boundary burst for every
     start/end, plus global residual coverage. Radius-zero unique coverage is
     only an anchor; it is not a complete selector objective and must not force
     one-frame-per-boundary behavior.
243. Do not claim that `budgeted_center_radius_decode`, learned context radius,
     GAS-VT `boundary_bracket_loss`, move25/move50 dilation, CellCF or
     left/right evaluation metrics already implement Oracle-like bilateral
     boundary bursts. Each provides only a partial component. The complete
     missing contract is deploy-visible transition centers plus capped
     left/center/right allocation, endpoint deduplication and residual global
     exact-K spending inside the V8 official-AdaTAD graph. Reuse the existing
     scorer/DP/diagnostics after V8 adjudication; do not create a synonymous
     selector or revive a ledger/local-cell route.
244. The canonical paper product is one offline-TAD pre-backbone acquisition
     plugin followed first by the official-derived AdaTAD/ActionFormer path.
     Do not rename it Online TAD, a new detector, or three independently
     deployed coarse/selector/detector models.
245. Radius-zero unique endpoint coverage is only a center anchor. Never ship
     it as the full objective or use it to suppress Oracle-like bilateral
     clustering. The complete target requires capped left/center/right burst
     semantics, endpoint fairness, overlap deduplication and residual global
     coverage.
246. Do not freeze `max_unselected_hole=2`, a burst radius or a 3--5-frame
     quota as scientific truth without a train-split Oracle reachability audit.
     Choose the weakest coverage cap that retains context while preserving
     measurable detector-mAP headroom; never select it from test mAP.
247. Do not run or interpret G1/G2 before corrected G0 has a terminal matched U
     comparison and the current bridge passes the real legal hard-swap
     alignment gate. If G0 does not beat U, adding feedback, companion, epochs
     or learning-rate searches is prohibited for that frozen K/G route.
248. Selection AUROC, endpoint recall, adjacency and boundary distance are
     mechanism gates only. They cannot replace official terminal-EMA TAD mAP,
     and FLOPs alone cannot replace measured full-stack latency/memory/energy
     including the dense cheap probe.
249. The boundary-burst plan is `designed_not_implemented`, not V9. A new
     version identity requires one exact implementation commit that extends the
     existing V8 scorer/DP, passes the preregistered contracts and does not
     duplicate selector, decoder, detector wrapper or worktree families.
250. During the official detector stage, TAD loss may update the detector and,
     after alignment, scorer/burst parameters. It must not freely rewrite the
     binary action head or coarse ASFormer trunk; a different ownership policy
     requires a separate gradient-conflict and coarse-calibration gate.
251. Do not promote EU-CRR or any post-VideoMAE coarse residual to V9 or the
     final method from a code review. Its current status is only
     `discussed_conditional_diagnostic_not_authorized`; G23 remains the frozen
     acquisition objective.
252. An exact-uniform U0/U1 fusion result cannot establish learned selection or
     Oracle-like boundary clustering. It tests frozen coarse representation
     value under fixed positions only. Never use it to replace transition-
     center, bilateral-burst, quota, fairness or residual-context evidence.
253. If an EU-CRR matrix is ever run, report all causal contrasts: `U1-U0`
     (fusion under uniform), `L1-L0` (fusion under learned), `L0-U0`
     (selection without fusion) and `L1-U1` (selection with fusion). A sole
     `L1-U1` gate cannot isolate feature reuse.
254. Failure of EU-CRR kills only post-VideoMAE coarse residual reuse; failure
     of a physical-grid matched test kills only that coordinate hypothesis.
     Neither result automatically kills the distinct Oracle-calibrated
     boundary-burst acquisition route.
255. Selected coarse hidden with a full-window ASFormer receptive field is not
     the same as feeding the complete dense coarse sequence to the detector.
     Describe it exactly, and once detector-side fusion is active do not call
     the method strict pre-backbone-only.
256. `research-wiki/duca_final_model_contract.md` is the canonical DUCA target.
     Do not reconstruct the final method from an older review, job, worktree or
     chat summary. Any intentional target change must update that contract,
     query pack, registry, decision history and log in the same turn.
257. The final coarse action head is binary action/background supervised. GT
     endpoints supervise the class-agnostic transition/burst selector only;
     never add a class-specific coarse start/end head and still call the method
     indirect boundary localization.
258. The desired selection is Oracle-like bilateral boundary clustering plus
     residual global coverage, not one observation per boundary and not
     unlimited Gaussian mass. Radius, quota and max-hole must come from the
     train-split Oracle reachability gate.
259. The two-stage curriculum is one deployed model. P0 skips the detector;
     official-60 begins with exact-uniform warmup, freezes coarse and then
     adapts scorer/burst plus detector. Do not relabel this as three separately
     trained/deployed models.
260. Do not use large action/transition loss weights to simulate a frontend
     phase. Enforce phase ownership structurally: skip the detector in P0,
     detach coarse hidden for selector losses, and use normalized bounded
     selector terms with explicit optimizer/gradient ownership.
261. Training-mode dense materialization proves heavy-backbone frame reduction
     only. A paper-level total-cost claim requires the deployment path with a
     low-resolution dense proxy and selected high-resolution materialization,
     including any random-access or second-decode overhead.
262. The primary backend keeps official-derived ActionFormer head/loss/NMS and
     uses selected-axis mapping plus original-time inverse metadata. Do not
     silently activate physical-grid or post-VideoMAE fusion inside a matched
     U/G0 experiment; each is a separate, explicitly named hypothesis.
263. 2026-07-22 起，DUCA 的完整论文路线固定为 V8 负诊断封存 -> R0 Oracle 可达性 ->
     R1 数学/代码/真实模型门禁 -> R2 P0 三臂机制 -> R3 same-commit U/G0 -> R4
     alignment 后 G1/G2 -> R5 三种子、K384/K256、第二 backend 与完整成本。Job
     `1178989` 不是论文主结果；其第一候选已经说明粗动作证据能学习但旧 scorer 未能转成
     更好边界分配。禁止把剩余 V8 候选、loss 下降或 plumbing gate 宣称为 final DUCA。
264. 不再发起开放式、重新发散主架构的 Pro 讨论。Pro 只允许在 R0 冻结 K/G/radius/quota、
     R1 exact-commit 长训前、R4 legal hard-swap alignment 三个边界做定向审查；在停止条件
     未触发前，不得借讨论重新引入 local-cell、actionness top-k、X3D 主线、dynamic MUST、
     新 selector/decoder/worktree 或 post-VideoMAE fusion。
265. R0 selected-axis 留出诊断必须同时匹配 dataset subset、evaluator subset 与 evaluator
     blocked-video 集。只把 dataset 限到 40 个视频、却让 evaluator 读取全部 training GT，
     会把其余视频当漏检；这是无效 mAP，禁止再次运行或引用。
266. 冻结 `transition_beta0 epoch_131` detector 曾在完整 training subset 上训练，因此
     R0 内部留出 mAP 的绝对值不是泛化结果。只允许使用同一 detector 下 Oracle-U 的
     相对可行域增益；论文主表仍以 R3 test terminal-EMA mAP 为准。
267. `fdf25f5` 只是 boundary-burst implementation candidate。独立 MAX、Linux focused、
     real CUDA gate、R0 与 R3 未闭环前，不得登记 V9、宣称超过 65 或启动 G1/G2。
268. K=384 的预算合同是逐样本 `selected_count=min(384, valid_len)`，不是跨短视频与
     尾窗求均值后强制等于 384。质量分析器已逐样本 fail-closed；P0 候选门禁只可读取
     该已验证合同并要求平均值不超过 384，禁止再次用错误的均值门禁浪费完整 P0。
269. 禁止 exporter 用 `budget=len(selected_positions)` 循环自证 exact-K。record 必须分别
     写入独立 requested K、`min(K,valid_len)` effective K、requested G 和实际 max-hole；
     analyzer 必须逐样本 fail-closed 并汇总 violation count/max，而不是无条件写布尔值。
270. P0 endpoint 评价必须与训练使用同一离散化：`floor(start)` 与 `ceil(end)-1`，并消费
     crop endpoint validity。窗口切口不是动作状态转变，不得作为真边界监督或 R0 Oracle。
271. R0 必须先于 P0 且以 `afterok` 阻断 DAG；只有同一 frozen detector 下受约束的双侧
     quota burst Oracle 相对 U 有正 mAP headroom，才允许启动 P0/official60。单 endpoint-hit
     solver、无条件 `ok=true` 或并行旁路 R0 均不可作为可达性门禁。
272. U arm 不得写入一个实际未加载的 P0 checkpoint hash。U 的 frontend artifact 应显式
     `not_applicable`；每个 learned arm 只绑定自己的 P0 winner。split manifest 必须封存并
     在每个消费者核验 annotation/train-list/holdout-list hash。
273. `899630a5ef4927e78ef4ca6b8cc51fdf754056da` 是当前唯一 boundary-burst exact
     candidate。不得回到 `fdf25f5` 或把 Linux 数值可移植测试提交误说成模型改版；其
     forward 合同仍是同一 center-conditioned R2Q3/R4Q5 quota support 和既有全局 DP。
274. 直通估计张量的 hard forward 可有浮点消去误差；测试必须用明确的微小容差核验
     0/1 配额形状，同时另验 exact quota/count。不得以 `torch.equal` 的逐位失败改写模型，
     也不得借容差放松逐样本 exact-K/G 与整数位置合同。
275. 独立 MAX 对 `899630a` 给出 GO 以及 exact CUDA gate 通过前，禁止提交正式长训。
     通过后只允许现有 R0->P0->gate->U/Gaussian/R2Q3/R4Q5 DAG；G1/G2 继续等待
     real legal hard-swap alignment，不得随四臂一并偷跑。
276. 独立 MAX 已对 `899630a` 给出 `HOLD_FIX_REQUIRED`。不得把 Linux `136+23` tests
     当成正式部署许可；四臂 runtime binder/gate mapping、pooled crop validity 与
     provenance/hash 链必须在新 exact commit 中全部关闭，并由全新独立 MAX 重审。
277. 当前不是重新设计主架构的时点。V8 已机制 HOLD，boundary-burst 主体未被 KILL；
     在 R0 projected-Oracle、R2 corrected-selector 和 R3 U/G0 结果出现前，禁止以 Pro
     讨论为由新增 fusion、dynamic budget、local-cell、新 selector 或第二 decoder。
278. 当前论文位置必须写作“R0 前的 R1 部署/证据合同修复”。不得写作主实验 running，
     也不得把 pooled validity focused test、静态审计或 runtime gate 当作 mAP 证据。
279. `aa3352ecf803c81d007a62ed5398667d9551684b` 已替代 `899630a` 成为唯一候选。
     `899630a` 的 HOLD blocker 不得再次修一遍；后续只消费 aa3352e 的全新独立 MAX、
     real CUDA gate、R0/P0/DAG 证据。远端 `139+23` tests 仍不是部署许可或 mAP 证据。
280. 只有独立 MAX `019f8647-ad93-70f3-a763-218f7552ac95` 对 aa3352e 给出
     `GO_TO_REAL_CUDA_GATE` 才能运行 CUDA gate。若 HOLD，只修其明确 blocker，禁止借机
     改 boundary-burst objective、K/G、official detector 或新增模型路线。
281. 独立 MAX 已对 `aa3352e` 给出 `HOLD_FIX_REQUIRED`。只允许修复两项：提交时冻结并由
     P0/frontend/gate 全链复核 AdaTAD pretrain path/SHA；在 real legal hard-swap alignment
     通过前由 production runtime binder 明确拒绝 G1/G2。不得借此改变 burst objective、
     K/G、official detector、selector/decoder 或另建 worktree。
282. `1178989` 已是旧 V8 的终局机制 HOLD（`FAILED/2:0`）；当前没有 boundary-burst 主实验
     running/queued。不得把 `139+23` 静态测试、独立审计或 CUDA gate 写成 R0 headroom、P0
     winner、terminal mAP 或论文主结果。
283. `f629ad79461941f405bc2028f087034abd17a840` 替代 `aa3352e` 为唯一 exact candidate。
     本次只修 submit-frozen pretrain identity 与 G1/G2 production lock；禁止再次重做这两项，
     也禁止把它解释成 scorer、burst objective、K/G、DP 或 official detector 改版。
284. 远端受影响面 `63+23` 通过只允许进入独立 MAX 复审。更宽历史集合的六个
     transition-only protocol mismatch 必须披露，但不得借此修改封存旧路线；只有 reviewer
     证明它们污染 boundary-burst 生产路径时才升级为当前 blocker。
285. 证据测试不得手工补入 production builder 实际不会写出的字段。训练审计、checkpoint
     metadata、terminal validator 与 aggregate 的成功夹具必须至少有一条链直接调用真实
     `build_training_audit()`；否则静态测试通过不能作为部署证据。
286. `7b9ad0b` 不是部署 GO。真实旧 sidecar 证明其 selected-axis audit producer 缺少
     `formal_protocol/training_profile`，会使合法 official-60 在终端评测前被 fail-closed。
     该问题已由 `86f7663` 最小修复，禁止再次用手工 JSON 夹具掩盖或重复修复。
287. `86f7663a94d628eace316d17e31db7043f731f75` 是当前唯一 exact candidate。远端
     `64+29` 只允许进入全新独立 MAX；在其明确给出 `GO_TO_REAL_CUDA_GATE` 前，不得运行
     CUDA/R0/P0/official-60，也不得改 objective、K/G、DP、AdaTAD 或新增路线。
288. 独立 MAX 已对 `86f7663` 给出 `HOLD_FIX_REQUIRED`。禁止把该提交继续描述成
     deployment GO，禁止提前提交 CUDA/R0/P0/official-60；只允许关闭审计列出的 R0 完整
     Oracle/bootstrap/identity、同可行域 simple-delta、crop-valid、原子日志和 no-mock 集成缺口。
289. simple `abs(delta p_action)` 若未经过与 learned policy 相同的 global exact-K/max-hole
     DP，就只能是 scorer 诊断，不能作为 selector 对照，也不能据此执行或跳过停止条件。
290. R0 不得仅凭单点 mAP 冻结 K/G/radius/quota。必须包含 unrestricted GT Oracle，使用
     official evaluator 做逐视频 bootstrap，并按预注册置信区间规则冻结唯一最弱可行几何。
291. R0 结果消费者必须独立重开 prediction/checkpoint/config/annotation/class-map/evaluation
     config、train-only subset 与 blocked-video hashes 并重算 official mAP；重新封签错误 JSON
     不能成为有效证据。
292. 当前不需要开放式 Pro 发散。关闭上述 blocker 后只运行一次新 exact-commit 有界逐行
     复审；R0 后审统计，R4 前审 legal hard-swap alignment，禁止借 HOLD 新增模型路线。
293. `4ec3e078a3aad834ffe504d74d414bf7e2b6fad3` 取代 `86f7663` 成为唯一候选，但只处于
     本地合同测试通过阶段。Linux/PyTorch 回归、真实 CUDA gate 和全新独立 MAX 未通过前，
     禁止提交 R0/P0/official-60，也不得登记 V9 或声称修复已获部署 GO。
294. `frontend_*_block_list` 以“哪个 consumer 使用”命名，而不是以文件内视频所属 split 命名：
     `train_block_list` 内是 holdout videos，`holdout_block_list` 内是 train videos。R0 holdout
     evaluator 的 blocked-video 文件必须来自 sealed `holdout_block_list`，其值精确等于
     manifest `train_videos`，剩余 evaluator target IDs 精确等于 `holdout_videos`。禁止再按文件名
     直觉把它改回 `train_block_list`。
295. unrestricted boundary-burst Oracle 的含义是取消首尾覆盖骨架与 max-hole/physical-cap 约束，
     仍保留 exact-K 和同一边界微簇目标。它可以因确定性并列规则自然选到首帧，但不得强制首尾帧。
296. 历史键 `pure_delta_topk_diagnostic` 现在只是兼容别名；正式对照是
     `pure_delta_same_feasible_dp`，必须与 learned policy 共用 global exact-K/max-hole DP。
297. Slurm journal 中任何 `PENDING`、部分文件或缺失 completion seal 都必须人工与 Slurm 对账，
     禁止自动重提；只有 commit/cluster/roles/jobs.tsv SHA 全匹配的完整 seal 才能幂等退出。
298. R0 预注册 headroom 是 `0.20` 个百分点，即内部 mAP fraction `0.002`，不是 `0.20` fraction。
     必须由逐视频 paired bootstrap 的 CI 下界严格超过该值，并按 R2Q3→R4Q5 冻结首个可行族。
299. `4ec3e07` 的远端 `109+23` tests、compile/bash/HEAD/clean 只允许进入全新独立 MAX 复审；
     不等于 CUDA、R0 或长训许可。只有明确的 `GO_TO_REAL_CUDA_GATE` 才能解锁下一门禁。
300. R0 的 `selected_weakest_projected_family` 必须成为后续 P0、真实模型 gate 与 R3 主锚点
     的唯一必过 learned family。Gaussian 与未被 R0 选中的 burst family 可以保留为诊断，
     但其失败不得阻断 R0 选中 family 与 matched U 的主结果；也不得无视 R0 决策而默认
     提交 U/Gaussian/R2Q3/R4Q5 全部长训。该传播合同在独立 MAX 裁决与 focused test 关闭前，
     继续阻断正式 DAG。
301. 独立 MAX `019f86e9-8aa0-75e1-8373-686265ac8b61` 对 `4ec3e07` 的裁决是
     `HOLD_FIX_REQUIRED`：唯一 P1 blocker 是 launcher 把 `train_block_list` 转成 holdout evaluator
     blocked JSON；错误测试又把该反向行为写成期望。它会被 finalizer fail-closed，不会产出假 mAP，
     但会浪费 R0 GPU。最小修复只能改用 `holdout_block_list` 并增加真实 split 语义测试。
302. `f90595d8620e42e8e3d74722f2ab48126c6b65f2` 是关闭规则 301 的唯一候选；远端
     `168 passed, 2 skipped`、强制 C3 `23 passed` 与 no-submit precheck 仍不是 CUDA/R0 许可。
     只有全新独立 MAX 明确 GO 后才可部署。
303. 新 MAX `019f8701-edaa-7e83-a572-49024b524098` 对 `f90595d` 的分阶段许可是：R0
     获准，CUDA/P0/official-60 不获准。不得把总裁决 `HOLD_FIX_REQUIRED` 误解为禁止运行 R0，
     也不得借 R0 获准偷跑 P0 或四臂长训。
304. P0 winner 必须从已哈希 `records_jsonl` 重跑生产 `analyze_jsonl` 并与 summary 逐字段一致；
     只核验 records/summary 各自 SHA 后信任 summary 不足以解锁后续实验。
305. `p0_real_gate.json` 必须被 P0 decision 以 path/SHA/commit/schema/ok 绑定，并由 full-model
     gate 与 official-60 消费端继续复核；生成但不进入证据链等同于未通过。
306. 四臂 aggregate 必须证明 annotation、class map、evaluation target/subset、AdaTAD pretrain
     与 frontend split annotation identity 跨臂完全相同；只分别校验每臂文件存在和哈希不足以称 matched。
307. official ASFormer 的固定 normalized-LF hash 必须由 production P0/full-model gate 核验；
     full-model CUDA batch 必须保留 pipeline 的 `gt_boundary_validity`。最终 suite 写入需原子化，
     但这些 P2 完整性修复不得改变模型、损失或 R0 结果。
308. R0 privileged boundary-burst 的配额语义是“每个有效端点中心必选，半径 R 内联合选择
     至少 Q 帧，适用时左右各至少一帧”。禁止再次把它实现为固定
     `center,-1,+1,-2,+2...` 的 nearest-Q union；邻近端点必须能够共享观测并联合满足 K/G。
309. Job `1179392` 只证明旧 fixed-nearest-Q Oracle 在真实密集动作窗口上会产生假不可行。
     它在 detector replay 前失败，没有 mAP，禁止用它否定 G2、R4Q5 或 boundary-burst，
     也禁止把它写成 R0 headroom 结果。
310. `22555a4e830ce24f9bb516897b1bb7f44b70c188` 是 corrected R0 exact-quota 候选，
     不是 V9 或经验支持。真实失败样本 replay、静态测试和零 MIP gap 只能授权精确复审；
     独立 MAX 明确 GO 前不得重排 R0，R0 通过前不得运行 P0/official-60。
311. 不得通过把 G2 临时放宽为 G10/G15 来掩盖 exact-quota solver 错误。只有 corrected R0
     先在冻结 G2 下形成有效 mAP/headroom 证据后，G10/G15 才能作为预注册覆盖-聚集消融；
     它们不是当前修复路径。
312. R0 只负责证明可行空间和 Oracle mAP 上限。P0 才训练 coarse/transition/burst，R3 才
     比较 terminal-EMA U/G0。不得把单样本 Oracle replay、P0 quality 或 detector loss
     代理当作最终 mAP 结论。
313. 性能证据必须位于执行关键路径。用户要求用 mAP 回答核心问题时，先盘点/提交/监控/解析
     实验，再做非阻塞审计和文档；禁止让开放式讨论、重复可见性认证或无变化记录占用 GPU 前置时间。
314. 独立 MAX 只在 R0 mAP 实现、R2/R3 最终模型与 U/G0、R4 hard-swap 与 G1/G2 这三类
     真正关键版本完成后各运行一次。小型字段、哈希、journal、aggregate、文档或启动器修复不得
     自动升级为完整审计门禁；多个有界 blocker 必须一次并行修完，再集中复审一次。
315. R 系列代码准备可并行，证据依赖只约束实际运行。不得以 R0/P0 尚未解锁为由拖延 R1/R4/R5
     的无冲突实现；下游通过 `afterok` 与 fail-closed consumer 保持协议正确。
316. Wiki 只在新决策、新提交、新 Job、真实失败、新 mAP、新成本或 claim 裁决时更新。连续两轮
     没有新代码、作业、mAP 或 KILL 信息时，必须停止重复流程并回到最短性能证据路径。
317. `run-experiment`、`monitor-experiment`、`analyze-results`、`experiment-queue` 可按需服务
     性能关键路径；审计型 skill、开放式 Pro 和重复独立 MAX 不得自动成为每个提交的前置工作流。
318. 显式四小时等截止时间采用时间盒：15 分钟内完成盘点，30 分钟内提交全部依赖安全实验或报告
     唯一物理阻塞；截止时先给可核验原始 mAP，并严格区分 terminal、diagnostic 与 running。
319. 当前四小时 R0-R5 交付只有在生产入口、正式配置、真实 backend、focused/contract tests、唯一
     exact commit 全部存在时才算代码完成。mock、sentinel、占位 backend、仅 `PRECHECK_ONLY`、TODO
     或“后续接入”不得登记为 implemented。
320. R0-R5 的完整部署必须包含 Slurm 已接受的有效 Job ID、dependency、exact commit、run root、
     manifest/hash 与终端产物路径；只生成 sbatch/config 或停在 gate 前不算 deployed。
321. R4 必须以真实合法 hard-swap selected RGB 和冻结 official detector 形成 signed utility/alignment；
     R5 必须包含三种子、K384/K256、真实第二 backend 与完整端到端成本。代理指标或伪 backend 不能替代。
322. 并行智能体只返回文字说明不算交付；必须由父任务检查、集成、测试并形成唯一关键提交，再做一次
     独立 MAX。通过后立即部署，不得把 MAX 后的小型工程修复继续扩成循环审核。
323. R0 的逐视频 bootstrap 只允许生产者调用官方 evaluator 一次。P0、family router、gate 和
     aggregate 消费者必须重开并哈希核验 prediction/evaluation/bootstrap，重算四族原始 mAP、
     bootstrap 差值和置信区间，但禁止再次执行全部 1000 次 evaluator resample。重复计算不增加
     统计证据，却会把模型实验拖慢数小时；它属于必须删除的工程性审计膨胀。
325. N16R4 `AssocMaxSubmitJobLimit` 是调度上限，不是模型失败。若逐 cell 提交超过上限，允许在
     一个 GPU allocation 内顺序执行若干原始、哈希绑定且 fail-closed 的 sbatch；必须保留成员表、
     原始 Job 取消记录、依赖与部署收据。禁止减少 cell、种子、预算、后端、成本项或把 generated
     config 冒充已提交实验。
326. 精确 `e49ef696` 的唯一正式 DAG 已提交。不得再次提交相同 R0--R5 配置；只能监控
     `1179795--1179865` 中登记的有效依赖链。六个 `1179828--1179833` 是未运行后取消的逐 cell
     重复项，不得计入结果、失败率或实验数量。
327. 真实 Job `1179517` 已证明串行执行 1000 次 x 4 族 official-evaluator bootstrap 可超过四小时，
     并阻塞全部 `afterok` 模型实验。R0 生产者必须在先证明相同 RNG 抽样序列、输出顺序与数值逐项
     等价后使用 Slurm 已分配 CPU 并行执行；下游消费者仍只能哈希核验并消费封存样本，不得重跑
     bootstrap。该优化是统计执行加速，不是模型、可行域、评价协议或实验配置改版。
328. “冻结 detector 回放”只表示回放时不更新参数，不表示 detector 没见过评测视频。当前 R0 的
     40-video frontend holdout 是在复用 `transition_beta0/epoch_131.pth` 之后创建，而该 checkpoint
     按完整 THUMOS `training` subset 训练，未消费 R0 block list。因此 R0 的 93--94 mAP 必须标为
     detector-seen training-internal diagnostic，禁止与正式 validation/test 的 64--65 mAP 比较，
     禁止写入论文主表或称为泛化上限。
329. 调用官方 mAP 实现不等于采用官方 benchmark 协议。R0 的 40-video training-internal subset 与
     1000 次自定义 paired bootstrap 只能做同 checkpoint、同视频、同预测链的内部相对诊断；论文
     主结果必须在完整官方 validation/test split 上按相同 terminal checkpoint/config/seed 单次评测，
     不得让自定义 bootstrap 代替或长期阻塞正式 mAP。需要方差时优先报告多种子 mean/std。
330. 若 R0 继续承担 clean headroom 或 family-selection 证据，detector 训练必须显式排除同一 40-video
     holdout，并封存 split-before-training 证据；否则只能把 R0 降级为 privileged mechanism diagnostic。
     当前 DAG 的下游 learned-policy official mAP 仍可运行，但不得以受污染 R0 的绝对分数支持论文 claim。
331. DUCA 论文 mAP 行必须同时满足：完整 THUMOS validation、OpenTAD `mAP`、tIoU 0.3--0.7、
     无 blocked validation videos、terminal epoch-59 EMA。P0 质量、一步梯度门禁、hard-swap alignment、
     bootstrap 与 cost profile 都不是准确率结果。
332. R0 bootstrap 只能作为可选的不确定性分析，不得阻塞标准 official point-mAP。尚未完成并通过
     terminal artifact 核验的 R5 cell 只能称 protocol-eligible，禁止预先认证为官方可比结果。
333. 正式对比臂不得依赖 R0、bootstrap 或无关诊断 Job。单个 learned arm 内部的 P0 -> real gate ->
     official60 -> evaluation 是权重消费的真实顺序，不属于跨实验相互等待，也不得伪装成可并行步骤。
334. Learned arms 固定使用 training-only P0 terminal epoch 19。它们相对 exact-uniform 多出的训练成本
     必须披露；不能据此否定 mAP 可比性，也不能谎称总训练成本完全匹配。
335. 真实 official-ASFormer 包装器的来源证据位于 `raw_actionness_source.probe.official_source`；
     `probe_module` 只允许作为旧测试夹具兼容入口。不得让夹具结构再次冒充生产结构。
336. `gt_boundary_validity` 的数据合同是逐样本布尔值完全相等，不是 Python 容器对象身份相同；DDP/
     batch 搬运允许重建 list/tuple。比较前必须显式移到同一设备，禁止用跨 CPU/CUDA 的 `torch.equal`。
337. Jobs `1180075`、`1180097`、`1180106` 均在 official-60 训练前因门禁合同错误退出，没有 mAP，
     不能登记为模型负结果。因相同已知门禁错误主动取消的伴随学习臂也不得计入失败率或方法比较。
338. 当前唯一 official-mAP 主队列是 exact commit `8d85929ea04dc40f1eb0c3cc806061ce3b071d3f`
     的 Jobs `1180111--1180114`。四者必须保持 `Dependency=(null)`；PENDING(Priority) 是调度等待，
     不是实验相互等待，也不是模型失败。
339. 只有 `map_protocol_audit.json` 中标记为 R3 official60 terminal evaluation，且运行时实际生成完整
     validation、tIoU 0.3--0.7、epoch-59 EMA 结果的行，才可称官方可比。R0/P0/gate/R4/cost 一律不得
     借用“mAP”字样进入论文主表。
340. `e49ef696` 是当前 DUCA 模型身份；`8d85929` 是同一模型的 official-mAP/evidence gate
     执行修复。不得把后者误写成 selector、decoder、detector、loss 或训练日程的新模型版本。
341. 当前 R2Q3/R4Q5 是 soft bilateral/quota burst objective；硬 DP 只保证 exact-K 和 max-hole。
     在 mandatory left/right mask 未进入 decoder 前，禁止声称硬双侧覆盖或硬局部配额。
342. 当前完整栈先密集解码并传输 160x160 粗输入，再在 heavy VideoMAE 前选帧。因此只能声称
     post-decode heavy-backbone processed-frame reduction，不能声称 K-only decode/H2D 或完整视频 I/O 降本。
343. 未来 R5 paper aggregate 必须从 raw predictions 独立重跑 OpenTAD evaluator，并要求 candidate/dense
     cost 的硬件、会话、软件、输入和 profiler 身份匹配；只验证 terminal JSON 自哈希或直接相除 p50 不够。
344. 当前 K384 seed-0 四臂 `1180111--1180114` 出 terminal EMA mAP 前，不恢复 24-cell 全矩阵。
     多种子、K256、TemporalMaxer 和完整成本是有条件扩展，不是当前门禁自动解锁项。
345. R0 93--94 是 detector-seen training-internal 机制诊断，不得继续担任唯一 family 选择权威、论文
     absolute mAP 或 official baseline。当前四臂直接比较所有 family，不再等待 R0/bootstrap。
346. 当前 official mAP 与中心误差、左右配额、soft-to-hard mismatch、梯度归属诊断完成后，最多选择
     一个结构修正。禁止同时实现 hard bilateral、hard-swap、ASFormer adaptation、true-time adapter 和
     adaptive burst/context split，制造无法归因的新版本。
347. 用户已显式要求全面部署 R0--R5，因此旧规则 338/344 中“唯一队列是 8d 四臂”和“暂不恢复
     24-cell 矩阵”已被后续决策取代。当前唯一执行身份是
     `codex/duca-boundary-burst-20260722@cd68d89dcc0854baa3c0107607086e801509b552`，正式根为
     `duca_boundary_cd68d89_parallel_20260722_205506`；不得再把旧队列描述成当前主实验。
348. `1180336--1180340` 是五个无相互依赖的正式 R0--R5 model bundles，`1180341` 只依赖 R5 做
     aggregate。R5 的 24 个 official-mAP cell 是 ActionFormer/TemporalMaxer x uniform/learned x
     K384/K256 x seeds 3407/5801/8123；四个正式 cost profile 仅限 ActionFormer seed 3407 的
     uniform/learned x K384/K256，并必须同会话配对 dense AdaTAD。
349. `2645e68` 的 Jobs `1180326--1180331` 在首个 soft-bilateral P0 batch 因 PyTorch 2.0 CUDA
     二维行重置错误退出，无 optimizer update、无 mAP；等价 `index_fill_` 修复已进入 `cd68d89`。
     旧 Jobs 只能作为启动故障记录，禁止作为模型负证据或与 cd68 结果合并。
350. R0 的四族 train-internal replay、P0/gate、R4 alignment 和 cost profile 均不是论文准确率行。
     只有完整 validation、terminal epoch-59 EMA、OpenTAD tIoU 0.3--0.7 的 R2/R3/R4/R5 行可进入
     mAP 表；在 terminal artifact 出现前状态只能是 `experiment_running`。
351. 五点预算曲线固定为 K=384/320/256/192/128；对应 max-hole G=2/2/3/4/6。不得为了看起来覆盖更强而把 K192/K128 强行设为 G2，也不得把放宽 G 隐藏在主表之外。
352. 旧 `cd68d89` 的 K384/K256 24 cells 不得重跑。新 `a00498e` 只增排 K320/K192/K128 36 cells；两者只在 terminal official aggregate 后合并。
353. `f4b2568` 的 Linux 门禁失败源于 `duca_selected_axis_training.py` 仍硬编码两预算/24 cells。它发生在正式增量训练前，无 optimizer update、无 mAP；修复后的唯一增量身份是 `a00498e15d69294f78d0abeadfb47bc456db0b0e`。
354. Jobs `1180356/1180357/1180358` 分别是新三档完整 TAD、增量聚合、五点 mAP 与选帧分布。`1180356 PENDING(AssocGrpGRES)` 是资源等待，不是提交失败；不得重复提交相同 36 cells。
355. 选帧分布只能作为机理证据。论文性能结论必须来自五档完整 validation、terminal epoch-59 EMA、OpenTAD tIoU 0.3--0.7 的三种子 mAP；不得用边界召回或端点距离代替 mAP。
356. “覆盖全部粗候选时间点”不等于“逐原始视频帧运行”。当前 DUCA 的 768 点候选网格约每 4 个源帧取一点，并在批张量中运行低分辨率 probe；不得误写成 native-FPS 逐帧推理。
357. 当前完整 768 点候选仍在 selector 前完成解码、160x160 变换和 H2D，因此现阶段只能主张重 backbone processed-frame 节省。未经实测，不得声称总解码、输入或端到端成本随 K 等比例下降。
358. 后续降低粗扫描成本只能做候选密度 matched 消融：固定重预算 K、selector、detector 和训练协议，比较密集粗扫、低频粗扫、低频加转变峰值局部补扫，并同时报告官方 mAP、短动作/边界诊断与完整成本。不得用 Uni-AdaFocus 的分类采样合同替代 TAD 的边界和物理覆盖合同。
359. 规则 347--354 的 `cd68d89/a00498e` 作业身份已被真实调度复核推翻：R4/R5 重复完整 bootstrap，且 `srun` step 继承全部作业 GPU，造成伪并行。旧 Jobs `1180336--1180341,1180356--1180358` 已取消并保留为执行故障历史；禁止再称其为当前正式队列或继续等待其结果。
360. 当前唯一 R0--R5 身份是 `9f97f2c7f081b10fbf1f63d0602a621c6b43a780` 与 Jobs `1180490--1180496`。R0/P0/U/G0/alignment 只能由 `1180493` 产生一次；R4/R5 只能消费其哈希封存 receipt，禁止再次在各自 bundle 中重建。
361. 多 GPU bundle 的每个并行子臂必须同时具有 `srun --exact`、`--gpus=1` 和 `--gpus-per-task=1`，并以真实 `TresPerStep=gres/gpu:1` 验收。仅在 sbatch 中申请多卡或在 shell 中加 `&` 不能证明并行。
362. 五预算不再拆成旧两档加新三档，而由 `9f97f2c` 的同一 R5 矩阵覆盖 K384/320/256/192/128、两后端、两策略和三种子。当前只有部署与并发证据；不得提前声称 official mAP 或成本收益。
363. 稀疏粗扫描的主重建输入是插值到完整 768 点网格的多维 temporal hidden；插值值本身就是 selector 证据，不再增加 observed-anchor mask 或最近 anchor 距离。禁止反复恢复这两个旁路输入，也禁止退化为只插值 `p_action`。
364. MS-TCN2、ASFormer、FACT、Video-Mamba-ASFormer 的独立 P0 粗分类结果只能做架构诊断。完整 TAD 比较前必须统一 temporal-hidden 输出语义；ASFormer temporal encoder hidden 与其他模型 spatial-stem hidden 的混合对照不公平，不能据此选择论文主后端。
365. `a00498e` selected-axis/TTDI Pro 审查已归档并对 `9f97f2c` 复核。由于两提交间只改 bundle
     启动脚本，selected-rank 时间扭曲、mandatory union 无接纳前 completeability 和 local-RGB-slope
     surrogate 仍是当前模型风险；禁止误报为 `9f97f2c` 已修复。
366. TTDI 是 `designed_pending_terminal_map` 候选，不是当前最终模型。只有 learned hard selection
     质量优于 uniform、但 official 高 tIoU mAP 不优时才解锁；否则先修 coarse/scorer/allocator。
367. 零初始化 true-time feature residual 与 physical-coordinate head/GT assignment 是两个结构变量。
     首个决定性实验只能加入前者；禁止一次性合并后把收益统一归因于“TTDI”。
368. Pro 回复给出的 2000/6000 updates、loss 权重和 55% alignment 阈值均是待验证提案，不得覆盖
     当前已部署协议或写成理论必然。当前 `1180490--1180496` 不因后验审查取消、篡改或重复提交。
369. 当前 `9f97f2c` 五预算来自同一模型提交，旧跨提交 merge 风险不直接作用于本轮 mAP；但正式
     cost parser 仍只接受 K384/K256，K320/K192/K128 成本不得在修复前声称完成。
370. 稀疏粗扫不再是 `designed`：唯一实现身份为
     `codex/duca-sparse-probe-interpolation-20260723@dd3c97cf5ee628c2b0b6f26ce976618e36b7cd45`，
     Gate `1180556` 已通过，四档完整 TAD Suite `1180557` 正在运行。禁止重复实现同一插值路径或重复提交同一套件。
371. d=1/2/3/4 必须只改变粗 probe 的真实计算 anchor 与 hidden-linear 重建；R2Q3、K384/G2、
     VideoMAE、official-derived AdaTAD/ActionFormer、seed 和 official-60 协议不得同时改变。插值
     action logits、temporal encoder hidden 和 policy hidden 均直接作为完整证据，不增加 mask/距离旁路。
372. Gate 的非零梯度与 MACs 单调下降只证明实现正确和理论成本趋势；四档优劣只能由完整
     validation、terminal epoch-59 EMA、OpenTAD tIoU 0.3--0.7 mAP 与实测总成本裁决。
373. combined DUCA 回归测试中 `test_boundary_burst_soft_and_hard_arms_share_local_utility_only`
     的精确相等断言在未修改基线 `4f81299` 上同样失败，根因是 training-mode 随机性；不得误报为
     sparse interpolation 回归，也不得借此修改无关 boundary-burst 代码。
374. 首次 suite 预检查只因登录环境旧 Python 缺少 `pathlib` 而停在 suite sbatch 提交前；Gate
     `1180556` 已完成后只提交了唯一 Suite `1180557`，回执随后以 shell 封存。该事件不是模型或训练失败，禁止重提重复 suite。
375. “免训练”必须拆成目标数据集免训练与严格零优化。冻结外部预训练 encoder 可以称
     target-train-free；测试时更新投影器、selector 或提示词不属于严格零优化；从未训练过的随机
     网络不得被包装成语义粗分类器。
376. 免训练模式禁止依据 THUMOS validation/test mAP 选择 prompt、融合权重、阈值或 stride；禁止
     密集运行 X3D/SlowFast/大视频 VLM 后只报告重 backbone 节省。外部 encoder 的 decode、预处理、
     H2D、FLOPs、latency 和 energy 必须计入总成本。
377. 首个 PhysTime 后续实验只能是相同 selected positions、相同 selected-axis detector 下的零初始化
     T1 true-time residual 与常量/打乱时间码对照；禁止同时修改 physical head、GT assignment、decode
     和 endpoint loss 后将差值归因于时间编码。
378. 五类主图只消费真实可比证据。粗分类 P0 无 detector 时禁止生成 TAD Pareto/预算 mAP；运行中
     实验不得预填曲线；内部 93--94 holdout、action AP/AUC 和 boundary proxy 均不得替代 terminal
     full-validation OpenTAD mAP。
## 2026-07-22 model-algorithm-first guardrail

- 不得把通用框架、复杂 schema、重复证书、日志美化或非必要验证器当成研究交付物；它们只有在影响模型行为、泄漏、指标真实性或可运行性时才可阻塞实验。
- 模型假设明确后必须优先形成最小端到端训练和 matched 性能对照。连续两轮只提高工程完整度而没有模型、实验、指标或明确 KILL 信息时，立即停止该工程支线。
- 当前空间路线不得让手工/GT 特权裁剪的枚举协议决定是否进入可学习模型；其结果最多是诊断。主实验应直接检验由真实 AdaTAD 检测损失优化的连续可变区域策略。
# 2026-07-23 论文叙事与绘图防回退规则

- 禁止继续沿用 `window-online`、zero-shot 主方法、teacher-utility Top-K 和 ledger 贡献叙事；
  当前任务始终是 offline TAD，当前主线是 transition-calibrated boundary-burst acquisition。
- 禁止把 R0--R5、G0--G2、四个 coarse backend 或多个 stride 平铺成“方法创新”；它们只能
  分别回答可达性、可学习性、有效性、归因性、效率与泛化问题。
- 禁止把内部 frozen-detector holdout 的 93--94 mAP、coarse AP/AUC 或边界 proxy 当作官方
  TAD 性能；主张必须由 terminal official mAP 与完整成本直接支持。
- 禁止用更多变体掩盖 matched uniform 未被稳定超过。主预算、关键组件和停止规则应在读取
  official test 结果前冻结。
- 禁止把 correlation plot 当作因果证据；机制图必须配套只改变一个因素的消融。
- 禁止提前把 TTDI 或 detector feedback 写成最终贡献；它们只能按预注册触发条件与 mAP
  结果升级。

# 2026-07-23 target-train-free Fast-only 防回退规则

- target-train-free 只指冻结 pre-backbone 选择前端，不能把仍在 THUMOS training 上训练的
  AdaTAD detector 称为整套 training-free；目标域 mAP 不得用于选择固定融合权重或阈值。
- 当前 SlowFast 冻结先验必须是 **Fast pathway only**：Slow pathway 与 lateral fusion 不执行。
  它是高成本强视频先验诊断，不是轻量主方法；禁止再次改回 Slow 分支或完整 SlowFast 融合。
- Fast CUDA preflight 已证明 `[1,16,256]` hidden、`slow_path_executed=false`、
  `lateral_fusion_executed=false`。`1180639/1180644/1180652` 的下载/旧合同退出均无 optimizer
  update，不得登记为性能负结果。
- 当前有效冻结先验作业为 Fast-only `1180653` 与 MobileNet 三臂 `1180654`；不得重复提交，
  除非对应 Job 真实失败且失败原因已经修复。

# 2026-07-23 R2/R3 P0 恢复防重复规则

- `1180491/1180492` 不是训练崩溃：四个 P0 已各完成 6000 次有效更新，失败仅来自后置
  BatchNorm buffer 初始化合同。禁止重跑这四个 P0。
- 只允许使用已哈希的四个 epoch-19 checkpoint，经 `487a178` 真实整模门禁后继续 official-60；
  当前唯一恢复 Jobs 为 `1180671--1180674`，不得因 pending 或无早期 mAP 重复提交。
- checkpoint 缺失可训练参数必须继续 fail-closed；只有明确登记在 receipt 中的非参数 buffer
  才能保留新阶段初值，禁止用 `strict=False` 泛化绕过状态错误。
- P0/实验标签与生产运行时策略 ID 是两个字段。禁止再把
  `boundary_burst_r2q3_{soft,hard,adapted}_*` 直接传给 selected-axis runtime binder；当前唯一
  映射实现是 `ca40c9c`，soft 标签映射 `boundary_burst_r2q3_soft_g0`，hard 标签映射
  `boundary_burst_r2q3_g0`。
- fail-closed 独立入口要求 `ARM_ROOT` 尚不存在；临时提交器只能创建其父级和日志目录，禁止预建
  arm 结果目录。`1180682--1180684` 因此在 optimizer 前退出，不得反复诊断为模型失败。
- 当前唯一有效恢复集合是正在训练的 R4Q5 `1180674` 与排队的 R2Q3 `1180685--1180687`。
  `1180671--1180673`、`1180682--1180684` 只保留为基础设施审计历史，不得重复提交或进入 mAP 表。

# 2026-07-23 稀疏 P0 与 MobileNet 恢复防重复规则

- 稀疏 Suite `1180557` 四个 P0 已完整完成，失败仅为同一 BatchNorm buffer 门禁合同。禁止重跑
  d=1/2/3/4 P0；只允许 `cee4ccd` 从四个已哈希 epoch-19 checkpoint 恢复，当前唯一 Job
  `1180696`。
- 冻结 MobileNet 语义模式必须直接保留预训练多类 logits；禁止在其后接随机、未训练的
  `LazyLinear` 再称为 ImageNet 语义先验。`1180654` 是 optimizer 前模型构建失败，不是性能证据；
  当前唯一修复 Job 为 `1180697@e30db0f`。
- `1180696/1180697` 的 `PENDING(AssocGrpGRES)` 仅是账户 GPU 配额等待，禁止因此重复提交。

# 2026-07-23 T1 P0 恢复防重复规则

- `1180637/1180638` 中的 R2Q3、true-time residual 与 reversed-time residual 已完成 P0；三者
  后置门禁失败仅为已知 BatchNorm buffer 合同。禁止重跑其 20-epoch P0。
- 唯一修复身份是 `26ce86d7810e8f7c0568dc045bb1db7240c66de2`，唯一恢复 Jobs 是
  `1180717/1180718/1180719`。`1180637` 仍运行仅因 exact-uniform 对照健康训练，禁止取消该对照
  或把顶层 RUNNING 误报为所有子臂健康。
- 三个 P0 哈希相同是当前课程合同的预期：T1 residual 只在 official-60 阶段生效。禁止据此怀疑
  checkpoint 覆盖，也禁止人为制造不同 P0 以破坏 matched comparison。
- `26ce86d` 的 `1180718/1180719` 在 official-60 optimizer 前因 T1 实验标签未登记到
  selected-axis runtime config 映射而失败；不得把它们计为 T1 性能结果或再次提交。`1180717`
  R2Q3 仍健康，继续使用原作业。
- T1 actual/reversed 的唯一有效恢复身份是
  `919aa555d1aa36191ee318477409dfbfdfb0e807`，唯一 Jobs 是 `1180731/1180732`。它们必须复用
  已封存 epoch-19 P0；禁止重跑 P0、重复 R2Q3，或用 `strict=False` 绕过绑定。

# 2026-07-23 R0 统计停止防重复规则

- `1180493` 已完整完成 1000 次 bootstrap；exit `2:0` 是预注册
  `KILL_PROJECTED_FEASIBLE_SET`，不是工程失败。禁止重跑 bootstrap、改 exit code 或把
  `1180494/1180495` 的 `DependencyNeverSatisfied` 当成需要修复的 Slurm 问题。
- R2Q3 虽有 `+0.6034 pp` 点估计，但 CI 下界 `-0.5300 pp` 未超过 `+0.20 pp` 门槛；R4Q5 与
  unrestricted Oracle 同样未通过。禁止按点估计事后强选 family 或删除置信区间。
- R0 的 90+ mAP 是 training-internal holdout frozen-detector replay，不是论文完整 validation
  terminal mAP。只能把它写作几何可达性负证据；正式训练臂的 terminal mAP 另行收割。

# 2026-07-23 Fast-only 终点防重复规则

- Job `1180653` 已用 commit `4c5604b4a0abde9e59f625d519934e855bfe1519` 完成唯一正式
  Fast-only K384/R2Q3 终点评估，Avg-mAP 为 `63.5297%`。禁止因低于目标就重复相同 seed、
  checkpoint、选择策略和训练协议。
- 在 matched exact-uniform 终点结果产生前，不得用历史 64--65 数值计算精确差值；当前只允许写
  “未达到约 65% 目标，尚未证明优于 matched uniform”。
- 后续诊断应定位 frozen Fast motion evidence、R2Q3 硬位置与 GT 边界的偏移，以及 selected-axis
  对非均匀时间间隔的扭曲；不得直接把 63.53 归咎于单一粗分类器，也不得把它提升为 train-free
  主方法支持证据。

# 2026-07-26 rate-curriculum Stage-2 recovery rule

- Job `1182391` completed the Stage-1 uniform warmup and its terminal
  `epoch_29.pth` is sealed by SHA-256
  `7233fa6944659f432f8deaf22448b4a25cf8794b1e912f59a4d5b3715d54b39e`.
  The Stage-2 failure was a zero-update legacy-P0 binder contract error, not
  a curriculum performance result. Do not rerun Stage 1 to repair it.
- Job `1190439` failed at zero runtime while its outer Slurm wrapper sourced
  the site profile under nounset mode. It ran no Python and produced no model
  evidence. The only allowed replacement is commit `b554f04` Job `1190528`,
  which fixes only that environment-source wrapper and reuses the exact
  checkpoint through the launcher hash and epoch-29 guards. Any later Stage-2
  retry must reuse the same sealed checkpoint unless a new Stage-1 experiment
  is explicitly authorized as a distinct method study.
- Stage-2 checkpoints at one-based epochs 5, 10, ..., 60 may receive only
  read-only, hash-recorded learning-curve diagnostics. They must never feed
  back into the running optimizer, alter the sampler, select the terminal
  checkpoint, or be reported as the formal final result. The first diagnostic
  attempt `1190606` failed before Python because it omitted the canonical
  environment source; it is not a result. The sole valid epoch-5 diagnostic is
  `1190626`, bound to the same `epoch_4.pth` SHA.

- The only completed e5 diagnostic is `1190643`, not `1190606`, `1190626`,
  `1190633`, or cancelled `1190637`. It is read-only and non-selecting. Its
  `60.521318%` official mAP and fixed 64-window quality export are valid
  learning-curve/mechanism evidence only, never a terminal comparator or a
  license to tune the family after observing the result.
- Job `1190528` completed 1,000 finite Stage-2 updates but failed closed on
  the first forward after its e10 validation, before AMP scaling. Treat it as
  an affected numerical failure with no terminal EMA result, not as a model
  failure or a reason to restart Stage 1. `epoch_9.pth` is the only admissible
  recovery source. First run exactly one read-only single-batch component
  diagnosis at the e10 schedule state; until it identifies the culprit, do not
  submit a recovery, change loss weights, alter the schedule, use
  `strict=False`, or rerun any healthy arm.

- User clarification on 2026-07-26: a bounded count of non-finite values is
  acceptable only when every affected batch is recorded and replayed from its
  untouched state. A pre-AMP non-finite `cost` is not an optimizer update:
  it must restore RNG, buffers, and custom replay state, then retry the same
  batch; it must never skip that batch or advance the optimizer, selector,
  scheduler, or EMA. Persistent non-finite loss still fails closed after the
  fixed bound.
- Job `1191745` failed before model construction because the diagnostic was
  invoked as a file and could not import the repository package. It is not
  numerical or offline TAD model evidence. The valid replacement is Job
  `1191754` at commit `65a4cfb31716f84c153af881a71fe05069637848`: it strictly
  restores the sealed Stage-2 epoch-9 model, optimizer, scheduler, EMA, and
  GradScaler in memory, executes batch 0 through a real AMP update and the
  1000-to-1001 selector transition, then probes batch 1 under eight controlled
  seeds. It writes no checkpoint, persists no optimizer/EMA state, and
  evaluates no mAP.
- Job `1191754` has eight of eight finite batch-1 outcomes. Across all trials,
  batch-0 gradients, the 49,914,588 post-update parameter values, and the
  56,069,713 post-update optimizer values remain finite; AMP scale remains
  8192, and batch 1 remains finite after selector and scheduler step 1001.
  This rejects persistent post-update contamination and an immediately
  deterministic schedule-boundary failure. The sealed epoch-9 checkpoint has
  no RNG state, so the exact original stochastic draw cannot be reconstructed;
  the remaining supported cause class is a one-shot pre-AMP forward transient
  in the stochastic/nondeterministic path, not an accepted persistent failure.
- Commit `9519760a26cd7fda08c3e648b1e7d7f459b3b6b` enables the bounded
  pre-AMP replay only for this Stage-2 curriculum (`max_nonfinite_loss_retries=8`).
  It atomically records `nonfinite_loss_*` counters to
  `stage2/update_audit.json`, restores state before every retry, and raises on
  exhaustion. Slurm code precheck `1191787` completed `0:0` with 15 focused
  tests passing; it has no training, checkpoint, or mAP output. A recovery may
  reuse only sealed `epoch_9.pth` and must report its final terminal epoch-59
  EMA OpenTAD official mAP without intermediate checkpoint selection.
- Commit `adc6fb13114584188da4ac17eeeab6d89d69d04f` and precheck Job
  `1191796` bind the only admissible recovery launcher to both sealed source
  hashes and an explicit recovery manifest. The rejected `--mem=62200M`
  scheduler submission created no job and is not model evidence; use the
  partition default single-GPU memory only. Do not duplicate the precheck or
  submit more than one Stage-2 continuation. Its sole endpoint is epoch-59
  EMA OpenTAD official mAP plus the update audit.
- Job `1191806` is the one authorized continuation and failed closed at
  epoch-10 batch 2 after exactly two finite updates. Its 8 bounded replays all
  remained non-finite, while the audit proves zero failed-batch state advance.
  Do not resubmit Stage-2, raise the replay bound, skip the batch, alter the
  schedule/losses, or call this a model-performance result. The only permitted
  follow-up is one read-only two-update prefix-state diagnosis from sealed
  epoch 9 that identifies the non-finite loss component and module boundary.

- **Resolved Stage-2 numerical cause (2026-07-26):** Do not spend another
  recovery run on the old `1191806` code path, call it a stochastic AMP event,
  or add `strict=False`/loss skipping. The fixed prefix identifies a
  deterministic FP16 ordering bug only in contribution distribution
  distillation: invalid logits were masked to finite `-65504` before division
  by `0.7`, becoming `-inf`, then zero targets formed `0 * -inf = NaN`.
  Commit `4c1f5384ae693c74a141619ded03196a72c594ed` scales before masking;
  read-only `1191854` verifies the historical batch is finite. A resumed
  Stage-2 run must use this commit, the same sealed inputs, strict bindings,
  same-batch replay audit, and terminal epoch-59 EMA official mAP only. Do
  not repeat the completed read-only diagnoses.

- **Current affected-arm continuation:** Precheck `1191874` completed and
  `1191880` is the only repaired Stage-2 continuation. Monitor that job and
  its `stage2/update_audit.json`; do not submit a second continuation while it
  is healthy or pending. Its only model endpoint is the terminal epoch-59 EMA
  OpenTAD official mAP.

- **Protocol-invalid repaired continuation:** `1191880` was deliberately
  cancelled after 700 finite updates, not because of a model or numerical
  failure. Its config inherited `intermediate_validation_selects_checkpoint=True`
  and wrote `best_validation_ema.json`; this violates the course prohibition
  on intermediate-mAP checkpoint selection even though the pointer did not
  change optimization, EMA, scheduler, selector state, or early stopping.
  Its epoch-15 EMA mAP `62.403751%` is audit-only and cannot be compared,
  selected, or reported as an offline TAD result. Do not resume that work
  directory or call it a healthy completed job. The only permissible successor
  must use the same sealed e9 source, explicitly set selection to false, and
  precheck the five-epoch read-only quality diagnostics before one fresh run.

- **Current valid continuation:** Precheck `1191956` completed `0:0` under
  `42dba3f90b37243e7965d18b6707e88e81bf7109`; it accepts only
  `learning_curve_only`, forbids `best_validation_ema.json`, and binds the
  original sealed e9 source. Job `1191957` is the one replacement. Do not
  submit another continuation while it is pending or healthy; its only
  performance endpoint is the epoch-59 EMA OpenTAD official mAP.

## 2026-07-27 joint-review snapshot guardrails

- Do not cite the 2026-07-26 joint review's `G0 not opened`, `source coverage
  zero`, `fixed HEAD 4c1f538`, or `full Stage-2 blocked` statements as current
  facts. PR #2 now passes the read-surface gate at `42dba3f9`; source
  adjudication is still open, and Job `1191957` is the sole valid continuation.
- Do not equate an open PR and immutable blob list with completed independent
  source review. A requested item becomes `CODE_FACT` only after the source
  has actually been read and adjudicated; otherwise retain `PARTNER_CLAIM` or
  `unresolved`.
- Do not copy plaintext proxy credentials into Wiki, reports, issue text or
  PR comments. Redacting current README, rotating the exposed credential and
  treating historical blobs are separate actions; none can be inferred from
  another.
- Do not auto-freeze reviewer-proposed D1 thresholds, Path A/B publication
  bands, arm counts, seeds, budgets or route-arbitration choices. Record them
  as `designed_reviewer_proposal` until explicitly reconciled with the
  canonical contract before result access.
- Do not mark every report-defined D0 subcheck passed after the fact. The
  accepted current fact is narrower: the deterministic FP16 mask-order defect
  was isolated and repaired, the historical batch is finite, and long-run
  continuation updates are healthy. Terminal offline TAD performance is still
  absent.
- Do not let this DUCA review cancel or rewrite ChronoTransport. It remains an
  independent parallel route under `AGENTS.md/RTK.md` unless a separate route
  decision changes that contract.

## DUCA K=192 course guardrails

- Do not call Job `1193437` a sampling-rate-only arm. Contribution distillation,
  detector-gradient feedback and full ASFormer adaptation are active in Stage
  2; the only changed model variable versus the K=384 course is the sampling
  budget and its required tensor shapes.
- Do not initialize K=192 Stage 2 from the K=384 Stage-1 checkpoint. K=192 has
  its own 30-epoch exact-uniform Stage 1, and Stage 2 must strictly load its
  terminal epoch-29 EMA artifact.
- Do not treat precheck failure `1193418` as model evidence, relax strict
  loading, or repeat its obsolete launcher path. Corrected precheck `1193433`
  already passed.
- Do not submit another K=192 formal course while `1193437` is pending or
  healthy. Do not use intermediate mAP for checkpoint selection.
- Do not infer learned-selector benefit at 25% from the K=384-to-K=192
  comparison alone. That attribution would require a separate terminal
  matched-uniform K=192 control.
- A formal `tools/test.py --metrics-json` run requires
  `post_processing.save_dict=True`, because the structured receipt hashes the
  saved final prediction file. Leaving `save_dict=False` can compute and print
  the full official metric and then fail only at evidence packaging. Treat
  that as a post-evaluation receipt defect, preserve the original exit code,
  and repair it with one exact-commit evaluation-only job against the same
  sealed terminal EMA checkpoint. Do not rerun training, loosen checkpoint
  loading, enable raw-prediction replay, or reinterpret the packaging failure
  as model evidence.

## 2026-07-27 scientific-claim guardrails

- Do not call the K=384 `65.385724%` result a fair official-60 endpoint. Stage
  1 trains the full detector for 30 epochs and Stage 2 trains it for another
  60, so the result consumes a 90-epoch model-optimization budget.
- Do not repair the comparison by choosing Stage-2 epoch 50. Its
  `65.650497%` is a useful best-observed curve diagnostic, but still consumes
  80 total training epochs. Best-checkpoint selection is admissible only when
  every arm uses the same held-out rule and maximum update budget.
- Do not claim the approximately `+0.896pp` difference from the 60-epoch
  `64.49%` uniform run is caused by learned selection or by any individual
  plugin. Training length, selector, contribution supervision,
  detector-gradient feedback and temporal-network adaptation are confounded.
- Do not describe the sparse K=384 selected-axis run as an untouched official
  AdaTAD baseline. Nominal head configuration, projection, cls/reg objectives
  and NMS remain official-derived, but the active `ActionFormer` and
  `AnchorFreeHead` source files, sparse sampling, target mapping and detector
  wrapper are project extensions. The source files are not byte-identical to
  upstream, execution parity is unproven, and there is no canonical official
  50% sparse-input configuration.
- Do not dismiss the historical approximately `65.696%` uniform result as
  unreliable. It used an explicit physical-grid detector head and therefore
  exposes a real model-coordinate gap relative to the `64.49%` selected-axis
  control; it is a different geometry, not a hidden duplicate baseline.
- Do not prioritize more loss plugins, broad seed sweeps, repository
  governance or deployment profiling before the main model question is
  resolved. The first scientific question is whether physical-time-aware
  assignment and regression recover the performance lost by interpreting
  nonuniform observations as equally spaced.
- Do not assume Stage 2 needs Stage 1 or can start from scratch. Test both under
  one total 60-epoch budget, with exact-uniform initialization and a
  preregistered schedule.

## 2026-07-27 pure-plugin and official-baseline guardrails

- Do not call the current 30+60 full-model course the original DUCA
  multi-course design. The original detector-facing contract is one total
  60-epoch/6,000-update course; any detector-free frontend P0 is accounted for
  separately and cannot silently become 30 extra detector-training epochs.
- Do not use the local approximately `68.29%` dense result as the canonical
  official AdaTAD score. The upstream VideoMAE-S `768/160` table reports
  `69.03%`; weight provenance, schedule and checkpoint selection must be
  matched before explaining the gap.
- Do not call K=384 selected-axis exact-uniform `64.49%` the native official
  1/2-downsampling baseline. It remains a useful wrapper control, while clean
  upstream native-uniform K=384 and K=192 baselines are separate missing
  experiments.
- Do not say that the active detector/head implementation is untouched merely
  because its nominal configuration and loss/NMS settings are official-derived.
  Exact commit `42dba3f9` extends both `ActionFormer` and `AnchorFreeHead`;
  disabled optional branches do not by themselves prove upstream equivalence.
- Do not present detector-head timestamp injection, physical-coordinate
  assignment/regression or a dedicated sparse head as the main evidence for a
  pure pre-backbone plugin. Such variants must be labeled diagnostic or
  enhanced integration and compared separately.
- Do not claim the final DUCA model is implemented or paper-ready. The current
  repository contains important selector, contribution and gradient
  mechanisms, but the recovered bounded-transport model has not yet been
  implemented or evaluated under a fair total-60 budget.
- Do not claim current frozen-prior results are fully training-free
  plug-and-play. They avoid target-domain selector training but still train the
  detector. A released detector plus selector with no optimization remains an
  untested stricter route.
- Do not infer that Stage 2 can start from scratch or that Stage 1 is necessary.
  Compare joint-from-scratch and short-warmup release under the same total
  update budget and shared checkpoint-selection rule.
- Do not launch more 30+60 repeats to repair the paper claim. First recover the
  clean dense/native-uniform baselines and run the compact total-60 model loop.
- Do not interpret the running K=192 course as learned-selector gain. It is
  over-budget, combines several trainable mechanisms and lacks a clean native
  K=192 uniform endpoint.
- Do not let baseline recovery consume the research agenda. It is the minimum
  identifiability condition for judging whether the theoretical selector
  improves a detector, after which effort returns to model analysis and gain.

## 2026-07-27 approved multi-scale alignment guardrails

- Do not treat one-frame hard-swap correlation as sufficient proof that a
  continuous detector gradient can guide the full exact-K selector. A
  one-frame swap is only the local finite-difference layer.
- Before enabling direct detector gradient in the fair total-60 final arm,
  also test dispersed 1%/5%/10% replacements, contiguous-block replacements
  and complete hard re-decoding after 0.25/0.5/1.0 density steps.
- Do not average away a multi-frame failure with a positive single-swap
  result. If multi-frame or global perturbations are stably anti-correlated,
  keep only detached normalized rank/transport supervision.
- All utility candidates, thresholds and gradient-scale choices must be
  frozen on a training-side holdout. Official test GT and test mAP cannot
  select the perturbation scale, loss weight, checkpoint or model arm.
- Do not launch another 30+60 course. The approved paper-facing matrix uses
  one shared maximum of 60 detector-training epochs and 6,000 successful
  updates.

## 2026-07-27 total-60 Pro-review absorption guardrails

- Do not cite the review's K192 `D`/intermediate-only status as current. The
  sealed terminal official result is `57.967272%`, but it remains a 90-epoch
  over-budget diagnostic without clean native K192 uniform.
- Do not turn reviewer-proposed density bounds, `4/K` CDF shift, DP
  gap/anchor constants, linear-rank RDD, publication deltas or cost ratios
  into frozen project facts. They remain `designed_reviewer_proposal` until
  clean-baseline, geometric-reachability and video-cluster power analysis
  freeze them before formal results.
- Do not call inverse-CDF, cumulative-quality sampling, dynamic programming
  or learned frame selection the paper novelty. The candidate novelty is the
  TAD-specific bounded exact-K geometry, hard-policy-validated task utility
  and strict detector-agnostic plugin evidence.
- Do not inverse-map proposals only after NMS in the strict-plugin path.
  Nonlinear time warps do not preserve IoU. Raw proposals must be mapped from
  q to physical t before unchanged official NMS.
- Do not keep the current gradient-derived contribution teacher when
  `G_rank` fails. Also do not overstate that failure as a proof that every
  possible task-utility target is impossible; one explicit hard-utility
  redesign is allowed, while repeated loss swapping is not.
- Do not run RDD-defined A1/A2/A3 long training before `G_rank` passes.
  Projection, decoder and coordinate code may be implemented in parallel,
  but those arms otherwise lack an admitted selector learning signal.
- Do not use a 1,000-update A4 fork after a completed 6,000-update A3 as a
  fair main result. It is a development gate only; a formal A4 must fit inside
  the same 6,000-update budget from the common initialization.
- Do not blanket-ban intermediate checkpoints or freely select them. Without
  a disjoint training-side selection set, use terminal EMA. With such a set,
  all arms must share the same maximum updates, evaluation frequency, metric
  and selection rule, and official test must not choose an epoch.
- Do not let PR #3 cleanup, review-surface governance or logging become the
  model critical path. They may proceed in parallel; clean baselines, the
  unique mathematical contract and `G_rank/G_direct` have priority.
- Do not indefinitely postpone the true frozen-detector train-free mode. Once
  parity and the shared decoder/coordinate contract pass, a minimal frozen
  baseline may run in parallel, with claims kept separate from task-adapted
  results.

## 2026-07-27 dynamic-K / RIME response guardrails

- Do not treat the 4,589-line takeover response as a frozen design or current
  implementation. Its scientific direction is accepted with major
  corrections; `DUCA-RIME`, strict nestedness, exact thresholds and the
  dynamic-K paper role remain `discussed`.
- Do not silently combine its two incompatible decoder contracts. The compact
  response requires independent-per-K exact-K sets, while the expanded report
  later requires strict nesting. Compare independent, strict nested and at
  most one predeclared weak-overlap family on a train-only Oracle, then freeze
  one. If sets are independent, call the target budget-policy value, not
  group-add marginal value.
- Do not revive the old `dynamic_must` implementation or its negative evidence
  as if it were RIME. The old prefix controller, selected-axis path and greedy
  center-radius decoder do not implement hard utility, paired risk, bounded
  transport, physical-time-before-NMS or K-bucket execution.
- Do not call dynamic K, scorer, ILP, inverse-CDF, nested prefix, cheap-global
  to sparse-heavy computation, risk calibration or dynamic frame selection
  individually novel. AdaFocus/AdaFocusV3/Uni-AdaFocus, AdaFrame, MGSampler,
  SMART, AdapTok and related work occupy those ingredients. Any claim must be
  the empirically closed TAD-specific combination.
- Do not freeze the response's `+0.75/+1.0pp`, 20/40% gap recovery, 25/30%
  cost reduction, density/gap constants or `G_rank` thresholds. First obtain
  clean-baseline video-cluster variance and power; preregister one table before
  formal results.
- Do not compare a dynamic mixed-K model only with uniform K=384/K=192.
  Include uniform positions under the identical per-video K sequence, clean
  per-K uniform or an equivalent complete control, identical mixed-K exposure
  and a K-histogram shuffle.
- Do not let the same video appear across detector training, hard-label
  generation, utility fitting, dual/risk calibration and certification.
  Different overlap settings on the same THUMOS validation videos are not
  independent populations.
- Do not use sliding windows as i.i.d. observations for budget or risk
  certificates. Freeze thresholds and report confidence at the complete-video
  cluster level.
- Do not hide hard-utility label generation, EMA refresh, K-bucket wait,
  decode/preprocess/H2D, selector/solver, head/NMS or energy from the cost
  ledger.
- Do not let `risk_infeasible -> Kmax` escape the average-budget accounting.
  Its frequency, realized K and cost must be certified.
- Do not implement or train the full dynamic model before clean parity,
  `q -> t -> NMS`, dynamic Oracle, decoder-family regret, `G_rank` and
  pair-risk gates are written and passed.

## 2026-07-27 dual-response comparison guardrails

- Do not merge the two takeover replies into a fictitious unanimous
  implementation spec. Their common scientific core is accepted, but their
  K grids, training forwards, losses, risk treatment, fallbacks and numeric
  gates conflict.
- Do not proliferate DUCA-METER, METER-TAD, MERTAD and MERTAD-Lite as
  simultaneous project models. Use the existing internal `DUCA-RIME` node
  until one decoder, training contract and empirical route are frozen; perform
  a publication-name search only after the route passes.
- Do not let strict nesting win because both reviewers prefer its narrative.
  Offline one-shot pre-backbone acquisition has no inherent progressive-cache
  requirement. Independent, strict nested and one weak-overlap family must be
  compared by train-only Oracle regret.
- Do not treat deterministic exact-K/gap/coordinate feasibility and learned
  paired-risk prediction as the same kind of guarantee. The former may be hard
  constraints; the latter is an empirical surrogate until independently
  calibrated and certified.
- Do not claim discrete strong duality, arbitrary-test-set exact average
  budget, distribution-shift risk coverage or mAP guarantees from the proposed
  price, Hoeffding or endpoint/tIoU sketches. State their assumptions and
  report realized violations.
- Do not use old snapshot statements or `.codex_tmp`/dirty-worktree variants as
  current canonical code facts. Verify the tracked target path and exact commit
  before changing implementation status.

## 2026-07-29 SparseHead decode-cross v8/v10 recovery guardrails

- Do not report Job `1201495` as a completed evidence suite or as a legitimate
  positive/negative model result. Four replay completions exist, but the
  explicit suite failed and no suite completion was written.
- Empty fatal findings have one frozen JSON type: array `[]`. A producer-side
  mapping `{}` is not interchangeable merely because both are empty; the exact
  producer/consumer schema must pass its regression and full suite.
- Do not change model, configuration, epoch-59 checkpoints, seed42, data,
  evaluator, thresholds or claim boundaries to repair
  `decode_cross_completion_fatal_log_findings_container_type_v1`.
- v9 is an immutable zero-job pre-submission diagnostic. Its missing
  repository-local Git author failure must not be repaired in place or reported
  as a runtime/model experiment.
- `1203046` is `sbatch --test-only`, not a Slurm experiment. The only formal
  successor is Job `1203047` at exact v10 commit/tree
  `c878fbe3a5e960671f03d93fff8367ed3414f5c5` /
  `8d3e73bb26544d1bcf7bfb61154d0b003f2658e0`.
- v10 gate SHA-256
  `e5516af02289d15dd1465f5387471bb1a3c357873980d22645c08acbf6aa141c`
  is valid CUDA/parity evidence only. Do not duplicate the running job or
  promote gate success to replay/suite/metric evidence.
- v10 selected-online / selected-EMA completion SHA-256 are
  `a4e727cf094127be7b91a4a13b140463ad9dc3e0c8c1bcfa3acb9887b5ff6dda` /
  `0c6f87617b1cbd6a5bc4a6be6e9a5a2174f8a5a568c2f24db7253c15a315b8dc`.
  They are valid `tested` replay components.
- Physical-online completion SHA-256
  `02384da2c71c93bdcd6ce003cd59451510c9d095e222653202f09f38b73b153f`
  is now the third valid `tested` component; its former direct-only row is
  superseded by the full dual-axis completion. These three completions still
  do not imply that physical-EMA or the explicit suite passed. Do not promote
  them to a route conclusion while Job `1203047` continues with physical-EMA.
- Physical-EMA direct/native Avg-mAP `0.5760868491267752` is
  `diagnostic_only` until its replay and validator write a valid completion.
  Do not count `DIRECT_INFERENCE_COMPLETE` as the fourth component or infer
  explicit-suite success from it.
- v8's eight metric rows remain `diagnostic_only` until the v10 fresh four-run
  chain and explicit suite pass. Do not begin model-negative Pro attribution
  from the incomplete v8 suite, and do not suppress a legitimate v10 negative
  result if the full evidence chain later validates it.

## 2026-07-29 SparseHead v10-terminal / v16 guardrails

- Do not describe Job `1203047` as still running. It is terminal `FAILED 1:0`.
  All four replay completions are valid `tested` components, but the explicit
  suite failed with
  `decode_cross_suite_checkpoint_binding_schema_shape_mismatch_v1`.
- Do not equate artifact-record dictionary shape with artifact identity.
  Preflight and gate may carry different metadata while binding the same
  canonical resolved path and file SHA. Continue to validate both records and
  state-dict hashes independently.
- Do not promote the four v10 components to a complete route result. Suite log
  SHA-256
  `68b7b3d34e587392bdac2df1eb2a36d971009d4c07165ef2a18157449ccb931f`
  proves the final consumer failed before a suite completion was written.
- v11–v15 are immutable zero-job roots. Do not repair them in place, reuse their
  partial logs, or count preparation/test-only identifiers as Slurm experiments.
  In particular, `1203916` is v16 `sbatch --test-only`, not a formal job.
- The only formal successor is Job `1203917` at v16 commit/tree
  `54e7f9abeaabf710a505f0a0f595a4eb3bb47f98` /
  `f8490f9c25c2e0e6958c406e19c83cc3d5a40535`. Do not submit another copy while
  it is pending/running.
- Do not treat v16 `78 passed`, preflight success, deployment identity or the
  now-passed CUDA gate (artifact SHA-256
  `0d2153effee84a0e1aa6410125bb291eb4ef4d41e4b40604f49d9e5868e0ada9`) as
  final metric evidence. The required order remains gate, four fresh replay
  completions, then explicit suite in one allocation. Selected-online direct,
  replay producer and formal validator have now completed; formal
  `DECODE_CROSS_COMPLETE.json` SHA-256 is
  `6937fc6b7b050fd7009ee967ceef446aebaa8b3daa695c7959106ff87048c038`
  with `status=tested`, `validation_pass=true` and
  `fatal_log_findings=[]`.
- v16 selected-online uniform/physical Avg-mAP
  `0.4125660433077075 / 0.5015355102106833` and producer SHA-256
  `97410d9855a3f6db859e36213bf6b201e10c96941a164b5588af02cdfba4ee20`
  are one reproducible formal `tested` component.
- v16 selected-EMA is now a second formal `tested` component:
  uniform/physical Avg-mAP
  `0.41283020792762315 / 0.5009785403306161`, formal completion SHA-256
  `4a1b405b7849f396e1b649da8895070e6176023c4a959c6d7fd9148f2bd8afe0`.
  Job `1203917` has entered `physical_online`; the two physical-arm completions
  and explicit suite remain mandatory. Do not turn close online/EMA values into
  a final model claim or begin final attribution before the suite.
- `physical_online` producer completion
  `d61d8fbf8b977b59b65eb87d55227904b2a5a2e6994e584226bda19a265b26eb`
  and physical/uniform Avg-mAP
  `0.5755558109390063 / 0.40107677185286417` now have formal completion
  `fd18348e6ae6ecf4bdc4390ca4620a109616582f7f77138ed137085e0df6c260`
  with all contracts passed, so this is the third `tested` component. Do not
  relabel the strong `+17.447903908614215 pp` axis gap as a final route result
  or start negative-method attribution before `physical_ema` and suite.
- `physical_ema` formal completion SHA-256
  `cd6da2f827524e0b9eb2b46c6cbbcc5b6e89243aa9cd8d7e45efafcb4cb6b565`
  passes every validator contract; uniform/physical Avg-mAP are
  `0.40296498031949024 / 0.5760868491267752`
  (`+17.312186880728497 pp`). All four formal components are now `tested`.
  Do not call this the route result or begin final attribution until the
  explicit suite artifact exists and Job `1203917` reaches a verified terminal
  state.
- If v16 produces a legal negative method result, preserve it and start the
  required multi-explanation Pro analysis. Do not reclassify it as engineering
  failure or silently change checkpoints, seed, data, evaluator, thresholds or
  model settings.

## 2026-07-29 SparseHead v16 terminal-analysis guardrails

- Do not describe Job `1203917` as running or suite-pending. It completed
  `0:0`; suite completion/validated SHA-256 are
  `ed2770c35cf9a3acd5fa80465eda1c34b3541ba3dea404c75388aaeffefbdc31` /
  `f2da143127b3a01aef7bda451e2351c494f72552f3810f604f895f4c0a7767d3`,
  both `validation_pass=true`.
- Do not call the result globally negative. Physical-time decode improves all
  four frozen conditions by `+8.81--+17.45 pp`; this is `tested` support for
  the decode-axis hypothesis and a rejection of harmless selected-rank decode.
- Do not use that gain to claim SparseHead/SDPQ superiority. The matched
  20-epoch SDPQ result remains below physical control, and cross-checkpoint
  selected/physical differences are descriptive, not causal training effects.
- Do not infer assignment/support causality, class behavior, calibration/NMS
  mechanism, multi-seed robustness or cost from the suite. Those fields are
  absent and require dedicated measurements.
- Do not launch a full retrain from this analysis. First perform an independent
  sealed-artifact mapper/NMS/GT evaluator and a fixed-window assignment/support
  audit. A repeated production evaluator is not an independent check.
- Do not revive uniform-rank decode as the default SparseHead baseline. Until
  falsified by an independent replay, all future matched sparse-head work must
  treat physical-time-before-NMS as the coordinate-correct baseline.
- Status is `tested`, not `empirically_supported` or `paper_ready`; single seed,
  no independent evaluator, no full cost and no cross-backend result remain
  hard claim boundaries.

## 2026-07-29 official-comparability and diagnostic-closure guardrails

- Do not put v16 VideoMAE/K384, historical `63.61`, the old documented
  ActionFormer `62.6`, or the released `66.83` in one matched-delta column.
  They are distinct protocol/evidence strata until a full official receipt is
  generated.
- An official ActionFormer main-table row must use exact upstream commit
  `61ea7eb9308a568b0cf45e3804830836e30061de`, tree
  `7b06c5261ba244788c942a0d73e304581bc35154`, config SHA-256
  `73f8aeaf7deef93aba57259badd4c454990ec1e0ce6eaa7c3434db44baaeeaf0`,
  README SHA-256
  `bdee4eb088a74e190935097742c7dbfaf254eb912f79729dccd73b9b36b33db8`
  and THUMOS archive MD5 `375f76ffbf7447af1035e694971ec9b2`.
- Do not accept schema-shaped JSON or `--skip-artifact-hash` as evidence. The
  official row requires live config/README/archive/data/annotation/class-map/
  feature/checkpoint/raw-prediction/log/evaluator/environment/run receipts,
  plus exact agreement between parsed official log and an independent rerun of
  the pinned official evaluator.
- Do not permit arbitrary matched-difference prefixes. The only current
  one-variable bundles are `selection_budget`, `head_projection` and
  `coordinate_geometry`; protected dataset, schedule, evaluator, NMS,
  checkpoint policy and integrity fields must remain identical.
- Do not call local `35 passed` a benchmark result. Commit
  `57917e7bf2b991478b4f6fc4ce1db5ca5878b68d` only makes the diagnostic and
  comparability tools `tested`. Remote artifacts, the official reproduction
  and matched method rows remain pending.
- Do not run the 64-window SDPQ support audit against a G1a ActionFormer
  checkpoint. It requires an exact compatible
  `SupportDecoupledPhysicalQueryHead` config/checkpoint pair; missing evidence
  is a blocked diagnostic, not permission to substitute another model.
- Do not launch structural rescue training from the v16 decode result. Run the
  independent v16 replay, support audit and official same-protocol anchor
  first; then choose at most one structural intervention after bounded Pro
  review.

## 2026-07-29 active official-reproduction guardrails

- Do not treat NaNs in the padded tail of a replay axis as model non-finites.
  For each window, only `axis[:native_valid_count]` must be finite and strictly
  increasing; every padded element must be NaN. In the frozen v16 capture all
  1,443 NaNs per axis are padding and all 792 valid prefixes are finite.
- Do not filter the OpenTAD THUMOS annotation with literal subset `test`.
  Logical evaluation `test` is explicitly bound to annotation subset
  `validation`; the frozen contract is 211 videos, 3,325 GT and 20 classes.
  Missing or mismatched counts must fail closed.
- Do not hard-code epoch 59 into the diagnostic SDPQ support audit. It must
  require an explicit expected epoch and exact state key. The available matched
  checkpoint is epoch 19 online-only; absent EMA must never fall back to online.
- Do not call test-only IDs `1204959`, `1204960` or `1204980` Slurm jobs.
  Job `1204961` failed before Python because `/bin/sh` cannot enable
  `pipefail`; preserve it. Job `1204981` is the sole Bash-wrapped successor.
- Do not reuse incomplete official download roots. v1 records missing proxy
  export and v2 records home-cache quota exhaustion. Only v3 contains the exact
  official THUMOS MD5 `375f76ff...ec9b2` and release-package SHA-256
  `e028f7e4...b034c929`.
- Do not use downloaded bytes alone as an official result. The released
  checkpoint must still generate raw predictions, pass the pinned official
  evaluator, match an independent recomputation and receive the strict
  `paper_main_table` verdict before any matched SparseHead comparison starts.

## 2026-07-29 sealed protocol and padding guardrails

- Do not call Job `1205131` a failed model experiment. Its inference/evaluator
  produced Avg-mAP `66.83`; the failure was the obsolete literal split-count
  contract `official_annotation_split_schema_contract_v1`. Also do not promote
  `66.83` yet: only successor Job `1205178` can issue the strict
  `main_table_eligible` verdict.
- Do not collapse nominal dataset cardinality and evaluated annotation
  cardinality. THUMOS nominal test is 213, the pinned official annotation DB
  contains 212 case-normalized `Test` videos, and the feature inventory has 413
  files because `video_test_0001292` is feature-only/unannotated. Exact set
  equality, canonical class mapping and full feature-content validation are
  required; a count-only check is insufficient.
- Do not reuse older Windows/text-converted config or README hashes as the
  official Git identity. The pinned Linux Git blob SHA-256 values are
  `c0ac0df560cd564941b56cd9391ad0bd5cea386d2e4b6cf9fc8ffcab821955cd`
  and `f0431584b4df0702fa08f961fb0038e1277f41c12b7df47b7d2bfed47e59af23`.
- Do not treat SDPQ Job `1205132` as negative model evidence. Its
  `sdpq_support_overlap_query_padding_mask_omission_v1` failure came from
  unmasked padded queries in the support-overlap branch. The repair must remain
  a post-branch multiplication by `query_mask`, with regression proof that
  valid-query outputs and gradients do not move.
- Do not launch or label a matched SparseHead row as paper-comparable without a
  live base-anchored source-diff attestation. Matching dataset names or metric
  thresholds is insufficient: the implementation must be tied to the pinned
  official ActionFormer base, same I3D bytes, annotation, 30-epoch schedule,
  seed/checkpoint rule, NMS and evaluator, with exactly one declared
  intervention. Current matched rows intentionally fail closed.
- Test-only IDs `1205176` and `1205177` are not jobs. Formal Jobs `1205178`
  (official anchor), `1205179` (support diagnostic) and `1205133` (independent
  replay) are unique; do not duplicate or cancel them while pending/running.

## 2026-07-29 official reseal and numeric-semantics guardrails

- Do not publish the `66.833392` result from Job `1205206` as an official
  main-table row yet. The metric is real, but the old record asserted seed `0`
  while official config/log evidence says `1234567891`; a fresh complete
  fifteen-receipt reseal is mandatory.
- Do not call Job `1205243` a negative model result. It preserved raw scores,
  masks, proposal geometry and every delta sign. Its failure signature
  `independent_recompute_semantic_match_drift_v1` is caused by sorting and
  float32 Soft-NMS semantics in the independent validator.
- Do not widen tolerances or switch back to NumPy stable sort/float64 to make
  independent closure pass. Keep the pinned PyTorch `2.0.1` CPU descending
  unstable sort, scalar float32 Soft-NMS, `libm expf` bit probe, exact
  pre/post-detection equality and `1e-4` aggregate-metric ceiling.
- Preserve Linux/predeployment roots
  `linux_executable_mode_fixture_restaging_v1`,
  `linux_executable_fixture_worktree_mode_drift_v1`,
  `github_tls_clone_termination_v1` and
  `independent_softnms_expf_ulp_mismatch_v1`. They are evidence that the
  current exact gate was earned, not disposable noise.
- Do not interpret the 647/647 SDPQ support audit from Job `1205240` as a
  performance rescue. It proves assignment/support observability only.
- Do not implement K384 by rank-coordinate gather/remap or fixed 768-window
  cropping and label it official-comparable. Preserve the original full-video
  grid, GT/time mapping, point stride and decoder; use one deterministic
  max-K384 mask/scatter intervention and measure actual skipped stage cost.
  Zero filling alone is not an efficiency result.
- Test-only IDs `1205384` and `1205398` are not jobs. Job `1205388` is a
  preserved one-second engineering failure,
  `slurm_module_function_unavailable_v1`; do not call it a validator or model
  result. Formal Job `1205400` is the only active independent-recompute
  successor and must not be duplicated while running.
- Test-only IDs `1205408` and `1205418` are not jobs. Job `1205409` is the
  preserved pre-inference probe failure
  `official_environment_probe_nms_import_order_v1`, not an official metric.
  Job `1205419` is terminal `FAILED 1:0`, not active: it reproduced `66.83` but
  failed the record builder with
  `official_released_train_log_default_serialization_omission_v1`. Do not
  publish it or duplicate it.
- The released train log may be normalized only by inserting the missing exact
  integer `model.fpn_start_level=0` whose provenance is the pinned official
  loader. Pin the raw released-log config hash, hash the normalization
  attestation, require exact normalized equality with the live source-expanded
  config, and reject every other missing field/value/type. This is provenance
  closure, not permission to ignore config drift.
- Preserve the failed v17 GitHub TLS clone root. The v18 runtime was created
  from a SHA-verified complete Git bundle at exact commit/tree `8b80c98` /
  `148a93e`; do not repair or overwrite either root. Its Linux suite passed
  `131 passed, 2 skipped`. Job `1205455` completed `0:0`; `1205454` is
  test-only. Its strict all-receipt verdict is `main_table_eligible=true` at
  Avg-mAP `66.833392`. Do not continue comparing SparseHead against historical
  `63.61`; the official paper comparator is now `66.83`.
- Do not treat deterministic stratified-uniform K384 and video-hash-random
  K384 as interchangeable primary rows. Use native-grid stratified-uniform as
  the preregistered main control; random is a secondary robustness control and
  must use a video-ID/global-seed pure function with no per-step, GT, teacher
  or evaluator dependence.
- Do not claim head efficiency from scattering zeros after dense convolution.
  Unselected query rows must not execute the three-layer classification and
  regression heads. Preserve original FPN indices and scatter selected center
  outputs back before the unchanged loss/decoder/NMS; report synchronized
  measured head and end-to-end cost in addition to theoretical MACs.

## 2026-07-29 official-native K384 no-repeat rules

- Never conflate `model.query_budget=384` with an input observation budget.
  Official I3D input remains dense; only native-grid head queries are selected.
- Do not call the current K384 method execution-only or unchanged-loss.
  `training.loss_support=selected_native_grid_queries` excludes unselected
  positives and negatives and changes the loss normalizer; this is the single
  declared method intervention together with sparse head execution.
- Do not compare only a sparse terminal checkpoint with the released
  `epoch_034.pth.tar`. The paper experiment requires dense and sparse controls
  trained under the same official 5-warmup + 30-epoch loop and evaluated from
  their terminal epoch-35 EMA states.
- Isolated head-path timing is not end-to-end wall-clock evidence. Report it
  only with `wall_clock_claim_allowed=false` until an end-to-end synchronized
  measurement with equal preprocessing, backbone, decoder and evaluator exists.
- `1205539` is a Slurm test-only ID; Job `1205541` is the unique formal gate.
  Pending priority is normal and must not trigger duplicate submission.
- Preserve the v1–v4 preflight roots and signatures
  `slurm_module_function_unavailable_v1`,
  `runtime_profile_source_under_nounset_v1`,
  `github_http2_remote_ref_transport_v1` and
  `github_remote_ref_live_check_transport_hang_v1`; none is model evidence.
- A failed CUDA gate authorizes correction of the actual sparse execution path
  and a new immutable run root, not training. A legal negative model result
  authorizes analysis and preregistered follow-up design, not silent threshold,
  budget, seed, data, checkpoint, loss or evaluator tuning.
- Decode-cross v16 remains a `tested` frozen single-seed diagnostic. Independent
  recomputation Job `1205400` is the sole active recomputation and must not be
  duplicated or cancelled merely because it is slow.

## 2026-07-29 packed-kernel recovery no-repeat rules

- Job `1205541` is no longer pending. Preserve its failed root and the exact
  signature `native_grid_sparse_head_microkernel_launch_and_scatter_slowdown_v1`.
  Do not reinterpret passed numerical equivalence as passed efficiency.
- Do not relax `1.05x`, remove synchronization, omit selector cost, average
  away a failed round or substitute theoretical MACs for real CUDA timing.
- The packed implementation must retain one Conv1d call per head layer across
  all samples/levels, physical radius-three dependency semantics, mask holes,
  zero boundaries, selected-output equivalence, exact zero unselected outputs
  and gradient equivalence. Optimizing by changing budget, loss support,
  precision, convolution weights or tolerance is forbidden.
- `1205566` is test-only and Job `1205567` is the sole successor. Do not submit
  another gate while it is pending/running.
- The pre-submission SCP interruption produced no partial remote file. Preserve
  signature `ssh_transport_interruption_during_pre_submission_receipt_copy_v1`;
  do not classify it as a code, CUDA or model failure.

## 2026-07-29 global-packed stop rules

- Job `1205567` and Job `1205569` are both terminal engineering failures.
  Preserve signatures
  `native_grid_sparse_head_packed_patch_materialization_and_microconv_slowdown_v1`
  and `native_grid_sparse_head_packed_gather_scatter_overhead_v1`; never
  reinterpret their passed equivalence as an efficiency or accuracy result.
- Do not retry the packed Conv1d or flattened-GEMM implementation unchanged.
  Their selector-inclusive speedups `0.4590x` and `0.4577x` are effectively
  identical and falsify “convolution arithmetic is the dominant overhead.”
- The only authorized final PyTorch prototype removes intermediate dense
  scatter/materialization and shares one cls/reg physical plan. It must retain
  exact raw invalid-hole influence at the first layer, zero masked hidden
  states, physical boundaries, autograd, selected-output equality and exact
  zero unselected outputs.
- If that prototype remains below `1.0x`, stop this execution implementation;
  do not create an unbounded sequence of micro-optimizations. The formal
  advancement threshold remains `>=1.05x` in every synchronized round with
  selector cost included.
- Preserve transport/provenance failures
  `github_https_clone_tls_termination_v1`,
  `bundle_clone_remote_head_unset_v1` and
  `github_remote_ref_dns_timeout_during_source_diff_v1`. Local live-ref
  attestation may unblock an engineering gate only; it cannot seal a paper
  main-table row.
- Never use released-checkpoint sparse inference, isolated-head timing or the
  sealed `66.833392` released anchor as a matched causal model result. Main-
  table comparison requires official same-commit dense/sparse retraining,
  terminal epoch-35 EMA, identical data/seed/evaluator, multiple preregistered
  seeds, independent receipts and end-to-end synchronized cost.
- `1205570` is test-only; Job `1205571` is the sole formal global-packed gate.
  It completed `0:0` and passed at `1.571574x` selector-inclusive median
  speedup with exact selected-output equivalence. Do not repeat this gate or
  reinterpret it as an accuracy/end-to-end result. It authorizes matched
  training only; `paper_metric_claim_allowed=false` and
  `end_to_end_wall_clock_claim_allowed=false` remain binding.
- Local live-ref source-diff SHA-256
  `68d2cf726cc8523094847337eb5ebe604ca5ad46cbe99d2a5fba2b78f45e67db`
  remains engineering-only despite being live and exact; provenance
  `e57fc3d618f86faacbd79cb77121796a18fed5fafdc0fe80506e40f9aba6237c`
  explicitly forbids paper sealing.
- Do not compare a retrained K384 model causally against the released
  epoch-034 `66.833392` anchor. Dense and sparse controls must be retrained
  from the same candidate commit with the exact official seed, 5-warmup +
  30-epoch loop, terminal epoch-35 EMA, data and evaluator. The first matched
  seed remains screening; a paper claim requires frozen structure and
  preregistered independent seeds.

## 2026-07-29 official matched-pair execution rules

- `1205572` is only `sbatch --test-only`; Job `1205573` is the sole formal
  matched screening pair. Do not duplicate, cancel, or launch an overlapping
  pair while it is pending/running.
- Do not call the sparse arm execution-only. Its K384 selector and
  `training_loss_support=selected_native_grid_queries` change which positive
  and negative queries contribute to optimization. The matched delta estimates
  this complete method intervention; kernel equivalence/speed is a separate
  engineering claim.
- Never select an implicit latest checkpoint. Both arms must train without
  resume for 35 executed epochs and evaluate explicit `epoch_035.pth.tar`
  `state_dict_ema`.
- Do not accept candidate-native metrics alone. Each raw `eval_results.pkl`
  must cover the exact 212-video official test set and be independently
  recomputed with clean official commit/tree
  `61ea7eb...` / `7b06c526...`.
- Single-seed Job `1205573` is screening even if its metrics are positive.
  Its schema deliberately forbids paper-main-table and end-to-end-cost claims.
  Main-table status requires preregistered paired independent seeds, aggregate
  uncertainty and synchronized full-pipeline cost.
- Preserve failed preflight root v1 and signatures
  `official_data_live_revalidation_import_scope_v1` and
  `preflight_failure_receipt_python_environment_unloaded_v1`. They are command
  environment/import failures before Slurm, not model or data failures. Never
  retry the repository-external import without activating the pinned
  environment and binding the audit root.
- Job `1205573` must not be resumed or requeued. It failed before training
  because a compute node could not resolve GitHub during redundant
  `validate_attestation_live`; preserve signature
  `compute_node_github_dns_during_redundant_live_source_diff_revalidation_v1`.
- Do not make live Internet availability a compute-node dependency. Build and
  hash the remote-bound attestation before submission, then validate the
  sealed artifact and all local Git/config/diff/allowlist content offline
  inside Slurm. Do not weaken or skip source validation.
- Preserve v24 clone signature
  `github_https_clone_tls_termination_during_audit_runtime_freeze_v1`.
  Verified bundle v25 is the only authorized recovery runtime; do not reuse
  the partial v24 path. Bundle verification must run with a Git repository as
  context; an SSH transport close is retriable only after confirming the
  destination bundle SHA.
- `1205579` is test-only; Job `1205580` is the only formal successor. Pending
  priority is normal. Do not duplicate it or use parent partial artifacts.

- Job `1205580` is now terminal and must not be resumed/requeued. It failed
  before any optimizer step because the official declared TensorBoard
  dependency was absent; preserve
  `official_declared_tensorboard_dependency_missing_v1`. Never interpret this
  as a model failure or copy its empty dense artifacts into a result record.
- The only authorized recovery environment is
  `/data/run01/sczc063/yuzibo/projects/python_envs/actionformer_tensorboard_2_20_0_20260730_v1`,
  receipt SHA-256
  `acc5909360970cfad1f390a4f5ab046a3876ac9378448b2f94da26ffb312ece2`.
  It must retain Python `3.10.20`, torch `2.0.1`, CUDA `11.8`, NumPy `1.23.5`
  and TensorBoard `2.20.0`; version or receipt drift is a hard failure.
- `1205583` is test-only. Job `1205584` is the sole formal successor and must
  not be duplicated or cancelled while pending/running. Its v3 root, exact
  candidate/audit commits, preflight, environment and submission receipts are
  immutable.
- Even a fully successful Job `1205584` is only single-seed screening.
  Do not put its dense/sparse delta in a paper main table. First independently
  reproduce both raw-prediction metrics and verify explicit epoch-35 EMA,
  exact 212-video coverage and pair completion; if the frozen decision rule
  passes, then run preregistered paired independent seeds and synchronized
  full-pipeline cost.

## 2026-07-29 official main-table preregistration no-repeat rules

- The five paired seeds are immutable:
  `1234567891/1423812477/737690612/1788897292/1322022747`, seed-set SHA-256
  `a4038a752aa46b97e5854c20574d65ece078bad6124e4778cc4269e75747c7c6`.
  Never substitute a failed or weak seed, stop after a favorable subset, or
  count repeated evaluator calls/videos as additional seeds.
- S0 continuation thresholds were frozen before metrics:
  `Delta Avg>=-1.00 pp`, `Delta @0.6>=-1.50 pp`, and
  `Delta @0.7>=-1.50 pp`. Do not relax them after Job `1205584`.
- Main accuracy preservation requires all five paired terminal-EMA results,
  Avg 95%-CI lower bound `>=-0.20 pp`, and @0.6/@0.7 lower bounds
  `>=-0.50 pp`. A point estimate is not a pass when its interval crosses a
  boundary.
- Never call the isolated-head `1.571574x` gate full-pipeline speedup. The
  formal boundary starts with official precomputed feature loading and ends
  with serialized detections. It requires median `>=1.05x`, lower CI
  `>1.00x`, and no short/medium/long stratum interval crossing unity.
- Do not call the cost boundary raw-video end-to-end: I3D extraction is
  upstream and unchanged. If raw-video system cost is reported, add identical
  extraction explicitly to both arms.
- The preregistered 2x2 cross uses the same dense-trained and selected-trained
  terminal EMA checkpoints under dense/K384 evaluation. It cannot select K,
  halo, normalizer, checkpoint or NMS.
- A 413-file count is insufficient for a paper run. Rehash expected IDs,
  content, shape and dtype inside each allocation and bind runtime
  effective-config/CLI/split/epoch35 EMA receipts.

- Job `1205584` must not be resumed or requeued. Preserve
  `official_actionformer_softnms_extension_abi_shadowed_by_opentad_v9arg_v1`.
  Its dense epoch-35 checkpoint completed training but never produced an EMA
  metric; sparse never started. Do not call it a dense result, reuse it in a
  pair, or skip dense training in the successor.
- Never let unqualified `import nms_1d_cpu` silently resolve the OpenTAD
  site-packages extension. The official ActionFormer caller requires the exact
  seven-argument ABI; the conflicting module requires `t1,t2`. A successor
  must use a receipt-bound isolated official extension, assert its resolved
  path/hash, and execute a real seven-argument Soft-NMS probe before training.
- Do not “fix” the official caller by adding OpenTAD-only `t1,t2` arguments.
  That would change evaluator/post-processing source semantics and break the
  official comparison. Repair environment/module provenance instead.

## 2026-07-30 official-comparable S0 successor no-repeat rules

- The only authorized NMS recovery environment is
  `/data/run01/sczc063/yuzibo/projects/python_envs/actionformer_official_runtime_20260730_v2`,
  receipt SHA-256
  `13d57c1161905f059204f7101f26029503a03da7f5eb44b81c418a0b97999f24`.
  The official seven-argument extension SHA-256 is
  `b67e0e41f9f55cd69e8b90cfc75a1947214365857d851a510047838ad49ed98d`;
  path/hash/import and a real call must all pass before training.
- Do not change THUMOS splits to generic defaults. Pinned upstream
  `configs/thumos_i3d.yaml` itself uses `validation` for training and `test`
  for evaluation. Candidate and official config bytes both hash to
  `c0ac0df560cd564941b56cd9391ad0bd5cea386d2e4b6cf9fc8ffcab821955cd`.
- Do not rely on a feature-file count for this recovery. The 413-file live
  ID/content/shape/dtype rehash exactly reproduced sealed manifest SHA-256
  `cda269dace393b9af1f6fcb87a9a531beed69e3c71279ba3ca2cee76e198d59a`;
  preserve receipt
  `73a2f714c100f541306d7d7f9c32e36481574d2ac6c5e78925ee4ee1dcca96b3`.
- `1205593` is test-only. Job `1205594` is the sole formal v4 successor.
  Pending priority is normal; do not duplicate, cancel, resume or seed it
  from Job `1205584`.
- A successful Job `1205594` is official-comparable S0 screening, not a paper
  row. Apply the preregistered delta thresholds first, then run all five fixed
  paired seeds, 2x2 attribution and synchronized detector-pipeline cost before
  any main-table or efficiency claim.

- Job `1205594` is terminal and must not be resumed or requeued. Preserve the
  recurring signature `official_environment_probe_nms_import_order_v1` and
  failure-analysis SHA-256
  `06bbc29e5f57b3b9a12f421f5ddd814487bf01733d0f0e5bbcc4c0551c877a41`.
  It failed before any test or training and contains no metric.
- Do not retry the same probe after checking only module path, SHA and ABI
  arity. The focused contract must additionally assert `import torch` precedes
  `import nms_1d_cpu`, because PyTorch loads `libc10.so` for the extension.
- Exact recovery commit/tree are
  `98f5b875315b4a2b5c6829f5d74ccce68f478e47` /
  `2e6b4bba6868c323d70c97140f7cbed044eb1a7b`; clean v28 bundle SHA-256 is
  `713a1d839e8e8ea50f141df9dba1feb44dc43c91dffbd4dd85bf8910bbdf9e24`.
  No successor may be submitted until its real remote ordered-import probe and
  focused suite pass.

- The v28 ordered-import probe and focused suite passed. `1205598` is only
  test-only; Job `1205599` is the sole formal successor. Pending priority is
  normal. Do not duplicate or cancel it.
- Job `1205599` must still retrain both arms from scratch and remains
  `paper_main_table_eligible=false`; the import-order fix authorizes execution,
  not a model or paper claim.
- Job `1205599` is running on g0030 and has passed its environment/source/test
  gates. Normal finite dense training through epoch 24 is not a reason to
  resubmit, cancel or infer performance.
- Dense Avg `66.583013` is a valid same-commit single-seed control, but it is
  not the paired result and must not be substituted for the released `66.83`
  anchor or combined with failed Job `1205584`. Wait for the sparse ARM and
  MATCHED_PAIR_COMPLETE before applying S0 thresholds.

## 2026-07-30 Job 1205599 legal-negative no-repeat rules

- Job `1205599` is terminal `COMPLETED 0:0`; do not resubmit, resume, requeue or
  relabel it as an engineering failure. Pair completion SHA-256 is
  `545e420aa1d437aedeffd15cb30390ceb0cfe4d6565d7eb35c53a8bf17ac76fd`.
- Exact same-commit dense/sparse Avg-mAP is `66.583013/43.919699`, delta
  `-22.663313 pp`; @0.6/@0.7 deltas are
  `-25.472328/-24.218944 pp`. The frozen S0 continuation rules fail. Do not
  launch the remaining four seeds or detector-pipeline cost for this
  intervention.
- Do not rescue the result by changing K, selector, loss support, normalizer,
  threshold, NMS, seed, split, data, checkpoint, decoder or evaluator after
  seeing test performance. Any redesigned arm requires a new preregistration
  and cannot overwrite this result.
- Do not call the result “SparseHead generally fails.” It rejects only the
  fixed K384 deterministic native-grid execution plus
  `selected_native_grid_queries` training-loss intervention.
- Do not promote this S0 row to a paper main table or efficiency claim. It is
  `paper_main_table_eligible=false`; the isolated `1.571574x` head
  microbenchmark does not offset the accuracy collapse and is not
  detector-pipeline cost.
- The only immediate evaluations authorized on the frozen checkpoints are the
  preregistered no-retraining 2x2 attribution and diagnostics. Cross-eval rows
  are not new seeds and must not be used to choose K, loss, checkpoint or test
  threshold.
- Post-NMS `eval_results.pkl` supports retained-output recall, boundary,
  duration, class and failure-video diagnostics only. It cannot prove pre-NMS
  proposal recall, suppressed-proposal behavior or full score calibration;
  do not overclaim NMS causality without new pre-NMS captures.

## 2026-07-30 S0 attribution closure no-repeat rules

- Do not repeat the no-retraining 2x2. Job `1205701` is terminal and receipt
  bound. Its Avg matrix is full×dense `66.583013`, full×K384 `45.784332`,
  selected×dense `64.537343`, selected×K384 `43.919699`.
- Do not blame selected-loss as the primary cause. Its average main effect is
  `-1.9552 pp`; K384 execution is `-20.7082 pp`; interaction is only
  `+0.1810 pp`.
- Do not claim score calibration/NMS is sufficient. Class-agnostic recall and
  rank-based fixed-topK recall already collapse, and same-label high-overlap
  support falls about 71%.
- Do not repeat the exact 64-window assignment audit. Job `1205799` completed
  `0:0`; suite SHA-256 is
  `475b61ddad4b0b56a86b2e2616ef2584b252c3169b4ad1268223f21d6e118567`.
  K384 positive retention is `16.9423%`, with `395/804` GT lacking candidates
  and `427/804` lacking assignments.
- Do not use that assignment audit as test performance or model selection. It
  uses the official `validation` training split, exactly 64 windows, no test
  GT, no backward and no new training.
- Do not treat historical `63.xx` PhysTime/random-sampling values as the
  matched baseline. The causal S0 baseline is the same-run official dense
  `66.583013`; the released `66.833392` is context only.
- Do not resurrect hard query deletion under a new selector name. Any method
  that zeros/removes unselected proposals or couples dense scaffold
  supervision to the sparse mask repeats the falsified mechanism.
- The only authorized continuing design is DCSR: dense cheap scaffold plus
  sparse expensive residual refinement. It remains `designed`; do not claim
  implementation or performance.
- Do not run official test-set budget/threshold sweeps for DCSR. Architecture
  and one final budget must be frozen on an internal holdout from the official
  training split before five paired official seeds.
- Do not claim efficiency from head-only microbenchmarks. Measure feature
  load/preprocess, H2D, scaffold, selector, refinement, scatter, decoder and
  NMS on matched hardware, and keep the claim limited to the measured
  precomputed-feature detector boundary.
- The S0 integrity audit is `WARN`, not PASS, because scope is one official
  seed and configured external cross-model file reviewers were unavailable.
  This warning must not be hidden or misrepresented as metric fraud.

## 2026-07-30 DCSR G0/G1 no-repeat and paper-boundary rules

- Do not repeat G0 identity under a different name. Job `1206168` completed
  `0:0`; receipt SHA-256
  `b87fc59ec6529e83e99f7bf5fbfb7f3bff5ec637060c62057da07a669a8c1ff4`
  proves exact state keys, native points, full masks, pre-decode outputs and
  final official Soft-NMS/timestamps.
- Do not call G0 a cheap-head, accuracy or efficiency result. It deliberately
  uses the full official dense head with residual disabled and records
  `metric_claim_allowed=false`, `efficiency_claim_allowed=false`.
- Do not confuse G1 with hard K384 deletion. G1 keeps a one-layer scaffold on
  every valid query and only sparsifies a signed three-layer residual; original
  full masks, targets and positive normalizer remain active.
- Do not use internal G1 numbers in a paper table. Its frozen 160/40
  official-validation manifest SHA-256 is
  `ba683bc5ddbb1fe219fab0545e9d808808d9b25fc9b32e7c5c0b6339b68b9bbb`;
  test records/GT/AP are not used.
- Do not resubmit formal array `1206273_[0-2]` while pending/running. Number
  `1206266` is only `sbatch --test-only`.
- Do not treat Jobs `1206160` or `1206166` as model negatives. They stopped
  before tensor comparison due distinct shell/export and repository-import
  launch contracts. Their v3/v4 roots must remain unchanged.
- Do not invoke repository tools as `python tools/...` in the formal launcher;
  use the tested `python -m tools...` entrypoints from the exact candidate
  root.
- A negative G1 gate is a legal internal model result. Preserve all per-seed
  values, run the preregistered Pro-level analysis, and do not tune or retrain
  before recording a new design decision.
- A positive G1 result still cannot support an official claim. G2--G4 must
  freeze selector/floors/budget before five disjoint paired official seeds and
  complete feature-to-final-detection cost.

## 2026-07-30 DCSR G1 closure no-repeat rules

- Do not repeat G1 or proceed to G2--G4. All three paired seeds completed and
  the frozen gate failed by `-7.556202 pp` Avg and
  `-11.043134/-11.019821 pp` at 0.6/0.7. More seeds cannot rescue a protocol
  that already triggered its preregistered kill.
- Do not call the G1 absolute Avg `49.25` an official ActionFormer result or
  compare it with historical `63.xx`, released `66.833392`, or official S0
  `66.583013`. G1 is a 40-video internal validation-holdout gate.
- Do not blame selected-query loss. G1 retains full-grid masks, targets,
  supervision and official normalization.
- Do not blame K384 support as the sole or dominant G1 cause. Scaffold-only is
  already `-7.4181 pp` versus dense, while the K384 support penalty versus
  all-query residual is `-1.2395 pp`.
- Do not claim the residual stayed identically dead. Residual final heads are
  nonzero by epoch 5. Five-epoch checkpoints do not identify first-step
  gradients or prove that optimization was useful.
- Do not claim calibration or NMS causality from score bins or post-NMS
  outputs. Score-conditioned TP rates are descriptive, and suppressed pre-NMS
  proposals are not observable.
- Do not convert no-training scaffold/all-query replays into newly trained arms
  or paper rows. Their completion explicitly records
  `diagnostic_only=true`, `training_performed=false`,
  `test_subset_used=false` and `paper_performance_row_allowed=false`.
- Do not claim efficiency, speedup, FLOPs, memory or energy. Complete
  synchronized feature-to-final-detection cost was never run after the G1
  accuracy kill.
- Do not revive the archived SparseHeadClean repository or hard K384 under a
  renamed selector. The canonical route is closed at G1 and its negative roots
  remain immutable.
- Any future official-quality dense proposal floor plus conditional residual
  compute is a new route. It is only `discussed` and requires a new
  preregistration, new name and explicit representation-equivalence gate before
  code or training.
- Preserve the three diagnostic deployment failures as engineering-only
  evidence:
  `diagnostic_deployment_inline_ssh_quoting_v1`,
  `diagnostic_deployment_nonlogin_module_function_v1`, and
  `diagnostic_deployment_profile_under_nounset_v1`. None is a model negative.
- Integrity status is `WARN`: internal scientific integrity passes, official
  paper comparability fails by design, and configured external cross-model
  reviewers were unavailable. Do not report an external review PASS.

## 2026-07-31 ODF-CR preregistration no-repeat rules

- ODF-CR is a separately named route. Do not relabel `official_identity` or
  `cheap_dense_scaffold`, reinterpret an old checkpoint, or overwrite DCSR G1
  semantics to obtain it.
- Do not reuse the already observed G1 holdout as the decision set. Holdout-v2
  must be selected only from the old train-160, contain exactly 40 videos, be
  disjoint from the old holdout-40, retain all classes on both sides and reject
  all test records.
- Do not reuse registered G1/prior official seeds. Frozen development seeds are
  `2026073101/2026073102/2026073103`; future official seeds must exclude all
  known development seeds.
- Three seeds on one holdout are training replicates, not three independent
  validation splits and not evidence of population-level generalization.
- Do not train K384 as a fifth arm or tune K after seeing results. Train only
  `d1_off/d1_all/d3_off/d3_all`; K384 is deterministic frozen replay with
  `stratified_uniform`, hash seed `2026073100` and exactly
  `min(384, valid_query_count)` residual queries per video.
- Do not start factorial training unless real-CUDA G0 proves bitwise
  zero-tolerance official-dense/`d3_off` equality and initialization contracts.
- Do not compare internal holdout-v2 absolute mAP with `63.xx`, released
  `66.833392` or official S0 `66.583013`. The matrix can select or kill the
  architecture, but cannot be a paper row or official benchmark claim.
- Do not silently rescue a negative residual-utility or K384-support gate by
  changing threshold, budget, selector, checkpoint, evaluator or seed. Record
  the legal negative result and analyze it before any new preregistration.
- Formal ODF-CR identity is exact commit/tree
  `01cdb78d2b7668098b6b13a1e49433d48fbc1a8d` /
  `e70d2956a197b1204e721239178e76152efe282b`, run root suffix
  `actionformer_odfcr_internal_20260731_v3`, array `1209259_[0-2]`, and G2
  Job `1209267`. Do not merge it with deployment v1/v2 roots or create another
  factorial array while this one is pending/running.
- `1209257` and `1209266` are `sbatch --test-only` scheduler estimates, not
  jobs. The only formal job IDs are `1209259` and `1209267`.
- The three real-CUDA G0 receipts already pass. Do not resubmit a standalone G0
  or treat another identity run as additional model evidence.
- Preserve `remote_profile_nonzero_under_errexit_v1`,
  `yaml_1_1_off_coercion_residual_support_v1`, and
  `aggregate_submit_gpu_count_missing_v1` as engineering-only events. The last
  was repaired by submitting only the missing G2 with explicit `--gpus=1`;
  the running main array was not and must not be duplicated.
- K384/G3 remain unauthorized until the completed G2 artifact has
  `residual_utility_gate_pass=true`. A false G2 is a legal negative result, not
  an engineering failure and not permission to tune the threshold.

## 2026-08-01 ODF-CR terminal no-repeat rules

- G2 is terminally false: mean `d3_all-d3_off=-0.1806 pp`, only `1/3` seeds
  positive, and @0.6 is `-2.7468 pp`. Do not submit K384/G3, tune K, change the
  gate, add seeds to rescue the same protocol or relabel a later run as this
  matrix.
- Do not call this an equivalence/non-inferiority result. With only three
  training seeds on one fixed holdout, the Avg interval is wide and not a
  population generalization interval.
- Do not say the residual branch was dead or that training failed. G0 proves
  exact zero initialization and paired identity; all jobs complete cleanly;
  `d3_all` has lower late training loss than `d3_off`.
- Do not infer calibration or NMS causality. Score-conditioned TP rates are
  descriptive, and only retained post-NMS outputs exist; suppressed proposals
  are unobservable.
- Do not infer gradient conflict, residual overshoot or optimizer failure as a
  fact. No residual norm, gate, activation, gradient norm or gradient-cosine
  telemetry was recorded.
- Do not generalize class or duration point estimates. SoccerPenalty and
  CricketShot lose while ThrowDiscus and TennisSwing gain; the longest duration
  bins contain only five and three GT instances.
- Do not reject conditional sparse routing in general. Only an all-valid
  residual on the depth-three floor was tested; K384/G3 never ran.
- Do preserve the positive internal floor-depth result: `d3_off-d1_off` is
  `+7.5600 pp` Avg and strongly positive at high IoU. It supports using an
  official-quality dense floor as a future design prerequisite, not an absolute
  benchmark, paper row or efficiency claim.
- Any low-LR, frozen-floor, calibrated, gated or new-holdout follow-up is a new
  preregistered experiment. It cannot be a silent repair of ODF-CR G2.
- Do not reactivate `sparsehead-official-matched-monitor`: terminal results,
  attribution and claim tracing are complete. Its configuration was archived
  recoverably only because self-deletion from the active heartbeat timed out.

### 2026-08-11 — DUCA P0 coordinate semantics are not tuning knobs

- Do not revive a two-generator uniform control, a 766 terminal endpoint at
  T=768/K=384, float linspace, banker rounding, tolerance repair, clipping, or
  deduplication. The accepted P0 contract has one integer-half-up canonical
  endpoint-inclusive generator and a bit-identical constant-density forward
  specialization.
- Do not retain or recreate selected-to-physical proposal conversion after NMS.
  Every non-sliding detector path must enter per-sample post-processing in
  `selected_q` and map exactly once to `physical_dense` before filtering, top-k,
  IoU, or unchanged NMS. Unknown or double coordinate state fails closed.
- Do not use P0 patch preparation as a pretext to run local/remote CPU checks,
  GPU/Slurm work, data traversal, metrics, validation/test, dynamic-K variants,
  Git push, or claims. Those require the later P1 admission decision from a
  fresh Pro turn.

### 2026-08-11 — accepted density-route v002 no-repeat rules

- Do not relabel a legacy slot allocator, actionness/boundary score, rank/top-k,
  quota, allocation, or soft-transport tensor as `duca_density_logits`. The
  accepted density route requires its own named per-time reader over dense
  `browser_memory` on the identity physical grid.
- Do not replace the named inverse-CDF decoder with raw top-k, a second global
  decoder, sort/clip/deduplicate/fill repair, or an unspecified tolerance-based
  constant shortcut. Exact constant logits must invoke the one canonical
  integer-half-up uniform generator; all nonconstant cases use the constrained
  density projection and fail closed on a violated contract.
- Do not map selected-q proposals after filtering, top-k, IoU, or NMS. The
  selected-q to physical-dense conversion is exactly once at the beginning of
  each per-sample `SingleStageDetector.post_processing` path.
- `PRO_P0_ROUTE_ADJUDICATION-v002` is a route specification, not experimental
  support. Do not start a command, test, data read, CPU/GPU/Slurm job, metric,
  push, result comparison, or paper claim until the later Pro-defined P1
  admission conditions are satisfied.

### 2026-08-12 — frozen nonconstant projection must not be weakened

- For a nonconstant serialized density target, do not replace the exact `Q=2^20` half-up target representation, exact DP/shortest-path search, or lexicographic `(E2,E_inf,E1,U1,position-vector)` comparison with host-language rounding, a weighted loss, tolerance, greedy rounding, local repair or stochastic/parallel tie behavior.
- Do not silently recover a projection failure with uniform positions, legacy selector output, clipping, deduplication, altered bounds or a second decoder. A nonfinite input, infeasibility, overflow, inconsistent comparator, certificate failure or reference mismatch must fail closed.
- Do not call property tests or same-process repeatability sufficient evidence. P0 requires independently implemented identity on matching serialized `(T,K,u,a)` inputs; that execution still needs a separate Pro authorization.

### 2026-08-25 — PJST first-mixing no-repeat and evidence rules

- Do not relabel the user-supplied PJST terminal text as a completed exact-DUCA Project browser receipt. The
  authoritative transport receipt was quarantined; the text is an independent scientific review only.
- Do not call sorted bootstrap entries 500/9500 a two-sided 95% interval for 10,000 samples. Freeze
  2.5%/97.5% quantiles, the index convention and interpolation rule before looking at outcomes.
- Do not describe support-weighted pair means as strict temporal integration or as removing all gap channels.
  The preferred first gate keeps the ordinary pair mean and changes only the physical derivative scale;
  support metadata remains audit-only until an isolated later ablation is justified.
- Do not claim identical selected RGB in an end-to-end PJST run whose gradients can change the selector.
  Freeze/replay selection for representation attribution, or report selector drift as mediation in a distinct
  total-effect experiment; never mix the two claims.
- Do not repeat dense/uniform/random, RankPack/TrueTime training, SingleClock retraining, H65 60-epoch
  compression, Query/Bridge, dynamic-K or continuous-cliplet matrices to answer the PJST first-mixing question.
- The authoritative fresh exact-DUCA v002 decision freezes derivative-only PJST-D1 and fixed-selector
  representation attribution. Do not reopen support-aware versus derivative-only or total-effect versus
  representation-effect discussion during implementation.
- Do not call a frozen contract or a passing focused test a PJST result. PRE_RUN requires an independent
  Critic PASS, and efficacy requires the matched frozen-selector OFF/ON formal experiment.
- Do not revive commits `877d893f` or `84325205` for PRE_RUN. Their correction loop is closed after the
  second equivalent integration defect: wrong clip-pair shape, unchunked checkpoint metadata and missing
  matched OFF instantiation. No third patch or recheck belongs to this implementation package.

### 2026-08-29 — native-tubelet attribution and dynamic-budget boundary

- Do not describe the current fixed `K=384` native-tubelet experiment as the final DUCA method or as
  dynamic budgeting. It is an attribution gate for selection location, coverage/redundancy, low-resolution
  context recycling and physical-time reconstruction.
- Do not repeat dense, random, arbitrary-frame, continuous-cliplet, H65 compression, UVT, Query-Bridge or
  PJST matrices to answer this gate. The matched comparison is native-tubelet uniform versus task-state
  coreset under the same 192-tubelet heavy input and downstream contract.
- A successful or non-killing fixed-budget result must lead to a true dynamic-budget phase. The dynamic
  candidate must change actually executed heavy clips and match or reduce mean realized VideoMAE compute;
  padding to the maximum followed by nominal K metadata is not an acceptable implementation.
- PRE_RUN Jobs `1260182/1260183` completed and unlocked the only formal pair, uniform `1260184` and coreset
  `1260185`, at exact commit `b33391126eac05e3353d322b973dda91741f0732`. Both arms completed 60 epochs,
  wrote epoch-59 checkpoints and completed official 211-video evaluation; do not duplicate or retrain them.
- The terminal log point estimates are uniform/coreset Avg-mAP `64.13%/62.81%` and mAP@0.7
  `42.45%/40.56%`. Preserve this diagnostic negative result; do not hide it as an engineering failure or
  promote it to a population-level conclusion.
- Both jobs failed only after metric computation because predictions were not saved, so the structured
  metric JSON, paired interval and measured-cost artifact are absent. Do not automatically rerun evaluation,
  infer cost, or enter dynamic-budget implementation before the neutral result returns to Pro.

### 2026-08-29 — window-level variable-compute successor no-repeat rules

- Pro has returned `PIVOT`. Do not tune or retrain the rejected fine-grained task-state coreset, and do not
  repeat the fixed uniform control. Reuse Job `1260184` epoch-59 EMA as the sole fixed 24-clip control.
- The successor changes allocation across windows, not positions within a fixed budget. Within each 16/20/24
  budget use deterministic uniform native-tubelet selection; do not reintroduce the rejected coreset score.
- Real variable compute requires separate 16/20/24 heavy-backbone executions. Padding every row to 24 clips,
  copying a requested budget into metadata or estimating cost from nominal K is inadmissible.
- Preserve the fixed detector grid and physical-time reconstruction. Do not change VideoMAE-S, Adapter,
  ActionFormer, loss, NMS, evaluator, split, seed, update count or checkpoint rule in this attribution step.
- Historical commit `36d75c146492a38eb8966c66ff6b2881938cf3c6` may supply the engineering pattern for K-bucketed
  backbone execution, original-order restoration and realized-work reporting. Its budget formula,
  selected-axis loader and scientific narrative are not part of the new method.

### 2026-08-29 — Coverage-v1 isolation and no-repeat rules

- Do not add boundary gradients, learned feature similarity, feature merging, temporal attention bias or
  dynamic K to the Coverage-v1 attribution. The only intended difference from matched H65 is Top-K versus
  deterministic temporal facility-location allocation at the same `K=384`.
- Do not call vectorized code, passing tests, improved unlabeled coverage statistics or a clean deployment a
  performance result. Scientific evidence requires matched full training, official predictions and the
  frozen paired whole-video analysis.
- Do not reuse the historical 65.13 point alone as the formal control when its predictions and exact runtime
  identity are not sealed for this comparison. Train the matched H65 control under the same submission.
- Do not respond to `AssocMaxSubmitJobLimit` by changing account/QOS, cancelling other projects' jobs,
  submitting outside Slurm or shrinking the experiment. Wait for capacity and submit the frozen PRE_RUN.

### 2026-08-30 — Coverage-v1 failed intervention gate: do not bypass or retune silently

- PRE_RUN Job `1261679` validly tested 200 unlabeled training samples and stopped before training. Do not call
  this an mAP failure, but do not launch the two 60-epoch arms: the frozen candidate failed its own set-change,
  anchor-coverage and max-gap criteria, with max-gap p95 worsening from `2` to `8`.
- Do not lower the preregistered thresholds, tune `M`, `sigma` or `K/M`, add hard gap repair, or reinterpret the
  same output as a pass. Each would change the frozen mechanism or decision rule and requires Pro.
- Do not continue calling the matched control “Top-K” without qualification. Current code uses H65
  budget-calibrated systematic sampling. Also distinguish a frozen scout/priority path from a trainable but
  matched VideoMAE-S/Adapter/ActionFormer backend.

### 2026-08-30 — DUCA-Marginal-v1 no-repeat and leakage boundaries

- Pro has returned `PIVOT`. Do not train Coverage-v1, retune its anchors, or revive PJST-D1, continuous cliplets,
  UVT, Query-Bridge, dense retraining, feature merging or new time encoding inside this task.
- Do not build a new dynamic-budget framework. Reuse the clean H65 `04c35a3b...` marginal controller,
  counterfactual utility, budget-calibrated sampling and acquisition surfaces; make only the changes needed for
  signed K256/K384/K512 marginal targets, nested prefixes, exact video-budget allocation and real K-bucket execution.
- K384 selected indices and predictions must reproduce the frozen H65 terminal EMA. K256 and K512 must be nested
  prefixes/supersets of the same H65 priority sequence; packet size 16 means non-contiguous observations, not a
  continuous 16-frame clip.
- Oracle and detector-loss labels are training-side diagnostics only. Fit the utility head on 160 training-side
  videos, hold out 40 from utility fitting, and never expose official test GT, oracle utility, teacher predictions
  or raw-prediction caches to deployment allocation.
- Do not run official test unless implementation parity, oracle headroom and predictability gates all pass. Do not
  treat an ambiguous threshold region as a pass, change the gates after seeing results, or launch full 60-epoch
  task-adapted training before this frozen-detector experiment returns to Pro.

### 2026-08-31 — Marginal short-window and code-identity boundaries

- Do not restore `384 * window_count` as the per-video target. The frozen target is
  `sum_i min(V_i,384)` over every window, including short windows.
- Do not execute a collapsed K256/K512 request a second time. If its actual observation count equals K384 actual
  cost, it is the K384 arm for loss, prediction, positions and execution. Do not count the requested label as an
  effective budget change.
- Do not pad a distinct nonbaseline arm to 384 or 512. It executes only the next 16-observation packet boundary;
  historical K384 is the sole 384-slot padding exception.
- Do not discuss or deploy an unpushed local candidate. Every Pro prompt must name the latest GitHub repository,
  branch, exact commit and key-file URLs. At this historical PRE_RUN stage the authoritative implementation was
  `feature/duca-marginal-budget-v1-20260830@f87555f7da362fe1a20d4ca08f7a68c975ed8280`.
- Job `1262073` is a zero-model-execution shell-launch failure and Job `1262075` is a focused-test contract failure.
  Do not retry either. Job `1262076` is the unique successful PRE_RUN on the corrected clean commit. Job `1262077` is
  its sole conditional successor and must not be duplicated manually. `PRE_RUN_PASS` establishes implementation
  eligibility only; it is not utility-head, headroom, mAP, significance or efficiency evidence.

### 2026-08-31 — Marginal gray-zone no-repeat boundary

- Do not rerun `select-k384`, `counterfactual-k256` or `counterfactual-k512`. Their immutable artifacts were produced
  once under `f87555f7...`; Job `1262098` reused them only after a summary-format fix and current-commit PRE_RUN.
- Do not hide the split provenance. The latest summary implementation is
  `feature/duca-marginal-budget-v1-20260830@f67d96fdf68a295eaa7f678f3dfc125530828889`, while the three producer
  artifacts retain `f87555f7...`. The only production-code delta is the evaluator block-list serialization adapter;
  config, checkpoint, data and pretrain hashes are unchanged.
- Do not call `+0.725589/+0.729004` percentage points a pass or a failure. It lies between the frozen strong-headroom
  and no-headroom boundaries. Do not train the utility head, run official test, invent a K320 balancing rule, lower
  thresholds or revive another route until Pro returns a new scientific decision.
- Every Pro request must include the repository, branch, exact latest commit and key-file GitHub URLs, plus the raw
  producer/summary provenance. A local path, old commit, branch name alone or prose description is insufficient.

### 2026-08-31 — Cap-release falsifier no-repeat boundary

- Pro has frozen exactly one follow-up: reuse the existing sealed K256/K384/K512 artifacts and change only
  `max_changed_fraction` from `0.5` to `1.0`. Do not rerun PRE_RUN, selection or counterfactual producers; do not add
  K320 or any other tier; do not train the utility head; do not access official test.
- The current implementation identity is
  `feature/duca-marginal-cap-release-falsifier-v1-20260831@d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`.
  Job `1262117` is the unique Evaluator. Do not duplicate, restart, change parameters or interpret it before terminal.
- If either cap-release point gain misses the original `+0.8/+1.0` gate, stop Marginal-v1 without bootstrap. If both
  pass, run only the frozen 10,000-replicate paired whole-video interval and return the result to Pro; predictor work
  is still not authorized automatically.

### 2026-08-31 — Marginal-v1 cap-release terminal no-repeat boundary

- Job `1262117` completed the unique cap-release diagnostic. The released result was only
  `+0.427310/+0.450280` percentage points over fixed K384, below both `+0.8/+1.0` gates, and worse than the capped
  oracle. Do not rerun it, bootstrap it, train the utility head, access official test, or tune the allocation after
  seeing this result.
- The preregistered consequence is to stop the current Marginal-v1 mechanism. Do not silently generalize this to all
  dynamic-budget methods, and do not revive an older route or invent a successor before Pro independently decides.
- Every result-return Pro request must include the repository, branch, exact commit and permanent GitHub links for
  the runner, allocator and focused test at `d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`, together with the raw terminal
  result path and immutable producer/summary provenance.

### 2026-08-31 — Joint-mAP neighborhood task boundary

- Pro has stopped the current additive Marginal-v1 mechanism. Do not resume cap sweeps, tier searches, utility-head
  training, official test, bootstrap or a new allocator under that mechanism.
- The only authorized analysis is the capped-to-released difference neighborhood. It must derive balanced states
  from actual per-window costs, enumerate the current 96 unique joint states, and never hard-code an arbitrary pairing
  for `video_validation_0000419`.
- Modify only `tools/bata/run_duca_marginal_frozen_h65_probe.py` and
  `tests/test_duca_marginal_budget.py`; keep `opentad/models/duca/dynamic_budget.py` byte-for-byte unchanged. Do not
  add a generic search framework, model class, configuration family or provenance system.
- This is same-holdout metric-oracle diagnosis. A best state is not deployable, confirmatory or paper-ready. Do not
  bootstrap after selecting among 96 development states, and do not promote the result beyond mechanism diagnosis.
- The next Pro return must again include the latest pushed repository, branch, exact commit and permanent runner,
  allocator and test URLs; local-only implementation is never sufficient.

### 2026-08-31 — Joint-mAP neighborhood terminal no-repeat boundary

- The sole 96-state diagnosis completed on
  `feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831@46812facc8773d9b4a9c21833cbe397c8aaa5a2d`.
  Do not rerun, extend or tune this neighborhood; do not add states, tiers, pairings or an after-selection bootstrap.
- No state passed the frozen joint gate, and no minimal legal transfer improved both Avg-mAP and mAP@0.7. Stop using
  video-level joint utility to repair this capped-to-released difference neighborhood. Do not train a predictor from
  the best development state or present that state as deployable or confirmatory.
- This does not reject every dynamic-computation mechanism. Do not revive an older route or choose a replacement from
  this diagnostic. Return the complete neutral evidence to Pro and wait for its independent scientific decision.
- Every such Pro return must include the latest repository, branch, exact commit and permanent GitHub URLs for the
  runner, unchanged allocator and focused test, plus the terminal JSON path and SHA-256.

### 2026-08-31 — Pro STOP boundary for additive Marginal-v1

- Pro has issued the terminal `STOP` for the existing Marginal-v1. Do not modify or rerun the frozen branch, revisit
  cap values, change the `+0.8/+1.0` gate, choose a different 96-state compromise, alter pairings or tie-breaks, train
  the utility head, add a bootstrap after state selection, or access official test.
- Treat `feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831@46812facc8773d9b4a9c21833cbe397c8aaa5a2d`
  as read-only negative evidence. No Builder, Critic, PRE_RUN, Evaluator or compute job remains for this mechanism.
- In paper text, say that window-level additive counterfactual detector loss was insufficient to rank video-level joint
  detection utility in the tested neighborhood. Do not use the internal label `single-item misranking primary`, and do
  not claim that `interaction_witness_count=0` rules out all Soft-NMS or window interactions.
- Do not broaden this terminal result to all dynamic budgets, the H65 priority sequence or the three budget values.
  Equally, do not infer an untested successor. Future dynamic-computation work requires a fresh Pro scientific
  hypothesis and a separately frozen task; it cannot be presented as a recovery or small revision of Marginal-v1.

### 2026-08-31 — Whole-video consistent-budget final-falsifier boundary

- Project-level Pro has authorized exactly one new mechanism test: a single ordered donor-recipient transfer where every
  donor window requests K256, every recipient window requests K512 and every other window requests K384. Do not call or
  modify the stopped Marginal allocator, mix tiers within a changed video, or present this as Marginal-v1 recovery.
- Candidate enumeration must finish before GT or metric access. Keep only candidates with real observation cost
  `<=47110`; both changed videos need an actual non-baseline window. Never use requested K, execution slots or padding as
  the cost.
- Reuse sealed predictions and the identical NMS/evaluator. Do not run detector/Scout forward, training, gradient,
  bootstrap or official test. Do not repeat dense, uniform, coreset, Coverage, PJST or old 96-state analyses.
- The practical gate remains simultaneous `+0.8 pp` Avg-mAP and `+1.0 pp` mAP@0.7. If no legal candidate passes, stop
  DUCA project-level method innovation within this frozen task/data/action-space/resource boundary. Do not add a third
  video, combine multiple transfers, change tiers, lower the gate, select-and-bootstrap or train a controller.
- The implementation must be pushed before review. Every Pro return must include repository, actual remote branch, exact
  commit and permanent runner/test plus unchanged allocator URLs.

### 2026-08-31 — Whole-video implementation identity

- Do not discuss or evaluate a local draft. The only current implementation is
  `feature/duca-whole-video-consistent-budget-falsifier-v1-20260831@33e4ed137c33eef07f0452b44506a6993bdf7535`,
  already pushed to <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702> and independently reviewed PASS.
- PRE_RUN Job `1262147` exposed only a deterministic proposal-row-order replay defect and is superseded. Do not rerun it
  or interpret it scientifically. Corrected PRE_RUN Job `1262161` passed the frozen identity, candidate-generation, cost
  and anchor-reproduction checks. The only formal experiment is Evaluator Job `1262162`; do not duplicate, tune or
  interpret it before its terminal artifact is available.
- Any later Pro prompt must cite the latest already-pushed repository, actual branch, exact commit and permanent runner,
  focused-test and unchanged-allocator URLs. Do not cite `c27d77...`, a local or unpushed revision, or only the repository
  homepage.

### 2026-08-31 — Whole-video evaluator node-failure recovery boundary

- Job `1262162` is terminal `NODE_FAIL`, not a method result. Node `g0022` went down after `500/704` candidates; no runner
  failure receipt or terminal `whole_video_consistent_budget_result.json` exists. Do not infer passing candidates, a gate
  result or project-level STOP from its partial stdout.
- Job `1262190` is the one exact same-task infrastructure recovery. It uses the unchanged `33e4ed...` clean snapshot,
  submission script, 704-candidate manifest, sealed predictions, evaluator, cost limit and `+0.8/+1.0` gate. Do not submit a
  third evaluator, change resources for scientific advantage, resume from partial candidate output or merge partial runs.

### 2026-08-31 — Whole-video falsifier terminal negative boundary

- Job `1262190` completed `704/704` legal candidates and produced the only terminal result. Do not rerun Job `1262162`,
  submit a third evaluator, merge its partial output, or recompute the same candidate set. The authoritative JSON is
  `/data/run01/sczc063/yuzibo/duca_whole_video_result_33e4ed13_20260831/whole_video_consistent_budget_result.json`,
  SHA-256 `40686fa73114eedfa14b3d34a01717aacb0b93f629f5a1e7f2ee27de300ad19c`.
- Zero of 704 legal states passed both `+0.8 pp` Avg-mAP and `+1.0 pp` mAP@0.7. Do not select the best Avg-only or
  high-tIoU-only state as a compromise, lower either threshold, add a third video, combine transfers, change K256/K384/K512,
  train a controller, add post-selection bootstrap or consume official test.
- Preserve the scope: this rules out sufficient headroom for one whole-video donor-recipient transfer in the current
  training-side controller holdout, H65 priority sequence, three-tier action space and resource boundary. It does not prove
  that every dynamic-computation mechanism, broader budget space or low-cost Scout is ineffective.
- The then-required scientific action was one fresh Pro adjudication with neutral evidence and the current public GitHub
  repository, actual branch, exact `33e4ed...` commit, runner, test and unchanged allocator URLs. That adjudication is now
  complete below; Codex did not revive an older route or propose a successor beforehand.

### 2026-08-31 — Pro terminal archive boundary for three-tier observation transfer

- Pro has now completed that adjudication with the latest public
  `feature/duca-whole-video-consistent-budget-falsifier-v1-20260831@33e4ed137c33eef07f0452b44506a6993bdf7535`
  implementation and issued `STOP`.
- Treat Marginal-v1, cap-release, the 96-state neighborhood and whole-video donor-recipient branches as read-only negative
  evidence. Do not rerun them, enlarge the development oracle, add transfers, change K tiers or thresholds, train a
  controller, bootstrap selected states, or use official validation/test for this route.
- Do not overgeneralize: this closes only the present THUMOS14 training-side holdout, frozen H65 detector/priority sequence,
  sealed K256/K384/K512 prediction and real-observation transfer boundary. It does not establish that every dynamic compute
  mechanism, Scout, budget space, budget-conditioned model, token/layer action or dataset fails.
- Do not auto-revive ChronoTransport, UVT, Fovea, Query-Bridge or another historical branch. No successor is authorized.
  Reopening requires a new mechanism outside this boundary with independent, preregistered matched-compute oracle headroom.
- Every future Pro discussion about a genuinely new mechanism must again include the latest pushed GitHub repository,
  actual branch, exact commit and permanent key-file URLs. Local-only or stale code descriptions are insufficient.

### 2026-08-31 — Multi-budget detector-adaptation isolation boundary

- The new `REVISE` does not reopen Marginal-v1 or the sealed 704-state oracle. Do not rerun, enlarge or reinterpret those
  states; they remain the negative evidence for a detector frozen at K384.
- In the first new experiment, preserve the existing nested K256/K384/K512 positions. Do not combine multi-budget detector
  training with budget-native per-K selection, a new selector, budget embedding, distillation, Gumbel-Softmax, a new Scout or
  boundary/classification head, DFT, Mamba, Block Drop, CUDA kernels or TensorRT.
- Do not call cross-budget mismatch a proven root cause. It is the tested hypothesis. Do not call the previous 704-candidate
  failure controller overfitting or a differentiability failure; no controller was trained in that falsifier.
- Do not start code, PRE_RUN or training until Pro freezes one matched update schedule and an exact train-side development
  video list that did not participate in parameter learning or rule selection. Do not silently reuse the old 40-video oracle
  split or import `30+60` from the superseded attachment.
- If the new matched-cost oracle fails the frozen `+0.8/+1.0` joint gate, stop the current K256/K384/K512 transfer route. If
  Avg-mAP recovers but high-tIoU does not, only a later Pro decision may open a separate conditioning experiment.

### 2026-08-31 — Full-data formal-comparison boundary

- Do not promote a 40-video holdout, a 160/40 development split, smoke run, pilot or shortened training to the formal comparison.
  After design freeze, both matched arms must be trained on the complete frozen training split.
- Do not access the complete official held-out evaluation split for training, checkpoint/threshold/rule selection, route choice
  or iterative debugging. Use it only for the preregistered final comparison, with the same evaluator, annotation semantics,
  class mapping and Soft-NMS for both arms.
- Do not silently treat the OpenTAD/DUCA 211-video `validation` convention as identical to the ActionFormer 212-video `test`
  convention. Pro must freeze the exact training/evaluation subset names and complete video-ID lists before Builder starts.
- The already-running Pro turn predates this rule. Do not append a follow-up, interrupt it or resubmit the prompt. If its terminal
  plan conflicts with full training or does not identify the complete held-out protocol, return that conflict in a new Pro turn.

### 2026-08-31 — Do not implement the superseded 160/40 Pro data protocol

- Pro v001's scientific `CONTINUE`, nested-budget isolation and 6,000-update recommendation are valid inputs to the next decision,
  but its 160-video train / 40-video development split is not the human-required formal comparison.
- Do not create the Builder branch, launch PRE_RUN, or report the 40-video development gate as paper evidence from that report.
- Do not silently replace 160/40 with a full-data split while keeping the rest of the report. The exact full training and held-out
  identities, 211/212 resolution and diagnostic-to-formal sequence require a new independent Pro adjudication.

### 2026-08-31 — Full-data Pro protocol: identity audit before any model work

- The new verified Pro decision has revoked the 160/40 formal split and all labeled training-side mAP/oracle gates. Do not revive
  them as a pilot, gate, checkpoint selector or paper comparison.
- Do not choose 211 or 212 by convention. Before model implementation, materialize annotation, loader and physical ID sets for
  `training` and `validation`, locate literal historical 211 IDs and a source-backed ActionFormer 212 source, and report every
  set difference. Missing media, decode failure, silent loader filtering or an unsupported exclusion is a blocker, not a reason
  to drop a video.
- The current branch may contain only the minimal read-only identity-audit tool and its focused test on base `04c35a3b...`.
  Until independent Critic and CPU Evaluator evidence returns to Pro, do not load checkpoints, create the multi-budget model,
  submit PRE_RUN/GPU/training, generate held-out predictions, read held-out temporal labels or calculate mAP.
- Do not replace the frozen task with progressive unfreezing, STE temperature annealing, K128/K192/K320 curves, ActivityNet,
  budget embeddings, distillation, Gumbel, a new Scout/head/selector, Mamba, Block Drop or deployment work. A missing local
  `research_project_analysis.md` and its pasted summary are not scientific authorization.

### 2026-08-31 — Irregular-time proposal is a hypothesis source, not an executable bundle

- Do not repeat the external proposal's four diagnoses as proven root causes. Non-contiguous tubelet pairing, rank-only time
  encoding, linear reconstruction and end-to-end gradient instability must remain separate, falsifiable explanations.
- Do not combine native tubelets, a 144/48 dual-stream selector, continuous-time rotary position encoding, Gaussian temporal
  splatting, Gumbel annealing and H65 distillation in one attribution experiment. A positive combined result would not identify
  which mechanism caused it.
- Do not quote the proposal's +0.8--1.5 pp, >=64.5, “50% end-to-end compute” or Dense-matching statements as evidence. They are
  untested targets; only the 50% reduction in heavy-input observation count is currently supported without hardware profiling.
- Do not call H65 simple Top-K. Its control identity is priority-modulated budget-calibrated systematic sampling with a uniform
  coverage floor. Do not replace it silently with the proposal's 144/48 allocation.
- Do not use the complete held-out evaluation labels to choose time encoding, reconstruction kernels, sampling allocation,
  curriculum, thresholds or routes. The current 211/212 identity audit and subsequent Pro admission remain ahead of all model work.
- Native tubelet acquisition is not a blank slate. Do not repeat the `b3339112...` fixed-K384 uniform/task-state coreset pair,
  the continuous 16-frame cliplet gate, or the FZ_CONTIG/JT_CONTIG bundles as if they isolated pairing distortion. A future valid
  test must hold the selected RGB set and every downstream surface fixed and change only within-tubelet physical continuity.
- Physical-time modeling has already produced full-training negative and positive results. Do not repeat a generic selected-axis
  versus physical-time comparison or cite it as untested; the closest matched RankPack/TrueTime result was +0.6208 Avg-mAP but
  remains single-seed and lacks paired-bootstrap closure. A new encoding must be explicitly distinguished from TrueTime.
- Hidden-linear sparse-to-dense reconstruction already has implementation and CUDA-gate evidence. Do not spend another cycle
  rebuilding that bridge. No formal nearest/linear/Gaussian kernel comparison with boundary metrics exists; only that isolated
  comparison remains scientifically open.
- Do not repeat the 30-uniform+60-joint curriculum as a fair curriculum result: it used 90 total epochs against a 60-epoch control.
  Do not rerun the completed homotopy or CellCF utility-distillation arms unchanged; all underperformed their relevant controls.
  A future curriculum/distillation claim requires identical successful-update budgets and a no-distillation matched arm.

### 2026-08-31 — Comprehensive Pro review: one data task, one conditional terminal experiment

- Do not interpret the GitHub/Wiki review as permission to begin multi-budget model implementation. The only current action is the
  full-data identity audit; it must return through an independent Critic and N16R4 CPU Evaluator to Pro.
- Do not choose 211 or 212 by count, convention or repository reputation. Every annotation/physical/loader/evaluator/prediction
  difference must have a source-backed explanation; unexplained differences block the route.
- After data admission, do not add budget embedding, distillation, Gumbel selection, a new Scout, TrueTime, reconstruction kernels,
  Mamba, Block Drop, TensorRT or a third arm to the first experiment. Its only variable is K384-only versus K256/K384/K512
  training exposure under the same H65 nested positions and trainable parameter set.
- If that complete two-arm, three-seed experiment fails its frozen mixed-workload gain, K384 safety, uncertainty or real-cost gates,
  stop the H65 K256/K384/K512 dynamic-budget mainline. Do not use another mechanism as a recovery loop for that result.

### 2026-08-31 — H65 system multi-budget PRE_RUN identity

- The only implementation is
  `feature/duca-h65-system-multibudget-exposure-v1-20260831@0d67d49c2fc4a5f50aa784f7809c0dd936492109`,
  with sole parent H65 `04c35a3b...`; a fresh independent Critic returned `PASS`.
- The sole PRE_RUN is Job `1262690` against clean snapshot
  `/data/run01/sczc063/yuzibo/duca_h65_multibudget_0d67d49c_20260831` and output root
  `/data/run01/sczc063/yuzibo/duca_h65_multibudget_prerun_0d67d49c_20260831`. Do not duplicate, restart, alter or interpret it
  before terminal.
- PRE_RUN success authorizes only the already frozen six complete training units. It is not model, mAP, interval, cost-saving or
  paper evidence; PRE_RUN failure must be classified before any correction or training submission.

### 2026-08-31 — Admitted identity evidence must use OpenTAD 200/211 literally

- Do not reopen the 211/212 question by count or naming convention. The clean `fdd2bcdd...` audit has materialized the literal
  sets: OpenTAD training annotation/loader/physical are identical 200; OpenTAD held-out annotation/loader/physical/evaluator and
  historical prediction IDs are identical 211; train-held-out intersection and decode failures are empty.
- Do not merge ActionFormer's 212-video `Test` set with OpenTAD's 211-video `validation`. Their only literal difference is
  `video_test_0000270`, which OpenTAD source line 11 excludes for wrong annotations. `video_test_0001292` is not an ActionFormer
  annotation video and must not be added to either evaluation set merely because it exists in physical/feature storage.
- Do not treat the first lowercase `test` invocation as a scientific or data failure. ActionFormer subset literals are
  case-sensitive `Test`/`Validation`; the preserved corrected CPU evaluation, with unchanged code and data, is the effective
  report and has SHA-256 `d7251c...`.
- `DATA_IDENTITY_PASS_211` is evidence for Pro admission, not self-executing permission. Until Pro signs the data boundary, do not
  create the multi-budget model branch, load Stage-1, run PRE_RUN/GPU/training, generate held-out predictions or compute mAP.
- Do not silently choose the later seed schedule. The comprehensive Wiki review says all three seeds; the newer route-integration
  report says seed 3407 first and 3408/3409 only after it passes. Return that conflict with the data evidence and follow Pro's
  explicit resolution.

### 2026-08-31 — Data admission and three-seed blindness are now settled

- Pro has admitted the literal 200-video training and 211-video OpenTAD held-out identities. Do not reopen 211/212, add physical
  extras, or run a parallel ActionFormer-212 comparison.
- The seed conflict is closed: execute `3407`, `3408`, and `3409` in that order, but do not read any held-out metric between seeds.
  Do not turn seed 3407 into a route gate or selectively omit later seeds.
- Do not call the current task detector-only adaptation. Its estimand is the total effect of changing Stage-2 budget exposure under
  the unchanged H65 Stage-2 trainable mask.
- Do not add a third arm, new selector/controller, budget embedding, distillation, Gumbel, Mamba, Block Drop, DFT, TensorRT,
  detector wrapper or another route. Do not resurrect the frozen-detector oracle or Coverage-v1.
- Do not open held-out labels or aggregate metrics until all six training units and all Control-K384, Candidate-K384 and
  Candidate-mixed predictions are sealed. After the one held-out opening, no code, method, checkpoint, probability, manifest,
  threshold or prediction regeneration is allowed.

### 2026-08-31 — Gemini post-admission advice does not change scientific authority

- Do not convert Gemini's proposed `0.1%` cost tolerance, exact file list, controller ablations or spatial-cropping fallback into
  Builder gates or follow-up tasks. They are external suggestions, not part of Pro's frozen task order.
- Do not claim that a PASS proves cross-budget representation mismatch was the unique root cause. It would support the total effect
  of H65 system-level multi-budget exposure under the frozen comparison.
- Do not claim that a FAIL proves an intrinsic temporal-discontinuity bottleneck or independently authorizes a project-wide pivot.
  It closes the current H65 K256/K384/K512 exposure route under its frozen protocol; Pro must decide any broader interpretation.

### 2026-08-31 — Do not fork or embellish the implemented multi-budget experiment

- The only implementation identity is `0d67d49c...` on
  `feature/duca-h65-system-multibudget-exposure-v1-20260831`. Do not create a parallel selector, detector wrapper, evaluation script
  or orchestration layer while this exact revision is under Critic review and formal execution.
- Do not reinterpret the deterministic `1454/3000/1546` budget occurrence schedule as a tunable hyperparameter sweep. Its calibrated
  probabilities and fixed mixed held-out manifest are frozen before held-out labels are opened.
- Do not use the label-free 200/211 preparation pass or the `25 passed` focused suite as model evidence. They establish executable
  identity only; mAP, paired intervals and cost benefit remain unknown.
- Do not read or report any one seed's held-out metric between training units. The one-time evaluator is the only authorized label
  opening after all nine prediction/cost seals exist.
- Do not call model-only timing end-to-end. The implemented cost record explicitly includes realized data-consumer wait and final
  per-video Soft-NMS, reports full-population wall time, and leaves framework overhead visible rather than attributing it to a model
  component.

### 2026-08-31 — Short-window PRE_RUN recovery is closed to one metadata correction

- Do not retry or reinterpret failed Job `1262690`. It reached no successful optimizer update and only exposed inactive `-1` padding
  in the short-window true-time metadata; it is not a model, optimization or performance result.
- The only recovery identity is `409f370a7ed14e7077bc87138196ab6abe459f99`, whose parent is `0d67d49c...`. Do not add a
  padding policy, clamp positions, change `valid_len`, alter the detector mask, or modify acquisition/model/config/data/statistics.
- Job `1262693` is the sole corrected PRE_RUN. Do not duplicate it or submit any complete training arm until it passes the frozen
  four-update checkpoint/probe contract.

### 2026-08-31 — Complete multi-budget training DAG is immutable

- Corrected PRE_RUN `1262693` passed; do not rerun it. The only full-training units are `1262696/1262697` (seed 3407),
  `1262698/1262699` (seed 3408) and `1262700/1262701` (seed 3409).
- Do not remove or weaken the `afterok` seed order, submit duplicates, inspect held-out metrics, select a seed, change the update
  budget, or regenerate calibration while these jobs run.
- A training terminal is an infrastructure/model-execution fact, not a held-out result. Generate predictions only after all six jobs
  complete successfully and terminal epoch-59 EMA identities are validated.

### 2026-08-31 — Do not revive the failed legacy-binder DAG

- Jobs `1262696/1262697` failed before training because this experiment was incorrectly routed through an unrelated legacy P0
  binder; Jobs `1262698`–`1262701` were cancelled without starting. Do not retry, resume, reinterpret or cite them as model evidence.
- The sole current implementation identity is `2b3b3243066a89e5a4be5acdb178c318fbeceac0`. Do not assign a false P0 variant, restore
  legacy binder artifacts, weaken the 6,000-successful-update audit, or add new contract machinery.
- PRE_RUN `1262715` passed and must not be duplicated. The only active full-training chain is `1262719/1262720` →
  `1262721/1262722` → `1262723/1262724`; preserve its strict seed order and blind held-out boundary.

### 2026-09-01 — Do not reuse the g0030-aborted job identities

- Jobs `1262719/1262720` never entered the batch script, and Jobs `1262721`–`1262724` were cancelled unstarted. Do not resume,
  retry, cite or reinterpret these six job IDs as training or scientific evidence.
- The only transport recovery is the same commit/calibration with `g0030` excluded. The active chain is `1262743/1262744` →
  `1262745/1262746` → `1262747/1262748`; do not create a parallel DAG or alter the frozen experiment.
