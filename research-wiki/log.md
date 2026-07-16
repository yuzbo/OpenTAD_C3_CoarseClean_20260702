---
type: wiki_log
append_only: true
---

# Research Wiki Log

- 2026-07-11：初始化 C3/DUCA research-wiki。
- 2026-07-11：逐轮读取主任务 191 轮，归档 158 条用户侧原始消息。
- 2026-07-11：登记实现代理、论文代理和早期目标任务的近期记录。
- 2026-07-11：登记 C3、PAction、GAS-VT、lattice、detector-aware、TrueTime、
  DUCA、MUST、X3D/SlowFast、physical-grid、CFPA、CVCR、ChronoTransport、
  PhysTime 路线。
- 2026-07-11：冻结当前裁决：70aa069 是待裁决 DUCA baseline，a5e1774 是最新
  审计代码；正式论文 claim 尚未闭环。
- 2026-07-11：wiki lint 通过：16 ideas、7 experiments、10 claims、47 edges、
  0 orphan nodes、0 curated broken links；query pack 2825 chars。
- 2026-07-11：纠正 ChronoTransport 过期状态：`92029ea` formal Stage-B P3 science gate 为负，Stage C/P5 未解锁；新增独立 negative experiment 节点，路线暂停。
- 2026-07-12：为无法读取本地工作区的 Pro reviewer 建立 GitHub 固定提交审查入口；仅同步
  ChronoTransport r1 规格、实现表面、原 Pro 记录、两轮独立复核与本地源码审计，明确排除
  数据、checkpoint、GPU 日志和新行为结果。审查仍止于 `REVISE_SPEC_BEFORE_PLAN`，不得借
  GitHub 同步越过到实现、profiling、Gate 1、新 seed 或 Stage C。
- 2026-07-12：完整归档 GitHub-visible Pro 终审，附件 SHA-256
  `07A5B4B519E64A39D7F84CE862F0E56117BFF2DB62206B6AE24BDD66768B19FE`，裁决
  `REVISE_SPEC_BEFORE_PLAN` / `GO_TO_SPEC_REVISION_ONLY`。本地独立复算确认 exposure、
  conformal rank、coverage 下界和 Stage-C 525/candidate；官方上游确认 GT-aware
  `random_trunc`、edge padding 与 all-time adapter。不同意原样照抄三处未闭合文本：split
  digest 字节协议、Gate-3 coverage margin 和 Gate-4 seed-level mAP bootstrap 必须在 r2
  中精确定义。当前状态为 `spec_revision_in_progress`，尚未实现或部署。
- 2026-07-12：完成 ChronoTransport r2 单文件书面规格 commit `d825520`，853 行，
  committed-blob 与 worktree 双重 SHA-256 均为
  `2551DC68F2FE94A204BAF722E8FC60143FD0D77B6024979F32EBC65BE4F69912`。完整吸收十项
  amendments，并额外冻结 split digest 字节协议、Gate-3 coverage margin 与 Gate-4
  seed-level mAP bootstrap。状态为 `written_spec_pending_spec_only_review`；尚未授权实现。
- 2026-07-12：空白上下文独立 agent 首轮 spec-only review 发现唯一 P1：Gate-4 detector-regret
  bootstrap 未定义 official-video sample unit。按 exact replacement 修复为单文件 commit
  `e4422f5`，870 行、47,546 bytes，SHA-256
  `87FA305CCAFC3A29176C3971F593489F86EDD23A4C02C1BFBDAE4144FCF34CF8`；复审最终返回
  `APPROVE_SPEC_FOR_PLAN`。状态升级为 `spec_approved`，只解锁 writing-plans，尚未实现。
- 2026-07-12: Created the executable CT-P3R-3S-r2 implementation plan after the independent
  `APPROVE_SPEC_FOR_PLAN` verdict. The plan preserves remote-only behavioral verification,
  implementation/registration commit separation, and the Gate-1-first hard stop chain. Status is
  `implementation_planned`; no new behavior is yet claimed implemented or tested by this entry.
- 2026-07-12: ChronoTransport r2 implementation batch 1 reached partial `tested` status on the remote
  CPU environment: protocol tests 7/7 and candidate/control/cache plus legacy-core tests 36/36. This
  does not unlock a Gate or scientific claim; runtime and later protocol surfaces remain in progress.
- 2026-07-12: ChronoTransport r2 runtime and window-risk implementation reached partial `tested`:
  runtime/integration 35/35 and risk/core 30/30 on the remote CPU environment. The implementation now
  uses all-row AdaTAD adapter writeback, detached historical cache with live current rows, distinct
  requested/executed ledgers, and the fixed D=23 mean/max window head. No Gate has run.
- 2026-07-12: Added and remotely tested Gate-1/Gate-2 pure adjudication (4/4) and Stage-C
  object-identity/loss-specific AMP primitives (4/4). These are synthetic implementation checks;
  no formal data was opened and no Gate result exists.
- 2026-07-12: Added registration/claim-chain primitives, Gate-1 CLI, r2 config overlays and a launcher
  requiring exact GPU1 plus Slurm allocation. Remote registration tests 4/4 and `bash -n` passed. The
  current SSH session has no active allocation, so no formal GPU experiment was started.
- 2026-07-12: Remote combined verification passed 110/110 in 84.58s. An independent implementation
  audit was then requested before freezing implementation commit I or registration R. Known incomplete
  surfaces remain explicitly unregistered and no formal Gate has run.
- 2026-07-12: Independent zero-context implementation audit returned
  `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION` with seven registration-blocking gaps. I/R and formal
  deployment remain locked; the passing 110-test suite is partial implementation evidence only.
- 2026-07-12: Added a GitHub-only Pro audit/discussion prompt pinned to implementation snapshot
  `4b07020acb2611c3f085488d2f678f3be037f1be`. The prompt requires independent re-audit of all seven
  blockers, official-upstream verification, complete patch proposals, and stops before any experiment,
  registration, remote action, commit, or PR.
- 2026-07-12: Published the GitHub-only Pro prompt in commit `6079135` and opened draft PR #1:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/pull/1`. The review target remains the
  immutable pre-prompt code snapshot `4b07020acb2611c3f085488d2f678f3be037f1be`.
- 2026-07-13: Fully archived and absorbed the 1,429-line external Pro GitHub code audit, original
  attachment SHA-256 `1B3A02373366A95654C00A5FE76F451F800D16A877B2688BB460674B25849142`.
  Verdict remains `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`: all seven independent-audit blockers
  are affirmed, with two new P0s for the wrong r2 config nesting and candidate-row pseudoreplication in
  Gate-3 conformal. Route B is retained; I/R, formal profile, Gate 1, new Stage-B seeds, Stage C, and
  Gate 4 remain locked. The review is source evidence, not a test rerun or experiment; its sandbox patch
  proposal is unavailable and explicitly not executed.
- 2026-07-13: Began the bounded ChronoTransport r2 implementation repair. Corrected the Stage-B/C
  config overlay to the inner VideoMAE runtime, separated Gate-3 `30 x 16 -> 30 maxima -> rank 28`
  conformal calibration from the fit-only 140-target rank-127 constant, and made formal profile
  validation reject missing/zero-sample required stages or `total_ms`. The config regression completed
  remote TDD RED (1 failed) then GREEN (1 passed). The first remote Torch-focused run returned 6 passed
  and 1 failed; the sole failure was an expected-error regex mismatch after the validator correctly
  rejected missing `total_ms`. Its message was aligned, but no retry result is claimed. Risk/profiler
  changes are therefore only `implemented`, not newly `tested`. Overall r2 remains partial and
  registration blocked; no formal profile, Gate, training, or experiment was started.
- 2026-07-13: Applied the independent foundation-slice revision without widening scope. Locked both
  r2 quantile helpers to exact ranks 28/127, strengthened the Gate-3 test to use 30
  distinct window maxima, neutralized inherited legacy `max_cache_age` to `None` in the resolved r2
  overlay so it carries neither age meaning, and
  split dynamic-runtime timing into independent `innovation` and `dense_adatad_adapter` stages. Formal profile
  tests now enumerate the required stage names explicitly. The final repaired focused remote run passed
  13/13 in 37.12 seconds. This marks only the foundation slice as remotely tested; whole-r2
  implementation, registration, profiling, Gates, and experiments remain blocked.
- 2026-07-13: Code-quality review found that the first strict implementation had narrowed the legacy
  generic conformal API and changed profile semantics without a schema bump. Added dedicated frozen-r2
  order-statistic helpers, restored the generic API, and introduced `chronotransport_profile_v2` with
  exact schema validation. Full-stack per-invocation aggregation, device-bound peak-memory reset, and
  dense-path instrumentation remain explicitly unimplemented in the later profiler batch. The new
  compatibility/schema tests observed a remote RED (2/2 failed as expected) before production changes;
  the repaired focused matrix then passed 13/13 in 37.30 seconds.
- 2026-07-13: Foundation regression closure: the complete remote `tests/test_chronotransport*.py`
  surface passed 128/128 in 199.80 seconds. An earlier 120-pass/8-fail attempt was invalid as a full
  regression verdict because the temporary scratch tree omitted required `scripts/` and `docs/` files;
  after copying those unchanged repository surfaces, all eight file-presence failures disappeared.
  Independent spec and code-quality reviewers both approved only this foundation slice.
- 2026-07-13: Completed remote TDD closure for the r2 manifest/protocol repair slice. The initial
  missing-symbol RED was followed by a source-vector integrity RED and strict canonical/type negative
  checks. The final focused manifest/protocol suite passed 27/27 in 53.81 seconds; the broader
  protocol/control/legacy-runner compatibility matrix passed 55/55 in 91.90 seconds. The tested scope
  now includes exact source-vector re-derivation, canonical raw bytes and SHA sidecar, duplicate-key
  rejection, strict scalar types, media-path freezing, hashed control/library identities, Stage-B/C
  exposure validation, and the legacy formal-runner hard lock. Registration, formal profile, Gates,
  and experiments remain blocked.
- 2026-07-13: Two independent follow-up reviewers returned `APPROVE_PROTOCOL_SLICE` and
  `APPROVE_PROTOCOL_QUALITY` after reproducing the repaired strict identity and canonical-path
  counterexamples. The bounded protocol/manifest repair was committed and pushed as `33378af`.
  This is an intermediate implementation commit, not the final implementation commit `I`; immutable
  registration `R`, profiling, all Gates, and scientific claims remain locked.
- 2026-07-13: The first Gate-1 exact-cost/full-stack draft passed 36/36 remote focused/adjudication/core
  checks in 37.88 seconds and correctly implemented direct P4-TRANSPORT B* plus the 20% hard threshold.
  Independent spec and quality reviews nevertheless returned `REVISE_GATE1_SLICE`: registration,
  manifest split, exact 23-item profile order, invocation IDs, factory/provenance, strict scalar types,
  and safety-override invalidation were not fully bound. The draft is retained as
  `implemented_under_revision`; no profile, B*, Gate-1 experiment fact, registration, or claim exists.
- 2026-07-13: A read-only Slurm audit proved that the old GPU guard confuses physical IDs with
  cgroup-remapped CUDA ordinals. Protected job `1137541` had physical `GRES IDX:4`, while its step
  exposed `SLURM_STEP_GPUS=4` and local `CUDA_VISIBLE_DEVICES=0`. The revised single-GPU contract is
  physical `SLURM_STEP_GPUS`/`SLURM_JOB_GPUS == 1`, `SLURM_GPUS_ON_NODE == 1`, and local
  `CUDA_VISIBLE_DEVICES == 0`. No current allocation is eligible; no training was started.
- 2026-07-13: The registration-bound Gate-1 repair froze exact 23x200 profiling, manifest-bound 30/30
  record artifacts, strict requested/executed ledgers, registered factory/environment provenance, and
  fixed 5000/20260711 adjudication. Remote focused verification passed 24/24 in 58.26 seconds and the
  complete ChronoTransport regression passed 203 tests with one protected-CUDA-only skip in 315.68
  seconds. Independent review returned `REVISE_GATE1_SLICE` because factory/Git/source identities,
  real manifest/checkpoint bytes, paired-replay regret provenance, launcher prechecks and exact 20%
  arithmetic were not closed. Status is `tested_then_rejected_under_repair`; no Gate result exists.
- 2026-07-13: Stage-C object-identity ownership and AMP overflow rollback reached remote focused
  15 passed/1 protected-CUDA skip and compatibility 27/27. The implementation restores registered
  mutable state while retaining GradScaler backoff and append-only retry audit, and fails after the
  fourth overflow attempt. Independent review then reproduced omitted-adapter, optimizer-hyperparameter,
  infinite-norm, hidden-Python-state, and caller-asserted-action fail-open cases and returned
  `BLOCK_STAGEC_SLICE`. Status is `tested_then_rejected_under_repair`; Stage C remains locked.
- 2026-07-13: Formal Stage B reached remote focused 20/20 in 117.87 seconds and compatibility 71/71 in
  199.51 seconds. The implementation covers exact three-seed/140-success exposure, exception-safe paired
  replay, fixed T+R optimization, strict nonfinite/AMP invalidation, and atomic resume/ledger hashes.
  Independent review returned `REVISE_STAGEB_SLICE_BEFORE_APPROVAL`: arbitrary factory execution,
  missing checkpoint strict-load, self-reported replay/order evidence and absent fit-only 140x16
  baseline remained. Status is `tested_then_rejected_under_repair`; no CUDA smoke, training run,
  calibration, or Gate evidence is claimed.
- 2026-07-13: Gate-1 deep-registration repair reached development focused 34/34 in 170.48 seconds, but
  self-audit kept the slice blocked because the production OpenTAD backend was missing and control
  action bytes were not regenerated from real inputs. A backend RED-to-GREEN sub-slice later passed
  6/6 and the external checkpoint-receipt contract passed 1/1. Real GPU invocation and the unresolved
  unsuffixed random-control seed remain pending; no Gate result exists.
- 2026-07-13: The second Stage-B formal repair passed frozen focused 32/32 and compatibility 71/71.
  Independent review still returned `REVISE_STAGEB_FORMAL_SLICE`: formal CLI omitted current I/R/source
  context validation, and training checkpoint completion preceded the separately written 140x16
  baseline without an atomic phase-completion marker. The implementation is under repair; no seed was
  trained.
- 2026-07-13: Stage-C repair reached focused 19 passed/1 protected-CUDA skip and compatibility
  59 passed/1 skip. A fresh independent review returned `REVISE_STAGEC_PRIMITIVE_SLICE` after reproducing
  wrong scheduler base-LR acceptance, no-op success callbacks, caller-hidden TRANSPORT execution,
  non-leaf autograd version breakage after rollback and an implicit legacy ownership fallback. Task 9,
  CUDA smoke and Stage C training remain locked.
- 2026-07-13: The only visible remote GPU job remained 1137541 on physical GPU4; no protected physical
  GPU1 allocation was available. No login-node training, formal profiler or experiment was started.
- 2026-07-13: At the user's request, a fresh maximum-strength independent agent completed a line-by-line
  Gate-1/registration/profile audit and returned `REVISE_GATE1_IMPLEMENTATION_SLICE`. Passing focused
  tests did not close production backend injection, caller-provided regret, media-timing bias,
  single-parent R/spec/output/cost identity, or the unresolved random/GPU rules. Independent Stage-B
  and Stage-C reviews also remained `REVISE`. The prioritized file-level audit was frozen in
  `OSS_AUDIT.md`; deployment remains locked and no experiment was started.
- 2026-07-13: A subsequent exact-SHA review returned `REVISE_GATE1_FROZEN`. Test-only raw
  profile/replay rows could still reach formal schemas, direct adjudication/unlock APIs bypassed formal
  repository/random-lock context, physical-GPU precheck was fail-open, Git blob/mode binding was
  incomplete, and result/terminal paths could collide. Gate-1 entered another TDD repair; no profile or
  Gate artifact exists.
- 2026-07-13: Stage-B was repaired to bind exact fit windows and registered action hashes, cross-check
  independent EMA against every EMA alias, use one logical predictor hash, and confine all outputs to
  the canonical R/seed root. Five adversarial tests passed and the full remote file passed 46/46 in
  130.27 seconds. Exact-SHA independent review returned `APPROVE_STAGEB_FROZEN`; this is an approved
  implementation slice, not a training run, and the GPU mapping spec blocker still forbids deployment.
- 2026-07-13: Stage-C reached remote 41 passed/1 protected-CUDA skip and compatibility 83 passed/1 skip,
  including a real depth-12 CPU overflow-retry fixture, but exact-SHA independent review returned
  `REVISE_STAGEC_FROZEN`: action identity was caller supplied and topology checks missed isomorphic block
  swaps/shared-container alias splits. Repair resumed; physical-GPU1 CUDA remains pending.
- 2026-07-13: A direct plan-to-files audit confirmed that formal Gates-2/3, Stage-C/matched-dense,
  Gate-4 and the full launcher/test surfaces are absent. Overall status remains
  `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`; I/R and experiment deployment remain forbidden.
- 2026-07-13: Gate-4 pure adjudication began with a remote missing-module RED, then passed 11/11 in
  40.60 seconds in isolated scratch `/data/run01/sczc063/yuzibo/ct_gate4_red_20260713_c45e2a`.
  The tested slice uses matched six-order timing blocks, official-video/seed bootstrap, per-seed raw
  prediction mAP rebuilding without cross-seed NMS, fit-Q1, CT-static regret, cost margin and diagnostic
  p95/memory/long-block energy/stage reports. It is `tested_under_review`; formal CLI, registered source
  coverage, Stage-C inputs and Gate-4 experiment evidence remain absent.
- 2026-07-13: Gate-4 pure adjudication added a raw-evidence report validator and passed 12/12 remotely
  in 40.47 seconds. This closes report-tamper detection only; registered profiling provenance and formal
  execution are still absent. Stage-C second repair reproduced five failures, then passed 5/5 targeted,
  46 passed/1 protected-CUDA skip focused, and 88 passed/1 skip compatibility. Its exact-SHA frozen
  files are under a new independent audit; no Stage-C training, I/R, or experiment was started.
- 2026-07-13: Refreshed Slurm allocation evidence: running jobs 1161501/1161502/1137541 expose physical
  GPU6/GPU4/GPU4 respectively. No physical GPU1 allocation exists, and none was reused for
  ChronoTransport. Formal CUDA smoke and deployment remain blocked without login-node execution.
- 2026-07-13: The new independent Stage-C exact-SHA audit reproduced two more P1 fail-open cases despite
  the 88-test repair run: dummy runtime evidence plus losses disconnected from its graph was accepted,
  and `.data` silently changed frozen heavy parameter bytes without a version bump. Both returned
  `SUCCESS`. Stage C reverted to `tested_then_rejected_under_repair`; Task 9 and deployment remain locked.
- 2026-07-13: The final independent verdict was `REVISE_STAGEC_FROZEN`. A third P1 accepted runtime
  summaries that omitted forced-dense/fallback/evidence-valid fields and used a duck-typed runtime.
  The reviewer also preserved the protected-CUDA skip and P2 gaps for success Python state, formal
  scaler/autocast, direct dependency hashes, and runner-owned canonical-window binding. A third TDD
  repair was assigned; it may not self-approve or deploy.
- 2026-07-13: Stage-C third-repair RED reproduced all 16 targeted failures remotely in 53.38 seconds,
  including separate toy and real-ViT detached-loss SUCCESS paths. Frozen scratch is
  `/data/run01/sczc063/yuzibo/ct_stagec_red3_20260713_4da91b`; GREEN implementation began afterward.
- 2026-07-13: Stage-C third-repair final bytes passed 56 focused, 134 eight-file compatibility and 34
  manifest/protocol tests remotely, with the single protected-GPU1 CUDA GradScaler/autocast skip retained.
  Exact files are frozen under `/data/run01/sczc063/yuzibo/ct_stagec_green3_20260713`; status is only
  `tested_under_review`. Frozen-byte hashing closes `.data` mutation but introduces O(frozen bytes)
  per-attempt overhead that must be measured on the protected GPU.
- 2026-07-13: That Stage-C GREEN was invalidated before independent review because extending autocast
  evidence to the audited risk forward changed `stage_c.py`. The updated exact bytes entered a full
  rerun; prior pass counts are retained as history but cannot certify the current SHA.
- 2026-07-13: Updated Stage-C SHA `36eb6148...a138` passed the exact eight-file compatibility matrix at
  134 passed/1 protected-CUDA skip in 107.55 seconds. The slice is `tested_under_review`; no independent
  approval or physical-GPU1 AMP evidence exists yet.
- 2026-07-13: Independent Stage-C GREEN3 review then found two more P1 candidates: post-forward
  `latest_signals` could be replaced or `.data`-modified before the risk forward, and successful updates
  did not enforce registered-buffer byte equality. Stage C reverted to
  `tested_then_rejected_under_review`; remote reproductions were requested in a fresh scratch.
- 2026-07-13: Gate-1 hardening self-verification reached 51 focused pass/1 strict-xfail, 11 profile/
  adjudication pass and 49 replay/core/pipeline pass. It remains `tested_under_review`, with random seed,
  GPU mapping and final source expansion intentionally blocked. Gates-2/3 initially reached 10 remote
  passes, but the module/test bytes changed afterward; the old result was rejected and an exact final-byte
  remote rerun was required before freezing or independent review.
- 2026-07-13: Gates-2/3 final bytes (`f7f8340` module, `c213013` CLI, `0e804d8` test) were copied to a
  fresh remote scratch, rehashed, and passed 10/10 in 43.32 seconds. The slice entered a separate
  independent audit; no report/unlock or formal execution was created.
- 2026-07-13: Independent exact-SHA review rejected both newly green-looking slices. Gate 1 retained
  contextless direct unlock, importable issuer/raw-runner formal minting, and missing binding between the
  in-memory registration and the exact regular `R` blob. Gates-2/3 lacked independent formal context
  guards and strong phase provenance; a remote adversarial example showed its seed bootstrap could PASS
  with CI `[0.0467,0.4867]` where global seed-cluster resampling gives `[-1.2,1.0]` and FAIL. Both await
  complete final verdicts before another repair; no formal artifacts or jobs were created.
- 2026-07-13: Gates-2/3 final verdict `REVISE_GATES23_FROZEN` added P0s for caller-written formal replay,
  text-checkpoint/arbitrary-ledger phase acceptance, and context-free public formal APIs; P1s covered the
  seed-cluster CI reversal, unbound registration commit, and a stale Gate-1 unlock call signature. A new
  RED-first repair was assigned; it may not self-approve.
- 2026-07-13: Gate-1 hardened final verdict `REVISE_GATE1_HARDENED` reported no P0 but four P1 classes:
  contextless public formal APIs, externally reusable formal construction helpers, missing exact binding
  from the in-memory registration to the regular `R` blob, and overwrite-capable result/terminal TOCTOU.
  A different agent received the RED-first repair; random/GPU locks remain unchanged.
- 2026-07-13: Gate-4 evaluator parity RED found local AP=1.0 versus official OpenTAD AP=0.5 for equal
  scores. The adjudicator was switched to the official evaluator; the new test passed and the full exact
  slice passed 13/13 remotely in 232.34 seconds. This remains a pure adjudicator, not formal Gate 4.
- 2026-07-13: Gate-1 Green2 produced an exact 15-file SHA-256 vector and passed the repair agent's remote
  focused suite at 57 passed/1 strict xfail plus 8 profile/adjudication compatibility tests. A fresh
  independent reviewer matched all 15 local bytes before starting review. Integration inspection found
  the frozen Stage-B CLI still uses the old Gate-1 unlock validator signature; this is a separate
  deployment blocker requiring RED-first repair and Stage-B re-freeze, not permission to weaken Gate 1.
- 2026-07-13: Independent Stage-C Green3 probes in
  `/data/run01/sczc063/yuzibo/tmp/audits/stagec_green3_independent_repro_20260713_a` reproduced three
  false `SUCCESS` outcomes: replacing `latest_signals`, mutating its logical bytes via `.data`, and
  mutating a registered model buffer via `.data`. Each changed logical values by max delta 7.0.
  Two further probes changed detector/feature forward outputs via `.data` and still passed VJP provenance.
  Green3 is `REVISE_STAGEC_GREEN3`; a five-case RED-first Green4 repair was assigned to a different agent.
- 2026-07-13: Gates-2/3 round2 synchronized final Gate-1 context plus a canonical 17-file dependency
  vector (`04e5572b...43cf`). Exact remote results were 16/16 focused and 61 passed/1 strict xfail Gate-1
  compatibility. Four exact SHA files entered an independent review as `tested_under_review`; no report,
  unlock, registration, or experiment was created.
- 2026-07-15: A fresh zero-context review matched Gate1 Green2 15/15 SHA and returned
  `REVISE_GATE1_GREEN2`: arbitrary-backend/callable profile construction and caller detector/batch replay
  can cross shared fixture/formal rebuild paths into formally accepted schema. R/current-byte and
  no-clobber checks passed but do not establish evidence origin. Gate1 entered another RED-first repair.
- 2026-07-15: Stage-B old Gate1 validator integration was reproduced RED-first, then repaired to pass
  exact repository root/current R/registration relpath. Remote Stage-B suite passed 47/47 in 132.98s in
  `/data/run01/sczc063/yuzibo/tmp/audits/stageb_gate1_context_red_20260715_a`; core SHA stayed frozen.
  A separate exact-SHA review returned `APPROVE_STAGEB_CONTEXT_REPAIR`.
- 2026-07-15: Read-only Slurm refresh returned no active jobs for `sczc063`; no protected physical-GPU1
  allocation exists to reuse. No ChronoTransport job was launched.
- 2026-07-15: Independent exact-byte Gates2/3 review reran 16/16 remotely but returned
  `REVISE_GATES23_ROUND2`: its four formal files are absent from the registration source set; terminal
  creation lacks full context/recomputation; two concurrent writers overwrite; and parent-directory
  symlinks pass. Correct fixed replay, Stage-B 140-row validation and global seed bootstrap were retained.
- 2026-07-15: Stage-C Green4 expanded the mutation boundary to 12 RED cases, then passed 12/12 targeted
  and 143 passed/1 protected-CUDA skip in the exact eight-file remote matrix. Candidate SHA are
  `e00f3730...fa6f` and `eaa047ba...abce` in
  `/data/run01/sczc063/yuzibo/tmp/audits/r2_stagec_green4_20260715_a`; a different agent began the
  mandatory exact-SHA review. Status remains `tested_under_independent_review`.
- 2026-07-15: The independent Green4 review returned `REVISE_STAGEC_GREEN4`: Tensor-valued ordinary
  Python module state still omitted storage identity, permitting equal-value storage rebind on success
  or overflow. Explicit forward evidence and registered buffers passed review; Green5 is required.
- 2026-07-15: Gate1 Green3 removed callable/backend and raw-row formal construction paths, separated
  fixture/formal row schemas, and passed 25 focused plus 169/1xfail broad remote tests and 30 Gates23
  compatibility tests. Exact bytes are `tested_under_independent_review`, not self-approved.
- 2026-07-15: Gates23 round3 repaired report-derived terminal semantics, exclusive lock/no-clobber and
  component-wise symlink rejection. Remote results were 21/21 focused and 30/30 corrected Gate1
  compatibility. Registration-vector integration and independent approval remain pending.
- 2026-07-15: Gates23 round3 received independent `APPROVE_GATES23_ROUND3_CODE` after exact 21/21 and an
  orchestration probe. Initial registration integration passed its suites but was independently rejected
  because `test_chronotransport_r2_gate1_hardening.py` remained outside the immutable exact source set;
  one final RED-first vector repair began.
- 2026-07-15: Stage-C Green5 reproduced two ordinary Python Tensor storage/alias failures, then passed
  targeted2, aggregate targeted14, and a 198-pass 12-file remote superset. Exact SHA `d994cefd...3fbe` /
  `6b9c113f...c837` entered a fresh independent review; physical-GPU1 CUDA/overhead evidence remains open.
- 2026-07-15: Generated
  docs/methods/2026-07-15-chronotransport-r2-current-github-pro-line-review-prompt.md for a future
  GitHub-only Pro line-by-line audit. The public branch was still 797a2df while newer implementation
  bytes remained local, so the prompt fails closed unless GitHub exposes a newer immutable descendant
  containing the complete current source/test surfaces and the approved spec hash. No Pro review,
  registration I/R, formal Gate, Slurm job, experiment result, or claim transition occurred.
- 2026-07-15: A new independent pre-deployment integrity audit returned
  `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`. It proved the exact registration vector omitted current
  Gate hardening/Gate4 files and all future runners, the random-control spec lock makes formal Gate1
  intentionally unreachable, the Stage-B factory read obsolete flat registration fields, real
  train-mode ActionFormer `loss_normalizer` conflicts with Stage-C success buffer immutability, and
  matched-dense/Gate4 formal runners are absent. The current governing Slurm rule also conflicts with
  the old physical-GPU1/CVD=1 contract. A nested-schema Stage-B regression/fix and current Gate source
  registration repair were started and passed local static compilation only. No I/R, Gate artifact,
  Job, result, or claim transition occurred.
- 2026-07-15: The first post-audit repair slice passed seven targeted remote CPU tests in 57.28 seconds
  under `/data/run01/sczc063/yuzibo/tmp/audits/ct_r2_integrity_fix_20260715_a`. Coverage is limited to
  nested Stage-B registration identity, split provenance, no-clobber publication and the exclusive
  Stage-B writer lock. This is focused implementation evidence, not full-suite approval, I/R, a formal
  Stage-B run, Gate evidence or a scientific result.
- 2026-07-15: The complete affected Stage-B and registration suites passed in the same isolated remote
  worktree: `89 passed, 1 xfailed in 310.86s`, with no failures. The repaired surfaces remain
  implementation-only evidence; frozen-protocol amendments, Stage-C/matched-dense/Gate-4 runners,
  complete registration, I/R and all formal jobs remain locked.
- 2026-07-15: A follow-on Gate-1-hardening/Gates-2/3/Gate-4-adjudicator compatibility matrix passed
  `43/43` remotely in 295.25 seconds. No formal Gate was invoked and no Gate-4 producer exists yet;
  the result only rules out regressions across those present test surfaces.
- 2026-07-15: A later Slurm refresh superseded the earlier empty-queue observation: multiple unrelated
  DUCA jobs owned by the same account were active or pending. None was a registered ChronoTransport
  allocation, and no job was reused, modified, cancelled or launched for ChronoTransport.
- 2026-07-15: Gates-2/3 half-publication recovery was implemented RED-first: existing replay/report is
  reusable only after exact recomputation produces identical canonical bytes; terminal, mismatched or
  non-regular artifacts remain fail closed. Remote targeted GREEN was 1/1 and the combined
  Gates-2/3+registration result was `59 passed, 1 xfailed` in 206.70 seconds. Exact bytes entered
  independent review; no Gate artifact or claim transition occurred.
- 2026-07-15: Drafted an explicitly unapproved minimal r2 protocol amendment proposal (SHA-256
  `30371FFC...8469F`) covering A1 fixed random seed 3407, A2 Slurm-assigned logical `cuda:0`, A3 matched
  train-mode `loss_normalizer` success/rollback traces, and A4 paired dense-reference/counterfactual
  per-window regret production. The frozen spec was not edited; status is `discussed/proposed`, and
  registration/formal execution remain locked pending explicit user approval plus spec-only review.
- 2026-07-15: Independent exact-byte review returned `APPROVE_GATES23_RECOVERY`, matching runner SHA
  `4CED5459...616885` and test SHA `10D13457...65AFB9`. It confirmed `_run_locked` integration,
  canonical-byte-only reuse, terminal refusal, regular-file/symlink enforcement, hard-link no-clobber
  and exclusive writer locking. This promotes only the bounded repair to code-approved; overall r2
  registration remains `NOT_READY` and no experiment/claim state changed.
- 2026-07-15: Refreshed the GitHub-only Pro line-review prompt after the Stage-B/registration and
  Gates2/3 recovery repairs. Current prompt SHA-256 is
  `241B3A2EE8BFF2B9983E82C3F5BF7B5DFB600F124457D00423B0E1A388138DDD`; its snapshot guard still
  stops on public HEAD 797a2df or missing current files. This is review input only, not a Pro response.
- 2026-07-15: Stage-B formal path/lock RED reproduced two bypasses: a symlink parent was accepted for
  the writer lock, and pre-validation `.resolve()` laundered a canonical R/seed symlink alias. The
  bounded lexical-lstat/O_NOFOLLOW/inode-safe cleanup repair passed targeted 5/5 and the full affected
  Stage-B+registration matrix `91 passed, 1 xfailed` in 291.82 seconds. Exact bytes entered independent
  review; no formal training or claim transition occurred.
- 2026-07-15: Independent exact-byte review returned `APPROVE_STAGEB_PATH_LOCK_HARDENING`, matching
  runner SHA `64B4A5AA...053B3D` and test SHA `1E7BB883...094CA`. It confirmed lexical component
  checks before resolution, real-loader missing-input refusal, O_EXCL/O_NOFOLLOW, inode-safe lock
  cleanup, hard-link no-clobber and unchanged formal R/Gate1 context. Only this bounded slice is
  code-approved; overall registration and every formal job remain locked.
- 2026-07-15: Refreshed the fail-closed GitHub Pro line-review prompt with the independently approved
  Stage-B path/lock slice. Current prompt SHA-256 is
  `B6D21D6073C9DDB3B6352D6DE8C92C893E0112DCFC6E832D80207928AAE335B0`; public GitHub snapshot
  guard and all no-result/no-approval statements remain unchanged.
- 2026-07-15: Gate-1 precheck parent-symlink RED was closed with lexical component checks and an
  independent R-derived output reconstruction. Remote evidence was focused 1/1, complete
  registration/precheck `38 passed, 1 xfailed`, and Gate-1 hardening/cost `25 passed`. Independent
  exact-byte review returned `APPROVE_GATE1_PRECHECK_PATH_HARDENING` for precheck SHA
  `0BE0EA8B...F76808` and test SHA `55916FBD...C10BA`. Only this bounded implementation slice changed
  to code-approved; formal Gate 1, I/R, all training and all claims remain locked.
- 2026-07-15: Stage-B partial-publication RED reproduced exact-existing completion/ledger refusal and
  added dangling periodic/final ledger plus final-pair recovery. The first 98/1xfail candidate was
  independently rejected for a phase dense-path alias and post-precheck pathname load. Replacement
  bytes added inside-lock regular-file state checks and O_NOFOLLOW/lstat/fstat inode-bound reads. Final
  evidence: targeted 5/5, Stage-B+registration 98 passed/1 xfailed, Gate compatibility 44/44, and
  `APPROVE_STAGEB_PARTIAL_PUBLICATION_RECOVERY` for SHAs `50F4469D...F4F84`,
  `47342FFE...A7670`, `9BB46DE2...E378D`. No formal job or result was produced.
- 2026-07-15: Refreshed the fail-closed GitHub Pro full line-review prompt with current Gate1 precheck
  and Stage-B partial-publication phenomena, the rejected intermediate candidate, final exact SHAs,
  missing Stage-C/Gate4 workflows, and unapproved A1--A4. Current prompt SHA-256 is
  `31E95F058603C3610608B8304F20DADE03EE76A1A64F9D2256936FE2D3FB3C1E`; public GitHub still must
  resolve beyond 797a2df with all current files or the reviewer must return `GITHUB_SNAPSHOT_INCOMPLETE`.
- 2026-07-15: Read-only `squeue` refresh found 13 running/pending jobs, all named DUCA/P0/S1 rather
  than ChronoTransport. No CT job was submitted or adopted; those unrelated jobs remain out of scope
  and untouched. Formal CT execution is still locked by A1--A4 and missing executable workflows.
- 2026-07-15: Gate-4 caller-raw-dict formal minting was reproduced RED-first, then locked before raw
  evidence parsing. Non-formal results now use the explicit test-only schema. Remote evidence was
  focused `1 passed`, forged-recomputed-hash targeted `1 passed in 105.28s`, full Gate-4 `13 passed in
  242.40s`, and registration-source focused `1 passed`; exact SHAs are `A581D713...B1A75F` /
  `5C0FFAF3...AB4506E`. Independent review returned
  `APPROVE_GATE4_CALLER_EVIDENCE_LOCK_FINAL`. This bounded approval created no official evidence
  producer, formal report, Gate result, registration or experiment.
- 2026-07-15: Refreshed the GitHub-only Pro prompt with the Gate-4 caller-evidence RED, explicit
  test-only replacement schema, remote regression counts and candidate hashes. Prompt SHA-256 is now
  `CE7E008B...59024`; the public GitHub snapshot guard remains unchanged and no Pro review occurred.
- 2026-07-15: Stage-C formal measured-cost evidence was hardened RED-first. The old validator accepted
  `cost_is_measured=False`; exact-true enforcement then exposed proxy-cost test fixtures and produced an
  intentionally retained intermediate full result of `39 failed, 32 passed, 1 skipped`. After binding an
  explicit test-only measured-cost table, remote focused passed `1/1` and full Stage-C passed `71 passed,
  1 skipped in 76.60s`. Exact SHAs are `5BDC1862...5577C4` / `C92FED39...3A262D7`; independent review
  returned `APPROVE_STAGEC_MEASURED_COST_FLAG_LOCK`. This does not provide immutable profile provenance,
  a formal runner, A3/A4, registration, training or a scientific result.
- 2026-07-15: Materialized `EXPERIMENT_AUDIT.md/.json` and the required forensic trace under
  `.aris/traces/experiment-audit/2026-07-15_run01/`. The readiness verdict is `FAIL` because the formal
  evidence chain and official scope are absent; no fabricated result was detected and no formal result
  exists. Separate read-only Codex reviewers were used, but cross-model-family independence cannot be
  attested. The GitHub-only Pro prompt was refreshed with the final Gate4 and Stage-C bounded reviews,
  current audit files and no-assumption instructions; final prompt SHA-256 is
  `F693E0DCD748BFA9AB93CB39A56EC0F18A5810AEC45543BF22A821F962C1B9DF`. A fresh `git ls-remote` at
  2026-07-15T21:07:55+08:00 confirmed that public GitHub still points at stale `797a2df`, so the prompt
  must return `GITHUB_SNAPSHOT_INCOMPLETE` until latest bytes are published.
- 2026-07-15: The persistent CT r2 execution goal was marked `blocked` after the same protocol-authority
  condition recurred across more than three goal turns. All amendment-neutral integrity work is exhausted;
  A1--A4 remain `proposed_unapproved`, so spec-only freeze/review, runners, I/R, PRECHECK and every formal
  Slurm stage remain forbidden. This is an authorization/specification block, not a Gate failure or route
  kill, and it produced no experiment result.
- 2026-07-15: The user supplied the first GitHub-only Pro response. It returned
  `GITHUB_SNAPSHOT_INCOMPLETE` after resolving `codex/chronotransport-r2-implementation` to the forbidden
  old SHA `797a2df...d836ee` with `ahead_by=0` and `behind_by=0`. Missing files, spec SHA and code were
  explicitly `NOT_EVALUATED_AFTER_FIRST_GATE_FAILURE`; therefore no implementation verdict exists. The
  verbatim response was archived as source SHA `AF4E4FA6...D691DDE`. No code, Gate or goal state changed.
- 2026-07-15: Refreshed the GitHub-only Pro prompt to require the archived first snapshot-gate response
  and to prohibit reuse of its old SHA/incomplete verdict. The next reviewer must fresh-resolve the
  newly published branch. Current prompt SHA-256 is
  `1D0E7FC160FFB1E30375BF2EC3BF9E44381EA3BAB1D8BB5DD8EA2249E3D93BF5`; this edit remains review-only
  and does not unlock A1--A4, registration or experiments.
- 2026-07-15: Absorbed the user-supplied 1,430-line GitHub Pro audit of immutable review-only SHA
  `b854adb4f4c9235580b5e58c3f3255db6e9adbc0`. The verbatim archive is 73,605 bytes and matches the
  attachment at SHA-256 `1A7B9D5A...C4C376`. Its overall
  `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION` verdict and the missing formal workflows, real
  ActionFormer Stage-C contract, A1--A4, source-vector, Slurm and measured-cost-provenance blockers are
  accepted. The suggested raw test glob is not accepted verbatim: 21 files match but only two changed
  integration tests are presently confirmed omissions, so a frozen classification manifest is required.
  The proposed Stage-C evidence interface remains a design sketch pending A3/A4. No implementation code,
  I/R, test run, CUDA/Slurm job, Gate or scientific result was produced.
- 2026-07-15: After the user authorized exact A1--A4, froze them in spec-only commit `537f692`, and kept
  all execution locks closed, refreshed the GitHub-only Pro full line-review prompt for a new immutable
  descendant snapshot. The prompt now requires a separate `e4422f5 → 537f692` spec-diff verdict, explicit
  classification of every ChronoTransport test, real ActionFormer paired-forward/`loss_normalizer`
  transaction review, Slurm-assigned logical `cuda:0` identity, implementation-grade RED tests and complete
  patches. Prompt SHA-256 is `9DDCABC19E6B38874EA97F5E4702C247D2DF8F485CE273E08E4A6515EBFEC3D0`.
  This is review-only documentation: no implementation, I/R, PRECHECK, test rerun, Job, Gate or experiment
  result was created.
- 2026-07-16: Archived and absorbed the user-supplied 2,019-line GitHub Pro review of immutable review-only
  SHA `1b6366d0acb712e8096c2cceb0f05e66b16d30d4`. The verbatim archive exactly matches the 86,871-byte
  attachment at SHA-256 `C61F9353...F1BC08`. The A1--A4 diff received `APPROVE_SPEC_FOR_PLAN`, so its
  specification status is now `spec_approved`; the implementation remains
  `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`, registration `NOT_READY`. Core A1/A2/Stage-C/workflow/source
  blockers were accepted after local source checks. Concrete patches were only partially adopted as design
  input: A2 needs producer-observed artifact identity and repo-wide migration; Stage-C loss/interface fields
  must come from registered ActionFormer semantics; all 21 matching tests need explicit classification;
  unauthorized/provenance-invalid runs are `INVALID_IMPLEMENTATION`, not automatic science FAIL. The stale
  `e4422f5`/GPU1 implementation plan is now explicitly forbidden unchanged. No production code, test run,
  CUDA/Slurm action, I/R, PRECHECK, Gate, training or scientific result was created.
- 2026-07-16: The user authorized full ChronoTransport r2 implementation and stop-chain execution. Replaced
  the stale `e4422f5`/physical-GPU1 plan with an A1--A4-approved dependency plan and created a locked
  execution tracker. A read-only N16R4 check found available public GPU capacity but only unrelated DUCA/S1
  jobs for this account; none was reused, modified or cancelled. Status is implementation repair
  `IN_PROGRESS`, registration `NOT_READY`, all E0--E5 stages `LOCKED`, and no experiment was launched.
- 2026-07-16: Completed and remotely tested implementation packages W0 and W1/A1. A frozen classification
  manifest now covers all 47 tracked ChronoTransport tests/tools/scripts and all 21 matching tests; exact
  `REQUIRED`/source-vector agreement rejects omissions and unclassified additions. Registration authority
  now binds approved spec `537f692` and exact hash, while `random_p2/p4/p8` require integer seed 3407 and
  context generation/replay independently recompute all per-window action hashes. The first remote run
  exposed a valid stale-template-hash RED (`43 passed, 1 xfailed, 1 failed`); after repair the affected
  contracts passed 9/9 and the control/manifest matrix passed 36/36. W2/A2 is now in progress. No I/R,
  PRECHECK, Slurm experiment, Gate result, training or scientific claim was produced.
- 2026-07-16: Completed the W2/A2 code contract through clean commit `c585ae5`. Registration now freezes a
  required model/software identity without a fabricated UUID; live producers record raw
  `CUDA_VISIBLE_DEVICES`/Slurm allocation fields, map the current CUDA PID to one full GPU UUID, enforce one
  visible device and logical `cuda:0`, and bind allocation/observed hashes into Gate-1 profile/replay/result,
  precheck and Stage-B evidence. The r2 launcher was renamed to the Slurm-single-GPU surface and never
  assigns scheduler visibility; the physical-GPU1 latency claim was removed. Initial remote contract runs
  retained three stale-fixture failures (78 pass/2 fail, then 81 pass/1 fail); all three were repaired and
  targeted 2/2 plus 1/1 passed. Final clean Gate-1/A2 regression was 78/78 with launcher `bash -n`. These are
  remote CPU implementation tests only: no Slurm GPU action, PRECHECK, I/R, Gate, training or science result
  occurred. W3 is now in progress and registration remains `NOT_READY`.
- 2026-07-16: Rechecked the user attachment at
  `attachments/90705a73-361e-4a4d-88eb-052abdebeef0/pasted-text.txt`. It is byte-identical to the existing
  86,871-byte verbatim archive of the `1b6366d` Pro review, SHA-256
  `C61F93531885040A3593DB7552E23B67B34DEC3D55095D71FCE5B6D2A1F1BC08`. It is recorded as a duplicate
  presentation of the same source, not a second independent audit; no duplicate archive or graph source was
  created, and the accepted `APPROVE_SPEC_FOR_PLAN` / `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`
  disposition is unchanged.
- 2026-07-16: Completed the non-I/R W3--W6 implementation candidate at
  `6c3606cc5161d415909a42741b3bc402278bf332`. During final cross-check, a transient Gate-4 implementation
  drift had replaced registered per-invocation margin and per-seed hard conditions with arm-level p50
  arithmetic; it was caught against the approved specification, restored, and covered by reversal tests.
  Eighteen changed/new files were SHA-256-identical in the remote CPU checkout. Targeted Gate 4 passed 32/32;
  the exact complete ChronoTransport suite passed `441 passed, 1 skipped, 2 warnings in 968.62s`; remote
  py-compile, four r2 launcher syntax checks and C3 compatibility 20/20 passed. The skip is the protected
  CUDA-only contract. These are implementation tests, not scientific results: exact-byte independent Pro
  review, implementation I, registration-only R and R-bound CUDA/Slurm PRECHECK remain absent, so E0--E5
  stay locked and no experiment was launched.
- 2026-07-16: Published a replacement GitHub-only Pro implementation-audit contract for a fresh immutable
  descendant of `6c3606c`. It requires a complete commit parent vector/tree certificate, exact spec/review
  hashes, exhaustive classification-driven line coverage, previous-finding closure, concrete P0/P1 RED tests
  and patches, and a binary `APPROVE_IMPLEMENTATION_FOR_REGISTRATION` versus
  `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION` verdict. Prompt SHA-256 is
  `663757DE188A3EB62E50FD9A0E00AE7D9768206E072475D9BA76EAEDE1647ABF`. The prompt is review-only and
  cannot create I/R, unlock CUDA/Slurm PRECHECK or start an experiment.
- 2026-07-16: Archived the user-supplied Pro snapshot-gate response for exact review SHA `92a18be` at
  SHA-256 `990E84F1...151FD1`. The reviewer independently proved the implementation-floor first parent, absence
  of a second parent, ahead 1/behind 0 and the exact eight-path docs-only diff, but its GitHub interface did not
  expose tree SHA or separate Git-object timestamps. It therefore correctly obeyed the then-current prompt and
  returned only `GITHUB_SNAPSHOT_INCOMPLETE`; no spec, code, test or registration verdict exists. Proposed
  remediation is a strict content-addressed fallback certificate, not relaxation to project-reported metadata.
  No I/R, PRECHECK, Slurm job or experiment was created.
- 2026-07-16: The user approved the strict equivalent-certificate repair. The replacement prompt keeps the
  complete Git Data commit object as Route A and allows Route B only after the reviewer proves interface field
  unavailability and independently verifies frozen SHA, exact first/second-parent probes, strict ancestry to
  all four anchors, the complete implementation-floor docs-only diff and every mandatory file through a
  SHA-pinned read. Tree/timestamps remain untrusted when unavailable. This is a docs-only review-transport
  change; implementation approval, I/R, PRECHECK and E0--E5 remain absent/locked.
- 2026-07-16: Finalized the approved replacement Pro prompt at 406 lines / 24,050 bytes, SHA-256
  `7F67EAB17A9A4280CE27431DA0DB5A3B6B219AEE4EC7A075D4AE7FD75D6866C8`. It also pins the verbatim
  `92a18be` snapshot response (`990E84F1...151FD1`) and its absorption (`EB7A5767...078B1`) so the next
  reviewer must distinguish prior tool failure from a code verdict. Publication remains docs-only and does
  not create I/R, PRECHECK or experiment evidence.
