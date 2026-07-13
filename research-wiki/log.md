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
