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
- 2026-07-15：纠正活动任务为 Spatial Zoom-only。S1 保持完整时间轴，只比较 matched
  dense160/224/256；DUCA 与时序选帧不属于此执行线。旧 `35204f5` 3x3 任务因每臂均出现
  `upsample_linear1d_backward_out_cuda` nondeterminism warning 而失效并取消，222 个
  checkpoint/sidecar 仅保留为诊断，无 sealed-test、最终 mAP、成本或 GO/KILL。
- 2026-07-15：在 `codex/spatial-zoom-s1-audit-fix-20260715` 实现 exact-2x deterministic
  temporal interpolation、formal strict determinism、真实 full-model AMP backward precheck、
  paired Bayesian video-cluster bootstrap、test-open crash recovery、原子证据发布和成本口径
  限定。本地 S1/train-iteration tests 为 `40 passed, 1 skipped`，required C3 regression 为
  `20 passed`；
  远端 replacement CUDA gate 与 3x3 matrix 尚未产生结果。
- 2026-07-15：提交 `64e71dd` 后创建 exact remote snapshot
  `opentad_spatial_zoom_s1_64e71dd_20260715_ghfast`。Job `1165648` 的 Linux 测试
  `41 passed`，真实 pretrained VideoMAE + AdaTAD AMP forward/backward 成功执行，但门禁因
  `backbone.model.backbone.fc_norm.{weight,bias}` 两个参数无梯度而 fail-closed。源码证明
  它们仅属于 `return_feat_map=False` 的分类池化出口，S1 dense TAD 路径在此前返回。
- 2026-07-15：将 formal precheck 升级为 v6：只允许上述两个精确参数缺梯度，要求观察集合
  与冻结白名单完全相等，并分别统计 trainable 与 gradient-required tensors；新增少项、额外
  断图和 component coverage 漂移测试。本地为 `43 passed, 1 skipped`。替换 CUDA gate 通过
  前保持 3x3 正式训练未排队。
- 2026-07-15：独立 `gpt-5.6-sol/max` 只读审计给出 `PASS_FOR_REMOTE_GATE`，无 P0/P1；
  其唯一 P2 是 component 与 global gradient counts 未闭合。提交前已增加所有 component 的
  trainable/required/unused/finite/nonzero 守恒验证及 under-report 反例测试。该门禁结论仍不
  等价于 S1 empirical GO。
- 2026-07-15：修复提交 `47842427eb373fb1f440b1661971a6a231a95f67` 已推送。
  新 exact snapshot `opentad_spatial_zoom_s1_4784242_20260715_ghfast`、suite
  `spatial_zoom_s1_4784242_20260715_2245` 和 manifest 均重新创建，未修改或复用旧快照。
- 2026-07-15：CUDA gate Job `1165667` COMPLETED 0:0 / PASS。三分辨率均为 339 个
  trainable、337 个 gradient-required/finite tensors；缺梯度集合精确等于两个 `fc_norm`，
  strict deterministic warn-only=false，formal_training_ready=true。
- 2026-07-15：正式 3x3 S1 matrix 已排队：`1165669-1165671`=dense160，
  `1165672-1165674`=dense224，`1165675-1165677`=dense256，各对应 seeds
  3407/3408/3409。状态提升到 `experiment_running`，sealed test/cost/GO-KILL 尚未完成。
- 2026-07-15：九任务均开始 epoch 0，初始 loss 有限。若 GradScaler overflow，S1 会恢复
  同批次 RNG/model buffers 并重放，成功前不推进 scheduler/EMA；单次 `AMP skipped batch`
  日志不是丢失 optimizer update。八次重试耗尽、raw loss 非有限或 successful-update parity
  不满足才判失败，并在 checkpoint/evidence 中记录全部 attempts。
- 2026-07-15 23:03+08:00：`1165669-1165677` 全部 RUNNING；dense160 已进入 epoch 2，
  dense224/256 已进入 epoch 1。最新 loss `1.0760-1.1135`，各臂 2-3 次 AMP attempt 已恢复；
  Traceback/OOM/non-finite/determinism/FAIL 计数均为 0。epoch 40 前无 validation mAP 是预注册行为。
- 2026-07-15 23:08+08:00: updated the two-hour Spatial Zoom S1 heartbeat
  into an idempotent end-to-end state machine: monitor the formal 3x3 matrix,
  validate checkpoint selection, open the sealed test once, run official test
  and trained-checkpoint cost profiling, issue the frozen GO/KILL diagnosis,
  publish only auditable summaries to GitHub, then invoke a verified Pro-tier
  code/protocol/result review and deactivate the automation.
- 2026-07-16 02:55+08:00: all formal jobs `1165669-1165677` failed closed while
  atomically writing checkpoints because the shared `/data` filesystem had only
  about 13 MiB free. Losses remained finite; this is infrastructure failure,
  not S1 performance evidence. Resume is forbidden, so the matrix is invalid.
- 2026-07-16 03:02+08:00: verified the 130 GiB canonical root with exactly 222
  checkpoints belongs to invalid commit `35204f58`. Removed only those forbidden
  `.pth` weights, preserved all 222 sidecars and diagnostics, and wrote purge
  receipt `59f1d9d3499eb3cd105478672805f9a19c15a73c8a747b1249bc7c2372ad9ecf`.
  Began a replacement fix that saves only gate-eligible checkpoints, cleans
  failed atomic temporaries, and enforces a 96 GiB free-space preflight.
- 2026-07-16 03:18+08:00: hash-validated and reclaimed the no-resume
  `47842427` failed matrix: 151 weights (94,195,092,514 bytes) plus nine
  temporaries (134,217,728 bytes). Preserved all sidecars and diagnostics;
  receipt file SHA is `b5237253eaa8d196957da47d5ebd2c07ae6537596b6e53e1e4348286c88d58d9`.
  `/data` free capacity rose to 217 GiB before the fresh replacement gate.
- 2026-07-16 03:29+08:00: storage-safe commit
  `0421a8d9f6982a6d4ec1fb590cd108581fa2bb83` passed CUDA gate Job `1165774`
  (0:0, 47 Linux tests, three-resolution deterministic full-model backward).
  Precheck internal SHA is `3d30ea5489b2ac7f07785dff94ed057ac420aebdd8762ab6df6c76a2ffb003ea`.
- 2026-07-16 03:31+08:00: submitted fresh epoch-0 3x3 Jobs `1165775-1165783`
  to canonical namespace `bf71376e2d57946a3f898d25b7dcc88cfc002549a9ed78656293f1a95316a8f7`.
  All nine received normal Slurm GPUs without physical-index override. No
  resume, test opening, profiling, or performance interpretation occurred.
- 2026-07-16 04:57+08:00: all nine storage-safe S1 cells remain healthy and
  running at epochs 11-15. Latest finite losses are `0.5830-0.7153`; retry
  counts are 2-4 with maximum same-batch retry `2/8`, followed by continued
  progress. Pre-gate checkpoint/evidence counts correctly remain zero and
  storage remains 217 GiB free. No hard-failure signature was found.
- 2026-07-16 06:58+08:00: formal S1 replacement remains healthy at epochs
  27-37. Latest finite losses are `0.4248-0.5028`; AMP retry counts/depth did
  not increase. No checkpoint or gate evidence exists before its frozen epoch,
  no hard failure was found, and `/data` still reports 217 GiB free.
- 2026-07-16 09:05+08:00: all nine S1 jobs remain healthy at epochs 43-55.
  Exactly 37 eligible checkpoints and 37 sidecars exist, with 34 completed gate
  evidence records and three evaluations still publishing; no temporary or
  selection file exists. Completed gate-only Avg-mAP values span 14.85-15.15
  and are not sealed-test evidence. Update/attempt parity, commit binding, and
  `official_test_opened=false` hold; hard-failure scans are clean and `/data`
  retains about 195 GiB free.
- 2026-07-16 11:00+08:00: Jobs `1165775-1165780` completed all 4,800 model
  updates and ten gate artifacts but failed only in post-training selection.
  Every rejected row was a finite zero-length proposal, which official OpenTAD
  retains as a zero-IoU false positive. An exact official-parity probe passed;
  dense160/3407 epoch59 recomputed gate Avg-mAP 64.7391 and mAP@0.6/0.7
  58.0680/46.1726. Fix `cbc63d0` passed 41 focused tests plus 20 C3 regression
  tests. No selection, sealed test, profile, or GO/KILL was allowed; dense256
  remains running and the strict contract requires a new formal namespace.
- 2026-07-16 11:25+08:00: formal replacement commit `18139b9` passed full
  CUDA gate Job `1166358` (`0:0`). Submitted fresh epoch-0 3x3 Jobs
  `1166361-1166369` to canonical namespace `d95a36db...`, without resume or
  physical-GPU override. An `afterany` dependency on old diagnostic Jobs
  `1165781-1165783` prevents old/new overlap. No sealed test, profile, or
  GO/KILL was opened.
- 2026-07-16 13:02+08:00: old Jobs `1165781-1165783` ended `FAILED 1:0`,
  completing invalidation of the `0421a8d9` matrix. Formal Jobs
  `1166361-1166369` released and run at epochs 7-11 with finite losses
  `0.6482-0.7552`; maximum same-batch AMP retry is `2/8`, followed by progress.
  No hard failure is present, pre-gate artifact counts remain zero, and storage
  has about 164 GiB free. No performance conclusion or test opening occurred.
- 2026-07-16 15:08+08:00: all formal S1 cells remain healthy at epochs 24-34.
  Latest finite losses are `0.4371-0.5452`; AMP retry totals are 2-4 with
  maximum depth `2/8`, followed by progress. The hard-failure scan and all
  pre-gate artifact counts remain empty, receipts retain their frozen hashes,
  and `/data` still has about 164 GiB free.
- 2026-07-16 17:09+08:00: all nine formal S1 cells remain `RUNNING` at epochs
  41-52 with finite losses `0.3440-0.4326`. AMP retries remain isolated at 2-4
  per cell and at most `2/8`; the hard-failure scan is empty. Allowed epoch
  41-51 checkpoint/evidence artifacts are appearing without selection. The
  approximately 15% training-monitor values remain explicitly excluded as a
  mismatched-corpus artifact rather than a gate score. No test was opened.
- 2026-07-16 19:10+08:00: dense160 Jobs `1166361-1166363` completed `0:0`
  with ten checkpoint/sidecar/evidence/prediction sets and valid selector
  records. All selected epoch 59 after 4,800 successful updates; gate Avg-mAP
  is `64.739055/64.842109/63.078053` for seeds 3407/3408/3409. Dense224 is at
  epochs 58-59 and dense256 at 52-53. The hard-failure and temporary-file scans
  remain empty. `official_test_read` remains false and no test/profile opened.
- 2026-07-16 19:45+08:00: dense224 Jobs `1166364-1166366` completed `0:0`
  with ten complete candidate artifact sets and valid selectors. Seeds
  3407/3408/3409 selected epochs 57/47/49 with gate Avg-mAP
  `65.695322/63.205058/63.783346`. Dense256 remains healthy at epochs 56-57
  with eight candidate sets per seed. Hard-failure/tmp scans remain empty;
  all completed selections keep `official_test_read=false`, and no test or
  profile has opened.
- 2026-07-16 21:28+08:00: all formal Jobs `1166361-1166369` completed `0:0`
  with ten frozen candidate sets and a valid gate-only selection per cell.
  Dense256 seeds selected epochs 51/57/51 with gate Avg-mAP
  `65.184669/63.316455/64.255928`. Exactly one global test-open certificate
  was published (file SHA `a6d1bf97...`, internal SHA `8627866a...`). Two
  duplicate long-running builder processes were terminated; only the oldest
  process published the canonical certificate and global marker.
- 2026-07-16 21:35-21:53+08:00: explicit 62.2 GB post-job submission was
  scheduler-rejected before a Job ID, then Job `1167230` failed closed before
  test read because host `SLURM_JOB_GPUS` was not the cgroup-local
  `nvidia-smi` selector. Diagnostic Jobs `1167232/1167238` established the
  physical-to-local mapping; full real-certificate/checkpoint preflight Job
  `1167239` passed with no test artifact or profile marker. Adapter SHA is
  `2693cac2...`; it does not change CUDA visibility or model code.
- 2026-07-16 21:53+08:00: submitted serial same-allocation official
  test/trained-checkpoint profile remediation Job `1167257` in the frozen
  nine-cell order. Receipt is `post_test_profile_resubmission_r1.json`.
  Official results, cost analysis, Pro review, and S1 GO/KILL remain pending.
- 2026-07-16 22:55+08:00: Job `1167257` remains healthy and completed the
  first frozen-order official test, dense256/seed3408: Avg-mAP `67.09`, with
  mAP@0.3-0.7 `82.14/77.76/70.36/59.53/45.67`. Canonical evidence file SHA is
  `10c0182...`; `official_test_read=true` and `paper_claim_allowed=false`.
  Its profile and the remaining eight cells are pending, so no GO/KILL or
  resolution selection is permitted.
- 2026-07-17 01:12+08:00: Job `1167257` failed `1:0` in the first profile's
  post-measurement validator with `formal S1 profile window identities must be
  unique`; no later cell started and no profile summary/sample/power artifact
  was published. Exact dataset replay found 792 official loader exposures but
  791 physical identities: `video_test_0001431:7680` is emitted twice by the
  inherited divisible-tail sliding-window behavior. The official workload
  includes both rows, so the uniqueness assertion, not the model or sealed
  test, is wrong. The started marker is retained and silent retry is forbidden
  pending an audited exposure-identity/recovery protocol. No GO/KILL or cost
  conclusion was issued.
- 2026-07-17 03:40+08:00: implemented an immutable post-profile recovery
  campaign without changing model/config/checkpoint/test semantics. Physical
  window identity may repeat; loader exposure identity is unique and
  ordinal-bound. The recovery certificate preserves and hashes the old failed
  marker/log, enforces the exact 792/791/one-duplicate topology, permits only a
  fixed post-processing Git path allowlist, and separates frozen training code
  from repair code. The matrix launcher reuses the one validated test evidence
  and executes the remaining cells once in frozen order on one allocation.
  Local focused tests report `44 passed, 1 skipped`. Deployment remains blocked
  until this implementation is committed, pushed, reconstructed as a clean
  remote snapshot, and passes its formal CUDA preflight.
- 2026-07-17 04:05+08:00: a clean `20b84d2` remote replay failed closed before
  certificate construction because historical bound configs/precheck were
  still validated against the repair checkout. No GPU task or new artifact was
  created. Commit `341cf97` adds exact historical-repository reconstruction:
  recorded source path, Git root, HEAD, clean status, full config matrix and
  precheck identity are all revalidated. Dirty/wrong-commit/wrong-path cases
  fail closed; local S1+C3 tests are `66 passed, 1 skipped`. Deployment remains
  pending a final clean snapshot, certificate and Slurm preflight.
- 2026-07-17 04:42+08:00: `abd1eff` passed 67 remote tests and produced recovery
  campaign `bb56f9d0283b12c0` (certificate SHA `1a0bc133...`). Gate `1167497`
  failed before Python because `/etc/profile` was sourced under `set -u`.
  Gate `1167500` preserved this fix and audited all nine large checkpoints, but
  the real preflight failed because the repair clone lacked the training
  snapshot's ignored relative `data/` mount. Neither Job opened a test/profile.
  Updated the formal launcher to execute repair code from the clean training
  working directory and added a no-open `PREFLIGHT_ONLY` mode. Local regression
  is `66 passed, 1 skipped`; a new certificate/gate is required.
- 2026-07-17 04:50+08:00: `2d988b2` passed 67 remote focused tests and issued
  recovery campaign `10105b8b590cd7fc` (certificate SHA `0f02a64b...`). Its
  no-open Gate `1167504` failed `127:0` in one second because the launcher used
  `python` before activating the OpenTAD environment. No Python preflight,
  test, or profile opened. The failed campaign retains its logs and self-hashed
  submission receipt (`80c230fe...`). Reordered environment activation before
  the first Python call and added a regression assertion; this launcher change
  requires one more clean commit/certificate/gate before any recovery matrix.
- 2026-07-17 05:03+08:00: clean snapshot `04111ad` passed 67 remote tests and
  issued campaign `e647d6feff89cfd7` (certificate SHA `b76fa4af...`). Gate
  `1167507` failed in three seconds before preflight because direct `sbatch`
  relocates `BASH_SOURCE` under `/var/spool/slurmd`, producing a false profile
  code root. No test/profile opened. Updated both formal launchers to require
  an explicit profile-source root and verify its certificate-bound commit and
  clean Git state. The failed campaign is preserved; another clean
  commit/certificate/gate is required before the matrix.
- 2026-07-17 05:32+08:00: `04f8c28` passed 67 remote tests and campaign
  `bc9bacf31bae3749` issued certificate SHA `77caf621...`. No-open Gate
  `1167512` completed `0:0` in `17:09`, validating 211 videos, 792 loader
  exposures, 791 physical windows, the exact duplicate topology, fingerprints,
  and reuse of the existing first-cell test evidence without publishing a
  profile. Submitted exactly one serial frozen-order recovery matrix,
  Job `1167516`, on one Slurm-assigned GPU. Gate/matrix receipt SHAs are
  `820a9721...` and `273bb526...`. Final matrix evidence and S1 GO/KILL remain
  pending; Pro and S2 remain blocked.
- 2026-07-17 07:39+08:00: serial recovery Job `1167516` failed `1:0` in the
  first dense256/seed3408 profile after running all 792 loader exposures. The
  formal cost validator rejected a sparse power trace; no profile summary,
  timing samples, raw power trace, or descriptor was published, and the next
  eight cells did not start. Preserved stdout/stderr/audit/marker file SHAs are
  `5d777a11.../22661dd3.../41079027.../eeac13f9...`. The one prior sealed-test
  result remains unchanged and was not reopened. Added a test-blind Slurm GPU
  cadence diagnostic comparing the inherited persistent `nvidia-smi` sampler
  against native NVML under matched CUDA load, with the existing 20 ms target
  and 100 ms limit unchanged. Local S1 verification is `49 passed, 1 skipped`;
  remote diagnostic evidence is pending, so no reprofile, GO/KILL, S2, or Pro
  review is authorized.
- 2026-07-17 07:58+08:00: test-blind power diagnostic Job `1167536` completed
  `0:0` on clean `7e75b43`. The inherited persistent `nvidia-smi` pipe failed
  the frozen cadence audit (`767` samples; median/P95/max gap
  `20.212/20.331/678.458` ms), while native NVML passed (`511` samples;
  `20.000/20.018/57.709` ms). Diagnostic file/internal SHAs are
  `14c12730.../596568ed...`; receipt SHA is `e32675c3...`. Implemented the
  formal switch to NVML handles resolved by the frozen Slurm-allocated GPU UUID
  with the 20 ms target and 100 ms limit unchanged, plus a chained recovery
  `v2` certificate that must retain the original failure, parent recovery,
  power failure, diagnostic commit, and diagnostic file. This is
  tested infrastructure only: combined local S1+C3 verification is `71 passed,
  1 skipped`; a new clean commit, certificate, no-open GPU gate and matrix
  remain required before S1 GO/KILL.
- 2026-07-17 08:52+08:00: pushed formal NVML commit `2f8eb06`, passed `72`
  exact remote tests, and issued chained campaign `02f8e8bf7c2d6d25`
  (certificate internal/file SHA `e70cccc3.../74ba2f55...`). No-open Gate
  `1167537` completed `0:0` in `20:57` with zero-byte stderr, validated reuse
  of the existing dense256/seed3408 test evidence, and published no profile.
  Its real Slurm mapping was logical `cuda:0` to physical GPU 1; UUID-resolved
  NVML passed at median/P95/max `20.000/20.025/57.848` ms while the inherited
  pipe failed at `674.014` ms. Submitted exactly one frozen-order serial matrix
  as Job `1167538`; it is pending by priority. Gate and matrix receipt internal
  SHAs are `a20341be.../a20768d5...`. S1 remains `experiment_running`; final
  cost/statistics, GO/KILL, Pro and S2 remain blocked.
- 2026-07-17 11:38+08:00: detected that the sole serial matrix Job `1167538`
  had failed `1:0` at `10:03+08:00` in its first dense256/seed3408 cell. It
  collected `107147` UUID-resolved native-NVML samples, but one
  `2413.519286` ms gap exceeded the unchanged `100` ms audit limit. Publication
  failed closed: no summary, latency trace, raw power trace, descriptor, or
  later cell exists, and the reused sealed-test evidence remains unchanged.
  Slurm MaxRSS was `63687456K` under a `62200M` allocation. Code audit found
  that the sampler is still an in-process Python thread; the earlier ten-second
  synthetic Gate cannot certify its long-tail cadence under the full high-RSS
  detector/finalizer path. Recorded all failure hashes and prohibited a silent
  rerun. The next admissible infrastructure step is an out-of-process
  UUID-bound NVML sidecar plus failed-trace preservation and a representative
  long-duration no-open stress Gate, with the 20/100 ms contract unchanged.
- 2026-07-17: implemented the admissible v3 recovery locally without changing
  the S1 model, data, checkpoints, test result, frozen order, cadence threshold,
  or statistical contract. The formal sampler is now a UUID-bound native-NVML
  child process on one dedicated CPU, with four detector CPUs, node-local raw
  trace, immutable attempt evidence, idempotent launcher salvage, and a
  separate parent-failure record when the parent fails after a sealed attempt.
  Added a recursive recovery certificate, full 792-exposure dense256/seed3408
  no-open Gate, formal Gate/hash propagation, and exactly-one serial-matrix
  guards. Local focused verification is `58 passed, 1 skipped`; required C3
  regressions are `20 passed`. State remains `experiment_running`: remote clean
  replay, independent audit, and the long Gate must pass before one replacement
  matrix may be submitted.
- 2026-07-17: independent max-level code audit returned HOLD with no P0 and
  five P1 gaps in the uncommitted v3 recovery: partial attempt publication,
  Gate runtime identity, report/trace closure in downstream consumers,
  concurrent matrix launch, and a mocked rather than real child lifecycle.
  Implemented all five without changing model, checkpoints, test evidence,
  profile order, cadence thresholds, or statistical contract. Gate evidence
  now binds its own actual UUID; matrix cells must match its stable
  hardware/software class and bind their allocation's actual UUID. A shared
  validator recomputes the raw trace hash/cadence in Gate, descriptor, and
  analyzer. Salvage completes only a missing hash-matching counterpart, and
  the matrix uses a persistent atomic lock plus self-hashed start/completion
  receipts. Added Linux real-subprocess lifecycle, early-crash, timeout and
  no-orphan tests. Local S1 result is `61 passed, 4 skipped`; the required C3
  regression is `20 passed`. Remote Linux execution and the full 792-exposure
  Gate remain mandatory before exactly one replacement matrix.
- 2026-07-17: clean remote snapshot `c1253e6` executed all Linux process tests
  before any Gate submission. It produced `84 passed, 1 failed`: on the loaded
  login node, the early-exit Python child had not started within the test's
  `150 ms` window, so the implementation correctly took its timeout path,
  killed the child and sealed a FAIL attempt, while the test expected the
  early-exit label. This is a test-timing defect, not a leaked process or Gate
  result. Raised only the early-exit observation window to 5 seconds; the
  explicit never-ready timeout remains 150 ms. No GPU job or evidence campaign
  was created from `c1253e6`.
- 2026-07-17: timing-fix commit `35c7c5f` passed the exact combined remote
  Linux suite with `85 passed`. A formal one-GPU/96GB request was rejected
  before submission by N16R4's 55GB-per-requested-GPU rule, so no Gate or
  evidence was created. Resource-only Jobs `1168504/1168506/1168509/1168510`
  established that an outer two-GPU/eight-CPU allocation provides `124400M`
  and can host one exact inner step with one GPU, five CPUs, and a finite
  96,000 MiB cgroup limit. Job `1168508` was a no-evidence interpreter-path
  diagnostic failure.
- 2026-07-17: implemented the site-specific execution repair without changing
  the S1 model, data, checkpoints, sealed-test result, profile order, cadence
  threshold, or statistics. Gate/cell/matrix launchers now enter one exact
  one-GPU/five-CPU/96000M Slurm step from the outer reservation, preserve
  Slurm's CUDA mapping, bind identities to `SLURM_STEP_GPUS`, and verify the
  tightest finite cgroup memory limit before evidence access. The second outer
  GPU is recorded as idle scheduling overhead, never model compute. Local S1
  verification is `62 passed, 4 skipped`; remote exact replay, a new v3
  certificate, and the full no-open Gate remain pending.
- 2026-07-17: independent max-level audit of resource-step commit `84a7144`
  returned `HOLD` with no P0 and four P1 lifecycle/evidence gaps. Implemented
  fail-closed Gate preflight ordering, evidence-guarded salvage, strict cgroup
  v2 and Slurm job/step GPU membership checks, logical-CUDA/NVML UUID closure,
  pre-lock in-memory matrix-start validation, per-profile/marker/descriptor
  start-receipt binding, exact-nine completion sealing, and mandatory
  completion validation in the final analyzer. The model, selected
  checkpoints, existing sealed-test result, frozen order, 20/100 ms cadence,
  and statistical thresholds are unchanged. Exact local regression is
  `98 passed, 5 skipped`; this is `tested_local` only. No new campaign, Gate,
  matrix, cost result, GO/KILL, S2, or Pro review has been created.
- 2026-07-17: completed the follow-up evidence-closure repair. The logical
  `cuda:0` UUID now comes from the public CUDA Driver API and must equal the
  step-scoped NVML UUID; all nine cells undergo a no-write dry-run before the
  matrix lock; every new official-test evidence file is bound to the canonical
  matrix-start receipt; the sole historical dense256/seed3408 exception is
  frozen by exact certificate path and hashes; descriptors are canonical-path
  only; and salvage no longer accepts marker-only evidence or suppresses its
  own failure. Exact local verification is `99 passed, 5 skipped`, with Python
  syntax, Bash syntax, and whitespace checks passing. The same independent
  max-level reviewer re-audited the current diff, found no P0/P1, and returned
  `DEPLOY` without modifying files. This authorizes only a clean commit, exact
  remote replay, and a new no-open Gate; no matrix or empirical conclusion is
  authorized until that Gate passes.
- 2026-07-17: pushed final profile/evidence commit `5bfdc36`; its clean remote
  snapshot passed the exact Linux suite with `104 passed`. Issued v3 recovery
  campaign `e3fccb9b12a5d24d`. No-open Gate Job `1168608` completed `0:0` in
  `00:58:39` on `g0041`, reused unchanged official-test evidence, exercised
  `792` loader exposures/`791` physical windows, and published no paper
  profile. Its dedicated NVML sidecar collected `110699` samples with
  median/P95/max gaps `20.000/20.022/63.098` ms, passing the unchanged
  `100` ms limit. After self-hash/artifact validation and an idempotency check,
  submitted exactly one frozen-order serial matrix as Job `1168823`; it is
  `experiment_running`. No final S1 evidence or GO/KILL exists yet.
- 2026-07-17: matrix Job `1168823` failed closed in its first
  dense256/seed3408 profile. The child exited normally and preserved `112107`
  samples, but three cadence gaps exceeded the unchanged `100` ms limit
  (maximum `146.048` ms). No descriptor or later cell was published. Kept the
  v3 campaign immutable and implemented a v4 evidence chain plus an in-memory
  sidecar trace buffer with post-stop atomic JSONL publication; no cadence,
  model, test, resource, or statistical threshold changed. Profiler/Gate now
  expose the trace mode. Combined local verification is
  `101 passed, 5 skipped`; status is `tested_local`. No new campaign, Gate,
  matrix, GO/KILL, S2, or Pro review has been created.
- 2026-07-17: completed a three-pass independent max-level audit of the local
  v4 repair. Pass one returned HOLD with three P1 gaps in official matrix-start
  validation, healthy-child cadence isolation, and parent/current code plus
  trace-mode binding. Pass two returned HOLD for one remaining raw-trace
  lifecycle gap. Fixed all four findings, including the exact relation
  `0 < start <= trace_first == ready_first <= trace_last <= finish`. Added
  rehashed wrong-ready-first and finish-before-last negatives. The exact
  combined suite is `102 passed, 5 skipped`; target negatives, `py_compile`,
  and `git diff --check` pass. Pass three found no P0/P1 and
  returned `DEPLOY` without changing files. State remains `tested_local`; no
  campaign, Gate, matrix, GO/KILL, S2, or Pro was created.
- 2026-07-17: pushed `bc9350e`; its clean remote snapshot passed `107` exact
  Linux tests and issued v4 campaign `6021eaba62337726`. Sole no-open Gate
  `1170341` failed `6:0` after the 96000M preflight and existing evidence hash,
  before any profile/test run or sidecar marker. Resource-only `1170342`
  diagnosed a strict identity-mapping bug: physical `SLURM_STEP_GPUS=1` was
  cgroup-renumbered to the only visible `CUDA_VISIBLE_DEVICES=0`, so
  `nvidia-smi -i 1` returned 6 while `-i 0` returned the allocated UUID. Kept
  the failed campaign immutable. The local repair records both identities,
  queries only the visible selector, and still cross-checks NVML UUID against
  logical `cuda:0`. Combined verification is `102 passed, 5 skipped`; state is
  `tested_local/audit_pending`, with no replacement campaign/Gate/matrix.
- 2026-07-17: independent read-only agent
  `019f70bc-5b43-71c0-9ca6-1122cb880eaf` audited the physical-slot versus
  cgroup-selector repair and returned `DEPLOY` with no P0/P1. It confirmed both
  Gate/formal-profile paths, UUID equality with logical `cuda:0`, provenance
  retention, and regression coverage; it changed no files. State is
  `tested_local`, authorizing only commit, exact remote replay, and one fresh
  no-open Gate.
- 2026-07-18: pushed cgroup-selector commit `43ac70b`; clean remote tests were
  `107 passed`. Campaign `20fe22c380fd38bd` no-open Gate `1170433` passed
  the full 792/791 path with 126218 samples and max gap `67.728` ms. The sole
  matrix `1170468` then completed ordinal-0 dense256/seed3408 and published one
  descriptor, but failed before ordinal-1 test open. The historical training
  snapshot's test guard rejected outer `SLURM_JOB_GPUS=2,4` despite the valid
  exact one-GPU step `SLURM_STEP_GPUS=2`. Preserved the failed campaign and
  lock unchanged; no completion receipt exists and the one cell is diagnostic
  only.
- 2026-07-18: designed and began implementing a step-scoped formal-test
  runtime recovery. It runs the unchanged official test path from a
  certificate-bound infrastructure commit, binds training/runtime commits and
  recovery identity into marker/evidence/descriptor, and recursively seals
  Job `1170468` failure evidence. Model/config/checkpoint/evaluator, sealed
  test, 20/100 ms cadence, resources, frozen order, and statistics remain
  unchanged. Status is `implemented_local`; no replacement campaign, Gate,
  matrix, analyzer, GO/KILL, S2, or Pro is authorized until focused tests and
  independent P0/P1 review pass.
- 2026-07-18: completed local step-scoped runtime recovery verification:
  `107 passed, 5 skipped`, Black, `git diff --check`, and Bash syntax checks
  passed. Independent read-only review returned `DEPLOY_READY_WITH_GATES`
  with no P0/P1. Its P2/P3 findings were closed before commit by narrowing the
  parent-to-v5 diff to nine exact infrastructure paths and requiring one
  structured stdout record to jointly identify ordinal 1/resolution 224/seed
  3409. State advanced to `tested_local`; only a clean runtime commit, exact
  remote replay, real-evidence v5 certificate, and one no-new-test-open Gate
  are authorized next.
- 2026-07-18: pushed runtime commit `6524e1b`; a clean Linux snapshot passed
  `112` exact tests. The first real-evidence v5 build failed closed during
  parent descriptor selection revalidation. Diagnosis showed the login shell
  loaded user-site NumPy `2.2.6`, whereas formal checkpoint selection used
  Conda NumPy `1.23.5`; disabling user-site packages reproduced every stored
  metric exactly. Began explicit `PYTHONNOUSERSITE=1` hardening in the Gate,
  matrix, and per-cell launchers. No campaign was published and no Gate or
  matrix was submitted.
- 2026-07-18: completed environment-provenance hardening. The v5 certificate
  now records and live-verifies the user-site flag, NumPy `1.23.5`, and fixed
  Conda path; offline `verify_checkout=False` validation remains portable.
  Focused verification passed `108` tests with `5` environment skips, and
  final independent review returned `DEPLOY_READY_WITH_GATES` with no P0/P1.
  State remains `tested_local` until this increment is committed, replayed
  remotely, and the real-evidence v5 certificate validates.
- 2026-07-18: pushed environment-pinned runtime `3d01d3b`; its clean Linux
  snapshot passed `113` exact tests. Created and revalidated real-evidence v5
  campaign `3180634880aa8de0` with certificate internal/file SHAs
  `a6d48b7c.../352bd5cc...`. Submitted exactly one no-open Gate,
  Job `1170765`, with receipt internal/file SHAs
  `d2d48a84.../0e1968ee...`. A login-node bare-Python receipt-writing error
  happened after `sbatch`; accounting was reconciled before atomically writing
  the receipt with Conda Python, and no duplicate Gate was submitted. The Job
  is running in the exact `1 GPU / 5 CPU / 96000 MiB` inner step on `g0024`.
  No replacement matrix, analyzer, GO/KILL, S2, or Pro is authorized yet.
- 2026-07-18: Gate `1170765` failed closed `2:0` before sidecar startup because
  profiler/Gate consumers accepted only the literal v4 recovery reason even
  though v5 inherited the exact buffered-sidecar contract. The immutable
  campaign contains only certificate, submission receipt, and stdout/stderr;
  no new test, sidecar evidence, profile, descriptor, or matrix was published,
  and the reused test-evidence hash remains unchanged. Implemented a narrow v6
  capability-family repair and recursive no-open failure certificate.
- 2026-07-18: independent reviewer
  `019f737e-a382-7901-a1a1-1673a98193eb` first returned HOLD with no P0/P1 and
  two P2 test-sufficiency gaps: the alias test did not reproduce the old
  double-stderr dictionary overwrite, and mixed-evidence tests omitted legacy
  and incomplete-role cases. Added double-stderr, hard-link, validator-alias,
  legacy/power mixing, and incomplete Gate/matrix regressions. Focused
  schema-compat verification is `52 passed`; the combined
  S1/train-engine/required-C3 suite is `157 passed, 5 skipped`. Final read-only
  verdict is `DEPLOY_READY_WITH_GATES` with no P0/P1/P2. Status remains
  `tested_local`; clean commit/push, remote replay, and real-evidence v6
  validation remain mandatory before any new Gate.
- 2026-07-18: committed and pushed the independently reviewed v6 runtime as
  `cef95485d1bfebccddb1055f30800ab081decaf7`. This closes local
  implementation provenance only; clean remote replay, real-evidence
  certificate validation, and a new no-open Gate remain pending.
- 2026-07-19: corrected the Spatial Zoom research object after explicit user
  review. The intended method selects native-density spatial crops/ROI tubes
  while preserving the full temporal axis; it is not full-frame downsampling.
  Reclassified the existing dense160/224/256 matrix as `R0 dense-resize
  headroom control`, froze further recovery deployment, and removed it as a
  mandatory prerequisite for crop sufficiency. The next authorized method task
  is a concrete Native-Crop S1 contract followed by one targeted Pro
  code/protocol review; learned ROI remains blocked until oracle/teacher-
  reference crop sufficiency passes preregistered criteria.
- 2026-07-20: archived and independently absorbed the Native-Crop S1 Pro
  review. Raw SHA-256 is
  `7AB0E10624A14FDF2FCABCBEF5EF435EB4994B83BC3E32F3240A3E0143CD44D5`;
  the original verdict is `PROCEED_NATIVE_CROP_S1`. Accepted the core findings
  that current code has no crop, old R0 is split-brain historical control,
  source-coordinate crop must precede resize, the 768/[B,384,768] detector
  contract should remain fixed, sufficiency must precede learned policy, and
  full-stack cost is mandatory. Added project qualifications: a finite
  candidate library is not a route-level oracle; output masked pooling cannot
  undo ViT padding-token mixing; 96/128/knot and numerical GO thresholds are
  provisional; formal teacher/test issues are deferred because the next slice
  uses no teacher and no official test. State remains `designed`. The sole
  authorized next task is a development-only source-geometry census and
  no-training global96/center-local128 vertical slice with source-pixel,
  no-resize, backward, detector-parity, no-leak, and cost-schema tests.
- 2026-07-20: implemented the development-only Native-Crop S1 vertical
  slice. The data path creates a `global96` letterbox and exact source-pixel
  `center-local128` crop immediately after Decord decode, before float/H2D;
  one shared VideoMAE-S instance encodes both views, fixed-mean fusion occurs
  on the 384-point tubelet axis, and deterministic 2x interpolation preserves
  the AdaTAD-derived `[B,384,768]` projection contract. Generic train/test
  entrypoints, teacher, oracle, official test, and paper claims remain
  fail-closed.
- 2026-07-20: completed a development-only source geometry and population
  audit. All 200 available decoded source videos are `320x180`; crop sizes
  96/112/128 require no padding. Census internal SHA is
  `73290dd5abbcac6e5a2da1945b8ebd5b44f2d62e5a570c549aee46679548a9f8`
  and sealed-test files opened is zero. The inherited 0.25 overlap omitted
  `video_validation_0000054`, whose only 0.7-second action is near the end;
  the isolated Native-Crop config now uses 0.5 overlap and closes the frozen
  fit-160/gate-40 population without modifying historical R0.
- 2026-07-20: the remote WIP focused Native-Crop suite passed `11 tests` and
  one real gate sample produced `global [1,3,768,96,96] uint8` and
  `local [1,3,768,128,128] uint8` from source `320x180`, with local source
  box `[96,26,224,154]` and zero padding. This is implementation evidence,
  not crop accuracy or cost evidence. A clean Git-bound CUDA full-model gate
  and independent P0/P1 review remain required before protocol discussion.
- 2026-07-20: the first independent max-agent implementation audit returned
  `HOLD`. It found two P0 defects before deployment: the geometry-census CLI
  argument had been placed on the wrong function signature, and synthetic
  uint8 views remained on CPU after the model moved to CUDA. Both were fixed.
  The same repair cycle froze the exact manifest/file/split hashes, generated
  an immutable 200-record training-only annotation (SHA
  `0985d3711ab31f404ff0be5a1ba75420796a6807d486410337078b38090bf749`),
  asserted fit/gate/development populations `160/40/129/664`, re-derived the
  geometry summary from all 200 source-bound records, matched the normalized
  candidate model/NMS to the reference AdaTAD-derived config, verified the
  complete 163/161/22,482,048 checkpoint/core contract, and added independent
  global/local branch-gradient, 48x8 ordering, exact-2x, checkpointed runtime
  grid, structured collate, and CLI wiring checks. Remote focused verification
  is now `16 passed`; second independent review and formal CUDA gate remain
  pending.
- 2026-07-20: the second max-agent pass returned `HOLD` with no P0 but two
  evidence-integrity P1s: expected commit did not bind working-tree source
  bytes to `HEAD`, and the self-hashed geometry census was not re-probed
  against current videos. It also found one P2 single-axis padding-ratio bug.
  The repair requires a full expected commit, completely clean worktree,
  tracked/byte-equal audited source blobs, and an in-gate re-probe of all 200
  video files including containment, size, geometry, rotation, frame count,
  and frame rate. Intersection-area math and forged/replaced-source negative
  tests were added. Remote focused verification is `17 passed`; the same
  auditor's third pass returned `DEPLOY` with `P0/P1/P2/P3 = 0`. Formal
  clean-commit remote replay and the Slurm CUDA gate remain pending.
- 2026-07-20: committed and pushed the reviewed Native-Crop vertical slice as
  `0bf59be877eeb6879166893641c12bc4e60a2b53`. A clean remote snapshot passed
  `173` focused Native-Crop/Spatial-Zoom/C3 tests. Formal one-GPU Slurm Gate
  Job `1174671` completed `0:0` on `g0059` in `00:01:40`; run root is
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/native_crop_s1_0bf59be_20260720_0225`.
  Precheck internal/file SHAs are
  `ba278a191905b492d78b07ec253857774a0311c363b70ceac8921159a855b0fc`
  /
  `b0cfe61261f39ef801be6b5800510d9feff54b0f0b73babfcc00d091a33bccde`.
  The gate closed source/HEAD provenance, 200-video geometry re-probe,
  `163/161/22,482,048` pretrained-core identity, `6x6/8x8` runtime grids,
  `[1,384,768]` detector input, finite nonzero gradients through both views,
  backbone/projection/head, and zero official-test/teacher/oracle use. This is
  implementation evidence only; crop sufficiency, mAP, measured cost,
  GO/KILL, learned ROI, and paper claims remain absent. The next task is a
  separate crop-sufficiency protocol freeze/discussion.
- 2026-07-20: archived and independently audited the Pro-authored
  `Native-Crop S2 Crop-Sufficiency Preregistration v1`. Raw SHA-256 is
  `E14ABFAB41FAFA3C3F411DF87D3148170872A190C274ED9B7EB2DD44C520C7D5`.
  The project accepts its source-coordinate crop object, same-runtime D160,
  fit/gate and official-test isolation, no-GT raw inference, evaluator parity,
  full-stack cost, and immutable evidence framework, but records
  `ACCEPT_WITH_MAJOR_REVISION / HOLD_IMPLEMENTATION`. Two decision-semantic
  blockers remain: the GT-visible lexicographic reference is not a certified
  library/global-mAP upper bound and cannot kill the library on failure; crop
  sufficiency must be separated from adaptive-selection headroom and
  deployable cost. v1.1 must also move gate GT creation after raw-output
  sealing, split detection and ABBA cost uncertainty by sampling unit,
  distinguish geometric coverage from model-conditioned reachability,
  disclose crop-schedule estimand, reserve future selector cost, and pass a
  result-blind power/Monte-Carlo audit. S2 remains `designed`; no formal
  implementation, queueing, test opening, learned policy, or paper claim is
  authorized.
- 2026-07-20: created the claim-driven Native-Crop paper experiment roadmap.
  It fixes the current position as `S2-P preregistration revision`, separates
  S2 crop sufficiency from the S3 deployable learned policy, and reserves
  official-test main evidence for the frozen final method. The paper-ready
  route now requires a three-seed THUMOS14 accuracy-cost anchor and Pareto
  curve, a TriDet secondary head, an ActivityNet-1.3 dataset census and run
  (with a pre-result FineAction fallback only if completeness fails), minimal
  mechanism ablations, full selector-inclusive cost, and immutable
  result-to-claim closure. Every stage has an explicit stop rule; no later
  experiment is authorized before S2 v1.1 is frozen and its predecessor gate
  passes.
- 2026-07-20: user corrected the final spatial method object. The target is not
  a fixed resolution, fixed source window, or discrete choice among 21
  `128x128` boxes. It is a Uni-AdaFocus-style continuous deformable ROI policy
  that regresses normalized `(cx,cy,w,h)` with variable center, width, height,
  scale, and aspect ratio over temporal groups. Official Uni-AdaFocus and
  AdaFocusV2 sources confirm deformable variable-size patches,
  interpolation-based end-to-end gradients, and the zero-size collapse,
  supervision, diversity, and stability risks. S1 remains valid
  infrastructure; fixed-21 S2 v1 and its v1.1 revision prompt are superseded as
  decisive gates and retained only as D0 controls. The unique next task is a
  continuous-RoI S2 v2 protocol covering differentiable crop, source-coordinate
  inference, box constraints, temporal coherence, matched controls, cost, and
  no-leak semantics.
- 2026-07-20: froze an implementation-targeted Pro prompt for
  `Continuous-RoI S2 Crop-Sufficiency Preregistration v2`, bound to immutable
  commit `6118cd50a3601d044dab690427ad9c756ce7d827`. The prompt consolidates the
  fixed-resolution/fixed-window ambiguity, absence of spatial GT, finite-search
  non-oracle semantics, variable-box anti-collapse and temporal-coherence
  requirements, differentiable/runtime crop parity, joint detector-gradient
  training, matched baselines, separate detection/cost inference families,
  complete cost accounting, no-leak evidence closure, and an exact
  implementation/Slurm output contract. This records a protocol-review request
  only; no Continuous-RoI implementation or experiment is yet authorized.
- 2026-07-20: archived and independently checked the 2,457-line Pro-authored
  `Continuous-RoI S2 Crop-Sufficiency Preregistration v2`. Raw SHA-256 is
  `9ADBD388AD41F79E9323612C25BE493332127B226EB2AA968832D14C5446582B`;
  its protocol-core hash and 894,274 proposed-parameter arithmetic reproduce.
  The project does not adopt the response's `V2_READY` verdict and records
  `ACCEPT_WITH_MAJOR_REVISION / HOLD_IMPLEMENTATION`. Blocking issues are:
  learned-policy S3 machinery is embedded in pre-policy S2; fixed/random/D0/LC
  controls are post-hoc overrides of a variable-box-trained checkpoint;
  `CR-PREF` has unmatched gate-GT privilege; confidence-objective convergence
  does not certify useful continuous spatial coverage; and the one-GPU
  high-memory, 512-GiB storage, 40-ms power-gap, and namespace-deletion rules
  contradict audited infrastructure. The unique next step is a narrow v2.1
  corrigendum; no implementation, queueing, test opening, or paper claim is
  authorized.
- 2026-07-20: froze the project-authored Continuous-RoI S2 v2.1 corrigendum.
  Its machine-readable protocol SHA-256 is
  `ef806b7cd37c704d14a54211b1d4e2f9fb88b75599da918272cc6acad157b3af`.
  The static validator passed eight contract families and exhaustively mapped
  all 128 outcome-state assignments; audit SHA-256 is
  `5af59b755dd4528fe3e4fd989bb20da71ee40e43ecb5add34083b8ae96057f9d`.
  The design trains `D160`, `G96`, and one selector-free common-support `U128`
  per seed, pairs fixed/variable references under equal privilege, and reserves
  the learned ROI policy for S3. This authorizes implementation only. Formal
  training, queueing, official-test access, and empirical claims remain
  blocked until focused tests and a full-model one-step CUDA Gate pass.
- 2026-07-20: committed model implementation `61878997adc4ca3d1de7396a804862d4c6943ee8`
  passed `61` focused tests in a clean remote snapshot. Formal one-GPU
  model-level Gate Job `1177561` completed `0:0`; evidence internal SHA is
  `51b3afec1af0d02197eb660daf719439dfb3c297262f51dde052aafbceecb2ef`.
  It verifies shared VideoMAE with two evaluations, detector-only and total
  gradients, optimizer coverage, `[1,384,768]`, external-geometry rejection,
  pretrained identity, logical `cuda:0` provenance, and zero official-test
  opens. This advances S2 from `implemented` to model-level `tested`; it does
  not authorize empirical claims by itself.
- 2026-07-20: implemented the formal development-training runtime binding for
  `D160/G96/U128 x seeds 3407/3408/3409`: immutable nine-cell configs,
  fit160/gate40 isolation, real-video inventory, exact 80 train batches per
  epoch, 60 epochs/4800 successful optimizer updates, success-only
  scheduler/EMA, AMP replay audit, final-EMA-only checkpoint, completion
  sidecars, one-GPU inner Slurm launcher, and idempotent matrix deployment.
  Local non-Torch checks pass `35`; the broader Windows Torch suite remains
  blocked by the known user-site `c10.dll` initialization error. A clean Linux
  exact suite and a single integrated real-development CUDA Gate are required
  before any of the nine training jobs may be submitted. Official test remains
  sealed and no mAP/cost/paper conclusion exists.
- 2026-07-20: closed the final runtime-authorization audit gap. Authorization
  now reopens each real Gate checkpoint and sidecar, verifies their current
  file hashes and canonical paths, reloads the checkpoint, and checks final
  EMA, optimizer state, scheduler progress, bound-config identity, Slurm
  identity, and work-directory ownership. Deleting, moving, replacing, or
  tampering with any entity fails closed. Local pure-logic regression is
  `36 passed`; Python compilation, both Slurm shell syntax checks, and
  `git diff --check` pass. The independent read-only final audit returned
  `DEPLOY_READY_WITH_GATES` with no P0/P1/P2. This is still a `tested`
  implementation state: clean Linux tests and the integrated real-data CUDA
  Gate remain mandatory before the nine formal jobs can be queued.
- 2026-07-21: clean remote commit `342f4526e6f72a0674d68f49391fe3180191db32`
  passed `70` exact Linux tests. Integrated Slurm Gate Job `1177616` passed
  the v3 full-model CUDA certificate and v2 200-video/nine-config runtime
  precheck, then failed closed before the first D160 optimizer step because
  `mmengine.Config` does not implement `__delitem__`. No runtime checkpoint,
  authorization, formal training job, or official-test access occurred. The
  failed Gate namespace is immutable. The runtime config builder now uses
  `Config.pop`, with a focused regression test; a new commit, clean snapshot,
  and new Gate namespace are required.
- 2026-07-21: clean remote fix commit
  `6b192ffd4f13dc8f1c33574771b70d730c5f83ea` passed `71` exact
  Linux tests. Replacement Gate Job `1177621` passed its v3 full-model
  certificate and v2 runtime precheck, then failed closed on the first D160
  backward because PyTorch 2.0 has no deterministic CUDA implementation for
  `upsample_linear1d_backward`. No runtime checkpoint, authorization, formal
  training job, or official-test access occurred. The D160/G96 comparator
  configs now select the already-audited exact-2x explicit interpolation
  (`384 -> 768`), which is forward/gradient equivalent to
  `align_corners=False` linear interpolation without the nondeterministic CUDA
  backward. The failed Gate namespace remains immutable; another clean commit
  and Gate are required. An independent read-only review found no P0/P1/P2
  in the bound-config-only deterministic replacement and returned
  `DEPLOY_READY_WITH_GATES`; real D160/G96/U128 CUDA backward remains the
  deciding evidence.
- 2026-07-21: deterministic-upsample runtime `6ee8a83` passed its clean Linux
  suite and integrated CUDA/runtime Gate, but its first authorized formal
  matrix failed before training. Jobs `1177641-1177646` failed immediately and
  `1177647-1177649` were cancelled before allocation because a Windows
  carriage return contaminated the exported `YUZIBO_ROOT`, resolving Conda
  under `/data/run01/sczc063/yuzibo^M`. Campaign `66cd32ff...` is frozen as
  deployment-failure evidence; it contains no S2 model result.
- 2026-07-21: commits `eea1f90` and `9a61da27` hardened the formal deployment
  surface: control/whitespace/comma rejection in Python and shell, exact
  canonical root, canonical launcher bound to the expected Git blob, per-job
  pre-sbatch rehash, v2 self-hashed environment/intent/receipt schemas, and
  POSIX behavior tests in the formal Gate. Independent review returned
  `DEPLOY_READY_WITH_GATES` with no P0/P1. The clean Linux exact suite passed
  `81`.
- 2026-07-21: integrated Gate Job `1177662` completed `0:0`. Full-model,
  runtime-precheck, and runtime-authorization SHAs are respectively
  `c633c3b7bc824c2f65800498621904fe307bc9be674375a3849c8c4e815f8c73`,
  `6c5b8f52fe99c2b865ebf8d58db60b579e62c6258a93f6c7276020a7eb077272`,
  and
  `62a0fb21809f9b297337fd17d8440c8c557bfca00ab609c934832abc25846a5f`.
  It validated two successful real-data updates per family, optimizer,
  scheduler, final EMA, exact Slurm identity and zero official-test access.
- 2026-07-21: the sole formal Continuous-RoI S2 development matrix was
  submitted under campaign `77c2149a...`; deployment intent/deployment SHAs
  are `96a805ca...a264` and `67227c44...c204`. Jobs
  `1177668-1177676` bind D160/G96/U128 to seeds 3407/3408/3409. Six tasks
  entered GPU execution and the remaining three waited only on `AssocGrpGRES`;
  all six running tasks completed epochs 0 and 1 and entered epoch 2. Epoch-1
  losses were finite and decreased from epoch 0; no
  Traceback/OOM/non-finite/fail marker was present. Status is
  `experiment_running`, not an empirical S2 result.
- 2026-07-21: the formal matrix reached seven completed cells. D160
  `1177668-1177670`, G96 `1177671-1177673`, and U128/3407 `1177674` all
  completed `0:0`; their self-reported completion records are `PASS` with
  exactly 4,800 successful updates, final-EMA-only checkpoints, matching EMA
  keys, and no official-test access. Final losses were D160
  `0.2190/0.2172/0.2115`, G96 `0.2259/0.2184/0.2219`, and U128/3407 `0.2517`.
  AMP skips were 3-4 per run with maximum same-batch retry 1-2, below the
  registered limit. U128/3408 and U128/3409 remained healthy in epochs 11 and
  9. Exact-nine validation, development inference, reference sweeps, cost
  evidence, and all S2 claims remain pending; status stays
  `experiment_running`.
- 2026-07-21: all nine formal Continuous-RoI S2 training Jobs
  `1177668-1177676` completed `0:0`. Strict live-artifact replay verified every
  bound config, raw/EMA checkpoint state, metadata sidecar and completion
  receipt: each cell has 60 epochs, 80 successful updates/epoch, exactly 4,800
  successful updates and final EMA only. Deployment/completion evidence forbids
  official-test use and no official-test Job, result, or artifact exists, but
  historical training had no syscall-level access instrumentation; no runtime
  zero-open claim is made. No hard training anomaly was found. A training-only
  exact-nine receipt finalizer and focused regressions were added; it cannot
  authorize reference inference or any S2 claim. The checkpoint scheduler
  states close at update 4,800 against a matched inherited 8,000-update cosine
  horizon; these are registered 60-epoch truncations, not completed cosine
  cycles.
- 2026-07-21: a post-training code/protocol audit found that the v2.1 reference
  phase is not safely executable. FS/VS share logits but not physical centers
  under the width/height-dependent decoder; exact Sobol identity, candidate-ID
  no-leak semantics, an annotation-free raw entrypoint, D0, privileged-join
  ties and several statistical definitions are under-specified or absent.
  Reference jobs were deliberately not submitted. Status remains
  `experiment_running`, with verdict
  `TRAINING_MATRIX_COMPLETE / REFERENCE_PROTOCOL_HOLD`; official test and S3
  remain sealed.
- 2026-07-21: the exact-nine training-only finalizer passed a three-round
  independent read-only audit. Intermediate HOLD findings covered injectable
  accounting, executable config loading, incomplete real-model/optimizer
  validation, orphan optimizer state and unrestricted pickle. The repaired
  code now uses live Slurm accounting, a pure-data config decoder, restricted
  `weights_only` checkpoint loading, real detector/optimizer strict loading,
  duplicate/orphan-state rejection and recursive intent/receipt/config/log/Git
  binding. Final verdict is `NO_P0_P1`; the remaining P2 is the pending formal
  Linux replay of all nine real D160/G96/U128 checkpoints. No reference or
  official-test Job was authorized.
- 2026-07-21: formal finalizer Job `1178693` failed closed before publishing
  `training_matrix_completion.json`. Root cause was validator provenance, not
  model state: the historical full-model Gate source hashes were compared with
  the newer finalizer checkout rather than the Gate-bound clean training
  snapshot at `9a61da27`. The repair makes the audited source root explicit;
  all commit and source hashes remain mandatory. Job `1178693` is immutable
  diagnostic evidence and no reference or official-test work was started.
- 2026-07-21: replacement finalizer Job `1178735` failed closed on a raw/EMA
  dtype check. Diagnostic Job `1178737` completed `0:0` and found only
  `module.rpn_head.loss_normalizer` (`float32` raw, `int64` EMA), caused by the
  official head reassigning an integer registered buffer during training. The
  repair emulates the DDP `module.` state prefix, keeps strict key/shape and
  parameter/EMA dtype checks, and records only raw registered-buffer casts
  accepted by the real loader. Neither failed Job produced a matrix receipt or
  authorized reference/official-test work.
- 2026-07-21: all-nine diagnostic Job `1178739` completed `0:0` and confirmed
  that every D160/G96/U128 seed has exactly the same sole dtype mismatch. The
  finalizer repair now classifies all parameter aliases with duplicate removal
  disabled, freezes the only permitted raw-buffer cast to
  `module.rpn_head.loss_normalizer`, and requires the generic and real-model
  mismatch sets to agree. This remains training-receipt validation only.
- 2026-07-21: dtype-audit commit `4543205` passed `83` tests in a clean Linux
  snapshot. Finalizer Job `1178742` exited `2:0` before Python because Slurm's
  `/bin/sh` wrapper rejected Bash-only `set -o pipefail`; no matrix receipt was
  created. The immutable failure authorizes only an explicit-Bash submission
  correction, with no change to runtime commit, evidence or protocol.
- 2026-07-21: explicit-Bash finalizer Job `1178744` completed `0:0` and sealed
  the exact-nine `PASS_TRAINING_ONLY` receipt; read-only replay Job `1178746`
  also completed `0:0` without changing file SHA-256 `14e0fac3...46db`.
  Internal receipt SHA-256 is `9eedfa1e...7dda5`. All real raw/EMA/optimizer
  strict loads passed and the only dtype exception in all nine cells was the
  frozen `module.rpn_head.loss_normalizer` buffer. Reference, official test,
  crop-sufficiency and paper claims remain closed; the sole next action is the
  result-blind v2.2 reference-protocol corrigendum.

- 2026-07-22: recorded a new `designed`, unimplemented Geometry-Residual-Depth
  Routing candidate for offline TAD after the user required a direct comparison
  between continuous ROI, free TokenSelect, A-MoD, and token merging. ROI is a
  structured native-token support prior, not a replacement for arbitrary token
  selection; a residual free-token budget covers disjoint evidence and A-MoD
  routes only later-block depth compute. The required matched matrix is Dense,
  ToMe-only, A-MoD-only, free TokenSelect, ROI-only, ROI+residual, and the
  three-way fusion. A-MoD is recorded as a pretrained-transformer-adaptable
  baseline from arXiv:2412.20875, not as established VideoMAE/AdaTAD evidence.
  No existing S2 receipt, metric, official test, or policy implementation was
  reinterpreted or reopened.

- 2026-07-23: froze the GeoRoute-AdaTAD final-target design and implementation
  matrix in `docs/methods/2026-07-23-georoute-adatad-design-and-experiment-plan.md`.
  The design corrects the depth-router schedule to an initial dense prefix plus
  alternating `Dense -> MoD` pairs: each MoD block consumes the immediately
  preceding dense block's full attention importance. It also records a
  single-heavy-forward soft-support warm-up followed by exact-K sparse
  fine-tuning as a hypothesis requiring direct detector-gradient and matched
  mAP/cost evidence. A Pro audit prompt was recorded beside the design. Status
  remains `designed`; no implementation, deployment, metric, official-test, or
  paper claim was created.

- 2026-07-23: archived the user-provided `GeoRoute-AdaTAD v1 实施裁决`
  attachment (SHA-256 `61A1918B36D811F178152F1E9DE60B464186D9C52678722BA679D617F4468E78`)
  and recorded a point-by-point absorption. The review correctly audits the
  historical baseline commit rather than a GeoRoute implementation, returns
  `HOLD`, and identifies the fixed-resample/two-VideoMAE/no-policy current
  route plus the absence of an independent quality loss. The project accepts
  native tubelet semantics, a one-heavy-forward P0, and alternating
  Dense-MoD. It does not freeze the review's uncalibrated numerical choices or
  make score-function policy gradients, CPU gather, or dense scatter the sole
  final implementation before matched P0 evidence. Status remains `designed`;
  no implementation, deployment, metric, official-test, or paper claim was
  created.

- 2026-07-23: implemented the local GeoRoute native-token vertical slice and
  its result-blind P0/P1/P2/P3 dispatcher. The P0 gate now requires real
  AdaTAD classification/regression losses, exact unique-K native selection,
  an independently counted one-heavy-forward path, required finite gradients,
  zeroed geometry regularizers, dense numerical reference parity, and
  detector-bound score-function evidence. The P1/P2 matrix now includes a
  fixed-lattice plus learned geometry side-channel control, so a gain cannot
  be attributed to token selection when it can be explained by geometry
  injection. Added a conditional theory package and external plot/table
  renderers that preserve raw seed rows and bind outputs to validated record
  hashes. Pure Python focused checks passed `20` tests; local Torch routing
  checks are blocked by the documented Windows `c10.dll` loader failure and
  must be rerun in the remote CUDA environment. Status is
  `implemented_local_pending_cuda_p0`: no mAP, cost, official-test, A-MoD,
  or paper claim exists.

- 2026-07-23: the first remote GeoRoute P0 submission was rejected by the
  N16R4 scheduler before any Slurm job was created because the outer request
  used one GPU with `96G`, while the site grants `55G` per outer GPU. No model
  code, CUDA forward, metric, or P0 JSON ran. The deployment and dispatcher
  now request the site-compliant two-GPU/eight-CPU outer allocation; their
  existing launchers enter a single-GPU/five-CPU/`96G` exact Slurm step for
  all model work. This is a resource-admission repair, not a model or protocol
  change. Status is `implemented_remote_p0_resubmission_pending`.

- 2026-07-23: the second remote GeoRoute P0 admission reached real CUDA
  detector forwards, but jobs `1180859` (dense), `1180860` (hybrid), and
  `1180861` (ROI score-function) all failed before emitting a P0 report.
  The common fail-closed traceback identified an eight-dimensional gather
  index for a seven-dimensional native `[B,T,K,3,2,16,16]` tubelet tensor.
  The P0 finalizer was also rejected because N16R4 requires a GPU declaration
  even for a control-plane job. Both defects are repaired with a shape
  regression test and a minimal one-GPU dispatcher allocation. The failed run
  namespace is diagnostic only; no metric, cost, official test, or P0 claim
  exists.

- 2026-07-23: before resubmission, the N16R4 routing suite exposed two
  additional fail-closed contract defects: uniform/random routes omitted their
  role counts, and the left-associated ST expression could produce a forward
  gate numerically near, but not bitwise equal to, one. The route now records
  those exact-K control roles and computes `1 + (soft - soft.detach())` so the
  hard forward is exact while its surrogate gradient remains attached. This is
  still implementation evidence only; a new remote suite and CUDA P0 must
  pass before any P1 job can exist.

- 2026-07-23: corrected CUDA P0 snapshot `cc35e4b` passed its remote focused
  suite. The site admitted only two P0 leaves: hybrid/ST Job `1180874`
  completed `0:0` with exact `K=32`, one packed VideoMAE forward, real
  AdaTAD `cls_loss`/`reg_loss` backward, finite nonzero scout/router gradients
  and 5.15 GB peak allocated memory. It is a single P0 leaf, not a P0-suite
  or model-performance result. Dense Job `1180873` failed closed before JSON:
  the full-token packed output differed from its dense reference by
  `6.5612793e-4`. Audit found that the debug reference was under `no_grad`,
  whereas the actual detector route was autograd-enabled and CUDA SDPA may
  dispatch a different numerical kernel. The repair retains the `1e-4`
  all-token criterion, runs the debug reference with matching autograd
  dispatch then detaches it, and makes that condition a validated P0 report
  field. Fresh Linux/CUDA verification remains mandatory; P1/P2/P3 stay
  closed.

- 2026-07-23: fresh model-path commit `4a9358d` passed the remote focused suite
  and completed all three independent GeoRoute CUDA one-step leaves: `1180906`
  (dense all-token parity), `1180907` (ROI score-function), and `1180927`
  (hybrid straight-through), each `COMPLETED 0:0`.  The dense report uses a
  matching-autograd numerical reference; hybrid records exact unique `K=32`,
  one packed heavy forward, real AdaTAD `cls_loss`/`reg_loss`, and nonzero
  geometry/residual scout gradients; score-function binds detector losses to
  the geometry scout.  The first P0-only finalizer `1180963` failed closed
  before output because the shell launcher executed a file path that could not
  import `tools`.  Commit `c2a3c69` switched it to module execution, passed
  local and remote focused checks, and replacement `1180966` completed `0:0`.
  It sealed suite SHA
  `a6f8ea041345cdc400c7f8a4f478c037cb66c8cfd3c19edb09d454ff363ce0b1` as
  `PASS_MECHANICAL_ONLY`, without a P1 job, training, evaluator, or official
  test.  This authorizes no performance or paper claim.

- 2026-07-23: implemented the GeoRoute P1/P2/P3 activation repair without
  replaying sealed P0.  `p1-bootstrap` recomputes and hash-validates the three
  P0 reports and `p0_finalization.json`, creates a new P1/P2/P3 namespace,
  submits only the seven one-seed matched P1 cells, and records result-gated
  P2/P3 successors.  The dispatcher action spelling was corrected from the
  invalid underscore form to its CLI hyphen form, control jobs no longer pin
  the N16R4-rejected `--mem=4G`, and package module entry points replace the
  fragile path invocation.  Focused contract checks pass `18/18`; this is
  implementation evidence pending a clean remote snapshot and P1 submission,
  not a development metric or paper claim.

- 2026-07-23: first GeoRoute P1 bootstrap Job `1181007` proved the dispatcher
  can reconstruct the sealed P0 parent, but the scheduler admitted only
  dense leaf `1181008` and rejected the second submission with
  `AssocMaxSubmitJobLimit`. The partial P1 namespace is immutable diagnostic
  evidence, not a P1 matrix. The one admitted leaf reached epoch 0 and
  surfaced a model-path defect that P0 missed: direct 2D `replicate` padding
  is unsupported for a non-16-divisible `[B,3,T,H,W]` video tensor. The
  minimal repair pads flattened NCHW frames and restores NCTHW without any
  resize or coordinate change; an exact 180x320 regression test and a fresh
  180x320 CUDA mechanical gate are now mandatory before a new P1 namespace.
  P2/P3, metrics, costs, official test and claims remain closed.

- 2026-07-23: submitted one new 180x320 hybrid/ST CUDA native-padding gate as
  Job `1181047`. A fresh P1 bootstrap first used Slurm `--test-only`, which
  was rejected with `AssocMaxSubmitJobLimit`; no replacement P1 bootstrap,
  namespace, P1 leaf, P2 leaf, or P3 leaf was submitted. The newly added
  all-leaf admission preflight therefore prevented a second invalid partial
  matrix. The account's unrelated queued/running work must free a slot before
  the sealed P1/P2/P3 DAG can be admitted.

- 2026-07-23: Job `1181047` failed after 21 seconds before emitting a gate
  report. The immutable stderr identifies the exact mechanical cause:
  `replication_pad2d_cuda` is not implemented for the uint8 decoded source
  frames. This is neither a numerical instability nor a model-quality result.
  The minimal replacement preserves source pixels and boundary semantics by
  appending the terminal rows/columns with `torch.cat`; it adds a combined
  bottom-plus-right uint8 regression alongside the 180x320 bottom-only case.
  The three unrecoverable dependency jobs `1180494`, `1180495`, and `1180496`
  were cancelled to release submit quota; no running job was changed. A fresh
  source-bound CUDA gate is required before P1, while P2/P3 remain closed.

- 2026-07-23: the byte-preserving repair passed the fresh 180x320 hybrid/ST
  CUDA Gate `1181172` (`PASS`, report SHA
  `ad362318cc017c234a4ebe5b4d5bbc6c10ffeed33629d15ba8baff5917d02cf3`):
  one heavy forward, exact-K=32, finite detector-to-router gradients, native
  12x20 source grid, and no official test. Bootstrap `1181177` then revealed
  that per-job `--test-only` admission is insufficient for an aggregate
  `MaxSubmitJobs=16` limit: 9 active jobs plus 7 leaves allowed the leaves but
  rejected the eighth selector. Jobs `1181187`--`1181193` were cancelled;
  dense `1181187` ran for 20 seconds and is invalid diagnostic evidence. The
  next minimal repair preflights aggregate headroom and compensates by
  cancelling leaves if any real post-preflight submission fails. The bootstrap
  itself consumes one slot, so its pre-submit condition is at most seven
  pre-existing active jobs. P1/P2/P3 remain unstarted and no result or claim
  changed.

- 2026-07-23: located and audited FlashVID (Fan et al., ICLR 2026 Oral,
  arXiv:2602.08024) and official code snapshot `983cce6`. Its headline is
  verified as a LLaVA-OneVision 10% visual-token retention-budget result with
  57.9/58.4 = 99.1% relative score, not 99.1% accuracy or TAD mAP. The official
  code runs a full vision tower under `torch.no_grad()` before ADTS/TSTM
  compression, and the paper's own efficiency table keeps vision-encoding time
  unchanged. GeoRoute absorbs only its joint relevance-diversity-motion
  correspondence hypothesis, retains P1 unchanged, and relegates a
  scout-only, FlashVID-inspired residual comparator to conditional P2. The
  detailed transfer audit is
  `docs/methods/reviews/2026-07-23-flashvid-literature-absorption.md`.

- 2026-07-27: resumed the existing Zoom+Token route as the frozen
  GeoRoute-AdaTAD P1 screen rather than creating a new method. Remote preflight
  bound clean snapshot `6a9bba6222c18a468c3bd410edac89a4afdea189`, the
  `PASS_MECHANICAL_ONLY` P0 suite SHA
  `a6f8ea041345cdc400c7f8a4f478c037cb66c8cfd3c19edb09d454ff363ce0b1`,
  all required assets, and 3 active jobs under `MaxSubmitJobs=16`. Bootstrap
  Job `1196062` completed `0:0` and atomically submitted dense `1196071`,
  fixed-lattice `1196072`, fixed-lattice-plus-geometry `1196073`, random
  `1196074`, free TokenSelect `1196075`, ROI `1196076`, hybrid `1196077`, and
  result-blind selector `1196078`. All seven leaves reached real Epoch 0
  batches; the selector is held by exact `afterok` dependencies. Shared-node
  `c10d` emitted transient port-bind warnings for two leaves, but both
  subsequently trained and the first audit found zero stderr lines and zero
  fatal-log matches across the matrix. The running namespace is
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_adatad_6a9bba62_p1p3_20260727_222913`.
  This advances the route only to `experiment_running`; no P1 metric, cost
  result, P2/P3 job, official test, empirical support, or paper claim exists.

- 2026-07-28: the complete status audit changed the GeoRoute P1 state from
  `experiment_running` to
  `failed_p1_infrastructure_storage_exhaustion_no_metric`. All seven matched
  leaves `1196071`--`1196077` failed `1:0` after roughly 64--68 minutes with
  `PytorchStreamWriter failed writing file` and `unexpected pos` during
  checkpoint publication. The common namespace consumed 63 GB because
  `checkpoint_interval=1` retained approximately 0.63 GB model/optimizer/EMA
  files for every cell and epoch; JuiceFS `/data` was 100% used with 3.1 GB
  free, while inode usage was only 1%. No P1 result JSON or selector receipt
  exists, and `1196078` remains dependency-held. This is immutable
  storage/deployment failure, not a dense/free/ROI/hybrid comparison. A
  replacement requires a new namespace, aggregate storage-capacity preflight,
  and final-EMA-only or explicitly bounded result-blind checkpoint retention.
  P2/P3, official test, empirical support, and all paper claims remain closed.

- 2026-07-28: performed the user-authorized checkpoint cleanup only after
  validating each GeoRoute P1 cell's highest usable checkpoint with CPU
  `torch.load`. The retained files are dense `epoch_13`,
  fixed-lattice-geometry `epoch_13`, fixed-lattice `epoch_14`, free
  `epoch_15`, hybrid `epoch_14`, random `epoch_15`, and ROI `epoch_15`; every
  retained mapping contains matching `epoch`, `state_dict`, `state_dict_ema`,
  `optimizer`, and `scheduler`. Deleted 107 lower, corrupt, or zero-byte
  `epoch_*.pth` files totaling 62,674,238,335 bytes (58.370 GiB). The failed
  namespace fell from 63 GB to 4.2 GB. Pretrained weights, sealed P0 evidence,
  receipts, configs, and logs were untouched. The retained checkpoints remain
  partial diagnostic artifacts; they do not authorize resume, P1 selection,
  P2/P3, official test, empirical support, or a paper claim.
  A post-cleanup `df` check reported 62 GB available on `/data` (98% used).

- 2026-07-28: completed the user's root-wide epoch-checkpoint retention request
  under `/data/run01/sczc063/yuzibo`. A bound inventory identified 48 remaining
  multi-epoch directories. The dry-run validated each retained highest epoch
  with CPU `torch.load` plus a non-empty model state, found no skipped or
  invalid retained checkpoint, and sealed plan SHA-256
  `e60f00ce6783ac6b858f107fbf06a5aff5d423d7e48b2139fa2412d2beab5e06`.
  Applying that exact manifest deleted 273 lower epoch files totaling
  155,107,184,454 bytes (144.455 GiB) plus 166 matching metadata/temp
  companions. Combined with the prior GeoRoute pass, 380 epoch checkpoint
  files and 217,781,422,789 checkpoint bytes (202.825 GiB) were removed. A second
  bound plan reported zero remaining multi-epoch directories among 278
  still-existing inventoried checkpoint directories and zero deletion
  candidates; `/data` reported 205 GB available (92% used). Pretrained weights,
  `best.pth`, configs, logs, and already-singleton checkpoint directories were
  untouched. The remote plan, apply receipt, and post-verification remain under
  `/data/run01/sczc063/yuzibo/checkpoint_cleanup_manifests/`.

- 2026-07-28: absorbed the exact-commit GeoRoute Pro verdict
  `HOLD_FOR_CORRECTNESS_FIX` and implemented the correctness replacement in one
  local batch. The replacement uses floor-native 176x320 support with an
  explicit validity mask, mask-aware exact-K, a coordinate-lineage packed
  VideoMAE Adapter, fixed full-frame/frozen geometry for the ROI-free `free`
  control, uniform-selected pooling across P1R, branch-aligned hybrid route
  gradients, summed temporal score likelihood, atomic final-only checkpoints,
  and same-commit aggregate storage preflight. P0R now seals a measured storage
  profile and packed-component trace; the result-blind selector first requires
  NativeTokenSelect to beat fixed/random/geometry-side-channel controls and
  cost less than dense, then conditionally tests whether hybrid geometry
  strictly adds. The entire P0R-to-P1R dependency graph is designed for one-shot
  submission, with all seven P1R cells running concurrently after
  `PASS_MECHANICAL_ONLY`. This is parallel scheduling but ordered scientific
  interpretation. The code is native-token evidence routing, not a sequential
  source-pixel crop or resized zoom. Local pure contracts pass, but clean-commit
  remote tensor tests and P0R are pending; status is only `implemented`, with
  no replacement metric, cost result, empirical support, or “Geometry Zoom”
  paper claim.

- 2026-07-28: froze the user-required N16R4 GitHub synchronization rule. Every
  remote `clone`, `fetch`, `pull`, `ls-remote`, or GitHub release download must
  use the login-node academic acceleration proxy in `RTK.md` from its first
  network attempt, then prove expected full HEAD, matching remote-tracking ref,
  and an empty worktree. Direct-first retries and uncommitted source copies do
  not satisfy provenance; proxy failure must fail closed.

- 2026-07-28: synced exact GitHub commit
  `45f5cca2e6b003478327511e3f38c8871b77084f` on N16R4 through the frozen
  academic acceleration proxy and verified identical HEAD/origin ref plus an
  empty worktree. Official-environment GeoRoute focused tests passed `58/58`
  and required C3 regressions passed `20/20`. The one-shot NativeTokenSelect-
  first run root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_nativefirst_45f5cca2_p0p3_20260728_1630`
  passed aggregate storage preflight (214,831,312,896 free bytes versus
  79,456,894,976 required) and submitted dense-parity `1199838`, hybrid-ST
  `1199839`, ROI-score-function `1199840`, and exact-afterok P0R finalizer
  `1199841`. The first audit found three `PENDING(Priority)` leaves and a
  `PENDING(Dependency)` finalizer. Status advances only to
  `experiment_running`; P0R pass, P1R metrics, cost results, empirical support,
  and paper claims remain absent.

- 2026-07-28: replacement P0R dense parity `1199838`, hybrid-ST `1199839`,
  and ROI score-function `1199840` all completed `0:0` in 21--24 seconds;
  finalizer `1199841` completed `0:0` and sealed suite
  `a0073394c3f0f505679797a4c22afeefda1d32adea7b4615e2eec4bbeed35077`
  as `PASS_MECHANICAL_ONLY`. Dense full-token numerical reference passed.
  Hybrid recorded 12 packed attention, MLP, and coordinate-lineage Adapter
  executions, zero dense Adapter executions, exact `K=32`, one heavy VideoMAE
  forward, real detector backward, and finite required gradients. P0R saved
  zero checkpoints and measured a same-commit final-checkpoint upper bound of
  869,336,982 bytes per cell. The finalizer then automatically submitted P1R
  dense `1199865`, fixed `1199866`, fixed-plus-geometry `1199867`, random
  `1199868`, free `1199869`, ROI `1199870`, hybrid `1199871`, and selector
  `1199872`. All seven leaves began as `PENDING(Priority)` and the selector as
  `PENDING(Dependency)`. Status remains `experiment_running`; no P1R metric,
  cost result, empirical support, or paper claim exists.

- 2026-07-28 17:25 CST: P1R dense `1199865` and fixed lattice `1199866`
  changed from priority-pending to running and reached Epoch 2; fixed-plus-
  geometry, random, free, ROI, and hybrid remained `PENDING(Priority)`, while
  selector `1199872` remained `PENDING(Dependency)`. Both active stderr files
  were empty, all logged losses were finite, and no Traceback, OOM, checkpoint
  writer failure, or non-finite loss appeared. Dense had two recovered AMP
  overflow retries; fixed had three retry attempts, all followed by successful
  updates and below the hard-failure threshold. Live memory readouts were
  15,226 MB and 5,331 MB, but are heartbeat diagnostics, not frozen cost
  evidence. No final checkpoint, stage result, selector receipt, or P1R metric
  exists.

- 2026-07-28 17:55 CST: P1R fixed-lattice-plus-geometry `1199867` and random
  `1199868` changed from priority-pending to running. Dense, fixed,
  fixed-plus-geometry, and random were at Epoch 10/9/3/2; free, ROI, and hybrid
  remained `PENDING(Priority)` and selector `1199872` remained
  `PENDING(Dependency)`. All four active stderr files were empty, losses were
  finite, and each arm recorded five recovered AMP retry attempts without
  exhaustion or any Traceback/OOM/checkpoint writer/non-finite-loss signature.
  No final checkpoint, stage result, selection receipt, or P1R metric existed.

- 2026-07-28 18:10 CST: the primary ROI-free NativeTokenSelect arm `1199869`
  changed from priority-pending to running. Dense, fixed,
  fixed-plus-geometry, random, and free were at Epoch 14/13/7/6/1; ROI and
  hybrid remained `PENDING(Priority)` and selector `1199872` remained
  dependency-held. All active stderr files were empty, losses finite, and AMP
  retry counts were 5/5/5/5/2 with successful recovery and no fatal signature.
  No final checkpoint, stage result, selector receipt, accuracy metric, or
  frozen cost result existed.

- 2026-07-28 18:43 CST: P1R crossed an infrastructure-invalid terminal
  boundary. Dense `1199865` and fixed lattice `1199866` completed `0:0`; each
  published exactly one atomic `epoch_19.pth`, zero temporary files, a passing
  storage receipt, and a commit-`45f5cca2` `PASS_DEVELOPMENT_ONLY` stage
  result. Dense recorded Avg-mAP/mAP@0.6/mAP@0.7
  `13.90/11.83/8.74`; fixed recorded `12.42/10.75/7.17`. Both profiles
  explicitly disallow paper-grade end-to-end and paper claims, and the cells
  are diagnostic rather than a matrix verdict.

- 2026-07-28 18:43 CST: free NativeTokenSelect `1199869` failed `1:0` in
  Epoch 8 with no checkpoint or stage result. It overlapped fixed on node
  `g0043`, logged an immediate C10d bind collision on localhost port `29400`,
  and continued only while fixed owned that TCPStore. Fixed logged
  `Training Over` at 18:36:35; free then emitted `Broken pipe` and
  `RendezvousConnectionError` and terminated. Its logged losses were finite,
  its four AMP skips recovered, and there was no OOM or non-finite loss/cost.
  Hybrid `1199871`, colocated with random `1199868` on `g0048`, also logged
  the same bind collision and is protocol-contaminated. Selector `1199872` is
  `PENDING(DependencyNeverSatisfied)` on failed `afterok:1199869`, with no
  selection receipt. Fixed-plus-geometry, random, ROI, and hybrid continue only
  for preserved diagnostics. The failed namespace will not be resumed, no
  candidate will be manually selected, and P2/P3 plus official test remain
  closed. This run neither supports nor refutes NativeTokenSelect or Geometry
  Zoom.

- 2026-07-28 19:10 CST: fixed-lattice-plus-geometry `1199867` and random
  `1199868` completed `0:0`. Each published exactly one atomic final
  checkpoint, zero temporary files, a passing storage receipt, and a
  `PASS_DEVELOPMENT_ONLY` stage result. Fixed-plus-geometry reported
  mAP@0.3--0.7 `16.74/15.56/13.34/10.40/7.09`, Avg-mAP `12.63`, and
  p50/p95/peak-allocated `999.16 ms/4475.58 ms/1817.76 MB`; random reported
  `16.71/15.13/13.28/10.76/7.53`, `12.68`, and
  `1812.36 ms/4806.06 ms/1816.86 MB`. Both used 64 unique of 220 valid tokens
  with zero duplicates, one heavy forward, 12 packed attention/MLP/Adapter
  calls, and zero dense Adapter calls. The profiles cover development-only
  model-and-postprocess execution, exclude the evaluator, include
  same-process loader wait, and do not permit paper-grade full-stack, energy,
  or paper claims.

- 2026-07-28 19:10 CST: hybrid `1199871` confirmed the predicted independent
  rendezvous collision on node `g0048`. Random logged `Training Over` at
  19:04:37; hybrid then lost the shared C10d store and ended `FAILED 1:0` at
  19:04:46 in Epoch 6 with `Broken pipe`/`RendezvousConnectionError`. It
  produced no checkpoint, temporary file, or stage result. Its five AMP retry
  attempts recovered, losses remained finite, and no OOM or non-finite
  loss/cost occurred. ROI `1199870` remains healthy in Epoch 8 with five
  recovered AMP retries. Selector `1199872` remains
  `PENDING(DependencyNeverSatisfied)` with no receipt; no manual selection,
  P2/P3, or official test is authorized.

- 2026-07-28 19:57 CST: ROI-only `1199870` completed `0:0` and closed the last
  running P1 leaf. Its development-only stage result reports
  Avg-mAP `13.18`, mAP@0.3--0.7 `16.66/15.64/13.37/11.28/8.95`, and
  p50/p95/peak-allocated `905.40 ms/4360.95 ms/1818.21 MB`. It selected
  exactly 64 unique of 220 valid tokens per tubelet with zero duplicates,
  straight-through ROI geometry, one heavy forward, 12 packed
  attention/MLP/Adapter calls, and zero dense Adapter calls. It retains one
  final checkpoint, zero temporary files, a passing storage receipt, and no
  GT/teacher/oracle/manual-ROI/raw-prediction-cache use.

- 2026-07-28 19:57 CST: final P1R accounting is five completed
  development-only cells (dense, fixed, fixed-plus-geometry, random, ROI-only)
  and two rendezvous failures without results (free NativeTokenSelect and
  hybrid). Every completed cell has one final checkpoint and no temporary
  file; both failed cells have zero checkpoints and zero stage results.
  Selector `1199872` remains `DependencyNeverSatisfied`, and no selection,
  P2/P3, or official-test artifact exists. The available profiles exclude the
  evaluator, include same-process loader wait, contain no energy receipt, and
  explicitly disallow paper-grade end-to-end claims. The final verdict is
  `NO_SCIENTIFIC_VERDICT_INFRASTRUCTURE_INVALID` for NativeTokenSelect and
  `NOT_AUTHORIZED_NATIVE_BASE_MISSING` for conditional geometry. The run is
  closed as `tested`, not `empirically_supported`; monitoring is complete.

- 2026-07-28 21:56 CST: user approved replacement plan A. A minimal
  infrastructure-only correction is locally implemented without changing the
  model, seven P1R arms, selector, seed, budget, data, initialization, or
  decision rule. GeoRoute train/test leaves no longer use implicit
  `torch.distributed.run --standalone`; they use c10d
  `127.0.0.1:0` and Slurm/job/stage/variant/seed/phase-bound rendezvous IDs.
  Every P0R leaf now runs two concurrent one-rank probes, verifies their
  observed `TORCHELASTIC_RUN_ID`, distinct actual `MASTER_PORT` values, and
  independent lifetimes, then hash-binds that same-leaf receipt into the P0
  CUDA report and final P0 suite. Stage/P0/deployment schemas advance to v3.
  Local compile and non-Torch focused/C3 checks pass `59/59`; the known Windows
  `c10.dll` failure still blocks the two Torch test modules, so authoritative
  Linux Torch and CUDA validation must occur on a clean N16R4 source. No new
  experiment has yet been submitted, the `45f5cca2` namespace remains
  immutable, and NativeTokenSelect/geometry claims remain closed.

- 2026-07-28 22:07 CST: clean source
  `a2ebd0604b4e5648b4f9bc4b3432541fae070393` synced through the academic
  proxy with full HEAD/origin/clean-tree parity and passed remote Linux
  GeoRoute plus required C3 tests `82/82`. New run root
  `georoute_nativefirst_a2ebd060_p0p3_20260728_2202` passed aggregate storage
  preflight (210,791,145,472 free versus 79,456,894,976 required bytes) and
  submitted P0R `1200510`--`1200512` plus finalizer `1200513`. Slurm colocated
  all P0 leaves on `g0003`. All three leaves then failed before any model
  forward or P0 report with the identical fail-closed message
  `GeoRoute rendezvous lifetime isolation was not demonstrated`; the finalizer
  became dependency-unsatisfied and no P1 was submitted. Inspection found a
  gate false negative: fixed 0.5/2.0-second worker durations did not account
  for torchrun parent teardown, so the nominal long worker could finish before
  the short parent returned even with independent stores. The namespace is
  immutable. A deterministic replacement is locally implemented: the long
  worker waits on a controller marker written only after complete short-parent
  exit, after which its continued liveness, distinct observed `MASTER_PORT`,
  exact run ID, and successful completion are required. Model and selector
  contracts remain frozen; no scientific claim changed.

- 2026-07-28 22:15 CST: deterministic-handshake source
  `bfee57904b3919480ce56b72429314eda508bf8e` synced through the academic
  proxy, matched full HEAD/origin/clean-tree state, and passed remote Linux
  tests `82/82`. P0R Jobs `1200550`--`1200552` in new immutable root
  `georoute_nativefirst_bfee5790_p0p3_20260728_2213` failed before model
  execution with `GeoRoute short runtime identity did not match torchrun`;
  finalizer `1200553` was dependency-unsatisfied and no P1 was submitted.
  Read-only Slurm diagnostic `1200560` then ran one real c10d
  `127.0.0.1:0` torchrun and observed dynamic `MASTER_PORT=57695`, exact
  `TORCHELASTIC_RUN_ID=diag-1200560`, and `MASTER_ADDR=g0024`. This classified
  the failure as a validator compatibility error: Torch exports the allocated
  node hostname to workers even though rendezvous used loopback port zero.
  The local gate now records `socket.gethostname()` in controller and probes
  and requires the observed master address to equal that exact node. It must
  pass a dedicated gate-only Slurm job before any third P0 deployment. No
  model, selector, or claim changed.

- 2026-07-28 22:20 CST: exact clean source
  `7be8363ea6e26b320bffafeb03f0e82d8b660779` passed remote Linux tests
  `82/82`. Dedicated gate-only Job `1200602` passed concurrent node-bound
  rendezvous isolation on `g0053`, with exact runtime IDs, distinct dynamic
  ports `54013/34325`, and the long worker alive after complete short-parent
  exit. P0R `1200611`--`1200613` then completed `0:0`; all three same-leaf
  rendezvous receipts and CUDA reports passed. The recomputable P0 suite at
  `georoute_nativefirst_7be8363e_p0p3_20260728_2222/control/p0_finalization.json`
  is `PASS_MECHANICAL_ONLY`, SHA-256
  `693034b276697e92ae915ea5f40cebdd5d01a76bad65f46e5639844654f210e9`.
  Finalizer `1200614` failed only after writing that receipt, before submitting
  any P1 job, because the atomic submit-cap guard observed `active=11`,
  `required_additional=8`, and `MaxSubmitJobs=16`. The namespace is retained
  unchanged and no result claim changed.

- 2026-07-28 22:26 CST: after exact job/path checks, only obsolete GeoRoute
  dependency-dead Jobs `1199872`, `1200513`, and `1200553` were cancelled to
  release submit capacity; DUCA `1181289` and all RIME jobs were not touched.
  Supported sealed-P0-parent bootstrap `1200652` completed `0:0` and created
  fresh root `georoute_nativefirst_7be8363e_p1p3_20260728_2225`. It atomically
  submitted the unchanged seven P1R leaves: dense `1200663`, fixed `1200664`,
  fixed-plus-geometry `1200665`, random `1200666`, free NativeTokenSelect
  `1200667`, ROI-only `1200668`, and hybrid `1200669`, plus automatic afterok
  selector `1200670`. All seven leaves entered `RUNNING` concurrently; initial
  scans found no traceback, OOM, rendezvous error, or non-finite loss/cost.
  State is `experiment_running`, not `empirically_supported`; P2/P3, official
  test, efficiency claim, and paper claim remain closed.

- 2026-07-28 23:59 CST: exact-source P1R reached a terminal protocol failure.
  Dense `1200663`, fixed `1200664`, fixed-plus-geometry `1200665`, random
  `1200666`, free NativeTokenSelect `1200667`, and hybrid `1200669` completed
  `0:0`, each with one atomic final checkpoint, no temporary file, a passing
  storage receipt, and a development-only stage result. Their
  Avg-mAP/mAP@0.6/mAP@0.7 are respectively `13.90/11.83/8.74`,
  `12.42/10.75/7.17`, `12.63/10.40/7.09`, `12.68/10.76/7.53`,
  `10.03/7.80/5.27`, and `13.23/11.35/8.81`. Free is descriptively below
  fixed, random, and fixed-plus-geometry by `2.39/2.65/2.60` Avg-mAP and
  `2.95/2.96/2.60` at mAP@0.6, so the current native selector does not satisfy
  the frozen native-base accuracy gate. Hybrid's gain over free cannot rescue
  or authorize geometry because the base must pass first.

- 2026-07-28 23:59 CST: ROI-only `1200668` completed Epoch 19 and wrote its
  unique final checkpoint, then failed `1:0` in development testing before
  prediction or stage-result publication. Decord raised
  `Unable to handle EOF ... DECORD_EOF_RETRY_MAX=10240` while a DataLoader
  worker retrieved final video frames. Storage preflight passed and there was
  no OOM, non-finite loss/cost, gradient skip, rendezvous error, or model
  failure; classification is development data/video-decode I/O. Selector
  `1200670` became `DependencyNeverSatisfied` with no receipt. The namespace is
  preserved without resume or manual selection; P2/P3 and official test remain
  absent. Because the frozen matrix and selector are incomplete, the available
  six-arm numbers are descriptive diagnostics only. Status is `tested`, not
  `empirically_supported`; no efficiency or paper claim is authorized.

- 2026-07-29 16:30 CST: archived and adjudicated the user-provided CER-TAD Pro
  review. Verdict is `ACCEPT_WITH_MAJOR_REVISION /
  READY_PREEXPERIMENT_ONLY`, not full acceptance. Free v1 is closed as the
  primary candidate on descriptive negative evidence; Hybrid v1, the failed
  namespace, and the old selector are not promoted or reused. Accepted
  mathematical facts are limited to the existing single-family ordered
  Plackett-Luce likelihood and correct risk-minimization sign. Full CER remains
  `discussed` because geometry representation is not isolated and dynamic role
  likelihood, critic, boundary, coverage, stability, and numerical gates are
  underspecified or post-hoc. Froze a new development-only D/K/M
  estimator/representation preexperiment: two-pass exact-index decode census,
  numerical PL/ST plus representation-isolation KATs, and six
  prediction-SHA-preserving diagnostic replays. No training, P2/P3, official
  test, efficiency claim, or paper claim is authorized by these gates.

- 2026-07-29 18:10 CST: implemented the frozen D/K/M preexperiment without
  launching training. Added independent absolute-coordinate,
  ROI-relative-coordinate, and geometry-projection switches; opt-in no-GT
  route telemetry; a two-pass full decode census; PL/ST and representation
  known-answer tests; and six conditional, prediction-SHA-preserving replay
  jobs. A post-implementation audit tightened positive-temperature and CUDA
  profiling guards, remote-path containment, source-artifact binding,
  inference-only `--not_eval`, deployment/result self-hashes, Slurm job
  identity, population consistency, and finalizer failure receipts. Python
  compile, Bash syntax, and `git diff --check` pass locally. State is
  `implemented_pending_remote_test`, not `tested`, `experiment_running`, or
  `empirically_supported`.

- 2026-07-29 16:43--17:30 CST: correction to the preceding prospective
  timestamp: the clean runtime snapshot was already deployed at approximately
  16:43 CST. Remote Linux tests passed `88/88`; the first Slurm test wrapper
  `1203087` failed before Python because `/bin/sh` did not support
  `set -o pipefail`, and corrected Bash Job `1203089` completed `0:0`.
  D/K/M Jobs `1203105`--`1203113` all completed `0:0` under
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_repiso_0c20f2e8_dkm_20260729_1643`.
  The census retrieved 272/272 items over two passes, every estimator and
  representation KAT passed, and all six Phase-M leaves reproduced prediction
  SHA on one 136-window population. Finalization
  `78b0598c70c9966dfd4e7bfa0cce35cfe3ec7d00ed016d0c3268a214e36e86fc`
  emitted `GO_PILOT_DESIGN_ONLY`, with training, P2/P3, official test, and
  paper claim all false.

- 2026-07-29: froze and implemented the independent
  `georoute_estimator_representation_pilot_v1` protocol. The six K=64,
  seed-3407, 20-epoch arms isolate residual PL versus ST with representation
  off, fixed-support representation on/off, ROI-support representation on/off,
  and ROI versus residual support under PL/representation-off. Added an
  independent contract, explicit P0 bindings for all representation switches,
  six-P0 -> six-training fail-closed Slurm DAG, final-only artifact receipts,
  full route/cost telemetry, and a non-promoting exploratory finalizer. The old
  selector, P2/P3, official test, confirmatory margin, and paper claim remain
  closed. State is `implemented_pending_remote_test_and_p0`, not
  `experiment_running` or `empirically_supported`.

- 2026-07-29: completed a post-implementation independent integrity audit of
  the six-arm pilot before deployment. Tightened fail-closed validation by
  recomputing every window descriptor, the complete development population,
  and telemetry summary from raw artifacts; rereading the raw cost profile;
  validating the full immutable arm binding and actual runtime estimator
  hyperparameters; binding train/test to one expected Slurm leaf; requiring
  the canonical P0 suite; and explicitly pinning the D/K/M runtime commit,
  source experiment commit, and finalization SHA. Added the two missing
  experiment graph nodes. Python compile, Bash syntax, `git diff --check`, and
  `34` focused pure contract/finalizer tests pass locally. State remains
  `implemented_pending_remote_test_and_p0`; no training has been submitted.

- 2026-07-29 18:03--18:08 CST: exact clean pilot runtime
  `02b6efe71bd9c62de304467adf0981799eba6b1e` passed remote Linux tests
  `108/108` and submitted six held-then-released P0 leaves
  `1203380`--`1203385` under
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_02b6efe7_20260729_1805`.
  All six failed mechanically before a P0 model report. The two single-tenant
  leaves passed rendezvous and then failed because script-mode P0 had not put
  the repository root on `sys.path` before importing `tools`; the four
  co-located leaves exceeded the old 30-second readiness bound and the old gate
  retained no child diagnostics. No training job ran. After exact state
  verification, only impossible pilot descendants `1203386`--`1203392` were
  canceled; no unrelated job was touched. Finalizer `1203393` then ran but
  exposed a sorted-JSON key-order bug in deployment validation, so no
  `pilot_finalization.json` exists. Deployment file SHA-256 is
  `ee91fa7a5c58371f4d2c57b75038896e87b6a6a9c669492a4fb81c26554529a6`.
  The namespace is preserved as infrastructure-failure evidence and supports no
  performance inference.

- 2026-07-29 18:26 CST: implemented the fresh-source mechanical repair without
  changing any frozen model or experimental factor. P0 now bootstraps the
  repository root before dynamic imports and launches by module. Rendezvous v4
  uses a collision-separated Slurm-job-scoped `127/8` address, kernel-assigned
  port, unique cell/phase ID, 120-second readiness bound, and a hashed bounded
  failure sidecar. Deployment/finalization schemas moved to v2; job maps are
  normalized by exact arm set rather than JSON key order and reject reused job
  IDs. The P0 finalizer and stage wrappers now run `afterany`; a stage wrapper
  requires a sealed PASS P0 suite before cell creation, so failure descendants
  terminate mechanically and the final `afterany` closeout can seal INCOMPLETE
  without performance inference. Same-node capability probes
  `1203460/1203461` proved N16R4 cannot satisfy `--resv-ports=2`, so reserved
  ports were rejected. Python compile and `38/38` focused pure tests pass
  locally. State is
  `implemented_repair_pending_remote_linux_same_node_gate_and_fresh_p0`, not
  `experiment_running`, `empirically_supported`, or `paper_ready`.

- 2026-07-29 18:34 CST: completed an independent pre-commit review of the pilot
  repair and hardened the remaining fail-safe edges. Rendezvous failures now
  terminate the complete torchrun process groups with bounded output draining,
  constrain output to the remote write boundary, and revalidate the receipt
  node from the same P0 leaf. P0-finalizer prevalidation/sealing failures write
  a hashed `pilot_p0_failure.json`; final-closeout validation/sealing failures
  write a hashed `INCOMPLETE_EXPLORATORY_PILOT` receipt before re-raising.
  Added behavior tests for both fail-safe receipts, P0-suite-before-cell
  enforcement, rendezvous timeout sidecar hashing, node mismatch rejection, and
  process termination. Focused pure tests now pass `42/42`; no model training,
  P2/P3, official test, or performance inference was opened.

- 2026-07-29: closed the final local execution-safety review before committing
  the fresh pilot source. The shared stage logger now launches every torchrun
  command in a new session, drains output with a bound, and terminates the whole
  process group on non-zero exit, interruption, or a stuck inherited pipe.
  Remote-write containment now uses structural `Path.relative_to` semantics and
  rejects both the boundary root and prefix-confusion paths such as
  `yuzibo_evil`. Added behavior tests for both properties and registered the
  repair itself as local source evidence pending remote verification. Python
  compile, Bash syntax, `git diff --check`, and the complete focused suite pass
  locally (`65/65`: GeoRoute repair/contract plus the required C3 regression
  files). State remains
  `mechanical_failure_repair_implemented_pending_remote_linux_and_fresh_p0`;
  no training, model result, official test, promotion, or paper claim was
  opened.

- 2026-07-29 18:49 CST: committed repair runtime
  `cbe0a08218a2f4550960f7c832f88c8cf77757c1`, pushed it through the
  `RTK.md` academic proxy, and verified full local/remote HEAD parity plus a
  clean tree. The exact clean N16R4 snapshot passed `118/118` remote Linux
  tests. After two pre-workload launcher diagnostics (one site memory-policy
  rejection before Job creation and `/bin/sh` rejecting Bash `pipefail` in
  Jobs `1203684/1203685`), explicit Bash same-node gate Jobs
  `1203689/1203690` ran concurrently on `g0005` and completed `0:0`. Their
  job-scoped hosts differ and all four actual TCPStore ports differ. A fresh
  six-arm pilot was then deployed under
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_cbe0a082_20260729_1849`.
  P0 Jobs `1203707`--`1203712` and finalizer `1203713` all completed `0:0`;
  suite
  `00b7c0e3251f3d384df91cf900267694918d1245b4a5803150e8e2e1465210d2`
  is `PASS_MECHANICAL_ONLY`. The six frozen 20-epoch leaves
  `1203714`--`1203719` are now running concurrently and closeout `1203720`
  remains dependency-held. State is `experiment_running`, with no metric,
  selector, P2/P3, official test, efficiency, Geometry Zoom, or paper claim.

- 2026-07-29: residual-PL Job `1203715` hard-failed on its first real batch
  after eight AMP retries reduced scale `32768` to `256`; it produced no
  checkpoint or metric. Failure JSON internal/file SHA-256 values are
  `3e5962a3893dc0768c4eea4f0ecd98c8448e4dca8954b71f96c816a46d2f8605`
  and
  `b60ebd0e42ed0b93351343bdf8a2e7c0bb741151a03d35cad2907b8fb7d0990c`;
  `train.out` and Slurm-stderr file SHA-256 values are
  `04cb838679861997fe8f066697c533c1acf004324d7636e814f83c25459dd094`
  and
  `1965f6fc27710d4ff5e76047200389901e5c585c261fed61a1133c1db3d0f88e`.
  The other five leaves continue only to preserve terminal provenance and
  closeout `1203720` must emit
  `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`; no partial contrast is allowed.
  Root-cause audit found the production `T=384/N=220/K=64` Plackett--Luce
  likelihood temporally accumulated in FP16 before GradScaler. Implemented a
  numerical-only FP32 likelihood and unchanged sum-then-batch-mean reduction,
  plus a P0-v4 AMP backward KAT whose objective exceeds FP16 range and whose
  source horizon is bound to decoded `180x320`, floor-native `11x20` support.
  State is `implemented_pending_clean_remote_linux_cuda_verification`, not an
  empirical result, promotion, Geometry Zoom claim, or paper claim.

- 2026-07-29: committed the estimator-equivalent PL AMP repair as
  `30f9ca6fff1572e2eabc6c1b6636c4cc23595a62`, pushed from the first GitHub
  request through the `RTK.md` academic proxy, and verified local HEAD, origin
  ref, clean remote snapshot and clean tree parity. The exact snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_georoute_30f9ca6f_20260729_plamp`
  passed the complete Linux suite `120/120`. Standalone CUDA KAT Job `1203873`
  completed `0:0` on `g0051`; its `T=384/N=220/K=64` check used FP16 source,
  FP32 likelihood/loss, objective `128637.0234375 > 65504`, and finite scaled
  gradients. Receipt internal/file SHA-256 values are
  `7d0ccc346b95180d02a5ddcf4253ac0278e83f39a6f7e434357c86067e3c8e84`
  and
  `75ef280473f5032fd734fb86f1f58207702c1999d34c5c7132d40ff5017ae4a4`.
  This closes only numerical correctness. The old five leaves still run to
  terminal provenance, `1203720` must seal INCOMPLETE, and no fresh performance
  run, P2/P3, official test, efficiency, Geometry Zoom, or paper claim is yet
  authorized.

- 2026-07-29 19:36 CST: completed a fresh history-free agent audit of the raw
  CER-TAD Pro review, its project absorption, frozen estimator/representation
  pilot, no-leak paths, production-horizon PL AMP repair, all-six finalizer, and
  14-job fail-closed deployer. The reviewer returned
  `DEPLOY_AFTER_OLD_CLOSEOUT_AND_CAPACITY`: no further Pro discussion is needed
  before the minimal six-arm exploratory pilot, but the old `cbe0a082`
  namespace must first seal
  `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, Slurm must admit all 14 jobs at
  once, and exact source `30f9ca6f` must pass all six expanded P0 leaves in a
  new namespace. This is code/protocol review evidence only, not a P0,
  performance, CER, Geometry Zoom, P2/P3, official-test, efficiency, or paper
  result.

- 2026-07-29 20:15 CST: the immutable `cbe0a082` pilot reached complete
  terminal provenance without intervention. Jobs
  `1203714/1203716`--`1203719` completed `0:0`; residual-PL `1203715` remained
  the sole hard failure. Afterany closeout `1203720` completed `0:0` and sealed
  schema-v2 `INCOMPLETE_EXPLORATORY_PILOT /
  PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, with
  `all_six_arms_passed=false`, an empty descriptive-contrast object, and
  selector/old-selector/P2/P3/official-test/paper-claim guards all false. Its
  canonical self-hash recomputed exactly as
  `738e9875de2e9e08408263fd7d359e60f5ba1ca1912d0fbb9062a462c58cbf3a`;
  file SHA-256 is
  `63f73a353e356bc77a7a701972f22f62620b35e46b0c8f3eba0fc3c9816db0cc`.
  No five-arm metric, epoch log, or partial checkpoint was interpreted. The old
  closeout gate is satisfied; only all-at-once 14-job capacity and fresh
  schema-v4 P0 remain before exact source `30f9ca6f` may run the replacement.

- 2026-07-29 20:23 CST: the replacement pilot passed both remaining deployment
  gates. With two unrelated active jobs, the all-or-none preflight admitted all
  14 jobs at once (`2 + 14 = MaxSubmitJobs 16`) without splitting the DAG or
  cancelling other work. Exact clean source
  `30f9ca6fff1572e2eabc6c1b6636c4cc23595a62` created
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_30f9ca6f_20260729_2023`.
  Deployment internal/file hashes are `09e837c5...` / `6bdd2742...`. P0 Jobs
  `1204015`--`1204020` and finalizer `1204021` completed `0:0`; suite
  `2aea448be4c8d72957b3c904bb22c5ae39689cb0010c3b18a4914bd71f5265ec`
  is self-hash-valid `PASS_MECHANICAL_ONLY`. The repaired score-function report
  is schema v4 and passed Job `1204016`, `180x320 -> 11x20/N=220`,
  `T=384/K=64`, FP16 source, FP32 likelihood/loss, objective
  `128637.0234375 > 65504`, and finite scaled gradients. All six frozen stage
  Jobs `1204022`--`1204027` run concurrently; afterany closeout `1204028`
  remains dependency-held. No selector, P2/P3, official test, efficiency,
  Geometry Zoom, CER, or paper claim is open.

- 2026-07-29 20:25 CST: residual-PL stage `1204023` hard-failed on real batch 0
  after eight AMP retries, including the scale-256 attempt, with no checkpoint
  or metric. Failure self/file SHA-256 values are `f70b0a54...` /
  `e36556f2...`; traceback SHA-256 is `405b761d...`. Five other stages continue
  only to terminal provenance and closeout `1204028` must seal INCOMPLETE. A
  single fresh history-free independent agent returned `HOLD -> REPAIR` and
  identified the missed contract: P0 ran the full model in FP32, then ran the
  AMP horizon only on isolated logits, so it never tested a scaled full-model
  optimizer update. The six-arm causal design remains frozen. A new source may
  only make an estimator-equivalent precision repair and add a full-graph
  autocast+GradScaler P0; no partial performance, normalization/clipping,
  P2/P3, official test, efficiency, CER, Geometry Zoom, or paper claim opens.

- 2026-07-29: implemented the independent audit's method-neutral repair.
  GeoRoute backbone schema v4 runs the complete low-cost scout/route graph in
  FP32 outside autocast. P0 schema v5 now runs the actual score-function model
  graph under FP16 autocast with GradScaler at the protocol floor `256`,
  unscales and checks required parameter gradients, and requires a successful
  zero-learning-rate optimizer step. The PL objective, estimator weight, arms,
  seed, K, epochs and contrasts are unchanged; normalization and clipping were
  not introduced. Local Python syntax and whitespace pass. Contract tests are
  `59 passed`; one test is blocked only by missing local `mmengine`, while the
  Torch suite cannot collect on Windows because `c10.dll` fails to load. State
  is `implemented_local_pending_clean_commit_linux_cuda`, not a new run or
  result.

- 2026-07-29 20:44 CST: exact repair source
  `c822add335c38a9f6c63e609237c4bfa9b9f468d` passed the complete clean remote
  Linux suite `121/121`. Standalone full-graph CUDA P0 Job `1204087` completed
  `0:0`. Its schema-v5 report is self-hash-valid
  (`4a9cd451...`; file `6dee7330...`) and binds same-leaf rendezvous
  (`74b7563d...`; file `b1e6c336...`). Evidence is
  `180x320 -> 11x20/N=220`, `T=384/K=64`, FP16 source, FP32
  likelihood/loss, `|objective|=128637.0234375 > 65504`, a real
  detector-plus-score-function autocast/GradScaler step at `256 -> 256`, finite
  required gradients, FP32 scout execution, a successful zero-LR optimizer
  update, and zero checkpoints. No metric, official test, selector, P2/P3,
  Geometry Zoom, CER or paper claim opened. This advances only the precision
  repair to `tested`; old closeout `1204028` must seal INCOMPLETE before a
  fresh all-six restart.

- 2026-07-29 21:49 CST: the immutable `30f9ca6f` run reached its required
  terminal closeout. Five surviving stages completed only for provenance;
  residual-PL `1204023` remained failed. Closeout `1204028` completed `0:0`
  and sealed `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, false all-six, empty
  contrasts and all promotion guards false. Finalization self/file SHA-256
  values are `60c9dab5...` / `6ad32b78...`. No five-arm performance was used.

- 2026-07-29 21:52 CST: with the old closeout and full-graph CUDA P0 passed,
  capacity preflight admitted exactly `active 2 + additional 14 =
  MaxSubmitJobs 16`. Exact clean source `c822add3` atomically deployed the
  unchanged six-arm/seed-3407/K64/20-epoch study at
  `georoute_estimator_representation_pilot_c822add3_20260729_2149`.
  Deployment self/file SHA-256 values are `7f445af5...` / `48f19fd8...`.
  P0 Jobs `1204301`--`1204306`, finalizer `1204307`, stages
  `1204308`--`1204313`, and closeout `1204314` are submitted. State is
  `experiment_running`; no metric, contrast, winner, P2/P3, official test,
  Geometry Zoom or paper claim is open.

- 2026-07-29 21:55 CST: all six fresh `c822add3` P0 leaves
  `1204301`--`1204306` and finalizer `1204307` completed `0:0`. The sealed
  schema-v5 suite is self-hash-valid `PASS_MECHANICAL_ONLY`
  (`f6f42367...`; file `114cef2b...`). Residual-PL stage `1204309`
  nevertheless failed `1:0` on real batch 0 after exhausting eight GradScaler
  retries from `32768` through `256`; there was no OOM or rendezvous error.
  Its self-hash-valid failure receipt is `5e556192...` (file `77c9eff7...`,
  traceback `17ec9adb...`), and the cell contains no checkpoint, prediction,
  metric, or stage result. This makes the fresh namespace incomplete and shows
  only that the current synthetic full-graph P0 is not sufficient evidence of
  real-batch AMP stability; it does not decide PL versus ST. Jobs
  `1204308/1204310`--`1204313` continue solely for terminal provenance, and
  closeout `1204314` must emit
  `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`. No resume, arm replacement,
  partial result, selector, P2/P3, official test, Geometry Zoom, or paper claim
  is authorized.

- 2026-07-29 22:47 CST: ROI-PL representation-on Job `1204313`, running only
  for terminal provenance after the namespace was already invalid, logged its
  eleventh AMP gradient skip at batch 111, retry 1/8, scale `64`. This crosses
  the registered `>10` hard-fail threshold. Slurm still reported `RUNNING`;
  logged loss/cost remained finite and no Traceback/OOM appeared. The process
  is not canceled, but neither later completion nor a final checkpoint can
  convert the arm into performance evidence. ROI-PL representation-off
  `1204312` remained at exactly ten skips at this audit point. Closeout
  `1204314` remains required to seal
  `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`.

- 2026-07-29 23:02 CST: ROI-PL representation-off Job `1204312` logged its
  eleventh AMP gradient skip at batch 63, retry 1/8, scale `64`, crossing the
  same registered hard-fail threshold as `1204313`. It remained `RUNNING` with
  finite logged loss/cost and no Traceback/OOM. Both ROI arms are now
  protocol-hard-failed and cannot supply their preregistered representation
  contrast, regardless of later process completion or checkpoint publication.
  They continue only for terminal provenance; no cancellation, partial
  inference, selector, test, or claim is opened.

- 2026-07-29 23:18 CST: correction and terminal closeout. The two preceding
  monitoring entries incorrectly promoted the generic cumulative `count>10`
  alert to a formal experiment hard-fail threshold. Exact source `c822add3`
  instead sets `max_amp_retries_per_batch=8`: the train engine restores and
  replays the same batch after a failed scaled optimizer attempt and raises only
  when one batch exhausts all eight retries without a successful update.
  Therefore the 11 cumulative failed attempts in each ROI-PL arm are
  numerical-stress telemetry (both reached scale `64`), not formal arm
  failures; Jobs `1204312/1204313` later completed all 20 epochs. This
  correction does not open their results, because residual-PL Job `1204309`
  remains the sole formal stage hard failure and breaks the all-six contract.
  Jobs `1204308` and `1204310`--`1204313` all completed `0:0`, each with one
  final epoch-19 checkpoint, one stage result, and zero temporary checkpoints.
  Afterany closeout `1204314` completed `0:0` and sealed schema-v2
  `INCOMPLETE_EXPLORATORY_PILOT /
  PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, `all_six_arms_passed=false`,
  cross-arm consistency false, empty contrasts, and every promotion guard
  false. Its canonical self/file SHA-256 values are
  `a02e551ba9007b49670103e2e4db3bf1c1d917cb5a7bb5c4dd724274b9379a2a`
  /
  `c95c1694dccbda2687b1b9e6e07bb9016ebe80181e2288d172874afa791d8f1c`.
  No five-arm metric, winner, P2/P3, official test, efficiency, Geometry Zoom,
  CER, or paper claim is inferred.

- 2026-07-29 23:45 CST: approved the next real-batch-first diagnosis rather
  than another blind six-arm rerun. The design freezes paired residual-PL/ST
  execution through the production train path, input/RNG fingerprints,
  loss-component telemetry and scaled/unscaled/clipped gradient localization.
  Job `1204309` did not preserve enough state for bitwise replay, so the new
  study is explicitly a deterministic same-config reproduction. Diagnostic
  retries may probe below scale `256`, but produce no checkpoint, metric,
  prediction, evaluator or official-test evidence and cannot become a training
  policy without a separately authorized repair and real-data stability gate.
  The publication boundary is also frozen: the 20-epoch single-seed pilot is
  not paper-comparable; a future claim requires an exact official AdaTAD
  reproduction plus matched native-source dense control, matched
  optimization/EMA/evaluator/NMS, disjoint multi-seed confirmation, sealed
  official test, and selector-inclusive decode-to-NMS latency/memory/energy.

- 2026-07-29: implemented the approved numerical-only real-batch AMP diagnosis
  at exact source `832caedd3713f477cb4b2f29a692acba9cd5a836`. The unchanged
  production training engine now exposes an opt-in observer for input/RNG,
  loss/audit and scaled/unscaled/clipped gradient state; matched residual-PL/ST
  leaves run held and concurrently, followed by one `afterany` finalizer. The
  DAG disables and audits checkpoints, predictions, evaluator, official test,
  metrics and performance inference. Deployment binds the distinct failed
  parent runtime `c822add3` and canonical closeout file hash; stage and wrapper
  failure evidence bind self-hash, runtime, arm, Slurm ID and rendezvous. Local
  pure protocol/train-engine checks pass `50/50`, required C3 regressions pass
  `20/20`, and Python compilation, Bash syntax and whitespace checks pass.
  Clean academic-proxy sync, remote Linux/CUDA validation and all three Jobs
  remain pending. The diagnostic and the single-seed 20-epoch pilot are not
  official-comparable results; a paper row still requires an exact official
  AdaTAD reproduction and same-recipe native-source controls, disjoint seeds,
  sealed test, matched evaluator/NMS, and full decode-to-NMS cost.

- 2026-07-30 00:40 CST: exact clean diagnostic source `832caedd` passed the
  combined remote Linux/Torch suite `98/98` and atomically deployed no-metric
  PL/ST Jobs `1204847/1204848` plus afterany finalizer `1204849` under
  `georoute_real_batch_amp_diag_832caedd_20260730_0040`. Capacity and storage
  preflights passed. Both stages failed symmetrically in six seconds before
  observer construction, data loading or model forward because
  `mmengine.Config` does not support `del cfg[key]`. The finalizer completed and
  sealed `DIAGNOSTIC_INCOMPLETE_NO_REPAIR`, empty performance metrics and all
  checkpoint/prediction/evaluator/test/claim guards false; self-hash is
  `feda83e084ece379faa07e828a88e017e5bb698eba7c78d1c4866c8cd09c77da`.
  No PL/ST numerical conclusion exists. Minimal repair source
  `64d991f96981a3e60b10f47d6d093d5457da9c60` replaces only the unsupported
  deletion with tested `Config.pop` and adds a real Config binder regression;
  local combined checks pass `71/71`. The failed namespace is immutable and a
  clean proxy-synced source plus fresh namespace are required.

- 2026-07-30 00:50 CST: clean `64d991f9` passed remote `99/99` and submitted
  PL/ST/finalizer Jobs `1204864/1204865/1204866`. Both leaves stopped before
  observer construction because the diagnostic treated
  `SlidingWindowDataset.block_list` as included IDs. Finalizer `1204866` sealed
  `DIAGNOSTIC_INCOMPLETE_NO_REPAIR`; self/file SHA-256 values are
  `3de84a8b5260485ca2b583be6a99f4994e92b20378dd3aaf1879de056803acd0`
  /
  `5a7d69afda7745d442de6dde1261123ea7bb771d1fe8dbcbbda748f76f64f37f`.
  Source `047f643f4f78f5a954364d4f9b8e694c93f16079` corrected the receipt to
  Fit-train/Gate-development and separately bound Gate/Fit exclusion lists.

- 2026-07-30 01:00 CST: exact clean `047f643f` passed remote `149/149` and ran
  PL/ST/finalizer Jobs `1204908/1204909/1204910`. Both arms reached one matched
  real batch with identical data/CPU/CUDA RNG hashes and finite forward losses,
  but strict deterministic error mode rejected
  `upsample_bilinear2d_backward_out_cuda` before any optimizer attempt. The
  historical pilot used deterministic warn-only mode, so this is a diagnostic
  execution mismatch, not PL/ST evidence. Finalizer `1204910` sealed
  `DIAGNOSTIC_INCOMPLETE_NO_REPAIR`; self/file SHA-256 values are
  `7755f777d4dbecb3c5024100f0752c3147dc70f81a4d099ba9e77ece6ae6deac`
  /
  `1aea037cda2504f3a4a3a7c57d2628c7242829189722a8c8d1e78a0af838c19f`.
  Candidate `861e9b1edba5baf1b96fe0d4ed1c3c08d1e2da58` explicitly binds and
  verifies the historical warn-only seed policy; local combined checks pass
  `71/71`. No metric, checkpoint, prediction, evaluator, official test, PL/ST
  numerical conclusion, or paper claim exists.

- 2026-07-30 01:07 CST: exact source
  `861e9b1edba5baf1b96fe0d4ed1c3c08d1e2da58` completed the matched PL/ST
  real-batch diagnosis under the historical warn-only seed policy. Jobs
  `1204944/1204945/1204946` completed `0:0`. PL failed nine scaled attempts
  (`65536` through `256`) only in `scout_score_function` and first succeeded at
  `128`; ST had zero failures and succeeded at `65536`. Data/CPU/CUDA RNG hashes
  matched. Finalizer emitted `ROOT_CAUSE_LOCALIZED_REPAIR_AUTHORIZED`
  (internal/file SHA-256
  `3960f747f0c5de9ba9e7de3046812f01f3474c67b63661c8382e78a4647b3c4c`
  /
  `d725e589e315434eca3fd0e0245cffa6e01e1b3490d10b6cae27ec361620a0d0`).
  No performance, checkpoint, prediction, evaluator or official-test artifact
  was created.

- 2026-07-30 01:32 CST: per-tubelet temporal-mean repair source `768e1a30` and
  fail-closed execution source
  `86ff1dde6ddb058ca9250f968972c255f19dab92` passed clean remote GeoRoute checks
  `124/124`, then launched strict stability-v1 Jobs
  `1205033/1205034/1205035`. PL passed two batches at scale `65536` before
  batch 3 produced two nonfinite `residual_head.weight` gradients. ST passed
  twenty batches before batch 21 produced detector-head nonfinite gradients.
  Both violated the preregistered 32-batch zero-skip contract; finalizer sealed
  `STABILITY_GATE_INCOMPLETE_HOLD` (internal/file SHA-256
  `aca065dc4d3dd32325909105ac461a9c32783a133b643bedfcfa8c48b0be1871`
  /
  `d62a017c656975495bb55e7059bd77b080c6f83d49a45a14156620566ea2100e`),
  with all performance/test/claim guards false. Exact official AdaTAD uses
  dynamic GradScaler and does not require zero skip, so v1 is a numerical HOLD,
  not an official-comparability or estimator verdict. A versioned,
  official-semantics, no-metric gate on an independent data order is required
  before freezing any paper experiment.

- 2026-07-30: heartbeat approved the separately versioned
  `stability_official_semantics_v2` design. It freezes diagnostic seed/order
  `4417`, 64 real-data batches, the default dynamic GradScaler, zero
  retry/replay, official scheduler/EMA per-batch transitions, at most two
  nonconsecutive skips, minimum/final scale `16384`, and a completely successful
  final-16 tail. The design is recorded in
  `docs/superpowers/specs/2026-07-30-georoute-official-semantics-amp-stability-v2-design.md`.
  It does not modify sealed v1 and does not authorize metrics, checkpoints,
  official test or a paper claim.

- 2026-07-30: implemented and locally verified the approved
  `stability_official_semantics_v2` contract. The production train engine now
  audits consumed batches, actual replays, scheduler advances and EMA updates;
  v2 alone captures batch-start RNG, rejects non-finite forward losses, uses
  the default GradScaler, tolerates at most the frozen bounded backoff, and
  validates the final stable tail. Deployment additionally requires the sealed
  stability-v1 HOLD, exact origin-ref parity and a hashed official AdaTAD
  reference. Focused diagnostic/v1/v2, train-engine, pilot and required C3
  regressions pass locally. Torch-importing Windows tests remain blocked by the
  documented user-site `c10.dll` failure and must pass on clean Linux before
  Slurm submission. The receipt explicitly says the development scheduler
  hyperparameters and full official recipe are unmatched, so no performance or
  paper claim is authorized.

- 2026-07-30: proxy-resolved and cloned exact v2 runtime source
  `27fba03cb6d4932ee10cb4545b97984dff28c28c`; full HEAD, origin-tracking ref
  and clean tree matched. The remote Linux/Torch suite passed `168/168`.
  Storage/capacity preflight observed `142833188864` free bytes and only
  `2/16` preexisting submissions. A fresh no-resume namespace
  `georoute_official_semantics_amp_stability_v2_27fba03c_20260730_0800`
  released PL/ST Jobs `1205588/1205589` in parallel plus afterany finalizer
  `1205590`; deployment self-hash
  `8c9ec92f927a1f5e7902c3abc0d5422eb8029b544a7b500dd51655daa17e543e`.
  All performance/test/claim guards remain false. This is
  `experiment_running`, not a numerical or model result.

- 2026-07-30: v2 became irreversibly HOLD while still running. PL Job
  `1205588` recorded nonconsecutive `scout_score_function` skips at zero-based
  batches `11`, `20` and `29`; the third backoff changed scale
  `16384 -> 8192`, crossing both the frozen `<=2` skip limit and `>=16384`
  scale floor. ST Job `1205589` had detector skips at batches `20/29` and
  reached `16384`. Both jobs continue naturally so finalizer `1205590` can seal
  canonical terminal evidence. No cancellation, resume, single-arm replacement,
  protocol freeze, performance inference or paper claim is allowed.

- 2026-07-30: sealed the official-semantics stability-v2 terminal record.
  PL Job `1205588` consumed 64 batches, completed 61 optimizer updates and
  recorded three nonconsecutive `scout_score_function` skips at `11/20/29`,
  ending at scale `8192`; ST Job `1205589` consumed 64 batches, completed 62
  updates and recorded detector skips at `20/29`, ending at `16384`. Both
  forwards remained finite, both completed a successful final-16 tail, and
  scheduler/EMA each advanced 64 times with zero retry/replay. Identical
  data/CPU-RNG/CUDA-RNG hashes confirm the bound order and initial stochastic
  state. PL terminated `FAILED 1:0`, ST `COMPLETED 0:0`, and afterany finalizer
  `1205590` `COMPLETED 0:0`, sealing
  `INCOMPLETE_OFFICIAL_SEMANTICS_AMP_STABILITY_V2 /
  OFFICIAL_SEMANTICS_AMP_STABILITY_V2_HOLD`. Finalization internal/file
  SHA-256 are
  `ab7ea3e5fca378532b689f8dce8d3ed57631ca78eec99b91a77a96a5e8e29d56` /
  `c7f59dbcec609430bdf4aafe99cc5ef3272ef93362b7f44ba74bcbc337c85ab0`;
  all self-hashes recomputed and all checkpoint/prediction/evaluator/test/
  temporary artifact counts are zero. This closes only the numerical gate:
  protocol freeze, performance inference, official test, P2/P3 and paper claims
  remain unauthorized.

- 2026-07-30: absorbed the user-provided GeoRoute numerical Pro audit, attachment
  SHA-256
  `22f5802f62689f687667f56ddd6aacb35e07242c213a591cf93a4e50942c6e83`,
  with verdict `ACCEPT_WITH_IMPLEMENTATION_CORRECTIONS`. Accepted the central
  `NEW_MATCHED_DIAGNOSIS_BEFORE_REPAIR` decision, the DDP scaled-bucket FP16-cast
  hypothesis, frozen PL/ST arms, seed `7367`, no-performance boundary, mechanism
  classes, and later two-anchor publication protocol. Corrected the matched RNG
  rule so all-batch data/CPU fingerprints and batch-zero CUDA RNG must match,
  while later CUDA divergence is recorded because only PL consumes Gumbel RNG.
  Added a mandatory same-commit Slurm CUDA/DDP KAT parent.

- 2026-07-30: implemented
  `georoute_pl_gradient_decomposition_diagnostic_v1`. Added an opt-in transient
  wrapper payload; event-connected analytic/actual residual-logit, grouped
  gradient and DDP GradBucket telemetry; a detached shadow matching PyTorch's
  cast-then-divide FP16 hook order; an observer wrapper that returns the
  authoritative hook Future; self-hashed KAT/binding/receipt/stage/deployment/
  finalization contracts; no-resume parallel Slurm leaves and afterany
  finalizer; and focused tests. Hard sampling, ordered likelihood, policy loss,
  temperature, K, baseline, temporal mean, standard hook, old schemas, official
  config, metrics/checkpoints/test surfaces, and paper claims remain unchanged.
  Python compilation passes. Local Torch tests remain unavailable because the
  Windows user-site `c10.dll` cannot initialize; clean remote Linux/CUDA
  validation is the next hard gate.

- 2026-07-30: the first clean remote full `tests/test_georoute*.py` regression
  on candidate `11a67f0b` reported `139 passed, 2 failed`. Both failures were
  legacy unit-test fakes that call `consume_detector_policy_loss` without
  constructing the new opt-in transient payload attribute. No real model,
  gradient, estimator, or receipt path failed. The compatibility correction
  reads the optional payload with `getattr(..., None)` and leaves all production
  and diagnostic semantics unchanged. Candidate `11a67f0b` is not deployable;
  a new exact source must pass the entire suite before the CUDA KAT.

- 2026-07-30: exact candidate `64551eda` passed the clean remote Linux/Torch
  suite `161/161`, then independent CUDA/DDP KAT Job `1207452` failed closed
  before backward. PyTorch DDP rejected the hook registration because postponed
  annotations represented `bucket: dist.GradBucket` as a string rather than the
  required runtime class object. Failure receipt self-hash
  `03ea04a02ed83d5931bfc8843f44aa62914985c6383a65009eab1f0f846b5cdb`
  was sealed with no checkpoint, prediction, metric, evaluator, test or claim.
  The namespace is not reused. The implementation correction removes postponed
  annotation evaluation from the hook module and adds an exact runtime-annotation
  regression; it does not change the bucket, authoritative Future, model,
  estimator, numerical hypothesis, or experiment protocol.

- 2026-07-30: exact candidate `fe6e7816` again passed the clean remote suite
  `161/161`; CUDA/DDP KAT Job `1207459` then passed concrete hook registration
  but failed closed at the KAT-only zero-argument DDP forward. PyTorch 2.0.1's
  DDP wrapper indexes `inputs[0]`, so the zero-argument micro-module raised
  `IndexError` before backward. Failure receipt self-hash
  `4d4a19219974c7b3bbb6c8bc6257626af5a33b3e7baec7f8c2f4c2784d2c1cfd`,
  file SHA-256
  `400e9787a7e11ab2e2c29639a9a9f96cf82ede33dfc0cd13674cbcbd71709ce9`.
  The failed namespace is sealed and not reused. The harness correction passes a
  zero-contribution dummy CUDA tensor to DDP; it cannot change the deliberately
  constructed parameter gradient, hook, bucket observation, or experiment.

- 2026-07-30: exact source
  `664180b6e2645aa3f9bde8b3a67fc7c224b3915c` passed the clean remote
  Linux/Torch suite `161/161`. CUDA/DDP KAT Job `1207467` completed `0:0`;
  receipt internal/file SHA-256 are
  `b4f5f197326c1a6e66315836515bd0de6b9310163499e14cc03a0c499093c47b`
  /
  `3b8f6814496d6a62a09039d707ca04cba08d37a18abc3f78e48596b26a56d7bb`.
  It verified the standard Future, bitwise observer noninterference, intentional
  finite-FP32-to-nonfinite-FP16 cast witness, and analytic gradient direction,
  with no checkpoint/prediction/metric/evaluator/test/claim.

- 2026-07-30: the first diagnostic DAG admission attempt failed before
  namespace creation and before any `sbatch`. The deployer requested parent
  `summary.failed_batch_indices`, but the sealed stability-v2 receipt schema
  uses `summary.skipped_batch_indices`. Added strict sorted/unique/nonnegative
  validation and a focused regression. This changes only copied provenance;
  it cannot affect model execution, gradient observation, estimator, RNG,
  optimizer or the frozen two-arm protocol. A fresh source and same-commit KAT
  are required before deployment.

- 2026-07-30: corrected exact source
  `33f721be83e0ad7f7a36e853491e7a14f148814b` was pushed through the
  academic proxy, cloned through that proxy into a fresh clean snapshot, and
  passed the complete remote suite `162/162`. Full HEAD and origin ref match and
  the tree is clean. Same-commit CUDA/DDP KAT Job `1207480` completed `0:0`;
  receipt self/file SHA-256 are
  `d31d34144e60bdde6103acc36cff42301ba7fbd80a40eb6f04ead63ddb6901b4`
  /
  `09e2ed0ec6f6e3372871ea00f0aa610027bbedd81d098b60ea6a2529aed0e6f4`.

- 2026-07-30: admitted the fresh seed-7367 no-performance mechanism diagnosis
  at root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_pl_gradient_decomposition_v1_33f721be_s7367_20260730_2300`.
  PL/ST Jobs `1207484/1207485` were released together and started in parallel on
  `g0067/g0050`; afterany finalizer `1207486` is dependency-held. Deployment
  self/file SHA-256 are
  `9e038d872f7869cd184a1835827c61c2c3527fa565bc7928ef17824f905297a2`
  /
  `12e785c78f1fdff3d39bd83222836548f3d803990cd1c62862c07ecbcd7ecc66`.
  Initial stderr is empty and all checkpoint/prediction/metric/evaluator/test/
  performance/paper surfaces remain closed. This is `experiment_running`, not a
  performance result.

- 2026-07-30: the gradient-decomposition DAG reached a complete sealed terminal
  state. PL/ST/finalizer Jobs `1207484/1207485/1207486` all completed `0:0`;
  stderr and fatal-signature scans are empty. Both arms consumed 64 matched
  batches with finite forward losses, zero retry/replay and 64 scheduler/EMA
  advances. PL completed 61 updates and skipped at `2/20/29`; all three failures
  had finite analytic/actual residual-logit gradients, finite FP32 pre-hook
  buckets and first became nonfinite in the detached FP16 cast, so every one was
  uniquely `DDP_FP16_CAST_OVERFLOW`. ST completed 62 updates and had
  detector-only FP32 pre-hook failures at `14/29`; these do not explain PL's
  scout-specific cast failures.

- 2026-07-30: finalizer `1207486` sealed
  `COMPLETE_GRADIENT_DECOMPOSITION_DIAGNOSTIC_ONLY /
  PL_NUMERICAL_MECHANISM_LOCALIZED_REPAIR_CLASS_IDENTIFIED`, repair class
  `DDP_FP16_CAST_OVERFLOW`. Finalization self/file SHA-256 are
  `52d4dfd698ed0679a976e6d468fb4b0d1ede9ea630df32f808115c9f118f681e`
  /
  `816819086374f964264d3a8bb4810842f97ef554d5661d2ec4a6b85fd135bc9c`.
  Independently recomputed all deployment/release/receipt/stage/finalizer
  self-hashes; the 22,072,228-byte namespace has zero checkpoint, prediction,
  metric, evaluator/NMS, official-test, latency, energy or temporary artifacts.
  This authorizes only one preregistered repair: disable DDP FP16 compression
  for every matched native arm and run a fresh independent-seed no-performance
  gate. No performance, P2/P3, Geometry Zoom or paper claim is open.

- 2026-07-30: preregistered and implemented
  `georoute_ddp_fp16_cast_repair_gate_v1`. The mechanically derived independent
  seed is `2307`; both matched PL/ST arms inherit the 64-batch
  official-prefix dynamic-GradScaler protocol and thresholds. The only changed
  factor is `solver.fp16_compress=false` in both arms. Added an exact
  gradient-parent admission gate and a same-commit real CUDA/DDP KAT that must
  preserve a finite scaled FP32 gradient of `70000` without a communication
  hook while a detached FP16 shadow overflows. Focused local AMP/repair tests
  pass `32/32`. Remote clean-source validation, KAT and Slurm execution remain
  pending; no performance surface is open.

- 2026-07-30: exact clean repair source
  `685f935e759d5d78f94e5f208997644e07bf4654` passed the complete remote
  GeoRoute suite `145/145`. Same-commit real CUDA/NCCL/DDP KAT Job `1207542`
  completed `0:0` and sealed
  `PASS_DDP_FP16_CAST_REPAIR_CUDA_KAT_ONLY`: no compression hook was
  registered, default FP32 reduction preserved a finite scaled gradient of
  `70000`, its detached FP16 shadow overflowed, unscale remained finite, and the
  optimizer update completed. KAT self/file SHA-256 are
  `257436d617b79413b4b790cda754d6dec56602d52edb07e50c03cdcd28f78b4f`
  /
  `d957514816f660a8eb43b922dfb3325baf36f1bbb706f398d0a54cc0a37df3ae`;
  the KAT emitted no performance artifact.

- 2026-07-30: after full HEAD/origin/clean-tree, parent, KAT, immutable-input,
  capacity and storage admission, deployed the fresh no-performance root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_ddp_fp16_cast_repair_gate_v1_685f935e_s2307_20260730_2314`.
  PL/ST Jobs `1207554/1207555` were released together and are running in
  parallel; afterany finalizer `1207556` is dependency-held. The sole
  intervention in both arms is `solver.fp16_compress: true -> false`.
  Deployment self/file SHA-256 are
  `da8e79727ec4ce758e23d996ac2b238568bee715493dd0e6dec767342e155451`
  /
  `380b85e781691e2956f978b828ba071ffec4192e0df8acaa7529ada9c281f3e0`.
  This is `experiment_running`; performance, official test, P2/P3, Geometry
  Zoom and paper claims remain closed.

- 2026-07-30: repair-gate PL/ST/finalizer Jobs `1207554/1207555/1207556`
  all completed `0:0`. Both arms consumed 64 batches with finite forward losses,
  skipped only batches `20/29`, made 62 successful updates, reached
  minimum/final scale `16384`, and completed the final-16 stable tail with zero
  retry/replay. Data and CPU RNG sequences, initial CUDA RNG, seed and immutable
  inputs match; skip delta is `0` and final-scale ratio is `1.0`. Finalizer
  sealed `COMPLETE_DDP_FP16_CAST_REPAIR_GATE_ONLY /
  DDP_FP16_CAST_REPAIR_GATE_PASS_MATCHED_FORMAL_PROTOCOL_FREEZE_AUTHORIZED`;
  self/file SHA-256 are
  `ad556812454f2ff02161587979ac99c33d9a4983b5c8fcd97d26efe47a936185`
  /
  `f8ef174c934b42ef4efb98e91f16ee4a0a79d0b9f0bbc9c3e174ad3b64bd77e3`.
  Independent validation recomputed receipt/stage/deployment/submission/final
  hashes and classification. The 255,761,847-byte namespace contains no
  wrapper failure, temporary, checkpoint, prediction, metric, evaluator/NMS or
  official-test artifact; logs contain no fatal signature. This validates only
  the registered no-compression numerical repair and authorizes freezing a
  matched formal protocol. Exact official reproduction, performance, official
  test, P2/P3, Geometry Zoom and paper claims remain closed.

- 2026-07-31: froze and implemented
  `georoute_official_comparable_protocol_v1`; no new Pro discussion is needed
  before its admission gate. The immutable external anchor is upstream OpenTAD
  commit `01c58b9f2370e914150cf94d392208a4e211c053` with exact official config
  SHA-256
  `5521b6ce28cc6770e662d3dfdd4621479bc228be6131e300a92285fb4961a49c`.
  Corrected a pre-deployment matchedness error: OpenTAD divides configured
  batch by world size, so config/global batch `2` over two ranks is local batch
  `1`, not global batch `4`. Added the no-performance two-arm real-data F0,
  two-rank FP32-DDP KAT, protocol/finalization contracts, atomic final
  checkpoint sidecar validation, two-rank Fit/Gate telemetry, 15-cell
  dense/fixed/random/ST/PL x three-seed development DAG, and a result-blind
  after-any selector requiring every-seed high-IoU control wins, every-seed
  dense-cost reduction and strict ST/PL accuracy-cost Pareto dominance.
  Geometry, official test, paper-grade efficiency and paper claims remain
  closed. Initial combined official-protocol/AMP/GeoRoute focused tests pass
  `85/85`; exact clean commit, remote Linux/CUDA checks and Slurm F0 remain
  pending.

- 2026-07-31: completed the pre-commit audit of the official-comparable
  implementation. Corrected the world-size-two KAT Slurm request from four
  reserved GPUs to two and restored the pinned AdaTAD deterministic-warn-only
  evaluation semantics. Combined protocol/AMP/GeoRoute/required-C3 checks now
  pass `105/105`, and all four new launchers pass `bash -n`. A wider local
  Windows collection is non-evidence because the host PyTorch installation
  cannot initialize `c10.dll`; the remote Linux suite remains mandatory.
  N16R4 reports `70 GiB` free and `MaxSubmitJobs=16`, with one unrelated
  running job and one unrelated dependency-failed pending job. The conservative
  four-job F0 admission gate has enough storage/capacity; F1 does not and stays
  closed without cancelling unrelated work or weakening its pre-registered
  storage/capacity guards.

- 2026-07-31: pushed exact official-comparable source
  `4a03339b13b0f65047ed0349615889ade06050e8` through the academic proxy and
  verified local and N16R4 HEAD/origin-ref/clean-tree parity. A separately
  proxied upstream snapshot is detached at
  `01c58b9f2370e914150cf94d392208a4e211c053`, with the registered official
  config SHA-256. Remote Linux focused tests passed `105/105` and the complete
  GeoRoute suite passed `153/153`. F0 root
  `georoute_official_comparable_preflight_v1_4a03339b_20260731_1145`
  submitted PL/ST/KAT/finalizer Jobs
  `1209272/1209273/1209274/1209275`; storage passed with 67,006,177,280 free
  versus 47,244,640,256 required bytes.

- 2026-07-31: F0 KAT Job `1209274` failed `1:0` after one second before
  Python/CUDA. Slurm reported `Memory required by task is not available`
  because the outer allocation did not reserve the inner
  `srun --mem=192000M` request. No KAT receipt, checkpoint, prediction,
  evaluator, metric or official-test artifact exists. PL/ST continue only to
  permit after-any finalizer `1209275` to seal the namespace incomplete; their
  outputs cannot substitute for the missing KAT. Implemented a resource-only
  replacement that binds both outer and inner two-GPU KAT steps to `32000M`;
  local focused repair checks pass `23/23`. No model, data, seed, threshold,
  selector or claim rule changed, and F1 remains closed.

- 2026-07-31: the first F0 namespace is now terminally sealed. PL/ST Jobs
  `1209272/1209273` completed their own 32-batch default-GradScaler contracts,
  but cannot substitute for the missing world-two KAT. After-any finalizer
  `1209275` emitted `INCOMPLETE_OFFICIAL_COMPARABLE_PREFLIGHT /
  OFFICIAL_COMPARABLE_PREFLIGHT_HOLD`; its internal/file SHA-256 are
  `f6da6db381260c40e6f90a07203e1eb1c38c50182cfda5b4e3edb0f52ec55cef` /
  `72a910b6ad2d79f895462cc8b0d6dc8c34e85774a810836b76713688e9387ca7`.
  Independent receipt validation found no checkpoint, prediction, evaluator,
  metric, official-test or temporary performance artifact.

- 2026-07-31: correction to the immediately preceding resource-repair entry.
  N16R4's submit Lua rejects every explicit `--mem` override and assigns 55 GB
  per requested GPU. The `32000M` source `5b447255` was therefore rejected by
  `sbatch --test-only` before any replacement namespace or Slurm job existed.
  The final resource-only wrapper requests two GPUs in the outer allocation
  and removes the inner memory override so the KAT inherits that allocation.
  This changes no scientific input or decision rule; a fresh exact-source,
  fresh-namespace all-three F0 PASS is still required before F1.

- 2026-07-31: source `2156811f6cab8c7cbb1882da764010e2ce08f0a9`
  passed remote focused `23/23` and complete GeoRoute `153/153`, but its
  replacement F0 was rejected before namespace creation or Slurm submission by
  the generic training storage gate: `45,628,272,640` free versus
  `47,244,640,256` required. A global bounded scan found only two new
  multi-epoch checkpoint directories, both in the old RIME run. After hashing
  all candidates and CPU-loading both epoch-59 keepers, ten epochs
  9/19/29/39/49 totaling `5,176,692,497` bytes were deleted. Postverification
  reloaded both keepers and wrote receipt
  `ad5396edb82e1724d89979a5d495d485a799299b1b81c63a7f9566ac87deafed`.
  No singleton/final/best/pretrained/data/cache/receipt/config/log was removed.

- 2026-07-31: implemented a versioned F0-only no-artifact storage contract in
  source `3d8c2b487fa983d6d6240b347177cc423a37748b`. F0 forbids checkpoint,
  prediction, metric, evaluator and official-test outputs, so it reserves
  512 MiB per leaf, 1 GiB shared overhead and 24 GiB filesystem safety instead
  of impossible checkpoint copies. The first sealed F0 occupied about 129 MB.
  F1's conservative 15-cell training storage contract is unchanged. Academic-
  proxy synchronization verified full HEAD/origin-ref/clean parity; remote
  focused tests passed `25/25` and all GeoRoute tests passed `154/154`.
  Fresh root
  `georoute_official_comparable_preflight_v1_3d8c2b48_20260731_122316`
  submitted PL/ST/KAT/finalizer Jobs `1209309/1209310/1209311/1209312`.

- 2026-07-31: the replacement F0 passed completely. All four jobs completed
  `0:0`; PL/ST each consumed 32 matched batches, recorded one default-scaler
  skip, ended at `32768`, and passed a final-16 stable tail. World-two default
  NCCL FP32 reduction/update passed and the detached FP16 shadow overflowed as
  required. No forbidden artifact exists. Finalizer emitted
  `PASS_OFFICIAL_COMPARABLE_PREFLIGHT_ONLY /
  FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED`; internal/file SHA-256 are
  `313da95faeae9e600965fe4ac5c7ad5816f652d5ff2c97cf9734f7028d888a3c` /
  `22f5dcab4c19d843bc807c5dd60e5f97605378617f67d3a3f507a7a768c57679`.
  This authorizes F1 execution only. F1 remains unsubmitted because current
  admission is `active=2, required=16, limit=16` and storage is
  `31,646,543,872` free versus `130,996,502,528` required; no unrelated job is
  cancelled and neither gate is relaxed.

- 2026-08-02: fully absorbed the 1,101-line Hybrid-centered Pro review
  (`b1a39b0869d03b50de9743df388c01665496ffebbb63bcb22b2efe908b196133`)
  and accepted `RUN_HYBRID_CAUSAL_PILOT_FIRST` with explicit evidence limits.
  The old ROI+TokenSelect Hybrid result remains descriptive motivation rather
  than causal proof; Fixed is defined as the deterministic row-major uniform
  K64 coverage control; geometry shift is temporal trajectory shift127; and a
  future PL-versus-ST claim must retain matched ST across confirmatory seeds.
  No additional Pro discussion is required before the frozen mechanical gate.

- 2026-08-02: implemented the development-only Hybrid causal route and
  fail-closed execution chain: exact K64 structured context8/ROI28/residual28
  conditional Plackett-Luce sampling with a route-private RNG, exact
  `cls_loss + reg_loss` policy risk, per-role telemetry and branch-gradient
  hooks, temporal geometry-misalignment control, nine frozen arms A0--A8,
  separate accuracy/telemetry and cost replays, all-terminal finalizers, and
  held all-or-none Slurm deployment. The P0 suite combines an A7 production
  forward/backward gate with the existing two-rank FP32-DDP KAT. New outer
  `sbatch` requests intentionally omit explicit `--mem` flags because site Lua
  assigns memory per GPU and rejects overrides; the pre-existing P0 launcher's
  conditional one-GPU inner-step compatibility branch remains unchanged and is
  not entered by this study's exact one-GPU outer allocation. Local
  pure-contract tests pass;
  Windows Torch is unavailable because `c10.dll` cannot load. Remote Linux/CUDA
  verification and P0 remain pending, so the route is only `implemented`, not
  `tested`, `empirically_supported`, or `paper_ready`.

- 2026-08-02: remote validation exposed and repaired one compatibility
  regression before any model execution. Source `a9e1f742` passed `170`
  GeoRoute tests with one skip but failed the inherited one-GPU inner-step
  memory-clause contract; its namespace remains failed provenance. Replacement
  source `0f64218d` restored the clause and passed pycompile, required C3
  `20/20`, complete GeoRoute `171/171` with one skip, and the real data/config
  binder SHA
  `202d8d75b024ae6f080caba461ba05c33edd0790b99d683c9751b4f449f2e78d`.

- 2026-08-02: the no-performance Hybrid P0 completed. Jobs
  `1213665/1213666/1213667` all exited `0:0`; final status is
  `PASS_MECHANICAL_ONLY`, suite internal/file SHA-256
  `6e86e8fec966dc9440140e746d3436926ce764f825f7081ad387b24dce9c8e71` /
  `142d0e64f67ddc5c4c72ff41e6cf2d346f4688e101d0116b4402c2c4905d1762`.
  It verified exact context8/ROI28/residual28 roles, distinct private role RNG,
  no global-RNG consumption, finite nonzero ROI/residual branch gradients, and
  default world-two FP32-DDP reduction/update. It authorizes performance
  training only and contains no checkpoint, prediction, or metric.

- 2026-08-02: released the held all-or-none nine-arm single-seed screen at exact
  runtime `0f64218d`. A0--A8 are Jobs `1213694--1213702`; all-terminal finalizer
  is Job `1213703`; deployment receipt SHA-256 is
  `6dc3abebdb662393dc6faa8eac4bdf052622c0ea041a2c7eec8f232365c2b3f9`.
  Storage and submission gates passed without cancelling unrelated work. The
  experiment stage is now `experiment_running`; no survivor, performance,
  cost, mechanism, official-test, or paper inference is open before sealed
  all-complete finalization.

- 2026-08-02: user corrected the route scope: fixed
  context8/ROI28/residual28 and exact K64 do not satisfy the original final
  objective. The final target is a temporally adaptive continuous ROI window
  together with adaptive token quantity. The running nine-arm study remains a
  matched-budget causal probe only: it can determine whether ROI/residual roles
  merit inclusion, but cannot promote fixed quotas as the final model. Current
  code already predicts per-tubelet `(cx,cy,w,h)`; dynamic token allocation is
  not implemented. One design question remains open: dynamic total `K_t` versus
  fixed total K with only a dynamic role split. No code, protocol, or running
  job was changed before that clarification and design approval.

- 2026-08-02: user resolved the dynamic-budget ambiguity: total heavy-token
  count `K_t` must vary over time, not merely the ROI/residual proportions under
  a fixed K64. Consequently the final design must jointly express a dynamic
  total budget and dynamic evidence-role allocation while recording actual
  executed K and measured cost. Temporal decision granularity remains the next
  design clarification; no implementation or running-pilot mutation was made.

- 2026-08-02: user accepted the native two-frame VideoMAE tubelet as the
  dynamic-policy unit. The intended model therefore makes 384 continuous ROI,
  total-`K_t`, and role-allocation decisions over each 768-frame window; it does
  not require 768 independent raw-frame heavy-token budgets or a new patch
  embedding. The remaining budget question is whether the sum over all 384
  `K_t` values is a hard per-window budget or may vary by window under a
  dataset-level cost constraint. No implementation or running job was changed.

- 2026-08-02: user selected the recommended staged budget design. Dynamic
  Stage 1 will enforce an exact configurable per-window total
  `sum_{t=1}^{384} K_t=B` while learning tubelet and role redistribution. Only a
  passing matched-budget stage may authorize a second design in which total B
  varies by window under an explicit cost contract. Direct expected-K or
  Lagrangian training is therefore out of scope for Stage 1. No code or running
  pilot was changed before the remaining architecture questions and design
  approval.

- 2026-08-02: user required context allocation to be fully dynamic as well.
  The intended Stage-1 policy therefore allocates
  `(K_context,t,K_ROI,t,K_residual,t)` jointly under the hard window budget and
  retains no deterministic context floor. A fixed uniform scaffold or post-hoc
  coverage repair is prohibited. Whether total `K_t` may equal zero at a
  tubelet remains the next explicit design question; no code or running job was
  changed.

- 2026-08-02: user allowed `K_t=0` in Dynamic Stage 1 and requested a settings
  ablation. The main design may therefore skip heavy tokens for a tubelet; a
  separately trained matched `K_t>=1` setting will test the value of full heavy
  temporal coverage. Zero-K execution may not silently select a fallback token,
  and any cheap detector carrier must be separately identified from actual
  heavy K. The exact zero-K detector representation remains the next design
  question; no implementation or running pilot was changed.

- 2026-08-02: user selected `masked zero carrier` as the Dynamic Stage-1 main
  representation for `K_t=0`: the heavy feature is zero, an explicit
  heavy-valid mask marks the absence downstream, no content-bearing scout/null
  substitute is inserted, and the carrier is excluded from executed-heavy-K
  accounting. `learned-null` and `scout-projection` are retained only as
  independent, separately trained ablations; they cannot be inference-time
  switches or share the main checkpoint's interpretation. This decision is
  `designed`, not implemented or tested; no running pilot was changed.

- 2026-08-02: the fixed-budget Hybrid causal pilot became terminal but not
  interpretable. All A0--A8 Jobs `1213694--1213702` completed `0:0` and nine
  stage results/final checkpoints exist. After-any finalizer `1213703` failed
  `1:0` after two seconds and atomically sealed
  `FAIL_UNTRUSTED_FINALIZER_INPUT` with empty contrasts (internal/file SHA-256
  `29a97472e0358e3379d7ce8b217eefb9368a2a7b3ebe0cfb493fa840fc66ebdd` /
  `42f83bfa264e92f1fcb8bfaa8e4a0b2c586c2b4b464de12f4f80424fa71fcd7c`).
  Read-only predicate replay proved a single deployment-validator defect:
  canonical JSON sorted the `jobs.stages` mapping, while the finalizer compared
  mapping insertion order to frozen arm order. Schema, commit, explicit
  `arm_order`, canonical hash, exact stage-key set, finalizer/dependency Job
  bindings, no-partial and no-test guards all passed. No metrics were read. The
  old namespace remains immutable; only a new versioned, hash-bound recovery
  finalizer may validate and interpret the complete population.

- 2026-08-02: user approved the recommended Dynamic Stage-1 allocator family:
  global constrained exact-`B` projection over all 384x220 physical native
  tokens. A physical patch may receive at most one of the dynamic
  context/ROI/residual roles; the selected union induces integer `K_t` and role
  counts, including `K_t=0`, without fixed quotas, a rounded count head, or
  post-hoc coverage repair. Code inspection found that the current native
  packed VideoMAE accepts only rectangular `[B,T,K]` indices and requires equal
  selected counts across chunk batches. Therefore the global allocator is
  `designed` only: its ragged execution section still requires user approval,
  and dummy/padded heavy tokens may not be mislabeled as exact-B compute. No
  model code or experiment was changed.

- 2026-08-02: user approved the full-window exact-`B` execution and independent
  cost-accounting contract. Exact B now means B unique physical tokens are
  selected, patch-embedded, and heavy-executed; it is not an equal-FLOPs claim.
  The main method retains full-window temporal adaptivity instead of enforcing a
  fixed quota inside each native 16-frame clip. Every route must separately
  receipt per-clip executed counts `b_c`, attention-pair cost
  `P=sum_c b_c^2`, actual ragged bucket and patch-embedding/attention/MLP/
  coordinate-lineage-Adapter calls, plus measured p50/p95, and must pass a
  cost/Pareto gate. Padding or dummy tokens cannot satisfy the exact-B ledger.
  This contract is `designed`, not implemented or tested; no model code, old
  pilot namespace, or running experiment was changed.

- 2026-08-02: user questioned whether an independent dynamic context utility
  `q_ctx(t,n)` is necessary and requested a Uni-AdaFocus training-method audit.
  Primary paper and official code commit `88464883` show that Uni-AdaFocus does
  not use PL/RL/ST for its current selector: it conditions policies on detached
  cheap global features, trains continuous spatial geometry with a
  deep-feature-interpolation auxiliary task loss, and trains temporal weights
  with a differentiable Monte Carlo expected-loss decomposition; hard focus
  indices and crop actions are detached from the heavy path. The resulting
  project recommendation is only `discussed`: replace a separately symmetric
  context head with one shared base physical-token utility plus ROI/residual
  modifiers, so context is the zero-modifier role and its selected count remains
  dynamic. Consider a Uni-inspired stop-gradient coarse-feature surrogate plus
  exact hard-forward ST as the main training candidate, with PL retained as a
  matched ablation rather than assumed default. Uni's fixed focus count, resized
  crop, classification proxy, full-frame size penalty, random second heavy
  branch, and early exit are not transplanted. No design approval, model code,
  experiment, or old pilot artifact changed.

- 2026-08-02: user approved Scheme A. The Dynamic Stage-1 main design removes
  the independent `q_ctx` head and uses
  `u_hard=q_base+max(0,delta_roi,delta_res)` for the unique global physical
  exact-B top-B; hard argmax supplies the operational context/ROI/residual role,
  making context the fully dynamic zero-modifier outcome. Backward uses a
  temperature-controlled log-sum-exp ST relaxation but cannot alter the hard
  support or executed-B ledger. The main estimator family is now a Uni-inspired
  stop-gradient coarse-feature surrogate plus exact hard-forward ST. PL is a
  separately trained matched ablation; recovery of the immutable fixed-quota
  A6/A7 pilot remains required for evidence but cannot automatically select the
  dynamic estimator. This decision is `designed`, not implemented or tested.
  Exact surrogate losses, gradient stops, degeneration guards and tests remain
  under section-by-section review; no model code, Job or old artifact changed.

- 2026-08-02: user approved the three-way gradient isolation and bounded proxy
  schedule. The cheap scout representation `Z` is jointly task-trained by a
  fit/train-only auxiliary TAD head, while the policy consumes `stopgrad(Z)`.
  The detector loss is active throughout training on the exact hard global
  top-B/ragged heavy path and reaches selected route scores through ST without
  depending on that bridge for detector learning. A backward-only global
  soft-budget projection with `0<p<1` and `sum p=B` aggregates detached scout
  features, supplies dense counterfactual TAD supervision to the policy and
  auxiliary head, never updates the scout/heavy backbone, never enters inference
  or the executed-B ledger, and is annealed to zero by successful optimizer step
  before a final hard-only phase. The main design adds no area, coverage,
  expected-cost, fixed-context, or fixed-`K_t` loss. This is `designed`, not
  implemented or tested; degeneration guards, exact schedule constants,
  ablations and tests remain under review. No model code, Job, or old artifact
  changed.
- `2026-08-02T09:52:19Z` ingest_paper: skipped existing paper wang2024_uniadafocus_spatialtemporal_dynamic.md (arxiv:2412.11228)

- 2026-08-02: user requested the exact Uni-AdaFocus deformable-patch formula and
  challenged whether the proposed SCNR-TAD geometry difference is necessary and
  correct. Re-audit of paper arXiv:2412.11228 and official commit `88464883`
  found that the paper's centre/height/width description is implemented with
  sigmoid top-left/size actions: source height/width are mapped to `96..224`
  pixels, top-left is mapped into the remaining legal interval, and the crop is
  resized to a fixed local input. In normalized centre coordinates this is
  algebraically the same bounded-interval family as the proposed formula. The
  real proposed changes are a native-grid minimum (`1/W_grid,1/H_grid`), native
  physical-token membership rather than resized crops, TAD proxy supervision,
  and omission of Uni Eq. 15's full-frame-seeking size regularizer. The latter
  differences are structurally motivated by exact hard B and high-IoU TAD, but
  the grid floor is not proven optimal and remains `discussed` pending user
  approval and a frozen scale-sensitivity ablation. No model code, experiment,
  Job, or old artifact changed.

- 2026-08-02: user approved the SCNR-TAD geometry contract. The main setting is
  now the Uni-equivalent bounded `(cx,cy,w,h)` mapping with runtime, axis-specific
  native-cell floors `w_min=1/W_grid` and `h_min=1/H_grid`; no Uni full-frame
  size penalty and no area/coverage/smoothness loss are allowed. The sole frozen
  scale sensitivity arm changes both axes to a 2x2-cell floor under otherwise
  matched training. Shared GeoRoute geometry code now accepts independent
  `(width,height)` limits, adds fail-closed `native_cells` resolution while
  preserving `static_normalized` as the historical default, and receipts the
  effective runtime floor plus the absence of a full-frame size penalty. Focused
  known-answer tests cover 11x20 -> `(1/20,1/11)` and `(2/20,2/11)`, independent
  axes, in-bounds gradients, legacy static behavior, and invalid floors. Static
  diff and Python compilation passed. Local tensor collection remains blocked by
  the pre-existing Windows `torch` `c10.dll` / WinError 1114 failure, so this is
  `implemented_pending_remote_tensor_test`, not `tested` or empirically
  supported. Experiment `exp:scnr-geometry-floor-sensitivity-v1` is `designed`
  and cannot launch before the genuine dynamic exact-B ragged Stage-1 executor,
  masked-zero carrier and approved proxy/ST path exist; no Job or performance
  experiment was started.

- 2026-08-02: SCNR native-cell geometry M0 passed remote tensor/regression
  validation at exact source
  `4be718449033e95dc6d15029ec4ef889397c9066`. A fresh N16R4 snapshot was cloned
  through the required academic proxy; full HEAD, origin branch ref and clean
  worktree matched. The focused geometry/routing file passed `36/36`. The full
  `tests/test_georoute*.py` set plus required C3 regressions passed
  `194 passed, 1 skipped` in the project OpenTAD Linux/Torch environment. No GPU
  allocation, dataset performance path, checkpoint, metric or Job was opened.
  The axis-specific floor primitive is therefore component-level `tested`, while
  `exp:scnr-geometry-floor-sensitivity-v1` remains `designed` and blocked on the
  true dynamic exact-B ragged executor, masked-zero carrier and proxy/ST training
  implementation.

- 2026-08-02: the approved dynamic ROI+TokenSelect Hybrid Stage-1 executor was
  implemented and tested. Commits `536e05b7` through `dfcbe692` added one global
  unique exact-`B` physical allocator, fully dynamic `K_t`, masked-zero empty
  tubelets, flat no-padding ragged VideoMAE/coordinate-lineage Adapter execution,
  the support-only hard ST path, a detached-scout exact-sum soft proxy and
  requested/unique/padded/executed plus `sum_c b_c^2` ledgers. Exact source
  `0c29a5e5` passed the relevant N16R4 Linux/Torch suite `92 passed`. First CUDA
  P0 Job `1215355` was retained as non-promoting diagnostic evidence: decoded
  `w,h` full extents were mistakenly used as ellipse semi-axes, producing
  context/ROI/residual `0/24573/3`. Corrected commit `dfcbe692` uses `w/2,h/2`;
  Job `1215358` completed `0:0`, sealed `PASS_NO_PERFORMANCE_P0`, exact
  selected/executed `B=24576`, zero padding, `K_t=48..83`, all required
  gradients, and role counts `4713/14292/5571`. Report file/internal SHA-256 are
  `285116b1ae02826f060d700b435253043a945e49aaecd5903aa0499cfb4abdb6` /
  `d2adbded39668f9945422e8e06dc5515d3d5ff20595b1e7bf234905e5bd0048d`.
  This is `tested` mechanical evidence, not performance or complementarity.

- 2026-08-02: real-data Fit-only dynamic policy-health Job `1215363` passed at
  exact clean source `7cf589f0ff583160c8e45b103e8ea4c316c10339`. It made 64
  successful updates in 66 attempts with two bounded AMP skip/replays, ended at
  scale `16384`, advanced scheduler/EMA exactly 64 times, and observed nonzero
  gradients for all nine required components on all 64 updates. Aggregate
  context/ROI/residual counts were `297230/984020/291614`, observed `K_t` was
  `17..218`, maximum soft-budget residual was `0.005859375`, and peak allocated
  CUDA memory was `7705451008` bytes. Artifact/no-leak audit found no metric,
  prediction, evaluator, checkpoint, teacher, oracle, official test or route-GT
  input. Report file/internal SHA-256 are
  `09e6b03fa747865cce0e1ed0ee54702f89723ae1b64c6b66ee5fdba7f8c3f3d8` /
  `fc457ba928743df12d68dcf3713128577d6b8cc175fe1196e0c2b730dfe5ac94`.
  This advances the route to real-Fit-prefix `tested`, not empirical support.

- 2026-08-02: the matched G2 two-native-cell ROI-floor arm was added and tested
  at exact source `8aa8e2a3c6649eca94d3ab714d0b122e4f7a5f97`. Its config differs
  from G1 only in `georoute_roi_extent_floor_cells=2`; local and clean remote
  focused tests passed `14/14`. Slurm CUDA P0 Job `1215364` completed `0:0` on
  `g0017` in 42 seconds. It sealed `PASS_NO_PERFORMANCE_P0`, exact
  selected/executed `B=24576`, zero padding, `K_t=49..85`, role counts
  context/ROI/residual `3899/15853/4824`, all required gradients, and peak
  allocated memory `5840570880` bytes. Report file/internal SHA-256 are
  `5103e024c7543de52946ae883b79c992096027a9d013b87a41912e3852957464` /
  `cb41492ea6723bbebb8beb3add8c515f2ab06f9dbfcaba482a67da48db222bbc`;
  an independent validator replay passed. M1 is complete and M2 is mechanically
  unblocked, but no floor-performance result exists.

- 2026-08-02: implemented the dynamic M2 accuracy-replay telemetry prerequisite
  at exact commit `7e5775e89c0e02428f9af2f6e13c4637a76c7850`.  The wrapper now emits
  sample-level ROI trajectories/distributions and floor saturation, complete
  `K_t` and operational-role distributions, plus fail-closed exact-B/true-ragged
  execution receipts.  Multi-sample attribution, invalid ROI bounds, lineage or
  uniqueness errors, padding, dense Adapter fallback and cost-ledger mismatch
  all abort.  Telemetry CPU copies are labelled accuracy-only and remain outside
  timed replay.  Static P0/health tests passed `14/14` locally; the Windows Torch
  tensor suite remained unavailable because `c10.dll` failed during import.
  Exact clean N16R4 Linux source passed the combined dynamic/P0/health suite
  `35/35`.  No GPU, dataset evaluation, checkpoint, metric, latency or energy
  result was produced.  M2 remains unstarted pending one matched development
  runner and a separately frozen selector-inclusive full-stack cost protocol.

- 2026-08-02: implemented and locally tested the matched dynamic native-floor M2
  protocol at exact source `ec8de9f51f85fc81031d82b79e30019d57a381b4`.
  G1/G2 are frozen as fresh seed-3407, 60-epoch arms with identical recipe/data/
  order, successful-update-only scheduler/EMA accounting, an atomic final
  epoch-59 checkpoint sidecar, and complete Gate accuracy/route telemetry.  Both
  successful arm receipts feed one physical-GPU serial
  `G1 -> G2 -> G2 -> G1` full decode/preprocess/H2D/model/postprocess/NMS replay.
  It disables diagnostic telemetry, records component timings and peak memory,
  uses a separate 20-ms NVML sidecar, and requires the validator to reintegrate
  raw power samples over retained monotonic energy windows.  The after-any
  finalizer fails closed on any missing/tampered lineage and reports descriptive
  single-seed deltas only. Python compile, Slurm Bash syntax, whitespace and
  focused contracts passed locally (`29 passed`). No remote precheck, training,
  checkpoint, prediction, metric, latency or energy result existed at this point.

- 2026-08-02: exact clean M2 runtime
  `9d6641a6c03644693e492d04a319b90fdad20238` passed remote Linux/Torch
  `76/76`, both arm prechecks, paired-cost precheck and finalizer precheck.
  Storage preflight observed `225,293,430,784` free versus
  `47,244,640,256` required bytes, and submit capacity allowed all four Jobs.
  The formal deployer nevertheless stopped before any Job submission when
  N16R4's submit Lua rejected the CPU-only finalizer `sbatch --test-only`.
  Failed root
  `/data/run01/sczc063/yuzibo/scnr_dynamic_floor_m2_9d6641a6_s3407_20260804_0507`
  contains only `control/storage_preflight.json`; a matching `squeue` query was
  empty. It will not be reused. Exact source
  `bad14693daa1fe414e56bf697c617e76f96eed48` applies the sole site-resource
  fix: finalizer requests 1 GPU/1 CPU, performs no model/cost work, and records
  the GPU as scheduling overhead. Local compile/whitespace and focused contracts
  pass `13/13`. Performance status remains not started pending fresh remote
  checks and a new namespace.

- 2026-08-02: replacement runtime
  `6ee97336775a09611f10423e07cafcea375e191a` passed the same remote
  Linux/Torch suite `76/76` and four fresh deployer prechecks. The immutable DAG
  was atomically submitted and released under
  `/data/run01/sczc063/yuzibo/scnr_dynamic_floor_m2_6ee97336_s3407_20260804_0525`:
  G1 `1216180`, G2 `1216181`, paired cost `1216182`, finalizer `1216183`.
  Deployment self/file SHA-256 are
  `a0504e45179957f20580b901e6ef7723d63c7b0ed445d8b3c35c3b5aaa02b89a` /
  `188da9dbf8cabffc1ab59cd90822e117adcd1457729c5a08806c916516ac8284`.
  Both fresh arm P0 receipts sealed `PASS_NO_PERFORMANCE_P0` and the two arms
  entered Epoch 0 on `g0024`; both first replayed matched batch 13 at scale
  `32768` with retry `1/8`. Cost remains afterok on both arms and finalizer
  remains afterany on both arms plus cost. Status is `experiment_running`; no
  complete checkpoint, prediction, metric, latency, energy or floor verdict is
  available or interpreted.
