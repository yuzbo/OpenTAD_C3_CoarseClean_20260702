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
