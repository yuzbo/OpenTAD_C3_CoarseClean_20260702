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
