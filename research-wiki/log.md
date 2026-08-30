---
type: wiki_log
append_only: true
---

# Research Wiki Log

- 2026-08-24：DSR6-KV 最小实现完成于 clean/pushed
  `3260cd39154069138c6b1757326372cc3b73754e`（父 `4e940b…`，分支
  `codex/zoomtoken-dsr6-kv-v001`）。前 6 层复用 FULL64，后 6 层复用固定 K32 的 MOD32-KV，
  保留全部非 detach K64 K/V 与全 K64 Adapter；无新参数、cache、transport 或 loss。静态、
  编译、Shell 与小型无数据 Torch 前后向通过；完整 pytest 受本机 DLL/OpenMMLab 环境阻断。
  独立 Critic `AUDIT_PASS`。
- 2026-08-24：结果盲 PRE_RUN `READY`。正确 N16R4 端点只读确认 canonical 411 MP4、0
  断链，注释/类别映射/VideoMAE-S/OpenTAD 环境存在，GitHub ref 精确指向 `3260cd39…`，
  拟定根 `/data/run01/sczc063/yuzibo/projects/zoomtoken_dsr6_3260cd39_seed42_20260824`
  不存在，`/data` 可用约 1.65 TB。尚未创建远端 source/root、未提交 Slurm、未读取结果；
  下一步仅请求唯一 seed42/双卡/60-epoch DSR6-KV 训练授权。

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

- 2026-08-05: dynamic-floor M2 reached an incomplete terminal state. G1/G2 Jobs
  `1216180/1216181` completed `0:0` and published hash-valid stage results, while
  paired-cost `1216182` failed `1:0` because its timed audit read nonexistent
  `packed.attention_pairs` instead of the native ragged executor's
  `packed.attention_pairs_per_window`. Finalizer `1216183` completed and sealed
  `INCOMPLETE_NO_FLOOR_INFERENCE`, missing paired cost, empty contrasts and all
  promotion guards false. No arm metric or partial cost was interpreted. The
  minimal execution-only recovery now validates clip counts and the current
  per-window attention-pair ledger and separately records frozen model runtime
  `6ee97336` versus its clean recovery-execution commit; model, checkpoints,
  data, ROI floors, budget and evaluation are unchanged. Python compile and the
  focused M2 suite pass locally (`15/15`). Full counterbalanced cost plus a new
  fail-closed finalizer remain required before any G1/G2 comparison.

- 2026-08-05: deployed the no-retraining dynamic-floor M2 cost recovery. Clean
  execution commit `c67e13e8` passed remote focused `50/50` and cost precheck;
  original runtime `6ee97336`, G1/G2 terminal states, and both stage-result hashes
  were revalidated unchanged. A first submission request using `afterok` on the
  old completed Jobs created no Job because they had aged out as accepted
  live-controller dependency targets. Recovery cost Job `1222672` and finalizer
  Job `1222673` were then submitted held, bound to a self-hashed recovery receipt,
  validated through the original fail-closed deployment validator, and released.
  The truthful Slurm DAG is cost with no dependency and finalizer
  `afterany:1222672`; the receipt separately preserves the frozen scientific DAG.
  Recovery deployment self/file SHA-256 are
  `12cbbb3f609adaa57ca9b29bf930bd124cd35c5f33aaa966fc6a9529c3d1de89` /
  `f67703bf4dc5d066f64a1bafa36d49be133a7b6701507ef8ef31d651a7d2fba7`.
  Failed artifacts remain archived. Status stays `experiment_running`; no live or
  partial pass metric is interpreted.

- 2026-08-05: first cost recovery `1222672` failed `1:0` at its first timed sample
  because `sparse_adapter.forward_ragged` is invoked directly and bypassed the
  profiler's module-forward hook. Finalizer `1222673` completed and again sealed
  `INCOMPLETE_NO_FLOOR_INFERENCE`; no paired profile, contrast or promotion was
  produced or interpreted. Its cost and finalization were preserved under
  `cost_failed_job1222672/` and
  `control/finalization_incomplete_job1222673.json`. Exact execution-only repair
  `6341927f` instruments the actual ragged method and names invalid stages; local
  `16/16`, remote `51/51` and cost precheck pass, while model/config remain
  unchanged. After arm state/hash revalidation, held cost/finalizer Jobs
  `1222700/1222701` were bound to recovery-v2 receipt self/file SHA-256
  `9370e5908718e0c6fef857c3b29ddeefbe5701bbc6a1c221f7ad7f828dac99e7` /
  `4968593b4172df4bbe7feeec9bad623ae188ffeeadedbd2b87c48a0bfa811fa3`,
  validated, and released. Status remains `experiment_running`; no live pass is read.

- 2026-08-05: second cost recovery `1222700` ran the complete four-pass order but
  failed `1:0` only at final profile validation. Producer and validator had
  duplicate cost-config construction: only producer forced
  `post_processing.sliding_window=True`, causing the receipted config hash to
  mismatch. Finalizer `1222701` sealed incomplete. The four raw files, manifest
  SHA-256 `c1fe0d6f...acf20`, and finalization self/file SHA-256
  `de38cb77...91e72` / `e28b38b...83d` were archived; manual salvage is forbidden
  because complete pass receipts and profile provenance were never published.
  Exact repair `011d2943` adds one shared cost-config builder used by producer,
  validator and fixture. Local `16/16`, clean remote `51/51` and cost precheck
  pass with no model/config/training change. After arm state/hash revalidation,
  held cost/finalizer Jobs `1222869/1222870` were bound to recovery-v3 receipt
  self/file SHA-256 `4cc8b764...584e2` / `623c2f95...81fe`, validated through the
  original finalizer, and released. Actual DAG is cost none and finalizer
  `afterany:1222869`; state remains `experiment_running` and no live metric is read.

- 2026-08-05: cost recovery `1222869` failed `1:0` before creating `cost/` because
  population preflight retained a call to deleted `_cost_config`; the exact
  traceback is `NameError` at `profile_georoute_dynamic_floor_m2.py:651`.
  Finalizer `1222870` completed and sealed `INCOMPLETE_NO_FLOOR_INFERENCE`; no pass
  sample, paired profile, contrast or promotion exists. Finalization self/file
  SHA-256 are `f586083c...9b75b` / `5bca01ea...7d59`. Exact repair `42923d9f`
  routes that call through the shared builder and adds an AST regression. Local
  `16/16`, remote clean `51/51` and precheck pass. After arm and empty-output
  revalidation, held Jobs `1222889/1222890` were bound to recovery-v4 receipt
  self/file SHA-256 `5fe63bce...c1e1c` / `5bd504a6...51226`, validated and
  released with actual DAG cost none/finalizer `afterany:1222889`. Status remains
  `experiment_running`; no live pass is read.

- 2026-08-06: cost `1222889` completed `0:0` and its profile validates under cost
  execution `42923d9f`, but finalizer `1222890` sealed incomplete because it ran
  frozen runtime source `6ee97336`; that old validator reconstructed pre-repair
  cost hashes without `sliding_window=True`. The incomplete finalization was
  archived with self/file SHA-256 `ecc7ec1b...63b00` / `3773cb28...ef97d`; valid
  cost/raw artifacts were not moved or rewritten. Repair `75e2adc8` binds model
  runtime, cost execution and finalizer execution separately and passes remote
  `52/52`, precheck and a no-number dry run. Finalizer-only Job `1223310` was held,
  bound to deployment self/file SHA-256 `8fe36543...8b9f0c` /
  `1eeff523...7f1ee`, validated and released without retraining or cost replay.
  Status remains `experiment_running`; no descriptive value is yet interpreted.

- 2026-08-06: finalizer-only Job `1223310` completed `0:0` and sealed
  `PASS_COMPLETE_DESCRIPTIVE_FLOOR_SENSITIVITY /
  COMPLETE_DESCRIPTIVE_ONLY_M3_REQUIRED_FOR_FLOOR_SELECTION` with `errors={}`.
  G1 minus G2 is `+5.78 pp` Avg-mAP and `+6.22 pp` high-IoU composite; model
  forward p50 is only `+0.438%`, while the aggregate end-to-end `+2.845%` is
  cold-order-sensitive. Both width/height floor saturation rates are zero.
  Dynamic `K_t=0` occurs in both arms, but selected context/ROI/residual counts
  collapse to `0/7/3,342,329` for G1 and `0/0/3,342,336` for G2. Therefore M2 is
  terminal `tested` descriptive evidence, not floor causality or operational
  Hybrid complementarity. Official test, paper claim and single-seed floor
  selection remain false. Same-family Type-A terminal audit passed A--F; no
  cross-family overlay was available.

- 2026-08-06: froze the next step as
  `exp:scnr-dynamic-role-calibration-diagnostic-v1` and implemented its local
  diagnostic surface. It measures all-valid/selected/unselected `q_base`, signed
  ROI modifier, residual modifier, pairwise differences, role wins and top1-top2
  margins only during out-of-band accuracy replay. It uses no target fractions,
  fixed quota, new `q_ctx`, GT, teacher, route mutation or timed-cost
  instrumentation. Python compile and whitespace pass; Windows Torch collection
  is blocked at the known `c10.dll` import boundary, so clean N16R4 tests and
  prediction-SHA-preserving frozen-checkpoint replay remain pending. M3 is held
  until this mechanism diagnosis is complete.

- 2026-08-06: hardened the role-calibration replay first under execution source
  `c48153885de5516abe9e854f0b8d1a8635824905`. The runner validates the frozen
  M2 accuracy config before instrumentation, binds model runtime `6ee97336`
  separately, requires source config/checkpoint/prediction hashes, exact population
  SHA and 136-window count, and automatically fails on prediction/population parity
  or analyzer boundary errors. Clean N16R4 tests passed `46/46` focused plus
  `20/20` required C3. Two over-memory `sbatch` attempts produced no Job ID. Held
  Jobs `1223595/1223596` were released with default memory but failed before
  inference because generic Phase-M added `--not_eval`, correctly rejected by the
  frozen M2 accuracy guard. They produced no prediction/calibration telemetry.
  Recovery `469dfe636a127541740843e1fb398d2db177f9f9` removed that flag only for the
  frozen M2 role replay and passed remote `48/48 + 20/20`, but replacement Jobs
  `1223601/1223602` exposed a second generic default: development profiling was
  enabled, while frozen M2 accuracy requires telemetry on and profiler off. They
  also failed before inference with no prediction/calibration telemetry. Recovery
  `b6c792fc66956cab0d4b0f18e756ecb675d20d93` preserves profile=false, separates
  model commit `6ee97336` from the clean diagnostic execution commit while retaining
  source checkpoint validation, passes real-config preflight and remote
  `49/49 + 20/20`. Held Jobs `1223615/1223616` were validated and released under a
  new namespace. Status is `experiment_running`; partial role telemetry remains
  unread.

- 2026-08-06: closed all later role-calibration replay attempts without a role
  result. Jobs `1223615/1223616` completed inference but the `b6c792fc` runner used
  the wrong output root; Jobs `1223625/1223626` completed inference but `0f97307d`
  expected the pre-formal diagnostic schema. Neither frozen attempt is salvaged.
  Exact source `2c39ce58791704a29745e9172565df42fba4723b` passed remote
  `51/51 + 20/20`, and Jobs `1223640/1223641` completed formal inference, but both
  failed exact historical prediction-SHA parity. Their role telemetry was not
  read. Failure-receipt self-hashes are not prediction hashes.

- 2026-08-06: compared only the source and replay prediction artifacts to localize
  the integrity failure. G1/G2 replay SHA-256 values are `3fa61c...` and
  `92c3e3...`; both retain the exact ordered 40-video set and 80,000 records.
  Exact `(video,label,start,end)` overlap is `76,660/80,000` and
  `78,387/80,000`, while video+label overlap is `79,925` and `79,934`. Thus the
  mismatch is not JSON formatting alone. Original M2 ran on `g0024`; replay ran on
  `g0044/g0048`, so source rerun drift and instrumentation remain confounded.

- 2026-08-06: implemented
  `exp:scnr-role-instrumentation-neutrality-pair-v1`. One Slurm job now executes
  source-formal role-OFF then role-ON checkpoint inference serially on one visible
  GPU, with exact config/checkpoint/seed/population/B/evaluation and
  `profile=false`; only the role-calibration extension changes. The runner
  validates formal no-leak schemas, forbids profiler artifacts, compares raw and
  semantic predictions without metrics, hard-fails on OFF/ON raw-SHA mismatch and
  never summarizes role statistics. Source parity remains required by the frozen
  role-diagnostic contract. Local Python/Bash/whitespace checks pass; Windows Torch
  tests stop at the known `c10.dll` boundary. Remote validation and execution are
  pending; no Pro discussion or model repair precedes this pair.

- 2026-08-06: closed the role-observer integrity chain. Same-GPU serial pair Jobs
  `1223686/1223687` showed OFF/ON neutrality but historical-source drift. Legacy
  OFF-A/OFF-B/ON Jobs `1223707/1223708` then showed baseline OFF/OFF prediction
  drift while all 136 route hashes remained equal. The first downstream warning
  localized ordinary replay nondeterminism to memory-efficient CUDA SDPA after
  routing. Strict math-SDPA Jobs `1223727/1223728` emitted zero such warnings and
  produced byte-identical OFF-A/OFF-B/ON predictions in each arm (G1
  `57886055...331e`, G2 `f0ce98ce...d834`). This proves observer neutrality but
  does not restore historical continuous-score or performance comparability.

- 2026-08-06: exact source `ede8af53d47c723a30902063f2f2bcdf260d340c`
  implemented a field-minimized categorical invariance bridge. It validated
  exact legacy/strict hard-role equality for `136/136` windows while keeping
  continuous modifiers, margins, geometry, predictions and performance closed.
  Authorized all-valid context/ROI/residual counts are
  `0/2,671/11,486,609` for G1 and `0/984/11,488,296` for G2; residual dominates
  every window before global top-B. Selected counts remain `0/7/3,342,329` and
  `0/0/3,342,336`. Thus branch-offset calibration is the first identifiable
  repair target; no Hybrid, floor, accuracy, cost, M3 or paper claim opens.

- 2026-08-06: preregistered
  `exp:scnr-residual-window-centering-probe-v1` at stage `designed`. The only
  intervention subtracts the differentiable all-valid 384-tubelet-window mean
  from `delta_residual` immediately before unchanged Scheme-A role argmax/top-B.
  `q_base`, ROI, exact-zero context, exact B, fully dynamic K including zero,
  ragged execution and masked-zero carrier remain unchanged; default mode is
  `none`. A strict deterministic frozen-G1/G2 checkpoint mechanism probe must
  restore structural context/ROI reachability and select at least one
  non-residual token before matched training is even designed. No further Pro
  discussion is needed for this one-variable probe.

- 2026-08-06: implemented the preregistered residual-window-centering probe.
  The opt-in model intervention subtracts only the differentiable all-valid
  full-window mean from `delta_residual` before unchanged Scheme-A role
  arbitration and exact-B selection; default behavior remains `none`. A new
  M2-bound no-metric runner performs serial strict math-SDPA duplicate replays,
  validates exact-B/ragged/no-leak and calibration receipts, and requires both
  raw-prediction and route-payload determinism before classifying structural
  context/ROI reachability. Local compilation, Bash syntax, whitespace,
  pure-contract tests, and standalone result-classification checks pass. The
  Torch-backed local suite remains blocked at the known Windows `c10.dll`
  loader boundary. Clean Linux/Torch regression and frozen G1/G2 Slurm probes
  are pending; no training, performance, floor, M3, official-test, or paper
  claim is open.

- 2026-08-06: exact source `091f9f9b57e68a4706a91d8b3b9176ddc88d0c6c`
  passed clean N16R4 GeoRoute/probe `90/90` and required C3 `20/20` regression.
  Frozen M2 Jobs `1223783/1223784` then completed `0:0`; every generated artifact
  hash and result self-hash validates. Within each arm, strict math-SDPA
  `centered_a/centered_b` predictions are byte-identical across 80,000 records
  and route payload hashes match. G1 selected context/ROI/residual counts are
  `168,733/421,121/2,752,482`; G2 counts are
  `186,976/429,896/2,725,464`. Both all-valid role sets also restore context and
  ROI reachability, and the maximum absolute centered residual mean is
  `3.04e-7`. Thus the registered structural gate passes in both arms. This
  supports residual branch-offset non-identifiability as a direct cause of the
  earlier categorical collapse and authorizes only a new matched development
  `none` versus centering training design. No mAP was evaluated, no training was
  performed, and Hybrid efficacy, cost, floor, M3, official-test and paper
  claims remain closed.

- 2026-08-06: froze
  `exp:scnr-residual-centering-matched-training-v1` without another Pro review.
  The two fresh seed-3407 cells both use the G1 `native_1cell_main` recipe and
  differ only in `georoute_branch_calibration_mode=none` versus
  `residual_window_center`. Both must train 60 epochs/9,600 successful updates
  from the same pretrained initialization and pass same-GPU strict math-SDPA
  duplicate Gate replay. No old M2 checkpoint, G2 floor reinterpretation, role
  target, resume, or second repair is allowed. After complete integrity, the
  preregistered screen requires positive mAP@0.6 and mAP@0.7 deltas plus
  nonnegative Avg-mAP delta. PASS opens only a separately frozen ABBA+BAAB
  full-stack cost study; multi-seed, M3, official test, efficiency,
  complementarity and paper claims remain closed. Status is `designed`, with no
  new training Job or metric yet.

- 2026-08-06: implemented the frozen matched residual-centering training DAG.
  A common constructor materializes both fresh G1 seed-3407 cells and binds a
  normalized complete-training-recipe hash, preventing optimizer, loss,
  augmentation, scheduler, detector/head, data or execution drift while allowing
  only receipt/path identity and the registered branch-calibration mode to vary.
  Per-arm execution performs fresh P0, 60 epochs/9,600 successful updates,
  final-EMA-only publication and serial strict `accuracy_a/accuracy_b` replay.
  The atomic deployer submits both stages held, persists a held `afterany`
  finalizer with their exact Job IDs, validates the deployment receipt and only
  then releases the DAG. Malformed/incomplete artifacts fail closed with empty
  contrasts. Python compilation, Bash syntax and local pure-contract/inherited-M2/
  required-C3 regression pass `57/57` (`10/10` new study tests). Local
  Torch-backed collection remains blocked at the known Windows `c10.dll` loader;
  clean N16R4 regression, exact commit, precheck and Slurm execution are pending.
  Stage is `implemented`, with no checkpoint, prediction, metric or performance
  conclusion.

- 2026-08-06: exact runtime
  `16137484c5ccad422e017e67a81c1a07d1ed2fbb` passed clean N16R4 Linux/Torch
  regression `93/93`. The deployment precheck passed both stage launchers, the
  finalizer, submit-capacity (`3 + 3 <= 16`) and storage (`203.7 GB` observed
  free versus `47.2 GB` required). Atomic deployment SHA
  `71b10681118c57a845deb33a3f0f98d269ae05ac4b6b9e0e0114182ee1998b59`
  created fresh root
  `scnr_residual_centering_matched_training_16137484_s3407_20260806_061352`,
  released control/center Jobs `1223819/1223820`, and bound after-any finalizer
  `1223821` to exactly those predecessors. Both no-performance P0 reports passed
  (`0f24c871...` / `edd5617a...`) and both fresh trainings entered epoch 0 on
  `g0059`. The experiment is now `experiment_running`; no checkpoint, duplicate
  accuracy, contrast, cost or empirical claim exists yet.

- 2026-08-09: reconciled the terminal residual-centering matched-training
  artifacts before any next experiment. Jobs `1223819/1223820/1223821` all
  completed `0:0`; each fresh cell completed 60 epochs/9,600 successful updates,
  published one epoch-59 EMA checkpoint, and passed byte-identical strict
  duplicate Gate prediction plus route/population/metric replay. Common
  complete-protocol SHA-256 is `34defbdbc30e7fff10bbb05d7e6665dd29b8128f8f03cd389250bca9e3e7493c`.
  Control versus centered Avg/mAP@.6/mAP@.7 is `10.52/8.90/6.98` versus
  `12.57/11.04/8.14`; deltas `+2.05/+2.14/+1.16 pp` pass every preregistered
  sign. Centering restores selected context/ROI/residual to
  `210,925/1,613,683/1,517,728` while control remains residual-only. Independent
  integrity review passes provenance, score handling, exactness, and executed
  path, with single-seed/development-only scope warnings. Finalization
  `2a9351a3...` authorizes paired cost only; no seed, efficiency, official-test,
  complementarity or paper claim opens. The matched-training node advances from
  `experiment_running` to `tested`.

- 2026-08-09: froze and implemented
  `exp:scnr-residual-centering-paired-cost-v1` without another Pro discussion.
  It consumes the two audited seed-3407 checkpoints without training and uses
  exactly one held Slurm Job/GPU, one continuous 20-ms NVML sidecar, 50 warmups
  per pass, and serial order `none,center,center,none,center,none,none,center`.
  Full scope is decode/preprocess/H2D/scout/route/patch/backbone/ragged adapter/
  head/postprocess/video NMS plus memory, energy, K_t and attention pairs.
  Four paired-pass ratios feed a 10,000-replicate video-cluster/pass-pair
  bootstrap. Seeds 3408/3409 open only when end-to-end-p50 and energy/sample
  center/control 95% upper bounds are both `<=1.05`; independent repeated Jobs
  remain mandatory for a paper efficiency claim. New contracts recursively bind
  the terminal training finalization/stages/checkpoints and separate model
  runtime from cost execution while rejecting any sensitive model/config diff.
  Local focused tests pass `8/8`; the combined cost, matched-training, inherited
  M2, and required C3 regression matrix passes `65/65`. Remote Linux validation
  and Slurm deployment remain pending.

- 2026-08-09: exact paired-cost execution
  `2eca86cfaf8804408a24567bb98bf4bd2046417b` passed clean N16R4 compile/Bash
  checks and the combined Linux/Torch regression `65/65`, then passed the frozen
  no-training cost precheck against both terminal seed-3407 source receipts.
  Atomic deployment
  `3e12809cfe799838d77c339b9b75af0b79e9c925723d6f9c5e50efd0507b72ab`
  bound and released the sole paired-cost Job `1233097` under
  `/data/run01/sczc063/yuzibo/scnr_residual_centering_paired_cost_2eca86cf_from16137484_s3407_20260809_112558`.
  The scheduler reports `PENDING (Priority)`, so the experiment advances to
  `experiment_running` but no cost, Pareto, seed authorization, or paper claim
  exists yet.

- 2026-08-12: accepted first-author decision
  `PRO_RUNTIME_FAIRNESS_AND_CAUSAL_STUDY_DECISION-v001` preserves the SCNR-Core
  route and its seed-3407 cost falsifier but revises the fair-runtime gate.
  Generic N16R4 allocation is only a carrier: a machine-generated GPU/software
  attestor must run after allocation and immutable-container launch yet before
  model, CUDA, data, checkpoint, warmup, metric, or cost work. All preflight and
  formal passes must match one runtime class; missing, late, requeued, fallback,
  or mismatched identities force no-survivor termination. This is a design
  decision, not an experiment. If the runtime and existing cost gate pass, the
  only next claim-bearing matrix is F/N/Q/D: full continuous-ROI SCNR-Core,
  no-ROI, free native-token, and dense controls. Residual calibration remains an
  anti-collapse diagnostic rather than the paper headline.

- 2026-08-14: ARIS CPR material pass (session `zoomtoken-aris-cpr-20260814-v001`),
  a first-author **proposal**, not an accepted decision. Paper endpoint restated
  per human constraint: reducing redundant spatial recomputation in end-to-end
  offline TAD while preserving boundary-sensitive/high-IoU detection; continuous
  ROI and F/N/Q/D are mechanism/ablation tools, not the headline. Three
  non-equivalent routes: (1) TAD-aware dynamic spatial compute routing
  (per-tubelet saliency + dynamic `K_t` + ragged heavy on selected tokens; ROI =
  optional prior, residual = anti-collapse calibration) — RECOMMENDED; (2)
  original SCNR exact-B full modifier family — REJECTED (mechanism bundling, M2
  role collapse `0/0/3,342,336`, overengineering); (3) aggressive spatiotemporal
  joint window budget — REJECTED (causal+systems risk, staged out by frozen
  exact-B rule). Clean routing candidate `cd6463df…` is static wiring evidence
  only; `b157433d…` is a closed detector-fixture negative. Replay/full training
  remain blocked pending a fresh Pro decision plus PRE_RUN/runtime/data/resource
  gates.

- 2026-08-14: P1 `{DN,U,R,Q}` transport recovery (session
  `zoomtoken-aris-p1-dnurq-20260814-v001`). Fresh exact-Project ChatGPT Pro has
  returned `PRO_DYNAMIC_SPATIAL_ROUTING_DECISION-v001 = REVISE`, **accepting the
  Q-core route**: paper endpoint is reducing redundant VideoMAE spatial heavy
  recomputation in end-to-end offline TAD while preserving boundary/high-IoU
  localization; continuous ROI and calibrated residual are demoted to later
  causal controls (G/N/F), not the headline. First real-video P1 screen frozen
  as `{DN, U, R, Q}` @ seed 3407 with `DO` (official dense) as anchor; `B=24576`
  (384 tubelet × 64) with dynamic token identity/position; seeds 3408/3409 only
  on admission; matched official THUMOS14/AdaTAD detector/loss/NMS/split/9,600
  updates/AMP/EMA; high-IoU and boundary estimands; selector-inclusive
  decode-to-NMS p50/p95 + memory/energy census; Pro success/failure/stop rules
  (Q beats uniform+random AND cost ≤0.85× dense; F/D Pareto metric lower bound
  ≥−0.50 pp and cost upper ≤0.85; three-seed F/N/Q thresholds). Durable packet
  written to `docs/aris/ARIS_P1_DNURQ_PACKET-2026-08-14.md`. Prior seed-3407
  residual-centering, cold paired cost, F/N/Q/D static wiring `AUDIT_PASS`, and
  `cd6463df…` are diagnostic or static only; no actual P1/real-video/GPU result
  exists. Replay/full training remain blocked on runtime-identity attestation +
  seed-3407 steady-cost admission + `PRE_RUN_READY`.

- 2026-08-14: result-blind P1 PRE_RUN intake returned `PRE_RUN_NOT_READY`.
  The Q-core route and its real-video first screen are unchanged; clean static
  candidate `a6b81095…` only establishes configuration/validator wiring, not
  runtime, accuracy, energy, or paper evidence. A six-existing-file,
  one-entry-point realization plan can add the five-arm deployment, allocation-
  time runtime attestation, selector-inclusive raw-video cost census, and
  no-survivor finalizer without a new platform. It made no edits because four
  evaluator semantics are not scientifically frozen: short-action partition,
  start/end boundary definition, high-IoU decomposition, and Q's dense cost
  comparator. These require one narrow scientific clarification; no P1 launch,
  replay, training, metric, or cost conclusion is opened.

- 2026-08-14: exact-Project Pro decision
  `PRO_G5_FOUR_DEFINITION_DECISION-v001 = CONTINUE` closed only the four P1 G5
  evaluator ambiguities. It fixes official source-video GT short actions as
  `0 < d <= 5.0` seconds (report-only); class-consistent score-ordered
  one-to-one inclusive-0.50 boundary matching with duration-normalized errors
  and separately counted unmatched instances (report-only); seven exhaustive
  GT high-IoU bins plus three unmatched-prediction bins (report-only); and DN
  as the sole Q p50/gross-energy denominator, with both one-sided 95% upper
  bounds `<=0.85`. The mandatory DO anchor is report-only. The full seed-3407
  `DO/DN/U/R/Q` matrix validly fails as `STOP_Q_CORE_P1`; any malformed or
  incomplete matrix is `NO_SURVIVOR_INVALID_P1`. The next bounded work is the
  already proposed six-file G1/G2/G5 realization, then independent Critic and
  result-blind Evaluator. No data, GPU, P1 result, performance, or cost evidence
  was produced by this decision.

- 2026-08-14: N16R4 read-only resource inventory confirmed official annotation
  and class-map availability, 200 development validation MP4s, a
  136-window/40-cluster manifest, a VideoMAE-S pretrain identity, an immutable
  UBI9.6 SIF and conda/pip lock. The first carrier witness failed before a CUDA
  kernel because of nounset/profile ordering; an independently reviewed
  agent-follows-doc witness `1238193` then completed in 32 seconds on RTX4090
  with Torch 2.0.1/NumPy 1.23.5 and a finite seeded CUDA matmul. This is runtime
  carrier evidence only—not P1, model efficacy, cost, fairness, or paper
  evidence. P1 remains closed pending a two-GPU preflight/leaf equality receipt,
  clean GitHub-resolvable `020bade…` origin reference and frozen preflight
  parent, official input-protocol binding, and a single-use empty run root with
  capacity/storage evidence.

- 2026-08-15: the clean focused P1 candidate
  `020bade11a448f07a8b9e2aef1dd05c6f5fae121` was recovered at
  `/data/run01/sczc063/yuzibo/runtime/zoomtoken_p1_020bade/source`, with
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git` origin and
  `refs/remotes/origin/codex/zoomtoken-p1-dnurq-v001` at the same SHA; the tree
  is clean. The verified repair parent exists remotely (file identity
  `f8ef174c...bd77e3`). One current-commit official-comparable F0 protocol gate
  was already submitted under
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_f0_020bade_20260815_0330`: PL/ST/
  world2/finalizer `1238199/1238200/1238201/1238202` began `PENDING`. It is the
  sole gate to monitor, never to resubmit. The RTX4090 carrier witness
  `1238193` (Torch 2.0.1, NumPy 1.23.5, CUDA visible) plus SIF
  `45c313...6e82` and lock `306927...8437` remain carrier facts. F0/env facts
  are neither efficacy, accuracy, cost nor paper evidence. Only terminal F0
  PASS with matched source/commit/protocol/runtime identities can mark P1
  deployable; every other terminal outcome keeps it closed.

- 2026-08-15: the first `020bade…` F0 is terminal but not a scientific event:
  world2 `1238201` completed `0:0`, while PL/ST `1238199/1238200` failed `1:0`
  before model load because the official DO config had no `georoute_protocol`.
  Finalizer `1238202` wrote `OFFICIAL_COMPARABLE_PREFLIGHT_HOLD`, `metrics={}`
  and `official_test=false`. This is a deterministic configuration-binding
  interface defect, not efficacy, accuracy, latency, energy, or cost evidence.
  Minimal clean successor `0351832f7e4203312b9ed9d6323a47ef4be33b2f` (parent
  `020bade…`) changes only the pilot contract and its focused test: it creates
  `georoute_protocol={}` only if absent before writing the existing status and
  never overwrites a present mapping. Independent default-agent audit passed and
  immutable-SIF/lock focused pytest passed `26` tests. Clean remote source is
  `/data/run01/sczc063/yuzibo/runtime/zoomtoken_p1_0351832f/source`, with the
  exact GitHub origin and `refs/remotes/origin/codex/zoomtoken-p1-f0-binding-v001`
  at the same clean SHA. The sole corrected no-performance F0 is already
  running under
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_f0_0351832f_20260815_0340`:
  PL/ST/world2/finalizer `1238210/1238211/1238212/1238213` initially
  `RUNNING/RUNNING/RUNNING/PENDING`; its submission receipt is
  `runtime/zoomtoken_p1_0351832f/preflight_submit_20260815_0340.json`. Only its
  terminal PASS with matched source/ref/protocol/input/runtime identities can
  make P1 deployable; it must not be resubmitted and cannot provide a paper
  result.

- 2026-08-15: read-only terminal monitoring of the sole corrected F0 found no
  admissible PASS: PL/ST/world2/finalizer `1238210/1238211/1238212/1238213`
  are `FAILED 1:0/FAILED 1:0/COMPLETED 0:0/FAILED 1:0`. The finalizer entered
  the immutable N16R4 environment but crashed in
  `finalize_georoute_official_comparable_preflight.py` while converting
  `summary["final_scale"] = None` with `float(...)`. No valid finalizer receipt,
  metric, official-test access, accuracy, latency, energy, cost or P1 admission
  exists. This is a second deterministic finalizer/receipt-representation
  implementation blocker, not a scientific failure. Automatic resubmission is
  prohibited; P1 remains closed as `F0_FINALIZER_FINAL_SCALE_NONE_TYPEERROR`
  pending a separately authorized bounded correction.

- 2026-08-15: terminal intake refined the second F0 root's exact failure
  chain. Its PL/ST cells `1238210/1238211` reached model construction and real
  training-data loading but failed before the first update because the native
  official AdaTAD backbone has no `set_successful_update_index`; world2
  `1238212` passed. The invocation had mistakenly treated the immutable official
  anchor as `GEOROUTE_SOURCE_CONFIG`, instead of using
  `configs/adatad/thumos/georoute_adatad_development_base.py` as selector source
  and the official config only as `GEOROUTE_OFFICIAL_REFERENCE_CONFIG`. The
  finalizer `1238213` then repeated the failure-receipt `final_scale=None`
  `float(...)` crash and wrote no finalization. There are no metrics,
  checkpoints, test access, efficacy or cost results.

- 2026-08-15: clean correction `a62e5fe5a0d2601a2b9bb3213ec027e56158e6ff`
  on `codex/zoomtoken-p1-f0-binding-v001` received independent audit PASS. It
  makes the successful-update hook optional only for dense-native `DO` (sparse
  arms retain it); initializes a missing `georoute_protocol` container without
  overwriting a present field; and maps absent/invalid/infinite failure
  `final_scale` to fail-closed HOLD. Relevant local/remote focused tests passed
  (`26 + 2`); one unmodified historical string assertion mismatch is unrelated.
  The sole new no-performance F0 is
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_f0_a62e5fe5_20260815_0355`, with
  PL/ST/world2/finalizer `1238221/1238222/1238223/1238224`. Only a matching
  terminal PASS can progress P1; no duplicate F0 or P1 submission is allowed.

- 2026-08-15: `a62e5fe5…` F0 is terminal PASS only for
  `FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED`: PL/ST/world2/finalizer
  `1238221/1238222/1238223/1238224` all completed `0:0`, while performance,
  paper and official-test permissions remained false. The initial P1 attempt
  failed closed before `sbatch`: its deployer incorrectly required the official
  DO config to match the formal Georoute bridge hash, so zero P1 jobs existed.
  This is an infrastructure binding negative, not model evidence. Clean pushed
  `f45fb8db9130c35d6f3b191e1292fa2a4a6c205c` fixes only that comparison:
  P1 uses `official_config_sha256`; formal continues to use
  `georoute_source_config_sha256`. Focused static checks passed `16`. The sole
  final no-performance F0 is
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_f0_f45fb8db_20260815_0405`, with
  PL/ST/world2/finalizer `1238229/1238230/1238231/1238232`. A matching PASS may
  release exactly one real THUMOS14 P1 `DO/DN/U/R/Q` matrix; no duplicate F0 or
  P1 submission is permitted.

- 2026-08-15: the `f45fb8db…` F0 advanced only the no-performance protocol
  gate. The first real P1 deployment then failed closed at `sbatch --test-only`:
  N16R4 supplies 55 GB per GPU (110 GB across the frozen two-GPU allocation),
  but the implementation requested `192000 MiB`; no P1 run namespace or Slurm
  job was created. This is a deterministic resource-shape binding failure, not
  scientific, accuracy, latency, energy, cost, or paper evidence. Clean pushed
  `c875ea62ea4f480a9039335a7c171c630d703a9e` changes only that request to the
  frozen two-GPU `96000 MiB` shape and passed 16 focused static checks. The sole
  current no-performance F0 is
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_f0_c875ea62_20260815_0415`, with
  PL/ST/world2/finalizer `1238287/1238288/1238289/1238290`. It is the only
  F0 population to monitor; no duplicate F0/P1 submission is permitted.

- 2026-08-15: the sole `c875ea62…` F0 is terminal PASS. PL/ST/world2/finalizer
  `1238287/1238288/1238289/1238290` all completed `0:0`; finalization records
  `PASS_OFFICIAL_COMPARABLE_PREFLIGHT_ONLY` and
  `FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED`, with empty performance metrics and
  every performance, paper and official-test permission false. The bound remote
  source is clean: `HEAD` and
  `refs/remotes/origin/codex/zoomtoken-p1-f0-binding-v001` both resolve to
  `c875ea62ea4f480a9039335a7c171c630d703a9e` at the frozen GitHub origin. Its
  two-rank KAT records RTX4090 ranks with Torch 2.0.1/CUDA 11.8. This is
  deployment/protocol evidence only: it makes exactly one atomic seed-3407 P1
  `DO/DN/U/R/Q` matrix deployable, but is not a result, metric, cost, energy or
  paper claim. P1 must still fail close on its own input and per-leaf runtime
  attestations before model/data/checkpoint access.

- 2026-08-15: final executable `519afd85ec18484c8cdd91c68ca3db52e404ad35`
  (branch `codex/zoomtoken-p1-f0-binding-v001`) fixed only P1's official-DO
  hash binding and uses the N16R4 site-default memory after the site rejected
  explicit `--mem`; 17 focused static checks passed. Its F0 root
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_f0_519afd85_20260815_0425` is
  terminal `FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED` with PL/ST/world2/finalizer
  `1238347/1238348/1238349/1238350` all `COMPLETED 0:0`; official test,
  performance and paper permissions remain false. The single real-video P1
  root `/data/run01/sczc063/yuzibo/zoomtoken_p1_dnurq_519afd85_seed3407_20260815_0430`
  was atomically released as `RELEASED_ATOMIC_P1_FIFTEEN_JOB_DAG`: runtime
  `1238367`, `DO/DN/U/R/Q` stages `1238368/1238369/1238370/1238371/1238372`,
  eight cost leaves `1238373–1238380`, and finalizer `1238381`. At intake the
  runtime preflight was Priority and all dependents pending. This is an
  experiment-running status only: no model, accuracy, latency, energy, cost,
  checkpoint, prediction or paper conclusion exists. Earlier F0 source-role,
  pre-submit hash, and resource-shape failures remain infrastructure negatives;
  no duplicate, requeue, retry or partial-arm interpretation is permitted.

- 2026-08-15: the one real-video P1 namespace
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_dnurq_519afd85_seed3407_20260815_0430`
  is terminal `INVALID_P1_MATRIX / NO_SURVIVOR_INVALID_P1`. Runtime `1238367`,
  stages `1238368–1238372`, cost leaves `1238373–1238380`, and finalizer
  `1238381` all exited within 1–2 seconds. No runtime-preflight, video, model,
  checkpoint, metric, cost, performance or paper evidence was created. The
  exact launch defects are: checking for `apptainer` before loading its module;
  retaining forbidden inner `srun --mem=192000M`; and finalizer validation of
  JSON insertion order (`tuple(mapping)`) after sorted-key serialization rather
  than the exact key set. This is an immutable launch implementation negative,
  not a Q-core falsification; this namespace cannot be retried, requeued,
  resumed, supplemented or read as a partial result.

- 2026-08-15: a distinct correction epoch is clean/pushed at
  `57ffdf32a26629d73ef161dffa90b85199441425`. It sources `/etc/profile` before
  nounset, loads apptainer only immediately before host container entry, inherits
  site-default inner-step memory, and validates stage/cost maps by exact key set.
  Focused checks (`17`) and `bash -n` pass; this is implementation evidence only.
  The sole active no-performance F0 is
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_f0_57ffdf32_20260815_0440`, with
  PL/ST/world2/finalizer `1238436/1238437/1238438/1238439`. Only a fully matching
  terminal PASS may authorize one new P1 epoch; no old P1 artifact or identifier
  is reusable, and no duplicate F0/P1 submission is permitted.

- 2026-08-15: the exact-commit F0 for
  `57ffdf32a26629d73ef161dffa90b85199441425` is terminal PASS at
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_f0_57ffdf32_20260815_0440`:
  PL/ST/world2/finalizer `1238436/1238437/1238438/1238439` all completed `0:0`
  and finalization is `FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED`. It contains no
  model result, test, performance, cost, or paper evidence. It released one
  distinct corrected P1 root
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_dnurq_57ffdf32_seed3407_20260815_0445`:
  runtime `1238455`; `DO/DN/U/R/Q` `1238456–1238460`; cost leaves
  `1238461–1238468`; and finalizer `1238469`. The scientific contract is
  unchanged, but this is a fresh implementation epoch. At intake the runtime
  job was pending by Priority, so no preflight, model, checkpoint, metric,
  latency, energy, cost, official-test, or paper conclusion exists. Monitor only
  this exact population; the invalid `519afd85…` epoch remains sealed and is
  never retried or reused.

- 2026-08-15: the distinct P1 namespace
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_dnurq_57ffdf32_seed3407_20260815_0445`
  is terminal `INVALID_P1_MATRIX / NO_SURVIVOR_INVALID_P1`. Runtime `1238455`,
  stages `1238456–1238460`, cost leaves `1238461–1238468`, and finalizer
  `1238469` all failed before video, model, runtime-attestation, checkpoint,
  metric, or cost evidence. The SIF lacks `git`, while the stage script repeated
  `git rev-parse` after container entry. This is a deterministic
  source-identity handoff infrastructure negative, not a Q-core falsification
  or real-video result; the namespace is sealed and cannot be retried, requeued,
  resumed, supplemented, or partially interpreted.

- 2026-08-15: clean pushed `99cb99a2694812cc0f58694001c24bb22a9bf578` makes
  only the source-identity handoff correction that avoids depending on
  inner-container `git`; bash syntax and the focused static suite (`17`) pass,
  and the remote source is clean. Its sole no-performance F0 is
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_f0_99cb99a2_20260815_0500`, with
  PL/ST/world2/finalizer `1238474/1238475/1238476/1238477`. It awaits central
  classification. No duplicate P1 epoch is submitted, and this F0 has no
  efficacy, test, metric, cost, energy, or paper-evidence meaning.

- 2026-08-15: exact-commit F0 for clean pushed
  `99cb99a2694812cc0f58694001c24bb22a9bf578` is terminal at
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_f0_99cb99a2_20260815_0500`: PL/ST/
  world2/finalizer `1238474/1238475/1238476/1238477` all completed `0:0`, and
  finalization is `FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED` with official test
  closed. It is no-performance protocol admission only. The sole distinct
  real-video P1 root
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_dnurq_99cb99a2_seed3407_20260815_0505`
  was atomically released with runtime `1238494`; `DO/DN/U/R/Q`
  `1238495–1238499`; cost leaves `1238500–1238507`; and finalizer `1238508`.
  No metric, runtime-attestation, video/model/checkpoint, latency, energy, cost,
  official-test, or paper result exists. Central alone monitors this epoch; no
  duplicate submission, cancellation, requeue, retry, or partial-arm inference
  is permitted.

- 2026-08-15: the P1 epoch
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_dnurq_99cb99a2_seed3407_20260815_0505`
  is terminal pre-data `INVALID_P1_MATRIX / NO_SURVIVOR_INVALID_P1`. Runtime
  `1238494`, stages `1238495–1238499`, cost leaves `1238500–1238507`, and
  finalizer `1238508` stopped before data, model, checkpoint, metric, or cost
  work because N16R4 RTX4090 `nvidia-smi` emits the literal MIG field `[N/A]`,
  while the runtime attestor accepted only `N/A`. This is a deterministic
  attestor-normalization infrastructure negative, not Q-core falsification or
  real-video evidence; the namespace is sealed and never retried or reused.

- 2026-08-15: independent single-GPU probe `1238509` reproduced `[N/A]`.
  Clean pushed `2683906203c46ef9201b263787bfe614b636d0b4` normalizes only
  `[N/A] -> N/A`; its focused suite (`17`) passes. Independent two-GPU P1
  runtime preflight `1238510` passes the exact RTX4090/SIF/Torch 2.0.1/CUDA 11.8/
  dependency class without data or model access. The sole new no-performance F0
  is `zoomtoken_p1_f0_26839062_20260815_0520`, jobs
  `1238515/1238516/1238517/1238518`, currently running. No duplicate full P1 is
  authorized before matching F0 PASS; none of these receipts is efficacy, test,
  metric, cost, energy, or paper evidence.

- 2026-08-15: exact-commit F0 for clean frozen
  `75c8f6e8c2f433c85ed8b8d488f3c867e5652d6b` passed at
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_f0_75c8f6e8_20260815_0545`:
  PL/ST/world2/finalizer `1238554/1238555/1238556/1238557` all completed `0:0`.
  Finalization is `PASS_OFFICIAL_COMPARABLE_PREFLIGHT_ONLY /
  FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED`; the world-two FP32 DDP KAT passed and
  official test remains closed. Its finalization file identity is
  `a2a146694f863ca32773294cc583b168e79d9cda44c10c6119875f5b63c9b9a8`.
  This is protocol admission only, not performance evidence.

- 2026-08-15: the sole real-video P1 root
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_dnurq_75c8f6e8_seed3407_20260815_0520`
  was atomically submitted under `ZOOMTOKEN_P1_DNURQ_V001`, seed 3407, with
  `DO/DN/U/R/Q`. Runtime preflight is `1238573`; accuracy cells are
  `1238574–1238578`; cost leaves `1238579–1238586`; and finalizer `1238587`.
  The real THUMOS14 validation inputs, official AdaTAD config, VideoMAE
  checkpoint, source commit/ref, SIF, dependency lock and prior F0 finalization
  are bound. Q retains global exact `B=24576`, dynamic `K_t`, ragged no-padding
  execution; Q/DN cost uses the frozen 136-window/40-cluster/10k-bootstrap
  one-sided upper-bound `<=0.85` gate, while DO is mandatory report-only.
  Official test remains closed. This is `experiment_running` only: no metric,
  checkpoint, latency, energy, cost, test, or paper result exists. Only a
  complete matching finalizer can determine the next step; no duplicate,
  cancellation, requeue, retry, or partial-arm inference is permitted.

- 2026-08-15: P1 root
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_dnurq_75c8f6e8_seed3407_20260815_0520`
  is terminal `NO_SURVIVOR_INVALID_P1` before video or model access. Runtime
  preflight passed, but all accuracy/cost leaves failed before data, model or
  checkpoint creation because the minimal SIF intentionally contains no `git`
  while the container stage runner repeated `git status`. Jobs `1238573–1238587`
  are terminal, with completed cells `0` and cost leaves `0`. This is a
  deterministic container source-identity infrastructure negative, not Q-core
  evidence; the namespace is sealed and is never retried, resumed or reused.

- 2026-08-15: clean pushed `fb36fdd6e6a042ce6ed50bde92c3a0db28bdb023` makes
  only the deterministic source-identity handoff closure: inside P1 containers
  it consumes the already validated outer source-identity receipt, while host
  and non-P1 execution retain `git status`. The focused static suite passes
  (`18`) and the remote source is clean/exact. The sole new no-performance F0
  root is `/data/run01/sczc063/yuzibo/zoomtoken_p1_f0_fb36fdd6_20260815_0528`,
  jobs `1238592/1238593/1238594/1238595`, currently running. No duplicate or
  resumed P1 is authorized before matching F0 PASS, and no efficacy, test,
  metric, cost, energy, or paper evidence exists.

- 2026-08-15: exact-commit F0 for `fb36fdd6e6a042ce6ed50bde92c3a0db28bdb023`
  is terminal PASS: PL/ST/world2/finalizer `1238592/1238593/1238594/1238595`
  all completed `0:0` and finalization is `FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED`.
  It is a no-performance protocol gate with no official-test opening.
  The sole distinct real-video P1 root
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_dnurq_fb36fdd6_seed3407_20260815_0535`
  is atomically submitted: runtime `1238611`; `DO/DN/U/R/Q` `1238612–1238616`;
  cost leaves `1238617–1238624`; finalizer `1238625`. It preserves the frozen
  THUMOS14/AdaTAD/VideoMAE, seed-3407, exact-`B=24576`, dynamic-`K_t`,
  ragged/no-padding, 136-window/40-cluster/10k-bootstrap and DN-only `<=0.85`
  contract; official test is closed. This is an experiment-running status only,
  not metric, checkpoint, latency, energy, cost or paper evidence. Monitor only
  this terminal path; no duplicate, cancellation, requeue, retry or partial-arm
  inference is permitted.

- 2026-08-15: a read-only Slurm terminal check of the sole `fb36fdd6…` P1
  population records runtime preflight `1238611` `COMPLETED 0:0`; accuracy
  cells `1238612–1238616`, all cost leaves `1238617–1238624`, and finalizer
  `1238625` are `FAILED 1:0`. A subsequent exact read-only finalization-receipt
  read failed at SSH banner exchange, so the terminal causal receipt is not
  available. This is a `PRE_RUN_NOT_READY` infrastructure blocker, not Q-core,
  accuracy, latency, energy, cost, checkpoint, official-test, or paper evidence.
  The root is terminal and is neither retried nor interpreted from partial arms.

- 2026-08-15: clean pushed `84475fab32e95d9b4e33982f70ce17ce2132bb1e` closes
  the deterministic outer-source-identity handoff. Its exact-commit F0 root
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_f0_84475fab_20260815_0540` passed:
  `1238630/1238631/1238632/1238633` completed `0:0`, finalization is
  `FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED`, performance inference is false, and
  the receipt identity is `41a812efeeb19b305985a8d8870fa4bba0400727b569b7d266cdad5d9ce3b17a`.
  This is protocol admission only.

- 2026-08-15: the sole new development P1 root
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_dnurq_84475fab_seed3407_20260815_0548`
  was atomically released: runtime `1238649`; `DO/DN/U/R/Q` `1238650–1238654`;
  cost leaves `1238655–1238662`; finalizer `1238663`. A read-only status check
  records runtime `COMPLETED 0:0`, stage jobs pending priority, and cost/finalizer
  pending dependency. It preserves THUMOS14 development, official AdaTAD/VideoMAE,
  seed 3407, exact `B=24576`, dynamic `K_t`, ragged/no-padding, 136/40/10k and
  DN-only `<=0.85`; official test is closed. It is `experiment_running` only:
  no metric, checkpoint, latency, energy, cost, efficacy or paper result exists.
  Monitor only this exact population, never infer partial arms, and leave unrelated
  Job `1237479` untouched.

- 2026-08-15: the `84475fab…` P1 finalization receipt is terminal
  `INVALID_P1_MATRIX / NO_SURVIVOR_INVALID_P1`. Runtime preflight `1238649`
  completed `0:0`; all five stage jobs `1238650–1238654`, all eight cost leaves
  `1238655–1238662`, and finalizer `1238663` ended `FAILED 1:0`. Finalization
  records `0/5` completed accuracy cells, `0/8` completed cost leaves,
  `q_survives_p1=false`, no partial-arm conclusion, five stage-failure receipts
  and eight missing cost leaves. It does not establish a Q-core or performance
  failure. This root is sealed: no retry, resume, requeue, cancellation, or
  partial inference; unrelated Job `1237479` remains untouched.

- 2026-08-15: the common five-arm pre-model/video cause is now classified from
  the terminal stage-failure receipts: the P1 runner pre-created `cell_root`
  for a storage receipt and bound the same existing path as formal training's
  required-fresh `work_dir`, causing `FileExistsError`. All cost leaves therefore
  lacked source stages. This is deterministic infrastructure, not Q-core,
  accuracy, latency, energy, cost, checkpoint, test or paper evidence.

- 2026-08-15: clean pushed `f7a516b9cf37d187deb09ec5e5362ab3947fb286` makes
  only this claim-preserving closure: storage preflight moves below `control/`
  without creating the formal work directory, while the durable pre-train
  failure path remains. Focused static checks (`20`) pass and the remote checkout
  is clean/exact. Its sole exact-commit F0 root
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_f0_f7a516b9_20260815_0600`, jobs
  `1238668/1238669/1238670/1238671`, is active. It is a no-performance recovery
  gate; no P1 or efficacy/cost inference is authorized before terminal matching
  PASS. Unrelated Job `1237479` remains untouched.

- 2026-08-15: exact-commit F0 `f7a516b9…` is terminal PASS. Jobs
  `1238668/1238669/1238670/1238671` all completed `0:0`; finalization is
  `PASS_OFFICIAL_COMPARABLE_PREFLIGHT_ONLY /
  FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED`, with both official-comparable leaves
  and world-two FP32 DDP KAT passing. The receipt binds the exact runtime commit,
  frozen protocol manifest and origin-ref parity; official test, prediction,
  checkpoint, performance and paper permissions remain false. This makes the
  frozen P1 deployable only. No P1 is submitted here, no efficacy/cost result
  exists, and unrelated Job `1237479` remains untouched.

- 2026-08-15: the subsequent `f7a516b9…` P1 is terminal before model/video
  access. Removing the initial storage pre-create was insufficient because
  `_run_logged` created `cell_root/train.out`, again making formal work_dir
  non-fresh; earlier `729b328b…` also exposed required DDP rank-order
  synchronization. This is deterministic launcher/work-directory infrastructure,
  not Q-core, accuracy, cost, efficacy or paper evidence.

- 2026-08-15: clean pushed `5491c580490648d5bcebc9fedb318f663df6158c` makes
  only the narrow closure: storage and launcher logs are under `control/`, the
  formal work_dir remains absent until rank 0 creates it, then ranks synchronize.
  Scientific settings, data, metrics, arms and protocol are unchanged; focused
  checks (`21`) pass. Exact-commit F0
  `/data/run01/sczc063/yuzibo/zoomtoken_p1_f0_5491c580_20260815_0640`, jobs
  `1238744/1238745/1238746/1238747`, is active. It is no-performance protocol
  recovery only; no P1 or efficacy/cost inference is authorized before a
  terminal matching PASS.

- 2026-08-15: exact-commit F0 `5491c580…` is terminal PASS. Jobs
  `1238744/1238745/1238746/1238747` all completed `0:0`; finalization is
  `PASS_OFFICIAL_COMPARABLE_PREFLIGHT_ONLY /
  FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED`, with both official-comparable leaves
  and world-two FP32 DDP KAT passing. Exact runtime commit, frozen input/protocol
  identities and origin-ref parity match; official test, prediction, checkpoint,
  performance and paper permissions remain false. This makes the frozen P1
  deployable only. No P1 is submitted here and no efficacy/cost result exists.

- 2026-08-15: the sole claim-bearing real-video development P1 is atomically
  running at `/data/run01/sczc063/yuzibo/zoomtoken_p1_dnurq_5491c580_seed3407_20260815_0650`:
  runtime `1238763`; `DO/DN/U/R/Q` `1238764–1238768`; cost leaves
  `1238769–1238776`; finalizer `1238777`. Runtime preflight passed. Checkpoint
  loading and epoch 0 are durable execution facts; DN/U/R/Q each report `25/79`
  successful updates and DO has started without a comparable cadence receipt yet.
  Frozen THUMOS14 development, official AdaTAD/VideoMAE, seed 3407, exact
  `B=24576`, dynamic `K_t`, ragged/no-padding, 136/40/10k and DN-only `<=0.85`
  remain unchanged; official test is closed. This is experiment-running only,
  not metric, latency, energy, cost, efficacy or paper evidence. Only the full
  finalizer may decide the next action; unrelated Job `1237479` remains untouched.

- 2026-08-15: durable scheduler intake for the same sole P1 records DN/U/R
  stages `1238765/1238766/1238767` as `COMPLETED 0:0`; DO `1238764` and
  Q `1238768` remain running, with the eight cost leaves and finalizer still
  dependency-pending. This is execution bookkeeping only: no stage output,
  partial-arm measurement, latency, energy, cost, efficacy or paper inference
  is admitted before the complete matching finalizer. Job `1237479` remains
  untouched.

- 2026-08-15: the sole `5491c580…` P1 population is terminal in its one
  authorized read-only status check. Runtime `1238763` and DN/U/R/Q
  `1238765/1238766/1238767/1238768` are `COMPLETED 0:0`; DO `1238764`,
  all eight cost leaves `1238769–1238776`, and finalizer `1238777` are
  `FAILED 1:0`. The monitor was authorized for status only and did not read a
  finalization receipt, so it does not infer the failure cause. The matching
  matrix is `PRE_RUN_NOT_READY`; no partial arm, metric, latency, energy, cost,
  efficacy or paper conclusion is admissible. This root is not submitted,
  canceled, retried, resumed, supplemented, or otherwise altered; Job
  `1237479` remains untouched.

- 2026-08-15: the authoritative `control/finalization.json` now supersedes the
  earlier status-only limitation for
  `zoomtoken_p1_dnurq_5491c580_seed3407_20260815_0650`. It seals the matrix as
  `INVALID_P1_MATRIX / NO_SURVIVOR_INVALID_P1`: completed accuracy cells `0/5`,
  completed cost leaves `0/8`, and `official_test_opened`, `paper_claim_allowed`,
  and `partial_arm_conclusion_allowed` are all false. DN/U/R/Q completed `0:0`
  but their stage results were rejected as `formal artifact changed: config_path`.
  DO reached an epoch-59 checkpoint and then failed in official dense test because
  no GeoRoute window telemetry was emitted. Cost leaves fail or are missing because
  the pass changed the frozen 136-window/40-video population. This is a terminal
  implementation/protocol-admission defect—not Q-core efficacy/cost evidence and
  not a scientific STOP. The root is sealed: do not inspect or promote metrics,
  retry, requeue, resume, supplement, or infer from individual arms; official test
  remains closed and Job `1237479` remains untouched. A future distinct epoch needs
  explicit new authority plus deterministic correction review.

- 2026-08-15: under the user's explicit experiment-first instruction, a distinct
  full-official epoch started from clean commit `b88a11ba…`, without reusing the
  sealed `5491c580…` finalizer or its 136-window/40-video population. Slurm array
  `1238928_[0-4]` runs five complete THUMOS14 training/evaluation arms: official
  AdaTAD dense reference, matched-source dense control, uniform selection, seeded
  random selection, and Q content-driven dynamic spatial routing. All arms use the
  full official training subset for 60 epochs and the official evaluation subset;
  all five reached epoch 0 with finite losses. Run root:
  `/data/run01/sczc063/yuzibo/zoomtoken_full_official_b88a11ba_seed3407_20260815_070546`.
  This is active training evidence only; no final accuracy, latency, energy, cost,
  or paper conclusion exists yet.

- 2026-08-15: the same complete five-arm experiment was submitted for independent
  seeds 3408 and 3409 as Slurm arrays `1238939_[0-4]` and `1238940_[0-4]` from
  clean commit `8b4f4ce9…`. Together with running seed 3407 array `1238928_[0-4]`,
  this is a 3-seed × 5-arm full-training matrix. The two new arrays are queued;
  no result or cross-seed inference exists yet.

- 2026-08-16: deterministic completion fixes were isolated in clean source `a6ff4921…`: the official dense arm now binds the absolute THUMOS14 annotation path, and completed checkpoints are resolved under the actual `gpu2_id0/checkpoint` directory. Completion array `1239607` reuses existing 60-epoch checkpoints only for official evaluation and trains only missing arm/seed cells. Matched-source full-compute seeds 3407 and 3408 completed official validation: Avg-mAP 66.42% and 67.14%; mAP@0.7 45.19% and 45.84%. These are admitted baseline measurements, not evidence that Q succeeds or fails. DO/U/R/Q missing cells and remaining evaluations continue under the same full official matrix; no final accuracy/cost/energy or paper claim is made before all five arms and three seeds finish.

- 2026-08-16: seed-3408 uniform and random sparse controls completed official validation. Uniform: Avg-mAP 60.05%, mAP@0.7 40.17%; random: Avg-mAP 61.53%, mAP@0.7 41.80%; the matched dense result is 67.14%/45.84%. This is evidence that naive equal-budget sparsification loses accuracy, not yet evidence for Q, because the task-aware route has not returned. Q and the other missing arm/seed cells continue under the same full official matrix.

- 2026-08-16: seed-3408 Q completed official validation from its full 60-epoch checkpoint: Avg-mAP 57.84%, mAP@0.7 36.93%. It is below matched dense by 9.30 Avg-mAP points and below uniform/random by 2.21/3.69 points. This is the first admissible method-level negative result for Q-core: content-only dynamic routing did not recover the lost spatial evidence for this seed. Remaining seeds continue; no cross-seed or cost claim is made yet.

- 2026-08-16 14:20: matched dense seed 3409 completed official validation: Avg-mAP 65.99%, with tIoU 0.3/0.4/0.5/0.6/0.7 = 81.04/76.65/68.82/58.41/45.02%. Together with matched dense seeds 3407/3408 (66.42/67.14 Avg-mAP), the full-compute reference is stable across three seeds. Seed-3409 U/R/Q and the remaining seed-3407 arms continue. Existing seed-3408 Q output has diagnostic telemetry disabled, so the current artifacts establish the negative score but do not identify raw selected-token positions or dynamic-K failure modes; a frozen-checkpoint replay with telemetry is the next diagnostic after a GPU slot opens.

- 2026-08-16: clean diagnostic commit `1080dc13…` implements that replay without retraining. It enables complete official-validation Q telemetry at local batch one and summarizes per-tubelet K distribution, zero-budget fraction, within-window K variation, geometry saturation and route-role fractions. Focused test: 1 passed; remote clean source `/data/run01/sczc063/yuzibo/runtime/zoomtoken_q_diag_1080dc13/source` matches the pushed commit. The replay waits for a submit slot and is not yet a result.

- 2026-08-16 14:50: uniform seed3409 completed official validation: Avg-mAP 60.55%, tIoU 0.3/0.4/0.5/0.6/0.7 = 75.62/70.75/63.04/53.00/40.32%. This closely matches uniform seed3408 60.05/40.17 and strengthens the sparse-control anchor. The first telemetry replay submission was rejected before job creation because its explicit 64GB request exceeded the one-GPU platform shape. Clean `27181b0f…` removed the explicit memory request, focused tests 2 passed, N16R4 resource test passed, and checkpoint-only official-validation replay `1239655` was submitted. It is queued, so no route telemetry result exists yet.
- 2026-08-16 15:16: seed3409 random and Q completed official validation. Random: Avg-mAP 61.41%, tIoU 0.3/0.4/0.5/0.6/0.7 = 75.97/71.40/63.43/54.30/41.95%. Q: Avg-mAP 53.81%, tIoU = 68.87/63.67/57.03/46.06/33.40%. Matched dense/uniform for the same seed are 65.99/60.55 Avg-mAP. Q is therefore below all same-source controls for a second independent seed. This is a repeated formal negative for content-only routing; the frozen-checkpoint telemetry job remains the evidence-bearing next step for root-cause analysis, while seed3407 and the official public configuration continue.

- 2026-08-17: completed a read-only full Wiki memory and data-resource audit
  (`WIKI_MEMORY_AUDIT-2026-08-17.md`). The decisive provenance correction is
  that published AdaTAD `Avg=69.03, mAP@0.7=48.27` is an upstream anchor, while
  dense `66.42/67.14/65.99` and all current Q/U/R outputs are matched-source
  THUMOS14 validation outputs, not an exact untouched AdaTAD reproduction.
  Canonical remote raw video is complete (411 MP4: train 200, validation 211,
  UID mismatch 0), annotations/class map and VideoMAE-S pretrain are present;
  the released AdaTAD checkpoint is not verified in known paths and its access/
  license receipt is absent. The next action is released-checkpoint evaluation
  in clean `01c58b9` official code, not another routing experiment. Future full
  runs save resumable `.pth` every five epochs unless untouched official code is
  more frequent (AdaTAD's official recipe remains every two epochs), retain at
  least three recovery points, and preserve final/final-EMA selection without
  intermediate cherry-picking. This audit ran no model, data, GPU, Slurm, metric
  or cost job.

- 2026-08-17: central read-only resource correction confirms that canonical
  THUMOS14 raw input is the shared root
  `/data/run01/sczc063/yuzibo/thumos14/raw_data/video`, with 411 valid MP4
  **symlinks** (200 training / 211 validation, zero broken, target volume about
  33G). Project configs/manifests must bind it directly without duplication and
  must not undercount it via `find -type f`. Feature identities are now explicit:
  OpenTAD I3D and InternVideo2 roots are absent; native MATR validation/test
  pickles are present (~3.33G/~3.69G) but its checkpoint is incomplete; SigLIP2
  has 823/~477M assets. These formats are non-interchangeable and none is a
  substitute for VideoMAE raw E2E AdaTAD. This is resource provenance only—no
  download, symlink, config change, remote job, metric or efficacy claim occurred.

- 2026-08-17: full remote video resource map was recorded read-only. THUMOS14
  remains the only ZoomToken input: canonical 411 valid symlinks map exactly to
  200 training and 211 validation annotations; two extra physical-store files
  (`video_test_0000270.mp4`, `video_test_0001292.mp4`) are noncanonical. Other
  inventory is deliberately not conflated with this route: MultiSports is still
  mostly archived, TOC-Bench and Charades are separate dataset trees, ActivityNet
  is unassembled, FineAction/HACS/EPIC-Kitchens/Ego4D videos are absent, and
  EventMATR provides only native feature pickles. No archive was extracted, no
  config/manifest was bound, and no experiment was authorized.

- 2026-08-17: read-only inspection of the historical seed-3407 full-matrix
  root records concrete source identities: DO snapshot inherits the official
  AdaTAD `e2e_thumos_videomae_s_768x1_160_adapter.py`; DN inherits
  `georoute_adatad_development_base.py`; Q inherits
  `georoute_dynamic_scnr_stage1_base.py`; U/R inherit Q, all against canonical
  shared THUMOS14 and VideoMAE-S pretrain. DN/Q/U/R contain
  `checkpoint_interval=60`. This is a historical matched-source configuration
  fact, not a result or exact official reproduction. The newly frozen policy
  requires Builder to set 5-epoch resumable checkpoints (latest three plus
  milestone/final; model/optimizer/scheduler/scaler/count/RNG restoration) for
  future non-untouched-official full runs; untouched official AdaTAD retains 2.
  No configuration was changed in this audit.

- 2026-08-17: user required the original AdaTAD baseline to become a **single
  shared execution**, with ZoomToken as sole owner. Prepared
  `docs/aris/ADATAD_SHARED_OFFICIAL_BASELINE_PACKET-2026-08-17.md`: one clean
  released-checkpoint evaluation first, then at most one clean untouched
  official training only if the released artifact is genuinely unavailable.
  Its final receipt must bind release/config/canonical 411/checkpoint-or-
  pretrain/seed/evaluator-NMS/EMA-final/runtime/result root; all other TAD
  projects consume it read-only. This does not relabel 66.xx matched-source
  dense or stop ZoomToken's non-execution Q-entry, conditional ROI/residual,
  recovery-checkpoint, review and PRE_RUN preparation. No job, download,
  experiment, metric, cost or claim was created by this governance update.

- 2026-08-17: the requested independent DSH review was submitted with the fixed
  anchored-standard / deepseek-official / deepseek-v4-pro / max / 256000 profile
  and a fresh empty session, but terminated before its first assistant response
  with `402 Insufficient Balance`. It is recorded as `NEEDS_ATTENTION`, not DSH
  PASS, code approval, a scientific verdict or a new experiment. Existing
  evidence boundaries remain: static wiring is not whole-implementation proof;
  66.xx stays matched-source; historical P1 runner/protocol failures remain
  infrastructure/protocol negatives. No automatic resubmission is permitted.

- 2026-08-17: accepted the current Track-B minimal-change plan only after
  rebinding Builder from obsolete clean `020bade…` to clean project-current
  `5491c580…`, which contains that lineage and the sealed P1 fixes. The bounded
  implementation corrects only formal-matrix admission: control-bound config
  identity, external shared-receipt DO handling, canonical physical
  136-window/40-video population identity, and five-epoch full-state recovery
  for future DN/U/R/Q. Q exact-B/dynamic-K_t/ragged/no-leak behavior and
  conditional G/N/F remain unchanged; the sealed 5491 root is not resumed or
  reused. This is implementation preparation, not a result or remote launch.

- 2026-08-17: a fresh DSH external reviewer session completed under the exact
  anchored-standard / deepseek-official / deepseek-v4-pro / max / 256000
  identity. `docs/aris/DSH_FORMAL_REVIEW_RECEIPT-2026-08-17.md` stores the
  first header, the sole `/^We need\b/` fingerprint and `turn/end=completed`;
  the visible Chinese report is separate. It preserves the Q-core endpoint and
  labels 5491 as protocol-admission failure, not efficacy. Its advisory
  implementation findings are: make any DO report-only cost path independent
  of GeoRoute telemetry, make 5-epoch recovery actually resumable only for
  unsealed bound cells, and bind a shared official receipt to the real
  69.03/48.27 anchor and provenance. These are inputs for the existing
  Builder→Critic→Evaluator chain, not a code approval, PRE_RUN, experiment,
  metric, cost or paper claim.

- 2026-08-17: the final fresh frozen-snapshot external review of clean
  `b798e9f…` was accepted after the two earlier fingerprint-miss sessions were
  quarantined. It preserved the Q-core mechanism and reported deterministic
  admission/recovery concerns: DO report-only cost must not require GeoRoute
  telemetry or a mismatched dense grid; the shared official receipt must bind
  the published anchor and result identity; and the advertised five-epoch
  recovery path must be actually reachable only for unsealed bound cells. These
  are advisory implementation inputs, not a Critic replacement, a route change,
  performance evidence, or experiment authority. The candidate remains held
  because the independent Critic returned no substantive audit after its bounded
  process recovery.

- 2026-08-18: a newly isolated read-only Critic completed the frozen
  `b798e9f…` audit on a clean worktree. It independently preserved the Q-core
  exact-budget, Q/DN cost-denominator, conditional G/N/F, fairness and
  no-leakage boundaries, but returned `IMPLEMENTATION_CORRECTION`: five-epoch
  recovery state is written yet no formal unsealed-cell resume dispatch can
  reach it, and runtime population agreement is not tied to the frozen ordered
  136-window/40-video manifest. The accepted external review additionally
  showed that report-only DO must not run a GeoRoute 220-token cost leaf and
  that shared official provenance must prevent matched-source `66.xx` from
  entering the official slot. These are admission-code defects, not Q efficacy,
  cost, official-baseline or paper evidence. A bounded Builder MCL is now the
  sole next step; no sealed root is resumed and no PRE_RUN/remote job starts.

- 2026-08-20: completed a read-only ROI-history audit after the user requested
  an official-comparable 60-epoch ROI experiment. The audit distinguishes the
  20-epoch ROI-only diagnostic (13.18 Avg-mAP, 8.95 mAP@0.7), a separate
  20-epoch ROI run that failed at decoding before formal evaluation, and the
  continuous-ROI 60-epoch training-only matrix from a valid official comparison.
  A remote search of the shared BATA run namespace found no omitted ZoomToken
  ROI terminal receipt. The shared official AdaTAD released-checkpoint receipt
  remains absent; 66.xx remains matched-source rather than official. Prepared
  `docs/aris/ROI_OFFICIAL_COMPARABLE_60EPOCH_MATERIAL-2026-08-20.md` for one
  narrow scientific decision on a fair ROI-first 60-epoch contract. No code,
  remote job, data access, metric, cost, or performance claim was created.

- 2026-08-20: ingested the bounded Track-B Builder successor
  `7f0d0eb018ca2e6c9d1774e71c214648a18ea88b` (parent `b798e9f…`). It changes
  eight existing admission paths only: report-only DO/shared official receipt,
  ordered 136-window/40-video manifest binding, and reachable five-epoch
  unsealed same-cell full-state resume. `py_compile`, shell syntax, and 44
  focused static tests passed in a clean worktree. An independent, read-only
  Critic recheck has been requested; no ROI arm, data, Torch, GPU, Slurm,
  remote job, cost measurement, result or paper claim was created.

- 2026-08-20: user clarified that the immediate objective is an effective ROI
  model rather than an over-designed first matrix. Narrowed the prepared ROI
  material to a real 60-epoch, seed-3407 ROI-only `G` versus same-source dense
  `DN` first stage, with shared official `DO` receipt only as report-only
  reference. First stage retains identical data/evaluator/recipe, fixed final
  selection and 5-epoch recovery checkpoints, and reports mAP plus same-runtime
  end-to-end latency/peak memory. Multi-arm, multi-seed and energy-bootstrap
  work is deferred until a credible first ROI signal; this simplification does
  not recast development evidence as an official or paper result.

- 2026-08-20: direct user-authorized Pro review was successfully submitted
  through iXBrowser profile 61 after earlier pre-submission transport failures
  were terminated without producing content. Oracle job `j-vqxlwg` created fresh
  Project conversation `6a870bf0-7768-83ea-8a03-4563579904f5` with
  `promptSubmitted=true`. The substantive review names only the pinned GitHub
  repository `yuzbo/OpenTAD_C3_CoarseClean_20260702`, branch
  `codex/zoomtoken-p1-fix-dn-cost-budget-label`, SHA `2e99ce0…`; no source code
  files or archives were uploaded. It will decide the minimal 60-epoch ROI G
  versus DN development experiment. No implementation, data, GPU, Slurm,
  metric, cost, training or claim was started by this consultation.

- 2026-08-20: 核验了用户提供的 ROI60 外部建议。采纳其将首轮收敛为同源
  `G`（ROI modifier-only）对 `DN`（dense）、seed 3407、60 epoch 的方向；不采纳其将 `7f0d0eb...`
  当作最新 GitHub 代码、无条件复用旧 DN、或在 final/EMA 间按 validation 择高的做法。指定 GitHub
  分支的 `2e99ce0...` 是 `7f0d0eb...` 后继，并已更正 DN 的稀疏 exact-B 标签。首轮 ROI 只能视为
  同源开发比较；共享 official AdaTAD receipt 前不得标为官方可比。报告中“latency 或 mAP@0.6 单独即
  升级”的逻辑已被拒绝；单种子需主要准确率与同机端到端时延共同支持才可以开展下一阶段。详见
  `docs/aris/ROI60_USER_SUPPLIED_PRO_REPORT_AUDIT-2026-08-20.md`。未启动代码、数据、GPU 或训练。

- 2026-08-20: 共享官方 AdaTAD released checkpoint 在已核验位置仍无可绑定副本，近似命名权重
  不能替代发布 artifact。因而按唯一 fallback，在 clean release `01c58b9...`、未经修改的官方
  THUMOS14 config、seed 42、canonical 411-video root、原始 validation evaluator/NMS 与 VideoMAE-S
  pretrain 上提交一次 60-epoch official reproduction。该作业仅处于运行状态；尚无 validation 指标，
  不得写成 `69.03/48.27` 已复现，也不提供 ROI、Q、成本或论文结论。启动绑定见
  `docs/aris/ADATAD_SHARED_OFFICIAL_REPRODUCTION_START_RECEIPT-2026-08-20.md`。

- 2026-08-21: ROI60 的 official-base 实现完成到 clean revision
  `321f1f767e730aad743ee1d4579803156ca0413f`。fresh external review 在前一候选中发现
  `GeoRouteSourceViews` 只有配置引用而未定义/注册；Builder 仅恢复了历史已审的 native-source +
  96×96 scout 变换，独立 Critic 确认语义与 lineage 一致且模型、预算、split、evaluator、NMS、
  recovery 未变。16 项静态测试通过；此前 N16R4 AMP/GeoRoute build smoke 通过。
- 2026-08-21: 用户已授权的真实 THUMOS14 ROI60 配对训练正式启动。远端源码 clean 且精确为
  `321f1f76…`；结果根
  `/data/run01/sczc063/yuzibo/bata_runs/zoomtoken_roi60_dn_g_seed3407_321f1f76_20260821`。
  DN job `1245897` 与 ROI-only G job `1245898` 原子提交后同时在 N16R4 运行；两臂 seed 3407、
  60 epoch、同一预训练/数据/评测/单卡资源，G 保持 exact `B=24576`、动态 `K_t`、ragged/
  no-padding、ROI on、residual off。每 5 epoch 保存 recovery，统一 final/final-EMA 规则。
  独立 direct-Decord smoke `1245895/1245896` 在 Python 进入数据前因集群进程
  `random_device` 异常失败，而正式 torchrun 已越过其短时故障点；它们不是模型或性能证据。
  共享 untouched AdaTAD job `1245842` 同时正常训练，当前也尚无终态指标。

- 2026-08-21: 上一条启动记录已被真实执行证据校正。首对 jobs `1245897/1245898` 均在 41 秒内因
  优化器参数组别名终止；G-only `1245908/1245909/1245910` 继续暴露并定位了 DDP 辅助损失图和
  实际 `ActionFormer.forward_train` 覆盖路径的问题。它们均未产生可解释的 validation 或成本结果。
  同源全计算 DN 的有效 60-epoch 运行是 job `1245907`，clean revision `d2b5de05…`，结果根
  `/data/run01/sczc063/yuzibo/bata_runs/zoomtoken_roi60_dn_g_seed3407_d2b5de05_20260821`；它已进入
  epoch 7，并发布 `recovery_epoch_4.pth`。最终 ROI-only G 的有效运行是 job `1245924`，clean
  revision `59960255…`，结果根
  `/data/run01/sczc063/yuzibo/bata_runs/zoomtoken_roi60_g_seed3407_59960255_20260821`；它已完成至少
  50 个有限损失的真实优化更新。共享 official AdaTAD `1245842` 已进入 epoch 29，并保持官方
  两轮 checkpoint 节奏。三个作业均仍在运行；这些事实只证明完整训练实质启动，不提供 ROI、
  `69.03`、效率或论文结果。

- 2026-08-21 03:44（北京时间）: ROI-only G job `1245924` 完成 epoch 4、进入 epoch 5，并在
  `/data/run01/sczc063/yuzibo/bata_runs/zoomtoken_roi60_g_seed3407_59960255_20260821/cells/g/seed3407/gpu1_id0/checkpoint/recovery_epoch_4.pth`
  写出首个约 628 MB 的 5-epoch 可恢复检查点。DN `1245907` 已进入 epoch 13，恢复点推进至
  epoch 4/9；共享官方 AdaTAD `1245842` 已进入 epoch 36，官方节奏检查点推进至 `epoch_35.pth`。
  三项作业均为 RUNNING，G 日志未见 Traceback、显存溢出或非有限数值硬故障。该记录只证明
  训练连续性与恢复机制，不是 ROI 性能、成本或 `69.03/48.27` 官方锚点复现证据。

- 2026-08-21 03:54（北京时间）: DN job `1245907` 进入 epoch 15，并新增第三个 5-epoch 周期
  恢复点 `recovery_epoch_14.pth`（约 628 MB）；共享官方 AdaTAD `1245842` 进入 epoch 38并新增
  `epoch_37.pth`；ROI-only G `1245924` 进入 epoch 6。三项作业仍为 RUNNING，检查范围内未见
  Traceback、显存溢出或非有限数值硬故障。该里程碑是训练连续性证据，不是 validation、ROI
  增益、端到端成本或官方锚点复现结果。

- 2026-08-21: 三项唯一 60-epoch 训练全部终态。共享 official AdaTAD `1245842`、matched-source
  DN `1245907`、ROI-only G `1245924` 均为 `COMPLETED 0:0`，运行时间分别为 `05:47:13`、
  `04:24:56`、`05:46:15`；三者均完成 epoch 59 的最后训练迭代，日志未见 Traceback、OOM 或
  非有限数值硬故障。DN/G 各保留 epoch 44/49/54 三个恢复点。指定结果根未发现
  `stage_result.json`、`finalization.json` 或 final-EMA validation 收据，因此本次只形成训练完成与
  可恢复性证据，没有 ROI、DN、`69.03/48.27` 或成本结果。不得重训或重复提交；下一动作是对
  既有终态 checkpoint 执行预注册选择和官方 evaluator 评测。

- 2026-08-21: 补充检查原始终态训练日志，纠正上一条“没有性能结果”的判断。官方 evaluator 已在
  日志末尾打印 validation：official AdaTAD reproduction 的 Avg-mAP/mAP@0.6/mAP@0.7 为
  `68.73/61.58/47.24`，DN 为 `64.73/56.14/43.26`，ROI-only G 为
  `61.49/53.42/39.99`。G 相对 DN 为 `-3.24/-2.72/-3.27` 个百分点，且 tIoU 0.3–0.7
  全部更低，构成当前 ROI-only 配置的准确率负结果。官方运行在最后 20 轮按 2 轮间隔完成 10 次
  validation；DN/G 由启动脚本设置为第 60 轮后单次 validation，所以没有中途性能曲线。三项均
  未产生完整端到端成本证据，不得据 token 预算声称效率收益。

- 2026-08-21: 为补齐 DN/G 的中途学习曲线，在不续训、不改变 final/final-EMA 选择、不打开 test
  split 的前提下，启动 epoch 44/49/54 recovery checkpoint 的 validation-only 评测。DN jobs 为
  `1246228/1246229/1246230`，G jobs 为 `1246231/1246232/1246233`，均绑定各自原始训练 revision、
  canonical THUMOS14 validation 与同一 evaluator/NMS。此前 `1246216/1246218/1246220/1246222/
  1246224/1246226` 因 Slurm `--wrap` 默认 `/bin/sh` 不支持 `source` 在 0 秒退出，未加载模型或数据，
  仅作为启动诊断保留。

- 2026-08-21: 六项中途 validation 全部 `COMPLETED 0:0` 并加载对应 EMA。epoch 44/49/54 的
  DN Avg-mAP/mAP@0.7 为 `65.50/44.01`、`65.06/43.44`、`64.84/43.36`；G 为
  `62.42/41.41`、`62.00/40.50`、`61.80/40.48`。G−DN Avg-mAP 稳定为
  `-3.08/-3.05/-3.04`，终态为 `-3.24`。这表明 ROI-only G 的准确率缺口贯穿训练后段，不能归因
  于最后几轮偶然退化；同一 G 配置不再延长训练或重复提交。

- 2026-08-21: 用户授权把本地 `E:\Released_FineDiving_Dataset` 完整上传至 N16R4
  `/data/run01/sczc063/yuzibo/datasets/FineDiving`，并继续准备必要 TAS 测评数据。FineDiving
  本地源已核定为 1,337,505 文件 / 101,123,416,820 字节；3,000 样本标注、2,251/749
  train/test、四套各 312,256 帧、135 个 MP4 和 15 个 ZIP 的结构/解码/CRC 检查均通过。
  远端上传仍为 `transfer_running`，最终必须通过全量相对路径+文件大小清单比对，当前不得称完整。
  FineGym 官方 annotations/categories/splits 已在远端完成 JSON 与官方计数校验；视频因官方表单
  授权边界未获取。标准 TAS 的 GTEA/50Salads/Breakfast 官方 MS-TCN/UVAST Zenodo feature/GT/
  split 包已在远端开始可续传下载，仍为 `download_running`；仅在字节数、ZIP CRC、解压和
  `COMPLETE` receipt 全部通过后才可用于测评。

- 2026-08-21: FineDiving 远端续传复核确认 `Untrimmed_Videos` 已达到本地精确基线
  135 文件 / 64,534,519,969 字节，`Lowresolution02_Trimmed_Video_Frames` 已达到
  312,256 文件 / 706,600,997 字节；未裁剪帧与原分辨率 trimmed 帧仍在传输/展开，
  `lowres01/005` 改为每 100 个样本目录写入一个可恢复断点后继续，整体仍保持
  `transfer_running`。官方 GTEA/50Salads/Breakfast Zenodo 下载进程和 16 路断点文件均存活，
  仍保持 `download_running`，不提前登记为可用。

- 2026-08-21: 用户澄清 TAS 数据要求为 GTEA、50Salads、Breakfast 的原始视频与原始标注，
  不是 MS-TCN/UVAST 预提取 feature 包。后者下载已停止但保留可恢复断点，不删除现有文件。
  新目标统一放在 `/data/run01/sczc063/yuzibo/datasets/TAS/raw`，且所有外部请求均显式使用
  N16R4 登录节点学术代理。Breakfast coarse/fine segmentation 已下载并通过完整 tar/gzip
  检查，3,930,212,562 字节原视频包仍在 Range 续传和终态校验；GTEA 官方 28 视频/71 类
  标注与 50Salads 官方 50 个 RGB AVI 因代理对 Dropbox/Dundee 旧主机返回 HTTP 503，保持
  低频 `download_retrying`。当前仅 Breakfast 标注可称完整，三套原视频均不得提前称可用。

- 2026-08-21 23:45（北京时间）：为严格分离 sparse adapter 与 ROI 的准确率影响，在 clean
  revision `1a18565bbee5fdb08969b754881d0b06f3429870` 上完成后主干三臂归因实现并通过独立
  Critic：A 复用已完成的未修改官方 AdaTAD job `1245842`；B 保持完整官方 dense VideoMAE
  前向，仅把空间平均聚合替换为全部 token 的 sparse adapter；C 与 B 共享同一主干、adapter、
  seed 和优化配方，仅在主干输出后加入 ROI `K=64` 支持选择。该设计不剪枝重骨干，因此只用于
  准确率/因果归因，不能作为计算节省证据。结果盲 PRE_RUN 验证 remote ref/HEAD clean、canonical
  411 视频与注释/类别/VideoMAE-S 权重、双卡 local batch 1/global batch 2、seed 42、官方增强、
  优化器/调度器、AMP/EMA/evaluator/NMS、全新结果根、存储和无重复作业后，提交并同时释放
  B job `1247290` 与 C job `1247291`；二者当前均为 `PENDING (Priority)`。A 未重复提交，当前无
  新性能结果。

- 2026-08-22：用户提供 FineGym 原始视频的完整 UTBox 共享链接及 2025-08-23 Google Drive
  备用源，并重申仅限学术使用、不得分发。远端已解析 UTBox `finegym_raw_videos` 全部 16 页：
  共 315 个视频 / 646,175,639,828 字节，文件名集合与 Drive 备份完全一致。由于远端 DNS 对
  `utexas.app.box.com` 的解析不可达，下载器通过学术代理连接 Box 官方边缘 IP、保留正确 Host
  获取逐文件签名链接；当前 16 个可续传 worker 正在运行，Box 解析错误和重试均为 0。状态为
  `download_running`；仅当 315 个文件逐一匹配 UTBox 官方 `itemSize` 并自动写入
  `DOWNLOAD_COMPLETE`、`DOWNLOAD_RECEIPT.txt` 和 `VALIDATION_REPORT.txt` 后才可称完整。

- 2026-08-22：用户要求把 Ego4D-NLQ 下载到远端。官方资料核实 2026 NLQ 挑战仍使用 Ego4D
  v2，精确范围为 `annotations + clips` 并用 `--benchmarks nlq` 过滤。远端已隔离安装官方
  Ego4D CLI v1.7.3，并在
  `/data/run01/sczc063/yuzibo/datasets/Ego4D-NLQ/download_ego4d_nlq.sh` 准备可续传启动器及
  NLQ annotation-to-clip UID 终态校验。预检以 exit 20 正确停止：本地和远端均不存在
  `~/.aws/credentials`。当前为 `download_blocked_by_credentials`，尚未下载任何 Ego4D
  payload；需用户完成官方许可并把仍有效的临时 AWS profile 安全配置到远端后才能启动。

- 2026-08-22 01:56（北京时间）：严格三臂归因矩阵的 B/C 两项训练已由调度等待转为真实运行。
  B（全部 token + sparse adapter，Slurm `1247290`）已完成 epoch 36、进入 epoch 37，最新恢复点为
  `epoch_35.pth`；C（ROI `K=64` + 同一 sparse adapter，Slurm `1247291`）已完成 epoch 33、进入
  epoch 34，最新恢复点为 `epoch_33.pth`。两项作业均从 2026-08-21 23:46 开始、各占两张 GPU，
  状态为 `RUNNING`；精确检索未发现 Traceback、CUDA OOM 或非有限损失。训练损失有限，但它不是
  detection validation 性能；必须等待预注册的 final/final-EMA 官方 validation 后，才比较 A→B
  的 adapter 影响和 B→C 的 ROI 增量影响。

- 2026-08-22 02:41（北京时间）：严格三臂归因矩阵形成首个同阶段过程性 validation。未修改官方
  A 在相同节点的 Avg-mAP/mAP@0.7 为 `67.88/46.19`；全部 token + sparse adapter 的 B 为
  `67.06/45.72`；ROI `K=64` + 同一 adapter 的 C 为 `67.86/46.14`。因此 A→B 为
  `-0.82/-0.47` 个百分点，B→C 为 `+0.80/+0.42` 个百分点；C 的 Avg-mAP 与 A 相差 `-0.02`。
  B/C jobs `1247290/1247291` 仍为 `RUNNING`，已推进到日志 epoch 43 附近并持续按两轮节奏保存
  checkpoint，未见 Traceback、OOM 或非有限损失。该观察只用于训练分布诊断，不改变预注册的
  final/final-EMA 模型选择，也不是最终性能或计算效率结论。

- 2026-08-22：完成旧 DN/G 与当前 A/B/C 的配置、包装器、选择器、adapter、损失图和训练协议
  回验。旧 DN 为 GeoRoute 同源全计算路径，终态 `64.73/43.26`；旧 G 在 VideoMAE 前执行
  全局 `B=24576`、动态 `K_t` 的原生 token 删除，终态 `61.49/39.99`。当前 B/C 均先完整执行
  稠密 VideoMAE，B 聚合全部 100 个空间 token，C 再固定选择 64 个 ROI token；首次同阶段
  验证为 `67.06/45.72` 与 `67.86/46.14`，B 后续中间验证为 `67.45/46.43`。因此旧 G 与
  当前 C 不具备实现同一性；当前结果只能支持后主干 ROI 聚合的精度归因，不能支持 VideoMAE
  重计算减少或真实成本下降。证据等级保持：A 为终态开发集性能，B/C 为中间运行性能。

- 2026-08-22：用户确认最终 ZoomToken 必须在 VideoMAE 重主干之前执行 ROI 原生 token 选择，
  重主干只能处理被选中的 token。当前后主干 B/C 归因矩阵继续仅作为诊断证据，不得表述为
  最终 ROI 方法或算力削减结果；旧 G 是主干前实现的负证据，但不等于否定所有主干前 ROI 设计。

- 2026-08-22：完成主干前固定 ROI 严格因果候选 `70dcbe1089866f6ee3821176eb41d2dc10ee8d14`。
  A 只读复用官方 job `1245842`；B 保留每 tubelet 全部 100 个原生 token；C 在 VideoMAE heavy
  前固定选择 ROI `K=64`；B/C 共用同一 true-ragged heavy path、sparse adapter、seed 42、双卡
  global/local batch `2/1` 和官方 60 轮配方。独立 Critic 修复并复核了 fixed-support regularization
  生命周期；目标环境又发现并修复 OpenTAD job-global batch 语义。双卡无指标 PRECHECK `1248828`
  `COMPLETED 0:0`，验证 B/C heavy token 为 `38,400/24,576`、单次 heavy、零 padding、输出形状相同。

- 2026-08-22：首次正式提交 `1248831/1248832` 在首批前因 common config 将 job-global batch 写成
  `1` 而失败，0 checkpoint、0 metric，不构成性能证据。修正 revision `70dcbe10…` 通过同一 Critic
  focused recheck 后，以新根提交 B job `1248835` 与 C job `1248834`；两项均已完成 canonical
  200-training/211-validation dataloader 构建、打印 epoch 0 started，并进入真实文件读取/首批处理。
  当前无 validation 或最终性能，下一证据为首个有效优化步、两轮一次的过程验证和 60 轮终态。

- 2026-08-22：主干前严格三臂的 B/C 正式训练终态完成。B job `1248835` 与 C job `1248834`
  均为 `COMPLETED 0:0`，各完成 60 轮、30 个两轮间隔 checkpoint 及 epoch-59 官方 validation；
  日志未见 Traceback、OOM 或非有限数值。A/B/C 的 Avg-mAP/mAP@0.6/mAP@0.7 分别为
  `68.73/61.58/47.24`、`68.51/61.19/46.27`、`68.22/61.01/45.35`。A→B 为
  `-0.22/-0.39/-0.97`，B→C 为 `-0.29/-0.18/-0.92`；这将旧 69→64→60 的大幅下降
  重新定位为不同 source/训练图/动态预算路径的混合效应，而不是 sparse adapter 或 fixed-ROI
  单独造成的必然损失。当前只具备单 seed 准确率证据，尚无端到端成本，不作效率或论文 claim。

- 2026-08-22：完成 `69 / 64 / 60` 四路径代码审计并记录于
  `docs/aris/ZOOMTOKEN_69_64_60_CODE_PATH_AUDIT-2026-08-22.md`。旧 DN/G 使用原生分辨率
  GeoRoute source、seed 3407、单卡 batch 1、不同 warmup/优化参数与 native-packed 路径；旧 G
  另含全窗动态 exact-B、可为零 `K_t` 和 auxiliary/proxy 损失，因此历史下降不是 adapter 与
  fixed-ROI 的纯两步效应。当前 `70dcbe10…` C 已逐层确认在原生 tubelet gather 后、VideoMAE
  patch embedding/blocks 前固定选择 ROI 64/100；不存在先运行稠密 VideoMAE 的隐藏分支。
  该审计不启动新训练或成本作业，下一证据仍是冻结 A/B/C checkpoint 的同硬件完整成本测量。

- 2026-08-22：用户确认新的主干前 ROI 实现必须以严格矩形 membership 为硬支持，而不能继续
  使用 ROI 分数全图 Top-K 并将其称为矩形。设计形成四条可区分机制臂：固定 8×8 严格矩形；
  矩形内 Token Select；连续宽高产生自然动态 `K_t` 的严格矩形；6×8 矩形核心加 16 个框外
  关键 free token。框外补充有独立上限，不能替换矩形核心或退回旧全局混排。完整规格位于
  `docs/superpowers/specs/2026-08-22-zoomtoken-strict-rectangle-roi-routing-design.md`；当前状态仅为
  `designed`，尚未实现、测试、训练或形成性能/成本证据。

- 2026-08-22：全新 exact-Project Pro 会话正常完成并返回 `REVISE`。它确认 `70dcbe10…` 的 C 是
  VideoMAE 前的原生 token 删除，但当前椭圆/高斯 Top-64 不是严格矩形。裁决把首个实验收窄为
  R1：`10×10` 网格九选一 `8×8` 完整矩形、固定 K64、单次 ragged 重主干；只比较现有 C 与
  R1 的 seed-42、60-epoch 高 tIoU 结果，通过后才测端到端成本。R2 框内选择、R3 动态矩形、
  R4 矩形加框外 token 均后置，不得与 R1 同时实现。此次讨论是设计裁决，不是代码、训练、
  性能或成本证据；原文保存在
  `docs/aris/ZOOMTOKEN_STRICT_RECTANGLE_ROI_PRO_RESULT-v002-2026-08-22.md`。

- 2026-08-22：严格矩形 R1 已从 `designed` 推进到 `experiment_running`。Builder 在 clean base
  `70dcbe10…` 上完成 R1，最终 clean/pushed revision 为 `9e25c6d38de8c993948025629181470b858682b4`；
  独立 Critic 最终 PASS。R1 在 `10×10` 原生网格上九选一完整 `8×8` 矩形，固定 K64，
  patch embedding 前 gather，单次 true-ragged heavy forward，零 padding，复用与 C 相同的
  sparse adapter、训练配方和官方 evaluator。目标环境 9 项无数据 Torch 检查通过。
  Evaluator 复核 411 个有效视频软链接、200/211 划分、20 类、预训练权重、历史 C endpoint 与空结果根后，
  启动唯一 R1 seed-42、60-epoch job `1249099`。作业已在 2×RTX4090 上进入 epoch 0；当前无
  R1 validation、成本或论文结论。下一证据是 5-epoch recovery 与预注册 validation/final-EMA。

- 2026-08-22：按用户批准的多分支计划完成 R2/R3/R4 及可区分对照的实现、独立审查、
  目标环境 PRE_RUN 与正式提交。clean/pushed revision 为
  `b1d9fa7b10209b23c4405b4be3965ee66f3c05f5`，parent 为 `aaf74a04…`；独立 Critic
  对 production selector、主干前边界、hard/soft support 一致性、公平性与无泄漏返回 PASS。
  目标环境的无数据检查为 `8 passed`，并核验 canonical THUMOS14 411 个有效视频软链接、
  200/211 划分、VideoMAE-S 权重、seed 42、双卡 global/local batch 2/1、官方 60 轮训练/
  evaluator/NMS、每 5 轮可恢复 checkpoint 与八个空结果根。
- 2026-08-22：八个单元已经 held-submit 后一次性释放。R2/R2-SHUF48/Q48-GLOBAL/R3/
  R3-AREA-SHIFT/R4 为 jobs `1249125–1249130`，均完成双 rank 初始化、200/211 dataloader
  构建并进入 epoch 0；R4-SHUF15/Q64-GLOBAL 为 `1249131/1249132`，已被 Slurm 接收，
  仅因账户 GPU 并发额度 `AssocGrpGRES` 等待自动启动。R2 是 8×8 eligibility 内 Top-48；
  R3 执行连续严格矩形的全部成员并允许自然动态 `K_t`；R4 是 7×7 无孔洞 core49 加
  框外 q_base Top15。两个 SHUF 对照在排序前对 q_base 与物理位置作稳定置换，软硬支持使用
  同一置换表面；Q48/Q64 是相同预算的全局内容选择。R1 `1249099` 未被修改并继续运行。
  这些都只是执行证据；当前没有新增 validation、成本、效率或论文结论。

- 2026-08-22 17:21 CST：只读监控确认 R1 job `1249099` 仍在 `g0024` 正常运行并进入
  epoch 21，已保存 `recovery_epoch_9/14/19.pth`，每 5 epoch 的 full-state 恢复合同开始形成
  真实运行证据。R2/R2-SHUF48/Q48-GLOBAL/R3/R3-AREA-SHIFT/R4 六项已进入 epoch 2；
  R4-SHUF15/Q64-GLOBAL 仍因 `AssocGrpGRES` 等待资源。所有已运行日志均未见 Traceback、
  OOM 或非有限数值，且尚无正式 validation；因此本次不产生准确率、成本或论文结论。

- 2026-08-22 17:29 CST：严格矩形矩阵的首个恢复点出现。R2-SHUF48 job `1249126`
  已发布 `recovery_epoch_4.pth`；R2/R2-SHUF48/Q48-GLOBAL/R3/R3-AREA-SHIFT/R4 六项
  均进入 epoch 4，R4-SHUF15/Q64-GLOBAL 仍因 `AssocGrpGRES` 等待。R1 继续进入 epoch 24，
  已有 epoch 9/14/19 恢复点。当前没有正式 validation、终态或硬故障；恢复点只证明恢复合同
  在真实训练中生效，不用于 checkpoint 选择或性能推断。
- 2026-08-22 17:38 CST：R1 job `1249099` 已进入 epoch 26，新生成 epoch 24 恢复点，按保留规则
  当前可见 epoch 14/19/24。R2/R2-SHUF48/Q48-GLOBAL/R3/R3-AREA-SHIFT/R4 六项均进入
  epoch 7，且全部生成首个 `recovery_epoch_4.pth`；R4-SHUF15/Q64-GLOBAL 仍因
  `AssocGrpGRES` 等待。未见正式 validation、终态或硬故障；本条仍只属于运行与恢复证据。
- 2026-08-22 17:48 CST：R2/R2-SHUF48/Q48-GLOBAL/R3/R3-AREA-SHIFT/R4 六项均进入
  epoch 10，并全部生成 `recovery_epoch_9.pth`，当前各自具备 epoch 4/9 两个恢复点；
  R4-SHUF15/Q64-GLOBAL 继续因 `AssocGrpGRES` 等待。R1 同期进入 epoch 29，保留
  epoch 14/19/24。当前没有正式 validation、终态或硬故障；新增信息只属于运行与恢复证据。
- 2026-08-22 21:18 CST：严格矩形路线产生首批正式中间验证。R1 已进入 epoch 57，最新
  Avg-mAP/mAP@0.6/mAP@0.7 为 `68.63/60.84/46.60`；相对 C 为
  `+0.41/-0.17/+1.25`，当前趋势满足 R1 预注册准确率门，但只能由 60 轮 final-EMA 最终裁决。
  R2/R2-SHUF48/Q48-GLOBAL/R3/R3-AREA-SHIFT/R4 的最新中间结果依次为
  `65.87/58.13/44.94`、`65.54/57.84/43.85`、`65.66/57.72/44.98`、
  `66.84/59.40/44.46`、`66.56/58.48/44.61`、`67.45/59.73/45.82`。
  R2 对乱序对照的 mAP@0.7 暂领先 `1.09`，但与全局 Top-48 几乎持平；R3 的 mAP@0.6
  优于面积轨迹错位 `0.92`，mAP@0.7 则低 `0.15`。六项均保留 epoch 39/44/49 恢复点；
  R4-SHUF15/Q64-GLOBAL 仍等待 GPU。没有硬故障，且不据中间结果选择 checkpoint。
- 2026-08-22 21:40 CST：R1 已进入 epoch 59 并发布 `epoch_59.pth`，但作业仍在运行，尚无
  final/final-EMA 终态。其最新中间 Avg-mAP/mAP@0.6/mAP@0.7 更新为
  `68.75/60.95/46.55`，相对 C 为 `+0.53/-0.06/+1.20`，继续满足预注册准确率符号。
  R2/R2-SHUF48/Q48-GLOBAL/R3/R3-AREA-SHIFT/R4 最新中间结果依次为
  `66.19/58.80/45.30`、`65.83/58.19/44.18`、`65.70/58.17/44.65`、
  `67.24/59.79/45.14`、`66.94/59.08/45.20`、`67.53/59.65/45.65`。
  R2 相对乱序与全局 Top-48 均出现正向差值；R3 相对面积轨迹错位仍未在 mAP@0.7 建立优势。
  两个 R4/Q64 对照继续等待 GPU；当前无硬故障，所有数字仍仅为过程证据，不用于 checkpoint 选择。

- 2026-08-22 21:50 CST：R1 job `1249099` 已完成 60 轮并以 `COMPLETED 0:0` 退出。epoch 59
  后的正式 EMA validation 为 tIoU 0.3/0.4/0.5/0.6/0.7 mAP
  `84.37/79.93/73.34/61.14/46.57`，Avg-mAP `69.07`；相对 C 为
  `+0.85/+0.13/+1.22`（Avg-mAP/mAP@0.6/mAP@0.7），三项预注册准确率条件全部通过。
  这是完整 8×8 矩形支持的 seed-42 准确率正证据，尚无配对端到端成本与多 seed 结论。
- 2026-08-22 21:50 CST：R4-SHUF15 job `1249131` 已从资源等待转为运行并进入 epoch 0，
  Q64-GLOBAL `1249132` 继续等待。R3/R3-AREA-SHIFT/R4 最新中间三指标分别为
  `67.40/59.55/45.62`、`67.03/59.57/44.94`、`67.88/60.05/46.35`；R3 相对错位
  对照为 `+0.37/-0.02/+0.68`。未见硬故障；本轮没有提交、取消、恢复或新增作业。

- 2026-08-22 22:00 CST：R2/R2-SHUF48/Q48-GLOBAL 已进入 epoch 54–55并发布
  `recovery_epoch_54.pth`。最新中间 Avg-mAP/mAP@0.6/mAP@0.7 分别为
  `66.28/58.76/44.75`、`65.90/58.24/43.80`、`65.77/58.51/44.74`；R2 相对乱序
  对照为 `+0.38/+0.52/+0.95`，相对全局 Top-48 为 `+0.51/+0.25/+0.01`。因此框内
  内容排序信号仍为正，但矩形 eligibility 在高 tIoU 上尚未与全局选择拉开。R4-SHUF15 进入
  epoch 3，Q64-GLOBAL 继续等待；未见硬故障，也未执行任何作业变更。

- 2026-08-22 22:10 CST：R3、R3-AREA-SHIFT 与 R4 出现新的官方中间 validation，
  Avg-mAP/mAP@0.6/mAP@0.7 分别为 `67.64/59.89/45.95`、`67.27/59.67/45.14`、
  `68.01/60.45/46.23`。R3 相对时间错位面积轨迹为 `+0.36/+0.22/+0.81`，当前三个
  指标均为正，但仍须由 60 轮终态裁决。R3 与 R4 已发布 `recovery_epoch_54.pth`；
  R4-SHUF15 已进入 epoch 6 并发布首个 `recovery_epoch_4.pth`。Q64-GLOBAL 继续因
  `AssocGrpGRES` 等待。未见 Traceback、显存溢出或非有限数值，也未执行作业变更。

- 2026-08-22 22:21 CST：R2/R2-SHUF48/Q48-GLOBAL 新一轮官方中间 validation 的
  Avg-mAP/mAP@0.6/mAP@0.7 为 `66.44/59.08/44.93`、`66.03/58.37/44.53`、
  `65.64/58.48/44.41`。R2 相对乱序对照为 `+0.41/+0.71/+0.40`，相对全局 Top-48
  为 `+0.80/+0.60/+0.52`；当前节点重新同时出现框内排序和矩形 eligibility 的正向信号，
  但仍不替代 60 轮终态。R3-AREA-SHIFT 已补齐 `recovery_epoch_54.pth`，至此六个先行
  单元均具备 epoch 54 恢复点；R4-SHUF15 已进入 epoch 8，Q64-GLOBAL 继续等待。未见硬故障。

- 2026-08-22 22:31 CST：R3、R3-AREA-SHIFT 与 R4 的新一轮官方中间 validation 更新为
  `67.89/60.17/46.31`、`67.42/59.95/45.00`、`68.03/60.41/46.11`
  （Avg-mAP/mAP@0.6/mAP@0.7）。R3 相对时间错位面积轨迹为 `+0.46/+0.22/+1.31`，
  延续当前节点的正向对齐信号，但仍须等待 60 轮 final/final-EMA。R4-SHUF15 已进入 epoch 11
  并发布 `recovery_epoch_9.pth`；Q64-GLOBAL 仍为 `PENDING (AssocGrpGRES)`。未见 Traceback、
  显存溢出或非有限数值；没有提交、取消、重排、恢复或新增任何作业。

- 2026-08-22 22:42 CST：R2-SHUF48 与 Q48-GLOBAL 的新一轮官方中间 validation 为
  `66.03/58.26/44.39` 与 `65.88/58.75/44.73`（Avg-mAP/mAP@0.6/mAP@0.7），两项均已进入
  epoch 59；R2-SHUF48 已写出 `epoch_59.pth`。R2 的最新可见值仍为
  `66.44/59.08/44.93`，因此其相对两项对照的异步差值 `+0.41/+0.82/+0.54` 与
  `+0.56/+0.33/+0.20` 只作过程记录，不作同阶段机制裁决。R4-SHUF15 已进入 epoch 14并保有
  epoch 4/9 恢复点，Q64-GLOBAL 仍为 `PENDING (AssocGrpGRES)`。未见 Traceback、显存溢出或
  非有限数值；没有终态 final/final-EMA，也没有提交、取消、重排、恢复或新增作业。

- 2026-08-22 22:53 CST：Q64-GLOBAL job `1249132` 已从 `AssocGrpGRES` 转为 `RUNNING`，
  完成双 rank 初始化并进入 epoch 1；没有人为重排或重复提交。R2、R2-SHUF48、Q48-GLOBAL、
  R3 与 R4 已写出 `epoch_59.pth`，R3-AREA-SHIFT 已进入 epoch 58；这些只是训练端点证据，
  尚非 final/final-EMA。最新中间结果为 R2/R2-SHUF48/Q48-GLOBAL
  `66.44/58.81/45.10`、`66.03/58.26/44.39`、`65.88/58.75/44.73`，以及
  R3/R3-AREA-SHIFT/R4 `67.83/60.00/46.56`、`67.27/59.76/44.60`、
  `68.02/60.52/46.20`。R4-SHUF15 已进入 epoch 17并发布 `recovery_epoch_14.pth`。
  未见 Traceback、显存溢出或非有限数值；不据中间结果选模。

- 2026-08-22 23:04 CST：只读调度核验确认 R2/R2-SHUF48/Q48-GLOBAL/R4 jobs
  `1249125/1249126/1249127/1249130` 已 `COMPLETED 0:0`；R3/R3-AREA-SHIFT 进入
  epoch 59，R4-SHUF15 进入 epoch 20，Q64-GLOBAL 进入 epoch 4。R4 日志在
  `Training Over` 前最后一次官方 validation 为 Avg-mAP/mAP@0.6/mAP@0.7
  `68.02/60.32/46.26`。本次单次状态检查没有取得三个先完成单元的终态日志路径，也未取得
  四个完成单元的结构化 final-EMA 收据，故不把既有中间值或 R4 日志末值提前标为 final-EMA。
  可见日志未见 Traceback、显存溢出或非有限数值；没有提交、取消、重排、恢复或新增作业。

- 2026-08-22 23:13 CST：同一只读核验取得六个完成单元在 `Training Over` 前的终态官方
  validation。R2/R2-SHUF48/Q48-GLOBAL 的 Avg-mAP/mAP@0.6/mAP@0.7 为
  `66.56/59.06/45.17`、`66.17/58.53/44.47`、`65.78/58.62/44.74`；R2 相对两项
  对照分别为 `+0.39/+0.53/+0.70` 与 `+0.78/+0.44/+0.43`。R3/R3-AREA-SHIFT
  为 `67.88/60.32/46.41` 与 `67.50/60.26/45.09`，时间对齐增量为
  `+0.38/+0.06/+1.32`。R4 为 `68.02/60.32/46.26`，但 R4-SHUF15 与
  Q64-GLOBAL 尚未终态，故不提前解释框外 free-token。R3 job `1249128` 与
  R3-AREA-SHIFT job `1249129` 已 `COMPLETED 0:0`；R4-SHUF15 运行至 epoch 23并保有
  epoch 9/14/19 恢复点，Q64-GLOBAL 运行至 epoch 6并发布首个
  `recovery_epoch_4.pth`。未见硬故障；没有作业变更、成本测量或新增实验。

- 2026-08-22 23:24 CST：只读核验确认 R4-SHUF15 job `1249131` 继续正常运行至
  epoch 25，并新发布 `recovery_epoch_24.pth`；按最近三份保留规则，当前恢复点为 epoch
  14/19/24。Q64-GLOBAL job `1249132` 运行至 epoch 9，仍保有首个 epoch-4 恢复点。
  两项均无新的正式 validation，日志未见 Traceback、显存溢出或非有限数值。本条仅是
  运行/恢复证据，不改变 R4 机制归因冻结，也没有作业变更、成本测量或新增实验。

- 2026-08-22 23:35 CST：Q64-GLOBAL job `1249132` 已进入 epoch 12并新发布
  `recovery_epoch_9.pth`，当前具备 epoch 4/9 两个可恢复点；R4-SHUF15 job `1249131`
  已进入 epoch 28，继续保有 epoch 14/19/24 三个最近恢复点。两项仍为 `RUNNING`，均无
  新的正式 validation，日志未见 Traceback、显存溢出或非有限数值。本条只确认训练连续性，
  不用于选择 checkpoint、裁决框外 free-token、启动成本或增加实验。

- 2026-08-22 23:45 CST：R4-SHUF15 job `1249131` 已进入 epoch 31并新发布
  `recovery_epoch_29.pth`，当前最近三个恢复点为 epoch 19/24/29；Q64-GLOBAL job
  `1249132` 已进入 epoch 15并新发布 `recovery_epoch_14.pth`，当前最近三个恢复点为
  epoch 4/9/14。两项仍为 `RUNNING`，均无新的正式 validation，日志未见 Traceback、
  显存溢出或非有限数值。本条只确认训练连续性，不用于选择 checkpoint、裁决框外 free-token、
  启动成本或增加实验。

- 2026-08-23 00:06 CST：只读调度与恢复点核验确认 R4-SHUF15 job `1249131` 和
  Q64-GLOBAL job `1249132` 仍为 `RUNNING`。R4-SHUF15 新发布
  `recovery_epoch_34.pth`，最近三个恢复点更新为 epoch 24/29/34；Q64-GLOBAL 新发布
  `recovery_epoch_19.pth`，最近三个恢复点更新为 epoch 9/14/19。本轮未取得新的正式
  validation、终态或硬错误回执，也未提交、取消、重排、恢复或新增作业。新增恢复点仅证明
  训练连续性，不用于选择 checkpoint、裁决框外 free-token、启动成本或增加实验。

- 2026-08-23 00:16 CST：R4-SHUF15 job `1249131` 仍为 `RUNNING`，已进入 epoch 40并
  新发布 `recovery_epoch_39.pth`，最近三个恢复点更新为 epoch 29/34/39。Q64-GLOBAL job
  `1249132` 仍为 `RUNNING`，已进入 epoch 23，最近三个恢复点保持 epoch 9/14/19。
  两项原始 stdout/stderr 未见 Traceback、显存溢出或非有限数值，也没有新的正式 validation
  或终态。本轮未提交、取消、重排、恢复或新增作业；恢复点不用于模型选择或机制裁决。

- 2026-08-23 00:26 CST：Q64-GLOBAL job `1249132` 仍为 `RUNNING`，新发布
  `recovery_epoch_24.pth`，最近三个恢复点更新为 epoch 14/19/24；R4-SHUF15 job
  `1249131` 同样保持 `RUNNING`，最近三个恢复点仍为 epoch 29/34/39。本轮未取得新的正式
  validation 或终态；新增恢复点只证明 Q64 全局对照持续完成训练更新，不用于选择 checkpoint、
  提前解释 R4 的框外 free-token，或启动成本与补充实验。

- 2026-08-23 00:46 CST：R4-SHUF15 job `1249131` 仍为 `RUNNING`，已进入 epoch 43；
  00:32:13 的官方中间 validation 给出 Avg-mAP `65.87`。本次单次只读快照未捕获该轮
  mAP@0.6/mAP@0.7，因此不计算与 R4 终态的完整三指标差值，也不提前解释框外 free-token。
  Q64-GLOBAL job `1249132` 已进入 epoch 31并发布 `recovery_epoch_29.pth`，最近三个恢复点
  更新为 epoch 19/24/29，尚无正式 validation。两项日志未见 Traceback、显存溢出或非有限数值；
  没有作业变更、成本测量或新增实验。

- 2026-08-23 00:56 CST：R4-SHUF15 job `1249131` 已进入 epoch 44，最新官方中间
  validation 为 Avg-mAP/mAP@0.6/mAP@0.7 `66.27/59.02/44.59`；相对 R4 终态
  `68.02/60.32/46.26` 暂低 `1.75/1.30/1.67` 个百分点。由于训练阶段不同且 SHUF15
  尚未完成 60 轮，该差值不用于机制裁决或 checkpoint 选择。Q64-GLOBAL job `1249132`
  已进入 epoch 34，最近三个恢复点仍为 epoch 19/24/29，尚无正式 validation。两项日志未见
  Traceback、显存溢出或非有限数值；没有作业变更、成本测量或新增实验。

- 2026-08-23 01:06 CST：R4-SHUF15 job `1249131` 与 Q64-GLOBAL job `1249132`
  仍为 `RUNNING`，分别新增 `recovery_epoch_44.pth` 与 `recovery_epoch_34.pth`；最近三个恢复点
  更新为 epoch 34/39/44 与 24/29/34。R4-SHUF15 最新中间 validation 仍为
  Avg-mAP/mAP@0.6/mAP@0.7 `66.27/59.02/44.59`，Q64-GLOBAL 尚无正式 validation。
  两项日志未见 Traceback、显存溢出或非有限数值；本次只读核验未提交、取消、重排、恢复或新增作业。
  新恢复点仅是训练连续性证据，不用于 checkpoint 选择或 R4 机制裁决。

- 2026-08-23 01:38 CST：本地 `E:\\Released_FineDiving_Dataset` 到远端
  `/data/run01/sczc063/yuzibo/datasets/FineDiving` 的授权上传完成。补传覆盖
  `Lowresolution01_Trimmed_Video_Frames` 与完整
  `Lowresolution005_Trimmed_Video_Frames`；目录级核验分别达到
  `312,256 / 354,562,258` 与 `312,256 / 249,019,138`（文件/字节）。最终对本地和远端
  全量生成排序后的“相对路径 + 文件大小”清单，双方均为 1,337,505 文件 / 101,123,416,820
  字节，逐条 `cmp` 完全一致。远端完成凭据写于数据集外部
  `/data/run01/sczc063/yuzibo/datasets/.transfer_state/FineDiving/COMPLETE`；状态由
  `transfer_running` 更新为 `transfer_complete_verified`。

- 2026-08-23 01:43 CST：完成同一样本的 ZoomToken token-selection 主图。clean 可视化代码
  `0b12c68e8559048a0bab617af2f420d2a2080f3d` 的 Slurm job `1250245` 在 `00:01:29`
  内 `COMPLETED 0:0`；固定样本为 THUMOS14 validation `video_test_0001339` 的首个滑窗，展示
  tubelet `32/96/160/224/288/352`。每行使用该方法自己的 checkpoint，并直接调用 production
  selector 后在 VideoMAE heavy forward 前停止；原色网格为选中 native token，灰色网格为未选中
  token，完整物理索引保存在 JSON。九个终态方法使用各自 `epoch_59.pth`；尚未终态的
  R4-SHUF15/Q64-GLOBAL 使用 `recovery_epoch_44.pth/recovery_epoch_39.pth`，在图中明确标为
  仅作定性观察。R3 与 R3-AREA-SHIFT 的全窗口 `K_t` 均跨 `56–70`，图中六列恰好均为 56。
  该产物验证了矩形/乱序/全局支持的空间形态，不提供准确率或成本结论，也不把不同 checkpoint
  的差异升级为反事实因果证据。终态回执为
  `docs/aris/ZOOMTOKEN_TOKEN_SELECTION_VISUALIZATION_RECEIPT-2026-08-23.md`。

- 2026-08-23 02:00 CST：在不改模型、不触碰训练作业的前提下，将 on-policy token-selection 可视化扩展到三个结果盲选的 THUMOS14 validation 样本：短动作板球 `video_test_0001194`、长持续举重 `video_test_0000058` 和多类田径 `video_test_0000367`。clean source `0b12c68e` 的只读路由 job `1250422` 在 `00:02:16` 内 `COMPLETED 0:0`，每个样本均生成 PNG/PDF/完整 physical-index JSON。九个终态方法继续使用各自 epoch-59 checkpoint；R4-SHUF15/Q64-GLOBAL 固定使用 epoch-49/39 recovery，仅作定性观察。完整 384-tubelet 支持统计显示 R1 在三个窗口内均为 1 种 mask/0 次转移，R3 系列仅 1–4 种 mask，而 R2、GLOBAL、R4 内容排序路线变化更频繁。该结果只说明若干几何路线在这些样本上的实际时间动态有限，不升级为准确率、成本或因果证据；终态回执追加在 `docs/aris/ZOOMTOKEN_TOKEN_SELECTION_VISUALIZATION_RECEIPT-2026-08-23.md`。

- 2026-08-23 02:12 CST：对既有严格矩形矩阵进行一次只读核验。R4-SHUF15 job
  `1249131` 与 Q64-GLOBAL job `1249132` 均仍为 `RUNNING`；前者新增
  `recovery_epoch_49.pth`（627,957,176 bytes），最近三个恢复点为 epoch 39/44/49；后者
  新增 `recovery_epoch_44.pth`（627,896,364 bytes），最近三个恢复点为 epoch 34/39/44。
  本次未取得新的正式 validation、终态或硬错误回执，也没有提交、取消、重排、恢复、重启或
  创建作业。新增 checkpoint 仅是训练连续性证据，不用于模型选择、R4 机制裁决或成本结论。

- 2026-08-23 03:07 CST：ZoomToken Project 的 fresh `gpt-5.5-pro` 对话
  `6a89ec16-90fc-83ea-a048-311b929ab876` 在两个附件与 GitHub 精确提交 `0b12c68e…` 上完成
  正式裁决。附件以 browser upload 发送，未内联或打包。裁决为 `REVISE` 后继续 `RC32-KV`：
  保留 R1 K64 严格矩形支持，只让同一 K32 mask 在全部 12 blocks 执行 query/output attention
  与 MLP；另外 K32 保留 K/V，并在 RC 臂使用前一 tubelet 同位置、窗口内、detached carry 与
  每 block 一个标量混合。最小四臂为 FULL64/DROP32/MOD32-KV/RC32-KV；不先实现 K24/K18、
  learned gate 或 ChronoTransport 重构。该记录属于 `designed`，下一步是 Builder 最小实现、
  独立 Critic 与结果盲 PRE_RUN，不构成性能或成本结果，也不改变既有 R4-SHUF15/Q64 作业。

- 2026-08-23 03:55 CST：完成 `RC32-KV` 冻结设计的最小实现、独立审查、结果盲 PRE_RUN 与
  真实训练提交。clean revision 为 `836f2ce4beafa8cbab513604dfa74be01a977a3c`，已推送至
  GitHub 分支 `codex/zoomtoken-rc32-kv-v001`；变更仅覆盖三项新 config、R1 base config、
  GeoRoute routing/wrapper、VideoMAE adapter、既有 N16R4 launcher 和 focused test。目标 N16R4
  环境的 8 项 Torch 测试通过；独立 Critic 在对照 Pro 原文后确认 K64 lineage、K32 refresh、
  同位置窗口内 carry 及无泄漏边界均符合设计并给出 PASS；同一 Evaluator 归一化为
  `PRE_RUN_READY/PASS`。FULL64 只读复用已完成 R1 job `1249099`，不重复训练。DROP32、
  MOD32-KV、RC32-KV 以 seed 42、双卡、60 轮、官方 THUMOS14 配方分别提交为 jobs
  `1250604/1250605/1250606`，共同根目录为
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_refresh_rc32_836f2ce4_seed42_20260823T0355`；
  首次状态为 RUNNING/RUNNING/RUNNING。当前只有实现与启动证据，尚无新增 validation、成本或
  论文结论；不提交 K24/K18、多 seed、重复 R1 或额外成本作业。

- 2026-08-23 04:03 CST：只读 Slurm/日志核验发现 RC32-KV 首个部署三项均已在训练前终止：
  DROP32 `1250604` 与 MOD32-KV `1250605` 各运行 34 秒，RC32-KV `1250606` 运行 30 秒，
  均为 `FAILED 1:0`。三个 traceback 的根因一致：`tools/train.py::_zoomtoken_recovery_contract`
  仍把 recovery 限定为旧冻结 route surface，未接受新 config schema，并在模型、视频、checkpoint
  和首个训练 batch 前抛出 `ValueError: ZoomToken recovery is restricted to the frozen route
  surfaces`。共同 namespace 无 recovery、validation、成本或模型结果；这是确定性的训练入口/准入
  实现缺陷，不是 DROP32、MOD32-KV、RC32-KV 或 carry 的科学结果。未提交、取消、恢复、重排或
  创建任何作业；既有 Q64-GLOBAL `1249132` 继续 RUNNING，未见新恢复点或正式 validation。

- 2026-08-23 04:14 CST：单次只读核验确认 Q64-GLOBAL job `1249132` 继续 `RUNNING`，累计运行
  `05:25:05`；新增 `recovery_epoch_49.pth`（627,896,364 bytes）与
  `recovery_epoch_54.pth`（627,896,428 bytes），最近三个恢复点为 epoch 44/49/54。未发现新的
  正式 validation、终态文件、Traceback、显存溢出或非有限数值。RC32-KV jobs
  `1250604/1250605/1250606` 的训练前失败终态不变。本轮没有提交、取消、恢复、重排或创建作业；
  新恢复点只属于运行连续性证据，不构成性能或成本结论，也不用于 checkpoint 选择。

- 2026-08-23 05:03 CST：单次只读 Slurm 终态核验确认 Q64-GLOBAL job `1249132` 已于
  04:56:25 以 `COMPLETED 0:0` 结束，累计运行 `06:06:57`；最近三个完整恢复点仍为
  epoch 44/49/54。本次检查未捕获正式 validation 行，也未发现结构化 finalization/result 文件，
  因此这里只把证据升级为“60 轮训练完成”，不从退出码或 checkpoint 推断准确率、框外
  free-token 机制或成本。RC32-KV jobs `1250604/1250605/1250606` 的既有训练前失败终态不变；
  本轮没有提交、取消、恢复、重排、重启或创建作业。

- 2026-08-23 05:34 CST：对同一既有 Q64-GLOBAL job `1249132` 的一次只读日志摄取取得
  `Training Over` 前最后一组冻结 EMA 官方 validation：tIoU 0.3/0.4/0.5/0.6/0.7 分别为
  `82.91/78.61/71.64/60.66/45.39`，Avg-mAP 为 `67.84`。作业终态仍为
  `COMPLETED 0:0`，没有结构化 finalization/result 文件。相对 R4 终态
  `68.02/60.32/46.26`（Avg-mAP/mAP@0.6/mAP@0.7），R4−Q64-GLOBAL 为
  `+0.18/-0.34/+0.87` 个百分点：这是单种子准确率证据，提示连续矩形 core 对高 tIoU
  可能有益，但不能替代 R4-SHUF15 冻结终态、多种子或完整成本。RC32-KV 三项训练入口失败
  终态不变；本轮没有提交、取消、恢复、重排、重启或创建作业。

- 2026-08-23：完成 RC32-KV 首次部署所暴露 recovery route-schema 缺口的最小修复。clean
  candidate `813012620dca991ff90121d0d9faf688f303d1ef` 已推送至
  `codex/zoomtoken-rc32-kv-v001`；相对原实现 `836f2ce4…` 只修改 `tools/train.py` 与
  `tests/test_zoomtoken_r1_refresh_carry_k32.py`。训练入口现在允许既定的 `R1-DROP32`、
  `R1-MOD32-KV`、`R1-RC32-KV` 使用原有完整 recovery contract，并继续拒绝未知 route；
  5-epoch/latest-three、model/EMA/optimizer/scheduler/scaler、update index、sampler 与 RNG
  恢复语义均未改变。目标 N16R4 clean source
  `/data/run01/sczc063/yuzibo/runtime/zoomtoken_rc32_81301262/source` 的实际三配置入口测试为
  `10 passed`，py_compile、diff 和 clean-source 检查通过；独立只读审查无条件 PASS。Windows
  本地 pytest 因 PyTorch `c10.dll` 初始化失败无法收集，但不是测试断言失败，目标 Linux 环境已
  覆盖该入口。没有提交 Slurm 作业、访问训练数据或运行模型；旧 jobs
  `1250604/1250605/1250606` 仍为不可恢复的训练前无效 epoch，因此本条只把修复升级为
  `implemented_and_target_tested`，不产生准确率、成本或论文结论。

- 2026-08-23 21:00 CST：clean revision
  `813012620dca991ff90121d0d9faf688f303d1ef` 完成结果盲 PRE_RUN 并提交新的独立
  RC32-KV seed-42 完整训练 epoch。PRE_RUN 复核了精确 HEAD/ref/clean、三份真实 config 的
  recovery 入口（目标环境 `10 passed`）、411 个规范 THUMOS14 视频软链接且 0 断链、annotation、
  class map、VideoMAE-S pretrain、1.7 TB 可用空间、空结果根与官方 evaluator/Soft-NMS/EMA/
  checkpoint 合同；三个 `sbatch --test-only` 均通过。FULL64 继续只读复用 R1 job `1249099`。
  DROP32/MOD32-KV/RC32-KV 分别提交为 jobs `1252179/1252180/1252181`，共同结果根为
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_refresh_rc32_81301262_seed42_20260823T2100`；
  首次状态均为 `PENDING (Priority)`，每项申请 1 node、2 GPU、8 CPU、8 hours，不显式覆盖站点
  memory，也不传 resume。当前只有 PRE_RUN/调度接收证据，没有新增准确率、成本或论文结论；
  不提交 K24/K18、多 seed、重复 FULL64 或成本作业。

- 2026-08-23 21:01 CST：只读 Slurm/日志核验确认三个修复后单元均已运行：DROP32
  `1252179` 位于 `g0063`，MOD32-KV `1252180` 与 RC32-KV `1252181` 位于 `g0066`
  的独立 Slurm 分配。三项日志均越过旧 recovery-contract 失败点，出现
  `Training Starts...` 与 `[Train]: Epoch 0 started`；未发现 Traceback、OOM 或非有限数值。
  这只是进入真实训练的运行证据，尚无 recovery、正式 validation、准确率或成本结果。

- 2026-08-24 01:29 CST：只读核验显示 DROP32/MOD32-KV/RC32-KV jobs
  `1252179/1252180/1252181` 均为 `RUNNING`，elapsed 约 `04:30:12`，已进入第 51 轮；
  三臂均保留 recovery epochs `39/44/49`。第 49 轮后的官方 validation
  Avg-mAP/mAP@0.6/mAP@0.7 分别为 `65.48/57.13/43.50`、
  `65.50/57.65/43.90`、`64.20/56.21/42.18`。未见 Traceback、OOM 或非有限训练故障，
  也尚无 final/final-EMA 或 selector-inclusive cost。RC32-KV 当前相对 MOD32-KV 为
  `-1.30/-1.44/-1.72` 个百分点，仅登记为负向过程信号；不据此提前选模或改变冻结路线。

- 2026-08-24 01:40 CST：三项修复后训练均继续为 `RUNNING` 并进入 epoch 53，未见
  Traceback、OOM 或非有限数值。第 51 轮后的官方中间 Avg-mAP/mAP@0.6/mAP@0.7 更新为
  DROP32 `65.79/57.89/43.63`、MOD32-KV `65.89/58.64/43.99`、RC32-KV
  `64.35/56.62/42.11`；RC32-KV 相对 MOD32-KV 仍为负向过程信号，不用于选模或提前停止。
  为防止平均 mAP 掩盖边界损失，在 clean RC32 worktree 新增只读 source-video 边界评估器，按
  冻结的 score-ordered greedy one-to-one、短动作、归一化起止误差和 high-IoU bins 计算，7 项
  focused 测试通过；它不进入训练路径，尚未消费终态 prediction。另生成可复现的中间
  计算量—准确率图：按 VideoMAE block 的 Q/K/V/output、attention、MLP 矩阵乘，DROP32、
  MOD32-KV、RC32-KV 约为 FULL64 的 `49.32%/58.11%/58.11%`。该横轴明确排除解码、H2D、
  scout、patch embedding、adapter、检测头、后处理与 NMS，当前没有 exact-checkpoint 端到端
  latency/energy，故不作效率结论。

- 2026-08-24 01:48 CST：为终态边界摄取完成 clean descendant
  `cb38b555fd0b564711a5941fb86aed703f3ad1f4` 并推送分支
  `codex/zoomtoken-rc32-boundary-eval-v001`。新增 final-evaluation launcher 只使用各 arm 的
  `checkpoint/epoch_59.pth`，绑定 canonical THUMOS14 validation 并保存
  `result_detection.json`；不接受 recovery checkpoint。独立只读审查发现并修复 `tools/test.py`
  原先将无 `module.` 前缀的 EMA state 直接加载到 DDP wrapper 的确定性错误，现改为加载
  `model.module`；非 EMA 路径不变。py_compile、bash syntax、diff check 与 8 项 focused tests
  全部通过。该 descendant 不改变正在运行的 `81301262…` 训练或 checkpoint，也未提交新的
  Slurm 作业；三项训练仍为 `RUNNING`。

- 2026-08-24 01:53 CST：三项训练均进入 epoch 54，未见 Traceback、OOM 或非有限数值。
  第 53 轮后的官方中间 Avg-mAP/mAP@0.6/mAP@0.7 为 DROP32
  `65.81/57.66/43.96`、MOD32-KV `66.23/59.19/44.46`、RC32-KV
  `64.59/56.93/42.85`。MOD32-KV 相对 DROP32 为 `+0.42/+1.53/+0.50`；RC32-KV
  相对 MOD32-KV 为 `-1.64/-2.26/-1.61`。已把可复现计算量—准确率图更新到该节点；
  仍不据中间 checkpoint 选模，继续等待 epoch-59 EMA 与终态 prediction。

- 2026-08-24 01:58 CST：独立只读核验定位已完成 FULL64/R1 的 final checkpoint 为
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_official_prebackbone_r1_9e25c6d3_seed42_20260822T080108Z/cells/r1_strict_rect8x8_prebackbone_sparse_adapter/seed42/gpu2_id0/checkpoint/epoch_59.pth`，
  且确认尚无 `result_detection.json`。终态诊断 descendant 更新为 clean
  `4e940b780da5a3cd0ea28ca420c5d1cb879818b5`，只为 final-evaluation launcher 增加 FULL64
  映射和回归断言。`bash -n`、8 项 focused tests、diff check 通过并已推送；未提交任何作业。

- 2026-08-24 02:55 CST：RC32 修复版完整矩阵终态。jobs
  `1252179/1252180/1252181` 均 `COMPLETED 0:0`，每个 cell 的
  `checkpoint/epoch_59.pth` 存在，日志出现 `Training Over...` 且没有 Traceback、OOM 或非有限数值。
  final-EMA Avg-mAP/mAP@0.6/mAP@0.7 为 DROP32 `66.11/57.83/44.88`、MOD32-KV
  `66.50/59.24/45.21`、RC32-KV `64.73/57.34/42.91`，只读 FULL64 为
  `69.07/61.14/46.57`。RC32-KV 对 A/B/C 的准确率门全部失败；MOD32-KV 相对 FULL64 也
  超过允许损失。按 Pro 预注册分支停止当前 temporal carry 与 KV-preserving K32 depth-allocation
  路线，不启动成本、边界补充、多 seed、K24/K18、learned gate 或蒸馏。最终可复现
  计算量—准确率图已生成；其 `49.32%/58.11%/58.11%` 仅为重块 matmul FLOPs 代理，
  不是端到端延迟或能耗。

- 2026-08-24 03:30 CST：fresh ZoomToken Project Pro 以完成的 seed-42 终态为唯一新证据，裁决
  `REVISE`，选择唯一候选 `DSR6-KV`。机制冻结为：严格 8×8/K64 支持不变；VideoMAE
  blocks 0–5 完整 K64 更新，blocks 6–11 对同一个每-tubelet K32 mask 执行
  query/output/MLP，全部 K64 保持不 detach 的 K/V context 并继续经过既有 Adapter。禁止 hidden
  carry、浅层 transport、新参数/损失、逐层动态预算、第二切分点、K24/K18 和新 seed。机械重块
  代理约 `79.055%`，不等于端到端速度。唯一新实验为 seed42、双卡、60 epoch 的 DSR6-KV；
  final-EMA 必须同时达到 Avg-mAP `68.57`、mAP@0.6 `60.64`、mAP@0.7 `46.07`，任一失败即
  `STOP_DEPTH_ROUTE` 且不测成本。当前状态为 `designed`，尚未实现或运行；next owner 为 Builder，
  必须先返回最小修改计划和 clean candidate，再经独立 Critic 与结果盲 Evaluator/PRE_RUN。

- 2026-08-24 04:20 CST：`CENTRAL-RUN-ZOOMTOKEN-DSR6-KV-SEED42-v001` 单次调度在
  `sbatch` 前 fail closed。只读门通过 exact GitHub SHA/ref、canonical THUMOS14 `411/0`、
  annotation/class map/VideoMAE-S/OpenTAD 环境文件、两个新根无碰撞、同名 job 不存在与
  `/data` 可用 `1651527008 KiB`。通过学术代理建立 immutable clean source
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_dsr6_src_3260cd39`，HEAD 和
  `refs/remotes/origin/codex/zoomtoken-dsr6-kv-v001` 均为
  `3260cd39154069138c6b1757326372cc3b73754e`。目标环境 focused precheck 随后因非登录 shell
  未 source `/etc/profile`、`module` 不可见而退出；没有执行 `sbatch --test-only` 或正式
  `sbatch`。复核确认结果根不存在、Slurm/sacct 无同名 job，`actual_attempt_count=0`。这是
  预提交环境初始化顺序问题，不是模型、训练、性能或成本证据；下一步须由新的单次调度按文档
  顺序初始化环境后重跑同一 precheck，科学合同不变。

- 2026-08-24 04:24 CST：机械重调度
  `CENTRAL-RUN-ZOOMTOKEN-DSR6-KV-SEED42-v002` 仍在 `sbatch` 前 fail closed。唯一允许的
  `source /etc/profile` 在调用方 `set -u` 下进入 `/etc/profile.d/apps-bin-path.sh` 时因未定义
  `XDG_DATA_DIRS` 退出，focused target-environment test 尚未执行。停止后再次确认 immutable
  source HEAD/ref exact 且 clean、requested result root absent、squeue/sacct exact job absent；
  `actual_attempt_count=0`。没有训练、GPU、checkpoint、validation、性能或成本证据。下一恢复需
  新授权仅允许在 source profile 期间局部 `set +u`，完成后恢复 `set -u` 再运行完全相同的检查。

- 2026-08-24 04:23:04 CST：最终机械调度
  `CENTRAL-RUN-ZOOMTOKEN-DSR6-KV-SEED42-v003` 通过。唯一 shell 修正为 source
  `/etc/profile` 期间 `set +u`、完成后恢复 `set -u`。N16R4 focused suite
  `11 passed in 41.46s`，launcher syntax 与 `sbatch --test-only` PASS；source HEAD/ref/clean 仍为
  `3260cd39154069138c6b1757326372cc3b73754e`。唯一正式训练 job `1252521` 已提交，初始
  `PENDING`，1 node/2 GPU/8 CPU/8 h，result root
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_dsr6_3260cd39_seed42_20260824`。该单元固定
  R1-DSR6-KV、seed42、60 epochs、global/local batch 2/1、canonical THUMOS14
  training→validation、5-epoch full-state recovery/latest3+final 与 epoch59 EMA。当前仅为启动证据；
  尚无准确率或成本结果，不允许重复、补臂、额外 seed 或成本。

- 2026-08-24 04:25 CST：job `1252521` 终态 `FAILED 1:0`，elapsed `00:00:02`。
  stdout 为空；stderr 为 `/etc/profile.d/Z97-byobu.sh: line 24: LC_BYOBU: unbound variable`；
  `cells/r1_dsr6_kv_prebackbone_sparse_adapter/seed42` 不存在。launcher 在新 Slurm shell 中仍以
  `set -u` source `/etc/profile`，所以提交端 precheck 的局部 nounset 修正没有进入 job。该事件发生
  在 cell、数据、模型、checkpoint 与 validation 前，是 infrastructure failure，不是 DSR6 结果。
  job 封存且不得 resume/requeue/retry；distinct epoch 需先修 launcher 的局部 profile 初始化、
  加 focused regression、独立审查并重做结果盲 PRE_RUN。模型、config 和准确率门保持不变。

- 2026-08-24 04:31 CST：launcher-only clean/pushed candidate
  `4eb40fe3eb67ea3511a16d26e38d6bdca3ca5c93` 形成，父提交 exact `3260cd39…`，只改
  `scripts/run_zoomtoken_official_prebackbone_bc_n16r4.sh` 与
  `tests/test_zoomtoken_r1_refresh_carry_k32.py`。N16R4 focused suite `12 passed in 39.91s`，
  HEAD/ref/clean 通过。fresh 独立 Critic 确认 launcher 边界、fail-closed 与 no-science-drift 正确，
  但因 shell regression 没有显式 `exit 0` 返回 `NEEDS_ATTENTION`。目标 Linux pytest/probe 实际
  PASS 不被主代理用来覆盖独立门；Evaluator/PRE_RUN 与新实验均未启动。

- 2026-08-24 04:40 CST：中央授权的 focused test-only correction 已形成 clean/pushed
  `c6327a891809aa30370b3b2d9bedab0dcfe0d326`（父 `4eb40fe3…`）。相对父提交只在 embedded
  shell regression 末尾增加显式 `exit 0`，没有再改 launcher、模型、config、数据、seed、资源、
  恢复、EMA 或阈值。N16R4 exact checkout 的同一 focused suite 为 `12 passed in 40.20s`；fresh
  独立 Critic 返回 `AUDIT_PASS`。fresh 结果盲 Evaluator 又提交了唯一 no-data/no-model 双卡
  job-shell witness `1252525`，终态 `COMPLETED 0:0`，证明真实 Slurm shell 中 exact profile
  boundary 成功且 nounset 已恢复。canonical THUMOS14 `411/0`、依赖、clean ref 通过；拟定 root
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_dsr6_c6327a89_seed42_20260824` 和 job
  `zt-dsr6-kv-s42-c6327a89` 均不存在。状态为 `PRE_RUN_READY`；没有正式 c632 训练或性能结果，
  旧 job `1252521`/root 继续封存。下一动作仅是请求一个新的 exact 单 cell 授权。

- 2026-08-24 04:46:31 CST：第一次 formal dispatch 因 PRE_RUN witness `1252525` 已占用原拟定
  exact job name，在 `sbatch` 前停止，actual attempt count 为 0、root 仍不存在。中央随后仅把
  正式名称替换为 `zt-dsr6-train-s42-c6327a89`，未改任何科学或运行字段。复核 clean source
  HEAD/ref exact `c6327a891809aa30370b3b2d9bedab0dcfe0d326`、新 job name/root 不存在后，唯一
  DSR6-KV 正式 job `1252527` 成功提交，初始 `PENDING`。result root 为
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_dsr6_c6327a89_seed42_20260824`。该 cell 固定
  canonical THUMOS14 training→validation、seed42、双卡 global/local batch 2/1、60 epochs、每 5
  epoch full-state recovery/latest3+final 与 epoch59 EMA；当前没有性能或成本结论，只进行事件驱动
  只读监控。

- 2026-08-24 04:48 CST：唯一 DSR6-KV job `1252527` 已在 `g0041` 转为 `RUNNING`，start time
  `04:46:36`。stderr 仅见模块加载、torchrun OMP 提示；stdout 已完成模型参数登记并明确出现
  `Training Starts` 与 `Epoch 0 started`。尚无 5-epoch recovery、validation 或终态指标，也未见
  Traceback/OOM/非有限 loss。该更新是运行证据，不是性能或效率证据；矩阵、final-EMA 选择与
  STOP_DEPTH_ROUTE 门保持不变。

- 2026-08-24 05:07 CST：唯一 DSR6-KV job `1252527` 仍在 `g0041` 正常 `RUNNING`，elapsed
  `00:21:05`。日志依次进入 epoch `0–5`；在 `05:06:03` 发布首个完整恢复点
  `cells/r1_dsr6_kv_prebackbone_sparse_adapter/seed42/gpu2_id0/checkpoint/recovery_epoch_4.pth`
  （`627,950,731` bytes），随后开始 epoch 5。未见 Traceback、OOM 或非有限 loss。该事件只证明
  每 5 epoch 的恢复合同已首次实际执行，并不提供中间准确率、最终性能或成本证据；不据此选模，
  继续等待新的恢复点、硬故障或 epoch59 EMA 终态。

- 2026-08-24 05:25 CST：job `1252527` 仍为 `RUNNING`，elapsed `00:39:15`，已依次进入
  epoch `0–10`。第二个恢复点 `recovery_epoch_9.pth` 于 `05:24:53` 发布，大小
  `627,950,731` bytes；首个 `recovery_epoch_4.pth` 仍存在且大小相同。未见 Traceback、OOM 或
  非有限 loss。两份文件证明每 5 epoch 的恢复节奏持续执行，但仍不提供准确率、成本或
  checkpoint 选择证据；继续保持 final epoch59 EMA 主结果与单作业边界。

- 2026-08-24 05:45 CST：job `1252527` 在 `g0041` 继续 `RUNNING`，elapsed `00:59:07`，已进入
  epoch 15。第三个恢复点 `recovery_epoch_14.pth` 于 `05:43:47` 发布（`627,952,846` bytes）；
  `recovery_epoch_4.pth` 与 `recovery_epoch_9.pth` 仍存在，形成冻结保留上限下的三份有效恢复点。
  日志仍无 Traceback、OOM 或非有限 loss。该记录证明恢复节奏和当前三份保留状态，不提供任何
  中间性能或效率证据；下一次恢复发布应同时验证最旧恢复点按 latest-3 规则被替换。

- 2026-08-24 06:03 CST：job `1252527` 仍为 `RUNNING`，elapsed `01:17:13`，已进入 epoch 20。
  `recovery_epoch_19.pth` 于 `06:02:39` 发布（`627,952,846` bytes）；checkpoint 目录当前严格只保留
  epochs `9/14/19`，epoch 4 已按 latest-3 规则移除。未见 Traceback、OOM 或非有限 loss。至此
  “每 5 epoch 发布、只保留最近三份”的恢复合同已在真实训练中完整验证；仍不读取中间性能，
  继续等待后续异常或 epoch59 EMA 终态。

- 2026-08-24 06:22 CST：job `1252527` 继续 `RUNNING`，elapsed `01:35:35`，已进入 epoch 25。
  `recovery_epoch_24.pth` 于 `06:21:40` 发布（`627,952,846` bytes），最近三份恢复点随之轮换为
  epochs `14/19/24`，epoch 9 已移除；未见 Traceback、OOM 或非有限 loss。该事件只确认既定恢复
  机制持续执行，不形成中间性能、效率或选模证据；继续等待硬故障或 epoch59 EMA 终态。

- 2026-08-24 06:41 CST：job `1252527` 仍为 `RUNNING`，elapsed `01:54:42`，已进入 epoch 30。
  `recovery_epoch_29.pth` 于 `06:40:47` 发布（`627,952,846` bytes），最近三份恢复点轮换为
  epochs `19/24/29`，epoch 14 已移除；未见 Traceback、OOM 或非有限 loss。该事件继续只证明恢复
  合同执行正常，不读取中间性能，也不改变 epoch59 EMA 的唯一判定规则。

- 2026-08-24 07:00 CST：job `1252527` 继续 `RUNNING`，elapsed `02:13:14`，已进入 epoch 35。
  `recovery_epoch_34.pth` 于 `06:59:34` 发布（`627,952,846` bytes），最近三份恢复点轮换为
  epochs `24/29/34`，epoch 19 已移除；未见 Traceback、OOM 或非有限 loss。该常规恢复事件不提供
  中间性能或效率证据，继续等待硬故障或 epoch59 EMA 终态。

- 2026-08-24 07:19 CST：job `1252527` 仍为 `RUNNING`，elapsed `02:32:34`，已进入 epoch 40。
  `recovery_epoch_39.pth` 于 `07:18:21` 发布（`627,952,846` bytes），最近三份恢复点轮换为
  epochs `29/34/39`，epoch 24 已移除；未见 Traceback、OOM 或非有限 loss。该常规恢复事件仅证明
  运行与可恢复性，尚无准确率、效率或终态证据。

- 2026-08-24 08:06 CST：job `1252527` 仍为 `RUNNING`，elapsed `03:19:16`，已进入 epoch 45。
  `recovery_epoch_44.pth` 于 `08:05:15` 发布（`627,952,846` bytes），最近三份恢复点轮换为
  epochs `34/39/44`，epoch 29 已移除；未见 Traceback、OOM 或非有限 loss。该常规恢复事件仅证明
  五轮恢复和 latest-3 保留合同继续真实执行，不提供准确率、效率或终态证据。

- 2026-08-24：严格 VideoMAE A-MoD 参照完成于 clean/pushed revision
  `a41714e9f9271906a2eb4505e3fedc590c838055`（branch `codex/zoomtoken-amod-v001`）。实现交替使用
  dense blocks `0/2/4/6/8/10` 与 A-MoD blocks `1/3/5/7/9/11`；后者按前一 dense block 的
  attention-probability column mean 稳定选择 top-400/800，仅更新选中 token 的 Attention+MLP，
  未选 token identity bypass，既有 Adapter 对全部 800 token 保持 dense。N16R4 无数据 focused
  测试 `8 passed`，独立 Critic `AUDIT_PASS`。该记录是实现/审查证据，不是准确率、延迟或能耗结果。

- 同日确认：上述 A-MoD 参照没有帧间 cache/carry，不能代替用户要求的帧时序特征保存和映射。
  跨帧方向继续保持 active；已准备新的 Project Pro 科学讨论，以冻结存储状态、跨帧/tubelet 对齐、
  当前帧变化评分、刷新和失效/场景切换、梯度及无未来帧泄漏边界。决定返回前不提交新的 A-MoD
  或 temporal-memory 正式训练，不干扰唯一运行中的 DSR6 job `1252527`。

- Pro 材料身份回验发现未执行的 v008 请求仍绑定设计提交 `412ab1db…`，因此在任何浏览器 mutation
  发生前将其替代为 v009。新 prepared Sources 为 `CURRENT_RESEARCH_STATE-v013.md`
  (`pfile-20260824T011407Z-72f874b72d`) 与 `MODEL_EXPERIMENT_HISTORY-v008.md`
  (`pfile-20260824T011407Z-2a86b70c1e`)，二者都绑定 implemented/tested/reviewed revision
  `a41714e9…`。新 exact Sources 请求为 `PROJECT_SOURCE_SYNC_REQUEST-v009.md`，fresh Pro 请求为
  `PRO_AMOD_TEMPORAL_MEMORY_ROUTE_ADJUDICATION_REQUEST-v002.md`，queue
  `msg-20260824T011426Z-339f9da9233e`，状态 queued/pending exact lease；尚未上传 Source，也尚未
  提交 Pro turn。

- 同轮完成时序实现入口的只读代码定位。原生 tubelet 提取与 sparse Adapter 已维护
  `(tubelet_index, spatial_index)` lineage；THUMOS sample 已提供 `video_name`、`window_start_frame`；
  ChronoTransport 已有可机械复用的 per-stream cache、detach policy、age、首 chunk 强制
  RECOMPUTE、non-finite fallback 和 current-state delta/cosine signals。当前缺失的是可靠的 token
  correspondence、遮挡/进出画面/scene-cut invalidation 和跨 window identity。实现不得按 DDP batch
  下标持久绑定 cache，也不得把 same spatial index 当成 same physical content；epoch-boundary resume
  应清空 live memory 并 warm-start，除非 Pro 明确冻结另一恢复语义。该定位不改变模型、不提交作业，
  但使 Pro 返回后可直接在明确代码接点实施。

- 2026-08-24 09:14 CST：只读核验 DSR6-KV job `1252527` 仍在 `g0041` `RUNNING`，elapsed
  `04:31:33`，日志已进入 epoch 51。`recovery_epoch_49.pth` 于 `08:52:13` 发布，最近三份恢复点
  为 epochs `39/44/49`；未见 Traceback、OOM 或非有限 loss。日志中的过程性 validation 不作为
  final-EMA 证据、不用于挑选 checkpoint，也不改变准确率门或成本关闭状态。

- 同日形成严格 A-MoD test-only clean successor
  `31e4b1e61a23c4f1b319249684c8f05da6734235`，相对 `a41714e9…` 只修改
  `tests/test_zoomtoken_amod_paper_exact.py`。新增测试分别用逐块唯一 marker 证明每个 A-MoD block
  消费紧邻前一 Dense block 的 routing score，并在官方 16-frame/160×160 几何上 hook 全部 12 个
  Adapter，证明每次输入均为 `[1,800,8]`。远端标准 OpenTAD 环境 CPU-only 运行结果为
  `10 passed in 35.05s`；未申请 GPU/Slurm、未读取数据或结果。该后继不改变模型或科学合同。

- 2026-08-24：fresh Project Pro 完成时序记忆路线裁决，结论为 `REVISE / APM32-CTX64`。冻结方法
  使用严格 R1 K64 支持、前一 tubelet detached pre-position patch memory、半径 2 mutual-nearest
  对齐与 `0.80` 阈值；有效匹配不足 32 时 K64 fallback，否则 K32 refresh/K64 context。
  `CUR32-CTX64` 共享 mask/fallback、仅使用当前 embedding，作为记忆替换的 matched control。
  该裁决是 preexecution design，不是准确率、延迟或能耗证据。

- 同日完成七文件初始候选 `435ab8dd6a102a96f26b2e37e2655e711277dcfd`。fresh 独立 Critic
  发现一个真实可达的批处理缺陷：各样本 fallback 数不同时，执行器仍错误要求批内刷新总数相等。
  两文件聚焦后继 `d985dfb8b0cba4f70c28770643145ee44cb451d2` 改为按实际 Query 数分桶，
  并新增同批 `64/96` 刷新数 known-answer。N16R4 标准 OpenTAD 环境在 exact clean SHA 上
  `13 passed in 34.74s`；fresh Critic 复核为 `AUDIT_PASS`。未使用数据、GPU、Slurm 或运行结果。

- fresh 结果盲 Evaluator 随后返回 `PRE_RUN_NOT_READY`。代码/配置/launcher 合同已通过，尚缺
  两项正式准入收据：Fit/train 单批有限 loss 前向/反向及精确执行账本；完整 model/optimizer/
  scheduler/scaler/EMA/RNG save-resume fixture，并证明 temporal memory 不被 checkpoint 保存。
  在两项证据形成前不提交 APM/CUR 60-epoch cell；DSR6 job `1252527` 独立且未被读取或干预。

- 随后完成结果盲执行 successor `e92df6a4737a10955722c6aedc2f079e0d285a18`，父提交为
  `d985dfb8…`，分支 `codex/zoomtoken-apm32-ctx64-v001`。该后继不改 APM/CUR 模型、配置或科学
  门，只补齐 production one-batch preflight 与 APM/CUR 的既有五轮完整恢复入口；保存—加载检查
  覆盖 model、EMA、optimizer、scheduler、scaler、训练计数、sampler 与 RNG，并拒绝 temporal
  memory 进入 checkpoint。preflight 不构建 validation/test loader，不输出指标。

- exact clean SHA 在 N16R4 标准 OpenTAD 环境执行 CPU-only focused suite，结果为
  `19 passed, 1 warning in 50.83s`；fresh 独立 Critic 返回 `AUDIT_PASS`。fresh 结果盲 Evaluator
  的终态仍为 `PRE_RUN_NOT_READY`，唯一剩余缺口是 APM/CUR 各一次实际双卡 Slurm 单批机械见证。
  资源请求已固化为 `ZOOMTOKEN_APM32_CTX64_PREFLIGHT_RUN_REQUEST-2026-08-24.md`；本轮没有
  GPU/Slurm 权限，因此未提交见证或正式训练，亦未触碰 DSR6 job `1252527`。

- 2026-08-24 10:50 CST 对唯一 DSR6-KV job `1252527` 做一次只读监控。scheduler 仍为
  `RUNNING`（`g0041`，elapsed `06:04:01`）；新发布 `recovery_epoch_54.pth`
  (`627,952,846` bytes)，latest-three 为 `44/49/54`，并已发布预注册
  `checkpoint/epoch_59.pth` (`627,933,747` bytes)。精确根未变，未见 Traceback、OOM 或
  non-finite loss。final validation 尚未终态，因此这些是训练/恢复执行证据，不是准确率、成本或
  checkpoint 选择证据；未读取或提升任何 live metric。

- 2026-08-24 11:03 CST：唯一正式 DSR6-KV job `1252527` 已终态 `COMPLETED 0:0`；运行区间
  `04:46:36–10:53:53`、elapsed `06:07:17`、节点 `g0041`。`epoch_59.pth` 和最近恢复点
  `44/49/54` 存在，硬故障扫描未见 Traceback/OOM/non-finite。精确结果根未发现独立
  result/finalization/metric JSON，本次有界终态快照没有取得可审计 final-EMA 三指标，因此
  准确率门保持 `UNADJUDICATED`，既不宣称 PASS 也不推定 STOP；成本与追加实验继续关闭。
  下一步仅为另行授权的不可变终态日志/EMA 只读结果摄取；不重训、不恢复、不补臂。

- 2026-08-24：按用户的实验进展/性能报告请求，对 job `1252527` 的精确不可变终态 stdout
  做只读摄取。训练在 epoch 59 后进入最终 validation；`eval_one_epoch` 在配置 `ema=True` 时先
  加载 `model_ema`，故终态 Avg-mAP/mAP@0.6/mAP@0.7 `67.38/59.34/46.01` 是预注册 final-EMA
  结果。相对门 `68.57/60.64/46.07` 分别低 `1.19/1.30/0.06` 点，裁决
  `STOP_DEPTH_ROUTE`。未启动成本、额外 seed、K24/K18 或结构补救；理论重块 FLOPs
  `79.055%` 继续只作算量代理，不解释为实际延迟或能耗。

- 2026-08-25：完整矩阵 Pro 科学复核返回 `REVISE`，选择性能优先的
  `R1-APM-C32/FULL64`。旧 APM32 同时改变帧间载体和 K32 深层更新，因果解释不单一；新实验
  保留既有 one-tubelet detached memory、半径 2 双向一致匹配、阈值 0.80、clip reset 与 K64
  fallback，只让 32 个可靠匹配位置使用前一表征与当前残差形成的输入载体，全部 K64 随后在
  12 个 VideoMAE block 和 Adapter 中完整更新。该路线不新增参数、loss 或 cache，不声称节省
  计算，也不恢复已经停止的深度稀疏分支。

- clean/pushed candidate `bffff43dad28ca1042602ad3a01ba2990b953c13`（父提交
  `e92df6a4…`，分支 `codex/zoomtoken-r1-apm-c32-full64-v001`）已在 N16R4 标准 OpenTAD 环境
  通过 py_compile、bash syntax 与 focused pytest（`22 passed in 72.88s`）；fresh 独立 Critic
  为 `AUDIT_PASS`，fresh 结果盲 Evaluator 为 `PRE_RUN_READY`。本机 pytest 因 Windows Torch
  DLL `WinError 1114` 未收集，未被计为通过。

- 唯一正式 job `1254008`（`zt-apm-c32-full64-s42-bffff43d`）已提交，结果根为
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_apm_full64_bffff43d_seed42_20260825`。首次启动核验时
  scheduler 为 `RUNNING`，节点 `g0003`；CUDA 11.8、Miniforge/OpenTAD 环境和双进程 DDP 入口
  已正常建立，未见启动 Traceback。该项只说明正式执行开始；尚无 final-EMA 准确率或效率证据，
  不读取或提升 live metric。

- 2026-08-25 00:45 CST：对 exact job/root 的一次只读训练循环核验确认，job `1254008` 仍在
  `g0003` 运行，日志于 `00:39:55` 写出 `Training Starts...`；硬故障扫描未见 Traceback、OOM、
  non-finite loss，尚无首个五轮恢复点。该事件仅证明训练循环已进入，不形成准确率或效率结论。

- 同轮对 exact candidate 的时序载体执行目标环境无数据 known-answer check，排除载体退化为当前
  表征恒等映射。224 个载体位置的 `alpha` min/mean/max 为
  `0.00061572/0.02699333/0.08087695`，载体相对当前表征最大绝对差为 `0.60094869`。只对
  tubelet 1 反传时，tubelet 0 memory 梯度为 `0`、当前 tubelet 梯度非零、其他 tubelet 梯度为
  `0`。这证明 frozen previous + live current residual 及 stop-gradient 语义真实生效；不使用真实
  数据，也不形成模型性能或效率证据。

- 同轮完成严格 VideoMAE A-MoD 深度实现复核。`31e4b1e6…` 中 12 层固定为偶数 Dense、奇数
  A-MoD；每个 A-MoD block 只消费紧邻前一 Dense block 的 attention column mean，稳定选择
  top-400，在所选子序列执行完整 MHSA+MLP，未选 token 对这两个预训练子块保持 identity bypass；
  AdaTAD Adapter 按冻结设计继续在完整 800-token 网格执行，且没有新增参数、路由 loss 或时序
  cache。模型允许 `capacity=1.0` 不是正式配置漂移，而是设计要求的 dense checkpoint parity
  验证入口；正式配置仍固定 `capacity=0.5`。现有 N16R4 `10 passed` 与独立 Critic 只证明实现，
  A-MoD 正式训练仍缺 capacity-1 official final-EMA parity 与实际双卡生产路径的结果盲见证。

- 2026-08-25 00:56 CST：完成 A-MoD 结果盲资源核验。官方 job `1245842` 的 final checkpoint
  `/data/run01/sczc063/yuzibo/projects/official_adatad_reproduction_run_01c58b9_seed42_commandfix_20260821/gpu2_id0/checkpoint/epoch_59.pth`
  存在；remote source `/data/run01/sczc063/yuzibo/projects/zoomtoken_amod_src_31e4b1e6`
  的 HEAD/ref 均为 `31e4b1e61a23c4f1b319249684c8f05da6734235` 且 dirty=0；独立结果根和
  job 名提交前不存在。`sbatch --test-only` 通过后提交唯一 capacity-1 final-EMA parity job
  `1254014`（`zt-amod-c1-parity-31e4b1e6`），初始为 `PENDING`。该 job 只验证新 A-MoD
  执行路径在 capacity=1 时与官方 Dense 等价，不训练参数、不代表 A-MoD-50 性能，也不干预
  正在运行的时序载体 job `1254008`。

- parity job `1254014` 在 `00:00:00` 终态 `FAILED 2:0`，stderr 为
  `/var/spool/slurmd/job1254014/slurm_script: 4: set: Illegal option -o pipefail`。Slurm `--wrap`
  默认由 `/bin/sh` 解释，尚未进入 profile、Python、checkpoint、数据或模型。该 root 保持不存在；
  不形成实现或性能证据。随后只把 wrapper 改为显式 `/bin/bash -c`，其余 source/checkpoint/
  config/data/evaluator/root 全部不变；再次通过 `sbatch --test-only` 后提交机械后继 job `1254016`
 （`zt-amod-c1-parity-31e4b1e6-b1`），初始 `PENDING`。不再复用或恢复 `1254014`。

- 2026-08-25 01:05 CST：A-MoD capacity-1 parity job `1254016` 终态 `FAILED 1:0`，运行
  `00:01:18`。它在模型构建和官方 checkpoint 读取后、validation 推理前退出：官方 epoch-59
  `state_dict` 与 `state_dict_ema` 各有 499 个键，全部统一保留 DDP `module.` 前缀；旧
  `tools/test.py` 却把 EMA 一律加载到无前缀的内部模型。这是确定性评估入口缺陷，不是 A-MoD
  准确率、训练或效率结果。clean/pushed 修正 `2d945e64bdccd09ae2e2916524562e3f388c5a2a`
  仅按统一前缀选择 DDP 外壳或内部模型，并拒绝混合/空命名空间；目标 N16R4 focused suite
  `20 passed`，fresh 独立 Critic `AUDIT_PASS`。本机 pytest 因 Windows Torch `c10.dll`
  初始化失败未能收集，未记作测试通过。

- 2026-08-25 01:17 CST：结果盲 Evaluator 在 exact clean remote source
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_amod_eval_prefix_src_2d945e64` 复核 HEAD/ref、
  官方 epoch-59 EMA、canonical THUMOS14 validation、配置与 capacity=1 override 后，完成唯一
  `sbatch --test-only` 并提交 distinct parity job `1254038`（`zt-amod-c1-parity-2d945e64`），
  初始 `PENDING/Priority`。结果根为
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_amod50_capacity1_parity_2d945e64_20260825`。
  该 job 只决定 A-MoD 实现是否与官方 Dense 数值等价；A-MoD-50 仍未训练，APM job `1254008`
  未被中断或重提。

- parity job `1254038` 在 `00:00:01` 终态 `FAILED 127:0`；Slurm 脚本再次由 `/bin/sh`
  解释，`source/module/torchrun` 均不可用，同时结果盲命令误绑定了不存在的旧
  `/data/run01/sczc063/yuzibo/raw_data/video`。它未进入 profile、Python、checkpoint、数据或模型，
  结果根仍不存在。只将作业壳改为显式 `#!/bin/bash`，并绑定已核验的 canonical
  `/data/run01/sczc063/yuzibo/thumos14/raw_data/video`（411 MP4、0 断链）；其余 frozen tuple
  不变。一次 `sbatch --test-only` 通过后提交 distinct job `1254040`
  （`zt-amod-c1-parity-2d945e64-b1`）。该 job 已加载官方 epoch-59 checkpoint、选择 EMA 并进入
  `Testing Starts`，尚无终态向量或 parity 结论。

- 2026-08-25 01:34 CST：Ego4D 官方访问获批后，将 AWS profile 仅安装到远端数据盘的私有
  凭据目录（目录 mode 700、文件 mode 600），未写入仓库、下载日志或 wiki。通过学术加速节点读取
  官方 v2 manifests 并逐对象核验 `ContentLength` 后，确认 clips manifest 不含 `NLQ` 标签，旧
  `--benchmarks nlq` 方案会选择 0 个视频；因此改为下载三份 `nlq_*.json`，再以其中 2,019 个唯一
  `clip_uid` 精确连接官方 clips manifest。冻结下载计划为 22,408 条语言查询、2,025 个文件、
  158,680,155,599 bytes，其中 clips 为 158,597,610,649 bytes；预检无缺失 S3 对象。启动前数据盘
  可用 1,668,814,585,856 bytes，完整下载后预计仍有 1,510,134,430,257 bytes，容量门禁通过。
  三份 NLQ annotations 与 66,687,246-byte primary metadata 已完成；官方 `ego4d-cli 1.7.3` 的
  `--skip-s3-checks --yes` 路径暴露 `expected_gb` 未初始化缺陷，隔离安装副本仅修正该分支后，
  可续传 clips 下载已在 PID `418299` 下进入传输。当前状态仅为 `download_running`，完成标记必须
  等 2,019 个 annotation-referenced clips 全部存在且非零、CLI 成功退出并写出最终验证/receipt。

- 2026-08-25 03:06 CST：A-MoD capacity=1 Dense 等价性 job `1254040`
  （`zt-amod-c1-parity-2d945e64-b1`）终态 `COMPLETED 0:0`，运行 `00:14:58`。使用共享官方
  epoch-59 EMA 与 canonical THUMOS14 validation 得到 mAP@0.3/0.4/0.5/0.6/0.7
  `83.46/79.45/71.96/61.59/47.20`，Avg-mAP `68.73`；官方记录为
  `83.46/79.44/71.94/61.58/47.24`，Avg-mAP 同为 `68.73`。最大绝对差 `0.04` 个百分点，
  判定 capacity=1 实现数值等价性通过但非逐位一致。该结果只证明 checkpoint/EMA、A-MoD
  gather/scatter 与官方 Dense 路径对齐，不是 A-MoD-50 方法性能或效率证据；A-MoD-50 仍未训练。
  帧间载体正式 job `1254008` 独立保持 `RUNNING`，两条路线尚未组合。

- 2026-08-25 02:45 CST：对 `R1-APM-C32/FULL64` exact job/root 的一次只读核验确认，job
  `1254008` 仍在 `g0003` 正常运行，累计运行 `02:07:47`，训练已进入 epoch 35。结果根的正式
  checkpoint 目录现保留 `recovery_epoch_24.pth`、`recovery_epoch_29.pth` 和
  `recovery_epoch_34.pth`，每份 `627,952,846` bytes，符合“每 5 epoch 保存、最近 3 份”恢复
  合同。日志扫描未见 Traceback、OOM 或 non-finite 硬故障。本条只记录执行与可恢复性证据；
  不读取或提升中间验证指标，终态 epoch-59 EMA 准确率和效率结论仍不存在。

- 2026-08-25 03:04 CST：对同一 `R1-APM-C32/FULL64` exact job/root 的只读核验确认，job
  `1254008` 仍在 `g0003` 正常运行，累计运行 `02:28:11`，训练已进入 epoch 40。新增
  `recovery_epoch_39.pth` 后，当前按“最近三份”策略保留 `recovery_epoch_29/34/39.pth`，
  每份 `627,952,846` bytes；epoch 24 恢复点已按预注册策略正常轮换。日志扫描未见
  Traceback、OOM 或 non-finite 硬故障。本条仍只记录执行与可恢复性证据，不读取或提升
  中间验证指标；终态 epoch-59 EMA 准确率和效率结论仍不存在。

- 2026-08-25（远端记录 03:49 CST）：对同一 `R1-APM-C32/FULL64` exact job/root 的一次只读
  核验显示 job `1254008` 仍在 `g0003` 运行，并新生成 `recovery_epoch_44.pth`
  （`627,952,846` bytes）；当前按“最近三份”策略保留 `recovery_epoch_34/39/44.pth`，epoch 29
  已正常轮换。日志扫描未见 Traceback、OOM 或 non-finite 硬故障。本条仅记录执行与可恢复性
  证据，不读取或提升中间验证指标；终态 epoch-59 EMA 准确率和效率结论仍不存在。

- 2026-08-25 04:34 CST：对同一 `R1-APM-C32/FULL64` exact job/root 的一次只读核验显示 job
  `1254008` 仍在 `g0003` 运行，并新生成 `recovery_epoch_49.pth`（`627,952,846` bytes）；当前按
  “最近三份”策略保留 `recovery_epoch_39/44/49.pth`，epoch 34 已正常轮换。日志扫描未见
  Traceback、OOM 或 non-finite 硬故障。本条仅记录执行与可恢复性证据，不读取或提升中间验证
  指标；终态 epoch-59 EMA 准确率和效率结论仍不存在。

- 2026-08-25 05:33 CST：对同一 `R1-APM-C32/FULL64` exact job/root 的一次只读核验显示 job
  `1254008` 仍在 `g0003` 运行，并新生成 `recovery_epoch_54.pth`（`627,952,846` bytes）；当前按
  “最近三份”策略保留 `recovery_epoch_44/49/54.pth`，epoch 39 已正常轮换。日志扫描未见
  Traceback、OOM 或 non-finite 硬故障。本条仅记录执行与可恢复性证据，不读取或提升中间验证
  指标；终态 epoch-59 EMA 准确率和效率结论仍不存在。

- 2026-08-25 06:18 CST：同一正式 job `1254008` 的预注册 `epoch_59.pth` 已落盘
  （`627,933,747` bytes），最近恢复点仍为 epoch 44/49/54；scheduler 仍为 `RUNNING`，本次
  核验未见 Traceback、OOM 或 non-finite 硬故障。这只证明 60 轮训练产物已经生成，不代表
  validation 或作业终态；在终态前不读取或提升 final EMA 指标，准确率和效率结论仍不存在。

- 2026-08-25 06:32 CST：`R1-APM-C32/FULL64` job `1254008` 在 `g0003` 终态
  `COMPLETED 0:0`，总运行 `05:54:00`。不可变日志已出现 `Training Over`，未见 Traceback、
  OOM 或 non-finite 硬故障；`epoch_59.pth` 与 recovery epoch 44/49/54 均存在。预注册 epoch-59
  final EMA 的 mAP@0.3/0.4/0.5/0.6/0.7 为 `83.74/78.93/72.41/60.43/45.60`，Avg-mAP
  `68.22`。相对 all-of 门 `68.73/61.58/47.24`，Avg-mAP/mAP@0.6/mAP@0.7 分别低
  `0.51/1.15/1.64` 个百分点，三项均失败，裁决为 `STOP_APM_MEMORY`：不启动成本、额外 seed
  或结构补救。这是当前 detached one-tubelet APM 载体在 FULL64 计算下的有效负结果，不是对所有
  时序去冗余路线的普遍否定；本实验没有计算节省，也没有 selector-inclusive 效率证据。

- 2026-08-25：项目状态复核发现严格矩形 R4-SHUF15 的终态此前未被摄取。只读 Slurm 与不可变
  日志核验确认 job `1249131` 已于 `2026-08-23 03:40:57 CST` 在 `g0022` 终态
  `COMPLETED 0:0`，运行 `05:50:51`；`epoch_59.pth` 和 recovery epoch 44/49/54 均存在，
  未见 Traceback、OOM 或 non-finite hard failure。`Training Over` 前的 final EMA 为
  mAP@0.3/0.4/0.5/0.6/0.7 `81.66/78.04/69.90/60.17/46.20`，Avg-mAP `67.19`。
  相对 R4 `68.02/60.32/46.26`，R4−R4-SHUF15 为 `+0.83/+0.15/+0.06`；其中 mAP@0.7
  未达到预注册的 `+0.30` 框外内容排序门，因此不能声称框外 learned ranking 有效，也不启动该
  机制的成本或多 seed。Q64-GLOBAL 终态为 `67.84/60.66/45.39`，R4−Q64 的
  `+0.18/-0.34/+0.87` 是交叉结果，不能替代失败的 R4-SHUF15 因果门。

- 2026-08-25 03:31 CST：Ego4D-NLQ v2 下载完成并通过冻结验证。自动续传共进入 5 次传输
  attempt，最终官方 CLI `exit=0`；远端目标包含 2,019/2,019 个 annotation-referenced clips，
  合计 158,597,610,649 bytes，与逐对象 S3 `ContentLength` 计划精确一致，且
  `missing_clips=0`、`zero_byte_clips=0`。三份 NLQ annotation 共覆盖 22,408 条 language queries；
  active 临时 `.mp4.*` 文件为 0。目标根已生成 `DOWNLOAD_COMPLETE`、
  `VALIDATION_REPORT.txt` 与 `DOWNLOAD_RECEIPT.txt`，下载进程和自动续传监督器均按预期退出；
  状态由 `download_running` 更新为 `download_complete_verified`。完成后数据盘仍可用
  1,507,860,164,608 bytes。该记录不包含 AWS 凭证内容。

- 2026-08-25：按用户明确的准确率—计算联合目标重新解释已完成的深度稀疏矩阵。原近无损门
  失败记录保持不变，但不再把它等同于效率候选全面终止。相对 FULL64，DSR6-KV/MOD32-KV/
  DROP32 的 VideoMAE 重块算量代理分别减少 `20.94%/41.89%/50.68%`，Avg-mAP 分别下降
  `1.69/2.57/2.96`，mAP@0.7 分别下降 `0.56/1.36/1.69`；三者保留为保守、中等、激进
  Pareto 候选。RC32-KV 因与 MOD32-KV 同代理成本而三项准确率更低，APM-C32/FULL64 因无
  重块节省却降低准确率，二者继续停止。当前没有运行新作业；下一项最小证据是复用既有 final
  checkpoints 做同硬件 FULL64/DSR6/MOD32/DROP32 完整端到端延迟、显存与能耗对比。代理 FLOPs
  在实测前不得写成真实速度、显存或能耗收益。

- 2026-08-25：在 exact ZoomToken Project 中完成一次 fresh Pro token 迁移/变换与 VideoMAE
  实现讨论。会话 `6a8d25c5-30ac-83e9-9f57-008f0167782c`、浏览器可核验 `Pro`，单轮终态
  `completed`，原始中文报告保存为
  `.cvpr-pro-lab/reviews/PRO_TOKEN_REUSE_PRIOR_ART_AND_VIDEOMAE_DESIGN_RESPONSE-v001.md`。
  Pro 返回 `PIVOT` 并设计 `R1-ACR16-Δ1-FKV`，但独立一手来源核验发现其遗漏最近邻 Eventful
  Transformers（ICCV 2023），该工作已经实现变化 token 检测、reference/buffer、选择性重算和
  稀疏/增量注意力。因此项目不全盘接受新颖性结论，也不直接交给 Builder；状态记录为
  `DISCUSSED / NEEDS_PRIOR_ART_CORRECTION`。核验回执位于
  `.cvpr-pro-lab/reviews/PRO_TOKEN_REUSE_PRIOR_ART_AND_VIDEOMAE_DESIGN_AUDIT-v001.md`。本轮无代码、
  GPU、Slurm、训练、准确率或效率变更。

- 2026-08-25：按用户要求清理 exact ZoomToken ChatGPT Project 的长期 Sources。远端 Source
  从 40 份降为 3 份，只保留 `PROJECT_CHARTER-v001.md`、
  `CURRENT_RESEARCH_STATE-v016.md` 与 `MODEL_EXPERIMENT_HISTORY-v011.md`；37 份旧版本、
  旧路线页和讨论专用 Pro/审查材料已从远端删除，本地 research-wiki、ledger 与正式证据保留。
  后续 Pro 请求、旧响应、审查和代码上下文只作为对应会话附件，不再长期占用 Project Sources。
- 2026-08-25：使用上述三份长期 Sources，并以附件方式提交正式请求、上一轮 Pro 回答和独立核验，
  在 exact ZoomToken Project 创建全新单轮 Pro 会话
  `6a8d706a-4170-83ea-b6bf-b6759d44020e`。终态为 `completed / STOP`：Eventful Transformers
  已覆盖 reference/buffer、变化 token 选择和稀疏/增量更新；ACR16 的剩余低秩差分与条件深度
  跳过不构成新的时序复用原理。独立算量复核确认主块理论节省上限 `9.446%`、已知骨干算术上限
  约 `8.80%`，不足以为完整链路同时达到 `>=5%` 延迟与能耗改善留出可信余量。因此 ACR16/
  Eventful-transfer 在实现前停止：0 个新训练单元，无 Builder、PRE_RUN、GPU 或 Slurm 行为，也无
  新准确率/效率证据。裁决不外推到全部时序冗余方向。

- 2026-08-25：按用户授权，在 exact ZoomToken Project 以一个未压缩文本附件包发起全新单轮 Pro
  会话 `6a8da9f2-3b60-83e9-87b5-6686919cf0e6`，固定 temporal revision `bffff43…`、
  A-MoD revision `a41714e…` 和三份长期 Sources。初始本地捕获超时后只 reattach 同一 session，
  无重提、无 follow-up；终态 `completed / STOP_BEFORE_IMPLEMENTATION`。Pro 比较 IC-DRU、
  OW-ECR、PCD-DRU，给出理想已知骨干节省上界约 `50.12%/7.66%/60.16%`，但认定前两种动态
  机制需要缺少准确率先验的约 25% refresh、可被已有变化路由/cache/ResidualViT/MoD 组件分解，
  窗口缓存则缺少全链路余量并增加约 `54 MiB/样本` 状态。因此三个精确定义候选均在实现前关闭，
  0 个 Builder/PRE_RUN/训练单元；没有新准确率、延迟、能耗或显存证据。该裁决不等于动态
  20%–30% refresh 或所有时序复用已经被实验否定。

- 2026-08-26：完整摄取用户提供的 `ZoomToken_开放式调研与科学裁决报告_R-PADT-v0.md` 与
  对话粘贴文本，并对来源身份、固定 revision 代码、理论成本、相关工作和科学机制做独立核验。
  项目内原 Pro 终态回执仍为 `TERMINAL_INCOMPLETE_NO_SCIENTIFIC_DECISION`，nonce/附件陈述存在
  矛盾，故外部报告不能直接成为同会话 Pro 许可。代码确认 T=384、K64、D=384、12 blocks，
  同时确认 dense Adapter、位置/时间身份和 anchor-copy 是报告未闭合的关键边界；R1 按 tubelet
  独立选择九个合法 8×8 框之一，故全 K64 锚点一一映射也不是既有事实。先导工作补入 STA/PVC，
  粘贴引用有多处错配。最终裁决为
  `PARTIAL_ACCEPT_REVISE_BEFORE_G4 / discussed`：吸收 clip-local 状态、identity gate、显式
  provenance、同预算反事实和完整链路成本原则，但不接受“完整前帧表征复用”表述，不冻结原始
  `L_p=2/R=4/m=16/Q=4`，没有代码、Builder、PRE_RUN、训练、准确率或效率状态变化。详细审计：
  `research-wiki/sources/2026-08-26-r-padt-v0-user-report-intake-audit.md`。

- 2026-08-26：用户完成 iXBrowser profile 61 登录后，只读核验 exact ZoomToken Project 与三份
  长期 Sources，并以附件方式创建唯一全新 Pro 会话
  `6a8e894d-bc84-83ea-8151-fb81a893a103`。Oracle `0.17.1` 解析网页 picker 为 `Pro`，会话
  `completed / EXITED_0`，无 follow-up、无 Source 变更。顶层裁决为 `STOP`：在固定
  `bffff43d...` 结构中，16 帧 clip 不是 Adapter 后完整状态单位，block-11-only cache 无法闭合
  changed-token 的逐层上下文，逐层 cache 又与 Eventful/STC 的事件化选择性重算高度重叠并可能
  损失 48-clip attention batching。独立代码核验支持主要状态依赖结论，但把完整并行损失标为需
  实测的系统推断；文献点验确认 Eventful 与 CVPR 2026 STC-Cacher 是真实近邻。项目接受对该
  精确 full-representation route 的 `STOP_BEFORE_IMPLEMENTATION`，不外推为全部时序复用无效。
  本轮无代码、配置、PRE_RUN、GPU/Slurm、训练或新准确率/效率结果。

- 2026-08-26：按用户要求在 exact ZoomToken Project 以附件方式发起新的前进路线 Pro 裁决，会话
  `6a8ec964-8984-83e9-afe5-7b363919f8d0`。初始捕获超时后只 reattach 同一 conversation，
  无重提、无 follow-up；终态 `completed / PIVOT`。Pro 从历史 hidden/KV/完整表征复用转向
  `ZoomToken-BPNS-R1`：当前观测、严格连续 `8x8/K64` 原生支持、全部 K64 完整执行 12 层与
  Adapter。现有 K100 与 R1 final-EMA 为 `68.51/61.19/46.27` 和
  `69.07/61.14/46.57`，故下一项证据冻结为零新增训练，优先同硬件重放 K100 job `1248835`
  与 R1 job `1249099`，测完整延迟、显存、能耗和边界质量；DSR6/MOD32/DROP32 为辅助点。
  项目接受科学 PIVOT，但把执行简化为精确 job/config/path/EMA 绑定和最小数值 parity，不新增
  checksum/协议框架。本轮未修改模型、未运行 GPU/Slurm/训练，也没有新增性能或成本结果。

- 2026-08-27：`ZoomToken-BPNS-R1` 同硬件 final-EMA 成本回放完成最小实现、独立审查、结果盲
  PRE_RUN 并进入正式执行。最终 clean/pushed candidate 为
  `b7357817d81127ab2d713b5471d008ea893efd35`（分支
  `codex/zoomtoken-bpns-r1-cost-v001`），只新增 profiler、N16R4 launcher 和 focused test 三个路径，
  不修改模型、训练配置、数据或 checkpoint。目标环境检查纠正了两个确定性实现问题：旧的 40-video/
  136-window 开发 population 不是本次官方 validation loader；功耗 sidecar 若在 CPU affinity 收窄后
  启动会错误继承 detector CPU 集合。最终合同使用 211 个 validation 视频、792 个有序 loader 项，
  并用 ordinal 身份保留其中一个官方重复内容项。Python/Shell 检查通过，focused pytest `6 passed`，
  fresh independent Critic 为 `AUDIT_PASS`。结果盲 Slurm job-shell 见证 job `1257250` 已
  `COMPLETED 0:0`，fresh Evaluator 为 `PRE_RUN_READY`。唯一正式 replay job `1257281`
  （`zt-bpns-formal-b7357817`）已在 `g0003` 进入 `RUNNING`，使用一张 Slurm GPU、5 CPU，
  以 ABBA+BAAB 顺序各重放 K100/R1 四次完整 validation。结果根为
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_r1_cost_b7357817_seed42_20260827`。
  当前仅形成实现、准入和运行证据；尚无新的终态准确率、延迟、显存、能耗或边界质量结论。

- 2026-08-27：正式 BPNS-R1 成本回放 job `1257281` 在 `g0003` 运行 `00:38:04` 后终态
  `FAILED 1:0`。首个 K100 完整 validation pass 得到 `mAP@0.7=46.246663`；profiler 的
  fail-closed 数值一致性门将其与预填历史值 `46.27` 比较并抛出 `RuntimeError`。该作业没有进入
  后续 R1/ABBA+BAAB 完整回放，结果根未发布 `profile.json` 或 `terminal_receipt.json`。
  因此这是确定性的回放准入/原始数值绑定失败，不是模型性能、延迟、显存、能耗或边界质量结果；
  任何局部产物均不解释。job 与结果根保持失败证据，未经新的独立检查和明确运行授权不得重提。

- 2026-08-28：按新的 Pro 唯一任务，从 `b7357817d81127ab2d713b5471d008ea893efd35` 建立
  最小 clean/pushed 候选 `e9323448f6cd78b99bb3de53fd9ffb55f3676d65`（分支
  `codex/zoomtoken-bpns-r1-parity-v002`）。候选只修改 BPNS cost profiler 的 accuracy-parity
  合同及 focused tests：六项 raw evaluator fraction 转未舍入百分点，与冻结 reported-2dp
  reference 在 inclusive `0.05 pp` 内比较，HALF_UP 两位展示不参与准入。Python/Shell/diff
  检查通过，focused pytest `13 passed`；fresh independent Critic 为 `PASS`，fresh result-blind
  Evaluator 在确认 411/0、211/792、两份 epoch-59 EMA、新 root/name 和 `sbatch --test-only`
  后给出 `PRE_RUN_READY`。唯一正式 job `1258299`（`zt-bpns-r1-pv2-e9323448`）于
  `2026-08-28 00:20:24 +08:00` 在 `g0048` 进入 `RUNNING`，使用 1 张 Slurm-visible GPU、
  5 CPU、8 小时上限，结果根为
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_r1_cost_parity_e9323448_seed42_20260828`。
  当前只形成实现、准入和运行证据；终态前不读取或解释任何局部准确率、延迟、显存、能耗、
  短动作或边界数值，也不重复提交。

- 2026-08-28：结果盲终态审计准备发现并经独立复核确认：profiler 顶层
  `comparison.r1_over_k100` 将每臂四个 pass 的全部窗口合并后计算 p50 与窗口平均能耗，
  不等同于 Pro 冻结的“每个完整 pass 先统计、再对四个 pass 取 arm-level median”主估计。
  `cost_samples.jsonl` 的每行已有 `arm`、`pass_index`、`end_to_end_serial_ms` 与包含摊销
  final-NMS 的 `gpu_energy_j`，因此终态可在不修改作业、不重放且不读取局部结果的前提下，
  严格重算每 pass p50、每 pass 总能耗及其四-pass 中位数。后续必须同时披露该冻结主估计与
  profiler pooled 描述性汇总，不能直接用后者应用 5% 论文门槛。

- 2026-08-28：在不读取 job `1258299` 的 live/partial 数值、不查询或干预 Slurm 的前提下，完成
  profiler 终态证据覆盖的结果盲代码/协议审计。直接产出为每臂一份 canonical prediction、
  `profile.pass_receipts` 内的逐 pass evaluator 向量、逐窗口 `cost_samples.jsonl`、功耗轨迹与终态
  profile/receipt；重复 pass prediction 只在内存中检查相等。原始行保存 `arm`、`pass_index`、
  有序 dataset-item 身份及成本字段，足以重建 population、逐 pass p50/总能耗和四-pass 中位数，
  因而这些属于可重算证据，不需要也不授权重放。当前未直接产出八份独立 prediction/evaluator
  文件、单独 population receipt、定量功耗 coverage/gap 统计、pass/window 时间区间或温度；不得
  事后合成这些材料，也不得把观察到的功耗顺序漂移称为热漂移。功耗样本缺失或非法会使积分失败，
  所以成功终态只能间接证明测量窗口完成积分。final video-level NMS 的时间与能耗被摊销到窗口行，
  终态报告必须披露。该审计没有形成性能结论、没有改变科学路线，也不授权取消、重启或追加实验；
  终态后应把直接、可重算和未测证据连同全部异常交给 fresh Pro，独立裁决其论文证据充分性。

- 2026-08-28：继续对 frozen candidate `e9323448…` 做结果盲 provenance 点验。accuracy reference
  合同直接保存 source revision/path/symbol/SHA；每臂 prediction JSON 虽会落盘，但不保存 prediction
  hash。`profile.json` 的 software 只包含 Python/Torch/CUDA 摘要，precheck 只保存 execution commit、
  population 数量和 checkpoint epoch/EMA 参数数/role，不内嵌完整命令、annotation/class-map/video/
  checkpoint 路径、完整 package lock 或成功侧 anomaly 清单。精确路径与命令仍可由冻结启动回执、
  launcher 和 Slurm 日志外部复原，终态交接必须把它们标为外部 provenance，而不是 profiler 直接
  产物；不得在运行后补造 prediction hash、命令字段或空 anomaly 记录。本次点验不读取结果、不改
  代码、不干预 job，也不改变 Pro 已冻结的任务或判据。

- 2026-08-28：唯一正式 BPNS-R1 v002 成本回放 job `1258299`
  （`zt-bpns-r1-pv2-e9323448`）在节点 `g0048` 从 `00:20:24` 运行至 `01:33:30 +08:00`，
  终态 `FAILED 1:0`，耗时 `01:13:06`。精确异常来自首个 R1 pass 的 accuracy-parity gate：
  未舍入 `mAP@0.6=61.0869609029443100 pp`，冻结 reference 为 `61.14 pp`，绝对差
  `0.0530390970556900 pp`，严格超过 inclusive `0.05 pp`。代码使用未舍入百分点和
  `difference > tolerance` 判定，focused test 也覆盖 `0.05` 通过、`0.050001` 失败，故该终止
  符合冻结合同，不是单位或比较方向错误。八 pass 未完成；正式 result root 存在但为空，
  `profile.json`、`terminal_receipt.json`、predictions、`cost_samples.jsonl`、
  `power_trace.jsonl` 及全部成本/短动作/边界终态证据均不存在。该轮只能分类为 replay
  admission/protocol failure；不能解释局部日志为模型性能、准确率保持或效率证据，也不得自动
  放宽门槛、补造产物、resume 或 duplicate。下一动作是把完整事实与未决合同交给 fresh Pro
  独立裁决；其返回前不追加实验。

- 2026-08-28：exact ZoomToken Project 的唯一有效全新 Pro 终态复盘完成。浏览器可见模型为
  `GPT-5.6 Pro`，六个附件均以 attachment-only 方式提交，实际科研提交 1 次、follow-up 0。
  Pro 返回 `REVISE`：v002 正确执行冻结合同，但 reported-2dp `61.14` 表示
  `[61.135,61.145)`，观测 `61.0869609029443100` 到最近兼容 raw 值只有
  `0.0480390970556900 pp`，所以该比较科学上是 `indeterminate`，不能承担二元硬准入。v002
  永久关闭为效率证据，BPNS-R1 仅保留单种子准确率可行、效率未知的窄主张。唯一下一任务是
  `ZOOMTOKEN-BPNS-R1-IDENTITY-GATED-FULL-STACK-REPLAY-v003`：硬门只覆盖执行身份与测量
  完整性，历史精度改为非阻塞三态诊断，完整保存八 pass 证据并按每臂四 pass 中位数计算主要
  p50/能耗估计。角色合同 `KEEP`；Git push、远端写入和 Slurm/GPU 动作需人工授权。本条只摄取
  Pro 裁决和任务，不新增模型、性能或成本结果。

- 2026-08-28：用户明确授权 Codex 自主推进正常 Git、远端部署和 Slurm/GPU 实验步骤。v003
  Builder 从 `e9323448f6cd78b99bb3de53fd9ffb55f3676d65` 建立最小 clean descendant
  `8a59d655005b9030d8ea5dc17ee2620844cb587b`，仅修改 profiler、focused tests 和 v003
  launcher。修正后的历史两位小数 parity 只作三态诊断；执行身份与测量完整性 fail closed；八个
  pass 分别保存 prediction/evaluator/hash；主估计为每臂四 pass 中位数；prediction hash 变化只
  作诊断；短动作/边界停止条件使用四个 R1 pass 全部劣于四个 K100 pass 的范围完全分离。
  local 与 N16R4 focused tests 均 `21 passed`；fresh independent Critic `PASS`，fresh result-blind
  Evaluator `PRE_RUN_READY`。候选已推送并部署到独立 clean source root。结果盲 precheck job
  `1258524` 于 `g0063` `COMPLETED 0:0`，返回 `PRECHECK_READY`、411 MP4、211/792 population、
  exact commit 与 data/checkpoint/config/evaluator identity，且明确不读取 validation metrics、
  不训练或 resume。唯一正式 job `1258526`（`zt-bpns-v003-8a59d655`）于
  `2026-08-28 11:58:39 +08:00` 在 `g0063` 开始，资源为 gpu partition、1 GPU、5 CPU、8 小时。
  当前只形成实现、准入与运行状态证据；终态前不读取或解释任何 partial 性能、成本或边界数值。

- 2026-08-28：唯一正式 v003 job `1258526` 在 `g0063` 从 `11:58:39` 运行至
  `17:32:11 +08:00`，终态 `FAILED 1:0 / FAILED_PROTOCOL_INVALID`，耗时 `05:33:32`。
  它按冻结顺序完成八个 K100/R1 validation pass，并分别保存 prediction、SHA 与六项未舍入
  evaluator vector；arm 内四次结果完全一致。随后 profile 阶段构造短动作 evaluator 时，因配置
  缺少 registry 必需的 `type` 字段而抛出确定性 `KeyError`。focused tests 覆盖统计函数但没有执行
  真实 evaluator 构造路径。结果根没有 `profile.json`、`cost_samples.jsonl`、
  `power_trace.jsonl` 或延迟、显存、能耗、短动作、边界汇总，因此 measurement completeness
  硬门未通过，冻结的四-pass-median 成本主估计不可计算。该轮是 evaluator 接口/协议失败，不是
  模型失败或科学负结果；八个 accuracy vectors 仅作终态诊断。job 不恢复、不重跑、不补造产物；
  完整事实交给一次 fresh exact-Project Pro 独立裁决，在其返回前不创建 successor 或选线。

- 2026-08-28：v003 的 fresh exact-Project post-result Pro 复盘已审计完成。conversation
  `6a919f06-bc94-83ea-b3e6-dd07f22375ee` 使用浏览器可见 `GPT-5.6 Pro`、七个 attachment-only
  文件、实际提交 1 次、follow-up 0；角色合同 `KEEP`。Pro 将 v003 判为工程缺陷触发、协议无效、
  科学无方向，并给出 `CONTINUE_ONCE_WITH_DECOUPLED_COST_CLOSURE`。八个 raw evaluator vector
  只支持固定 seed/checkpoint 的准确率可行性诊断；R1−K100 为 Avg-mAP `+0.5353 pp`，mAP@0.6
  `-0.1042 pp`，其余阈值混合但接近，不能承担效率或边界主张。唯一 v004 从 `8a59d655…` 建立
  clean minimal descendant，逐 pass 先原子保存 raw cost/power/prediction SHA/receipt，八 pass 后
  才运行非计时短动作与边界诊断；主 p50 与 complete-pass energy 的四-pass 中位数比均须
  `<=0.95`。只允许一个新 Slurm job；若 raw acquisition 前再次协议失败，则不授权 v005 或更多
  BPNS replay，效率保持未知并转交 fresh Pro。

- 2026-08-29：用户确认粘贴的 `CONTINUE_COMPOSITE_PROBE` 是最新 Pro 指示，并要求
  不中断唯一 v004 成本作业的同时开始独立 `ZoomToken-R1-TAR32-FKV` Builder。新 probe
  从 exact base `2d945e64bdccd09ae2e2916524562e3f388c5a2a` 建立独立分支/工作树：R1
  连续 K64 支持不变，偶数 VideoMAE 层更新全 K64，奇数层用紧邻前一完整层
  attention column mean 每 tubelet 选 exact K32，只为 K32 计算 Query/output/MLP，但全
  K64 继续提供 Key/Value 上下文并执行现有 Adapter。它是 composition-first research
  probe，不是已成立论文方法或 novelty。v004 job `1260095` 的 candidate/source/result root/
  watcher 均保持冻结；两条证据链不混用 partial 结果，各自终态后返回 fresh Pro。

- 2026-08-29：`R1-TAR32-FKV` 已从 Builder 进入唯一正式训练。最小 clean/pushed
  candidate 为 `b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7`，分支
  `codex/zoomtoken-r1-tar32-fkv-v001`，相对 base `2d945e64…` 只修改冻结允许的
  backbone/config/launcher/focused-test/training-route allowlist 表面。N16R4 CPU focused
  suites 为 `32 passed, 1 skipped` 与独立 strict-R1 `9 passed`；fresh Critic `PASS`，fresh
  result-blind Evaluator `PRE_RUN_READY`。首次实际 pre-run job `1260162` 仅因 `/bin/sh`
  wrapper 在 Python/CUDA 前 exit 127，封存为操作失败；replacement `1260163` 在 `g0048`
  `COMPLETED 0:0`，真实 canonical batch 的 CUDA AMP forward/backward 有限，route ledger
  为 `[64,32]x6`、fallback/failure 为 0，未读取 accuracy 或 prediction。唯一正式
  seed-42、60-epoch training job `1260166`（`zt-r1-tar32-fkv-s42-e60-v001`）于
  `2026-08-29 02:29:38 +08:00` 在 `g0059` 开始，2 GPU、8 CPU、16 小时；source root
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_tar32_fkv_src_b0a1ca11`，result root
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_tar32_fkv_v001_seed42_20260830`。
  后端 watcher `j-ucv5ag` 只等待终态；运行期间不读取或解释 partial 性能。v004 job
  `1260095` 及其 watcher `j-sjvtib` 保持独立、未修改、未重排。

- 2026-08-29：BPNS-R1 v004 唯一正式 job `1260095` 在 `g0059` 从 `00:37:38` 运行至
  `06:10:03 +08:00`，终态 `COMPLETED 0:0`。八 pass 严格按
  `K100,R1,R1,K100,R1,K100,K100,R1` 完成，4/4 prediction SHA 每臂一致并匹配冻结锚点；
  6,336 cost rows、929,889 power rows、pass receipts、profile、terminal receipt、短动作和边界
  诊断均完整。按 raw rows 独立重算的 R1/K100 四-pass 中位数比为：p50 `0.9849289616`、
  gross energy `0.9350002508`、p95 `0.9780185512`、peak allocated/reserved memory
  `0.7512973880/0.6896551724`。能耗通过 5% 门，但 p50 只下降 1.51% 并失败；按冻结联合规则
  终态为 `STOP_BPNS_R1_EFFICIENCY_HEADLINE`。短动作与起止边界差异小且混合，不支持边界保护
  主张。K100 pass 3 存在 `2804.82 ms` 功耗采样间隙，coverage 仍完整且冻结协议无 gap 阈值；
  作为能耗不确定性披露，但不改变延迟失败且不授权重跑。终态包只进入一次 fresh Pro。

- 2026-08-29：v004 fresh exact-Project post-result Pro 复盘已审计完成。conversation
  `6a92c125-e2e4-83e9-87f8-3123c9287afc` 使用浏览器可见 `GPT-5.6 Pro`、七个
  attachment-only 文件、实际科研提交 1 次、follow-up 0；裁决 `PIVOT`，角色合同 `KEEP`。
  Pro 确认 `STOP_BPNS_R1_EFFICIENCY_HEADLINE`，将 BPNS-R1 冻结为空间支持可行性、局部
  GPU/显存/能耗归因和负系统结果；不授权 v005、重放、额外 seed 或边界保护主张。唯一原子任务为
  `ZOOMTOKEN-R1-TAR32-FKV-TERMINAL-VALIDATION-AND-K100-MATCHED-FULL-STACK-COST-CLOSURE-v001`：
  先绑定原始 TAR32 权威并只读验收 job `1260166`；有效模型输出条件下，模型代码零修改，只执行
	  一次 K100/TAR32 同 GPU 八-pass 成本闭环，功耗按 pass-local gap `<=100 ms` fail closed；任何
	  终态随后只进入一次 fresh Pro，不自动重跑或增加 seed/第三臂。

- 2026-08-29：`CPTC-vFinal-20260829` 已冻结 TAR32 的终态执行语义：当前成本对照若获准
  应为 `R1/FULL64` 与 `R1-TAR32-FKV`，K100 仅属于后继残差探针。training job
  `1260166` 的 exact candidate、epoch-59 EMA checkpoint 与终态身份有效，但没有官方 final
  validation；单一 Critic 因此给出 `PASS_WITH_BLOCKER` 并只授权一次 evaluation-only completion。
  唯一 evaluation-only submission job `1261121` 在 `g0067` 运行 4 秒后 `FAILED 2:0`，在模型、
  checkpoint、loader 和 evaluator 前被外部 launcher 拒绝。原因是 Builder 用顶层 regular-file
  查询统计视频，而 canonical 411 个 MP4 是递归目录内的有效软链接；只读重建为顶层 regular=0、
  recursive symlink=411、follow-links regular=411。结果根未创建，prediction/metric/训练/resume/
  参数更新均为 0；这是工程/协议 blocker，不是 TAR32 科学结果。最终路线明确“一次提交”，因此
  不静默重提；一行 `find -L` 修正与完整 blocker 先进入 fresh Project Pro 独立裁决，Residual
	  Probe 仍冻结。

- 2026-08-29：pre-model 清单 blocker 的 fresh exact-Project `GPT-5.6 Pro` 复盘已完成。
  conversation `6a92d5cb-ac50-83ea-8e0f-3ca229ce9ba7` 使用 9 个 attachment-only 文件、实际
  提交 1 次、follow-up 0；裁决 `REVISE_AND_CONTINUE`，角色合同 `REVISE`。Pro 区分 scheduler
  submission 与 scientific attempt：job `1261121` 没有进入科学执行，因此只授权一次
  `RPL1_EVALUATION_ONLY_COMPLETION`，其计数为 scheduler ordinal 2 / scientific-attempt ordinal 1，
  第三次提交禁止。若冻结准确率门通过，只形成 `ACCURACY_ADMITTED_PENDING_FRESH_PRO`；当前不授权
  成本或 `ZT-CPTC-RP-K100-v001`。

- 2026-08-29：TAR32 replacement 的一行 `find -L` launcher 修正通过 Builder 身份检查、独立
  Critic `PASS` 和结果盲 Evaluator `PRE_RUN_READY_REPLACEMENT`。唯一替代 job `1261142`
  （`zt-r1-tar32-eval-b0a1`）于 `21:11:36 +08:00` 提交、`21:11:38` 在 `g0067` 开始，
  使用 2 GPU / 8 CPU，只加载冻结 epoch-59 `state_dict_ema` 并运行 official validation；不训练、
  resume 或更新参数。FastCtx watcher `j-sdfhnd` 每 300 秒只检测终态，Codex 不读取 live/partial
  accuracy、prediction、route 或其他指标，也不创建第三提交、成本或 successor。

- 2026-08-29：唯一 TAR32 replacement job `1261142` 已在 `g0067` 终态 `COMPLETED 0:0`，运行
  `00:14:58`；candidate、epoch-59 EMA、canonical 211-video/792-window population、官方
  evaluator/Soft-NMS、prediction 与 receipt SHA 均匹配。官方未舍入 Average-mAP/mAP@0.6/
  mAP@0.7 为 `64.9811408/57.3707378/43.6691029`，低于冻结 R1 reference
  `69.07/61.14/46.57`。从冻结双臂 prediction 重建的短动作 mAP 下降 `3.3175919 pp` 并失败；
  start/end median normalized boundary-error 比 `1.0925926/1.0178819` 均通过。四项准确率/
  短动作门失败、两项边界门通过，因此有效终态为
  `STOP_R1_TAR32_FKV_EXACT_COMPOSITION`。该结果只否定当前 single-seed exact composition
  的 accuracy admission；没有测 latency、energy、memory、多 seed 或机制因果。第三提交、成本和
  residual successor 均保持冻结，完整终态包只进入一次 fresh Project Pro。

- 2026-08-30：TAR32 终态 fresh exact-Project Pro 复盘已完成。conversation
  `6a930db4-fb90-83ea-ae8b-16e5028b6a45` 使用浏览器可见 `GPT-5.6 Sol / Power=Pro
  (5 of 5)`、8 个 attachment-only 文件、实际科研提交 1 次、follow-up 0；裁决 `PIVOT`，
  角色合同 `KEEP`。Pro 接受该单种子准确率负结果并冻结
  `STOP_R1_TAR32_FKV_EXACT_COMPOSITION`，不授权 TAR32 成本、第三次评测、重训、附加 seed、
  原地 residual rescue 或 `ZT-CPTC-RP-K100-v001`。负结果只否定 R1/K64 与固定半更新的精确
  组合，不外推到全部 CPTC。唯一下一任务为 `ZT-CPTC-K100-TAR50-INTERACTION-FALSIFIER-001`：
  在 native K100 上补齐 `[K100,K50]x6` 交互归因单元，绑定 strict A-MoD capacity=1 job
  `1254040`，只允许一个 seed-42、60-epoch、final-EMA official-validation Slurm 提交；任何正式
  失败直接回 Pro，不 retry/resume/replacement，任何有效终态也必须先 fresh Pro，且不自动测成本。

- 2026-08-30：K100-TAR50 Builder 候选 `fac88624723aed08175a947025a7f1d8a2af3171`
  已 clean/pushed。候选只添加 task-specific reference alias、precheck/formal launcher 与 focused
  tests；模型路径仍是已验证的 strict A-MoD capacity=0.5。N16R4 focused suite `14 passed`，job
  `1254040` 的 checkpoint/prediction/reference-config SHA 与任务 alias config SHA 均在远端重算
  匹配；fresh Critic `PASS`、fresh result-blind Evaluator `PRE_RUN_READY`。首次非科学 precheck
  在 job 创建前被 Slurm `AssocMaxSubmitJobLimit` 拒绝，Job ID 不存在、formal submission 仍为
  `0/1`、result root 未创建、训练/评测未开始。共享提交槽由 machine-side `sleep 300` waiter
  `j-00wllb` 静默等待；不取消或修改无关作业，也不绕过 precheck。

- 2026-08-30：非科学 precheck job `1261670` 已 `COMPLETED 0:0 / PRECHECK_READY`，随后唯一
  formal job `1261680` 在 `g0087` 运行 `00:00:38` 后终态 `FAILED 1:0`。训练到 epoch 0、首个成功
  optimizer update 前即因 `successful update indexing requires a GeoRoute backbone` 退出；result root
  只有 launch/terminal receipt 与 training log，没有 checkpoint、official prediction/vector、短动作、
  边界或成本证据。该终态是 `ENGINEERING_OR_PROTOCOL_BLOCKER`，不触发六门，也不支持 family stop。
  同一终态审计还确认冻结 prose 的 full-800 K/V、per-tubelet K50 语义与未修改的 strict A-MoD
  实现不一致：继承 odd block 在全局 flattened top-400 selected tensor 上执行 attention，K/V 也只有
  selected-400。正式提交已用尽 `1/1`；两项问题原样交 fresh Project Pro，不自动修复或重跑。

- 2026-08-31：用户手工转交一份 Pro 风格 `PIVOT / KEEP` 响应，提出 `ZoomToken-RACER24` 与
  `ZOOMTOKEN-COMPOSITE-SPRINT-AGENTS-ORDER-v001`。该材料没有可绑定的 exact Project ID、conversation、
  nonce、浏览器模型/effort、附件、提交计数、Oracle transcript/meta 或 terminal receipt，因此按
  provenance-warning 的来源陈述摄取，而不冒充浏览器审计证据。其可执行提案是：保持 BPNS K64、在
  blocks `{4,6,8,10}` 做 per-tubelet 24/64 selected-Q/full-KV，parameter-free completion 后恢复 dense
  carrier，并让既有 Adapter处理全部 token。材料中“job `1258299` 仍运行”和“base 已有 full-KV helper”
  两项与权威终态/代码不符，已明确隔离。当前只允许同一 Builder 补交 Iteration-0 MCL；没有代码、
  microbenchmark、训练、成本或 contingency successor 授权，也没有新的北京时间 deadline。

- 2026-08-31：RACER24 Iteration-0 只读 `MINIMAL_CHANGE_PLAN` 已完成。代码核验确认现有
  `Attention.forward` 只能同源生成 Q/K/V，native packed/ragged 路径也是 selected-only self-attention；
  因而 selected-Q/full-KV 与未选 token completion 必须作为最小新机制实现，不能用已有 helper冒充。
  MCL 将文件面限制为 `vit_adapter.py`、一个 config、一个 focused test、一个 task-specific profiler和
  一个 microbenchmark launcher；matched real-shape gate 为至少 200 次、p50 至少 `1.08x`、peak memory
  不超过 dense control `5%`。当前仍无模型代码编辑或实验，等待来源绑定或明确的 Iteration-0 实施授权。

- 2026-08-31：用户明确授权后，RACER24 clean candidate
  `5ebaa74f611bb3a43c3042700a78b92a9e5e74fb` 完成 Builder、两轮确定性审查修正、fresh Critic
  `PASS`、N16R4 focused tests `16 passed` 与 result-blind Evaluator `PRE_RUN_READY`。首次 job
  `1262067` 只因 Slurm `/bin/sh` wrapper 缺少 `source/module/python` 在 1 秒内退出，没有 Python、模型或
  CUDA 执行；独立 Evaluator将其判为 pre-execution blocker，并准入一次显式 Bash replacement。

- 2026-08-31：唯一科学微基准 job `1262068` 在 `g0041` 完成 50 warmup 与每臂 200 次 real-shape
  matched block 测量。dense/RACER24 p50 为 `1.33477/5.34684 ms`，p95 为 `1.36554/5.51236 ms`；
  p50 speedup 仅 `0.24964x`。RACER24 peak allocated/reserved 为 dense 的 `1.98884/1.84615x`。
  `>=1.08x` speed 与 `<=1.05x` memory 门全部失败，故终态为
  `STOP_RACER24_ITERATION0_AND_RETURN_TO_PRO`。这是当前 exact block-path 实现的有效负结果，不是
  accuracy、full-stack TAD、energy 或整个方法家族结论；不训练、不调参、不救援、不打开后继候选。

- 2026-08-31：fresh exact-Project Pro conversation `6a94842b-1370-83ea-a13c-2cc492170597` 已终态。
  唯一 prompt 绑定了最新 GitHub branch 与 exact commit URL；model picker `Pro` 已验证，提交 `1`、
  follow-up `0`。Pro 裁决 `PIVOT / STOP_RACER24_ITERATION0 / KEEP`，把 RACER24 负结果降为带
  pre-push deployment deviation 披露的 decision-grade evidence，不重跑、不升级为论文 claim。
  浏览器上传日志为 6 个文件，而 Pro 文本声称读取 7 个并列出一个未在日志中的旧 BPNS 回执；该差异
  作为 provenance discrepancy 保存，不重提。

- 2026-08-31：Pro 独立下达唯一任务 `ZOOMTOKEN-GRIDFUSE32-L6-GATED-v001`。该候选保持 R1 K64、
  8 个 temporal tubelet 和 dense Adapter，blocks 6–11 用固定相邻 pair 把完整 Q/K/V/MLP 物理序列
  从 N512 缩到 N256，再将 block residual 广播回 dense carrier。G0/G1/G2 是单一门控链；前门失败
  即终态并 fresh Pro，不开启 rescue、第二候选、sweep 或额外 seed。

- 2026-08-31：GridFuse32-L6 最小实现以 clean/pushed candidate
  `0b734ab839973b2c945b012f066db8222d235bb9` 固定。GitHub branch 与 exact commit 均经 fresh fetch/
  ls-remote 核验。N16R4 精确 checkout 在独立进程中通过 GridFuse `9`、R1 regression `12`、
  strict-rectangle `8` 项测试；fresh Critic 为 `PASS`。初次结果盲 Evaluator 发现 G2 未绑定 G1 checkpoint
  lineage；最小修正加入 canonical path、SHA256、epoch59、`state_dict_ema` 四项绑定后，新的结果盲
  Evaluator 返回 `PRE_RUN_READY`。这只说明可进入非科学 precheck，不构成 G0 加速证据。

- 2026-08-31：非科学 precheck jobs `1262078/1262079` 都在测试前因计算节点无法解析 GitHub 而停止，
  未执行模型或 G0。最终 launcher-only 修正把 fresh-fetch 放在登录节点，并让 Slurm action 核验 clean
  exact HEAD 与 persistent remote-tracking ref 同时等于 reviewed SHA；模型、门槛和结果解释未改变。最终
  exact candidate 的 fresh Critic/Evaluator 为 `PASS/PRE_RUN_READY`。

- 2026-08-31：最终非科学 precheck job `1262089` 以 `COMPLETED 0:0 / PRECHECK_READY` 终态，
  launcher 内的 focused suite 再次为 `9 passed`。随后只提交一个正式 G0 job `1262090`
  (`zt-gf32-l6-g0`，1 GPU，4 CPU，2 小时)，提交时为 `PENDING/Priority`。本轮只测冻结的六层
  real-shape segment gate；G1/G2 未开放，运行中不读取或解释 partial timing/memory。

- 2026-08-31：正式 G0 job `1262090` 于 `g0030` 运行 `00:00:15` 后终态 `FAILED 2:0`。
  segment profiler 在 `build_detector` 构建 pre-processing pipeline 时因 `Rearrange` 未注册到 mmengine
  transform registry 而退出；任何 warmup、alternating timing 或 memory measurement 都未开始。
  `terminal_receipt.json` 已生成，`profile.json` 缺失，因而分类为
  `GRIDFUSE32_L6_G0_ENGINEERING_OR_PROTOCOL_BLOCKER`，不是效率负结果。正式 G0 提交数 `1/1` 已用尽，
  G1/G2 未开放；不静默补 import、不重跑，先向 fresh exact-Project Pro 交回完整终态。

- 2026-08-31：fresh exact-Project Pro conversation `6a9494ad-dab4-83ea-83f6-e9cc2fabc722` 以
  GPT-5.6 Pro、八个 attachment-only 文件、submission `1`、follow-up `0` 完整返回；prompt 明确绑定
  GitHub repository、branch 与 exact `0b734ab8…` commit。裁决为
  `REVISE / CONTINUE_ONCE_WITH_EXACT-CONSTRUCTION-WITNESSED_G0_REPLACEMENT`，角色合同 `REVISE`。
  旧任务与 job `1262090` 保持 scheduler ordinal 1 终态；新任务
  `ZOOMTOKEN-GRIDFUSE32-L6-G0-CONSTRUCTION-WITNESS-AND-RPL1-v001` 只允许在 production construction
  witness、fresh Critic 与 result-blind Evaluator 通过后提交一个 scheduler-2/scientific-1 replacement。
  无第三次提交；G1/G2 仍关闭。

- 2026-08-31：唯一 construction-witness candidate
  `b5993faaaa59be318557ca314697e38c4b39b6a1` 已 clean push，且后续 Pro 请求固定携带 repository、branch 与
  exact commit GitHub 链接。候选只修改获准的 profiler/launcher/test，N16R4 GridFuse/R1/strict-rectangle
  suites 为 `12/12/8 passed`。Slurm witness job `1262099` 在 `g0063` 以 `FAILED 2:0` 终态：canonical
  registry、真实 detector construction 和 checkpoint strict load 已通过，但第一段 dense real-shape dry ledger
  在 Adapter 中触发 `ragged Adapter temporal axis differs from pretrained Adapter`。未进入 timing、memory、
  prediction、metric、gate 或训练，因此没有科学结果；不再 repair/review/submit replacement，立即 fresh Pro。

- 2026-08-31：fresh exact-Project conversation `6a949bec-1334-83ea-b410-a47ecdd451f7` 以 verified
  `GPT-5.6 Pro`、八个 attachment-only 文件、submission `1`、follow-up `0` 完整返回；prompt 与 response
  均绑定 repository、branch 和 exact `b5993faa…` commit。裁决 `REVISE`，永久关闭错误的 8-tubelet
  segment-G0，并把 blocker 定性为 production 384-tubelet Adapter 与旧 witness shape 的协议构造不匹配，
  不是 GridFuse 科学失败。唯一任务变为一个无 replacement 的 production-full-window atomic G0：同一单次
  job 内完成 full-window construction/ledger、100 warmups、500 alternating measurements 和冻结 gate；任一
  blocker 或 gate failure 都永久停止 exact GridFuse32-L6，完整 pass 也只返回 fresh Pro，不自动训练。

- 2026-08-31：production-full-window atomic G0 最终 clean/pushed candidate 为
  `dde46aee17f10bf793e5407055fc7b3416d93205`，GitHub branch 与 exact commit 已绑定。实现只修改 Pro
  许可的 profiler、focused test 和 launcher，真实使用 `B1/T384/K64/N24576`、48 个 attention buckets
  和 temporal-384 Adapter；G1/G2、训练、full-stack 与独立 GPU witness 均关闭。N16R4 GridFuse/R1/
  strict-rectangle suites 为 `16/12/8 passed`，fresh Critic `PASS`，fresh result-blind Evaluator
  `PRE_RUN_READY_ATOMIC_FULLWINDOW_G0`。唯一 final scheduler-ordinal-2/scientific-measurement-1 job
  `1262108` 于 `2026-08-31T05:38:35+08:00` 在 `g0041` 启动，1 GPU、4 CPU、2 小时；终态前不读取
  或解释 partial timing/memory，且没有 replacement 或 ordinal 3。

- 2026-08-31：唯一 job `1262108` 在 `g0041` 终态 `FAILED 3:0`，但完整写出了 construction
  witness、`profile.json` 和 `terminal_receipt.json`；非零退出来自冻结 gate failure，而非执行或产物
  blocker。真实 `B1/T384/K64/N24576`、48 bucket、dense temporal-384 Adapter、100 warmups/arm
  和 500 alternating samples/arm 均完成。dense/candidate p50 为 `178.500099/314.885696 ms`，
  speedup `0.56687268x < 1.35x`；allocated/reserved memory ratio 均为 `1.0`。终态为
  `STOP_GRIDFUSE32_L6_EXACT_ROUTE_VALID_G0_NEGATIVE`：exact GridFuse32-L6 路线永久停止，不修复、不重跑、
  不开 G1/G2。最新实现审查身份固定为 GitHub commit `dde46aee…`，下一动作只有携带 repository、branch、
  exact commit 链接及完整终态证据的一次 fresh exact-Project Pro 裁决。
