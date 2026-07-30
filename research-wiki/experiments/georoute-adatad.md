---
type: experiment
node_id: exp:georoute-adatad
title: "GeoRoute-AdaTAD native spatial routing"
stage: implemented
status: official_semantics_stability_v2_running
outcome: no_performance_result_waiting_v2_finalizer
updated: 2026-07-30
---

# GeoRoute-AdaTAD native spatial routing

## Question

At a fixed native VideoMAE token budget, does ROI-free NativeTokenSelect first
beat matched fixed, random, and geometry-side-channel controls at lower total
cost than dense, and only then does continuous geometry add further high-tIoU
benefit without extra total cost?

## Current evidence

- Local implementation exists for native `2 x 16 x 16` tubelet routing,
  ROI-only, free-token, hybrid, the fixed-lattice geometry-side-channel
  control, P0 CUDA checks, a result-blind P0/P1/P2/P3 dispatcher, theory, and
  external paper plotting/table tools.
- Pure Python contract, DAG, result-schema and paper-tool checks passed
  `20` tests on 2026-07-23.
- The local Windows Torch runtime fails while loading `c10.dll`; this is an
  environment failure, not CUDA evidence. The Torch-dependent routing tests
  and the only meaningful P0 verdict remain pending on N16R4 CUDA.
- The first N16R4 P0 submission created no Slurm jobs because the deployment
  wrapper requested one outer GPU with `96G`, above the site's `55G/GPU`
  outer-allocation rule. The model code did not execute and emitted no P0
  report. The deployment code now requests a site-compliant two-GPU outer
  allocation and the existing launcher still executes exactly one GPU, five
  CPUs, and `96G` in an inner Slurm step.
- The second N16R4 P0 attempt created jobs `1180859`--`1180861`, but all
  three failed before a P0 JSON after reaching real AdaTAD detector forward:
  `_gather_selected_native_tubelets` expanded `[B,T,K]` to eight rather than
  seven native-video dimensions. The failed namespace and logs are preserved
  as diagnostics. A separate scheduler defect also rejected the CPU-only P0
  finalizer because this site requires every batch job to declare a GPU. Both
  issues are now under a minimal code fix. The accompanying remote routing
  suite also exposed missing uniform/random role accounting and a roundoff
  residue in the ST hard-forward gate; both are patched and still await the
  corrected remote suite and CUDA P0. No metric, cost, official test, or P0
  pass claim resulted.
- The next corrected snapshot passed its full remote focused suite, but the
  scheduler admitted only two of the three P0 leaves. `1180874` (hybrid,
  straight-through) independently passed its one-step CUDA gate: exact
  `K=32`, one packed VideoMAE forward, real AdaTAD classification/regression
  backward, finite nonzero scout/router gradients, and 5.15 GB peak allocated
  memory. This is a leaf-level implementation fact, not a suite verdict. The
  all-token dense leaf `1180873` fail-closed before a JSON because its P0
  reference used `no_grad`, which may dispatch CUDA SDPA differently from the
  real training forward (`max_abs_error=6.5612793e-4`). The repair keeps the
  `1e-4` criterion unchanged, executes the reference with matching autograd
  dispatch, immediately detaches it, and records that mode in the P0
  contract. It still requires a fresh CUDA verification.
- No development metric, cost result, official-test record, paper claim, or
  A-MoD experiment exists.
- **Authoritative current P0 evidence:** the model-path commit
  [`4a9358d`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/4a9358d1fba4bde9aa7693a94f7e4dfc95d31ecc)
  produced three CUDA one-step reports in
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_adatad_4a9358d_p0_20260723_102943`:
  dense all-token parity Job `1180906`, score-function ROI Job `1180907`, and
  straight-through hybrid Job `1180927`, all `COMPLETED 0:0`.  The dense report
  uses matching autograd dispatch and passed its frozen combined numerical
  comparison; hybrid has exact unique `K=32`, one packed heavy VideoMAE forward,
  real AdaTAD classification/regression loss, and finite nonzero geometry and
  residual scout gradients.  The score-function report binds real detector
  losses to the geometry scout.
- The P0-only finalizer first failed closed as Job `1180963` before writing a
  report because its file-path Python entrypoint could not import the repository
  `tools` package.  It is preserved as diagnostic evidence.  The minimal module
  entrypoint repair
  [`c2a3c69`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/c2a3c69d9006b32b1ca18f4ce66222b59550c45f)
  passed remote shell/focused checks; replacement Job `1180966` completed
  `0:0` and sealed `p0_finalization.json` as `PASS_MECHANICAL_ONLY`, suite SHA
  `a6f8ea041345cdc400c7f8a4f478c037cb66c8cfd3c19edb09d454ff363ce0b1`.
  It did not dispatch P1.  This is not accuracy, total-cost, or paper evidence.
- The P1/P2/P3 activation repair is implemented locally: a new P1 bootstrap
  recomputes the sealed P0 suite before submitting the seven matched P1 cells,
  freezes P2/P3 successor policy in its receipt, and never replays P0.  The
  dispatcher now uses the valid hyphenated action names and an N16R4-compatible
  control allocation without an explicit `--mem` override.  Focused contract
  checks pass `18/18`.  This remains `implemented` until the fresh committed
  remote snapshot has passed its Linux checks and submitted P1.
- P1 bootstrap Job `1181007` reached its dispatcher, accepted only its first
  dense leaf (`1181008`), then hit the account's `AssocMaxSubmitJobLimit` on
  the second leaf. The partial namespace is invalid diagnostic evidence and
  will never be resumed. The admitted dense leaf reached epoch 0 and exposed
  a real-input-only boundary branch. P0's 160x160 synthetic input did not
  traverse it. The first repair flattened independent batch/time axes, but
  Job `1181047` then exposed that CUDA does not implement `replicate` padding
  for the uint8 decoded frames used by the actual path. The failed 21-second
  gate namespace is immutable diagnostic evidence. The replacement appends
  the final row and column with byte-preserving concatenation, preserving
  native pixels and the same bottom/right replication semantics without a
  resize or interpolation. Exact-value bottom-only and bottom-plus-right
  regressions plus a fresh 180x320 CUDA mechanical gate are required before a
  fresh P1 namespace may be submitted. There is still no P1 metric, cost
  evidence, P2/P3 training, official test, or paper claim.
- The three dead `DependencyNeverSatisfied` jobs `1180494`--`1180496` were
  cancelled without touching active work. P1 will be retried only after the
  native-padding gate passes and the account admits the entire P1 matrix and
  selector; P2/P3 remain result-gated descendants.
- The first post-gate bootstrap `1181177` found a second scheduler-only
  failure: its `--test-only` checks passed, but seven real leaves consumed the
  final account slot before the selector could be submitted. Jobs
  `1181187`--`1181193` were cancelled immediately; dense `1181187` ran for
  only 20 seconds and is invalid diagnostic evidence. The account limit is
  `MaxSubmitJobs=16`; at dispatch it had nine active submissions, whereas the
  complete P1 matrix requires eight additional slots. The dispatcher now
  checks this arithmetic before a P1 root is created and cancels already
  submitted leaves if a later real submission is rejected. Because the
  bootstrap itself also counts against the cap, P1 remains held until there
  are at most seven active submissions before the bootstrap is queued.
- On 2026-07-27 the account had three active submissions, so clean snapshot
  [`6a9bba62`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/6a9bba6222c18a468c3bd410edac89a4afdea189)
  passed the source/P0/asset/capacity preflight. Bootstrap Job `1196062`
  completed `0:0`, validated sealed P0 suite SHA
  `a6f8ea041345cdc400c7f8a4f478c037cb66c8cfd3c19edb09d454ff363ce0b1`,
  and atomically published `p1_bootstrap.json` and `p1_submission.json` under
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_adatad_6a9bba62_p1p3_20260727_222913`.
  The matched seed-3407 leaves are dense `1196071`, fixed lattice `1196072`,
  fixed lattice plus geometry `1196073`, random `1196074`, free TokenSelect
  `1196075`, ROI `1196076`, and hybrid `1196077`; all seven reached real
  Epoch 0 batches with no stderr or fatal-log match at the first audit.
  Selector `1196078` is pending with exact `afterok` dependencies on all seven
  leaves. P2/P3 and official test remain unopened. Partial training losses are
  heartbeat evidence only and authorize no performance, cost, or paper claim.
- On 2026-07-28 accounting showed that every leaf `1196071`--`1196077` had
  failed `1:0` after roughly 64--68 minutes. Each log ended in
  `PytorchStreamWriter failed writing file` while saving a checkpoint, followed
  by `unexpected pos`; this was common to dense and all sparse variants.
  The namespace occupied 63 GB because the frozen config saved an approximately
  0.63 GB model/optimizer/EMA checkpoint every epoch. At audit time the JuiceFS
  `/data` filesystem was 100% used with 3.1 GB free, while inodes remained 99%
  free. No P1 result JSON was published. Selector `1196078` therefore remains
  `DependencyNeverSatisfied`/pending and has emitted no selection receipt.
  This is storage/deployment failure, not evidence for or against dense, free,
  ROI, hybrid, or geometry. The namespace remains immutable; any replacement
  must use a new commit/namespace, preflight aggregate checkpoint headroom, and
  retain only final EMA or an explicitly bounded result-blind checkpoint set.
  P2/P3, official test, empirical support, and paper claims remain closed.
- A user-authorized conservative cleanup then validated checkpoints with CPU
  `torch.load`, requiring matching `epoch` plus `state_dict`,
  `state_dict_ema`, `optimizer`, and `scheduler`. It retained exactly one
  highest-loadable file per cell: dense `epoch_13`, fixed-lattice-geometry
  `epoch_13`, fixed-lattice `epoch_14`, free `epoch_15`, hybrid `epoch_14`,
  random `epoch_15`, and ROI `epoch_15`. It deleted 107 other
  `epoch_*.pth` files totaling 62,674,238,335 bytes (58.370 GiB), reducing the
  namespace from 63 GB to 4.2 GB without touching pretrained weights, P0,
  receipts, configs, or logs. These retained partial checkpoints are
  diagnostic recovery artifacts only; the failed namespace remains
  non-resumable and contains no P1 decision.
  A post-cleanup filesystem check reported 62 GB available on `/data`
  (98% used).
- A second user-authorized `/data/run01/sczc063/yuzibo` retention pass used a
  hash-bound dry-run manifest, validated the highest checkpoint in every one of
  48 remaining multi-epoch directories with CPU `torch.load`, and deleted 273
  lower epoch files totaling 155,107,184,454 bytes (144.455 GiB) plus 166
  matching metadata/temp companions. Its
  post-verification found zero multi-epoch directories among 278 still-existing
  inventoried checkpoint directories; `/data` then reported 205 GB available
  (92% used). The plan SHA-256 is
  `e60f00ce6783ac6b858f107fbf06a5aff5d423d7e48b2139fa2412d2beab5e06`.
  This storage operation does not change the failed P1 evidence status.

## Correctness replacement (2026-07-28)

- The exact-commit Pro review returned `HOLD_FOR_CORRECTNESS_FIX` and is
  archived with SHA-256
  `e71e1964b75c68c3b05467ba571112e2bd540afa2ce791f991c5cf68ee078600`.
  It found six blockers: replicated 180x320-to-192x320 support without
  validity, a dense-lattice Adapter under packed attention/MLP, learned
  geometry in `free`, unmatched logit versus uniform pooling, a
  branch-misaligned hybrid surrogate, and unbounded checkpoint/storage
  behavior.
- The replacement is locally `implemented`: floor-native 176x320 support and
  an explicit validity mask; mask-aware exact-K; coordinate-lineage packed
  VideoMAE Adapter; fixed full-frame/frozen geometry in `free`; common
  uniform-selected pooling; branch-aligned hybrid route gradients with summed
  temporal score likelihood; atomic final-only checkpointing; and
  same-commit aggregate storage preflight.
- P0R now measures and seals a same-commit storage profile and checks packed
  attention, MLP, and Adapter execution with zero dense Adapter calls. P1R
  retains the seven matched arms but removes A-MoD, FlashVID-inspired
  residuals, and all other post-selection innovations.
- The selector is hierarchical. NativeTokenSelect must beat fixed lattice,
  random, and fixed-lattice-plus-geometry on the frozen accuracy criterion and
  cost less than dense. Only then may hybrid geometry advance, and only by
  beating free, random, and the geometry side-channel without greater total
  cost. Otherwise Route B advances or learned routing stops.
- Deployment remains one shot: three Slurm P0R siblings gate a finalizer; a
  mechanical pass automatically submits all seven P1R arms concurrently; a
  result-blind `afterok(all)` selector alone may create P2/P3 descendants.
  Thus scheduling is parallel while scientific interpretation is ordered.
- The implementation is native-token evidence routing, not a sequential
  TokenSelect-then-source-crop forward and not a resized crop. The term
  “Geometry Zoom” remains disallowed unless geometry strictly adds after the
  native base and the later multi-seed, cost, diagnostic, and generalization
  evidence closes.
- Local non-tensor contracts and static checks pass; Windows cannot load the
  Torch runtime. Until the clean remote snapshot passes focused tensor tests
  and P0R, the replacement is not `tested`. No replacement P1R metric, cost
  result, empirical support, or paper claim exists.

## Correctness replacement deployment (2026-07-28)

- Clean GitHub snapshot
  `45f5cca2e6b003478327511e3f38c8871b77084f` was obtained on N16R4 through
  the frozen academic acceleration proxy. Full HEAD, origin tracking ref, and
  an empty worktree matched.
- In the official OpenTAD Conda environment, all focused GeoRoute tensor,
  contract, DAG, storage, checkpoint, and paper-tool tests passed `58/58`;
  required C3 regressions passed `20/20`.
- One-shot run root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_nativefirst_45f5cca2_p0p3_20260728_1630`
  passed aggregate preflight with 214,831,312,896 observed free bytes versus
  79,456,894,976 required bytes, including atomic-publication peak.
- P0R Jobs are dense parity `1199838`, hybrid straight-through `1199839`, and
  ROI score-function `1199840`; result-blind finalizer `1199841` has exact
  `afterok` dependencies. At the first audit all three leaves are
  `PENDING(Priority)` and the finalizer is `PENDING(Dependency)`. P1R has not
  yet been submitted. This advances the replacement only to
  `experiment_running`; it is not a P0R pass or performance evidence.
- All three P0R leaves subsequently completed `0:0` in 21--24 seconds and
  finalizer `1199841` completed `0:0`. Sealed suite SHA
  `a0073394c3f0f505679797a4c22afeefda1d32adea7b4615e2eec4bbeed35077`
  is `PASS_MECHANICAL_ONLY`: dense full-token numerical reference passed;
  every checked hybrid block recorded 12 packed attention, MLP, and
  coordinate-lineage Adapter calls with zero dense Adapter calls; exact
  `K=32`, one heavy VideoMAE forward, real detector backward, and finite
  required scout/router/adapter/detector gradients passed. P0R wrote zero
  checkpoints. The same-commit final-only checkpoint upper bound is
  869,336,982 bytes per cell plus the sealed auxiliary/safety allowance.
- The finalizer automatically submitted matched P1R dense `1199865`, fixed
  lattice `1199866`, fixed-lattice-plus-geometry `1199867`, random `1199868`,
  free NativeTokenSelect `1199869`, ROI `1199870`, and hybrid `1199871`, plus
  exact `afterok(all)` selector `1199872`. Initial state is
  `PENDING(Priority)` for all seven leaves and `PENDING(Dependency)` for the
  selector. P1R performance and cost remain absent.
- At the 2026-07-28 17:25 CST heartbeat, dense `1199865` and fixed lattice
  `1199866` had advanced to `RUNNING` and were both in Epoch 2; the other five
  leaves remained `PENDING(Priority)` and selector `1199872` remained
  `PENDING(Dependency)`. Both stderr files were empty, losses were finite, and
  no Traceback/OOM/checkpoint failure appeared. Dense recorded two recovered
  AMP overflow retries and fixed recorded three retry attempts, all below the
  hard-failure threshold and followed by successful updates. Their live memory
  readouts (15,226 MB dense and 5,331 MB fixed) are heartbeat diagnostics only,
  not the frozen P1R cost result. No final checkpoint, stage result, selection
  receipt, or P1R metric exists yet.
- At 17:55 CST, fixed-lattice-plus-geometry `1199867` and random `1199868`
  also advanced to `RUNNING`; dense/fixed/fixed-plus-geometry/random were at
  Epoch 10/9/3/2. Free, ROI, and hybrid remained `PENDING(Priority)` and the
  selector remained dependency-held. All four active stderr files were empty,
  losses were finite, and each arm had five recovered AMP retry attempts with
  no retry exhaustion or fatal signature. There was still no final checkpoint,
  stage result, selector receipt, or P1R metric.
- At 18:10 CST, the primary ROI-free NativeTokenSelect arm `1199869` advanced
  to `RUNNING`; dense/fixed/fixed-plus-geometry/random/free were at Epoch
  14/13/7/6/1. ROI and hybrid remained `PENDING(Priority)`. Active stderr files
  remained empty, losses finite, and recovered AMP retry counts were
  5/5/5/5/2 without exhaustion. No final checkpoint, stage result, selector
  receipt, accuracy metric, or frozen cost result existed.
- At the 18:43 CST audit, dense `1199865` and fixed lattice `1199866`
  completed `0:0` and each published one atomic `epoch_19.pth`, zero temporary
  files, a passing storage receipt, and a `PASS_DEVELOPMENT_ONLY` stage result
  bound to runtime commit `45f5cca2e6b003478327511e3f38c8871b77084f`.
  Dense reported Avg-mAP `13.90`, mAP@0.6 `11.83`, and mAP@0.7 `8.74`;
  fixed lattice reported `12.42`, `10.75`, and `7.17`. Their profiles remain
  development-only and explicitly disallow paper-grade end-to-end and paper
  claims. These completed arms are retained as diagnostic cell outputs, not a
  P1R decision.
- Primary free NativeTokenSelect `1199869` failed `1:0` in Epoch 8 without a
  checkpoint or stage result. This is a launch-isolation failure, not a model
  or numerical failure: fixed and free overlapped on node `g0043`; free logged
  an immediate localhost TCPStore bind collision on port `29400`; fixed logged
  `Training Over` at 18:36:35; and free then lost that C10d store with
  `Broken pipe`/`RendezvousConnectionError`. Free had four recovered AMP retry
  attempts, finite logged losses, and no OOM or non-finite loss/cost signature.
  Hybrid `1199871` also overlapped random `1199868` on `g0048` and logged the
  same port collision.
- At 19:10 CST, fixed-lattice-plus-geometry `1199867` and random `1199868`
  completed `0:0`. Each has exactly one atomic final checkpoint, zero temporary
  files, a passing storage receipt, and a development-only stage result.
  Fixed-plus-geometry reported mAP@0.3--0.7
  `16.74/15.56/13.34/10.40/7.09` and Avg-mAP `12.63`; random reported
  `16.71/15.13/13.28/10.76/7.53` and `12.68`. Relative to fixed lattice,
  the geometry side-channel diagnostic changed Avg-mAP by `+0.21` but
  mAP@0.6/0.7 by `-0.35/-0.08`; random changed them by
  `+0.26/+0.01/+0.36`. These are single-seed descriptive cell deltas only.
  Both sparse controls selected exactly 64 unique of 220 valid tokens per
  tubelet with zero duplicates, one heavy forward, 12 packed
  attention/MLP/Adapter calls, and zero dense Adapter calls.
- Their development-only model-and-postprocess profiles reported
  p50/p95/peak-allocated `999.16 ms/4475.58 ms/1817.76 MB` for
  fixed-plus-geometry and `1812.36 ms/4806.06 ms/1816.86 MB` for random.
  The evaluator is excluded, same-process loader wait is included, and no
  paper-grade full-stack or energy claim is allowed.
- Hybrid `1199871` then failed `1:0` in Epoch 6 with zero checkpoints, zero
  temporary files, and no stage result. Random logged `Training Over` at
  19:04:37; hybrid immediately lost the shared C10d store and terminated at
  19:04:46 with the same `Broken pipe`/`RendezvousConnectionError`. Its five
  AMP retry attempts had recovered, losses remained finite, and no OOM or
  non-finite loss/cost occurred.
- ROI-only `1199870` completed `0:0` at 19:57:27. It published one atomic
  final checkpoint, zero temporary files, a passing storage receipt, and a
  development-only stage result. It reported Avg-mAP `13.18` and mAP@0.3--0.7
  `16.66/15.64/13.37/11.28/8.95`. Its route selected exactly 64 unique of
  220 valid tokens per tubelet with zero duplicates, straight-through geometry,
  one heavy forward, 12 packed attention/MLP/Adapter calls, and zero dense
  Adapter calls. Its p50/p95/peak-allocated profile was
  `905.40 ms/4360.95 ms/1818.21 MB`. These jobs cannot rescue or manually
  complete the hierarchical decision.
- Selector `1199872` is now `PENDING(DependencyNeverSatisfied)` because
  `afterok:1199869` failed. It has emitted no selection receipt and cannot
  authorize P2/P3. The seven-arm P1R matrix is therefore scientifically
  invalid: it neither supports nor refutes NativeTokenSelect, and geometry
  remains unauthorized. The namespace is immutable and will not be resumed.
  A future replacement requires a fresh exact commit and namespace plus a
  per-leaf collision-free rendezvous endpoint, followed by the complete frozen
  gate and matrix; no such replacement is launched from this failed run.

## Final P1R diagnostic closure

All P1 leaves are terminal. Values below are development-only diagnostics at
seed 3407; deltas are percentage points relative to fixed lattice.

| Arm / Job | Terminal state | Avg-mAP | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 | Delta Avg / @0.6 / @0.7 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense `1199865` | completed | 13.90 | 17.89 | 16.45 | 14.59 | 11.83 | 8.74 | +1.48 / +1.08 / +1.57 |
| fixed `1199866` | completed | 12.42 | 16.33 | 14.90 | 12.98 | 10.75 | 7.17 | 0 / 0 / 0 |
| fixed + geometry `1199867` | completed | 12.63 | 16.74 | 15.56 | 13.34 | 10.40 | 7.09 | +0.21 / -0.35 / -0.08 |
| random `1199868` | completed | 12.68 | 16.71 | 15.13 | 13.28 | 10.76 | 7.53 | +0.26 / +0.01 / +0.36 |
| free NativeTokenSelect `1199869` | failed: rendezvous | -- | -- | -- | -- | -- | -- | unavailable |
| ROI-only `1199870` | completed | 13.18 | 16.66 | 15.64 | 13.37 | 11.28 | 8.95 | +0.76 / +0.53 / +1.78 |
| hybrid `1199871` | failed: rendezvous | -- | -- | -- | -- | -- | -- | unavailable |

ROI-only is descriptively `+0.50/+0.52/+1.42` over random and
`+0.55/+0.88/+1.86` over fixed-plus-geometry in
Avg-mAP/mAP@0.6/mAP@0.7, and is `-0.72/-0.55/+0.21` versus dense. It is a
geometry-only diagnostic, not the ROI-free native base and not a substitute
for free NativeTokenSelect.

The only available cost record is the non-paper-grade development
model-and-postprocess profile:

| Arm | K / valid | Route / estimator | Unique / duplicates | Heavy forwards | Packed Attn / MLP / Adapter | Dense Adapter | p50 / p95 ms | Peak MB |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 220 / 220 | dense / none | 220 / 0 | 1 | 12 / 12 / 12 | 0 | 1237.37 / 4350.93 | 2681.03 |
| fixed | 64 / 220 | uniform / none | 64 / 0 | 1 | 12 / 12 / 12 | 0 | 1203.75 / 5226.15 | 1816.40 |
| fixed + geometry | 64 / 220 | uniform + geometry side-channel / none | 64 / 0 | 1 | 12 / 12 / 12 | 0 | 999.16 / 4475.58 | 1817.76 |
| random | 64 / 220 | random / none | 64 / 0 | 1 | 12 / 12 / 12 | 0 | 1812.36 / 4806.06 | 1816.86 |
| free NativeTokenSelect | -- | failed before result | -- | -- | -- | -- | -- | -- |
| ROI-only | 64 / 220 | ROI / straight-through | 64 / 0 | 1 | 12 / 12 / 12 | 0 | 905.40 / 4360.95 | 1818.21 |
| hybrid | -- | failed before result | -- | -- | -- | -- | -- | -- |

The profile explicitly excludes the evaluator, includes same-process loader
wait, has no energy receipt, and sets paper-grade end-to-end permission false.
It is not the required complete decode-to-NMS cost, so no efficiency or Pareto
claim is available. Five completed arms each retain exactly one final
checkpoint and no temporary file; free and hybrid retain neither checkpoint
nor stage result. All five stage results bind commit `45f5cca2`, prohibit paper
claims and official-test opening, and contain no GT, teacher, oracle, manual
ROI, or raw-prediction-cache use.

Selector `1199872` remains `DependencyNeverSatisfied`; no selection receipt,
P2/P3 artifact, or official-test artifact exists. Therefore:

- NativeTokenSelect: `NO_SCIENTIFIC_VERDICT_INFRASTRUCTURE_INVALID`.
- Conditional geometry: `NOT_AUTHORIZED_NATIVE_BASE_MISSING`.
- P2/P3: not authorized and not launched.
- Paper boundary: implementation/P0 mechanical facts and descriptive control
  cells only; no learned-routing, Geometry Zoom, efficiency, generalization,
  or paper-ready result.

## Rendezvous correctness replacement

- The replacement removes implicit `torch.distributed.run --standalone` from
  every development train/test leaf. Each launch now uses c10d,
  `127.0.0.1:0`, and a rendezvous ID bound to Slurm job, stage, variant, seed,
  and train/test phase. Stage-result schema v3 hashes both launch receipts.
- Every P0R leaf now runs two simultaneous one-rank Gloo probes before its
  model gate. Both probes must reach ready state, expose distinct actual
  `MASTER_PORT` values and exact `TORCHELASTIC_RUN_ID` values, and the long
  probe must remain alive after the short probe exits. The gate executes no
  model forward and opens no dataset or official test.
- The P0 CUDA report and rendezvous receipt are same-leaf bound by Slurm job
  ID, canonical path, file SHA-256, gate SHA-256, runtime commit, and
  non-symlink policy. The P0 finalizer requires three distinct bound Slurm
  receipts before it may emit `PASS_MECHANICAL_ONLY`.
- Local compile, DAG, P0 contract, checkpoint, paper-tool, storage, and
  required C3 non-Torch checks pass `59/59`. The complete local Torch suite is
  blocked by the documented Windows `c10.dll` failure, so Linux Torch,
  concurrent-rendezvous, CUDA P0R, and all scientific evidence remain pending
  on a clean N16R4 snapshot.
- This is an infrastructure-only correction. The model, seven P1R arms,
  selector, seeds, budgets, data, initialization, update schedule, and
  hierarchical decision rule are unchanged. The failed `45f5cca2` namespace
  remains immutable and cannot be pooled with the replacement.
- Clean source `a2ebd0604b4e5648b4f9bc4b3432541fae070393` passed remote
  Linux GeoRoute plus required C3 tests `82/82`. Run root
  `georoute_nativefirst_a2ebd060_p0p3_20260728_2202` submitted P0R Jobs
  `1200510`--`1200512` and finalizer `1200513`; Slurm intentionally colocated
  all three leaves on `g0003`. All leaves failed before a model forward because
  the first isolation gate used fixed 0.5/2.0-second worker lifetimes. Torchrun
  parent shutdown overhead allowed the nominal long worker to finish before
  the short parent returned, making
  `long_probe_alive_after_short_exit=False` without establishing a store
  collision. No P0 model report, checkpoint, stage result, selector, P1/P2/P3,
  or official-test artifact exists in that immutable namespace.
- The local replacement removes that timing assumption. Both probes now enter
  their independent stores concurrently; the long worker blocks on a
  `short.exited` marker that the gate writes only after the short torchrun
  parent has fully exited. The long parent must still be alive, observe that
  marker, retain a distinct observed `MASTER_PORT`, and finish successfully.
  This deterministic peer-exit handshake remains locally `implemented` until a
  new clean source and namespace pass N16R4 P0R.
- Deterministic-handshake source
  `bfee57904b3919480ce56b72429314eda508bf8e` again passed remote Linux tests
  `82/82`. P0R `1200550`--`1200552` under
  `georoute_nativefirst_bfee5790_p0p3_20260728_2213` failed before model
  execution with `short runtime identity did not match torchrun`; finalizer
  `1200553` was dependency-unsatisfied. Read-only Slurm diagnostic `1200560`
  then directly observed `TORCHELASTIC_RUN_ID=diag-1200560`, dynamic
  `MASTER_PORT=57695`, and `MASTER_ADDR=g0024`. Thus endpoint and run ID were
  correct; Torch materialized the worker master address as the allocated node
  hostname rather than the literal rendezvous endpoint address. The validator
  is now locally changed to bind `MASTER_ADDR` to the probe's exact
  `socket.gethostname()` on that Slurm node. Before another P0 namespace is
  allowed, the complete gate will run alone in Slurm and must publish a valid
  receipt.

## Active exact-source replacement run

- Clean source `7be8363ea6e26b320bffafeb03f0e82d8b660779` matched full
  local/origin/remote HEAD with a clean worktree after first-request academic
  proxy sync, and remote Linux GeoRoute plus required C3 tests passed `82/82`.
- Gate-only Slurm Job `1200602` passed
  `PASS_CONCURRENT_RENDEZVOUS_ISOLATION` on `g0053`. It observed exact short and
  long run IDs, distinct actual ports `54013/34325`, and a live long worker
  after the short torchrun parent had fully exited. Receipt SHA-256 is
  `bbfd46f507e7da16e0880df0d78067a37659164aaef2158c802f81de112acec9`.
- P0R Jobs `1200611`--`1200613` all completed `0:0` under
  `georoute_nativefirst_7be8363e_p0p3_20260728_2222`. Each P0 model report
  passed and hash-bound its same-leaf isolation receipt. The recomputable suite
  at `control/p0_finalization.json` is `PASS_MECHANICAL_ONLY`, schema v3, with
  suite SHA-256
  `693034b276697e92ae915ea5f40cebdd5d01a76bad65f46e5639844654f210e9`.
  Aggregate storage preflight observed 210,744,381,440 free versus
  79,456,894,976 required bytes.
- Finalizer `1200614` failed after sealing P0 but before any P1 submission:
  obsolete dependency-dead jobs made the fail-before-first-submit capacity
  guard report `active=11, required_additional=8, MaxSubmitJobs=16`. This is a
  scheduler-admission failure, not a model, P0, or rendezvous failure. The P0
  namespace remains immutable.
- Only obsolete GeoRoute dependency holds `1199872`, `1200513`, and `1200553`
  were cancelled after path/job provenance checks; DUCA `1181289` and all RIME
  work were left untouched. Supported sealed-parent bootstrap `1200652`
  completed `0:0` and created fresh P1 root
  `georoute_nativefirst_7be8363e_p1p3_20260728_2225`.
- Dense `1200663`, fixed `1200664`, fixed-plus-geometry `1200665`, random
  `1200666`, free NativeTokenSelect `1200667`, and hybrid `1200669` completed
  `0:0`. Each has one atomic final `epoch_19.pth`, zero temporary files, a
  passing storage receipt, and a `PASS_DEVELOPMENT_ONLY` stage result bound to
  source `7be8363e`. No arm used route GT, teacher, oracle, manual ROI, official
  test, or a raw-prediction cache.
- ROI-only `1200668` completed all 20 training epochs and wrote exactly one
  final checkpoint, then failed `1:0` during development testing. Its DataLoader
  reached `mmaction` decord frame loading and raised
  `Unable to handle EOF ... DECORD_EOF_RETRY_MAX=10240` while retrieving the
  last video frames. It has no `result_detection.json` or `stage_result.json`
  and no temporary file. Storage preflight passed; there was no OOM, non-finite
  loss/cost, gradient skip, rendezvous error, or model failure. The root cause
  is classified as development data/video-decode I/O.
- Result-blind selector `1200670` is `DependencyNeverSatisfied` and emitted no
  selection receipt. No P2/P3 or official-test artifact exists. The namespace
  is preserved without resume or manual selection.

Available development-only metrics and profiles are:

| Arm | Avg-mAP | mAP@0.3 | @0.4 | @0.5 | @0.6 | @0.7 | p50 / p95 ms | Peak MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 13.90 | 17.89 | 16.45 | 14.59 | 11.83 | 8.74 | 1306.88 / 5265.10 | 2681.03 |
| fixed | 12.42 | 16.33 | 14.90 | 12.98 | 10.75 | 7.17 | 1472.61 / 4620.84 | 1816.40 |
| fixed + geometry | 12.63 | 16.74 | 15.56 | 13.34 | 10.40 | 7.09 | 1408.03 / 4412.79 | 1817.76 |
| random | 12.68 | 16.71 | 15.13 | 13.28 | 10.76 | 7.53 | 932.64 / 4458.63 | 1816.86 |
| free NativeTokenSelect | 10.03 | 13.90 | 12.49 | 10.68 | 7.80 | 5.27 | 1073.14 / 4388.70 | 1818.00 |
| ROI-only | failed during development decode | -- | -- | -- | -- | -- | -- | -- |
| hybrid | 13.23 | 16.98 | 15.40 | 13.63 | 11.35 | 8.81 | 1115.63 / 4411.67 | 1817.76 |

All successful sparse arms select exactly 64 unique tokens from 220 valid
tokens per tubelet, with zero duplicates, one heavy backbone forward, 12
packed attention/MLP/coordinate-lineage Adapter calls, and zero dense Adapter
calls. Dense uses all 220. Hybrid allocates 28 ROI, 28 residual, and 8 context
tokens. The profile is model-and-postprocess-only, includes same-process loader
wait, excludes the evaluator, has no energy measurement, and is not
paper-grade end-to-end cost.

Descriptively, free loses to fixed/random/fixed-plus-geometry by
`-2.39/-2.65/-2.60` Avg-mAP, `-2.95/-2.96/-2.60` at mAP@0.6, and
`-1.90/-2.26/-1.82` at mAP@0.7. It is also `-3.87/-4.03/-3.47` versus dense.
Thus the current free selector does not satisfy the preregistered native-base
accuracy gate even before considering paper-grade total cost. Hybrid is
`+3.20/+3.55/+3.54` over free and `+0.55/+0.59/+1.28` over random, but the
hierarchy forbids using hybrid to rescue a failed native base. Because ROI
failed and the frozen selector emitted no receipt, these are descriptive
diagnostics rather than a formal selector verdict. Final status is
`tested_p1_data_decode_failure_no_selector`, not `empirically_supported` or
`paper_ready`.

## 2026-07-29 Pro review adjudication and preexperiment

The user-provided CER-TAD Pro review has been archived and independently checked
against the exact implementation and experiment contract. Project verdict is
`ACCEPT_WITH_MAJOR_REVISION / READY_PREEXPERIMENT_ONLY`.

Accepted findings are: Free v1 is residual-only with fixed full-frame geometry;
its ST amplitude surrogate gives no direct membership gradient to unselected
tokens; Hybrid uses deterministic quotas and has no hard-route likelihood; the
current geometry/coordinate adapter confounds support and representation; and
the current path is native-token routing rather than Geometry Zoom. The
single-family Gumbel-top-k / ordered Plackett-Luce likelihood and positive
risk-minimization policy-loss sign are mathematically coherent.

Not accepted for deployment are the full eleven-arm matrix, post-result
`+0.50/+0.30 pp` gates, dynamic role budget, critic, boundary head,
coverage/stability objectives, and immediate CER promotion. They are either
incompatible with the old selector, lack a variance/power basis, or remain
underspecified.

The new development-only preexperiment has three gates:

1. D: retrieve every exact development sliding-window item for two complete
   passes; any Decord failure stops all descendants.
2. K: pass frozen PL probability/sign/selected-and-unselected-gradient and
   support/representation isolation known-answer tests.
3. M: replay dense, fixed, fixed-plus-geometry, random, Free v1, and Hybrid v1
   under opt-in telemetry and reproduce every original prediction SHA-256.

ROI is excluded from parity because it has no source prediction. D/K/M outputs
are diagnostic only and cannot complete the old selector or authorize P2/P3.
The exact frozen design is
`docs/methods/2026-07-29-georoute-estimator-representation-preexperiment.md`.

The D/K/M preexperiment is now `tested_complete_go_pilot_design_only`.
Runtime commit
`0c20f2e89e6af8bac0e3612776e03f80c0a9f3fb`, Jobs
`1203105`--`1203113`, and finalization SHA
`78b0598c70c9966dfd4e7bfa0cce35cfe3ec7d00ed016d0c3268a214e36e86fc`
are sealed. D retrieved 272/272 items over two complete passes with zero
failure; K passed likelihood, risk-sign, ST/PL reachability, and representation
isolation checks; all six M replays preserved prediction SHA and a common
136-window population. No training, P2/P3, official test, or paper claim was
opened. Full diagnostics are in
`docs/methods/2026-07-29-georoute-estimator-preexperiment-results.md`.

The independent six-arm study
`georoute_estimator_representation_pilot_v1` is now
`experiment_running_after_passed_fresh_p0`. It
fixes K=64, seed 3407, 20 epochs,
absolute VideoMAE position on, and all other data/optimizer/runtime settings.
Its four preregistered contrasts are residual PL minus ST with representation
off, fixed representation on minus off, ROI representation on minus off, and
ROI minus residual support under PL/representation-off. Six P0 leaves and six
training leaves are parallel; the finalizer emits descriptive single-seed
contrasts without a winner or promotion. The old selector, P2/P3, and official
test are absent. Frozen protocol:
`docs/methods/2026-07-29-georoute-estimator-pilot.md`.

The first deployment of that protocol is sealed as infrastructure-invalid, not
as a pilot result. Exact clean runtime
`02b6efe71bd9c62de304467adf0981799eba6b1e` passed remote Linux tests
`108/108` and was deployed at
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_02b6efe7_20260729_1805`.
P0 Jobs `1203380`--`1203385` all ended `FAILED 1:0`. Jobs `1203380` and
`1203383`, each alone on its node, passed the concurrent rendezvous gate and
then failed before the CUDA model gate because script-mode execution had not
inserted the repository root before importing `tools`. The four co-located
leaves on `g0005` and `g0003` exceeded the old 30-second readiness bound; the
old gate discarded child output on this path, so this is classified as an
under-instrumented rendezvous-readiness failure rather than a model failure.
No P0 model report, training checkpoint, stage result, or metric was produced.
After all P0 leaves were terminal, impossible descendants
`1203386`--`1203392` were canceled without touching unrelated jobs. Finalizer
`1203393` ran and exposed a fourth mechanical defect: its validator compared
JSON mapping insertion order after `sort_keys=True` serialization. It therefore
failed before writing `pilot_finalization.json`. The immutable deployment file
hash is
`ee91fa7a5c58371f4d2c57b75038896e87b6a6a9c669492a4fb81c26554529a6`;
the absence of a finalization is explicitly retained as failure evidence.

The repair is `implemented_pending_remote_linux_and_fresh_p0`. P0 now launches
as a module and inserts the repository root before any dynamic import.
Rendezvous v4 encodes the decimal Slurm job into a unique audited `127/8`
address, retains a kernel-assigned port and unique cell/phase run ID, raises the
readiness bound to 120 seconds, and writes a hashed sidecar containing command,
return code, marker state, selected Slurm environment, and bounded child output
on failure while terminating the entire torchrun process group. It also
revalidates the gate node from the same P0 leaf. Two same-node Slurm capability
probes, Jobs `1203460/1203461`,
both showed that this site cannot reserve two step ports, so
`--resv-ports` is rejected rather than silently assumed. Deployment schema v2
normalizes exact arm-key sets independent of JSON order and rejects duplicate
job IDs. P0 finalization, fail-closed stage wrappers, and final closeout all use
`afterany`; a stage wrapper checks the sealed PASS P0 suite before creating a
cell, so a P0 failure reaches an auditable INCOMPLETE finalization without
launching training or inferring performance. P0-finalizer and final-closeout
prevalidation/sealing exceptions write hashed fail-safe receipts before
re-raising. This repair changes no frozen
model, arm, seed, K, epoch budget, estimator, representation switch, data,
contrast, or claim rule.

The repair is now remotely verified and deployed without changing the frozen
study. Exact runtime
`cbe0a08218a2f4550960f7c832f88c8cf77757c1` was synced through the academic
proxy to an exact clean snapshot and passed all GeoRoute plus required C3
remote Linux tests `118/118`. Same-node gate Jobs `1203689/1203690` ran
concurrently on `g0005` and completed `0:0`, using distinct job-scoped
loopback hosts and four distinct actual TCPStore ports. The fresh pilot root
is
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_cbe0a082_20260729_1849`.
P0 Jobs `1203707`--`1203712` and afterany P0 finalizer `1203713` completed
`0:0`; suite SHA
`00b7c0e3251f3d384df91cf900267694918d1245b4a5803150e8e2e1465210d2`
is `PASS_MECHANICAL_ONLY`. Six frozen leaves `1203714`--`1203719` started in
parallel. Residual-PL Job `1203715` hard-failed on real batch 0 after eight AMP
retries (scale `32768` to `256`) and produced no checkpoint or metric. The
other five leaves completed `0:0` only for terminal provenance. Afterany
closeout `1203720` completed `0:0` and sealed
`INCOMPLETE_EXPLORATORY_PILOT /
PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, `all_six_arms_passed=false`, an
empty contrast set, and false selector/P2/P3/official-test/paper-claim guards.
Its canonical self-hash is
`738e9875de2e9e08408263fd7d359e60f5ba1ca1912d0fbb9062a462c58cbf3a`;
file SHA-256 is
`63f73a353e356bc77a7a701972f22f62620b35e46b0c8f3eba0fc3c9816db0cc`.
The failure is
numerical correctness: a finite per-tubelet Plackett--Luce likelihood was
temporally reduced in FP16 over the real 384-tubelet horizon and overflowed
before GradScaler could recover. The local repair preserves the estimator and
sum-then-batch-mean objective while evaluating half/bfloat likelihood and its
temporal reduction in FP32. P0 schema v4 now requires an AMP backward KAT bound
to the actual `180x320`, floor-native `11x20` grid (`N=220`) and `K=64`.
Repair commit `30f9ca6fff1572e2eabc6c1b6636c4cc23595a62` is proxy-synced to
an exact clean snapshot and passed the complete remote Linux suite `120/120`.
Standalone CUDA KAT Job `1203873` completed `0:0` on `g0051`; the AMP horizon
check recorded FP16 source, FP32 likelihood/loss, objective
`128637.0234375 > 65504`, and finite scaled gradients. Its receipt internal/file
SHA-256 values are
`7d0ccc346b95180d02a5ddcf4253ac0278e83f39a6f7e434357c86067e3c8e84`
/
`75ef280473f5032fd734fb86f1f58207702c1999d34c5c7132d40ff5017ae4a4`.
This state is `old_namespace_sealed_incomplete_pending_capacity_and_fresh_p0`;
it is not a result, winner, P2/P3, official test, efficiency claim, Geometry
Zoom claim, or paper claim.

An independent history-free agent subsequently returned
`DEPLOY_AFTER_OLD_CLOSEOUT_AND_CAPACITY` after a read-only audit of the raw Pro
review, project absorption, frozen six-arm intervention, numerical repair,
no-leak contracts, and fail-closed 14-job DAG. This is protocol-review evidence
only. It authorizes no partial submission and does not replace the old
INCOMPLETE closeout or the new source's six expanded P0 leaves. The old
closeout gate is now satisfied; the capacity and fresh-P0 gates remain.

Both remaining gates subsequently passed without changing the study. At
deployment, two unrelated jobs were active and the complete 14-job DAG exactly
filled `MaxSubmitJobs=16`. Exact clean source `30f9ca6f` created
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_30f9ca6f_20260729_2023`.
P0 Jobs `1204015`--`1204020` and finalizer `1204021` completed `0:0`; sealed
suite self-hash
`2aea448be4c8d72957b3c904bb22c5ae39689cb0010c3b18a4914bd71f5265ec`
is `PASS_MECHANICAL_ONLY`. The repaired PL arm passed schema-v4 production AMP
evidence at `180x320`, `N=220`, `T=384`, `K=64`, FP32 likelihood/loss and
finite scaled gradients. Six frozen stages `1204022`--`1204027` now run
concurrently, with afterany closeout `1204028`. State is
`experiment_running`; no result or promotion surface is open.

That P0 coverage was still insufficient. Residual-PL stage `1204023`
hard-failed on real batch 0 after eight AMP retries, including the scale-256
attempt, and wrote only a self-hash-valid failure receipt
`f70b0a541cbfbbcf6595e8dfe7d7ef46ce16426d09a5ff7bc9fc921273c9eb81`.
A single fresh independent agent found that the P0 full model ran without
autocast or GradScaler and the subsequent AMP KAT was disconnected from model
parameters. Verdict is `HOLD -> REPAIR`, not estimator rejection. The next
numerical source must keep the scientific matrix fixed, force the complete
score-function route/scout precision path out of FP16 overflow, and gate a real
full-model scaled optimizer update. The five surviving stages completed only
for terminal provenance. Closeout `1204028` completed `0:0` and sealed
`PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, empty contrasts, and all promotion
guards false. Its self/file SHA-256 values are
`60c9dab575e65830b7b849437963de2c7f789743caedb130b499c142c49c76ab`
/
`6ad32b7822042685b353f378c6eb9ea14be061e7f22d7db7288a129cbe080f06`.

The estimator-equivalent repair is now numerically `tested` at exact source
`c822add335c38a9f6c63e609237c4bfa9b9f468d`. Its exact clean Linux snapshot
passed `121/121`. Standalone CUDA P0 Job `1204087` completed `0:0`; schema-v5
report self-hash
`4a9cd451e59417b6e606e841bcda47ebd5dd9b8b4c45ee3cfd42c0e0922d88aa`
records the real detector-plus-score-function graph under FP16 autocast,
GradScaler `256 -> 256`, finite required gradients, a successful zero-LR
optimizer update, and FP32 scout execution. The subordinate arithmetic evidence
remains `T384/N220/K64`, FP16 source, FP32 likelihood/loss and
`128637.0234375 > 65504`. It created zero checkpoints and no metric, test or
claim. No temporal normalization, clipping, weight, arm, seed, budget, epoch or
contrast changed. This passes only the replacement numerical gate. After the
old closeout sealed INCOMPLETE, capacity preflight admitted the complete
replacement DAG at exactly `active 2 + additional 14 = MaxSubmitJobs 16`.
Exact clean source `c822add3` created root
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_c822add3_20260729_2149`.
Deployment self/file SHA-256 values are
`7f445af550cf359b3d13d174f1199fa9b45192223461291f73f66a001c97b202`
/
`48f19fd88741361e9290ae6f444ccbd38f07030c82bc0c2726abad514de1158b`.
P0 Jobs `1204301`--`1204306`, P0 finalizer `1204307`, stage wrappers
`1204308`--`1204313`, and afterany closeout `1204314` were submitted.
All six P0 leaves plus finalizer passed `0:0`, and suite self-hash
`f6f423670c9c2417aadfca97c67d794427ee337c359ba2d2509faee53a5ccdb6`
records `PASS_MECHANICAL_ONLY`. Residual-PL stage `1204309` then hard-failed
on real batch 0 after all eight AMP retries down to scale `256`, with no
checkpoint, metric, or stage result. Failure receipt self/file SHA-256 values
are
`5e55619291285a36a8410be9582a79f293b33cd135053b4c6a3967e9d8beb5c8`
/
`77c9eff76c5881f8daa45ef68dd0d0d71bec2f5e171f31e2a8a97fc734b9f3c5`.
The other five stages and closeout `1204314` completed `0:0`; each surviving
stage has exactly one final epoch-19 checkpoint, one stage result, and zero
temporary checkpoints. ROI-PL representation-on/off Jobs `1204313/1204312`
each logged 11 cumulative failed optimizer attempts before successful replay,
with scale reaching `64`. This is numerical-stress telemetry, not a formal
failure: the source hard-fail rule is exhaustion of eight retries within one
batch, not cumulative count `>10` across training. Closeout `1204314` sealed
schema-v2 `INCOMPLETE_EXPLORATORY_PILOT /
PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, `all_six_arms_passed=false`, empty
contrasts, and all selector/P2/P3/official-test/paper guards false. Its
canonical self/file SHA-256 values are
`a02e551ba9007b49670103e2e4db3bf1c1d917cb5a7bb5c4dd724274b9379a2a`
/
`c95c1694dccbda2687b1b9e6e07bb9016ebe80181e2288d172874afa791d8f1c`.
Thus `1204309` is the sole formal stage hard failure, and the current synthetic
full-graph P0 is not a sufficient real-batch AMP stability certificate. This is
not a PL/ST or representation scientific verdict and opens no result surface,
rerun, selector, P2/P3, official test, Geometry Zoom, or claim.

The approved real-batch PL/ST cause analysis is implemented at exact source
`832caedd3713f477cb4b2f29a692acba9cd5a836`. It adds an opt-in observer to the
unchanged production train engine, a two-arm held-and-parallel stage DAG, and an
`afterany` numerical-only finalizer. Every path forbids checkpoints,
predictions, evaluator output, official test and performance inference. Local
protocol tests are `50/50` and required C3 regressions are `20/20`; clean
N16R4 Linux/CUDA validation and jobs are pending. This implementation does not
authorize a blind six-arm rerun.

The paper protocol is a separate future object. The current development recipe
differs from the official AdaTAD anchor in data population, batch size,
warmup/optimization surface, validation/checkpoint workflow and
evaluator/NMS. No diagnostic or 20-epoch pilot number is officially comparable.
Only an exact official reproduction plus same-recipe native-source dense and
candidate arms, disjoint seeds, sealed test, and full decode-to-NMS
latency/memory/energy may become paper evidence.

The first diagnostic deployment used exact clean source `832caedd`, whose
combined remote Linux/Torch suite passed `98/98`. PL `1204847` and ST `1204848`
then failed identically before config publication or model execution because
the binder called unsupported `mmengine.Config.__delitem__`. Finalizer
`1204849` sealed a self-hashed `DIAGNOSTIC_INCOMPLETE_NO_REPAIR` receipt with
no arms, metrics, checkpoints, predictions, evaluator or official test.
Minimal repair `64d991f96981a3e60b10f47d6d093d5457da9c60` uses `Config.pop` and
adds a real binder regression; local combined checks pass `71/71`. This
authorized only clean remote replay and a fresh diagnostic namespace.

That `64d991f9` namespace (Jobs `1204864/1204865/1204866`) was sealed
`DIAGNOSTIC_INCOMPLETE_NO_REPAIR` because the binder inverted
`SlidingWindowDataset.block_list` semantics before observer execution. Exact
source `047f643f4f78f5a954364d4f9b8e694c93f16079` corrected the receipt to
Fit-train/Gate-development while retaining Gate/Fit as the corresponding
exclusion lists and passed the remote suite `149/149`. Jobs
`1204908/1204909` then reached one identical real batch with matching
data/CPU/CUDA RNG hashes and finite forward losses, but both stopped before
backward because strict deterministic error mode rejected
`upsample_bilinear2d_backward_out_cuda`. Finalizer `1204910` again sealed
`DIAGNOSTIC_INCOMPLETE_NO_REPAIR` (self/file SHA-256
`7755f777d4dbecb3c5024100f0752c3147dc70f81a4d099ba9e77ece6ae6deac`
/
`1aea037cda2504f3a4a3a7c57d2628c7242829189722a8c8d1e78a0af838c19f`).
The failed parent pilot used the ordinary non-formal train path and therefore
deterministic warn-only mode. Source
`861e9b1edba5baf1b96fe0d4ed1c3c08d1e2da58` bound that same seed policy and
completed the matched diagnosis: PL failed nine loss scales through `256` only
in `scout_score_function` and first succeeded at `128`, while ST succeeded at
`65536`. Finalizer `1204946` authorized only a localized numerical repair.

Source `768e1a30` implemented a per-tubelet temporal mean for the PL
score-function loss; source `86ff1dde6ddb058ca9250f968972c255f19dab92`
fail-closed all diagnostic-parent and input bindings and passed clean remote
GeoRoute checks `124/124`. The corresponding stability-v1 Jobs
`1205033/1205034/1205035` were sealed
`INCOMPLETE_REAL_DATA_AMP_STABILITY_GATE / STABILITY_GATE_INCOMPLETE_HOLD`.
PL passed two batches at scale `65536` before two nonfinite
`residual_head.weight` gradients; ST passed twenty batches before detector-head
nonfinite gradients at the same scale. Both forwards remained finite and no
performance artifact was created. Because this gate required zero skips whereas
the exact official AdaTAD path uses dynamic GradScaler, it is not an official
comparability predicate and cannot freeze a paper protocol. No PL/ST, accuracy,
efficiency, or paper conclusion exists yet.

The separately versioned official-transition-semantics v2 is now terminal
`tested` at exact runtime source
`27fba03cb6d4932ee10cb4545b97984dff28c28c`. Its clean remote Linux/Torch
suite passed `168/168`; PL/ST Jobs `1205588/1205589` ran in
`georoute_official_semantics_amp_stability_v2_27fba03c_20260730_0800`, with
afterany finalizer `1205590`. It binds the sealed v1 HOLD and official reference
hash but explicitly does not match official scheduler hyperparameters or the
full recipe. Therefore it remains a no-metric numerical gate and cannot enter a
paper comparison.

The v2 gate is sealed HOLD. PL consumed 64 batches, made 61 successful updates,
and skipped scaler updates at batches `11`, `20` and `29`, with the third
backoff reaching `8192`. This exceeds the registered two-skip bound and violates
the `16384` scale floor. ST consumed 64 batches, made 62 successful updates,
and its two skips at `20/29` ended at scale `16384`. Both had finite forwards
and stable final-16 tails, but these facts do not undo PL's registered failure.
Finalizer `1205590` completed with
`INCOMPLETE_OFFICIAL_SEMANTICS_AMP_STABILITY_V2 /
OFFICIAL_SEMANTICS_AMP_STABILITY_V2_HOLD`; no rerun, performance inference,
formal protocol freeze, or paper comparison is allowed.

The accepted next step is not a repair or a performance rerun. The new
`georoute_pl_gradient_decomposition_diagnostic_v1` is implemented as a
no-performance matched PL/ST mechanism diagnosis at seed `7367`. It instruments
the production scaled-backward/DDP FP16 path while leaving hard sampling,
ordered likelihood, policy loss, temperature `0.7`, `K=64`, baseline,
temporal mean, GradScaler transitions, and the authoritative communication hook
unchanged. A separate real CUDA/DDP KAT must pass before the two parallel
64-batch leaves and `afterany` finalizer may be submitted. The diagnosis can
only distinguish FP16-cast overflow, upstream score nonfinite, scout VJP
nonfinite, shared detector overflow, or ambiguity. Even a unique mechanism
class authorizes only one minimal repair followed by a new no-performance gate.
It cannot produce or authorize mAP, cost, official-test, Geometry Zoom, P2/P3,
or paper claims.

## Frozen decision logic

P0R proves only implementation facts. P1R first tests whether ROI-free
NativeTokenSelect survives fixed, random, geometry-side-channel, and dense-cost
controls. A failed native base stops learned routing rather than being rescued
by extra geometry. Only after the native base passes can corrected hybrid
geometry be tested as an add-on. P2 and P3 are result-blind descendants; they
cannot start full training until their parent decision receipt authorizes them.

## Evidence outputs

- Theory: `docs/methods/georoute_adatad_theory.md`
- Figure/claim plan: `docs/methods/georoute_adatad_paper_evidence_and_figures.md`
- Runtime P0: `tools/bata/run_georoute_p0_gate.py`
- Result schema/analysis: `tools/bata/georoute_result_schema.py` and
  `tools/bata/analyze_georoute_results.py`
- External figures/tables: `tools/bata/plot_georoute_paper.py` and
  `tools/bata/render_georoute_paper_tables.py`
