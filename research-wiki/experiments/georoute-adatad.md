---
type: experiment
node_id: exp:georoute-adatad
title: "GeoRoute-AdaTAD native spatial routing"
stage: implemented
status: rendezvous_correctness_fix_implemented_pending_remote_validation
outcome: prior_p1r_infrastructure_invalid_replacement_not_yet_running
updated: 2026-07-28
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
