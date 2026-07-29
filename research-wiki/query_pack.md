---
type: query_pack
updated: 2026-07-30
max_chars: 8000
---

# Research Query Pack

## Current Active Route: NativeTokenSelect-first GeoRoute-AdaTAD (2026-07-28)

- 2026-07-29 Pro-review absorption verdict:
  `ACCEPT_WITH_MAJOR_REVISION / READY_PREEXPERIMENT_ONLY`. Free
  NativeTokenSelect v1 is closed as the primary candidate on strong descriptive
  negative evidence; this does not manufacture the missing old selector receipt.
  Hybrid v1, the failed namespace, and the old Free-first selector are not
  promoted or reused. The current implementation is source-native token routing,
  not Geometry Zoom. The proposed CER-TAD context/geometry/residual model remains
  `discussed`: dynamic role allocation, critic, boundary head, coverage/stability
  objectives, and the review-proposed eleven-arm matrix are not implementation
  ready. The immediate independent preexperiment is only (D) a two-pass
  full-development exact-index decode census, (K) numerical PL/ST and
  representation-isolation KATs, and (M) prediction-SHA-preserving diagnostic
  replay of the six old arms that have both a final checkpoint and prediction.
  ROI lacks a source prediction and is excluded from parity. Any decode, KAT,
  prediction-hash, or population mismatch stops before training. The
  review-proposed `+0.50/+0.30 pp` margins are not accepted as confirmatory
  thresholds because they lack an independent variance/power basis. Full design:
  `docs/methods/2026-07-29-georoute-estimator-representation-preexperiment.md`.
  D/K/M is now `tested_complete_go_pilot_design_only` at runtime commit
  `0c20f2e89e6af8bac0e3612776e03f80c0a9f3fb`. Jobs
  `1203105`--`1203113` all completed `0:0`; the two-pass census retrieved
  `272/272` items, every KAT passed, and all six Phase-M leaves preserved
  prediction SHA and a common 136-window population. Sealed finalization
  `78b0598c70c9966dfd4e7bfa0cce35cfe3ec7d00ed016d0c3268a214e36e86fc`
  emitted `GO_PILOT_DESIGN_ONLY`, with training/P2/P3/official test/paper claim
  all false. Results:
  `docs/methods/2026-07-29-georoute-estimator-preexperiment-results.md`.
  A new independent study,
  `georoute_estimator_representation_pilot_v1`, is
  `c822add3_sealed_incomplete_no_performance_inference`.
  Six single-seed, 20-epoch, K=64 arms isolate PL versus ST, fixed-support
  representation, ROI-support representation, and ROI versus residual support.
  The first clean runtime `02b6efe7` passed remote Linux tests `108/108`, but
  P0 Jobs `1203380`--`1203385` all failed before a model result: the two
  single-tenant leaves passed the rendezvous gate and then exposed a script-mode
  `tools` import defect, while four co-located leaves exceeded the old
  30-second readiness threshold. No training leaf ran. Descendants
  `1203386`--`1203392` were canceled only after becoming impossible; finalizer
  `1203393` then exposed a JSON-key-order validation defect and produced no
  pilot finalization. This namespace is immutable deployment-failure evidence,
  not model evidence. The v2 repair launches P0 by module, installs the source
  root before dynamic imports, uses a decimal-Slurm-job-scoped `127/8` address
  plus a kernel-assigned port and unique run ID, raises readiness to 120 seconds
  with a hashed failure sidecar, terminates the entire torchrun process group on
  failure, validates the current node and arm maps independent of sorted JSON
  order, and changes all gating dependencies to `afterany` while stage wrappers
  fail before cell creation unless the sealed P0 suite passed. P0-finalizer and
  final-closeout prevalidation exceptions also produce hashed mechanical
  failure/INCOMPLETE receipts. Thus both success and failure paths reach a
  terminal non-promoting record. N16R4 Jobs
  `1203460/1203461` proved `srun --resv-ports=2` is unavailable
  (`Requires more ports than can be reserved`), so that mechanism is explicitly
  rejected. Fresh runtime
  `cbe0a08218a2f4550960f7c832f88c8cf77757c1` was proxy-synced with exact
  HEAD/origin/clean-tree parity and passed remote Linux tests `118/118`.
  Independent same-node Jobs `1203689/1203690` completed `0:0` concurrently
  on `g0005` with distinct job-scoped loopback hosts and four distinct actual
  TCPStore ports. The fresh pilot root is
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_cbe0a082_20260729_1849`.
  P0 Jobs `1203707`--`1203712` and finalizer `1203713` all completed `0:0`;
  sealed suite
  `00b7c0e3251f3d384df91cf900267694918d1245b4a5803150e8e2e1465210d2`
  is `PASS_MECHANICAL_ONLY`. Six leaves `1203714`--`1203719` started the
  frozen study in parallel. Residual-PL Job `1203715` hard-failed on real
  batch 0 after eight AMP retries from scale `32768` to `256`, with no
  checkpoint or metric. The other five leaves completed `0:0` only for terminal
  provenance. Afterany closeout `1203720` completed `0:0` and sealed schema-v2
  status `INCOMPLETE_EXPLORATORY_PILOT` with decision
  `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, `all_six_arms_passed=false`, an
  empty contrast set, and all selector/P2/P3/official-test/paper-claim guards
  false. Its canonical self-hash is
  `738e9875de2e9e08408263fd7d359e60f5ba1ca1912d0fbb9062a462c58cbf3a`;
  file SHA-256 is
  `63f73a353e356bc77a7a701972f22f62620b35e46b0c8f3eba0fc3c9816db0cc`.
  The four contrasts cannot be completed from this namespace.
  Root cause is a finite per-tubelet PL likelihood being temporally summed in
  FP16 over the 384-tubelet production horizon; the float64 `T=1` KAT and
  synthetic P0 missed it. A numerical-only FP32 likelihood/reduction repair
  and mandatory AMP-shaped `T=384/N=220/K=64` P0 KAT are now implemented
  in exact commit `30f9ca6fff1572e2eabc6c1b6636c4cc23595a62`.
  The validator binds this to the real decoded `180x320`, floor-native `11x20`
  source grid rather than a mismatched synthetic capacity. The commit was
  proxy-synced to an exact clean remote snapshot, passed the complete Linux
  suite `120/120`, and standalone CUDA KAT Job `1203873` completed `0:0`.
  Its objective magnitude was `128637.0234375 > 65504`, all scaled gradients
  were finite, and receipt internal/file SHA-256 values are
  `7d0ccc346b95180d02a5ddcf4253ac0278e83f39a6f7e434357c86067e3c8e84`
  /
  `75ef280473f5032fd734fb86f1f58207702c1999d34c5c7132d40ff5017ae4a4`.
  The numerical repair plus the sealed old closeout permits only an all-at-once
  capacity check and a future fresh full six-arm P0/run, not a result. There is
  no resume, partial-performance inference,
  old selector, automatic promotion, P2/P3, official test, efficiency claim,
  Geometry Zoom claim, or paper claim.
  A fresh history-free agent independently audited the Pro absorption,
  six-arm contract, numerical repair, no-leak paths, all-six finalizer, and
  14-job fail-closed deployer. Its verdict is
  `DEPLOY_AFTER_OLD_CLOSEOUT_AND_CAPACITY`: no further Pro discussion is needed.
  The old-closeout condition is now satisfied; Slurm must still admit all 14
  new jobs at once. This is code/protocol review evidence, not a model result.
  That capacity gate has now passed atomically: active jobs `2` plus all `14`
  new jobs exactly matched `MaxSubmitJobs=16`. Exact clean source
  `30f9ca6fff1572e2eabc6c1b6636c4cc23595a62` deployed a new namespace at
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_30f9ca6f_20260729_2023`.
  P0 Jobs `1204015`--`1204020` and afterany finalizer `1204021` completed
  `0:0`; suite
  `2aea448be4c8d72957b3c904bb22c5ae39689cb0010c3b18a4914bd71f5265ec`
  is self-hash-valid `PASS_MECHANICAL_ONLY`. The repaired score-function report
  is schema v4 and binds Job `1204016`, `180x320 -> 11x20/N=220`,
  `T=384/K=64`, FP16 source, FP32 likelihood/loss,
  `|objective|=128637.0234375 > 65504`, and finite scaled gradients. All six
  frozen stage Jobs `1204022`--`1204027` are running in parallel; afterany
  closeout is `1204028`. No selector, P2/P3, official test, or claim is open.
  The repaired residual-PL stage `1204023` nevertheless hard-failed on real
  batch 0 after eight AMP retries, again at floor scale `256`, with no
  checkpoint or metric. Its failure self/file SHA-256 values are
  `f70b0a541cbfbbcf6595e8dfe7d7ef46ce16426d09a5ff7bc9fc921273c9eb81`
  /
  `e36556f20a5cdbf138779fd46efc2f31462aaec19dc49296bb23fa94180edb5e`.
  A single fresh independent agent audited the discrepancy and returned
  `HOLD -> REPAIR`: the P0 model backward was FP32 without autocast/GradScaler,
  while its AMP KAT differentiated only isolated route logits. It therefore
  missed the scaled full graph through detector, scout, adapter, and backbone.
  The other five stages completed only for terminal provenance; no partial
  result is valid. Closeout Job `1204028` completed `0:0` and sealed
  `INCOMPLETE_EXPLORATORY_PILOT /
  PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, with `all_six_arms_passed=false`,
  an empty contrast set, and selector/P2/P3/official-test/paper guards false.
  Its self/file SHA-256 values are
  `60c9dab575e65830b7b849437963de2c7f789743caedb130b499c142c49c76ab`
  /
  `6ad32b7822042685b353f378c6eb9ea14be061e7f22d7db7288a129cbe080f06`.
  The next source must keep the six-arm science fixed, execute the
  score-function scout/likelihood path safely in FP32, and pass a full-model
  autocast+GradScaler optimizer-update P0 before any further replacement.
  Exact repair source
  `c822add335c38a9f6c63e609237c4bfa9b9f468d` is now `tested` for the
  numerical gate: backbone schema v4 keeps the complete scout/route forward and
  backward in FP32 outside autocast; P0 schema v5 runs the actual score-function
  model graph under FP16 autocast and GradScaler at floor scale `256`, unscales
  and audits required parameter gradients, then requires a successful
  zero-learning-rate optimizer step. The exact clean Linux snapshot passed
  `121/121`. Standalone Slurm Job `1204087` completed `0:0`; its self-hash-valid
  report records `T384/N220/K64`, FP16 source, FP32 likelihood/loss,
  `|objective|=128637.0234375 > 65504`, full-graph scale `256 -> 256`, finite
  required gradients, FP32 scout execution, a successful optimizer update and
  zero checkpoints. The estimator loss, weight, six arms and contrasts are
  unchanged. The old closeout and all-at-once capacity gates then passed.
  With `active=2`, the full 14-job DAG exactly filled `MaxSubmitJobs=16`.
  Exact source `c822add3` created fresh root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_c822add3_20260729_2149`.
  P0 Jobs `1204301`--`1204306`, P0 finalizer `1204307`, six stage wrappers
  `1204308`--`1204313`, and afterany closeout `1204314` were submitted
  atomically. All six P0 leaves and the finalizer completed `0:0`; sealed suite
  `f6f423670c9c2417aadfca97c67d794427ee337c359ba2d2509faee53a5ccdb6`
  is self-hash-valid `PASS_MECHANICAL_ONLY`. Residual-PL stage `1204309`
  nevertheless hard-failed on real batch 0 after exhausting all eight AMP
  retries from scale `32768` through `256`. Its failure receipt is
  self-hash-valid
  (`5e55619291285a36a8410be9582a79f293b33cd135053b4c6a3967e9d8beb5c8`;
  file
  `77c9eff76c5881f8daa45ef68dd0d0d71bec2f5e171f31e2a8a97fc734b9f3c5`)
  and the cell contains no checkpoint, metric, or stage result. The other five
  stages and closeout `1204314` completed `0:0`; each surviving stage has one
  final epoch-19 checkpoint, one stage result, and zero temporary checkpoints,
  but all are terminal provenance only. ROI-PL Jobs `1204312/1204313` each
  logged 11 cumulative failed AMP optimizer attempts before later successful
  updates, with scale reaching `64`. The earlier monitor classified cumulative
  count `>10` as a protocol hard failure; this is corrected. The source contract
  sets `max_amp_retries_per_batch=8`, and hard-fails only when one batch cannot
  produce a successful update after those retries. The cumulative counter is a
  numerical-stress alert, not a sealed experiment decision threshold.
  Closeout `1204314` sealed schema-v2
  `INCOMPLETE_EXPLORATORY_PILOT /
  PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, with
  `all_six_arms_passed=false`, empty contrasts, and all promotion guards false.
  Its canonical self/file SHA-256 values are
  `a02e551ba9007b49670103e2e4db3bf1c1d917cb5a7bb5c4dd724274b9379a2a`
  /
  `c95c1694dccbda2687b1b9e6e07bb9016ebe80181e2288d172874afa791d8f1c`.
  Thus `1204309` is the sole formal stage hard failure. The run shows that the
  current synthetic full-graph P0 is insufficient evidence of real-batch AMP
  stability; it does not yield a PL/ST, representation, efficiency, or Geometry
  Zoom verdict. No rerun, partial contrast, selector, P2/P3, official test, or
  claim is authorized.
  The approved next step is now implemented locally at exact source
  `832caedd3713f477cb4b2f29a692acba9cd5a836` as
  `exp:georoute-real-batch-amp-diagnostic-v1`: matched residual-PL/ST on the
  production data/training path, with input/RNG hashes and scaled/unscaled/
  clipped gradient localization. The old job lacks sample indices and RNG
  state, so this is not described as bitwise replay. It emits no metrics,
  checkpoint, prediction or test evidence. Only a localized cause can authorize
  a minimal repair and a fresh real-data stability gate. The observer is
  opt-in; parent pilot runtime/file hash, stage and wrapper-failure
  self-hashes, Slurm IDs and rendezvous are bound. Local pure checks pass
  `50/50`, required C3 regressions pass `20/20`, and Python/Bash/whitespace
  checks pass. Clean proxy-synced N16R4 Linux/CUDA validation and all three
  Slurm jobs are still pending, so no numerical diagnosis exists yet.
  The first exact-source attempt is now sealed as an infrastructure failure:
  clean `832caedd` passed the remote Linux/Torch suite `98/98`, but both PL
  `1204847` and ST `1204848` failed before observer/model execution because
  `mmengine.Config` has no `__delitem__`. Afterany finalizer `1204849` sealed
  `DIAGNOSTIC_INCOMPLETE_NO_REPAIR` with no performance artifacts. Minimal
  source `64d991f96981a3e60b10f47d6d093d5457da9c60` uses `Config.pop` and adds a
  real binder regression; local combined checks pass `71/71`. Its clean replay
  passed remote `99/99`, but PL/ST Jobs `1204864/1204865` stopped before the
  observer because the binding inverted exclusion-list semantics; finalizer
  `1204866` sealed no-repair. Corrected source `047f643f` passed remote
  `149/149`; Jobs `1204908/1204909` reached an identical real batch with matched
  input/CPU/CUDA RNG hashes and finite forward losses, then both stopped before
  backward because the diagnostic enabled strict deterministic error mode,
  unlike the historical pilot's warn-only policy. Finalizer `1204910` sealed
  `DIAGNOSTIC_INCOMPLETE_NO_REPAIR` (self-hash `7755f777...`). Exact candidate
  `861e9b1edba5baf1b96fe0d4ed1c3c08d1e2da58` restored and receipt-bound the
  historical deterministic warn-only seed policy. Its clean matched run
  `1204944/1204945/1204946` sealed
  `ROOT_CAUSE_LOCALIZED_REPAIR_AUTHORIZED`: PL failed nine scaled attempts from
  `65536` through `256` only in `scout_score_function` and first succeeded at
  `128`, whereas matched ST had zero failures and succeeded at `65536`; the
  batch/input/CPU/CUDA RNG hashes matched and no performance artifact existed.
  Source `768e1a30` then implemented an explicit per-tubelet temporal mean for
  the PL score-function loss, and exact source
  `86ff1dde6ddb058ca9250f968972c255f19dab92` fail-closed the real-data gate's
  parent/input bindings; clean remote GeoRoute checks passed `124/124`.
  Stability-v1 Jobs `1205033/1205034/1205035` are now sealed HOLD. Repaired PL
  passed its first two batches at `65536` but had two nonfinite
  `residual_head.weight` gradients on batch 3; ST passed twenty batches at
  `65536` but had detector-head nonfinite gradients on batch 21. Both therefore
  violated the preregistered 32-batch zero-skip rule. This is a failure of that
  numerical gate, not a performance or PL/ST verdict. The exact official AdaTAD
  path uses dynamic GradScaler and does not require zero skip, so stability-v1
  cannot be called an official-comparability criterion; an official-semantics,
  no-metric replacement gate must be separately frozen on an independent data
  order before any performance run.
  That replacement is now implemented and locally `tested` as
  `exp:georoute-real-data-amp-stability-v2`: seed/order `4417`, 64 batches,
  default dynamic GradScaler, zero retry/replay, official scheduler/EMA
  per-batch transitions, at most two nonconsecutive skips, scale floor
  `16384`, and a fully successful final-16 tail. Its binding explicitly records
  that only transition cadence is matched: scheduler hyperparameters and the
  full official recipe are not matched, so the gate is not performance
  comparable. Remote clean-Linux verification and the two-arm Slurm execution
  remain pending; it produces no metric, checkpoint, test or paper evidence.
  The single-seed 20-epoch pilot is not an official paper result. A future
  confirmatory study must include both an exact official AdaTAD reproduction
  and a matched native-source dense control, then match updates, effective batch
  size, EMA, evaluator/NMS, sealed-test policy and decode-to-NMS
  latency/memory/energy across disjoint seeds. The current GeoRoute development
  recipe uses a development-only population, batch size `1`, warmup `2`, no
  validation evaluator/NMS and final-only checkpointing, whereas the official
  AdaTAD anchor uses its official split/evaluator recipe, effective batch
  `2`, warmup `5`, and scheduled validation/checkpointing. Their numbers must
  never share a paper comparison row.
- Objective: first test whether detector-supervised, ROI-free exact-K selection
  of source-native VideoMAE tubelets protects high-tIoU offline TAD at lower
  measured total cost. Only after that base passes may continuous geometry be
  tested as a strict add-on.
- Active replacement status:
  `p1r_data_decode_failure_no_selector_exact_source_7be8363e`
  (`tested`, not `empirically_supported`). The external Pro audit of exact
  commit `df3e54e0c6776544dba20807b2ec100e1a399654` returned
  `HOLD_FOR_CORRECTNESS_FIX`. The local replacement now implements floor-native
  176x320 support with a validity mask, mask-aware exact-K, a
  coordinate-lineage packed Adapter, a truly geometry-free `free` control,
  common uniform-selected pooling, branch-aligned hybrid gradients,
  final-only atomic checkpoints, and same-commit aggregate storage preflight.
  Clean GitHub snapshot `45f5cca2e6b003478327511e3f38c8871b77084f`
  was synced through the frozen academic proxy; GeoRoute remote focused tests
  passed `58/58` and required C3 regressions passed `20/20`. P0R Jobs
  `1199838`--`1199840` plus afterok finalizer `1199841` were submitted under
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_nativefirst_45f5cca2_p0p3_20260728_1630`.
  Aggregate storage preflight passed with 214,831,312,896 free versus
  79,456,894,976 required bytes. P0R Jobs `1199838`--`1199840` and finalizer
  `1199841` all completed `0:0`; sealed suite
  `a0073394c3f0f505679797a4c22afeefda1d32adea7b4615e2eec4bbeed35077`
  is `PASS_MECHANICAL_ONLY`, with full-token parity, packed
  attention/MLP/Adapter execution, zero dense Adapter calls, real detector
  backward, route gradients, zero checkpoints, and same-commit storage
  measurement verified. It automatically submitted seven P1R leaves
  `1199865`--`1199871` and selector `1199872`. Dense `1199865` and fixed
  `1199866`, fixed-plus-geometry `1199867`, and random `1199868` completed
  `0:0`, each with one final checkpoint, zero temporary files, passing storage
  receipt, and a development-only stage result. Their diagnostic
  Avg-mAP/mAP@0.6/mAP@0.7 are respectively `13.90/11.83/8.74`,
  `12.42/10.75/7.17`, `12.63/10.40/7.09`, and `12.68/10.76/7.53`.
  Fixed-plus-geometry and random used exact unique `K=64` of 220, zero
  duplicates, one heavy forward, 12 packed attention/MLP/Adapter calls, and
  zero dense Adapter calls. Their profiles are development-only
  model-and-postprocess diagnostics, not paper-grade full-stack or energy
  evidence. Primary free NativeTokenSelect `1199869` failed `1:0` in Epoch 8
  without checkpoint or stage result because its implicit torchrun localhost
  port `29400` collided with fixed on shared node `g0043`. Hybrid `1199871`
  independently confirmed the same defect on `g0048`: random ended training
  at 19:04:37, then hybrid lost the shared store and failed `1:0` in Epoch 6
  with no checkpoint or stage result. Neither failure showed OOM or non-finite
  loss/cost. ROI-only `1199870` completed `0:0` with
  Avg-mAP/mAP@0.6/mAP@0.7 `13.18/11.28/8.95`, exact unique `K=64`, one heavy
  forward, and p50/p95/peak `905.40 ms/4360.95 ms/1818.21 MB`. All five
  completed cells have one final checkpoint, zero temporary files, passing
  storage receipts, and development-only stage results. The cost scope is
  model-and-postprocess only, excludes the evaluator, has no energy receipt,
  and is not paper-grade end-to-end evidence. Free and hybrid have no
  checkpoint or stage result. Selector `1199872` is
  `DependencyNeverSatisfied` and has no receipt; P2/P3 and official test are
  absent. Thus this P1R matrix is infrastructure-invalid, supplies no
  NativeTokenSelect or conditional-geometry verdict, and supports no
  efficiency or paper claim. The approved replacement now removes implicit
  standalone, binds train/test to kernel-assigned `127.0.0.1:0` endpoints and
  unique Slurm/cell/phase rendezvous IDs, adds a real same-node concurrent
  lifetime gate using observed `TORCHELASTIC_RUN_ID` and `MASTER_PORT`, and
  hash-binds each P0 model report to its same-leaf isolation receipt. Local
  non-Torch compile/focused/C3 checks pass `59/59`.
  Clean source `a2ebd0604b4e5648b4f9bc4b3432541fae070393` passed remote
  Linux tests `82/82`, but P0R `1200510`--`1200512` all failed before model
  execution because the first gate's fixed 0.5/2.0-second probe durations
  conflated torchrun parent teardown with store lifetime. Finalizer `1200513`
  was dependency-unsatisfied; no P0/P1 result exists. A deterministic
  replacement now keeps the long worker blocked until the controller observes
  complete short-parent exit and publishes a peer-exit marker; it still
  requires a fresh commit, namespace, and P0R.
  Deterministic source `bfee57904b3919480ce56b72429314eda508bf8e` also passed
  `82/82`, but P0R `1200550`--`1200552` failed before model execution because
  the gate required literal `MASTER_ADDR=127.0.0.1`. Slurm diagnostic `1200560`
  observed the correct dynamic port `57695` and run ID but
  `MASTER_ADDR=g0024`, the allocated node hostname. The local validator now
  binds master address to exact `socket.gethostname()`; a gate-only Slurm pass
  was mandatory before another P0 namespace.
  Exact clean source `7be8363ea6e26b320bffafeb03f0e82d8b660779`
  passed remote Linux tests `82/82`. Gate-only Job `1200602` then passed
  concurrent rendezvous isolation on `g0053` with exact run IDs, distinct
  dynamic ports `54013/34325`, and the long worker alive after complete
  short-parent exit. P0R Jobs `1200611`--`1200613` all completed `0:0`; their
  three same-leaf isolation receipts and CUDA reports sealed suite
  `693034b276697e92ae915ea5f40cebdd5d01a76bad65f46e5639844654f210e9`
  as `PASS_MECHANICAL_ONLY`. Finalizer `1200614` failed only after writing that
  receipt because obsolete dependency-dead jobs made the submit-cap preflight
  report `active=11, required_additional=8, MaxSubmitJobs=16`; no P1 job was
  partially submitted in that namespace. After cancelling only obsolete
  GeoRoute Jobs `1199872`, `1200513`, and `1200553` while leaving DUCA/RIME
  untouched, sealed-parent bootstrap `1200652` completed `0:0` into fresh root
  `georoute_nativefirst_7be8363e_p1p3_20260728_2225`. Dense `1200663`, fixed
  `1200664`, fixed-plus-geometry `1200665`, random `1200666`, free
  NativeTokenSelect `1200667`, and hybrid `1200669` completed `0:0`, each with
  one final checkpoint, zero temporary files, passing storage receipt, and a
  development-only stage result. Their Avg-mAP/mAP@0.6/mAP@0.7 are
  `13.90/11.83/8.74`, `12.42/10.75/7.17`, `12.63/10.40/7.09`,
  `12.68/10.76/7.53`, `10.03/7.80/5.27`, and `13.23/11.35/8.81`.
  Free is descriptively worse than fixed, random, and fixed-plus-geometry by
  `2.39/2.65/2.60` Avg-mAP and by `2.95/2.96/2.60` at mAP@0.6, so it does not
  satisfy the preregistered native-base accuracy condition. Hybrid's descriptive
  gain over free cannot authorize geometry because the native base did not
  pass. ROI-only `1200668` finished training and wrote its unique final
  checkpoint, then failed `1:0` during development testing when decord could
  not retrieve final video frames before `DECORD_EOF_RETRY_MAX=10240`; it has no
  prediction or stage result. This is a data/video-decode I/O failure, not OOM,
  non-finite loss/cost, storage, model, or rendezvous failure. Selector
  `1200670` is `DependencyNeverSatisfied` with no receipt. Therefore the matrix
  is protocol-incomplete and supplies no formal selector verdict; P2/P3 and
  official test remain closed. Available timing is model-and-postprocess-only,
  excludes evaluator and energy, and permits no paper-grade efficiency claim.
- Historical P1 status remains
  `failed_p1_infrastructure_storage_exhaustion_no_metric`. The sealed P0 parent from
  [`4a9358d`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/4a9358d1fba4bde9aa7693a94f7e4dfc95d31ecc)
  remains `PASS_MECHANICAL_ONLY`. Clean dispatcher snapshot
  [`6a9bba62`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/6a9bba6222c18a468c3bd410edac89a4afdea189)
  completed bootstrap Job `1196062` (`0:0`) and atomically submitted the seven
  matched P1 leaves `1196071`--`1196077`. All seven later failed `1:0` while
  publishing per-epoch checkpoints after `/data` reached 100% usage; the
  namespace had accumulated 63 GB. Result-blind selector `1196078` remains
  dependency-held and emitted no decision. This is immutable infrastructure
  failure evidence only: no P1 mAP, cost, A-MoD result, empirical support,
  official-test evidence, or paper claim exists.
- P0R contains three mechanical CUDA leaves. A one-shot Slurm dependency graph
  submits all seven P1R arms automatically and concurrently only after the P0R
  finalizer emits `PASS_MECHANICAL_ONLY`. Parallel scheduling does not alter the
  causal selector order.
- P1R is the first scientific screen: matched dense, fixed lattice,
  lattice-plus-geometry side-channel, random, ROI-free NativeTokenSelect
  (`free`), ROI-only, and corrected hybrid, all with uniform pooling and the
  packed Adapter. The native base must beat fixed, random, and the geometry
  side-channel while costing less than dense. Geometry is considered only
  afterward and must strictly improve on free, random, and the geometry
  side-channel without higher total cost. Otherwise Route B advances or learned
  routing stops. P2 promotes only the authorized route to seeds/budgets; P3 is
  frozen second-detector/dataset and sealed-test closure.
- The prior quota hold was cleared, but P1 is now storage-held. P0 replacement Gate `1181172` passed the
  real uint8 180x320 path; roots `1181007` and `1181177` remain immutable
  scheduler diagnostics only. The fresh namespace is
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_adatad_6a9bba62_p1p3_20260727_222913`.
  Its bootstrap and submission receipts bind the sealed P0 suite SHA
  `a6f8ea041345cdc400c7f8a4f478c037cb66c8cfd3c19edb09d454ff363ce0b1`.
  The replacement uses a new namespace, result-blind aggregate
  storage-capacity preflight, and one atomic final checkpoint per cell.
  The failed namespace was pruned conservatively on 2026-07-28: seven
  highest-loadable per-cell checkpoints were retained and 58.370 GiB of
  intermediate/corrupt epoch files were removed. This does not make the
  namespace resumable or create P1 evidence.
  A subsequent user-authorized root-wide retention pass validated the highest
  loadable checkpoint in 48 additional multi-epoch directories and deleted 273
  lower epoch files (144.455 GiB) plus 166 matching metadata/temp companions.
  Together the two passes removed 380 epoch checkpoint files
  (202.825 GiB); a bound post-verification found zero multi-epoch directories
  among the 278 still-existing inventoried checkpoint directories, and `/data`
  reported 205 GB available. Pretrained weights, `best.pth`, configs, logs, and
  single-checkpoint directories were not changed.
  P2/P3 remain absent and result-gated. The code is native-token evidence
  routing, not a sequential second crop/resized zoom; “Geometry Zoom” remains
  unauthorized unless the conditional geometry gate and later paper evidence
  close.
- FlashVID was audited as a VLLM reference, not a GeoRoute result. Its 10%
  retention result is 57.9/58.4 = 99.1% relative score after a full vision
  encoder, so it cannot support native-pre-backbone or detector-gradient
  claims. Its relevance-diversity-motion correspondence principle is only a
  conditional P2 scout-side comparator after a P1 hybrid win.

## Continuous-RoI / Native-Crop Record (Frozen or Held)

- The original spatial goal is source-coordinate, variable `(cx, cy, w, h)`
  crop tubes at native local pixel density while retaining the full temporal
  axis. Dense 160/224/256 resizing is R0 headroom control, not a crop result.
- Continuous-RoI S2 exact-nine training (`1177668`--`1177676`) is sealed as
  `PASS_TRAINING_ONLY`: 60 epochs, 4,800 successful updates per cell,
  final-EMA-only, and no official-test opening. It is neither crop
  sufficiency, cost, mAP, nor a learned-policy result.
- Its fixed/variable reference protocol is `HOLD`: common physical centers,
  Sobol generator identity, candidate-ID authority, no-GT raw entrypoint and
  privileged join/tie/statistics are not jointly frozen. Only a result-blind
  v2.2 corrigendum is allowed; no official test or S3 learned policy follows.
- Native-Crop S1 merely established source-native crop data/model/gradient and
  no-leak mechanics. The fixed 128 candidate library is D0 diagnostic only.
  Historical S1/R0 campaigns must not be resumed, combined, or presented as
  crop GO/KILL evidence.

## C3 / DUCA Historical Baselines and Negative Memory

- Project-wide target remains offline, task-aware redundant-computation
  removal with protected mAP@0.6/0.7 and full decode-to-NMS cost. It is not
  causal or Online TAD.
- DUCA is a frozen, unproven full-window candidate. Its honest contract is
  `offline_full_window + runtime_generated + cache_free + jointly_trained`.
  It uses a low-cost coarse probe, transition/boundary-sensitive selection,
  fixed-K positions and AdaTAD-derived components; it cannot be called an
  unmodified official AdaTAD plugin or a paper method before matched evidence.
- Do not revive these mistakes: actionness top-k as the final selector;
  post-hoc gap repair/uniform scaffolds that hide learning failure; old-commit
  mAP as current evidence; smoke/gradient checks as utility evidence; dense
  X3D as a low-cost main probe; dynamic MUST as a main contribution; or FLOPs
  without trained end-to-end cost.
- Known failure mechanisms remain valuable: actionness focuses action interiors
  rather than boundaries; complex coverage constraints can collapse toward
  uniform; GAS-VT train/apply mismatch and hard repair invalidated its main
  reading; selected-axis geometry can damage high-tIoU; and requested,
  effective, unique, padded and actual backbone budgets must be logged
  separately.

## Non-Negotiable Evidence Rules

1. Match commit, data, pretrained initialization, updates, effective batch
   size, AMP, EMA, seeds, token budget, detector/head, evaluator and effective
   NMS before comparing selection methods. Pair an exact official reproduction
   with a matched native-source dense control whenever preprocessing changes.
2. Report high-tIoU, short-action/boundary diagnostics and measured full-stack
   p50/p95 latency, memory and energy. FLOPs or random-init profiling alone
   cannot establish efficiency.
3. Training-only, smoke, precheck, pending or failed jobs never become
   empirical support. Test/validation GT, teacher signals and raw prediction
   caches must never participate in inference decisions.
4. Any route that fails its matched control narrows or dies instead of gaining
   extra selector heads or loss weights. Preserve failed evidence in the
   experiment record and `anti_repetition.md`.

## Pointers

- GeoRoute implementation and gates:
  `research-wiki/experiments/georoute-adatad.md`.
- Current GeoRoute hypothesis and decisions:
  `research-wiki/ideas/geo-route-adatad.md`.
- Native-Crop S2 hold:
  `research-wiki/experiments/native-crop-s2-crop-sufficiency.md`.
- Full historical source and decision record: `research-wiki/log.md`,
  `research-wiki/decision_history.md`, `research-wiki/anti_repetition.md`,
  and `research-wiki/source_registry.md`.
