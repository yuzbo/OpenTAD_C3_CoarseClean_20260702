---
type: experiment
node_id: exp:chronotransport-r2-implementation-verification
title: "ChronoTransport r2 implementation verification"
idea: idea:chronotransport
verdict: revise_implementation_before_registration
confidence: high
commit: "6c3606cc5161d415909a42741b3bc402278bf332"
review_snapshot: "1b6366d0acb712e8096c2cceb0f05e66b16d30d4"
jobs: "remote CPU focused pytest in workdirs/chronotransport_r2/repo"
updated: 2026-07-16
---

# ChronoTransport r2 Implementation Verification

## W3--W6 complete candidate verification (2026-07-16)

- Candidate `6c3606cc5161d415909a42741b3bc402278bf332` is explicitly review-only and is not I or R.
- W3 binds lexical no-follow inputs, same-descriptor bytes, imported-module origins and formal
  entrypoints. Gate-4 media is checked before and immediately after decode using the retained descriptor.
- W4 exposes real ActionFormer/AnchorFreeHead batch-two per-window task losses from the same logits,
  targets, normalizer and aggregate reduction.
- W5/W6 implement paired CT/matched-dense 4,200-success workflows, exact resume/checkpoint ledgers,
  post-Stage-C Gate 3 and formal official-population Gate 4 with immutable seed shards and finalizer.
- Gate 4 includes decode/preprocess/H2D/model/postprocess/full-video NMS, deferred CUDA events without a
  hidden mid-forward sync, official NumPy quicksort tie semantics, official-video/seed bootstraps,
  matched-invocation margin, all five registered per-seed thresholds and long-block 10-Hz NVML energy.
- All 18 changed/new implementation files were SHA-256-identical between the local candidate and remote
  verification checkout. Targeted Gate-4 regression was `32 passed in 53.56s`; final complete
  ChronoTransport CPU regression was `441 passed, 1 skipped, 2 warnings in 968.62s`. Remote py-compile and
  four-launcher `bash -n` exited zero; required C3 compatibility was `20 passed in 20.42s`.
- The one skip is the protected CUDA-only surface. Local pytest is not cited because local PyTorch DLL
  initialization fails with `WinError 1114`; local py-compile and `git diff --check` passed.

Overall verdict intentionally remains `revise_implementation_before_registration` until a fresh
GitHub-only exact-byte reviewer returns `APPROVE_IMPLEMENTATION_FOR_REGISTRATION`. Registration is
`NOT_READY`; no PRECHECK, Slurm GPU job, Gate, training or scientific result exists.

## Verified manifest/protocol repair slice

- Exact 200-video label-free manifest/deep re-derivation, canonical raw-byte and sidecar validation,
  duplicate-key rejection, Stage-B hash-bound exposure artifact, Stage-C balance/cursor validation,
  frozen control/library identities, and the legacy formal-runner hard lock are remotely `tested`.
- The TDD chain recorded the expected missing-symbol RED (1 collection error, 35.37 seconds), then a
  source-vector integrity RED where a registry-provided sampled index could be trusted, followed by
  strict-type/canonical-byte negative checks. After repair, the focused remote manifest/protocol suite
  passed 27/27 in 53.81 seconds.
- The remote protocol/control/legacy-runner compatibility matrix passed 55/55 in 91.90 seconds. Local
  `py_compile` and `git diff --check` also passed; these local checks are static evidence only.
- This tested slice does not change the overall `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION` verdict,
  create registration `I/R`, or unlock any formal Gate.

## Verified scope

### Gate-1 cost/profile slice under revision

The first exact-cost/full-stack implementation draft passed a remote focused/adjudication/core matrix
of 36/36 in 37.88 seconds, correctly deriving `B*` from direct `periodic4_transport` total-ms p50 and
enforcing the 20% dense-saving hard condition. Independent specification and code-quality reviews both
returned `REVISE_GATE1_SLICE`: arbitrary factories/provenance and arbitrary 30+30 record IDs were not
bound to registration; the exact 23-item profile set/order was not frozen; strict scalar schemas and
the safety-override invalidation were incomplete. This draft is `implemented_under_revision`, not an
approved profiler, formal cost artifact, or Gate-1 result. Deep registration-bound repair is in
progress.

The registration-bound repair now freezes the exact 23-candidate by 200-invocation profile plan,
derives provenance from deep registration v2, binds Gate-1 record artifacts to the exact 30/30
manifest splits, recomputes executed-action hashes, and rejects coercive ledger/safety paths. Its
remote focused matrix passed 24/24 in 58.26 seconds; a complete ChronoTransport regression passed
203 tests with one protected-CUDA-only skip in 315.68 seconds. Independent review still returned
`REVISE_GATE1_SLICE`: factory/source/Git identities remained self-declared, the manifest/checkpoint
were not re-read from real files, regret rows were not bound to paired replay, launcher prechecks were
incomplete, and exact 20% saving could fail from floating arithmetic. Status is
`tested_then_rejected_under_repair`; it is not a measured profile, Gate-1 result, `I`, or `R`.

### Stage-C ownership and overflow retry slice

The Stage-C primitive now snapshots/restores RNG, model tensors/buffers and Python state, optimizer,
EMA, scheduler, diagnostics, profiler, sampler and cursors across AMP overflow while retaining only
GradScaler backoff and an append-only retry audit. It enforces object-identity A/T/R ownership and the
initial-attempt-plus-three-retry limit. Remote focused verification passed 15 tests with one protected
CUDA-only smoke skipped; compatibility checks passed 27/27. Independent adversarial review then
returned `BLOCK_STAGEC_SLICE` with reproduced fail-open cases: omitted adapters and wrong optimizer
hyperparameters were accepted; infinite aggregate norm could succeed; incomplete Python state escaped
rollback; and actual action changes could hide behind an unchanged caller string. Status is
`tested_then_rejected_under_repair`; Stage-C training remains locked behind Gates 1--3.

### Formal Stage-B slice pending independent review

The first repaired formal Stage-B path implements exact three-seed/140-success exposure, paired RNG and
transform restoration including exception paths, fixed FP32 T+R optimization, EMA/scheduler success
counters, strict nonfinite/AMP invalidation, and atomic resume/ledger provenance. The remote focused
suite passed 20/20 in 117.87 seconds and the compatibility regression passed 71/71 in 199.51 seconds.
CUDA one-update smoke remains blocked by the absence of an eligible protected physical-GPU1
allocation. Independent review returned `REVISE_STAGEB_SLICE_BEFORE_APPROVAL`: the formal CLI still
accepted an arbitrary factory, the dense checkpoint was hashed but not strict-loaded, paired replay
and candidate-order evidence were callback-self-reported, and the fit-only 140x16 rank-127 baseline
was absent. Status is `tested_then_rejected_under_repair`, not trained or calibrated.

- Protocol canonicalization, label-free split/window helpers, Stage-B exposure and Stage-C exposure:
  remote `tests/test_chronotransport_r2_protocol.py`, 7 passed.
- Frozen r2 candidate library, motion/random exact-count controls, dual-age cache contract, and legacy
  core cache regression: remote focused suite, 36 passed.
- Runtime all-row adapter writeback, current-row live gradient, detached historical cache,
  requested/executed action separation, forced-dense/integration regressions: remote focused suite,
  35 passed.
- Fixed window-level D=23 mean/max quantile head, true-age feature, dense external safety semantics,
  and core scheduler regressions: remote focused suite, 30 passed.
- Gate 1 equal-cost oracle-headroom and Gate 2 matched TRANSPORT/HOLD pure adjudicators: remote
  synthetic focused suite, 4 passed.
- Stage-C object-identity ownership and loss-specific AMP gradient assignment: remote focused suite,
  4 passed. Overflow retry and the formal 4,200-update runner remain pending.
- Pre-Gate1 registration schema/claim chain, Gate 1 CLI, r2 Stage-B/C config overlays, and guarded
  GPU1 launcher: remote 4 tests passed plus launcher `bash -n`.

## Second/third repair audit cycle (2026-07-13)

Gate 1 reached a development-only 34/34 focused pass in 170.48 seconds after replacing self-declared
source/manifest/checkpoint evidence with Git/filesystem context, exact canonical bytes, controlled
checkpoint content addressing, paired replay provenance, formal precheck and exact decimal 20-percent
arithmetic. That pass is not an approval: self-audit still found the production OpenTAD profiler backend
missing at runtime and motion/random controls bound only by claimed hashes. A new RED-to-GREEN backend
slice now has six focused checks passing in 44.59 seconds and an external checkpoint-receipt contract
passing one check in 42.05 seconds, but real GPU invocation remains pending. The frozen specification is
also ambiguous about which run seed backs the three unsuffixed Gate-1 random comparators. The code now
fails closed when `control_seed` is absent; no seed or Gate-1 result has been guessed.

The second formal Stage-B repair passed a frozen remote focused suite of 32/32 in 129.76 seconds and a
71/71 compatibility matrix in 120.76 seconds, including strict dense checkpoint load, fixed repository
factory, sealed paired replay, real 16-order probes, full 140x16 rank-127 fit baseline, FP32, strict
resume and atomic prefix ledgers. Independent review nevertheless returned
`REVISE_STAGEB_FORMAL_SLICE`: the CLI validated only registration payload, not the current formal
`I/R` repository context, and it could leave a `COMPLETE` training checkpoint before the fit baseline
phase had an atomic completion marker. Both are implementation-integrity defects; no Stage-B seed has
been trained.

Stage-C repair cycles passed 19 focused checks with one protected-CUDA skip and 59 compatibility checks
with one skip, but a third independent adversarial review returned `REVISE_STAGEC_PRIMITIVE_SLICE`.
Reproduced fail-open cases included self-declared base-LR markers hiding wrong scheduler base LRs,
no-op success callbacks, caller-hidden TRANSPORT actions, implicit legacy ownership fallback and
Parameter version changes that made a restored non-leaf graph unusable after overflow. Earlier repairs
did correctly close omitted-adapter, infinite aggregate norm, non-persistent buffer and action-hash
gaps. Status remains `tested_then_rejected_under_repair`; Task-9 training and CUDA smoke are locked.

The remote Slurm audit still shows only job 1137541 on physical GPU4. There is no eligible protected
physical-GPU1 allocation, and no formal profiling or training job has started.

A user-requested independent maximum-strength line-by-line audit then returned
`REVISE_GATE1_IMPLEMENTATION_SLICE`. It confirmed the exact candidate/count/bootstrap arithmetic but
found that production profiling still admitted an injected backend, Gate-1 accepted caller regret,
media verification biased the first candidate timing, registration did not prove single-parent R-only
or pin the approved spec strongly enough, random seed and physical-GPU mapping remained unresolved,
and output/cost/provenance identities were incomplete. The focused Stage-B review independently
confirmed that a multi-parent R and a four-field pseudo-unlock were accepted. The latest Stage-C review
closed its previous five issues but returned `REVISE_STAGEC_PRIMITIVE_SLICE` for buffer version changes,
incomplete scheduler/global-state coherence and full-model parameter cloning. The complete actionable
report is `OSS_AUDIT.md`. These are implementation-integrity findings, not experiment results.

## Evidence boundary

These are behavioral implementation checks, not Gate 1--4 results. Formal Stage B/C, full-stack
timing, registration, and formal GPU1 deployment are not yet verified by this record. No scientific
claim is unlocked.

## Exact-SHA audit and fourth repair cycle (2026-07-13)

An independent reviewer matched the Gate-1 frozen hashes 11/11 and still returned
`REVISE_GATE1_FROZEN`: raw test rows reached formal schemas, paired evidence remained caller-owned,
direct adjudication/unlock paths bypassed formal R/random-lock checks, physical-GPU precheck was
fail-open, Git blob/mode binding was incomplete, and result/terminal paths could collide. Another TDD
repair is in progress; no formal profile was run.

Stage-B then added exact baseline-window/action binding, EMA cross-representation equality, a single
logical predictor hash, and canonical R/seed output confinement. Remote adversarial checks passed 5/5;
the full Stage-B file passed 46/46 in 130.27 seconds in isolated scratch
`/data/run01/sczc063/yuzibo/ct_stageb_p1_20260713_a61f09`. Exact-SHA independent source review returned
`APPROVE_STAGEB_FROZEN`. This approval covers the code slice only; no Stage-B seed was trained and the
GPU mapping spec blocker remains.

Stage-C's real-model repair passed 41 tests with one protected-CUDA skip and an 83-test compatibility
matrix with one skip. Independent exact-SHA review nevertheless found two P1s: caller-supplied action
identity could hide the action used by the forward, and unordered parameter-set checks could miss a
depth-12 block swap. Runtime-owned action evidence, exact topology/alias graphs, and memoized shared
Python-state rollback are now under repair.

A plan-to-files audit additionally found the formal Gate-2/3, Stage-C/matched-dense, Gate-4, launcher,
and corresponding test surfaces absent. Overall status remains `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`.

Gate-1 hardening then passed 51 focused tests with one strict expected failure for the unresolved GPU
mapping specification, plus 11 profile/adjudication and 49 replay/core/pipeline compatibility tests.
The 12 exact source/test/launcher SHA values were then reviewed by an agent that did not implement the
repair. It has already found three P1 paths: contextless direct unlock, importable token/issuer/raw-runner
surfaces that can mint formal evidence from caller objects, and failure to bind the in-memory registration
mapping to the exact regular Git blob at `R`. Terminal-marker TOCTOU is still being assessed. Gate 1 is
therefore `tested_then_rejected_under_review`, not Gate-1 PASS; the final verdict may add findings.

A new formal Gates-2/3 module/CLI/test reached 10 remote passes, but the implementer subsequently changed
the module and test SHA while strengthening phase-marker binding. The obsolete result was rejected; the
final exact bytes were then resynchronized to a fresh scratch and passed 10/10 in 43.32 seconds with all
three remote hashes rechecked. Independent review nevertheless reproduced a Gate-changing seed-bootstrap
error: the implementation produced CI `[0.0467, 0.4867]` and PASS where global seed-cluster resampling
produced `[-1.2, 1.0]` and FAIL. It also found formal report/unlock context bypasses and phase validation
that accepts arbitrary checkpoint/self-reported predictor/140-row ledger inputs. Gates 2/3 are
`tested_then_rejected_under_review`; no report or unlock exists.

After that audit, a new isolated Gate-4 pure adjudicator followed a remote missing-module RED and passed
12/12 synthetic adversarial tests in 40.47 seconds after adding full raw-evidence report recomputation
and tamper rejection. It preserves complete D/C/S matched blocks for
latency, uses official video as the metric/regret outer unit, resamples seeds internally, rebuilds each
seed's mAP from raw predictions and GT without cross-seed NMS, excludes timing repetitions from metric
and regret samples, and applies every approved Gate-4 hard threshold. p95, throughput, peak memory,
10-Hz long-block energy semantics, stage medians, mAP@0.3--0.7/Avg-mAP and duration quartiles are
diagnostics only. This is `tested_under_review`, not a formal Gate-4 runner or result; source registration,
Stage-C completion inputs, the CLI/launcher and an independent source audit are still required.

An evaluator-parity RED then showed the local Gate-4 AP routine returned `1.0` for an equal-score case
where the repository's official OpenTAD evaluator returned `0.5`. Gate 4 now calls the official evaluator
for every bootstrap reconstruction. The parity test passed alone, and the exact final slice passed 13/13
remotely in 232.34 seconds (`gate4.py` SHA `cc91d11...`, test SHA `e2cea83...`). The longer runtime is
expected from rebuilding official AP; the slice remains pure/tested-under-review, not formal Gate 4.

The second Stage-C repair reproduced all five new adversarial failures before implementation, then
passed those 5/5, the focused Stage-C/ViT set at 46 passed/1 protected-CUDA skip, and an eight-file
compatibility matrix at 88 passed/1 skip. The repair removes caller action assertions, freezes the
runtime-executed schedule after the real forward, checks exact ordered module/parameter/buffer topology
and alias multiplicity, and restores shared mutable graphs with memoized identity preservation. Exact
SHA files are frozen at `/data/run01/sczc063/yuzibo/ct_stagec_final2_20260713_6c117f` and are now in a
fresh independent audit. This is `tested_under_review`, not Stage-C training or CUDA approval; the only
skip is the unavailable protected physical-GPU1 GradScaler smoke.

That audit then reproduced two further P1 fail-open paths against the exact frozen SHA: a dummy real
runtime forward could be followed by A/T/R losses disconnected from that forward graph and still return
`SUCCESS`; and `frozen_parameter.data.add_(...)` could alter heavy bytes without a Tensor version bump,
also returning `SUCCESS`. The final verdict was `REVISE_STAGEC_FROZEN` after a third P1 showed that a
runtime summary missing all forced-dense/fallback/evidence-valid fields also passed; the primitive used
duck typing instead of binding the production runtime. P2 gaps cover success-path Python state, exact
GradScaler/autocast evidence, direct-dependency hashes, and runner-owned canonical window order. Stage C
is therefore back to `tested_then_rejected_under_repair`; Task 9 and deployment remain locked pending
another RED-to-GREEN repair and a new independent review.

Third-repair RED is frozen at `/data/run01/sczc063/yuzibo/ct_stagec_red3_20260713_4da91b`:
16/16 adversarial cases failed as intended in 53.38 seconds. Two focused toy/real-ViT detached-loss
probes separately confirmed `DID NOT RAISE` and a real `SUCCESS` path before implementation. The matrix
covers detached losses, heavy/projection/head `.data` mutation on success and overflow, missing/wrongly
typed summary fields, a duck runtime, and success-path Python-state contamination. GREEN is in progress.

Third-repair candidate bytes under `/data/run01/sczc063/yuzibo/ct_stagec_green3_20260713` passed 56/56
focused with one protected-CUDA skip, 134/134 in the eight-file compatibility matrix with the same skip,
and 34/34 manifest/protocol tests. The implementation binds the production runtime class
and forward identity, requires an exact summary schema, proves detector/feature/risk loss provenance from
the audited forward outputs, hashes logical frozen-parameter bytes, and constrains success Python-state
changes. A subsequent static review extended autocast proof to the audited risk forward and changed the
Stage-C SHA, so those pass counts no longer certify the current bytes; a full exact-byte rerun is in
progress. The final SHA `36eb6148...a138` then passed the exact eight-file matrix again at 134 passed,
1 protected-CUDA skip in 107.55 seconds; the unchanged protocol/manifest dependencies retain their 34/34
result. Status is now `tested_under_review`, not approved. Per-attempt frozen-byte hashing is O(frozen
model bytes), so its real GPU1 training overhead is an explicit P2 measurement gap.

The next independent exact-SHA review found two further P1 candidates before finishing remote probes:
`runtime.latest_signals` is checked only after the detector forward and can be replaced or `.data`-
modified before the audited risk forward; and the successful path never performs bitwise registered-
buffer equality, allowing silent buffer `.data` mutations to persist. The slice is therefore back to
`tested_then_rejected_under_review`; pass counts remain valid test history but do not approve Stage C.

Those candidates were then independently reproduced against the exact Green3 bytes in
`/data/run01/sczc063/yuzibo/tmp/audits/stagec_green3_independent_repro_20260713_a`. Three real
forward/autograd/optimizer success attempts replaced `latest_signals`, changed its logical values via
`.data`, or changed a registered `toy_input` buffer via `.data`; all three incorrectly returned
`SUCCESS`, with maximum logical-value delta 7.0. Two additional attempts changed the captured detector
output or feature output via `.data` after the forward boundary; both still passed the current VJP
provenance check and updated parameters. The exact eight-file SHA set matched 8/8 and its existing suite
remained 62 passed/1 protected-CUDA skip. Green3 is therefore formally `REVISE_STAGEC_GREEN3`; a
five-case RED-first Green4 repair is in progress.

Green4 expanded the boundary suite to 12 cases: the five original signal/buffer mutations, four
equal-value storage rebinds, feature-output reference replacement, and two post-success state-advance
buffer/training contaminations. Against the old bytes these groups failed 5, 4, 1, and 2 tests as
intended. Final candidate bytes bind signals/detector/feature tensors by reference, storage, metadata,
version, and logical value; recheck them before risk/loss and after success-state advance; and validate
all registered buffers plus training/Python state while retaining overflow rollback. Remote Green was
12/12 in 49.61 seconds and the exact eight-file matrix was 143 passed, 1 protected-CUDA skip, 1 existing
PyTorch scheduler-order warning in 89.13 seconds. Frozen candidate SHA are
`e00f3730...fa6f` (`stage_c.py`) and `eaa047ba...abce` (test), under
`/data/run01/sczc063/yuzibo/tmp/audits/r2_stagec_green4_20260715_a`. Status is
`tested_under_independent_review`, not approved or deployed.

Independent exact-SHA review returned `REVISE_STAGEC_GREEN4`. The explicit runtime signals, detector/
feature outputs and registered buffers are now storage-bound correctly, but ordinary Tensor-valued
Python attributes in `_module_python_state` still snapshot reference/version/value/shape/stride without
storage cdata/pointer/size. An equal-value `tensor.data = tensor.data.clone()` can therefore split aliases
on success or survive overflow restoration. This is one P1 requiring a focused Green5 RED-first repair.
The physical-GPU1 CUDA AMP test and real hashing/clone overhead remain unmeasured.

Green5 reproduced two ordinary Python Tensor-state defects on exact Green4 (`2 failed` in 46.60s):
equal-value storage rebind split a base/view alias on both success and overflow without raising. The new
candidate binds layout, storage cdata/pointer/nbytes/offset and alias-relevant identity and fails closed
when exact rollback is impossible. Remote Green passed 2/2 targeted, all Green4+5 14/14, and a documented
12-file stricter superset at 198 passed/1 protected-CUDA skip/1 existing scheduler warning in 129.10s.
Candidate SHA are `d994cefd...3fbe` and `6b9c113f...c837` under
`/data/run01/sczc063/yuzibo/tmp/audits/r2_stagec_green5_20260715_a`; status is
`tested_under_independent_review`.

## Gate-1 Green2 frozen identity

The second Gate-1 hardening repair is frozen for independent review as the following exact 15-file
SHA-256 vector. This is an identity record only, not Gate-1 PASS:

```text
3e0319fa496ad65ae7d59d97f62cdd7ffd60e44f82a8005158a60b01a97c14a6  opentad/models/chronotransport/registration.py
9a99b02a24e58fa1532c31bbbf71b9754cf2eafb68e14e5f33d0cc1ffa517000  opentad/models/chronotransport/full_stack_profiler.py
f5fe45fa6595777447a30b2f3201607c436971c740c00291411ee524a36af088  opentad/models/chronotransport/replay.py
2a86e126336e09e2ab5e62e4241cbb64400218d0cd9f97f547f3a0c47cb59805  opentad/models/chronotransport/adjudication.py
7fb61e87d604440385f220ba1c563420dbc76c6efd0f670ee3069a94ccd389fe  opentad/models/chronotransport/gate1_unlock.py
307f70bd4752426e7c4c3563b54d3f0105aee48c140caf57a262606a36fea13c  tools/bata/chronotransport_r2_profile_factory.py
0d5c124b84126a958ad2a84344aa33fec580e8809dc6c9ad6b098ed24dea2e68  tools/bata/profile_chronotransport_r2_full_stack.py
88d0511b51e3b3a4ca8ab377a3d8b3660deab3c36510aa070796ee9abf935e76  tools/bata/chronotransport_r2_opentad_profile_backend.py
531e500c7ff564da69d038d4cf26129533ebbbc4ec5e442fcabe3c7d2d66627a  tools/bata/chronotransport_r2_gate1_replay_factory.py
0a09769d895bf21c4d4cf7797435f5cca4efab2307504af588d934c3dcdc7abf  tools/bata/validate_chronotransport_r2_precheck.py
89427965bf7982435db7ca8535916df1d328b1388b6493c9bcf49f9965985858  tools/bata/run_chronotransport_r2_gate1.py
6537593b55a8a19736543ed90a96f198c69140150dd5ceadcabf26dcd1bae4e8  scripts/run_chronotransport_r2_gate1_gpu1.sh
df6597935ed6d168415dd3e1161e04e24949a374c39499922ef295ebcbd9cf3d  tests/test_chronotransport_r2_gate1_hardening.py
19ca391b08d4233e25c159f3df53e550ba0dd1a3454dbb783b0fe979bd04c36c  tests/test_chronotransport_r2_gate1_cost_profile.py
27adbaf6bd4b2e79884369fdd724cc73e27681f1a114900b82a914827023d315  tests/test_chronotransport_r2_registration.py
```

The independent reviewer matched 15/15 local bytes before review. Integration inspection also found
that `tools/bata/train_chronotransport_r2_stage_b.py` still calls the old Gate-1 unlock validator
signature and therefore cannot consume the Green2 unlock. The frozen Stage-B core remains untouched;
the CLI integration must be repaired RED-first, remotely re-run, and independently re-frozen.

A new zero-context exact-SHA review returned `REVISE_GATE1_GREEN2`. Profile construction still exposes
an arbitrary-backend `RegisteredProfileInvocation` and an any-callable profiler; formal and fixture
candidate/container representations share the same rebuild path, so a fixture can be retagged and
rehashed into bytes accepted by the formal validator. Replay likewise exposes a raw detector/batch
executor, while formal and fixture replay/record artifacts share row validation and rebuild logic;
synthetic rows can transitively reach a formally accepted record/unlock under an otherwise valid R
context. Mandatory R/current bytes and no-clobber publication passed review, but do not repair the
evidence-origin defect. Gate1 is `tested_then_rejected_under_repair`, not approved.

The Stage-B integration defect was separately reproduced RED-first in
`/data/run01/sczc063/yuzibo/tmp/audits/stageb_gate1_context_red_20260715_a`: the old helper rejected the
mandatory `repository_root` argument. The CLI now forwards repository root, current R, and registration
relative path to the Gate-1 unlock validator. Exact remote Stage-B tests passed 47/47 in 132.98 seconds;
the Stage-B core SHA remains `097af0d6...43f1`. Updated CLI SHA is `7a837198...f87` and test SHA is
`7d85c066...5d4`. A separate exact-SHA reviewer returned `APPROVE_STAGEB_CONTEXT_REPAIR`; the formal
Stage-B slice is frozen again, but cannot run until Gate1 itself is approved.

Gate1 Green3 removed the any-callable/arbitrary-backend profile API and raw replay executor, moved formal
execution behind exact repository-owned sessions, made profile candidate/invocation and replay/record row
schemas structurally disjoint, and made the formal replay validator require stored derived regret rather
than rebuilding formal rows from raw inputs. RED was 3 intended failures in 56.79 seconds. Corrected final
scratch `/data/run01/sczc063/yuzibo/tmp/audits/gate1_green3_final_20260715_b` passed 25 focused tests,
169 passed/1 expected random-lock xfail/1 existing TypedStorage warning in 396.97 seconds, plus 30/30
Gates23 compatibility. Eight final SHA values were matched local/remote. Status is
`tested_under_independent_review`; the implementing agent did not self-approve.

## Gates 2/3 round2 frozen identity

The RED-first Gates 2/3 repair synchronized the final Gate-1 mandatory context and a 17-file direct
dependency vector. In remote scratch
`/data/run01/sczc063/yuzibo/tmp/audits/r2_gates23_tdd_round2_20260713a`, the focused suite passed
16/16 in 41.83 seconds and final Gate-1 compatibility passed 61 with one strict xfail. Stage-B
compatibility passed 43 and failed two stale scratch tests that omitted the already-required candidate
action-hash mapping; the separate real Stage-B CLI old-signature blocker remains. Final files under
independent review are:

```text
cf205bcfed188f2ad99643d3c9a27d28f3b58e044dc542b88aad91b9591f68c4  opentad/models/chronotransport/gates23.py
3c350ad5a9f43f923005167d3dcf63680a113efbcc63def0ecc3ee1efdb8cb1e  tools/bata/run_chronotransport_r2_gates23.py
52990df327d028a725cd66b1d12d45bb5a2949a8675746fdafd499accf95269c  tools/bata/chronotransport_r2_gates23_replay_factory.py
f0eb72a2fcb840b1aab9cb3d85629a03dc42d4b01fc59f64f8704887e0195d34  tests/test_chronotransport_r2_gates23.py
```

The canonical 17-dependency vector SHA-256 is
`04e5572b82de31bbc69de199382793a51bbec6b7117ba7b99195174ee18543cf`. Status is
`tested_under_review`, not Gate-2/3 PASS and not an unlock.

Independent exact-byte review returned `REVISE_GATES23_ROUND2` despite a fresh 16/16 remote pass in
41.12 seconds. All four formal Gates2/3 paths are absent from the registration required-source set, so
the formal registration contract cannot currently represent this slice. The public terminal builder
also accepted caller strings and emitted `SUCCESS` without repository/Gate1/report recomputation; two
concurrent CLI writers both exited zero and overwrote one output; and a regular leaf reached through a
symlinked parent was accepted. Fixed replay ownership, full Stage-B checkpoint/140-row ledger validation,
global seed-cluster bootstrap, and formal/test replay schema separation passed review. Round3 repairs the
three local defects first; registration-vector integration remains separately blocked until both Gate1
and Gates2/3 bytes are stable.

Gates23 round3 now derives terminal state only from an exact revalidated report, emits no authoritative
terminal before report creation, acquires an exclusive run lock before existence checks, publishes by
fsync plus atomic hard-link no-clobber, and rejects symlinks in every existing path component. RED covered
four original defects plus repository-root symlink handling. Final focused remote result is 21/21 in
52.32 seconds; corrected Gate1 Green3 compatibility is 30/30 in 96.02 seconds. Four final files are frozen
at `b3b7dfcd...f8113fc`, `00406234...02204`, `9ac5805c...9114a`, and `c30ec389...250e4`.
Registration still lacks these final source paths, so status is `tested_pending_registration_integration`
and formal execution remains fail-closed.

Independent review returned `APPROVE_GATES23_ROUND3_CODE`: exact four-file hashes matched, a separate
21/21 remote suite passed, and an orchestration probe confirmed no pre-report terminal, lock lifetime,
and ordered no-clobber publication. The first registration integration added the four Gates23 paths and
passed 36/1xfail registration, 30 Gate1+Gates23, and 58 StageB tests, but review returned
`REVISE_REGISTRATION_GATES23_INTEGRATION` because the frozen Gate1 Green3 hardening test itself remained
outside the exact source set. No other current Gate1/Gates23/StageB file omission was found; one final
RED-first source-vector repair is in progress.

## Remote provenance

Environment: `/data/run01/sczc063/yuzibo/conda_envs/opentad`; isolated verification workdir:
`/data/run01/sczc063/yuzibo/workdirs/chronotransport_r2/repo`. The bounded protocol repair commit is
`33378af`; it is not the final implementation commit `I`.

Remote scheduling audit found no reusable physical-GPU1 allocation. On protected job `1137541`,
Slurm reported physical `GRES IDX:4` and the in-step read-only probe reported
`SLURM_STEP_GPUS=4`, `CUDA_VISIBLE_DEVICES=0`, confirming task/cgroup ordinal remapping. The old
launcher invariant `CUDA_VISIBLE_DEVICES=1` is therefore invalid under a single-GPU protected step.
The corrected guard must require physical `SLURM_STEP_GPUS` (or `SLURM_JOB_GPUS`) exactly `1`,
`SLURM_GPUS_ON_NODE=1`, and remapped local `CUDA_VISIBLE_DEVICES=0`; no current allocation satisfies
that contract. Formal GPU1 execution remains unauthorized and login-node training remains forbidden.

The 2026-07-13 20:40 +08:00 refresh also found running jobs `1161501` on physical GPU6 and `1161502`
on physical GPU4; `1137541` remained on physical GPU4. None is a physical-GPU1 allocation, and no
ChronoTransport job was submitted into those unrelated runs.

The 2026-07-15 refresh returned no active Slurm jobs for `sczc063`. Therefore there is still no
protected physical-GPU1 allocation to reuse; this is not evidence that login-node execution is allowed.

## Combined regression

Remote combined static/focused verification passed: 110 tests in 84.58 seconds, including every new
r2 test, existing ChronoTransport core/integration/formal-Stage-B/repository contracts, and the two
required C3 focused suites. This confirms the currently implemented surfaces only; it does not fill the
known missing Gate-3/Gate-4 adjudicators, overflow retry, full formal runners, or create registration R.

## Independent audit

Two independent follow-up reviewers approved the bounded protocol repair as
`APPROVE_PROTOCOL_SLICE` and `APPROVE_PROTOCOL_QUALITY` after reproducing and closing strict-type,
path, source-vector, canonical-byte, and rehashed-identity fail-open cases. The approved slice was
committed and pushed as `33378af`. These approvals are not approval of full r2 registration or Gates.

A fresh no-conversation-context agent returned `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`. Seven
registration-blocking gaps remain: Gate 3/4, executable r2 Stage B/C/matched dense, overflow retry, B*
and exact cost feasibility, full-stack profiling/provenance, strict derived registration validation,
and a registration-bound fixed Gate-1 input chain. I/R and formal deployment remain locked.

## External Pro GitHub audit

A GitHub-only Pro review of immutable snapshot `4b07020acb2611c3f085488d2f678f3be037f1be`
independently affirmed all seven blockers and found two additional P0 defects in surfaces that the
project-reported suite had treated as covered:

1. The r2 config overlay targets `model.backbone.chronotransport` instead of the actual inner
   `model.backbone.backbone.chronotransport` runtime.
2. The conformal helper flattens `window×candidate` residuals instead of taking a per-window maximum
   before rank 28 over the 30 calibration windows.

It also records incomplete formal manifest, per-window Spearman, Stage-C exposure/resume, exact-cost
ledger, and measured-stage profiler contracts. This audit is repository evidence, not an experiment or
independent test rerun. Verdict remains `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`; registration
readiness is `NOT_READY`, and the formal execution chain remains locked.

## 2026-07-15 pre-deployment integrity audit

A fresh read-only independent audit of the current dirty bytes again returned
`REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`. Exact blockers are recorded in
`research-wiki/sources/2026-07-15-chronotransport-r2-predeployment-integrity-audit.md`.
The audit found a real Stage-B reachability defect: the repository factory used historical flat
registration aliases although the validated registration embeds the manifest and Stage-B exposure
artifacts. A RED-first regression and nested-artifact validator were added. The already-present
Gate-1 hardening test and Gate-4 pure adjudicator/test were also added to the current exact source
vector. Only local static compilation has run on these new bytes.

The seven new targeted cases subsequently passed remotely in 57.28 seconds at
`/data/run01/sczc063/yuzibo/tmp/audits/ct_r2_integrity_fix_20260715_a`. They cover only nested
manifest/exposure/split provenance, no-clobber publication and the exclusive Stage-B writer lock;
the prior sentence is retained as the chronological pre-run state, not the current evidence state.

The full affected Stage-B plus registration suites subsequently passed in the same isolated worktree:
`89 passed, 1 xfailed in 310.86s`. There were no failures. This is broader implementation regression
evidence for the repaired surfaces, not a Stage-B training run, a registration approval, a Gate
artifact, or a scientific result.

The Gate-1 hardening, Gates-2/3 and Gate-4-adjudicator compatibility matrix then passed there as well:
`43 passed in 295.25s`. It covers only existing code contracts and does not constitute formal Gate
execution or Gate-4 evidence production.

A RED-first repair then addressed Gates-2/3 interruption after replay/report publication. Recovery is
allowed only when the existing immutable artifact bytes equal a fresh exact recomputation; a terminal,
byte mismatch, symlink/non-regular path or competing writer remains fail closed. Remote results in the
same isolated worktree were targeted `1 passed in 37.95s` and combined Gates-2/3+registration
`59 passed, 1 xfailed in 206.70s`. An independent exact-byte review returned
`APPROVE_GATES23_RECOVERY`, matching runner SHA `4CED5459...616885` and test SHA
`10D13457...65AFB9`. Status is `tested_and_bounded_code_approved`; it produced no formal Gate or
scientific evidence and does not approve the overall implementation for registration.

This does not resolve the unapproved random-control seed, current Slurm-vs-physical-GPU1 protocol
conflict, Stage-C train-mode buffer semantics, per-window batch-two detector-regret production path,
A-only matched-dense runner, formal Gate-4 runner, complete source vector, remote CUDA verification or
the mandatory second implementation approval. Registration remains `NOT_READY`; no formal execution
is authorized.

An unapproved minimal protocol amendment proposal was drafted at
`docs/methods/2026-07-15-chronotransport-r2-minimal-protocol-amendment-proposal.md` (SHA-256
`30371FFC17B02DF615FF0D772B93BADF30CF0A3AB84E36325CBF5A71EFD8469F`). It proposes exact A1--A4
decisions for the random seed, Slurm-assigned device, successful `loss_normalizer` trace, and paired
Stage-C per-window regret forward. Status is `proposed_unapproved`; the approved spec bytes and every
formal lock remain unchanged.

Stage-B formal path/lock hardening was then exercised RED-first. The old CLI followed a canonical-root
symlink alias before validation and its run-lock cleanup could remove a same-name replacement; the
symlink-parent lock path was also accepted. The repair performs lexical component checks before
resolution, opens locks with exclusive/no-follow semantics, and cleans up only the opened inode.
Remote results were targeted `5 passed in 44.70s` and full Stage-B+registration
`91 passed, 1 xfailed in 291.82s`. Independent exact-byte review returned
`APPROVE_STAGEB_PATH_LOCK_HARDENING`, matching runner SHA `64B4A5AA...053B3D` and test SHA
`1E7BB883...094CA`. Status is `tested_and_bounded_code_approved`; this does not change Stage-B/Gate
execution state or approve the complete implementation for registration.

Gate-1 precheck parent-path aliasing was then reproduced RED-first and repaired without resolving
caller paths before inspection. The exact candidate passed a focused Linux symlink regression
(`1 passed`), the complete registration/precheck suite (`38 passed, 1 xfailed`), and Gate-1
hardening/cost compatibility (`25 passed`). Independent review returned
`APPROVE_GATE1_PRECHECK_PATH_HARDENING` for precheck SHA `0BE0EA8B...F76808` and test SHA
`55916FBD...C10BA`. Status is `tested_and_bounded_code_approved`; no Gate artifact, unlock, I/R,
training job or scientific result was produced.

Stage-B partial-publication recovery was exercised at dangling periodic/final ledger and completed
training-pair boundaries. An initially green candidate was independently rejected because its public
phase builder still followed dense-checkpoint aliases and the CLI used pathname existence/load checks
after the pre-lock validation. The replacement exact bytes use inside-lock regular-file state checks,
descriptor/inode-bound checkpoint reads, exact-byte-only artifact reuse, full temporary-baseline phase
preflight, and impossible-state refusal. Final remote evidence is targeted `5 passed`, affected
Stage-B+registration `98 passed, 1 xfailed`, and Gate compatibility `44 passed`. Independent review
returned `APPROVE_STAGEB_PARTIAL_PUBLICATION_RECOVERY` for SHAs `50F4469D...F4F84`,
`47342FFE...A7670`, and `9BB46DE2...E378D`. Status is `tested_and_bounded_code_approved`; no formal
training or scientific result exists.

A Gate-4 integrity RED then demonstrated that the public pure adjudicator could accept arbitrary
caller-owned timing/metric/regret mappings with `formal=True` and emit the same report schema used by
the synthetic tests. The bounded repair now refuses every valid-parameter `formal=True` call before
raw evidence parsing and labels non-formal output as
`chronotransport-r2-gate4-test-only-v1` / `test_only_unregistered_raw_mappings`. Remote evidence in the
same isolated worktree is focused `1 passed`, forged-payload-with-recomputed-hash targeted `1 passed
in 105.28s`, full Gate-4 synthetic `13 passed in 242.40s`, and current registration-source coverage
`1 passed`. Exact SHAs are `A581D713...B1A75F` for `gate4.py` and `5C0FFAF3...AB4506E` for its test.
Independent exact-byte review returned `APPROVE_GATE4_CALLER_EVIDENCE_LOCK_FINAL`; status is
`tested_and_bounded_code_approved`. This intentionally removes a fake-formal path but does not
implement the official evidence producer, formal runner, Stage-C completion chain or Gate-4 experiment.

A follow-up reachability audit found that `_validate_formal_runtime_summary` accepted
`cost_is_measured=False` because the field was only type-checked. A RED reproduced the missing refusal.
Moving the field to the exact-true set initially exposed that all formal toy/ViT fixtures still used
proxy cost: the first full rerun honestly failed `39 failed, 32 passed, 1 skipped in 63.44s`. The fixtures
were then changed to an explicit test-only three-group measured-cost table. Final isolated-remote evidence
is focused `1 passed, 71 deselected in 45.26s` and full Stage-C `71 passed, 1 skipped in 76.60s` for exact
SHAs `5BDC1862...5577C4` and `C92FED39...3A262D7`. Independent exact-byte review returned
`APPROVE_STAGEC_MEASURED_COST_FLAG_LOCK`; status is `tested_and_bounded_code_approved`. This closes only
the false boolean-evidence path: immutable registered cost-profile identity, the 4,200-update runner,
matched-dense workflow and A3/A4 contracts remain missing.

## 2026-07-15 GitHub Pro review of immutable review-only snapshot

The branch snapshot `b854adb4f4c9235580b5e58c3f3255db6e9adbc0` passed the external snapshot gate but
received the overall verdict `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`. The review independently
confirmed the absent Stage-C/matched-dense/Gate-4 production chain, mismatch between the Stage-C canonical
Tensor hook and the real ActionFormer loss dictionary, train-mode `loss_normalizer` specification conflict,
unapproved A1--A4, incomplete registration source vector, Slurm device conflict and lack of immutable
measured-cost provenance. It preserved the bounded Gate-4 test-only adjudicator approval and did not claim
a P0 in its read surface.

Local read-only fact checking confirmed that the four named workflow paths are absent, that
`ActionFormer.forward_train` returns a dictionary, that `AnchorFreeHead.losses` mutates
`loss_normalizer` in training, and that the two changed integration tests are absent from
`REQUIRED_REGISTRATION_SOURCE_PATHS`. The reviewer’s broad automatic glob suggestion was narrowed:
`tests/test_chronotransport*.py` discovers 21 files while the vector includes 14, leaving five additional
old/general files besides the two confirmed omissions. Formal completeness must therefore use an explicit
classification manifest rather than filename prefix alone. Its proposed Stage-C evidence interface and
launcher path remain unapproved design suggestions.

The verbatim source and detailed disposition are archived at SHA-256
`1A7B9D5AEA47302AC7BCB29DB9EF54DAD97CF3D45DF1536691CB9B536EC4C376`. The reviewer did not run tests,
CUDA or Slurm; this record likewise started no experiment. Status remains
`REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`, A1--A4 `proposed_unapproved`, registration `NOT_READY`, and
all formal Gate/Stage-C execution absent.

## 2026-07-15 A1--A4 spec-only successor and fresh Pro input

The user subsequently authorized the exact A1--A4 decisions. Commit
`537f692189cf0c5a6ee7d40ad8c4ed1032bf1d37` changed only the governing specification; its SHA-256 is
`E79DFAAB8F9B0093E96CBD6B46BEF4ECF8D6433009E2DCB922AD0F4C473B27A6`. This supersedes only the
historical `proposed_unapproved` protocol-authority blocker. It does not implement A1--A4, repair the
production workflows, approve registration, or alter the prior `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`
status.

The refreshed GitHub-only Pro input has SHA-256
`9DDCABC19E6B38874EA97F5E4702C247D2DF8F485CE273E08E4A6515EBFEC3D0`. It requires a fresh immutable
descendant snapshot, a separate spec-diff verdict, and a full production implementation verdict. No tests,
CUDA/Slurm execution, training, formal Gate, or scientific result were produced while preparing it.

## 2026-07-16 complete Pro review at `1b6366d`

The external reviewer verified review-only SHA
`1b6366d0acb712e8096c2cceb0f05e66b16d30d4`, tree
`3fc64c72cf26b77f041d059f51385f29e5e85462`, its single parent `537f692...1d37`, and the current
spec hash. The A1--A4 diff received `APPROVE_SPEC_FOR_PLAN`; the implementation received
`REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`. The reviewer executed no tests, CUDA, Slurm, training,
profiling or evaluation, so this changes specification-review status only and creates no experiment fact.

Current-source checking confirmed the core P0s. `registration.py` still binds `e4422f5` and the old spec
hash, while its formal random lock rejects both random candidates and any `control_seed`. The Gate-1 shell
and backend still require physical GPU1/CVD=1. The Stage-C audit hook requires exactly one top-level model
forward returning a differentiable Tensor and rejects all successful buffer changes, whereas real
`ActionFormer.forward_train` returns a loss dictionary and `AnchorFreeHead.losses` advances
`loss_normalizer` in train mode. The five expected Stage-C/matched/Gate-4 workflow paths remain absent.

The review's implementation sketches are not frozen interfaces. The A2 patch lacks the allocation/step/
device artifact schema and does not cover the full-stack profiler CLI, Stage-B CLI, profile backend,
Gates-2/3 claim key or stale implementation plan. Its Stage-C dataclass fixes a loss-key set and detector
feature field not specified by A3/A4. Its source-classification example saw 19 tests; the checkout contains
21, including `test_chronotransport_opentad_replay.py` and `test_chronotransport_stage_a_smoke.py`, and all
must be individually classified before any source vector is frozen.

The permanent-kill list is narrowed at implementation violations: an unauthorized job or provenance
failure invalidates and quarantines that run, requiring repair/re-review/rerun; it does not by itself prove a
science Gate failure. Formal Gate failure, deliberate post-hoc protocol changes or unrecoverable
contamination retain their frozen-route semantics.

The verbatim review is archived as 86,871 bytes / 2,019 lines with SHA-256
`C61F93531885040A3593DB7552E23B67B34DEC3D55095D71FCE5B6D2A1F1BC08`; its detailed disposition is
in the paired absorption record. Status is A1--A4 `spec_approved`, implementation
`REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`, registration `NOT_READY`, and
`experiment_running=false`. No implementation code or experiment was changed or launched during
absorption.

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
