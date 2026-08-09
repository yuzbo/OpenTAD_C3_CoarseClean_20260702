# Research Log

## 2026-08-09 — Historical negative control was split from production numeric admission

- Fully read external adjudication `U-PRO-STAGEA-SPLIT-LEGACY-NC-1`, source
  SHA-256 `4f900ba49d0d7884a0a2beecbaf809542bb80e19bb42fee65c8bd003cfd2e082`.
  Its `GO_SPLIT_HISTORICAL_NEGATIVE_CONTROL` verdict is accepted: the old raw
  FP32 row-mass statistic is gauge-dependent and cannot require historical-bug
  reproduction on a repaired real trajectory.
- Implemented a deterministic `T96/K48` code-regression negative control and a
  production gate that stops at the first eligible successful-update
  `T768/K384` capture. Legacy production output is explicitly
  `diagnostic_only/admission_effect=none`; synchronized owner/update identity,
  current row/column/order/dual-logZ/edge/gradient/hard-path/shift oracles and
  no-metric failure receipts are explicit.
- Focused Slurm prechecks `1233451` and `1233452` failed before Python because
  `--wrap`/non-interactive Bash lacked `module/source`. No numerical assertion
  ran. The corrected canonical-interpreter precheck `1233456` completed `0:0`
  and established old-fixed-fixture failure plus current-solver structural pass.
- Local syntax, shell and contract checks pass; `tests/test_duca_paper_full200_contract.py`
  reports `21 passed, 1 skipped`. The change is not yet a clean commit and has
  not passed the full authoritative code, natural-short, two-rank numeric or
  exact-211 release chain. State is `implemented / focused-tested /
  pending_clean_commit_and_authoritative_gates`.
- No model, loss, budget, threshold, seed, data split, checkpoint rule or
  evaluator changed. No Stage-A job was released, no mAP was opened and no
  method-performance claim is allowed.

## 2026-08-06 — Repaired numeric gate completed its bounded search but legacy trigger was absent

- Exact AMP-replay source `4b766457b5abd8247f7e054d64bf6eb725183493`
  passed clean Linux/PyTorch code gate `1223270` with 149 tests; receipt SHA-256
  is `b73c73f1048bb0d39342f37309da7d43b7258355615d8d621e1db6d2b842ee0a`.
  Real natural-short gate `1223282` also passed; receipt SHA-256 is
  `4a481eb1b7f7a94039470ddf08060a8d9de6570b39011cd68574d1326f3f5fd7`.
- Two-rank release-gate job `1223308` ran for 5m19s and failed `1:0` at
  `/data/run01/sczc063/yuzibo/rime_preflight/duca_paper_release_gates_4b766457_20260806_004306`.
  Exact terminal exception on both ranks is `old production FP32 guard did not
  trigger within 100 updates`; immutable log SHA-256 is
  `7b36ccc2c8866d7e5e5a526361a97d85cffa3286a9d104f1c518a7ff74d408ab`.
  The execution therefore crossed the earlier immediate-capture and pre-step AMP
  overflow failure surfaces and reached the terminal bounded-search predicate.
- No numeric receipt was written, so the failed process did not persist per-rank
  target-capture counts, maximum legacy drift or AMP-replay counts. It is known
  only that no rank produced a target whose deprecated raw-message statistic
  exceeded its legacy envelope before the terminal check. Exact-211 did not run,
  no aggregate receipt exists and no Stage-A job was submitted.
- This is no longer authorized for automatic engineering repair. The required
  trigger is the superseded raw alpha/beta row-mass guard already adjudicated as
  gauge-dependent and non-structural, while the repaired solver's accepted
  invariants are dual-logZ, brute-force/FP64 agreement, row/column/order,
  edge-flow, finite gradients and unchanged hard path. Whether historical-bug
  reproduction remains a valid release prerequisite is a scientific gate-design
  decision.
- Current state is
  `gate_failed_closed / legacy_trigger_absent /
  scientific_gate_adjudication_required`. Do not delete the requirement, extend
  the search, change its envelope or retry automatically. A bounded Pro review is
  now warranted to choose between retaining the negative-control requirement,
  replacing it with actual-production-tensor oracle coverage, or requiring a
  separate immutable historical-replay fixture. No performance metric was opened
  and Stage B remains sealed.

## 2026-08-06 — Numeric gate exposed a second gate-versus-production AMP mismatch

- Exact source `751ce695f6bb4681dc26f8669ea7c4f01acb875b` passed clean
  Linux/PyTorch code gate `1223013` with 148 tests; receipt SHA-256 is
  `5e707aaf5e93c552fed45d8a1da8ccb8cc690fc9047d0ae55a3afbd8978a1f0d`.
  It also passed real natural-short-window gate `1223116`; receipt SHA-256 is
  `697a8df66603e0da22529898cbcd9aa2400662ee44cdaa36f8e547c7574b3626`.
  The latter used all 200 training videos, found 43 natural short rows and zero
  subquantum rows, and used neither validation/test nor synthetic input.
- Release-gate job `1223142` failed `1:0` at root
  `/data/run01/sczc063/yuzibo/rime_preflight/duca_paper_release_gates_751ce695_20260806_002924`.
  Both ranks raised `actual learned full-model backward produced non-finite
  gradients`; immutable log SHA-256 is
  `3a1bc78e7ed3442480b79a9b4873e8c41319eed060cd19f597e61ccd9624cda8`.
  No numeric, exact-211 or aggregate receipt exists and no Stage-A job was
  submitted.
- Registered signature
  `numeric_gate_pre_step_gradient_finiteness_bypasses_formal_amp_replay`. The
  formal Stage-A config freezes eight AMP retries per batch, and the production
  train engine lets `GradScaler.step/update` classify an overflow, restores RNG,
  model buffers and custom selector replay state, and advances optimizer,
  scheduler and selector only on a successful step. The gate instead failed on
  unscaled gradients before GradScaler could skip and lower its scale.
- The bounded correction mirrors the existing production replay contract inside
  the numeric gate. Every DDP rank synchronizes the step outcome; skipped attempts
  restore the same state and do not count as successful updates or captures.
  Successful attempts still require all finite gradients, and exhausting the
  unchanged eight-retry limit fails closed. The initial GradScaler, model, solver,
  objective, data, budget, thresholds and paper question remain unchanged.
- This diagnosis is `ENGINEERING_STATUS`, not evidence that model gradients are
  acceptable. That conclusion requires a fresh exact-commit two-rank gate to
  complete a real successful update and all frozen solver oracles. No metric was
  opened; Stage B remains sealed.

## 2026-08-06 — Numeric release gate stopped on an over-strong per-update capture check

- Published and installed exact test-repair source
  `dfe787f1e39e09c55ddb3c459367eba081cb5abf` in clean remote checkout
  `/data/run01/sczc063/yuzibo/OpenTAD_DUCA_PAPER_dfe787f1`; complete transport
  bundle SHA-256 is
  `c9991fe26f87bbb43cc45edf4bcb59b9040f0a5ffca54bd4ae8de30501623527`.
- Authoritative code gate `1222944` completed `0:0` with 147 Linux/PyTorch tests;
  receipt SHA-256 is
  `d341f91a32aa02930b1e40b4f84ed3eab0a1cb08c02b5a5a1d044a5b09349918`.
  Real natural-short-window heavy-backbone gate `1222951` completed `0:0`;
  receipt SHA-256 is
  `8e85e81b0b995e26599025061a45d941ca5aeae68d49b15abb64604f4d76e5a5`.
- Combined release-gate job `1222954` failed `1:0` at fresh root
  `/data/run01/sczc063/yuzibo/rime_preflight/duca_paper_release_gates_dfe787f1_20260806_000536`.
  Exact exception: `actual training update did not reach T768/K384 solver`.
  Exact-211 did not run, no release-gate receipt exists and no Stage-A cell was
  submitted.
- Registered failure signature
  `numeric_gate_per_update_t768_k384_capture_requirement`. The runner promised to
  reproduce the legacy condition within at most 100 real full-model updates, but
  immediately required every successful rank-local update to contain a natural
  `T=768,K=384` row. Natural short rows legitimately use a smaller effective K,
  and the two DDP ranks can observe different valid lengths on an update.
- The bounded repair treats a non-target update as an allowed search step while
  all ranks still execute one synchronized `MAX` trigger reduction every update.
  The terminal owner reduction continues to fail closed unless an exact target
  reproduces the legacy guard within 100 updates. Per-rank target-capture counts
  are recorded for diagnosis. No solver, model, loss, budget meaning, threshold,
  data, seed, checkpoint, evaluator or paper question changed.
- No loss value, checkpoint, prediction, partial metric or mAP was opened. This is
  `ENGINEERING_STATUS`; Stage B remains sealed and a fresh exact-commit four-gate
  chain is required before Stage-A release.

## 2026-08-05 — First four-gate release attempt stopped on FP64 test-reference dtype

- Transported exact source `e0a58ab2e576522ccf335a8fea44bdcef71e490b`
  to clean checkout
  `/data/run01/sczc063/yuzibo/OpenTAD_DUCA_PAPER_e0a58ab2` using complete Git
  bundle SHA-256
  `6e0baad404221d32d80f41b12190d0cb4b9a090ace4fae7906f9c4214c3b2636`.
  The registered runtime annotation/video links and all three asset hashes were
  reverified while Git remained clean; the old `7e893569` root was untouched.
- Submitted authoritative code gate `1222939` at fresh root
  `/data/run01/sczc063/yuzibo/rime_preflight/duca_paper_code_gate_e0a58ab2_20260805_235718`.
  It failed `1:0` after `146 passed / 1 failed`; no gate receipt exists and no
  short-window, numeric, exact-211 or Stage-A job was submitted.
- Exact failure signature:
  `linux_fp64_oracle_expected_ones_dtype_mismatch`. The small brute-force FP64
  distribution comparison passed, then the normalization assertion constructed
  a default-FP32 `torch.ones` reference and PyTorch raised `Double did not match
  Float`. This is deterministic test construction, not solver/model behavior.
- Applied the bounded test-only correction: construct the normalization identity
  with `torch.ones_like` on the actual slot-mass tensor. No model, solver, loss,
  budget, threshold, data, seed, checkpoint, evaluator or paper question changed.
  The failed root/log remain immutable; a new commit and entirely fresh code-gate
  root are required. No loss, checkpoint, prediction, partial metric or mAP was
  opened.

## 2026-08-05 — N16R4 access identity corrected and made persistent

- Root cause of the false remote-block conclusion: the network endpoint,
  ParaCloud login label and Slurm cluster name were collapsed into the bare
  command `ssh N16R4`, before inspecting the repository's existing registered
  recovery script. With no prior `~/.ssh/config`, OpenSSH treated `N16R4` as a
  DNS hostname and failed before authentication.
- Recovered the exact registered access tuple from immutable local evidence:
  gateway `ssh.cn-zhongwei-1.paracloud.com`, port `22`, login
  `sczc063@BSCC-N16R4`, key `C:/Users/skywalker/.ssh/id_rsa`, and the required
  RSA compatibility options. No private-key contents or credentials were read.
- Added local SSH aliases `N16R4` and `BSCC-N16R4`. A fresh `ssh N16R4`
  read-only probe succeeded and verified remote host `ln01`, effective user
  `sczc063`, `/usr/bin/sbatch`, `/usr/bin/squeue`, and Slurm
  `ClusterName=n16r4`. The first alias read exposed inherited
  `CodexSandboxUsers` ACLs; these were removed and the config ACL was matched to
  the existing private-key owner/SYSTEM/Administrators scope before the passing
  probe. This was another local pre-network failure, not remote unavailability.
- Froze the diagnostic order in `RTK.md` and anti-repetition memory: expand the
  alias first, verify the read-only remote identity second, and only then perform
  Slurm operations. Remote Stage-A release is no longer connection-blocked, but
  no gate/job/result is claimed by this access repair itself.

## 2026-08-05 — Stage-A four-gate release chain implemented locally

- Continued the accepted `GO_MINIMAL_SOLVER_REPAIR` route without another Pro
  round. The scientific matrix remains unchanged: four ActionFormer arms, three
  fixed seeds, all 200 training videos, exact 211 validation videos, 60 epochs,
  6000 successful updates and terminal epoch-59 EMA.
- Added a production-like learned numeric gate. It runs the formal DUCA config
  through two-rank DDP, global batch two and AMP, captures an actual
  `T=768,K=384` physical exact-K solver input, requires the superseded raw
  unnormalized guard to reproduce within at most 100 successful updates, and
  applies the frozen FP32/FP64 slot, gradient, dual-logZ, edge-flow, row-mass and
  Viterbi checks to the repaired solver. It creates no checkpoint, prediction or
  metric and cannot support a paper-performance claim.
- Added an exact-211 metadata/physical-UID gate. It builds the formal sliding
  index without decoding video, running a model, generating predictions or
  accessing metrics; it requires the complete official 211-ID set, unique
  `(video_id,window_start_frame)` keys, unique source-frame-sequence UIDs and the
  historical `video_test_0001431/7680` key exactly once.
- Bound code, natural-short, numeric and exact-211 receipts by exact commit,
  path and SHA-256 through the matrix manifest, both Stage-A submitters, every
  training/cell receipt and the final twelve-cell seal. Roots must remain under
  the registered remote base and outside the Git worktree.
- Two independent read-only audits found and closed a DDP rank-local failure
  deadlock, incomplete tensor/config/asset validation, missing four-gate release
  bindings and missing external-root enforcement. Validators now require an
  explicit external SHA-256 and independently verify the captured tensor
  artifact and per-rank summaries. A final clean-commit audit additionally
  required an explicit bound for exceptions raised before a DDP collective; the
  numeric runner now uses elastic worker supervision, NCCL asynchronous error
  handling, a receipt-bound 600-second process-group timeout and a fixed
  14,400-second outer process watchdog that also bounds non-collective CUDA
  stalls.
- Local Bash/Python static checks pass. The paper contract suite reports
  `17 passed / 1 Windows-only skip`; the repository-mandated C3 checks report
  `23 passed`. The Windows host still cannot initialize Torch `c10.dll`, so no
  local GPU/model execution is claimed. Authoritative Linux code/short/numeric/
  exact-211 receipts and a fresh Stage-A transaction do not yet exist. No loss,
  checkpoint, partial metric or mAP was opened; Stage B remains sealed.
- The implementation and bounded-failure commits through `3a0563fb` were pushed
  to `origin/codex/duca-rime-20260727`, and a fresh independent read-only audit
  returned `GO` with no P0/P1 for the release-gate scope. Remote release is not
  claimed: this desktop session exposes no SSH/Slurm connector and
  `ssh -o BatchMode=yes N16R4` fails with an unresolved hostname. No job ID,
  remote receipt or transaction root was fabricated; execution remains blocked
  only on restoring the registered N16R4 endpoint.

## 2026-08-05 — Minimal Stage-A solver repair accepted and implemented locally

- Fully read and SHA-bound `U-PRO-STAGEA-MINIMAL-SOLVER-REPAIR-1` (SHA-256
  `58e761262430c5ecead0f923fca93dd1a9576742e644d8c63428c517cab834b8`).
  Accepted its `GO_MINIMAL_SOLVER_REPAIR`, deterministic duplicate-window root
  cause, AMP precision-boundary diagnosis and scaled log-semiring direction.
- Did not adopt the report verbatim. A generic pre-backbone execution journal is
  unnecessary for the proven dataset-source duplicate and would expand crash/API
  semantics. Proposed numeric thresholds are not treated as calibrated facts;
  the solver pseudocode was independently rederived so both forward and backward
  message scales and restored log partition remain exact.
- Implemented unique canonical sliding starts while preserving the physical
  `(video,start)` key and snippet-center annotation endpoints. The formal
  `N=2688,W=768,S=384` case now emits terminal start 1920 exactly once.
- Kept AMP/FP32 coverage probabilities and log probabilities at FP32; explicit
  FP64 diagnostic input stays FP64. Replaced the gauge-dependent raw slot-mass
  envelope with global exact-K gauge centering, per-slot normalized alpha/beta
  messages, carried scales, restored logZ and an independent backward/source
  partition identity. Existing row, column, ordering, finite-gradient and hard
  Viterbi contracts remain fail closed.
- Added focused regressions for exact terminal enumeration, short/exact/overflow
  windows, small-graph brute-force marginals/logZ/gradients, additive-gauge
  invariance, FP32-versus-FP64 long-chain agreement, AMP dtype preservation and
  the production-shaped `T=768,K=384` high-dynamic-range backward path.
- `py_compile`, `git diff --check`, direct CPU numerical oracles and the full
  `T=768,K=384` stress pass. Normal pytest collection is skipped on Windows by
  the repository c10.dll guard. A fresh independent read-only diff audit returned
  GO; its FP64 dtype caveat was corrected.
- Current status is `implemented / local_direct_tested /
  authoritative_Linux_gate_pending`. No Stage-A job was submitted, no old root or
  receipt was reused, and no metric/mAP was opened. Stage B and all extensions
  remain blocked.
- The exact implementation and test commit is
  `cb077a77d48d9776028fa4d88fcf5b3ca1d9e357`; it is not yet a production gate
  receipt or experiment source authorization. It has been pushed to GitHub
  branch `codex/duca-rime-20260727`; the mandatory C3 plus Stage-A contract
  suites passed locally as `38 passed / 1 skipped`, while the three Torch-heavy
  modules remain explicitly Windows-skipped and therefore require Linux gate
  execution.

## 2026-08-04 — Stage A terminally failed closed; no metric was opened

- Exact source `7e8935692b732f2958ba3c20787ae19c86f7b15c`, clean checkout,
  protocol/submission/release hashes and both prerequisite gate receipts were
  reverified unchanged. The failure is therefore inside the released execution,
  not repository or transaction identity drift.
- Control jobs `1215390`, `1215392`, and `1215394` ended `FAILED 1:0` after a
  dense terminal cell and fixed-uniform training. Their immutable log SHA-256
  values are `8d679ef897263bb0e93cb7721230d351ad1a5a235072891e5970fcb65635c623`,
  `0a75f5003fd6b1dbafaace72d1896c23d6c4accda8acde6fe51902330916e38f`, and
  `9f90c4ce9e47670975d3fddfcdb24049ae8334e449887e760b51a7c648fb3943`.
  All three failed the fixed-requested-K384 evaluation ledger's unique-key
  predicate at line 722. Rows 721/722 are byte-identical and both identify
  `(video_test_0001431, window_start_frame=7680)`; canonical row SHA-256 is
  `400d197499d056b2874aa1646cccd56cbc83378ea24e2b76e3bf47a75d2b2fb6`.
  Every K/shape/protocol predicate on that row passed. Register
  `paper_exact_uniform_eval_duplicate_window_ledger_key` and require diagnosis
  of dataset-sample identity versus duplicate ledger emission before any repair;
  post-hoc deduplication is not authorized.
- Learned-DUCA jobs `1215391`, `1215393`, and `1215395` ended `FAILED 1:0` on
  `physical exact-K raw slot-mass drift exceeds the FP32 normalization envelope`.
  Their log SHA-256 values are
  `12290ad8d8479bec90c3eed3292899a267152ac4d633c450b1e35af1a732907b`,
  `b1cbe7c87be37fc9f4d41d99ce057f1dedec8eaafdf84bb8bd333a3f74ade65a`, and
  `9a9f3123a09bcbb9691028ced5a7f6ace7eafe980db5f26adcc97d291e23ce5e`.
  This is classified as a repeat/continuation of
  `physical_exactk_long_chain_fp32_slot_mass_loss`; its recovery quota was
  already consumed and rule 139 forbids threshold relaxation or automatic
  redeployment.
- Only the three dense cell receipts exist. Fixed-uniform has three training
  receipts but no terminal cell receipt; mixed-K never started; learned DUCA has
  no terminal training/cell receipt; no matrix receipt exists. Exact seal job
  `1215396` was cancelled after `DependencyNeverSatisfied` made sealing
  impossible. Stage B and all extensions remain disabled.
- No training loss, checkpoint, partial/single-seed metric, incomplete matrix or
  mAP was opened. This is terminal `ENGINEERING_STATUS`, not performance
  evidence; no comparison with AdaTAD mAP=65 is valid.

## 2026-08-03 — Exact-K repair passed dual gates; fresh Stage A released

- Published exact numerical-repair source
  `7e8935692b732f2958ba3c20787ae19c86f7b15c`. Its transport bundle SHA-256 is
  `e95b36a53f661c7b98063aeda7dabcc45d2254d5decf353e44578ec928e01476`;
  fresh remote checkout is
  `/data/run01/sczc063/yuzibo/OpenTAD_DUCA_PAPER_7e893569` with verified runtime
  annotation/video bindings and a clean exact commit.
- Slurm code gate `1215388` completed `0:0` with 140 Linux/PyTorch tests. It now
  includes the small brute-force Gibbs test and a `T=768,K=384` high-dynamic-
  range finite-backward regression. Receipt SHA-256 is
  `92b31ab9a9d33580c3b27a367a3866d5ffdf5cf2f9e34a186384ef42e6e0bbf0`.
- Real natural-short-window gate `1215389` completed `0:0`; receipt SHA-256 is
  `904dacf22d6d88cba8206367c6b824da002f717e7227ad1f310afa08a4a89b16`.
  It covered all 200 training videos, found no sub-quantum window and completed
  all four actual selector/gather/VideoMAE paths without padding or repetition.
- Released fresh immutable root
  `/data/run01/sczc063/yuzibo/rime_runs/duca_paper_stage_a_7e893569_20260803_030306`.
  Protocol/submission/released-receipt SHA-256 values are
  `33903a7eee1089202295161368edf4e5e5b3470c88b072c9fa00f82f15fefaf3`,
  `1f4b14569f45977d6b606febe7c42898d401258ee5f73b918b56d197d65bb105`, and
  `cea74e4bd231e8687e583f2fa98ff5dacec3c41e2ec5352396e097e7abfb01f7`.
  Jobs are controls/DUCA `1215390/1215391`, `1215392/1215393`,
  `1215394/1215395`, with seal `1215396` afterok all six. Every training group
  entered `RUNNING`; DUCA error scans remained empty beyond the old immediate
  failure boundary.
- No training loss, checkpoint, single-seed value, partial metric or incomplete
  matrix was opened. This is `ENGINEERING_STATUS`; Stage B remains disabled and
  no paper-admissible empirical conclusion exists.

## 2026-08-03 — Stage-A learned exact-K numeric failure and narrow stabilization

- Recovery root
  `/data/run01/sczc063/yuzibo/rime_runs/duca_paper_stage_a_00f54dfe_retry1_20260803_023358`
  passed both prerequisite identities but all three learned-DUCA jobs failed in
  epoch 0: `1215378`, `1215380`, and `1215382` raised `physical exact-K slot
  marginals do not sum to one` from
  `structured_selection.py::_physical_row_forward_backward`. Their log SHA-256
  values are `5371743766d85d7df461682e9b498ffbcd25c332b6021fd50a646e6f234b4b1b`,
  `7db05504b28713b0d8a19ffe840d042de7d0af2b36da7ebb1502965b46cddad2`, and
  `5d688a5b2171f6a4e24d66c428ff7db60c9016f7c47d0803501cf1d1b429a780`.
  Register `physical_exactk_long_chain_fp32_slot_mass_loss`.
- The isolated controls `1215377/1215379/1215381` remained operational, proving
  the seven-job failure isolation, but were cancelled because no result from the
  old source can enter a new commit-bound matrix. Seal `1215383` was also
  cancelled. No loss, checkpoint, partial metric or mAP was opened.
- Three independent read-only audits found that graph reachability and finite
  partition checks passed before the failure. Physical forward/backward was only
  tested at `T=6,K=3`; its long chain accumulates in FP32 and directly
  exponentiates `alpha+beta-logZ`. The real short-window gate covered a
  no-gradient single-sample path and could not establish training stability.
- Diagnostic jobs `1215384` and `1215385` exited before Python on module/profile
  bootstrap errors. Corrected diagnostic `1215386` established the expected
  FP32-versus-FP64 precision gap. Scale sweep `1215387` reproduced the production
  invariant failure at `T=768,K=384` for score scales 16/32/64; per-slot
  log-domain normalization and FP64 both passed with finite gradients. The scale
  diagnostic log SHA-256 is
  `c2200fc76264e1d3d42d89bf6e5b2ac1fee305751cf84adc9ba217714e57ef9b`.
- Chose log-domain slot normalization rather than FP64 because it is the
  mathematical categorical normalization already implied by the Gibbs marginal
  and does not impose FP64 selector cost. The graph, log partition, Viterbi hard
  path, budget, loss and model architecture remain unchanged; column occupancy
  and ordered expectations still fail closed. A post-patch independent audit
  identified that unconstrained normalization could hide a uniformly scaled DP
  error, so the implementation now rejects pre-normalization log-mass drift
  outside a conservative FP32 accumulation envelope before projecting. Added a small brute-force
  equivalence test, a `T=768,K=384` high-dynamic-range backward regression, and
  included the physical structured-selection suite in the formal code gate.
  Current state: `implemented / local_non_torch_checked / remote_gate_pending`.

## 2026-08-03 — Corrected Stage-A code-gate engineering failures and narrow repair

- Transported exact clean source `75b9ba3d2053675ef83902e03dd4ff705c235244`
  to a fresh N16R4 checkout using bundle SHA-256
  `74ff3e99666128053af94166c66e7cf850d7c815c7ff2cd86471efc1040677e5`.
  The old failed source/root remained untouched and no metric was opened.
- Code-gate job `1215366` failed `127:0` before its root existed because Slurm
  `--wrap` used POSIX `sh`, which does not implement `source`. Register
  `slurm_wrap_posix_sh_source_not_found`; retry only with explicit
  `/bin/bash -lc` and a new root.
- The bounded launcher retry `1215367` entered the authoritative Linux/PyTorch
  suite and stopped at `90 passed / 1 failed`. The sub-quantum q=16 decoder
  already failed closed correctly, but the exception message differed from the
  focused regression's frozen wording. Register
  `subquantum_failclosed_exception_message_contract_mismatch`.
- Implemented the narrow message-only repair. It changes no model, loss,
  requested/effective K semantics, data, seed, checkpoint, evaluator, threshold,
  or metric. Both failed jobs and their roots/logs remain immutable; no passing
  receipt, Stage-A manifest, or experiment transaction exists yet.
- Repair source `00f54dfecb6a536224958b1cd64d2daa5b8ca982` passed code gate
  `1215368` (`91 passed`, receipt SHA-256
  `5cb16630ea07f38db6dc9a14d9bcd18efa2a0c9ab8f408dd0c9fde3610c26185`)
  and real natural-short-window gate `1215369` (receipt SHA-256
  `2eec808e36d9eb92a8f22eee67d5a00588e4f114e6b27dbaca9a56a495b29d89`).
  The latter enumerated full-200, found 43 natural short samples and zero
  sub-quantum samples, and completed all four actual heavy-backbone executions
  without padding/repetition. This remains engineering-only evidence.
- The first formal root
  `/data/run01/sczc063/yuzibo/rime_runs/duca_paper_stage_a_00f54dfe_20260803_023009`
  failed before training: jobs `1215370`–`1215375` shared `runtime annotation
  binding drift`; seal `1215376` was cancelled. Register
  `missing_runtime_thumos_relative_bindings`. The exact clean checkout lacked
  ignored OpenTAD runtime data symlinks; no cell artifact or metric exists.
- After verifying that the prior formal link targets exactly match the frozen
  annotation/class-map hashes, restored the runtime links, verified all four
  configs and retained clean commit status. The signature's one-time recovery
  released root
  `/data/run01/sczc063/yuzibo/rime_runs/duca_paper_stage_a_00f54dfe_retry1_20260803_023358`.
  Protocol/submission/released-receipt SHA-256 values are respectively
  `b4baa6b60954c00dc906740d801a170cc079021192cc2c7c2c81f7f5bc209366`,
  `8def0bce9a0447b9a3d25f6a171452ba9e15ad2f5fdf937571052d2915f3e19e`, and
  `c49d8f3f1b017ec11ef7ad1ca3c246e2798fb7892af183544809421bb658c97c`.
  Jobs are `1215377/1215378`, `1215379/1215380`, `1215381/1215382` for the
  three control/DUCA seed pairs, with seal `1215383`. Metrics and Stage B remain
  sealed.

## 2026-08-03 — Stage-A natural-short-window corrigendum accepted and implemented

- Fully read and hash-registered `U-PRO-STAGEA-SHORT-K-CORRIGENDUM-1` (SHA-256
  `20901b2f5acd8da44f00aa0a2b86985ec59670855f211643ca4707888291058f`).
  Its `IMPLEMENT` verdict is accepted with two repository/cluster realizations:
  q=16 sub-quantum windows fail closed, and the twelve logical cells use a
  seven-job scheduler DAG to respect `MaxSubmitJobs=16`.
- The mixed schedule remains `(8,12,16,24)` over requested
  `(192,256,384,512)` with nominal requested mean 384. Natural windows use
  `K_eff=min(K_req,floor(L/16)*16)`; actual execution must prove
  `K_backbone=K_unique=K_eff<=K_req`. Padding, repetition, video deletion and
  length-conditioned requests remain prohibited.
- Implemented separate requested/feasible/unique/backbone accounting,
  successful-update-only and AMP-retry-safe training ledgers, actual
  BackboneWrapper/inner-VideoMAE tensor receipts, exact-211 evaluation ledger
  checks, and a real full-data natural-short-window heavy-backbone Slurm gate.
  The corrected matrix manifest cannot be created without that gate receipt.
- The fresh scheduler shape is three per-seed sequential control jobs, three
  independent per-seed DUCA jobs and one dependent seal. The old `2df0103e`
  transaction and its six unopened partial receipts remain immutable and cannot
  be reused. No metric has been read; Stage B remains sealed.
- Evidence level is currently `implemented`; final local checks, independent
  audit, exact-clean Linux/Slurm verification and fresh deployment are pending.
- Final local non-Torch verification passed `15` tests with one Linux-only loader
  test skipped. The local Windows Torch runtime remains unusable because
  `c10.dll` initialization fails, so selector/detector Torch status is deferred
  to the authoritative Slurm gate.
- Two independent read-only audits verified the actual selector/gather/backbone
  K chain and the scoped real-data preflight. Their valid hardening suggestions
  were implemented: every training/cell/matrix/submission receipt now binds the
  short-window gate hash, the old `2df0103e` source and root are explicitly
  rejected, and the final released submission receipt has a SHA-256 sidecar.
- Deployment-order review found a P0 circular dependency in the old code-gate
  launcher: it tried to create a formal manifest before the new prerequisite
  gate existed. The launcher now performs only clean Linux/PyTorch verification
  and records `short_window_gate_pending=true`; the enforced order is code gate,
  real natural-short-window gate, new manifest/root, then seven-job release.
- A final chain audit found that ordering was initially procedural rather than
  receipt-enforced. Added a standalone clean-code-gate validator; the short-window
  gate now consumes that exact commit/hash-bound receipt, and the manifest,
  submission, training, cell and seal receipts bind both prerequisite hashes.
- Post-repair independent audit returned `GO` with no P0/P1: all heredoc argument
  mappings align, the two-gate chain is non-bypassable and acyclic, the seven-job
  DAG is isolated, the old source/root are denied and Stage B remains sealed.

## 2026-08-02 — Full-data Stage A failed closed on mixed-K short windows

- Seed jobs `1213712` (5801), `1213713` (8123), and `1213714` (12011) all
  terminated `FAILED 1:0` in the third sequential arm. The exact shared exception
  is `ValueError: uniform_mixed_k forbids effective-K shrinkage on a short
  window`; register failure signature
  `paper_full200_uniform_mixed_k_short_window_exact_requested_k_infeasible`.
- Immutable failed-log SHA-256 values are
  `9ed49fa701b13c99960c0ef5fa88e597021120fe16bc3d810ad60c6293ff0879`,
  `dae2a78d35157b4d6efdc93c31e9f7452789ae69c263b819ac1b3fe404c6e0da`, and
  `604aa86707635f00c93de7d8af526fa9b1356e94371ca2eca3a07d66f513217a`.
- Exact source remained clean at
  `2df0103ec1c26ff7cff7ed15f399e78e640df211`; protocol and submission manifest
  hashes remained
  `ef8c1991fe5a638607cc1ac96e1da55802985000ce0a47c0b17160349ebc392f` and
  `9ee9e4a7ba0f413ccf3a348bc3096487bd71f67949aeaa147ec6c31385a65f9a`.
- Seal job `1213715` became `DependencyNeverSatisfied` and was cancelled by
  exact ID. Six dense/uniform cell receipts exist, but they are an incomplete
  matrix and their metrics were not opened. The learned DUCA fixed-K384 arm did
  not run because it followed the failed mixed-K arm in each grouped job.
- This is a scientific protocol inconsistency: the old mixed-K design requires
  exact requested-K execution and forbids shrinkage/padding, whereas full-200
  inclusion necessarily presents short windows. Shrinkage, exclusion, padding,
  or a different feasible schedule each changes a frozen scientific dimension.
  The monitor therefore made no code change and launched no retry. Stage B stays
  blocked and no performance conclusion exists.
- Heartbeat `duca-rime` was paused after recording this terminal state. It may be
  resumed only after an explicit mixed-K short-window semantic corrigendum and a
  fresh hash-bound transaction are approved.

## 2026-08-02 — Grouped full-data Stage-A transaction released

- Published exact source commit `2df0103ec1c26ff7cff7ed15f399e78e640df211`
  and installed clean remote checkout
  `/data/run01/sczc063/yuzibo/OpenTAD_DUCA_PAPER_2df0103e`.
- Authoritative exact-commit Slurm gate `1213711` completed with exit `0:0` and
  37 Linux/PyTorch tests. Protocol-manifest SHA-256 is
  `ef8c1991fe5a638607cc1ac96e1da55802985000ce0a47c0b17160349ebc392f`; gate-
  receipt SHA-256 is
  `12b83daa514094ecdc5dc967e85eda9227dd5b17c04f7d92c5571b63635f823e`.
- Released production root
  `/data/run01/sczc063/yuzibo/rime_runs/duca_paper_stage_a_2df0103e_20260802_120351`.
  Submission-manifest SHA-256 is
  `9ee9e4a7ba0f413ccf3a348bc3096487bd71f67949aeaa147ec6c31385a65f9a`;
  released-receipt SHA-256 is
  `96c7268937761f1aca9ab19b7ed18e36ab001d99de84a96af9af6b8b74b92759`.
- Seed jobs are `1213712` (5801), `1213713` (8123), and `1213714` (12011).
  Each executes four independently rooted, independently seeded logical arms.
  Matrix seal job `1213715` uses exact `afterok` dependencies on all three seed
  jobs. The manifest records 12 logical cells, four scheduler jobs,
  `grouping_only=true`, `phase_b=false`, and `single_seed_claim=false`.
- Initial scheduler state was `PENDING (AssocGrpGRES)` for all seed jobs and
  `PENDING (Dependency)` for the seal. This is `ENGINEERING_STATUS`; no training
  value, partial evaluation, single-seed metric, or incomplete-matrix result was
  opened or interpreted.
- First state change: seed jobs `1213712/1213713/1213714` entered `RUNNING`; seal
  `1213715` remains `PENDING (Dependency)`. No metric was opened.
- Existing heartbeat `duca-rime` was reactivated at a 15-minute interval and
  rebound exclusively to this transaction. It may perform at most one bounded
  recovery for a new, reproducible, protocol-invariant engineering signature;
  scientific/protocol changes fail closed. It cannot start Stage B or inspect
  partial metrics, and it pauses after a valid full-matrix seal or terminal stop.

## 2026-08-02 — Stage-A Linux gate passed; scheduler grouping required

- Published paper-matrix commit `1ec4faaa3afbc4d65bb16dade16de03e2bf4b457`
  and installed a clean remote checkout with the registered ignored runtime data
  symlinks. Exact annotation identity is 200 training plus 211 validation videos;
  the class map has 20 nonempty rows.
- Authoritative Slurm gate `1213710` completed with exit `0:0`; 37 Linux/PyTorch
  contract tests passed. Protocol-manifest SHA-256 is
  `0d929cb8d2698bc69ef2cc454ff35d54ffefe3c1a3b03aedb8fd0b15686b2901` and
  gate-receipt SHA-256 is
  `355a4382908165a72625bf51bce3f7d5d8d7eab987facf605686ae23d545242b`.
- The exact post-gate scheduler snapshot contained 12 other user-owned jobs.
  The original 13-job representation cannot be submitted under
  `MaxSubmitJobs=16`; no unrelated job was canceled.
- Implemented an execution-equivalent grouped DAG: three two-GPU seed jobs, each
  running four independently rooted/seeded logical arms sequentially, plus one
  dependent seal. The 12 logical cells, models, data, updates, checkpoints,
  evaluation and per-cell receipts are unchanged. A fresh exact-commit Slurm
  gate is required before release.
- This is `ENGINEERING_STATUS`; no model metric was read or interpreted.

## 2026-08-02 — Paper-facing full-200/exact-211 Stage A implemented

- The user authorized direct implementation and execution; another Pro round is
  not required before Stage A because the scientific question, arms, data,
  update count, seeds, checkpoint rule and release boundary are fully frozen.
- Implemented four ActionFormer arms by three seeds: dense T768, exact-uniform
  K384, mixed-K training with exact-uniform K384 evaluation, and jointly learned
  DUCA fixed K384 positions. Dynamic K is explicitly deferred to Stage B.
- Corrected the method identity: DUCA uses a jointly optimized train-only
  ASFormer frontend, not a frozen external coarse checkpoint. Its scan cost is
  included in full-stack cost.
- Added full-200 stateless two-rank exposure validation, exact 6000-update audit,
  terminal EMA compaction/training receipts, exact-211 prediction/merge/NMS/
  evaluator validation, a SHA-bound 12-cell matrix manifest, transactional held
  Slurm submission, and a dependent full-matrix seal.
- Corrected a test-entry regression that had accidentally nested the old
  protected `seed=3407` rule under the new paper protocol. Registered paper
  seeds `5801/8123/12011` now have a dedicated tested request validator. Source
  resolved-config identity is sealed before permitted runtime path overrides,
  while the effective runtime config has a separate hash.
- Local Python/Bash checks pass. The new focused suite reports `11 passed` and
  one Windows-only skip for the authoritative Linux loader test; the local host
  still cannot initialize its PyTorch DLL. Clean Linux/Slurm verification and
  submission remain pending. No metric was inspected and no paper-admissible
  empirical conclusion exists.

## 2026-08-02 — User stopped simulation-first work and restored model priority

- The user explicitly rejected further complex candidate-free Admission
  engineering before DUCA model feasibility is established. The unexecuted
  52-by-500 simulation, 24-by-200 MC calibration and 4M reference work are no
  longer active prerequisites and will consume no compute without new explicit
  authorization.
- The simulation design is preserved as negative-history/protocol provenance;
  production Admission remains `NO_GO`, but this historical gate no longer
  blocks a paper-facing DUCA experiment.
- A read-only repository audit found no `PAPER_ADMISSIBLE_RESULT`. Existing
  20-video measurements, salvage checkpoints and candidate-free fixtures remain
  engineering-only and will not be used to explain model performance.
- The connected window-local DUCA path exists, but the official experiment still
  needs a full-200 configuration, exact-211 evaluator completeness, a frozen
  training-only coarse-scan checkpoint, exact protocol/pretrain assets and
  homogeneous dynamic-K execution. H-RIME remains replay/oracle-only and is
  deferred until window-local DUCA feasibility is established.
- The recommended first paper-facing panel is ActionFormer/K384 over seeds
  5801/8123/12011, comparing dense, uniform fixed-K, uniform mixed-K, DUCA
  fixed-K position selection, DUCA dynamic-K and exact realized-cost
  `U-same-K`. Every trainable arm uses all 200 training videos and the complete
  211-video OpenTAD evaluation; paper metrics include official mAP, high-IoU,
  short-action and measured full-stack cost.
- No model code was changed and no experiment was submitted in this decision
  update. The next implementation requires review of this compact feasibility
  design.

## 2026-08-02 — V2.1 full-simulation execution protocol found incomplete

- Began direct execution of the user-approved candidate-free Admission v2.1
  stages from clean branch tip `77acd054965d4e5527c170cedf3073a3fe7cb04b`.
- Repository audit confirmed that the 52-scenario registry, one-outer executor,
  scenario/MC evaluators and receipt validators exist, but no full runner, task
  manifest, shard writer, strict reducer, resumable artifact contract or
  terminal execution receipt writer exists.
- Identified a new P0 protocol boundary: the helper defaults every outer to
  primary stream zero/diagnostic stream one; five shift profiles regenerate the
  same deterministic streams without a registered common-random-number policy;
  and no producer defines the 4,800 operational streams or the two independent
  2M halves and their concatenated 4M reference. These choices can alter random
  dependence or binary64 output and were not invented locally.
- Candidate-free scalar microbenchmarks measured about 8.85 seconds for 10,000
  multiplier replicates, 0.063 seconds for one 10,000-replicate maxT reduction,
  and 0.559 seconds for ten delete-1000 recomputations. These are local capacity
  diagnostics only, not statistical or model results.
- Read-only N16R4 inquiry found `MaxArraySize=1001`, account
  `MaxSubmitJobs=16`, a GPU-only partition and default eight CPUs per GPU. The
  OpenTAD environment is Python 3.10.20/NumPy 1.23.5, not the frozen Python
  3.11.7/NumPy 1.23.5 simulation runtime. A naive one-task-per-outer release is
  therefore invalid.
- Froze the recommended content-addressed shard/reducer design, alternative
  rejection reasons, exact release gates, and one narrow Pro execution prompt in
  `docs/superpowers/specs/2026-08-02-duca-admission-v2-1-full-simulation-execution-design.md`.
- Current state is `execution_protocol_review_required / full_simulation_not_run /
  production_admission_NO_GO`. No real-video worker, training, holdout, Phase 1+,
  learned H-RIME or official-final action was started. No paper-admissible
  empirical conclusion exists.

## 2026-08-01 — Final v2.1 narrow repair accepted and Stage A--D implemented

- Fully read and hash-registered `U-PRO-V21-FINAL-REPAIR-1`:
  `C:/Users/skywalker/.codex/attachments/a09b8a5b-c1ae-462a-94e3-c2681c29ad86/pasted-text.txt`,
  SHA-256
  `9b62a23d29d1cd74063f34127a64ba7a100805a3a60456ec9212163a8398da04`,
  68,504 bytes.
- Accepted the core route and bounded protocol implementation. Independently
  fixed exact-zero to binary64 positive zero, the 64-value type-1 median to
  zero-based index 31, and finite/positive requirements for all nonzero-branch
  scales, bounds and MC normalizers.
- Implemented deterministic source roles/reserves/triplets, canonical ranks,
  the role-specific connected 32-by-8 incidence, 192-cell manifest, 12-metric
  registry, independent role contrast, positive two-point product multiplier
  with `kappa=1`, fixed-scale maxT, prefix-extensible MC and delete-one-1000
  jackknife.
- Implemented the exact 52-scenario and 24-scenario MC-calibration registries,
  generators, one-outer executor, acceptance aggregators and strict receipt
  validators. These code paths are candidate-free. The full 52-by-500 and
  24-by-200 executions have not run and remain blockers, not results.
- Implemented the 37-row runtime registry, planned-cell/worker bindings,
  mandatory independent evidence verifiers for any `PASSED` receipt, explicit
  old-v2 rejection and descriptor-bound allowlisted POSIX publication. An Ubuntu
  smoke check passed fresh-root, atomic hard-link, parent-hash and symlink
  rejection paths; authoritative exact-clean Slurm receipts remain absent.
- Focused checks currently pass (`49 passed, 2 skipped` on Windows); the two
  skips are POSIX-only and are not counted as Linux/Slurm evidence. The broad
  non-Torch DUCA/C3 contract suite passes (`247 passed, 5 skipped`). The full
  Torch collection remains unavailable on this Windows host because `c10.dll`
  fails to initialize; that is an environment limitation, not a test pass.
- No detector/backbone/loss/budget/selector/checkpoint/evaluator code changed.
  Real-video workers, scale-fit, calibration, holdout, Phase 1+, learned H-RIME,
  full-200 refit and official-final remain unauthorized.
- Full adjudication and implementation status are recorded in
  `docs/superpowers/plans/2026-08-01-duca-admission-v2-1-final-repair-implementation.md`.
- Evidence status remains:
  `No paper-admissible empirical conclusion is available yet`.

## 2026-07-31 — Proposed v2.1 corrigendum independently rejected as implementation-ready

- Fully read and hash-registered `U-PRO-V21-CORRIGENDUM-1`:
  `C:/Users/skywalker/.codex/attachments/39eb4169-2000-4a3e-ba57-5bfe441bab1f/pasted-text.txt`,
  SHA-256
  `12a324b2eb43086397a0e54d5c64dae84c86fec96cef03ab0c8decc095cc7f37`,
  67,868 bytes.
- Verified the live local/GitHub branch identity at
  `d3e9814afd16739dadc273f181deb9a065c151d4`, tree
  `c2f74b963bc6291b96a3d18133fb90c9eb3e3901`, with a clean worktree before
  this documentation update.
- Accepted the positive factor-multiplier direction, fixed equal-video
  estimating equation, fixed-scale maxT naming, deterministic MC extension,
  structural/numeric-tail separation and scoped trusted-program threat model.
- Did not accept `protocol_implementation = GO` as written. The incidence uses
  mutable manifest row rank while demanding row-permutation invariance; the
  tail pseudocode puts a calibration/holdout block in `video_id`; role/triplet
  formation and reserve assignment are not byte-level deterministic.
- Independently checked the proposed finite scalar on the exact 32-by-8 graph.
  Product weighting without the scalar has expected row/process/interaction
  variance gains `117/64`, `69/64`, and `181/64`. Multiplying by
  `(32/31)(8/7)` changes them to `468/217`, `276/217`, and `724/217`.
  Owen--Eckles supports product factor weights and qualified conservatism, not
  this asserted finite-level scalar.
- Found that the coverage grid is not fully reproducible and has no
  non-vacuity/power acceptance, while the batch-jackknife MC half-width is not
  calibrated for the nonsmooth maxT quantile and joint bounds.
- Found that runtime receipt enums, planned-cell/parent bindings,
  status-to-authorization invariants and cluster attestations remain open.
- Recorded the final narrow correction contract and a ready-to-send Pro prompt
  in
  `docs/superpowers/plans/2026-07-31-duca-v2-1-corrigendum-independent-audit.md`.
  No protocol/model code, real-video worker, holdout, training, full-200 refit
  or official-final action was started.
- Evidence status remains
  `No paper-admissible empirical conclusion is available yet`.

## 2026-07-30 — Final v2.1 Pro response conditionally accepted; statistical corrigendum required

- Fully read and hash-registered `U-PRO-V21-FINAL-1`:
  `C:/Users/skywalker/.codex/attachments/934f541a-db6e-4bc8-94cc-272905f3d42c/pasted-text.txt`,
  SHA-256
  `9e7efa045f0b2a01dfc52755a6376205346bf76673483b61573dd55951d7c871`,
  94,487 bytes.
- Accepted the selected-axis mainline, role-level 32/32/32 coverage,
  margin-free Admission, full-200 five-by-forty OOF/DDP2 refit, exact-211
  release, H-RIME stage order and AdapTok naming/novelty boundaries.
- Did not accept `GO_IMPLEMENT_V2_1` verbatim. Independent audit found that
  sparse multinomial empty-row handling changes the equal-video estimand
  without a coverage proof; the fixed-scale maxT is mislabeled as studentized;
  binary two-stream agreement is not a calibrated Monte Carlo error criterion;
  and the catastrophic holdout-max/calibration-max ratio is count-dependent and
  unstable.
- Classified administrator network/mount/read-only attestations as unresolved
  until the threat model and N16R4 feasibility are explicit. Exact repository,
  path, process, shard and receipt controls remain hard; observations may not be
  labeled enforced.
- Recorded that the current H-RIME Stage-1 finalizer does not yet implement the
  future primary/maxT/shuffle-null/absolute-surrogate gate. No H-RIME code
  change is authorized before Admission and selected-axis Phase 1 close.
- Prepared a narrow statistical-protocol corrigendum prompt and post-decision
  implementation order in
  `docs/superpowers/plans/2026-07-30-duca-v2-1-pro-response-adjudication.md`.
  No model code, calibration, training, Phase 1, full-200 artifact or
  official-final action was started.
- Evidence status remains
  `No paper-admissible empirical conclusion is available yet`.

## 2026-07-30 — Next phase requires bounded Pro scientific adjudication

- Recovered the full project state from `query_pack.md`,
  `anti_repetition.md`, the Admission-v2.1 repair plan and the approved H-RIME
  specification. Local branch and GitHub branch were clean and identical at
  `505cc4e48b2511a13ad4936c295dfcb084d8d7fd`.
- Determined that formal training is not authorized. Admission v2.1 still lacks
  frozen role-level window coverage, exact crossed uncertainty/catastrophic
  statistics, a candidate-independent NI policy and truthful enforceable versus
  observed isolation semantics. The all-200-video OOF/global-batch-two refit is
  also not implemented.
- The next discussion is therefore a bounded decision review, not general
  ideation. It must return formulas, deterministic algorithms, role manifests,
  receipt schemas, tests, file-level implementation order and explicit
  GO/NO-GO gates. The exact Linux/PyTorch code gate may proceed as independent
  engineering verification; no calibration, model training or official-final
  evaluation may start before the scientific decision is frozen.
- No new empirical result was produced. The evidence status remains
  `No paper-admissible empirical conclusion is available yet`.

## 2026-07-29 — Official full-train boundary corrected

- Audited the complete Phase-4 training and evaluation path in response to the
  user's official-comparability correction.
- Confirmed that the registered training set has 200 videos, while the current
  Phase-4 trainer exports the `detector_selector_train` block list and trains
  RIME plus matched controls on only 100. This is a valid development partition
  but cannot support the paper main table or direct published-number claims.
- Confirmed that the evaluation path requires `block_list=None`, the exact
  registered official-final video-key set, and rejects missing or extra
  predictions. OpenTAD's data contract intentionally excludes two
  malformed/empty THUMOS test videos, leaving a complete comparable set of 211.
- Identified the required refit contract: frozen method, leakage-safe OOF
  targets for all 200 training videos, global effective batch two, 60 epochs,
  100 optimizer updates per epoch, 6000 total updates, matched controls and the
  unchanged upstream evaluator/NMS.
- Hard-disabled the current Phase-4 cell entrypoint. No official-final
  experiment may start until the full-200 refit producer, receipts and code
  gate exist. No current result is paper-admissible.
- Current focused checks pass (`66 passed`); the broad non-Torch contract suite
  passes (`219 passed, 3 skipped`). Full Torch collection remains unavailable
  on the Windows host due the known `c10.dll` WinError 1114, so an exact
  clean-commit Linux/PyTorch Slurm gate is still required.

## 2026-07-29 — Admission v2.1 audit adjudicated; impossible window contract found

- Fully read and registered `U-PRO-ADMISSION-V21-1`. Accepted its current
  formal-v2 `NO-GO`, all twelve P0 findings, preservation of the pure
  selected-axis model, and the real-video/full-model/disjoint-holdout direction.
- Did not accept the proposed protocol verbatim. Independently identified four
  blockers: the fixed natural-window coverage is physically infeasible, the
  sparse video/process bootstrap and catastrophic bound are underdefined,
  `2 * reporting quantum` is not a scientific NI-margin justification, and
  repository-level Slurm code cannot prove hard network/mount/object-lock
  isolation without cluster support.
- Read-only remote verification bound split manifest SHA-256
  `41349cd39a6a550b6e1613de968577b1605c93902edd52a88309121b9e90c057`.
  Its `detector_selector_train` pool has 100 videos. Production-compatible
  enumeration from immutable annotation metadata gives 70 full-only, 30
  short-only, zero with both, and fixed short-bin counts 7/13/10. Thus `3 x 32`
  is count-feasible, but the proposed per-video full-plus-short rule and
  minimum-eight first short bin are not.
- Implemented the noncontroversial Stage-0 response. Old Admission v2 formal
  calibration/admission now fails closed; the random-head path is
  engineering-fixture-only; old v2 receipts require explicit
  historical-read-only parsing and cannot authorize production entrypoints.
- Added a pure metadata v2.1 feasibility auditor and focused tests. It mirrors
  `SlidingWindowDataset.split_video_to_windows`, emits a finite exclusive
  content-bound typed failure, consumes no decoded frames or candidate output,
  and never authorizes Phase 1.
- Full v2.1 calibration and Phase 1 remain unstarted. The next decision must
  freeze role-level window coverage, exact crossed statistics, a justified NI
  margin and enforceable isolation claims. Phase 4 and official-final remain
  sealed; no model-performance or paper-admissible empirical conclusion exists.

## 2026-07-29 — DUCA acquisition-v2 selected-axis implementation

- Accepted the core Pro adjudication with bounded implementation corrections:
  the paper mainline is the pure pre-backbone selected-axis acquisition plugin;
  physical-time head injection remains a separately named integration route.
- Implemented and locally audited the selected-axis coordinate contract,
  standard-head restoration, exact-K/no-padding execution, GT remapping,
  inverse mapping before NMS, content-bound Admission v2, numeric-null
  calibration, and pre-candidate scientific-protocol freeze.
- Commit `70cf49de82a9d0ed889ed94af9604edd61070e55` was pushed and transported by
  SHA-bound Git bundle to a clean N16R4 checkout. Bundle SHA-256 is
  `5347836c57564a151f10818908d36e7488211ca66a72e9bf4356c70405c6e9af`.
- Authoritative Slurm code-gate job `1204048` failed closed with exit `1:0`;
  247 tests passed and one stale assertion still expected the superseded
  `duca_protected_physical_v1` Phase-1 protocol instead of
  `duca_rime_selected_axis_plugin_v2`. No gate receipt was produced.
- Failure signature:
  `phase1_uniform_test_expected_superseded_physical_protocol`.
  The correction updates only the regression expectation and adds assertions
  that the selected-axis plugin uses the standard detector head. The focused
  local regression passes. No model, loss, data, threshold, checkpoint,
  metric, or scientific protocol changed.
- The failed preflight root is immutable:
  `/data/run01/sczc063/yuzibo/rime_preflight/duca_acquisition_v2_70cf49de_20260729_202923`.
  Numeric calibration, scientific admission, Phase 1, Phase 4, and
  official-final were not started.
- Remediation commit `119db2f83756281729506632a18bfed607794d13`
  passed authoritative Slurm code-gate job `1204067` with exit `0:0`.
  Gate-receipt SHA-256 is
  `d664a619007f1cafbd4e52f2fd6a053fb0e3b5336dcb2be2b16302912286e5c8`.
- A post-gate evidence audit stopped the workflow before numeric calibration.
  No non-fixture training-target JSONL, acquisition data manifest, or
  `duca_acquisition_ni_margin_source_v1` exists in the registered remote
  assets. Code-gate fixture targets and block lists are forbidden substitutes.
- An independent specification-to-code audit found a P0 protocol gap in the
  current numeric calibrator: it runs synthetic head-feature fixtures while
  declaring `train_only_calibration`, does not consume real role-scoped videos,
  has no video/process-grouped null distribution or independent process
  launches, and does not freeze the registered normalized statistic and
  denominator floor. Therefore the calibrator is an implemented draft, not a
  valid Admission-v2 producer.
- Admission v2, Phase 1, Phase 4, and official-final remain unopened. The next
  authorized work is to repair and preregister the real-data numeric
  calibration protocol and generate its train/calibration-only input assets;
  no performance claim is available.

## 2026-07-27 — DUCA-RIME four-stage implementation

- Recorded user approval for direct four-stage execution.
- Marked the earlier total-60 bounded-density plan as superseded by the
  dynamic-budget RIME adjudication.
- Implemented Phase 1 exact-K/geometry/cost evidence, Phase 2 `U-mixed-K` and
  causal gates, Phase 3 candidate/ablation matrix, and Phase 4 formal matrix.
- Corrected the mixed-K schedule to a per-video stateless 60-entry exposure
  with exact mean K=384.
- Corrected K192 to `fixed_floor_budget_position_only`; it cannot support a
  dynamic-budget claim.
- Corrected full-stack cost matching to exact `U-same-K` replay.
- Corrected the profiler to consume `effective_k`.
- Added an explicit Phase-4 authorization → Phase-2 receipt → budget protocol
  path/hash → checkpoint audit → terminal identity binding.
- Independent MAX audit found and corrected a deployment-blocking dense
  evidence SHA variable-name mismatch between the Phase-3 controller and
  Phase-3/4 submitters.
- Deployment preflight also corrected the one legacy Phase-1 gate invocation
  to run through `bash`, because that retained script is intentionally tracked
  without an executable bit.
- Rejected fabrication of a trained commit for the historical exact-uniform
  checkpoint whose surviving log records `git_head=unknown`. The Phase-1
  no-probe/probe cost pair now uses one byte-identical, SHA-bound checkpoint
  trained at `cb89586a92b8b0a8349ecc9551bc50aa97982360`; the launcher and seal
  require identical checkpoint SHA, trained commit, epoch, and EMA state.
  The no-probe arm drops only the registered probe/transition state that its
  configuration does not build, so the common heavy path is weight-identical.
- Retrieved the official released AdaTAD VideoMAE-S/ActionFormer checkpoint
  from the source-linked Google Drive file through its direct user-content
  endpoint. Its size is `200938640` bytes and SHA-256 is
  `21dbb9efe9f62d3089696c3c535edd27e8b8d9c14a06a21aac5738ec82bfab97`.
- Pre-registered the exact K/cost ladder, K384/K192 panel semantics,
  `weak_overlap` decoder, risk rule, O4 calibration gates, and 2 s / 8 s
  duration strata in both the submitting commit and submission manifest.
- The first released transaction at commit `57965bec` failed at the code gate
  before any experiment work because Slurm `--wrap` executes under `/bin/sh`
  and therefore rejected the Bash-only `source` builtin. All downstream jobs
  remained dependency-blocked. The submitter now explicitly enters
  `/bin/bash -lc` before the CUDA/Miniforge bootstrap; the failed root and
  scheduler records are retained as negative deployment evidence.
- The next code gate at commit `c0e7e036` reached the full remote Torch suite
  and stopped on two stale test fixtures: a `protected_e2e` RIME construction
  requested bridge scale `0.0` despite the registered `1.0` contract, and a
  no-padding ledger fixture omitted `irregular_dense_valid_len`. The fixtures
  were corrected to exercise, rather than violate, their production contracts;
  no model behavior or gate threshold was relaxed.
- The diagnostic rerun at commit `8667d057` cleared those failures and exposed
  one final stale floor-protocol fixture that constructed non-uniform RIME
  without the required official trainable ASFormer evidence source. The fixture
  now supplies that registered source contract; production behavior and every
  scientific gate remain unchanged.
- At commit `3f8e3ca1`, the exact remote suite passed all 158 tests. Its first
  unobstructed config-matrix run then exposed an over-broad generic assertion:
  the two evaluation-only Phase-1 paired cost profilers were being checked as
  train/evaluation-result configs. The matrix now keeps batch-size-one and saved
  predictions mandatory for trainable/formal-evaluation configs, while applying
  the stronger relevant contract to those two cost-only configs: test batch
  size one, zero loader workers, no saved accuracy predictions, no accuracy
  claim, and a byte-identical paired checkpoint identity.
- Current state: `implemented/tested`; remote code gate and Slurm deployment
  remain pending. No empirical or paper-ready claim has been made.

## 2026-07-28 — DUCA-RIME final transaction released

- Froze the physical protocol on implementation commit
  `f510741b32075c5c4e729d4207a549886a6dd064`; manifest SHA-256 is
  `5d28d1d37e698b5f17156245f55da62a82dc5b537c32fe70104f1be231e605d8`.
- Released the immutable fail-closed transaction at
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_four_phase_f510741b_20260728_094811`.
  Submission-manifest SHA-256 is
  `c74c351bad04dd7bfc6701ca5205e419116694d21736419776cec1f3cdb7ada6`.
- Code-gate job `1197889` completed successfully on the exact clean commit:
  all 158 focused tests passed and all 24 registered configs passed their
  stage-specific matrix.
- Slurm released Phase 1 (`1197890`), dense ActionFormer (`1197891`), and dense
  TriDet (`1197892`) into `RUNNING`. Phase 2 (`1197893`) and the Phase-3/4
  controller (`1197894`) remain dependency-gated as registered.
- Current scientific state is `experiment_running`, not
  `empirically_supported` or `paper_ready`.
- Early monitoring then found both dense jobs failed before their first
  optimizer update: their configs correctly set `dataset.val=None` and disabled
  validation intervals, but omitted the generic trainer's explicit
  `seal_eval_dataloaders_during_training=True` switch. The Phase-3 dependency
  failed closed. Phase 1 and all remaining jobs in this now-uncompletable
  transaction were canceled by exact job ID; the root and logs are retained.
- Added the missing seal to both dense configs and bound it in the launcher
  precheck, config matrix, and focused tests. This is an orchestration fix, not
  a model or scientific-protocol change.
- The next immutable transaction at commit `1ff54baf` passed its code gate.
  Phase 1 and dense ActionFormer ran normally, and ActionFormer reached update
  50. Dense TriDet then failed on its first backward pass because it alone still
  inherited reentrant VideoMAE gradient checkpointing; DDP reported a parameter
  marked ready twice. The transaction again failed closed, and jobs `1197975`,
  `1197976`, `1197979`, and `1197980` were canceled by exact ID.
- Dense TriDet now explicitly sets `with_cp=False`, as the already-working dense
  ActionFormer and protected/RIME bases do. The launcher and code gate reject
  any future dense reference with checkpointing re-enabled. This changes memory
  use, not the detector, objective, data, update count, or publication claim.
- The corrected dense precheck passed in Slurm job `1198049`. Diagnostic TriDet
  job `1198059` then completed 50 stable updates and was intentionally canceled;
  it is smoke evidence only.
- Released the active immutable transaction on commit
  `d9d454cd49a3e7a87694fc948601d00ff4043cb0` at
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_four_phase_d9d454cd_20260728_101256`.
  Protocol SHA-256 is
  `c4dfc31a64b56a93366c43443883df535e572eed38df63878fe11d3e00193a70`;
  submission-manifest SHA-256 is
  `ed374ae81991ca8241c0b01ab6588f13ea292b967b18a58115ec3f735440b038`.
- Code gate `1198113` passed. Phase 1 `1198114`, dense ActionFormer `1198115`,
  and dense TriDet `1198116` are running; both dense references passed update
  50. Phase 2 `1198117` and controller `1198118` remain correctly
  dependency-gated.
- Recorded `Pair-Risk Graph RIME` as a `discussed` post-v1 candidate. It is not
  designed, implemented, tested, or authorized to alter the frozen transaction.
- Monitoring snapshot `2026-07-28 10:40 CST`: Phase 1 job `1198114` failed
  closed after 12m44s in `finalize_duca_rime_inference_ledger.py`. Uniform-K384
  ledger line 64 for `video_validation_0000686` recorded
  `dense_valid_len=231`, `effective_k=unique_k=231`, but
  `backbone_input_k=padded_k=384`, violating the frozen no-padding cost
  contract. Its development localization JSON exists, but no evaluation
  receipt was emitted. Phase 2 job `1198117` is
  `DependencyNeverSatisfied`; controller `1198118` remains dependency-pending.
  Dense ActionFormer `1198115` and dense TriDet `1198116` continued into epoch
  9 with zero observed Traceback, OOM, non-finite-loss, or gradient-skip
  matches. No official-final data were opened and no downstream stage is
  authorized.
- Audited the unexpectedly high Phase-1 terminal mAP. The split manifest has
  200 `training` videos and uses a 180-ID block list to retain only 20
  certification-development videos. Historical checkpoint configs train on
  the same THUMOS `training` subset, and no checkpoint-specific exclusion
  manifest was found, so the 20-video measurements are high-confidence
  in-sample sanity controls. Pooled mAP was independently recomputed from the
  immutable predictions and exactly matched the terminal JSON; no
  self-normalization or `top_k=None` inflation was found. These numbers must
  not be compared with the upstream 69.03 official validation result.
- User froze a paper-responsibility contract: partial, training-domain,
  intermediate, small-subset, single-seed, unmatched, or missing-receipt
  results may be retained only as engineering status and must never be used to
  explain model performance or support the paper. Froze the full contract in
  mandatory-read `research-wiki/query_pack.md`. Until a complete comparable
  experiment exists, the correct empirical statement is that no paper-
  admissible conclusion is available; the only alternative is a self-contained
  theoretical analysis with explicit assumptions and limits.

## 2026-07-28 — CBCG-RIME external review absorbed and adjudicated

- Fully read and registered external review `U-PRO-CBCG-1`. Its overall
  freeze-v1 → same-K oracle → learned-head hold → causal/full-stack gates route
  was conditionally accepted.
- Corrected three stale execution claims in the review: the code gate has
  already passed; Phase 1 has failed closed on the no-padding ledger; Phase
  2/3/4 are dependency-blocked even though the two dense references were still
  running at the last verified snapshot. “Wait for the four-stage run to finish”
  and “immediately apply Patch A” are therefore not the current execution plan.
- Retained `CBCG-RIME` only as a working refinement of the `discussed`
  Pair-Risk Graph idea. It narrows generic pair risk to calibrated
  boundary-coverage failure on consecutive physical-selection edges, while
  preserving video-level risk for K selection.
- Recorded hard design blockers before implementation: path-to-edge regret
  attribution is underidentified without balanced perturbations and stability
  evidence; source/sink gap masses are not yet mathematically normalized;
  sparse complexity requires an enforced span cap; gap-only confounding,
  hard/soft energy equality, cross-fit calibration, and bit-exact risk-off
  behavior need explicit tests.
- The report's linked sandbox patches, hashes, and reported test count are
  unavailable in the repository and remain `PARTNER_CLAIM`; no code from them
  was applied and no implementation/test status was promoted.
- Standardized the comparison name to
  `AdapTok-inspired TAD budget allocation baseline`. The official AdapTok paper
  and repository still require direct provenance registration before
  publication.
- No empirical performance conclusion was added. The current project still has
  no `PAPER_ADMISSIBLE_RESULT`.

## 2026-07-28 — Risk granularity corrected and four-stage terminal state verified

- Resolved the user annotation about AdaTAD's 768-point input. The current
  controller does not compute whole-video risk: it pools cheap `[B,T,D]`
  evidence per training crop or inference sliding window and returns a
  window-level `[B,M]` utility/risk panel.
- Froze the unit distinction for future designs and writing: model decisions are
  per 768-candidate window; cross-fitting/statistics are grouped by video; costs
  are the per-video sum over actual heavy inputs for all windows, including
  overlap. `U-same-K` must replay each `(video_id, window_start_frame)` before
  aggregation.
- Compared three abstractions. Whole-video scalar K was rejected for v1 because
  it is not implemented and can wash out local short-action/boundary demand.
  Window-local K is the current recommended route. A hierarchical video prior
  plus window residual is deferred until window-local v1 establishes genuine
  headroom and cross-window correlation.
- Independently queried Slurm at `2026-07-28 14:47 CST`. Code gate `1198113`
  completed; Phase 1 `1198114` failed the no-padding ledger; dense ActionFormer
  `1198115` and dense TriDet `1198116` both failed after their 60-epoch training
  loops during checkpoint compaction; Phase 2 `1198117` and controller
  `1198118` remain `DependencyNeverSatisfied`.
- Both dense raw epoch-59 checkpoints exist, but neither terminal EMA,
  training/evaluation receipt, checkpoint binding, nor any Phase-1/2/3/4
  terminal receipt exists. The latest transaction is therefore terminally
  failed closed, not complete.
- Identified the dense post-training failure surface:
  `python tools/bata/compact_duca_rime_checkpoint.py` cannot resolve
  `from tools.bata import duca_p0_training` in the released environment.
  No remote state was changed.
- Preserved the raw checkpoints as possible inputs to a future, separately
  hash-bound salvage transaction. They are engineering artifacts, not positive
  experimental evidence.
- No model code or launcher was modified because the corrected risk/execution
  design is awaiting user approval under the brainstorming gate. No empirical
  performance conclusion was added.

## 2026-07-28 — Whole-video budget correction and H-RIME proposal

- The user clarified that offline cheap scanning can plan over the complete
  video even though AdaTAD continues to execute and detect on 768-candidate
  windows. Accepted this as a substantive correction to the earlier
  window-only final-model recommendation.
- Distinguished a total video quota `B_v` from a uniform per-window K. The
  proposed hierarchy is video-level budget prediction, joint window-level
  `K_vw` allocation, and existing within-window exact-K physical selection.
- Audited the current code contract: videos are flattened into overlapping
  window rows; the current controller, target/replay path, and ledger are
  per-window; no whole-video joint decision exists. H-RIME is recorded as
  `discussed/designed/awaiting_user_approval`, not implemented or tested.
- Froze truthful overlap accounting. Under the current no-cache backend, heavy
  work in overlapping windows is recomputed, so formal cost is
  `sum_w K_vw`; unique physical frames and the duplicate ratio are diagnostics,
  not compute savings.
- Promoted a held-out same-total-heavy-cost allocation oracle ahead of learned
  implementation. It must compare uniform, independent-window, and joint
  video-level allocation and stop the route if cross-window redistribution has
  no material high-IoU/short-action headroom.
- Reframed the failed four-stage RIME path as a required window-local baseline
  and infrastructure source, not automatically the final publication model.
  Phase-1 execution and dense checkpoint closure remain prerequisites.
- No model/launcher code or remote scheduler state was changed. No empirical
  performance claim was made.

## 2026-07-28 — Pro H-RIME report absorbed and implementation authorized

- Fully read and registered `U-PRO-HRIME-1`. The user approved the adjudicated
  route and authorized implementation.
- Accepted the main Approach-C architecture: shared full-video cheap scan,
  normalized total-video budget, exact MCKP per-window allocation, reuse of the
  existing exact-K selector, homogeneous-K heavy dispatch, and unchanged
  AdaTAD/NMS.
- Froze the corrected design at
  `docs/superpowers/specs/2026-07-28-hrime-v1-budget-conserving-design.md` and its
  implementation plan at
  `docs/superpowers/plans/2026-07-28-hrime-v1-implementation.md`.
- Did not accept the report verbatim. Numeric accuracy/calibration/cost gates
  remain proposals until a training/calibration-only pre-registration; raw caps
  are projected to reachable effective-K totals; official merge/NMS replay must
  validate the additive oracle; risk evaluation extends beyond ECE; endpoints,
  multiplicity, seed interpretation and deterministic MCKP ties are explicit.
- Independently audited the current repository. It has a reusable exact-K
  decoder and per-window replay, but flat window datasets, no whole-video
  planner/allocator, no grouped two-pass dispatch and no shared video scan.
  H-RIME was therefore `implementation_started`, not implemented/tested at that
  audit point.
- Independently verified the failed dense raw checkpoints:
  ActionFormer source job `1198115`, size `623799387`, SHA-256
  `cd92f3d499360c834f7ddd6ccfd5cba172c870bf6922de566b2b7e3878680e11`;
  TriDet source job `1198116`, size `411540059`, SHA-256
  `8940dbe756e8abfa3f7c8b042f3c658b26898d5c805d2876011a4e7510d11e12`.
  Both are epoch-59 raw checkpoints with EMA state but lack complete embedded
  commit/variant/seed provenance.
- Froze a new immutable salvage requirement. The failed root remains unchanged,
  and recovered artifacts must bind source job/path/size/hash plus explicitly
  external provenance.
- Scheduler energy fields were unavailable/zero and no trusted GPU monitor was
  active, so no energy claim is permitted.
- Directly registered official AdapTok and EVATok paper/code sources. Per-video
  adaptive allocation/routing is prior-art context, not H-RIME's novelty claim.
- No model performance claim was added. The project still has no
  `PAPER_ADMISSIBLE_RESULT`.

## 2026-07-28 — Stage-0 repair and H-RIME deterministic core implemented

- Repaired the short-window execution contract. Candidate requests now map to a
  homogeneous, quantum-aligned effective K before heavy execution; a 231-valid
  window maps `(192,256,384,512)` to `(192,224,224,224)`, and the heavy tensor
  width/ledger use 192 or 224 without replicated inactive tail.
- Changed all dense checkpoint compactors to clean-cwd module invocation and
  added a manifest-driven recovery tool/launcher. It validates the failed root,
  source job IDs, exact epoch-59 path/size/SHA/schema/EMA keys, writes only to a
  fresh recovery root, records missing embedded provenance honestly, keeps the
  original job state `FAILED`, and makes no energy claim.
- Added a dual-mode recovery DAG. `fresh_train` remains the default;
  `salvage` requires a frozen manifest/hash and only redirects the standard
  downstream checkpoint-evidence pointers. The failed transaction is never
  modified. Phase 4 is forced sealed in this recovery DAG.
- Implemented `hrime_exact_equality_mckp_v1`: canonical effective-K alias
  deduplication, reachable-cap projection, exact equality DP, frozen int64
  score quantization, deterministic objective/risk/lexicographic tie-break,
  and solver-input/assignment hashes.
- Implemented stable complete-video window groups, shared-scan receipt
  contracts, video budget-plan hashes, exact-K replay rows for the existing
  selector, and homogeneous-K dispatch/inverse-restoration plans. These are
  deterministic contracts, not yet a connected learned/shared-scan runtime.
- Independent code audit caught and corrected non-16-aligned K acceptance,
  fractional integer truncation, and insufficient feasible-set/plan/replay hash
  binding before merge.
- Local Python compilation, Bash syntax and 46 focused no-Torch tests passed.
  Torch-dependent tests are not claimed locally because the Windows host cannot
  load the CUDA-linked `torch` DLL; authoritative Slurm verification remains
  pending.
- No experiment performance claim was made. The same-total-cost oracle has not
  run, learned H-RIME is not implemented, and no paper-admissible empirical
  conclusion is available.
- The mandatory zero-context MAX deployment audit returned an initial `NO-GO`:
  the repaired fixed-window execution ledger still wrote a null physical
  protocol hash and did not distinguish raw/requested, reachable, realized,
  projection-unused and solver-unused budgets. No Slurm job was submitted.
- Corrected the blocker by requiring the exact physical-protocol SHA-256 in
  every Stage-0 ledger row, adding all five budget fields with explicit
  `window_fixed_request` / `stage0_engineering_window_execution` scope, and
  making the finalizer aggregate and fail closed on inconsistent budget truth.
  The expanded focused non-Torch suite then passed 64 tests. A fresh audit on the
  corrected clean commit remains mandatory before deployment.
- The clean-commit re-audit returned a second `NO-GO`: the selector producer was
  strict, but `run_duca_rime_phase1_uniform_eval.sh` did not pass the protocol
  hash or demand explicit budget truth from the finalizer, so a legacy row could
  still be sealed. Corrected the consumer chain: the launcher now requires and
  hash-checks the protocol manifest, invokes the finalizer as a module with the
  expected hash plus `--require-explicit-budget-truth`, and the parent pipeline
  exports the binding before K384/K192. The expanded focused non-Torch suite
  now passes 66 tests. Slurm remains unsubmitted pending another clean audit.
- A third zero-context MAX audit on commit `534da568` returned `GO`. A
  commit-bound physical protocol and immutable two-backend salvage manifest
  were frozen, but the first recovery submission met the account-level
  `AssocMaxSubmitJobLimit` after four held jobs had been created. The nested
  `sbatch` failure did not trigger the parent-only `ERR` cleanup trap.
- Canceled unreleased jobs `1199974`--`1199977` and the route-local stale
  `DependencyNeverSatisfied` jobs `1198117`/`1198118`; no unrelated job was
  changed. Enabled Bash `errtrace` in the four-phase, Phase-3 and Phase-4
  transactional submitters so future nested failures cancel their held prefix.
  The partial submission root remains non-reusable evidence; no experiment or
  performance claim was created.
- Fresh MAX audit returned `GO` on exact clean commit
  `902168a12bc92babd62b6cb1877ce7137f56cea0`. Froze the new commit-bound
  physical protocol (`1823826b...e7e34e`) and salvage manifest
  (`b4f5b7fd...d85a0e`), then atomically released recovery jobs
  `1199978`--`1199983` under
  `duca_rime_recovery_902168a1_20260728_183709`.
- The submission manifest is
  `fd6fef65ac01e7830c6b5e337684b19a3bad65c1432f819cfecb32e83dfefb85`;
  the receipt is released, dense recovery is explicitly engineering-only, the
  original failed jobs remain failed, Phase 4 is disabled, and official-final
  remains sealed. The first snapshot is scheduler-pending with no terminal
  receipts, so no empirical claim is available.

## 2026-07-28 — H-RIME Stage-1 oracle and execution-proof surface implemented

- Implemented the complete development-only Stage-1 strategy matrix:
  same-total uniform, independent window RIME, joint GT oracle, joint allocation
  with uniform positions, and feasibility-preserving shuffled null. Every video
  uses the exact same reachable total effective K across strategies.
- Added an explicit pre-execution preregistration builder. It freezes one
  primary endpoint, video bootstrap, intersection-union multiplicity family,
  noninferiority/materiality gates, guardrails, surrogate thresholds, MCKP
  numeric/tie contracts, evaluator semantics and official-final exclusion.
  No scientific threshold has a result-derived default.
- Connected each Stage-1 replay to the existing exact-K selector while preserving
  strict RIME-full parameter keys and shapes. Oracle permission, decision role,
  GT provenance, requested/effective K, no-padding execution and video-total
  budget truth all fail closed.
- Extended the test runtime to produce a machine-verifiable execution receipt:
  exact sliding-window dataset coverage, model-forward batch count, window
  counts by video, distributed aggregation, pre/post-NMS proposal counts,
  per-video NMS call count, post-NMS prediction SHA-256, official evaluator
  call/success, resolved config identities and implementation source hashes.
  The terminal also records `strict_exact_v1` checkpoint compatibility.
- Rejected negative oracle risk weights at both freeze and validation entry
  points, and rejected prediction artifacts missing any expected development
  video instead of silently interpreting them as empty.
- Local Python compilation and Bash syntax passed. The pure/non-Torch focused
  suites passed 61 Stage-1/RIME tests plus 23 repository-mandated C3 regression
  tests. The new runtime-receipt and strict model-loading tests remain
  `remote_torch_pending` because the Windows host cannot load the CUDA-linked
  Torch DLL.
- A fresh independent diff review additionally required that the shuffled null
  not collapse to the joint oracle for an entire budget anchor. Planning now
  records every degenerate video and fails before writing its output root if no
  non-identity, histogram-preserving feasible allocation exists for that anchor.
- Stage-1 is `implemented/local_non_torch_tested`, not executed. It authorizes no
  learned H-RIME training until Stage-0 closes, a clean-commit remote test and
  independent deployment audit pass, the complete preregistration is frozen,
  and the full development oracle receipt passes. No paper-admissible empirical
  conclusion is available.

## 2026-07-28 — Stage-1 remote verification and Stage-0 recovery failure absorbed

- Pushed exact Stage-1 implementation commit
  `577e748ffb3fe452a57094d3d0bb5f022c32f739` and checked it out cleanly on the
  remote Linux/Torch environment.
- Fourteen targeted remote Torch tests passed in 48.62 seconds: strict
  RIME-full-to-Stage-1 architecture loading, short-window replay,
  cross-window aggregation/NMS/official-evaluator execution receipts, the
  Stage-1 oracle core, and exact expected prediction-video keys. Stage-1 is now
  `remote_torch_tested`, not `experiment_run`.
- Rechecked recovery jobs `1199978`--`1199983`. The code gate passed. Phase 1
  failed because `tools/test.py` received the base config's relative VideoMAE
  initialization path. Both dense salvage arms failed with exit code 126
  because the generated wrapper directly executed a `100644` shell script.
  These are deployment failures, not model or performance evidence.
- Phase 2/controller were verified as `DependencyNeverSatisfied` and canceled by
  exact IDs `1199982` and `1199983`. The failed root, logs, released submission
  receipt, and original failed job states remain untouched.
- Implemented the deployment repair: Phase-1 dense evaluation now requires,
  hash-checks, passes and records the exact absolute VideoMAE initialization;
  the recovery submitter invokes both salvage commands through explicit Bash;
  the salvage launcher mode is restored to `100755`. Bash syntax and 36 focused
  launcher/salvage tests pass locally.
- Independent repair review found that salvage checked and used the VideoMAE
  initialization but did not explicitly carry its SHA-256 into the terminal
  recovery receipt. Added a second in-process hash check plus explicit path/hash
  fields to the source/recovery evidence; this closes provenance rather than
  changing execution.
- This repair still requires a clean commit, remote prechecks, an independent
  audit, new commit-bound manifests and a fresh immutable Slurm transaction.
  No paper-admissible empirical conclusion is available.

## 2026-07-28 — Corrected Stage-0 recovery transaction released

- Committed and pushed the deployment repair as exact source commit
  `0ab242f31be8de7b7da806b645d3aa60d02d8d88`.
- Local compilation/Bash checks and 82 focused tests passed. A clean remote
  checkout then passed the same 82 Linux/Torch tests. An independent
  clean-commit review returned code-level `GO`.
- Remote `PRECHECK_ONLY` passed for Phase-1 dense, exact-uniform and paired-cost
  launchers. Both salvage prechecks reloaded the original raw checkpoints,
  verified 499 ActionFormer and 462 TriDet EMA/state keys, wrote nothing, kept
  source jobs `FAILED`, and confirmed official-final exclusion.
- Froze the commit-bound physical protocol with SHA-256
  `2f11c12d62451c7ec41b54ac889058617f56f889e6f289cfe865a47eb03ff9f9`
  and the fresh two-backend salvage manifest with SHA-256
  `faab636144d0855f2d8f26d6c7298459302b3c84508bdc2da24b1b864013772d`.
- Atomically released jobs `1200135`--`1200140` at
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_0ab242f3_20260728_201613`.
  Submission-manifest SHA-256 is
  `b996543dfe57bc3678799591f38f0e96e76da971eb8d5a4f7a4edbb15aa3d04d`.
  The first snapshot has the code gate priority-pending and every child under
  exact fail-closed dependencies.
- Dense salvage remains
  `engineering_dense_reference_recovery_not_method_evidence`; original jobs and
  roots remain failed/immutable. Phase 4 is disabled, official-final is sealed,
  and no paper-admissible empirical conclusion is available.

## 2026-07-28 — Corrected Stage-0 recovery transaction failed closed

- Rechecked exact transaction
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_0ab242f3_20260728_201613`
  at `21:02 CST`. The deployed checkout remains clean at exact commit
  `0ab242f31be8de7b7da806b645d3aa60d02d8d88`; the submission, physical-protocol
  and salvage-manifest hashes remain exactly registered.
- Code gate `1200135` completed. Phase 1 `1200136` failed in the exact-uniform
  evaluator because its actual `tools/test.py` command did not override the
  base config's repository-relative VideoMAE initialization, despite the
  absolute path being required elsewhere. This is a runtime/precheck coverage
  defect, not performance evidence.
- ActionFormer/TriDet salvage `1200137`/`1200138` compacted raw EMA checkpoints
  and ran their engineering evaluations, then failed structured evidence
  finalization: the evaluator subset was frozen as `training`, while
  `tools/test.py` classified neither salvage role and therefore expected
  `validation`. No terminal dense checkpoint evidence or recovery receipt was
  produced.
- Phase 2/controller `1200139`/`1200140` were verified as
  `DependencyNeverSatisfied` and canceled by exact ID. No unrelated job was
  changed.
- Required Phase-1, dense, Phase-2 and Phase-3 terminal receipts are absent.
  Partial salvage and evaluation artifacts remain engineering diagnostics only.
  Phase 4 stayed disabled, official-final stayed sealed, and no paper-admissible
  empirical conclusion is available.

## 2026-07-28 — Recovery-v3 evaluator contracts implemented locally

- Treated the user's repair/redeploy instruction as approval of the already
  recorded minimal design and froze the implementation addendum at
  `docs/superpowers/specs/2026-07-28-stage0-recovery-v3-contract-repair-design.md`.
- Repaired exact-uniform Phase-1 evaluation so both budgets require,
  hash-check, resolve, actually override and receipt-bind the absolute VideoMAE
  initialization.
- Added a separate dense-reference protocol predicate and validator. This avoids
  the invalid shortcut of adding dense protocols to ordinary RIME formal
  routing, whose later payload requires trainable selector contracts absent
  from dense references.
- Dense evaluation now validates its engineering role, backend, no-selector
  path, 768-window dense execution, `with_cp=False`, training development
  subset/block list, saved predictions, pretrain identity and official-final
  exclusion before emitting a dedicated terminal schema.
- The new regression tests failed before implementation and passed afterward.
  Local Python compilation, Bash syntax, launcher, salvage, RIME contract,
  H-RIME, prediction-metric and mandatory C3 regressions completed with
  `96 passed`.
- State remains `implemented/local_tested/remote_pending`. No new experiment was
  yet submitted, no result was interpreted, and official-final remains sealed.

## 2026-07-28 — Recovery-v3 remotely verified and redeployed

- Pushed exact implementation commit
  `bbf051410839f7bec36b0f2cc085de0cd5041cad`; an independent MAX deployment
  audit returned `GO`.
- Created a fresh clean remote checkout. Slurm code preflight `1200405`
  completed 193 authoritative Linux/Torch contract tests and emitted its gate
  receipt.
- The first combined launcher preflight `1200441` intentionally failed closed
  because the production salvage output parent must exist while the production
  transaction root must remain absent before submission. No production root or
  salvage output was created. A separate same-source, same-commit preflight
  manifest preserved both contracts.
- Full launcher preflight `1200462` completed Phase-1 dense, uniform K384,
  uniform K192, paired-cost and both salvage checks. The two salvage checks
  audited 499 ActionFormer and 462 TriDet EMA keys, wrote no output, preserved
  source jobs as `FAILED`, and excluded official-final.
- Froze production physical protocol
  `69a9cc0b85aaa647a5641f3c00eadd9b8405e8435d3ed5820aae3949df210f4c`
  and production salvage manifest
  `f7c09b017a4e973211c0f816f55de506d68046801886066dfff3555f15942aef`.
- Atomically released jobs `1200483`--`1200488` at
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_bbf05141_20260728_215335`.
  Submission-manifest SHA-256 is
  `53a633c162dd69ec3bdfd291e8df97d8e79619d9b688808d0dfad36127abc265`.
- At `22:04 CST`, production code gate `1200483` was complete; Phase 1 and both
  salvage arms were running; Phase 2/3 remained dependency-held. The deployment
  monitor was rebound to these exact identities. Phase 4 remains disabled,
  official-final remains sealed, and no paper-admissible empirical conclusion
  is available.

## 2026-07-28 — Recovery-v3 failed closed; recovery-v4 bridge designed

- Production code gate `1200483` completed. Dense salvage jobs
  `1200485`/`1200486` then both failed before inference at the exact
  `tools/test.py` guard `formal evaluation checkout differs from
  DUCA_EXPECTED_COMMIT`.
- The checkpoint salvage itself had passed and the immutable sources remained
  unchanged. The launcher required `DUCA_RIME_EXPECTED_COMMIT` but omitted the
  explicit bridge to the evaluator's canonical `DUCA_EXPECTED_COMMIT`.
- Because the transaction could no longer produce all required receipts, Phase
  1 `1200484`, Phase 2 `1200487`, and controller `1200488` were canceled by
  exact ID. No unrelated job was changed. Phase 4 was never opened.
- Froze the recovery-v4 design: explicitly export the canonical commit variable
  and make `PRECHECK_ONLY` execute the same environment lookup and Git identity
  comparison as formal evaluation. Silent evaluator fallback was rejected
  because it would weaken fail-closed launcher diagnostics.
- Recovery-v3 remains immutable failed engineering evidence. No performance
  result was read or interpreted, and no paper-admissible empirical conclusion
  is available.
- Implemented the recovery-v4 bridge in the salvage launcher. It overwrite-
  exports the canonical evaluator commit and runs an evaluator-equivalent
  environment/Git identity probe before both precheck and actual execution.
  The new regression was red before the fix and green afterward; local
  compilation, Bash syntax and the expanded focused suite passed `97` tests.

## 2026-07-28 — Recovery-v4 remotely verified and redeployed

- Pushed exact implementation commit
  `1b44fe3a35042d28c55b9e838f69107bd1461810`; an independent clean-commit
  deployment audit returned `GO`.
- Slurm code preflight `1200583` completed 194 authoritative Linux/Torch
  contract tests. Full launcher/runtime preflight `1200601` then passed all
  Phase-1 and dense-salvage checks, including a deliberate stale canonical
  evaluator-commit injection that the new overwrite-bridge corrected before the
  evaluator-equivalent identity probe. Precheck wrote no production output.
- Froze production physical protocol
  `2d416cddd923aa46693ad5361979558e845252947fcb50491cd5cc6c6e70be8c`
  and production salvage manifest
  `2fb3f9c1a7623e059f855227c34d7614ef2fb6c9e29ee5461e29b4cf5f107d11`.
- Atomically released jobs `1200627`--`1200632` at
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_1b44fe3a_20260728_221502`.
  Submission-manifest SHA-256 is
  `ca72b350ccd7227671554e6e413281cd7059c97f5c3161e2ed93c7a087549767`;
  released receipt SHA-256 is
  `eca9e24a06ad7ff2a187066f2f255eb4d764a2f3c5362734444463fa6c128449`.
- Code gate `1200627` and both salvage jobs `1200629`/`1200630` completed with
  exit `0:0`. Both salvage arms crossed the former commit-identity failure,
  completed formal engineering evaluation and emitted checkpoint evidence plus
  passing recovery receipts. The receipts retain their original source jobs as
  `FAILED` and restrict claims to engineering dense-reference recovery.
- At `22:25 CST`, Phase 1 `1200628` was running; Phase 2 `1200631` and Phase-3
  controller `1200632` remained dependency-held. Monitoring was rebound to these
  exact identities. Phase 4 remains disabled, official-final remains sealed,
  and no paper-admissible empirical conclusion is available.

## 2026-07-28 — Recovery-v4 Phase 1 failed closed on mask handoff

- Phase 1 job `1200628` failed at `22:32:51 CST`, exit `1:0`, during the first
  actual exact-uniform K384 forward. No Phase-1 terminal receipt was produced.
- Exact exception:
  `ValueError: dynamic RIME backbone requires an aligned [B,K] mask` from
  `BackboneWrapper._prepare_dynamic_temporal_bucket`.
- The traceback shows `ActionFormer.forward_test` called
  `self.backbone(inputs)` without the dataset mask because the exact-uniform
  baseline has no `duca_rime_physical` selector. Its backbone nevertheless has
  the dynamic temporal bucket enabled and requires that aligned mask.
- The existing uniform launcher precheck only parsed and asserted config,
  protocol, budget and pretrain identity. It did not construct the model or run
  the tensor-forward boundary, so it could not catch the mismatch.
- Phase 2 job `1200631` became `DependencyNeverSatisfied`; Phase-3 controller
  `1200632` remains dependency-held. The two dense salvage receipts remain
  passing engineering recovery evidence. Phase 4 remains disabled and
  official-final remains sealed.
- This diagnosis used only execution state, traceback and tensor-contract
  provenance. No intermediate or terminal performance value was reported or
  interpreted.

## 2026-07-28 — Recovery-v5 mask handoff implemented

- User authorized a complete correction, redeployment, and bounded self-healing
  monitor.
- Implemented one shared ActionFormer/TriDet backbone handoff. A dynamic
  temporal bucket now always receives the exact aligned detector mask; ordinary
  backbones preserve the legacy invocation; physical RIME paired with a
  non-dynamic backbone fails closed.
- Added a focused runtime contract test for both backends, AST coverage of all
  train/test call sites, Slurm code-gate inclusion, and frozen dynamic-bucket
  assertions in the Phase-1 uniform precheck.
- Python compilation, Bash syntax and `git diff --check` passed. Two independent
  read-only patch audits returned `GO`. Local Torch execution is explicitly not
  claimed because Windows failed loading `c10.dll`; the authoritative
  Linux/Torch gate is pending.
- The repair changes no model objective, budget, split, checkpoint,
  hyperparameter, metric or paper claim. Recovery-v4 remains immutable failed
  engineering evidence, Phase 4 is disabled, official-final is sealed, and no
  paper-admissible empirical conclusion is available.

## 2026-07-28 — Recovery-v5 remote gate queued; monitor upgraded

- Pushed and cleanly installed exact commit
  `74de620d8fafc365694aa1f400318a401add3ecc`.
- The focused Linux/PyTorch detector-backbone mask suite passed all 10 tests.
- Canceled only recovery-v4 dependency-impossible jobs `1200631` and `1200632`
  by exact ID; old roots, logs and valid dense salvage receipts remain
  immutable.
- Initial Slurm gate `1201029` waited on `AssocGrpGRES`. A CPU-only replacement
  was attempted to avoid unnecessary GPU use, but cluster submission policy
  rejected it before creating a job because the only partition requires an
  explicit GPU request. The standard 1-GPU gate was restored as `1201057`; it
  remains queued on `AssocGrpGRES`, not failed.
- Prepared a fresh deployment script at
  `/data/run01/sczc063/yuzibo/rime_prerequisites/deploy_duca_rime_recovery_v5_74de620d.sh`
  with SHA-256
  `780cd27f36a68d307a4fd90168a96dfe1db3a34e530c9f332a594e78a3b769a1`.
  It targets the fresh root
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_74de620d_20260728_233000`
  and cannot run before the commit-bound gate receipt passes.
- Upgraded automation `duca-rime` from 30-minute passive observation to a
  15-minute bounded self-healing contract: one retry per unique deterministic
  protocol-preserving engineering failure, new commit/manifests/root required,
  and fail-closed escalation for repeated, ambiguous, scientific, data,
  numerical or model-quality failures. Phase 4 and official-final remain sealed.

## 2026-07-28 — Recovery-v5 preflight passed; bounded retry released production DAG

- Slurm preflight `1201057` completed with exit `0:0`. Its receipt binds exact
  source commit `74de620d8fafc365694aa1f400318a401add3ecc` and has SHA-256
  `740bc46cff9db814dc8e6c1ae5ad9051db6c6bc9503979969515268462cf0af3`.
- The first hash-frozen deployment invocation failed before producing any
  protocol, manifest or production root. The exact traceback was
  `FileNotFoundError: data/thumos-14/annotations/thumos_14_anno.json`; a clean
  clone had no ignored runtime annotation/video symlinks. The failure log and
  unused root identity ending in `20260728_233000` remain preserved.
- Classified the unique failure as
  `missing_runtime_data_symlinks_before_protocol_freeze`. Restored only the
  established checkout-local symlinks to immutable datasets, verified both
  targets and reverified that the deployed source checkout remained Git-clean.
  This changed no model, protocol, loss, budget, split, checkpoint,
  hyperparameter or metric.
- Consumed the one allowed automatic retry for that signature. Retry script
  `/data/run01/sczc063/yuzibo/rime_prerequisites/deploy_duca_rime_recovery_v5_74de620d_retry1.sh`
  has SHA-256
  `22698937ff31b4fc696df2a73b0e737c2eeca5fdcd660b8bfcd5c9b46faba635`.
- Released fresh immutable root
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_74de620d_20260728_235100`
  with physical-protocol SHA-256
  `65db63c4b3ebb7f407099efe0f3a97670c19359b0a6f680cb44114645cb3b244`,
  salvage-manifest SHA-256
  `af3c466e6d2f61ea9284de540e5b353bbecb1609c0fb40face172c7d1e642acf`,
  submission-manifest SHA-256
  `6290a5f0bbe15128f8313fec1aaf3003b306e1e7b7c6a3259a3fd21b46beaeb5`,
  and released-receipt SHA-256
  `81d5ce399a0568df744908f7068dc53746b00b7b9f5b4df08d4a5d429c54c95e`.
- Atomically released jobs `1201169`--`1201174`: production code gate,
  Phase 1, ActionFormer salvage, TriDet salvage, Phase 2 and Phase-3 controller.
  Production code gate `1201169` completed with exit `0:0`; its exact-commit
  receipt SHA-256 is
  `7de03703c23ae79772b8598bea7de3fbaa0db85bffc58d71f467e9f7294045e4`.
  Phase 1 `1201170` and both salvage jobs `1201171`/`1201172` are running;
  Phase 2 `1201173` and controller `1201174` remain dependency-held. Phase 4
  remains disabled, official-final remains sealed, and no performance value or
  paper-admissible empirical conclusion was produced.

## 2026-07-29 — Recovery-v5 dense salvage arms completed

- ActionFormer salvage `1201171` completed with exit `0:0`. Checkpoint-evidence
  SHA-256 is
  `72699b01de350c36a2fa6243215aad0bc0294c6c21cf68c07565e1e4d6df9832`;
  terminal recovery-receipt SHA-256 is
  `2a245ad1209fe8986da612754fbd47c68656e9c136ecd0e448798319232cf5bf`.
- TriDet salvage `1201172` completed with exit `0:0`. Checkpoint-evidence
  SHA-256 is
  `5549264c89dccfc7adec06e7ea14c41c1650d07879a138be8779efab96a5689c`;
  terminal recovery-receipt SHA-256 is
  `37e6980daecc3b77ae406d3be0b5cfaca43fc5fb39e3f389f095b0ec2246f3a1`.
- Both terminal receipts bind exact recovery commit
  `74de620d8fafc365694aa1f400318a401add3ecc`, retain original source jobs
  `1198115`/`1198116` and source commit
  `d9d454cd49a3e7a87694fc948601d00ff4043cb0` as failed evidence, do not
  reclassify either source job, exclude official-final, and retain the claim
  scope `engineering_dense_reference_recovery_not_method_evidence`.
- Phase 1 `1201170` remains running. Phase 2 `1201173` and Phase-3 controller
  `1201174` remain dependency-held. This update is `ENGINEERING_STATUS`; no
  performance value was inspected or interpreted.

## 2026-07-29 — Recovery-v5 Phase 1 failed; Recovery-v6 engineering repair implemented

- Phase 1 `1201170` failed with exit `1:0`; no Phase-1 terminal receipt exists.
  The exact-uniform K192 short-window path reached the real VideoMAE adapter,
  where an eight-token runtime temporal axis was reshaped using nominal
  `temporal_size=192`. The exact terminal exception was
  `RuntimeError: shape '[-1, 192, 10, 10, 96]' is invalid for input of size
  1075200`.
- Registered unique failure signature
  `vit_adapter_static_temporal_axis_on_dynamic_k_bucket`. This is deterministic,
  reproducible and protocol-preserving engineering correctness, so it is
  eligible for one bounded fresh recovery.
- Canceled only dependency-impossible jobs `1201173`/`1201174` by exact ID.
  Recovery-v5 root, scheduler state, production gate and both valid dense
  salvage receipts remain immutable. Phase 4 was never opened.
- Recovery-v6 derives the adapter's runtime temporal count from
  `N / (h * w)`, rejects non-integral token geometry, and does not mutate the
  configured nominal temporal size. Added red-before/green-after runtime and
  failure-contract tests and added `vit_adapter.py` to the Slurm code-gate
  compilation surface.
- Static compilation, Bash syntax and `git diff --check` pass. Remote Torch
  verification, exact commit publication, commit-bound manifests and a new
  transaction root remain pending. An independent read-only audit returned
  `GO` for the bounded diff, confirmed that active dynamic configs use
  `VisionTransformerAdapter`, and found no scientific-protocol change. No
  performance value was inspected or interpreted.

## 2026-07-29 — Recovery-v6 exact commit published; Slurm gate queued

- Published exact repair commit
  `5a599e909aca58751711979e8c9e5b68ab6cab72`.
- A direct remote GitHub clone encountered a transient TLS termination before
  creating a checkout. Built a complete bundle from the already-pushed branch,
  SHA-256
  `6e4052a5ae4f8e74a2cbfa12303415712b5b41b84906ff1b5c27fd8853edca48`,
  and used it to create clean detached checkout
  `/data/run01/sczc063/yuzibo/OpenTAD_DUCA_HRIME_5a599e90`.
- Restored the required checkout-local annotation/video symlinks to the
  established immutable datasets, verified both targets, exact HEAD and clean
  Git status before submission.
- Preflight submission script SHA-256 is
  `3e02fabb176d93d5dc125992c55bd80e0188fd85519a2dcd2b0be240e7903a35`.
  It submitted Slurm code gate `1201390` at
  `/data/run01/sczc063/yuzibo/rime_preflight/duca_rime_recovery_v6_5a599e90_20260729_003219`;
  the first snapshot is scheduler-pending on priority.
- No Recovery-v6 production protocol, salvage manifest, submission manifest,
  root or DAG exists yet. Phase 4 and official-final remain sealed.

## 2026-07-29 — Recovery-v6 preflight passed; production DAG released

- Slurm preflight `1201390` completed with exit `0:0`; exact-commit receipt
  SHA-256 is
  `bef1f6446ceab601b910bfee0f21d0d0d95a297e426455bf682a064f3f4fb2be`.
- Deployment script SHA-256 is
  `f44ff20e8a7acf134581fb460c1eb1188da02070c09aff7bf2bb9cb20e89c8f9`.
  It generated fresh commit-bound physical protocol
  `94ebe87782e5375afe71ed1506f13e3c812d105f018a3ccdf24eea450f0a35f9`,
  production salvage manifest
  `61f7cfec47b0a467b1f8e616487686937b51bc96098ca15e776c31ff024fa7f0`,
  submission manifest
  `759fe6e97b10edf03128b6b2244dbab6cbc3e5009d7fdf1d8d9f5319d5d3375a`,
  and released receipt
  `007cee9134ebdba67563681b6bbc3a5e1cecbcf7ad998c688d1cd131bcdbd691`.
- Released jobs `1201416`--`1201421` at
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_5a599e90_20260729_003600`.
  They are production gate, Phase 1, ActionFormer salvage, TriDet salvage,
  Phase 2 and Phase-3 controller with the exact registered dependency map.
- The first scheduler snapshot had production gate `1201416` running and every
  child dependency-held. Phase 4 remains disabled, official-final remains
  sealed, and this is only `ENGINEERING_STATUS`.
- Production gate `1201416` subsequently completed with exit `0:0`; its
  exact-commit receipt SHA-256 is
  `34152cfe1fb6c008f4cd20d11f3ed1c6dd19f980caf45d2b1069a029a065146d`.
  Phase 1 `1201417` and salvage jobs `1201418`/`1201419` subsequently entered
  `RUNNING`; Phase 2 `1201420` and controller `1201421` remain dependency-held.

## 2026-07-29 — Recovery-v6 dense salvage arms completed

- ActionFormer salvage `1201418` completed with exit `0:0`; checkpoint-evidence
  SHA-256 is
  `f5b4f231686fe9aec9e79545ee2eba010d4004e07d285dae05830bb2ede8d7a3`
  and recovery-receipt SHA-256 is
  `45590ba3a02a06526cf1ad16d217c33c98e77d2c24aeea7509a8a1bee2adcbf1`.
- TriDet salvage `1201419` completed with exit `0:0`; checkpoint-evidence
  SHA-256 is
  `d979e854a3f75f49f58c5d168bcee5eb5716bcdcb1af6cb5f2595b9a21669327`
  and recovery-receipt SHA-256 is
  `ba3e7ddaa310bdf36a78723738545de2b99c76560f28d97742c257ee7538257a`.
- Both receipts bind exact Recovery-v6 commit, preserve original source jobs
  `1198115`/`1198116` as failed without reclassification, exclude
  official-final, and retain engineering-only claim scope.
- Phase 1 `1201417` remains running; Phase 2 `1201420` and controller `1201421`
  remain dependency-held. No performance value was inspected or interpreted.

## 2026-07-29 — Recovery-v6 failed closed at protected full-model gate

- Phase 1 `1201417` failed with exit `1:0`; no Phase-1 pipeline receipt exists.
  The terminal schema was `ProtectedPhysicalGateFailure`, status
  `p1_p2_full_model_gate_failed`, with exact error `protected physical
  full-model gate failed: exact-uniform physical and selected-axis detector
  losses disagree`.
- Registered failure signature
  `protected_physical_exact_uniform_selected_axis_loss_equivalence_gate_failed`.
  The immutable log path is
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_5a599e90_20260729_003600/logs/rime-phase1-1201417.out`;
  SHA-256 is
  `0b9aedc943139e024939fa16bf5cf3007c7ae387e74f04bdae823551e3baee29`.
- This is a frozen scientific/admission gate failure, not a deterministic
  protocol-preserving engineering defect. The bounded monitor made no code or
  protocol change and did not automatically retry.
- Canceled only exact dependency-impossible jobs `1201420`/`1201421`. Production
  gate `1201416` and both dense-salvage terminal receipts remain valid and
  immutable. Phase 4 was never opened and official-final remains sealed.
- This record is `ENGINEERING_STATUS` only. No model-quality or paper-admissible
  performance conclusion was drawn.

## 2026-07-29 — Recovery-v6 gate premise and thresholds independently audited

- Rechecked the exact gate implementation, ActionFormer physical head, integer
  exact-uniform anchors, commit history, tests and immutable failure log.
- The operational stop remains correct: the frozen gate rejected the run, so no
  child experiment or official-final access was allowed.
- The scientific root cause is not established. Integer round-half-to-even
  anchors generally induce a piecewise-linear, non-affine selected-to-physical
  map. The head's local stride approximation, center sampling, regression-range
  assignment, normalized offsets and IoU/GIoU objective do not guarantee exact
  loss equality with the legacy selected-axis parameterization.
- Commit `ce5d03ebf` introduced the `1e-4` loss/proposal and `1e-6` target/score
  tolerances without a registered derivation, null-repeat calibration or
  FP16/FP32 error study. They are engineering tolerances, not validated
  scientific thresholds.
- The failure artifact omits full versus short-padded provenance, offending loss
  key, error magnitude, applicable threshold and FP32 replay. It cannot
  distinguish numerical miss, semantic implementation bug and over-strong gate
  premise.
- Corrected verdict:
  `gate_failed_closed / root_cause_not_identified / gate_validity_under_review`.
  No performance number or paper claim is authorized.

## 2026-07-29 — Gate contract root-cause class established; diagnostics committed

- The result is in post-failure scientific-contract analysis, not model
  performance analysis. No paper-admissible empirical result exists.
- The universal gate premise is now rejected as a theorem for the implemented
  general case. At `T=768, K=384`, round-half-to-even integer anchors produce
  382 steps of two and one step of three, hence a non-affine coordinate warp.
  The physical head's target assignment and IoU/GIoU loss are not guaranteed to
  be scalar-loss conjugate to the selected-axis formulation under that warp.
- The exact observed component in job `1201417` remains unresolved. Its
  immutable log records no window role, loss key, error magnitude, threshold
  comparison or FP32 replay. The correct state is
  `universal_loss_equivalence_premise_invalid /
  observed_mismatch_component_unresolved`.
- Diagnostic-only commit
  `69136de3ed8d8f977c78cfe5258dae3d57f7e238` records affine applicability,
  per-loss errors, unchanged tolerance, AMP state and a separate diagnostic-only
  FP32 replay; failures produce exclusive JSON outside the Git worktree.
- The commit changes no model, loss, budget, threshold, data, checkpoint,
  metric, or admission outcome. Compilation, `git diff --check`, and the
  focused diagnostic/evidence suite passed with `32 passed`.
- The diagnostic commit and research analysis were pushed to
  `codex/duca-rime-20260727`; direct remote-ref verification confirmed the
  published branch identity before preparing the Pro review.
- One bounded Pro review is now warranted to select the paper architecture and
  replace the invalid universal equality premise with a justified scientific
  gate. No new transaction or Phase 4 access is authorized by this analysis.

## 2026-07-29 — Pure selected-axis architecture accepted and acquisition-v2 implemented

- Fully read and registered `U-PRO-PURE-PLUGIN-1`. Accepted its
  `CONDITIONAL GO`: organizational Route C, Route A as the only paper mainline,
  physical-time head as an isolated integration/diagnostic arm, and replacement
  of the invalid general loss-equivalence gate.
- Implemented selected-axis GT mapping, standard-head restoration,
  exact-once proposal remap before official NMS, full implemented-map
  diagnostics, and selected-axis ActionFormer/TriDet configs. The existing
  fully gated `AnchorFreeHead` was not split into subclasses because the
  selected-axis route never enables its physical branch; subclass extraction
  is deferred until a physical integration experiment is separately
  authorized.
- Implemented a Slurm/CUDA acquisition-v2 runtime producer. It strict-loads
  real dense checkpoints into standard and selected heads, checks full/short
  exact-K execution, GT roundtrip, NMS ordering, AMP nulls and complete
  state/RNG/debug restoration.
- Removed external JSON finalization. The runtime producer now writes the final
  finite, exclusive, content-bound receipt directly. Verification re-hashes
  config, checkpoint, data, split, code-gate, calibration, scientific protocol
  and NI-margin-source artifacts and re-binds exact Git/runtime identity.
- Added pre-candidate scientific-protocol anchoring to a clean commit/tree,
  fresh candidate output root, training/calibration NI-margin source, endpoint,
  multiplicity, guardrails and stopping rules. Phase 1, Phase 3 and all Phase 4
  entrypoints require the same verified admission-v2 prerequisite.
- Local compilation, Bash syntax and `git diff --check` passed. The pure
  admission/geometry/stage/launcher suites plus mandatory C3 regressions passed
  `95 tests`; one warning is unrelated. Independent read-only audits found no
  P0 in the selected-axis or admission chain.
- The Torch model suite remains `remote_torch_pending`: the Windows host fails
  loading PyTorch `c10.dll` with WinError 1114. This is not model evidence.
- Current state is `implemented/local_tested/remote_pending`; no experiment has
  run, no performance value was interpreted, Phase 4 and official-final remain
  sealed, and no paper-admissible empirical conclusion exists.
