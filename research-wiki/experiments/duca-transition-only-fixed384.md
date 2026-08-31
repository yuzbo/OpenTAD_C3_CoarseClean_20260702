---
type: experiment
node_id: exp:duca-transition-only-fixed384
title: "DUCA Shared-ASFormer Transition-Only fixed-384"
status: tested
outcome: signed_score_space_proximal_candidate_exact_cuda_and_real_loader_gates_pending
added: 2026-07-11
---

## 2026-07-16 Round-2 method verdict

The external method/paper review returned `REDESIGN`: commit `7525efb` is not
the final train candidate. Its transition evidence and detached hard policy-
utility ingredients are retained, while global DP/homotopy and Gram proximal
are proposed to be replaced by one-per-exact-uniform-cell deformation and
different-cell weighted logistic flips. The project accepts this architecture
only at `designed` status. No Local-cell code, gate, pilot, mAP or cost exists,
so this experiment node remains `tested` for the predecessor implementation
and does not support C3/C4/C7.

## 2026-07-16 exact-commit Round-1 Pro audit

The reviewer inspected GitHub commit `7525efb` and returned
`GO_TO_REAL_GATE`, not permission for pilot or training. It found no static P0
model blocker and accepted the signed score-space proximal derivation as a
local selector-score descent identity. It independently classifies the route
as detached hard-swap selection-policy utility, not direct detector-gradient
learning, and official AdaTAD components with a wrapper, not source-identical
AdaTAD.

Three P1 classes remain before any pilot: the current gate is synthetic and
does not execute the real loader/DDP/train engine; all-zero or single-value
utility can yield a vacuous perfect alignment; and the measured utility mixes
frame content with selected-axis geometry and renewed assignment. AMP/DDP is
therefore unresolved and a fail-closed real THUMOS loader CUDA gate is the sole
allowed next experiment. This audit changes no claim or experiment status.

## 2026-07-15 signed counterfactual score-space repair

GitHub commit `7525efb2e07214615a59c482443246174a6adaf1` supersedes
`a6903ae` as the current code-review candidate. Exact gate Job `1165646` on
`a6903ae` correctly failed because candidate-only relative distillation did not
produce a detector-utility-aligned selector descent direction. A no-op-anchored
categorical softmax was also rejected as the formal objective: it fixes the
optimum semantics but cannot guarantee every candidate's local signed direction
when multiple swaps share a removed frame.

The replacement builds the hard-swap incidence matrix `A`, normalizes detached
utility `u = L_baseline - L_swap`, solves `(A A^T) v = u`, and applies a
stop-gradient proximal target to `d = A s`. At the current point this gives
`A(-grad_s L)` proportional to signed normalized utility, including shared-
remove candidates. The implementation also keeps the proximal loss in FP32
under outer AMP, fails closed on invalid swap indices, audits mixed batches per
sample, keeps all-short summaries finite, restores Python/NumPy/Torch RNG for
the detector teacher, and makes the formal gate reject dirty trees while
hash-binding core implementation files.

Clean remote CPU/Linux focused verification is `160 passed, 7 skipped`.
Independent reviewer `019f6620-3469-79d1-84b3-7b8bbd42b2d1` verified the
mathematical sign and shared-remove derivation but retained `HOLD` until AMP,
dirty-tree, all-short and mixed-batch issues were fixed; those deterministic
issues are addressed in `7525efb`. CUDA-only autocast execution, a clean
exact-commit CUDA gate, a real THUMOS loader/sample gate, forced-overflow pilot,
and matched mAP are still absent. Status is therefore only `tested`, not
`experiment_running`, `empirically_supported`, or `paper_ready`. Shell failure
Job `1165650` and canceled Job `1165654` are deployment diagnostics only.

## 2026-07-15 successful-update replacement implementation

Commit `a6903ae036d7b4bfd0c25752c51f020b20427fff` supersedes the deployment
qualification of `043be401`, whose Jobs `1164700-1164703` remain invalid
diagnostics. The replacement replays AMP-skipped materialized batches with
Python/NumPy/Torch RNG and forward-mutated state rollback. LR, ModelEma, and
DUCA schedules advance only after real optimizer updates. Checkpoint interval
remains five epochs; the sole primary result is terminal epoch 131
`state_dict_ema`.

The evidence finalizer reopens the checkpoint and recomputes official OpenTAD
mAP from the prediction JSON under frozen detector/evaluation/data/evaluator
hashes. Slurm submission is idempotent and preserves `jobid;cluster`. Local
checks are `80 passed, 3 skipped`; independent auditors
`019f631d-1917-7f13-b982-6b433b2b3924` and
`019f6603-d9b2-7e20-8a26-57739fa78561` found and closed all blockers; final
verdict is `GO`. Fresh exact CUDA gate and forced-overflow pilot remain
required, so no replacement long run or mAP exists and C3/C4 are unproven.

# DUCA Shared-ASFormer Transition-Only fixed-384

## 2026-07-13 corrected P0 rerun

The only admissible current implementation is commit
`40eb86ee69e19b3105f9ddd6a977fb7693f724ad` on
`codex/duca-transition-only-20260711`. It fixes the exact-uniform reference,
keeps structured boundary supervision as an FP32 soft expected-neighborhood-
mass surrogate, makes every active four-arm auxiliary loss FP32 under AMP, and
hash-binds post-run evidence to the exact code, configs, protocol, gate, seed,
and run manifest.

Formal CUDA gate `1161590` completed with `ok=true`, exact uniform, finite
selector/backbone/head updates, and exact optimizer coverage. The current code
uses `static_graph=True`, reentrant checkpointing, and a scorer-connected FP32
zero counterfactual loss for all-short batches. An independent exact-commit
audit and a multi-iteration counterfactual DDP startup pilot remain mandatory;
the four matched full trains have not been submitted. C3/C4 remain unproven.

Jobs `1161505-1161508`, `1161536-1161539`, gate `1161545`, and pilot `1161548`
are invalid deployment diagnostics. They respectively exposed batch-varying
static graphs, checkpoint ready-twice failures, an unsupported non-reentrant
checkpoint workaround, and unused-parameter reduction failure. None is a
performance result.

Intermediate failures are retained as protocol evidence, not results. Jobs
`1161482-1161485` used an invalid hard-coded CUDA device after Slurm remapping;
formal gate `1161489` found official-ASFormer FP16 backward NaNs; Job `1161492`
found direct calibration declared without an artifact; Job `1161494` found
short-window padding/duplicates incorrectly treated as a feasible one-swap.
Those defects are repaired in the current commit, but only completed full
training can establish performance.

## 2026-07-13 selection-quality follow-up

EMA epoch 89 from legacy beta=0 Job `1159416` received a selector-only quality
audit. The coarse classifier is only moderate, learned transition ranking is
worse than raw actionness change, and learned selected positions lose radius-1
coverage to exact uniform despite a small exact-hit gain. The detailed
diagnostic is `duca-selection-quality-epoch89.md`; it remains invalidated legacy
evidence and does not update C3/C4 or corrected commit `0ea4e15`.

## Claim under test

Binary coarse action-state transitions can guide an exact-budget/max-gap
pre-backbone selector more effectively than a matched exact-uniform policy,
without direct boundary heads or detector-gradient leakage into the coarse
classifier.

## Implementation provenance

- Base: `a5e1774b9941312569ca645341da1abad339db61`.
- Isolated branch: `codex/duca-transition-only-20260711`.
- Official ASFormer: `ChinaYi/ASFormer@e1bbe4f3ed083748f91467c51a63ac2a8b9277ad`.
- Normalized-LF `ASFormer/model.py` SHA256:
  `e075ee4825a201cfe324d5fbfb1332c0800f532e85b9d3809f6ca5180381c600`.

## Engineering evidence

- Gate `1159350`: `COMPLETED`, 26 focused tests passed.
- Official scaled ActionFormerHead train/test and optimizer step passed.
- Detector-only gradient: coarse probe `0`, transition scorer `16.4741`.
- Coverage-only gradient: coarse probe `0`, transition scorer `0.1542`.
- Transition loss: spatial stem and ASFormer encoder non-zero; ASFormer binary
  action head and decoder `0`.
- Exact selected count `8/8` and observed max unselected hole `1` in the scaled
  proof.
- Optimizer exact coverage and component LRs passed.

## Cost smoke

Scaled selector-only smoke (`T=16`, `K=8`, 16x16, RTX 4090) reported median
12.63 ms and p90 12.92 ms. This is an implementation smoke, not paper evidence,
not trained-checkpoint cost, and not full-stack cost.

## P0 matched matrix

1. Exact-uniform fixed-384 in the same structured feasible family.
2. Direct-boundary `a5e` architecture retrained to the matched 13200-step
   expected horizon.
3. Transition-only fixed-384 with beta=0.
4. Transition-only fixed-384 with beta ramping to 0.25.

## Remaining gates

Full training, three seeds, coarse AUROC/AUPRC/ECE, transition recall, matched
high-tIoU comparison, one-swap counterfactual alignment, selected-axis geometry,
trained full-stack cost, and a second detector remain unverified.

## 2026-07-11 hardening audit

- Commit `8e38ccaa99c37e1ffaefe5651a1bc74caf3afa81` supersedes the first
  implementation commit `fc98eca` for all future evidence.
- Restored legacy spatial-stem hidden as the default direct-baseline behavior;
  transition-only explicitly opts into same-forward ASFormer encoder hidden.
- Detector attribution now backpropagates real `cls_loss + reg_loss`, not an
  input-energy surrogate. Training supervision provenance is explicit and
  inference-use fields are false.
- Optimizer exclusions are applied before DDP registration. Direct and
  transition P0 variants share coarse-component LRs and the same
  `val_start/anchor=47` fields; the exact current schedule first evaluates
  after one-based epoch 52.
- Formal P0 launchers require a clean tree and a commit/config/source/checkpoint
  bound CUDA gate JSON. A boolean environment flag is not accepted as evidence.
- Slurm `1159383` is the clean-commit formal gate for real VideoMAE Adapter,
  `T=768 -> K=384`, 160x160, and ActionFormerHead detector-only backward. It is
  `PENDING` at this update. No P0 full-train job has been submitted.

## Current audited run set

- Current HEAD: `8bfc0e549434591b9bf1a9cd5563deb0da388f92`.
- Formal gate `1159395`: `COMPLETED`. Linux focused tests `37 passed, 1
  skipped`; real T768/K384 detector-only gradients were scorer `0.023908`,
  VideoMAE adapter `22.9283`, ActionFormerHead `290.7146`, coarse probe `0`;
  selected count `384`, max hole `6`, CUDA peak allocation `2.111 GB`.
- `1159387/1159389/1159390` failed on an AMP padded-window coverage NaN and
  `1159388` was cancelled to preserve same-commit comparisons. Root cause was
  FP16 underflow of `clamp_min(1e-8)` after autocast `conv1d`; commit `41de232`
  keeps coverage computation in FP32 and adds a CUDA padded-window test.
- Hash-bound P0 seed-0 jobs now running: exact uniform `1159414`, direct-a5
  `1159415`, transition beta0 `1159416`, transition beta0.25 `1159417`.
  All use gate `1159395`.

## 2026-07-12 interim evaluation

### Critical protocol invalidation

The label `exact-uniform` below is historically preserved but invalid. In
commit `8bfc0e5`, T=768/K=384 midpoint-distance reference logits collapse to a
single value, so Viterbi follows a tie-break path rather than uniform sampling.
The decoded path overlaps true rounded endpoint linspace at only 47.135%, has
179.695-frame mean rank-aligned absolute error, and gap histogram
`{1:358, 10:1, 16:24}`. Job `1159414` and its eventual 55.67 best Avg-mAP are
therefore a degenerate-DP diagnostic, not a uniform baseline.

The same defect invalidates the intended alpha=0 starting point of Jobs
`1159416/1159417`; their learned-policy results remain diagnostic, but cannot
support a continuous-homotopy or matched-uniform superiority claim. Historical
uniform anchors are real but unmatched: Job `1150701` reached 64.352 with
native stride-2 adapter ActionFormer, and Job `1150842` reached 65.696 with a
grid-aware detector.

Commit `0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d` fixes the reference to
rounded endpoint linspace and makes the formal gate/validator require exact
decoded-position equality. Remote focused verification passed 26 tests with 2
skips. A new formal CUDA gate and corrected P0 rerun have not yet been
submitted, so the fix is `tested`, not `experiment_running`.

The first scheduled validation completed after logged epoch 51. Raw seed-0
metrics are:

| Variant | Avg-mAP | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 |
|---|---:|---:|---:|---:|---:|---:|
| invalid alpha0 DP tie-break | 53.68 | 70.94 | 64.29 | 55.82 | 46.00 | 31.35 |
| direct-a5 | 55.77 | 73.43 | 68.06 | 59.24 | 45.92 | 32.19 |
| transition beta=0 | 63.07 | 79.60 | 73.66 | 65.82 | 54.97 | 41.28 |
| transition beta=0.25 | 62.02 | 78.59 | 73.18 | 64.75 | 53.62 | 39.98 |

The next two scheduled evaluations give this Avg-mAP trend:

| Logged epoch before evaluation | invalid alpha0 control | direct-a5 | beta=0 | beta=0.25 |
|---:|---:|---:|---:|---:|
| 51 | 53.68 | 55.77 | 63.07 | 62.02 |
| 56 | 54.42 | 55.41 | 62.17 | 62.79 |
| 61 | 55.11 | 55.82 | 63.74 | 63.21 |
| 66 | 55.46 | 56.15 | 63.39 | 63.17 |
| 71 | 55.37 | 56.02 | 63.93 | 63.55 |

At epoch 71, beta=0 reports 80.18/75.50/67.12/55.51/41.33 and beta=0.25
reports 79.17/74.37/66.17/55.75/42.28 at tIoU 0.3:0.7. Beta=0 leads by 0.38
Avg-mAP, while beta=0.25 leads at tIoU 0.7 by 0.95. Its protected bridge has
reached about 0.247/0.25 by logged epoch 76, so the emerging hypothesis is a
possible high-IoU diagnostic rather than a valid causal gain; the broken
homotopy prevents a paper claim. All jobs remained numerically healthy around
epoch 72-76, but numerical health does not repair the invalid protocol.

## 2026-07-12 13:35 +0800 diagnostic monitor

All four legacy jobs remain `RUNNING` after about 14h20m. The invalid alpha0
control reached epoch 82 with best/latest Avg-mAP 55.50; direct-a5 reached
epoch 81 with best/latest 56.34; beta=0 reached epoch 86 with best/latest
63.98; beta=0.25 reached epoch 86 with best 63.55 and latest 63.40. Latest
beta=0 IoU-wise mAP is 79.73/74.59/66.74/56.40/42.44, while beta=0.25 is
79.34/74.48/65.97/55.09/42.12. All four Slurm stderr files are empty and no
Traceback, OOM, executed non-finite failure, ValueError, or hard FAIL was
found. These remain protocol-invalidated diagnostics and do not update C3/C4.

## 2026-07-12 21:37 +0800 diagnostic monitor

All four legacy jobs remain `RUNNING` after about 22h23m. The invalid alpha0
control is at epoch 118 with best/latest Avg-mAP 55.67/55.38; direct-a5 is at
epoch 116 with best/latest 57.60/57.60; beta=0 is at epoch 121 with best/latest
64.34/63.30; beta=0.25 is at epoch 121 with best/latest 63.55/63.04. Latest
beta=0 IoU-wise mAP is 79.23/74.43/66.20/55.78/40.85, while beta=0.25 is
78.38/73.27/66.02/55.45/42.10. All Slurm stderr files remain empty. The
beta=0 diagnostic now reaches the historical 64.352 native-stride uniform
anchor numerically but cannot be compared causally because its homotopy start
and the current control are invalid. C3/C4 remain unproven.

## 2026-07-12 23:38 +0800 diagnostic monitor

All jobs remain `RUNNING` after about 24h24m. Invalid-alpha0/direct reached
epoch 126 and beta0/beta0.25 reached epoch 131. Best/latest diagnostic Avg-mAP
is 55.67/55.22, 57.71/57.71, 64.34/63.87, and 63.55/63.27 respectively.
Latest beta0/beta0.25 mAP@0.7 is 41.76/43.12, but this cannot establish a
bridge benefit under the invalid homotopy. All stderr files remain empty; the
jobs are near completion but have not exited, so no final status is recorded.

## Final completed diagnostic results (2026-07-13)

All four jobs completed with Slurm exit code 0. Each produced 17 evaluations,
an epoch-131 checkpoint, and a final evaluation. Slurm stderr was empty for all
runs; no Traceback, CUDA OOM, or runtime non-finite skip was found.

| Invalidated variant | Best epoch | Best Avg | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 | Final Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| invalid alpha0 DP tie-break | 91 | 55.67 | 71.93 | 66.68 | 58.38 | 47.10 | 34.27 | 55.21 |
| direct-a5 | 121 | 57.71 | 74.38 | 68.96 | 60.91 | 49.54 | 34.78 | 57.50 |
| transition beta=0 | 91 | 64.34 | 79.92 | 75.18 | 67.83 | 56.81 | 41.97 | 64.20 |
| transition beta=0.25 | 71 | 63.55 | 79.17 | 74.37 | 66.17 | 55.75 | 42.28 | 63.14 |

The beta=0.25 bridge trails beta=0 by 0.79 best Avg-mAP. Its best-checkpoint
mAP@0.7 is only 0.31 higher and its final mAP@0.7 is 0.52 lower, so there is no
stable detector-gradient benefit even as a diagnostic. Beta=0 is numerically
within 0.012 Avg-mAP of historical Job 1150701, but that historical uniform
anchor is protocol-unmatched and cannot validate C3. Full-run selected-count
and max-hole distributions were not emitted; final-batch effective-budget
means of 223.5/227.5 reflect short valid windows and are not run aggregates.

Verdict: the runs establish numerical stability and show that learned
transition scoring can reach about 64 Avg-mAP under this implementation. They
do not test matched exact-uniform superiority or the intended homotopy, and do
not support C3 or C4. Corrected commit `0ea4e15` still requires a new formal
CUDA gate and replacement matched runs before any empirical claim.

The strict result-to-claim gate is `C3=no`, `C4=no`, confidence `high`.
Execution integrity passed, but protocol integrity failed. Independent reviewer
delegation was unavailable, so this local judgment is pending external review.

## 2026-07-15 corrected deployment chain

The corrected route has not produced a valid mAP result. The deployment chain
is preserved because each failure invalidates the next layer of evidence:

- `f048c31` / gate `1162051`: CUDA gate passed, but audit found the structured
  DP surrogate still changed shape with valid length.
- `34b9106` / gate `1162124`: fixed-shape surrogate gate passed, but all-short
  direct slots assigned mass to invalid positions.
- `77fc7a4` / gate `1163434`: failed because the test helper exercised the
  legacy decoder instead of the configured structured policy.
- `a5b3c67` / gate `1163435`: formal gate passed after test-path correction.
  Pilot `1163437` then failed before training because preparation and runtime
  resolved different config hashes.
- `51330c8` / gate `1163439`: formal gate passed in 6m02s with finite optimizer
  updates, exact K=384, max hole 7, and clean exit. It remained HOLD because
  profiling and other configuration values could still be inherited, and the
  generated job did not persist the resolved environment.
- `cff479e`: introduces the single canonical environment contract, preparation
  TSV/SHA binding, runtime reconstruction and byte comparison, and tests that
  cover all environment reads in the base P0 config. Local matrix/suite tests
  passed, but exact-commit independent review and CUDA evidence are pending.
- `cff479e` real-batch follow-up: static pilot Jobs `1163456`, `1163460`, and
  no-checkpoint/static Job `1163471` failed when the parameter-use graph
  changed. No-checkpoint/dynamic/find-unused Job `1163472` completed all ten
  real optimizer iterations across varying effective K with about 8.47 GB
  peak CUDA memory. Its epoch-end save failed only because `/data` was then
  full; this is training-graph PASS but deployment/storage FAIL.
- Current `28908e2de974ff90fe1e16e8f12a02085742f9f7` fixes the shared four-arm
  protocol to `with_cp=False/static_graph=False/find_unused_parameters=True`,
  adds four pilot configs and a machine-readable ten-step training probe, and
  requires commit/core-gate/shared-protocol-bound pilot evidence before formal
  suite preparation. Local and clean-remote focused tests are both 62 passed,
  1 skipped. The exact CUDA gate and four-arm pilot have not run yet.

Status: `tested`, not `experiment_running`. No corrected full train is running.
The next allowed sequence is independent exact-commit audit, fresh formal
gate, the complete four-arm DDP pilot, then and only then the four matched
seed-0 arms.

## 2026-07-13 external exact-commit audit

The external review of `0ea4e15` independently preserves the `HOLD` verdict and
does not upgrade the legacy diagnostics. It additionally found that the
legacy/direct `stable_selection` route in `acquisition.py` retains the same
midpoint-reference family, so Job `1159415` is not a clean corrected direct
control either. The review did not access cluster logs or rerun jobs; its new
performance thresholds are proposed specifications, not empirical results.

The proposed DUCA-FSU replacement is recorded separately as an unimplemented
idea. No experiment status or claim status changes follow from that proposal.

## 2026-07-15 exact AMP and mixed-length feasible-family repair

- Commit `18dc1cdf75ae40e8e2068d4213ff037313829bec` fixes a CUDA autocast
  cache hazard: a no-grad counterfactual teacher previously created detached
  FP16 casts that the later main detector forward could reuse. Exact gate
  `1164279` completed and proved finite real-model gradients and optimizer
  updates, but this gate is stale after the next code change.
- Four-arm DDP pilot `1164286` completed uniform's real 10-step arm. Direct-a5
  then failed on its first batch with `every active structured slot assignment
  must sum to one`; transition arms did not run. This is a failed deployment
  diagnostic, not a result.
- Root cause: the hard path used per-sample `valid_count/effective_k`, but the
  soft surrogate always solved T=768/K=384 and only afterward masked invalid
  suffix positions and treated the first effective-K ranks as active. Real
  batches contained mixed effective budgets such as 368.5, 306 and 286.
- Commit `043be401ba2b694342dc395f263e9a9858628d69` now runs both hard and
  relaxed DP on the same valid prefix, effective K and max-hole constraint,
  then pads inactive slots/suffixes with exact zeros. It keeps the dynamic DDP
  contract `with_cp=False/static_graph=False/find_unused_parameters=True`.
- Clean Linux focused verification: `122 passed, 5 skipped`. Independent max
  exact-commit review: GO, P0=0, P1=0. Exact CUDA gate `1164318` completed at
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_043be40_exact_gate_20260715_1500`.
- Four-arm pilot `1164319` completed with 10/10 optimizer steps for every arm,
  full/mixed/all-short budget coverage, complete trainable parameter-group
  gradient coverage and no fatal scan hit. Pilot root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_043be40_ddp_pilot_20260715_1500`.
- Hash-bound formal root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_043be40_p0_formal_seed0_20260715_1647`.
  Jobs `1164700/1164701/1164702/1164703` are exact-uniform, direct-a5,
  transition beta=0 and transition counterfactual respectively; all entered
  RUNNING under Slurm allocations.

Status: `experiment_running`. No matched mAP or post-run claim evidence exists
yet. C3/C4 remain unproven.

### Initial formal startup check

All four formal jobs reached real epoch-0 step `50/99` with finite logged
losses and no fatal scan hit. Uniform and transition beta=0 both logged total
loss about 5.61 during their shared warmup; transition counterfactual added a
finite distillation term about 0.327. Direct-a5 logged total loss about 9.03,
mostly because it owns explicit start/end/context/boundary auxiliary losses, so
raw total loss is not comparable across arms. Peak logged memory is about
8.68 GB. This is startup stability only, not mAP evidence.

### 2026-07-15 successful-update invalidation

The exact-commit Pro audit identified that the fixed epoch loop does not replay
GradScaler-skipped optimizer updates. Live logs then confirmed the defect is
active in every formal arm:

| Job/arm | Completed epoch end | Expected updates | Schedule step | Deficit |
| --- | ---: | ---: | ---: | ---: |
| `1164700` exact-uniform | 25 | 2600 | 2596 | 4 |
| `1164701` direct-a5 | 25 | 2600 | 2597 | 3 |
| `1164702` transition beta=0 | 25 | 2600 | 2596 | 4 |
| `1164703` transition counterfactual | 24 | 2500 | 2496 | 4 |

With the fixed 132-epoch loop, these runs cannot produce the declared 13,200
successful updates. They remain useful only as terminal diagnostics and cannot
enter a matched formal table or support C3/C4. A replacement suite requires
state-exact same-batch AMP replay, exact loader/update assertions, one frozen
checkpoint rule, artifact/evaluator SHA binding, and a new exact-commit gate
plus multi-batch pilot.

## 2026-07-15 earlier exact-commit Pro audit absorption

Attachment `48c9c615-e001-40cb-8207-951cb504198f` audited exact commit
`043be401` with repository visibility but without remote Slurm artifacts. Its
verdict is GO for completing the four seed-0 jobs and HOLD for paper claims;
no confirmed P0 was found. Local recheck accepts two claim/protocol risks:

This historical GO did not include remote Slurm artifacts and is superseded by
the successful-update invalidation above. The remaining mechanism and
checkpoint-policy observations are retained as source history.

- The current counterfactual objective is candidate-relative one-swap ranking.
  It has no no-op anchor and therefore does not learn signed acceptance versus
  retaining the baseline selection.
- Repeated evaluations use the THUMOS test split while the post-run contract
  permits `best_or_final`. The current suite must seal final one-based epoch
  132 EMA as its sole primary result and treat intermediate mAP as diagnostic.

The review also corrects the first evaluation to one-based epoch 52, confirms
that the actionness head is binary while the shared transition route receives
train-only endpoint supervision, and identifies selected-axis detector
geometry as an unresolved mechanism confound. It does not change experiment
or claim status. The project's absorption is `PARTIAL_ACCEPT`: no-op softmax
is a candidate rather than a unique solution; two-rank DDP is unnecessary for
the current one-GPU suite; and a physical-time head remains gated by a
fixed-selection geometry diagnostic.

Before any mAP was emitted, the study-level primary-result protocol was sealed
at 2026-07-15T12:16:50Z. Local artifact:
`docs/methods/2026-07-15-duca-043be401-primary-result-protocol.json`; SHA-256
`AAC0FCA8671AE6F58CF4C9B5D4D40282BE714AA354028246E86504FD39C89B48`.
The same read-only bytes exist at the formal run root and in each variant log
directory. Final one-based epoch 132 `state_dict_ema` is the only primary
result; intermediate THUMOS test mAP cannot select a checkpoint.

At the independent 2026-07-15 20:16 CST protocol check, Jobs
`1164700-1164702` had entered epoch 27 and `1164703` epoch 25; no evaluation
had run. The earlier 18:55 monitor found checkpoint `epoch_14.pth` in all arms
and no fatal scan hit. Status remains
`experiment_running`; C3/C4 remain unproven.
