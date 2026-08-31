---
id: exp:duca-global-curriculum-official60
type: experiment
status: experiment_running
updated: 2026-07-21
---

# DUCA global-curriculum official-60

## Question

在同一个 P0、同一个全局 exact-K/max-gap 可行集和同一个官方 AdaTAD 协议下，
状态转变驱动的跨区域预算分配能否超过精确均匀采样；受保护的下游检测梯度和
训练期均匀 companion 是否分别带来 terminal mAP 增益？

## Exact implementation

- Branch: `codex/duca-global-curriculum-20260721`
- Commit: `63e25eb17e523d369f73434ed4d9b6446608861a`
- GitHub:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-global-curriculum-20260721`
- Remote snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_global_63e25eb_20260721`
- Focused evidence: parent P0 revision `158 passed, 3 skipped`; final launcher
  revisions resolve all four quality/aggregation tools through `python -m`;
  exact commit `63e25eb` passed the EMA group-audit regression and related
  curriculum contracts: `21 passed`, `py_compile`, diff, exact-HEAD and
  clean-tree checks.

## Shared model contract

- Offline TAD, fixed K=384, selected-axis `global_structured_topk`, hard
  max-gap G=2 and cross-region quota transfer.
- Shared P0 coarse action/transition checkpoint and the same official
  AdaTAD/ActionFormer detector path.
- Actionness remains binary coarse supervision. Selection uses transition,
  boundary and uncertainty evidence; it is not actionness top-k.
- Protected detector feedback may update only the transition scorer. It must
  not rewrite the action head or coarse ASFormer representation.
- Training and inference use the same learned hard policy for learned arms;
  G2's uniform companion is training-only and is absent at inference.

## Matched arms

| Arm | Difference | Causal question |
| --- | --- | --- |
| U | Exact-uniform K=384 | Same-commit detector baseline |
| G0 | Global learned policy, detector feedback off | Does coarse/transition supervision learn useful allocation? |
| G1 | G0 plus protected detector feedback | Does downstream TAD feedback improve mAP? |
| G2 | G1 plus one-pass training-only uniform companion | Does stable detector input improve learning without changing inference? |

For G2, companion rows have zero selector bridge and learned rows are scaled
by batch-size / learned-row-count, so their expected detector-gradient mass is
matched to G1 rather than silently halved.

## P0 bounded variants

The three P0 arms are not three model families. They share the same coarse
probe, official ASFormer temporal module, transition scorer, losses and global
selector. Only the component learning rates differ:

| Variant | Coarse trunk | Action head | Transition scorer |
| --- | ---: | ---: | ---: |
| Control | `2.5e-5` | `5e-5` | `1e-4` |
| Coarse-first moderate | `5e-5` | `1e-4` | `2.5e-5` |
| Coarse-first strong | `1e-4` | `2e-4` | `5e-5` |

All three use actionness/transition/boundary weights `1.0/0.10/16.0`.
Holdout selection uses exact/radius-zero boundary recall, short-action exact
both-endpoint coverage, endpoint distance, transition AUROC and coarse
AUROC/AUPRC lift. Radius-one recall remains diagnostic only.

## Deployment

- Active Slurm Job: `1178989`
- Run root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_global_63e25eb_serial_20260721_2120`
- Receipt: `submission/receipt.json`, schema
  `duca_two_stage_serial_submission_v1`; the training-only split is regenerated
  from the same seed `3407`. Receipt SHA-256 is
  `b1b7892b282b265a77efc7df42a034bb2d1c7fde415471f06b6179fa07d42c85`;
  split SHA-256 is
  `7b381a38d4a0d66f5746c768df2a9c2ab7f27e6c93e83a99e707eaa6300217a7`.
- Current submission: `1178989` at 2026-07-21 21:20 +08:00 after a successful
  login-node precheck; initial state `PENDING`.
- Execution order: P0 real-model gate and candidates, P0 selection, U/G0/G1/G2
  full-model gates, then four official-60 arms sequentially.

### 2026-07-21 21:00 status

- Job `1178975` remains `PENDING (Priority)` with zero runtime, so the corrected
  real-CUDA P0 gate has not executed and none of its three P0 candidates has
  started. This is queue state, not model evidence.
- The deployed candidate grid is complete: control `(2.5e-5, 5e-5, 1e-4)`,
  coarse-first moderate `(5e-5, 1e-4, 2.5e-5)`, and coarse-first strong
  `(1e-4, 2e-4, 5e-5)` for coarse trunk, action head and transition scorer.
  Architecture and losses remain identical across candidates.
- The subsequent matched matrix remains U/G0/G1/G2 from the same V8 model;
  no local-cell, X3D, MUST, dynamic-budget or alternate detector experiment
  was added.
- The predecessor selected-axis Job `1178642` is still running wave two near
  epoch 52/53. Its sealed wave-one terminal EMA values remain exact-uniform
  `64.4579977` and direct-0.25 `63.7101546`; homotopy and companion have no
  terminal mAP yet.

## Evidence boundary

The code is `tested_focused` and the experiment is `experiment_running`.
There is no completed CUDA full-model gate, selected P0 checkpoint, terminal
EMA mAP or greater-than-65 result yet. Jobs `1178642` and `1178863` are
predecessor/local-cell diagnostics and cannot be substituted into this table.

## 2026-07-21 pre-runtime P0 audit

Job `1178911` had accumulated zero runtime and remained pending when the P0
selector was re-audited. The model family and U/G0/G1/G2 causal matrix remain
valid, but the exact job is no longer admissible for execution because its P0
winner rule uses saturated radius-one recall as both a hard gate and the first
ranking key. Its three candidates also vary auxiliary loss scales while keeping
the transition scorer four times faster than the random-initialized coarse
trunk. A replacement may reuse the four official arms only after the same
global model receives the bounded P0 metric/LR correction and exact tests.

The queue started Job `1178911` before replacement deployment completed. It
ran for `00:09:01`, entered the first old P0 arm, and was then cancelled. Its
logs and partial epoch remain immutable diagnostic history; they are not
eligible P0 or paper evidence. The first replacement Job `1178927` was
cancelled at zero runtime after Job `1178863` exposed a shared script-path
failure in post-training quality export. Exact commit `2c403a8` invokes the
quality tools as repository modules; Job `1178933` was the next historical
replacement.

## 2026-07-21 complete-entry correction

The two downstream aggregators in `2c403a8` still used file-path execution and
both reproducibly failed to import `tools.bata`. Commit `6b6363e` changes only
those calls to module entry points and adds a regression test; model weights,
architecture, P0 variants and U/G0/G1/G2 are unchanged. Job `1178933` began
while a temporary compatibility-audit file was still present in its exact
snapshot and failed the clean-tree guard after three seconds, before any model
construction or optimizer update. The temporary file was removed and the old
snapshot reverified clean, but the job remains immutable failed history. Job
`1178947` was the sole admissible replacement at that checkpoint.

## 2026-07-21 P0 optimizer-gate classifier correction

Job `1178947` started at 20:24 +08:00 and failed after one minute in the real
P0 gate, before any optimizer update. The model optimizer was already grouping
parameters correctly. The gate used an over-broad rule that classified every
official-ASFormer parameter containing `conv_out` as an action head, even
though official ASFormer attention layers also contain internal `conv_out`
projections. This made a valid temporal-trunk optimizer group appear to mix the
coarse-trunk and action-head learning rates.

Commit `9138156` makes the gate use the same exact topology rule as
`ActionFormer.get_optim_groups()`: only `encoder.conv_out.*` and
`decoders.<stage>.conv_out.*` are binary action heads; attention-layer
`conv_out` parameters remain temporal-trunk parameters. Four explicit name
cases are regression-tested. No selector, decoder, loss, P0 duration,
U/G0/G1/G2 setting, AdaTAD component or ActionFormerHead was changed.
At the `9138156` checkpoint, Job `1178975` was the only active replacement and
still had to pass the corrected real CUDA gate before any P0 candidate could
train. The following section records its immutable outcome and successor.

## 2026-07-21 21:20 EMA gate correction

Job `1178975` left queue and executed one production P0 step. The logged total
loss was finite (`1.5725`); actionness, transition and transition-boundary raw
objectives were active, requested/effective K were both 384, detector loss was
zero and the detector path was explicitly `skipped`. Gradient and optimizer
audits confirmed both coarse and transition-scorer groups updated. The job
failed only because its EMA check observed the first nonzero-gradient parameter
from each group; the 0.001 EMA fraction of one step rounded to zero on at least
one representative FP32 tensor.

Commit `63e25eb` replaces that single-parameter verdict with an audit over all
trainable parameters in each existing group and preserves representative
deltas as diagnostics. It changes no model or training hyperparameter. The
remote test copy passed `21` affected tests, Python compilation and diff checks.
Fresh exact snapshot `opentad_duca_global_63e25eb_20260721` was cloned through
the academic proxy; precheck passed and Job `1178989` was submitted. Status
remains `experiment_running`; no P0 winner or terminal mAP exists yet.

## 2026-07-21 21:27 no-rewrite checkpoint

- A fresh read-only audit still resolves the active implementation to V8 at
  exact commit `63e25eb17e523d369f73434ed4d9b6446608861a`. The isolated code
  tree is clean and synchronized with its remote branch.
- Job `1178989` remains `PENDING (Priority)` with zero runtime. No failed gate
  or model result exists, so there is no basis for another selector, decoder,
  model class, config family or worktree.
- Predecessor Job `1178642` remains numerically finite. At the latest read,
  homotopy had entered epoch 58 and uniform-companion had entered epoch 57 of
  60. Neither arm had emitted `terminal_evaluation.json`; the only sealed
  terminal values remain exact-uniform `64.4579977` and direct-0.25
  `63.7101546`.
- The no-rewrite rule is binding: current evidence may update the V8 registry
  and experiment state, but cannot create a synonymous global selector or
  fall back to one-frame-per-cell geometry.

## 2026-07-22 03:31 terminal P0 mechanism verdict

Job `1178989` finished `FAILED/2:0` after `05:32:02`, but the failure is an
intentional fail-closed mechanism verdict rather than a runtime crash. All
three learning-rate profiles completed 5/10/15/20-epoch checkpoints and quality
exports. `frontend_decision.json` reports
`HOLD_FRONTEND_MECHANISM_FAILED`, `eligible_count=0`, `winner=null`.

Across all 12 candidates, coarse action AUROC reached at most `0.619653`, while
the learned policy transition AUROC remained below pure `abs(delta p_action)`
for every checkpoint. Learned endpoint distance was worse than exact uniform
for every checkpoint (`uniform_minus_learned` from `-0.058885` to `-0.098942`).
Radius-zero recall gain was small/inconsistent (`-0.025613` to `+0.010858`).
Therefore V8 is sealed as a negative scorer-mechanism result and correctly did
not start U/G0/G1/G2. It must not be rerun or used as terminal-mAP evidence.

## 2026-07-21 21:41 real P0 gate passed

- Job `1178989` started on Slurm GPU `g0006`. The exact-commit real gate wrote
  `p0_real_gate.json` with `ok=true` and clean binding to `63e25eb`.
- The gate used a real two-row THUMOS batch. It built the complete AdaTAD
  object but executed zero detector/backbone/head calls during P0. The frozen
  detector was byte invariant.
- Gradient ownership passed: actionness updated only the 62-parameter coarse
  group, while transition and boundary supervision updated only the
  six-parameter transition scorer. All 68 trainable parameters saw gradients.
- One production AMP/AdamW/scheduler/EMA update completed. Group-wide EMA
  evidence changed 60/62 coarse parameters and 5/6 scorer parameters; the
  previous representative-tensor false negative is closed.
- Hard selection remained exact K=384 with observed max unselected hole 2.
  Only the three declared losses were active and every inactive loss was a
  graph-free zero.
- The first sequential candidate, `lr_control_c25_a50_s100`, entered epoch 0.
  At step 60 its loss was finite, detector path was explicitly skipped and
  requested/effective K were both 384. This is successful gate/runtime
  evidence, not holdout quality or terminal mAP.

## 2026-07-21 22:07 predecessor four-arm terminal verdict

Predecessor Job `1178642` completed all four epoch-59 EMA evaluations on exact
commit `cb89586`. These are diagnostic evidence for the superseded V5
training contract, not results from current V8:

| V5 arm | Avg-mAP | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 | Delta vs U |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-uniform | 64.4580 | 79.7557 | 75.5604 | 67.5863 | 56.7664 | 42.6212 | 0.0000 |
| direct-0.25 | 63.7102 | 79.4111 | 74.5551 | 66.6419 | 55.4667 | 42.4759 | -0.7478 |
| homotopy-0.25 | 63.0601 | 78.3237 | 73.6741 | 66.0751 | 55.0551 | 42.1723 | -1.3979 |
| homotopy + uniform companion | 63.6931 | 79.3542 | 73.9892 | 66.7425 | 56.2650 | 42.1145 | -0.7649 |

The old learned endpoint is therefore the bottleneck: a uniform-to-learned
schedule alone made the result worse, and the training-only companion recovered
about 0.6330 Avg-mAP relative to homotopy but still did not reach uniform. Do
not rerun these V5 arms under new names. Current V8 remains a distinct bounded
test because it adds repaired P0, freezes the coarse branch for official-60 and
restricts protected detector feedback to the transition scorer.

At the same checkpoint, V8 Job `1178989` remained healthy in the first control
P0 candidate and had entered epoch 6/20. Actionness BCE was approximately
0.68--0.69, detector execution remained skipped, and no Traceback, OOM or
non-finite collapse was observed. Holdout selection and the other two P0
candidates remain pending.

## 2026-07-21 22:14 V8 P0 live checkpoint

- Exact Job `1178989` remained `RUNNING` on `g0006` at `00:40:33` elapsed.
- The first control-LR candidate completed epoch 7 and entered epoch 8/20.
  Its latest logged loss was finite (`1.4643`); actionness BCE was `0.6840`,
  transition raw loss was `6.7155`, and transition-boundary raw loss was
  `0.0135`.
- The detector path remained explicitly `skipped`, requested/effective K were
  both `384`, and no Traceback, OOM, ValueError or non-finite event was found.
- This remains runtime evidence only. No sealed holdout winner, V8 arm, or V8
  terminal mAP exists yet; status stays `experiment_running`.

## 2026-07-21 22:22 V8 P0 and repository binding

- The exact implementation tree remained clean at `63e25eb`; fetch reported
  zero commits ahead/behind and explicit push returned `Everything up-to-date`.
- The control-LR candidate completed epoch 8 and entered epoch 9/20. Latest
  logged loss was finite (`1.4665`), with actionness BCE `0.6855`, weighted
  transition loss `0.6396`, and weighted transition-boundary loss `0.1415`.
- Detector execution remained skipped and no error/non-finite signal appeared.
  No winner or V8 terminal mAP exists, so this does not change claim status.

## 2026-07-21 exact-commit Pro objective audit

- A Pro static review of exact commit `63e25eb` is archived byte-for-byte at
  `docs/methods/reviews/2026-07-21-63e25eb-duca-v8-unique-endpoint-pro-review-raw.txt`;
  independent absorption is in
  `docs/methods/2026-07-21-63e25eb-duca-v8-unique-endpoint-pro-review-absorption.md`.
- Static verification confirms that the same radius-4 Gaussian
  `transition_target` feeds both distribution supervision and the active
  `exp(-neighborhood_mass)` coverage loss. The repository already contains an
  exact structured event-probability primitive, but V8 does not use it.
- Static verification also confirms that G1/G2 use a surrogate transport and
  the current serial gate does not execute the preregistered real hard-swap
  alignment before writing `formal_training_unlocked=True`. This blocks a
  detector-aligned-gradient claim even if plumbing gates pass.
- The review's proposed radius-one exact event objective is not accepted as
  written: with `max-hole=2`, every internal three-position event is guaranteed
  to contain a selection, so its exact coverage probability is one and its
  gradient is zero. A corrected objective must use a nontrivial event such as
  rounded `radius=0` and pass a brute-force headroom/gradient gate.
- Job `1178989` remains immutable and `experiment_running`; this review changes
  neither its code nor its result status. Its outputs are diagnostic until the
  objective and bridge-alignment contracts are separately resolved.

## 2026-07-21 23:40 first P0 LR profile terminal diagnostic

- Job `1178989` remained healthy on `g0006`. The first profile,
  `lr_control_c25_a50_s100`, completed all 20 P0 epochs and wrote
  `completion.json` with `ok=true`, exact-commit binding, train-only data use
  and checkpoints at epochs 5/10/15/20. This proves protocol completion only;
  it is not a mechanism-gate pass or a holdout-winner decision.
- Coarse action evidence improved: video-macro AUROC rose from `0.562321` at
  epoch 5 to `0.624512` at epoch 20; pooled AUROC/AUPRC at epoch 20 were
  `0.614948/0.346425`. The cheap binary branch therefore learned a real but
  still moderate actionness signal.
- The structured selector did not convert that signal into better boundary
  allocation. At epoch 20, learned radius-zero endpoint recall was `0.151525`
  versus uniform `0.142912` (delta `+0.008613`, bootstrap CI
  `[-0.010467, 0.026975]`), while radius-one recall was `0.883237` versus
  `0.999775`. Mean endpoint distance was also worse: `0.538120` versus uniform
  `0.477457`, with uniform-minus-learned CI `[-0.088083, -0.037186]`.
- Both-endpoint coverage showed the same failure: learned/uniform were
  `0.016453/0.039503` at radius zero and `0.779621/0.999549` at radius one.
  Exact K remained `384` and mean max-hole was `1.916667`, so this is a
  placement-quality failure inside the feasible family, not a budget or
  coverage-contract violation.
- The learned policy's transition discrimination was weaker than the simple
  coarse-state derivative: radius-zero AUROC was `0.521321` for the policy and
  `0.553237` for `abs(delta p_action)`; radius-one AUROC was
  `0.505647/0.541698`. Pure-delta top-k is only a scorer diagnostic because its
  mean max-hole was `36.73`, not an admissible selector baseline.
- This profile therefore fails the declared endpoint-distance,
  both-endpoint-coverage and transition-versus-delta quality tests. It must not
  be called the P0 winner merely because `completion.json` says `ok=true`.
  The second profile, `lr_coarse50_action100_scorer25`, had entered epoch 3
  with finite losses; the third profile, holdout decision and every V8
  official-60 arm remained pending. No V8 terminal mAP exists.
