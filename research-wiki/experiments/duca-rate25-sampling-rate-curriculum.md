---
type: experiment
id: duca-rate25-sampling-rate-curriculum
status: experiment_running
updated: 2026-07-27
---

# DUCA 25% sampling-rate curriculum

## Question

Measure the offline TAD performance distribution after reducing the DUCA
sampling budget from `K=384/768` (50%) to `K=192/768` (25%), while preserving
the current complete curriculum. This is a cross-sampling-rate comparison, not
a sampling-rate-only model or a contribution ablation.

## Matched contract

The comparison anchor is running Job `1191957` at commit
`42dba3f90b37243e7965d18b6707e88e81bf7109`. The candidate keeps:

- seed `3407`, data, annotations, detector, evaluator, optimizer schedules,
  epoch counts, EMA semantics and diagnostic cadence;
- Stage 1: 30 epochs of exact uniform sampling, with contribution
  distillation and detector-gradient feedback disabled in both arms;
- Stage 2: 60 epochs of learned sampling rate, `cls+reg` contribution
  distillation, `density_transport_st` detector gradient and full ASFormer
  adaptation;
- diagnostic-only intermediate validation and terminal
  `epoch_59/state_dict_ema` OpenTAD official mAP as the sole performance
  endpoint.

The single changed model variable is the sampling budget. Its required
shape consequences are:

| Surface | Anchor | Candidate |
|---|---:|---:|
| dense temporal window | 768 | 768 |
| selected frames | 384 | 192 |
| temporal chunks | 24 | 12 |
| VideoMAE total frames | 384 | 192 |
| detector projection length | 384 | 192 |

Focused resolved-config tests normalize both arms and reject every difference
outside these budget, required-shape, contract-label, profile and work-directory
fields.

## Implementation and verification

- Branch: `codex/duca-rate25-curriculum-20260727`
- Exact deployment commit:
  `ed0d4900bffe3546997ea1f00ae806d82cad55f2`
- Remote clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_rate25_ed0d490_20260727`
- Linux contract/two-stage tests: `18 passed`
- Linux C3 regression tests: `23 passed`
- Python compilation and launcher syntax checks: passed

Precheck Job `1193418` failed before any update because the launcher parsed the
strict Stage-2 config before declaring the expected Stage-1 artifact path. This
was a launcher-order failure, not model evidence. The minimal repair declares a
precheck-only pending hash and the deterministic future checkpoint path before
config parsing; immediately before real Stage-2 construction it still requires
the Stage-1 epoch-29 file and replaces the pending hash with the artifact's
actual SHA-256. No `strict=False`, missing-key tolerance or checkpoint reuse was
introduced.

Corrected GPU precheck Job `1193433` completed `0:0`. Its manifest SHA-256 is
`0b8d4270939cc9ff7b25fb4831a5be107938135738e42bdefc70a4d26315af3f`.

## Formal run

- Job: `1193437`
- State at `2026-07-27 01:31 +08:00`: `RUNNING`, Stage-1 epoch 0 on `g0015`
- Run root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate25_ed0d490_formal_20260727_013015`
- Submission logs:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate25_ed0d490_submit_20260727_012855`
- Slurm dependency: none
- Initial health: manifest commit matched; training entered epoch 0; no
  Traceback, OOM, non-finite loss or fail-closed receipt.

Do not submit a duplicate while Job `1193437` is pending or healthy. Do not
select a checkpoint from intermediate mAP. Report every five-epoch quality
diagnostic as learning-curve evidence only, and report performance only after
the terminal epoch-59 EMA official evaluation.

## Evidence boundary

This experiment can measure the performance and selection-quality shift from
50% to 25% under the complete current course. By itself it cannot estimate the
benefit of learned sampling versus uniform sampling at 25%, because no separate
matched K=192 terminal uniform control is part of this run.

## Runtime update at 2026-07-27 01:37 +08:00

Job `1193437` completed Stage-1 epoch 0 and entered epoch 1. The terminal
epoch-0 receipt is finite (`Loss=5.4105`, `cls_loss=0.8852`,
`reg_loss=0.4453`, memory `5994MB`) with exact requested/effective budget
`192/192`, `policy_alpha=0`, contribution weight `0` and detector-gradient
weight `0`, as required for uniform warmup.

One startup batch required two bounded AMP overflow restores: batch 17 replayed
at scales `32768` and `16384` before the epoch completed normally. This is an
accepted finite recovery event, not a non-finite-loss failure or model result.
No Traceback, OOM, fail-closed receipt or terminal mAP exists.

## Runtime update at 2026-07-27 02:08 +08:00

Job `1193437` completed Stage-1 epochs 0 through 4 and sealed
`checkpoint/epoch_4.pth`; its scheduled one-based epoch-5 diagnostic
evaluation is currently in progress. The epoch-4 terminal receipt remains
finite (`Loss=4.7960`, `cls_loss=0.4619`, `reg_loss=0.3513`, memory
`5994MB`) with exact requested/effective budget `192/192`,
`policy_alpha=0`, contribution weight `0` and detector-gradient weight `0`.

In addition to the two startup restores at epoch 0 batch 17, epoch 2 batch 69
required one bounded replay at scale `8192` and then completed normally. Thus
there are three accepted AMP replay attempts across two batches, with no
Traceback, OOM, non-finite-loss failure, replay exhaustion, selection pointer,
Stage-2 initialization or terminal offline TAD result. The in-progress
diagnostic cannot select a checkpoint.

## Runtime update at 2026-07-27 02:44 +08:00

Job `1193437` completed its one-based epoch-5 EMA diagnostic and entered
Stage-1 epoch 8. The early exact-uniform warmup curve is `8.730398%`
Average-mAP, with `19.988334/12.780593/6.844830/3.013880/1.024351%` at
tIoU `0.3/0.4/0.5/0.6/0.7`. It is a five-epoch learning-curve point from
`epoch_4.pth/state_dict_ema`, not a terminal offline TAD result and not a
checkpoint-selection signal.

The K=192 source snapshot remains clean at exact commit
`ed0d4900bffe3546997ea1f00ae806d82cad55f2`; the job has no dependency,
selection pointer, Traceback, OOM, non-finite-loss failure or fail-closed
receipt. The separate AP/AUC/Brier/ECE and transition-boundary quality export
has not started yet; the launcher runs that sealed-checkpoint diagnostic stage
after Stage-1 training rather than during the active five-epoch validation.

## Runtime update at 2026-07-27 03:14 +08:00

Job `1193437` completed its one-based epoch-10 EMA diagnostic and entered
Stage-1 epoch 10. The exact-uniform K=192 warmup curve is `27.82%`
Average-mAP, with `50.03/39.49/28.41/15.44/5.75%` at tIoU
`0.3/0.4/0.5/0.6/0.7`. This is an intermediate learning-curve point from
`epoch_9.pth/state_dict_ema`; it cannot select a checkpoint, establish
terminal offline TAD performance, or be compared as the learned K=192
selector result.

The formal job remains `RUNNING` on `g0015` at clean exact commit
`ed0d4900bffe3546997ea1f00ae806d82cad55f2`, with no dependency, selection
pointer, Traceback, OOM, non-finite-loss failure or fail-closed receipt. The
separate AP/AUC/Brier/ECE and transition-boundary quality export remains
deferred until the Stage-1 checkpoint set is sealed.

## Runtime update at 2026-07-27 03:37 +08:00

Job `1193437` remains `RUNNING` at the clean exact K=192 commit and has
entered Stage-1 epoch 14. The latest sealed checkpoint remains `epoch_9.pth`;
the next one-based epoch-15 EMA diagnostic is not yet complete. There is no
Stage-2 initialization, AP/AUC/Brier/ECE quality export, selection pointer,
Traceback, OOM, non-finite-loss failure or fail-closed receipt.

The K=384 comparison endpoint is now sealed: evaluation-only Job `1193610`
completed `0:0` and wrote the exact-commit epoch-59 EMA receipt at
`65.385724%` Average-mAP
(`80.193191/75.662461/68.607247/58.581766/43.883956%`). This does not alter
the K=192 attribution boundary: only K=192 terminal Stage-2 performance can
answer the 50%-to-25% full-course shift, and a separate matched K=192 uniform
terminal control is still required to isolate learned-selection gain.

## Runtime update at 2026-07-27 04:07 +08:00

Job `1193437` sealed its one-based epoch-15 EMA diagnostic from
`epoch_14.pth/state_dict_ema` and entered Stage-1 epoch 16. The exact-uniform
K=192 warmup curve is `35.616501%` Average-mAP, with
`59.176347/48.839082/36.417352/22.762136/10.887588%` at tIoU
`0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions. This is
intermediate learning-curve evidence only; it cannot select a checkpoint,
establish learned-selector performance, or serve as the terminal K=192
offline TAD result.

The exact source remains clean, the Slurm dependency is null, and no
`best_validation_ema.json`, Stage-2 initialization, Traceback, OOM,
non-finite-loss failure or fail-closed receipt exists. AP/AUC/Brier/ECE and
transition-boundary quality exports remain deferred until the complete
Stage-1 checkpoint set is sealed.

## Runtime update at 2026-07-27 05:07 +08:00

Job `1193437` sealed its one-based epoch-20 EMA diagnostic from
`epoch_19.pth/state_dict_ema` and entered Stage-1 epoch 23. The exact-uniform
K=192 warmup curve is `43.132056%` Average-mAP, with
`65.227910/55.799303/44.794554/32.149065/17.689446%` at tIoU
`0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions. This remains
intermediate learning-curve evidence only; it cannot select a checkpoint,
establish learned-selector performance, or serve as terminal K=192 offline
TAD evidence.

The job remains healthy at its clean exact commit with no Slurm dependency,
`best_validation_ema.json`, Stage-2 initialization, Traceback, OOM,
non-finite-loss failure or fail-closed receipt. AP/AUC/Brier/ECE and
transition-boundary quality exports remain deferred until Stage 1 seals all
scheduled checkpoints.

## Runtime update at 2026-07-27 06:07 +08:00

Job `1193437` sealed its one-based epoch-25 EMA diagnostic from
`epoch_24.pth/state_dict_ema`. The exact-uniform K=192 warmup curve is
`49.091036%` Average-mAP, with
`69.158801/61.778057/51.280304/38.465483/24.772537%` at tIoU
`0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions. This remains
intermediate learning-curve evidence only and cannot select a checkpoint or
establish learned-selector/terminal K=192 performance.

All 30 Stage-1 training epochs then completed and sealed
`checkpoint/epoch_29.pth` with SHA-256
`141e4c1f3ce7b1b11a477fecf59478694055b8897102180137f007a825fe2595`.
The one-based epoch-30 EMA evaluation is still in progress, and Stage-2 strict
initialization has not begun. The clean exact source and null dependency are
unchanged; no selection pointer, Traceback, OOM, non-finite-loss failure or
fail-closed receipt exists. AP/AUC/Brier/ECE and transition-boundary quality
exports must complete before interpreting the Stage-1 evidence set.

## Runtime update at 2026-07-27 06:37 +08:00

The one-based epoch-30 EMA diagnostic from the sealed
`epoch_29.pth/state_dict_ema` completed at `51.954148%` Average-mAP, with
`70.747919/64.400867/54.530413/41.835973/28.255569%` at tIoU
`0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions. This closes
the Stage-1 exact-uniform warmup learning curve, but it is not the terminal
full-course K=192 offline TAD result and cannot be used to select a
checkpoint.

The launcher has begun the contract-required selection-quality export for
one-based epoch 5 from its sealed EMA checkpoint; 211 validation videos are
being processed and no analyzer summary exists yet. AP/AUC/Brier/ECE and
transition-boundary support therefore remain pending, and strict Stage-2
initialization has not begun. No selection pointer, Traceback, OOM,
non-finite-loss failure or fail-closed receipt exists.

## Runtime update at 2026-07-27 07:10 +08:00

Job `1193437` remains `RUNNING` at clean exact commit
`ed0d4900bffe3546997ea1f00ae806d82cad55f2`. Stage-1 quality summaries are
now sealed for one-based epochs 5, 10 and 15:

| Epoch | Macro AP | Macro AUC | Pooled AP | Pooled AUC | Brier | ECE |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.332623 | 0.485973 | 0.302061 | 0.488838 | 0.233573 | 0.118097 |
| 10 | 0.378387 | 0.542316 | 0.347483 | 0.546702 | 0.222322 | 0.060039 |
| 15 | 0.406062 | 0.575196 | 0.363583 | 0.567331 | 0.217861 | 0.024695 |

The transition macro policy AUPRC at radii `r0/r1/r2/r4/r8` is
`0.030985/0.069763/0.106559/0.172717/0.280760` at epoch 5,
`0.029421/0.069232/0.107586/0.174352/0.282603` at epoch 10, and
`0.031783/0.071428/0.110247/0.177410/0.287755` at epoch 15. These are
coarse-evidence diagnostics, not official OpenTAD mAP.

Because this phase is exact-uniform warmup, the selected geometry is
unchanged across these checkpoints: action enrichment `0.996621`, mean
endpoint distance `0.964181`, mean maximum unselected hole `3.735113`,
boundary recall `0.276691/0.759128/1.0/1.0/1.0`, and bilateral endpoint
coverage `0.089063/0.584968/1.0/1.0/1.0` at radii `r0/r1/r2/r4/r8`.
Every paired learned-minus-uniform quantity is exactly zero by construction.
This is a contract sanity check and must not be reported as learned-selector
gain.

The summaries cover 211 videos, 487 windows and 355,592 frame observations.
GT is evaluation-only, the budget is matched, and there are zero budget or
max-hole contract violations; short valid windows explain the observed
effective count range `112..192` under requested `K=192`. Epoch-20 quality
export is currently running. Stage-2 has not initialized, and there is still
no `best_validation_ema.json`, Traceback, OOM, non-finite event or fail-closed
receipt.

## Runtime update at 2026-07-27 07:37 +08:00

The one-based epoch-20 and epoch-25 Stage-1 quality summaries have completed:

| Epoch | Macro AP | Macro AUC | Pooled AP | Pooled AUC | Brier | ECE |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.410136 | 0.578777 | 0.364094 | 0.570474 | 0.216209 | 0.012080 |
| 25 | 0.410307 | 0.579620 | 0.367067 | 0.571312 | 0.215715 | 0.021642 |

The transition macro policy AUPRC at `r0/r8` is `0.030358/0.290133` at
epoch 20 and `0.031584/0.292148` at epoch 25. Relative to epoch 15, the
epoch-25 macro AP/AUC gains are only `+0.004245/+0.004424`, while pooled AP
gains `+0.003484`; the Stage-1 coarse evidence is therefore approaching a
plateau. ECE reaches `0.012080` at epoch 20 and rebounds to `0.021642` at
epoch 25, still below the epoch-15 value. These remain diagnostic trends,
not checkpoint-selection or terminal offline TAD evidence.

As required for exact-uniform Stage 1, selection geometry and every paired
learned-minus-uniform delta remain unchanged and exactly matched. The
epoch-30 quality export is processing at least 242 of 487 windows. Job
`1193437` remains healthy with null dependency; source HEAD is independently
verified clean at
`ed0d4900bffe3546997ea1f00ae806d82cad55f2`, matching `manifest.json`.
Stage-2 has not initialized, and focused scans found no Traceback, OOM,
runtime error, non-finite event, `best_validation_ema.json` or fail-closed
receipt.

## Runtime update at 2026-07-27 08:07 +08:00

The one-based epoch-30 Stage-1 quality summary completed, sealing all six
scheduled checkpoints:

| Epoch | Macro AP | Macro AUC | Pooled AP | Pooled AUC | Brier | ECE |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0.418318 | 0.584329 | 0.372005 | 0.574455 | 0.215742 | 0.021818 |

Epoch-30 transition macro policy AUPRC is `0.032568/0.294064` at `r0/r8`.
Across epochs 5 to 30, macro AP/AUC improved by
`+0.085695/+0.098356`, while Brier/ECE fell by
`-0.017831/-0.096279`. The exact-uniform selection geometry and every
learned-minus-uniform comparison remain invariant by construction, so this
closes the Stage-1 coarse-evidence curve without creating learned-selector or
terminal offline TAD evidence.

Stage-2 started at `07:46:38` from
`epoch_29.pth/state_dict_ema`, declared epoch 29, with observed SHA-256
`141e4c1f3ce7b1b11a477fecf59478694055b8897102180137f007a825fe2595`.
That SHA matches the sealed Stage-1 artifact and the configured binding. The
initializer source performs `load_state_dict(..., strict=True)` and resets
only the approved `frame_selector._loss_weight_schedule_step`; the runtime log
acknowledges the same checkpoint, state key and epoch. This is the required
strict full-model initialization, not a cross-K or partial-state load.

By completion of Stage-2 epoch 2, `update_audit.json` records 300 attempted
batches, 300 successful optimizer updates, 300 EMA updates, 300 scheduler
updates and 300 DUCA schedule updates. One AMP-skipped attempt was restored
and replayed successfully on its first bounded retry; there were zero
non-finite-loss attempts, non-finite replays or replay exhaustions. This is an
acceptable isolated AMP event, not an infinite-loss failure. Stage-2 epoch 3
has started with finite loss (`4.4504` at step 299), finite cls/reg losses
(`0.2594/0.2745`) and exact requested/effective budget `192/192`.

The current `duca_detector_grad_w=0` and zero teacher-utility term are the
pre-bridge portion of the registered homotopy, not disabled plugins: the
contract delays the detector bridge for 2,100 successful steps and then ramps
it over 1,500 steps toward weight `0.25`. Contribution distillation,
`density_transport_st` and full ASFormer adaptation remain configured for
their scheduled phases. Job `1193437` is healthy with null dependency and a
clean exact source commit; there is no Traceback, OOM, runtime error,
non-finite loss, `best_validation_ema.json` or fail-closed receipt.

## Runtime update at 2026-07-27 08:37 +08:00

Stage-2 sealed its one-based epoch-5 intermediate EMA diagnostic from
`checkpoint/epoch_4.pth` at `53.426779%` Average-mAP, with
`71.631253/64.833503/56.501860/44.442697/29.724581%` at tIoU
`0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions. The
checkpoint SHA-256 is
`d611a51f50a889cf08048a303206e1a06db37403de94f0910350158588b09fd3`;
the structured intermediate receipt SHA-256 is
`a56f83cc09bd5ac3f58c723a8439d91ea347d024b74e3ac7e734a0b60ac908e9`.

Relative to the Stage-1 epoch-30 EMA diagnostic, this is `+1.472631pp`
Average-mAP, with
`+0.883334/+0.432636/+1.971447/+2.606724/+1.469012pp` across the five tIoU
thresholds. This is a registered intermediate learning-curve point only. It
must not select a checkpoint, serve as terminal K=192 performance, or be
attributed to learned selection versus uniform without the separate matched
K=192 terminal control.

The update audit at epoch 4 records 500 attempted batches and 500 successful
optimizer/EMA/scheduler/DUCA-schedule updates. Two isolated AMP-skipped
attempts were each restored and replayed successfully on retry 1/8; there are
zero non-finite-loss attempts or replay exhaustions. The run has entered
epoch 5. Direct detector-gradient and contribution-teacher weights are still
in their registered pre-bridge delay, so the current intermediate gain cannot
be assigned to those later scheduled terms. Job state, null dependency,
clean exact source, exact `K=192` budget and no-best-checkpoint contract remain
healthy; no Traceback, OOM, runtime error, non-finite loss or fail-closed
receipt exists.

## Runtime update at 2026-07-27 09:37 +08:00

Stage-2 sealed its one-based epoch-10 intermediate EMA diagnostic from
`checkpoint/epoch_9.pth` at `54.515667%` Average-mAP, with
`72.941354/65.978947/57.882904/45.013669/30.761460%` at tIoU
`0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions. Checkpoint
SHA-256 is
`b5169d2949cc9a89ab0a286c1e61a6e567465be4b36cd4ca32e1534ac91ef63d`;
the structured receipt SHA-256 is
`e91049677bca9f8685520c487f04872dd4bf9f708f4834ba6f2a5218c5fcf6bd`.

This is `+1.088889pp` over the epoch-5 Stage-2 point and `+2.561519pp` over
the Stage-1 epoch-30 diagnostic. Versus epoch 5, the per-threshold changes are
`+1.310101/+1.145444/+1.381044/+0.570972/+1.036879pp`. It remains
non-selecting intermediate evidence and cannot establish terminal K=192
performance or learned-selector gain over a matched K=192 uniform endpoint.

The schedule has now crossed the first plugin-activation boundary. At
successful step 1,099, cls/reg contribution-distillation losses are finite
and nonzero (`0.0115/0.0113`) and `duca_detector_grad_w=0.0015`, directly
confirming that contribution distillation and the `density_transport_st`
gradient path have entered training. The separate
`duca_detector_utility_w` remains zero at this point, so later utility-bridge
effects are not yet present.

After the evaluation, epoch 10 completed and the audit reached 1,100
successful optimizer/EMA/scheduler/DUCA-schedule updates. The cumulative AMP
skip count remains two, both recovered by one bounded replay; non-finite-loss
attempts and replay exhaustions remain zero. Epoch 11 is running. Job state,
null dependency, clean exact commit, exact-budget contract and absence of
`best_validation_ema.json` or hard failure remain unchanged.

## Runtime update at 2026-07-27 10:16 +08:00

Stage-2 sealed its one-based epoch-15 intermediate EMA diagnostic from
`checkpoint/epoch_14.pth` at `55.403415%` Average-mAP, with
`73.319852/67.646858/58.673394/45.968481/31.408491%` at tIoU
`0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions. Checkpoint
SHA-256 is
`c56240b3181b5907555f07ff16c838b9a005e0ab2fe8169306dae009a269e94e`;
the structured receipt SHA-256 is
`0465943b24adac7647e6a8232d40b7a275e176a91e16cb691b118b72d07873b6`.

This is `+0.887748pp` over the epoch-10 Stage-2 point and `+3.449267pp`
over the Stage-1 epoch-30 diagnostic. Versus epoch 10, the per-threshold
changes are
`+0.378498/+1.667911/+0.790490/+0.954812/+0.647031pp`. The strongest
five-epoch increment is now at tIoU 0.4. This remains a registered,
non-selecting learning-curve point: it cannot select a checkpoint, establish
terminal K=192 performance, or isolate learned selection from the missing
matched K=192 uniform endpoint.

The epoch-14 audit records 1,500 attempted batches and 1,500 successful
optimizer/EMA/scheduler/DUCA-schedule updates. A third isolated AMP skip was
restored and replayed successfully on retry 1/8; non-finite-loss attempts,
non-finite-loss replays and replay exhaustions remain zero. At step 1,499,
cls/reg contribution-distillation losses remain finite and nonzero
(`0.7043/0.6853`) and `duca_detector_grad_w=0.0365`; requested and effective
budgets remain exactly `192/192`. The separate detector-utility weight is
still zero under its registered delay. Epoch 15 has started. Job state, null
dependency, clean exact source, no-best-checkpoint contract and hard-error
scans remain healthy.

## Runtime update at 2026-07-27 11:07 +08:00

Stage-2 sealed its one-based epoch-20 intermediate EMA diagnostic from
`checkpoint/epoch_19.pth` at `56.050489%` Average-mAP, with
`73.483220/67.033731/58.709834/47.921247/33.104414%` at tIoU
`0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions. Checkpoint
SHA-256 is
`5e213343bd5f2f4994d47ca7b042c7c41f9e40a4705d096b133333ee454ae276`;
the structured receipt SHA-256 is
`39e743dcc3585b0cd2fa05e97dc0a6e109bca490d110a777a17706403f909e01`.

This is `+0.647074pp` over the epoch-15 Stage-2 point and `+4.096341pp`
over the Stage-1 epoch-30 diagnostic. Versus epoch 15, the per-threshold
changes are
`+0.163368/-0.613127/+0.036440/+1.952766/+1.695923pp`. Thus this interval's
gain is concentrated at high tIoU 0.6/0.7, while tIoU 0.4 decreased. This
remains a registered, non-selecting learning-curve point and cannot establish
terminal K=192 performance or learned-selector gain over a missing matched
K=192 uniform endpoint. Its apparent `-9.335235pp` gap to the sealed K=384
terminal result is an intermediate-to-terminal comparison, not the final
cross-budget effect.

The epoch-19 audit records 2,000 attempted batches and 2,000 successful
optimizer/EMA/scheduler/DUCA-schedule updates. Four isolated AMP skips have
each been restored and replayed successfully on retry 1/8; non-finite-loss
attempts, non-finite-loss replays and replay exhaustions remain zero. At step
1,999, cls/reg contribution-distillation losses are finite
(`2.5477/2.5470`), `duca_detector_grad_w=0.1248`, and requested/effective
budget is exactly `192/192`; the separate detector-utility weight remains
zero. Total loss increased to a finite `7.2839` as the contribution terms
grew, so its later trend remains a diagnostic to monitor rather than a hard
failure. Epoch 20 has started. Job state, null dependency, clean exact source,
no-best-checkpoint contract and hard-error scans remain healthy.

## Failure isolation and epoch-24 recovery at 2026-07-27 12:19 +08:00

Formal Job `1193437` ended `FAILED/1:0` at `11:43:54 +08:00`, after Stage-2
epoch 24 had completed and `checkpoint/epoch_24.pth` had been durably saved.
The failure occurred 147/396 batches into the one-based epoch-25
non-selecting EMA evaluation. A Decord worker exhausted the default 10,240
EOF retries while retrieving the final frames from a `BytesIO` video. This
is an evaluation data-decoder failure, not OOM, non-finite loss, a model
update failure, or terminal K=192 performance. It produced no epoch-25 mAP
receipt, and no training update occurred after the epoch-24 checkpoint.

The sealed recovery state is:

- Stage-1 `epoch_29.pth`, SHA-256
  `141e4c1f3ce7b1b11a477fecf59478694055b8897102180137f007a825fe2595`;
- Stage-2 `epoch_24.pth`, SHA-256
  `d37cad6e1fcbf9078f9e186c0735f291461332572df67ef6df16ab05db3c00f6`,
  containing strict model/EMA, optimizer, scheduler, GradScaler and epoch
  state at scheduler and selector step 2,500;
- source update audit SHA-256
  `ec1536dfd68d16144e242c8ca7ee10828b5de05d5fea4f26b968e21b7a1dcf9d`,
  with 2,500 successful optimizer/EMA/scheduler/DUCA-schedule updates, 2,506
  optimizer attempts, six bounded AMP skips, five replayed batches, maximum
  retry 2/8, and zero non-finite-loss attempts or replay exhaustion;
- source Stage-2 train log SHA-256
  `009d2cee168c27071b7ecc9fd2d25c07d6812898fe33ea41371ecc4e4dd04467`;
- no `best_validation_ema.json` and therefore no intermediate checkpoint
  selection.

GPU precheck Job `1194469` completed `0:0`. Its recovery manifest SHA-256 is
`44a545805e11051a80c91f834292915efee2c870a9d0c54265b24d097c9a8d75`;
the independent recovery launcher SHA-256 is
`66353a213053bc0981bf349734ee1e699c9e312722e2c8a36f5f620e481229bf`.
The only runtime repair is Decord's own bounded retry setting,
`DECORD_EOF_RETRY_MAX=20480`; model code, config, exact commit
`ed0d4900bffe3546997ea1f00ae806d82cad55f2`, budget and all sampling plugins
remain unchanged. All state-dict loads remain strict.

Formal recovery Job `1194471` started at `12:19:03 +08:00` with null
dependency and a fresh root
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate25_ed0d490_e24_recovery_20260727_121900`.
Before any new update it must successfully reproduce the missing epoch-25
EMA evaluation from the sealed source checkpoint. Only then may it resume
epochs 25--59. The continuation audit must contain exactly 3,500 successful
updates and combine with the sealed source audit to exactly 6,000; terminal
evidence remains epoch-59 EMA official OpenTAD mAP only.

The source checkpoint does not contain global Python/NumPy/PyTorch RNG state.
The recovery therefore restores model, EMA, optimizer, scheduler, GradScaler
and DUCA schedule exactly, then uses the registered deterministic seed 3407,
but it is not a bit-exact continuation of the original random stream. This
limitation is sealed in the recovery manifest and must accompany any terminal
claim; it is protocol provenance, not model-performance evidence.

At `12:26:54 +08:00`, Job `1194471` reached 169/396 evaluation batches,
crossing the source failure position of 147/396 without Decord error,
Traceback, OOM or non-finite evidence. No continuation work directory or new
training audit exists yet, so the evaluate-before-update gate remains intact.
This validates the bounded decoder repair at the observed failure boundary;
it is not an mAP result and does not yet prove completion of the full
evaluation.

## Recovered epoch-25 EMA diagnostic at 2026-07-27 12:41 +08:00

Job `1194471` completed all 396 evaluation batches and sealed the missing
one-based epoch-25 EMA diagnostic from source `epoch_24.pth/state_dict_ema`.
Official OpenTAD Average-mAP is `56.646995%`, with
`73.566280/67.812407/59.759908/48.284381/33.811996%` at tIoU
`0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions. The
prediction SHA-256 is
`17c9fce0f909eed7d08e82ba3cd133c68bf681c75635b3e2edeb946c1674d422`;
the structured evaluation receipt SHA-256 is
`e9069dc8d621268014f32761759f3a590f2ff9d85ac822a9ec09425423894638`.
It binds exact commit
`ed0d4900bffe3546997ea1f00ae806d82cad55f2`, checkpoint SHA-256
`d37cad6e1fcbf9078f9e186c0735f291461332572df67ef6df16ab05db3c00f6`,
the validation annotations and evaluator source.

The point is `+0.596506pp` over the epoch-20 EMA diagnostic and
`+4.692847pp` over the Stage-1 epoch-30 EMA endpoint. Relative to epoch 20,
the per-threshold changes are
`+0.083060/+0.778676/+1.050074/+0.363134/+0.707582pp`. Its apparent
`-8.738729pp` gap to the sealed K=384 terminal endpoint is
intermediate-to-terminal and must not be reported as the final 25%-versus-50%
budget effect. Without a terminal K=192 matched-uniform control, it also
cannot establish learned-selector gain over uniform.

The evaluate-before-update gate passed at `12:37:51`; only afterward did the
launcher strictly restore the sealed epoch-24 model/EMA, optimizer,
scheduler, GradScaler and DUCA schedule and start epoch 25 at `12:38:28`.
The recovery remains subject to the recorded global-RNG continuity
limitation. At `12:41:24`, Job `1194471` is running with null dependency and
no Decord error, Traceback, OOM, non-finite evidence or checkpoint-selection
pointer. This mAP remains a non-selecting intermediate learning-curve point,
not terminal K=192 offline TAD performance.

By `12:44:08 +08:00`, the continuation completed epoch 25 and entered epoch
26. The first 100 resumed updates are finite and advance the restored selector
schedule from step 2,500 through 2,599. At the final epoch-25 batch, total
loss is finite (`11.6996`), cls/reg losses are finite
(`0.2304/0.2485`), cls/reg contribution-distillation losses are finite and
active (`5.1453/5.0894`), `duca_detector_grad_w=0.2260`, and requested and
effective budgets remain exactly `192/192`. No hard-error signature or replay
exhaustion is present. The large but finite contribution terms remain a trend
diagnostic, not a non-finite failure.

## Recovery audit at one-based epoch 30, 2026-07-27 13:07 +08:00

Job `1194471` completed continuation epochs 25--29 and sealed
`checkpoint/epoch_29.pth` with SHA-256
`4ec40e031af6087ff4db509df333e62da3514440d55d3615907cdd8ec2acd2dc`.
The checkpoint contains strict model/EMA, optimizer, scheduler and GradScaler
state at epoch 29, scheduler step 3,000, selector schedule step 3,000 and
finite GradScaler scale 1,024. The continuation update audit SHA-256 is
`fc9860d56fc80980f1bdedd050ebf390f15644daf5f16f7b4fbb29576f1f81f4`;
it records exactly 500 successful optimizer/EMA/scheduler/DUCA-schedule
updates, 500 optimizer attempts, zero AMP-skipped attempts, zero non-finite
loss attempts and zero replay exhaustion.

At the final epoch-29 batch, total loss remains finite (`12.4112`), cls/reg
losses are `0.2235/0.2378`, cls/reg contribution-distillation losses are
`5.6252/5.5110`, and the registered homotopy reaches
`duca_detector_grad_w=0.2500` and schedule progress `1.0000`. Requested and
effective budget is `192/192` there. Two earlier logged batch means had
effective budgets `128.5` and `153.0`; this is not global budget drift:
the exact implementation defines `effective_k=min(K, valid_len)`, and the
source Job `1193437` contains the same short-valid-sequence cases. The
requested budget remains 192 and ordinary full-length batches remain
`192/192`.

The non-selecting one-based epoch-30 EMA evaluation from the sealed
epoch-29 checkpoint started immediately after save and is still running.
There is no new mAP yet, no `best_validation_ema.json`, and no Decord error,
Traceback, OOM, non-finite event or fail-closed receipt.

## Recovered epoch-30 EMA diagnostic at 2026-07-27 13:38 +08:00

Job `1194471` completed the one-based epoch-30 EMA evaluation from sealed
`checkpoint/epoch_29.pth/state_dict_ema`. Official OpenTAD Average-mAP is
`57.464558%`, with
`74.527791/69.177787/60.361168/49.020347/34.235695%` at tIoU
`0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions. The
structured intermediate-evaluation JSON SHA-256 is
`30fdb7579505e04619f932a0673701ed6733152540ad7092733b4c225750ea39`;
it binds `epoch_29.pth`, `state_dict_ema`, the evaluator source SHA-256
`e855e70d41d087d039a90ecdb8f3cc3efece209130417320edf35062b8503fd4`
and the metric vector. This diagnostic did not retain a raw-prediction path.

The point is `+0.817563pp` over the recovered epoch-25 EMA diagnostic and
`+5.510410pp` over the Stage-1 epoch-30 EMA endpoint. Its apparent
`-7.921166pp` gap to the sealed K=384 terminal endpoint remains an
intermediate-to-terminal comparison, not the final 25%-versus-50% budget
effect. It cannot select a checkpoint, establish terminal K=192 offline TAD
performance, or isolate learned selection from the missing matched K=192
uniform endpoint.

After evaluation, continuation epochs 30 and 31 completed and epoch 32
started. The `13:38 +08:00` continuation-audit snapshot SHA-256 is
`8f33d67200fcacb8bd0a91c53d0cd4b479a8d7e35b7e7449403c69f70fb15f80`;
it records exactly 700 successful optimizer/EMA/scheduler/DUCA-schedule
updates and 700 attempts, with zero AMP skips, non-finite losses, replay
restorations or replay exhaustion. At the final epoch-31 batch, total loss
is finite (`12.2577`), cls/reg losses are `0.2188/0.2354`, active cls/reg
contribution-distillation losses are `5.4511/5.5401`,
`duca_detector_grad_w=0.2500`, schedule progress is `1.0000`, and requested
and effective budget is `192/192`. Job `1194471` remains running with null
dependency and no Decord error, Traceback, OOM, hard non-finite event,
`FAIL` receipt or `best_validation_ema.json`. The sealed global-RNG
continuity limitation still applies.

## Recovered epoch-35 EMA diagnostic at 2026-07-27 14:12 +08:00

Job `1194471` completed continuation epochs 32--34 and sealed
`checkpoint/epoch_34.pth` with SHA-256
`2da7de83c0e9feb1f3b267deed7a41593680305b2491d47d4352a81254ac4a02`.
The non-selecting one-based epoch-35 EMA evaluation from
`epoch_34.pth/state_dict_ema` then completed with official OpenTAD
Average-mAP `57.921948%`, with
`74.668088/69.304388/61.149872/49.416606/35.070788%` at tIoU
`0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions. The
structured intermediate-evaluation JSON SHA-256 is
`00d22c18c188738d63b9f065a912ba08dfb74cf5d78df7e24d2fabe48d1784cc`;
it binds the checkpoint path, `state_dict_ema` and evaluator source SHA-256
`e855e70d41d087d039a90ecdb8f3cc3efece209130417320edf35062b8503fd4`.

The point is `+0.457390pp` over epoch 30, `+1.274953pp` over epoch 25 and
`+5.967800pp` over the Stage-1 endpoint. Relative to epoch 30, per-threshold
changes are
`+0.140296/+0.126601/+0.788703/+0.396260/+0.835092pp`, so the latest
five-epoch improvement is concentrated at tIoU 0.5 and 0.7. Its apparent
`-7.463776pp` gap to the sealed K=384 terminal endpoint remains an
intermediate-to-terminal comparison. It cannot select a checkpoint,
establish terminal K=192 offline TAD performance, or identify learned
selection gain without a matched K=192 uniform endpoint.

The continuation audit at epoch 34 records exactly 1,000 successful
optimizer/EMA/scheduler/DUCA-schedule updates and 1,000 attempted batches.
One epoch-33 batch produced three bounded AMP-overflow skips at replay
attempts 1--3/8, with scale reduced from 512 to 128; exact state restoration
allowed the same batch to complete afterward. Thus optimizer attempts are
1,003, replay-state restorations are three and replayed-batch count is one,
while non-finite-loss attempts, loss replays and replay exhaustion remain
zero. This is an accepted finite transient under the registered bounded AMP
contract, not model-failure evidence. Job `1194471` entered epoch 35 at
`14:11:11`, remains running from a clean detached snapshot at exact commit
`ed0d4900bffe3546997ea1f00ae806d82cad55f2` with null dependency, and has no
Decord error, Traceback, OOM, hard non-finite event, `FAIL` receipt or
`best_validation_ema.json`. The global-RNG continuity limitation remains
sealed.

## Recovered epoch-40 EMA diagnostic at 2026-07-27 15:08 +08:00

Job `1194471` completed continuation epochs 35--39 and sealed
`checkpoint/epoch_39.pth` with SHA-256
`c1a9c6393920f1189000800362564239d5b9ee38ef94c00373f6fe6551b1f445`.
The non-selecting one-based epoch-40 EMA evaluation from
`epoch_39.pth/state_dict_ema` completed with official OpenTAD Average-mAP
`58.116412%`, with
`73.838536/69.371290/61.536349/50.339880/35.496002%` at tIoU
`0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions. The
structured intermediate-evaluation JSON SHA-256 is
`5ef548dd4ab4cb8f7bdadc5938ce3670d1c075620a35279df39b99cc953f547e`;
it binds `epoch_39.pth`, `state_dict_ema` and evaluator source SHA-256
`e855e70d41d087d039a90ecdb8f3cc3efece209130417320edf35062b8503fd4`.

The point is `+0.194464pp` over epoch 35, `+0.651854pp` over epoch 30 and
`+6.162264pp` over the Stage-1 endpoint. Relative to epoch 35, per-threshold
changes are
`-0.829552/+0.066902/+0.386477/+0.923274/+0.425214pp`: the average gain
comes from tIoU 0.4--0.7, especially tIoU 0.6, while tIoU 0.3 regresses.
Its apparent `-7.269312pp` gap to the sealed K=384 terminal endpoint remains
an intermediate-to-terminal comparison. This point cannot select a
checkpoint, establish terminal K=192 offline TAD performance, or identify
learned-selection gain without a matched K=192 uniform endpoint.

The epoch-40 continuation audit SHA-256 is
`8bb3290894c0d3642a5326149313272c1357e862445468dcce3dd344a2cd21ee`.
It records exactly 1,600 successful optimizer/EMA/scheduler/DUCA-schedule
updates from 1,600 attempted batches and 1,603 optimizer attempts. The only
three AMP skips and state restorations remain the previously sealed single
epoch-33 replayed batch; maximum observed retries remain 3/8. Non-finite-loss
attempts, loss replays and all replay exhaustions remain zero. At the final
epoch-40 batch, total loss is finite (`12.3632`), cls/reg losses are
`0.1950/0.2267`, active cls/reg contribution-distillation losses are
`5.5061/5.6228`, detector-gradient weight is `0.2500`, and schedule progress
is `1.0000`. The `172/192` effective/requested budget there is a known
`effective_k=min(K, valid_len)` short-valid-sequence cap, not global budget
drift. By `15:11 +08:00`, Job `1194471` completed epoch 41 and entered epoch
42; it remains running with null dependency from the clean detached exact commit
`ed0d4900bffe3546997ea1f00ae806d82cad55f2`, and has no Decord error,
Traceback, OOM, hard non-finite event, `FAIL` receipt or
`best_validation_ema.json`. The sealed global-RNG continuity limitation
still applies.

## Runtime update at 2026-07-27 15:37 +08:00

Recovery Job `1194471` completed continuation epochs 40--44 and sealed
`checkpoint/epoch_44.pth` with SHA-256
`fabe373abd4e7f2f982bf6a6fad26e7d022930ea4da8789d400f613200c8c9ea`.
The epoch-44 continuation-audit SHA-256 is
`6991f412001928853a3976baa07dc5c2b4bb46c288cc9788e3ad2c4bb2219413`;
it records exactly 2,000 successful optimizer/EMA/scheduler/DUCA-schedule
updates from 2,000 attempted batches and 2,003 optimizer attempts. The three
AMP skips/restorations remain confined to the previously sealed single
epoch-33 batch; non-finite-loss attempts and all replay exhaustions remain
zero.

The non-selecting one-based epoch-45 EMA evaluation is in progress and has no
metric JSON yet. Job `1194471` remains running with null dependency from a
clean detached snapshot at exact commit
`ed0d4900bffe3546997ea1f00ae806d82cad55f2`; no Decord error, Traceback, OOM,
hard non-finite event, `FAIL` receipt or `best_validation_ema.json` exists.
The latest completed performance point therefore remains epoch 40, not epoch
45 or the terminal epoch 59.

## Recovered epoch-45 EMA diagnostic at 2026-07-27 15:44 +08:00

Recovery Job `1194471` completed the non-selecting one-based epoch-45 EMA
evaluation from sealed `checkpoint/epoch_44.pth/state_dict_ema`. Official
OpenTAD Average-mAP is `57.877041%`, with
`74.416763/68.892371/61.103833/49.558953/35.413287%` at tIoU
`0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions. The
checkpoint SHA-256 remains
`fabe373abd4e7f2f982bf6a6fad26e7d022930ea4da8789d400f613200c8c9ea`;
the intermediate-evaluation JSON SHA-256 is
`4e13c794f1cf5ad709081e227833abf422be245f102ce97feb7a9f2f2b902670`.

The point is `-0.239371pp` versus epoch 40 and `+5.922893pp` over the
Stage-1 endpoint. Relative to epoch 40, tIoU 0.3 changes by `+0.578227pp`
while tIoU 0.4--0.7 change by
`-0.478919/-0.432516/-0.780927/-0.082715pp`. Its apparent `-7.508683pp`
gap to the sealed K=384 terminal endpoint remains an intermediate-to-terminal
comparison. It cannot select a checkpoint, establish terminal K=192 offline
TAD performance, or isolate learned-selection gain without a matched K=192
uniform endpoint.

The evaluation completed without the earlier Decord failure and Job
`1194471` entered continuation epoch 45 at `15:43:49 +08:00`. The sealed
epoch-44 audit still records 2,000 successful updates, three bounded AMP
restores from one epoch-33 batch, zero non-finite-loss attempts and zero
replay exhaustion. The job remains running with null dependency from exact
commit `ed0d4900bffe3546997ea1f00ae806d82cad55f2`, with no intermediate
checkpoint-selection pointer. The global-RNG continuity limitation remains
sealed.

## Recovered epoch-50 and epoch-55 EMA diagnostics at 2026-07-27 17:21 +08:00

Recovery Job `1194471` completed the non-selecting one-based epoch-50 EMA
evaluation from `checkpoint/epoch_49.pth/state_dict_ema`. Official OpenTAD
Average-mAP is `58.383005%`, with
`74.477950/69.245107/61.193389/50.712331/36.286249%` at tIoU
`0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions. The
checkpoint SHA-256 is
`90d6c4cb791a1b908c5c4cfcf2123a1c3aac3a721498ce26b2b18a819d708161`;
the intermediate-evaluation JSON SHA-256 is
`6ca0f449398bf75686921b38e7767a697b6cacc97b23b83e41e57ab1554f1a22`.
This is `+0.505964pp` versus epoch 45 and `+6.428857pp` over the Stage-1
endpoint.

The one-based epoch-55 EMA evaluation then completed from
`checkpoint/epoch_54.pth/state_dict_ema`. Official OpenTAD Average-mAP is
`58.082562%`, with
`73.934313/69.065870/61.031079/49.908786/36.472761%` at tIoU
`0.3/0.4/0.5/0.6/0.7`, again over 211 videos and 422,000 predictions. The
checkpoint SHA-256 is
`59d24803e63efd2d0177e5d5baf106fc4dd0c01a737c2053740b1c3148177fcc`;
the intermediate-evaluation JSON SHA-256 is
`77cdc89b79d4a460a7024f71c914739b153e33c927c588ccc19f76ec3b8867cb`.
This is `-0.300443pp` versus epoch 50 and `+6.128414pp` over Stage-1.
Per-threshold changes from epoch 50 are
`-0.543637/-0.179237/-0.162310/-0.803545/+0.186512pp`.

The epoch-54 continuation audit SHA-256 is
`a42d6e162b96b5fa6cb96c56eff8144be3911db3a56292106f65a2d8054365be`.
It records exactly 3,000 successful optimizer/EMA/scheduler/DUCA-schedule
updates from 3,000 attempted batches and 3,003 optimizer attempts. Combined
with the sealed source's 2,500 updates, the course has completed 5,500 of
6,000 Stage-2 updates. The only continuation AMP skips/restorations remain
three attempts from the single epoch-33 batch; non-finite-loss attempts,
loss replays and all replay exhaustions remain zero. Job `1194471` entered
epoch 55 at `17:16:00 +08:00`, remains `RUNNING` with null dependency from
the clean detached exact commit
`ed0d4900bffe3546997ea1f00ae806d82cad55f2`, and has no Decord error,
Traceback, OOM, hard non-finite event, `FAIL` receipt or
`best_validation_ema.json`.

Both points are learning-curve diagnostics only. The slight epoch-55 decline
cannot select epoch 50 under this sealed course. Neither point is terminal
K=192 offline TAD performance or learned-selector evidence against the still
missing matched K=192 uniform endpoint. The sole terminal evidence remains
the predeclared epoch-59 EMA official evaluation.

## Epoch-59 training completion and terminal evaluation at 2026-07-27 17:49 +08:00

Recovery Job `1194471` completed continuation epochs 55--59 at
`17:44:07 +08:00` and sealed
`stage2/work/gpu1_id0/checkpoint/epoch_59.pth`, SHA-256
`4a5389506263b8fd76ca3de6ce3475dee64cc0d9ed1ca73c896692c8db288455`.
The continuation audit SHA-256 is
`36fc64d4542ce671b4c891f8b8270a51b629ad4b514165f80fd66b439a7451f0`.
It records exactly 3,500 successful optimizer, scheduler, EMA and DUCA
schedule updates from 3,500 attempted continuation batches. Combined with
the sealed source's 2,500 updates, Stage 2 has completed exactly
6,000/6,000 successful updates.

The only continuation AMP events remain three bounded overflow skips and
state restorations on one epoch-33 batch; that batch subsequently completed.
There are zero non-finite-loss attempts, loss replays, replay exhaustions or
failed-batch state advances. The final epoch-59 losses are finite:
total `12.4513`, classification/regression `0.1843/0.2128`, and
classification/regression contribution distillation `5.5674/5.6760`.
No `best_validation_ema.json` exists.

The predeclared terminal `epoch_59.pth/state_dict_ema` OpenTAD evaluation is
now running. At `17:49:28 +08:00`, it had processed 126/396 evaluation
batches. Job `1194471` remains `RUNNING` on `g0043` with null dependency from
exact commit `ed0d4900bffe3546997ea1f00ae806d82cad55f2`. There is no terminal
metric receipt yet and no Decord error, Traceback, OOM, hard non-finite event
or `FAIL` receipt. The terminal K=192 result must not be inferred from the
epoch-50/55 diagnostics; the sealed global-RNG continuity limitation remains
part of the recovery provenance.

## Sealed epoch-59 EMA terminal result at 2026-07-27 18:20 +08:00

The explicit terminal OpenTAD evaluator loaded the predeclared
`checkpoint/epoch_59.pth/state_dict_ema` and reproduced the automatic
one-based epoch-60 EMA diagnostic exactly. Over 211 validation videos and
422,000 predictions, official Average-mAP is `57.967272%`, with
`73.907179/68.926135/61.194230/49.841145/35.967670%` at tIoU
`0.3/0.4/0.5/0.6/0.7`. The terminal checkpoint SHA-256 remains
`4a5389506263b8fd76ca3de6ce3475dee64cc0d9ed1ca73c896692c8db288455`.
The terminal receipt SHA-256 is
`febc59d463476bcf6a1a0d77f237a54f12c59ce3028ce623fba9844c07fada04`;
the prediction SHA-256 is
`719ba43b0f76f5647b2394a23b622aff0bac0c17a54d15613c4d9dbdb57d02d0`;
and the combined 6,000-update audit SHA-256 is
`f13ef3f8650c4fe75f795ddf255b32a377ea4b7c31d4bdc007f0121114ba97a1`.
The receipt binds exact commit
`ed0d4900bffe3546997ea1f00ae806d82cad55f2`, evaluator source SHA-256
`e855e70d41d087d039a90ecdb8f3cc3efece209130417320edf35062b8503fd4`,
and evaluation payload SHA-256
`5d9227079f3c5bbf5e02a951eb3ca85c3c96fc886285f90ca52fa1bebdecc4b5`.

Relative to the Stage-1 epoch-29 EMA diagnostic (`51.954148%`), the terminal
Stage-2 point is `+6.013124pp`. Relative to the sealed K=384 30+60 course
(`65.385724%`), it is `-7.418452pp`; this is a cross-budget comparison between
two over-budget courses, not a learned-selector-versus-uniform effect. It is
also `-0.415733pp` below the non-selecting epoch-50 diagnostic, and the sealed
contract therefore correctly prevents post-hoc checkpoint selection.

This result is terminal evidence only for the K=192 30+60 diagnostic course.
It consumed 90 full-model epochs, lacks a matched native K=192 uniform
terminal control, and inherited the documented non-bit-exact global-RNG
continuity limitation after the epoch-24 recovery. It is not a fair total-60
paper result, does not prove that learned sampling beats uniform sampling at
25%, and does not validate the final pure pre-backbone model. At `18:24
+08:00`, Slurm Job `1194471` remained `RUNNING` only for post-evaluation
selector-quality export; the formal stderr was empty, no
`best_validation_ema.json` existed, and no Decord error, Traceback, OOM, hard
non-finite event, replay exhaustion or `FAIL` receipt was present.

## Post-terminal Stage-2 quality epochs 5 and 10 at 2026-07-27 18:46 +08:00

The post-terminal exporter completed the one-based Stage-2 epoch-5 and
epoch-10 selection-quality summaries over 487 windows from all 211 validation
videos. Their SHA-256 values are respectively
`fcbeb7b77f3dc574639261399848baaf6e0172809a3d924d443d390529c01864`
and
`c6677d757340c8fc721fa289743f638087e2291ad3b812eca5c9c92e6c7d910d`.
Both summaries report zero budget and max-hole contract violations.

At epoch 5, coarse evidence has macro AUPRC/AUROC
`0.421082/0.585920`; pooled AUPRC/AUROC is
`0.376125/0.578415`, with Brier/ECE `0.215769/0.027737`. At epoch 10,
these values are `0.425132/0.590137`,
`0.378734/0.581840`, and `0.216125/0.035281`. Discrimination therefore
improves only slightly from epoch 5 to 10, while calibration becomes worse.

The learned-minus-matched-uniform geometry is small and mixed. At epoch 5,
action enrichment changes by `+0.014202`, boundary recall at radii 0/1 by
`-0.008874/+0.020527`, R2Q3 bilateral endpoint recall by `+0.004879`,
and R4Q5 bilateral endpoint recall by `-0.023647`. At epoch 10, the
corresponding changes are `+0.014851`,
`-0.006059/+0.013539`, `+0.014461`, and `-0.026231`. Learned selection
reduces mean endpoint distance by only `0.011653` and `0.007479` at the two
points, while increasing mean maximum hole by `0.024641` and `0.030801`.

These diagnostics do not show a strong, uniformly better learned sampling
geometry. The narrow-radius and R2Q3 changes are marginal, while exact-boundary
recall and wider R4Q5 bilateral support remain worse than matched uniform.
This is consistent with the terminal K=192 result failing to establish a
selector advantage, but it is explanatory evidence only: these summaries do
not replace the missing full-model matched-uniform terminal control. At
`18:47 +08:00`, Job `1194471` remained `RUNNING` while exporting epoch-15
quality records; formal stderr remained empty and no hard failure or
best-checkpoint pointer appeared.

## Post-terminal Stage-2 quality epochs 15 and 20 at 2026-07-27 19:10 +08:00

The one-based Stage-2 epoch-15 and epoch-20 selection-quality summaries
completed with SHA-256
`a8b0917ad46cac2921b2534318cec1eb89a2d62a1f9ce4d5bd299a44d1fb824e`
and
`3de85f7579650c007d23af1d22b73e461ab03226478fdd4c315a10c45eb7f757`.
Both cover 487 windows and 211 videos and record zero budget or max-hole
contract violations.

Coarse macro AUPRC/AUROC rises to `0.431067/0.594522` at epoch 15 and
`0.437216/0.599782` at epoch 20; pooled AUPRC/AUROC reaches
`0.381589/0.586731` and `0.387066/0.592015`. Brier remains nearly flat
at `0.216540/0.216509`, while ECE worsens further to
`0.045660/0.050933`. Thus ranking discrimination is improving slowly, but
the score is becoming progressively less calibrated.

At epoch 20, learned-minus-matched-uniform action enrichment remains only
`+0.013665`. Boundary recall changes by `+0.003399` at radius 0 and
`+0.017566` at radius 1. Only the radius-1 change excludes zero under the
video-cluster bootstrap (`95% CI [0.001394, 0.034104]`); the exact-radius
change does not (`[-0.019502, 0.026367]`). R2Q3 bilateral endpoint recall
changes by `+0.015936` with CI `[-0.006958, 0.033390]`, while R4Q5
bilateral endpoint recall changes by `-0.026594` with CI
`[-0.033483, -0.021020]`. Mean endpoint distance improves by only
`0.020965` with a zero-crossing CI, and mean maximum hole worsens by
`0.121150`.

The evidence now isolates a specific failure shape rather than a general
selector gain: the policy obtains a small radius-1 boundary-recall benefit,
but does not improve exact boundary support or R2Q3 paired support
significantly and consistently harms wider R4Q5 paired support. This weak
geometry is a plausible contributor to the low high-tIoU terminal result, but
it remains an explanatory association rather than a causal ablation. At
`19:17 +08:00`, Job `1194471` remained `RUNNING` and was exporting epoch-25
quality records; formal stderr remained empty and no new hard failure was
present.

## Post-terminal Stage-2 quality epochs 25, 30 and 35 at 2026-07-27 19:46 +08:00

The one-based Stage-2 epoch-25/30/35 selection-quality summaries completed
with SHA-256
`cfab3759813577fe1187d9f2a6c8340642c991f92577523eff8409d3ac5e8af6`,
`719da69f91bcab1decfac2dfe47600fa7d5f014151f4d51a505e4f59e0589f0a`,
and
`284f91c9dde2c01b8782a9ea6d3581ea07aa61eed10fc34556bc7ce0b0934ef6`.
Each covers 487 windows and 211 videos with zero budget or max-hole contract
violations.

Coarse macro AUPRC/AUROC increases across these points from
`0.439054/0.602468` to `0.440312/0.604029` and
`0.441664/0.605025`. Pooled AUPRC/AUROC similarly reaches
`0.397513/0.601299` at epoch 35. Brier improves slightly to `0.215398`,
while ECE remains poor at `0.048672`, still substantially above the epoch-5
value `0.027737`. Learned-minus-uniform action enrichment grows from
`+0.016965` at epoch 25 to `+0.024621` at epoch 35.

The geometry becomes more polarized rather than uniformly better. At epoch
35, radius-1 boundary recall improves `+0.031290` with 95% CI
`[0.013792, 0.053063]`, and mean endpoint distance improves `0.047062`
with CI `[0.013641, 0.078358]`. Exact-radius recall remains inconclusive at
`+0.016536` with CI `[-0.002361, 0.035797]`, and R2Q3 bilateral endpoint
recall remains inconclusive at `+0.013493` with CI
`[-0.007245, 0.034074]`. In contrast, R4Q5 bilateral endpoint recall
worsens by `-0.045869` with CI `[-0.053979, -0.037741]`, and mean maximum
hole worsens by `0.501027`. The wider-support deficit and hole penalty both
grow monotonically from epochs 25 through 35.

This curve supports a sharper model diagnosis: training increasingly enriches
action evidence and moves some samples closer to one nearby boundary, but it
does not learn paired start/end preservation and progressively sacrifices
wider bilateral support. That mechanism is compatible with weak high-tIoU
localization, although only a controlled objective/geometry ablation can make
the causal claim. At `19:47 +08:00`, Job `1194471` remained `RUNNING` after
the epoch-35 quality summary, with empty formal stderr, no hard error and no
best-checkpoint pointer.

## Post-terminal Stage-2 quality epochs 40 and 45 at 2026-07-27 20:21 +08:00

The one-based Stage-2 epoch-40 and epoch-45 selection-quality summaries
completed with SHA-256
`a9d361a4f0c7d9f56095bfc19210ce979e3652db9262a23c2a3ebd26619dfa50`
and
`224f87458dceab37276142d08c3f92dae36674bb2f466f7e1d0810af86c17994`.
Each covers 487 windows, 211 videos and 355,592 frame observations.

Coarse macro AUPRC/AUROC is `0.443028/0.605677` at epoch 40 and
`0.443816/0.606281` at epoch 45. Pooled Brier improves slightly from
`0.215309` to `0.215244`, while ECE worsens from `0.048988` to
`0.049396`. Learned-minus-matched-uniform action enrichment is
`+0.025338/+0.025265`.

The geometry diagnosis is stable. Radius-1 boundary recall improves by
`+0.039993` (95% CI `[0.022085, 0.060272]`) at epoch 40 and
`+0.034523` (`[0.016535, 0.051169]`) at epoch 45. Mean endpoint distance
improves by `0.036928` (`[0.008255, 0.066371]`) and `0.054350`
(`[0.022962, 0.084692]`). Exact-radius recall and R2Q3 bilateral endpoint
recall remain inconclusive. R4Q5 bilateral endpoint recall remains
significantly worse by `-0.046030` (`[-0.053938, -0.037155]`) and
`-0.046575` (`[-0.055881, -0.038003]`), while maximum hole worsens by
`0.542094/0.570842`.

Thus later training does not repair the core paired-boundary deficit. It
preserves a small action-enrichment and narrow-radius benefit while widening
holes and sacrificing broad start/end support, which remains a plausible
explanation for weak high-tIoU localization rather than a causal plugin
claim. At `20:21 +08:00`, Job `1194471` remained `RUNNING` with null
dependency at exact clean commit
`ed0d4900bffe3546997ea1f00ae806d82cad55f2`; formal stderr was empty,
there was no hard error or best-checkpoint pointer, and epoch-50 quality
export was in progress.

## Post-terminal Stage-2 quality epoch 50 at 2026-07-27 20:24 +08:00

The epoch-50 summary completed with SHA-256
`cdb63e8babcd239967b68dd95818c2c4e3fc0d4d865d340c0bc25afcfcc1c2a4`,
again covering 487 windows, 211 videos and 355,592 frame observations.
Coarse macro AUPRC/AUROC is `0.444491/0.606776`; pooled AUPRC/AUROC is
`0.401329/0.604658`, and pooled Brier/ECE is `0.215020/0.047492`.
Learned-minus-uniform action enrichment is `+0.024000`.

Radius-1 boundary recall remains significantly positive at `+0.040996`
(95% CI `[0.020502, 0.058475]`), and endpoint distance improves
`0.049390` (`[0.019040, 0.077725]`). Exact-radius recall
`+0.009768` (`[-0.011669, 0.027545]`) and R2Q3 bilateral recall
`+0.009305` (`[-0.010956, 0.026838]`) remain inconclusive. R4Q5
bilateral recall remains significantly worse at `-0.043515`
(`[-0.052543, -0.036561]`), and maximum hole worsens `0.579055`.
Epoch 50 therefore confirms rather than resolves the same late-training
failure shape.

## Post-terminal Stage-2 quality epochs 55/60 and final Slurm closure at 2026-07-27 20:54 +08:00

The epoch-55 and epoch-60 selection-quality summaries completed with
SHA-256
`f74cb521dd3c16af3bb6fc42a476f9ead8e69deee9fcac31999149ce5877e19f`
and
`fb53b13243235be945e30fd9b2b9bede7cfb2f3558b80c763d83f84c574fc2e3`.
Each covers the same 487 windows, 211 videos and 355,592 frame observations.

At epoch 55, coarse macro AUPRC/AUROC is `0.444667/0.607080`, pooled
Brier/ECE is `0.214853/0.046050`, and learned-minus-matched-uniform action
enrichment is `+0.025671`. Radius-1 boundary recall improves by `+0.034622`
(95% CI `[0.013418, 0.055163]`) and endpoint distance improves by
`0.035913` (`[0.006682, 0.066797]`). Exact-radius recall and R2Q3 bilateral
support remain inconclusive. R4Q5 bilateral support remains significantly
worse by `-0.042437` (`[-0.048871, -0.034500]`), while maximum hole worsens
by `0.570842`.

At terminal one-based epoch 60, coarse macro AUPRC/AUROC reaches
`0.445240/0.607663`, pooled Brier/ECE is `0.214704/0.044528`, and
learned-minus-matched-uniform action enrichment is `+0.024107`.
Radius-1 boundary recall remains significantly positive at `+0.036524`
(`[0.018434, 0.054814]`) and endpoint distance improves `0.040021`
(`[0.009562, 0.067923]`). Exact-radius recall and R2Q3 bilateral support
still have intervals crossing zero. R4Q5 bilateral support remains
significantly worse by `-0.041925` (`[-0.049793, -0.033066]`), and maximum
hole reaches its worst observed late value, `+0.587269`.

The complete post-terminal curve therefore closes with the same mechanism
diagnosis: small action enrichment and narrow local boundary proximity are
stable, but broad paired-boundary protection is consistently sacrificed and
holes continue to grow. This supports the need for a bounded
localization-preserving transport model; it does not establish a
learned-selector gain over matched uniform.

Slurm Job `1194471` completed at `20:47:16 +08:00` with state
`COMPLETED`, exit `0:0`, elapsed `08:28:13` and null queue presence. The
exact commit remains
`ed0d4900bffe3546997ea1f00ae806d82cad55f2`. The terminal model/evaluation
evidence is unchanged: official Avg-mAP `57.967272%`, checkpoint SHA-256
`4a5389506263b8fd76ca3de6ce3475dee64cc0d9ed1ca73c896692c8db288455`,
receipt SHA-256
`febc59d463476bcf6a1a0d77f237a54f12c59ce3028ce623fba9844c07fada04`,
prediction SHA-256
`719ba43b0f76f5647b2394a23b622aff0bac0c17a54d15613c4d9dbdb57d02d0`,
and combined update-audit SHA-256
`f13ef3f8650c4fe75f795ddf255b32a377ea4b7c31d4bdc007f0121114ba97a1`.
The combined audit still records exactly 6,000 successful optimizer,
scheduler and EMA updates, nine bounded AMP replay restorations, zero
non-finite-loss attempts and zero replay exhaustion. A fresh scan found no
Traceback, OOM, Decord failure or hard `FAIL`; the only matched `fail_*`
tokens are configuration field names. No `best_validation_ema.json` exists.
The documented non-bit-exact global-RNG recovery limitation remains.
