# DUCA Dynamic-K RIME Four-Stage Adjudication

## Status

- Approval date: `2026-07-27`
- Deployment date: `2026-07-28`
- Approval: `user_approved`
- Design: `designed`
- Implementation: `implemented`
- Focused local verification: `tested`
- Remote authoritative verification: `passed`
- Latest transaction: `failed_closed_without_terminal_receipts`
- Dense reference training: `training_loop_finished_but_jobs_failed_at_checkpoint_compaction`
- Phase 2/3/4: `blocked_by_failed_dependency`
- Empirical support: `not_yet_empirically_supported`
- Paper status: `not_yet_paper_ready`
- Stage-0 recovery implementation: `implemented / local_non_torch_tested`
- Stage-0 recovery deployment: `not_yet_submitted`

## Objective

Test whether cheap coarse evidence can allocate real pre-backbone temporal
compute and protect high-IoU localization through risk-aware, exact-K,
monotone physical-time acquisition.

## Frozen experimental DAG

```text
remote code gate
    -> Phase 1 execution/geometry/cost closure
    -> Phase 2 U-mixed-K + O1/O2/O3/O4 + K384/K192 protocols
    -> Phase 3 six train arms + U-same-K + development seal
    -> Phase 4 authorization
    -> 2 detectors x 2 budget panels x 3 fresh seeds
    -> formal matrix seal
```

Dense ActionFormer and TriDet references are trained in parallel after the code
gate and become prerequisites for Phase 3/4 cost evidence.

## Algorithms and trainable outputs

| Phase | Output | Model status |
|---|---|---|
| 1 | exact-K physical decoder, coordinate inverse, controls, ledger/cost profiler | algorithm/infrastructure |
| 2 | `U-mixed-K`, cross-fitted targets, counterfactual gates, frozen price protocols | trainable baseline + protocol |
| 3 | `RIME-full` and causal ablation family | first new model candidate |
| 4 | fresh-seed ActionFormer/TriDet replicas and formal evidence matrix | validation of the same candidate |

## Formal arms

Phase 3 train arms:

- `RIME-full`;
- `U-fixed`;
- `F-bound`;
- `D-no-risk`;
- `AdapTok-TAD`;
- `D-shuffle`.

`U-same-K` is evaluation-only and reuses the RIME checkpoint with an immutable
exact-K replay. It is not a separately trained model.

Phase 4 trains `RIME-full` and `U-fixed` for each formal cell. `U-same-K`
remains evaluation-only from the corresponding RIME source.

## Corrected panel semantics

| Panel | Allocation | Allowed claim |
|---|---|---|
| K384 | frozen-price dynamic K | dynamic allocation and learned positions |
| K192 | forced exact K=192 | learned positions at the floor budget only |

## Corrected cost semantics

Candidate and `U-same-K` must consume the same realized per-window K map keyed by
`(video_id, window_start_frame)` within tolerance. Per-window costs are then
aggregated per video. `U-fixed` is retained for accuracy, not variable-K cost
matching. Candidate, matched control, and dense reference all use measured
full-stack latency, throughput, and peak memory. Energy is reported only when a
trusted, registered power source is available; it is unavailable for the latest
failed transaction and cannot support a claim.

## Provenance requirements

- exact clean Git commit;
- positive Slurm job IDs and explicit dependencies;
- immutable external run roots;
- split manifest and assignment hashes;
- checkpoint SHA-256 plus trained-commit evidence;
- Phase-2 receipt and both formal protocol hashes;
- Phase-4 authorization bound to the same Phase-2 receipt and protocol;
- checkpoint audit and terminal identity;
- official-final video set opened only after Phase-3 GO.

## Pre-registered protocol inputs

These values are frozen in the submitting Git commit rather than selected from
Phase-2 or later outcomes:

| Input | Frozen value | Interpretation |
|---|---:|---|
| candidate K / cost proxy | `192,256,384,512` | exact heavy RGB frames; no Kmax padding |
| primary / floor target | `384 / 192` | K384 dynamic panel; K192 learned-position-only panel |
| decoder family | `weak_overlap` | must pass the independent-oracle regret gate |
| risk weight / threshold | `1.0 / 0.35` | finite-budget utility and feasibility rule |
| O4 max Brier / ECE | `0.25 / 0.10` | pre-test calibration ceilings |
| O4 min low-risk coverage | `0.25` | rejects a vacuous always-fallback risk head |
| O4 max low-risk failure | `0.35` | observed failure may not exceed the accepted-risk ceiling |
| short / medium duration | `2.0 s / 8.0 s` | inherited from the registered DUCA physical P3 strata |

## Deployment ledger

Aborted immutable submission (retained as negative deployment evidence):

- implementation commit:
  `f510741b32075c5c4e729d4207a549886a6dd064`;
- external run root:
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_four_phase_f510741b_20260728_094811`;
- physical protocol manifest:
  `/data/run01/sczc063/yuzibo/rime_prerequisites/duca_protected_physical_protocol_f510741b.json`,
  SHA-256 `5d28d1d37e698b5f17156245f55da62a82dc5b537c32fe70104f1be231e605d8`;
- submission manifest:
  `submission_manifest.json`, SHA-256
  `c74c351bad04dd7bfc6701ca5205e419116694d21736419776cec1f3cdb7ada6`;
- code-gate job `1197889`: `COMPLETED`, receipt passed on the exact commit;
- Phase-1 job `1197890`: `CANCELLED` after the transaction became
  uncompletable;
- dense ActionFormer job `1197891`: `FAILED` before optimizer update 1 because
  its sealed-validation switch was missing;
- dense TriDet job `1197892`: `FAILED` before optimizer update 1 for the same
  reason;
- Phase-2 job `1197893`: `CANCELLED` before execution;
- Phase-3 controller job `1197894`: `CANCELLED` after its dense dependency
  failed closed;
- no Phase-3 child or Phase-4 formal job was submitted.

The code gate recorded all 158 focused tests passing and all 24 registered
configs satisfying their stage-specific contracts.

The next transaction must use the commit that adds
`seal_eval_dataloaders_during_training=True` to both dense configs and enforces
it in precheck; no output from this aborted root may be reused as positive
evidence.

Second aborted immutable submission (also retained; no positive evidence may be
reused):

- implementation commit:
  `1ff54baf782194faf1403296186a465031f26dd9`;
- external run root:
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_four_phase_1ff54baf_20260728_095843`;
- physical protocol SHA-256:
  `ce5debb5e436e0e6b6919747680553a5ad947c1938c540f80217165288d6c65a`;
- submission-manifest SHA-256:
  `114821c948345de3beb98455c892921b757f84f68bf80b2a745fd6d0a5bce77b`;
- code gate `1197974`: `COMPLETED`;
- Phase 1 `1197975`: `CANCELLED`;
- dense ActionFormer `1197976`: `CANCELLED` after demonstrating stable
  optimization through update 50;
- dense TriDet `1197977`: `FAILED` on backward 1 from reentrant VideoMAE
  checkpointing plus DDP;
- Phase 2 `1197979` and controller `1197980`: `CANCELLED` before execution.

The next commit must additionally freeze
`model.backbone.backbone.with_cp=False` for dense TriDet and reject drift in
both the launcher precheck and code-gate config matrix.

Latest immutable submission (terminally failed closed):

- implementation commit:
  `d9d454cd49a3e7a87694fc948601d00ff4043cb0`;
- external run root:
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_four_phase_d9d454cd_20260728_101256`;
- physical protocol SHA-256:
  `c4dfc31a64b56a93366c43443883df535e572eed38df63878fe11d3e00193a70`;
- submission-manifest SHA-256:
  `ed374ae81991ca8241c0b01ab6588f13ea292b967b18a58115ec3f735440b038`;
- code gate `1198113`: `COMPLETED`, 158 tests and 24 configs passed;
- Phase 1 `1198114`: `FAILED` after 12m44s on the exact-K/no-padding ledger
  gate. Uniform-K384 line 64 (`video_validation_0000686`, window 0) had
  `dense_valid_len=231`, `effective_k=unique_k=231`, and
  `backbone_input_k=padded_k=384`;
- dense ActionFormer `1198115`: `FAILED` after its 60-epoch loop printed
  `Training Over`; raw `epoch_59.pth` exists, but checkpoint compaction stopped
  at `from tools.bata import duca_p0_training` with
  `ModuleNotFoundError: No module named 'tools'`;
- dense TriDet `1198116`: `FAILED` at the same compaction import after its
  60-epoch training loop; raw `epoch_59.pth` exists;
- neither dense arm emitted `terminal_ema.pth`, `training_receipt.json`,
  evaluation evidence, or checkpoint binding;
- Phase 2 `1198117`: `DependencyNeverSatisfied`;
- Phase-3/4 controller `1198118`: dependency-pending and unauthorized.

At the `2026-07-28 14:47 CST` verification, the expected Phase-1, Phase-2,
Phase-3, and Phase-4 terminal receipts were all absent. The repository contains
no registered job/artifact literally named `4B`; the user reference is mapped
to this four-stage transaction by context. This transaction is not a completed
experiment and produced no admissible model comparison.

Verified raw-checkpoint recovery inputs:

| Backend | Source path | Size | SHA-256 |
|---|---|---:|---|
| ActionFormer | `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_four_phase_d9d454cd_20260728_101256/dense_actionformer/train/gpu1_id0/checkpoint/epoch_59.pth` | `623799387` | `cd92f3d499360c834f7ddd6ccfd5cba172c870bf6922de566b2b7e3878680e11` |
| TriDet | `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_four_phase_d9d454cd_20260728_101256/dense_tridet/train/gpu1_id0/checkpoint/epoch_59.pth` | `411540059` | `8940dbe756e8abfa3f7c8b042f3c658b26898d5c805d2876011a4e7510d11e12` |

Both contain `epoch`, `state_dict`, `optimizer`, `scheduler`, `state_dict_ema`
and `grad_scaler`, with `epoch=59`. They do not embed a complete
commit/variant/seed audit. Any recovery must use a new immutable, hash-bound
salvage transaction and explicitly label external provenance; the failed root
is never modified or reclassified as successful.

Stage-0 recovery code now exists locally:

- short-window heavy execution is quantum-aligned and uses its true effective K
  without replicated inactive tail;
- ledger equality is
  `effective_k=unique_k=backbone_input_k=padded_k`;
- compaction uses clean-cwd module invocation;
- salvage is a new manifest/hash-bound transaction with exact source path,
  size, hash, job, epoch, EMA schema and explicit external-provenance receipt;
- the old failed root remains immutable and its source jobs remain `FAILED`;
- the recovery DAG supports explicit `fresh_train` or `salvage` dense mode and
  forces official-final Phase 4 sealed.

Local `py_compile`, `bash -n`, and the expanded 64-test focused non-Torch
suite passed. The Windows
host cannot load the repository's CUDA-linked PyTorch DLL, so Torch-dependent
selector/detector tests remain `remote_pending`; no authoritative Stage-0
receipt exists until the exact clean commit passes Slurm precheck/code gate.

The mandatory pre-deployment MAX audit initially returned `NO-GO` because the
Stage-0 fixed-window ledger still emitted a null physical-protocol hash and did
not explicitly separate requested/raw, short-window reachable, realized,
projection-unused and solver-unused budgets. The ledger and its finalizer now
require the exact protocol SHA-256, seal all five budget fields, scope them as
`stage0_engineering_window_execution`, aggregate their totals and fail closed on
any inconsistency. A new clean-commit audit is still required before Slurm
release.

Raw Phase-1 development measurements produced before the fail-closed ledger
gate were:

| Control | Avg mAP | mAP@0.7 | Receipt |
|---|---:|---:|---|
| local dense A | 86.9867 | 74.9371 | passed |
| local dense B | 86.9867 | 74.9371 | passed |
| released dense | 92.8823 | 84.3745 | passed |
| uniform K384 | 90.6759 | 80.3625 | missing |

These are 20-video development controls with different checkpoint identities;
they are not a method comparison. Uniform K384 is inadmissible as a completed
Phase-1 artifact because its no-padding receipt did not pass.

The 20 videos are selected from the 200-video THUMOS `training` subset by a
180-video block list. The historical dense/uniform checkpoints follow configs
whose training dataset is the same `training` subset; an exact per-checkpoint
exclusion manifest was not found. The terminal values are therefore treated as
high-confidence in-sample sanity measurements, not held-out accuracy.
Independent recomputation with the pooled repository mAP evaluator reproduced
all terminal values exactly. The high numbers are not caused by self-score
normalization, `top_k=None`, or averaging the secondary per-video diagnostic
mAP; the primary problem is training exposure and the narrow 20-video scope.

Pre-deployment dense launcher precheck `1198049` completed. A dedicated TriDet
smoke job `1198059` was deliberately canceled after stable update 50; its
partial output is diagnostic only and is not admissible positive evidence.

## Stop rules

- Any failed hash, split, coordinate, exact-K, no-padding, or cost-match check
  stops the dependent stage.
- Phase-3 NO-GO prevents official-final evaluation.
- Phase-4 formal failure forbids the general-plugin paper claim; negative
  evidence is retained rather than silently rerouted.
