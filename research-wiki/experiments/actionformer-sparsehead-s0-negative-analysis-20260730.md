# ActionFormer SparseHead K384 Official S0 Negative Analysis

Date: 2026-07-30

Experiment status: `tested`

Claim status: `empirically_supported` only for the frozen
K384 + `selected_native_grid_queries` intervention

Paper status: `paper_ready=false`

## Outcome

Slurm Job `1205599` (`af-k384-pair-r5`) completed `0:0` on g0030 in
`00:19:21`. It retrained both arms from scratch under candidate commit/tree
`d86a4acda21e35a1609f19f1a46bc470ee18b7e1` /
`327c032a1ab3c14d0e34d6339df36f8a33ec6907`, official THUMOS
`validation`-train / `test`-eval data, seed `1234567891`, the official
5-warmup + 30-optimizer-epoch schedule, no resume, terminal
`epoch_035.pth.tar:state_dict_ema`, the same seven-argument official Soft-NMS
extension, and the pinned official evaluator.

`MATCHED_PAIR_COMPLETE.json` has SHA-256
`545e420aa1d437aedeffd15cb30390ceb0cfe4d6565d7eb35c53a8bf17ac76fd`.
It reports `validation_pass=true`, no issues, and every frozen comparability
flag true. Dense/sparse ARM completion SHA-256 values are
`a15b0526ef9a75a0fe32c0798b609c738781ab5c063c53df165ace6cbcdf138a` /
`fc682cfb01b9ed6639f821938922051edc2afa55490f504170eb7e3a6fd49037`.
Independent metric attestation SHA-256 values are
`59a9d037faf0418e226f184b87c66d484c7a64b81a911692ba65d44c1cc195d7` /
`499b0e7f34b854ca6c9915a95d0c522106dda859079a50a7c4f28bbd6ede65ba`.
Each arm covers exactly 212 official test videos and 42,400 retained
predictions.

| endpoint | dense full-grid | K384 selected-loss | sparse - dense |
|---|---:|---:|---:|
| Avg-mAP | 66.583013 | 43.919699 | -22.663313 pp |
| mAP@0.3 | 81.908495 | 64.925248 | -16.983246 pp |
| mAP@0.4 | 77.952035 | 56.642845 | -21.309190 pp |
| mAP@0.5 | 71.285498 | 45.952641 | -25.332858 pp |
| mAP@0.6 | 58.255505 | 32.783177 | -25.472328 pp |
| mAP@0.7 | 43.513530 | 19.294586 | -24.218944 pp |

The relative Avg-mAP loss is `34.037681%`. The preregistered S0 continuation
bounds were `Delta Avg >= -1.00 pp` and both high-IoU deltas
`>= -1.50 pp`. All three primary bounds fail by very large margins.
Therefore the frozen S0 verdict is:

`KILL_CURRENT_K384_SELECTED_LOSS_INTERVENTION`

The five-seed main study and detector-pipeline cost study are not authorized.
This is a legal model/method negative result, not an engineering failure.

## Official comparability and paper boundary

The causal comparator is the newly trained same-commit dense arm
(`66.583013`), not the released official anchor (`66.833392`). The released
anchor remains contextual validation that the matched dense arm reproduced the
expected regime. Historical PhysTime/OpenTAD/AdaTAD values are different
methods, checkpoints and/or evaluation contracts and are not matched
comparators.

The result is official-comparable for the preregistered single-seed S0
question, but S0 was frozen as `paper_main_table_eligible=false`. It may be
reported as an appendix/negative-ablation result with all receipts. It cannot
support an accuracy-preserving or efficiency claim, cannot be promoted to the
main table, and cannot be generalized to “all SparseHead/PhysTime methods
fail.”

## What the current intervention actually changes

K384 is not execution-only. The selector allocates a fixed total of 384
native-grid queries across FPN levels. The model then:

1. executes the classification/regression heads only for those selected
   physical positions;
2. scatters zero at every unselected output position;
3. uses the selected mask for decoding, so unselected positions cannot emit a
   proposal; and
4. uses the same selected mask as training loss support, so unselected queries
   receive neither classification nor regression gradient and the EMA loss
   normalizer is driven only by selected positives.

Focused tests prove selected-position head outputs are exactly equivalent to
dense head outputs for the same weights and native geometry. That lowers the
probability of a packed-kernel arithmetic defect, but it does not make the
complete intervention equivalent to dense training or dense proposal
coverage.

## Competing explanations

### H1: structural query/proposal coverage bottleneck

Supporting evidence:

- unselected positions are intentionally absent from decoding;
- degradation grows from `-16.98 pp` at IoU 0.3 to roughly `-24--25.5 pp`
  at IoU 0.5--0.7;
- this pattern is consistent with missing action centers/boundaries and a
  harder high-IoU localization ceiling.

Counterevidence:

- the large `-16.98 pp` loss already at IoU 0.3 suggests coverage alone may
  not explain the full collapse.

Falsifiable prediction:

- the dense-trained checkpoint evaluated with K384 queries should lose
  strongly if inference coverage is the dominant factor; the loss should be
  amplified at high IoU.

Minimal decisive diagnostic:

- the preregistered no-retraining 2x2 checkpoint cross-evaluation.

### H2: selected-loss supervision and optimization collapse

Supporting evidence:

- the sparse arm trains classification and regression only on selected
  queries;
- the positive count and EMA loss normalizer therefore differ from official
  dense training;
- all IoU endpoints decline, consistent with a model learned under a much
  smaller and differently distributed supervision support.

Counterevidence:

- the present final metrics alone cannot separate training damage from
  inference query removal.

Falsifiable prediction:

- the selected-loss checkpoint evaluated with dense queries remains far below
  the dense-trained/dense-evaluated control; if it recovers substantially,
  inference coverage rather than optimization dominates.

Minimal decisive diagnostic:

- the same 2x2 cross-evaluation plus selected positive/negative counts,
  regression-positive coverage, EMA normalizer trajectories and gradient-scale
  summaries from the preserved training logs/checkpoints.

### H3: score calibration and Soft-NMS density mismatch

Supporting evidence:

- sparse query density changes the score population consumed by fixed
  threshold/top-k/Soft-NMS settings;
- the low-IoU decline leaves room for ranking/calibration damage in addition
  to boundary coverage.

Counterevidence:

- post-NMS AP falls too broadly and too strongly for calibration to be assumed
  as the sole cause.

Falsifiable prediction:

- pre-NMS class-aware proposal recall stays near dense while post-NMS recall
  and score reliability collapse; a training-internal, frozen calibration
  replay recovers a substantial fraction without changing segments.

Minimal decisive diagnostic:

- first measure retained-output score/overlap/recall behavior; a true NMS
  attribution additionally requires newly captured pre-NMS proposals and
  logits. Post-NMS `eval_results.pkl` alone cannot identify suppressed
  proposals or background-logit calibration.

### H4: implementation or evaluator defect

Supporting evidence:

- the effect size is large enough that a coordinate/mask defect must be
  explicitly considered.

Counterevidence:

- selected-position dense equivalence, native-geometry tests, exact official
  data/config/source hashes, terminal EMA receipts, 212-video coverage and an
  independent pinned-official evaluator all pass;
- both raw prediction files contain the same fixed count and official video
  ID set.

Falsifiable prediction:

- off-diagonal 2x2 rows or synthetic index-to-timestamp audits expose a
  systematic mismatch inconsistent with the intended mask semantics.

Minimal decisive diagnostic:

- state-dict compatibility gate, no-retraining 2x2 cross-eval and a
  per-video selected-index/timestamp contract audit.

## Ranked judgment and next action

Current ranking is:

1. combined structural coverage + selected-loss supervision bottleneck;
2. optimization/normalizer drift within the selected-loss arm;
3. score calibration/Soft-NMS interaction;
4. implementation/evaluator defect.

The immediate authorized work is diagnostic only:

1. freeze an explicit S0 KILL receipt;
2. execute the preregistered 2x2 using the two existing epoch-35 EMA
   checkpoints, with no training and no model selection;
3. compute per-class, duration, boundary, retained-proposal recall/top-k and
   failed-video diagnostics from both raw files;
4. separately audit assignment/support observability and training-log
   normalizer/gradient evidence;
5. decide whether a new, separately preregistered method should decouple
   training supervision from inference execution, add adaptive/boundary-aware
   coverage, raise the budget, or terminate this SparseHead formulation.

No threshold, seed, data, checkpoint, evaluator, K, loss or decoder may be
changed to rescue Job `1205599`.

## Completed no-retraining 2x2 attribution

Slurm Job `1205701` completed `0:0`. Attribution completion SHA-256 is
`d0bffe87cfb582b1b0649da3833e9fe0147db5a0a78500b6700fb78019323afb`.
Both off-diagonal cells reuse the exact two terminal epoch-35 EMA checkpoints,
official data/config/evaluator and add no training, seed or model selection.

| training support | dense evaluation queries | K384 evaluation queries |
|---|---:|---:|
| full native-grid | 66.583013 | 45.784332 |
| selected native-grid | 64.537343 | 43.919699 |

Exact off-diagonal metrics are:

- full-train × K384: Avg `0.4578433218148983`; mAP@0.3--0.7
  `0.6445610125678265/0.5780821439202886/0.48367460937713663/0.35681513496191763/0.22608370824732202`;
- selected-train × dense: Avg `0.6453734282343033`; mAP@0.3--0.7
  `0.8221767537918664/0.7719218183080583/0.6922817647178611/0.5469645815932217/0.39352222276050963`.

The average K384 execution main effect is `-20.7082 pp`; the selected-loss
training main effect is `-1.9552 pp`; the interaction is only `+0.1810 pp`.
Selected-loss is almost neutral at IoU 0.3 but costs `3.56/4.16 pp` at
IoU 0.6/0.7 under dense evaluation. Thus the high-IoU training penalty is
real but secondary. The full-trained checkpoint still collapses under K384,
whereas the selected-trained checkpoint returns to `64.54` under dense
queries. This falsifies the initial “combined damage of similar magnitude”
hypothesis: inference proposal/query removal is dominant and selected-loss
damage is a smaller, nearly additive factor.

The 2x2 remains single-seed descriptive causal attribution. It is
`paper_main_table_eligible=false`, adds zero independent seeds and has no cost
claim.

## Completed retained-output diagnostics

The post-NMS diagnostic artifact SHA-256 is
`a6b7fa0c4a41aac75ae2fb4cb4fcfbe68cf48bc7d2c813b37485b35998838791`.
It covers `3358` official test GT instances.

- official mean Recall@1x/5x at tIoU 0.7 falls from `52.25/71.08%` to
  `30.21/43.25%`;
- oracle class-aware retained-output recall at tIoU
  0.3/0.4/0.5/0.6/0.7 falls from
  `96.55/95.86/93.92/87.61/76.50%` to
  `86.45/81.09/71.86/59.71/42.41%`;
- class-agnostic recall@0.7 is still only `44.55%` versus `80.85%`, so
  classification cannot explain the collapse;
- fixed-topK class-aware recall@0.7 for K=1/10/20/50/100/200 is
  `3.72/26.89/41.48/61.26/70.49/76.50%` dense versus
  `3.19/19.21/26.65/34.90/40.02/42.41%` sparse. The widening gap with K
  rejects a pure score-reranking explanation;
- mean best IoU drops from `0.76734` to `0.60237`;
- score median/p95 drops from `0.05106/0.4661` to `0.03846/0.2420`, so
  calibration is plausibly secondary;
- same-label overlap-pair rate at tIoU >=0.7 drops about `71%`;
- every one of 20 classes is negative, with the largest losses for LongJump,
  PoleVault, HammerThrow and JavelinThrow. Common 4--16 second actions are
  especially damaged.

These are retained post-NMS outputs. They cannot prove pre-NMS proposal recall,
identify suppressed proposals, isolate background-logit calibration or support
an NMS-only causal claim.

## Completed assignment/support audit

Audit implementation commit/tree are
`465b2bc284d5c3b62ec9e21023052b5eabddf260` /
`da1e515398017345deb4c39d98751ade0a8aa8db`. Slurm Job `1205799`
completed `0:0` in `00:00:41`. Suite/producer/rows/sample-seal SHA-256 values
are
`475b61ddad4b0b56a86b2e2616ef2584b252c3169b4ad1268223f21d6e118567` /
`ca7e97a4124e49eb2ac30e949bcd50d4407998e8518eb72c8c6c8c8bb3f86e8b` /
`a73b6f69c8655fed584774d131388ebf4974cf001f3efd9f492a952251e96b7f` /
`d02f1de5fe9320cea47011b4af253001db77ecb7aadff83b8185a3350c7c55f4`.
Validation passes with no issues.

This diagnostic uses exactly 64 deterministic official THUMOS `validation`
training windows, no test GT, loss, backward, optimizer, training or model
selection. It is not an official performance row.

| support statistic | dense | exact K384 |
|---|---:|---:|
| valid/selected queries | 142,623 | 24,350 |
| assigned positives | 2,721 | 461 |
| positive retention | 100% | 16.9423% |
| GT with no candidate | 0 / 804 | 395 / 804 |
| GT with no assignment | 67 / 804 | 427 / 804 |

Per-level positive retention is only
`18.11/18.42/16.67/16.67/15.31/19.72%`, while maximum selected-center gaps
grow `12/24/48/96/192/384` feature-grid units across FPN levels. Common
duration buckets retain only about `15--20%` of dense candidate/assignment
support; the 8--16 second bucket retains `14.81/14.64%`. No invalid,
filtering, rounding or non-finite counter is nonzero.

This explains why more/better training alone cannot repair the method: the
hard K384 path removes nearly half of sampled GT from candidate reachability
and more than half from assignment, while inference permanently forbids
unselected queries from proposing actions.

## Contradiction check

There is no material contradiction among the evidence:

- physical-time decode improvements in earlier PhysTime replays were
  within-method decoder counterfactuals over frozen tensors; they did not
  establish official ActionFormer accuracy or recover proposals deleted before
  decoding;
- the assignment audit shows severe selected support loss, while the 2x2 shows
  only a roughly `2 pp` learned-weight penalty. These are compatible: dense
  evaluation restores proposal opportunities for the sparse-trained weights,
  but cannot make the K384 path emit deleted queries;
- lower score tails coexist with rank-based and class-agnostic recall collapse,
  so calibration can contribute without being the dominant cause;
- the historical `63.xx` random-sampling/PhysTime values use different
  checkpoints or evaluation contracts. The valid causal baseline here is the
  paired `66.583013`; the exact K384 method is far below both regimes.

## Final ranked root-cause judgment

1. **Structural proposal/query/support deletion -- high confidence.**
   Directly supported by the approximately `-20.7 pp` K384 main effect,
   `395/804` GT without candidates, `427/804` without assignments and
   class-agnostic/topK recall collapse.
2. **Selected-loss supervision and high-IoU optimization damage -- moderate
   confidence, secondary.** About `-2.0 pp` Avg and `-3.6/-4.2 pp` at
   IoU 0.6/0.7 under dense evaluation.
3. **Score calibration/Soft-NMS density interaction -- moderate-to-low
   confidence, secondary.** Score tails compress, but segment/topK support is
   already absent before any monotonic recalibration could restore it.
4. **Implementation/evaluator defect -- low confidence.** Source/data/runtime,
   selected-position equivalence, native geometry, official evaluation,
   independent recomputation and all three terminal suites pass.
5. **Seed/data uncertainty -- affects magnitude, not the current decision.**
   A single seed cannot establish a population effect, but it is not a license
   to continue a preregistered intervention that misses all gates by more than
   20 pp.

## Route decision and claim boundary

The exact hard K384 selected-loss formulation is closed and must not be
resurrected. The SparseHead research question is not universally refuted, but
it may continue only through a new design that preserves dense proposal and
supervision support while sparsifying expensive refinement.

That design is frozen as DCSR in
`actionformer-sparsehead-dcsr-official-prereg-20260730.md` with status
`designed`. It is not yet implemented or empirically supported.

The integrity audit is recorded in
`actionformer-sparsehead-s0-integrity-audit-20260730.md`. Its overall status is
`WARN` solely because evidence is one official seed plus diagnostics and the
configured external cross-model file reviewer was unavailable. The manual
source/receipt audit found no fake GT, self-normalization, phantom result or
official-evaluator mismatch.

Paper-safe result: an official-comparable, fully receipted negative appendix
finding for the exact intervention and diagnostic attribution of its dominant
failure. Forbidden result: any positive efficiency/main-table/general
SparseHead claim.

The monitoring heartbeat is now `PAUSED`: every authorized S0 diagnostic is
terminal and no additional hard-K384 submission is allowed.
