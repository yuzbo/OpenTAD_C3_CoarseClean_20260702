# DUCA e49ef696 R0--R5 Pro audit absorption

## Source identity

- Audited repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- Audited model commit: `e49ef69605e1f98a7217957483f93a8a64bfc348`
- Current execution commit: `8d85929ea04dc40f1eb0c3cc806061ce3b071d3f`
- Identity relation: `8d85929` changes metric/evidence/runtime gates only. It does not
  change the DUCA selector, decoder, detector, losses or schedule relative to `e49ef696`.
- Raw archive:
  `docs/methods/reviews/2026-07-22-e49ef696-duca-r0-r5-pro-audit-raw.txt`
- Raw/archive SHA-256:
  `1D0F9909D2C3DF3966DED0B9F71BFA0A73F9CA2B8D7C68DF15F64265EC8AD636`
- Raw/archive size: `53,578` bytes, `1,404` lines.

## Project verdict

```text
SUBSTANTIAL_ACCEPT_CODE_DIAGNOSIS
PARTIAL_ACCEPT_RECOMMENDED_PLAN
REJECT_STALE_SERIAL_DAG_AND_UNCONDITIONAL_MATRIX_EXPANSION
```

The review is strong and mostly correct about the implemented model. It is not
accepted wholesale because its execution recommendation predates the official-mAP
protocol correction and the independent four-arm replacement queue.

## Line-by-line verification outcome

| Finding | Verdict | Verified implementation fact | Consequence |
|---|---|---|---|
| H1: R2Q3/R4Q5 are not hard bilateral/quota guarantees | Confirmed | `transition_only.py` builds a soft burst utility over the whole radius; `structured_selection.py` hard-decodes only exact-K and max-hole. No mandatory left/right mask is passed to Viterbi. | Paper language must say soft bilateral burst objective, not hard guarantee. A hard decoder is a conditional successor, not current functionality. |
| H2: dense low-resolution decode/H2D occurs before selection | Confirmed | The data pipeline decodes and transforms the complete 160x160 sequence; the full-stack profiler moves the complete batch to GPU before model selection. | Current evidence supports reducing heavy VideoMAE processed frames after decode, not K-only video I/O or end-to-end decode savings. |
| H3: R5 aggregation does not rerun official evaluation from raw predictions | Confirmed | `aggregate_duca_r5_paper_matrix.py` validates terminal JSON and hashes but does not reopen predictions and recompute OpenTAD mAP. The older R3 aggregator does. | Future paper aggregate must independently reopen raw predictions and rerun the evaluator. No retraining is required. |
| H4: candidate/dense cost pairing is not fail-closed | Confirmed | Cost profiles record hardware/session fingerprints, but aggregation divides p50 values without enforcing matching fingerprints and paired-session identity. | Cost ratios are diagnostic until pair identity is enforced. |
| H5: dense baseline commit is not pinned to the historical identity | Confirmed | Launcher/validator accepts a syntactically valid 40-hex SHA and consistency with the supplied SHA; it does not require the sealed historical dense source. | Bind a sealed dense receipt containing commit, checkpoint, config and source hashes; an expected SHA should also be checked. |
| M1: transition discovery uses GT-derived supervision | Confirmed | P0 supervision derives action/transition/boundary validity from training GT. | Describe the method as GT-supervised transition discovery from indirect low-cost state evidence, never boundary-unsupervised. |
| M2: P0 transition losses do not adapt coarse hidden features | Confirmed | The base config sets `auxiliary_hidden_gradient_scale=0.0`. | This is a deliberate gradient-ownership choice, not end-to-end coarse adaptation; diagnose before changing it. |
| M3: true-time mapping repairs coordinates, not internal detector geometry | Confirmed | Inputs are processed on selected-rank positions; true-time metadata maps GT and outputs around the detector. | Do not claim an irregular-time-aware detector. A true-time adapter is a separate hypothesis. |
| M4: G2 changes training distribution | Confirmed | The companion/uniform intervention changes which observations the detector sees. | Report G2 as a training-distribution intervention, not merely gradient normalization. |
| M5: max-hole allows transfer but constrains it | Confirmed | Global DP permits nonuniform allocation under exact-K while bounding the largest uncovered run. | It is neither local-cell sampling nor unconstrained Oracle concentration. |
| M6: P0 family/epoch selection has multiplicity | Confirmed | Candidate/family selection occurs before the formal terminal run. | Keep selection training-only, sealed and disclose multiplicity. Do not use validation mAP to select P0. |
| M7: gradient clipping order is correct | Confirmed | No correction needed in the reviewed path. | Retain runtime audit of resolved values. |
| M8: paper cost table is incomplete | Confirmed | Lazy high-resolution materialization and fully paired decode/H2D accounting are not implemented. | Limit the current cost claim and add a deployment cost path only if the paper needs total-system savings. |
| M9: bundled jobs can remain scientifically independent | Qualified agreement | Batching is acceptable only when every cell has an independent config, checkpoint, terminal evaluation and receipt. | Job count is not experiment count; terminal artifacts decide admissibility. |

Relevant focused tests on the current exact execution snapshot passed remotely:
`96 passed, 1 warning in 83.63s`. These tests establish that existing contracts
are stable. They do not refute H1/H3/H4/H5 because those are missing contracts,
not regressions covered by the current tests. The local Windows attempt remains
blocked by the known Torch `c10.dll` loader issue.

## Recommendations accepted

1. Correct paper wording for soft burst geometry, GT supervision, selected-axis
   detector processing and post-decode heavy-backbone savings.
2. Repair H3/H4/H5 on an evidence-only path before a future R5 paper aggregate.
3. Use fail-closed identity, prediction and cost-pair checks.
4. Select at most one structural model change after official terminal mAP and a
   decisive diagnostic identify the bottleneck.
5. Treat multi-seed/K256/second-backend runs as conditional expansion rather than
   evidence already earned by a gate.

## Recommendations not accepted unchanged

1. **Do not continue the old serial e49 DAG.** Its R0 absolute mAP is contaminated
   training-internal evidence and its bootstrap blocked the mAP critical path. It
   was correctly replaced by independent official-validation jobs.
2. **Do not let R0 choose the only learned family.** R0 may remain a mechanism
   diagnostic, but it cannot authorize a paper family from detector-seen 40-video
   replay. The current four-arm suite directly measures all families.
3. **Do not immediately force a mandatory bilateral decoder.** The review correctly
   identifies the missing hard guarantee, but a hard constraint around a wrong
   predicted center can waste budget. First measure center error, left/right quota
   satisfaction and soft-to-hard mismatch on current R2Q3/R4Q5 outputs.
4. **Do not launch the complete 24-cell R5 matrix before the K384 seed-0 result.**
   First answer whether any learned arm beats matched exact-uniform on official mAP.
5. **Do not implement all five proposed model options.** Hard bilateral decode,
   legal hard-swap utility, protected ASFormer adaptation, true-time geometry and
   adaptive burst/context split are competing hypotheses. Choose exactly one after
   the current result localizes the failure.
6. Three seeds should be reported as paired deltas and mean plus standard deviation;
   `n=3` is too small for a strong distributional significance claim.

## Current evidence and bounded action order

Current formal root:
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_independent_8d85929_formal_20260722_1820`.

Independent jobs are exact-uniform `1180111`, Gaussian G0 `1180112`, R2Q3 G0
`1180113` and R4Q5 G0 `1180114`. All have `Dependency=(null)`. At absorption
time all four are running; uniform passed the real AMP/DDP/full-model gate and
entered official-60, while the learned arms are in finite P0. No terminal
epoch-59 EMA mAP exists, so efficacy remains `experiment_running` and unproven.

Execution order:

1. Finish the current four official-validation arms and compare terminal EMA mAP.
2. In parallel, implement evidence-only H3/H4/H5 corrections without retraining.
3. Use current artifacts to diagnose H1/M2/M3: center error, bilateral counts,
   quota satisfaction, hard-selection mismatch and gradient ownership.
4. If learned selection is not better, choose exactly one bounded model correction
   from the diagnosed cause and rerun U versus the corrected learned arm.
5. Expand to seeds, K256, TemporalMaxer and full paired cost only after the corrected
   learned arm has a credible official-mAP signal.

No model code was changed as part of this absorption.
