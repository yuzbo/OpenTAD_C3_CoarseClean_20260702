# Post-v1 Ideas: H-RIME and CBCG-RIME

## Priority candidate: H-RIME whole-video budget allocation

### State

`user_approved`, `designed`, `implementation_started`,
`deterministic_core_implemented`, `deterministic_core_locally_tested`,
`not_yet_end_to_end_implemented`, `not_empirically_supported`

### User correction and decision

The user correctly distinguished whole-video planning from AdaTAD execution.
Offline access makes a cheap whole-video scan legitimate. AdaTAD may continue to
detect on 768-candidate windows after a separate planner predicts one total
video budget and distributes it across those windows.

The current detector runtime still flattens videos into independent windows,
predicts risk/K per row, and writes per-window ledgers. The repository now
implements the deterministic H-RIME contract/core—stable video grouping,
canonical short-window effective-K aliases, reachable-budget projection,
exact-equality MCKP, replay generation, and homogeneous-K dispatch planning—but
not the learned/shared-scan detector integration. H-RIME is therefore a new
model candidate, not a reinterpretation of the current controller. The
corrected governing specification is
`docs/superpowers/specs/2026-07-28-hrime-v1-budget-conserving-design.md`; the
user has approved it and authorized implementation.

### Implemented deterministic core receipt

`opentad/models/duca/hrime.py` freezes:

- solver `hrime_exact_equality_mckp_v1`;
- `int64`, scale `1,000,000`, `ROUND_HALF_EVEN`, strict registered
  quantization tolerance and deterministic tie-break;
- raw/reachable/realized/projection-unused/solver-unused budget separation;
- `(192,256,384,512) -> (192,224,224,224)` behavior for valid length 231;
- exact equality or fail closed;
- group, scan-contract, plan, solver-input, assignment, replay and dispatch
  hashes;
- `hrime_joint_video_exact_mckp` replay consumption by the existing exact-K
  selector.

Focused pure-CPU tests include brute-force MCKP agreement, unreachable caps,
non-quantized/fractional rejection, deterministic ties/hashes, alias-tamper
rejection, grouping, replay and dispatch restoration. These tests establish
software-contract correctness only; they are not a detector experiment.

### One-sentence method

H-RIME performs a cheap full-video evidence pass, predicts a
length-normalized total heavy-compute budget, jointly allocates quantum-aligned
K values across overlapping 768-candidate windows, and then reuses the exact-K
physical selector and unchanged AdaTAD detector inside each window.

### Three-level contract

1. A **video planner** consumes inference-visible cheap evidence for all windows
   and a pooled video token, then predicts utility/risk over a registered
   total-budget-density panel.
2. A **window allocator** chooses one feasible K for every window under the hard
   total-video budget with an exact discrete solver.
3. The existing **position selector** chooses exact physical positions inside
   each window, after which the unchanged variable-length heavy backbone,
   detector head, physical inverse map, and official NMS run.

Whole-video budget is `B_v=sum_w c_vw(K_vw)`. It is not a common scalar K copied
to every window.

### Mathematical candidate

For video `v`, let windows be `w=1...W_v`, execution quantum `q=16`, candidate
requests be `(192,256,384,512)`, and define feasible effective executions as

`K_vw_feasible = unique({q*floor(min(k,L_vw)/q): k in candidates})`,

discarding zero. The allocator records both the candidate request and effective
execution so short-window aliases cannot be double-counted as distinct choices.

A global head selects a normalized budget density using a frozen,
training-only calibrated price:

`j_v = argmax_j U_v(j) - beta*R_v(j) - lambda*C_v(j)`.

The selected density becomes a duration/window-count-aware `B_v`. Given local
panels `u_vw(k)` and `r_vw(k)`, the first allocator solves

`max sum_w [u_vw(K_vw) - beta*r_vw(K_vw)]`

subject to exact equality with the largest reachable effective-K total no
greater than the raw cap. Raw, reachable and realized budgets remain distinct in
the ledger. With four nominal K candidates this is an exact multiple-choice
knapsack over deduplicated effective options, not a learned heuristic.
Adjacent-window interaction or a shared heavy-feature cache is a separately
falsified extension.

### Training and inference contract

- Build train-only, video-grouped, cross-fitted per-window counterfactual
  utility/risk panels.
- Construct a surrogate video Pareto frontier by solving the exact allocator at
  registered budgets, and train the global/local panels on held-out-video folds.
- Additive detector loss is a training surrogate, not an official video-level
  mAP oracle claim.
- At inference, enumerate one video's windows, run the cheap scan, predict
  `B_v`, solve all `K_vw`, dispatch homogeneous effective-K buckets, execute
  AdaTAD per window, and emit both a video receipt and exact window ledgers.
- Validation/test GT, teachers, counterfactual ledgers, raw prediction caches,
  and batch-composition signals remain forbidden in decisions.

### Cost truth and causal ladder

The default sliding dataset overlaps windows by 25%. Without a real
cross-window heavy-feature cache, the formal execution cost is
`E_v=sum_w K_vw`; unique selected physical frames `U_v` and `E_v/U_v` are only
diagnostics.

The minimum comparisons are dense, fixed uniform K, window-local RIME,
whole-video uniform allocation, exact `U-same-total`, exact per-window K replay,
H-RIME without the video token, shuffled window panels, and the
AdapTok-inspired TAD allocation baseline.

Before learned implementation, a held-out same-total-heavy-cost oracle must show
material cross-window redistribution headroom over uniform and independent
window policies on high-IoU, short-action, or pair-support behavior. Stop if
that oracle fails, if gains come from higher realized cost/padding/leakage, or
if full-stack latency does not fall.

## Optional within-window candidate: CBCG-RIME

### State

`discussed`, `not_yet_designed`, `not_implemented`, `not_tested`

This is an optional within-window extension of H-RIME. It is not authorized to
change the frozen DUCA-RIME four-stage transaction or precede the H-RIME
same-total-cost oracle.

## 2026-07-28 external-review decision

Source `U-PRO-CBCG-1` was fully absorbed and independently reviewed. The
decision is **conditional acceptance, not verbatim approval**:

- accept the narrowing from a generic pair-risk graph to calibrated
  **boundary-coverage failure** on consecutive physical selections;
- accept a matched-K oracle before any learned edge head, while retaining the
  existing window-level budget-sufficiency risk only for K selection;
- accept independent per-K decoding as the default, explicit source/sink
  coverage, one energy for hard Viterbi and soft forward-backward, and a sparse
  physical graph with measured decoder overhead;
- accept that Patch B and production integration remain on hold until the
  oracle, learnability, calibration, causal, and cost gates pass;
- use `AdapTok-inspired TAD budget allocation baseline`, not “official AdapTok
  reproduction”, for the current direct-transfer arm.

The following parts of the external report are not accepted as written:

1. Patch A is not an immediate production action. The current v1 transaction
   failed its Phase-1 no-padding ledger, and its downstream phases are blocked.
   The execution contract and a genuinely held-out development protocol must
   be repaired before a new oracle transaction is authorized.
2. The report's linked sandbox patches and reported synthetic tests are
   unavailable here. They remain `PARTNER_CLAIM`, not implementation or test
   evidence.
3. Regressing a path-level positive detector regret onto edge-incidence vectors
   is generally underidentified: consecutive-edge incidences, total gap, and
   source/sink terms are strongly collinear. A valid oracle must use balanced
   edge perturbations or a regularized inverse problem with rank, condition
   number, bootstrap stability, and held-out-video tests.
4. “Gap-mass normalization” is incomplete until source and sink gaps are
   explicitly defined as path-dependent physical intervals and all nonnegative
   masses are normalized to sum to one. Weighted mean, max, and CVaR aggregation
   must be oracle-compared rather than chosen post hoc.
5. `O(KTW)` is only a valid bound when predecessor and successor spans are
   fail-closed at a registered `W`; the current physical-gap graph can otherwise
   degenerate toward `O(KT^2)`. Actual edge count and span distributions must be
   reported.
6. Gap length is both a proposed feature and a direct gap penalty, so a
   gap-only baseline and residualized content-risk ablation are mandatory.
   Source/internal/sink potentials must be identical between hard and soft
   decoders, and `risk-off` must be bit-exact with v1.

## One-sentence method

CBCG-RIME, as a working refinement, makes calibrated boundary-coverage failure
an edge cost inside the physical exact-K position decoder. The existing
window-level risk remains a budget/K signal; the new path risk may change
positions only if it survives a same-K causal test. No whole-video risk
aggregation is implemented.

## Algorithm candidate

For candidate time points `i<j`, predict node utility `u_j` and consecutive
boundary-coverage risk `r_ij` from cheap inference-visible evidence. For each
registered budget K, solve the physical-time DAG recurrence

`DP[k,j] = u_j + max_i(DP[k-1,i] - lambda_r * r_ij - lambda_g * gap_ij)`

over allowed monotone edges, exact cardinality, required anchors, and the frozen
physical gap cap. A forward-backward relaxation supplies training gradients.
The first admissible oracle keeps K fixed per video and changes positions only.
Only after that gate passes may a separately calibrated path-risk summary be
considered alongside the existing K-level controller under the same frozen-price
and no-padding cost contract as v1.

## Publishable distinction

The candidate contribution is not adaptive token count or first use of exact-K
graph decoding, both of which would overclaim. It is a falsifiable attempt to
place calibrated TAD boundary-coverage risk on consecutive physical-selection
edges, with realized-K-matched controls and high-IoU/short-action falsifiers.
The `AdapTok-inspired TAD budget allocation baseline` remains mandatory.

## Minimal falsification ladder

1. **Execution and split closure:** first repair and revalidate real effective-K
   execution/no-padding, then establish checkpoint-specific training exclusion
   and video-grouped cross-fitting. The failed 20-video training-domain control
   is not an oracle population.
2. **Oracle headroom:** using privileged counterfactual targets only for
   held-out development analysis, fix checkpoint, backend, per-window realized
   K, NMS, subset, and seed, and vary positions only. Pair by
   `(video_id, window_start_frame)` and bootstrap by video. Report both signed
   and positive regret, edge-design rank/conditioning, bootstrap stability,
   `mAP@0.7`, short-action mAP, boundary coverage, and pair support.
3. **Target learnability:** cross-fit the pair-risk target and require useful
   ranking plus calibrated low-risk coverage using inference-visible evidence.
4. **Decoder causality:** compare full graph, risk-off graph, gap-only,
   residualized-risk, feasibility-preserving shuffled edges, v1 `weak_overlap`,
   `D-no-risk`, and exact `U-same-K`.
5. **Full-stack check:** include graph-decoder overhead in latency, energy,
   throughput, and memory.

## Stop rules

- Stop before a learned edge implementation if the complete held-out same-K
  oracle misses its pre-registered primary materiality gate over the v1 decoder
  at matched K. The margin must be frozen from training/calibration roles and
  measurement noise before reading the complete development result; the
  external report's `0.5` point value is a proposal, not a repository fact.
- Stop if cross-fitted pair risk is uncalibrated or cannot cover a non-vacuous
  low-risk subset.
- Stop if gains disappear under edge shuffle, fail high-IoU/short/pair-support
  gates, or cost matching to `U-same-K` fails.
- Stop if the edge inverse problem is rank-deficient/unstable or if a gap-only
  explanation accounts for the apparent benefit.
- Do not open official-final data or promote this candidate while v1 lacks a
  passing physical/no-padding receipt and a complete development adjudication.

## Deferred alternatives

- **Conformal-risk RIME:** use a train-only upper risk bound for conservative K
  selection if v1 fails calibration rather than position coverage.
- **Sequential RIME:** acquire uniform anchors then a second information-gain
  round only if one-shot acquisition leaves clear oracle headroom; this has the
  highest latency, cache, and AdapTok-overlap risk.
