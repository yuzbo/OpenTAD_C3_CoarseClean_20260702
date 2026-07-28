# H-RIME v1 Stage-1 Same-Total Development Oracle

## State

- User authorization: `approved`
- Design: `designed`
- Deterministic planner/statistics: `implemented`
- Replay/detector/evaluator integration: `implemented`
- Focused non-Torch verification: `tested`
- Torch/runtime verification: `remote_pending`
- Preregistration artifact: `not_yet_frozen`
- Development oracle matrix: `not_yet_run`
- Learned H-RIME: `not_authorized`
- Empirical support: `not_yet_empirically_supported`
- Paper status: `not_yet_paper_ready`

## Scientific question

At the same reachable total heavy-frame count for every complete development
video, does joint redistribution of effective K across that video's ordered
768-frame windows provide material official localization headroom over both
uniform allocation and independent window-local RIME?

The experimental unit and bootstrap unit are the complete video. Windows never
serve as independent statistical samples.

## Implemented matrix

For each preregistered anchor budget, one exact-equality MCKP plan produces:

1. `uniform_same_total`;
2. `independent_exact_total`;
3. `joint_oracle`;
4. `joint_same_k_uniform_positions`;
5. `shuffled_null`.

Every strategy has the same per-video reachable and realized total effective K.
Short-window nominal aliases are deduplicated before optimization. Scores use
the frozen int64 quantization and deterministic objective/risk/lexicographic
tie rule. GT is permitted only for development-oracle planning/evaluation and
is never passed to the runtime selector.

## Evidence contract

Each cell must prove all of the following in immutable receipts:

- exact replay/ledger window identity and no-padding effective-K execution;
- strict RIME-full EMA compatibility with zero missing/unexpected keys;
- complete `ThumosSlidingDataset` traversal and exact per-video window counts;
- actual cross-window result aggregation;
- one NMS invocation for every expected video, with pre/post counts;
- saved post-NMS prediction bytes and SHA-256;
- successful official OpenTAD mAP evaluator invocation on that same object;
- checkpoint, annotation, evaluator, NMS, config and implementation identities;
- official-final remains sealed.

Missing prediction keys fail closed. Explicit empty prediction lists remain
valid. A shuffled null may be degenerate for an individual video whose K
histogram cannot be permuted, but an anchor that is degenerate for every video
fails during planning. The complete strategy-by-anchor matrix is required; a
partial matrix cannot authorize Stage 2.

## Admission rule

Before execution, an exact clean Git commit must freeze a hash-bound
preregistration containing:

- one primary endpoint and direction;
- materiality, lower-confidence-bound and noninferiority thresholds;
- video bootstrap sample count and seed;
- the full intersection-union primary family;
- optional guardrails;
- surrogate Spearman, sign-agreement and worst-rank-error thresholds;
- a finite non-negative oracle risk weight.

All registered primary cells, guardrails and surrogate checks must pass.
Suggested Pro-report numbers are not defaults and cannot be chosen after
observing the development matrix.

## Current verification

Local compilation and launcher syntax pass. Sixty-one focused Stage-1/RIME
non-Torch tests and twenty-three required C3 regression tests pass. The
Torch-dependent strict-architecture and end-to-end merge/NMS/evaluator receipt
tests require the remote Linux environment.

Correct empirical statement:

`No paper-admissible empirical conclusion is available yet.`
