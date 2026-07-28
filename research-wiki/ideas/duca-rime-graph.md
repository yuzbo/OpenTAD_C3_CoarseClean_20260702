# Idea: Pair-Risk Graph RIME

## State

`discussed`, `not_yet_designed`, `not_implemented`, `not_tested`

This is a post-v1 candidate. It is not authorized to change the frozen
DUCA-RIME four-stage transaction.

## One-sentence method

Pair-Risk Graph RIME makes localization risk an edge cost inside the physical
exact-K position decoder, so budget and positions are chosen from a structured
utility-risk path rather than from independent frame scores plus a video-level
risk scalar.

## Algorithm candidate

For candidate time points `i<j`, predict node utility `u_j` and consecutive
selection risk `r_ij` from cheap inference-visible evidence. For each registered
budget K, solve the physical-time DAG recurrence

`DP[k,j] = u_j + max_i(DP[k-1,i] - lambda_r * r_ij - lambda_g * gap_ij)`

over allowed monotone edges, exact cardinality, required anchors, and the frozen
physical gap cap. A forward-backward relaxation supplies training gradients.
The budget controller compares the best structured path at each K using the
same frozen-price and no-padding cost contract as v1.

## Publishable distinction

The candidate contribution is not adaptive token count. It is structured
pair/boundary-risk protection inside an exact physical-time acquisition graph,
with realized-K-matched controls and high-IoU/short-action falsifiers. The named
AdapTok-TAD arm remains mandatory.

## Minimal falsification ladder

1. **Oracle headroom:** using privileged counterfactual targets only for
   train/development analysis, test whether a pairwise graph oracle improves
   `mAP@0.7`, short-action mAP, or pair support at exactly the same K.
2. **Target learnability:** cross-fit the pair-risk target and require useful
   ranking plus calibrated low-risk coverage using inference-visible evidence.
3. **Decoder causality:** compare full graph, risk-off graph, shuffled edges,
   v1 `weak_overlap`, `D-no-risk`, and exact `U-same-K`.
4. **Full-stack check:** include graph-decoder overhead in latency, energy,
   throughput, and memory.

## Stop rules

- Stop before implementation if the pairwise oracle has less than a
  pre-registered `0.5` point `mAP@0.7` headroom over the v1 decoder at matched K.
- Stop if cross-fitted pair risk is uncalibrated or cannot cover a non-vacuous
  low-risk subset.
- Stop if gains disappear under edge shuffle, fail high-IoU/short/pair-support
  gates, or cost matching to `U-same-K` fails.
- Do not open official-final data or promote this candidate while v1 remains
  `experiment_running`.

## Deferred alternatives

- **Conformal-risk RIME:** use a train-only upper risk bound for conservative K
  selection if v1 fails calibration rather than position coverage.
- **Sequential RIME:** acquire uniform anchors then a second information-gain
  round only if one-shot acquisition leaves clear oracle headroom; this has the
  highest latency, cache, and AdapTok-overlap risk.
