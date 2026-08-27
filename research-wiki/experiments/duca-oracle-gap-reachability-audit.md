---
type: experiment_audit
node_id: exp:duca-oracle-gap-reachability-audit
status: tested_diagnostic
updated: 2026-07-13
method_commit: 0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d
worktree_head: 84bcb2b62684688315365066c566a7a6a8b695fc
paper_claim_allowed: false
---

# DUCA Oracle Gap and Reachability Audit

## Question

Can a deployable, low-cost frame selector approach the practical full-frame
AdaTAD baseline, and why does the current learned policy not approach the
GT-boundary Oracle?

The target remains offline full-window TAD with lower decode-to-output cost and
protected high-tIoU localization. Fixed T=768/K=384 is an attribution setting,
not the final dynamic-budget claim.

## Evidence boundary

| System | Best Avg-mAP | mAP@0.7 | Evidence role | Strictly matched here? |
|---|---:|---:|---|---|
| Historical full-frame stride-1 AdaTAD | 68.97 | 47.46 | practical dense reference | no |
| Grid-aware exact uniform K=384, Job 1150842 | 65.696 | 44.05 | historical uniform reference | no |
| Legacy transition beta=0, Job 1159416 | 64.34 | 41.97 | invalidated diagnostic | no |
| Legacy transition beta=0.25, Job 1159417 | 63.55 | 42.28 | invalidated diagnostic | no |
| GT-boundary Oracle, Job 1001959 | 76.67 | 65.83 | privileged diagnostic | no |

The two legacy transition runs used the defective uniform homotopy start. The
corrected transition helper at `0ea4e15` has focused tests but no replacement
full train. None of the rows above forms a same-commit causal comparison.

## Why the Oracle is unusually strong

The historical Oracle uses GT segments during train, validation, and test. For
every start/end it selects the center plus/minus radius 2 first, then fills the
remaining K positions uniformly from the other positions. It also remaps GT
segments onto the label-dependent selected axis before detector training and
evaluation.

Consequently, the Oracle does more than identify informative RGB frames. Its
selection set and temporal coordinate transform both carry privileged label
information. It guarantees dense boundary support, compresses less useful
regions, and makes the detector's ordinal geometry easier. The 76.67 result is
therefore evidence that GT-informed temporal allocation can help this detector;
it is not a learnable ceiling and not a deployable target.

For a deployable selector S=f(X), S adds no target information beyond the full
video X. For the Oracle S=f(X,Y), the selected positions can encode Y. The
value gap between those two policies is privileged-information value, not an
optimization error that more epochs necessarily remove.

## Current root causes

1. **Weak observability.** The legacy epoch-89 coarse probe has AUROC 0.6214,
   AUPRC 0.4111, and ECE 0.1710. Action and background probability ranges
   overlap heavily.
2. **The learned ranker trails the audited hand proxy.** Learned transition AP
   and AUROC trail `abs_delta_p_action + uncertainty_peak` at every audited
   radius. The exporter mislabeled that compound score as raw transition;
   pure `abs(delta p_action)` remains unmeasured in this diagnostic.
3. **Global ranking destroys a very strong coverage prior.** Learned selection
   reaches r1 endpoint recall 0.8437 versus exact uniform 0.9991, and both-end
   r1 coverage 0.7108 versus 0.9982. Mean max hole grows from 1.83 to 11.86.
4. **The coverage loss can reward clustering.** `local_boundary_coverage_loss`
   takes `-log` of an unnormalized local occupancy sum. The sum can exceed one,
   so the loss becomes negative and keeps rewarding duplicate local mass.
5. **The detector bridge is only a soft surrogate.** Forward uses hard Viterbi
   RGB positions, while backward differentiates through soft mixtures. Its
   direction has not been validated against feasible hard one-swap detector
   loss. The legacy beta=0.25 run does not show a stable Avg-mAP benefit.
6. **The allowed hole is too loose for K=T/2.** G=15 is feasible but far weaker
   than the one-to-two-frame spacing naturally supplied by exact uniform.
7. **Geometry and cost remain unresolved.** The detector trains on an ordinal
   selected axis and predictions are remapped afterward. The current profile
   does not establish complete decode-to-output savings.
8. **A control-path implementation bug remains.** The legacy/direct
   `stable_selection` arm still uses midpoint targets that collapse to equal
   logits at T=768/K=384. It is not exact uniform.

The transition-only architecture is conceptually consistent with the original
indirect-boundary idea: the coarse path learns action/background state and the
selector consumes temporal changes rather than direct endpoint heads. The
failure is behavioral and optimization-related, not a return to actionness
top-k. Detector loss updates the transition scorer through the bridge but is
intentionally blocked from the coarse probe, so this is protected multi-loss
training rather than fully shared end-to-end adaptation.

## Reachability verdict

Approaching or matching a finite, practical full-frame AdaTAD is plausible but
unproven. Historical exact uniform is already within about 3.27 Avg-mAP of the
68.97 reference under a different protocol, so half-rate heavy computation is
not obviously information-starved. Matching the full-information Bayes optimum
is generally impossible unless discarded frames are conditionally redundant
given the selected frames, timestamps, and dense low-cost evidence.

The current global transition top-k policy is not a credible route to the
Oracle. A credible route must preserve coverage first and learn only the
residual allocation that the scout can predict reliably.

## Decisive next design

Preferred falsifiable direction:

1. Start from exact uniform or a one-per-stratum coverage skeleton.
2. Permit only a bounded set of exact-K/max-gap feasible local swaps or segment
   quota changes; do not rerank all K positions globally.
3. Keep dense low-cost state/transition features, original timestamps, and
   support widths visible to the downstream model.
4. Train a utility head with train-only hard counterfactual detector-loss
   differences for the same feasible swaps used at inference. Stop-gradient
   utility distillation is a new claim, not evidence for the old bridge claim.
5. Before full training, compare pure `abs(delta p_action)`, the compound hand
   proxy, and the learned ranker separately; require learned residual utility
   to preserve r1 and both-endpoint coverage near exact uniform and show
   positive held-out hard-swap rank correlation.

The subsequent `1fc7037` Pro review narrows this proposal to DUCA-CellCF:
one-per-uniform-cell local deformation with hard counterfactual utility
distillation. It is a `discussed` bounded appeal, not implemented evidence.

The alternative with the strongest literature support is a dense-light,
sparse-heavy residual path: retain cheap features at every time point, refresh
heavy semantics at selected points, and reconstruct a dense physical-time
feature grid. This may be more capable than literal frame deletion but changes
the method from a pure selector to adaptive heavy-feature refresh.

## Claim status

- Current DUCA transition-only: `tested`, with legacy
  `completed_protocol_invalid_diagnostic` evidence.
- Corrected learned-policy performance: `unproven`.
- Oracle-level deployable performance: unsupported target.
- Matching practical dense AdaTAD at lower full-stack cost: plausible research
  hypothesis, not empirically supported.
- Paper status: `HOLD`.
