# Result-to-claim findings

## F001 — Residual centering passes the narrow seed-3407 accuracy screen

- `claim_supported`: yes, for the preregistered THUMOS14 Gate development
  screen only.
- Confidence: high for the authorization decision; medium for generalization.
- Evidence: fresh matched G1 cells, seed 3407, exact B=24576, 60 epochs/9,600
  successful updates, exact duplicate replay, common protocol/population, and
  centered-minus-control `+2.05 pp` Avg-mAP, `+2.14 pp` mAP@0.6, `+1.16 pp`
  mAP@0.7.
- Allowed conclusion: the single residual-window centering repair has detector
  utility on this development screen and deserves measured-cost evaluation.
- Not allowed: multi-seed significance, general Hybrid efficacy,
  complementarity, floor selection, efficiency, official-test, cross-detector,
  or final-method claims.

Recommended wording:

> On THUMOS14 Gate development, one fresh seed-3407 matched G1 comparison at
> B=24576 passed the preregistered residual-centering screen, improving
> Avg-mAP by 2.05 pp, mAP@0.6 by 2.14 pp, and mAP@0.7 by 1.16 pp. This
> authorizes, but does not establish, a counterbalanced full-stack cost study.

## F002 — Cost and paper claims remain unsupported

- `claim_supported`: no.
- Missing evidence: same-GPU ABBA+BAAB decode-to-NMS latency/energy/memory,
  cost non-inferiority decision, seeds 3408/3409, independent repeated cost
  Jobs, matched dense/fixed/random/free-token baselines, short-action/boundary
  analyses, second detector/dataset, and sealed official test.
- Next decision order: paired cost -> cost non-inferiority -> conditional
  seeds 3408/3409 -> broader paper matrix.
- Another Pro discussion is not required before the already specified paired
  cost experiment.
