# DUCA Admission v2.1 Pro response adjudication

Date: `2026-07-30`

Status: `corrigendum_returned_but_not_implementation_ready`

Source: `U-PRO-V21-FINAL-1`

Code snapshot reviewed by the external report:
`240423184a57849594fc1548f23b6acf8fbd4a94`

Source SHA-256:
`9e7efa045f0b2a01dfc52755a6376205346bf76673483b61573dd55951d7c871`

## Executive decision

The report is not accepted verbatim.

Its research direction and most architectural decisions are accepted, but the
proposed uncertainty and catastrophic gates are not yet implementation-ready.
The report therefore authorizes a final protocol-correction discussion and
pure engineering checks; it does not yet authorize production Admission v2.1,
Phase 1, formal training, learned H-RIME, Phase 4, or official-final.

The requested response returned as `U-PRO-V21-CORRIGENDUM-1`. Independent
audit found five remaining P0 issues and supersedes the execution decision in
this document. See
`docs/superpowers/plans/2026-07-31-duca-v2-1-corrigendum-independent-audit.md`.

The correct current evidence statement remains:

`No paper-admissible empirical conclusion is available yet`.

## Accepted decisions

1. Keep the offline-TAD pure selected-axis pre-backbone plugin as the paper
   mainline. Keep the physical-time head as a separately named diagnostic.
2. Admission v2.1 is an executor/semantics gate, not a model-performance or
   noninferiority experiment.
3. Use three disjoint 32-video development roles:
   `scale_fit`, `calibration`, and `admission_holdout`.
4. Each role contains 22 natural-full videos and 10 natural-short videos. Four
   long videos are metadata-only reserve. Coverage is role-level, not
   per-video. Synthetic crops, padding and replicated frames are forbidden.
5. Keep role assignment metadata-only and candidate-independent. Rank tertiles,
   deterministic hash tie-breaking and immutable manifests are acceptable.
6. Keep old Admission v2 permanently superseded and unable to authorize a new
   stage.
7. Accept `NO-GO_FOR_NI` for Admission. No mAP NI margin is invented from
   reporting precision, calibration variance or candidate output.
8. Preserve exact code/data/receipt identity, process separation, planned-cell
   completeness, atomic exclusive shard publication and truthful
   enforced/attested/observed fields.
9. Accept a separate full-200, five-by-forty OOF design after method freeze.
   Accept two-GPU DDP with local batch one as the only formal global-batch-two
   implementation, subject to an explicit global loss-normalization and AMP
   synchronization proof.
10. Keep exact-211 OpenTAD evaluation sealed as one final transaction.
11. Preserve the order:
   Admission v2.1 -> selected-axis Phase 1 -> window-local development ->
   same-realized-cost H-RIME oracle -> learned H-RIME only after oracle pass ->
   full-200 refit -> one-shot exact-211 evaluation.
12. Use the name `AdapTok-inspired TAD budget allocation baseline`; do not
   claim an official reproduction or first adaptive video allocation.

## Accepted with wording corrections

### Role balance

The long-video three-item blocks provide deterministic local rank
stratification. They do not, by themselves, prove a prespecified bound on
duration or window-count mean imbalance. Receipts may report achieved
imbalance, but the paper may not claim statistical balance unless an objective
and acceptance bound are preregistered.

### Max-family terminology

The report's statistic uses one bootstrap-estimated marginal scale per metric:

`max_c ((delta_c^b - delta_hat_c) / s_c)`.

That is a fixed-scale standardized maxT construction. It is not a strict
bootstrap-t/studentized procedure with a replicate-specific standard error.
The final protocol must either:

- call and justify it as `single_step_fixed_scale_standardized_maxT`; or
- specify a valid nested/replicate-specific studentization algorithm.

No implementation may silently use the word `studentized` for the former.

## Rejected or unresolved as written

### 1. Sparse crossed-bootstrap empty-cell semantics

The proposed multinomial column multiplicities can give a video
`D_v^b = 0`. Dropping that video and changing the replicate denominator turns
the frozen equal-video estimand into a random conditional estimand. The report
does not establish coverage for that operation.

The final adjudication must choose exactly one of:

1. a positive multiway multiplier scheme that cannot create empty video
   denominators;
2. a rigorously justified conditional multinomial algorithm with a complete
   redraw rule; or
3. a denser incidence design whose registered estimator avoids the ambiguity.

It must provide a hand-computed example, simulation coverage under row,
process and interaction effects, and a deterministic seed/quantile contract.

### 2. Monte Carlo stability

Requiring two independent 50k streams to return an identical binary pass
vector is conservative but is not a calibrated Monte Carlo error criterion.
Near a boundary it may fail because of simulation noise even when the
underlying procedure is stable.

The replacement must bind:

- a numerical Monte Carlo error estimate for the critical value and every
  simultaneous upper bound;
- an absolute or scale-normalized tolerance fixed before candidate output;
- an escalation rule from 100k to 200k;
- a fail-closed rule after 200k.

Binary agreement may remain a diagnostic, but not the sole convergence proof.

### 3. Catastrophic max-over-max ratio

`max(holdout discrepancy) / max(calibration null discrepancy) <= 1` is not a
family-wise calibrated test. Its false-failure probability depends strongly on
the unequal number of videos, processes, windows and components in the two
roles. A tiny nonzero calibration maximum also creates an unstable ratio.

The replacement must separate:

- exact catastrophes: missing tensor, shape/dtype mismatch, nonfinite value,
  missing planned cell, shard/ledger failure;
- numeric-tail evidence: a candidate-independent, count-aware bound defined at
  the video/process cluster level.

A split-conformal cluster maximum, simultaneous standardized bound, or another
finite-sample method may be proposed, but its experimental unit, multiplicity,
sample-count dependence and attainable alpha must be explicit. The raw
max-over-max rule is rejected.

### 4. Runtime-isolation threat model

Exact clean checkout, hashes, input allowlists, symlink-escape rejection,
fresh roots, independent Slurm processes and immutable shards are hard gates.
Observed mount, route, socket and environment fingerprints are diagnostics.

The report additionally makes administrator-attested network deny, mount
namespace allowlisting and read-only mounts mandatory. That is not accepted
without a stated threat model and a written N16R4 feasibility receipt. An
audited exact program does not automatically require an adversarial-malware
sandbox to establish executor semantics. The final review must decide whether
these controls are:

- required scientific anti-leakage controls;
- optional security strengthening; or
- unavailable limitations that must be reported truthfully.

No receipt may claim a control is enforced when it is only observed.

### 5. Existing H-RIME oracle finalizer

The current Stage-1 finalizer is not the gate described by the report. It uses
configurable NI-style thresholds and mean guardrails, does not implement the
required simultaneous guardrail intervals or explicit shuffle/null
superiority, and audits rank error rather than the proposed absolute official
delta envelope.

This mismatch is recorded now, but no H-RIME code change is authorized before
Admission v2.1 and selected-axis Phase 1 close. Before the oracle is run, the
gate must be replaced by one preregistered primary cost/endpoint and an exact
secondary family.

## Immediate authorization

Allowed now:

1. preserve and publish this adjudication;
2. run the exact-clean Linux/PyTorch code gate for the current fail-closed
   code snapshot;
3. conduct read-only metadata checks;
4. prepare simulation-only prototypes for competing crossed-uncertainty
   definitions, with no candidate data;
5. obtain a cluster administrator feasibility statement for the proposed
   isolation controls.

Not allowed now:

- production Admission v2.1 workers or holdout opening;
- any Phase-1/2/3/4 model training;
- learned H-RIME implementation or training;
- full-200 target generation/refit;
- official-final access.

## Next bounded Pro discussion

The next discussion is a statistical-protocol corrigendum, not an architecture
or novelty ideation round. It must return one executable decision for the four
unresolved items above and must not reopen already accepted model semantics.

### Discussion prompt

```text
You are the final statistical and experiment-protocol adjudicator for DUCA-RIME.
Work read-only. Do not modify the repository, launch jobs, or report any model
performance.

Repository:
https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
Branch:
codex/duca-rime-20260727
Code snapshot underlying the prior review:
240423184a57849594fc1548f23b6acf8fbd4a94

First verify remote -> branch -> HEAD and show the commit/tree identity. Then
read in full:
- research-wiki/query_pack.md
- research-wiki/anti_repetition.md
- research-wiki/source_registry.md
- docs/superpowers/plans/2026-07-29-duca-admission-v2-1-repair-plan.md
- docs/superpowers/plans/2026-07-30-duca-v2-1-pro-response-adjudication.md
- docs/superpowers/specs/2026-07-28-hrime-v1-budget-conserving-design.md

Inspect the exact current code for the old Admission gate, feasibility auditor,
evidence writer, training authorization, H-RIME core and Stage-1 finalizer.

Do not rediscuss selected-axis versus physical-head architecture, the 32/32/32
role counts, NO-GO_FOR_NI for Admission, full-200 rather than 100-video formal
refit, DDP2 global batch two, exact-211 final evaluation, AdapTok naming, or the
oracle-before-learned-H-RIME order. Those decisions are frozen.

Adjudicate only:

1. Crossed uncertainty
   - Give the exact estimand for sparse video x independent-process cells.
   - Choose one exact resampling/weighting algorithm.
   - Resolve D_v=0 without changing the equal-video estimand silently.
   - State whether the max-family method is fixed-scale standardized maxT or
     truly studentized; provide formulas and coverage assumptions.
   - Freeze quantile convention, seed, replicate count, MC error measurement,
     escalation and final failure rule.
   - Provide hand-computed and simulation tests.

2. Catastrophic numeric tails
   - Reject or repair the calibration-max/holdout-max ratio.
   - Separate exact structural catastrophes from numeric tails.
   - Give a count-aware, candidate-independent, video/process-cluster method
     with an attainable one-sided alpha and multiplicity rule.
   - Specify all-zero, tiny denominator, nonfinite and unequal-window-count
     behavior.

3. Runtime isolation
   - State the explicit threat model.
   - Classify every control as repository-enforced, cluster-attested,
     observed-only, optional strengthening, or unavoidable limitation.
   - Decide whether network deny/mount namespace/read-only mount attestation is
     scientifically mandatory for executor admission or merely security
     hardening. Justify the answer without overclaiming.
   - Return an exact receipt schema and fail-closed conditions.

4. H-RIME oracle gate correction
   - Reconcile the current Stage-1 finalizer with the frozen future gate.
   - Freeze one primary cost/endpoint (default candidate: realized mean K=384,
     mAP@0.7 joint oracle minus independent-window RIME).
   - Define simultaneous guardrails, shuffle/null attribution and absolute
     surrogate-error calibration.
   - Keep this as a future patch after Phase-1 closure; do not authorize learned
     H-RIME now.

For every conclusion use:
[REPO_FACT], [EXTERNAL_FACT], [INFERENCE], [DECISION], or [BLOCKER].

Required output:
A. repository identity receipt
B. executive GO/NO-GO for protocol implementation
C. exact corrected mathematics
D. deterministic pseudocode
E. machine-readable decision manifest
F. focused unit/simulation tests
G. file-by-file minimal patch plan
H. explicit actions still unauthorized

If the exact repository snapshot or a required source cannot be read, stop with
REPOSITORY_NOT_VERIFIED. Do not fill gaps with plausible prose.
```

## Post-corrigendum implementation order

Only after the corrigendum returns an implementation-ready `GO`:

1. add the v2.1 role generator and schema while preserving old-v2 rejection;
2. add the corrected crossed uncertainty and numeric-tail modules;
3. add pure deterministic and simulation tests;
4. add truthful runtime-isolation receipts;
5. pass local non-Torch checks and exact-clean Linux/Torch/DDP gates;
6. produce a metadata-only role manifest;
7. obtain any required cluster attestation;
8. implement real-video/full-model process workers and the held Slurm DAG;
9. run scale-fit, seal calibration, and open holdout once;
10. release Phase 1 only from a verified v2.1 receipt.
