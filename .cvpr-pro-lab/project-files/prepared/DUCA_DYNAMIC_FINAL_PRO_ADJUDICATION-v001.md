---
title: DUCA_DYNAMIC_FINAL_PRO_ADJUDICATION
version: v001
date: 2026-08-14
stage: FINAL_SCIENTIFIC_ROUND_MATERIAL_ONLY
author_role: coordinator
route_id: DUCA_HIERARCHICAL_DYNAMIC_PHYSICAL_ACQUISITION-v001
evidence_class: STATIC_PLAN_REVIEW_ONLY / NOT_EXECUTED
supersedes: DUCA_DYNAMIC_ROUTE_AND_REAL_VIDEO_EXPERIMENT-v001
---

# Final Pro adjudication packet — Dynamic DUCA definitions

## Exact decision requested

This is the third and final allowed scientific round for the current
dynamic-budget route: DeepSeek round-1 proposed Candidate B, and a fresh
Project Pro previously froze it at plan-only status. The independent Critic
then closed the role correction loop as `DYNAMIC_ROUTE_B_PLAN_BLOCKED`.

Fresh exact-Project Pro must make one decisive disposition:

- **FREEZE** both unresolved definitions and specify their complete, unique,
  implementation-independent contracts, then state the bounded next admission;
- **REVISE**, **PIVOT**, or **STOP** the dynamic route if either definition
  cannot support an interesting, isolatable Route-B mechanism.

Pro must not turn fixed K into the final route. Fixed K remains only an
attribution/control/fallback. No answer may authorize code, tests, role work,
PRE_RUN, data access, official validation/test access, GPU, Slurm, remote work,
pilot, training, inference, evaluation, metrics, costs, or paper claims.

## Scientific mechanism that remains under adjudication

Route B is hierarchical Dynamic DUCA, not generic adaptive sampling. An outer
per-video/window discrete policy selects heavy-frame K from train-only hard
detector utility minus paired-boundary/high-IoU failure risk and frozen realized
cost. One shared K-independent inner physical density then emits exactly K
strictly increasing physical positions; detector proposals are mapped from the
compact acquisition axis to physical time before unchanged filtering, top-k,
IoU, NMS, or evaluation.

Its potentially interesting prediction is that videos/windows require different
heavy-compute budgets for localization risk, and that this protects high-IoU
boundaries at matched realized cost where uniform or learned fixed-K cannot.
That prediction has no experimental support. It remains contingent on the
definitions below and on later O1--O4 kill gates.

## Binding Critic return and the two ambiguities

Source return:
`.cvpr-pro-lab/critic-returns/CRITIC_DUCA_DYNAMIC_ROUTE_B_PLAN_STATIC_REVIEW-v001.md`.
Its evidence class is `STATIC_PLAN_REVIEW_ONLY / NOT_EXECUTED`.

### F1 — one canonical constant-density / uniform identity is absent

The frozen general decoder says: for midpoint quantiles, take the first bin
whose CDF is at least the quantile, invert within the physical unit bin, then
project to strictly increasing integers by squared error with the
lexicographically smaller tuple breaking ties. At `T_v=768,K=384` and constant
mass, that contract selects `(0,2,...,766)`. The same plan separately mandates
the canonical-uniform tuple
`floor((2j+1)T_v/(2K))=(1,3,...,767)`.

Pro must choose and state a single exact convention that resolves this conflict:
the canonical uniform identity, the CDF/in-bin coordinate convention, the
integer objective and tie rule, and the constant-density specialization must be
mutually identical. It must preserve a physical exact-K contract and explain
whether the resulting fixed-K control and constant-density path are a valid
common attribution anchor. If no such definition is scientifically defensible,
Pro must REVISE/PIVOT/STOP rather than leave an implementation choice.

### F2 — K-shuffle cannot yet isolate content-to-budget causality

The six-arm plan says K-shuffle computes Dynamic-B assignments then applies a
nonce-seeded permutation within exact valid-length/candidate-set strata, keeping
the K histogram and cost while breaking content correspondence. It omits the
actual nonce/seed, the canonical row ordering, and the permutation algorithm.

Pro must freeze all of those elements—or reject the control as unable to isolate
the outer dynamic-budget mechanism. The definition must give an unambiguous
mapping for every eligible row and preserve the predetermined stratum, K
histogram, cost accounting, and train/FIT-CAL-HOLD firewall. It must state the
fallback disposition for collapsed strata or invalid inputs rather than permit
silent reshuffling.

## Existing real-video contract — context only, not authorization

Any later route retained by Pro must keep the official THUMOS14/OpenTAD-AdaTAD
comparison contract: training-population-only video-disjoint FIT/CAL/HOLD;
official validation/test inaccessible to selection; unchanged detector/loss/
pre-NMS physical mapping/NMS/evaluator; and six sealed arms:

1. native dense;
2. canonical physical uniform K384;
3. learned fixed K384 with the shared density reader and no outer policy;
4. full dynamic K Route B;
5. dynamic K-shuffle; and
6. dynamic no-risk with only the paired-risk contribution masked.

O1 dynamic-oracle headroom, O2 exact-K transport, O3 train-only hard-utility
predictability, O4 paired-risk superiority over actionness/transition/no-risk/
K-shuffle, and decode-to-final-serialization full-stack cost are hard gates.
None has run. A future N16R4 one-seed real-video pilot is conceivable only after
all structural, PRE_RUN, O1--O4, resource, and later Pro gates; formal training
requires a successful pilot and separate authorization.

## Provenance and negative boundary

- The repository root is a dirty DUCA worktree whose base `a6bdc084...` is a
  SparseHead commit, not a DUCA revision or admissible experimental identity.
- Existing `PrefixMarginalUtilityBudgetController` / center-radius code is
  Candidate-A historical negative infrastructure, not a Route-B dependency.
- Untracked fixed-K `density_decode.py` is prototype-only; old U/O/R and sealed
  replacement packages are terminal negative evidence and forbidden reuse.
- There is no Route-B production patch, no PRE_RUN, no data traversal, no pilot,
  no training, no official metric, no cost result, and no paper claim.

The final decision must preserve this boundary. It may authorize at most the
next explicitly bounded planning step; it cannot upgrade static reasoning to
empirical support.
