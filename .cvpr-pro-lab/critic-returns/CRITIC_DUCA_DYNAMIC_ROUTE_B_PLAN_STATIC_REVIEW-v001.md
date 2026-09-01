---
title: CRITIC_DUCA_DYNAMIC_ROUTE_B_PLAN_STATIC_REVIEW
version: v001
date: 2026-08-14
dispatch_id: DUCA-ARIS-R-DYNAMIC-PLAN-v001
message_id: msg-20260814T085600Z-dynamic-route-b-static-review
role: Critic
terminal: DYNAMIC_ROUTE_B_PLAN_BLOCKED
evidence_class: STATIC_PLAN_REVIEW_ONLY / NOT_EXECUTED
next_owner: DUCA Coordinator terminal hold
single_recovery: none
---

# Durable Critic terminal receipt

## Terminal verdict

`DYNAMIC_ROUTE_B_PLAN_BLOCKED`

The frozen Builder package is not a self-consistent implementation contract.
This is a terminal no-correction disposition; no Builder correction, Pro turn,
Evaluator intake, PRE_RUN admission, role dispatch, or execution is authorized.

## Findings

### F1 — `IMPLEMENTATION_CORRECTION`: constant-density decoder identity is contradictory

In `DUCA_DYNAMIC_ROUTE_B_IMMUTABLE_PLAN-v001.md`, the general decoder
specification first uses midpoint quantiles, chooses the first CDF bin satisfying
`cdf[t] >= u`, computes `r = x - 0.5`, and projects by squared error with the
lexicographically smaller tuple winning ties (lines 264--276 and 367--381).
The same document separately requires constant logits to return
`floor((2*j+1)*T_v/(2*K))` bit-for-bit (lines 280--285).

For the frozen nominal case `T_v=768, K=384`, constant mass gives
`u_j=(2j+1)/768`, `bin=2j`, `x=2j+1`, and `r_j=2j+0.5`. Both `2j` and `2j+1`
have equal squared error, so the stated lexicographic rule selects `2j`; the
asserted specialization selects `2j+1`. The resulting tuples are respectively
`(0,2,...,766)` and `(1,3,...,767)`.

This breaks the required canonical-uniform identity, makes the fixed-K control
and constant-density path non-identical, and leaves no unique decoder behavior
for a future implementation. The defect is deterministic and does not require
runtime evidence.

Smallest owner/action would ordinarily be Builder clarification of one exact
decoder convention under the existing Pro route, but this dispatch expressly
authorizes no correction or recheck. Therefore the dependency remains terminally
unsatisfied.

### F2 — `IMPLEMENTATION_CORRECTION`: K-shuffle is not reproducibly frozen

The plan requires a "nonce-seeded permutation" within exact valid-length and
candidate-set strata (lines 244--248 and 418--425), but freezes neither the
nonce/seed nor the canonical row ordering and permutation algorithm. Unlike the
FIT/CAL/HOLD split and bootstrap, this control therefore has no exact reproducible
mapping. Different conforming implementations can generate different
content-to-K assignments and hence different O4/pilot conclusions.

This leaves the mandatory K-shuffle invalid-control attack unresolved. The
smallest owner/action would ordinarily be Builder specification of the exact
permutation identity, but no correction or recheck is authorized.

## Scope and boundary

The transport worktree and revision `6515ebf501b3c49be57ad4c37fc84d2fb4ae66d7`
were treated only as a review container and never as DUCA scientific identity.
No repository code/config/wiki change, command, test, import, interpreter, data,
model, checkpoint, official-validation/test, GPU, Slurm, remote, Git, browser,
Sources, Pro, metric, cost, or role-dispatch action was performed.

`next_owner=DUCA Coordinator terminal hold`

