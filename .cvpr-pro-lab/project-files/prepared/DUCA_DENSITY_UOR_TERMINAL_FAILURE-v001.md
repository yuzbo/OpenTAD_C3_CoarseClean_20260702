---
title: DUCA_DENSITY_UOR_TERMINAL_FAILURE
version: v001
date: 2026-08-14
stage: DRAFT
author_role: coordinator
github_url: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_commit: 6576789468c1a7692d49b2ba94a638e01e7970f4
protocol_revision: DUCA_DENSITY_REACHABILITY_PROTOCOL-v001
parent_decision: PRO_DUCA_DENSITY_UOR_IMPLEMENTATION-v001
parent_receipt: CRITIC_DUCA_DENSITY_UOR_FOCUSED_IMPLEMENTATION_RECHECK-v001
evidence_class: IMPLEMENTATION_CORRECTION_BLOCKED_PRE_RESULT
---

# Terminal U/O/R implementation-loop failure packet

## Frozen chain and evidence boundary

The accepted Pro disposition authorized one minimal tracked production package,
one independent Critic review, at most one focused Builder correction, and one
focused Critic recheck. The package remains pre-result: no data, held-out or
official-validation access, runtime, remote/GPU/Slurm activity, training,
inference, evaluation, bootstrap, metrics, cost evidence, or claims occurred.

The final focused Builder correction is the clean commit
`6576789468c1a7692d49b2ba94a638e01e7970f4` with a reported 21 focused no-data
checks passing. It is not PRE_RUN admission or an experiment result.

## First-fail terminal finding

The independently rebound Critic's sole focused recheck reports
`REACHABILITY_PRE_RUN_BLOCKED / NEEDS_ATTENTION` and
`IMPLEMENTATION_CORRECTION`, with `SCIENTIFIC_AMBIGUITY=NONE`.

The corrected DUCA configurations define reachability entrypoint gates, but the
generic CLI-protection helper does not recognize the DUCA route/stage labels. As
a result, `tools/train.py` and `tools/test.py` accept arbitrary `--cfg-options`
before the later gate check. A caller can set the reachability gate's `required`
field false, disabling the gate before dataset/checkpoint/output and other
firewall bindings are verified. The Critic classifies this as a reachable
equivalent of the earlier launch-bypass defect and therefore withholds PRE_RUN
admission.

The first review's loader/firewall and sealing changes were structurally present
within static scope, but cannot close the package while this pre-gate override
bypass remains reachable.

## Terminal disposition and decision required

This is the second equivalent deterministic implementation defect in the same
frozen correction area. The one-correction policy therefore terminates the loop:
there is no third Builder correction, no further Critic recheck, no Evaluator
PRE_RUN, and no execution authority.

The Critic describes, but does not authorize, one possible simplification:
for `DUCA_DENSITY_REACHABILITY_PROTOCOL-v001`, reject every nonempty
`--cfg-options` before any merge rather than expanding the generic gate taxonomy.
It is not an adopted design and must not be implemented without a new scientific
decision.

Fresh exact-Project Pro must choose exactly one disposition:

1. `STOP` the U/O/R route; or
2. authorize a genuinely simplified replacement implementation/protocol,
   including the exact bounded Builder → independent Critic → existing Evaluator
   sequence and all no-execution/held-out safeguards.

The Pro review must not create another theory round, silently extend the old
correction loop, promote untracked prototype evidence, or grant any experiment
authority.
