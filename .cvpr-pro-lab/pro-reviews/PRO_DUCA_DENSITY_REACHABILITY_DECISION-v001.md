---
doc_id: PRO_DUCA_DENSITY_REACHABILITY_DECISION
version: v001
stage: DRAFT
author_role: pro-scientific-first-author-agent
intake_role: coordinator
status: accepted_local
scientific_decision: REVISE
evidence_class: BLOCKED_PRE_RESULT
project_id: g-p-6a796fef9a00819194024cf1de3bd697
project_url: https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697/project
fixed_commit: a6bdc084cc145c80b6b2c68d0a38f0deea3e8518
turn_id: duca-density-critique
nonce: DUCA-ARIS-DENSITY-v001-20260813T222019Z
dispatch_id: CENTRAL-ARIS-PRO-DUCA-DENSITY-v001
model: gpt-5.5-pro
effort: MAX_EFFORT_NOT_SEPARATELY_EXPOSED
raw_transcript: C:/Users/skywalker/.fastctx/jobs/j-e408ux/output.log
sources_status: inline self-contained packet; no Source mutation in this turn
---

# Training-side density reachability decision

## Decision

`REVISE`. The proposed official-validation GT-boundary oracle-density versus
exact-uniform comparison is not a valid upper-bound or deployability test. It
uses privileged labels for route selection, does not establish reader
reachability, and cannot kill the route merely because a lower confidence bound
crosses zero. No data, GPU, Slurm, metric, result, or paper claim is authorized.

## Replaced falsifier

Freeze one immutable video-disjoint split of the official training population:
`FIT` for detector/reader fitting and `CAL` for one reachability evaluation.
Official validation must be absent from all mounts and manifests. With one frozen
FIT terminal detector and identical production wrappers, compare:

- `U`: canonical exact-uniform positions through a constant density-logit path;
- `O`: privileged GT-boundary density, diagnostic only;
- `R`: deploy-visible `browser_memory -> duca_density_logits`, with no GT,
  teacher, cached prediction, or detector result entering inference.

The evaluator must compute dataset-level official mAP afresh for each of 10,000
paired video-cluster bootstrap resamples. The minimum worthwhile Avg-mAP effect
is +0.50 pp. `O` alone never supports a learned-density claim. Continue only if
both `O-U` and `R-U` clear their predeclared effect/lower-bound gates and
high-tIoU protections. Kill only when a relevant predeclared upper bound is
below the minimum worthwhile effect; otherwise hold.

## Authorized bounded next work

Evaluator may freeze `DUCA_DENSITY_REACHABILITY_PROTOCOL-v001` without data
access or execution. It must specify the FIT/CAL firewall, U/O/R tuple,
privileged-oracle boundary, bootstrap/evaluator unit, thresholds, output seal,
and required receipt schema. Builder is not yet authorized to write production
code; Critic remains idle until a complete tracked package exists.

## Evidence boundary

The ARIS density decoder is untracked dirty-worktree prototype code atop this
commit, with only local compile/fixture checks and an incomplete N16R4
pure-Python geometry probe. It is not production code, PRE_RUN evidence,
evaluator evidence, or an experiment result. Production reader/config/evaluator/
Slurm/full-training wiring remains absent.

