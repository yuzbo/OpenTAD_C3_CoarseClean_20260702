---
doc_id: MODEL_EXPERIMENT_HISTORY
version: v006
status: prepared_pending_central_project_source_sync
date: 2026-08-11
supersedes: MODEL_EXPERIMENT_HISTORY-v005.md
stage: DRAFT
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
evidence_class: BLOCKED_PRE_RESULT
---

# DUCA model and experiment history

## 2026-08-11 — Pro resolved the P0 density mechanism ambiguity

Fresh Oracle Project decision `PRO_P0_ROUTE_ADJUDICATION-v002` supersedes the
ambiguous v001 density definition with
`DUCA_FIXEDK_BOUNDED_DENSITY_QUANTILE_ACQUISITION-v002`. The route uses a newly
named `duca_density_logits` tensor from a density-only reader attached to dense
`browser_memory`, not any legacy slot/allocation/top-k/rank/quota output. It
defines per-sample positive trapezoidal density mass, endpoint-inclusive
inverse-CDF quantiles, exact-constant canonical-uniform degeneration, and a
deterministic constrained integer hard decoder.

The same decision permits two independent claim-neutral repairs: one canonical
integer-half-up uniform generator and exactly-once selected-q-to-physical-dense
transport before NMS. It rejects interpreting a legacy selector signal as a
density field and does not authorize any learning surrogate, experiment, or
result claim.

No experiment changed state. There are no executed patches/tests, remote
commands, datasets, checkpoints, metrics, costs, GPU/Slurm jobs, result
comparisons, or paper results. The next evidence is a Builder minimal change
plan, then a bounded patch and static closure review under the Pro decision.
