# Experiment Audit Report

**Date**: 2026-07-15
**Auditor**: independent read-only Codex reviewer agents
**Project**: ChronoTransport CT-P3R-3S-r2 pre-experiment implementation
**Overall Verdict**: FAIL — not ready for registration or formal execution
**Integrity Status**: fail

This verdict does not mean that a fabricated paper result was found. It means that no formal result
exists and the current implementation cannot yet produce one through the registered stop-chain.

## Checks

### A. Ground Truth Provenance: FAIL

The implemented Gate-4 statistic surface explicitly refuses `formal=True` for caller-supplied mappings
and labels every usable output test-only (`opentad/models/chronotransport/gate4.py:646`, `:666`, `:727`).
That is a valid bounded lock, but the repository has no formal producer binding official population,
predictions/GT, post-Stage-C checkpoints and registration R. Therefore GT provenance cannot be proved
for the missing formal workflow. No reviewed runtime path showed GT, teacher, replay ledger or raw
prediction cache entering scheduler decisions.

### B. Score Normalization and Cost Evidence: WARN

The pure Gate-4 test adjudicator calls the repository's official AP routine
(`opentad/models/chronotransport/gate4.py:12`, `:436`) and no metric divided by the model's own maximum
was found. Formal Stage-C now requires `cost_is_measured is True`
(`opentad/models/chronotransport/stage_c.py:799`, `:815`, `:827`). However, that boolean is not an
immutable profile identity: no formal runner binds the cost artifact hash, hardware/environment,
producer identity or requested/executed cost bytes to I/R.

### C. Result File Existence: PASS — no phantom result claim detected

There is no registered implementation commit I, registration-only R, formal Gate report, Stage-B/C
checkpoint chain, Gate-4 result or ChronoTransport Slurm Job ID. Repository memory consistently labels
the route as pre-experiment. Consequently there is no number to validate and no number that may be used
in a paper claim.

### D. Dead Code and Reachability: FAIL

The Stage-C primitive enforces a 4,200-update ceiling (`opentad/models/chronotransport/stage_c.py:2477`),
but the planned runner work remains unchecked (`docs/superpowers/plans/2026-07-12-chronotransport-ct-p3r-3s-r2-implementation.md:160`).
Formal Stage-C, matched-dense and Gate-4 runners, their launchers, the final validator and runner tests
are absent. The current primitive also requires exactly one top-level model forward
(`opentad/models/chronotransport/stage_c.py:1061`), while the unresolved A3/A4 contract must define the
real detector loss-normalizer update and paired dense/counterfactual forwards.

### E. Scope Assessment: FAIL

Only focused unit/integration tests and synthetic/test-only Gate-4 adjudication have run. There are no
140-window formal Stage-B seeds, no independent 30/30 calibration/evaluation Gate-2/3 run, no 4,200
successful Stage-C updates, no matched-dense run and no official full-video Gate-4 population. The
approved specification requires the latter (`docs/superpowers/specs/2026-07-12-chronotransport-ct-p3r-3s-r2-design.md:683`).

### F. Evaluation Type: synthetic_proxy / test_only

Current Gate-4 evidence is explicitly `chronotransport-r2-gate4-test-only-v1`; no formal `real_gt`
evaluation exists. The current tests validate implementation invariants, not scientific performance.

## Independently Approved Bounded Locks

- `APPROVE_GATE4_CALLER_EVIDENCE_LOCK_FINAL`: caller-owned raw mappings cannot mint formal Gate-4
  evidence; this does not implement a formal producer.
- `APPROVE_STAGEC_MEASURED_COST_FLAG_LOCK`: formal Stage-C rejects proxy-cost summaries; this does not
  authenticate a cost profile or approve Stage C.

## Action Items

1. Obtain explicit approval of A1--A4, freeze a new spec-only commit and obtain an independent spec-only
   review. Do not reinterpret the proposal in implementation code.
2. Implement registration-bound Stage-C and matched-dense runners, exact 4,200-success ledgers,
   checkpoint/resume/EMA/LR/exposure identity and A3/A4 semantics.
3. Implement a repository-owned Gate-4 evidence producer and validator that binds the frozen official
   population/order, raw predictions/GT, live full-stack timing, static identity, post-Stage-C Gate-3
   unlock and clean detached R.
4. Close the exact registration source vector, create clean I and single-parent registration-only R,
   run `PRECHECK_ONLY=1`, then execute Gate 1 → Stage B → Gates 2/3 → Stage C → Gate 4 with permanent
   stop on any scientific Gate failure.

## Claim Impact

- Core speed/accuracy claim: **unsupported** — no formal experiment exists.
- Gate-4 caller-evidence isolation: **supported as a bounded code invariant only**.
- Stage-C measured-cost flag: **supported as a bounded code invariant only**.
- Registration readiness, experiment completion, paper readiness and deployment: **unsupported**.

## Reviewer-Independence Limitation

The reviews were performed by separate read-only Codex agents. The orchestrator cannot attest that
they use a different model family. The GitHub-only Pro prompt in
`docs/methods/2026-07-15-chronotransport-r2-current-github-pro-line-review-prompt.md` is therefore still
required for an external Pro review after the latest bytes are published to an immutable GitHub SHA.
