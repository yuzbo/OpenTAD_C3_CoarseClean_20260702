---
type: source
source_id: source:chronotransport-r2-pro-github-code-audit-response
date: 2026-07-13
verdict: REVISE_IMPLEMENTATION_BEFORE_REGISTRATION
review_mode: external_pro_github_only_source_audit
review_target: 4b07020acb2611c3f085488d2f678f3be037f1be
---

# ChronoTransport r2 Pro GitHub Code Audit — Absorption

## Source certificate

- Original attachment: `bf3c8b10-951f-4765-87d5-53c6ba02b7dd/pasted-text.txt`.
- Original attachment SHA-256:
  `1B3A02373366A95654C00A5FE76F451F800D16A877B2688BB460674B25849142`.
- Original byte/line identity: 69,815 bytes, UTF-8 without BOM, 1,429 CRLF-delimited lines.
- Exact text archive (newline-normalized only):
  `research-wiki/sources/2026-07-13-chronotransport-r2-pro-github-code-audit-response.md`.
- Archive SHA-256 after CRLF-to-LF normalization:
  `5419A46D7269AFAB9EB1E61A0270E82069916A361DDCBDB5B468BD0D1B0BFBA6`; normalized-text
  equality with the attachment was verified.
- GitHub review target: repository `yuzbo/OpenTAD_C3_CoarseClean_20260702`, commit
  `4b07020acb2611c3f085488d2f678f3be037f1be`.
- The reviewer found the r2 specification blob at the target commit identical to approved commit
  `e4422f5`; Git blob `0f54c9392f512c29f7ef59eb0afda61fe8dfa5f2`.
- The reviewer did not independently recompute the specification SHA-256. The project-provided
  normative value remains
  `87FA305CCAFC3A29176C3971F593489F86EDD23A4C02C1BFBDAE4144FCF34CF8`.

## Evidence boundary

This is a source/repository audit, not an experiment. It did not run the proposed patches, did not
rerun the project-reported 110-test suite, did not open formal profile/replay/evaluation data, and did
not produce any r2 Gate result. The report's code interpretations, counterexamples, repair plan, and
test contracts are review evidence or proposals. They are not new latency, mAP, coverage, calibration,
or training facts.

The report links a proposed patch at
`sandbox:/mnt/data/chronotransport_r2_registration_blocking_patch_proposal.md`. That artifact is not
available in this workspace or on the fixed GitHub tree. Every patch in it is explicitly marked
`NOT_EXECUTED_BY_REVIEWER`; it is therefore recorded only as an unavailable proposal, never as applied
or verified code.

## Absorption verdict

We accept the mandatory verdict:

```text
REVISE_IMPLEMENTATION_BEFORE_REGISTRATION
```

Route A (register and run the current implementation) remains rejected. Route B (complete the
spec-preserving implementation, verify it, and obtain a second independent implementation approval)
is selected. Route C (permanent freeze) remains the mandatory fallback if the minimal patch set cannot
pass re-audit or a later hard science gate fails. This does not invalidate the approved r2 specification
and does not erase the historical `92029ea` negative result.

## Seven previously known registration blockers — accepted

1. Gate 3/Gate 4 adjudicators and launchers are absent.
2. Executable formal r2 Stage B, Stage C, and matched-dense workflows are absent; the old runner uses
   the legacy six schedules and old split/training procedure.
3. Transactional AMP overflow backoff, mutable-state snapshot/restore, identical-batch retry, success-
   only counter advancement, and the retry cap are absent.
4. The scheduler does not require registered `B*` plus exact measured requested-action cost.
5. A full-stack invocation-level profiler and provenance-complete raw cost artifact are absent.
6. Registration accepts caller-authored identity and does not rederive the complete chain from bytes.
7. Gate-1 population, library, budget, bootstrap, and input artifacts remain caller-controlled rather
   than registration-bound.

These seven findings agree with the preceding independent audit. The project-reported 110 tests remain
valid evidence for their tested subset only and cannot close any of these workflow gaps.

## Two newly established P0 blockers — accepted

### P0-A — r2 config overlay is at the wrong nesting level

The r2 overlay writes `model.backbone.chronotransport`, while the actual VideoMAE adapter runtime is
under `model.backbone.backbone.chronotransport`. Until resolved-config/build tests prove otherwise,
the overlay may either fail construction or leave legacy inner values such as `max_cache_age=8`. The
latter would repair long HOLD/TRANSPORT schedules, change executed action identity, and invalidate
formal samples. No r2 profile or Gate input may be generated from this configuration.

### P0-B — Gate-3 conformal uses the wrong independent unit

The existing helper flattens `window×candidate` residuals. The r2 specification requires one score per
calibration window: first take the maximum residual over all 16 candidates, then take rank 28 over the
30 window maxima. The review's counterexample (27 all-zero windows and three windows with one residual
of 100) correctly gives `q_conf=100`; flattening 480 rows incorrectly gives zero. The flattened helper
must be unreachable from r2 Gate 3 because it can cause undercoverage and a false PASS.

## Additional accepted implementation findings

- The current protocol helpers do not yet form an executable, immutable, label-free one-window manifest
  for exactly 200 videos with the complete media/index/config/data identities. The formal path must not
  use GT-aware `random_trunc`.
- The legacy Stage-B summary pools rows for Spearman and treats degeneracy as zero. r2 requires one rho
  for each complete 16-candidate `seed×window` vector and fail-closed behavior when ranks are degenerate.
- Stage-C exposure generation lacks the 525-per-candidate validator, per-seed/combined hashes, resume
  prefix validation, and CT/matched-dense shadow-ledger binding.
- Requested and executed costs must remain distinct. A repaired schedule's additive proxy cannot be
  recorded as the exact measured cost of the originally requested candidate.
- Profiler summaries must not satisfy completeness by inserting unmeasured `count=0, p50=0, p95=0`
  stages. Formal full-stack evidence is based on directly measured invocation `total_ms` rows.
- The registered candidate library must fix exact names, order, hash, and tie behavior; arbitrary mapping
  order or extra candidate names cannot affect Gate-1 selection.
- Formal Stage C must bind object-identity optimizer ownership end to end; generic name-based grouping is
  not evidence of A/T/R exact-once membership.
- `transport.py` should use non-mutating `age.clamp(...)`, because in-place clamping can corrupt a caller
  tensor even if the current runtime happens to pass a temporary tensor.
- Paired replay still needs candidate-order permutation, materialized-pixel identity, resume-prefix, and
  loader-cursor tests.
- Dense fail-closed artifacts need the same requested/executed identity, cost-validity, and selection-
  exclusion fields as non-dense artifacts.
- Canonical JSON byte rules for NFC/non-ASCII data must be documented with golden vectors; the current
  `ensure_ascii=True` choice may stay if it is explicitly frozen.

## What is not absorbed as fact

- The unavailable sandbox patch proposal is not treated as present, applied, complete, or tested.
- `PATCH_BLOCKED_BY_MISSING_FACT` interfaces are not silently invented and are not implementation facts.
- The reviewer's source deductions about tensor shapes and semantics are not runtime observations.
- Project-reported tests are not relabeled as independently reproduced tests.
- The GitHub connector's blob equality is not relabeled as an independently recomputed specification
  SHA-256.
- Missing implementation surfaces do not reopen or rewrite the approved r2 scientific specification.
- No r2 experiment, Gate, deploy, latency, metric, or paper claim is created by this review.

## Mandatory repair order absorbed into the route

1. Fix the resolved inner-runtime config and window-level simultaneous conformal unit.
2. Build and validate the complete immutable 200/140/30/30 label-free window manifest.
3. Build invocation-level full-stack profiling and provenance-complete exact cost artifacts.
4. Bind scheduler feasibility and Gate 1 to the registered library, exact cost, `B*`, population, and
   fixed statistical constants.
5. Implement formal Stage B, registered Gate 2, and Gate 3 with exact successful-update, exposure,
   rank, per-window statistic, and hierarchical-bootstrap contracts.
6. Implement transactional Stage C and matched dense with 4,200 successful updates, 8,400 exposures,
   overflow retry, resume validation, and identical successful-batch hashes.
7. Implement Gate 4 on the official full-video population with six-order timing and the specified
   one-sided hierarchical confidence bounds.
8. Generate identity from immutable inputs, add deep validation, create all formal GPU1 launchers, and
   require `PRECHECK_ONLY=1`, clean-tree, exact HEAD, Slurm/GPU1, unlock-chain, and output-root checks.
9. Run remote controlled CPU/CUDA verification without opening formal Gate data, then obtain a fresh
   independent `APPROVE_IMPLEMENTATION_FOR_REGISTRATION` verdict.

## Current state and hard lock

```text
scientific specification: designed / approved
implementation: partial
tested: partial, project-reported subset only
registration readiness: NOT_READY
experiment_running: false
empirically_supported (r2): false
paper_ready: false
deploy: false
```

Until the second implementation review approves the repaired candidate, the following remain forbidden:
implementation commit `I`, registration commit `R`, formal profiling, Gate 1, new Stage-B seeds, Stage C,
and Gate 4. Any repair/fallback/hash/cost/update/resume/provenance violation is
`INVALID_IMPLEMENTATION`, not a science PASS or FAIL. Even if all four Gates later pass, `deploy=false`
and `paper=false` remain frozen until a separate result-to-claim decision.
