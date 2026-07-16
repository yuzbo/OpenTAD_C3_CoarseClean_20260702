---
type: source_record
title: "ChronoTransport r1 GitHub-visible Pro review absorption"
source_attachment_sha256: "07A5B4B519E64A39D7F84CE862F0E56117BFF2DB62206B6AE24BDD66768B19FE"
repository_archive_sha256: "1B7C2A377ADA97B5F6165E699EEA6B0DF6023794A66261DED047117D82F05EB1"
reviewed_repository_commit: "1f5f7254a390f183121e6c4b7cebcebd2f2954d1"
reviewed_spec_commit: "02199f8eb34fc0e34e342afcc357269df457d091"
verdict: REVISE_SPEC_BEFORE_PLAN
status: absorbed_and_locally_verified
updated: 2026-07-12
---

# ChronoTransport r1 GitHub-visible Pro Review Absorption

## Source and evidence boundary

The complete 1,924-line Pro response is archived in
[`2026-07-12-chronotransport-r1-pro-github-review-response.md`](2026-07-12-chronotransport-r1-pro-github-review-response.md).
The external attachment and the repository archive have different byte hashes
because the archive normalizes line endings. The review inspected the public,
commit-pinned repository at `1f5f7254a390f183121e6c4b7cebcebd2f2954d1`.

This record is a review absorption and static verification. It is not a new
GPU result and does not upgrade historical P3 numbers to independently verified
experiment facts.

## Executive absorption

The review verdict is `REVISE_SPEC_BEFORE_PLAN`, with final action
`GO_TO_SPEC_REVISION_ONLY`. It confirms that `02199f8` correctly froze:

- the pre-adapter heavy-cache boundary;
- all-row original AdaTAD temporal-adapter execution;
- `hard_cache_validity_age=47` versus transport embedding cap 8;
- requested/executed action and cost separation.

It finds ten patch-ready amendments are still required before implementation:

1. live current-row tensor versus detached recurrent-cache aliases;
2. one label-free hash-frozen 768-point window per train video;
3. exact-count group-wise `motion_topk` and random controls;
4. block-rotated Stage-B exposure;
5. removal of the oracle-shuffle Gate-1 tautology;
6. Gate-3 minimum support, all-fit constant replay and window-outer statistics;
7. executable Stage-C loss-specific autograd and AMP retry;
8. clustered one-sided Gate-4 inference and balanced timing order;
9. immutable pre-Gate1 two-commit registration;
10. population-scoped claim flags with `deploy=false` and `paper=false`.

## Independent verification performed locally

The following review claims were independently recomputed or checked against
the pinned repository and fixed official upstream sources:

- The old offsets 0/4/8 preserve
  `candidate mod 4 = canonical video position mod 4` for all three seeds.
- For `candidate=(p+5*b+offset) mod 16`, each seed exposes twelve candidates
  nine times and four candidates eight times.
- The three-seed total is candidate 0--3 at 27 exposures and 4--15 at 26,
  totaling 420.
- The exact tails are `[8..15,0..3]`, `[12..15,0..7]` and `[0..11]`.
- In the first 128 windows, each candidate appears twice in each `p mod 4`
  class per seed, and six times after aggregating three seeds.
- Stage C has 8,400 window exposures per seed and exactly 525 exposures per
  candidate.
- The conformal ranks are 28 for `n=30` and 127 for `n=140`.
- A one-sided 95% exact lower bound for 18/18 coverage is approximately
  0.846682, so requiring its lower bound to reach 0.85 would be nearly
  impossible at the minimum legal support.
- Official OpenTAD `random_trunc` uses GT intersection in crop selection,
  while short frame-index vectors are padded with `numpy.pad(mode="edge")`.
- Official AdaTAD applies attention residual, MLP residual and then its
  full-temporal-axis adapter; the adapter Conv1d has symmetric padding and no
  causal mask.
- The historical local runtime only writes adapted values to effective
  RECOMPUTE rows, confirming it is a pre-r1 implementation rather than an r1
  implementation result.

## Points not accepted verbatim

The main verdict and protocol direction are accepted, but the response is not
copied blindly. Three remaining textual ambiguities must be closed in the r2
specification:

1. The phrase "canonical SHA-256 ordering with seed 3407" does not define
   exact input bytes, field separators, text encoding or tie-breaking. The r2
   specification must define an ASCII/UTF-8 domain-separated digest and raw-byte
   lexicographic ordering.
2. Gate 3 reports `min_seed_selected(upper-regret)` without first naming the
   coverage margin. The r2 specification must define
   `coverage_margin=upper-regret` and how windows with no non-dense selection
   are handled.
3. Gate-4 mAP bootstrap does not explicitly say whether resampled seed metrics
   are concatenated or averaged. The r2 specification must reconstruct mAP for
   every resampled seed on the resampled video multiset and use their arithmetic
   mean as the replicate statistic.

These are closure fixes, not new modeling choices, seeds, losses, candidates or
thresholds.

## Current decision

ChronoTransport remains `spec_revision_in_progress`. No implementation,
profiling, Gate 1, new seed or Stage C is unlocked until:

1. all ten amendments plus the three closure fixes are written into a new spec;
2. the exact spec SHA-256 is computed;
3. a spec-only reviewer returns `APPROVE_SPEC_FOR_PLAN`.

After approval, implementation must follow TDD and formal execution must obey
the sequential Gate-1-to-Gate-4 stop chain.
