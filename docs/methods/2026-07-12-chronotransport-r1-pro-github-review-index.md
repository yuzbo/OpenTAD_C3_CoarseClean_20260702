# ChronoTransport r1 Pro GitHub Review Index

## Purpose

This page is the public, repository-native entry point for an independent Pro
review of the ChronoTransport `CT-P3R-3S-r1` written specification. It exists
because the reviewer cannot read the local `E:\` workspace.

The review must be performed from one immutable Git commit. Open this page from
a commit-pinned GitHub `blob/<full-sha>/...` URL and keep the same full SHA for
every repository link. Do not silently switch to the default branch during the
review.

## Current decision state

- Written specification commit: `02199f8`
- Written specification SHA-256:
  `871420261BD1C19CC515218A6016A91ED7D553B73740AB41C2E02AA7F96609F9`
- Current review status: `REVISE_SPEC_BEFORE_PLAN`
- `writing-plans`, disputed implementation, profiling, new seeds, Gate 1 and
  Stage C remain locked.
- This package contains source and written-review evidence only. It does not
  contain datasets, checkpoints, GPU logs or new behavioral results.

## Mandatory reading order

### Round 1: sealed review

Read the current specification and repository code before reading any previous
reviewer conclusion:

1. [Current r1 written specification](../superpowers/specs/2026-07-11-chronotransport-ct-p3r-3s-r1-design.md)
2. [ChronoTransport model package](../../opentad/models/chronotransport/)
3. [VideoMAE/AdaTAD integration surface](../../opentad/models/backbones/vit_adapter.py)
4. [Anchor-free head state surface](../../opentad/models/dense_heads/anchor_free_head.py)
5. [Optimizer construction](../../opentad/cores/optimizer.py)
6. [ChronoTransport configs](../../configs/adatad/thumos/)
7. [ChronoTransport tests](../../tests/)
8. [ChronoTransport runners and validators](../../tools/bata/)

Form a provisional verdict before proceeding to Round 2.

### Round 2: adversarial comparison

Read the review records in this order:

1. [Original Pro review](../../research-wiki/sources/2026-07-11-chronotransport-ct-p3r-3s-pro-review-raw.md)
2. [Structured absorption of the original Pro review](../../research-wiki/sources/2026-07-11-chronotransport-ct-p3r-3s-pro-review-absorption.md)
3. [Local source audit](../../research-wiki/sources/2026-07-11-chronotransport-r1-local-source-audit.md)
4. [First blank-context independent review](../../research-wiki/sources/2026-07-11-chronotransport-r1-independent-agent-review.md)
5. [Second blank-context review of commit 02199f8](../../research-wiki/sources/2026-07-12-chronotransport-r1-spec-independent-agent-review.md)

The previous reviews are arguments to verify, not ground truth.

## Integrity manifest

| Artifact | SHA-256 |
| --- | --- |
| r1 written specification | `871420261BD1C19CC515218A6016A91ED7D553B73740AB41C2E02AA7F96609F9` |
| original Pro review repository copy | `E1ACD208DB8B04E55540FB62A7FA0D772D86021AB9C70EE6E8C7645E4B6959FC` |
| Pro review absorption | `7D95B82322F6730578021FC299E134881B5FF12E8461CB3274FC74C1F3589AF7` |
| local source audit | `54BAEC2E271FA1B14C24E2B64C6F4851DA148E2E41F702F916B4E828F1FE79CF` |
| first independent review | `E03CC35DA82770B689E15984065F28D9318651E3DC2E7FCC5235B8B245D7462F` |
| second independent review | `44CB6CAD040A985DC4125E6CFA128F0195A08E3CA6A6F4474CEA5BEFDF84670E` |

The original external attachment was separately registered with SHA-256
`E7971A22044B384092B833A1137F8EC0B543B504D271078CBCB4198F96D35CAF`.
That attachment hash and the repository-copy hash differ because the repository
copy is a Markdown archival representation; neither should be substituted for
the other.

## Fixed upstream references

- OpenTAD: `sming256/OpenTAD` at
  `1aa8ca4ac5e846b1e8ff69298dd6607121a01589`
- AdaTAD: `sming256/AdaTAD` at
  `25e06c720e450298ca5267fda6927f3591dcdfef`
- VideoMAE: `MCG-NJU/VideoMAE` at
  `14ef8d856287c94ef1f985fe30f958eb4ec2c55d`

Only primary-source upstream code may be used to support upstream semantic
claims.

## Evidence boundary

Every conclusion must be tagged as one of:

- `REPOSITORY_FACT`: directly supported by the pinned repository commit;
- `EXPERIMENT_FACT`: directly supported by a raw registered artifact;
- `REVIEWER_REPORT`: asserted by a prior reviewer but not independently proven;
- `INFERENCE`: reasoned from available evidence;
- `PROPOSAL`: a requested amendment or new design choice.

The historical Stage-B result remains negative and Stage C/P5 remains locked.
No review document upgrades a proposal, smoke test or static audit into an
experiment result.

## Review-only stop boundary

This GitHub synchronization authorizes reading and written review only. It does
not authorize code edits, remote access, GPU work, experiment launch, pushing a
reviewer patch, or changing gates after observing results.
