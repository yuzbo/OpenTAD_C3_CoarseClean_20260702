---
type: external_review_absorption
source_file: research-wiki/sources/2026-07-16-chronotransport-r2-github-pro-snapshot-gate-92a18be.md
source_sha256: 990E84F1D09116257D684090163BACB3F579ACA7290BADCB4D9FC6CFDA151FD1
review_sha: 92a18bec2f5f247446083a8eb50fe889f367c23e
disposition: accepted_snapshot_gate_and_approved_equivalent_certificate_prompt_repair
updated: 2026-07-16
---

# Absorption: GitHub snapshot gate at `92a18be`

## Accepted facts

- The reviewer fresh-resolved the branch to exact SHA
  `92a18bec2f5f247446083a8eb50fe889f367c23e`.
- It independently established that the SHA differs from the three anchors, that the implementation-floor
  compare is ahead 1/behind 0 with merge base `6c3606c`, that `^1` is `6c3606c`, and that `^2` does not
  exist through the available revision probe.
- It independently enumerated the exact eight changed paths after the implementation floor; all are audit or
  research-memory documents. It found no post-floor production, test, config, launcher or registration change.
- Its available normalized GitHub interface did not expose `tree.sha` or separate Git-object author and
  committer timestamps. Under the exact prompt then in force, returning only `GITHUB_SNAPSHOT_INCOMPLETE`
  was correct fail-closed behavior.

## Evidence boundary

This response is not an implementation review and is not a registration verdict. It did not check the spec
hash, source classification, production code, tests, Stage C, Gate 4 or Slurm contracts. It neither approves
nor rejects candidate `6c3606c`; W7 remains open, registration remains `NOT_READY`, and E0--E5 remain locked.

The missing fields are a reviewer-tool capability gap rather than evidence that the Git object or repository
bytes are inconsistent. Local/GitHub API observations of the tree SHA cannot be substituted into this review
as independent reviewer evidence.

## Approved prompt repair (not a review verdict)

The user approved this repair on 2026-07-16. Keep the full Git Data commit object as the preferred certificate.
If the reviewer explicitly proves that its
GitHub interface does not expose that object, accept an equivalent content-addressed certificate only when all
of the following are independently established: fresh immutable SHA resolution; strict compares to every
anchor; exact `^1` and negative `^2`; exact implementation-floor changed-path list; all subsequent reads pinned
to the SHA; and exact blob/content verification for every mandatory file. Missing any one of those conditions
still returns `GITHUB_SNAPSHOT_INCOMPLETE`.

Separate author/committer timestamps are informational and do not protect implementation bytes. Exact tree
SHA is useful but is redundant for this audit when the reviewer proves the immutable commit identity, parent
structure, complete post-floor diff scope and every audited file's SHA-pinned content. This proposed fallback
removes a tool-specific false stop without allowing project-reported metadata or moving-branch reads.

No code, I/R, PRECHECK, Gate, training or experiment is authorized by this absorption.
