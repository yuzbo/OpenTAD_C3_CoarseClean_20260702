---
updated: 2026-07-06
status: active
scope: Absorbed Pro review gate for unavailable DUCA-TAD commit 5f87baf
out-of-scope: Line-by-line source-code findings, detector mAP claims, or implementation PASS/WARN/FAIL
---

# DUCA-TAD 5f87baf Review-Blocked Absorption

Raw record: `docs/methods/reviews/2026-07-06-duca-5f87baf-review-blocked-raw.txt`

Requested target commit: `5f87baf`

Requested target branch: `codex/gas-vt-stage23-detector-aware-20260706`

## Core Verdict

This is a valid external **review gate**, but it is not a valid **line-by-line code review** of the target implementation.

The reviewer could access the public repository, but the requested commit and branch were not visible on GitHub. Because the target source could not be opened, the reviewer correctly refused to infer implementation quality from adjacent branches.

Machine-readable status from the review:

```text
DUCA_TAD_REVIEW_GATE=HOLD_TARGET_COMMIT_AND_BRANCH_NOT_VISIBLE
TARGET_COMMIT_VISIBLE=NO
TARGET_BRANCH_VISIBLE=NO
LINE_BY_LINE_REVIEW_DONE=NO
PRECHECK_ONLY_ALLOWED=NO
SHORT_SMOKE_ALLOWED=NO
FULL_TRAINING_ALLOWED=NO
PAPER_CLAIM_ALLOWED=NO
REQUEST=RE_PUSH_FIXED_COMMIT_OR_PROVIDE_DIFF_BUNDLE
```

## What This Review Proves

- The public repository is reachable.
- The requested fixed commit `5f87baf` was not reachable by the external reviewer.
- The requested branch `codex/gas-vt-stage23-detector-aware-20260706` was not reachable by the external reviewer.
- A GitHub-only review cannot validate Stage1/2/3/4 implementation completeness until the exact branch or commit is pushed and visible.
- No paper claim, mAP claim, end-to-end claim, or PRECHECK/full-training release should be justified by this review alone.

## What This Review Does Not Prove

- It does not prove that Stage2/Stage3/Stage4 are missing locally.
- It does not prove that the local implementation is correct.
- It does not inspect `detector_teacher_utility.py`, detector-aware selector code, TrueTime code, Stage4 validators, launchers, or tests line by line.
- It does not find source-level bugs in the target commit.
- It does not validate or invalidate any detector mAP result.

## Absorbed P0 Gate

Before asking for another external code review:

1. Push the intended branch or fixed commit to GitHub.
2. Verify the commit URL is externally reachable.
3. Provide the exact branch, full SHA, and repository URL.
4. If GitHub visibility cannot be guaranteed, provide a fixed zip or diff bundle.
5. Keep result-to-claim gates fail-closed until a visible fixed-source review and real mAP evidence exist.

## Checklist For Next Pro Review

The next review must explicitly confirm:

- `TARGET_COMMIT_VISIBLE=YES`
- `TARGET_BRANCH_VISIBLE=YES`
- `LINE_BY_LINE_REVIEW_DONE=YES`
- target commit full SHA
- inspected file list
- P0/P1/P2 findings with file and line references
- whether Stage2 teacher utility is train-only and provenance-locked
- whether val/test paths forbid GT, teacher utility, raw prediction cache, and dense teacher payload
- whether selector inputs are deployable observables only
- whether fixed budgets are truly fixed and dynamic budget is calibrated across videos
- whether selected positions remain local dense-time indices, not selected-rank indices
- whether TrueTime geometry preserves physical time for assignment, decode, and NMS
- whether ST hard selector receives detector-loss gradients in a real detector forward path
- whether Stage4 rejects placeholders, fake mAP, missing SHA, and precheck-only evidence

## Claim Policy After Absorption

Allowed wording:

- "The external reviewer blocked the review because the requested commit and branch were not visible."
- "This is a visibility/provenance failure, not a source-code PASS or FAIL."
- "A real code review must be repeated after pushing a fixed, reachable commit."

Forbidden wording:

- "The external reviewer found no code issues."
- "The external review passed."
- "Stage2/3/4 are externally verified."
- "Full training or paper claims are unlocked by this review."

## Immediate Action

Treat this as a GitHub synchronization problem and review-provenance problem. The technical implementation should be judged by local tests, remote prechecks, detector mAP experiments, and a repeat external review against a reachable fixed commit.
