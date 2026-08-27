# DUCA semantic-indirect dynamic budget cycle2 — terminal receipt

- Scientific question: whether a low-cost scout can predict 0/1 actionness and
  boundary importance, then deterministically acquire timestamp-preserving
  frames under a dynamic per-video/window K that genuinely reduces heavy
  VideoMAE/ActionFormer computation.
- Frozen clean base: `6125654b946cc30c614428ce1141f1903b015867`
- Terminal clean candidate: `d80022e963a8ad21d390c785cbd8a4c23f41484a`
- Worktree: `C:/Users/skywalker/.codex/worktrees/duca-semantic-cycle2-20260817`
- Evidence class: static code/compile review only; no data, GPU, remote job,
  training, evaluation, mAP, or full-stack cost result.

## Terminal decision

`BLOCKED_PRE_RUN / CYCLE2_IMPLEMENTATION_PACKAGE_CLOSED`.

The three permitted claim-preserving implementation corrections are exhausted.
The dynamic-K heavy-path focused fix received an independent static pass, but
the final review found three remaining execution-admission defects: the shared
dense arm is a placeholder rather than an actual runtime builder; shared
detector/loss/NMS/evaluator/update/seed fields are not bound to the real
training objects; and the checkpoint lifecycle lacks latest-three, milestone,
final, and final-EMA artifacts. Windows PyTorch additionally fails to load
`c10.dll`, so the relevant runtime fixture could not be executed.

No fourth Builder repair, Evaluator PRE_RUN, or N16R4 experiment is admissible
for this candidate. The scientific route is not empirically refuted; fixed K
remains a control/fallback and no effect-size claim is licensed.

## DSH transport

The final correct-parameter DSH job `j-w89vrp` exited `1` with zero output
lines and no session/header/first reasoning line/completed turn. It is neither
PASS nor FAIL. Related transport receipts are in
`docs/dsh/DUCA_DSH_CYCLE2_FINAL_REVIEW_RECEIPT-2026-08-18.md` and
`docs/dsh/DUCA_DSH_CYCLE2_FINAL_RETRY_RECEIPT-2026-08-18.md`.

## Handoff

- next_owner: Coordinator terminal hold
- next_action: preserve the blocker; do not run an experiment from this
  candidate.
- dependency: a new user-authorized clean implementation cycle, if the project
  is to continue.
- expected_return_at: none
- single_recovery: none

