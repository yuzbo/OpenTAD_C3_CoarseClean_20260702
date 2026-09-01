# CRITIC_PJST_D1_CYCLE3_FOCUSED_REVIEW-v001

## Binding

- You are the sole independent Critic for the frozen PJST-D1 Cycle-3 focused-correction snapshot.
- Review only `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle3-builder-20260826` at exact clean commit `5a4fef786f71f191cd9bb4fe3cb32334a0ba61b5`.
- Clean scientific parent: `b2ccfccab5b4912b59954afcc9b0364955327f7c`; prior Cycle-3 candidate: `a367063f58746a87314e60cedcd7165bf992cc0f`.
- Read-only code/config/test/launcher review. Do not edit, commit, access data/browser/GPU/Slurm/remote hosts, train, infer, evaluate, or produce metrics.
- Write only the durable review receipt requested below outside the reviewed Git worktree.

Read fully:

1. Worktree `AGENTS.md`, `RTK.md`, `research-wiki/query_pack.md`, `research-wiki/anti_repetition.md`.
2. Accepted scientific contract: `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/pro-reviews/runs/duca-pjst-derivative-causal-freeze-v002/PRO_DUCA_PJST_DERIVATIVE_CAUSAL_FREEZE-v002.md`.
3. Builder task and return:
   - `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/role-messages/BUILDER_PJST_D1_CYCLE3_FOCUSED_CORRECTION-v001.md`
   - `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/role-returns/BUILDER_PJST_D1_CYCLE3_FOCUSED_CORRECTION-v001.md`
4. Exact diff `a367063f..5a4fef78` and all changed implementation/config/test/validator/launcher files.

## Required review

Attack correctness and design fidelity, not novelty. Verify with file:line and exact symbols:

1. **Real runtime reachability:** selector-updated metadata reaches both train/test backbone calls through actual production detector classes, including ActionFormer overrides, without breaking other backbone callers.
2. **Metadata identity:** ON extracts the real H65 selected physical positions, dense valid length, and contiguous prefix mask; `[B,24,8] -> [B*24,8]` ordering matches flattened frames for B>1. OFF performs no PJST helper/cast/division/kwargs path.
3. **Frozen acquisition:** matched arms actually replay/freeze the learned H65 semantic nonuniform K384 acquisition rather than silently selecting exact-uniform frames or allowing selector drift. Prove the resolved configs and all relevant loss/update schedules, not just literal child lines.
4. **Derivative transform:** exact-uniform samples bypass before float arithmetic; invalid/partial pairs inside irregular rows stay byte-identical; valid pairs implement exactly the accepted formula and preserve dtype/gradient; no support-weighted appearance, learned gate, extra PatchEmbed, parameter or state key.
5. **Checkpointing:** ON chunk-dim 0 slices every metadata tensor with the exact frame chunk; ON dim 2 fails before execution; OFF base behavior stays compatible; callback arguments match `cp.checkpoint` and no full-batch metadata leaks through closure capture.
6. **Physical decode:** selected-to-physical remap executes exactly once on raw segments before threshold/filter/top-k/IoU/NMS and still fail-closes on missing/wrong axis metadata. No score/NMS/evaluator semantics changed.
7. **Config/launcher/validator:** same Stage-1 epoch-29 checkpoint and selector exposure, seed 3407, Stage-2 60 epochs/6000 successful updates, every-five-epoch resumable state, fixed final/final-EMA rule, canonical THUMOS14/annotation/category/pretrain paths, N16R4 env/distributed bindings, distinct clean roots, PRECHECK_ONLY and real `tools/train.py` execution. The validator must prove production contracts rather than merely grep strings.
8. **Tests:** assess whether the focused tests exercise production call paths and distinguish the old failures. Do not accept signature-only or source-text-only checks as proof of runtime behavior. Local Windows skip is not a pass; state exactly what must run on N16R4 before PRE_RUN.
9. **Scope:** inspect the four-line `actionformer.py` change and every other diff for unintended model/science changes. Compare state-dict keys and parameter counts where feasible without data.

Run only useful no-data checks whose failure would change the verdict: `py_compile`, validator, launcher syntax, focused pytest if the environment can import Torch, and `git diff --check`. Never label a skipped test as passed.

## Verdict and durable return

Return exactly one of:

- `PJST_D1_FOCUSED_STATIC_PASS`: implementation is faithful and sufficiently complete to hand to Evaluator for N16R4 Linux focused tests/PRE_RUN; list remaining evaluator-only gates.
- `BLOCKED_PRE_RUN`: identify every deterministic blocking defect, its evidence, the smallest claim-preserving fix, and whether defects are equivalent to a previously reported failure.

Write `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/role-returns/CRITIC_PJST_D1_CYCLE3_FOCUSED_REVIEW-v001.md` with exact reviewed commit/worktree cleanliness, findings, commands/results, evidence class, verdict, and `next_owner`. Do not claim efficacy. No correction is authorized by this review itself.
