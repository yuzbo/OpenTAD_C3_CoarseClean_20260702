---
queue_name: BUILDER_DUCA_P0_CANONICAL_TRANSPORT-v001
parent_decision: PRO_P0_BLOCKER_DECISION-v001
stage: DRAFT/P0
fixed_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
workspace: C:/Users/skywalker/.codex/worktrees/duca-p0-63a726a4
execution: prohibited
---

# Builder P0 task

Use only the clean detached worktree above. Do not read, resume, cherry-pick,
or rely on any previous Builder/Critic worktree state. Make one static patch
attempt, touching at most ten production/test files, and do not execute Python,
pytest, training, evaluation, Slurm, remote commands, or access dataset/metric
artifacts.

Implement exactly the Pro P0 repair:

1. Introduce one canonical uniform generator for prefix-contiguous valid length
   `Tv`: `K_eff=min(384,16*floor(Tv/16))`; fail closed for `Tv<16`; for
   `j=0..K_eff-1`, use
   `floor((2*j*(Tv-1)+(K_eff-1))/(2*(K_eff-1)))`. No float linspace, banker
   rounding, tolerance repair, clipping, deduplication, padding/repetition, or
   second generator.
2. Replace the two incompatible current call sites so the clean uniform control
   and the selector's constant-density hard-forward specialization are
   bit-identical to this generator.
3. Add a detector-agnostic selected-to-physical raw-proposal adapter at the
   entry of each per-sample `SingleStageDetector.post_processing`, before any
   filtering, top-k, IoU, or unchanged NMS. Preserve `selected_q` (`[0,K_eff]`)
   and `physical_dense` (`[0,Tv]`) tags, fail closed for unknown/double mapping,
   and remove/neutralize the old post-NMS conversion without changing detector
   architecture, assignment, head, loss, NMS callable/config, evaluator, split,
   class map, or method route.
4. Write focused static fixture/test code and configuration-reconciliation
   artifacts, but do not run them.

Return one durable Markdown receipt with changed paths, contract mapping,
unrun test plan, uniform-spec JSON, config-diff JSON, patch identifier, explicit
no-execution attestation, and blockers. This is correctness preparation only;
it is not implementation evidence, performance evidence, or a result.
