# DUCA-RIME Anti-Repetition Memory

Last updated: `2026-07-28`

These failures and naming mistakes must not be repeated.

1. Do not call this Online TAD. The task is offline temporal action detection.
2. Do not call a designed, implemented, tested, or running method
   `empirically_supported`.
3. Do not call four experimental phases four final models.
4. Do not use official-final GT, teacher outputs, cached raw predictions, or
   counterfactual ledgers in an inference decision.
5. Do not select a checkpoint, threshold, price, loss weight, or method variant
   on the official-final set.
6. Do not interpret a fixed-K checkpoint evaluated at several K values as
   mixed-K detector headroom. O1 requires the `U-mixed-K` checkpoint.
7. Do not use a mutable global optimizer-step schedule for mixed-K exposure.
   Use the stateless `(epoch + sample_index) mod 60` schedule.
8. Do not count AMP retries as successful detector updates. Every formal train
   arm requires exactly 6000 successful updates.
9. Do not pad variable-K execution to Kmax and then claim physical cost savings.
10. Do not compare variable-K RIME cost only with `U-fixed`; use exact
    `U-same-K` replay for realized-cost matching.
11. Do not read only legacy `effective_budget` in the profiler; the RIME ledger
    uses `effective_k`.
12. Do not claim K192 is a dynamic-budget panel. It is the minimum-budget,
    learned-position stress panel with forced K=192.
13. Do not treat K=384 historical 90-round or K=192 30+60 results as fair
    6000-update paper results.
14. Do not let wrapper/native path differences masquerade as selector gains.
15. Do not use single-frame exchange alone to justify a global allocation
    policy; multi-scale counterfactual and null controls remain required.
16. Do not launch Phase 4 unless the Phase-3 development receipt explicitly
    authorizes it.
17. Do not accept a Phase-4 cell unless authorization, Phase-2 receipt, budget
    protocol path/hash, checkpoint audit, and terminal identity form one exact
    provenance chain.
18. Do not write `paper_ready` merely because the matrix was submitted or
    completed; the registered statistical and cost gates must all pass.
19. Do not rely on Slurm's default `/bin/sh` for the environment bootstrap;
    every held job must explicitly enter `/bin/bash -lc`.
20. Do not apply train-batch or saved-prediction assertions to the paired
    Phase-1 cost-only profilers. Enforce their relevant contract instead: test
    batch one, zero workers, no accuracy output/claim, and identical checkpoint
    identity.
21. Do not set `dataset.val=None` without also setting
    `workflow.seal_eval_dataloaders_during_training=True`; otherwise the generic
    trainer attempts to build a validation loader from `None` before the first
    optimizer update.
22. Do not leave reentrant VideoMAE gradient checkpointing enabled in the
    single-GPU DDP dense TriDet reference. It marks a shared parameter ready
    twice on the first backward pass; bind `with_cp=False` in config and
    precheck, matching the dense ActionFormer execution contract.
