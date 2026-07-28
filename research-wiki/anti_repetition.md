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
23. Do not label a short window as effective K when the heavy backbone still
    consumes the configured fixed K. The `d9d454cd` Phase-1 K384 control exposed
    `effective_k=231` with `backbone_input_k=padded_k=384`; either dispatch a
    real K231-compatible execution or account for the actual K384 cost.
24. Do not compare Phase-1 mAP on 20 videos drawn from the checkpoint's
    `training` domain with the upstream 69.03 official validation mAP. A
    deterministic evaluation role is not held out from a historical checkpoint
    unless its training manifest proves exclusion; use split-aware retraining or
    cross-fitting before interpreting detector headroom.
25. Do not use a small subset, training-domain result, intermediate epoch,
    smoke run, single-seed pilot, partial matrix, or missing-receipt artifact to
    explain model performance to the user. These are engineering status only
    and cannot enter the paper.
26. Do not fill an empirical evidence gap with a suggestive partial number.
    State that no paper-admissible conclusion exists, or give a complete
    theoretical analysis with explicit assumptions and limitations.
27. Do not treat an external review's linked sandbox patch, hash, or reported
    synthetic test count as repository implementation or verification when the
    artifact is not present and independently inspected.
28. Do not describe the four-stage DAG as generically “running” after Phase 1
    has failed closed. Report each component separately: dense references may
    run while Phase 2/3/4 remain dependency-blocked.
29. Do not fit and evaluate edge hazards on the same videos or infer unique
    edge causality from a collinear path-incidence regression. Require
    video-level cross-fitting, balanced perturbations, rank/conditioning, and
    bootstrap stability.
30. Do not claim cross-K gap-mass normalization until path-dependent source,
    internal, and sink gaps are explicitly nonnegative and normalized to sum to
    one.
31. Do not claim `O(KTW)` merely because the graph is sparse in common cases.
    Enforce a registered predecessor/successor span cap and fail closed on
    overflow; otherwise report the possible `O(KT^2)` degeneration.
32. Do not let gap length serve simultaneously as an unexamined learned-risk
    explanation and a direct regularizer. Include gap-only, residualized-risk,
    risk-off, and feasibility-preserving edge-shuffle controls.
33. Do not call the current `AdapTok-TAD` arm an official reproduction. Use
    `AdapTok-inspired TAD budget allocation baseline` unless the official
    tokenizer, scorer, latent allocation, and training path are actually
    transferred and audited.
34. Do not implement or integrate a learned CBCG edge head before the repaired
    v1 execution receipt, held-out same-K oracle, target learnability,
    calibration, causal, and full-stack cost gates pass.
35. Do not call the current controller output “video-level risk.” It summarizes
    one 768-candidate training crop or inference window and predicts a
    window-level per-K risk panel; no cross-window video aggregation exists.
36. Do not treat windows from one video as independent statistical units.
    Cross-fit, calibration, paired tests, and bootstrap must keep all windows
    from the same video in one group.
37. Do not cost-match only a video-average K. Replay the exact
    `(video_id, window_start_frame) -> effective_k` map, then aggregate actual
    heavy frames, latency, and energy per video, including sliding overlap.
38. Do not call a dense reference complete because its log says
    `Training Over` or because `epoch_59.pth` exists. Terminal EMA compaction,
    evaluation, checkpoint binding, and the registered receipt must all pass.
39. Do not reuse raw checkpoints from a failed immutable root by silently
    rerunning post-processing in place. Any salvage must be a new hash-bound
    transaction that names the original job IDs and source checkpoint hashes.
40. Do not leave `DependencyNeverSatisfied` jobs described as active
    experiments. Their downstream evidence does not exist.
41. Do not confuse one whole-video total budget `B_v` with one K copied to every
    768-candidate window. The intended hierarchy predicts a total quota, jointly
    allocates `K_vw`, and still executes AdaTAD per window.
42. Do not claim that the current repository already **executes** learned
    video-level allocation. H-RIME's deterministic budget/allocation/replay
    contracts now exist, but the dataset path, learned planner, real shared scan,
    detector replay and full ledger are not connected.
43. Do not predict or compare a raw duration-blind video budget. Normalize the
    budget by valid duration/window opportunity, freeze its price on
    training-only video groups, and enforce the exact per-video hard constraint.
44. Do not count unique selected frames across overlapping windows as actual
    heavy compute unless a real cross-window feature cache executes them once.
    Without that cache, formal cost is the sum of per-window heavy executions.
45. Do not implement a hierarchical video planner before a held-out,
    same-total-heavy-cost oracle establishes cross-window allocation headroom
    over uniform and independent-window policies.
46. Do not treat the numeric gates suggested by `U-PRO-HRIME-1` as empirical
    facts. Freeze the primary endpoint, materiality/noninferiority margins,
    multiplicity and calibration thresholds from training/calibration roles in
    a hash-bound manifest before reading the complete development matrix.
47. Do not require equality to an arithmetically unreachable raw video cap.
    Project to a reachable sum of deduplicated effective-K choices, then report
    raw cap, reachable target, realized total, projection unused budget and
    solver unused budget separately.
48. Do not compare policies by nominal K or requested budget when short windows
    alias several requests to one effective K. Cost matching and allocation use
    the effective-K assignment.
49. Do not call a budget-density panel duration-normalized merely because it is
    expressed as rho. Report realized density, window-valid-length opportunity,
    unreachable-cap rate and unused-budget distribution.
50. Do not promote an additive window-loss allocation oracle without replaying
    the selected assignments through the full prediction merge, official NMS
    and evaluator. Audit surrogate rank/sign agreement and worst error.
51. Do not infer true risk monotonicity from a monotone neural parameterization.
    Report Brier, reliability, risk-coverage, violations and pre-registered
    worst-group calibration in addition to ECE.
52. Do not use “one endpoint significant and two positive” as an admission rule.
    Pre-register one primary endpoint plus multiplicity and noninferiority
    guardrails.
53. Do not present three seeds as a precise normal-variance estimate. Report
    every seed and use video-cluster bootstrap for the primary uncertainty.
54. Do not allow floating-point MCKP ties to depend on device/library accident.
    Freeze score quantization, dtype, solver version, deterministic tie-break and
    assignment hash.
55. Do not claim energy savings for the current transaction. Scheduler energy
    fields are zero/unavailable and no trusted GPU power monitor was active.
56. Do not copy H-RIME sandbox patches or reported test counts from the external
    review: those artifacts are absent. Implement and verify the registered
    design in this repository.
57. Do not pass fractional or non-quantum-aligned costs into H-RIME and let
    integer conversion silently change the problem. Raw budgets and every
    effective K must be exact integers; effective K is divisible by 16.
58. Do not emit an H-RIME replay from feasible aliases or an MCKP result that is
    not hash-bound to the exact video budget plan. Recompute the feasible-set
    and plan-input hashes before generating replay rows.
59. Do not call a grouping/dispatch **plan** a measured shared-scan or heavy
    execution saving. A shared scan counts as implemented only when the runtime
    executes it once per complete video and the full-stack ledger proves reuse.
60. Do not assume a top-level Bash `ERR` trap covers `sbatch` failures inside a
    submitter function. Transactional held-DAG submitters must enable `errtrace`
    and a failed partial submission must be canceled before any job is released.
