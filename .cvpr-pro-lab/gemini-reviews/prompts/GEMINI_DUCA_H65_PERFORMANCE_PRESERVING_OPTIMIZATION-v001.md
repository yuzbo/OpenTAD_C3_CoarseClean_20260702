# Gemini independent consultation: preserve H65 performance while improving DUCA

You are an independent senior researcher in temporal action detection, efficient video understanding, curriculum learning, and continual/multi-condition adaptation. Conduct a read-only, code-grounded scientific consultation about how DUCA can preserve the main capability of the clean H65 model while pursuing further improvement, instead of repeatedly destroying H65 performance.

Respond in Chinese. State the exact model identity used. You are an external technical adviser, not the project's scientific decision-maker: Pro remains the sole scientific lead. Your report does not authorize code changes, held-out inspection, experiment submission, or route selection. Explicitly separate verified facts, mechanism hypotheses, and recommendations.

## Authoritative repository identities

- Repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- Current complete Wiki branch: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-wiki-complete-sync-20260831>
- Exact Wiki/evidence commit: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/aece35372ac8d4a37ceff4ec7f88a1aff0896fb6>
- Clean H65 scientific base: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/04c35a3b76897e6c1569eeede41ed3aecaf7f854>
- Current multi-budget implementation branch: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-h65-system-multibudget-exposure-v1-20260831>
- Exact current implementation commit: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/2b3b3243066a89e5a4be5acdb178c318fbeceac0>
- Frozen-detector whole-video falsifier: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>

Use the local worktrees when available and inspect exact code rather than relying only on prose:

- Current implementation worktree: `C:/Users/skywalker/.codex/worktrees/duca-h65-system-multibudget-exposure-v1-20260831/OpenTAD_C3_CoarseClean_20260702`
- Complete Wiki worktree: `C:/Users/skywalker/.codex/worktrees/duca-wiki-complete-sync-20260831/OpenTAD_C3_CoarseClean_20260702`

Read the following current files completely before drawing conclusions:

- `PAPER_PROGRESS.md`
- `research-wiki/query_pack.md`
- `research-wiki/anti_repetition.md`
- `research-wiki/decision_history.md`
- `research-wiki/experiments/duca-multi-budget-detector-adaptation.md`
- `research-wiki/sources/2026-08-31-pro-duca-full-data-identity-admission-v001.md`
- `research-wiki/sources/2026-08-31-duca-full-data-comparable-protocol-v001.md`
- `research-wiki/sources/2026-08-31-agy-gemini-duca-post-admission-optimization-v001.md`

Inspect the actual H65 and current implementation code, including the diff `04c35a3b...2b3b3243`, relevant configs, acquisition/selection, true-time mapping, variable-length VideoMAE execution, trainable-parameter groups, successful-update/EMA clocks, prediction sealing, and focused tests. List any requested file or commit that you could not inspect.

## Current admitted evidence

1. Dense AdaTAD is the 100%-heavy-observation reference at about 68.73 Avg-mAP. Clean H65 uses 384/768 heavy observations and achieves about 65.13 Avg-mAP and 43.31 mAP@0.7. Observation reduction is not yet a matched end-to-end latency claim.
2. H65 is not ordinary global Top-K. It uses action/transition evidence with budget-calibrated systematic sampling, a uniform-coverage floor, deterministic ordered indices, a 30-epoch Stage-1 and 60-epoch Stage-2 curriculum, controlled detector feedback, physical-time remapping, and a fixed 384-point detector grid.
3. No strictly comparable sparse DUCA candidate has yet exceeded clean H65. Training compression, continuous clips, protected end-to-end bridges, local-cell allocation, task-state native-tubelet coreset, Coverage-v1, and frozen-detector K256/K384/K512 transfer all failed their stated tests or evidence gates. Do not recommend repeating them under new names.
4. Historical values such as 65.3857 and 65.696 use different identities or detector geometry and cannot be promoted as a clean improvement over 65.13. PJST-D1 has an Avg-mAP point decrease and no completed paired interval. TrueTime has a small single-seed gain only within a lower-performing matched family.
5. The old frozen-detector action space produced zero passing candidates among 704 legal whole-video transfers. This stops that action space, not dynamic compute in general. Cross-budget representation mismatch remains a hypothesis, not an established root cause.
6. The currently authorized experiment compares K384-only Stage-2 exposure with K256/K384/K512 Stage-2 exposure on the complete 200-video training population, seeds 3407/3408/3409, 6,000 successful updates per arm, sealed predictions, a single complete 211-video held-out opening, and 10,000 paired whole-video bootstrap replicates.
7. The candidate keeps K384 exposure at 50%; the frozen occurrence plan is K256/K384/K512 = 1454/3000/1546, calibrated to match actual heavy observations. No budget embedding, distillation, new selector/controller, Gumbel, Mamba, Block Drop, DFT, or deployment optimization is present.
8. Exact commit `2b3b3243...` passed focused tests, independent Critic review, and PRE_RUN. Seed 3407 Control/Candidate training completed 6,000 successful updates each; seed 3408 is running and seed 3409 is pending. Held-out predictions, mAP, confidence intervals, and cost results do not yet exist and must remain unopened. Do not inspect remote training outputs or infer performance from training progress.

## Scientific question for this consultation

How should the project preserve the capabilities that make H65 the strongest sparse baseline while still allowing a new mechanism to improve performance or real compute efficiency? We want a preservation-first scientific strategy, not a broad redesign and not rescue tuning of the currently running frozen experiment.

## Required analysis

1. Reconstruct the clean H65 capability stack from code. Identify which components are most plausibly responsible for its performance and classify each as code-confirmed, evidence-supported, or merely hypothesized. Include training duration/update budget as a possible confound.
2. Explain, route by route, why previous changes plausibly damaged H65 performance. Do not turn an interpretation into a proven cause when no single-variable evidence exists.
3. Audit `04c35a3b...2b3b3243` for performance preservation. Distinguish:
   - initial K384 forward-path parity;
   - preservation after 6,000 multi-budget updates;
   - true heavy-compute matching;
   - trainable-parameter drift or catastrophic forgetting;
   - detector calibration, physical-time mapping, short-window collapse, batch homogeneity, RNG, optimizer and EMA clocks.
4. State what the current experiment can and cannot establish about H65 preservation. Assess whether the K384 safety gate of -0.2 pp is scientifically sufficient and what additional evidence is needed without reopening held-out tuning.
5. Produce a ranked preservation-first optimization map. For each candidate mechanism, specify:
   - the H65 capability it protects;
   - the one scientific variable it changes;
   - a falsifiable prediction;
   - the matched control;
   - the smallest full-data experiment that could decide it;
   - the main confound and stop rule.
   Do not combine multiple mechanisms into one candidate. You may reject all familiar mechanisms and formulate a better one.
6. Give exactly one recommended next scientific question for each possible terminal outcome of the current experiment:
   - K384 unsafe;
   - K384 safe but mixed-budget gain absent;
   - point gain present but paired interval crosses zero;
   - all gates pass.
   These are advisory questions for Pro, not implementation authorization.
7. Explicitly evaluate preservation techniques such as function-preserving residual/adapters, H65 teacher anchoring or distillation, K384 rehearsal, gradient constraints, selective freezing, budget-specific lightweight normalization/calibration, or other alternatives. Do not assume any is correct; compare their attribution quality, risk of merely copying H65, and ability to yield a publishable mechanism.
8. Define a publication-grade experimental ladder that never sacrifices the clean H65 anchor: full 200-video training, fair update/compute treatment, complete 211-video held-out evaluation, three seeds, paired uncertainty, real end-to-end cost, and only then broader benchmarks. Keep the matrix minimal.
9. End with a concise recommendation to Pro that states what must remain immutable, what evidence is still missing, and which single next question would have the highest information gain after the current sealed experiment terminates.

## Required output structure

A. Files and exact commits inspected
B. Clean H65 capability stack and evidence level
C. Why previous routes lost performance
D. Preservation audit of the current implementation
E. What the running experiment can and cannot prove
F. Ranked one-variable preservation-first optimization map
G. Outcome-conditioned next scientific questions
H. Minimal publication-grade experiment ladder
I. Concise advisory recommendation to Pro

End exactly with:

`GEMINI_DUCA_H65_PRESERVATION_ADVISORY_READY`
