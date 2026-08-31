# Gemini independent consultation: DUCA post-admission model improvement

You are an independent senior researcher in video understanding, efficient video models, and temporal action detection. Conduct a read-only, evidence-grounded review of the current DUCA scientific state and advise how to improve the model without repeating disproven routes or creating an uninterpretable multi-variable system.

Respond in Chinese. State the exact model identity you are using. You are an independent technical adviser, not the scientific decision maker: Pro remains the project's scientific lead, and your recommendations do not authorize code changes or experiments.

## Authoritative public repository and current state

- Repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- Current Wiki/evidence branch: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-wiki-complete-sync-20260831>
- Exact Wiki/evidence commit: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/a6c246a66ac9a81e94e2b592da15b79192a74150>
- Current Builder branch, presently still equal to the admitted H65 base and not yet implemented: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-h65-system-multibudget-exposure-v1-20260831>
- Admitted H65 base commit: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/04c35a3b76897e6c1569eeede41ed3aecaf7f854>
- Full-data identity audit code: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/fdd2bcdddf3f23f3546244adf90c4427ed022837>
- Whole-video frozen-detector falsifier: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>

Read the following exact public files completely before drawing conclusions:

- Current research query pack: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/a6c246a66ac9a81e94e2b592da15b79192a74150/research-wiki/query_pack.md>
- Anti-repetition memory: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/a6c246a66ac9a81e94e2b592da15b79192a74150/research-wiki/anti_repetition.md>
- Decision history: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/a6c246a66ac9a81e94e2b592da15b79192a74150/research-wiki/decision_history.md>
- Paper progress: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/a6c246a66ac9a81e94e2b592da15b79192a74150/PAPER_PROGRESS.md>
- Full-data identity audit result: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/a6c246a66ac9a81e94e2b592da15b79192a74150/research-wiki/experiments/duca-full-data-identity-audit-v1.md>
- Latest Pro admission and task order, preserved verbatim: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/a6c246a66ac9a81e94e2b592da15b79192a74150/research-wiki/sources/2026-08-31-pro-duca-full-data-identity-admission-v001.md>
- Previous Pro route integration report: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/a6c246a66ac9a81e94e2b592da15b79192a74150/research-wiki/sources/2026-08-31-pro-duca-comprehensive-route-integration-v001.md>
- Previous Gemini full-history review, which is advisory and contains recorded overclaims: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/a6c246a66ac9a81e94e2b592da15b79192a74150/research-wiki/sources/2026-08-31-agy-gemini-comprehensive-wiki-code-review-v001.md>

When local worktrees are available, inspect the exact code rather than relying only on prose. If a URL or file cannot be read, list it explicitly and distinguish code-confirmed facts, evidence-confirmed facts, and hypotheses.

## Admitted scientific facts

1. Full official data identity is now admitted: 200 training videos and the full OpenTAD 211-video `validation` held-out set. The ActionFormer 212-video set is source-tracing only and differs by `video_test_0000270`; it must not be merged into the OpenTAD population.
2. Held-out labels and metrics remain unopened for the new experiment. All six training units and all predictions must be sealed before the single held-out opening.
3. Dense AdaTAD is a reference upper bound around 68.73 Avg-mAP. H65 fixed K384 is the only reliable sparse baseline, around 65.13 Avg-mAP and 43.31 mAP@0.7, with half the heavy-input observations but without a matched end-to-end latency claim.
4. The frozen-detector K256/K384/K512 transfer action space failed its training-side privileged oracle gate: zero of 704 legal whole-video transfers achieved both +0.8 pp Avg-mAP and +1.0 pp mAP@0.7 at no greater actual observation cost. This stops that specific frozen-detector action space, not dynamic compute in general.
5. Coverage-v1, fine-grained task-state coreset, protected end-to-end bridges, local-cell allocation, and several compression routes already failed their stated tests. Do not recommend repeating them under new names.
6. Cross-budget representation/calibration mismatch is the strongest untested explanation for the frozen-detector failure, not an established root cause.

## Sole Pro-authorized experiment

The only authorized next task is **H65 system-level multi-budget exposure adaptation** on full 200/211 data, three seeds `3407, 3408, 3409`, and one held-out opening.

- Control: H65 Stage-2 system trained with K384 exposure only.
- Candidate: the same H65 Stage-2 system exposed to nested K256/K384/K512 observations during training.
- Both arms start from the same admitted H65 Stage-1 epoch-29 `state_dict_ema`, use the same trainable parameter set, optimizer, schedule, 6,000 successful updates, terminal update-6000 `state_dict_ema`, data, post-processing, evaluator, and seed.
- The candidate keeps `p(384)=0.5`; `p(256)` and `p(512)` must be calibrated from measured actual observation means so expected real heavy observations match the K384 control. Invalid cost monotonicity or probability blocks the run.
- Nested sets must satisfy `S256 subset S384 subset S512` while preserving the H65 producer order. K384 forced mode must reproduce the admitted H65 path exactly.
- Only minimal migration is allowed: true variable-length VideoMAE execution, packet/tensor alignment, actual observation counting, K384 parity, full-211 prediction sealing, official evaluation/cost outputs, focused tests, and minimal Slurm launchers.
- No new selector or controller, Gumbel-Softmax, Mamba, DFT, Block Drop, detector wrapper, distillation, budget embedding, detector/loss/NMS/evaluator change, or workflow framework is authorized in this experiment.
- K384 safety gate: candidate K384 versus matched control must be at least -0.2 pp for Avg-mAP and mAP@0.7.
- Mixed-budget primary gate: at least +0.8 pp Avg-mAP and +1.0 pp mAP@0.7, both 10,000 whole-video paired-bootstrap 95% interval lower bounds above zero, and actual total heavy observations no greater than control.
- If the gate fails, this route stops without rescue tuning. If it passes, a deployment-visible controller may be considered only in a later Pro decision.

## Questions requiring your independent analysis

1. Reconstruct the admitted H65 data flow and identify exactly what “system-level multi-budget exposure” changes and what it keeps fixed. Verify the answer against code.
2. Audit whether the proposed Control/Candidate pair isolates cross-budget adaptation cleanly. Identify any hidden variable that could invalidate attribution, especially selector schedules, trainable masks, variable-length batching, physical-time remapping, short-window collapse, EMA/update clocks, and per-seed randomness.
3. Assess whether nested K256/K384/K512 exposure with actual-cost-calibrated probabilities is the strongest minimal next falsifier. If not, explain the defect without replacing it with a multi-variable redesign.
4. Give concrete implementation advice for the current Builder: the minimum code surfaces, invariants, focused tests, and pre-run failures that matter scientifically. Do not design workflow/provenance machinery.
5. Analyze likely success and failure mechanisms. Separate established evidence, strongest untested hypotheses, and speculative alternatives. Do not call a hypothesis a proven root cause.
6. If this experiment passes, state the smallest scientifically justified next model improvement and the necessary ablation. If it fails, state what is actually ruled out and what remains scientifically open. Do not pre-authorize either branch.
7. Explain how this route could become a publishable paper contribution: central claim, required full-data evidence, uncertainty analysis, cost measurements, ablations, and claim boundaries. Distinguish heavy-observation reduction from real end-to-end latency.
8. Identify any conflict between the latest Pro task order and the current GitHub code/Wiki that must be returned to Pro before implementation proceeds.

Use this output structure:

A. Files and commits actually inspected
B. Verified current model structure
C. Audit of the frozen experiment design
D. Minimal Builder implementation recommendations
E. Ranked success/failure mechanisms
F. Model-improvement path after PASS and after FAIL
G. Publication path and claim boundaries
H. Blocking conflicts or missing evidence
I. Concise recommendation to Pro

End exactly with:

`GEMINI_DUCA_POST_ADMISSION_OPTIMIZATION_READY`
