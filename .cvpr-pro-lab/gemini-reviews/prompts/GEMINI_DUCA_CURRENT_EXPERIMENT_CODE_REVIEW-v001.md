# Gemini independent review: current DUCA experiment and code

You are an independent senior reviewer in computer vision, video understanding, and temporal action detection. Analyze the current DUCA experiment and its latest public implementation. Do not agree with the prior Pro `STOP` by default; judge it independently from code, experimental design, and numerical evidence.

Respond in Chinese and explicitly state the actual model identity you are using.

## Latest public code identity

- Repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- Branch: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831>
- Exact commit: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>
- Runner: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tools/bata/run_duca_whole_video_consistent_budget_falsifier.py>
- Focused test: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tests/test_duca_whole_video_consistent_budget_falsifier.py>
- Three-tier allocator: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/opentad/models/duca/dynamic_budget.py>

Inspect these GitHub files when accessible. If any link cannot be accessed, list it and distinguish code-confirmed facts from inferences based on the supplied evidence.

## Model background

1. The deployable base is fixed-budget H65. For each 768-frame window, a low-cost Scout produces an action/boundary-related priority sequence. A deterministic rule selects K384 high-resolution observations. VideoMAE-S is followed by a Temporal Adapter, ActionFormer/AdaTAD head, physical-time inverse mapping, and Soft-NMS.
2. Shared dense AdaTAD has Avg-mAP 68.73. H65 30+60 official validation has Avg-mAP 65.13 and mAP@0.7 43.31. The change from 768 to 384 proves that heavy-input observation count is halved, but there is no matched-hardware end-to-end latency, throughput, memory, or energy measurement.
3. Native-tubelet uniform K384 has Avg-mAP/mAP@0.7 64.13/42.45. Task-state coreset has 62.81/40.56.
4. Later diagnostics freeze the detector and use the same H65 priority sequence to construct nested K256/K384/K512 sealed predictions. They are training-side privileged-oracle budget-transfer tests, with no detector or Scout forward pass, training, or gradients, and they are not deployable controllers.
5. The final whole-video experiment requests K256 for every window of a donor video, K512 for every window of a recipient video, and K384 for all other videos. Candidates are generated before label or metric access. Only candidates with total actual observation cost no greater than fixed K384 are retained. The same sealed predictions, physical coordinate mapping, Soft-NMS, and evaluator are then used for enumeration.

## Terminal evidence

- 40 training-side controller-holdout videos and 124 windows.
- Fixed K384 actual observation cost: 47110.
- 1560 ordered video pairs and 704 legal candidates.
- Fixed/capped/released reproduction error: 0.0 percentage points for all six metrics.
- Job 1262190 completed 704/704 candidates; fixed plus candidates used 705 evaluator calls.
- Fixed K384: Avg-mAP 88.131197%, mAP@0.7 76.270583%.
- Best Avg-mAP candidate: +0.694215 pp Avg-mAP, -0.043632 pp mAP@0.7, cost 46982.
- Best mAP@0.7 candidate: -0.235922 pp Avg-mAP, +0.496998 pp mAP@0.7, cost 46854.
- Best joint-gate candidate: +0.147383 pp Avg-mAP, +0.489786 pp mAP@0.7, cost 45830.
- Preregistered continue gate: delta Avg-mAP at least +0.8 pp, delta mAP@0.7 at least +1.0 pp, and cost no greater than 47110.
- Passing candidate count: 0.
- There is no paired bootstrap, official validation/test, population uncertainty, deployable policy, or end-to-end efficiency measurement.
- The scoped STOP covers only the current THUMOS14 training-side holdout, frozen H65 detector and priority sequence, sealed K256/K384/K512 action space, actual observation cost, and resource scope. It does not extend to every dynamic-compute method, Scout, budget-conditioned model, internal token/layer action, or dataset.

## Required independent review

1. Reconstruct the model data flow and the actual code path of the latest whole-video falsifier in technically accurate but accessible language.
2. Check whether the code really guarantees candidate generation before label/metric access, consistent within-video budgets, actual-cost rather than requested-K accounting, preservation of sealed prediction order, the same Soft-NMS/evaluator, and a fair fixed anchor.
3. Identify any implementation error, leakage, unfair comparison, statistical misreading, or evidence gap that would change the scientific conclusion. Do not elevate style issues into scientific blockers.
4. Independently judge whether zero passing candidates is enough to stop this action space. Explain why a predictor/controller should or should not be trained when privileged-oracle headroom is below the gate.
5. Rank failure explanations as supported, most plausible but unconfirmed, and alternatives. In particular, assess the hypothesis that a detector trained only at K384 has cross-budget representation or calibration mismatch at K256/K512.
6. Quantify compute redundancy precisely: what can the 768-to-384 change support as a claim, and can the 47110-to-45830 observation reduction of 2.72% be presented as a method gain?
7. State paper-eligible claims, prohibited claims, and whether the scoped STOP is scientifically justified.
8. Propose a genuinely distinct next mechanism and its cheapest falsifier only if the evidence supports opening one. Otherwise recommend keeping the STOP.

Use this structure:

A. Code visibility and confirmed facts
B. Model structure
C. Correctness of the experimental implementation
D. Performance and computation conclusions
E. Ranked failure causes
F. Independent judgment of the STOP
G. Paper claim boundaries
H. Whether a new mechanism is justified and its minimum-cost falsifier
