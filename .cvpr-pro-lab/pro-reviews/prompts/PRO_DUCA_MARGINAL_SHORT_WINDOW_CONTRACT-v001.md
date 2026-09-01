# DUCA-Marginal-v1 short-window contract adjudication

**Nonce:** `DUCA-MARGINAL-SHORT-WINDOW-CONTRACT-v001-20260831`

You remain the independent scientific head of DUCA. Codex is implementing the already frozen `DUCA-Marginal-v1` counterfactual probe and has not launched any GPU job. Issue one direct scientific decision: **CONTINUE**, **REVISE**, or **STOP**. This turn is only to remove a real contradiction in the short-window budget contract; do not choose a different research route, add a new model component, start a hyperparameter search, or authorize detector training.

## Frozen experiment that must otherwise remain unchanged

- Clean base: `04c35a3b76897e6c1569eeede41ed3aecaf7f854`
- Branch: `feature/duca-marginal-budget-v1-20260830`
- Frozen H65 terminal checkpoint: `/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823/gpu1_id0/checkpoint/epoch_59.pth`
- SHA-256: `dafcfbd0b1e0a13c400789e73ee13a20cf69551813ef62fc8185fde609806a1c`
- State: `state_dict_ema`, epoch 59
- Main tiers: requested `K in {256,384,512}`; one packet is 16 non-contiguous H65-ranked observations.
- A: Fixed-H65-384; B: train-side Oracle-Reallocate-384; C: Learned-Reallocate-384.
- 160/40 video-level training-side split, seed 3407. No official-test access in this probe.
- H65 Scout, detector, loss, decoding, NMS, annotation, class map and evaluator remain frozen.
- K384 must reproduce the sealed H65 behavior; utility labels are detached signed losses `L256-L384` and `L384-L512`.
- No padding every arm to 512; different heavy budgets must cause genuinely different heavy input lengths.
- Secondary mean-K320 arms remain unresolved and are not part of this question.

## Newly verified real-data fact

Codex inspected the authoritative THUMOS14 annotation at `/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json` using the frozen `feature_stride=4` sliding-window interpretation. The detector-training side contains 200 videos:

- 43 videos have fewer than 512 valid dense observations;
- 62 videos have fewer than 768 valid dense observations;
- the minimum is 67 valid observations.

The sliding-window dataset shifts the final window to full length only when the video itself is long enough. Therefore short windows are common and reachable, not an exotic corner case.

## Contradiction that blocks faithful execution

Your previous frozen report simultaneously required:

1. per-video total budget `384 * actual_window_count` and the focused assertion `sum(K_dynamic) == 384 * num_windows`;
2. short windows billed by actual valid length, with unused budget reallocated and not counted as padding;
3. 16-observation packets and real variable heavy tensor shapes;
4. K384 prediction parity with the existing H65 path for every window.

For a one-window video with only 67 valid observations, these cannot all be true at once. Its maximum unique actual cost is 67 (or 64 if whole 16-observation packets are mandatory), so it cannot meet an exact cost target of 384 within the same video. The existing H65 K384 path pads invalid slots and executes the nominal K384 tensor; shortening that heavy tensor changes the execution and may break the required prediction parity. The current draft runner therefore fails closed when `valid_count < 512`, which would reject many formal training-side videos and is not an acceptable final implementation.

## Required authoritative output

Freeze one coherent, executable rule set. State exact formulas and behavior, not general advice:

1. Define `actual_cost(V,K)` for valid length `V` and requested tier `K`, including whether it is `min(V,K)`, rounded down/up to a 16-observation packet, or another precise rule. State how a partial final packet is executed and billed.
2. Define the exact per-video target budget. Choose whether it is nominal `384*N`, baseline actual cost `sum actual_cost(V_i,384)`, or another explicit quantity.
3. State the redistribution domain and fallback when a video's target is infeasible: within-video only, across-video, exclusion, fixed baseline, or another precise action. Preserve train/holdout isolation and no test leakage.
4. State exactly what K384 parity must mean for short windows: identical requested indices, identical valid prefix, identical predictions under the historical padded path, or a separately named admissible comparison.
5. State how to define `L256`, `L384`, and `L512` when two or more requested tiers collapse to the same valid prefix, and whether such windows train the utility head.
6. State how Oracle and Learned allocation metrics count the fraction assigned to K256/K512 when actual costs collapse.
7. Confirm whether the 160/40 split still includes all 200 training-side videos.
8. Give the minimal Builder changes and focused tests needed to implement your answer. Do not modify VideoMAE, ActionFormer/AdaTAD head, loss, NMS, split, annotation, class map, official evaluator, Scout or checkpoint.
9. Return one next action with owner and an absolute Beijing-time deadline. If the contract cannot be made coherent without changing the scientific question, say **STOP**.

The answer must be self-contained and must not rely on prior chat context. Treat the facts above as the complete material for this adjudication and correct any inconsistency directly. Do not authorize GPU execution unless the short-window contract is fully frozen.
