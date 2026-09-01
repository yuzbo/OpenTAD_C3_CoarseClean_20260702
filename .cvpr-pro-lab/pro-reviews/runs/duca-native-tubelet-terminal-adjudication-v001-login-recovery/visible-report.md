# PIVOT

**Nonce:** `DUCA-NATIVE-TUBELET-CORESET-TERMINAL-v001-20260829`

The exact H65-derived fixed-budget native-tubelet coreset is terminal as a candidate. Do not tune its score weights, endpoint rule, maximum-gap rule, or packing behavior, and do not retrain it. Pivot from **fine-grained “which tubelets?” selection** to a single test of **window-level “how much heavy computation?” allocation**, while retaining deterministic uniform native-tubelet sampling inside each assigned budget.

## Scientific interpretation

### What the completed experiment falsifies

On seed `3407`, under the frozen implementation and matched `K=384` high-resolution RGB budget, the task-state coreset failed its operational prediction:

* Avg-mAP was `62.81`, versus `64.13` for deterministic uniform selection: `−1.32` percentage points.
* The deficit increased at the localization-sensitive thresholds: `−2.03` at tIoU `0.6` and `−1.89` at tIoU `0.7`.
* It was lower at every reported tIoU threshold.

Therefore, this experiment rejects the **implemented package-level hypothesis** that the frozen H65 task-state score, endpoint forcing, maximum unselected gap of seven tubelets, and current VideoMAE packing/reconstruction path provide a better fixed-budget allocation than deterministic uniform native-tubelet selection.

It does **not** establish a statistically significant negative population effect, because paired predictions and a paired whole-video interval are absent. It also does not specifically falsify short-action preservation because no duration-stratified result was produced.

It does not falsify:

* all low-cost scout models;
* all sparse heavy-compute designs;
* dynamic budgeting;
* uniform sampling at variable budgets;
* alternative ways of presenting nonadjacent tubelets to VideoMAE;
* any unique causal explanation for the loss;
* an efficiency claim, because realized compute and end-to-end cost were not measured.

### Evaluation-only sealing decision

**Do not run a standalone uniform-versus-coreset sealing pass.**

The existing point estimates already settle the next scientific action. The candidate needed positive preservation evidence to remain the fixed-budget route; instead, it is lower at every threshold and materially lower at high tIoU. A paired interval could produce either:

* a confidently negative effect, which would strengthen rejection; or
* an interval crossing zero, which would still provide no positive evidence for continuing the exact coreset.

Neither outcome would make this coreset the next method. The interval would refine retrospective reporting, not change route selection.

The existing uniform epoch-59 exponential-moving-average checkpoint may nevertheless be evaluated once in the new task to create a sealed paired control for the new dynamic-budget arm. The coreset checkpoint is not reopened.

## Ranked mechanism explanations

Only the performance difference is observed. None of the following explanations is causally established.

| Rank  | Explanation                                                                                       | Evidence and limitation                                                                                                                                                                                                                                                                                                                                                                                           |
| ----- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **Fine-grained selection-score mismatch**                                                         | The selected tubelet identities are the intended arm difference, and the coreset is worse across all thresholds. The frozen combination of actionness, boundary importance, and novelty may not rank the visual evidence most useful to the downstream detector. No boundary-distance, discarded-context, or oracle-utility diagnostic was sealed, so this remains a hypothesis rather than a demonstrated cause. |
| **2** | **Artificial adjacency inside VideoMAE clips**                                                    | Nonadjacent physical tubelets are packed as neighboring positions in conventional 16-frame clips. An irregular coreset can therefore create different and potentially more harmful temporal mixing than deterministic uniform selection. The larger losses at high tIoU are compatible with this explanation, but there is no adjacency-neutral intervention, so causality is unproven.                           |
| **3** | **Endpoint and maximum-gap constraints protect coarse coverage but not action-boundary evidence** | Forced window endpoints and a maximum gap of seven tubelets prevent gross temporal holes. They do not guarantee paired observations immediately before and after actual action starts and ends, nor preserve boundary-centered local density. No realized action-boundary coverage statistics are available.                                                                                                      |
| **4** | **Interaction with context recycling or physical-time reconstruction**                            | These components are shared across the two arms and therefore cannot independently explain the contrast. They could amplify errors caused by irregular selections, but there is no discriminating observation supporting that interaction.                                                                                                                                                                        |

The next experiment should not attempt to distinguish these four explanations one by one. The publication-relevant question is whether the frozen scout contains useful **coarse compute-demand information** even though its fine-grained tubelet ranking failed.

# Frozen next scientific task

## Task: task-state-guided variable uniform tubelet budget

### Scientific hypothesis

The frozen scout may be insufficiently accurate to choose individual high-value tubelets, while still being accurate enough to rank which video windows are difficult and which are temporally redundant.

Accordingly:

> Keep high-demand windows at the fixed uniform control density, reduce heavy computation only for lower-demand windows, and use deterministic uniform native-tubelet selection within every assigned budget.

The differential prediction is that this mechanism will preserve official detection accuracy—especially high-tIoU and short-action performance—while executing fewer VideoMAE clips on average.

The principal alternative explanation is that the frozen task-state signal is not reliable even at window level; in that case, assigning reduced budgets from it will disproportionately damage action or boundary-bearing windows.

## Exact mechanism

Build from frozen revision:

`b33391126eac05e3353d322b973dda91741f0732`

Create one successor branch:

`codex/duca-dynamic-native-tubelet-budget-20260829`

Use exactly the same frozen H65 Stage-1 epoch-29 exponential-moving-average scout bound to the completed coreset experiment. Do not retrain or modify the scout.

For each window, derive three scalar quantities from the scout’s raw valid-tubelet outputs:

1. mean two-frame actionness;
2. 90th percentile of two-frame boundary importance;
3. 90th percentile of temporal novelty.

Within each whole video, convert each quantity to a percentile rank over that video’s windows. Combine the three percentile ranks using the **same fixed component weights already used by the completed coreset**. This produces one window-demand score. No ground truth, detector predictions, validation metrics, or checkpoint-dependent information may enter this score.

For a video containing `W` windows:

* sort windows by demand score, with earlier physical start time as the deterministic tie-break;
* let `q = floor(W / 2)`;
* assign the lowest `q` windows **16 VideoMAE clips**;
* assign the highest `q` windows **24 VideoMAE clips**;
* when `W` is odd, assign the single median window **20 VideoMAE clips**.

This gives exactly:

$$
\frac{1}{W}\sum_w B_w = 20\ \text{clips per window},
$$

including odd `W`. The fixed control executes `24` clips per window, so the new mechanism executes **16.67% fewer heavy clips by construction**.

Within each assigned budget, select native two-frame tubelets deterministically and uniformly on the valid physical-time grid:

* 16 clips: 128 tubelets, 256 high-resolution frames;
* 20 clips: 160 tubelets, 320 high-resolution frames;
* 24 clips: 192 tubelets, 384 high-resolution frames.

Selected tubelets remain ordered by physical time and use the existing packing, low-resolution context recycling, physical-time residual, and reconstruction to the 384-point native-tubelet grid. The task introduces no task-state ranking among tubelets.

Windows with different budgets must be grouped for VideoMAE execution so that only `16`, `20`, or `24` clips are actually processed. Padding all windows to 24 heavy clips is an implementation failure, even if outputs are masked afterward.

## Frozen implementation boundary

The Builder may modify only:

* the existing native-tubelet budget assignment path;
* the existing deterministic uniform native-tubelet selection call so it accepts the three frozen counts;
* the minimal VideoMAE batching/grouping surface required to execute the actual clip count;
* one dynamic-budget configuration;
* focused tests for this mechanism;
* minimal counters needed to report realized heavy execution.

The Builder must not change:

* VideoMAE-S weights or architecture;
* Adapter or ActionFormer;
* detector head, losses, optimizer, learning-rate schedule, or update count;
* low-resolution context recycling;
* physical-time residual or reconstruction;
* data split, augmentations, annotations, or class mapping;
* non-maximum suppression;
* official evaluator;
* checkpoint-selection rule;
* scout weights or existing score-component weights;
* control checkpoint;
* short-action definition after training begins.

No new selector family, general scheduler, budget-prediction network, score search, budget curve, or parallel implementation is authorized.

# Training and evaluation

## Formal training

Run exactly one new training arm:

* full official THUMOS14 training data;
* seed `3407`;
* same 60-epoch schedule as jobs `1260184` and `1260185`;
* same optimizer and all matched training settings;
* checkpoints at least every five epochs;
* final result from epoch-59 `state_dict_ema`;
* no validation-based epoch selection;
* one GPU formal training job.

Do not retrain dense, fixed uniform, random, coreset, arbitrary-frame, or continuous-cliplet controls. Do not run multiple budget formulas.

## Control

The control is the immutable epoch-59 exponential-moving-average checkpoint from uniform job `1260184`.

Evaluate it once with prediction saving enabled. It must reproduce, within `1e-6` percentage points:

| Metric  | Required reproduction |
| ------- | --------------------: |
| Avg-mAP |                 64.13 |
| mAP@0.3 |                 79.26 |
| mAP@0.4 |                 74.89 |
| mAP@0.5 |                 67.44 |
| mAP@0.6 |                 56.61 |
| mAP@0.7 |                 42.45 |

Failure to reproduce these values invalidates the paired comparison; it triggers only a bounded evaluator-path correction, not model retraining or a new scientific route.

## Required official-data outputs

For both the fixed uniform control and the new dynamic arm:

* official metrics at tIoU `0.3, 0.4, 0.5, 0.6, 0.7`;
* Avg-mAP;
* predictions for all 211 THUMOS14 evaluation videos;
* exact video-ID equality;
* `post_processing.save_dict=True`;
* unchanged non-maximum suppression and official evaluator.

Required files:

```text
uniform_control/metrics_epoch59_ema.json
uniform_control/result_detection.json
dynamic_mean20/metrics_epoch59_ema.json
dynamic_mean20/result_detection.json
dynamic_mean20/budget_trace.jsonl
dynamic_mean20/realized_compute.json
duration_stratified_metrics.json
boundary_error_diagnostics.json
paired_video_bootstrap.json
```

`budget_trace.jsonl` must contain, for every video window:

* video ID and physical window start;
* the three raw aggregate scout quantities;
* their within-video ranks;
* combined demand score;
* assigned clip budget;
* actually executed clip count;
* selected tubelet and high-resolution frame counts.

`realized_compute.json` must report:

* total and mean executed VideoMAE clips;
* total selected high-resolution frames;
* total heavy temporal tokens;
* scout computation separately;
* VideoMAE and end-to-end model-forward timing under matched hardware, batch, precision, and data-loader settings.

The hard scientific compute identity is the number of clips actually processed. Wall-clock measurements determine whether a runtime-speed claim is allowed.

## Short-action and boundary evidence

Freeze the short-action threshold before detector training as the bottom quartile of ground-truth action duration on the THUMOS14 training split. Apply that fixed duration threshold to the official evaluation split.

Report:

* short-action Avg-mAP;
* short-action mAP@0.7;
* median normalized start error and end error for one-to-one matched true positives at tIoU at least `0.5`.

The boundary-error result is diagnostic; official mAP remains primary.

## Paired statistic

Use a paired whole-video bootstrap:

* baseline: fixed uniform;
* candidate: dynamic mean-20 budget;
* sampling unit: whole video;
* 211 video IDs;
* 10,000 resamples with replacement;
* identical sampled IDs and multiplicities for both arms;
* bootstrap random seed `3407`;
* recompute metrics from predictions and ground truth for every resample;
* two-sided percentile 95% interval using the 2.5th and 97.5th percentiles.

Report intervals for:

* dynamic minus uniform Avg-mAP;
* mAP@0.7;
* short-action Avg-mAP.

# Decision and stop rules

The dynamic mechanism advances only if all of the following hold:

1. **Execution correctness:** every window executes exactly its assigned number of heavy clips; no hidden 24-clip padding or unused VideoMAE work occurs.
2. **Compute:** mean executed heavy clips are exactly `20.0` per window, versus `24.0` for control.
3. **Average detection non-inferiority:** the lower endpoint of the paired 95% interval for dynamic-minus-uniform Avg-mAP is greater than `−0.50` percentage points.
4. **Boundary protection:** dynamic-minus-uniform mAP@0.7 is at least `−0.50` points.
5. **Short-action protection:** dynamic-minus-uniform short-action Avg-mAP is at least `−0.50` points.
6. **Validity:** all 211 videos, checkpoint identity, evaluator, NMS, split, and model-selection rules are matched.

The mechanism is stopped if any correctness or compute condition fails, or if any accuracy protection condition fails. There is no automatic threshold adjustment, second budget formula, coreset repair, or second training run under this task.

An outcome that meets the point-estimate guards but misses the paired non-inferiority interval is **unresolved and not claimable**; it does not authorize silent margin changes or an additional run.

# Dynamic-budget judgment

**True dynamic budgeting remains scientifically justified for this one experiment.**

The fixed-budget result tested whether the scout could identify better individual tubelets at a constant budget. It did not test whether aggregate scout evidence can identify windows that tolerate reduced heavy sampling. The new mechanism isolates that surviving hypothesis:

* no learned tubelet ranking;
* no extra compute for any window;
* high-demand windows retain the full control budget;
* low-demand windows execute fewer clips;
* mean VideoMAE work is strictly below the control.

Fixed `K=384` remains the matched control and fallback, not the headline method.

# Result-to-claim boundary

If the task passes, the admissible statement is:

> On the single frozen THUMOS14 seed and backend, a task-state-guided per-window budget of 16, 20, or 24 uniformly selected native-tubelet clips preserved detection within the pre-specified non-inferiority margins while executing 16.67% fewer VideoMAE clips than the fixed 24-clip control.

It would not yet justify claims of:

* statistically stable performance across training seeds;
* improvement over the fixed control;
* superiority to dense AdaTAD;
* general dynamic-budget effectiveness;
* cross-dataset or cross-detector generality;
* end-to-end speedup unless the matched runtime measurement is lower;
* a causal explanation for the failed coreset.

If the task fails, the admissible statement is:

> The frozen H65 task-state signal did not support this pre-specified window-level budget reduction under the matched THUMOS14 experiment.

That failure would terminate this particular demand score and `16/20/24` budget rule. It would not establish that every possible dynamic-budget method is impossible.

# Builder → independent Critic → Evaluator handoff

**Builder**

Implement only the frozen mechanism and focused correctness tests. Required tests cover deterministic budget assignment, no ground-truth access, exact per-video mean of 20 clips, exact tubelet counts, physical-time ordering, fixed 384-point reconstruction, frozen scout parameters, expected detector gradients, and absence of padded heavy execution.

**Independent Critic**

Review the exact Builder revision once for:

* faithful window-demand computation;
* absence of validation/test ground truth or detector-prediction leakage;
* actual rather than nominal variable VideoMAE execution;
* unchanged detector, losses, schedule, NMS, evaluator, and control path;
* no hidden score or threshold search;
* numerical parity of the fixed-24 path with revision `b333911…`.

Only a defect that changes model behavior, fairness, leakage, coordinate semantics, or realized compute may block execution.

**Evaluator**

Run the full dynamic training, seal the fixed control evaluation, perform official evaluation, cost accounting, duration-stratified analysis, boundary diagnostics, and the paired bootstrap. Return raw outputs and a factual pass/fail application of the frozen rules; do not select the next route.

**Absolute deadline:** `2026-09-04T23:59:00+08:00`.

```text
next_owner: Builder, followed by independent Critic, then Evaluator
next_action: implement and execute the single task-state-guided 16/20/24-clip dynamic uniform native-tubelet budget experiment
dependency: revision b33391126eac05e3353d322b973dda91741f0732; the exact frozen H65 Stage-1 epoch-29 EMA scout; uniform job 1260184 epoch-59 EMA checkpoint; official THUMOS14 data and evaluator; VideoMAE-S pretraining asset; authorized GPU execution
expected_return_at: 2026-09-04T23:59:00+08:00
```
