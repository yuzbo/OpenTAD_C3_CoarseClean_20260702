# DUCA H65 60-epoch compression: learning-rate and curriculum-rate adjudication

Nonce: `DUCA-H65-60-LR-CURRICULUM-PRO-v001-20260824`

You are the independent Scientific First-Author and a severe reviewer for a temporal action detection study. Work from the quantitative evidence and attached code, not from generic learning-rate folklore. Respond in rigorous but readable Chinese. Do not hand the choice back to the human or Codex.

## Exact project and evidence boundary

- Project: DUCA, OpenTAD/AdaTAD on THUMOS14.
- Repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Clean H65 implementation revision: `04c35a3b76897e6c1569eeede41ed3aecaf7f854`.
- The attached files are the authoritative local code truth for the two schedules and the scheduler implementation.
- This discussion concerns optimization/compression of the already established H65 semantic indirect, non-uniform, per-frame selector. It is not a request to redesign the selector, introduce dynamic K, revive continuous cliplets, or repeat dense/uniform/random controls.
- Do not infer terminal results from still-running Jobs `1251622`/`1251782` in older material. The terminal numbers below have since been collected and are authoritative for this turn.

## Frozen model and protocol

All comparisons use the same H65 model structure and input contract: fixed `K=384`; non-uniform semantic indirect frame selection; selected high-resolution RGB frames enter VideoMAE-S; the selected-rank representation and historical physical-time handling are unchanged; AdaTAD/ActionFormer detector, losses, NMS, THUMOS14 split/evaluator, seed `3407`, optimizer family, and per-parameter-group base learning rates are held fixed. A parameter-key audit found the original and compressed candidates architecturally identical (`579` keys). Thus the observed difference is not explained by adding/removing model modules.

## Observed results

1. Original H65 schedule, 90 total epochs:
   - Stage 1: 30 epochs / 3000 successful updates, uniform K384 pretraining.
   - Stage 2: 60 epochs / 6000 successful updates, curriculum to learned semantic sampling and joint training.
   - Terminal epoch-59 EMA: Avg-mAP `65.1257`, mAP@0.7 `43.3137`.
   - Diagnostic trajectory: after Stage-2 epoch 29 (60 total epochs), Avg-mAP about `63.795`; after Stage-2 epoch 39 (70 total), about `64.48`; a diagnostic maximum around Stage-2 epoch 49 (80 total) was about `65.65`; terminal selection remains preregistered epoch-59 EMA, not post-hoc best.
2. Compressed schedule, 60 total epochs:
   - Stage 1: 20 epochs / 2000 updates.
   - Stage 2: 40 epochs / 4000 updates.
   - Terminal epoch-39 EMA: Avg-mAP `62.4648`, mAP@0.7 `39.9434`.
3. Therefore the 90-to-60 compression loses `2.6609` Avg-mAP and `3.3703` at IoU 0.7. Even against the original schedule observed at the same total epoch 60, the compressed allocation is lower by roughly `1.33` Avg-mAP. This strongly suggests that the degradation is not merely “30 fewer terminal epochs”.

## Exact schedule differences that must be causally separated

- Stage-1 duration/update count: `30/3000` -> `20/2000`.
- Stage-2 duration/update count: `60/6000` -> `40/4000`.
- Semantic curriculum transition: `3000` -> `2000` Stage-2 updates.
- Detector-feedback schedule: warmup `1000` + transition `2000` -> warmup `667` + transition `1333` updates.
- Stage-2 cosine scheduler horizon: `max_epoch=60` -> `40`; Stage-1 horizon `30` -> `20`.
- Stage-2 warmup remains 3 epochs, so its fraction of Stage 2 changes.
- Base parameter-group learning rates are unchanged. Current scheduler uses one absolute scalar `eta_min`, so a nonzero common floor can distort parameter-group LR ratios.

## Human scientific constraints

1. Keep every parameter-group base LR unchanged in the first attribution round.
2. Preserve the original 30-epoch uniform Stage 1; do not compress or rerun it when its epoch-29 EMA can be reused.
3. Adjust Stage-2 decay speed/horizon before considering a higher peak LR.
4. Preserve useful nonzero LR late in joint training, e.g. a longer cosine horizon or a short flat tail.
5. Only after gradient stability and genuine underfitting are demonstrated may one parameter group receive a small, isolated LR change; never multiply every group by one factor.
6. Checkpoints every 5 epochs; final/final-EMA rule fixed in advance; no post-hoc intermediate selection.
7. Avoid repeating already completed 20+40 or original 30+30 diagnostics when they can be reused as anchors.

## Required adversarial analysis

First, attack the causal interpretation. Decide which mechanisms are most likely responsible for the large drop and which are merely correlated:

- insufficient uniform Stage-1 representation stabilization;
- curriculum changing too quickly in update time;
- detector feedback entering too early;
- Stage-2 cosine decay reaching low LR before the selector/detector co-adaptation converges;
- reduced total joint updates;
- EMA lag or checkpoint-selection effects;
- any overlooked implementation difference visible in the attached configs/scheduler.

Explicitly explain why the 20+40 result can be lower than the original schedule at the same total epoch 60 even though the model is identical.

Second, issue exactly one scientific disposition: `CONTINUE`, `REVISE`, `PIVOT`, or `STOP`. If continuing/revising, freeze the smallest decision-changing experiment matrix. You must choose exact scheduler formulas and hyperparameters, not just say “tune LR”. At minimum adjudicate these candidate families:

- `LongCosine-90`: reuse Stage-1 epoch-29 EMA; Stage 2 runs 30 epochs but evaluates a cosine schedule whose horizon remains 90 Stage-2 epochs, keeping terminal LR well above zero.
- `CosineFloorTail`: reuse Stage-1 epoch-29 EMA; Stage 2 runs 30 epochs; decay each parameter group by a shared multiplicative factor to a relative floor (candidate `0.2x`) by epoch 25, then hold for 5 epochs. This needs a relative-factor implementation to preserve group ratios.
- A strictly better single alternative, if justified, that remains scheduler/curriculum-only and does not alter the model.

Then decide whether curriculum-rate variation should be tested in the same round or only after the scheduler winner. The current proposed second step is to compare the original transition (`3000`, feedback `1000+2000`) against a faster transition (`2000`, feedback `667+1333`) while fixing the scheduler winner. You may revise these numbers, but must freeze exact values.

## Required output

Return one self-contained report containing:

1. A causal diagnosis ranked by expected contribution, grounded in the observed trajectory and code.
2. A table of the minimum new runs, with Stage-1 checkpoint, Stage-2 epochs/updates, warmup, scheduler equation/horizon/floor/tail, curriculum and feedback timings, seed, terminal checkpoint, and purpose.
3. A stopping rule after round 1 that prevents a grid search. State which result triggers scheduler selection, curriculum testing, or termination.
4. The exact metrics needed beyond Avg-mAP: mAP@0.7, selector/detector/transition gradient norms by parameter group, LR traces, selector entropy or concentration, selected-frame displacement from uniform, and any EMA-vs-online diagnostic you consider essential. Distinguish health diagnostics from claim evidence.
5. The minimal Builder change surface and focused tests, especially how to implement a per-group-relative LR floor without changing base LR ratios and how to prove resume fidelity.
6. Independent Critic checks and Evaluator PRE_RUN gates before N16R4 submission.
7. A clear claim boundary: this matrix studies whether schedule compression preserves H65, not whether a new DUCA mechanism is superior.
8. `next_owner`, `next_action`, `dependency`, and a concrete expected return.

Be severe: reject any arm whose outcome cannot distinguish update-budget, decay-horizon, and curriculum-rate explanations. But do not propose another broad theory round or repeat controls already completed.
