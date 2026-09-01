# Oracle Browser Transcript

Conversation: https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697-duca/c/6a8b95e8-7d08-83ea-a6d7-9d4fa59976f2

## Prompt

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

### File: ../OpenTAD_DUCA_H65First_MatchedCycle6_20260822/configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py
Lines: 1-93
```python
 1 | _base_ = ["./duca_sampling_rate_both_asformer_full_adapt_fixed384_official60.py"]
 2 | 
 3 | # Stage 1 is deliberately a real full-model uniform-K=384 warmup.  The
 4 | # detector learns on the same observations it will see in the uniform control,
 5 | # while the ASFormer coarse probe is optimized only by binary actionness and
 6 | # transition supervision.  No learned sampling or detector-to-policy bridge is
 7 | # enabled in this phase.
 8 | duca_stage1_end_epoch = 30
 9 | duca_stage1_steps_per_epoch = 100
10 | duca_stage1_total_steps = duca_stage1_end_epoch * duca_stage1_steps_per_epoch
11 | 
12 | duca_sampling_rate_contract = dict(
13 |     route="DUCA_RATE_CURRICULUM_STAGE1_UNIFORM384",
14 |     task="offline_temporal_action_detection",
15 |     stage="uniform_k384_full_model_coarse_convergence",
16 |     pre_backbone_plugin=True,
17 |     exact_budget=384,
18 |     detector_policy="exact_uniform",
19 |     coarse_supervision=[
20 |         "binary_actionness",
21 |         "state_transition_distribution",
22 |         "transition_boundary_support",
23 |     ],
24 |     detector_gradient_to_selector=False,
25 |     paper_claim_allowed=False,
26 | )
27 | 
28 | model = dict(
29 |     frame_selector=dict(
30 |         inference_policy_alpha=0.0,
31 |         training_uniform_companion_fraction=0.0,
32 |         # Stage 1 has no learned-policy rows, so companion-only gradient
33 |         # normalization is inapplicable and must not leak from the joint base.
34 |         training_uniform_companion_normalize_learned_gradient=False,
35 |         detector_gradient_mode="density_transport_st",
36 |         detector_contribution_distillation_weight=0.0,
37 |         coarse_trunk_lr=5.0e-5,
38 |         action_head_lr=1.0e-4,
39 |         transition_scorer_lr=5.0e-5,
40 |         loss_weights=dict(
41 |             actionness=1.0,
42 |             transition=0.50,
43 |             transition_boundary=2.0,
44 |         ),
45 |         loss_weight_schedule=dict(
46 |             _delete_=True,
47 |             type="progressive_joint",
48 |             shape="linear",
49 |             warmup_steps=0,
50 |             transition_steps=1,
51 |             actionness=dict(start=1.0, end=1.0),
52 |             transition=dict(start=0.50, end=0.50),
53 |             transition_boundary=dict(start=2.0, end=2.0),
54 |             policy_alpha=dict(start=0.0, end=0.0),
55 |             detector_gradient=dict(start=0.0, end=0.0),
56 |             detector_contribution=dict(start=0.0, end=0.0),
57 |             asformer_adapt=dict(start=0.0, end=0.0),
58 |         ),
59 |         actionness_source_cfg=dict(
60 |             policy_hidden_gradient_scope="asformer_full_encoder",
61 |         ),
62 |     ),
63 | )
64 | 
65 | scheduler = dict(
66 |     type="LinearWarmupCosineAnnealingLR",
67 |     warmup_epoch=3,
68 |     max_epoch=duca_stage1_end_epoch,
69 | )
70 | 
71 | workflow = dict(
72 |     formal_protocol="",
73 |     training_profile="duca_rate_curriculum_stage1_uniform384",
74 |     checkpoint_interval=5,
75 |     val_loss_interval=-1,
76 |     val_eval_interval=5,
77 |     val_eval_interval_anchor_epoch=5,
78 |     val_start_epoch=4,
79 |     intermediate_validation_role="stage1_learning_curve_only",
80 |     intermediate_validation_selects_checkpoint=False,
81 |     end_epoch=duca_stage1_end_epoch,
82 |     formal_successful_update_contract=False,
83 |     expected_train_batches_per_epoch=duca_stage1_steps_per_epoch,
84 |     expected_successful_optimizer_updates=duca_stage1_total_steps,
85 |     max_amp_retries_per_batch=8,
86 |     fail_on_amp_replay_exhaustion=True,
87 |     require_finite_train_loss=True,
88 |     primary_checkpoint_epoch=duca_stage1_end_epoch - 1,
89 |     primary_checkpoint_state_key="state_dict_ema",
90 |     checkpoint_criterion="terminal_epoch_29_state_dict_ema",
91 | )
92 | 
93 | work_dir = "exps/thumos/adatad/duca_sampling_rate_curriculum_stage1_uniform384"
```

### File: ../OpenTAD_DUCA_H65First_MatchedCycle6_20260822/configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py
Lines: 1-142
```python
  1 | _base_ = ["./duca_sampling_rate_both_asformer_full_adapt_fixed384_official60.py"]
  2 | 
  3 | import os
  4 | 
  5 | 
  6 | def _required(name):
  7 |     value = os.environ.get(name, "")
  8 |     if not value:
  9 |         raise ValueError(f"{name} is required for DUCA rate curriculum stage 2")
 10 |     return value
 11 | 
 12 | 
 13 | duca_stage1_checkpoint = _required("DUCA_STAGE1_CHECKPOINT")
 14 | duca_stage1_checkpoint_sha256 = _required("DUCA_STAGE1_CHECKPOINT_SHA256")
 15 | duca_stage1_checkpoint_epoch = int(_required("DUCA_STAGE1_CHECKPOINT_EPOCH"))
 16 | 
 17 | # Stage 2 has a single, fresh 6,000-step optimizer schedule.  The first half
 18 | # retains coarse supervision while smoothly turning on learned sampling and
 19 | # detector feedback; the second half is TAD-led but never drops semantic and
 20 | # transition supervision to zero.
 21 | duca_stage2_half_steps = 3000
 22 | 
 23 | # Explicit admission metadata.  These values already come from the inherited
 24 | # official-60 training recipe; spelling them out lets the fail-closed Stage-2
 25 | # validator verify the frozen seed and successful-update budget before loading
 26 | # the Stage-1 handoff.
 27 | seed = 3407
 28 | total_epochs = 60
 29 | max_updates = 6000
 30 | 
 31 | duca_sampling_rate_contract = dict(
 32 |     route="DUCA_RATE_CURRICULUM_STAGE2_JOINT384",
 33 |     task="offline_temporal_action_detection",
 34 |     stage="low_lr_joint_rate_adaptation_then_tad_led_joint_training",
 35 |     pre_backbone_plugin=True,
 36 |     stage1_initialization="full_uniform_k384_ema_model",
 37 |     optimizer_scheduler_amp_state_reset=True,
 38 |     detector_gradient="density_transport_st",
 39 |     final_loss_emphasis=dict(
 40 |         detector=1.0,
 41 |         actionness=0.25,
 42 |         transition=0.10,
 43 |         transition_boundary=0.25,
 44 |     ),
 45 |     paper_claim_allowed=False,
 46 | )
 47 | 
 48 | model = dict(
 49 |     frame_selector=dict(
 50 |         coarse_trunk_lr=1.0e-5,
 51 |         action_head_lr=2.0e-5,
 52 |         transition_scorer_lr=5.0e-5,
 53 |         loss_weights=dict(
 54 |             actionness=1.0,
 55 |             transition=0.50,
 56 |             transition_boundary=2.0,
 57 |         ),
 58 |         loss_weight_schedule=dict(
 59 |             _delete_=True,
 60 |             type="progressive_joint",
 61 |             shape="cosine",
 62 |             warmup_steps=0,
 63 |             transition_steps=duca_stage2_half_steps,
 64 |             actionness=dict(
 65 |                 start=1.0,
 66 |                 end=0.25,
 67 |                 warmup_steps=0,
 68 |                 transition_steps=duca_stage2_half_steps,
 69 |             ),
 70 |             transition=dict(
 71 |                 start=0.50,
 72 |                 end=0.10,
 73 |                 warmup_steps=0,
 74 |                 transition_steps=duca_stage2_half_steps,
 75 |             ),
 76 |             transition_boundary=dict(
 77 |                 start=2.0,
 78 |                 end=0.25,
 79 |                 warmup_steps=0,
 80 |                 transition_steps=duca_stage2_half_steps,
 81 |             ),
 82 |             policy_alpha=dict(
 83 |                 start=0.0,
 84 |                 end=1.0,
 85 |                 warmup_steps=0,
 86 |                 transition_steps=duca_stage2_half_steps,
 87 |             ),
 88 |             detector_gradient=dict(
 89 |                 start=0.0,
 90 |                 end=0.25,
 91 |                 warmup_steps=1000,
 92 |                 transition_steps=2000,
 93 |             ),
 94 |             detector_contribution=dict(
 95 |                 start=0.0,
 96 |                 end=1.0,
 97 |                 warmup_steps=1000,
 98 |                 transition_steps=2000,
 99 |             ),
100 |             asformer_adapt=dict(
101 |                 start=0.0,
102 |                 end=1.0,
103 |                 warmup_steps=0,
104 |                 transition_steps=duca_stage2_half_steps,
105 |             ),
106 |         ),
107 |         actionness_source_cfg=dict(
108 |             policy_hidden_gradient_scope="asformer_full_encoder",
109 |         ),
110 |     ),
111 | )
112 | 
113 | workflow = dict(
114 |     # This curriculum candidate uses the same full THUMOS training and
115 |     # validation protocol as the official-60 arms, but its phase boundary is
116 |     # deliberately outside the frozen selected-axis evidence runtime.  It
117 |     # becomes paper-comparable only after the measured model result is sealed.
118 |     formal_protocol="",
119 |     # This is a new curriculum candidate, not one of the sealed legacy P0
120 |     # variants.  Leaving the inherited P0 contract enabled routes it through
121 |     # the legacy variant binder before model initialization.
122 |     formal_successful_update_contract=False,
123 |     training_profile="duca_rate_curriculum_stage2_joint384",
124 |     # Keep five-epoch validation strictly diagnostic.  The Stage-2 course is
125 |     # always judged by epoch-59 EMA, never by a curve-best checkpoint.
126 |     intermediate_validation_role="learning_curve_only",
127 |     intermediate_validation_selects_checkpoint=False,
128 |     # A pre-AMP NaN/Inf may be replayed only from the untouched batch state.
129 |     # The engine records every event and still fails closed after this bound.
130 |     max_nonfinite_loss_retries=8,
131 |     training_update_audit_json=os.environ.get("DUCA_STAGE2_UPDATE_AUDIT_JSON", ""),
132 |     model_initialization=dict(
133 |         enabled=True,
134 |         checkpoint_path=duca_stage1_checkpoint,
135 |         checkpoint_sha256=duca_stage1_checkpoint_sha256,
136 |         state_key="state_dict_ema",
137 |         expected_checkpoint_epoch=duca_stage1_checkpoint_epoch,
138 |         reset_state_keys=["frame_selector._loss_weight_schedule_step"],
139 |     ),
140 | )
141 | 
142 | work_dir = "exps/thumos/adatad/duca_sampling_rate_curriculum_stage2_joint384"
```

### File: ../OpenTAD_DUCA_H65First_MatchedCycle6_20260822/configs/adatad/thumos/duca_h65_60_stage1_uniform20.py
Lines: 1-44
```python
 1 | """H65-60 Stage 1: the historical H65 uniform warmup compressed to 20 epochs."""
 2 | 
 3 | _base_ = ["./duca_sampling_rate_curriculum_stage1_uniform384.py"]
 4 | 
 5 | duca_stage1_end_epoch = 20
 6 | duca_stage1_steps_per_epoch = 100
 7 | duca_stage1_total_steps = 2000
 8 | 
 9 | duca_sampling_rate_contract = dict(
10 |     route="DUCA_H65_60_STAGE1_UNIFORM20",
11 |     stage="uniform_k384_full_model_coarse_convergence_compressed20",
12 |     curriculum_only_change=True,
13 |     model_change_allowed=False,
14 | )
15 | 
16 | scheduler = dict(
17 |     type="LinearWarmupCosineAnnealingLR",
18 |     warmup_epoch=2,
19 |     max_epoch=duca_stage1_end_epoch,
20 | )
21 | 
22 | workflow = dict(
23 |     training_profile="duca_h65_60_stage1_uniform20",
24 |     end_epoch=duca_stage1_end_epoch,
25 |     expected_train_batches_per_epoch=duca_stage1_steps_per_epoch,
26 |     expected_successful_optimizer_updates=duca_stage1_total_steps,
27 |     primary_checkpoint_epoch=duca_stage1_end_epoch - 1,
28 |     primary_checkpoint_state_key="state_dict_ema",
29 |     checkpoint_criterion="terminal_epoch_19_state_dict_ema",
30 | )
31 | 
32 | seed = 3407
33 | total_epochs = 20
34 | max_updates = 2000
35 | checkpoint_interval_epochs = 5
36 | checkpoint_policy = dict(
37 |     resumable=True,
38 |     keep_latest=3,
39 |     milestones=True,
40 |     final=True,
41 |     final_ema=True,
42 | )
43 | paper_claim_allowed = False
44 | work_dir = "exps/thumos/adatad/duca_h65_60_stage1_uniform20"
```

### File: ../OpenTAD_DUCA_H65First_MatchedCycle6_20260822/configs/adatad/thumos/duca_h65_60_stage2_transition20_joint20.py
Lines: 1-104
```python
  1 | """H65-60 Stage 2: 20-epoch cosine transition plus 20-epoch full joint training."""
  2 | 
  3 | _base_ = ["./duca_sampling_rate_curriculum_stage2_joint384.py"]
  4 | 
  5 | duca_stage2_transition_steps = 2000
  6 | duca_stage2_total_steps = 4000
  7 | 
  8 | duca_sampling_rate_contract = dict(
  9 |     route="DUCA_H65_60_STAGE2_TRANSITION20_JOINT20",
 10 |     stage="low_lr_joint_rate_adaptation20_then_tad_led_joint20",
 11 |     stage1_initialization="full_uniform_k384_epoch19_ema_model",
 12 |     optimizer_scheduler_amp_state_reset=True,
 13 |     curriculum_only_change=True,
 14 |     model_change_allowed=False,
 15 | )
 16 | 
 17 | model = dict(
 18 |     frame_selector=dict(
 19 |         loss_weight_schedule=dict(
 20 |             _delete_=True,
 21 |             type="progressive_joint",
 22 |             shape="cosine",
 23 |             warmup_steps=0,
 24 |             transition_steps=duca_stage2_transition_steps,
 25 |             actionness=dict(
 26 |                 start=1.0,
 27 |                 end=0.25,
 28 |                 warmup_steps=0,
 29 |                 transition_steps=duca_stage2_transition_steps,
 30 |             ),
 31 |             transition=dict(
 32 |                 start=0.50,
 33 |                 end=0.10,
 34 |                 warmup_steps=0,
 35 |                 transition_steps=duca_stage2_transition_steps,
 36 |             ),
 37 |             transition_boundary=dict(
 38 |                 start=2.0,
 39 |                 end=0.25,
 40 |                 warmup_steps=0,
 41 |                 transition_steps=duca_stage2_transition_steps,
 42 |             ),
 43 |             policy_alpha=dict(
 44 |                 start=0.0,
 45 |                 end=1.0,
 46 |                 warmup_steps=0,
 47 |                 transition_steps=duca_stage2_transition_steps,
 48 |             ),
 49 |             # Historical H65 uses 1,000 + 2,000 steps. Scaling both intervals
 50 |             # by 2/3 preserves their ratio and completes feedback at step 2,000.
 51 |             detector_gradient=dict(
 52 |                 start=0.0,
 53 |                 end=0.25,
 54 |                 warmup_steps=667,
 55 |                 transition_steps=1333,
 56 |             ),
 57 |             detector_contribution=dict(
 58 |                 start=0.0,
 59 |                 end=1.0,
 60 |                 warmup_steps=667,
 61 |                 transition_steps=1333,
 62 |             ),
 63 |             asformer_adapt=dict(
 64 |                 start=0.0,
 65 |                 end=1.0,
 66 |                 warmup_steps=0,
 67 |                 transition_steps=duca_stage2_transition_steps,
 68 |             ),
 69 |         ),
 70 |     ),
 71 | )
 72 | 
 73 | scheduler = dict(
 74 |     type="LinearWarmupCosineAnnealingLR",
 75 |     warmup_epoch=3,
 76 |     max_epoch=40,
 77 | )
 78 | 
 79 | workflow = dict(
 80 |     training_profile="duca_h65_60_stage2_transition20_joint20",
 81 |     end_epoch=40,
 82 |     expected_train_batches_per_epoch=100,
 83 |     expected_successful_optimizer_updates=duca_stage2_total_steps,
 84 |     primary_checkpoint_epoch=39,
 85 |     primary_checkpoint_state_key="state_dict_ema",
 86 |     checkpoint_criterion="terminal_epoch_39_state_dict_ema",
 87 |     model_initialization=dict(
 88 |         expected_checkpoint_epoch=19,
 89 |     ),
 90 | )
 91 | 
 92 | seed = 3407
 93 | total_epochs = 40
 94 | max_updates = 4000
 95 | checkpoint_interval_epochs = 5
 96 | checkpoint_policy = dict(
 97 |     resumable=True,
 98 |     keep_latest=3,
 99 |     milestones=True,
100 |     final=True,
101 |     final_ema=True,
102 | )
103 | paper_claim_allowed = False
104 | work_dir = "exps/thumos/adatad/duca_h65_60_stage2_transition20_joint20"
```

### File: ../OpenTAD_DUCA_H65First_MatchedCycle6_20260822/opentad/cores/scheduler.py
Lines: 1-212
```python
  1 | import math
  2 | import warnings
  3 | from bisect import bisect_right
  4 | from collections import Counter
  5 | from torch.optim.lr_scheduler import _LRScheduler
  6 | 
  7 | 
  8 | def build_scheduler(cfg, optimizer, dataloader_len):
  9 |     scheduler_type = cfg["type"]
 10 |     cfg.pop("type")
 11 | 
 12 |     max_epoch = cfg["max_epoch"]
 13 | 
 14 |     if scheduler_type == "LinearWarmupCosineAnnealingLR":
 15 |         cfg["warmup_epoch"] *= dataloader_len
 16 |         cfg["max_epoch"] *= dataloader_len
 17 |         scheduler = LinearWarmupCosineAnnealingLR(optimizer, **cfg)
 18 |     elif scheduler_type == "LinearWarmupMultiStepLR":
 19 |         cfg.pop("max_epoch")
 20 |         cfg["warmup_epoch"] *= dataloader_len
 21 |         cfg["milestones"] = [dataloader_len * step for step in cfg["milestones"]]
 22 |         scheduler = LinearWarmupMultiStepLR(optimizer, **cfg)
 23 |     elif scheduler_type == "MultiStepLR":
 24 |         cfg.pop("max_epoch")
 25 |         cfg["milestones"] = [dataloader_len * step for step in cfg["milestones"]]
 26 |         scheduler = LinearWarmupMultiStepLR(optimizer, warmup_epoch=0, **cfg)
 27 |     else:
 28 |         raise f"Optimizer {scheduler_type} is not supported so far."
 29 | 
 30 |     return scheduler, max_epoch
 31 | 
 32 | 
 33 | class LinearWarmupCosineAnnealingLR(_LRScheduler):
 34 |     """
 35 |     Sets the learning rate of each parameter group to follow a linear warmup schedule
 36 |     between warmup_start_lr and base_lr followed by a cosine annealing schedule between
 37 |     base_lr and eta_min.
 38 | 
 39 |     .. warning::
 40 |         It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
 41 |         after each iteration as calling it after each epoch will keep the starting lr at
 42 |         warmup_start_lr for the first epoch which is 0 in most cases.
 43 | 
 44 |     .. warning::
 45 |         passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
 46 |         It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
 47 |         :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
 48 |         epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
 49 |         train and validation methods.
 50 | 
 51 |     Example:
 52 |         >>> layer = nn.Linear(10, 1)
 53 |         >>> optimizer = Adam(layer.parameters(), lr=0.02)
 54 |         >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epoch=10, max_epoch=40)
 55 |         >>> #
 56 |         >>> # the default case
 57 |         >>> for epoch in range(40):
 58 |         ...     # train(...)
 59 |         ...     # validate(...)
 60 |         ...     scheduler.step()
 61 |         >>> #
 62 |         >>> # passing epoch param case
 63 |         >>> for epoch in range(40):
 64 |         ...     scheduler.step(epoch)
 65 |         ...     # train(...)
 66 |         ...     # validate(...)
 67 |     """
 68 | 
 69 |     def __init__(
 70 |         self,
 71 |         optimizer,
 72 |         warmup_epoch,
 73 |         max_epoch,
 74 |         warmup_start_lr=0.0,
 75 |         eta_min=1e-8,
 76 |         last_epoch=-1,
 77 |     ):
 78 |         """
 79 |         Args:
 80 |             optimizer (Optimizer): Wrapped optimizer.
 81 |             warmup_epoch (int): Maximum number of iterations for linear warmup
 82 |             max_epoch (int): Maximum number of iterations
 83 |             warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
 84 |             eta_min (float): Minimum learning rate. Default: 0.
 85 |             last_epoch (int): The index of last epoch. Default: -1.
 86 |         """
 87 |         self.warmup_epoch = warmup_epoch
 88 |         self.max_epoch = max_epoch
 89 |         self.warmup_start_lr = warmup_start_lr
 90 |         self.eta_min = eta_min
 91 | 
 92 |         super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
 93 | 
 94 |     def get_lr(self):
 95 |         """
 96 |         Compute learning rate using chainable form of the scheduler
 97 |         """
 98 |         if not self._get_lr_called_within_step:
 99 |             warnings.warn(
100 |                 "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
101 |                 UserWarning,
102 |             )
103 | 
104 |         if self.last_epoch == 0:
105 |             return [self.warmup_start_lr] * len(self.base_lrs)
106 |         elif self.last_epoch < self.warmup_epoch:
107 |             return [
108 |                 group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epoch - 1)
109 |                 for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
110 |             ]
111 |         elif self.last_epoch == self.warmup_epoch:
112 |             return self.base_lrs
113 |         elif (self.last_epoch - 1 - self.max_epoch) % (2 * (self.max_epoch - self.warmup_epoch)) == 0:
114 |             return [
115 |                 group["lr"]
116 |                 + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epoch - self.warmup_epoch))) / 2
117 |                 for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
118 |             ]
119 | 
120 |         return [
121 |             (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epoch) / (self.max_epoch - self.warmup_epoch)))
122 |             / (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epoch - 1) / (self.max_epoch - self.warmup_epoch)))
123 |             * (group["lr"] - self.eta_min)
124 |             + self.eta_min
125 |             for group in self.optimizer.param_groups
126 |         ]
127 | 
128 |     def _get_closed_form_lr(self):
129 |         """
130 |         Called when epoch is passed as a param to the `step` function of the scheduler.
131 |         """
132 |         if self.last_epoch < self.warmup_epoch:
133 |             return [
134 |                 self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epoch - 1)
135 |                 for base_lr in self.base_lrs
136 |             ]
137 | 
138 |         return [
139 |             self.eta_min
140 |             + 0.5
141 |             * (base_lr - self.eta_min)
142 |             * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epoch) / (self.max_epoch - self.warmup_epoch)))
143 |             for base_lr in self.base_lrs
144 |         ]
145 | 
146 | 
147 | class LinearWarmupMultiStepLR(_LRScheduler):
148 |     """Decays the learning rate of each parameter group by gamma once the
149 |     number of epoch reaches one of the milestones. Notice that such decay can
150 |     happen simultaneously with other changes to the learning rate from outside
151 |     this scheduler. When last_epoch=-1, sets initial lr as lr.
152 | 
153 |     Args:
154 |         optimizer (Optimizer): Wrapped optimizer.
155 |         milestones (list): List of epoch indices. Must be increasing.
156 |         gamma (float): Multiplicative factor of learning rate decay.
157 |             Default: 0.1.
158 |         last_epoch (int): The index of last epoch. Default: -1.
159 |         verbose (bool): If ``True``, prints a message to stdout for
160 |             each update. Default: ``False``.
161 | 
162 |     Example:
163 |         >>> # Assuming optimizer uses lr = 0.05 for all groups
164 |         >>> # lr = 0.05     if epoch < 30
165 |         >>> # lr = 0.005    if 30 <= epoch < 80
166 |         >>> # lr = 0.0005   if epoch >= 80
167 |         >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
168 |         >>> for epoch in range(100):
169 |         >>>     train(...)
170 |         >>>     validate(...)
171 |         >>>     scheduler.step()
172 |     """
173 | 
174 |     def __init__(
175 |         self,
176 |         optimizer,
177 |         milestones,
178 |         gamma=0.1,
179 |         last_epoch=-1,
180 |         warmup_epoch=0,
181 |         warmup_start_lr=0,
182 |     ):
183 |         self.milestones = Counter(milestones)
184 |         self.gamma = gamma
185 |         self.warmup_epoch = warmup_epoch
186 |         self.warmup_start_lr = warmup_start_lr
187 |         super(LinearWarmupMultiStepLR, self).__init__(optimizer, last_epoch)
188 | 
189 |     def get_lr(self):
190 |         if not self._get_lr_called_within_step:
191 |             warnings.warn(
192 |                 "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
193 |             )
194 | 
195 |         # last epoch actually means last iter
196 |         if self.last_epoch == self.warmup_epoch:
197 |             return self.base_lrs
198 |         elif self.last_epoch == 0:
199 |             return [self.warmup_start_lr] * len(self.base_lrs)
200 |         elif self.last_epoch < self.warmup_epoch:
201 |             return [
202 |                 group["lr"] + (base_lr - self.warmup_start_lr) / self.warmup_epoch
203 |                 for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
204 |             ]
205 |         elif self.last_epoch in self.milestones:
206 |             return [group["lr"] * self.gamma for group in self.optimizer.param_groups]
207 |         else:
208 |             return [group["lr"] for group in self.optimizer.param_groups]
209 | 
210 |     def _get_closed_form_lr(self):
211 |         milestones = list(sorted(self.milestones.elements()))
212 |         return [base_lr * self.gamma ** bisect_right(milestones, self.last_epoch) for base_lr in self.base_lrs]
```

## Answer

Nonce：`DUCA-H65-60-LR-CURRICULUM-PRO-v001-20260824`

# DUCA H65 60-epoch 学习率与课程速率科学裁决

## 一、唯一科学裁决

**SCIENTIFIC_DISPOSITION：REVISE**

保留 H65 的全部模型、选帧、物理时间、检测器和评估合同；废止 `20+40` 作为可接受的 60-epoch 压缩方案。下一步不改峰值 LR、不改任何参数组基准 LR、不改模型，只复用原始 Stage‑1 epoch‑29 EMA，以两个 30-epoch Stage‑2 调度器实验完成 LR 归因。课程速率不与调度器同时变化；只有调度器归因完成后，才允许一个且仅一个 `3000 → 2000` 的课程速率实验。

核心判断是：

> `20+40` 的下降不是“模型容量不足”，也不能简单归咎于少了 30 个终末 epoch。它同时削弱了 Stage‑1 起点、缩短了每个课程状态得到的绝对优化时间，并让 Stage‑2 在第 4000 次更新附近把所有参数组拉向共同的绝对 LR 下限。现有证据首先要求修正优化时钟，而不是提高峰值 LR。

---

# 二、因果诊断

## 2.1 按预期贡献排序

| 排名 | 机制                                            | 判断                               | 证据强度与解释                                                                                                                                                          |
| -- | --------------------------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | **Stage‑1 表征稳定不足，且 Stage‑2 重置优化器后无法继承优化动量**   | 最可能是大幅下降的首要来源之一                  | `30/3000 → 20/2000` 不仅少 1000 次更新，还把 Stage‑1 warmup 从 3 epoch 改成 2 epoch。进入 Stage‑2 时只加载 epoch‑19 EMA 模型，而优化器、调度器和 AMP 状态全部重置，因此较弱的表示不能靠继承动量补偿。                   |
| 2  | **Stage‑2 余弦 horizon 过短，导致有效 LR 暴露显著下降**      | 与 Stage‑1 并列为主要原因；这是本轮首先可干净验证的因素 | 40-epoch 压缩臂虽然有 4000 次 Stage‑2 更新，但归一化累计 LR 暴露低于原始日程的前 3000 次 Stage‑2 更新。末段还被共同绝对 `eta_min` 拉平。                                                                  |
| 3  | **Stage‑2 总联合更新从 6000 降到 4000**               | 明显贡献，但不足以单独解释全部 2.6609 pp        | 原始曲线在 Stage‑2 epoch 29 后仍从约 63.795 提升到 epoch 39 的约 64.48，并在 epoch 49 附近达到约 65.65。联合适配确实尚未完成。但原始和压缩日程都观察到 4000 次 Stage‑2 更新时，二者仍相差约 2.02 pp，因此少 2000 次末段更新不是唯一原因。 |
| 4  | **课程和 detector feedback 在绝对 update time 中过快** | 次要但仍需一次隔离验证                      | 压缩课程与调度器大体同时按 2/3 缩放；在课程完成点，两条日程的 LR 因子反而近似。因此主要问题不是“课程与 LR 相位完全错位”，而是每种课程状态得到的绝对更新数太少，并且从更弱的 Stage‑1 起点进入。                                                      |
| 5  | **EMA lag 或终点 checkpoint 规则**                 | 低贡献，不足以解释主差距                     | 两臂都按预注册 terminal EMA 评价，没有 checkpoint 选择不公平。原始诊断峰值约 65.65 到 terminal 65.1257 的差约 0.52 pp，远小于压缩损失 2.6609 pp。EMA 需要监控，但不是首因。                                       |

---

## 2.2 当前调度器揭示的定量问题

对任意参数组 (i)，当前 warmup 后的余弦形式为：

[
\eta_i(s)=\eta_{\min}
+\frac{1}{2}\left(\eta_i^{\mathrm{base}}-\eta_{\min}\right)
\left[
1+\cos\left(
\pi\frac{s-W}{H-W}
\right)
\right],
]

其中：

* (s)：调度器 update index；
* (W)：warmup updates；
* (H)：余弦 horizon；
* 当前 `eta_min=1e-8`，且对所有参数组相同。

以下“累计 LR 暴露”是将每次更新的 LR 除以该组 base LR 后求和。它不是等价于训练效果的论文指标，但能严格说明“4000 次更新不一定比 3000 次更新具有更大的优化剂量”。

| 日程                            | updates |    warmup / horizon | 终点相对 LR 因子 | 累计相对 LR 暴露，约 |
| ----------------------------- | ------: | ------------------: | ---------: | -----------: |
| 原始 Stage‑1                    |    3000 |          300 / 3000 |        近 0 |       1500.5 |
| 压缩 Stage‑1                    |    2000 |          200 / 2000 |        近 0 |       1000.5 |
| 原始 Stage‑2 前 30 epoch         |    3000 |          300 / 6000 |     0.5413 |       2404.3 |
| 原始 Stage‑2 前 40 epoch         |    4000 |          300 / 6000 |     0.2742 |       2809.8 |
| 压缩 Stage‑2 完整 40 epoch        |    4000 |          300 / 4000 |        近 0 |       2000.5 |
| 拟议 `LongCosine-90` 前 30 epoch |    3000 |          300 / 9000 |     0.7806 |       2646.2 |
| 拟议 `RelativeCosineHold-25`    |    3000 | 300 / 6000，2500 后保持 |     0.6753 |       2437.4 |

这直接解释了一个表面悖论：

> 压缩 `20+40` 虽然比原始总 epoch 60 的 `30+30` 多了 1000 次 Stage‑2 更新，但前者完整 Stage‑2 的累计 LR 暴露约为 2000.5，后者前 3000 次更新约为 2404.3。压缩臂实际上获得了更少的高效联合优化剂量。

### 绝对 `eta_min` 的额外问题

Stage‑2 中可见的 selector LR 至少包括：

* coarse trunk：`1e-5`
* action head：`2e-5`
* transition scorer：`5e-5`

基准比例是 (1:2:5)。当前余弦达到 horizon 时，三组都变成 `1e-8`，比例变成 (1:1:1)。在最后几十次更新内，这种比例坍缩已经很明显。因此下一轮必须使用共同的**乘法因子**：

[
\eta_i(s)=\eta_i^{\mathrm{base}}f(s),
]

而不是给所有参数组加同一个绝对 floor。

---

## 2.3 为什么“课程过快”不是当前第一嫌疑

压缩日程同时把：

* semantic transition：`3000 → 2000`
* detector feedback：`1000+2000 → 667+1333`
* cosine horizon：`6000 → 4000`

大致按相同比例缩放。

按所附 scheduler 公式，在关键位置：

| 事件                       |        原始日程 LR 因子 |        压缩日程 LR 因子 |
| ------------------------ | ----------------: | ----------------: |
| detector feedback 开始     | step 1000：约 0.963 |  step 667：约 0.976 |
| curriculum / feedback 完成 | step 3000：约 0.541 | step 2000：约 0.564 |

因此，压缩日程并没有让 detector feedback 在一个显著更低或更高的 LR 相位进入。真正变化的是：

1. 进入课程时的 Stage‑1 表征更弱；
2. 每个课程区间得到的绝对更新数减少；
3. 课程完成后剩余训练处于快速衰减区，并最终归零。

所以课程速率必须测试，但应排在调度器归因之后，不能与调度器同时改变。

---

# 三、候选调度器裁决

## 3.1 `LongCosine-90`：准入为诊断臂

Stage‑2 仍只运行 3000 次成功更新，但余弦 horizon 固定为 9000 次更新。

令 (W=300)，对 (300\le s\le3000)：

[
f_{\mathrm{LC90}}(s)
====================

\frac{1}{2}
\left[
1+\cos\left(
\pi\frac{s-300}{9000-300}
\right)
\right].
]

每个参数组：

[
\eta_i(s)=\eta_i^{\mathrm{base}}f_{\mathrm{LC90}}(s).
]

终点：

[
f_{\mathrm{LC90}}(3000)=0.7805935327.
]

作用：最直接检验“压缩失败是否主要由过早衰减导致”。

风险：终点 LR 较高，可能仍在持续变化，EMA lag 可能变大。因此它是强诊断臂，不预设为最终最佳日程。

---

## 3.2 原提议 `CosineFloorTail-0.2`：拒绝运行

原提议为：

* 3 epoch warmup；
* epoch 25 前余弦降至 `0.2×base_lr`；
* 最后 5 epoch 保持 `0.2×`。

它不适合当前第一轮，原因不是 `0.2` 一定错误，而是它不能检验当前最重要的因果问题。

按该形状：

* step 2000 的相对 LR 约为 0.298；
* terminal 为 0.2；
* 3000 次更新累计相对 LR 暴露约为 1570。

这比已经存在的原始 `30+30` anchor 的 2404 明显更低。若它失败，无法区分：

* 30-epoch update budget 本身不足；
* 0.2 floor 太低；
* decay 开始太早；
* curriculum 尚未完成时 LR 已经过低。

因此该臂的信息价值不足，应当删除，而不是把它加入调参网格。

---

## 3.3 更好的替代：`RelativeCosineHold-25`

该臂严格匹配原始 60-epoch Stage‑2 余弦形状到第 2500 次成功更新，此后不再衰减，保持最后 500 次更新。

令：

[
f_{60}(s)
=========

\frac{1}{2}
\left[
1+\cos\left(
\pi\frac{s-300}{6000-300}
\right)
\right].
]

则：

[
f_{\mathrm{RCH25}}(s)=
\begin{cases}
f_{60}(s), & 300\le s\le2500,[4pt]
f_{60}(2500), & 2500<s\le3000.
\end{cases}
]

其中：

[
f_{60}(2500)=0.6753187776.
]

参数组 LR：

[
\eta_i(s)=\eta_i^{\mathrm{base}}f_{\mathrm{RCH25}}(s).
]

该替代比任意指定 `0.2` 更有因果价值：

* 前 25 epoch 保持原始日程的已知优化形状；
* 只测试最后 5 epoch 是否应该停止衰减；
* 参数组 LR 比例严格不变；
* 给 EMA 和 selector/detector 协同适配一个固定 LR 尾段；
* 若相对原始 `30+30` anchor 有提升，可直接归因于末段衰减策略，而不是峰值 LR 或课程变化。

---

# 四、最小新运行矩阵

## 4.1 只读复用，不得重跑的锚点

| 锚点                                                         | 用途                                                                 |
| ---------------------------------------------------------- | ------------------------------------------------------------------ |
| 原始 `30+60` terminal epoch‑59 EMA：65.1257 / mAP@0.7 43.3137 | H65 保真目标                                                           |
| 原始 Stage‑2 epoch‑29 EMA：Avg 约 63.795                       | 同总 epoch 60 的旧调度器 anchor；必须从现有 official-eval receipt 补封其精确 mAP@0.7 |
| 压缩 `20+40` terminal：62.4648 / 39.9434                      | 已完成负面压缩结果，不得重复                                                     |

`30+30` 的精确 Avg 和 mAP@0.7 必须在 PRE_RUN 前从已有 checkpoint/prediction/evaluator receipt 封存；缺少 mAP@0.7 不能靠重训补齐。

## 4.2 新运行

| Run                      | Stage‑1 checkpoint                                  |                            Stage‑2 |                Warmup | Scheduler                                                        | Curriculum / feedback                                                              | Seed | Terminal rule                                                                  | 目的                     |
| ------------------------ | --------------------------------------------------- | ---------------------------------: | --------------------: | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ---: | ------------------------------------------------------------------------------ | ---------------------- |
| `R1-LC90-C3000`          | 原始 Stage‑1 epoch‑29 `state_dict_ema`，路径与 SHA‑256 固定 | 30 epoch / 3000 successful updates | 3 epoch / 300 updates | `LongCosine-90`；horizon 9000；无绝对 floor；terminal factor 0.7805935 | 原始 transition 3000；feedback 1000 warmup + 2000 transition                          | 3407 | 每 5 epoch checkpoint；primary=`Stage2 epoch29 state_dict_ema`；同时保存 final online | 检验全局减慢 LR 衰减是否恢复 H65   |
| `R1-RCH25-C3000`         | 同上                                                  |                          30 / 3000 |               3 / 300 | 原始 6000-horizon 曲线到 step 2500；之后保持 factor 0.6753188 至 step 3000  | 同上                                                                                 | 3407 | 同上                                                                             | 检验仅停止最后 5 epoch 衰减是否足够 |
| `R2-<winner>-C2000`，条件运行 | 同上                                                  |                          30 / 3000 | 与 Round‑1 winner 完全相同 | 使用 Round‑1 选定调度器，不得再改任何 scheduler 参数                             | transition 2000；feedback 667 warmup + 1333 transition；最后 1000 updates 为完整 joint 状态 | 3407 | 同上                                                                             | 唯一 curriculum-rate 归因  |

Round‑1 中不得加入 curriculum-rate 变化。Round‑2 不得同时修改 warmup、floor、horizon 或峰值 LR。

选择 `2000 / 667+1333`，而不是新增 `1500 / 500+1000`，是因为前者已经是实际完成过的压缩课程定义；本轮只需要在恢复原始 Stage‑1 和选定 scheduler 后隔离其作用。再引入 1500 会形成新的速率网格。

---

# 五、停止与选择规则

以下是开发期的**工程保真门**，不是统计等效性论文声明。

定义：

* (A_{30})：现有原始 Stage‑2 epoch‑29 EMA 的精确 Avg-mAP；
* (H_{30})：同一 checkpoint 的精确 mAP@0.7；
* (A_{90}=65.1257)；
* (H_{90}=43.3137)。

## 5.1 Round‑1 调度器选择

一个新 scheduler 只有同时满足以下条件，才能替换原始 60-horizon anchor：

[
\mathrm{Avg}*{new} \ge A*{30}+0.50
]

并且

[
\mathrm{mAP@0.7}*{new} \ge H*{30}-0.20.
]

选择规则：

1. 仅一个新臂通过：选该臂。
2. 两个都通过，Avg 差至少 0.10 pp：选 Avg 更高者。
3. 两者 Avg 差小于 0.10 pp：选 `RelativeCosineHold-25`，因为它离原始日程更近，并具有明确的固定尾段。
4. 两者都未通过：原始 60-horizon `30+30` anchor 保持为 scheduler winner；不得为此再创建第三个 scheduler。

## 5.2 Round‑1 后何时停止、何时测试课程

### 直接停止并冻结 60-epoch 日程

若 Round‑1 winner 同时满足：

[
\mathrm{Avg}\ge64.6257
]

和

[
\mathrm{mAP@0.7}\ge42.8137,
]

即两项都在原始 terminal 的 0.50 pp 工程保真带内，则不再测试课程速率。直接冻结该 scheduler 与原始 `3000 / 1000+2000` curriculum。

### 进入唯一课程实验

若 Round‑1 winner 通过全部健康和身份门，但没有同时进入上述保真带，则只运行一个 `C2000` 对照。

### 立即终止本包

出现任一情况，不得运行课程实验：

* Stage‑1 checkpoint、模型 key、optimizer group membership 或 base LR 不一致；
* scheduler clock 与 successful optimizer update 不一致；
* resume 结果不忠实；
* 非有限 loss/gradient；
* 新 scheduler 意外越过 3000 updates 或发生 cosine rebound；
* final EMA、online 或 official evaluator 身份不完整；
* mAP@0.7 的原始 `30+30` anchor 无法从已有证据封存。

## 5.3 Round‑2 课程选择和最终终止

`C2000` 只有满足：

[
\mathrm{Avg}*{C2000}
\ge
\mathrm{Avg}*{C3000}+0.50
]

且

[
\mathrm{mAP@0.7}*{C2000}
\ge
\mathrm{mAP@0.7}*{C3000}-0.20
]

才替换原始课程。否则保留 `C3000`。

完成该一次比较后，最终选定的 60-epoch 日程仍必须达到：

* Avg-mAP ≥ 64.6257；
* mAP@0.7 ≥ 42.8137。

任何一项未达到，即终止 60-epoch H65 保真压缩，正式保留原始 `30+60` 日程。不得继续：

* 提高所有参数组 LR；
* 扫描多个 floor；
* 修改 Stage‑2 warmup；
* 尝试 1500/2500 等其他 curriculum；
* 根据中间 checkpoint 选最佳；
* 自动授权单参数组 LR 调整。

当前证据不足以授权任何单参数组 LR 改动。

---

# 六、必须采集的指标

## 6.1 模型质量与日程选择证据

只有以下 terminal 指标可以改变调度器/课程决定：

* terminal epoch‑29 EMA official Avg-mAP；
* terminal epoch‑29 EMA mAP@0.7；
* 完整 mAP@0.3、0.4、0.5、0.6、0.7 向量；
* 同一 terminal online state 的指标仅作为 EMA 健康诊断，不得取代 EMA；
* successful optimizer updates 必须恰为 3000。

中间每 5 epoch 的官方评价只能画学习曲线，不得选择 checkpoint。

## 6.2 健康诊断，不是 claim evidence

### LR 与更新时钟

逐 successful update 记录：

* scheduler step；
* successful optimizer update count；
* curriculum schedule step；
* 每个参数组的 base LR 和实际 LR；
* 相对因子 (f(s))；
* 任意 AMP retry、skip 或 replay。

相对调度器在非零 warmup 点后必须满足：

[
\frac{\eta_i(s)}{\eta_j(s)}
===========================

\frac{\eta_i^{base}}{\eta_j^{base}}
]

至数值容差 `1e-12`。

### 梯度与实际更新

至少分组记录：

* selector coarse trunk；
* selector action head；
* transition scorer；
* VideoMAE backbone；
* adapter；
* ActionFormer detector/head。

每 100 次成功更新汇总：

* clipping 前梯度 L2 norm：median、p90、p99；
* clipping 后梯度 norm；
* 参数 norm；
* 相对更新幅度 (|\Delta\theta|_2/|\theta|_2)；
* clipping 触发率；
* nonfinite 或 AMP replay 次数。

在 steps `666/667`、`999/1000`、`1999/2000`、`2499/2500`、`2999/3000` 附近必须保留细粒度记录。

### selector 状态

在相同固定诊断视频/窗口上记录：

* H65 已有归一化采样 density/rate 的归一化熵；
* top 10% 时间位置的质量占比；
* 有效支持大小；
* 相对 canonical uniform 的平均绝对位移；
* 位移 p95、最大位移；
* 相邻物理间隔分布；
* checkpoint 间所选帧集合和位移变化。

不得为了诊断另定义一个带温度的 softmax 或改写 selector。

### EMA 健康

在 terminal 同时保存但不混合选择：

* online official metrics；
* EMA official metrics；
* (|\theta_{\mathrm{online}}-\theta_{\mathrm{EMA}}|/|\theta_{\mathrm{online}}|)；
* 最后 5 epoch 的 online/EMA loss 和 selector-state 漂移。

这些量用于判断 `LongCosine-90` 是否仍处于高 LR 漂移，但不得以 online 胜过 EMA 为由改变预注册 checkpoint。

---

# 七、Builder 最小改动面

## 7.1 允许修改

1. `opentad/cores/scheduler.py`
2. 两个新的 Round‑1 Stage‑2 配置文件
3. 一个在 Round‑1 结束后实例化的 Round‑2 配置模板
4. scheduler 单元测试
5. 必要时增加一个只读 diagnostics hook 及其测试

## 7.2 禁止修改

* selector 源码和 selector loss；
* density/transition/actionness 定义；
* VideoMAE、Adapter、ActionFormer；
* optimizer family；
* 参数组 membership；
* 任意参数组 base LR；
* augmentation、seed、数据顺序；
* NMS、official evaluator；
* selected-rank/physical-time 路径；
* K=384；
* Stage‑1 checkpoint；
* final/final-EMA 规则。

## 7.3 相对 LR 实现要求

新增 scheduler 必须从 `base_lrs` 闭式计算：

```python
lr_i = base_lr_i * factor(successful_update_index)
```

禁止：

```python
lr_i = absolute_eta_min + ...
```

也禁止从当前 `group["lr"]` 递归累计，以免浮点漂移和 resume 分叉。

需要修正或隔离当前 `build_scheduler` 的两个风险：

1. `cfg.pop("type")` 和 epoch 乘法会原地修改传入配置；必须先复制配置。
2. 调度器不得在超过冻结 horizon/total updates 后产生 cosine rebound。超出 3000 successful updates 必须 fail closed。

新 scheduler 最好直接使用：

* `warmup_updates=300`
* `cosine_horizon_updates=9000` 或 `6000`
* `hold_after_update=None` 或 `2500`
* `total_successful_updates=3000`

而不是再次依赖未经验证的 `dataloader_len × epoch` 推导。

## 7.4 配置一致性

压缩 Stage‑2 子配置引入了 `duca_stage2_transition_steps=2000`，但其基类仍定义 `duca_stage2_half_steps=3000`。即使模型 schedule 已使用新变量，解析后的配置中也可能同时保留两个不同的顶层值。

新配置必须只有一个规范字段，并保证以下字段一致：

* `workflow.end_epoch=30`
* `total_epochs=30`
* `max_updates=3000`
* `expected_successful_optimizer_updates=3000`
* scheduler total update contract
* terminal epoch `29`
* checkpoint interval `5`

同时确认实际 engine 读取的是 `workflow.checkpoint_interval` 还是顶层 `checkpoint_interval_epochs`，不得仅在未消费的别名里写入 “5”。

## 7.5 必须通过的测试

* LR 公式关键点：steps 0、299、300、1000、2500、3000；
* LC90 terminal factor 精确匹配 `0.7805935327`；
* RCH25 terminal factor 精确匹配 `0.6753187776`；
* 所有参数组比例保持；
* config 构建两次不发生 mutation 或二次乘 100；
* scheduler 不因 extra step 重新升高；
* AMP skipped update 不推进 LR 或 curriculum；
* Stage‑2 从 Stage‑1 checkpoint 初始化与 Stage‑2 内 resume 严格区分；
* 在 toy model 上比较：

  * uninterrupted 3000-step；
  * 在 step 1737 checkpoint/resume 后完成 3000-step；
* 两条路径的模型参数、optimizer state、scheduler state、EMA、RNG、LR trace、curriculum step 完全一致或达到冻结的逐字段数值容差；
* 579 个模型 key、shape 和 optimizer group membership 与原 H65 完全一致。

---

# 八、Independent Critic 检查

Critic 必须独立验证：

1. Round‑1 两臂唯一的科学差异是 scheduler。
2. 两臂使用同一 epoch‑29 Stage‑1 EMA SHA。
3. 579 个 key 相同不能替代 optimizer group membership 审计；后者也必须相同。
4. base LR 数组逐项相同。
5. curriculum 权重、transition 3000、feedback 1000+2000 完全相同。
6. successful-update clock、scheduler clock、loss-schedule clock逐次对齐。
7. 绝对 `eta_min` 没有通过继承重新进入新 scheduler。
8. `duca_stage2_half_steps` 等陈旧字段不存在歧义。
9. `workflow.end_epoch`、`max_updates`、实际循环和 checkpoint epoch 一致。
10. final EMA 是唯一 primary checkpoint；intermediate evaluation 不选择模型。
11. resume 不改变数据顺序、RNG、EMA 或调度器相位。
12. Stage‑1 未被重新训练，旧 `20+40` 和 `30+30` 未被重复。
13. selector、VideoMAE、detector、loss、NMS、evaluator、K 和物理时间路径没有变化。
14. `LongCosine-90` 的高 terminal LR 不被错误描述为“已收敛”；EMA/online 诊断标签正确。
15. `RelativeCosineHold-25` 前 2500 updates 的 factor trace符合冻结公式。

任一 P0 身份差异均阻止提交，而不是作为新的实验变量接受。

---

# 九、Evaluator PRE_RUN 门

在任何 N16R4 提交之前，Evaluator 必须封存：

* clean revision `04c35a3b76897e6c1569eeede41ed3aecaf7f854` 及 scheduler patch identity；
* 两个完全解析后的 config；
* Stage‑1 epoch‑29 EMA 路径、SHA‑256、checkpoint epoch、state key；
* 原始 `30+30` anchor 的精确 Avg 和 mAP@0.7；
* seed `3407`；
* Stage‑2 3000 successful update 合同；
* scheduler expected LR trace；
* curriculum expected trace；
* checkpoint interval 5；
* final 与 final-EMA 输出路径；
* terminal EMA primary rule；
* official THUMOS14 split、class map、NMS 和 evaluator identity；
* 新鲜且不存在的输出根目录；
* literal launcher argv、cwd、环境和资源身份；
* resume/invalidation 规则；
* 不允许读取中间 metric 后改变课程或 scheduler；
* 每个窗口实际 selected K、unique K、VideoMAE executed K 均为 384。

训练成本应报告：

* Stage‑1 checkpoint 为复用，但完整方法训练成本仍按历史 3000 Stage‑1 updates 加本次 3000 Stage‑2 updates说明；
* 本次新增 GPU wall time、AMP retries 和成功更新数；
* scheduler/diagnostic hook 开销；
* 不得把复用 Stage‑1 描述为“训练成本为零”。

推理时模型结构和 K 不变，因此 scheduler 不产生新的推理机制或效率 claim；仍需检查没有因实现错误改变 full-stack execution。

---

# 十、论文与主张边界

这组实验只研究：

> 在不改变 H65 机制、模型、K、检测器和评估器的条件下，能否通过正确的 Stage‑2 LR horizon、末段 LR 和课程速率，将原始 `30+60` 日程压缩为 `30+30` 而基本保留性能。

它不支持以下主张：

* 新 DUCA 机制优于已有方法；
* H65 selector 的科学优势被重新证明；
* dynamic K 有效；
* TrueTime、cliplet、Query bridge 或知识传递有效；
* 60-epoch 日程优于 90-epoch 日程；
* 单 seed 结果构成论文级统计证据；
* 中间 epoch 49 的诊断峰值可以取代 terminal checkpoint；
* 调度器优化带来推理成本下降。

---

# 十一、下一责任合同

```text
next_owner:
  Builder

next_action:
  在 clean H65 revision 上实现一个按 successful-update 驱动的
  per-group-relative cosine/hold scheduler；
  生成 R1-LC90-C3000 和 R1-RCH25-C3000 两个解析配置；
  完成公式、参数组比例、config 不变性、AMP-skip 时钟和 resume-fidelity 测试。
  不提交训练作业。

dependency:
  1. 原始 Stage-1 epoch-29 EMA checkpoint 路径与 SHA-256；
  2. 原始 Stage-2 epoch-29 EMA 的精确 official Avg 和 mAP@0.7 receipt；
  3. 579-key 及 optimizer-group membership 审计；
  4. Independent Critic PASS；
  5. Evaluator PRE_RUN_READY；
  6. 新鲜 N16R4 输出身份。

expected_return:
  - scheduler patch 与逐步公式 receipt；
  - 两个 resolved-config diff；
  - uninterrupted-versus-resume identity receipt；
  - exact expected LR traces；
  - Critic closure；
  - Evaluator PRE_RUN manifest；
  - 随后两个 Round-1 terminal-EMA 结果、mAP@0.7、完整 LR/gradient/
    selector/EMA 健康诊断；
  - 按冻结规则唯一输出：
      a) 冻结一个 30+30 scheduler；
      b) 运行唯一 C2000 curriculum 对照；
      c) 或终止 60-epoch H65 保真压缩并保留 30+60。
```

最终执行上限是 **两次 Round‑1 新训练，加至多一次条件式 curriculum 训练**。该上限之后没有 floor、warmup、transition 或全局 LR 网格。
