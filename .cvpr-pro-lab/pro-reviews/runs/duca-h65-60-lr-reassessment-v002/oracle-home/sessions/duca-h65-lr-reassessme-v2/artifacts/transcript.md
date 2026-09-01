# Oracle Browser Transcript

Conversation: https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697-duca/c/6a8c0302-b5e4-83ea-b87e-3bcaa8130dde

## Prompt

请完整阅读附加任务和冻结配置。作为 DUCA 独立 Scientific First-Author Agent，以最严厉审稿人视角给出唯一终稿。Nonce: DUCA-H65-60-LR-SCHEDULE-REASSESS-v002-20260824

### File: .cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_H65_60_LR_SCHEDULE_REASSESSMENT-v002.md
Lines: 1-77
```md
 1 | # DUCA H65：60 轮压缩性能下降与学习率日程复核
 2 | 
 3 | Nonce: `DUCA-H65-60-LR-SCHEDULE-REASSESS-v002-20260824`
 4 | 
 5 | 你是 DUCA 项目的独立 Scientific First-Author Agent、训练动力学专家和最严厉的实验审稿人。请在同一个回答中完成因果诊断、现有实验设计复核和终态结果后的决策树。不得把路线选择交回人类或 Codex；必须给出唯一的 `CONTINUE`、`REVISE`、`PIVOT` 或 `STOP`。
 6 | 
 7 | ## 1. 项目与代码身份
 8 | 
 9 | - Exact ChatGPT Project: `DUCA`, `g-p-6a796fef9a00819194024cf1de3bd697`
10 | - GitHub: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
11 | - 历史 H65 evidence commit: `42dba3f90b37243e7965d18b6707e88e81bf7109`
12 | - 课程压缩代码 revision: `04c35a3b76897e6c1569eeede41ed3aecaf7f854`
13 | - 当前 schedule-only attribution clean revision: `ae3642a138c5b2e1ac2daad75a6d43d17cdb6c2f`
14 | - 当前实验只讨论 H65 训练日程；First-Mixing SingleClock 是独立的时间表示实验，禁止混为因果解释。
15 | 
16 | ## 2. 冻结 H65 模型与公平性合同
17 | 
18 | H65 使用低成本 ASFormer 语义侦察器学习动作性、状态转移和边界证据，经确定性 sampling-rate/density transport 形成非均匀逐帧位置；不是小模型直接预测 frame index。固定 `K=384`，选中的 RGB 帧按原时间排序，送入相同 VideoMAE-S、Adapter、ActionFormer、loss、NMS、THUMOS14 training/validation split 与 evaluator。
19 | 
20 | 当前 schedule-only attribution 中，下列内容全部冻结不变：模型结构、K384 非均匀逐帧选择、selected RGB、数据、seed `3407`、损失终值、检测器、评估器、AdamW、weight decay、backbone freeze、参数组成员和各组基础 LR。任何新建议不得借机改变 selector、Query、Bridge、dynamic K、连续 cliplet、loss 权重终值或基础 LR。
21 | 
22 | ## 3. 真实终态证据
23 | 
24 | 1. 历史 `30+60`：Stage-1 exact-uniform K384 训练 30 epoch/3000 successful updates；Stage-2 learned H65 训练 60 epoch/6000 successful updates。终态 epoch-59 EMA：Avg-mAP `65.1257`，mAP@0.7 `43.3137`。
25 | 2. 失败的压缩版 `20+40`：Stage-1 20 epoch/2000 updates；Stage-2 40 epoch/4000 updates。终态 epoch-39 EMA：Avg-mAP `62.4648`，mAP@0.7 `39.9434`。
26 | 3. 差值：Avg-mAP `-2.6609 pp`，mAP@0.7 `-3.3703 pp`。
27 | 4. Stage-1 终点也明显不同：30 epoch `59.4231`，20 epoch `49.5389`。因此失败的 `20+40` 不是“只少训 30 epoch”：它从不成熟的 epoch-19 EMA handoff，且同时压缩 LR、semantic/policy transition、feedback clock 与 full-joint tail。
28 | 5. 历史 `65.696` 是改造 physical-grid、协议未完全匹配的探索锚点，不得替代上述 matched H65 终态。
29 | 6. 所有数字均为单 seed，且历史 checkpoint 缺完整 RNG/DataLoader 恢复状态；它们可用于日程归因，但不是论文级多 seed 主张。
30 | 
31 | ## 4. 已接受的上一轮 Pro 方案与当前正在运行的严格归因
32 | 
33 | 上一轮独立 Pro 已给出 `REVISE`，要求保留成熟的 30-epoch Stage-1 epoch-29 EMA，不提高任何基础 LR，只运行 30-epoch/3000-update Stage-2 的两个 schedule-only 臂：
34 | 
35 | ### A. `AM-RPCH25`
36 | 
37 | - successful update 1–500：历史绝对 warmup 到各组 base LR；
38 | - 501–1500：保持 `1.0× base LR`；
39 | - 1501–2500：相对 cosine 从 `1.0×` 降到 `0.25×`；
40 | - 2501–3000：保持 `0.25× base LR`；
41 | - 参数组 LR 比例始终不变；禁止统一绝对 `eta_min`。
42 | 
43 | ### B. `LongCosine-H6000`
44 | 
45 | - 保持历史 6000-update cosine horizon；
46 | - 只运行前 3000 successful updates，因此终点保留较高非零 LR；
47 | - 其目的只是区分“相对 plateau/cosine/hold”与“历史长 horizon 截断”的 LR 形状，不是新模型。
48 | 
49 | 两臂共同保留成熟 Stage-1 epoch-29 EMA、2000-step semantic/policy transition、1000-step detector-feedback warmup + 1000-step transition、1000-step full-joint tail；每 5 epoch 保存可恢复 checkpoint；固定 epoch-29 final 与 final-EMA，不按中间 validation 选择。
50 | 
51 | 远端 N16R4 Jobs `1252979`、`1252980` 正在完整训练，PRE_RUN、恢复检查和路径 smoke 已通过，但尚无终态 validation、mAP 或成本结论。禁止读取中途 loss 后猜测优胜者，也禁止为了本次讨论追加第三个训练臂。
52 | 
53 | ## 5. 需要你严厉回答的问题
54 | 
55 | 1. 重新核验性能下降的证据排序：Stage-1 handoff 不成熟、Stage-2 更新量减半、LR 提前衰减、semantic supervision 退火、feedback timing、joint tail、EMA lag。哪些是确定事实，哪些只能由终态/已有曲线区分？不得编造数值归因。
56 | 2. 判断当前 A/B 两臂是否真正做到“模型不变、基础 LR 不变、只做 Stage-2 schedule attribution”；指出任何隐藏混杂项或无效比较。
57 | 3. 判断 30+30（总 60 epoch）是否存在理论或经验上的无损恢复保证。明确 LR 形状能修复什么，缺失 3000 次 Stage-2 minibatch/optimizer/EMA exposure 又不能修复什么。
58 | 4. 给出终态结果后的唯一决策树：
59 |    - 若 A 或 B 恢复到历史锚点的可接受邻域，下一步是否只补 seeds，还是还需检查 schedule 稳定性？
60 |    - 若两者都明显低于历史锚点，但梯度稳定、终点曲线仍上升，应优先增加 Stage-2 exposure、单独微调某参数组 LR，还是承认 60-epoch 预算不可无损压缩？
61 |    - 若只有高 IoU 恶化、Avg-mAP 接近，应检查哪些边界/时间/selector 梯度现象？
62 | 5. 冻结可执行成功/失败阈值。请优先给出基于 matched historical anchor 的 Avg-mAP、mAP@0.7、训练稳定性与终端斜率判据；不得用中途最好 checkpoint。
63 | 6. 指定最少的只读诊断：历史 Stage-2 checkpoint 轨迹、各参数组 LR 曲线、未加权/加权语义 loss、梯度范数、selector 位移/熵、online-EMA gap。每项必须说明观察到什么会改变下一步。
64 | 7. 判断在 A/B 终态出来前是否应继续调参。若不应，明确 `HOLD_NEW_TUNING_UNTIL_TERMINAL`。
65 | 8. 训练日程恢复不是科学创新。请明确其结果到论文 claim 的边界。
66 | 
67 | ## 6. 必须返回的终稿
68 | 
69 | - 唯一科学裁决及一句话原因；
70 | - 性能下降的证据分层诊断；
71 | - 对 A/B 两臂的严格正确性/混杂审查；
72 | - 终态后的 if/then 决策树与停止规则；
73 | - 不超过一个后续动作；若当前证据不足，应明确等待哪些终态，而不是设计新实验；
74 | - Critic 与 Evaluator 必须验证的事实；
75 | - `next_owner / next_action / dependency / expected_return_at`。
76 | 
77 | 请用严肃、直接、外部评审可理解的中文作答。不要引入新的模型路线，不要重复 dense/uniform/random 对照，不要把正在运行的中间日志当作结果。
```

### File: ../OpenTAD_DUCA_H65_LRSchedule_20260824/configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py
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

### File: ../OpenTAD_DUCA_H65_LRSchedule_20260824/configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py
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

### File: ../OpenTAD_DUCA_H65_LRSchedule_20260824/configs/adatad/thumos/duca_h65_60_stage1_uniform20.py
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

### File: ../OpenTAD_DUCA_H65_LRSchedule_20260824/configs/adatad/thumos/duca_h65_60_stage2_transition20_joint20.py
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

### File: ../OpenTAD_DUCA_H65_LRSchedule_20260824/configs/adatad/thumos/duca_h65_60_stage2_am_rpch25.py
Lines: 1-113
```python
  1 | """H65-60 attribution: mature Stage-1 plus AM-RPCH25 Stage-2."""
  2 | 
  3 | _base_ = ["./duca_sampling_rate_curriculum_stage2_joint384.py"]
  4 | 
  5 | 
  6 | duca_stage2_transition_steps = 2000
  7 | duca_stage2_total_steps = 3000
  8 | 
  9 | duca_sampling_rate_contract = dict(
 10 |     route="DUCA_H65_60_STAGE2_AM_RPCH25",
 11 |     stage="mature_stage1_then_area_matched_relative_plateau_cosine_hold",
 12 |     stage1_initialization="full_uniform_k384_epoch29_ema_model",
 13 |     optimizer_scheduler_amp_state_reset=True,
 14 |     curriculum_only_change=True,
 15 |     model_change_allowed=False,
 16 | )
 17 | 
 18 | model = dict(
 19 |     frame_selector=dict(
 20 |         loss_weight_schedule=dict(
 21 |             _delete_=True,
 22 |             type="progressive_joint",
 23 |             shape="cosine",
 24 |             warmup_steps=0,
 25 |             transition_steps=duca_stage2_transition_steps,
 26 |             actionness=dict(
 27 |                 start=1.0,
 28 |                 end=0.25,
 29 |                 warmup_steps=0,
 30 |                 transition_steps=duca_stage2_transition_steps,
 31 |             ),
 32 |             transition=dict(
 33 |                 start=0.50,
 34 |                 end=0.10,
 35 |                 warmup_steps=0,
 36 |                 transition_steps=duca_stage2_transition_steps,
 37 |             ),
 38 |             transition_boundary=dict(
 39 |                 start=2.0,
 40 |                 end=0.25,
 41 |                 warmup_steps=0,
 42 |                 transition_steps=duca_stage2_transition_steps,
 43 |             ),
 44 |             policy_alpha=dict(
 45 |                 start=0.0,
 46 |                 end=1.0,
 47 |                 warmup_steps=0,
 48 |                 transition_steps=duca_stage2_transition_steps,
 49 |             ),
 50 |             detector_gradient=dict(
 51 |                 start=0.0,
 52 |                 end=0.25,
 53 |                 warmup_steps=1000,
 54 |                 transition_steps=1000,
 55 |             ),
 56 |             detector_contribution=dict(
 57 |                 start=0.0,
 58 |                 end=1.0,
 59 |                 warmup_steps=1000,
 60 |                 transition_steps=1000,
 61 |             ),
 62 |             asformer_adapt=dict(
 63 |                 start=0.0,
 64 |                 end=1.0,
 65 |                 warmup_steps=0,
 66 |                 transition_steps=duca_stage2_transition_steps,
 67 |             ),
 68 |         ),
 69 |     ),
 70 | )
 71 | 
 72 | scheduler = dict(
 73 |     _delete_=True,
 74 |     type="RelativeSuccessfulUpdateLR",
 75 |     mode="am_rpch25",
 76 |     max_epoch=30,
 77 |     total_updates=duca_stage2_total_steps,
 78 |     warmup_updates=500,
 79 |     plateau_updates=1000,
 80 |     decay_updates=1000,
 81 |     hold_updates=500,
 82 |     terminal_factor=0.25,
 83 |     horizon_updates=6000,
 84 | )
 85 | 
 86 | workflow = dict(
 87 |     training_profile="duca_h65_60_stage2_am_rpch25",
 88 |     checkpoint_interval=5,
 89 |     require_resumable_training_state=True,
 90 |     end_epoch=30,
 91 |     expected_train_batches_per_epoch=100,
 92 |     expected_successful_optimizer_updates=duca_stage2_total_steps,
 93 |     primary_checkpoint_epoch=29,
 94 |     primary_checkpoint_state_key="state_dict_ema",
 95 |     checkpoint_criterion="terminal_epoch_29_state_dict_ema",
 96 |     intermediate_validation_role="learning_curve_only",
 97 |     intermediate_validation_selects_checkpoint=False,
 98 |     model_initialization=dict(expected_checkpoint_epoch=29),
 99 | )
100 | 
101 | seed = 3407
102 | total_epochs = 30
103 | max_updates = duca_stage2_total_steps
104 | checkpoint_interval_epochs = 5
105 | checkpoint_policy = dict(
106 |     resumable=True,
107 |     keep_latest=3,
108 |     milestones=True,
109 |     final=True,
110 |     final_ema=True,
111 | )
112 | paper_claim_allowed = False
113 | work_dir = "exps/thumos/adatad/duca_h65_60_stage2_am_rpch25"
```

### File: ../OpenTAD_DUCA_H65_LRSchedule_20260824/configs/adatad/thumos/duca_h65_60_stage2_longcosine_h6000.py
Lines: 1-19
```python
 1 | """H65-60 attribution: mature Stage-1 plus truncated historical cosine."""
 2 | 
 3 | _base_ = ["./duca_h65_60_stage2_am_rpch25.py"]
 4 | 
 5 | 
 6 | duca_sampling_rate_contract = dict(
 7 |     route="DUCA_H65_60_STAGE2_LONGCOSINE_H6000",
 8 |     stage="mature_stage1_then_truncated_historical_cosine_horizon",
 9 | )
10 | 
11 | scheduler = dict(
12 |     mode="longcosine_h6000",
13 | )
14 | 
15 | workflow = dict(
16 |     training_profile="duca_h65_60_stage2_longcosine_h6000",
17 | )
18 | 
19 | work_dir = "exps/thumos/adatad/duca_h65_60_stage2_longcosine_h6000"
```

## Answer

Nonce: `DUCA-H65-60-LR-SCHEDULE-REASSESS-v002-20260824`

# 最终科学裁决

**SCIENTIFIC_DECISION: `CONTINUE`**

**控制状态：`HOLD_NEW_TUNING_UNTIL_TERMINAL`**

**一句话原因：** 当前 A/B 两臂已经是足以回答“成熟 Stage‑1 后，哪一种 3000-update Stage‑2 LR 日程包最有可能恢复 H65”的最小严格比较；但它们尚无终态，任何第三臂、参数组 LR 微调或中途优胜者判断都会破坏归因。

这里的 `CONTINUE` 只表示：**继续并完成已经运行的 Jobs `1252979`、`1252980`，同时冻结结果判据。** 它不表示 30+30 已恢复历史性能，也不表示 60 epoch 可以无损替代 30+60。

---

# 一、证据状态与可接受锚点

本次裁决仅接受以下 matched H65 终态作为日程锚点：

| 训练合同   |      Stage‑1 |      Stage‑2 |     终态 Avg-mAP |        mAP@0.7 |
| ------ | -----------: | -----------: | -------------: | -------------: |
| 历史 H65 | 3000 updates | 6000 updates |    **65.1257** |    **43.3137** |
| 失败压缩版  | 2000 updates | 4000 updates |    **62.4648** |    **39.9434** |
| 差值     |              |              | **−2.6609 pp** | **−3.3703 pp** |

Stage‑1 自身的终点是：

* 30 epoch：`59.4231`
* 20 epoch：`49.5389`
* 差值：`−9.8842 pp`

这些数字支持日程诊断，但均为单 seed；历史 checkpoint 又缺少完整 RNG/DataLoader 恢复状态，因此不能把它们包装成论文级因果估计或方差结论。历史 `65.696` 来自 physical-grid/协议不完全匹配的探索身份，继续排除。（任务说明 L24–L29）

---

# 二、性能下降的证据分层诊断

## 2.1 第一层：确定发生了什么

### 1. Stage‑1 handoff 明显不成熟——最强的直接前置证据

20-epoch Stage‑1 比 30-epoch Stage‑1 低 `9.8842 pp`。因此失败的 20+40 从 epoch‑19 EMA 开始时，检测器、ASFormer 语义表征以及均匀 K384 下的联合模型都没有达到历史 handoff 状态。

这是目前最强的事实证据，但必须严格区分：

* **已经确定：** 两个 handoff 状态质量不同。
* **尚未确定：** 这 `9.8842 pp` 中有多少最终转化成了 `−2.6609 pp`。
* **不得声称：** Stage‑1 不成熟单独解释了全部终态下降。

当前 A/B 都改用同一成熟 epoch‑29 EMA，因而可以检验“不成熟 handoff 是否是压缩失败的必要因素”，但由于 A/B 同时把 Stage‑2 缩短到 3000 updates，它们仍不是 Stage‑1 handoff 的纯单变量实验。

### 2. Stage‑2 实际优化暴露量减少——不可由 LR 形状等价替换

历史 Stage‑2 有 6000 次成功 optimizer update；20+40 只有 4000；当前 30+30 A/B 进一步只有 3000。由此减少的不只是“训练轮数”，还包括：

* 见到的 minibatch 与增强样本数；
* AdamW 一阶、二阶矩的演化次数；
* weight decay 施加次数；
* selector—detector 协同适应次数；
* full-joint 状态下的随机梯度暴露；
* EMA 更新次数。

这是第二强的确定事实。其精确贡献尚未被单独隔离。

### 3. LR 被提前压缩——确定的日程变化，效果尚待 A/B 终态

历史 Stage‑2 的 legacy 日程是 5-epoch warmup、60-epoch cosine；失败 20+40 改为 3-epoch warmup、40-epoch cosine，因而 warmup、衰减终点及低 LR 区间都被按短 horizon 提前。（历史配置的 `warmup_epoch=5` 可由冻结代码确认。）

当前 A/B 恢复了 500-update warmup，但使用两种不同的 3000-update 后续日程。它们可以判断哪一个**完整 LR 日程包**更有效，不能把结果细拆为“plateau 单独贡献多少”或“cosine 单独贡献多少”。

### 4. semantic/policy supervision 退火提前——确定存在，尚未单独归因

历史 Stage‑2 中：

* actionness、transition、transition-boundary；
* `policy_alpha`；
* `asformer_adapt`

在 3000 updates 内完成过渡。

失败 20+40 把这一过程缩为 2000 updates。当前 A/B 同样使用 2000-update 过渡，因此它们与历史 30+60 相比仍然改变了 semantic/policy clock。

这可能造成两类相反现象：

* 过早减弱边界与 transition 监督，产生 semantic forgetting；
* 更早开放 learned policy，使 detector 反馈在 selector 尚不稳定时主导。

目前没有数值证据可以选择其中之一。

### 5. detector-feedback timing 被压缩——确定存在，效果未隔离

历史路径是：

* 1000-step feedback warmup；
* 2000-step transition；
* 之后约 3000-step full-joint tail。

失败 20+40 将 feedback 改成 `667+1333`，并只留下 2000-step tail。当前 A/B 改为：

* 1000-step warmup；
* 1000-step transition；
* 1000-step full-joint tail。

因此 A/B 保留了绝对 1000-step 无反馈 warmup，却仍减少了反馈渐进期和 full-joint tail。若两臂都失败，不能只怪 LR；剩余 joint exposure 过短仍是同等合理解释。

### 6. EMA lag 是更新量减少的下游效应，不是独立主因

冻结实现以 `decay=0.999` 创建新的 Stage‑2 EMA，并在每次成功 optimizer update 后更新。

初始 Stage‑1 权重在终态 EMA 中的直接系数约为：

* 3000 updates：`0.999^3000 ≈ 4.97%`
* 4000 updates：`≈ 1.83%`
* 6000 updates：`≈ 0.247%`

因此较短 Stage‑2 确实具有更强的初始状态记忆和更明显的在线模型滞后风险。但 EMA lag 与参数轨迹、更新量完全耦合，不能在没有 online-versus-EMA 诊断时被提升为主要原因。

## 2.2 当前因果优先级

本次冻结的证据排序是：

1. **Stage‑1 handoff 不成熟：直接观测到的最强前置差异；贡献量未知。**
2. **Stage‑2 minibatch/optimizer/joint exposure 减少：确定且无法由 LR 面积替代。**
3. **semantic/policy、feedback 与 full-joint 时钟压缩：确定存在，彼此耦合。**
4. **LR 形状与累计 LR 剂量变化：A/B 当前直接检验的因素。**
5. **EMA lag：可测的次级机制，不得先验归为主因。**

---

# 三、A/B 两臂的严格正确性与隐藏混杂审查

## 3.1 A 与 B 之间：总体上是有效的 Stage‑2 LR 日程包比较

冻结代码满足以下关键条件：

* B 直接继承 A，除 `scheduler.mode`、route/stage 元数据、training profile 和 work directory 外不改变模型或课程。
* 测试明确比较完整 resolved configs，并在删除这些允许字段后要求二者完全相等。
* 两臂均固定 seed `3407`、3000 updates、epoch‑29 EMA checkpoint、同一 2000-step semantic transition 与同一 feedback clock。
* scheduler 以成功 optimizer update 为单位，不按 DataLoader 长度再次缩放。
* 所有参数组乘以同一个相对因子，保持基础 LR 比例。
* 中断恢复测试要求 LR 轨迹逐元素相同。
* 训练引擎仅在 optimizer step 真正成功后推进 DUCA schedule、LR scheduler 和 EMA。
* Stage‑1 `state_dict_ema` 被严格加载，唯一允许重置的是 `frame_selector._loss_weight_schedule_step`；optimizer、scheduler 和 AMP 状态不从 Stage‑1 继承。
* launcher 要求 exact commit、clean checkout、相同 epoch‑29 checkpoint SHA 和固定 seed。

因此，**A 与 B 的运行若最终身份收据通过，可以被称为“模型不变、基础 LR 不变的 Stage‑2 LR-schedule-package attribution”。**

## 3.2 但 A/B 并不是“等 LR 面积、只比较曲线几何”

冻结 scheduler 公式给出：

### A：AM-RPCH25

* update 1：`0`
* update 500：`1.0×`
* update 501–1500：`1.0×`
* update 2000：`0.625×`
* update 2500–3000：`0.25×`
* 3000 次更新的离散累计倍率：**`1999.625`**

### B：LongCosine-H6000

* update 500：`1.0×`
* update 1500：约 `0.92063×`
* update 2000：约 `0.82743×`
* update 2500：约 `0.70771×`
* update 3000：**`0.571157×`**
* 3000 次更新的离散累计倍率：约 **`2366.228`**

因此 B 的累计 LR 剂量比 A 高约 **18.33%**。公式与冻结点由 scheduler 及测试直接定义。

由此必须限制解释：

* B 胜出，不能单独归因于“长 cosine 形状”；也可能只是累计步长更大。
* A 胜出，不能单独归因于 plateau 或 terminal hold；可能是更充分的后期稳定化。
* 该比较仍然有效，因为**完整 LR 日程本来就是处理变量**；无效的是把它进一步拆成尚未实验隔离的微观机制。

## 3.3 LongCosine-H6000 不是 legacy scheduler 的 bit-exact 重放

历史 legacy scheduler 使用统一绝对 `eta_min=1e-8`；B 使用纯相对倍率。因此 B 应准确称为：

> **historical-horizon relative cosine analogue**

而不是：

> **exact historical scheduler truncated at update 3000**

在前 3000 updates 内二者差异通常很小，但对于不同基础 LR 参数组并非数学上完全相同；当前设计选择相对日程是为了严格保持组间 LR 比例。

## 3.4 A/B 与历史 30+60 不是 schedule-only 单变量比较

A/B 相对于历史锚点同时改变：

* Stage‑2 updates：6000 → 3000；
* semantic/policy transition：3000 → 2000；
* detector feedback transition：2000 → 1000；
* full-joint tail：约 3000 → 1000；
* EMA exposure：6000 → 3000；
* legacy absolute-floor cosine → relative scheduler。

所以 A/B 可以回答：

> 在成熟 Stage‑1 handoff 下，某个 3000-update 压缩日程包能否经验性恢复历史终态？

不能回答：

> LR 形状单独造成了历史与压缩结果的全部差异。

## 3.5 仍需终态收据关闭的隐藏工程混杂

有一个需要明确指出的薄弱点：A/B 配置中的 `formal_successful_update_contract=False`。虽然 launcher、重试逻辑、update audit 和恢复测试都很强，但核心训练入口不会像正式 DUCA contract 那样自动验证每轮恰好 100 次更新。

因此 Critic 必须以终态收据验证，而不是从配置字段推定：

* `successful_optimizer_updates = 3000`
* `scheduler_updates = 3000`
* `ema_updates = 3000`
* `duca_schedule_updates = 3000`
* 每 epoch 实际 train batches = 100
* 无 replay exhaustion
* 恢复前后 scheduler、EMA、DataLoader epoch state 连续
* 两臂运行时参数组成员与基础 LR 清单相同
* 两臂使用同一 Stage‑1 checkpoint SHA
* 两臂使用相同数据、annotation、pretrain 和 evaluator hash

任一不满足，相关 arm 应标为 **INVALID_ATTRIBUTION**，只能按同一身份重跑，不能改参数。

---

# 四、30+30 是否有无损恢复保证

**没有理论保证，也没有现有经验保证。**

即使令 3000-update 日程的 LR 积分等于 6000-update 日程，也不能推出终态等价。原因不是抽象的“深度网络很复杂”，而是这里存在明确的路径依赖：

1. AdamW 的矩估计依赖每一个按顺序出现的梯度；
2. selector 的 hard/straight-through 路径使局部梯度随采样位置变化；
3. semantic、policy 与 detector-feedback 权重都随更新步变化；
4. 当前 selector 会改变下一次重型 detector 所看到的输入；
5. EMA 对整个参数轨迹加权，而不是只看最终参数；
6. 少掉的 3000 个 minibatch 包含无法由更大 LR 合成的随机增强和梯度方向。

LR 形状可以修复：

* 过早进入极低 LR；
* 过短的高 LR 适应区；
* 在 policy 打开时步长已不足；
* terminal checkpoint 尚未经过稳定 LR 区间。

LR 形状不能修复：

* 未见到的 3000 个 minibatch；
* 缺失的 AdamW/weight-decay 更新；
* 缺失的 selector—detector 协同适应；
* 缺失的 full-joint tail；
* 缺失的 EMA averaging exposure。

因此，“30+30 无损恢复”只能是终态、多 seed 后得到的经验结果，不能从 LR 面积或训练动力学常识预先推出。

---

# 五、冻结终态成功、失败与斜率判据

令任一 arm 的终态 epoch‑29 EMA 为：

[
\Delta_{\mathrm{Avg}}=
\mathrm{Avg\text{-}mAP}_{arm}-65.1257
]

[
\Delta_{0.7}=
\mathrm{mAP@0.7}_{arm}-43.3137
]

## 5.1 身份与稳定性准入门

在读指标前必须全部通过：

* exact revision `ae3642a138c5b2e1ac2daad75a6d43d17cdb6c2f`；
* clean checkout 和 launcher 绑定；
* 同一 Stage‑1 epoch‑29 EMA SHA；
* epoch‑29 `final` 与 `final-EMA` 均存在；
* 四个成功时钟均为 3000；
* scheduler factor trace 与冻结公式逐点一致；
* 无 NaN/Inf、无 replay exhaustion、无未恢复的 AMP skip；
* official THUMOS14 evaluator、NMS、class map 和 annotation 身份一致；
* 中间 validation 从未选择 checkpoint。

任一失败，先判 `INVALID_RUN`，不得讨论恢复或失败。

## 5.2 终态性能分区

### `RECOVERED_NEIGHBORHOOD`

必须同时满足：

* Avg-mAP ≥ **64.6257**，即距历史不超过 `−0.50 pp`；
* mAP@0.7 ≥ **42.8137**，即距历史不超过 `−0.50 pp`；
* 无终端恶化斜率；
* 稳定性门全部通过。

这是单 seed 的“进入可接受邻域”，不是统计等价证明。

### `BORDERLINE`

* Avg-mAP 或 mAP@0.7 中至少一项低于 `−0.50 pp`；
* 但两项都没有低于 `−1.00 pp`；
* 训练稳定。

对应数值区间为：

* Avg-mAP：`[64.1257, 64.6257)`
* mAP@0.7：`[42.3137, 42.8137)`

该区间不允许宣布无损恢复。

### `CLEAR_FAILURE`

满足任一：

* Avg-mAP < **64.1257**；
* mAP@0.7 < **42.3137**；
* 训练/身份准入门失败；
* 终端出现明确崩塌。

### `HIGH_IOU_ONLY_FAILURE`

* Avg-mAP ≥ **64.6257**
* 但 mAP@0.7 < **42.8137**

这不是成功，也不得由较低 tIoU 的平均提升掩盖。

## 5.3 终端斜率

只使用固定的 Stage‑2 EMA validation 点：

* update 2000：epoch 19
* update 2500：epoch 24
* update 3000：epoch 29

定义每 500 updates 的终端趋势：

[
S=\frac{m_{29}-m_{19}}{2}
]

分别计算 `S_Avg` 和 `S_0.7`。

* **仍在上升：** `S_Avg ≥ +0.10 pp/500` 或 `S_0.7 ≥ +0.15 pp/500`，并且 epoch‑29 不低于 epoch‑24。
* **明显下降：** `S_Avg ≤ −0.10` 且 epoch‑29 比 epoch‑24 低至少 `0.30 pp`；或 `S_0.7 ≤ −0.15` 且 epoch‑29 比 epoch‑24 低至少 `0.50 pp`。
* **其余：** plateau/noisy，不据此虚构收敛方向。

斜率只决定后续诊断，不选择中间 checkpoint。

---

# 六、终态后的唯一 if/then 决策树

## 分支 A：至少一臂进入 `RECOVERED_NEIGHBORHOOD`

**THEN：**

1. 先完成一次现有日志的只读 schedule-stability seal；
2. 不再运行任何第三种 schedule；
3. 稳定性通过后，唯一训练扩展是对选中 schedule 与历史 30+60 recipe 做配对多 seed 确认。

这回答了“只补 seeds 还是先检查稳定性”：**稳定性检查必须先做，但它是只读封印，不是新训练；通过后只补配对 seeds。**

若两臂都通过：

* 首先选终态 Avg-mAP 更高者；
* 若 Avg 差小于 `0.20 pp`，选择 mAP@0.7 更高且不低于另一臂 `0.20 pp` 者；
* 若两项差异均小于 `0.20 pp`，选择 **A / AM-RPCH25**，因为它有明确 terminal hold、较低终端 LR 和较低累计 LR 剂量，终态更适合作为硬 3000-update recipe。

不得用中间最佳值选 winner。

## 分支 B：两臂都明显低于锚点，但梯度稳定且终端仍上升

**THEN：**

* 立即承认：**当前 30+30、总 60-epoch 合同没有实现无损压缩。**
* 优先级选择是 **增加 Stage‑2 exposure**，不是单独调某个参数组基础 LR。
* 后续若仍需要判定 exposure 是否为瓶颈，唯一合理的最小实验是成熟 Stage‑1 后增加 1000 次 Stage‑2 update；它已经不再是 60-epoch 无损压缩实验。

选择 exposure 而非参数组 LR 的原因是：当前没有任何证据指向某个参数组的基础 LR 错误，而“稳定且仍上升”直接表明模型尚未完成当前轨迹。

本终稿**不授权该新实验**；当前必须先等 A/B 终态。

## 分支 C：两臂都低于锚点且终端 plateau 或下降

**THEN：**

* 对“60 epoch 无损恢复 H65”作停止判定；
* 保留历史 30+60 作为性能 recipe；
* 不再进行参数组 LR sweep；
* 不用 Query、Bridge、TrueTime、dynamic K 或 loss 改动挽救这一 schedule-attribution 子问题。

## 分支 D：Avg-mAP 接近，但只有高 IoU 明显恶化

**THEN：**

不新增训练，先执行只读边界诊断。优先检查：

* start/end boundary error 是否增大；
* 短动作与密集边界视频是否集中退化；
* transition-boundary 未加权 loss 是否在权重退火后反而升高；
* selector 是否从 start/end 邻域向动作内部迁移；
* 最大采样空洞、相邻位置跨度和 boundary-near frame coverage 是否恶化；
* detector regression 梯度进入 selector 后，transition scorer 梯度是否被压低；
* online 模型是否保住高 IoU、只有 EMA 落后。

若这些现象存在，应把问题归为 **boundary-support/feedback-clock failure**，而不是泛化为“学习率不够大”。

---

# 七、最少只读诊断及其决策意义

| 只读诊断                     | 必须观察的量                                                                   | 什么观察会改变下一步                                                                              |
| ------------------------ | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| 历史 Stage‑2 checkpoint 轨迹 | 按 500 updates 对齐历史 6000-update、A、B 的 terminal-EMA Avg 和 @0.7             | 若历史在 update 3000 也明显低、之后才上升，支持 exposure 瓶颈；若历史 update 3000 已接近最终锚点而 A/B 低，支持课程/LR 路径问题  |
| 参数组 LR 曲线                | 每组 base LR、实际 LR、倍率、累计倍率、resume 连续性                                      | 任一组比例变化或时钟偏移使 arm 无效；B 优势若与额外 18.33% LR 剂量一致，只能归为 schedule package                      |
| 未加权/加权语义 loss            | actionness、transition、transition-boundary 原始 loss 与乘权后贡献                 | 权重下降而原始 boundary loss 上升，支持 semantic forgetting；原始 loss 稳定则降低该解释优先级                     |
| 参数组梯度范数                  | detector、coarse trunk、action head、transition scorer 在 1000/2000 切换点附近的范数 | feedback 打开后 selector 梯度坍缩/爆炸，支持 timing 冲突；全程稳定且指标仍升，支持 exposure 不足                     |
| selector 位移与熵            | 相对 Stage‑1/uniform 的位置位移、分布熵、最大空洞、边界邻域覆盖                                 | 熵塌缩或从边界转向动作内部，解释高 IoU 损失；位置分布稳定则转向 detector/EMA exposure                                |
| online–EMA gap           | final 与 final-EMA 的 Avg、@0.7 及 checkpoint 差距                             | online 比 EMA 高 ≥`0.50 pp` Avg 或 ≥`0.75 pp` @0.7 且 gap 正在收窄，支持 EMA lag；二者都低则排除 EMA 为主要解释 |

若某项没有在当前运行中被记录，必须标为 `NOT_OBSERVED`；**不得为了补齐诊断重新训练。**

---

# 八、Critic 与 Evaluator 必须验证的事实

## Independent Critic

Critic 必须确认：

1. 两个 job 的 exact commit、clean checkout、launcher argv 和环境绑定；
2. Stage‑1 checkpoint path、SHA、epoch 和 `state_dict_ema` 完全相同；
3. resolved configs 除 schedule mode 与允许的身份字段外完全相同；
4. 运行时 optimizer 参数组名称、成员、base LR、weight decay 完全相同；
5. A/B 实际 LR 轨迹与冻结公式逐点一致；
6. 3000 个 successful update 与 scheduler/EMA/DUCA clocks 一一对应；
7. resume 后 RNG、DataLoader epoch、GradScaler、optimizer、scheduler、EMA 均连续；
8. semantic/policy、feedback 和 joint-tail 时钟与配置一致；
9. 没有混入 First-Mixing SingleClock、Bridge、Query、dynamic K、loss 终值或 selector 修改；
10. 历史锚点与当前代码的差异只被用于 recipe-level attribution，不被伪装成 bit-exact paired continuation。

Critic 返回：

* `SCHEDULE_ATTRIBUTION_IDENTITY_PASS`
* 或 `SCHEDULE_ATTRIBUTION_INVALID`

## Result-blind Evaluator

Evaluator 必须在两臂都终态后同时解封，并验证：

1. 仅使用 epoch‑29 final 与 final-EMA；
2. primary metric 固定为 final-EMA；
3. 中间 validation 不参与 checkpoint 或 arm 选择；
4. official Avg-mAP 和 mAP@0.7 使用同一 evaluator；
5. 应用本终稿冻结的邻域、失败和斜率阈值；
6. 同时报告 online、EMA、训练稳定性、终端斜率和 read-only 诊断；
7. 明确标注 `single_seed_schedule_attribution_only`；
8. 不生成 H65 方法优势、训练成本优势或论文创新结论。

---

# 九、论文 claim 边界

无论 A/B 最终结果如何，**训练日程恢复都不是 DUCA 的科学创新。**

若成功，它最多支持：

> 在 seed 3407 和冻结 H65 模型合同下，成熟 Stage‑1 handoff 配合某一 3000-update Stage‑2 日程，可以把终态恢复到历史 30+60 的预注册邻域。

多 seed 后，才可以进一步说该 recipe 较稳定。

它不能支持：

* DUCA 优于 uniform/dense；
* learned acquisition 本身有效；
* dynamic K 有效；
* TrueTime 有效；
* 90→60 epoch 一般可以无损压缩；
* LR schedule 是论文方法贡献；
* 推理成本或 full-stack efficiency 改善；
* 65.696 是 matched H65 结果；
* 单 seed 差异具有统计显著性。

在论文中，该结果最多属于训练协议、复现细节或附录中的优化稳定性分析。

---

# 十、当前唯一后续动作与返回合同

```text
next_owner:
  independent_result_blind_evaluator
  with_critic_identity_signoff

next_action:
  TERMINAL_A_B_SEAL_ONLY
  collect_and_jointly_unseal_jobs_1252979_and_1252980
  apply_frozen_terminal_thresholds
  produce_read_only_diagnostics
  authorize_no_new_training

dependency:
  both_jobs_reach_epoch_29_terminal
  final_and_final_ema_checkpoints_exist
  exact_3000_successful_update_audits_pass
  stage1_checkpoint_sha_matches
  official_evaluator_identity_matches
  no_intermediate_checkpoint_selection

expected_return_at:
  immediately_after_both_terminal_packages_are_complete
  and_before_any_new_config_edit_parameter_tuning_or_launch
```

**最终执行口令：`HOLD_NEW_TUNING_UNTIL_TERMINAL`。**
