# Oracle Browser Transcript

Conversation: https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697-duca/c/6a8ba7fe-82cc-83ea-bf10-1e0b5e4edfda

## Prompt

请完整读取附加的裁决任务与四个冻结配置，作为 DUCA Scientific First-Author Agent 输出终稿。Nonce: DUCA-H65-60-LR-SCHEDULE-v001-20260824

### File: .cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_H65_60_LR_SCHEDULE_ADJUDICATION-v001.md
Lines: 1-73
```md
 1 | # DUCA H65：90→60 epoch 压缩后性能下降的训练动力学裁决
 2 | 
 3 | Nonce: `DUCA-H65-60-LR-SCHEDULE-v001-20260824`
 4 | 
 5 | 你是 DUCA 项目的独立 Scientific First-Author Agent 和最严厉的优化/实验审稿人。请基于下列冻结代码、配置与真实终态结果，分析为什么 H65 从历史 30+60 epoch 压缩为 20+40 epoch 后性能下降，并给出唯一、最小、可直接实现的后续训练方案。不要把路线选择交回人类或 Codex；必须返回一个 `CONTINUE`、`REVISE`、`PIVOT` 或 `STOP`。
 6 | 
 7 | ## 项目与代码身份
 8 | 
 9 | - Exact ChatGPT Project: `DUCA`, `g-p-6a796fef9a00819194024cf1de3bd697`
10 | - GitHub: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
11 | - H65 current branch: `codex/duca-h65-60-curriculum-20260823`
12 | - Clean HEAD: `04c35a3b76897e6c1569eeede41ed3aecaf7f854`
13 | - 60-epoch Stage-2 training revision: `87ff0883651a631d48468ab4f9d6392f587c15e4`
14 | - Historical H65 evidence anchor: `42dba3f90b37243e7965d18b6707e88e81bf7109`
15 | 
16 | 本次附加四个精确配置文件。模型代码没有在 90→60 的归因实验中改变。
17 | 
18 | ## 冻结模型合同
19 | 
20 | H65 是低成本 ASFormer 语义侦察器学习动作性与边界/状态转移证据，再由确定性 sampling-rate/density transport 间接形成非均匀逐帧位置；不是小模型直接预测 frame index。固定 `K=384`，按原始时间排序选中的 RGB 帧，进入同一 VideoMAE-S、Adapter、ActionFormer、loss、NMS、THUMOS14 training/validation split 与官方 evaluator。基础 optimizer 仍为 AdamW，base LR `1e-4`、weight decay `0.05`、backbone frozen、adapter LR `2e-4`。Stage-1 的 selector 参数组 LR 为 `5e-5/1e-4/5e-5`；Stage-2 为 `1e-5/2e-5/5e-5`。这些基础 LR、最终损失权重、模型结构、输入方式、冻结策略均未因压缩而改变。
21 | 
22 | ## 两个终态结果
23 | 
24 | 1. 历史日程：Stage-1 uniform K384 训练 30 epoch / 3000 successful updates；Stage-2 learned/joint 训练 60 epoch / 6000 successful updates。终态 epoch-59 EMA：Avg-mAP `65.1257`，mAP@0.7 `43.3137`。
25 | 2. 压缩日程：Stage-1 20 epoch / 2000 updates；Stage-2 40 epoch / 4000 updates，其中 20 epoch transition + 20 epoch joint。终态 epoch-39 EMA：Avg-mAP `62.4648`，mAP@0.7 `39.9434`。
26 | 3. 差值：Avg-mAP `-2.6609 pp`，mAP@0.7 `-3.3703 pp`。这是同一模型结构下的课程/优化负结果，不是 H65 科学机制被否定。
27 | 4. Stage-1 本身也显示成熟度差异：30-epoch endpoint Avg-mAP `59.4231`，20-epoch endpoint `49.5389`。请判断这是否主要说明 uniform detector/scout warmup 尚未成熟，还是存在更具体的 LR/curriculum 交互。
28 | 
29 | ## 实际改变的训练动力学
30 | 
31 | - Stage-1 scheduler 从 warmup 3 / cosine horizon 30 改为 warmup 2 / horizon 20。
32 | - Stage-2 从 warmup 5 / cosine horizon 60 改为 warmup 3 / horizon 40。
33 | - Stage-2 所有 actionness、transition、boundary、policy-alpha、ASFormer-adapt 的 cosine transition 从 3000 updates 压到 2000。
34 | - detector-gradient 与 detector-contribution 从历史 warmup 1000 + transition 2000 改为 667 + 1333。
35 | - 因而压缩并非只少训练 30 epoch：它同时让 LR 更早衰减、语义监督更快减弱、learned sampling/feedback 更早增大，并缩短 full-joint 后段。
36 | - 仍在运行的 First-Mixing SingleClock TrueTime 实验是独立表示实验，不得用它解释上述 90→60 的性能下降，也不得等待它才能给出本训练裁决。
37 | 
38 | ## 用户冻结的优化原则
39 | 
40 | 1. 保持所有参数组基础 LR 不变；不允许统一乘倍数。
41 | 2. 保留原始 30-epoch uniform Stage-1，不再压缩。
42 | 3. 若总训练预算必须为 60 epoch，则只调整 Stage-2 的 decay 速度/时间尺度，不提高峰值 LR。
43 | 4. 联合训练后段必须保留足够的非零 LR，可通过延长 cosine horizon 或短暂平坦尾段实现。
44 | 5. 只有明确证据表明某一参数组欠拟合且梯度稳定，才允许后续单独小幅调整该组 LR。
45 | 6. 不改变 H65 模型、selected RGB、K、损失终值、检测器、数据、seed=3407、official evaluator、final/final-EMA 规则。
46 | 7. 不重复 dense/uniform/random 等已经多次完成的通用对照；本轮只做 H65 schedule attribution。
47 | 
48 | ## 请重点攻击并裁决
49 | 
50 | 1. 对 `-2.6609 pp` 的原因给出按证据排序的机制解释：Stage-1 不成熟、Stage-2 总更新不足、cosine 过早接近零、语义监督退火过快、policy/feedback 开启过快、joint tail 不足，各自应该如何用已有曲线或无需训练的诊断区分？不要凭直觉定量归因。
51 | 2. “把 90 epoch 挤成 60 epoch”是否在固定每 epoch 100 updates 时，本质上减少了 1/3 optimizer exposure，因此无法仅靠 LR schedule 无损恢复？请明确可恢复上限与不可恢复部分。
52 | 3. 在保留 30-epoch Stage-1 的前提下，冻结一个 30-epoch Stage-2 方案。必须写出逐 update 的：warmup、cosine/hold 区间、policy alpha、actionness、transition、boundary、detector gradient/contribution、ASFormer adapt 时间轴，以及 terminal LR fraction。不得只说“调慢一点”。
53 | 4. 严厉比较下列思路并选择唯一主方案，必要时可提出一个更好的单一方案：
54 |    - `LongCosine-90`：Stage-2 只跑 3000 updates，但沿用历史 6000-update LR horizon，使终点仍保留非零 LR；
55 |    - `RelativeCosine+Hold`：在 3000 updates 内完成较慢的 transition，并给 full-joint 留一个短的非零 LR 平坦尾段；
56 |    - 其他不改峰值 LR、可严格说明的 schedule。
57 | 5. 冻结最多 2–3 个正式完整训练臂。优先复用已经完成的 30-epoch Stage-1 epoch-29 EMA，禁止重复 Stage-1；每个臂必须给出唯一变量、预计信息增益、成功/失败阈值和何时停止继续调参。
58 | 6. 判断是否需要 matched `30+60` 只读 anchor 之外的 OFF 重训；若不需要，明确禁止重复。
59 | 7. 冻结训练正确性检查：实际 successful updates、每参数组 LR 曲线、各 curriculum 权重曲线、梯度范数/非有限更新、checkpoint 恢复、每 5 epoch 可恢复 `.pth`、final/final-EMA 不按中间验证挑选。
60 | 8. 给出结果解释边界：单 seed 达到什么程度只能说明 schedule recovery，何时才值得增加 seeds；不得把训练日程恢复宣称为新科学贡献。
61 | 
62 | ## 必须返回的终稿
63 | 
64 | - 唯一裁决与一句话原因；
65 | - 对性能下降的证据排序诊断；
66 | - 精确 30-epoch Stage-2 schedule 表/公式；
67 | - 最小配置改动（允许文件、字段、禁止改动）；
68 | - 最多 2–3 臂实验矩阵、seed、6000? 或 3000 successful-update 解释、checkpoint/cost/stop rule；
69 | - Critic 与 Evaluator PRE_RUN 必须核验的事实；
70 | - 失败后下一步是增加训练 exposure、单独 LR 微调，还是承认 60-epoch 预算不可无损压缩；
71 | - `next_owner / next_action / dependency / expected_return_at`。
72 | 
73 | 请用严肃、直接、外部评审可理解的中文作答。不要引入新 selector、Query、Bridge、dynamic K、连续 cliplet 或新的论文故事。
```

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

## Answer

# DUCA H65 90→60 epoch 训练日程终局裁决

**Nonce：`DUCA-H65-60-LR-SCHEDULE-v001-20260824`**

## 1. 唯一科学裁决

**SCIENTIFIC_DECISION：`REVISE`**

**一句话原因：**`20+40` 负结果不是 H65 机制失效，而是把 Stage-1 成熟度、Stage-2 更新暴露、LR 衰减、语义退火、policy/feedback 开启和 full-joint 尾段同时压缩；下一步必须复用成熟的 Stage-1 epoch-29 EMA，只运行一个严格冻结的 30-epoch Stage-2 相对 LR 日程及一个 LR 形状归因臂，不改模型或基础 LR。

本裁决不等待 First-Mixing SingleClock TrueTime 实验，也不允许用该表示实验解释本次 `-2.6609 pp`。

---

## 2. 对性能下降的证据排序诊断

冻结结果为：

* 历史 `30+60`：Avg-mAP `65.1257`，mAP@0.7 `43.3137`。
* 压缩 `20+40`：Avg-mAP `62.4648`，mAP@0.7 `39.9434`。
* 差值：Avg-mAP `-2.6609 pp`，mAP@0.7 `-3.3703 pp`。
* Stage-1 终点：30 epoch `59.4231`，20 epoch `49.5389`，相差 `9.8842 pp`。

### 2.1 排名与证据等级

| 排名 | 机制                                   |      当前证据强度 | 严格解释                                                                                                                |
| -- | ------------------------------------ | ----------: | ------------------------------------------------------------------------------------------------------------------- |
| 1  | **Stage-1 handoff 不成熟**              |      最强直接证据 | 相同 uniform-K384 阶段的终点差 `9.8842 pp`。这证明 epoch-19 EMA 不是与 epoch-29 EMA 等价的初始化，但不能把最终 `-2.6609 pp` 全部归因于它。             |
| 2  | **Stage-2 optimizer exposure 减少**    |        确定事实 | 历史 Stage-2 有 6000 次成功更新，压缩实验只有 4000；在恢复 30-epoch Stage-1 后，60-epoch 总预算只剩 3000 次 Stage-2 更新，即历史的一半。                 |
| 3  | **LR horizon 过短并在终点坍缩**              | 强机制证据，效果量未知 | Stage-2 从 warmup 500 / horizon 6000 变成 warmup 300 / horizon 4000；压缩终点进入绝对 `eta_min` 邻域，最后的 full-joint 更新几乎失去有效步长。   |
| 4  | **full-joint 尾段不足**                  |      确定日程差异 | 历史在 3000-step transition 后有 3000-step full-joint；压缩版只有 2000-step full-joint。恢复 30+30 后，最多只能保留 1000-step full-joint。 |
| 5  | **语义监督与 policy/ASFormer adapt 退火过快** | 配置事实，因果量未识别 | actionness、transition、boundary、policy-alpha、ASFormer-adapt 的 3000-step transition 被压至 2000。                         |
| 6  | **detector feedback 开启过早**           | 配置事实，因果量未识别 | 无反馈期从 1000 step 缩到 667，完整 feedback 从 step 3000 提前到 step 2000。                                                       |
| 7  | **EMA 滞后或实现恢复问题**                    |       尚无正证据 | EMA 可能放大短尾段问题，但没有证据证明 EMA 本身导致下降；checkpoint/resume 状态不完整则属于实验无效，而不是模型负结果。                                           |

### 2.2 为什么不能说“主要就是 Stage-1”

Stage-1 的 `9.8842 pp` 差异是最强异常信号，但它与最终 learned Stage-2 的 Avg-mAP 不在同一训练阶段，不能相加或直接折算：

* Stage-2 可以恢复部分较差初始化；
* Stage-1 终点同时受 1000 次缺失更新和更快 LR 衰减影响；
* 没有 `30e Stage-1 + 40e Stage-2 compressed schedule` 与 `20e Stage-1 + 60e Stage-2` 的交叉单元。

因此，**Stage-1 不成熟排第一，但其对最终 `-2.6609 pp` 的数值贡献仍不可识别**。

### 2.3 无需新训练即可完成的区分诊断

在启动新臂前，必须只读提取已有日志和 checkpoint：

1. **Stage-1 曝露与 LR 的区分**
   用历史 30-epoch Stage-1 的 epoch-19 EMA，与压缩 Stage-1 epoch-19 EMA 在相同 evaluator 下比较：

   * 两者接近：缺失 epoch 20–29 的 1000 次更新占主导；
   * 历史 epoch-19 已明显更高：20-horizon 提前衰减本身占有重要作用。

2. **Stage-2 缺更新量的诊断**
   从历史 Stage-2 每 5 epoch checkpoint 计算：
   [
   \Delta_{0:3000},\quad \Delta_{3000:4000},\quad \Delta_{4000:6000}.
   ]
   若 4000–6000 仍持续提高，缺失 exposure 是不可忽略的；若早已平台而压缩版提前停滞，LR/curriculum 时钟更可疑。

3. **语义退火过快**
   同时画：

   * 未乘权重的 actionness、transition、boundary loss；
   * 乘权重后的实际贡献；
   * coarse trunk、action head、transition scorer 梯度范数。
     若未加权 loss 仍在改善，但加权梯度在 step 2000 前快速消失，说明监督退火过早。

4. **feedback 开启过早**
   对齐 step 667、1000、2000、3000，检查：

   * selector entropy / top-mass concentration；
   * 相对 exact-uniform 的平均、p95、最大位移；
   * detector、adapter、三个 selector 参数组的梯度范数；
   * 非有限更新或梯度尖峰。
     若这些量在 feedback 开启点发生突变，而非随 LR 平滑变化，则 feedback clock 是独立问题。

5. **joint tail 不足**
   检查最后 5、10 epoch：

   * EMA 与 online 的差；
   * Avg-mAP 与 mAP@0.7 的终端斜率；
   * detector/selector loss 是否仍有单调下降。
     终点仍有明显正斜率，说明不是 checkpoint 选择问题，而是尚未完成优化。

---

## 3. 60 epoch 能恢复什么，不能恢复什么

### 3.1 不可恢复的事实

历史训练总共：

[
3000\text{ Stage-1 updates}+6000\text{ Stage-2 updates}=9000.
]

严格 60-epoch 合同只能有：

[
3000+3000=6000.
]

因此总 optimizer exposure 减少：

[
\frac{9000-6000}{9000}=33.3%.
]

而 Stage-2 exposure 直接减半：

[
6000\rightarrow3000.
]

任何 LR scheduler 都不能生成缺失的：

* 3000 次随机 minibatch 梯度观测；
* 3000 次 optimizer 状态转移；
* 3000 次 EMA 更新；
* 3000 次数据增强与样本顺序暴露；
* 3000 次 selector—detector 联合适应机会。

所以，**日程可以恢复“时钟压缩造成的损失”，但不能保证恢复“更新次数本身提供的优化收益”**。

### 3.2 可恢复上限不能从两个终点数值推断

本轮材料只支持：

* 可观察待恢复差值最多为 `2.6609 / 3.3703 pp`；
* 60-epoch 可恢复其中多少，当前区间仍是未知的；
* `65.1257 / 43.3137` 是历史只读锚，不是 3000-step Stage-2 的理论可达保证。

任何“Stage-1 占 70%”“LR 占 1.5 pp”之类的定量分解都会是编造。

### 3.3 LR 累积暴露的辅助计算

忽略极小绝对 `eta_min`，把各 step 的 LR 除以其参数组 base LR，可得到近似 normalized LR exposure：

* 历史 Stage-2：warmup 500 + 6000 horizon，约 `3000.5` base-LR-update equivalents；
* 失败的压缩 Stage-2：warmup 300 + 4000 horizon，约 `2000.5`；
* 本裁决主方案：约 `1999.6`；
* LongCosine-90 截断到 3000 updates：约 `2366.7`。

这不是收敛定理，但它说明主方案**不会通过隐性增加累计 LR 面积作弊**；它保持与失败压缩版近乎相同的累计 LR 暴露，只重排其时间位置并留下非零尾段。

---

# 4. 唯一主方案：Area-Matched Relative Plateau–Cosine–Hold

方案 ID：

**`DUCA_H65_60_STAGE2_AM_RPCH25-v001`**

不是简单 LongCosine-90，也不是 warmup 后立即一路 cosine 到 floor，而是：

> **500-step 历史绝对 warmup → 1000-step peak-LR plateau → 1000-step 相对 cosine 到 0.25× → 500-step 0.25× hold。**

它比两个原候选更合理：

* 相比 **LongCosine-90**：保留真正的低 LR terminal phase，终点不是历史训练的“中途 LR 状态”；
* 相比普通 **RelativeCosine+Hold**：加入 1000-step peak plateau，避免在 Stage-2 只有 3000 次更新时过早损失有效 LR 暴露；
* 相比失败的 40-epoch cosine：累计 LR 面积近似匹配，但不在 final joint tail 坍缩到绝对 `eta_min`；
* 相比统一绝对 floor：始终保持所有参数组的 LR 比例。

当前 scheduler 使用单一绝对 `eta_min=1e-8`，并把 epoch 数乘以 dataloader 长度转为 iteration 时钟；这会在终点把不同 base LR 的参数组拉向相同绝对值。
训练引擎已经只在成功 optimizer step 后推进 DUCA schedule、scheduler 和 EMA，因此新 scheduler 必须沿用这一 successful-update 时钟，而不能按 attempted batch 或 wall-clock epoch 计数。

---

## 5. 精确 3000-update LR 定义

令 (n\in{1,\ldots,3000}) 为**即将执行的第 (n) 次成功 optimizer update**，定义所有参数组共同的相对倍率 (g_n)：

[
g_n =
\begin{cases}
\dfrac{n-1}{499},
&1\le n\le 500,[6pt]
1,
&501\le n\le 1500,[6pt]
0.25+
0.375\left[
1+\cos\left(\pi\dfrac{n-1500}{1000}\right)
\right],
&1501\le n\le 2500,[8pt]
0.25,
&2501\le n\le 3000.
\end{cases}
]

每个参数组 (i)：

[
\operatorname{LR}_i(n)=\operatorname{baseLR}_i\cdot g_n.
]

### 5.1 关键性质

* 第 1 次 update 使用当前 historical warmup 语义下的零起点；
* 第 500 次达到 base LR；
* update 501–1500 保持 base LR；
* update 1501–2500 相对 cosine；
* update 2501–3000 保持 `0.25× base LR`；
* terminal LR fraction：**`0.25`**；
* 所有参数组比值从头到尾保持不变；
* 禁止单一绝对 `eta_min`；
* 禁止对 base LR 再乘预先常数；
* 禁止按 epoch 长度变化或失败 AMP attempt 推进时钟。

### 5.2 terminal 实际 LR

| 参数组                 | 冻结 base LR | terminal LR |
| ------------------- | ---------: | ----------: |
| detector base group |   `1.0e-4` |    `2.5e-5` |
| adapter             |   `2.0e-4` |    `5.0e-5` |
| coarse trunk        |   `1.0e-5` |    `2.5e-6` |
| action head         |   `2.0e-5` |    `5.0e-6` |
| transition scorer   |   `5.0e-5` |   `1.25e-5` |

例如 transition scorer 与 coarse trunk 的 `5:1` 比值必须在 terminal 仍是 `5:1`，而不是一起变成 `1e-8`。

---

# 6. 精确 curriculum / feedback 时间轴

令 (u) 为当前 update 前已经完成的成功 optimizer updates。

定义主语义 transition：

[
H(u)=
\frac{1-\cos\left(
\pi\frac{\min(\max(u,0),2000)}{2000}
\right)}{2}.
]

则：

[
\begin{aligned}
w_{\rm action}(u) &= 1.0-0.75H(u),\
w_{\rm transition}(u) &= 0.50-0.40H(u),\
w_{\rm boundary}(u) &= 2.0-1.75H(u),\
\alpha_{\rm policy}(u) &= H(u),\
a_{\rm ASFormer}(u) &= H(u).
\end{aligned}
]

定义 detector feedback 时钟：

[
D(u)=
\begin{cases}
0,&u\le1000,[4pt]
\dfrac{1-\cos\left(\pi\dfrac{u-1000}{1000}\right)}{2},
&1000<u<2000,[8pt]
1,&u\ge2000.
\end{cases}
]

于是：

[
\begin{aligned}
g_{\rm detector}(u)&=0.25D(u),\
c_{\rm detector}(u)&=D(u).
\end{aligned}
]

其中：

* `detector_gradient = 0.25 D(u)`；
* `detector_contribution = D(u)`；
* 前 1000 updates 完全没有 detector feedback；
* updates 1001–2000 用 cosine 开启；
* updates 2001–3000 为完整 final joint；
* **full-joint tail 恰好 1000 successful updates**。

### 6.1 时间轴汇总

| completed updates | LR 状态       | 语义/policy      | detector feedback | 科学作用                                    |
| ----------------: | ----------- | -------------- | ----------------- | --------------------------------------- |
|           `0–500` | 0→1 warmup  | 缓慢启动           | OFF               | 避免刚切换 learned sampling 就承受大步长           |
|        `500–1000` | 1.0 plateau | 继续 transition  | OFF               | 给成熟 Stage-1 表征适应新 Stage-2 loss          |
|       `1000–1500` | 1.0 plateau | 中后段 transition | 0→0.5             | feedback 开始，但不同时衰减 LR                   |
|       `1500–2000` | 1.0→0.625   | transition 收束  | 0.5→1             | policy、ASFormer、detector feedback 同步到终值 |
|       `2000–2500` | 0.625→0.25  | final          | final             | 低 LR full-joint refinement              |
|       `2500–3000` | 0.25 hold   | final          | final             | 稳定 terminal EMA，禁止 LR 坍缩                |

---

# 7. 为什么不把 LongCosine-90 作为主方案

LongCosine-90 在 3000 updates 后的相对 LR 为：

[
g_{\rm LC90}(3000)
==================

\frac{1+\cos\left(
\pi\frac{3000-500}{6000-500}
\right)}{2}
\approx 0.571157.
]

它有两个问题：

1. final checkpoint 仍处于历史 6000-step cosine 的中段，不是一个明确的终端优化状态；
2. 它用更高的 normalized LR exposure 补偿缺失 updates，若性能更好，无法区分是“更好的形状”还是“累计有效步长更大”。

因此：

* **AM-RPCH25 是主方案**；
* LongCosine-90 只保留为一个完整的 schedule-attribution arm；
* 不把 LongCosine 的胜出解释为无损压缩，只解释为“3000-step 情况下需要更高的后段有效 LR”。

---

# 8. 最小正式实验矩阵：仅两臂

两个臂复用**同一份 Stage-1 epoch-29 `state_dict_ema`**，不重跑 Stage-1。

| Arm                        | Stage-1           |                 Stage-2 | LR                                                               | curriculum / feedback                                             |   seed | 唯一目的                                           |
| -------------------------- | ----------------- | ----------------------: | ---------------------------------------------------------------- | ----------------------------------------------------------------- | -----: | ---------------------------------------------- |
| **A：AM-RPCH25，主方案**        | epoch-29 EMA，只读复用 | 30 epoch / 3000 updates | 500 warmup + 1000 plateau + 1000 relative cosine→0.25 + 500 hold | semantic/policy 2000；feedback 1000+1000；1000-step full-joint tail | `3407` | 测试在不增加累计 LR 面积的情况下，重新安排 decay 与 tail 能否恢复      |
| **B：LongCosine-H6000，归因臂** | 完全相同              | 30 epoch / 3000 updates | warmup 500，历史 6000-step horizon，运行到 update 3000 即停               | 与 A 完全相同                                                          | `3407` | 测试更高后段 LR exposure 是否比 terminal floor/hold 更重要 |

### 8.1 “6000 updates”不得混淆

每个新模型的总训练身份是：

* Stage-1：历史已有 `3000` updates；
* Stage-2：新运行 `3000` updates；
* 模型总 exposure：`6000` updates；
* **不是 Stage-2 6000 updates**。

本轮新增计算只包含：

[
2\times3000=6000
]

次 Stage-2 successful updates；Stage-1 计算不重复，但必须在成本表中标注为共享前置成本。

### 8.2 不运行第三臂

不增加：

* 另一 floor；
* 另一 warmup；
* 另一 transition length；
* per-group LR；
* 30+45；
* 30+60 复现；
* selector、Query、Bridge、dynamic K、cliplet。

两臂完成后，LR schedule 搜索即终止。

---

# 9. 不需要 OFF 或 30+60 重训

**禁止重新训练 matched `30+60` OFF/anchor。**

理由：

* 历史 `30+60` 终态已经作为冻结只读锚；
* 本任务是 H65 schedule recovery，不是重新估计 baseline；
* 重训同一 anchor 不增加对 60-epoch 压缩机制的识别力；
* 新实验唯一需要的是同一 Stage-1 epoch-29 EMA 的不可变 checkpoint hash。

若 anchor 的 checkpoint/config/evaluator 身份无法核验，应当：

> **阻断比较并返回 identity blocker，而不是默认重跑。**

---

# 10. 配置与代码的最小改动边界

## 10.1 允许修改

最多允许：

1. `opentad/cores/scheduler.py`

   * 新增一个 relative-factor successful-update scheduler；
   * 同一实现通过参数支持 AM-RPCH25 与 LongCosine-H6000；
   * 不修改旧 scheduler 行为。

2. 新增：

   * `configs/adatad/thumos/duca_h65_60_stage2_am_rpch25.py`
   * `configs/adatad/thumos/duca_h65_60_stage2_longcosine_h6000.py`

3. 一个 focused scheduler/resume test 文件。

4. 一个只负责绑定现有 launcher 参数、输出根和 update-audit 路径的运行包装文件；不得改变训练参数。

## 10.2 两个配置必须显式覆盖

```python
seed = 3407
total_epochs = 30
max_updates = 3000
workflow.end_epoch = 30
workflow.expected_train_batches_per_epoch = 100
workflow.expected_successful_optimizer_updates = 3000
workflow.primary_checkpoint_epoch = 29
workflow.primary_checkpoint_state_key = "state_dict_ema"
workflow.checkpoint_criterion = "terminal_epoch_29_state_dict_ema"
workflow.model_initialization.expected_checkpoint_epoch = 29
```

共同 curriculum：

```python
transition_steps = 2000

detector_gradient.warmup_steps = 1000
detector_gradient.transition_steps = 1000

detector_contribution.warmup_steps = 1000
detector_contribution.transition_steps = 1000
```

必须消除继承的歧义：

* 不得让 base 中 `duca_stage2_half_steps=3000` 继续控制任一新臂；
* resolved config 中所有相关字段必须明确显示 `2000` 或 `1000+1000`；
* `total_epochs`、`workflow.end_epoch`、`max_updates` 和 scheduler stop 必须一致；
* LongCosine 的 `horizon=6000` 只是 scheduler horizon，绝不能把 run length 扩成 60 epoch。

## 10.3 禁止修改

禁止触碰：

* H65 selector、ASFormer、sampling-rate/density transport；
* selected RGB 或物理时间排序；
* `K=384`；
* VideoMAE-S、Adapter、ActionFormer；
* loss 起点、终值或 loss 类型；
* optimizer 类型、base LR、weight decay；
* backbone frozen 状态；
* NMS、split、official evaluator；
* seed；
* final/final-EMA 规则；
* Stage-1 checkpoint 内容；
* First-Mixing、TrueTime、Query、Bridge、dynamic K、连续 cliplet。

---

# 11. Checkpoint、resume 与更新正确性

## 11.1 Stage-1→Stage-2 handoff

必须绑定：

* exact checkpoint 路径；
* SHA-256；
* epoch=`29`；
* state key=`state_dict_ema`；
* Stage-2 初始化时重置 optimizer、scheduler、AMP scaler；
* `_loss_weight_schedule_step` 只在这次 Stage-1 handoff 时归零。

## 11.2 Stage-2 中断恢复

从 Stage-2 milestone 恢复时必须同时恢复：

* online model；
* EMA model；
* optimizer；
* scheduler；
* AMP scaler；
* RNG；
* successful update counter；
* DUCA curriculum step；
* update audit。

**禁止在 Stage-2 resume 时再次把 curriculum step 归零。**

## 11.3 每 5 epoch 保存

保存 epoch：

`4, 9, 14, 19, 24, 29`

每个 `.pth` 必须是真正可恢复 full-state checkpoint，而不只是模型权重。

必须做一次无真实训练数据的 resume-fidelity 测试：

* 连续推进 3000 synthetic scheduler steps；
* 与在 step 500、1000、1500、2000、2500 中断恢复后的轨迹逐元素比较；
* LR、curriculum、EMA/update counters 必须完全一致。

---

# 12. 强制附加日志

每个 successful update 或至少每个 epoch 输出结构化记录：

1. **每参数组 LR**

   * group name；
   * base LR；
   * current LR；
   * relative factor；
   * 参数组 LR 比例。

2. **每参数组梯度范数**

   * detector；
   * adapter；
   * coarse trunk；
   * action head；
   * transition scorer；
   * clip 前、clip 后；
   * non-finite 次数。

3. **curriculum**

   * actionness weight；
   * transition weight；
   * boundary weight；
   * policy alpha；
   * detector gradient；
   * detector contribution；
   * ASFormer adapt。

4. **selector 行为**

   * entropy；
   * normalized entropy；
   * top-10% / top-25% mass；
   * 位置集中度；
   * 相对 exact-uniform 的 mean/p95/max absolute displacement；
   * 相邻采样间隔分布。

5. **训练时钟**

   * attempted batches；
   * successful optimizer updates；
   * AMP skipped attempts；
   * scheduler updates；
   * EMA updates；
   * DUCA schedule updates。

终态必须满足：

[
\text{optimizer}=\text{scheduler}=\text{EMA}=
\text{DUCA schedule}=3000.
]

6. **checkpoint 指标**

   * online；
   * EMA；
   * Avg-mAP；
   * mAP@0.7；
   * 仅作为 learning curve；
   * 不得选择 epoch-14、19、24 等中间 best。

---

# 13. 预注册结果判定与停止规则

定义：

[
R_{\rm Avg}
===========

\frac{M_{\rm Avg}-62.4648}{2.6609},
\qquad
R_{0.7}
=======

\frac{M_{0.7}-39.9434}{3.3703}.
]

这里 (M) 只取 terminal epoch-29 EMA。

## 13.1 三层结果解释

### A. 单 seed 近似无损恢复

同时满足：

[
M_{\rm Avg}\ge64.6257,
\qquad
M_{0.7}\ge42.8137.
]

即两个指标都距离历史 anchor 不超过 `0.50 pp`。

结论只能是：

> seed 3407 下，60-epoch schedule 达到 operational near-recovery。

之后才值得为该压缩配方增加两个预注册 seeds。

### B. 强 schedule recovery，但不是无损

同时满足：

[
R_{\rm Avg}\ge0.75,\qquad R_{0.7}\ge0.75,
]

即：

* Avg-mAP ≥ `64.4605`；
* mAP@0.7 ≥ `42.4711`。

结论：

> 日程修订恢复了大部分损失，但剩余差距与缺失 exposure 一致；不得称为无损压缩。

不继续搜索第三个 scheduler。

### C. schedule-only 失败

若最佳臂在任一关键指标上：

[
R<0.50,
]

即低于：

* Avg-mAP `63.7953`，或
* mAP@0.7 `41.6286`，

则停止所有 60-epoch LR 日程搜索。

同样，若 Avg-mAP 有恢复但 mAP@0.7 不高于压缩结果 `39.9434`，也判定失败：不能用低 IoU 增益掩盖高 IoU 恶化。

## 13.2 两臂之间的机制判断

* B 相对 A 的 Avg-mAP ≥ `+0.50 pp`，且 mAP@0.7 不下降：高 LR exposure 比 floor-tail 更重要；
* A 相对 B 的 Avg-mAP ≥ `+0.50 pp`，且 mAP@0.7 不下降：低 LR full-joint terminal phase 更重要；
* 差异绝对值 `<0.50 pp`：LR 形状没有实质可识别优势，保留 A 作为默认，因为它具有明确 terminal phase 和相对 LR 比例合同。

无论结果如何，**不增加第三个 schedule**。

---

# 14. 失败后的唯一顺序

## 硬 60-epoch 预算不允许放松

两臂均未达到近似无损门时：

> **承认 60-epoch 总预算在当前 seed、固定 base LR 和固定模型合同下不可无损压缩。**

不再：

* 调 floor；
* 改 warmup；
* 改 policy 时间；
* 增加单组 LR；
* 选择中间 best；
* 加 Query/Bridge/dynamic K。

## 允许放松总预算

下一步只能是一个 exposure-dose arm：

* 复用 Stage-1 epoch-29 EMA；
* Stage-2 `45 epoch / 4500 successful updates`；
* 使用本轮两臂中预注册判定胜出的 LR 形式；
* 不重跑 30+60 anchor。

这是对“缺 optimizer exposure”的直接检验。

## 单参数组 LR 微调

它不是本轮失败后的下一步。只有在更长 exposure 下同时看到：

* 某一参数组梯度稳定且长期显著低于其他组；
* 该组对应未加权 loss 持续欠拟合；
* 没有 non-finite、尖峰或 selector collapse；
* 延长 exposure 不能修复；

才允许新的独立裁决考虑一个参数组的小幅 LR 改动。不得从本轮两个单 seed 结果直接调 LR。

---

# 15. 独立 Critic 必须核验

Critic 在任何训练前必须给出 `PASS/BLOCKED`：

1. 两臂的 resolved config 除 LR scheduler 身份、route/work_dir 外完全相同；
2. Stage-1 checkpoint 的 epoch、state key、SHA-256 完全相同；
3. 所有参数组名称、参数集合、base LR、weight decay、冻结状态相同；
4. 新 scheduler 使用相对 multiplier，无共同绝对 floor；
5. AM-RPCH25 的 3000-step LR 轨迹逐点符合公式，累计 normalized exposure 约 `1999.6`；
6. LongCosine terminal fraction 为约 `0.571157`；
7. curriculum 为 `2000`，feedback 为 `1000+1000`，full-joint tail 为 `1000`；
8. scheduler、curriculum 和 EMA 只在 successful optimizer step 后前进；
9. AMP replay 不增加任何正式时钟；
10. Stage-2 resume 不重置 curriculum；
11. inherited `6000`、`60 epoch`、`duca_stage2_half_steps=3000` 没有泄漏到运行长度或课程；
12. 模型、输入、K、loss、NMS、evaluator、seed 均未变化；
13. checkpoint 每 5 epoch 可完整恢复；
14. 无任何 TrueTime/Bridge/dynamic-K 路径进入本实验。

任一项失败都是 PRE_RUN blocker，不得解释模型指标。

---

# 16. Evaluator PRE_RUN 必须冻结

Evaluator 只能在 Critic PASS 后给出 `PRE_RUN_READY`，并冻结：

* clean revision 与 scheduler patch identity；
* 两个完整 resolved config；
* Stage-1 checkpoint hash；
* seed `3407`；
* 每臂恰好 `3000` successful Stage-2 updates；
* terminal epoch-29 final 与 final-EMA；
* intermediate validation 仅作 learning curve；
* 官方 THUMOS14 split/evaluator/NMS identity；
* 只读历史 anchor：

  * `65.1257 / 43.3137`；
  * 压缩负结果 `62.4648 / 39.9434`；
* 本裁决的 recovery ratios 和 stop thresholds；
* 结果在两臂完成、身份和 update receipts 全部通过前不得读取；
* 训练成本分别记录：

  * GPU-hours；
  * wall-clock；
  * AMP replay；
  * validation cost；
  * 共享 Stage-1 成本与新增 Stage-2 成本分开；
* 不得按中间验证结果取消、延长或修改另一臂。

---

# 17. 科学与论文声明边界

单 seed 最多支持：

* 这个固定 seed 下，某种 schedule 是否恢复了压缩损失；
* LR tail 与高后段 LR exposure 的相对机制信号；
* 60-epoch recipe 是否值得增加 seeds。

单 seed不能支持：

* H65 普遍优于其他方法；
* 60-epoch 日程具有跨 seed 稳健性；
* H65 科学机制被重新证明；
* schedule 是论文创新；
* state of the art；
* TrueTime 或 dynamic K 的任何结论。

本次训练日程恢复是**工程与归因工作**，不是新的论文贡献。

`20+40` 负结果同样不能否定 H65：它否定的只是“把多条优化时钟按比例一起压缩即可保留性能”这一具体训练假设。

---

# 18. 最终执行合同

```text
next_owner:
  Builder

next_action:
  实现 relative successful-update scheduler；
  新增 AM-RPCH25 与 LongCosine-H6000 两个 Stage-2 配置；
  生成逐 update LR/curriculum 轨迹与 resume-fidelity 测试；
  不运行 Stage-1，不修改模型。

dependency:
  1. clean HEAD 04c35a3b76897e6c1569eeede41ed3aecaf7f854
  2. 历史 Stage-1 epoch-29 state_dict_ema 的精确路径与 SHA-256
  3. Builder focused tests PASS
  4. 独立 Critic PASS
  5. Evaluator PRE_RUN_READY
  6. 两个新输出根在启动前均不存在
  7. official evaluator 与 final/final-EMA 规则冻结

expected_return_at:
  AFTER_BOTH_REGISTERED_ARMS_REACH_EXACTLY_3000_SUCCESSFUL_STAGE2_UPDATES
  AND_TERMINAL_EPOCH29_FINAL_AND_FINAL_EMA_RECEIPTS_ARE_SEALED
```

**最终路线：复用 30-epoch Stage-1；运行 AM-RPCH25 主臂和 LongCosine-H6000 单一归因臂；两臂之后停止 scheduler 搜索。**
