# Oracle Browser Transcript

Conversation: https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697-duca/c/6a8c09ca-0844-83ea-9d6e-ad5fe5f73a50

## Prompt

请完整阅读附加任务、冻结配置与终态收据。作为 DUCA 独立 Scientific First-Author Agent 和最严厉训练动力学审稿人，给出唯一中文终稿。Nonce: DUCA-H65-60-COMPRESSION-DIAGNOSIS-v003-20260824

### File: .cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_H65_60_COMPRESSION_DIAGNOSIS-v003.md
Lines: 1-72
```md
 1 | # DUCA H65：90→60 轮压缩性能下降诊断与后续调参裁决
 2 | 
 3 | Nonce: `DUCA-H65-60-COMPRESSION-DIAGNOSIS-v003-20260824`
 4 | 
 5 | 你是 DUCA 项目的独立 Scientific First-Author Agent 与最严厉的训练动力学审稿人。请直接阅读本文件以及附带的冻结配置/收据，给出一份可执行的中文终稿。不要把问题交回给人类，也不要重新设计 DUCA 模型或改变数据、检测器、采样语义和评估协议。
 6 | 
 7 | ## 1. 唯一问题
 8 | 
 9 | 解释为什么历史 H65 从 90 轮（30+60）压缩到 60 轮后出现约 2.66 pp Avg-mAP、3.37 pp mAP@0.7 的下降；区分哪些原因已有证据支持，哪些仍是待检验假设。然后冻结当前两条 30+30 学习率实验结束后的唯一调整决策树。
10 | 
11 | ## 2. 已冻结事实
12 | 
13 | ### 历史 H65 90 轮
14 | 
15 | - Stage-1：30 epochs / 3000 successful optimizer updates。
16 | - Stage-2：60 epochs / 6000 successful optimizer updates。
17 | - seed=3407，K=384，H65 语义间接非均匀逐帧选择。
18 | - epoch-59 EMA 官方验证：Avg-mAP=65.1257；mAP@0.3/0.4/0.5/0.6/0.7 = 80.2808/75.7109/68.5475/57.7757/43.3137。
19 | - 该 checkpoint 缺 RNG/DataLoader 状态，因此不是完整恢复合同，但终态评估本身有效。
20 | 
21 | ### 已失败的 60 轮压缩
22 | 
23 | - Stage-1：20 epochs / 2000 updates。
24 | - Stage-2：40 epochs / 4000 updates。
25 | - 其余数据、seed、K、检测器和 evaluator 保持同一实验族。
26 | - epoch-39 EMA：Avg-mAP=62.4648；mAP@0.3/0.4/0.5/0.6/0.7 = 78.0914/73.4479/65.0772/55.7639/39.9434。
27 | - 相对 H65：Avg-mAP -2.6609 pp；mAP@0.7 -3.3703 pp。
28 | - Stage-1 终点：30-epoch EMA Avg-mAP=59.4231；20-epoch EMA=49.5389，差 -9.8842 pp。同为 epoch-20 时，原日程=50.8707、压缩日程=49.5389，差 -1.3318 pp。
29 | 
30 | 因此旧 20+40 失败不能简单归因于“少 30 轮”或“峰值 LR 太小”。它同时改变了 Stage-1 成熟度、Stage-2 暴露、semantic/policy transition、feedback warmup/decay、full-joint tail、cosine horizon 和 EMA exposure。
31 | 
32 | ### 当前正在运行的 30+30 A/B（尚无终态，不得推断）
33 | 
34 | 两臂复用同一个成熟 Stage-1 epoch-29 EMA；固定模型、K384、数据、seed=3407、损失、检测器、官方 evaluator、参数组与各组基础 LR；Stage-2 均为 30 epochs / 3000 successful updates，每 5 epoch 保存恢复 checkpoint。
35 | 
36 | 1. `AM-RPCH25`：500-update warmup；501–1500 为 1.0×；之后衰减到 0.25× 并保留平坦尾段；累计相对 LR 面积 1999.625。
37 | 2. `LongCosine-H6000`：沿 6000-update 历史 horizon 只执行前 3000 updates；第 3000 update 仍为 0.571157×；累计相对 LR 面积约 2366.228，比 AM 高 18.33%。
38 | 
39 | 两臂共有：2000-step semantic/policy transition、1000-step feedback warmup + 1000-step cosine、约 1000-step full-joint tail。二者仍相对历史 90 轮共同缩短 Stage-2 6000→3000、semantic/policy 3000→2000、feedback 2000→1000、joint tail 3000→1000、EMA exposure 6000→3000。
40 | 
41 | Jobs 1252979/1252980 当前仍 RUNNING；中间 validation 不能用于挑 checkpoint 或宣称终态。当前已经冻结：在它们结束前禁止第三 scheduler、参数组 LR 微调或中途优胜者选择。
42 | 
43 | ### 已冻结的终态门
44 | 
45 | - “恢复到 H65 邻域”：Avg-mAP ≥64.6257 且 mAP@0.7 ≥42.8137。
46 | - “明确失败”：Avg-mAP <64.1257 或 mAP@0.7 <42.3137。
47 | - 介于两者之间为灰区，不得包装成无损压缩。
48 | 
49 | ## 3. 不允许改变的科学身份
50 | 
51 | - H65 模型结构、间接语义选帧、选中 RGB、K=384、VideoMAE-S/Adapter/ActionFormer、loss、NMS、split、官方 evaluator。
52 | - fixed-K 本轮只做训练日程归因；不引入 dynamic K、TrueTime、Query-Bridge、Fovea 或新 selector。
53 | - 不重复 dense/uniform/random 基线，不以中间 checkpoint 选最优。
54 | - 不把单 seed 的调度结果升级为论文级效率或稳定性结论。
55 | 
56 | ## 4. 你必须给出的裁决
57 | 
58 | 请输出唯一 `CONTINUE / REVISE / STOP_60_EPOCH_COMPRESSION`，并逐项回答：
59 | 
60 | 1. 对 20+40 大幅下降给出按证据强弱排序的因果诊断：Stage-1 handoff、Stage-2 总更新数、LR 面积/曲线、课程时钟、反馈时钟、联合训练尾段、EMA 暴露、欠拟合/优化不稳定，各自哪些已被证据支持，哪些只是推测。
61 | 2. 解释为什么“保持基础 LR 不变、保留 30-epoch Stage-1、调整 Stage-2 decay 而非整体抬高峰值 LR、保留非零尾段”是或不是正确策略。
62 | 3. 对当前 A/B 的结果分别冻结动作：
63 |    - 若某臂进入恢复邻域；
64 |    - 若两臂都在灰区；
65 |    - 若两臂都明确失败且终端仍在上升；
66 |    - 若两臂都失败且已平台或下降。
67 | 4. 如确需下一项实验，只能选择一个最便宜、最能区分机制的正式 H65-compatible 训练。给出精确 Stage-1/Stage-2 epochs、成功 update 数、scheduler 公式或关键拐点、semantic/feedback/joint-tail 时钟、EMA、基础 LR 是否变化、唯一停止规则。不得列无界网格。
68 | 5. 明确是否存在“60 epoch 在保持其他结构不变时原则上无法等价 90 epoch”的证据；若没有，请说明目前能支持的最窄结论。
69 | 6. 给出终态数据需要如何分析：训练/验证曲线、梯度与 LR、selector/semantic losses、EMA-vs-final、high-IoU，哪些是诊断，哪些可决定下一步；禁止 post-hoc cherry-pick。
70 | 7. 输出 `next_owner / next_action / dependency / expected_return_at`。
71 | 
72 | 请以严肃但清晰的科研语言书写，首先给一句裁决，再给因果诊断、结果分支和唯一后续动作。不要声称 Jobs 1252979/1252980 已产生终态 mAP。
```

### File: .cvpr-pro-lab/receipts/H65_CURRICULUM_TERMINAL_COMPARISON-v001.md
Lines: 1-38
```md
 1 | # H65 curriculum terminal comparison — MATERIAL RESULT
 2 | 
 3 | - completed_at: `2026-08-24T06:15:13+08:00`
 4 | - evidence_class: `FULL_TRAINING / OFFICIAL_VALIDATION / SINGLE_SEED / EMA_TERMINAL`
 5 | - dataset: canonical THUMOS14 validation, `211` videos
 6 | - evaluator_sha256: `e855e70d41d087d039a90ecdb8f3cc3efece209130417320edf35062b8503fd4`
 7 | - seed: `3407`
 8 | - independent_evaluator: `evaluate_h65_curriculum_terminal`, terminal read-only verification accepted
 9 | 
10 | ## Original 30+60 schedule
11 | 
12 | - job: `1251782`, `COMPLETED 0:0`
13 | - source revision: `04c35a3b76897e6c1569eeede41ed3aecaf7f854`
14 | - run root: `/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823`
15 | - frozen result: `gpu1_id0/intermediate_validation/epoch_060_ema.json`
16 | - checkpoint: `gpu1_id0/checkpoint/epoch_59.pth`, `state_dict_ema`
17 | - Avg-mAP / mAP@0.3/0.4/0.5/0.6/0.7: `65.1257 / 80.2808 / 75.7109 / 68.5475 / 57.7757 / 43.3137`
18 | 
19 | ## Compressed 20+40 schedule
20 | 
21 | - job: `1251622`, `COMPLETED 0:0`
22 | - source revision: `87ff0883651a631d48468ab4f9d6392f587c15e4`
23 | - run root: `/data/run01/sczc063/yuzibo/duca_h65_60_stage2_transition20_joint20_87ff0883_20260823`
24 | - frozen result: `gpu1_id0/intermediate_validation/epoch_040_ema.json`
25 | - checkpoint: `gpu1_id0/checkpoint/epoch_39.pth`, `state_dict_ema`
26 | - Avg-mAP / mAP@0.3/0.4/0.5/0.6/0.7: `62.4648 / 78.0914 / 73.4479 / 65.0772 / 55.7639 / 39.9434`
27 | 
28 | ## Frozen comparison
29 | 
30 | The compressed schedule changes training duration/curriculum but not the H65 selector mechanism. Relative to the original schedule, it changes Avg-mAP / mAP@0.3/0.4/0.5/0.6/0.7 by `-2.6609 / -2.1894 / -2.2630 / -3.4703 / -2.0117 / -3.3703` percentage points. Therefore the 20+40 compression does not preserve the 30+60 H65 endpoint under this seed. This is negative evidence about schedule compression, not a falsification of semantic indirect selection or of the separately trained SingleClock representation gate.
31 | 
32 | The independent Evaluator reproduced the two Slurm terminal states, config seeds, checkpoint epochs/update counts, video counts, evaluator identity, metrics and all reported deltas from the raw artifacts. Both immutable terminal checkpoints lack `rng_state` and `data_loader_state`. Their frozen EMA inference is usable for this diagnostic comparison, but the missing recovery fields prevent paper-level replication admission and must not be fabricated.
33 | 
34 | - next_owner: `DUCA Coordinator`
35 | - next_action: preserve this negative schedule result; continue the already frozen SingleClock/legacy-bootstrap evidence chain without changing the model
36 | - dependency: terminal Jobs `1252482` and `1252515`
37 | - expected_return_at: their formal terminal events
38 | - single_recovery: `none`
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

# 裁决：`CONTINUE`

**Nonce：`DUCA-H65-60-COMPRESSION-DIAGNOSIS-v003-20260824`**

这里的 `CONTINUE` 仅表示：**让已经冻结的 `AM-RPCH25` 与 `LongCosine-H6000` 两个 30+30 作业运行到各自唯一的 terminal EMA，并按预先定义的门自动裁决。**它不表示 60 轮压缩已经成功，也不授权第三个 60 轮 scheduler、参数组 LR 微调、峰值 LR 上调或中间 checkpoint 选择。Jobs `1252979/1252980` 仍无已知终态，本报告不推断其 mAP。

---

## 一、核心科学判断

历史 20+40 的下降不是一个可被“学习率太小”概括的单因素现象，而是一个**耦合压缩失败**：

[
\text{weaker Stage-1 handoff}
\times
\text{fewer Stage-2 updates}
\times
\text{shorter curriculum/feedback clocks}
\times
\text{shorter full-joint tail}
\times
\text{earlier LR decay}
\times
\text{shorter EMA exposure}.
]

现有证据中，**Stage‑1 交接成熟度不足是最强的直接信号**；Stage‑2 更新与 full-joint 暴露不足是下一高优先级解释；LR 曲线/面积是当前 A/B 正在直接检验的因素；课程、反馈和 EMA 目前仍是未隔离假设。

严禁把各项差异机械地加成“各贡献多少 pp”。这些因素在同一训练轨迹中相互作用，现有设计不是因子实验。

---

# 二、20+40 下降的因果诊断排序

## 1. Stage‑1 handoff 不成熟——**最强支持**

历史 30 轮 Stage‑1 terminal EMA 为 `59.4231`，压缩 20 轮仅为 `49.5389`，交接点相差 `−9.8842 pp`。而在同为第 20 轮时，原 30 轮日程为 `50.8707`，压缩日程为 `49.5389`，只差 `−1.3318 pp`。〔任务文件 L21–30〕

这支持两个结论：

1. 压缩日程自身的较短 cosine horizon 已经在相同更新数下造成约 `1.33 pp` 差异；
2. 更大的 `9.88 pp` terminal handoff 差距主要在于压缩模型没有经历后续 10 轮成熟过程。

因此，20+40 的 Stage‑2 是从明显更弱的 detector/coarse-semantic 状态启动的。这会迫使联合阶段同时承担：

* 补 detector 基础能力；
* 学习非均匀选择；
* 开启 detector-to-selector feedback；
* 完成 ASFormer/selector 适配。

但不能据此宣称最终 `−2.6609 pp` 全由 Stage‑1 造成。Stage‑2 明显补回了部分 handoff 差距，故 Stage‑1 是强因果候选而不是完整解释。

## 2. Stage‑2 successful updates 减少——**中强支持，尚未隔离**

历史 Stage‑2 有 `6000` 次成功更新，旧压缩只有 `4000` 次；当前 30+30 A/B 更进一步只有 `3000` 次。〔Stage‑2 历史配置 L17–29；20+40 配置 L73–94；任务文件 L34–39〕

“更新数确实减少”是事实；“这造成了多少性能损失”尚未单独测得。但它是非常有力的解释，因为减少的不只是一般训练时间，而是 learned sampling、detector feedback 和最终 detector-led joint optimization 共同生效后的更新数。

如果当前 A/B 在 terminal 前仍持续上升，则这一解释将明显增强；若已经平台或下降，则单纯增加更新的解释减弱。

## 3. full-joint tail 缩短——**重要且可操作，但未直接证明**

在历史 30+60 中：

* semantic/policy 与 feedback 在约 update 3000 完成；
* 随后还有约 `3000` 次完整联合更新。

旧 20+40 中：

-相关时钟在 update 2000 左右完成；
-剩余约 `2000` 次 full-joint tail。

当前 30+30 A/B 中：

-时钟同样在 update 2000 左右完成；
-只剩约 `1000` 次 full-joint tail。〔任务文件 L36–39〕

因此，当前 A/B 即使修复了 Stage‑1，也仍把完整联合阶段压缩到历史的三分之一。该因素与 Stage‑2 总更新数高度相关，但科学含义更具体：问题可能不是“总共少训练”，而是**selector 已 fully active 后，detector 与采样策略共同沉降的时间不足**。

## 4. LR horizon、面积与尾部非零性——**配置事实明确，性能因果正在由 A/B 检验**

旧 20+40 把 Stage‑2 cosine horizon 从 60 轮缩成 40 轮；Stage‑1 同轮次比较已经表明，较短 horizon 会造成可观差异。因此，“过早进入低 LR 区域”有直接旁证，但还不是 Stage‑2 的独立结果。

当前 A/B 在相同 Stage‑1、相同 3000 updates、相同课程时钟下，只改变 LR 曲线：

* `AM-RPCH25`：累计相对 LR 面积 `1999.625`；
* `LongCosine-H6000`：面积约 `2366.228`，高 `18.33%`，update 3000 仍为 `0.571157×`。〔任务文件 L34–39；AM 配置 L72–84；LongCosine 配置 L11–17〕

所以：

* 若 LongCosine 单独恢复，支持“过早衰减/面积不足”；
* 若 AM 单独恢复，说明较高总面积并非必要，平稳的非零 floor/tail 更重要；
* 若两者均恢复，不能宣称某一 LR 形状是唯一原因；
* 若两者均失败，只能说明**在 3000 Stage‑2 updates 下，调整 LR 曲线不足以恢复 H65**。

累计 LR 面积不是有效参数位移的等价物；后者还依赖梯度尺度、方向和 clipping。

## 5. semantic/policy curriculum clock——**已发生改变，因果未隔离**

历史 transition 为 `3000` steps，旧压缩和当前 A/B 为 `2000` steps。actionness、transition、boundary 权重与 policy alpha 因此更快完成迁移。〔历史 Stage‑2 配置 L58–105；20+40 配置 L17–69；AM 配置 L18–68〕

它可能造成：

* policy 在 detector 尚未适应时过早主导输入；
* coarse supervision 过早减弱；
* detector 所见输入分布变化过快。

但没有 selector loss、梯度或策略熵证据时，这仍是机制假设。当前 A/B 两臂使用相同 2000-step clock，因此不能通过两臂差异隔离该因素。

## 6. feedback clock——**已发生改变，因果未隔离**

历史 feedback 为 `1000 warmup + 2000 transition`；旧 20+40 为 `667+1333`；当前 A/B 为 `1000+1000`。〔历史配置 L88–99；20+40 配置 L49–62；AM 配置 L50–61〕

旧压缩是按总长度近似比例加速；当前 A/B 则保留 1000-step warmup、压缩 transition。两者都改变了 detector gradient 和 detector contribution 进入 selector 的时间结构。

目前不能判断它是“过早反馈”“反馈变化过陡”还是无关因素。必须依赖梯度范数、loss 分量和 selector 行为分析，而不能凭最终 mAP 讲故事。

## 7. EMA exposure/滞后——**弱假设**

历史 Stage‑2 EMA 经历 6000 次联合更新，旧压缩 4000 次，当前 A/B 3000 次。若模型在后期仍快速移动，terminal EMA 可能落后于 online model。

但当前材料没有给出：

* EMA decay 的有效时间常数；
* online-final 与 EMA-final 的差异；
* 后期权重变化率。

因此 EMA 目前只能列为待检验假设。即使 online-final 更好，冻结 primary 仍是 terminal EMA，不能 post-hoc 改用 online checkpoint。

## 8. 欠拟合还是优化不稳定——**当前无法判定；欠拟合先验更强**

目前只有 terminal 指标，没有足够训练曲线、梯度或 entropy 收据。

* 若 loss 持续下降、梯度有限、validation terminal 仍上升、selector 未塌缩，支持欠拟合/暴露不足；
* 若出现梯度尖峰、loss 振荡、selector 熵骤降、online–EMA 强烈分离或高 IoU 反复退化，才支持不稳定。

不能仅凭 20+40 性能低就称其“训练不稳定”。

## 9. 实现差异——**当前证据不支持其为主因**

独立终态收据复现了 seed、checkpoint epoch、更新数、211 个 validation videos、evaluator 身份及所有 delta，并把变化界定为训练时长/课程变化，而非 H65 selector 机制变化。〔终态收据 L28–32〕

两个历史 terminal checkpoint 缺 `rng_state` 与 `data_loader_state`，这阻止完整 resume/论文级复现，但不否定其冻结 terminal EMA 比较。没有证据表明 `−2.6609 pp` 是 evaluator 或 selector 实现漂移造成的。

---

# 三、为什么当前策略是正确的

## 1. 保留 30 轮 Stage‑1：正确且必要

这是对最强 confound 的直接修复。继续从 20 轮 Stage‑1 调 LR，只会把“较弱初始化”和“Stage‑2 scheduler”继续混在一起。

## 2. 保持基础 LR 不变：正确

历史 30+60 已经证明这组基础 LR 至少能够到达 H65 邻域。当前没有任何证据显示峰值 LR 太小。

整体抬高峰值 LR 会同时改变：

* detector/backbone/adapter 的绝对更新尺度；
* coarse trunk/action head/transition scorer 的尺度；
* feedback 开启时的耦合强度；
* AMP、clipping 和 EMA 行为。

这不是对压缩机制的最小修复，而是开启新的优化网格。

## 3. 先改变 decay/horizon，而非峰值：正确

旧压缩同时缩短 horizon 和训练暴露。当前 A/B 保持峰值与参数组比例，直接测试：

* 更大的累计 LR 面积是否必要；
* 非零 tail 是否能防止联合阶段过早冻结。

`RelativeSuccessfulUpdateLR` 对各参数组乘同一个相对因子，能够保持原有组间 LR 比例，优于使用一个绝对 `eta_min` 使不同 base-LR 组发生比例扭曲。

## 4. 保留非零尾段：正确，但不能替代更新数

AM 的 `0.25×` hold 与 LongCosine 在 update 3000 的 `0.571157×` 都避免 terminal 前 LR 近零。它们可以检验“旧压缩是否过早失去可学习性”，但不能凭空替代少掉的 3000 次 historical Stage‑2 updates，也不能恢复被压缩的 full-joint/EMA exposure。

因此当前策略是**正确的第一诊断**，不是预先保证成功的解决方案。

---

# 四、当前 A/B 的冻结终态决策树

令每臂 terminal EMA 的：

[
A_i=\text{Avg-mAP},\qquad H_i=\text{mAP@0.7}.
]

分类严格沿用冻结门：

* **恢复邻域**：(A_i\ge64.6257) 且 (H_i\ge42.8137)；
* **明确失败**：(A_i<64.1257) 或 (H_i<42.3137)；
* **灰区**：其余情况。

## 分支 1：至少一臂进入恢复邻域

**动作：接受一条 30+30 schedule 作为单 seed 的 H65-60 schedule-feasibility 结果；不再运行第三个 scheduler。**

若两臂均通过，按以下预冻结规则选择：

[
S_i=\min(A_i-64.6257,\ H_i-42.8137),
]

选择 (S_i) 较大者，即对两个门具有更大最小安全余量的 arm。若两者 (S_i) 相差不超过 `0.10 pp`，选择 `LongCosine-H6000`，因为它最接近历史 horizon、非历史拐点更少。

允许结论仅为：

> 在 seed 3407、冻结 H65 结构和协议下，该 30+30 schedule 恢复到了预注册 H65 邻域。

不允许升级为：

* 60 轮与 90 轮严格等价；
* 多 seed 稳定；
* 训练效率论文结论；
* selector 或 dynamic-K 论文结论。

## 分支 2：没有 arm 通过，但至少一臂处于灰区

包括“两臂均灰区”以及“一臂灰区、一臂明确失败”。

**动作：拒绝“60 轮无损压缩”表述；不再做第三个 60 轮 scheduler。仅允许下文唯一的 `+1000 full-joint updates` 延长诊断。**

理由是：灰区已经未通过预注册恢复门。再启动一个全新的 30+30 clock/scheduler 组合，会成为 validation-guided 的第三次 60 轮搜索，而且同时修改 semantic、feedback 和 tail，无法干净隔离原因。

## 分支 3：两臂均明确失败，但最佳臂 terminal 仍在上升

“仍在上升”现在冻结为：

* 使用 Stage‑2 epoch `19/24/29` 三个预定 EMA validation 点；
* 对 Avg-mAP 和 mAP@0.7 分别作关于 epoch 的三点最小二乘直线；
* 两条斜率均大于 0；
* epoch 29 在两项指标上均不低于 epoch 24。

**动作：同样只允许唯一的 `+1000 full-joint updates` 延长诊断。**

这时最诚实的假设是 3000 个 Stage‑2 updates 截断了尚未完成的优化，而不是继续发明一个 60 轮 scheduler。

## 分支 4：两臂均明确失败，且最佳臂已经平台或下降

**动作：`STOP_60_EPOCH_COMPRESSION`。**

不再允许：

* 第三个 30+30 scheduler；
* 整体抬高 peak LR；
* 单参数组 LR 微调；
* 改 semantic/feedback clock；
* 用 intermediate best checkpoint 挽救；
* 复跑 20+40。

保留 30+60 为当前 H65 训练参考，把本轮结论记为 schedule-compression negative evidence。

---

# 五、唯一允许的后续训练：`H65-TAIL-EXPOSURE-EXT1000-v001`

这不是第三个 60 轮方案，而是**在 60 轮未通过后，最便宜地判定“是否只是 Stage‑2/full-joint 暴露不足”**。一旦触发该实验，60-epoch no-loss compression 本身已经被停止；该实验最多建立一个 70-epoch 工程折中。

## 1. Parent arm 的确定

在没有通过 arm 时，定义恢复缺口：

[
D_i=\max(64.6257-A_i,\ 42.8137-H_i).
]

* 灰区分支：只在非明确失败 arm 中选 (D_i) 最小者；
* 双失败上升分支：只在满足上述 `RISING` 定义的 arm 中选 (D_i) 最小者；
* 若差值不超过 `0.10 pp`，选择 `LongCosine-H6000`。

这是一项预注册的 parent 选择规则，不允许人工看曲线后改选。

## 2. 精确训练合同

| 项目                             | 冻结值                                                                        |
| ------------------------------ | -------------------------------------------------------------------------- |
| Stage‑1                        | 30 epochs / 3000 successful updates；同一 epoch‑29 EMA handoff                |
| Stage‑2                        | 从选定 arm 的完整 update‑3000 checkpoint 继续到 40 epochs / 4000 successful updates |
| 总训练                            | 30+40，即 70 epochs                                                          |
| seed                           | 3407                                                                       |
| 模型/数据/K/检测器/loss/NMS/evaluator | 全部不变                                                                       |
| semantic/policy/asformer clock | 保持 2000 steps，禁止 reset                                                     |
| feedback clock                 | 1000 warmup + 1000 transition，update 2000 完成，禁止 reset                      |
| full-joint tail                | 从原 1000 扩展为 2000 updates                                                   |
| base LR                        | 不变                                                                         |
| checkpoint                     | 每 5 epochs；primary 为 Stage‑2 epoch 39 terminal EMA                         |
| intermediate validation        | learning-curve only，不选 checkpoint                                          |
| online/EMA                     | terminal EMA 唯一主结果；online 只诊断                                              |

## 3. Scheduler 必须是原 arm 的原生连续延伸

不得重启 scheduler，不得在 update 3001 重新 warmup。

* 若 parent 为 `AM-RPCH25`：前 3000 update 的因子轨迹必须逐点一致；update `2501–4000` 均保持 `0.25×`。
* 若 parent 为 `LongCosine-H6000`：继续同一个 6000-update historical horizon；前 3000 update 逐点一致，update 3000 为 `0.571157×`，之后沿同一 cosine 继续衰减。Builder 必须导出 update `1–4000` 的权威 factor trace；不接受根据 epoch 重新计算的近似曲线。

## 4. Resume fidelity 是硬门

延长训练必须恢复：

* online model；
* EMA shadow；
* optimizer；
* scheduler；
* AMP scaler；
* successful-update counter；
* curriculum schedule step；
* RNG；
* DataLoader/sampler 状态。

不得把 terminal `state_dict_ema` 重新载入模型后重置 optimizer。若完整状态缺失，延长实验 fail-closed，不得改为一次新的 30+40 全量重跑来扩大本轮搜索。

## 5. 唯一停止规则

只读取 update 4000 / Stage‑2 epoch 39 terminal EMA：

* 若同时满足 `Avg-mAP ≥64.6257` 和 `mAP@0.7 ≥42.8137`：保留 30+40 作为单 seed 工程候选，并把“Stage‑2/full-joint 暴露不足”提升为主要解释；
* 否则：停止全部 H65 schedule compression 调参，保留 30+60 参考。

无第四个 scheduler，无中间 checkpoint 挽救，无峰值 LR 调整。

---

# 六、是否已有证据表明“60 轮原则上无法等价 90 轮”

**没有。**

当前材料只能支持最窄的结论：

> 在 seed 3407 和冻结 H65 实验族下，具体的 20+40 压缩日程不能保持 30+60 terminal EMA，下降为 `−2.6609 pp Avg-mAP` 和 `−3.3703 pp mAP@0.7`。〔终态收据 L19–32〕

它不能证明不存在另一条 60 轮优化轨迹。

即使当前两条 30+30 均失败，也只能说明：

> 已测试的 20+40、AM-RPCH25 30+30 和 LongCosine-H6000 30+30 在该单 seed 下不足以恢复 H65 邻域。

届时停止 60 轮压缩，是基于**实验资源纪律与避免无界 scheduler 搜索**，不是数学上的不可能性证明。

---

# 七、终态数据分析与决策权限

## 1. 结果读取前的 admission gate

在读取 mAP 前，独立 Evaluator 必须确认：

* 正确的 Stage‑1 epoch‑29 EMA checkpoint 及 hash；
* seed 3407；
* Stage‑2 正好 3000 successful updates；
* resolved config 与 scheduler mode；
* 参数组 base LR 未变；
* scheduler factor/LR trace 与累计面积；
* terminal epoch‑29 online 与 EMA 均存在；
* evaluator hash、211 videos、split 与 NMS 一致；
* 无 intermediate checkpoint selection；
* checkpoint 具备完整 resume state。

任一身份不一致，结果是 `INVALID`，不是科学成功或失败。

## 2. 能决定分支的量

只有以下信息可以决定下一步：

1. terminal EMA Avg-mAP；
2. terminal EMA mAP@0.7；
3. 若两臂明确失败，预冻结的 epoch 19/24/29 terminal-slope 分类；
4. 运行身份、更新数或 evaluator 失败可直接使结果无效。

## 3. 只用于机制诊断、不能选 checkpoint 的量

### LR 与梯度

逐 successful update 记录：

* 每个参数组实际 LR；
* detector/backbone/adapter/coarse trunk/action head/transition scorer 的梯度 L2 范数；
* 建议同时报告 (L_2/\sqrt{\text{parameter count}})；
* median、p95、max、zero-gradient rate；
* gradient clipping 次数；
* AMP retry/nonfinite 事件。

用途：

* LongCosine 是否真的提供更大有效更新，而不只是名义面积；
* feedback 开启处是否出现梯度冲击；
* terminal 是欠拟合还是不稳定。

### semantic/selector

记录：

* actionness loss；
* transition loss；
* transition-boundary loss；
* detector loss；
* policy alpha、detector gradient、detector contribution 与 ASFormer adapt 的实际 schedule 值；
* actionness/transition score entropy；
* top-mass concentration；
* selected-frame 相对 exact-uniform 的平均、p95、最大位移；
* gap 分布；
* selector 与 uniform 的重合率；
* 是否出现位置塌缩或大幅来回摆动。

除非发生非有限值、K384 身份破坏或实现合同失败，这些量不能改变预注册终态门。

### EMA 与 online

必须同时报告 terminal online 与 terminal EMA 的全部阈值 mAP。

* online 显著优于 EMA：支持 EMA lag；
* online 与 EMA 均低：反对“仅 EMA 滞后”解释；
* EMA 优于 online：支持后期轨迹仍有噪声。

无论何种情况，primary 仍是 terminal EMA，禁止改用 online 挽救。

### high-IoU

mAP@0.7 是恢复门的一部分，不只是附加分析。mAP@0.3–0.6、类别、动作时长和 boundary error 可用于解释，但不能替代 Avg-mAP 与 mAP@0.7 的双门。

历史阈值下降并非严格随 IoU 单调增加——最大损失出现在 0.5 和 0.7 附近——所以不能未经边界误差分析就宣称这是“纯 boundary failure”。

---

# 八、独立审查链

若触发唯一延长实验：

**Builder** 只能增加一个 extension config/resume entry，冻结 update 1–4000 的 LR trace、第一步 continuation 一致性和完整 resume 字段；不得改模型、loss、selector 或基础 LR。

**Independent Critic** 必须验证：

* update 1–3000 与 parent arm 逐点相同；
* optimizer/scheduler/EMA/RNG/DataLoader 未 reset；
* curriculum step 没有重新从 0 开始；
* update 3001 起已处于 full-joint 状态；
* 只增加 1000 successful updates；
* terminal epoch 39 EMA 是唯一 checkpoint；
* 无 intermediate metric-driven 选择。

**Evaluator PRE_RUN** 必须在训练前封存 revision、config、parent checkpoint、resume state、evaluator、输出根、成功更新数和停止规则。Critic 或 PRE_RUN 任一不通过，不执行该延长。

---

# 九、最终反主张

本轮不得声称：

* 20+40 否定了 H65 语义间接选帧；
* 60 轮在原则上不可能恢复；
* LR 是唯一原因；
* mAP@0.7 下降证明了单一边界机制；
* 单 seed scheduler 结果证明训练效率或稳定性；
* intermediate peak 可以替代 terminal EMA；
* 当前 A/B 已有终态结果。

---

```text
next_owner: Independent DUCA Evaluator

next_action:
  等待 Jobs 1252979/1252980 各自形成正式 terminal event；
  先完成只读身份、3000-successful-update、scheduler/LR trace、
  terminal online/EMA、evaluator 与 resumable-state 审计；
  再按本报告的双门、灰区和 terminal-slope 规则自动分类。
  在此之前不得读取中间结果作路线选择，也不得创建第三个 scheduler。

dependency:
  两个 A/B 作业的正式 terminal checkpoints、terminal official-validation JSON、
  successful-update audit、完整 LR/gradient/selector diagnostics，
  以及 full resume-state receipt。

expected_return_at:
  两个 A/B 作业均形成正式 terminal event并完成独立只读封存之时。
```
