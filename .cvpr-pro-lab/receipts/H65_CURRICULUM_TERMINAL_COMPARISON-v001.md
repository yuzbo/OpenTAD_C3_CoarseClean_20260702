# H65 curriculum terminal comparison — MATERIAL RESULT

- completed_at: `2026-08-24T06:15:13+08:00`
- evidence_class: `FULL_TRAINING / OFFICIAL_VALIDATION / SINGLE_SEED / EMA_TERMINAL`
- dataset: canonical THUMOS14 validation, `211` videos
- evaluator_sha256: `e855e70d41d087d039a90ecdb8f3cc3efece209130417320edf35062b8503fd4`
- seed: `3407`
- independent_evaluator: `evaluate_h65_curriculum_terminal`, terminal read-only verification accepted

## Original 30+60 schedule

- job: `1251782`, `COMPLETED 0:0`
- source revision: `04c35a3b76897e6c1569eeede41ed3aecaf7f854`
- run root: `/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823`
- frozen result: `gpu1_id0/intermediate_validation/epoch_060_ema.json`
- checkpoint: `gpu1_id0/checkpoint/epoch_59.pth`, `state_dict_ema`
- Avg-mAP / mAP@0.3/0.4/0.5/0.6/0.7: `65.1257 / 80.2808 / 75.7109 / 68.5475 / 57.7757 / 43.3137`

## Compressed 20+40 schedule

- job: `1251622`, `COMPLETED 0:0`
- source revision: `87ff0883651a631d48468ab4f9d6392f587c15e4`
- run root: `/data/run01/sczc063/yuzibo/duca_h65_60_stage2_transition20_joint20_87ff0883_20260823`
- frozen result: `gpu1_id0/intermediate_validation/epoch_040_ema.json`
- checkpoint: `gpu1_id0/checkpoint/epoch_39.pth`, `state_dict_ema`
- Avg-mAP / mAP@0.3/0.4/0.5/0.6/0.7: `62.4648 / 78.0914 / 73.4479 / 65.0772 / 55.7639 / 39.9434`

## Frozen comparison

The compressed schedule changes training duration/curriculum but not the H65 selector mechanism. Relative to the original schedule, it changes Avg-mAP / mAP@0.3/0.4/0.5/0.6/0.7 by `-2.6609 / -2.1894 / -2.2630 / -3.4703 / -2.0117 / -3.3703` percentage points. Therefore the 20+40 compression does not preserve the 30+60 H65 endpoint under this seed. This is negative evidence about schedule compression, not a falsification of semantic indirect selection or of the separately trained SingleClock representation gate.

The independent Evaluator reproduced the two Slurm terminal states, config seeds, checkpoint epochs/update counts, video counts, evaluator identity, metrics and all reported deltas from the raw artifacts. Both immutable terminal checkpoints lack `rng_state` and `data_loader_state`. Their frozen EMA inference is usable for this diagnostic comparison, but the missing recovery fields prevent paper-level replication admission and must not be fabricated.

- next_owner: `DUCA Coordinator`
- next_action: preserve this negative schedule result; continue the already frozen SingleClock/legacy-bootstrap evidence chain without changing the model
- dependency: terminal Jobs `1252482` and `1252515`
- expected_return_at: their formal terminal events
- single_recovery: `none`
