# H65 30+30 LR-schedule terminal results — evidence packet

- evidence_class: `FULL_TRAINING / OFFICIAL_VALIDATION / SINGLE_SEED / TERMINAL_EMA / SCHEDULE_ATTRIBUTION_ONLY`
- frozen_revision: `ae3642a138c5b2e1ac2daad75a6d43d17cdb6c2f`
- seed: `3407`
- shared_stage1: `30 epochs / 3000 successful updates / epoch-29 state_dict_ema`
- stage2_per_arm: `30 epochs / 3000 successful updates`
- run_root: `/data/run01/sczc063/yuzibo/duca_h65_lr60_formal_ae3642a1_20260824`
- official_validation: canonical THUMOS14 validation, 211 videos, 3325 ground-truth instances
- primary_checkpoint: Stage-2 `epoch_29.pth / state_dict_ema`
- checkpoint_interval: every 5 epochs

## Arm A — AM-RPCH25

- Slurm job: `1252979`
- schedule: 500-update warmup, 1000-update plateau, 1000-update decay, 500-update hold, terminal factor `0.25`
- epoch 19 EMA: Avg-mAP `62.70`, mAP@0.7 `40.34`
- epoch 24 EMA: Avg-mAP `63.35`, mAP@0.7 `40.98`
- epoch 29 terminal EMA: Avg-mAP `63.22`; mAP@0.3/0.4/0.5/0.6/0.7 = `79.29/73.86/66.09/55.59/41.25`
- terminal status: training and official validation completed (`Training Over`)

## Arm B — LongCosine-H6000

- Slurm job: `1252980`
- schedule: historical 6000-update cosine horizon truncated after its first 3000 updates; terminal factor `0.571157`
- epoch 19 EMA: Avg-mAP `62.62`, mAP@0.7 `39.97`
- epoch 24 EMA: Avg-mAP `63.58`, mAP@0.7 `40.95`
- epoch 29 terminal EMA: Avg-mAP `63.56`; mAP@0.3/0.4/0.5/0.6/0.7 = `79.66/74.55/66.58/56.01/41.01`
- terminal status: training and official validation completed (`Training Over`)

## Frozen comparison references

- Historical 30+60 reproduction terminal EMA: Avg-mAP `65.1257`, mAP@0.7 `43.3137`.
- Previous 20+40 compression terminal EMA: Avg-mAP `62.4648`, mAP@0.7 `39.9434`.
- Recovery neighborhood: Avg-mAP `>=64.6257` and mAP@0.7 `>=42.8137`.
- Clear failure: Avg-mAP `<64.1257` or mAP@0.7 `<42.3137`.

Both new 30+30 arms are clear failures. Neither meets the preregistered rising-tail condition because epoch 29 Avg-mAP is below epoch 24 (`63.22<63.35` and `63.56<63.58`), even though mAP@0.7 rises slightly. Under the prior frozen decision tree this selects `STOP_60_EPOCH_COMPRESSION` and does not authorize a third scheduler or a `+1000` continuation.

## Evidence boundary

These results isolate two Stage-2 LR-tail shapes after a mature Stage-1 handoff. They do not isolate curriculum-clock, feedback-clock, full-joint-exposure, or EMA-lag effects. They do not falsify H65 semantic indirect frame selection, establish multi-seed stability, or support a training-efficiency claim.
