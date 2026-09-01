# EVALUATOR_H65_COMPRESSION_READONLY_POSTMORTEM-v001

- status: `COMPLETED / READ_ONLY`
- completed_at: `2026-08-24`
- evaluator_binding: `E:/DeskTop/TAD/OpenTAD_DUCA_H65_LRSchedule_20260824`
- evaluator_revision: `ae3642a138c5b2e1ac2daad75a6d43d17cdb6c2f`
- binding_status: `clean`
- parent_decision: `PRO_DUCA_H65_60_COMPRESSION_POSTMORTEM-v001 / STOP`
- evidence_class: `FULL_TRAINING / OFFICIAL_VALIDATION / SINGLE_SEED / TERMINAL_EMA`; the 30+30 arms are also `SCHEDULE_ATTRIBUTION_ONLY`
- mutation: `none`
- new_training_or_evaluation: `none`

## Frozen evidence

| Run | Revision | Terminal EMA Avg-mAP | mAP@0.7 |
|---|---|---:|---:|
| H65 30+60 | `04c35a3b76897e6c1569eeede41ed3aecaf7f854` | 65.1257 | 43.3137 |
| H65 20+40 | `87ff0883651a631d48468ab4f9d6392f587c15e4` | 62.4648 | 39.9434 |
| H65 30+30 AM-RPCH25 | `ae3642a138c5b2e1ac2daad75a6d43d17cdb6c2f` | 63.22 | 41.25 |
| H65 30+30 LongCosine-H6000 | `ae3642a138c5b2e1ac2daad75a6d43d17cdb6c2f` | 63.56 | 41.01 |

All four terminal records use seed `3407`, canonical THUMOS14 validation with
211 videos, OpenTAD evaluator source SHA-256
`e855e70d41d087d039a90ecdb8f3cc3efece209130417320edf35062b8503fd4`,
and `state_dict_ema`. Their terminal checkpoints are epoch 59, 39, 29, and 29,
respectively.

## Bounded diagnostic outcome

| Requested diagnostic | Status | Evidence and interpretation |
|---|---|---|
| Historical and compressed trajectories aligned at Stage-2 successful updates 2000/2500/3000 | `NOT_OBSERVED` | Existing checkpoint/evaluation artifacts are epoch-granular and do not provide one frozen EMA record at all three exact update counts for all four runs. The strongest exposure falsifier therefore remains unclosed. |
| Online versus EMA official metrics at matched checkpoints | `NOT_OBSERVED` | Formal result JSON files evaluate `state_dict_ema`; no matched official online-state metrics exist. EMA lag cannot be quantified or promoted to a primary cause. |
| Unweighted boundary/transition loss | `OBSERVED_INCOMPLETE` | Existing `log.json` records raw fields including `transition_distribution_loss`, `transition_boundary_coverage_loss`, `actionness_bce_loss`, detector classification/regression losses and contribution-distillation losses. They are not sealed as a common exact-update table and do not by themselves prove a boundary-clock causal failure. |
| Parameter-group movement and Adam moment state | `NOT_OBSERVED` | No frozen per-group displacement or first/second-moment norm artifacts were found. It is not possible to distinguish insufficient movement from a different optimization path. |
| Selector entropy, physical maximum gap and boundary coverage | `NOT_OBSERVED` | Same-named training losses or weights are not executed selector geometry. No aligned selected-position ledger reports entropy, actual physical gaps, endpoint coverage or short-action coverage. |

## Identity and fairness

The terminal evaluator, seed, split, state key and terminal-checkpoint policy are
consistent enough for the bounded schedule-stop decision. The historical 30+60
and 20+40 checkpoints lack `rng_state` and `data_loader_state`; a complete common
audit of AMP skips, replay behavior, resume continuity and data order is not
available across all four runs. This blocks paper-level replication admission,
but it does not make the observed terminal schedule failures disappear.

`65.696` is not a matched H65 anchor and is excluded from this comparison.

## Verdict

- `STOP_60_EPOCH_COMPRESSION`: `SUPPORTED`.
- `H65_COMPRESSION_STOP_CAUSAL_AUDIT`: `BLOCKED` for a unique causal mechanism.
- The existing evidence supports stopping additional learning-rate decay,
  warmup, hold, terminal-factor and stage-ratio sweeps.
- The missing 3000 Stage-2 updates and shortened semantic/policy,
  detector-feedback and full-joint exposure remain the strongest explanation,
  but are not uniquely or numerically proved.
- EMA lag, boundary-support failure and parameter-state divergence remain
  `NOT_OBSERVED` secondary hypotheses.
- H65 semantic indirect nonuniform frame selection is not falsified.

## Handoff

- next_owner: `DUCA Coordinator`
- next_action: freeze 30+60 terminal EMA as the current H65 training recipe; retain the three compression failures; perform no new schedule training
- dependency: accepted Pro STOP and the immutable terminal artifacts listed above
- expected_return_at: this terminal receipt
- single_recovery: `none`

No update to `PAPER_PROGRESS.md` is required: its current account already states
that the compression search stopped and that joint-exposure, EMA and boundary
mechanisms are not independently isolated.
