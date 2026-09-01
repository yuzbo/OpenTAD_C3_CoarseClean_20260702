# H65 OFF terminal checkpoint audit — INFERENCE_READY / RECOVERY_DEVIATION

- completed_at: `2026-08-24T06:08:01+08:00`
- job: `1251782`, Slurm terminal `COMPLETED 0:0`
- evidence_status: `TRAINING_COMPLETED / INFERENCE_CAPABLE / RESUME_AUDIT_FAILED`
- run_root: `/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823`
- checkpoint: `gpu1_id0/checkpoint/epoch_59.pth`
- checkpoint_sha256: `dafcfbd0b1e0a13c400789e73ee13a20cf69551813ef62fc8185fde609806a1c`
- config_sha256: `73a5b75bce219b3df725a8e6f97a273a7ee6dd1e67c661fd33f261a681563867`
- stage1_checkpoint: `/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth`
- stage1_sha256: `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`

The frozen audit code accepted the exact Stage-1 epoch-29 EMA binding, seed `3407`, 60 epochs, 6000 successful updates, five-epoch checkpoint interval, epoch-59 terminal state, scheduler step 6000, and the presence of final weights, EMA weights, optimizer, scheduler, and AMP scaler. The run-copied child config and the clean evidence checkout config are byte-identical; the latter was used only because the run directory did not copy its relative base config.

The evidence audit was corrected and frozen at Git revision `3089300f`: the legacy H65 OFF checkpoint may omit recovery-only fields for read-only final/EMA inference, while a Clock ON checkpoint still fails closed without them. The H65 OFF checkpoint contains one registered SingleClock scalar in final and EMA, both exactly `0.0`. The resolved detector configuration has `single_clock_admission=False`, so the physical-time residual is not passed into the backbone; the registered-zero scalar is therefore an OFF architecture identity, not evidence of an active SingleClock intervention. N16R4 focused verification passed `20/20` tests.

The immutable checkpoint still has neither `rng_state` nor `data_loader_state`. Its complete top-level key set is `epoch`, `state_dict`, `state_dict_ema`, `optimizer`, `scheduler`, `grad_scaler`, and `successful_optimizer_updates`. Consequently it is eligible for frozen external-reference inference in the ON/gate-zero terminal comparison, but it does not satisfy the preregistered complete-resume contract. If all efficacy gates later pass while this deviation remains, the result is capped at `REVISE_WITHOUT_MORE_TIME_MODULES` and is not paper-claim admissible.

- next_owner: `DUCA Coordinator`
- next_action: wait for SingleClock and legacy-bootstrap terminal evidence; audit those artifacts, then run the already frozen ON/gate-zero/H65-OFF read-only terminal evaluation if the Clock ON audit and old-pair harm gate pass
- dependency: terminal SingleClock checkpoint and legacy RankPack/TrueTime bootstrap; the H65 OFF recovery deviation limits evidence grade but no longer blocks read-only inference
- expected_return_at: after the remaining running jobs reach terminal state
- single_recovery: `none`; the missing fields are absent from the immutable checkpoint and must not be fabricated
