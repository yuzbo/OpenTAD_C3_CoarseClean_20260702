# EVALUATOR_PJST_D1_CYCLE4_N16R4_PRERUN_AND_LAUNCH-v002

This is the single non-equivalent transport recovery of v001. The GitHub TLS clone failure was not a code or PRE_RUN failure.

- role: independent Evaluator
- process_id: codex-evaluator-pjst-d1-cycle4-n16r4-v002
- frozen_revision: `c195b97c46acae166e0721fcb412b70221ae7d49`
- remote_checkout: `/data/run01/sczc063/yuzibo/projects/duca_pjst_d1_cycle4_c195b97c_recovery_20260826`
- transport_bundle: `/data/run01/sczc063/yuzibo/tmp/duca_pjst_cycle4_transport_20260826/duca-pjst-cycle4-c195b97c-20260826.bundle`
- transport_bundle_sha256: `f9eb8997c993409a0153bcafdf7f281efa4096d76a31070b9a8e6e05cb6009ce`
- stage1_checkpoint: `/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth`
- stage1_sha256: `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`
- seed: `3407`
- formal_work_root: `/data/run01/sczc063/yuzibo/duca_pjst_d1_c195b97c_20260826`

The Coordinator has already established the new checkout from the verified bundle and observed exact HEAD with empty porcelain. Reverify these bindings read-only, then execute the same objective and stop rules in `EVALUATOR_PJST_D1_CYCLE4_N16R4_PRERUN_AND_LAUNCH-v001.md` from step 2 onward.

There is no further transport retry. If the complete focused suite, config/checkpoint validator, compile/syntax/resource/duplicate-job checks all pass with zero skips, declare `PRE_RUN_READY` and immediately submit exactly two formal jobs: matched `STAGE2_OFF` and `STAGE2_ON`, distinct work roots under the formal root, same revision/checkpoint/seed/config/evaluator. Return the two authoritative `jobid;cluster` receipts and initial scheduler states. Otherwise stop with the exact single blocker and no job submission.

