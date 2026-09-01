# EVALUATOR_PJST_D1_CYCLE4_C73E8418_PRERUN_AND_LAUNCH-v001

- verdict: `PRE_RUN_READY / FORMAL_EXPERIMENT_RUNNING`
- revision: `c73e8418de31cdcb2a445ff58a1e33ab9ab6a508`
- remote_checkout: `/data/run01/sczc063/yuzibo/projects/duca_pjst_d1_cycle4_c73e8418_20260826` (clean)
- focused_tests: `31 passed, 0 failed, 0 skipped`
- validator/config/shell/compile: PASS
- dataset: canonical THUMOS14 annotation/class map and `raw_data/video` resolved for train/val/test
- Stage-1: epoch 29 EMA, SHA256 `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`
- production_initialization: OFF/ON both strict-loaded 579 states; no missing/unexpected keys; block-0 identity scalar exists and equals exact zero; `single_clock_admission=False` in both.
- formal_jobs: OFF `1256372`, ON `1256373`
- output_roots: `/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/{off,on}`
- initial_runtime_evidence: both jobs RUNNING on N16R4; both reached epoch 0 update 50 with finite loss (`4.2918` OFF, `4.2920` ON), fixed K=384, no traceback/OOM/nonfinite loss.
- evidence_class: execution/PRE_RUN evidence only. No terminal mAP, confidence interval, cost, PASS/KILL, or paper claim.
- next_owner: Evaluator
- next_action: monitor both jobs to terminal 60-epoch final-EMA evaluation, then perform frozen matched identity/statistical analysis.
- dependency: successful completion of both matched arms and official evaluator artifacts.
- expected_return_at: after both formal jobs complete; based on initial throughput, approximately later on 2026-08-26 Beijing time, subject to validation runtime.
- single_recovery: one resume from the latest valid five-epoch checkpoint only if an infrastructure interruption occurs; never select an intermediate checkpoint by validation.
