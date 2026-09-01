# EVALUATOR_PJST_D1_TERMINAL_EVAL_FAILURE_DIAGNOSIS-v004

- verdict: `TERMINAL_DIAGNOSIS / COMMON_DETERMINISTIC_ENVIRONMENT_WIRING_DEFECT`
- jobs: OFF `1257240 FAILED 1:0`; ON `1257241 FAILED 1:0`; each elapsed 35 seconds on node `g0048`
- first_causal_error: `Config.fromfile` raised `ValueError: DUCA_STAGE1_CHECKPOINT is required for DUCA rate curriculum stage 2`
- classification: common launcher submission-environment binding defect; not a resource, data, checkpoint-content, evaluator, model-efficacy, or arm-specific failure
- admissible_artifacts: `NONE`; no terminal-evaluation output, prediction, sidecar, metrics, or partial result exists
- smallest_recovery: submit one exact OFF and ON replacement with the frozen Stage-1 checkpoint path, SHA256 and epoch-29 environment variables; no code/model/config/science change
- current_scientific_question: can derivative-only PJST-D1 improve H65 first-mixing representation under frozen matched selection?
- next_owner: same Evaluator
- next_action: one exact failed-job replacement and dependency-bound bundled bootstrap submission
- dependency: corrected immutable environment binding
- expected_return_at: `2026-08-27T18:00:00+08:00`
- single_recovery: one same-task infrastructure recovery; no duplicate accepted job

