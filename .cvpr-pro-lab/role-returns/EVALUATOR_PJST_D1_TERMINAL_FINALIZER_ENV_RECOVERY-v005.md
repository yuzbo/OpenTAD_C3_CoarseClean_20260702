# EVALUATOR_PJST_D1_TERMINAL_FINALIZER_ENV_RECOVERY-v005

- verdict: `START / TERMINAL_RESULT_FINALIZATION_RUNNING`
- clean_remote_checkout: `/data/run01/sczc063/yuzibo/projects/duca_pjst_d1_terminal_bundle_4204937_20260827`
- revision: `4204937a933c7a48854b623efefc7fd662e98805`; porcelain empty
- config_interface: exactly `DUCA_STAGE1_CHECKPOINT`, `DUCA_STAGE1_CHECKPOINT_SHA256`, `DUCA_STAGE1_CHECKPOINT_EPOCH`
- frozen_stage1: epoch 29; SHA256 `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3` verified
- checks: focused finalizer tests `4 passed`; launcher shell syntax PASS
- replacement_evaluations: OFF `1257283` -> `off/terminal_eval_v2`; ON `1257284` -> `on/terminal_eval_v2`; both pending at submission
- frozen_inference: epoch-59 `state_dict_ema`, seed/config/checkpoint/data/evaluator bindings unchanged
- bundled_finalizer: job `1257285`; dependency `afterok:1257283:1257284`; pending dependency; `cpus-per-task=4`
- outputs: `bootstrap_shards_v2` and `bootstrap_v2`
- current_scientific_question: can derivative-only PJST-D1 improve H65 first-mixing representation under frozen matched selection?
- next_owner: same Evaluator
- next_action: monitor exact three-job DAG to terminal and return sealed predictions, point estimates, paired CI and frozen gate
- dependency: successful replacement evaluations and bundled 16-shard/10,000-draw finalizer
- expected_return_at: `2026-08-27T18:00:00+08:00`
- single_recovery: consumed; no further equivalent rerun

