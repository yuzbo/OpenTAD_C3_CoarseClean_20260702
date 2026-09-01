# EVALUATOR_PJST_D1_TERMINAL_BOOTSTRAP_BUNDLE_SUBMIT-v003

- verdict: `NEEDS_ATTENTION / UPSTREAM_TERMINAL_EVALUATION_FAILED`
- evaluation_jobs: OFF `1257240 FAILED 1:0`; ON `1257241 FAILED 1:0`
- duplicate_submission: `NONE`; Evaluator did not resubmit either evaluation
- clean_remote_checkout: `/data/run01/sczc063/yuzibo/projects/duca_pjst_d1_terminal_bundle_4204937_20260827`
- revision: `4204937a933c7a48854b623efefc7fd662e98805`; porcelain empty
- checks: focused finalizer tests `4 passed`; three launcher syntax checks PASS
- bundle_submission: Slurm rejected `afterok:1257240:1257241` with `Job dependency problem`; no bundle job ID exists
- evidence_boundary: no terminal predictions, bootstrap, confidence interval, gate, or new efficacy conclusion
- current_scientific_question: can derivative-only PJST-D1 improve H65 first-mixing representation under the frozen matched selector and official evaluation contract?
- next_owner: same Evaluator for read-only failure diagnosis
- next_action: inspect exact scheduler/stdout/stderr/runtime artifacts for jobs 1257240/1257241 and return one common or arm-specific deterministic failure cause
- dependency: immutable failed job artifacts
- expected_return_at: `2026-08-27T10:20:00+08:00`
- single_recovery: one bounded same-task read-only failure diagnosis; no resubmission

