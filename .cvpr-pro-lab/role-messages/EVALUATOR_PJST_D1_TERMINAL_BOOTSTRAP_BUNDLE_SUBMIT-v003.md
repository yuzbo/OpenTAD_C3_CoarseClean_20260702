# Evaluator — submit and monitor one-allocation PJST-D1 terminal bootstrap

Consume `PJST_D1_TERMINAL_BOOTSTRAP_BUNDLE_STATIC_PASS` on exact clean commit `4204937a933c7a48854b623efefc7fd662e98805`. Reuse the registered evaluation-only identity and workspace; do not register or wake another Evaluator.

First read-only reconcile only the already accepted terminal-evaluation jobs:

- OFF job `1257240`, target `/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/off/terminal_eval`;
- ON job `1257241`, target `/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/on/terminal_eval`.

Do not resubmit either evaluation. Deploy a fresh clean remote checkout of exact `4204937a` under `/data/run01/sczc063/yuzibo/projects/`, prove HEAD and empty porcelain, and rerun only the focused finalizer test plus shell syntax check. Then submit exactly one N16R4 Slurm job using `run_duca_pjst_d1_terminal_bootstrap_bundle_n16r4.sbatch`, with `afterok:1257240:1257241` dependency, OFF/ON prediction paths from those outputs, frozen shard/output roots, and bounded CPU parallelism. Never submit an array or separate merge job.

If scheduler capacity prevents this single dependency job, monitor the two existing evaluation jobs and retry the same unaccepted bundle submission once after a slot is freed; this is transport recovery, not a scientific retry. Unknown submission state allows one read-only scheduler reconciliation and forbids retransmission until resolved.

After acceptance, return immediate `START` with clean checkout, checks, exact evaluation states, bundle job ID/dependency/paths/CPU budget, then monitor the exact three-job DAG to terminal. Validate:

- OFF and ON sidecars both bind epoch-59 `state_dict_ema`, canonical 211-video population, exact prediction files and matching evaluator/config identity;
- all 16 shards cover `[0,10000)` once and merge produces the exact 10,000-draw paired result;
- point estimates, CI bounds and frozen PASS/KILL/INCONCLUSIVE rules are reported without post-hoc changes;
- no training, update, alternate checkpoint, new seed, new arm, data/split/evaluator/NMS change, or efficacy overclaim.

Write the durable terminal receipt required by `EVALUATOR_PJST_D1_CYCLE4_TERMINAL_FINALIZER-v001.md` and include current_scientific_question / next_owner / next_action / dependency / expected_return_at / single_recovery.

