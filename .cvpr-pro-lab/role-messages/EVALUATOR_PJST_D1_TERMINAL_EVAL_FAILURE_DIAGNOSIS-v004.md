# Evaluator — read-only diagnosis of failed PJST-D1 terminal inference

Inspect only immutable artifacts for OFF job `1257240` and ON job `1257241`. Use `sacct`, each exact Slurm stdout/stderr, and any output-side failure artifact. Do not modify code/config/data/checkpoints, resubmit either evaluation, submit bootstrap, or infer efficacy.

Return a durable `TERMINAL_DIAGNOSIS` with:

- exact job state, exit code, elapsed time, node and allocation;
- the first causal error and the smallest relevant surrounding stack/log lines for each arm;
- whether both arms share one deterministic transport/launcher defect, an environment/resource failure, a data/checkpoint/evaluator identity failure, or distinct causes;
- whether any prediction/sidecar/output artifact was partially created and whether it is admissible (default inadmissible unless fully sealed);
- the smallest claim-preserving recovery surface, if objectively identifiable, without implementing or submitting it;
- current_scientific_question / next_owner / next_action / dependency / expected_return_at / single_recovery.

If the failure is a deterministic finalizer implementation defect, hand to the existing Builder for one focused correction and then the same Critic. If it is resource/transient infrastructure, hand back to Evaluator for one exact rerun only after the cause is resolved. If evidence is insufficient, return `NEEDS_ATTENTION` rather than guessing.

