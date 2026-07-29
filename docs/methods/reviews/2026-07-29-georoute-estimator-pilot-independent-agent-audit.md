# GeoRoute estimator pilot independent-agent audit

Date: 2026-07-29

Reviewer: fresh default agent `independent_pilot_integrity`, created with
`fork_turns=none`.

Scope: read-only audit of the user-provided CER-TAD Pro review, its project
absorption, the D/K/M preexperiment and six-arm pilot protocols, the estimator
and representation implementation, the production-horizon AMP repair, the
deployment/finalization DAG, and focused tests. The reviewer did not modify
files and did not treat remote artifacts that it had not opened as fresh
runtime evidence.

## Verdict

`DEPLOY_AFTER_OLD_CLOSEOUT_AND_CAPACITY`

No additional Pro discussion or model change is required before the smallest
frozen six-arm exploratory pilot. Deployment remains conditional on:

1. the old `cbe0a082` namespace reaching all-terminal closeout and sealing
   `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`;
2. Slurm admitting the complete 14-job DAG in one submission preflight; and
3. exact clean runtime `30f9ca6fff1572e2eabc6c1b6636c4cc23595a62`
   passing all six expanded P0 leaves in a new namespace.

## Findings

- **A — PASS/WARN, review absorption.** Free v1 is closed, Hybrid and the old
  selector are not promoted, the current method is not called Zoom, and the
  full CER model remains `discussed`. The eleven-arm CER proposal and
  post-result numerical margins were not silently promoted.
- **B — PASS, no leak.** The frozen binding and stage validation prohibit GT,
  teacher, oracle, raw-prediction cache, manual ROI, and official-test inputs
  to the route. Evaluation is development-only.
- **C — PASS, result integrity.** The finalizer requires all six valid stage
  results before emitting any contrast. A failed or missing arm produces an
  empty contrast set and forbids partial-five-arm inference.
- **D — PASS, causal matrix.** The six arms, seed `3407`, `K=64`, 20 epochs,
  representation switches, four contrasts, uniform pooling, and final-only
  checkpoint policy are frozen.
- **E — PASS after repair, numerical integrity.** The production AMP KAT binds
  `T=384`, `N=220`, and `K=64`, uses an FP16 source with FP32 likelihood and
  policy loss, requires an objective beyond FP16 range, and checks finite
  scaled gradients. The repair preserves the original sum-then-batch-mean
  estimator semantics.
- **F — PASS, deployment integrity.** The deployer rejects an existing
  namespace, preflights all 14 jobs before creating it, submits held P0 leaves,
  uses all-terminal `afterany` finalization, and forbids selector reuse,
  P2/P3, official test, and paper claims. Reused names in non-mutating
  `sbatch --test-only` probes are only a warning.
- **G — DEPLOY after gates.** The failed residual-PL cell cannot be resumed and
  the other five old cells cannot be interpreted. The replacement must be a
  complete fresh six-arm run from `30f9ca6f`.

## Evidence boundary

This audit is code and protocol review evidence. It is not a substitute for
the old closeout receipt, the new per-arm P0 suite, six completed stage
results, measured development metrics, multi-seed confirmation, full-stack
cost, Geometry Zoom, official test, or paper evidence.
