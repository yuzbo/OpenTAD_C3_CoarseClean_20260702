# Stage-0 Recovery v4 Evaluator Commit Bridge

Status: `user_authorized / implemented / local_tested / remote_pending`

## Failure evidence

Recovery-v3 transaction
`/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_bbf05141_20260728_215335`
was exact-commit clean and passed its code gate. Dense salvage jobs `1200485`
and `1200486` both copied and hash-verified their registered EMA state without
mutating the failed source transaction, then stopped before inference at the
same `tools/test.py` guard:

`formal evaluation checkout differs from DUCA_EXPECTED_COMMIT`

The launcher required and used `DUCA_RIME_EXPECTED_COMMIT`, while the formal
evaluator deliberately reads the cross-launcher canonical variable
`DUCA_EXPECTED_COMMIT`. The salvage launcher did not bridge the two names. Its
checkpoint-only precheck did not enter the evaluator identity guard and
therefore did not detect the omission.

## Chosen repair

1. In `run_duca_rime_dense_salvage.sh`, overwrite-export
   `DUCA_EXPECTED_COMMIT="${DUCA_RIME_EXPECTED_COMMIT}"` after the exact clean
   checkout has been verified and before both precheck and actual evaluation.
2. In the `PRECHECK_ONLY` path, run an identity probe that reads
   `DUCA_EXPECTED_COMMIT` from the environment exactly as `tools/test.py` does,
   compares it with `DUCA_RIME_EXPECTED_COMMIT`, resolves `git rev-parse HEAD`
   in `DUCA_RIME_REPO_ROOT`, and rejects any mismatch.
3. Add a regression test requiring the bridge and identity probe to precede
   both the precheck branch and `torchrun`.
4. Preserve every scientific and provenance boundary: source jobs remain
   failed, source roots remain immutable, dense salvage remains
   engineering-only, Phase 4 remains disabled, and official-final remains
   sealed.

## Rejected alternative

Do not make `tools/test.py` silently fall back from `DUCA_EXPECTED_COMMIT` to
`DUCA_RIME_EXPECTED_COMMIT`. The formal evaluator's canonical environment name
is shared across launchers; a fallback would hide missing launcher bindings and
weaken fail-closed diagnostics.

## Deployment gate

Recovery v4 requires:

1. red-to-green regression plus the complete local focused suite;
2. a fresh exact implementation commit and independent deployment audit;
3. a fresh remote checkout, authoritative Slurm code gate, and both salvage
   prechecks exercising the new identity probe;
4. new commit-bound physical and salvage manifests;
5. a fresh transaction root and new held/released DAG.

No file or artifact from recovery v3 may be reused as a v4 terminal output.
