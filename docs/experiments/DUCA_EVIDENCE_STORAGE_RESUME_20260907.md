# Evidence storage-failure continuation

The six source jobs failed while writing the training audit with `Disk quota
exceeded`. Their checkpoints are read-only inputs, not terminal results:

| Arms | Source commit | Last saved epoch | Successful updates |
|---|---|---:|---:|
| A1, A6 | 1570a72507491899a50767700a35a04eee3f5fe9 | 39 | 4000 |
| F, A3, A4, A5 | 73bdd34ae21c6c675a00d1927b8af7c12f9edc05 | 29 | 3000 |

This repair changes only continuation plumbing. Model/dataset/update source
under `opentad/` and the saved RNG helpers must be unchanged relative to the
explicit source commit. Existing source-file, resolved-config, data, pretrain,
ledger, seed and schedule checks remain strict. The old runtime configuration
digest is reconstructed by changing only `work_dir` back to the checkpoint's
original directory. The old checkpoint and its metadata are never rewritten.

New training audits bind the current clean SHA and retain the source checkpoint,
source SHA, old audit digest, old job and successful-update boundary in
`resume_lineage`. These are resumed runs with explicit two-commit provenance,
not fresh 6000-update runs entirely produced by the new commit. The original
epoch records, optimizer, scheduler, EMA, GradScaler and RNG state are restored.

## Validation and deployment

Run the focused relocation tests and existing Evidence/C3 tests, then deploy a
clean exact-SHA checkout. `run_duca_evidence_storage_resume_precheck_n16r4.sbatch`
runs the CUDA admission and one real resumed epoch for each of the six source
checkpoints. Set `DUCA_REPO_ROOT`, `DUCA_EVIDENCE_EXPECTED_COMMIT` and a fresh
`DUCA_RESUME_PRECHECK_ROOT` before submission. It retains the full 6000-update
recipe and learning-rate trajectory, writes an explicitly nonterminal precheck
checkpoint, and never makes that checkpoint eligible for formal continuation.

After these checks pass, reuse `run_duca_evidence_recovery_train_array_n16r4.sbatch`
with `DUCA_SEEDS=8261`, the arm's array index, a new `DUCA_RUN_ROOT`, the original
`DUCA_RESUME_CHECKPOINT`, its `DUCA_RESUME_SOURCE_COMMIT`, and `PRECHECK_ONLY=0`.
Each arm must resume its original source checkpoint, not a precheck checkpoint.
Retain the original failed job/log/checkpoint and register new independent
terminal evaluation jobs. Final evaluation still requires epoch 59 EMA, 6000
successful optimizer/scheduler/EMA updates and the official evaluator receipt.

The earlier storage probes used two epochs of 100 batches each (200 batches
total), not 200 batches per epoch. They demonstrated new writes but did not
test continuation from the original formal checkpoint or prove the storage
quota cannot be exhausted again.
