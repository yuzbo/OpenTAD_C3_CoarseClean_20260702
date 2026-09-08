# Held-control query recovery

On 2026-09-08 at approximately 15:50 Beijing time, the A100 supplementary coordinator's recorded PID was absent on its recorded login host. Its final traceback identified `held_controls.release_controls`: `squeue -h --me -o %i|%T` returned a Slurm socket timeout, and the uncaught `CalledProcessError` terminated the coordinator. The earlier dependency-query repair covered a different call site.

The release query now has a 60-second timeout. A failed query leaves the held-job state and submissions unchanged, records `WAITING_SLURM_QUERY` with the actual error, and retries the scheduling pass after 60 seconds. Successful reads retain the existing release conditions and concurrency limits. This change does not retry `sbatch`, alter the frozen model, or change any training configuration.

Validation used `python -m unittest test_held_controls test_slurm_query_recovery -v`: all 13 tests passed locally and in the existing A100 environment. The new checks inject both a Slurm error and a timeout, verify unchanged job state and no release on a failed read, then verify recovery without duplicate submissions. They also check that the heartbeat retains the prior task states and exposes the query failure.

Only the failed A100 supplementary coordinator was restarted, as revision `backfill-1h-held-controls-v3-release-query-retry`. A subsequent live observation confirmed a fresh heartbeat and the same four held A100 control IDs; the N16 held control also retained its ID. Both primary coordinators and all training processes continued unchanged. Production model commit remains `b70ae056c495b43ca3f305fe438926b97b2723b5`.

Operational receipts and logs stay outside this repository under the execution package's `official_adatad_audit/`: `audit_secondary_a100_inspect_20260908_1551.json`, `audit_release_query_tests_20260908.log`, `audit_release_query_remote_tests_20260908.log`, and `audit_release_query_activation_20260908.log`. These establish coordinator recovery, not new model performance.
