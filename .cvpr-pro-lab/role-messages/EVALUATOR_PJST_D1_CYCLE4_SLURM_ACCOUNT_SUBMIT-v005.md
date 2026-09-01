# EVALUATOR_PJST_D1_CYCLE4_SLURM_ACCOUNT_SUBMIT-v005

Submission-only recovery after an authoritative `sbatch` pre-creation rejection. No job was created and the complete exact-commit PRE_RUN already passed (`30 passed`, zero skip/fail; validator PASS; OFF/ON launcher prechecks PASS).

Read-only Slurm association evidence:

```text
cluster=n16r4 account=sczc063 user=sczc063 qos=gpugpu,normal partition=gpu
```

- source checkout: `/data/run01/sczc063/yuzibo/projects/duca_pjst_d1_cycle4_c195b97c_recovery_20260826`
- exact revision: `c195b97c46acae166e0721fcb412b70221ae7d49`
- checkpoint: `/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth`
- checkpoint SHA-256: `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`
- output root: `/data/run01/sczc063/yuzibo/duca_pjst_d1_c195b97c_20260826`

Do not rerun tests and do not modify code. Reverify exact HEAD/clean state, both output subdirectories absent, and no active jobs named `duca-pjst-c4-off` or `duca-pjst-c4-on`. Then submit exactly two jobs from the source checkout using the unchanged launcher, but override stale scheduler directives at the `sbatch` command line:

- `--account=sczc063 --partition=gpu --qos=normal`
- distinct job names `duca-pjst-c4-off` / `duca-pjst-c4-on`
- exact `MODE`, repository root, checkpoint path/SHA/epoch, output root, `PYTHONNOUSERSITE=1`, and distinct valid `MASTER_PORT` values.

Return exact `jobid;cluster` responses and initial `squeue` states. If the first submission succeeds and the second fails, do not resubmit either; report the exact partial state. No efficacy interpretation.

