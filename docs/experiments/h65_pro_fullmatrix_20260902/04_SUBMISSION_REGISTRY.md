# Submission Registry

Status: PRECHECKED, NOT DEPLOYED.

The frozen 28-row source matrix is `03_EXPERIMENT_MATRIX.csv`. The formal Slurm submitter writes live job ids outside the git checkout at:

`/data/run01/sczc063/yuzibo/h65_pro_fullmatrix_20260902_submission/<commit>/submission_registry.csv`

This keeps the repository clean for `tools/train.py` and `tools/test.py` formal exact-commit checks.

Final implementation commit `bd8623754a4375c39eb5c941893c606cffbcd6de` passed remote `PRECHECK_ONLY=1` for the complete 28-row matrix. Formal deployment is blocked by Slurm account/QOS submit limits (`AssocMaxSubmitJobLimit`); no final-commit registry exists until the scheduler accepts all required train/eval jobs.

One superseded trial submission from an older intermediate commit accepted `REF-D768` train/eval ids `1265842` and `1265843` before the limit fired. Those jobs were canceled and the old registry was marked `CANCELED_SUPERSEDED_COMMIT`. They must not be used for final H65-Pro results.
