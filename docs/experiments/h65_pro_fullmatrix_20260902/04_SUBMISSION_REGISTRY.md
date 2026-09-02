# Submission Registry

Status: PENDING remote submission.

The frozen 28-row source matrix is `03_EXPERIMENT_MATRIX.csv`. The formal Slurm submitter writes live job ids outside the git checkout at:

`/data/run01/sczc063/yuzibo/h65_pro_fullmatrix_20260902_submission/<commit>/submission_registry.csv`

This keeps the repository clean for `tools/train.py` and `tools/test.py` formal exact-commit checks.
