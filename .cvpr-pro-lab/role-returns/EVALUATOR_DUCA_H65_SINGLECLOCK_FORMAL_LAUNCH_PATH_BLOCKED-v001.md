# Evaluator formal-launch return — output-root binding blocked

- **Revision:** `b2ccfccab5b4912b59954afcc9b0364955327f7c`
- **PRE_RUN:** Job `1253016`, `COMPLETED 0:0`, `49 passed`
- **Failed formal Job:** `1253017`, `FAILED 1:0` after 10 seconds
- **Evidence class:** pre-execution launcher binding failure; no efficacy evidence.

The submission command placed Slurm stdout/stderr inside the terminal result root
before the launcher started.  The frozen launcher correctly requires that result
root not to exist, so it stopped immediately with
`terminal evaluation output root already exists`.  No tests, checkpoint load,
data iteration, inference, metric, bootstrap or cost action occurred.

PRE_RUN remains valid because neither source, revision, resources, checkpoints nor
protocol changed.  One transport-only resubmission may place Slurm logs in an
independent log directory and give the unchanged launcher a new absent result
root.

- **next_owner:** Evaluator
- **next_action:** submit the identical formal launcher with corrected log/result paths
- **dependency:** new result root must be absent before submission
- **expected_return:** a formal Slurm job that reaches launcher execution
- **single_recovery:** one formal-launch path recovery now authorized
