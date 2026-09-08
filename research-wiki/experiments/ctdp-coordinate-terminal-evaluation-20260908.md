# CT-DP coordinate-corrected terminal evaluation

The independent evaluator is based on training commit
`fe1c53db1b2a7e467a6af2eccfe8b7636f667a81`, preserving its model and
configuration files, including the selected-axis ground-truth and proposal
mapping for G0/G1. It ports the terminal evaluation entry point from
`11ced13ac6b72091d26405c5d6f152f09914c22d` without changing the official
evaluator, dataset, seed, AMP, batch size or checkpoint selection.

The old evaluator remains bound to `78cde1be` for its historical results.
This evaluator rejects those old checkpoints; changing evaluation code cannot
repair training performed with incorrect coordinates. It requires epoch-59
EMA and 6000 actual optimizer/scheduler/EMA updates for terminal receipts.

The one-batch inference PRECHECK consumes the four existing epoch-1 checkpoints
from the already-passed training admission 1278007. It does not retrain models,
repeat that admission, or produce performance metrics. New G0/G1 terminal
evaluation must wait for jobs 1278011/1278012 to finish and pass terminal checks.

Local validation: 45 tests passed, one CUDA witness skipped; Python compilation
passed. The coordinate test could not be collected because the local Windows
Torch c10.dll failed to load. Exact-SHA Linux coordinate tests and Slurm
inference PRECHECK are still required before deployment can be called ready.
Runtime validation and job identities are recorded in the coordination catalog,
not retrospectively assigned to this implementation note.
