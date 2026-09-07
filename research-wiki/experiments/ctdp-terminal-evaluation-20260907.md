# CT-DP independent terminal evaluation

The training identity remains `78cde1be1cb8b3acc7d750afc92ea740f2a03d06`.
This change does not edit the model, dataset, training recipe, NMS or official evaluator.
The independent `tools/test.py` entry reads an explicit epoch-59 EMA checkpoint,
validates its embedded arm/seed/source contract and all actual AdamW steps, runs
fresh complete inference, then seals the official evaluator's output JSON.
Old 5996/5997-update checkpoints remain invalid and are not relabelled.

`scripts/run_ctdp_terminal_eval_n16r4.sbatch` uses `CTDP_EVAL_REPO`,
`CTDP_EVAL_COMMIT`, `CTDP_TRAIN_ROOT`, and a fresh `CTDP_EVAL_ROOT`.
`PRECHECK_ONLY=1` runs focused CUDA tests and one inference batch for each G0-G3
using the already retained two-epoch/six-update precheck checkpoints. It never
opens official metrics or writes a terminal receipt. Formal evaluation uses
array indices 0-3, each dependent on its corresponding completed training job.

Formal receipts are written to `EVAL_ROOT/gN/gpu1_id0/official_eval/metrics.json`.
They identify both training and evaluator commits and bind the terminal state,
actual update counts, resolved evaluation config, predictions and official mAP.
Until GPU precheck and full evaluation finish, this is evaluator implementation,
not new performance evidence. Baselines near 69/65 must not be replaced with
lower modified-recipe reference results when judging improvement.
