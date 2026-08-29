# ZoomToken CPTC TAR32 terminal evidence ingestion map

## Scope

This is a result-blind map of the frozen evaluation-only code path for Slurm job
`1261142`. It was prepared without accessing Slurm, the live result root, logs or
partial metrics. It does not change the task, gate or scientific route.

## Frozen artifact contract

The launcher defines:

- result root:
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_tar32_fkv_eval_only_b0a1ca11_seed42_20260829`
- cell root: `<result-root>/gpu2_id0`
- launch receipt: `<result-root>/launch_receipt.tsv`
- terminal receipt: `<result-root>/terminal_receipt.tsv`
- evaluation log: `<result-root>/evaluation.log`
- official prediction: `<result-root>/gpu2_id0/result_detection.json`

On successful `torchrun`, `result_detection.json` must exist and be non-empty.
The terminal receipt is written after either successful or failed `torchrun` and
records the Job ID/name, exit code, frozen checkpoint SHA, official-result path and
SHA when present, and evaluation-log path and SHA.

## What each artifact proves

- `launch_receipt.tsv` proves the execution binding: candidate/config/checkpoint,
  seed 42, rank 2, canonical 211-video / 792-window population, official evaluator
  plus configured Soft-NMS, and the `[64,32]x6` route with all-K64 K/V and Adapter.
- `terminal_receipt.tsv` proves the terminal command exit and binds the prediction
  and log identities. It does not itself prove accuracy.
- `result_detection.json` is the sole guaranteed official prediction artifact.
  The launcher does not enable raw-prediction export, so an `outputs/` directory
  must not be assumed.
- `evaluation.log` contains the official evaluator output. There is no separately
  persisted metrics JSON in this path.

`tools/test.py` loads `checkpoint["state_dict_ema"]` when EMA is enabled and calls
`eval_one_epoch` without a training or update path. Rank 0 writes
`result_detection.json`, constructs the configured evaluator, calls `evaluate()`,
and logs its output.

## Minimal terminal ingestion order

1. Read `terminal_receipt.tsv`; verify Job ID/name, exit code, paths and recorded SHAs.
2. Verify `launch_receipt.tsv` schema, candidate, checkpoint, population and route.
3. Require a non-empty official prediction on success and recompute its SHA against
   the terminal receipt.
4. Read `evaluation.log` only after terminal state and extract the official Avg-mAP,
   mAP@0.6 and mAP@0.7 values.
5. Reconstruct short-action mAP and start/end boundary median error from the frozen
   official prediction plus canonical annotation. Label these values as reconstructed,
   not directly emitted metrics.
6. Apply the frozen gate and classify the complete terminal package. Even a pass ends
   at `ACCURACY_ADMITTED_PENDING_FRESH_PRO`; it does not authorize cost.

## Source anchors

- launcher artifact paths and receipts:
  `scripts/run_zoomtoken_r1_tar32_fkv_eval_only_n16r4.sh:22-35,89-149`
- EMA loading and evaluation call: `tools/test.py:496-523`
- prediction serialization and evaluator call: `opentad/cores/test_engine.py:222-265`
- frozen gate:
  `docs/aris/ZOOMTOKEN_R1_TAR32_FKV_TERMINAL_AUTHORITY_BINDING_RECEIPT-2026-08-29.md:61-75`
