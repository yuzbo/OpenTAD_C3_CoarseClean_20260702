# ZoomToken CPTC K100-TAR50 interaction falsifier terminal receipt

## Frozen authority

- Task: `ZT-CPTC-K100-TAR50-INTERACTION-FALSIFIER-001`
- Pro authority conversation: `6a930db4-fb90-83ea-ae8b-16e5028b6a45`
- Base: `2d945e64bdccd09ae2e2916524562e3f388c5a2a`
- Clean/pushed candidate: `fac88624723aed08175a947025a7f1d8a2af3171`
- Branch: `codex/zoomtoken-k100-tar50-interaction-v001`
- Remote source: `/data/run01/sczc063/yuzibo/projects/zoomtoken_k100_tar50_src_fac88624`
- Result root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_k100_tar50_interaction_fac88624_seed42_20260830`
- Log root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_k100_tar50_logs_fac88624_20260830`
- Formal submission count: `1 / 1`

## Scheduler terminal state

- Formal Job ID / name: `1261680` / `zt-k100-tar50-s42`
- Slurm state / exit: `FAILED` / `1:0`
- Submitted: `2026-08-30 13:00:57+08:00`
- Started: `2026-08-30 13:06:39+08:00`
- Ended: `2026-08-30 13:07:17+08:00`
- Elapsed / node: `00:00:38` / `g0087`
- Allocated resources: 2 GPUs, 8 CPUs, 16-hour requested walltime
- Scheduler reason field at terminal query: `AssocGrpGRES`

The terminal-only waiter `j-cz81o5` emitted this terminal state and then exited. No retry, resume, replacement, second seed or cost job was created.

## Execution failure

The launcher reached epoch 0 but failed before the first successful optimizer update. The terminal training log records:

```text
ValueError: successful update indexing requires a GeoRoute backbone
```

The exception is raised from `opentad/cores/train_engine.py` when the inherited configuration enables successful-update indexing but the strict A-MoD backbone does not expose the GeoRoute callback required by that generic training hook. This is a deterministic launcher/configuration compatibility failure, not an observed model-quality failure.

The terminal receipt reports:

- `train_exit_code=1`
- `final_ema_eval_exit_code=125`
- `primary_checkpoint_present=false`
- `official_result_present=false`
- `retry_resume_replacement=false`
- `cost_measurement=false`

The result root contains only `launch_receipt.tsv`, `terminal_receipt.tsv` and `training.log`. There is no checkpoint, official validation prediction/vector, short-action diagnostic, boundary diagnostic or cost measurement.

## Evidence classification

Classification: `ENGINEERING_OR_PROTOCOL_BLOCKER`.

This run supports only the claim that the frozen launcher/configuration did not produce a first successful update. It does not support or refute the accuracy of native K100 plus fixed-half transformation, does not trigger the six scientific gates, and does not support any latency, memory, energy, boundary or generalization claim. In particular, it does not authorize `STOP_FIXED_HALF_UPDATE_ATTENTION_COLUMN_IDENTITY_BYPASS_FAMILY`, which requires an identity-valid, protocol-complete official validation result.

## Specification–implementation conflict requiring fresh Pro adjudication

The frozen prose described the candidate as native K100 with `[K100,K50]x6`, full K/V over all 800 tokens and exact identity bypass. Candidate `fac88624...`, however, makes no model-code change relative to `2d945e64...`; it reuses strict A-MoD capacity 0.5. In that inherited implementation, odd blocks rank the flattened 800-token sequence globally, gather the selected top 400 tokens, and run attention on the selected tensor. Consequently, odd-block K/V are selected-400 rather than full-800, and the selection is a global flattened top-400 rather than an explicit per-tubelet K50 contract.

The terminal failure and this specification–implementation mismatch must be presented separately to a fresh Project Pro. No local interpretation may convert either issue into a scientific route decision, and the frozen one-submission limit prohibits an automatic corrected rerun.

## Next mandatory action

Submit this complete terminal evidence, the actual inherited model semantics, all known uncertainty and the no-retry constraint to exactly one fresh conversation in the exact ZoomToken Project. Pro must independently classify engineering, protocol and scientific evidence, may reject the current framing, and must return exactly one next task with explicit Beijing deadlines. No experiment or cost measurement is authorized before that adjudication.
