# BA-FDR Successful-Update Repair

The user-authorized six-route audit found a training-contract defect on
2026-09-07. Original training SHA: `539287fa8a035765afd7e79863ce77278bef83f2`.
The six original stdout/stderr pairs were read; they report completion and no
overflow because the driver did not log GradScaler skips.

Direct CPU reads of terminal optimizer state give:

| Arm | Job | Actual AdamW steps | Declared updates / scheduler |
|---|---|---:|---:|
| D160 teacher | 1267884 | 5996 | 6000 |
| G96 | 1268698 | 5997 | 6000 |
| U16 uniform | 1269124 | 5996 | 6000 |
| Late fusion | 1269129 | 5996 | 6000 |
| No distillation | 1269137 | 5994 | 6000 |
| Full distillation | 1269297 | 5995 | 6000 |

Each checkpoint's initialized optimizer states agree on the reported actual
count. `train_epoch` advanced scheduler, EMA and the named successful counter
unconditionally after `GradScaler.step`. A self-hashed receipt did not catch
this semantic error. Existing mAP values and receipts are preserved as
protocol-invalid diagnostic results, not strict-6000 final performance.
This defect alone does not explain the large accuracy gap.

The minimal repair replays the same batch after AMP overflow, restores both
student and teacher forward buffers and RNG, bounds retries at eight, fails
on non-finite forward loss, and advances scheduler/EMA/counters only after an
actual successful update. Epoch checks compare counters to AdamW state and
scheduler state. Teacher and evaluator consumers reject deficient terminal
checkpoints. PRECHECK now runs two real three-batch epochs in its own directory.
Models, router/KD losses, data, seeds, batch size and learning-rate settings
are unchanged. No old checkpoint is resumed or given synthetic missing steps.

Next: focused CPU/CUDA tests, exact clean source, two-GPU real PRECHECK for
D160 and every student arm, then new teacher/student training namespaces as
account capacity permits. FULL must wait for a valid new terminal teacher.
No replacement training or CUDA pass is claimed by this implementation note.
Historical BAFDR jobs 1267920/1267921 remain strictly read-only.

Local verification: 41 focused/static/C3 tests passed; one runtime module was
skipped because this Windows Torch installation cannot load its DLL. Python
compilation, both shell syntax checks and whitespace checks passed. A fresh
bounded code review found no model/loss change and raised checkpoint-use
ambiguity; the CLI now explicitly accepts `--checkpoint` for evaluation only.
Real Linux execution and CUDA/two-GPU data witnesses remain pending.
