# CT-DP successful-update repair

The four c0fae67a runs (1267229-1267232) reached epoch 59, but their AdamW
state steps were 5997/5996/5996/5997 while the scheduler reached 6000.
The retained log.json files ended with finite losses. The completed jobs'
stdout/stderr could not be located in the known source/run directories; this
does not invalidate the checkpoint evidence. Old artifacts are preserved.

The cause is in tools/train.py: only the unrelated S1 route enabled the
existing same-batch AMP replay and exhaustion failure. CT-DP used the legacy
skip-and-advance behavior. This repair explicitly connects CT-DP to that
existing replay implementation without claiming S1 model identity. The
CT-Tubelet, B-AMoD, sampler, physical head and original LR recipe are unchanged.

Formal CT-DP checks 60 complete epochs of 100 batches and equal actual
optimizer, scheduler and EMA counts after every epoch. Checkpoints include
these counters, source identity, GradScaler and RNG states. Old deficient
checkpoints cannot be resumed into or relabeled as a corrected result.
The real-video PRECHECK is separately labeled and runs two epochs of three
batches; it cannot produce a formal 6000-update result.

Branch: codex/duca-ctdp-successful-updates-20260907, based on c0fae67a.
The final local focused suite passed 40 tests with one CUDA-only test skipped
on Windows; py_compile and launcher bash syntax passed. A fresh read-only
review found no blocking defect in this patch. Exact-SHA N16R4 CUDA
witness/PRECHECK must pass before formal replacement jobs. No new performance is
available, and a budget repair alone does not establish mechanism efficacy.

The initial exact-SHA remote CPU run at 3a9f3dfe also passed 40 tests with
the CUDA witness skipped. The GPU admission submission was rejected with
AssocMaxSubmitJobLimit and did not receive a job ID. Its stdout/stderr were
read; this is a scheduler limit, not a model failure. The existing
run_duca_ctdp_cuda_gate_n16r4.sbatch now contains the exact-SHA tests and
four real-video prechecks directly, so the continuation need not rebuild a
quoted shell command. The successor SHA must run this admission before
formal G0-G3 are resubmitted.

GPU admission 1276617 at 0aa72a60 failed in the CUDA witness after 48
other tests passed. Its stdout/stderr identify an uninitialized distributed
process group in the logging-only reduce_loss call: the witness deliberately
uses one unwrapped process. The test now replaces that logging reduction with
the single-process identity. Real CUDA GradScaler overflow, optimizer, scheduler,
EMA, RNG/buffer replay and checkpoint assertions remain active; production model
and training code are unchanged. The successor must rerun the full admission,
including all four real-video prechecks, before replacement formal training.
The revised local non-Torch suite passes 40 tests with one CUDA witness skipped;
py_compile and diff checks pass. The expanded Windows geometry collection still
fails loading c10.dll, so its execution belongs to the exact-SHA Linux admission.
