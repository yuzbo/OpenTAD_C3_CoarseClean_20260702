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
