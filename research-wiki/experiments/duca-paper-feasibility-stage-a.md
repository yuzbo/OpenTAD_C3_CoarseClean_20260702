# DUCA paper-feasibility Stage A

## Status

- Decision: `user_approved`
- Design: `designed`
- Implementation: `implemented`
- Local focused verification: `11_passed / 1_Linux_loader_test_skipped_on_Windows`
- Authoritative Linux/Slurm verification: `passed / job_1213711 / 37_tests`
- Experiment: `experiment_running / three_seed_jobs_running`
- Empirical support: `not_yet_empirically_supported`
- Paper status: `not_yet_paper_ready`

## Question

At an exact heavy input of K=384, does jointly learned, task-aware temporal
position selection improve official THUMOS14 localization over exact-uniform
selection under the same ActionFormer, initialization, training exposure,
checkpoint and evaluator contract?

## Frozen Stage-A matrix

Four arms use seeds `5801`, `8123`, and `12011`:

1. dense ActionFormer at T=768;
2. exact-uniform fixed K384;
3. stateless mixed-K training with exposure counts `(8,12,16,24)` over
   `(192,256,384,512)`, evaluated at exact-uniform K384;
4. DUCA jointly optimized ASFormer evidence with learned fixed K384 positions.

Every cell trains on all 200 `training` videos with two-process DDP, global
batch two, 60 epochs, 100 successful updates per epoch and terminal epoch-59
EMA. Training has no validation loader. Evaluation uses exactly the complete
211-video OpenTAD `validation` set, standard sliding-window merge, NMS and mAP.

The mixed-K arm is a detector-robustness training control. It is not a dynamic
inference method. The DUCA ASFormer frontend is jointly trained from the 200
training videos and is not a frozen external checkpoint; its scan cost belongs
to full-stack DUCA cost.

## Implemented evidence chain

- source and resolved-config SHA-256 per arm;
- exact Git commit and clean checkout;
- VideoMAE, annotation and class-map SHA-256;
- exact 200/211 annotation identities;
- two-rank full-train exposure replay for all 60 epochs;
- exactly 6000 optimizer/scheduler/EMA updates;
- terminal EMA compaction and training receipt;
- exact 211 prediction keys plus executed merge/NMS/evaluator receipt;
- transactional held submission for 12 cells and one dependent matrix seal.

The primary scheduler representation is three two-GPU seed jobs. Each seed job
executes the same four logical cells sequentially, followed by one dependent
seal job. This four-job grouping exists only to satisfy the account's immutable
`MaxSubmitJobs=16` while other user-owned jobs are active. It does not share
weights, RNG state, work directories or receipts across logical cells and does
not change the frozen 12-cell scientific matrix.

The exact deployed source is commit
`2df0103ec1c26ff7cff7ed15f399e78e640df211`. Authoritative gate job `1213711`
completed with 37 Linux/PyTorch tests. The production transaction root is
`/data/run01/sczc063/yuzibo/rime_runs/duca_paper_stage_a_2df0103e_20260802_120351`.
Seed jobs `1213712`, `1213713`, and `1213714` contain the twelve independent
logical cells; matrix seal job `1213715` depends on all three. At release, the
seed jobs were scheduler-pending on `AssocGrpGRES` and the seal was dependency-
pending. The first monitored state change placed all three seed jobs in
`RUNNING`, while the seal remained dependency-pending. No metric has been
opened.

No single cell, seed, intermediate checkpoint or incomplete matrix may support a
performance statement. Until all twelve terminal receipts pass, the status is
only `ENGINEERING_STATUS`.

## Conditional Stage B

Stage B remains blocked until Stage A is complete and full-200, training-only
out-of-fold per-K utility/risk targets exist. It will add three dynamic
mean-K384 runs and the corresponding evaluation-only exact same-realized-K
uniform replays. H-RIME, TriDet and K192 remain deferred.
