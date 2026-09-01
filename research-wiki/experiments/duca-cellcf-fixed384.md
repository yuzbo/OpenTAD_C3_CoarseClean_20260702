---
type: experiment
node_id: exp:duca-cellcf-fixed384
title: "DUCA-CellCF fixed-384 matched suite"
status: tested
empirical_state: cellcf_adaptive_allocation_claim_killed_formal_cost_pair_missing
updated: 2026-07-19
---

# DUCA-CellCF fixed-384 matched suite

## 2026-07-19 result interpretation

All three model arms completed at the immutable `1642f26` terminal EMA:
exact-uniform `63.8594`, transition-beta0 `64.2755`, CellCF `64.0610`
Avg-mAP. Exact source/math review further proves that one-per-cell CellCF
cannot transfer quota between regions and that its actual observations are
presented to the detector on fixed uniform anchor coordinates.

The suite is therefore retained as `tested` diagnostic evidence and kills
CellCF's adaptive-allocation main-method interpretation. It does not robustly
prove transition-beta0 superiority from one seed, does not test direct
detector gradients and does not close C7. Cost schema compatibility passed,
but the repeated formal cost pair and dense full-stack comparison remain
missing. No replacement CellCF training is authorized.

## Frozen arms

1. `uniform`: exact-uniform one frame per cell.
2. `transition_beta0`: learned transition-first local deformation without
   detector-derived utility.
3. `cellcf`: the same policy plus detached distinct-cell hard-flip utility.

All arms use offline THUMOS14, K=384/T=768, the same official-component
AdaTAD/ActionFormer backend, 132 epochs, checkpoint every five epochs, exactly
13,200 successful updates, and terminal `epoch_131.pth/state_dict_ema`.

## Evidence DAG

`clean commit -> Linux focused tests -> synthetic gate -> real THUMOS CUDA
gate -> three-arm forced-overflow DDP pilot -> three terminal-EMA full trains
-> independent metric recomputation -> result-to-claim`.

## Current evidence

Local compile and shell syntax pass. CellCF contract tests: 27 passed and three
Windows Torch tests intentionally skipped for Linux. Required C3 regression
tests: 23 passed. The clean remote Linux snapshot passed 62 focused tests.
Independent read-only audit found no P0/P1 model-path defect
after checking cells, short windows, coordinate separation, signed utility,
teacher restoration, loss aggregation, gradient ownership and optimizer
coverage. Its two evidence-chain findings were repaired before deployment.

Implementation commit `475634e1be4a77ad1d9bc6bcf5f4bed04c3d6f31` is pushed.
The CPU synthetic gate passed at
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_475634e_synth_debug_20260716_201424/synthetic_gate.json`,
SHA-256
`ada3a32faaa496924a867ee616309ef06c5c3b653135b828f03107ac9ec7519c`.
Job `1167135` failed before Python from profile/nounset ordering; Job `1167140`
then failed closed because canonical THUMOS environment variables were not
exported. The environment-corrected real-loader gate Job `1167145` passed and
its pure validator revalidated artifact SHA-256
`e0f762fb1387fc823ca1b8ab5b2c291052897b24a75f712d1b6ba9e810b6d7f3`.
It covered real full/mixed/all-short THUMOS batches, positive and negative
detached local-flip utility, one forced AMP skip and same-batch replay, one
successful optimizer/EMA/LR/selector update, and complete trainable-group
gradient coverage. The three-arm forced-overflow DDP pilot Job `1167146`
passed under
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_475634e_ddp_pilot_20260716_202319`.
Each arm had ten successful optimizer/EMA/LR/selector updates, one forced
skip/replay, all full/mixed/all-short K patterns and complete gradient groups;
CellCF had nonzero utility in nine of ten steps. Pilot SHA-256 is
`1c180572683e5dafea00cea7364253b1a5fcc7a24b1916d34642c831de7929c0`.
A post-pilot deployment audit nevertheless found five P1 evidence-handoff
defects covering seed/job hashes, terminal artifact reopening, cost/checkpoint
binding, cost DAG completion and receipt idempotence. No full train was
submitted. The replacement commit must rerun every gate. This node supports no
performance, compute-saving or paper-readiness claim.

## 2026-07-16 evidence-DAG replacement

Commit `3a0f5ae54d1dbd23ff170cda8a4706f5ed0d38d3` on
`codex/duca-cellcf-20260716` closes the five handoff findings. Preparation now
hash-binds the exact commit, seed, manifest, generated sbatch files, job names,
dependency roles and target Slurm cluster. Submission uses intent/receipt
records and reopens live/accounting job identity; only the specific normal
`squeue: Invalid job id specified` transition may fall back to `sacct`, while
all other query failures remain fail-closed. The mandatory DAG is three arms
to aggregate, aggregate to trained-checkpoint cost, and aggregate plus cost to
formal completion. Aggregate status is only `runs_complete_cost_pending`;
`complete` requires reopening all terminal artifacts and cost profiles.

Cost is bound to the CellCF `epoch_131.pth` SHA, `state_dict_ema`, seed,
commit and train/eval config hashes. The comparison remains CellCF frontend
versus a bare exact-uniform frontend lower bound using the same trained
downstream weights; it does not claim dense full-stack savings.

Local checks on the replacement tree: `84 passed, 1 skipped` for all CellCF,
cost and submission contracts; required C3 regressions `23 passed`; Python
compile, shell syntax and `git diff --check` passed. Independent read-only max
review task `019f6af9-7f66-7ea2-9bd8-38cfb75b92c8` first found the completed
receipt fallback P1, then returned `GO_TO_EXACT_COMMIT_GATE` with P0/P1=0 after
the repair and target-cluster reproduction. These are implementation checks,
not mAP or cost results. Gates `1167145/1167146` bind old commit `475634e` and
cannot authorize the replacement. Fresh Linux tests, synthetic gate, real
THUMOS CUDA gate and three-arm DDP pilot are required before any 132-epoch
submission.

## 2026-07-16 replacement gate and pilot

Clean snapshot
`/data/run01/sczc063/yuzibo/projects/opentad_duca_cellcf_3a0f5ae_20260716`
passed 87 CellCF/cost/submission tests and 23 required C3 regressions. The CPU
synthetic gate passed with SHA-256
`1d8234e9c186e1726f08fa59cd892c63f02fd9a47d498d8afe304f05fed9adad`.

Real gate Job `1167220` failed in zero seconds because a blank line preceded
the shebang; Job `1167221` reached a Slurm-visible RTX 4090 but correctly
failed closed before model training because canonical THUMOS variables were
not sourced. These are deployment diagnostics, not model evidence.
Environment-corrected Job `1167222` completed in 86 seconds. Its artifact at
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_3a0f5ae_real_gate_envfix_20260716_2126/real_loader_gate.json`
was independently reopened by the pure validator; SHA-256 is
`b128f58755dcf6bd924fad60d8a84c02a1f149ec0fa70119b836fb61be0c4334`.
It binds commit `3a0f5ae`, real THUMOS annotations/classes/videos, the official
ASFormer source, VideoMAE checkpoint, full/mixed/all-short windows, official
ActionFormer cls+reg, forced AMP replay and one successful update.

The gate-bound three-arm 10-update DDP pilot is Job `1167227`, run root
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_3a0f5ae_ddp_pilot_20260716_2130`.
It completed successfully in 4:46. All arms reached 10/10 successful updates,
replayed one forced overflow, covered full/mixed/all-short batches and had
complete parameter-group gradients. CellCF had nonzero distinct-cell utility
on 9/10 steps. Pilot artifact SHA-256 is
`f199f4dc14aeef8c03ad91838e31281f93bd551ab4a45c01284e48d4aa3d8085`.

## 2026-07-16 invalid formal submission and replacement

The first prepared formal root was
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_3a0f5ae_formal_seed0_20260716_2138`.
Its submitter created arm Jobs `1167234-1167236`, then received no valid Slurm
binding for the aggregate dependency. Because Bash `set -e` was not inherited
through nested command substitutions, the old code continued and wrote
`SUBMITTED` receipts with null job IDs for aggregate/cost/completion. No such
three downstream jobs existed. The arms were cancelled after 95 seconds and
the whole suite is protocol-invalid; it must never be used for mAP or cost.

The working-tree replacement explicitly checks every prepared-binding,
`sbatch`, normalization, receipt write and live-validation substitution. It
rejects empty, malformed, wrong-cluster and job-ID-only responses; only exact
`jobid;cluster` may create a receipt. Dynamic transaction tests confirm invalid
responses leave only `INTENT_RECORDED`, while a valid binding writes
`SUBMITTED`. Local focused checks are 110 passed/5 skipped plus 23 required C3
regressions. This repair changes the evidence code, so the old gate/pilot may
explain the failure but cannot authorize a replacement run. Independent
review, a new exact commit, clean Linux tests, synthetic gate, real-loader CUDA
gate and DDP pilot are required before resubmission.

## 2026-07-16 exact transaction replacement

Commit `b8cd29f621d410b720f12380b3095dd39574e01f` is pushed on
`codex/duca-cellcf-20260716`. It requires canonical positive
`jobid;cluster`, validates scheduler identity and dependency before receipt
creation and after durable receipt reopening, and fsyncs both file and parent
directory. When Slurm removes satisfied `afterok` IDs, every removed
predecessor must be a unique same-cluster `COMPLETED/0:0` allocation; a started
target additionally requires full-second `predecessor End <= target Start`
proof with `SLURM_TIME_FORMAT=standard`.

Local focused result is 127 passed/5 skipped plus 23 C3 regressions. Independent
max review task `019f6b35-56d5-7043-a9d0-3854a4f6d018` returned `GO` with
P0/P1=0 after adversarial dependency and crash-window rechecks. Clean
accelerated snapshot
`/data/run01/sczc063/yuzibo/projects/opentad_duca_cellcf_b8cd29f_20260716`
passed 155 focused tests/3 skipped plus 23 C3 regressions. The exact synthetic
gate passed at
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_b8cd29f_synth_20260716_2250/synthetic_gate.json`;
SHA-256 is
`9606f6325e05767e7b748b85e73352cdc52a439b382541a4dd5ef66ca855a76f`.

The first real-gate wrapper retry, Job `1167338`, failed in one second with
exit `127` because `#!/usr/bin/env bash` did not initialize the cluster
`module` function. Python and the model entrypoint never ran, so this is an
immutable deployment diagnostic rather than model evidence. The environment-
corrected login-shell Job `1167345` completed `0:0` in 62 seconds. Its artifact
is
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_b8cd29f_real_gate_envfix_20260716_2335/real_loader_gate.json`;
independent pure validation passed and SHA-256 is
`c4f6b5ce7d2bb830236ee51cef6d2b5ac5965bd4b84811a12cb2e86eb039b673`.
Gate-bound three-arm DDP pilot Job `1167348` completed `0:0` in 4:38. All
three arms reached 10/10 successful optimizer/EMA/scheduler/selector updates,
replayed one forced AMP overflow, covered full/mixed/all-short batches and had
complete parameter-group gradients. CellCF produced nonzero distinct-cell
utility on 9/10 successful steps. Pilot artifact SHA-256 is
`572e47440c54da558f6320148549de8fd62204d0f524b410f53400fe02249270`.

The prepared formal root
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_b8cd29f_formal_seed0_20260716_2355`
passed manifest validation, but submission fail-closed while reopening the
second arm. On this cluster, pending allocations can have an empty `sacct
Comment` while both live `squeue Comment` and immutable `sacct SubmitLine
--comment=...` retain the exact token. Uniform/transition Jobs
`1167359/1167360` remained pending for zero runtime and were cancelled; no
CellCF, aggregate, cost or completion job was created. This suite is invalid.
The validator is being repaired to permit blank accounting comments only when
one exact SubmitLine comment exists, while rejecting missing, duplicate and
conflicting comments. That evidence-code change requires a new exact commit
and fresh gates before formal resubmission. No full train, terminal mAP or
formal cost evidence exists.

## 2026-07-17 scheduler-policy and live-dependency replacements

Commit `4bf648556383a7a9816320be2245338eff0a3045` strengthened live job
identity, SubmitLine, stored batch-script hashing and crash-safe receipts.
Clean Linux tests, synthetic gate, real-loader Job `1167461` and DDP pilot Job
`1167464` passed. Its formal root created arm Jobs `1167469-1167471`, but the
n16r4 submit policy rejected aggregate because that generated post-run job did
not request a GPU. The arms were cancelled at zero runtime. No aggregate,
cost, completion or result is admissible from that root.

Commit `522925e000aa0a26e18937a019ecc60c1ee1bb3b` added one generic GPU to
aggregate and completion. It passed clean Linux tests (204/3 skipped plus 23
C3), synthetic SHA-256
`930f2bb8664046a050583ce39a0bcd9de07a82447d7e4ba964152ed93b25522e`,
real-loader Job `1167473` with artifact SHA-256
`f3146cab3144b04f449f8b9ad496813e7c1c705762dc70fd552ebb06d0074d7f`,
and DDP pilot Job `1167474` with artifact SHA-256
`95ec215b9053783b8e49befb53ee2db25f44e92694172a658bac437ec724964e`.
Formal Jobs `1167475-1167478` were created, then all cancelled with zero
runtime when live Slurm represented the canonical aggregate dependency as
comma-separated `afterok:id(unfulfilled)` tokens. The old validator rejected
that rendering before receipt creation. This is an immutable deployment
diagnostic, not model evidence.

## 2026-07-17 current exact formal suite

Commit `1642f265e48391418a7c8a4a087e33e2b7bf6899` strictly separates the
canonical submitted dependency from Slurm's annotated live rendering. It
accepts only positive unique `afterok:id(unfulfilled)` tokens and retains exact
SubmitLine, scheduler-script SHA and predecessor completion/time-order proof.
Independent max review returned GO with P0/P1=0. Local broad verification was
196 passed/7 skipped; clean Linux was 212 passed/3 skipped plus 23 C3.

The exact synthetic gate SHA-256 is
`3dd4750cc97d0287b647125264a5495626cb87df6aec6b099b4aed48a523e5cd`.
Real-loader CUDA gate Job `1167479` completed `0:0`; artifact SHA-256 is
`3d630a323e79c694f663c31151c070fd46943296937ceafdd5f9bcacfcbd7cde`.
DDP pilot Job `1167480` completed `0:0`; independently reopened artifact
SHA-256 is
`8e6a59e92f12b15ec1e7c3671104959c0533c9ba9b68dd36550c0294c8b48cd3`.
Each arm had 11 attempted/10 successful updates, complete gradient coverage
and full/mixed/all-short K coverage; CellCF utility was nonzero on 9/10 steps.

Formal root is
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_1642f26_formal_seed0_20260717_0200`.
Jobs: `1167481` uniform, `1167482` transition-beta0, `1167483` CellCF,
`1167484` aggregate after all arms, `1167485` cost after aggregate, and
`1167486` completion after aggregate plus cost. All six prepared, receipt and
scheduler-stored script hashes match. The three arms use equal 2-day limits;
post-run jobs use equal 1-day limits. At the 12:45 CST audit, exact-uniform,
transition-beta0 and CellCF had completed 91/90/83 epochs with exactly
9,100/9,000/8,300 successful optimizer, EMA, scheduler and selector updates.
Checkpoints exist through epochs 89/89/79 at the
unchanged five-epoch interval. AMP replay counts are 5/6/5, all isolated replay
1/8 events with no exhaustion; every logged loss is finite, Slurm stderr is
empty and memory remains about 8.56 GB.
Requested K is 384; logged effective-K means span 190--384 under legal short
windows. CellCF's logged counterfactual distillation loss is nonzero and finite
(latest 0.1664), evidence that the objective executes but not that utility or
mAP improves. Receipt, generated-script and scheduler-stored-script SHA-256
values still agree for all six jobs, and the live dependency DAG remains exact.
Status is only `experiment_running`; no terminal-EMA mAP or formal cost
evidence exists.

## Training-budget interpretation

`epoch_131.pth` is zero-based naming for 132 completed epochs and 13,200
successful optimizer/LR/EMA/selector updates. This schedule was deliberately
chosen to match the historical separated-training exposure of about 13,080
updates and remove the original DUCA undertraining confound (5,940 updates).
It is an admissible fully-trained diagnostic for the three matched arms, but it
is not automatically the final paper training recipe.

The repository AdaTAD base protocol ends at 60 epochs. A defensible efficiency
claim therefore needs a second, same-commit contract in which exact-uniform,
transition-beta0 and CellCF all use the official 60-epoch schedule and equal
successful-update/wall-clock accounting. The current epoch 59, 89 and 131
checkpoints may be evaluated as a predeclared convergence trajectory only.
Epoch 59 was trained under a scheduler whose horizon is 132 epochs, so it is
not equivalent to an independently trained official 60-epoch model and cannot
replace that matched experiment.

The current formal suite remains unchanged and must finish. Final reporting
must include per-model training GPU-hours, peak memory, counterfactual-training
overhead, inference p50/p95 and a break-even calculation between extra
training cost and per-video inference savings. If CellCF improves only at 132
epochs but not at 60 epochs/equal compute, training efficiency is unsupported
and the 132-epoch result belongs only in the convergence or sufficient-exposure
analysis.

## 2026-07-17 evidence tooling and latest progress

Post-run evidence tooling is frozen separately at
`2a0f848f7dbf17b7bcb40aa7a996954e8f87c4de` on
`codex/duca-cellcf-evidence-20260717`. It passed 303 Linux tests with three
skips. It does not alter or retrospectively authorize the model training at
`1642f26`. Its implemented/tested scope is: explicit `exposure132` and
`official60` profiles, profile-bound prepare/submit reopening, raw Slurm and
full-stack cost replay, epoch-59/89/131 convergence inspection, exclusive
artifact publication, and break-even inputs.

At the 16:45 CST audit, Jobs `1167481/1167482/1167483` were still RUNNING and
had started epochs 125/123/114. Their latest five-epoch checkpoints were
124/119/109. Jobs `1167484/1167485/1167486` remained dependency-pending.
There was no Traceback, OOM, ValueError, executed non-finite loss or FAIL.
Losses remained finite, requested K remained 384, and memory was about
8.56 GB. The occasional effective-K value below 384 is legal for short valid
windows and is not a dynamic-budget decision. No terminal `epoch_131` metric,
aggregate, trained-checkpoint cost or completion artifact exists, so C3/C4/C7
remain unproven.

## 2026-07-17 terminal raw results

All three immutable training arms finished `COMPLETED/0:0`. Terminal
`epoch_131.pth/state_dict_ema` metrics are:

| arm | Avg-mAP | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-uniform | 63.8594 | 78.8009 | 73.4968 | 66.5040 | 56.8974 | 43.5978 |
| transition-beta0 | 64.2755 | 78.9614 | 74.4893 | 67.2996 | 57.4936 | 43.1336 |
| CellCF | 64.0610 | 78.8992 | 74.6776 | 66.6185 | 56.2856 | 43.8241 |

Transition-beta0 is `+0.4161` percentage points over exact-uniform. CellCF is
`-0.2145` below transition-beta0 and `+0.2016` over uniform. The one-seed raw
result therefore does not support the claim that current CellCF utility
improves transition-only selection. Aggregate Job `1167484` completed `0:0`;
trained-checkpoint cost Job `1167485` is running and completion Job `1167486`
remains dependency-pending. Status remains `experiment_running`; C3/C4/C7 and
paper readiness cannot advance before the six-job DAG and repaired external
evidence seal complete.

## 2026-07-17 post-run evidence hardening status

Evidence-only commit `787569e` is invalid for deployment because its exact
Linux suite reproduced a transient swap/restore operation that escaped the
terminal stat checks. Replacement commit
`9e96967a158534b014aacde57c1b78bd1591e71a` installs the directory mutation
monitor before each evidence read/hash and retains final hash/identity
revalidation. Independent max review returned GO with no P0/P1.

The exact clean snapshot passed `14` finalizer tests and `253` broad evidence
tests with `--basetemp` on the formal `/data/run01` `fuseblk` mount. The gate
includes an independent-process swap/write/delete/restore attack. This moves
the post-run implementation back to `tested`, not `experiment_running`: no
replacement post-run DAG has yet been submitted. The immutable model results
remain `experiment_running` because cost Job `1167485` and completion Job
`1167486` are still unfinished.

## 2026-07-18 cost-recovery status

The preceding status is historical. Original cost Job `1167485` subsequently
finished `FAILED/1:0` after 2:10:53, and original completion Job `1167486` was
cancelled without runtime. Neither result, the original six-job ledger nor the
completed training arms was rewritten.

Two fail-closed recovery attempts are retained as diagnostics:

- `cost_recovery_5ab3042_v1` found that N16R4 rejects a one-node completion
  job without a GPU request. Diagnostic Job `1170338` was cancelled.
- `cost_recovery_67a8a0a_v1` found that live `JobHeldUser` state can precede
  visibility of the exact accounting `SubmitLine`. Held Job `1170354` was
  cancelled by transaction rollback.

Evidence commit `e153c96bfa0f37b9d4b82046e05b1bbce70dfe50`
requests one logical GPU for both jobs and retries the unchanged strict
accounting check for at most 20 seconds while jobs remain held. It passed 230
exact Linux CellCF tests, compile/Bash/clean-tree checks and an independent
review with P0=0/P1=0.

The current admissible recovery root is
`cost_recovery_e153c96_v1`. Job `1170366` performs the terminal-checkpoint
cost pair; Job `1170367` depends on `afterok:1170366` and validates final-suite
evidence. At submission audit both were `PENDING`, with cost waiting on
priority and completion waiting on the exact dependency. The immutable
recovery manifest SHA-256 is
`e595768d3ddfeccb47d32d5fd0e1a476cbb81b9587ba66165d5e7ef66e8d6c4a`;
the ledger SHA-256 is
`96e8f27ad9e6f47f31f30d340e3cee8a3389a173f675113a52eb34ccef00d2b2`.

Status remains `experiment_running`. This pair measures CellCF versus a bare
uniform frontend; it does not establish dense full-stack savings. C3/C4/C7
and paper readiness remain unproven until this recovery, post-run evidence and
external sealing complete.

### Recovery runtime failure

Job `1170366` later failed `1:0` after 1,357 seconds. The profiler generated
seven component `*_cpu_enqueue_ms` fields, but the strict
`duca_full_stack_cost._derived_sample` schema rejected those producer-owned
fields as unsupported before any cost artifact was published. Dependent Job
`1170367` entered `DependencyNeverSatisfied` and was cancelled with zero
runtime. This is a cost-evidence producer/consumer contract defect, not a model
training result. No replacement was submitted.

### CPU-enqueue data-contract repair and real GPU gate

Evidence commit
`4ce69c852bdbd902046b47bc6019ae11e850dbe4` repairs the producer/consumer
contract without changing the canonical `duca-full-stack-cost-v1` latency
semantics. The producer and summary validator now share one exact seven-key
allowlist:

- `backbone_wrapper_total_cpu_enqueue_ms`
- `coarse_probe_cpu_enqueue_ms`
- `frame_selector_total_cpu_enqueue_ms`
- `head_cpu_enqueue_ms`
- `heavy_backbone_cpu_enqueue_ms`
- `neck_cpu_enqueue_ms`
- `projection_cpu_enqueue_ms`

These fields are finite, nonnegative, raw-only diagnostics. They are retained
in each raw sample but excluded from canonical stage p50/p95 and serial-latency
sums. Unknown `*_cpu_enqueue_ms` fields remain fail-closed. The profiler now
validates each measured sample before appending it, so future producer/schema
drift fails at sample zero rather than after a complete 500-sample run.

Verification:

- local broad CellCF/cost suite: `259 passed, 10 skipped`;
- exact clean Linux snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_cellcf_evidence_4ce69c8_20260718`:
  `279 passed in 113.22s`, plus compile, Bash syntax, exact-HEAD and clean-tree
  checks;
- diagnostic Job `1170932` produced valid two-sample JSONL/JSON/TSV artifacts,
  but the job itself is `FAILED/1:0` because its temporary post-profile heredoc
  lost Python quote characters. It is not a passed gate;
- replacement real-model GPU gate Job `1170940` is `COMPLETED/0:0` in 43
  seconds. It used the formal CellCF config and terminal
  `epoch_131.pth/state_dict_ema`, with one warmup and two measured windows.

The passing gate root is
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_schema_gate_4ce69c8_20260718_110645_+0800`.
Its strict receipt is `schema_gate.json`, SHA-256
`f69bc872993fc778b2ceaf6b1a179721861aa57c176d0e12b5869a3913e14758`.
The measured JSONL and summary JSON SHA-256 values are respectively
`ae3f3ac474f95349c620813618a983ed2937c95627569716521cc033f08fbfb3`
and
`e39fe74155bde9a28854110e270accb6697f813a2f2ec316a80853a53e56617b`.
The receipt verifies exact JSONL/summary raw-sample equality, exact summary
reconstruction, all seven fields, canonical-stage exclusion, the clean exact
commit and all artifact hashes.

The data contract is therefore `tested`, not a paper cost result. No new
500-sample formal cost recovery has been submitted. C7, dense full-stack
savings, break-even and paper readiness remain unproven.
